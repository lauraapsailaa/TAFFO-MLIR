#include "Taffo/Transforms/LowerToArithPass.h"
#include "Taffo/Dialect/Attributes.h"
#include "Taffo/Dialect/Taffo.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Taffo/Dialect/Ops.h"

namespace mlir::taffo {
#define GEN_PASS_DEF_LOWERTOARITHPASS
#include "Taffo/Transforms/Passes.h.inc"
} // namespace mlir::taffo

using namespace ::mlir::taffo;

namespace mlir {
class LowerToArithPass
    : public mlir::taffo::impl::LowerToArithPassBase<LowerToArithPass> {
public:
  using LowerToArithPassBase::LowerToArithPassBase;

  class TaffoToArithTypeConverter : public mlir::TypeConverter {
  public:
    TaffoToArithTypeConverter(MLIRContext *ctx) {
      addConversion([](Type type) { return type; });
      addConversion([ctx](RealType type) -> Type {
        return IntegerType::get(ctx, type.getBitwidth(),
                                IntegerType::SignednessSemantics::Signless);
      });

      addTargetMaterialization(
          [&](mlir::OpBuilder &builder, mlir::TypeRange resultType,
              mlir::ValueRange inputs, mlir::Location loc,
              mlir::Type) -> llvm::SmallVector<mlir::Value> {
            if (inputs.size() != 1) {
              return {};
            }

            auto CastToRealOp =
                builder.create<mlir::UnrealizedConversionCastOp>(
                    loc, resultType, inputs);

            llvm::SmallVector<mlir::Value> result;
            result.push_back(CastToRealOp.getResult(0));
            return result;
          });

      addSourceMaterialization(
          [&](mlir::OpBuilder &builder, mlir::Type resultType,
              mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
            if (inputs.size() != 1) {
              return {};
            }

            auto CastToRealOp =
                builder.create<mlir::UnrealizedConversionCastOp>(
                    loc, resultType, inputs);

            return CastToRealOp.getResult(0);
          });
    }
  };

  static int getExp(Value v) {
    return ::llvm::dyn_cast<RealType>(v.getType()).getExponent();
  }

  static bool getSignd(Value v) {
    return ::llvm::dyn_cast<RealType>(v.getType()).getSignd();
  }

  static int intWidth(Value v) {
    return cast<IntegerType>(v.getType()).getWidth();
  }

  // Sign/zero-extend up to `width` per the operand's own signedness.
  // Never truncates.
  static Value extendToWidth(ImplicitLocOpBuilder &b, Value v, Value ogV,
                             int width) {
    if (intWidth(v) >= width)
      return v;
    return getSignd(ogV) ? b.create<arith::ExtSIOp>(b.getIntegerType(width), v)
                               .getResult()
                         : b.create<arith::ExtUIOp>(b.getIntegerType(width), v)
                               .getResult();
  }

  // Truncate to `width` if wider; no-op otherwise.
  static Value truncToWidth(ImplicitLocOpBuilder &b, Value v, int width) {
    if (intWidth(v) == width)
      return v;
    return b.create<arith::TruncIOp>(b.getIntegerType(width), v).getResult();
  }

  // Integer constant with the same type as `like`.
  static Value constLike(ImplicitLocOpBuilder &b, Value like, int64_t value) {
    return b.create<arith::ConstantOp>(b.getIntegerAttr(like.getType(), value))
        .getResult();
  }

  template <typename T1, typename T2>
  static LogicalResult ConvertAddCommon(T1 op, T2 adaptor,
                                        ConversionPatternRewriter &rewriter) {

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    RealType resType = ::llvm::dyn_cast<RealType>(op->getResult(0).getType());

    const int targetWidth = resType.getBitwidth();

    Value ogLhs = op.getLhs();
    Value ogRhs = op.getRhs();
    int lhsExp = getExp(ogLhs);
    int rhsExp = getExp(ogRhs);
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    // Compute at the widest width; narrow only the final result.
    const int workWidth = std::max({targetWidth, intWidth(lhs), intWidth(rhs)});
    auto buildIntAttr = [workWidth](Builder b, int64_t value) -> IntegerAttr {
      return b.getIntegerAttr(b.getIntegerType(workWidth), value);
    };
    lhs = extendToWidth(b, lhs, ogLhs, workWidth);
    rhs = extendToWidth(b, rhs, ogRhs, workWidth);
    auto finish = [&](Value res) {
      rewriter.replaceOp(op, truncToWidth(b, res, targetWidth));
    };

    // reconcile arguments of different signedness
    if (resType.getSignd() && (getSignd(ogRhs) != getSignd(ogLhs))) {

      if (!getSignd(ogLhs)) {
        lhsExp += 1;
        lhs = b.create<arith::ShRUIOp>(
                   lhs, b.create<arith::ConstantOp>(buildIntAttr(b, 1)))
                  .getResult();
      }

      if (!getSignd(ogRhs)) {
        rhsExp += 1;
        rhs = b.create<arith::ShRUIOp>(
                   rhs, b.create<arith::ConstantOp>(buildIntAttr(b, 1)))
                  .getResult();
      }
    }

    int expDiff = std::abs(rhsExp - lhsExp);
    // if the difference in exponent is large enough that largest number of
    // the smaller operand cannot be represented by the larger operand,
    // we delete the op
    if (expDiff > workWidth) {
      Value maxExpArg = rhsExp > lhsExp ? rhs : lhs;
      finish(maxExpArg);
      return success();
    }

    Value res;

    if (expDiff != 0) {
      Value to_shift = rhsExp < lhsExp ? rhs : lhs;
      Value no_shift = rhsExp > lhsExp ? rhs : lhs;

      arith::ConstantOp shift_amount =
          b.create<arith::ConstantOp>(buildIntAttr(b, expDiff));
      Value ShOp =
          resType.getSignd()
              ? b.create<arith::ShRSIOp>(to_shift, shift_amount).getResult()
              : b.create<arith::ShRUIOp>(to_shift, shift_amount).getResult();
      res = b.create<arith::AddIOp>(no_shift, ShOp);
    } else {
      res = b.create<arith::AddIOp>(lhs, rhs);
    }

    int resExpDiff = resType.getExponent() - std::max(rhsExp, lhsExp);

    if (resExpDiff == 0) {
      finish(res);
      return success();
    }

    arith::ConstantOp align_res =
        b.create<arith::ConstantOp>(buildIntAttr(b, std::abs(resExpDiff)));
    res = resExpDiff > 0
              ? resType.getSignd()
                    ? b.create<arith::ShRSIOp>(res, align_res).getResult()
                    : b.create<arith::ShRUIOp>(res, align_res).getResult()
              : b.create<arith::ShLIOp>(res, align_res).getResult();
    finish(res);
    return success();
  }

  struct ConvertAdd : public OpConversionPattern<AddOp> {
    ConvertAdd(mlir::MLIRContext *context)
        : OpConversionPattern<AddOp>(context) {}

    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(AddOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
      return ConvertAddCommon<AddOp, OpAdaptor>(op, adaptor, rewriter);
    }
  };

  template <typename T1, typename T2>
  static LogicalResult ConvertSubCommon(T1 op, T2 adaptor,
                                        ConversionPatternRewriter &rewriter) {

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    RealType resType = ::llvm::dyn_cast<RealType>(op->getResult(0).getType());

    const int targetWidth = resType.getBitwidth();

    Value ogLhs = op.getLhs();
    Value ogRhs = op.getRhs();
    int lhsExp = getExp(ogLhs);
    int rhsExp = getExp(ogRhs);
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    // Compute at the widest width; narrow only the final result.
    const int workWidth = std::max({targetWidth, intWidth(lhs), intWidth(rhs)});
    auto buildIntAttr = [workWidth](Builder b, int64_t value) -> IntegerAttr {
      return b.getIntegerAttr(b.getIntegerType(workWidth), value);
    };
    lhs = extendToWidth(b, lhs, ogLhs, workWidth);
    rhs = extendToWidth(b, rhs, ogRhs, workWidth);
    auto finish = [&](Value res) {
      rewriter.replaceOp(op, truncToWidth(b, res, targetWidth));
    };

    // reconcile arguments of different signedness (identical to ConvertAdd)
    if (resType.getSignd() && (getSignd(ogRhs) != getSignd(ogLhs))) {

      if (!getSignd(ogLhs)) {
        lhsExp += 1;
        lhs = b.create<arith::ShRUIOp>(
                   lhs, b.create<arith::ConstantOp>(buildIntAttr(b, 1)))
                  .getResult();
      }

      if (!getSignd(ogRhs)) {
        rhsExp += 1;
        rhs = b.create<arith::ShRUIOp>(
                   rhs, b.create<arith::ConstantOp>(buildIntAttr(b, 1)))
                  .getResult();
      }
    }

    int expDiff = std::abs(rhsExp - lhsExp);

    // If one operand's exponent dwarfs the other's, the smaller-magnitude
    // operand is negligible at this bitwidth. Unlike addition, subtraction
    // is NOT symmetric here: if rhs dominates, the result is approximately
    // -rhs (not +rhs), since lhs contributes ~0.
    if (expDiff > workWidth) {
      if (rhsExp > lhsExp) {
        Value zero = b.create<arith::ConstantOp>(buildIntAttr(b, 0));
        Value negRhs = b.create<arith::SubIOp>(zero, rhs).getResult();
        finish(negRhs);
      } else {
        finish(lhs);
      }
      return success();
    }

    // Align both operands to the same (coarser/larger) exponent by shifting
    // whichever has the SMALLER exponent right by expDiff. alignedLhs /
    // alignedRhs are tracked explicitly so the final SubIOp's operand order
    // always matches the original lhs - rhs.
    Value alignedLhs = lhs;
    Value alignedRhs = rhs;

    if (expDiff != 0) {
      arith::ConstantOp shift_amount =
          b.create<arith::ConstantOp>(buildIntAttr(b, expDiff));
      if (lhsExp < rhsExp) {
        alignedLhs =
            resType.getSignd()
                ? b.create<arith::ShRSIOp>(lhs, shift_amount).getResult()
                : b.create<arith::ShRUIOp>(lhs, shift_amount).getResult();
      } else {
        alignedRhs =
            resType.getSignd()
                ? b.create<arith::ShRSIOp>(rhs, shift_amount).getResult()
                : b.create<arith::ShRUIOp>(rhs, shift_amount).getResult();
      }
    }

    Value res = b.create<arith::SubIOp>(alignedLhs, alignedRhs);

    int resExpDiff = resType.getExponent() - std::max(rhsExp, lhsExp);

    if (resExpDiff == 0) {
      finish(res);
      return success();
    }

    arith::ConstantOp align_res =
        b.create<arith::ConstantOp>(buildIntAttr(b, std::abs(resExpDiff)));
    res = resExpDiff > 0
              ? resType.getSignd()
                    ? b.create<arith::ShRSIOp>(res, align_res).getResult()
                    : b.create<arith::ShRUIOp>(res, align_res).getResult()
              : b.create<arith::ShLIOp>(res, align_res).getResult();
    finish(res);
    return success();
  }

  struct ConvertSub : public OpConversionPattern<SubOp> {
    ConvertSub(mlir::MLIRContext *context)
        : OpConversionPattern<SubOp>(context) {}

    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(SubOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
      return ConvertSubCommon<SubOp, OpAdaptor>(op, adaptor, rewriter);
    }
  };

  template <typename T1, typename T2>
  static LogicalResult ConvertMultCommon(T1 op, T2 adaptor,
                                         ConversionPatternRewriter &rewriter) {

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    RealType resType = ::llvm::dyn_cast<RealType>(op->getResult(0).getType());

    const int resWidth = resType.getBitwidth();
    const int ogWidth = resWidth;

    Value ogLhs = op.getLhs();
    Value ogRhs = op.getRhs();
    int lhsExp = getExp(ogLhs);
    int rhsExp = getExp(ogRhs);
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    // Wide type must hold the full product of the operands as given.
    // Equals 2*resWidth when both operands are resWidth wide (original).
    const int extWidth = std::max(resWidth, intWidth(lhs) + intWidth(rhs));

    auto buildWideAttr = [extWidth](Builder b, int64_t value) -> IntegerAttr {
      return b.getIntegerAttr(b.getIntegerType(extWidth), value);
    };

    // reconcile arguments of different signedness; the shift constant takes
    // each operand's own type, since operands may not be resWidth wide.
    if (resType.getSignd() && (getSignd(ogRhs) != getSignd(ogLhs))) {

      if (!getSignd(ogLhs)) {
        lhsExp += 1;
        lhs = b.create<arith::ShRUIOp>(lhs, constLike(b, lhs, 1)).getResult();
      }

      if (!getSignd(ogRhs)) {
        rhsExp += 1;
        rhs = b.create<arith::ShRUIOp>(rhs, constLike(b, rhs, 1)).getResult();
      }
    }

    int implicitExp = rhsExp + lhsExp + resType.getBitwidth();
    int expDiff = resType.getExponent() - implicitExp;

    // extend operands
    // Skip the extension when an operand is already extWidth wide
    // (extsi/extui require a strictly wider result).
    if (intWidth(lhs) < extWidth)
      lhs = resType.getSignd()
                ? b.create<arith::ExtSIOp>(b.getIntegerType(extWidth), lhs)
                      .getResult()
                : b.create<arith::ExtUIOp>(b.getIntegerType(extWidth), lhs)
                      .getResult();
    if (intWidth(rhs) < extWidth)
      rhs = resType.getSignd()
                ? b.create<arith::ExtSIOp>(b.getIntegerType(extWidth), rhs)
                      .getResult()
                : b.create<arith::ExtUIOp>(b.getIntegerType(extWidth), rhs)
                      .getResult();

    Value res = b.create<arith::MulIOp>(lhs, rhs).getResult();

    // this is probably unnecessary as I expect shifts by 0 get folded away
    // TODO: check if this check is necessary
    if (expDiff == 0) {
      rewriter.replaceOp(op, truncToWidth(b, res, ogWidth));
      return success();
    }

    int correctionFactor = expDiff + ogWidth;
    arith::ConstantOp align_res = b.create<arith::ConstantOp>(
        buildWideAttr(b, std::abs(correctionFactor)));

    // this should never be < 0, but it's best to check (if it is, it means
    // that VRA has been broken)
    res = correctionFactor > 0
              ? resType.getSignd()
                    ? b.create<arith::ShRSIOp>(res, align_res).getResult()
                    : b.create<arith::ShRUIOp>(res, align_res).getResult()
              : b.create<arith::ShLIOp>(res, align_res).getResult();

    res = truncToWidth(b, res, ogWidth);
    rewriter.replaceOp(op, res);
    return success();
  }

  struct ConvertMult : public OpConversionPattern<MultOp> {
    ConvertMult(mlir::MLIRContext *context)
        : OpConversionPattern<MultOp>(context) {}

    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(MultOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
      return ConvertMultCommon<MultOp, OpAdaptor>(op, adaptor, rewriter);
    }
  };

  template <typename T1, typename T2>
  static LogicalResult ConvertDivCommon(T1 op, T2 adaptor,
                                        ConversionPatternRewriter &rewriter) {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    RealType resType = ::llvm::dyn_cast<RealType>(op->getResult(0).getType());
    const int resWidth = resType.getBitwidth();
    const int ogWidth = resWidth;

    Value ogNumerator = op.getLhs();
    Value ogDenom = op.getRhs();
    int numExp = getExp(ogNumerator);
    int denomExp = getExp(ogDenom);
    Value numerator = adaptor.getLhs();
    Value denominator = adaptor.getRhs();

    // Wide type must hold the widest operand shifted left by resWidth.
    // Equals 2*resWidth when both operands are resWidth wide (original).
    const int extWidth =
        std::max({resWidth, intWidth(numerator), intWidth(denominator)}) +
        resWidth;

    auto buildWideAttr = [extWidth](Builder b, int64_t value) -> IntegerAttr {
      return b.getIntegerAttr(b.getIntegerType(extWidth), value);
    };

    // reconcile arguments of different signedness; the shift constant takes
    // each operand's own type, since operands may not be resWidth wide.
    if (resType.getSignd() && (getSignd(ogNumerator) != getSignd(ogDenom))) {
      if (!getSignd(ogNumerator)) {
        numExp += 1;
        numerator =
            b.create<arith::ShRUIOp>(numerator, constLike(b, numerator, 1))
                .getResult();
      }
      if (!getSignd(ogDenom)) {
        denomExp += 1;
        denominator =
            b.create<arith::ShRUIOp>(denominator, constLike(b, denominator, 1))
                .getResult();
      }
    }

    // Extend operands to wide type.
    numerator =
        resType.getSignd()
            ? b.create<arith::ExtSIOp>(b.getIntegerType(extWidth), numerator)
                  .getResult()
            : b.create<arith::ExtUIOp>(b.getIntegerType(extWidth), numerator)
                  .getResult();
    denominator =
        resType.getSignd()
            ? b.create<arith::ExtSIOp>(b.getIntegerType(extWidth), denominator)
                  .getResult()
            : b.create<arith::ExtUIOp>(b.getIntegerType(extWidth), denominator)
                  .getResult();

    // Scale numerator by shifting left to preserve fractional precision.
    arith::ConstantOp scale_const =
        b.create<arith::ConstantOp>(buildWideAttr(b, resWidth));
    Value numerator_scaled =
        b.create<arith::ShLIOp>(numerator, scale_const).getResult();

    // Perform division.
    Value res = resType.getSignd()
                    ? b.create<arith::DivSIOp>(numerator_scaled, denominator)
                          .getResult()
                    : b.create<arith::DivUIOp>(numerator_scaled, denominator)
                          .getResult();

    // Compute correction factor.
    // Expected effective exponent after division: (numExp - denomExp -
    // resWidth) We correct by: resType.getExponent() - (numExp - denomExp -
    // resWidth)
    int correctionFactor = resType.getExponent() + resWidth - numExp + denomExp;

    if (correctionFactor != 0) {
      arith::ConstantOp correction_const = b.create<arith::ConstantOp>(
          buildWideAttr(b, std::abs(correctionFactor)));
      res = correctionFactor > 0
                ? (resType.getSignd()
                       ? b.create<arith::ShRSIOp>(res, correction_const)
                             .getResult()
                       : b.create<arith::ShRUIOp>(res, correction_const)
                             .getResult())
                : b.create<arith::ShLIOp>(res, correction_const).getResult();
    }

    // Truncate the result back to narrow type.
    res = b.create<arith::TruncIOp>(b.getIntegerType(ogWidth), res).getResult();
    rewriter.replaceOp(op, res);
    return success();
  }

  struct ConvertDiv : public OpConversionPattern<DivOp> {
    ConvertDiv(mlir::MLIRContext *context)
        : OpConversionPattern<DivOp>(context) {}

    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(DivOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
      return ConvertDivCommon<DivOp, OpAdaptor>(op, adaptor, rewriter);
    }
  };

  struct ConvertCastToReal : public OpConversionPattern<CastToRealOp> {
    ConvertCastToReal(mlir::MLIRContext *context)
        : OpConversionPattern<CastToRealOp>(context) {}

    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(CastToRealOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {

      ImplicitLocOpBuilder b(op.getLoc(), rewriter);

      FloatType fType = ::llvm::dyn_cast<FloatType>(op.getFrom().getType());
      int ogWidth = fType.getWidth();

      if (ogWidth > 64) {
        op->emitOpError()
            << "Conversion from floats bigger than f64 is not yet supported";
        return failure();
      }

      RealType resType = ::llvm::dyn_cast<RealType>(op->getResult(0).getType());

      if (resType.getBitwidth() > 64) {
        op->emitOpError()
            << "Conversion to fixpoints bigger than 64 is not yet supported";
        return failure();
      }

      auto buildIntAttr = [ogWidth](Builder b, int64_t value) -> IntegerAttr {
        return b.getIntegerAttr(b.getIntegerType(ogWidth), value);
      };

      // imagine a world where you are a programmer who decides the method
      // "getFPMantissaWidth" returns the width of the mantissa field + 1
      // because it accounts for the implicit bit for some godforsaken reason
      int mantissaBitwidth = fType.getFPMantissaWidth() - 1;

      IntegerType destType = b.getIntegerType(resType.getBitwidth());

      arith::BitcastOp bitcast =
          b.create<arith::BitcastOp>(b.getIntegerType(ogWidth), op.getFrom());

      int normalized_exp = resType.getExponent();

      // compute actual exponent in place
      IntegerAttr dtExp = buildIntAttr(
          b, (uint64_t)(std::abs(normalized_exp) << mantissaBitwidth));
      arith::ConstantOp dtExp_const = b.create<arith::ConstantOp>(dtExp);
      Value actual_exp =
          normalized_exp > 0
              ? b.create<arith::SubIOp>(bitcast, dtExp_const).getResult()
              : b.create<arith::AddIOp>(bitcast, dtExp_const).getResult();

      arith::BitcastOp bitcast_back =
          b.create<arith::BitcastOp>(fType, actual_exp);

      Value res =
          (resType.getSignd())
              ? b.create<arith::FPToSIOp>(destType, bitcast_back).getResult()
              : b.create<arith::FPToUIOp>(destType, bitcast_back).getResult();

      rewriter.replaceOp(op, res);
      return success();
    }
  };

  struct ConvertCastToFloat : public OpConversionPattern<CastToFloatOp> {
    ConvertCastToFloat(mlir::MLIRContext *context)
        : OpConversionPattern<CastToFloatOp>(context) {}

    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(CastToFloatOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {

      ImplicitLocOpBuilder b(op.getLoc(), rewriter);

      FloatType fType = ::llvm::dyn_cast<FloatType>(op.getRes().getType());

      int targetWidth = fType.getWidth();

      RealType fromType =
          ::llvm::dyn_cast<RealType>(op->getOperand(0).getType());

      if (fromType.getBitwidth() > 64) {
        op->emitOpError()
            << "Conversion from fixpoints bigger than 64 is not yet supported";
        return failure();
      }

      if (targetWidth > 64) {
        op->emitOpError()
            << "Conversion to floats bigger than f64 is not yet supported";
        return failure();
      }

      if (fromType.getExponent() >
          llvm::APFloat::semanticsMaxExponent(fType.getFloatSemantics())) {
        // maybe add an option for saturating conversion in the future?
        op->emitOpError()
            << "Target float type too small to represent real value";
        return failure();
      }

      auto buildIntAttr = [targetWidth](Builder b,
                                        int64_t value) -> IntegerAttr {
        return b.getIntegerAttr(b.getIntegerType(targetWidth), value);
      };

      // imagine a world where you are a programmer who decides the method
      // "getFPMantissaWidth" returns the width of the mantissa field + 1
      // because it accounts for the implicit bit for some godforsaken reason
      int mantissaBitwidth = fType.getFPMantissaWidth() - 1;

      Value conv =
          (fromType.getSignd())
              ? b.create<arith::SIToFPOp>(fType, adaptor.getFrom()).getResult()
              : b.create<arith::UIToFPOp>(fType, adaptor.getFrom()).getResult();

      // NOTE: there is no need to account for exponent bias here, as it's
      // already implicitly accounted for

      arith::BitcastOp bitcast =
          b.create<arith::BitcastOp>(b.getIntegerType(targetWidth), conv);

      // compute actual exponent in place
      IntegerAttr dtExp = buildIntAttr(
          b, (uint64_t)(std::abs(fromType.getExponent()) << mantissaBitwidth));
      arith::ConstantOp dtExp_const = b.create<arith::ConstantOp>(dtExp);
      Value actual_exp =
          fromType.getExponent() > 0
              ? b.create<arith::AddIOp>(bitcast, dtExp_const).getResult()
              : b.create<arith::SubIOp>(bitcast, dtExp_const).getResult();

      // bitcast back
      arith::BitcastOp bitcast_back =
          b.create<arith::BitcastOp>(fType, actual_exp);

      // FIX: the "add to the exponent bits directly" trick above only
      // correctly scales NON-ZERO values.
      Value zeroInt = b.create<arith::ConstantOp>(
          b.getIntegerAttr(b.getIntegerType(fromType.getBitwidth()), 0));
      Value isZero = b.create<arith::CmpIOp>(arith::CmpIPredicate::eq,
                                             adaptor.getFrom(), zeroInt);
      Value zeroFloat = b.create<arith::ConstantOp>(b.getFloatAttr(fType, 0.0));
      Value res = b.create<arith::SelectOp>(isZero, zeroFloat,
                                            bitcast_back.getResult());

      rewriter.replaceOp(op, res);
      return success();
    }
  };

  // Handles the case where raise-to-taffo inserts a GENERIC
  // builtin.unrealized_conversion_cast
  struct ConvertGenericCastToFloat
      : public OpConversionPattern<UnrealizedConversionCastOp> {
    ConvertGenericCastToFloat(mlir::MLIRContext *context)
        : OpConversionPattern<UnrealizedConversionCastOp>(context) {}

    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {

      if (op->getNumOperands() != 1 || op->getNumResults() != 1)
        return failure();

      RealType fromType =
          ::llvm::dyn_cast<RealType>(op->getOperand(0).getType());
      FloatType fType = ::llvm::dyn_cast<FloatType>(op->getResult(0).getType());

      // Only claim the specific real->float bridging case; let every other
      // unrealized_conversion_cast (including the reverse direction, and
      // any unrelated type pairs) pass through untouched for other
      // patterns / reconcile-unrealized-casts to handle.
      if (!fromType || !fType)
        return failure();

      ImplicitLocOpBuilder b(op.getLoc(), rewriter);

      int targetWidth = fType.getWidth();

      if (fromType.getBitwidth() > 64) {
        op->emitOpError()
            << "Conversion from fixpoints bigger than 64 is not yet supported";
        return failure();
      }

      if (targetWidth > 64) {
        op->emitOpError()
            << "Conversion to floats bigger than f64 is not yet supported";
        return failure();
      }

      if (fromType.getExponent() >
          llvm::APFloat::semanticsMaxExponent(fType.getFloatSemantics())) {
        op->emitOpError()
            << "Target float type too small to represent real value";
        return failure();
      }

      auto buildIntAttr = [targetWidth](Builder b,
                                        int64_t value) -> IntegerAttr {
        return b.getIntegerAttr(b.getIntegerType(targetWidth), value);
      };

      int mantissaBitwidth = fType.getFPMantissaWidth() - 1;

      Value conv =
          (fromType.getSignd())
              ? b.create<arith::SIToFPOp>(fType, adaptor.getOperands()[0])
                    .getResult()
              : b.create<arith::UIToFPOp>(fType, adaptor.getOperands()[0])
                    .getResult();

      arith::BitcastOp bitcast =
          b.create<arith::BitcastOp>(b.getIntegerType(targetWidth), conv);

      IntegerAttr dtExp = buildIntAttr(
          b, (uint64_t)(std::abs(fromType.getExponent()) << mantissaBitwidth));
      arith::ConstantOp dtExp_const = b.create<arith::ConstantOp>(dtExp);
      Value actual_exp =
          fromType.getExponent() > 0
              ? b.create<arith::AddIOp>(bitcast, dtExp_const).getResult()
              : b.create<arith::SubIOp>(bitcast, dtExp_const).getResult();

      arith::BitcastOp bitcast_back =
          b.create<arith::BitcastOp>(fType, actual_exp);

      // The "add to the exponent bits directly" trick above only
      // correctly scales NON-ZERO values.
      Value zeroInt = b.create<arith::ConstantOp>(
          b.getIntegerAttr(b.getIntegerType(fromType.getBitwidth()), 0));
      Value isZero = b.create<arith::CmpIOp>(arith::CmpIPredicate::eq,
                                             adaptor.getOperands()[0], zeroInt);
      Value zeroFloat = b.create<arith::ConstantOp>(b.getFloatAttr(fType, 0.0));
      Value res = b.create<arith::SelectOp>(isZero, zeroFloat,
                                            bitcast_back.getResult());

      rewriter.replaceOp(op, res);
      return success();
    }
  };

  struct ConvertBitcastToReal : public OpConversionPattern<BitcastToRealOp> {
    ConvertBitcastToReal(mlir::MLIRContext *context)
        : OpConversionPattern<BitcastToRealOp>(context) {}

    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(BitcastToRealOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {

      rewriter.replaceOp(op, adaptor.getFrom());
      return success();
    }
  };

  struct ConvertBitcastToInt : public OpConversionPattern<BitcastToIntOp> {
    ConvertBitcastToInt(mlir::MLIRContext *context)
        : OpConversionPattern<BitcastToIntOp>(context) {}

    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(BitcastToIntOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {

      rewriter.replaceOp(op, adaptor.getFrom());
      return success();
    }
  };

  struct ConvertAlign : public OpConversionPattern<AlignOp> {
    ConvertAlign(mlir::MLIRContext *context)
        : OpConversionPattern<AlignOp>(context) {}

    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(AlignOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {

      ImplicitLocOpBuilder b(op.getLoc(), rewriter);

      RealType source = op.getFrom().getType();
      RealType target = op.getRes().getType();

      if (source.getBitwidth() > target.getBitwidth()) {
        // shift first
        int expDiff = target.getExponent() - source.getExponent();
        arith::ConstantOp align_res = b.create<arith::ConstantOp>(
            b.getIntegerAttr(b.getIntegerType(source.getBitwidth()), expDiff));
        Value shifted =
            source.getSignd()
                ? b.create<arith::ShRSIOp>(adaptor.getFrom(), align_res)
                      .getResult()
                : b.create<arith::ShRUIOp>(adaptor.getFrom(), align_res)
                      .getResult();
        // then trunc
        Value res = b.create<arith::TruncIOp>(
            b.getIntegerType(target.getBitwidth()), shifted);
        rewriter.replaceOp(op, res);
      } else {
        Value source_val = adaptor.getFrom();
        // we need this check because ext to same datatype isn't a valid op
        // (idk why they don't simply have a folder for this)
        if (source.getBitwidth() < target.getBitwidth()) {
          // ext first
          source_val = source.getSignd()
                           ? b.create<arith::ExtSIOp>(
                                  b.getIntegerType(target.getBitwidth()),
                                  adaptor.getFrom())
                                 .getResult()
                           : b.create<arith::ExtUIOp>(
                                  b.getIntegerType(target.getBitwidth()),
                                  adaptor.getFrom())
                                 .getResult();
        }

        // then shift
        int expDiff = target.getExponent() - source.getExponent();
        arith::ConstantOp align_res =
            b.create<arith::ConstantOp>(b.getIntegerAttr(
                b.getIntegerType(target.getBitwidth()), std::abs(expDiff)));
        Value res =
            expDiff > 0
                ? source.getSignd()
                      ? b.create<arith::ShRSIOp>(source_val, align_res)
                            .getResult()
                      : b.create<arith::ShRUIOp>(source_val, align_res)
                            .getResult()
                : b.create<arith::ShLIOp>(source_val, align_res).getResult();
        rewriter.replaceOp(op, res);
      }
      return success();
    }
  };

  struct ConvertLoopRegionTypes : public OpConversionPattern<scf::ForOp> {
    ConvertLoopRegionTypes(mlir::MLIRContext *context)
        : OpConversionPattern<scf::ForOp>(context) {}

    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {

      Block *body = op.getBody();

      Region &region = op->getRegion(0);

      rewriter.startOpModification(op);
      auto terminator = cast<scf::YieldOp>(body->getTerminator());
      SmallVector<Value> terminatorRes;
      if (failed(rewriter.getRemappedValues(terminator->getOperands(),
                                            terminatorRes)))
        return failure();
      rewriter.modifyOpInPlace(terminator,
                               [&] { terminator->setOperands(terminatorRes); });

      rewriter.finalizeOpModification(op);

      if (failed(rewriter.convertRegionTypes(&region, *getTypeConverter())))
        return failure();

      ImplicitLocOpBuilder b(op.getLoc(), rewriter);

      auto newOp =
          b.create<scf::ForOp>(adaptor.getLowerBound(), adaptor.getUpperBound(),
                               adaptor.getStep(), adaptor.getInitArgs());

      // We do not need the empty block created by rewriter.
      rewriter.eraseBlock(newOp.getBody(0));
      // Inline the type converted region from the original operation.
      rewriter.inlineRegionBefore(op.getRegion(), newOp.getRegion(),
                                  newOp.getRegion().end());

      rewriter.replaceOp(op, newOp);
      return success();
    }
  };

  // The matmul reduction loops don't use scf.for iter_args at all (confirmed
  // via debug instrumentation: numRegionIterArgs == 0 for every one of
  // them).
  struct RewriteMemAccumulator : public OpRewritePattern<memref::AllocaOp> {
    const TaffoToArithTypeConverter &typeConverter;

    RewriteMemAccumulator(const TaffoToArithTypeConverter &tc,
                          MLIRContext *context)
        : OpRewritePattern<memref::AllocaOp>(context), typeConverter(tc) {}

    // Walks up from `user` looking for an affine.for/scf.for ancestor that
    // is nested INSIDE the alloca's own block (i.e. the loop containing
    // this use). Returns nullptr if `user` is directly in the alloca's own
    // block (not inside any loop relative to the alloca).
    Operation *findEnclosingLoop(Operation *user, Block *allocaBlock) const {
      Operation *cur = user;
      while (cur && cur->getBlock() != allocaBlock) {
        if (isa<scf::ForOp>(cur->getParentOp()))
          return cur->getParentOp();
        cur = cur->getParentOp();
      }
      return nullptr;
    }

    LogicalResult matchAndRewrite(memref::AllocaOp allocaOp,
                                  PatternRewriter &rewriter) const override {
      auto memrefType = dyn_cast<MemRefType>(allocaOp.getType());
      if (!memrefType || !memrefType.getElementType().isF32() ||
          memrefType.getRank() != 0) {
        return failure();
      }

      Block *allocaBlock = allocaOp->getBlock();

      memref::StoreOp initStore;
      Operation *loop = nullptr;
      memref::LoadOp bodyLoad;
      memref::StoreOp bodyStore;
      SmallVector<memref::LoadOp> postLoopLoads;

      for (Operation *user : allocaOp->getUsers()) {
        Operation *enclosingLoop = findEnclosingLoop(user, allocaBlock);

        if (!enclosingLoop) {
          // Directly in the alloca's own block: either the init store, or
          // a post-loop load.
          if (auto store = dyn_cast<memref::StoreOp>(user)) {
            if (initStore) {
              return failure();
            }
            initStore = store;
          } else if (auto load = dyn_cast<memref::LoadOp>(user)) {
            postLoopLoads.push_back(load);
          } else {
            return failure();
          }
          continue;
        }

        // Inside a loop.
        if (loop && loop != enclosingLoop) {
          return failure();
        }
        loop = enclosingLoop;

        if (auto load = dyn_cast<memref::LoadOp>(user)) {
          if (bodyLoad) {
            return failure();
          }
          bodyLoad = load;
        } else if (auto store = dyn_cast<memref::StoreOp>(user)) {
          if (bodyStore) {
            return failure();
          }
          bodyStore = store;
        } else {
          return failure();
        }
      }

      if (!initStore || !loop || !bodyLoad || !bodyStore) {
        return failure();
      }

      // Verify the body's load -> cast -> accumulate -> cast -> store chain.
      if (!bodyLoad->getResult(0).hasOneUse()) {
        return failure();
      }
      Operation *afterLoad = *bodyLoad->getResult(0).getUsers().begin();
      // Accept either a generic bridging cast OR a real, already-annotated
      // taffo.cast2real
      Operation *castToReal = nullptr;
      if (isa<UnrealizedConversionCastOp>(afterLoad) ||
          isa<CastToRealOp>(afterLoad)) {
        if (afterLoad->getNumResults() == 1 &&
            isa<RealType>(afterLoad->getResult(0).getType()))
          castToReal = afterLoad;
      }
      if (!castToReal) {
        return failure();
      }

      if (!castToReal->getResult(0).hasOneUse()) {
        return failure();
      }
      Operation *afterCast = *castToReal->getResult(0).getUsers().begin();

      // Some accumulators (e.g. fc3) have an intermediate taffo.align
      // (exponent realignment) between the cast and the actual add
      Operation *alignOp = nullptr;
      Operation *accumOp = afterCast;
      if (isa<AlignOp>(afterCast)) {
        alignOp = afterCast;
        if (!alignOp->getResult(0).hasOneUse()) {
          return failure();
        }
        accumOp = *alignOp->getResult(0).getUsers().begin();
      }

      if (!isa<AddOp>(accumOp)) {
        return failure();
      }

      // The native representation used throughout this rewrite is always
      // accumOp's own result type (what's actually produced and persisted
      // each iteration)

      if (!accumOp->getResult(0).hasOneUse()) {
        return failure();
      }
      Operation *afterAccum = *accumOp->getResult(0).getUsers().begin();
      // Same widening for the exit side: accept either cast form.
      Operation *castToFloat = nullptr;
      if (isa<UnrealizedConversionCastOp>(afterAccum) ||
          isa<CastToFloatOp>(afterAccum)) {
        if (afterAccum->getNumResults() == 1 &&
            afterAccum->getResult(0).getType().isF32())
          castToFloat = afterAccum;
      }
      if (!castToFloat) {
        return failure();
      }

      if (bodyStore.getValueToStore() != castToFloat->getResult(0)) {
        return failure();
      }

      RealType accumType = cast<RealType>(accumOp->getResult(0).getType());
      Type nativeType = typeConverter.convertType(accumType);
      if (!nativeType)
        return failure();

      // --- All checks passed: rewrite. ---
      ImplicitLocOpBuilder b(allocaOp.getLoc(), rewriter);

      // 1. New alloca with the native element type.
      auto newMemrefType = MemRefType::get({}, nativeType);
      b.setInsertionPoint(allocaOp);
      auto newAlloca = b.create<memref::AllocaOp>(newMemrefType);

      // 2. Init store: every accumulator init value seen in this codebase
      // is the literal constant 0.0
      b.setInsertionPoint(initStore);
      Value initF32 = initStore.getValueToStore();
      auto initConstOp = initF32.getDefiningOp<arith::ConstantOp>();
      bool initIsZero = false;
      if (initConstOp) {
        if (auto fAttr = dyn_cast<FloatAttr>(initConstOp.getValue()))
          initIsZero = fAttr.getValue().isZero();
      }
      if (!initIsZero) {
        return failure();
      }
      Value nativeZero =
          b.create<arith::ConstantOp>(rewriter.getZeroAttr(nativeType));
      b.create<memref::StoreOp>(nativeZero, newAlloca, initStore.getIndices());
      rewriter.eraseOp(initStore);

      // 3. Body: load/store natively.
      b.setInsertionPoint(bodyLoad);
      auto newBodyLoad =
          b.create<memref::LoadOp>(newAlloca, bodyLoad.getIndices());
      auto reboxToReal =
          b.create<BitcastToRealOp>(accumType, newBodyLoad.getResult());
      // Redirect the correct op's uses
      // same as before.
      if (alignOp) {
        rewriter.replaceOp(alignOp, reboxToReal.getResult());
        rewriter.eraseOp(castToReal);
      } else {
        rewriter.replaceOp(castToReal, reboxToReal.getResult());
      }
      rewriter.eraseOp(bodyLoad);

      b.setInsertionPoint(bodyStore);
      auto reboxToInt =
          b.create<BitcastToIntOp>(nativeType, accumOp->getResult(0));
      b.create<memref::StoreOp>(reboxToInt.getResult(), newAlloca,
                                bodyStore.getIndices());
      rewriter.eraseOp(bodyStore);
      rewriter.eraseOp(castToFloat);

      // 4. Post-loop loads:
      for (auto load : postLoopLoads) {
        b.setInsertionPoint(load);
        auto newLoad = b.create<memref::LoadOp>(newAlloca, load.getIndices());
        auto reboxToReal =
            b.create<BitcastToRealOp>(accumType, newLoad.getResult());
        auto resultCast = b.create<UnrealizedConversionCastOp>(
            TypeRange{rewriter.getF32Type()},
            ValueRange{reboxToReal.getResult()});
        rewriter.replaceOp(load, resultCast.getResults());
      }

      rewriter.eraseOp(allocaOp);
      return success();
    }
  };

  struct RewriteLoopAccumulator : public OpRewritePattern<scf::ForOp> {
    const TaffoToArithTypeConverter &typeConverter;

    RewriteLoopAccumulator(const TaffoToArithTypeConverter &tc,
                           MLIRContext *context)
        : OpRewritePattern<scf::ForOp>(context), typeConverter(tc) {}

    LogicalResult matchAndRewrite(scf::ForOp forOp,
                                  PatternRewriter &rewriter) const override {
      if (forOp.getInitArgs().size() != 1 || forOp.getNumResults() != 1)
        return failure();
      if (!forOp.getRegionIterArgs()[0].getType().isF32() ||
          !forOp.getResultTypes()[0].isF32())
        return failure();

      BlockArgument iterArg = forOp.getRegionIterArgs()[0];

      if (!iterArg.hasOneUse())
        return failure();
      Operation *afterIterArg = *iterArg.getUsers().begin();
      Operation *castToReal = nullptr;
      if (isa<UnrealizedConversionCastOp>(afterIterArg) ||
          isa<CastToRealOp>(afterIterArg)) {
        if (afterIterArg->getNumResults() == 1 &&
            isa<RealType>(afterIterArg->getResult(0).getType()))
          castToReal = afterIterArg;
      }
      if (!castToReal)
        return failure();

      if (!castToReal->getResult(0).hasOneUse())
        return failure();
      Operation *afterCast = *castToReal->getResult(0).getUsers().begin();

      if (isa<AlignOp>(afterCast))
        return failure();

      if (!isa<AddOp>(afterCast))
        return failure();
      Operation *accumOp = afterCast;

      if (!accumOp->getResult(0).hasOneUse())
        return failure();
      Operation *afterAccum = *accumOp->getResult(0).getUsers().begin();
      Operation *castToFloat = nullptr;
      if (isa<UnrealizedConversionCastOp>(afterAccum) ||
          isa<CastToFloatOp>(afterAccum)) {
        if (afterAccum->getNumResults() == 1 &&
            afterAccum->getResult(0).getType().isF32())
          castToFloat = afterAccum;
      }
      if (!castToFloat)
        return failure();

      auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
      if (yieldOp.getOperand(0) != castToFloat->getResult(0))
        return failure();

      RealType accumType = cast<RealType>(accumOp->getResult(0).getType());
      Type nativeType = typeConverter.convertType(accumType);
      if (!nativeType)
        return failure();

      ImplicitLocOpBuilder b(forOp.getLoc(), rewriter);

      b.setInsertionPoint(forOp);
      Value initF32 = forOp.getInitArgs()[0];
      auto initReal = b.create<CastToRealOp>(accumType, initF32);
      auto initNative =
          b.create<BitcastToIntOp>(nativeType, initReal.getResult());

      auto newForOp = b.create<scf::ForOp>(
          forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep(),
          ValueRange{initNative.getResult()});

      rewriter.eraseBlock(newForOp.getBody());
      rewriter.inlineRegionBefore(forOp.getRegion(), newForOp.getRegion(),
                                  newForOp.getRegion().end());

      Block *newBody = newForOp.getBody();
      BlockArgument oldIterArg = newBody->getArgument(1);
      BlockArgument newIterArg =
          newBody->insertArgument(1u, nativeType, oldIterArg.getLoc());

      b.setInsertionPoint(castToReal);
      auto reboxToReal = b.create<BitcastToRealOp>(accumType, newIterArg);
      rewriter.replaceOp(castToReal, reboxToReal.getResult());

      b.setInsertionPoint(castToFloat);
      auto reboxToInt =
          b.create<BitcastToIntOp>(nativeType, accumOp->getResult(0));
      rewriter.replaceOp(castToFloat, reboxToInt.getResult());

      oldIterArg.dropAllUses();
      newBody->eraseArgument(2u);

      rewriter.replaceOp(forOp, newForOp.getResult(0));
      return success();
    }
  };

  struct RewritePassthroughLoop : public OpRewritePattern<scf::ForOp> {
    RewritePassthroughLoop(MLIRContext *context)
        : OpRewritePattern<scf::ForOp>(context) {}

    RealType findInnerAccumType(scf::ForOp innerForOp) const {
      // Recursive, not just one level deep: for an outermost wrapper
      // whose own "inner loop" is itself another wrapper (not the
      // accumulator loop directly), the BitcastToRealOp sits two or more
      // levels deeper still
      RealType found = nullptr;
      innerForOp->walk([&](BitcastToRealOp bcast) {
        found = cast<RealType>(bcast.getResult().getType());
        return WalkResult::interrupt();
      });
      return found;
    }

    LogicalResult matchAndRewrite(scf::ForOp forOp,
                                  PatternRewriter &rewriter) const override {
      if (forOp.getInitArgs().size() != 1 || forOp.getNumResults() != 1)
        return failure();
      if (!forOp.getRegionIterArgs()[0].getType().isF32() ||
          !forOp.getResultTypes()[0].isF32())
        return failure();

      BlockArgument iterArg = forOp.getRegionIterArgs()[0];
      if (!iterArg.hasOneUse())
        return failure();

      // The body must yield exactly the result of a single inner scf.for,
      // itself living directly in this same body
      Block *body = forOp.getBody();
      auto yieldOp = cast<scf::YieldOp>(body->getTerminator());
      if (yieldOp.getNumOperands() != 1)
        return failure();
      auto innerForOp = yieldOp.getOperand(0).getDefiningOp<scf::ForOp>();
      if (!innerForOp || innerForOp->getBlock() != body)
        return failure();
      if (innerForOp.getInitArgs().size() != 1 ||
          innerForOp.getNumResults() != 1)
        return failure();

      Type innerResultType = innerForOp.getResultTypes()[0];
      if (innerResultType.isF32())
        return failure(); // inner loop not yet rewritten -- not our turn yet

      RealType accumType = findInnerAccumType(innerForOp);
      if (!accumType)
        return failure();

      // --- All checks passed: rewrite. ---
      ImplicitLocOpBuilder b(forOp.getLoc(), rewriter);

      b.setInsertionPoint(forOp);
      Value initF32 = forOp.getInitArgs()[0];
      auto initReal = b.create<CastToRealOp>(accumType, initF32);
      auto initNative =
          b.create<BitcastToIntOp>(innerResultType, initReal.getResult());

      auto newForOp = b.create<scf::ForOp>(
          forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep(),
          ValueRange{initNative.getResult()});

      rewriter.eraseBlock(newForOp.getBody());
      rewriter.inlineRegionBefore(forOp.getRegion(), newForOp.getRegion(),
                                  newForOp.getRegion().end());

      Block *newBody = newForOp.getBody();
      BlockArgument oldIterArg = newBody->getArgument(1);
      BlockArgument newIterArg =
          newBody->insertArgument(1u, innerResultType, oldIterArg.getLoc());

      // Walk the (possibly empty) chain of single-use, single-result
      // bridge ops from oldIterArg forward to wherever it actually feeds
      SmallVector<Operation *> deadChain;
      {
        Value cur = oldIterArg;
        while (cur.hasOneUse()) {
          Operation *user = *cur.getUsers().begin();
          if (user == innerForOp.getOperation())
            break; // reached the end of the chain -- don't include innerForOp
          if (user->getNumResults() != 1)
            break; // chain broken; bail out safely
          deadChain.push_back(user);
          cur = user->getResult(0);
        }
      }

      // Rewire the inner loop's own init operand directly to the new,
      // already-native outer iter_arg
      innerForOp.getInitArgsMutable()[0].set(newIterArg);

      // Now that innerForOp no longer references the chain, erase it in
      // reverse order (last-fed-op first), so each op has zero remaining
      // uses at the moment it's erased.
      for (Operation *deadOp : llvm::reverse(deadChain))
        rewriter.eraseOp(deadOp);

      newBody->eraseArgument(2u);

      auto newYieldOp = cast<scf::YieldOp>(newBody->getTerminator());
      b.setInsertionPoint(newYieldOp);
      b.create<scf::YieldOp>(ValueRange{innerForOp.getResult(0)});
      rewriter.eraseOp(newYieldOp);

      rewriter.replaceOp(forOp, newForOp.getResult(0));
      return success();
    }
  };

  struct RewriteFinalResultBridge : public OpRewritePattern<scf::ForOp> {
    const TaffoToArithTypeConverter &typeConverter;

    RewriteFinalResultBridge(const TaffoToArithTypeConverter &tc,
                             MLIRContext *context)
        : OpRewritePattern<scf::ForOp>(context), typeConverter(tc) {}

    LogicalResult matchAndRewrite(scf::ForOp forOp,
                                  PatternRewriter &rewriter) const override {
      if (forOp.getNumResults() != 1)
        return failure();
      Type resultType = forOp.getResultTypes()[0];
      if (resultType.isF32() || isa<RealType>(resultType))
        return failure();

      if (forOp.getResult(0).use_empty())
        return failure();

      // Load-bearing guard:
      for (Operation *user : forOp.getResult(0).getUsers()) {
        if (isa<scf::YieldOp>(user))
          return failure();
      }

      // Idempotency guard:
      for (Operation *user : forOp.getResult(0).getUsers()) {
        if (isa<BitcastToRealOp>(user))
          return failure();
      }

      RealType accumType = nullptr;
      forOp->walk([&](BitcastToRealOp bcast) {
        accumType = cast<RealType>(bcast.getResult().getType());
        return WalkResult::interrupt();
      });
      if (!accumType)
        return failure();

      ImplicitLocOpBuilder b(forOp.getLoc(), rewriter);
      b.setInsertionPointAfter(forOp);
      auto resultToReal =
          b.create<BitcastToRealOp>(accumType, forOp.getResult(0));
      auto resultToFloat = b.create<UnrealizedConversionCastOp>(
          TypeRange{rewriter.getF32Type()},
          ValueRange{resultToReal.getResult()});

      rewriter.replaceAllUsesExcept(forOp.getResult(0),
                                    resultToFloat.getResult(0), resultToReal);
      return success();
    }
  };

  // Converts one weight tensor's DenseElementsAttr from float to
  // fixed-point, using the same op sequence ConvertCastToReal emits at
  // runtime, but folded once via createOrFold instead of run per-element.
  // Returns null (with `diag` set) if folding doesn't fully collapse to a
  // constant -- callers must skip the optimization for this global rather
  // than leave a partially-folded expression behind.
  static DenseElementsAttr preconvertWeightTensor(OpBuilder &builder,
                                                  Location loc,
                                                  DenseElementsAttr floatData,
                                                  RealType targetType,
                                                  std::string &diag) {
    auto floatTensorType = cast<RankedTensorType>(floatData.getType());
    FloatType fType = cast<FloatType>(floatTensorType.getElementType());
    int ogWidth = fType.getWidth();

    if (ogWidth > 64) {
      diag = "source float width > 64 not supported";
      return nullptr;
    }
    if (targetType.getBitwidth() > 64) {
      diag = "target fixed-point width > 64 not supported";
      return nullptr;
    }

    // Same mantissa/exponent-field math as ConvertCastToReal.
    int mantissaBitwidth = fType.getFPMantissaWidth() - 1;
    IntegerType srcIntType = builder.getIntegerType(ogWidth);
    IntegerType destType = builder.getIntegerType(targetType.getBitwidth());
    auto destTensorType =
        RankedTensorType::get(floatTensorType.getShape(), destType);

    // Constant tensor to fold against.
    Value constTensor =
        builder.create<arith::ConstantOp>(loc, floatTensorType, floatData);

    // Reinterpret float bits as int.
    Value bitcast = builder.createOrFold<arith::BitcastOp>(
        loc, RankedTensorType::get(floatTensorType.getShape(), srcIntType),
        constTensor);

    // Shift the exponent field by the target fixed-point exponent.
    int normalized_exp = targetType.getExponent();
    auto dtExpSplat = DenseElementsAttr::get(
        RankedTensorType::get(floatTensorType.getShape(), srcIntType),
        builder.getIntegerAttr(srcIntType, (uint64_t)(std::abs(normalized_exp)
                                                      << mantissaBitwidth)));
    Value dtExpConst = builder.create<arith::ConstantOp>(
        loc, RankedTensorType::get(floatTensorType.getShape(), srcIntType),
        dtExpSplat);
    Value actualExp =
        normalized_exp > 0
            ? builder.createOrFold<arith::SubIOp>(loc, bitcast, dtExpConst)
            : builder.createOrFold<arith::AddIOp>(loc, bitcast, dtExpConst);

    // Reinterpret back as float (now scaled by 2^-exponent), then truncate
    // to the target integer type.
    Value bitcastBack =
        builder.createOrFold<arith::BitcastOp>(loc, floatTensorType, actualExp);
    Value res = targetType.getSignd()
                    ? builder.createOrFold<arith::FPToSIOp>(loc, destTensorType,
                                                            bitcastBack)
                    : builder.createOrFold<arith::FPToUIOp>(loc, destTensorType,
                                                            bitcastBack);

    // Every input was constant, so `res` should be a fresh arith.constant
    // createOrFold inserted -- extract its data directly.
    if (auto constOp = res.getDefiningOp<arith::ConstantOp>()) {
      if (auto elements = dyn_cast<DenseElementsAttr>(constOp.getValue()))
        return elements;
    }

    diag = "constant folding did not fully collapse to a single constant";
    return nullptr;
  }

  // Finds every krnl.global feeding a CastToRealOp (via a load) with a
  // consistent target RealType, and builds its pre-converted integer
  // twin. Returns original-global -> new-global (or null if declined).
  static DenseMap<Operation *, Operation *>
  preconvertConstantWeights(Operation *module,
                            const TaffoToArithTypeConverter &typeConverter) {
    DenseMap<Operation *, Operation *> converted;
    OpBuilder builder(module->getContext());

    int totalCastToReal = 0, notFromLoad = 0, loadNotFromGlobal = 0,
        alreadyHandled = 0, reachedEnd = 0;

    module->walk([&](CastToRealOp castOp) {
      totalCastToReal++;
      // Must come from a load directly off a krnl.global.
      Operation *loadOp = castOp.getFrom().getDefiningOp();
      if (!loadOp || !(isa<::mlir::affine::AffineLoadOp>(loadOp) ||
                       isa<memref::LoadOp>(loadOp))) {
        notFromLoad++;
        return;
      }
      Value memrefOperand = loadOp->getOperand(0);
      Operation *globalOp = memrefOperand.getDefiningOp();
      if (!globalOp || globalOp->getName().getStringRef() != "krnl.global") {
        loadNotFromGlobal++;
        return;
      }
      if (converted.count(globalOp)) {
        alreadyHandled++;
        return; // already handled or already declined
      }
      reachedEnd++;

      // Every load off this global must agree on the same RealType.
      RealType targetType = cast<RealType>(castOp.getResult().getType());
      bool consistent = true;
      for (Operation *user : memrefOperand.getUsers()) {
        bool isLoad = isa<::mlir::affine::AffineLoadOp>(user) ||
                      isa<memref::LoadOp>(user);
        if (!isLoad)
          continue;
        for (Operation *loadUser : user->getResult(0).getUsers()) {
          auto otherCast = dyn_cast<CastToRealOp>(loadUser);
          if (!otherCast)
            continue;
          if (cast<RealType>(otherCast.getResult().getType()) != targetType)
            consistent = false;
        }
      }
      if (!consistent) {
        llvm::errs() << "[weight-preconvert] declined: inconsistent RealType "
                        "across loads\n";
        converted[globalOp] = nullptr;
        return;
      }

      // Pull the float data out and convert it.
      auto valueAttr = globalOp->getAttr("value");
      auto floatData = dyn_cast_or_null<DenseElementsAttr>(valueAttr);
      if (!floatData) {
        llvm::errs() << "[weight-preconvert] declined: value attr is not "
                        "DenseElementsAttr\n";
        converted[globalOp] = nullptr;
        return;
      }

      builder.setInsertionPoint(globalOp);
      std::string diag;
      DenseElementsAttr intData = preconvertWeightTensor(
          builder, globalOp->getLoc(), floatData, targetType, diag);
      if (!intData) {
        globalOp->emitRemark() << "skipping weight pre-conversion: " << diag;
        converted[globalOp] = nullptr;
        return;
      }

      // Rebuild krnl.global explicitly with its three known attributes
      // (name, shape, value), rather than generically copying whatever
      // getAttrDictionary() returns -- these are likely stored as ODS
      // properties, not plain discardable attributes, so a generic copy
      // from the dictionary can silently miss them.
      auto origMemrefType = cast<MemRefType>(globalOp->getResult(0).getType());
      Type nativeElemType = typeConverter.convertType(targetType);
      auto newMemrefType =
          MemRefType::get(origMemrefType.getShape(), nativeElemType);

      auto origNameAttr = cast<StringAttr>(globalOp->getAttr("name"));
      auto newNameAttr =
          builder.getStringAttr(origNameAttr.getValue().str() + "_fixedpoint");
      auto shapeAttr = globalOp->getAttr("shape");

      SmallVector<NamedAttribute> newAttrs;
      newAttrs.push_back(builder.getNamedAttr("name", newNameAttr));
      newAttrs.push_back(builder.getNamedAttr("shape", shapeAttr));
      newAttrs.push_back(builder.getNamedAttr("value", intData));

      OperationState state(globalOp->getLoc(), "krnl.global");
      state.addAttributes(newAttrs);
      state.addTypes({newMemrefType});
      Operation *newOp = builder.create(state);
      llvm::errs() << "[weight-preconvert] created new global, parent block="
                   << (newOp->getBlock() != nullptr) << ", text: ";
      newOp->print(llvm::errs());
      llvm::errs() << "\n";
      converted[globalOp] = newOp;
    });

    int genuineSuccesses = 0;
    for (auto &entry : converted)
      if (entry.second)
        genuineSuccesses++;

    llvm::errs() << "[weight-preconvert] total CastToRealOp visited: "
                 << totalCastToReal << ", not from load: " << notFromLoad
                 << ", load not from krnl.global: " << loadNotFromGlobal
                 << ", already handled: " << alreadyHandled
                 << ", reached conversion attempt: " << reachedEnd
                 << ", genuinely converted: " << genuineSuccesses << "\n";
    return converted;
  }

  // Redirects a weight load+cast to load from the pre-converted global
  // instead, replacing the expensive CastToRealOp with a cheap
  // BitcastToRealOp. Read-only over the map built by Step 1.
  struct RewriteConstantWeightLoad : public OpRewritePattern<CastToRealOp> {
    const DenseMap<Operation *, Operation *> &convertedGlobals;

    RewriteConstantWeightLoad(
        const DenseMap<Operation *, Operation *> &converted,
        MLIRContext *context)
        : OpRewritePattern<CastToRealOp>(context), convertedGlobals(converted) {
    }

    LogicalResult matchAndRewrite(CastToRealOp castOp,
                                  PatternRewriter &rewriter) const override {
      // Weight reads seen in practice are memref.load, not affine.load --
      // handle both, mirroring Step 1's own check exactly.
      Operation *loadOp = castOp.getFrom().getDefiningOp();
      bool isAffineLoad = loadOp && isa<::mlir::affine::AffineLoadOp>(loadOp);
      bool isMemrefLoad = loadOp && isa<memref::LoadOp>(loadOp);
      if (!isAffineLoad && !isMemrefLoad)
        return failure();

      Value memrefOperand = loadOp->getOperand(0);
      Operation *globalOp = memrefOperand.getDefiningOp();
      if (!globalOp)
        return failure();

      auto it = convertedGlobals.find(globalOp);
      if (it == convertedGlobals.end() || !it->second)
        return failure(); // not eligible, or declined by Step 1

      Operation *newGlobal = it->second;
      RealType resType = cast<RealType>(castOp.getResult().getType());

      // Same indices, new (integer) memref, cheap bitcast instead of the
      // full runtime conversion.
      ImplicitLocOpBuilder b(castOp.getLoc(), rewriter);
      b.setInsertionPoint(loadOp);
      Value newLoadResult;
      if (isAffineLoad) {
        auto load = cast<::mlir::affine::AffineLoadOp>(loadOp);
        newLoadResult = b.create<::mlir::affine::AffineLoadOp>(
                             newGlobal->getResult(0), load.getAffineMap(),
                             load.getMapOperands())
                            .getResult();
      } else {
        auto load = cast<memref::LoadOp>(loadOp);
        newLoadResult =
            b.create<memref::LoadOp>(newGlobal->getResult(0), load.getIndices())
                .getResult();
      }
      auto bridged = b.create<BitcastToRealOp>(resType, newLoadResult);
      rewriter.replaceOp(castOp, bridged.getResult());

      // Old load cleaned up by the greedy driver's own dead-code removal
      // once it has no remaining uses.
      return success();
    }
  };

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    mlir::Operation *module = getOperation();

    TaffoToArithTypeConverter typeConverter(context);

    // Run this FIRST, as a separate, plain greedy rewrite
    {
      DenseMap<Operation *, Operation *> convertedWeights =
          preconvertConstantWeights(module, typeConverter);

      RewritePatternSet accumPatterns(context);
      accumPatterns.add<RewriteMemAccumulator>(typeConverter, context);
      accumPatterns.add<RewriteLoopAccumulator>(typeConverter, context);
      accumPatterns.add<RewritePassthroughLoop>(context);
      accumPatterns.add<RewriteFinalResultBridge>(typeConverter, context);
      accumPatterns.add<RewriteConstantWeightLoad>(convertedWeights, context);
      GreedyRewriteConfig accumConfig;
      accumConfig.setMaxIterations(100);
      if (failed(applyPatternsGreedily(module, std::move(accumPatterns),
                                       accumConfig))) {
        signalPassFailure();
        return;
      }
    }

    ConversionTarget target(*context);
    target.markUnknownOpDynamicallyLegal([](Operation *op) { return true; });
    target.addIllegalDialect<TaffoDialect>();

    // builtin.unrealized_conversion_cast isn't part of TaffoDialect, so it
    // falls under markUnknownOpDynamicallyLegal's catch-all (legal by
    // default)
    // bridges elsewhere) remains legal and untouched.
    target.addDynamicallyLegalOp<UnrealizedConversionCastOp>(
        [](UnrealizedConversionCastOp op) {
          if (op->getNumOperands() != 1 || op->getNumResults() != 1)
            return true; // legal: not the pattern we care about
          bool isRealToFloat =
              ::llvm::isa<RealType>(op->getOperand(0).getType()) &&
              ::llvm::isa<FloatType>(op->getResult(0).getType());
          return !isRealToFloat; // illegal (needs conversion) only when true
        });

    RewritePatternSet patterns(context);

    patterns
        .add<ConvertAdd, ConvertSub, ConvertMult, ConvertDiv, ConvertCastToReal,
             ConvertCastToFloat, ConvertGenericCastToFloat, ConvertBitcastToInt,
             ConvertBitcastToReal, ConvertAlign>(typeConverter, context);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
    target.addDynamicallyLegalOp<scf::ForOp>(
        [&](scf::ForOp op) { return typeConverter.isLegal(op); });

    RewritePatternSet loopPatterns(context);
    loopPatterns.add<ConvertLoopRegionTypes>(typeConverter, context);
    if (failed(
            applyPartialConversion(module, target, std::move(loopPatterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace mlir