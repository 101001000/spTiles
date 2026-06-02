#pragma once

#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"
#include "common.hpp"

struct AddFOpConversion final
    : mlir::OpConversionPattern<mlir::cuda_tile::AddFOp> {
    using mlir::OpConversionPattern<mlir::cuda_tile::AddFOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        mlir::cuda_tile::AddFOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter &rewriter) const override {
        mlir::Value lhs = adaptor.getOperands()[0];
        mlir::Value rhs = adaptor.getOperands()[1];

        if (lhs.getType() != rhs.getType()) {
            return rewriter.notifyMatchFailure(op, "operand type mismatch");
        }

        if (!mlir::isa<mlir::FloatType>(lhs.getType())) {
            return rewriter.notifyMatchFailure(op, "expected scalar float");
        }

        mlir::Value result = mlir::spirv::FAddOp::create(
            rewriter,
            op.getLoc(),
            lhs.getType(),
            lhs,
            rhs);

        rewriter.replaceOp(op, result);
        return mlir::success();
    }
};


struct LoadViewTkoOpConversion final
    : mlir::OpConversionPattern<mlir::cuda_tile::LoadViewTkoOp> {
    LoadViewTkoOpConversion(
        mlir::MLIRContext *context,
        CudaTileSpmdLoweringContext &lowering_context)
        : mlir::OpConversionPattern<mlir::cuda_tile::LoadViewTkoOp>(context),
          lowering_context(lowering_context) {}

    mlir::LogicalResult matchAndRewrite(
        mlir::cuda_tile::LoadViewTkoOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter &rewriter) const override {
        mlir::Location loc = op.getLoc();

        mlir::Value pview =
            lowering_context.get_forwarded_value(op->getOperand(0));
        mlir::Value token = adaptor.getOperands().back();

        auto partition_it = lowering_context.partition_views.find(pview);
        if (partition_it == lowering_context.partition_views.end()) {
            return rewriter.notifyMatchFailure(op, "unknown partition view");
        }

        mlir::Value base =
            lowering_context.get_forwarded_value(partition_it->second.base);

        auto array_it = lowering_context.entry_arrays.find(base);
        if (array_it == lowering_context.entry_arrays.end()) {
            return rewriter.notifyMatchFailure(
                op, "partition view base is not a known entry array");
        }

        mlir::Value linear_id =
            lowering_context.create_global_linear_id(
                op.getOperation(),
                rewriter);

        mlir::Value active;
        mlir::Value index;
        lowering_context.create_linearized_array_access(
            rewriter,
            loc,
            array_it->second,
            linear_id,
            active,
            index);

        mlir::Value ptr =
            lowering_context.create_storage_buffer_access(
                rewriter,
                loc,
                base,
                index);

        mlir::Value loaded = mlir::spirv::LoadOp::create(
            rewriter,
            loc,
            ptr,
            nullptr,
            nullptr);

        llvm::SmallVector<mlir::Value, 2> replacements;
        replacements.push_back(loaded);
        replacements.push_back(token);

        rewriter.replaceOp(op, replacements);
        return mlir::success();
    }

private:
    CudaTileSpmdLoweringContext &lowering_context;
};

struct StoreViewTkoOpConversion final
    : mlir::OpConversionPattern<mlir::cuda_tile::StoreViewTkoOp> {
    StoreViewTkoOpConversion(
        mlir::MLIRContext *context,
        CudaTileSpmdLoweringContext &lowering_context)
        : mlir::OpConversionPattern<mlir::cuda_tile::StoreViewTkoOp>(context),
          lowering_context(lowering_context) {}

    mlir::LogicalResult matchAndRewrite(
        mlir::cuda_tile::StoreViewTkoOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter &rewriter) const override {
        mlir::Location loc = op.getLoc();

        mlir::Value value = adaptor.getOperands()[0];
        mlir::Value pview =
            lowering_context.get_forwarded_value(op->getOperand(1));
        mlir::Value token = adaptor.getOperands().back();

        auto partition_it = lowering_context.partition_views.find(pview);
        if (partition_it == lowering_context.partition_views.end()) {
            return rewriter.notifyMatchFailure(op, "unknown partition view");
        }

        mlir::Value base =
            lowering_context.get_forwarded_value(partition_it->second.base);

        auto array_it = lowering_context.entry_arrays.find(base);
        if (array_it == lowering_context.entry_arrays.end()) {
            return rewriter.notifyMatchFailure(
                op, "partition view base is not a known entry array");
        }

        if (!mlir::isa<mlir::spirv::PointerType>(base.getType())) {
            return rewriter.notifyMatchFailure(
                op, "expected SPIR-V pointer base");
        }

        if (!mlir::isa<mlir::FloatType>(value.getType())) {
            return rewriter.notifyMatchFailure(
                op, "expected scalar floating-point value");
        }

        mlir::Value linear_id =
            lowering_context.create_global_linear_id(
                op.getOperation(),
                rewriter);

        mlir::Value active;
        mlir::Value index;
        lowering_context.create_linearized_array_access(
            rewriter,
            loc,
            array_it->second,
            linear_id,
            active,
            index);

        auto selection_op = mlir::spirv::SelectionOp::create(
            rewriter,
            loc,
            mlir::spirv::SelectionControl::None);

        mlir::Region &region = selection_op.getBody();

        auto *header_block = new mlir::Block();
        auto *then_block = new mlir::Block();
        auto *merge_block = new mlir::Block();

        region.push_back(header_block);
        region.push_back(then_block);
        region.push_back(merge_block);

        rewriter.setInsertionPointToStart(header_block);

        mlir::spirv::BranchConditionalOp::create(
            rewriter,
            loc,
            active,
            then_block,
            mlir::ValueRange{},
            merge_block,
            mlir::ValueRange{});

        rewriter.setInsertionPointToStart(then_block);

        mlir::Value ptr =
            lowering_context.create_storage_buffer_access(
                rewriter,
                loc,
                base,
                index);

        mlir::spirv::StoreOp::create(
            rewriter,
            loc,
            ptr,
            value,
            nullptr,
            nullptr);

        mlir::spirv::BranchOp::create(
            rewriter,
            loc,
            merge_block);

        rewriter.setInsertionPointToStart(merge_block);
        mlir::spirv::MergeOp::create(rewriter, loc);

        rewriter.setInsertionPointAfter(selection_op);

        llvm::SmallVector<mlir::Value, 1> replacements;
        replacements.push_back(token);

        rewriter.replaceOp(op, replacements);
        return mlir::success();
    }

private:
    CudaTileSpmdLoweringContext &lowering_context;
};

struct ConstantOpConversion final
    : mlir::OpConversionPattern<mlir::cuda_tile::ConstantOp> {
    using mlir::OpConversionPattern<
        mlir::cuda_tile::ConstantOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        mlir::cuda_tile::ConstantOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter &rewriter) const override {
        if (op->getNumResults() != 1) {
            return rewriter.notifyMatchFailure(
                op, "expected one result");
        }

        mlir::Attribute value_attr = op->getAttr("value");
        if (!value_attr) {
            return rewriter.notifyMatchFailure(
                op, "expected value attribute");
        }

        auto dense_attr = mlir::dyn_cast<mlir::DenseElementsAttr>(value_attr);
        if (!dense_attr || !dense_attr.isSplat()) {
            return rewriter.notifyMatchFailure(
                op, "expected splat dense elements attribute");
        }

        mlir::Location loc = op.getLoc();
        mlir::Type element_type = dense_attr.getElementType();

        if (auto integer_type = mlir::dyn_cast<mlir::IntegerType>(element_type)) {
            if (integer_type.getWidth() != 32) {
                return rewriter.notifyMatchFailure(
                    op, "expected i32 constant");
            }

            llvm::APInt value = dense_attr.getSplatValue<llvm::APInt>();

            mlir::Value constant = mlir::spirv::ConstantOp::create(
                rewriter,
                loc,
                element_type,
                rewriter.getIntegerAttr(element_type, value));

            rewriter.replaceOp(op, constant);
            return mlir::success();
        }

        auto float_type = mlir::dyn_cast<mlir::FloatType>(element_type);
        if (float_type && float_type.getWidth() == 32) {
            llvm::APFloat value = dense_attr.getSplatValue<llvm::APFloat>();

            mlir::Value constant = mlir::spirv::ConstantOp::create(
                rewriter,
                loc,
                element_type,
                mlir::FloatAttr::get(element_type, value));

            rewriter.replaceOp(op, constant);
            return mlir::success();
        }

        return rewriter.notifyMatchFailure(
            op, "unsupported constant element type");
    }
};

struct GetTileBlockIdOpConversion final
    : mlir::OpConversionPattern<mlir::cuda_tile::GetTileBlockIdOp> {
    using mlir::OpConversionPattern<
        mlir::cuda_tile::GetTileBlockIdOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        mlir::cuda_tile::GetTileBlockIdOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter &rewriter) const override {
        if (op->getNumResults() != 3) {
            return rewriter.notifyMatchFailure(
                op, "expected three results");
        }

        mlir::Location loc = op.getLoc();
        mlir::Type i32_type = rewriter.getI32Type();

        mlir::Value workgroup_id =
            mlir::spirv::getBuiltinVariableValue(
                op.getOperation(),
                mlir::spirv::BuiltIn::WorkgroupId,
                i32_type,
                rewriter);

        if (!workgroup_id) {
            return rewriter.notifyMatchFailure(
                op, "failed to get WorkgroupId builtin");
        }

        mlir::Value block_id_x =
            mlir::spirv::CompositeExtractOp::create(
                rewriter,
                loc,
                i32_type,
                workgroup_id,
                rewriter.getI32ArrayAttr({0}));

        mlir::Value block_id_y =
            mlir::spirv::CompositeExtractOp::create(
                rewriter,
                loc,
                i32_type,
                workgroup_id,
                rewriter.getI32ArrayAttr({1}));

        mlir::Value block_id_z =
            mlir::spirv::CompositeExtractOp::create(
                rewriter,
                loc,
                i32_type,
                workgroup_id,
                rewriter.getI32ArrayAttr({2}));

        llvm::SmallVector<mlir::Value, 3> replacements;
        replacements.push_back(block_id_x);
        replacements.push_back(block_id_y);
        replacements.push_back(block_id_z);

        rewriter.replaceOp(op, replacements);
        return mlir::success();
    }
};

struct ReshapeOpConversion final
    : mlir::OpConversionPattern<mlir::cuda_tile::ReshapeOp> {
    using mlir::OpConversionPattern<
        mlir::cuda_tile::ReshapeOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        mlir::cuda_tile::ReshapeOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter &rewriter) const override {
        if (adaptor.getOperands().size() != 1 || op->getNumResults() != 1) {
            return rewriter.notifyMatchFailure(
                op, "expected one operand and one result");
        }

        rewriter.replaceOp(op, adaptor.getOperands()[0]);
        return mlir::success();
    }
};

struct ForOpConversion final
    : mlir::OpConversionPattern<mlir::cuda_tile::ForOp> {
    ForOpConversion(
        mlir::MLIRContext *context,
        CudaTileSpmdLoweringContext &lowering_context)
        : mlir::OpConversionPattern<mlir::cuda_tile::ForOp>(context),
          lowering_context(lowering_context) {}

    mlir::LogicalResult matchAndRewrite(
        mlir::cuda_tile::ForOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter &rewriter) const override {
        if (op->getNumRegions() != 1 || op->getRegion(0).empty()) {
            return rewriter.notifyMatchFailure(
                op, "expected one non-empty region");
        }

        mlir::Region &old_region = op->getRegion(0);
        if (!old_region.hasOneBlock()) {
            return rewriter.notifyMatchFailure(
                op, "expected one body block");
        }

        unsigned result_count = op->getNumResults();
        if (op->getNumOperands() != result_count + 3) {
            return rewriter.notifyMatchFailure(
                op, "invalid operand count");
        }

        mlir::Block &old_body = old_region.front();
        if (old_body.getNumArguments() != result_count + 1) {
            return rewriter.notifyMatchFailure(
                op, "invalid body argument count");
        }

        mlir::Operation *continue_op = old_body.getTerminator();
        if (!continue_op ||
            continue_op->getName().getStringRef() != "cuda_tile.continue") {
            return rewriter.notifyMatchFailure(
                op, "expected cuda_tile.continue terminator");
        }

        if (continue_op->getNumOperands() != result_count) {
            return rewriter.notifyMatchFailure(
                op, "continue operand count mismatch");
        }

        mlir::ValueRange operands = adaptor.getOperands();
        mlir::Value lower_bound = operands[0];
        mlir::Value upper_bound = operands[1];
        mlir::Value step = operands[2];

        auto is_i32 = [](mlir::Value value) {
            auto type = mlir::dyn_cast<mlir::IntegerType>(value.getType());
            return type && type.getWidth() == 32;
        };

        if (!is_i32(lower_bound) || !is_i32(upper_bound) || !is_i32(step)) {
            return rewriter.notifyMatchFailure(
                op, "expected scalar i32 bounds");
        }

        llvm::SmallVector<mlir::Value> init_values;
        llvm::SmallVector<mlir::Type> result_types;
        init_values.reserve(result_count);
        result_types.reserve(result_count);

        for (unsigned i = 0; i < result_count; ++i) {
            mlir::Value init = operands[i + 3];
            init_values.push_back(init);
            result_types.push_back(init.getType());
        }

        mlir::Location loc = op.getLoc();
        mlir::Type i32_type = rewriter.getI32Type();

        mlir::OperationState state(
            loc,
            mlir::spirv::LoopOp::getOperationName());

        state.addTypes(result_types);
        state.addAttribute(
            "loop_control",
            mlir::spirv::LoopControlAttr::get(
                rewriter.getContext(),
                mlir::spirv::LoopControl::None));
        state.addRegion();

        auto loop_op =
            mlir::cast<mlir::spirv::LoopOp>(rewriter.create(state));

        mlir::Region &region = loop_op.getBody();

        auto *entry_block = new mlir::Block();
        auto *header_block = new mlir::Block();
        auto *body_block = new mlir::Block();
        auto *continue_block = new mlir::Block();
        auto *merge_block = new mlir::Block();

        region.push_back(entry_block);
        region.push_back(header_block);
        region.push_back(body_block);
        region.push_back(continue_block);
        region.push_back(merge_block);

        header_block->addArgument(i32_type, loc);
        body_block->addArgument(i32_type, loc);
        continue_block->addArgument(i32_type, loc);

        for (mlir::Type type : result_types) {
            header_block->addArgument(type, loc);
            body_block->addArgument(type, loc);
            continue_block->addArgument(type, loc);
            merge_block->addArgument(type, loc);
        }

        llvm::SmallVector<mlir::Value> entry_args;
        entry_args.push_back(lower_bound);
        entry_args.append(init_values.begin(), init_values.end());

        rewriter.setInsertionPointToStart(entry_block);
        mlir::spirv::BranchOp::create(
            rewriter,
            loc,
            header_block,
            entry_args);

        llvm::SmallVector<mlir::Value> header_args;
        for (mlir::BlockArgument arg : header_block->getArguments()) {
            header_args.push_back(arg);
        }

        llvm::SmallVector<mlir::Value> merge_args;
        for (unsigned i = 0; i < result_count; ++i) {
            merge_args.push_back(header_block->getArgument(i + 1));
        }

        rewriter.setInsertionPointToStart(header_block);

        mlir::Value condition = mlir::spirv::ULessThanOp::create(
            rewriter,
            loc,
            i32_type,
            header_block->getArgument(0),
            upper_bound);

        mlir::spirv::BranchConditionalOp::create(
            rewriter,
            loc,
            condition,
            body_block,
            header_args,
            merge_block,
            merge_args);

        for (unsigned i = 0; i < old_body.getNumArguments(); ++i) {
            old_body.getArgument(i).replaceAllUsesWith(
                body_block->getArgument(i));
        }

        while (!old_body.empty()) {
            old_body.front().moveBefore(body_block, body_block->end());
        }

        lowering_context.continue_targets[continue_op] =
            ContinueTargetInfo{continue_block, step};

        llvm::SmallVector<mlir::Value> backedge_args;
        for (mlir::BlockArgument arg : continue_block->getArguments()) {
            backedge_args.push_back(arg);
        }

        rewriter.setInsertionPointToStart(continue_block);
        mlir::spirv::BranchOp::create(
            rewriter,
            loc,
            header_block,
            backedge_args);

        llvm::SmallVector<mlir::Value> final_values;
        for (mlir::BlockArgument arg : merge_block->getArguments()) {
            final_values.push_back(arg);
        }

        rewriter.setInsertionPointToStart(merge_block);
        mlir::spirv::MergeOp::create(
            rewriter,
            loc,
            final_values);

        rewriter.replaceOp(op, loop_op->getResults());
        return mlir::success();
    }

private:
    CudaTileSpmdLoweringContext &lowering_context;
};

struct ContinueOpConversion final
    : mlir::OpConversionPattern<mlir::cuda_tile::ContinueOp> {
    ContinueOpConversion(
        mlir::MLIRContext *context,
        CudaTileSpmdLoweringContext &lowering_context)
        : mlir::OpConversionPattern<mlir::cuda_tile::ContinueOp>(context),
          lowering_context(lowering_context) {}

    mlir::LogicalResult matchAndRewrite(
        mlir::cuda_tile::ContinueOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter &rewriter) const override {
        auto it = lowering_context.continue_targets.find(op.getOperation());
        if (it == lowering_context.continue_targets.end()) {
            return rewriter.notifyMatchFailure(
                op, "missing continue target");
        }

        mlir::Block *continue_block = it->second.continue_block;
        mlir::Value step = it->second.step;

        mlir::Block *body_block = op->getBlock();
        if (!body_block || body_block->getNumArguments() == 0) {
            return rewriter.notifyMatchFailure(
                op, "invalid loop body block");
        }

        mlir::Location loc = op.getLoc();
        mlir::Type i32_type = rewriter.getI32Type();

        mlir::Value next_iv = mlir::spirv::IAddOp::create(
            rewriter,
            loc,
            i32_type,
            body_block->getArgument(0),
            step);

        llvm::SmallVector<mlir::Value> continue_args;
        continue_args.push_back(next_iv);

        for (mlir::Value value : adaptor.getOperands()) {
            continue_args.push_back(value);
        }

        rewriter.replaceOpWithNewOp<mlir::spirv::BranchOp>(
            op,
            continue_block,
            continue_args);

        return mlir::success();
    }

private:
    CudaTileSpmdLoweringContext &lowering_context;
};

struct GetIndexSpaceShapeOpConversion final
    : mlir::OpConversionPattern<mlir::cuda_tile::GetIndexSpaceShapeOp> {
    GetIndexSpaceShapeOpConversion(
        mlir::MLIRContext *context,
        CudaTileSpmdLoweringContext &lowering_context)
        : mlir::OpConversionPattern<
              mlir::cuda_tile::GetIndexSpaceShapeOp>(context),
          lowering_context(lowering_context) {}

    mlir::LogicalResult matchAndRewrite(
        mlir::cuda_tile::GetIndexSpaceShapeOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter &rewriter) const override {
        if (op->getNumOperands() != 1) {
            return rewriter.notifyMatchFailure(
                op, "expected one operand");
        }

        mlir::Value pview =
            lowering_context.get_forwarded_value(op->getOperand(0));

        auto partition_it = lowering_context.partition_views.find(pview);
        if (partition_it == lowering_context.partition_views.end()) {
            return rewriter.notifyMatchFailure(
                op, "unknown partition view");
        }

        auto tensor_it = lowering_context.tensor_views.find(
            partition_it->second.tensor_view);
        if (tensor_it == lowering_context.tensor_views.end()) {
            return rewriter.notifyMatchFailure(
                op, "partition view tensor view is not known");
        }

        llvm::ArrayRef<int> tile_shape =
            partition_it->second.type.getTileShape().asArrayRef();

        if (op->getNumResults() != tile_shape.size()) {
            return rewriter.notifyMatchFailure(
                op, "result count does not match tile rank");
        }

        if (tensor_it->second.shape.size() != tile_shape.size()) {
            return rewriter.notifyMatchFailure(
                op, "tensor view shape rank does not match tile rank");
        }

        mlir::Location loc = op.getLoc();
        mlir::Type i32_type = rewriter.getI32Type();
        llvm::SmallVector<mlir::Value> results;
        results.reserve(tile_shape.size());

        for (uint32_t dim = 0; dim < tile_shape.size(); ++dim) {
            mlir::Value shape_dim =
                lowering_context.get_forwarded_value(
                    tensor_it->second.shape[dim]);

            auto integer_type =
                mlir::dyn_cast<mlir::IntegerType>(shape_dim.getType());
            if (!integer_type || integer_type.getWidth() != 32) {
                return rewriter.notifyMatchFailure(
                    op, "expected i32 tensor view shape dimension");
            }

            int32_t tile_dim = static_cast<int32_t>(tile_shape[dim]);

            mlir::Value c_tile_dim = mlir::spirv::ConstantOp::create(
                rewriter,
                loc,
                i32_type,
                rewriter.getI32IntegerAttr(tile_dim));

            mlir::Value c_tile_dim_minus_one =
                mlir::spirv::ConstantOp::create(
                    rewriter,
                    loc,
                    i32_type,
                    rewriter.getI32IntegerAttr(tile_dim - 1));

            mlir::Value numerator = mlir::spirv::IAddOp::create(
                rewriter,
                loc,
                i32_type,
                shape_dim,
                c_tile_dim_minus_one);

            mlir::Value result = mlir::spirv::UDivOp::create(
                rewriter,
                loc,
                i32_type,
                numerator,
                c_tile_dim);

            results.push_back(result);
        }

        rewriter.replaceOp(op, results);
        return mlir::success();
    }

private:
    CudaTileSpmdLoweringContext &lowering_context;
};


template <typename SourceOp, typename SpirvOp>
struct ScalarIntegerBinaryOpConversion final
    : mlir::OpConversionPattern<SourceOp> {
    using mlir::OpConversionPattern<SourceOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        SourceOp op,
        typename mlir::OpConversionPattern<SourceOp>::OpAdaptor adaptor,
        mlir::ConversionPatternRewriter &rewriter) const override {
        mlir::Value lhs = adaptor.getOperands()[0];
        mlir::Value rhs = adaptor.getOperands()[1];

        if (lhs.getType() != rhs.getType()) {
            return rewriter.notifyMatchFailure(op, "operand type mismatch");
        }

        if (!mlir::isa<mlir::IntegerType>(lhs.getType())) {
            return rewriter.notifyMatchFailure(op, "expected integer operands");
        }

        mlir::Value result = SpirvOp::create(
            rewriter,
            op.getLoc(),
            lhs.getType(),
            lhs,
            rhs);

        rewriter.replaceOp(op, result);
        return mlir::success();
    }
};

using AddIOpConversion =
    ScalarIntegerBinaryOpConversion<
        mlir::cuda_tile::AddIOp,
        mlir::spirv::IAddOp>;

using SubIOpConversion =
    ScalarIntegerBinaryOpConversion<
        mlir::cuda_tile::SubIOp,
        mlir::spirv::ISubOp>;

using MulIOpConversion =
    ScalarIntegerBinaryOpConversion<
        mlir::cuda_tile::MulIOp,
        mlir::spirv::IMulOp>;

using DivIOpConversion =
    ScalarIntegerBinaryOpConversion<
        mlir::cuda_tile::DivIOp,
        mlir::spirv::UDivOp>;

using RemIOpConversion =
    ScalarIntegerBinaryOpConversion<
        mlir::cuda_tile::RemIOp,
        mlir::spirv::UModOp>;

using AndIOpConversion =
    ScalarIntegerBinaryOpConversion<
        mlir::cuda_tile::AndIOp,
        mlir::spirv::LogicalAndOp>;

using XOrIOpConversion =
    ScalarIntegerBinaryOpConversion<
        mlir::cuda_tile::XOrIOp,
        mlir::spirv::LogicalNotEqualOp>;


struct AssumeOpConversion final
    : mlir::OpConversionPattern<mlir::cuda_tile::AssumeOp> {
    using mlir::OpConversionPattern<
        mlir::cuda_tile::AssumeOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        mlir::cuda_tile::AssumeOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter &rewriter) const override {
        if (adaptor.getOperands().size() != 1 || op->getNumResults() != 1) {
            return rewriter.notifyMatchFailure(
                op, "expected one operand and one result");
        }

        rewriter.replaceOp(op, adaptor.getOperands()[0]);
        return mlir::success();
    }
};

struct MakeTokenOpConversion final
    : mlir::OpConversionPattern<mlir::cuda_tile::MakeTokenOp> {
    using mlir::OpConversionPattern<
        mlir::cuda_tile::MakeTokenOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        mlir::cuda_tile::MakeTokenOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter &rewriter) const override {
        mlir::Value token = mlir::spirv::ConstantOp::create(
            rewriter,
            op.getLoc(),
            rewriter.getI32Type(),
            rewriter.getI32IntegerAttr(0));

        rewriter.replaceOp(op, token);
        return mlir::success();
    }
};

struct MakeTensorViewConversion final
    : mlir::OpConversionPattern<mlir::cuda_tile::MakeTensorViewOp> {
    using mlir::OpConversionPattern<
        mlir::cuda_tile::MakeTensorViewOp>::OpConversionPattern;
    
    mlir::LogicalResult matchAndRewrite(
        mlir::cuda_tile::MakeTensorViewOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter &rewriter) const override {
        mlir::Value token = mlir::spirv::ConstantOp::create(
            rewriter,
            op.getLoc(),
            rewriter.getI32Type(),
            rewriter.getI32IntegerAttr(0));

        rewriter.replaceOp(op, token);
        return mlir::success();
    }
};

struct MakePartitionViewConversion final
    : mlir::OpConversionPattern<mlir::cuda_tile::MakePartitionViewOp> {
    using mlir::OpConversionPattern<
        mlir::cuda_tile::MakePartitionViewOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        mlir::cuda_tile::MakePartitionViewOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter &rewriter) const override {
        mlir::Value token = mlir::spirv::ConstantOp::create(
            rewriter,
            op.getLoc(),
            rewriter.getI32Type(),
            rewriter.getI32IntegerAttr(0));

        rewriter.replaceOp(op, token);
        return mlir::success();
    }
};

struct CmpIOpConversion final
    : mlir::OpConversionPattern<mlir::cuda_tile::CmpIOp> {
    using mlir::OpConversionPattern<
        mlir::cuda_tile::CmpIOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        mlir::cuda_tile::CmpIOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter &rewriter) const override {
        mlir::Value lhs = adaptor.getOperands()[0];
        mlir::Value rhs = adaptor.getOperands()[1];

        if (lhs.getType() != rhs.getType()) {
            return rewriter.notifyMatchFailure(op, "operand type mismatch");
        }

        mlir::Type bool_type = rewriter.getI1Type();

        if (attr_contains(op.getOperation(), "comparison_predicate", "less_than")) {
            mlir::Value result = mlir::spirv::ULessThanOp::create(
                rewriter,
                op.getLoc(),
                bool_type,
                lhs,
                rhs);

            rewriter.replaceOp(op, result);
            return mlir::success();
        }

        if (attr_contains(op.getOperation(), "comparison_predicate", "not_equal")) {
            mlir::Value result = mlir::spirv::INotEqualOp::create(
                rewriter,
                op.getLoc(),
                bool_type,
                lhs,
                rhs);

            rewriter.replaceOp(op, result);
            return mlir::success();
        }

        return rewriter.notifyMatchFailure(
            op, "unsupported comparison predicate");
    }

private:
    static bool attr_contains(
        mlir::Operation *op,
        llvm::StringRef name,
        llvm::StringRef text) {
        mlir::Attribute attr = op->getAttr(name);
        if (!attr) {
            return false;
        }

        std::string value;
        llvm::raw_string_ostream stream(value);
        attr.print(stream);
        stream.flush();

        return llvm::StringRef(value).contains(text);
    }
};

struct SelectOpConversion final
    : mlir::OpConversionPattern<mlir::cuda_tile::SelectOp> {
    using mlir::OpConversionPattern<
        mlir::cuda_tile::SelectOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        mlir::cuda_tile::SelectOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter &rewriter) const override {
        mlir::Value condition = adaptor.getOperands()[0];
        mlir::Value true_value = adaptor.getOperands()[1];
        mlir::Value false_value = adaptor.getOperands()[2];

        if (true_value.getType() != false_value.getType()) {
            return rewriter.notifyMatchFailure(op, "value type mismatch");
        }

        mlir::Value result = mlir::spirv::SelectOp::create(
            rewriter,
            op.getLoc(),
            true_value.getType(),
            condition,
            true_value,
            false_value);

        rewriter.replaceOp(op, result);
        return mlir::success();
    }
};

struct MinIOpConversion final
    : mlir::OpConversionPattern<mlir::cuda_tile::MinIOp> {
    using mlir::OpConversionPattern<
        mlir::cuda_tile::MinIOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        mlir::cuda_tile::MinIOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter &rewriter) const override {
        mlir::Value lhs = adaptor.getOperands()[0];
        mlir::Value rhs = adaptor.getOperands()[1];

        if (lhs.getType() != rhs.getType()) {
            return rewriter.notifyMatchFailure(op, "operand type mismatch");
        }

        auto integer_type = mlir::dyn_cast<mlir::IntegerType>(lhs.getType());
        if (!integer_type) {
            return rewriter.notifyMatchFailure(op, "expected integer operands");
        }

        mlir::Value condition = mlir::spirv::ULessThanOp::create(
            rewriter,
            op.getLoc(),
            rewriter.getI1Type(),
            lhs,
            rhs);

        mlir::Value result = mlir::spirv::SelectOp::create(
            rewriter,
            op.getLoc(),
            lhs.getType(),
            condition,
            lhs,
            rhs);

        rewriter.replaceOp(op, result);
        return mlir::success();
    }
};

struct ReturnOpConversion final
    : mlir::OpConversionPattern<mlir::cuda_tile::ReturnOp> {
    using mlir::OpConversionPattern<
        mlir::cuda_tile::ReturnOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        mlir::cuda_tile::ReturnOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter &rewriter) const override {
        if (!adaptor.getOperands().empty()) {
            return rewriter.notifyMatchFailure(
                op, "expected return without operands");
        }

        rewriter.replaceOpWithNewOp<mlir::spirv::ReturnOp>(op);
        return mlir::success();
    }
};