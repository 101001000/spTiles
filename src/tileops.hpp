#pragma once

#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "common.hpp"
#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"

#include <iostream>

static bool is_cuda_tile_op(mlir::Operation *op) { return op->getName().getDialectNamespace() == "cuda_tile"; }

static mlir::Value get_remapped_index_or_self(mlir::Value value, mlir::ConversionPatternRewriter &rewriter) {
	while (true) {
		if (mlir::Value remapped_value = rewriter.getRemappedValue(value)) {
			if (remapped_value != value)
				return remapped_value;
		}

		auto assume_op = value.getDefiningOp<mlir::cuda_tile::AssumeOp>();
		if (!assume_op)
			return value;

		value = assume_op->getOperand(0);
	}
}

static void erase_dead_cuda_tile_ops(mlir::ModuleOp root_module) {
	bool changed = true;

	while (changed) {
		changed = false;

		llvm::SmallVector<mlir::Operation *> ops;

		root_module.walk([&](mlir::Operation *op) {
			if (!is_cuda_tile_op(op)) {
				return;
			}

			if (op->hasTrait<mlir::OpTrait::IsTerminator>()) {
				return;
			}

			if (!op->use_empty()) {
				llvm::errs() << "live: " << op->getName().getStringRef() << "\n";

				for (mlir::Value result : op->getResults()) {
					for (mlir::OpOperand &use : result.getUses()) {
						llvm::errs() << "  used by: " << use.getOwner()->getName().getStringRef() << " operand " << use.getOperandNumber() << "\n";
					}
				}

				return;
			}

			ops.push_back(op);
		});

		for (mlir::Operation *op : llvm::reverse(ops)) {
			llvm::errs() << "erase dead: " << op->getName().getStringRef() << "\n";
			op->erase();
			changed = true;
		}
	}
}

struct TileMaterializer {

	LoweringContext &lowering_context;

	TileMaterializer(LoweringContext &lowering_context) : lowering_context(lowering_context) {}

	mlir::Value materialize_tile(mlir::Value value, mlir::Location loc, mlir::ConversionPatternRewriter &rewriter) {
		if (mlir::Value remapped_value = rewriter.getRemappedValue(value)) {
			if (remapped_value != value) {
				return remapped_value;
			}
		}

		mlir::Operation *op = value.getDefiningOp();
		assert(op && "block arguments should have been remapped");

		if (auto load_op = mlir::dyn_cast<mlir::cuda_tile::LoadViewTkoOp>(op))
			return materialize_load_view_tko(load_op, loc, rewriter);

		if (auto addf_op = mlir::dyn_cast<mlir::cuda_tile::AddFOp>(op))
			return materialize_fadd(addf_op, loc, rewriter);

		if (auto reshape_op = mlir::dyn_cast<mlir::cuda_tile::ReshapeOp>(op))
			return materialize_tile(reshape_op.getSource(), loc, rewriter);

		llvm_unreachable("Unsupported operation to materialize");
	}

	mlir::Value materialize_partition_view_element_ptr(mlir::Value partition_view, llvm::SmallVector<mlir::Value> indices, mlir::Value element_index, mlir::Location loc, mlir::ConversionPatternRewriter &rewriter) {
		PartitionViewInfo &pview_info = lowering_context.partition_views[partition_view];
		TensorViewInfo &tview_info = lowering_context.tensor_views[pview_info.tensor_view];

		mlir::Type i32_type = rewriter.getI32Type();

		llvm::SmallDenseMap<mlir::Operation *, llvm::SmallVector<mlir::Value, 3>, 4> materialized_tile_block_ids;

		for (mlir::Value &index : indices) {
			if (mlir::Value remapped_index = rewriter.getRemappedValue(index)) {
				if (remapped_index != index) {
					index = remapped_index;
					continue;
				}
			}

			auto get_tile_block_id_op = index.getDefiningOp<mlir::cuda_tile::GetTileBlockIdOp>();
			if (!get_tile_block_id_op)
				continue;

			auto op_result = llvm::cast<mlir::OpResult>(index);
			unsigned result_index = op_result.getResultNumber();

			auto it = materialized_tile_block_ids.find(get_tile_block_id_op.getOperation());
			if (it == materialized_tile_block_ids.end()) {
				llvm::SmallVector<mlir::Value, 3> tile_block_ids = materialize_get_tile_block_id(get_tile_block_id_op, partition_view, loc, rewriter);

				it = materialized_tile_block_ids.try_emplace(get_tile_block_id_op.getOperation(), std::move(tile_block_ids)).first;
			}

			index = it->second[result_index];
		}

		mlir::Value tile_offset = indices[0];

		mlir::Value stride;
		for (unsigned i = 1; i < indices.size(); ++i) {
			IndexValue &extent = tview_info.shape[i - 1];

			mlir::Value extent_value;
			if (extent.is_static()) {
				extent_value = mlir::spirv::ConstantOp::create(rewriter, loc, i32_type, rewriter.getIntegerAttr(i32_type, extent.static_value));
			} else {
				extent_value = get_remapped_index_or_self(extent.dynamic_value, rewriter);
			}

			if (stride)
				stride = mlir::spirv::IMulOp::create(rewriter, loc, i32_type, stride, extent_value);
			else
				stride = extent_value;

			mlir::Value term = mlir::spirv::IMulOp::create(rewriter, loc, i32_type, indices[i], stride);
			tile_offset = mlir::spirv::IAddOp::create(rewriter, loc, i32_type, tile_offset, term);
		}

		int32_t total_tile_size = 1;
		for (int32_t value : pview_info.tile_shape)
			total_tile_size *= value;

		mlir::Value total_tile_size_value = mlir::spirv::ConstantOp::create(rewriter, loc, i32_type, rewriter.getIntegerAttr(i32_type, total_tile_size));

		tile_offset = mlir::spirv::IMulOp::create(rewriter, loc, i32_type, tile_offset, total_tile_size_value);
		tile_offset = mlir::spirv::IAddOp::create(rewriter, loc, i32_type, tile_offset, element_index);

		mlir::Value base_ptr = rewriter.getRemappedValue(tview_info.base_ptr);
		if (!base_ptr)
			base_ptr = tview_info.base_ptr;

		mlir::Value member_index = mlir::spirv::ConstantOp::create(rewriter, loc, i32_type, rewriter.getIntegerAttr(i32_type, 0));

		return mlir::spirv::AccessChainOp::create(rewriter, loc, base_ptr, mlir::ValueRange{member_index, tile_offset});
	}

	mlir::Value materialize_global_linear_index(mlir::Operation *op, mlir::Location loc, mlir::ConversionPatternRewriter &rewriter) {
		mlir::Type i32_type = rewriter.getI32Type();

		mlir::Value global_id = mlir::spirv::getBuiltinVariableValue(op, mlir::spirv::BuiltIn::GlobalInvocationId, i32_type, rewriter);

		mlir::Value num_workgroups = mlir::spirv::getBuiltinVariableValue(op, mlir::spirv::BuiltIn::NumWorkgroups, i32_type, rewriter);

		mlir::Value workgroup_size = mlir::spirv::getBuiltinVariableValue(op, mlir::spirv::BuiltIn::WorkgroupSize, i32_type, rewriter);

		auto extract = [&](mlir::Value value, int32_t index) -> mlir::Value { return mlir::spirv::CompositeExtractOp::create(rewriter, loc, value, llvm::ArrayRef<int32_t>{index}); };

		mlir::Value x = extract(global_id, 0);
		mlir::Value y = extract(global_id, 1);
		mlir::Value z = extract(global_id, 2);

		mlir::Value num_workgroups_x = extract(num_workgroups, 0);
		mlir::Value num_workgroups_y = extract(num_workgroups, 1);

		mlir::Value workgroup_size_x = extract(workgroup_size, 0);
		mlir::Value workgroup_size_y = extract(workgroup_size, 1);

		mlir::Value global_size_x = mlir::spirv::IMulOp::create(rewriter, loc, num_workgroups_x, workgroup_size_x);

		mlir::Value global_size_y = mlir::spirv::IMulOp::create(rewriter, loc, num_workgroups_y, workgroup_size_y);

		mlir::Value y_offset = mlir::spirv::IMulOp::create(rewriter, loc, y, global_size_x);

		mlir::Value xy = mlir::spirv::IAddOp::create(rewriter, loc, x, y_offset);

		mlir::Value xy_size = mlir::spirv::IMulOp::create(rewriter, loc, global_size_x, global_size_y);

		mlir::Value z_offset = mlir::spirv::IMulOp::create(rewriter, loc, z, xy_size);

		return mlir::spirv::IAddOp::create(rewriter, loc, xy, z_offset);
	}

	mlir::Value materialize_partition_local_linear_index(mlir::Operation *op, mlir::Value partition_view, mlir::Location loc, mlir::ConversionPatternRewriter &rewriter) {
		PartitionViewInfo &pview_info = lowering_context.partition_views[partition_view];

		mlir::Type i32_type = rewriter.getI32Type();

		int32_t total_tile_size = 1;
		for (int32_t value : pview_info.tile_shape)
			total_tile_size *= value;

		mlir::Value global_linear_id = materialize_global_linear_index(op, loc, rewriter);

		mlir::Value total_tile_size_value = mlir::spirv::ConstantOp::create(rewriter, loc, i32_type, rewriter.getIntegerAttr(i32_type, total_tile_size));

		return mlir::spirv::UModOp::create(rewriter, loc, global_linear_id, total_tile_size_value);
	}

	mlir::Value materialize_load_view_tko(mlir::cuda_tile::LoadViewTkoOp op, mlir::Location loc, mlir::ConversionPatternRewriter &rewriter) {
		llvm::SmallVector<mlir::Value> indices(op.getIndex().begin(), op.getIndex().end());
		mlir::Value partition = op.getView();
		mlir::Value partition_local_index = materialize_partition_local_linear_index(op, partition, loc, rewriter);
		mlir::Value element_ptr = materialize_partition_view_element_ptr(op.getView(), indices, partition_local_index, loc, rewriter);
		return mlir::spirv::LoadOp::create(rewriter, loc, element_ptr);
	}

	mlir::Value materialize_fadd(mlir::cuda_tile::AddFOp op, mlir::Location loc, mlir::ConversionPatternRewriter &rewriter) {
		mlir::Value lhs = materialize_tile(op.getLhs(), loc, rewriter);
		mlir::Value rhs = materialize_tile(op.getRhs(), loc, rewriter);
		return mlir::spirv::FAddOp::create(rewriter, loc, lhs.getType(), lhs, rhs);
	}

	llvm::SmallVector<mlir::Value, 3> materialize_get_tile_block_id(mlir::cuda_tile::GetTileBlockIdOp op, mlir::Value partition_view, mlir::Location loc, mlir::ConversionPatternRewriter &rewriter) {
		PartitionViewInfo &pview_info = lowering_context.partition_views[partition_view];
		TensorViewInfo &tview_info = lowering_context.tensor_views[pview_info.tensor_view];

		mlir::Type i32_type = rewriter.getI32Type();

		auto make_i32_constant = [&](int64_t value) -> mlir::Value { return mlir::spirv::ConstantOp::create(rewriter, loc, i32_type, rewriter.getIntegerAttr(i32_type, value)); };

		auto ceil_div_static = [](int64_t lhs, int64_t rhs) -> int64_t { return (lhs + rhs - 1) / rhs; };

		auto materialize_ceil_div = [&](IndexValue lhs, int64_t rhs) -> mlir::Value {
			if (lhs.is_static()) {
				return make_i32_constant(ceil_div_static(lhs.static_value, rhs));
			}

			mlir::Value lhs_value = get_remapped_index_or_self(lhs.dynamic_value, rewriter);

			if (rhs == 1) {
				return lhs_value;
			}

			mlir::Value rhs_value = make_i32_constant(rhs);
			mlir::Value offset = make_i32_constant(rhs - 1);
			mlir::Value numerator = mlir::spirv::IAddOp::create(rewriter, loc, lhs_value, offset);

			return mlir::spirv::UDivOp::create(rewriter, loc, numerator, rhs_value);
		};

		auto materialize_num_tile_blocks = [&](int64_t dim) -> mlir::Value {
			IndexValue tensor_dim = dim < tview_info.shape.size() ? tview_info.shape[dim] : IndexValue::from_static(1);
			int64_t tile_dim = dim < pview_info.tile_shape.size() ? pview_info.tile_shape[dim] : 1;

			return materialize_ceil_div(tensor_dim, tile_dim);
		};

		int32_t total_tile_size = 1;
		for (int32_t value : pview_info.tile_shape)
			total_tile_size *= value;

		mlir::Value global_linear_id = materialize_global_linear_index(op.getOperation(), loc, rewriter);
		mlir::Value total_tile_size_value = make_i32_constant(total_tile_size);
		mlir::Value tile_linear_id = mlir::spirv::UDivOp::create(rewriter, loc, global_linear_id, total_tile_size_value);

		mlir::Value num_tile_blocks_x = materialize_num_tile_blocks(0);
		mlir::Value num_tile_blocks_y = materialize_num_tile_blocks(1);

		mlir::Value tile_block_x = mlir::spirv::UModOp::create(rewriter, loc, tile_linear_id, num_tile_blocks_x);

		mlir::Value div_x = mlir::spirv::UDivOp::create(rewriter, loc, tile_linear_id, num_tile_blocks_x);

		mlir::Value tile_block_y = mlir::spirv::UModOp::create(rewriter, loc, div_x, num_tile_blocks_y);

		mlir::Value tile_block_z = mlir::spirv::UDivOp::create(rewriter, loc, div_x, num_tile_blocks_y);

		return {tile_block_x, tile_block_y, tile_block_z};
	}
};

struct StoreViewTkoOpConversion final : mlir::OpConversionPattern<mlir::cuda_tile::StoreViewTkoOp> {
	StoreViewTkoOpConversion(mlir::MLIRContext *context, TileMaterializer &materializer) : mlir::OpConversionPattern<mlir::cuda_tile::StoreViewTkoOp>(context), materializer(materializer) {}

	mlir::LogicalResult matchAndRewrite(mlir::cuda_tile::StoreViewTkoOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) const override {
		mlir::Location loc = op.getLoc();

		mlir::Value element_value = materializer.materialize_tile(op.getTile(), loc, rewriter);
		mlir::Value partition_local_index = materializer.materialize_partition_local_linear_index(op.getOperation(), op.getView(), loc, rewriter);

		llvm::SmallVector<mlir::Value> indices(op.getIndex().begin(), op.getIndex().end());

		mlir::Value element_ptr = materializer.materialize_partition_view_element_ptr(op.getView(), indices, partition_local_index, loc, rewriter);

		mlir::spirv::StoreOp::create(rewriter, loc, element_ptr, element_value);

		rewriter.eraseOp(op);

		return mlir::success();
	}

  private:
	TileMaterializer &materializer;
};

struct AssumeOpConversion : public mlir::OpConversionPattern<mlir::cuda_tile::AssumeOp> {
	using mlir::OpConversionPattern<mlir::cuda_tile::AssumeOp>::OpConversionPattern;

	mlir::LogicalResult matchAndRewrite(mlir::cuda_tile::AssumeOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) const override {
		rewriter.replaceOp(op, adaptor.getOperands()[0]);
		return mlir::success();
	}
};

template <typename OpTy> struct EraseOpConversion : public mlir::OpConversionPattern<OpTy> {
	using Base = mlir::OpConversionPattern<OpTy>;
	using OpAdaptor = typename Base::OpAdaptor;

	using Base::Base;

	mlir::LogicalResult matchAndRewrite(OpTy op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) const override {
		if (op->getNumResults() != 0) {
			return rewriter.notifyMatchFailure(op, "cannot erase metadata op with results");
		}

		rewriter.eraseOp(op);
		return mlir::success();
	}
};

struct ReturnOpConversion final : mlir::OpConversionPattern<mlir::cuda_tile::ReturnOp> {
	using mlir::OpConversionPattern<mlir::cuda_tile::ReturnOp>::OpConversionPattern;

	mlir::LogicalResult matchAndRewrite(mlir::cuda_tile::ReturnOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) const override {
		if (!adaptor.getOperands().empty()) {
			return rewriter.notifyMatchFailure(op, "expected return without operands");
		}

		rewriter.replaceOpWithNewOp<mlir::spirv::ReturnOp>(op);
		return mlir::success();
	}
};