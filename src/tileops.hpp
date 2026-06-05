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

static bool attr_contains(mlir::Operation *op, llvm::StringRef name, llvm::StringRef text) {
	mlir::Attribute attr = op->getAttr(name);

	std::string value;
	llvm::raw_string_ostream stream(value);
	attr.print(stream);
	stream.flush();

	return llvm::StringRef(value).contains(text);
}

struct TileMaterializer {

	LoweringContext &lowering_context;

	TileMaterializer(LoweringContext &lowering_context) : lowering_context(lowering_context) {}

	mlir::Value materialize_scalar(mlir::Value value, mlir::Location loc, mlir::ConversionPatternRewriter &rewriter, mlir::Value partition_view = {}) {
		if (auto it = value_map.find(value); it != value_map.end())
			return it->second;

		if (mlir::Value remapped_value = rewriter.getRemappedValue(value)) {
			if (remapped_value != value)
				return remapped_value;
		}

		if (auto block_arg = mlir::dyn_cast<mlir::BlockArgument>(value))
			return block_arg;

		mlir::Operation *op = value.getDefiningOp();

		if (auto assume_op = mlir::dyn_cast<mlir::cuda_tile::AssumeOp>(op))
			return materialize_scalar(assume_op->getOperand(0), loc, rewriter, partition_view);

		if (auto addi_op = mlir::dyn_cast<mlir::cuda_tile::AddIOp>(op))
			return materialize_binopi<mlir::cuda_tile::AddIOp, mlir::spirv::IAddOp>(addi_op, loc, rewriter, partition_view);

		if (auto subi_op = mlir::dyn_cast<mlir::cuda_tile::SubIOp>(op))
			return materialize_binopi<mlir::cuda_tile::SubIOp, mlir::spirv::ISubOp>(subi_op, loc, rewriter, partition_view);

		if (auto muli_op = mlir::dyn_cast<mlir::cuda_tile::MulIOp>(op))
			return materialize_binopi<mlir::cuda_tile::MulIOp, mlir::spirv::IMulOp>(muli_op, loc, rewriter, partition_view);

		if (auto divi_op = mlir::dyn_cast<mlir::cuda_tile::DivIOp>(op))
			return materialize_binopi<mlir::cuda_tile::DivIOp, mlir::spirv::UDivOp>(divi_op, loc, rewriter, partition_view);

		if (auto remi_op = mlir::dyn_cast<mlir::cuda_tile::RemIOp>(op))
			return materialize_binopi<mlir::cuda_tile::RemIOp, mlir::spirv::UModOp>(remi_op, loc, rewriter, partition_view);

		if (auto andi_op = mlir::dyn_cast<mlir::cuda_tile::AndIOp>(op))
			return materialize_binopi<mlir::cuda_tile::AndIOp, mlir::spirv::LogicalAndOp>(andi_op, loc, rewriter, partition_view);

		if (auto xori_op = mlir::dyn_cast<mlir::cuda_tile::XOrIOp>(op))
			return materialize_binopi<mlir::cuda_tile::XOrIOp, mlir::spirv::LogicalNotEqualOp>(xori_op, loc, rewriter, partition_view);

		if (auto mini_op = mlir::dyn_cast<mlir::cuda_tile::MinIOp>(op))
			return materialize_mini(mini_op, loc, rewriter, partition_view);

		if (auto cmpi_op = mlir::dyn_cast<mlir::cuda_tile::CmpIOp>(op))
			return materialize_cmpi(cmpi_op, loc, rewriter, partition_view);

		if (auto select_op = mlir::dyn_cast<mlir::cuda_tile::SelectOp>(op))
			return materialize_select(select_op, loc, rewriter, partition_view);

		if (auto constant_op = mlir::dyn_cast<mlir::cuda_tile::ConstantOp>(op))
			return materialize_constant(constant_op, loc, rewriter);

		if (auto get_tile_block_id_op = mlir::dyn_cast<mlir::cuda_tile::GetTileBlockIdOp>(op)) {
			assert(partition_view && "get_tile_block_id requires a partition view");

			auto result = llvm::cast<mlir::OpResult>(value);
			llvm::SmallVector<mlir::Value, 3> block_ids = materialize_get_tile_block_id(get_tile_block_id_op, partition_view, loc, rewriter);

			return block_ids[result.getResultNumber()];
		}

		return materialize_tile(value, loc, rewriter);
	}

	mlir::Value materialize_tile(mlir::Value value, mlir::Location loc, mlir::ConversionPatternRewriter &rewriter) {
		if (mlir::Value remapped_value = rewriter.getRemappedValue(value)) {
			if (remapped_value != value) {
				return remapped_value;
			}
		}

		if (auto it = value_map.find(value); it != value_map.end())
			return it->second;

		mlir::Operation *op = value.getDefiningOp();
		assert(op && "block arguments should have been remapped");

		if (auto load_op = mlir::dyn_cast<mlir::cuda_tile::LoadViewTkoOp>(op))
			return materialize_load_view_tko(load_op, loc, rewriter);

		if (auto constant_op = mlir::dyn_cast<mlir::cuda_tile::ConstantOp>(op))
			return materialize_constant(constant_op, loc, rewriter);

		if (auto addf_op = mlir::dyn_cast<mlir::cuda_tile::AddFOp>(op))
			return materialize_fadd(addf_op, loc, rewriter);

		if (auto mmaf_op = mlir::dyn_cast<mlir::cuda_tile::MmaFOp>(op))
			return materialize_mmaf(mmaf_op, loc, rewriter);

		if (auto reshape_op = mlir::dyn_cast<mlir::cuda_tile::ReshapeOp>(op))
			return materialize_tile(reshape_op.getSource(), loc, rewriter);

		if (auto ftof_op = mlir::dyn_cast<mlir::cuda_tile::FToFOp>(op))
			return materialize_ftof(ftof_op, loc, rewriter);

		if (auto get_index_space_shape_op = mlir::dyn_cast<mlir::cuda_tile::GetIndexSpaceShapeOp>(op))
			return materialize_get_index_space_shape(get_index_space_shape_op, loc, rewriter);

		if (auto for_op = mlir::dyn_cast<mlir::cuda_tile::ForOp>(op)) {
			auto result = llvm::cast<mlir::OpResult>(value);
			return materialize_for(for_op, result.getResultNumber(), loc, rewriter);
		}

		op->emitError() << "unsupported operation to materialize tile value: " << op->getName();
		llvm::report_fatal_error("cuda_tile lowering failed");
	}

	template <typename CudaOp, typename SpirvOp> mlir::Value materialize_binopi(CudaOp op, mlir::Location loc, mlir::ConversionPatternRewriter &rewriter, mlir::Value partition_view = {}) {
		mlir::Value lhs = materialize_scalar(op.getLhs(), loc, rewriter, partition_view);
		mlir::Value rhs = materialize_scalar(op.getRhs(), loc, rewriter, partition_view);
		return SpirvOp::create(rewriter, loc, lhs.getType(), lhs, rhs);
	}

	mlir::Value materialize_cmpi(mlir::cuda_tile::CmpIOp op, mlir::Location loc, mlir::ConversionPatternRewriter &rewriter, mlir::Value partition_view = {}) {
		mlir::Value lhs = materialize_scalar(op.getLhs(), loc, rewriter, partition_view);
		mlir::Value rhs = materialize_scalar(op.getRhs(), loc, rewriter, partition_view);

		mlir::Type i1_type = rewriter.getI1Type();

		if (attr_contains(op.getOperation(), "comparison_predicate", "less_than"))
			return mlir::spirv::ULessThanOp::create(rewriter, loc, i1_type, lhs, rhs);

		if (attr_contains(op.getOperation(), "comparison_predicate", "not_equal"))
			return mlir::spirv::INotEqualOp::create(rewriter, loc, i1_type, lhs, rhs);

		op->emitError() << "unsupported cmpi predicate";
		llvm::report_fatal_error("cuda_tile lowering failed");
	}

	mlir::Value materialize_select(mlir::cuda_tile::SelectOp op, mlir::Location loc, mlir::ConversionPatternRewriter &rewriter, mlir::Value partition_view = {}) {
		mlir::Value condition = materialize_scalar(op.getCond(), loc, rewriter, partition_view);
		mlir::Value true_value = materialize_scalar(op.getValIfTrue(), loc, rewriter, partition_view);
		mlir::Value false_value = materialize_scalar(op.getValIfFalse(), loc, rewriter, partition_view);

		return mlir::spirv::SelectOp::create(rewriter, loc, true_value.getType(), condition, true_value, false_value);
	}

	mlir::Value materialize_mini(mlir::cuda_tile::MinIOp op, mlir::Location loc, mlir::ConversionPatternRewriter &rewriter, mlir::Value partition_view = {}) {
		mlir::Value lhs = materialize_scalar(op.getLhs(), loc, rewriter, partition_view);
		mlir::Value rhs = materialize_scalar(op.getRhs(), loc, rewriter, partition_view);

		mlir::Value condition = mlir::spirv::ULessThanOp::create(rewriter, loc, rewriter.getI1Type(), lhs, rhs);

		return mlir::spirv::SelectOp::create(rewriter, loc, lhs.getType(), condition, lhs, rhs);
	}

	mlir::Value materialize_partition_view_element_ptr(mlir::Value partition_view, llvm::SmallVector<mlir::Value> indices, mlir::Value element_index, mlir::Location loc, mlir::ConversionPatternRewriter &rewriter) {
		PartitionViewInfo &pview_info = lowering_context.partition_views[partition_view];
		TensorViewInfo &tview_info = lowering_context.tensor_views[pview_info.tensor_view];

		mlir::Type i32_type = rewriter.getI32Type();

		for (mlir::Value &index : indices)
			index = materialize_scalar(index, loc, rewriter, partition_view);

		mlir::Value tile_offset = indices[0];

		mlir::Value stride;
		for (unsigned i = 1; i < indices.size(); ++i) {
			IndexValue &extent = tview_info.shape[i - 1];

			mlir::Value extent_value;
			if (extent.is_static()) {
				extent_value = mlir::spirv::ConstantOp::create(rewriter, loc, i32_type, rewriter.getIntegerAttr(i32_type, extent.static_value));
			} else {
				extent_value = materialize_scalar(extent.dynamic_value, loc, rewriter, partition_view);
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

	mlir::Value materialize_ftof(mlir::cuda_tile::FToFOp op, mlir::Location loc, mlir::ConversionPatternRewriter &rewriter) { return materialize_tile(op.getOperand(), loc, rewriter); }

	mlir::Value materialize_mmaf(mlir::cuda_tile::MmaFOp op, mlir::Location loc, mlir::ConversionPatternRewriter &rewriter) {
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

	mlir::Value materialize_for(mlir::cuda_tile::ForOp op, unsigned result_index, mlir::Location loc, mlir::ConversionPatternRewriter &rewriter) {
	mlir::OpBuilder::InsertionGuard guard(rewriter);

	mlir::Type i32_type = rewriter.getI32Type();

	unsigned result_count = op->getNumResults();

	mlir::Value lower_bound = materialize_scalar(op->getOperand(0), loc, rewriter);
	mlir::Value upper_bound = materialize_scalar(op->getOperand(1), loc, rewriter);
	mlir::Value step = materialize_scalar(op->getOperand(2), loc, rewriter);

	llvm::SmallVector<mlir::Value> init_values;
	for (unsigned i = 0; i < result_count; ++i)
		init_values.push_back(materialize_tile(op->getOperand(3 + i), loc, rewriter));

	llvm::SmallVector<mlir::Type> result_types;
	for (mlir::Value value : init_values)
		result_types.push_back(value.getType());

	mlir::OperationState state(loc, mlir::spirv::LoopOp::getOperationName());
	state.addTypes(result_types);
	state.addAttribute(
		"loop_control",
		mlir::spirv::LoopControlAttr::get(
			rewriter.getContext(),
			mlir::spirv::LoopControl::None));
	state.addRegion();

	auto loop_op = mlir::cast<mlir::spirv::LoopOp>(rewriter.create(state));
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

	auto block_args = [](mlir::Block *block) {
		return llvm::SmallVector<mlir::Value>(block->getArguments().begin(), block->getArguments().end());
	};

	llvm::SmallVector<mlir::Value> entry_args;
	entry_args.push_back(lower_bound);
	entry_args.append(init_values.begin(), init_values.end());

	rewriter.setInsertionPointToStart(entry_block);
	mlir::spirv::BranchOp::create(rewriter, loc, header_block, entry_args);

	rewriter.setInsertionPointToStart(header_block);

	mlir::Value condition = mlir::spirv::ULessThanOp::create(
		rewriter,
		loc,
		rewriter.getI1Type(),
		header_block->getArgument(0),
		upper_bound);

	llvm::SmallVector<mlir::Value> merge_args(header_block->getArguments().begin() + 1, header_block->getArguments().end());

	mlir::spirv::BranchConditionalOp::create(
		rewriter,
		loc,
		condition,
		body_block,
		block_args(header_block),
		merge_block,
		merge_args);

	mlir::Block &old_body = *op.getBody();
	auto continue_op = mlir::cast<mlir::cuda_tile::ContinueOp>(old_body.getTerminator());

	llvm::SmallVector<std::pair<mlir::Value, mlir::Value>> old_values;

	auto map_value = [&](mlir::Value original, mlir::Value replacement) {
		if (auto it = value_map.find(original); it != value_map.end())
			old_values.push_back({original, it->second});
		else
			old_values.push_back({original, mlir::Value()});

		value_map[original] = replacement;
	};

	map_value(old_body.getArgument(0), body_block->getArgument(0));

	for (unsigned i = 0; i < result_count; ++i)
		map_value(old_body.getArgument(i + 1), body_block->getArgument(i + 1));

	rewriter.setInsertionPointToStart(body_block);

	llvm::SmallVector<mlir::Value> continue_values;
	for (unsigned i = 0; i < result_count; ++i)
		continue_values.push_back(materialize_tile(continue_op->getOperand(i), loc, rewriter));

	mlir::Value next_index = mlir::spirv::IAddOp::create(rewriter, loc, body_block->getArgument(0), step);

	llvm::SmallVector<mlir::Value> continue_args;
	continue_args.push_back(next_index);
	continue_args.append(continue_values.begin(), continue_values.end());

	mlir::spirv::BranchOp::create(rewriter, loc, continue_block, continue_args);

	for (auto it = old_values.rbegin(); it != old_values.rend(); ++it) {
		if (it->second)
			value_map[it->first] = it->second;
		else
			value_map.erase(it->first);
	}

	rewriter.setInsertionPointToStart(continue_block);
	mlir::spirv::BranchOp::create(rewriter, loc, header_block, block_args(continue_block));

	rewriter.setInsertionPointToStart(merge_block);
	mlir::spirv::MergeOp::create(rewriter, loc, block_args(merge_block));

	return loop_op->getResult(result_index);
}

	mlir::Value materialize_get_index_space_shape(mlir::cuda_tile::GetIndexSpaceShapeOp op, mlir::Location loc, mlir::ConversionPatternRewriter &rewriter) {
		mlir::Type i32_type = rewriter.getI32Type();

		return mlir::spirv::ConstantOp::create(rewriter, loc, i32_type, rewriter.getIntegerAttr(i32_type, 5));
	}

	mlir::Value materialize_constant(mlir::cuda_tile::ConstantOp op, mlir::Location loc, mlir::ConversionPatternRewriter &rewriter) {
		auto dense_attr = mlir::cast<mlir::DenseElementsAttr>(op->getAttr("value"));
		mlir::Type element_type = dense_attr.getElementType();

		if (auto integer_type = mlir::dyn_cast<mlir::IntegerType>(element_type)) {
			llvm::APInt value = dense_attr.getSplatValue<llvm::APInt>();

			return mlir::spirv::ConstantOp::create(rewriter, loc, integer_type, rewriter.getIntegerAttr(integer_type, value));
		}

		llvm::APFloat value = dense_attr.getSplatValue<llvm::APFloat>();

		return mlir::spirv::ConstantOp::create(rewriter, loc, element_type, mlir::FloatAttr::get(element_type, value));
	}

	llvm::SmallDenseMap<mlir::Value, mlir::Value, 16> value_map;
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
