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

	enum class MaterializationCacheKind : unsigned {
		scalar,
		tile_element,
		global_linear_index,
		partition_local_linear_index,
	};

	struct MaterializationCacheKey {
		MaterializationCacheKind kind;
		mlir::Block *block = nullptr;
		mlir::Operation *op = nullptr;
		mlir::Value value;
		mlir::Value value2;

		bool operator==(const MaterializationCacheKey &other) const { return kind == other.kind && block == other.block && op == other.op && value == other.value && value2 == other.value2; }
	};

	struct MaterializationCacheKeyHash {
		size_t operator()(const MaterializationCacheKey &key) const {
			auto opaque_value = [](mlir::Value value) -> const void * { return value ? value.getAsOpaquePointer() : nullptr; };

			return static_cast<size_t>(llvm::hash_combine(static_cast<unsigned>(key.kind), key.block, key.op, opaque_value(key.value), opaque_value(key.value2)));
		}
	};

	static bool can_reuse_cached_value(mlir::Value value, mlir::ConversionPatternRewriter &rewriter) {
		mlir::Operation *def_op = value.getDefiningOp();
		if (!def_op)
			return true;

		mlir::Block *block = rewriter.getInsertionBlock();
		if (def_op->getBlock() != block)
			return false;

		auto insertion_point = rewriter.getInsertionPoint();
		return insertion_point == block->end() || def_op->isBeforeInBlock(&*insertion_point);
	}

	template <typename Fn> mlir::Value get_or_materialize(MaterializationCacheKey key, mlir::ConversionPatternRewriter &rewriter, Fn &&fn) {
		key.block = rewriter.getInsertionBlock();

		if (auto it = materialization_cache.find(key); it != materialization_cache.end()) {
			if (can_reuse_cached_value(it->second, rewriter))
				return it->second;
		}

		mlir::Value value = fn();
		materialization_cache[key] = value;
		return value;
	}

	std::unordered_map<MaterializationCacheKey, mlir::Value, MaterializationCacheKeyHash> materialization_cache;
	llvm::DenseMap<std::pair<mlir::Block *, int64_t>, mlir::Value> i32_constant_map;

	mlir::Value materialize_scalar(mlir::Value value, mlir::Location loc, mlir::ConversionPatternRewriter &rewriter, mlir::Value partition_view = {}) {
		if (auto it = value_map.find(value); it != value_map.end())
			return it->second;

		if (mlir::Value remapped_value = rewriter.getRemappedValue(value)) {
			if (remapped_value != value)
				return remapped_value;
		}

		if (auto block_arg = mlir::dyn_cast<mlir::BlockArgument>(value))
			return block_arg;

		MaterializationCacheKey key;
		key.kind = MaterializationCacheKind::scalar;
		key.value = value;
		key.value2 = partition_view;

		return get_or_materialize(key, rewriter, [&]() -> mlir::Value {
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

			if (auto get_index_space_shape_op = mlir::dyn_cast<mlir::cuda_tile::GetIndexSpaceShapeOp>(op)) {
				auto result = llvm::cast<mlir::OpResult>(value);
				return materialize_get_index_space_shape(get_index_space_shape_op, result.getResultNumber(), loc, rewriter);
			}

			if (auto constant_op = mlir::dyn_cast<mlir::cuda_tile::ConstantOp>(op))
				return materialize_constant(constant_op, loc, rewriter);

			if (auto get_tile_block_id_op = mlir::dyn_cast<mlir::cuda_tile::GetTileBlockIdOp>(op)) {
				assert(partition_view && "get_tile_block_id requires a partition view");

				auto result = llvm::cast<mlir::OpResult>(value);
				llvm::SmallVector<mlir::Value, 3> block_ids = materialize_get_tile_block_id(get_tile_block_id_op, partition_view, loc, rewriter);

				return block_ids[result.getResultNumber()];
			}

			op->emitError() << "unsupported operation to materialize scale value: " << op->getName();
			llvm::report_fatal_error("cuda_tile lowering failed");
		});
	}

	mlir::Value materialize_tile_element(mlir::Value value, mlir::Location loc, mlir::ConversionPatternRewriter &rewriter, mlir::Value element_index = {}) {
		if (mlir::Value remapped_value = rewriter.getRemappedValue(value)) {
			if (remapped_value != value) {
				return remapped_value;
			}
		}

		if (auto it = value_map.find(value); it != value_map.end())
			return it->second;

		mlir::Operation *op = value.getDefiningOp();
		assert(op && "block arguments should have been remapped");

		if (!element_index) {
			element_index = get_or_create_global_linear_index(op, loc, rewriter);
		}

		MaterializationCacheKey key;
		key.kind = MaterializationCacheKind::tile_element;
		key.value = value;
		key.value2 = element_index;

		return get_or_materialize(key, rewriter, [&]() -> mlir::Value {
			if (auto load_op = mlir::dyn_cast<mlir::cuda_tile::LoadViewTkoOp>(op))
				return materialize_load_view_tko(load_op, loc, rewriter, element_index);

			if (auto addf_op = mlir::dyn_cast<mlir::cuda_tile::AddFOp>(op))
				return materialize_fadd(addf_op, loc, rewriter, element_index);

			if (auto mmaf_op = mlir::dyn_cast<mlir::cuda_tile::MmaFOp>(op))
				return materialize_mmaf_for(mmaf_op, loc, rewriter, element_index);

			if (auto reshape_op = mlir::dyn_cast<mlir::cuda_tile::ReshapeOp>(op))
				return materialize_tile_element(reshape_op.getSource(), loc, rewriter, element_index);

			if (auto ftof_op = mlir::dyn_cast<mlir::cuda_tile::FToFOp>(op))
				return materialize_tile_element(ftof_op.getOperand(), loc, rewriter, element_index);

			if (auto for_op = mlir::dyn_cast<mlir::cuda_tile::ForOp>(op)) {
				auto result = llvm::cast<mlir::OpResult>(value);
				return materialize_for(for_op, result.getResultNumber(), loc, rewriter, element_index);
			}

			op->emitError() << "unsupported operation to materialize tile value: " << op->getName();
			llvm::report_fatal_error("cuda_tile lowering failed");
		});
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

	mlir::Value get_or_create_global_linear_index(mlir::Operation *op, mlir::Location loc, mlir::ConversionPatternRewriter &rewriter) {

		MaterializationCacheKey key;
		key.kind = MaterializationCacheKind::global_linear_index;

		return get_or_materialize(key, rewriter, [&]() -> mlir::Value {
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
		});
	}

	mlir::Value get_or_create_partition_local_linear_index(mlir::Operation *op, mlir::Value partition_view, mlir::Location loc, mlir::ConversionPatternRewriter &rewriter, mlir::Value element_index) {

		MaterializationCacheKey key;
		key.kind = MaterializationCacheKind::partition_local_linear_index;
		key.op = op;
		key.value = partition_view;
		key.value2 = element_index;

		return get_or_materialize(key, rewriter, [&]() -> mlir::Value {
			PartitionViewInfo &pview_info = lowering_context.partition_views[partition_view];

			mlir::Type i32_type = rewriter.getI32Type();

			int32_t total_tile_size = 1;
			for (int32_t value : pview_info.tile_shape)
				total_tile_size *= value;

			mlir::Value total_tile_size_value = mlir::spirv::ConstantOp::create(rewriter, loc, i32_type, rewriter.getIntegerAttr(i32_type, total_tile_size));

			return mlir::spirv::UModOp::create(rewriter, loc, element_index, total_tile_size_value);
		});
	}

	mlir::Value get_or_create_i32_constant(int64_t value, mlir::Location loc, mlir::ConversionPatternRewriter &rewriter) {
		auto key = std::make_pair(rewriter.getInsertionBlock(), value);

		if (auto it = i32_constant_map.find(key); it != i32_constant_map.end())
			return it->second;

		mlir::Type i32_type = rewriter.getI32Type();
		mlir::Value result = mlir::spirv::ConstantOp::create(rewriter, loc, i32_type, rewriter.getIntegerAttr(i32_type, value));

		i32_constant_map[key] = result;
		return result;
	}

	mlir::Value materialize_load_view_tko(mlir::cuda_tile::LoadViewTkoOp op, mlir::Location loc, mlir::ConversionPatternRewriter &rewriter, mlir::Value element_index) {
		llvm::SmallVector<mlir::Value> indices(op.getIndex().begin(), op.getIndex().end());
		mlir::Value partition = op.getView();
		mlir::Value partition_local_index = get_or_create_partition_local_linear_index(op, partition, loc, rewriter, element_index);
		mlir::Value element_ptr = materialize_partition_view_element_ptr(op.getView(), indices, partition_local_index, loc, rewriter);
		return mlir::spirv::LoadOp::create(rewriter, loc, element_ptr);
	}

	mlir::Value materialize_fadd(mlir::cuda_tile::AddFOp op, mlir::Location loc, mlir::ConversionPatternRewriter &rewriter, mlir::Value element_index) {
		mlir::Value lhs = materialize_tile_element(op.getLhs(), loc, rewriter, element_index);
		mlir::Value rhs = materialize_tile_element(op.getRhs(), loc, rewriter, element_index);
		return mlir::spirv::FAddOp::create(rewriter, loc, lhs.getType(), lhs, rhs);
	}

	mlir::Value materialize_mmaf_for(mlir::cuda_tile::MmaFOp op, mlir::Location loc, mlir::ConversionPatternRewriter &rewriter, mlir::Value element_index) {
		mlir::OpBuilder::InsertionGuard guard(rewriter);

		mlir::Type i32_type = rewriter.getI32Type();

		auto lhs_type = op.getLhs().getType();
		auto rhs_type = op.getRhs().getType();
		auto acc_type = op.getAcc().getType();

		llvm::ArrayRef<int64_t> lhs_shape = lhs_type.getShape();
		llvm::ArrayRef<int64_t> rhs_shape = rhs_type.getShape();
		llvm::ArrayRef<int64_t> acc_shape = acc_type.getShape();

		assert(lhs_shape.size() == 2 && rhs_shape.size() == 2 && acc_shape.size() == 2);

		int64_t m = lhs_shape[0];
		int64_t k = lhs_shape[1];
		int64_t n = rhs_shape[1];

		assert(rhs_shape[0] == k);
		assert(acc_shape[0] == m);
		assert(acc_shape[1] == n);

		mlir::Value output_tile_size = get_or_create_i32_constant(m * n, loc, rewriter);

		element_index = mlir::spirv::UModOp::create(rewriter, loc, i32_type, element_index, output_tile_size);

		mlir::Value n_value = get_or_create_i32_constant(n, loc, rewriter);

		mlir::Value row = mlir::spirv::UDivOp::create(rewriter, loc, i32_type, element_index, n_value);
		mlir::Value col = mlir::spirv::UModOp::create(rewriter, loc, i32_type, element_index, n_value);

		mlir::Value initial_result = materialize_scalar(op.getAcc(), loc, rewriter);

		llvm::SmallVector<mlir::Type> result_types;
		result_types.push_back(initial_result.getType());

		mlir::OperationState state(loc, mlir::spirv::LoopOp::getOperationName());
		state.addTypes(result_types);
		state.addAttribute("loop_control", mlir::spirv::LoopControlAttr::get(rewriter.getContext(), mlir::spirv::LoopControl::None));
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
		header_block->addArgument(initial_result.getType(), loc);

		body_block->addArgument(i32_type, loc);
		body_block->addArgument(initial_result.getType(), loc);

		continue_block->addArgument(i32_type, loc);
		continue_block->addArgument(initial_result.getType(), loc);

		merge_block->addArgument(initial_result.getType(), loc);

		auto block_args = [](mlir::Block *block) { return llvm::SmallVector<mlir::Value>(block->getArguments().begin(), block->getArguments().end()); };

		rewriter.setInsertionPointToStart(entry_block);

		mlir::Value zero = get_or_create_i32_constant(0, loc, rewriter);
		mlir::spirv::BranchOp::create(rewriter, loc, header_block, mlir::ValueRange{zero, initial_result});

		rewriter.setInsertionPointToStart(header_block);

		mlir::Value k_index = header_block->getArgument(0);
		mlir::Value current_result = header_block->getArgument(1);
		mlir::Value condition = mlir::spirv::ULessThanOp::create(rewriter, loc, rewriter.getI1Type(), k_index, get_or_create_i32_constant(k, loc, rewriter));

		mlir::spirv::BranchConditionalOp::create(rewriter, loc, condition, body_block, block_args(header_block), merge_block, mlir::ValueRange{current_result});

		rewriter.setInsertionPointToStart(body_block);

		mlir::Value body_k_index = body_block->getArgument(0);
		mlir::Value body_result = body_block->getArgument(1);

		mlir::Value lhs_row_offset = mlir::spirv::IMulOp::create(rewriter, loc, i32_type, row, get_or_create_i32_constant(k, loc, rewriter));
		mlir::Value lhs_index = mlir::spirv::IAddOp::create(rewriter, loc, i32_type, lhs_row_offset, body_k_index);

		mlir::Value rhs_row_offset = mlir::spirv::IMulOp::create(rewriter, loc, i32_type, body_k_index, get_or_create_i32_constant(n, loc, rewriter));
		mlir::Value rhs_index = mlir::spirv::IAddOp::create(rewriter, loc, i32_type, rhs_row_offset, col);

		mlir::Value lhs = materialize_tile_element(op.getLhs(), loc, rewriter, lhs_index);
		mlir::Value rhs = materialize_tile_element(op.getRhs(), loc, rewriter, rhs_index);

		mlir::Value product = mlir::spirv::FMulOp::create(rewriter, loc, lhs.getType(), lhs, rhs);
		mlir::Value next_result = mlir::spirv::FAddOp::create(rewriter, loc, body_result.getType(), body_result, product);
		mlir::Value next_k_index = mlir::spirv::IAddOp::create(rewriter, loc, i32_type, body_k_index, get_or_create_i32_constant(1, loc, rewriter));

		mlir::spirv::BranchOp::create(rewriter, loc, continue_block, mlir::ValueRange{next_k_index, next_result});

		rewriter.setInsertionPointToStart(continue_block);
		mlir::spirv::BranchOp::create(rewriter, loc, header_block, block_args(continue_block));

		rewriter.setInsertionPointToStart(merge_block);
		mlir::spirv::MergeOp::create(rewriter, loc, block_args(merge_block));

		return loop_op->getResult(0);
	}

	mlir::Value materialize_mmaf(mlir::cuda_tile::MmaFOp op, mlir::Location loc, mlir::ConversionPatternRewriter &rewriter, mlir::Value element_index) {
		mlir::Type i32_type = rewriter.getI32Type();

		auto make_i32_constant = [&](int64_t value) -> mlir::Value { return mlir::spirv::ConstantOp::create(rewriter, loc, i32_type, rewriter.getIntegerAttr(i32_type, value)); };

		auto lhs_type = op.getLhs().getType();
		auto rhs_type = op.getRhs().getType();
		auto acc_type = op.getAcc().getType();

		llvm::ArrayRef<int64_t> lhs_shape = lhs_type.getShape();
		llvm::ArrayRef<int64_t> rhs_shape = rhs_type.getShape();
		llvm::ArrayRef<int64_t> acc_shape = acc_type.getShape();

		assert(lhs_shape.size() == 2 && rhs_shape.size() == 2 && acc_shape.size() == 2);

		int64_t m = lhs_shape[0];
		int64_t k = lhs_shape[1];
		int64_t n = rhs_shape[1];

		assert(rhs_shape[0] == k);
		assert(acc_shape[0] == m);
		assert(acc_shape[1] == n);

		mlir::Value output_tile_size = make_i32_constant(m * n);

		element_index = mlir::spirv::UModOp::create(rewriter, loc, i32_type, element_index, output_tile_size);

		mlir::Value n_value = make_i32_constant(n);

		mlir::Value row = mlir::spirv::UDivOp::create(rewriter, loc, i32_type, element_index, n_value);

		mlir::Value col = mlir::spirv::UModOp::create(rewriter, loc, i32_type, element_index, n_value);

		mlir::Value result = materialize_scalar(op.getAcc(), loc, rewriter);

		for (int64_t k_index = 0; k_index < k; ++k_index) {
			mlir::Value lhs_row_offset = mlir::spirv::IMulOp::create(rewriter, loc, i32_type, row, make_i32_constant(k));

			mlir::Value lhs_index = mlir::spirv::IAddOp::create(rewriter, loc, i32_type, lhs_row_offset, make_i32_constant(k_index));

			mlir::Value rhs_row_offset = mlir::spirv::IMulOp::create(rewriter, loc, i32_type, make_i32_constant(k_index), make_i32_constant(n));

			mlir::Value rhs_index = mlir::spirv::IAddOp::create(rewriter, loc, i32_type, rhs_row_offset, col);

			mlir::Value lhs = materialize_tile_element(op.getLhs(), loc, rewriter, lhs_index);

			mlir::Value rhs = materialize_tile_element(op.getRhs(), loc, rewriter, rhs_index);

			mlir::Value product = mlir::spirv::FMulOp::create(rewriter, loc, lhs.getType(), lhs, rhs);

			result = mlir::spirv::FAddOp::create(rewriter, loc, result.getType(), result, product);
		}

		return result;
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

		mlir::Value global_linear_id = get_or_create_global_linear_index(op.getOperation(), loc, rewriter);
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

	mlir::Value materialize_for(mlir::cuda_tile::ForOp op, unsigned result_index, mlir::Location loc, mlir::ConversionPatternRewriter &rewriter, mlir::Value element_index) {
		mlir::OpBuilder::InsertionGuard guard(rewriter);

		mlir::Type i32_type = rewriter.getI32Type();

		unsigned result_count = op->getNumResults();

		mlir::Value lower_bound = materialize_scalar(op->getOperand(0), loc, rewriter);
		mlir::Value upper_bound = materialize_scalar(op->getOperand(1), loc, rewriter);
		mlir::Value step = materialize_scalar(op->getOperand(2), loc, rewriter);

		llvm::SmallVector<mlir::Value> init_values;
		for (unsigned i = 0; i < result_count; ++i)
			init_values.push_back(materialize_scalar(op->getOperand(3 + i), loc, rewriter));

		llvm::SmallVector<mlir::Type> result_types;
		for (mlir::Value value : init_values)
			result_types.push_back(value.getType());

		mlir::OperationState state(loc, mlir::spirv::LoopOp::getOperationName());
		state.addTypes(result_types);
		state.addAttribute("loop_control", mlir::spirv::LoopControlAttr::get(rewriter.getContext(), mlir::spirv::LoopControl::None));
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

		auto block_args = [](mlir::Block *block) { return llvm::SmallVector<mlir::Value>(block->getArguments().begin(), block->getArguments().end()); };

		llvm::SmallVector<mlir::Value> entry_args;
		entry_args.push_back(lower_bound);
		entry_args.append(init_values.begin(), init_values.end());

		rewriter.setInsertionPointToStart(entry_block);
		mlir::spirv::BranchOp::create(rewriter, loc, header_block, entry_args);

		rewriter.setInsertionPointToStart(header_block);

		mlir::Value condition = mlir::spirv::ULessThanOp::create(rewriter, loc, rewriter.getI1Type(), header_block->getArgument(0), upper_bound);

		llvm::SmallVector<mlir::Value> merge_args(header_block->getArguments().begin() + 1, header_block->getArguments().end());

		mlir::spirv::BranchConditionalOp::create(rewriter, loc, condition, body_block, block_args(header_block), merge_block, merge_args);

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
			continue_values.push_back(materialize_tile_element(continue_op->getOperand(i), loc, rewriter, element_index));

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

	mlir::Value materialize_get_index_space_shape(mlir::cuda_tile::GetIndexSpaceShapeOp op, unsigned result_index, mlir::Location loc, mlir::ConversionPatternRewriter &rewriter) {
		mlir::Value partition_view = op->getOperand(0);

		PartitionViewInfo &pview_info = lowering_context.partition_views[partition_view];
		TensorViewInfo &tview_info = lowering_context.tensor_views[pview_info.tensor_view];

		mlir::Type i32_type = rewriter.getI32Type();

		auto make_i32_constant = [&](int64_t value) -> mlir::Value { return mlir::spirv::ConstantOp::create(rewriter, loc, i32_type, rewriter.getIntegerAttr(i32_type, value)); };

		auto ceil_div_static = [](int64_t lhs, int64_t rhs) -> int64_t { return (lhs + rhs - 1) / rhs; };

		IndexValue tensor_dim = result_index < tview_info.shape.size() ? tview_info.shape[result_index] : IndexValue::from_static(1);

		int64_t tile_dim = result_index < pview_info.tile_shape.size() ? pview_info.tile_shape[result_index] : 1;

		if (tensor_dim.is_static())
			return make_i32_constant(ceil_div_static(tensor_dim.static_value, tile_dim));

		mlir::Value tensor_dim_value = get_remapped_index_or_self(tensor_dim.dynamic_value, rewriter);

		if (tile_dim == 1)
			return tensor_dim_value;

		mlir::Value tile_dim_value = make_i32_constant(tile_dim);
		mlir::Value offset = make_i32_constant(tile_dim - 1);
		mlir::Value numerator = mlir::spirv::IAddOp::create(rewriter, loc, i32_type, tensor_dim_value, offset);

		return mlir::spirv::UDivOp::create(rewriter, loc, i32_type, numerator, tile_dim_value);
	}

	mlir::Value materialize_constant(mlir::cuda_tile::ConstantOp op, mlir::Location loc, mlir::ConversionPatternRewriter &rewriter) {
		auto dense_attr = mlir::cast<mlir::DenseElementsAttr>(op->getAttr("value"));
		mlir::Type element_type = dense_attr.getElementType();

		if (auto integer_type = mlir::dyn_cast<mlir::IntegerType>(element_type)) {
			llvm::APInt value = dense_attr.getSplatValue<llvm::APInt>();

			if (integer_type.isInteger(32))
				return get_or_create_i32_constant(value.getSExtValue(), loc, rewriter);

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

		mlir::Value element_value = materializer.materialize_tile_element(op.getTile(), loc, rewriter);
		mlir::Value global_linear_index = materializer.get_or_create_global_linear_index(op.getOperation(), loc, rewriter);
		mlir::Value partition_local_index = materializer.get_or_create_partition_local_linear_index(op.getOperation(), op.getView(), loc, rewriter, global_linear_index);

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
