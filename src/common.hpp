#pragma once

#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include "cuda_tile/Dialect/CudaTile/IR/Types.h"
#include <optional>

struct IndexValue {
	enum class Kind { Static, Dynamic };

	Kind kind;
	int64_t static_value = 0;
	mlir::Value dynamic_value;

	static IndexValue from_static(int64_t value) { return IndexValue{Kind::Static, value, nullptr}; }

	static IndexValue from_dynamic(mlir::Value value) { return IndexValue{Kind::Dynamic, 0, value}; }

	bool is_static() const { return kind == Kind::Static; }

	bool is_dynamic() const { return kind == Kind::Dynamic; }
};

struct TensorViewInfo {
	mlir::Value base_ptr;
	llvm::SmallVector<IndexValue> shape;
	llvm::SmallVector<IndexValue> strides;
};

struct PartitionViewInfo {
	mlir::Value tensor_view;
	llvm::SmallVector<int32_t> tile_shape;
	std::optional<mlir::cuda_tile::PaddingValue> padding_value;
};

struct ContinueTargetInfo {
	mlir::Block *continue_block = nullptr;
	mlir::Value step;
};

struct LoweringContext {
	llvm::DenseMap<mlir::Value, TensorViewInfo> tensor_views;
	llvm::DenseMap<mlir::Value, PartitionViewInfo> partition_views;
};

static bool is_op(mlir::Operation *op, llvm::StringRef name) { return op->getName().getStringRef() == name; }

static bool is_entry_array_base_type(mlir::Type type) { return mlir::isa<mlir::spirv::PointerType>(type); }

static bool is_i32_type(mlir::Type type) {
	auto integer_type = mlir::dyn_cast<mlir::IntegerType>(type);
	return integer_type && integer_type.getWidth() == 32;
}