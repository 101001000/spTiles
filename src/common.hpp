#pragma once

#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include "cuda_tile/Dialect/CudaTile/IR/Types.h"

struct EntryArrayInfo {
    mlir::Value array;
    llvm::SmallVector<mlir::Value, 3> lengths;
    llvm::SmallVector<mlir::Value, 3> strides;
    uint32_t dimensions = 0;
};

struct TensorViewInfo {
    mlir::Value base;
    llvm::SmallVector<mlir::Value> shape;
    mlir::Type type;
    mlir::Operation *source_op = nullptr;
};

struct PartitionViewInfo {
    mlir::Value tensor_view;
    mlir::Value base;
    mlir::cuda_tile::PartitionViewType type;
    mlir::Operation *source_op = nullptr;
};

struct ContinueTargetInfo {
    mlir::Block *continue_block = nullptr;
    mlir::Value step;
};

struct CudaTileSpmdLoweringContext {
    CudaTileSpmdLoweringContext(
        uint32_t local_size_x,
        llvm::DenseMap<mlir::Value, EntryArrayInfo> &entry_arrays,
        llvm::DenseMap<mlir::Value, TensorViewInfo> &tensor_views,
        llvm::DenseMap<mlir::Value, PartitionViewInfo> &partition_views,
        llvm::DenseMap<mlir::Value, mlir::Value> &forwarded_values)
        : local_size_x(local_size_x),
          entry_arrays(entry_arrays),
          tensor_views(tensor_views),
          partition_views(partition_views),
          forwarded_values(forwarded_values) {}

    uint32_t local_size_x = 16;

    llvm::DenseMap<mlir::Value, EntryArrayInfo> &entry_arrays;
    llvm::DenseMap<mlir::Value, TensorViewInfo> &tensor_views;
    llvm::DenseMap<mlir::Value, PartitionViewInfo> &partition_views;
    llvm::DenseMap<mlir::Value, mlir::Value> &forwarded_values;

    llvm::DenseMap<mlir::Operation *, ContinueTargetInfo> continue_targets;


    mlir::Value get_forwarded_value(mlir::Value value) const {
        while (true) {
            auto it = forwarded_values.find(value);
            if (it == forwarded_values.end()) {
                return value;
            }

            value = it->second;
        }
    }

    mlir::Value get_or_create_lane_id(
        mlir::Operation *op,
        mlir::OpBuilder &builder) const {
        mlir::Location loc = op->getLoc();
        mlir::Type i32_type = builder.getI32Type();

        mlir::Value local_id =
            mlir::spirv::getBuiltinVariableValue(
                op,
                mlir::spirv::BuiltIn::LocalInvocationId,
                i32_type,
                builder);

        return mlir::spirv::CompositeExtractOp::create(
            builder,
            loc,
            i32_type,
            local_id,
            builder.getI32ArrayAttr({0}));
    }

    mlir::Value get_or_create_workgroup_id_x(
        mlir::Operation *op,
        mlir::OpBuilder &builder) const {
        mlir::Location loc = op->getLoc();
        mlir::Type i32_type = builder.getI32Type();

        mlir::Value workgroup_id =
            mlir::spirv::getBuiltinVariableValue(
                op,
                mlir::spirv::BuiltIn::WorkgroupId,
                i32_type,
                builder);

        return mlir::spirv::CompositeExtractOp::create(
            builder,
            loc,
            i32_type,
            workgroup_id,
            builder.getI32ArrayAttr({0}));
    }

    mlir::Value create_global_linear_id(
        mlir::Operation *op,
        mlir::OpBuilder &builder) const {
        mlir::Location loc = op->getLoc();
        mlir::Type i32_type = builder.getI32Type();

        mlir::Value lane = get_or_create_lane_id(op, builder);
        mlir::Value workgroup_id_x =
            get_or_create_workgroup_id_x(op, builder);

        mlir::Value c_local_size_x = mlir::spirv::ConstantOp::create(
            builder,
            loc,
            i32_type,
            builder.getI32IntegerAttr(
                static_cast<int32_t>(local_size_x)));

        mlir::Value group_base = mlir::spirv::IMulOp::create(
            builder,
            loc,
            i32_type,
            workgroup_id_x,
            c_local_size_x);

        return mlir::spirv::IAddOp::create(
            builder,
            loc,
            i32_type,
            group_base,
            lane);
    }

    mlir::Value create_storage_buffer_access(
        mlir::OpBuilder &builder,
        mlir::Location loc,
        mlir::Value base,
        mlir::Value idx) const {
        mlir::Type i32_type = builder.getI32Type();

        mlir::Value zero = mlir::spirv::ConstantOp::create(
            builder,
            loc,
            i32_type,
            builder.getI32IntegerAttr(0));

        llvm::SmallVector<mlir::Value, 2> indices;
        indices.push_back(zero);
        indices.push_back(idx);

        return mlir::spirv::AccessChainOp::create(
            builder,
            loc,
            base,
            indices);
    }

    void create_linearized_array_access(
        mlir::OpBuilder &builder,
        mlir::Location loc,
        const EntryArrayInfo &info,
        mlir::Value linear_id,
        mlir::Value &active,
        mlir::Value &index) const {
        mlir::Type i32_type = builder.getI32Type();

        mlir::Value total_elements = info.lengths[0];
        for (uint32_t dim = 1; dim < info.dimensions; ++dim) {
            total_elements = mlir::spirv::IMulOp::create(
                builder,
                loc,
                i32_type,
                total_elements,
                info.lengths[dim]);
        }

        active = mlir::spirv::ULessThanOp::create(
            builder,
            loc,
            linear_id,
            total_elements);

        mlir::Value c0 = mlir::spirv::ConstantOp::create(
            builder,
            loc,
            i32_type,
            builder.getI32IntegerAttr(0));

        mlir::Value safe_linear_id = mlir::spirv::SelectOp::create(
            builder,
            loc,
            i32_type,
            active,
            linear_id,
            c0);

        mlir::Value x = safe_linear_id;
        if (info.dimensions > 1) {
            x = mlir::spirv::UModOp::create(
                builder,
                loc,
                i32_type,
                safe_linear_id,
                info.lengths[0]);
        }

        index = mlir::spirv::IMulOp::create(
            builder,
            loc,
            i32_type,
            x,
            info.strides[0]);

        if (info.dimensions > 1) {
            mlir::Value linear_after_x = mlir::spirv::UDivOp::create(
                builder,
                loc,
                i32_type,
                safe_linear_id,
                info.lengths[0]);

            mlir::Value y = linear_after_x;
            if (info.dimensions > 2) {
                y = mlir::spirv::UModOp::create(
                    builder,
                    loc,
                    i32_type,
                    linear_after_x,
                    info.lengths[1]);
            }

            mlir::Value y_offset = mlir::spirv::IMulOp::create(
                builder,
                loc,
                i32_type,
                y,
                info.strides[1]);

            index = mlir::spirv::IAddOp::create(
                builder,
                loc,
                i32_type,
                index,
                y_offset);

            if (info.dimensions > 2) {
                mlir::Value z = mlir::spirv::UDivOp::create(
                    builder,
                    loc,
                    i32_type,
                    linear_after_x,
                    info.lengths[1]);

                mlir::Value z_offset = mlir::spirv::IMulOp::create(
                    builder,
                    loc,
                    i32_type,
                    z,
                    info.strides[2]);

                index = mlir::spirv::IAddOp::create(
                    builder,
                    loc,
                    i32_type,
                    index,
                    z_offset);
            }
        }
    }
};