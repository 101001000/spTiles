#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseSet.h"

#include <optional>

namespace {

    
struct CudaTileModuleToSpirvPass
    : public mlir::PassWrapper<CudaTileModuleToSpirvPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CudaTileModuleToSpirvPass)

    llvm::StringRef getArgument() const override {
        return "cuda-tile-module-to-spirv-module";
    }

    llvm::StringRef getDescription() const override {
        return "Lower CUDA Tile module shell to SPIR-V module shell.";
    }

    void getDependentDialects(mlir::DialectRegistry &registry) const override {
        registry.insert<mlir::spirv::SPIRVDialect>();
    }

    void runOnOperation() override {
        mlir::ModuleOp root_module = getOperation();

        replace_modules(root_module);
        replace_entries(root_module);

        collect_tokens(root_module);
        collect_assumes(root_module);
        collect_reshapes(root_module);
        collect_tensor_views(root_module);
        collect_partition_views(root_module);

        lower_cuda_tile_ops(root_module);

        erase_lowered_cuda_tile_ops(root_module);

        attach_spirv_abi_attrs(root_module);

        replace_returns(root_module);

        root_module.walk([&](mlir::Operation *op) {
            if (op->getName().getDialectNamespace() != "cuda_tile") {
                return;
            }

            
            op->emitError("cuda_tile op left after lowering: ") << op->getName().getStringRef();

            llvm::errs() << "\n=== remaining cuda_tile op ===\n";
            op->print(llvm::errs(), mlir::OpPrintingFlags().printGenericOpForm());
            llvm::errs() << "\n==============================\n";
            llvm::errs().flush();
        });
    }

private:

    uint32_t local_size_x = 16;

    struct EntryArrayInfo {
        mlir::Value array;
        llvm::SmallVector<mlir::Value, 3> lengths;
        llvm::SmallVector<mlir::Value, 3> strides;
        uint32_t dimensions = 0;
    };

    struct TensorViewInfo {
        mlir::Value base;
        llvm::SmallVector<mlir::Value> shape;
        mlir::Type type; // Esto debería ser mlir::cuda_tile::TensorViewType
        mlir::Operation *source_op;
    };

    struct PartitionViewInfo {
        mlir::Value tensor_view;
        mlir::Value base;
        mlir::cuda_tile::PartitionViewType type;
        mlir::Operation *source_op;
    };

    llvm::DenseSet<mlir::Value> tokens;
    llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Attribute>> assumptions;
    llvm::DenseMap<mlir::Value, EntryArrayInfo> entry_arrays;
    llvm::DenseMap<mlir::Value, TensorViewInfo> tensor_views;
    llvm::DenseMap<mlir::Value, PartitionViewInfo> partition_views;
    llvm::DenseMap<mlir::Value, mlir::Value> forwarded_values;
    llvm::DenseMap<mlir::Value, mlir::Value> lowered_values;
    llvm::DenseSet<mlir::Operation *> lowered_ops;

    mlir::Value get_forwarded_value(mlir::Value value) {
        while (true) {
            auto it = forwarded_values.find(value);
            if (it == forwarded_values.end()) {
                return value;
            }

            value = it->second;
        }
    }

    void add_forwarded_value(mlir::Value result, mlir::Value source) {
        forwarded_values[result] = get_forwarded_value(source);
    }

    mlir::Value get_lowered_value(mlir::Value value) {
        mlir::Value original_value = value;
        value = get_forwarded_value(value);

        auto it = lowered_values.find(value);
        if (it != lowered_values.end()) {
            return it->second;
        }

        if (is_already_lowered_type(value.getType())) {
            return value;
        }

        llvm::errs() << "\n=== missing lowered value ===\n";

        llvm::errs() << "value: ";
        value.print(llvm::errs());
        llvm::errs() << "\n";

        llvm::errs() << "value type: ";
        value.getType().print(llvm::errs());
        llvm::errs() << "\n";

        llvm::errs() << "=============================\n";
        llvm::errs().flush();

        return {};
    }

    static bool is_op(mlir::Operation *op, llvm::StringRef name) {
        return op->getName().getStringRef() == name;
    }

    static bool is_entry_array_base_type(mlir::Type type) {
        return mlir::isa<mlir::spirv::PointerType>(type);
    }

    static bool is_i32_type(mlir::Type type) {
        auto integer_type = mlir::dyn_cast<mlir::IntegerType>(type);
        return integer_type && integer_type.getWidth() == 32;
    }

    static bool is_already_lowered_type(mlir::Type type) {
        return mlir::isa<mlir::IntegerType>(type) ||
               mlir::isa<mlir::FloatType>(type) ||
               mlir::isa<mlir::spirv::PointerType>(type);
    }

    std::string type_to_string(mlir::Type type) {
        std::string text;
        llvm::raw_string_ostream stream(text);
        type.print(stream);
        stream.flush();
        return text;
    }

    mlir::Type get_storage_buffer_f32_runtime_array_ptr_type(
        mlir::OpBuilder &builder) {
        mlir::Type f32_type = builder.getF32Type();

        auto runtime_array_type = mlir::spirv::RuntimeArrayType::get(
            f32_type,
            /*stride=*/4);

        llvm::SmallVector<mlir::Type, 1> member_types;
        member_types.push_back(runtime_array_type);

        auto struct_type = mlir::spirv::StructType::get(
            member_types,
            /*offsetInfo=*/0);

        return mlir::spirv::PointerType::get(
            struct_type,
            mlir::spirv::StorageClass::StorageBuffer);
    }

    mlir::Type convert_entry_type(mlir::Type type, mlir::OpBuilder &builder) {
        std::string text = type_to_string(type);

        if (llvm::StringRef(text).contains("tile<ptr<f32>>")) {
            return get_storage_buffer_f32_runtime_array_ptr_type(builder);
        }

        if (llvm::StringRef(text).contains("tile<i32>")) {
            return builder.getI32Type();
        }

        return type;
    }

    mlir::FunctionType convert_entry_function_type(
        mlir::FunctionType function_type,
        mlir::OpBuilder &builder) {
        llvm::SmallVector<mlir::Type> inputs;
        llvm::SmallVector<mlir::Type> results;

        for (mlir::Type input_type : function_type.getInputs()) {
            inputs.push_back(convert_entry_type(input_type, builder));
        }

        for (mlir::Type result_type : function_type.getResults()) {
            results.push_back(convert_entry_type(result_type, builder));
        }

        return builder.getFunctionType(inputs, results);
    }

    mlir::Value get_or_create_lane_id(mlir::Operation *op,
                                    mlir::OpBuilder &builder) {
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

    mlir::Value get_or_create_workgroup_id_x(mlir::Operation *op,
                                             mlir::OpBuilder &builder) {
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

    mlir::Value create_global_linear_id(mlir::Operation *op,
                                        mlir::OpBuilder &builder) {
        mlir::Location loc = op->getLoc();
        mlir::Type i32_type = builder.getI32Type();

        mlir::Value lane = get_or_create_lane_id(op, builder);
        mlir::Value workgroup_id_x = get_or_create_workgroup_id_x(op, builder);

        mlir::Value c_local_size_x = mlir::spirv::ConstantOp::create(
            builder,
            loc,
            i32_type,
            builder.getI32IntegerAttr(static_cast<int32_t>(local_size_x)));

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

    mlir::Value create_storage_buffer_access(mlir::OpBuilder &builder,
                                             mlir::Location loc,
                                             mlir::Value base,
                                             mlir::Value idx) {
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
        mlir::Value &index) {
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

    void collect_assumes(mlir::ModuleOp root_module) {
        root_module.walk([&](mlir::Operation *op) {
            if (!is_op(op, "cuda_tile.assume")) {
                return;
            }

            mlir::Value input = op->getOperand(0);
            mlir::Value result = op->getResult(0);
            mlir::Value base = get_forwarded_value(input);

            if (auto predicate = op->getAttr("predicate")) {
                assumptions[base].push_back(predicate);
            }

            add_forwarded_value(result, input);
            lowered_ops.insert(op);
        });
    }

    void lower_load_view_tko(mlir::Operation *op) {
        mlir::OpBuilder builder(op);
        mlir::Location loc = op->getLoc();

        mlir::Value pview = get_forwarded_value(op->getOperand(0));
        mlir::Value token = op->getOperand(op->getNumOperands() - 1);

        auto partition_it = partition_views.find(pview);
        if (partition_it == partition_views.end()) {
            op->emitError("unknown partition view");
            return;
        }

        mlir::Value base = get_forwarded_value(partition_it->second.base);

        auto array_it = entry_arrays.find(base);
        if (array_it == entry_arrays.end()) {
            op->emitError("partition view base is not a known entry array");
            signalPassFailure();
            return;
        }

        mlir::Value linear_id = create_global_linear_id(op, builder);
        mlir::Value active;
        mlir::Value index;
        create_linearized_array_access(
            builder,
            loc,
            array_it->second,
            linear_id,
            active,
            index);

        mlir::Value ptr = create_storage_buffer_access(
            builder,
            loc,
            base,
            index);

        mlir::Value loaded = mlir::spirv::LoadOp::create(
            builder,
            loc,
            ptr,
            nullptr,
            nullptr);

        lowered_values[op->getResult(0)] = loaded;
        add_forwarded_value(op->getResult(1), token);
        lowered_ops.insert(op);
    }

    void lower_get_tile_block_id(mlir::Operation *op) {
        if (op->getNumResults() != 3) {
            op->emitError("expected get_tile_block_id to produce 3 results");
            return;
        }

        mlir::OpBuilder builder(op);
        mlir::Location loc = op->getLoc();
        mlir::Type i32_type = builder.getI32Type();

        mlir::Value workgroup_id =
            mlir::spirv::getBuiltinVariableValue(
                op,
                mlir::spirv::BuiltIn::WorkgroupId,
                i32_type,
                builder);

        if (!workgroup_id) {
            op->emitError("failed to get or create WorkgroupId builtin");
            return;
        }

        mlir::Value block_id_x =
            mlir::spirv::CompositeExtractOp::create(
                builder,
                loc,
                i32_type,
                workgroup_id,
                builder.getI32ArrayAttr({0}));

        mlir::Value block_id_y =
            mlir::spirv::CompositeExtractOp::create(
                builder,
                loc,
                i32_type,
                workgroup_id,
                builder.getI32ArrayAttr({1}));

        mlir::Value block_id_z =
            mlir::spirv::CompositeExtractOp::create(
                builder,
                loc,
                i32_type,
                workgroup_id,
                builder.getI32ArrayAttr({2}));

        lowered_values[op->getResult(0)] = block_id_x;
        lowered_values[op->getResult(1)] = block_id_y;
        lowered_values[op->getResult(2)] = block_id_z;
        lowered_ops.insert(op);
    }

    void lower_get_index_space_shape(mlir::Operation *op) {
        if (op->getNumOperands() != 1) {
            op->emitError("expected get_index_space_shape with exactly one operand");
            return;
        }

        mlir::Value pview = get_forwarded_value(op->getOperand(0));

        auto partition_it = partition_views.find(pview);
        if (partition_it == partition_views.end()) {
            op->emitError("unknown partition view");
            return;
        }

        auto tensor_it = tensor_views.find(partition_it->second.tensor_view);
        if (tensor_it == tensor_views.end()) {
            op->emitError("partition view tensor view is not known");
            return;
        }

        llvm::ArrayRef<int> tile_shape =
            partition_it->second.type.getTileShape().asArrayRef();

        if (op->getNumResults() != tile_shape.size()) {
            op->emitError("get_index_space_shape result count does not match tile rank");
            return;
        }

        if (tensor_it->second.shape.size() != tile_shape.size()) {
            op->emitError("tensor view shape rank does not match tile rank");
            return;
        }

        mlir::OpBuilder builder(op);
        mlir::Location loc = op->getLoc();
        mlir::Type i32_type = builder.getI32Type();

        for (uint32_t dim = 0; dim < tile_shape.size(); ++dim) {
            mlir::Value shape_dim =
                get_forwarded_value(tensor_it->second.shape[dim]);

            if (!is_i32_type(shape_dim.getType())) {
                op->emitError("expected i32 tensor view shape dimension");
                return;
            }

            int32_t tile_dim = static_cast<int32_t>(tile_shape[dim]);

            mlir::Value c_tile_dim = mlir::spirv::ConstantOp::create(
                builder,
                loc,
                i32_type,
                builder.getI32IntegerAttr(tile_dim));

            mlir::Value c_tile_dim_minus_one = mlir::spirv::ConstantOp::create(
                builder,
                loc,
                i32_type,
                builder.getI32IntegerAttr(tile_dim - 1));

            mlir::Value numerator = mlir::spirv::IAddOp::create(
                builder,
                loc,
                i32_type,
                shape_dim,
                c_tile_dim_minus_one);

            mlir::Value result = mlir::spirv::UDivOp::create(
                builder,
                loc,
                i32_type,
                numerator,
                c_tile_dim);

            lowered_values[op->getResult(dim)] = result;
        }

        lowered_ops.insert(op);
    }

    void collect_reshapes(mlir::ModuleOp root_module) {
        root_module.walk([&](mlir::Operation *op) {
            if (!is_op(op, "cuda_tile.reshape")) {
                return;
            }

            if (op->getNumOperands() != 1 || op->getNumResults() != 1) {
                op->emitError("expected cuda_tile.reshape with one operand and one result");
                signalPassFailure();
                return;
            }

            add_forwarded_value(op->getResult(0), op->getOperand(0));
            lowered_ops.insert(op);
        });
    }

    void lower_constant(mlir::Operation *op) {
        if (op->getNumOperands() != 0 || op->getNumResults() != 1) {
            op->emitError("expected cuda_tile.constant with no operands and one result");
            signalPassFailure();
            return;
        }

        mlir::Attribute value_attr = op->getAttr("value");
        if (!value_attr) {
            op->emitError("expected cuda_tile.constant value attribute");
            signalPassFailure();
            return;
        }

        auto dense_attr = mlir::dyn_cast<mlir::DenseElementsAttr>(value_attr);
        if (!dense_attr) {
            op->emitError("expected cuda_tile.constant value to be a dense elements attribute");
            signalPassFailure();
            return;
        }

        if (!dense_attr.isSplat()) {
            op->emitError("only splat cuda_tile.constant values are supported");
            signalPassFailure();
            return;
        }

        mlir::OpBuilder builder(op);
        mlir::Location loc = op->getLoc();
        mlir::Type element_type = dense_attr.getElementType();

        if (is_i32_type(element_type)) {
            llvm::APInt value = dense_attr.getSplatValue<llvm::APInt>();

            mlir::Value constant = mlir::spirv::ConstantOp::create(
                builder,
                loc,
                element_type,
                builder.getIntegerAttr(element_type, value));

            lowered_values[op->getResult(0)] = constant;
            lowered_ops.insert(op);
            return;
        }

        auto float_type = mlir::dyn_cast<mlir::FloatType>(element_type);
        if (float_type && float_type.getWidth() == 32) {
            llvm::APFloat value = dense_attr.getSplatValue<llvm::APFloat>();

            mlir::Value constant = mlir::spirv::ConstantOp::create(
                builder,
                loc,
                element_type,
                mlir::FloatAttr::get(element_type, value));

            lowered_values[op->getResult(0)] = constant;
            lowered_ops.insert(op);
            return;
        }

        op->emitError("unsupported cuda_tile.constant element type: ")
            << element_type;
        signalPassFailure();
    }


    void lower_store_view_tko(mlir::Operation *op) {
        if (op->getNumResults() != 1) {
            op->emitError("expected store_view_tko to produce one token result");
            return;
        }

        mlir::OpBuilder builder(op);
        mlir::Location loc = op->getLoc();

        mlir::Value value = get_lowered_value(op->getOperand(0));
        mlir::Value pview = get_forwarded_value(op->getOperand(1));
        mlir::Value token = op->getOperand(op->getNumOperands() - 1);

        auto partition_it = partition_views.find(pview);
        if (partition_it == partition_views.end()) {
            op->emitError("unknown partition view");
            return;
        }

        mlir::Value base = get_forwarded_value(partition_it->second.base);

        auto array_it = entry_arrays.find(base);
        if (array_it == entry_arrays.end()) {
            op->emitError("partition view base is not a known entry array");
            signalPassFailure();
            return;
        }

        auto base_type = mlir::dyn_cast<mlir::spirv::PointerType>(base.getType());
        if (!base_type) {
            op->emitError("expected SPIR-V pointer base for store_view_tko, got ")
                << base.getType();
            return;
        }

        if (!mlir::isa<mlir::FloatType>(value.getType())) {
            op->emitError("expected scalar floating-point value for store_view_tko, got ")
                << value.getType();
            return;
        }

        mlir::Value linear_id = create_global_linear_id(op, builder);
        mlir::Value active;
        mlir::Value index;
        create_linearized_array_access(
            builder,
            loc,
            array_it->second,
            linear_id,
            active,
            index);

        auto selection_op = mlir::spirv::SelectionOp::create(
            builder,
            loc,
            mlir::spirv::SelectionControl::None);

        mlir::Region &region = selection_op.getBody();
        auto *header_block = new mlir::Block();
        auto *then_block = new mlir::Block();
        auto *merge_block = new mlir::Block();

        region.push_back(header_block);
        region.push_back(then_block);
        region.push_back(merge_block);

        builder.setInsertionPointToStart(header_block);

        mlir::spirv::BranchConditionalOp::create(
            builder,
            loc,
            active,
            then_block,
            mlir::ValueRange{},
            merge_block,
            mlir::ValueRange{});

        builder.setInsertionPointToStart(then_block);

        mlir::Value ptr = create_storage_buffer_access(
            builder,
            loc,
            base,
            index);

        mlir::spirv::StoreOp::create(
            builder,
            loc,
            ptr,
            value,
            nullptr,
            nullptr);

        mlir::spirv::BranchOp::create(
            builder,
            loc,
            merge_block);

        builder.setInsertionPointToStart(merge_block);
        mlir::spirv::MergeOp::create(builder, loc);

        builder.setInsertionPointAfter(selection_op);

        add_forwarded_value(op->getResult(0), token);
        lowered_ops.insert(op);
    }

    void lower_addf(mlir::Operation *op) {
        if (op->getNumOperands() != 2) {
            op->emitError("expected cuda_tile.addf with exactly two operands");
            return;
        }

        if (op->getNumResults() != 1) {
            op->emitError("expected cuda_tile.addf with exactly one result");
            return;
        }

        mlir::Value lhs = get_lowered_value(op->getOperand(0));
        mlir::Value rhs = get_lowered_value(op->getOperand(1));

        if (lhs.getType() != rhs.getType()) {
            op->emitError("cuda_tile.addf operands must have the same lowered type, got ")
                << lhs.getType() << " and " << rhs.getType();
            return;
        }

        mlir::Type scalar_type = lhs.getType();

        if (!mlir::isa<mlir::FloatType>(scalar_type)) {
            op->emitError("NEW addf error: expected scalar floating-point operands, got ")
                << scalar_type;
            return;
        }

        mlir::OpBuilder builder(op);

        mlir::Value result = mlir::spirv::FAddOp::create(
            builder,
            op->getLoc(),
            scalar_type,
            lhs,
            rhs);

        lowered_values[op->getResult(0)] = result;
        lowered_ops.insert(op);
    }


    void lower_for(mlir::Operation *op) {
        if (op->getNumRegions() != 1 || op->getRegion(0).empty()) {
            op->emitError("expected cuda_tile.for with one non-empty region");
            signalPassFailure();
            return;
        }

        mlir::Region &old_region = op->getRegion(0);
        if (!old_region.hasOneBlock()) {
            op->emitError("expected cuda_tile.for with one body block");
            signalPassFailure();
            return;
        }

        unsigned result_count = op->getNumResults();
        if (op->getNumOperands() != result_count + 3) {
            op->emitError("expected cuda_tile.for operands as lower, upper, step, iter_args...");
            signalPassFailure();
            return;
        }

        mlir::Block &old_body = old_region.front();
        if (old_body.getNumArguments() != result_count + 1) {
            op->emitError("cuda_tile.for body arguments must be iv plus iter args");
            signalPassFailure();
            return;
        }

        mlir::Operation *continue_op = old_body.getTerminator();
        if (!continue_op || !is_op(continue_op, "cuda_tile.continue")) {
            op->emitError("expected cuda_tile.for body to end with cuda_tile.continue");
            signalPassFailure();
            return;
        }

        if (continue_op->getNumOperands() != result_count) {
            continue_op->emitError("cuda_tile.continue operand count must match cuda_tile.for result count");
            signalPassFailure();
            return;
        }

        mlir::Value lower_bound = get_lowered_value(op->getOperand(0));
        mlir::Value upper_bound = get_lowered_value(op->getOperand(1));
        mlir::Value step = get_lowered_value(op->getOperand(2));

        if (!lower_bound || !upper_bound || !step) {
            signalPassFailure();
            return;
        }

        llvm::SmallVector<mlir::Value> init_values;
        llvm::SmallVector<mlir::Type> result_types;
        init_values.reserve(result_count);
        result_types.reserve(result_count);

        for (unsigned i = 0; i < result_count; ++i) {
            mlir::Value init = get_lowered_value(op->getOperand(i + 3));
            if (!init) {
                signalPassFailure();
                return;
            }

            init_values.push_back(init);
            result_types.push_back(init.getType());
        }

        mlir::OpBuilder builder(op);
        mlir::Location loc = op->getLoc();
        mlir::Type i32_type = builder.getI32Type();

        mlir::OperationState state(loc, mlir::spirv::LoopOp::getOperationName());
        state.addTypes(result_types);
        state.addAttribute(
            "loop_control",
            mlir::spirv::LoopControlAttr::get(
                builder.getContext(),
                mlir::spirv::LoopControl::None));
        state.addRegion();

        auto loop_op = mlir::cast<mlir::spirv::LoopOp>(builder.create(state));
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

        lowered_values[header_block->getArgument(0)] = header_block->getArgument(0);
        lowered_values[body_block->getArgument(0)] = body_block->getArgument(0);
        lowered_values[continue_block->getArgument(0)] = continue_block->getArgument(0);

        for (mlir::Type type : result_types) {
            header_block->addArgument(type, loc);
            body_block->addArgument(type, loc);
            continue_block->addArgument(type, loc);
            merge_block->addArgument(type, loc);
        }

        for (unsigned i = 0; i < result_count; ++i) {
            lowered_values[header_block->getArgument(i + 1)] = header_block->getArgument(i + 1);
            lowered_values[body_block->getArgument(i + 1)] = body_block->getArgument(i + 1);
            lowered_values[continue_block->getArgument(i + 1)] = continue_block->getArgument(i + 1);
            lowered_values[merge_block->getArgument(i)] = merge_block->getArgument(i);
        }

        llvm::SmallVector<mlir::Value> entry_args;
        entry_args.push_back(lower_bound);
        entry_args.append(init_values.begin(), init_values.end());

        builder.setInsertionPointToStart(entry_block);
        mlir::spirv::BranchOp::create(builder, loc, header_block, entry_args);

        llvm::SmallVector<mlir::Value> header_args;
        for (mlir::BlockArgument arg : header_block->getArguments()) {
            header_args.push_back(arg);
        }

        llvm::SmallVector<mlir::Value> merge_args;
        for (unsigned i = 0; i < result_count; ++i) {
            merge_args.push_back(header_block->getArgument(i + 1));
        }

        builder.setInsertionPointToStart(header_block);
        mlir::Value condition = mlir::spirv::ULessThanOp::create(
            builder,
            loc,
            header_block->getArgument(0),
            upper_bound);

        mlir::spirv::BranchConditionalOp::create(
            builder,
            loc,
            condition,
            body_block,
            header_args,
            merge_block,
            merge_args);

        for (unsigned i = 0; i < old_body.getNumArguments(); ++i) {
            old_body.getArgument(i).replaceAllUsesWith(body_block->getArgument(i));
        }

        while (!old_body.empty() && &old_body.front() != continue_op) {
            old_body.front().moveBefore(body_block, body_block->end());
        }

        lower_cuda_tile_ops_in_block(body_block);

        llvm::SmallVector<mlir::Value> yielded_values;
        yielded_values.reserve(result_count);
        for (mlir::Value operand : continue_op->getOperands()) {
            mlir::Value yielded = get_lowered_value(operand);
            if (!yielded) {
                signalPassFailure();
                return;
            }

            yielded_values.push_back(yielded);
        }

        builder.setInsertionPointToEnd(body_block);
        mlir::Value next_iv = mlir::spirv::IAddOp::create(
            builder,
            loc,
            i32_type,
            body_block->getArgument(0),
            step);

        llvm::SmallVector<mlir::Value> continue_args;
        continue_args.push_back(next_iv);
        continue_args.append(yielded_values.begin(), yielded_values.end());

        mlir::spirv::BranchOp::create(
            builder,
            loc,
            continue_block,
            continue_args);

        llvm::SmallVector<mlir::Value> backedge_args;
        for (mlir::BlockArgument arg : continue_block->getArguments()) {
            backedge_args.push_back(arg);
        }

        builder.setInsertionPointToStart(continue_block);
        mlir::spirv::BranchOp::create(
            builder,
            loc,
            header_block,
            backedge_args);

        llvm::SmallVector<mlir::Value> final_values;
        for (mlir::BlockArgument arg : merge_block->getArguments()) {
            final_values.push_back(arg);
        }

        builder.setInsertionPointToStart(merge_block);
        mlir::spirv::MergeOp::create(builder, loc, final_values);

        builder.setInsertionPointAfter(loop_op);

        for (unsigned i = 0; i < result_count; ++i) {
            lowered_values[op->getResult(i)] = loop_op->getResult(i);
        }

        continue_op->erase();
        lowered_ops.insert(op);
    }

    template <typename Callback>
    void lower_ops_by_name(mlir::ModuleOp root_module,
                        llvm::StringRef op_name,
                        Callback callback) {
        llvm::SmallVector<mlir::Operation *> ops;

        root_module.walk([&](mlir::Operation *op) {
            if (is_op(op, op_name)) {
                ops.push_back(op);
            }
        });

        for (mlir::Operation *op : ops) {
            if (!op->getBlock() || lowered_ops.contains(op)) {
                continue;
            }

            callback(op);
        }
    }

    void lower_cuda_tile_ops_in_block(mlir::Block *block) {
        llvm::SmallVector<mlir::Operation *> ops;

        for (mlir::Operation &op : *block) {
            ops.push_back(&op);
        }

        for (mlir::Operation *op : ops) {
            if (!op->getBlock() || lowered_ops.contains(op)) {
                continue;
            }

            if (is_op(op, "cuda_tile.get_tile_block_id")) {
                lower_get_tile_block_id(op);
            } else if (is_op(op, "cuda_tile.get_index_space_shape")) {
                lower_get_index_space_shape(op);
            } else if (is_op(op, "cuda_tile.for")) {
                lower_for(op);
            } else if (is_op(op, "cuda_tile.load_view_tko")) {
                lower_load_view_tko(op);
            } else if (is_op(op, "cuda_tile.addf")) {
                lower_addf(op);
            } else if (is_op(op, "cuda_tile.store_view_tko")) {
                lower_store_view_tko(op);
            } else if (is_op(op, "cuda_tile.constant")) {
                lower_constant(op);
            }
        }
    }

    void lower_cuda_tile_ops(mlir::ModuleOp root_module) {
        lower_ops_by_name(root_module, "cuda_tile.constant",
                        [&](mlir::Operation *op) {
                            lower_constant(op);
                        });

        lower_ops_by_name(root_module, "cuda_tile.get_tile_block_id",
                        [&](mlir::Operation *op) {
                            lower_get_tile_block_id(op);
                        });

        lower_ops_by_name(root_module, "cuda_tile.get_index_space_shape",
                        [&](mlir::Operation *op) {
                            lower_get_index_space_shape(op);
                        });

        lower_ops_by_name(root_module, "cuda_tile.for",
                        [&](mlir::Operation *op) {
                            lower_for(op);
                        });

        lower_ops_by_name(root_module, "cuda_tile.load_view_tko",
                        [&](mlir::Operation *op) {
                            lower_load_view_tko(op);
                        });

        lower_ops_by_name(root_module, "cuda_tile.addf",
                        [&](mlir::Operation *op) {
                            lower_addf(op);
                        });

        lower_ops_by_name(root_module, "cuda_tile.store_view_tko",
                        [&](mlir::Operation *op) {
                            lower_store_view_tko(op);
                        });
    }

    void erase_lowered_cuda_tile_ops(mlir::ModuleOp root_module) {
        bool erased_any = true;

        while (erased_any) {
            erased_any = false;
            llvm::SmallVector<mlir::Operation *> ops;

            root_module.walk([&](mlir::Operation *op) {
                if (lowered_ops.contains(op)) {
                    ops.push_back(op);
                }
            });

            for (mlir::Operation *op : llvm::reverse(ops)) {
                if (!op->use_empty()) {
                    op->emitError("cuda_tile op being used for removal: ") << op->getName().getStringRef();
                    continue;
                }

                lowered_ops.erase(op);
                op->erase();
                erased_any = true;
            }
        }
    }

    void collect_tensor_views(mlir::ModuleOp root_module) {
        root_module.walk([&](mlir::Operation *op) {
            if (!is_op(op, "cuda_tile.make_tensor_view")) {
                return;
            }

            auto segment_sizes =
                op->getAttrOfType<mlir::DenseI32ArrayAttr>("operandSegmentSizes");

            if (!segment_sizes || segment_sizes.size() != 3) {
                op->emitError("expected make_tensor_view operand segment sizes");
                signalPassFailure();
                return;
            }

            uint32_t base_count = static_cast<uint32_t>(segment_sizes[0]);
            uint32_t shape_count = static_cast<uint32_t>(segment_sizes[1]);
            uint32_t stride_count = static_cast<uint32_t>(segment_sizes[2]);

            if (base_count != 1 ||
                op->getNumOperands() != base_count + shape_count + stride_count) {
                op->emitError("invalid make_tensor_view operand segments");
                signalPassFailure();
                return;
            }

            mlir::Value result = op->getResult(0);

            TensorViewInfo info;
            info.base = get_forwarded_value(op->getOperand(0));
            info.type = result.getType();
            info.source_op = op;

            for (uint32_t i = 0; i < shape_count; ++i) {
                info.shape.push_back(
                    get_forwarded_value(op->getOperand(base_count + i)));
            }

            tensor_views[result] = std::move(info);
            lowered_ops.insert(op);
        });
    }


   void collect_partition_views(mlir::ModuleOp root_module) {
        root_module.walk([&](mlir::Operation *op) {
            if (!is_op(op, "cuda_tile.make_partition_view")) {
                return;
            }

            mlir::Value tensor_view = get_forwarded_value(op->getOperand(0));
            mlir::Value result = op->getResult(0);
            

            auto tensor_it = tensor_views.find(tensor_view);
            if (tensor_it == tensor_views.end()) {
                op->emitError("partition view operand is not a known tensor view");
                signalPassFailure();
                return;
            }

            auto partition_type =
            mlir::dyn_cast<mlir::cuda_tile::PartitionViewType>(result.getType());

            if (!partition_type) {
                op->emitError("result is not a cuda_tile partition view");
                signalPassFailure();
                return;
            }

            PartitionViewInfo info;
            info.tensor_view = tensor_view;
            info.base = tensor_it->second.base;
            info.type = partition_type;
            info.source_op = op;

            partition_views[result] = info;
            lowered_ops.insert(op);
        });
    }

    void collect_tokens(mlir::ModuleOp root_module) {
        root_module.walk([&](mlir::Operation *op) {
            if (op->getName().getStringRef() != "cuda_tile.make_token") {
                return;
            }

            tokens.insert(op->getResult(0));
            lowered_ops.insert(op);
        });
    }

    void replace_modules(mlir::ModuleOp root_module) {
        llvm::SmallVector<mlir::Operation *> modules;

        root_module.walk([&](mlir::Operation *op) {
            if (is_op(op, "cuda_tile.module")) {
                modules.push_back(op);
            }
        });

        for (mlir::Operation *module : modules) {
            if (module->getNumRegions() != 1 || module->getRegion(0).empty()) {
                module->emitError("expected cuda_tile.module with one non-empty region");
                signalPassFailure();
                return;
            }

            mlir::OpBuilder builder(module);

            auto spirv_module = mlir::spirv::ModuleOp::create(
                builder,
                module->getLoc(),
                mlir::spirv::AddressingModel::Logical,
                mlir::spirv::MemoryModel::GLSL450);

            auto vce_triple = mlir::spirv::VerCapExtAttr::get(
                mlir::spirv::Version::V_1_0,
                llvm::ArrayRef<mlir::spirv::Capability>{
                    mlir::spirv::Capability::Shader},
                llvm::ArrayRef<mlir::spirv::Extension>{
                    mlir::spirv::Extension::SPV_KHR_storage_buffer_storage_class},
                module->getContext());

            spirv_module->setAttr(
                mlir::spirv::ModuleOp::getVCETripleAttrName(),
                vce_triple);

            spirv_module->setAttr(
                mlir::spirv::getTargetEnvAttrName(),
                mlir::spirv::getDefaultTargetEnv(module->getContext()));

            if (auto sym_name = module->getAttrOfType<mlir::StringAttr>(
                    mlir::SymbolTable::getSymbolAttrName())) {
                spirv_module->setAttr(mlir::SymbolTable::getSymbolAttrName(),
                                      sym_name);
            }

            // Move the CUDA Tile module body into the SPIR-V module shell.
            spirv_module.getRegion().takeBody(module->getRegion(0));

            module->erase();
        }
    }

    void collect_entry_array_params(mlir::spirv::FuncOp func) {
        if (func.getBody().empty()) {
            return;
        }

        mlir::Block &entry_block = func.getBody().front();
        unsigned arg_count = entry_block.getNumArguments();

        for (unsigned arg = 0; arg < arg_count;) {
            mlir::BlockArgument array = entry_block.getArgument(arg);

            if (!is_entry_array_base_type(array.getType())) {
                ++arg;
                continue;
            }

            unsigned metadata_begin = arg + 1;
            unsigned metadata_end = metadata_begin;

            while (metadata_end < arg_count &&
                   !is_entry_array_base_type(
                       entry_block.getArgument(metadata_end).getType())) {
                mlir::BlockArgument metadata = entry_block.getArgument(metadata_end);

                if (!is_i32_type(metadata.getType())) {
                    func.emitError("expected i32 array metadata argument");
                    signalPassFailure();
                    return;
                }

                ++metadata_end;
            }

            unsigned metadata_count = metadata_end - metadata_begin;

            if (metadata_count == 0 || metadata_count % 2 != 0) {
                func.emitError("expected array arguments as ptr, lengths..., strides...");
                signalPassFailure();
                return;
            }

            unsigned dimensions = metadata_count / 2;

            if (dimensions < 1 || dimensions > 3) {
                func.emitError("only 1D, 2D and 3D arrays are supported");
                signalPassFailure();
                return;
            }

            EntryArrayInfo info;
            info.array = array;
            info.dimensions = dimensions;

            for (unsigned dim = 0; dim < dimensions; ++dim) {
                info.lengths.push_back(
                    entry_block.getArgument(metadata_begin + dim));
            }

            for (unsigned dim = 0; dim < dimensions; ++dim) {
                info.strides.push_back(
                    entry_block.getArgument(metadata_begin + dimensions + dim));
            }

            entry_arrays[array] = std::move(info);
            arg = metadata_end;
        }
    }

    void replace_entries(mlir::ModuleOp root_module) {
        llvm::SmallVector<mlir::Operation *> entries;

        root_module.walk([&](mlir::Operation *op) {
            if (is_op(op, "cuda_tile.entry")) {
                entries.push_back(op);
            }
        });

        for (mlir::Operation *entry : entries) {
            auto sym_name = entry->getAttrOfType<mlir::StringAttr>(
                mlir::SymbolTable::getSymbolAttrName());

            auto type_attr =
                entry->getAttrOfType<mlir::TypeAttr>("function_type");

            if (!sym_name || !type_attr) {
                entry->emitError("expected sym_name and function_type");
                signalPassFailure();
                return;
            }

            auto function_type =
                llvm::dyn_cast<mlir::FunctionType>(type_attr.getValue());

            if (!function_type) {
                entry->emitError("expected FunctionType");
                signalPassFailure();
                return;
            }

            mlir::OpBuilder builder(entry);

            mlir::FunctionType converted_function_type =
            convert_entry_function_type(function_type, builder);

            auto spirv_func = mlir::spirv::FuncOp::create(
                builder,
                entry->getLoc(),
                sym_name.getValue(),
                converted_function_type,
                mlir::spirv::FunctionControl::None);

            spirv_func.getBody().takeBody(entry->getRegion(0));

            mlir::Block &entry_block = spirv_func.getBody().front();

            if (entry_block.getNumArguments() != converted_function_type.getNumInputs()) {
                entry->emitError("entry block argument count does not match function type");
                signalPassFailure();
                return;
            }

            for (unsigned i = 0; i < entry_block.getNumArguments(); ++i) {
                entry_block.getArgument(i).setType(converted_function_type.getInput(i));
            }

            collect_entry_array_params(spirv_func);

            entry->erase();
        }
    }

    void erase_unused_spirv_func_args(mlir::ModuleOp root_module) {
        llvm::SmallVector<mlir::spirv::FuncOp> funcs;

        root_module.walk([&](mlir::spirv::FuncOp func) {
            funcs.push_back(func);
        });

        for (mlir::spirv::FuncOp func : funcs) {
            if (func.getBody().empty()) {
                continue;
            }

            mlir::Block &entry_block = func.getBody().front();
            mlir::FunctionType function_type = func.getFunctionType();

            if (entry_block.getNumArguments() != function_type.getNumInputs()) {
                func.emitError("function body argument count does not match function type");
                signalPassFailure();
                return;
            }

            llvm::SmallVector<char> keep_args(entry_block.getNumArguments(), 0);
            llvm::SmallVector<mlir::Type> new_inputs;
            new_inputs.reserve(function_type.getNumInputs());

            for (unsigned i = 0; i < entry_block.getNumArguments(); ++i) {
                if (!entry_block.getArgument(i).use_empty()) {
                    keep_args[i] = 1;
                    new_inputs.push_back(function_type.getInput(i));
                }
            }

            if (new_inputs.size() == function_type.getNumInputs()) {
                continue;
            }

            for (int64_t i = static_cast<int64_t>(entry_block.getNumArguments()) - 1;
                 i >= 0;
                 --i) {
                if (!keep_args[static_cast<unsigned>(i)]) {
                    entry_block.eraseArgument(static_cast<unsigned>(i));
                }
            }

            func.setType(mlir::FunctionType::get(
                func.getContext(),
                new_inputs,
                function_type.getResults()));
        }
    }

    void attach_spirv_abi_attrs(mlir::ModuleOp root_module) {
        llvm::SmallVector<mlir::spirv::FuncOp> funcs;

        root_module.walk([&](mlir::spirv::FuncOp func) {
            funcs.push_back(func);
        });

        for (mlir::spirv::FuncOp func : funcs) {
            mlir::MLIRContext *context = func.getContext();

            llvm::SmallVector<int32_t, 3> workgroup_size;
            workgroup_size.push_back(local_size_x);
            workgroup_size.push_back(1);
            workgroup_size.push_back(1);

            func->setAttr(
                mlir::spirv::getEntryPointABIAttrName(),
                mlir::spirv::getEntryPointABIAttr(context, workgroup_size));

            llvm::StringRef arg_abi_attr_name =
                mlir::spirv::getInterfaceVarABIAttrName();

            mlir::FunctionType function_type = func.getFunctionType();

            for (unsigned i = 0; i < function_type.getNumInputs(); ++i) {
                mlir::Type arg_type = function_type.getInput(i);
                std::optional<mlir::spirv::StorageClass> storage_class = std::nullopt;

                if (!mlir::isa<mlir::spirv::PointerType>(arg_type)) {
                    storage_class = mlir::spirv::StorageClass::StorageBuffer;
                }

                func.setArgAttr(
                    i,
                    arg_abi_attr_name,
                    mlir::spirv::getInterfaceVarABIAttr(
                        /*descriptorSet=*/0,
                        /*binding=*/i,
                        storage_class,
                        context));
            }
        }
    }

    void replace_returns(mlir::ModuleOp root_module) {
        llvm::SmallVector<mlir::Operation *> returns;

        root_module.walk([&](mlir::Operation *op) {
            if (is_op(op, "cuda_tile.return")) {
                returns.push_back(op);
            }
        });

        for (mlir::Operation *ret : returns) {
            mlir::OpBuilder builder(ret);

            mlir::spirv::ReturnOp::create(builder, ret->getLoc());

            ret->erase();
        }
    }
};

} // namespace

std::unique_ptr<mlir::Pass> create_cuda_tile_module_to_spirv_pass() {
    return std::make_unique<CudaTileModuleToSpirvPass>();
}