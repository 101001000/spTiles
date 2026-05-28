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
        collect_and_erase_assumes(root_module);
        collect_tensor_views(root_module);
        collect_partition_views(root_module);

        lower_cuda_tile_ops(root_module);

        erase_metadata_ops(root_module);

        erase_unused_spirv_func_args(root_module);
        attach_spirv_abi_attrs(root_module);

        replace_returns(root_module);
    }

private:

    uint32_t local_size_x = 1024;

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
        uint32_t tile_size;
    };

    llvm::DenseSet<mlir::Value> tokens;
    llvm::DenseMap<mlir::Value, mlir::Value> value_aliases;
    llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Attribute>> assumptions;
    llvm::DenseMap<mlir::Value, TensorViewInfo> tensor_views;
    llvm::DenseMap<mlir::Value, PartitionViewInfo> partition_views;

    static bool is_op(mlir::Operation *op, llvm::StringRef name) {
        return op->getName().getStringRef() == name;
    }

    mlir::Value canonical_value(mlir::Value value) {
        while (true) {
            auto it = value_aliases.find(value);
            if (it == value_aliases.end()) {
                return value;
            }

            value = it->second;
        }
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

    void lower_load_view_tko(mlir::Operation *op) {
        mlir::OpBuilder builder(op);
        mlir::Location loc = op->getLoc();

        mlir::Value pview = op->getOperand(0);
        mlir::Value block_id_x = canonical_value(op->getOperand(1));
        mlir::Value token = op->getOperand(2);

        auto it = partition_views.find(pview);
        if (it == partition_views.end()) {
            op->emitError("unknown partition view");
            return;
        }

        mlir::Value base = it->second.base;
        uint32_t tile_size = it->second.tile_size;

        mlir::Type i32_type = builder.getI32Type();

        mlir::Value c_tile_size = mlir::spirv::ConstantOp::create(
            builder,
            loc,
            i32_type,
            builder.getI32IntegerAttr(tile_size));

        mlir::Value c0 = mlir::spirv::ConstantOp::create(
            builder,
            loc,
            i32_type,
            builder.getI32IntegerAttr(0));

        mlir::Value lane = get_or_create_lane_id(op, builder);

        mlir::Value active = mlir::spirv::ULessThanOp::create(
            builder,
            loc,
            lane,
            c_tile_size);

        mlir::Value safe_lane = mlir::spirv::SelectOp::create(
            builder,
            loc,
            i32_type,
            active,
            lane,
            c0);

        mlir::Value tile_base = mlir::spirv::IMulOp::create(
            builder,
            loc,
            i32_type,
            block_id_x,
            c_tile_size);

        mlir::Value idx = mlir::spirv::IAddOp::create(
            builder,
            loc,
            i32_type,
            tile_base,
            safe_lane);

        mlir::Value ptr = create_storage_buffer_access(
            builder,
            loc,
            base,
            idx);

        mlir::Value loaded = mlir::spirv::LoadOp::create(
            builder,
            loc,
            ptr,
            nullptr,
            nullptr);

        op->getResult(0).replaceAllUsesWith(loaded);
        op->getResult(1).replaceAllUsesWith(token);
        op->erase();
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

        op->getResult(0).replaceAllUsesWith(block_id_x);
        op->getResult(1).replaceAllUsesWith(block_id_y);
        op->getResult(2).replaceAllUsesWith(block_id_z);

        op->erase();
    }


    void lower_store_view_tko(mlir::Operation *op) {
        if (op->getNumOperands() != 4) {
            op->emitError("expected store_view_tko with exactly 4 operands");
            return;
        }

        if (op->getNumResults() != 1) {
            op->emitError("expected store_view_tko to produce one token result");
            return;
        }

        mlir::OpBuilder builder(op);
        mlir::Location loc = op->getLoc();

        mlir::Value value = op->getOperand(0);
        mlir::Value pview = op->getOperand(1);
        mlir::Value block_id_x = canonical_value(op->getOperand(2));
        mlir::Value token = op->getOperand(3);

        auto it = partition_views.find(pview);
        if (it == partition_views.end()) {
            op->emitError("unknown partition view");
            return;
        }

        mlir::Value base = it->second.base;
        int64_t tile_size = it->second.tile_size;

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

        mlir::Type i32_type = builder.getI32Type();

        mlir::Value c_tile_size = mlir::spirv::ConstantOp::create(
            builder,
            loc,
            i32_type,
            builder.getI32IntegerAttr(static_cast<int32_t>(tile_size)));

        mlir::Value lane = get_or_create_lane_id(op, builder);

        mlir::Value active = mlir::spirv::ULessThanOp::create(
            builder,
            loc,
            lane,
            c_tile_size);

        mlir::Value tile_base = mlir::spirv::IMulOp::create(
            builder,
            loc,
            i32_type,
            block_id_x,
            c_tile_size);

        mlir::Value idx = mlir::spirv::IAddOp::create(
            builder,
            loc,
            i32_type,
            tile_base,
            lane);

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
            idx);

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

        op->getResult(0).replaceAllUsesWith(token);
        op->erase();
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

        mlir::Value lhs = op->getOperand(0);
        mlir::Value rhs = op->getOperand(1);

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

        op->getResult(0).replaceAllUsesWith(result);
        op->erase();
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
            if (!op->getBlock()) {
                continue;
            }

            callback(op);
        }
    }

    void lower_cuda_tile_ops(mlir::ModuleOp root_module) {
        lower_ops_by_name(root_module, "cuda_tile.get_tile_block_id",
                        [&](mlir::Operation *op) {
                            lower_get_tile_block_id(op);
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

    void erase_metadata_ops(mlir::ModuleOp root_module) {
        llvm::SmallVector<mlir::Operation *> ops;

        root_module.walk([&](mlir::Operation *op) {
            if (is_op(op, "cuda_tile.make_tensor_view") ||
                is_op(op, "cuda_tile.make_partition_view") ||
                is_op(op, "cuda_tile.make_token")) {
                ops.push_back(op);
            }
        });

        for (mlir::Operation *op : llvm::reverse(ops)) {
            if (!op->use_empty()) {
                op->emitError("cannot erase metadata op because it still has users");

                for (mlir::Operation *user : op->getUsers()) {
                    user->emitError("still uses metadata op that could not be erased");
                }

                continue;
            }

            op->erase();
        }
    }

    void collect_tensor_views(mlir::ModuleOp root_module) {
        root_module.walk([&](mlir::Operation *op) {
            if (!is_op(op, "cuda_tile.make_tensor_view")) {
                return;
            }

            mlir::Value result = op->getResult(0);

            TensorViewInfo info;
            info.base = canonical_value(op->getOperand(0));
            info.type = result.getType();
            info.source_op = op;

            for (mlir::Value operand : op->getOperands().drop_front()) {
                info.shape.push_back(canonical_value(operand));
            }

            tensor_views[result] = std::move(info);
        });
    }


   void collect_partition_views(mlir::ModuleOp root_module) {
        root_module.walk([&](mlir::Operation *op) {
            if (!is_op(op, "cuda_tile.make_partition_view")) {
                return;
            }

            mlir::Value tensor_view = op->getOperand(0);
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

            uint32_t tile_size = 1;
            for (uint32_t dim : partition_type.getTileShape().asArrayRef()) {
                tile_size *= dim;
            }
            info.tile_size = tile_size;

            partition_views[result] = info;
        });
    }

    void collect_and_erase_assumes(mlir::ModuleOp root_module) {
        llvm::SmallVector<mlir::Operation *> assume_ops;

        root_module.walk([&](mlir::Operation *op) {
            if (op->getName().getStringRef() == "cuda_tile.assume") {
                assume_ops.push_back(op);
            }
        });

        for (mlir::Operation *assume_op : assume_ops) {
            mlir::Value input = assume_op->getOperand(0);
            mlir::Value result = assume_op->getResult(0);
            mlir::Value base = canonical_value(input);

            if (auto predicate = assume_op->getAttr("predicate")) {
                assumptions[base].push_back(predicate);
            }

            value_aliases[result] = base;

            // Replace the refined value with the original base value.
            result.replaceAllUsesWith(base);

            assume_op->erase();
        }
    }


    void collect_tokens(mlir::ModuleOp root_module) {
        root_module.walk([&](mlir::Operation *op) {
            if (op->getName().getStringRef() != "cuda_tile.make_token") {
                return;
            }

            tokens.insert(op->getResult(0));
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