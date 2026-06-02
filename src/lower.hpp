#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/IR/BuiltinDialect.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseSet.h"

#include "tileops.hpp"
#include "common.hpp"

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
        registry.insert<
            mlir::cuda_tile::CudaTileDialect,
            mlir::spirv::SPIRVDialect>();
    }

    void runOnOperation() override {
        mlir::ModuleOp root_module = getOperation();

        replace_modules(root_module);
        replace_entries(root_module);

        collect_tokens(root_module);
        collect_assumes(root_module);
        collect_tensor_views(root_module);
        collect_partition_views(root_module);

        if (mlir::failed(apply_spmd_value_conversion(root_module))) {
            signalPassFailure();
            return;
        }

        erase_lowered_cuda_tile_ops(root_module);

        attach_spirv_abi_attrs(root_module);
    }

private:

    uint32_t local_size_x = 16;

    llvm::DenseSet<mlir::Value> tokens;
    llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Attribute>> assumptions;
    llvm::DenseMap<mlir::Value, EntryArrayInfo> entry_arrays;
    llvm::DenseMap<mlir::Value, TensorViewInfo> tensor_views;
    llvm::DenseMap<mlir::Value, PartitionViewInfo> partition_views;
    llvm::DenseMap<mlir::Value, mlir::Value> forwarded_values;
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

    mlir::LogicalResult apply_spmd_value_conversion(mlir::ModuleOp root_module) {
        mlir::MLIRContext &context = getContext();

        mlir::ConversionTarget target(context);
        mlir::RewritePatternSet patterns(&context);

        target.addLegalDialect<mlir::spirv::SPIRVDialect>();
        target.addLegalDialect<mlir::BuiltinDialect>();

        target.addIllegalDialect<mlir::cuda_tile::CudaTileDialect>();

        CudaTileSpmdLoweringContext lowering_context(
            local_size_x,
            entry_arrays,
            tensor_views,
            partition_views,
            forwarded_values);

        patterns.add<ConstantOpConversion>(&context);
        patterns.add<GetTileBlockIdOpConversion>(&context);
        patterns.add<GetIndexSpaceShapeOpConversion>(&context, lowering_context);
        patterns.add<LoadViewTkoOpConversion>(&context, lowering_context);
        patterns.add<StoreViewTkoOpConversion>(&context, lowering_context);
        patterns.add<ReshapeOpConversion>(&context);
        patterns.add<AddFOpConversion>(&context);
        patterns.add<AddIOpConversion>(&context);
        patterns.add<SubIOpConversion>(&context);
        patterns.add<MulIOpConversion>(&context);
        patterns.add<DivIOpConversion>(&context);
        patterns.add<MinIOpConversion>(&context);
        patterns.add<RemIOpConversion>(&context);
        patterns.add<CmpIOpConversion>(&context);
        patterns.add<AndIOpConversion>(&context);
        patterns.add<XOrIOpConversion>(&context);
        patterns.add<SelectOpConversion>(&context);
        patterns.add<AssumeOpConversion>(&context);
        patterns.add<ReturnOpConversion>(&context);
        patterns.add<ForOpConversion>(&context, lowering_context);
        patterns.add<ContinueOpConversion>(&context, lowering_context);
        
        patterns.add<MakeTokenOpConversion>(&context);
        patterns.add<MakeTensorViewConversion>(&context);
        patterns.add<MakePartitionViewConversion>(&context);

        return mlir::applyPartialConversion(
            root_module,
            target,
            std::move(patterns));
    }

    void add_forwarded_value(mlir::Value result, mlir::Value source) {
        forwarded_values[result] = get_forwarded_value(source);
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
};

} // namespace

std::unique_ptr<mlir::Pass> create_cuda_tile_module_to_spirv_pass() {
    return std::make_unique<CudaTileModuleToSpirvPass>();
}
