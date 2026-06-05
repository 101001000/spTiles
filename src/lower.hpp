#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include "common.hpp"
#include "tileops.hpp"

#include <optional>

namespace {

static bool is_cuda_tile_op(mlir::Operation *op) { return op && op->getName().getDialectNamespace() == "cuda_tile"; }

static void print_value_chain(mlir::Value value, unsigned depth = 0) {
	for (unsigned i = 0; i < depth; ++i)
		llvm::errs() << "  ";

	llvm::errs() << "- value type: ";
	value.getType().print(llvm::errs());
	llvm::errs() << "\n";

	if (auto block_arg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
		for (unsigned i = 0; i < depth; ++i)
			llvm::errs() << "  ";

		llvm::errs() << "  block argument #" << block_arg.getArgNumber() << "\n";
		return;
	}

	mlir::Operation *def = value.getDefiningOp();
	if (!def)
		return;

	for (unsigned i = 0; i < depth; ++i)
		llvm::errs() << "  ";

	llvm::errs() << "  produced by: " << def->getName().getStringRef() << "\n";

	if (!is_cuda_tile_op(def))
		return;

	if (depth >= 4)
		return;

	for (unsigned i = 0; i < def->getNumOperands(); ++i) {
		for (unsigned j = 0; j < depth; ++j)
			llvm::errs() << "  ";

		llvm::errs() << "  operand " << i << ":\n";
		print_value_chain(def->getOperand(i), depth + 2);
	}
}

static void debug_remaining_cuda_tile_ops(mlir::ModuleOp root_module) {
	root_module.walk([&](mlir::Operation *op) {
		if (!is_cuda_tile_op(op))
			return;

		if (op->hasTrait<mlir::OpTrait::IsTerminator>())
			return;

		llvm::errs() << "\nremaining cuda_tile op: " << op->getName().getStringRef() << "\n";
		llvm::errs() << "op: ";
		op->print(llvm::errs());
		llvm::errs() << "\n";

		for (unsigned i = 0; i < op->getNumOperands(); ++i) {
			llvm::errs() << "operand " << i << ":\n";
			print_value_chain(op->getOperand(i), 1);
		}
	});
}

struct CudaTileModuleToSpirvPass : public mlir::PassWrapper<CudaTileModuleToSpirvPass, mlir::OperationPass<mlir::ModuleOp>> {
	MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CudaTileModuleToSpirvPass)

	llvm::StringRef getArgument() const override { return "cuda-tile-module-to-spirv-module"; }

	llvm::StringRef getDescription() const override { return "Lower CUDA Tile module shell to SPIR-V module shell."; }

	static llvm::SmallVector<IndexValue> get_index_values(llvm::ArrayRef<int64_t> static_values, mlir::ValueRange dynamic_values) {
		llvm::SmallVector<IndexValue> values;
		unsigned dynamic_index = 0;

		for (int64_t value : static_values) {
			if (mlir::ShapedType::isDynamic(value)) {
				values.push_back(IndexValue::from_dynamic(dynamic_values[dynamic_index++]));
			} else {
				values.push_back(IndexValue::from_static(value));
			}
		}

		return values;
	}

	void collect_tensor_views(mlir::ModuleOp root_module) {
		root_module.walk([&](mlir::cuda_tile::MakeTensorViewOp op) {
			auto tensor_view_type = op.getResult().getType();

			TensorViewInfo info;
			info.base_ptr = op.getBase();
			info.shape = get_index_values(tensor_view_type.getShape(), op.getDynamicShape());
			info.strides = get_index_values(tensor_view_type.getStrides(), op.getDynamicStrides());

			lowering_context.tensor_views[op.getResult()] = std::move(info);
		});
	}

	void collect_partition_views(mlir::ModuleOp root_module) {
		root_module.walk([&](mlir::cuda_tile::MakePartitionViewOp op) {
			auto partition_view_type = op.getResult().getType();

			PartitionViewInfo info;
			info.tensor_view = op.getTensorView();
			info.tile_shape = llvm::SmallVector<int32_t>(partition_view_type.getTileShape().asArrayRef());

			if (auto padding_value = partition_view_type.getPaddingValue()) {
				info.padding_value = padding_value.getValue();
			}

			lowering_context.partition_views[op.getResult()] = std::move(info);
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

			auto spirv_module = mlir::spirv::ModuleOp::create(builder, module->getLoc(), mlir::spirv::AddressingModel::Logical, mlir::spirv::MemoryModel::GLSL450);

			auto vce_triple = mlir::spirv::VerCapExtAttr::get(mlir::spirv::Version::V_1_0, llvm::ArrayRef<mlir::spirv::Capability>{mlir::spirv::Capability::Shader}, llvm::ArrayRef<mlir::spirv::Extension>{mlir::spirv::Extension::SPV_KHR_storage_buffer_storage_class}, module->getContext());

			spirv_module->setAttr(mlir::spirv::ModuleOp::getVCETripleAttrName(), vce_triple);

			spirv_module->setAttr(mlir::spirv::getTargetEnvAttrName(), mlir::spirv::getDefaultTargetEnv(module->getContext()));

			if (auto sym_name = module->getAttrOfType<mlir::StringAttr>(mlir::SymbolTable::getSymbolAttrName())) {
				spirv_module->setAttr(mlir::SymbolTable::getSymbolAttrName(), sym_name);
			}

			// Move the CUDA Tile module body into the SPIR-V module shell.
			spirv_module.getRegion().takeBody(module->getRegion(0));

			module->erase();
		}
	}

	std::string type_to_string(mlir::Type type) {
		std::string text;
		llvm::raw_string_ostream stream(text);
		type.print(stream);
		stream.flush();
		return text;
	}

	mlir::Type get_storage_buffer_f32_runtime_array_ptr_type(mlir::OpBuilder &builder) {
		mlir::Type f32_type = builder.getF32Type();

		auto runtime_array_type = mlir::spirv::RuntimeArrayType::get(f32_type,
		                                                             /*stride=*/4);

		llvm::SmallVector<mlir::Type, 1> member_types;
		member_types.push_back(runtime_array_type);

		auto struct_type = mlir::spirv::StructType::get(member_types,
		                                                /*offsetInfo=*/0);

		return mlir::spirv::PointerType::get(struct_type, mlir::spirv::StorageClass::StorageBuffer);
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

	mlir::FunctionType convert_entry_function_type(mlir::FunctionType function_type, mlir::OpBuilder &builder) {
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

	void replace_entries(mlir::ModuleOp root_module) {
		llvm::SmallVector<mlir::Operation *> entries;

		root_module.walk([&](mlir::Operation *op) {
			if (is_op(op, "cuda_tile.entry")) {
				entries.push_back(op);
			}
		});

		for (mlir::Operation *entry : entries) {
			auto sym_name = entry->getAttrOfType<mlir::StringAttr>(mlir::SymbolTable::getSymbolAttrName());

			auto type_attr = entry->getAttrOfType<mlir::TypeAttr>("function_type");

			if (!sym_name || !type_attr) {
				entry->emitError("expected sym_name and function_type");
				signalPassFailure();
				return;
			}

			auto function_type = llvm::dyn_cast<mlir::FunctionType>(type_attr.getValue());

			if (!function_type) {
				entry->emitError("expected FunctionType");
				signalPassFailure();
				return;
			}

			mlir::OpBuilder builder(entry);

			mlir::FunctionType converted_function_type = convert_entry_function_type(function_type, builder);

			auto spirv_func = mlir::spirv::FuncOp::create(builder, entry->getLoc(), sym_name.getValue(), converted_function_type, mlir::spirv::FunctionControl::None);

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
	void attach_spirv_abi_attrs(mlir::ModuleOp root_module) {
		llvm::SmallVector<mlir::spirv::FuncOp> funcs;

		root_module.walk([&](mlir::spirv::FuncOp func) { funcs.push_back(func); });

		for (mlir::spirv::FuncOp func : funcs) {
			mlir::MLIRContext *context = func.getContext();

			llvm::SmallVector<int32_t, 3> workgroup_size;
			workgroup_size.push_back(1024);
			workgroup_size.push_back(1);
			workgroup_size.push_back(1);

			func->setAttr(mlir::spirv::getEntryPointABIAttrName(), mlir::spirv::getEntryPointABIAttr(context, workgroup_size));

			llvm::StringRef arg_abi_attr_name = mlir::spirv::getInterfaceVarABIAttrName();

			mlir::FunctionType function_type = func.getFunctionType();

			for (unsigned i = 0; i < function_type.getNumInputs(); ++i) {
				mlir::Type arg_type = function_type.getInput(i);
				std::optional<mlir::spirv::StorageClass> storage_class = std::nullopt;

				if (!mlir::isa<mlir::spirv::PointerType>(arg_type)) {
					storage_class = mlir::spirv::StorageClass::StorageBuffer;
				}

				func.setArgAttr(i, arg_abi_attr_name,
				                mlir::spirv::getInterfaceVarABIAttr(
				                    /*descriptorSet=*/0,
				                    /*binding=*/i, storage_class, context));
			}
		}
	}

	void getDependentDialects(mlir::DialectRegistry &registry) const override { registry.insert<mlir::cuda_tile::CudaTileDialect, mlir::spirv::SPIRVDialect>(); }

	void runOnOperation() override {
		mlir::ModuleOp root_module = getOperation();

		collect_tensor_views(root_module);
		collect_partition_views(root_module);

		replace_modules(root_module);
		replace_entries(root_module);

		mlir::MLIRContext &context = getContext();

		mlir::ConversionTarget target(context);
		mlir::RewritePatternSet patterns(&context);

		target.addLegalDialect<mlir::spirv::SPIRVDialect>();
		target.addLegalDialect<mlir::BuiltinDialect>();

		target.addIllegalOp<mlir::cuda_tile::ReturnOp>();
		target.addIllegalOp<mlir::cuda_tile::StoreViewTkoOp>();

		target.markUnknownOpDynamicallyLegal([](mlir::Operation *) { return true; });

		TileMaterializer materializer(lowering_context);

		patterns.add<ReturnOpConversion>(&context);
		patterns.add<StoreViewTkoOpConversion>(&context, materializer);

		if (mlir::failed(mlir::applyPartialConversion(root_module, target, std::move(patterns)))) {
			debug_remaining_cuda_tile_ops(root_module);
			signalPassFailure();
			return;
		}

		erase_dead_cuda_tile_ops(root_module);
		attach_spirv_abi_attrs(root_module);
	}

  private:
	LoweringContext lowering_context;
};

} // namespace

std::unique_ptr<mlir::Pass> create_cuda_tile_module_to_spirv_pass() { return std::make_unique<CudaTileModuleToSpirvPass>(); }
