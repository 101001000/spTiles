#include "cuda_tile/Bytecode/Reader/BytecodeReader.h"
#include "cuda_tile/Dialect/CudaTile/IR/Dialect.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Target/SPIRV/Serialization.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FileSystem.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"

#include "lower.hpp"

std::unique_ptr<mlir::Pass> create_cuda_tile_module_to_spirv_pass();

static mlir::LogicalResult
save_spirv_binary(mlir::spirv::ModuleOp spirv_module,
                  llvm::StringRef output_path) {
    llvm::SmallVector<uint32_t, 0> binary;

    if (mlir::failed(mlir::spirv::serialize(spirv_module, binary))) {
        return mlir::failure();
    }

    std::error_code error_code;
    llvm::raw_fd_ostream output(output_path, error_code,
                                llvm::sys::fs::OF_None);

    if (error_code) {
        llvm::errs() << "error: could not open output file '" << output_path
                     << "': " << error_code.message() << "\n";
        return mlir::failure();
    }

    output.write(reinterpret_cast<const char *>(binary.data()),
                 binary.size() * sizeof(uint32_t));
    output.flush();

    return mlir::success();
}

static mlir::OwningOpRef<mlir::Operation *>
load_cuda_tile_mlir(llvm::MemoryBufferRef buffer_ref,
                    mlir::MLIRContext &context) {
    llvm::SourceMgr source_mgr;
    source_mgr.AddNewSourceBuffer(
        llvm::MemoryBuffer::getMemBufferCopy(buffer_ref.getBuffer(),
                                             buffer_ref.getBufferIdentifier()),
        llvm::SMLoc());

    mlir::SourceMgrDiagnosticHandler diagnostic_handler(source_mgr, &context);

    mlir::ParserConfig parser_config(&context);
    return mlir::parseSourceFile(source_mgr, parser_config);
}

static mlir::OwningOpRef<mlir::Operation *>
load_cuda_tile_tilebc(llvm::MemoryBufferRef buffer_ref,
                      mlir::MLIRContext &context) {
    if (!mlir::cuda_tile::isTileIRBytecode(buffer_ref)) {
        llvm::errs() << "error: input is not CUDA Tile bytecode\n";
        return {};
    }

    return mlir::cuda_tile::readBytecode(buffer_ref, context);
}

static mlir::OwningOpRef<mlir::Operation *>
load_cuda_tile_file(llvm::StringRef file_path, mlir::MLIRContext &context) {
    auto file_or_error = llvm::MemoryBuffer::getFileOrSTDIN(file_path);

    if (!file_or_error) {
        llvm::errs() << "error: could not open input file '" << file_path
                     << "': " << file_or_error.getError().message() << "\n";
        return {};
    }

    llvm::MemoryBufferRef buffer_ref = (*file_or_error)->getMemBufferRef();

    if (mlir::cuda_tile::isTileIRBytecode(buffer_ref)) {
        return load_cuda_tile_tilebc(buffer_ref, context);
    }

    return load_cuda_tile_mlir(buffer_ref, context);
}

int main(int argc, char **argv) {
    llvm::InitLLVM init_llvm(argc, argv);

    if (argc != 2) {
        llvm::errs() << "usage: " << argv[0] << " input.{mlir,tilebc}\n";
        return 1;
    }

    mlir::DialectRegistry registry;
    registry.insert<mlir::cuda_tile::CudaTileDialect,
                    mlir::spirv::SPIRVDialect>();

    mlir::MLIRContext context(registry);
    context.loadDialect<mlir::cuda_tile::CudaTileDialect,
                        mlir::spirv::SPIRVDialect>();

    mlir::OwningOpRef<mlir::Operation *> op =
        load_cuda_tile_file(argv[1], context);

    if (!op) {
        llvm::errs() << "error: failed to load CUDA Tile input\n";
        return 1;
    }

    if (mlir::failed(mlir::verify(*op))) {
        llvm::errs() << "error: CUDA Tile IR verification failed\n";
        return 1;
    }

    llvm::outs() << "loaded CUDA Tile input successfully\n";
    op->print(llvm::outs());
    llvm::outs() << "\n";

    mlir::OwningOpRef<mlir::ModuleOp> wrapper(
    mlir::ModuleOp::create((*op)->getLoc()));

    wrapper->getBody()->getOperations().push_back(op.release());

    mlir::PassManager pm(&context);

    // The intermediate IR is intentionally invalid as real SPIR-V.
    pm.enableVerifier(false);

    pm.addPass(create_cuda_tile_module_to_spirv_pass());

    pm.addNestedPass<mlir::spirv::ModuleOp>(mlir::spirv::createSPIRVLowerABIAttributesPass());

    if (mlir::failed(pm.run(*wrapper))) {
        llvm::errs() << "error: CUDA Tile to SPIR-V module pass failed\n";
        return 1;
    }

    llvm::outs() << "converted CUDA Tile module shell to SPIR-V module shell\n";
    wrapper->print(llvm::outs());
    llvm::outs() << "\n";

     mlir::spirv::ModuleOp spirv_module = nullptr;

    wrapper->walk([&](mlir::spirv::ModuleOp module) {
        spirv_module = module;
        return mlir::WalkResult::interrupt();
    });

    if (!spirv_module) {
        llvm::errs() << "error: no spirv.module found after lowering\n";
        return 1;
    }

    if (mlir::failed(mlir::verify(spirv_module))) {
        llvm::errs() << "error: generated SPIR-V module verification failed\n";
        return 1;
    }


    std::string output_path = "./output.spv";

    if (mlir::failed(save_spirv_binary(spirv_module, output_path))) {
        llvm::errs() << "error: failed to serialize SPIR-V binary\n";
        return 1;
    }

    llvm::outs() << "wrote SPIR-V binary to " << output_path << "\n";

    return 0;
}