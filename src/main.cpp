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
    registry.insert<mlir::cuda_tile::CudaTileDialect>();

    mlir::MLIRContext context(registry);
    context.loadDialect<mlir::cuda_tile::CudaTileDialect>();

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

    return 0;
}