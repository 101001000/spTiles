# CUDA Tile to SPIR-V translator

Small MLIR-based translator that loads CUDA Tile IR and lowers a supported subset of it to a SPIR-V compute module.

## What it does

The tool accepts a CUDA Tile input file in either textual MLIR form or Tile IR bytecode form:

```bash
./sp_tiles_translate input.{mlir,tilebc}
```

It then:

1. Loads the CUDA Tile module.
2. Verifies the input IR.
3. Lowers the CUDA Tile module shell to a SPIR-V module shell.
4. Lowers a supported subset of CUDA Tile operations to SPIR-V.
5. Writes the generated SPIR-V binary to:

```bash
./output.spv
```

The lowering currently focuses on a small subset of CUDA Tile programs, including elementwise operations such as add, matrix multiply accumulation through `mmaf`, and simple control-flow constructs such as `for`.

## Execution model

The generated SPIR-V kernel is parallelized at the tile-element level.

Each invocation computes one element of a logical tile. A fixed workgroup size is used, and the translator maps the global invocation id to:

* the logical tile being processed,
* the local element inside that tile,
* and the corresponding tensor element in storage-buffer memory.

For tiled matrix multiplication, this means each output tile is decomposed into independent scalar element computations, while the accumulation across the reduction dimension is lowered to SPIR-V control flow.

## Examples

### Vector add

Translate the CUDA Tile bytecode:

```bash
./sp_tiles_translate ../shaders/vector_add.tilebc
```

Run the generated SPIR-V binary:

```bash
./sp_tiles_translate_test vector_add output.spv 2048
```

### Matrix multiplication

Translate the CUDA Tile bytecode:

```bash
./sp_tiles_translate ../shaders/matmul.tilebc
```

Run the generated SPIR-V binary:

```bash
./sp_tiles_translate_test matmul output.spv 256 256 256
```

The matrix multiplication test arguments are:

```text
M N K
```

so the example above runs:

```text
C[M x N] = A[M x K] @ B[K x N]
```

## Expected output

A successful translation prints messages similar to:

```text
loaded CUDA Tile input successfully
converted CUDA Tile module shell to SPIR-V module shell
wrote SPIR-V binary to ./output.spv
```

A successful test run prints a confirmation such as:

```text
matmul OK: 256x256 @ 256x256
```

or:

```text
vector_add OK: 2048 elements
```

## Notes

This is an experimental lowering pass for CUDA Tile kernels targeting SPIR-V compute shaders. The implementation is intentionally limited and currently supports only a small subset of CUDA Tile operations and types, with the main focus on `f32` storage-buffer based kernels.
