# CUDA Tile to SPIR-V translator

Small MLIR-based translator that loads CUDA Tile IR and lowers it to a SPIR-V module.

## What it does

The tool accepts a CUDA Tile input file in either textual MLIR form or Tile IR bytecode form:

```bash
./sp_tiles_translate input.{mlir,tilebc}
```

It then:

1. Loads the CUDA Tile module.
2. Verifies the input IR.
3. Lowers the CUDA Tile module shell to a SPIR-V module shell.
4. Lowers the supported CUDA Tile operations to SPIR-V operations.
5. Writes the generated SPIR-V binary to:

```bash
./output.spv
```

## Currently supported lowering

The current lowering handles the basic vector-add style path:

- `cuda_tile.module` to `spirv.module`
- `cuda_tile.entry` to `spirv.func`
- `cuda_tile.get_tile_block_id`
- `cuda_tile.load_view_tko`
- `cuda_tile.addf`
- `cuda_tile.store_view_tko`
- `cuda_tile.return`

The generated SPIR-V uses a fixed local size of:

```text
16 x 1 x 1
```

## Example

```bash
./sp_tiles_translate ../shaders/vector_add.tilebc
```

Expected result:

```text
loaded CUDA Tile input successfully
converted CUDA Tile module shell to SPIR-V module shell
wrote SPIR-V binary to ./output.spv
```

## Notes

This is an experimental lowering pass focused on CUDA Tile kernels that operate over `f32` storage buffers.