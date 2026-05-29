import os

import cuda.tile as ct
import cuda.tile.compilation as ct_compilation


@ct.kernel
def vector_add(a, b, c, tile_m: ct.Constant[int], tile_n: ct.Constant[int]):
    # Get the 1D pid.
    pid = ct.bid(0)

    flat_tile_size = tile_m * tile_n

    # Load input tiles as 1D contiguous chunks.
    a_flat = ct.load(a, index=(pid,), shape=(flat_tile_size,))
    b_flat = ct.load(b, index=(pid,), shape=(flat_tile_size,))

    # Work with the tiles as 2D values internally.
    a_tile = ct.reshape(a_flat, (tile_m, tile_n))
    b_tile = ct.reshape(b_flat, (tile_m, tile_n))

    result = a_tile + b_tile

    # Store back as a 1D contiguous chunk.
    ct.store(c, index=(pid,), tile=ct.reshape(result, (flat_tile_size,)))


def compile_only():
    output_file = "vector_add2D.tilebc"
    tile_m = 16
    tile_n = 32
    tile_size = tile_m * tile_n

    array_constraint = ct_compilation.ArrayConstraint(
        ct.float32,
        1,
        index_dtype=ct.int32,
        stride_lower_bound_incl=0,
        alias_groups=[],
        may_alias_internally=False,
        stride_constant=(1,),
        shape_divisible_by=tile_size,
        base_addr_divisible_by=1,
    )

    signature = ct_compilation.KernelSignature(
        parameters=[
            array_constraint,
            array_constraint,
            array_constraint,
            tile_m,
            tile_n,
        ],
        calling_convention=ct_compilation.CallingConvention.cutile_python_v1(),
        symbol="vector_add",
    )

    ct_compilation.export_kernel(
        vector_add,
        signatures=[signature],
        output_file=output_file,
        gpu_code="sm_100",
        output_format="tileir_bytecode",
        bytecode_version="13.1",
    )

    print(f"wrote {output_file}")


if __name__ == "__main__":
    compile_only()