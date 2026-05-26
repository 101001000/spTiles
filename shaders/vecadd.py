import os

import cuda.tile as ct
import cuda.tile.compilation as ct_compilation


@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    # Get the 1D pid.
    pid = ct.bid(0)

    # Load input tiles.
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    b_tile = ct.load(b, index=(pid,), shape=(tile_size,))

    # Perform elementwise addition.
    result = a_tile + b_tile

    # Store result.
    ct.store(c, index=(pid,), tile=result)


def compile_only():
    output_file = "vector_add.tilebc"
    tile_size = 16

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
            tile_size,
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