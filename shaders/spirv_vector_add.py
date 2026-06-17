from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

try:
    from .spirv_runtime import (
        LOCAL_SIZE_X,
        ceil_div,
        default_spirv_path,
        run_spirv_kernel,
        tensor_1d_descriptor,
    )
except ImportError:
    from spirv_runtime import (
        LOCAL_SIZE_X,
        ceil_div,
        default_spirv_path,
        run_spirv_kernel,
        tensor_1d_descriptor,
    )


VECTOR_ADD_TILE_SIZE = 16


def _torch_to_f32_numpy(tensor: torch.Tensor, name: str) -> np.ndarray:
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be a 1D tensor.")
    if tensor.dtype != torch.float32:
        raise ValueError(
            f"{name} must be torch.float32; the current SPIR-V lowering is f32-only."
        )
    return tensor.detach().cpu().contiguous().numpy()


def _vector_add_dispatch(element_count: int) -> tuple[int, int, int]:
    return (ceil_div(element_count, LOCAL_SIZE_X), 1, 1)


def spirv_vector_add(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    spirv_path: str | Path | None = None,
    entry_name: str = "vector_add",
) -> torch.Tensor:
    """
    Execute a generated SPIR-V vector-add kernel directly from Python with Kompute.

    The cuTile Python ABI passes each 1D tensor as three storage buffers:
    data, size, stride. The exported vector_add kernel was compiled for f32.
    """
    if A.shape != B.shape:
        raise ValueError(f"Input tensors must have the same shape: {A.shape} != {B.shape}")
    if A.device != B.device:
        raise ValueError("Input tensors must be on the same torch device.")

    if spirv_path is None:
        spirv_path = default_spirv_path()

    original_device = A.device
    a = _torch_to_f32_numpy(A, "A")
    b = _torch_to_f32_numpy(B, "B")
    element_count = a.shape[0]
    if element_count % VECTOR_ADD_TILE_SIZE != 0:
        raise ValueError(
            f"vector_add expects a size divisible by {VECTOR_ADD_TILE_SIZE}; "
            f"got {element_count}."
        )

    c = np.zeros(element_count, dtype=np.float32)
    parameter_arrays = [
        *tensor_1d_descriptor(a),
        *tensor_1d_descriptor(b),
        *tensor_1d_descriptor(c),
    ]
    output = run_spirv_kernel(
        spirv_path=spirv_path,
        entry_name=entry_name,
        parameter_arrays=parameter_arrays,
        output_indices=[6],
        dispatch=_vector_add_dispatch(element_count),
    )[0][:element_count].copy()

    result = torch.from_numpy(output)
    if original_device.type != "cpu":
        result = result.to(original_device)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the generated SPIR-V vector_add kernel.")
    parser.add_argument("spirv_path", nargs="?", default=None)
    parser.add_argument("--size", type=int, default=2048)
    parser.add_argument("--entry", default="vector_add")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    A = torch.arange(args.size, dtype=torch.float32)
    B = torch.arange(1000, 1000 + args.size, dtype=torch.float32)
    C = spirv_vector_add(A, B, spirv_path=args.spirv_path, entry_name=args.entry)
    torch.testing.assert_close(C, A + B, rtol=1e-4, atol=1e-4)
    print(f"vector_add OK: {args.size} elements")


if __name__ == "__main__":
    main()
