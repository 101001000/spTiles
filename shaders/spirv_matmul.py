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
        tensor_2d_descriptor,
    )
except ImportError:
    from spirv_runtime import (
        LOCAL_SIZE_X,
        ceil_div,
        default_spirv_path,
        run_spirv_kernel,
        tensor_2d_descriptor,
    )


MATMUL_TILE_M = 32
MATMUL_TILE_N = 32


def _torch_to_f32_numpy(tensor: torch.Tensor, name: str) -> np.ndarray:
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be a 2D tensor.")
    if tensor.dtype != torch.float32:
        raise ValueError(
            f"{name} must be torch.float32; the current SPIR-V lowering is f32-only."
        )
    return tensor.detach().cpu().contiguous().numpy()


def _matmul_dispatch(m: int, n: int) -> tuple[int, int, int]:
    tile_size = MATMUL_TILE_M * MATMUL_TILE_N
    output_tile_count = ceil_div(m, MATMUL_TILE_M) * ceil_div(n, MATMUL_TILE_N)
    invocation_count = output_tile_count * tile_size
    return (ceil_div(invocation_count, LOCAL_SIZE_X), 1, 1)


def spirv_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    spirv_path: str | Path | None = None,
    entry_name: str = "matmul_fp32",
) -> torch.Tensor:
    """
    Execute a generated SPIR-V matmul kernel directly from Python with Kompute.

    The cuTile Python ABI passes each 2D tensor as five storage buffers:
    data, dim0, dim1, stride0, stride1. The current lowering is f32-only.
    """
    if A.shape[1] != B.shape[0]:
        raise ValueError(
            f"Incompatible matrices: K dimension of A ({A.shape[1]}) must match "
            f"K dimension of B ({B.shape[0]})"
        )
    if A.device != B.device:
        raise ValueError("Input tensors must be on the same torch device.")

    if spirv_path is None:
        spirv_path = default_spirv_path()

    original_device = A.device
    a = _torch_to_f32_numpy(A, "A")
    b = _torch_to_f32_numpy(B, "B")
    m, k_a = a.shape
    k_b, n = b.shape
    if k_a != k_b:
        raise ValueError("Input matrices have incompatible K dimensions.")

    c = np.zeros((m, n), dtype=np.float32)
    parameter_arrays = [
        *tensor_2d_descriptor(a),
        *tensor_2d_descriptor(b),
        *tensor_2d_descriptor(c),
    ]
    output_flat = run_spirv_kernel(
        spirv_path=spirv_path,
        entry_name=entry_name,
        parameter_arrays=parameter_arrays,
        output_indices=[10],
        dispatch=_matmul_dispatch(m, n),
    )[0]
    output = output_flat.reshape(m, n).copy()

    result = torch.from_numpy(output)
    if original_device.type != "cpu":
        result = result.to(original_device)
    return result


def cutile_matmul_spirv(
    A: torch.Tensor,
    B: torch.Tensor,
    persistent: bool = False,
    *,
    spirv_path: str | Path | None = None,
    entry_name: str = "matmul_fp32",
) -> torch.Tensor:
    if persistent:
        raise NotImplementedError(
            "The generated SPIR-V matmul path does not expose a persistent kernel."
        )
    return spirv_matmul(A, B, spirv_path=spirv_path, entry_name=entry_name)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the generated SPIR-V matmul kernel.")
    parser.add_argument("spirv_path", nargs="?", default=None)
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--entry", default="matmul_fp32")
    return parser.parse_args()


def _demo_values(count: int, modulus: int, offset: int, scale: float) -> torch.Tensor:
    values = [(i % modulus - offset) * scale for i in range(count)]
    return torch.tensor(values, dtype=torch.float32)


def main() -> None:
    args = _parse_args()
    A = _demo_values(args.m * args.k, 17, 8, 0.25).reshape(args.m, args.k)
    B = _demo_values(args.k * args.n, 13, 6, 0.5).reshape(args.k, args.n)
    C = spirv_matmul(A, B, spirv_path=args.spirv_path, entry_name=args.entry)
    torch.testing.assert_close(C, A @ B, rtol=1e-4, atol=1e-3)
    print(f"matmul OK: {args.m}x{args.k} @ {args.k}x{args.n}")


if __name__ == "__main__":
    main()
