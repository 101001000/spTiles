from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, Iterable

import kp
import numpy as np


LOCAL_SIZE_X = 1024


def check_kompute_api() -> None:
    missing = [name for name in ("Manager", "OpAlgoDispatch") if not hasattr(kp, name)]
    if missing:
        raise RuntimeError(
            "This Kompute Python package is too old for the cuTile SPIR-V ABI. "
            f"Missing: {', '.join(missing)}. Install a recent kp package from "
            "the Kompute repository."
        )


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_spirv_path() -> Path:
    build_output = repo_root() / "build" / "output.spv"
    if build_output.exists():
        return build_output
    return repo_root() / "output.spv"


def ceil_div(lhs: int, rhs: int) -> int:
    return (lhs + rhs - 1) // rhs


def read_spirv(path: str | Path, entry_name: str = "main") -> bytes:
    data = Path(path).read_bytes()
    if not data or len(data) % 4 != 0:
        raise ValueError(f"Invalid SPIR-V file size for {path}")
    words = list(struct.unpack(f"<{len(data) // 4}I", data))
    if words[0] != 0x07230203:
        raise ValueError(f"Invalid SPIR-V magic number for {path}")
    patched = rename_entry_point_for_kompute(words, entry_name)
    return struct.pack(f"<{len(patched)}I", *patched)


def rename_entry_point_for_kompute(words: list[int], entry_name: str) -> list[int]:
    # Kompute's high-level pipeline builder uses pName="main". cuTile exports
    # symbols such as "matmul_fp32", so adjust only the OpEntryPoint string.
    if entry_name == "main":
        return words

    OP_ENTRY_POINT = 15
    index = 5
    found_requested_entry = False
    found_main_entry = False
    patched = words.copy()

    while index < len(patched):
        instruction = patched[index]
        word_count = instruction >> 16
        opcode = instruction & 0xFFFF
        if word_count == 0:
            raise ValueError("Invalid SPIR-V instruction with zero word count.")

        if opcode == OP_ENTRY_POINT and word_count >= 4:
            name_word_start = index + 3
            max_name_words = word_count - 3
            name_bytes = b"".join(
                word.to_bytes(4, "little")
                for word in patched[name_word_start : name_word_start + max_name_words]
            )
            nul = name_bytes.find(b"\0")
            if nul == -1:
                raise ValueError("Invalid SPIR-V OpEntryPoint without a null-terminated name.")

            current_name = name_bytes[:nul].decode("utf-8")
            name_word_count = (nul + 4) // 4
            if current_name == "main":
                found_main_entry = True
            elif current_name == entry_name:
                found_requested_entry = True
                replacement = b"main\0".ljust(name_word_count * 4, b"\0")
                patched[name_word_start : name_word_start + name_word_count] = struct.unpack(
                    f"<{name_word_count}I", replacement
                )

        index += word_count

    if found_main_entry or found_requested_entry:
        return patched
    raise ValueError(f"Entry point {entry_name!r} was not found in the SPIR-V module.")


def make_tensor(manager: Any, data: np.ndarray) -> Any:
    if not hasattr(manager, "tensor_t"):
        raise RuntimeError(
            "Installed Kompute does not expose Manager.tensor_t. A newer Kompute "
            "Python build is required because this runner binds float32 and int32 buffers."
        )
    return manager.tensor_t(np.ascontiguousarray(data))


def tensor_1d_descriptor(tensor: np.ndarray) -> list[np.ndarray]:
    data = np.ascontiguousarray(tensor, dtype=np.float32).reshape(-1)
    stride = data.strides[0] // data.itemsize
    return [
        data,
        np.array([data.shape[0]], dtype=np.int32),
        np.array([stride], dtype=np.int32),
    ]


def tensor_2d_descriptor(tensor: np.ndarray) -> list[np.ndarray]:
    data = np.ascontiguousarray(tensor, dtype=np.float32)
    dim0, dim1 = data.shape
    stride0 = data.strides[0] // data.itemsize
    stride1 = data.strides[1] // data.itemsize
    return [
        data.reshape(-1),
        np.array([dim0], dtype=np.int32),
        np.array([dim1], dtype=np.int32),
        np.array([stride0], dtype=np.int32),
        np.array([stride1], dtype=np.int32),
    ]


def _sync_device_op(params: list[Any]) -> Any:
    op_type = getattr(kp, "OpTensorSyncDevice", None) or getattr(kp, "OpSyncDevice", None)
    if op_type is None:
        raise RuntimeError("Installed Kompute does not expose a device sync operation.")
    return op_type(params)


def _sync_local_op(params: list[Any]) -> Any:
    op_type = getattr(kp, "OpTensorSyncLocal", None) or getattr(kp, "OpSyncLocal", None)
    if op_type is None:
        raise RuntimeError("Installed Kompute does not expose a local sync operation.")
    return op_type(params)


def _read_output_tensor(output_tensor: Any) -> np.ndarray:
    if hasattr(output_tensor, "vector"):
        return np.array(output_tensor.vector(), dtype=np.float32, copy=True)
    if hasattr(output_tensor, "data"):
        return np.array(output_tensor.data(), dtype=np.float32, copy=True)
    raise RuntimeError("Unable to read data back from the Kompute output tensor.")


def run_spirv_kernel(
    *,
    spirv_path: str | Path,
    entry_name: str,
    parameter_arrays: Iterable[np.ndarray],
    output_indices: Iterable[int],
    dispatch: tuple[int, int, int],
) -> list[np.ndarray]:
    check_kompute_api()
    manager = kp.Manager()
    params = [make_tensor(manager, array) for array in parameter_arrays]
    output_tensors = [params[index] for index in output_indices]

    algorithm = manager.algorithm(params, read_spirv(spirv_path, entry_name), dispatch)
    sequence = manager.sequence()
    (
        sequence.record(_sync_device_op(params))
        .record(kp.OpAlgoDispatch(algorithm))
        .record(_sync_local_op(output_tensors))
        .eval()
    )
    return [_read_output_tensor(output_tensor) for output_tensor in output_tensors]
