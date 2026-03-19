"""
orbench_io_py.py - Python utilities for ORBench v2 input.bin writing.

This module is used by tasks/*/gen_data.py to generate:
  - input.bin (header + TensorDesc + ParamEntry + raw tensor bytes)

Binary layout (little-endian):
  FileHeader (32 bytes)
  TensorDesc * num_tensors (64 bytes each)
  ParamEntry * num_params (48 bytes each)
  padding to 64B-aligned data_offset
  raw tensor bytes (tightly packed, each tensor has its own recorded offset)
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict, Union, Any, Optional


DTYPE_INT32 = 0
DTYPE_FLOAT32 = 1
DTYPE_FLOAT64 = 2


def _align64(x: int) -> int:
    return (x + 63) & ~63


def _dtype_to_code(dtype: str) -> int:
    d = dtype.lower()
    if d in ("int32", "i32"):
        return DTYPE_INT32
    if d in ("float32", "f32"):
        return DTYPE_FLOAT32
    if d in ("float64", "f64"):
        return DTYPE_FLOAT64
    raise ValueError(f"Unsupported dtype: {dtype}")


def _dtype_size(code: int) -> int:
    if code == DTYPE_INT32:
        return 4
    if code == DTYPE_FLOAT32:
        return 4
    if code == DTYPE_FLOAT64:
        return 8
    raise ValueError(f"Bad dtype code: {code}")


def _as_bytes(arr: Any, dtype_code: int) -> Tuple[bytes, int]:
    """
    Convert a 1D array-like to raw bytes and element count.
    Supports: Python lists/tuples, bytes/bytearray, numpy arrays (if installed), array('i'/'f'/'d').
    """
    # bytes already
    if isinstance(arr, (bytes, bytearray, memoryview)):
        b = bytes(arr)
        elem_size = _dtype_size(dtype_code)
        if len(b) % elem_size != 0:
            raise ValueError("Raw bytes length not divisible by element size")
        return b, len(b) // elem_size

    # numpy
    try:
        import numpy as np  # type: ignore

        if isinstance(arr, np.ndarray):
            if arr.ndim != 1:
                raise ValueError("Only 1D tensors are supported")
            # Ensure dtype matches
            if dtype_code == DTYPE_INT32:
                a = arr.astype(np.int32, copy=False)
            elif dtype_code == DTYPE_FLOAT32:
                a = arr.astype(np.float32, copy=False)
            elif dtype_code == DTYPE_FLOAT64:
                a = arr.astype(np.float64, copy=False)
            else:
                raise ValueError("Bad dtype code")
            return a.tobytes(order="C"), int(a.size)
    except Exception:
        pass

    # array module
    try:
        import array as py_array

        if isinstance(arr, py_array.array):
            b = arr.tobytes()
            return b, len(arr)
    except Exception:
        pass

    # Python list/tuple
    if isinstance(arr, (list, tuple)):
        count = len(arr)
        if dtype_code == DTYPE_INT32:
            return struct.pack("<" + "i" * count, *arr), count
        if dtype_code == DTYPE_FLOAT32:
            return struct.pack("<" + "f" * count, *arr), count
        if dtype_code == DTYPE_FLOAT64:
            return struct.pack("<" + "d" * count, *arr), count

    raise TypeError(f"Unsupported tensor data type: {type(arr)}")


def write_input_bin(
    out_path: str,
    tensors: List[Tuple[str, str, Any]],
    params: Dict[str, int],
    version: int = 1,
) -> None:
    """
    Write ORBench v2 input.bin.

    Args:
        out_path: output file path
        tensors: list of (name, dtype_str, data_1d)
        params: dict of key -> int64 value
        version: file format version (default 1)
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Prepare tensor raw bytes and descriptors
    tensor_descs = []
    tensor_blobs = []

    for name, dtype_str, data in tensors:
        if len(name) > 31:
            raise ValueError(f"Tensor name too long (max 31): {name}")
        dtype_code = _dtype_to_code(dtype_str)
        blob, count = _as_bytes(data, dtype_code)
        size_bytes = len(blob)
        tensor_descs.append(
            {
                "name": name,
                "dtype": dtype_code,
                "count": int(count),
                "size_bytes": int(size_bytes),
                "offset": 0,  # fill later
            }
        )
        tensor_blobs.append(blob)

    # Prepare params
    param_items = list(params.items())
    for k, v in param_items:
        if len(k) > 31:
            raise ValueError(f"Param key too long (max 31): {k}")
        if not isinstance(v, int):
            raise ValueError(f"Param value must be int: {k}={v}")

    num_tensors = len(tensor_descs)
    num_params = len(param_items)

    header_size = 32
    tensor_desc_size = 64 * num_tensors
    param_size = 48 * num_params
    meta_size = header_size + tensor_desc_size + param_size
    data_offset = _align64(meta_size)

    # Compute offsets for each tensor (tightly packed in data region)
    cur = data_offset
    for i, td in enumerate(tensor_descs):
        td["offset"] = cur
        cur += td["size_bytes"]

    # Write file
    with open(out_path, "wb") as f:
        # FileHeader
        magic = b"ORBENCH\x00"
        f.write(
            struct.pack(
                "<8siiiiq",
                magic,
                int(version),
                int(num_tensors),
                int(num_params),
                int(data_offset),
                0,
            )
        )

        # TensorDesc array
        for td in tensor_descs:
            name_bytes = td["name"].encode("utf-8")[:31] + b"\x00"
            name_bytes = name_bytes.ljust(32, b"\x00")
            f.write(
                struct.pack(
                    "<32siiqqq",
                    name_bytes,
                    int(td["dtype"]),
                    0,
                    int(td["count"]),
                    int(td["offset"]),
                    int(td["size_bytes"]),
                )
            )

        # ParamEntry array
        for k, v in param_items:
            key_bytes = k.encode("utf-8")[:31] + b"\x00"
            key_bytes = key_bytes.ljust(32, b"\x00")
            f.write(struct.pack("<32sq", key_bytes, int(v)))

        # Padding to data_offset
        cur_pos = f.tell()
        if cur_pos > data_offset:
            raise RuntimeError("Internal error: meta_size exceeded data_offset")
        f.write(b"\x00" * (data_offset - cur_pos))

        # Raw tensor bytes
        for blob in tensor_blobs:
            f.write(blob)








