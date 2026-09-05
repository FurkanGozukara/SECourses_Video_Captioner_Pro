"""Bounded, unmapped safetensors reads for Windows checkpoint streaming."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path

import torch


_DTYPES = {
    "BOOL": torch.bool, "U8": torch.uint8, "I8": torch.int8,
    "I16": torch.int16, "U16": torch.uint16,
    "F16": torch.float16, "BF16": torch.bfloat16,
    "I32": torch.int32, "U32": torch.uint32, "F32": torch.float32,
    "I64": torch.int64, "U64": torch.uint64, "F64": torch.float64,
    "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2,
}


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate safetensors header key: {key}")
        result[key] = value
    return result


@dataclass(frozen=True)
class _TensorInfo:
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int

    def get_shape(self) -> list[int]:
        return list(self.shape)

    def get_dtype(self) -> str:
        return self.dtype


class StreamCheckpoint:
    """Read one CPU tensor at a time without mapping the entire checkpoint.

    Even safetensors 0.8's pread backend creates a temporary copy-on-write map
    to inspect its header. Windows charges that map against the commit limit.
    This reader validates the header using ordinary file reads, then reads each
    tensor directly into its owned Torch allocation. It implements only the
    metadata/full-tensor operations needed by our single-file loader.
    """

    def __init__(self, path: Path):
        self._file = path.open("rb", buffering=0)
        try:
            size = os.fstat(self._file.fileno()).st_size
            prefix = self._file.read(8)
            if len(prefix) != 8:
                raise ValueError("Missing safetensors header length")
            length = int.from_bytes(prefix, "little")
            if not 2 <= length <= min(100_000_000, size - 8):
                raise ValueError("Invalid safetensors header length")
            raw = self._file.read(length)
            if len(raw) != length or not raw.startswith(b"{"):
                raise ValueError("Invalid or truncated safetensors JSON header")
            header = json.loads(raw, object_pairs_hook=_unique_object)
            if not isinstance(header, dict):
                raise ValueError("Safetensors header must be an object")
            self._offset = 8 + length
            self._metadata = header.pop("__metadata__", {})
            if not isinstance(self._metadata, dict) or any(
                not isinstance(value, str) for value in self._metadata.values()
            ):
                raise ValueError("Safetensors metadata must contain strings")
            self._tensors: dict[str, _TensorInfo] = {}
            for name, entry in header.items():
                if not isinstance(entry, dict):
                    raise ValueError(f"Invalid tensor entry: {name}")
                dtype, shape, offsets = (
                    entry.get("dtype"), entry.get("shape"), entry.get("data_offsets")
                )
                if not isinstance(dtype, str) or dtype not in _DTYPES:
                    raise ValueError(f"Unsupported safetensors dtype for {name}: {dtype}")
                if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
                    raise ValueError(f"Invalid tensor shape: {name}")
                if not isinstance(offsets, list) or len(offsets) != 2 or any(
                    type(n) is not int or n < 0 for n in offsets
                ):
                    raise ValueError(f"Invalid tensor offsets: {name}")
                start, end = offsets
                expected = math.prod(shape) * _DTYPES[dtype].itemsize
                if end - start != expected or end > size - self._offset:
                    raise ValueError(f"Tensor size or file bounds mismatch: {name}")
                self._tensors[name] = _TensorInfo(dtype, tuple(shape), start, end)
            cursor = 0
            for info in sorted(self._tensors.values(), key=lambda item: (item.start, item.end)):
                if info.start != cursor:
                    raise ValueError("Safetensors data has gaps or overlapping tensors")
                cursor = info.end
            if cursor != size - self._offset:
                raise ValueError("Safetensors file has incomplete or trailing tensor data")
        except BaseException:
            self._file.close()
            raise

    def __enter__(self) -> StreamCheckpoint:
        return self

    def __exit__(self, *_exc) -> None:
        self._file.close()

    def metadata(self) -> dict[str, str]:
        return dict(self._metadata)

    def keys(self) -> list[str]:
        return sorted(self._tensors)

    def get_slice(self, name: str) -> _TensorInfo:
        return self._tensors[name]

    def get_tensor(self, name: str) -> torch.Tensor:
        info = self._tensors[name]
        storage = torch.empty(info.end - info.start, dtype=torch.uint8, device="cpu")
        self._file.seek(self._offset + info.start)
        with memoryview(storage.numpy()) as buffer:
            read = 0
            while read < len(buffer):
                count = self._file.readinto(buffer[read:])
                if not count:
                    raise ValueError(f"Truncated tensor data: {name}")
                read += count
        return storage.view(_DTYPES[info.dtype]).reshape(info.shape)
