"""Shared streaming helpers for the Video Captioner Pro model tools."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import struct
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Iterable, Iterator, Mapping, Sequence


DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


@dataclass(frozen=True)
class TensorRecord:
    name: str
    dtype: str
    shape: tuple[int, ...]
    path: Path
    start: int
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class TensorSpec:
    dtype: str
    shape: tuple[int, ...]

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * DTYPE_BYTES[self.dtype]


class ProgressLine:
    def __init__(self, label: str, total_items: int, total_bytes: int) -> None:
        self.label = label
        self.total_items = total_items
        self.total_bytes = total_bytes
        self.started = time.perf_counter()
        self.last_draw = 0.0

    def update(self, item: int, completed_bytes: int, name: str, *, force: bool = False) -> None:
        now = time.perf_counter()
        if not force and now - self.last_draw < 0.15 and item < self.total_items:
            return
        self.last_draw = now
        elapsed = max(now - self.started, 1e-9)
        rate = completed_bytes / elapsed
        remaining = max(self.total_bytes - completed_bytes, 0)
        eta = remaining / rate if rate else 0.0
        short_name = name if len(name) <= 72 else "..." + name[-69:]
        line = (
            f"\r{self.label}: tensor {item}/{self.total_items} | "
            f"{completed_bytes / 1e9:.2f}/{self.total_bytes / 1e9:.2f} GB | "
            f"ETA {format_duration(eta)} | {short_name}"
        )
        print(line.ljust(150), end="", flush=True)
        if item >= self.total_items:
            print(flush=True)


def format_duration(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def read_safetensors_header(path: Path) -> tuple[dict[str, str], dict[str, TensorRecord]]:
    path = path.resolve()
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"Not a safetensors file (short header): {path}")
        header_length = struct.unpack("<Q", raw_length)[0]
        if header_length <= 0 or header_length > 100_000_000:
            raise ValueError(f"Invalid safetensors header length {header_length}: {path}")
        raw_header = handle.read(header_length)
    header = json.loads(raw_header.rstrip(b" \t\r\n\0").decode("utf-8"))
    metadata = dict(header.pop("__metadata__", {}) or {})
    data_start = 8 + header_length
    records: dict[str, TensorRecord] = {}
    for name, item in header.items():
        begin, end = item["data_offsets"]
        record = TensorRecord(
            name=name,
            dtype=item["dtype"],
            shape=tuple(item["shape"]),
            path=path,
            start=data_start + int(begin),
            end=data_start + int(end),
        )
        expected = math.prod(record.shape) * DTYPE_BYTES[record.dtype]
        if record.nbytes != expected:
            raise ValueError(
                f"Tensor byte-size mismatch for {name}: header={record.nbytes}, expected={expected}"
            )
        records[name] = record
    return metadata, records


def discover_tensor_records(source: Path) -> tuple[dict[str, str], dict[str, TensorRecord]]:
    """Read a single file or a sharded HF folder without materializing tensors."""
    source = source.resolve()
    if source.is_file():
        return read_safetensors_header(source)
    if not source.is_dir():
        raise FileNotFoundError(source)

    single = source / "model.safetensors"
    index_path = source / "model.safetensors.index.json"
    if single.exists():
        return read_safetensors_header(single)
    if not index_path.exists():
        raise FileNotFoundError(f"No model.safetensors or index in {source}")

    with index_path.open("r", encoding="utf-8") as handle:
        index = json.load(handle)
    weight_map: dict[str, str] = index.get("weight_map", {})
    if not weight_map:
        raise ValueError(f"Empty weight_map: {index_path}")

    by_shard: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        by_shard.setdefault(shard, []).append(name)

    records: dict[str, TensorRecord] = {}
    shard_metadata: dict[str, str] = {}
    for shard, expected_names in sorted(by_shard.items()):
        metadata, shard_records = read_safetensors_header(source / shard)
        shard_metadata.update(metadata)
        missing = set(expected_names) - shard_records.keys()
        if missing:
            raise ValueError(f"{shard} is missing {len(missing)} indexed tensors")
        for name in expected_names:
            if name in records:
                raise ValueError(f"Duplicate tensor in index: {name}")
            records[name] = shard_records[name]

    extra = set(records) - set(weight_map)
    missing = set(weight_map) - set(records)
    if extra or missing:
        raise ValueError(f"Index/header mismatch: {len(missing)} missing, {len(extra)} extra")
    return shard_metadata, records


def normalize_metadata(metadata: Mapping[str, object]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in metadata.items():
        if isinstance(value, str):
            normalized[str(key)] = value
        elif isinstance(value, (dict, list, tuple, bool)) or value is None:
            normalized[str(key)] = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
        else:
            normalized[str(key)] = str(value)
    return normalized


def make_safetensors_header(
    specs: Mapping[str, TensorSpec], metadata: Mapping[str, object]
) -> tuple[bytes, list[str]]:
    header: dict[str, object] = {"__metadata__": normalize_metadata(metadata)}
    offset = 0
    order = list(specs)
    for name in order:
        spec = specs[name]
        header[name] = {
            "dtype": spec.dtype,
            "shape": list(spec.shape),
            "data_offsets": [offset, offset + spec.nbytes],
        }
        offset += spec.nbytes
    raw = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    padded_length = (len(raw) + 7) & ~7
    return struct.pack("<Q", padded_length) + raw + b" " * (padded_length - len(raw)), order


def copy_range(source: BinaryIO, target: BinaryIO, start: int, length: int, chunk_size: int = 16 << 20) -> None:
    source.seek(start)
    remaining = length
    while remaining:
        chunk = source.read(min(chunk_size, remaining))
        if not chunk:
            raise EOFError(f"Unexpected EOF with {remaining} bytes left")
        target.write(chunk)
        remaining -= len(chunk)


def hash_range(path: Path, start: int, length: int, chunk_size: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        handle.seek(start)
        remaining = length
        while remaining:
            chunk = handle.read(min(chunk_size, remaining))
            if not chunk:
                raise EOFError(path)
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def sha256_file(path: Path, chunk_size: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_cpu_tensor(handle: BinaryIO, tensor: object) -> None:
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(type(tensor))
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()
    array = tensor.detach().contiguous().view(torch.uint8).numpy()
    array.tofile(handle)


class RawFileCache:
    def __init__(self) -> None:
        self.path: Path | None = None
        self.handle: BinaryIO | None = None

    def get(self, path: Path) -> BinaryIO:
        if path != self.path:
            self.close()
            self.path = path
            self.handle = path.open("rb")
        assert self.handle is not None
        return self.handle

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
        self.handle = None
        self.path = None

    def __enter__(self) -> "RawFileCache":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class SafeOpenCache:
    def __init__(self) -> None:
        self.path: Path | None = None
        self.context: object | None = None
        self.reader: object | None = None

    def get_tensor(self, record: TensorRecord):
        from safetensors import safe_open

        if record.path != self.path:
            self.close()
            self.path = record.path
            self.context = safe_open(str(record.path), framework="pt", device="cpu")
            self.reader = self.context.__enter__()
        assert self.reader is not None
        return self.reader.get_tensor(record.name)

    def close(self) -> None:
        if self.context is not None:
            self.context.__exit__(None, None, None)
        self.path = None
        self.context = None
        self.reader = None

    def __enter__(self) -> "SafeOpenCache":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


COPY_PATTERNS = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "tokenizer*",
    "vocab*",
    "merges*",
    "chat_template*",
    "special_tokens_map*",
    "added_tokens*",
    "spk_dict.pt",
)


def copy_model_support_files(source_dir: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    seen: set[Path] = set()
    for pattern in COPY_PATTERNS:
        for source in sorted(source_dir.glob(pattern)):
            if not source.is_file() or source in seen:
                continue
            seen.add(source)
            target = out_dir / source.name
            shutil.copy2(source, target)
            copied.append(target)
    if not (out_dir / "config.json").exists():
        raise FileNotFoundError(f"config.json was not found in {source_dir}")
    return copied


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(partial, path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def infer_source_repo(model_dir: Path) -> str:
    name = model_dir.name
    known = {
        "TimeChat-Captioner-GRPO-7B": "yaolily/TimeChat-Captioner-GRPO-7B",
        "AVoCaDO": "AVoCaDO-Captioner/AVoCaDO",
        "Qwen3-Omni-30B-A3B-Thinking": "Qwen/Qwen3-Omni-30B-A3B-Thinking",
        "Qwen3-Omni-30B-A3B-Instruct": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "Qwen3-Omni-30B-A3B-Captioner": "Qwen/Qwen3-Omni-30B-A3B-Captioner",
    }
    return known.get(name, str(model_dir.resolve()))


def infer_family(config_path: Path) -> str:
    with config_path.open("r", encoding="utf-8") as handle:
        model_type = json.load(handle).get("model_type", "unknown")
    if model_type == "qwen2_5_omni":
        return "qwen2_5_omni"
    if model_type == "qwen3_omni_moe":
        return "qwen3_omni_moe"
    return str(model_type)


def ensure_utf8_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def total_spec_bytes(specs: Mapping[str, TensorSpec]) -> int:
    return sum(spec.nbytes for spec in specs.values())


def iter_chunks(values: Sequence[str], size: int) -> Iterator[Sequence[str]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]
