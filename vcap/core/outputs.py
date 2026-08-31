"""Collision-safe run allocation, atomic output writing, and metadata."""

from __future__ import annotations

import json
import os
import re
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .logs import AppLog, get_log
from .paths import sanitize_filename

_RUN_PATTERN = re.compile(r"^(?:batch_)?(\d{4,})_.+$", re.IGNORECASE)
_OUTPUT_SUFFIXES = {
    "txt": ".txt",
    "json": ".json",
    "srt": ".srt",
    "vtt": ".vtt",
    "jsonl": ".jsonl",
    "reasoning": "",
}


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def model_short_name(model_key: str) -> str:
    """Reduce a model variant key to its stable output-directory name."""

    normalized = str(model_key or "model").strip().casefold()
    if normalized.startswith("qwen3"):
        return "qwen3"
    token = normalized.split("_", 1)[0]
    return sanitize_filename(token or "model", max_len=48)


def allocate_run_dir(
    outputs_root: str | os.PathLike[str],
    model_short: str,
    kind: str = "single",
) -> Path:
    """Atomically allocate the next monotonic single or batch run directory."""

    normalized_kind = str(kind).casefold()
    if normalized_kind not in {"single", "batch"}:
        raise ValueError("kind must be 'single' or 'batch'")
    root = Path(outputs_root)
    root.mkdir(parents=True, exist_ok=True)
    maximum = 0
    try:
        for child in root.iterdir():
            try:
                if not child.is_dir():
                    continue
                match = _RUN_PATTERN.match(child.name)
                if match:
                    maximum = max(maximum, int(match.group(1)))
            except OSError:
                continue
    except OSError:
        pass
    short = sanitize_filename(model_short or "model", max_len=48)
    number = maximum + 1
    prefix = "batch_" if normalized_kind == "batch" else ""
    while True:
        candidate = root / f"{prefix}{number:04d}_{short}"
        try:
            candidate.mkdir(parents=False, exist_ok=False)
            return candidate
        except FileExistsError:
            number += 1


class OutputWriter:
    """Write text and JSON outputs atomically beneath an optional root."""

    def __init__(self, root: str | os.PathLike[str] | None = None) -> None:
        self.root = Path(root).resolve(strict=False) if root is not None else None

    def _path(self, path: str | os.PathLike[str]) -> Path:
        target = Path(path)
        if self.root is not None and not target.is_absolute():
            target = self.root / target
        return target

    def atomic_write(self, path: str | os.PathLike[str], data: str | bytes) -> Path:
        """Replace a destination atomically with text or binary data."""

        target = self._path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        mode = "wb" if isinstance(data, bytes) else "w"
        kwargs: dict[str, Any] = {} if isinstance(data, bytes) else {"encoding": "utf-8", "newline": "\n"}
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode=mode,
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
                delete=False,
                **kwargs,
            ) as handle:
                temporary_name = handle.name
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, target)
        finally:
            if temporary_name is not None:
                try:
                    Path(temporary_name).unlink(missing_ok=True)
                except OSError:
                    pass
        return target

    def write_text(self, path: str | os.PathLike[str], text: str) -> Path:
        """Atomically write UTF-8 text."""

        return self.atomic_write(path, str(text))

    def write_json(
        self,
        path: str | os.PathLike[str],
        obj: Any,
        pretty: bool = True,
    ) -> Path:
        """Atomically serialize a JSON value as UTF-8."""

        payload = json.dumps(
            obj,
            ensure_ascii=False,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
            default=str,
        )
        if pretty:
            payload += "\n"
        return self.atomic_write(path, payload)

    def caption_output_paths(
        self,
        out_dir: str | os.PathLike[str],
        stem: str,
        formats: Iterable[str],
    ) -> dict[str, Path]:
        """Build output paths for requested caption and reasoning formats."""

        directory = self._path(out_dir)
        safe_stem = sanitize_filename(Path(str(stem)).stem or "caption")
        result: dict[str, Path] = {}
        for raw_format in formats:
            output_format = str(raw_format).casefold().lstrip(".")
            if output_format not in _OUTPUT_SUFFIXES:
                raise ValueError(f"Unsupported caption format: {raw_format}")
            if output_format == "reasoning":
                filename = "reasoning.txt" if safe_stem.casefold() == "caption" else f"{safe_stem}_reasoning.txt"
            else:
                filename = f"{safe_stem}{_OUTPUT_SUFFIXES[output_format]}"
            result[output_format] = directory / filename
        return result


class MetadataBuilder:
    """Construct and persist the versioned run metadata document."""

    def __init__(self) -> None:
        self.data: dict[str, Any] | None = None

    def build(
        self,
        app_version: str,
        model_info: dict[str, Any],
        settings: dict[str, Any],
        items_results: list[dict[str, Any]],
        timings: dict[str, Any],
        gpu_info: dict[str, Any] | None,
        extra: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Build a complete JSON-safe metadata mapping."""

        timestamp = datetime.now(timezone.utc).isoformat()
        self.data = {
            "_meta": {
                "format": "secourses_vcap_metadata",
                "version": 1,
                "created_at": timestamp,
            },
            "timestamp": timestamp,
            "app_version": str(app_version),
            "model_info": _json_safe(model_info),
            "settings": _json_safe(settings),
            "items_results": _json_safe(items_results),
            "timings": _json_safe(timings),
            "gpu_info": _json_safe(gpu_info) if gpu_info is not None else None,
            "extra": _json_safe(extra) if extra is not None else {},
        }
        return deepcopy(self.data)

    def write(self, path: str | os.PathLike[str]) -> Path:
        """Write the most recently built metadata document."""

        if self.data is None:
            raise RuntimeError("build() must be called before write()")
        return OutputWriter().write_json(path, self.data, pretty=True)


def load_metadata(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load and validate a Video Captioner Pro metadata document."""

    target = Path(path)
    try:
        with target.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read metadata: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Metadata root must be a JSON object")
    meta = data.get("_meta")
    if not isinstance(meta, dict) or meta.get("format") != "secourses_vcap_metadata":
        raise ValueError("Not a SECourses Video Captioner Pro metadata file")
    if meta.get("version") != 1:
        raise ValueError(f"Unsupported metadata version: {meta.get('version')}")
    required_types = {
        "app_version": str,
        "model_info": dict,
        "settings": dict,
        "items_results": list,
        "timings": dict,
    }
    for key, expected in required_types.items():
        if not isinstance(data.get(key), expected):
            raise ValueError(f"Metadata field '{key}' must be {expected.__name__}")
    return data


class RunLog:
    """Attach ``run_log.txt`` to the shared application logger for a context."""

    def __init__(self, run_dir: str | os.PathLike[str], app_log: AppLog | None = None) -> None:
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / "run_log.txt"
        self.app_log = app_log or get_log()

    def __enter__(self) -> "RunLog":
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.app_log.attach_file(self.path)
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.app_log.detach_file(self.path)


__all__ = [
    "MetadataBuilder",
    "OutputWriter",
    "RunLog",
    "allocate_run_dir",
    "load_metadata",
    "model_short_name",
]
