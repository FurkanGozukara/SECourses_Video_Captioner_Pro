"""Collision-safe run allocation, atomic output writing, and metadata."""

from __future__ import annotations

import json
import os
import re
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .logs import AppLog, get_log
from .paths import natural_sort_key, normalize_path, sanitize_filename

_RUN_PATTERN = re.compile(r"^(?:batch_)?(\d{4,})_.+$", re.IGNORECASE)
_OUTPUT_SUFFIXES = {
    "txt": ".txt",
    "json": ".json",
    "srt": ".srt",
    "vtt": ".vtt",
    "jsonl": ".jsonl",
    "reasoning": "",
}
_MAX_HISTORY_METADATA_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class RunSummary:
    """One lightweight, failure-tolerant entry in the output run history."""

    run_dir: str
    name: str
    kind: str
    model_key: str
    created: float
    items: int
    counts: dict[str, int]
    preview: str
    metadata_path: str | None


def _history_json(path: Path) -> dict[str, Any] | None:
    try:
        if path.stat().st_size > _MAX_HISTORY_METADATA_BYTES:
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _history_preview(folder: Path) -> str:
    try:
        top_level = [
            path
            for path in folder.glob("*.txt")
            if path.name.casefold() != "run_log.txt"
        ]
        nested = [
            path
            for path in folder.glob("*/*.txt")
            if path.name.casefold() != "run_log.txt"
        ]
    except OSError:
        return ""
    candidates = sorted(top_level, key=lambda item: natural_sort_key(item.name))
    candidates.extend(
        sorted(nested, key=lambda item: natural_sort_key(str(item.relative_to(folder))))
    )
    for path in candidates:
        try:
            return path.read_text(encoding="utf-8").strip()[:160]
        except (OSError, UnicodeError):
            continue
    return ""


def _history_model_key(payload: Mapping[str, Any] | None) -> str:
    if not isinstance(payload, Mapping):
        return ""
    model_info = payload.get("model_info")
    if isinstance(model_info, Mapping):
        value = (
            model_info.get("variant_key")
            or model_info.get("model_key")
            or model_info.get("alias")
        )
        if value:
            return str(value)
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping) and metadata.get("model_key"):
        return str(metadata["model_key"])
    settings = payload.get("settings")
    if isinstance(settings, Mapping):
        value = (
            settings.get("model_key")
            or settings.get("variant_key")
            or settings.get("whisper_model")
        )
        if value:
            return str(value)
    extra = payload.get("extra")
    params = extra.get("params") if isinstance(extra, Mapping) else None
    if isinstance(params, Mapping) and params.get("model"):
        return str(params["model"])
    return str(payload.get("model_key") or payload.get("variant_key") or "")


def _history_items_and_counts(payload: Mapping[str, Any] | None) -> tuple[int, dict[str, int]]:
    if not isinstance(payload, Mapping):
        return 0, {}
    values = payload.get("items_results")
    if isinstance(values, list):
        counts: dict[str, int] = {}
        for item in values:
            if not isinstance(item, Mapping):
                continue
            status = str(item.get("status") or "").strip().casefold()
            if status:
                counts[status] = counts.get(status, 0) + 1
        return len(values), counts
    messages = payload.get("messages")
    if isinstance(messages, list):
        return len(messages), {}
    extra = payload.get("extra")
    raw_counts = extra.get("counts") if isinstance(extra, Mapping) else payload.get("counts")
    counts = {}
    if isinstance(raw_counts, Mapping):
        for key, value in raw_counts.items():
            if str(key).casefold() == "total":
                continue
            try:
                counts[str(key)] = int(value)
            except (TypeError, ValueError):
                continue
    try:
        items = int(payload.get("items") or sum(counts.values()))
    except (TypeError, ValueError):
        items = 0
    return max(0, items), counts


def _history_kind(
    folder: Path,
    payload: Mapping[str, Any] | None,
    chat_marker: bool,
    regeneration_marker: bool = False,
) -> str:
    if regeneration_marker:
        return "regenerate"
    if chat_marker or "_chat_" in f"_{folder.name.casefold()}_":
        return "chat"
    extra = payload.get("extra") if isinstance(payload, Mapping) else None
    recorded_kind = str(extra.get("kind") or "").casefold() if isinstance(extra, Mapping) else ""
    if recorded_kind in {"whisper_transcription", "transcribe", "transcription"}:
        return "transcribe"
    if folder.name.casefold().endswith("_whisper") and payload is not None:
        return "transcribe"
    if folder.name.casefold().startswith("batch_"):
        return "batch"
    settings = payload.get("settings") if isinstance(payload, Mapping) else None
    if isinstance(settings, Mapping):
        typed = settings.get("typed_job")
        output = typed.get("output") if isinstance(typed, Mapping) else None
        selected = output.get("kind") if isinstance(output, Mapping) else settings.get("output_kind")
        if str(selected).casefold() in {"single", "batch"}:
            return str(selected).casefold()
    return "single" if payload is not None else "other"


def _regeneration_item_preview(payload: Mapping[str, Any] | None) -> str:
    if not isinstance(payload, Mapping):
        return ""
    values = payload.get("items_results")
    if not isinstance(values, list):
        return ""
    names: list[str] = []
    for item in values:
        if not isinstance(item, Mapping):
            continue
        outputs = item.get("outputs")
        raw_values = list(outputs.values()) if isinstance(outputs, Mapping) else []
        raw_values.extend([item.get("caption_path"), item.get("path")])
        raw = next((str(value) for value in raw_values if value), "")
        if raw:
            path = Path(raw)
            label = f"{path.parent.name}/{path.name}" if path.parent.name.casefold().endswith("_segments") else path.name
            if label not in names:
                names.append(label)
    return ", ".join(names[:3])


def _transcription_preview(folder: Path, payload: Mapping[str, Any] | None) -> str:
    preview = _history_preview(folder)
    if preview or not isinstance(payload, Mapping):
        return preview
    items = payload.get("items_results")
    if not isinstance(items, list):
        return ""
    for item in items:
        if not isinstance(item, Mapping):
            continue
        result = item.get("result")
        if isinstance(result, Mapping) and str(result.get("text") or "").strip():
            return str(result["text"]).strip()[:160]
        raw_files = item.get("files")
        candidates = list(raw_files) if isinstance(raw_files, (list, tuple)) else []
        outputs = item.get("outputs")
        if isinstance(outputs, Mapping):
            candidates.extend(outputs.values())
        for raw in candidates:
            path = Path(str(raw))
            if not path.is_absolute():
                path = folder / path
            try:
                if path.suffix.casefold() == ".txt" and path.is_file():
                    text = path.read_text(encoding="utf-8").strip()
                elif path.suffix.casefold() == ".json" and path.is_file():
                    document = _history_json(path)
                    text = str(document.get("text") or "").strip() if document else ""
                else:
                    continue
            except (OSError, UnicodeError):
                continue
            if text:
                return text[:160]
    return ""


def _caption_preview(folder: Path, payload: Mapping[str, Any] | None) -> str:
    """Preview a caption from its run folder or metadata-recorded split path."""

    preview = _history_preview(folder)
    if preview or not isinstance(payload, Mapping):
        return preview
    items = payload.get("items_results")
    if not isinstance(items, list):
        return ""
    for item in items:
        if not isinstance(item, Mapping):
            continue
        outputs = item.get("outputs")
        if isinstance(outputs, Mapping):
            candidates = [
                outputs.get("merged_caption"),
                outputs.get("txt"),
                outputs.get("video_caption"),
                outputs.get("audio_caption"),
            ]
            for raw in candidates:
                if not raw:
                    continue
                path = Path(str(raw))
                if not path.is_absolute():
                    path = folder / path
                try:
                    text = path.read_text(encoding="utf-8").strip() if path.is_file() else ""
                except (OSError, UnicodeError):
                    text = ""
                if text:
                    return text[:160]
        segments = item.get("segments")
        if isinstance(segments, list):
            text = " ".join(
                str(segment.get("caption") or "").strip()
                for segment in segments
                if isinstance(segment, Mapping) and segment.get("caption")
            ).strip()
            if text:
                return text[:160]
    return ""


def list_recent_runs(
    outputs_root: str | os.PathLike[str],
    limit: int = 30,
) -> list[RunSummary]:
    """Return newest output runs first without failing on damaged run folders."""

    maximum = max(0, int(limit))
    if maximum == 0:
        return []
    try:
        root = normalize_path(outputs_root)
        folders = [path for path in root.iterdir() if path.is_dir()]
    except (OSError, TypeError, ValueError):
        return []
    summaries: list[RunSummary] = []
    for folder in folders:
        metadata = folder / "metadata.json"
        regeneration = folder / "editor_regeneration_metadata.json"
        chat = folder / "chat.json"
        run_log = folder / "run_log.txt"
        try:
            markers = [path for path in (metadata, regeneration, chat, run_log) if path.is_file()]
        except OSError:
            continue
        if not markers:
            continue
        is_regeneration = regeneration in markers and metadata not in markers
        data_path = (
            metadata
            if metadata in markers
            else regeneration
            if regeneration in markers
            else chat
            if chat in markers
            else None
        )
        payload = _history_json(data_path) if data_path is not None else None
        items, counts = _history_items_and_counts(payload)
        kind = _history_kind(folder, payload, chat in markers, is_regeneration)
        try:
            created = float(folder.stat().st_mtime)
        except OSError:
            continue
        summaries.append(
            RunSummary(
                run_dir=str(folder.resolve(strict=False)),
                name=folder.name,
                kind=kind,
                model_key=_history_model_key(payload),
                created=created,
                items=items,
                counts=counts,
                preview=(
                    _regeneration_item_preview(payload)
                    if is_regeneration
                    else _transcription_preview(folder, payload)
                    if kind == "transcribe"
                    else _caption_preview(folder, payload)
                ),
                metadata_path=(
                    str(data_path.resolve(strict=False)) if data_path is not None else None
                ),
            )
        )
    summaries.sort(key=lambda item: (-item.created, natural_sort_key(item.name)))
    return summaries[:maximum]


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
    "RunSummary",
    "allocate_run_dir",
    "list_recent_runs",
    "load_metadata",
    "model_short_name",
]
