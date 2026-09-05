"""Caption review, correction, regeneration, and approved-dataset export."""

from __future__ import annotations

import html
import hashlib
import json
import math
import os
import queue
import re
import threading
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, TypedDict

import gradio as gr
from PIL import Image, ImageDraw

from vcap import OUTPUTS_DIR
from vcap.core.archive import zip_directory
from vcap.core.caption_stats import calculate_caption_statistics, render_caption_statistics
from vcap.core.captions_post import (
    apply_replacements,
    caption_stats,
    diff_html,
    finalize_caption,
    parse_replace_pairs,
)
from vcap.core.dataset_captions import (
    DEFAULT_CAPTION_MERGE_TEMPLATE,
    render_caption_template,
)
from vcap.core.export import export_dataset, read_flags, write_flags
from vcap.core.media import make_thumbnail, preview_safe_media, probe_media
from vcap.core.outputs import OutputWriter
from vcap.core.paths import (
    guess_kind_by_extension,
    list_media_files,
    natural_sort_key,
    normalize_path,
    open_in_file_manager,
    reveal_in_file_manager,
)
from vcap.models.registry import MODEL_SPECS, all_variant_choices, variant_to_family
from vcap.pipeline.job import InputItem, JobSpec, OutputSpec
from vcap.prompts.presets import default_preset_for, get_preset, list_presets
from vcap.ui.components import action_button

if TYPE_CHECKING:
    from vcap.ui.app import UiContext


CAPTION_EXTENSIONS = (".txt", ".json", ".srt", ".vtt", ".jsonl")
_RUN_DIR_RE = re.compile(r"^(?:batch_)?\d{4,}_.+", re.IGNORECASE)
_IGNORED_JSON_NAMES = {
    ".vcap_flags.json",
    "metadata.json",
    "captions_index.json",
    "summary.json",
    "split_manifest.json",
    "vcap_model_info.json",
}
_IGNORED_CAPTION_NAMES = {
    "run_log.txt",
    "reasoning.txt",
    "editor_regeneration_metadata.json",
}
_TABLE_HEADERS = ["#", "File", "Caption", "Len", "Tokens", "Flag", "Status"]
_DEFAULT_FILTER = {
    "search": "",
    "regex": False,
    "min_length": None,
    "max_length": None,
    "min_tokens": None,
    "max_tokens": None,
    "flag": "all",
    "status": "all",
    "token_limit": 512,
}


class EditorItem(TypedDict, total=False):
    media_path: str | None
    source_media_path: str | None
    caption_path: str
    caption: str
    caption_field: str | None
    caption_formats: list[str]
    video_caption_path: str | None
    audio_caption_path: str | None
    video_caption: str
    audio_caption: str
    kind: str
    duration: float | None
    chars: int
    tokens: int
    flag: str | None
    status: str
    segment_index: int
    start_s: float
    end_s: float
    segment_media_path: str | None
    split_mode: str
    encode_codec: str
    encode_crf: int
    encode_preset: str
    encode_audio_bitrate: str
    batch_run_item: bool


class EditorState(TypedDict, total=False):
    folder: str
    items: list[EditorItem]
    selected_index: int | None
    filter: dict[str, Any]
    dirty: bool
    page: int
    page_size: int
    last_edit: float
    draft_caption: str | None


def new_editor_state(folder: str | os.PathLike[str] = "") -> EditorState:
    """Return the JSON-safe state shared by editor callbacks."""

    return {
        "folder": os.fspath(folder),
        "items": [],
        "selected_index": None,
        "filter": dict(_DEFAULT_FILTER),
        "dirty": False,
        "page": 1,
        "page_size": 25,
        "last_edit": 0.0,
        "draft_caption": None,
    }


def _scan_directories(root: Path, recursive: bool) -> list[Path]:
    if recursive:
        return [root]
    directories = [root]
    try:
        directories.extend(
            child
            for child in root.iterdir()
            if child.is_dir()
            and child.name.casefold() not in {"video_caption", "audio_caption"}
            and (
                (_RUN_DIR_RE.match(child.name) and (child / "metadata.json").is_file())
                or _contains_caption_sidecars(child)
            )
        )
    except (OSError, PermissionError):
        pass
    return directories


def _contains_caption_sidecars(folder: Path) -> bool:
    try:
        return any(
            path.is_file()
            and not path.name.startswith(".")
            and path.suffix.casefold() in CAPTION_EXTENSIONS
            and path.name.casefold() not in _IGNORED_CAPTION_NAMES
            and path.name.casefold() not in _IGNORED_JSON_NAMES
            for path in folder.iterdir()
        )
    except (OSError, PermissionError):
        return False


def _walk_caption_files(root: Path, recursive: bool) -> list[Path]:
    iterators = (
        [root.rglob("*")]
        if recursive
        else [directory.iterdir() for directory in _scan_directories(root, False)]
    )
    found: list[Path] = []
    for iterator in iterators:
        try:
            paths = iterator
            for path in paths:
                try:
                    suffix = path.suffix.casefold()
                    if not path.is_file() or path.name.startswith(".") or suffix not in CAPTION_EXTENSIONS:
                        continue
                    try:
                        relative_parts = path.relative_to(root).parts[:-1]
                    except ValueError:
                        relative_parts = path.parts[:-1]
                    if any(
                        str(part).casefold() in {"video_caption", "audio_caption"}
                        for part in relative_parts
                    ):
                        continue
                    if path.name.casefold() in _IGNORED_CAPTION_NAMES:
                        continue
                    if suffix == ".json" and path.name.casefold() in _IGNORED_JSON_NAMES:
                        continue
                    found.append(path)
                except (OSError, PermissionError):
                    continue
        except (OSError, PermissionError):
            continue
    return sorted(found, key=lambda value: (natural_sort_key(value), str(value).casefold(), str(value)))


def _read_caption(path: Path) -> tuple[str, str | None]:
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "", None
    if path.suffix.casefold() != ".json":
        return raw.rstrip("\r\n"), None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return raw.rstrip("\r\n"), None
    if isinstance(value, Mapping):
        for key in _JSON_CAPTION_KEYS:
            if isinstance(value.get(key), str):
                return str(value[key]), key
    return json.dumps(value, ensure_ascii=False, indent=2), None


_JSON_CAPTION_KEYS = ("caption", "text", "final_caption", "description")


def _sync_json_sidecar(caption_path: Path, text: str) -> None:
    """Mirror an edited ``.txt`` caption into the ``text``/``caption`` field of its JSON sibling."""

    sidecar = caption_path.with_suffix(".json")
    if caption_path.suffix.casefold() != ".txt" or not sidecar.is_file():
        return
    try:
        document = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(document, dict):
        return
    key = next((name for name in _JSON_CAPTION_KEYS if isinstance(document.get(name), str)), None)
    if key is not None and document[key] != text:
        document[key] = text
        OutputWriter().write_json(sidecar, document, pretty=True)


def _write_caption(path: Path, text: str, field: str | None = None) -> Path:
    """Atomically write text while preserving common JSON caption wrappers."""

    value = str(text)
    _sync_json_sidecar(path, value)
    if path.suffix.casefold() == ".json":
        existing: Any = None
        try:
            existing = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
        except (OSError, json.JSONDecodeError):
            pass
        if field and isinstance(existing, dict):
            existing[field] = value
            return OutputWriter().write_json(path, existing, pretty=True)
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = {"text": value}
        return OutputWriter().write_json(path, parsed, pretty=True)
    return OutputWriter().write_text(path, value + ("\n" if value and not value.endswith("\n") else ""))


def _flag_key(root: Path, item: Mapping[str, Any]) -> str | None:
    for raw in (item.get("media_path"), item.get("caption_path")):
        if not raw:
            continue
        try:
            return normalize_path(str(raw)).relative_to(root).as_posix()
        except (OSError, ValueError):
            continue
    return None


def _failed_media(root: Path, recursive: bool) -> tuple[set[str], set[str]]:
    paths: set[str] = set()
    names: set[str] = set()
    iterator = (
        root.rglob("metadata.json")
        if recursive
        else (
            directory / "metadata.json"
            for directory in _scan_directories(root, False)
            if (directory / "metadata.json").is_file()
        )
    )
    for metadata_path in iterator:
        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for result in data.get("items_results", []) if isinstance(data, dict) else []:
            if not isinstance(result, dict) or str(result.get("status", "")).casefold() != "failed":
                continue
            raw_path = str(result.get("path") or "")
            if raw_path:
                try:
                    paths.add(str(normalize_path(raw_path)).casefold())
                except Exception:
                    paths.add(raw_path.casefold())
                names.add(Path(raw_path).name.casefold())
    return paths, names


def _path_values(value: Any) -> list[str]:
    if isinstance(value, (str, os.PathLike)):
        text = os.fspath(value).strip()
        return [text] if text else []
    if isinstance(value, Mapping):
        values: list[str] = []
        for key in ("path", "name", "input", "source", "media_path", "file"):
            if key in value:
                values.extend(_path_values(value[key]))
        if not values:
            for entry in value.values():
                values.extend(_path_values(entry))
        return values
    if isinstance(value, (list, tuple, set)):
        values = []
        for entry in value:
            values.extend(_path_values(entry))
        return values
    return []


def _metadata_candidate(raw: str, metadata_path: Path) -> Path:
    text = str(raw).strip().strip('"').strip("'")
    direct = normalize_path(text)
    raw_path = Path(text)
    if raw_path.is_absolute():
        return direct
    relative = normalize_path(metadata_path.parent / raw_path)
    return relative if relative.exists() or not direct.exists() else direct


def _metadata_entries(document: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    entries: list[Mapping[str, Any]] = []
    for key in ("items", "items_results"):
        value = document.get(key)
        if isinstance(value, list):
            entries.extend(entry for entry in value if isinstance(entry, Mapping))
    return entries


def _entry_output_paths(entry: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("outputs", "output", "caption_path", "output_path"):
        if key in entry:
            values.extend(_path_values(entry[key]))
    return values


def _entry_source_paths(entry: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("input", "source_media_path", "source_path", "source", "media_path", "path", "file"):
        if key in entry:
            values.extend(_path_values(entry[key]))
    return values


def _is_caption_match(raw: str, caption_path: Path, metadata_path: Path) -> bool:
    try:
        candidate = _metadata_candidate(raw, metadata_path)
    except Exception:
        return False
    return _path_identity(candidate) == _path_identity(caption_path) or (
        candidate.stem.casefold() == caption_path.stem.casefold()
        and candidate.suffix.casefold() in CAPTION_EXTENSIONS
    )


def _path_identity(path: str | os.PathLike[str]) -> str:
    """Return a Unicode-safe path key with Windows-style case folding."""

    try:
        value = os.path.normcase(str(normalize_path(path)))
    except Exception:
        value = os.path.normcase(os.fspath(path))
    return value.casefold()


def _metadata_media_candidates(
    document: Mapping[str, Any],
    metadata_path: Path,
    caption_path: Path,
) -> list[Path]:
    entries = _metadata_entries(document)
    matched = [
        entry
        for entry in entries
        if any(_is_caption_match(raw, caption_path, metadata_path) for raw in _entry_output_paths(entry))
    ]
    if not matched:
        matched = [
            entry
            for entry in entries
            if any(Path(raw).stem.casefold() == caption_path.stem.casefold() for raw in _entry_source_paths(entry))
        ]
    if not matched and len(entries) == 1:
        matched = entries

    raw_candidates: list[str] = []
    for entry in matched:
        raw_candidates.extend(_entry_source_paths(entry))
    settings = document.get("settings")
    settings_map = settings if isinstance(settings, Mapping) else {}
    configured = [
        *_path_values(settings_map.get("input_files")),
        *_path_values(settings_map.get("input_path")),
        *_path_values(document.get("input_files")),
        *_path_values(document.get("input_path")),
    ]
    matching_configured = [
        raw for raw in configured if Path(raw).stem.casefold() == caption_path.stem.casefold()
    ]
    raw_candidates.extend(matching_configured)
    if len(configured) == 1 or caption_path.stem.casefold() == "caption":
        raw_candidates.extend(configured)

    candidates: list[Path] = []
    seen: set[str] = set()
    for raw in raw_candidates:
        try:
            candidate = _metadata_candidate(raw, metadata_path)
        except Exception:
            continue
        if guess_kind_by_extension(candidate) not in {"video", "audio", "image"}:
            continue
        key = str(candidate).casefold()
        if key not in seen:
            seen.add(key)
            candidates.append(candidate)
    return candidates


def _batch_document_records(
    document: Mapping[str, Any],
    document_path: Path,
) -> list[dict[str, Any]]:
    """Normalize a captions index or legacy batch metadata into lookup rows."""

    records: list[dict[str, Any]] = []
    captions = document.get("captions")
    if isinstance(captions, Mapping):
        for raw_caption, raw_record in captions.items():
            if not isinstance(raw_record, Mapping):
                continue
            source_values = _path_values(raw_record.get("source_path"))
            if not source_values:
                continue
            try:
                caption = _metadata_candidate(str(raw_caption), document_path)
                source = _metadata_candidate(source_values[0], document_path)
            except Exception:
                continue
            records.append(
                {
                    "caption_path": str(caption),
                    "source_path": str(source),
                    "kind": str(raw_record.get("kind") or guess_kind_by_extension(source)),
                    "start_s": raw_record.get("start_s"),
                    "end_s": raw_record.get("end_s"),
                    "output_key": str(raw_record.get("output_key") or ""),
                    "metadata_path": str(document_path),
                }
            )
        return records

    for entry in _metadata_entries(document):
        sources = _entry_source_paths(entry)
        if not sources:
            continue
        try:
            source = _metadata_candidate(sources[0], document_path)
        except Exception:
            continue
        groups: list[Mapping[str, Any]] = [entry]
        segments = entry.get("segments")
        if isinstance(segments, list):
            groups.extend(value for value in segments if isinstance(value, Mapping))
        for group in groups:
            output_records: list[tuple[str, str]] = []
            raw_outputs = group.get("outputs")
            if isinstance(raw_outputs, Mapping):
                for output_key, raw_value in raw_outputs.items():
                    output_records.extend(
                        (str(output_key), raw_caption)
                        for raw_caption in _path_values(raw_value)
                    )
            for key in ("output", "caption_path", "output_path"):
                output_records.extend(
                    (key, raw_caption) for raw_caption in _path_values(group.get(key))
                )
            for output_key, raw_caption in output_records:
                normalized_key = output_key.casefold()
                if normalized_key.startswith("transcript_") or normalized_key == "reasoning":
                    continue
                try:
                    caption = _metadata_candidate(raw_caption, document_path)
                except Exception:
                    continue
                if caption.suffix.casefold() not in CAPTION_EXTENSIONS:
                    continue
                records.append(
                    {
                        "caption_path": str(caption),
                        "source_path": str(source),
                        "kind": str(entry.get("kind") or guess_kind_by_extension(source)),
                        "start_s": group.get("start_s"),
                        "end_s": group.get("end_s"),
                        "output_key": output_key,
                        "metadata_path": str(document_path),
                    }
                )
    return records


def _read_batch_records(run_dir: Path) -> list[dict[str, Any]]:
    """Prefer a batch captions index and fall back to its metadata."""

    for name in ("captions_index.json", "metadata.json"):
        path = run_dir / name
        if not path.is_file():
            continue
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(document, Mapping):
            records = _batch_document_records(document, path)
            if records or name == "captions_index.json":
                return records
    return []


def _batch_caption_lookup(scan_root: Path) -> dict[str, dict[str, Any]]:
    """Load at most the newest 50 batch index documents once per editor scan."""

    roots: list[Path] = []
    root_keys: set[str] = set()
    for candidate in (scan_root, scan_root.parent, OUTPUTS_DIR):
        normalized = normalize_path(candidate)
        key = _path_identity(normalized)
        if key not in root_keys:
            roots.append(normalized)
            root_keys.add(key)
    run_dirs: dict[str, Path] = {}
    for root in roots:
        if (root / "captions_index.json").is_file() or (root / "metadata.json").is_file():
            run_dirs[_path_identity(root)] = root
        try:
            for child in root.glob("batch_*"):
                if child.is_dir() and (
                    (child / "captions_index.json").is_file()
                    or (child / "metadata.json").is_file()
                ):
                    run_dirs[_path_identity(child)] = child
        except (OSError, PermissionError):
            continue

    def newest(path: Path) -> float:
        values: list[float] = []
        for candidate in (path / "captions_index.json", path / "metadata.json"):
            try:
                values.append(candidate.stat().st_mtime)
            except OSError:
                pass
        return max(values, default=0.0)

    lookup: dict[str, dict[str, Any]] = {}
    for run_dir in sorted(run_dirs.values(), key=newest, reverse=True)[:50]:
        for record in _read_batch_records(run_dir):
            lookup.setdefault(_path_identity(record["caption_path"]), record)
    return lookup


def resolve_media_from_metadata(
    caption_path: str | os.PathLike[str],
    scan_root: str | os.PathLike[str] | None = None,
    batch_lookup: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[Path | None, Path | None]:
    """Return an existing source and the best metadata path candidate for a caption."""

    caption = normalize_path(caption_path)
    root = normalize_path(scan_root) if scan_root is not None else None
    directories = [caption.parent, *caption.parent.parents]
    for directory in directories:
        if root is not None:
            try:
                directory.relative_to(root)
            except ValueError:
                break
        metadata_path = directory / "metadata.json"
        if metadata_path.is_file():
            try:
                document = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                document = None
            if isinstance(document, Mapping):
                candidates = _metadata_media_candidates(document, metadata_path, caption)
                existing = next((candidate for candidate in candidates if candidate.is_file()), None)
                if existing is not None or candidates:
                    return existing, existing or candidates[0]
        if root is not None and directory == root:
            break
    lookup = batch_lookup if batch_lookup is not None else _batch_caption_lookup(root or caption.parent)
    record = lookup.get(_path_identity(caption)) if isinstance(lookup, Mapping) else None
    if isinstance(record, Mapping) and record.get("source_path"):
        source = normalize_path(str(record["source_path"]))
        return (source if source.is_file() else None), source
    return None, None


_CLIP_STEM_RE = re.compile(r"^clip_(\d+)$", re.IGNORECASE)


def _finite_window(value: Mapping[str, Any]) -> tuple[float, float] | None:
    try:
        start = float(value.get("start_s"))
        end = float(value.get("end_s"))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        return None
    return max(0.0, start), end


def _segment_context_from_metadata(
    caption_path: Path,
    scan_root: Path,
) -> dict[str, Any]:
    """Recover a segment's source, window, and optional persisted clip."""

    clip_match = _CLIP_STEM_RE.match(caption_path.stem)
    index_hint = int(clip_match.group(1)) if clip_match else None
    is_segment_dir = caption_path.parent.name.casefold().endswith("_segments")
    context: dict[str, Any] = {}

    sidecars = [caption_path] if caption_path.suffix.casefold() == ".json" else [caption_path.with_suffix(".json")]
    for sidecar in sidecars:
        try:
            document = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(document, Mapping):
            continue
        window = _finite_window(document)
        if window:
            context.update(start_s=window[0], end_s=window[1])
        for key in ("source_media_path", "source_path", "source", "path"):
            raw = document.get(key)
            if raw:
                context["source_media_path"] = str(_metadata_candidate(str(raw), sidecar))
                break
        raw_index = document.get("index", document.get("segment_index"))
        if raw_index is not None:
            try:
                context["segment_index"] = int(raw_index)
            except (TypeError, ValueError):
                pass

    directories = [caption_path.parent, *caption_path.parent.parents]
    for directory in directories:
        try:
            directory.relative_to(scan_root)
        except ValueError:
            break
        metadata_path = directory / "metadata.json"
        if not metadata_path.is_file():
            if directory == scan_root:
                break
            continue
        try:
            document = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            document = None
        if not isinstance(document, Mapping):
            continue
        settings = document.get("settings")
        if isinstance(settings, Mapping):
            for key in ("split_mode", "encode_codec", "encode_crf", "encode_preset", "encode_audio_bitrate"):
                if key in settings:
                    context[key] = settings[key]
        for entry in _metadata_entries(document):
            entry_matches = any(
                _is_caption_match(raw, caption_path, metadata_path)
                for raw in _entry_output_paths(entry)
            )
            if entry_matches:
                window = _finite_window(entry)
                if window:
                    context.update(start_s=window[0], end_s=window[1])
                raw_index = entry.get("index", entry.get("segment_index", index_hint))
                if raw_index is not None:
                    try:
                        context["segment_index"] = int(raw_index)
                    except (TypeError, ValueError):
                        pass
                source_candidates = _entry_source_paths(entry)
                if source_candidates:
                    context["source_media_path"] = str(
                        _metadata_candidate(source_candidates[0], metadata_path)
                    )
                clip_candidates = _path_values(entry.get("clip_path"))
                if clip_candidates:
                    produced = _metadata_candidate(clip_candidates[0], metadata_path)
                    if produced.is_file():
                        context["segment_media_path"] = str(produced)
            segments = entry.get("segments")
            if not isinstance(segments, list):
                continue
            source_candidates = _entry_source_paths(entry)
            source = source_candidates[0] if source_candidates else None
            for position, raw_segment in enumerate(segments, start=1):
                if not isinstance(raw_segment, Mapping):
                    continue
                raw_index = raw_segment.get("index", raw_segment.get("segment_index", position))
                try:
                    segment_index = int(raw_index)
                except (TypeError, ValueError):
                    segment_index = position
                output_match = any(
                    _is_caption_match(raw, caption_path, metadata_path)
                    for raw in _entry_output_paths(raw_segment)
                )
                if not output_match and index_hint is not None:
                    output_match = segment_index == index_hint
                if not output_match:
                    continue
                window = _finite_window(raw_segment)
                if window:
                    context.update(start_s=window[0], end_s=window[1])
                context["segment_index"] = segment_index
                if source:
                    context["source_media_path"] = str(_metadata_candidate(source, metadata_path))
                media_candidates = _path_values(raw_segment.get("media_path"))
                if media_candidates:
                    produced = _metadata_candidate(media_candidates[0], metadata_path)
                    if produced.is_file():
                        context["segment_media_path"] = str(produced)
                break
        if context.get("start_s") is not None and context.get("source_media_path"):
            break
        if directory == scan_root:
            break

    if (is_segment_dir or context.get("start_s") is not None) and "segment_index" not in context and index_hint is not None:
        context["segment_index"] = index_hint
    if is_segment_dir and not context.get("source_media_path"):
        source_stem = caption_path.parent.name[: -len("_segments")]
        candidates = [
            path
            for path in caption_path.parent.parent.glob(f"{source_stem}.*")
            if guess_kind_by_extension(path) in {"video", "audio", "image"}
        ]
        source = next((path for path in candidates if path.is_file()), None)
        if source is not None:
            context["source_media_path"] = str(source)
    if "segment_index" in context and not context.get("segment_media_path"):
        source_raw = context.get("source_media_path")
        source = Path(str(source_raw)) if source_raw else None
        index = int(context["segment_index"])
        candidates: list[Path] = []
        if source is not None:
            candidates.append(caption_path.parent.parent / f"{source.stem}_clips" / f"clip_{index:04d}{source.suffix}")
        if caption_path.parent.name.casefold().endswith("_segments"):
            clips_name = caption_path.parent.name[: -len("_segments")] + "_clips"
            clips_dir = caption_path.parent.with_name(clips_name)
            candidates.extend(clips_dir.glob(f"clip_{index:04d}.*") if clips_dir.is_dir() else [])
        produced = next((candidate for candidate in candidates if candidate.is_file()), None)
        if produced is not None:
            context["segment_media_path"] = str(produced)
    return context


def _enrich_segment_item(item: EditorItem, root: Path) -> None:
    caption_path = Path(str(item["caption_path"]))
    context = _segment_context_from_metadata(caption_path, root)
    if not context:
        return
    item.update(context)  # type: ignore[typeddict-item]
    source = context.get("source_media_path")
    if source:
        item["source_media_path"] = str(source)
        if Path(str(source)).is_file():
            item["media_path"] = str(source)
            item["kind"] = guess_kind_by_extension(source)
            if item.get("status") == "no media":
                item["status"] = "ok" if str(item.get("caption") or "").strip() else "empty"


def _item_from_pair(
    media: Path | None,
    caption_path: Path,
    formats: Iterable[Path],
    root: Path,
    flags: Mapping[str, Any],
    failed_paths: set[str],
    failed_names: set[str],
) -> EditorItem:
    caption, caption_field = _read_caption(caption_path) if caption_path.is_file() else ("", None)
    stats = caption_stats(caption)
    status = "no media" if media is None else ("ok" if caption.strip() else "empty")
    if media is not None:
        try:
            normalized = str(normalize_path(media)).casefold()
        except Exception:
            normalized = str(media).casefold()
        if normalized in failed_paths or media.name.casefold() in failed_names:
            status = "failed"
    item: EditorItem = {
        "media_path": str(media) if media is not None else None,
        "caption_path": str(caption_path),
        "caption": caption,
        "caption_field": caption_field,
        "caption_formats": [str(path) for path in formats],
        "kind": guess_kind_by_extension(media) if media is not None else "unknown",
        "duration": None,
        "chars": stats["chars"],
        "tokens": stats["approx_tokens"],
        "flag": None,
        "status": status,
    }
    video_part = caption_path.parent / "video_caption" / f"{caption_path.stem}.txt"
    audio_part = caption_path.parent / "audio_caption" / f"{caption_path.stem}.txt"
    item["video_caption_path"] = str(video_part) if video_part.is_file() else None
    item["audio_caption_path"] = str(audio_part) if audio_part.is_file() else None
    item["video_caption"] = _read_caption(video_part)[0] if video_part.is_file() else ""
    item["audio_caption"] = _read_caption(audio_part)[0] if audio_part.is_file() else ""
    key = _flag_key(root, item)
    raw_flag = flags.get(key) if key else None
    if isinstance(raw_flag, Mapping):
        raw_flag = raw_flag.get("flag", raw_flag.get("approved"))
    flag = str(raw_flag).casefold() if raw_flag is not None else ""
    if flag in {"approved", "approve", "true", "1", "yes"}:
        item["flag"] = "approved"
    elif flag in {"rejected", "reject", "false", "0", "no"}:
        item["flag"] = "rejected"
    return item


def scan_folder(folder: str | os.PathLike[str], recursive: bool = False) -> list[EditorItem]:
    """Pair media and captions without probing or generating thumbnails."""

    root = normalize_path(folder, must_exist=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    if recursive:
        media_files = list_media_files(
            root, recursive=True, kinds=("video", "audio", "image")
        )
    else:
        media_files = sorted(
            {
                path
                for directory in _scan_directories(root, False)
                for path in list_media_files(
                    directory, recursive=False, kinds=("video", "audio", "image")
                )
            },
            key=lambda value: (natural_sort_key(value), str(value).casefold(), str(value)),
        )
    caption_files = _walk_caption_files(root, recursive)
    batch_lookup = _batch_caption_lookup(root)
    run_records = _read_batch_records(root)
    captions_by_key: dict[tuple[str, str], list[Path]] = {}
    for path in caption_files:
        captions_by_key.setdefault((str(path.parent).casefold(), path.stem.casefold()), []).append(path)
    for candidates in captions_by_key.values():
        candidates.sort(key=lambda value: CAPTION_EXTENSIONS.index(value.suffix.casefold()))

    flags = read_flags(root)
    failed_paths, failed_names = _failed_media(root, recursive)
    used_captions: set[str] = set()
    generic_assignments: set[str] = set()
    direct_counts: dict[str, int] = {}
    for media in media_files:
        key = str(media.parent).casefold()
        direct_counts[key] = direct_counts.get(key, 0) + 1

    items: list[EditorItem] = []
    for media in media_files:
        exact = captions_by_key.get((str(media.parent).casefold(), media.stem.casefold()), [])
        selected = exact[0] if exact else None
        formats = list(exact)
        if selected is None:
            generic_parents = [media.parent]
            if media.parent.name.casefold() == "clips":
                generic_parents.append(media.parent.parent)
            for parent in generic_parents:
                generic = captions_by_key.get((str(parent).casefold(), "caption"), [])
                parent_key = str(parent).casefold()
                direct_ok = direct_counts.get(str(media.parent).casefold(), 0) == 1
                if generic and parent_key not in generic_assignments and (parent != media.parent or direct_ok):
                    selected, formats = generic[0], list(generic)
                    generic_assignments.add(parent_key)
                    break
        if selected is None:
            selected, formats = media.with_suffix(".txt"), []
        else:
            used_captions.update(str(path.resolve(strict=False)).casefold() for path in formats)
        item = _item_from_pair(media, selected, formats, root, flags, failed_paths, failed_names)
        _enrich_segment_item(item, root)
        items.append(item)

    for caption_path in caption_files:
        if str(caption_path.resolve(strict=False)).casefold() in used_captions:
            continue
        formats = captions_by_key[(str(caption_path.parent).casefold(), caption_path.stem.casefold())]
        if caption_path != formats[0]:
            continue
        used_captions.update(str(path.resolve(strict=False)).casefold() for path in formats)
        resolved_media, source_media = resolve_media_from_metadata(
            caption_path, root, batch_lookup
        )
        item = _item_from_pair(
            resolved_media,
            caption_path,
            formats,
            root,
            flags,
            failed_paths,
            failed_names,
        )
        if source_media is not None:
            item["source_media_path"] = str(source_media)
        _enrich_segment_item(item, root)
        items.append(item)

    # Numbered batch run directories keep captions in mirrored/next-to-source
    # destinations. Their index still provides a complete editor queue.
    records_by_window: dict[tuple[str, Any, Any], list[dict[str, Any]]] = {}
    for record in run_records:
        window_key = (
            _path_identity(str(record["source_path"])),
            record.get("start_s"),
            record.get("end_s"),
        )
        records_by_window.setdefault(window_key, []).append(record)
    preferred_records: list[dict[str, Any]] = []
    for records in records_by_window.values():
        main = [
            record
            for record in records
            if str(record.get("output_key") or "").casefold()
            not in {"video_caption", "audio_caption"}
        ]
        video = [
            record
            for record in records
            if str(record.get("output_key") or "").casefold() == "video_caption"
        ]
        preferred_records.extend(main or video or records)

    run_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in preferred_records:
        caption_path = normalize_path(str(record["caption_path"]))
        if not caption_path.is_file():
            continue
        key = (_path_identity(caption_path.parent), caption_path.stem.casefold())
        run_groups.setdefault(key, []).append(record)
    for records in run_groups.values():
        records.sort(
            key=lambda value: CAPTION_EXTENSIONS.index(
                Path(str(value["caption_path"])).suffix.casefold()
            )
        )
        selected_record = records[0]
        caption_path = normalize_path(str(selected_record["caption_path"]))
        if _path_identity(caption_path) in used_captions:
            continue
        formats = [normalize_path(str(value["caption_path"])) for value in records]
        used_captions.update(_path_identity(value) for value in formats)
        source = normalize_path(str(selected_record["source_path"]))
        item = _item_from_pair(
            source if source.is_file() else None,
            caption_path,
            formats,
            root,
            flags,
            failed_paths,
            failed_names,
        )
        item["source_media_path"] = str(source)
        item["batch_run_item"] = True
        window = _finite_window(selected_record)
        if window:
            item.update(start_s=window[0], end_s=window[1])
        items.append(item)

    def item_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
        raw = str(item.get("media_path") or item.get("caption_path") or "")
        try:
            relative = normalize_path(raw).relative_to(root)
        except (OSError, ValueError):
            relative = Path(raw)
        return tuple(natural_sort_key(part) for part in relative.parts)

    return sorted(items, key=item_key)


def _optional_int(value: Any) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Filter values must be finite")
    integer = int(number)
    return None if integer <= 0 else integer


def _matches_filter(item: Mapping[str, Any], spec: Mapping[str, Any]) -> bool:
    search = str(spec.get("search") or "")
    haystack = f"{item.get('media_path') or item.get('caption_path') or ''}\n{item.get('caption') or ''}"
    if search:
        if spec.get("regex"):
            if re.search(search, haystack, flags=re.IGNORECASE) is None:
                return False
        elif search.casefold() not in haystack.casefold():
            return False
    chars, tokens = int(item.get("chars") or 0), int(item.get("tokens") or 0)
    limits = (
        (_optional_int(spec.get("min_length")), chars, "min"),
        (_optional_int(spec.get("max_length")), chars, "max"),
        (_optional_int(spec.get("min_tokens")), tokens, "min"),
        (_optional_int(spec.get("max_tokens")), tokens, "max"),
    )
    if any(limit is not None and ((mode == "min" and value < limit) or (mode == "max" and value > limit)) for limit, value, mode in limits):
        return False
    wanted_flag = str(spec.get("flag") or "all").casefold()
    if wanted_flag == "unflagged" and item.get("flag") is not None:
        return False
    if wanted_flag in {"approved", "rejected"} and item.get("flag") != wanted_flag:
        return False
    wanted_status = str(spec.get("status") or "all").casefold()
    if wanted_status == "over token limit":
        return tokens > int(spec.get("token_limit") or 512)
    return wanted_status == "all" or item.get("status") == wanted_status


def filter_items(items: Iterable[Mapping[str, Any]], filter_spec: Mapping[str, Any] | None = None, **updates: Any) -> list[Mapping[str, Any]]:
    """Filter editor items by search, counts, flag, and status."""

    spec = dict(_DEFAULT_FILTER)
    spec.update(dict(filter_spec or {}))
    spec.update(updates)
    if spec.get("search") and spec.get("regex"):
        re.compile(str(spec["search"]), flags=re.IGNORECASE)
    return [item for item in items if _matches_filter(item, spec)]


def filtered_indices(state: Mapping[str, Any]) -> list[int]:
    spec = state.get("filter") or _DEFAULT_FILTER
    return [index for index, item in enumerate(state.get("items") or []) if _matches_filter(item, spec)]


def pagination_math(total_items: int, page: int, page_size: int) -> tuple[int, int, int, int]:
    """Return clamped page, total pages, start, and exclusive end."""

    size, total = max(1, int(page_size)), max(0, int(total_items))
    pages = max(1, math.ceil(total / size))
    selected = min(pages, max(1, int(page)))
    start = (selected - 1) * size
    return selected, pages, start, min(total, start + size)


def paginate_items(items: list[Any], page: int, page_size: int) -> tuple[list[Any], int, int]:
    selected, pages, start, end = pagination_math(len(items), page, page_size)
    return items[start:end], selected, pages


def _segment_clock(seconds: float, *, milliseconds: bool = False) -> str:
    precision = 3 if milliseconds else 1
    factor = 10**precision
    value = math.floor(max(0.0, float(seconds)) * factor + 0.5) / factor
    hours = int(value // 3600)
    minutes = int((value % 3600) // 60)
    remainder = value % 60
    seconds_text = f"{remainder:0{3 + precision}.{precision}f}"
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds_text}"
    return f"{minutes:02d}:{seconds_text}"


def editor_item_label(item: Mapping[str, Any], scan_root: str | os.PathLike[str] | None = None) -> str:
    """Return the stable row/gallery label for an editor item."""

    caption = Path(str(item.get("caption_path") or "caption"))
    if item.get("batch_run_item"):
        source = Path(
            str(item.get("source_media_path") or item.get("media_path") or "source")
        )
        return f"{caption.name} - {source.name}"
    is_segment = item.get("segment_index") is not None or caption.parent.name.casefold().endswith("_segments")
    if is_segment:
        try:
            name = caption.relative_to(normalize_path(scan_root)).as_posix() if scan_root else caption.as_posix()
        except (OSError, ValueError):
            name = f"{caption.parent.name}/{caption.name}"
        if item.get("start_s") is not None and item.get("end_s") is not None:
            name += f" · {_segment_clock(float(item['start_s']))}–{_segment_clock(float(item['end_s']))}"
        return name
    return Path(str(item.get("media_path") or caption)).name


def _page_rows(state: EditorState) -> tuple[list[list[Any]], str]:
    indices = filtered_indices(state)
    page, pages, start, end = pagination_math(len(indices), int(state.get("page", 1)), int(state.get("page_size", 25)))
    state["page"] = page
    rows: list[list[Any]] = []
    token_limit = int((state.get("filter") or {}).get("token_limit") or 512)
    selected = state.get("selected_index")
    for global_index in indices[start:end]:
        item = state["items"][global_index]
        name = editor_item_label(item, state.get("folder"))
        preview = re.sub(r"\s+", " ", str(item.get("caption") or "")).strip()
        preview = preview if len(preview) <= 120 else preview[:117].rstrip() + "..."
        token_count = int(item.get("tokens") or 0)
        flag = str(item.get("flag") or "-")
        if token_count > token_limit:
            flag = f"⚠ {flag}" if flag != "-" else "⚠"
        row_number = f"▶ {global_index + 1}" if selected is not None and int(selected) == global_index else str(global_index + 1)
        rows.append([row_number, name, preview, int(item.get("chars") or 0), token_count, flag, item.get("status") or "empty"])
    showing = "0" if not indices else f"{start + 1}-{end}"
    return rows, f"**Page {page} / {pages}** · showing {showing} of {len(indices)}"


def _counter_markdown(state: Mapping[str, Any]) -> str:
    items = state.get("items") or []
    captioned = sum(bool(str(item.get("caption") or "").strip()) for item in items)
    approved = sum(item.get("flag") == "approved" for item in items)
    failed = sum(item.get("status") == "failed" for item in items)
    return f"**{len(items)} items** · {captioned} captioned · {approved} approved · {failed} failed"


def _state_token_limit(state: Mapping[str, Any] | None) -> int:
    return max(1, int(((state or {}).get("filter") or {}).get("token_limit") or 512))


def _stats_markdown(item: Mapping[str, Any] | None, token_limit: int = 512) -> str:
    if not item:
        return "**No item selected.**"
    stats = caption_stats(str(item.get("caption") or ""))
    raw = str(
        item.get("media_path")
        or item.get("source_media_path")
        or item.get("caption_path")
        or ""
    )
    duration = item.get("duration")
    duration_text = f" · {float(duration):.2f}s" if duration is not None else ""
    line = f"**{stats['chars']} chars** · {stats['words']} words · ~{stats['approx_tokens']} tokens{duration_text}<br>`{html.escape(raw)}`"
    limit = max(1, int(token_limit or 512))
    if int(stats["approx_tokens"]) > limit:
        line += (
            f"<br><span class='vc-warn'>⚠ {int(stats['approx_tokens'])} tokens > limit {limit} "
            "(may be truncated by the trainer's text encoder)</span>"
        )
    return line


def _refresh_item(item: EditorItem, caption: str) -> None:
    item["caption"] = str(caption)
    stats = caption_stats(str(caption))
    item["chars"], item["tokens"] = stats["chars"], stats["approx_tokens"]
    if item.get("status") != "failed":
        item["status"] = "no media" if not item.get("media_path") else ("ok" if str(caption).strip() else "empty")


def _selection_payload(
    state: EditorState,
    cache_dir: Path,
    *,
    load_preview: bool = True,
) -> tuple[Any, Any, Any, Any, str, str]:
    selected = state.get("selected_index")
    if selected is None or not (0 <= int(selected) < len(state.get("items") or [])):
        return (
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            gr.update(value="No preview selected.", visible=True),
            "",
            _stats_markdown(None),
        )
    item = state["items"][int(selected)]
    video, audio, image = gr.update(value=None, visible=False), gr.update(value=None, visible=False), gr.update(value=None, visible=False)
    media_path = item.get("media_path")
    source_media_path = item.get("source_media_path") or media_path
    placeholder = gr.update(value="", visible=False)
    if not media_path:
        if source_media_path:
            placeholder = gr.update(
                value=f"Source media not found: {source_media_path}",
                visible=True,
            )
        else:
            placeholder = gr.update(value="No media available for this caption.", visible=True)
    elif not Path(media_path).is_file():
        placeholder = gr.update(value=f"Source media not found: {media_path}", visible=True)
    if load_preview and media_path and Path(media_path).is_file():
        try:
            info = probe_media(media_path)
            item["duration"], item["kind"] = info.duration, ("video" if info.has_video else info.kind)
            safe_path = preview_safe_media(media_path, cache_dir)
            safe = str(safe_path)
            if info.has_video:
                if safe_path.suffix.casefold() in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
                    image = gr.update(value=safe, visible=True)
                    placeholder = gr.update(
                        value="Preview shows the first frame; this video is not browser-playable.",
                        visible=True,
                    )
                else:
                    video = gr.update(value=safe, visible=True)
            elif info.kind == "audio":
                audio = gr.update(value=safe, visible=True)
            elif info.kind == "image":
                image = gr.update(value=safe, visible=True)
            else:
                placeholder = gr.update(value="Media preview unavailable.", visible=True)
        except Exception:
            placeholder = gr.update(value="Media preview unavailable.", visible=True)
    state["dirty"], state["draft_caption"] = False, None
    token_limit = _state_token_limit(state)
    return video, audio, image, placeholder, str(item.get("caption") or ""), _stats_markdown(item, token_limit)


def caption_parts_payload(state: EditorState | Mapping[str, Any] | None) -> tuple[str, str]:
    """Return the selected item's read-only video and audio caption parts."""

    current = dict(state or {})
    items = list(current.get("items") or [])
    selected = current.get("selected_index")
    if selected is None or not (0 <= int(selected) < len(items)):
        return "", ""
    item = items[int(selected)]
    return str(item.get("video_caption") or ""), str(item.get("audio_caption") or "")


def _thumbnail_identity(path: Path) -> str:
    try:
        stat = path.stat()
        raw = f"{path.resolve(strict=False)}\0{stat.st_size}\0{stat.st_mtime_ns}"
    except OSError:
        raw = str(path.resolve(strict=False))
    return hashlib.sha256(raw.encode("utf-8", errors="surrogatepass")).hexdigest()[:28]


def _editor_icon(path: Path, target: Path, kind: str) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (320, 190), (32, 38, 48))
    draw = ImageDraw.Draw(image)
    color = (77, 184, 160) if kind == "audio" else (123, 139, 168)
    draw.rounded_rectangle((82, 28, 238, 162), radius=18, fill=(46, 55, 69), outline=color, width=4)
    symbol = "AUDIO" if kind == "audio" else "CAPTION"
    draw.text((160, 96), symbol, fill=(232, 238, 245), anchor="mm")
    image.save(target, format="PNG")
    return target


def editor_thumbnail(item: Mapping[str, Any], cache_dir: str | os.PathLike[str]) -> Path:
    """Return a path+mtime cached thumbnail for one editor item."""

    raw = item.get("media_path") or item.get("source_media_path") or item.get("caption_path")
    source = Path(str(raw or "caption"))
    cache = normalize_path(cache_dir)
    target = cache / f"{_thumbnail_identity(source)}_thumb.png"
    if target.is_file() and target.stat().st_size:
        return target
    kind = str(item.get("kind") or guess_kind_by_extension(source))
    if source.is_file() and kind in {"video", "video_no_audio", "image"}:
        try:
            return make_thumbnail(source, target, at_seconds=0.0, width=320)
        except Exception:
            pass
    return _editor_icon(source, target, "audio" if kind == "audio" else "caption")


def editor_page_gallery(
    state: Mapping[str, Any],
    cache_dir: str | os.PathLike[str],
) -> tuple[list[tuple[str, str]], int | None]:
    """Build current-page gallery entries and the synchronized local selection."""

    indices = filtered_indices(state)
    _, _, start, end = pagination_math(
        len(indices), int(state.get("page", 1)), int(state.get("page_size", 25))
    )
    page_indices = indices[start:end]
    values: list[tuple[str, str]] = []
    for global_index in page_indices:
        item = (state.get("items") or [])[global_index]
        name = editor_item_label(item, state.get("folder"))
        excerpt = re.sub(r"\s+", " ", str(item.get("caption") or "")).strip()
        if len(excerpt) > 88:
            excerpt = excerpt[:85].rstrip() + "..."
        label = name + (f"\n{excerpt}" if excerpt else "\nNo caption")
        values.append((str(editor_thumbnail(item, cache_dir)), label))
    selected = state.get("selected_index")
    local = page_indices.index(int(selected)) if selected is not None and int(selected) in page_indices else None
    return values, local


def _scope_indices(state: Mapping[str, Any], scope: str) -> list[int]:
    return filtered_indices(state) if str(scope).casefold().startswith("filtered") else list(range(len(state.get("items") or [])))


def _count_matches(text: str, find: str, regex: bool, case_sensitive: bool, whole_words: bool) -> int:
    expression = find if regex else re.escape(find)
    if whole_words:
        expression = rf"(?<!\w)(?:{expression})(?!\w)"
    return len(re.findall(expression, text, flags=0 if case_sensitive else re.IGNORECASE))


def find_replace_preview(
    items: Iterable[Mapping[str, Any]],
    find: str,
    replacement: str,
    *,
    regex: bool = False,
    case_sensitive: bool = False,
    whole_words: bool = False,
    limit: int = 10,
) -> dict[str, Any]:
    """Return match/file counts and the first ten changed captions."""

    needle = str(find)
    if not needle:
        raise ValueError("Find text cannot be empty")
    pairs = parse_replace_pairs([(needle, str(replacement))]) or [(needle, str(replacement))]
    changed_files, matches, previews = 0, 0, []
    for item in items:
        old = str(item.get("caption") or "")
        count = _count_matches(old, needle, regex, case_sensitive, whole_words)
        if not count:
            continue
        new = apply_replacements(old, pairs, regex=regex, case_insensitive=not case_sensitive, whole_words=whole_words)
        if new == old:
            continue
        changed_files, matches = changed_files + 1, matches + count
        if len(previews) < max(0, int(limit)):
            previews.append({"path": str(item.get("media_path") or item.get("caption_path") or ""), "old": old, "new": new})
    return {"files_changed": changed_files, "replacement_count": matches, "previews": previews}


preview_find_replace = find_replace_preview


def _preview_html(result: Mapping[str, Any]) -> str:
    content = f"<p><strong>{int(result.get('replacement_count', 0))} replacements</strong> across {int(result.get('files_changed', 0))} files.</p>"
    for preview in result.get("previews", []):
        content += f"<h4>{html.escape(Path(str(preview['path'])).name)}</h4>{diff_html(str(preview['old']), str(preview['new']))}"
    return content


def editor_item_modality(item: Mapping[str, Any] | None) -> str:
    """Return the prompt-preset modality for an editor item."""

    if not item:
        return "video"
    kind = str(item.get("kind") or "")
    raw = item.get("media_path")
    if kind in {"video", "video_no_audio"} or raw:
        try:
            info = probe_media(str(raw))
            if info.has_video:
                return "video_audio" if info.has_audio else "video"
            kind = info.kind
        except Exception:
            pass
    return kind if kind in {"audio", "image", "text"} else "video"


def regeneration_prompt_choices(
    variant_key: str,
    item: Mapping[str, Any] | None,
    current: str | None = None,
) -> tuple[list[tuple[str, str]], str | None]:
    """Filter regeneration prompts and retain a valid current selection."""

    choices, selected, _ = resolve_regeneration_prompt_choices(
        variant_key,
        item,
        current,
    )
    return choices, selected


def resolve_regeneration_prompt_choices(
    variant_key: str,
    item: Mapping[str, Any] | None,
    current: str | None = None,
) -> tuple[list[tuple[str, str]], str | None, str | None]:
    """Return compatible choices, a choice-safe value, and an unavailable message."""

    family = variant_to_family(str(variant_key))
    modality = editor_item_modality(item)
    presets = list_presets(family, modality)
    choices = [(f"{preset.group} · {preset.label}", preset.id) for preset in presets]
    ids = {preset.id for preset in presets}
    if not choices:
        model_label = MODEL_SPECS[family].label
        message = (
            f"No prompt preset supports {modality} input with {model_label}; "
            "pick another model"
        )
        return [], None, message

    selected = str(current or "")
    if selected not in ids:
        try:
            default_id = default_preset_for(family, modality).id
        except Exception:
            default_id = ""
        selected = default_id if default_id in ids else presets[0].id
    return choices, selected, None


def _handler_error(total_outputs: int, state: Any, message: str) -> tuple[Any, ...]:
    """Build a complete Gradio handler tuple ending in one status message."""

    return state, *[gr.skip() for _ in range(max(0, total_outputs - 2))], message


def editor_filter_handler(
    current: EditorState,
    values: Iterable[Any],
    *,
    initial_state: EditorState | None = None,
    preview_cache: str | os.PathLike[str] = ".",
) -> tuple[Any, ...]:
    """Apply the editor filter and always return its complete ten outputs."""

    raw = list(values)
    if len(raw) not in {8, 9}:
        return _handler_error(10, current, "<span class='vc-err'>Invalid filter input count.</span>")
    next_state = deepcopy(current or initial_state or new_editor_state())
    next_state["filter"] = {
        "search": raw[0], "regex": bool(raw[1]), "min_length": raw[2],
        "max_length": raw[3], "min_tokens": raw[4], "max_tokens": raw[5],
        "flag": raw[6], "status": raw[7],
        "token_limit": raw[8] if len(raw) == 9 else (current.get("filter") or {}).get("token_limit", 512),
    }
    try:
        next_state["filter"]["token_limit"] = _optional_int(next_state["filter"].get("token_limit")) or 512
        if next_state["filter"]["search"] and next_state["filter"]["regex"]:
            re.compile(str(next_state["filter"]["search"]), flags=re.IGNORECASE)
        matches = filtered_indices(next_state)
    except (ValueError, re.error) as exc:
        return _handler_error(
            10,
            current,
            f"<span class='vc-err'>Invalid filter: {html.escape(str(exc))}</span>",
        )
    next_state["page"] = 1
    next_state["selected_index"] = matches[0] if matches else None
    rows, page_text = _page_rows(next_state)
    return (
        next_state,
        rows,
        page_text,
        *_selection_payload(next_state, Path(preview_cache), load_preview=False),
        f"Filter matched {len(matches)} item(s).",
    )


def editor_flag_handler(
    current: EditorState,
    flag: str,
    *,
    default_folder: str | os.PathLike[str] = ".",
    preview_cache: str | os.PathLike[str] = ".",
) -> tuple[Any, ...]:
    """Persist an approval flag and always return its complete eleven outputs."""

    next_state = deepcopy(current or new_editor_state(default_folder))
    selected = next_state.get("selected_index")
    if selected is None or not (0 <= int(selected) < len(next_state.get("items") or [])):
        return _handler_error(11, next_state, "No caption is selected.")
    indices_before = filtered_indices(next_state)
    old_position = indices_before.index(int(selected)) if int(selected) in indices_before else 0
    item = next_state["items"][int(selected)]
    if next_state.get("dirty"):
        draft = str(next_state.get("draft_caption") or "")
        _write_caption(Path(str(item["caption_path"])), draft, item.get("caption_field"))
        _refresh_item(item, draft)
        next_state["dirty"], next_state["draft_caption"] = False, None
    root = normalize_path(next_state.get("folder") or default_folder)
    key = _flag_key(root, item)
    if key is None:
        return _handler_error(
            11,
            next_state,
            "<span class='vc-err'>Selected item is outside the scanned folder.</span>",
        )
    flags = read_flags(root)
    flags[key] = flag
    write_flags(root, flags)
    item["flag"] = flag
    remaining = filtered_indices(next_state)
    if remaining:
        position = min(
            len(remaining) - 1,
            (remaining.index(int(selected)) + 1) if int(selected) in remaining else old_position,
        )
        next_state["selected_index"] = remaining[position]
    rows, page_text = _page_rows(next_state)
    message = f"Marked {Path(str(item.get('media_path') or item['caption_path'])).name} {flag}."
    return (
        next_state,
        rows,
        _counter_markdown(next_state),
        page_text,
        *_selection_payload(next_state, Path(preview_cache)),
        message,
    )


def editor_save_handler(
    current: EditorState,
    text: str,
    *,
    initial_state: EditorState | None = None,
    quiet: bool = False,
) -> tuple[Any, ...]:
    """Atomically save the selected caption and return all five UI outputs."""

    next_state = deepcopy(current or initial_state or new_editor_state())
    selected = next_state.get("selected_index")
    if selected is None or not (0 <= int(selected) < len(next_state.get("items") or [])):
        return next_state, gr.skip(), gr.skip(), _stats_markdown(None), "No caption is selected."
    item = next_state["items"][int(selected)]
    try:
        path = Path(str(item["caption_path"]))
        value = str(text or "")
        _write_caption(path, value, item.get("caption_field"))
        _refresh_item(item, value)
        next_state["dirty"], next_state["draft_caption"] = False, None
        rows, _ = _page_rows(next_state)
        message = f"Saved {path}."
        return next_state, rows, _counter_markdown(next_state), _stats_markdown(item, _state_token_limit(next_state)), (gr.skip() if quiet else message)
    except Exception as exc:
        return (
            next_state,
            gr.skip(),
            gr.skip(),
            _stats_markdown(item, _state_token_limit(next_state)),
            f"<span class='vc-err'>{html.escape(str(exc))}</span>",
        )


def editor_export_handler(
    current: EditorState,
    destination: str | os.PathLike[str],
    copy_media: bool = True,
    extension: str = ".txt",
    include_caption_only: bool = False,
    create_zip: bool = False,
) -> str:
    """Export approved editor items and return a UI-ready summary."""

    try:
        report = export_dataset(
            current.get("items") or [],
            destination,
            only_approved=True,
            copy_media=bool(copy_media),
            caption_ext=extension,
            include_caption_only=bool(include_caption_only),
        )
        message = (
            f"Exported {report.exported} approved item(s); no-media {report.no_media}; "
            f"not-approved {report.not_approved}; errors {report.error_count}."
        )
        if report.segment_full_source_fallbacks:
            message += (
                f" {report.segment_full_source_fallbacks} segment captions exported against the full source "
                "because no clip window was recorded."
            )
        if report.errors:
            message += " " + " | ".join(report.errors[:3])
        if create_zip:
            archive_path = zip_directory(report.out_root, Path(str(report.out_root) + ".zip"))
            size_mib = archive_path.stat().st_size / float(1024**2)
            message += f" ZIP: {archive_path} ({size_mib:.2f} MiB)."
        return message
    except Exception as exc:
        return f"<span class='vc-err'>{html.escape(str(exc))}</span>"


def _regeneration_formats(item: Mapping[str, Any]) -> tuple[str, ...]:
    paths = [Path(str(item.get("caption_path") or "caption.txt"))]
    paths.extend(Path(str(path)) for path in item.get("caption_formats") or [])
    formats = [
        path.suffix.casefold().lstrip(".")
        for path in paths
        if path.suffix.casefold().lstrip(".") in {"txt", "json", "srt", "vtt", "jsonl"}
    ]
    return tuple(dict.fromkeys(formats or ["txt"]))


def editor_regeneration_log(item: Mapping[str, Any]) -> str:
    source = Path(str(item.get("source_media_path") or item.get("media_path") or "media"))
    index = int(item.get("segment_index") or 0)
    if index and item.get("start_s") is not None and item.get("end_s") is not None:
        return (
            f"Regenerating clip {index} "
            f"({_segment_clock(float(item['start_s']), milliseconds=True)}–"
            f"{_segment_clock(float(item['end_s']), milliseconds=True)}) of {source.name}"
        )
    return f"Regenerating {Path(str(item.get('caption_path') or source)).name}"


def build_editor_regeneration_spec(
    settings: Mapping[str, Any],
    item: Mapping[str, Any],
    *,
    variant: str,
    prompt_id: str,
    override: str = "",
    outputs_root: str | os.PathLike[str],
) -> JobSpec:
    """Build a one-item regeneration job, constraining known segments to their clip."""

    caption_path = Path(str(item["caption_path"]))
    formats = _regeneration_formats(item)
    produced_raw = item.get("segment_media_path")
    produced = Path(str(produced_raw)) if produced_raw else None
    known_window = item.get("start_s") is not None and item.get("end_s") is not None
    if produced is not None and produced.is_file():
        media_path = produced
        trim_start, trim_end = 0.0, None
    else:
        media_raw = item.get("source_media_path") or item.get("media_path")
        if not media_raw:
            raise ValueError("The selected caption has no media to regenerate.")
        media_path = Path(str(media_raw))
        trim_start = float(item["start_s"]) if known_window else float(settings.get("trim_start_s") or 0.0)
        raw_trim_end = float(item["end_s"]) if known_window else settings.get("trim_end_s")
        trim_end = float(raw_trim_end) if raw_trim_end not in (None, "") else None
    if not media_path.is_file():
        raise FileNotFoundError(f"Media not found: {media_path}")

    merged = dict(settings)
    merged.update(
        model_key=variant,
        variant_key=variant,
        prompt_preset_id=prompt_id,
        system_prompt="",
        overwrite_existing=True,
        output_formats=list(formats),
        trim_start_s=trim_start,
        trim_end_s=trim_end,
        segment_mode="whole",
        scene_detect_enabled=False,
        audio_caption_source="none",
        video_caption_source="generate",
    )
    override_prompt = str(override or "").strip()
    if override_prompt:
        merged["user_prompt"] = override_prompt
    else:
        merged.pop("user_prompt", None)
    output = OutputSpec(
        kind="batch",
        outputs_root=str(outputs_root),
        batch_output_dir=str(caption_path.parent),
        mirror_names=False,
        overwrite=True,
    )
    spec = JobSpec.from_settings(
        merged,
        [
            InputItem(
                path=str(media_path),
                trim_start_s=trim_start,
                trim_end_s=trim_end,
            )
        ],
        output,
    )
    return replace(
        spec,
        preprocess=replace(spec.preprocess, trim_start_s=trim_start, trim_end_s=trim_end),
        split=replace(spec.split, mode="whole"),
        post=replace(spec.post, formats=formats),
        internal={
            **dict(spec.internal or {}),
            "output_dirs": [str(caption_path.parent)],
            "output_stems": [caption_path.stem],
            "metadata_name": "editor_regeneration_metadata.json",
            "continue_on_error": True,
        },
    )


def rebuild_caption_parts_after_regeneration(
    settings: Mapping[str, Any],
    item: Mapping[str, Any],
    generated_video_caption: str,
) -> str:
    """Update the clean video part and rebuild a merged editor caption."""

    raw_video_path = item.get("video_caption_path")
    if not raw_video_path:
        return str(generated_video_caption)
    video_path = Path(str(raw_video_path))
    caption_path = Path(str(item["caption_path"]))
    video_text = str(generated_video_caption).strip()
    _write_caption(video_path, video_text)
    raw_audio_path = item.get("audio_caption_path")
    audio_text = ""
    if raw_audio_path and Path(str(raw_audio_path)).is_file():
        audio_text = _read_caption(Path(str(raw_audio_path)))[0]
    merged = render_caption_template(
        str(settings.get("caption_merge_template") or DEFAULT_CAPTION_MERGE_TEMPLATE),
        {
            "VIDEO_CAPTION": video_text,
            "AUDIO_CAPTION": audio_text,
            "TRANSCRIPT": "",
            "SOUND_CAPTION": "",
            "FILENAME": caption_path.stem,
        },
    )
    _write_caption(caption_path, merged, item.get("caption_field"))
    return merged


def build(ctx: "UiContext") -> None:
    """Render and wire the full Caption Editor tab."""

    initial_state = new_editor_state(ctx.outputs_dir)
    state = gr.State(initial_state)
    regeneration_backup = gr.State({})
    autosave_timer = gr.Timer(0.5)
    preview_cache = ctx.temp_dir / "editor_previews"
    registry = ctx.settings_registry
    model_entry = next((entry for entry in registry.entries() if entry.key == "model_key"), None)

    with gr.Row():
        folder = gr.Textbox(
            value=str(ctx.outputs_dir), label="Caption folder",
            info="Scan media/caption sidecars or SECourses run directories.", scale=7,
        )
        scan = action_button("Scan", "cyan", scale=1, min_width=92)
        open_folder = action_button("📂 Open folder", "amber", scale=1, min_width=118)
        reveal_selected = action_button("📍 Reveal file", "crimson", scale=1, min_width=124)
        recursive = gr.Checkbox(
            value=False, label="Recursive",
            info="Include nested batch folders and run clip directories.", scale=1,
        )

    with gr.Accordion("Filters", open=False):
        with gr.Row():
            search = gr.Textbox(label="Contains", info="Search file paths and caption text.", scale=4)
            search_regex = gr.Checkbox(label="Regex", value=False, info="Interpret Contains as a regular expression.")
            flag_filter = gr.Dropdown(
                choices=[("All", "all"), ("Approved", "approved"), ("Rejected", "rejected"), ("Unflagged", "unflagged")],
                value="all", label="Flag", info="Limit the review queue by approval state.",
            )
            status_filter = gr.Dropdown(
                choices=[
                    ("All", "all"),
                    ("Empty", "empty"),
                    ("Failed", "failed"),
                    ("No media", "no media"),
                    ("Over token limit", "over token limit"),
                ],
                value="all", label="Status", info="Show empty or failed captions only.",
            )
        with gr.Row():
            min_length = gr.Number(label="Min chars", precision=0, minimum=0, info="Minimum caption length; 0 = no limit.")
            max_length = gr.Number(label="Max chars", precision=0, minimum=0, info="Maximum caption length; 0 = no limit.")
            min_tokens = gr.Number(label="Min tokens", precision=0, minimum=0, info="Minimum approximate token count; 0 = no limit.")
            max_tokens = gr.Number(label="Max tokens", precision=0, minimum=0, info="Maximum approximate token count; 0 = no limit.")
            token_limit = gr.Number(
                value=512,
                label="Token limit (warn)",
                precision=0,
                minimum=1,
                info="Warn and filter captions that may exceed the trainer text encoder limit.",
                elem_id="vc_editor_token_limit",
            )
            apply_filters = action_button("Apply filters", "blue", min_width=126)

    status = gr.Markdown("Ready to scan.", elem_classes=["vc-status"])
    with gr.Row(equal_height=False):
        with gr.Column(scale=5, min_width=520):
            view_mode = gr.Radio(
                choices=["Table", "Gallery"],
                value="Table",
                label="View",
                elem_id="vc_editor_view_mode",
            )
            table = gr.Dataframe(
                value=[], headers=_TABLE_HEADERS,
                datatype=["str", "str", "str", "number", "number", "str", "str"],
                type="array", interactive=False, show_search="none", max_height=520,
                pinned_columns=2, static_columns=list(range(7)),
                column_widths=[55, 180, "46%", 70, 75, 90, 100], wrap=False,
                buttons=["copy", "fullscreen"], label="Review queue",
                elem_id="vc_editor_review_table",
            )
            gallery = gr.Gallery(
                value=[],
                label="Review gallery",
                columns=4,
                rows=3,
                height=520,
                object_fit="cover",
                allow_preview=False,
                visible=False,
                elem_id="vc_editor_gallery",
            )
            counters = gr.Markdown(_counter_markdown(initial_state))
            with gr.Row():
                previous_page = action_button("Prev page", "indigo", scale=1)
                page_label = gr.Markdown("**Page 1 / 1** · showing 0 of 0", scale=3)
                next_page = action_button("Next page", "sky", scale=1)
                page_size = gr.Dropdown(
                    choices=[10, 25, 50, 100, 200], value=25, label="Per page",
                    info="Rows shown on each editor page.", scale=1, min_width=105,
                )

        with gr.Column(scale=5, min_width=480):
            preview_placeholder = gr.Markdown(
                "No preview selected.",
                elem_classes=["vc-status"],
            )
            video = gr.Video(label="Video preview", visible=False, interactive=False, height=390)
            audio = gr.Audio(label="Audio preview", visible=False, interactive=False)
            image = gr.Image(label="Image preview", visible=False, interactive=False, type="filepath", height=390)
            caption = gr.Textbox(
                label="Caption", lines=12, max_lines=18, buttons=["copy"],
                info="Edits are saved after a short pause when autosave is enabled.",
            )
            with gr.Accordion(
                "Caption parts (read-only)",
                open=False,
                elem_id="vc_editor_caption_parts",
            ):
                video_caption_part = gr.Textbox(
                    label="Video caption",
                    lines=5,
                    max_lines=10,
                    interactive=False,
                    elem_classes=["vc-mono"],
                    elem_id="vc_editor_video_caption_part",
                )
                audio_caption_part = gr.Textbox(
                    label="Audio caption",
                    lines=5,
                    max_lines=10,
                    interactive=False,
                    elem_classes=["vc-mono"],
                    elem_id="vc_editor_audio_caption_part",
                )
            stats = gr.Markdown(_stats_markdown(None))
            with gr.Row():
                autosave = gr.Checkbox(
                    value=True, label="Autosave on edit",
                    info="Atomically save after 0.7 seconds without another edit.", scale=2,
                )
                save = action_button("💾 Save", "green", scale=1)
            with gr.Row():
                previous_item = action_button("⬅ Prev", "violet", scale=1)
                next_item = action_button("➡ Next", "slate", scale=1)
                approve = action_button("✅ Approve", "emerald", scale=1)
                reject = action_button("❌ Reject", "rose", scale=1)
                gr.HTML('<span class="vc-help" title="Left/Right: previous/next; Ctrl+S: save; Ctrl+Enter: approve; Ctrl+Delete: reject">&#9432;</span>', min_width=24)

    # Canonical editor hooks plus aliases consumed by T6's global shortcut map.
    hk_ed_prev = gr.Button("Editor previous", elem_id="hk_ed_prev", visible="hidden")
    hk_ed_next = gr.Button("Editor next", elem_id="hk_ed_next", visible="hidden")
    hk_ed_save = gr.Button("Editor save", elem_id="hk_ed_save", visible="hidden")
    hk_ed_approve = gr.Button("Editor approve", elem_id="hk_ed_approve", visible="hidden")
    hk_ed_reject = gr.Button("Editor reject", elem_id="hk_ed_reject", visible="hidden")
    hk_prev_alias = gr.Button("Editor previous alias", elem_id="hk_prev", visible="hidden")
    hk_next_alias = gr.Button("Editor next alias", elem_id="hk_next", visible="hidden")
    hk_save_alias = gr.Button("Editor save alias", elem_id="hk_save", visible="hidden")

    state.change(
        caption_parts_payload,
        inputs=state,
        outputs=[video_caption_part, audio_caption_part],
        queue=False,
        trigger_mode="multiple",
        show_progress="hidden",
        api_visibility="private",
    )

    with gr.Accordion("🔁 Regenerate selected", open=False) as regeneration_accordion:
        variants = all_variant_choices()
        default_variant = (
            str(model_entry.default)
            if model_entry is not None
            else next((key for _, key in variants if key == "qwen3_omni_instruct_int4"), variants[0][1])
        )
        prompts, initial_regen_prompt, initial_regen_error = resolve_regeneration_prompt_choices(
            default_variant,
            None,
        )
        with gr.Row():
            regen_variant = gr.Dropdown(
                choices=variants, value=default_variant, label="Model variant",
                info="The selected checkpoint replaces only this caption.", scale=3,
            )
            regen_prompt = gr.Dropdown(
                choices=prompts, value=initial_regen_prompt, label="Prompt preset",
                info="Choose a task compatible with the media and model.", scale=3,
            )
        regen_prompt_value = gr.State(initial_regen_prompt)
        regen_prompt_error = gr.State(initial_regen_error or "")
        regen_override = gr.Textbox(
            label="User prompt override", lines=4,
            info="Leave blank to use the selected prompt preset unchanged.",
        )
        with gr.Row():
            regenerate = action_button("Regenerate selected", "fuchsia", elem_id="vc_editor_regenerate_selected")
            regenerate_all = action_button(
                "Regenerate all in current filter", "navy",
                elem_id="vc_editor_regenerate_all",
            )
            keep_new = action_button("Keep new", "lime")
            revert = action_button("Revert", "red")
        regenerate_all_targets = gr.State([])
        with gr.Row(
            visible=False,
            elem_id="vc_editor_regenerate_all_confirmation",
            elem_classes=["vc-confirm-bar"],
        ) as regenerate_all_confirmation:
            regenerate_all_question = gr.Markdown("Regenerate 0 captions?")
            regenerate_all_yes = action_button(
                "✔ Yes, regenerate", "maroon", variant="stop",
                scale=0, min_width=160,
                elem_id="vc_editor_regenerate_all_yes",
            )
            regenerate_all_keep = action_button(
                "✖ Keep current captions", "steel",
                scale=0, min_width=208,
                elem_id="vc_editor_regenerate_all_keep",
            )
        regen_status = gr.Markdown(
            initial_regen_error or "No regeneration is pending.",
            elem_classes=["vc-status"],
        )
        regen_diff = gr.HTML("")

    def update_regeneration_prompts(
        variant_key: str,
        current: EditorState,
        current_prompt: str | None,
        previous_error: str,
    ) -> tuple[Any, str | None, str, Any]:
        selected = (current or {}).get("selected_index")
        items = (current or {}).get("items") or []
        item = items[int(selected)] if selected is not None and 0 <= int(selected) < len(items) else None
        try:
            choices, selected_prompt, message = resolve_regeneration_prompt_choices(
                str(variant_key), item, current_prompt
            )
            status = message or ("No regeneration is pending." if previous_error else gr.skip())
            return (
                gr.update(choices=choices, value=selected_prompt),
                selected_prompt,
                message or "",
                status,
            )
        except Exception as exc:
            message = f"Could not filter regeneration prompts: {exc}"
            return gr.update(choices=[], value=None), None, message, message

    regeneration_prompt_inputs = [
        regen_variant,
        state,
        regen_prompt_value,
        regen_prompt_error,
    ]
    regeneration_prompt_outputs = [
        regen_prompt,
        regen_prompt_value,
        regen_prompt_error,
        regen_status,
    ]

    def with_regeneration_prompt_update(
        result: tuple[Any, ...],
        variant_key: str,
        current_prompt: str | None,
        previous_error: str,
    ) -> tuple[Any, ...]:
        next_state = result[0] if result else initial_state
        return (
            *result,
            *update_regeneration_prompts(
                variant_key,
                next_state,
                current_prompt,
                previous_error,
            ),
        )

    if model_entry is not None:
        def mirror_main_model_to_regeneration(
            variant_key: str,
            current: EditorState,
            current_prompt: str | None,
            previous_error: str,
        ) -> tuple[Any, ...]:
            return (
                gr.update(value=variant_key),
                *update_regeneration_prompts(
                    variant_key,
                    current,
                    current_prompt,
                    previous_error,
                ),
            )

        model_entry.component.change(
            mirror_main_model_to_regeneration,
            inputs=[
                model_entry.component,
                state,
                regen_prompt_value,
                regen_prompt_error,
            ],
            outputs=[regen_variant, *regeneration_prompt_outputs],
            queue=False,
            trigger_mode="multiple",
            show_progress="hidden",
            api_visibility="private",
        )

    regen_prompt.input(
        lambda value: value,
        inputs=regen_prompt,
        outputs=regen_prompt_value,
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )

    regen_variant.change(
        update_regeneration_prompts,
        inputs=regeneration_prompt_inputs,
        outputs=regeneration_prompt_outputs,
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    regeneration_accordion.expand(
        update_regeneration_prompts,
        inputs=regeneration_prompt_inputs,
        outputs=regeneration_prompt_outputs,
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )

    with gr.Accordion("🔎 Find & replace across folder", open=False):
        with gr.Row():
            find_text = gr.Textbox(label="Find", info="Text or regular expression to locate.")
            replace_text = gr.Textbox(label="Replace", info="Replacement text applied by the shared post-processor.")
            replace_scope = gr.Radio(
                choices=["Filtered items", "All items"], value="Filtered items", label="Scope",
                info="Choose the current filtered queue or the complete scan.",
            )
        with gr.Row():
            replace_regex = gr.Checkbox(value=False, label="Regex", info="Treat Find as a regular expression.")
            replace_case = gr.Checkbox(value=False, label="Case sensitive", info="Match letter case exactly.")
            replace_whole = gr.Checkbox(value=True, label="Whole word", info="Exclude matches embedded inside longer words.")
            preview_replace = action_button("Preview", "purple")
            apply_replace = action_button("Apply", "orange")
        replace_result = gr.HTML("")

    with gr.Accordion("➕ Bulk edit", open=False):
        with gr.Row():
            bulk_prefix = gr.Textbox(label="Prefix", info="Text placed before each scoped caption.")
            bulk_suffix = gr.Textbox(label="Suffix", info="Text placed after each scoped caption.")
            bulk_trigger = gr.Textbox(label="Trigger word", info="Optional dataset trigger token.")
            trigger_position = gr.Dropdown(
                choices=[("Prefix", "prefix"), ("Suffix", "suffix"), ("None", "none")],
                value="prefix", label="Trigger position", info="Place the trigger before or after the caption.",
            )
            bulk_scope = gr.Radio(
                choices=["Filtered items", "All items"], value="Filtered items", label="Scope",
                info="Apply changes to the visible queue or every scanned item.",
            )
        with gr.Row():
            bulk_apply = action_button("Apply bulk edit", "teal")
            strip_whitespace = action_button("Strip edges", "yellow")
            collapse_newlines = action_button("Collapse newlines", "bronze")
        bulk_result = gr.Markdown("", elem_classes=["vc-status"])

    with gr.Accordion("📤 Export approved", open=False):
        with gr.Row():
            export_destination = gr.Textbox(
                value=str(ctx.outputs_dir / "approved_dataset"), label="Destination",
                info="Approved media/caption pairs are copied here without deleting sources.", scale=5,
            )
            export_copy_media = gr.Checkbox(value=True, label="Copy media", info="Copy media files along with captions.")
            export_caption_only = gr.Checkbox(
                value=False,
                label="Include caption-only items",
                info="Export approved captions without available media as caption files only.",
            )
            export_extension = gr.Dropdown(
                choices=[".txt", ".json", ".srt"], value=".txt", label="Caption extension",
                info="Extension used for exported captions.",
            )
            export_zip = gr.Checkbox(
                value=False,
                label="Also create ZIP archive",
                info="Write a ZIP archive next to the exported folder.",
                elem_id="vc_editor_export_zip",
            )
            export_button = action_button("Export approved only", "pink")
        export_result = gr.Markdown("", elem_classes=["vc-status"])

    with gr.Accordion(
        "📊 Dataset statistics", open=False,
        elem_id="vc_editor_statistics_accordion",
    ):
        compute_statistics = action_button(
            "📊 Compute statistics", "olive",
            elem_id="vc_editor_compute_statistics",
        )
        statistics_output = gr.Markdown(
            "Scan a folder, then compute statistics for every item in the scan.",
            elem_id="vc_editor_dataset_statistics",
        )

    # Event handlers are kept in this module so their pure transforms remain testable.
    def scan_handler(
        raw_folder: str,
        recurse: bool,
        size: int,
        variant_key: str = default_variant,
        current_prompt: str | None = None,
        previous_error: str = "",
    ) -> tuple[Any, ...]:
        try:
            items = scan_folder(raw_folder, bool(recurse))
            next_state = new_editor_state(normalize_path(raw_folder))
            next_state["items"], next_state["page_size"] = items, int(size or 25)
            matches = filtered_indices(next_state)
            next_state["selected_index"] = matches[0] if matches else None
            rows, page_text = _page_rows(next_state)
            selection = _selection_payload(next_state, preview_cache, load_preview=True)
            message = f"Scanned {len(items)} review item(s) in {next_state['folder']}."
            ctx.app_log.log(message, scope="editor")
            result = (
                next_state,
                rows,
                _counter_markdown(next_state),
                page_text,
                *selection,
                message,
            )
        except Exception as exc:
            empty = new_editor_state(raw_folder)
            ctx.app_log.error(f"Editor scan failed: {exc}", scope="editor")
            result = (
                empty,
                [],
                _counter_markdown(empty),
                "**Page 1 / 1**",
                *_selection_payload(empty, preview_cache, load_preview=False),
                f"<span class='vc-err'>{html.escape(str(exc))}</span>",
            )
        return with_regeneration_prompt_update(
            result,
            variant_key,
            current_prompt,
            previous_error,
        )

    scan.click(
        scan_handler,
        inputs=[
            folder,
            recursive,
            page_size,
            regen_variant,
            regen_prompt_value,
            regen_prompt_error,
        ],
        outputs=[
            state,
            table,
            counters,
            page_label,
            video,
            audio,
            image,
            preview_placeholder,
            caption,
            stats,
            status,
            *regeneration_prompt_outputs,
        ],
        show_progress="minimal", api_visibility="private",
    )
    ctx.states["editor_open_binding"] = {
        "folder": folder,
        "recursive": recursive,
        "scan_fn": scan_handler,
        "inputs": [
            folder,
            recursive,
            page_size,
            regen_variant,
            regen_prompt_value,
            regen_prompt_error,
        ],
        "outputs": [
            state,
            table,
            counters,
            page_label,
            video,
            audio,
            image,
            preview_placeholder,
            caption,
            stats,
            status,
            *regeneration_prompt_outputs,
        ],
    }

    def refresh_gallery(current: EditorState, mode: str = "Gallery") -> Any:
        if str(mode) != "Gallery":
            return gr.skip()
        try:
            values, selected = editor_page_gallery(current or initial_state, preview_cache)
            return gr.update(value=values, selected_index=selected)
        except Exception as exc:
            ctx.app_log.warn(f"Editor gallery refresh failed: {exc}", scope="editor")
            return gr.update(value=[], selected_index=None)

    state.change(
        refresh_gallery,
        inputs=[state, view_mode],
        outputs=gallery,
        queue=False,
        trigger_mode="always_last",
        show_progress="hidden",
        api_visibility="private",
    )
    state.change(
        fn=None,
        inputs=state,
        outputs=[],
        js=r"""
        (value) => {
          const key = `${value?.folder || ''}|${value?.selected_index ?? ''}|${value?.page || 1}`;
          const selectMarkedRow = (attempt) => {
            const root = document.getElementById('vc_editor_review_table');
            const row = Array.from(root?.querySelectorAll('tbody tr') || []).find((item) =>
              (item.querySelector('td')?.textContent || '').trim().startsWith('▶')
            );
            if (row) {
              if (window.__vcapEditorSelectedMarker !== key) {
                window.__vcapEditorSelectedMarker = key;
                (row.querySelector('td') || row).click();
              }
            } else if (attempt < 8) {
              window.setTimeout(() => selectMarkedRow(attempt + 1), 50);
            }
          };
          window.setTimeout(() => selectMarkedRow(0), 0);
          return [];
        }
        """,
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    def toggle_editor_view(mode: str, current: EditorState) -> tuple[Any, Any]:
        gallery_update = refresh_gallery(current, mode)
        if isinstance(gallery_update, dict):
            gallery_update = {**gallery_update, "visible": str(mode) == "Gallery"}
        else:
            gallery_update = gr.update(visible=str(mode) == "Gallery")
        return gr.update(visible=str(mode) == "Table"), gallery_update

    view_mode.change(
        toggle_editor_view,
        inputs=[view_mode, state],
        outputs=[table, gallery],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )

    def open_folder_handler(raw_folder: str) -> str:
        if not str(raw_folder or "").strip():
            return "<span class='vc-warn'>Enter a caption folder first.</span>"
        ok, message = open_in_file_manager(normalize_path(raw_folder))
        return f"<span class='{'vc-ok' if ok else 'vc-err'}'>{html.escape(message)}</span>"

    def reveal_selected_handler(current: EditorState) -> str:
        selected = (current or {}).get("selected_index")
        items = (current or {}).get("items") or []
        if selected is None or not (0 <= int(selected) < len(items)):
            return "<span class='vc-warn'>Select a caption first.</span>"
        item = items[int(selected)]
        raw = item.get("media_path") or item.get("source_media_path")
        if raw and not Path(str(raw)).is_file():
            return f"<span class='vc-warn'>Source media not found: {html.escape(str(raw))}</span>"
        raw = raw or item.get("caption_path")
        ok, message = reveal_in_file_manager(str(raw or ""))
        return f"<span class='{'vc-ok' if ok else 'vc-err'}'>{html.escape(message)}</span>"

    open_folder.click(
        open_folder_handler,
        inputs=folder,
        outputs=status,
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    reveal_selected.click(
        reveal_selected_handler,
        inputs=state,
        outputs=status,
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )

    def filter_handler(
        current: EditorState,
        search_value: str,
        regex_value: bool,
        min_length_value: Any,
        max_length_value: Any,
        min_tokens_value: Any,
        max_tokens_value: Any,
        flag_value: str,
        status_value: str,
        token_limit_value: Any,
        variant_key: str,
        current_prompt: str | None,
        previous_error: str,
    ) -> tuple[Any, ...]:
        result = editor_filter_handler(
            current,
            [
                search_value,
                regex_value,
                min_length_value,
                max_length_value,
                min_tokens_value,
                max_tokens_value,
                flag_value,
                status_value,
                token_limit_value,
            ],
            initial_state=initial_state,
            preview_cache=preview_cache,
        )
        return with_regeneration_prompt_update(
            result,
            variant_key,
            current_prompt,
            previous_error,
        )

    filter_inputs = [
        state,
        search,
        search_regex,
        min_length,
        max_length,
        min_tokens,
        max_tokens,
        flag_filter,
        status_filter,
        token_limit,
        regen_variant,
        regen_prompt_value,
        regen_prompt_error,
    ]
    filter_outputs = [
        state,
        table,
        page_label,
        video,
        audio,
        image,
        preview_placeholder,
        caption,
        stats,
        status,
        *regeneration_prompt_outputs,
    ]
    for trigger in (apply_filters.click, token_limit.change, status_filter.change):
        trigger(
            filter_handler,
            inputs=filter_inputs,
            outputs=filter_outputs,
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        )

    def page_handler(current: EditorState, direction: int, selected_size: int) -> tuple[EditorState, list[list[Any]], str]:
        next_state = deepcopy(current or initial_state)
        next_state["page_size"] = int(selected_size or 25)
        next_state["page"] = int(next_state.get("page", 1)) + int(direction)
        rows, label = _page_rows(next_state)
        return next_state, rows, label

    previous_page.click(lambda current, size: page_handler(current, -1, size), [state, page_size], [state, table, page_label], queue=False, show_progress="hidden", api_visibility="private")
    next_page.click(lambda current, size: page_handler(current, 1, size), [state, page_size], [state, table, page_label], queue=False, show_progress="hidden", api_visibility="private")
    page_size.change(lambda current, size: page_handler(current, 0, size), [state, page_size], [state, table, page_label], queue=False, show_progress="hidden", api_visibility="private")

    def select_handler(
        current: EditorState,
        variant_key: str,
        current_prompt: str | None,
        previous_error: str,
        evt: gr.SelectData,
    ) -> tuple[Any, ...]:
        next_state = deepcopy(current or initial_state)
        row = int(evt.index[0] if isinstance(evt.index, (tuple, list)) else evt.index)
        indices = filtered_indices(next_state)
        _, _, start, end = pagination_math(len(indices), int(next_state.get("page", 1)), int(next_state.get("page_size", 25)))
        page_indices = indices[start:end]
        if 0 <= row < len(page_indices):
            next_state["selected_index"] = page_indices[row]
        selected_number = int(next_state.get("selected_index") or 0) + 1
        result = (
            next_state,
            *_selection_payload(next_state, preview_cache),
            f"Selected item {selected_number}.",
        )
        return with_regeneration_prompt_update(
            result,
            variant_key,
            current_prompt,
            previous_error,
        )

    table.select(
        select_handler,
        inputs=[state, regen_variant, regen_prompt_value, regen_prompt_error],
        outputs=[
            state,
            video,
            audio,
            image,
            preview_placeholder,
            caption,
            stats,
            status,
            *regeneration_prompt_outputs,
        ],
        show_progress="minimal", api_visibility="private",
    )

    def gallery_select_handler(
        current: EditorState,
        variant_key: str,
        current_prompt: str | None,
        previous_error: str,
        evt: gr.SelectData,
    ) -> tuple[Any, ...]:
        next_state = deepcopy(current or initial_state)
        row = int(evt.index[0] if isinstance(evt.index, (tuple, list)) else evt.index)
        indices = filtered_indices(next_state)
        _, _, start, end = pagination_math(
            len(indices), int(next_state.get("page", 1)), int(next_state.get("page_size", 25))
        )
        page_indices = indices[start:end]
        if 0 <= row < len(page_indices):
            next_state["selected_index"] = page_indices[row]
        selected_number = int(next_state.get("selected_index") or 0) + 1
        rows, _ = _page_rows(next_state)
        result = (
            next_state,
            rows,
            *_selection_payload(next_state, preview_cache),
            f"Selected item {selected_number}.",
        )
        return with_regeneration_prompt_update(
            result,
            variant_key,
            current_prompt,
            previous_error,
        )

    gallery.select(
        gallery_select_handler,
        inputs=[state, regen_variant, regen_prompt_value, regen_prompt_error],
        outputs=[
            state,
            table,
            video,
            audio,
            image,
            preview_placeholder,
            caption,
            stats,
            status,
            *regeneration_prompt_outputs,
        ],
        show_progress="minimal",
        api_visibility="private",
    )

    def mark_dirty(current: EditorState, text: str) -> tuple[EditorState, str, list[list[Any]]]:
        next_state = deepcopy(current or initial_state)
        selected = next_state.get("selected_index")
        if selected is not None and 0 <= int(selected) < len(next_state.get("items") or []):
            item = next_state["items"][int(selected)]
            _refresh_item(item, str(text or ""))
            next_state.update(dirty=True, draft_caption=str(text or ""), last_edit=time.monotonic())
            rows, _ = _page_rows(next_state)
            return next_state, _stats_markdown(item, _state_token_limit(next_state)), rows
        rows, _ = _page_rows(next_state)
        return next_state, _stats_markdown(None), rows

    caption.input(
        mark_dirty, inputs=[state, caption], outputs=[state, stats, table],
        queue=False, show_progress="hidden", trigger_mode="always_last", api_visibility="private",
    )

    def save_handler(current: EditorState, text: str, quiet: bool = False) -> tuple[Any, ...]:
        result = editor_save_handler(
            current,
            text,
            initial_state=initial_state,
            quiet=quiet,
        )
        message = result[-1]
        if isinstance(message, str):
            if "vc-err" in message:
                ctx.app_log.error(f"Caption save failed: {message}", scope="editor")
            elif message != "No caption is selected.":
                ctx.app_log.log(message, scope="editor")
        return result

    for trigger in (save.click, hk_ed_save.click, hk_save_alias.click):
        trigger(
            save_handler, inputs=[state, caption], outputs=[state, table, counters, stats, status],
            queue=False, show_progress="hidden", api_visibility="private",
        )

    def autosave_handler(current: EditorState, enabled: bool, textbox_value: str) -> tuple[Any, ...]:
        if not enabled or not current or not current.get("dirty"):
            return (gr.skip(),) * 5
        if time.monotonic() - float(current.get("last_edit") or 0.0) < 0.7:
            return (gr.skip(),) * 5
        saved_draft = str(current.get("draft_caption") or "")
        result = list(save_handler(current, saved_draft, True))
        current_text = str(textbox_value or "")
        if isinstance(result[0], dict):
            saved_state = result[0]
            selected = saved_state.get("selected_index")
            if selected is not None and 0 <= int(selected) < len(saved_state.get("items") or []):
                selected_item = saved_state["items"][int(selected)]
                if current_text != saved_draft:
                    _refresh_item(selected_item, current_text)
                    saved_state.update(
                        dirty=True,
                        draft_caption=current_text,
                        last_edit=time.monotonic(),
                    )
                result[1], _ = _page_rows(saved_state)
                result[2] = _counter_markdown(saved_state)
                result[3] = _stats_markdown(selected_item, _state_token_limit(saved_state))
        return tuple(result)

    ctx.states["editor_mark_dirty_handler"] = mark_dirty
    ctx.states["editor_autosave_handler"] = autosave_handler

    autosave_timer.tick(
        autosave_handler, inputs=[state, autosave, caption], outputs=[state, table, counters, stats, status],
        queue=False, show_progress="hidden", api_visibility="private",
    )

    def navigate_handler(
        current: EditorState,
        direction: int,
        variant_key: str,
        current_prompt: str | None,
        previous_error: str,
    ) -> tuple[Any, ...]:
        next_state = deepcopy(current or initial_state)
        if next_state.get("dirty") and next_state.get("selected_index") is not None:
            selected_item = next_state["items"][int(next_state["selected_index"])]
            _write_caption(Path(str(selected_item["caption_path"])), str(next_state.get("draft_caption") or ""), selected_item.get("caption_field"))
            next_state["dirty"] = False
        indices = filtered_indices(next_state)
        if not indices:
            result = (
                next_state,
                *_selection_payload(next_state, preview_cache, load_preview=False),
                "No items match the current filter.",
            )
        else:
            selected = next_state.get("selected_index")
            try:
                position = indices.index(int(selected)) if selected is not None else 0
            except ValueError:
                position = 0
            position = min(len(indices) - 1, max(0, position + int(direction)))
            next_state["selected_index"] = indices[position]
            result = (
                next_state,
                *_selection_payload(next_state, preview_cache),
                f"Item {position + 1} of {len(indices)} in the filtered queue.",
            )
        return with_regeneration_prompt_update(
            result,
            variant_key,
            current_prompt,
            previous_error,
        )

    for trigger, direction in (
        (previous_item.click, -1), (hk_ed_prev.click, -1), (hk_prev_alias.click, -1),
        (next_item.click, 1), (hk_ed_next.click, 1), (hk_next_alias.click, 1),
    ):
        trigger(
            lambda current, variant, prompt, error, step=direction: navigate_handler(
                current,
                step,
                variant,
                prompt,
                error,
            ),
            inputs=[state, regen_variant, regen_prompt_value, regen_prompt_error],
            outputs=[
                state,
                video,
                audio,
                image,
                preview_placeholder,
                caption,
                stats,
                status,
                *regeneration_prompt_outputs,
            ],
            show_progress="minimal", api_visibility="private",
        )

    def flag_handler(
        current: EditorState,
        flag: str,
        variant_key: str,
        current_prompt: str | None,
        previous_error: str,
    ) -> tuple[Any, ...]:
        result = editor_flag_handler(
            current,
            flag,
            default_folder=ctx.outputs_dir,
            preview_cache=preview_cache,
        )
        message = str(result[-1])
        if "vc-err" in message or message == "No caption is selected.":
            ctx.app_log.error(re.sub(r"<[^>]+>", "", message), scope="editor")
        else:
            ctx.app_log.log(message, scope="editor")
        return with_regeneration_prompt_update(
            result,
            variant_key,
            current_prompt,
            previous_error,
        )

    for trigger, flag in ((approve.click, "approved"), (hk_ed_approve.click, "approved"), (reject.click, "rejected"), (hk_ed_reject.click, "rejected")):
        trigger(
            lambda current, variant, prompt, error, value=flag: flag_handler(
                current,
                value,
                variant,
                prompt,
                error,
            ),
            inputs=[state, regen_variant, regen_prompt_value, regen_prompt_error],
            outputs=[
                state,
                table,
                counters,
                page_label,
                video,
                audio,
                image,
                preview_placeholder,
                caption,
                stats,
                status,
                *regeneration_prompt_outputs,
            ],
            show_progress="minimal", api_visibility="private",
        )

    def preview_replace_handler(current: EditorState, find: str, replacement: str, regex: bool, case: bool, whole: bool, scope: str) -> str:
        try:
            selected_items = [current["items"][index] for index in _scope_indices(current, scope)]
            result = find_replace_preview(selected_items, find, replacement, regex=regex, case_sensitive=case, whole_words=whole)
            return _preview_html(result)
        except Exception as exc:
            return f"<p class='vc-err'>{html.escape(str(exc))}</p>"

    preview_replace.click(
        preview_replace_handler,
        inputs=[state, find_text, replace_text, replace_regex, replace_case, replace_whole, replace_scope],
        outputs=replace_result, queue=False, show_progress="hidden", api_visibility="private",
    )

    def apply_replace_handler(current: EditorState, find: str, replacement: str, regex: bool, case: bool, whole: bool, scope: str) -> tuple[Any, ...]:
        next_state = deepcopy(current or initial_state)
        try:
            target_indices = _scope_indices(next_state, scope)
            selected_items = [next_state["items"][index] for index in target_indices]
            preview = find_replace_preview(selected_items, find, replacement, regex=regex, case_sensitive=case, whole_words=whole)
            pairs = parse_replace_pairs([(find, replacement)]) or [(find, replacement)]
            for index in target_indices:
                item = next_state["items"][index]
                old = str(item.get("caption") or "")
                new = apply_replacements(old, pairs, regex=regex, case_insensitive=not case, whole_words=whole)
                if new != old:
                    _write_caption(Path(str(item["caption_path"])), new, item.get("caption_field"))
                    _refresh_item(item, new)
            next_state["dirty"] = False
            rows, _ = _page_rows(next_state)
            selected = next_state.get("selected_index")
            selected_item = next_state["items"][int(selected)] if selected is not None else None
            message = f"Applied {preview['replacement_count']} replacement(s) across {preview['files_changed']} file(s)."
            ctx.app_log.log(message, scope="editor")
            return next_state, rows, _counter_markdown(next_state), (selected_item.get("caption", "") if selected_item else ""), _stats_markdown(selected_item, _state_token_limit(next_state)), _preview_html(preview), message
        except Exception as exc:
            ctx.app_log.error(f"Find/replace failed: {exc}", scope="editor")
            error = f"<span class='vc-err'>{html.escape(str(exc))}</span>"
            return next_state, gr.skip(), gr.skip(), gr.skip(), gr.skip(), error, error

    apply_replace.click(
        apply_replace_handler,
        inputs=[state, find_text, replace_text, replace_regex, replace_case, replace_whole, replace_scope],
        outputs=[state, table, counters, caption, stats, replace_result, status],
        show_progress="minimal", api_visibility="private",
    )

    def bulk_handler(current: EditorState, prefix: str, suffix: str, trigger_word: str, position: str, scope: str, mode: str) -> tuple[Any, ...]:
        next_state = deepcopy(current or initial_state)
        changed = 0
        try:
            for index in _scope_indices(next_state, scope):
                item = next_state["items"][index]
                old = str(item.get("caption") or "")
                if mode == "strip":
                    new = old.strip()
                elif mode == "newlines":
                    new = re.sub(r"[ \t]*\n+[ \t]*", " ", old).strip()
                else:
                    new = finalize_caption(old, prefix=prefix, suffix=suffix, trigger=trigger_word, trigger_mode=position)
                if new != old:
                    _write_caption(Path(str(item["caption_path"])), new, item.get("caption_field"))
                    _refresh_item(item, new)
                    changed += 1
            next_state["dirty"] = False
            rows, _ = _page_rows(next_state)
            selected = next_state.get("selected_index")
            selected_item = next_state["items"][int(selected)] if selected is not None else None
            message = f"Updated {changed} caption(s)."
            ctx.app_log.log(message, scope="editor")
            return next_state, rows, _counter_markdown(next_state), (selected_item.get("caption", "") if selected_item else ""), _stats_markdown(selected_item, _state_token_limit(next_state)), message, message
        except Exception as exc:
            ctx.app_log.error(f"Bulk edit failed: {exc}", scope="editor")
            error = f"<span class='vc-err'>{html.escape(str(exc))}</span>"
            return next_state, gr.skip(), gr.skip(), gr.skip(), gr.skip(), error, error

    bulk_inputs = [state, bulk_prefix, bulk_suffix, bulk_trigger, trigger_position, bulk_scope]
    for trigger, mode in ((bulk_apply.click, "bulk"), (strip_whitespace.click, "strip"), (collapse_newlines.click, "newlines")):
        trigger(
            lambda *args, selected_mode=mode: bulk_handler(*args, selected_mode),
            inputs=bulk_inputs, outputs=[state, table, counters, caption, stats, bulk_result, status],
            show_progress="minimal", api_visibility="private",
        )

    def export_handler(
        current: EditorState,
        destination: str,
        copy_media: bool,
        extension: str,
        include_caption_only: bool,
        create_zip: bool,
    ) -> str:
        message = editor_export_handler(
            current,
            destination,
            copy_media,
            extension,
            include_caption_only,
            create_zip,
        )
        if "vc-err" in message:
            ctx.app_log.error(f"Approved export failed: {message}", scope="editor")
        else:
            ctx.app_log.log(message, scope="editor")
        return message

    export_button.click(
        export_handler,
        inputs=[state, export_destination, export_copy_media, export_extension, export_caption_only, export_zip],
        outputs=export_result, show_progress="minimal", api_visibility="private",
    )

    def statistics_handler(current: EditorState, trigger_word: str) -> str:
        try:
            calculated = calculate_caption_statistics(
                (current or {}).get("items") or [],
                str(trigger_word or ""),
            )
            return render_caption_statistics(calculated)
        except Exception as exc:
            return f"<span class='vc-err'>{html.escape(str(exc))}</span>"

    compute_statistics.click(
        statistics_handler,
        inputs=[state, ctx.caption_handles.controls["trigger_word"]],
        outputs=statistics_output,
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )

    class _RegenerationSink:
        def __init__(self, events: "queue.Queue[tuple[str, Any]]") -> None:
            self.events = events

        def on_log(self, message: str, level: str = "info", scope: str | None = None) -> None:
            # PipelineClient already mirrors worker logs into AppLog. Raw loader
            # and native-library stderr must never replace the editor status.
            del message, level, scope

        @staticmethod
        def _status(event: Any) -> str | None:
            message = str(getattr(event, "message", event) or "").strip()
            data = dict(getattr(event, "data", {}) or {})
            phase = str(data.get("phase") or "").casefold()
            lowered = message.casefold()
            if any(token in lowered for token in ("info:", "debug:", ".dll", "llama-server:")):
                return None
            if "llama" in phase or "llama-server" in lowered:
                return "Starting llama-server (GGUF)..."
            if "load" in phase or "loading" in lowered or "checkpoint" in lowered:
                return "Loading model..."
            if any(token in lowered for token in ("caption", "generat", "segment", "clip")):
                return "Captioning clip..."
            return message or None

        def on_progress(self, event: Any) -> None:
            status = self._status(event)
            if status:
                self.events.put(("progress", status))

        def on_item(self, event: Any) -> None:
            status = self._status(event)
            if status:
                self.events.put(("progress", status))

    ctx.states["editor_regeneration_sink_class"] = _RegenerationSink

    def request_regenerate_all(
        current: EditorState,
        variant: str,
        prompt_id: str | None,
    ) -> tuple[list[int], Any, str, str]:
        indices = [
            index
            for index in filtered_indices(current or initial_state)
            if (current or initial_state)["items"][index].get("media_path")
            and Path(str((current or initial_state)["items"][index]["media_path"])).is_file()
        ]
        count = len(indices)
        if not count:
            return [], gr.update(visible=False), "Regenerate 0 captions?", "<span class='vc-warn'>No filtered captions have available media.</span>"
        items = (current or initial_state).get("items") or []
        for index in indices:
            _, _, message = resolve_regeneration_prompt_choices(
                variant,
                items[index],
                prompt_id,
            )
            if message:
                return [], gr.update(visible=False), "Regenerate 0 captions?", message
        return indices, gr.update(visible=True), f"Regenerate {count} captions?", f"Ready to regenerate {count} caption(s)."

    regenerate_all.click(
        request_regenerate_all,
        inputs=[state, regen_variant, regen_prompt_value],
        outputs=[regenerate_all_targets, regenerate_all_confirmation, regenerate_all_question, regen_status],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    regenerate_all_keep.click(
        lambda: ([], gr.update(visible=False), "Regeneration cancelled; captions were not changed."),
        outputs=[regenerate_all_targets, regenerate_all_confirmation, regen_status],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )

    def regenerate_all_handler(
        current: EditorState,
        target_indices: list[int],
        variant: str,
        prompt_id: str | None,
        override: str,
        *runtime_values: Any,
    ):
        next_state = deepcopy(current or initial_state)
        indices = [
            int(index)
            for index in (target_indices or [])
            if 0 <= int(index) < len(next_state.get("items") or [])
        ]
        if not indices:
            yield gr.update(visible=False), "<span class='vc-warn'>No captions are queued for regeneration.</span>", next_state, gr.skip(), gr.skip(), gr.skip(), gr.skip(), []
            return
        target_items = [next_state["items"][index] for index in indices]
        for target_item in target_items:
            _, _, message = resolve_regeneration_prompt_choices(
                variant,
                target_item,
                prompt_id,
            )
            if message:
                yield gr.update(visible=False), message, next_state, gr.skip(), gr.skip(), gr.skip(), gr.skip(), []
                return
        try:
            settings = registry.values_to_dict(runtime_values)
            caption_paths = [Path(str(item["caption_path"])) for item in target_items]
            variant_to_family(variant)
            specs: list[tuple[JobSpec, str]] = []
            for target_item in target_items:
                _, item_prompt_id, message = resolve_regeneration_prompt_choices(
                    variant, target_item, prompt_id
                )
                if message or not item_prompt_id:
                    raise ValueError(message or "No compatible regeneration prompt is available")
                get_preset(item_prompt_id)
                specs.append(
                    (
                        build_editor_regeneration_spec(
                            settings,
                            target_item,
                            variant=variant,
                            prompt_id=item_prompt_id,
                            override=override,
                            outputs_root=ctx.outputs_dir,
                        ),
                        editor_regeneration_log(target_item),
                    )
                )
        except Exception as exc:
            yield gr.update(visible=False), f"<span class='vc-err'>{html.escape(str(exc))}</span>", next_state, gr.skip(), gr.skip(), gr.skip(), gr.skip(), []
            return

        events: "queue.Queue[tuple[str, Any]]" = queue.Queue()
        terminal: dict[str, Any] = {}

        def work() -> None:
            try:
                set_mode = getattr(ctx.pipeline_client, "set_subprocess_mode", None)
                if callable(set_mode):
                    set_mode(bool(settings.get("subprocess_mode", True)))
                else:
                    ctx.pipeline_client.subprocess_mode = bool(settings.get("subprocess_mode", True))
                results = []
                sink = _RegenerationSink(events)
                for spec, log_message in specs:
                    ctx.app_log.log(log_message, scope="editor")
                    events.put(("log", log_message))
                    results.append(ctx.pipeline.run_job(spec, sink))
                terminal["results"] = results
            except BaseException as exc:
                terminal["error"] = exc
            finally:
                events.put(("terminal", None))

        threading.Thread(target=work, daemon=True, name="vcap-editor-regenerate-all").start()
        yield gr.update(visible=False), f"Starting regeneration of {len(indices)} captions...", next_state, gr.skip(), gr.skip(), gr.skip(), gr.skip(), []
        while True:
            kind, payload = events.get()
            if kind == "terminal":
                break
            yield gr.skip(), html.escape(str(payload)), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()
        if "error" in terminal:
            exc = terminal["error"]
            ctx.app_log.error(f"Editor filtered regeneration failed: {exc}", scope="editor")
            yield gr.skip(), f"<span class='vc-err'>{html.escape(str(exc))}</span>", next_state, gr.skip(), gr.skip(), gr.skip(), gr.skip(), []
            return

        changed = 0
        for index, caption_path in zip(indices, caption_paths):
            try:
                new_caption, field = _read_caption(caption_path)
                item = next_state["items"][index]
                item["caption_field"] = field or item.get("caption_field")
                _refresh_item(item, new_caption)
                changed += 1
            except Exception as exc:
                ctx.app_log.warn(f"Could not refresh regenerated caption {caption_path}: {exc}", scope="editor")
        next_state["dirty"] = False
        rows, _ = _page_rows(next_state)
        selected = next_state.get("selected_index")
        selected_item = next_state["items"][int(selected)] if selected is not None else None
        caption_value = str(selected_item.get("caption") or "") if selected_item else ""
        counts: dict[str, int] = {}
        for result in terminal.get("results", []):
            for key, value in dict(getattr(result, "counts", {}) or {}).items():
                counts[key] = counts.get(key, 0) + int(value or 0)
        message = (
            f"Regenerated {changed}/{len(indices)} caption(s); "
            f"failed {int(counts.get('failed', 0) or 0)}."
        )
        ctx.app_log.log(message, scope="editor")
        yield (
            gr.update(visible=False),
            message,
            next_state,
            rows,
            _counter_markdown(next_state),
            caption_value,
            _stats_markdown(selected_item, _state_token_limit(next_state)),
            [],
        )

    def regenerate_handler(
        current: EditorState,
        variant: str,
        prompt_id: str | None,
        override: str,
        *runtime_values: Any,
    ):
        next_state = deepcopy(current or initial_state)
        selected = next_state.get("selected_index")
        if selected is None or not (0 <= int(selected) < len(next_state.get("items") or [])):
            yield gr.skip(), "<span class='vc-err'>No caption is selected.</span>", *(gr.skip() for _ in range(6))
            return
        item = next_state["items"][int(selected)]
        _, prompt_id, message = resolve_regeneration_prompt_choices(
            variant,
            item,
            prompt_id,
        )
        if message:
            yield gr.skip(), message, *(gr.skip() for _ in range(6))
            return
        media_path = item.get("segment_media_path") or item.get("source_media_path") or item.get("media_path")
        if not media_path or not Path(str(media_path)).is_file():
            yield gr.skip(), "<span class='vc-err'>The selected caption has no media to regenerate.</span>", *(gr.skip() for _ in range(6))
            return
        caption_path = Path(str(item["caption_path"]))
        old = str(item.get("caption") or "")
        if next_state.get("dirty"):
            _write_caption(caption_path, old, item.get("caption_field"))
            next_state["dirty"], next_state["draft_caption"] = False, None
        try:
            old_raw = caption_path.read_text(encoding="utf-8")
        except OSError:
            old_raw = None
        old_field = item.get("caption_field")
        old_video_raw: str | None = None
        video_part_path = (
            Path(str(item["video_caption_path"]))
            if item.get("video_caption_path")
            else None
        )
        if video_part_path is not None:
            try:
                old_video_raw = video_part_path.read_text(encoding="utf-8")
            except OSError:
                old_video_raw = None
        try:
            settings = registry.values_to_dict(runtime_values)
        except ValueError as exc:
            yield gr.skip(), f"<span class='vc-err'>{html.escape(str(exc))}</span>", *(gr.skip() for _ in range(6))
            return
        try:
            get_preset(prompt_id)
            variant_to_family(variant)
            spec = build_editor_regeneration_spec(
                settings,
                item,
                variant=variant,
                prompt_id=prompt_id,
                override=override,
                outputs_root=ctx.outputs_dir,
            )
        except Exception as exc:
            yield gr.skip(), f"<span class='vc-err'>{html.escape(str(exc))}</span>", *(gr.skip() for _ in range(6))
            return

        events: "queue.Queue[tuple[str, Any]]" = queue.Queue()
        terminal: dict[str, Any] = {}

        def work() -> None:
            try:
                set_mode = getattr(ctx.pipeline_client, "set_subprocess_mode", None)
                if callable(set_mode):
                    set_mode(bool(settings.get("subprocess_mode", True)))
                else:
                    ctx.pipeline_client.subprocess_mode = bool(settings.get("subprocess_mode", True))
                log_message = editor_regeneration_log(item)
                ctx.app_log.log(log_message, scope="editor")
                events.put(("log", log_message))
                terminal["result"] = ctx.pipeline.run_job(spec, _RegenerationSink(events))
            except BaseException as exc:
                terminal["error"] = exc
            finally:
                events.put(("terminal", None))

        threading.Thread(target=work, daemon=True, name="vcap-editor-regenerate").start()
        yield gr.skip(), "Starting regeneration...", *(gr.skip() for _ in range(6))
        while True:
            kind, payload = events.get()
            if kind == "terminal":
                break
            yield gr.skip(), html.escape(str(payload)), *(gr.skip() for _ in range(6))
        if "error" in terminal:
            exc = terminal["error"]
            ctx.app_log.error(f"Editor regeneration failed: {exc}", scope="editor")
            yield gr.skip(), f"<span class='vc-err'>{html.escape(str(exc))}</span>", *(gr.skip() for _ in range(6))
            return
        try:
            generated, field = _read_caption(caption_path)
            new = rebuild_caption_parts_after_regeneration(settings, item, generated)
            if video_part_path is not None:
                item["video_caption"] = generated
                ctx.app_log.log(
                    f"Updated {video_part_path} and rebuilt {caption_path.name} with its audio caption.",
                    scope="editor",
                )
            item["caption_field"] = field or item.get("caption_field")
            _refresh_item(item, new)
            next_state["dirty"] = False
            rows, _ = _page_rows(next_state)
            backup = {
                "caption_path": str(caption_path),
                "old": old,
                "raw": old_raw,
                "field": old_field,
                "video_caption_path": str(video_part_path) if video_part_path is not None else None,
                "video_raw": old_video_raw,
            }
            message = f"Regenerated {caption_path.name}. Review the diff, then keep or revert."
            ctx.app_log.log(message, scope="editor")
            yield diff_html(old, new), message, next_state, new, _stats_markdown(item, _state_token_limit(next_state)), rows, _counter_markdown(next_state), backup
        except Exception as exc:
            ctx.app_log.error(f"Could not load regenerated caption: {exc}", scope="editor")
            yield gr.skip(), f"<span class='vc-err'>{html.escape(str(exc))}</span>", *(gr.skip() for _ in range(6))

    def keep_handler(backup: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        if not backup:
            return "No regenerated caption is awaiting a decision.", {}
        ctx.app_log.log(f"Kept regenerated caption {backup.get('caption_path')}", scope="editor")
        return "Regenerated caption kept.", {}

    keep_new.click(
        keep_handler, inputs=regeneration_backup, outputs=[regen_status, regeneration_backup],
        queue=False, show_progress="hidden", api_visibility="private",
    )

    def revert_handler(current: EditorState, backup: dict[str, Any]) -> tuple[Any, ...]:
        if not backup:
            return gr.skip(), "No regenerated caption is awaiting a decision.", *(gr.skip() for _ in range(5)), {}
        next_state = deepcopy(current or initial_state)
        try:
            path, old = Path(str(backup["caption_path"])), str(backup.get("old") or "")
            if backup.get("raw") is not None:
                OutputWriter().write_text(path, str(backup["raw"]))
            else:
                _write_caption(path, old, backup.get("field"))
            video_path_value = backup.get("video_caption_path")
            if video_path_value and backup.get("video_raw") is not None:
                OutputWriter().write_text(
                    Path(str(video_path_value)),
                    str(backup["video_raw"]),
                )
            for candidate in next_state.get("items") or []:
                if str(candidate.get("caption_path")) == str(path):
                    _refresh_item(candidate, old)
                    candidate["caption_field"] = backup.get("field")
                    if video_path_value and Path(str(video_path_value)).is_file():
                        candidate["video_caption"] = _read_caption(
                            Path(str(video_path_value))
                        )[0]
                    break
            selected = next_state.get("selected_index")
            selected_item = next_state["items"][int(selected)] if selected is not None else None
            rows, _ = _page_rows(next_state)
            ctx.app_log.log(f"Reverted regenerated caption {path}", scope="editor")
            return "", "Reverted to the previous caption.", next_state, old, _stats_markdown(selected_item, _state_token_limit(next_state)), rows, _counter_markdown(next_state), {}
        except Exception as exc:
            return gr.skip(), f"<span class='vc-err'>{html.escape(str(exc))}</span>", *(gr.skip() for _ in range(5)), backup

    revert.click(
        revert_handler, inputs=[state, regeneration_backup],
        outputs=[regen_diff, regen_status, state, caption, stats, table, counters, regeneration_backup],
        show_progress="minimal", api_visibility="private",
    )
    ctx.states["editor_state"] = state
    ctx.states["editor_regeneration_binding"] = {
        "prompt_update_handler": update_regeneration_prompts,
        "scan_handler": scan_handler,
        "select_handler": select_handler,
        "gallery_select_handler": gallery_select_handler,
        "navigate_handler": navigate_handler,
        "flag_handler": flag_handler,
        "prompt_inputs": regeneration_prompt_inputs,
        "prompt_outputs": regeneration_prompt_outputs,
        "regenerate": regenerate,
        "regenerate_handler": regenerate_handler,
        "regenerate_inputs": [state, regen_variant, regen_prompt_value, regen_override],
        "regenerate_outputs": [
            regen_diff,
            regen_status,
            state,
            caption,
            stats,
            table,
            counters,
            regeneration_backup,
        ],
        "regenerate_all_yes": regenerate_all_yes,
        "regenerate_all_confirmation": regenerate_all_confirmation,
        "regenerate_all_handler": regenerate_all_handler,
        "regenerate_all_inputs": [
            state,
            regenerate_all_targets,
            regen_variant,
            regen_prompt_value,
            regen_override,
        ],
        "regenerate_all_outputs": [
            regenerate_all_confirmation,
            regen_status,
            state,
            table,
            counters,
            caption,
            stats,
            regenerate_all_targets,
        ],
        "wired": False,
    }


def wire(ctx: "UiContext") -> None:
    """Wire registry-wide regeneration events after every tab has registered."""

    binding = ctx.states.get("editor_regeneration_binding")
    if not isinstance(binding, dict):
        raise RuntimeError("editor_tab.build() must run before wire()")
    if binding.get("wired"):
        return

    registry_components = ctx.settings_registry.components()
    confirm_event = binding["regenerate_all_yes"].click(
        lambda: gr.update(visible=False),
        outputs=binding["regenerate_all_confirmation"],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    regenerate_all_event = confirm_event.then(
        binding["regenerate_all_handler"],
        inputs=[*binding["regenerate_all_inputs"], *registry_components],
        outputs=binding["regenerate_all_outputs"],
        concurrency_id="gpu_queue",
        concurrency_limit=1,
        show_progress="hidden",
        api_visibility="private",
    )
    regenerate_event = binding["regenerate"].click(
        binding["regenerate_handler"],
        inputs=[*binding["regenerate_inputs"], *registry_components],
        outputs=binding["regenerate_outputs"],
        concurrency_id="gpu_queue",
        concurrency_limit=1,
        show_progress="hidden",
        api_visibility="private",
    )
    binding.update(
        {
            "wired": True,
            "registry_components": registry_components,
            "regenerate_event": regenerate_event,
            "regenerate_all_event": regenerate_all_event,
        }
    )


__all__ = [
    "CAPTION_EXTENSIONS",
    "EditorItem",
    "EditorState",
    "build",
    "caption_parts_payload",
    "editor_export_handler",
    "editor_filter_handler",
    "editor_flag_handler",
    "editor_page_gallery",
    "editor_save_handler",
    "editor_thumbnail",
    "filter_items",
    "filtered_indices",
    "find_replace_preview",
    "new_editor_state",
    "paginate_items",
    "pagination_math",
    "preview_find_replace",
    "rebuild_caption_parts_after_regeneration",
    "regeneration_prompt_choices",
    "resolve_regeneration_prompt_choices",
    "resolve_media_from_metadata",
    "scan_folder",
    "wire",
]
