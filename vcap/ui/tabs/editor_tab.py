"""Caption review, correction, regeneration, and approved-dataset export."""

from __future__ import annotations

import html
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

from vcap.core.captions_post import (
    apply_replacements,
    caption_stats,
    diff_html,
    finalize_caption,
    parse_replace_pairs,
)
from vcap.core.export import export_dataset, read_flags, write_flags
from vcap.core.media import preview_safe_media, probe_media
from vcap.core.outputs import OutputWriter
from vcap.core.paths import (
    guess_kind_by_extension,
    list_media_files,
    natural_sort_key,
    normalize_path,
    open_in_file_manager,
    reveal_in_file_manager,
)
from vcap.models.registry import all_variant_choices, variant_to_family
from vcap.pipeline.job import InputItem, JobSpec, OutputSpec, PostSpec
from vcap.prompts.presets import get_preset, list_presets
from vcap.ui.components import action_button

if TYPE_CHECKING:
    from vcap.ui.app import UiContext


CAPTION_EXTENSIONS = (".txt", ".json", ".srt")
_RUN_DIR_RE = re.compile(r"^(?:batch_)?\d{4,}_.+", re.IGNORECASE)
_IGNORED_JSON_NAMES = {
    ".vcap_flags.json",
    "metadata.json",
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
}


class EditorItem(TypedDict, total=False):
    media_path: str | None
    source_media_path: str | None
    caption_path: str
    caption: str
    caption_field: str | None
    caption_formats: list[str]
    kind: str
    duration: float | None
    chars: int
    tokens: int
    flag: str | None
    status: str


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
        for key in ("caption", "text", "final_caption", "description"):
            if isinstance(value.get(key), str):
                return str(value[key]), key
    return json.dumps(value, ensure_ascii=False, indent=2), None


def _write_caption(path: Path, text: str, field: str | None = None) -> Path:
    """Atomically write text while preserving common JSON caption wrappers."""

    value = str(text)
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
    for key in ("input", "source", "media_path", "path", "file"):
        if key in entry:
            values.extend(_path_values(entry[key]))
    return values


def _is_caption_match(raw: str, caption_path: Path, metadata_path: Path) -> bool:
    try:
        candidate = _metadata_candidate(raw, metadata_path)
    except Exception:
        return False
    return candidate == caption_path or (
        candidate.stem.casefold() == caption_path.stem.casefold()
        and candidate.suffix.casefold() in CAPTION_EXTENSIONS
    )


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


def resolve_media_from_metadata(
    caption_path: str | os.PathLike[str],
    scan_root: str | os.PathLike[str] | None = None,
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
    return None, None


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
        items.append(_item_from_pair(media, selected, formats, root, flags, failed_paths, failed_names))

    for caption_path in caption_files:
        if str(caption_path.resolve(strict=False)).casefold() in used_captions:
            continue
        formats = captions_by_key[(str(caption_path.parent).casefold(), caption_path.stem.casefold())]
        if caption_path != formats[0]:
            continue
        used_captions.update(str(path.resolve(strict=False)).casefold() for path in formats)
        resolved_media, source_media = resolve_media_from_metadata(caption_path, root)
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


def _page_rows(state: EditorState) -> tuple[list[list[Any]], str]:
    indices = filtered_indices(state)
    page, pages, start, end = pagination_math(len(indices), int(state.get("page", 1)), int(state.get("page_size", 25)))
    state["page"] = page
    rows: list[list[Any]] = []
    for global_index in indices[start:end]:
        item = state["items"][global_index]
        name = Path(str(item.get("media_path") or item.get("caption_path") or "")).name
        preview = re.sub(r"\s+", " ", str(item.get("caption") or "")).strip()
        preview = preview if len(preview) <= 120 else preview[:117].rstrip() + "..."
        rows.append([global_index + 1, name, preview, int(item.get("chars") or 0), int(item.get("tokens") or 0), item.get("flag") or "-", item.get("status") or "empty"])
    showing = "0" if not indices else f"{start + 1}-{end}"
    return rows, f"**Page {page} / {pages}** · showing {showing} of {len(indices)}"


def _counter_markdown(state: Mapping[str, Any]) -> str:
    items = state.get("items") or []
    captioned = sum(bool(str(item.get("caption") or "").strip()) for item in items)
    approved = sum(item.get("flag") == "approved" for item in items)
    failed = sum(item.get("status") == "failed" for item in items)
    return f"**{len(items)} items** · {captioned} captioned · {approved} approved · {failed} failed"


def _stats_markdown(item: Mapping[str, Any] | None) -> str:
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
    return f"**{stats['chars']} chars** · {stats['words']} words · ~{stats['approx_tokens']} tokens{duration_text}<br>`{html.escape(raw)}`"


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
    placeholder = gr.update(visible=False)
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
            safe = str(preview_safe_media(media_path, cache_dir))
            if info.has_video:
                video = gr.update(value=safe, visible=True)
            elif info.kind == "audio":
                audio = gr.update(value=safe, visible=True)
            elif info.kind == "image":
                image = gr.update(value=safe, visible=True)
        except Exception:
            placeholder = gr.update(value="Media preview unavailable.", visible=True)
    state["dirty"], state["draft_caption"] = False, None
    return video, audio, image, placeholder, str(item.get("caption") or ""), _stats_markdown(item)


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


def _prompt_choices() -> list[tuple[str, str]]:
    return [(f"{preset.group} · {preset.label}", preset.id) for preset in list_presets()]


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
    if len(raw) != 8:
        return _handler_error(10, current, "<span class='vc-err'>Invalid filter input count.</span>")
    next_state = deepcopy(current or initial_state or new_editor_state())
    next_state["filter"] = {
        "search": raw[0], "regex": bool(raw[1]), "min_length": raw[2],
        "max_length": raw[3], "min_tokens": raw[4], "max_tokens": raw[5],
        "flag": raw[6], "status": raw[7],
    }
    try:
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
        return next_state, rows, _counter_markdown(next_state), _stats_markdown(item), (gr.skip() if quiet else message)
    except Exception as exc:
        return (
            next_state,
            gr.skip(),
            gr.skip(),
            _stats_markdown(item),
            f"<span class='vc-err'>{html.escape(str(exc))}</span>",
        )


def editor_export_handler(
    current: EditorState,
    destination: str | os.PathLike[str],
    copy_media: bool = True,
    extension: str = ".txt",
    include_caption_only: bool = False,
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
        if report.errors:
            message += " " + " | ".join(report.errors[:3])
        return message
    except Exception as exc:
        return f"<span class='vc-err'>{html.escape(str(exc))}</span>"


def build(ctx: "UiContext") -> None:
    """Render and wire the full Caption Editor tab."""

    initial_state = new_editor_state(ctx.outputs_dir)
    state = gr.State(initial_state)
    regeneration_backup = gr.State({})
    autosave_timer = gr.Timer(0.5)
    preview_cache = ctx.temp_dir / "editor_previews"
    registry = ctx.settings_registry
    registry_components = registry.components()
    model_entry = next((entry for entry in registry.entries() if entry.key == "model_key"), None)

    with gr.Row(elem_classes=["vc-compact-row"]):
        folder = gr.Textbox(
            value=str(ctx.outputs_dir), label="Caption folder",
            info="Scan media/caption sidecars or SECourses run directories.", scale=7,
        )
        scan = action_button("Scan", "cyan", size="md", scale=1, min_width=92)
        open_folder = action_button("📂 Open folder", "amber", size="md", scale=1, min_width=118)
        reveal_selected = action_button("📍 Reveal selected file", "crimson", size="md", scale=1, min_width=158)
        recursive = gr.Checkbox(
            value=False, label="Recursive",
            info="Include nested batch folders and run clip directories.", scale=1,
        )

    with gr.Accordion("Filters", open=False):
        with gr.Row(elem_classes=["vc-compact-row"]):
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
                ],
                value="all", label="Status", info="Show empty or failed captions only.",
            )
        with gr.Row(elem_classes=["vc-compact-row"]):
            min_length = gr.Number(label="Min chars", precision=0, minimum=0, info="Minimum caption length; 0 = no limit.")
            max_length = gr.Number(label="Max chars", precision=0, minimum=0, info="Maximum caption length; 0 = no limit.")
            min_tokens = gr.Number(label="Min tokens", precision=0, minimum=0, info="Minimum approximate token count; 0 = no limit.")
            max_tokens = gr.Number(label="Max tokens", precision=0, minimum=0, info="Maximum approximate token count; 0 = no limit.")
            apply_filters = action_button("Apply filters", "blue", size="md", min_width=126)

    status = gr.Markdown("Ready to scan.", elem_classes=["vc-status"])
    with gr.Row(equal_height=False):
        with gr.Column(scale=5, min_width=520):
            table = gr.Dataframe(
                value=[], headers=_TABLE_HEADERS,
                datatype=["number", "str", "str", "number", "number", "str", "str"],
                type="array", interactive=False, show_search="none", max_height=520,
                pinned_columns=2, static_columns=list(range(7)),
                column_widths=[55, 180, "46%", 70, 75, 90, 100], wrap=False,
                buttons=["copy", "fullscreen"], label="Review queue",
            )
            counters = gr.Markdown(_counter_markdown(initial_state))
            with gr.Row(elem_classes=["vc-compact-row"]):
                previous_page = action_button("Prev page", "indigo", size="md", scale=1)
                page_label = gr.Markdown("**Page 1 / 1** · showing 0 of 0", scale=3)
                next_page = action_button("Next page", "sky", size="md", scale=1)
                page_size = gr.Dropdown(
                    choices=[25, 50, 100], value=25, label="Per page",
                    info="Rows shown on each editor page.", scale=1, min_width=105,
                )

        with gr.Column(scale=5, min_width=480):
            preview_placeholder = gr.Markdown(
                "No preview selected.",
                elem_classes=["vc-status", "vc-preview-placeholder"],
            )
            video = gr.Video(label="Video preview", visible=False, interactive=False, elem_classes=["vc-preview"])
            audio = gr.Audio(label="Audio preview", visible=False, interactive=False, elem_classes=["vc-preview"])
            image = gr.Image(label="Image preview", visible=False, interactive=False, type="filepath", elem_classes=["vc-preview"])
            caption = gr.Textbox(
                label="Caption", lines=12, max_lines=18, buttons=["copy"],
                info="Edits are saved after a short pause when autosave is enabled.",
            )
            stats = gr.Markdown(_stats_markdown(None))
            with gr.Row(elem_classes=["vc-compact-row"]):
                autosave = gr.Checkbox(
                    value=True, label="Autosave on edit",
                    info="Atomically save after 0.7 seconds without another edit.", scale=2,
                )
                save = action_button("💾 Save", "green", size="md", scale=1)
            with gr.Row(elem_classes=["vc-compact-row"]):
                previous_item = action_button("⬅ Prev", "violet", size="md", scale=1)
                next_item = action_button("➡ Next", "slate", size="md", scale=1)
                approve = action_button("✅ Approve", "emerald", size="md", scale=1)
                reject = action_button("❌ Reject", "rose", size="md", scale=1)
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

    with gr.Accordion("🔁 Regenerate selected", open=False):
        variants = all_variant_choices()
        default_variant = (
            str(model_entry.default)
            if model_entry is not None
            else next((key for _, key in variants if key == "qwen3_omni_instruct_int8"), variants[0][1])
        )
        prompts = _prompt_choices()
        with gr.Row(elem_classes=["vc-compact-row"]):
            regen_variant = gr.Dropdown(
                choices=variants, value=default_variant, label="Model variant",
                info="The selected checkpoint replaces only this caption.", scale=3,
            )
            regen_prompt = gr.Dropdown(
                choices=prompts, value=prompts[0][1] if prompts else None, label="Prompt preset",
                info="Choose a task compatible with the media and model.", scale=3,
            )
        regen_override = gr.Textbox(
            label="User prompt override", lines=4,
            info="Leave blank to use the selected prompt preset unchanged.",
        )
        with gr.Row(elem_classes=["vc-compact-row"]):
            regenerate = action_button("🔁 Regenerate", "fuchsia", size="md")
            keep_new = action_button("Keep new", "lime", size="md")
            revert = action_button("Revert", "red", size="md")
        regen_status = gr.Markdown("No regeneration is pending.", elem_classes=["vc-status"])
        regen_diff = gr.HTML("")

    if model_entry is not None:
        model_entry.component.change(
            lambda value: gr.update(value=value),
            inputs=model_entry.component,
            outputs=regen_variant,
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        )

    with gr.Accordion("🔎 Find & replace across folder", open=False):
        with gr.Row(elem_classes=["vc-compact-row"]):
            find_text = gr.Textbox(label="Find", info="Text or regular expression to locate.")
            replace_text = gr.Textbox(label="Replace", info="Replacement text applied by the shared post-processor.")
            replace_scope = gr.Radio(
                choices=["Filtered items", "All items"], value="Filtered items", label="Scope",
                info="Choose the current filtered queue or the complete scan.",
            )
        with gr.Row(elem_classes=["vc-compact-row"]):
            replace_regex = gr.Checkbox(value=False, label="Regex", info="Treat Find as a regular expression.")
            replace_case = gr.Checkbox(value=False, label="Case sensitive", info="Match letter case exactly.")
            replace_whole = gr.Checkbox(value=False, label="Whole word", info="Exclude matches embedded inside longer words.")
            preview_replace = action_button("Preview", "purple", size="md")
            apply_replace = action_button("Apply", "orange", size="md")
        replace_result = gr.HTML("")

    with gr.Accordion("➕ Bulk edit", open=False):
        with gr.Row(elem_classes=["vc-compact-row"]):
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
        with gr.Row(elem_classes=["vc-compact-row"]):
            bulk_apply = action_button("Apply bulk edit", "teal", size="md")
            strip_whitespace = action_button("Strip edges", "yellow", size="md")
            collapse_newlines = action_button("Collapse newlines", "bronze", size="md")
        bulk_result = gr.Markdown("", elem_classes=["vc-status"])

    with gr.Accordion("📤 Export approved", open=False):
        with gr.Row(elem_classes=["vc-compact-row"]):
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
            export_button = action_button("Export approved only", "pink", size="md")
        export_result = gr.Markdown("", elem_classes=["vc-status"])

    # Event handlers are kept in this module so their pure transforms remain testable.
    def scan_handler(raw_folder: str, recurse: bool, size: int) -> tuple[Any, ...]:
        try:
            items = scan_folder(raw_folder, bool(recurse))
            next_state = new_editor_state(normalize_path(raw_folder))
            next_state["items"], next_state["page_size"] = items, int(size or 25)
            matches = filtered_indices(next_state)
            next_state["selected_index"] = matches[0] if matches else None
            rows, page_text = _page_rows(next_state)
            selection = _selection_payload(next_state, preview_cache, load_preview=False)
            message = f"Scanned {len(items)} review item(s) in {next_state['folder']}."
            ctx.app_log.log(message, scope="editor")
            return next_state, rows, _counter_markdown(next_state), page_text, *selection, message
        except Exception as exc:
            empty = new_editor_state(raw_folder)
            ctx.app_log.error(f"Editor scan failed: {exc}", scope="editor")
            return empty, [], _counter_markdown(empty), "**Page 1 / 1**", *_selection_payload(empty, preview_cache, load_preview=False), f"<span class='vc-err'>{html.escape(str(exc))}</span>"

    scan.click(
        scan_handler, inputs=[folder, recursive, page_size],
        outputs=[state, table, counters, page_label, video, audio, image, preview_placeholder, caption, stats, status],
        show_progress="minimal", api_visibility="private",
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

    def filter_handler(current: EditorState, *values: Any) -> tuple[Any, ...]:
        return editor_filter_handler(
            current,
            values,
            initial_state=initial_state,
            preview_cache=preview_cache,
        )

    apply_filters.click(
        filter_handler,
        inputs=[state, search, search_regex, min_length, max_length, min_tokens, max_tokens, flag_filter, status_filter],
        outputs=[state, table, page_label, video, audio, image, preview_placeholder, caption, stats, status],
        queue=False, show_progress="hidden", api_visibility="private",
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

    def select_handler(current: EditorState, evt: gr.SelectData) -> tuple[Any, ...]:
        next_state = deepcopy(current or initial_state)
        row = int(evt.index[0] if isinstance(evt.index, (tuple, list)) else evt.index)
        indices = filtered_indices(next_state)
        _, _, start, end = pagination_math(len(indices), int(next_state.get("page", 1)), int(next_state.get("page_size", 25)))
        page_indices = indices[start:end]
        if 0 <= row < len(page_indices):
            next_state["selected_index"] = page_indices[row]
        selected_number = int(next_state.get("selected_index") or 0) + 1
        return next_state, *_selection_payload(next_state, preview_cache), f"Selected item {selected_number}."

    table.select(
        select_handler, inputs=state,
        outputs=[state, video, audio, image, preview_placeholder, caption, stats, status],
        show_progress="minimal", api_visibility="private",
    )

    def mark_dirty(current: EditorState, text: str) -> tuple[EditorState, str]:
        next_state = deepcopy(current or initial_state)
        selected = next_state.get("selected_index")
        if selected is not None and 0 <= int(selected) < len(next_state.get("items") or []):
            item = next_state["items"][int(selected)]
            _refresh_item(item, str(text or ""))
            next_state.update(dirty=True, draft_caption=str(text or ""), last_edit=time.monotonic())
            return next_state, _stats_markdown(item)
        return next_state, _stats_markdown(None)

    caption.input(
        mark_dirty, inputs=[state, caption], outputs=[state, stats],
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
        if current_text != saved_draft and isinstance(result[0], dict):
            saved_state = result[0]
            selected = saved_state.get("selected_index")
            if selected is not None and 0 <= int(selected) < len(saved_state.get("items") or []):
                selected_item = saved_state["items"][int(selected)]
                _refresh_item(selected_item, current_text)
                saved_state.update(
                    dirty=True,
                    draft_caption=current_text,
                    last_edit=time.monotonic(),
                )
                result[1], _ = _page_rows(saved_state)
                result[2] = _counter_markdown(saved_state)
                result[3] = _stats_markdown(selected_item)
        return tuple(result)

    autosave_timer.tick(
        autosave_handler, inputs=[state, autosave, caption], outputs=[state, table, counters, stats, status],
        queue=False, show_progress="hidden", api_visibility="private",
    )

    def navigate_handler(current: EditorState, direction: int) -> tuple[Any, ...]:
        next_state = deepcopy(current or initial_state)
        if next_state.get("dirty") and next_state.get("selected_index") is not None:
            selected_item = next_state["items"][int(next_state["selected_index"])]
            _write_caption(Path(str(selected_item["caption_path"])), str(next_state.get("draft_caption") or ""), selected_item.get("caption_field"))
            next_state["dirty"] = False
        indices = filtered_indices(next_state)
        if not indices:
            return next_state, *_selection_payload(next_state, preview_cache, load_preview=False), "No items match the current filter."
        selected = next_state.get("selected_index")
        try:
            position = indices.index(int(selected)) if selected is not None else 0
        except ValueError:
            position = 0
        position = min(len(indices) - 1, max(0, position + int(direction)))
        next_state["selected_index"] = indices[position]
        return next_state, *_selection_payload(next_state, preview_cache), f"Item {position + 1} of {len(indices)} in the filtered queue."

    for trigger, direction in (
        (previous_item.click, -1), (hk_ed_prev.click, -1), (hk_prev_alias.click, -1),
        (next_item.click, 1), (hk_ed_next.click, 1), (hk_next_alias.click, 1),
    ):
        trigger(
            lambda current, step=direction: navigate_handler(current, step), inputs=state,
            outputs=[state, video, audio, image, preview_placeholder, caption, stats, status],
            show_progress="minimal", api_visibility="private",
        )

    def flag_handler(current: EditorState, flag: str) -> tuple[Any, ...]:
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
        return result

    for trigger, flag in ((approve.click, "approved"), (hk_ed_approve.click, "approved"), (reject.click, "rejected"), (hk_ed_reject.click, "rejected")):
        trigger(
            lambda current, value=flag: flag_handler(current, value), inputs=state,
            outputs=[state, table, counters, page_label, video, audio, image, preview_placeholder, caption, stats, status],
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
            return next_state, rows, _counter_markdown(next_state), (selected_item.get("caption", "") if selected_item else ""), _stats_markdown(selected_item), _preview_html(preview), message
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
            return next_state, rows, _counter_markdown(next_state), (selected_item.get("caption", "") if selected_item else ""), _stats_markdown(selected_item), message, message
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
    ) -> str:
        message = editor_export_handler(
            current,
            destination,
            copy_media,
            extension,
            include_caption_only,
        )
        if "vc-err" in message:
            ctx.app_log.error(f"Approved export failed: {message}", scope="editor")
        else:
            ctx.app_log.log(message, scope="editor")
        return message

    export_button.click(
        export_handler,
        inputs=[state, export_destination, export_copy_media, export_extension, export_caption_only],
        outputs=export_result, show_progress="minimal", api_visibility="private",
    )

    class _RegenerationSink:
        def __init__(self, events: "queue.Queue[tuple[str, Any]]") -> None:
            self.events = events

        def on_log(self, message: str, level: str = "info", scope: str | None = None) -> None:
            del level
            self.events.put(("log", f"[{scope}] {message}" if scope else message))

        def on_progress(self, event: Any) -> None:
            self.events.put(("progress", getattr(event, "message", str(event))))

        def on_item(self, event: Any) -> None:
            self.events.put(("progress", getattr(event, "message", str(event))))

    def regenerate_handler(
        current: EditorState,
        variant: str,
        prompt_id: str,
        override: str,
        *runtime_values: Any,
    ):
        next_state = deepcopy(current or initial_state)
        selected = next_state.get("selected_index")
        if selected is None or not (0 <= int(selected) < len(next_state.get("items") or [])):
            yield gr.skip(), "<span class='vc-err'>No caption is selected.</span>", *(gr.skip() for _ in range(6))
            return
        item = next_state["items"][int(selected)]
        media_path = item.get("media_path")
        if not media_path or not Path(media_path).is_file():
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
        try:
            settings = registry.values_to_dict(runtime_values)
        except ValueError as exc:
            yield gr.skip(), f"<span class='vc-err'>{html.escape(str(exc))}</span>", *(gr.skip() for _ in range(6))
            return
        settings.update(
            model_key=variant,
            variant_key=variant,
            prompt_preset_id=prompt_id,
            system_prompt=None,
            user_prompt=(str(override).strip() or None),
            overwrite_existing=True,
            output_formats=[caption_path.suffix.lstrip(".") or "txt"],
        )
        output = OutputSpec(
            kind="batch", outputs_root=str(ctx.outputs_dir), batch_output_dir=str(caption_path.parent),
            mirror_names=False, overwrite=True,
        )
        try:
            get_preset(prompt_id)
            variant_to_family(variant)
            spec = JobSpec.from_settings(settings, [InputItem(path=media_path)], output)
            spec = replace(
                spec,
                post=PostSpec(formats=(caption_path.suffix.lstrip(".") or "txt",)),
                internal={
                    "output_dirs": [str(caption_path.parent)],
                    "output_stems": [caption_path.stem],
                    "metadata_name": "editor_regeneration_metadata.json",
                },
            )
        except Exception as exc:
            yield gr.skip(), f"<span class='vc-err'>{html.escape(str(exc))}</span>", *(gr.skip() for _ in range(6))
            return

        events: "queue.Queue[tuple[str, Any]]" = queue.Queue()
        terminal: dict[str, Any] = {}

        def work() -> None:
            try:
                ctx.pipeline_client.subprocess_mode = bool(settings.get("subprocess_mode", True))
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
            new, field = _read_caption(caption_path)
            item["caption_field"] = field or item.get("caption_field")
            _refresh_item(item, new)
            next_state["dirty"] = False
            rows, _ = _page_rows(next_state)
            backup = {
                "caption_path": str(caption_path),
                "old": old,
                "raw": old_raw,
                "field": old_field,
            }
            message = f"Regenerated {caption_path.name}. Review the diff, then keep or revert."
            ctx.app_log.log(message, scope="editor")
            yield diff_html(old, new), message, next_state, new, _stats_markdown(item), rows, _counter_markdown(next_state), backup
        except Exception as exc:
            ctx.app_log.error(f"Could not load regenerated caption: {exc}", scope="editor")
            yield gr.skip(), f"<span class='vc-err'>{html.escape(str(exc))}</span>", *(gr.skip() for _ in range(6))

    regenerate.click(
        regenerate_handler,
        inputs=[state, regen_variant, regen_prompt, regen_override, *registry_components],
        outputs=[regen_diff, regen_status, state, caption, stats, table, counters, regeneration_backup],
        concurrency_id="gpu_queue", concurrency_limit=1, show_progress="hidden", api_visibility="private",
    )

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
            for candidate in next_state.get("items") or []:
                if str(candidate.get("caption_path")) == str(path):
                    _refresh_item(candidate, old)
                    candidate["caption_field"] = backup.get("field")
                    break
            selected = next_state.get("selected_index")
            selected_item = next_state["items"][int(selected)] if selected is not None else None
            rows, _ = _page_rows(next_state)
            ctx.app_log.log(f"Reverted regenerated caption {path}", scope="editor")
            return "", "Reverted to the previous caption.", next_state, old, _stats_markdown(selected_item), rows, _counter_markdown(next_state), {}
        except Exception as exc:
            return gr.skip(), f"<span class='vc-err'>{html.escape(str(exc))}</span>", *(gr.skip() for _ in range(5)), backup

    revert.click(
        revert_handler, inputs=[state, regeneration_backup],
        outputs=[regen_diff, regen_status, state, caption, stats, table, counters, regeneration_backup],
        show_progress="minimal", api_visibility="private",
    )
    ctx.states["editor_state"] = state


__all__ = [
    "CAPTION_EXTENSIONS",
    "EditorItem",
    "EditorState",
    "build",
    "editor_export_handler",
    "editor_filter_handler",
    "editor_flag_handler",
    "editor_save_handler",
    "filter_items",
    "filtered_indices",
    "find_replace_preview",
    "new_editor_state",
    "paginate_items",
    "pagination_math",
    "preview_find_replace",
    "resolve_media_from_metadata",
    "scan_folder",
]
