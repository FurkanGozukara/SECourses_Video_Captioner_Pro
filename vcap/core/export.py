"""Dataset folder discovery, Musubi TOML generation, flags, and export."""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import tomli_w

from .logs import get_log
from .outputs import OutputWriter
from .paths import list_media_files, normalize_path, sanitize_filename, sort_paths_natural

_REPEAT_PREFIX = re.compile(r"^(\d+)_(.+)$")
_FLAGS_FILE = ".vcap_flags.json"


@dataclass(frozen=True)
class DatasetFolder:
    """One direct-media dataset directory and its parsed repeat prefix."""

    path: Path
    name: str
    num_repeats: int
    kind: str
    media_count: int
    has_repeat_prefix: bool


@dataclass(frozen=True)
class ExportReport:
    """Summary of an approved-item dataset export."""

    out_root: Path
    exported: int
    skipped: int
    rejected: int
    no_media: int
    media_files: list[Path]
    caption_files: list[Path]
    errors: list[str]

    @property
    def exported_count(self) -> int:
        """Alias used by UI counters."""

        return self.exported

    @property
    def skipped_count(self) -> int:
        """Alias used by UI counters."""

        return self.skipped

    @property
    def not_approved(self) -> int:
        """Number excluded by the approved-only filter."""

        return self.rejected

    @property
    def error_count(self) -> int:
        """Number of per-item export errors."""

        return len(self.errors)


def _repeat_parts(name: str) -> tuple[int, str, bool]:
    match = _REPEAT_PREFIX.match(str(name))
    if not match:
        return 1, str(name), False
    return max(1, int(match.group(1))), match.group(2), True


def discover_dataset_folders(root: str | os.PathLike[str]) -> list[DatasetFolder]:
    """Find root/direct-child media folders and parse ``N_name`` repeat counts."""

    parent = normalize_path(root, must_exist=True)
    if not parent.is_dir():
        raise NotADirectoryError(parent)
    candidates: list[Path] = []
    direct = list_media_files(parent, recursive=False, kinds=("video", "image"))
    if direct:
        candidates.append(parent)
    try:
        children = [
            child
            for child in parent.iterdir()
            if child.is_dir() and not child.name.startswith(".")
        ]
    except OSError:
        children = []
    for child in sort_paths_natural(children):
        if list_media_files(child, recursive=False, kinds=("video", "image")):
            candidates.append(child)
    result: list[DatasetFolder] = []
    for folder in candidates:
        videos = list_media_files(folder, recursive=False, kinds=("video",))
        images = list_media_files(folder, recursive=False, kinds=("image",))
        if videos and images:
            kind = "mixed"
        elif videos:
            kind = "video"
        else:
            kind = "image"
        repeats, clean_name, explicit = _repeat_parts(folder.name)
        result.append(
            DatasetFolder(
                path=folder,
                name=clean_name,
                num_repeats=repeats,
                kind=kind,
                media_count=len(videos) + len(images),
                has_repeat_prefix=explicit,
            )
        )
    return result


def _toml_path(path: Path) -> str:
    return path.resolve(strict=False).as_posix()


def write_kohya_musubi_toml(
    dataset_root: str | os.PathLike[str],
    out_path: str | os.PathLike[str],
    *,
    kind: str = "video",
    resolution: tuple[int, int] = (960, 544),
    caption_extension: str = ".txt",
    batch_size: int = 1,
    enable_bucket: bool = True,
    bucket_no_upscale: bool = False,
    num_repeats: int = 1,
    target_frames: Sequence[int] | None = None,
    frame_extraction: str = "head",
    frame_stride: int = 1,
    frame_sample: int = 1,
    max_frames: int = 129,
    source_fps: float | None = None,
    cache_directory_name: str = "cache_dir",
) -> Path:
    """Write the proven ``[[datasets]]`` plus ``[general]`` Musubi TOML shape."""

    selected_kind = str(kind).casefold()
    if selected_kind not in {"video", "image"}:
        raise ValueError("kind must be 'video' or 'image'")
    width, height = int(resolution[0]), int(resolution[1])
    if width <= 0 or height <= 0:
        raise ValueError("resolution dimensions must be positive")
    caption_ext = str(caption_extension or ".txt")
    if not caption_ext.startswith("."):
        caption_ext = "." + caption_ext
    root = normalize_path(dataset_root, must_exist=True)
    discovered = discover_dataset_folders(root)
    folders = [
        folder
        for folder in discovered
        if folder.kind == selected_kind or (folder.kind == "mixed" and selected_kind == "video")
    ]
    if not folders:
        raise ValueError(f"No {selected_kind} dataset folders found in {root}")
    frames = [max(1, int(value)) for value in (target_frames or [81])]
    datasets: list[dict[str, Any]] = []
    default_repeats = max(1, int(num_repeats))
    cache_name = str(cache_directory_name or "").strip()
    for folder in folders:
        entry: dict[str, Any] = {
            f"{selected_kind}_directory": _toml_path(folder.path),
            "num_repeats": folder.num_repeats if folder.has_repeat_prefix else default_repeats,
        }
        if selected_kind == "video":
            entry.update(
                {
                    "target_frames": frames,
                    "frame_extraction": str(frame_extraction),
                    "frame_stride": max(1, int(frame_stride)),
                    "frame_sample": max(1, int(frame_sample)),
                    "max_frames": max(1, int(max_frames)),
                }
            )
            if source_fps is not None and float(source_fps) > 0:
                entry["source_fps"] = float(source_fps)
        if cache_name:
            raw_cache = Path(cache_name)
            cache_path = raw_cache / folder.path.name if raw_cache.is_absolute() else folder.path / raw_cache
            entry["cache_directory"] = _toml_path(cache_path)
        datasets.append(entry)
    document = {
        "datasets": datasets,
        "general": {
            "resolution": [width, height],
            "caption_extension": caption_ext,
            "batch_size": max(1, int(batch_size)),
            "enable_bucket": bool(enable_bucket),
            "bucket_no_upscale": bool(bucket_no_upscale),
        },
    }
    payload = tomli_w.dumps(document)
    payload = re.sub(r",(\s*])", r"\1", payload)
    target = normalize_path(out_path)
    return OutputWriter().write_text(target, payload)


def _flag_key(value: str | os.PathLike[str]) -> str | None:
    text = os.fspath(value).replace("\\", "/").strip()
    while text.startswith("./"):
        text = text[2:]
    path = PurePosixPath(text)
    if not text or path.is_absolute() or ".." in path.parts:
        return None
    return path.as_posix()


def read_flags(folder: str | os.PathLike[str]) -> dict[str, Any]:
    """Read relative-path approval/rejection flags; malformed files yield an empty map."""

    root = normalize_path(folder)
    path = root / _FLAGS_FILE
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        get_log().warn(f"Could not read flags from '{path}': {exc}", scope="export")
        return {}
    if not isinstance(value, dict):
        return {}
    result: dict[str, Any] = {}
    for raw_key, flag in value.items():
        key = _flag_key(str(raw_key))
        if key is not None:
            result[key] = flag
    return result


def write_flags(folder: str | os.PathLike[str], flags: Mapping[str, Any]) -> Path:
    """Atomically persist editor flags keyed by normalized relative paths."""

    root = normalize_path(folder)
    root.mkdir(parents=True, exist_ok=True)
    normalized: dict[str, Any] = {}
    for raw_key, flag in flags.items():
        key = _flag_key(str(raw_key))
        if key is not None:
            normalized[key] = flag
    return OutputWriter().write_json(root / _FLAGS_FILE, normalized, pretty=True)


def _item_value(item: Any, *names: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        for name in names:
            if name in item:
                return item[name]
        return default
    for name in names:
        if hasattr(item, name):
            return getattr(item, name)
    return default


def _approved(item: Any, media: Path) -> bool:
    def flag_value(value: Any) -> bool:
        if isinstance(value, Mapping):
            return flag_value(value.get("approved", value.get("flag", False)))
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value == 1
        return str(value).strip().casefold() in {
            "approved",
            "approve",
            "yes",
            "true",
            "1",
        }

    explicit = _item_value(item, "approved", default=None)
    if explicit is not None:
        return flag_value(explicit)
    flag = _item_value(item, "flag", "status", default=None)
    if flag is not None:
        return flag_value(flag)

    configured_root = _item_value(item, "flags_root", "dataset_root", "source_root", default=None)
    candidates: list[Path] = []
    if configured_root:
        candidates.append(normalize_path(configured_root))
    current = media.parent
    while True:
        if (current / _FLAGS_FILE).is_file() and current not in candidates:
            candidates.append(current)
        if current.parent == current:
            break
        current = current.parent
    for root in candidates:
        try:
            key = media.relative_to(root).as_posix()
        except ValueError:
            continue
        stored_flags = read_flags(root)
        if key in stored_flags:
            return flag_value(stored_flags[key])
    return False


def _caption_for(item: Any, media: Path) -> tuple[str | None, Path | None]:
    text = _item_value(item, "caption", "caption_text", "text", default=None)
    if text is not None:
        return str(text), None
    caption_path = _item_value(item, "caption_path", default=None)
    source = normalize_path(caption_path) if caption_path else media.with_suffix(".txt")
    try:
        return source.read_text(encoding="utf-8"), source
    except OSError:
        return None, source


def _relative_parent(item: Any) -> Path:
    raw = _item_value(item, "relative_path", "relative", default=None)
    if not raw:
        return Path()
    key = _flag_key(str(raw))
    if key is None:
        return Path()
    path = Path(*PurePosixPath(key).parts)
    return path if not path.suffix else path.parent


def _unique_target(directory: Path, filename: str, caption_ext: str) -> Path:
    candidate = directory / filename
    if not candidate.exists() and not candidate.with_suffix(caption_ext).exists():
        return candidate
    stem, suffix = candidate.stem, candidate.suffix
    counter = 1
    while True:
        value = directory / f"{stem}_{counter:04d}{suffix}"
        if not value.exists() and not value.with_suffix(caption_ext).exists():
            return value
        counter += 1


def _unique_caption_target(directory: Path, stem: str, caption_ext: str) -> Path:
    candidate = directory / f"{stem}{caption_ext}"
    if not candidate.exists():
        return candidate
    counter = 1
    while True:
        value = directory / f"{stem}_{counter:04d}{caption_ext}"
        if not value.exists():
            return value
        counter += 1


def export_dataset(
    items: Iterable[Any],
    out_root: str | os.PathLike[str],
    *,
    only_approved: bool = True,
    copy_media: bool = True,
    caption_ext: str = ".txt",
    flat: bool = True,
    include_caption_only: bool = False,
) -> ExportReport:
    """Export approved pairs and optional caption-only items with explicit counts."""

    root = normalize_path(out_root)
    root.mkdir(parents=True, exist_ok=True)
    extension = str(caption_ext or ".txt")
    if not extension.startswith("."):
        extension = "." + extension
    exported = skipped = rejected = no_media = 0
    media_outputs: list[Path] = []
    caption_outputs: list[Path] = []
    errors: list[str] = []
    for item in items:
        raw_media = _item_value(item, "media_path", "path", "media", "file", default=item if isinstance(item, (str, os.PathLike)) else None)
        if raw_media is None:
            raw_media = _item_value(item, "source_media_path", default=None)
        raw_caption_path = _item_value(item, "caption_path", default=None)
        try:
            media = normalize_path(raw_media) if raw_media else None
            if media is not None and not media.is_file():
                media = None
            approval_reference = media
            if approval_reference is None and raw_caption_path:
                approval_reference = normalize_path(raw_caption_path)
            if approval_reference is None:
                approval_reference = root / "caption.txt"
            if only_approved and not _approved(item, approval_reference):
                rejected += 1
                continue
            if media is None:
                no_media += 1
                if not include_caption_only:
                    skipped += 1
                    continue
            caption_reference = media or approval_reference
            caption, caption_source = _caption_for(item, caption_reference)
            if caption is None:
                skipped += 1
                errors.append(f"Missing caption for {caption_reference}")
                continue
            destination_dir = root if flat else root / _relative_parent(item)
            destination_dir.mkdir(parents=True, exist_ok=True)
            if media is not None:
                safe_name = sanitize_filename(media.name)
                media_target = _unique_target(destination_dir, safe_name, extension)
                caption_target = media_target.with_suffix(extension)
                if copy_media:
                    shutil.copy2(media, media_target)
                    media_outputs.append(media_target)
                else:
                    media_outputs.append(media)
            else:
                source_name = caption_source.name if caption_source is not None else Path(str(raw_caption_path or "caption.txt")).name
                safe_stem = Path(sanitize_filename(source_name)).stem or "caption"
                caption_target = _unique_caption_target(destination_dir, safe_stem, extension)
            OutputWriter().write_text(
                caption_target,
                caption + ("\n" if caption and not caption.endswith("\n") else ""),
            )
            caption_outputs.append(caption_target)
            exported += 1
        except Exception as exc:
            skipped += 1
            errors.append(f"{raw_media or '<unknown>'}: {exc}")
    return ExportReport(
        out_root=root,
        exported=exported,
        skipped=skipped,
        rejected=rejected,
        no_media=no_media if not include_caption_only else 0,
        media_files=media_outputs,
        caption_files=caption_outputs,
        errors=errors,
    )


__all__ = [
    "DatasetFolder",
    "ExportReport",
    "discover_dataset_folders",
    "export_dataset",
    "read_flags",
    "write_flags",
    "write_kohya_musubi_toml",
]
