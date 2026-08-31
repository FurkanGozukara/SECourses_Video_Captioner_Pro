"""Cross-platform path, filename, and media discovery helpers."""

from __future__ import annotations

import ctypes
import os
import re
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Iterable, Sequence, TypeVar

TPath = TypeVar("TPath", str, Path)
_NATURAL_PARTS = re.compile(r"(\d+)")
_WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "CONIN$",
    "CONOUT$",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}

MEDIA_EXTENSIONS: dict[str, set[str]] = {
    "video": {
        ".mp4",
        ".mkv",
        ".mov",
        ".webm",
        ".avi",
        ".m4v",
        ".ts",
        ".mts",
        ".m2ts",
        ".wmv",
        ".flv",
        ".mpg",
        ".mpeg",
        ".3gp",
        ".ogv",
    },
    "audio": {
        ".wav",
        ".mp3",
        ".flac",
        ".m4a",
        ".aac",
        ".ogg",
        ".opus",
        ".wma",
        ".aiff",
        ".aif",
        ".amr",
        ".mka",
    },
    "image": {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".bmp",
        ".tiff",
        ".tif",
        ".gif",
        ".heic",
        ".avif",
    },
    "text": {".txt", ".md"},
}


def _expand_percent_vars(value: str) -> str:
    """Expand Windows-style percent variables even when running on POSIX."""

    def replace(match: re.Match[str]) -> str:
        return os.environ.get(match.group(1), match.group(0))

    return re.sub(r"%([^%]+)%", replace, value)


def normalize_path(p: str | os.PathLike[str], must_exist: bool = False) -> Path:
    """Normalize a quoted or mixed-separator path to an absolute path."""

    value = os.fspath(p).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1].strip()
    value = _expand_percent_vars(os.path.expandvars(os.path.expanduser(value)))

    if os.name == "nt":
        value = value.replace("/", "\\")
    else:
        value = value.replace("\\", "/")

    normalized = os.path.normpath(value)
    result = Path(normalized).resolve(strict=False)
    if must_exist and not result.exists():
        raise FileNotFoundError(f"Path does not exist: {result}")
    return result


def sanitize_filename(name: str, max_len: int = 200) -> str:
    """Return a portable filename while preserving readable Unicode text."""

    limit = max(1, int(max_len))
    cleaned: list[str] = []
    for character in str(name):
        if character in '<>:"|?*/\\' or unicodedata.category(character) == "Cc":
            cleaned.append("_")
        else:
            cleaned.append(character)
    value = "".join(cleaned).strip(" .")
    if not value:
        value = "unnamed"

    base = value.split(".", 1)[0].upper()
    if base in _WINDOWS_RESERVED:
        value = f"file_{value}"

    if len(value) > limit:
        suffix = Path(value).suffix
        if suffix and len(suffix) < limit:
            value = value[: limit - len(suffix)].rstrip(" .") + suffix
        else:
            value = value[:limit].rstrip(" .")
    value = value.strip(" .")
    if not value:
        value = "_"[:limit]
    if value.split(".", 1)[0].upper() in _WINDOWS_RESERVED:
        prefix = "file_"
        value = (prefix + value)[:limit].rstrip(" .") or "_"
    return value


def natural_sort_key(s: str | os.PathLike[str]) -> tuple[tuple[int, object], ...]:
    """Build a deterministic, case-insensitive natural-sort key."""

    text = Path(os.fspath(s)).name.casefold()
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in _NATURAL_PARTS.split(text)
        if part
    )


def sort_paths_natural(paths: Iterable[TPath]) -> list[TPath]:
    """Natural-sort paths identically on Windows and POSIX."""

    return sorted(
        paths,
        key=lambda item: (
            natural_sort_key(item),
            os.fspath(item).casefold(),
            os.fspath(item),
        ),
    )


def guess_kind_by_extension(path: str | os.PathLike[str]) -> str:
    """Guess a media kind from its lowercase file extension."""

    suffix = Path(os.fspath(path)).suffix.casefold()
    for kind, extensions in MEDIA_EXTENSIONS.items():
        if suffix in extensions:
            return kind
    return "unknown"


def _is_hidden_or_system(path: Path) -> bool:
    if path.name.startswith("."):
        return True
    try:
        attributes = path.stat().st_file_attributes  # type: ignore[attr-defined]
        return bool(attributes & 0x2 or attributes & 0x4)
    except (AttributeError, OSError):
        return False


def list_media_files(
    folder: str | os.PathLike[str],
    recursive: bool = False,
    kinds: Sequence[str] = ("video", "audio", "image"),
) -> list[Path]:
    """Discover non-hidden media files, tolerating inaccessible directories."""

    root = normalize_path(folder)
    allowed = {
        extension
        for kind in kinds
        for extension in MEDIA_EXTENSIONS.get(str(kind).casefold(), set())
    }
    if not root.is_dir() or not allowed:
        return []

    found: list[Path] = []

    def scan(directory: Path) -> None:
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    item = Path(entry.path)
                    if _is_hidden_or_system(item):
                        continue
                    try:
                        if entry.is_file(follow_symlinks=True):
                            if item.suffix.casefold() in allowed:
                                found.append(item)
                        elif recursive and entry.is_dir(follow_symlinks=False):
                            scan(item)
                    except (OSError, PermissionError):
                        continue
        except (OSError, PermissionError):
            return

    scan(root)
    return sort_paths_natural(found)


def discover_allowed_paths() -> list[str]:
    """Return drive roots or mount points suitable for a file-serving allow-list."""

    candidates: list[str] = []
    if os.name == "nt":
        try:
            mask = int(ctypes.windll.kernel32.GetLogicalDrives())
        except Exception:
            mask = 0
        for index in range(26):
            if mask & (1 << index):
                candidates.append(f"{chr(65 + index)}:\\")
    else:
        candidates.extend(["/", "/mnt", "/media", "/home", "/workspace", "/Volumes"])
        mounts = Path("/proc/mounts")
        try:
            for line in mounts.read_text(encoding="utf-8", errors="replace").splitlines():
                fields = line.split()
                if len(fields) >= 2:
                    candidates.append(
                        fields[1]
                        .replace("\\040", " ")
                        .replace("\\011", "\t")
                        .replace("\\134", "\\")
                    )
        except OSError:
            pass

    result: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            resolved = str(Path(candidate).resolve(strict=False))
        except (OSError, RuntimeError):
            resolved = os.path.abspath(candidate)
        key = os.path.normcase(resolved)
        if key not in seen:
            seen.add(key)
            result.append(resolved)
    return result


def collision_safe_path(path: str | os.PathLike[str]) -> Path:
    """Return a non-existing sibling path using incrementing ``_NNNN`` suffixes."""

    target = Path(path)
    if not target.exists():
        return target
    match = re.match(r"^(.*)_(\d{4})$", target.stem)
    if match:
        base, counter = match.group(1), int(match.group(2)) + 1
    else:
        base, counter = target.stem, 1
    while True:
        candidate = target.with_name(f"{base}_{counter:04d}{target.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def is_subpath(child: str | os.PathLike[str], parent: str | os.PathLike[str]) -> bool:
    """Return whether ``child`` is equal to or lies below ``parent``."""

    try:
        normalized_child = normalize_path(child)
        normalized_parent = normalize_path(parent)
        normalized_child.relative_to(normalized_parent)
        return True
    except (OSError, ValueError, RuntimeError):
        return False


def display_path(path: str | os.PathLike[str], max_len: int = 80) -> str:
    """Render a path compactly by eliding its middle when necessary."""

    text = os.fspath(path)
    limit = max(1, int(max_len))
    if len(text) <= limit:
        return text
    if limit <= 3:
        return "." * limit
    left = (limit - 3) // 2
    right = limit - 3 - left
    return f"{text[:left]}...{text[-right:]}"


def open_in_file_manager(path: str | os.PathLike[str]) -> tuple[bool, str]:
    """Open a file or directory with the platform file manager without raising."""

    try:
        target = normalize_path(path)
        if not target.exists():
            return False, f"Path does not exist: {target}"
        if os.name == "nt":
            os.startfile(str(target))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(target)])
        else:
            subprocess.Popen(["xdg-open", str(target)])
        return True, f"Opened: {target}"
    except Exception as exc:
        return False, f"Could not open path: {exc}"


def reveal_in_file_manager(file: str | os.PathLike[str]) -> tuple[bool, str]:
    """Reveal a file in its parent file manager without raising."""

    try:
        target = normalize_path(file)
        if not target.exists():
            return False, f"Path does not exist: {target}"
        if os.name == "nt":
            subprocess.Popen(["explorer", "/select,", str(target)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", str(target)])
        else:
            parent = target if target.is_dir() else target.parent
            subprocess.Popen(["xdg-open", str(parent)])
        return True, f"Revealed: {target}"
    except Exception as exc:
        return False, f"Could not reveal path: {exc}"


__all__ = [
    "MEDIA_EXTENSIONS",
    "collision_safe_path",
    "discover_allowed_paths",
    "display_path",
    "guess_kind_by_extension",
    "is_subpath",
    "list_media_files",
    "natural_sort_key",
    "normalize_path",
    "open_in_file_manager",
    "reveal_in_file_manager",
    "sanitize_filename",
    "sort_paths_natural",
]
