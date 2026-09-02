"""Small, dependency-free archive helpers."""

from __future__ import annotations

import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

from .paths import normalize_path


@dataclass(frozen=True)
class ExtractReport:
    """Summary of a streamed ZIP extraction."""

    destination: str
    files: int
    total_bytes: int
    skipped: list[str]


def _zip_member_name(info: zipfile.ZipInfo) -> str:
    name = str(info.filename)
    if info.flag_bits & 0x800:
        return name
    try:
        repaired = name.encode("cp437").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return name
    return repaired


def _zip_target(destination: Path, name: str) -> Path | None:
    normalized = name.replace("\\", "/")
    relative = PurePosixPath(normalized)
    if not normalized or not relative.parts:
        return None
    if relative.is_absolute() or any(part == ".." for part in relative.parts):
        return None
    if relative.parts and (":" in relative.parts[0] or relative.parts[0] in {"", "."}):
        return None
    target = (destination / Path(*relative.parts)).resolve(strict=False)
    if target == destination:
        return None
    try:
        target.relative_to(destination)
    except ValueError:
        return None
    return target


def extract_zip(
    src: str | os.PathLike[str],
    dst: str | os.PathLike[str],
    *,
    progress_cb: Any = None,
) -> ExtractReport:
    """Safely stream a ZIP into ``dst`` while reporting rejected members."""

    source = normalize_path(src, must_exist=True)
    destination = normalize_path(dst)
    destination.mkdir(parents=True, exist_ok=True)
    destination = destination.resolve(strict=False)
    skipped: list[str] = []
    extracted: list[tuple[zipfile.ZipInfo, str, Path]] = []
    with zipfile.ZipFile(source, mode="r") as archive:
        for info in archive.infolist():
            name = _zip_member_name(info)
            normalized = name.replace("\\", "/")
            parts = PurePosixPath(normalized).parts
            if info.is_dir() or normalized.endswith("/"):
                continue
            if "__MACOSX" in parts or (parts and parts[-1] == ".DS_Store"):
                skipped.append(name)
                continue
            target = _zip_target(destination, name)
            if target is None:
                skipped.append(name)
                continue
            extracted.append((info, name, target))

        files = 0
        total_bytes = 0
        total_files = len(extracted)
        for info, name, target in extracted:
            target.parent.mkdir(parents=True, exist_ok=True)
            checked = target.resolve(strict=False)
            try:
                checked.relative_to(destination)
            except ValueError:
                skipped.append(name)
                continue
            written = 0
            with archive.open(info, mode="r") as reader, checked.open("wb") as writer:
                while True:
                    chunk = reader.read(1024 * 1024)
                    if not chunk:
                        break
                    writer.write(chunk)
                    written += len(chunk)
            files += 1
            total_bytes += written
            if callable(progress_cb):
                progress_cb(files, total_files, name)
    return ExtractReport(str(destination), files, total_bytes, skipped)


def zip_directory(
    src: str | os.PathLike[str],
    dst: str | os.PathLike[str],
) -> Path:
    """Stream every file below ``src`` into an atomic UTF-8 ZIP archive."""

    source = normalize_path(src, must_exist=True)
    if not source.is_dir():
        raise NotADirectoryError(source)
    target = normalize_path(dst)
    target.parent.mkdir(parents=True, exist_ok=True)
    target_resolved = target.resolve(strict=False)

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
        temporary_path = Path(temporary_name)
        with zipfile.ZipFile(
            temporary_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            allowZip64=True,
        ) as archive:
            for path in sorted(source.rglob("*"), key=lambda value: value.as_posix().casefold()):
                if not path.is_file():
                    continue
                resolved = path.resolve(strict=False)
                if resolved in {target_resolved, temporary_path.resolve(strict=False)}:
                    continue
                archive.write(path, arcname=path.relative_to(source).as_posix())
        os.replace(temporary_path, target)
        temporary_name = None
        return target
    finally:
        if temporary_name:
            try:
                Path(temporary_name).unlink(missing_ok=True)
            except OSError:
                pass


__all__ = ["ExtractReport", "extract_zip", "zip_directory"]
