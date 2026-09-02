from __future__ import annotations

import struct
import zipfile
from pathlib import Path

from vcap.core.archive import ExtractReport, extract_zip


def _add_unflagged_utf8_name(path: Path, name: str, data: bytes) -> None:
    encoded = name.encode("utf-8")
    placeholder = b"x" * len(encoded)
    with zipfile.ZipFile(path, "a") as archive:
        archive.writestr(placeholder.decode("ascii"), data)
    raw = bytearray(path.read_bytes())
    cursor = 0
    replacements = 0
    while True:
        index = raw.find(placeholder, cursor)
        if index < 0:
            break
        raw[index : index + len(encoded)] = encoded
        if raw[index - 30 : index - 26] == b"PK\x03\x04":
            flags_offset = index - 24
        elif raw[index - 46 : index - 42] == b"PK\x01\x02":
            flags_offset = index - 38
        else:
            cursor = index + len(encoded)
            continue
        flags = struct.unpack_from("<H", raw, flags_offset)[0] & ~0x800
        struct.pack_into("<H", raw, flags_offset, flags)
        replacements += 1
        cursor = index + len(encoded)
    assert replacements == 2
    path.write_bytes(raw)


def test_extract_zip_unicode_security_layout_and_byte_counts(tmp_path: Path) -> None:
    archive_path = tmp_path / "input.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("日本語/görüntü.txt", b"caption")
        archive.writestr("nested/deep/data.bin", b"12345")
        archive.writestr("../evil.txt", b"evil")
        archive.writestr("__MACOSX/._junk", b"junk")
        archive.writestr("nested/.DS_Store", b"junk")
    _add_unflagged_utf8_name(archive_path, "café.txt", b"coffee")

    progress: list[tuple[int, int, str]] = []
    destination = tmp_path / "out"
    report = extract_zip(
        archive_path,
        destination,
        progress_cb=lambda done, total, name: progress.append((done, total, name)),
    )

    assert isinstance(report, ExtractReport)
    assert report.destination == str(destination.resolve())
    assert report.files == 3
    assert report.total_bytes == len(b"caption12345coffee")
    assert (destination / "日本語" / "görüntü.txt").read_bytes() == b"caption"
    assert (destination / "nested" / "deep" / "data.bin").read_bytes() == b"12345"
    assert (destination / "café.txt").read_bytes() == b"coffee"
    assert not (tmp_path / "evil.txt").exists()
    assert "../evil.txt" in report.skipped
    assert "__MACOSX/._junk" in report.skipped
    assert "nested/.DS_Store" in report.skipped
    assert progress[-1][0:2] == (3, 3)
