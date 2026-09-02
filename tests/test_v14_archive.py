from __future__ import annotations

from pathlib import Path
import zipfile

from vcap.core.archive import zip_directory


def test_zip_directory_streams_unicode_and_skips_itself(tmp_path: Path) -> None:
    source = tmp_path / "dataset"
    nested = source / "日本語"
    nested.mkdir(parents=True)
    (nested / "görüntü.txt").write_text("caption", encoding="utf-8")
    archive = source / "export.zip"

    result = zip_directory(source, archive)

    assert result == archive
    with zipfile.ZipFile(result) as opened:
        assert opened.namelist() == ["日本語/görüntü.txt"]
        assert opened.read("日本語/görüntü.txt").decode("utf-8") == "caption"


def test_zip_directory_replaces_existing_archive(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "caption.txt").write_text("new", encoding="utf-8")
    archive = tmp_path / "source.zip"
    archive.write_bytes(b"old")

    zip_directory(source, archive)

    with zipfile.ZipFile(archive) as opened:
        assert opened.read("caption.txt") == b"new"
