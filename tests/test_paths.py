from __future__ import annotations

import os
from pathlib import Path

import pytest

from vcap.core.paths import (
    collision_safe_path,
    display_path,
    guess_kind_by_extension,
    is_subpath,
    list_media_files,
    natural_sort_key,
    normalize_path,
    sanitize_filename,
    sort_paths_natural,
)


def test_normalize_quotes_environment_and_mixed_separators(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    nested = tmp_path / "mixed" / "folder"
    nested.mkdir(parents=True)
    monkeypatch.setenv("VCAP_TEST_ROOT", str(tmp_path))
    if os.name == "nt":
        raw = f'  "%VCAP_TEST_ROOT%/mixed\\folder"  '
    else:
        raw = '  "$VCAP_TEST_ROOT\\mixed//folder"  '
    assert normalize_path(raw, must_exist=True) == nested.resolve()
    missing = normalize_path(f'"{tmp_path / "not yet"}"')
    assert missing.is_absolute()
    with pytest.raises(FileNotFoundError):
        normalize_path(missing, must_exist=True)


def test_sanitize_unicode_reserved_and_controls() -> None:
    value = sanitize_filename(' vöyager 日本語<>:"/\\|?*\x00.mp4. ')
    assert "vöyager 日本語" in value
    assert not any(character in value for character in '<>:"/\\|?*\x00')
    assert not value.endswith((".", " "))
    assert sanitize_filename("CON.txt").casefold().startswith("file_con")
    assert sanitize_filename("LPT9").casefold().startswith("file_lpt9")
    assert sanitize_filename("***")
    assert len(sanitize_filename("x" * 400 + ".txt", max_len=40)) <= 40


def test_natural_sort_and_media_discovery(tmp_path: Path) -> None:
    names = ["clip10.mp4", "clip2.mp4", "clip1.mp4", "vöyager 日本語.mp4"]
    for name in names:
        (tmp_path / name).write_bytes(b"")
    (tmp_path / ".hidden.mp4").write_bytes(b"")
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "clip3.wav").write_bytes(b"")

    direct = list_media_files(tmp_path)
    assert [path.name for path in direct[:3]] == ["clip1.mp4", "clip2.mp4", "clip10.mp4"]
    assert ".hidden.mp4" not in {path.name for path in direct}
    recursive = list_media_files(tmp_path, recursive=True, kinds=("video", "audio"))
    assert nested / "clip3.wav" in recursive
    assert guess_kind_by_extension("sample.GIF") == "image"
    assert guess_kind_by_extension("sample.bin") == "unknown"


def test_path_helpers_and_collision(tmp_path: Path) -> None:
    original = tmp_path / "name.txt"
    original.write_text("x", encoding="utf-8")
    first = collision_safe_path(original)
    assert first.name == "name_0001.txt"
    first.write_text("x", encoding="utf-8")
    assert collision_safe_path(first).name == "name_0002.txt"
    assert is_subpath(first, tmp_path)
    assert not is_subpath(tmp_path, first)
    assert len(display_path("a" * 100, 20)) == 20
    sorted_names = sort_paths_natural(["a11", "a2", "a1"])
    assert sorted_names == ["a1", "a2", "a11"]
    assert natural_sort_key("a2") < natural_sort_key("a10")
