from __future__ import annotations

from pathlib import Path

import pytest

from vcap.models import downloads
from vcap.models.downloads import DeleteReport, delete_variant_files, variant_disk_usage


VARIANT = "qwen3_omni_instruct_int4"


def _patch_folder(monkeypatch: pytest.MonkeyPatch, root: Path, folder: Path) -> None:
    monkeypatch.setattr(downloads, "MODELS_DIR", root)
    monkeypatch.setattr(downloads, "resolve_model_dir", lambda _key: folder)


def test_variant_usage_and_delete_include_partial_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "models"
    folder = root / VARIANT
    nested = folder / "nested"
    nested.mkdir(parents=True)
    (folder / "model.safetensors.part").write_bytes(b"partial")
    (nested / "config.json").write_bytes(b"{}")
    _patch_folder(monkeypatch, root, folder)

    assert variant_disk_usage(VARIANT) == 9
    report = delete_variant_files(VARIANT)

    assert isinstance(report, DeleteReport)
    assert report.variant_key == VARIANT
    assert report.folder == str(folder.resolve())
    assert report.files_removed == 2
    assert report.bytes_freed == 9
    assert report.errors == []
    assert not folder.exists()
    assert variant_disk_usage(VARIANT) == 0


def test_delete_refuses_a_resolved_folder_outside_models_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "models"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.bin").write_bytes(b"keep")
    _patch_folder(monkeypatch, root, outside)

    with pytest.raises(ValueError, match="outside MODELS_DIR"):
        delete_variant_files(VARIANT)
    assert (outside / "keep.bin").exists()


def test_delete_collects_file_errors_without_raising(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "models"
    folder = root / VARIANT
    folder.mkdir(parents=True)
    locked = folder / "locked.part"
    locked.write_bytes(b"locked")
    _patch_folder(monkeypatch, root, folder)
    original_unlink = Path.unlink

    def fail_locked(path: Path, *args, **kwargs):
        if path.name == "locked.part":
            raise PermissionError("synthetic denial")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_locked)
    report = delete_variant_files(VARIANT)

    assert report.files_removed == 0
    assert report.bytes_freed == 0
    assert any("synthetic denial" in error for error in report.errors)
    assert locked.exists()
