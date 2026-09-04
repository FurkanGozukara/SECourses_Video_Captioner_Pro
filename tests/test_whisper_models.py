from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from vcap.core.subprocess_runner import CancelledError
from vcap.whisper import models


def test_catalog_order_and_hardcoded_sizes() -> None:
    assert [item.alias for item in models.WHISPER_MODELS] == [
        "large-v1",
        "large-v3",
        "large-v3-turbo",
        "large-v2",
        "distil-large-v3.5",
        "distil-large-v3",
        "distil-large-v2",
        "medium",
        "medium.en",
        "distil-medium.en",
        "small",
        "small.en",
        "distil-small.en",
        "base",
        "base.en",
        "tiny",
        "tiny.en",
    ]
    assert all(item.repo_id and "/" in item.repo_id for item in models.WHISPER_MODELS)
    assert all(item.size_bytes > 0 for item in models.WHISPER_MODELS)
    assert models.get_model("LARGE-V1").repo_id == "Systran/faster-whisper-large-v1"
    assert models.get_model("missing") is None


def test_model_paths_labels_and_readiness(tmp_path: Path) -> None:
    folder = models.model_dir("owner/repo", tmp_path)
    assert folder == (tmp_path / "whisper" / "owner--repo").resolve()
    assert not models.is_model_ready("owner/repo", tmp_path)

    folder.mkdir(parents=True)
    (folder / "config.json").write_text("{}", encoding="utf-8")
    (folder / "model.bin").write_bytes(b"model")
    assert models.is_model_ready("owner/repo", tmp_path)
    assert models.local_size_bytes("owner/repo", tmp_path) == 7

    partial = folder / "model.bin.incomplete"
    partial.write_bytes(b"")
    assert not models.is_model_ready("owner/repo", tmp_path)
    partial.unlink()
    (folder / "nested").mkdir()
    (folder / "nested" / "download.part").write_bytes(b"x")
    assert not models.is_model_ready("owner/repo", tmp_path)


def test_format_size_and_choices(tmp_path: Path) -> None:
    assert models.format_size(3_089_578_414) == "3.09 GB"
    assert models.format_size(78_203_619) == "78.2 MB"
    choices = models.model_choices(tmp_path)
    assert len(choices) == len(models.WHISPER_MODELS)
    assert choices[0][1] == "large-v1"
    assert "3.09 GB" in choices[0][0]


def test_download_model_uses_snapshot_and_reports_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict] = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        target = Path(kwargs["local_dir"])
        (target / "config.json").write_text("{}", encoding="utf-8")
        (target / "model.bin").write_bytes(b"model")
        return str(target)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    progress: list[dict] = []
    result = models.download_model("tiny", tmp_path, progress_cb=progress.append)

    assert result == models.model_dir("tiny", tmp_path)
    assert models.is_model_ready("tiny", tmp_path)
    assert calls[0]["repo_id"] == "Systran/faster-whisper-tiny"
    assert calls[0]["allow_patterns"] == models.MODEL_FILE_PATTERNS
    assert progress[-1]["fraction"] == 1.0
    assert set(progress[-1]) == {
        "fraction",
        "bytes",
        "total",
        "speed_bps",
        "message",
        "file",
    }


def test_download_cancelled_before_network_keeps_not_ready(tmp_path: Path) -> None:
    with pytest.raises(CancelledError):
        models.download_model("tiny", tmp_path, cancel_check=lambda: True)
    assert not models.is_model_ready("tiny", tmp_path)


def test_download_checks_remaining_disk_with_five_percent_headroom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        models.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=100, used=99, free=1),
    )

    with pytest.raises(OSError, match="Not enough free disk space"):
        models.download_model("tiny", tmp_path)


def test_download_cancelled_after_transfer_stays_not_ready_until_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_snapshot_download(**kwargs):
        target = Path(kwargs["local_dir"])
        (target / "config.json").write_text("{}", encoding="utf-8")
        (target / "model.bin").write_bytes(b"model")
        return str(target)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    checks = {"count": 0}

    def cancel_after_transfer() -> bool:
        checks["count"] += 1
        return checks["count"] >= 3

    with pytest.raises(CancelledError):
        models.download_model("tiny", tmp_path, cancel_check=cancel_after_transfer)
    assert not models.is_model_ready("tiny", tmp_path)
    assert (models.model_dir("tiny", tmp_path) / ".cancelled.incomplete").is_file()

    models.download_model("tiny", tmp_path, cancel_check=lambda: False)
    assert models.is_model_ready("tiny", tmp_path)


def test_delete_model_returns_removed_bytes(tmp_path: Path) -> None:
    folder = models.model_dir("tiny", tmp_path)
    folder.mkdir(parents=True)
    (folder / "config.json").write_bytes(b"{}")
    (folder / "model.bin").write_bytes(b"12345")
    assert models.delete_model("tiny", tmp_path) == 7
    assert not folder.exists()


def test_refresh_sizes_matches_allow_patterns(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeApi:
        def model_info(self, _repo_id, *, files_metadata):
            assert files_metadata is True
            return SimpleNamespace(
                siblings=[
                    SimpleNamespace(rfilename="config.json", size=2),
                    SimpleNamespace(rfilename="model.bin", size=10),
                    SimpleNamespace(rfilename="README.md", size=1000),
                ]
            )

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)
    refreshed = models.refresh_sizes()
    assert set(refreshed) == {item.alias for item in models.WHISPER_MODELS}
    assert set(refreshed.values()) == {12}
