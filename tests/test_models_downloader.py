from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def downloader_module():
    downloader_path = Path(__file__).resolve().parents[2] / "Models_Downloader.py"
    spec = importlib.util.spec_from_file_location("vcap_test_models_downloader_paths", downloader_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_catalog_hf_subfolders_are_the_fifteen_root_keys(downloader_module) -> None:
    expected_keys = {
        "timechat_bf16",
        "timechat_int8",
        "timechat_int4",
        "avocado_bf16",
        "avocado_int8",
        "avocado_int4",
        "qwen3_omni_instruct_bf16",
        "qwen3_omni_instruct_int8",
        "qwen3_omni_instruct_int4",
        "qwen3_omni_thinking_bf16",
        "qwen3_omni_thinking_int8",
        "qwen3_omni_thinking_int4",
        "qwen3_omni_captioner_bf16",
        "qwen3_omni_captioner_int8",
        "qwen3_omni_captioner_int4",
    }
    actual = {
        key: spec["hf_subfolder"]
        for key, spec in downloader_module.MODEL_CATALOG.items()
        if "hf_subfolder" in spec
    }

    assert actual == {key: key for key in expected_keys}


def test_remote_index_uses_and_caches_alternate_subfolder(
    downloader_module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls: list[str] = []

    class FakeApi:
        repo_info_calls = 0

        def repo_info(self, **_kwargs):
            self.repo_info_calls += 1
            return SimpleNamespace(sha="fixture-commit")

        def list_repo_tree(self, _repo_id, *, path_in_repo: str, **_kwargs):
            calls.append(path_in_repo)
            if path_in_repo == "timechat_bf16":
                error = RuntimeError("not found")
                error.response = SimpleNamespace(status_code=404)
                raise error
            return [
                SimpleNamespace(
                    path=f"{path_in_repo}/config.json",
                    size=2,
                    lfs=SimpleNamespace(sha256="a" * 64),
                )
            ]

    api = FakeApi()
    monkeypatch.setattr(downloader_module, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(downloader_module, "HfApi", lambda: api)
    caplog.set_level("INFO", logger=downloader_module.__name__)

    first = downloader_module.load_remote_index(
        "timechat_bf16",
        downloader_module.DEFAULT_REPO_ID,
        "timechat_bf16",
    )
    cache = json.loads((tmp_path / "remote_index_timechat_bf16.json").read_text(encoding="utf-8"))

    assert calls == ["timechat_bf16", "Video_Captioner_Pro/timechat_bf16"]
    assert first[0][0].filename == "Video_Captioner_Pro/timechat_bf16/config.json"
    assert first[0][1] == "config.json"
    assert cache["subfolder"] == "Video_Captioner_Pro/timechat_bf16"
    assert "returned 404; trying" in caplog.text

    second = downloader_module.load_remote_index(
        "timechat_bf16",
        downloader_module.DEFAULT_REPO_ID,
        "timechat_bf16",
    )

    assert [(item.filename, relative) for item, relative in second] == [
        (item.filename, relative) for item, relative in first
    ]
    assert calls == ["timechat_bf16", "Video_Captioner_Pro/timechat_bf16"]
    assert api.repo_info_calls == 1


def test_model_disk_space_preflight_uses_remaining_bytes_and_margin(
    downloader_module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        downloader_module.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=10_000, used=9_900, free=100),
    )

    with pytest.raises(downloader_module.DownloadError) as exc_info:
        downloader_module._check_model_disk_space(
            "timechat_int4",
            tmp_path,
            expected_bytes=1_000,
            already_present_bytes=200,
        )

    message = str(exc_info.value)
    assert message.startswith("Not enough free disk space for timechat_int4: need about ")
    assert " GB more on " in message


def test_ensure_disk_failure_emits_vcap_status_error(
    downloader_module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    remote = downloader_module.RemoteFile(
        "repo",
        "timechat_int4/weights.bin",
        1_000,
        "sha256",
        "a" * 64,
        "commit",
    )
    monkeypatch.setattr(
        downloader_module,
        "load_remote_index",
        lambda *_args, **_kwargs: [(remote, "weights.bin")],
    )
    monkeypatch.setattr(
        downloader_module.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=10_000, used=9_900, free=100),
    )

    class FakeDownloader:
        def set_status_key(self, _key):
            return None

    assert not downloader_module.ensure_model("timechat_int4", tmp_path, FakeDownloader())
    payloads = [
        json.loads(line.removeprefix("VCAP_STATUS "))
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("VCAP_STATUS ")
    ]
    assert payloads[-1]["state"] == "error"
    assert payloads[-1]["message"].startswith(
        "Not enough free disk space for timechat_int4:"
    )
