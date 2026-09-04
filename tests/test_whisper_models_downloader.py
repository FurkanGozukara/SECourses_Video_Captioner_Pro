from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def downloader_module():
    path = Path(__file__).resolve().parents[2] / "Models_Downloader.py"
    spec = importlib.util.spec_from_file_location("whisper_models_downloader", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_whisper_catalog_lists_all_aliases(downloader_module, tmp_path: Path) -> None:
    statuses = downloader_module.collect_whisper_statuses(tmp_path)

    assert len(statuses) == 17
    assert statuses[0]["key"] == "whisper:large-v1"
    assert statuses[-1]["key"] == "whisper:tiny.en"
    assert all(status["family"] == "whisper" for status in statuses)


@pytest.mark.parametrize("flag", ["--ensure", "--verify"])
def test_cli_routes_whisper_keys_without_caption_downloads(
    downloader_module,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    flag: str,
) -> None:
    calls: list[tuple[str, Path]] = []

    def routed(key: str, target_root: Path) -> bool:
        calls.append((key, target_root))
        return True

    monkeypatch.setattr(downloader_module, "ensure_whisper_model", routed)
    monkeypatch.setattr(downloader_module, "verify_whisper_model", routed)

    exit_code = downloader_module.run_cli(
        [flag, "whisper:tiny", "--target-root", str(tmp_path)]
    )

    assert exit_code == 0
    assert calls == [("whisper:tiny", tmp_path.resolve())]
