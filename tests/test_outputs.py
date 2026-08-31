from __future__ import annotations

import json
from pathlib import Path

import pytest

from vcap.core.outputs import (
    MetadataBuilder,
    OutputWriter,
    RunLog,
    allocate_run_dir,
    load_metadata,
    model_short_name,
)
from vcap.core.logs import get_log


def test_model_short_names() -> None:
    assert model_short_name("qwen3_omni_instruct_int8") == "qwen3"
    assert model_short_name("timechat_bf16") == "timechat"
    assert model_short_name("avocado_int4") == "avocado"


def test_run_numbering_is_monotonic_across_kinds(tmp_path: Path) -> None:
    single = allocate_run_dir(tmp_path, "qwen3", "single")
    batch = allocate_run_dir(tmp_path, "timechat", "batch")
    again = allocate_run_dir(tmp_path, "avocado", "single")
    assert single.name == "0001_qwen3"
    assert batch.name == "batch_0002_timechat"
    assert again.name == "0003_avocado"
    with pytest.raises(ValueError):
        allocate_run_dir(tmp_path, "bad", "other")


def test_output_writer_and_metadata_round_trip(tmp_path: Path) -> None:
    writer = OutputWriter()
    paths = writer.caption_output_paths(tmp_path, "vöyager 日本語", ["txt", "json", "reasoning"])
    assert paths["reasoning"].name.endswith("_reasoning.txt")
    writer.write_text(paths["txt"], "hello 日本語")
    writer.write_json(paths["json"], {"caption": "hello"})
    assert paths["txt"].read_text(encoding="utf-8") == "hello 日本語"
    assert json.loads(paths["json"].read_text(encoding="utf-8"))["caption"] == "hello"

    builder = MetadataBuilder()
    data = builder.build(
        "1.0.0",
        {"key": "qwen3"},
        {"fps": 2},
        [{"path": "clip.mp4", "status": "done"}],
        {"total": 1.2},
        {"name": "GPU"},
        {"note": "ok"},
    )
    metadata_path = tmp_path / "metadata.json"
    builder.write(metadata_path)
    assert load_metadata(metadata_path) == data
    metadata_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        load_metadata(metadata_path)


def test_run_log_attaches_and_detaches(tmp_path: Path) -> None:
    log = get_log()
    run_dir = tmp_path / "run"
    with RunLog(run_dir):
        log.log("inside-run-log", console=False)
    before = (run_dir / "run_log.txt").read_text(encoding="utf-8")
    log.log("after-detach", console=False)
    after = (run_dir / "run_log.txt").read_text(encoding="utf-8")
    assert "inside-run-log" in before
    assert after == before
