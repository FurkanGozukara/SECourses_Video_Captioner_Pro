from __future__ import annotations

import json
import os
from pathlib import Path

from vcap.core.outputs import RunSummary, list_recent_runs


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_list_recent_runs_handles_all_run_kinds_and_damaged_metadata(tmp_path: Path) -> None:
    single = tmp_path / "0001_qwen3"
    _write_json(
        single / "metadata.json",
        {
            "model_info": {"variant_key": "qwen3_omni_instruct_int4"},
            "items_results": [{"status": "done"}],
        },
    )
    (single / "caption.txt").write_text("A Unicode caption: görüntü 日本語", encoding="utf-8")

    (single / "aaa").mkdir()
    (single / "aaa" / "nested.txt").write_text("nested must not win", encoding="utf-8")

    batch = tmp_path / "batch_0002_timechat"
    _write_json(
        batch / "metadata.json",
        {
            "model_info": {"variant_key": "timechat_int4"},
            "items_results": [
                {"status": "done"},
                {"status": "done"},
                {"status": "failed"},
            ],
        },
    )
    nested = batch / "nested"
    nested.mkdir()
    (nested / "clip.txt").write_text("nested preview", encoding="utf-8")

    chat = tmp_path / "0003_chat_qwen3"
    _write_json(
        chat / "chat.json",
        {"metadata": {"model_key": "qwen3_omni_thinking_int4"}, "messages": [{}, {}]},
    )

    damaged = tmp_path / "0004_damaged"
    damaged.mkdir()
    (damaged / "metadata.json").write_text("{broken", encoding="utf-8")
    (damaged / "caption.txt").write_text("still visible", encoding="utf-8")

    junk = tmp_path / "random-folder"
    junk.mkdir()
    (junk / "caption.txt").write_text("not a run", encoding="utf-8")

    for index, folder in enumerate((single, batch, chat, damaged), start=1):
        os.utime(folder, (100 + index, 100 + index))

    runs = list_recent_runs(tmp_path)

    assert all(isinstance(item, RunSummary) for item in runs)
    assert [item.name for item in runs] == [
        "0004_damaged",
        "0003_chat_qwen3",
        "batch_0002_timechat",
        "0001_qwen3",
    ]
    by_name = {item.name: item for item in runs}
    assert by_name["0001_qwen3"].kind == "single"
    assert by_name["0001_qwen3"].model_key == "qwen3_omni_instruct_int4"
    assert by_name["0001_qwen3"].counts == {"done": 1}
    assert by_name["0001_qwen3"].preview.startswith("A Unicode caption")
    assert by_name["batch_0002_timechat"].kind == "batch"
    assert by_name["batch_0002_timechat"].items == 3
    assert by_name["batch_0002_timechat"].counts == {"done": 2, "failed": 1}
    assert by_name["batch_0002_timechat"].preview == "nested preview"
    assert by_name["0003_chat_qwen3"].kind == "chat"
    assert by_name["0003_chat_qwen3"].items == 2
    assert by_name["0004_damaged"].model_key == ""
    assert by_name["0004_damaged"].counts == {}
    assert len(by_name["0001_qwen3"].preview) <= 160
    assert list_recent_runs(tmp_path, limit=2) == runs[:2]


def test_list_recent_runs_missing_root_is_empty(tmp_path: Path) -> None:
    assert list_recent_runs(tmp_path / "missing") == []
