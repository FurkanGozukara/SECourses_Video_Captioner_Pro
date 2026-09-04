"""Regression tests for the v1.6.0 follow-up fixes made after the real-model Chrome pass."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from vcap import PRESETS_DEFAULT_DIR
from vcap.core.paths import exclude_caption_sidecars, list_media_files
from vcap.models import torch_compile
from vcap.models.registry import MODEL_SPECS
from vcap.pipeline.job import ItemResult, JobResult
from vcap.ui import components
from vcap.ui.app import build_app
from vcap.ui.tabs import caption_tab


@pytest.fixture(scope="module")
def app() -> Any:
    demo = build_app()
    try:
        yield demo
    finally:
        demo.vcap_context.pipeline_client.shutdown()


def _dataset_layout(root: Path) -> None:
    (root / "clip.mp4").write_bytes(b"0")
    (root / "clip.txt").write_text("merged caption", encoding="utf-8")
    (root / "clip_transcript.txt").write_text("transcript", encoding="utf-8")
    (root / "notes.txt").write_text("a real text prompt", encoding="utf-8")
    (root / "video_caption").mkdir()
    (root / "video_caption" / "clip.txt").write_text("video part", encoding="utf-8")
    (root / "audio_caption").mkdir()
    (root / "audio_caption" / "clip.txt").write_text("audio part", encoding="utf-8")
    nested = root / "sub ünicode"
    nested.mkdir()
    (nested / "fırlatma 日本語.mp4").write_bytes(b"0")
    (nested / "fırlatma 日本語.txt").write_text("merged", encoding="utf-8")
    (nested / "fırlatma 日本語_0002.txt").write_text("stale collision file", encoding="utf-8")


# N1-5: a captioned dataset folder must never feed its own caption files back as text inputs.
def test_exclude_caption_sidecars_keeps_media_and_real_text_only(tmp_path: Path) -> None:
    _dataset_layout(tmp_path)
    found = list_media_files(tmp_path, recursive=True, kinds=("video", "audio", "image", "text"))
    kept = exclude_caption_sidecars(found)
    names = sorted(path.name for path in kept)
    assert names == ["clip.mp4", "fırlatma 日本語.mp4", "notes.txt"]


def test_folder_scan_ignores_caption_parts_and_sidecars(tmp_path: Path) -> None:
    _dataset_layout(tmp_path)
    selected, summary = components._folder_scan(str(tmp_path), True)
    names = sorted(Path(item).name for item in selected)
    assert names == ["clip.mp4", "fırlatma 日本語.mp4", "notes.txt"]
    assert "2 videos" in summary and "1 texts" in summary


# D32: prompt-preset token overrides are clamped to the selected family's cap.
def test_select_prompt_clamps_token_override_to_family_cap(app: Any) -> None:
    handler = app.vcap_context.states["caption_prompt_select_handler"]
    variables = ["ohwx", "English", "English", "English", "detailed", "", "", "person"]
    outputs = handler("timechat_flatten_wan", "qwen3_omni_instruct_int4", *variables)
    token_update = outputs[8]
    assert isinstance(token_update, dict)
    # The slider keeps the global maximum (so bound changes can never invalidate
    # a value in flight); the value itself is clamped to the family cap.
    assert token_update["maximum"] == caption_tab._GLOBAL_MAX_NEW_TOKENS
    assert 1 <= token_update["value"] <= 8192
    outputs = handler("qwen3_video_dense", "avocado_int4", *variables)
    token_update = outputs[8]
    assert isinstance(token_update, dict)
    avocado_cap = int(MODEL_SPECS["avocado"].limits.max_new_tokens_cap)
    assert token_update["value"] <= avocado_cap


# D31: a preset that names a tier-hidden variant ships choices that contain it.
def test_preset_model_key_adapter_includes_hidden_variant(app: Any) -> None:
    adapter = app.vcap_context.states["preset_value_adapters"]["model_key"]
    update = adapter({"model_key": "qwen3_omni_instruct_gguf_q8", "show_all_variants": False})
    assert isinstance(update, dict)
    assert update["value"] == "qwen3_omni_instruct_gguf_q8"
    assert any(key == "qwen3_omni_instruct_gguf_q8" for _, key in update["choices"])
    fallback = adapter({"model_key": "not_a_variant"})
    assert fallback["value"] == caption_tab._INITIAL_VARIANT


# N1-6: the dataset presets scan subfolders by default.
@pytest.mark.parametrize(
    "name",
    [
        "Dataset clips - add Whisper audio captions to existing captions.json",
        "Dataset clips - video + audio captions (Qwen3-Omni + Whisper).json",
        "Dataset clips - video + sound captions (Qwen3-Omni + Captioner).json",
    ],
)
def test_dataset_presets_scan_subfolders(name: str) -> None:
    data = json.loads((PRESETS_DEFAULT_DIR / name).read_text(encoding="utf-8"))
    assert data["settings"]["batch_recursive"] is True


# D30: Dynamo keeps recompiling the decode loop instead of silently going eager.
def test_recompile_limits_are_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    import types

    torch = pytest.importorskip("torch")
    config = getattr(getattr(torch, "_dynamo", None), "config", None)
    if config is None:
        pytest.skip("torch._dynamo unavailable")
    names = [n for n in ("cache_size_limit", "recompile_limit") if hasattr(config, n)]
    assert names
    for name in names:
        monkeypatch.setattr(config, name, 8, raising=False)
    messages: list[str] = []
    torch_compile._raise_recompile_limits(messages.append)
    for name in names:
        assert getattr(config, name) >= torch_compile._RECOMPILE_LIMIT
    assert any("recompile limit" in message for message in messages)


# N1-8: the files suffix only names files for single runs.
def _result(items: list[ItemResult]) -> JobResult:
    return JobResult(items=items, counts={"total": len(items), "done": len(items)}, run_dir="", metadata_path="", elapsed=1.0)


def test_result_summary_names_files_only_for_single_runs() -> None:
    single = _result([ItemResult(0, "a.mp4", "video", "done", merged_caption_path="/x/a.txt")])
    batch = _result(
        [
            ItemResult(0, "a.mp4", "video", "done", merged_caption_path="/x/a.txt"),
            ItemResult(1, "b.mp4", "video", "done", merged_caption_path="/x/b.txt"),
        ]
    )
    _, single_message, _, _ = caption_tab._result_summary(single)
    _, batch_message, _, _ = caption_tab._result_summary(batch)
    assert "files:" in single_message
    assert "files:" not in batch_message
