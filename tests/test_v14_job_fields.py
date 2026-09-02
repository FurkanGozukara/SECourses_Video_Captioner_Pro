from __future__ import annotations

from pathlib import Path

from vcap.core.subprocess_runner import build_child_env
from vcap.models.registry import MODEL_SPECS
from vcap.pipeline.job import InputItem, JobSpec, OutputSpec


def test_v14_fields_round_trip_from_settings(tmp_path: Path) -> None:
    settings = {
        "model_key": "qwen3_omni_instruct_int4",
        "seed": 123456,
        "max_caption_chars": 777,
        "context_carry_words": 91,
        "fade_threshold": 8.5,
        "encode_codec": "hevc_nvenc",
        "encode_crf": 23,
        "encode_preset": "slow",
        "encode_audio_bitrate": "256k",
        "quality_frames": 16,
        "total_pixel_cap": 123_456_789,
        "oom_retries": 4,
        "pinned_ram_budget_gb": 24.5,
        "gguf_max_frames": 48,
        "gguf_jpeg_quality": 87,
        "gguf_threads": 12,
        "gguf_batch_size": 1024,
        "gguf_ubatch_size": 256,
        "gguf_flash_attn": "off",
        "gguf_cache_reuse": 128,
        "gguf_ignore_tier_context": True,
        "gguf_extra_args": "--no-mmap --mlock",
        "summarize_segments": True,
        "summary_prompt": "Summarize in {{LANGUAGE}}.",
        "batch_include_kinds": ["video", "text"],
        "batch_name_filter": "clip_*;*.md",
    }
    spec = JobSpec.from_settings(
        settings,
        [InputItem("prompt", kind="text", text_prompt_only=True, text="hello")],
        OutputSpec(kind="batch", outputs_root=tmp_path),
    )

    assert spec.generation.seed == 123456
    assert spec.post.max_caption_chars == 777
    assert spec.context_carry_words == 91
    assert spec.split.fade_threshold == 8.5
    assert (
        spec.split.encode_codec,
        spec.split.encode_crf,
        spec.split.encode_preset,
        spec.split.encode_audio_bitrate,
        spec.split.quality_frames,
    ) == ("hevc_nvenc", 23, "slow", "256k", 16)
    assert spec.preprocess.total_pixel_cap == 123_456_789
    assert spec.model.offload.pinned_ram_budget_gb == 24.5
    assert spec.runtime.oom_retries == 4
    assert spec.runtime.gguf_max_frames == 48
    assert spec.runtime.gguf_jpeg_quality == 87
    assert spec.runtime.gguf_threads == 12
    assert spec.runtime.gguf_batch_size == 1024
    assert spec.runtime.gguf_ubatch_size == 256
    assert spec.runtime.gguf_flash_attn == "off"
    assert spec.runtime.gguf_cache_reuse == 128
    assert spec.runtime.gguf_ignore_tier_context is True
    assert spec.runtime.gguf_extra_args == "--no-mmap --mlock"
    assert spec.summarize_segments is True
    assert spec.summary_prompt == "Summarize in {{LANGUAGE}}."
    assert spec.output.include_kinds == ("video", "text")
    assert spec.output.name_filter == "clip_*;*.md"

    restored = JobSpec.from_dict(spec.to_dict())
    assert restored.to_dict() == spec.to_dict()


def test_unparsable_generation_and_media_values_use_family_defaults(tmp_path: Path) -> None:
    family = MODEL_SPECS["avocado"]
    defaults = {item.name: item.default for item in family.param_schema}
    spec = JobSpec.from_settings(
        {
            "model_key": "avocado_int4",
            "temperature": "not-a-number",
            "top_p": "not-a-number",
            "top_k": "not-a-number",
            "max_new_tokens": "not-a-number",
            "fps": "not-a-number",
            "max_frames": "not-a-number",
            "max_pixels": "not-a-number",
        },
        [InputItem("prompt", kind="text", text_prompt_only=True, text="hello")],
        OutputSpec(outputs_root=tmp_path),
    )
    assert spec.generation.temperature == defaults["temperature"]
    assert spec.generation.top_p == defaults["top_p"]
    assert spec.generation.top_k == defaults["top_k"]
    assert spec.generation.max_new_tokens == defaults["max_new_tokens"]
    assert spec.preprocess.fps == defaults["fps"]
    assert spec.preprocess.max_frames == defaults["max_frames"]
    assert spec.preprocess.max_pixels == defaults["max_pixels"]


def test_build_child_env_sets_stable_cuda_device_order(monkeypatch) -> None:
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    assert build_child_env()["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
    assert build_child_env()["CUDA_DEVICE_ORDER"] == "FASTEST_FIRST"
