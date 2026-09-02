from __future__ import annotations

from pathlib import Path

from vcap.pipeline.job import InputItem, JobSpec, ModelChoice, OutputSpec


def _job(tmp_path: Path, settings: dict[str, object]) -> JobSpec:
    return JobSpec.from_settings(
        settings,
        [InputItem("hello", kind="text", text_prompt_only=True, text="hello")],
        OutputSpec(outputs_root=tmp_path),
    )


def test_every_v14b_field_maps_and_round_trips(tmp_path: Path) -> None:
    spec = _job(
        tmp_path,
        {
            "no_repeat_ngram_size": 6,
            "dedupe_repeated_sentences": False,
            "gguf_min_p": 0.17,
            "gguf_repeat_last_n": 222,
            "gguf_presence_penalty": 0.4,
            "gguf_frequency_penalty": -0.3,
            "subtitle_min_cue_s": 1.2,
            "subtitle_max_line_chars": 48,
            "summary_max_new_tokens": 1536,
            "caption_join_separator": ", ",
            "adaptive_threshold": 4.5,
            "reject_black_luma": 24,
            "reject_silence_rms": 0.0045,
            "context_carry_prompt": "Earlier: {{CONTEXT}}",
            "plan_slack_mib": 768,
            "oom_degrade_factor": 0.8,
            "gguf_fit_headroom_mib": 2048,
            "gguf_startup_timeout_s": 1200,
            "gguf_stream_idle_timeout_s": 75,
        },
    )

    assert spec.generation.no_repeat_ngram_size == 6
    assert spec.post.dedupe_repeated_sentences is False
    assert spec.post.subtitle_min_cue_s == 1.2
    assert spec.post.subtitle_max_line_chars == 48
    assert spec.post.join_separator == ", "
    assert spec.preprocess.adaptive_threshold == 4.5
    assert spec.split.reject_black_luma == 24
    assert spec.split.reject_silence_rms == 0.0045
    assert spec.context_carry_prompt == "Earlier: {{CONTEXT}}"
    assert spec.summary_max_new_tokens == 1536
    assert spec.model.offload.plan_slack_mib == 768
    assert spec.runtime.oom_degrade_factor == 0.8
    assert spec.runtime.gguf_min_p == 0.17
    assert spec.runtime.gguf_repeat_last_n == 222
    assert spec.runtime.gguf_presence_penalty == 0.4
    assert spec.runtime.gguf_frequency_penalty == -0.3
    assert spec.runtime.gguf_fit_headroom_mib == 2048
    assert spec.runtime.gguf_startup_timeout_s == 1200
    assert spec.runtime.gguf_stream_idle_timeout_s == 75
    assert JobSpec.from_dict(spec.to_dict()).to_dict() == spec.to_dict()


def test_v14b_field_clamps_and_default_variant(tmp_path: Path) -> None:
    spec = _job(
        tmp_path,
        {
            "no_repeat_ngram_size": 99,
            "subtitle_min_cue_s": -1,
            "subtitle_max_line_chars": 999,
            "summary_max_new_tokens": 1,
            "caption_join_separator": "0123456789abcdefghijklmnop",
            "adaptive_threshold": 99,
            "reject_black_luma": -5,
            "reject_silence_rms": 4,
            "plan_slack_mib": 99_999,
            "oom_degrade_factor": 0.1,
            "gguf_min_p": 7,
            "gguf_repeat_last_n": -1,
            "gguf_presence_penalty": 9,
            "gguf_frequency_penalty": -9,
            "gguf_fit_headroom_mib": -2,
            "gguf_startup_timeout_s": 1,
            "gguf_stream_idle_timeout_s": 9999,
        },
    )

    assert ModelChoice().variant_key == "qwen3_omni_instruct_int4"
    assert spec.generation.no_repeat_ngram_size == 20
    assert spec.post.subtitle_min_cue_s == 0.0
    assert spec.post.subtitle_max_line_chars == 200
    assert spec.post.join_separator == "0123456789abcdef"
    assert spec.summary_max_new_tokens == 64
    assert spec.preprocess.adaptive_threshold == 50.0
    assert spec.split.reject_black_luma == 0
    assert spec.split.reject_silence_rms == 0.1
    assert spec.model.offload.plan_slack_mib == 8192
    assert spec.runtime.oom_degrade_factor == 0.5
    assert spec.runtime.gguf_min_p == 1.0
    assert spec.runtime.gguf_repeat_last_n == 0
    assert spec.runtime.gguf_presence_penalty == 2.0
    assert spec.runtime.gguf_frequency_penalty == -2.0
    assert spec.runtime.gguf_fit_headroom_mib == 0
    assert spec.runtime.gguf_startup_timeout_s == 60
    assert spec.runtime.gguf_stream_idle_timeout_s == 3600
