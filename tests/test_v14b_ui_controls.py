from __future__ import annotations

from vcap.ui.app import build_app
from vcap.ui.tabs.caption_tab import gguf_control_updates


EXPECTED = {
    "no_repeat_ngram_size": (0, "int", 0, 20, None, "generation"),
    "dedupe_repeated_sentences": (True, "bool", None, None, None, "postprocessing"),
    "gguf_min_p": (0.05, "float", 0.0, 1.0, None, "runtime"),
    "gguf_repeat_last_n": (64, "int", 0, 4096, None, "runtime"),
    "gguf_presence_penalty": (0.0, "float", -2.0, 2.0, None, "runtime"),
    "gguf_frequency_penalty": (0.0, "float", -2.0, 2.0, None, "runtime"),
    "subtitle_min_cue_s": (0.5, "float", 0.0, 5.0, None, "output"),
    "subtitle_max_line_chars": (0, "int", 0, 200, None, "output"),
    "summary_max_new_tokens": (1024, "int", 64, 8192, None, "output"),
    "caption_join_separator": (" ", "str", None, None, None, "postprocessing"),
    "adaptive_threshold": (2.0, "float", 0.1, 50.0, None, "preprocessing"),
    "reject_black_luma": (16, "int", 0, 255, None, "preprocessing"),
    "reject_silence_rms": (0.001, "float", 0.0, 0.1, None, "preprocessing"),
    "context_carry_prompt": (
        "Context from the previous segment (do not repeat it): {{CONTEXT}}",
        "str",
        None,
        None,
        None,
        "splitting",
    ),
    "plan_slack_mib": (512, "int", 0, 8192, None, "model"),
    "oom_degrade_factor": (0.75, "float", 0.5, 0.95, None, "runtime"),
    "gguf_fit_headroom_mib": (1536, "int", 0, 8192, None, "runtime"),
    "gguf_startup_timeout_s": (900, "int", 60, 3600, None, "runtime"),
    "gguf_stream_idle_timeout_s": (120, "int", 0, 3600, None, "runtime"),
    "chat_repetition_penalty": (1.0, "float", 0.5, 2.0, None, "chat"),
    "ffmpeg_path": ("", "str", None, None, None, "global"),
    "prompt_library_selection": ("", "str", None, None, None, "prompt_library"),
    "prompt_library_name": ("", "str", None, None, None, "prompt_library"),
}

EXPECTED_ELEM_IDS = {
    "no_repeat_ngram_size": "vc_no_repeat_ngram_size",
    "dedupe_repeated_sentences": "vc_dedupe_repeated_sentences",
    "gguf_min_p": "vc_gguf_min_p",
    "gguf_repeat_last_n": "vc_gguf_repeat_last_n",
    "gguf_presence_penalty": "vc_gguf_presence_penalty",
    "gguf_frequency_penalty": "vc_gguf_frequency_penalty",
    "subtitle_min_cue_s": "vc_subtitle_min_cue_s",
    "subtitle_max_line_chars": "vc_subtitle_max_line_chars",
    "summary_max_new_tokens": "vc_summary_max_new_tokens",
    "caption_join_separator": "vc_caption_join_separator",
    "adaptive_threshold": "vc_adaptive_threshold",
    "reject_black_luma": "vc_reject_black_luma",
    "reject_silence_rms": "vc_reject_silence_rms",
    "context_carry_prompt": "vc_context_carry_prompt",
    "plan_slack_mib": "vc_plan_slack_mib",
    "oom_degrade_factor": "vc_oom_degrade_factor",
    "gguf_fit_headroom_mib": "vc_gguf_fit_headroom_mib",
    "gguf_startup_timeout_s": "vc_gguf_startup_timeout_s",
    "gguf_stream_idle_timeout_s": "vc_gguf_stream_idle_timeout_s",
    "chat_repetition_penalty": "vc_chat_repetition_penalty",
    "ffmpeg_path": "vc_ffmpeg_path",
    "prompt_library_selection": "vc_my_prompts",
    "prompt_library_name": "vc_prompt_name",
}


def test_every_v14b_control_is_registered_exactly() -> None:
    demo = build_app()
    try:
        entries = {entry.key: entry for entry in demo.vcap_context.registry.entries()}
        assert entries["model_key"].default == "qwen3_omni_instruct_int4"
        for key, expected in EXPECTED.items():
            entry = entries[key]
            actual = (
                entry.default,
                entry.kind,
                entry.minimum,
                entry.maximum,
                entry.choices,
                entry.section,
            )
            assert actual == expected, key
            assert entry.description, key
        assert entries["chat_repetition_penalty"].in_metadata is False
        assert entries["ffmpeg_path"].in_preset is False
        assert entries["ffmpeg_path"].in_metadata is False
        assert entries["prompt_library_selection"].in_preset is False
        assert entries["prompt_library_selection"].in_metadata is False
        assert entries["prompt_library_name"].in_preset is False
        assert entries["prompt_library_name"].in_metadata is False
    finally:
        demo.vcap_context.pipeline.shutdown()


def test_gguf_disables_no_repeat_ngram_and_transformers_plan_controls() -> None:
    gguf = gguf_control_updates("qwen3_omni_instruct_gguf_q4")
    for key in ("no_repeat_ngram_size", "plan_slack_mib"):
        assert gguf[key]["interactive"] is False
        assert gguf[key]["info"].endswith("(not used by GGUF)")

    transformers = gguf_control_updates("qwen3_omni_instruct_int4")
    for key in ("no_repeat_ngram_size", "plan_slack_mib"):
        assert transformers[key]["interactive"] is True
        assert "not used by GGUF" not in transformers[key]["info"]


def test_caption_join_separator_is_limited_to_sixteen_characters() -> None:
    demo = build_app()
    try:
        component = next(
            item
            for item in demo.get_config_file()["components"]
            if item.get("props", {}).get("elem_id") == "vc_caption_join_separator"
        )
        assert component["props"]["max_length"] == 16
    finally:
        demo.vcap_context.pipeline.shutdown()


def test_every_v14b_setting_control_has_stable_id_and_info_text() -> None:
    demo = build_app()
    try:
        entries = {entry.key: entry for entry in demo.vcap_context.registry.entries()}
        config = {
            item["id"]: item.get("props", {})
            for item in demo.get_config_file()["components"]
        }
        for key, elem_id in EXPECTED_ELEM_IDS.items():
            props = config[entries[key].component._id]
            assert props.get("elem_id") == elem_id, key
            assert props.get("info"), key
    finally:
        demo.vcap_context.pipeline.shutdown()
