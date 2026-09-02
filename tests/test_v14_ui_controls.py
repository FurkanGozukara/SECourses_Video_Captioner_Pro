from __future__ import annotations

import gradio as gr

from vcap.ui.app import build_app
from vcap.ui.tabs.caption_tab import (
    DEFAULT_SUMMARY_PROMPT,
    gguf_control_updates,
    render_prompt_preserving_edits,
)


EXPECTED = {
    "seed": (-1, "int", -1, 2147483647, None),
    "chat_seed": (-1, "int", -1, 2147483647, None),
    "max_caption_chars": (0, "int", 0, 100000, None),
    "context_carry_words": (60, "int", 10, 400, None),
    "fade_threshold": (12.0, "float", 1, 100, None),
    "encode_codec": ("libx264", "str", None, None, ("libx264", "h264_nvenc", "libx265", "hevc_nvenc")),
    "encode_crf": (18, "int", 0, 51, None),
    "encode_preset": ("veryfast", "str", None, None, ("ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower")),
    "encode_audio_bitrate": ("192k", "str", None, None, ("96k", "128k", "192k", "256k", "320k")),
    "quality_frames": (8, "int", 4, 32, None),
    "total_pixel_cap": (0, "int", 0, 400000000, None),
    "oom_retries": (2, "int", 0, 4, None),
    "pinned_ram_budget_gb": (0.0, "float", 0, 1024, None),
    "swap_slots": (2, "int", None, None, (1, 2, 3, 4)),
    "gguf_max_frames": (32, "int", 1, 128, None),
    "gguf_jpeg_quality": (90, "int", 50, 100, None),
    "gguf_threads": (0, "int", 0, 256, None),
    "gguf_batch_size": (2048, "int", 64, 8192, None),
    "gguf_ubatch_size": (512, "int", 32, 4096, None),
    "gguf_flash_attn": ("auto", "str", None, None, ("auto", "on", "off")),
    "gguf_cache_reuse": (0, "int", 0, 4096, None),
    "gguf_ignore_tier_context": (False, "bool", None, None, None),
    "gguf_extra_args": ("", "str", None, None, None),
    "summarize_segments": (False, "bool", None, None, None),
    "summary_prompt": (DEFAULT_SUMMARY_PROMPT, "str", None, None, None),
    "batch_include_kinds": (["video", "audio", "image", "text"], "list", None, None, ("video", "audio", "image", "text")),
    "batch_name_filter": ("", "str", None, None, None),
}


def test_every_v14_contract_control_is_registered_exactly() -> None:
    demo = build_app()
    try:
        entries = {entry.key: entry for entry in demo.vcap_context.registry.entries()}
        for key, expected in EXPECTED.items():
            entry = entries[key]
            actual = (entry.default, entry.kind, entry.minimum, entry.maximum, entry.choices)
            assert actual == expected, key
        logs = entries["logs_dir"]
        assert logs.kind == "str"
        assert logs.section == "global"
        assert logs.in_preset is False and logs.in_metadata is False
        assert entries["chat_seed"].section == "chat"
        assert entries["chat_seed"].in_metadata is False
    finally:
        demo.vcap_context.pipeline.shutdown()


def test_gguf_backend_disables_transformers_only_controls() -> None:
    updates = gguf_control_updates("qwen3_omni_instruct_gguf_q4", False)
    assert updates["gguf_options"]["visible"] is True
    assert updates["gguf_option"]["interactive"] is True
    disabled = {
        "attention_backend",
        "block_swap_auto",
        "blocks_to_swap",
        "swap_slots",
        "offload_experts",
        "pin_cpu",
        "pinned_ram_budget_gb",
        "torch_compile",
        "torch_compile_mode",
        "use_cache",
    }
    for key in disabled:
        assert updates[key]["interactive"] is False
        assert updates[key]["info"].endswith("(not used by GGUF)")

    transformer = gguf_control_updates("qwen3_omni_instruct_int4", False)
    assert transformer["gguf_options"]["visible"] is False
    assert transformer["gguf_option"]["interactive"] is False
    assert all(transformer[key]["interactive"] is True for key in disabled)
    assert all("not used by GGUF" not in transformer[key]["info"] for key in disabled)


def test_prompt_variable_render_preserves_manual_edits() -> None:
    variables = {
        "TRIGGER": "ohwx",
        "LANGUAGE": "English",
        "SOURCE_LANGUAGE": "English",
        "TARGET_LANGUAGE": "English",
        "CAPTION_LENGTH": "detailed",
        "AVOID": "",
        "SUBJECT_CLASS": "person",
        "EXTRA_INSTRUCTIONS": "",
    }
    _, auto_system, auto_user, tracked = render_prompt_preserving_edits(
        "wan22_t2v_dense", variables, "", "", {"system": "", "user": ""}
    )
    description, kept_system, kept_user, _ = render_prompt_preserving_edits(
        "wan22_t2v_dense",
        {**variables, "LANGUAGE": "Turkish"},
        "my manual system",
        str(auto_user),
        tracked,
    )
    assert kept_system == gr.skip()
    assert kept_user != gr.skip()
    assert "Prompt edited manually — Reset prompts to preset re-renders it." in description
