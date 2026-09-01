from __future__ import annotations

from pathlib import Path

from vcap.models.base import GenParams
from vcap.models.llamacpp_backend import server_plan_for_vram
from vcap.models.omni_common import OmniCaptionerBase
from vcap.models.registry import MODEL_SPECS
from vcap.pipeline.job import InputItem, JobSpec, OutputSpec
from vcap.pipeline.runner import _context_limit, _model_gen
from vcap.ui.components import context_usage_text
from vcap.ui.tabs.chat_tab import _chatbot_messages, _tokens_line


def _spec(tmp_path: Path, **settings: object) -> JobSpec:
    return JobSpec.from_settings(
        {"model_key": "qwen3_omni_instruct_int4", **settings},
        [InputItem(path="", kind="text", text_prompt_only=True, text="hi")],
        OutputSpec(kind="single", outputs_root=str(tmp_path)),
    )


def test_requested_context_flows_from_settings_to_backends(tmp_path: Path) -> None:
    spec = _spec(tmp_path, context_tokens=12000)
    assert spec.generation.context_tokens == 12000
    assert _model_gen(spec).context_tokens == 12000
    assert _context_limit(spec, MODEL_SPECS["qwen3_omni_instruct"]) == 12000
    # Blank or oversized requests fall back to the family cap.
    assert _spec(tmp_path).generation.context_tokens is None
    assert _context_limit(_spec(tmp_path), MODEL_SPECS["qwen3_omni_instruct"]) == 32768
    assert _context_limit(_spec(tmp_path, context_tokens=99_999), MODEL_SPECS["timechat"]) == 32768
    assert _spec(tmp_path, context_tokens="").generation.context_tokens is None
    backend = OmniCaptionerBase("qwen3_omni_instruct")
    assert backend._context_limit(GenParams(context_tokens=12000)) == 12000
    assert backend._context_limit(GenParams(context_tokens=99_999)) == 32768
    assert backend._context_limit(GenParams()) == 32768


def test_job_clamps_frames_to_the_family_cap(tmp_path: Path) -> None:
    def frames(model_key: str, value: object) -> int:
        spec = JobSpec.from_settings(
            {"model_key": model_key, "max_frames": value},
            [InputItem(path="", kind="text", text_prompt_only=True, text="hi")],
            OutputSpec(kind="single", outputs_root=str(tmp_path)),
        )
        return spec.preprocess.max_frames

    # A Qwen3 value left behind when switching to a 7B family is clamped, not rejected.
    assert frames("timechat_int8", 768) == 160
    assert frames("avocado_int4", 768) == 256
    assert frames("qwen3_omni_instruct_int4", 768) == 768
    assert frames("qwen3_omni_instruct_int4", 240) == 240
    assert frames("timechat_int8", 0) == 160


def test_llamacpp_reserves_the_requested_window_within_the_tier() -> None:
    assert server_plan_for_vram(32.0, requested_context=12000).context_size == 12000
    assert server_plan_for_vram(32.0, requested_context=32768).context_size == 32768
    assert server_plan_for_vram(24.0, requested_context=32768).context_size == 16384
    assert server_plan_for_vram(16.0, requested_context=24576).context_size == 8192


def test_kv_cache_estimates_follow_each_family() -> None:
    qwen3 = MODEL_SPECS["qwen3_omni_instruct"].limits
    seven_b = MODEL_SPECS["timechat"].limits
    assert qwen3.kv_bytes_per_token == 48 * 4 * 128 * 4
    assert seven_b.kv_bytes_per_token == 28 * 4 * 128 * 4
    assert abs(qwen3.kv_cache_gb(32768) - 3.0) < 1e-6
    assert abs(qwen3.kv_cache_gb(16384) - 1.5) < 1e-6
    assert abs(seven_b.kv_cache_gb(32768) - 1.75) < 1e-6
    assert MODEL_SPECS["qwen3_omni_thinking"].limits.kv_bytes_per_token == qwen3.kv_bytes_per_token
    assert MODEL_SPECS["qwen3_omni_captioner"].limits.kv_bytes_per_token == qwen3.kv_bytes_per_token
    assert MODEL_SPECS["avocado"].limits.kv_bytes_per_token == seven_b.kv_bytes_per_token


def test_context_statistics_formatting() -> None:
    assert context_usage_text(None, 32768) == "—"
    assert context_usage_text(0, 32768) == "—"
    assert context_usage_text(4096, 32768) == "4,096 / 32,768 (12%)"
    assert context_usage_text(4096, 0) == "4,096"
    assert context_usage_text("bad", "worse") == "—"
    assert _tokens_line() == "**Tokens:** — · **Speed:** — · **Context:** —"
    assert _tokens_line(12, "3.00 tok/s", 1500, 16384).endswith("**Context:** 1,500 / 16,384 (9%)")
    shown = _chatbot_messages(
        [
            {"role": "user", "content": "look", "media": ["C:/clips/a.mp4", "/tmp/b.png"]},
            {"role": "assistant", "content": "ok"},
            {"role": "system", "content": "hidden"},
        ]
    )
    assert shown == [
        {"role": "user", "content": "look\n\n📎 a.mp4 · b.png"},
        {"role": "assistant", "content": "ok"},
    ]
