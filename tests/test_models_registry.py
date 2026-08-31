from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path

import pytest

from vcap.models import attention
from vcap.models.registry import (
    MODEL_SPECS,
    all_variant_choices,
    resolve_model_dir,
    variant_is_ready,
    variant_to_family,
)
from vcap.models.vram_presets import (
    VRAM_TIERS,
    VramPreset,
    allowed_variants,
    apply_preset,
    auto_tier,
    preset_for,
)


def test_registry_integrity_and_unique_variant_keys() -> None:
    assert set(MODEL_SPECS) == {
        "timechat",
        "avocado",
        "qwen3_omni_instruct",
        "qwen3_omni_thinking",
        "qwen3_omni_captioner",
    }
    keys: list[str] = []
    for family, spec in MODEL_SPECS.items():
        assert spec.family == family
        assert spec.variants
        assert spec.capabilities
        assert spec.default_prompt_preset
        assert all(param.description for param in spec.param_schema)
        assert all(
            param.min is None or param.max is None or param.min <= param.default <= param.max
            for param in spec.param_schema
            if isinstance(param.default, (int, float))
        )
        for variant in spec.variants:
            keys.append(variant.key)
            assert variant_to_family(variant.key) == family
            assert resolve_model_dir(variant.key).name == variant.folder_name
            assert variant.size_gb > 0
    assert len(keys) == len(set(keys))


def test_variant_choices_are_complete_and_stable() -> None:
    choices = all_variant_choices()
    registered = [variant.key for spec in MODEL_SPECS.values() for variant in spec.variants]
    assert [key for _, key in choices] == registered
    assert len({key for _, key in choices}) == len(choices)
    assert all(label.endswith("GB)") for label, _ in choices)
    gguf_ready, gguf_detail = variant_is_ready("qwen3_omni_instruct_gguf_q4")
    assert isinstance(gguf_ready, bool)
    assert gguf_detail


def test_local_model_info_source_repositories() -> None:
    models_root = Path(__file__).parents[1] / "models"
    expected = {
        "timechat": "yaolily/TimeChat-Captioner-GRPO-7B",
        "avocado": "AVoCaDO-Captioner/AVoCaDO",
        "qwen3_omni_instruct": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "qwen3_omni_thinking": "Qwen/Qwen3-Omni-30B-A3B-Thinking",
        "qwen3_omni_captioner": "Qwen/Qwen3-Omni-30B-A3B-Captioner",
    }
    for folder in models_root.iterdir():
        info_path = folder / "vcap_model_info.json"
        if not info_path.is_file():
            continue
        prefix = next(key for key in expected if folder.name.startswith(key))
        data = json.loads(info_path.read_text(encoding="utf-8"))
        assert data["source_repo"] == expected[prefix]


def test_limits_and_qwen_video_budget_are_monotonic() -> None:
    assert MODEL_SPECS["timechat"].limits.compute_max_duration() == 60
    assert MODEL_SPECS["avocado"].limits.compute_max_duration() == 100
    limits = MODEL_SPECS["qwen3_omni_instruct"].limits
    base = limits.compute_max_duration(1.0, 128 * 32 * 32)
    faster = limits.compute_max_duration(2.0, 128 * 32 * 32)
    larger = limits.compute_max_duration(1.0, 256 * 32 * 32)
    assert base > faster
    assert base > larger
    assert limits.min_pixels < limits.default_max_pixels
    assert limits.context_tokens >= limits.max_new_tokens_cap
    assert limits.compute_max_duration(include_audio=False) > limits.compute_max_duration()
    avocado_params = {
        param.name: param.default for param in MODEL_SPECS["avocado"].param_schema
    }
    assert avocado_params["max_new_tokens"] == 2_048


def test_vram_tiers_presets_and_allowed_variants_are_monotonic() -> None:
    assert auto_tier(5.0) == 6
    assert auto_tier(23.6) == 24
    assert auto_tier(90) == 80
    for family in MODEL_SPECS:
        previous: set[str] = set()
        for tier in VRAM_TIERS:
            current = set(allowed_variants(family, tier))
            assert previous <= current
            previous = current
    assert allowed_variants("qwen3_omni_instruct", 6) == []
    assert preset_for("timechat", 24).variant_scheme == "bf16"
    assert preset_for("qwen3_omni_instruct", 24).variant_scheme == "int4_convrot_w4a8"
    assert preset_for("qwen3_omni_instruct", 48).variant_scheme == "int8_convrot"
    with pytest.raises(ValueError, match="not offered"):
        preset_for("qwen3_omni_instruct", 6)


def test_vram_preset_fields_have_descriptions_and_apply_without_mutation() -> None:
    assert all(item.metadata.get("description") for item in fields(VramPreset))
    original = {"keep": 1, "fps": 99}
    preset = preset_for("avocado", 16)
    applied = apply_preset(original, preset)
    assert original == {"keep": 1, "fps": 99}
    assert applied["keep"] == 1
    assert applied["fps"] == preset.fps
    assert applied["offload"] == preset.offload


def test_attention_resolution_and_flash_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        attention,
        "probe_available",
        lambda: {
            "auto": True,
            "flash_attention_2": False,
            "sdpa": True,
            "sage": False,
            "xformers": False,
            "eager": True,
        },
    )
    implementation, context = attention.resolve("auto", "timechat")
    assert implementation == "sdpa"
    with context:
        pass
    assert attention.resolve("flash_attention_2", "timechat")[0] == "sdpa"
    assert attention.resolve("eager", "avocado")[0] == "eager"
    with pytest.raises(ValueError, match="Unknown attention"):
        attention.resolve("made_up", "timechat")


def test_loader_gpu_selection_has_only_opt_in_dev_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    from vcap.models.loader import (
        _enforce_dev_gpu_guard,
        _normalize_device,
        _resolve_gpu_selection,
        _selected_physical_gpu,
    )

    monkeypatch.delenv("VCAP_DEV_FORCE_GPU", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("VCAP_WORKER_GPU", "3")
    assert _normalize_device("cuda:2") == ("cuda:2", 2)
    assert _selected_physical_gpu(None, 0) == 3
    assert _selected_physical_gpu(2, 0) == 2
    assert _resolve_gpu_selection("cuda:0", 3) == ("cuda:0", 0, 3)
    _enforce_dev_gpu_guard(2)

    monkeypatch.delenv("VCAP_WORKER_GPU", raising=False)
    assert _resolve_gpu_selection("cuda:0", 2) == ("cuda:2", 2, 2)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,2")
    assert _selected_physical_gpu(None, 1) == 2
    assert _resolve_gpu_selection("cuda:0", 2) == ("cuda:1", 1, 2)
    with pytest.raises(RuntimeError, match="not present"):
        _resolve_gpu_selection("cuda:0", 3)

    monkeypatch.setenv("VCAP_DEV_FORCE_GPU", "1")
    _enforce_dev_gpu_guard(1)
    with pytest.raises(RuntimeError, match="rejects selected physical GPU 2"):
        _enforce_dev_gpu_guard(2)


def test_offload_map_targets_selected_process_device() -> None:
    from vcap.models.offload import OffloadPlan, build_device_map

    placement = build_device_map(
        "avocado",
        OffloadPlan(),
        20.0,
        device_index=2,
        physical_gpu_index=2,
    )
    assert placement.device_map == {"": 2}
    assert set(placement.explicit_device_map.values()) == {2}
