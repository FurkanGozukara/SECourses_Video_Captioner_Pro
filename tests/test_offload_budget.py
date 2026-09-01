from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from vcap.models import offload
from vcap.models.offload import (
    BudgetHint,
    CheckpointLayout,
    OffloadPlan,
    checkpoint_layout,
    estimate_activation_bytes,
    plan_block_swap,
)


GIB = 2**30


def _write_safetensors_header(path: Path, header: dict[str, object]) -> None:
    raw_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(len(raw_header).to_bytes(8, "little") + raw_header)


def test_checkpoint_layout_reads_only_the_safetensors_header(tmp_path: Path) -> None:
    checkpoint = tmp_path / "synthetic.safetensors"
    _write_safetensors_header(
        checkpoint,
        {
            "__metadata__": {"format": "pt"},
            "thinker.model.layers.0.weight": {
                "dtype": "F16",
                "shape": [2, 3],
                "data_offsets": [0, 12],
            },
            "thinker.model.layers.0.bias": {
                "dtype": "F32",
                "shape": [3],
                "data_offsets": [12, 24],
            },
            "thinker.model.layers.2.weight": {
                "dtype": "BF16",
                "shape": [2, 2],
                "data_offsets": [24, 32],
            },
            "thinker.model.layers.2.quant": {
                "dtype": "F8_E4M3",
                "shape": [5],
                "data_offsets": [32, 37],
            },
            "thinker.model.embed_tokens.weight": {
                "dtype": "I8",
                "shape": [5, 2],
                "data_offsets": [37, 47],
            },
            "thinker.visual.mask": {
                "dtype": "BOOL",
                "shape": [2],
                "data_offsets": [47, 49],
            },
            "thinker.lm_head.scale": {
                "dtype": "F64",
                "shape": [],
                "data_offsets": [49, 57],
            },
        },
    )

    layout = checkpoint_layout(checkpoint)

    assert layout.path == checkpoint
    assert layout.layer_count == 3
    assert layout.layer_bytes == (24, 0, 13)
    assert layout.non_layer_bytes == 20
    assert layout.total_bytes == 57


def test_activation_estimates_follow_both_family_formulas() -> None:
    qwen_config = SimpleNamespace(
        text_config=SimpleNamespace(
            num_hidden_layers=48,
            num_key_value_heads=4,
            head_dim=128,
            hidden_size=2_048,
            num_attention_heads=16,
            vocab_size=152_064,
        )
    )
    qwen_hint = BudgetHint(
        max_frames=32,
        max_pixels=128 * 32 * 32,
        fps=2.0,
        max_new_tokens=2_048,
        context_tokens=32_768,
        media_kinds=("video",),
    )
    qwen_estimate = estimate_activation_bytes(
        "qwen3_omni_instruct", qwen_config, qwen_hint
    )
    assert qwen_estimate == 1_588_609_024

    timechat_config = SimpleNamespace(
        num_hidden_layers=28,
        num_key_value_heads=4,
        hidden_size=3_584,
        num_attention_heads=28,
        vocab_size=152_064,
    )
    timechat_hint = BudgetHint(
        max_frames=32,
        max_pixels=128 * 28 * 28,
        fps=2.0,
        max_new_tokens=2_048,
        context_tokens=32_768,
        media_kinds=("video",),
    )
    timechat_estimate = estimate_activation_bytes("timechat", timechat_config, timechat_hint)
    assert timechat_estimate == 1_351_106_560
    # Same vision constant for both families now; Qwen3's larger KV cache and MoE
    # prefill buffers make it the bigger estimate at equal frame counts.
    assert GIB < timechat_estimate < qwen_estimate < 16 * GIB

    audio_only = estimate_activation_bytes(
        "qwen3_omni_captioner",
        SimpleNamespace(),
        BudgetHint(
            max_frames=2,
            max_pixels=128 * 32 * 32,
            fps=2.0,
            max_new_tokens=32,
            media_kinds=("audio",),
        ),
        observed_bytes=3 * GIB,
    )
    assert audio_only == 3 * GIB
    assert estimate_activation_bytes(
        "avocado", SimpleNamespace(), BudgetHint(media_kinds=("text",)), observed_bytes=20 * GIB
    ) == 16 * GIB


def _uniform_layout() -> CheckpointLayout:
    return CheckpointLayout(
        path=Path("synthetic.safetensors"),
        layer_count=8,
        layer_bytes=(GIB,) * 8,
        non_layer_bytes=2 * GIB,
        total_bytes=10 * GIB,
    )


def test_plan_block_swap_auto_and_resident_modes() -> None:
    layout = _uniform_layout()
    budget = plan_block_swap(
        OffloadPlan(),
        layout,
        free_vram_bytes=10 * GIB,
        total_vram_bytes=12 * GIB,
        activation_bytes=GIB,
    )

    # slack = 512 MiB + 0.5 x 1 GiB activation = 1 GiB; weights budget = 10 - 2 - 1 - 1 = 6 GiB
    # -> minus 2 GiB non-layer and 2 x 1 GiB slots leaves 2 resident layers.
    assert budget.mode == "block_swap"
    assert (budget.resident_layers, budget.swapped_layers, budget.slots) == (2, 6, 2)
    assert budget.resident_weight_bytes == 6 * GIB
    assert budget.expected_peak_bytes == 8 * GIB
    assert budget.pinned_bytes == 6 * GIB
    assert budget.allocator_slack_bytes == GIB
    assert budget.notes == (
        "Block swap: 2/8 decoder layers resident, 6 swapped (6.00 GiB pinned), "
        "2 slots x 1024 MiB; GPU weights 6.0 GiB; activation estimate 1.0 GiB; "
        "allocator slack 1.0 GiB; reserve 2.0 GiB; expected peak 8.0 of 10.0 GiB free",
    )
    assert budget.summary()["expected_peak_gib"] == 8.0
    assert budget.summary()["allocator_slack_gib"] == 1.0

    resident = plan_block_swap(
        OffloadPlan(),
        layout,
        free_vram_bytes=14 * GIB,
        total_vram_bytes=16 * GIB,
        activation_bytes=GIB,
    )
    assert resident.mode == "resident"
    assert (resident.resident_layers, resident.swapped_layers, resident.slots) == (8, 0, 0)


def test_plan_block_swap_forced_integer_legacy_and_tiny_vram_notes() -> None:
    layout = _uniform_layout()
    forced = plan_block_swap(
        OffloadPlan("all"),
        layout,
        free_vram_bytes=10 * GIB,
        total_vram_bytes=12 * GIB,
        activation_bytes=GIB,
    )
    assert forced.mode == "forced_resident"
    assert (forced.resident_layers, forced.swapped_layers, forced.slots) == (8, 0, 0)
    assert forced.notes[1] == (
        "expected peak 12.0 GiB exceeds free VRAM minus reserve by 4.0 GiB; "
        "Windows will page into shared memory \u2014 use auto or a smaller layer count"
    )

    explicit = plan_block_swap(
        OffloadPlan(7),
        layout,
        free_vram_bytes=10 * GIB,
        total_vram_bytes=12 * GIB,
        activation_bytes=GIB,
    )
    assert explicit.mode == "block_swap"
    assert (explicit.resident_layers, explicit.swapped_layers, explicit.slots) == (7, 1, 1)
    assert "respecting the requested resident layer count" in explicit.notes[1]

    legacy = plan_block_swap(
        OffloadPlan("auto", True),
        layout,
        free_vram_bytes=10 * GIB,
        total_vram_bytes=12 * GIB,
        activation_bytes=GIB,
    )
    assert legacy.mode == "legacy"
    assert (legacy.resident_layers, legacy.swapped_layers, legacy.slots) == (8, 0, 0)
    assert "Legacy Accelerate offload" in legacy.notes[1]
    assert "block swap is disabled" in legacy.notes[1]

    tiny = plan_block_swap(
        OffloadPlan(),
        layout,
        free_vram_bytes=4 * GIB,
        total_vram_bytes=8 * GIB,
        activation_bytes=GIB,
        ram_available_bytes=5 * GIB,
    )
    assert tiny.mode == "block_swap"
    assert (tiny.resident_layers, tiny.swapped_layers, tiny.slots) == (0, 8, 2)
    assert "Even with zero resident decoder layers" in tiny.notes[1]
    assert "some layers will stay pageable" in tiny.notes[2]


def test_plan_block_swap_stages_towers_when_it_buys_resident_layers() -> None:
    layout = CheckpointLayout(
        path=Path("synthetic.safetensors"),
        layer_count=8,
        layer_bytes=(GIB,) * 8,
        non_layer_bytes=2 * GIB,
        total_bytes=10 * GIB,
        tower_bytes=int(1.5 * GIB),
    )
    # slack = 0.5 + 0.5 x 2 = 1.5 GiB.
    # Plain: weights budget 10 - 2 - 2 - 1.5 = 4.5 -> (4.5 - 2 - 2) = 0 resident layers.
    # Staged: dense 0.5 GiB, prefill phase max(2, 1.5 + 1) = 2.5 -> budget 4.0 ->
    # (4.0 - 0.5 - 2) = 1 resident layer, so staging wins.
    budget = plan_block_swap(
        OffloadPlan(),
        layout,
        free_vram_bytes=10 * GIB,
        total_vram_bytes=12 * GIB,
        activation_bytes=2 * GIB,
    )
    assert budget.mode == "block_swap"
    assert budget.stage_towers is True
    assert (budget.resident_layers, budget.swapped_layers, budget.slots) == (1, 7, 2)
    assert budget.resident_weight_bytes == int(0.5 * GIB) + 3 * GIB
    assert budget.expected_peak_bytes == int(0.5 * GIB) + 3 * GIB + int(2.5 * GIB) + int(1.5 * GIB)
    assert "towers staged on CPU between prefills (1.5 GiB)" in budget.notes[0]
    assert budget.summary()["stage_towers"] is True

    # Without towers in the layout nothing changes.
    plain = plan_block_swap(
        OffloadPlan(),
        _uniform_layout(),
        free_vram_bytes=10 * GIB,
        total_vram_bytes=12 * GIB,
        activation_bytes=2 * GIB,
    )
    assert plain.stage_towers is False
    assert plain.resident_layers == 0

    # When both layouts leave zero resident layers, staging still wins on peak alone.
    tiny = plan_block_swap(
        OffloadPlan(),
        layout,
        free_vram_bytes=5 * GIB,
        total_vram_bytes=8 * GIB,
        activation_bytes=2 * GIB,
    )
    assert (tiny.resident_layers, tiny.stage_towers) == (0, True)
    assert tiny.expected_peak_bytes == int(0.5 * GIB) + 2 * GIB + int(2.5 * GIB) + int(1.5 * GIB)


def test_observed_activation_ratio_scales_fresh_estimates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(offload, "TEMP_DIR", tmp_path)
    assert offload.observed_activation_ratio("variant") == 1.0
    offload.record_observed_activation_bytes("variant", 3 * GIB, planned_bytes=GIB)
    assert offload.observed_activation_ratio("variant") == 3.0
    offload.record_observed_activation_bytes("variant", GIB, planned_bytes=GIB)
    # blends towards the latest observation but never below the fresh value
    assert offload.observed_activation_ratio("variant") == 2.0
    offload.record_observed_activation_bytes("variant", 50 * GIB, planned_bytes=GIB)
    assert offload.observed_activation_ratio("variant") == offload.OBSERVED_RATIO_MAX

    hint = BudgetHint(max_frames=40, max_pixels=256 * 32 * 32, fps=2.0, max_new_tokens=256, media_kinds=("video",))
    base = estimate_activation_bytes("qwen3_omni_instruct", None, hint)
    doubled = estimate_activation_bytes("qwen3_omni_instruct", None, hint, observed_ratio=2.0)
    assert doubled == 2 * base
    assert estimate_activation_bytes("qwen3_omni_instruct", None, hint, observed_ratio=0.1) == int(0.75 * base) or (
        abs(estimate_activation_bytes("qwen3_omni_instruct", None, hint, observed_ratio=0.1) - 0.75 * base) <= 1
    )


def test_observed_activation_cache_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(offload, "TEMP_DIR", tmp_path)

    assert offload.observed_activation_bytes("variant") == 0
    offload.record_observed_activation_bytes("variant", 123)
    offload.record_observed_activation_bytes("variant", 100)
    assert offload.observed_activation_bytes("variant") == 123
    offload.record_observed_activation_bytes("variant", 456)
    assert offload.observed_activation_bytes("variant") == 456

    cache_path = tmp_path / "vram_activation_cache.json"
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert payload["variant"]["bytes"] == 456
    assert math.isfinite(payload["variant"]["timestamp"])

    cache_path.write_text("not JSON", encoding="utf-8")
    assert offload.observed_activation_bytes("variant") == 0
    offload.record_observed_activation_bytes("variant", 789)
    assert offload.observed_activation_bytes("variant") == 789
