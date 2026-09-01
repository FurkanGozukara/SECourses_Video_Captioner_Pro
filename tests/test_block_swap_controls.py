"""Block-swap controls: translation helpers, legacy migration, job parsing, and the UI preview."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from vcap.core.registry import SettingsRegistry
from vcap.models import offload
from vcap.models.offload import BudgetHint, CheckpointLayout, OffloadPlan
from vcap.pipeline.job import JobSpec, OutputSpec
from vcap.ui.tabs import caption_tab


GIB = 2**30


def _write_safetensors_header(path: Path, header: dict[str, object]) -> None:
    raw_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(len(raw_header).to_bytes(8, "little") + raw_header)


def _write_synthetic_model_folder(folder: Path, *, layers: int, layer_mib: int) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    header: dict[str, object] = {"__metadata__": {"format": "pt"}}
    per_layer = layer_mib * 2**20
    for index in range(layers):
        header[f"thinker.model.layers.{index}.weight"] = {
            "dtype": "U8",
            "shape": [per_layer],
            "data_offsets": [0, per_layer],
        }
    header["thinker.model.embed_tokens.weight"] = {"dtype": "U8", "shape": [GIB], "data_offsets": [0, GIB]}
    header["thinker.visual.blocks"] = {"dtype": "U8", "shape": [GIB // 2], "data_offsets": [0, GIB // 2]}
    _write_safetensors_header(folder / "model.safetensors", header)
    (folder / "config.json").write_text(
        json.dumps(
            {
                "thinker_config": {
                    "model_type": "qwen3_omni_moe_thinker",
                    "text_config": {
                        "num_hidden_layers": layers,
                        "num_key_value_heads": 4,
                        "hidden_size": 2048,
                        "num_attention_heads": 16,
                        "head_dim": 128,
                        "vocab_size": 152_064,
                    },
                }
            }
        ),
        encoding="utf-8",
    )


def test_block_swap_control_translation_round_trips() -> None:
    assert offload.family_layer_count("qwen3_omni_instruct") == 48
    assert offload.family_layer_count("timechat") == 28
    with pytest.raises(KeyError):
        offload.family_layer_count("unknown")

    assert offload.block_swap_to_gpu_layers(True, 17, 48) == "auto"
    assert offload.block_swap_to_gpu_layers(False, 0, 48) == 48
    assert offload.block_swap_to_gpu_layers(False, 11, 48) == 37
    assert offload.block_swap_to_gpu_layers(False, 99, 28) == 0
    assert offload.block_swap_to_gpu_layers(False, "bad", 28) == 28

    assert offload.gpu_layers_to_block_swap("auto", 48) == (True, 0)
    assert offload.gpu_layers_to_block_swap(None, 48) == (True, 0)
    assert offload.gpu_layers_to_block_swap("all", 48) == (False, 0)
    assert offload.gpu_layers_to_block_swap("37", 48) == (False, 11)
    assert offload.gpu_layers_to_block_swap(12, 28) == (False, 16)
    assert offload.gpu_layers_to_block_swap(500, 28) == (False, 0)
    assert offload.gpu_layers_to_block_swap("garbage", 48) == (True, 0)

    for auto, swapped in ((True, 0), (False, 0), (False, 11), (False, 48)):
        assert offload.gpu_layers_to_block_swap(
            offload.block_swap_to_gpu_layers(auto, swapped, 48), 48
        ) == (auto, swapped)


def test_migrate_legacy_gpu_layers_uses_family_layer_count_and_keeps_new_keys() -> None:
    legacy = {"model_key": "timechat_int4", "gpu_layers": "20", "fps": 2.0}
    migrated = offload.migrate_legacy_gpu_layers(legacy)
    assert legacy == {"model_key": "timechat_int4", "gpu_layers": "20", "fps": 2.0}
    assert migrated == {
        "model_key": "timechat_int4",
        "fps": 2.0,
        "block_swap_auto": False,
        "blocks_to_swap": 8,
    }
    assert offload.migrate_legacy_gpu_layers({"gpu_layers": "auto"}) == {
        "block_swap_auto": True,
        "blocks_to_swap": 0,
    }
    # Unknown variants fall back to the 48-layer families.
    assert offload.migrate_legacy_gpu_layers({"model_key": "nope", "gpu_layers": 40}) == {
        "model_key": "nope",
        "block_swap_auto": False,
        "blocks_to_swap": 8,
    }
    # Explicit new keys win; the legacy key is simply dropped.
    assert offload.migrate_legacy_gpu_layers(
        {"gpu_layers": "all", "block_swap_auto": True, "blocks_to_swap": 3}
    ) == {"block_swap_auto": True, "blocks_to_swap": 3}
    assert offload.migrate_legacy_gpu_layers({"fps": 1.0}) == {"fps": 1.0}


def test_registry_migrations_rewrite_legacy_keys_before_coercion() -> None:
    registry = SettingsRegistry()
    registry.register("auto", object(), True, kind="bool")
    registry.register("count", object(), 0, kind="int", minimum=0, maximum=48)

    def migrate(settings: dict[str, Any]) -> dict[str, Any]:
        legacy = settings.pop("legacy", None)
        if legacy is not None and "count" not in settings:
            settings["auto"] = False
            settings["count"] = int(legacy)
        return settings

    registry.add_migration(migrate)
    source = {"legacy": "7"}
    values, warnings = registry.coerce(source)
    assert values == {"auto": False, "count": 7}
    assert warnings == []
    assert source == {"legacy": "7"}  # the caller's mapping is untouched

    values, warnings = registry.coerce({"legacy": "7", "count": 3})
    assert values == {"auto": True, "count": 3}
    assert warnings == []
    with pytest.raises(TypeError):
        registry.add_migration("not callable")  # type: ignore[arg-type]


def test_plan_model_folder_previews_the_loader_plan_without_torch(tmp_path: Path, monkeypatch) -> None:
    folder = tmp_path / "qwen3_synthetic"
    _write_synthetic_model_folder(folder, layers=48, layer_mib=256)
    monkeypatch.setattr(offload, "observed_activation_ratio", lambda key: 1.0)

    config = offload.planning_config(folder)
    assert config["model_type"] == "qwen3_omni_moe_thinker"
    assert config["text_config"]["num_hidden_layers"] == 48

    hint = BudgetHint(max_frames=32, max_pixels=128 * 32 * 32, fps=1.0, max_new_tokens=512, context_tokens=8_192)
    budget = offload.plan_model_folder(
        "qwen3_omni_instruct",
        "qwen3_omni_instruct_int8",
        folder,
        OffloadPlan("auto", vram_reserve_gb=2.0, swap_slots=2),
        hint,
        free_vram_bytes=12 * GIB,
        total_vram_bytes=16 * GIB,
        ram_available_bytes=64 * GIB,
    )
    assert budget.mode == "block_swap"
    assert budget.layer_count == 48
    assert 0 < budget.resident_layers < 48
    assert budget.resident_layers + budget.swapped_layers == 48
    assert budget.expected_peak_bytes <= 12 * GIB - 2 * GIB

    manual = offload.plan_model_folder(
        "qwen3_omni_instruct",
        "qwen3_omni_instruct_int8",
        folder,
        OffloadPlan(48 - 5, vram_reserve_gb=2.0, swap_slots=2),
        hint,
        free_vram_bytes=12 * GIB,
        total_vram_bytes=16 * GIB,
    )
    assert (manual.resident_layers, manual.swapped_layers) == (43, 5)

    # The header is parsed once per (size, mtime) signature.
    calls: list[Path] = []
    original = offload.checkpoint_layout

    def counting(path: Path, **kwargs: object) -> CheckpointLayout:
        calls.append(Path(path))
        return original(path, **kwargs)

    monkeypatch.setattr(offload, "checkpoint_layout", counting)
    offload._LAYOUT_CACHE.clear()
    offload.cached_checkpoint_layout(folder / "model.safetensors")
    offload.cached_checkpoint_layout(folder / "model.safetensors")
    assert len(calls) == 1


def _job_settings(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "model_key": "qwen3_omni_instruct_int8",
        "user_prompt": "Describe the clip.",
        "max_new_tokens": 64,
        "fps": 1.0,
        "max_frames": 24,
        "max_pixels": 131_072,
        "output_formats": ["txt"],
    }
    values.update(overrides)
    return values


@pytest.mark.parametrize(
    ("settings", "expected_layers"),
    [
        ({"block_swap_auto": True, "blocks_to_swap": 30}, "auto"),
        ({"block_swap_auto": False, "blocks_to_swap": 0}, 48),
        ({"block_swap_auto": False, "blocks_to_swap": 11}, 37),
        ({"block_swap_auto": "false", "blocks_to_swap": "60"}, 0),
        ({"model_key": "timechat_int4", "block_swap_auto": False, "blocks_to_swap": 8}, 20),
        # An explicit legacy value still wins over the UI controls.
        ({"gpu_layers": "all", "block_swap_auto": False, "blocks_to_swap": 11}, "all"),
        ({}, "auto"),
    ],
)
def test_job_offload_parsing_translates_block_swap_controls(
    settings: dict[str, Any], expected_layers: int | str, tmp_path: Path
) -> None:
    spec = JobSpec.from_settings(_job_settings(**settings), [], OutputSpec(outputs_root=tmp_path))
    assert spec.model.offload.gpu_layers == expected_layers
    assert JobSpec.from_json(spec.to_json()).model.offload == spec.model.offload


def test_block_swap_preview_shows_auto_count_and_manual_override(tmp_path: Path, monkeypatch) -> None:
    folder = tmp_path / "qwen3_omni_instruct_int8"
    _write_synthetic_model_folder(folder, layers=48, layer_mib=256)
    monkeypatch.setattr(caption_tab, "variant_is_ready", lambda key: (True, "ready"))
    monkeypatch.setattr(caption_tab, "resolve_model_dir", lambda key: folder)
    monkeypatch.setattr(
        caption_tab,
        "resource_snapshot",
        lambda index: {"vram_free_gb": 12.0, "vram_total_gb": 16.0, "ram_free_gb": 64.0},
    )
    monkeypatch.setattr(offload, "observed_activation_ratio", lambda key: 1.0)
    common: dict[str, Any] = dict(
        gpu_index=0,
        reserve_gb=2.0,
        swap_slots=2,
        offload_experts=False,
        pin_cpu=True,
        fps_value=1.0,
        frames_value=32,
        pixels_value=128 * 32 * 32,
        output_tokens=512,
        context_value=8_192,
        duration=0.0,
        modality="video_audio",
    )

    slider, note = caption_tab.block_swap_preview("qwen3_omni_instruct_int8", True, 0, **common)
    assert slider["interactive"] is False
    assert 0 < slider["value"] < 48
    assert f"{slider['value']} block-swapped" in note
    assert "Automatic plan" in note
    assert "48 layers" in slider["info"]

    # A resident model reports the free VRAM it saw before loading; that figure is the basis.
    pong = {
        "ev": "pong",
        "loaded_variant": "qwen3_omni_instruct_int8",
        "block_swap": {"free_vram_gib": 14.0, "resident_layers": 40, "layer_count": 48, "swapped_layers": 8},
    }
    loaded_slider, loaded_note = caption_tab.block_swap_preview(
        "qwen3_omni_instruct_int8", True, 0, pong=pong, **common
    )
    assert loaded_slider["value"] < slider["value"]
    assert "before the resident model was placed" in loaded_note
    assert "Loaded now: 40/48 resident, 8 swapped" in loaded_note

    manual_slider, manual_note = caption_tab.block_swap_preview("qwen3_omni_instruct_int8", False, 20, **common)
    assert manual_slider == {"value": 20, "interactive": True, "info": slider["info"]}
    assert "Manual plan" in manual_note
    assert "28 of 48 decoder layers on GPU" in manual_note
    assert "20 block-swapped" in manual_note

    # A whole-decoder override that does not fit warns about paging instead of hiding it.
    _, forced_note = caption_tab.block_swap_preview("qwen3_omni_instruct_int8", False, 0, **common)
    assert "48 of 48 decoder layers on GPU" in forced_note
    assert "vc-warn" in forced_note

    # Slider values above the family's layer count clamp to it.
    clamped_slider, _ = caption_tab.block_swap_preview("timechat_int4", False, 40, **common)
    assert clamped_slider["value"] == 28

    gguf_slider, gguf_note = caption_tab.block_swap_preview("qwen3_omni_instruct_gguf_q8", False, 5, **common)
    assert gguf_slider["interactive"] is False
    assert "llama-server" in gguf_note

    experts_slider, experts_note = caption_tab.block_swap_preview(
        "qwen3_omni_instruct_int8", False, 5, **{**common, "offload_experts": True}
    )
    assert experts_slider["interactive"] is False
    assert "Legacy Accelerate" in experts_note

    monkeypatch.setattr(caption_tab, "variant_is_ready", lambda key: (False, "missing model.safetensors"))
    missing_slider, missing_note = caption_tab.block_swap_preview("qwen3_omni_instruct_int8", True, 0, **common)
    assert missing_slider["value"] == 0
    assert "Download the checkpoint" in missing_note


def test_caption_tab_controls_drive_the_job_offload_plan(tmp_path: Path) -> None:
    """Whatever the Gradio controls hold is what the job runs with; nothing is re-derived."""

    import gradio as gr

    from vcap.core.logs import get_log
    from vcap.core.presets import PresetStore
    from vcap.pipeline.client import PipelineClient
    from vcap.ui.app import UiContext

    client = PipelineClient(subprocess_mode=True)
    ctx = UiContext(
        settings_registry=SettingsRegistry(),
        preset_store=PresetStore(Path("presets"), Path("presets_default")),
        pipeline_client=client,
        app_log=get_log(),
    )
    try:
        with gr.Blocks():
            caption_tab.build(ctx)
        registry = ctx.settings_registry
        settings = registry.defaults()
        assert settings["block_swap_auto"] is True
        assert "gpu_layers" not in settings

        def offload_for(**overrides: Any) -> Any:
            values = {**settings, **overrides}
            return JobSpec.from_settings(values, [], OutputSpec(outputs_root=tmp_path)).model.offload

        automatic = offload_for()
        assert (automatic.gpu_layers, automatic.vram_reserve_gb, automatic.swap_slots, automatic.pin_cpu) == (
            "auto",
            2.0,
            2,
            True,
        )
        manual = offload_for(
            block_swap_auto=False,
            blocks_to_swap=30,
            vram_reserve_gb=4.5,
            swap_slots=3,
            pin_cpu=False,
            offload_experts=False,
        )
        assert (manual.gpu_layers, manual.vram_reserve_gb, manual.swap_slots, manual.pin_cpu) == (18, 4.5, 3, False)
        seven_b = offload_for(model_key="timechat_int4", block_swap_auto=False, blocks_to_swap=30)
        assert seven_b.gpu_layers == 0  # 28-layer family: a larger swap count clamps to all layers
        whole_decoder = offload_for(block_swap_auto=False, blocks_to_swap=0)
        assert whole_decoder.gpu_layers == 48
        # A preset saved before the controls existed carries gpu_layers only; it is
        # translated into the controls, and the job then reads the controls.
        legacy = {key: value for key, value in settings.items() if key not in {"block_swap_auto", "blocks_to_swap"}}
        migrated, warnings = registry.coerce({**legacy, "gpu_layers": "40"})
        assert warnings == []
        assert (migrated["block_swap_auto"], migrated["blocks_to_swap"]) == (False, 8)
        assert offload_for(**migrated).gpu_layers == 40
    finally:
        client.shutdown()


def test_gguf_loader_forwards_the_ui_vram_reserve(tmp_path: Path, monkeypatch) -> None:
    from vcap.models import llamacpp_backend, llamacpp_install, loader
    from vcap.models.registry import get_variant

    variant = get_variant("qwen3_omni_instruct_gguf_q8")
    for name in variant.gguf_files:
        (tmp_path / name).write_bytes(b"gguf")
    captured: dict[str, Any] = {}

    class FakeBackend:
        def __init__(self, family: str, **kwargs: Any) -> None:
            captured["family"] = family
            captured.update(kwargs)
            self.load_peak_vram_gb = 1.25
            self.loaded = None

        def start(self, progress_cb: Any) -> None:
            captured["started"] = True

        def block_swap_summary(self) -> dict[str, Any]:
            return {"mode": "llamacpp_fit"}

    monkeypatch.setattr(llamacpp_backend, "LlamaCppCaptioner", FakeBackend)
    monkeypatch.setattr(llamacpp_install, "ensure_llamacpp", lambda progress_cb: tmp_path / "llama-server.exe")
    monkeypatch.setattr(loader, "resource_snapshot", lambda index: {"vram_total_gb": 32.0})
    monkeypatch.setattr(loader, "_gguf_checkpoint_bytes", lambda folder, variant: 4)

    plan = OffloadPlan("auto", vram_reserve_gb=3.5, swap_slots=2)
    loaded = loader._load_llamacpp_model(
        variant.key,
        device="cuda:0",
        device_index=0,
        gpu_index=0,
        offload=plan,
        progress_cb=None,
        model_dir=tmp_path,
        context_size=16_384,
    )
    assert captured["started"] is True
    assert captured["vram_reserve_gb"] == 3.5
    assert captured["context_size"] == 16_384
    assert captured["vram_total_gb"] == 32.0
    assert loaded.offload is plan
    assert loaded.load_report.block_swap == {"mode": "llamacpp_fit"}
    assert isinstance(loaded.model, FakeBackend)
