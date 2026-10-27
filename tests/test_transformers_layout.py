"""Load the shipped checkpoint naming into real current Transformers models."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from safetensors import safe_open
from safetensors.torch import save_file

from vcap.models.layout import language_model, thinker_layout
from vcap.models.quant import convrot


def _tiny_model(family: str):
    if family == "qwen3":
        from transformers import Qwen3OmniMoeThinkerForConditionalGeneration as Model
        from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
            Qwen3OmniMoeAudioEncoderConfig as Audio,
            Qwen3OmniMoeTextConfig as Text,
            Qwen3OmniMoeThinkerConfig as Config,
            Qwen3OmniMoeVisionEncoderConfig as Vision,
        )
        text_extra = dict(moe_intermediate_size=32, num_experts=4, num_experts_per_tok=2, head_dim=16)
        vision_extra = dict(num_position_embeddings=16, deepstack_visual_indexes=[0, 1])
        audio_extra = dict(n_window_infer=32, conv_chunksize=32)
    else:
        from transformers import Qwen2_5OmniThinkerForConditionalGeneration as Model
        from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
            Qwen2_5OmniAudioEncoderConfig as Audio,
            Qwen2_5OmniTextConfig as Text,
            Qwen2_5OmniThinkerConfig as Config,
            Qwen2_5OmniVisionEncoderConfig as Vision,
        )
        text_extra = dict(rope_parameters={
            "rope_type": "default", "rope_theta": 10000.0, "mrope_section": [2, 3, 3],
        })
        vision_extra = dict(fullatt_block_indexes=[0, 1], window_size=16)
        audio_extra = {}
    text = Text(
        hidden_size=64, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=256,
        max_position_embeddings=128, **text_extra,
    )
    vision = Vision(
        depth=2, hidden_size=32, num_heads=2, out_hidden_size=64,
        intermediate_size=32, patch_size=4, temporal_patch_size=2,
        spatial_merge_size=2, **vision_extra,
    )
    audio = Audio(
        d_model=32, encoder_layers=1, encoder_attention_heads=2,
        encoder_ffn_dim=32, output_dim=64, num_mel_bins=8,
        max_source_positions=16, n_window=8, **audio_extra,
    )
    config = Config(audio_config=audio, vision_config=vision, text_config=text)
    config._attn_implementation = "sdpa"
    return Model(config).eval()


def _legacy_tensors(model: nn.Module) -> dict[str, torch.Tensor]:
    tensors = {}
    for name, value in model.state_dict().items():
        if name.startswith("model.language_model."):
            name = "model." + name[len("model.language_model.") :]
        elif name.startswith(("model.visual.", "model.audio_tower.")):
            name = name[len("model.") :]
        name = "thinker." + name
        value = value.detach().contiguous()
        if name.endswith(".experts.gate_up_proj"):
            prefix = name[: -len(".gate_up_proj")]
            for expert, packed in enumerate(value):
                gate, up = packed.chunk(2, dim=0)
                tensors[f"{prefix}.{expert}.gate_proj.weight"] = gate.clone()
                tensors[f"{prefix}.{expert}.up_proj.weight"] = up.clone()
        elif name.endswith(".experts.down_proj"):
            prefix = name[: -len(".down_proj")]
            for expert, weight in enumerate(value):
                tensors[f"{prefix}.{expert}.down_proj.weight"] = weight.clone()
        else:
            tensors[name] = value.clone()
    return tensors


@pytest.mark.parametrize("family", ["qwen3", "qwen2_5"])
@pytest.mark.parametrize("reader", ["stream", "safetensors"])
def test_stream_legacy_checkpoint_into_current_real_model(
    tmp_path: Path, family: str, reader: str, monkeypatch,
) -> None:
    if reader == "safetensors":
        # Exercise the real reader used on Linux as well as our Windows reader.
        monkeypatch.setattr(convrot, "_open_checkpoint", lambda path: safe_open(
            str(path), framework="pt", device="cpu",
        ))
    torch.manual_seed(31)
    reference = _tiny_model(family)
    checkpoint = tmp_path / "model.safetensors"
    save_file(_legacy_tensors(reference), checkpoint)
    with torch.device("meta"):
        restored = _tiny_model(family)
    placements = []
    convrot.apply_quantized_checkpoint(
        restored, checkpoint, device="cpu", dtype=torch.float32,
        layer_device=lambda index: placements.append(index) or "cpu", tower_offload=False,
    )
    assert set(placements) == {0, 1}
    assert not any(value.is_meta for value in (*restored.parameters(), *restored.buffers()))
    for name, expected in reference.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], expected, rtol=0, atol=0)
    with torch.inference_mode():
        inputs = torch.tensor([[3, 4, 5, 6]])
        expected = reference(input_ids=inputs, use_cache=False).logits
        actual = restored(input_ids=inputs, use_cache=False).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("scheme", ["int8_convrot", "int4_convrot_w4a8"])
@pytest.mark.parametrize("reader", ["stream", "safetensors"])
def test_quant_metadata_and_experts_follow_current_model_layout(
    tmp_path: Path, scheme: str, reader: str, monkeypatch,
) -> None:
    if reader == "safetensors":
        monkeypatch.setattr(convrot, "_open_checkpoint", lambda path: safe_open(
            str(path), framework="pt", device="cpu",
        ))
    reference = _tiny_model("qwen3")
    tensors = _legacy_tensors(reference)
    selected = [
        name[: -len(".weight")] for name in tensors
        if name.endswith(".weight") and (
            name.endswith(".self_attn.q_proj.weight") or ".mlp.experts." in name
        )
    ]
    layers = {}
    for name in selected:
        weight = tensors[name + ".weight"]
        shape = tuple(weight.shape)
        stored_shape = shape if scheme == "int8_convrot" else (shape[0], shape[1] // 2)
        tensors[name + ".weight"] = torch.zeros(stored_shape, dtype=torch.int8)
        tensors[name + ".weight_scale"] = torch.ones((shape[0], 1), dtype=torch.float32)
        layers[name] = {"weight_shape": shape, "convrot_groupsize": 4}
    checkpoint = tmp_path / "model.safetensors"
    save_file(tensors, checkpoint, metadata={"_quantization_metadata": json.dumps({
        "scheme": scheme, "group_size": 4, "scale_layout": "row", "layers": layers,
    })})
    with torch.device("meta"):
        model = _tiny_model("qwen3")
    report = convrot.apply_quantized_checkpoint(
        model, checkpoint, device="cpu", dtype=torch.float32, tower_offload=False,
    )
    assert report.quantized_layers == len(selected)
    expected_type = convrot.ConvRotInt8Experts if scheme == "int8_convrot" else convrot.ConvRotInt4Experts
    for layer in language_model(model).layers:
        assert isinstance(layer.mlp.experts, expected_type)
        assert torch.count_nonzero(layer.mlp.experts.gate_up_weight) == 0
        assert torch.all(layer.mlp.experts.down_weight_scale == 1)
        assert layer.self_attn.q_proj.weight.dtype == torch.int8
    with torch.inference_mode():
        logits = model(input_ids=torch.tensor([[3, 4, 5, 6]]), use_cache=False).logits
    assert torch.isfinite(logits).all()


def test_current_decoder_used_for_swap_compile_and_outer_model() -> None:
    from vcap.models.loader import _block_swap_layers
    from vcap.models.torch_compile import _language_target

    model = _tiny_model("qwen3")
    decoder = model.model.language_model
    assert _block_swap_layers(model, "qwen3_omni_instruct", 2) is decoder.layers
    assert _language_target(model) is decoder
    assert _language_target(SimpleNamespace(thinker=model)) is decoder


@pytest.mark.parametrize("inferred", [False, True])
def test_legacy_dispatch_maps_names_but_preserves_inferred_runtime_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, inferred: bool,
) -> None:
    import accelerate
    from vcap.models import loader
    from vcap.models.offload import OffloadPlan

    model = _tiny_model("qwen3")
    legacy = {"audio_tower": 0, "visual": 0, "model.embed_tokens": 0, "model.layers.0": "cpu"}
    runtime = {"model": 0, "lm_head": "cpu"}
    observed = {}
    monkeypatch.setattr(accelerate, "infer_auto_device_map", lambda *_args, **_kwargs: runtime)
    monkeypatch.setattr(accelerate, "dispatch_model", lambda _model, **kwargs: observed.update(kwargs))
    monkeypatch.setattr(loader, "TEMP_DIR", tmp_path)
    placement = SimpleNamespace(explicit_device_map=legacy, max_memory=None, no_split_modules=())
    loader._dispatch_after_cpu_load(
        model, "qwen3_omni_instruct", OffloadPlan(gpu_layers="all" if inferred else 1),
        placement, torch.float32, device_index=0,
    )
    expected = runtime if inferred else {
        "model.audio_tower": 0, "model.visual": 0,
        "model.language_model.embed_tokens": 0, "model.language_model.layers.0": "cpu",
    }
    assert observed["device_map"] == expected
    assert "audio_tower" in legacy  # The reusable plan must not be mutated.


def test_tower_offload_wraps_actual_current_forward_owners(tmp_path: Path, monkeypatch) -> None:
    model = _tiny_model("qwen3")
    towers = (model.model.audio_tower, model.model.visual)
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.touch()
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device: (1, 1))
    original_empty = torch.empty
    monkeypatch.setattr(torch, "empty", lambda *args, **kwargs: original_empty(
        *args, **{**kwargs, "device": "cpu"},
    ))
    names = convrot._offload_prefill_towers_if_needed(model, checkpoint, torch.device("cuda:0"), force=True)
    assert names == ("model.audio_tower", "model.visual")
    for name, tower in zip(names, towers):
        assert model.get_submodule(name).module is tower
    assert convrot._offload_prefill_towers_if_needed(
        model, checkpoint, torch.device("cuda:0"), force=True,
    ) == ()


def test_legacy_layout_remains_supported() -> None:
    model = nn.Module()
    model.model = nn.Module()
    layout = thinker_layout(model)
    for name in ("model.embed_tokens.weight", "audio_tower.proj.weight", "lm_head.weight"):
        assert layout.checkpoint_name(name) == name
    assert language_model(model) is model.model
