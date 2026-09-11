"""Regression tests for rebuilding rotary buffers that meta-device construction leaves empty.

transformers 5.17 rewrote ``Qwen3OmniMoeVisionRotaryEmbedding`` to be config driven
(``compute_axial_rope_parameters`` plus an ``original_inv_freq`` buffer, no ``dim``/``theta``),
which made ``apply_quantized_checkpoint`` fail with
``Checkpoint load left meta tensors: ['visual.rotary_pos_emb.inv_freq', ...]``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from vcap.models.quant import convrot  # noqa: E402


def _vision_config() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=1152,
        num_attention_heads=16,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "axial"},
    )


class AxialVisionRotary(nn.Module):
    """Mirror of transformers>=5.17 ``Qwen3OmniMoeVisionRotaryEmbedding``."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.rope_type = config.rope_parameters["rope_type"]
        inv_freq, self.attention_scaling = self.compute_axial_rope_parameters(config)
        self.inv_freq = nn.Buffer(inv_freq, persistent=False)
        self.original_inv_freq = nn.Buffer(inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_axial_rope_parameters(config, device=None, **kwargs):
        base = config.rope_parameters["rope_theta"]
        dim = config.hidden_size // config.num_attention_heads
        spatial_dim = dim // 2
        inv_freq = 1.0 / (
            base ** (torch.arange(0, spatial_dim, 2, dtype=torch.float) / spatial_dim)
        )
        return inv_freq.to(device), 1.0


class LegacyVisionRotary(nn.Module):
    """Mirror of transformers<=5.16 ``Qwen3OmniMoeVisionRotaryEmbedding``."""

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.inv_freq = nn.Buffer(inv_freq, persistent=False)


class TextRotary(nn.Module):
    """Mirror of the transformers 5.x text rotary embedding, including scaled rope types."""

    def __init__(self, config, rope_init_fn=None) -> None:
        super().__init__()
        self.config = config
        self.rope_type = config.rope_parameters["rope_type"]
        rope_init_fn = rope_init_fn or self.compute_default_rope_parameters
        inv_freq, self.attention_scaling = rope_init_fn(config)
        self.inv_freq = nn.Buffer(inv_freq, persistent=False)
        self.original_inv_freq = nn.Buffer(inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_default_rope_parameters(config, device=None, **kwargs):
        base = config.rope_parameters["rope_theta"]
        dim = config.head_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        return inv_freq.to(device), 1.0


def _meta_buffers(module: nn.Module) -> list[str]:
    return [name for name, value in module.named_buffers() if value.device.type == "meta"]


def test_axial_vision_rotary_is_rebuilt_from_config() -> None:
    config = _vision_config()
    with torch.device("meta"):
        module = AxialVisionRotary(config)
    assert _meta_buffers(module) == ["inv_freq", "original_inv_freq"]

    convrot._materialize_meta_buffers(module, torch.device("cpu"))

    expected = AxialVisionRotary(config)
    assert _meta_buffers(module) == []
    assert torch.equal(module.inv_freq, expected.inv_freq)
    assert torch.equal(module.original_inv_freq, expected.original_inv_freq)
    assert module.attention_scaling == expected.attention_scaling


def test_legacy_dim_theta_vision_rotary_is_still_rebuilt() -> None:
    with torch.device("meta"):
        module = LegacyVisionRotary(36)

    convrot._materialize_meta_buffers(module, torch.device("cpu"))

    assert _meta_buffers(module) == []
    assert torch.equal(module.inv_freq, LegacyVisionRotary(36).inv_freq)


def test_text_rotary_default_type_uses_module_recipe() -> None:
    config = SimpleNamespace(
        head_dim=128, rope_parameters={"rope_theta": 1_000_000.0, "rope_type": "default"}
    )
    with torch.device("meta"):
        module = TextRotary(config)

    convrot._materialize_meta_buffers(module, torch.device("cpu"))

    assert _meta_buffers(module) == []
    assert torch.equal(module.inv_freq, TextRotary(config).inv_freq)
    assert torch.equal(module.original_inv_freq, module.inv_freq)


def test_scaled_rope_type_resolves_through_transformers_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rope_utils = pytest.importorskip("transformers.modeling_rope_utils")

    def scaled(config, device=None, **kwargs):
        return torch.full((4,), 0.25) * config.rope_parameters["factor"], 0.5

    monkeypatch.setitem(rope_utils.ROPE_INIT_FUNCTIONS, "vcap_test_scaled", scaled)
    config = SimpleNamespace(
        head_dim=8,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "vcap_test_scaled", "factor": 2.0},
    )
    with torch.device("meta"):
        module = TextRotary(config, rope_init_fn=scaled)
    module.attention_scaling = None

    convrot._materialize_meta_buffers(module, torch.device("cpu"))

    # The registry entry must win over the module's own default recipe for scaled types.
    assert _meta_buffers(module) == []
    assert torch.equal(module.inv_freq, torch.full((4,), 0.5))
    assert torch.equal(module.original_inv_freq, torch.full((4,), 0.5))
    assert module.attention_scaling == 0.5


def test_apply_quantized_checkpoint_loads_visual_tower_with_axial_rotary(tmp_path: Path) -> None:
    from safetensors.torch import save_file

    config = _vision_config()

    class TinyThinker(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.visual = nn.Module()
            self.visual.rotary_pos_emb = AxialVisionRotary(config)
            self.visual.proj = nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)

    with torch.device("meta"):
        model = TinyThinker()
    assert _meta_buffers(model) == [
        "visual.rotary_pos_emb.inv_freq",
        "visual.rotary_pos_emb.original_inv_freq",
    ]
    weight = torch.arange(4, dtype=torch.bfloat16).reshape(2, 2)
    checkpoint = tmp_path / "model.safetensors"
    save_file({"thinker.visual.proj.weight": weight}, checkpoint)

    report = convrot.apply_quantized_checkpoint(
        model, checkpoint, device="cpu", dtype=torch.bfloat16, tower_offload=False
    )

    assert report.bf16_layers == 1
    assert _meta_buffers(model) == []
    assert not any(parameter.device.type == "meta" for parameter in model.parameters())
    assert torch.equal(model.visual.proj.weight, weight)
    expected = AxialVisionRotary(config)
    assert torch.equal(model.visual.rotary_pos_emb.inv_freq, expected.inv_freq)
    assert torch.equal(model.visual.rotary_pos_emb.original_inv_freq, expected.inv_freq)


def test_rotary_without_recipe_reports_owner_and_transformers_version(tmp_path: Path) -> None:
    from safetensors.torch import save_file

    class OpaqueRotary(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inv_freq = nn.Buffer(torch.empty(4, device="meta"), persistent=False)

    model = nn.Module()
    model.visual = nn.Module()
    model.visual.rotary_pos_emb = OpaqueRotary()
    model.visual.proj = nn.Linear(2, 2, bias=False, dtype=torch.bfloat16, device="meta")
    checkpoint = tmp_path / "model.safetensors"
    save_file({"thinker.visual.proj.weight": torch.ones(2, 2, dtype=torch.bfloat16)}, checkpoint)

    with pytest.raises(
        RuntimeError,
        match=r"left meta tensors.*visual\.rotary_pos_emb\.inv_freq.*OpaqueRotary.*transformers",
    ):
        convrot.apply_quantized_checkpoint(
            model, checkpoint, device="cpu", dtype=torch.bfloat16, tower_offload=False
        )


def test_qwen3_omni_thinker_meta_buffers_are_all_rebuilt() -> None:
    transformers = pytest.importorskip("transformers")
    from transformers import Qwen3OmniMoeThinkerForConditionalGeneration
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
        Qwen3OmniMoeAudioEncoderConfig,
        Qwen3OmniMoeTextConfig,
        Qwen3OmniMoeThinkerConfig,
        Qwen3OmniMoeVisionEncoderConfig,
    )

    text = Qwen3OmniMoeTextConfig(
        hidden_size=64,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_experts=4,
        num_experts_per_tok=2,
        vocab_size=256,
        max_position_embeddings=128,
    )
    vision = Qwen3OmniMoeVisionEncoderConfig(
        depth=2,
        hidden_size=32,
        num_heads=2,
        out_hidden_size=64,
        intermediate_size=32,
        num_position_embeddings=16,
        deepstack_visual_indexes=[0, 1],
        patch_size=4,
        temporal_patch_size=2,
        spatial_merge_size=2,
    )
    audio = Qwen3OmniMoeAudioEncoderConfig(
        d_model=32,
        encoder_layers=1,
        encoder_attention_heads=2,
        encoder_ffn_dim=32,
        output_dim=64,
        num_mel_bins=8,
        max_source_positions=16,
        n_window=8,
        n_window_infer=32,
        conv_chunksize=32,
    )
    config = Qwen3OmniMoeThinkerConfig(audio_config=audio, vision_config=vision, text_config=text)
    with torch.device("meta"):
        model = Qwen3OmniMoeThinkerForConditionalGeneration(config)
    assert "visual.rotary_pos_emb.inv_freq" in _meta_buffers(model)

    convrot._materialize_meta_buffers(model, torch.device("cpu"))

    assert _meta_buffers(model) == [], f"transformers {transformers.__version__} left meta buffers"
    rotary = model.visual.rotary_pos_emb
    if hasattr(rotary, "config"):
        fresh = type(rotary)(rotary.config)
    else:
        fresh = type(rotary)(rotary.dim, rotary.theta)
    assert torch.equal(rotary.inv_freq, fresh.inv_freq)
    text_rotary = model.model.rotary_emb
    fresh_text = type(text_rotary)(text_rotary.config)
    assert torch.equal(text_rotary.inv_freq, fresh_text.inv_freq)
    assert torch.equal(text_rotary.original_inv_freq, fresh_text.inv_freq)
