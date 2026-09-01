from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_apply_checkpoint_honors_decoder_layer_device_for_bf16(tmp_path: Path) -> None:
    import torch
    from torch import nn
    from safetensors.torch import save_file

    from vcap.models.quant.convrot import apply_quantized_checkpoint

    class Rotary(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dim = 4
            self.theta = 10_000.0
            self.register_buffer(
                "inv_freq",
                torch.empty(2, dtype=torch.float32, device="meta"),
                persistent=False,
            )

    class Experts(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_parameter(
                "gate_up_proj",
                nn.Parameter(
                    torch.empty((1, 4, 2), dtype=torch.bfloat16, device="meta"),
                    requires_grad=False,
                ),
            )
            self.register_parameter(
                "down_proj",
                nn.Parameter(
                    torch.empty((1, 2, 2), dtype=torch.bfloat16, device="meta"),
                    requires_grad=False,
                ),
            )

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(2, 2, bias=True, dtype=torch.bfloat16, device="meta")
            self.mlp = nn.Module()
            self.mlp.experts = Experts()
            self.rotary = Rotary()
            self.register_buffer(
                "loaded_buffer",
                torch.empty(2, dtype=torch.float32, device="meta"),
            )

    class TinyThinker(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([Layer(), Layer()])

    model = TinyThinker()
    tensors: dict[str, torch.Tensor] = {}
    for index in range(2):
        prefix = f"thinker.model.layers.{index}"
        offset = index * 20
        tensors[f"{prefix}.proj.weight"] = (
            torch.arange(4, dtype=torch.bfloat16).reshape(2, 2) + offset
        )
        tensors[f"{prefix}.proj.bias"] = torch.tensor(
            [offset + 4, offset + 5], dtype=torch.bfloat16
        )
        tensors[f"{prefix}.loaded_buffer"] = torch.tensor(
            [offset + 6, offset + 7], dtype=torch.float32
        )
        tensors[f"{prefix}.mlp.experts.0.gate_proj.weight"] = torch.full(
            (2, 2), offset + 8, dtype=torch.bfloat16
        )
        tensors[f"{prefix}.mlp.experts.0.up_proj.weight"] = torch.full(
            (2, 2), offset + 9, dtype=torch.bfloat16
        )
        tensors[f"{prefix}.mlp.experts.0.down_proj.weight"] = torch.full(
            (2, 2), offset + 10, dtype=torch.bfloat16
        )
    checkpoint = tmp_path / "model.safetensors"
    save_file(tensors, checkpoint)

    placements: list[int] = []

    def decoder_device(index: int) -> str:
        placements.append(index)
        return "cpu"

    report = apply_quantized_checkpoint(
        model,
        checkpoint,
        device="meta",
        dtype=torch.bfloat16,
        layer_device=decoder_device,
        tower_offload=False,
    )

    assert report.bf16_layers == 2
    assert set(placements) == {0, 1}
    for index, layer in enumerate(model.model.layers):
        offset = index * 20
        assert {tensor.device.type for tensor in layer.parameters()} == {"cpu"}
        assert {tensor.device.type for tensor in layer.buffers()} == {"cpu"}
        assert torch.equal(layer.proj.weight, tensors[f"thinker.model.layers.{index}.proj.weight"])
        assert torch.equal(
            layer.mlp.experts.gate_up_proj[0, :2],
            torch.full((2, 2), offset + 8, dtype=torch.bfloat16),
        )
        assert torch.equal(
            layer.mlp.experts.gate_up_proj[0, 2:],
            torch.full((2, 2), offset + 9, dtype=torch.bfloat16),
        )
        assert torch.equal(
            layer.mlp.experts.down_proj[0],
            torch.full((2, 2), offset + 10, dtype=torch.bfloat16),
        )


def test_last_token_logits_hook_slices_only_when_enabled() -> None:
    import torch
    from torch import nn

    from vcap.models.loader import _install_last_token_logits_hook

    root = nn.Module()
    root.lm_head = nn.Linear(4, 3, bias=False)
    hidden = torch.arange(40, dtype=torch.float32).reshape(2, 5, 4)

    _install_last_token_logits_hook(root, True)
    sliced = root.lm_head(hidden)
    assert sliced.shape == (2, 1, 3)
    assert torch.equal(sliced, root.lm_head(hidden[:, -1:, :]))

    root._vcap_last_token_logits = False
    full = root.lm_head(hidden)
    assert full.shape == (2, 5, 3)


def test_loader_wires_block_swap_budget_and_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import psutil
    import torch
    from torch import nn

    from vcap.models import loader
    from vcap.models.offload import (
        BlockSwapBudget,
        BudgetHint,
        CheckpointLayout,
        OffloadPlan,
    )

    variant_key = "qwen3_omni_instruct_int8"
    family = "qwen3_omni_instruct"
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.touch()
    expected_layer_name = loader._FAMILY_LAYOUTS[family][1][-1]
    layer_type = type(expected_layer_name, (nn.Module,), {"forward": lambda self, x: x})

    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([layer_type(), layer_type()])
            self.lm_head = nn.Linear(4, 2, bias=False)
            self.config = SimpleNamespace()

    layout = CheckpointLayout(checkpoint, 2, (100, 100), 50, 250)
    budget = BlockSwapBudget(
        layer_count=2,
        resident_layers=1,
        swapped_layers=1,
        slots=2,
        layer_bytes=100,
        non_layer_bytes=50,
        resident_weight_bytes=350,
        activation_bytes=456,
        reserve_bytes=2**30,
        free_vram_bytes=8 * 2**30,
        total_vram_bytes=10 * 2**30,
        expected_peak_bytes=806,
        pinned_bytes=100,
        mode="block_swap",
        notes=("budget summary", "budget warning"),
    )
    calls: dict[str, object] = {}

    monkeypatch.setattr(loader, "resolve_attention", lambda *_args: ("sdpa", nullcontext()))
    monkeypatch.setattr(loader, "_model_types", lambda _architecture: (object, object))
    monkeypatch.setattr(
        loader,
        "_thinker_config",
        lambda *_args: SimpleNamespace(text_config=SimpleNamespace()),
    )
    monkeypatch.setattr(loader, "checkpoint_layout", lambda path: layout)
    monkeypatch.setattr(loader, "observed_activation_bytes", lambda key: 123)
    monkeypatch.setattr(loader, "observed_activation_ratio", lambda key: 1.5)

    def fake_estimate(family_arg, config, hint, *, observed_bytes=0, observed_ratio=1.0):
        calls["estimate"] = (family_arg, config, hint, observed_ratio)
        return 456

    def fake_plan(plan, layout_arg, **kwargs):
        calls["plan"] = (plan, layout_arg, kwargs)
        return budget

    monkeypatch.setattr(loader, "estimate_activation_bytes", fake_estimate)
    monkeypatch.setattr(loader, "plan_block_swap", fake_plan)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=12 * 2**30))

    def fake_stream(*_args, **kwargs):
        calls["stream"] = kwargs
        return TinyModel(), SimpleNamespace(quantized_layers=7, bf16_layers=3)

    monkeypatch.setattr(loader, "_stream_single_checkpoint", fake_stream)

    class Manager:
        def summary(self):
            return {"manager_ready": True, "slot_bytes": 100}

        def remove(self):
            return None

    def fake_install(cls, root, layers, **kwargs):
        calls["install"] = (root, layers, kwargs)
        manager = Manager()
        root._vcap_block_swap = True
        root._vcap_block_swap_manager = manager
        return manager

    monkeypatch.setattr(loader.BlockSwapManager, "install", classmethod(fake_install))
    monkeypatch.setattr(loader, "_processor", lambda *_args: SimpleNamespace())
    monkeypatch.setattr(loader, "_load_generation_config", lambda *_args: None)

    total = 10 * 2**30
    free = 8 * 2**30
    reserved = 1 * 2**30
    fractions: list[tuple[float, object]] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device: (free, total))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _device: reserved)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _device: 3 * 2**30)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device: 4 * 2**30)
    monkeypatch.setattr(
        torch.cuda,
        "set_per_process_memory_fraction",
        lambda fraction, device: fractions.append((fraction, device)),
    )

    hint = BudgetHint(max_frames=12, max_new_tokens=64)
    loaded = loader.load_model(
        variant_key,
        device="cuda:0",
        attention="sdpa",
        offload=OffloadPlan(gpu_layers="auto", swap_slots=2),
        budget_hint=hint,
        hf_dir=tmp_path,
    )

    estimate_call = calls["estimate"]
    assert estimate_call[0] == family
    assert estimate_call[2] is hint
    assert estimate_call[3] == 1.5
    plan_call = calls["plan"]
    assert plan_call[1] is layout
    assert plan_call[2]["free_vram_bytes"] == free
    assert plan_call[2]["total_vram_bytes"] == total
    assert plan_call[2]["activation_bytes"] == 456
    stream_kwargs = calls["stream"]
    assert stream_kwargs["layer_device"](0) == "cuda:0"
    assert stream_kwargs["layer_device"](1) == "cpu"
    assert stream_kwargs["tower_offload"] is False
    install_kwargs = calls["install"][2]
    assert install_kwargs["resident"] == 1
    assert install_kwargs["slots"] == 2
    assert install_kwargs["pin_budget_bytes"] == 6 * 2**30
    assert loaded.load_report.device_map == {"": "cuda:0"}
    assert loaded.load_report.quantized_layers == 7
    assert loaded.load_report.bf16_layers == 3
    assert loaded.load_report.resident_bytes == 3 * 2**30
    assert loaded.load_report.activation_estimate_bytes == 456
    assert loaded.load_report.block_swap["manager_ready"] is True
    assert loaded.load_report.vram_cap_bytes == int(total * 0.875)
    assert fractions[0][0] == pytest.approx(0.875)
    assert loaded.model._vcap_last_token_logits is True


def test_model_cache_reuses_plan_that_covers_hint_and_keys_last_token_logits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vcap.models import loader
    from vcap.models.offload import BudgetHint

    created: list[object] = []
    gib = 2**30

    def fake_load(_variant_key: str, **kwargs):
        hint = kwargs.get("budget_hint")
        planned = int(hint.max_frames) * gib if hint is not None else 0
        value = SimpleNamespace(
            model=SimpleNamespace(config=None),
            spec=SimpleNamespace(family="timechat"),
            variant=SimpleNamespace(key="variant"),
            load_report=SimpleNamespace(activation_estimate_bytes=planned),
        )
        created.append(value)
        return value

    # One "GiB per frame" keeps the arithmetic obvious: the plan made for 8 frames
    # covers any later job needing 8 or fewer, and must be rebuilt for 9.
    monkeypatch.setattr(
        loader,
        "estimate_activation_bytes",
        lambda _family, _config, hint, **_kwargs: int(hint.max_frames) * gib,
    )
    monkeypatch.setattr(loader, "observed_activation_bytes", lambda _variant: 0)
    monkeypatch.setattr(loader, "observed_activation_ratio", lambda _variant: 1.0)
    monkeypatch.setattr(loader, "load_model", fake_load)
    monkeypatch.setattr(loader, "unload_model", lambda _loaded: None)
    cache = loader.ModelCache()

    first = cache.load("variant", budget_hint=BudgetHint(max_frames=8), last_token_logits=True)
    assert cache.load("variant", budget_hint=BudgetHint(max_frames=8), last_token_logits=True) is first
    assert cache.load("variant", budget_hint=BudgetHint(max_frames=6), last_token_logits=True) is first
    assert len(created) == 1
    second = cache.load("variant", budget_hint=BudgetHint(max_frames=9), last_token_logits=True)
    assert second is not first
    assert len(created) == 2
    cache.load("variant", budget_hint=BudgetHint(max_frames=9), last_token_logits=False)
    assert len(created) == 3


def test_model_cache_reuses_loads_without_a_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    from vcap.models import loader
    from vcap.models.offload import BudgetHint

    created: list[object] = []

    def fake_load(_variant_key: str, **_kwargs):
        value = SimpleNamespace(model=object())
        created.append(value)
        return value

    monkeypatch.setattr(loader, "load_model", fake_load)
    monkeypatch.setattr(loader, "unload_model", lambda _loaded: None)
    cache = loader.ModelCache()
    first = cache.load("variant", budget_hint=BudgetHint(max_frames=8))
    # GGUF/legacy loads carry no activation plan, so any hint reuses them.
    assert cache.load("variant", budget_hint=BudgetHint(max_frames=800)) is first
    assert len(created) == 1
