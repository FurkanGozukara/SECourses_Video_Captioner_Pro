from __future__ import annotations

import copy
from itertools import chain

import pytest
import torch
from torch import nn

from vcap.models.block_swap import BlockSwapManager


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")

_WIDTH = 64
_LAYERS = 8


class _MixedLayer(nn.Module):
    def __init__(self, width: int = _WIDTH) -> None:
        super().__init__()
        self.linear = nn.Linear(width, width, dtype=torch.bfloat16)
        self.register_buffer("quant_code", torch.randint(-8, 8, (width,), dtype=torch.int8))
        self.register_buffer("quant_scale", torch.rand(width, dtype=torch.float32) / 32)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        correction = self.quant_code.to(torch.float32) * self.quant_scale
        return torch.tanh(self.linear(value) + correction.to(value.dtype))


class _Stack(nn.Module):
    def __init__(self, count: int = _LAYERS, width: int = _WIDTH) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_MixedLayer(width) for _ in range(count)])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            value = layer(value)
        return value


def _models(resident: int) -> tuple[_Stack, _Stack, torch.device]:
    torch.manual_seed(1234)
    template = _Stack().eval()
    baseline = copy.deepcopy(template).eval()
    swapped = copy.deepcopy(template).eval()
    device = torch.device("cuda", torch.cuda.current_device())
    baseline.to(device)
    for layer in swapped.layers[:resident]:
        layer.to(device)
    return baseline, swapped, device


def _assert_layer_devices(model: _Stack, resident: int, device: torch.device) -> None:
    for index, layer in enumerate(model.layers):
        expected = device.type if index < resident else "cpu"
        tensors = chain(layer.parameters(), layer.buffers())
        assert all(tensor.device.type == expected for tensor in tensors)


@pytest.mark.parametrize("resident", [0, 3, 7])
@pytest.mark.parametrize("slots", [2, 3])
def test_block_swap_matches_resident_stack_and_allocations_are_stable(
    resident: int,
    slots: int,
) -> None:
    baseline, swapped, device = _models(resident)
    manager = BlockSwapManager.install(
        swapped,
        swapped.layers,
        resident=resident,
        slots=slots,
        device=device,
    )
    generator = torch.Generator(device="cpu").manual_seed(9876)
    allocation_after_second = None

    try:
        with torch.inference_mode():
            for step in range(20):
                batch = 1 + (step * 7) % 6
                value = torch.randn(batch, _WIDTH, generator=generator, dtype=torch.bfloat16).to(device)
                expected = baseline(value)
                actual = swapped(value)
                assert torch.equal(actual, expected)
                _assert_layer_devices(swapped, resident, device)
                del actual, expected, value
                torch.cuda.synchronize(device)
                allocated = torch.cuda.memory_allocated(device)
                if step == 1:
                    allocation_after_second = allocated
                if step == 19:
                    assert allocated == allocation_after_second

        stats = manager.stats()
        assert stats["forwards"] == 20
        assert stats["layer_loads"] == 20 * (_LAYERS - resident)
        assert stats["bytes_h2d"] > 0
        manager.reset_stats()
        assert manager.stats()["forwards"] == 0
    finally:
        manager.remove()


def test_signature_mismatch_names_first_different_tensor() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    root = _Stack(count=3)
    root.layers[2] = _MixedLayer(_WIDTH + 1)

    with pytest.raises(ValueError, match=r"signature mismatch.*linear\.weight"):
        BlockSwapManager.install(root, root.layers, resident=0, slots=2, device=device)

    assert not hasattr(root, "_vcap_block_swap")


def test_remove_releases_slots_and_clears_root_flags() -> None:
    _baseline, swapped, device = _models(resident=3)
    torch.cuda.synchronize(device)
    before = torch.cuda.memory_allocated(device)
    manager = BlockSwapManager.install(
        swapped,
        swapped.layers,
        resident=3,
        slots=3,
        device=device,
        pin=False,
    )
    installed = torch.cuda.memory_allocated(device)

    assert installed > before
    assert swapped._vcap_block_swap is True
    assert swapped._vcap_block_swap_manager is manager
    manager.remove()
    torch.cuda.synchronize(device)

    assert torch.cuda.memory_allocated(device) == before
    assert manager.summary()["installed"] is False
    assert not hasattr(swapped, "_vcap_block_swap")
    assert not hasattr(swapped, "_vcap_block_swap_manager")
    _assert_layer_devices(swapped, 3, device)


def test_pageable_fallback_when_both_pinning_methods_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FailedCudart:
        @staticmethod
        def cudaHostRegister(_pointer: int, _nbytes: int, _flags: int) -> int:
            return 1

        @staticmethod
        def cudaHostUnregister(_pointer: int) -> int:
            return 0

    real_empty = torch.empty

    def fail_pinned_empty(*args, **kwargs):
        if kwargs.get("pin_memory"):
            raise RuntimeError("pinned allocator unavailable")
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch.cuda, "cudart", lambda: _FailedCudart())
    monkeypatch.setattr(torch, "empty", fail_pinned_empty)
    baseline, swapped, device = _models(resident=6)
    messages: list[str] = []
    manager = BlockSwapManager.install(
        swapped,
        swapped.layers,
        resident=6,
        slots=2,
        device=device,
        pin=True,
        log=messages.append,
    )

    try:
        assert manager.pinned_bytes == 0
        assert manager.pageable_bytes > 0
        assert manager.summary()["pin_method"] == "pageable"
        assert sum("remaining layers use pageable buffers" in message for message in messages) == 1
        with torch.inference_mode():
            value = torch.randn(2, _WIDTH, dtype=torch.bfloat16, device=device)
            assert torch.equal(swapped(value), baseline(value))
        _assert_layer_devices(swapped, 6, device)
    finally:
        manager.remove()


def test_lazy_kickoff_when_swapped_layer_is_called_directly() -> None:
    _baseline, swapped, device = _models(resident=1)
    reference = copy.deepcopy(swapped.layers[4]).to(device).eval()
    manager = BlockSwapManager.install(
        swapped,
        swapped.layers,
        resident=1,
        slots=2,
        device=device,
        pin=False,
    )

    try:
        with torch.inference_mode():
            for batch in (1, 3):
                value = torch.randn(batch, _WIDTH, dtype=torch.bfloat16, device=device)
                expected = reference(value)
                actual = swapped.layers[4](value)
                assert torch.equal(actual, expected)
                assert all(tensor.device.type == "cpu" for tensor in swapped.layers[4].parameters())
                assert all(tensor.device.type == "cpu" for tensor in swapped.layers[4].buffers())
        assert manager.stats()["forwards"] == 2
        assert manager.stats()["layer_loads"] >= 2
    finally:
        manager.remove()
