from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from vcap.models.quant import convrot


class _DenseMlp(nn.Module):
    def __init__(self, linear_type, *, device: str = "cpu", dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        del dtype
        self.gate_proj = linear_type(256, 64, group_size=256, device=device)
        self.up_proj = linear_type(256, 64, group_size=256, device=device)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)


def _randomize(module: _DenseMlp) -> None:
    for projection in (module.gate_proj, module.up_proj):
        projection.weight.copy_(
            torch.randint(
                -128,
                128,
                projection.weight.shape,
                dtype=torch.int8,
                device=projection.weight.device,
            )
        )
        projection.weight_scale.copy_(
            torch.rand(
                projection.weight_scale.shape,
                dtype=torch.float32,
                device=projection.weight_scale.device,
            )
            * 0.02
            + 0.001
        )


@pytest.mark.parametrize(
    "linear_type",
    [convrot.ConvRotInt8Linear, convrot.ConvRotInt4W4A8Linear],
)
def test_dense_gate_up_fusion_matches_unfused_cpu(linear_type) -> None:
    torch.manual_seed(123)
    module = _DenseMlp(linear_type)
    _randomize(module)
    hidden_states = torch.randn(3, 256)
    before = module(hidden_states)
    original_bytes = sum(
        tensor.numel() * tensor.element_size()
        for projection in (module.gate_proj, module.up_proj)
        for tensor in projection.buffers()
    )

    assert convrot._fuse_quantized_qkv(module) == 1
    after = module(hidden_states)
    shared = module.gate_proj.shared
    fused_bytes = sum(tensor.numel() * tensor.element_size() for tensor in shared.buffers())

    torch.testing.assert_close(after, before, rtol=1e-6, atol=1e-6)
    assert fused_bytes == original_bytes
    assert not any(
        isinstance(child, convrot._ConvRotLinearBase) for child in module.modules()
    )


def test_hadamard_lookup_is_hoisted_after_first_module_call(monkeypatch) -> None:
    calls = 0
    original = convrot._regular_hadamard

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(convrot, "_regular_hadamard", counted)
    layer = convrot.ConvRotInt8Linear(256, 32, group_size=256)
    layer.weight.zero_()
    layer.weight_scale.fill_(1.0)
    values = torch.randn(1, 256)
    layer(values)
    layer(values)
    assert calls == 1


@pytest.mark.parametrize(
    "linear_type",
    [convrot.ConvRotInt8Linear, convrot.ConvRotInt4W4A8Linear],
)
def test_dense_gate_up_fusion_is_bit_identical_on_cuda(linear_type) -> None:
    if os.environ.get("VCAP_GPU_TESTS") != "1" or not torch.cuda.is_available():
        pytest.skip("set VCAP_GPU_TESTS=1 for the exact CUDA ConvRot kernel test")
    torch.cuda.set_device(0)
    torch.manual_seed(321)
    module = _DenseMlp(linear_type, device="cuda").eval()
    _randomize(module)
    hidden_states = torch.randn(1, 256, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        before = module(hidden_states)
        torch.cuda.synchronize()
        assert convrot._fuse_quantized_qkv(module) == 1
        after = module(hidden_states)
        torch.cuda.synchronize()
    assert torch.equal(after, before)
