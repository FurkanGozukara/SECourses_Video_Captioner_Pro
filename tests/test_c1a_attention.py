from __future__ import annotations

from types import SimpleNamespace

import pytest

from vcap.models import attention


def test_attention_descriptions_and_registration() -> None:
    descriptions = attention.describe_available()
    assert set(descriptions) == set(attention.ATTENTION_CHOICES)
    assert all(value for value in descriptions.values())

    available = attention.probe_available()
    from transformers import AttentionInterface

    for backend in ("sage", "xformers"):
        implementation, _ = attention.resolve(backend, "timechat", "bfloat16")
        if available[backend]:
            assert implementation == backend
            assert backend in AttentionInterface().keys()
        else:
            assert implementation == "sdpa"


@pytest.mark.parametrize(
    "implementation",
    [attention.sage_attention_forward, attention.xformers_attention_forward],
)
def test_custom_attention_uses_sdpa_for_unsafe_cpu_tensors(implementation) -> None:
    torch = pytest.importorskip("torch")
    generator = torch.Generator().manual_seed(7)
    query = torch.randn(1, 2, 4, 8, generator=generator)
    key = torch.randn(1, 2, 4, 8, generator=generator)
    value = torch.randn(1, 2, 4, 8, generator=generator)
    module = SimpleNamespace(is_causal=True, num_key_value_groups=1)

    actual, weights = implementation(
        module,
        query,
        key,
        value,
        None,
        is_causal=True,
    )
    expected = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        is_causal=True,
    ).transpose(1, 2).contiguous()
    assert weights is None
    torch.testing.assert_close(actual, expected)
