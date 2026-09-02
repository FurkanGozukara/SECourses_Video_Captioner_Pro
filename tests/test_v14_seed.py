from __future__ import annotations

from types import SimpleNamespace

from vcap.models.base import (
    Callbacks,
    CaptionTiming,
    ChatResult,
    GenParams,
    TokenUsage,
)
from vcap.models.omni_common import _sampling_seed, _seed_torch
from vcap.pipeline.chat import ChatResponse


def test_sampled_seed_is_resolved_and_applied_to_cpu_and_cuda(monkeypatch) -> None:
    messages: list[str] = []
    callbacks = Callbacks(progress=lambda message, *_args: messages.append(str(message)))
    assert _sampling_seed(GenParams(do_sample=True, temperature=0.6, seed=123), callbacks, "test") == 123
    monkeypatch.setattr("vcap.models.omni_common.secrets.randbits", lambda _bits: 4_000_000_001)
    assert _sampling_seed(GenParams(do_sample=True, temperature=0.6, seed=-1), callbacks, "test") == 4_000_000_001

    calls: list[tuple[str, int]] = []
    fake_torch = SimpleNamespace(
        manual_seed=lambda seed: calls.append(("cpu", seed)),
        cuda=SimpleNamespace(manual_seed_all=lambda seed: calls.append(("cuda", seed))),
    )
    _seed_torch(fake_torch, 123)
    assert calls == [("cpu", 123), ("cuda", 123)]


def test_greedy_and_zero_temperature_runs_record_no_seed() -> None:
    messages: list[str] = []
    callbacks = Callbacks(progress=lambda message, *_args: messages.append(str(message)))
    assert _sampling_seed(GenParams(do_sample=False, temperature=0.6, seed=10), callbacks, "test") is None
    assert _sampling_seed(GenParams(do_sample=True, temperature=0.0, seed=10), callbacks, "test") is None
    assert messages == ["Sampling is enabled but temperature is 0; greedy decoding is used."]


def test_chat_response_exposes_the_seed_used_by_generation() -> None:
    result = ChatResult(
        text="answer",
        raw_text="answer",
        usage=TokenUsage(10, 2, "eos", 77),
        timing=CaptionTiming(0.1, 0.2, 10.0, 0.3, 2),
    )
    response = ChatResponse.from_result("qwen3_omni_instruct_int4", result)
    assert response.seed == 77
    assert ChatResponse.from_dict(response.to_dict()).seed == 77
