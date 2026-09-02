from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vcap.models.base import Callbacks, GenParams, MediaInput, PromptSpec
from vcap.models.loader import _load_generation_config, resolve_stop_token_ids
from vcap.models.qwen3_omni import (
    Qwen3OmniAudioCaptioner,
    Qwen3OmniInstructCaptioner,
    Qwen3OmniThinkingCaptioner,
)
from vcap.models.registry import MODEL_SPECS


class _Tokenizer:
    pad_token_id = 303
    unk_token_id = None
    unk_token = None

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return {"<|im_end|>": 101, "<|endoftext|>": 202}.get(token)


class _Inputs(dict):
    def to(self, _value):
        return self


class _Processor:
    def __init__(self, decoded: str = "A single coherent caption paragraph.") -> None:
        self.tokenizer = _Tokenizer()
        self.decoded = decoded
        self.template_calls: list[tuple[list[dict], dict]] = []

    def apply_chat_template(self, conversation, **kwargs):
        self.template_calls.append((conversation, kwargs))
        return "rendered"

    def __call__(self, **_kwargs):
        import torch

        return _Inputs(
            input_ids=torch.tensor([[11, 12, 13]], dtype=torch.long),
            attention_mask=torch.ones((1, 3), dtype=torch.long),
        )

    def batch_decode(self, _ids, **_kwargs):
        return [self.decoded]


class _Model:
    def __init__(self, *, fail_flash_once: bool = False) -> None:
        self.generation_config = SimpleNamespace(eos_token_id=[101, 202], pad_token_id=303)
        self.calls: list[dict] = []
        self.fail_flash_once = fail_flash_once
        self.switched_to: str | None = None

    def generate(self, **kwargs):
        import torch

        self.calls.append(kwargs)
        if self.fail_flash_once:
            self.fail_flash_once = False
            raise RuntimeError("FlashAttention only supports bf16 on this architecture")
        eos = torch.tensor([[kwargs["eos_token_id"][0]]], dtype=torch.long)
        return torch.cat((kwargs["input_ids"], eos), dim=1)

    def set_attn_implementation(self, implementation: str) -> None:
        self.switched_to = implementation


def _loaded(family: str, processor: _Processor, model: _Model, attention: str = "sdpa"):
    import torch

    return SimpleNamespace(
        model=model,
        processor=processor,
        spec=MODEL_SPECS[family],
        device="cpu",
        dtype=torch.bfloat16,
        attention=attention,
    )


@pytest.mark.parametrize(
    ("family", "sampled", "temperature", "max_tokens"),
    [
        ("timechat", False, 0.0, 9_216),
        ("qwen3_omni_captioner", True, 0.6, 2_048),
    ],
)
def test_loader_generation_config_has_family_defaults_and_stop_ids(
    tmp_path: Path,
    family: str,
    sampled: bool,
    temperature: float,
    max_tokens: int,
) -> None:
    model = SimpleNamespace(generation_config=SimpleNamespace(), config=SimpleNamespace())
    processor = SimpleNamespace(tokenizer=_Tokenizer())
    _load_generation_config(model, tmp_path, processor, MODEL_SPECS[family])
    assert model.generation_config.eos_token_id == [101, 202]
    assert model.generation_config.pad_token_id == 303
    assert model.generation_config.do_sample is sampled
    assert model.generation_config.temperature == temperature
    assert model.generation_config.max_new_tokens == max_tokens


def test_stop_token_resolution_uses_documented_fallbacks() -> None:
    processor = SimpleNamespace(tokenizer=SimpleNamespace(
        pad_token_id=None,
        unk_token_id=None,
        convert_tokens_to_ids=lambda _token: None,
    ))
    assert resolve_stop_token_ids(processor) == ([151_645, 151_643], 151_643)


def test_transformers_generation_passes_eos_and_reports_finish_reason() -> None:
    processor = _Processor()
    model = _Model()
    messages: list[str] = []
    captioner = Qwen3OmniInstructCaptioner(_loaded("qwen3_omni_instruct", processor, model))
    result = captioner.caption(
        MediaInput(kind="text", text="Describe a storm."),
        gen=GenParams(max_new_tokens=32, no_repeat_ngram_size=4),
        cb=Callbacks(progress=lambda message, _payload=None: messages.append(str(message))),
    )
    call = model.calls[-1]
    assert call["no_repeat_ngram_size"] == 4
    assert call["eos_token_id"] == [101, 202]
    assert call["pad_token_id"] == 303
    assert result.usage["finish_reason"] == "eos"
    assert result.timing["generated_tokens"] == 1
    assert result.timing["tokens_per_s"] >= 0
    assert any("Generation finished: 1 new tokens" in message for message in messages)


def test_captioner_ignores_prompt_and_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    processor = _Processor("Detailed audio caption.")
    model = _Model()
    monkeypatch.setattr(
        "vcap.models.omni_common.probe_media",
        lambda _path: SimpleNamespace(
            kind="audio",
            duration=1.0,
            has_audio=True,
            has_video=False,
            error=None,
        ),
    )
    monkeypatch.setattr(
        "vcap.models.omni_common.read_audio",
        lambda _path: np.zeros(16_000, dtype=np.float32),
    )
    messages: list[str] = []
    captioner = Qwen3OmniAudioCaptioner(_loaded("qwen3_omni_captioner", processor, model))
    result = captioner.caption(
        MediaInput(path="ignored.wav", kind="audio"),
        PromptSpec(system_prompt="custom system", user_prompt="custom user"),
        gen=GenParams(
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            max_new_tokens=2_048,
            do_sample=True,
        ),
        cb=Callbacks(progress=lambda message, _payload=None: messages.append(str(message))),
    )
    conversation = processor.template_calls[-1][0]
    assert [part["type"] for part in conversation[-1]["content"]] == ["audio"]
    assert result.usage.finish_reason == "eos"
    assert any("prompt-free; ignoring" in message for message in messages)


def test_thinking_template_defaults_and_reasoning_split() -> None:
    processor = _Processor("<think>inspect the image</think>A clear final answer.")
    model = _Model()
    captioner = Qwen3OmniThinkingCaptioner(
        _loaded("qwen3_omni_thinking", processor, model)
    )
    result = captioner.caption(
        MediaInput(kind="text", text="What is visible?"),
        gen=GenParams(
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            max_new_tokens=2_048,
            do_sample=True,
            enable_thinking=True,
        ),
    )
    template_kwargs = processor.template_calls[-1][1]
    generation_kwargs = model.calls[-1]
    assert "no_repeat_ngram_size" not in generation_kwargs
    assert template_kwargs["enable_thinking"] is True
    assert generation_kwargs["temperature"] == 0.6
    assert generation_kwargs["top_p"] == 0.95
    assert generation_kwargs["top_k"] == 20
    assert result.reasoning == "inspect the image"
    assert result.text == "A clear final answer."


def test_flash_first_forward_retries_with_sdpa() -> None:
    processor = _Processor()
    model = _Model(fail_flash_once=True)
    loaded = _loaded("qwen3_omni_instruct", processor, model, "flash_attention_2")
    result = Qwen3OmniInstructCaptioner(loaded).caption(
        MediaInput(kind="text", text="Describe this."),
        gen=GenParams(max_new_tokens=8),
    )
    assert len(model.calls) == 2
    assert model.switched_to == "sdpa"
    assert loaded.attention == "sdpa"
    assert result.usage.finish_reason == "eos"
