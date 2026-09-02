from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from vcap.models.base import GenParams
from vcap.models.llamacpp_backend import LlamaCppCaptioner
from vcap.pipeline.job import InputItem, JobSpec, OutputSpec


def _backend(tmp_path: Path, spec: JobSpec) -> LlamaCppCaptioner:
    return LlamaCppCaptioner(
        "qwen3_omni_instruct",
        variant_key="qwen3_omni_instruct_gguf_q4",
        server_path=tmp_path / "llama-server",
        model_dir=tmp_path,
        runtime=spec,
    )


def test_payload_is_built_from_runtime_spec_without_a_server(tmp_path: Path) -> None:
    spec = JobSpec.from_settings(
        {
            "model_key": "qwen3_omni_instruct_gguf_q4",
            "temperature": 0.7,
            "do_sample": True,
            "gguf_min_p": 0.12,
            "gguf_repeat_last_n": 333,
            "gguf_presence_penalty": 0.45,
            "gguf_frequency_penalty": -0.25,
        },
        [InputItem("hello", kind="text", text_prompt_only=True)],
        OutputSpec(outputs_root=tmp_path),
    )
    backend = _backend(tmp_path, spec)
    generation = GenParams(**asdict(spec.generation))
    payload = backend._payload([{"role": "user", "content": "hello"}], generation)

    assert payload["min_p"] == 0.12
    assert payload["repeat_last_n"] == 333
    assert payload["presence_penalty"] == 0.45
    assert payload["frequency_penalty"] == -0.25


def test_greedy_payload_omits_sampling_only_min_p_and_zero_penalties(tmp_path: Path) -> None:
    spec = JobSpec.from_settings(
        {
            "model_key": "qwen3_omni_instruct_gguf_q4",
            "gguf_min_p": 0.9,
            "gguf_repeat_last_n": 0,
            "gguf_presence_penalty": 0,
            "gguf_frequency_penalty": 0,
        },
        [],
        OutputSpec(outputs_root=tmp_path),
    )
    payload = _backend(tmp_path, spec)._payload(
        [{"role": "user", "content": "hello"}],
        GenParams(do_sample=False, temperature=0.7),
    )

    assert "min_p" not in payload
    assert payload["repeat_last_n"] == 0
    assert "presence_penalty" not in payload
    assert "frequency_penalty" not in payload
