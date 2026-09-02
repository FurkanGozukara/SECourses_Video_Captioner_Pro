from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from vcap.models.base import Callbacks, GenParams
from vcap.models.llamacpp_backend import (
    LlamaCppCaptioner,
    frame_budget_for_video,
    server_plan_for_vram,
)


def _backend(tmp_path: Path, **kwargs) -> LlamaCppCaptioner:
    return LlamaCppCaptioner(
        "qwen3_omni_instruct",
        variant_key="qwen3_omni_instruct_gguf_q4",
        server_path=tmp_path / "llama-server.exe",
        model_dir=tmp_path,
        **kwargs,
    )


def test_runtime_server_flags_and_extra_args_are_appended_last(tmp_path: Path) -> None:
    runtime = SimpleNamespace(
        gguf_threads=7,
        gguf_batch_size=1024,
        gguf_ubatch_size=256,
        gguf_flash_attn="off",
        gguf_cache_reuse=128,
        gguf_extra_args='--threads-http 3 --alias "caption model"',
        gguf_max_frames=24,
        gguf_jpeg_quality=82,
        gguf_ignore_tier_context=False,
    )
    backend = _backend(tmp_path, runtime=SimpleNamespace(runtime=runtime))
    command = backend._server_command(tmp_path / "llama-server.exe", 12345)

    assert "--no-webui" in command
    assert command[command.index("-np") + 1] == "1"
    assert command[command.index("--threads") + 1] == "7"
    assert command[command.index("-b") + 1] == "1024"
    assert command[command.index("-ub") + 1] == "256"
    assert command[command.index("-fa") + 1] == "off"
    assert command[command.index("--cache-reuse") + 1] == "128"
    assert command[-4:] == ["--threads-http", "3", "--alias", "caption model"]
    assert backend.runtime_options.max_frames == 24
    assert backend.runtime_options.jpeg_quality == 82


def test_frame_budget_applies_each_cap_in_order() -> None:
    budget = frame_budget_for_video(
        duration_s=20.0,
        fps=2.0,
        sampling_strategy="keyframe",
        max_frames=30,
        gguf_max_frames=20,
        context_size=4096,
        max_new_tokens=1024,
        max_pixels=256 * 32 * 32,
        audio_tokens=260,
    )

    assert budget.requested_frames == 40
    assert budget.selected_frames == 8
    assert budget.per_frame_tokens == 256
    assert budget.context_frame_cap == 8
    assert "Maximum frames" in budget.messages[0]
    assert "GGUF maximum frames" in budget.messages[1]
    assert "context budget" in budget.messages[2]
    assert "keyframe" in budget.messages[-1]


def test_frame_budget_has_a_one_frame_floor() -> None:
    budget = frame_budget_for_video(
        duration_s=0.01,
        fps=0.01,
        sampling_strategy="uniform",
        max_frames=0,
        gguf_max_frames=1,
        context_size=512,
        max_new_tokens=2048,
        max_pixels=32 * 32,
        audio_tokens=1000,
    )
    assert budget.requested_frames == 1
    assert budget.selected_frames == 1
    assert budget.context_frame_cap == 1


def test_tier_context_clamp_can_be_bypassed(tmp_path: Path) -> None:
    clamped = server_plan_for_vram(12.0, requested_context=32_768)
    bypassed = server_plan_for_vram(
        12.0,
        requested_context=32_768,
        ignore_tier_context=True,
    )
    assert clamped.context_size == 8192
    assert clamped.context_was_clamped is True
    assert bypassed.context_size == 32_768
    assert bypassed.context_tier_ignored is True

    runtime = SimpleNamespace(gguf_ignore_tier_context=True)
    backend = _backend(
        tmp_path,
        runtime=SimpleNamespace(runtime=runtime),
        vram_total_gb=12.0,
        context_size=32_768,
    )
    command = backend._server_command(tmp_path / "llama-server.exe", 12345)
    assert command[command.index("-c") + 1] == "32768"


def test_sampling_seed_is_sent_and_greedy_omits_it(monkeypatch, tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    messages = [{"role": "user", "content": "hello"}]
    fixed = backend._payload(messages, GenParams(do_sample=True, temperature=0.7, seed=1234))
    monkeypatch.setattr("vcap.models.llamacpp_backend.secrets.randbits", lambda bits: 4_000_000_001)
    random_seed = backend._payload(
        messages,
        GenParams(do_sample=True, temperature=0.7, seed=-1),
    )
    greedy = backend._payload(messages, GenParams(do_sample=False, seed=99))

    assert fixed["seed"] == 1234
    assert random_seed["seed"] == 4_000_000_001
    assert "seed" not in greedy
    json.dumps(fixed)


def test_sse_uses_transport_chunks_and_throttles_progress(monkeypatch, tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    backend._base_url = "http://127.0.0.1:12345"
    seen_chunk_sizes: list[object] = []

    class Response:
        status_code = 200
        text = ""

        def iter_lines(self, chunk_size=1):
            seen_chunk_sizes.append(chunk_size)
            for _ in range(40):
                yield b'data: {"choices":[{"delta":{"content":"x"}}]}'
            yield (
                b'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":40},'
                b'"timings":{"prompt_ms":10,"predicted_ms":100,"predicted_per_second":400}}'
            )
            yield b"data: [DONE]"

        def close(self) -> None:
            return None

    class Session:
        def post(self, *_args, **_kwargs):
            return Response()

    backend._session = Session()
    monkeypatch.setattr(
        "vcap.models.llamacpp_backend.resource_snapshot",
        lambda _index: {"vram_used_gb": 1.0},
    )
    monkeypatch.setattr("vcap.core.progress.time.monotonic", lambda: 0.0)
    updates: list[dict] = []
    result = backend._stream_request(
        {"messages": [], "stream": True, "seed": 77},
        Callbacks(progress=lambda _message, payload: updates.append(dict(payload))),
    )

    assert seen_chunk_sizes == [None]
    assert result.completion_tokens == 40
    assert result.seed == 77
    assert len(updates) == 1
    assert updates[0]["new_tokens"] == 1
