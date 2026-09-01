from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import socket
import threading
import time
from types import SimpleNamespace

import numpy as np

from vcap.core.media import VideoFrames
from vcap.core.subprocess_runner import CancelToken
from vcap.models import llamacpp_backend
from vcap.models.base import Callbacks, GenParams, MediaInput, PreprocessParams
from vcap.models.llamacpp_backend import (
    LlamaCppCaptioner,
    find_free_port,
    parse_sse_events,
    server_plan_for_vram,
)
from vcap.models.registry import get_variant
from vcap.models.vram_presets import allowed_variants


def _backend(tmp_path: Path, family: str = "qwen3_omni_instruct") -> LlamaCppCaptioner:
    return LlamaCppCaptioner(
        family,
        variant_key=f"{family}_gguf_q4",
        server_path=tmp_path / "llama-server.exe",
        model_dir=tmp_path,
    )


def test_registry_gguf_repos_files_sizes_and_vram_tiers() -> None:
    instruct = get_variant("qwen3_omni_instruct_gguf_q4")
    assert instruct.gguf_repo == "ggml-org/Qwen3-Omni-30B-A3B-Instruct-GGUF"
    assert instruct.gguf_files == (
        "Qwen3-Omni-30B-A3B-Instruct-Q4_K_M.gguf",
        "mmproj-Qwen3-Omni-30B-A3B-Instruct-Q8_0.gguf",
    )
    assert instruct.gguf_file_sizes == (18_557_053_952, 1_325_020_128)
    captioner = get_variant("qwen3_omni_captioner_gguf_q8")
    assert captioner.gguf_repo == "mradermacher/Qwen3-Omni-30B-A3B-Captioner-GGUF"
    assert captioner.gguf_files == (
        "Qwen3-Omni-30B-A3B-Captioner.Q8_0.gguf",
        "Qwen3-Omni-30B-A3B-Captioner.mmproj-Q8_0.gguf",
    )
    assert captioner.gguf_file_sizes == (32_484_494_272, 1_325_020_384)
    assert get_variant("qwen3_omni_instruct_gguf_q4_k_m") is instruct
    tier_24 = allowed_variants("qwen3_omni_instruct", 24)
    tier_32 = allowed_variants("qwen3_omni_instruct", 32)
    assert "qwen3_omni_instruct_gguf_q4" in tier_24
    assert "qwen3_omni_instruct_gguf_q8" not in tier_24
    assert "qwen3_omni_instruct_gguf_q8" in tier_32


def test_gguf_download_falls_back_to_mirror_and_flattens_file(monkeypatch, tmp_path: Path) -> None:
    variant = get_variant("qwen3_omni_captioner_gguf_q4")
    filename = variant.gguf_files[0]
    payload = b"byte-identical mirror fixture"
    folder = tmp_path / variant.folder_name
    folder.mkdir()
    calls: list[tuple[str, str]] = []

    def fake_hf_hub_download(**kwargs):
        calls.append((kwargs["repo_id"], kwargs["filename"]))
        if len(calls) == 1:
            raise OSError("primary unavailable")
        downloaded = Path(kwargs["local_dir"]) / Path(kwargs["filename"])
        downloaded.parent.mkdir(parents=True, exist_ok=True)
        downloaded.write_bytes(payload)
        return str(downloaded)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)

    downloaded = llamacpp_backend._hf_download(
        variant,
        filename,
        folder,
        len(payload),
        hashlib.sha256(payload).hexdigest(),
        None,
        None,
    )
    target = folder / filename

    assert calls == [
        (variant.gguf_repo, filename),
        ("MonsterMMORPG/Wan_GGUF", f"{variant.folder_name}/{filename}"),
    ]
    assert downloaded == target.resolve()
    assert target.read_bytes() == payload
    assert not (folder / variant.folder_name / filename).exists()


def test_message_building_audio_uses_base64_wav(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "clip.mp3"
    source.write_bytes(b"stub")
    backend = _backend(tmp_path)
    monkeypatch.setattr(
        "vcap.models.llamacpp_backend.probe_media",
        lambda _path: SimpleNamespace(kind="audio", duration=1.0, has_audio=True, has_video=False, error=None),
    )
    monkeypatch.setattr(
        "vcap.models.llamacpp_backend.read_audio",
        lambda _path: np.zeros(16_000, dtype=np.float32),
    )
    prepared = backend.build_messages(MediaInput(path=source, kind="audio"), "Describe every audible event.")
    user = prepared.messages[-1]
    assert user["role"] == "user"
    assert [part["type"] for part in user["content"]] == ["input_audio", "text"]
    encoded = user["content"][0]["input_audio"]["data"]
    assert base64.b64decode(encoded).startswith(b"RIFF")
    assert user["content"][0]["input_audio"]["format"] == "wav"


def test_video_building_uses_bounded_frames_and_separate_audio(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"stub")
    backend = _backend(tmp_path)
    monkeypatch.setattr(
        "vcap.models.llamacpp_backend.probe_media",
        lambda _path: SimpleNamespace(kind="video", duration=10.0, has_audio=True, has_video=True, error=None),
    )

    frame_options = {}

    def fake_frames(_path, **kwargs):
        frame_options.update(kwargs)
        count = kwargs["num_frames"]
        return VideoFrames(
            frames=np.zeros((count, 32, 32, 3), dtype=np.uint8),
            timestamps=[float(index) for index in range(count)],
            fps_effective=1.0,
            orig_size=(32, 32),
            resized_size=(32, 32),
            total_frames=300,
        )

    monkeypatch.setattr("vcap.models.llamacpp_backend.read_video_frames", fake_frames)
    monkeypatch.setattr(
        "vcap.models.llamacpp_backend.read_audio",
        lambda _path: np.zeros(160_000, dtype=np.float32),
    )
    prepared = backend.build_messages(
        MediaInput(path=source),
        pre=PreprocessParams(
            max_frames=16,
            sampling_strategy="keyframe",
            max_pixels=32 * 32,
            min_pixels=32 * 32,
        ),
    )
    types = [part["type"] for part in prepared.messages[-1]["content"]]
    assert types.count("image_url") == 8
    assert types.count("input_audio") == 1
    assert types[-1] == "text"
    assert prepared.warnings and "not interleaved" in prepared.warnings[0]
    assert frame_options["sampling"] == "keyframe"
    assert frame_options["target_fps"] == 2.0


def test_captioner_gguf_ignores_supplied_prompt(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "clip.wav"
    source.write_bytes(b"stub")
    backend = _backend(tmp_path, "qwen3_omni_captioner")
    monkeypatch.setattr(
        "vcap.models.llamacpp_backend.probe_media",
        lambda _path: SimpleNamespace(
            kind="audio", duration=1.0, has_audio=True, has_video=False, error=None
        ),
    )
    monkeypatch.setattr(
        "vcap.models.llamacpp_backend.read_audio",
        lambda _path: np.zeros(16_000, dtype=np.float32),
    )
    prepared = backend.build_messages(MediaInput(path=source, kind="audio"), "ignore me")
    assert [part["type"] for part in prepared.messages[-1]["content"]] == ["input_audio"]
    assert any("prompt-free; ignoring" in warning for warning in prepared.warnings)


def test_sse_parser_and_mocked_streaming_response(monkeypatch, tmp_path: Path) -> None:
    events = list(
        parse_sse_events(
            [
                b": keep-alive",
                b'data: {"choices":[{"delta":{"content":"storm"}}]}',
                b'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":2}}',
                b"data: [DONE]",
            ]
        )
    )
    assert events[0]["choices"][0]["delta"]["content"] == "storm"
    assert events[1]["usage"]["completion_tokens"] == 2

    backend = _backend(tmp_path)
    backend._base_url = "http://127.0.0.1:12345"

    class FakeResponse:
        status_code = 200
        text = ""

        def iter_lines(self, chunk_size=1):
            del chunk_size
            yield b'data: {"choices":[{"delta":{"content":"Lightning "}}]}'
            yield b'data: {"choices":[{"delta":{"content":"strikes."}}]}'
            yield (
                b'data: {"choices":[],"usage":{"prompt_tokens":21,"completion_tokens":4},'
                b'"timings":{"prompt_ms":500,"predicted_ms":1000,"predicted_per_second":4.0}}'
            )
            yield b"data: [DONE]"

        def close(self):
            pass

    class FakeSession:
        def post(self, *args, **kwargs):
            assert args[0].endswith("/v1/chat/completions")
            assert kwargs["json"]["stream"] is True
            return FakeResponse()

    backend._session = FakeSession()
    monkeypatch.setattr(
        "vcap.models.llamacpp_backend.resource_snapshot",
        lambda _index: {"vram_used_gb": 20.5},
    )
    result = backend._stream_request(
        {"messages": [], "stream": True},
        Callbacks(),
    )
    assert result.content == "Lightning strikes."
    assert result.prompt_tokens == 21
    assert result.completion_tokens == 4
    assert result.prefill_s == 0.5
    assert result.decode_s == 1.0
    assert result.tok_per_s == 4.0
    assert result.peak_vram_gb == 20.5


def test_streaming_cancel_closes_inflight_sse_and_stops_server(monkeypatch, tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    backend._base_url = "http://127.0.0.1:12345"
    response_closed = threading.Event()
    server_stopped = threading.Event()

    class BlockingResponse:
        status_code = 200
        text = ""

        def iter_lines(self, chunk_size=1):
            del chunk_size
            yield b'data: {"choices":[{"delta":{"content":"partial"}}]}'
            response_closed.wait(5)

        def close(self):
            response_closed.set()

    class FakeSession:
        def post(self, *args, **kwargs):
            del args, kwargs
            return BlockingResponse()

    backend._session = FakeSession()
    monkeypatch.setattr(backend, "stop", lambda: server_stopped.set())
    monkeypatch.setattr(
        "vcap.models.llamacpp_backend.resource_snapshot",
        lambda _index: {"vram_used_gb": 20.0},
    )
    token = CancelToken()
    timer = threading.Timer(0.05, token.cancel)
    timer.start()
    started = time.monotonic()
    try:
        result = backend._stream_request(
            {"messages": [], "stream": True},
            Callbacks(cancel=token),
        )
    finally:
        timer.cancel()
    assert result.cancelled
    assert response_closed.is_set() and server_stopped.is_set()
    assert time.monotonic() - started < 2.0


def test_free_port_and_server_command_contains_no_quantized_kv_flags(tmp_path: Path) -> None:
    port = find_free_port()
    assert 0 < port < 65_536
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", port))
    backend = _backend(tmp_path)
    command = backend._server_command(tmp_path / "llama-server.exe", port)
    assert command[:3] == [str(tmp_path / "llama-server.exe"), "--model", str(tmp_path / backend.variant.gguf_files[0])]
    assert "--mmproj" in command
    assert "--jinja" in command
    assert command[command.index("-c") + 1] == "32768"
    # -ngl must stay unset: llama.cpp aborts fitting when n_gpu_layers is user-set.
    assert "-ngl" not in command
    assert command[command.index("--fit") + 1] == "on"
    # 2 GB reserve + 1,536 MiB multimodal-projector headroom
    assert command[command.index("--fit-target") + 1] == "3584"
    assert command[command.index("--device") + 1] == "CUDA0"
    assert not {"-ctk", "-ctv", "--cache-type-k", "--cache-type-v"}.intersection(command)


def test_server_plan_keeps_tier_contexts_and_delegates_offload_to_fit(tmp_path: Path) -> None:
    assert server_plan_for_vram(24.0).context_size == 16_384
    assert server_plan_for_vram(24.0).gpu_layers is None
    assert server_plan_for_vram(32.0).context_size == 32_768
    assert server_plan_for_vram(32.0).gpu_layers is None
    assert server_plan_for_vram(32.0, q8_weights=True).gpu_layers is None
    assert server_plan_for_vram(32.0).fit is True
    assert server_plan_for_vram(32.0).fit_target_mib == 3_584
    assert server_plan_for_vram(32.0).n_cpu_moe is None

    backend = LlamaCppCaptioner(
        "qwen3_omni_instruct",
        variant_key="qwen3_omni_instruct_gguf_q4",
        server_path=tmp_path / "llama-server.exe",
        model_dir=tmp_path,
        device_index=2,
        gpu_index=2,
        vram_total_gb=24.0,
        vram_reserve_gb=3.5,
    )
    command = backend._server_command(tmp_path / "llama-server.exe", 12345)
    assert command[command.index("-c") + 1] == "16384"
    assert "-ngl" not in command
    assert command[command.index("--fit") + 1] == "on"
    assert command[command.index("--fit-target") + 1] == "5120"
    assert command[command.index("--device") + 1] == "CUDA2"
    assert command[command.index("--main-gpu") + 1] == "2"


def test_server_fit_environment_overrides(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VCAP_LLAMACPP_GPU_LAYERS", "17")
    monkeypatch.setenv("VCAP_LLAMACPP_CONTEXT_SIZE", "12288")
    monkeypatch.setenv("VCAP_LLAMACPP_FIT_TARGET_MIB", "3072")
    monkeypatch.setenv("VCAP_LLAMACPP_FIT", "0")
    monkeypatch.setenv("VCAP_LLAMACPP_N_CPU_MOE", "11")

    backend = _backend(tmp_path)
    command = backend._server_command(tmp_path / "llama-server.exe", 12345)

    assert command[command.index("-ngl") + 1] == "17"
    assert command[command.index("-c") + 1] == "12288"
    assert command[command.index("--fit") + 1] == "off"
    assert command[command.index("--fit-target") + 1] == "3072"
    assert command[command.index("--n-cpu-moe") + 1] == "11"
    assert backend.block_swap_summary()["fit"] is False


def test_fit_report_parses_final_server_placement_and_is_json_safe(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    backend._server_command(tmp_path / "llama-server.exe", 12345)
    with backend._log_lock:
        backend._log_tail.extend(
            [
                "common_params_fit_impl: n_gpu_layers = 35",
                "tensor blk.0.ffn_gate_exps.weight (512 MiB) buffer type overridden to CPU",
                "tensor blk.1.ffn_up_exps.weight (512 MiB) buffer type overridden to CPU",
                "load_tensors: offloaded 37/49 layers to GPU",
                "llama_context: n_ctx = 16384",
                "W common_fit_params: failed to fit params to free device memory: example, abort",
            ]
        )

    report = backend.block_swap_summary()
    assert report["fit_error"] == "failed to fit params to free device memory: example, abort"

    assert report["backend"] == "llamacpp"
    assert report["mode"] == "fit"
    assert report["fit_target_mib"] == 3_584
    assert report["requested_gpu_layers"] is None
    assert report["n_gpu_layers"] == 37
    assert report["n_gpu_layers_total"] == 49
    assert report["n_cpu_moe"] == 2
    assert report["n_ctx"] == 16_384
    assert backend.context_size == 16_384
    assert json.loads(json.dumps(report)) == report


def test_generation_payload_maps_sampling_controls(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    payload = backend._payload(
        [{"role": "user", "content": "hello"}],
        GenParams(temperature=0.7, top_p=0.9, top_k=33, max_new_tokens=123, do_sample=True),
    )
    assert payload["temperature"] == 0.7
    assert payload["top_p"] == 0.9
    assert payload["top_k"] == 33
    assert payload["max_tokens"] == 123
    assert payload["stream"] is True
    json.dumps(payload)
