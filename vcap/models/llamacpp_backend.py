"""llama.cpp server backend for multimodal Qwen3-Omni GGUF models."""

from __future__ import annotations

import atexit
import base64
from collections import deque
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import queue
import socket
import subprocess
import threading
import time
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence
from urllib.parse import quote
import wave
import weakref

import numpy as np
from PIL import Image, ImageOps
import requests

from vcap.core import console_progress
from vcap.core.gpu import resource_snapshot, vram_tier_for_gb
from vcap.core.logs import get_log
from vcap.core.media import probe_media, read_audio, read_video_frames
from vcap.core.subprocess_runner import CancelledError, build_child_env, kill_process_tree
from vcap.prompts.postprocess import POST_PROCESSORS, PostResult, plain
from vcap.prompts.presets import PromptPreset, get_preset, render_prompt

from .base import (
    BaseCaptioner,
    Callbacks,
    CaptionResult,
    CaptionTiming,
    ChatMessage,
    ChatResult,
    GenParams,
    MediaInput,
    MediaPart,
    PreprocessParams,
    PromptSpec,
    TokenUsage,
    normalize_chat_history,
    truncate_chat_history,
)
from .llamacpp_install import download_resumable, ensure_llamacpp
from .omni_common import split_thinking
from .registry import (
    MODEL_SPECS,
    VariantSpec,
    get_variant,
    resolve_model_dir,
    variant_is_ready,
    variant_to_family,
)


ProgressCallback = Callable[..., None]
HF_RESOLVE_ROOT = "https://huggingface.co"
DEFAULT_CONTEXT_SIZE = 32_768
DEFAULT_STARTUP_TIMEOUT_S = 900.0


@dataclass(frozen=True)
class LlamaCppServerPlan:
    """Tier-aware llama-server context and GPU-layer limits."""

    vram_tier_gb: int
    context_size: int
    gpu_layers: int


def server_plan_for_vram(
    total_vram_gb: float,
    *,
    requested_context: int = DEFAULT_CONTEXT_SIZE,
    q8_weights: bool = False,
) -> LlamaCppServerPlan:
    """Return a conservative Qwen3-Omni server plan for one physical GPU."""

    observed = float(total_vram_gb or 32.0)
    tier = vram_tier_for_gb(observed)
    requested = max(4_096, int(requested_context))
    if tier <= 16:
        return LlamaCppServerPlan(tier, min(requested, 8_192), 20)
    if tier <= 24:
        # Q4_K_M + mmproj leaves too little headroom for a fully offloaded 32K
        # context on a 24 GB card. This plan stays below that measured peak.
        return LlamaCppServerPlan(tier, min(requested, 16_384), 36)
    gpu_layers = 36 if q8_weights and tier < 48 else 999
    return LlamaCppServerPlan(tier, min(requested, 32_768), gpu_layers)


def _cancelled(token: object | None) -> bool:
    if token is None:
        return False
    for name in ("is_cancelled", "is_set"):
        method = getattr(token, name, None)
        if callable(method):
            try:
                return bool(method())
            except Exception:
                return False
    return bool(getattr(token, "cancelled", False))


def _emit(callback: ProgressCallback | None, message: str, **data: Any) -> None:
    get_log().log(message, scope="llama.cpp")
    if callback is None:
        return
    payload = {"message": message, **data}
    for args in ((message, payload), (payload,), (message,)):
        try:
            callback(*args)
            return
        except TypeError:
            continue


def _callback(callback: ProgressCallback | None, message: str, **data: Any) -> None:
    if callback is None:
        return
    payload = {"message": message, **data}
    for args in ((message, payload), (payload,), (message,)):
        try:
            callback(*args)
            return
        except TypeError:
            continue


def _delta_callback(callback: ProgressCallback | None, delta: str = "", reasoning_delta: str = "") -> None:
    if callback is None or (not delta and not reasoning_delta):
        return
    payload = {"delta": delta, "reasoning_delta": reasoning_delta}
    for args in ((delta, payload), (payload,), (delta,)):
        try:
            callback(*args)
            return
        except TypeError:
            continue


def build_llamacpp_chat_messages(
    history: Sequence[ChatMessage | Mapping[str, Any]],
    media_content: Sequence[Mapping[str, Any]] = (),
    system_prompt: str | None = None,
) -> list[dict[str, Any]]:
    """Build OpenAI-compatible messages with media in the first user turn."""

    normalized = normalize_chat_history(history)
    inherited_system = next((item.content for item in normalized if item.role == "system"), "")
    selected_system = str(system_prompt) if system_prompt is not None else inherited_system
    text_messages = [item for item in normalized if item.role != "system"]
    if not text_messages or text_messages[-1].role != "user":
        raise ValueError("Chat history must end with the current user message")
    first_user = next((index for index, item in enumerate(text_messages) if item.role == "user"), None)
    if first_user is None:
        raise ValueError("Chat history needs at least one user message")
    result: list[dict[str, Any]] = []
    if selected_system.strip():
        result.append({"role": "system", "content": selected_system.strip()})
    for index, message in enumerate(text_messages):
        if index == first_user and media_content:
            content = [dict(item) for item in media_content]
            if message.content:
                content.append({"type": "text", "text": message.content})
            result.append({"role": message.role, "content": content})
        else:
            if not message.content:
                raise ValueError(f"The {message.role} message is empty")
            result.append({"role": message.role, "content": message.content})
    return result


def find_free_port(host: str = "127.0.0.1") -> int:
    """Return an ephemeral TCP port selected by the operating system."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, 0))
        return int(listener.getsockname()[1])


def parse_sse_events(lines: Iterable[bytes | str]) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from an OpenAI-compatible SSE byte stream."""

    for raw in lines:
        text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
        text = text.strip()
        if not text or text.startswith(":") or not text.startswith("data:"):
            continue
        data = text[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            event = json.loads(data)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid SSE JSON from llama-server: {data[:500]}") from exc
        if isinstance(event, dict):
            yield event


def _verify_file(
    path: Path,
    expected_sha256: str | None,
    progress_cb: ProgressCallback | None,
    cancel: object | None,
) -> None:
    if not expected_sha256:
        return
    expected = expected_sha256.removeprefix("sha256:").casefold()
    digest = hashlib.sha256()
    total = path.stat().st_size
    done = 0
    last_report = time.monotonic()
    _emit(progress_cb, f"Verifying SHA-256: {path.name}")
    with path.open("rb") as handle:
        while True:
            if _cancelled(cancel):
                raise CancelledError(f"verification cancelled: {path.name}")
            chunk = handle.read(16 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            done += len(chunk)
            now = time.monotonic()
            if now - last_report >= 2.0:
                _emit(
                    progress_cb,
                    f"Verifying {path.name}: {done * 100 / max(total, 1):.1f}%",
                    fraction=done / max(total, 1),
                    bytes_done=done,
                    bytes_total=total,
                )
                last_report = now
    actual = digest.hexdigest()
    if actual != expected:
        raise RuntimeError(f"SHA-256 mismatch for {path.name}: got {actual}, expected {expected}")


def _hf_tqdm_class(
    progress_cb: ProgressCallback | None,
    cancel: object | None,
    label: str,
) -> type[Any]:
    from tqdm.auto import tqdm

    class CallbackTqdm(tqdm):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._vcap_last_report = 0.0
            super().__init__(*args, **kwargs)

        def update(self, n: int | float = 1) -> bool | None:
            if _cancelled(cancel):
                raise CancelledError(f"download cancelled: {label}")
            result = super().update(n)
            now = time.monotonic()
            if now - self._vcap_last_report >= 2.0:
                total = int(self.total or 0)
                done = int(self.n)
                fraction = done / total if total else None
                _emit(
                    progress_cb,
                    f"Downloading {label}: {done / 1e9:.2f}"
                    + (f" / {total / 1e9:.2f} GB ({done * 100 / total:.1f}%)" if total else " GB"),
                    fraction=fraction,
                    bytes_done=done,
                    bytes_total=total or None,
                )
                self._vcap_last_report = now
            return result

    return CallbackTqdm


def _hf_download(
    variant: VariantSpec,
    filename: str,
    folder: Path,
    expected_size: int | None,
    expected_sha256: str | None,
    progress_cb: ProgressCallback | None,
    cancel: object | None,
) -> Path:
    label = f"{variant.gguf_repo}/{filename}"
    target = folder / filename
    _emit(progress_cb, f"Downloading {label} with Hugging Face resumable cache")
    try:
        from huggingface_hub import hf_hub_download

        downloaded = Path(
            hf_hub_download(
                repo_id=str(variant.gguf_repo),
                filename=filename,
                revision="main",
                local_dir=folder,
                user_agent="SECourses-Video-Captioner-Pro/1.0",
                force_download=bool(target.is_file() and expected_size and target.stat().st_size != expected_size),
                tqdm_class=_hf_tqdm_class(progress_cb, cancel, label),
            )
        ).resolve(strict=True)
    except ImportError:
        encoded_name = quote(filename, safe="/")
        url = f"{HF_RESOLVE_ROOT}/{variant.gguf_repo}/resolve/main/{encoded_name}?download=true"
        downloaded = download_resumable(
            url,
            target,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
            progress_cb=progress_cb,
            cancel=cancel,
            label=label,
        )
        return downloaded
    if expected_size is not None and downloaded.stat().st_size != expected_size:
        raise RuntimeError(
            f"Incomplete download for {filename}: got {downloaded.stat().st_size}, expected {expected_size} bytes"
        )
    _verify_file(downloaded, expected_sha256, progress_cb, cancel)
    target.with_name(target.name + ".part").unlink(missing_ok=True)
    _emit(progress_cb, f"Downloaded {label}", fraction=1.0, bytes_total=downloaded.stat().st_size)
    return downloaded


def ensure_gguf(
    variant_key: str,
    progress_cb: ProgressCallback | None = None,
    cancel: object | None = None,
) -> Path:
    """Resumably fetch a registered non-gated GGUF and its mmproj from HF."""

    variant = get_variant(variant_key)
    if variant.scheme != "gguf" or variant.backend != "llamacpp":
        raise ValueError(f"{variant_key} is not a llama.cpp GGUF variant")
    if not variant.gguf_repo or not variant.gguf_files:
        raise RuntimeError(f"{variant.key} has incomplete GGUF registry metadata")
    ready, _ = variant_is_ready(variant.key)
    folder = resolve_model_dir(variant.key)
    if ready:
        _emit(progress_cb, f"GGUF model is ready: {folder}", fraction=1.0)
        return folder
    folder.mkdir(parents=True, exist_ok=True)
    sizes = variant.gguf_file_sizes or ()
    hashes = variant.gguf_sha256 or ()
    for index, filename in enumerate(variant.gguf_files):
        if _cancelled(cancel):
            raise CancelledError(f"GGUF download cancelled before {filename}")
        _hf_download(
            variant,
            filename,
            folder,
            sizes[index] if index < len(sizes) else None,
            hashes[index] if index < len(hashes) else None,
            progress_cb,
            cancel,
        )
    total_bytes = sum((folder / name).stat().st_size for name in variant.gguf_files)
    family = variant_to_family(variant.key)
    qwen_variant = family.removeprefix("qwen3_omni_").title()
    metadata = {
        "variant_key": variant.key,
        "backend": "llamacpp",
        "source_repo": f"Qwen/Qwen3-Omni-30B-A3B-{qwen_variant}",
        "artifact_repo": variant.gguf_repo,
        "files": list(variant.gguf_files),
        "total_bytes": total_bytes,
    }
    (folder / "vcap_model_info.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    ready, detail = variant_is_ready(variant.key)
    if not ready:
        raise RuntimeError(f"GGUF download did not produce a ready model: {detail}")
    _emit(progress_cb, f"GGUF model ready: {folder}", fraction=1.0, bytes_total=total_bytes)
    return folder


def _default_gen(family: str) -> GenParams:
    values = {item.name: item.default for item in MODEL_SPECS[family].param_schema}
    return GenParams(
        temperature=float(values.get("temperature", 0.0)),
        top_p=float(values.get("top_p", 1.0)),
        top_k=int(values.get("top_k", 0)),
        repetition_penalty=float(values.get("repetition_penalty", 1.0)),
        max_new_tokens=int(values.get("max_new_tokens", 2048)),
        do_sample=bool(values.get("do_sample", False)),
        enable_thinking=bool(values.get("enable_thinking", True)),
    )


def _default_pre(family: str) -> PreprocessParams:
    limits = MODEL_SPECS[family].limits
    return PreprocessParams(
        fps=limits.default_fps,
        max_frames=int(limits.max_frames or 768),
        max_pixels=limits.default_max_pixels,
        min_pixels=limits.min_pixels,
        use_audio_in_video="video_audio" in MODEL_SPECS[family].capabilities,
    )


def _infer_part(media: MediaInput) -> MediaPart:
    if media.kind == "text" or (media.path is None and media.text is not None):
        return MediaPart("text", media.path, media.text, media.start, media.end)
    if media.path is None:
        raise ValueError("MediaInput needs a path, text, or mixed-media parts")
    info = probe_media(media.path)
    mapping = {
        "video": "video_audio",
        "video_no_audio": "video",
        "audio": "audio",
        "image": "image",
        "text": "text",
    }
    kind = media.kind or mapping.get(info.kind)
    if kind is None:
        raise ValueError(f"Unsupported or unreadable media {media.path}: {info.error or info.kind}")
    return MediaPart(kind, media.path, media.text, media.start, media.end)


def _parts(media: MediaInput) -> list[MediaPart]:
    return list(media.parts) if media.parts else [_infer_part(media)]


def _text_for_part(part: MediaPart) -> str:
    if part.text is not None:
        return str(part.text)
    if part.path is None:
        return ""
    try:
        return Path(part.path).read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return Path(part.path).read_text(encoding="utf-8", errors="replace")


def _slice_audio(samples: np.ndarray, start: float | None, end: float | None) -> np.ndarray:
    if end is not None and float(end) <= float(start or 0.0):
        raise ValueError("end must be greater than start")
    first = max(0, int(round(float(start or 0.0) * 16_000)))
    last = len(samples) if end is None else max(first, int(round(float(end) * 16_000)))
    sliced = np.ascontiguousarray(samples[first:last], dtype=np.float32)
    if sliced.size == 0:
        raise ValueError("The selected audio interval is empty")
    return sliced


def _wav_base64(samples: np.ndarray) -> str:
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = (clipped * 32767.0).round().astype("<i2", copy=False)
    buffer = BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(pcm.tobytes())
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _image_data_url(image: Image.Image | np.ndarray) -> str:
    converted = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")
    converted = converted.convert("RGB")
    buffer = BytesIO()
    converted.save(buffer, format="JPEG", quality=90, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _native_video_part(path: Path) -> dict[str, Any]:
    suffix = path.suffix.casefold().lstrip(".")
    video_format = suffix if suffix in {"mp4", "ogg"} else "auto"
    return {
        "type": "input_video",
        "input_video": {
            "data": base64.b64encode(path.read_bytes()).decode("ascii"),
            "format": video_format,
        },
    }


@dataclass
class _PreparedMessage:
    messages: list[dict[str, Any]]
    warnings: list[str]
    preset: PromptPreset | None


@dataclass
class _StreamResult:
    content: str
    reasoning: str
    prompt_tokens: int
    completion_tokens: int
    prefill_s: float
    decode_s: float
    tok_per_s: float
    total_s: float
    peak_vram_gb: float
    cancelled: bool
    finish_reason: str


class LlamaCppCaptioner(BaseCaptioner):
    """Persistent local llama-server wrapper for one Qwen3-Omni GGUF."""

    def __init__(
        self,
        family: str = "qwen3_omni_instruct",
        loaded: Any | None = None,
        *,
        variant_key: str | None = None,
        server_path: str | os.PathLike[str] | None = None,
        model_dir: str | os.PathLike[str] | None = None,
        context_size: int = DEFAULT_CONTEXT_SIZE,
        video_mode: str | None = None,
        device_index: int = 0,
        gpu_index: int = 0,
        vram_total_gb: float | None = None,
    ) -> None:
        super().__init__(family, loaded)
        if not family.startswith("qwen3_omni_"):
            raise ValueError(f"llama.cpp backend only supports Qwen3-Omni, not {family}")
        if variant_key is None:
            variant_key = next(
                variant.key for variant in self.spec.variants if variant.backend == "llamacpp"
            )
        variant = get_variant(variant_key)
        if variant.backend != "llamacpp" or variant_to_family(variant.key) != family:
            raise ValueError(f"{variant_key} is not a llama.cpp variant for {family}")
        self.variant: VariantSpec = variant
        self.model_dir = (
            Path(model_dir).expanduser().resolve(strict=False)
            if model_dir is not None
            else resolve_model_dir(variant.key)
        )
        self.server_path = (
            Path(server_path).expanduser().resolve(strict=False)
            if server_path is not None
            else None
        )
        self.requested_context_size = max(4096, int(context_size))
        self.device_index = max(0, int(device_index))
        self.gpu_index = max(0, int(gpu_index))
        self.vram_total_gb = float(vram_total_gb or 32.0)
        self.server_plan = server_plan_for_vram(
            self.vram_total_gb,
            requested_context=self.requested_context_size,
            q8_weights=self.variant.key.endswith("_gguf_q8"),
        )
        self.context_size = self.server_plan.context_size
        selected_video_mode = (video_mode or os.environ.get("VCAP_LLAMACPP_VIDEO_MODE", "frames_audio")).strip().casefold()
        if selected_video_mode not in {"frames_audio", "native"}:
            raise ValueError("video_mode must be 'frames_audio' or 'native'")
        self.video_mode = selected_video_mode
        self._process: subprocess.Popen[str] | None = None
        self._port: int | None = None
        self._base_url: str | None = None
        self._log_thread: threading.Thread | None = None
        self._log_tail: deque[str] = deque(maxlen=120)
        self._startup_progress: ProgressCallback | None = None
        self._lifecycle_lock = threading.RLock()
        self._caption_lock = threading.Lock()
        self._session = requests.Session()
        self.load_seconds = 0.0
        self.load_peak_vram_gb = 0.0
        reference = weakref.ref(self)

        def stop_at_exit() -> None:
            backend = reference()
            if backend is not None:
                backend.stop()

        atexit.register(stop_at_exit)

    @property
    def base_url(self) -> str | None:
        return self._base_url

    @property
    def is_running(self) -> bool:
        process = self._process
        return process is not None and process.poll() is None and self._base_url is not None

    @property
    def server_log_tail(self) -> str:
        return "\n".join(self._log_tail)

    def _server_command(self, server: Path, port: int) -> list[str]:
        if not self.variant.gguf_files or len(self.variant.gguf_files) < 2:
            raise RuntimeError(f"{self.variant.key} does not define a model and mmproj")
        model = self.model_dir / self.variant.gguf_files[0]
        mmproj = self.model_dir / self.variant.gguf_files[1]
        context_size = int(
            os.environ.get("VCAP_LLAMACPP_CONTEXT_SIZE", self.server_plan.context_size)
        )
        gpu_layers = int(
            os.environ.get("VCAP_LLAMACPP_GPU_LAYERS", self.server_plan.gpu_layers)
        )
        return [
            str(server),
            "--model",
            str(model),
            "--mmproj",
            str(mmproj),
            "-c",
            str(max(4_096, context_size)),
            "--jinja",
            "-ngl",
            str(max(0, gpu_layers)),
            "--device",
            f"CUDA{self.device_index}",
            "--split-mode",
            "none",
            "--main-gpu",
            str(self.device_index),
            "--port",
            str(port),
            "--host",
            "127.0.0.1",
        ]

    def _read_server_logs(self, process: subprocess.Popen[str]) -> None:
        stream = process.stdout
        if stream is None:
            return
        try:
            for raw in iter(stream.readline, ""):
                line = raw.rstrip("\r\n")
                if not line:
                    continue
                self._log_tail.append(line)
                _emit(self._startup_progress, f"llama-server: {line}")
        finally:
            try:
                stream.close()
            except OSError:
                pass

    def start(
        self,
        progress_cb: ProgressCallback | None = None,
        cancel: object | None = None,
        *,
        timeout_s: float = DEFAULT_STARTUP_TIMEOUT_S,
    ) -> "LlamaCppCaptioner":
        """Start llama-server and wait until ``GET /health`` reports ready."""

        with self._lifecycle_lock:
            if self.is_running:
                return self
            ready, detail = variant_is_ready(self.variant.key)
            if not ready:
                raise FileNotFoundError(f"{self.variant.key} is not ready: {detail}")
            server = self.server_path or ensure_llamacpp(progress_cb, cancel)
            self.server_path = server
            port = find_free_port()
            command = self._server_command(server, port)
            _emit(progress_cb, f"Starting llama-server for {self.variant.key} on 127.0.0.1:{port}")
            creationflags = 0
            popen_options: dict[str, Any] = {}
            if os.name == "nt":
                creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "CREATE_NO_WINDOW", 0)
            else:
                popen_options["start_new_session"] = True
            started = time.perf_counter()
            baseline = float(
                resource_snapshot(self.gpu_index).get("vram_used_gb", 0.0) or 0.0
            )
            self._log_tail.clear()
            self._startup_progress = progress_cb
            try:
                process = subprocess.Popen(
                    command,
                    cwd=str(server.parent),
                    env=build_child_env(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    creationflags=creationflags,
                    **popen_options,
                )
            except OSError as exc:
                self._startup_progress = None
                raise RuntimeError(f"Could not start llama-server: {exc}") from exc
            self._process = process
            self._port = port
            self._base_url = f"http://127.0.0.1:{port}"
            self._log_thread = threading.Thread(
                target=self._read_server_logs,
                args=(process,),
                name=f"vcap-llama-server-{port}",
                daemon=True,
            )
            self._log_thread.start()

            deadline = time.monotonic() + max(10.0, float(timeout_s))
            peak = baseline
            last_health_error = ""
            try:
                while time.monotonic() < deadline:
                    peak = max(
                        peak,
                        float(
                            resource_snapshot(self.gpu_index).get("vram_used_gb", 0.0)
                            or 0.0
                        ),
                    )
                    if _cancelled(cancel):
                        self.stop()
                        raise CancelledError("llama-server startup cancelled")
                    return_code = process.poll()
                    if return_code is not None:
                        tail = self.server_log_tail or "(no server output)"
                        raise RuntimeError(
                            f"llama-server exited with code {return_code} during startup. Log tail:\n{tail}"
                        )
                    try:
                        response = self._session.get(f"{self._base_url}/health", timeout=2)
                        health_ready = response.status_code == 200
                        if not health_ready:
                            last_health_error = f"HTTP {response.status_code}: {response.text[-500:]}"
                        response.close()
                        if health_ready:
                            self.load_seconds = time.perf_counter() - started
                            self.load_peak_vram_gb = peak
                            _emit(
                                progress_cb,
                                f"llama-server ready in {self.load_seconds:.1f}s; "
                                f"peak GPU {self.gpu_index} VRAM {peak:.2f} GiB",
                                fraction=1.0,
                            )
                            return self
                    except requests.RequestException as exc:
                        last_health_error = str(exc)
                    time.sleep(0.25)
                tail = self.server_log_tail or "(no server output)"
                self.stop()
                raise RuntimeError(
                    f"llama-server did not become healthy within {timeout_s:.0f}s ({last_health_error}). "
                    f"Log tail:\n{tail}"
                )
            except Exception:
                if self._process is not None:
                    self.stop()
                raise
            finally:
                self._startup_progress = None

    def stop(self) -> None:
        """Terminate the server process tree and release the local endpoint."""

        with self._lifecycle_lock:
            process = self._process
            self._process = None
            self._base_url = None
            self._port = None
            if process is not None and process.poll() is None:
                kill_process_tree(process.pid, grace=2.0)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
            thread = self._log_thread
            self._log_thread = None
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=2.0)

    close = stop

    def _validate_parts(self, parts: list[MediaPart]) -> None:
        if self.spec.family == "qwen3_omni_captioner":
            if len(parts) != 1 or parts[0].type != "audio":
                raise ValueError("Qwen3-Omni Captioner accepts exactly one audio input and no prompt")
            return
        for part in parts:
            if part.type not in self.spec.capabilities:
                raise ValueError(f"{self.spec.label} does not support {part.type} input")

    @staticmethod
    def _bounds(part: MediaPart, pre: PreprocessParams) -> tuple[float | None, float | None]:
        return (
            part.start if part.start is not None else pre.start,
            part.end if part.end is not None else pre.end,
        )

    def _media_content(
        self,
        parts: list[MediaPart],
        pre: PreprocessParams,
        cancel: object | None,
    ) -> tuple[list[dict[str, Any]], list[str], list[str]]:
        content: list[dict[str, Any]] = []
        warnings: list[str] = []
        text_parts: list[str] = []
        for part in parts:
            if _cancelled(cancel):
                raise CancelledError("media preparation cancelled")
            start, end = self._bounds(part, pre)
            if part.type in {"video", "video_audio"}:
                if part.path is None:
                    raise ValueError("Video parts require a path")
                path = Path(part.path).expanduser().resolve(strict=True)
                info = probe_media(path)
                if not info.has_video:
                    raise ValueError(f"No video stream found in {path}")
                use_native = self.video_mode == "native" and start is None and end is None
                if use_native:
                    content.append(_native_video_part(path))
                else:
                    duration = max(0.0, float((end if end is not None else info.duration) or 0.0) - float(start or 0.0))
                    target_frames = int(math.ceil(duration * 0.75)) if duration > 0 else 12
                    target_frames = max(8, min(16, target_frames))
                    target_frames = max(1, min(target_frames, int(pre.max_frames)))
                    decoded = read_video_frames(
                        path,
                        start=start,
                        end=end,
                        target_fps=float(pre.fps),
                        num_frames=target_frames,
                        max_frames=target_frames,
                        min_frames=min(4, target_frames),
                        max_pixels=int(pre.max_pixels),
                        min_pixels=int(pre.min_pixels or self.spec.limits.min_pixels),
                        size_multiple=32,
                        sampling=pre.sampling_strategy,
                        cancel_token=cancel,
                    )
                    content.extend(
                        {"type": "image_url", "image_url": {"url": _image_data_url(frame)}}
                        for frame in decoded.frames
                    )
                    warnings.append(
                        f"Video used {len(decoded.frames)} chronological still frames plus separate audio; "
                        "frame/audio tokens are not interleaved by timestamp."
                    )
                include_audio = bool(pre.use_audio_in_video and info.has_audio)
                if include_audio:
                    samples = _slice_audio(read_audio(path), start, end)
                    content.append(
                        {"type": "input_audio", "input_audio": {"data": _wav_base64(samples), "format": "wav"}}
                    )
                elif pre.use_audio_in_video and not info.has_audio:
                    warnings.append("The video has no audio track; captioning used visual input only.")
            elif part.type == "audio":
                if part.path is None:
                    raise ValueError("Audio parts require a path")
                path = Path(part.path).expanduser().resolve(strict=True)
                info = probe_media(path)
                duration = max(0.0, float((end if end is not None else info.duration) or 0.0) - float(start or 0.0))
                if self.spec.family == "qwen3_omni_captioner" and duration > 30.0:
                    warnings.append("Qwen3-Omni Captioner works best with audio no longer than 30 seconds.")
                samples = _slice_audio(read_audio(path), start, end)
                content.append(
                    {"type": "input_audio", "input_audio": {"data": _wav_base64(samples), "format": "wav"}}
                )
            elif part.type == "image":
                if part.path is None:
                    raise ValueError("Image parts require a path")
                with Image.open(part.path) as image:
                    converted = ImageOps.exif_transpose(image).convert("RGB")
                    content.append(
                        {"type": "image_url", "image_url": {"url": _image_data_url(converted)}}
                    )
            elif part.type == "text":
                text = _text_for_part(part).strip()
                if text:
                    text_parts.append(text)
        return content, warnings, text_parts

    def _resolve_prompt(
        self,
        prompt: PromptSpec | str | None,
        media_types: list[str],
        text_parts: list[str],
    ) -> tuple[str | None, str, PromptPreset | None, str | None]:
        if isinstance(prompt, str):
            prompt = PromptSpec(user_prompt=prompt)
        prompt = prompt or PromptSpec()
        preset_id = prompt.preset_id
        if preset_id is None:
            if media_types == ["audio"]:
                preset_id = "qwen3_captioner_promptfree" if self.spec.family == "qwen3_omni_captioner" else "qwen3_audio_caption"
            elif media_types and all(value == "image" for value in media_types):
                preset_id = "qwen3_image_describe"
            elif len(set(media_types)) > 1:
                preset_id = "qwen3_joint_describe"
            elif not media_types and text_parts:
                preset_id = None
            else:
                preset_id = self.spec.default_prompt_preset
        preset = get_preset(preset_id) if preset_id else None
        system, user = render_prompt(preset, prompt.variables) if preset else (None, "")
        if prompt.system_prompt is not None:
            system = prompt.system_prompt
        if prompt.user_prompt is not None:
            user = prompt.user_prompt
        if self.spec.family == "qwen3_omni_captioner":
            warning = None
            if (system or "").strip() or (user or "").strip() or text_parts:
                warning = "Qwen3-Omni Captioner is prompt-free; ignoring the provided prompt text"
            return None, "", preset, warning
        all_text = [value for value in text_parts if value]
        if user:
            all_text.append(user)
        return system, "\n\n".join(all_text).strip(), preset, None

    def build_messages(
        self,
        media: MediaInput,
        prompt: PromptSpec | str | None = None,
        pre: PreprocessParams | None = None,
        cancel: object | None = None,
    ) -> _PreparedMessage:
        """Build typed OpenAI message parts without making an HTTP request."""

        parts = _parts(media)
        self._validate_parts(parts)
        pre = pre or _default_pre(self.spec.family)
        content, warnings, text_parts = self._media_content(parts, pre, cancel)
        media_types: list[str] = []
        for part in parts:
            if part.type in {"video", "video_audio"}:
                media_types.append("video")
            elif part.type != "text":
                media_types.append(part.type)
        system, user, preset, prompt_warning = self._resolve_prompt(prompt, media_types, text_parts)
        if prompt_warning:
            warnings.append(prompt_warning)
            get_log().warn(prompt_warning, scope="llama.cpp")
        if user:
            content.append({"type": "text", "text": user})
        if not content:
            raise ValueError("The user message is empty")
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})
        return _PreparedMessage(messages, warnings, preset)

    def _payload(self, messages: list[dict[str, Any]], gen: GenParams) -> dict[str, Any]:
        sampled = bool(gen.do_sample and gen.temperature > 0)
        payload: dict[str, Any] = {
            "model": self.variant.key,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": float(gen.temperature if sampled else 0.0),
            "top_p": float(gen.top_p if sampled else 1.0),
            "top_k": int(gen.top_k if sampled else 0),
            "max_tokens": min(int(gen.max_new_tokens), self.spec.limits.max_new_tokens_cap),
            "repeat_penalty": float(gen.repetition_penalty),
            "reasoning_format": "none",
        }
        if self.spec.family == "qwen3_omni_thinking":
            payload["chat_template_kwargs"] = {"enable_thinking": bool(gen.enable_thinking)}
            if not gen.enable_thinking:
                payload["reasoning_effort"] = "none"
        return payload

    def _stream_request(
        self,
        payload: dict[str, Any],
        callbacks: Callbacks,
        *,
        preserve_server_on_cancel: bool = False,
    ) -> _StreamResult:
        if not self._base_url:
            raise RuntimeError("llama-server is not running")
        endpoint = f"{self._base_url}/v1/chat/completions"
        events: queue.Queue[tuple[str, Any]] = queue.Queue()
        shared: dict[str, Any] = {}

        def request_worker() -> None:
            response: requests.Response | None = None
            try:
                response = self._session.post(
                    endpoint,
                    json=payload,
                    headers={"Content-Type": "application/json", "Authorization": "Bearer no-key"},
                    stream=True,
                    timeout=(30, None),
                )
                shared["response"] = response
                if response.status_code >= 400:
                    detail = response.text[-5000:]
                    raise RuntimeError(f"llama-server HTTP {response.status_code}: {detail}")
                for line in response.iter_lines(chunk_size=1):
                    events.put(("line", line))
                events.put(("done", None))
            except Exception as exc:
                events.put(("error", exc))
            finally:
                if response is not None:
                    response.close()

        worker = threading.Thread(target=request_worker, name="vcap-llama-sse", daemon=True)
        request_started = time.perf_counter()
        worker.start()
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        usage: dict[str, Any] = {}
        timings: dict[str, Any] = {}
        first_token_at: float | None = None
        chunks = 0
        cancelled = False
        server_finish_reason: str | None = None
        peak_state = {
            "value": float(
                resource_snapshot(self.gpu_index).get("vram_used_gb", 0.0) or 0.0
            )
        }
        stop_vram_monitor = threading.Event()

        def monitor_vram() -> None:
            while not stop_vram_monitor.wait(0.1):
                try:
                    used = float(
                        resource_snapshot(self.gpu_index).get("vram_used_gb", 0.0) or 0.0
                    )
                except Exception:
                    continue
                peak_state["value"] = max(peak_state["value"], used)

        vram_monitor = threading.Thread(
            target=monitor_vram,
            name="vcap-llama-vram",
            daemon=True,
        )
        vram_monitor.start()
        progress_key = f"llamacpp-caption-{id(events)}"
        try:
            complete = False
            while not complete:
                if _cancelled(callbacks.cancel):
                    cancelled = True
                    response = shared.get("response")
                    if response is not None:
                        response.close()
                    if not preserve_server_on_cancel:
                        self.stop()
                    break
                try:
                    kind, value = events.get(timeout=0.1)
                except queue.Empty:
                    continue
                if kind == "done":
                    complete = True
                    continue
                if kind == "error":
                    if cancelled:
                        break
                    raise RuntimeError(f"llama-server streaming request failed: {value}") from value
                for event in parse_sse_events([value]):
                    if isinstance(event.get("usage"), dict):
                        usage = event["usage"]
                    if isinstance(event.get("timings"), dict):
                        timings = event["timings"]
                    choices = event.get("choices")
                    if not isinstance(choices, list) or not choices:
                        continue
                    choice = choices[0] if isinstance(choices[0], dict) else {}
                    if choice.get("finish_reason") is not None:
                        server_finish_reason = str(choice["finish_reason"])
                    delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
                    text = delta.get("content")
                    reasoning = delta.get("reasoning_content")
                    if isinstance(text, str) and text:
                        content_parts.append(text)
                    if isinstance(reasoning, str) and reasoning:
                        reasoning_parts.append(reasoning)
                    _delta_callback(
                        callbacks.delta,
                        text if isinstance(text, str) else "",
                        reasoning if isinstance(reasoning, str) else "",
                    )
                    if (isinstance(text, str) and text) or (isinstance(reasoning, str) and reasoning):
                        now = time.perf_counter()
                        if first_token_at is None:
                            first_token_at = now
                        chunks += 1
                        speed = max(0, chunks - 1) / max(now - first_token_at, 1e-9)
                        message = f"Generating: {chunks} streamed chunks | {speed:.2f} tok/s"
                        console_progress.show_progress_line(message, key=progress_key)
                        _callback(callbacks.progress, message, new_tokens=chunks, tok_per_s=speed)
        finally:
            console_progress.finalize_progress_line(key=progress_key)
            worker.join(timeout=5.0)
            stop_vram_monitor.set()
            vram_monitor.join(timeout=1.0)
            try:
                peak_state["value"] = max(
                    peak_state["value"],
                    float(
                        resource_snapshot(self.gpu_index).get("vram_used_gb", 0.0)
                        or 0.0
                    ),
                )
            except Exception:
                pass
        ended = time.perf_counter()
        prompt_tokens = int(usage.get("prompt_tokens") or timings.get("prompt_n") or 0)
        completion_tokens = int(usage.get("completion_tokens") or timings.get("predicted_n") or chunks)
        prefill_s = float(timings.get("prompt_ms") or 0.0) / 1000.0
        decode_s = float(timings.get("predicted_ms") or 0.0) / 1000.0
        if prefill_s <= 0:
            prefill_s = max(0.0, (first_token_at or ended) - request_started)
        if decode_s <= 0:
            decode_s = max(0.0, ended - (first_token_at or ended))
        tok_per_s = float(timings.get("predicted_per_second") or 0.0)
        if tok_per_s <= 0 and decode_s > 0:
            tok_per_s = max(0, completion_tokens - 1) / decode_s
        if cancelled:
            finish_reason = "cancelled"
        elif str(server_finish_reason or "").casefold() == "length":
            finish_reason = "length"
        elif completion_tokens >= int(payload.get("max_tokens") or completion_tokens + 1):
            finish_reason = "length"
        else:
            finish_reason = "eos"
        return _StreamResult(
            "".join(content_parts),
            "".join(reasoning_parts),
            prompt_tokens,
            completion_tokens,
            prefill_s,
            decode_s,
            tok_per_s,
            ended - request_started,
            peak_state["value"],
            cancelled,
            finish_reason,
        )

    def _postprocess(self, raw: str, preset: PromptPreset | None) -> tuple[PostResult, str]:
        reasoning = ""
        answer = raw
        if self.spec.family == "qwen3_omni_thinking":
            reasoning, answer = split_thinking(raw)
        processor_name = preset.post_processor if preset and preset.post_processor else "plain"
        processor = POST_PROCESSORS.get(processor_name, plain)
        return processor(answer, {}), reasoning

    def chat(
        self,
        history: Sequence[ChatMessage | Mapping[str, Any]],
        media: MediaInput | None = None,
        *,
        system_prompt: str | None = None,
        gen: GenParams | None = None,
        pre: PreprocessParams | None = None,
        cb: Callbacks | None = None,
    ) -> ChatResult:
        """Stream a multi-turn OpenAI-compatible multimodal completion."""

        if self.spec.family == "qwen3_omni_captioner":
            raise ValueError(
                "Qwen3-Omni Captioner does not support chat; pick Qwen3-Omni Instruct or Thinking."
            )
        callbacks = cb or Callbacks()
        generation = gen or _default_gen(self.spec.family)
        generation = GenParams(
            temperature=generation.temperature,
            top_p=generation.top_p,
            top_k=generation.top_k,
            repetition_penalty=generation.repetition_penalty,
            max_new_tokens=min(generation.max_new_tokens, self.spec.limits.max_new_tokens_cap),
            do_sample=generation.do_sample,
            use_cache=generation.use_cache,
            enable_thinking=generation.enable_thinking,
        )
        preprocessing = pre or _default_pre(self.spec.family)
        normalized = normalize_chat_history(history)
        parts = _parts(media) if media is not None else []
        if parts:
            self._validate_parts(parts)
        with self._caption_lock:
            if not self.is_running:
                self.start(callbacks.progress, callbacks.cancel)
            _emit(callbacks.progress, "Preparing llama.cpp chat context")
            media_content: list[dict[str, Any]] = []
            warnings: list[str] = []
            text_parts: list[str] = []
            if parts:
                media_content, warnings, text_parts = self._media_content(
                    parts,
                    preprocessing,
                    callbacks.cancel,
                )
            if text_parts:
                merged = list(normalized)
                first_user = next(
                    (index for index, item in enumerate(merged) if item.role == "user"),
                    None,
                )
                if first_user is None:
                    raise ValueError("Chat history needs at least one user message")
                current = merged[first_user]
                merged[first_user] = ChatMessage(
                    "user",
                    "\n\n".join([*text_parts, current.content]).strip(),
                )
                normalized = merged

            def estimate_tokens(candidate: Sequence[ChatMessage]) -> int:
                characters = sum(len(item.content) for item in candidate)
                if system_prompt:
                    characters += len(system_prompt)
                media_allowance = 512 * len(media_content)
                return max(1, math.ceil(characters / 4) + 12 * len(candidate) + media_allowance)

            retained, dropped_turns, estimated_tokens = truncate_chat_history(
                normalized,
                estimate_tokens,
                self.context_size,
            )
            context_warning = ""
            if dropped_turns:
                context_warning = (
                    f"Context limit: dropped {dropped_turns} oldest conversation "
                    f"turn{'s' if dropped_turns != 1 else ''}; the media turn was kept."
                )
                warnings.append(context_warning)
                _emit(
                    callbacks.progress,
                    context_warning,
                    dropped_turns=dropped_turns,
                    context_trimmed=True,
                )
            messages = build_llamacpp_chat_messages(retained, media_content, system_prompt)
            for warning in warnings:
                if warning == context_warning:
                    continue
                _emit(callbacks.progress, warning)
            stream = self._stream_request(
                self._payload(messages, generation),
                callbacks,
                preserve_server_on_cancel=True,
            )
        if stream.reasoning:
            reasoning = stream.reasoning.strip()
            answer = stream.content.strip()
            raw = f"<think>{reasoning}</think>{answer}"
        elif self.spec.family == "qwen3_omni_thinking":
            raw = stream.content.strip()
            reasoning, answer = split_thinking(raw)
        else:
            raw = stream.content.strip()
            reasoning, answer = "", raw
        detail = (
            "stopped by EOS"
            if stream.finish_reason == "eos"
            else f"reached max_new_tokens {generation.max_new_tokens}"
            if stream.finish_reason == "length"
            else "cancelled"
        )
        _emit(
            callbacks.progress,
            f"Chat finished: {stream.completion_tokens} new tokens in {stream.total_s:.1f}s ({detail})",
            new_tokens=stream.completion_tokens,
            tok_per_s=stream.tok_per_s,
            cancelled=stream.cancelled,
            finish_reason=stream.finish_reason,
        )
        return ChatResult(
            text=answer,
            raw_text=raw,
            reasoning=reasoning,
            usage=TokenUsage(stream.prompt_tokens, stream.completion_tokens, stream.finish_reason),
            timing=CaptionTiming(
                stream.prefill_s,
                stream.decode_s,
                stream.tok_per_s,
                stream.total_s,
                stream.completion_tokens,
            ),
            peak_vram_gb=stream.peak_vram_gb,
            cancelled=stream.cancelled,
            warnings=tuple(warnings),
            retained_history=tuple(retained),
            dropped_turns=dropped_turns,
            context_tokens=stream.prompt_tokens or estimated_tokens,
        )

    def caption(
        self,
        media: MediaInput,
        prompt: PromptSpec | str | None = None,
        gen: GenParams | None = None,
        pre: PreprocessParams | None = None,
        cb: Callbacks | None = None,
    ) -> CaptionResult:
        """Prepare media, stream one chat completion, and normalize its result."""

        callbacks = cb or Callbacks()
        generation = gen or _default_gen(self.spec.family)
        preprocessing = pre or _default_pre(self.spec.family)
        try:
            with self._caption_lock:
                if not self.is_running:
                    self.start(callbacks.progress, callbacks.cancel)
                _emit(callbacks.progress, "Preparing llama.cpp multimodal message")
                prepared = self.build_messages(media, prompt, preprocessing, callbacks.cancel)
                for warning in prepared.warnings:
                    _emit(callbacks.progress, warning)
                stream = self._stream_request(self._payload(prepared.messages, generation), callbacks)
        except CancelledError:
            self.stop()
            raise
        if stream.reasoning:
            raw = f"<think>{stream.reasoning}</think>{stream.content}"
        else:
            raw = stream.content
        post, reasoning = self._postprocess(raw.strip(), prepared.preset)
        if stream.finish_reason == "eos":
            finish_detail = "stopped by EOS"
        elif stream.finish_reason == "length":
            token_cap = min(int(generation.max_new_tokens), self.spec.limits.max_new_tokens_cap)
            finish_detail = f"reached max_new_tokens {token_cap}"
        else:
            finish_detail = "cancelled"
        _emit(
            callbacks.progress,
            f"Generation finished: {stream.completion_tokens} new tokens in "
            f"{stream.total_s:.1f}s ({finish_detail})",
            new_tokens=stream.completion_tokens,
            tok_per_s=stream.tok_per_s,
            cancelled=stream.cancelled,
            finish_reason=stream.finish_reason,
        )
        return CaptionResult(
            text=post.text,
            raw_text=raw.strip(),
            reasoning=reasoning,
            structured=post.structured,
            segments=list(post.segments),
            usage=TokenUsage(stream.prompt_tokens, stream.completion_tokens, stream.finish_reason),
            timing=CaptionTiming(
                stream.prefill_s,
                stream.decode_s,
                stream.tok_per_s,
                stream.total_s,
                stream.completion_tokens,
            ),
            peak_vram_gb=stream.peak_vram_gb,
            cancelled=stream.cancelled,
            warnings=tuple(prepared.warnings),
        )


__all__ = [
    "DEFAULT_CONTEXT_SIZE",
    "LlamaCppServerPlan",
    "LlamaCppCaptioner",
    "build_llamacpp_chat_messages",
    "ensure_gguf",
    "find_free_port",
    "parse_sse_events",
    "server_plan_for_vram",
]
