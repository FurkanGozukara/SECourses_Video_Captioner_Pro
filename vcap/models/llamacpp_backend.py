"""llama.cpp server backend for multimodal Qwen3-Omni GGUF models.

``use_cache`` is not applicable to llama-server: its KV cache is managed by the
server, so the UI disables that Transformers-only control for GGUF variants.
"""

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
import re
import secrets
import shlex
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
from vcap.core.progress import UiThrottle
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
    chat_media_parts,
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
GGUF_MIRRORS: tuple[tuple[str, str], ...] = (
    ("MonsterMMORPG/Wan_GGUF", "{folder_name}/{filename}"),
)
DEFAULT_CONTEXT_SIZE = 32_768
DEFAULT_STARTUP_TIMEOUT_S = 900.0


@dataclass(frozen=True)
class LlamaCppRuntimeOptions:
    """Validated llama-server controls copied from ``JobSpec.runtime``."""

    max_frames: int = 32
    jpeg_quality: int = 90
    threads: int = 0
    batch_size: int = 2_048
    ubatch_size: int = 512
    flash_attn: str = "auto"
    cache_reuse: int = 0
    ignore_tier_context: bool = False
    extra_args: str = ""
    min_p: float = 0.05
    repeat_last_n: int = 64
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    fit_headroom_mib: int = 1_536
    startup_timeout_s: int = 900
    stream_idle_timeout_s: int = 120

    @classmethod
    def from_spec(cls, spec: Any | None) -> "LlamaCppRuntimeOptions":
        runtime = getattr(spec, "runtime", spec)

        def integer(name: str, default: int, minimum: int, maximum: int) -> int:
            try:
                value = int(getattr(runtime, name, default))
            except (TypeError, ValueError):
                value = default
            return max(minimum, min(maximum, value))

        def number(name: str, default: float, minimum: float, maximum: float) -> float:
            try:
                value = float(getattr(runtime, name, default))
            except (TypeError, ValueError):
                value = default
            return max(minimum, min(maximum, value))

        flash_attn = str(getattr(runtime, "gguf_flash_attn", "auto") or "auto").casefold()
        if flash_attn not in {"auto", "on", "off"}:
            flash_attn = "auto"
        return cls(
            max_frames=integer("gguf_max_frames", 32, 1, 128),
            jpeg_quality=integer("gguf_jpeg_quality", 90, 50, 100),
            threads=integer("gguf_threads", 0, 0, 256),
            batch_size=integer("gguf_batch_size", 2_048, 64, 8_192),
            ubatch_size=integer("gguf_ubatch_size", 512, 32, 4_096),
            flash_attn=flash_attn,
            cache_reuse=integer("gguf_cache_reuse", 0, 0, 4_096),
            ignore_tier_context=bool(
                getattr(runtime, "gguf_ignore_tier_context", False)
            ),
            extra_args=str(getattr(runtime, "gguf_extra_args", "") or ""),
            min_p=number("gguf_min_p", 0.05, 0.0, 1.0),
            repeat_last_n=integer("gguf_repeat_last_n", 64, 0, 4_096),
            presence_penalty=number("gguf_presence_penalty", 0.0, -2.0, 2.0),
            frequency_penalty=number("gguf_frequency_penalty", 0.0, -2.0, 2.0),
            fit_headroom_mib=integer("gguf_fit_headroom_mib", 1_536, 0, 8_192),
            startup_timeout_s=integer("gguf_startup_timeout_s", 900, 60, 3_600),
            stream_idle_timeout_s=integer("gguf_stream_idle_timeout_s", 120, 0, 3_600),
        )


@dataclass(frozen=True)
class GgufFrameBudget:
    """Resolved frame count and an audit trail of every reducing cap."""

    requested_frames: int
    selected_frames: int
    per_frame_tokens: int
    audio_tokens: int
    context_frame_cap: int
    messages: tuple[str, ...]


def frame_budget_for_video(
    *,
    duration_s: float,
    fps: float,
    sampling_strategy: str,
    max_frames: int,
    gguf_max_frames: int,
    context_size: int,
    max_new_tokens: int,
    max_pixels: int,
    audio_tokens: int = 0,
) -> GgufFrameBudget:
    """Resolve the GGUF still-frame plan from user, runtime, and context caps."""

    resolved_fps = max(0.01, float(fps))
    requested = max(1, int(math.ceil(max(0.0, float(duration_s)) * resolved_fps)))
    per_frame_tokens = max(1, int(math.ceil(max(1, int(max_pixels)) / float(32 * 32))))
    audio = max(0, int(audio_tokens))
    available = int(context_size) - max(0, int(max_new_tokens)) - audio - 512
    context_cap = max(1, available // per_frame_tokens)
    selected = requested
    messages: list[str] = []

    def apply_cap(limit: int, label: str) -> None:
        nonlocal selected
        cap = max(1, int(limit))
        if selected > cap:
            before = selected
            selected = cap
            messages.append(f"GGUF frame count capped from {before} to {selected} by {label}.")

    apply_cap(max_frames, "Maximum frames")
    apply_cap(gguf_max_frames, "GGUF maximum frames")
    apply_cap(context_cap, f"the {int(context_size)}-token context budget")
    messages.append(
        f"GGUF frame plan selected {selected} of {requested} frame(s) at "
        f"{resolved_fps:g} fps ({str(sampling_strategy or 'fps')}); estimated "
        f"{per_frame_tokens} tokens/frame and {audio} audio tokens."
    )
    return GgufFrameBudget(
        requested,
        selected,
        per_frame_tokens,
        audio,
        context_cap,
        tuple(messages),
    )


@dataclass(frozen=True)
class LlamaCppServerPlan:
    """Tier-aware llama-server context and device-memory fitting settings."""

    vram_tier_gb: int
    context_size: int
    gpu_layers: int | None  # None leaves -ngl unset so llama.cpp's fitter can choose
    fit: bool
    fit_target_mib: int
    n_cpu_moe: int | None
    context_tier_ignored: bool = False
    context_was_clamped: bool = False


# llama.cpp's fitter sizes weights, KV cache, and text compute buffers, but not the
# multimodal projector's image/audio encoding buffers. Measured on a 32 GB GPU, a
# 2,048 MiB target left only ~0.7 GiB free at generation peak; this headroom is added
# on top of the configured reserve so the reserve is honoured in practice.
MTMD_FIT_HEADROOM_MIB = 1_536


def server_plan_for_vram(
    total_vram_gb: float,
    *,
    requested_context: int = DEFAULT_CONTEXT_SIZE,
    q8_weights: bool = False,
    fit_target_mib: int = 2_048 + MTMD_FIT_HEADROOM_MIB,
    ignore_tier_context: bool = False,
) -> LlamaCppServerPlan:
    """Return a Qwen3-Omni server plan that lets llama.cpp fit device memory."""

    del q8_weights  # Kept for API compatibility; --fit now handles every GGUF size.
    observed = float(total_vram_gb or 32.0)
    tier = vram_tier_for_gb(observed)
    requested = max(4_096, int(requested_context))
    target = max(0, int(fit_target_mib))
    tier_context = 8_192 if tier <= 16 else 16_384 if tier <= 24 else 32_768
    ignored = bool(ignore_tier_context)
    context_size = requested if ignored else min(requested, tier_context)
    clamped = not ignored and context_size < requested
    if ignored:
        get_log().log("Context tier clamp bypassed", scope="llama.cpp")
    elif clamped:
        get_log().warn(
            f"Context clamped to {context_size} by the {tier} GB VRAM tier",
            scope="llama.cpp",
        )
    # -ngl stays unset: llama.cpp aborts fitting when n_gpu_layers is user-set.
    return LlamaCppServerPlan(
        tier,
        context_size,
        None,
        True,
        target,
        None,
        ignored,
        clamped,
    )


_FALSE_ENV_VALUES = {"0", "false", "no", "off"}
_GPU_LAYERS_RE = re.compile(r"\bn_gpu_layers\b\s*(?:=|:)\s*(-?\d+)", re.IGNORECASE)
_GPU_OFFLOAD_RE = re.compile(
    r"\boffloaded\s+(\d+)\s*/\s*(\d+)\s+layers?\s+to\s+GPU\b",
    re.IGNORECASE,
)
_CPU_MOE_RE = re.compile(r"\bn_cpu_moe\b\s*(?:=|:)\s*(\d+)", re.IGNORECASE)
_CONTEXT_RE = re.compile(r"\bn_ctx\b\s*(?:=|:)\s*(\d+)", re.IGNORECASE)
_BLOCK_INDEX_RE = re.compile(r"\bblk\\?\.(\d+)\\?\.", re.IGNORECASE)


def _env_non_negative_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return int(default)
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative integer, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {raw!r}")
    return value


def _server_plan_from_env(plan: LlamaCppServerPlan) -> LlamaCppServerPlan:
    fit_raw = os.environ.get("VCAP_LLAMACPP_FIT")
    fit = (
        plan.fit
        if fit_raw is None or not fit_raw.strip()
        else fit_raw.strip().casefold() not in _FALSE_ENV_VALUES
    )
    n_cpu_raw = os.environ.get("VCAP_LLAMACPP_N_CPU_MOE")
    n_cpu_moe = plan.n_cpu_moe
    if n_cpu_raw is not None and n_cpu_raw.strip():
        n_cpu_moe = _env_non_negative_int("VCAP_LLAMACPP_N_CPU_MOE", 0)
    gpu_layers = plan.gpu_layers
    layers_raw = os.environ.get("VCAP_LLAMACPP_GPU_LAYERS")
    if layers_raw is not None and layers_raw.strip():
        gpu_layers = _env_non_negative_int("VCAP_LLAMACPP_GPU_LAYERS", 0)
    elif gpu_layers is None and not fit:
        gpu_layers = 999  # no fitter and no explicit request: offload everything, as before
    return LlamaCppServerPlan(
        plan.vram_tier_gb,
        max(4_096, _env_non_negative_int("VCAP_LLAMACPP_CONTEXT_SIZE", plan.context_size)),
        gpu_layers,
        fit,
        _env_non_negative_int("VCAP_LLAMACPP_FIT_TARGET_MIB", plan.fit_target_mib),
        n_cpu_moe,
        plan.context_tier_ignored,
        plan.context_was_clamped,
    )


def _parse_fit_log(lines: Iterable[str]) -> dict[str, Any]:
    """Extract final fitted placement values from llama.cpp startup output."""

    parsed: dict[str, Any] = {}
    cpu_moe_layers: set[int] = set()
    for raw in lines:
        line = str(raw)
        if "failed to fit params" in line:
            parsed["fit_error"] = line.split("common_fit_params:", 1)[-1].strip()
        match = _GPU_LAYERS_RE.search(line)
        if match:
            parsed["n_gpu_layers"] = max(0, int(match.group(1)))
        match = _GPU_OFFLOAD_RE.search(line)
        if match:
            parsed["n_gpu_layers"] = int(match.group(1))
            parsed["n_gpu_layers_total"] = int(match.group(2))
        match = _CPU_MOE_RE.search(line)
        if match:
            parsed["n_cpu_moe"] = int(match.group(1))
        match = _CONTEXT_RE.search(line)
        if match:
            parsed["n_ctx"] = int(match.group(1))
        lowered = line.casefold()
        if (
            "cpu" in lowered
            and "overrid" in lowered
            and ("exps" in lowered or "expert" in lowered)
        ):
            match = _BLOCK_INDEX_RE.search(line)
            if match:
                cpu_moe_layers.add(int(match.group(1)))
    if cpu_moe_layers and "n_cpu_moe" not in parsed:
        parsed["n_cpu_moe"] = len(cpu_moe_layers)
    return parsed


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
    *,
    turn_media: Any = None,
) -> list[dict[str, Any]]:
    """Build OpenAI-compatible messages.

    ``media_content`` goes into the first user turn (legacy first-turn
    attachments); ``turn_media(index, message)`` returns the encoded content
    for a message's own attachments, indexed within the non-system messages.
    """

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
        content: list[dict[str, Any]] = []
        if index == first_user and media_content:
            content.extend(dict(item) for item in media_content)
        if turn_media is not None and message.media:
            content.extend(dict(item) for item in turn_media(index, message))
        if content:
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
    target = folder / filename
    sources = [(str(variant.gguf_repo), filename)]
    sources.extend(
        (repo_id, path_template.format(folder_name=variant.folder_name, filename=filename))
        for repo_id, path_template in GGUF_MIRRORS
    )
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        hf_hub_download = None

    for source_index, (repo_id, source_filename) in enumerate(sources):
        label = f"{repo_id}/{source_filename}"
        _emit(progress_cb, f"Downloading {label} with Hugging Face resumable cache")
        try:
            if _cancelled(cancel):
                raise CancelledError(f"download cancelled: {label}")
            if hf_hub_download is None:
                encoded_name = quote(source_filename, safe="/")
                url = f"{HF_RESOLVE_ROOT}/{repo_id}/resolve/main/{encoded_name}?download=true"
                return download_resumable(
                    url,
                    target,
                    expected_size=expected_size,
                    expected_sha256=expected_sha256,
                    progress_cb=progress_cb,
                    cancel=cancel,
                    label=label,
                )
            downloaded = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=source_filename,
                    revision="main",
                    local_dir=folder,
                    user_agent="SECourses-Video-Captioner-Pro/1.0",
                    force_download=bool(target.is_file() and expected_size and target.stat().st_size != expected_size),
                    tqdm_class=_hf_tqdm_class(progress_cb, cancel, label),
                )
            ).resolve(strict=True)
            if downloaded != target.resolve():
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(downloaded, target)
                downloaded = target.resolve(strict=True)
            if expected_size is not None and downloaded.stat().st_size != expected_size:
                raise RuntimeError(
                    f"Incomplete download for {filename}: got {downloaded.stat().st_size}, expected {expected_size} bytes"
                )
            _verify_file(downloaded, expected_sha256, progress_cb, cancel)
            target.with_name(target.name + ".part").unlink(missing_ok=True)
            _emit(progress_cb, f"Downloaded {label}", fraction=1.0, bytes_total=downloaded.stat().st_size)
            return downloaded
        except CancelledError:
            raise
        except Exception as exc:
            if _cancelled(cancel):
                raise CancelledError(f"download cancelled: {label}") from exc
            _emit(progress_cb, f"Download failed from {label}: {exc}")
            if source_index + 1 >= len(sources):
                raise
    raise RuntimeError(f"No download source is configured for {filename}")


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


def _image_data_url(image: Image.Image | np.ndarray, quality: int = 90) -> str:
    converted = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")
    converted = converted.convert("RGB")
    buffer = BytesIO()
    converted.save(
        buffer,
        format="JPEG",
        quality=max(50, min(100, int(quality))),
        optimize=True,
    )
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
    seed: int | None


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
        vram_reserve_gb: float = 2.0,
        runtime: Any | None = None,
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
        self.vram_reserve_gb = float(vram_reserve_gb)
        if not math.isfinite(self.vram_reserve_gb) or self.vram_reserve_gb < 0:
            raise ValueError("vram_reserve_gb must be a non-negative finite number")
        self.runtime_options = LlamaCppRuntimeOptions.from_spec(runtime)
        self.server_plan = server_plan_for_vram(
            self.vram_total_gb,
            requested_context=self.requested_context_size,
            q8_weights=self.variant.key.endswith("_gguf_q8"),
            fit_target_mib=(
                int(round(self.vram_reserve_gb * 1_024))
                + self.runtime_options.fit_headroom_mib
            ),
            ignore_tier_context=self.runtime_options.ignore_tier_context,
        )
        self._active_server_plan = self.server_plan
        self.context_size = self.server_plan.context_size
        selected_video_mode = (video_mode or os.environ.get("VCAP_LLAMACPP_VIDEO_MODE", "frames_audio")).strip().casefold()
        if selected_video_mode not in {"frames_audio", "native"}:
            raise ValueError("video_mode must be 'frames_audio' or 'native'")
        self.video_mode = selected_video_mode
        self._process: subprocess.Popen[str] | None = None
        self._port: int | None = None
        self._base_url: str | None = None
        self._log_thread: threading.Thread | None = None
        self._log_tail: deque[str] = deque(maxlen=1_000)
        self._log_lock = threading.Lock()
        self._fit_lock = threading.Lock()
        self.fit_report: dict[str, Any] = self._fit_report_for_plan(self.server_plan)
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
        with self._log_lock:
            return "\n".join(self._log_tail)

    def configure_runtime(self, spec: Any | None) -> None:
        """Apply ``JobSpec.runtime`` fields before the next server request."""

        generation = getattr(spec, "generation", getattr(spec, "gen", None))
        if int(getattr(generation, "no_repeat_ngram_size", 0) or 0) > 0:
            get_log().warn(
                "no_repeat_ngram_size is ignored by llama.cpp",
                scope="llama.cpp",
            )
        options = LlamaCppRuntimeOptions.from_spec(spec)
        if options == self.runtime_options:
            return
        if self.is_running:
            get_log().warn(
                "GGUF server options changed; restarting llama-server before the next request.",
                scope="llama.cpp",
            )
            self.stop()
        self.runtime_options = options
        self.server_plan = server_plan_for_vram(
            self.vram_total_gb,
            requested_context=self.requested_context_size,
            q8_weights=self.variant.key.endswith("_gguf_q8"),
            fit_target_mib=(
                int(round(self.vram_reserve_gb * 1_024))
                + options.fit_headroom_mib
            ),
            ignore_tier_context=options.ignore_tier_context,
        )
        self._active_server_plan = self.server_plan
        self.context_size = self.server_plan.context_size
        with self._fit_lock:
            self.fit_report = self._fit_report_for_plan(self.server_plan)

    @staticmethod
    def _fit_report_for_plan(plan: LlamaCppServerPlan) -> dict[str, Any]:
        return {
            "backend": "llamacpp",
            "mode": "fit" if plan.fit else "manual",
            "fit": bool(plan.fit),
            "fit_target_mib": int(plan.fit_target_mib),
            "requested_gpu_layers": None if plan.gpu_layers is None else int(plan.gpu_layers),
            "requested_n_cpu_moe": plan.n_cpu_moe,
            "requested_context_size": int(plan.context_size),
            "context_tier_ignored": bool(plan.context_tier_ignored),
            "context_was_clamped": bool(plan.context_was_clamped),
            "n_gpu_layers": None,
            "n_cpu_moe": plan.n_cpu_moe,
            "n_ctx": int(plan.context_size),
        }

    def _update_fit_report_from_log_tail(self) -> None:
        with self._log_lock:
            lines = list(self._log_tail)
        parsed = _parse_fit_log(lines)
        with self._fit_lock:
            self.fit_report.update(parsed)
            fitted_context = self.fit_report.get("n_ctx")
        if isinstance(fitted_context, int) and fitted_context > 0:
            self.context_size = fitted_context

    def block_swap_summary(self) -> dict[str, Any]:
        """Return the llama.cpp fit result in the common JSON-safe report slot."""

        self._update_fit_report_from_log_tail()
        with self._fit_lock:
            return json.loads(json.dumps(self.fit_report))

    def _server_command(self, server: Path, port: int) -> list[str]:
        if not self.variant.gguf_files or len(self.variant.gguf_files) < 2:
            raise RuntimeError(f"{self.variant.key} does not define a model and mmproj")
        model = self.model_dir / self.variant.gguf_files[0]
        mmproj = self.model_dir / self.variant.gguf_files[1]
        plan = _server_plan_from_env(self.server_plan)
        self._active_server_plan = plan
        self.context_size = plan.context_size
        with self._fit_lock:
            self.fit_report = self._fit_report_for_plan(plan)
        command = [
            str(server),
            "--model",
            str(model),
            "--mmproj",
            str(mmproj),
            "-c",
            str(plan.context_size),
            "--jinja",
            "--no-webui",
            "-np",
            "1",
            "-b",
            str(self.runtime_options.batch_size),
            "-ub",
            str(self.runtime_options.ubatch_size),
            "-fa",
            self.runtime_options.flash_attn,
        ]
        if self.runtime_options.threads > 0:
            command.extend(["--threads", str(self.runtime_options.threads)])
        if self.runtime_options.cache_reuse > 0:
            command.extend(["--cache-reuse", str(self.runtime_options.cache_reuse)])
        if plan.gpu_layers is not None:
            command.extend(["-ngl", str(plan.gpu_layers)])
        command.extend(["--fit", "on" if plan.fit else "off", "--fit-target", str(plan.fit_target_mib)])
        if plan.n_cpu_moe is not None:
            command.extend(["--n-cpu-moe", str(plan.n_cpu_moe)])
        command.extend(
            [
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
        )
        if self.runtime_options.extra_args.strip():
            try:
                command.extend(shlex.split(self.runtime_options.extra_args))
            except ValueError as exc:
                raise ValueError(f"Invalid GGUF extra arguments: {exc}") from exc
        return command

    def _read_server_logs(self, process: subprocess.Popen[str]) -> None:
        stream = process.stdout
        if stream is None:
            return
        try:
            for raw in iter(stream.readline, ""):
                line = raw.rstrip("\r\n")
                if not line:
                    continue
                with self._log_lock:
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
        timeout_s: float | None = None,
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
            if self._active_server_plan.context_tier_ignored:
                _emit(progress_cb, "Context tier clamp bypassed")
            elif self._active_server_plan.context_was_clamped:
                _emit(
                    progress_cb,
                    f"Context clamped to {self._active_server_plan.context_size} by the "
                    f"{self._active_server_plan.vram_tier_gb} GB VRAM tier",
                )
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
            with self._log_lock:
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

            resolved_timeout = (
                float(self.runtime_options.startup_timeout_s)
                if timeout_s is None
                else float(timeout_s)
            )
            deadline = time.monotonic() + max(10.0, resolved_timeout)
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
                            self._update_fit_report_from_log_tail()
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
                    f"llama-server did not become healthy within {resolved_timeout:.0f}s ({last_health_error}). "
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
        gen: GenParams | None = None,
    ) -> tuple[list[dict[str, Any]], list[str], list[str]]:
        content: list[dict[str, Any]] = []
        warnings: list[str] = []
        text_parts: list[str] = []
        generation = gen or _default_gen(self.spec.family)
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
                duration = max(
                    0.0,
                    float((end if end is not None else info.duration) or 0.0)
                    - float(start or 0.0),
                )
                include_audio = bool(pre.use_audio_in_video and info.has_audio)
                use_native = self.video_mode == "native" and start is None and end is None
                if use_native:
                    content.append(_native_video_part(path))
                else:
                    audio_tokens = (
                        int(
                            math.ceil(
                                duration
                                * float(getattr(self.spec.limits, "audio_tokens_per_s", 13.0))
                            )
                        )
                        if include_audio
                        else 0
                    )
                    budget = frame_budget_for_video(
                        duration_s=duration,
                        fps=float(pre.fps),
                        sampling_strategy=pre.sampling_strategy,
                        max_frames=int(pre.max_frames),
                        gguf_max_frames=self.runtime_options.max_frames,
                        context_size=self.context_size,
                        max_new_tokens=int(generation.max_new_tokens),
                        max_pixels=int(pre.max_pixels),
                        audio_tokens=audio_tokens,
                    )
                    target_frames = budget.selected_frames
                    for message in budget.messages:
                        get_log().log(message, scope="llama.cpp")
                    warnings.extend(budget.messages)
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
                        adaptive_threshold=float(pre.adaptive_threshold),
                        cancel_token=cancel,
                    )
                    content.extend(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": _image_data_url(
                                    frame,
                                    self.runtime_options.jpeg_quality,
                                )
                            },
                        }
                        for frame in decoded.frames
                    )
                    warnings.append(
                        f"Video used {len(decoded.frames)} chronological still frames plus separate audio; "
                        "frame/audio tokens are not interleaved by timestamp."
                    )
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
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": _image_data_url(
                                    converted,
                                    self.runtime_options.jpeg_quality,
                                )
                            },
                        }
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
        gen: GenParams | None = None,
    ) -> _PreparedMessage:
        """Build typed OpenAI message parts without making an HTTP request."""

        parts = _parts(media)
        self._validate_parts(parts)
        pre = pre or _default_pre(self.spec.family)
        content, warnings, text_parts = self._media_content(parts, pre, cancel, gen)
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
            "repeat_last_n": int(self.runtime_options.repeat_last_n),
            "reasoning_format": "none",
        }
        if sampled:
            payload["min_p"] = float(self.runtime_options.min_p)
        if self.runtime_options.presence_penalty != 0:
            payload["presence_penalty"] = float(self.runtime_options.presence_penalty)
        if self.runtime_options.frequency_penalty != 0:
            payload["frequency_penalty"] = float(self.runtime_options.frequency_penalty)
        if self.spec.family == "qwen3_omni_thinking":
            payload["chat_template_kwargs"] = {"enable_thinking": bool(gen.enable_thinking)}
            if not gen.enable_thinking:
                payload["reasoning_effort"] = "none"
        if sampled:
            requested_seed = int(getattr(gen, "seed", -1))
            payload["seed"] = (
                secrets.randbits(32) if requested_seed < 0 else requested_seed
            )
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
                    timeout=(30, self.runtime_options.stream_idle_timeout_s or None),
                )
                shared["response"] = response
                if response.status_code >= 400:
                    detail = response.text[-5000:]
                    raise RuntimeError(f"llama-server HTTP {response.status_code}: {detail}")
                for line in response.iter_lines(chunk_size=None):
                    events.put(("line", line))
                events.put(("done", None))
            except requests.Timeout as exc:
                idle = self.runtime_options.stream_idle_timeout_s
                events.put(
                    (
                        "error",
                        RuntimeError(
                            f"llama-server produced no data for {idle} second(s); generation timed out"
                        ),
                    )
                )
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
        progress_throttle = UiThrottle(0.1)
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
                    terminal_chunk = choice.get("finish_reason") is not None
                    if terminal_chunk:
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
                    has_delta = (isinstance(text, str) and bool(text)) or (
                        isinstance(reasoning, str) and bool(reasoning)
                    )
                    if has_delta:
                        now = time.perf_counter()
                        if first_token_at is None:
                            first_token_at = now
                        chunks += 1
                    if has_delta or terminal_chunk:
                        now = time.perf_counter()
                        if progress_throttle.should_emit(
                            force=chunks == 1 or terminal_chunk
                        ):
                            reported_tokens = (
                                int(
                                    usage.get("completion_tokens")
                                    or timings.get("predicted_n")
                                    or chunks
                                )
                                if terminal_chunk
                                else chunks
                            )
                            speed = (
                                float(timings.get("predicted_per_second") or 0.0)
                                if terminal_chunk
                                else max(0, chunks - 1)
                                / max(now - (first_token_at or now), 1e-9)
                            )
                            message = (
                                f"Generation finished: {reported_tokens} tokens | {speed:.2f} tok/s"
                                if terminal_chunk
                                else f"Generating: {chunks} streamed chunks | {speed:.2f} tok/s"
                            )
                            console_progress.show_progress_line(message, key=progress_key)
                            progress_data: dict[str, Any] = {
                                "new_tokens": reported_tokens,
                                "tok_per_s": speed,
                            }
                            if terminal_chunk:
                                progress_data.update(
                                    finish_reason=server_finish_reason,
                                    prompt_tokens=int(
                                        usage.get("prompt_tokens")
                                        or timings.get("prompt_n")
                                        or 0
                                    ),
                                    context_limit=self.context_size,
                                )
                            _callback(callbacks.progress, message, **progress_data)
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
            int(payload["seed"]) if "seed" in payload else None,
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
            seed=getattr(generation, "seed", -1),
            context_tokens=getattr(generation, "context_tokens", None),
        )
        preprocessing = pre or _default_pre(self.spec.family)
        normalized = normalize_chat_history(history)
        # Legacy first-turn media plus each user turn's own attachments, probed once.
        legacy_parts = _parts(media) if media is not None else []
        part_by_path: dict[str, MediaPart] = {}
        for item in normalized:
            for raw, part in zip(item.media, chat_media_parts(item.media)):
                part_by_path[raw] = part
        all_parts = legacy_parts + [part_by_path[raw] for item in normalized for raw in item.media]
        if all_parts:
            self._validate_parts(all_parts)
        with self._caption_lock:
            if not self.is_running:
                self.start(callbacks.progress, callbacks.cancel)
            _emit(callbacks.progress, "Preparing llama.cpp chat context")
            first_content: list[dict[str, Any]] = []
            warnings: list[str] = []
            text_parts: list[str] = []
            if legacy_parts:
                first_content, warnings, text_parts = self._media_content(
                    legacy_parts,
                    preprocessing,
                    callbacks.cancel,
                    generation,
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
                    current.media,
                )
                normalized = merged

            def estimate_tokens(candidate: Sequence[ChatMessage]) -> int:
                characters = sum(len(item.content) for item in candidate)
                if system_prompt:
                    characters += len(system_prompt)
                # Each attachment becomes sampled frames and/or audio; budget generously.
                attachments = len(legacy_parts) + sum(len(item.media) for item in candidate)
                return max(1, math.ceil(characters / 4) + 12 * len(candidate) + 512 * attachments)

            retained, dropped_turns, estimated_tokens = truncate_chat_history(
                normalized,
                estimate_tokens,
                self.context_size,
            )
            context_warning = ""
            if dropped_turns:
                context_warning = (
                    f"Context limit: dropped {dropped_turns} oldest conversation "
                    f"turn{'s' if dropped_turns != 1 else ''}; the first turn was kept."
                )
                warnings.append(context_warning)
                _emit(
                    callbacks.progress,
                    context_warning,
                    dropped_turns=dropped_turns,
                    context_trimmed=True,
                )
            # Encode attachments only for the turns that survived truncation.
            turn_content: list[list[dict[str, Any]]] = []
            for item in retained:
                if item.role == "system":
                    continue
                content: list[dict[str, Any]] = []
                if item.media:
                    content, item_warnings, _ = self._media_content(
                        [part_by_path[raw] for raw in item.media],
                        preprocessing,
                        callbacks.cancel,
                        generation,
                    )
                    warnings.extend(item_warnings)
                turn_content.append(content)
            messages = build_llamacpp_chat_messages(
                retained,
                first_content,
                system_prompt,
                turn_media=lambda index, message: turn_content[index],
            )
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
            prompt_tokens=stream.prompt_tokens,
            context_limit=self.context_size,
        )
        return ChatResult(
            text=answer,
            raw_text=raw,
            reasoning=reasoning,
            usage=TokenUsage(
                stream.prompt_tokens,
                stream.completion_tokens,
                stream.finish_reason,
                stream.seed,
            ),
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
            context_limit=self.context_size,
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
                prepared = self.build_messages(
                    media,
                    prompt,
                    preprocessing,
                    callbacks.cancel,
                    generation,
                )
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
            prompt_tokens=stream.prompt_tokens,
            context_limit=self.context_size,
        )
        return CaptionResult(
            text=post.text,
            raw_text=raw.strip(),
            reasoning=reasoning,
            structured=post.structured,
            segments=list(post.segments),
            usage=TokenUsage(
                stream.prompt_tokens,
                stream.completion_tokens,
                stream.finish_reason,
                stream.seed,
            ),
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
    "GgufFrameBudget",
    "LlamaCppServerPlan",
    "LlamaCppCaptioner",
    "LlamaCppRuntimeOptions",
    "build_llamacpp_chat_messages",
    "ensure_gguf",
    "find_free_port",
    "frame_budget_for_video",
    "parse_sse_events",
    "server_plan_for_vram",
]
