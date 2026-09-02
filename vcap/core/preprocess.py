"""Model-aware frame planning, resizing, normalization, and clip analysis."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from .media import (
    MediaError,
    is_encoder_error,
    probe_media,
    read_audio,
    read_video_frames,
    resolve_video_encoder,
    run_ffmpeg,
    video_encode_args,
)
from .paths import normalize_path
from .subprocess_runner import CancelToken


@dataclass(frozen=True)
class FrameSamplingParams:
    """Frame-count and cadence constraints for model video decoding."""

    strategy: Literal["fps", "uniform", "keyframe", "adaptive"] = "fps"
    fps: float = 2.0
    num_frames: int | None = None
    max_frames: int = 160
    min_frames: int = 4
    frame_factor: int = 2


@dataclass(frozen=True)
class ResolutionParams:
    """Pixel-area and divisibility constraints for model inputs."""

    max_pixels: int
    min_pixels: int
    size_multiple: int = 28
    keep_aspect: bool = True
    max_ratio: float = 200.0


@dataclass(frozen=True)
class ClipQuality:
    """Sampled visual and audio quality measurements for one clip."""

    duration: float
    mean_luma: float
    black_ratio: float
    static_score: float
    sharpness: float
    silence_ratio: float
    has_audio: bool
    has_video: bool = True


@dataclass(frozen=True)
class AutoRejectRules:
    """Thresholds used to remove clips unsuitable for training/captioning."""

    min_duration_s: float = 0.0
    max_black_ratio: float = 1.0
    max_static_score: float = -1.0
    min_sharpness: float = 0.0
    require_audio: bool = False
    max_silence_ratio: float = 1.0


def _round_frame_count(value: int, minimum: int, maximum: int, factor: int) -> int:
    minimum = max(1, int(minimum))
    maximum = max(minimum, int(maximum))
    factor = max(1, int(factor))
    count = min(maximum, max(minimum, int(value)))
    if factor == 1:
        return count
    floored = (count // factor) * factor
    if floored >= minimum:
        return min(maximum, floored)
    ceiled = math.ceil(minimum / factor) * factor
    return min(maximum, max(minimum, ceiled))


def plan_frame_sampling(
    duration_s: float,
    source_fps: float,
    params: FrameSamplingParams,
) -> tuple[int, float, list[float]]:
    """Plan a factor-aligned sample count, actual cadence, and timestamps."""

    duration = float(duration_s)
    source_rate = float(source_fps)
    if not math.isfinite(duration) or duration <= 0:
        return 0, 0.0, []
    if not math.isfinite(source_rate) or source_rate <= 0:
        raise ValueError("source_fps must be positive")
    strategy = str(params.strategy).casefold()
    if strategy not in {"fps", "uniform", "keyframe", "adaptive"}:
        raise ValueError("strategy must be 'fps', 'uniform', 'keyframe', or 'adaptive'")
    minimum = max(1, int(params.min_frames))
    maximum = max(minimum, int(params.max_frames))
    requested_rate = min(source_rate, max(1e-9, float(params.fps)))
    if params.num_frames is not None:
        raw_count = int(params.num_frames)
    elif strategy == "uniform":
        raw_count = maximum
    elif strategy == "keyframe":
        raw_count = min(maximum, max(minimum, int(math.ceil(duration * requested_rate))))
    elif strategy == "adaptive":
        # Adaptive decoding may later choose motion-rich frames, but its time budget
        # is fixed here so the processor always sees a deterministic cadence.
        raw_count = min(maximum, max(minimum, int(math.ceil(duration * requested_rate))))
    else:
        raw_count = int(math.ceil(duration * requested_rate))
    count = _round_frame_count(raw_count, minimum, maximum, params.frame_factor)
    effective_fps = count / duration
    timestamps = [min(duration - 1e-9, index / effective_fps) for index in range(count)]
    return count, effective_fps, timestamps


def smart_resize(
    h: int,
    w: int,
    factor: int,
    min_pixels: int | None,
    max_pixels: int | None,
) -> tuple[int, int]:
    """Apply the qwen-omni-utils aspect-preserving resize rule."""

    height, width = int(h), int(w)
    multiple = int(factor)
    if height <= 0 or width <= 0 or multiple <= 0:
        raise ValueError("h, w, and factor must be positive")
    minimum = 4 * multiple * multiple if min_pixels is None else int(min_pixels)
    maximum = 16_384 * multiple * multiple if max_pixels is None else int(max_pixels)
    if minimum <= 0 or maximum < minimum:
        raise ValueError("max_pixels must be greater than or equal to positive min_pixels")
    ratio = max(height, width) / min(height, width)
    if ratio > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 200, got {ratio:g}")

    def round_by(value: float) -> int:
        return round(value / multiple) * multiple

    def floor_by(value: float) -> int:
        return max(multiple, math.floor(value / multiple) * multiple)

    def ceil_by(value: float) -> int:
        return max(multiple, math.ceil(value / multiple) * multiple)

    resized_h = max(multiple, round_by(height))
    resized_w = max(multiple, round_by(width))
    if resized_h * resized_w > maximum:
        beta = math.sqrt((height * width) / maximum)
        resized_h = floor_by(height / beta)
        resized_w = floor_by(width / beta)
    elif resized_h * resized_w < minimum:
        beta = math.sqrt(minimum / (height * width))
        resized_h = ceil_by(height * beta)
        resized_w = ceil_by(width * beta)
    return resized_h, resized_w


def token_budget_estimate(
    model_family: str,
    n_frames: int,
    h: int,
    w: int,
    audio_seconds: float,
) -> dict[str, int | float | str]:
    """Estimate processor video/audio tokens from the verified Omni formulas."""

    family = str(model_family).casefold().replace("_", "-")
    frames = max(0, int(n_frames))
    height, width = max(1, int(h)), max(1, int(w))
    seconds = max(0.0, float(audio_seconds))
    if family in {"qwen2.5-omni", "qwen2-5-omni", "qwen25-omni", "timechat", "avocado"}:
        canonical = "qwen2.5-omni"
        factor = 28
        temporal_factor = 2
        audio_rate = 25.0
        tokens_per_group = (height * width) / float(factor * factor)
        temporal_groups = math.ceil(frames / temporal_factor)
        video_tokens = math.ceil(temporal_groups * tokens_per_group)
        # Kept as a separate per-frame heuristic for UI comparison with older
        # Qwen calculators; the processor-accurate total uses temporal groups.
        tokens_per_frame = (height * width) / float(factor * factor * 4)
    elif family in {"qwen3-omni", "qwen3omni", "qwen3-omni-moe"}:
        canonical = "qwen3-omni"
        factor = 32
        temporal_factor = 2
        audio_rate = 13.0
        tokens_per_group = (height * width) / float(factor * factor)
        temporal_groups = math.ceil(frames / temporal_factor)
        video_tokens = math.ceil(temporal_groups * tokens_per_group)
        tokens_per_frame = video_tokens / frames if frames else 0.0
    else:
        raise ValueError(f"Unsupported model family: {model_family}")
    audio_tokens = math.ceil(seconds * audio_rate)
    total = int(video_tokens + audio_tokens)
    return {
        "model_family": canonical,
        "factor": factor,
        "temporal_factor": temporal_factor,
        "n_frames": frames,
        "height": height,
        "width": width,
        "tokens_per_frame": float(tokens_per_frame),
        "tokens_per_temporal_group": float(tokens_per_group),
        "video_tokens": int(video_tokens),
        "audio_tokens_per_second": audio_rate,
        "audio_tokens": int(audio_tokens),
        "total_input_tokens": total,
        "total_tokens": total,
    }


def fits_context(
    budget: dict[str, Any], context_tokens: int, reserve_output_tokens: int
) -> bool:
    """Return whether an estimated input plus output reserve fits the context."""

    input_tokens = int(budget.get("total_input_tokens", budget.get("total_tokens", 0)) or 0)
    return input_tokens + max(0, int(reserve_output_tokens)) <= max(0, int(context_tokens))


def _silence_ratio(
    samples: np.ndarray,
    sample_rate: int = 16000,
    threshold: float = 0.001,
) -> float:
    if samples.size == 0:
        return 1.0
    window = max(1, int(round(sample_rate * 0.02)))
    pad = (-samples.size) % window
    if pad:
        samples = np.pad(samples, (0, pad))
    windows = samples.reshape(-1, window).astype(np.float64, copy=False)
    rms = np.sqrt(np.mean(np.square(windows), axis=1))
    return float(np.mean(rms < max(0.0, float(threshold))))


def analyze_clip_quality(
    path: str | os.PathLike[str],
    *,
    sample_frames: int = 8,
    start_s: float | None = None,
    end_s: float | None = None,
    black_luma: int = 16,
    silence_rms: float = 0.001,
) -> ClipQuality:
    """Measure visual/audio quality for a whole source or a planned time range."""

    source = normalize_path(path, must_exist=True)
    info = probe_media(source)
    if not info.has_video and not info.has_audio:
        raise MediaError(f"No video or audio stream found in {source}")
    beginning = max(0.0, float(start_s or 0.0))
    ending = float(end_s) if end_s is not None else info.duration
    if info.duration is not None:
        beginning = min(beginning, float(info.duration))
        ending = float(info.duration) if ending is None else min(max(0.0, ending), float(info.duration))
    if ending is not None and ending <= beginning:
        raise ValueError("end_s must be greater than start_s")
    duration = max(0.0, float(ending) - beginning) if ending is not None else float(info.duration or 0.0)

    gray_frames: list[np.ndarray] = []
    luma_values: list[float] = []
    black_values: list[float] = []
    sharpness_values: list[float] = []
    if info.has_video:
        count = max(1, int(sample_frames))
        decoded = read_video_frames(
            source,
            start=beginning,
            end=ending,
            sampling="uniform",
            num_frames=count,
            max_frames=count,
            min_frames=1,
            max_pixels=640 * 640,
            min_pixels=None,
            size_multiple=2,
        )
        for frame in decoded.frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            gray_frames.append(gray)
            luma_values.append(float(np.mean(gray)))
            black_values.append(float(np.mean(gray <= max(0, min(255, int(black_luma))))))
            sharpness_values.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
    differences = [
        float(np.mean(cv2.absdiff(previous, current)))
        for previous, current in zip(gray_frames, gray_frames[1:])
    ]
    has_audio = bool(info.has_audio)
    silence = 1.0
    if has_audio:
        try:
            samples = read_audio(source, sample_rate=16000)
            start_sample = min(samples.size, max(0, int(round(beginning * 16000))))
            end_sample = samples.size if ending is None else min(samples.size, max(0, int(round(ending * 16000))))
            silence = _silence_ratio(
                samples[start_sample:end_sample],
                16000,
                max(0.0, float(silence_rms)),
            )
        except Exception:
            silence = 1.0
    return ClipQuality(
        duration=duration,
        mean_luma=float(np.mean(luma_values)) if luma_values else 0.0,
        black_ratio=float(np.mean(black_values)) if black_values else 0.0,
        static_score=float(np.mean(differences)) if differences else 0.0,
        sharpness=float(np.mean(sharpness_values)) if sharpness_values else 0.0,
        silence_ratio=silence,
        has_audio=has_audio,
        has_video=bool(info.has_video),
    )


def should_reject(
    quality: ClipQuality, rules: AutoRejectRules
) -> tuple[bool, list[str]]:
    """Evaluate quality metrics against auto-reject rules."""

    reasons: list[str] = []
    if quality.duration + 1e-9 < max(0.0, float(rules.min_duration_s)):
        reasons.append(
            f"too short ({quality.duration:.2f}s < {float(rules.min_duration_s):.2f}s)"
        )
    if quality.has_video and quality.black_ratio > float(rules.max_black_ratio):
        reasons.append(
            f"mostly black ({quality.black_ratio:.1%} > {float(rules.max_black_ratio):.1%})"
        )
    if quality.has_video and rules.max_static_score >= 0 and quality.static_score <= float(rules.max_static_score):
        reasons.append(
            f"too static (frame difference {quality.static_score:.3f} <= {float(rules.max_static_score):.3f})"
        )
    if quality.has_video and quality.sharpness < max(0.0, float(rules.min_sharpness)):
        reasons.append(
            f"low sharpness ({quality.sharpness:.2f} < {float(rules.min_sharpness):.2f})"
        )
    if rules.require_audio and not quality.has_audio:
        reasons.append("audio track required")
    if quality.has_audio and quality.silence_ratio > float(rules.max_silence_ratio):
        reasons.append(
            f"mostly silent ({quality.silence_ratio:.1%} > {float(rules.max_silence_ratio):.1%})"
        )
    return bool(reasons), reasons


def normalize_clip_for_model(
    src: str | os.PathLike[str],
    dst: str | os.PathLike[str],
    *,
    target_fps: float,
    max_pixels: int,
    min_pixels: int,
    size_multiple: int,
    keep_audio: bool,
    audio_sr: int = 16000,
    encoder: str = "libx264",
    crf: int = 18,
    preset: str = "veryfast",
    audio_bitrate: str = "192k",
    cancel: CancelToken | None = None,
) -> Path:
    """Create a deterministic encoded MP4 at exact FPS, geometry, and audio format."""

    source = normalize_path(src, must_exist=True)
    target = normalize_path(dst)
    info = probe_media(source)
    if not info.has_video or not info.width or not info.height:
        raise MediaError(f"No video stream found in {source}")
    rate = float(target_fps)
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("target_fps must be positive")
    multiple = math.lcm(max(1, int(size_multiple)), 2)
    out_h, out_w = smart_resize(info.height, info.width, multiple, min_pixels, max_pixels)
    audio_ready = (
        info.has_audio
        and info.audio_codec == "aac"
        and info.audio_channels == 1
        and info.audio_sample_rate == int(audio_sr)
    )
    requested_encoder = str(encoder or "libx264").strip().casefold()
    active_encoder = resolve_video_encoder(requested_encoder)
    requested_codec = "hevc" if active_encoder in {"libx265", "hevc_nvenc"} else "h264"
    selected_bitrate = str(audio_bitrate or "192k").strip().casefold()
    if selected_bitrate not in {"96k", "128k", "192k", "256k", "320k"}:
        selected_bitrate = "192k"
    video_ready = (
        source.suffix.casefold() == ".mp4"
        and info.video_codec in ({"hevc", "h265"} if requested_codec == "hevc" else {"h264", "avc1"})
        and info.width == out_w
        and info.height == out_h
        and info.fps is not None
        and abs(float(info.fps) - rate) <= max(0.001, rate * 0.0005)
    )
    desired_audio_ready = audio_ready if keep_audio else not info.has_audio
    if video_ready and desired_audio_ready:
        return source
    if source.resolve() == target.resolve():
        raise ValueError("dst must differ from src when normalization is required")
    target.parent.mkdir(parents=True, exist_ok=True)
    duration = float(info.duration or 0.0)
    silent_input = keep_audio and not info.has_audio

    def arguments_for(selected_encoder: str) -> list[str]:
        arguments: list[str] = ["-y", "-i", str(source)]
        if silent_input:
            arguments += ["-f", "lavfi", "-i", f"anullsrc=r={max(1, int(audio_sr))}:cl=mono"]
        arguments += [
            "-map",
            "0:v:0",
            "-vf",
            f"fps={rate:.12g},scale={out_w}:{out_h}:flags=lanczos",
            *video_encode_args(selected_encoder, crf, preset),
            "-pix_fmt",
            "yuv420p",
        ]
        if keep_audio:
            arguments += [
                "-map",
                "1:a:0" if silent_input else "0:a:0",
                "-c:a",
                "aac",
                "-ac",
                "1",
                "-ar",
                str(max(1, int(audio_sr))),
                "-b:a",
                selected_bitrate,
            ]
            if silent_input:
                arguments += ["-shortest"]
        else:
            arguments += ["-an"]
        arguments += ["-movflags", "+faststart", str(target)]
        return arguments

    try:
        run_ffmpeg(
            arguments_for(active_encoder),
            total_duration=duration or None,
            cancel_token=cancel,
        )
    except Exception as exc:
        if active_encoder == "libx264" or not is_encoder_error(exc):
            raise
        from .logs import get_log

        get_log().warn(
            f"FFmpeg encoder {active_encoder} failed; falling back to libx264.",
            scope="encode",
        )
        run_ffmpeg(
            arguments_for("libx264"),
            total_duration=duration or None,
            cancel_token=cancel,
        )
    return target


def ensure_audio_track(
    src: str | os.PathLike[str], dst: str | os.PathLike[str]
) -> Path:
    """Mux a duration-matched silent AAC track when a video has no audio."""

    source = normalize_path(src, must_exist=True)
    info = probe_media(source)
    if not info.has_video:
        raise MediaError(f"No video stream found in {source}")
    if info.has_audio:
        return source
    target = normalize_path(dst)
    if source.resolve() == target.resolve():
        raise ValueError("dst must differ from src when adding audio")
    target.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(
        [
            "-y",
            "-i",
            str(source),
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=16000:cl=mono",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-shortest",
            "-movflags",
            "+faststart",
            str(target),
        ],
        total_duration=float(info.duration or 0.0) or None,
    )
    return target


__all__ = [
    "AutoRejectRules",
    "ClipQuality",
    "FrameSamplingParams",
    "ResolutionParams",
    "analyze_clip_quality",
    "ensure_audio_track",
    "fits_context",
    "normalize_clip_for_model",
    "plan_frame_sampling",
    "should_reject",
    "smart_resize",
    "token_budget_estimate",
]
