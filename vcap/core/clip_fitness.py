"""Trainer frame-count fitness, duration suggestions, and bucket previews."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from .media import MediaInfo, probe_media


TRAINER_TARGETS: dict[str, dict[str, Any]] = {
    "wan": {
        "label": "Wan",
        "frame_multiple": 4,
        "frame_offset": 1,
        "frame_rule": "4k+1",
        "frames": 81,
        "fps": 16.0,
        "default_frames": 81,
        "default_fps": 16.0,
        "min_frames": 5,
        "max_frames": 129,
        "recommended_frames": [17, 33, 49, 65, 81, 97, 113, 129],
        "buckets": {"480p": (832, 480), "720p": (1280, 720)},
        "resolution_multiple": 16,
    },
    "hunyuan": {
        "label": "Hunyuan",
        "frame_multiple": 4,
        "frame_offset": 1,
        "frame_rule": "4k+1",
        "frames": 129,
        "fps": 24.0,
        "default_frames": 129,
        "default_fps": 24.0,
        "min_frames": 5,
        "max_frames": 257,
        "recommended_frames": [49, 65, 81, 97, 113, 129, 161, 193, 225, 257],
        "buckets": {"480p": (848, 480), "720p": (1280, 720), "1080p": (1920, 1088)},
        "resolution_multiple": 16,
    },
    "ltx2": {
        "label": "LTX 2.x",
        "frame_multiple": 8,
        "frame_offset": 1,
        "frame_rule": "8k+1",
        "frames": 121,
        "fps": 25.0,
        "default_frames": 121,
        "default_fps": 25.0,
        "min_frames": 9,
        "max_frames": 257,
        "recommended_frames": [9, 17, 25, 33, 41, 49, 65, 81, 97, 121, 129, 161, 193, 225, 257],
        "buckets": {"480p": (832, 480), "720p": (1280, 720), "1080p": (1920, 1080)},
        "resolution_multiple": 8,
    },
    "minimax_h3": {
        "label": "MiniMax H3",
        "frame_multiple": 17,
        "frame_offset": 5,
        "frame_rule": "17n+5",
        "frames": 124,
        "fps": 24.0,
        "default_frames": 124,
        "default_fps": 24.0,
        "min_frames": 124,
        "max_frames": 345,
        "recommended_frames": list(range(124, 346, 17)),
        "buckets": {"480p": (832, 480), "720p": (1280, 736), "1080p": (1920, 1088)},
        "resolution_multiple": 32,
    },
    "custom": {
        "label": "Custom",
        "frame_multiple": 1,
        "frame_offset": 0,
        "frame_rule": "custom",
        "frames": 1,
        "fps": 24.0,
        "default_frames": 1,
        "default_fps": 24.0,
        "min_frames": 1,
        "max_frames": 100_000,
        "recommended_frames": [1],
        "buckets": {"480p": (854, 480), "720p": (1280, 720), "1080p": (1920, 1080)},
        "resolution_multiple": 2,
    },
}


@dataclass(frozen=True)
class FitnessReport:
    """Trainer compatibility result for one clip."""

    ok: bool
    frames_available: int
    frames_needed: int
    suggested_frames: int | None
    warnings: list[str]
    bucket: tuple[int, int]


def is_clip_fitness_plan_path(path: str | os.PathLike[str], source_root: str | os.PathLike[str]) -> bool:
    """Return whether a scanned path lives inside a generated clip-fitness plan folder."""

    candidate = Path(path).resolve(strict=False)
    root = Path(source_root).resolve(strict=False)
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        return False
    return any(part.casefold() == "clip_fitness" for part in relative.parts[:-1])


def _target_config(target: str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(target, Mapping):
        base_name = str(target.get("name", target.get("target", "custom"))).casefold()
        base = dict(TRAINER_TARGETS.get(base_name, TRAINER_TARGETS["custom"]))
        base.update(dict(target))
        if "frames" in target and "default_frames" not in target:
            base["default_frames"] = int(target["frames"])
        if "target_frames" in target and "default_frames" not in target:
            raw_frames = target["target_frames"]
            if isinstance(raw_frames, (list, tuple)):
                raw_frames = raw_frames[0] if raw_frames else 1
            base["default_frames"] = int(raw_frames)
        if "fps" in target and "default_fps" not in target:
            base["default_fps"] = float(target["fps"])
        if "frame_factor" in target and "frame_multiple" not in target:
            base["frame_multiple"] = int(target["frame_factor"])
        if "recommended_frames" not in target and (
            "frames" in target or "target_frames" in target or "default_frames" in target
        ):
            base["recommended_frames"] = [int(base["default_frames"])]
        return base
    key = str(target).casefold().strip()
    if key not in TRAINER_TARGETS:
        raise ValueError(f"Unknown trainer target: {target}")
    return dict(TRAINER_TARGETS[key])


def _valid_floor(frames: int, config: Mapping[str, Any]) -> int | None:
    value = int(frames)
    multiple = max(1, int(config.get("frame_multiple", 1)))
    offset = int(config.get("frame_offset", 0))
    minimum = max(1, int(config.get("min_frames", 1)))
    maximum = max(minimum, int(config.get("max_frames", value)))
    value = min(value, maximum)
    if value < minimum:
        return None
    candidate = ((value - offset) // multiple) * multiple + offset
    while candidate > value:
        candidate -= multiple
    return candidate if candidate >= minimum else None


def suggest_clip_length(
    target: str | Mapping[str, Any], fps: float
) -> list[tuple[int, float]]:
    """Return valid trainer frame counts with their clip durations."""

    config = _target_config(target)
    rate = float(fps or config.get("default_fps", 0.0))
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("fps must be positive")
    raw = config.get("recommended_frames") or [config.get("default_frames", 1)]
    counts = sorted({int(value) for value in raw if _valid_floor(int(value), config) == int(value)})
    return [(frames, frames / rate) for frames in counts]


def _bucket_dimensions(
    bucket: str | tuple[int, int] | list[int] | Mapping[str, Any]
) -> tuple[int, int, int]:
    if isinstance(bucket, Mapping):
        width = int(bucket.get("width", 0))
        height = int(bucket.get("height", 0))
        multiple = max(1, int(bucket.get("multiple", 2)))
    elif isinstance(bucket, str):
        standard = {
            "480p": (854, 480),
            "720p": (1280, 720),
            "1080p": (1920, 1080),
        }
        if bucket.casefold() not in standard:
            raise ValueError(f"Unknown resolution bucket: {bucket}")
        width, height = standard[bucket.casefold()]
        multiple = 2
    else:
        if len(bucket) != 2:
            raise ValueError("bucket must contain width and height")
        width, height = int(bucket[0]), int(bucket[1])
        multiple = 2
    if width <= 0 or height <= 0:
        raise ValueError("bucket dimensions must be positive")
    return width, height, multiple


def _round_dimension(value: float, multiple: int, minimum: int | None = None) -> int:
    rounded = max(multiple, int(round(value / multiple)) * multiple)
    if minimum is not None:
        rounded = min(rounded, minimum)
        rounded = max(multiple, (rounded // multiple) * multiple)
    return rounded


def resolution_bucket_preview(
    w: int,
    h: int,
    bucket: str | tuple[int, int] | list[int] | Mapping[str, Any],
    policy: Literal["keep_ar", "letterbox", "crop", "area"],
) -> tuple[int, int, dict[str, Any]]:
    """Preview resize geometry plus crop/padding details for a bucket policy."""

    source_w, source_h = int(w), int(h)
    if source_w <= 0 or source_h <= 0:
        raise ValueError("source dimensions must be positive")
    target_w, target_h, multiple = _bucket_dimensions(bucket)
    if (source_w < source_h) != (target_w < target_h):
        target_w, target_h = target_h, target_w
    selected = str(policy).casefold()
    if selected == "keep_ar":
        scale = min(target_w / source_w, target_h / source_h)
        out_w = _round_dimension(source_w * scale, multiple, target_w)
        out_h = _round_dimension(source_h * scale, multiple, target_h)
        info = {"operation": "resize", "scale": scale, "crop": (0, 0, 0, 0), "pad": (0, 0, 0, 0)}
        return out_w, out_h, info
    if selected == "letterbox":
        scale = min(target_w / source_w, target_h / source_h)
        content_w = min(target_w, _round_dimension(source_w * scale, multiple, target_w))
        content_h = min(target_h, _round_dimension(source_h * scale, multiple, target_h))
        left = (target_w - content_w) // 2
        top = (target_h - content_h) // 2
        info = {
            "operation": "letterbox",
            "scale": scale,
            "content": (content_w, content_h),
            "crop": (0, 0, 0, 0),
            "pad": (left, top, target_w - content_w - left, target_h - content_h - top),
        }
        return target_w, target_h, info
    if selected == "crop":
        scale = max(target_w / source_w, target_h / source_h)
        scaled_w = max(target_w, _round_dimension(source_w * scale, multiple))
        scaled_h = max(target_h, _round_dimension(source_h * scale, multiple))
        left = (scaled_w - target_w) // 2
        top = (scaled_h - target_h) // 2
        info = {
            "operation": "crop",
            "scale": scale,
            "scaled": (scaled_w, scaled_h),
            "crop": (left, top, scaled_w - target_w - left, scaled_h - target_h - top),
            "pad": (0, 0, 0, 0),
        }
        return target_w, target_h, info
    if selected == "area":
        scale = math.sqrt((target_w * target_h) / (source_w * source_h))
        out_w = _round_dimension(source_w * scale, multiple)
        out_h = _round_dimension(source_h * scale, multiple)
        info = {"operation": "area", "scale": scale, "crop": (0, 0, 0, 0), "pad": (0, 0, 0, 0)}
        return out_w, out_h, info
    raise ValueError("policy must be 'keep_ar', 'letterbox', 'crop', or 'area'")


def _media_info(value: Any) -> MediaInfo:
    if isinstance(value, MediaInfo):
        return value
    if isinstance(value, (str, os.PathLike, Path)):
        return probe_media(value)
    path = getattr(value, "path", None)
    if path is not None:
        probed = probe_media(path)
        if probed.kind != "unknown":
            return probed
    if isinstance(value, Mapping):
        path = value.get("path")
        if path:
            probed = probe_media(path)
            if probed.kind != "unknown":
                return probed
        return MediaInfo(
            path=Path(str(path or ".")),
            kind="video",
            duration=float(value.get("duration", value.get("duration_s", 0.0)) or 0.0),
            width=int(value.get("width", 0) or 0) or None,
            height=int(value.get("height", 0) or 0) or None,
            fps=float(value.get("fps", 0.0) or 0.0) or None,
            nb_frames=int(value.get("nb_frames", value.get("frames", 0)) or 0) or None,
            has_video=True,
            has_audio=bool(value.get("has_audio", False)),
        )
    raise TypeError("media_info_or_clip must be MediaInfo, a path, clip, or mapping")


def evaluate_clip(
    media_info_or_clip: Any,
    target: str | Mapping[str, Any],
    resolution_bucket: str | tuple[int, int] | list[int] | Mapping[str, Any],
    policy: Literal["keep_ar", "letterbox", "crop", "area"] = "keep_ar",
) -> FitnessReport:
    """Evaluate whether a clip supplies enough frames for a trainer target."""

    info = _media_info(media_info_or_clip)
    config = _target_config(target)
    needed = int(config.get("default_frames", 1))
    if info.nb_frames is not None and info.nb_frames > 0:
        available = int(info.nb_frames)
    else:
        rate = float(info.fps or config.get("default_fps", 0.0) or 0.0)
        available = max(0, int(math.floor(float(info.duration or 0.0) * rate + 1e-9)))
    suggested = _valid_floor(available, config)
    warnings: list[str] = []
    if available < needed:
        warnings.append(
            f"will be dropped by trainer: only {available} frames, needs {needed}"
        )
    elif suggested is not None and suggested != available:
        warnings.append(
            f"trainer will use {suggested} valid frames from {available} available"
        )

    bucket_value: str | tuple[int, int] | list[int] | Mapping[str, Any] = resolution_bucket
    if isinstance(resolution_bucket, str):
        configured = config.get("buckets", {}).get(resolution_bucket.casefold())
        if configured:
            bucket_value = {
                "width": configured[0],
                "height": configured[1],
                "multiple": config.get("resolution_multiple", 2),
            }
    if info.width and info.height:
        out_w, out_h, _ = resolution_bucket_preview(
            info.width, info.height, bucket_value, policy
        )
    else:
        out_w, out_h, _ = _bucket_dimensions(bucket_value)
    return FitnessReport(
        ok=available >= needed,
        frames_available=available,
        frames_needed=needed,
        suggested_frames=suggested,
        warnings=warnings,
        bucket=(out_w, out_h),
    )


def sub_split_plan(
    duration_s: float, target_seconds: float, overlap_s: float
) -> list[tuple[float, float]]:
    """Return fixed sub-clip boundaries with optional overlap."""

    duration = max(0.0, float(duration_s))
    target = float(target_seconds)
    overlap = max(0.0, float(overlap_s))
    if target <= 0:
        raise ValueError("target_seconds must be positive")
    if overlap >= target:
        raise ValueError("overlap_s must be smaller than target_seconds")
    result: list[tuple[float, float]] = []
    start = 0.0
    while start < duration - 1e-9:
        end = min(duration, start + target)
        result.append((start, end))
        if end >= duration - 1e-9:
            break
        start += target - overlap
    return result


__all__ = [
    "FitnessReport",
    "TRAINER_TARGETS",
    "evaluate_clip",
    "is_clip_fitness_plan_path",
    "resolution_bucket_preview",
    "sub_split_plan",
    "suggest_clip_length",
]
