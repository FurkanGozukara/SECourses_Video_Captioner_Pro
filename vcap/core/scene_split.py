"""Scene detection, segment planning, and verified FFmpeg clip splitting."""

from __future__ import annotations

import json
import math
import os
import subprocess
import threading
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Sequence

from .logs import AppLog, get_log
from .media import (
    MediaError,
    MediaInfo,
    extract_audio,
    extract_frames_to_files,
    find_ffprobe,
    probe_media,
    run_ffmpeg,
)
from .outputs import OutputWriter
from .paths import normalize_path, sanitize_filename
from .subprocess_runner import CancelToken, CancelledError

ProgressCallback = Callable[[float, str], None]
SPLIT_MANIFEST_NAME = "split_manifest.json"


@dataclass(frozen=True)
class SceneDetectParams:
    """Settings used for PySceneDetect and post-detection cleanup."""

    threshold: float = 27.0
    min_scene_len_s: float = 1.0
    max_scene_len_s: float = 0.0
    merge_short_scenes: bool = True
    merge_below_s: float = 1.0
    fade_detection: bool = False
    detector: Literal["content", "adaptive", "threshold"] = "content"
    downscale: int = 0
    start_s: float | None = None
    end_s: float | None = None


@dataclass(frozen=True)
class SceneRange:
    """Half-open time/frame range describing one planned clip."""

    start_s: float
    end_s: float
    start_frame: int | None = None
    end_frame: int | None = None

    @property
    def duration_s(self) -> float:
        """Return the non-negative scene duration in seconds."""

        return max(0.0, float(self.end_s) - float(self.start_s))


@dataclass(frozen=True)
class SegmentPlan:
    """Complete preprocessing segmentation decision for one source."""

    segments: list[SceneRange]
    warnings: list[str]
    source_duration: float


@dataclass(frozen=True)
class ClipInfo:
    """Verified information about a physically written split clip."""

    index: int
    path: Path
    start_s: float
    end_s: float
    expected_frames: int | None
    actual_frames: int | None
    mode_used: str


def _emit_progress(callback: ProgressCallback | None, fraction: float, message: str) -> None:
    if callback is None:
        return
    try:
        callback(min(1.0, max(0.0, float(fraction))), str(message))
    except Exception:
        pass


def _cancelled(cancel: object | None) -> bool:
    if cancel is None:
        return False
    for name in ("is_cancelled", "is_set"):
        method = getattr(cancel, name, None)
        if callable(method):
            try:
                return bool(method())
            except Exception:
                return False
    return False


def _check_cancel(cancel: object | None) -> None:
    if _cancelled(cancel):
        raise CancelledError("Scene/clip operation cancelled")


def _timecode_frames(value: Any) -> int | None:
    if value is None:
        return None
    for name in ("frame_num", "frames", "frame", "get_frames"):
        try:
            raw = getattr(value, name, None)
            if raw is None:
                continue
            frame = int(raw() if callable(raw) else raw)
            if frame >= 0:
                return frame
        except Exception:
            continue
    try:
        frame = int(value)
        return frame if frame >= 0 else None
    except (TypeError, ValueError):
        return None


def _timecode_seconds(value: Any) -> float:
    """Read a PySceneDetect timecode without using deprecated accessors."""

    raw = getattr(value, "seconds", None)
    if raw is not None:
        return float(raw)
    getter = getattr(value, "get_seconds", None)
    if callable(getter):
        return float(getter())
    return float(value)


def _as_scene(value: SceneRange | Sequence[float]) -> SceneRange:
    if isinstance(value, SceneRange):
        return value
    if len(value) < 2:
        raise ValueError("A segment needs start and end values")
    start_frame = int(value[2]) if len(value) > 2 and value[2] is not None else None
    end_frame = int(value[3]) if len(value) > 3 and value[3] is not None else None
    return SceneRange(float(value[0]), float(value[1]), start_frame, end_frame)


def _clean_scenes(scenes: Iterable[SceneRange | Sequence[float]]) -> list[SceneRange]:
    result: list[SceneRange] = []
    for value in scenes:
        try:
            scene = _as_scene(value)
        except (TypeError, ValueError, IndexError):
            continue
        if not (math.isfinite(scene.start_s) and math.isfinite(scene.end_s)):
            continue
        if scene.end_s <= scene.start_s:
            continue
        result.append(scene)
    return sorted(result, key=lambda item: (item.start_s, item.end_s))


def detect_scenes(
    video_path: str | os.PathLike[str],
    params: SceneDetectParams,
    progress_cb: ProgressCallback | None = None,
    cancel: CancelToken | None = None,
) -> list[SceneRange]:
    """Detect video scenes with PySceneDetect 0.7; bad inputs return an empty list."""

    log = get_log()
    try:
        from scenedetect import SceneManager, open_video
        from scenedetect.detectors import AdaptiveDetector, ContentDetector, ThresholdDetector

        source = normalize_path(video_path, must_exist=True)
        info = probe_media(source)
        if not info.has_video:
            raise MediaError(f"No video stream found in {source}")
        detector_name = str(params.detector).casefold()
        if detector_name not in {"content", "adaptive", "threshold"}:
            raise ValueError(f"Unknown scene detector: {params.detector}")

        video = open_video(str(source))
        fps = float(getattr(video, "frame_rate", None) or info.fps or 30.0)
        min_scene_frames = max(1, int(round(max(0.0, params.min_scene_len_s) * fps)))
        manager = SceneManager()
        if params.downscale > 0:
            manager.auto_downscale = False
            manager.downscale = max(1, int(params.downscale))
        else:
            manager.auto_downscale = True

        if detector_name == "adaptive":
            manager.add_detector(
                AdaptiveDetector(
                    adaptive_threshold=max(0.0, float(params.threshold)),
                    min_scene_len=min_scene_frames,
                )
            )
        elif detector_name == "threshold":
            manager.add_detector(
                ThresholdDetector(
                    threshold=max(0.0, float(params.threshold)),
                    min_scene_len=min_scene_frames,
                    fade_bias=0.0,
                    add_final_scene=True,
                )
            )
        else:
            manager.add_detector(
                ContentDetector(
                    threshold=max(0.0, float(params.threshold)),
                    min_scene_len=min_scene_frames,
                )
            )
        if params.fade_detection and detector_name != "threshold":
            manager.add_detector(
                ThresholdDetector(
                    threshold=12.0,
                    min_scene_len=min_scene_frames,
                    fade_bias=0.0,
                    add_final_scene=True,
                )
            )

        source_duration = float(info.duration or 0.0)
        if source_duration <= 0:
            try:
                source_duration = _timecode_seconds(getattr(video, "duration", None))
            except (TypeError, ValueError):
                source_duration = 0.0
        start_s = max(0.0, float(params.start_s or 0.0))
        end_s = float(params.end_s) if params.end_s is not None else source_duration
        if source_duration > 0:
            start_s = min(start_s, source_duration)
            end_s = min(max(start_s, end_s), source_duration)
        if end_s <= start_s:
            raise ValueError("Scene detection end must be greater than start")
        if start_s > 0:
            video.seek(start_s)

        total_frames = max(1, int(round((end_s - start_s) * fps)))
        seen_frames = 0
        last_fraction = -1.0
        poller_stop = threading.Event()

        def publish(value: Any = None, force: bool = False) -> None:
            nonlocal last_fraction
            _check_cancel(cancel)
            frame = _timecode_frames(value)
            if frame is None:
                frame = _timecode_frames(getattr(video, "position", None))
            if frame is None:
                frame = seen_frames
            relative = max(0, frame - int(round(start_s * fps)))
            fraction = min(0.99, relative / total_frames)
            if force or fraction - last_fraction >= 0.005:
                last_fraction = fraction
                _emit_progress(progress_cb, fraction, "Detecting scene cuts")

        original_read = getattr(video, "read", None)
        if callable(original_read):
            try:
                def read_with_progress(*args: Any, **kwargs: Any) -> Any:
                    nonlocal seen_frames
                    _check_cancel(cancel)
                    frame_data = original_read(*args, **kwargs)
                    has_frame = frame_data is not None
                    if isinstance(frame_data, tuple) and frame_data:
                        has_frame = bool(frame_data[0]) if isinstance(frame_data[0], bool) else frame_data[0] is not None
                    if has_frame:
                        seen_frames += 1
                        publish(int(round(start_s * fps)) + seen_frames)
                    return frame_data

                setattr(video, "read", read_with_progress)
            except Exception:
                pass

        def poll_position() -> None:
            while not poller_stop.wait(0.2):
                try:
                    publish()
                except CancelledError:
                    return

        poller = threading.Thread(target=poll_position, daemon=True, name="vcap-scene-progress")
        poller.start()

        def scene_callback(*args: Any, **kwargs: Any) -> None:
            _check_cancel(cancel)
            for value in (*args, *kwargs.values()):
                if _timecode_frames(value) is not None:
                    publish(value, force=True)
                    return
            publish(force=True)

        detect_kwargs: dict[str, Any] = {
            "video": video,
            "end_time": end_s,
            "show_progress": False,
            "callback": scene_callback,
        }
        try:
            manager.detect_scenes(**detect_kwargs)
        except TypeError as exc:
            if "callback" not in str(exc).casefold():
                raise
            detect_kwargs.pop("callback", None)
            manager.detect_scenes(**detect_kwargs)
        finally:
            poller_stop.set()
            poller.join(timeout=0.5)
        _check_cancel(cancel)

        ranges: list[SceneRange] = []
        for start_tc, end_tc in manager.get_scene_list(start_in_scene=True):
            detected_start = max(start_s, _timecode_seconds(start_tc))
            detected_end = min(end_s, _timecode_seconds(end_tc))
            if detected_end <= detected_start:
                continue
            ranges.append(
                SceneRange(
                    detected_start,
                    detected_end,
                    max(0, int(_timecode_frames(start_tc) or 0)),
                    max(0, int(_timecode_frames(end_tc) or 0)),
                )
            )
        if not ranges and end_s > start_s:
            ranges = [
                SceneRange(
                    start_s,
                    end_s,
                    int(round(start_s * fps)),
                    int(round(end_s * fps)),
                )
            ]
        if params.merge_short_scenes and params.merge_below_s > 0:
            ranges = merge_short_scenes(ranges, params.merge_below_s)
        if params.max_scene_len_s > 0:
            ranges = cap_scene_lengths(ranges, params.max_scene_len_s)
        _emit_progress(progress_cb, 1.0, f"Detected {len(ranges)} scene(s)")
        return ranges
    except CancelledError:
        raise
    except Exception as exc:
        log.warn(f"Scene detection failed for '{video_path}': {exc}", scope="scene")
        _emit_progress(progress_cb, 1.0, "Scene detection failed")
        return []


def merge_short_scenes(
    scenes: Iterable[SceneRange | Sequence[float]], min_len_s: float
) -> list[SceneRange]:
    """Merge ranges shorter than ``min_len_s`` into an adjacent scene."""

    result = _clean_scenes(scenes)
    minimum = max(0.0, float(min_len_s))
    if minimum <= 0 or len(result) < 2:
        return result
    while len(result) > 1:
        short_index = next(
            (index for index, scene in enumerate(result) if scene.duration_s + 1e-9 < minimum),
            None,
        )
        if short_index is None:
            break
        if short_index == 0:
            neighbor = 1
        elif short_index == len(result) - 1:
            neighbor = short_index - 1
        else:
            left_gap = abs(result[short_index].start_s - result[short_index - 1].end_s)
            right_gap = abs(result[short_index + 1].start_s - result[short_index].end_s)
            neighbor = short_index - 1 if left_gap <= right_gap else short_index + 1
        low, high = sorted((short_index, neighbor))
        first, second = result[low], result[high]
        merged = SceneRange(
            min(first.start_s, second.start_s),
            max(first.end_s, second.end_s),
            first.start_frame if first.start_s <= second.start_s else second.start_frame,
            second.end_frame if second.end_s >= first.end_s else first.end_frame,
        )
        result[low : high + 1] = [merged]
    return result


def fixed_length_segments(
    duration_s: float, chunk_s: float, overlap_s: float
) -> list[SceneRange]:
    """Build half-open fixed-duration segments covering a source from zero."""

    duration = max(0.0, float(duration_s))
    chunk = float(chunk_s)
    overlap = max(0.0, float(overlap_s))
    if chunk <= 0:
        raise ValueError("chunk_s must be positive")
    if overlap >= chunk:
        raise ValueError("overlap_s must be smaller than chunk_s")
    if duration <= 0:
        return []
    result: list[SceneRange] = []
    start = 0.0
    step = chunk - overlap
    while start < duration - 1e-9:
        end = min(duration, start + chunk)
        result.append(SceneRange(start, end))
        if end >= duration - 1e-9:
            break
        start += step
    return result


def cap_scene_lengths(
    scenes: Iterable[SceneRange | Sequence[float]],
    max_len_s: float,
    overlap_s: float = 0.0,
) -> list[SceneRange]:
    """Split long scenes into clips no longer than ``max_len_s``."""

    maximum = float(max_len_s)
    if maximum <= 0:
        return _clean_scenes(scenes)
    result: list[SceneRange] = []
    for scene in _clean_scenes(scenes):
        local = fixed_length_segments(scene.duration_s, maximum, overlap_s)
        fps = None
        if (
            scene.start_frame is not None
            and scene.end_frame is not None
            and scene.duration_s > 0
        ):
            fps = (scene.end_frame - scene.start_frame) / scene.duration_s
        for piece in local:
            start = scene.start_s + piece.start_s
            end = min(scene.end_s, scene.start_s + piece.end_s)
            start_frame = (
                int(round(scene.start_frame + piece.start_s * fps))
                if fps and scene.start_frame is not None
                else None
            )
            end_frame = (
                int(round(scene.start_frame + piece.end_s * fps))
                if fps and scene.start_frame is not None
                else None
            )
            result.append(SceneRange(start, end, start_frame, end_frame))
    return result


def enforce_model_limit(
    segments: Iterable[SceneRange | Sequence[float]],
    model_max_s: float | None,
    overlap_s: float,
    log: AppLog | str | None,
) -> tuple[list[SceneRange], list[str]]:
    """Re-split overlong clips and return user-facing limit warnings."""

    cleaned = _clean_scenes(segments)
    limit = float(model_max_s or 0.0)
    if limit <= 0:
        return cleaned, []
    output: list[SceneRange] = []
    warnings: list[str] = []
    label = str(log).strip() if isinstance(log, str) and str(log).strip() else "model"
    logger = log if hasattr(log, "warn") else None
    for index, scene in enumerate(cleaned, start=1):
        if scene.duration_s <= limit + 1e-9:
            output.append(scene)
            continue
        pieces = cap_scene_lengths([scene], limit, overlap_s)
        duration_label = f"{scene.duration_s:.1f}".rstrip("0").rstrip(".")
        limit_label = f"{limit:.1f}".rstrip("0").rstrip(".")
        warning = (
            f"Scene {index} ({duration_label} s) exceeds {label}'s {limit_label} s limit "
            f"-> split into {len(pieces)} clips"
        )
        warnings.append(warning)
        if logger is not None:
            try:
                logger.warn(warning, scope="scene")
            except Exception:
                pass
        output.extend(pieces)
    return output, warnings


def _trainer_seconds(trainer_target: Any) -> float | None:
    if trainer_target is None:
        return None
    if isinstance(trainer_target, (int, float)):
        return float(trainer_target) if float(trainer_target) > 0 else None
    if isinstance(trainer_target, dict):
        for key in ("target_seconds", "seconds", "duration_s"):
            if key in trainer_target and float(trainer_target[key]) > 0:
                return float(trainer_target[key])
        frames = trainer_target.get("frames") or trainer_target.get("default_frames") or trainer_target.get("target_frames")
        if isinstance(frames, (list, tuple)):
            frames = frames[0] if frames else None
        fps = trainer_target.get("fps") or trainer_target.get("default_fps") or trainer_target.get("target_fps")
        if frames and fps and float(fps) > 0:
            return float(frames) / float(fps)
    if isinstance(trainer_target, str):
        try:
            from .clip_fitness import TRAINER_TARGETS

            config = TRAINER_TARGETS.get(trainer_target.casefold())
            if config:
                return float(config["default_frames"]) / float(config["default_fps"])
        except Exception:
            return None
    return None


def plan_segments(
    media_info: MediaInfo,
    *,
    mode: Literal["whole", "scenes", "fixed", "trainer"] = "whole",
    scene_params: SceneDetectParams | None = None,
    fixed_chunk_s: float = 0.0,
    model_max_duration_s: float | None = None,
    trainer_target: Any = None,
    sub_split_overlap_s: float = 0.0,
    trim_start_s: float | None = None,
    trim_end_s: float | None = None,
) -> SegmentPlan:
    """Plan all clip boundaries for the shared pipeline in one place."""

    duration = float(media_info.duration or 0.0)
    if duration <= 0:
        return SegmentPlan([], ["Source duration is unavailable."], duration)
    start = min(duration, max(0.0, float(trim_start_s or 0.0)))
    end = duration if trim_end_s is None else min(duration, max(0.0, float(trim_end_s)))
    if end <= start:
        return SegmentPlan([], ["Trim end must be greater than trim start."], duration)
    selected_mode = str(mode).casefold()
    warnings: list[str] = []
    if selected_mode == "whole":
        fps = float(media_info.fps or 0.0)
        segments = [
            SceneRange(
                start,
                end,
                int(round(start * fps)) if fps > 0 else None,
                int(round(end * fps)) if fps > 0 else None,
            )
        ]
    elif selected_mode == "scenes":
        params = scene_params or SceneDetectParams()
        detection_params = replace(params, start_s=start, end_s=end)
        segments = detect_scenes(media_info.path, detection_params)
        if not segments:
            segments = [SceneRange(start, end)]
            warnings.append("Scene detection failed; using the selected range as one clip.")
    elif selected_mode in {"fixed", "trainer"}:
        chunk = float(fixed_chunk_s)
        if selected_mode == "trainer":
            chunk = _trainer_seconds(trainer_target) or chunk
        if chunk <= 0:
            segments = [SceneRange(start, end)]
            warnings.append(f"No valid {selected_mode} clip length was supplied; using one clip.")
        else:
            segments = [
                SceneRange(start + item.start_s, start + item.end_s)
                for item in fixed_length_segments(end - start, chunk, sub_split_overlap_s)
            ]
    else:
        raise ValueError("mode must be 'whole', 'scenes', 'fixed', or 'trainer'")
    segments, limit_warnings = enforce_model_limit(
        segments,
        model_max_duration_s,
        sub_split_overlap_s,
        "model",
    )
    warnings.extend(limit_warnings)
    return SegmentPlan(segments, warnings, duration)


def _fraction(value: Any) -> float | None:
    try:
        text = str(value or "").strip()
        if not text or text in {"N/A", "0/0"}:
            return None
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            return float(numerator) / float(denominator)
        return float(text)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _probe_video_details(path: Path) -> dict[str, Any]:
    ffprobe = find_ffprobe()
    if not ffprobe:
        return {}
    try:
        completed = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=r_frame_rate,avg_frame_rate,start_time,pix_fmt,nb_frames",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
        streams = json.loads(completed.stdout or "{}").get("streams") or []
        return streams[0] if completed.returncode == 0 and streams else {}
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return {}


def _probe_frame_count(path: Path) -> int | None:
    ffprobe = find_ffprobe()
    if not ffprobe or not path.is_file():
        return None
    commands = [
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "packet=flags",
            "-of",
            "csv=p=0",
            str(path),
        ],
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_packets",
            "-show_entries",
            "stream=nb_read_packets",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
    ]
    for index, command in enumerate(commands):
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
                check=False,
            )
            if completed.returncode != 0:
                continue
            if index == 0:
                flags = [line.strip().strip(",") for line in completed.stdout.splitlines()]
                count = sum(bool(flag) and "D" not in flag for flag in flags)
            else:
                raw = completed.stdout.strip()
                count = int(raw) if raw.isdigit() else 0
            if count > 0:
                return count
        except (OSError, subprocess.SubprocessError, ValueError):
            continue
    info = probe_media(path)
    return int(info.nb_frames) if info.nb_frames else None


def _reasonable_av_timing(path: Path) -> bool:
    """Reject copied audio that starts late or is severely shorter/longer than video."""

    ffprobe = find_ffprobe()
    if not ffprobe:
        return True
    try:
        completed = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "stream=codec_type,start_time,duration",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
        if completed.returncode != 0:
            return True
        timing: dict[str, tuple[float | None, float | None]] = {}
        for stream in json.loads(completed.stdout or "{}").get("streams") or []:
            kind = str(stream.get("codec_type") or "").casefold()
            if kind not in {"video", "audio"} or kind in timing:
                continue
            try:
                start = float(stream["start_time"]) if stream.get("start_time") not in {None, "N/A"} else None
            except (TypeError, ValueError):
                start = None
            try:
                duration = float(stream["duration"]) if stream.get("duration") not in {None, "N/A"} else None
            except (TypeError, ValueError):
                duration = None
            timing[kind] = (start, duration)
        if "audio" not in timing or "video" not in timing:
            return True
        video_start, video_duration = timing["video"]
        audio_start, audio_duration = timing["audio"]
        if video_start is not None and audio_start is not None and abs(audio_start - video_start) > 0.25:
            return False
        if video_duration is not None and audio_duration is not None:
            if audio_duration > video_duration + max(0.25, 0.05 * max(0.0, video_duration)):
                return False
            if audio_duration < max(0.0, video_duration - 1.0):
                return False
        return True
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return True


def _is_decodable(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        import av

        with av.open(str(path), mode="r") as container:
            if not container.streams.video:
                return False
            return next(container.decode(container.streams.video[0]), None) is not None
    except Exception:
        return False


def _normalize_split_ranges(
    segments: Iterable[SceneRange | Sequence[float]], fps: float
) -> list[SceneRange]:
    normalized: list[SceneRange] = []
    previous_end_us: int | None = None
    half_frame = 0.5 / fps if fps > 0 else 0.0005

    def snap_us(seconds: float) -> int:
        if fps > 0:
            frame = int(round(max(0.0, seconds) * fps))
            return int(round(frame * 1_000_000.0 / fps))
        return int(round(max(0.0, seconds) * 1_000_000.0))

    for scene in _clean_scenes(segments):
        start_us = snap_us(scene.start_s)
        end_us = snap_us(scene.end_s)
        if previous_end_us is not None and abs(scene.start_s - previous_end_us / 1_000_000.0) <= half_frame + 1e-6:
            start_us = previous_end_us
        if end_us <= start_us:
            end_us = start_us + (int(round(1_000_000.0 / fps)) if fps > 0 else 1000)
        start_frame = int(round(start_us * fps / 1_000_000.0)) if fps > 0 else scene.start_frame
        end_frame = int(round(end_us * fps / 1_000_000.0)) if fps > 0 else scene.end_frame
        normalized.append(
            SceneRange(start_us / 1_000_000.0, end_us / 1_000_000.0, start_frame, end_frame)
        )
        previous_end_us = end_us
    return normalized


def _output_name(template: str, index: int) -> str:
    try:
        rendered = str(template).format(index=index)
    except (KeyError, ValueError, IndexError) as exc:
        raise ValueError(f"Invalid name_template: {exc}") from exc
    rendered = sanitize_filename(Path(rendered).stem or f"clip_{index:04d}")
    return f"{rendered}.mp4"


def split_video(
    video_path: str | os.PathLike[str],
    segments: Iterable[SceneRange | Sequence[float]],
    out_dir: str | os.PathLike[str],
    *,
    mode: Literal["copy", "precise"] = "copy",
    keep_audio: bool = True,
    name_template: str = "clip_{index:04d}",
    encoder: str = "libx264",
    crf: int = 18,
    progress_cb: ProgressCallback | None = None,
    cancel: CancelToken | None = None,
) -> list[ClipInfo]:
    """Write verified clips, falling back from keyframe copy to precise encoding."""

    if mode not in {"copy", "precise"}:
        raise ValueError("mode must be 'copy' or 'precise'")
    source = normalize_path(video_path, must_exist=True)
    info = probe_media(source)
    if not info.has_video:
        raise MediaError(f"No video stream found in {source}")
    directory = normalize_path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    details = _probe_video_details(source)
    r_fps = _fraction(details.get("r_frame_rate"))
    avg_fps = _fraction(details.get("avg_frame_rate"))
    fps = float(avg_fps or info.fps or r_fps or 0.0)
    source_is_cfr = bool(r_fps and avg_fps and abs(r_fps - avg_fps) <= max(1e-6, 0.002 * r_fps))
    rational_fps = str(details.get("r_frame_rate") or "") if source_is_cfr else ""
    try:
        source_start = max(0.0, float(details.get("start_time") or 0.0))
    except (TypeError, ValueError):
        source_start = 0.0
    pix_fmt = str(details.get("pix_fmt") or "")
    planned = _normalize_split_ranges(segments, fps)
    if not planned:
        OutputWriter().write_json(directory / SPLIT_MANIFEST_NAME, {"source": str(source), "clips": []})
        return []

    written: list[ClipInfo] = []
    manifest_clips: list[dict[str, Any]] = []
    used_targets: set[Path] = set()
    total = len(planned)
    for index, segment in enumerate(planned, start=1):
        _check_cancel(cancel)
        target = directory / _output_name(name_template, index)
        if target in used_targets:
            target = target.with_name(f"{target.stem}_{index:04d}{target.suffix}")
        used_targets.add(target)
        duration = segment.duration_s
        if duration <= 0:
            continue
        expected = (
            segment.end_frame - segment.start_frame
            if segment.start_frame is not None
            and segment.end_frame is not None
            and segment.end_frame > segment.start_frame
            and (source_is_cfr or info.is_vfr is not True)
            else None
        )
        message = f"Splitting clip {index}/{total}"
        _emit_progress(progress_cb, (index - 1) / total, message)

        def progress(local: float) -> None:
            _emit_progress(progress_cb, (index - 1 + local) / total, message)

        def clear_target() -> None:
            try:
                target.unlink(missing_ok=True)
            except OSError:
                pass

        def run(arguments: list[str]) -> bool:
            clear_target()
            try:
                run_ffmpeg(
                    arguments,
                    progress_cb=progress,
                    total_duration=duration,
                    cancel_token=cancel,
                )
                return _is_decodable(target)
            except CancelledError:
                raise
            except Exception as exc:
                get_log().warn(f"Clip {index} split attempt failed: {exc}", scope="split")
                return False

        start_text = f"{segment.start_s:.6f}"
        duration_text = f"{duration:.6f}"

        def attempt_copy() -> tuple[bool, str]:
            base = ["-y", "-ss", start_text, "-i", str(source), "-t", duration_text]
            if keep_audio and info.has_audio:
                if run(base + ["-c", "copy", "-avoid_negative_ts", "make_zero", "-movflags", "+faststart", str(target)]):
                    if probe_media(target).has_audio and _reasonable_av_timing(target):
                        return True, "copy"
                if run(
                    base
                    + [
                        "-map",
                        "0:v:0",
                        "-map",
                        "0:a?",
                        "-c:v",
                        "copy",
                        "-c:a",
                        "aac",
                        "-b:a",
                        "192k",
                        "-avoid_negative_ts",
                        "make_zero",
                        "-movflags",
                        "+faststart",
                        str(target),
                    ]
                ) and _reasonable_av_timing(target):
                    return True, "copy+aac"
            if run(
                base
                + [
                    "-map",
                    "0:v:0",
                    "-c:v",
                    "copy",
                    "-an",
                    "-avoid_negative_ts",
                    "make_zero",
                    "-movflags",
                    "+faststart",
                    str(target),
                ]
            ):
                return True, "copy-video-only"
            return False, "copy-failed"

        def attempt_precise() -> tuple[bool, str]:
            seek = max(0.0, segment.start_s - 1.0)
            read_limit = duration + 2.0
            absolute_start = segment.start_s + source_start
            absolute_end = segment.end_s + source_start
            base = [
                "-y",
                "-ss",
                f"{seek:.6f}",
                "-t",
                f"{read_limit:.6f}",
                "-copyts",
                "-i",
                str(source),
                "-map",
                "0:v:0",
            ]
            video_args = [
                "-vf",
                f"trim=start={absolute_start:.6f}:end={absolute_end:.6f},setpts=PTS-STARTPTS",
            ]
            if rational_fps:
                video_args += ["-r", rational_fps]
            video_args += ["-c:v", str(encoder), "-preset", "veryfast", "-crf", str(max(0, min(51, int(crf))))]
            if pix_fmt and pix_fmt in {"yuv420p", "yuv420p10le", "yuv422p", "yuv444p"}:
                video_args += ["-pix_fmt", pix_fmt]
            else:
                video_args += ["-pix_fmt", "yuv420p"]
            if keep_audio and info.has_audio:
                audio_args = [
                    "-map",
                    "0:a?",
                    "-af",
                    f"atrim=start={absolute_start:.6f}:end={absolute_end:.6f},asetpts=PTS-STARTPTS",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                ]
                if run(base + audio_args + video_args + ["-movflags", "+faststart", str(target)]) and _reasonable_av_timing(target):
                    return True, "precise+aac"
            if run(base + video_args + ["-an", "-movflags", "+faststart", str(target)]):
                return True, "precise-video-only"
            return False, "precise-failed"

        def exact_enough() -> tuple[bool, int | None, str]:
            actual = _probe_frame_count(target)
            tolerance_frames = 1
            if expected is not None and actual is not None:
                difference = abs(actual - expected)
                return difference <= tolerance_frames, actual, f"{actual} frames vs expected {expected}"
            actual_duration = float(probe_media(target).duration or 0.0)
            tolerance_seconds = max(0.03, 0.75 / fps) if fps > 0 else 0.03
            return (
                abs(actual_duration - duration) <= tolerance_seconds,
                actual,
                f"{actual_duration:.3f}s vs expected {duration:.3f}s",
            )

        ok = False
        used = ""
        if mode == "copy":
            ok, used = attempt_copy()
            if ok:
                exact, _, reason = exact_enough()
                if not exact:
                    get_log().warn(
                        f"Clip {index} stream copy is inexact ({reason}); retrying precise encode.",
                        scope="split",
                    )
                    ok = False
            if not ok:
                ok, used = attempt_precise()
        else:
            ok, used = attempt_precise()
            if not ok:
                ok, used = attempt_copy()
        if not ok or not _is_decodable(target):
            raise MediaError(f"Could not split clip {index} ({segment.start_s:.3f}-{segment.end_s:.3f}s)")

        exact, actual, reason = exact_enough()
        if not exact:
            get_log().warn(f"Clip {index} verification tolerance exceeded: {reason}", scope="split")
        mode_used = "precise" if used.startswith("precise") else "copy"
        clip = ClipInfo(
            index=index,
            path=target,
            start_s=segment.start_s,
            end_s=segment.end_s,
            expected_frames=expected,
            actual_frames=actual,
            mode_used=mode_used,
        )
        written.append(clip)
        entry = asdict(clip)
        entry["path"] = target.name
        entry.update(
            {
                "file": target.name,
                "start_us": int(round(segment.start_s * 1_000_000)),
                "end_us": int(round(segment.end_s * 1_000_000)),
                "start_frame": segment.start_frame,
                "end_frame": segment.end_frame,
                "verified_exact": exact,
                "split_mode": mode_used,
                "audio_fallback": used,
            }
        )
        manifest_clips.append(entry)
        _emit_progress(progress_cb, index / total, f"Finished clip {index}/{total}")

    manifest = {
        "source": str(source),
        "source_duration": info.duration,
        "source_fps": fps or None,
        "source_fps_rational": rational_fps or None,
        "source_is_cfr": source_is_cfr,
        "mode_requested": mode,
        "keep_audio": bool(keep_audio),
        "encoder": str(encoder),
        "crf": int(crf),
        "clips": manifest_clips,
    }
    OutputWriter().write_json(directory / SPLIT_MANIFEST_NAME, manifest, pretty=True)
    return written


def extract_segment_audio(
    video_path: str | os.PathLike[str],
    segments: Iterable[SceneRange | Sequence[float]],
    out_dir: str | os.PathLike[str],
    sr: int = 16000,
    *,
    progress_cb: ProgressCallback | None = None,
    cancel: CancelToken | None = None,
) -> list[Path]:
    """Extract one mono PCM WAV file per segment."""

    source = normalize_path(video_path, must_exist=True)
    directory = normalize_path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    planned = _clean_scenes(segments)
    result: list[Path] = []
    for index, segment in enumerate(planned, start=1):
        _check_cancel(cancel)
        target = directory / f"audio_{index:04d}.wav"
        extract_audio(source, target, sample_rate=max(1, int(sr)), mono=True, start=segment.start_s, end=segment.end_s)
        result.append(target)
        _emit_progress(progress_cb, index / max(1, len(planned)), f"Extracted audio {index}/{len(planned)}")
    return result


def extract_segment_frames(
    video_path: str | os.PathLike[str],
    segments: Iterable[SceneRange | Sequence[float]],
    out_dir: str | os.PathLike[str],
    *,
    sampling: Literal["uniform", "fps", "keyframe"] = "uniform",
    target_fps: float | None = None,
    num_frames: int | None = None,
    max_frames: int | None = None,
    min_frames: int = 1,
    max_pixels: int | None = None,
    min_pixels: int | None = None,
    size_multiple: int = 28,
    image_format: Literal["png", "jpg", "jpeg"] = "png",
    progress_cb: ProgressCallback | None = None,
    cancel: CancelToken | None = None,
) -> list[list[Path]]:
    """Decode and save sampled frame files in one subdirectory per segment."""

    source = normalize_path(video_path, must_exist=True)
    directory = normalize_path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    planned = _clean_scenes(segments)
    result: list[list[Path]] = []
    for index, segment in enumerate(planned, start=1):
        _check_cancel(cancel)
        segment_dir = directory / f"segment_{index:04d}"
        paths = extract_frames_to_files(
            source,
            segment_dir,
            prefix="frame",
            image_format=image_format,
            start=segment.start_s,
            end=segment.end_s,
            sampling=sampling,
            target_fps=target_fps,
            num_frames=num_frames,
            max_frames=max_frames,
            min_frames=min_frames,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
            size_multiple=size_multiple,
            cancel_token=cancel,
        )
        result.append(paths)
        _emit_progress(progress_cb, index / max(1, len(planned)), f"Extracted frames {index}/{len(planned)}")
    return result


__all__ = [
    "ClipInfo",
    "SPLIT_MANIFEST_NAME",
    "SceneDetectParams",
    "SceneRange",
    "SegmentPlan",
    "cap_scene_lengths",
    "detect_scenes",
    "enforce_model_limit",
    "extract_segment_audio",
    "extract_segment_frames",
    "fixed_length_segments",
    "merge_short_scenes",
    "plan_segments",
    "split_video",
]
