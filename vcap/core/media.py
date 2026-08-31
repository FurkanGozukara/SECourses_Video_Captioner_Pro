"""FFmpeg, ffprobe, PyAV, audio, frame, preview, and thumbnail helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import shutil
import subprocess
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Sequence

import numpy as np
from PIL import Image, ImageOps

from .paths import guess_kind_by_extension, normalize_path, sanitize_filename


class MediaError(RuntimeError):
    """Raised when probing, decoding, or transforming media fails."""


@dataclass(frozen=True)
class MediaInfo:
    """Best-effort media stream and container metadata."""

    path: Path
    kind: Literal["video", "video_no_audio", "audio", "image", "text", "unknown"]
    duration: float | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    nb_frames: int | None = None
    has_video: bool = False
    has_audio: bool = False
    audio_sample_rate: int | None = None
    audio_channels: int | None = None
    video_codec: str | None = None
    audio_codec: str | None = None
    container: str | None = None
    size_bytes: int = 0
    is_vfr: bool | None = None
    rotation: int = 0
    error: str | None = None


@dataclass(frozen=True)
class VideoFrames:
    """Decoded RGB frames and their source timing/geometry."""

    frames: np.ndarray
    timestamps: list[float]
    fps_effective: float
    orig_size: tuple[int, int]
    resized_size: tuple[int, int]
    total_frames: int


def _common_windows_binary(name: str) -> Iterable[Path]:
    executable = f"{name}.exe"
    roots = [
        Path(r"C:\ffmpeg\bin"),
        Path(r"C:\Program Files\ffmpeg\bin"),
        Path(r"C:\Program Files (x86)\ffmpeg\bin"),
        Path(__file__).resolve().parents[2] / "ffmpeg" / "bin",
        Path(__file__).resolve().parents[3] / "ffmpeg" / "bin",
    ]
    for root in roots:
        yield root / executable


@lru_cache(maxsize=1)
def find_ffmpeg() -> str | None:
    """Locate FFmpeg on PATH, in common Windows folders, or via imageio-ffmpeg."""

    found = shutil.which("ffmpeg")
    if found:
        return found
    if os.name == "nt":
        for candidate in _common_windows_binary("ffmpeg"):
            if candidate.is_file():
                return str(candidate)
    try:
        import imageio_ffmpeg

        candidate = imageio_ffmpeg.get_ffmpeg_exe()
        if candidate and Path(candidate).is_file():
            return str(candidate)
    except Exception:
        pass
    return None


@lru_cache(maxsize=1)
def find_ffprobe() -> str | None:
    """Locate ffprobe beside FFmpeg, on PATH, or in common Windows folders."""

    found = shutil.which("ffprobe")
    if found:
        return found
    ffmpeg = find_ffmpeg()
    if ffmpeg:
        sibling = Path(ffmpeg).with_name("ffprobe.exe" if os.name == "nt" else "ffprobe")
        if sibling.is_file():
            return str(sibling)
    if os.name == "nt":
        for candidate in _common_windows_binary("ffprobe"):
            if candidate.is_file():
                return str(candidate)
    return None


def _fraction(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text or text in {"N/A", "0/0"}:
        return None
    try:
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            divisor = float(denominator)
            return float(numerator) / divisor if divisor else None
        return float(text)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _positive_float(*values: Any) -> float | None:
    for value in values:
        try:
            parsed = float(value)
            if math.isfinite(parsed) and parsed >= 0:
                return parsed
        except (TypeError, ValueError):
            continue
    return None


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
        return parsed if parsed >= 0 else None
    except (TypeError, ValueError):
        return None


def _rotation_for_stream(stream: dict[str, Any]) -> int:
    value: Any = stream.get("tags", {}).get("rotate") if isinstance(stream.get("tags"), dict) else None
    if value is None:
        for side_data in stream.get("side_data_list", []) or []:
            if isinstance(side_data, dict) and side_data.get("rotation") is not None:
                value = side_data.get("rotation")
                break
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return 0


def _is_primary_video_stream(stream: object) -> bool:
    if not isinstance(stream, dict) or stream.get("codec_type") != "video":
        return False
    disposition = stream.get("disposition")
    attached = disposition.get("attached_pic", 0) if isinstance(disposition, dict) else 0
    try:
        return not bool(int(attached or 0))
    except (TypeError, ValueError):
        return True


def _unknown_info(path: Path, error: object) -> MediaInfo:
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    return MediaInfo(path=path, kind="unknown", size_bytes=size, error=str(error))


def probe_media(path: str | os.PathLike[str]) -> MediaInfo:
    """Probe a file without raising; invalid inputs return ``kind='unknown'``."""

    try:
        target = normalize_path(path)
    except Exception as exc:
        return _unknown_info(Path(str(path)), exc)
    if not target.is_file():
        return _unknown_info(target, "File does not exist")
    try:
        size = target.stat().st_size
    except OSError:
        size = 0
    extension_kind = guess_kind_by_extension(target)

    if extension_kind == "text":
        return MediaInfo(path=target, kind="text", size_bytes=size)
    if extension_kind == "image":
        try:
            with Image.open(target) as image:
                transposed = ImageOps.exif_transpose(image)
                width, height = transposed.size
                container = str(image.format or target.suffix.lstrip(".")).casefold()
            return MediaInfo(
                path=target,
                kind="image",
                width=width,
                height=height,
                container=container,
                size_bytes=size,
            )
        except Exception as exc:
            return _unknown_info(target, exc)

    ffprobe = find_ffprobe()
    if not ffprobe:
        return _unknown_info(target, "ffprobe was not found")
    try:
        completed = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(target),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
        if completed.returncode != 0:
            tail = (completed.stderr or "ffprobe failed").strip()[-2000:]
            return _unknown_info(target, tail)
        payload = json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        return _unknown_info(target, exc)

    payload_dict = payload if isinstance(payload, dict) else {}
    streams = payload_dict.get("streams", [])
    if not isinstance(streams, list):
        streams = []
    video = next(
        (stream for stream in streams if _is_primary_video_stream(stream)),
        None,
    )
    audio = next(
        (stream for stream in streams if isinstance(stream, dict) and stream.get("codec_type") == "audio"),
        None,
    )
    raw_format = payload_dict.get("format")
    format_data = raw_format if isinstance(raw_format, dict) else {}
    has_video, has_audio = video is not None, audio is not None
    if has_video:
        kind: Literal["video", "video_no_audio", "audio", "image", "text", "unknown"] = (
            "video" if has_audio else "video_no_audio"
        )
    elif has_audio:
        kind = "audio"
    else:
        return _unknown_info(target, "No audio or video streams found")

    duration = _positive_float(
        format_data.get("duration"),
        video.get("duration") if video else None,
        audio.get("duration") if audio else None,
    )
    avg_fps = _fraction(video.get("avg_frame_rate")) if video else None
    real_fps = _fraction(video.get("r_frame_rate")) if video else None
    fps = avg_fps or real_fps
    is_vfr: bool | None = None
    if avg_fps is not None and real_fps is not None:
        is_vfr = abs(avg_fps - real_fps) > max(0.01, real_fps * 0.001)
    return MediaInfo(
        path=target,
        kind=kind,
        duration=duration,
        width=_positive_int(video.get("width")) if video else None,
        height=_positive_int(video.get("height")) if video else None,
        fps=fps,
        nb_frames=_positive_int(video.get("nb_frames")) if video else None,
        has_video=has_video,
        has_audio=has_audio,
        audio_sample_rate=_positive_int(audio.get("sample_rate")) if audio else None,
        audio_channels=_positive_int(audio.get("channels")) if audio else None,
        video_codec=str(video.get("codec_name")) if video and video.get("codec_name") else None,
        audio_codec=str(audio.get("codec_name")) if audio and audio.get("codec_name") else None,
        container=str(format_data.get("format_name")) if format_data.get("format_name") else None,
        size_bytes=size,
        is_vfr=is_vfr,
        rotation=_rotation_for_stream(video) if video else 0,
    )


def _is_cancelled(token: object | None) -> bool:
    if token is None:
        return False
    for name in ("is_cancelled", "is_set"):
        method = getattr(token, name, None)
        if callable(method):
            try:
                return bool(method())
            except Exception:
                return False
    return False


def _raise_cancelled(message: str) -> None:
    try:
        from .subprocess_runner import CancelledError

        raise CancelledError(message)
    except ImportError as exc:
        raise MediaError(message) from exc


def _parse_ffmpeg_clock(value: str) -> float | None:
    try:
        hours, minutes, seconds = value.split(":", 2)
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except (TypeError, ValueError):
        return None


def run_ffmpeg(
    args: Sequence[str | os.PathLike[str]],
    progress_cb: Callable[[float], None] | None = None,
    total_duration: float | None = None,
    cancel_token: object | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run FFmpeg, parse machine progress, cancel its tree, and raise useful errors."""

    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise MediaError("ffmpeg was not found")
    command = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-nostats",
        "-progress",
        "pipe:1",
        *[os.fspath(argument) for argument in args],
    ]
    popen_options: dict[str, Any] = {}
    if os.name == "nt":
        popen_options["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        popen_options["start_new_session"] = True
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            **popen_options,
        )
    except OSError as exc:
        raise MediaError(f"Could not start ffmpeg: {exc}") from exc
    output_queue: queue.Queue[str | None] = queue.Queue()
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    def read_stdout() -> None:
        assert process.stdout is not None
        try:
            for line in process.stdout:
                output_queue.put(line)
        finally:
            output_queue.put(None)

    def read_stderr() -> None:
        assert process.stderr is not None
        for line in process.stderr:
            stderr_lines.append(line)

    stdout_thread = threading.Thread(target=read_stdout, daemon=True, name="vcap-ffmpeg-stdout")
    stderr_thread = threading.Thread(target=read_stderr, daemon=True, name="vcap-ffmpeg-stderr")
    stdout_thread.start()
    stderr_thread.start()
    stream_done = False
    cancelled = False
    while not stream_done or process.poll() is None:
        if _is_cancelled(cancel_token):
            cancelled = True
            try:
                from .subprocess_runner import kill_process_tree

                kill_process_tree(process.pid, grace=0.5)
            except Exception:
                process.kill()
            break
        try:
            line = output_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if line is None:
            stream_done = True
            continue
        stdout_lines.append(line)
        key, separator, raw_value = line.strip().partition("=")
        if not separator:
            continue
        seconds: float | None = None
        if key in {"out_time_us", "out_time_ms"}:
            try:
                seconds = int(raw_value) / 1_000_000.0
            except ValueError:
                pass
        elif key == "out_time":
            seconds = _parse_ffmpeg_clock(raw_value)
        if seconds is not None and progress_cb is not None and total_duration and total_duration > 0:
            try:
                progress_cb(min(1.0, max(0.0, seconds / total_duration)))
            except Exception:
                pass
        if key == "progress" and raw_value == "end" and progress_cb is not None:
            try:
                progress_cb(1.0)
            except Exception:
                pass

    if cancelled:
        process.wait(timeout=5)
    else:
        process.wait()
    stdout_thread.join(timeout=2)
    stderr_thread.join(timeout=2)
    stdout = "".join(stdout_lines)
    stderr = "".join(stderr_lines)
    completed = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    if cancelled:
        _raise_cancelled("FFmpeg operation cancelled")
    if process.returncode != 0:
        tail = "\n".join(stderr.splitlines()[-40:])[-5000:]
        raise MediaError(f"FFmpeg failed with exit code {process.returncode}:\n{tail}")
    return completed


def extract_audio(
    src: str | os.PathLike[str],
    dst_wav: str | os.PathLike[str],
    sample_rate: int = 16000,
    mono: bool = True,
    start: float | None = None,
    end: float | None = None,
) -> Path:
    """Extract PCM WAV audio with optional time bounds."""

    source = normalize_path(src, must_exist=True)
    target = normalize_path(dst_wav)
    target.parent.mkdir(parents=True, exist_ok=True)
    arguments: list[str] = ["-y"]
    if start is not None:
        arguments.extend(["-ss", f"{max(0.0, float(start)):.6f}"])
    arguments.extend(["-i", str(source)])
    if end is not None:
        beginning = max(0.0, float(start or 0.0))
        duration = float(end) - beginning
        if duration <= 0:
            raise ValueError("end must be greater than start")
        arguments.extend(["-t", f"{duration:.6f}"])
    arguments.extend(
        [
            "-vn",
            "-ac",
            "1" if mono else "2",
            "-ar",
            str(max(1, int(sample_rate))),
            "-c:a",
            "pcm_s16le",
            str(target),
        ]
    )
    run_ffmpeg(arguments, total_duration=(float(end) - float(start or 0)) if end is not None else None)
    return target


def read_audio(path: str | os.PathLike[str], sample_rate: int = 16000) -> np.ndarray:
    """Decode an audio stream to mono float32 samples through an FFmpeg pipe."""

    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise MediaError("ffmpeg was not found")
    target = normalize_path(path, must_exist=True)
    completed = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(target),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(max(1, int(sample_rate))),
            "-f",
            "f32le",
            "pipe:1",
        ],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        error = completed.stderr.decode("utf-8", errors="replace")[-3000:]
        raise MediaError(f"Could not decode audio: {error}")
    return np.frombuffer(completed.stdout, dtype="<f4").astype(np.float32, copy=True)


def trim_media(
    src: str | os.PathLike[str],
    dst: str | os.PathLike[str],
    start: float,
    end: float,
    mode: Literal["copy", "precise"] = "copy",
    keep_audio: bool = True,
) -> Path:
    """Trim media by stream copy or frame-accurate re-encoding."""

    source = normalize_path(src, must_exist=True)
    target = normalize_path(dst)
    target.parent.mkdir(parents=True, exist_ok=True)
    beginning = max(0.0, float(start))
    duration = float(end) - beginning
    if duration <= 0:
        raise ValueError("end must be greater than start")
    if mode not in {"copy", "precise"}:
        raise ValueError("mode must be 'copy' or 'precise'")
    if mode == "copy":
        arguments = ["-y", "-ss", f"{beginning:.6f}", "-i", str(source), "-t", f"{duration:.6f}"]
        if keep_audio:
            arguments.extend(["-map", "0:v?", "-map", "0:a?"])
        else:
            arguments.extend(["-map", "0:v?", "-an"])
        arguments.extend(["-c", "copy", "-avoid_negative_ts", "make_zero", str(target)])
    else:
        info = probe_media(source)
        arguments = ["-y", "-i", str(source), "-ss", f"{beginning:.6f}", "-t", f"{duration:.6f}"]
        if info.has_video:
            arguments.extend(["-map", "0:v:0", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18", "-pix_fmt", "yuv420p"])
        else:
            arguments.append("-vn")
        if keep_audio and info.has_audio:
            arguments.extend(["-map", "0:a:0?"])
            audio_suffix = target.suffix.casefold()
            if audio_suffix == ".wav":
                arguments.extend(["-c:a", "pcm_s16le"])
            elif audio_suffix == ".mp3":
                arguments.extend(["-c:a", "libmp3lame", "-q:a", "2"])
            elif audio_suffix == ".flac":
                arguments.extend(["-c:a", "flac"])
            else:
                arguments.extend(["-c:a", "aac", "-b:a", "192k"])
        else:
            arguments.append("-an")
        if target.suffix.casefold() == ".mp4":
            arguments.extend(["-movflags", "+faststart"])
        arguments.append(str(target))
    run_ffmpeg(arguments, total_duration=duration)
    return target


def _round_to_factor(value: int, factor: int, mode: str = "nearest") -> int:
    factor = max(1, int(factor))
    if mode == "floor":
        return max(factor, math.floor(value / factor) * factor)
    if mode == "ceil":
        return max(factor, math.ceil(value / factor) * factor)
    return max(factor, round(value / factor) * factor)


def _smart_resize(
    height: int,
    width: int,
    factor: int,
    min_pixels: int | None,
    max_pixels: int | None,
) -> tuple[int, int]:
    factor = max(1, int(factor))
    minimum = max(1, int(min_pixels)) if min_pixels is not None else None
    maximum = max(1, int(max_pixels)) if max_pixels is not None else None
    if minimum is not None and maximum is not None and maximum < minimum:
        raise ValueError("max_pixels must be greater than or equal to min_pixels")
    if maximum is not None and maximum < factor * factor:
        raise ValueError("max_pixels is too small for one size_multiple tile")
    resized_h = _round_to_factor(height, factor)
    resized_w = _round_to_factor(width, factor)
    if maximum is not None and resized_h * resized_w > maximum:
        scale = math.sqrt((height * width) / maximum)
        resized_h = _round_to_factor(max(1, int(height / scale)), factor, "floor")
        resized_w = _round_to_factor(max(1, int(width / scale)), factor, "floor")
    if minimum is not None and resized_h * resized_w < minimum:
        scale = math.sqrt(minimum / (height * width))
        resized_h = _round_to_factor(max(1, int(math.ceil(height * scale))), factor, "ceil")
        resized_w = _round_to_factor(max(1, int(math.ceil(width * scale))), factor, "ceil")
    area = resized_h * resized_w
    if (minimum is None or area >= minimum) and (maximum is None or area <= maximum):
        return resized_h, resized_w

    # Very narrow media can make the direct Qwen rounding cross a pixel bound.
    # Find the closest feasible pair of integer tiles while minimizing aspect drift.
    tile_area = factor * factor
    minimum_tiles = max(1, math.ceil((minimum or 1) / tile_area))
    maximum_tiles = math.floor((maximum or max(area, minimum or area)) / tile_area)
    if maximum_tiles < minimum_tiles:
        raise ValueError("No dimensions divisible by size_multiple fit the pixel bounds")
    aspect = width / max(1, height)
    desired_area = min(max(height * width, minimum or 1), maximum or height * width)
    best: tuple[float, int, int] | None = None
    for height_tiles in range(1, maximum_tiles + 1):
        low_width = max(1, math.ceil(minimum_tiles / height_tiles))
        high_width = maximum_tiles // height_tiles
        if low_width > high_width:
            continue
        ideal_width = aspect * height_tiles
        width_tiles = min(high_width, max(low_width, int(round(ideal_width))))
        candidate_area = height_tiles * width_tiles * tile_area
        candidate_aspect = width_tiles / height_tiles
        aspect_error = abs(math.log(max(candidate_aspect, 1e-12) / max(aspect, 1e-12)))
        area_error = abs(math.log(candidate_area / max(desired_area, 1)))
        score = aspect_error * 10.0 + area_error
        if best is None or score < best[0]:
            best = (score, height_tiles, width_tiles)
    if best is None:
        raise ValueError("Could not find resize dimensions within the pixel bounds")
    resized_h, resized_w = best[1] * factor, best[2] * factor
    return resized_h, resized_w


def _rounded_frame_count(
    count: int,
    factor: int | None,
    minimum: int,
    maximum: int | None,
) -> int:
    value = max(minimum, int(count))
    if factor and factor > 1:
        value = _round_to_factor(value, factor)
    if maximum is not None:
        cap = max(1, int(maximum))
        if factor and factor > 1 and cap >= factor:
            cap = _round_to_factor(cap, factor, "floor")
        value = min(value, cap)
    return max(1, value)


def _uniform_indexes(length: int, count: int) -> list[int]:
    if length <= 0 or count <= 0:
        return []
    if count >= length:
        return list(range(length))
    return [int(index) for index in np.linspace(0, length - 1, count).round()]


def _frame_timestamp(frame: Any, stream: Any, fallback: float) -> float:
    timestamp = frame.time
    if timestamp is None and frame.pts is not None and stream.time_base is not None:
        timestamp = float(frame.pts * stream.time_base)
    return float(fallback if timestamp is None else timestamp)


def _seek_decode_targets(
    av_module: Any,
    source: Path,
    desired_times: Sequence[float],
    *,
    beginning: float,
    ending: float | None,
    source_fps: float,
    cancel_token: object | None,
) -> tuple[list[np.ndarray], list[float], int, int, int, int, int]:
    """Seek to keyframes and decode forward only far enough for each target."""

    arrays: list[np.ndarray] = []
    timestamps: list[float] = []
    decoded_count = 0
    with av_module.open(str(source), mode="r") as container:
        if not container.streams.video:
            raise MediaError(f"No video stream found in {source}")
        stream = container.streams.video[0]
        stream_frames = int(stream.frames or 0)
        original_width = int(stream.codec_context.width or 0)
        original_height = int(stream.codec_context.height or 0)
        tolerance = 0.5 / max(source_fps, 1.0)
        for target in desired_times:
            if _is_cancelled(cancel_token):
                _raise_cancelled("Video frame decoding cancelled")
            if stream.time_base is not None:
                try:
                    container.seek(
                        max(0, int(float(target) / float(stream.time_base))),
                        stream=stream,
                        backward=True,
                        any_frame=False,
                    )
                except Exception:
                    container.seek(0, backward=True, any_frame=False)
            selected: Any | None = None
            selected_time: float | None = None
            previous: Any | None = None
            previous_time: float | None = None
            for offset, frame in enumerate(container.decode(stream)):
                if _is_cancelled(cancel_token):
                    _raise_cancelled("Video frame decoding cancelled")
                decoded_count += 1
                timestamp = _frame_timestamp(
                    frame,
                    stream,
                    beginning + offset / max(source_fps, 1.0),
                )
                if timestamp + 1e-6 < beginning:
                    continue
                if ending is not None and timestamp >= ending + 1e-6:
                    break
                previous, previous_time = frame, timestamp
                if timestamp + tolerance >= float(target):
                    selected, selected_time = frame, timestamp
                    break
            if selected is None:
                selected, selected_time = previous, previous_time
            if selected is None:
                if arrays:
                    arrays.append(arrays[-1])
                    timestamps.append(timestamps[-1])
                continue
            arrays.append(selected.to_ndarray(format="rgb24"))
            timestamps.append(float(selected_time if selected_time is not None else target))
    return (
        arrays,
        timestamps,
        decoded_count,
        stream_frames,
        original_width,
        original_height,
        len(desired_times),
    )


def _adaptive_frame_indexes(arrays: Sequence[np.ndarray], threshold: float = 2.0) -> list[int]:
    """Keep visually distinct frames using a small grayscale mean-difference test."""

    if not arrays:
        return []

    def thumbnail(array: np.ndarray) -> np.ndarray:
        image = Image.fromarray(array, mode="RGB").convert("L")
        image.thumbnail((64, 64), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.float32)

    kept = [0]
    previous = thumbnail(arrays[0])
    for index, array in enumerate(arrays[1:], start=1):
        current = thumbnail(array)
        difference = float(np.mean(np.abs(current - previous)))
        if difference >= float(threshold):
            kept.append(index)
            previous = current
    if len(kept) < 2 and len(arrays) >= 2:
        kept.append(len(arrays) - 1)
    return sorted(set(kept))


def read_video_frames(
    path: str | os.PathLike[str],
    *,
    start: float | None = None,
    end: float | None = None,
    target_fps: float | None = None,
    start_s: float | None = None,
    end_s: float | None = None,
    fps: float | None = None,
    num_frames: int | None = None,
    max_frames: int | None = None,
    min_frames: int = 1,
    max_pixels: int | None = None,
    min_pixels: int | None = None,
    size_multiple: int = 28,
    keep_aspect: bool = True,
    sampling: Literal["uniform", "fps", "keyframe", "adaptive"] = "uniform",
    cancel_token: object | None = None,
    round_frames_to: int | None = None,
) -> VideoFrames:
    """Decode sampled RGB frames with Qwen-style smart resizing."""

    del keep_aspect  # Pixel-area scaling has no alternate target aspect to apply.
    if start_s is not None:
        if start is not None and not math.isclose(float(start), float(start_s)):
            raise ValueError("start and start_s specify different values")
        start = float(start_s)
    if end_s is not None:
        if end is not None and not math.isclose(float(end), float(end_s)):
            raise ValueError("end and end_s specify different values")
        end = float(end_s)
    if fps is not None:
        if target_fps is not None and not math.isclose(float(target_fps), float(fps)):
            raise ValueError("target_fps and fps specify different values")
        target_fps = float(fps)
    source = normalize_path(path, must_exist=True)
    if sampling not in {"uniform", "fps", "keyframe", "adaptive"}:
        raise ValueError("sampling must be 'uniform', 'fps', 'keyframe', or 'adaptive'")
    minimum_count = max(2 if sampling == "adaptive" else 1, int(min_frames))
    if max_frames is not None and int(max_frames) < minimum_count:
        raise ValueError("max_frames must be greater than or equal to min_frames")
    info = probe_media(source)
    if not info.has_video:
        raise MediaError(f"No video stream found in {source}")
    try:
        import av
    except ImportError as exc:
        raise MediaError("PyAV is required to decode video frames") from exc

    beginning = max(0.0, float(start or 0.0))
    ending = float(end) if end is not None else info.duration
    if info.duration is not None and ending is not None:
        ending = min(float(ending), float(info.duration))
    if ending is None and info.nb_frames and info.fps:
        ending = beginning + max(0.0, float(info.nb_frames) / float(info.fps))
    if ending is not None and ending <= beginning:
        raise ValueError("end must be greater than start")
    source_fps = float(info.fps or 0.0)
    requested_count: int | None = None
    if num_frames is not None:
        requested_count = _rounded_frame_count(num_frames, round_frames_to, minimum_count, max_frames)
    elif sampling == "uniform" and max_frames is not None:
        requested_count = _rounded_frame_count(max_frames, round_frames_to, minimum_count, max_frames)

    desired_times: list[float] | None = None
    if ending is not None and ending > beginning:
        last_time = max(beginning, ending - (0.5 / source_fps if source_fps > 0 else 1e-6))
        if sampling == "uniform" and requested_count is not None:
            desired_times = [
                float(value)
                for value in np.linspace(beginning, last_time, requested_count)
            ]
        elif sampling in {"fps", "adaptive"}:
            fps = float(target_fps or source_fps or 1.0)
            if fps <= 0:
                raise ValueError("target_fps must be positive")
            count = max(minimum_count, int(math.ceil((ending - beginning) * fps)))
            if num_frames is not None:
                count = min(count, max(1, int(num_frames)))
            count = _rounded_frame_count(count, round_frames_to, minimum_count, max_frames)
            desired_times = [float(value) for value in np.linspace(beginning, last_time, count)]

    decoded_arrays: list[np.ndarray] = []
    decoded_times: list[float] = []
    decoded_count = 0
    last_array: np.ndarray | None = None
    last_timestamp = beginning
    if sampling == "uniform" and desired_times is not None:
        (
            decoded_arrays,
            decoded_times,
            decoded_count,
            stream_frame_count,
            original_width,
            original_height,
            _,
        ) = _seek_decode_targets(
            av,
            source,
            desired_times,
            beginning=beginning,
            ending=ending,
            source_fps=source_fps,
            cancel_token=cancel_token,
        )
        original_width = int(original_width or info.width or 0)
        original_height = int(original_height or info.height or 0)
    else:
        with av.open(str(source), mode="r") as container:
            if not container.streams.video:
                raise MediaError(f"No video stream found in {source}")
            stream = container.streams.video[0]
            if sampling == "keyframe":
                try:
                    stream.codec_context.skip_frame = "NONKEY"
                except Exception:
                    pass
            if beginning > 0 and stream.time_base:
                try:
                    container.seek(
                        max(0, int(beginning / float(stream.time_base))),
                        stream=stream,
                        backward=True,
                        any_frame=False,
                    )
                except Exception:
                    pass
            target_index = 0
            for frame in container.decode(stream):
                if _is_cancelled(cancel_token):
                    _raise_cancelled("Video frame decoding cancelled")
                decoded_count += 1
                timestamp = _frame_timestamp(
                    frame,
                    stream,
                    beginning + (decoded_count - 1) / max(source_fps, 1.0),
                )
                if timestamp + 1e-6 < beginning:
                    continue
                if ending is not None and timestamp >= ending + 1e-6:
                    break
                array = frame.to_ndarray(format="rgb24")
                last_array, last_timestamp = array, timestamp
                if desired_times is None:
                    decoded_arrays.append(array)
                    decoded_times.append(timestamp)
                else:
                    tolerance = 0.5 / max(source_fps, target_fps or 0.0, 1.0)
                    while target_index < len(desired_times) and timestamp + tolerance >= desired_times[target_index]:
                        decoded_arrays.append(array)
                        decoded_times.append(timestamp)
                        target_index += 1
                    if target_index >= len(desired_times):
                        break
            if desired_times is not None and last_array is not None:
                while len(decoded_arrays) < len(desired_times):
                    decoded_arrays.append(last_array)
                    decoded_times.append(last_timestamp)

            stream_frame_count = int(stream.frames or 0)
            original_width = int(stream.codec_context.width or info.width or 0)
            original_height = int(stream.codec_context.height or info.height or 0)

    if sampling == "uniform" and desired_times is not None and decoded_arrays:
        while len(decoded_arrays) < len(desired_times):
            decoded_arrays.append(decoded_arrays[-1])
            decoded_times.append(decoded_times[-1])

    if sampling == "keyframe" and len(decoded_arrays) < 2 and ending is not None:
        fallback_count = requested_count
        if fallback_count is None:
            fallback_count = _rounded_frame_count(
                num_frames or max_frames or max(2, minimum_count),
                round_frames_to,
                minimum_count,
                max_frames,
            )
        last_time = max(beginning, ending - (0.5 / source_fps if source_fps > 0 else 1e-6))
        fallback_times = [
            float(value) for value in np.linspace(beginning, last_time, fallback_count)
        ]
        (
            decoded_arrays,
            decoded_times,
            fallback_decoded,
            fallback_stream_frames,
            fallback_width,
            fallback_height,
            _,
        ) = _seek_decode_targets(
            av,
            source,
            fallback_times,
            beginning=beginning,
            ending=ending,
            source_fps=source_fps,
            cancel_token=cancel_token,
        )
        decoded_count += fallback_decoded
        stream_frame_count = stream_frame_count or fallback_stream_frames
        original_width = original_width or fallback_width
        original_height = original_height or fallback_height

    if sampling == "adaptive" and decoded_arrays:
        indexes = _adaptive_frame_indexes(decoded_arrays)
        decoded_arrays = [decoded_arrays[index] for index in indexes]
        decoded_times = [decoded_times[index] for index in indexes]

    if desired_times is None:
        cap = requested_count
        if sampling in {"fps", "adaptive"} and decoded_arrays:
            fps = float(target_fps or source_fps or 1.0)
            count = max(minimum_count, int(math.ceil(len(decoded_arrays) * fps / max(source_fps, fps))))
            cap = _rounded_frame_count(count, round_frames_to, minimum_count, max_frames)
        elif sampling == "keyframe" and num_frames is not None:
            cap = _rounded_frame_count(num_frames, round_frames_to, minimum_count, max_frames)
        elif max_frames is not None:
            cap = min(len(decoded_arrays), int(max_frames))
        if cap is not None and len(decoded_arrays) > cap:
            indexes = _uniform_indexes(len(decoded_arrays), cap)
            decoded_arrays = [decoded_arrays[index] for index in indexes]
            decoded_times = [decoded_times[index] for index in indexes]

    if not decoded_arrays:
        raise MediaError(f"No frames could be decoded from {source}")
    while len(decoded_arrays) < minimum_count:
        decoded_arrays.append(decoded_arrays[-1])
        decoded_times.append(decoded_times[-1])
    if round_frames_to and round_frames_to > 1 and len(decoded_arrays) % round_frames_to:
        target_count = _rounded_frame_count(
            len(decoded_arrays), round_frames_to, minimum_count, max_frames
        )
        if target_count < len(decoded_arrays):
            indexes = _uniform_indexes(len(decoded_arrays), target_count)
            decoded_arrays = [decoded_arrays[index] for index in indexes]
            decoded_times = [decoded_times[index] for index in indexes]
        else:
            while len(decoded_arrays) < target_count:
                decoded_arrays.append(decoded_arrays[-1])
                decoded_times.append(decoded_times[-1])

    if original_width <= 0 or original_height <= 0:
        original_height, original_width = decoded_arrays[0].shape[:2]
    resized_h, resized_w = _smart_resize(
        original_height,
        original_width,
        max(1, int(size_multiple)),
        min_pixels,
        max_pixels,
    )
    resized_arrays: list[np.ndarray] = []
    for array in decoded_arrays:
        if array.shape[1] == resized_w and array.shape[0] == resized_h:
            resized_arrays.append(array.astype(np.uint8, copy=False))
        else:
            image = Image.fromarray(array, mode="RGB")
            image = image.resize((resized_w, resized_h), Image.Resampling.LANCZOS)
            resized_arrays.append(np.asarray(image, dtype=np.uint8))
    frames = np.stack(resized_arrays).astype(np.uint8, copy=False)
    clip_duration = float(ending - beginning) if ending is not None else 0.0
    if clip_duration > 0:
        effective_fps = len(decoded_times) / clip_duration
    elif len(decoded_times) > 1 and decoded_times[-1] > decoded_times[0]:
        effective_fps = (len(decoded_times) - 1) / (decoded_times[-1] - decoded_times[0])
    else:
        effective_fps = float(target_fps or source_fps or 0.0)
    total_frames = int(info.nb_frames or stream_frame_count or decoded_count)
    return VideoFrames(
        frames=frames,
        timestamps=[float(value) for value in decoded_times],
        fps_effective=float(effective_fps),
        orig_size=(original_width, original_height),
        resized_size=(resized_w, resized_h),
        total_frames=total_frames,
    )


def extract_frames_to_files(
    path: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    *,
    prefix: str = "frame",
    image_format: str = "png",
    format: str | None = None,
    quality: int = 95,
    **frame_options: Any,
) -> list[Path]:
    """Decode frames and save them as numbered PNG or JPEG files."""

    frames = read_video_frames(path, **frame_options)
    selected_format = str(format or image_format).casefold().lstrip(".")
    if selected_format == "jpeg":
        selected_format = "jpg"
    if selected_format not in {"png", "jpg"}:
        raise ValueError("image_format must be 'png', 'jpg', or 'jpeg'")
    directory = normalize_path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    safe_prefix = sanitize_filename(prefix or "frame", max_len=80)
    result: list[Path] = []
    for index, array in enumerate(frames.frames, start=1):
        target = directory / f"{safe_prefix}_{index:06d}.{selected_format}"
        image = Image.fromarray(array, mode="RGB")
        if selected_format == "jpg":
            image.save(target, format="JPEG", quality=min(100, max(1, int(quality))), optimize=True)
        else:
            image.save(target, format="PNG")
        result.append(target)
    return result


def _preview_identity(path: Path) -> str:
    stat = path.stat()
    raw = f"{path.resolve()}\0{stat.st_size}\0{stat.st_mtime_ns}".encode(
        "utf-8", errors="surrogatepass"
    )
    return hashlib.sha256(raw).hexdigest()[:32]


def _browser_video(info: MediaInfo) -> bool:
    suffix = info.path.suffix.casefold()
    video = (info.video_codec or "").casefold()
    audio = (info.audio_codec or "").casefold() or None
    if suffix == ".mp4":
        return video in {"h264", "av1"} and audio in {None, "aac", "mp3"}
    if suffix == ".webm":
        return video in {"vp8", "vp9", "av1"} and audio in {None, "opus", "vorbis"}
    return False


def preview_safe_media(path: str | os.PathLike[str], cache_dir: str | os.PathLike[str]) -> Path:
    """Return a browser-safe source or a content-identity cached conversion."""

    source = normalize_path(path, must_exist=True)
    cache = normalize_path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    info = probe_media(source)
    if info.kind == "unknown":
        raise MediaError(info.error or f"Unsupported media: {source}")
    identity = _preview_identity(source)

    if info.kind == "image":
        orientation = 1
        try:
            with Image.open(source) as image:
                orientation = int(image.getexif().get(274, 1) or 1)
        except Exception:
            pass
        if source.suffix.casefold() in {".png", ".jpg", ".jpeg", ".webp", ".gif"} and orientation == 1:
            return source
        target = cache / f"{identity}_preview.png"
        if target.is_file() and target.stat().st_size:
            return target
        temporary = target.with_name(f".{target.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            with Image.open(source) as image:
                converted = ImageOps.exif_transpose(image)
                if converted.mode not in {"RGB", "RGBA"}:
                    converted = converted.convert("RGBA" if "transparency" in converted.info else "RGB")
                converted.save(temporary, format="PNG")
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return target

    if info.kind == "audio":
        if source.suffix.casefold() in {".wav", ".mp3"}:
            return source
        target = cache / f"{identity}_preview.wav"
        if target.is_file() and target.stat().st_size:
            return target
        temporary = target.with_name(f".{target.stem}.{os.getpid()}.{threading.get_ident()}.wav")
        try:
            run_ffmpeg(["-y", "-i", str(source), "-vn", "-c:a", "pcm_s16le", str(temporary)])
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return target

    if info.has_video:
        if _browser_video(info):
            return source
        target = cache / f"{identity}_preview.mp4"
        if target.is_file() and target.stat().st_size:
            return target
        temporary = target.with_name(f".{target.stem}.{os.getpid()}.{threading.get_ident()}.mp4")
        can_copy_video = (info.video_codec or "").casefold() in {"h264", "av1"}
        try:
            if can_copy_video:
                try:
                    audio_codec = (info.audio_codec or "").casefold()
                    run_ffmpeg(
                        [
                            "-y",
                            "-i",
                            str(source),
                            "-map",
                            "0:v:0",
                            "-map",
                            "0:a?",
                            "-c:v",
                            "copy",
                            "-c:a",
                            "copy" if audio_codec in {"", "aac", "mp3"} else "aac",
                            "-movflags",
                            "+faststart",
                            str(temporary),
                        ]
                    )
                except MediaError:
                    temporary.unlink(missing_ok=True)
                    can_copy_video = False
            if not can_copy_video:
                run_ffmpeg(
                    [
                        "-y",
                        "-i",
                        str(source),
                        "-map",
                        "0:v:0",
                        "-map",
                        "0:a?",
                        "-c:v",
                        "libx264",
                        "-preset",
                        "veryfast",
                        "-crf",
                        "23",
                        "-pix_fmt",
                        "yuv420p",
                        "-c:a",
                        "aac",
                        "-b:a",
                        "160k",
                        "-movflags",
                        "+faststart",
                        str(temporary),
                    ]
                )
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return target

    return source


def make_thumbnail(
    path: str | os.PathLike[str],
    out_png: str | os.PathLike[str],
    at_seconds: float | None = None,
    width: int = 320,
) -> Path:
    """Create a width-constrained PNG thumbnail for an image or video."""

    source = normalize_path(path, must_exist=True)
    target = normalize_path(out_png)
    target.parent.mkdir(parents=True, exist_ok=True)
    target_width = max(1, int(width))
    info = probe_media(source)
    if info.kind == "image":
        with Image.open(source) as image:
            converted = ImageOps.exif_transpose(image).convert("RGB")
            if converted.width != target_width:
                height = max(1, round(converted.height * target_width / converted.width))
                converted = converted.resize((target_width, height), Image.Resampling.LANCZOS)
            converted.save(target, format="PNG")
        return target
    if info.has_video:
        timestamp = float(at_seconds) if at_seconds is not None else min(1.0, (info.duration or 0.0) * 0.1)
        run_ffmpeg(
            [
                "-y",
                "-ss",
                f"{max(0.0, timestamp):.6f}",
                "-i",
                str(source),
                "-frames:v",
                "1",
                "-vf",
                f"scale={target_width}:-2",
                str(target),
            ]
        )
        return target
    raise MediaError(f"Cannot create a visual thumbnail for {info.kind} media")


__all__ = [
    "MediaError",
    "MediaInfo",
    "VideoFrames",
    "extract_audio",
    "extract_frames_to_files",
    "find_ffmpeg",
    "find_ffprobe",
    "make_thumbnail",
    "preview_safe_media",
    "probe_media",
    "read_audio",
    "read_video_frames",
    "run_ffmpeg",
    "trim_media",
]
