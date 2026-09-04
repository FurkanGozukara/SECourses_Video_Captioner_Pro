"""Unified media-to-caption job runner with deferred model loading."""

from __future__ import annotations

import inspect
import json
import math
import os
import re
import shutil
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from vcap import TEMP_DIR, VERSION
from vcap.core import console_progress, gpu
from vcap.core.captions_post import (
    Segment,
    clamp_segments_to_window,
    dedupe_repeated_sentences,
    finalize_caption,
    write_caption_outputs,
)
from vcap.core.dataset_captions import (
    caption_unit_paths,
    render_caption_template,
    render_transcript,
    resolve_captioner_variant,
    transcript_segments,
)
from vcap.core.logs import get_log
from vcap.core.media import (
    MediaInfo,
    extract_audio,
    filter_media_paths,
    probe_media,
    trim_media,
)
from vcap.core.outputs import MetadataBuilder, OutputWriter, RunLog, allocate_run_dir, model_short_name
from vcap.core.paths import exclude_caption_sidecars, list_media_files, normalize_path, sanitize_filename
from vcap.core.preprocess import (
    AutoRejectRules,
    analyze_clip_quality,
    normalize_clip_for_model,
    should_reject,
)
from vcap.core.progress import ProgressEvent, ProgressSink, ProgressTracker, UiThrottle
from vcap.core.scene_split import SceneDetectParams, SceneRange, plan_segments, split_video
from vcap.core.subprocess_runner import CancelToken, CancelledError
from vcap.models.registry import MODEL_SPECS, ModelSpec, variant_to_family

from .job import InputItem, ItemResult, JobResult, JobSpec, ModelChoice, TranscriptSpec


_TOTAL_STEPS = 8
_FAKE_LOCK = threading.RLock()
_FAKE_CAPTIONER: "_FakeCaptioner | None" = None
_FAKE_VARIANT: str | None = None


def _split_layout(spec: JobSpec) -> bool:
    return bool(spec.audio_caption.enabled)


def _needs_whisper(spec: JobSpec) -> bool:
    return bool(spec.transcript.enabled or spec.audio_caption.needs_whisper)


def _captioner_vram_tier(spec: JobSpec) -> float:
    selected = str(spec.model.vram_preset or "auto").strip().casefold()
    if selected != "auto":
        try:
            return float(selected)
        except (TypeError, ValueError):
            pass
    try:
        return float(gpu.resource_snapshot(spec.runtime.gpu_index).get("vram_total_gb", 0.0) or 0.0)
    except Exception:
        return 0.0


def _sound_captioner_variant(spec: JobSpec) -> str:
    return resolve_captioner_variant(
        spec.audio_caption.model_key,
        spec.model.variant_key,
        vram_tier=_captioner_vram_tier(spec),
    )


class _NullSink:
    def on_log(self, message: str, level: str = "info", scope: str | None = None) -> None:
        del message, level, scope

    def on_progress(self, event: ProgressEvent) -> None:
        del event

    def on_item(self, event: ProgressEvent) -> None:
        del event


def _call_sink(method: Any, *args: Any, **kwargs: Any) -> None:
    try:
        method(*args, **kwargs)
    except Exception:
        pass


class _Emitter:
    def __init__(self, sink: ProgressSink | None, tracker: ProgressTracker) -> None:
        self.sink: ProgressSink = sink or _NullSink()
        self.tracker = tracker
        self._current_event_index: int | None = None
        self._console_key = ("pipeline", id(self))
        self._console_throttle = UiThrottle(0.5)
        self.run_dir: str | None = None

    def log(self, text: object, level: str = "info", scope: str = "pipeline") -> None:
        message = str(text)
        get_log().log(
            message,
            level=level,
            scope=scope,
            console=os.environ.get("VCAP_WORKER", "") != "1",
        )
        _call_sink(self.sink.on_log, message, level=level, scope=scope)

    def start_item(self, tracker_index: int, label: str, event_index: int | None = None) -> None:
        self.tracker.start_item(tracker_index, label)
        self._current_event_index = tracker_index if event_index is None else event_index

    @staticmethod
    def _clock(seconds: float | None) -> str:
        if seconds is None:
            return "unknown"
        total = max(0, int(round(float(seconds))))
        hours, remainder = divmod(total, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _payload(
        self,
        event_index: int | None,
        *,
        data: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        snapshot = self.tracker.to_dict()
        return {
            "fraction_in_item": snapshot["fraction_in_item"],
            "step_index": snapshot["step_index"],
            "total_steps": snapshot["total_steps"],
            "eta_seconds": snapshot["eta_seconds"],
            "elapsed": snapshot["elapsed"],
            "processed": snapshot["total_items"] - snapshot["remaining"],
            "done": snapshot["processed"],
            "remaining": snapshot["remaining"],
            "total": snapshot["total_items"],
            "elapsed_s": snapshot["elapsed_s"],
            "eta_s": snapshot["eta_s"],
            "segment_completed": snapshot["segment_completed"],
            "segment_total": snapshot["segment_total"],
            "item_index": event_index,
            "item_elapsed_s": snapshot["item_elapsed_s"],
            "status_line": self.tracker.status_line(),
            "run_dir": self.run_dir,
            **dict(data or {}),
        }

    def _console_status(
        self,
        message: str,
        payload: Mapping[str, Any],
        *,
        force: bool = False,
        finalize: bool = False,
    ) -> None:
        if os.environ.get("VCAP_WORKER") == "1" and os.environ.get("VCAP_CONSOLE_PROGRESS_PARENT") == "1":
            return
        if not self._console_throttle.should_emit(force=force):
            return
        speed = payload.get("tok_per_s", payload.get("tokens_per_second"))
        display_message = message
        message_parts = message.rsplit("|", 1)
        if speed is not None and len(message_parts) == 2 and message_parts[1].strip().casefold().endswith("tok/s"):
            display_message = message_parts[0].rstrip()
        line = (
            f"[{int(payload.get('processed', 0))}/{int(payload.get('total', 0))}] {display_message}"
            f" | elapsed {self._clock(payload.get('elapsed_s'))}"
            f" | ETA {self._clock(payload.get('eta_s'))}"
        )
        try:
            if speed is not None and float(speed) > 0:
                line += f" | {float(speed):.1f} tok/s"
        except (TypeError, ValueError):
            pass
        if finalize:
            console_progress.finalize_progress_line(self._console_key, line)
        else:
            console_progress.show_progress_line(line, key=self._console_key)

    def _emit_running_item(
        self,
        message: str,
        event_index: int | None,
        payload: dict[str, Any],
        *,
        prethrottled: bool = False,
    ) -> None:
        if "new_tokens" in payload and not prethrottled:
            throttle = getattr(self, "_generate_item_throttle", None)
            if throttle is None:
                throttle = UiThrottle(0.1)
                self._generate_item_throttle = throttle
            terminal = "finish_reason" in payload or bool(payload.get("cancelled"))
            if not throttle.should_emit(force=terminal):
                return
        _call_sink(
            self.sink.on_item,
            ProgressEvent(
                message=message,
                fraction=self.tracker.overall_fraction,
                item_index=event_index,
                total_items=self.tracker.total_items,
                status="running",
                data=payload,
                kind="item",
            ),
        )

    def progress(
        self,
        tracker_index: int,
        event_index: int,
        message: str,
        fraction_in_item: float,
        *,
        step_index: int,
        data: Mapping[str, Any] | None = None,
    ) -> None:
        del tracker_index
        progress_data = dict(data or {})
        high_frequency = "new_tokens" in progress_data
        if high_frequency:
            throttle = getattr(self, "_generate_progress_throttle", None)
            if throttle is None:
                throttle = UiThrottle(0.1)
                self._generate_progress_throttle = throttle
            terminal = "finish_reason" in progress_data or bool(progress_data.get("cancelled"))
            if not throttle.should_emit(force=terminal):
                return
        self._current_event_index = event_index
        self.tracker.set_step(
            message,
            min(1.0, max(0.0, float(fraction_in_item))),
            total_steps=_TOTAL_STEPS,
            step_index=step_index,
        )
        payload = self._payload(event_index, data=progress_data)
        _call_sink(
            self.sink.on_progress,
            ProgressEvent(
                message=message,
                fraction=self.tracker.overall_fraction,
                item_index=event_index,
                total_items=self.tracker.total_items,
                data=payload,
            ),
        )
        self._emit_running_item(
            message,
            event_index,
            payload,
            prethrottled=high_frequency,
        )
        self._console_status(message, payload)

    def start_segment(self, total_segments: int) -> None:
        """Start timing a known per-item segment plan."""

        self.tracker.start_segment(total_segments)

    def finish_segment(
        self,
        tracker_index: int,
        event_index: int,
        segment_index: int,
        total_segments: int,
        message: str,
    ) -> None:
        """Complete one segment and publish its single-item ETA."""

        self.tracker.finish_segment()
        self.progress(
            tracker_index,
            event_index,
            message,
            0.56 + 0.30 * segment_index / max(1, total_segments),
            step_index=6,
            data={"segment_index": segment_index, "total_segments": total_segments},
        )

    def phase_progress(
        self,
        message: str,
        fraction: float | None,
        *,
        data: Mapping[str, Any] | None = None,
    ) -> None:
        """Emit download/load progress without disturbing the active item fraction."""

        event_index = self._current_event_index
        payload = self._payload(event_index, data=data)
        resolved_fraction = self.tracker.overall_fraction if fraction is None else min(1.0, max(0.0, float(fraction)))
        _call_sink(
            self.sink.on_progress,
            ProgressEvent(
                message=message,
                fraction=resolved_fraction,
                item_index=event_index,
                total_items=self.tracker.total_items,
                data=payload,
            ),
        )
        self._emit_running_item(message, event_index, payload)
        self._console_status(message, payload)

    def finish(
        self,
        tracker_index: int,
        event_index: int,
        status: str,
        message: str,
        seconds: float,
        *,
        outputs: Mapping[str, Any] | None = None,
        clip_path: str | None = None,
    ) -> None:
        tracker_status = (
            status
            if status in {"done", "skipped", "failed"}
            else "skipped"
            if status in {"unsupported", "cancelled"}
            else "failed"
        )
        self.tracker.finish_item(tracker_status, seconds)
        payload = self._payload(
            event_index,
            data={
                "elapsed": max(0.0, float(seconds)),
                "item_elapsed_s": max(0.0, float(seconds)),
                "outputs": {str(key): str(value) for key, value in dict(outputs or {}).items()},
                "clip_path": clip_path,
            },
        )
        _call_sink(
            self.sink.on_item,
            ProgressEvent(
                message=message,
                fraction=self.tracker.overall_fraction,
                item_index=event_index,
                total_items=self.tracker.total_items,
                status=status,
                data=payload,
                kind="item",
            ),
        )
        self._console_status(
            message,
            payload,
            force=True,
            finalize=self.tracker.remaining == 0,
        )


def _last_saved_clip(result: ItemResult) -> str | None:
    """Return the newest segment clip that still exists on disk, if clips were kept."""

    for record in reversed(list(result.segments or [])):
        raw = record.get("media_path") if isinstance(record, dict) else None
        if raw and Path(str(raw)).is_file():
            return str(raw)
    return None


@dataclass
class _ResolvedInput:
    tracker_index: int
    result_index: int
    item: InputItem
    path: Path | None
    source_root: Path | None
    info: MediaInfo | None = None
    kind: str = "unknown"
    capability: str = ""
    status: str = "pending"
    message: str = ""
    out_dir: Path | None = None
    stem: str = "caption"


@dataclass(frozen=True)
class _SegmentSource:
    index: int
    path: Path | None
    start_s: float
    end_s: float
    media_start: float | None
    media_end: float | None
    persistent_clip: bool = False

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


def _is_cancelled(cancel: object | None) -> bool:
    if cancel is None:
        return False
    method = getattr(cancel, "is_cancelled", None)
    return bool(method()) if callable(method) else False


def _check_cancel(cancel: object | None) -> None:
    if _is_cancelled(cancel):
        raise CancelledError("Caption job cancelled")


def _traceback_tail(limit: int = 18) -> str:
    lines = traceback.format_exc().rstrip().splitlines()
    return "\n".join(lines[-max(1, int(limit)) :])


def _record_value(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    try:
        return asdict(value)
    except (TypeError, ValueError):
        if isinstance(value, Mapping):
            return dict(value)
        attributes = getattr(value, "__dict__", None)
        return dict(attributes) if isinstance(attributes, Mapping) else {"value": str(value)}


def _status_counts(
    items: Iterable[ItemResult],
    *,
    include_audio: bool = False,
) -> dict[str, int]:
    item_list = list(items)
    result = {
        "total": 0,
        "done": 0,
        "skipped": 0,
        "failed": 0,
        "unsupported": 0,
        "cancelled": 0,
    }
    audio_paths: set[str] = set()
    for item in item_list:
        result["total"] += 1
        result[item.status] = result.get(item.status, 0) + 1
        if item.audio_caption_path:
            audio_paths.add(str(item.audio_caption_path))
        for segment in item.segments:
            outputs = segment.get("outputs") if isinstance(segment, Mapping) else None
            if isinstance(outputs, Mapping) and outputs.get("audio_caption"):
                audio_paths.add(str(outputs["audio_caption"]))
    if include_audio:
        result["audio_captions"] = len(audio_paths)
        result["no_speech"] = sum(
            1
            for item in item_list
            for segment in item.segments
            if isinstance(segment, Mapping) and bool(segment.get("no_speech"))
        )
    return result


def _result_index(spec: JobSpec, local_index: int) -> int:
    mapping = spec.internal.get("index_map")
    if isinstance(mapping, Sequence) and local_index < len(mapping):
        try:
            return int(mapping[local_index])
        except (TypeError, ValueError):
            pass
    return local_index


def _inside_root(path: Path, root: Path | None) -> bool:
    if root is None:
        return False
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _resolve_inputs(spec: JobSpec, log_cb: Any | None = None) -> list[_ResolvedInput]:
    expanded: list[tuple[InputItem, Path | None]] = []
    configured_root: Path | None = None
    if spec.output.source_root:
        try:
            configured_root = normalize_path(spec.output.source_root)
        except Exception:
            configured_root = None
    for item in spec.inputs:
        if item.text_prompt_only:
            expanded.append((item, None))
            continue
        try:
            path = normalize_path(item.path)
        except Exception:
            path = Path(str(item.path)).resolve(strict=False)
        if path.is_dir() and spec.output.kind == "batch":
            files = list_media_files(
                path,
                recursive=spec.output.recursive,
                kinds=("video", "audio", "image", "text"),
            )
            files = exclude_caption_sidecars(files)
            before_filters = len(files)
            files = filter_media_paths(
                files,
                spec.output.include_kinds,
                spec.output.name_filter,
            )
            skipped = before_filters - len(files)
            if skipped and callable(log_cb):
                log_cb(f"Batch filters skipped {skipped} file(s).")
            source_root = configured_root if _inside_root(path, configured_root) else path
            expanded.extend((InputItem(file, item.kind), source_root) for file in files)
        else:
            source_root = configured_root if _inside_root(path, configured_root) else None
            expanded.append((item, source_root))
    resolved: list[_ResolvedInput] = []
    for local_index, (item, root) in enumerate(expanded):
        path = None if item.text_prompt_only else normalize_path(item.path)
        resolved.append(
            _ResolvedInput(
                tracker_index=local_index,
                result_index=_result_index(spec, local_index),
                item=item,
                path=path,
                source_root=root,
            )
        )
    return resolved


def _required_capability(info: MediaInfo | None, item: InputItem, spec: JobSpec, model: ModelSpec) -> tuple[str, str]:
    if item.text_prompt_only:
        return "text", "text"
    if info is None:
        return "", "unknown"
    if info.has_video and spec.preprocess.max_frames == 0 and "audio" in model.capabilities:
        return "audio", "video"
    explicit = item.kind.casefold()
    if explicit not in {"", "auto"}:
        if explicit in {"video_audio", "video", "audio", "image", "text"}:
            return explicit, explicit
    if info.kind == "video":
        capability = "video_audio" if spec.preprocess.use_audio_in_video else "video"
        return capability, "video"
    if info.kind == "video_no_audio":
        capability = "video_audio" if model.limits.requires_audio_track else "video"
        return capability, "video"
    if info.kind in {"audio", "image", "text"}:
        return info.kind, info.kind
    return "", info.kind


def _capability_message(model: ModelSpec, capability: str, kind: str) -> str:
    alternatives = [
        candidate.label
        for candidate in MODEL_SPECS.values()
        if capability in candidate.capabilities
    ]
    suffix = (
        f" Models that do support it: {', '.join(alternatives)}."
        if alternatives
        else ""
    )
    label = "audio-only" if kind == "audio" else kind.replace("_", "-")
    return f"{model.label} does not support {label} input.{suffix}"


def _safe_relative_parent(path: Path, root: Path | None) -> Path:
    if root is None:
        return Path()
    try:
        parts = path.relative_to(root).parent.parts
    except ValueError:
        return Path()
    return Path(*(sanitize_filename(part, max_len=100) for part in parts))


def _assign_batch_outputs(spec: JobSpec, resolved: list[_ResolvedInput]) -> Path:
    batch_root = normalize_path(
        spec.output.batch_output_dir
        or (Path(spec.output.outputs_root) / "batch_captions")
    )
    seen: dict[tuple[str, str], int] = {}
    for entry in resolved:
        if entry.path is None:
            parent = Path()
            base = "text_prompt"
        else:
            parent = _safe_relative_parent(entry.path, entry.source_root) if spec.output.mirror_names else Path()
            base = sanitize_filename(entry.path.stem or "caption")
        key = (str(parent).casefold(), base.casefold())
        seen[key] = seen.get(key, 0) + 1
        suffix = f"_{seen[key]:04d}" if seen[key] > 1 else ""
        entry.stem = f"{base}{suffix}"
        entry.out_dir = (
            entry.path.parent
            if spec.output.save_next_to_source and entry.path is not None
            else batch_root / parent
        )
    return batch_root


def _assign_single_outputs(run_dir: Path, resolved: list[_ResolvedInput]) -> None:
    seen: dict[str, int] = {}
    for entry in resolved:
        base = (
            "text_prompt"
            if entry.path is None
            else sanitize_filename(entry.path.stem or "caption")
        )
        key = base.casefold()
        seen[key] = seen.get(key, 0) + 1
        entry.stem = f"{base}_{seen[key]:04d}" if seen[key] > 1 else base
        entry.out_dir = run_dir


def _apply_preassigned_outputs(spec: JobSpec, resolved: list[_ResolvedInput]) -> bool:
    stems = spec.internal.get("output_stems")
    directories = spec.internal.get("output_dirs")
    if not isinstance(stems, Sequence) or not isinstance(directories, Sequence):
        return False
    if len(stems) != len(resolved) or len(directories) != len(resolved):
        return False
    for entry, stem, directory in zip(resolved, stems, directories):
        entry.stem = sanitize_filename(str(stem) or "caption")
        entry.out_dir = normalize_path(str(directory))
    return True


def _public_probe_error(error: object) -> str:
    detail = str(error or "").casefold()
    if "moov atom not found" in detail:
        return "unreadable media (ffmpeg: moov atom not found)"
    if "invalid data found when processing input" in detail:
        return "unreadable media (ffmpeg: Invalid data found when processing input)"
    return "unreadable media (ffmpeg could not inspect the file)"


def _probe_and_classify(
    spec: JobSpec,
    resolved: list[_ResolvedInput],
    model: ModelSpec,
    emitter: _Emitter | None = None,
) -> None:
    fake_captioner = os.environ.get("VCAP_FAKE_CAPTIONER", "").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }
    for entry in resolved:
        if entry.item.text_prompt_only:
            entry.info = None
        elif entry.path is None:
            entry.status = "unsupported"
            entry.message = "Input has no path."
            continue
        else:
            entry.info = probe_media(entry.path)
            if entry.info.kind == "unknown":
                entry.status = "unsupported"
                entry.kind = "unknown"
                raw_error = str(entry.info.error or entry.path)
                entry.message = _public_probe_error(raw_error)
                if emitter is not None:
                    emitter.log(
                        f"Could not inspect {entry.path}: {raw_error}",
                        level="warning",
                        scope="inputs",
                    )
                continue
        if spec.audio_caption.enabled and spec.audio_caption.video_source == "existing":
            if entry.info is None:
                entry.status = "unsupported"
                entry.kind = "text"
                entry.message = "Existing-caption audio passes require a video or audio file."
                continue
            if entry.info.kind not in {"video", "video_no_audio", "audio"}:
                entry.status = "unsupported"
                entry.kind = entry.info.kind
                entry.message = "Existing-caption audio passes support video and audio inputs only."
                continue
            entry.kind = "video" if entry.info.has_video else "audio"
            entry.capability = (
                "video_audio"
                if entry.info.has_video and entry.info.has_audio
                else "video"
                if entry.info.has_video
                else "audio"
            )
            continue
        capability, kind = _required_capability(entry.info, entry.item, spec, model)
        entry.capability, entry.kind = capability, kind
        if capability == "audio" and kind == "video" and entry.info is not None and not entry.info.has_audio:
            entry.status = "unsupported"
            entry.message = "Visual frames are disabled, but the video has no audio track."
            continue
        if not capability or (
            capability not in model.capabilities
            and not (fake_captioner and entry.item.text_prompt_only)
        ):
            entry.status = "unsupported"
            entry.message = _capability_message(model, capability or kind, kind)


def _apply_batch_skip(spec: JobSpec, resolved: list[_ResolvedInput]) -> None:
    if spec.output.kind != "batch" or spec.output.overwrite:
        return
    for entry in resolved:
        if entry.status != "pending" or entry.out_dir is None:
            continue
        paths = caption_unit_paths(entry.out_dir, entry.stem)
        targets: list[Path]
        if not _split_layout(spec):
            targets = [paths.merged]
        elif spec.audio_caption.video_source == "existing":
            targets = [paths.audio]
            has_video_caption = paths.video.is_file() or paths.merged.is_file()
            if entry.path is not None:
                has_video_caption = has_video_caption or entry.path.with_suffix(".txt").is_file()
            if has_video_caption and spec.audio_caption.write_merged:
                targets.append(paths.merged)
        elif spec.audio_caption.write_merged:
            targets = [paths.merged]
        else:
            targets = [paths.video, paths.audio]
        if targets and all(target.is_file() for target in targets):
            entry.status = "skipped"
            entry.message = f"Caption output already exists: {targets[-1]}"


def _apply_batch_limit(spec: JobSpec, resolved: list[_ResolvedInput]) -> None:
    """Keep only the first N pending batch entries after existing-output skips."""

    limit = spec.output.limit_items
    if spec.output.kind != "batch" or limit <= 0:
        return
    selected = 0
    for entry in resolved:
        if entry.status != "pending":
            continue
        if selected < limit:
            selected += 1
            continue
        entry.status = "skipped"
        entry.message = f"Excluded by batch limit of {limit} item(s)."


def _allocate_job_dir(spec: JobSpec) -> Path:
    override = spec.internal.get("run_dir")
    if override:
        directory = normalize_path(str(override))
        directory.mkdir(parents=True, exist_ok=True)
        return directory
    return allocate_run_dir(
        spec.output.outputs_root,
        model_short_name(spec.model.variant_key),
        spec.output.kind,
    )


def _metadata_settings(spec: JobSpec) -> dict[str, Any]:
    if spec.settings:
        return dict(spec.settings)
    typed = spec.to_dict()
    typed.pop("inputs", None)
    typed.pop("internal", None)
    return {"typed_job": typed}


def _metadata_finish_reason(items: Sequence[ItemResult]) -> str | list[str] | None:
    reasons: list[str] = []
    for item in items:
        for segment in item.segments:
            usage = segment.get("usage")
            reason = usage.get("finish_reason") if isinstance(usage, Mapping) else None
            if reason not in {None, ""} and str(reason) not in reasons:
                reasons.append(str(reason))
    if not reasons:
        return None
    return reasons[0] if len(reasons) == 1 else reasons


def _metadata_transcript_summary(
    spec: JobSpec,
    items: Sequence[ItemResult],
) -> dict[str, Any] | None:
    if not _needs_whisper(spec):
        return None
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for item in sorted(items, key=lambda value: value.index):
        raw = item.transcript
        if not isinstance(raw, Mapping):
            continue
        if raw.get("error"):
            errors.append(
                {
                    "item_index": item.index,
                    "path": item.path,
                    "error": str(raw["error"]),
                }
            )
            continue
        probability = raw.get("language_probability", raw.get("probability"))
        duration = float(raw.get("duration_s", raw.get("duration", 0.0)) or 0.0)
        elapsed = float(raw.get("elapsed_s", raw.get("elapsed", 0.0)) or 0.0)
        segment_count = int(raw.get("segment_count", raw.get("segments_count", 0)) or 0)
        word_count = int(raw.get("word_count", raw.get("words_count", 0)) or 0)
        record = {
            "item_index": item.index,
            "path": item.path,
            "model": str(raw.get("model") or spec.transcript.whisper.get("model") or ""),
            "language": raw.get("language"),
            "probability": probability,
            "language_probability": probability,
            "duration": duration,
            "duration_s": duration,
            "elapsed": elapsed,
            "elapsed_s": elapsed,
            "segments": segment_count,
            "segment_count": segment_count,
            "words": word_count,
            "word_count": word_count,
            "files": [str(value) for value in raw.get("files", []) or []],
            "injected": bool(raw.get("injected", False)),
        }
        records.append(record)
    if len(records) == 1 and not errors:
        return records[0]
    models = list(dict.fromkeys(record["model"] for record in records if record["model"]))
    languages = list(
        dict.fromkeys(str(record["language"]) for record in records if record["language"])
    )
    summary: dict[str, Any] = {
        "model": (
            models[0]
            if len(models) == 1
            else models
            if models
            else str(spec.transcript.whisper.get("model") or "")
        ),
        "language": languages[0] if len(languages) == 1 else languages,
        "probability": records[0]["probability"] if len(records) == 1 else None,
        "language_probability": records[0]["probability"] if len(records) == 1 else None,
        "duration": sum(float(record["duration"]) for record in records),
        "duration_s": sum(float(record["duration_s"]) for record in records),
        "elapsed": sum(float(record["elapsed"]) for record in records),
        "elapsed_s": sum(float(record["elapsed_s"]) for record in records),
        "segments": sum(int(record["segment_count"]) for record in records),
        "segment_count": sum(int(record["segment_count"]) for record in records),
        "words": sum(int(record["word_count"]) for record in records),
        "word_count": sum(int(record["word_count"]) for record in records),
        "files": list(
            dict.fromkeys(path for record in records for path in record["files"])
        ),
        "injected": any(bool(record["injected"]) for record in records),
        "items": records,
    }
    if errors:
        summary["errors"] = errors
    return summary


def _write_batch_summary(
    run_dir: Path,
    items: Sequence[ItemResult],
    elapsed: float,
    limit_items: int = 0,
    *,
    include_audio: bool = False,
) -> Path:
    counts = _status_counts(items, include_audio=include_audio)
    count_keys = ["total", "done", "skipped", "failed", "unsupported", "cancelled"]
    if include_audio:
        count_keys.extend(["audio_captions", "no_speech"])
    summary = {
        "counts": {
            key: int(counts.get(key, 0))
            for key in count_keys
        },
        "limit_items": max(0, int(limit_items)),
        "processing_time_seconds": max(0.0, float(elapsed)),
        "items": [
            {
                "status": item.status,
                "path": item.path,
                "elapsed": max(0.0, float(item.elapsed)),
                "elapsed_s": max(0.0, float(item.elapsed)),
            }
            for item in sorted(items, key=lambda value: value.index)
        ],
    }
    return OutputWriter().write_json(run_dir / "summary.json", summary, pretty=False)


def _write_captions_index(
    spec: JobSpec,
    run_dir: Path,
    items: Sequence[ItemResult],
) -> Path:
    """Write the batch caption-to-source lookup consumed by Caption Editor."""

    caption_suffixes = {".txt", ".json", ".srt", ".vtt", ".jsonl"}
    indexed: dict[str, dict[str, Any]] = {}

    def add_outputs(
        outputs: Mapping[str, Any],
        item: ItemResult,
        *,
        start_s: Any = None,
        end_s: Any = None,
    ) -> None:
        for output_key, raw in outputs.items():
            normalized_key = str(output_key).casefold()
            if normalized_key.startswith("transcript_") or normalized_key == "reasoning":
                continue
            if not isinstance(raw, (str, os.PathLike)):
                continue
            caption = normalize_path(raw)
            if caption.suffix.casefold() not in caption_suffixes:
                continue
            indexed[str(caption)] = {
                "source_path": str(normalize_path(item.path)),
                "kind": item.kind,
                "start_s": start_s,
                "end_s": end_s,
                "output_key": str(output_key),
            }

    by_index = {index: value for index, value in enumerate(spec.inputs)}
    for item in sorted(items, key=lambda value: value.index):
        input_item = by_index.get(int(item.index))
        add_outputs(
            item.outputs,
            item,
            start_s=getattr(input_item, "trim_start_s", None),
            end_s=getattr(input_item, "trim_end_s", None),
        )
        for segment in item.segments:
            if not isinstance(segment, Mapping):
                continue
            add_outputs(
                dict(segment.get("outputs") or {}),
                item,
                start_s=segment.get("start_s"),
                end_s=segment.get("end_s"),
            )
    return OutputWriter().write_json(
        run_dir / "captions_index.json",
        {
            "_meta": {"format": "secourses_vcap_captions_index", "version": 1},
            "captions": indexed,
        },
        pretty=True,
    )


def _write_metadata(
    spec: JobSpec,
    items: list[ItemResult],
    run_dir: Path,
    elapsed: float,
    peak_vram_gb: float,
    *,
    load_report: Any | None = None,
    shared_gpu_memory_peak_gb: float = 0.0,
    shared_gpu_memory_excess_peak_gb: float = 0.0,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    try:
        family = variant_to_family(spec.model.variant_key)
        model = MODEL_SPECS[family]
        model_info = {
            "variant_key": spec.model.variant_key,
            "family": family,
            "label": model.label,
            "attention": spec.model.attention,
            "vram_preset": spec.model.vram_preset,
            "compile": spec.runtime.compile,
        }
    except KeyError:
        model_info = {"variant_key": spec.model.variant_key}
    gpu_info = gpu.resource_snapshot(spec.runtime.gpu_index)
    gpu_info["peak_vram_gb"] = float(peak_vram_gb)
    block_swap = getattr(load_report, "block_swap", None)
    gpu_info.update(
        block_swap=dict(block_swap) if isinstance(block_swap, Mapping) else None,
        vram_reserve_gb=float(spec.model.offload.vram_reserve_gb),
        activation_estimate_gb=float(
            getattr(load_report, "activation_estimate_bytes", 0) or 0
        )
        / float(2**30),
        vram_cap_gb=float(getattr(load_report, "vram_cap_bytes", 0) or 0)
        / float(2**30),
        shared_gpu_memory_peak_gb=max(0.0, float(shared_gpu_memory_peak_gb)),
        shared_gpu_memory_excess_peak_gb=max(0.0, float(shared_gpu_memory_excess_peak_gb)),
    )
    name = str(spec.internal.get("metadata_name") or "metadata.json")
    target = run_dir / sanitize_filename(name)
    builder = MetadataBuilder()
    metadata_extra = {
        "counts": _status_counts(items, include_audio=_split_layout(spec)),
        **dict(extra or {}),
    }
    transcript_summary = _metadata_transcript_summary(spec, items)
    if transcript_summary is not None:
        metadata_extra.setdefault("transcript", transcript_summary)
    builder.build(
        VERSION,
        model_info,
        _metadata_settings(spec),
        [item.to_dict() for item in sorted(items, key=lambda value: value.index)],
        {"elapsed_s": max(0.0, elapsed)},
        gpu_info,
        metadata_extra,
    )
    source_root = normalize_path(spec.output.source_root) if spec.output.source_root else None
    explicit_metadata = dict(
        finish_reason=_metadata_finish_reason(items),
        sampling_strategy=spec.preprocess.sampling_strategy,
        context_carry_over=bool(spec.context_carry_over),
        source_root=str(source_root) if source_root is not None else None,
        processing_time_seconds=max(0.0, float(elapsed)),
        batch_limit_items=spec.output.limit_items,
    )
    if builder.data is not None:
        builder.data.update(explicit_metadata)
    builder.write(target)
    if spec.output.kind == "batch" and target.name.casefold() == "metadata.json":
        _write_batch_summary(
            run_dir,
            items,
            elapsed,
            spec.output.limit_items,
            include_audio=_split_layout(spec),
        )
        _write_captions_index(spec, run_dir, items)
    return target


def _empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


class _FakeCaptioner:
    """CPU-only worker-protocol test double enabled by ``VCAP_FAKE_CAPTIONER``."""

    def __init__(self, variant_key: str) -> None:
        self.variant_key = variant_key

    def caption(self, media: Any, prompt: Any, gen: Any, pre: Any, cb: Any) -> Any:
        del prompt, gen, pre
        from vcap.models.base import CaptionResult, CaptionTiming, TokenUsage

        sleep_s = max(0.0, float(os.environ.get("VCAP_FAKE_CAPTION_SLEEP", "0.02")))
        deadline = time.monotonic() + sleep_s
        while time.monotonic() < deadline:
            if _is_cancelled(getattr(cb, "cancel", None)):
                raise CancelledError("Fake caption cancelled")
            time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
        path = Path(str(media.path)) if getattr(media, "path", None) else None
        pattern = os.environ.get("VCAP_FAKE_FAIL_PATTERN", "")
        if pattern and path is not None and pattern.casefold() in path.name.casefold():
            raise RuntimeError(f"Synthetic fake-caption failure for {path.name}")
        base = os.environ.get("VCAP_FAKE_CAPTION_TEXT", "A canned caption.")
        label = path.stem if path is not None else "text prompt"
        text = f"{base} [{label}]"
        start = float(getattr(media, "start", None) or 0.0)
        end_value = getattr(media, "end", None)
        duration = max(0.0, float(end_value) - start) if end_value is not None else 0.0
        if duration <= 0 and path is not None and path.is_file():
            duration = float(probe_media(path).duration or 0.0)
        segments = [(0.0, max(0.01, duration), text)] if duration > 0 else []
        if callable(getattr(cb, "progress", None)):
            cb.progress("Fake generation complete", {"tok_per_s": 100.0, "new_tokens": 8})
        return CaptionResult(
            text=text,
            raw_text=text,
            segments=segments,
            usage=TokenUsage(4, 8),
            timing=CaptionTiming(0.0, sleep_s, 100.0, sleep_s),
        )


def _media_budget_hint(spec: JobSpec, resolved: Sequence["_ResolvedInput"]) -> Any | None:
    """Derive the frame count and media kinds the job will really use from probed inputs.

    The VRAM plan is computed once per load, before any item runs. Using the actual
    clip durations instead of the preset's worst-case ``max_frames`` keeps more
    decoder layers resident for short clips while staying exact for long ones.
    """

    from vcap.models.offload import BudgetHint

    kinds: set[str] = set()
    frames_needed = 0
    saw_visual = False
    fps = max(0.25, float(spec.preprocess.fps or 0.25))
    max_frames = max(0, int(spec.preprocess.max_frames))
    strategy = str(getattr(spec.preprocess, "sampling_strategy", "fps") or "fps").casefold()
    for entry in resolved:
        if entry.status not in {"pending", "done"}:
            continue
        info = entry.info
        kind = str(entry.kind or (info.kind if info is not None else "unknown")).casefold()
        if entry.capability == "audio":
            kinds.add("audio")
            continue
        if kind in {"video", "video_no_audio", "video_audio"}:
            kinds.add("video")
            saw_visual = True
            duration = float(getattr(info, "duration", None) or 0.0) if info is not None else 0.0
            if strategy != "fps" or duration <= 0.0:
                frames_needed = max_frames
            else:
                frames_needed = max(frames_needed, min(max_frames, int(math.ceil(duration * fps)) + 2))
        elif kind == "image":
            kinds.add("image")
            saw_visual = True
            frames_needed = max(frames_needed, 2)
        elif kind == "audio":
            kinds.add("audio")
        elif kind == "text":
            kinds.add("text")
    if not kinds:
        return None
    return BudgetHint(
        max_frames=frames_needed if saw_visual else 0,
        media_kinds=tuple(sorted(kinds)),
    )


def _emit_model_download_progress(emitter: Any, message: Any, payload: Any = None) -> None:
    """Keep readiness checks in the log instead of jumping the job bar to 100%."""

    data = payload if isinstance(payload, Mapping) else {}
    if str(data.get("state") or "").casefold() == "ready":
        emitter.log(str(message), scope="models")
        return
    fraction = data.get("fraction") if isinstance(data, Mapping) else None
    try:
        parsed_fraction = float(fraction) if fraction is not None else None
    except (TypeError, ValueError):
        parsed_fraction = None
    emitter.phase_progress(
        str(message),
        parsed_fraction,
        data={"phase": "model_download", **dict(data)},
    )


class _ModelSession:
    def __init__(
        self,
        spec: JobSpec,
        emitter: _Emitter,
        cancel: CancelToken,
        media_hint: Any | None = None,
    ) -> None:
        self.spec = spec
        self.emitter = emitter
        self.cancel = cancel
        self.media_hint = media_hint
        self.captioner: Any | None = None
        self.loaded: Any | None = None
        self.load_report: Any | None = None
        self.peak_vram_gb = 0.0
        self.shared_gpu_memory_peak_gb = 0.0
        self.shared_gpu_memory_excess_peak_gb = 0.0
        self._compile_prepared = False
        self._patched_loader = False

    def _prepare_compile(self) -> None:
        if self._compile_prepared or not self.spec.runtime.compile:
            return
        self._compile_prepared = True
        from vcap.models.registry import variant_to_family
        from vcap.models.torch_compile import DEFAULT_COMPILE_MODE, prepare_compile_env

        requested_mode = str(
            self.spec.settings.get("compile_mode")
            or self.spec.settings.get("torch_compile_mode")
            or DEFAULT_COMPILE_MODE
        )
        plan = prepare_compile_env(
            True,
            mode=requested_mode,
            family=variant_to_family(self.spec.model.variant_key),
        )
        os.environ.update(plan.env_updates)
        self.emitter.log(
            f"torch.compile mode: {plan.requested_mode} ({plan.mode})",
            scope="compile",
        )
        for warning in plan.warnings:
            self.emitter.log(warning, level="warning", scope="compile")

    def ensure(self) -> Any:
        global _FAKE_CAPTIONER, _FAKE_VARIANT
        if self.captioner is not None:
            return self.captioner
        _check_cancel(self.cancel)
        self._prepare_compile()
        if os.environ.get("VCAP_FAKE_CAPTIONER", "").strip().casefold() in {"1", "true", "yes", "on"}:
            with _FAKE_LOCK:
                reused = (
                    _FAKE_CAPTIONER is not None
                    and _FAKE_VARIANT == self.spec.model.variant_key
                )
                if not reused:
                    _FAKE_CAPTIONER = _FakeCaptioner(self.spec.model.variant_key)
                    _FAKE_VARIANT = self.spec.model.variant_key
                self.captioner = _FAKE_CAPTIONER
            self.emitter.log(
                f"{'Reusing resident' if reused else 'Loading'} {self.spec.model.variant_key}",
                scope="models",
            )
            return self.captioner

        from vcap.models import captioner_for_loaded
        from vcap.models import downloads, loader
        from vcap.models.offload import BudgetHint, OffloadPlan

        load_function = loader.load_model
        loader_is_patched = not (
            getattr(load_function, "__module__", "") == "vcap.models.loader"
            and getattr(load_function, "__name__", "") == "load_model"
        )
        self._patched_loader = loader_is_patched
        resident_variant = (
            loader.MODEL_CACHE.loaded_variant_key()
            if not loader_is_patched
            else None
        )

        def download_progress(message: Any, payload: Any = None) -> None:
            _emit_model_download_progress(self.emitter, message, payload)

        if (
            not loader_is_patched
            and resident_variant != self.spec.model.variant_key
            and os.environ.get("VCAP_SKIP_MODEL_ENSURE", "") != "1"
        ):
            ready, detail = downloads.ensure_model(
                self.spec.model.variant_key,
                progress_cb=download_progress,
                cancel=self.cancel,
            )
            if not ready:
                raise FileNotFoundError(detail)

        def load_progress(*args: Any) -> None:
            message = str(args[0]) if args else "Loading model"
            if message.startswith("llama-server: "):
                self.emitter.log(message, scope="llama.cpp")
                return
            fraction: float | None = None
            data: dict[str, Any] = {"phase": "model_load"}
            if len(args) > 1 and isinstance(args[1], Mapping):
                data.update(dict(args[1]))
                try:
                    raw_fraction = data.get("fraction")
                    fraction = float(raw_fraction) if raw_fraction is not None else None
                except (TypeError, ValueError):
                    fraction = None
            elif len(args) > 1 and isinstance(args[1], (int, float)):
                fraction = float(args[1])
            self.emitter.phase_progress(message, fraction, data=data)

        max_memory: dict[Any, str] | None = None
        if self.spec.model.offload.max_memory:
            max_memory = {
                int(key) if str(key).isdigit() else key: value
                for key, value in self.spec.model.offload.max_memory.items()
            }
        offload = OffloadPlan(
            gpu_layers=self.spec.model.offload.gpu_layers,
            offload_experts=self.spec.model.offload.offload_experts,
            max_memory=max_memory,
            pin_cpu=self.spec.model.offload.pin_cpu,
            vram_reserve_gb=self.spec.model.offload.vram_reserve_gb,
            swap_slots=self.spec.model.offload.swap_slots,
            pinned_ram_budget_gb=self.spec.model.offload.pinned_ram_budget_gb,
            plan_slack_mib=self.spec.model.offload.plan_slack_mib,
        )
        model_spec = MODEL_SPECS[variant_to_family(self.spec.model.variant_key)]
        hint_frames = _effective_max_frames(self.spec, model_spec)
        hint_kinds: tuple[str, ...] = ()
        if self.media_hint is not None:
            if getattr(self.media_hint, "max_frames", None) is not None:
                hint_frames = min(int(hint_frames), int(self.media_hint.max_frames))
            hint_kinds = tuple(getattr(self.media_hint, "media_kinds", ()) or ())
        budget_hint = BudgetHint(
            max_frames=hint_frames,
            max_pixels=self.spec.preprocess.max_pixels,
            fps=self.spec.preprocess.fps,
            max_new_tokens=self.spec.generation.max_new_tokens,
            context_tokens=_context_limit(self.spec, model_spec),
            media_kinds=hint_kinds,
        )
        physical_gpu_index = int(self.spec.runtime.gpu_index)
        isolated_worker = os.environ.get("VCAP_WORKER", "").strip() == "1"
        process_device_index = 0 if isolated_worker else physical_gpu_index
        from vcap.models.torch_compile import DEFAULT_COMPILE_MODE

        compile_mode = str(
            self.spec.settings.get("compile_mode")
            or self.spec.settings.get("torch_compile_mode")
            or DEFAULT_COMPILE_MODE
        )
        load_kwargs = {
            "device": f"cuda:{process_device_index}",
            "gpu_index": physical_gpu_index,
            "attention": self.spec.model.attention,
            "offload": offload,
            "progress_cb": load_progress,
            "compile_model": self.spec.runtime.compile,
            "compile_mode": compile_mode,
        }
        try:
            accepts_budget_hint = "budget_hint" in inspect.signature(loader.load_model).parameters
        except (TypeError, ValueError):
            accepts_budget_hint = False
        if accepts_budget_hint:
            load_kwargs["budget_hint"] = budget_hint
        try:
            accepts_runtime = "runtime" in inspect.signature(loader.load_model).parameters
        except (TypeError, ValueError):
            accepts_runtime = False
        if accepts_runtime:
            load_kwargs["runtime"] = self.spec.runtime
        cached_before = loader.MODEL_CACHE.loaded if not loader_is_patched else None
        if cached_before is None:
            self.emitter.log(
                f"Loading {self.spec.model.variant_key}",
                scope="models",
            )
        self.loaded = (
            load_function(self.spec.model.variant_key, **load_kwargs)
            if loader_is_patched
            else loader.MODEL_CACHE.load(self.spec.model.variant_key, **load_kwargs)
        )
        if cached_before is not None and self.loaded is cached_before:
            self.emitter.log(
                f"Reusing resident {self.spec.model.variant_key}",
                scope="models",
            )
        elif cached_before is not None:
            self.emitter.log(
                f"Loading {self.spec.model.variant_key}",
                scope="models",
            )
        report = getattr(self.loaded, "load_report", None)
        self.load_report = report
        self.peak_vram_gb = max(self.peak_vram_gb, float(getattr(report, "peak_vram_gb", 0.0) or 0.0))
        self.captioner = (
            self.loaded
            if callable(getattr(self.loaded, "caption", None))
            else captioner_for_loaded(self.loaded)
        )
        configure_runtime = getattr(self.captioner, "configure_runtime", None)
        if callable(configure_runtime):
            configure_runtime(self.spec)
        return self.captioner

    def sample_shared_gpu_memory(self) -> None:
        """Record process-level WDDM spill after an item without requiring Task C."""

        sampler = getattr(gpu, "shared_gpu_memory_usage", None)
        if not callable(sampler):
            return
        try:
            usage = sampler()
            shared_gb = float(usage.get("shared_gb", 0.0) or 0.0) if isinstance(usage, Mapping) else 0.0
        except Exception:
            return
        self.shared_gpu_memory_peak_gb = max(self.shared_gpu_memory_peak_gb, shared_gb)
        # WDDM counts pinned (page-locked) host memory as "shared usage", so the
        # block-swap buffers legitimately appear here. Only usage beyond the pinned
        # bytes plus the driver's own baseline indicates that device allocations
        # were paged into system memory.
        block_swap = getattr(self.load_report, "block_swap", None)
        pinned_gb = 0.0
        if isinstance(block_swap, Mapping):
            try:
                pinned_gb = float(block_swap.get("pinned_gib", 0.0) or 0.0)
            except (TypeError, ValueError):
                pinned_gb = 0.0
        excess_gb = max(0.0, shared_gb - pinned_gb - 0.6)
        self.shared_gpu_memory_excess_peak_gb = max(self.shared_gpu_memory_excess_peak_gb, excess_gb)
        if excess_gb > 0.5:
            self.emitter.log(
                f"WDDM shared GPU memory beyond the pinned block-swap buffers: {excess_gb:.2f} GiB "
                f"(shared {shared_gb:.2f} GiB, pinned {pinned_gb:.2f} GiB) — VRAM is being paged into "
                "system memory; increase the VRAM reserve or lower the resident layer count",
                "warning",
                "gpu",
            )

    def unload(self) -> None:
        global _FAKE_CAPTIONER, _FAKE_VARIANT
        if os.environ.get("VCAP_FAKE_CAPTIONER", "").strip().casefold() in {"1", "true", "yes", "on"}:
            if not self.spec.runtime.keep_model_loaded:
                with _FAKE_LOCK:
                    _FAKE_CAPTIONER = None
                    _FAKE_VARIANT = None
            return
        if self.loaded is None:
            return
        if self._patched_loader:
            unload = getattr(self.loaded, "unload", None)
            if callable(unload):
                unload()
            self.loaded = None
            self.captioner = None
            return
        from vcap.models.loader import MODEL_CACHE, unload_model

        if MODEL_CACHE.loaded is self.loaded:
            MODEL_CACHE.unload()
        else:
            unload_model(self.loaded)
        self.loaded = None
        self.captioner = None


def loaded_variant_key() -> str | None:
    """Return the worker's currently resident variant without loading Torch."""

    if _FAKE_VARIANT:
        return _FAKE_VARIANT
    try:
        from vcap.models.loader import MODEL_CACHE

        loaded = MODEL_CACHE.loaded
        return str(loaded.variant.key) if loaded is not None else None
    except Exception:
        return None


def loaded_block_swap_summary() -> dict[str, Any] | None:
    """Return the cached model's JSON-safe block-swap load summary, if any."""

    try:
        from vcap.models.loader import MODEL_CACHE

        loaded = MODEL_CACHE.loaded
        summary = getattr(getattr(loaded, "load_report", None), "block_swap", None)
        return dict(summary) if isinstance(summary, Mapping) else None
    except Exception:
        return None


def unload_cached_model(
    unless_variant: str | None = None,
) -> dict[str, Any] | None:
    """Release a resident model unless it is the selected variant."""

    global _FAKE_CAPTIONER, _FAKE_VARIANT
    with _FAKE_LOCK:
        if _FAKE_VARIANT is not None:
            if _FAKE_VARIANT == unless_variant:
                return None
            released = _FAKE_VARIANT
            _FAKE_CAPTIONER = None
            _FAKE_VARIANT = None
            return {"variant_key": released, "released": True, "fake": True}
    try:
        from vcap.models.loader import MODEL_CACHE

        report = MODEL_CACHE.unload(unless_variant=unless_variant)
        return report.to_dict() if report is not None else None
    except Exception as exc:
        return {"error": str(exc)}


def _context_limit(spec: JobSpec, model: ModelSpec) -> int:
    """Return the job's effective context window: the request capped by the model."""

    cap = int(model.limits.context_tokens)
    requested = getattr(spec.generation, "context_tokens", None)
    return min(cap, int(requested)) if requested else cap


def _family_frame_cap(model: ModelSpec) -> int | None:
    if model.limits.max_frames is not None:
        return int(model.limits.max_frames)
    parameter = next(
        (item for item in model.param_schema if item.name == "max_frames"),
        None,
    )
    return int(parameter.max) if parameter is not None and parameter.max is not None else None


def _effective_max_frames(spec: JobSpec, model: ModelSpec) -> int:
    selected = max(0, int(spec.preprocess.max_frames))
    if selected == 0 and "audio" not in model.capabilities:
        selected = 4
    cap = _family_frame_cap(model)
    if cap is not None:
        selected = min(selected, int(cap))
    return selected


def _original_max_frames(spec: JobSpec) -> int:
    value: Any = spec.settings.get("max_frames", spec.preprocess.max_frames)
    nested = spec.settings.get("preprocess")
    if isinstance(nested, Mapping) and "max_frames" in nested:
        value = nested["max_frames"]
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return max(0, int(spec.preprocess.max_frames))


def _log_job_generation_contracts(
    spec: JobSpec,
    model: ModelSpec,
    resolved: Sequence[_ResolvedInput],
    emitter: _Emitter,
) -> None:
    requested = _original_max_frames(spec)
    cap = _family_frame_cap(model)
    if cap is not None and requested > int(cap):
        emitter.log(
            f"Maximum frames {requested} exceeds the {model.label} cap {int(cap)}; using {int(cap)}.",
            "warning",
            "preprocess",
        )
    if requested == 0:
        if "audio" in model.capabilities:
            if any(
                entry.status == "pending"
                and entry.capability == "audio"
                and entry.info is not None
                and entry.info.has_video
                and entry.info.has_audio
                for entry in resolved
            ):
                emitter.log(
                    "Visual frames disabled (Maximum frames = 0): captioning the audio track only.",
                    scope="preprocess",
                )
        else:
            emitter.log(
                f"Maximum frames 0 is below the {model.label} minimum 4; using 4.",
                "warning",
                "preprocess",
            )
    if spec.generation.do_sample and spec.generation.temperature <= 0:
        emitter.log(
            "Sampling is enabled but temperature is 0; greedy decoding is used.",
            "warning",
            "generation",
        )
    token_cap = int(model.limits.max_new_tokens_cap)
    if int(spec.generation.max_new_tokens) > token_cap:
        emitter.log(
            f"Maximum new tokens {int(spec.generation.max_new_tokens)} exceeds the "
            f"{model.label} cap {token_cap}; using {token_cap}.",
            "warning",
            "generation",
        )
    if int(model.limits.size_multiple) == 28 and int(spec.preprocess.max_pixels) > 602_112:
        emitter.log(
            f"Maximum pixels {int(spec.preprocess.max_pixels):,} exceeds the {model.label} "
            "per-frame ceiling 602,112; using 602,112.",
            "warning",
            "preprocess",
        )
    if spec.summarize_segments and model.family not in {
        "qwen3_omni_instruct",
        "qwen3_omni_thinking",
    }:
        emitter.log(
            f"Summary skipped: {model.label} cannot take text-only input",
            "warning",
            "summary",
        )


def _effective_model_limit(spec: JobSpec, model: ModelSpec, include_audio: bool) -> float | None:
    if spec.preprocess.max_frames == 0 and "audio" in model.capabilities:
        if model.limits.max_duration_s is not None:
            registry_limit = float(model.limits.max_duration_s)
        else:
            available = max(
                1,
                _context_limit(spec, model) - spec.generation.max_new_tokens,
            )
            registry_limit = available / max(1.0, model.limits.audio_tokens_per_s)
    else:
        registry_limit = model.limits.compute_max_duration(
            spec.preprocess.fps,
            spec.preprocess.max_pixels,
            context=_context_limit(spec, model),
            reserve_tokens=spec.generation.max_new_tokens,
            include_audio=include_audio,
        )
    explicit = spec.split.model_max_duration_s
    values = [value for value in (registry_limit, explicit) if value is not None and value > 0]
    return min(values) if values else None


def _scene_params(spec: JobSpec) -> SceneDetectParams:
    return SceneDetectParams(
        threshold=spec.split.scene_threshold,
        min_scene_len_s=spec.split.scene_min_len_s,
        max_scene_len_s=spec.split.scene_max_len_s,
        merge_short_scenes=spec.split.merge_short_scenes,
        merge_below_s=spec.split.merge_below_s,
        fade_detection=spec.split.fade_detection,
        fade_threshold=spec.split.fade_threshold,
        detector=spec.split.scene_detector,
        downscale=spec.split.scene_downscale,
    )


def _trim_source(
    spec: JobSpec,
    entry: _ResolvedInput,
    source: Path,
    work_dir: Path,
    emitter: _Emitter,
) -> tuple[Path, MediaInfo, float]:
    assert entry.info is not None
    info = entry.info
    item_trim = (
        entry.item.trim_start_s is not None
        or entry.item.trim_end_s is not None
    )
    if spec.output.kind == "batch" and not item_trim:
        return source, info, 0.0
    start = max(
        0.0,
        float(entry.item.trim_start_s or 0.0)
        if item_trim
        else spec.preprocess.trim_start_s,
    )
    end = entry.item.trim_end_s if item_trim else spec.preprocess.trim_end_s
    if info.duration is not None:
        start = min(start, info.duration)
        end = info.duration if end is None else min(max(0.0, end), info.duration)
    if end is not None and end <= start:
        raise ValueError("Trim end must be greater than trim start")
    needs_trim = start > 1e-9 or (
        end is not None and info.duration is not None and end < info.duration - 1e-6
    )
    if not needs_trim or info.kind in {"image", "text"}:
        return source, info, 0.0
    if end is None:
        raise ValueError("Cannot trim a source whose duration is unavailable")
    if spec.output.save_processed_files and entry.out_dir is not None:
        target_dir = entry.out_dir / f"{entry.stem}_processed"
    else:
        target_dir = work_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"trimmed{source.suffix or '.mp4'}"
    scope = "per-item clip" if item_trim else "single-file"
    emitter.log(
        f"Trimming {source.name}: {start:.3f}s to {end:.3f}s ({scope} range)",
        scope="preprocess",
    )
    trim_media(
        source,
        target,
        start,
        end,
        mode=spec.split.cut_mode,
        keep_audio=True,
        encode_codec=spec.split.encode_codec,
        encode_crf=spec.split.encode_crf,
        encode_preset=spec.split.encode_preset,
        encode_audio_bitrate=spec.split.encode_audio_bitrate,
    )
    trimmed_info = probe_media(target)
    if trimmed_info.kind == "unknown":
        raise RuntimeError(f"Trimmed media could not be probed: {trimmed_info.error}")
    return target, trimmed_info, start


def _is_dataset_job(spec: JobSpec) -> bool:
    return bool(spec.output.save_clips or spec.split.mode == "trainer")


def _plan_item_segments(
    spec: JobSpec,
    info: MediaInfo,
    model: ModelSpec,
    emitter: _Emitter,
) -> list[SceneRange]:
    if info.kind in {"image", "text"}:
        return [SceneRange(0.0, 0.0)]
    mode = spec.split.mode
    if mode == "scenes" and not info.has_video:
        emitter.log("Scene detection is unavailable for audio; using whole/fixed model-limit splitting.", "warning")
        mode = "whole"
    include_audio = info.has_audio and spec.preprocess.use_audio_in_video
    plan = plan_segments(
        info,
        mode=mode,
        scene_params=_scene_params(spec),
        fixed_chunk_s=spec.split.fixed_chunk_s,
        model_max_duration_s=_effective_model_limit(spec, model, include_audio),
        trainer_target=spec.split.trainer_target,
        sub_split_overlap_s=spec.split.overlap_s,
    )
    for warning in plan.warnings:
        emitter.log(warning, "warning", "segments")
    if not plan.segments:
        raise ValueError("No valid media segments were planned")
    return plan.segments


def _materialize_segments(
    spec: JobSpec,
    entry: _ResolvedInput,
    source: Path | None,
    info: MediaInfo | None,
    segments: list[SceneRange],
    work_dir: Path,
    emitter: _Emitter,
    cancel: CancelToken,
) -> list[_SegmentSource]:
    if source is None or info is None or info.kind in {"image", "text"}:
        return [_SegmentSource(1, source, 0.0, 0.0, None, None)]
    persist = _is_dataset_job(spec)
    audio_only_video = bool(info.has_video and entry.capability == "audio")
    physical = len(segments) > 1 or persist or audio_only_video
    if not physical:
        segment = segments[0]
        return [
            _SegmentSource(
                1,
                source,
                segment.start_s,
                segment.end_s,
                segment.start_s,
                segment.end_s,
            )
        ]
    if persist:
        reason = "Save produced clips is enabled" if spec.output.save_clips else "Trainer split mode is selected"
        emitter.log(
            f"Produced clips will be persisted because {reason}. Clips are otherwise temporary.",
            scope="split",
        )
    else:
        emitter.log(
            "Produced clips are temporary; enable Save produced clips or select Trainer split mode to persist them.",
            scope="split",
        )
    if persist and entry.out_dir is not None:
        clip_dir = entry.out_dir / f"{entry.stem}_clips"
    else:
        clip_dir = work_dir / "clips"
    clip_dir.mkdir(parents=True, exist_ok=True)
    if info.has_video and not audio_only_video:
        clips = split_video(
            source,
            segments,
            clip_dir,
            mode=spec.split.cut_mode,
            keep_audio=True,
            encoder=spec.split.encode_codec,
            crf=spec.split.encode_crf,
            preset=spec.split.encode_preset,
            audio_bitrate=spec.split.encode_audio_bitrate,
            progress_cb=lambda fraction, message: emitter.log(
                f"{message} ({fraction * 100:.1f}%)", scope="split"
            ),
            cancel=cancel,
        )
        return [
            _SegmentSource(
                clip.index,
                clip.path,
                clip.start_s,
                clip.end_s,
                None,
                None,
                persist,
            )
            for clip in clips
        ]
    outputs: list[_SegmentSource] = []
    for index, segment in enumerate(segments, start=1):
        _check_cancel(cancel)
        target = clip_dir / f"clip_{index:04d}.wav"
        extract_audio(
            source,
            target,
            sample_rate=spec.preprocess.audio_sample_rate,
            mono=True,
            start=segment.start_s,
            end=segment.end_s,
        )
        outputs.append(
            _SegmentSource(index, target, segment.start_s, segment.end_s, None, None, persist)
        )
    return outputs


def _reject_rules(spec: JobSpec) -> AutoRejectRules:
    return AutoRejectRules(
        min_duration_s=spec.split.reject_min_duration_s,
        max_black_ratio=spec.split.reject_max_black_ratio,
        max_static_score=spec.split.reject_max_static_score,
        min_sharpness=spec.split.reject_min_sharpness,
        require_audio=spec.split.reject_require_audio,
        max_silence_ratio=spec.split.reject_max_silence_ratio,
    )


def _normalize_segment(
    spec: JobSpec,
    entry: _ResolvedInput,
    segment: _SegmentSource,
    model: ModelSpec,
    work_dir: Path,
    cancel: CancelToken,
) -> _SegmentSource:
    if not spec.preprocess.normalize_clip or segment.path is None:
        return segment
    segment_info = probe_media(segment.path)
    if not segment_info.has_video:
        return segment
    if spec.output.save_processed_files and entry.out_dir is not None:
        directory = entry.out_dir / f"{entry.stem}_processed"
    else:
        directory = work_dir / "normalized"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"normalized_{segment.index:04d}.mp4"
    normalized = normalize_clip_for_model(
        segment.path,
        target,
        target_fps=spec.preprocess.fps,
        max_pixels=spec.preprocess.max_pixels,
        min_pixels=spec.preprocess.min_pixels or model.limits.min_pixels,
        size_multiple=model.limits.size_multiple,
        keep_audio=spec.preprocess.use_audio_in_video,
        audio_sr=spec.preprocess.audio_sample_rate,
        encoder=spec.split.encode_codec,
        crf=spec.split.encode_crf,
        preset=spec.split.encode_preset,
        audio_bitrate=spec.split.encode_audio_bitrate,
        cancel=cancel,
    )
    return replace(segment, path=normalized)


def _model_prompt(
    spec: JobSpec,
    *,
    model_family: str | None = None,
    modality: str | None = None,
    emitter: _Emitter | None = None,
    item_label: str = "input",
) -> Any:
    from vcap.models.base import PromptSpec as ModelPromptSpec

    preset_id = spec.prompt.preset_id
    system_prompt = spec.prompt.system_prompt
    user_prompt = spec.prompt.user_prompt
    if preset_id and model_family and modality:
        try:
            from vcap.prompts.presets import (
                default_preset_for,
                get_preset,
                render_prompt,
            )

            selected = get_preset(preset_id)
            compatible = (
                model_family in selected.applies_to_models
                and (
                    modality in selected.modalities
                    or (
                        model_family == "timechat"
                        and modality == "video_audio"
                        and "video" in selected.modalities
                    )
                )
            )
            if not compatible:
                replacement = default_preset_for(model_family, modality)
                rendered_selected = render_prompt(selected, spec.prompt.variables)
                rendered_replacement = render_prompt(replacement, spec.prompt.variables)
                # UI prompt fields contain the selected preset's rendered text. Preserve a
                # genuinely custom override, but never reuse an automatic video prompt for
                # an audio/image item in a heterogeneous batch.
                automatic_system = system_prompt in {None, rendered_selected[0]}
                automatic_user = user_prompt in {None, rendered_selected[1]}
                preset_id = replacement.id
                if automatic_system:
                    system_prompt = rendered_replacement[0]
                if automatic_user:
                    user_prompt = rendered_replacement[1]
                if emitter is not None:
                    emitter.log(
                        f"Prompt preset {selected.id} does not support {modality}; "
                        f"using {replacement.id} for {item_label}.",
                        scope="prompts",
                    )
        except KeyError:
            # Custom/third-party preset IDs remain usable through their direct fields.
            pass

    return ModelPromptSpec(
        preset_id=preset_id,
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        variables=dict(spec.prompt.variables),
    )


class _PipelineTranscriptSink:
    """Bridge Whisper worker events into the pipeline's existing progress stream."""

    def __init__(self, emitter: _Emitter, entry: _ResolvedInput) -> None:
        self.emitter = emitter
        self.entry = entry

    def on_log(self, message: str, level: str = "info") -> None:
        self.emitter.log(message, level=level, scope="transcript")

    def on_download(self, payload: dict[str, Any]) -> None:
        message = str(payload.get("message") or "Downloading Whisper model")
        self.emitter.log(message, scope="transcript")
        try:
            fraction = max(0.0, min(1.0, float(payload.get("fraction") or 0.0)))
        except (TypeError, ValueError):
            fraction = 0.0
        self.emitter.progress(
            self.entry.tracker_index,
            self.entry.result_index,
            message,
            0.02 + 0.05 * fraction,
            step_index=1,
        )

    def on_progress(self, payload: dict[str, Any]) -> None:
        raw_fraction = payload.get("fraction", payload.get("progress", 0.0))
        try:
            fraction = max(0.0, min(1.0, float(raw_fraction or 0.0)))
        except (TypeError, ValueError):
            fraction = 0.0
        self.emitter.progress(
            self.entry.tracker_index,
            self.entry.result_index,
            str(payload.get("message") or "Transcribing speech"),
            0.05 + 0.10 * fraction,
            step_index=1,
        )

    def on_segment(self, payload: dict[str, Any]) -> None:
        text = str(payload.get("text") or "").strip()
        if text:
            self.emitter.log(text, scope="transcript")

    def on_item_done(self, payload: dict[str, Any]) -> None:
        del payload

    def on_item_error(self, payload: dict[str, Any]) -> None:
        message = str(payload.get("message") or "Whisper transcription failed")
        self.emitter.log(message, level="warning", scope="transcript")


def _run_item_transcript(
    spec: JobSpec,
    entry: _ResolvedInput,
    source: Path,
    run_dir: Path,
    emitter: _Emitter,
    cancel: CancelToken,
) -> tuple[Any | None, dict[str, Any] | None, dict[str, str]]:
    """Run one transcript request without importing Whisper at module import time."""

    from vcap.whisper.client import build_request, run_transcription
    from vcap.whisper.params import TranscriptOutputOptions, WhisperParams

    params = WhisperParams.from_dict(spec.transcript.whisper)
    output = TranscriptOutputOptions(
        formats=tuple(spec.transcript.formats),
        file_suffix=spec.transcript.file_suffix,
    )
    request = build_request(
        params,
        output,
        [
            {
                "index": entry.result_index,
                "path": str(source),
                "out_dir": str(entry.out_dir),
                "stem": entry.stem,
                "trim_start_s": 0.0,
                "trim_end_s": None,
            }
        ],
    )
    emitter.log(f"Transcribing speech for {entry.stem}", scope="transcript")
    outcome = run_transcription(
        request,
        sink=_PipelineTranscriptSink(emitter, entry),
        cancel=cancel,
        request_dir=run_dir / ".work" / "whisper",
    )
    if bool(getattr(outcome, "cancelled", False)) or cancel.is_cancelled():
        raise CancelledError("Whisper transcription cancelled")
    result = getattr(outcome, "results", {}).get(entry.result_index)
    items = list(getattr(outcome, "items", []) or [])
    item_payload = next(
        (
            item
            for item in items
            if int(item.get("item_index", entry.result_index)) == entry.result_index
        ),
        items[0] if items else {},
    )
    if result is None:
        if bool(item_payload.get("skipped")):
            message = str(item_payload.get("message") or "no audio track")
            emitter.log(f"Whisper skipped for {entry.stem}: {message}", scope="transcript")
            return None, {"skipped": True, "message": message}, {}
        error = str(
            item_payload.get("message")
            or getattr(outcome, "error", "")
            or "Whisper returned no transcript"
        )
        raise RuntimeError(error)

    data = result.to_dict()
    files = [str(path) for path in item_payload.get("files", []) or []]
    data["files"] = files
    outputs: dict[str, str] = {}
    for path_text in files:
        path = Path(path_text)
        key = f"transcript_{path.suffix.casefold().lstrip('.') or len(outputs) + 1}"
        outputs[key] = str(path)
    result_segments = list(getattr(result, "segments", []) or [])
    segment_count = len(result_segments)
    word_count = sum(
        1
        for segment in result_segments
        for word in (getattr(segment, "words", None) or [])
        if str(getattr(word, "word", "") or "").strip()
    )
    if not word_count:
        word_count = len(str(getattr(result, "text", "") or "").split())
    language = str(getattr(result, "language", None) or "unknown")
    elapsed = float(getattr(result, "elapsed_s", 0.0) or 0.0)
    probability = getattr(result, "language_probability", None)
    duration = float(getattr(result, "duration_s", 0.0) or 0.0)
    data.update(
        probability=probability,
        duration=duration,
        elapsed=elapsed,
        segment_count=segment_count,
        word_count=word_count,
        injected=False,
        injection_windows=[],
    )
    emitter.log(
        f"Transcript: {segment_count} segments, {word_count} words, language {language}, {elapsed:.1f} s",
        scope="transcript",
    )
    return result, data, outputs


_TRANSCRIPT_TOKEN_RE = re.compile(r"\{\{\s*TRANSCRIPT\s*\}\}", re.IGNORECASE)


def _prompt_has_transcript_token(prompt: Any) -> bool:
    template = getattr(prompt, "user_prompt", None)
    if template is None and getattr(prompt, "preset_id", None):
        try:
            from vcap.prompts.presets import get_preset

            template = get_preset(str(prompt.preset_id)).user_prompt
        except KeyError:
            template = None
    return bool(_TRANSCRIPT_TOKEN_RE.search(str(template or "")))


def _transcript_clock(seconds: float) -> str:
    tenths = max(0, int(round(float(seconds) * 10.0)))
    hours, remainder = divmod(tenths, 36_000)
    minutes, remainder = divmod(remainder, 600)
    whole_seconds, fraction = divmod(remainder, 10)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{fraction}"
    return f"{minutes:02d}:{whole_seconds:02d}.{fraction}"


def _prompt_with_transcript(
    prompt: Any,
    transcript_result: Any | None,
    start_s: float,
    end_s: float,
    transcript_spec: TranscriptSpec,
) -> tuple[Any, str]:
    """Fill the transcript token last and optionally append the configured wrapper."""

    from vcap.prompts.presets import _render_template, get_preset, render_prompt

    transcript_text = ""
    if transcript_result is not None:
        from vcap.whisper.engine import text_between

        transcript_text = text_between(transcript_result, start_s, end_s)
    variables = dict(getattr(prompt, "variables", {}) or {})
    variables["TRANSCRIPT"] = transcript_text

    direct_user = getattr(prompt, "user_prompt", None)
    template_user = direct_user
    preset = None
    preset_id = getattr(prompt, "preset_id", None)
    if template_user is None and preset_id:
        try:
            preset = get_preset(preset_id)
            template_user = preset.user_prompt
        except KeyError:
            preset = None

    has_token = bool(_TRANSCRIPT_TOKEN_RE.search(str(template_user or "")))
    user_prompt = direct_user
    if direct_user is not None:
        user_prompt = _TRANSCRIPT_TOKEN_RE.sub(transcript_text, str(direct_user))
    elif has_token and preset is not None:
        _, user_prompt = render_prompt(preset, variables)

    if transcript_spec.inject_prompt and not has_token and transcript_result is not None:
        wrapper = _render_template(transcript_spec.prompt_wrapper, variables) or ""
        if user_prompt is None and preset is not None:
            _, user_prompt = render_prompt(preset, variables)
        base = str(user_prompt or "").strip()
        user_prompt = "\n\n".join(part for part in (base, wrapper.strip()) if part)

    return replace(prompt, user_prompt=user_prompt, variables=variables), transcript_text


def _structured_with_transcript(
    structured: Any,
    transcript: Mapping[str, Any] | None,
) -> Any:
    if not transcript:
        return structured
    if isinstance(structured, Mapping):
        return {**dict(structured), "transcript": dict(transcript)}
    if structured is None:
        return {"transcript": dict(transcript)}
    return {"caption": structured, "transcript": dict(transcript)}


def _model_gen(spec: JobSpec) -> Any:
    from vcap.models.base import GenParams as ModelGenParams

    return ModelGenParams(**asdict(spec.generation))


def _model_pre(spec: JobSpec, model: ModelSpec, override: Mapping[str, Any] | None = None) -> Any:
    from vcap.models.base import PreprocessParams

    values = {
        "fps": spec.preprocess.fps,
        "max_frames": _effective_max_frames(spec, model),
        "max_pixels": spec.preprocess.max_pixels,
        "min_pixels": spec.preprocess.min_pixels or model.limits.min_pixels,
        "total_pixel_cap": spec.preprocess.total_pixel_cap,
        "adaptive_threshold": spec.preprocess.adaptive_threshold,
        "use_audio_in_video": spec.preprocess.use_audio_in_video,
        "sampling_strategy": spec.preprocess.sampling_strategy,
    }
    values.update(dict(override or {}))
    supported = {item.name for item in fields(PreprocessParams)}
    return PreprocessParams(**{key: value for key, value in values.items() if key in supported})


def _media_input(entry: _ResolvedInput, segment: _SegmentSource, spec: JobSpec, model: ModelSpec) -> Any:
    from vcap.models.base import MediaInput

    if entry.item.text_prompt_only:
        text = entry.item.text if entry.item.text is not None else str(entry.item.path)
        return MediaInput(kind="text", text=text)
    if entry.kind == "text" and segment.path is not None:
        return MediaInput(path=segment.path, kind="text")
    kind = entry.capability
    if entry.kind == "video" and entry.capability != "audio":
        segment_info = probe_media(segment.path) if segment.path is not None else entry.info
        if model.limits.requires_audio_track:
            kind = "video_audio"
        elif spec.preprocess.use_audio_in_video and segment_info is not None and segment_info.has_audio:
            kind = "video_audio"
        else:
            kind = "video"
    return MediaInput(
        path=segment.path,
        kind=kind,
        start=segment.media_start,
        end=segment.media_end,
    )


def _caption_progress(emitter: _Emitter, tracker_index: int, event_index: int, segment_index: int, total: int) -> Any:
    def callback(*args: Any) -> None:
        message = str(args[0]) if args else "Generating caption"
        data: dict[str, Any] = {}
        if len(args) > 1 and isinstance(args[1], Mapping):
            data.update(args[1])
        local = 0.58 + 0.32 * (segment_index - 1 + 0.5) / max(1, total)
        emitter.progress(
            tracker_index,
            event_index,
            message,
            local,
            step_index=6,
            data={"segment_index": segment_index, "total_segments": total, **data},
        )

    return callback


def _degrade_pre(
    current: dict[str, Any],
    model: ModelSpec,
    factor: float = 0.75,
) -> tuple[dict[str, Any], str] | None:
    scale = max(0.5, min(0.95, float(factor)))
    minimum_pixels = int(current.get("min_pixels") or model.limits.min_pixels)
    pixels = int(current["max_pixels"])
    if pixels > minimum_pixels:
        lowered = max(minimum_pixels, int(math.floor(pixels * scale)))
        if lowered < pixels:
            updated = dict(current)
            updated["max_pixels"] = lowered
            return updated, f"max_pixels {pixels} -> {lowered}"
    frames = int(current["max_frames"])
    if frames > 4:
        lowered_frames = max(4, int(math.floor(frames * scale)))
        if lowered_frames < frames:
            updated = dict(current)
            updated["max_frames"] = lowered_frames
            return updated, f"max_frames {frames} -> {lowered_frames}"
    fps = float(current["fps"])
    if fps > 1.0:
        updated = dict(current)
        updated["fps"] = 1.0
        return updated, f"fps {fps:g} -> 1"
    return None


def _caption_with_oom_recovery(
    spec: JobSpec,
    model: ModelSpec,
    session: _ModelSession,
    prompt: Any,
    media: Any,
    callback: Any,
    cancel: CancelToken,
    emitter: _Emitter,
) -> Any:
    from vcap.models.base import Callbacks

    captioner = session.ensure()
    gen = _model_gen(spec)
    pre_values = {
        "fps": spec.preprocess.fps,
        "max_frames": _effective_max_frames(spec, model),
        "max_pixels": spec.preprocess.max_pixels,
        "min_pixels": spec.preprocess.min_pixels or model.limits.min_pixels,
        "total_pixel_cap": spec.preprocess.total_pixel_cap,
        "adaptive_threshold": spec.preprocess.adaptive_threshold,
        "use_audio_in_video": spec.preprocess.use_audio_in_video,
        "sampling_strategy": spec.preprocess.sampling_strategy,
    }
    retries = 0
    compile_retried = False
    while True:
        _check_cancel(cancel)
        try:
            result = captioner.caption(
                media,
                prompt=prompt,
                gen=gen,
                pre=_model_pre(spec, model, pre_values),
                cb=Callbacks(progress=callback, cancel=cancel),
            )
            if isinstance(result, str):
                from vcap.models.base import CaptionResult

                result = CaptionResult(text=result, raw_text=result)
            if getattr(result, "cancelled", False) or _is_cancelled(cancel):
                raise CancelledError("Caption generation cancelled")
            return result
        except CancelledError:
            raise
        except Exception as exc:
            if not compile_retried and session.loaded is not None:
                from vcap.models.torch_compile import (
                    is_compile_runtime_error,
                    restore_eager_model,
                )

                if is_compile_runtime_error(exc):
                    fallback = restore_eager_model(session.loaded, exc)
                    if fallback is not None:
                        compile_retried = True
                        exc.__traceback__ = None
                        _empty_cuda_cache()
                        emitter.log(
                            f"torch.compile mode '{fallback.mode}' failed for {fallback.family} "
                            f"({fallback.reason}); restored {fallback.restored_modules} compiled "
                            "module(s), disabled this family/mode for the process, and retrying "
                            "the same segment once in eager mode.",
                            "warning",
                            "compile",
                        )
                        continue
            if not gpu.is_oom_error(exc) or retries >= spec.runtime.oom_retries:
                raise
            _empty_cuda_cache()
            degraded = _degrade_pre(pre_values, model, spec.runtime.oom_degrade_factor)
            if degraded is None:
                raise
            pre_values, description = degraded
            retries += 1
            emitter.log(
                f"CUDA OOM during caption generation; cleared cache and reduced {description}. "
                f"Retry {retries}/{spec.runtime.oom_retries}.",
                "warning",
                "oom",
            )


def _finalize_text(spec: JobSpec, text: str) -> str:
    return finalize_caption(
        text,
        prefix=spec.post.prefix,
        suffix=spec.post.suffix,
        trigger=spec.post.trigger,
        trigger_mode=spec.post.trigger_mode,
        replace_pairs=spec.post.replace_pairs,
        replace_opts={
            "regex": spec.post.replace_regex,
            "case_insensitive": spec.post.replace_case_insensitive,
            "whole_words": spec.post.replace_whole_words,
        },
        collapse_whitespace=spec.post.collapse_whitespace,
        max_length=spec.post.max_caption_chars,
        dedupe_repeated_sentences=spec.post.dedupe_repeated_sentences,
        join_separator=spec.post.join_separator,
    )


def _finalize_cue_text(spec: JobSpec, text: str, *, dedupe: bool | None = None) -> str:
    """Apply find/replace to a cue without caption-level text injection."""

    return finalize_caption(
        str(text),
        replace_pairs=spec.post.replace_pairs,
        replace_opts={
            "regex": spec.post.replace_regex,
            "case_insensitive": spec.post.replace_case_insensitive,
            "whole_words": spec.post.replace_whole_words,
        },
        collapse_whitespace=spec.post.collapse_whitespace,
        trigger_mode="none",
        max_length=spec.post.max_caption_chars,
        dedupe_repeated_sentences=(
            spec.post.dedupe_repeated_sentences if dedupe is None else bool(dedupe)
        ),
        join_separator=spec.post.join_separator,
    )


_STRUCTURED_CAPTION_KEYS = frozenset(
    {
        "text",
        "caption",
        "description",
        "answer",
        "summary",
        "segment_detail_caption",
        "camera_state",
        "video_background",
        "storyline",
        "shooting_style",
        "speech_content",
        "acoustics_content",
    }
)


def _finalize_structured(spec: JobSpec, value: Any, field_name: str | None = None) -> Any:
    """Apply caption cleanup to textual leaves written as JSON or JSONL."""

    if isinstance(value, str):
        return (
            _finalize_cue_text(spec, value, dedupe=False)
            if field_name is None or field_name.casefold() in _STRUCTURED_CAPTION_KEYS
            else value
        )
    if isinstance(value, Mapping):
        return {
            str(key): _finalize_structured(spec, item, str(key))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_finalize_structured(spec, item, field_name) for item in value]
    if isinstance(value, tuple):
        return [_finalize_structured(spec, item, field_name) for item in value]
    return value


def _context_excerpt(text: str, word_limit: int = 60) -> str:
    words = str(text).split()
    return " ".join(words[-max(1, int(word_limit)) :])


def _prompt_with_context(
    prompt: Any,
    previous_text: str,
    model: ModelSpec,
    word_limit: int = 60,
    context_prompt: str = "Context from the previous segment (do not repeat it): {{CONTEXT}}",
) -> Any:
    excerpt = _context_excerpt(previous_text, word_limit)
    if not excerpt:
        return prompt
    base_user = getattr(prompt, "user_prompt", None)
    if base_user is None:
        preset_id = getattr(prompt, "preset_id", None) or model.default_prompt_preset
        if preset_id:
            try:
                from vcap.prompts.presets import get_preset, render_prompt

                _, base_user = render_prompt(get_preset(preset_id), dict(getattr(prompt, "variables", {}) or {}))
            except KeyError:
                base_user = ""
    rendered_excerpt = json.dumps(excerpt, ensure_ascii=False)
    context = str(context_prompt).replace("{{CONTEXT}}", rendered_excerpt)
    user_prompt = f"{str(base_user or '').rstrip()}\n\n{context}".lstrip()
    return replace(prompt, user_prompt=user_prompt)


def _supports_context_carry_over(model: ModelSpec) -> bool:
    return model.family in {"avocado", "qwen3_omni_instruct", "qwen3_omni_thinking"}


def _segment_output_location(entry: _ResolvedInput, segment: _SegmentSource) -> tuple[Path, str]:
    assert entry.out_dir is not None
    if segment.persistent_clip and segment.path is not None:
        return segment.path.parent, segment.path.stem
    directory = entry.out_dir / f"{entry.stem}_segments"
    return directory, f"clip_{segment.index:04d}"


def _time_label(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def _summary_time_label(seconds: float, include_hours: bool = False) -> str:
    total = max(0, int(round(float(seconds))))
    if include_hours:
        hours, remainder = divmod(total, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    minutes, secs = divmod(total, 60)
    return f"{minutes:02d}:{secs:02d}"


def _summary_input(
    spec: JobSpec,
    model: ModelSpec,
    records: Sequence[Mapping[str, Any]],
    emitter: _Emitter,
) -> str:
    from vcap.prompts.presets import _render_template

    heading = str(_render_template(spec.summary_prompt, dict(spec.prompt.variables)) or "").strip()
    include_hours = any(float(record.get("end_s") or 0.0) >= 3600.0 for record in records)
    lines = [
        f"{_summary_time_label(float(record['start_s']), include_hours)}-"
        f"{_summary_time_label(float(record['end_s']), include_hours)} "
        f"{str(record.get('caption') or '').strip()}"
        for record in records
    ]
    input_token_budget = max(1, _context_limit(spec, model) - spec.summary_max_new_tokens)
    character_budget = max(1, input_token_budget * 4)
    removed = 0
    while len(lines) > 1 and len(heading) + 2 + sum(len(line) + 1 for line in lines) > character_budget:
        lines.pop(0)
        removed += 1
    prompt = heading + ("\n\n" if heading and lines else "") + "\n".join(lines)
    if len(prompt) > character_budget:
        if len(heading) >= character_budget:
            heading = heading[:character_budget].rstrip()
        available = max(0, character_budget - len(heading) - (2 if heading else 0))
        newest = lines[-1][-available:] if available else ""
        prompt = heading + ("\n\n" if heading and newest else "") + newest
        removed += 1
    if removed:
        emitter.log(
            f"Summary prompt exceeded the context window; truncated {removed} oldest caption(s).",
            "warning",
            "summary",
        )
    return prompt


def _summarize_item(
    spec: JobSpec,
    entry: _ResolvedInput,
    model: ModelSpec,
    session: _ModelSession,
    records: Sequence[Mapping[str, Any]],
    emitter: _Emitter,
    cancel: CancelToken,
) -> tuple[str, dict[str, Any], dict[str, Any], Path | None]:
    if (
        not spec.summarize_segments
        or len(records) < 2
        or model.family not in {"qwen3_omni_instruct", "qwen3_omni_thinking"}
    ):
        return "", {}, {}, None
    from vcap.models.base import MediaInput, PromptSpec as ModelPromptSpec

    _check_cancel(cancel)
    emitter.progress(
        entry.tracker_index,
        entry.result_index,
        f"Summarizing {len(records)} segments",
        0.91,
        step_index=7,
    )
    summary_input = _summary_input(spec, model, records, emitter)
    summary_spec = replace(
        spec,
        generation=replace(
            spec.generation,
            max_new_tokens=spec.summary_max_new_tokens,
        ),
    )
    result = _caption_with_oom_recovery(
        summary_spec,
        model,
        session,
        ModelPromptSpec(),
        MediaInput(kind="text", text=summary_input),
        _caption_progress(
            emitter,
            entry.tracker_index,
            entry.result_index,
            len(records),
            len(records),
        ),
        cancel,
        emitter,
    )
    summary = _finalize_cue_text(spec, str(result.text))
    assert entry.out_dir is not None
    path = OutputWriter().write_text(
        entry.out_dir / f"{entry.stem}_summary.txt",
        summary + ("\n" if summary and not summary.endswith("\n") else ""),
    )
    return (
        summary,
        _record_value(getattr(result, "usage", None)),
        _record_value(getattr(result, "timing", None)),
        path,
    )


def _process_item(
    spec: JobSpec,
    entry: _ResolvedInput,
    run_dir: Path,
    model: ModelSpec,
    session: _ModelSession,
    emitter: _Emitter,
    cancel: CancelToken,
) -> ItemResult:
    started = time.perf_counter()
    assert entry.out_dir is not None
    entry.out_dir.mkdir(parents=True, exist_ok=True)
    worker_tag = sanitize_filename(str(spec.internal.get("worker_id") or "main"))
    work_dir = run_dir / ".work" / f"{worker_tag}_{entry.result_index:04d}"
    work_dir.mkdir(parents=True, exist_ok=True)
    peak = 0.0
    try:
        _check_cancel(cancel)
        emitter.progress(entry.tracker_index, entry.result_index, "Preparing input", 0.03, step_index=1)
        if entry.item.text_prompt_only:
            source, info, trim_offset = None, None, 0.0
        else:
            assert entry.path is not None and entry.info is not None
            source, info, trim_offset = _trim_source(spec, entry, entry.path, work_dir, emitter)
        transcript_result: Any | None = None
        transcript_data: dict[str, Any] | None = None
        transcript_outputs: dict[str, str] = {}
        if (
            _needs_whisper(spec)
            and entry.kind in {"video", "audio"}
            and source is not None
            and (info is None or info.has_audio)
        ):
            try:
                transcript_result, transcript_data, transcript_outputs = _run_item_transcript(
                    spec,
                    entry,
                    source,
                    run_dir,
                    emitter,
                    cancel,
                )
            except CancelledError:
                raise
            except Exception as exc:
                transcript_data = {"error": f"{type(exc).__name__}: {exc}"}
                emitter.log(
                    f"Transcript failed; continuing without speech text: {type(exc).__name__}: {exc}",
                    level="warning",
                    scope="transcript",
                )
        elif _needs_whisper(spec) and info is not None and not info.has_audio:
            emitter.log(
                f"Whisper skipped for {entry.stem}: no audio track",
                scope="transcript",
            )
        emitter.progress(entry.tracker_index, entry.result_index, "Planning segments", 0.18, step_index=2)
        segments = (
            [SceneRange(0.0, 0.0)]
            if info is None
            else _plan_item_segments(spec, info, model, emitter)
        )
        emitter.log(f"Planned {len(segments)} segment(s) for {entry.stem}", scope="segments")
        emitter.progress(entry.tracker_index, entry.result_index, "Splitting media", 0.30, step_index=3)
        sources = _materialize_segments(spec, entry, source, info, segments, work_dir, emitter, cancel)
        accepted: list[dict[str, Any]] = []
        rejected_count = 0
        combined_cues: list[Segment] = []
        combined_reasoning: list[str] = []
        total_sources = len(sources)
        prompt_modality = entry.capability if entry.capability else entry.kind
        item_prompt = _model_prompt(
            spec,
            model_family=model.family,
            modality=prompt_modality,
            emitter=emitter,
            item_label=entry.stem,
        )
        previous_final_text = ""
        context_enabled = (
            spec.context_carry_over
            and total_sources > 1
            and _supports_context_carry_over(model)
        )
        for source_position, segment in enumerate(sources):
            _check_cancel(cancel)
            emitter.start_segment(total_sources)
            record: dict[str, Any] = {
                "index": segment.index,
                "start_s": trim_offset + segment.start_s,
                "end_s": trim_offset + segment.end_s,
                "media_path": str(segment.path) if segment.path is not None else None,
            }
            if _split_layout(spec):
                record.update(
                    media_start=segment.media_start,
                    media_end=segment.media_end,
                    persistent_clip=bool(segment.persistent_clip),
                )
            if spec.split.auto_reject and segment.path is not None:
                emitter.progress(
                    entry.tracker_index,
                    entry.result_index,
                    f"Checking clip {segment.index}/{total_sources}",
                    0.37,
                    step_index=4,
                )
                try:
                    quality = analyze_clip_quality(
                        segment.path,
                        sample_frames=spec.split.quality_frames,
                        start_s=segment.media_start,
                        end_s=segment.media_end,
                        black_luma=spec.split.reject_black_luma,
                        silence_rms=spec.split.reject_silence_rms,
                    )
                    reject, reasons = should_reject(quality, _reject_rules(spec))
                    record["quality"] = asdict(quality)
                except Exception as exc:
                    reject, reasons = False, []
                    record["quality_error"] = f"{type(exc).__name__}: {exc}"
                    emitter.log(
                        f"Auto-reject analysis failed for clip {segment.index}; continuing: "
                        f"{type(exc).__name__}: {exc}",
                        "warning",
                        "fitness",
                    )
                if reject:
                    rejected_count += 1
                    record.update(status="rejected", reasons=reasons)
                    accepted.append(record)
                    emitter.log(
                        f"Rejected clip {segment.index}: {', '.join(reasons)}",
                        "warning",
                        "fitness",
                    )
                    emitter.finish_segment(
                        entry.tracker_index,
                        entry.result_index,
                        segment.index,
                        total_sources,
                        f"Rejected clip {segment.index}/{total_sources}",
                    )
                    continue
            emitter.progress(
                entry.tracker_index,
                entry.result_index,
                f"Normalizing clip {segment.index}/{total_sources}",
                0.46,
                step_index=5,
            )
            model_segment = _normalize_segment(spec, entry, segment, model, work_dir, cancel)
            emitter.progress(
                entry.tracker_index,
                entry.result_index,
                f"Captioning clip {segment.index}/{total_sources}",
                0.56 + 0.30 * (segment.index - 1) / max(1, total_sources),
                step_index=6,
            )
            media = _media_input(entry, model_segment, spec, model)
            segment_prompt = item_prompt
            if context_enabled and source_position > 0 and previous_final_text:
                segment_prompt = _prompt_with_context(
                    item_prompt,
                    previous_final_text,
                    model,
                    spec.context_carry_words,
                    spec.context_carry_prompt,
                )
                emitter.log(
                    f"Applied previous-segment context to clip {segment.index}/{total_sources}.",
                    scope="prompts",
                )
            transcript_injected = bool(
                transcript_result is not None
                and (
                    spec.transcript.inject_prompt
                    or _prompt_has_transcript_token(segment_prompt)
                )
            )
            segment_prompt, clip_transcript = _prompt_with_transcript(
                segment_prompt,
                transcript_result,
                segment.start_s,
                segment.end_s,
                spec.transcript,
            )
            clip_word_count = len(clip_transcript.split())
            if transcript_result is not None:
                window_start = trim_offset + segment.start_s
                window_end = trim_offset + segment.end_s
                detail = (
                    f"{clip_word_count} words"
                    if clip_word_count
                    else "no speech in this window"
                )
                if transcript_injected:
                    emitter.log(
                        "Transcript injected into the prompt for clip "
                        f"{segment.index} ({_transcript_clock(window_start)}–"
                        f"{_transcript_clock(window_end)}): {detail}",
                        scope="transcript",
                    )
                else:
                    emitter.log(
                        "Transcript not injected into the prompt for clip "
                        f"{segment.index} ({_transcript_clock(window_start)}–"
                        f"{_transcript_clock(window_end)}): {detail}",
                        scope="transcript",
                    )
                if transcript_data is not None:
                    transcript_data["injected"] = bool(
                        transcript_data.get("injected", False) or transcript_injected
                    )
                    windows = transcript_data.setdefault("injection_windows", [])
                    if isinstance(windows, list):
                        windows.append(
                            {
                                "clip": segment.index,
                                "start_s": window_start,
                                "end_s": window_end,
                                "word_count": clip_word_count,
                                "injected": transcript_injected,
                            }
                        )
            caption_result = _caption_with_oom_recovery(
                spec,
                model,
                session,
                segment_prompt,
                media,
                _caption_progress(
                    emitter,
                    entry.tracker_index,
                    entry.result_index,
                    segment.index,
                    total_sources,
                ),
                cancel,
                emitter,
            )
            raw_caption_text = str(caption_result.text)
            removed_repetitions = 0
            if spec.post.dedupe_repeated_sentences:
                _, removed_repetitions = dedupe_repeated_sentences(raw_caption_text)
            final_text = _finalize_text(spec, raw_caption_text)
            if removed_repetitions > 0:
                emitter.log(
                    f"Removed {removed_repetitions} repeated sentence(s)",
                    scope="postprocess",
                )
            final_structured = _finalize_structured(
                spec,
                getattr(caption_result, "structured", None),
            )
            if transcript_data is not None:
                clip_transcript_data: dict[str, Any] | None = {
                    "start_s": trim_offset + segment.start_s,
                    "end_s": trim_offset + segment.end_s,
                    "text": clip_transcript,
                    "word_count": clip_word_count,
                    "injected": transcript_injected,
                }
                if _split_layout(spec):
                    clip_transcript_data["segments"] = transcript_segments(
                        transcript_result,
                        segment.start_s,
                        segment.end_s,
                    )
            else:
                clip_transcript_data = None
            final_structured = _structured_with_transcript(
                final_structured,
                clip_transcript_data,
            )
            local_cues = [
                Segment(float(start), float(end), _finalize_cue_text(spec, str(text)))
                for start, end, text in list(getattr(caption_result, "segments", []) or [])
            ]
            if not local_cues and final_text and segment.duration_s > 0:
                local_cues = [Segment(0.0, segment.duration_s, _finalize_cue_text(spec, raw_caption_text))]
            if segment.duration_s > 0:
                local_cues = clamp_segments_to_window(
                    local_cues,
                    0.0,
                    segment.duration_s,
                    min_duration_s=spec.post.subtitle_min_cue_s,
                )
            else:
                local_cues = []
            cue_offset = trim_offset + segment.start_s
            combined_cues.extend(
                Segment(cue_offset + cue.start_s, cue_offset + cue.end_s, cue.text)
                for cue in local_cues
            )
            reasoning = str(getattr(caption_result, "reasoning", "") or "")
            if reasoning:
                combined_reasoning.append(
                    f"[{_time_label(cue_offset)} - {_time_label(trim_offset + segment.end_s)}]\n{reasoning}"
                )
            formats = list(spec.post.formats)
            if spec.post.save_reasoning and reasoning and "reasoning" not in formats:
                formats.append("reasoning")
            paths: dict[str, Path] = {}
            if len(sources) > 1 or segment.persistent_clip:
                # Per-clip files only make sense for multi-segment items or
                # persisted clips; a single segment is covered by the item files.
                segment_dir, segment_stem = _segment_output_location(entry, segment)
                paths = write_caption_outputs(
                    segment_dir,
                    segment_stem,
                    formats,
                    text=final_text,
                    structured=final_structured,
                    segments=local_cues,
                    reasoning=reasoning,
                    max_line_chars=spec.post.subtitle_max_line_chars,
                    always_include_txt=not _split_layout(spec),
                )
            peak = max(peak, float(getattr(caption_result, "peak_vram_gb", 0.0) or 0.0))
            timing = getattr(caption_result, "timing", None)
            tok_per_s = float(getattr(timing, "tok_per_s", 0.0) or 0.0)
            if tok_per_s:
                emitter.log(f"Clip {segment.index}: {tok_per_s:.2f} tok/s", scope="caption")
            for warning in getattr(caption_result, "warnings", ()) or ():
                emitter.log(str(warning), "warning", "caption")
            usage = _record_value(getattr(caption_result, "usage", None))
            record.update(
                status="done",
                caption=final_text,
                structured=final_structured,
                outputs={key: str(path) for key, path in paths.items()},
                reasoning_saved=bool(reasoning and spec.post.save_reasoning),
                usage=usage,
                finish_reason=usage.get("finish_reason"),
                timing=_record_value(timing),
                peak_vram_gb=float(getattr(caption_result, "peak_vram_gb", 0.0) or 0.0),
                transcript=clip_transcript_data,
            )
            accepted.append(record)
            previous_final_text = final_text
            emitter.finish_segment(
                entry.tracker_index,
                entry.result_index,
                segment.index,
                total_sources,
                f"Captioned clip {segment.index}/{total_sources}",
            )

        done_records = [record for record in accepted if record.get("status") == "done"]
        if not done_records:
            elapsed = time.perf_counter() - started
            message = f"All {rejected_count} segment(s) were auto-rejected."
            return ItemResult(
                entry.result_index,
                str(entry.path or entry.item.path),
                entry.kind,
                "skipped",
                message,
                segments=accepted,
                outputs=dict(transcript_outputs),
                elapsed=elapsed,
                peak_vram_gb=peak,
                gpu_index=spec.runtime.gpu_index,
                transcript=transcript_data,
            )
        if len(done_records) == 1:
            combined_text = str(done_records[0]["caption"])
            combined_structured = done_records[0].get("structured")
        else:
            blocks = [
                f"[{_time_label(float(record['start_s']))} - {_time_label(float(record['end_s']))}]\n"
                f"{record['caption']}"
                for record in done_records
            ]
            combined_text = "\n\n".join(blocks)
            combined_structured = [
                {
                    "start_s": record["start_s"],
                    "end_s": record["end_s"],
                    "text": record["caption"],
                    "structured": record.get("structured"),
                }
                for record in done_records
            ]
        summary = ""
        summary_usage: dict[str, Any] = {}
        summary_timing: dict[str, Any] = {}
        summary_path: Path | None = None
        try:
            summary, summary_usage, summary_timing, summary_path = _summarize_item(
                spec,
                entry,
                model,
                session,
                done_records,
                emitter,
                cancel,
            )
        except CancelledError:
            raise
        except Exception as exc:
            emitter.log(
                f"Summary generation failed; keeping segment captions: {type(exc).__name__}: {exc}",
                "warning",
                "summary",
            )
        if summary:
            if isinstance(combined_structured, Mapping):
                combined_structured = {**dict(combined_structured), "summary": summary}
            else:
                combined_structured = {"segments": combined_structured, "summary": summary}
        combined_structured = _structured_with_transcript(combined_structured, transcript_data)
        emitter.progress(entry.tracker_index, entry.result_index, "Writing combined outputs", 0.94, step_index=7)
        formats = list(spec.post.formats)
        if spec.post.save_reasoning and combined_reasoning and "reasoning" not in formats:
            formats.append("reasoning")
        combined_paths = write_caption_outputs(
            entry.out_dir,
            entry.stem,
            formats,
            text=combined_text,
            structured=combined_structured,
            segments=combined_cues,
            reasoning="\n\n".join(combined_reasoning),
            max_line_chars=spec.post.subtitle_max_line_chars,
            always_include_txt=not _split_layout(spec),
        )
        if summary_path is not None:
            combined_paths["summary"] = summary_path
        combined_paths.update({key: Path(value) for key, value in transcript_outputs.items()})
        elapsed = time.perf_counter() - started
        emitter.progress(
            entry.tracker_index,
            entry.result_index,
            "Video caption complete" if _split_layout(spec) else "Item complete",
            0.86 if _split_layout(spec) else 1.0,
            step_index=6 if _split_layout(spec) else 8,
        )
        return ItemResult(
            index=entry.result_index,
            path=str(entry.path or entry.item.path),
            kind=entry.kind,
            status="done",
            message=f"Captioned {len(done_records)} segment(s); rejected {rejected_count}.",
            outputs={key: str(path) for key, path in combined_paths.items()},
            segments=accepted,
            elapsed=elapsed,
            peak_vram_gb=max(peak, session.peak_vram_gb),
            gpu_index=spec.runtime.gpu_index,
            summary=summary,
            summary_usage=summary_usage,
            summary_timing=summary_timing,
            transcript=transcript_data,
        )
    finally:
        if not spec.audio_caption.needs_captioner:
            shutil.rmtree(work_dir, ignore_errors=True)


def _existing_video_caption(
    spec: JobSpec,
    entry: _ResolvedInput,
    emitter: _Emitter,
) -> tuple[str, Path | None]:
    """Read or bootstrap the clean video-caption source for existing mode."""

    assert entry.out_dir is not None
    paths = caption_unit_paths(entry.out_dir, entry.stem)
    candidates = [paths.video, paths.merged]
    if entry.path is not None:
        candidates.append(entry.path.with_suffix(".txt"))
    seen: set[str] = set()
    for candidate in candidates:
        identity = os.path.normcase(str(candidate.resolve(strict=False)))
        if identity in seen:
            continue
        seen.add(identity)
        if not candidate.is_file():
            continue
        text = candidate.read_text(encoding="utf-8", errors="replace")
        if candidate != paths.video:
            OutputWriter().write_text(paths.video, text)
            emitter.log(
                f"Copied existing video caption from {candidate} to {paths.video}",
                scope="dataset_captions",
            )
        return text.rstrip("\r\n"), paths.video
    return "", None


def _process_existing_item(
    spec: JobSpec,
    entry: _ResolvedInput,
    run_dir: Path,
    emitter: _Emitter,
    cancel: CancelToken,
) -> ItemResult:
    """Prepare one unsplit existing-caption unit without loading the main model."""

    started = time.perf_counter()
    assert entry.out_dir is not None and entry.path is not None and entry.info is not None
    entry.out_dir.mkdir(parents=True, exist_ok=True)
    _check_cancel(cancel)
    transcript_result: Any | None = None
    transcript_data: dict[str, Any] | None = None
    transcript_outputs: dict[str, str] = {}
    if (
        _needs_whisper(spec)
        and entry.info.kind in {"video", "video_no_audio", "audio"}
        and entry.info.has_audio
    ):
        try:
            transcript_result, transcript_data, transcript_outputs = _run_item_transcript(
                spec,
                entry,
                entry.path,
                run_dir,
                emitter,
                cancel,
            )
        except CancelledError:
            raise
        except Exception as exc:
            transcript_data = {"error": f"{type(exc).__name__}: {exc}"}
            emitter.log(
                f"Transcript failed; continuing without speech text: {type(exc).__name__}: {exc}",
                level="warning",
                scope="transcript",
            )
    elif _needs_whisper(spec) and not entry.info.has_audio:
        emitter.log(
            f"Whisper skipped for {entry.stem}: no audio track",
            scope="transcript",
        )
    video_caption, video_path = _existing_video_caption(spec, entry, emitter)
    duration = float(entry.info.duration or 0.0)
    clip_transcript = render_transcript(
        transcript_result,
        "plain",
        start_s=0.0,
        end_s=duration or None,
    )
    clip_transcript_data = (
        {
            "start_s": 0.0,
            "end_s": duration,
            "text": clip_transcript,
            "word_count": len(clip_transcript.split()),
            "injected": False,
            "segments": transcript_segments(transcript_result, 0.0, duration or None),
        }
        if transcript_data is not None
        else None
    )
    record = {
        "index": 1,
        "start_s": 0.0,
        "end_s": duration,
        "media_path": str(entry.path),
        "media_start": 0.0,
        "media_end": duration or None,
        "persistent_clip": False,
        "status": "done",
        "caption": video_caption,
        "structured": _structured_with_transcript(None, transcript_data),
        "outputs": {},
        "transcript": clip_transcript_data,
        "existing_video_missing": video_path is None,
    }
    outputs = dict(transcript_outputs)
    if video_path is not None:
        outputs["video_caption"] = str(video_path)
    elapsed = time.perf_counter() - started
    message = (
        f"Existing video caption loaded for {entry.path.name}."
        if video_path is not None
        else f"No existing video caption for {entry.path.name}; audio caption will be saved separately"
    )
    return ItemResult(
        index=entry.result_index,
        path=str(entry.path),
        kind=entry.kind,
        status="done",
        message=message,
        outputs=outputs,
        segments=[record],
        elapsed=elapsed,
        gpu_index=spec.runtime.gpu_index,
        transcript=transcript_data,
        video_caption_path=str(video_path) if video_path is not None else None,
        audio_caption_source=spec.audio_caption.source,
    )


def _finalize_sound_text(spec: JobSpec, text: str) -> str:
    """Apply caption cleanup without video-caption prefix/suffix/trigger injection."""

    return finalize_caption(
        text,
        replace_pairs=spec.post.replace_pairs,
        replace_opts={
            "regex": spec.post.replace_regex,
            "case_insensitive": spec.post.replace_case_insensitive,
            "whole_words": spec.post.replace_whole_words,
        },
        collapse_whitespace=spec.post.collapse_whitespace,
        trigger_mode="none",
        max_length=spec.post.max_caption_chars,
        dedupe_repeated_sentences=spec.post.dedupe_repeated_sentences,
        join_separator=spec.post.join_separator,
    )


def _caption_sound_windows(
    spec: JobSpec,
    session: _ModelSession,
    media_path: Path,
    start_s: float | None,
    end_s: float | None,
    work_dir: Path,
    emitter: _Emitter,
    cancel: CancelToken,
    *,
    unit_label: str,
) -> tuple[str, int]:
    """Caption one audio timeline in prompt-free windows no longer than 30 seconds."""

    info = probe_media(media_path)
    if not info.has_audio:
        emitter.log(f"{unit_label}: no audio track; sound caption is empty", scope="sound_captions")
        return "", 0
    duration = float(info.duration or 0.0)
    begin = max(0.0, float(start_s or 0.0))
    finish = duration if end_s is None else min(duration or float(end_s), max(begin, float(end_s)))
    if finish <= begin:
        emitter.log(f"{unit_label}: empty audio window; sound caption is empty", scope="sound_captions")
        return "", 0
    windows: list[tuple[float, float]] = []
    cursor = begin
    while cursor < finish - 1e-6:
        window_end = min(finish, cursor + 30.0)
        windows.append((cursor, window_end))
        cursor = window_end
    emitter.log(
        f"{unit_label}: {len(windows)} Captioner audio window(s), maximum 30 s each",
        scope="sound_captions",
    )
    from vcap.models.base import Callbacks, MediaInput, PromptSpec as ModelPromptSpec

    captioner = session.ensure()
    captioner_model = MODEL_SPECS["qwen3_omni_captioner"]
    work_dir.mkdir(parents=True, exist_ok=True)
    texts: list[str] = []
    for index, (window_start, window_end) in enumerate(windows, start=1):
        _check_cancel(cancel)
        audio_path = work_dir / f"audio_{index:04d}.wav"
        extract_audio(
            media_path,
            audio_path,
            sample_rate=16_000,
            mono=True,
            start=window_start,
            end=window_end,
            cancel_token=cancel,
        )
        result = captioner.caption(
            MediaInput(path=audio_path, kind="audio"),
            prompt=ModelPromptSpec(),
            gen=_model_gen(spec),
            pre=_model_pre(
                replace(spec, preprocess=replace(spec.preprocess, max_frames=0, use_audio_in_video=False)),
                captioner_model,
            ),
            cb=Callbacks(cancel=cancel),
        )
        if isinstance(result, str):
            value = result
        else:
            if getattr(result, "cancelled", False):
                raise CancelledError("Sound caption generation cancelled")
            value = str(getattr(result, "text", "") or "")
        if value.strip():
            texts.append(value.strip())
    return _finalize_sound_text(spec, " ".join(texts)), len(windows)


def _record_output_location(
    entry: _ResolvedInput,
    record: Mapping[str, Any],
    record_count: int,
) -> tuple[Path, str] | None:
    if record_count <= 1 and not bool(record.get("persistent_clip")):
        return None
    media_path = Path(str(record.get("media_path") or ""))
    if bool(record.get("persistent_clip")) and media_path.name:
        return media_path.parent, media_path.stem
    assert entry.out_dir is not None
    return entry.out_dir / f"{entry.stem}_segments", f"clip_{int(record.get('index', 1)):04d}"


def _combined_video_caption(records: Sequence[Mapping[str, Any]]) -> str:
    if len(records) == 1:
        return str(records[0].get("caption") or "")
    return "\n\n".join(
        f"[{_time_label(float(record.get('start_s', 0.0) or 0.0))} - "
        f"{_time_label(float(record.get('end_s', 0.0) or 0.0))}]\n"
        f"{str(record.get('caption') or '')}"
        for record in records
    )


def _combined_sound_caption(records: Sequence[Mapping[str, Any]]) -> str:
    populated = [record for record in records if str(record.get("sound_caption") or "").strip()]
    if not populated:
        return ""
    if len(records) == 1:
        return str(populated[0].get("sound_caption") or "").strip()
    return "\n\n".join(
        f"[{_time_label(float(record.get('start_s', 0.0) or 0.0))} - "
        f"{_time_label(float(record.get('end_s', 0.0) or 0.0))}]\n"
        f"{str(record.get('sound_caption') or '').strip()}"
        for record in populated
    )


def _caption_parts_structured(
    structured: Any,
    *,
    video_caption: str,
    audio_caption: str,
    merged_caption: str,
) -> dict[str, Any]:
    if isinstance(structured, Mapping):
        value = dict(structured)
    elif isinstance(structured, list):
        value = {"segments": structured}
    elif structured is None:
        value = {}
    else:
        value = {"caption": structured}
    value.update(
        video_caption=video_caption,
        audio_caption=audio_caption,
        merged_caption=merged_caption,
    )
    return value


def _existing_structured_output(outputs: Mapping[str, str], fallback: Any) -> Any:
    raw = outputs.get("json")
    if raw:
        try:
            with Path(raw).open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    return fallback


def _write_text_caption(path: Path, text: str) -> Path:
    payload = str(text)
    return OutputWriter().write_text(
        path,
        payload + ("\n" if payload and not payload.endswith("\n") else ""),
    )


def _render_and_write_unit(
    spec: JobSpec,
    out_dir: Path,
    stem: str,
    filename: str,
    video_caption: str,
    transcript_text: str,
    sound_caption: str,
    structured: Any,
    outputs: dict[str, str],
    emitter: _Emitter,
    *,
    allow_merged: bool = True,
) -> tuple[str, str, bool]:
    """Render and atomically write one split-layout caption unit."""

    paths = caption_unit_paths(out_dir, stem)
    if video_caption or allow_merged:
        outputs["video_caption"] = str(_write_text_caption(paths.video, video_caption))
    has_audio = bool(transcript_text.strip() or sound_caption.strip())
    audio_caption = ""
    no_speech = not has_audio
    if has_audio:
        audio_caption = render_caption_template(
            spec.audio_caption.template,
            {
                "TRANSCRIPT": transcript_text,
                "SOUND_CAPTION": sound_caption,
                "FILENAME": filename,
            },
        )
    elif spec.audio_caption.empty_policy == "placeholder":
        audio_caption = spec.audio_caption.empty_text.strip()
    if audio_caption:
        outputs["audio_caption"] = str(_write_text_caption(paths.audio, audio_caption))
    else:
        outputs.pop("audio_caption", None)
        if spec.output.overwrite:
            try:
                paths.audio.unlink(missing_ok=True)
            except OSError as exc:
                emitter.log(f"Could not remove stale audio caption {paths.audio}: {exc}", "warning", "dataset_captions")
        emitter.log(
            (
                f"No speech detected in {filename}; merged caption is the video caption only"
                if no_speech
                else f"Audio caption template rendered empty for {filename}; merged caption uses video only"
            ),
            scope="dataset_captions",
        )
    merged_caption = render_caption_template(
        spec.audio_caption.merge_template,
        {
            "VIDEO_CAPTION": video_caption,
            "AUDIO_CAPTION": audio_caption,
            "TRANSCRIPT": transcript_text,
            "SOUND_CAPTION": sound_caption,
            "FILENAME": filename,
        },
    )
    if not audio_caption:
        merged_caption = video_caption
    if spec.audio_caption.write_merged and allow_merged:
        outputs["merged_caption"] = str(_write_text_caption(paths.merged, merged_caption))
        outputs["txt"] = outputs["merged_caption"]
    else:
        outputs.pop("merged_caption", None)
        outputs.pop("txt", None)
    if "json" in spec.post.formats:
        json_path = out_dir / f"{sanitize_filename(str(stem) or 'caption')}.json"
        value = _caption_parts_structured(
            structured,
            video_caption=video_caption,
            audio_caption=audio_caption,
            merged_caption=merged_caption if allow_merged else "",
        )
        outputs["json"] = str(OutputWriter().write_json(json_path, value, pretty=True))
    return audio_caption, merged_caption, no_speech


def _run_sound_caption_phase(
    spec: JobSpec,
    resolved: Sequence[_ResolvedInput],
    results: dict[int, ItemResult],
    main_session: _ModelSession,
    emitter: _Emitter,
    cancel: CancelToken,
    run_dir: Path,
) -> tuple[_ModelSession | None, bool]:
    """Run prompt-free Captioner inference for every completed produced unit."""

    if not spec.audio_caption.needs_captioner:
        return None, False
    variant = _sound_captioner_variant(spec)
    captioner_limit = MODEL_SPECS["qwen3_omni_captioner"].limits.max_new_tokens_cap
    sound_tokens = min(int(spec.generation.max_new_tokens), int(captioner_limit))
    if sound_tokens != spec.generation.max_new_tokens:
        emitter.log(
            f"Sound-caption max_new_tokens clamped from {spec.generation.max_new_tokens} "
            f"to the Captioner limit {sound_tokens}.",
            "warning",
            "sound_captions",
        )
    sound_spec = replace(
        spec,
        model=replace(spec.model, variant_key=variant),
        generation=replace(spec.generation, max_new_tokens=sound_tokens),
        preprocess=replace(spec.preprocess, max_frames=0, use_audio_in_video=False),
    )
    sound_session = main_session if variant == spec.model.variant_key else _ModelSession(sound_spec, emitter, cancel)
    units: list[tuple[_ResolvedInput, ItemResult, dict[str, Any]]] = []
    entries = {entry.result_index: entry for entry in resolved}
    for result in sorted(results.values(), key=lambda item: item.index):
        entry = entries.get(result.index)
        if entry is None or result.status != "done":
            continue
        for record in result.segments:
            if record.get("status") == "done" and record.get("media_path"):
                units.append((entry, result, record))
    emitter.log(f"Phase: sound captions ({len(units)} unit(s)) with {variant}", scope="sound_captions")
    cancelled = False
    for position, (entry, result, record) in enumerate(units, start=1):
        if _is_cancelled(cancel):
            cancelled = True
            break
        emitter.start_item(entry.tracker_index, entry.stem, entry.result_index)
        emitter.progress(
            entry.tracker_index,
            entry.result_index,
            f"Sound captions {position}/{len(units)}",
            0.86 + 0.08 * position / max(1, len(units)),
            step_index=7,
            data={"phase": "sound_captions", "phase_index": position, "phase_total": len(units)},
        )
        media_path = Path(str(record["media_path"]))
        unit_work = run_dir / ".work" / "sound_captions" / f"{entry.result_index:04d}_{int(record.get('index', 1)):04d}"
        result.sound_caption_model = variant
        try:
            sound, windows = _caption_sound_windows(
                sound_spec,
                sound_session,
                media_path,
                record.get("media_start"),
                record.get("media_end"),
                unit_work,
                emitter,
                cancel,
                unit_label=media_path.name,
            )
            record["sound_caption"] = sound
            record["audio_windows"] = windows
            result.audio_windows += windows
        except CancelledError:
            cancelled = True
            break
        except Exception as exc:
            record["sound_caption"] = ""
            record["audio_windows"] = 0
            record["sound_caption_error"] = f"{type(exc).__name__}: {exc}"
            emitter.log(
                f"Sound caption failed for {media_path.name}; continuing: {type(exc).__name__}: {exc}",
                "error",
                "sound_captions",
            )
        finally:
            sound_session.sample_shared_gpu_memory()
            shutil.rmtree(unit_work, ignore_errors=True)
    if (
        sound_session is not main_session
        and spec.runtime.keep_model_loaded
        and sound_session.captioner is not None
    ):
        emitter.log(
            f"Captioner remains loaded; the next caption run reloads {spec.model.variant_key}",
            scope="models",
        )
    return sound_session, cancelled


def _run_merge_phase(
    spec: JobSpec,
    resolved: Sequence[_ResolvedInput],
    results: dict[int, ItemResult],
    emitter: _Emitter,
    cancel: CancelToken,
) -> bool:
    """Write split caption parts, merged files, and augmented JSON outputs."""

    entries = {entry.result_index: entry for entry in resolved}
    eligible = [result for result in results.values() if result.status == "done"]
    emitter.log(f"Phase: merging captions ({len(eligible)} item(s))", scope="dataset_captions")
    cancelled = False
    for position, result in enumerate(sorted(eligible, key=lambda item: item.index), start=1):
        if _is_cancelled(cancel):
            cancelled = True
            result.status = "cancelled"
            result.message = "Cancelled before merging captions."
            continue
        entry = entries.get(result.index)
        if entry is None or entry.out_dir is None:
            continue
        emitter.start_item(entry.tracker_index, entry.stem, entry.result_index)
        emitter.progress(
            entry.tracker_index,
            entry.result_index,
            f"Merging captions {position}/{len(eligible)}",
            0.94 + 0.06 * position / max(1, len(eligible)),
            step_index=8,
            data={"phase": "merge", "phase_index": position, "phase_total": len(eligible)},
        )
        records = [record for record in result.segments if record.get("status") == "done"]
        try:
            for record in records:
                location = _record_output_location(entry, record, len(result.segments))
                if location is None:
                    continue
                segment_dir, segment_stem = location
                transcript_data = record.get("transcript")
                transcript_text = (
                    render_transcript(transcript_data, spec.audio_caption.transcript_style)
                    if spec.audio_caption.needs_whisper
                    else ""
                )
                media_path = Path(str(record.get("media_path") or segment_stem))
                segment_outputs = dict(record.get("outputs") or {})
                audio_text, merged_text, no_speech = _render_and_write_unit(
                    spec,
                    segment_dir,
                    segment_stem,
                    media_path.stem,
                    str(record.get("caption") or ""),
                    transcript_text,
                    str(record.get("sound_caption") or ""),
                    record.get("structured"),
                    segment_outputs,
                    emitter,
                    allow_merged=not bool(record.get("existing_video_missing")),
                )
                record.update(
                    outputs=segment_outputs,
                    audio_caption=audio_text,
                    merged_caption=merged_text,
                    no_speech=no_speech,
                    audio_caption_source=spec.audio_caption.source,
                    sound_caption_model=result.sound_caption_model,
                    video_caption_path=segment_outputs.get("video_caption"),
                    audio_caption_path=segment_outputs.get("audio_caption"),
                    merged_caption_path=segment_outputs.get("merged_caption"),
                )
            video_caption = _combined_video_caption(records)
            transcript_text = (
                render_transcript(result.transcript, spec.audio_caption.transcript_style)
                if spec.audio_caption.needs_whisper
                else ""
            )
            sound_caption = _combined_sound_caption(records)
            existing_missing = bool(records) and all(bool(record.get("existing_video_missing")) for record in records)
            outputs = dict(result.outputs)
            combined_structured = _existing_structured_output(
                outputs,
                (
                    records[0].get("structured")
                    if len(records) == 1
                    else [record.get("structured") for record in records]
                ),
            )
            audio_text, merged_text, no_speech = _render_and_write_unit(
                spec,
                entry.out_dir,
                entry.stem,
                entry.path.stem if entry.path is not None else entry.stem,
                video_caption,
                transcript_text,
                sound_caption,
                combined_structured,
                outputs,
                emitter,
                allow_merged=not existing_missing,
            )
            paths = caption_unit_paths(entry.out_dir, entry.stem)
            result.outputs = outputs
            result.video_caption_path = outputs.get("video_caption")
            result.audio_caption_path = outputs.get("audio_caption")
            result.merged_caption_path = outputs.get("merged_caption")
            result.audio_caption_source = spec.audio_caption.source
            if existing_missing:
                result.message = f"No existing video caption for {entry.path.name if entry.path else entry.stem}; audio caption saved separately"
            else:
                result.message = (
                    f"Captioned {len(records)} segment(s); audio caption saved"
                    if audio_text
                    else f"Captioned {len(records)} segment(s); merged caption uses video only"
                )
            if no_speech:
                for record in records:
                    record.setdefault("no_speech", True)
            del paths, merged_text
        except CancelledError:
            cancelled = True
            result.status = "cancelled"
            result.message = "Cancelled while merging captions."
        except Exception as exc:
            result.status = "failed"
            result.message = f"Could not merge caption parts: {type(exc).__name__}: {exc}"
            result.traceback_tail = _traceback_tail()
            emitter.log(result.message, "error", "dataset_captions")
    return cancelled


def _cleanup_item_work_dirs(spec: JobSpec, run_dir: Path, results: Iterable[ItemResult]) -> None:
    worker_tag = sanitize_filename(str(spec.internal.get("worker_id") or "main"))
    for result in results:
        shutil.rmtree(run_dir / ".work" / f"{worker_tag}_{result.index:04d}", ignore_errors=True)
    shutil.rmtree(run_dir / ".work" / "sound_captions", ignore_errors=True)


def _populate_split_result_paths(
    spec: JobSpec,
    resolved: Sequence[_ResolvedInput],
    results: Mapping[int, ItemResult],
) -> None:
    """Record all split artifacts that exist, including skipped batch items."""

    by_index = {entry.result_index: entry for entry in resolved}
    for result in results.values():
        result.audio_caption_source = spec.audio_caption.source
        entry = by_index.get(result.index)
        if entry is None or entry.out_dir is None:
            continue
        paths = caption_unit_paths(entry.out_dir, entry.stem)
        if paths.video.is_file():
            result.video_caption_path = str(paths.video)
            result.outputs.setdefault("video_caption", str(paths.video))
        if paths.audio.is_file():
            result.audio_caption_path = str(paths.audio)
            result.outputs.setdefault("audio_caption", str(paths.audio))
        if spec.audio_caption.write_merged and paths.merged.is_file():
            result.merged_caption_path = str(paths.merged)
            result.outputs.setdefault("merged_caption", str(paths.merged))
            result.outputs.setdefault("txt", str(paths.merged))


def _run_job_local(spec: JobSpec, sinks: ProgressSink | None, cancel: CancelToken) -> JobResult:
    started = time.perf_counter()
    try:
        model = MODEL_SPECS[variant_to_family(spec.model.variant_key)]
        model_error = ""
    except KeyError as exc:
        model = next(iter(MODEL_SPECS.values()))
        model_error = str(exc)
    filter_messages: list[str] = []
    resolved = _resolve_inputs(spec, filter_messages.append)
    preassigned_outputs = _apply_preassigned_outputs(spec, resolved)
    if spec.output.kind == "batch" and not preassigned_outputs:
        _assign_batch_outputs(spec, resolved)
    run_dir = _allocate_job_dir(spec)
    if spec.output.kind == "single" and not preassigned_outputs:
        _assign_single_outputs(run_dir, resolved)
    tracker = ProgressTracker(len(resolved), [entry.stem for entry in resolved])
    emitter = _Emitter(sinks, tracker)
    emitter.run_dir = str(run_dir)
    results: dict[int, ItemResult] = {}
    peak_vram = 0.0
    metadata_path = run_dir / str(spec.internal.get("metadata_name") or "metadata.json")
    session = _ModelSession(spec, emitter, cancel)
    sound_session: _ModelSession | None = None
    finished_indices: set[int] = set()
    with RunLog(run_dir):
        emitter.log(
            f"Starting {spec.output.kind} job with {len(resolved)} resolved input(s) using "
            f"{spec.model.variant_key}."
        )
        for message in filter_messages:
            emitter.log(message, scope="inputs")
        if _split_layout(spec):
            emitter.log("Phase: captions", scope="dataset_captions")
        if spec.audio_caption.needs_whisper and not spec.transcript.enabled:
            emitter.log(
                "Whisper transcript enabled by the audio caption setting",
                scope="transcript",
            )
        existing_mode = bool(
            _split_layout(spec) and spec.audio_caption.video_source == "existing"
        )
        scene_detection_on = str(
            spec.settings.get("scene_detect_enabled", "")
        ).strip().casefold() in {"1", "true", "yes", "on"}
        if existing_mode and (spec.split.mode != "whole" or scene_detection_on):
            emitter.log(
                "Scene splitting is ignored when existing video captions are reused; input files are the caption units.",
                "warning",
                "segments",
            )
        if spec.output.kind == "batch" and (
            spec.preprocess.trim_start_s > 1e-9 or spec.preprocess.trim_end_s is not None
        ):
            emitter.log("Trim range is ignored for folder batches", "warning", "preprocess")
        if model_error and not existing_mode:
            for entry in resolved:
                entry.status = "failed"
                entry.kind = "unknown"
                entry.message = f"Unknown model variant: {spec.model.variant_key}"
        else:
            _probe_and_classify(spec, resolved, model, emitter)
            if not existing_mode and not spec.internal.get("suppress_job_contract_logs"):
                _log_job_generation_contracts(spec, model, resolved, emitter)
            _apply_batch_skip(spec, resolved)
            _apply_batch_limit(spec, resolved)
            # The VRAM plan is made once at the first load; size it from the media
            # this job really contains rather than the preset's worst-case frames.
            session.media_hint = _media_budget_hint(spec, resolved)

        actionable: list[_ResolvedInput] = []
        for entry in resolved:
            if entry.status == "pending":
                actionable.append(entry)
                continue
            emitter.start_item(entry.tracker_index, entry.stem, entry.result_index)
            result = ItemResult(
                index=entry.result_index,
                path=str(entry.path or entry.item.path),
                kind=entry.kind,
                status=entry.status,
                message=entry.message,
                gpu_index=spec.runtime.gpu_index,
            )
            results[entry.result_index] = result
            emitter.log(entry.message, "warning" if entry.status == "unsupported" else "info")
            emitter.finish(entry.tracker_index, entry.result_index, entry.status, entry.message, 0.0)
            finished_indices.add(entry.result_index)

        if not actionable:
            emitter.log("Nothing requires captioning; no caption model was loaded.")
        cancelled_job = False
        for position, entry in enumerate(actionable):
            if cancelled_job or _is_cancelled(cancel):
                emitter.start_item(entry.tracker_index, entry.stem, entry.result_index)
                result = ItemResult(
                    entry.result_index,
                    str(entry.path or entry.item.path),
                    entry.kind,
                    "cancelled",
                    "Cancelled before processing.",
                    gpu_index=spec.runtime.gpu_index,
                )
                results[entry.result_index] = result
                emitter.finish(
                    entry.tracker_index,
                    entry.result_index,
                    "cancelled",
                    result.message,
                    0.0,
                )
                finished_indices.add(entry.result_index)
                continue
            emitter.start_item(entry.tracker_index, entry.stem, entry.result_index)
            item_started = time.perf_counter()
            try:
                result = (
                    _process_existing_item(spec, entry, run_dir, emitter, cancel)
                    if existing_mode
                    else _process_item(spec, entry, run_dir, model, session, emitter, cancel)
                )
                results[entry.result_index] = result
                peak_vram = max(peak_vram, result.peak_vram_gb)
                if not _split_layout(spec):
                    emitter.finish(
                        entry.tracker_index,
                        entry.result_index,
                        result.status,
                        result.message,
                        result.elapsed,
                        outputs=result.outputs,
                        clip_path=_last_saved_clip(result),
                    )
                    finished_indices.add(entry.result_index)
            except CancelledError as exc:
                cancelled_job = True
                elapsed = time.perf_counter() - item_started
                result = ItemResult(
                    entry.result_index,
                    str(entry.path or entry.item.path),
                    entry.kind,
                    "cancelled",
                    str(exc),
                    elapsed=elapsed,
                    gpu_index=spec.runtime.gpu_index,
                )
                results[entry.result_index] = result
                emitter.log(str(exc), "warning")
                emitter.finish(
                    entry.tracker_index,
                    entry.result_index,
                    "cancelled",
                    result.message,
                    elapsed,
                )
                finished_indices.add(entry.result_index)
            except Exception as exc:
                elapsed = time.perf_counter() - item_started
                tail = _traceback_tail()
                message = f"{type(exc).__name__}: {exc}"
                result = ItemResult(
                    entry.result_index,
                    str(entry.path or entry.item.path),
                    entry.kind,
                    "failed",
                    message,
                    elapsed=elapsed,
                    traceback_tail=tail,
                    gpu_index=spec.runtime.gpu_index,
                )
                results[entry.result_index] = result
                emitter.log(f"Item failed: {message}\n{tail}", "error")
                emitter.finish(
                    entry.tracker_index,
                    entry.result_index,
                    "failed",
                    message,
                    elapsed,
                )
                finished_indices.add(entry.result_index)
            finally:
                session.sample_shared_gpu_memory()

        if _split_layout(spec):
            sound_session, sound_cancelled = _run_sound_caption_phase(
                spec,
                resolved,
                results,
                session,
                emitter,
                cancel,
                run_dir,
            )
            cancelled_job = cancelled_job or sound_cancelled
            merge_cancelled = _run_merge_phase(
                spec,
                resolved,
                results,
                emitter,
                cancel,
            )
            cancelled_job = cancelled_job or merge_cancelled
            by_result_index = {entry.result_index: entry for entry in resolved}
            for result in sorted(results.values(), key=lambda item: item.index):
                if result.index in finished_indices:
                    continue
                entry = by_result_index.get(result.index)
                if entry is None:
                    continue
                emitter.start_item(entry.tracker_index, entry.stem, entry.result_index)
                emitter.finish(
                    entry.tracker_index,
                    entry.result_index,
                    result.status,
                    result.message,
                    result.elapsed,
                    outputs=result.outputs,
                    clip_path=_last_saved_clip(result),
                )
                finished_indices.add(result.index)
            _populate_split_result_paths(spec, resolved, results)
            _cleanup_item_work_dirs(spec, run_dir, results.values())

        if not spec.runtime.keep_model_loaded:
            final_session = (
                sound_session
                if sound_session is not None and sound_session.captioner is not None
                else session
            )
            final_session.unload()
            emitter.log("Model unloaded at the end of the job.", scope="models")
        ordered = sorted(results.values(), key=lambda item: item.index)
        elapsed = time.perf_counter() - started
        try:
            active_session = sound_session or session
            metadata_path = _write_metadata(
                spec,
                ordered,
                run_dir,
                elapsed,
                max(peak_vram, session.peak_vram_gb, active_session.peak_vram_gb),
                load_report=session.load_report,
                shared_gpu_memory_peak_gb=max(
                    session.shared_gpu_memory_peak_gb,
                    active_session.shared_gpu_memory_peak_gb,
                ),
                shared_gpu_memory_excess_peak_gb=max(
                    session.shared_gpu_memory_excess_peak_gb,
                    active_session.shared_gpu_memory_excess_peak_gb,
                ),
            )
        except Exception as exc:
            emitter.log(f"Could not write metadata: {exc}", "error", "metadata")
            raise
        counts = _status_counts(ordered, include_audio=_split_layout(spec))
        emitter.log(
            "Job finished: "
            + ", ".join(f"{key}={value}" for key, value in counts.items() if key != "total")
            + f" in {elapsed:.2f}s."
        )
    return JobResult(
        items=ordered,
        counts=counts,
        run_dir=str(run_dir),
        metadata_path=str(metadata_path),
        elapsed=elapsed,
    )


class _GpuPrefixSink:
    def __init__(self, sink: ProgressSink | None, gpu_index: int, index_map: Sequence[int]) -> None:
        self.sink: ProgressSink = sink or _NullSink()
        self.gpu_index = gpu_index
        self.index_map = list(index_map)

    def _index(self, index: int | None) -> int | None:
        # Child specs already carry an index_map, so streamed indexes are global.
        return index

    def on_log(self, message: str, level: str = "info", scope: str | None = None) -> None:
        _call_sink(self.sink.on_log, f"[GPU {self.gpu_index}] {message}", level=level, scope=scope)

    def on_progress(self, event: ProgressEvent) -> None:
        _call_sink(
            self.sink.on_progress,
            replace(
                event,
                message=f"[GPU {self.gpu_index}] {event.message}",
                item_index=self._index(event.item_index),
                data={**event.data, "gpu_index": self.gpu_index},
            ),
        )

    def on_item(self, event: ProgressEvent) -> None:
        _call_sink(
            self.sink.on_item,
            replace(
                event,
                message=f"[GPU {self.gpu_index}] {event.message}",
                item_index=self._index(event.item_index),
                data={**event.data, "gpu_index": self.gpu_index},
            ),
        )


def _compile_env_updates(enabled: bool) -> dict[str, str]:
    if not enabled:
        return {}
    try:
        from vcap.models.torch_compile import prepare_compile_env

        return dict(prepare_compile_env(True).env_updates)
    except Exception:
        return {}


def _round_robin_partitions(
    entries: Sequence[_ResolvedInput], partition_count: int
) -> list[list[_ResolvedInput]]:
    partitions: list[list[_ResolvedInput]] = [
        [] for _ in range(max(1, int(partition_count)))
    ]
    for index, entry in enumerate(entries):
        partitions[index % len(partitions)].append(entry)
    return partitions


def _run_multi_gpu(spec: JobSpec, sinks: ProgressSink | None, cancel: CancelToken) -> JobResult:
    from queue import Queue

    from vcap.core.subprocess_runner import WorkerProcess, build_child_env

    started = time.perf_counter()
    filter_messages: list[str] = []
    expanded = _resolve_inputs(spec, filter_messages.append)
    if not expanded:
        local_spec = replace(
            spec,
            runtime=replace(spec.runtime, gpu_indices=(spec.runtime.gpu_index,)),
        )
        return _run_job_local(local_spec, sinks, cancel)
    run_dir = _allocate_job_dir(spec)
    contract_model: ModelSpec | None = None
    contract_entries: list[_ResolvedInput] = []
    try:
        contract_model = MODEL_SPECS[variant_to_family(spec.model.variant_key)]
        for entry in expanded:
            candidate = replace(entry)
            if candidate.path is not None:
                candidate.info = probe_media(candidate.path)
            candidate.capability, candidate.kind = _required_capability(
                candidate.info,
                candidate.item,
                spec,
                contract_model,
            )
            contract_entries.append(candidate)
    except (KeyError, OSError):
        contract_model = None

    with RunLog(run_dir):
        for message in filter_messages:
            get_log().log(message, scope="inputs")
            if sinks is not None:
                _call_sink(sinks.on_log, message, level="info", scope="inputs")
        get_log().log(
            f"Starting multi-GPU job on devices {', '.join(map(str, spec.runtime.gpu_indices))}.",
            scope="pipeline",
            console=os.environ.get("VCAP_WORKER", "") != "1",
        )
        if contract_model is not None:
            class _ContractEmitter:
                @staticmethod
                def log(
                    message: object,
                    level: str = "info",
                    scope: str = "pipeline",
                ) -> None:
                    get_log().log(str(message), level=level, scope=scope)
                    if sinks is not None:
                        _call_sink(
                            sinks.on_log,
                            str(message),
                            level=level,
                            scope=scope,
                        )

            _log_job_generation_contracts(
                spec,
                contract_model,
                contract_entries,
                _ContractEmitter(),  # type: ignore[arg-type]
            )
    if spec.output.kind == "single":
        _assign_single_outputs(run_dir, expanded)
    else:
        _assign_batch_outputs(spec, expanded)
    pre_results: list[ItemResult] = []
    partition_entries = expanded
    if spec.output.kind == "batch" and spec.output.limit_items > 0:
        try:
            limit_model = MODEL_SPECS[variant_to_family(spec.model.variant_key)]
            _probe_and_classify(spec, expanded, limit_model, emitter)
        except KeyError:
            pass
        _apply_batch_skip(spec, expanded)
        _apply_batch_limit(spec, expanded)
        pre_results = [
            ItemResult(
                entry.result_index,
                str(entry.path or entry.item.path),
                entry.kind,
                entry.status,
                entry.message,
            )
            for entry in expanded
            if entry.status != "pending"
        ]
        partition_entries = [entry for entry in expanded if entry.status == "pending"]
    gpu_indices = tuple(spec.runtime.gpu_indices)
    partitions = _round_robin_partitions(partition_entries, len(gpu_indices))
    messages: Queue[tuple[int, str, Any]] = Queue()
    workers: dict[int, WorkerProcess] = {}
    threads: list[threading.Thread] = []

    active_partitions = [
        (gpu_index, part)
        for gpu_index, part in zip(gpu_indices, partitions)
        if part
    ]
    def pump(gpu_index: int, partition: list[_ResolvedInput]) -> None:
        worker = WorkerProcess()
        workers[gpu_index] = worker
        try:
            env = build_child_env(
                gpu_index,
                extra={
                    **_compile_env_updates(spec.runtime.compile),
                    "VCAP_MULTI_GPU_CHILD": "1",
                },
            )
            worker.start(
                [sys.executable, "-u", "-m", "vcap.pipeline.worker", "--gpu", str(gpu_index)],
                cwd=Path(__file__).resolve().parents[2],
                env=env,
                name=f"gpu-{gpu_index}",
            )
            event_stream = worker.events()
            ready = False
            for event in event_stream:
                if event.get("ev") == "ready":
                    ready = True
                    break
            if not ready:
                raise RuntimeError("worker exited before ready")
            index_map = [entry.result_index for entry in partition]
            child_inputs = [entry.item for entry in partition]
            child_spec = replace(
                spec,
                inputs=child_inputs,
                runtime=replace(spec.runtime, gpu_index=gpu_index, gpu_indices=(gpu_index,)),
                internal={
                    **spec.internal,
                    "run_dir": str(run_dir),
                    "metadata_name": f"metadata_gpu_{gpu_index}.json",
                    "worker_id": f"gpu_{gpu_index}",
                    "index_map": index_map,
                    "output_stems": [entry.stem for entry in partition],
                    "output_dirs": [str(entry.out_dir) for entry in partition],
                    "suppress_job_contract_logs": True,
                },
            )
            worker.send({"cmd": "run_job", "job": child_spec.to_dict()})
            for event in event_stream:
                messages.put((gpu_index, "event", event))
                if event.get("ev") in {"result", "error"}:
                    break
        except Exception as exc:
            messages.put((gpu_index, "failure", f"{type(exc).__name__}: {exc}"))
        finally:
            messages.put((gpu_index, "done", None))

    for gpu_index, partition in active_partitions:
        thread = threading.Thread(
            target=pump,
            args=(gpu_index, partition),
            daemon=True,
            name=f"vcap-multi-gpu-{gpu_index}",
        )
        threads.append(thread)
        thread.start()
    result_items: list[ItemResult] = list(pre_results)
    failures: dict[int, str] = {}
    finished: set[int] = set()
    cancel_sent_at: float | None = None
    while len(finished) < len(active_partitions):
        if _is_cancelled(cancel):
            if cancel_sent_at is None:
                cancel_sent_at = time.monotonic()
                for worker in list(workers.values()):
                    try:
                        worker.send({"cmd": "cancel"})
                    except Exception:
                        pass
            elif time.monotonic() - cancel_sent_at >= 5.0:
                for worker in list(workers.values()):
                    worker.kill_tree(grace=0.25)
        try:
            gpu_index, kind, value = messages.get(timeout=0.1)
        except Exception:
            continue
        partition = next(part for index, part in active_partitions if index == gpu_index)
        index_map = [entry.result_index for entry in partition]
        prefixed = _GpuPrefixSink(sinks, gpu_index, index_map)
        if kind == "done":
            finished.add(gpu_index)
            continue
        if kind == "failure":
            failures[gpu_index] = str(value)
            prefixed.on_log(str(value), "error", "worker")
            continue
        event = value
        ev = event.get("ev")
        if ev in {"log", "stdout"}:
            prefixed.on_log(
                str(event.get("text", "")),
                str(event.get("level", "info")),
                event.get("scope"),
            )
        elif ev == "progress":
            fields = {
                key: event.get(key)
                for key in (
                    "message",
                    "fraction",
                    "item_index",
                    "total_items",
                    "status",
                    "data",
                    "timestamp",
                    "kind",
                )
                if event.get(key) is not None
            }
            prefixed.on_progress(ProgressEvent(**fields))
        elif ev == "item":
            prefixed.on_item(
                ProgressEvent(
                    message=str(event.get("message", "")),
                    item_index=event.get("index"),
                    status=event.get("status"),
                    kind="item",
                    data={**dict(event.get("data") or {}), "gpu_index": gpu_index},
                )
            )
        elif ev == "result":
            child = JobResult.from_dict(event["job_result"])
            result_items.extend(child.items)
        elif ev == "error":
            failures[gpu_index] = str(event.get("message", "Worker failed"))
    for thread in threads:
        thread.join(timeout=2.0)
    for gpu_index, worker in list(workers.items()):
        try:
            if worker.is_alive():
                worker.send({"cmd": "exit"})
                worker.wait(timeout=2.0)
        except Exception:
            worker.kill_tree(grace=0.25)
    completed_indices = {item.index for item in result_items}
    for gpu_index, message in failures.items():
        partition = next(part for index, part in active_partitions if index == gpu_index)
        for entry in partition:
            if entry.result_index not in completed_indices:
                result_items.append(
                    ItemResult(
                        entry.result_index,
                        str(entry.path or entry.item.path),
                        entry.kind,
                        "failed",
                        message,
                        gpu_index=gpu_index,
                    )
                )
                completed_indices.add(entry.result_index)
    for entry in expanded:
        if entry.result_index in completed_indices:
            continue
        was_cancelled = _is_cancelled(cancel)
        result_items.append(
            ItemResult(
                entry.result_index,
                str(entry.path or entry.item.path),
                entry.kind,
                "cancelled" if was_cancelled else "failed",
                "Cancelled while the GPU worker was stopping."
                if was_cancelled
                else "GPU worker exited without returning an item result.",
            )
        )
    result_items.sort(key=lambda item: item.index)
    if _split_layout(spec):
        _populate_split_result_paths(
            spec,
            expanded,
            {item.index: item for item in result_items},
        )
    elapsed = time.perf_counter() - started
    metadata = _write_metadata(
        spec,
        result_items,
        run_dir,
        elapsed,
        max((item.peak_vram_gb for item in result_items), default=0.0),
        extra={"multi_gpu_indices": list(gpu_indices)},
    )
    return JobResult(
        result_items,
        _status_counts(result_items, include_audio=_split_layout(spec)),
        str(run_dir),
        str(metadata),
        elapsed,
    )


def run_job(
    spec: JobSpec,
    sinks: ProgressSink | None,
    cancel: CancelToken | None = None,
) -> JobResult:
    """Run one single/batch job without importing Gradio or eagerly importing Torch."""

    if not isinstance(spec, JobSpec):
        spec = JobSpec.from_dict(spec)  # type: ignore[arg-type]
    token = cancel or CancelToken()
    multi_gpu = (
        spec.output.kind == "batch"
        and len(spec.runtime.gpu_indices) > 1
        and spec.runtime.subprocess_mode
        and os.environ.get("VCAP_MULTI_GPU_CHILD", "") != "1"
    )
    if multi_gpu:
        return _run_multi_gpu(spec, sinks, token)
    return _run_job_local(spec, sinks, token)


__all__ = [
    "loaded_block_swap_summary",
    "loaded_variant_key",
    "run_job",
    "unload_cached_model",
]
