"""Unified media-to-caption job runner with deferred model loading."""

from __future__ import annotations

import json
import math
import os
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
    apply_replacements,
    finalize_caption,
    write_caption_outputs,
)
from vcap.core.logs import get_log
from vcap.core.media import MediaInfo, extract_audio, probe_media, trim_media
from vcap.core.outputs import MetadataBuilder, OutputWriter, RunLog, allocate_run_dir, model_short_name
from vcap.core.paths import list_media_files, normalize_path, sanitize_filename
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

from .job import InputItem, ItemResult, JobResult, JobSpec


_TOTAL_STEPS = 8
_FAKE_LOCK = threading.RLock()
_FAKE_CAPTIONER: "_FakeCaptioner | None" = None
_FAKE_VARIANT: str | None = None


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
            "item_index": event_index,
            "item_elapsed_s": snapshot["item_elapsed_s"],
            "status_line": self.tracker.status_line(),
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

    def _emit_running_item(self, message: str, event_index: int | None, payload: dict[str, Any]) -> None:
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
        self._current_event_index = event_index
        self.tracker.set_step(
            message,
            min(1.0, max(0.0, float(fraction_in_item))),
            total_steps=_TOTAL_STEPS,
            step_index=step_index,
        )
        payload = self._payload(event_index, data=data)
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
        self._emit_running_item(message, event_index, payload)
        self._console_status(message, payload)

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
    ) -> None:
        tracker_status = status if status in {"done", "skipped", "failed"} else "failed"
        self.tracker.finish_item(tracker_status, seconds)
        payload = self._payload(
            event_index,
            data={"elapsed": max(0.0, float(seconds)), "item_elapsed_s": max(0.0, float(seconds))},
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


def _status_counts(items: Iterable[ItemResult]) -> dict[str, int]:
    result = {
        "total": 0,
        "done": 0,
        "skipped": 0,
        "failed": 0,
        "unsupported": 0,
        "cancelled": 0,
    }
    for item in items:
        result["total"] += 1
        result[item.status] = result.get(item.status, 0) + 1
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


def _resolve_inputs(spec: JobSpec) -> list[_ResolvedInput]:
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
                kinds=("video", "audio", "image"),
            )
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
        entry.out_dir = batch_root / parent
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


def _probe_and_classify(spec: JobSpec, resolved: list[_ResolvedInput], model: ModelSpec) -> None:
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
                entry.message = f"Unsupported or unreadable input: {entry.info.error or entry.path}"
                continue
        capability, kind = _required_capability(entry.info, entry.item, spec, model)
        entry.capability, entry.kind = capability, kind
        if not capability or capability not in model.capabilities:
            entry.status = "unsupported"
            entry.message = _capability_message(model, capability or kind, kind)


def _apply_batch_skip(spec: JobSpec, resolved: list[_ResolvedInput]) -> None:
    if spec.output.kind != "batch" or spec.output.overwrite:
        return
    for entry in resolved:
        if entry.status != "pending" or entry.out_dir is None:
            continue
        target = entry.out_dir / f"{entry.stem}.txt"
        if target.is_file():
            entry.status = "skipped"
            entry.message = f"Caption already exists: {target}"


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


def _write_batch_summary(run_dir: Path, items: Sequence[ItemResult], elapsed: float) -> Path:
    counts = _status_counts(items)
    summary = {
        "counts": {
            key: int(counts.get(key, 0))
            for key in ("total", "done", "skipped", "failed", "unsupported", "cancelled")
        },
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


def _write_metadata(
    spec: JobSpec,
    items: list[ItemResult],
    run_dir: Path,
    elapsed: float,
    peak_vram_gb: float,
    *,
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
    name = str(spec.internal.get("metadata_name") or "metadata.json")
    target = run_dir / sanitize_filename(name)
    builder = MetadataBuilder()
    builder.build(
        VERSION,
        model_info,
        _metadata_settings(spec),
        [item.to_dict() for item in sorted(items, key=lambda value: value.index)],
        {"elapsed_s": max(0.0, elapsed)},
        gpu_info,
        {"counts": _status_counts(items), **dict(extra or {})},
    )
    source_root = normalize_path(spec.output.source_root) if spec.output.source_root else None
    explicit_metadata = dict(
        finish_reason=_metadata_finish_reason(items),
        sampling_strategy=spec.preprocess.sampling_strategy,
        context_carry_over=bool(spec.context_carry_over),
        source_root=str(source_root) if source_root is not None else None,
        processing_time_seconds=max(0.0, float(elapsed)),
    )
    if builder.data is not None:
        builder.data.update(explicit_metadata)
    builder.write(target)
    if spec.output.kind == "batch" and target.name.casefold() == "metadata.json":
        _write_batch_summary(run_dir, items, elapsed)
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


class _ModelSession:
    def __init__(self, spec: JobSpec, emitter: _Emitter, cancel: CancelToken) -> None:
        self.spec = spec
        self.emitter = emitter
        self.cancel = cancel
        self.captioner: Any | None = None
        self.loaded: Any | None = None
        self.peak_vram_gb = 0.0
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
                if _FAKE_CAPTIONER is None or _FAKE_VARIANT != self.spec.model.variant_key:
                    _FAKE_CAPTIONER = _FakeCaptioner(self.spec.model.variant_key)
                    _FAKE_VARIANT = self.spec.model.variant_key
                self.captioner = _FAKE_CAPTIONER
            self.emitter.log(f"Using CPU fake captioner for {self.spec.model.variant_key}", scope="models")
            return self.captioner

        from vcap.models import captioner_for_loaded
        from vcap.models import downloads, loader
        from vcap.models.offload import OffloadPlan

        load_function = loader.load_model
        loader_is_patched = not (
            getattr(load_function, "__module__", "") == "vcap.models.loader"
            and getattr(load_function, "__name__", "") == "load_model"
        )
        self._patched_loader = loader_is_patched

        def download_progress(message: Any, payload: Any = None) -> None:
            data = payload if isinstance(payload, Mapping) else {}
            fraction = data.get("fraction") if isinstance(data, Mapping) else None
            try:
                parsed_fraction = float(fraction) if fraction is not None else None
            except (TypeError, ValueError):
                parsed_fraction = None
            self.emitter.phase_progress(
                str(message),
                parsed_fraction,
                data={"phase": "model_download", **dict(data)},
            )

        if not loader_is_patched and os.environ.get("VCAP_SKIP_MODEL_ENSURE", "") != "1":
            ready, detail = downloads.ensure_model(
                self.spec.model.variant_key,
                progress_cb=download_progress,
                cancel=self.cancel,
            )
            if not ready:
                raise FileNotFoundError(detail)

        def load_progress(*args: Any) -> None:
            message = str(args[0]) if args else "Loading model"
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
        self.loaded = (
            load_function(self.spec.model.variant_key, **load_kwargs)
            if loader_is_patched
            else loader.MODEL_CACHE.load(self.spec.model.variant_key, **load_kwargs)
        )
        report = getattr(self.loaded, "load_report", None)
        self.peak_vram_gb = max(self.peak_vram_gb, float(getattr(report, "peak_vram_gb", 0.0) or 0.0))
        self.captioner = (
            self.loaded
            if callable(getattr(self.loaded, "caption", None))
            else captioner_for_loaded(self.loaded)
        )
        return self.captioner

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


def unload_cached_model() -> None:
    """Release the persistent worker model, including the CPU fake hook."""

    global _FAKE_CAPTIONER, _FAKE_VARIANT
    with _FAKE_LOCK:
        _FAKE_CAPTIONER = None
        _FAKE_VARIANT = None
    try:
        from vcap.models.loader import MODEL_CACHE

        MODEL_CACHE.unload()
    except Exception:
        pass


def _effective_model_limit(spec: JobSpec, model: ModelSpec, include_audio: bool) -> float | None:
    registry_limit = model.limits.compute_max_duration(
        spec.preprocess.fps,
        spec.preprocess.max_pixels,
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
    if spec.output.kind == "batch":
        return source, info, 0.0
    start = max(0.0, spec.preprocess.trim_start_s)
    end = spec.preprocess.trim_end_s
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
    emitter.log(f"Trimming {source.name}: {start:.3f}s to {end:.3f}s")
    trim_media(source, target, start, end, mode=spec.split.cut_mode, keep_audio=True)
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
    physical = len(segments) > 1 or persist
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
    if info.has_video:
        clips = split_video(
            source,
            segments,
            clip_dir,
            mode=spec.split.cut_mode,
            keep_audio=True,
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
                and modality in selected.modalities
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


def _model_gen(spec: JobSpec) -> Any:
    from vcap.models.base import GenParams as ModelGenParams

    return ModelGenParams(**asdict(spec.generation))


def _model_pre(spec: JobSpec, model: ModelSpec, override: Mapping[str, Any] | None = None) -> Any:
    from vcap.models.base import PreprocessParams

    values = {
        "fps": spec.preprocess.fps,
        "max_frames": spec.preprocess.max_frames,
        "max_pixels": spec.preprocess.max_pixels,
        "min_pixels": spec.preprocess.min_pixels or model.limits.min_pixels,
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
    if entry.kind == "video":
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


def _degrade_pre(current: dict[str, Any], model: ModelSpec) -> tuple[dict[str, Any], str] | None:
    minimum_pixels = int(current.get("min_pixels") or model.limits.min_pixels)
    pixels = int(current["max_pixels"])
    if pixels > minimum_pixels:
        lowered = max(minimum_pixels, int(math.floor(pixels * 0.75)))
        if lowered < pixels:
            updated = dict(current)
            updated["max_pixels"] = lowered
            return updated, f"max_pixels {pixels} -> {lowered}"
    frames = int(current["max_frames"])
    if frames > 4:
        lowered_frames = max(4, int(math.floor(frames * 0.75)))
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
        "max_frames": spec.preprocess.max_frames,
        "max_pixels": spec.preprocess.max_pixels,
        "min_pixels": spec.preprocess.min_pixels or model.limits.min_pixels,
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
            if not gpu.is_oom_error(exc) or retries >= 2:
                raise
            _empty_cuda_cache()
            degraded = _degrade_pre(pre_values, model)
            if degraded is None:
                raise
            pre_values, description = degraded
            retries += 1
            emitter.log(
                f"CUDA OOM during caption generation; cleared cache and reduced {description}. "
                f"Retry {retries}/2.",
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
    )


def _finalize_cue_text(spec: JobSpec, text: str) -> str:
    """Apply find/replace to a cue without caption-level text injection."""

    if not spec.post.replace_pairs:
        return str(text)
    return apply_replacements(
        str(text),
        spec.post.replace_pairs,
        regex=spec.post.replace_regex,
        case_insensitive=spec.post.replace_case_insensitive,
        whole_words=spec.post.replace_whole_words,
    )


def _context_excerpt(text: str, word_limit: int = 60) -> str:
    words = str(text).split()
    return " ".join(words[-max(1, int(word_limit)) :])


def _prompt_with_context(prompt: Any, previous_text: str, model: ModelSpec) -> Any:
    excerpt = _context_excerpt(previous_text)
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
    context = f"Context from the previous segment (do not repeat it): {json.dumps(excerpt, ensure_ascii=False)}"
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
            record: dict[str, Any] = {
                "index": segment.index,
                "start_s": trim_offset + segment.start_s,
                "end_s": trim_offset + segment.end_s,
                "media_path": str(segment.path) if segment.path is not None else None,
            }
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
                        start_s=segment.media_start,
                        end_s=segment.media_end,
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
                segment_prompt = _prompt_with_context(item_prompt, previous_final_text, model)
                emitter.log(
                    f"Applied previous-segment context to clip {segment.index}/{total_sources}.",
                    scope="prompts",
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
            final_text = _finalize_text(spec, raw_caption_text)
            local_cues = [
                Segment(float(start), float(end), _finalize_cue_text(spec, str(text)))
                for start, end, text in list(getattr(caption_result, "segments", []) or [])
            ]
            if not local_cues and final_text and segment.duration_s > 0:
                local_cues = [Segment(0.0, segment.duration_s, _finalize_cue_text(spec, raw_caption_text))]
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
                    structured=getattr(caption_result, "structured", None),
                    segments=local_cues,
                    reasoning=reasoning,
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
                structured=getattr(caption_result, "structured", None),
                outputs={key: str(path) for key, path in paths.items()},
                reasoning_saved=bool(reasoning and spec.post.save_reasoning),
                usage=usage,
                finish_reason=usage.get("finish_reason"),
                timing=_record_value(timing),
                peak_vram_gb=float(getattr(caption_result, "peak_vram_gb", 0.0) or 0.0),
            )
            accepted.append(record)
            previous_final_text = final_text

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
                elapsed=elapsed,
                peak_vram_gb=peak,
                gpu_index=spec.runtime.gpu_index,
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
        )
        elapsed = time.perf_counter() - started
        emitter.progress(entry.tracker_index, entry.result_index, "Item complete", 1.0, step_index=8)
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
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _run_job_local(spec: JobSpec, sinks: ProgressSink | None, cancel: CancelToken) -> JobResult:
    started = time.perf_counter()
    try:
        model = MODEL_SPECS[variant_to_family(spec.model.variant_key)]
        model_error = ""
    except KeyError as exc:
        model = next(iter(MODEL_SPECS.values()))
        model_error = str(exc)
    resolved = _resolve_inputs(spec)
    preassigned_outputs = _apply_preassigned_outputs(spec, resolved)
    if spec.output.kind == "batch" and not preassigned_outputs:
        _assign_batch_outputs(spec, resolved)
    run_dir = _allocate_job_dir(spec)
    if spec.output.kind == "single" and not preassigned_outputs:
        _assign_single_outputs(run_dir, resolved)
    tracker = ProgressTracker(len(resolved), [entry.stem for entry in resolved])
    emitter = _Emitter(sinks, tracker)
    results: dict[int, ItemResult] = {}
    peak_vram = 0.0
    metadata_path = run_dir / str(spec.internal.get("metadata_name") or "metadata.json")
    session = _ModelSession(spec, emitter, cancel)
    with RunLog(run_dir):
        emitter.log(
            f"Starting {spec.output.kind} job with {len(resolved)} resolved input(s) using "
            f"{spec.model.variant_key}."
        )
        if spec.output.kind == "batch" and (
            spec.preprocess.trim_start_s > 1e-9 or spec.preprocess.trim_end_s is not None
        ):
            emitter.log("Trim range is ignored for folder batches", "warning", "preprocess")
        if model_error:
            for entry in resolved:
                entry.status = "failed"
                entry.kind = "unknown"
                entry.message = f"Unknown model variant: {spec.model.variant_key}"
        else:
            _probe_and_classify(spec, resolved, model)
            _apply_batch_skip(spec, resolved)

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

        if not actionable:
            emitter.log("Nothing requires captioning; the model was not loaded.")
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
                continue
            emitter.start_item(entry.tracker_index, entry.stem, entry.result_index)
            item_started = time.perf_counter()
            try:
                result = _process_item(spec, entry, run_dir, model, session, emitter, cancel)
                results[entry.result_index] = result
                peak_vram = max(peak_vram, result.peak_vram_gb)
                emitter.finish(
                    entry.tracker_index,
                    entry.result_index,
                    result.status,
                    result.message,
                    result.elapsed,
                )
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

        if not spec.runtime.keep_model_loaded:
            session.unload()
            emitter.log("Model unloaded at the end of the job.", scope="models")
        ordered = sorted(results.values(), key=lambda item: item.index)
        elapsed = time.perf_counter() - started
        try:
            metadata_path = _write_metadata(
                spec,
                ordered,
                run_dir,
                elapsed,
                max(peak_vram, session.peak_vram_gb),
            )
        except Exception as exc:
            emitter.log(f"Could not write metadata: {exc}", "error", "metadata")
            raise
        counts = _status_counts(ordered)
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
    expanded = _resolve_inputs(spec)
    if not expanded:
        local_spec = replace(
            spec,
            runtime=replace(spec.runtime, gpu_indices=(spec.runtime.gpu_index,)),
        )
        return _run_job_local(local_spec, sinks, cancel)
    run_dir = _allocate_job_dir(spec)
    with RunLog(run_dir):
        get_log().log(
            f"Starting multi-GPU job on devices {', '.join(map(str, spec.runtime.gpu_indices))}.",
            scope="pipeline",
            console=os.environ.get("VCAP_WORKER", "") != "1",
        )
    if spec.output.kind == "single":
        _assign_single_outputs(run_dir, expanded)
    else:
        _assign_batch_outputs(spec, expanded)
    gpu_indices = tuple(spec.runtime.gpu_indices)
    partitions = _round_robin_partitions(expanded, len(gpu_indices))
    messages: Queue[tuple[int, str, Any]] = Queue()
    workers: dict[int, WorkerProcess] = {}
    threads: list[threading.Thread] = []

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

    active_partitions = [(gpu_index, part) for gpu_index, part in zip(gpu_indices, partitions) if part]
    for gpu_index, partition in active_partitions:
        thread = threading.Thread(
            target=pump,
            args=(gpu_index, partition),
            daemon=True,
            name=f"vcap-multi-gpu-{gpu_index}",
        )
        threads.append(thread)
        thread.start()
    result_items: list[ItemResult] = []
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
        _status_counts(result_items),
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


__all__ = ["loaded_variant_key", "run_job", "unload_cached_model"]
