"""Parent-side in-process/persistent-worker API used by the UI layer."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping

from vcap import APP_DIR
from vcap.core import console_progress
from vcap.core.progress import ProgressEvent, ProgressSink, UiThrottle
from vcap.core.subprocess_runner import (
    CancelToken,
    CancelledError,
    WorkerProcess,
    build_child_env,
)

from .job import JobResult, JobSpec
from .chat import ChatRequest, ChatResponse


_EOF = object()


class _NullSink:
    def on_log(self, message: str, level: str = "info", scope: str | None = None) -> None:
        del message, level, scope

    def on_progress(self, event: ProgressEvent) -> None:
        del event

    def on_item(self, event: ProgressEvent) -> None:
        del event


def _sink_call(method: Any, *args: Any, **kwargs: Any) -> None:
    try:
        method(*args, **kwargs)
    except Exception:
        pass


def _relay_app_log(
    message: str,
    *,
    level: str = "info",
    scope: str | None = None,
) -> None:
    """Mirror a worker log into the parent process independently of a UI sink."""

    try:
        from vcap.core.logs import get_log

        get_log().log(message, level=level, scope=scope)
    except Exception:
        pass


def _compile_child_plan(enabled: bool, gpu_index: int) -> tuple[dict[str, str], list[str], str]:
    """Probe compile support in a disposable child so the UI parent never imports Torch."""

    if not enabled:
        return {}, [], "eager"
    code = (
        "import json; "
        "from vcap.models.torch_compile import prepare_compile_env; "
        "p=prepare_compile_env(True); "
        "print('VCAP_COMPILE_PLAN '+json.dumps({'env_updates':p.env_updates,'warnings':p.warnings,'mode':p.mode}))"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(APP_DIR),
            env=build_child_env(gpu_index),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            check=False,
        )
    except Exception as exc:
        return {}, [f"Could not probe torch.compile environment: {exc}"], "eager"
    marker = "VCAP_COMPILE_PLAN "
    line = next(
        (value[len(marker) :] for value in reversed(completed.stdout.splitlines()) if value.startswith(marker)),
        None,
    )
    if line is None:
        detail = (completed.stderr or completed.stdout or f"exit code {completed.returncode}").strip()[-1000:]
        return {}, [f"Could not probe torch.compile environment: {detail}"], "eager"
    try:
        data = json.loads(line)
        updates = {str(key): str(value) for key, value in dict(data.get("env_updates") or {}).items()}
        warnings = [str(value) for value in data.get("warnings") or []]
        return updates, warnings, str(data.get("mode") or "eager")
    except Exception as exc:
        return {}, [f"Invalid torch.compile probe result: {exc}"], "eager"


class PipelineClient:
    """Run jobs locally or through one auto-restarting persistent worker."""

    def __init__(self, subprocess_mode: bool) -> None:
        self.subprocess_mode = bool(subprocess_mode)
        self._worker: WorkerProcess | None = None
        self._worker_gpu: int | None = None
        self._worker_compile: bool | None = None
        self._events: queue.Queue[Any] = queue.Queue()
        self._reader: threading.Thread | None = None
        self._run_lock = threading.RLock()
        self._state_lock = threading.RLock()
        self._busy = False
        self._last_activity = time.monotonic()
        self._idle_minutes = 0.0
        self._idle_unloaded = True
        self._idle_exited = False
        self._force_requested = threading.Event()
        self._external_cancel_at: float | None = None
        self._local_chat_cancel: CancelToken | None = None
        self._selected_variant: str | None = None
        self._release_pending = False
        self._resident_variant: str | None = None
        self._idle_released_variant: str | None = None
        self._worker_output_tail: deque[str] = deque(maxlen=1000)
        self._crash_logs_written: set[int] = set()
        self._shutdown = threading.Event()
        self._console_key = ("pipeline-client", id(self))
        self._console_throttle = UiThrottle(0.5)
        self._console_active = False
        self._idle_thread = threading.Thread(
            target=self._idle_loop,
            daemon=True,
            name="vcap-pipeline-idle",
        )
        self._idle_thread.start()

    def _reader_loop(self, worker: WorkerProcess, event_queue: queue.Queue[Any]) -> None:
        try:
            for event in worker.events():
                event_queue.put(event)
        finally:
            event_queue.put(_EOF)

    @staticmethod
    def _wait_for_worker_vram(gpu_index: int | None) -> None:
        if gpu_index is None:
            return
        try:
            from vcap.core import gpu

            previous = float(
                gpu.resource_snapshot(int(gpu_index)).get("vram_used_gb", 0.0)
                or 0.0
            )
        except Exception:
            return
        stable_samples = 0
        deadline = time.monotonic() + 3.0
        while stable_samples < 4 and time.monotonic() < deadline:
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
            try:
                current = float(
                    gpu.resource_snapshot(int(gpu_index)).get(
                        "vram_used_gb", 0.0
                    )
                    or 0.0
                )
            except Exception:
                return
            if abs(previous - current) < 0.05:
                stable_samples += 1
            else:
                stable_samples = 0
            previous = current

    def _stop_worker(self, *, graceful: bool = True, unload: bool = False) -> None:
        with self._state_lock:
            worker = self._worker
            busy = self._busy
            worker_gpu = self._worker_gpu
        if worker is None:
            return
        if worker.is_alive() and graceful:
            try:
                if unload and not busy:
                    worker.send({"cmd": "unload"})
                elif busy:
                    worker.send({"cmd": "cancel"})
                worker.send({"cmd": "exit"})
                worker.wait(timeout=5.5)
            except Exception:
                worker.kill_tree(grace=0.25)
        elif worker.is_alive():
            worker.kill_tree(grace=0.25)
        worker_exited = not worker.is_alive()
        with self._state_lock:
            if self._worker is worker:
                self._worker = None
                self._worker_gpu = None
                self._worker_compile = None
                self._reader = None
                self._resident_variant = None
        if worker_exited:
            self._wait_for_worker_vram(worker_gpu)

    def set_subprocess_mode(self, enabled: bool) -> None:
        """Switch execution modes without leaving a second model resident."""

        selected = bool(enabled)
        with self._state_lock:
            previous = self.subprocess_mode
            self.subprocess_mode = selected
            worker = self._worker
            busy = self._busy
        if not selected:
            if worker is not None:
                self._stop_worker(graceful=True, unload=True)
            with self._state_lock:
                self._idle_unloaded = True
                self._idle_exited = False
                self._last_activity = time.monotonic()
            return
        if previous or busy:
            return
        try:
            from .runner import unload_cached_model

            unload_cached_model()
        finally:
            with self._state_lock:
                self._idle_unloaded = True
                self._last_activity = time.monotonic()

    @staticmethod
    def _clock(seconds: Any) -> str:
        if seconds is None:
            return "unknown"
        try:
            total = max(0, int(round(float(seconds))))
        except (TypeError, ValueError, OverflowError):
            return "unknown"
        hours, remainder = divmod(total, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _render_progress_console(
        self,
        message: str,
        data: Mapping[str, Any],
        *,
        terminal: bool = False,
    ) -> None:
        if not self._console_throttle.should_emit(force=terminal):
            return
        speed = data.get("tok_per_s", data.get("tokens_per_second"))
        display_message = message
        message_parts = message.rsplit("|", 1)
        if speed is not None and len(message_parts) == 2 and message_parts[1].strip().casefold().endswith("tok/s"):
            display_message = message_parts[0].rstrip()
        line = (
            f"[{int(data.get('processed', 0))}/{int(data.get('total', 0))}] {display_message}"
            f" | elapsed {self._clock(data.get('elapsed_s'))}"
            f" | ETA {self._clock(data.get('eta_s'))}"
        )
        try:
            if speed is not None and float(speed) > 0:
                line += f" | {float(speed):.1f} tok/s"
        except (TypeError, ValueError):
            pass
        self._console_active = not terminal
        if terminal:
            console_progress.finalize_progress_line(self._console_key, line)
        else:
            console_progress.show_progress_line(line, key=self._console_key)

    def _forward(self, event: Mapping[str, Any], sink: ProgressSink) -> JobResult | None:
        kind = str(event.get("ev", ""))
        if kind in {"log", "stdout"}:
            level = str(event.get("level") or ("warning" if event.get("source") == "stderr" else "info"))
            message = str(event.get("text", ""))
            self._worker_output_tail.append(message)
            scope = event.get("scope")
            _relay_app_log(message, level=level, scope=scope)
            _sink_call(
                sink.on_log,
                message,
                level=level,
                scope=scope,
            )
        elif kind == "progress":
            fields = {
                key: event[key]
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
                if key in event
            }
            progress_event = ProgressEvent(**fields)
            self._render_progress_console(progress_event.message, progress_event.data)
            _sink_call(sink.on_progress, progress_event)
        elif kind == "item":
            item_event = ProgressEvent(
                message=str(event.get("message", "")),
                item_index=event.get("index"),
                status=event.get("status"),
                data=dict(event.get("data") or {}),
                kind="item",
            )
            if item_event.status != "running":
                terminal = int(item_event.data.get("remaining", 1)) == 0
                self._render_progress_console(item_event.message, item_event.data, terminal=terminal)
            _sink_call(sink.on_item, item_event)
        elif kind == "result":
            return JobResult.from_dict(event["job_result"])
        elif kind == "unloaded":
            pass
        return None

    def _persist_worker_crash(self, worker: WorkerProcess) -> None:
        pid = worker.pid
        if pid is None or pid in self._crash_logs_written:
            return
        self._crash_logs_written.add(pid)
        try:
            from vcap.core.logs import get_log

            path = get_log().write_worker_crash(
                pid,
                [
                    f"Pipeline worker {pid} exited with code {worker.returncode}.",
                    *self._worker_output_tail,
                ],
            )
            _relay_app_log(
                f"Pipeline worker crashed; captured output in {path}",
                level="error",
                scope="worker",
            )
        except Exception:
            pass

    def _ensure_worker(self, gpu_index: int, compile_enabled: bool, sink: ProgressSink) -> WorkerProcess:
        with self._state_lock:
            worker = self._worker
            reusable = (
                worker is not None
                and worker.is_alive()
                and self._worker_gpu == gpu_index
                and self._worker_compile == compile_enabled
            )
        if reusable:
            return worker  # type: ignore[return-value]
        self._stop_worker(graceful=True)
        env_updates, warnings, mode = _compile_child_plan(compile_enabled, gpu_index)
        if compile_enabled:
            _relay_app_log(
                f"torch.compile probe selected {mode}",
                level="info",
                scope="compile",
            )
            _sink_call(
                sink.on_log,
                f"torch.compile probe selected {mode}",
                level="info",
                scope="compile",
            )
            for warning in warnings:
                _relay_app_log(warning, level="warning", scope="compile")
                _sink_call(sink.on_log, warning, level="warning", scope="compile")
        worker = WorkerProcess().start(
            [sys.executable, "-u", "-m", "vcap.pipeline.worker", "--gpu", str(gpu_index)],
            cwd=APP_DIR,
            env=build_child_env(
                gpu_index,
                extra={**env_updates, "VCAP_CONSOLE_PROGRESS_PARENT": "1"},
            ),
            name=f"pipeline-gpu-{gpu_index}",
        )
        self._worker_output_tail.clear()
        event_queue: queue.Queue[Any] = queue.Queue()
        reader = threading.Thread(
            target=self._reader_loop,
            args=(worker, event_queue),
            daemon=True,
            name=f"vcap-worker-events-{gpu_index}",
        )
        with self._state_lock:
            self._worker = worker
            self._worker_gpu = gpu_index
            self._worker_compile = compile_enabled
            self._events = event_queue
            self._reader = reader
            self._idle_unloaded = False
            self._idle_exited = False
        reader.start()
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            try:
                event = event_queue.get(timeout=0.1)
            except queue.Empty:
                if not worker.is_alive():
                    break
                continue
            if event is _EOF:
                break
            if event.get("ev") == "ready":
                return worker
            self._forward(event, sink)
        self._persist_worker_crash(worker)
        self._stop_worker(graceful=False)
        raise RuntimeError("Caption worker did not become ready")

    def _drain_stale_events(self, sink: ProgressSink) -> None:
        while True:
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                return
            if event is _EOF:
                continue
            self._forward(event, sink)

    def _release_after_selection_change(self, used_variant: str) -> None:
        with self._state_lock:
            selected = self._selected_variant
            pending = self._release_pending
        # A selection made before a run is not a request to unload the model the
        # run just loaded. Only a selection that arrived while that run was busy
        # may release its resident variant when the run finishes.
        if not pending or selected is None or selected == str(used_variant):
            with self._state_lock:
                self._release_pending = False
            return
        try:
            outcome = self.release_model(unless_variant=selected)
            released = outcome.get("released") if isinstance(outcome, Mapping) else None
            if released is not None:
                from vcap.core.logs import get_log

                get_log().log(
                    f"Released {released} after the model selection changed to {selected}",
                    scope="models",
                )
            elif isinstance(outcome, Mapping) and outcome.get("error"):
                from vcap.core.logs import get_log

                get_log().warn(
                    f"Could not release the previous model after selecting {selected}: "
                    f"{outcome['error']}",
                    scope="models",
                )
        except Exception as exc:
            try:
                from vcap.core.logs import get_log

                get_log().warn(
                    f"Could not release the previous model after selecting {selected}: {exc}",
                    scope="models",
                )
            except Exception:
                pass
        finally:
            with self._state_lock:
                self._release_pending = False

    def run_job(
        self,
        spec: JobSpec,
        sinks: ProgressSink | None,
        cancel: CancelToken | None = None,
    ) -> JobResult:
        """Run a job and synchronously forward every worker event to ``sinks``."""

        self.record_job_variant(spec.model.variant_key)
        sink: ProgressSink = sinks or _NullSink()
        token = cancel or CancelToken()
        if not self.subprocess_mode:
            from .runner import run_job as run_in_process

            local_spec = replace(
                spec,
                runtime=replace(spec.runtime, subprocess_mode=False, gpu_indices=(spec.runtime.gpu_index,)),
            )
            with self._run_lock:
                with self._state_lock:
                    self._busy = True
                    self._idle_minutes = max(0.0, local_spec.runtime.idle_unload_minutes)
                    self._idle_unloaded = not local_spec.runtime.keep_model_loaded
                    self._idle_exited = False
                try:
                    result = run_in_process(local_spec, sink, token)
                    with self._state_lock:
                        self._resident_variant = (
                            local_spec.model.variant_key
                            if local_spec.runtime.keep_model_loaded
                            else None
                        )
                        self._idle_released_variant = None
                    return result
                finally:
                    with self._state_lock:
                        self._busy = False
                        self._last_activity = time.monotonic()
                        self._idle_unloaded = not local_spec.runtime.keep_model_loaded
                    self._release_after_selection_change(
                        local_spec.model.variant_key
                    )

        worker_spec = replace(spec, runtime=replace(spec.runtime, subprocess_mode=True))
        with self._run_lock:
            worker = self._ensure_worker(
                worker_spec.runtime.gpu_index,
                worker_spec.runtime.compile,
                sink,
            )
            self._drain_stale_events(sink)
            with self._state_lock:
                self._busy = True
                self._idle_minutes = max(0.0, worker_spec.runtime.idle_unload_minutes)
                self._idle_unloaded = False
                self._idle_exited = False
            self._force_requested.clear()
            with self._state_lock:
                self._external_cancel_at = None
            cancel_sent_at: float | None = None
            worker.send({"cmd": "run_job", "job": worker_spec.to_dict()})
            try:
                while True:
                    with self._state_lock:
                        external_cancel_at = self._external_cancel_at
                    if external_cancel_at is not None and cancel_sent_at is None:
                        cancel_sent_at = external_cancel_at
                    if token.is_cancelled() and cancel_sent_at is None:
                        try:
                            worker.send({"cmd": "cancel"})
                        except Exception:
                            pass
                        cancel_sent_at = time.monotonic()
                    if self._force_requested.is_set():
                        worker.kill_tree(grace=0.25)
                        raise CancelledError("Caption worker was force-cancelled")
                    if cancel_sent_at is not None and time.monotonic() - cancel_sent_at >= 5.0:
                        worker.kill_tree(grace=0.25)
                        raise CancelledError("Caption worker did not stop within the 5 second cancel grace period")
                    try:
                        event = self._events.get(timeout=0.1)
                    except queue.Empty:
                        if not worker.is_alive():
                            self._persist_worker_crash(worker)
                            raise RuntimeError(f"Caption worker exited unexpectedly (code {worker.returncode})")
                        continue
                    if event is _EOF:
                        if self._force_requested.is_set() or cancel_sent_at is not None:
                            raise CancelledError("Caption worker exited during cancellation")
                        self._persist_worker_crash(worker)
                        raise RuntimeError(f"Caption worker exited unexpectedly (code {worker.returncode})")
                    result = self._forward(event, sink)
                    if result is not None:
                        with self._state_lock:
                            self._resident_variant = (
                                worker_spec.model.variant_key
                                if worker_spec.runtime.keep_model_loaded
                                else None
                            )
                            self._idle_released_variant = None
                        return result
                    if event.get("ev") == "error":
                        message = str(event.get("message", "Worker error"))
                        trace = str(event.get("traceback", "")).strip()
                        raise RuntimeError(message + (f"\n{trace}" if trace else ""))
            finally:
                with self._state_lock:
                    self._busy = False
                    self._last_activity = time.monotonic()
                    self._external_cancel_at = None
                self._release_after_selection_change(
                    worker_spec.model.variant_key
                )
                if self._console_active:
                    console_progress.finalize_progress_line(self._console_key)
                    self._console_active = False

    def chat(
        self,
        request: ChatRequest | Mapping[str, Any],
        on_event: Callable[[dict[str, Any]], None] | None = None,
        cancel: CancelToken | None = None,
    ) -> ChatResponse:
        """Run one streamed assistant turn through the shared model worker/cache."""

        from .chat import run_chat

        selected = ChatRequest.from_dict(request)
        token = cancel or CancelToken()
        settings = selected.settings
        used_variant = str(settings.get("model_key") or "")
        self.record_job_variant(used_variant)
        requested_mode = bool(settings.get("subprocess_mode", self.subprocess_mode))
        if requested_mode != self.subprocess_mode:
            self.set_subprocess_mode(requested_mode)
        keep_loaded = bool(settings.get("keep_model_loaded", True))
        idle_minutes = max(0.0, float(settings.get("idle_unload_minutes", 10.0) or 0.0))
        console_key = ("pipeline-chat", id(self))

        def publish(event: Mapping[str, Any]) -> None:
            item = dict(event)
            if item.get("ev") in {"log", "stdout"}:
                level = str(
                    item.get("level")
                    or ("warning" if item.get("source") == "stderr" else "info")
                )
                _relay_app_log(
                    str(item.get("text") or ""),
                    level=level,
                    scope=item.get("scope"),
                )
            if on_event is not None:
                try:
                    on_event(item)
                except Exception:
                    pass
            if item.get("ev") != "status":
                return
            message = str(item.get("message") or "Chat is running")
            data = dict(item.get("data") or {})
            speed = data.get("tok_per_s", data.get("tokens_per_second"))
            display_message = message
            message_parts = message.rsplit("|", 1)
            if (
                speed is not None
                and len(message_parts) == 2
                and message_parts[1].strip().casefold().endswith("tok/s")
            ):
                display_message = message_parts[0].rstrip()
            line = f"Chat | {display_message}"
            try:
                if speed is not None and float(speed) > 0:
                    line += f" | {float(speed):.1f} tok/s"
            except (TypeError, ValueError):
                pass
            terminal = bool(data.get("finish_reason"))
            if terminal:
                console_progress.finalize_progress_line(console_key, line)
            else:
                console_progress.show_progress_line(line, key=console_key)

        if not self.subprocess_mode:
            with self._run_lock:
                with self._state_lock:
                    self._busy = True
                    self._local_chat_cancel = token
                    self._idle_minutes = idle_minutes
                    self._idle_unloaded = not keep_loaded
                    self._idle_exited = False
                try:
                    response = run_chat(selected, publish, token)
                    with self._state_lock:
                        self._resident_variant = used_variant if keep_loaded else None
                        self._idle_released_variant = None
                    return response
                finally:
                    with self._state_lock:
                        self._busy = False
                        self._local_chat_cancel = None
                        self._last_activity = time.monotonic()
                        self._idle_unloaded = not keep_loaded
                    self._release_after_selection_change(used_variant)
                    console_progress.finalize_progress_line(console_key)

        gpu_index = max(0, int(settings.get("gpu_index", 0) or 0))
        compile_enabled = bool(settings.get("torch_compile", settings.get("compile", False)))
        with self._run_lock:
            worker = self._ensure_worker(gpu_index, compile_enabled, _NullSink())
            self._drain_stale_events(_NullSink())
            with self._state_lock:
                self._busy = True
                self._local_chat_cancel = token
                self._idle_minutes = idle_minutes
                self._idle_unloaded = False
                self._idle_exited = False
                self._external_cancel_at = None
            self._force_requested.clear()
            cancel_sent = False
            worker.send({"cmd": "chat", "payload": selected.to_dict()})
            try:
                while True:
                    with self._state_lock:
                        externally_cancelled = self._external_cancel_at is not None
                    if (token.is_cancelled() or externally_cancelled) and not cancel_sent:
                        try:
                            worker.send({"cmd": "cancel"})
                        except Exception:
                            pass
                        cancel_sent = True
                    if self._force_requested.is_set():
                        worker.kill_tree(grace=0.25)
                        raise CancelledError("Chat worker was force-cancelled")
                    try:
                        event = self._events.get(timeout=0.1)
                    except queue.Empty:
                        if not worker.is_alive():
                            self._persist_worker_crash(worker)
                            raise RuntimeError(
                                f"Chat worker exited unexpectedly (code {worker.returncode})"
                            )
                        continue
                    if event is _EOF:
                        self._persist_worker_crash(worker)
                        raise RuntimeError(
                            f"Chat worker exited unexpectedly (code {worker.returncode})"
                        )
                    publish(event)
                    kind = str(event.get("ev") or "")
                    if kind == "chat_result":
                        with self._state_lock:
                            self._resident_variant = used_variant if keep_loaded else None
                            self._idle_released_variant = None
                        return ChatResponse.from_dict(event.get("result") or {})
                    if kind == "error":
                        message = str(event.get("message") or "Chat worker error")
                        trace = str(event.get("traceback") or "").strip()
                        raise RuntimeError(message + (f"\n{trace}" if trace else ""))
            finally:
                with self._state_lock:
                    self._busy = False
                    self._local_chat_cancel = None
                    self._last_activity = time.monotonic()
                    self._external_cancel_at = None
                    self._idle_unloaded = not keep_loaded
                self._release_after_selection_change(used_variant)
                console_progress.finalize_progress_line(console_key)

    def cancel(self, force: bool = False) -> None:
        """Request cooperative cancellation or immediately kill the worker tree."""

        with self._state_lock:
            worker = self._worker
            busy = self._busy
            local_chat_cancel = self._local_chat_cancel
        if local_chat_cancel is not None:
            local_chat_cancel.cancel()
        if worker is None or not worker.is_alive() or not busy:
            return
        if force:
            self._force_requested.set()
            worker.kill_tree(grace=0.25)
            return
        try:
            worker.send({"cmd": "cancel"})
            with self._state_lock:
                self._external_cancel_at = time.monotonic()
        except Exception:
            worker.kill_tree(grace=0.25)

    def release_model(
        self,
        unless_variant: str | None = None,
        timeout_s: float = 30.0,
        lock_timeout_s: float = 2.0,
    ) -> dict[str, Any]:
        """Release the resident model while keeping an idle worker available."""

        with self._state_lock:
            if self._busy:
                return {"busy": True}
        # A health ping or a job that is still spawning its worker can hold the run
        # lock for a moment; wait briefly so a selection change is not dropped.
        if not self._run_lock.acquire(timeout=max(0.0, float(lock_timeout_s))):
            return {"busy": True}

        deferred: list[Any] = []
        try:
            with self._state_lock:
                if self._busy:
                    return {"busy": True}
                worker = self._worker

            if not self.subprocess_mode:
                try:
                    from .runner import loaded_variant_key, unload_cached_model

                    resident = loaded_variant_key()
                    outcome = unload_cached_model(
                        unless_variant=unless_variant
                    )
                except Exception as exc:
                    return {"error": str(exc)}
                has_error = isinstance(outcome, Mapping) and "error" in outcome
                released = (
                    outcome.get("variant_key")
                    if isinstance(outcome, Mapping) and not has_error
                    else None
                )
                result = {
                    "ev": "unloaded",
                    "resident": resident,
                    "released": released,
                    "skipped": resident is not None and outcome is None,
                    "report": outcome,
                }
                if released is not None:
                    with self._state_lock:
                        self._idle_unloaded = True
                        self._resident_variant = None
                        self._last_activity = time.monotonic()
                return result

            if worker is None or not worker.is_alive():
                return {
                    "ev": "unloaded",
                    "resident": None,
                    "released": None,
                    "skipped": False,
                    "report": None,
                }

            try:
                worker.send(
                    {
                        "cmd": "unload",
                        "unless_variant": unless_variant,
                    }
                )
                deadline = time.monotonic() + max(0.05, float(timeout_s))
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return {"error": "Model release timed out"}
                    try:
                        event = self._events.get(timeout=min(0.1, remaining))
                    except queue.Empty:
                        continue
                    if isinstance(event, Mapping):
                        kind = str(event.get("ev") or "")
                        if kind == "unloaded":
                            result = dict(event)
                            if result.get("released") is not None:
                                with self._state_lock:
                                    self._idle_unloaded = True
                                    self._resident_variant = None
                                    self._last_activity = time.monotonic()
                            return result
                        if kind == "error":
                            return {
                                "error": str(
                                    event.get("message") or "Worker error"
                                )
                            }
                        if kind in {"log", "stdout"}:
                            self._forward(event, _NullSink())
                            continue
                    deferred.append(event)
            except Exception as exc:
                return {"error": str(exc)}
        finally:
            for event in deferred:
                self._events.put(event)
            self._run_lock.release()

    def record_variant_selection(self, variant_key: str) -> bool:
        """Record the latest UI selection without waiting for model release."""

        selected = str(variant_key)
        with self._state_lock:
            self._selected_variant = selected
            if self._busy:
                self._release_pending = True
                return True
        return False

    def record_job_variant(self, variant_key: str) -> None:
        """Make the submitted variant authoritative over stale pre-run UI events."""

        selected = str(variant_key)
        with self._state_lock:
            self._selected_variant = selected
            if not self._busy:
                self._release_pending = False

    def select_variant(self, variant_key: str) -> dict[str, Any]:
        """Record a model selection and release any different resident model."""

        selected = str(variant_key)
        if self.record_variant_selection(selected):
            return {"busy": True, "deferred": True}
        return self.release_recorded_variant(selected)

    def release_recorded_variant(self, variant_key: str) -> dict[str, Any]:
        """Release for a selection only while it is still the latest choice."""

        selected = str(variant_key)
        with self._state_lock:
            if self._selected_variant != selected:
                return {"superseded": True}
            if self._busy:
                self._release_pending = True
                return {"busy": True, "deferred": True}
            self._release_pending = False
        return self.release_model(unless_variant=selected)

    def unload(self) -> None:
        """Release the idle worker or in-app model cache."""

        with self._state_lock:
            worker = self._worker
            busy = self._busy
        if worker is not None and worker.is_alive() and not busy:
            worker.send({"cmd": "unload"})
            with self._state_lock:
                self._idle_unloaded = True
                self._resident_variant = None
        elif not busy and not self.subprocess_mode:
            from .runner import unload_cached_model

            unload_cached_model()
            with self._state_lock:
                self._idle_unloaded = True
                self._resident_variant = None

    def ping(self, timeout_s: float = 0.6) -> dict[str, Any]:
        """Report the resident model and its block-swap summary without loading anything.

        Returns a ``pong`` mapping (``loaded_variant`` and ``block_swap`` may be
        ``None``), ``{"busy": True}`` while a job owns the worker, or
        ``{"error": ...}`` when the worker did not answer in time.
        """

        if not self.subprocess_mode:
            try:
                from .runner import loaded_block_swap_summary, loaded_variant_key

                return {
                    "ev": "pong",
                    "loaded_variant": loaded_variant_key(),
                    "block_swap": loaded_block_swap_summary(),
                }
            except Exception as exc:
                return {"error": str(exc)}

        with self._state_lock:
            worker = self._worker
            busy = self._busy
        if busy:
            return {"busy": True}
        if worker is None or not worker.is_alive():
            return {"ev": "pong", "loaded_variant": None, "block_swap": None}
        if not self._run_lock.acquire(blocking=False):
            return {"busy": True}
        deferred: list[Any] = []
        try:
            with self._state_lock:
                if self._busy:
                    return {"busy": True}
            worker.send({"cmd": "ping"})
            deadline = time.monotonic() + max(0.05, float(timeout_s))
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    event = self._events.get(timeout=min(0.1, remaining))
                except queue.Empty:
                    continue
                if isinstance(event, Mapping) and event.get("ev") == "pong":
                    return dict(event)
                if isinstance(event, Mapping) and event.get("ev") in {"log", "stdout"}:
                    self._forward(event, _NullSink())
                    continue
                deferred.append(event)
            return {"error": "Worker health ping timed out"}
        except Exception as exc:
            return {"error": str(exc)}
        finally:
            for event in deferred:
                self._events.put(event)
            self._run_lock.release()

    def _idle_loop(self) -> None:
        while not self._shutdown.wait(0.25):
            with self._state_lock:
                worker = self._worker
                busy = self._busy
                idle_minutes = self._idle_minutes
                idle_for = time.monotonic() - self._last_activity
                unloaded = self._idle_unloaded
                exited = self._idle_exited
                resident_variant = self._resident_variant
                released_variant = self._idle_released_variant
            if busy or idle_minutes <= 0:
                continue
            threshold = idle_minutes * 60.0
            if worker is None or not worker.is_alive():
                if self.subprocess_mode or unloaded or idle_for < threshold:
                    continue
                try:
                    from .runner import unload_cached_model

                    unload_cached_model()
                    with self._state_lock:
                        self._idle_unloaded = True
                        self._resident_variant = None
                        self._idle_released_variant = resident_variant
                    if resident_variant:
                        _relay_app_log(
                            f"Idle for {idle_minutes:g} min: released {resident_variant} "
                            "from the in-process cache",
                            scope="models",
                        )
                    released_variant = resident_variant
                except Exception:
                    pass
                continue
            if not unloaded and idle_for >= threshold:
                try:
                    worker.send({"cmd": "unload"})
                    with self._state_lock:
                        self._idle_unloaded = True
                        self._resident_variant = None
                        self._idle_released_variant = resident_variant
                    if resident_variant:
                        _relay_app_log(
                            f"Idle for {idle_minutes:g} min: released {resident_variant}; "
                            "the worker remains ready for reuse",
                            scope="models",
                        )
                    released_variant = resident_variant
                except Exception:
                    pass
            if not exited and idle_for >= threshold * 2.0:
                try:
                    worker.send({"cmd": "exit"})
                    with self._state_lock:
                        self._idle_exited = True
                    variant = released_variant or resident_variant or "resident model"
                    _relay_app_log(
                        f"Idle for {idle_minutes:g} min: released {variant} and stopped the worker",
                        scope="models",
                    )
                except Exception:
                    worker.kill_tree(grace=0.25)

    def shutdown(self) -> None:
        """Stop the idle timer and terminate the persistent worker tree."""

        self._shutdown.set()
        self._stop_worker(graceful=True)
        if not self.subprocess_mode:
            try:
                from .runner import unload_cached_model

                unload_cached_model()
            except Exception:
                pass
        if self._idle_thread is not threading.current_thread():
            self._idle_thread.join(timeout=1.0)

    def __enter__(self) -> "PipelineClient":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.shutdown()


__all__ = ["PipelineClient"]
