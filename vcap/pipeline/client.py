"""Parent-side in-process/persistent-worker API used by the UI layer."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from vcap import APP_DIR
from vcap.core.progress import ProgressEvent, ProgressSink
from vcap.core.subprocess_runner import (
    CancelToken,
    CancelledError,
    WorkerProcess,
    build_child_env,
)

from .job import JobResult, JobSpec


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
        self._idle_unloaded = False
        self._idle_exited = False
        self._force_requested = threading.Event()
        self._external_cancel_at: float | None = None
        self._shutdown = threading.Event()
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

    def _stop_worker(self, *, graceful: bool = True) -> None:
        with self._state_lock:
            worker = self._worker
        if worker is None:
            return
        if worker.is_alive() and graceful:
            try:
                worker.send({"cmd": "exit"})
                worker.wait(timeout=5.5)
            except Exception:
                worker.kill_tree(grace=0.25)
        elif worker.is_alive():
            worker.kill_tree(grace=0.25)
        with self._state_lock:
            if self._worker is worker:
                self._worker = None
                self._worker_gpu = None
                self._worker_compile = None
                self._reader = None

    def _forward(self, event: Mapping[str, Any], sink: ProgressSink) -> JobResult | None:
        kind = str(event.get("ev", ""))
        if kind in {"log", "stdout"}:
            level = str(event.get("level") or ("warning" if event.get("source") == "stderr" else "info"))
            _sink_call(
                sink.on_log,
                str(event.get("text", "")),
                level=level,
                scope=event.get("scope"),
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
            _sink_call(sink.on_progress, ProgressEvent(**fields))
        elif kind == "item":
            _sink_call(
                sink.on_item,
                ProgressEvent(
                    message=str(event.get("message", "")),
                    item_index=event.get("index"),
                    status=event.get("status"),
                    data=dict(event.get("data") or {}),
                    kind="item",
                ),
            )
        elif kind == "result":
            return JobResult.from_dict(event["job_result"])
        return None

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
            _sink_call(sink.on_log, f"torch.compile probe selected {mode}", level="info", scope="compile")
            for warning in warnings:
                _sink_call(sink.on_log, warning, level="warning", scope="compile")
        worker = WorkerProcess().start(
            [sys.executable, "-u", "-m", "vcap.pipeline.worker", "--gpu", str(gpu_index)],
            cwd=APP_DIR,
            env=build_child_env(gpu_index, extra=env_updates),
            name=f"pipeline-gpu-{gpu_index}",
        )
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

    def run_job(
        self,
        spec: JobSpec,
        sinks: ProgressSink | None,
        cancel: CancelToken | None = None,
    ) -> JobResult:
        """Run a job and synchronously forward every worker event to ``sinks``."""

        sink: ProgressSink = sinks or _NullSink()
        token = cancel or CancelToken()
        if not self.subprocess_mode:
            from .runner import run_job as run_in_process

            local_spec = replace(
                spec,
                runtime=replace(spec.runtime, subprocess_mode=False, gpu_indices=(spec.runtime.gpu_index,)),
            )
            return run_in_process(local_spec, sink, token)

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
                            raise RuntimeError(f"Caption worker exited unexpectedly (code {worker.returncode})")
                        continue
                    if event is _EOF:
                        if self._force_requested.is_set() or cancel_sent_at is not None:
                            raise CancelledError("Caption worker exited during cancellation")
                        raise RuntimeError(f"Caption worker exited unexpectedly (code {worker.returncode})")
                    result = self._forward(event, sink)
                    if result is not None:
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

    def cancel(self, force: bool = False) -> None:
        """Request cooperative cancellation or immediately kill the worker tree."""

        with self._state_lock:
            worker = self._worker
            busy = self._busy
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

    def unload(self) -> None:
        """Ask an idle persistent worker to release its model."""

        with self._state_lock:
            worker = self._worker
            busy = self._busy
        if worker is not None and worker.is_alive() and not busy:
            worker.send({"cmd": "unload"})
            with self._state_lock:
                self._idle_unloaded = True

    def _idle_loop(self) -> None:
        while not self._shutdown.wait(0.25):
            with self._state_lock:
                worker = self._worker
                busy = self._busy
                idle_minutes = self._idle_minutes
                idle_for = time.monotonic() - self._last_activity
                unloaded = self._idle_unloaded
                exited = self._idle_exited
            if worker is None or not worker.is_alive() or busy or idle_minutes <= 0:
                continue
            threshold = idle_minutes * 60.0
            if not unloaded and idle_for >= threshold:
                try:
                    worker.send({"cmd": "unload"})
                    with self._state_lock:
                        self._idle_unloaded = True
                except Exception:
                    pass
            if not exited and idle_for >= threshold * 2.0:
                try:
                    worker.send({"cmd": "exit"})
                    with self._state_lock:
                        self._idle_exited = True
                except Exception:
                    worker.kill_tree(grace=0.25)

    def shutdown(self) -> None:
        """Stop the idle timer and terminate the persistent worker tree."""

        self._shutdown.set()
        self._stop_worker(graceful=True)
        if self._idle_thread is not threading.current_thread():
            self._idle_thread.join(timeout=1.0)

    def __enter__(self) -> "PipelineClient":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.shutdown()


__all__ = ["PipelineClient"]
