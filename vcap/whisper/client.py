"""Parent-side client for the Whisper JSON-lines worker."""

from __future__ import annotations

import json
import os
import queue
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from vcap import APP_DIR, MODELS_DIR, TEMP_DIR
from vcap.core import console_progress
from vcap.core.logs import get_log
from vcap.core.paths import normalize_path
from vcap.core.subprocess_runner import (
    CancelToken,
    WorkerProcess,
    build_child_env,
    iter_json_lines,
    kill_process_tree,
)

from .engine import TranscriptResult
from .params import TranscriptOutputOptions, WhisperParams


class TranscriptionSink(Protocol):
    def on_log(self, message: str, level: str = "info") -> None: ...
    def on_download(self, payload: dict) -> None: ...
    def on_progress(self, payload: dict) -> None: ...
    def on_segment(self, payload: dict) -> None: ...
    def on_item_done(self, payload: dict) -> None: ...
    def on_item_error(self, payload: dict) -> None: ...


@dataclass
class TranscriptionOutcome:
    ok: bool
    items: list[dict]
    results: dict[int, TranscriptResult]
    elapsed_s: float
    cancelled: bool
    error: str | None


def _normalized_item(raw: Mapping[str, Any], fallback_index: int) -> dict[str, Any]:
    item = dict(raw)
    item["index"] = int(item.get("index", fallback_index))
    if item.get("path"):
        item["path"] = str(normalize_path(item["path"]))
    if item.get("out_dir"):
        item["out_dir"] = str(normalize_path(item["out_dir"]))
    item.setdefault("trim_start_s", 0.0)
    item.setdefault("trim_end_s", None)
    return item


def build_request(
    params: WhisperParams,
    output: TranscriptOutputOptions,
    items: list[dict],
    *,
    models_dir: Path | None = None,
    skip_existing: bool = False,
    action: str = "transcribe",
) -> dict:
    """Build the stable JSON-safe worker request payload."""

    normalized_action = str(action).strip().casefold()
    if normalized_action not in {"transcribe", "ensure_model", "probe_runtime"}:
        raise ValueError(f"Unsupported Whisper action: {action}")
    return {
        "action": normalized_action,
        "params": params.to_dict(),
        "output": output.to_dict(),
        "models_dir": str(normalize_path(models_dir or MODELS_DIR)),
        "items": [
            _normalized_item(item, index)
            for index, item in enumerate(items)
            if isinstance(item, Mapping)
        ],
        "skip_existing": bool(skip_existing),
    }


def _python_executable(value: str | None) -> str:
    if value:
        return str(normalize_path(value))
    venv_python = APP_DIR / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return str(venv_python.resolve(strict=False) if venv_python.is_file() else Path(sys.executable))


def _kill_worker_tree(worker: WorkerProcess, grace: float = 0.2) -> None:
    """Stop through the wrapper, then retry its shared tree killer if needed."""

    try:
        worker.kill_tree(grace=grace)
    except Exception:
        pass
    if not worker.is_alive():
        return
    process = getattr(worker, "_process", None)
    pid = getattr(process, "pid", None)
    if pid is not None:
        kill_process_tree(int(pid), grace=grace)


def _sink_call(sink: TranscriptionSink | None, method: str, *args: Any) -> None:
    callback = getattr(sink, method, None) if sink is not None else None
    if callable(callback):
        try:
            callback(*args)
        except Exception as exc:
            get_log().warn(f"Whisper event sink {method} failed: {exc}", scope="whisper")


def _mirror_log(message: str, level: str) -> None:
    logger = get_log()
    if level == "warning":
        logger.warn(message, scope="whisper")
    elif level == "error":
        logger.error(message, scope="whisper")
    else:
        logger.log(message, level=level, scope="whisper")


def _handle_event(
    event: Mapping[str, Any],
    sink: TranscriptionSink | None,
    item_payloads: list[dict],
    results: dict[int, TranscriptResult],
) -> tuple[dict[str, Any] | None, str | None]:
    kind = str(event.get("event") or event.get("ev") or "")
    payload = dict(event)
    payload.pop("ev", None)
    if kind == "stdout":
        text = str(payload.get("text") or "").strip()
        if text:
            _mirror_log(text, "info")
            _sink_call(sink, "on_log", text, "info")
        return None, None
    payload["event"] = kind
    if kind == "log":
        message = str(payload.get("message") or payload.get("text") or "")
        level = str(payload.get("level") or "info").casefold()
        _mirror_log(message, level)
        _sink_call(sink, "on_log", message, level)
    elif kind == "download":
        _sink_call(sink, "on_download", payload)
    elif kind == "progress":
        _sink_call(sink, "on_progress", payload)
        console_progress.show_progress_line(
            str(payload.get("message") or "Whisper transcription"),
            key=("whisper", payload.get("item_index", 0)),
            min_interval=0.25,
        )
    elif kind == "segment":
        _sink_call(sink, "on_segment", payload)
    elif kind == "item_done":
        item_payloads.append(payload)
        raw_result = payload.get("result")
        if isinstance(raw_result, Mapping):
            results[int(payload.get("item_index", 0))] = TranscriptResult.from_dict(raw_result)
        _sink_call(sink, "on_item_done", payload)
        console_progress.finalize_progress_line(
            key=("whisper", payload.get("item_index", 0)),
            final_text=str(payload.get("files", ["Transcription complete"])[0]),
        )
    elif kind == "item_error":
        item_payloads.append(payload)
        _sink_call(sink, "on_item_error", payload)
        message = str(payload.get("message") or "Whisper item failed")
        _mirror_log(message, "error")
        console_progress.finalize_progress_line(
            key=("whisper", payload.get("item_index", 0)),
            final_text=message,
        )
    elif kind in {"runtime", "model_loaded"}:
        message = str(payload.get("message") or "")
        if not message and kind == "model_loaded":
            message = (
                f"Whisper model {payload.get('model', '')} loaded in "
                f"{float(payload.get('load_s') or 0.0):.1f}s"
            )
        if message:
            _mirror_log(message, "info")
            _sink_call(sink, "on_log", message, "info")
    elif kind == "done":
        return payload, None
    elif kind == "error":
        message = str(payload.get("message") or "Whisper worker failed")
        _mirror_log(message, "error")
        _sink_call(sink, "on_log", message, "error")
        return None, message
    return None, None


def run_transcription(
    request: dict,
    *,
    sink: TranscriptionSink | None = None,
    cancel: CancelToken | None = None,
    request_dir: Path | None = None,
    python: str | None = None,
    timeout_s: float | None = None,
) -> TranscriptionOutcome:
    """Spawn the Whisper worker, forward events, and assemble its outcome."""

    started = time.monotonic()
    cancel_token = cancel or CancelToken()
    directory = normalize_path(request_dir or (TEMP_DIR / "whisper"))
    directory.mkdir(parents=True, exist_ok=True)
    descriptor, request_name = tempfile.mkstemp(
        prefix="request_", suffix=".json", dir=directory
    )
    os.close(descriptor)
    request_path = normalize_path(request_name)
    request_path.write_text(
        json.dumps(request, indent=2, ensure_ascii=False),
        encoding="utf-8",
        newline="\n",
    )
    params = WhisperParams.from_dict(request.get("params") or {})
    extra_env: dict[str, Any] = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    if params.device == "cpu":
        extra_env["CUDA_VISIBLE_DEVICES"] = None
        environment = build_child_env(extra=extra_env)
    else:
        environment = build_child_env(params.gpu_index, extra=extra_env)
    command = [
        _python_executable(python),
        "-m",
        "vcap.whisper.worker",
        "--request",
        str(request_path),
    ]
    worker = WorkerProcess()
    item_payloads: list[dict] = []
    results: dict[int, TranscriptResult] = {}
    done_payload: dict[str, Any] | None = None
    fatal_error: str | None = None
    cancelled = False
    timed_out = False
    events_queue: queue.Queue[dict[str, Any] | None] = queue.Queue()

    def read_events() -> None:
        try:
            # WorkerProcess.events delegates parsing to iter_json_lines. Keep the
            # imported parser part of this client contract for alternate workers.
            _ = iter_json_lines
            for event in worker.events():
                if (
                    str(event.get("ev") or "") == "stdout"
                    and cancel_token.is_cancelled()
                ):
                    continue
                events_queue.put(event)
        finally:
            events_queue.put(None)

    try:
        worker.start(command, cwd=APP_DIR, env=environment, name="whisper")
        reader = threading.Thread(target=read_events, name="whisper-events", daemon=True)
        reader.start()
        output_finished = False
        cancel_sent_at: float | None = None
        while not output_finished:
            now = time.monotonic()
            if timeout_s is not None and timeout_s > 0 and now - started > timeout_s:
                timed_out = True
                fatal_error = f"Whisper worker timed out after {float(timeout_s):g} seconds"
                _kill_worker_tree(worker)
                break
            if cancel_token.is_cancelled() and cancel_sent_at is None:
                cancelled = True
                cancel_sent_at = now
                try:
                    worker.send("cancel")
                except RuntimeError:
                    pass
            if cancel_sent_at is not None and now - cancel_sent_at >= 3.0 and worker.is_alive():
                _kill_worker_tree(worker)
            try:
                event = events_queue.get(timeout=0.1)
            except queue.Empty:
                if not worker.is_alive() and not reader.is_alive():
                    break
                continue
            if event is None:
                output_finished = True
                continue
            completed, error = _handle_event(event, sink, item_payloads, results)
            if completed is not None:
                done_payload = completed
                cancelled = cancelled or bool(completed.get("cancelled", False))
            if error is not None:
                fatal_error = error
        reader.join(timeout=2.0)
        if worker.is_alive():
            try:
                worker.wait(timeout=3.0)
            except Exception:
                _kill_worker_tree(worker)
        return_code = worker.returncode
        if (
            fatal_error is None
            and not cancelled
            and not timed_out
            and done_payload is None
            and return_code not in {0, None}
        ):
            fatal_error = f"Whisper worker exited with code {return_code}"
    except OSError as exc:
        fatal_error = f"Could not start Whisper worker: {exc}"
        _mirror_log(fatal_error, "error")
    finally:
        if worker.is_alive():
            _kill_worker_tree(worker)
        try:
            request_path.unlink()
        except OSError:
            pass

    item_payloads.sort(key=lambda item: int(item.get("item_index", 0)))
    elapsed = (
        float(done_payload.get("elapsed_s") or 0.0)
        if done_payload is not None
        else time.monotonic() - started
    )
    ok = bool(done_payload and done_payload.get("ok")) and not cancelled and fatal_error is None
    return TranscriptionOutcome(
        ok=ok,
        items=item_payloads,
        results=results,
        elapsed_s=elapsed,
        cancelled=cancelled,
        error=fatal_error,
    )


__all__ = [
    "TranscriptionOutcome",
    "TranscriptionSink",
    "build_request",
    "run_transcription",
]
