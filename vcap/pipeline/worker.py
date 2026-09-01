"""Persistent JSON-lines caption worker entry point."""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import re
import select
import sys
import threading
import time
import traceback
from dataclasses import asdict
from typing import Any, Mapping, TextIO

from vcap.core.logs import setup_utf8_stdio
from vcap.core.progress import ProgressEvent, ProgressSink
from vcap.core.subprocess_runner import CancelToken

from .job import JobSpec


_NO_INPUT = object()
_STDIN_BUFFER = bytearray()
_STDIN_EOF = False
_STAMPED_LOG_RE = re.compile(
    r"^\[(?P<timestamp>\d{2}:\d{2}:\d{2})\](?: \[(?P<scope>[^\]]+)\])? (?P<text>.*)$"
)


class _ProtocolWriter:
    def __init__(self, stream: TextIO) -> None:
        self.stream = stream
        self.lock = threading.RLock()
        self._recent_logs: dict[tuple[str, str, str], float] = {}

    def emit(self, event: dict[str, Any]) -> None:
        event = dict(event)
        if event.get("ev") == "stdout":
            match = _STAMPED_LOG_RE.fullmatch(str(event.get("text", "")))
            if match is not None:
                event = {
                    "ev": "log",
                    "level": "warning" if event.get("source") == "stderr" else "info",
                    "text": match.group("text"),
                    "timestamp": match.group("timestamp"),
                }
                if match.group("scope"):
                    event["scope"] = match.group("scope")
        payload = json.dumps(event, ensure_ascii=False, default=str, separators=(",", ":"))
        with self.lock:
            if event.get("ev") == "log":
                now = time.monotonic()
                key = (
                    str(event.get("level") or "info"),
                    str(event.get("scope") or ""),
                    str(event.get("text") or ""),
                )
                previous = self._recent_logs.get(key)
                if previous is not None and now - previous < 0.5:
                    return
                self._recent_logs[key] = now
                if len(self._recent_logs) > 64:
                    cutoff = now - 2.0
                    self._recent_logs = {
                        item: seen for item, seen in self._recent_logs.items() if seen >= cutoff
                    }
            self.stream.write(payload + "\n")
            self.stream.flush()


class _ProtocolBuffer(io.RawIOBase):
    def __init__(self, owner: "_RedirectStream") -> None:
        self.owner = owner

    def writable(self) -> bool:
        return True

    def write(self, data: bytes | bytearray) -> int:
        raw = bytes(data)
        self.owner.write(raw.decode("utf-8", errors="replace"))
        return len(raw)

    def flush(self) -> None:
        self.owner.flush()


class _RedirectStream(io.TextIOBase):
    """Line-buffer stray output into protocol-safe ``stdout`` events."""

    def __init__(self, protocol: _ProtocolWriter, source: str) -> None:
        super().__init__()
        self.protocol = protocol
        self.source = source
        self._buffer = ""
        self._lock = threading.RLock()
        self.buffer = _ProtocolBuffer(self)  # type: ignore[assignment]

    @property
    def encoding(self) -> str:
        return "utf-8"

    @property
    def errors(self) -> str:
        return "replace"

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return False

    def fileno(self) -> int:
        return -1

    def reconfigure(self, **kwargs: Any) -> None:
        del kwargs

    def write(self, value: str) -> int:
        text = str(value)
        with self._lock:
            self._buffer += text
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                if line.endswith("\r"):
                    line = line[:-1]
                if line:
                    self.protocol.emit({"ev": "stdout", "text": line, "source": self.source})
        return len(text)

    def flush(self) -> None:
        with self._lock:
            if self._buffer:
                line, self._buffer = self._buffer, ""
                self.protocol.emit({"ev": "stdout", "text": line, "source": self.source})


class _WorkerSink(ProgressSink):
    def __init__(self, protocol: _ProtocolWriter) -> None:
        self.protocol = protocol

    def on_log(self, message: str, level: str = "info", scope: str | None = None) -> None:
        event: dict[str, Any] = {"ev": "log", "level": level, "text": message}
        if scope is not None:
            event["scope"] = scope
        self.protocol.emit(event)

    def on_progress(self, event: ProgressEvent) -> None:
        self.protocol.emit({"ev": "progress", **asdict(event)})

    def on_item(self, event: ProgressEvent) -> None:
        self.protocol.emit(
            {
                "ev": "item",
                "index": event.item_index,
                "status": event.status,
                "message": event.message,
                "data": event.data,
            }
        )


class _Server:
    def __init__(self, protocol: _ProtocolWriter) -> None:
        self.protocol = protocol
        self.sink = _WorkerSink(protocol)
        self._lock = threading.RLock()
        self._active_thread: threading.Thread | None = None
        self._cancel: CancelToken | None = None
        self._closing = threading.Event()

    def _active(self) -> bool:
        with self._lock:
            return self._active_thread is not None and self._active_thread.is_alive()

    def run(self, raw_job: Any) -> None:
        if self._active():
            self.protocol.emit({"ev": "error", "message": "Worker is already running a job", "traceback": ""})
            return
        try:
            spec = raw_job if isinstance(raw_job, JobSpec) else JobSpec.from_dict(raw_job)
        except Exception as exc:
            self.protocol.emit(
                {
                    "ev": "error",
                    "message": f"Invalid job payload: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
            return
        token = CancelToken()

        def execute() -> None:
            try:
                from .runner import run_job

                result = run_job(spec, self.sink, token)
                self.protocol.emit({"ev": "result", "job_result": result.to_dict()})
            except BaseException as exc:
                self.protocol.emit(
                    {
                        "ev": "error",
                        "message": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                )
            finally:
                with self._lock:
                    self._cancel = None
                    self._active_thread = None

        thread = threading.Thread(target=execute, name="vcap-worker-job", daemon=False)
        with self._lock:
            self._cancel = token
            self._active_thread = thread
        thread.start()

    def chat(self, raw_request: Any) -> None:
        """Run one streamed chat turn without replacing the resident model cache."""

        if self._active():
            self.protocol.emit({"ev": "error", "message": "Worker is already running a job", "traceback": ""})
            return
        try:
            from .chat import ChatRequest

            request = ChatRequest.from_dict(raw_request or {})
        except Exception as exc:
            self.protocol.emit(
                {
                    "ev": "error",
                    "message": f"Invalid chat payload: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
            return
        token = CancelToken()

        def execute() -> None:
            result: Any | None = None
            error: BaseException | None = None
            try:
                from .chat import run_chat

                result = run_chat(request, self.protocol.emit, token)
            except BaseException as exc:
                error = exc
            finally:
                with self._lock:
                    self._cancel = None
                    self._active_thread = None
            if error is not None:
                self.protocol.emit(
                    {
                        "ev": "error",
                        "message": f"{type(error).__name__}: {error}",
                        "traceback": "".join(
                            traceback.format_exception(type(error), error, error.__traceback__)
                        ),
                    }
                )
            elif result is not None:
                self.protocol.emit({"ev": "chat_result", "result": result.to_dict()})

        thread = threading.Thread(target=execute, name="vcap-worker-chat", daemon=False)
        with self._lock:
            self._cancel = token
            self._active_thread = thread
        thread.start()

    def cancel(self) -> None:
        with self._lock:
            token = self._cancel
        if token is not None:
            token.cancel()
            self.protocol.emit({"ev": "log", "level": "warning", "text": "Cancellation requested"})
        else:
            self.protocol.emit({"ev": "log", "level": "info", "text": "No job is running"})

    def unload(self, request: Mapping[str, Any] | None = None) -> None:
        if self._active():
            self.protocol.emit({"ev": "error", "message": "Cannot unload while a job is running", "traceback": ""})
            return
        try:
            from .runner import loaded_variant_key, unload_cached_model

            resident = loaded_variant_key()
            raw_unless = request.get("unless_variant") if request is not None else None
            unless = raw_unless if isinstance(raw_unless, str) else None
            outcome = unload_cached_model(unless_variant=unless)
        except Exception as exc:
            self.protocol.emit(
                {"ev": "error", "message": f"Could not unload model: {exc}", "traceback": traceback.format_exc()}
            )
            return

        has_error = isinstance(outcome, Mapping) and "error" in outcome
        released = (
            outcome.get("variant_key")
            if isinstance(outcome, Mapping) and not has_error
            else None
        )
        skipped = resident is not None and outcome is None
        if has_error:
            self.protocol.emit(
                {
                    "ev": "error",
                    "message": str(outcome.get("error") or "Could not unload model"),
                    "traceback": "",
                }
            )
        self.protocol.emit(
            {
                "ev": "unloaded",
                "resident": resident,
                "released": released,
                "skipped": skipped,
                "report": outcome,
            }
        )
        if released is not None:
            self.protocol.emit({"ev": "log", "level": "info", "text": "Model unloaded"})
        elif skipped:
            self.protocol.emit(
                {
                    "ev": "log",
                    "level": "info",
                    "text": f"Keeping {resident} loaded",
                }
            )
        elif resident is None and not has_error:
            self.protocol.emit(
                {"ev": "log", "level": "info", "text": "No model is loaded"}
            )

    def ping(self) -> None:
        try:
            from .runner import loaded_block_swap_summary, loaded_variant_key

            loaded = loaded_variant_key()
            block_swap = loaded_block_swap_summary()
        except Exception:
            loaded = None
            block_swap = None
        self.protocol.emit(
            {"ev": "pong", "loaded_variant": loaded, "block_swap": block_swap}
        )

    def close(self) -> None:
        self._closing.set()
        self.cancel()
        with self._lock:
            thread = self._active_thread
        if thread is not None:
            thread.join(timeout=5.0)


def _install_redirects(protocol: _ProtocolWriter) -> None:
    stdout = _RedirectStream(protocol, "stdout")
    stderr = _RedirectStream(protocol, "stderr")
    sys.stdout = stdout
    sys.stderr = stderr
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SECourses Video Captioner Pro worker")
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args(argv)


def _poll_stdin(timeout_s: float = 0.1) -> str | None | object:
    """Poll the protocol pipe without a GIL-holding Windows ``readline``."""

    global _STDIN_EOF

    newline = _STDIN_BUFFER.find(b"\n")
    if newline >= 0:
        raw = bytes(_STDIN_BUFFER[: newline + 1])
        del _STDIN_BUFFER[: newline + 1]
        return raw.decode("utf-8", errors="replace")
    if _STDIN_EOF:
        if _STDIN_BUFFER:
            raw = bytes(_STDIN_BUFFER)
            _STDIN_BUFFER.clear()
            return raw.decode("utf-8", errors="replace")
        return None
    descriptor = sys.stdin.fileno()
    if os.name != "nt":
        readable, _, _ = select.select([descriptor], [], [], max(0.0, timeout_s))
        if not readable:
            return _NO_INPUT
        chunk = os.read(descriptor, 65_536)
        if not chunk:
            _STDIN_EOF = True
        else:
            _STDIN_BUFFER.extend(chunk)
        return _poll_stdin(0.0)
    try:
        import ctypes
        import msvcrt
        from ctypes import wintypes

        handle = wintypes.HANDLE(msvcrt.get_osfhandle(descriptor))
        available = wintypes.DWORD()
        ok = ctypes.windll.kernel32.PeekNamedPipe(
            handle,
            None,
            0,
            None,
            ctypes.byref(available),
            None,
        )
        if not ok:
            return None
        if available.value <= 0:
            time.sleep(max(0.0, timeout_s))
            return _NO_INPUT
        chunk = os.read(descriptor, min(65_536, int(available.value)))
        if not chunk:
            _STDIN_EOF = True
        else:
            _STDIN_BUFFER.extend(chunk)
        return _poll_stdin(0.0)
    except Exception:
        time.sleep(max(0.0, timeout_s))
        return _NO_INPUT


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["VCAP_WORKER"] = "1"
    os.environ["VCAP_WORKER_GPU"] = str(args.gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    setup_utf8_stdio()
    real_stdout = sys.stdout
    protocol = _ProtocolWriter(real_stdout)
    _install_redirects(protocol)
    server = _Server(protocol)
    protocol.emit({"ev": "ready"})
    while True:
        raw = _poll_stdin()
        if raw is _NO_INPUT:
            continue
        if raw is None:
            server.close()
            break
        line = raw.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("request must be a JSON object")
        except Exception as exc:
            protocol.emit({"ev": "error", "message": f"Invalid protocol line: {exc}", "traceback": ""})
            continue
        command = str(request.get("cmd", "")).casefold()
        if command == "run_job":
            server.run(request.get("job", request.get("spec")))
        elif command == "chat":
            server.chat(request.get("payload", request.get("request", request)))
        elif command == "cancel":
            server.cancel()
        elif command == "unload":
            server.unload(request)
        elif command == "ping":
            server.ping()
        elif command == "exit":
            server.close()
            break
        else:
            protocol.emit({"ev": "error", "message": f"Unknown command: {command or '<empty>'}", "traceback": ""})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
