"""UTF-8 JSON-lines workers, cancellation, and process-tree termination."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from contextlib import suppress
from typing import Any, Iterable, Iterator, Mapping, Sequence, TextIO

from .gpu import cuda_visible_devices_env


class CancelledError(RuntimeError):
    """Raised when cooperative or subprocess work is cancelled."""


class CancelToken:
    """Thread-safe cancellation event with a temporary confirmation arm."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._armed_until = 0.0

    def arm_confirmation(self, window_s: float = 7) -> None:
        """Arm destructive cancellation for a short confirmation window."""

        with self._lock:
            self._armed_until = time.monotonic() + max(0.0, float(window_s))

    def is_armed(self) -> bool:
        """Return whether the confirmation window remains active."""

        with self._lock:
            if self._armed_until <= time.monotonic():
                self._armed_until = 0.0
                return False
            return True

    def cancel(self) -> None:
        """Set the cancellation event and clear confirmation state."""

        with self._lock:
            self._armed_until = 0.0
        self._event.set()

    def is_cancelled(self) -> bool:
        """Return whether cancellation has been requested."""

        return self._event.is_set()

    def reset(self) -> None:
        """Clear cancellation and confirmation state."""

        self._event.clear()
        with self._lock:
            self._armed_until = 0.0


def build_child_env(
    gpu_index: int | str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Build the UTF-8, unbuffered environment shared by worker processes."""

    env = {str(key): str(value) for key, value in os.environ.items()}
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    if gpu_index is not None:
        env.update(cuda_visible_devices_env(gpu_index))
    if extra:
        for key, value in extra.items():
            if value is None:
                env.pop(str(key), None)
            else:
                env[str(key)] = str(value)
    if os.name == "nt":
        for key in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
            value = env.get(key)
            if value is None:
                continue
            tokens = [
                token.strip()
                for token in value.split(",")
                if token.strip()
                and token.replace(" ", "").casefold()
                != "expandable_segments:true"
            ]
            if tokens:
                env[key] = ",".join(tokens)
            else:
                env.pop(key, None)
        # Without expandable segments the caching allocator keeps large transient prefill
        # blocks cached; with the worker's per-process VRAM cap in place, a garbage-collection
        # threshold makes it release unused cached blocks before the cap is reached.
        configured = ",".join(
            env.get(key, "") for key in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF")
        ).casefold()
        if "garbage_collection_threshold" not in configured:
            existing = env.get("PYTORCH_ALLOC_CONF", "").strip()
            env["PYTORCH_ALLOC_CONF"] = (
                f"{existing},garbage_collection_threshold:0.6" if existing else "garbage_collection_threshold:0.6"
            )
    return env


def _as_text_line(raw: object) -> str:
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw)


def iter_json_lines(stream: Iterable[str] | TextIO) -> Iterator[dict[str, Any]]:
    """Yield protocol events, wrapping plain or malformed lines as stdout events."""

    for raw in stream:
        text = _as_text_line(raw).rstrip("\r\n")
        if not text:
            continue
        try:
            decoded = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            decoded = None
        if isinstance(decoded, dict) and "ev" in decoded:
            yield decoded
        else:
            yield {"ev": "stdout", "text": text}


def _snapshot_processes(pid: int) -> tuple[Any | None, list[Any]]:
    try:
        import psutil

        parent = psutil.Process(int(pid))
        return parent, parent.children(recursive=True)
    except Exception:
        return None, []


def kill_process_tree(pid: int, grace: float = 2.0) -> None:
    """Terminate a process group and its pre-snapshotted descendants."""

    target_pid = int(pid)
    if target_pid <= 0 or target_pid == os.getpid():
        return
    parent, descendants = _snapshot_processes(target_pid)
    target_group: int | None = None
    if os.name != "nt":
        with suppress(Exception):
            candidate_group = os.getpgid(target_pid)
            if candidate_group != os.getpgrp():
                target_group = candidate_group

    if os.name == "nt":
        with suppress(Exception):
            os.kill(target_pid, signal.CTRL_BREAK_EVENT)
    else:
        if target_group is not None:
            with suppress(Exception):
                os.killpg(target_group, signal.SIGTERM)
        else:
            with suppress(Exception):
                os.kill(target_pid, signal.SIGTERM)

    for child in reversed(descendants):
        with suppress(Exception):
            child.terminate()
    if parent is not None:
        with suppress(Exception):
            parent.terminate()

    deadline = time.monotonic() + max(0.0, float(grace))
    while time.monotonic() < deadline:
        alive = False
        if parent is not None:
            with suppress(Exception):
                alive = bool(parent.is_running()) and parent.status() != "zombie"
        else:
            with suppress(Exception):
                os.kill(target_pid, 0)
                alive = True
        if not alive:
            break
        time.sleep(0.05)

    if os.name == "nt":
        with suppress(Exception):
            subprocess.run(
                ["taskkill", "/PID", str(target_pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
            )
    else:
        if target_group is not None:
            with suppress(Exception):
                os.killpg(target_group, signal.SIGKILL)
        else:
            with suppress(Exception):
                os.kill(target_pid, signal.SIGKILL)

    for child in reversed(descendants):
        with suppress(Exception):
            if child.is_running():
                child.kill()
    if parent is not None:
        with suppress(Exception):
            if parent.is_running():
                parent.kill()
        with suppress(Exception):
            parent.wait(timeout=1)


class WorkerProcess:
    """One line-buffered child implementing the application's JSON-lines protocol."""

    def __init__(self) -> None:
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.RLock()
        self.name = "worker"

    def start(
        self,
        cmd: Sequence[str | os.PathLike[str]],
        cwd: str | os.PathLike[str] | None = None,
        env: Mapping[str, str] | None = None,
        name: str = "worker",
    ) -> "WorkerProcess":
        """Launch a child in its own process group and return this wrapper."""

        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise RuntimeError(f"Worker '{self.name}' is already running")
            popen_options: dict[str, Any] = {}
            if os.name == "nt":
                popen_options["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            else:
                popen_options["start_new_session"] = True
            self.name = str(name)
            self._process = subprocess.Popen(
                [os.fspath(part) for part in cmd],
                cwd=os.fspath(cwd) if cwd is not None else None,
                env=dict(env) if env is not None else build_child_env(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                **popen_options,
            )
        return self

    def send(self, obj: Any) -> None:
        """Write and flush one UTF-8 JSON command line."""

        with self._lock:
            process = self._process
            if process is None or process.poll() is not None or process.stdin is None:
                raise RuntimeError(f"Worker '{self.name}' is not running")
            process.stdin.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
            process.stdin.flush()

    def events(self) -> Iterator[dict[str, Any]]:
        """Yield parsed protocol events until the child closes stdout."""

        with self._lock:
            process = self._process
        if process is None or process.stdout is None:
            return
        yield from iter_json_lines(process.stdout)

    def is_alive(self) -> bool:
        """Return whether the child has not exited."""

        with self._lock:
            return self._process is not None and self._process.poll() is None

    @property
    def returncode(self) -> int | None:
        """Return the child's current return code, if exited."""

        with self._lock:
            return self._process.poll() if self._process is not None else None

    def kill_tree(self, grace: float = 2.0) -> None:
        """Stop this worker and all of its descendants."""

        with self._lock:
            process = self._process
        if process is None or process.poll() is not None:
            return
        kill_process_tree(process.pid, grace=grace)
        with suppress(Exception):
            process.wait(timeout=max(1.0, grace + 1.0))

    def wait(self, timeout: float | None = None) -> int:
        """Wait for child exit and return its integer return code."""

        with self._lock:
            process = self._process
        if process is None:
            raise RuntimeError("Worker has not been started")
        return int(process.wait(timeout=timeout))


__all__ = [
    "CancelToken",
    "CancelledError",
    "WorkerProcess",
    "build_child_env",
    "iter_json_lines",
    "kill_process_tree",
]
