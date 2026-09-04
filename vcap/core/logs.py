"""Application-wide console, ring-buffer, and per-run logging."""

from __future__ import annotations

import sys
import threading
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import TextIO

from . import console_progress

_STATUS_TEXT_REPLACEMENTS = {
    "Ã¢â‚¬Â¦": "...",
    "â€¦": "...",
    "â†’": "->",
    "âž¡": "->",
    "â€œ": '"',
    "â€\x9d": '"',
    "â€˜": "'",
    "â€™": "'",
    "â€“": "-",
    "â€”": "-",
    "â€‘": "-",
    "â—\x8f": "*",
    "â‰¥": ">=",
    "â‰¤": "<=",
    "âœ…": "[OK]",
    "âŒ": "[ERROR]",
    "âš ï¸": "[WARNING]",
    "â„¹ï¸": "[INFO]",
    "ðŸ›‘": "[STOP]",
    "…": "...",
    "→": "->",
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "–": "-",
    "—": "-",
    "‑": "-",
    "●": "*",
    "⚠️": "[WARNING]",
    "✅": "[OK]",
    "❌": "[ERROR]",
    "ℹ️": "[INFO]",
    "🛑": "[STOP]",
}


def setup_utf8_stdio() -> None:
    """Configure writable process streams for loss-tolerant UTF-8 output."""

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace", write_through=True)
        except (AttributeError, OSError, ValueError):
            continue


def clean_status_text(s: object) -> str:
    """Repair common UTF-8/CP1252 mojibake and console-hostile glyphs."""

    value = "" if s is None else str(s)
    for _ in range(2):
        if not any(token in value for token in ("â", "Ã", "ð", "Â")):
            break
        try:
            repaired = value.encode("latin-1").decode("utf-8")
        except (UnicodeDecodeError, UnicodeEncodeError):
            break
        if repaired == value:
            break
        value = repaired
    for bad, good in _STATUS_TEXT_REPLACEMENTS.items():
        value = value.replace(bad, good)
    return value.replace("\ufffd", "")


class AppLog:
    """Thread-safe singleton logger with a bounded revisioned history."""

    _instance: "AppLog | None" = None
    _instance_lock = threading.Lock()

    def __new__(cls) -> "AppLog":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        with self._instance_lock:
            if self._initialized:
                return
            self._lock = threading.RLock()
            self._lines: deque[tuple[int, str]] = deque(maxlen=5000)
            self._revision = 0
            self._files: dict[Path, TextIO] = {}
            self._file_last_messages: dict[Path, str] = {}
            self._persistence_dir: Path | None = None
            self._daily_path: Path | None = None
            self._daily_file: TextIO | None = None
            self._initialized = True

    def configure_persistence(self, directory: str | Path, keep_files: int = 14) -> Path:
        """Persist every AppLog line to a daily UTF-8 append-only file."""

        target = Path(directory).resolve(strict=False)
        with self._lock:
            if target != self._persistence_dir:
                if self._daily_file is not None:
                    try:
                        self._daily_file.close()
                    except OSError:
                        pass
                self._daily_file = None
                self._daily_path = None
                self._persistence_dir = target
            target.mkdir(parents=True, exist_ok=True)
            files = sorted(
                (path for path in target.glob("app_????-??-??.log") if path.is_file()),
                key=lambda path: path.name,
                reverse=True,
            )
            for stale in files[max(1, int(keep_files)) :]:
                try:
                    stale.unlink()
                except OSError:
                    pass
            return self._ensure_daily_file_locked()

    def _ensure_daily_file_locked(self) -> Path:
        directory = self._persistence_dir
        if directory is None:
            from vcap import LOGS_DIR

            directory = Path(LOGS_DIR).resolve(strict=False)
            self._persistence_dir = directory
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"app_{datetime.now():%Y-%m-%d}.log"
        if target != self._daily_path or self._daily_file is None:
            if self._daily_file is not None:
                try:
                    self._daily_file.close()
                except OSError:
                    pass
            self._daily_file = target.open("a", encoding="utf-8", newline="\n")
            self._daily_path = target
            files = sorted(
                (path for path in directory.glob("app_????-??-??.log") if path.is_file()),
                key=lambda path: path.name,
                reverse=True,
            )
            for stale in files[14:]:
                try:
                    stale.unlink()
                except OSError:
                    pass
        return target

    @property
    def current_log_path(self) -> Path:
        with self._lock:
            return self._ensure_daily_file_locked()

    def write_worker_crash(self, pid: int | None, lines: object) -> Path:
        """Persist a crashed worker's captured merged stderr/stdout tail."""

        with self._lock:
            daily = self._ensure_daily_file_locked()
            target = daily.parent / f"worker_{int(pid or 0)}.log"
            if isinstance(lines, str):
                text = lines
            else:
                try:
                    text = "\n".join(str(value) for value in lines)  # type: ignore[arg-type]
                except TypeError:
                    text = str(lines)
            target.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")
            return target

    @property
    def revision(self) -> int:
        """Return the latest monotonically increasing log revision."""

        with self._lock:
            return self._revision

    def log(
        self,
        msg: object,
        level: str = "info",
        scope: str | None = None,
        console: bool = True,
    ) -> str:
        """Append a timestamped message to every configured sink."""

        del level  # Kept for structured callers; the required line format is stable.
        message = clean_status_text(msg)
        raw_lines = message.splitlines() or [""]
        rendered: list[str] = []
        with self._lock:
            for raw_line in raw_lines:
                timestamp = datetime.now().strftime("%H:%M:%S")
                scope_part = f" [{clean_status_text(scope)}]" if scope else ""
                line = f"[{timestamp}]{scope_part} {raw_line}"
                self._revision += 1
                self._lines.append((self._revision, line))
                rendered.append(line)
                if console:
                    console_progress.log(line)
                try:
                    self._ensure_daily_file_locked()
                    assert self._daily_file is not None
                    self._daily_file.write(line + "\n")
                    self._daily_file.flush()
                except (OSError, ValueError):
                    if self._daily_file is not None:
                        try:
                            self._daily_file.close()
                        except OSError:
                            pass
                    self._daily_file = None
                stale: list[Path] = []
                for path, handle in self._files.items():
                    persisted_message = line.split("] ", 1)[-1]
                    if self._file_last_messages.get(path) == persisted_message:
                        continue
                    try:
                        handle.write(line + "\n")
                        handle.flush()
                        self._file_last_messages[path] = persisted_message
                    except (OSError, ValueError):
                        stale.append(path)
                for path in stale:
                    self._close_attached(path)
        return "\n".join(rendered)

    def warn(self, msg: object, scope: str | None = None, console: bool = True) -> str:
        """Log a warning-level message."""

        return self.log(msg, level="warning", scope=scope, console=console)

    def error(self, msg: object, scope: str | None = None, console: bool = True) -> str:
        """Log an error-level message."""

        return self.log(msg, level="error", scope=scope, console=console)

    def debug(self, msg: object, scope: str | None = None, console: bool = True) -> str:
        """Log a debug-level message."""

        return self.log(msg, level="debug", scope=scope, console=console)

    def exception(self, msg: object, scope: str | None = None, console: bool = True) -> str:
        """Log a message followed by the active exception traceback."""

        trace = traceback.format_exc()
        detail = str(msg) if trace.strip() == "NoneType: None" else f"{msg}\n{trace.rstrip()}"
        return self.log(detail, level="error", scope=scope, console=console)

    def attach_file(self, path: str | Path) -> Path:
        """Attach an append-only UTF-8 run log, idempotently."""

        target = Path(path).resolve(strict=False)
        with self._lock:
            if target in self._files:
                return target
            target.parent.mkdir(parents=True, exist_ok=True)
            last_message = ""
            try:
                lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
                if lines:
                    last_message = lines[-1].split("] ", 1)[-1]
            except OSError:
                pass
            self._files[target] = target.open("a", encoding="utf-8", newline="\n")
            if last_message:
                self._file_last_messages[target] = last_message
        return target

    def _close_attached(self, target: Path) -> None:
        handle = self._files.pop(target, None)
        self._file_last_messages.pop(target, None)
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass

    def detach_file(self, path: str | Path) -> None:
        """Flush, close, and detach a previously attached run log."""

        target = Path(path).resolve(strict=False)
        with self._lock:
            self._close_attached(target)

    def tail(self, n: int) -> str:
        """Return the newest ``n`` lines as one newline-delimited string."""

        count = max(0, int(n))
        with self._lock:
            if count == 0:
                return ""
            return "\n".join(line for _, line in list(self._lines)[-count:])

    def tail_snapshot(self, n: int) -> tuple[list[str], int]:
        """Return a bounded recent window and its revision atomically."""

        count = max(0, int(n))
        with self._lock:
            recent = list(self._lines)[-count:] if count else []
            return [line for _, line in recent], self._revision

    def snapshot(self, since_revision: int = 0) -> tuple[list[str], int]:
        """Return lines newer than a revision and the current revision."""

        threshold = max(0, int(since_revision))
        with self._lock:
            lines = [line for revision, line in self._lines if revision > threshold]
            return lines, self._revision

    def snapshot_for_poll(
        self,
        since_revision: int = 0,
        recovery_limit: int = 300,
    ) -> tuple[list[str], int, bool]:
        """Return a cursor-safe snapshot for a periodically polled consumer.

        A browser can retain a revision from an older logger lifetime or apply
        timer responses out of order.  Such a cursor is ahead of this logger
        and cannot ever match a future line until the revision catches up.
        Return a bounded recent window in that case so the consumer can replace
        its view and reset its cursor in one response.
        """

        threshold = max(0, int(since_revision))
        limit = max(0, int(recovery_limit))
        with self._lock:
            current = self._revision
            if threshold > current:
                recent = list(self._lines)[-limit:] if limit else []
                return [line for _, line in recent], current, True
            lines = [line for revision, line in self._lines if revision > threshold]
            return lines, current, False


_APP_LOG = AppLog()


def get_log() -> AppLog:
    """Return the process-wide :class:`AppLog` singleton."""

    return _APP_LOG


__all__ = ["AppLog", "clean_status_text", "get_log", "setup_utf8_stdio"]
