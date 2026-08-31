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
            self._initialized = True

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
                stale: list[Path] = []
                for path, handle in self._files.items():
                    try:
                        handle.write(line + "\n")
                        handle.flush()
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
            self._files[target] = target.open("a", encoding="utf-8", newline="\n")
        return target

    def _close_attached(self, target: Path) -> None:
        handle = self._files.pop(target, None)
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

    def snapshot(self, since_revision: int = 0) -> tuple[list[str], int]:
        """Return lines newer than a revision and the current revision."""

        threshold = max(0, int(since_revision))
        with self._lock:
            lines = [line for revision, line in self._lines if revision > threshold]
            return lines, self._revision


_APP_LOG = AppLog()


def get_log() -> AppLog:
    """Return the process-wide :class:`AppLog` singleton."""

    return _APP_LOG


__all__ = ["AppLog", "clean_status_text", "get_log", "setup_utf8_stdio"]
