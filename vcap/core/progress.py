"""UI-agnostic item, step, rate, and sink progress primitives."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Iterable, Protocol, runtime_checkable


def format_duration(seconds: float | int | None) -> str:
    """Format seconds compactly for human status lines."""

    if seconds is None:
        return "unknown"
    total = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def format_eta(seconds: float | int | None) -> str:
    """Format an ETA label, including the unknown state."""

    return f"ETA {format_duration(seconds)}"


def format_bytes(value: float | int | None) -> str:
    """Format a byte count using binary-sized display units."""

    if value is None:
        return "unknown"
    amount = max(0.0, float(value))
    units = ("B", "KB", "MB", "GB", "TB", "PB")
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            return f"{int(amount)} B" if unit == "B" else f"{amount:.1f} {unit}"
        amount /= 1024.0
    return f"{amount:.1f} PB"


class ProgressTracker:
    """Track per-item steps, aggregate counts, elapsed time, and ETA."""

    def __init__(self, total_items: int, item_labels: Iterable[str] | None = None) -> None:
        self.total_items = max(0, int(total_items))
        labels = list(item_labels or [])
        self.item_labels = labels[: self.total_items]
        self._started_at = time.monotonic()
        self._lock = threading.RLock()
        self._current_index: int | None = None
        self._current_label = ""
        self._step_desc = ""
        self._step_fraction = 0.0
        self._step_total: int | None = None
        self._step_index: int | None = None
        self._statuses: dict[int, str] = {}
        self._durations: list[float] = []

    def start_item(self, i: int, label: str | None = None) -> None:
        """Begin a zero-based item and reset its active step state."""

        index = int(i)
        if self.total_items and not 0 <= index < self.total_items:
            raise IndexError(f"Item index {index} is outside 0..{self.total_items - 1}")
        with self._lock:
            self._current_index = index
            if label is not None:
                self._current_label = str(label)
            elif 0 <= index < len(self.item_labels):
                self._current_label = str(self.item_labels[index])
            else:
                self._current_label = f"Item {index + 1}"
            self._step_desc = ""
            self._step_fraction = 0.0
            self._step_total = None
            self._step_index = None

    def set_step(
        self,
        desc: str,
        fraction_in_item: float,
        total_steps: int | None = None,
        step_index: int | None = None,
    ) -> None:
        """Update the active item's step description and fractional completion."""

        with self._lock:
            self._step_desc = str(desc)
            self._step_fraction = min(1.0, max(0.0, float(fraction_in_item)))
            self._step_total = max(1, int(total_steps)) if total_steps is not None else None
            self._step_index = max(0, int(step_index)) if step_index is not None else None

    def finish_item(self, status: str, seconds: float) -> None:
        """Finish the active item as done, skipped, or failed."""

        normalized = str(status).casefold()
        if normalized not in {"done", "skipped", "failed"}:
            raise ValueError("status must be 'done', 'skipped', or 'failed'")
        with self._lock:
            if self._current_index is None:
                raise RuntimeError("start_item() must be called before finish_item()")
            previous = self._statuses.get(self._current_index)
            self._statuses[self._current_index] = normalized
            if previous is None and normalized != "skipped":
                self._durations.append(max(0.0, float(seconds)))
            self._step_fraction = 1.0

    @property
    def elapsed(self) -> float:
        """Seconds elapsed since tracker construction."""

        return max(0.0, time.monotonic() - self._started_at)

    @property
    def processed(self) -> int:
        """Number of successfully completed items."""

        with self._lock:
            return sum(status == "done" for status in self._statuses.values())

    @property
    def skipped(self) -> int:
        """Number of skipped items."""

        with self._lock:
            return sum(status == "skipped" for status in self._statuses.values())

    @property
    def failed(self) -> int:
        """Number of failed items."""

        with self._lock:
            return sum(status == "failed" for status in self._statuses.values())

    @property
    def remaining(self) -> int:
        """Number of items not yet assigned a terminal status."""

        return max(0, self.total_items - self.processed - self.skipped - self.failed)

    @property
    def eta_seconds(self) -> float | None:
        """Estimate remaining time from completed non-skipped item durations."""

        with self._lock:
            if not self._durations or self.remaining <= 0:
                return 0.0 if self.remaining <= 0 else None
            return (sum(self._durations) / len(self._durations)) * self.remaining

    @property
    def overall_fraction(self) -> float:
        """Return terminal items plus the active item's step as a 0..1 fraction."""

        if self.total_items <= 0:
            return 1.0
        with self._lock:
            terminal = len(self._statuses)
            active = 0.0
            if self._current_index is not None and self._current_index not in self._statuses:
                active = self._step_fraction
            return min(1.0, max(0.0, (terminal + active) / self.total_items))

    def status_line(self) -> str:
        """Render a multi-line item, step, ETA, finish-time, and rate summary."""

        with self._lock:
            index = self._current_index
            label = self._current_label
            fraction = self._step_fraction
            step_desc = self._step_desc
            step_index = self._step_index
            step_total = self._step_total
        current_number = (index + 1) if index is not None else 0
        first = (
            f"Item {current_number}/{self.total_items}: {label} ({fraction * 100:.1f}%)"
            if index is not None
            else f"Items 0/{self.total_items}"
        )
        step_prefix = "Step"
        if step_index is not None and step_total is not None:
            display_index = 1 if step_index == 0 else step_index
            step_prefix = f"Step {display_index}/{step_total}"
        second = f"{step_prefix}: {step_desc or 'Waiting'}"
        eta = self.eta_seconds
        finish = "unknown"
        if eta is not None:
            finish = (datetime.now() + timedelta(seconds=eta)).strftime("%H:%M:%S")
        terminal_non_skipped = self.processed + self.failed
        rate = terminal_non_skipped / max(self.elapsed, 1e-9)
        third = (
            f"Total {self.processed}/{self.total_items} processed, {self.skipped} skipped, "
            f"{self.failed} failed, {self.remaining} remaining. {format_eta(eta)}. "
            f"Finish {finish}."
        )
        fourth = f"Elapsed {format_duration(self.elapsed)}. Rate {rate:.2f} items/s."
        return "\n".join((first, second, third, fourth))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable progress snapshot."""

        with self._lock:
            return {
                "total_items": self.total_items,
                "current_index": self._current_index,
                "current_label": self._current_label,
                "step": self._step_desc,
                "fraction_in_item": self._step_fraction,
                "step_index": self._step_index,
                "total_steps": self._step_total,
                "elapsed": self.elapsed,
                "eta_seconds": self.eta_seconds,
                "processed": self.processed,
                "skipped": self.skipped,
                "failed": self.failed,
                "remaining": self.remaining,
                "overall_fraction": self.overall_fraction,
            }


class TokenSpeedMeter:
    """Measure generated token throughput from cumulative token counts."""

    def __init__(self) -> None:
        self._started_at: float | None = None
        self._tokens = 0
        self._lock = threading.Lock()

    def start(self) -> "TokenSpeedMeter":
        """Reset and start the meter."""

        with self._lock:
            self._started_at = time.monotonic()
            self._tokens = 0
        return self

    def update(self, n_tokens: int) -> float:
        """Set the cumulative token count and return tokens per second."""

        with self._lock:
            if self._started_at is None:
                self._started_at = time.monotonic()
            self._tokens = max(0, int(n_tokens))
            elapsed = max(time.monotonic() - self._started_at, 1e-9)
            return self._tokens / elapsed

    @property
    def tok_per_s(self) -> float:
        """Return the current token rate."""

        with self._lock:
            if self._started_at is None:
                return 0.0
            return self._tokens / max(time.monotonic() - self._started_at, 1e-9)


class UiThrottle:
    """Gate high-frequency producers to a minimum UI emission interval."""

    def __init__(self, min_interval: float = 0.2) -> None:
        self.min_interval = max(0.0, float(min_interval))
        self._last_emit: float | None = None
        self._lock = threading.Lock()

    def should_emit(self, force: bool = False) -> bool:
        """Return true when an update should be emitted now."""

        now = time.monotonic()
        with self._lock:
            if force or self._last_emit is None or now - self._last_emit >= self.min_interval:
                self._last_emit = now
                return True
            return False


@dataclass(frozen=True)
class ProgressEvent:
    """Structured progress or item event passed to sinks."""

    message: str
    fraction: float | None = None
    item_index: int | None = None
    total_items: int | None = None
    status: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    kind: str = "progress"


@runtime_checkable
class ProgressSink(Protocol):
    """Consumer contract for logs, progress updates, and item updates."""

    def on_log(self, message: str, level: str = "info", scope: str | None = None) -> None: ...

    def on_progress(self, event: ProgressEvent) -> None: ...

    def on_item(self, event: ProgressEvent) -> None: ...


class MultiSink:
    """Fan progress calls out to multiple independent sinks."""

    def __init__(self, *sinks: ProgressSink | Iterable[ProgressSink]) -> None:
        if len(sinks) == 1 and not hasattr(sinks[0], "on_log"):
            self.sinks = list(sinks[0])  # type: ignore[arg-type]
        else:
            self.sinks = list(sinks)  # type: ignore[list-item]

    def on_log(self, message: str, level: str = "info", scope: str | None = None) -> None:
        """Broadcast a log message."""

        for sink in tuple(self.sinks):
            sink.on_log(message, level=level, scope=scope)

    def on_progress(self, event: ProgressEvent) -> None:
        """Broadcast a progress event."""

        for sink in tuple(self.sinks):
            sink.on_progress(event)

    def on_item(self, event: ProgressEvent) -> None:
        """Broadcast an item event."""

        for sink in tuple(self.sinks):
            sink.on_item(event)


__all__ = [
    "MultiSink",
    "ProgressEvent",
    "ProgressSink",
    "ProgressTracker",
    "TokenSpeedMeter",
    "UiThrottle",
    "format_bytes",
    "format_duration",
    "format_eta",
]
