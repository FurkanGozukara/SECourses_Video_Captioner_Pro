from __future__ import annotations

import time

from vcap.core.progress import (
    MultiSink,
    ProgressEvent,
    ProgressTracker,
    TokenSpeedMeter,
    UiThrottle,
    format_bytes,
    format_duration,
    format_eta,
)


def test_formatters() -> None:
    assert format_duration(45) == "45s"
    assert format_duration(200) == "3m 20s"
    assert format_duration(8100) == "2h 15m"
    assert format_eta(45) == "ETA 45s"
    assert format_bytes(1024) == "1.0 KB"


def test_tracker_counts_fraction_eta_and_status() -> None:
    tracker = ProgressTracker(3, ["one", "two", "three"])
    tracker.start_item(0)
    tracker.set_step("decode", 0.5, total_steps=2, step_index=1)
    assert 0 < tracker.overall_fraction < 1
    tracker.finish_item("done", 2.0)
    tracker.start_item(1)
    tracker.finish_item("skipped", 0.0)
    assert tracker.processed == 1
    assert tracker.skipped == 1
    assert tracker.failed == 0
    assert tracker.remaining == 1
    assert tracker.eta_seconds == 2.0
    status = tracker.status_line()
    assert "processed" in status and "ETA" in status and "Rate" in status
    snapshot = tracker.to_dict()
    assert snapshot["processed"] == 1
    tracker.start_item(2)
    tracker.finish_item("failed", 1.0)
    assert tracker.overall_fraction == 1.0


def test_speed_throttle_and_multi_sink() -> None:
    meter = TokenSpeedMeter().start()
    time.sleep(0.01)
    assert meter.update(10) > 0
    assert meter.tok_per_s > 0
    throttle = UiThrottle(10)
    assert throttle.should_emit()
    assert not throttle.should_emit()
    assert throttle.should_emit(force=True)

    class Sink:
        def __init__(self) -> None:
            self.calls: list[object] = []

        def on_log(self, message: str, level: str = "info", scope: str | None = None) -> None:
            self.calls.append((message, level, scope))

        def on_progress(self, event: ProgressEvent) -> None:
            self.calls.append(event)

        def on_item(self, event: ProgressEvent) -> None:
            self.calls.append(event.status)

    first, second = Sink(), Sink()
    sinks = MultiSink(first, second)
    event = ProgressEvent("working", fraction=0.5, status="done")
    sinks.on_log("hello", scope="test")
    sinks.on_progress(event)
    sinks.on_item(event)
    assert len(first.calls) == len(second.calls) == 3
