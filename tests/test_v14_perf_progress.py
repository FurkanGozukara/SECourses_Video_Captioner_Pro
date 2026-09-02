from __future__ import annotations

from io import StringIO
from types import SimpleNamespace

import torch

from vcap.core import console_progress
from vcap.core.progress import ProgressTracker
from vcap.models.base import Callbacks
from vcap.models.omni_common import OmniCaptionerBase
from vcap.pipeline.runner import _Emitter


class _Sink:
    def __init__(self) -> None:
        self.progress = []
        self.items = []

    def on_log(self, *_args, **_kwargs) -> None:
        return None

    def on_progress(self, event) -> None:
        self.progress.append(event)

    def on_item(self, event) -> None:
        self.items.append(event)


class _Tty(StringIO):
    def __init__(self) -> None:
        super().__init__()
        self.flushes = 0

    def isatty(self) -> bool:
        return True

    def flush(self) -> None:
        self.flushes += 1
        super().flush()


def test_transformers_stopping_throttles_but_checks_cancel_each_token(monkeypatch) -> None:
    clock = [0.0]
    monkeypatch.setattr("vcap.core.progress.time.monotonic", lambda: clock[0])
    console_updates: list[str] = []
    monkeypatch.setattr(
        "vcap.models.omni_common.console_progress.show_progress_line",
        lambda message, **_kwargs: console_updates.append(message),
    )

    class CancelAfter:
        calls = 0

        def is_cancelled(self) -> bool:
            self.calls += 1
            return self.calls >= 50

    cancel = CancelAfter()
    callback_updates: list[dict] = []

    def callback(_message, payload) -> None:
        callback_updates.append(dict(payload))

    captioner = object.__new__(OmniCaptionerBase)
    captioner.loaded = SimpleNamespace(device="cpu")
    stopping, _timing = captioner._stopping(10, Callbacks(progress=callback, cancel=cancel), 0.0)
    criterion = stopping[0]
    result_ids: list[int] = []
    for count in range(1, 51):
        clock[0] = count * 0.01
        result = criterion(torch.zeros((1, 10 + count), dtype=torch.long), None)
        result_ids.append(result.data_ptr())

    assert cancel.calls == 50
    assert len(set(result_ids)) == 1
    assert bool(result.item()) is True
    assert callback_updates[0]["new_tokens"] == 1
    assert callback_updates[-1]["new_tokens"] == 50
    assert len(callback_updates) == len(console_updates)
    assert len(callback_updates) <= 6


def test_emitter_forces_first_and_terminal_generate_updates(monkeypatch) -> None:
    clock = [0.0]
    monkeypatch.setattr("vcap.core.progress.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(_Emitter, "_console_status", lambda *_args, **_kwargs: None)
    tracker = ProgressTracker(1, ["clip"])
    tracker.start_item(0, "clip")
    sink = _Sink()
    emitter = _Emitter(sink, tracker)

    for count in range(1, 50):
        clock[0] = count * 0.01
        emitter.progress(
            0,
            0,
            f"Generating: {count}",
            0.5,
            step_index=6,
            data={"new_tokens": count, "tok_per_s": 20.0},
        )
    clock[0] = 0.495
    emitter.progress(
        0,
        0,
        "Generation finished",
        0.9,
        step_index=6,
        data={"new_tokens": 50, "tok_per_s": 20.0, "finish_reason": "length"},
    )

    assert sink.progress[0].data["new_tokens"] == 1
    assert sink.progress[-1].data["new_tokens"] == 50
    assert sink.progress[-1].data["finish_reason"] == "length"
    assert len(sink.progress) == len(sink.items)
    assert len(sink.progress) <= 6

    emitter.progress(0, 0, "Saving", 0.95, step_index=7, data={"phase": "save"})
    emitter.progress(0, 0, "Saved", 1.0, step_index=8, data={"phase": "save"})
    assert [event.message for event in sink.progress[-2:]] == ["Saving", "Saved"]


def test_tty_console_progress_is_keyed_throttled_and_terminal_is_forced(monkeypatch) -> None:
    clock = [0.0]
    monkeypatch.setattr("vcap.core.console_progress.time.monotonic", lambda: clock[0])
    stream = _Tty()
    key = object()

    for count in range(20):
        clock[0] = count * 0.01
        console_progress.show_progress_line(
            f"{count}% working",
            key=key,
            stream=stream,
            min_interval=0.1,
        )
    before_terminal_flushes = stream.flushes
    console_progress.show_progress_line("100% complete", key=key, stream=stream)
    console_progress.finalize_progress_line(key=key, stream=stream)

    assert before_terminal_flushes <= 2
    assert "100% complete" in stream.getvalue()
