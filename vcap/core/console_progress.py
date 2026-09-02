"""Thread-safe console logs with stable, non-interleaving progress rows."""

from __future__ import annotations

import os
import shutil
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Hashable, TextIO

_LOCK = threading.RLock()
_DEFAULT_KEY = object()
_STATES: dict[int, tuple[TextIO, "_ProgressState"]] = {}
_CURSOR_SUPPORT: dict[tuple[str, int], bool] = {}


@dataclass
class _ProgressState:
    lines: OrderedDict[Hashable, str] = field(default_factory=OrderedDict)
    rendered_line_count: int = 0
    last_progress_len: int = 0
    legacy_visible_key: Hashable | None = None
    plain_last_emit: dict[Hashable, float] = field(default_factory=dict)
    plain_last_text: dict[Hashable, str] = field(default_factory=dict)


def _isatty(stream: TextIO) -> bool:
    try:
        return bool(stream.isatty())
    except Exception:
        return False


def _terminal_width() -> int:
    try:
        return shutil.get_terminal_size(fallback=(100, 20)).columns
    except Exception:
        return 100


def _safe_text(text: object) -> str:
    return "".join(character if character >= " " else " " for character in str(text))


def _visible_text(text: object) -> str:
    value = _safe_text(text)
    return value[: max(1, _terminal_width() - 1)]


def _state_for(stream: TextIO) -> _ProgressState:
    key = id(stream)
    existing = _STATES.get(key)
    if existing is None or existing[0] is not stream:
        existing = (stream, _ProgressState())
        _STATES[key] = existing
    return existing[1]


def _enable_windows_vt(stream: TextIO) -> bool:
    try:
        import msvcrt
        from ctypes import wintypes

        import ctypes

        handle = wintypes.HANDLE(msvcrt.get_osfhandle(stream.fileno()))
        mode = wintypes.DWORD()
        kernel32 = ctypes.windll.kernel32
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        if mode.value & 0x0004:
            return True
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except Exception:
        return False


def _supports_cursor(stream: TextIO) -> bool:
    if not _isatty(stream):
        return False
    try:
        file_id = stream.fileno()
    except Exception:
        file_id = id(stream)
    cache_key = (os.name, file_id)
    if cache_key not in _CURSOR_SUPPORT:
        _CURSOR_SUPPORT[cache_key] = (
            _enable_windows_vt(stream)
            if os.name == "nt"
            else os.environ.get("TERM", "").strip().casefold() != "dumb"
        )
    return _CURSOR_SUPPORT[cache_key]


def _clear_block(stream: TextIO, state: _ProgressState) -> None:
    if state.rendered_line_count <= 0:
        return
    if _supports_cursor(stream):
        stream.write("\r\x1b[2K")
        for _ in range(state.rendered_line_count - 1):
            stream.write("\x1b[1A\r\x1b[2K")
    else:
        stream.write("\r" + " " * max(state.last_progress_len, _terminal_width()) + "\r")
    state.rendered_line_count = 0
    state.last_progress_len = 0


def _render_block(stream: TextIO, state: _ProgressState) -> None:
    if not state.lines:
        stream.flush()
        return
    if _supports_cursor(stream):
        lines = [_visible_text(text) for text in state.lines.values()]
        stream.write("\r")
        for index, text in enumerate(lines):
            if index:
                stream.write("\n\r")
            stream.write(text)
        state.rendered_line_count = len(lines)
        state.last_progress_len = len(lines[-1])
    else:
        key = state.legacy_visible_key
        if key not in state.lines:
            key = next(reversed(state.lines))
            state.legacy_visible_key = key
        text = _visible_text(state.lines[key])
        stream.write("\r" + text)
        state.rendered_line_count = 1
        state.last_progress_len = len(text)
    stream.flush()


def _update_existing(stream: TextIO, state: _ProgressState, key: Hashable) -> bool:
    if not _supports_cursor(stream) or state.rendered_line_count != len(state.lines):
        return False
    keys = list(state.lines)
    distance = len(keys) - keys.index(key) - 1
    text = _visible_text(state.lines[key])
    if distance:
        stream.write(f"\x1b[{distance}A")
    stream.write("\r\x1b[2K" + text)
    if distance:
        stream.write(f"\x1b[{distance}B\r")
        last = _visible_text(state.lines[keys[-1]])
        if last:
            stream.write(f"\x1b[{len(last)}C")
        state.last_progress_len = len(last)
    else:
        state.last_progress_len = len(text)
    stream.flush()
    return True


def show_progress_line(
    text: str,
    key: Hashable | None = None,
    stream: TextIO | None = None,
    min_interval: float = 0.1,
) -> None:
    """Create or update one keyed progress row at a bounded refresh rate."""

    output = stream or sys.stdout
    resolved = _DEFAULT_KEY if key is None else key
    with _LOCK:
        state = _state_for(output)
        value = _safe_text(text)
        now = time.monotonic()
        last_emit = state.plain_last_emit.get(resolved)
        terminal = "100%" in value or "100.0%" in value
        if not _isatty(output):
            state.lines[resolved] = value
            last_text = state.plain_last_text.get(resolved)
            if last_text is None or terminal or last_emit is None or now - last_emit >= 1.0:
                print(value, file=output, flush=True)
                state.plain_last_emit[resolved] = now
                state.plain_last_text[resolved] = value
            return

        existed = resolved in state.lines
        state.lines[resolved] = str(text)
        state.legacy_visible_key = resolved
        interval = max(0.0, float(min_interval))
        if existed and not terminal and last_emit is not None and now - last_emit < interval:
            return
        state.plain_last_emit[resolved] = now
        state.plain_last_text[resolved] = value
        if existed and _update_existing(output, state, resolved):
            return
        if (
            not existed
            and _supports_cursor(output)
            and state.rendered_line_count == len(state.lines) - 1
        ):
            value = _visible_text(state.lines[resolved])
            output.write("\n\r" if state.rendered_line_count else "\r")
            output.write(value)
            state.rendered_line_count += 1
            state.last_progress_len = len(value)
            output.flush()
            return
        _clear_block(output, state)
        _render_block(output, state)


def finalize_progress_line(
    key: Hashable | None = None,
    final_text: str | None = None,
    stream: TextIO | None = None,
) -> None:
    """Remove one progress row and optionally commit its final text."""

    output = stream or sys.stdout
    resolved = _DEFAULT_KEY if key is None else key
    with _LOCK:
        state = _state_for(output)
        if not _isatty(output):
            previous = state.plain_last_text.pop(resolved, None)
            state.plain_last_emit.pop(resolved, None)
            state.lines.pop(resolved, None)
            if final_text is not None and final_text != previous:
                print(final_text, file=output, flush=True)
            return
        _clear_block(output, state)
        state.lines.pop(resolved, None)
        if state.legacy_visible_key == resolved:
            state.legacy_visible_key = next(reversed(state.lines), None)
        if final_text is not None:
            output.write(_visible_text(final_text) + "\n")
        _render_block(output, state)


def clear_all(stream: TextIO | None = None) -> None:
    """Clear every live progress row for a stream."""

    output = stream or sys.stdout
    with _LOCK:
        state = _state_for(output)
        if _isatty(output):
            _clear_block(output, state)
        state.lines.clear()
        state.plain_last_emit.clear()
        state.plain_last_text.clear()
        state.legacy_visible_key = None
        output.flush()


def log(message: str, stream: TextIO | None = None) -> None:
    """Print a normal log line above all active progress rows."""

    output = stream or sys.stdout
    with _LOCK:
        state = _state_for(output)
        if state.rendered_line_count and _isatty(output):
            _clear_block(output, state)
            print(message, file=output, flush=False)
            _render_block(output, state)
        else:
            print(message, file=output, flush=True)


__all__ = ["clear_all", "finalize_progress_line", "log", "show_progress_line"]
