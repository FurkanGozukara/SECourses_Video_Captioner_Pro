"""One-request JSON-lines child process for Whisper transcription."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import select
import sys
import tempfile
import threading
import time
import traceback
from contextlib import redirect_stdout, suppress
from pathlib import Path
from typing import Any, Callable, Mapping

from vcap import TEMP_DIR
from vcap.core.paths import normalize_path
from vcap.core.subprocess_runner import CancelledError

_PROTOCOL_STDOUT = sys.stdout
_EMIT_LOCK = threading.Lock()
_NO_INPUT = object()
_STDIN_BUFFER = bytearray()
_STDIN_EOF = False


def _emit(event: str, **payload: Any) -> None:
    record = {"event": event, "ev": event, **payload}
    with _EMIT_LOCK:
        _PROTOCOL_STDOUT.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        _PROTOCOL_STDOUT.flush()


def _poll_stdin(timeout_s: float = 0.1) -> str | None | object:
    """Poll a pipe without Windows TextIO.readline holding the GIL."""

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
        if chunk:
            _STDIN_BUFFER.extend(chunk)
        else:
            _STDIN_EOF = True
        return _poll_stdin(0.0)
    try:
        import ctypes
        import msvcrt
        from ctypes import wintypes

        handle = wintypes.HANDLE(msvcrt.get_osfhandle(descriptor))
        available = wintypes.DWORD()
        ok = ctypes.windll.kernel32.PeekNamedPipe(
            handle, None, 0, None, ctypes.byref(available), None
        )
        if not ok:
            return None
        if available.value <= 0:
            time.sleep(max(0.0, timeout_s))
            return _NO_INPUT
        chunk = os.read(descriptor, min(65_536, int(available.value)))
        if chunk:
            _STDIN_BUFFER.extend(chunk)
        else:
            _STDIN_EOF = True
        return _poll_stdin(0.0)
    except Exception:
        time.sleep(max(0.0, timeout_s))
        return _NO_INPUT


def _cancel_listener(cancelled: threading.Event) -> None:
    while True:
        try:
            line = _poll_stdin()
        except (OSError, ValueError):
            return
        if line is _NO_INPUT:
            continue
        if line is None:
            return
        value = line.strip()
        if value.casefold() == "cancel" or value.strip('"').casefold() == "cancel":
            cancelled.set()
            return


def _load_engine_class():
    target = (
        os.environ.get("VCAP_WHISPER_ENGINE_FACTORY")
        or os.environ.get("VCAP_WHISPER_FAKE_ENGINE")
        or ""
    ).strip()
    if not target:
        from .engine import WhisperEngine

        return WhisperEngine
    module_name, separator, attribute = target.partition(":")
    if not separator:
        module_name, separator, attribute = target.rpartition(".")
    if not module_name or not attribute:
        raise ValueError(
            "VCAP_WHISPER_ENGINE_FACTORY must be in 'module:attribute' form"
        )
    return getattr(importlib.import_module(module_name), attribute)


def _library_state(directories: list[str]) -> tuple[bool, bool]:
    names: list[str] = []
    for directory in directories:
        try:
            names.extend(
                path.name.casefold()
                for path in Path(directory).iterdir()
                if path.is_file()
            )
        except OSError:
            continue
    try:
        import ctranslate2

        package = Path(ctranslate2.__file__).resolve(strict=False).parent
        names.extend(
            path.name.casefold() for path in package.iterdir() if path.is_file()
        )
    except Exception:
        pass
    return any("cublas" in name for name in names), any("cudnn" in name for name in names)


def _probe_runtime(directories: list[str]) -> dict[str, Any]:
    cuda_devices = 0
    try:
        import ctranslate2

        cuda_devices = max(0, int(ctranslate2.get_cuda_device_count()))
    except Exception as exc:
        _emit("log", level="warning", message=f"CTranslate2 runtime probe failed: {exc}")
    cublas, cudnn = _library_state(directories)
    device = "cuda" if cuda_devices > 0 else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    return {
        "device": device,
        "compute_type": compute_type,
        "cuda_devices": cuda_devices,
        "cublas": cublas,
        "cudnn": cudnn,
        "message": (
            f"CTranslate2 runtime probe: {cuda_devices} CUDA device(s), "
            f"cuBLAS={'ready' if cublas else 'not found'}, "
            f"cuDNN={'ready' if cudnn else 'not found'}"
        ),
    }


def _engine_progress(payload: Mapping[str, Any], item_index: int | None) -> None:
    data = dict(payload)
    stage = str(data.pop("stage", "progress"))
    event = stage if stage in {"download", "runtime", "model_loaded"} else "progress"
    if event == "progress" and item_index is not None:
        data["item_index"] = item_index
    _emit(event, **data)


def _requested_output_path(item: Mapping[str, Any], output: Any) -> Path | None:
    from vcap.core.paths import sanitize_filename

    from .writers import FORMAT_EXTENSIONS

    if not output.formats:
        return None
    first = output.formats[0]
    extension = FORMAT_EXTENSIONS[first]
    out_dir = normalize_path(item.get("out_dir") or Path(item["path"]).parent)
    stem = str(item.get("stem") or Path(str(item["path"])).stem)
    base = sanitize_filename(f"{stem}{output.file_suffix}")
    exact = normalize_path(out_dir / f"{base}.{extension}")
    if not output.add_timestamp:
        return exact if exact.is_file() else None
    try:
        matches = sorted(out_dir.glob(f"{base}-*.{extension}"))
    except OSError:
        return None
    return normalize_path(matches[-1]) if matches else None


def _trimmed_media(
    source: Path,
    trim_start_s: float,
    trim_end_s: float | None,
) -> tuple[Path, float, float, Callable[[], None]]:
    """Decode, slice, and stage a 16 kHz mono WAV below TEMP_DIR/whisper."""

    if trim_start_s <= 0 and trim_end_s is None:
        return source, 0.0, 0.0, lambda: None
    import numpy as np
    import soundfile as sf
    from faster_whisper import decode_audio

    sampling_rate = 16_000
    audio = np.ascontiguousarray(
        decode_audio(str(source), sampling_rate=sampling_rate), dtype=np.float32
    ).reshape(-1)
    full_duration = float(audio.shape[-1]) / sampling_rate
    start = max(0.0, min(float(trim_start_s), full_duration))
    end = full_duration if trim_end_s is None else max(0.0, min(float(trim_end_s), full_duration))
    if end <= start:
        raise ValueError(f"Trim end ({end:g}s) must be later than trim start ({start:g}s)")
    sliced = audio[round(start * sampling_rate) : round(end * sampling_rate)]
    temp_root = normalize_path(TEMP_DIR / "whisper")
    temp_root.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix="trim_", suffix=".wav", dir=temp_root)
    os.close(descriptor)
    staged = normalize_path(name)
    sf.write(staged, sliced, sampling_rate, subtype="PCM_16")

    def cleanup() -> None:
        with suppress(OSError):
            staged.unlink()

    return staged, start, full_duration, cleanup


def _offset_result(result: Any, offset_s: float, full_duration_s: float) -> Any:
    if offset_s:
        for segment in result.segments:
            segment.start += offset_s
            segment.end += offset_s
            for word in segment.words:
                word.start += offset_s
                word.end += offset_s
    if full_duration_s > 0:
        result.duration_s = full_duration_s
    return result


def _coerce_result(value: Any):
    from .engine import TranscriptResult

    if isinstance(value, TranscriptResult):
        return value
    if isinstance(value, Mapping):
        return TranscriptResult.from_dict(value)
    raise TypeError(f"Whisper engine returned {type(value).__name__}, expected TranscriptResult")


def _run_request(
    request: Mapping[str, Any],
    directories: list[str],
    cancelled: threading.Event,
) -> int:
    from .params import TranscriptOutputOptions, WhisperParams

    action = str(request.get("action") or "transcribe").strip().casefold()
    started = time.perf_counter()
    if action == "probe_runtime":
        _emit("runtime", **_probe_runtime(directories))
        _emit(
            "done",
            ok=True,
            items_done=0,
            items_failed=0,
            items_skipped=0,
            elapsed_s=time.perf_counter() - started,
        )
        return 0
    if action not in {"ensure_model", "transcribe"}:
        raise ValueError(f"Unknown Whisper worker action: {action}")

    params = WhisperParams.from_dict(request.get("params") or {})
    output = TranscriptOutputOptions.from_dict(request.get("output") or {})
    models_dir_raw = request.get("models_dir")
    models_dir = normalize_path(models_dir_raw) if models_dir_raw else None
    current_item: dict[str, int | None] = {"index": None}

    def log(message: str, level: str = "info") -> None:
        _emit("log", level=level, message=str(message))

    def progress(payload: Mapping[str, Any]) -> None:
        _engine_progress(payload, current_item["index"])

    engine_class = _load_engine_class()
    engine = engine_class(
        params,
        models_dir=models_dir,
        log=log,
        progress=progress,
        cancel_check=cancelled.is_set,
    )
    items_done = 0
    items_failed = 0
    items_skipped = 0
    try:
        if action == "ensure_model":
            path = engine.ensure_model()
            _emit(
                "done",
                ok=True,
                items_done=0,
                items_failed=0,
                items_skipped=0,
                elapsed_s=time.perf_counter() - started,
                path=str(path),
            )
            return 0

        engine.load()
        skip_existing = bool(request.get("skip_existing", False))
        for position, raw_item in enumerate(request.get("items") or []):
            if cancelled.is_set():
                break
            item = dict(raw_item) if isinstance(raw_item, Mapping) else {}
            item_index = int(item.get("index", position))
            current_item["index"] = item_index
            existing = _requested_output_path(item, output) if skip_existing else None
            if existing is not None:
                items_skipped += 1
                _emit(
                    "item_done",
                    item_index=item_index,
                    files=[str(existing)],
                    result=None,
                    skipped=True,
                )
                continue
            cleanup: Callable[[], None] = lambda: None
            try:
                source = normalize_path(item.get("path") or "", must_exist=True)
                trim_start = max(0.0, float(item.get("trim_start_s") or 0.0))
                raw_trim_end = item.get("trim_end_s")
                trim_end = float(raw_trim_end) if raw_trim_end is not None else None
                media, offset, full_duration, cleanup = _trimmed_media(
                    source, trim_start, trim_end
                )

                def on_segment(segment: Any) -> None:
                    _emit(
                        "segment",
                        item_index=item_index,
                        id=int(segment.id),
                        start=float(segment.start) + offset,
                        end=float(segment.end) + offset,
                        text=str(segment.text),
                    )

                result = _coerce_result(engine.transcribe(media, on_segment=on_segment))
                result = _offset_result(result, offset, full_duration)
                out_dir = normalize_path(item.get("out_dir") or source.parent)
                stem = str(item.get("stem") or source.stem)
                from .writers import write_transcript_files

                files = write_transcript_files(
                    result,
                    out_dir,
                    stem,
                    output,
                    normalize=params.normalize_word_timestamps,
                    highlight_words=params.highlight_words,
                )
                items_done += 1
                _emit(
                    "item_done",
                    item_index=item_index,
                    files=[str(path) for path in files],
                    result=result.to_dict(),
                    skipped=False,
                )
            except CancelledError:
                cancelled.set()
                log("Whisper transcription cancellation requested", "warning")
                break
            except Exception as exc:
                items_failed += 1
                _emit(
                    "item_error",
                    item_index=item_index,
                    message=str(exc),
                    traceback=traceback.format_exc(),
                )
            finally:
                cleanup()
        was_cancelled = cancelled.is_set()
        _emit(
            "done",
            ok=not was_cancelled and items_failed == 0,
            items_done=items_done,
            items_failed=items_failed,
            items_skipped=items_skipped,
            elapsed_s=time.perf_counter() - started,
            cancelled=was_cancelled,
        )
        return 2 if was_cancelled else 0 if items_failed == 0 else 1
    finally:
        with suppress(Exception):
            engine.unload()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SECourses Whisper worker")
    parser.add_argument("--request", required=True, help="UTF-8 JSON request path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run one request while reserving stdout exclusively for JSON records."""

    args = _parse_args(argv)
    cancelled = threading.Event()
    listener = threading.Thread(
        target=_cancel_listener,
        args=(cancelled,),
        name="whisper-cancel-listener",
        daemon=True,
    )
    listener.start()
    try:
        request_path = normalize_path(args.request, must_exist=True)
        request = json.loads(request_path.read_text(encoding="utf-8"))
        if not isinstance(request, Mapping):
            raise ValueError("Whisper request must be a JSON object")
        from .cuda_runtime import enable_cuda_runtime_autodiscovery

        directories = enable_cuda_runtime_autodiscovery()
        # Third-party libraries occasionally print banners; keep the protocol clean.
        with redirect_stdout(sys.stderr):
            return _run_request(request, directories, cancelled)
    except CancelledError as exc:
        _emit("log", level="warning", message=str(exc))
        _emit(
            "done",
            ok=False,
            items_done=0,
            items_failed=0,
            items_skipped=0,
            elapsed_s=0.0,
            cancelled=True,
        )
        return 2
    except Exception as exc:
        _emit("error", message=str(exc), traceback=traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
