"""Whisper model catalogue and resumable Hugging Face downloads."""

from __future__ import annotations

import fnmatch
import os
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from vcap import MODELS_DIR
from vcap.core import console_progress
from vcap.core.logs import get_log
from vcap.core.paths import normalize_path
from vcap.core.subprocess_runner import CancelledError

MODEL_FILE_PATTERNS = [
    "config.json",
    "preprocessor_config.json",
    "model.bin",
    "tokenizer.json",
    "vocabulary.*",
]


@dataclass(frozen=True)
class WhisperModelInfo:
    alias: str
    repo_id: str
    size_bytes: int
    english_only: bool
    note: str


# Sizes were refreshed through HfApi.model_info(files_metadata=True) on 2026-09-04.
WHISPER_MODELS: tuple[WhisperModelInfo, ...] = (
    WhisperModelInfo(
        "large-v1",
        "Systran/faster-whisper-large-v1",
        3_089_578_414,
        False,
        "Best-quality large-v1 default.",
    ),
    WhisperModelInfo(
        "large-v3",
        "Systran/faster-whisper-large-v3",
        3_090_835_702,
        False,
        "Latest full multilingual model.",
    ),
    WhisperModelInfo(
        "large-v3-turbo",
        "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
        1_621_665_983,
        False,
        "Fast multilingual large-v3 variant.",
    ),
    WhisperModelInfo(
        "large-v2",
        "Systran/faster-whisper-large-v2",
        3_089_578_858,
        False,
        "Previous full multilingual model.",
    ),
    WhisperModelInfo(
        "distil-large-v3.5",
        "distil-whisper/distil-large-v3.5-ct2",
        1_516_479_656,
        False,
        "Distilled multilingual large-v3.5.",
    ),
    WhisperModelInfo(
        "distil-large-v3",
        "Systran/faster-distil-whisper-large-v3",
        1_516_479_628,
        False,
        "Distilled multilingual large-v3.",
    ),
    WhisperModelInfo(
        "distil-large-v2",
        "Systran/faster-distil-whisper-large-v2",
        1_516_108_953,
        False,
        "Distilled multilingual large-v2.",
    ),
    WhisperModelInfo(
        "medium",
        "Systran/faster-whisper-medium",
        1_530_571_735,
        False,
        "Balanced multilingual model.",
    ),
    WhisperModelInfo(
        "medium.en",
        "Systran/faster-whisper-medium.en",
        1_530_457_748,
        True,
        "English-only medium model.",
    ),
    WhisperModelInfo(
        "distil-medium.en",
        "Systran/faster-distil-whisper-medium.en",
        792_060_626,
        True,
        "Fast distilled English medium model.",
    ),
    WhisperModelInfo(
        "small",
        "Systran/faster-whisper-small",
        486_212_372,
        False,
        "Compact multilingual model.",
    ),
    WhisperModelInfo(
        "small.en",
        "Systran/faster-whisper-small.en",
        486_098_798,
        True,
        "Compact English-only model.",
    ),
    WhisperModelInfo(
        "distil-small.en",
        "Systran/faster-distil-whisper-small.en",
        335_542_354,
        True,
        "Fast distilled English small model.",
    ),
    WhisperModelInfo(
        "base",
        "Systran/faster-whisper-base",
        147_882_941,
        False,
        "Small multilingual model.",
    ),
    WhisperModelInfo(
        "base.en",
        "Systran/faster-whisper-base.en",
        147_769_510,
        True,
        "Small English-only model.",
    ),
    WhisperModelInfo(
        "tiny",
        "Systran/faster-whisper-tiny",
        78_203_619,
        False,
        "Smallest multilingual model.",
    ),
    WhisperModelInfo(
        "tiny.en",
        "Systran/faster-whisper-tiny.en",
        78_090_594,
        True,
        "Smallest English-only model.",
    ),
)

_MODELS_BY_ALIAS = {item.alias: item for item in WHISPER_MODELS}


def get_model(alias: str) -> WhisperModelInfo | None:
    """Return catalogue metadata for an alias, case-insensitively."""

    return _MODELS_BY_ALIAS.get(str(alias or "").strip().casefold())


def whisper_models_root(models_dir: Path | None = None) -> Path:
    """Return the application-owned Whisper model root."""

    base = MODELS_DIR if models_dir is None else normalize_path(models_dir)
    return normalize_path(Path(base) / "whisper")


def model_dir(alias: str, models_dir: Path | None = None) -> Path:
    """Return a stable visible directory for an alias or repository id."""

    safe_alias = str(alias or "").strip().replace("/", "--").replace("\\", "--")
    if not safe_alias or safe_alias in {".", ".."}:
        raise ValueError("Whisper model alias must not be empty")
    target = normalize_path(whisper_models_root(models_dir) / safe_alias)
    try:
        target.relative_to(whisper_models_root(models_dir))
    except ValueError as exc:
        raise ValueError(f"Invalid Whisper model alias: {alias!r}") from exc
    return target


def _has_partial_files(folder: Path) -> bool:
    try:
        return any(
            path.is_file()
            and (
                path.name.casefold().endswith(".part")
                or path.name.casefold().endswith(".incomplete")
            )
            for path in folder.rglob("*")
        )
    except OSError:
        return True


def _cleanup_completed_resume_files(folder: Path) -> None:
    """Drop stale Hub markers only after the complete model files exist."""

    if not (folder / "config.json").is_file() or not (folder / "model.bin").is_file():
        return
    for path in folder.rglob("*"):
        if path.is_file() and path.name.casefold().endswith((".part", ".incomplete")):
            try:
                path.unlink()
            except OSError:
                continue


def is_model_ready(alias: str, models_dir: Path | None = None) -> bool:
    """Return whether the minimum CTranslate2 files exist without partials."""

    folder = model_dir(alias, models_dir)
    try:
        return (
            folder.is_dir()
            and (folder / "config.json").is_file()
            and (folder / "model.bin").is_file()
            and not _has_partial_files(folder)
        )
    except OSError:
        return False


def local_size_bytes(alias: str, models_dir: Path | None = None) -> int:
    """Return bytes occupied by one model folder, including resume state."""

    folder = model_dir(alias, models_dir)
    if not folder.is_dir():
        return 0
    total = 0
    try:
        for root, _, filenames in os.walk(folder):
            for filename in filenames:
                try:
                    total += max(0, int((Path(root) / filename).stat().st_size))
                except OSError:
                    continue
    except OSError:
        return total
    return total


def format_size(num_bytes: int) -> str:
    """Format a byte count using decimal model-download units."""

    size = max(0, int(num_bytes))
    if size >= 1_000_000_000:
        return f"{size / 1_000_000_000:.2f} GB"
    if size >= 1_000_000:
        return f"{size / 1_000_000:.1f} MB"
    if size >= 1_000:
        return f"{size / 1_000:.1f} KB"
    return f"{size} B"


def model_label(alias: str, models_dir: Path | None = None) -> str:
    """Return the dropdown label with size and local state."""

    info = get_model(alias)
    size = info.size_bytes if info else local_size_bytes(alias, models_dir)
    state = " \u2713 downloaded" if is_model_ready(alias, models_dir) else ""
    return f"{alias} \u2014 {format_size(size)}{state}"


def model_choices(models_dir: Path | None = None) -> list[tuple[str, str]]:
    """Return Gradio ``(label, value)`` model choices in contract order."""

    return [(model_label(item.alias, models_dir), item.alias) for item in WHISPER_MODELS]


def _matches_model_file(filename: str) -> bool:
    return any(fnmatch.fnmatch(filename, pattern) for pattern in MODEL_FILE_PATTERNS)


def refresh_sizes() -> dict[str, int]:
    """Query current Hugging Face metadata for maintaining the hardcoded table."""

    from huggingface_hub import HfApi

    api = HfApi()
    refreshed: dict[str, int] = {}
    for model in WHISPER_MODELS:
        info = api.model_info(model.repo_id, files_metadata=True)
        refreshed[model.alias] = sum(
            int(sibling.size or 0)
            for sibling in info.siblings
            if _matches_model_file(str(sibling.rfilename))
        )
    return refreshed


def _remote_size(repo_id: str) -> int:
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo_id, files_metadata=True)
    return sum(
        int(sibling.size or 0)
        for sibling in info.siblings
        if _matches_model_file(str(sibling.rfilename))
    )


def _check_free_disk(folder: Path, total: int, present: int) -> None:
    remaining = max(0, int(total) - max(0, int(present)))
    required = int(remaining * 1.05 + 0.999999)
    if required <= 0:
        return
    free = int(shutil.disk_usage(folder).free)
    if free >= required:
        return
    shortfall = required - free
    drive = folder.resolve(strict=False).drive or folder.resolve(strict=False).anchor
    raise OSError(
        f"Not enough free disk space for Whisper model: need about "
        f"{format_size(shortfall)} more on {drive or folder}"
    )


class _DownloadProgress:
    def __init__(
        self,
        alias: str,
        total: int,
        initial: int,
        callback: Callable[[dict], None] | None,
        cancel_check: Callable[[], bool] | None,
    ) -> None:
        self.alias = alias
        self.total = max(1, int(total))
        self.base = min(self.total, max(0, int(initial)))
        self.callback = callback
        self.cancel_check = cancel_check
        self._lock = threading.RLock()
        self._bars: dict[int, tuple[int, int, str]] = {}
        self._last_bytes = self.base
        self._last_time = time.monotonic()
        self._started = self._last_time
        self._speed = 0.0
        self._console_key = ("whisper-download", alias)

    def check_cancel(self) -> None:
        if self.cancel_check is not None and self.cancel_check():
            raise CancelledError(f"Whisper model download cancelled: {self.alias}")

    def update_bar(self, key: int, current: int, total: int, filename: str) -> None:
        self.check_cancel()
        with self._lock:
            self._bars[key] = (max(0, int(current)), max(0, int(total)), filename)
            # huggingface_hub may expose both transfer and reconstruction bars for
            # the same bytes, so summing bars double-counts a download.
            active_done = max(
                (
                    min(current_value, total_value or current_value)
                    for current_value, total_value, _ in self._bars.values()
                ),
                default=0,
            )
            completed = min(self.total, self.base + active_done)
            now = time.monotonic()
            elapsed = max(1e-6, now - self._started)
            transferred = max(0, completed - self.base)
            if transferred > 0:
                self._speed = transferred / elapsed
            self._last_bytes = completed
            self._last_time = now
            fraction = min(1.0, completed / self.total)
            clean_file = Path(filename).name if filename else "model files"
            message = f"{self.alias} {clean_file} {fraction * 100:.1f}%"
            payload = {
                "fraction": fraction,
                "bytes": completed,
                "total": self.total,
                "speed_bps": float(self._speed),
                "message": message,
                "file": clean_file,
            }
            if self.callback is not None:
                self.callback(payload)
            if fraction < 1.0:
                speed_text = (
                    f" at {format_size(int(self._speed))}/s"
                    if self._speed > 0
                    else ""
                )
                console_progress.show_progress_line(
                    f"Whisper download {self.alias}: {fraction * 100:5.1f}% "
                    f"({format_size(completed)} / {format_size(self.total)})"
                    f"{speed_text}",
                    key=self._console_key,
                    stream=sys.stderr,
                    min_interval=0.25,
                )

    def finish(self, *, cancelled: bool = False, completed: bool = False) -> None:
        if cancelled:
            final = f"Whisper download {self.alias}: cancelled; resume state preserved"
        elif completed:
            final = f"Whisper download {self.alias}: 100% ({format_size(self.total)})"
        else:
            final = f"Whisper download {self.alias}: failed; resume state preserved"
        console_progress.finalize_progress_line(
            key=self._console_key,
            final_text=final,
            stream=sys.stderr,
        )


def _make_tqdm_class(progress: _DownloadProgress):
    from tqdm.auto import tqdm

    class WhisperDownloadTqdm(tqdm):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs.setdefault("file", sys.stderr)
            kwargs.setdefault("dynamic_ncols", True)
            # The app renders its own rate-limited line. Native concurrent tqdm
            # bars can interleave with worker JSON after stderr is merged.
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)
            progress.update_bar(
                id(self),
                int(getattr(self, "n", kwargs.get("initial", 0)) or 0),
                int(getattr(self, "total", kwargs.get("total", 0)) or 0),
                str(getattr(self, "desc", kwargs.get("desc", "")) or ""),
            )

        def update(self, n: int | float = 1) -> bool | None:
            if self.disable:
                self.n = float(getattr(self, "n", 0) or 0) + n
                result = None
            else:
                result = super().update(n)
            progress.update_bar(
                id(self),
                int(getattr(self, "n", 0) or 0),
                int(getattr(self, "total", 0) or 0),
                str(getattr(self, "desc", "") or ""),
            )
            return result

    return WhisperDownloadTqdm


def download_model(
    alias: str,
    models_dir: Path | None = None,
    *,
    progress_cb: Callable[[dict], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> Path:
    """Download or resume a CTranslate2 Whisper snapshot into its visible folder."""

    from huggingface_hub import snapshot_download

    model = get_model(alias)
    repo_id = model.repo_id if model is not None else str(alias).strip()
    if model is None and "/" not in repo_id:
        raise KeyError(f"Unknown Whisper model alias: {alias}")
    target = model_dir(alias, models_dir)
    target.mkdir(parents=True, exist_ok=True)
    if is_model_ready(alias, models_dir):
        total = model.size_bytes if model is not None else local_size_bytes(alias, models_dir)
        payload = {
            "fraction": 1.0,
            "bytes": total,
            "total": total,
            "speed_bps": 0.0,
            "message": f"{alias} already downloaded",
            "file": "",
        }
        if progress_cb is not None:
            progress_cb(payload)
        return target

    if cancel_check is not None and cancel_check():
        raise CancelledError(f"Whisper model download cancelled: {alias}")
    total = model.size_bytes if model is not None else _remote_size(repo_id)
    if total <= 0:
        raise RuntimeError(f"Hugging Face returned no downloadable model files for {repo_id}")
    present = min(total, local_size_bytes(alias, models_dir))
    _check_free_disk(target, total, present)
    tracker = _DownloadProgress(alias, total, present, progress_cb, cancel_check)
    cancelled = False
    completed = False
    try:
        tracker.update_bar(0, 0, 0, "model files")
        get_log().log(
            f"Downloading Whisper model {alias} from {repo_id} to {target}",
            scope="whisper",
            console=False,
        )
        snapshot_download(
            repo_id=repo_id,
            local_dir=target,
            allow_patterns=MODEL_FILE_PATTERNS,
            token=os.environ.get("HF_TOKEN") or None,
            tqdm_class=_make_tqdm_class(tracker),
        )
        tracker.check_cancel()
        _cleanup_completed_resume_files(target)
        if not is_model_ready(alias, models_dir):
            raise RuntimeError(f"Whisper model download is incomplete at {target}")
        if progress_cb is not None:
            progress_cb(
                {
                    "fraction": 1.0,
                    "bytes": total,
                    "total": total,
                    "speed_bps": float(tracker._speed),
                    "message": f"{alias} download complete",
                    "file": "",
                }
            )
        get_log().log(
            f"Whisper model download finished: {target}",
            scope="whisper",
            console=False,
        )
        completed = True
        return target
    except CancelledError:
        cancelled = True
        try:
            (target / ".cancelled.incomplete").touch()
        except OSError:
            pass
        get_log().warn(
            f"Whisper model download cancelled; resume state preserved at {target}",
            scope="whisper",
            console=False,
        )
        raise
    except Exception:
        try:
            (target / ".download.incomplete").touch()
        except OSError:
            pass
        raise
    finally:
        tracker.finish(cancelled=cancelled, completed=completed)


def delete_model(alias: str, models_dir: Path | None = None) -> int:
    """Delete one Whisper model folder and return the number of bytes removed."""

    root = whisper_models_root(models_dir).resolve(strict=False)
    target = model_dir(alias, models_dir).resolve(strict=False)
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Refusing to delete outside the Whisper model root: {target}") from exc
    size = local_size_bytes(alias, models_dir)
    if target.exists() or target.is_symlink():
        if target.is_symlink():
            target.unlink()
        else:
            shutil.rmtree(target)
    return size


__all__ = [
    "MODEL_FILE_PATTERNS",
    "WHISPER_MODELS",
    "WhisperModelInfo",
    "delete_model",
    "download_model",
    "format_size",
    "get_model",
    "is_model_ready",
    "local_size_bytes",
    "model_choices",
    "model_dir",
    "model_label",
    "refresh_sizes",
    "whisper_models_root",
]
