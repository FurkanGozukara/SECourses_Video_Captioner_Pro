#!/usr/bin/env python3
"""
Resumable model downloader for SECourses Video Captioner Pro.

Model contents are discovered from Hugging Face at runtime. Large files use the
proven SECourses multi-connection range engine, exact .part/.part.json resumes,
published LFS SHA-256 or Git blob verification, and atomic final installation.
"""

from huggingface_hub import HfApi, get_hf_file_metadata, get_token, hf_hub_url, list_repo_files
from huggingface_hub import constants as hf_constants
from huggingface_hub.utils import build_hf_headers
import argparse
import collections
import concurrent.futures
import ctypes
import fnmatch
import hashlib
import json
import logging
import os
import random
import re
import requests
import shutil
import sys
import threading
import time
import urllib.parse
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)
for _noisy in ("huggingface_hub", "httpx", "httpx2", "httpcore", "httpcore2", "urllib3", "requests", "filelock"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

SCRIPT_DIR = Path(__file__).parent.resolve()
if (SCRIPT_DIR / "vcap").is_dir():
    PROJECT_DIR = SCRIPT_DIR
elif (SCRIPT_DIR / "SECourses_Video_Captioner_Pro" / "vcap").is_dir():
    PROJECT_DIR = SCRIPT_DIR / "SECourses_Video_Captioner_Pro"
else:
    PROJECT_DIR = SCRIPT_DIR / "SECourses_Video_Captioner_Pro"
DEFAULT_MODELS_DIR = PROJECT_DIR / "models"
CACHE_DIR = PROJECT_DIR.parent / "download_cache"
DEFAULT_REPO_ID = "MonsterMMORPG/Wan_GGUF"
INDEX_TTL_SECONDS = 24 * 60 * 60
NESTED_MODEL_PREFIX = "Video_Captioner_Pro/"
WHISPER_KEY_PREFIX = "whisper:"

MODEL_CATALOG = {
    "timechat_bf16": {"label": "TimeChat-Captioner-GRPO-7B — BF16", "hf_subfolder": "timechat_bf16", "family": "timechat", "variant": "bf16", "approx_gb": 17.9, "modalities": ["video_audio"]},
    "timechat_int8": {"label": "TimeChat-Captioner-GRPO-7B — INT8 ConvRot", "hf_subfolder": "timechat_int8", "family": "timechat", "variant": "int8_convrot", "approx_gb": 10.3, "modalities": ["video_audio"]},
    "timechat_int4": {"label": "TimeChat-Captioner-GRPO-7B — INT4 ConvRot (W4A8)", "hf_subfolder": "timechat_int4", "family": "timechat", "variant": "int4_convrot_w4a8", "approx_gb": 6.5, "modalities": ["video_audio"]},
    "avocado_bf16": {"label": "AVoCaDO — BF16", "hf_subfolder": "avocado_bf16", "family": "avocado", "variant": "bf16", "approx_gb": 17.9, "modalities": ["video", "video_audio"]},
    "avocado_int8": {"label": "AVoCaDO — INT8 ConvRot", "hf_subfolder": "avocado_int8", "family": "avocado", "variant": "int8_convrot", "approx_gb": 10.3, "modalities": ["video", "video_audio"]},
    "avocado_int4": {"label": "AVoCaDO — INT4 ConvRot (W4A8)", "hf_subfolder": "avocado_int4", "family": "avocado", "variant": "int4_convrot_w4a8", "approx_gb": 6.5, "modalities": ["video", "video_audio"]},
    "qwen3_omni_instruct_bf16": {"label": "Qwen3-Omni-30B-A3B Instruct — BF16", "hf_subfolder": "qwen3_omni_instruct_bf16", "family": "qwen3_omni_instruct", "variant": "bf16", "approx_gb": 63.4, "modalities": ["video", "audio", "image", "text"]},
    "qwen3_omni_instruct_int8": {"label": "Qwen3-Omni-30B-A3B Instruct — INT8 ConvRot", "hf_subfolder": "qwen3_omni_instruct_int8", "family": "qwen3_omni_instruct", "variant": "int8_convrot", "approx_gb": 33.0, "modalities": ["video", "audio", "image", "text"]},
    "qwen3_omni_instruct_int4": {"label": "Qwen3-Omni-30B-A3B Instruct — INT4 ConvRot (W4A8)", "hf_subfolder": "qwen3_omni_instruct_int4", "family": "qwen3_omni_instruct", "variant": "int4_convrot_w4a8", "approx_gb": 17.8, "modalities": ["video", "audio", "image", "text"]},
    "qwen3_omni_thinking_bf16": {"label": "Qwen3-Omni-30B-A3B Thinking — BF16", "hf_subfolder": "qwen3_omni_thinking_bf16", "family": "qwen3_omni_thinking", "variant": "bf16", "approx_gb": 63.4, "modalities": ["video", "audio", "image", "text"]},
    "qwen3_omni_thinking_int8": {"label": "Qwen3-Omni-30B-A3B Thinking — INT8 ConvRot", "hf_subfolder": "qwen3_omni_thinking_int8", "family": "qwen3_omni_thinking", "variant": "int8_convrot", "approx_gb": 33.0, "modalities": ["video", "audio", "image", "text"]},
    "qwen3_omni_thinking_int4": {"label": "Qwen3-Omni-30B-A3B Thinking — INT4 ConvRot (W4A8)", "hf_subfolder": "qwen3_omni_thinking_int4", "family": "qwen3_omni_thinking", "variant": "int4_convrot_w4a8", "approx_gb": 17.8, "modalities": ["video", "audio", "image", "text"]},
    "qwen3_omni_captioner_bf16": {"label": "Qwen3-Omni-30B-A3B Captioner — BF16", "hf_subfolder": "qwen3_omni_captioner_bf16", "family": "qwen3_omni_captioner", "variant": "bf16", "approx_gb": 63.4, "modalities": ["audio"]},
    "qwen3_omni_captioner_int8": {"label": "Qwen3-Omni-30B-A3B Captioner — INT8 ConvRot", "hf_subfolder": "qwen3_omni_captioner_int8", "family": "qwen3_omni_captioner", "variant": "int8_convrot", "approx_gb": 33.0, "modalities": ["audio"]},
    "qwen3_omni_captioner_int4": {"label": "Qwen3-Omni-30B-A3B Captioner — INT4 ConvRot (W4A8)", "hf_subfolder": "qwen3_omni_captioner_int4", "family": "qwen3_omni_captioner", "variant": "int4_convrot_w4a8", "approx_gb": 17.8, "modalities": ["audio"]},
    "qwen3_omni_instruct_gguf_q4": {"label": "Qwen3-Omni-30B-A3B Instruct — GGUF Q4_K_M", "family": "qwen3_omni_instruct", "variant": "gguf_q4", "approx_gb": 19.9, "modalities": ["video", "audio", "image", "text"], "source": "app_gguf"},
    "qwen3_omni_instruct_gguf_q8": {"label": "Qwen3-Omni-30B-A3B Instruct — GGUF Q8_0", "family": "qwen3_omni_instruct", "variant": "gguf_q8", "approx_gb": 33.8, "modalities": ["video", "audio", "image", "text"], "source": "app_gguf"},
    "qwen3_omni_thinking_gguf_q4": {"label": "Qwen3-Omni-30B-A3B Thinking — GGUF Q4_K_M", "family": "qwen3_omni_thinking", "variant": "gguf_q4", "approx_gb": 19.9, "modalities": ["video", "audio", "image", "text"], "source": "app_gguf"},
    "qwen3_omni_thinking_gguf_q8": {"label": "Qwen3-Omni-30B-A3B Thinking — GGUF Q8_0", "family": "qwen3_omni_thinking", "variant": "gguf_q8", "approx_gb": 33.8, "modalities": ["video", "audio", "image", "text"], "source": "app_gguf"},
    "qwen3_omni_captioner_gguf_q4": {"label": "Qwen3-Omni-30B-A3B Captioner — GGUF Q4_K_M", "family": "qwen3_omni_captioner", "variant": "gguf_q4", "approx_gb": 19.9, "modalities": ["audio"], "source": "app_gguf"},
    "qwen3_omni_captioner_gguf_q8": {"label": "Qwen3-Omni-30B-A3B Captioner — GGUF Q8_0", "family": "qwen3_omni_captioner", "variant": "gguf_q8", "approx_gb": 33.8, "modalities": ["audio"], "source": "app_gguf"},
}

GGUF_KEYS = frozenset(key for key, spec in MODEL_CATALOG.items() if spec.get("source") == "app_gguf")


def emit_status(
    key: str,
    state: str,
    message: str,
    *,
    fraction: Optional[float] = None,
    bytes_done: Optional[int] = None,
    bytes_total: Optional[int] = None,
) -> None:
    """Emit one UTF-8 JSON status record for the app/downloader bridge."""

    normalized_state = {"failed": "error"}.get(str(state).casefold(), str(state).casefold())
    allowed = {"downloading", "verifying", "ready", "error", "skipped", "missing"}
    if normalized_state not in allowed:
        normalized_state = "error"
    normalized_fraction = None
    if fraction is not None:
        with suppress(TypeError, ValueError):
            normalized_fraction = min(1.0, max(0.0, float(fraction)))
    payload = {
        "key": str(key),
        "state": normalized_state,
        "fraction": normalized_fraction,
        "bytes_done": int(bytes_done) if bytes_done is not None else None,
        "bytes_total": int(bytes_total) if bytes_total is not None else None,
        "message": " ".join(str(message).split()),
    }
    print("VCAP_STATUS " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")), flush=True)

# =============================================================================
# Transfer engine (retained from the proven SECourses downloader)
# =============================================================================

DOWNLOAD_CONFIG = {
    # HTTP starts conservatively and scales only for a genuinely large queue.
    # Native Xet remains the preferred path for Xet-backed weights.  The HTTP
    # fallback therefore optimizes for portable resumes and non-Xet servers,
    # where opening 96 sockets was slower on both the Linux benchmark host and
    # typical consumer Windows machines.
    "num_connections": 16,
    "http_adaptive_connections": 1,
    "http_max_connections": 32,
    # A file is cut into pieces and every piece is fetched with its own range
    # request. Giving each connection one enormous range instead lets a single
    # shaped or unlucky flow decide the speed of the whole file. Pieces scale
    # with the file, from this size up to the cap while aiming for the target
    # count, so a 15 GB model spends its time moving bytes instead of opening
    # requests while a 100 MB file keeps fine-grained pieces and retries.
    "piece_size": 16 * 1024 * 1024,
    "piece_size_max": 32 * 1024 * 1024,
    "pieces_per_file_target": 512,
    # The model CDN is fast when a flow is fresh but can shape a long-lived
    # flow heavily. Rotate after a couple of ranges and stagger the time limit
    # so the fleet never falls into one synchronized slow wave.
    "pieces_per_connection": 2,
    "max_connection_seconds": 30.0,
    "connection_age_jitter": 0.35,
    # Prefer Hugging Face's native Rust/Xet range stream for large Xet-backed
    # files. It remains optional: the HTTP range engine is the complete fallback
    # on older installs and for highly fragmented resumes.
    "xet_enabled": 1,
    "xet_threshold": 256 * 1024 * 1024,
    "xet_concurrency": 32,
    # Zero means unlimited.  A single Xet stream group is reused for every
    # missing interval, so a fragmented resume stays on Xet instead of falling
    # back to HTTP merely because an interruption left many small gaps.
    "xet_max_resume_gaps": 0,
    "xet_range_flush_size": 32 * 1024 * 1024,
    "xet_high_memory_threshold": 64 * 1024 * 1024 * 1024,
    # After this fair chance, a connection moving less than this fraction of
    # the current per-connection fleet average is dropped and dialed again.
    # The absolute floor further down catches the whole fleet crawling.
    "slow_connection_grace": 8.0,
    "slow_connection_fraction": 0.35,
    "read_block_size": 1024 * 1024,
    # Received bytes are batched to this size before they reach the file, which
    # keeps writes large and, because pieces are handed out in order, close
    # together on disk.
    "write_block_size": 4 * 1024 * 1024,
    # Files below this size use one connection, but several files run at once.
    "parallel_threshold": 16 * 1024 * 1024,
    "small_file_workers": 8,
    # Hashing below this size finishes before a progress line would help.
    "verify_progress_threshold": 64 * 1024 * 1024,
    "metadata_workers": 16,
    "max_retries": 8,
    "retry_delay": 1.0,
    "max_retry_delay": 20.0,
    "connect_timeout": 15.0,
    "read_timeout": 30.0,
    "metadata_timeout": 30.0,
    # Signed CDN links expire; they are refreshed well before that happens.
    "url_refresh_seconds": 900.0,
    "state_flush_interval": 2.0,
    # Speed is reported over this sliding window instead of since the start, so
    # the number on screen is the speed right now.
    "speed_window": 12.0,
    "progress_interval": 0.4,
    "non_tty_progress_interval": 5.0,
    "hash_block_size": 8 * 1024 * 1024,
    # A connection still below this rate after that many seconds is dropped and
    # replaced rather than allowed to hold the file back.
    "min_piece_rate": 128 * 1024,
    "min_piece_seconds": 20.0,
    # Whole-file guards: restart the transfer this many times, and give up only
    # after nothing at all has arrived for this long.
    "file_attempts": 6,
    "stall_timeout": 300.0,
    # Nonzero only in explicit downloader tests; preserves a resumable partial.
    "max_bytes_for_test": 0,
}


# Any setting above can be overridden for a single run, which is how values are
# compared on the machine that actually shows the problem rather than guessed:
#   SECOURSES_DL_MAX_CONNECTION_SECONDS=3 python Models_Downloader.py --ensure timechat_bf16
for _key, _default in list(DOWNLOAD_CONFIG.items()):
    _override = os.environ.get("SECOURSES_DL_" + _key.upper())
    if _override:
        with suppress(ValueError):
            DOWNLOAD_CONFIG[_key] = type(_default)(_override)

RESUME_STATE_VERSION = 2

_CONTENT_RANGE_RE = re.compile(r"^bytes\s+(\d+)-(\d+)/(\d+)$", re.IGNORECASE)
_HEX_40_RE = re.compile(r"^[0-9a-f]{40}$")
_HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")
_LEGACY_PART_RE = re.compile(r"\.part(\d+)$")

_HF_ENDPOINT = str(getattr(hf_constants, "ENDPOINT", "https://huggingface.co"))
_HF_ENDPOINT_HOST = (urllib.parse.urlsplit(_HF_ENDPOINT).hostname or "huggingface.co").lower()


class DownloadError(RuntimeError):
    """A transfer failed in a way the caller should report."""


class RangeNotSupported(DownloadError):
    """The server refused to serve byte ranges."""


class _TooSlow(DownloadError):
    """A connection fell far behind the fleet and is redialed immediately."""


class _Cancelled(RuntimeError):
    """Raised inside workers once the file has been abandoned."""


@dataclass(frozen=True)
class RemoteFile:
    """Everything needed to fetch and verify one repository file."""

    repo_id: str
    filename: str
    size: int
    digest_algorithm: Optional[str]
    digest: Optional[str]
    commit_hash: str

    @property
    def key(self) -> Tuple[str, str]:
        return (self.repo_id, self.filename)


def _digest_from_etag(etag: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Turn a Hugging Face etag or blob id into a verifiable digest.

    LFS weights expose a plain sha256. Small files such as config.json are
    stored in git, so their id is the sha1 of ``blob <size>\\0<content>``.
    """
    if not etag:
        return None, None
    value = etag.replace('"', "").replace("W/", "").strip().lower()
    if value.startswith("sha256:"):
        value = value[7:]
    if _HEX_64_RE.match(value):
        return "sha256", value
    if _HEX_40_RE.match(value):
        return "git-sha1", value
    return None, None


def _reserve_blocks(path: Path, size: int) -> bool:
    """Claim the whole file from the allocator before anything is written.

    A staging file that grows while a dozen connections write into it makes the
    file system extend its block map over and over, and every later write pays
    for the longer map. Reserving the space once keeps that cost flat for the
    whole transfer. ``fallocate`` is called directly rather than through
    ``os.posix_fallocate`` because the libc wrapper quietly falls back to
    writing gigabytes of zeroes on file systems that cannot do this.
    """
    if not sys.platform.startswith("linux") or size <= 0:
        return False
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.fallocate.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_longlong, ctypes.c_longlong
        ]
        fd = os.open(str(path), os.O_WRONLY)
        try:
            return libc.fallocate(fd, 0, 0, size) == 0
        finally:
            os.close(fd)
    except (OSError, AttributeError, ValueError):
        return False


def _enable_sparse_file(path: Path) -> bool:
    """Keep a staging file sparse so out-of-order writes stay cheap.

    Ranges land all over the file, and on NTFS the first write near the end of a
    15 GB file otherwise forces Windows to zero-fill everything before it.
    POSIX file systems already behave this way.
    """
    if os.name != "nt":
        return True
    try:
        import ctypes.wintypes as wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateFileW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        kernel32.CreateFileW.restype = wintypes.HANDLE
        kernel32.DeviceIoControl.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
            wintypes.LPVOID,
        ]
        kernel32.DeviceIoControl.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.CreateFileW(
            str(path),
            0x40000000,  # GENERIC_WRITE
            0x00000003,  # FILE_SHARE_READ | FILE_SHARE_WRITE
            None,
            3,  # OPEN_EXISTING
            0x80,  # FILE_ATTRIBUTE_NORMAL
            None,
        )
        if handle == wintypes.HANDLE(-1).value:
            return False
        try:
            returned = wintypes.DWORD(0)
            return bool(
                kernel32.DeviceIoControl(
                    handle,
                    0x000900C4,  # FSCTL_SET_SPARSE
                    None,
                    0,
                    None,
                    0,
                    ctypes.byref(returned),
                    None,
                )
            )
        finally:
            kernel32.CloseHandle(handle)
    except (OSError, ValueError, AttributeError, ImportError):
        return False


def _allocated_file_bytes(path: Path) -> int:
    """Return physical bytes occupied by ``path`` as portably as possible.

    ``st_size`` is the logical length and therefore reports a newly-created
    50 GB sparse staging file as full even when almost no blocks are allocated.
    POSIX exposes 512-byte block counts; Windows exposes the equivalent through
    GetCompressedFileSizeW.  Falling back to the logical size is conservative
    for non-sparse/unknown file systems.
    """
    try:
        stat = path.stat()
    except OSError:
        return 0

    if os.name == "nt":
        try:
            import ctypes.wintypes as wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.GetCompressedFileSizeW.argtypes = [
                wintypes.LPCWSTR,
                ctypes.POINTER(wintypes.DWORD),
            ]
            kernel32.GetCompressedFileSizeW.restype = wintypes.DWORD
            high = wintypes.DWORD(0)
            ctypes.set_last_error(0)
            low = int(kernel32.GetCompressedFileSizeW(str(path), ctypes.byref(high)))
            if low == 0xFFFFFFFF and ctypes.get_last_error() != 0:
                raise OSError(ctypes.get_last_error(), "GetCompressedFileSizeW failed")
            return min(stat.st_size, (int(high.value) << 32) | low)
        except (OSError, ValueError, AttributeError, ImportError):
            return stat.st_size

    blocks = getattr(stat, "st_blocks", None)
    if isinstance(blocks, int) and blocks >= 0:
        return min(stat.st_size, blocks * 512)
    return stat.st_size


def _total_memory_bytes() -> int:
    """Best-effort physical-memory size without making psutil mandatory."""
    with suppress(Exception):
        import psutil

        return int(psutil.virtual_memory().total)
    if os.name == "posix":
        with suppress(OSError, ValueError, AttributeError):
            return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    if os.name == "nt":
        with suppress(Exception):
            import ctypes.wintypes as wintypes

            class _MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", wintypes.DWORD),
                    ("memory_load", wintypes.DWORD),
                    ("total_physical", ctypes.c_ulonglong),
                    ("available_physical", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong),
                    ("available_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended_virtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.total_physical)
    return 0


def _fsync_parent(path: Path) -> None:
    """Durably record a rename on file systems which support directory fsync."""
    if os.name != "posix":
        return
    descriptor = None
    try:
        descriptor = os.open(str(path.parent), os.O_RDONLY)
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        if descriptor is not None:
            os.close(descriptor)


class _SpeedMeter:
    """Throughput over a sliding window rather than since the download began."""

    def __init__(self, window: float):
        self._window = max(1.0, float(window))
        self._lock = threading.Lock()
        self._total = 0
        self._samples = collections.deque()

    def add(self, count: int) -> None:
        with self._lock:
            self._total += count

    @property
    def total(self) -> int:
        with self._lock:
            return self._total

    def rate(self) -> float:
        now = time.monotonic()
        with self._lock:
            total = self._total
            self._samples.append((now, total))
            while len(self._samples) > 2 and now - self._samples[0][0] > self._window:
                self._samples.popleft()
            oldest_at, oldest_total = self._samples[0]
        span = now - oldest_at
        if span <= 0.05:
            return 0.0
        return max(0.0, (total - oldest_total) / span)


class _Link:
    """One worker's HTTP session, replaced when it misbehaves or grows old.

    A connection that has just been slow or failed rarely improves, so it is
    thrown away and dialed again. Healthy sockets are reused briefly to avoid
    needless handshakes, then rotated before CDN shaping can drag down a whole
    synchronized fleet.
    """

    __slots__ = (
        "_factory",
        "_limit",
        "_max_age",
        "_age_jitter",
        "_session",
        "_used",
        "_expires",
    )

    def __init__(self, factory, limit: int, max_age: float, age_jitter: float = 0.0):
        self._factory = factory
        self._limit = max(1, int(limit))
        self._max_age = max(1.0, float(max_age))
        self._age_jitter = min(0.9, max(0.0, float(age_jitter)))
        self._session = None
        self._used = 0
        self._expires = 0.0

    def session(self):
        # Both limits here are backstops. A genuinely slow connection is
        # noticed against the rest of the fleet inside _fetch_piece and
        # dropped long before either arm triggers.
        now = time.monotonic()
        if (
            self._session is None
            or self._used >= self._limit
            or now >= self._expires
        ):
            self.drop()
            self._session = self._factory()
            self._used = 0
            spread = self._max_age * self._age_jitter
            self._expires = now + random.uniform(
                self._max_age - spread,
                self._max_age + spread,
            )
        self._used += 1
        return self._session

    def drop(self) -> None:
        if self._session is not None:
            with suppress(Exception):
                self._session.close()
            self._session = None
            self._expires = 0.0


class _Progress:
    """When bytes last reached the file, and how many connections are moving."""

    __slots__ = ("_lock", "last", "active")

    def __init__(self):
        self._lock = threading.Lock()
        self.last = time.monotonic()
        self.active = 0

    def touch(self) -> None:
        self.last = time.monotonic()

    def enter(self) -> None:
        with self._lock:
            self.active += 1

    def leave(self) -> None:
        with self._lock:
            self.active -= 1


class _Ranges:
    """The byte ranges of a staging file that are already on disk.

    This is the whole resume story: half-open ``[start, end)`` intervals, merged
    as they arrive and saved next to the staging file. It does not care which
    connection wrote what or how the file was cut up, so a download started by
    one version resumes under another, and an interrupted run continues from the
    exact byte it reached rather than from a piece or segment boundary.
    """

    __slots__ = ("_items", "_lock", "_total")

    def __init__(self, items: Optional[Iterable[Sequence[int]]] = None):
        self._lock = threading.Lock()
        self._items: List[List[int]] = []
        self._total = 0
        for start, end in items or []:
            self.add(int(start), int(end))

    def add(self, start: int, end: int) -> None:
        self.add_many(((start, end),))

    def add_many(self, items: Iterable[Sequence[int]]) -> None:
        """Merge several completed ranges in one pass.

        Xet can return thousands of sub-megabyte chunks out of order. Batching
        their interval updates avoids sorting the complete map once per chunk.
        """
        additions = [
            [int(start), int(end)]
            for start, end in items
            if int(end) > int(start)
        ]
        if not additions:
            return
        with self._lock:
            merged: List[List[int]] = []
            for item in sorted(self._items + additions):
                if merged and item[0] <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], item[1])
                else:
                    merged.append(item)
            self._items = merged
            self._total = sum(stop - begin for begin, stop in merged)

    def prefix(self, start: int, limit: int) -> int:
        """How many bytes from ``start`` are already present, up to ``limit``."""
        with self._lock:
            for begin, stop in self._items:
                if begin > start:
                    break
                if start < stop:
                    return min(stop - start, limit)
        return 0

    def missing(self, start: int, stop: int) -> List[Tuple[int, int]]:
        """Return the exact gaps in ``[start, stop)`` that are not on disk."""
        if stop <= start:
            return []
        with self._lock:
            cursor = start
            gaps: List[Tuple[int, int]] = []
            for begin, end in self._items:
                if end <= cursor:
                    continue
                if begin >= stop:
                    break
                if begin > cursor:
                    gaps.append((cursor, min(begin, stop)))
                cursor = max(cursor, min(end, stop))
                if cursor >= stop:
                    break
            if cursor < stop:
                gaps.append((cursor, stop))
            return gaps

    @property
    def total(self) -> int:
        with self._lock:
            return self._total

    def snapshot(self) -> List[List[int]]:
        with self._lock:
            return [list(item) for item in self._items]


_REPO_INDEX_CACHE: Dict[str, Tuple[str, Dict[str, RemoteFile]]] = {}
_REPO_INDEX_LOCK = threading.Lock()


def repo_index(repo_id: str) -> Tuple[str, Dict[str, RemoteFile]]:
    """List a repository once, with the size and digest of every file.

    One ``repo_info`` call replaces a per-file metadata request, which for a
    repository with hundreds of entries is the difference between one second and
    several minutes before the first byte is transferred.
    """
    with _REPO_INDEX_LOCK:
        cached = _REPO_INDEX_CACHE.get(repo_id)
    if cached is not None:
        return cached

    info = HfApi().repo_info(repo_id, files_metadata=True)
    commit_hash = str(getattr(info, "sha", "") or "main")
    files: Dict[str, RemoteFile] = {}
    for sibling in getattr(info, "siblings", None) or []:
        name = getattr(sibling, "rfilename", None)
        size = getattr(sibling, "size", None)
        if not name or not isinstance(size, int):
            continue
        lfs = getattr(sibling, "lfs", None)
        etag = getattr(lfs, "sha256", None) if lfs is not None else None
        if not etag:
            etag = getattr(sibling, "blob_id", None)
        algorithm, digest = _digest_from_etag(etag)
        files[name] = RemoteFile(repo_id, name, size, algorithm, digest, commit_hash)

    result = (commit_hash, files)
    with _REPO_INDEX_LOCK:
        _REPO_INDEX_CACHE[repo_id] = result
    return result


def list_repo_file_names(repo_id: str) -> List[str]:
    """Repository file list, reusing the cached index when one exists."""
    try:
        return sorted(repo_index(repo_id)[1])
    except Exception:
        return sorted(list_repo_files(repo_id))


class RobustDownloader:
    """Resumable, verified downloader built on parallel byte ranges.

    A large file is cut into pieces that many connections claim in file
    order, each piece fetched with its own range request. Nothing
    depends on a single connection keeping its speed, so one shaped or
    unlucky flow costs a piece rather than the rest of the download, and the
    connections keep writing to one moving point in the file instead of a dozen
    places gigabytes apart.

    Bytes land in a sparse ``.part`` file at absolute offsets and the file is
    installed only once its digest matches the repository. Every byte that
    reaches disk is recorded in a map saved alongside it, so an interrupted run
    resumes at exactly the byte it reached.
    """

    def __init__(self, config: Dict, skip_verify: bool = False):
        self.config = dict(DOWNLOAD_CONFIG)
        self.config.update(config or {})
        self.skip_verify = skip_verify

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.verified_cache_file = CACHE_DIR / "verified_files_cache.json"
        self.verified_cache = self._load_json_cache(self.verified_cache_file)
        self._cache_lock = threading.Lock()

        self._progress_lock = threading.Lock()
        self._active_progress = False
        self._last_progress_len = 0
        try:
            self._interactive = bool(sys.stdout.isatty())
        except (AttributeError, OSError):
            self._interactive = False
        self._last_throttled_at = 0.0
        self._status_key: Optional[str] = None
        self._last_status_at = 0.0
        self._status_lock = threading.Lock()

        self._remote_cache: Dict[Tuple[str, str], RemoteFile] = {}
        self._xet_cache: Dict[Tuple[str, str], Tuple[str, str]] = {}
        self._remote_lock = threading.Lock()
        self._url_cache: Dict[Tuple[str, str], Tuple[str, float]] = {}
        self._url_refreshing: set = set()
        self._url_lock = threading.Lock()

        try:
            self._hf_token = _hf_token()
        except Exception:
            self._hf_token = None
        self._hf_headers = dict(
            build_hf_headers(token=self._hf_token if self._hf_token else False)
        )

    # ------------------------------- plumbing --------------------------------

    def _create_session(self, pool: int = 4) -> requests.Session:
        """A session with its own connection pool.

        Transfer workers each hold one and replace it periodically, so their
        sockets are never reused for longer than a handful of pieces. Retries
        are handled per piece, so the adapter itself does not retry.
        """
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=pool,
            pool_maxsize=pool,
            max_retries=0,
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def _load_json_cache(self, filepath: Path) -> Dict:
        if filepath.exists():
            try:
                with open(filepath, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                return data if isinstance(data, dict) else {}
            except (IOError, OSError, json.JSONDecodeError) as exc:
                logger.warning(f"Could not load cache {filepath.name}: {exc}")
        return {}

    def _save_json_cache(self, filepath: Path, data: Dict) -> None:
        try:
            temp_file = filepath.with_suffix(".tmp")
            with open(temp_file, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2)
            temp_file.replace(filepath)
        except (IOError, OSError) as exc:
            logger.warning(f"Could not save cache {filepath.name}: {exc}")

    def _get_terminal_width(self) -> int:
        try:
            return shutil.get_terminal_size(fallback=(100, 20)).columns
        except Exception:
            return 100

    def set_status_key(self, key: Optional[str]) -> None:
        """Select the catalog key used by throttled transfer status records."""

        with self._status_lock:
            self._status_key = key
            self._last_status_at = 0.0

    def _emit_progress_status(
        self,
        text: str,
        state: str,
        bytes_done: Optional[int],
        bytes_total: Optional[int],
    ) -> None:
        if self._status_key is None or bytes_done is None or bytes_total is None:
            return
        now = time.monotonic()
        with self._status_lock:
            if now - self._last_status_at < 2.0 and int(bytes_done) < int(bytes_total):
                return
            self._last_status_at = now
            key = self._status_key
        emit_status(
            key,
            state,
            text,
            fraction=int(bytes_done) / max(1, int(bytes_total)),
            bytes_done=int(bytes_done),
            bytes_total=int(bytes_total),
        )

    def show_progress_line(
        self,
        text: str,
        *,
        state: str = "downloading",
        bytes_done: Optional[int] = None,
        bytes_total: Optional[int] = None,
    ) -> None:
        self._emit_progress_status(text, state, bytes_done, bytes_total)
        with self._progress_lock:
            if not self._interactive:
                # Piped output (the app runs this downloader as a subprocess)
                # gets whole lines at a slow cadence instead of carriage returns.
                now = time.monotonic()
                interval = max(0.1, float(self.config["non_tty_progress_interval"]))
                if now - self._last_throttled_at >= interval:
                    print(text, flush=True)
                    self._last_throttled_at = now
                return
            text = text[: max(1, self._get_terminal_width() - 1)]
            sys.stdout.write("\r" + text)
            if self._last_progress_len > len(text):
                sys.stdout.write(" " * (self._last_progress_len - len(text)))
                sys.stdout.write("\r" + text)
            sys.stdout.flush()
            self._last_progress_len = len(text)
            self._active_progress = True

    def finalize_progress_line(self, final_text: Optional[str] = None) -> None:
        with self._progress_lock:
            if not self._interactive:
                if final_text is not None:
                    print(final_text, flush=True)
                self._last_throttled_at = 0.0
                self._last_progress_len = 0
                self._active_progress = False
                return
            if final_text is not None:
                final_text = final_text[: max(1, self._get_terminal_width() - 1)]
                sys.stdout.write("\r" + final_text)
                if self._last_progress_len > len(final_text):
                    sys.stdout.write(" " * (self._last_progress_len - len(final_text)))
                sys.stdout.write("\n")
                sys.stdout.flush()
            elif self._active_progress:
                sys.stdout.write("\n")
                sys.stdout.flush()
            self._last_progress_len = 0
            self._active_progress = False

    def log(self, msg: str) -> None:
        with self._progress_lock:
            if self._active_progress:
                sys.stdout.write("\r" + " " * max(self._last_progress_len, 1) + "\r")
                self._last_progress_len = 0
                self._active_progress = False
            print(msg, flush=True)

    @staticmethod
    def format_bytes(bytes_val: float) -> str:
        bytes_val = max(0.0, float(bytes_val))
        for unit in ["B", "KB", "MB", "GB"]:
            if bytes_val < 1024.0:
                return f"{bytes_val:.1f} {unit}"
            bytes_val /= 1024.0
        return f"{bytes_val:.1f} TB"

    @staticmethod
    def format_time(seconds: float) -> str:
        if seconds < 0 or seconds != seconds:
            return "0s"
        seconds = int(seconds)
        if seconds < 60:
            return f"{seconds}s"
        if seconds < 3600:
            mins, secs = divmod(seconds, 60)
            return f"{mins}m {secs}s" if secs else f"{mins}m"
        hours, rest = divmod(seconds, 3600)
        mins = rest // 60
        return f"{hours}h {mins}m" if mins else f"{hours}h"

    # ------------------------------- metadata --------------------------------

    def _get_file_metadata(self, url: str):
        """Resolve HF metadata, retrying rate limits and rejecting a bad token."""
        token = self._hf_token if self._hf_token else False
        for attempt in range(5):
            try:
                return get_hf_file_metadata(
                    url,
                    token=token,
                    timeout=float(self.config["metadata_timeout"]),
                )
            except Exception as exc:
                status = self._status_code(exc)
                if status == 401 and token is not False:
                    self.log("[WARNING] Hugging Face rejected the saved token; retrying anonymously")
                    token = False
                    self._hf_token = None
                    self._hf_headers = dict(build_hf_headers(token=False))
                    continue
                if status == 429 and attempt < 4:
                    response = getattr(exc, "response", None)
                    retry_after = getattr(response, "headers", {}).get("Retry-After") if response is not None else None
                    try:
                        delay = min(60.0, max(1.0, float(retry_after)))
                    except (TypeError, ValueError):
                        delay = min(30.0, 2.0 ** attempt)
                    self.log(f"[RETRY] Hugging Face rate limit; retrying metadata in {delay:.0f}s")
                    time.sleep(delay)
                    continue
                raise

    def get_remote_file(self, repo_id: str, filename: str) -> RemoteFile:
        """Size, digest and pinned commit for one repository file."""
        key = (repo_id, filename)
        with self._remote_lock:
            cached = self._remote_cache.get(key)
        if cached is not None:
            return cached

        url = hf_hub_url(repo_id, filename)
        metadata = self._get_file_metadata(url)
        size = getattr(metadata, "size", None)
        if not isinstance(size, int):
            raise DownloadError(f"{filename}: the repository did not report a file size")
        algorithm, digest = _digest_from_etag(getattr(metadata, "etag", None))
        remote = RemoteFile(
            repo_id,
            filename,
            size,
            algorithm,
            digest,
            str(getattr(metadata, "commit_hash", "") or "main"),
        )
        # The same response already carries the resolved CDN link, so the first
        # range request does not need another round trip to resolve it.
        location = str(getattr(metadata, "location", "") or url)
        self._remember_remote(
            remote,
            location,
            getattr(metadata, "xet_file_data", None),
        )
        return remote

    def _remember_remote(
        self,
        remote: RemoteFile,
        location: Optional[str] = None,
        xet_file_data=None,
    ) -> None:
        with self._remote_lock:
            self._remote_cache[remote.key] = remote
            file_hash = str(getattr(xet_file_data, "file_hash", "") or "")
            refresh_route = str(getattr(xet_file_data, "refresh_route", "") or "")
            if file_hash and refresh_route:
                self._xet_cache[remote.key] = (file_hash, refresh_route)
        if location:
            with self._url_lock:
                self._url_cache[remote.key] = (
                    location,
                    time.monotonic() + float(self.config["url_refresh_seconds"]),
                )

    def prefetch_metadata(self, requests_list: Sequence[Tuple[str, str]]) -> Dict[Tuple[str, str], RemoteFile]:
        """Resolve many files at once instead of one blocking call per file."""
        wanted = [key for key in dict.fromkeys(requests_list) if key not in self._remote_cache]
        if not wanted:
            return dict(self._remote_cache)

        by_repo: Dict[str, List[str]] = collections.defaultdict(list)
        for repo_id, filename in wanted:
            by_repo[repo_id].append(filename)

        # Repositories contributing several files are cheaper to list in one go.
        leftovers: List[Tuple[str, str]] = []
        for repo_id, filenames in by_repo.items():
            if len(filenames) < 3:
                leftovers.extend((repo_id, name) for name in filenames)
                continue
            try:
                _, index = repo_index(repo_id)
            except Exception as exc:
                logger.debug(f"Could not list {repo_id}: {exc}")
                leftovers.extend((repo_id, name) for name in filenames)
                continue
            for name in filenames:
                remote = index.get(name)
                if remote is None:
                    leftovers.append((repo_id, name))
                else:
                    self._remember_remote(remote)

        if leftovers:
            workers = max(1, min(int(self.config["metadata_workers"]), len(leftovers)))
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(self.get_remote_file, repo_id, name): (repo_id, name)
                    for repo_id, name in leftovers
                }
                for future in concurrent.futures.as_completed(futures):
                    with suppress(Exception):
                        future.result()
        return dict(self._remote_cache)

    # ------------------------------ verification -----------------------------

    def _verified_key(self, remote: RemoteFile) -> str:
        return f"{remote.repo_id}/{remote.filename}"

    def is_file_verified(self, remote: RemoteFile, filepath: Path) -> bool:
        """True when this exact file was verified earlier and never touched."""
        if not remote.digest:
            return False
        with self._cache_lock:
            record = self.verified_cache.get(self._verified_key(remote))
        if not isinstance(record, dict):
            return False
        try:
            stat = filepath.stat()
        except OSError:
            return False
        return (
            record.get("sha256") == remote.digest
            and record.get("size") == stat.st_size
            and abs(float(record.get("mtime", 0.0)) - stat.st_mtime) < 1.0
        )

    def mark_file_verified(self, remote: RemoteFile, filepath: Path) -> None:
        if not remote.digest:
            return
        try:
            stat = filepath.stat()
        except OSError:
            return
        with self._cache_lock:
            self.verified_cache[self._verified_key(remote)] = {
                "sha256": remote.digest,
                "algorithm": remote.digest_algorithm,
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "verified_at": time.time(),
            }
            self._save_json_cache(self.verified_cache_file, self.verified_cache)

    def compute_digest(
        self,
        filepath: Path,
        algorithm: str,
        size: int,
        display: str,
        show_progress: bool = True,
    ) -> str:
        """Hash a file, showing progress because 15 GB is not instant."""
        if algorithm == "git-sha1":
            digest = hashlib.sha1()
            digest.update(b"blob %d\0" % size)
        else:
            digest = hashlib.new(algorithm)
        block = int(self.config["hash_block_size"])
        # Small files hash faster than a progress line is worth reading.
        show_progress = show_progress and size >= int(self.config["verify_progress_threshold"])
        read = 0
        started = time.monotonic()
        last_update = 0.0
        with open(filepath, "rb") as handle:
            while True:
                data = handle.read(block)
                if not data:
                    break
                digest.update(data)
                read += len(data)
                if not show_progress:
                    continue
                now = time.monotonic()
                if now - last_update >= 0.25:
                    percent = (read * 100.0 / size) if size else 100.0
                    speed = read / max(0.001, now - started)
                    self.show_progress_line(
                        f"[VERIFYING] {display}: {percent:.1f}% "
                        f"({self.format_bytes(read)}/{self.format_bytes(size)}) "
                        f"{self.format_bytes(speed)}/s",
                        state="verifying",
                        bytes_done=read,
                        bytes_total=size,
                    )
                    last_update = now
        return digest.hexdigest().lower()

    # ------------------------------ URL handling -----------------------------

    def _headers_for(self, url: str) -> Dict[str, str]:
        host = (urllib.parse.urlsplit(url).hostname or "").lower()
        if host == _HF_ENDPOINT_HOST:
            return dict(self._hf_headers)
        # Storage links are pre-signed. Forwarding the Hugging Face token to a
        # CDN host would leak it without granting any extra access.
        return {"user-agent": self._hf_headers.get("user-agent", "secourses-model-downloader")}

    def _download_target(self, remote: RemoteFile) -> Tuple[str, Dict[str, str]]:
        """The current CDN link for a file, refreshed before it can expire.

        Every range request asks for this, so the refresh happens outside the
        lock and only one thread does it. The others keep using the link they
        have, which still works: the refresh window is deliberately far shorter
        than the lifetime of a signed link.
        """
        with self._url_lock:
            cached = self._url_cache.get(remote.key)
            fresh = cached is not None and time.monotonic() < cached[1]
            if fresh or (cached is not None and remote.key in self._url_refreshing):
                return cached[0], self._headers_for(cached[0])
            self._url_refreshing.add(remote.key)

        url = hf_hub_url(remote.repo_id, remote.filename, revision=remote.commit_hash or "main")
        try:
            metadata = get_hf_file_metadata(url, timeout=float(self.config["metadata_timeout"]))
            location = str(getattr(metadata, "location", "") or url)
            self._remember_remote(
                remote,
                location,
                getattr(metadata, "xet_file_data", None),
            )
        except Exception as exc:
            logger.debug(f"Could not refresh the download link for {remote.filename}: {exc}")
            location = url
        with self._url_lock:
            self._url_cache[remote.key] = (
                location,
                time.monotonic() + float(self.config["url_refresh_seconds"]),
            )
            self._url_refreshing.discard(remote.key)
        return location, self._headers_for(location)

    def _expire_download_target(self, remote: RemoteFile) -> None:
        with self._url_lock:
            self._url_cache.pop(remote.key, None)

    @staticmethod
    def _status_code(exc: BaseException) -> Optional[int]:
        response = getattr(exc, "response", None)
        return getattr(response, "status_code", None)

    def _is_retryable(self, exc: BaseException) -> bool:
        if isinstance(exc, (DownloadError, requests.exceptions.RequestException, OSError)):
            if isinstance(exc, RangeNotSupported):
                return False
            status = self._status_code(exc)
            if status is not None and status not in (408, 429) and status < 500:
                return False
            return True
        return False

    def _friendly_error(self, exc: BaseException) -> str:
        status = self._status_code(exc)
        if isinstance(exc, requests.exceptions.SSLError):
            return "TLS verification failed. Check the clock, CA certificates, proxy or firewall."
        if isinstance(exc, requests.exceptions.Timeout):
            return "The connection kept timing out. Re-run to resume from the saved bytes."
        if isinstance(exc, requests.exceptions.ConnectionError):
            return "Could not reach Hugging Face. Check the connection, proxy, DNS or firewall."
        if status == 401:
            return "Authentication failed. Run `hf auth login` if the repository is private."
        if status == 404:
            return "The file does not exist in the repository."
        if status == 429:
            return "Hugging Face rate-limited the request. Wait a moment, then re-run to resume."
        if status is not None and status >= 500:
            return f"Hugging Face returned HTTP {status}. Re-run later to resume."
        if isinstance(exc, PermissionError):
            return "The destination is locked. Close whatever is using the file and retry."
        if isinstance(exc, OSError) and (
            getattr(exc, "errno", None) == 28 or getattr(exc, "winerror", None) == 112
        ):
            return "Not enough free disk space for this download."
        return str(exc) or type(exc).__name__

    # ------------------------------ resume state -----------------------------

    @staticmethod
    def _staging_path(target: Path) -> Path:
        return Path(str(target) + ".part")

    @staticmethod
    def _state_path(target: Path) -> Path:
        return Path(str(target) + ".part.json")

    def _write_state(self, target: Path, remote: RemoteFile, ranges: _Ranges) -> None:
        payload = {
            "version": RESUME_STATE_VERSION,
            "repo_id": remote.repo_id,
            "filename": remote.filename,
            "commit_hash": remote.commit_hash,
            "digest": remote.digest,
            "size": remote.size,
            "ranges": ranges.snapshot(),
        }
        path = self._state_path(target)
        try:
            # Publish only ranges whose staged data has reached stable storage.
            # If power is lost after this point, recovery may conservatively
            # redownload newer bytes but cannot trust unwritten ones.
            staging = self._staging_path(target)
            if staging.is_file():
                # Windows rejects fsync on a read-only descriptor (errno 9).
                descriptor = os.open(str(staging), os.O_RDWR)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            temp = Path(str(path) + ".tmp")
            with open(temp, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            temp.replace(path)
            _fsync_parent(path)
        except OSError as exc:
            logger.debug(f"Could not save the resume map for {target.name}: {exc}")

    def _read_state(self, target: Path, remote: RemoteFile) -> Optional[_Ranges]:
        """Rebuild the saved byte map, rejecting anything that is not this file."""
        path = self._state_path(target)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            self.log(f"[WARNING] Ignoring an unreadable resume map for {target.name}")
            return None

        if not isinstance(data, dict):
            return None
        identity_changed = (
            data.get("repo_id") != remote.repo_id
            or data.get("filename") != remote.filename
            or data.get("size") != remote.size
            or data.get("digest") != remote.digest
            # A digest identifies content across commits. Without one, the
            # pinned commit is the only safe way to distinguish revisions.
            or (
                remote.digest is None
                and data.get("commit_hash") != remote.commit_hash
            )
        )
        if identity_changed:
            self.log(f"[WARNING] The saved ranges belong to another revision of {target.name}; starting over")
            return None

        try:
            on_disk = self._staging_path(target).stat().st_size
        except OSError:
            return None

        # Version 1 recorded one entry per connection as [start, end, done],
        # where only the first ``done`` bytes of that range had landed. Reading
        # it here is what lets a download interrupted by the previous release
        # continue instead of starting the file again.
        if data.get("version") == 1:
            entries = [
                (entry[0], entry[0] + entry[2])
                for entry in data.get("segments") or []
                if isinstance(entry, list) and len(entry) == 3 and all(isinstance(v, int) for v in entry)
            ]
        elif data.get("version") == RESUME_STATE_VERSION:
            entries = [
                (entry[0], entry[1])
                for entry in data.get("ranges") or []
                if isinstance(entry, list) and len(entry) == 2 and all(isinstance(v, int) for v in entry)
            ]
        else:
            return None

        # Anything past the end of the staging file was never actually written.
        clipped = [
            (start, min(end, remote.size, on_disk))
            for start, end in entries
            if 0 <= start < min(end, remote.size, on_disk)
        ]
        return _Ranges(clipped) if clipped else None

    def _discard_staging(self, target: Path) -> None:
        with suppress(OSError):
            self._staging_path(target).unlink()
        with suppress(OSError):
            self._state_path(target).unlink()
        with suppress(OSError):
            Path(str(self._state_path(target)) + ".tmp").unlink()
        # Numbered chunks written by releases before the single staging file.
        # iterdir rather than glob, because a file name is not a glob pattern.
        with suppress(OSError):
            for path in target.parent.iterdir():
                if path.name.startswith(target.name + ".part") and _LEGACY_PART_RE.search(path.name):
                    with suppress(OSError):
                        path.unlink()

    def _create_staging(self, staging: Path, size: int) -> None:
        """Make sure the staging file exists and is exactly ``size`` long.

        Reserving the length up front means every worker writes into a file that
        is already the right shape, so no write has to extend it. On Windows the
        file is also marked sparse, which keeps the reservation free instead of
        zero-filling gigabytes before the first byte arrives.
        """
        staging.parent.mkdir(parents=True, exist_ok=True)
        if not staging.exists():
            with open(staging, "wb"):
                pass
            _enable_sparse_file(staging)
        with open(staging, "r+b") as handle:
            if handle.seek(0, os.SEEK_END) != size:
                handle.truncate(size)
        _reserve_blocks(staging, size)

    # ------------------------------- transfer --------------------------------

    def _connection_count(self, pieces: int, file_size: int = 0) -> int:
        """Choose a bounded HTTP fleet from file size and machine resources."""
        requested = max(1, int(self.config["num_connections"]))
        if not int(self.config.get("http_adaptive_connections", 0)):
            return max(1, min(requested, pieces))

        floor = max(1, min(requested, 16))
        ceiling = max(floor, min(32, int(self.config.get("http_max_connections", 32))))
        memory = _total_memory_bytes()
        cpus = os.cpu_count() or 1
        # Low-resource systems remain at 16.  Bigger fleets are reserved for a
        # long queue, so fragmented resumes do not pay a large socket-startup
        # penalty merely because their ranges are scattered.
        machine_cap = floor
        if cpus >= 16 and memory >= 32 * 1024**3:
            machine_cap = min(24, ceiling)
        if cpus >= 24 and memory >= 64 * 1024**3:
            machine_cap = ceiling
        # The measured 48 GiB Hugging Face case was fastest at 16. Extra sockets
        # are useful mainly when a much larger object provides a long enough
        # queue to amortize handshakes, shaping waves, and connection rotation.
        if file_size and file_size < 64 * 1024**3:
            machine_cap = floor
        elif file_size and file_size < 128 * 1024**3:
            machine_cap = min(machine_cap, 24)
        if pieces < 512:
            machine_cap = floor
        elif pieces < 1024:
            machine_cap = min(machine_cap, 24)
        return max(1, min(machine_cap, pieces))

    def _piece_size_for(self, size: int) -> int:
        # Pieces grow with the file so a huge model keeps each request open
        # long enough to matter, without a lost piece ever costing much. The
        # byte map records what actually landed, so resuming does not depend
        # on the file being cut the same way twice.
        floor = max(1, int(self.config["piece_size"]))
        cap = max(floor, int(self.config["piece_size_max"]))
        target = max(1, int(self.config["pieces_per_file_target"]))
        return min(cap, max(floor, size // target))

    @staticmethod
    def _check_range_response(response, start: int, end: int, size: int) -> None:
        content_range = (response.headers.get("Content-Range") or "").strip()
        match = _CONTENT_RANGE_RE.fullmatch(content_range)
        if match is None:
            raise DownloadError(f"missing or invalid Content-Range: {content_range!r}")
        if tuple(int(value) for value in match.groups()) != (start, end, size):
            raise DownloadError(
                f"the server answered {content_range!r} instead of bytes {start}-{end}/{size}"
            )
        encoding = (response.headers.get("Content-Encoding") or "identity").lower()
        if encoding not in ("", "identity"):
            raise DownloadError(f"unexpected Content-Encoding {encoding!r} on a byte range")

    def _xet_details(self, remote: RemoteFile):
        """Return the optional native Xet backend and this file's Xet identity."""
        if (
            not int(self.config["xet_enabled"])
            or remote.size < int(self.config["xet_threshold"])
        ):
            return None
        try:
            import hf_xet

            if not all(
                hasattr(hf_xet, name)
                for name in ("XetConfig", "XetFileInfo", "XetSession")
            ):
                return None
            with self._remote_lock:
                cached = self._xet_cache.get(remote.key)
            if cached is None:
                url = hf_hub_url(
                    remote.repo_id,
                    remote.filename,
                    revision=remote.commit_hash or "main",
                )
                metadata = self._get_file_metadata(url)
                if getattr(metadata, "size", None) != remote.size:
                    raise DownloadError("Xet metadata reported a different file size")
                _, digest = _digest_from_etag(getattr(metadata, "etag", None))
                if remote.digest and digest != remote.digest:
                    raise DownloadError("Xet metadata belongs to a different file revision")
                self._remember_remote(
                    remote,
                    str(getattr(metadata, "location", "") or url),
                    getattr(metadata, "xet_file_data", None),
                )
                with self._remote_lock:
                    cached = self._xet_cache.get(remote.key)
            if cached is None:
                return None
            return hf_xet, cached[0], cached[1]
        except Exception as exc:
            logger.debug(f"Native Xet is unavailable for {remote.filename}: {exc}")
            return None

    def _new_xet_session(self, hf_xet):
        """Build the measured fast Xet profile without overcommitting small PCs."""
        concurrency = max(1, min(124, int(self.config["xet_concurrency"])))
        override = os.environ.get("HF_XET_FIXED_DOWNLOAD_CONCURRENCY")
        if override:
            with suppress(ValueError):
                concurrency = max(1, min(124, int(override)))

        total_memory = _total_memory_bytes()
        if total_memory and total_memory < 8 * 1024**3:
            concurrency = min(concurrency, 8)
        elif total_memory and total_memory < 16 * 1024**3:
            concurrency = min(concurrency, 12)
        elif total_memory and total_memory < 32 * 1024**3:
            concurrency = min(concurrency, 16)
        elif total_memory and total_memory < int(self.config["xet_high_memory_threshold"]):
            concurrency = min(concurrency, 24)

        updates = {
            "client.enable_adaptive_concurrency": False,
            "client.ac_initial_download_concurrency": concurrency,
            "client.ac_min_download_concurrency": concurrency,
            "client.ac_max_download_concurrency": concurrency,
            "client.max_idle_connections": max(16, concurrency),
        }
        gib = 1024**3
        mib = 1024**2
        if total_memory and total_memory < 8 * gib:
            updates.update(
                {
                    "reconstruction.min_reconstruction_fetch_size": 64 * mib,
                    "reconstruction.max_reconstruction_fetch_size": 512 * mib,
                    "reconstruction.download_buffer_size": 256 * mib,
                    "reconstruction.download_buffer_perfile_size": 128 * mib,
                    "reconstruction.download_buffer_limit": 512 * mib,
                    "reconstruction.min_prefetch_buffer": 128 * mib,
                }
            )
        elif total_memory and total_memory < 16 * gib:
            updates.update(
                {
                    "reconstruction.min_reconstruction_fetch_size": 128 * mib,
                    "reconstruction.max_reconstruction_fetch_size": 1 * gib,
                    "reconstruction.download_buffer_size": 512 * mib,
                    "reconstruction.download_buffer_perfile_size": 256 * mib,
                    "reconstruction.download_buffer_limit": 1 * gib,
                    "reconstruction.min_prefetch_buffer": 256 * mib,
                }
            )
        elif total_memory and total_memory < 32 * gib:
            updates.update(
                {
                    "reconstruction.max_reconstruction_fetch_size": 2 * gib,
                    "reconstruction.download_buffer_size": 1 * gib,
                    "reconstruction.download_buffer_perfile_size": 256 * mib,
                    "reconstruction.download_buffer_limit": 2 * gib,
                    "reconstruction.min_prefetch_buffer": 512 * mib,
                }
            )
        # Keep hf_xet's bounded reconstruction-buffer defaults on every host.
        # Larger buffers increased peak RSS roughly in proportion to model size
        # (several GiB for a single 20 GiB model) without being portable to the
        # ordinary Windows and Linux computers this downloader targets.
        config = hf_xet.XetConfig().with_config(updates)
        return hf_xet.XetSession(config), concurrency

    def _run_xet(
        self,
        remote: RemoteFile,
        target: Path,
        display: str,
        ranges: _Ranges,
        gaps: Sequence[Tuple[int, int]],
        details,
        show_progress: bool,
        shared_meter: Optional[_SpeedMeter],
    ) -> None:
        """Stream exact missing ranges through native Xet and retain our map."""
        hf_xet, file_hash, refresh_route = details
        session, concurrency = self._new_xet_session(hf_xet)
        info = hf_xet.XetFileInfo(file_hash, remote.size)
        group = session.new_download_stream_group(
            token_refresh_url=refresh_route,
            token_refresh_headers=dict(self._hf_headers),
            custom_headers={
                "user-agent": self._hf_headers.get(
                    "user-agent", "secourses-model-downloader"
                )
            },
        )
        staging = self._staging_path(target)
        meter = _SpeedMeter(float(self.config["speed_window"]))
        pending: List[Tuple[int, int]] = []
        pending_bytes = 0
        flush_size = max(1, int(self.config["xet_range_flush_size"]))
        already_done = ranges.total
        action = "RESUMING" if already_done else "DOWNLOADING"
        started = time.monotonic()
        last_state = started
        last_progress = 0.0

        def commit_ranges() -> None:
            nonlocal pending, pending_bytes
            if pending:
                ranges.add_many(pending)
                pending = []
                pending_bytes = 0

        if show_progress:
            detail = self.format_bytes(remote.size)
            if already_done:
                detail += f", {self.format_bytes(already_done)} already on disk"
            self.log(
                f"[{action}] {display} ({detail}) with native Xet "
                f"({concurrency} streams)"
            )

        try:
            with open(staging, "r+b", buffering=0) as handle:
                for gap_start, gap_stop in gaps:
                    stream = group.download_unordered_stream(
                        info,
                        start=gap_start,
                        end=gap_stop,
                    )
                    try:
                        for relative, data in stream:
                            relative = int(relative)
                            absolute = gap_start + relative
                            stop = absolute + len(data)
                            if (
                                not data
                                or relative < 0
                                or absolute < gap_start
                                or stop > gap_stop
                            ):
                                raise DownloadError("Xet returned an invalid byte range")
                            handle.seek(absolute)
                            view = memoryview(data)
                            written = 0
                            while written < len(view):
                                step = handle.write(view[written:])
                                if not step:
                                    raise DownloadError(
                                        "the destination stopped accepting Xet writes"
                                    )
                                written += step
                            pending.append((absolute, stop))
                            pending_bytes += len(data)
                            meter.add(len(data))

                            now = time.monotonic()
                            if pending_bytes >= flush_size:
                                commit_ranges()
                            if now - last_state >= float(self.config["state_flush_interval"]):
                                commit_ranges()
                                self._write_state(target, remote, ranges)
                                last_state = now
                            if (
                                show_progress
                                and now - last_progress >= float(self.config["progress_interval"])
                            ):
                                current = min(remote.size, ranges.total + pending_bytes)
                                speed = meter.rate()
                                remaining = max(0, remote.size - current)
                                eta = self.format_time(remaining / speed) if speed > 0 else "--"
                                self.show_progress_line(
                                    f"[{action}] {display}: "
                                    f"{current * 100.0 / max(1, remote.size):.1f}% "
                                    f"({self.format_bytes(current)}/{self.format_bytes(remote.size)}) "
                                    f"{self.format_bytes(speed)}/s ETA {eta} | Xet {concurrency} streams",
                                    bytes_done=current,
                                    bytes_total=remote.size,
                                )
                                last_progress = now
                    finally:
                        with suppress(Exception):
                            stream.cancel()
            commit_ranges()
            if ranges.total != remote.size:
                raise DownloadError(
                    f"{self.format_bytes(remote.size - ranges.total)} are still missing"
                )
        except KeyboardInterrupt:
            with suppress(Exception):
                session.sigint_abort()
            raise
        except DownloadError:
            raise
        except Exception as exc:
            raise DownloadError(f"native Xet transfer failed: {exc}") from exc
        finally:
            commit_ranges()
            self._write_state(target, remote, ranges)
            if shared_meter is not None:
                shared_meter.add(meter.total)
            if show_progress:
                self.finalize_progress_line()

        elapsed = max(0.001, time.monotonic() - started)
        if show_progress and meter.total:
            self.log(
                f"[TRANSFERRED] {display}: {self.format_bytes(meter.total)} in "
                f"{self.format_time(elapsed)} ({self.format_bytes(meter.total / elapsed)}/s)"
            )

    def _try_run_xet(
        self,
        remote: RemoteFile,
        target: Path,
        display: str,
        ranges: _Ranges,
        show_progress: bool,
        shared_meter: Optional[_SpeedMeter],
    ) -> bool:
        details = self._xet_details(remote)
        if details is None:
            return False
        gaps = ranges.missing(0, remote.size)
        if not gaps:
            return False
        configured_limit = int(self.config.get("xet_max_resume_gaps", 0))
        if show_progress and ranges.total and len(gaps) > 1:
            suffix = ""
            if configured_limit > 0 and len(gaps) > configured_limit:
                suffix = f" (legacy limit {configured_limit} ignored safely)"
            self.log(
                f"[RESUME] {display}: native Xet will fetch "
                f"{len(gaps)} exact missing ranges{suffix}"
            )
        self._run_xet(
            remote,
            target,
            display,
            ranges,
            gaps,
            details,
            show_progress,
            shared_meter,
        )
        return True

    def _fetch_piece(
        self,
        remote: RemoteFile,
        link: _Link,
        handle,
        ranges: _Ranges,
        start: int,
        stop: int,
        cancel: threading.Event,
        meter: _SpeedMeter,
        progress: _Progress,
    ) -> None:
        """Fetch ``[start, stop)`` and record every byte that reaches the file.

        Whatever arrives before a failure is written and kept, so a retry asks
        only for the bytes that are genuinely still missing, whether the retry
        happens seconds later or on a re-run days later.
        """
        block = int(self.config["read_block_size"])
        flush_at = int(self.config["write_block_size"])
        timeout = (float(self.config["connect_timeout"]), float(self.config["read_timeout"]))
        min_rate = float(self.config["min_piece_rate"])
        min_seconds = float(self.config["min_piece_seconds"])
        slow_grace = float(self.config["slow_connection_grace"])
        slow_fraction = float(self.config["slow_connection_fraction"])
        max_retries = max(1, int(self.config["max_retries"]))
        buffer = bytearray()
        offset = start
        # ``stuck`` counts only the failures that moved nothing. A connection
        # that dies after delivering part of its range has still made the file
        # smaller, so it earns a fresh budget; the hard cap is what guarantees
        # the loop ends even against a server that fails in tiny increments.
        stuck = 0
        attempts = 0

        def flush() -> None:
            """Put the batch on disk, then record those bytes as present."""
            nonlocal buffer, offset
            moved = len(buffer)
            if not moved:
                return
            with memoryview(buffer) as view:
                handle.seek(offset)
                written = 0
                while written < moved:
                    step = handle.write(view[written:])
                    if not step:
                        raise DownloadError("the destination stopped accepting writes")
                    written += step
            # Only bytes that survived the write are claimed, so a crash between
            # these two lines costs a retry rather than a corrupt file.
            ranges.add(offset, offset + moved)
            offset += moved
            buffer = bytearray()
            meter.add(moved)
            progress.touch()
            test_limit = int(self.config.get("max_bytes_for_test", 0))
            if test_limit > 0 and meter.total >= test_limit:
                raise KeyboardInterrupt

        while offset < stop:
            if cancel.is_set():
                raise _Cancelled()
            url, base_headers = self._download_target(remote)
            headers = dict(base_headers)
            headers["Range"] = f"bytes={offset}-{stop - 1}"
            headers["Accept-Encoding"] = "identity"
            reached = offset
            try:
                with link.session().get(
                    url, headers=headers, stream=True, timeout=timeout
                ) as response:
                    if response.status_code in (401, 403, 410):
                        self._expire_download_target(remote)
                        raise DownloadError(f"the download link expired (HTTP {response.status_code})")
                    if response.status_code == 200:
                        raise RangeNotSupported("the server ignored the Range header")
                    if response.status_code != 206:
                        response.raise_for_status()
                        raise DownloadError(f"unexpected HTTP status {response.status_code}")
                    self._check_range_response(response, offset, stop - 1, remote.size)

                    raw = response.raw
                    began = time.monotonic()
                    received = 0
                    while offset + len(buffer) < stop:
                        if cancel.is_set():
                            raise _Cancelled()
                        data = raw.read(min(block, stop - offset - len(buffer)))
                        if not data:
                            break
                        buffer += data
                        received += len(data)
                        if len(buffer) >= flush_at:
                            flush()
                        elapsed = time.monotonic() - began
                        # A connection that is still crawling after a fair
                        # chance is replaced instead of holding the file back:
                        # below an absolute floor, or far behind what the
                        # other connections are moving right now.
                        if elapsed >= min_seconds and received / elapsed < min_rate:
                            raise DownloadError(
                                f"this connection settled at "
                                f"{self.format_bytes(received / elapsed)}/s"
                            )
                        if elapsed >= slow_grace:
                            fleet = meter.rate() / max(1, progress.active)
                            if (
                                fleet * slow_fraction > min_rate
                                and received / elapsed < fleet * slow_fraction
                            ):
                                raise _TooSlow(
                                    f"{self.format_bytes(received / elapsed)}/s "
                                    f"against a fleet average of "
                                    f"{self.format_bytes(fleet)}/s"
                                )

                flush()
                if offset >= stop:
                    return
                raise DownloadError("the connection closed before the range was complete")
            except (_Cancelled, RangeNotSupported):
                raise
            except Exception as exc:
                # Whatever went wrong, that socket has had its chance.
                link.drop()
                if cancel.is_set():
                    raise _Cancelled() from exc
                # Bytes already in hand are still good; only the rest is retried.
                with suppress(Exception):
                    flush()
                buffer = bytearray()
                attempts += 1
                stuck = 0 if offset > reached else stuck + 1
                if (
                    stuck >= max_retries
                    or attempts >= max_retries * 8
                    or not self._is_retryable(exc)
                ):
                    raise DownloadError(
                        f"bytes {offset}-{stop - 1}: {self._friendly_error(exc)}"
                    ) from exc
                if isinstance(exc, _TooSlow):
                    # The path was slow, not broken; dial again without a pause.
                    continue
                delay = min(
                    float(self.config["retry_delay"]) * (2 ** max(0, stuck - 1)),
                    float(self.config["max_retry_delay"]),
                )
                if cancel.wait(delay):
                    raise _Cancelled()

    def _run_pieces(
        self,
        remote: RemoteFile,
        target: Path,
        display: str,
        ranges: _Ranges,
        show_progress: bool,
        shared_meter: Optional[_SpeedMeter] = None,
    ) -> None:
        """Fetch every missing piece of one file over several connections."""
        staging = self._staging_path(target)
        self._create_staging(staging, remote.size)

        if self._try_run_xet(
            remote, target, display, ranges, show_progress, shared_meter
        ):
            return

        piece = self._piece_size_for(remote.size)
        work: List[Tuple[int, int]] = []
        for piece_start in range(0, remote.size, piece):
            piece_stop = min(piece_start + piece, remote.size)
            # Asking only for genuine gaps preserves every byte from an older
            # run even when this release uses a different piece size.
            work.extend(ranges.missing(piece_start, piece_stop))
        workers = self._connection_count(len(work), remote.size)
        per_session = max(1, int(self.config["pieces_per_connection"]))
        max_age = float(self.config["max_connection_seconds"])
        age_jitter = float(self.config["connection_age_jitter"])
        meter = _SpeedMeter(float(self.config["speed_window"]))
        cancel = threading.Event()
        progress = _Progress()
        errors: List[BaseException] = []
        errors_lock = threading.Lock()
        counter = [0]
        counter_lock = threading.Lock()
        already_done = ranges.total

        def claim() -> Optional[Tuple[int, int]]:
            with counter_lock:
                index = counter[0]
                if index >= len(work):
                    return None
                counter[0] = index + 1
                return work[index]

        action = "RESUMING" if already_done else "DOWNLOADING"
        if show_progress:
            detail = self.format_bytes(remote.size)
            if already_done:
                detail += f", {self.format_bytes(already_done)} already on disk"
            self.log(f"[{action}] {display} ({detail}) over {workers} connections")

        def worker() -> None:
            # Pieces are claimed in file order, so the connections stay clustered
            # around one moving point in the file rather than writing to a dozen
            # places gigabytes apart.
            link = _Link(self._create_session, per_session, max_age, age_jitter)
            try:
                with open(staging, "r+b", buffering=0) as handle:
                    while not cancel.is_set():
                        claimed = claim()
                        if claimed is None:
                            return
                        start, stop = claimed
                        progress.enter()
                        try:
                            self._fetch_piece(
                                remote, link, handle, ranges,
                                start, stop, cancel, meter, progress,
                            )
                        finally:
                            progress.leave()
            except _Cancelled:
                pass
            except BaseException as exc:  # noqa: BLE001 - reported to the caller
                cancel.set()
                with errors_lock:
                    errors.append(exc)
            finally:
                link.drop()

        started = time.monotonic()
        stall_timeout = float(self.config["stall_timeout"])
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="hf-piece"
        )
        interrupted: Optional[BaseException] = None
        try:
            pending = {executor.submit(worker) for _ in range(workers)}
            last_state = time.monotonic()
            while pending:
                _, pending = concurrent.futures.wait(
                    pending, timeout=float(self.config["progress_interval"])
                )
                now = time.monotonic()
                current = ranges.total
                if show_progress:
                    speed = meter.rate()
                    remaining = max(0, remote.size - current)
                    eta = self.format_time(remaining / speed) if speed > 0 else "--"
                    self.show_progress_line(
                        f"[{action}] {display}: {current * 100.0 / max(1, remote.size):.1f}% "
                        f"({self.format_bytes(current)}/{self.format_bytes(remote.size)}) "
                        f"{self.format_bytes(speed)}/s ETA {eta} | {progress.active} conns "
                        f"@ {self.format_bytes(speed / max(1, progress.active))}/s",
                        bytes_done=current,
                        bytes_total=remote.size,
                    )
                if now - last_state >= float(self.config["state_flush_interval"]):
                    self._write_state(target, remote, ranges)
                    last_state = now
                if not cancel.is_set() and now - progress.last > stall_timeout:
                    # Nothing has reached the file for minutes. Stop the workers
                    # and let the caller resume rather than sit here forever.
                    cancel.set()
                    with errors_lock:
                        errors.append(
                            DownloadError(
                                f"nothing arrived for {self.format_time(stall_timeout)}"
                            )
                        )
        except BaseException as exc:  # KeyboardInterrupt included on purpose
            cancel.set()
            interrupted = exc
        finally:
            executor.shutdown(wait=True)
            self._write_state(target, remote, ranges)
            if show_progress:
                self.finalize_progress_line()

        if shared_meter is not None:
            shared_meter.add(meter.total)
        if interrupted is not None:
            raise interrupted
        if errors:
            # A server that ignores Range says so on every connection at once,
            # and that answer decides the fallback, so it outranks the rest.
            raise next(
                (exc for exc in errors if isinstance(exc, RangeNotSupported)), errors[0]
            )

        if ranges.total != remote.size:
            raise DownloadError(
                f"{self.format_bytes(remote.size - ranges.total)} are still missing"
            )
        elapsed = max(0.001, time.monotonic() - started)
        if show_progress and meter.total:
            self.log(
                f"[TRANSFERRED] {display}: {self.format_bytes(meter.total)} in "
                f"{self.format_time(elapsed)} ({self.format_bytes(meter.total / elapsed)}/s)"
            )

    def _stream_whole_file(
        self,
        remote: RemoteFile,
        target: Path,
        display: str,
        show_progress: bool,
    ) -> None:
        """One-connection fallback for servers that refuse byte ranges."""
        staging = self._staging_path(target)
        self._discard_staging(target)
        self._create_staging(staging, 0)
        timeout = (float(self.config["connect_timeout"]), float(self.config["read_timeout"]))
        url, headers = self._download_target(remote)
        headers = dict(headers)
        headers["Accept-Encoding"] = "identity"
        meter = _SpeedMeter(float(self.config["speed_window"]))
        started = time.monotonic()
        # A fallback can run alongside other small-file downloads. Give it a
        # private Session: requests.Session is not safe to mutate/use from many
        # threads at once.
        with self._create_session(1) as session:
            with session.get(url, headers=headers, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                with open(staging, "wb") as handle:
                    for data in response.iter_content(chunk_size=int(self.config["read_block_size"])):
                        if not data:
                            continue
                        handle.write(data)
                        meter.add(len(data))
                        if show_progress:
                            current = meter.total
                            speed = meter.rate()
                            self.show_progress_line(
                                f"[DOWNLOADING] {display}: {current * 100.0 / max(1, remote.size):.1f}% "
                                f"({self.format_bytes(current)}/{self.format_bytes(remote.size)}) "
                                f"{self.format_bytes(speed)}/s",
                                bytes_done=current,
                                bytes_total=remote.size,
                            )
        if show_progress:
            elapsed = max(0.001, time.monotonic() - started)
            self.finalize_progress_line(
                f"[TRANSFERRED] {display}: {self.format_bytes(meter.total)} in "
                f"{self.format_time(elapsed)}"
            )

    def _check_free_space(self, target: Path, size: int, display: str) -> None:
        """Refuse a transfer that cannot possibly fit, before any bytes move."""
        try:
            # Space the staging file already holds is space this does not need
            # again, which is what keeps a resume from being refused on a disk
            # that is nearly full precisely because the download is on it.
            staged = _allocated_file_bytes(self._staging_path(target))
        except OSError:
            staged = 0
        needed = size - staged
        if needed <= 0:
            return
        try:
            free = shutil.disk_usage(target.parent).free
        except OSError:
            return
        if free < needed + (64 << 20):
            raise DownloadError(
                f"{display} needs {self.format_bytes(needed)} but only "
                f"{self.format_bytes(free)} is free on {target.parent}"
            )

    def _install(self, staging: Path, target: Path) -> None:
        attempts = 5
        for attempt in range(attempts):
            try:
                os.replace(staging, target)
                _fsync_parent(target)
                return
            except PermissionError:
                if attempt == attempts - 1:
                    raise
                delay = min(0.25 * (2 ** attempt), 2.0)
                self.log(f"[RETRY] {target.name} is locked; replacing it again in {delay:.2f}s")
                time.sleep(delay)

    def _already_complete(self, remote: RemoteFile, target: Path, display: str) -> bool:
        if not target.is_file():
            return False
        try:
            stat = target.stat()
        except OSError:
            return False
        if stat.st_size != remote.size:
            return False
        if self.skip_verify or not remote.digest:
            self.log(f"[SKIP] {display} is already complete ({self.format_bytes(remote.size)})")
            return True
        if self.is_file_verified(remote, target):
            self.log(f"[SKIP] {display} is already complete and verified")
            return True
        actual = self.compute_digest(target, remote.digest_algorithm, remote.size, display)
        if actual == remote.digest:
            self.finalize_progress_line(f"[SKIP] {display} is already complete and verified")
            self.mark_file_verified(remote, target)
            return True
        self.finalize_progress_line(
            f"[WARNING] {display} does not match the repository copy and is being downloaded again"
        )
        return False

    def ensure_file(
        self,
        remote: RemoteFile,
        target: Path,
        *,
        show_progress: bool = True,
        shared_meter: Optional[_SpeedMeter] = None,
        display: Optional[str] = None,
        retry_corrupt: bool = True,
    ) -> bool:
        """Download, verify and install one file, resuming whatever exists."""
        # Repositories can repeat basenames across folders, so the caller can pass
        # the relative path that tells them apart.
        display = display or target.name

        try:
            target.parent.mkdir(parents=True, exist_ok=True)

            if self._already_complete(remote, target, display):
                self._discard_staging(target)
                return True

            ranges = self._read_state(target, remote)
            if ranges is None:
                self._discard_staging(target)
                ranges = _Ranges()
            self._check_free_space(target, remote.size, display)

            # One transfer normally covers the whole file. When the network
            # disappears underneath it, the byte map is simply picked back up:
            # only a restart that gains nothing at all counts against the
            # budget, so a bad line costs time rather than the download.
            budget = max(1, int(self.config["file_attempts"]))
            stuck = 0
            attempts = 0
            refusals = 0
            unranged = False
            while True:
                reached = ranges.total
                try:
                    if unranged:
                        self._stream_whole_file(remote, target, display, show_progress)
                    else:
                        self._run_pieces(
                            remote, target, display, ranges, show_progress, shared_meter
                        )
                    break
                except RangeNotSupported:
                    # Falling back means restarting the file on one connection,
                    # so a CDN that answers one request with 200 instead of 206
                    # must not be able to throw away gigabytes on its own.
                    refusals = 0 if ranges.total > reached else refusals + 1
                    if ranges.total and refusals < 2:
                        self.log(
                            f"[RETRY] {display}: a range request came back unranged; "
                            f"trying again before giving up {self.format_bytes(ranges.total)}"
                        )
                        time.sleep(5.0)
                        continue
                    self.log(f"[WARNING] {display}: this server does not serve byte ranges")
                    # The single-stream fallback runs through the same retry
                    # budget below, so a drop at 14 GB is not the end of it.
                    unranged = True
                    continue
                except (DownloadError, requests.exceptions.RequestException, OSError) as exc:
                    attempts += 1
                    stuck = 0 if ranges.total > reached else stuck + 1
                    if stuck >= budget or attempts >= budget * 8:
                        raise
                    delay = min(5.0 * stuck, 30.0)
                    self.log(
                        f"[RETRY] {display}: {self._friendly_error(exc)}; "
                        f"resuming from {self.format_bytes(ranges.total)} "
                        f"({ranges.total * 100.0 / max(1, remote.size):.1f}%) in {delay:.0f}s"
                    )
                    time.sleep(delay)

            staging = self._staging_path(target)
            actual_size = staging.stat().st_size
            if actual_size != remote.size:
                raise DownloadError(
                    f"the staged file is {self.format_bytes(actual_size)} but the repository "
                    f"reports {self.format_bytes(remote.size)}"
                )

            if not self.skip_verify and remote.digest:
                actual = self.compute_digest(staging, remote.digest_algorithm, remote.size, display)
                if actual != remote.digest:
                    self.finalize_progress_line(
                        f"[ERROR] {display} failed verification and was discarded"
                    )
                    self.log(f"  Expected: {remote.digest}")
                    self.log(f"  Got:      {actual}")
                    self._discard_staging(target)
                    if retry_corrupt:
                        self.log(f"[RETRY] {display}: downloading it again from scratch")
                        return self.ensure_file(
                            remote,
                            target,
                            show_progress=show_progress,
                            shared_meter=shared_meter,
                            display=display,
                            retry_corrupt=False,
                        )
                    return False
                self.finalize_progress_line(f"[VERIFIED] {display}: {actual[:16]}...")
            elif not remote.digest:
                self.log(f"[INFO] {display}: the repository publishes no digest, size was checked")

            self._install(staging, target)
            with suppress(OSError):
                self._state_path(target).unlink()
            self.mark_file_verified(remote, target)
            self.log(f"[OK] {display} is ready ({self.format_bytes(remote.size)})")
            return True

        except KeyboardInterrupt:
            self.finalize_progress_line()
            self.log(f"[INTERRUPTED] {display}: the transferred bytes were kept for the next run")
            raise
        except Exception as exc:
            self.finalize_progress_line()
            self.log(f"[FAILED] {display}: {self._friendly_error(exc)}")
            self.log("[RESUME] The bytes already fetched were kept; re-run to continue from there.")
            return False

    # -------------------------------- public API -----------------------------

    def download_file(
        self,
        repo_id: str,
        filename: str,
        local_dir: Path,
        local_filename: Optional[str] = None,
    ) -> bool:
        """Fetch one repository file into ``local_dir``."""
        target = Path(local_dir) / (local_filename if local_filename is not None else filename)
        try:
            remote = self.get_remote_file(repo_id, filename)
        except Exception as exc:
            self.log(f"[FAILED] {filename}: {self._friendly_error(exc)}")
            return False
        return self.ensure_file(remote, target)

    def download_many(
        self,
        items: Sequence[Tuple[str, str, Path, str]],
    ) -> Tuple[int, int, List[str]]:
        """Fetch a batch of files, small ones several at a time.

        Repositories with many modest files spend more time on connection setup
        than bandwidth, so those run
        concurrently. Large weights still get every connection to themselves.
        """
        if not items:
            return 0, 0, []

        self.prefetch_metadata([(repo_id, name) for repo_id, name, _, _ in items])

        resolved: List[Tuple[RemoteFile, Path, str]] = []
        failed: List[str] = []
        for repo_id, remote_name, target_dir, local_name in items:
            try:
                remote = self.get_remote_file(repo_id, remote_name)
            except Exception as exc:
                self.log(f"[FAILED] {remote_name}: {self._friendly_error(exc)}")
                failed.append(remote_name)
                continue
            resolved.append((remote, Path(target_dir) / local_name, str(local_name)))

        threshold = int(self.config["parallel_threshold"])
        small = [entry for entry in resolved if entry[0].size < threshold]
        large = [entry for entry in resolved if entry[0].size >= threshold]
        successful = 0
        total = len(resolved)
        index = 0

        if small:
            workers = max(1, min(int(self.config["small_file_workers"]), len(small)))
            self.log(f"[BATCH] Fetching {len(small)} smaller file(s) over {workers} parallel downloads")
            meter = _SpeedMeter(float(self.config["speed_window"]))
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        self.ensure_file,
                        remote,
                        target,
                        show_progress=False,
                        shared_meter=meter,
                        display=label,
                    ): label
                    for remote, target, label in small
                }
                for future in concurrent.futures.as_completed(futures):
                    label = futures[future]
                    index += 1
                    try:
                        ok = future.result()
                    except KeyboardInterrupt:
                        raise
                    except Exception as exc:
                        self.log(f"[FAILED] {label}: {self._friendly_error(exc)}")
                        ok = False
                    if ok:
                        successful += 1
                    else:
                        failed.append(label)
            elapsed_note = self.format_bytes(meter.total)
            self.log(f"[BATCH] {len(small)} smaller file(s) done ({elapsed_note} transferred)")

        for remote, target, label in large:
            index += 1
            self.log(f"\n[{index}/{total}] {label} ({self.format_bytes(remote.size)})")
            if self.ensure_file(remote, target, display=label):
                successful += 1
            else:
                failed.append(label)

        return successful, len(failed), failed

    def download_repo(
        self,
        repo_id: str,
        local_dir: Path,
        exclude_patterns: Optional[List[str]] = None,
        include_prefixes: Optional[List[str]] = None,
        strip_prefix: Optional[str] = None,
    ) -> Tuple[int, int]:
        """Download every matching file from a repository.

        Args:
            repo_id: Repository such as ``MonsterMMORPG/Wan_GGUF``.
            local_dir: Directory that receives the files.
            exclude_patterns: Substrings/suffixes that skip a file.
            include_prefixes: Optional repository paths to restrict the download to.
            strip_prefix: Optional repository prefix removed from local paths.

        Returns:
            ``(successful_count, failed_count)``
        """
        if exclude_patterns is None:
            exclude_patterns = [".gitattributes", ".gitignore"]

        prefixes = [value.strip("/") for value in include_prefixes or [] if value and value.strip("/")]
        strip = strip_prefix.strip("/") if strip_prefix and strip_prefix.strip("/") else None

        self.log(f"[INFO] Listing files in repository: {repo_id}")
        try:
            names = list_repo_file_names(repo_id)
        except Exception as exc:
            self.log(f"[ERROR] Could not list the repository: {self._friendly_error(exc)}")
            return 0, 1

        items: List[Tuple[str, str, Path, str]] = []
        for name in names:
            if any(pattern in name or name.endswith(pattern) for pattern in exclude_patterns):
                continue
            if prefixes and not any(name == prefix or name.startswith(prefix + "/") for prefix in prefixes):
                continue
            local_name = name
            if strip:
                token = strip + "/"
                if name.startswith(token):
                    local_name = name[len(token):]
                elif name == strip:
                    continue
            items.append((repo_id, name, Path(local_dir), local_name))

        self.log(f"[INFO] Found {len(items)} files to download")
        Path(local_dir).mkdir(parents=True, exist_ok=True)

        successful, failed_count, failed_files = self.download_many(items)
        if failed_files:
            self.log(f"\n[WARNING] Failed to download {len(failed_files)} file(s):")
            for name in failed_files:
                self.log(f"  - {name}")
        return successful, failed_count

# =============================================================================
# Video Captioner Pro catalog, status, and CLI
# =============================================================================

def _hf_token() -> Optional[str]:
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    with suppress(Exception):
        return get_token()
    return None


def _status_code(exc: BaseException) -> Optional[int]:
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None)


def _hf_call(action: Callable[[object], Any], description: str, *, quiet: bool = False) -> Any:
    """Run one Hub API operation with rate-limit and bad-token recovery."""
    token: object = _hf_token() or False
    for attempt in range(5):
        try:
            return action(token)
        except Exception as exc:
            status = _status_code(exc)
            if status == 401 and token is not False:
                if not quiet:
                    print(f"[WARNING] Hugging Face rejected the saved token while {description}; retrying anonymously", flush=True)
                token = False
                continue
            if status == 429 and attempt < 4:
                response = getattr(exc, "response", None)
                retry_after = getattr(response, "headers", {}).get("Retry-After") if response is not None else None
                try:
                    delay = min(60.0, max(1.0, float(retry_after)))
                except (TypeError, ValueError):
                    delay = min(30.0, 2.0 ** attempt)
                if not quiet:
                    print(f"[RETRY] Hugging Face rate limit while {description}; retrying in {delay:.0f}s", flush=True)
                time.sleep(delay)
                continue
            raise


def _safe_cache_key(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", key)


def _index_cache_path(key: str) -> Path:
    return CACHE_DIR / f"remote_index_{_safe_cache_key(key)}.json"


def _subfolder_candidates(subfolder: str) -> Tuple[str, ...]:
    """Return the requested Hub folder followed by its alternate layout."""
    requested = subfolder.strip("/")
    if requested.startswith(NESTED_MODEL_PREFIX):
        alternate = requested[len(NESTED_MODEL_PREFIX):]
    elif "/" not in requested:
        alternate = NESTED_MODEL_PREFIX + requested
    else:
        alternate = ""
    return (requested, alternate) if alternate and alternate != requested else (requested,)


def _relative_path(remote_path: str, subfolder: str) -> str:
    prefix = subfolder.strip("/")
    token = prefix + "/" if prefix else ""
    if prefix and remote_path == prefix:
        return PurePosixPath(remote_path).name
    if token and not remote_path.startswith(token):
        raise DownloadError(f"repository returned a path outside {subfolder}: {remote_path}")
    relative = remote_path[len(token):] if token else remote_path
    parsed = PurePosixPath(relative)
    if not relative or parsed.is_absolute() or any(part in ("", ".", "..") for part in parsed.parts):
        raise DownloadError(f"unsafe repository path: {remote_path}")
    return parsed.as_posix()


def _remote_to_json(remote: RemoteFile, relative_path: str) -> Dict[str, Any]:
    return {"path": remote.filename, "relative_path": relative_path, "size": remote.size, "digest_algorithm": remote.digest_algorithm, "digest": remote.digest, "commit_hash": remote.commit_hash}


def _remote_from_json(repo_id: str, item: Dict[str, Any]) -> Tuple[RemoteFile, str]:
    remote_path = str(item["path"])
    relative = str(item["relative_path"])
    size = int(item["size"])
    if size < 0:
        raise ValueError("negative remote size")
    return (RemoteFile(repo_id, remote_path, size, item.get("digest_algorithm"), item.get("digest"), str(item.get("commit_hash") or "main")), relative)


def load_remote_index(key: str, repo_id: str, subfolder: str, *, refresh: bool = False, quiet: bool = False) -> List[Tuple[RemoteFile, str]]:
    """Discover every file below one model folder and cache it for 24 hours."""
    repo_id = repo_id.strip()
    subfolder = subfolder.strip("/")
    if not repo_id or not subfolder:
        raise DownloadError("both repository and subfolder must be non-empty")
    candidates = _subfolder_candidates(subfolder)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _index_cache_path(key)
    if not refresh and cache_path.is_file():
        try:
            with open(cache_path, "r", encoding="utf-8") as handle:
                cached = json.load(handle)
            fresh = time.time() - float(cached.get("fetched_at", 0)) < INDEX_TTL_SECONDS
            cached_subfolder = str(cached.get("subfolder") or "").strip("/")
            if cached.get("version") == 1 and cached.get("repo_id") == repo_id and cached_subfolder in candidates and fresh and isinstance(cached.get("files"), list):
                restored = [_remote_from_json(repo_id, item) for item in cached["files"]]
                if restored:
                    return restored
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            pass

    api = HfApi()
    try:
        info = _hf_call(lambda token: api.repo_info(repo_id=repo_id, repo_type="model", token=token), f"resolving {repo_id}", quiet=quiet)
        commit_hash = str(getattr(info, "sha", "") or "main")
    except Exception as exc:
        if _status_code(exc) == 404:
            raise DownloadError(f"remote model folder {repo_id}/{subfolder} is not available yet") from exc
        raise DownloadError(f"could not list {repo_id}/{subfolder}: {exc}") from exc

    tree = []
    resolved_subfolder = subfolder
    for index, candidate in enumerate(candidates):
        try:
            tree = _hf_call(
                lambda token, candidate=candidate: list(api.list_repo_tree(repo_id, path_in_repo=candidate, recursive=True, expand=True, revision=commit_hash, repo_type="model", token=token)),
                f"listing {repo_id}/{candidate}",
                quiet=quiet,
            )
            resolved_subfolder = candidate
            break
        except Exception as exc:
            if _status_code(exc) == 404 and index + 1 < len(candidates):
                alternate = candidates[index + 1]
                logger.info(f"[FALLBACK] {repo_id}/{candidate} returned 404; trying {repo_id}/{alternate}")
                continue
            if _status_code(exc) == 404:
                raise DownloadError(f"remote model folder {repo_id}/{subfolder} is not available yet") from exc
            raise DownloadError(f"could not list {repo_id}/{candidate}: {exc}") from exc

    files: List[Tuple[RemoteFile, str]] = []
    for entry in tree:
        remote_path, size = getattr(entry, "path", None), getattr(entry, "size", None)
        if not isinstance(remote_path, str) or not isinstance(size, int):
            continue
        lfs = getattr(entry, "lfs", None)
        etag = getattr(lfs, "sha256", None) if lfs is not None else None
        algorithm, digest = _digest_from_etag(etag or getattr(entry, "blob_id", None))
        relative = _relative_path(remote_path, resolved_subfolder)
        files.append((RemoteFile(repo_id, remote_path, size, algorithm, digest, commit_hash), relative))
    files.sort(key=lambda pair: pair[1].casefold())
    if not files:
        raise DownloadError(f"no files were found in {repo_id}/{resolved_subfolder}")

    payload = {"version": 1, "fetched_at": time.time(), "repo_id": repo_id, "subfolder": resolved_subfolder, "commit_hash": commit_hash, "files": [_remote_to_json(remote, relative) for remote, relative in files]}
    temp_path = Path(str(cache_path) + ".tmp")
    try:
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, cache_path)
        _fsync_parent(cache_path)
    except OSError as exc:
        logger.warning(f"Could not cache {cache_path.name}: {exc}")
        with suppress(OSError):
            temp_path.unlink()
    return files


def _filtered_files(files: Sequence[Tuple[RemoteFile, str]], patterns: Sequence[str]) -> List[Tuple[RemoteFile, str]]:
    if not patterns:
        return list(files)
    selected = []
    for item in files:
        relative = item[1]
        basename = PurePosixPath(relative).name
        if any(fnmatch.fnmatchcase(relative, pattern.replace("\\", "/")) or fnmatch.fnmatchcase(basename, pattern.replace("\\", "/")) for pattern in patterns):
            selected.append(item)
    if not selected:
        raise DownloadError(f"--only patterns matched no files: {', '.join(patterns)}")
    return selected


def _target_for(model_dir: Path, relative_path: str) -> Path:
    parsed = PurePosixPath(relative_path)
    if parsed.is_absolute() or any(part in ("", ".", "..") for part in parsed.parts):
        raise DownloadError(f"unsafe local relative path: {relative_path}")
    return model_dir.joinpath(*parsed.parts)


def _directory_bytes(path: Path) -> int:
    total = 0
    if path.is_dir():
        with suppress(OSError):
            for item in path.rglob("*"):
                if item.is_file() and not item.name.endswith(".part") and not item.name.endswith(".part.json"):
                    with suppress(OSError):
                        total += item.stat().st_size
    return total


def _bundled_local_manifest(key: str) -> List[Tuple[str, int, str]]:
    """Read the generated local SHA manifest when the HF folder is unpublished."""

    document = PROJECT_DIR / "docs" / "MODEL_FILES.md"
    try:
        text = document.read_text(encoding="utf-8")
    except OSError:
        return []
    heading = re.search(rf"^##\s+{re.escape(key)}\s*$", text, flags=re.MULTILINE)
    if heading is None:
        return []
    tail = text[heading.end():]
    next_heading = re.search(r"^##\s+", tail, flags=re.MULTILINE)
    section = tail[:next_heading.start()] if next_heading else tail
    rows: List[Tuple[str, int, str]] = []
    pattern = re.compile(
        r"^\|\s*`([^`]+)`\s*\|\s*([\d,]+)\s*\|[^|]*\|\s*`([0-9a-fA-F]{64})`\s*\|",
        flags=re.MULTILINE,
    )
    for name, raw_size, digest in pattern.findall(section):
        rows.append((name, int(raw_size.replace(",", "")), digest.casefold()))
    return rows


def _verify_bundled_manifest(
    key: str,
    target_root: Path,
    downloader: RobustDownloader,
    manifest: Sequence[Tuple[str, int, str]],
) -> bool:
    model_dir = target_root / key
    total = sum(size for _, size, _ in manifest)
    emit_status(key, "verifying", f"using bundled manifest for {len(manifest)} local file(s)", fraction=0.0, bytes_done=0, bytes_total=total)
    for name, size, expected in manifest:
        path = model_dir / name
        if not path.is_file():
            emit_status(key, "missing", f"missing {name}")
            return False
        actual_size = path.stat().st_size
        if actual_size != size:
            emit_status(key, "error", f"{name} has {actual_size} bytes; expected {size}")
            return False
        actual = downloader.compute_digest(path, "sha256", size, name)
        if actual != expected:
            emit_status(key, "error", f"SHA-256 mismatch for {name}")
            return False
        downloader.log(f"[VERIFIED] {name}: {actual[:16]}...")
    emit_status(key, "ready", f"{len(manifest)} local file(s), size and SHA-256 verified", fraction=1.0, bytes_done=total, bytes_total=total)
    return True


def _remote_index_unpublished(error: BaseException) -> bool:
    text = str(error).casefold()
    return "not available yet" in text or "no files were found" in text or "not published" in text


def _local_presence_fallback(
    key: str,
    target_root: Path,
    subfolder: str,
    error: BaseException,
) -> Dict[str, Any]:
    """Report local presence and sizes when published remote digests are unavailable."""

    model_dir = target_root / key
    local_files: List[Path] = []
    if model_dir.is_dir():
        with suppress(OSError):
            local_files = [
                path
                for path in model_dir.rglob("*")
                if path.is_file()
                and not path.name.endswith(".part")
                and not path.name.endswith(".part.json")
            ]
    local_bytes = _directory_bytes(model_dir)
    manifest = _bundled_local_manifest(key)
    missing = wrong_size = 0
    if manifest:
        for name, expected_size, _ in manifest:
            candidate = model_dir / name
            try:
                actual_size = candidate.stat().st_size
            except OSError:
                missing += 1
                continue
            if not candidate.is_file() or actual_size != expected_size:
                wrong_size += 1
        ready = missing == 0 and wrong_size == 0
        required_files: Optional[int] = len(manifest)
    else:
        ready = bool(local_files)
        required_files = len(local_files) if local_files else None
    size_gb = f"{local_bytes / 1_000_000_000:.2f}".rstrip("0").rstrip(".")
    if _remote_index_unpublished(error):
        remote_message = f"Remote folder {subfolder} is not published yet"
    else:
        remote_message = f"Remote index for {subfolder} is unavailable"
    if ready:
        message = (
            f"{remote_message}; local files are present ({size_gb} GB) "
            "but cannot be verified against published digests."
        )
    else:
        problems = []
        if missing:
            problems.append(f"{missing} missing")
        if wrong_size:
            problems.append(f"{wrong_size} wrong size")
        if not local_files:
            problems.append("no local files")
        message = f"{remote_message}; local files are incomplete ({', '.join(problems) or 'not ready'})."
    return {
        "key": key,
        "label": MODEL_CATALOG.get(key, {}).get("label", key),
        "ready": ready,
        "local_bytes": local_bytes,
        "expected_bytes": local_bytes if ready else None,
        "required_files": required_files,
        "verified_files": 0,
        "message": message,
        "verification_mode": "local_presence_and_size",
    }


def _partial_bytes(target: Path, remote: RemoteFile) -> int:
    try:
        staging_path = Path(str(target) + ".part")
        if not staging_path.is_file():
            return 0
        with open(Path(str(target) + ".part.json"), "r", encoding="utf-8") as handle:
            state = json.load(handle)
        if state.get("repo_id") != remote.repo_id or state.get("filename") != remote.filename or state.get("size") != remote.size or state.get("digest") != remote.digest:
            return 0
        ranges = state.get("ranges") if state.get("version") == RESUME_STATE_VERSION else []
        return min(remote.size, _Ranges(item for item in ranges if isinstance(item, list) and len(item) == 2).total)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return 0


def model_status(key: str, files: Sequence[Tuple[RemoteFile, str]], target_root: Path, downloader: RobustDownloader, *, rehash: bool = False) -> Dict[str, Any]:
    model_dir = target_root / key
    expected_bytes = sum(remote.size for remote, _ in files)
    local_bytes = missing = wrong_size = unverified = verified = 0
    for remote, relative in files:
        target = _target_for(model_dir, relative)
        try:
            stat = target.stat()
        except OSError:
            missing += 1
            local_bytes += _partial_bytes(target, remote)
            continue
        local_bytes += stat.st_size
        if not target.is_file() or stat.st_size != remote.size:
            wrong_size += 1
        elif not remote.digest:
            verified += 1
        elif rehash:
            actual = downloader.compute_digest(target, remote.digest_algorithm or "sha256", remote.size, relative)
            if actual == remote.digest:
                downloader.mark_file_verified(remote, target)
                verified += 1
            else:
                unverified += 1
        elif downloader.is_file_verified(remote, target):
            verified += 1
        else:
            unverified += 1
    ready = bool(files) and missing == 0 and wrong_size == 0 and unverified == 0
    if ready:
        detail = f"{len(files)} file(s), {downloader.format_bytes(expected_bytes)}, size and SHA verified"
    else:
        problems = []
        if missing: problems.append(f"{missing} missing")
        if wrong_size: problems.append(f"{wrong_size} wrong size")
        if unverified: problems.append(f"{unverified} need SHA verification")
        detail = ", ".join(problems) or "not ready"
    return {"key": key, "label": MODEL_CATALOG.get(key, {}).get("label", key), "ready": ready, "local_bytes": local_bytes, "expected_bytes": expected_bytes, "required_files": len(files), "verified_files": verified, "message": detail}


def _resolve_target_root(value: Optional[str]) -> Path:
    raw = value or os.environ.get("VCAP_MODELS_DIR") or str(DEFAULT_MODELS_DIR)
    return Path(os.path.expandvars(os.path.expanduser(raw))).resolve()


def _check_model_disk_space(
    key: str,
    target_root: Path,
    expected_bytes: int,
    already_present_bytes: int,
) -> None:
    """Fail before transfer when the remaining model plus margin cannot fit."""

    remaining = max(0, int(expected_bytes) - max(0, int(already_present_bytes)))
    required = int(remaining * 1.05 + 0.999999)
    if required <= 0:
        return
    free = int(shutil.disk_usage(target_root).free)
    if free >= required:
        return
    shortfall_gb = (required - free) / float(1024 ** 3)
    resolved = target_root.resolve()
    drive = resolved.drive or resolved.anchor or str(resolved)
    raise DownloadError(
        f"Not enough free disk space for {key}: need about {shortfall_gb:.1f} GB more on {drive}"
    )


def _app_gguf_modules():
    """Import the app GGUF registry/backend only when a GGUF key is used."""

    project = str(PROJECT_DIR)
    if project not in sys.path:
        sys.path.insert(0, project)
    from vcap.models import registry as app_registry
    from vcap.models.llamacpp_backend import ensure_gguf

    return app_registry, ensure_gguf


def _app_whisper_modules():
    """Import the app Whisper catalogue/downloader only when requested."""

    project = str(PROJECT_DIR)
    if project not in sys.path:
        sys.path.insert(0, project)
    try:
        from vcap.whisper import models as whisper_models
    except Exception as exc:
        raise RuntimeError(
            f"Whisper backend could not be imported from {PROJECT_DIR}: {exc}"
        ) from exc
    return whisper_models


def _whisper_alias(key: str) -> Optional[str]:
    value = str(key or "")
    if not value.casefold().startswith(WHISPER_KEY_PREFIX):
        return None
    return value[len(WHISPER_KEY_PREFIX):]


def collect_whisper_statuses(target_root: Path) -> List[Dict[str, Any]]:
    whisper_models = _app_whisper_modules()
    results: List[Dict[str, Any]] = []
    for model in whisper_models.WHISPER_MODELS:
        key = f"{WHISPER_KEY_PREFIX}{model.alias}"
        ready = whisper_models.is_model_ready(model.alias, target_root)
        local_bytes = whisper_models.local_size_bytes(model.alias, target_root)
        results.append({
            "key": key,
            "label": model.alias,
            "ready": ready,
            "local_bytes": local_bytes,
            "expected_bytes": model.size_bytes,
            "required_files": 2,
            "verified_files": 2 if ready else 0,
            "message": "config.json and model.bin present" if ready else "not downloaded",
            "approx_gb": model.size_bytes / 1_000_000_000,
            "family": "whisper",
            "variant": model.alias,
            "modalities": ["audio", "video"],
        })
    return results


def ensure_whisper_model(key: str, target_root: Path) -> bool:
    whisper_models = _app_whisper_modules()
    alias = _whisper_alias(key)
    if not alias or whisper_models.get_model(alias) is None:
        emit_status(key, "error", f"unknown Whisper model alias: {alias or key}")
        return False

    def callback(payload: Dict[str, Any]) -> None:
        message = str(payload.get("message") or f"downloading {alias}")
        emit_status(
            key,
            "skipped" if "already downloaded" in message.casefold() else "downloading",
            message,
            fraction=payload.get("fraction"),
            bytes_done=payload.get("bytes"),
            bytes_total=payload.get("total"),
        )

    try:
        path = whisper_models.download_model(
            alias,
            models_dir=target_root,
            progress_cb=callback,
        )
        ready = whisper_models.is_model_ready(alias, target_root)
        info = whisper_models.get_model(alias)
        emit_status(
            key,
            "ready" if ready else "error",
            f"Whisper model ready at {path}" if ready else f"Whisper model is incomplete at {path}",
            fraction=1.0 if ready else None,
            bytes_done=info.size_bytes if ready and info else whisper_models.local_size_bytes(alias, target_root),
            bytes_total=info.size_bytes if info else None,
        )
        return ready
    except KeyboardInterrupt:
        emit_status(key, "error", "cancelled; resumable Hugging Face state was preserved")
        raise
    except Exception as exc:
        emit_status(key, "error", str(exc))
        return False


def verify_whisper_model(key: str, target_root: Path) -> bool:
    whisper_models = _app_whisper_modules()
    alias = _whisper_alias(key)
    if not alias or whisper_models.get_model(alias) is None:
        emit_status(key, "error", f"unknown Whisper model alias: {alias or key}")
        return False
    ready = whisper_models.is_model_ready(alias, target_root)
    local_bytes = whisper_models.local_size_bytes(alias, target_root)
    info = whisper_models.get_model(alias)
    emit_status(
        key,
        "ready" if ready else "missing",
        "config.json and model.bin present; no partial files" if ready else "model is missing, incomplete, or has partial files",
        fraction=1.0 if ready else None,
        bytes_done=min(local_bytes, info.size_bytes) if info else local_bytes,
        bytes_total=info.size_bytes if info else None,
    )
    return ready


def _gguf_status(key: str, target_root: Path) -> Dict[str, Any]:
    app_registry, _ = _app_gguf_modules()
    variant = app_registry.get_variant(key)
    folder = target_root / variant.folder_name
    files = variant.gguf_files or ()
    sizes = variant.gguf_file_sizes or ()
    expected = sum(int(value) for value in sizes)
    local = _directory_bytes(folder)
    missing: List[str] = []
    wrong: List[str] = []
    for index, name in enumerate(files):
        path = folder / name
        if not path.is_file():
            missing.append(name)
            continue
        wanted = int(sizes[index]) if index < len(sizes) else 0
        if wanted and path.stat().st_size != wanted:
            wrong.append(name)
    ready = bool(files) and not missing and not wrong
    if ready:
        message = f"{len(files)} GGUF file(s), exact sizes present"
    else:
        problems = []
        if missing:
            problems.append("missing " + ", ".join(missing))
        if wrong:
            problems.append("wrong size " + ", ".join(wrong))
        message = "; ".join(problems) or "not ready"
    return {
        "key": key,
        "label": MODEL_CATALOG[key]["label"],
        "ready": ready,
        "local_bytes": local,
        "expected_bytes": expected,
        "required_files": len(files),
        "verified_files": len(files) if ready else 0,
        "message": message,
    }


def ensure_gguf_model(key: str, target_root: Path) -> bool:
    app_registry, ensure_gguf = _app_gguf_modules()
    emit_status(key, "downloading", "checking registered third-party GGUF files", fraction=0.0)

    def callback(message: Any, payload: Any = None) -> None:
        data = payload if isinstance(payload, dict) else {}
        state = "verifying" if "verif" in str(message).casefold() else "downloading"
        emit_status(
            key,
            state,
            str(message),
            fraction=data.get("fraction"),
            bytes_done=data.get("bytes_done"),
            bytes_total=data.get("bytes_total"),
        )

    original_root = app_registry.MODELS_DIR
    app_registry.MODELS_DIR = target_root
    try:
        current = _gguf_status(key, target_root)
        if not current["ready"]:
            _check_model_disk_space(
                key,
                target_root,
                current["expected_bytes"],
                current["local_bytes"],
            )
        ensure_gguf(key, callback)
        result = _gguf_status(key, target_root)
        emit_status(
            key,
            "ready" if result["ready"] else "error",
            result["message"],
            fraction=1.0 if result["ready"] else None,
            bytes_done=result["expected_bytes"] if result["ready"] else result["local_bytes"],
            bytes_total=result["expected_bytes"],
        )
        return bool(result["ready"])
    except KeyboardInterrupt:
        emit_status(key, "error", "cancelled; resumable downloads were preserved")
        raise
    except Exception as exc:
        emit_status(key, "error", str(exc))
        return False
    finally:
        app_registry.MODELS_DIR = original_root


def verify_gguf_model(key: str, target_root: Path, downloader: RobustDownloader) -> bool:
    app_registry, _ = _app_gguf_modules()
    variant = app_registry.get_variant(key)
    files = variant.gguf_files or ()
    sizes = variant.gguf_file_sizes or ()
    hashes = variant.gguf_sha256 or ()
    folder = target_root / variant.folder_name
    total = sum(int(value) for value in sizes)
    emit_status(key, "verifying", f"re-hashing {len(files)} local GGUF file(s)", fraction=0.0, bytes_done=0, bytes_total=total)
    downloader.set_status_key(key)
    try:
        for index, name in enumerate(files):
            path = folder / name
            if not path.is_file():
                emit_status(key, "missing", f"missing {name}")
                return False
            expected_size = int(sizes[index]) if index < len(sizes) else path.stat().st_size
            if path.stat().st_size != expected_size:
                emit_status(key, "error", f"{name} has the wrong size")
                return False
            expected_hash = str(hashes[index]).casefold() if index < len(hashes) else ""
            actual_hash = downloader.compute_digest(path, "sha256", expected_size, name)
            if expected_hash and actual_hash != expected_hash:
                emit_status(key, "error", f"SHA-256 mismatch for {name}")
                return False
        result = _gguf_status(key, target_root)
        emit_status(key, "ready", f"{len(files)} GGUF file(s), size and SHA-256 verified", fraction=1.0, bytes_done=total, bytes_total=total)
        return bool(result["ready"])
    except KeyboardInterrupt:
        emit_status(key, "error", "verification cancelled")
        raise
    except Exception as exc:
        emit_status(key, "error", str(exc))
        return False
    finally:
        downloader.set_status_key(None)


def _selection_source(key: str, repo_override: Optional[str], subfolder_override: Optional[str]) -> Tuple[str, str]:
    return repo_override or DEFAULT_REPO_ID, subfolder_override or MODEL_CATALOG[key]["hf_subfolder"]


def ensure_model(key: str, target_root: Path, downloader: RobustDownloader, *, repo_override: Optional[str] = None, subfolder_override: Optional[str] = None, only_patterns: Sequence[str] = (), refresh_index: bool = False) -> bool:
    if key in GGUF_KEYS:
        return ensure_gguf_model(key, target_root)
    repo_id, subfolder = _selection_source(key, repo_override, subfolder_override)
    emit_status(key, "downloading", f"checking {repo_id}/{subfolder}", fraction=0.0)
    downloader.set_status_key(key)
    try:
        files = _filtered_files(load_remote_index(key, repo_id, subfolder, refresh=refresh_index), only_patterns)
        current = model_status(key, files, target_root, downloader)
        if current["ready"]:
            downloader.log(
                f"[SKIP] {key} is already complete and verified "
                f"({downloader.format_bytes(current['expected_bytes'])})"
            )
            emit_status(
                key,
                "skipped",
                current["message"],
                fraction=1.0,
                bytes_done=current["expected_bytes"],
                bytes_total=current["expected_bytes"],
            )
            return True
        _check_model_disk_space(
            key,
            target_root,
            current["expected_bytes"],
            current["local_bytes"],
        )
        expected = sum(remote.size for remote, _ in files)
        emit_status(key, "downloading", f"{len(files)} file(s), {downloader.format_bytes(expected)}", fraction=0.0, bytes_done=0, bytes_total=expected)
        model_dir = target_root / key
        for remote, _ in files:
            downloader._remember_remote(remote)
        items = [(remote.repo_id, remote.filename, model_dir, relative) for remote, relative in files]
        successful, failed_count, failed_files = downloader.download_many(items)
        if failed_count or successful != len(items):
            message = f"{failed_count} file(s) failed"
            if failed_files:
                message += f": {', '.join(failed_files[:3])}"
            emit_status(key, "error", message)
            return False
        final = model_status(key, files, target_root, downloader)
        emit_status(
            key,
            "ready" if final["ready"] else "error",
            final["message"],
            fraction=1.0 if final["ready"] else None,
            bytes_done=final["local_bytes"],
            bytes_total=final["expected_bytes"],
        )
        return bool(final["ready"])
    except KeyboardInterrupt:
        emit_status(key, "error", "cancelled; resumable partial files were preserved")
        raise
    except DownloadError as exc:
        emit_status(key, "error", str(exc))
        return False
    except Exception as exc:
        if only_patterns:
            emit_status(key, "error", str(exc))
            return False
        fallback = _local_presence_fallback(key, target_root, subfolder, exc)
        if fallback["ready"]:
            emit_status(
                key,
                "skipped",
                fallback["message"],
                fraction=1.0,
                bytes_done=fallback["local_bytes"],
                bytes_total=fallback["local_bytes"],
            )
            return True
        emit_status(key, "error", fallback["message"] if _remote_index_unpublished(exc) else str(exc))
        return False
    finally:
        downloader.set_status_key(None)


def verify_model(key: str, target_root: Path, downloader: RobustDownloader, *, repo_override: Optional[str] = None, subfolder_override: Optional[str] = None, only_patterns: Sequence[str] = (), refresh_index: bool = False) -> bool:
    if key in GGUF_KEYS:
        return verify_gguf_model(key, target_root, downloader)
    repo_id, subfolder = _selection_source(key, repo_override, subfolder_override)
    emit_status(key, "verifying", "re-hashing local files", fraction=0.0)
    downloader.set_status_key(key)
    try:
        try:
            files = _filtered_files(load_remote_index(key, repo_id, subfolder, refresh=refresh_index), only_patterns)
        except Exception as exc:
            if only_patterns:
                raise
            fallback = _local_presence_fallback(key, target_root, subfolder, exc)
            emit_status(
                key,
                "skipped" if fallback["ready"] else "missing",
                fallback["message"],
                fraction=1.0 if fallback["ready"] else None,
                bytes_done=fallback["local_bytes"],
                bytes_total=fallback["local_bytes"] if fallback["ready"] else None,
            )
            return bool(fallback["ready"])
        result = model_status(key, files, target_root, downloader, rehash=True)
        emit_status(
            key,
            "ready" if result["ready"] else "error",
            result["message"],
            fraction=1.0 if result["ready"] else None,
            bytes_done=result["local_bytes"],
            bytes_total=result["expected_bytes"],
        )
        return bool(result["ready"])
    except KeyboardInterrupt:
        emit_status(key, "error", "verification cancelled")
        raise
    except Exception as exc:
        emit_status(key, "error", str(exc))
        return False
    finally:
        downloader.set_status_key(None)


def collect_statuses(target_root: Path, downloader: RobustDownloader, *, repo_override: Optional[str] = None, refresh_index: bool = False, quiet: bool = False) -> List[Dict[str, Any]]:
    results = []
    for key, spec in MODEL_CATALOG.items():
        try:
            if key in GGUF_KEYS:
                result = _gguf_status(key, target_root)
            else:
                files = load_remote_index(key, repo_override or DEFAULT_REPO_ID, spec["hf_subfolder"], refresh=refresh_index, quiet=quiet)
                result = model_status(key, files, target_root, downloader)
        except Exception as exc:
            if key in GGUF_KEYS:
                result = {"key": key, "label": spec["label"], "ready": False, "local_bytes": _directory_bytes(target_root / key), "expected_bytes": None, "required_files": None, "verified_files": 0, "message": str(exc)}
            else:
                result = _local_presence_fallback(key, target_root, spec["hf_subfolder"], exc)
        result.update(approx_gb=spec["approx_gb"], family=spec["family"], variant=spec["variant"], modalities=list(spec["modalities"]))
        results.append(result)
    return results


def print_catalog(statuses: Sequence[Dict[str, Any]], target_root: Path) -> None:
    print("SECourses Video Captioner Pro model catalog")
    print("=" * 108)
    print(f"{'KEY':38} {'APPROX':>9}  {'STATUS':12} LABEL")
    print("-" * 108)
    for item in statuses:
        status = "READY" if item["ready"] else "NOT READY"
        print(f"{item['key']:38} {item['approx_gb']:7.1f} GB  {status:12} {item['label']}")
        if item.get("message") and (not item["ready"] or item.get("verification_mode")):
            print(f"{'':52} {item['message']}")
    print(f"\nModels directory: {target_root}")


def print_whisper_catalog(statuses: Sequence[Dict[str, Any]], target_root: Path) -> None:
    print("\nWhisper speech models")
    print("=" * 108)
    print(f"{'KEY':38} {'SIZE':>9}  {'STATUS':12} MODEL")
    print("-" * 108)
    for item in statuses:
        status = "READY" if item["ready"] else "NOT READY"
        print(f"{item['key']:38} {item['approx_gb']:7.2f} GB  {status:12} {item['label']}")
    print(f"\nWhisper directory: {target_root / 'whisper'}")


def interactive_selection(target_root: Path, downloader: RobustDownloader, *, repo_override: Optional[str], refresh_index: bool) -> List[str]:
    statuses = collect_statuses(target_root, downloader, repo_override=repo_override, refresh_index=refresh_index)
    whisper_statuses = collect_whisper_statuses(target_root)
    print("\nSECourses Video Captioner Pro Models")
    print("=" * 90)
    caption_keys = list(MODEL_CATALOG)
    for index, (key, item) in enumerate(zip(caption_keys, statuses), 1):
        marker = "READY" if item["ready"] else "not ready"
        print(f"{index:2}. {MODEL_CATALOG[key]['label']} (~{MODEL_CATALOG[key]['approx_gb']:.1f} GB) [{marker}]")
    print("\nWhisper speech models")
    print("-" * 90)
    whisper_keys = [item["key"] for item in whisper_statuses]
    for index, item in enumerate(whisper_statuses, len(caption_keys) + 1):
        marker = "READY" if item["ready"] else "not ready"
        print(f"{index:2}. {item['label']} ({item['approx_gb']:.2f} GB) [{marker}]")
    keys = caption_keys + whisper_keys
    print("\nEnter numbers separated by commas, ranges such as 1-3, A for all, or Q to quit.")
    while True:
        try:
            answer = input("Selection: ").strip()
        except EOFError:
            return []
        if answer.lower() in ("q", "quit", "exit"):
            return []
        if answer.lower() in ("a", "all"):
            return keys
        chosen, valid = [], True
        for token in [part.strip() for part in answer.split(",") if part.strip()]:
            if "-" in token:
                bounds = token.split("-", 1)
                if len(bounds) != 2 or not all(value.strip().isdigit() for value in bounds):
                    valid = False
                    break
                first, last = (int(value.strip()) for value in bounds)
                if first > last:
                    first, last = last, first
                chosen.extend(range(first, last + 1))
            elif token.isdigit():
                chosen.append(int(token))
            else:
                valid = False
                break
        if valid and chosen and all(1 <= value <= len(keys) for value in chosen):
            return [keys[value - 1] for value in dict.fromkeys(chosen)]
        print("Invalid selection. Use comma-separated numbers and ranges from the list.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and verify SECourses Video Captioner Pro models", formatter_class=argparse.RawDescriptionHelpFormatter, epilog="""Examples:
  python Models_Downloader.py
  python Models_Downloader.py --list
  python Models_Downloader.py --ensure timechat_int4
  python Models_Downloader.py --ensure avocado_int8 --ensure timechat_int8
  python Models_Downloader.py --ensure whisper:large-v1
  python Models_Downloader.py --verify qwen3_omni_instruct_bf16
  python Models_Downloader.py --verify whisper:tiny
  python Models_Downloader.py --status --json
  python Models_Downloader.py --all
""")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--ensure", action="append", metavar="KEY", help="download/verify one catalog key; repeatable")
    mode.add_argument("--all", action="store_true", help="download and verify every catalog model")
    mode.add_argument("--list", action="store_true", help="list catalog keys, sizes, and local readiness")
    mode.add_argument("--verify", metavar="KEY", help="re-hash one local model without downloading")
    mode.add_argument("--status", action="store_true", help="show readiness for every catalog model")
    parser.add_argument("--json", action="store_true", help="emit JSON with --status")
    parser.add_argument("--target-root", metavar="DIR", help="models root; defaults to VCAP_MODELS_DIR or the app models folder")
    parser.add_argument("--repo", metavar="REPO_ID", help=f"override the Hugging Face repo (default: {DEFAULT_REPO_ID})")
    parser.add_argument("--subfolder", metavar="PATH", help="override one selected model's HF subfolder (testing)")
    parser.add_argument("--only", action="append", default=[], metavar="GLOB", help="download/verify only matching files (testing; repeatable)")
    parser.add_argument("--refresh-index", action="store_true", help="ignore the 24-hour remote listing cache")
    parser.add_argument("--connections", type=int, metavar="N", help="pin HTTP range connections (1-128)")
    parser.add_argument("--xet-concurrency", type=int, metavar="N", default=DOWNLOAD_CONFIG["xet_concurrency"], help=f"native Xet streams (1-124, default: {DOWNLOAD_CONFIG['xet_concurrency']})")
    parser.add_argument("--no-xet", action="store_true", help="disable optional hf_xet fast path")
    parser.add_argument("--max-bytes-for-test", type=int, metavar="N", help="test-only: interrupt after approximately N transferred bytes and keep .part state")
    return parser


def run_cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.json and not args.status:
        parser.error("--json requires --status")
    if args.connections is not None and not 1 <= args.connections <= 128:
        parser.error("--connections must be between 1 and 128")
    if not 1 <= args.xet_concurrency <= 124:
        parser.error("--xet-concurrency must be between 1 and 124")
    if args.max_bytes_for_test is not None and args.max_bytes_for_test <= 0:
        parser.error("--max-bytes-for-test must be greater than zero")

    target_root = _resolve_target_root(args.target_root)
    config: Dict[str, Any] = {"xet_concurrency": args.xet_concurrency}
    if args.connections is not None:
        config.update(num_connections=args.connections, http_adaptive_connections=0)
    if args.no_xet or args.max_bytes_for_test:
        config["xet_enabled"] = 0
    if args.max_bytes_for_test:
        config["max_bytes_for_test"] = args.max_bytes_for_test
    downloader = RobustDownloader(config)

    if args.list or args.status:
        if args.subfolder or args.only or args.max_bytes_for_test:
            parser.error("--subfolder, --only, and --max-bytes-for-test require one --ensure/--verify")
        statuses = collect_statuses(target_root, downloader, repo_override=args.repo, refresh_index=args.refresh_index, quiet=bool(args.json))
        whisper_statuses = collect_whisper_statuses(target_root)
        if args.status and args.json:
            all_statuses = [*statuses, *whisper_statuses]
            print(json.dumps({item["key"]: item for item in all_statuses}, indent=2, ensure_ascii=False))
        else:
            print_catalog(statuses, target_root)
            print_whisper_catalog(whisper_statuses, target_root)
        return 0

    verifying = False
    if args.ensure:
        selected = list(dict.fromkeys(args.ensure))
    elif args.all:
        selected = list(MODEL_CATALOG)
    elif args.verify:
        selected, verifying = [args.verify], True
    else:
        selected = interactive_selection(target_root, downloader, repo_override=args.repo, refresh_index=args.refresh_index)
        if not selected:
            print("No models selected.")
            return 0

    whisper_models = _app_whisper_modules()
    whisper_keys = {
        f"{WHISPER_KEY_PREFIX}{model.alias}" for model in whisper_models.WHISPER_MODELS
    }
    unknown = [key for key in selected if key not in MODEL_CATALOG and key not in whisper_keys]
    if unknown:
        parser.error(f"unknown model key(s): {', '.join(unknown)}")
    if args.subfolder and len(selected) != 1:
        parser.error("--subfolder requires exactly one --ensure/--verify key")
    if args.only and len(selected) != 1:
        parser.error("--only requires exactly one --ensure/--verify key")
    if args.max_bytes_for_test and (verifying or len(selected) != 1):
        parser.error("--max-bytes-for-test requires exactly one --ensure key")
    if any(key in whisper_keys for key in selected) and (args.repo or args.subfolder or args.only or args.max_bytes_for_test):
        parser.error("--repo, --subfolder, --only, and --max-bytes-for-test are not supported for whisper:<alias>")
    target_root.mkdir(parents=True, exist_ok=True)
    all_ok = True
    for key in selected:
        if key in whisper_keys:
            ok = verify_whisper_model(key, target_root) if verifying else ensure_whisper_model(key, target_root)
        else:
            kwargs = dict(repo_override=args.repo, subfolder_override=args.subfolder, only_patterns=args.only, refresh_index=args.refresh_index)
            ok = verify_model(key, target_root, downloader, **kwargs) if verifying else ensure_model(key, target_root, downloader, **kwargs)
        all_ok = all_ok and ok
    return 0 if all_ok else 1


def main() -> None:
    try:
        raise SystemExit(run_cli())
    except KeyboardInterrupt:
        print("\nDownload cancelled. Partial .part and .part.json files were preserved; run the same command again to resume.", flush=True)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
