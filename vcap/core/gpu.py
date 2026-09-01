"""GPU and host-memory inspection without importing Torch."""

from __future__ import annotations

import csv
import ctypes
import html
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from io import StringIO
from typing import Any

VRAM_TIERS = (6, 8, 10, 12, 16, 24, 32, 48, 80)
TIER_LABELS = {tier: f"{tier} GB" for tier in VRAM_TIERS}

_PDH_FMT_LARGE = 0x00000400
_PDH_MORE_DATA = 0x800007D2


class _PdhFormattedValueData(ctypes.Union):
    _fields_ = [
        ("longValue", ctypes.c_long),
        ("doubleValue", ctypes.c_double),
        ("largeValue", ctypes.c_longlong),
        ("ansiStringValue", ctypes.c_char_p),
        ("wideStringValue", ctypes.c_wchar_p),
    ]


class _PdhFormattedCounterValue(ctypes.Structure):
    _anonymous_ = ("value",)
    _fields_ = [("CStatus", ctypes.c_uint32), ("value", _PdhFormattedValueData)]


class _PdhFormattedCounterValueItemW(ctypes.Structure):
    _fields_ = [("szName", ctypes.c_wchar_p), ("FmtValue", _PdhFormattedCounterValue)]


@dataclass(frozen=True)
class GpuInfo:
    """Static and current-memory information for one NVIDIA GPU."""

    index: int
    name: str
    total_gb: float
    free_gb: float
    used_gb: float
    compute_capability: tuple[int, int] | None
    driver_version: str | None
    is_default: bool = False


def _decode_nvml(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _default_gpu_index() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        first = visible.split(",", 1)[0].strip()
        if first.isdigit():
            return int(first)
    return 0


def _list_gpus_nvml() -> list[GpuInfo]:
    try:
        import pynvml

        pynvml.nvmlInit()
    except Exception:
        return []

    result: list[GpuInfo] = []
    try:
        driver = _decode_nvml(pynvml.nvmlSystemGetDriverVersion())
        default_index = _default_gpu_index()
        for index in range(int(pynvml.nvmlDeviceGetCount())):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            compute: tuple[int, int] | None = None
            try:
                major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                compute = (int(major), int(minor))
            except Exception:
                pass
            gib = float(1024**3)
            result.append(
                GpuInfo(
                    index=index,
                    name=_decode_nvml(pynvml.nvmlDeviceGetName(handle)),
                    total_gb=float(memory.total) / gib,
                    free_gb=float(memory.free) / gib,
                    used_gb=float(memory.used) / gib,
                    compute_capability=compute,
                    driver_version=driver,
                    is_default=index == default_index,
                )
            )
    except Exception:
        return []
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    return result


def _run_nvidia_smi(fields: list[str]) -> str | None:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return None
    try:
        completed = subprocess.run(
            [
                executable,
                f"--query-gpu={','.join(fields)}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def _list_gpus_smi() -> list[GpuInfo]:
    field_sets = [
        ["index", "name", "memory.total", "memory.free", "compute_cap", "driver_version"],
        ["index", "name", "memory.total", "memory.free", "driver_version"],
    ]
    default_index = _default_gpu_index()
    for fields in field_sets:
        output = _run_nvidia_smi(fields)
        if not output:
            continue
        try:
            rows = list(csv.reader(StringIO(output)))
            parsed: list[GpuInfo] = []
            for row in rows:
                values = [column.strip() for column in row]
                index = int(values[0])
                total = float(values[2]) / 1024.0
                free = float(values[3]) / 1024.0
                cursor = 4
                compute = None
                if "compute_cap" in fields:
                    parts = values[cursor].split(".")
                    cursor += 1
                    if len(parts) >= 2:
                        compute = (int(parts[0]), int(parts[1]))
                driver = values[cursor] if cursor < len(values) else None
                parsed.append(
                    GpuInfo(
                        index=index,
                        name=values[1],
                        total_gb=total,
                        free_gb=free,
                        used_gb=max(0.0, total - free),
                        compute_capability=compute,
                        driver_version=driver,
                        is_default=index == default_index,
                    )
                )
            return parsed
        except (IndexError, TypeError, ValueError):
            continue
    return []


def list_gpus() -> list[GpuInfo]:
    """Enumerate NVIDIA GPUs via NVML, falling back to ``nvidia-smi``."""

    return _list_gpus_nvml() or _list_gpus_smi()


def vram_tier_for_gb(total_gb: float) -> int:
    """Map physical VRAM to the nearest supported capacity tier at or below it."""

    rounded = max(0, int(float(total_gb) + 0.5))
    eligible = [tier for tier in VRAM_TIERS if tier <= rounded]
    return eligible[-1] if eligible else VRAM_TIERS[0]


def _nvml_memory_for_index(gpu_index: int) -> tuple[float, float, float] | None:
    pynvml: Any | None = None
    try:
        import pynvml as imported_pynvml

        pynvml = imported_pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gib = float(1024**3)
        return memory.used / gib, memory.total / gib, memory.free / gib
    except Exception:
        return None
    finally:
        if pynvml is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


def _pdh_status_code(status: Any) -> int:
    """Normalize signed or ctypes PDH_STATUS values for constant comparisons."""

    return int(getattr(status, "value", status)) & 0xFFFFFFFF


def _configure_pdh_signatures(pdh: Any) -> None:
    """Set 64-bit-safe signatures on real ctypes functions while leaving mocks alone."""

    function_type = getattr(ctypes, "_CFuncPtr", None)
    if function_type is None:
        return
    signatures = {
        "PdhOpenQueryW": (
            [ctypes.c_wchar_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)],
            ctypes.c_long,
        ),
        "PdhAddEnglishCounterW": (
            [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)],
            ctypes.c_long,
        ),
        "PdhCollectQueryData": ([ctypes.c_void_p], ctypes.c_long),
        "PdhGetFormattedCounterArrayW": (
            [
                ctypes.c_void_p,
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.c_void_p,
            ],
            ctypes.c_long,
        ),
        "PdhCloseQuery": ([ctypes.c_void_p], ctypes.c_long),
    }
    for name, (argtypes, restype) in signatures.items():
        function = getattr(pdh, name, None)
        if isinstance(function, function_type):
            function.argtypes = argtypes
            function.restype = restype


def _pdh_counter_array_total(pdh: Any, counter: ctypes.c_void_p) -> int | None:
    buffer_size = ctypes.c_uint32(0)
    item_count = ctypes.c_uint32(0)
    status = pdh.PdhGetFormattedCounterArrayW(
        counter,
        _PDH_FMT_LARGE,
        ctypes.byref(buffer_size),
        ctypes.byref(item_count),
        None,
    )
    status_code = _pdh_status_code(status)
    if status_code == 0 and item_count.value == 0:
        return 0
    if status_code != _PDH_MORE_DATA or buffer_size.value <= 0:
        return None

    buffer = ctypes.create_string_buffer(buffer_size.value)
    status = pdh.PdhGetFormattedCounterArrayW(
        counter,
        _PDH_FMT_LARGE,
        ctypes.byref(buffer_size),
        ctypes.byref(item_count),
        buffer,
    )
    if _pdh_status_code(status) != 0:
        return None
    items = ctypes.cast(buffer, ctypes.POINTER(_PdhFormattedCounterValueItemW))
    total = 0
    for index in range(item_count.value):
        if items[index].FmtValue.CStatus in {0, 1}:
            total += max(0, int(items[index].FmtValue.largeValue))
    return total


def _pdh_memory_totals(instance_object: str) -> dict[str, float]:
    """Sum WDDM shared/dedicated usage counters for one PDH object instance filter."""

    if os.name != "nt":
        return {}

    query = ctypes.c_void_p()
    pdh: Any | None = None
    try:
        pdh = ctypes.windll.pdh
        _configure_pdh_signatures(pdh)
        if _pdh_status_code(pdh.PdhOpenQueryW(None, 0, ctypes.byref(query))) != 0:
            return {}

        counters: dict[str, ctypes.c_void_p] = {}
        for key, counter_name in (
            ("shared_gb", "Shared Usage"),
            ("dedicated_gb", "Dedicated Usage"),
        ):
            counter = ctypes.c_void_p()
            path = f"\\{instance_object}\\{counter_name}"
            status = pdh.PdhAddEnglishCounterW(query, path, 0, ctypes.byref(counter))
            if _pdh_status_code(status) != 0:
                return {}
            counters[key] = counter

        if _pdh_status_code(pdh.PdhCollectQueryData(query)) != 0:
            return {}
        totals = {key: _pdh_counter_array_total(pdh, counter) for key, counter in counters.items()}
        if any(value is None for value in totals.values()):
            return {}
        gib = float(2**30)
        return {key: float(value) / gib for key, value in totals.items() if value is not None}
    except Exception:
        return {}
    finally:
        if pdh is not None and query.value:
            try:
                pdh.PdhCloseQuery(query)
            except Exception:
                pass


def shared_gpu_memory_usage(pid: int | None = None) -> dict[str, float]:
    """Return one process's WDDM shared and dedicated GPU memory in GiB via PDH.

    Pinned (page-locked) host memory is counted as shared usage by WDDM, so a
    block-swapped model legitimately shows its pinned layers here; only the
    portion beyond the pinned bytes indicates paging of device allocations.
    """

    try:
        process_id = os.getpid() if pid is None else int(pid)
    except (TypeError, ValueError):
        return {}
    return _pdh_memory_totals(f"GPU Process Memory(pid_{process_id}_*)")


def shared_gpu_memory_total() -> dict[str, float]:
    """Return system-wide WDDM shared and dedicated GPU memory in GiB (all adapters)."""

    return _pdh_memory_totals("GPU Adapter Memory(*)")


def resource_snapshot(gpu_index: int = 0) -> dict[str, float | int]:
    """Capture current VRAM and system RAM use in GiB."""

    vram = _nvml_memory_for_index(int(gpu_index))
    if vram is None:
        match = next((gpu for gpu in list_gpus() if gpu.index == int(gpu_index)), None)
        vram = (
            (match.used_gb, match.total_gb, match.free_gb)
            if match is not None
            else (0.0, 0.0, 0.0)
        )
    try:
        import psutil

        memory = psutil.virtual_memory()
        gib = float(1024**3)
        ram_used, ram_total, ram_free = memory.used / gib, memory.total / gib, memory.available / gib
    except Exception:
        ram_used = ram_total = ram_free = 0.0
    snapshot: dict[str, float | int] = {
        "gpu_index": int(gpu_index),
        "vram_used_gb": float(vram[0]),
        "vram_total_gb": float(vram[1]),
        "vram_free_gb": float(vram[2]),
        "ram_used_gb": float(ram_used),
        "ram_total_gb": float(ram_total),
        "ram_free_gb": float(ram_free),
        "timestamp": time.time(),
    }
    shared = shared_gpu_memory_total()
    if "shared_gb" in shared:
        snapshot["shared_used_gb"] = float(shared["shared_gb"])
    return snapshot


def _meter_row(label: str, used: float, total: float, detail: str = "") -> str:
    percent = min(100.0, max(0.0, used / total * 100.0)) if total > 0 else 0.0
    safe_label = html.escape(label)
    safe_detail = html.escape(detail)
    return (
        '<div class="vc-meter__row">'
        f'<div class="vc-meter__label"><span>{safe_label}</span>'
        f'<span>{used:.1f} / {total:.1f} GB{safe_detail}</span></div>'
        '<div class="vc-meter__track" role="progressbar" '
        f'aria-label="{safe_label}" aria-valuemin="0" aria-valuemax="100" '
        f'aria-valuenow="{percent:.1f}"><span class="vc-meter__fill" '
        f'style="width:{percent:.1f}%"></span></div></div>'
    )


def render_resource_meter_html(
    snapshot: dict[str, Any], peak_vram_gb: float | None = None
) -> str:
    """Render compact, theme-agnostic VRAM and RAM meter markup."""

    vram_used = float(snapshot.get("vram_used_gb", 0.0) or 0.0)
    vram_total = float(snapshot.get("vram_total_gb", 0.0) or 0.0)
    ram_used = float(snapshot.get("ram_used_gb", 0.0) or 0.0)
    ram_total = float(snapshot.get("ram_total_gb", 0.0) or 0.0)
    peak = f"; peak {float(peak_vram_gb):.1f} GB" if peak_vram_gb is not None else ""
    rows = (
        '<div class="vc-meter">'
        + _meter_row("VRAM", vram_used, vram_total, peak)
        + _meter_row("RAM", ram_used, ram_total)
    )
    if "shared_used_gb" in snapshot:
        shared_used = float(snapshot.get("shared_used_gb", 0.0) or 0.0)
        shared_total = float(snapshot.get("shared_total_gb", ram_total / 2.0) or 0.0)
        rows += _meter_row("Shared GPU memory (WDDM, all processes)", shared_used, shared_total)
    return rows + "</div>"


def is_oom_error(exc_or_text: object) -> bool:
    """Recognize common Torch, CUDA, cuBLAS, and accelerator OOM signatures."""

    if exc_or_text is None:
        return False
    class_name = type(exc_or_text).__name__.casefold()
    text = f"{class_name}: {exc_or_text}".casefold()
    strong = (
        "outofmemoryerror",
        "out of memory",
        "cudaerrormemoryallocation",
        "cublas_status_alloc_failed",
        "cublasstatusallocfailed",
        "cudnn_status_alloc_failed",
        "cuda error: memory allocation",
    )
    if any(pattern in text for pattern in strong):
        return True
    generic = "failed to allocate" in text or "ran out of memory" in text
    gpu_context = any(token in text for token in ("cuda", "gpu", "vram", "cublas", "device"))
    if generic and gpu_context:
        return True
    accelerator = "acceleratorerror" in text and (
        "error code 2" in text or "code: 2" in text or "code=2" in text
    )
    return accelerator and ("cuda" in text or "memory" in text)


def cuda_visible_devices_env(index: int | str) -> dict[str, str]:
    """Return an environment overlay selecting exactly one CUDA device."""

    return {"CUDA_VISIBLE_DEVICES": str(index).strip()}


def vram_cap_env_disabled() -> bool:
    """Return whether VCAP_VRAM_HARD_CAP explicitly disables the allocator cap."""

    return os.environ.get("VCAP_VRAM_HARD_CAP", "").strip().casefold() in {"0", "false", "off"}


__all__ = [
    "GpuInfo",
    "TIER_LABELS",
    "cuda_visible_devices_env",
    "is_oom_error",
    "list_gpus",
    "render_resource_meter_html",
    "resource_snapshot",
    "shared_gpu_memory_total",
    "shared_gpu_memory_usage",
    "vram_cap_env_disabled",
    "vram_tier_for_gb",
]
