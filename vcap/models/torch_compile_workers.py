"""Windows-safe TorchInductor worker-pool configuration and warmup."""

from __future__ import annotations

import os
import sys
import threading
import time
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from typing import Any, MutableMapping


MIN_COMPILE_THREADS = 1
MAX_COMPILE_THREADS = 32
DEFAULT_COMPILE_THREADS = 8
COMPILE_THREADS_ENV = "TORCHINDUCTOR_COMPILE_THREADS"
WORKER_START_ENV = "TORCHINDUCTOR_WORKER_START"
_SUPPORTED_START_METHODS = {"subprocess", "fork", "spawn"}
_CONFIG_LOCK = threading.Lock()


@dataclass(frozen=True)
class CompileWorkerSettings:
    threads: int
    worker_start: str


@dataclass(frozen=True)
class CompileWorkerWarmupHandle:
    threads: int
    worker_start: str
    started_at: float
    pool: Any = None
    futures: tuple[Any, ...] = ()
    error: str = ""


@dataclass(frozen=True)
class CompileWorkerWarmupResult:
    requested_threads: int
    active_workers: int
    worker_start: str
    elapsed_seconds: float
    ready: bool
    detail: str


def normalize_compile_threads(value: Any = None, *, env: MutableMapping[str, str] | None = None) -> int:
    """Resolve and clamp a requested parallel compilation worker count."""

    if value is None or str(value).strip() == "":
        source = os.environ if env is None else env
        value = source.get(COMPILE_THREADS_ENV, DEFAULT_COMPILE_THREADS)
    try:
        threads = int(round(float(value)))
    except (TypeError, ValueError):
        threads = DEFAULT_COMPILE_THREADS
    return max(MIN_COMPILE_THREADS, min(MAX_COMPILE_THREADS, threads))


def prepare_compile_worker_env(
    env: MutableMapping[str, str],
    value: Any = None,
    *,
    platform: str | None = None,
) -> CompileWorkerSettings:
    """Write platform-correct Inductor worker settings to an environment."""

    resolved_platform = sys.platform if platform is None else platform
    threads = normalize_compile_threads(value, env=env)
    if resolved_platform == "win32":
        start = "spawn"
    else:
        start = str(env.get(WORKER_START_ENV, "subprocess")).strip() or "subprocess"
        if start not in _SUPPORTED_START_METHODS:
            start = "subprocess"
    env[COMPILE_THREADS_ENV] = str(threads)
    env[WORKER_START_ENV] = start
    return CompileWorkerSettings(threads, start)


def configure_compile_workers(value: Any = None) -> CompileWorkerSettings:
    """Configure Inductor before its first asynchronous compilation."""

    with _CONFIG_LOCK:
        settings = prepare_compile_worker_env(os.environ, value)
        _apply_loaded_inductor_config(settings.threads, settings.worker_start)
    return settings


def _ready_probe(delay_seconds: float) -> int:
    if delay_seconds > 0:
        time.sleep(delay_seconds)
    return os.getpid()


def _async_runtime() -> tuple[Any, int]:
    from torch._inductor.async_compile import AsyncCompile, get_compile_threads

    return AsyncCompile, int(get_compile_threads())


def start_compile_worker_warmup(value: Any = None) -> CompileWorkerWarmupHandle:
    """Start all configured Inductor workers without blocking model loading."""

    settings = configure_compile_workers(value)
    started = time.perf_counter()
    if settings.threads <= 1:
        return CompileWorkerWarmupHandle(settings.threads, settings.worker_start, started)
    try:
        async_compile, configured = _async_runtime()
        if configured != settings.threads:
            return CompileWorkerWarmupHandle(
                settings.threads,
                settings.worker_start,
                started,
                error=f"requested {settings.threads} workers but Inductor reports {configured}",
            )
        pool = async_compile.process_pool()
        futures = tuple(pool.submit(_ready_probe, 0.2) for _ in range(settings.threads))
        async_compile.use_process_pool()
        return CompileWorkerWarmupHandle(settings.threads, settings.worker_start, started, pool, futures)
    except Exception as exc:
        return CompileWorkerWarmupHandle(
            settings.threads, settings.worker_start, started, error=f"{type(exc).__name__}: {exc}"
        )


def _result(
    handle: CompileWorkerWarmupHandle,
    active: int,
    ready: bool,
    detail: str,
) -> CompileWorkerWarmupResult:
    return CompileWorkerWarmupResult(
        handle.threads,
        active,
        handle.worker_start,
        max(0.0, time.perf_counter() - handle.started_at),
        ready,
        detail,
    )


def finish_compile_worker_warmup(
    handle: CompileWorkerWarmupHandle,
    *,
    timeout_seconds: float = 120.0,
) -> CompileWorkerWarmupResult:
    """Wait for worker startup and confirm that Inductor marked its pool ready."""

    if handle.error:
        return _result(handle, 0, False, handle.error)
    if handle.threads <= 1:
        return _result(handle, 1, True, "serial compilation")
    deadline = time.perf_counter() + max(0.1, float(timeout_seconds))
    pids: set[int] = set()
    try:
        for future in handle.futures:
            pids.add(int(future.result(timeout=max(0.0, deadline - time.perf_counter()))))
    except (FuturesTimeoutError, TimeoutError) as exc:
        return _result(handle, len(pids), False, f"worker startup timed out: {exc}")
    except Exception as exc:
        return _result(handle, len(pids), False, f"worker startup failed: {type(exc).__name__}: {exc}")
    active = len(pids)
    processes = getattr(handle.pool, "_processes", None)
    if isinstance(processes, dict):
        active = max(active, sum(1 for process in processes.values() if process and process.is_alive()))
    try:
        async_compile, _ = _async_runtime()
        while not async_compile.use_process_pool():
            if time.perf_counter() >= deadline:
                return _result(handle, active, False, "workers started but Inductor did not mark the pool ready")
            time.sleep(0.01)
    except Exception as exc:
        return _result(handle, active, False, f"pool readiness failed: {type(exc).__name__}: {exc}")
    return _result(handle, active, True, f"worker probes completed on {active} process(es)")


def _apply_loaded_inductor_config(threads: int, worker_start: str) -> None:
    try:
        import torch._inductor.config as config
    except (ImportError, AttributeError):
        return
    current_threads = int(getattr(config, "compile_threads", threads) or threads)
    current_start = str(getattr(config, "worker_start_method", worker_start))
    if current_threads != threads or current_start != worker_start:
        try:
            from torch._inductor.async_compile import AsyncCompile, shutdown_compile_workers

            shutdown_compile_workers()
            if AsyncCompile.pool.cache_info().currsize:
                AsyncCompile.pool().shutdown(wait=True)
                AsyncCompile.pool.cache_clear()
        except (ImportError, AttributeError):
            pass
    config.compile_threads = threads
    config.worker_start_method = worker_start


__all__ = [
    "COMPILE_THREADS_ENV",
    "CompileWorkerSettings",
    "CompileWorkerWarmupHandle",
    "CompileWorkerWarmupResult",
    "DEFAULT_COMPILE_THREADS",
    "WORKER_START_ENV",
    "configure_compile_workers",
    "finish_compile_worker_warmup",
    "normalize_compile_threads",
    "prepare_compile_worker_env",
    "start_compile_worker_warmup",
]
