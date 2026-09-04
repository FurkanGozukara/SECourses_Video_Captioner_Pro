"""Robust Torch compile setup with toolchain discovery and graceful fallbacks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile
import threading
from typing import Any, Callable, Literal

from vcap import TEMP_DIR
from vcap.core.logs import get_log

from .torch_compile_workers import configure_compile_workers


CompileReadiness = Literal["full", "triton_only", "cudagraphs_only", "unavailable"]
CompileMode = Literal["full", "triton_only", "cudagraphs", "eager"]


@dataclass(frozen=True)
class CompileModeOption:
    """One user-facing, generation-safe Inductor tuning mode."""

    value: str
    label: str
    description: str


DEFAULT_COMPILE_MODE = "default"
COMPILE_MODE_OPTIONS: tuple[CompileModeOption, ...] = (
    CompileModeOption(
        "default",
        "Inductor default (recommended)",
        "Balanced compilation without explicitly enabling CUDA graph replay.",
    ),
    CompileModeOption(
        "max-autotune-no-cudagraphs",
        "Max autotune (no CUDA graphs)",
        "Longer first-run tuning while avoiding DynamicCache-unsafe CUDA graph replay.",
    ),
)
_COMPILE_MODE_ALIASES = {
    "full": "default",
    "inductor": "default",
    "max_autotune_no_cudagraphs": "max-autotune-no-cudagraphs",
    "reduce_overhead": "reduce-overhead",
    "cuda_graphs": "cudagraphs",
    "cuda-graphs": "cudagraphs",
}
_INTERNAL_COMPILE_MODES = {
    *(option.value for option in COMPILE_MODE_OPTIONS),
    "cudagraphs",
    "reduce-overhead",
}

_CACHE_DIR = Path(__file__).resolve().parent
_VCVARS_CACHE = _CACHE_DIR / ".vcvars_cache.json"
_PROBE_CACHE = _CACHE_DIR / ".compile_env_cache.json"
_CACHE_LOCK = threading.RLock()
_PROBE_MEMORY: dict[str, "CompileEnvReport"] = {}
_DISABLED_COMPILE_MODES: dict[tuple[str, str], str] = {}


@dataclass(frozen=True)
class CompileEnvReport:
    """C++/Triton discovery result used by the compile-mode decision ladder."""

    os: str
    cl_on_path: bool
    vswhere_path: str | None
    vs_install_path: str | None
    vcvars_bat: str | None
    msvc_version: str | None
    gcc_version: str | None
    triton_ok: bool
    triton_windows: bool
    inductor_ready: CompileReadiness
    messages: list[str]
    triton_version: str | None = None
    cl_path: str | None = None


@dataclass(frozen=True)
class CompilePlan:
    """Executable compile choice plus environment overlay and fallback policy."""

    mode: CompileMode
    env_updates: dict[str, str]
    torch_compile_kwargs: dict[str, Any]
    warnings: list[str]
    fallback_modes: tuple[str, ...] = field(default_factory=tuple)
    requested_mode: str = "eager"

    @property
    def enabled(self) -> bool:
        return self.mode != "eager"


@dataclass(frozen=True)
class CompileFallback:
    """Details of a compiled model restored in place for an eager retry."""

    family: str
    mode: str
    restored_modules: int
    reason: str


def compile_mode_choices() -> list[tuple[str, str]]:
    """Return stable Gradio label/value pairs for validated compile modes."""

    return [(option.label, option.value) for option in COMPILE_MODE_OPTIONS]


def compile_mode_values() -> tuple[str, ...]:
    """Return the user-facing compile-mode registry values."""

    return tuple(option.value for option in COMPILE_MODE_OPTIONS)


def normalize_compile_mode(mode: object) -> str:
    """Normalize current and legacy compile-mode setting values."""

    value = str(mode or DEFAULT_COMPILE_MODE).strip().casefold()
    value = _COMPILE_MODE_ALIASES.get(value, value)
    return value if value in _INTERNAL_COMPILE_MODES else DEFAULT_COMPILE_MODE


def _compile_key(family: object, mode: object) -> tuple[str, str]:
    return (
        str(family or "unknown").strip().casefold() or "unknown",
        normalize_compile_mode(mode),
    )


def _disable_compile_mode(family: object, mode: object, reason: str) -> None:
    with _CACHE_LOCK:
        _DISABLED_COMPILE_MODES[_compile_key(family, mode)] = str(reason)


def disabled_compile_modes() -> dict[tuple[str, str], str]:
    """Return process-local family/mode failures that now force eager loading."""

    with _CACHE_LOCK:
        return dict(_DISABLED_COMPILE_MODES)


def _json_read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}


def _json_write(path: Path, value: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".partial")
        partial.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(partial, path)
    except OSError:
        pass


def _mtime(path: str | Path | None) -> int:
    if not path:
        return -1
    try:
        return int(Path(path).stat().st_mtime_ns)
    except OSError:
        return -1


def _which(name: str, env: dict[str, str] | None = None) -> str | None:
    return shutil.which(name, path=(env or os.environ).get("PATH", ""))


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _resolve_vswhere() -> str | None:
    candidate = _which("vswhere.exe") or _which("vswhere")
    if candidate and Path(candidate).is_file():
        return str(Path(candidate).resolve(strict=False))
    roots = (
        os.environ.get("ProgramFiles(x86)"),
        os.environ.get("ProgramFiles"),
        r"C:\Program Files (x86)",
        r"C:\Program Files",
    )
    for root in roots:
        if not root:
            continue
        path = Path(root) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
        if path.is_file():
            return str(path.resolve(strict=False))
    if os.name == "nt":
        try:
            import winreg

            for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
                try:
                    with winreg.OpenKey(hive, r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\vswhere.exe") as key:
                        value, _ = winreg.QueryValueEx(key, None)
                    if value and Path(value).is_file():
                        return str(Path(value).resolve(strict=False))
                except OSError:
                    continue
        except ImportError:
            pass
    return None


def _query_vs(vswhere: str) -> str | None:
    try:
        completed = subprocess.run(
            [
                vswhere,
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return next((line.strip() for line in completed.stdout.splitlines() if line.strip()), None)


def _default_vs_installs() -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for root in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles"), r"C:\Program Files (x86)", r"C:\Program Files"):
        base = Path(root or "") / "Microsoft Visual Studio"
        if not base.is_dir():
            continue
        try:
            versions = sorted((item for item in base.iterdir() if item.is_dir()), reverse=True)
        except OSError:
            continue
        for version in versions:
            try:
                editions = sorted((item for item in version.iterdir() if item.is_dir()), reverse=True)
            except OSError:
                continue
            for edition in editions:
                key = os.path.normcase(str(edition))
                if key not in seen:
                    seen.add(key)
                    result.append(edition)
    return result


def _find_vcvars(installation: str | None) -> tuple[str | None, str | None]:
    candidates: list[Path] = []
    env_install = os.environ.get("VSINSTALLDIR")
    if env_install:
        candidates.append(Path(env_install))
    if installation:
        candidates.append(Path(installation))
    candidates.extend(_default_vs_installs())
    seen: set[str] = set()
    for install in candidates:
        key = os.path.normcase(str(install.resolve(strict=False)))
        if key in seen:
            continue
        seen.add(key)
        for relative in (
            Path("VC/Auxiliary/Build/vcvars64.bat"),
            Path("VC/Auxiliary/Build/vcvarsall.bat"),
            Path("Common7/Tools/VsDevCmd.bat"),
        ):
            batch = install / relative
            if batch.is_file():
                return str(install.resolve(strict=False)), str(batch.resolve(strict=False))
    return installation, None


def _parse_env_block(stdout: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key and not key.startswith("=") and "\x00" not in key:
            result[key] = value
    return result


def capture_vcvars_env(vcvars_bat: str | os.PathLike[str]) -> dict[str, str]:
    """Capture and cache the environment delta produced by ``vcvars64.bat``."""

    batch = Path(vcvars_bat).expanduser().resolve(strict=True)
    cache_key = f"{os.path.normcase(str(batch))}|{_mtime(batch)}"
    with _CACHE_LOCK:
        cached = _json_read(_VCVARS_CACHE)
        entry = cached.get("entries", {}).get(cache_key)
        if isinstance(entry, dict) and isinstance(entry.get("env"), dict):
            return {str(key): str(value) for key, value in entry["env"].items()}

    name = batch.name.lower()
    if name == "vcvarsall.bat":
        invocation = ["cmd.exe", "/d", "/s", "/c", "call", str(batch), "amd64", "&&", "set"]
    elif name == "vsdevcmd.bat":
        invocation = [
            "cmd.exe", "/d", "/s", "/c", "call", str(batch),
            "-arch=amd64", "-host_arch=amd64", "&&", "set",
        ]
    else:
        invocation = ["cmd.exe", "/d", "/s", "/c", "call", str(batch), "&&", "set"]
    try:
        completed = subprocess.run(
            invocation,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"vcvars activation failed: {exc}") from exc
    if completed.returncode != 0:
        tail = "\n".join((completed.stdout + "\n" + completed.stderr).splitlines()[-30:])
        raise RuntimeError(f"{batch.name} exited with code {completed.returncode}: {tail}")
    captured = _parse_env_block(completed.stdout)
    base = dict(os.environ)
    delta = {key: value for key, value in captured.items() if base.get(key) != value}
    if not _which("cl.exe", {**base, **delta}):
        raise RuntimeError("cl.exe was not found after vcvars activation; install the MSVC C++ workload")
    with _CACHE_LOCK:
        payload = _json_read(_VCVARS_CACHE)
        entries = payload.setdefault("entries", {})
        entries.clear()
        entries[cache_key] = {"path": str(batch), "mtime_ns": _mtime(batch), "env": delta}
        payload["format_version"] = "1.0"
        _json_write(_VCVARS_CACHE, payload)
    return delta


def _command_version(command: list[str], env: dict[str, str] | None = None) -> str | None:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            env=env,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = (completed.stdout + "\n" + completed.stderr).strip()
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    compiler_line = next((line for line in lines if "compiler version" in line.lower()), None)
    return compiler_line or next(iter(lines), None)


def _probe_triton() -> tuple[bool, str | None, str | None]:
    if importlib.util.find_spec("triton") is None:
        return False, None, "Triton is not installed"
    version = _package_version("triton") or _package_version("triton-windows") or "unknown"
    return True, str(version), None


def _probe_torch_compile() -> tuple[bool, str | None]:
    """Detect the torch.compile API from package metadata without importing Torch."""

    if importlib.util.find_spec("torch") is None:
        return False, "PyTorch is not installed"
    version = _package_version("torch")
    if version:
        match = re.match(r"\s*(\d+)(?:\.(\d+))?", version)
        if match and (int(match.group(1)), int(match.group(2) or 0)) < (2, 0):
            return False, f"PyTorch {version} predates torch.compile"
    return True, None


def _report_key(cl: str | None, vswhere: str | None, vcvars: str | None) -> str:
    data = {
        "platform": platform.system(),
        "force_no_msvc": os.environ.get("VCAP_FORCE_NO_MSVC", ""),
        "cl": [cl, _mtime(cl)],
        "vswhere": [vswhere, _mtime(vswhere)],
        "vcvars": [vcvars, _mtime(vcvars)],
        "path": os.environ.get("PATH", ""),
        "torch_version": _package_version("torch"),
        "triton_version": _package_version("triton") or _package_version("triton-windows"),
    }
    return json.dumps(data, sort_keys=True)


def probe_compile_environment(force: bool = False) -> CompileEnvReport:
    """Inspect MSVC/GCC, Triton, and the strongest usable compile mode."""

    system = platform.system()
    windows = os.name == "nt"
    forced_no_msvc = windows and os.environ.get("VCAP_FORCE_NO_MSVC", "").strip().lower() in {"1", "true", "yes", "on"}
    initial_cl = None if forced_no_msvc else _which("cl.exe") if windows else None
    vswhere = _resolve_vswhere() if windows else None
    installation = _query_vs(vswhere) if vswhere and not forced_no_msvc else None
    installation, vcvars = _find_vcvars(installation) if windows and not forced_no_msvc else (installation, None)
    key = _report_key(initial_cl, vswhere, vcvars)
    with _CACHE_LOCK:
        if not force and key in _PROBE_MEMORY:
            return _PROBE_MEMORY[key]
        if not force:
            disk = _json_read(_PROBE_CACHE)
            if disk.get("key") == key and isinstance(disk.get("report"), dict):
                try:
                    report = CompileEnvReport(**disk["report"])
                    _PROBE_MEMORY[key] = report
                    return report
                except (TypeError, ValueError):
                    pass

    messages: list[str] = []
    compile_env = dict(os.environ)
    cl_path = initial_cl
    if forced_no_msvc:
        messages.append("VCAP_FORCE_NO_MSVC=1: MSVC discovery deliberately disabled")
    elif windows and not cl_path and vcvars:
        try:
            compile_env.update(capture_vcvars_env(vcvars))
            cl_path = _which("cl.exe", compile_env)
            if cl_path:
                messages.append(f"MSVC environment available through {Path(vcvars).name}")
        except RuntimeError as exc:
            messages.append(str(exc))
    msvc_version = _command_version([cl_path], compile_env) if cl_path else None

    gcc_path = None
    gcc_version = None
    if not windows:
        gcc_path = _which("g++") or _which("gcc") or _which("c++")
        if gcc_path:
            gcc_version = _command_version([gcc_path, "--version"])
        else:
            messages.append("gcc/g++ was not found on PATH")
    elif not cl_path:
        messages.append("MSVC C++ build tools were not found")

    triton_ok, triton_version, triton_error = _probe_triton()
    if triton_error:
        messages.append(triton_error)
    torch_compile, torch_error = _probe_torch_compile()
    if torch_error:
        messages.append(torch_error)

    compiler_ok = bool(cl_path if windows else gcc_path)
    if torch_compile and triton_ok and compiler_ok:
        readiness: CompileReadiness = "full"
    elif torch_compile and triton_ok:
        readiness = "triton_only"
    elif torch_compile:
        readiness = "cudagraphs_only"
    else:
        readiness = "unavailable"
    report = CompileEnvReport(
        os=system,
        cl_on_path=bool(initial_cl),
        vswhere_path=vswhere,
        vs_install_path=installation,
        vcvars_bat=vcvars,
        msvc_version=msvc_version,
        gcc_version=gcc_version,
        triton_ok=triton_ok,
        triton_windows=bool(windows and _package_version("triton-windows")),
        inductor_ready=readiness,
        messages=messages,
        triton_version=triton_version,
        cl_path=cl_path,
    )
    with _CACHE_LOCK:
        _PROBE_MEMORY[key] = report
        _json_write(_PROBE_CACHE, {"format_version": "1.0", "key": key, "report": asdict(report)})
    return report


def prepare_compile_env(
    enable: bool,
    *,
    mode: str = DEFAULT_COMPILE_MODE,
    compile_threads: int = 8,
    family: str | None = None,
) -> CompilePlan:
    """Choose full Inductor, Triton-only, CUDA graphs, or eager execution."""

    raw_mode = str(mode or DEFAULT_COMPILE_MODE).strip().casefold()
    requested_mode = normalize_compile_mode(raw_mode)
    if not enable:
        return CompilePlan(
            "eager",
            {},
            {},
            ["torch.compile disabled"],
            requested_mode=requested_mode,
        )
    disabled_reason = None
    if family:
        with _CACHE_LOCK:
            disabled_reason = _DISABLED_COMPILE_MODES.get(_compile_key(family, requested_mode))
    if disabled_reason:
        return CompilePlan(
            "eager",
            {},
            {},
            [
                f"torch.compile mode '{requested_mode}' is disabled for {family} in this process "
                f"after a runtime failure ({disabled_reason}); using eager execution"
            ],
            requested_mode=requested_mode,
        )
    report = probe_compile_environment()
    messages = list(report.messages)
    normalized_raw = _COMPILE_MODE_ALIASES.get(raw_mode, raw_mode)
    if normalized_raw not in _INTERNAL_COMPILE_MODES:
        messages.insert(
            0,
            f"Unknown torch.compile mode '{raw_mode}'; using '{DEFAULT_COMPILE_MODE}'",
        )
    cache_dir = (TEMP_DIR / "torchinductor").resolve(strict=False)
    env_updates = {
        "TORCHINDUCTOR_CACHE_DIR": str(cache_dir),
        "TORCHINDUCTOR_COMPILE_THREADS": str(max(1, min(32, int(compile_threads)))),
    }
    if os.name == "nt":
        env_updates["TORCHINDUCTOR_WORKER_START"] = "spawn"
    if report.vcvars_bat and report.cl_path and not report.cl_on_path:
        try:
            env_updates.update(capture_vcvars_env(report.vcvars_bat))
        except RuntimeError as exc:
            messages.append(str(exc))

    if requested_mode == "cudagraphs" and report.inductor_ready != "unavailable":
        return CompilePlan(
            "cudagraphs",
            env_updates,
            {"backend": "cudagraphs", "dynamic": None},
            [
                "Legacy CUDA graphs mode selected; DynamicCache decoding may require the "
                "automatic segment-level eager retry",
                *messages,
            ],
            ("eager",),
            requested_mode,
        )
    if report.inductor_ready == "full":
        return CompilePlan(
            "full",
            env_updates,
            {"backend": "inductor", "mode": requested_mode, "dynamic": None},
            messages,
            ("cudagraphs", "eager"),
            requested_mode,
        )
    if report.inductor_ready == "triton_only":
        warning = "C++ build tools not found — using Triton-only Inductor fallback"
        env_updates.update(
            {
                "TORCHINDUCTOR_CPP_WRAPPER": "0",
                "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS": "TRITON",
                "TORCHINDUCTOR_MAX_AUTOTUNE_CONV_BACKENDS": "TRITON",
            }
        )
        return CompilePlan(
            "triton_only",
            env_updates,
            {"backend": "inductor", "mode": "default", "dynamic": None},
            [warning, *messages],
            ("cudagraphs", "eager"),
            requested_mode,
        )
    if report.inductor_ready == "cudagraphs_only":
        return CompilePlan(
            "cudagraphs",
            env_updates,
            {"backend": "cudagraphs", "dynamic": None},
            ["Inductor is unavailable; using the no-codegen CUDA graphs backend", *messages],
            ("eager",),
            requested_mode,
        )
    why = "; ".join(messages) or "PyTorch does not expose torch.compile"
    return CompilePlan(
        "eager",
        {},
        {},
        [f"torch.compile unavailable: {why}; install VS Build Tools / gcc for the full path"],
        requested_mode=requested_mode,
    )


def _emit(progress_cb: Callable[..., None] | None, message: str) -> None:
    get_log().log(message, scope="compile")
    if progress_cb:
        try:
            progress_cb(message)
        except TypeError:
            try:
                progress_cb({"message": message})
            except TypeError:
                pass


def _offloaded(model: Any) -> bool:
    if bool(getattr(model, "_vcap_block_swap", False)):
        return True
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        devices = {str(value) for value in device_map.values()}
        return any(value in {"cpu", "disk"} for value in devices)
    return False


def _language_target(model: Any) -> Any:
    thinker = getattr(model, "thinker", None)
    if thinker is not None:
        model = thinker
    return getattr(model, "model", model)


def _toolchain_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    markers = (
        "cl.exe", "msvc", "gcc", "g++", "c++ compiler", "cppcompileerror",
        "invalidcompiler", "compiler not found", "failed to compile", "ninja",
        "vcvars", "link.exe", "torchinductor",
    )
    return any(marker in text for marker in markers)


def _numerics_guards(torch: Any) -> None:
    try:
        torch._inductor.config.emulate_precision_casts = True
        torch._inductor.config.eager_numerics.division_rounding = True
        torch._inductor.config.eager_numerics.use_pytorch_libdevice = True
        torch._inductor.config.fx_graph_cache = True
    except (AttributeError, RuntimeError):
        pass


def _exception_chain(exc: BaseException) -> list[BaseException]:
    pending = [exc]
    result: list[BaseException] = []
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        result.append(current)
        cause = getattr(current, "__cause__", None)
        context = getattr(current, "__context__", None)
        nested = getattr(current, "exceptions", None)
        if isinstance(cause, BaseException):
            pending.append(cause)
        if isinstance(context, BaseException):
            pending.append(context)
        if isinstance(nested, (list, tuple)):
            pending.extend(item for item in nested if isinstance(item, BaseException))
    return result


def is_compile_runtime_error(exc: BaseException) -> bool:
    """Return whether an exception originated in Dynamo, Inductor, or CUDA graphs."""

    markers = (
        "torch.compile",
        "torch._dynamo",
        "torchinductor",
        "inductor",
        "backendcompilerfailed",
        "cudagraph",
        "cuda graph",
        "overwritten by a subsequent run",
        "stream is capturing",
        "capture_status",
    )
    compile_types = {
        "BackendCompilerFailed",
        "CppCompileError",
        "InductorError",
        "InvalidCompiler",
        "TorchRuntimeError",
        "Unsupported",
    }
    for current in _exception_chain(exc):
        error_type = type(current)
        module = str(getattr(error_type, "__module__", "")).casefold()
        name = str(getattr(error_type, "__name__", ""))
        text = f"{name}: {current}".casefold()
        if module.startswith(("torch._dynamo", "torch._inductor")):
            return True
        if name in compile_types and module.startswith("torch"):
            return True
        if any(marker in text for marker in markers) or _toolchain_error(current):
            return True
    return False


def _modules(model: Any) -> list[Any]:
    modules = getattr(model, "modules", None)
    if callable(modules):
        try:
            values = list(modules())
            if values:
                return values
        except Exception:
            pass
    return [model]


def _delete_instance_attrs(value: Any, names: tuple[str, ...]) -> None:
    namespace = getattr(value, "__dict__", {})
    for name in names:
        if name not in namespace:
            continue
        try:
            delattr(value, name)
        except (AttributeError, TypeError):
            pass


def _reset_compiler_runtime() -> None:
    try:
        import torch

        compiler_reset = getattr(getattr(torch, "compiler", None), "reset", None)
        dynamo_reset = getattr(getattr(torch, "_dynamo", None), "reset", None)
        reset = compiler_reset if callable(compiler_reset) else dynamo_reset
        if callable(reset):
            reset()
    except Exception:
        pass


def release_compiled_model(model: Any) -> int:
    """Restore compiled forwards and discard per-model compiler state."""

    try:
        modules = _modules(model)
    except Exception:
        modules = [model]
    if not any(module is model for module in modules):
        modules.insert(0, model)

    restored = 0
    seen: set[int] = set()
    attributes = (
        "_vcap_compiled",
        "_vcap_original_forward",
        "_vcap_original_disabled_forward",
        "_vcap_compile_plan",
        "_vcap_compile_family",
        "_vcap_compile_requested_mode",
        "_vcap_compile_disabled",
    )
    for module in modules:
        identity = id(module)
        if identity in seen:
            continue
        seen.add(identity)
        restored_module = False
        try:
            original = getattr(module, "_vcap_original_forward", None)
        except Exception:
            original = None
        if callable(original):
            try:
                module.forward = original
                restored_module = True
            except Exception:
                pass
        try:
            disabled_original = getattr(
                module, "_vcap_original_disabled_forward", None
            )
        except Exception:
            disabled_original = None
        if callable(disabled_original):
            try:
                module.forward = disabled_original
                restored_module = True
            except Exception:
                pass
        try:
            _delete_instance_attrs(module, attributes)
        except Exception:
            pass
        if restored_module:
            restored += 1

    if restored:
        try:
            _reset_compiler_runtime()
        except Exception:
            pass
    return restored


def restore_eager_model(loaded_or_model: Any, exc: BaseException) -> CompileFallback | None:
    """Restore compiled forwards in place, disable the failed mode, and keep weights loaded."""

    is_loaded = all(
        hasattr(loaded_or_model, name) for name in ("model", "spec", "load_report")
    )
    loaded = loaded_or_model if is_loaded else None
    model = loaded_or_model.model if is_loaded else loaded_or_model
    modules = _modules(model)
    plan = getattr(model, "_vcap_compile_plan", None)
    if plan is None:
        plan = next(
            (getattr(module, "_vcap_compile_plan", None) for module in modules if getattr(module, "_vcap_compiled", False)),
            None,
        )
    active = any(
        bool(getattr(module, "_vcap_compiled", False))
        and callable(getattr(module, "_vcap_original_forward", None))
        for module in modules
    )
    if not active:
        return None

    family = str(
        getattr(getattr(loaded, "spec", None), "family", None)
        or getattr(model, "_vcap_compile_family", None)
        or "unknown"
    )
    mode = normalize_compile_mode(
        getattr(plan, "requested_mode", None)
        or getattr(model, "_vcap_compile_requested_mode", None)
        or getattr(plan, "mode", None)
    )
    reason = " ".join(f"{type(exc).__name__}: {exc}".split())[:1200]
    restored = 0
    for module in modules:
        original = getattr(module, "_vcap_original_forward", None)
        if callable(original):
            module.forward = original
            restored += 1
        disabled_original = getattr(module, "_vcap_original_disabled_forward", None)
        if callable(disabled_original):
            module.forward = disabled_original
        _delete_instance_attrs(
            module,
            (
                "_vcap_compiled",
                "_vcap_original_forward",
                "_vcap_original_disabled_forward",
                "_vcap_compile_plan",
                "_vcap_compile_family",
                "_vcap_compile_requested_mode",
            ),
        )
    _delete_instance_attrs(
        model,
        (
            "_vcap_compile_plan",
            "_vcap_compile_family",
            "_vcap_compile_requested_mode",
        ),
    )
    setattr(model, "_vcap_compile_disabled", True)
    _disable_compile_mode(family, mode, reason)
    if loaded is not None:
        setattr(loaded, "_vcap_compile_disabled", True)
        report = getattr(loaded, "load_report", None)
        try:
            loaded.load_report = replace(report, compile_mode="eager")
        except (TypeError, ValueError, AttributeError):
            try:
                report.compile_mode = "eager"
            except (AttributeError, TypeError):
                pass
    _reset_compiler_runtime()
    return CompileFallback(family, mode, restored, reason)


_RECOMPILE_LIMIT = 64


def _raise_recompile_limits(progress_cb: Any = None) -> None:
    """Let Dynamo keep specializing the decode loop instead of falling back to eager.

    Autoregressive decoding changes sequence and cache shapes every step; with the
    stock limit of 8 recompiles Dynamo stops compiling the frame after a few
    tokens and the rest of the run silently executes eagerly.
    """

    try:
        import torch  # local import keeps the UI process free of torch
    except Exception:
        return
    config = getattr(getattr(torch, "_dynamo", None), "config", None)
    if config is None:
        return
    for name in ("cache_size_limit", "recompile_limit", "accumulated_cache_size_limit", "accumulated_recompile_limit"):
        try:
            current = int(getattr(config, name))
        except (AttributeError, TypeError, ValueError):
            continue
        if current < _RECOMPILE_LIMIT:
            try:
                setattr(config, name, _RECOMPILE_LIMIT)
            except Exception:
                continue
    _emit(progress_cb, f"torch.compile recompile limit set to {_RECOMPILE_LIMIT} for the decode loop")


def apply_compile(
    model: Any,
    plan: CompilePlan,
    scope: str = "language_model",
    *,
    progress_cb: Callable[..., None] | None = None,
    family: str | None = None,
) -> Any:
    """Compile the resident text decoder while retaining automatic fallbacks."""

    if not plan.enabled:
        for warning in plan.warnings:
            _emit(progress_cb, warning)
        return model
    if _offloaded(model):
        _emit(progress_cb, "torch.compile skipped: CPU-offloaded or block-swapped decoder layers are not compile-safe")
        return model
    import torch

    os.environ.update(plan.env_updates)
    Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    configure_compile_workers(os.environ.get("TORCHINDUCTOR_COMPILE_THREADS", "8"))
    _numerics_guards(torch)
    if plan.mode == "triton_only":
        try:
            torch._inductor.config.cpp_wrapper = False
            torch._inductor.config.max_autotune_gemm_backends = "TRITON"
            torch._inductor.config.max_autotune_conv_backends = "TRITON"
        except (AttributeError, RuntimeError):
            pass
    for warning in plan.warnings:
        _emit(progress_cb, warning)
    _emit(
        progress_cb,
        "capturing CUDA graphs on the first generation"
        if plan.mode == "cudagraphs"
        else "compiling - first run is slow (1-5 min)",
    )

    target = _language_target(model) if scope == "language_model" else model
    if getattr(target, "_vcap_compiled", False):
        return model
    for module in _modules(target):
        if module.__class__.__name__.startswith("ConvRot"):
            try:
                module._vcap_original_disabled_forward = module.forward
                module.forward = torch.compiler.disable(module.forward)
            except (AttributeError, TypeError):
                pass

    original_forward = target.forward
    _raise_recompile_limits(progress_cb)
    try:
        compiled_forward = torch.compile(original_forward, **plan.torch_compile_kwargs)
    except Exception as exc:
        mode = plan.requested_mode or plan.mode
        compile_family = family or "unknown"
        _disable_compile_mode(compile_family, mode, f"{type(exc).__name__}: {exc}")
        for module in _modules(target):
            disabled_original = getattr(module, "_vcap_original_disabled_forward", None)
            if callable(disabled_original):
                module.forward = disabled_original
            _delete_instance_attrs(module, ("_vcap_original_disabled_forward",))
        _emit(
            progress_cb,
            f"torch.compile mode '{mode}' failed during setup ({type(exc).__name__}: {exc}); "
            "using eager execution",
        )
        return model

    def guarded_forward(*args: Any, **kwargs: Any) -> Any:
        if plan.mode == "cudagraphs" or plan.requested_mode == "reduce-overhead":
            mark_step = getattr(getattr(torch, "compiler", None), "cudagraph_mark_step_begin", None)
            if callable(mark_step):
                mark_step()
        return compiled_forward(*args, **kwargs)

    target.forward = guarded_forward
    target._vcap_compiled = True
    target._vcap_original_forward = original_forward
    target._vcap_compile_plan = plan
    target._vcap_compile_family = family or "unknown"
    target._vcap_compile_requested_mode = plan.requested_mode
    model._vcap_compile_plan = plan
    model._vcap_compile_family = family or "unknown"
    model._vcap_compile_requested_mode = plan.requested_mode
    return model


def clear_inductor_caches() -> dict[str, Any]:
    """Remove app-owned Triton/Inductor caches and cached vcvars discovery."""

    username = os.environ.get("USERNAME") or os.environ.get("USER") or "user"
    temp_root = Path(tempfile.gettempdir())
    candidates = {
        Path(os.environ.get("TRITON_CACHE_DIR", Path.home() / ".triton" / "cache")),
        Path(os.environ.get("TORCHINDUCTOR_CACHE_DIR", TEMP_DIR / "torchinductor")),
        temp_root / f"torchinductor_{username}",
        _VCVARS_CACHE,
        _PROBE_CACHE,
    }
    removed: list[str] = []
    errors: list[str] = []
    for path in candidates:
        try:
            if path.is_dir():
                shutil.rmtree(path)
                removed.append(str(path))
            elif path.is_file():
                path.unlink()
                removed.append(str(path))
        except OSError as exc:
            errors.append(f"{path}: {exc}")
    with _CACHE_LOCK:
        _PROBE_MEMORY.clear()
    return {"removed": removed, "errors": errors}


__all__ = [
    "COMPILE_MODE_OPTIONS",
    "DEFAULT_COMPILE_MODE",
    "CompileEnvReport",
    "CompileFallback",
    "CompileModeOption",
    "CompilePlan",
    "apply_compile",
    "capture_vcvars_env",
    "clear_inductor_caches",
    "compile_mode_choices",
    "compile_mode_values",
    "disabled_compile_modes",
    "is_compile_runtime_error",
    "normalize_compile_mode",
    "prepare_compile_env",
    "probe_compile_environment",
    "release_compiled_model",
    "restore_eager_model",
]
