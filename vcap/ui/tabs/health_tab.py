"""Environment, GPU, attention, disk, and model-management diagnostics."""

from __future__ import annotations

import html
import hashlib
import importlib.metadata
import importlib.util
import os
import platform
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import gradio as gr

from vcap.core import gpu
from vcap.core.media import find_ffmpeg
from vcap.core.paths import open_in_file_manager
from vcap.core.subprocess_runner import build_child_env, kill_process_tree
from vcap.models.attention import probe_available
from vcap.models.downloads import _parse_status, ensure_model, format_status_line
from vcap.models.llamacpp_install import describe_runtime, ensure_llamacpp
from vcap.models.registry import (
    MODEL_SPECS,
    all_variant_choices,
    get_variant,
    resolve_model_dir,
    variant_is_ready,
    variant_size_gb,
)
from vcap.models.torch_compile import clear_inductor_caches, probe_compile_environment
from vcap.ui.components import action_button, render_progress_html

if TYPE_CHECKING:
    from vcap.ui.app import UiContext


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _torch_cuda_version() -> str:
    """Read Torch's generated version module without importing Torch/CUDA."""

    try:
        spec = importlib.util.find_spec("torch")
        if spec is None or spec.origin is None:
            return "unavailable"
        version_file = Path(spec.origin).parent / "version.py"
        text = version_file.read_text(encoding="utf-8", errors="replace")
        match = re.search(
            r"^cuda(?:\s*:[^=]+)?\s*=\s*['\"]([^'\"]*)['\"]",
            text,
            flags=re.MULTILINE,
        )
        return (match.group(1) or "CPU-only") if match else "unknown"
    except Exception:
        return "unknown"


def _ffmpeg_description() -> str:
    executable = find_ffmpeg()
    if not executable:
        return "not found"
    try:
        completed = subprocess.run(
            [executable, "-version"], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=8, check=False,
        )
        first = completed.stdout.splitlines()[0] if completed.stdout else "version unavailable"
        return f"{executable} ({first})"
    except Exception as exc:
        return f"{executable} ({exc})"


def _disk_line(label: str, path: Path) -> str:
    candidate = path
    while not candidate.exists() and candidate.parent != candidate:
        candidate = candidate.parent
    try:
        usage = shutil.disk_usage(candidate)
        gib = float(1024**3)
        return f"{label}: {usage.free / gib:.1f} GiB free of {usage.total / gib:.1f} GiB ({path})"
    except OSError as exc:
        return f"{label}: unavailable ({exc})"


def environment_report(ctx: "UiContext") -> str:
    """Build a copy-friendly report without importing Torch in the UI process."""

    attention = probe_available()
    lines = [
        "SECourses Video Captioner Pro environment",
        f"OS: {platform.platform()}",
        f"Python: {platform.python_version()} ({sys.executable})",
        f"Torch: {_package_version('torch')}",
        f"Torch CUDA build: {_torch_cuda_version()}",
        f"Transformers: {_package_version('transformers')}",
        f"Gradio: {_package_version('gradio')}",
        f"FFmpeg: {_ffmpeg_description()}",
        "Attention: " + ", ".join(
            f"{name}={'available' if available else 'missing'}"
            for name, available in attention.items()
            if name in {"flash_attention_2", "sage", "xformers", "sdpa"}
        ),
        _disk_line("Models disk", ctx.models_dir),
        _disk_line("Outputs disk", ctx.outputs_dir),
        _disk_line("Temp disk", ctx.temp_dir),
    ]
    devices = gpu.list_gpus()
    if not devices:
        lines.append("GPUs: telemetry unavailable")
    for item in devices:
        compute = ".".join(str(value) for value in item.compute_capability) if item.compute_capability else "unknown"
        lines.append(
            f"GPU {item.index}: {item.name}; {item.total_gb:.1f} GiB; "
            f"{item.free_gb:.1f} GiB free; compute {compute}; driver {item.driver_version or 'unknown'}"
        )
    return "\n".join(lines)


def _gpu_rows() -> list[list[Any]]:
    rows: list[list[Any]] = []
    for item in gpu.list_gpus():
        compute = ".".join(str(value) for value in item.compute_capability) if item.compute_capability else "-"
        rows.append([
            item.index, item.name, round(item.total_gb, 2), round(item.free_gb, 2),
            round(item.used_gb, 2), compute, item.driver_version or "-", "yes" if item.is_default else "",
        ])
    return rows


def _local_bytes(folder: Path) -> int:
    total = 0
    if not folder.is_dir():
        return total
    try:
        for path in folder.rglob("*"):
            try:
                if path.is_file():
                    total += path.stat().st_size
            except OSError:
                continue
    except (OSError, PermissionError):
        pass
    return total


def model_inventory() -> tuple[list[list[Any]], list[str]]:
    """Return every registry variant row and its stable key order."""

    rows: list[list[Any]] = []
    keys: list[str] = []
    for spec in MODEL_SPECS.values():
        for variant in spec.variants:
            ready, _ = variant_is_ready(variant.key)
            keys.append(variant.key)
            rows.append([
                f"{spec.label} · {variant.label}",
                variant.scheme,
                round(variant_size_gb(variant.key), 2),
                ("✓ Ready" if ready else "✗ Not ready"),
                _local_bytes(resolve_model_dir(variant.key)),
            ])
    return rows, keys


def _find_downloader(ctx: "UiContext") -> Path | None:
    override = os.environ.get("SECOURSES_VCAP_DOWNLOADER", "").strip().strip('"').strip("'")
    candidates = [
        Path(override).expanduser() if override else None,
        ctx.app_dir.parent / "Models_Downloader.py",
        ctx.app_dir / "Models_Downloader.py",
    ]
    return next((path.resolve(strict=False) for path in candidates if path is not None and path.is_file()), None)


def selected_model_action_key(value: Any) -> str:
    """Validate the dropdown value received by a model action click."""

    key = str(value or "").strip()
    if not key:
        raise ValueError("Choose a model variant first")
    get_variant(key)
    return key


def _selected_status(key: str, message: str) -> str:
    size_bytes = _variant_disk_usage_bytes(key)
    return (
        f"Selected: {html.escape(key)} · On disk: {size_bytes / (1024 ** 3):.2f} GB "
        f"({size_bytes:,} bytes)  \n{message}"
    )


def _variant_disk_usage_bytes(key: str, usage_fn: Any = None) -> int:
    """Read one variant's on-disk usage, with a pre-F1 compatibility fallback."""

    if usage_fn is None:
        try:
            from vcap.models.downloads import variant_disk_usage
        except ImportError:
            return _local_bytes(resolve_model_dir(key))
        usage_fn = variant_disk_usage
    try:
        return max(0, int(usage_fn(key) or 0))
    except Exception:
        return _local_bytes(resolve_model_dir(key))


def request_model_delete(
    variant_key: str,
    pong: Mapping[str, Any] | None = None,
    usage_fn: Any = None,
) -> dict[str, Any]:
    """Return the inline delete-confirmation state or a plain blocked reason."""

    key = selected_model_action_key(variant_key)
    worker = dict(pong or {})
    if worker.get("busy"):
        return {
            "state": "blocked",
            "variant_key": key,
            "message": "A caption job is running; model files cannot be deleted yet.",
        }
    if str(worker.get("loaded_variant") or "") == key:
        return {
            "state": "blocked",
            "variant_key": key,
            "message": f"{key} is resident in the worker; unload it before deleting its files.",
        }
    size_bytes = _variant_disk_usage_bytes(key, usage_fn)
    if size_bytes <= 0:
        return {
            "state": "blocked",
            "variant_key": key,
            "message": f"No on-disk files were found for {key}.",
        }
    label = str(get_variant(key).label)
    return {
        "state": "confirm",
        "variant_key": key,
        "size_bytes": size_bytes,
        "question": f"Delete {label} ({size_bytes / (1024 ** 3):.2f} GB) from disk?",
    }


def delete_model_files_report(variant_key: str, delete_fn: Any = None) -> str:
    """Delete a variant through F1's helper and render its complete report."""

    key = selected_model_action_key(variant_key)
    if delete_fn is None:
        try:
            from vcap.models.downloads import delete_variant_files
        except ImportError:
            return "<span class='vc-warn'>Delete model files becomes available after the backend update.</span>"
        delete_fn = delete_variant_files
    try:
        report = delete_fn(key)
        value = dict(report) if isinstance(report, Mapping) else {
            name: getattr(report, name)
            for name in ("variant_key", "folder", "files_removed", "bytes_freed", "errors")
            if hasattr(report, name)
        }
        files = int(value.get("files_removed", 0) or 0)
        freed = int(value.get("bytes_freed", 0) or 0)
        errors = [str(item) for item in (value.get("errors") or [])]
        message = f"Removed {files} file(s) and freed {freed:,} bytes ({freed / (1024 ** 3):.2f} GB) for {key}."
        if errors:
            message += " Errors: " + " | ".join(errors[:6])
        css = "vc-ok" if not errors else ("vc-warn" if files else "vc-err")
        return f"<span class='{css}'>{html.escape(message)}</span>"
    except Exception as exc:
        return f"<span class='vc-err'>Could not delete {html.escape(key)}: {html.escape(str(exc))}</span>"


def render_update_status(status: Any) -> str:
    """Color an UpdateStatus message by success and behind/ahead state."""

    if isinstance(status, Mapping):
        ok = bool(status.get("ok"))
        behind = int(status.get("behind", 0) or 0)
        message = str(status.get("message") or "Update check returned no message.")
    else:
        ok = bool(getattr(status, "ok", False))
        behind = int(getattr(status, "behind", 0) or 0)
        message = str(getattr(status, "message", "Update check returned no message."))
    css = "vc-err" if not ok else ("vc-warn" if behind > 0 else "vc-ok")
    return f"<span class='{css}'>{html.escape(message)}</span>"


def _cancelled(token: object | None) -> bool:
    if token is None:
        return False
    for name in ("is_cancelled", "is_set"):
        method = getattr(token, name, None)
        if callable(method):
            try:
                return bool(method())
            except Exception:
                return False
    return bool(getattr(token, "cancelled", False))


def _callback(callback: Any, message: str, payload: dict[str, Any]) -> None:
    if callback is None:
        return
    for args in ((message, payload), (payload,), (message,)):
        try:
            callback(*args)
            return
        except TypeError:
            continue


def verify_local_files(
    checks: Sequence[tuple[Path, int | None, str | None]],
    progress_cb: Any = None,
    cancel: object | None = None,
) -> tuple[bool, str]:
    """Verify local file sizes and SHA-256 values with five-second progress."""

    normalized: list[tuple[Path, int, str | None]] = []
    for path, expected_size, expected_sha in checks:
        candidate = Path(path)
        if not candidate.is_file():
            return False, f"missing {candidate.name}"
        actual_size = candidate.stat().st_size
        if expected_size is not None and actual_size != int(expected_size):
            return False, f"{candidate.name} has {actual_size} bytes; expected {int(expected_size)}"
        normalized.append((candidate, actual_size, expected_sha))
    total = sum(size for _, size, _ in normalized)
    aggregate_done = 0
    _callback(
        progress_cb,
        f"Verifying {len(normalized)} local file(s)",
        {"state": "verifying", "fraction": 0.0, "bytes_done": 0, "bytes_total": total},
    )
    for path, size, expected_sha in normalized:
        digest = hashlib.sha256()
        file_done = 0
        last_report = time.monotonic()
        with path.open("rb") as handle:
            while True:
                if _cancelled(cancel):
                    return False, "verification cancelled"
                chunk = handle.read(16 * 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                file_done += len(chunk)
                now = time.monotonic()
                if now - last_report >= 5.0:
                    done = aggregate_done + file_done
                    _callback(
                        progress_cb,
                        f"Verifying {path.name}: {file_done * 100.0 / max(size, 1):.1f}%",
                        {
                            "state": "verifying",
                            "fraction": done / max(total, 1),
                            "bytes_done": done,
                            "bytes_total": total,
                        },
                    )
                    last_report = now
        actual_sha = digest.hexdigest().casefold()
        expected = str(expected_sha or "").removeprefix("sha256:").casefold()
        if expected and actual_sha != expected:
            return False, f"SHA-256 mismatch for {path.name}: got {actual_sha}, expected {expected}"
        aggregate_done += size
        _callback(
            progress_cb,
            f"Verified {path.name}",
            {
                "state": "verifying",
                "fraction": aggregate_done / max(total, 1),
                "bytes_done": aggregate_done,
                "bytes_total": total,
            },
        )
    return True, f"{len(normalized)} file(s), {total} bytes, size and SHA-256 verified"


def verify_local_gguf(
    variant_key: str,
    progress_cb: Any = None,
    cancel: object | None = None,
) -> tuple[bool, str]:
    """Verify one registered GGUF variant without invoking the HF downloader."""

    variant = get_variant(variant_key)
    if variant.backend != "llamacpp" or not variant.gguf_files:
        raise ValueError(f"{variant_key} is not a registered GGUF variant")
    sizes = variant.gguf_file_sizes or ()
    hashes = variant.gguf_sha256 or ()
    folder = resolve_model_dir(variant.key)
    checks = [
        (
            folder / name,
            sizes[index] if index < len(sizes) else None,
            hashes[index] if index < len(hashes) else None,
        )
        for index, name in enumerate(variant.gguf_files)
    ]
    return verify_local_files(checks, progress_cb, cancel)


def _gguf_readiness() -> str:
    lines = []
    for spec in MODEL_SPECS.values():
        for variant in spec.variants:
            if variant.backend != "llamacpp":
                continue
            ready, detail = variant_is_ready(variant.key)
            lines.append(f"- `{variant.key}`: **{'ready' if ready else 'not ready'}** ({html.escape(detail)})")
    return "\n".join(lines) or "No GGUF variants are registered."


def _runtime_report() -> str:
    runtime = describe_runtime()
    version = str(runtime.get("version_output") or "not runnable").splitlines()[0]
    return (
        f"**Pinned:** `{runtime['pinned_tag']}` (minimum `b{runtime['minimum_build']}`)  \n"
        f"**Install:** `{runtime['install_path']}`  \n"
        f"**llama-server:** {'found and runnable' if runtime['runs'] else 'missing or not runnable'}  \n"
        f"**Version:** {html.escape(version)}  \n"
        f"**CUDA build:** {'yes' if runtime['cuda_build'] else 'no'}\n\n"
        f"**GGUF readiness**\n{_gguf_readiness()}"
    )


def _compile_report(force: bool = False) -> str:
    report = probe_compile_environment(force=force)
    compiler = report.msvc_version or report.gcc_version or "not found"
    notes = " | ".join(report.messages[-3:]) if report.messages else "No probe warnings."
    return (
        f"**Readiness:** `{report.inductor_ready}`  \n"
        f"**C++ toolchain:** {html.escape(compiler)}  \n"
        f"**Triton:** {'ready' if report.triton_ok else 'not ready'}"
        + (f" (`{report.triton_version}`)" if report.triton_version else "")
        + f"  \n{html.escape(notes)}"
    )


def _pipeline_ping(ctx: "UiContext", timeout_s: float = 0.6) -> dict[str, Any]:
    """Read model health from the local cache or an idle persistent worker."""

    client = ctx.pipeline_client
    ping = getattr(client, "ping", None)
    if callable(ping):
        try:
            return dict(ping(timeout_s=timeout_s))
        except Exception as exc:
            return {"error": str(exc)}
    if not bool(getattr(client, "subprocess_mode", True)):
        try:
            from vcap.pipeline.runner import loaded_block_swap_summary, loaded_variant_key

            return {
                "ev": "pong",
                "loaded_variant": loaded_variant_key(),
                "block_swap": loaded_block_swap_summary(),
            }
        except Exception as exc:
            return {"error": str(exc)}

    state_lock = getattr(client, "_state_lock", None)
    try:
        if state_lock is not None:
            with state_lock:
                worker = getattr(client, "_worker", None)
                busy = bool(getattr(client, "_busy", False))
        else:
            worker = getattr(client, "_worker", None)
            busy = bool(getattr(client, "_busy", False))
    except Exception as exc:
        return {"error": str(exc)}
    if busy:
        return {"busy": True}
    if worker is None or not worker.is_alive():
        return {"ev": "pong", "loaded_variant": None, "block_swap": None}

    run_lock = getattr(client, "_run_lock", None)
    if run_lock is None or not run_lock.acquire(blocking=False):
        return {"busy": True}
    events = getattr(client, "_events", None)
    deferred: list[Any] = []
    try:
        if bool(getattr(client, "_busy", False)):
            return {"busy": True}
        worker.send({"cmd": "ping"})
        deadline = time.monotonic() + max(0.05, float(timeout_s))
        while time.monotonic() < deadline:
            try:
                event = events.get(timeout=min(0.1, deadline - time.monotonic()))
            except queue.Empty:
                continue
            if isinstance(event, Mapping) and event.get("ev") == "pong":
                return dict(event)
            deferred.append(event)
        return {"error": "Worker health ping timed out"}
    except Exception as exc:
        return {"error": str(exc)}
    finally:
        if events is not None:
            for event in deferred:
                events.put(event)
        run_lock.release()


def render_model_health(pong: Mapping[str, Any] | None) -> str:
    """Format a worker pong as compact loaded-model and block-swap status."""

    data = dict(pong or {})
    if data.get("busy"):
        return "**Loaded model:** worker is busy; status will refresh after the active job."
    if data.get("error"):
        return f"<span class='vc-warn'>Model health unavailable: {html.escape(str(data['error']))}</span>"
    loaded = data.get("loaded_variant")
    if not loaded:
        return "**Loaded model:** none"
    lines = [f"**Loaded model:** `{html.escape(str(loaded))}`"]
    summary = data.get("block_swap")
    if not isinstance(summary, Mapping):
        lines.append("**Block swap:** not active or unavailable")
        return "  \n".join(lines)

    mode = html.escape(str(summary.get("mode") or "active"))
    resident = summary.get("resident_layers")
    total = summary.get("layer_count")
    swapped = summary.get("swapped_layers")
    slots = summary.get("slots")
    details: list[str] = []
    if resident is not None and total is not None:
        details.append(f"{int(resident)}/{int(total)} resident")
    if swapped is not None:
        details.append(f"{int(swapped)} swapped")
    if slots is not None:
        details.append(f"{int(slots)} slot(s)")
    if summary.get("pinned_gib") is not None:
        details.append(f"{float(summary['pinned_gib']):.2f} GiB pinned")
    lines.append(f"**Block swap:** `{mode}`" + (f"; {', '.join(details)}" if details else ""))
    peak = summary.get("expected_peak_gib")
    reserve = summary.get("reserve_gib")
    if peak is not None or reserve is not None:
        peak_text = f"{float(peak):.2f} GiB" if peak is not None else "unknown"
        reserve_text = f"{float(reserve):.2f} GiB" if reserve is not None else "unknown"
        lines.append(f"**Expected peak:** {peak_text}; **reserve:** {reserve_text}")
    return "  \n".join(lines)


def _model_health_report(ctx: "UiContext") -> str:
    return render_model_health(_pipeline_ping(ctx))


def build(ctx: "UiContext") -> None:
    """Render live health information and streaming model actions."""

    report_text = environment_report(ctx)
    with gr.Row(equal_height=False):
        with gr.Column(scale=5, min_width=480):
            gr.Markdown("### Environment")
            report = gr.Code(
                value=report_text, language=None, lines=15, max_lines=22,
                label="Environment report", buttons=["copy"], interactive=False,
            )
            with gr.Row():
                copy_report = action_button("Copy environment report", "cyan")
                check_updates = action_button(
                    "🔎 Check for updates",
                    "cobalt",
                    elem_id="vc_check_for_updates",
                )
            update_status = gr.Markdown(
                "<span class='vc-help'>Update status has not been checked.</span>",
                elem_classes=["vc-status"],
                elem_id="vc_update_status",
            )

        with gr.Column(scale=5, min_width=480):
            gr.Markdown("### GPUs")
            gpu_table = gr.Dataframe(
                value=_gpu_rows(),
                headers=["#", "Name", "VRAM GiB", "Free GiB", "Used GiB", "Compute", "Driver", "Default"],
                datatype=["number", "str", "number", "number", "number", "str", "str", "str"],
                type="array", interactive=False, show_search="none", max_height=270,
                pinned_columns=2, static_columns=list(range(8)), buttons=["copy", "fullscreen"],
                label="GPU inventory",
            )
            gpu_default = int(ctx.states.get("gpu_index_default", 0) or 0)
            meter = gr.HTML(gpu.render_resource_meter_html(gpu.resource_snapshot(gpu_default)))

    with gr.Row(equal_height=False):
        with gr.Column(scale=6, min_width=520):
            gr.Markdown("### llama.cpp runtime")
            runtime_status = gr.Markdown(_runtime_report(), elem_classes=["vc-status"])
            with gr.Row():
                install_runtime = action_button("Install / repair llama.cpp", "emerald")
                refresh_runtime = action_button("Refresh", "amber")
            runtime_progress = gr.HTML(render_progress_html(0.0, "Ready", "Runtime inspection complete."))
            runtime_log = gr.Textbox(
                value="", label="llama.cpp install log", lines=8, max_lines=12,
                interactive=False, autoscroll=True, elem_classes=["vc-log"],
            )

        with gr.Column(scale=4, min_width=420):
            gr.Markdown("### torch.compile toolchain")
            compile_status = gr.Markdown(_compile_report(), elem_classes=["vc-status"])
            clear_compile = action_button("Clear compile caches", "orange")
            gr.Markdown("### Loaded model & block swap")
            model_health_status = gr.Markdown(
                _model_health_report(ctx), elem_classes=["vc-status"]
            )
            with gr.Row():
                refresh_model_health = action_button("Refresh model status", "cyan")
                unload_model = action_button(
                    "⏏ Unload model", "plum",
                    elem_id="vc_health_unload_model",
                )
                open_logs = action_button(
                    "📂 Open logs folder", "berry",
                    elem_id="vc_open_logs_folder_health",
                )

    gr.Markdown("### Models")
    initial_rows, initial_keys = model_inventory()
    model_table = gr.Dataframe(
        value=initial_rows,
        headers=["Variant", "Scheme", "Size GB", "Ready", "Local bytes"],
        datatype=["str", "str", "number", "str", "number"],
        type="array", interactive=False, show_search="filter", max_height=520,
        column_widths=[440, 170, 100, 150, 140],
        pinned_columns=1, static_columns=list(range(5)), buttons=["copy", "fullscreen"],
        label="Model variants",
    )
    variant_choices = all_variant_choices()
    initial_variant_key = variant_choices[0][1] if variant_choices else ""
    initial_delete_check = (
        request_model_delete(initial_variant_key, _pipeline_ping(ctx))
        if initial_variant_key
        else {"state": "blocked"}
    )
    initial_model_message = "Ready for Download or Verify."
    if initial_delete_check.get("state") == "blocked":
        initial_model_message += (
            " <span class='vc-warn'>"
            f"{html.escape(str(initial_delete_check.get('message') or 'Delete unavailable.'))}"
            "</span>"
        )
    with gr.Row():
        variant = gr.Dropdown(
            choices=variant_choices, value=initial_variant_key or None,
            label="Model action", info="Pick a variant to download or verify.", scale=5,
        )
        download = action_button("📥 Download", "sky")
        cancel = action_button("Cancel", "red")
        verify = action_button("🔍 Verify", "violet")
        delete_model = action_button(
            "🗑 Delete model files",
            "olive",
            elem_id="vc_health_delete_model_files",
            interactive=initial_delete_check.get("state") == "confirm",
        )
        open_models = action_button("Open models folder", "teal")
    delete_confirmation_state = gr.State({})
    with gr.Row(
        visible=False,
        elem_id="vc_health_delete_model_confirmation",
        elem_classes=["vc-confirm-bar"],
    ) as delete_confirmation:
        delete_question = gr.Markdown("Delete the selected model files from disk?")
        delete_yes = action_button(
            "✔ Yes, delete",
            "berry",
            scale=0,
            min_width=132,
            variant="stop",
            elem_id="vc_health_delete_model_yes",
        )
        delete_keep = action_button(
            "✖ Keep files",
            "slate",
            scale=0,
            min_width=124,
            elem_id="vc_health_delete_model_keep",
        )
    model_status = gr.Markdown(
        _selected_status(initial_variant_key, initial_model_message)
        if initial_variant_key
        else "Ready.",
        elem_classes=["vc-status"],
    )
    model_progress = gr.HTML(render_progress_html(0.0, "Ready", "Choose a model action."))
    download_log = gr.Textbox(
        value="", label="Model action log", lines=14, max_lines=18,
        interactive=False, autoscroll=True, elem_classes=["vc-log"],
    )

    meter_timer = gr.Timer(2.0)
    model_health_timer = gr.Timer(3.0)
    gpu_index_state = ctx.states.get("gpu_index")
    if gpu_index_state is not None:
        meter_timer.tick(
            lambda selected: gpu.render_resource_meter_html(gpu.resource_snapshot(int(selected or 0))),
            inputs=gpu_index_state, outputs=meter,
            queue=False, show_progress="hidden", api_visibility="private",
        )
    else:
        meter_timer.tick(
            lambda: gpu.render_resource_meter_html(gpu.resource_snapshot(gpu_default)),
            outputs=meter, queue=False, show_progress="hidden", api_visibility="private",
        )

    copy_report.click(
        fn=None, inputs=report, outputs=[],
        js="(text) => { navigator.clipboard.writeText(text || ''); return []; }",
        queue=False, show_progress="hidden", api_visibility="private",
    )

    def check_updates_handler() -> str:
        try:
            from vcap.core.updates import check_for_updates
        except ImportError:
            return "<span class='vc-warn'>Update check becomes available after the backend update.</span>"
        try:
            return render_update_status(check_for_updates(ctx.app_dir))
        except Exception as exc:
            return f"<span class='vc-err'>Update check failed: {html.escape(str(exc))}</span>"

    check_updates.click(
        check_updates_handler,
        outputs=update_status,
        concurrency_id="vc-update-check",
        concurrency_limit=1,
        show_progress="hidden",
        api_visibility="private",
    )

    def refresh_model_card(
        selected_key: str,
        confirmation: Mapping[str, Any] | None,
    ) -> tuple[str, Any, Any]:
        pong = _pipeline_ping(ctx)
        busy = bool(pong.get("busy")) or ctx.get_active_cancel() is not None
        if busy:
            pong = {**pong, "busy": True}
        delete_check = request_model_delete(selected_key, pong) if selected_key else {"state": "blocked"}
        delete_enabled = delete_check.get("state") == "confirm" and not bool(confirmation)
        detail = render_model_health(pong)
        if delete_check.get("state") == "blocked":
            detail += f"  \n<span class='vc-warn'>{html.escape(str(delete_check.get('message') or 'Delete unavailable.'))}</span>"
        return detail, gr.update(interactive=not busy), gr.update(interactive=delete_enabled)

    refresh_model_health.click(
        refresh_model_card,
        inputs=[variant, delete_confirmation_state],
        outputs=[model_health_status, unload_model, delete_model],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    model_health_timer.tick(
        refresh_model_card,
        inputs=[variant, delete_confirmation_state],
        outputs=[model_health_status, unload_model, delete_model],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )

    def unload_health(selected_gpu: Any, selected_key: str) -> tuple[str, Any, Any]:
        from vcap.ui.tabs.caption_tab import unload_model_report

        message = unload_model_report(ctx.pipeline_client, int(selected_gpu or 0))
        check = request_model_delete(str(selected_key), _pipeline_ping(ctx))
        return (
            f"{message}\n\n{_model_health_report(ctx)}",
            gr.update(interactive=True),
            gr.update(interactive=check.get("state") == "confirm"),
        )

    if gpu_index_state is not None:
        unload_model.click(
            unload_health,
            inputs=[gpu_index_state, variant],
            outputs=[model_health_status, unload_model, delete_model],
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        )
    else:
        unload_model.click(
            lambda selected_key: unload_health(gpu_default, selected_key),
            inputs=variant,
            outputs=[model_health_status, unload_model, delete_model],
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        )

    def open_logs_handler() -> str:
        ctx.logs_dir.mkdir(parents=True, exist_ok=True)
        ok, message = open_in_file_manager(ctx.logs_dir)
        return f"<span class='{'vc-ok' if ok else 'vc-err'}'>{html.escape(message)}</span>"

    open_logs.click(
        open_logs_handler,
        outputs=model_health_status,
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )

    def install_runtime_handler():
        events: "queue.Queue[tuple[str, Any]]" = queue.Queue()
        terminal: dict[str, Any] = {}

        def callback(message: Any, payload: Any = None) -> None:
            events.put(("line", (str(message), dict(payload) if isinstance(payload, Mapping) else {})))

        def work() -> None:
            try:
                terminal["path"] = ensure_llamacpp(progress_cb=callback)
            except BaseException as exc:
                terminal["error"] = exc
            finally:
                events.put(("terminal", None))

        threading.Thread(target=work, daemon=True, name="vcap-llamacpp-install-ui").start()
        lines = ["Starting llama.cpp install / repair"]
        yield "\n".join(lines), render_progress_html(0.0, "Starting", lines[-1]), gr.skip()
        while True:
            kind, value = events.get()
            if kind == "terminal":
                break
            message, payload = value
            lines.append(message)
            lines = lines[-300:]
            fraction = payload.get("fraction")
            yield (
                "\n".join(lines),
                render_progress_html(float(fraction or 0.0), "Installing llama.cpp", message),
                gr.skip(),
            )
        if "error" in terminal:
            message = str(terminal["error"])
            lines.append(f"ERROR: {message}")
            yield (
                "\n".join(lines),
                render_progress_html(0.0, "Failed", message),
                f"<span class='vc-err'>{html.escape(message)}</span>\n\n{_runtime_report()}",
            )
            return
        message = f"llama.cpp ready: {terminal['path']}"
        lines.append(message)
        yield "\n".join(lines), render_progress_html(1.0, "Ready", message), _runtime_report()

    install_runtime.click(
        install_runtime_handler,
        outputs=[runtime_log, runtime_progress, runtime_status],
        concurrency_id="llamacpp_install",
        concurrency_limit=1,
        show_progress="hidden",
        api_visibility="private",
    )
    refresh_runtime.click(
        _runtime_report,
        outputs=runtime_status,
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )

    def clear_compile_handler() -> str:
        result = clear_inductor_caches()
        removed = len(result.get("removed") or [])
        errors = result.get("errors") or []
        message = f"Cleared {removed} compile cache path(s)."
        if errors:
            message += " " + " | ".join(str(value) for value in errors[:3])
        return f"{_compile_report(force=True)}\n\n{html.escape(message)}"

    clear_compile.click(
        clear_compile_handler,
        outputs=compile_status,
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )

    def select_model(evt: gr.SelectData) -> Any:
        row = int(evt.index[0] if isinstance(evt.index, (tuple, list)) else evt.index)
        _, keys = model_inventory()
        return keys[row] if 0 <= row < len(keys) else gr.skip()

    model_table.select(
        select_model, outputs=variant,
        queue=False, show_progress="hidden", api_visibility="private",
    )
    def selected_model_status(value: str) -> tuple[str, Any, dict[str, Any], Any]:
        key = str(value or "")
        try:
            pong = _pipeline_ping(ctx)
            if ctx.get_active_cancel() is not None:
                pong = {**pong, "busy": True}
            check = request_model_delete(key, pong)
            message = "Ready for Download or Verify."
            if check.get("state") == "blocked":
                message += f" <span class='vc-warn'>{html.escape(str(check.get('message') or 'Delete unavailable.'))}</span>"
            return (
                _selected_status(key, message),
                gr.update(interactive=check.get("state") == "confirm"),
                {},
                gr.update(visible=False),
            )
        except Exception as exc:
            return (
                f"<span class='vc-err'>{html.escape(str(exc))}</span>",
                gr.update(interactive=False),
                {},
                gr.update(visible=False),
            )

    variant.change(
        selected_model_status,
        inputs=variant,
        outputs=[model_status, delete_model, delete_confirmation_state, delete_confirmation],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )

    def request_delete_handler(key: str) -> tuple[Any, ...]:
        try:
            pong = _pipeline_ping(ctx)
            if ctx.get_active_cancel() is not None:
                pong = {**pong, "busy": True}
            check = request_model_delete(key, pong)
            if check.get("state") != "confirm":
                reason = str(check.get("message") or "Model files cannot be deleted right now.")
                return (
                    {},
                    gr.update(visible=False),
                    gr.skip(),
                    gr.update(interactive=False),
                    _selected_status(str(key), f"<span class='vc-warn'>{html.escape(reason)}</span>"),
                )
            return (
                check,
                gr.update(visible=True),
                f"⚠ {html.escape(str(check['question']))}",
                gr.update(interactive=False),
                _selected_status(str(key), "Waiting for delete confirmation."),
            )
        except Exception as exc:
            return (
                {},
                gr.update(visible=False),
                gr.skip(),
                gr.update(interactive=False),
                f"<span class='vc-err'>{html.escape(str(exc))}</span>",
            )

    def keep_model_files(key: str) -> tuple[Any, ...]:
        try:
            check = request_model_delete(key, _pipeline_ping(ctx))
            enabled = check.get("state") == "confirm" and ctx.get_active_cancel() is None
        except Exception:
            enabled = False
        return (
            {},
            gr.update(visible=False),
            gr.update(interactive=enabled),
            _selected_status(str(key), "Kept the selected model files."),
        )

    def confirm_delete_handler(state: Mapping[str, Any] | None) -> tuple[Any, ...]:
        key = str((state or {}).get("variant_key") or "")
        if not key:
            return (
                "<span class='vc-warn'>No model deletion is awaiting confirmation.</span>",
                gr.skip(),
                gr.update(visible=False),
                gr.update(interactive=False),
                {},
            )
        pong = _pipeline_ping(ctx)
        if ctx.get_active_cancel() is not None:
            pong = {**pong, "busy": True}
        check = request_model_delete(key, pong)
        if check.get("state") != "confirm":
            reason = str(check.get("message") or "Model files cannot be deleted right now.")
            return (
                _selected_status(key, f"<span class='vc-warn'>{html.escape(reason)}</span>"),
                gr.skip(),
                gr.update(visible=False),
                gr.update(interactive=False),
                {},
            )
        message = delete_model_files_report(key)
        post_check = request_model_delete(key, _pipeline_ping(ctx))
        return (
            _selected_status(key, message),
            model_inventory()[0],
            gr.update(visible=False),
            gr.update(interactive=post_check.get("state") == "confirm"),
            {},
        )

    delete_model.click(
        request_delete_handler,
        inputs=variant,
        outputs=[
            delete_confirmation_state,
            delete_confirmation,
            delete_question,
            delete_model,
            model_status,
        ],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    delete_keep.click(
        keep_model_files,
        inputs=variant,
        outputs=[delete_confirmation_state, delete_confirmation, delete_model, model_status],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    delete_yes.click(
        confirm_delete_handler,
        inputs=delete_confirmation_state,
        outputs=[model_status, model_table, delete_confirmation, delete_model, delete_confirmation_state],
        concurrency_id="model_delete",
        concurrency_limit=1,
        show_progress="hidden",
        api_visibility="private",
    )

    cancel_event = threading.Event()
    cancel_armed = gr.State(0.0)
    cancel_arm_timer = gr.Timer(1.0)

    def download_handler(key: str):
        try:
            key = selected_model_action_key(key)
        except Exception as exc:
            message = str(exc)
            yield "", f"<span class='vc-err'>{html.escape(message)}</span>", gr.skip(), render_progress_html(0.0, "Failed", message)
            return
        cancel_event.clear()
        events: "queue.Queue[tuple[str, Any]]" = queue.Queue()
        terminal: dict[str, Any] = {}

        def callback(message: Any, payload: Any = None) -> None:
            parsed = dict(payload) if isinstance(payload, Mapping) else {}
            events.put(("line", (format_status_line(str(message), parsed), parsed)))

        def work() -> None:
            try:
                terminal["result"] = ensure_model(key, callback, cancel_event)
            except BaseException as exc:
                terminal["error"] = exc
            finally:
                events.put(("terminal", None))

        threading.Thread(target=work, daemon=True, name="vcap-model-download-ui").start()
        lines = [f"Selected: {key}", f"Starting download for {key}"]
        yield "\n".join(lines), _selected_status(key, "Downloading..."), gr.skip(), render_progress_html(0.0, "Starting", lines[-1])
        while True:
            kind, value = events.get()
            if kind == "terminal":
                break
            message, payload = value
            lines.append(message)
            lines = lines[-500:]
            fraction = payload.get("fraction")
            state_name = str(payload.get("state") or "Downloading").replace("_", " ").title()
            yield (
                "\n".join(lines),
                _selected_status(key, html.escape(message)),
                gr.skip(),
                render_progress_html(float(fraction or 0.0), state_name, message),
            )
        if "error" in terminal:
            exc = terminal["error"]
            ctx.app_log.error(f"Model download failed: {exc}", scope="models")
            yield "\n".join([*lines, f"ERROR: {exc}"]), _selected_status(key, f"<span class='vc-err'>{html.escape(str(exc))}</span>"), model_inventory()[0], render_progress_html(0.0, "Failed", str(exc))
            return
        ready, detail = terminal.get("result", (False, "No result"))
        message = f"{key}: {detail}"
        yield "\n".join([*lines, message]), _selected_status(key, f"<span class='{'vc-ok' if ready else 'vc-err'}'>{html.escape(message)}</span>"), model_inventory()[0], render_progress_html(1.0 if ready else 0.0, "Ready" if ready else "Failed", message)

    download.click(
        download_handler, inputs=variant, outputs=[download_log, model_status, model_table, model_progress],
        concurrency_id="model_download", concurrency_limit=1,
        show_progress="hidden", api_visibility="private",
    )

    def cancel_handler(armed_at: float) -> tuple[Any, str, float]:
        now = time.monotonic()
        if armed_at and now - float(armed_at) <= 6.0:
            cancel_event.set()
            return gr.update(value="Cancel"), "Cancellation requested.", 0.0
        return (
            gr.update(value="Click again to confirm"),
            "Click Cancel again within 6 seconds to stop the active model action.",
            now,
        )

    cancel.click(
        cancel_handler,
        inputs=cancel_armed,
        outputs=[cancel, model_status, cancel_armed],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )

    def expire_cancel_handler(armed_at: float) -> tuple[Any, Any]:
        if armed_at and time.monotonic() - float(armed_at) > 6.0:
            return gr.update(value="Cancel"), 0.0
        return gr.skip(), gr.skip()

    cancel_arm_timer.tick(
        expire_cancel_handler,
        inputs=cancel_armed,
        outputs=[cancel, cancel_armed],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )

    def verify_handler(key: str):
        try:
            key = selected_model_action_key(key)
        except Exception as exc:
            message = str(exc)
            yield "", f"<span class='vc-err'>{html.escape(message)}</span>", gr.skip(), render_progress_html(0.0, "Failed", message)
            return
        cancel_event.clear()
        selected_variant = get_variant(key)
        downloader = _find_downloader(ctx)
        if selected_variant.backend != "llamacpp" and downloader is None:
            ready, detail = variant_is_ready(key)
            yield (
                f"Registry check only: {detail}",
                _selected_status(key, f"<span class='{'vc-ok' if ready else 'vc-err'}'>{html.escape(detail)}</span>"),
                model_inventory()[0],
                render_progress_html(1.0 if ready else 0.0, "Registry check", detail),
            )
            return
        events: "queue.Queue[tuple[str, Any]]" = queue.Queue()
        terminal: dict[str, Any] = {}

        def work() -> None:
            if selected_variant.backend == "llamacpp":
                try:
                    def callback(message: Any, payload: Any = None) -> None:
                        parsed = dict(payload) if isinstance(payload, Mapping) else {}
                        events.put(("line", (format_status_line(str(message), parsed), parsed)))

                    terminal["local_result"] = verify_local_gguf(key, callback, cancel_event)
                    if cancel_event.is_set():
                        terminal["cancelled"] = True
                except BaseException as exc:
                    terminal["error"] = exc
                finally:
                    events.put(("terminal", None))
                return
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
            kwargs: dict[str, Any] = {} if os.name == "nt" else {"start_new_session": True}
            process: subprocess.Popen[str] | None = None
            try:
                assert downloader is not None
                process = subprocess.Popen(
                    [sys.executable, "-u", str(downloader), "--verify", key],
                    cwd=str(ctx.app_dir), env=build_child_env(),
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", errors="replace", bufsize=1,
                    creationflags=creationflags, **kwargs,
                )
                assert process.stdout is not None
                output_lines: "queue.Queue[str | None]" = queue.Queue()

                def read_output() -> None:
                    try:
                        for raw in iter(process.stdout.readline, ""):
                            output_lines.put(raw)
                    finally:
                        output_lines.put(None)

                reader = threading.Thread(
                    target=read_output,
                    daemon=True,
                    name="vcap-model-verify-output",
                )
                reader.start()
                while True:
                    if cancel_event.is_set():
                        kill_process_tree(process.pid)
                        terminal["cancelled"] = True
                        break
                    try:
                        line = output_lines.get(timeout=0.1)
                    except queue.Empty:
                        if process.poll() is not None and not reader.is_alive():
                            break
                        continue
                    if line is None:
                        break
                    if line.strip():
                        text = line.rstrip()
                        parsed = _parse_status(text) or {}
                        parsed_message = str(parsed.get("message") or "")
                        if "cannot be verified against published digests" in parsed_message:
                            terminal["verification_detail"] = parsed_message
                        events.put(("line", (format_status_line(text, parsed), parsed)))
                reader.join(timeout=2.0)
                terminal["code"] = process.wait(timeout=10) if process.poll() is None else process.returncode
            except BaseException as exc:
                terminal["error"] = exc
                if process is not None and process.poll() is None:
                    kill_process_tree(process.pid)
            finally:
                events.put(("terminal", None))

        threading.Thread(target=work, daemon=True, name="vcap-model-verify-ui").start()
        lines = [f"Selected: {key}", f"Verifying {key}"]
        yield "\n".join(lines), _selected_status(key, "Verifying..."), gr.skip(), render_progress_html(0.0, "Verifying", lines[-1])
        while True:
            kind, value = events.get()
            if kind == "terminal":
                break
            message, payload = value
            lines.append(message)
            lines = lines[-500:]
            fraction = payload.get("fraction")
            yield (
                "\n".join(lines),
                _selected_status(key, html.escape(message)),
                gr.skip(),
                render_progress_html(float(fraction or 0.0), "Verifying", message),
            )
        if terminal.get("cancelled"):
            message = "Model verification cancelled."
            yield "\n".join([*lines, message]), _selected_status(key, message), model_inventory()[0], render_progress_html(0.0, "Cancelled", message)
            return
        if "error" in terminal:
            message = str(terminal["error"])
            yield "\n".join([*lines, f"ERROR: {message}"]), _selected_status(key, f"<span class='vc-err'>{html.escape(message)}</span>"), model_inventory()[0], render_progress_html(0.0, "Failed", message)
            return
        if "local_result" in terminal:
            ok, detail = terminal["local_result"]
        else:
            ready, registry_detail = variant_is_ready(key)
            detail = str(terminal.get("verification_detail") or registry_detail)
            code = int(terminal.get("code") or 0)
            ok = code == 0 and ready
        message = f"Verification {'passed' if ok else 'failed'} for {key}: {detail}"
        ctx.app_log.log(message, scope="models")
        yield "\n".join([*lines, message]), _selected_status(key, f"<span class='{'vc-ok' if ok else 'vc-err'}'>{html.escape(message)}</span>"), model_inventory()[0], render_progress_html(1.0 if ok else 0.0, "Verified" if ok else "Failed", message)

    verify.click(
        verify_handler, inputs=variant, outputs=[download_log, model_status, model_table, model_progress],
        concurrency_id="model_download", concurrency_limit=1,
        show_progress="hidden", api_visibility="private",
    )

    def open_models_handler() -> str:
        ok, message = open_in_file_manager(ctx.models_dir)
        return f"<span class='{'vc-ok' if ok else 'vc-err'}'>{html.escape(message)}</span>"

    open_models.click(
        open_models_handler, outputs=model_status,
        queue=False, show_progress="hidden", api_visibility="private",
    )


__all__ = [
    "build",
    "environment_report",
    "delete_model_files_report",
    "model_inventory",
    "render_update_status",
    "render_model_health",
    "request_model_delete",
    "selected_model_action_key",
    "verify_local_files",
    "verify_local_gguf",
]
