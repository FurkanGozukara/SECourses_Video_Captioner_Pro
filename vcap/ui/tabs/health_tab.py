"""Environment, GPU, attention, disk, and model-management diagnostics."""

from __future__ import annotations

import html
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
from pathlib import Path
from typing import TYPE_CHECKING, Any

import gradio as gr

from vcap.core import gpu
from vcap.core.media import find_ffmpeg
from vcap.core.paths import open_in_file_manager
from vcap.core.subprocess_runner import build_child_env, kill_process_tree
from vcap.models.attention import probe_available
from vcap.models.downloads import ensure_model
from vcap.models.registry import (
    MODEL_SPECS,
    all_variant_choices,
    resolve_model_dir,
    variant_is_ready,
    variant_size_gb,
)
from vcap.ui.components import action_button

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
            copy_report = action_button("Copy environment report", "cyan", size="md")

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
    with gr.Row(elem_classes=["vc-compact-row"]):
        variant = gr.Dropdown(
            choices=variant_choices, value=variant_choices[0][1] if variant_choices else None,
            label="Model action", info="Pick a variant to download or verify.", scale=5,
        )
        download = action_button("📥 Download", "sky", size="md")
        cancel = action_button("Cancel", "red", size="md")
        verify = action_button("🔍 Verify", "violet", size="md")
        open_models = action_button("Open models folder", "teal", size="md")
    model_status = gr.Markdown("Ready.", elem_classes=["vc-status"])
    download_log = gr.Textbox(
        value="", label="Model action log", lines=14, max_lines=18,
        interactive=False, autoscroll=True, elem_classes=["vc-log"],
    )

    meter_timer = gr.Timer(2.0)
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

    def select_model(evt: gr.SelectData) -> Any:
        row = int(evt.index[0] if isinstance(evt.index, (tuple, list)) else evt.index)
        _, keys = model_inventory()
        return keys[row] if 0 <= row < len(keys) else gr.skip()

    model_table.select(
        select_model, outputs=variant,
        queue=False, show_progress="hidden", api_visibility="private",
    )

    cancel_event = threading.Event()

    def download_handler(key: str):
        cancel_event.clear()
        events: "queue.Queue[tuple[str, Any]]" = queue.Queue()
        terminal: dict[str, Any] = {}

        def callback(message: Any, payload: Any = None) -> None:
            del payload
            events.put(("line", str(message)))

        def work() -> None:
            try:
                terminal["result"] = ensure_model(key, callback, cancel_event)
            except BaseException as exc:
                terminal["error"] = exc
            finally:
                events.put(("terminal", None))

        threading.Thread(target=work, daemon=True, name="vcap-model-download-ui").start()
        lines = [f"Starting download for {key}"]
        yield "\n".join(lines), f"Downloading `{key}`...", gr.skip()
        while True:
            kind, value = events.get()
            if kind == "terminal":
                break
            lines.append(str(value))
            lines = lines[-500:]
            yield "\n".join(lines), html.escape(str(value)), gr.skip()
        if "error" in terminal:
            exc = terminal["error"]
            ctx.app_log.error(f"Model download failed: {exc}", scope="models")
            yield "\n".join([*lines, f"ERROR: {exc}"]), f"<span class='vc-err'>{html.escape(str(exc))}</span>", model_inventory()[0]
            return
        ready, detail = terminal.get("result", (False, "No result"))
        message = f"{key}: {detail}"
        yield "\n".join([*lines, message]), f"<span class='{'vc-ok' if ready else 'vc-err'}'>{html.escape(message)}</span>", model_inventory()[0]

    download.click(
        download_handler, inputs=variant, outputs=[download_log, model_status, model_table],
        concurrency_id="model_download", concurrency_limit=1,
        show_progress="hidden", api_visibility="private",
    )

    cancel.click(
        lambda: (cancel_event.set(), "Cancellation requested.")[1], outputs=model_status,
        queue=False, show_progress="hidden", api_visibility="private",
    )

    def verify_handler(key: str):
        cancel_event.clear()
        downloader = _find_downloader(ctx)
        if downloader is None:
            ready, detail = variant_is_ready(key)
            yield f"Registry check only: {detail}", f"<span class='{'vc-ok' if ready else 'vc-err'}'>{html.escape(detail)}</span>", model_inventory()[0]
            return
        events: "queue.Queue[tuple[str, Any]]" = queue.Queue()
        terminal: dict[str, Any] = {}

        def work() -> None:
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
            kwargs: dict[str, Any] = {} if os.name == "nt" else {"start_new_session": True}
            process: subprocess.Popen[str] | None = None
            try:
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
                        events.put(("line", line.rstrip()))
                reader.join(timeout=2.0)
                terminal["code"] = process.wait(timeout=10) if process.poll() is None else process.returncode
            except BaseException as exc:
                terminal["error"] = exc
                if process is not None and process.poll() is None:
                    kill_process_tree(process.pid)
            finally:
                events.put(("terminal", None))

        threading.Thread(target=work, daemon=True, name="vcap-model-verify-ui").start()
        lines = [f"Verifying {key}"]
        yield "\n".join(lines), f"Verifying `{key}`...", gr.skip()
        while True:
            kind, value = events.get()
            if kind == "terminal":
                break
            lines.append(str(value))
            lines = lines[-500:]
            yield "\n".join(lines), html.escape(str(value)), gr.skip()
        if terminal.get("cancelled"):
            message = "Model verification cancelled."
            yield "\n".join([*lines, message]), message, model_inventory()[0]
            return
        if "error" in terminal:
            message = str(terminal["error"])
            yield "\n".join([*lines, f"ERROR: {message}"]), f"<span class='vc-err'>{html.escape(message)}</span>", model_inventory()[0]
            return
        ready, detail = variant_is_ready(key)
        code = int(terminal.get("code") or 0)
        ok = code == 0 and ready
        message = f"Verification {'passed' if ok else 'failed'} for {key}: {detail}"
        ctx.app_log.log(message, scope="models")
        yield "\n".join([*lines, message]), f"<span class='{'vc-ok' if ok else 'vc-err'}'>{html.escape(message)}</span>", model_inventory()[0]

    verify.click(
        verify_handler, inputs=variant, outputs=[download_log, model_status, model_table],
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


__all__ = ["build", "environment_report", "model_inventory"]
