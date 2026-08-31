"""Command-line entry point for SECourses Video Captioner Pro."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from vcap import APP_DIR, APP_NAME, TEMP_DIR, VERSION, ensure_app_dirs

# Gradio reads its cache location during import, so this must stay above every
# import from vcap.ui or gradio itself.
os.environ.setdefault("GRADIO_TEMP_DIR", str(TEMP_DIR / "gradio"))
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr

from vcap.core.gpu import list_gpus
from vcap.core.logs import setup_utf8_stdio
from vcap.core.paths import discover_allowed_paths
from vcap.ui.app import build_app
from vcap.ui.theme import HOTKEYS_HEAD, build_css, build_theme


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=APP_NAME)
    parser.add_argument("--share", action="store_true", help="Create a temporary public Gradio share link.")
    parser.add_argument("--server-name", default="127.0.0.1", help="Interface/address for the Gradio server.")
    parser.add_argument("--server-port", type=int, default=7860, help="TCP port for the Gradio server.")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the local URL automatically.")
    parser.add_argument("--inbrowser", action="store_true", help="Open the local URL automatically.")
    return parser


def _startup_banner() -> None:
    print("=" * 76)
    print(f"{APP_NAME} v{VERSION}")
    devices = list_gpus()
    if devices:
        for device in devices:
            default = " [default]" if device.is_default else ""
            print(
                f"GPU {device.index}: {device.name} | {device.total_gb:.1f} GB total | "
                f"{device.free_gb:.1f} GB free{default}"
            )
    else:
        print("GPU: no NVIDIA device detected (telemetry unavailable)")
    print("=" * 76)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    setup_utf8_stdio()
    ensure_app_dirs()
    (TEMP_DIR / "gradio").mkdir(parents=True, exist_ok=True)
    _startup_banner()

    demo = build_app()
    gr.set_static_paths([APP_DIR])
    favicon = APP_DIR / "assets" / "favicon.svg"
    demo.queue(default_concurrency_limit=1, max_size=64).launch(
        share=bool(args.share),
        server_name=str(args.server_name),
        server_port=int(args.server_port),
        inbrowser=bool(args.inbrowser or not args.no_browser),
        show_error=True,
        theme=build_theme(),
        css=build_css(),
        head=HOTKEYS_HEAD,
        allowed_paths=discover_allowed_paths(),
        favicon_path=str(favicon),
        max_file_size="8gb",
        footer_links=[],
    )


if __name__ == "__main__":
    main()
