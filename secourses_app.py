"""Command-line entry point for SECourses Video Captioner Pro."""

from __future__ import annotations

import argparse
import os
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

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
    parser.add_argument(
        "--server-name",
        default=None,
        help="Interface/address for the Gradio server. Omitted by default so Gradio picks its own.",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=None,
        help="TCP port for the Gradio server. Omitted by default so Gradio takes the next free port.",
    )
    parser.add_argument("--no-browser", action="store_true", help="Do not open the local URL automatically.")
    parser.add_argument("--inbrowser", action="store_true", help="Open the local URL automatically.")
    return parser


# Gradio serves ``favicon_path`` at /favicon.ico but never emits a <link> for it,
# so browsers fall back to the implicit probe and can keep a stale default cached.
# Declaring it explicitly also tells them the payload is SVG rather than an .ico.
FAVICON_HEAD = '<link rel="icon" type="image/svg+xml" href="/favicon.ico">'


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
    # Passing server_name/server_port pins Gradio to exactly that address and port,
    # so an occupied port aborts the launch instead of rolling to the next one.
    # Leave both out unless the user asked for them.
    server_kwargs: dict[str, object] = {}
    if args.server_name is not None:
        server_kwargs["server_name"] = str(args.server_name)
    if args.server_port is not None:
        server_kwargs["server_port"] = int(args.server_port)

    demo.queue(default_concurrency_limit=1, max_size=64).launch(
        share=bool(args.share),
        **server_kwargs,
        inbrowser=bool(args.inbrowser or not args.no_browser),
        show_error=True,
        theme=build_theme(),
        css=build_css(),
        head=FAVICON_HEAD + HOTKEYS_HEAD,
        allowed_paths=discover_allowed_paths(),
        favicon_path=str(favicon),
        max_file_size="8gb",
        footer_links=[],
    )


if __name__ == "__main__":
    main()
