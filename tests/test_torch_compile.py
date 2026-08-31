from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def test_compile_probe_does_not_import_torch_in_parent_process() -> None:
    root = Path(__file__).parents[1]
    code = (
        "import json,sys; "
        "from vcap.models.torch_compile import probe_compile_environment; "
        "report=probe_compile_environment(force=True); "
        "print(json.dumps({'torch_loaded':'torch' in sys.modules,'ready':report.inductor_ready}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=True,
    )
    data = json.loads(completed.stdout.splitlines()[-1])
    assert data["torch_loaded"] is False
    assert data["ready"] in {"full", "triton_only", "cudagraphs_only", "unavailable"}


def test_compile_default_uses_direct_cuda_graphs() -> None:
    from vcap.models.torch_compile import prepare_compile_env

    plan = prepare_compile_env(True)
    if plan.enabled:
        assert plan.mode == "cudagraphs"
        assert plan.torch_compile_kwargs["backend"] == "cudagraphs"
