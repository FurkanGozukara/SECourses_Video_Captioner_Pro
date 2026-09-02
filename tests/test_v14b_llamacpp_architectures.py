from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from vcap.models import llamacpp_install


def _fake_completed(stdout: str, *, returncode: int = 0) -> SimpleNamespace:
    return SimpleNamespace(stdout=stdout, stderr="", returncode=returncode)


def test_detect_cuda_architectures_parses_supported_compute_capabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append((command, kwargs))
        return _fake_completed("8.9\n9.0\n12.0\n10.0\n")

    monkeypatch.setattr(llamacpp_install.subprocess, "run", fake_run)

    assert llamacpp_install._detect_cuda_architectures() == [
        "89-real",
        "90-real",
        "120-real",
        "100-real",
    ]
    assert calls == [
        (
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            {
                "capture_output": True,
                "text": True,
                "encoding": "utf-8",
                "errors": "replace",
                "timeout": 15,
                "check": False,
            },
        )
    ]


def test_detect_cuda_architectures_deduplicates_multi_gpu_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        llamacpp_install.subprocess,
        "run",
        lambda *_args, **_kwargs: _fake_completed("8.9\n8.9\n 9.0 \n8.9\n"),
    )

    assert llamacpp_install._detect_cuda_architectures() == ["89-real", "90-real"]


@pytest.mark.parametrize(
    ("stdout", "returncode"),
    [("garbage\n8.x\n9\n", 0), ("8.9\n", 1)],
)
def test_detect_cuda_architectures_returns_empty_for_unusable_output(
    monkeypatch: pytest.MonkeyPatch,
    stdout: str,
    returncode: int,
) -> None:
    monkeypatch.setattr(
        llamacpp_install.subprocess,
        "run",
        lambda *_args, **_kwargs: _fake_completed(stdout, returncode=returncode),
    )

    assert llamacpp_install._detect_cuda_architectures() == []


def test_linux_build_commands_honor_verbatim_environment_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    override = "89-real;90-real"
    monkeypatch.setenv("VCAP_LLAMACPP_CUDA_ARCHS", override)
    monkeypatch.setattr(
        llamacpp_install,
        "_detect_cuda_architectures",
        lambda: pytest.fail("environment override must bypass detection"),
    )

    _clone, configure, _compile = llamacpp_install._linux_build_commands(tmp_path)

    assert f"-DCMAKE_CUDA_ARCHITECTURES={override}" in configure


def test_linux_build_commands_use_detected_architectures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("VCAP_LLAMACPP_CUDA_ARCHS", raising=False)
    monkeypatch.setattr(
        llamacpp_install,
        "_detect_cuda_architectures",
        lambda: ["89-real", "120-real"],
    )

    _clone, configure, _compile = llamacpp_install._linux_build_commands(tmp_path)

    assert "-DCMAKE_CUDA_ARCHITECTURES=89-real;120-real" in configure
