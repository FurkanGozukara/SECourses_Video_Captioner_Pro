from __future__ import annotations

import os
from pathlib import Path

import pytest

import vcap.whisper.cuda_runtime as cuda_runtime


def test_discovers_nvidia_site_package_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "site-packages"
    expected = []
    for package in ("cublas", "cudnn", "cuda_runtime"):
        directory = root / "nvidia" / package / "bin"
        directory.mkdir(parents=True)
        expected.append(str(directory.resolve()))
    monkeypatch.setattr(cuda_runtime, "_candidate_site_package_roots", lambda: [str(root)])
    monkeypatch.setattr(cuda_runtime, "_discover_dirs_from_nvidia_modules", lambda: [])
    monkeypatch.setattr(cuda_runtime, "_discover_dirs_from_cuda_env", lambda: [])

    discovered = cuda_runtime.discover_cuda_runtime_dirs()

    assert set(expected).issubset(discovered)


def test_enable_is_idempotent_and_prepends_runtime_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "nvidia" / "cublas" / "bin"
    runtime.mkdir(parents=True)
    preloads: list[list[str]] = []
    registrations: list[list[str]] = []
    monkeypatch.setattr(cuda_runtime, "_CUDA_RUNTIME_CONFIGURED", False)
    monkeypatch.setattr(cuda_runtime, "_CONFIGURED_DIRECTORIES", [])
    monkeypatch.setattr(cuda_runtime, "discover_cuda_runtime_dirs", lambda: [str(runtime)])
    monkeypatch.setattr(
        cuda_runtime, "_preload_libraries", lambda directories: preloads.append(list(directories))
    )
    monkeypatch.setattr(
        cuda_runtime,
        "_register_windows_dll_directories",
        lambda directories: registrations.append(list(directories)),
    )
    monkeypatch.setattr(cuda_runtime.platform, "system", lambda: "Windows")
    monkeypatch.setenv("PATH", str(tmp_path / "existing"))

    first = cuda_runtime.enable_cuda_runtime_autodiscovery()
    second = cuda_runtime.enable_cuda_runtime_autodiscovery()

    assert first == second == [str(runtime)]
    assert os.environ["PATH"].split(";", 1)[0] == str(runtime)
    assert preloads == [[str(runtime)]]
    assert registrations == [[str(runtime)]]


def test_enable_never_raises_when_discovery_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cuda_runtime, "_CUDA_RUNTIME_CONFIGURED", False)
    monkeypatch.setattr(
        cuda_runtime,
        "discover_cuda_runtime_dirs",
        lambda: (_ for _ in ()).throw(OSError("broken layout")),
    )
    assert cuda_runtime.enable_cuda_runtime_autodiscovery() == []
