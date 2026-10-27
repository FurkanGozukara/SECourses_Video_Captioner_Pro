from __future__ import annotations

import base64
import hashlib
from importlib.metadata import FileHash, PackagePath
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from vcap.core import cuda_runtime


def test_child_environment_excludes_foreign_runtime_but_preserves_driver(tmp_path, monkeypatch):
    own = tmp_path / "venv-runtime"
    foreign = tmp_path / "system-cudnn"
    driver = tmp_path / "nvidia-driver"
    for path in (own, foreign, driver):
        path.mkdir()
    (foreign / "libcudnn_engines_tensor_ir.so.9").write_bytes(b"foreign")
    (driver / "libcuda.so.1").write_bytes(b"driver")
    monkeypatch.setattr(cuda_runtime, "runtime_library_dirs", lambda: [str(own)])
    variable = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
    original = {variable: os.pathsep.join([str(foreign), str(driver), str(own)]), "UNCHANGED": "yes"}

    result = cuda_runtime.cuda_runtime_env(original)

    assert result[variable].split(os.pathsep) == [str(own), str(driver)]
    assert result["UNCHANGED"] == "yes"
    assert str(foreign) in original[variable]


def test_discovery_uses_current_venv_not_other_environments(tmp_path, monkeypatch):
    root = tmp_path / "venv/site-packages"
    torch_lib = root / "torch/lib"
    toolkit_lib = root / "nvidia/cu13/lib"
    for path in (torch_lib, toolkit_lib):
        path.mkdir(parents=True)
    monkeypatch.setattr(cuda_runtime.sys, "prefix", str(tmp_path / "venv"))
    monkeypatch.setattr(cuda_runtime, "_distribution", lambda name: SimpleNamespace(locate_file=lambda name: root / name))

    assert cuda_runtime.runtime_library_dirs() == [str(torch_lib), str(toolkit_lib)]
    monkeypatch.setattr(cuda_runtime.sys, "prefix", str(tmp_path / "another-venv"))
    assert cuda_runtime.runtime_library_dirs() == []


@pytest.fixture
def cudnn_install(tmp_path, monkeypatch):
    root = tmp_path / "site-packages"
    library = root / "nvidia/cudnn/lib/libcudnn.so.9"
    library.parent.mkdir(parents=True)
    library.write_bytes(b"CUDA 13 cuDNN library")
    record = PackagePath("nvidia/cudnn/lib/libcudnn.so.9")
    digest = base64.urlsafe_b64encode(hashlib.sha256(library.read_bytes()).digest()).decode().rstrip("=")
    record.hash = FileHash(f"sha256={digest}")
    cudnn = SimpleNamespace(version="9.20.0.48", files=[record], locate_file=lambda name: root / name)
    packages = {"torch": SimpleNamespace(requires=['nvidia-cudnn-cu13==9.20.0.48; platform_system == "Linux"']),
                "nvidia-cudnn-cu13": cudnn}
    monkeypatch.setattr(cuda_runtime.sys, "platform", "linux")
    monkeypatch.setattr(cuda_runtime, "_distribution", packages.get)
    monkeypatch.setattr(cuda_runtime, "_backend_requirement", lambda: None)
    return library, packages


def test_detects_cudnn_overwrite_even_when_package_version_matches(cudnn_install):
    library, _ = cudnn_install
    assert cuda_runtime.runtime_problems(verify_files=True) == []
    library.write_bytes(b"CUDA 12 wheel overwrote the same path")
    assert any("differs from its wheel" in problem for problem in cuda_runtime.runtime_problems(verify_files=True))


def test_detects_extra_cudnn_sublibrary_and_overlapping_distribution(cudnn_install):
    library, packages = cudnn_install
    (library.parent / "libcudnn_engines_tensor_ir.so.9").write_bytes(b"newer sublibrary")
    packages["nvidia-cudnn-cu12"] = SimpleNamespace(version="9.27.0.42")
    problems = cuda_runtime.runtime_problems()
    assert any("nvidia-cudnn-cu12 is installed" in problem for problem in problems)
    assert any("Untracked cuDNN sublibrary" in problem for problem in problems)


def test_linux_preload_keeps_allocator_but_removes_foreign_cuda(monkeypatch):
    monkeypatch.setattr(cuda_runtime.sys, "platform", "linux")
    monkeypatch.setattr(cuda_runtime, "runtime_library_dirs", lambda: ["/venv/cuda"])
    result = cuda_runtime.cuda_runtime_env({"LD_PRELOAD": "/system/libcudnn.so.9:/usr/lib/libjemalloc.so"})
    assert result["LD_PRELOAD"] == "/usr/lib/libjemalloc.so"
