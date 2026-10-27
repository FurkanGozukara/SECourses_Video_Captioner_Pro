from __future__ import annotations

import vcap.whisper.cuda_runtime as cuda_runtime


def test_whisper_uses_shared_runtime_and_configures_once(monkeypatch):
    directories = ["/venv/torch/lib", "/venv/nvidia/cu13/lib"]
    calls = []
    monkeypatch.setattr(cuda_runtime, "_CUDA_RUNTIME_CONFIGURED", False)
    monkeypatch.setattr(cuda_runtime, "configure_cuda_runtime", lambda: calls.append(1) or directories)
    assert cuda_runtime.enable_cuda_runtime_autodiscovery() == directories
    assert cuda_runtime.enable_cuda_runtime_autodiscovery() == directories
    assert calls == [1]


def test_cpu_remains_available_when_runtime_configuration_fails(monkeypatch):
    monkeypatch.setattr(cuda_runtime, "_CUDA_RUNTIME_CONFIGURED", False)
    monkeypatch.setattr(cuda_runtime, "configure_cuda_runtime", lambda: (_ for _ in ()).throw(OSError("missing DLL")))
    assert cuda_runtime.enable_cuda_runtime_autodiscovery() == []
