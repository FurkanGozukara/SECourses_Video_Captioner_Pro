from __future__ import annotations

import io
import os
from pathlib import Path
import stat

import pytest

from vcap.models import llamacpp_install


def test_linux_without_cuda_release_asset_chooses_source_build(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[Path, Path]] = []
    expected_server = tmp_path / "built" / "llama-server"

    monkeypatch.setattr(llamacpp_install, "APP_DIR", tmp_path)
    monkeypatch.setattr(llamacpp_install, "_external_install", lambda: None)
    monkeypatch.setattr(
        llamacpp_install,
        "_release_assets",
        lambda _progress: [
            {
                "name": f"llama-{llamacpp_install.LLAMACPP_TAG}-bin-ubuntu-x64.zip",
                "browser_download_url": "https://example.invalid/cpu.zip",
            }
        ],
    )
    monkeypatch.setattr(llamacpp_install, "_emit", lambda *args, **kwargs: None)
    monkeypatch.setattr(llamacpp_install.os, "name", "posix")
    monkeypatch.setattr(llamacpp_install.sys, "platform", "linux")

    def fake_build(root: Path, target: Path, **_kwargs: object) -> Path:
        calls.append((root, target))
        return expected_server

    monkeypatch.setattr(llamacpp_install, "_build_linux_cuda", fake_build)

    result = llamacpp_install.ensure_llamacpp()

    install_root = (tmp_path / "llamacpp").resolve(strict=False)
    assert result == expected_server
    assert calls == [(install_root, install_root / llamacpp_install.LLAMACPP_TAG)]


def test_linux_build_uses_expected_mocked_subprocess_commands(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "llamacpp"
    target = root / llamacpp_install.LLAMACPP_TAG
    nvcc = tmp_path / "cuda" / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.write_text("stub", encoding="utf-8")
    calls: list[tuple[list[str], dict[str, str]]] = []

    monkeypatch.setattr(llamacpp_install, "_find_nvcc", lambda: nvcc)
    monkeypatch.setattr(llamacpp_install, "_detect_cuda_architectures", lambda: [])
    monkeypatch.setattr(llamacpp_install, "_emit", lambda *args, **kwargs: None)

    class FakeProcess:
        def __init__(self) -> None:
            self.pid = 12345
            self.stdout = io.StringIO("mocked subprocess output\n")

        def poll(self) -> int:
            return 0

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return 0

    def fake_popen(command: list[str], **kwargs: object) -> FakeProcess:
        env = kwargs["env"]
        assert isinstance(env, dict)
        calls.append((list(command), {str(key): str(value) for key, value in env.items()}))
        if command[0:2] == ["cmake", "--build"]:
            build_bin = root / "build" / llamacpp_install.LLAMACPP_TAG / "bin"
            build_bin.mkdir(parents=True)
            for name in ("llama-server", "llama-mtmd-cli", "libllama.so", "libllama.so.1"):
                (build_bin / name).write_text(name, encoding="utf-8")
        return FakeProcess()

    monkeypatch.setattr(llamacpp_install.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        llamacpp_install,
        "_validate_install",
        lambda folder: folder / "llama-server",
    )
    monkeypatch.setattr(
        llamacpp_install,
        "_version_build",
        lambda _server: (llamacpp_install.LLAMACPP_BUILD, "version b10621"),
    )

    server = llamacpp_install._build_linux_cuda(
        root,
        target,
        progress_cb=None,
        cancel=None,
    )

    source = root / "src" / llamacpp_install.LLAMACPP_TAG
    build = root / "build" / llamacpp_install.LLAMACPP_TAG
    assert calls[0][0] == [
        "git",
        "clone",
        "--branch",
        llamacpp_install.LLAMACPP_TAG,
        "--depth",
        "1",
        "https://github.com/ggml-org/llama.cpp.git",
        str(source),
    ]
    assert calls[1][0] == [
        "cmake",
        "-S",
        str(source),
        "-B",
        str(build),
        "-DGGML_CUDA=ON",
        "-DGGML_NATIVE=OFF",
        "-DLLAMA_CURL=OFF",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    assert calls[2][0][0:5] == ["cmake", "--build", str(build), "--config", "Release"]
    assert calls[2][0][5:7] == ["-j", str(max(1, os.cpu_count() or 1))]
    assert calls[2][0][-3:] == ["--target", "llama-server", "llama-mtmd-cli"]
    assert calls[1][1]["CUDACXX"] == str(nvcc)
    assert calls[2][1]["CUDACXX"] == str(nvcc)
    assert server == target / "llama-server"
    assert {path.name for path in target.iterdir()} >= {
        "llama-server",
        "llama-mtmd-cli",
        "libllama.so",
        "libllama.so.1",
        "vcap_install.json",
    }
    if os.name != "nt":
        assert (target / "llama-server").stat().st_mode & stat.S_IXUSR
    build_log = root / "downloads" / llamacpp_install.LLAMACPP_TAG / "build.log"
    log_text = build_log.read_text(encoding="utf-8")
    assert "$ git clone --branch b10621 --depth 1" in log_text
    assert "-DGGML_CUDA=ON" in log_text
    assert "mocked subprocess output" in log_text


def test_find_nvcc_uses_path_then_standard_cuda_location(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path_nvcc = tmp_path / "path-cuda" / "nvcc"
    path_nvcc.parent.mkdir()
    path_nvcc.write_text("stub", encoding="utf-8")
    monkeypatch.setattr(llamacpp_install.shutil, "which", lambda _name: str(path_nvcc))
    assert llamacpp_install._find_nvcc() == path_nvcc.resolve(strict=False)

    fallback_nvcc = tmp_path / "usr-local-cuda" / "nvcc"
    fallback_nvcc.parent.mkdir()
    fallback_nvcc.write_text("stub", encoding="utf-8")
    monkeypatch.setattr(llamacpp_install.shutil, "which", lambda _name: None)
    assert llamacpp_install._find_nvcc(fallback_nvcc) == fallback_nvcc.resolve(strict=False)


def test_find_nvcc_failure_tells_user_to_install_cuda_toolkit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(llamacpp_install.shutil, "which", lambda _name: None)

    with pytest.raises(RuntimeError) as caught:
        llamacpp_install._find_nvcc(tmp_path / "missing" / "nvcc")

    message = str(caught.value)
    assert "CUDA toolkit compiler nvcc was not found" in message
    assert "driver alone is not enough" in message
    assert "System & Models" in message


def test_streaming_subprocess_failure_names_log_and_last_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FailedProcess:
        def __init__(self) -> None:
            self.pid = 12345
            self.stdout = io.StringIO("compiler failed clearly\n")

        def poll(self) -> int:
            return 2

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return 2

    monkeypatch.setattr(llamacpp_install.subprocess, "Popen", lambda *args, **kwargs: FailedProcess())
    monkeypatch.setattr(llamacpp_install, "_emit", lambda *args, **kwargs: None)
    log_path = tmp_path / "downloads" / "build.log"

    with pytest.raises(RuntimeError) as caught:
        llamacpp_install._run_streaming_command(
            ["cmake", "--build", "build"],
            cwd=tmp_path,
            env={},
            log_path=log_path,
            label="Building llama.cpp CUDA tools",
            progress_cb=None,
            cancel=None,
        )

    message = str(caught.value)
    assert "failed with exit code 2" in message
    assert f"See {log_path}" in message
    assert "compiler failed clearly" in message
    assert "compiler failed clearly" in log_path.read_text(encoding="utf-8")


def test_build_if_needed_cli_reuses_ensure_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    server = tmp_path / "llama-server"
    calls: list[bool] = []
    monkeypatch.setattr(llamacpp_install, "setup_utf8_stdio", lambda: None)

    def fake_ensure(*, force: bool = False) -> Path:
        calls.append(force)
        return server

    monkeypatch.setattr(llamacpp_install, "ensure_llamacpp", fake_ensure)

    assert llamacpp_install._main(["--build-if-needed"]) == 0
    assert calls == [False]
