from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from vcap.core.python_runtime import project_python


@pytest.fixture(scope="module")
def isolated_app(tmp_path_factory):
    app = tmp_path_factory.mktemp("child_python") / "app with spaces ü"
    app.mkdir()
    venv = app / "venv"
    command = [sys.executable, "-m", "venv", "--without-pip"]
    if os.name != "nt":
        command.append("--symlinks")
    command.append(str(venv))
    hidden = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    subprocess.run(command, check=True, capture_output=True, timeout=60, **hidden)
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    purelib = subprocess.check_output(
        [str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
        text=True, timeout=30, **hidden,
    ).strip()
    (Path(purelib) / "vcap_venv_dependency.py").write_text("VALUE = 'venv only'\n", encoding="utf-8")
    return app, python


@pytest.mark.parametrize("mode", ["default", "explicit", "quoted"])
def test_child_interpreter_keeps_venv_prefix_and_dependencies(isolated_app, mode) -> None:
    app, python = isolated_app
    override = None if mode == "default" else str(python)
    if mode == "quoted":
        override = f'"{python}"'
    selected = project_python(app, override)
    assert selected == python
    hidden = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    output = subprocess.check_output(
        [str(selected), "-c", "import json,sys,vcap_venv_dependency as d; print(json.dumps({'prefix':sys.prefix,'value':d.VALUE}))"],
        text=True, timeout=30, **hidden,
    )
    result = json.loads(output)
    assert Path(result["prefix"]) == app / "venv"
    assert result["value"] == "venv only"
    if os.name != "nt":
        assert selected.is_symlink()
        assert selected != selected.resolve()


def test_downloader_and_whisper_select_the_same_unresolved_venv(isolated_app, monkeypatch) -> None:
    from vcap.models import downloads
    from vcap.whisper import client

    app, python = isolated_app
    monkeypatch.setattr(downloads, "APP_DIR", app)
    monkeypatch.setattr(client, "APP_DIR", app)
    assert downloads._venv_python() == python
    assert client._python_executable(None) == str(python)
    assert client._python_executable(str(python)) == str(python)


def test_fallback_keeps_running_interpreter_path(tmp_path: Path, monkeypatch) -> None:
    executable = tmp_path / "custom venv" / "python"
    executable.parent.mkdir()
    executable.touch()
    monkeypatch.setattr(sys, "executable", str(executable))
    assert project_python(tmp_path / "no bundled venv") == executable


def test_automatic_model_download_uses_real_child_environment(isolated_app, monkeypatch) -> None:
    from vcap.models import downloads

    app, python = isolated_app
    ready = app / "downloaded.json"
    script = app.parent / "Models_Downloader.py"
    script.write_text(
        "import json,sys\n"
        "from pathlib import Path\n"
        "import vcap_venv_dependency as dependency\n"
        f"Path({str(ready)!r}).write_text(json.dumps({{'prefix':sys.prefix,'value':dependency.VALUE}}))\n"
        "print('VCAP_STATUS ' + json.dumps({'key':sys.argv[-1],'state':'ready','fraction':1.0,'message':'verified'}), flush=True)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(downloads, "APP_DIR", app)
    monkeypatch.setattr(downloads, "_find_downloader", lambda: script)
    monkeypatch.setattr(downloads, "variant_is_ready", lambda _key: (ready.is_file(), "fixture verified"))
    messages = []
    success, detail = downloads.ensure_model("qwen3_omni_instruct_int4", lambda message, *_args: messages.append(message))
    assert success and "fixture verified" in detail
    result = json.loads(ready.read_text(encoding="utf-8"))
    assert Path(result["prefix"]) == app / "venv"
    assert result["value"] == "venv only"
    assert any("verified (100%)" in message for message in messages)


@pytest.mark.parametrize("layout", ["bundled", "legacy", "both", "override"])
def test_downloader_source_selection_updates_with_git(tmp_path: Path, monkeypatch, layout) -> None:
    from vcap.models import downloads

    app = tmp_path / "app"
    app.mkdir()
    bundled = app / "Models_Downloader.py"
    legacy = tmp_path / "Models_Downloader.py"
    override = tmp_path / "custom downloader.py"
    monkeypatch.setattr(downloads, "APP_DIR", app)
    monkeypatch.delenv("SECOURSES_VCAP_DOWNLOADER", raising=False)
    if layout in {"bundled", "both", "override"}:
        bundled.touch()
    if layout in {"legacy", "both", "override"}:
        legacy.touch()
    if layout == "override":
        override.touch()
        monkeypatch.setenv("SECOURSES_VCAP_DOWNLOADER", str(override))
    expected = override if layout == "override" else legacy if layout == "legacy" else bundled
    assert downloads._find_downloader() == expected
