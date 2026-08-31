from __future__ import annotations

import json
import sys
import time

import pytest

import vcap.core.subprocess_runner as subprocess_runner
from vcap.core.subprocess_runner import CancelToken, WorkerProcess, build_child_env, iter_json_lines


def test_json_lines_and_worker_events(tmp_path) -> None:
    code = (
        "import json,sys; "
        "print(json.dumps({'ev':'ready','value':1}), flush=True); "
        "print('plain output', flush=True); "
        "line=sys.stdin.readline(); print(line, end='', flush=True)"
    )
    worker = WorkerProcess().start(
        [sys.executable, "-u", "-c", code],
        cwd=tmp_path,
        env=build_child_env(),
        name="echo",
    )
    worker.send({"ev": "echo", "text": "vöyager 日本語"})
    events = list(worker.events())
    assert worker.wait(timeout=5) == 0
    assert events[0] == {"ev": "ready", "value": 1}
    assert events[1] == {"ev": "stdout", "text": "plain output"}
    assert events[2]["ev"] == "echo"

    parsed = list(iter_json_lines([json.dumps({"ev": "ok"}) + "\n", "not json\n", "{}\n"]))
    assert parsed == [
        {"ev": "ok"},
        {"ev": "stdout", "text": "not json"},
        {"ev": "stdout", "text": "{}"},
    ]


def test_kill_tree_and_cancel_token(tmp_path) -> None:
    code = (
        "import subprocess,sys,time; "
        "subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        "time.sleep(60)"
    )
    worker = WorkerProcess().start([sys.executable, "-c", code], cwd=tmp_path, name="sleep")
    time.sleep(0.3)
    assert worker.is_alive()
    worker.kill_tree(grace=0.2)
    assert not worker.is_alive()

    token = CancelToken()
    assert not token.is_armed() and not token.is_cancelled()
    token.arm_confirmation(1)
    assert token.is_armed()
    token.cancel()
    assert token.is_cancelled() and not token.is_armed()
    token.reset()
    assert not token.is_cancelled()


def test_windows_child_env_removes_expandable_segments_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subprocess_runner.os, "name", "nt")
    monkeypatch.setenv(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True,garbage_collection_threshold:0.8",
    )
    env = build_child_env(
        extra={"PYTORCH_ALLOC_CONF": "max_split_size_mb:512,expandable_segments:True"}
    )

    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "garbage_collection_threshold:0.8"
    assert env["PYTORCH_ALLOC_CONF"] == "max_split_size_mb:512"
