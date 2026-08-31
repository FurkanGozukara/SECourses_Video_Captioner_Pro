from __future__ import annotations

import os
import threading
import time

import pytest

from vcap.pipeline.chat import ChatRequest
from vcap.pipeline.client import PipelineClient


def _request(history: list[dict[str, str]]) -> ChatRequest:
    return ChatRequest.from_dict(
        {
            "settings": {
                "model_key": "qwen3_omni_instruct_int4",
                "gpu_index": 0,
                "subprocess_mode": True,
                "keep_model_loaded": True,
                "idle_unload_minutes": 10,
            },
            "history": history,
            "media": [],
            "generation": {
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 0,
                "max_new_tokens": 64,
                "enable_thinking": False,
            },
        }
    )


def test_pipeline_client_chat_worker_streams_and_reuses_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VCAP_FAKE_CHAT", "1")
    monkeypatch.setenv("VCAP_SKIP_MODEL_ENSURE", "1")
    client = PipelineClient(subprocess_mode=True)
    try:
        first_events: list[dict] = []
        first = client.chat(
            _request([{"role": "user", "content": "What is in the image?"}]),
            first_events.append,
        )
        worker = client._worker
        assert worker is not None and worker.is_alive()
        assert first.finish_reason == "eos"
        assert first.text.startswith("Mock answer to:")
        assert any(event.get("ev") == "delta" for event in first_events)

        second_events: list[dict] = []
        second = client.chat(
            _request(
                [
                    {"role": "user", "content": "What is in the image?"},
                    {"role": "assistant", "content": first.text},
                    {"role": "user", "content": "Answer again in exactly five words."},
                ]
            ),
            second_events.append,
        )
        assert client._worker is worker and worker.is_alive()
        assert first.text in second.text
        assert second.finish_reason == "eos"
        assert len([event for event in second_events if event.get("ev") == "delta"]) >= 2
    finally:
        client.shutdown()
    assert not client._worker


def test_chat_import_path_stays_torch_free() -> None:
    command = (
        "import sys; import vcap.pipeline.chat, vcap.pipeline.client; "
        "assert 'torch' not in sys.modules; print('torch-free')"
    )
    completed = __import__("subprocess").run(
        [os.fspath(__import__("sys").executable), "-c", command],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "torch-free" in completed.stdout


def test_cooperative_chat_stop_retains_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VCAP_FAKE_CHAT", "1")
    monkeypatch.setenv("VCAP_FAKE_CHAT_DELAY", "0.3")
    client = PipelineClient(subprocess_mode=True)
    state: dict[str, object] = {}

    def execute() -> None:
        try:
            state["result"] = client.chat(
                _request([{"role": "user", "content": "Please keep generating."}])
            )
        except BaseException as exc:
            state["error"] = exc

    thread = threading.Thread(target=execute, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while (client._worker is None or not client._busy) and time.monotonic() < deadline:
        time.sleep(0.01)
    worker = client._worker
    try:
        assert worker is not None and worker.is_alive()
        client.cancel(force=False)
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert "error" not in state
        result = state["result"]
        assert getattr(result, "cancelled") is True
        assert client._worker is worker and worker.is_alive()

        follow_up = client.chat(
            _request([{"role": "user", "content": "A fresh request."}])
        )
        assert follow_up.finish_reason == "eos"
        assert client._worker is worker and worker.is_alive()
    finally:
        client.shutdown()
