from __future__ import annotations

import os
import queue
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from vcap.core.logs import get_log
from vcap.pipeline.client import PipelineClient
from vcap.pipeline.job import InputItem, JobResult, JobSpec, OutputSpec
from vcap.ui.app import build_app
from vcap.ui.components import poll_log_value
from vcap.ui.tabs.recover_tab import refresh_recent_run_choices


@pytest.fixture(scope="module")
def full_app():
    app = build_app()
    try:
        yield app
    finally:
        app.vcap_context.pipeline_client.shutdown()


def _dependency_for_function(app: Any, name: str) -> dict[str, Any]:
    config = app.get_config_file()
    matches = [
        dependency
        for dependency in config["dependencies"]
        if getattr(app.fns[dependency["id"]].fn, "__name__", "") == name
    ]
    assert len(matches) == 1, name
    return matches[0]


def test_log_poll_recovers_a_future_cursor_with_the_recent_tail() -> None:
    app_log = get_log()
    marker = f"F4 stale cursor marker {time.time_ns()}"
    app_log.log(marker, scope="test", console=False)
    current_revision = app_log.revision

    rendered, revision = poll_log_value(
        app_log,
        current_revision + 100,
        "stale browser contents",
    )

    assert revision == current_revision
    assert marker in rendered
    assert "stale browser contents" not in rendered
    assert len(rendered.splitlines()) <= 300


def test_pipeline_cancel_keeps_forwarding_into_the_same_app_log(tmp_path: Path) -> None:
    class FakeWorker:
        def __init__(self) -> None:
            self.alive = True
            self.returncode = 0
            self.sent: list[dict[str, Any]] = []

        def is_alive(self) -> bool:
            return self.alive

        def send(self, payload: dict[str, Any]) -> None:
            self.sent.append(dict(payload))

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            self.alive = False
            return 0

        def kill_tree(self, grace: float = 0.0) -> None:
            del grace
            self.alive = False

    class LogSink:
        def on_log(
            self,
            message: str,
            level: str = "info",
            scope: str | None = None,
        ) -> None:
            del message, level, scope

        def on_progress(self, _event: Any) -> None:
            pass

        def on_item(self, _event: Any) -> None:
            pass

    settings = {
        "model_key": "qwen3_omni_instruct_int4",
        "prompt_preset_id": "custom",
        "user_prompt": "Describe this.",
        "subprocess_mode": True,
        "gpu_index": 0,
    }
    spec = JobSpec.from_settings(
        settings,
        [InputItem(path="", kind="text", text_prompt_only=True, text="Describe this.")],
        OutputSpec(outputs_root=tmp_path),
    )
    client = PipelineClient(subprocess_mode=True)
    worker = FakeWorker()
    client._worker = worker  # type: ignore[assignment]
    client._worker_gpu = spec.runtime.gpu_index
    client._worker_compile = spec.runtime.compile
    client._events = queue.Queue()
    outcome: dict[str, Any] = {}
    app_log = get_log()
    before = app_log.revision

    def run() -> None:
        outcome["result"] = client.run_job(spec, LogSink())

    thread = threading.Thread(target=run, name="f4-cancel-reproduction")
    thread.start()
    deadline = time.monotonic() + 3.0
    while not client._busy and time.monotonic() < deadline:
        time.sleep(0.01)
    assert client._busy

    client.cancel()
    client._events.put(
        {
            "ev": "log",
            "text": "cancelled job terminal log",
            "scope": "pipeline",
        }
    )
    client._events.put(
        {
            "ev": "result",
            "job_result": {
                "items": [],
                "counts": {"cancelled": 1},
                "run_dir": str(tmp_path),
                "metadata_path": str(tmp_path / "metadata.json"),
                "elapsed": 0.1,
            },
        }
    )
    thread.join(timeout=3.0)
    try:
        assert not thread.is_alive()
        assert outcome["result"].counts["cancelled"] == 1
        assert any(payload.get("cmd") == "cancel" for payload in worker.sent)
        marker = f"after cancel {time.time_ns()}"
        app_log.log(marker, scope="test", console=False)
        lines, revision = app_log.snapshot(before)
        assert revision == app_log.revision
        assert any("cancelled job terminal log" in line for line in lines)
        assert any(marker in line for line in lines)
    finally:
        client.shutdown()


def test_editor_and_recover_registry_events_are_wired_after_full_build(full_app: Any) -> None:
    ctx = full_app.vcap_context
    registry_ids = [component._id for component in ctx.settings_registry.components()]
    selected = _dependency_for_function(full_app, "regenerate_handler")
    regenerate_all = _dependency_for_function(full_app, "regenerate_all_handler")
    recover_load = _dependency_for_function(full_app, "load_handler")

    assert selected["inputs"][-len(registry_ids) :] == registry_ids
    assert len(selected["inputs"]) == len(registry_ids) + 4
    assert regenerate_all["inputs"][-len(registry_ids) :] == registry_ids
    assert len(regenerate_all["inputs"]) == len(registry_ids) + 5
    assert recover_load["inputs"][-len(registry_ids) :] == registry_ids
    assert len(recover_load["inputs"]) == len(registry_ids) + 3
    assert ctx.states["editor_regeneration_binding"]["registry_components"] == ctx.settings_registry.components()
    assert ctx.states["recover_wire_binding"]["registry_components"] == ctx.settings_registry.components()


def test_model_change_returns_immediately_and_one_run_is_submitted(
    full_app: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ctx = full_app.vcap_context
    client = ctx.pipeline_client
    release_started = threading.Event()
    allow_release = threading.Event()
    submitted: list[JobSpec] = []
    recorded: list[str] = []
    selected = "qwen3_omni_thinking_int4"

    def record_variant(variant_key: str) -> bool:
        recorded.append(str(variant_key))
        return False

    def slow_release(variant_key: str) -> dict[str, Any]:
        release_started.set()
        allow_release.wait(timeout=3.0)
        return {
            "released": None,
            "report": None,
            "selected": variant_key,
        }

    def fake_run_job(spec: JobSpec, _sink: Any, _token: Any) -> JobResult:
        submitted.append(spec)
        return JobResult(
            items=[],
            counts={"done": 0, "skipped": 0, "failed": 0, "cancelled": 0},
            run_dir=str(tmp_path),
            metadata_path=str(tmp_path / "metadata.json"),
            elapsed=0.01,
        )

    monkeypatch.setattr(client, "record_variant_selection", record_variant)
    monkeypatch.setattr(client, "release_recorded_variant", slow_release)
    monkeypatch.setattr(client, "run_job", fake_run_job)

    changed = ctx.states["caption_model_change_handler"]
    started_at = time.monotonic()
    changed(selected)
    elapsed = time.monotonic() - started_at
    assert elapsed < 0.5
    assert release_started.wait(timeout=1.0)

    settings = ctx.settings_registry.defaults()
    settings["model_key"] = selected
    values = ctx.settings_registry.dict_to_values(settings)
    run_caption = ctx.states["caption_run_handler"]
    try:
        updates = list(run_caption(*values, [], "upload", "unknown"))
    finally:
        allow_release.set()

    assert recorded == [selected]
    assert len(submitted) == 1
    assert submitted[0].model.variant_key == selected
    assert "Waiting for the previous model to unload" in updates[0][1]


def test_recent_run_refresh_selects_the_newest_and_is_hooked_to_jobs(
    tmp_path: Path,
    full_app: Any,
) -> None:
    old = tmp_path / "0001_old" / "metadata.json"
    new = tmp_path / "0002_new" / "metadata.json"
    old.parent.mkdir()
    new.parent.mkdir()
    old.write_text("{}", encoding="utf-8")
    new.write_text("{}", encoding="utf-8")
    os.utime(old, ns=(1_000_000_000, 1_000_000_000))
    os.utime(new, ns=(2_000_000_000, 2_000_000_000))

    choices, selected = refresh_recent_run_choices(tmp_path)
    assert choices[0][1] == str(new)
    assert selected == str(new)

    ctx = full_app.vcap_context
    config = full_app.get_config_file()
    recent_outputs = [
        component._id
        for component in ctx.states["recover_recent_binding"]["refresh_outputs"]
    ]
    caption_event = next(
        dependency
        for dependency in config["dependencies"]
        if dependency.get("api_name") == "caption"
    )
    assert any(
        dependency.get("trigger_after") == caption_event["id"]
        and dependency["outputs"] == recent_outputs
        for dependency in config["dependencies"]
    )
    recover_tab_id = ctx.states["recover_tab_component"]._id
    assert any(
        any(
            target_id == recover_tab_id and target_event == "select"
            for target_id, target_event in dependency.get("targets", [])
        )
        and dependency["outputs"] == recent_outputs
        for dependency in config["dependencies"]
    )
