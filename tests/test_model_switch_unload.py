"""Model switches must release the previous model completely.

Covers the loader release contract (`unload_model`, `UnloadReport`, `ModelCache.unload`),
the compiled-state release, the worker protocol (`unload` with `unless_variant`), the
pipeline client (`release_model`, `select_variant`, deferred release after a job), and the
Caption tab wiring that triggers a release when the model selection changes.
"""

from __future__ import annotations

import gc
import io
import json
import threading
import time
import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vcap.pipeline.job import InputItem, JobSpec, OutputSpec


# --------------------------------------------------------------------------- helpers


class _Handle:
    def __init__(self) -> None:
        self.removed = False

    def remove(self) -> None:
        self.removed = True


class _Manager:
    def __init__(self) -> None:
        self.removed = False

    def remove(self) -> None:
        self.removed = True


class _Held:
    """Weak-referenceable stand-in for a model object (SimpleNamespace is not)."""


class _Backend:
    def __init__(self) -> None:
        self.stopped = 0

    def stop(self) -> None:
        self.stopped += 1


class _DuckModule:
    """Duck-typed module like the doubles in test_c7_torch_compile_fallback."""

    def __init__(self, children: list[Any] | None = None) -> None:
        self._children = list(children or [])

    def modules(self) -> list[Any]:
        return [self, *self._children]

    def forward(self) -> str:
        return "eager"


def _snapshot_sequence(values: list[float]):
    """``resource_snapshot`` stand-in yielding successive VRAM readings, then the last forever."""

    calls: list[int] = []

    def snapshot(gpu_index: int = 0) -> dict[str, float]:
        calls.append(int(gpu_index))
        used = float(values[min(len(calls) - 1, len(values) - 1)])
        return {"vram_used_gb": used, "vram_total_gb": 32.0, "vram_free_gb": 32.0 - used}

    snapshot.calls = calls  # type: ignore[attr-defined]
    return snapshot


def _loaded(
    model: Any,
    *,
    backend: str = "transformers",
    key: str = "timechat_int4",
    family: str = "timechat",
    device: str = "cpu",
):
    from vcap.models.loader import LoadReport, LoadedModel

    report = LoadReport(
        seconds=1.0,
        peak_vram_gb=1.0,
        checkpoint_bytes=1,
        attention="sdpa",
        device_map={"": device},
    )
    spec = SimpleNamespace(family=family, label=family)
    variant = SimpleNamespace(
        key=key,
        backend=backend,
        scheme="gguf" if backend == "llamacpp" else "int4_convrot_w4a8",
        label=key,
    )
    return LoadedModel(model, object(), spec, variant, report, device, None, "sdpa")


def _cache_fakes(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Patch ``load_model``/``unload_model`` with recorders and return the call order."""

    from vcap.models import loader

    order: list[str] = []

    def fake_load(variant_key: str, **kwargs: Any) -> Any:
        del kwargs
        order.append(f"load:{variant_key}")
        return SimpleNamespace(
            model=object(),
            processor=None,
            variant=SimpleNamespace(key=variant_key, backend="transformers"),
            spec=SimpleNamespace(family="timechat"),
            load_report=SimpleNamespace(activation_estimate_bytes=0, block_swap=None),
        )

    def fake_unload(loaded: Any, **kwargs: Any) -> Any:
        del kwargs
        order.append(f"unload:{loaded.variant.key}")
        loaded.model = None
        return loader.UnloadReport(0.0, 8.0, 1.5, 6.5, variant_key=loaded.variant.key)

    monkeypatch.setattr(loader, "load_model", fake_load)
    monkeypatch.setattr(loader, "unload_model", fake_unload)
    return order


def _settings(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "model_key": "timechat_int4",
        "prompt_preset_id": "custom",
        "user_prompt": "Describe this input.",
        "system_prompt": None,
        "fps": 2.0,
        "max_frames": 8,
        "max_pixels": 131_072,
        "min_pixels": 4_096,
        "use_audio_in_video": False,
        "output_formats": ["txt"],
        "keep_model_loaded": True,
        "idle_unload_minutes": 0,
        "gpu_index": 0,
    }
    values.update(overrides)
    return values


def _resident_runner(monkeypatch: pytest.MonkeyPatch, key: str | None) -> tuple[dict[str, Any], list[Any]]:
    """Patch the runner's resident-model functions with an in-memory stand-in."""

    import vcap.pipeline.runner as runner

    resident: dict[str, Any] = {"key": key}
    calls: list[Any] = []

    def loaded_variant_key() -> str | None:
        return resident["key"]

    def unload_cached_model(unless_variant: str | None = None) -> dict[str, Any] | None:
        calls.append(unless_variant)
        if resident["key"] is None or resident["key"] == unless_variant:
            return None
        released, resident["key"] = resident["key"], None
        return {"variant_key": released, "released": True, "freed_vram_gb": 6.5}

    monkeypatch.setattr(runner, "loaded_variant_key", loaded_variant_key)
    monkeypatch.setattr(runner, "unload_cached_model", unload_cached_model)
    return resident, calls


# --------------------------------------------------------------------------- loader


def test_unload_report_carries_release_details_and_is_json_safe() -> None:
    from vcap.models.loader import UnloadReport

    report = UnloadReport(
        0.5,
        10.0,
        2.0,
        8.0,
        variant_key="timechat_int4",
        backend="transformers",
        released=True,
        host_before_gb=12.0,
        host_after_gb=5.0,
        notes=("settled",),
    )
    data = report.to_dict()
    assert data["variant_key"] == "timechat_int4"
    assert data["backend"] == "transformers"
    assert data["released"] is True
    assert data["freed_vram_gb"] == 8.0
    assert data["notes"] == ["settled"]
    assert json.loads(json.dumps(data)) == data

    legacy = UnloadReport(0.1, 1.0, 1.0, 0.0)
    assert legacy.variant_key is None
    assert legacy.backend == ""
    assert legacy.released is True
    assert legacy.notes == ()


def test_unload_model_without_a_model_reports_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    from vcap.models import loader

    snapshot = _snapshot_sequence([3.0])
    monkeypatch.setattr(loader, "resource_snapshot", snapshot)
    started = time.perf_counter()
    report = loader.unload_model(None)
    assert time.perf_counter() - started < 0.5
    assert report.variant_key is None
    assert report.backend == ""
    assert report.released is True
    assert report.freed_vram_gb == 0.0


def test_unload_model_releases_transformers_runtime_completely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    from vcap.models import loader, torch_compile

    class Decoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(4, 4)

        def forward(self, x: Any) -> Any:
            return self.proj(x)

    class Root(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = Decoder()
            self.lm_head = torch.nn.Linear(4, 4)

        def forward(self, x: Any) -> Any:
            return self.lm_head(self.model(x))

    manager = _Manager()
    hook = _Handle()
    resets: list[bool] = []
    monkeypatch.setattr(torch_compile, "_reset_compiler_runtime", lambda: resets.append(True))
    monkeypatch.setattr(loader, "resource_snapshot", _snapshot_sequence([9.0, 2.5]))

    def build() -> tuple[Any, weakref.ref]:
        model = Root()
        # What apply_compile() leaves behind on the decoder and the root.
        original = model.model.forward
        model.model.forward = lambda *args, **kwargs: original(*args, **kwargs)
        model.model._vcap_compiled = True
        model.model._vcap_original_forward = original
        model.model._vcap_compile_plan = object()
        model._vcap_compile_plan = object()
        model._vcap_compile_family = "timechat"
        # What load_model() attaches at runtime.
        model._vcap_block_swap_manager = manager
        model._vcap_last_token_logits_hook = hook
        model._vcap_last_token_logits = True
        return _loaded(model), weakref.ref(model)

    loaded, ref = build()
    report = loader.unload_model(loaded)

    assert manager.removed, "block-swap manager must be removed"
    assert hook.removed, "last-token logits hook must be removed"
    assert loaded.model is None and loaded.processor is None
    gc.collect()
    assert ref() is None, "no reference to the released model may survive"
    assert resets, "Dynamo/Inductor caches must be reset when a compiled model is released"
    assert report.variant_key == "timechat_int4"
    assert report.backend == "transformers"
    assert report.released is True
    assert report.vram_before_gb == pytest.approx(9.0)
    assert report.vram_after_gb == pytest.approx(2.5)
    assert report.freed_vram_gb == pytest.approx(6.5)
    assert report.host_before_gb > 0.0 and report.host_after_gb > 0.0
    assert not any("still referenced" in note for note in report.notes)


def test_unload_model_clears_convrot_device_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    from vcap.models import loader
    from vcap.models.quant import convrot

    monkeypatch.setattr(loader, "resource_snapshot", _snapshot_sequence([2.0, 2.0]))
    monkeypatch.setitem(convrot._HADAMARD_CACHE, (4, "cpu", torch.float32), torch.eye(4))
    monkeypatch.setitem(convrot._DECODE_DISPATCH_CACHE, ("cpu", 4), torch.arange(4))
    assert convrot._HADAMARD_CACHE and convrot._DECODE_DISPATCH_CACHE

    loader.unload_model(_loaded(torch.nn.Linear(2, 2)))

    assert convrot._HADAMARD_CACHE == {}
    assert convrot._DECODE_DISPATCH_CACHE == {}
    assert convrot.clear_device_caches() == 0


def test_unload_model_reports_lingering_references(monkeypatch: pytest.MonkeyPatch) -> None:
    from vcap.models import loader

    monkeypatch.setattr(loader, "resource_snapshot", _snapshot_sequence([9.0, 9.0]))
    held = _Held()
    loaded = _loaded(held)
    report = loader.unload_model(loaded)
    assert loaded.model is None
    assert report.released is False
    assert any("still referenced" in note for note in report.notes)
    assert held is not None  # the test itself is the lingering referrer


def test_unload_model_stops_llama_server_and_waits_for_vram_to_settle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vcap.models import loader

    snapshot = _snapshot_sequence([24.0, 24.0, 15.0, 8.0, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5])
    monkeypatch.setattr(loader, "resource_snapshot", snapshot)
    backend = _Backend()
    loaded = _loaded(
        backend,
        backend="llamacpp",
        key="qwen3_omni_instruct_gguf_q4",
        family="qwen3_omni_instruct",
        device="cuda:0",
    )
    report = loader.unload_model(loaded, wait_s=5.0)
    assert backend.stopped == 1
    assert loaded.model is None
    assert report.backend == "llamacpp"
    assert report.variant_key == "qwen3_omni_instruct_gguf_q4"
    assert report.vram_before_gb == pytest.approx(24.0)
    assert report.vram_after_gb == pytest.approx(2.5)
    assert report.freed_vram_gb == pytest.approx(21.5)
    assert len(snapshot.calls) >= 5, "the release must poll until the driver returned the VRAM"
    assert report.seconds < 5.0


def test_unload_model_settle_wait_honors_the_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    from vcap.models import loader

    # Usage keeps falling slowly: the wait must stop at ``wait_s`` and say so.
    snapshot = _snapshot_sequence([20.0] + [20.0 - 0.5 * step for step in range(1, 200)])
    monkeypatch.setattr(loader, "resource_snapshot", snapshot)
    loaded = _loaded(_Backend(), backend="llamacpp", key="qwen3_omni_instruct_gguf_q8", family="qwen3_omni_instruct")
    started = time.perf_counter()
    report = loader.unload_model(loaded, wait_s=0.4)
    elapsed = time.perf_counter() - started
    assert 0.3 <= elapsed < 3.0
    assert any("settle" in note.casefold() or "timed out" in note.casefold() for note in report.notes)


def test_model_cache_switch_unloads_previous_model_before_loading_next(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vcap.models import loader

    order = _cache_fakes(monkeypatch)
    cache = loader.ModelCache()
    first = cache.load("timechat_int4")
    assert cache.loaded_variant_key() == "timechat_int4"
    second = cache.load("avocado_int4")
    assert second is not first
    assert order == ["load:timechat_int4", "unload:timechat_int4", "load:avocado_int4"]
    assert cache.loaded_variant_key() == "avocado_int4"
    assert cache.load("avocado_int4") is second
    assert order[-1] == "load:avocado_int4"


def test_model_cache_unload_unless_variant_keeps_a_matching_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vcap.models import loader

    order = _cache_fakes(monkeypatch)
    cache = loader.ModelCache()
    assert cache.unload(unless_variant="avocado_int4") is None
    cache.load("timechat_int4")
    assert cache.unload(unless_variant="timechat_int4") is None
    assert cache.loaded is not None
    assert order == ["load:timechat_int4"]
    report = cache.unload(unless_variant="avocado_int4")
    assert report is not None and report.variant_key == "timechat_int4"
    assert cache.loaded is None
    assert cache.loaded_variant_key() is None
    assert order == ["load:timechat_int4", "unload:timechat_int4"]
    assert cache.unload() is None


def test_model_cache_leaves_nothing_behind_when_the_new_load_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vcap.models import loader

    order = _cache_fakes(monkeypatch)
    cache = loader.ModelCache()
    cache.load("timechat_int4")
    real_load = loader.load_model

    def failing_load(variant_key: str, **kwargs: Any) -> Any:
        if variant_key == "avocado_int4":
            order.append(f"load:{variant_key}")
            raise RuntimeError("synthetic load failure")
        return real_load(variant_key, **kwargs)

    monkeypatch.setattr(loader, "load_model", failing_load)
    with pytest.raises(RuntimeError, match="synthetic load failure"):
        cache.load("avocado_int4")
    assert cache.loaded is None and cache.loaded_variant_key() is None
    assert order == ["load:timechat_int4", "unload:timechat_int4", "load:avocado_int4"]


# --------------------------------------------------------------------------- torch_compile


def test_release_compiled_model_restores_forwards_and_resets_compiler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vcap.models import torch_compile

    convrot = _DuckModule()
    convrot._vcap_original_disabled_forward = convrot.forward
    convrot.forward = lambda: "disabled"
    root = _DuckModule([convrot])
    original = root.forward
    root.forward = lambda: "compiled"
    root._vcap_compiled = True
    root._vcap_original_forward = original
    root._vcap_compile_plan = object()
    root._vcap_compile_family = "timechat"
    root._vcap_compile_requested_mode = "default"
    root._vcap_compile_disabled = False

    resets: list[bool] = []
    monkeypatch.setattr(torch_compile, "_reset_compiler_runtime", lambda: resets.append(True))

    assert torch_compile.release_compiled_model(root) == 2
    assert root.forward() == "eager"
    assert convrot.forward() == "eager"
    for name in (
        "_vcap_compiled",
        "_vcap_original_forward",
        "_vcap_compile_plan",
        "_vcap_compile_family",
        "_vcap_compile_requested_mode",
        "_vcap_compile_disabled",
    ):
        assert name not in vars(root), name
    assert "_vcap_original_disabled_forward" not in vars(convrot)
    assert resets == [True]

    assert torch_compile.release_compiled_model(root) == 0
    assert torch_compile.release_compiled_model(SimpleNamespace()) == 0
    assert torch_compile.release_compiled_model(None) == 0
    assert resets == [True], "nothing to restore means no second reset"


# --------------------------------------------------------------------------- runner / worker


def test_unload_cached_model_returns_report_and_honors_unless_variant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vcap.pipeline.runner as runner
    from vcap.models import loader

    order = _cache_fakes(monkeypatch)
    monkeypatch.setattr(loader, "MODEL_CACHE", loader.ModelCache())
    loader.MODEL_CACHE.load("timechat_int4")
    assert runner.loaded_variant_key() == "timechat_int4"

    assert runner.unload_cached_model(unless_variant="timechat_int4") is None
    assert runner.loaded_variant_key() == "timechat_int4"

    outcome = runner.unload_cached_model(unless_variant="avocado_int4")
    assert outcome is not None
    assert outcome["variant_key"] == "timechat_int4"
    assert outcome["released"] is True
    assert outcome["freed_vram_gb"] == pytest.approx(6.5)
    assert runner.loaded_variant_key() is None
    assert runner.unload_cached_model() is None
    assert order == ["load:timechat_int4", "unload:timechat_int4"]


def test_unload_cached_model_handles_the_fake_captioner(monkeypatch: pytest.MonkeyPatch) -> None:
    import vcap.pipeline.runner as runner

    monkeypatch.setattr(runner, "_FAKE_CAPTIONER", object())
    monkeypatch.setattr(runner, "_FAKE_VARIANT", "timechat_int4")
    assert runner.loaded_variant_key() == "timechat_int4"
    assert runner.unload_cached_model(unless_variant="timechat_int4") is None
    assert runner.loaded_variant_key() == "timechat_int4"
    outcome = runner.unload_cached_model(unless_variant="avocado_int4")
    assert outcome == {"variant_key": "timechat_int4", "released": True, "fake": True}
    assert runner.loaded_variant_key() is None
    assert runner._FAKE_CAPTIONER is None


def test_worker_unload_command_reports_skip_and_release(monkeypatch: pytest.MonkeyPatch) -> None:
    from vcap.pipeline.worker import _ProtocolWriter, _Server

    resident, _calls = _resident_runner(monkeypatch, "timechat_int4")
    stream = io.StringIO()
    server = _Server(_ProtocolWriter(stream))
    server.unload({"cmd": "unload", "unless_variant": "timechat_int4"})
    assert resident["key"] == "timechat_int4"
    server.unload({"cmd": "unload", "unless_variant": "avocado_int4"})
    assert resident["key"] is None
    server.unload({"cmd": "unload"})
    server.unload()

    events = [json.loads(line) for line in stream.getvalue().splitlines()]
    unloaded = [event for event in events if event["ev"] == "unloaded"]
    assert [(event["resident"], event["released"], event["skipped"]) for event in unloaded] == [
        ("timechat_int4", None, True),
        ("timechat_int4", "timechat_int4", False),
        (None, None, False),
        (None, None, False),
    ]
    assert unloaded[0]["report"] is None
    assert unloaded[1]["report"]["freed_vram_gb"] == pytest.approx(6.5)
    logs = [event["text"] for event in events if event["ev"] == "log"]
    assert "Model unloaded" in logs
    assert any("Keeping timechat_int4 loaded" in text for text in logs)
    assert not any(event["ev"] == "error" for event in events)


def test_worker_refuses_to_unload_while_a_job_is_running(monkeypatch: pytest.MonkeyPatch) -> None:
    from vcap.pipeline.worker import _ProtocolWriter, _Server

    _resident_runner(monkeypatch, "timechat_int4")
    stream = io.StringIO()
    server = _Server(_ProtocolWriter(stream))
    release = threading.Event()
    thread = threading.Thread(target=release.wait, name="fake-active-job")
    thread.start()
    try:
        server._active_thread = thread
        server.unload({"cmd": "unload", "unless_variant": "avocado_int4"})
    finally:
        release.set()
        thread.join()
    events = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert events and events[0]["ev"] == "error"
    assert not any(event["ev"] == "unloaded" for event in events)


# --------------------------------------------------------------------------- client


def test_pipeline_client_select_variant_in_process_releases_only_other_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vcap.pipeline.client import PipelineClient

    resident, calls = _resident_runner(monkeypatch, "timechat_int4")
    client = PipelineClient(subprocess_mode=False)
    try:
        kept = client.select_variant("timechat_int4")
        assert kept["skipped"] is True
        assert kept["released"] is None
        assert kept["resident"] == "timechat_int4"
        assert resident["key"] == "timechat_int4"

        outcome = client.select_variant("avocado_int4")
        assert outcome["released"] == "timechat_int4"
        assert outcome["skipped"] is False
        assert outcome["report"]["freed_vram_gb"] == pytest.approx(6.5)
        assert resident["key"] is None

        empty = client.release_model()
        assert empty["released"] is None and empty["resident"] is None and empty["skipped"] is False
        assert calls[:3] == ["timechat_int4", "avocado_int4", None]
    finally:
        client.shutdown()


def test_pipeline_client_defers_release_until_the_running_job_finishes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import vcap.pipeline.runner as runner
    from vcap.pipeline.client import PipelineClient

    monkeypatch.setenv("VCAP_FAKE_CAPTIONER", "1")
    monkeypatch.setenv("VCAP_FAKE_CAPTION_SLEEP", "0.8")
    monkeypatch.setattr(runner, "_FAKE_CAPTIONER", None)
    monkeypatch.setattr(runner, "_FAKE_VARIANT", None)
    client = PipelineClient(subprocess_mode=False)
    try:

        def run(variant: str) -> tuple[threading.Thread, dict[str, Any], threading.Event]:
            spec = JobSpec.from_settings(
                _settings(model_key=variant, subprocess_mode=False),
                [InputItem("deferred release text", text_prompt_only=True)],
                OutputSpec(outputs_root=tmp_path / "runs"),
            )
            done = threading.Event()
            outcome: dict[str, Any] = {}

            def work() -> None:
                try:
                    outcome["result"] = client.run_job(spec, None)
                except BaseException as exc:  # pragma: no cover - surfaced by the assertion below
                    outcome["error"] = exc
                finally:
                    done.set()

            thread = threading.Thread(target=work, name="deferred-release-job")
            thread.start()
            deadline = time.monotonic() + 5.0
            while runner.loaded_variant_key() != variant and time.monotonic() < deadline:
                time.sleep(0.01)
            assert runner.loaded_variant_key() == variant, "the fake job did not start"
            return thread, outcome, done

        # 1. Selection changes to another model while the job runs: release is deferred.
        thread, outcome, done = run("timechat_int4")
        assert client.select_variant("avocado_int4") == {"busy": True, "deferred": True}
        assert runner.loaded_variant_key() == "timechat_int4", "never unload under a running job"
        assert done.wait(timeout=15.0) and "error" not in outcome
        thread.join()
        assert outcome["result"].counts["done"] == 1
        assert runner.loaded_variant_key() is None, "the old model must go as soon as the job ends"

        # 2. Selecting the model the job itself uses keeps it resident.
        thread, outcome, done = run("avocado_int4")
        assert client.select_variant("avocado_int4") == {"busy": True, "deferred": True}
        assert done.wait(timeout=15.0) and "error" not in outcome
        thread.join()
        assert runner.loaded_variant_key() == "avocado_int4"

        # 3. Idle: an immediate release.
        released = client.select_variant("timechat_int4")
        assert released["released"] == "avocado_int4"
        assert runner.loaded_variant_key() is None
    finally:
        client.shutdown()


def test_pipeline_client_release_model_through_the_worker_protocol(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from vcap.pipeline.client import PipelineClient

    monkeypatch.setenv("VCAP_FAKE_CAPTIONER", "1")
    client = PipelineClient(subprocess_mode=True)
    try:
        spec = JobSpec.from_settings(
            _settings(subprocess_mode=True),
            [InputItem("worker release text", text_prompt_only=True)],
            OutputSpec(outputs_root=tmp_path / "runs"),
        )
        result = client.run_job(spec, None)
        assert result.counts["done"] == 1
        worker = client._worker
        assert worker is not None and worker.is_alive()
        assert client.ping(timeout_s=5.0)["loaded_variant"] == "timechat_int4"

        kept = client.select_variant("timechat_int4")
        assert kept["ev"] == "unloaded"
        assert kept["skipped"] is True and kept["released"] is None
        assert client.ping(timeout_s=5.0)["loaded_variant"] == "timechat_int4"

        released = client.select_variant("avocado_int4")
        assert released["ev"] == "unloaded"
        assert released["released"] == "timechat_int4"
        assert released["skipped"] is False
        assert released["report"]["released"] is True
        assert client.ping(timeout_s=5.0)["loaded_variant"] is None
        assert client._worker is worker and worker.is_alive(), "the worker stays for the next model"

        empty = client.release_model()
        assert empty["ev"] == "unloaded" and empty["released"] is None

        # A selection change while the worker job runs is released once the job returns.
        monkeypatch.setenv("VCAP_FAKE_CAPTION_SLEEP", "0.8")
        client.shutdown()  # restart the worker so the new sleep applies
        client = PipelineClient(subprocess_mode=True)
        done = threading.Event()
        outcome: dict[str, Any] = {}

        def work() -> None:
            try:
                outcome["result"] = client.run_job(spec, None)
            except BaseException as exc:  # pragma: no cover
                outcome["error"] = exc
            finally:
                done.set()

        thread = threading.Thread(target=work, name="worker-deferred-release")
        thread.start()
        deadline = time.monotonic() + 10.0
        while not client._busy and time.monotonic() < deadline:
            time.sleep(0.01)
        assert client._busy
        assert client.select_variant("avocado_int4") == {"busy": True, "deferred": True}
        assert done.wait(timeout=30.0) and "error" not in outcome
        thread.join()
        assert outcome["result"].counts["done"] == 1
        assert client.ping(timeout_s=5.0)["loaded_variant"] is None
    finally:
        client.shutdown()


# --------------------------------------------------------------------------- UI wiring


def test_caption_tab_model_change_releases_the_previous_model() -> None:
    import gradio as gr

    from vcap.core.logs import get_log
    from vcap.core.presets import PresetStore
    from vcap.core.registry import SettingsRegistry
    from vcap.ui.app import UiContext
    from vcap.ui.tabs import caption_tab

    class FakeClient:
        subprocess_mode = True

        def __init__(self) -> None:
            self.selected: list[str] = []

        def select_variant(self, variant_key: str) -> dict[str, Any]:
            self.selected.append(str(variant_key))
            return {
                "ev": "unloaded",
                "resident": "timechat_int4",
                "released": "timechat_int4",
                "skipped": False,
                "report": {"variant_key": "timechat_int4", "released": True, "freed_vram_gb": 6.5},
            }

        def ping(self, timeout_s: float = 0.6) -> dict[str, Any]:
            del timeout_s
            return {"ev": "pong", "loaded_variant": None, "block_swap": None}

        def set_subprocess_mode(self, enabled: bool) -> None:
            self.subprocess_mode = bool(enabled)

        def shutdown(self) -> None:
            pass

    client = FakeClient()
    ctx = UiContext(
        settings_registry=SettingsRegistry(),
        preset_store=PresetStore(Path("presets"), Path("presets_default")),
        pipeline_client=client,  # type: ignore[arg-type]
        app_log=get_log(),
    )
    with gr.Blocks() as demo:
        handles = caption_tab.build(ctx)
    model_key = handles.controls["model_key"]
    release = [
        fn
        for fn in demo.fns.values()
        if getattr(fn.fn, "__name__", "") == "release_previous_model"
        and (model_key._id, "change") in list(fn.targets)
    ]
    assert len(release) == 1, "the model dropdown must trigger release_previous_model on change"
    before = get_log().revision
    release[0].fn("avocado_int4")
    assert client.selected == ["avocado_int4"]
    lines, _ = get_log().snapshot(before)
    assert any("timechat_int4" in line and "avocado_int4" in line for line in lines)
