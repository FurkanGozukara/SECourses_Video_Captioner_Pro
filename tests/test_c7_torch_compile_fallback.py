from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


def test_compile_mode_registry_hides_dynamic_cache_unsafe_modes() -> None:
    from vcap.models.torch_compile import (
        DEFAULT_COMPILE_MODE,
        compile_mode_choices,
        compile_mode_values,
        normalize_compile_mode,
    )

    assert DEFAULT_COMPILE_MODE == "default"
    assert compile_mode_values() == ("default", "max-autotune-no-cudagraphs")
    assert [value for _label, value in compile_mode_choices()] == list(compile_mode_values())
    assert "cudagraphs" not in compile_mode_values()
    assert "reduce-overhead" not in compile_mode_values()
    assert normalize_compile_mode("full") == "default"
    assert normalize_compile_mode("not-a-mode") == DEFAULT_COMPILE_MODE


def test_compile_runtime_error_detection_includes_cudagraph_and_dynamo() -> None:
    from vcap.models.torch_compile import is_compile_runtime_error

    backend_error = type(
        "BackendCompilerFailed",
        (RuntimeError,),
        {"__module__": "torch._dynamo.exc"},
    )("backend failed")
    assert is_compile_runtime_error(backend_error)
    assert is_compile_runtime_error(
        RuntimeError(
            "Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run"
        )
    )
    assert not is_compile_runtime_error(RuntimeError("ordinary caption model error"))


class _Decoder:
    def modules(self) -> list[Any]:
        return [self]

    def forward(self) -> str:
        return "eager output"


class _RootModel:
    def __init__(self) -> None:
        self.model = _Decoder()

    def modules(self) -> list[Any]:
        return [self, self.model]


def test_failing_compiled_generation_restores_eager_and_retries_segment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import torch

    from vcap.core.subprocess_runner import CancelToken
    from vcap.models.base import CaptionResult
    from vcap.models.registry import MODEL_SPECS
    from vcap.models.torch_compile import (
        CompilePlan,
        apply_compile,
        disabled_compile_modes,
        prepare_compile_env,
    )
    from vcap.pipeline.job import InputItem, JobSpec, OutputSpec
    from vcap.pipeline import runner

    import vcap.models.torch_compile as compile_module

    monkeypatch.setattr(compile_module, "_DISABLED_COMPILE_MODES", {})
    monkeypatch.setattr(compile_module, "configure_compile_workers", lambda _value: None)
    monkeypatch.setattr(compile_module, "_reset_compiler_runtime", lambda: None)

    def broken_compile(_forward: Any, **_kwargs: Any) -> Any:
        def broken_forward(*_args: Any, **_forward_kwargs: Any) -> Any:
            raise RuntimeError(
                "Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run"
            )

        return broken_forward

    monkeypatch.setattr(torch, "compile", broken_compile)
    monkeypatch.setattr(runner, "_empty_cuda_cache", lambda: None)

    root = _RootModel()
    plan = CompilePlan(
        "cudagraphs",
        {
            "TORCHINDUCTOR_CACHE_DIR": str(tmp_path / "inductor"),
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
        },
        {"backend": "cudagraphs", "dynamic": None},
        [],
        ("eager",),
        "cudagraphs",
    )
    apply_compile(root, plan, family="timechat")
    assert getattr(root.model, "_vcap_compiled", False)

    loaded = SimpleNamespace(
        model=root,
        spec=SimpleNamespace(family="timechat"),
        load_report=SimpleNamespace(compile_mode="cudagraphs"),
    )
    calls: list[str] = []

    class Captioner:
        def caption(self, *_args: Any, **_kwargs: Any) -> CaptionResult:
            calls.append("caption")
            text = root.model.forward()
            return CaptionResult(text=text, raw_text=text)

    class Session:
        def __init__(self) -> None:
            self.loaded = loaded

        def ensure(self) -> Captioner:
            return Captioner()

    class Emitter:
        def __init__(self) -> None:
            self.logs: list[tuple[str, str, str]] = []

        def log(self, message: str, level: str = "info", scope: str = "pipeline") -> None:
            self.logs.append((message, level, scope))

    spec = JobSpec.from_settings(
        {
            "model_key": "timechat_int4",
            "torch_compile": True,
            "torch_compile_mode": "cudagraphs",
            "fps": 1.0,
            "max_frames": 4,
            "max_pixels": 65_536,
            "max_new_tokens": 8,
            "use_audio_in_video": False,
        },
        [InputItem(path="", kind="text", text_prompt_only=True, text="test")],
        OutputSpec(kind="single", outputs_root=tmp_path),
    )
    emitter = Emitter()
    result = runner._caption_with_oom_recovery(
        spec,
        MODEL_SPECS["timechat"],
        Session(),
        prompt="test",
        media=SimpleNamespace(path=None),
        callback=lambda *_args: None,
        cancel=CancelToken(),
        emitter=emitter,
    )

    assert result.text == "eager output"
    assert calls == ["caption", "caption"]
    assert not getattr(root.model, "_vcap_compiled", False)
    assert root.model.forward() == "eager output"
    assert loaded.load_report.compile_mode == "eager"
    assert ("timechat", "cudagraphs") in disabled_compile_modes()
    disabled_plan = prepare_compile_env(True, mode="cudagraphs", family="timechat")
    assert disabled_plan.mode == "eager"
    assert any("using eager execution" in warning for warning in disabled_plan.warnings)
    assert any(
        level == "warning"
        and scope == "compile"
        and "retrying the same segment once in eager mode" in message
        for message, level, scope in emitter.logs
    )


def test_pipeline_item_is_done_after_broken_compiled_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import torch

    from vcap.core.subprocess_runner import CancelToken
    from vcap.models import loader
    from vcap.models.base import CaptionResult
    from vcap.models.registry import MODEL_SPECS
    from vcap.models.torch_compile import CompilePlan, apply_compile
    from vcap.pipeline.job import InputItem, JobSpec, OutputSpec
    from vcap.pipeline.runner import run_job

    import vcap.models.torch_compile as compile_module
    from vcap.pipeline import runner

    monkeypatch.setattr(compile_module, "_DISABLED_COMPILE_MODES", {})
    monkeypatch.setattr(compile_module, "configure_compile_workers", lambda _value: None)
    monkeypatch.setattr(compile_module, "_reset_compiler_runtime", lambda: None)
    monkeypatch.setattr(runner, "_empty_cuda_cache", lambda: None)

    def broken_compile(_forward: Any, **_kwargs: Any) -> Any:
        def broken_forward(*_args: Any, **_forward_kwargs: Any) -> Any:
            raise RuntimeError("BackendCompilerFailed: deliberate torch.compile test failure")

        return broken_forward

    monkeypatch.setattr(torch, "compile", broken_compile)
    root = _RootModel()
    plan = CompilePlan(
        "full",
        {
            "TORCHINDUCTOR_CACHE_DIR": str(tmp_path / "inductor"),
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
        },
        {"backend": "inductor", "mode": "default", "dynamic": None},
        [],
        ("cudagraphs", "eager"),
        "default",
    )
    apply_compile(root, plan, family="qwen3_omni_instruct")

    class Loaded:
        def __init__(self) -> None:
            self.model = root
            self.spec = MODEL_SPECS["qwen3_omni_instruct"]
            self.variant = SimpleNamespace(key="qwen3_omni_instruct_int4")
            self.load_report = SimpleNamespace(peak_vram_gb=0.0, compile_mode="default")
            self.calls = 0

        def caption(self, *_args: Any, **_kwargs: Any) -> CaptionResult:
            self.calls += 1
            text = self.model.model.forward()
            return CaptionResult(text=text, raw_text=text)

    loaded = Loaded()

    def fake_load_model(_variant_key: str, **_kwargs: Any) -> Loaded:
        return loaded

    monkeypatch.setattr(loader, "load_model", fake_load_model)

    class Sink:
        def __init__(self) -> None:
            self.logs: list[tuple[str, str, str | None]] = []

        def on_log(self, message: str, level: str = "info", scope: str | None = None) -> None:
            self.logs.append((message, level, scope))

        def on_progress(self, _event: Any) -> None:
            pass

        def on_item(self, _event: Any) -> None:
            pass

    settings = {
        "model_key": "qwen3_omni_instruct_int4",
        "prompt_preset_id": "custom",
        "user_prompt": "Test fallback.",
        "torch_compile": True,
        "torch_compile_mode": "default",
        "compile_mode": "default",
        "keep_model_loaded": False,
        "subprocess_mode": False,
        "output_formats": ["txt"],
        "max_new_tokens": 8,
    }
    spec = JobSpec.from_settings(
        settings,
        [InputItem("deliberate broken compile", text_prompt_only=True)],
        OutputSpec(kind="single", outputs_root=tmp_path / "runs"),
    )
    sink = Sink()
    result = run_job(spec, sink, CancelToken())

    assert result.counts["done"] == 1
    assert result.counts["failed"] == 0
    assert loaded.calls == 2
    assert any(
        level == "warning"
        and scope == "compile"
        and "retrying the same segment once in eager mode" in message
        for message, level, scope in sink.logs
    )
