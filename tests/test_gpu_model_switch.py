"""Real-GPU verification that switching models releases the previous one completely.

Opt-in: ``VCAP_GPU_TESTS=1`` (and the referenced checkpoints on disk). Runs on physical GPU
``VCAP_GPU_TEST_INDEX`` (default 0) and needs a few minutes. ``VCAP_GPU_TESTS_COMPILE=1``
additionally exercises the Inductor-compiled path (several minutes of compilation).

Whole-GPU NVML readings include every other process on the card (the desktop, browsers),
so the strict assertions are process-level (the PyTorch allocator must be empty, the
released objects must be dead, the unload report's own before/after delta must cover the
model) and the NVML comparisons keep a noise tolerance.
"""

from __future__ import annotations

import gc
import os
import time
import weakref
from pathlib import Path
from typing import Any

import psutil
import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("VCAP_GPU_TESTS", "").strip() != "1",
    reason="set VCAP_GPU_TESTS=1 to run real-GPU model switch tests",
)

GPU = int(os.environ.get("VCAP_GPU_TEST_INDEX", "0") or 0)
ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "temp" / "codex_BS" / "media" / "video20s.mp4"
NOISE_GB = 1.5  # other processes on a display GPU move by up to about a gigabyte
MODEL_GB = 4.0  # every variant used here occupies well over this once resident


def _nvml_used_gb(index: int = GPU) -> float:
    import pynvml

    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        return pynvml.nvmlDeviceGetMemoryInfo(handle).used / 2**30
    finally:
        pynvml.nvmlShutdown()


def _cuda_baseline_gb() -> float:
    """Create this process's CUDA context first so it is not counted as a leak."""

    import torch

    device = torch.device(f"cuda:{GPU}")
    torch.cuda.init()
    torch.zeros(1, device=device)
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    return _nvml_used_gb()


def _torch_process_mib() -> tuple[float, float]:
    import torch

    device = torch.device(f"cuda:{GPU}")
    return (
        torch.cuda.memory_allocated(device) / 2**20,
        torch.cuda.memory_reserved(device) / 2**20,
    )


def _assert_allocator_empty() -> None:
    allocated, reserved = _torch_process_mib()
    assert allocated == 0.0 and reserved == 0.0, (
        f"PyTorch still holds {allocated:.1f} MiB allocated / {reserved:.1f} MiB reserved after unload"
    )


def _llama_pids() -> set[int]:
    found: set[int] = set()
    for proc in psutil.process_iter(["name"]):
        try:
            if str(proc.info["name"] or "").casefold().startswith("llama-server"):
                found.add(proc.pid)
        except Exception:
            continue
    return found


def _require(*variants: str) -> None:
    from vcap.models.registry import variant_is_ready

    for key in variants:
        ready, detail = variant_is_ready(key)
        if not ready:
            pytest.skip(f"{key} is not available: {detail}")


def _hint(context: int) -> Any:
    from vcap.models.offload import BudgetHint

    return BudgetHint(
        max_frames=8,
        max_pixels=200_704,
        fps=1.0,
        max_new_tokens=16,
        context_tokens=context,
        media_kinds=("video",),
    )


def _load(variant: str, **extra: Any) -> Any:
    from vcap.models.loader import MODEL_CACHE
    from vcap.models.offload import OffloadPlan

    kwargs: dict[str, Any] = {
        "device": f"cuda:{GPU}",
        "gpu_index": GPU,
        "attention": "auto",
        "offload": OffloadPlan(),
        "progress_cb": None,
        "budget_hint": _hint(8192 if variant.endswith(("_q4", "_q8")) else 32768),
    }
    kwargs.update(extra)
    return MODEL_CACHE.load(variant, **kwargs)


def _caption(loaded: Any) -> str:
    from vcap.models import captioner_for_loaded
    from vcap.models.base import GenParams, MediaInput, PreprocessParams
    from vcap.models.registry import MODEL_SPECS

    spec = MODEL_SPECS[loaded.spec.family]
    pre = PreprocessParams(
        fps=1.0,
        max_frames=8,
        max_pixels=spec.limits.default_max_pixels,
        min_pixels=spec.limits.min_pixels,
        use_audio_in_video="video_audio" in spec.capabilities,
    )
    result = captioner_for_loaded(loaded).caption(MediaInput(path=VIDEO), None, GenParams(max_new_tokens=16), pre)
    return result.text


@pytest.fixture(autouse=True)
def _clean_cache() -> Any:
    from vcap.models.loader import MODEL_CACHE

    MODEL_CACHE.unload()
    yield
    MODEL_CACHE.unload()


def test_switching_transformers_models_releases_the_previous_one() -> None:
    _require("timechat_int4", "avocado_int4")
    if not VIDEO.is_file():
        pytest.skip(f"missing sample video {VIDEO}")
    from vcap.models.loader import MODEL_CACHE

    start = _cuda_baseline_gb()
    first = _load("timechat_int4")
    assert _caption(first)
    one_model = _nvml_used_gb()
    assert one_model > start + MODEL_GB, "the first model did not land on the GPU"
    old_model = weakref.ref(first.model)
    old_processor = weakref.ref(first.processor)
    first = None

    second = _load("avocado_int4")
    gc.collect()
    assert old_model() is None and old_processor() is None, "the previous model is still referenced"
    assert MODEL_CACHE.loaded_variant_key() == "avocado_int4"
    switched = _nvml_used_gb()
    assert switched <= one_model + NOISE_GB, f"two models resident: {switched:.2f} GB vs {one_model:.2f} GB"
    assert _caption(second)
    new_model = weakref.ref(second.model)
    second = None

    report = MODEL_CACHE.unload()
    gc.collect()
    assert report is not None and report.released is True
    assert report.variant_key == "avocado_int4" and report.backend == "transformers"
    assert new_model() is None
    assert not any("still referenced" in note for note in report.notes)
    assert report.freed_vram_gb >= MODEL_GB, report
    _assert_allocator_empty()
    after = _nvml_used_gb()
    assert after <= one_model - MODEL_GB, f"the model's VRAM is still in use: {after:.2f} GB vs {one_model:.2f} GB"
    assert after <= start + NOISE_GB, f"VRAM not released: {after:.2f} GB vs {start:.2f} GB at start"


def test_switching_from_gguf_to_transformers_stops_llama_server() -> None:
    _require("qwen3_omni_instruct_gguf_q4", "timechat_int4")
    from vcap.models.loader import MODEL_CACHE

    start = _cuda_baseline_gb()
    baseline_pids = _llama_pids()
    gguf = _load("qwen3_omni_instruct_gguf_q4")
    server_pids = _llama_pids() - baseline_pids
    assert server_pids, "llama-server did not start"
    with_server = _nvml_used_gb()
    assert with_server > start + 10.0
    backend = weakref.ref(gguf.model)
    gguf = None

    switched = _load("timechat_int4")
    gc.collect()
    assert not (_llama_pids() & server_pids), "llama-server survived the switch"
    assert backend() is None
    assert MODEL_CACHE.loaded_variant_key() == "timechat_int4"
    plan = switched.load_report.block_swap or {}
    if float(plan.get("total_vram_gib", 0.0) or 0.0) >= 20.0:
        # The new plan must have seen the freed VRAM, not the dying server's allocation.
        assert int(plan.get("swapped_layers", 0) or 0) == 0, plan
    assert _nvml_used_gb() <= with_server - 8.0, "the GGUF server's VRAM was not returned before the next load"
    switched = None

    report = MODEL_CACHE.unload()
    assert report is not None and report.released is True
    assert report.freed_vram_gb >= MODEL_GB, report
    _assert_allocator_empty()
    assert _nvml_used_gb() <= start + NOISE_GB
    assert not (_llama_pids() - baseline_pids)


def test_unloading_gguf_waits_for_the_driver_to_return_vram() -> None:
    _require("qwen3_omni_instruct_gguf_q4")
    from vcap.models.loader import MODEL_CACHE

    start = _nvml_used_gb()
    _load("qwen3_omni_instruct_gguf_q4")
    report = MODEL_CACHE.unload()
    assert report is not None and report.backend == "llamacpp"
    assert report.released is True
    assert report.freed_vram_gb >= 10.0, report
    assert report.vram_after_gb <= start + NOISE_GB, report
    assert not any("timed out" in note.casefold() for note in report.notes), report.notes
    assert _nvml_used_gb() <= start + NOISE_GB


@pytest.mark.skipif(
    os.environ.get("VCAP_GPU_TESTS_COMPILE", "").strip() != "1",
    reason="set VCAP_GPU_TESTS_COMPILE=1 to exercise the Inductor-compiled release (slow)",
)
def test_unload_after_compiled_generation_releases_everything(tmp_path: Path) -> None:
    _require("timechat_int4")
    if not VIDEO.is_file():
        pytest.skip(f"missing sample video {VIDEO}")
    import sys

    from vcap.models.loader import MODEL_CACHE
    from vcap.pipeline.job import InputItem, JobSpec, OutputSpec
    from vcap.pipeline.runner import run_job

    start = _cuda_baseline_gb()
    settings = {
        "model_key": "timechat_int4",
        "gpu_index": GPU,
        "subprocess_mode": False,
        "keep_model_loaded": True,
        "idle_unload_minutes": 0,
        "compile": True,
        "compile_mode": "default",
        "max_new_tokens": 16,
        "max_frames": 8,
        "fps": 1.0,
        "output_formats": ["txt"],
        "prompt_preset_id": None,
    }
    spec = JobSpec.from_settings(settings, [InputItem(str(VIDEO))], OutputSpec(kind="single", outputs_root=str(tmp_path)))
    result = run_job(spec, None)
    assert result.counts["done"] == 1
    loaded = MODEL_CACHE.loaded
    assert loaded is not None
    model_ref = weakref.ref(loaded.model)
    decoder_ref = weakref.ref(getattr(loaded.model, "model", loaded.model))
    loaded = None

    report = MODEL_CACHE.unload()
    gc.collect()
    assert report is not None and report.released is True
    assert model_ref() is None and decoder_ref() is None
    assert report.freed_vram_gb >= MODEL_GB, report
    _assert_allocator_empty()
    assert _nvml_used_gb() <= start + NOISE_GB
    assert "torch._dynamo" in sys.modules


def test_worker_protocol_switch_releases_between_jobs(tmp_path: Path) -> None:
    _require("timechat_int4", "avocado_int4")
    if not VIDEO.is_file():
        pytest.skip(f"missing sample video {VIDEO}")
    from vcap.pipeline.client import PipelineClient
    from vcap.pipeline.job import InputItem, JobSpec, OutputSpec

    def job(variant: str) -> JobSpec:
        settings = {
            "model_key": variant,
            "gpu_index": GPU,
            "subprocess_mode": True,
            "keep_model_loaded": True,
            "idle_unload_minutes": 0,
            "max_new_tokens": 16,
            "max_frames": 8,
            "fps": 1.0,
            "output_formats": ["txt"],
            "prompt_preset_id": None,
        }
        return JobSpec.from_settings(settings, [InputItem(str(VIDEO))], OutputSpec(kind="single", outputs_root=str(tmp_path)))

    start = _nvml_used_gb()
    client = PipelineClient(subprocess_mode=True)
    try:
        assert client.run_job(job("timechat_int4"), None).counts["done"] == 1
        one_model = _nvml_used_gb()
        assert one_model > start + MODEL_GB
        assert client.ping(timeout_s=5.0)["loaded_variant"] == "timechat_int4"

        assert client.run_job(job("avocado_int4"), None).counts["done"] == 1
        assert client.ping(timeout_s=5.0)["loaded_variant"] == "avocado_int4"
        assert _nvml_used_gb() <= one_model + NOISE_GB, "two models resident in the worker"

        kept = client.select_variant("avocado_int4")
        assert kept["skipped"] is True
        released = client.select_variant("timechat_int4")
        assert released["released"] == "avocado_int4"
        assert released["report"]["released"] is True
        assert released["report"]["freed_vram_gb"] >= MODEL_GB, released
        assert client.ping(timeout_s=5.0)["loaded_variant"] is None
        time.sleep(0.5)
        after = _nvml_used_gb()
        assert after <= one_model - MODEL_GB, f"the worker still holds the model: {after:.2f} GB vs {one_model:.2f} GB"
        assert after <= start + NOISE_GB
    finally:
        client.shutdown()
    time.sleep(1.0)
    assert _nvml_used_gb() <= start + NOISE_GB
