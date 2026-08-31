from __future__ import annotations

from vcap.core.gpu import (
    GpuInfo,
    cuda_visible_devices_env,
    is_oom_error,
    list_gpus,
    render_resource_meter_html,
    resource_snapshot,
    vram_tier_for_gb,
)


def test_gpu_enumeration_snapshot_and_html_without_torch() -> None:
    gpus = list_gpus()
    assert isinstance(gpus, list)
    assert all(isinstance(gpu, GpuInfo) for gpu in gpus)
    snapshot = resource_snapshot(0)
    assert snapshot["ram_total_gb"] >= 0
    assert snapshot["vram_total_gb"] >= 0
    html = render_resource_meter_html(snapshot, peak_vram_gb=1.5)
    assert 'class="vc-meter"' in html
    assert "VRAM" in html and "RAM" in html


def test_tier_oom_and_visibility_helpers() -> None:
    assert vram_tier_for_gb(5) == 6
    assert vram_tier_for_gb(7.9) == 8
    assert vram_tier_for_gb(31.8) == 32
    assert vram_tier_for_gb(64) == 48
    assert is_oom_error("CUDA error: out of memory")
    assert is_oom_error("CUBLAS_STATUS_ALLOC_FAILED")
    assert is_oom_error("AcceleratorError: CUDA error code 2")
    assert not is_oom_error("ordinary CPU exception")
    assert cuda_visible_devices_env(1) == {"CUDA_VISIBLE_DEVICES": "1"}
