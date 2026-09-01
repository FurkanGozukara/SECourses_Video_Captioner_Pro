from __future__ import annotations

import ctypes
from types import SimpleNamespace

import pytest

from vcap.core import gpu
from vcap.core.gpu import (
    GpuInfo,
    cuda_visible_devices_env,
    is_oom_error,
    list_gpus,
    render_resource_meter_html,
    resource_snapshot,
    shared_gpu_memory_usage,
    vram_cap_env_disabled,
    vram_tier_for_gb,
)


def test_gpu_enumeration_snapshot_and_html_without_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gpu,
        "shared_gpu_memory_total",
        lambda: {"shared_gb": 1.25, "dedicated_gb": 0.5},
    )
    gpus = list_gpus()
    assert isinstance(gpus, list)
    assert all(isinstance(gpu, GpuInfo) for gpu in gpus)
    snapshot = resource_snapshot(0)
    assert snapshot["ram_total_gb"] >= 0
    assert snapshot["vram_total_gb"] >= 0
    assert snapshot["shared_used_gb"] == 1.25
    html = render_resource_meter_html(snapshot, peak_vram_gb=1.5)
    assert 'class="vc-meter"' in html
    assert "VRAM" in html and "RAM" in html
    assert "Shared GPU memory (WDDM" in html


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


class _FakePdh:
    def __init__(self) -> None:
        gib = 2**30
        self.values = {
            11: [gib // 4, gib // 2],
            12: [gib, gib // 4],
        }
        self.paths: list[str] = []
        self.collect_calls = 0
        self.close_calls = 0

    def PdhOpenQueryW(self, source, user_data, query_ptr) -> int:  # noqa: N802
        query_ptr._obj.value = 10
        return 0

    def PdhAddEnglishCounterW(self, query, path, user_data, counter_ptr) -> int:  # noqa: N802
        self.paths.append(path)
        counter_ptr._obj.value = 11 if path.endswith("Shared Usage") else 12
        return 0

    def PdhCollectQueryData(self, query) -> int:  # noqa: N802
        self.collect_calls += 1
        return 0

    def PdhGetFormattedCounterArrayW(  # noqa: N802
        self,
        counter,
        value_format,
        buffer_size_ptr,
        item_count_ptr,
        buffer,
    ) -> int:
        values = self.values[counter.value]
        item_count_ptr._obj.value = len(values)
        buffer_size_ptr._obj.value = len(values) * ctypes.sizeof(
            gpu._PdhFormattedCounterValueItemW
        )
        if buffer is None:
            return gpu._PDH_MORE_DATA
        items = ctypes.cast(
            buffer, ctypes.POINTER(gpu._PdhFormattedCounterValueItemW)
        )
        for index, value in enumerate(values):
            items[index].szName = None
            items[index].FmtValue.CStatus = 0
            items[index].FmtValue.largeValue = value
        return 0

    def PdhCloseQuery(self, query) -> int:  # noqa: N802
        self.close_calls += 1
        return 0


def test_shared_gpu_memory_usage_sums_mocked_pdh_instances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakePdh()
    monkeypatch.setattr(gpu.os, "name", "nt")
    monkeypatch.setattr(gpu.ctypes, "windll", SimpleNamespace(pdh=fake))

    usage = shared_gpu_memory_usage(4242)

    assert usage == {"shared_gb": 0.75, "dedicated_gb": 1.25}
    assert fake.paths == [
        r"\GPU Process Memory(pid_4242_*)\Shared Usage",
        r"\GPU Process Memory(pid_4242_*)\Dedicated Usage",
    ]
    assert fake.collect_calls == 1
    assert fake.close_calls == 1


def test_shared_gpu_memory_usage_returns_empty_on_pdh_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakePdh()
    fake.PdhCollectQueryData = lambda query: 1  # type: ignore[method-assign]
    monkeypatch.setattr(gpu.os, "name", "nt")
    monkeypatch.setattr(gpu.ctypes, "windll", SimpleNamespace(pdh=fake))
    assert shared_gpu_memory_usage(4242) == {}
    assert fake.close_calls == 1

    monkeypatch.setattr(gpu.os, "name", "posix")
    assert shared_gpu_memory_usage(4242) == {}


@pytest.mark.parametrize("value", ["0", " false ", "OFF"])
def test_vram_cap_disabled_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("VCAP_VRAM_HARD_CAP", value)
    assert vram_cap_env_disabled()


@pytest.mark.parametrize("value", [None, "", "1", "true", "no"])
def test_vram_cap_enabled_values(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv("VCAP_VRAM_HARD_CAP", raising=False)
    else:
        monkeypatch.setenv("VCAP_VRAM_HARD_CAP", value)
    assert not vram_cap_env_disabled()
