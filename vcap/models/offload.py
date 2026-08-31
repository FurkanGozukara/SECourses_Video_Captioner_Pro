"""VRAM budgeting and conservative decoder-layer placement helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from vcap.core.gpu import resource_snapshot


DeviceTarget = int | str


@dataclass(frozen=True)
class OffloadPlan:
    """Requested GPU residency and CPU-offload policy."""

    gpu_layers: int | str = field(
        default="all", metadata={"description": "Decoder layers kept resident on the selected GPU."}
    )
    offload_experts: bool = field(
        default=False, metadata={"description": "Keep Qwen3 MoE expert banks on CPU when possible."}
    )
    max_memory: dict[DeviceTarget, str] | None = field(
        default=None, metadata={"description": "Optional Accelerate max-memory override."}
    )
    pin_cpu: bool = field(
        default=False, metadata={"description": "Pin CPU-resident model tensors for faster transfer."}
    )

    def __post_init__(self) -> None:
        value = self.gpu_layers
        if value != "all" and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
            raise ValueError("gpu_layers must be 'all' or a non-negative integer")


@dataclass(frozen=True)
class DeviceMapPlan:
    """Both Accelerate auto-loading and explicit meta-loader placement data."""

    device_map: str | dict[str, DeviceTarget]
    max_memory: dict[DeviceTarget, str] | None
    explicit_device_map: dict[str, DeviceTarget]
    no_split_modules: tuple[str, ...]

    def from_pretrained_kwargs(self) -> dict[str, object]:
        """Return kwargs accepted by Hugging Face ``from_pretrained``."""

        result: dict[str, object] = {"device_map": self.device_map}
        if self.max_memory:
            result["max_memory"] = self.max_memory
        return result


_FAMILY_LAYOUTS: dict[str, tuple[int, tuple[str, ...]]] = {
    "timechat": (28, ("Qwen2_5OmniAudioEncoder", "Qwen2_5OmniVisionEncoder", "Qwen2_5OmniDecoderLayer")),
    "avocado": (28, ("Qwen2_5OmniAudioEncoder", "Qwen2_5OmniVisionEncoder", "Qwen2_5OmniDecoderLayer")),
    "qwen3_omni_instruct": (48, ("Qwen3OmniMoeAudioEncoder", "Qwen3OmniMoeVisionEncoder", "Qwen3OmniMoeThinkerTextDecoderLayer")),
    "qwen3_omni_thinking": (48, ("Qwen3OmniMoeAudioEncoder", "Qwen3OmniMoeVisionEncoder", "Qwen3OmniMoeThinkerTextDecoderLayer")),
    "qwen3_omni_captioner": (48, ("Qwen3OmniMoeAudioEncoder", "Qwen3OmniMoeThinkerTextDecoderLayer")),
}


def _memory_limits(
    plan: OffloadPlan,
    vram_free_gb: float,
    *,
    device_index: int,
    physical_gpu_index: int,
) -> dict[DeviceTarget, str]:
    if plan.max_memory:
        return dict(plan.max_memory)
    gpu_gib = max(1, int(max(0.0, float(vram_free_gb)) - 0.5))
    ram_free = float(
        resource_snapshot(int(physical_gpu_index)).get("ram_free_gb", 0.0) or 0.0
    )
    cpu_gib = max(4, int(max(4.0, ram_free - 4.0)))
    return {int(device_index): f"{gpu_gib}GiB", "cpu": f"{cpu_gib}GiB"}


def build_device_map(
    model_family: str,
    plan: OffloadPlan,
    vram_free_gb: float,
    *,
    device_index: int = 0,
    physical_gpu_index: int | None = None,
) -> DeviceMapPlan:
    """Build Accelerate and explicit maps at decoder-layer granularity."""

    try:
        layer_count, no_split = _FAMILY_LAYOUTS[model_family]
    except KeyError as exc:
        raise KeyError(f"Unknown model family for offload placement: {model_family}") from exc

    selected_device = max(0, int(device_index))
    selected_physical = (
        selected_device if physical_gpu_index is None else max(0, int(physical_gpu_index))
    )
    requested = layer_count if plan.gpu_layers == "all" else min(layer_count, int(plan.gpu_layers))
    explicit: dict[str, DeviceTarget] = {
        "audio_tower": selected_device,
        "visual": selected_device,
        "model.embed_tokens": selected_device,
        "model.rotary_emb": selected_device,
        "model.norm": selected_device,
        "lm_head": "cpu" if requested < layer_count else selected_device,
    }
    for index in range(layer_count):
        explicit[f"model.layers.{index}"] = selected_device if index < requested else "cpu"
        if plan.offload_experts and model_family.startswith("qwen3_"):
            explicit[f"model.layers.{index}.mlp.experts"] = "cpu"

    all_resident = requested == layer_count and not plan.offload_experts and plan.max_memory is None
    auto_map: str | dict[str, DeviceTarget] = {"": selected_device} if all_resident else "auto"
    return DeviceMapPlan(
        device_map=auto_map,
        max_memory=(
            None
            if all_resident
            else _memory_limits(
                plan,
                vram_free_gb,
                device_index=selected_device,
                physical_gpu_index=selected_physical,
            )
        ),
        explicit_device_map=explicit,
        no_split_modules=no_split,
    )


def estimate_layers_on_gpu(
    checkpoint_bytes_by_layer: Mapping[object, int] | Sequence[int],
    budget_gb: float,
) -> int:
    """Count the leading decoder layers that fit within a byte budget."""

    values = (
        list(checkpoint_bytes_by_layer.values())
        if isinstance(checkpoint_bytes_by_layer, Mapping)
        else list(checkpoint_bytes_by_layer)
    )
    budget = max(0, int(float(budget_gb) * 1024**3))
    used = 0
    count = 0
    for raw in values:
        size = max(0, int(raw))
        if used + size > budget:
            break
        used += size
        count += 1
    return count


def vram_budget_gb(
    gpu_index: int,
    reserve_gb: float = 1.4,
    kv_gb: float = 0.0,
    vision_peak_gb: float = 0.0,
) -> float:
    """Return usable weight VRAM after runtime, KV, and vision reserves."""

    snapshot = resource_snapshot(int(gpu_index))
    free = float(snapshot.get("vram_free_gb", 0.0) or 0.0)
    return max(0.0, free - max(0.0, reserve_gb) - max(0.0, kv_gb) - max(0.0, vision_peak_gb))


__all__ = [
    "DeviceMapPlan",
    "OffloadPlan",
    "build_device_map",
    "estimate_layers_on_gpu",
    "vram_budget_gb",
]
