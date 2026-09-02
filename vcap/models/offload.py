"""VRAM budgeting and conservative decoder-layer placement helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import re
import struct
import time
from typing import Any, Mapping, Sequence

from vcap import TEMP_DIR
from vcap.core.gpu import resource_snapshot


DeviceTarget = int | str


@dataclass(frozen=True)
class OffloadPlan:
    """Requested GPU residency, block-swap, and legacy CPU-offload policy.

    ``gpu_layers`` is ``"auto"`` (fit the free VRAM minus the reserve), ``"all"``
    (force every decoder layer resident), or an integer count of resident decoder
    layers. Any non-resident layer is block-swapped from pinned host memory unless
    the legacy Accelerate path is selected through ``offload_experts``/``max_memory``.
    """

    gpu_layers: int | str = field(
        default="auto",
        metadata={"description": "Resident decoder layers: 'auto', 'all', or a count; the rest are block-swapped."},
    )
    offload_experts: bool = field(
        default=False, metadata={"description": "Legacy Accelerate expert offload; disables block swap."}
    )
    max_memory: dict[DeviceTarget, str] | None = field(
        default=None, metadata={"description": "Legacy Accelerate max-memory override; disables block swap."}
    )
    pin_cpu: bool = field(
        default=True, metadata={"description": "Pin swapped decoder layers in host memory."}
    )
    vram_reserve_gb: float = field(
        default=2.0, metadata={"description": "Dedicated VRAM to keep free at the expected generation peak."}
    )
    swap_slots: int = field(
        default=2, metadata={"description": "GPU staging slots for block swap (2 = one layer of prefetch)."}
    )
    pinned_ram_budget_gb: float = field(
        default=0.0,
        metadata={
            "description": "Maximum pinned host RAM in GiB; zero keeps 6 GiB of total RAM free."
        },
    )
    plan_slack_mib: int = field(
        default=512,
        metadata={"description": "Fixed CUDA allocator slack included in the block-swap VRAM plan."},
    )

    def __post_init__(self) -> None:
        value = self.gpu_layers
        if isinstance(value, str):
            normalized = value.strip().casefold()
            if normalized not in {"auto", "all"}:
                raise ValueError("gpu_layers must be 'auto', 'all', or a non-negative integer")
            object.__setattr__(self, "gpu_layers", normalized)
        elif not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("gpu_layers must be 'auto', 'all', or a non-negative integer")
        reserve = float(self.vram_reserve_gb)
        if reserve < 0.0 or reserve != reserve:
            raise ValueError("vram_reserve_gb must be a non-negative number")
        object.__setattr__(self, "vram_reserve_gb", reserve)
        slots = self.swap_slots
        if isinstance(slots, bool) or not isinstance(slots, int) or not 1 <= slots <= 4:
            raise ValueError("swap_slots must be an integer between 1 and 4")
        pinned_budget = float(self.pinned_ram_budget_gb)
        if pinned_budget < 0.0 or not math.isfinite(pinned_budget):
            raise ValueError("pinned_ram_budget_gb must be a non-negative number")
        object.__setattr__(self, "pinned_ram_budget_gb", min(1024.0, pinned_budget))
        slack_mib = int(self.plan_slack_mib)
        if not 0 <= slack_mib <= 8192:
            raise ValueError("plan_slack_mib must be between 0 and 8192")
        object.__setattr__(self, "plan_slack_mib", slack_mib)

    @property
    def uses_legacy_offload(self) -> bool:
        """Return whether the Accelerate hook path is explicitly requested."""

        return bool(self.offload_experts) or self.max_memory is not None

    @property
    def block_swap_enabled(self) -> bool:
        """Return whether non-resident decoder layers use block swap."""

        return not self.uses_legacy_offload and self.gpu_layers != "all"


@dataclass(frozen=True)
class BudgetHint:
    """What the caller knows about the upcoming job, for the activation estimate."""

    max_frames: int | None = None
    max_pixels: int | None = None
    fps: float | None = None
    max_new_tokens: int | None = None
    context_tokens: int | None = None
    media_kinds: tuple[str, ...] = ()


@dataclass(frozen=True)
class CheckpointLayout:
    """Per-decoder-layer byte sizes read from a safetensors header."""

    path: Path
    layer_count: int
    layer_bytes: tuple[int, ...]
    non_layer_bytes: int
    total_bytes: int
    tower_bytes: int = 0  # audio_tower.* + visual.* (prefill-only encoders), included in non_layer_bytes


@dataclass(frozen=True)
class BlockSwapBudget:
    """Resolved residency plan for one load."""

    layer_count: int
    resident_layers: int
    swapped_layers: int
    slots: int
    layer_bytes: int
    non_layer_bytes: int
    resident_weight_bytes: int
    activation_bytes: int
    reserve_bytes: int
    free_vram_bytes: int
    total_vram_bytes: int
    expected_peak_bytes: int
    pinned_bytes: int
    mode: str
    notes: tuple[str, ...]
    allocator_slack_bytes: int = 0
    stage_towers: bool = False
    tower_bytes: int = 0

    def summary(self) -> dict[str, Any]:
        """Return a JSON-safe summary with GiB values rounded to two decimals."""

        gib = float(2**30)
        return {
            "mode": self.mode,
            "layer_count": int(self.layer_count),
            "resident_layers": int(self.resident_layers),
            "swapped_layers": int(self.swapped_layers),
            "slots": int(self.slots),
            "layer_mib": round(self.layer_bytes / 2**20, 1),
            "non_layer_gib": round(self.non_layer_bytes / gib, 2),
            "resident_weight_gib": round(self.resident_weight_bytes / gib, 2),
            "activation_estimate_gib": round(self.activation_bytes / gib, 2),
            "allocator_slack_gib": round(self.allocator_slack_bytes / gib, 2),
            "stage_towers": bool(self.stage_towers),
            "tower_gib": round(self.tower_bytes / gib, 2),
            "reserve_gib": round(self.reserve_bytes / gib, 2),
            "free_vram_gib": round(self.free_vram_bytes / gib, 2),
            "total_vram_gib": round(self.total_vram_bytes / gib, 2),
            "expected_peak_gib": round(self.expected_peak_bytes / gib, 2),
            "pinned_gib": round(self.pinned_bytes / gib, 2),
            "notes": list(self.notes),
        }


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


def family_layer_count(family: str) -> int:
    """Return the decoder-layer count of a model family."""

    try:
        return _FAMILY_LAYOUTS[family][0]
    except KeyError as exc:
        raise KeyError(f"Unknown model family for block swap: {family}") from exc


def block_swap_to_gpu_layers(auto: bool, blocks_to_swap: int, layer_count: int) -> int | str:
    """Translate the block-swap controls into an :class:`OffloadPlan` ``gpu_layers`` value.

    ``auto`` keeps the loader's fit-to-free-VRAM policy; otherwise ``blocks_to_swap``
    decoder layers (capped at the family's layer count) stay in pinned RAM and the
    rest remain resident. Zero swapped layers keeps the whole decoder on the GPU.
    """

    if auto:
        return "auto"
    layers = max(0, int(layer_count))
    try:
        swapped = int(blocks_to_swap)
    except (TypeError, ValueError, OverflowError):
        swapped = 0
    return layers - min(layers, max(0, swapped))


def gpu_layers_to_block_swap(gpu_layers: int | str | None, layer_count: int) -> tuple[bool, int]:
    """Translate a ``gpu_layers`` value into ``(automatic, blocks_to_swap)``.

    ``auto`` (and anything unparseable) maps to the automatic plan, ``all`` to zero
    swapped layers, and a resident count ``N`` to ``layer_count - N`` swapped layers.
    """

    if gpu_layers is None or isinstance(gpu_layers, bool):
        return True, 0
    if isinstance(gpu_layers, str):
        text = gpu_layers.strip().casefold()
        if not text or text == "auto":
            return True, 0
        if text == "all":
            return False, 0
        try:
            resident = int(float(text))
        except ValueError:
            return True, 0
    else:
        try:
            resident = int(gpu_layers)
        except (TypeError, ValueError, OverflowError):
            return True, 0
    layers = max(0, int(layer_count))
    return False, layers - min(layers, max(0, resident))


def migrate_legacy_gpu_layers(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Rewrite a legacy ``gpu_layers`` setting into the block-swap controls.

    Older presets and run metadata stored ``gpu_layers`` (``auto``/``all``/count).
    The UI now stores ``block_swap_auto`` and ``blocks_to_swap`` instead; when only
    the legacy key is present it is translated using the selected family's layer
    count and removed. The input mapping is never mutated.
    """

    result = dict(settings)
    if "gpu_layers" not in result:
        return result
    legacy = result.pop("gpu_layers")
    if "block_swap_auto" in result or "blocks_to_swap" in result:
        return result
    layer_count = 48
    variant = result.get("model_key") or result.get("variant_key")
    if variant:
        try:
            from .registry import variant_to_family

            layer_count = family_layer_count(variant_to_family(str(variant)))
        except KeyError:
            pass
    auto, swapped = gpu_layers_to_block_swap(legacy, layer_count)
    result["block_swap_auto"] = auto
    result["blocks_to_swap"] = swapped
    return result


def planning_config(folder: str | Path) -> dict[str, Any]:
    """Read the thinker configuration from ``config.json`` without Transformers.

    Qwen3-Omni checkpoints nest the thinker under ``thinker_config``; Qwen2.5-Omni
    thinkers expose ``text_config`` at the root. Either shape satisfies
    :func:`estimate_activation_bytes`.
    """

    path = Path(folder) / "config.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid model config in {path}: expected a JSON object")
    inner = payload.get("thinker_config")
    return inner if isinstance(inner, dict) else payload


_LAYOUT_CACHE: dict[str, tuple[tuple[int, int], CheckpointLayout]] = {}


def cached_checkpoint_layout(safetensors_path: str | Path) -> CheckpointLayout:
    """Return :func:`checkpoint_layout` for a file, reusing it while size and mtime match."""

    path = Path(safetensors_path)
    stat = path.stat()
    signature = (int(stat.st_size), int(stat.st_mtime_ns))
    key = str(path.resolve(strict=False))
    cached = _LAYOUT_CACHE.get(key)
    if cached is not None and cached[0] == signature:
        return cached[1]
    layout = checkpoint_layout(path)
    _LAYOUT_CACHE[key] = (signature, layout)
    return layout


def plan_model_folder(
    family: str,
    variant_key: str,
    folder: str | Path,
    plan: OffloadPlan,
    hint: BudgetHint | None,
    *,
    free_vram_bytes: int,
    total_vram_bytes: int,
    ram_available_bytes: int | None = None,
    ram_total_bytes: int | None = None,
) -> BlockSwapBudget:
    """Resolve the residency plan for a local checkpoint folder without loading it.

    This mirrors the loader's planning inputs (safetensors header, config, activation
    estimate with the variant's observed ratio) so the UI can preview what ``auto``
    will resolve to before a job starts.
    """

    root = Path(folder)
    config = planning_config(root)
    activation = int(
        estimate_activation_bytes(
            family,
            config,
            hint,
            observed_ratio=observed_activation_ratio(variant_key),
        )
    )
    layout = cached_checkpoint_layout(root / "model.safetensors")
    return plan_block_swap(
        plan,
        layout,
        free_vram_bytes=free_vram_bytes,
        total_vram_bytes=total_vram_bytes,
        activation_bytes=activation,
        ram_available_bytes=ram_available_bytes,
        ram_total_bytes=ram_total_bytes,
    )


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
    requested = (
        layer_count if plan.gpu_layers in {"all", "auto"} else min(layer_count, int(plan.gpu_layers))
    )
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


def pinned_ram_budget_bytes(
    plan: OffloadPlan,
    total_ram_bytes: int,
) -> int:
    """Resolve the user pin limit; zero leaves 6 GiB of total host RAM unpinned."""

    total = max(0, int(total_ram_bytes))
    requested = max(0.0, float(plan.pinned_ram_budget_gb))
    if requested > 0:
        return min(total, int(requested * 2**30))
    return max(0, total - 6 * 2**30)


def checkpoint_layout(
    safetensors_path: str | Path,
    *,
    strip_prefix: str = "thinker.",
) -> CheckpointLayout:
    """Read per-decoder-layer byte sizes from a safetensors header only."""

    path = Path(safetensors_path)
    with path.open("rb") as checkpoint:
        raw_length = checkpoint.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"Invalid safetensors header in {path}: missing length prefix")
        header_length = struct.unpack("<Q", raw_length)[0]
        raw_header = checkpoint.read(header_length)
        if len(raw_header) != header_length:
            raise ValueError(f"Invalid safetensors header in {path}: truncated JSON header")

    try:
        header = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid safetensors header JSON in {path}") from exc
    if not isinstance(header, dict):
        raise ValueError(f"Invalid safetensors header in {path}: expected a JSON object")

    dtype_bytes = {
        "BOOL": 1,
        "U8": 1,
        "I8": 1,
        "I16": 2,
        "U16": 2,
        "F16": 2,
        "BF16": 2,
        "I32": 4,
        "U32": 4,
        "F32": 4,
        "I64": 8,
        "U64": 8,
        "F64": 8,
    }
    layer_pattern = re.compile(r"^model\.layers\.(\d+)\.")
    tower_pattern = re.compile(r"^(audio_tower|visual)\.")
    bytes_by_layer: dict[int, int] = {}
    non_layer_bytes = 0
    tower_bytes = 0
    total_bytes = 0

    for raw_name, tensor in header.items():
        if raw_name == "__metadata__":
            continue
        if not isinstance(raw_name, str) or not isinstance(tensor, dict):
            raise ValueError(f"Invalid tensor entry in safetensors header: {raw_name!r}")
        dtype = str(tensor.get("dtype", "")).upper()
        item_size = 1 if dtype.startswith("F8") else dtype_bytes.get(dtype)
        if item_size is None:
            raise ValueError(f"Unsupported safetensors dtype {dtype!r} for tensor {raw_name!r}")
        shape = tensor.get("shape")
        if not isinstance(shape, list):
            raise ValueError(f"Invalid shape for tensor {raw_name!r}")
        elements = 1
        for dimension in shape:
            if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0:
                raise ValueError(f"Invalid shape for tensor {raw_name!r}")
            elements *= dimension
        tensor_bytes = elements * item_size
        total_bytes += tensor_bytes

        name = raw_name[len(strip_prefix):] if strip_prefix and raw_name.startswith(strip_prefix) else raw_name
        match = layer_pattern.match(name)
        if match is None:
            non_layer_bytes += tensor_bytes
            if tower_pattern.match(name):
                tower_bytes += tensor_bytes
        else:
            index = int(match.group(1))
            bytes_by_layer[index] = bytes_by_layer.get(index, 0) + tensor_bytes

    layer_count = max(bytes_by_layer, default=-1) + 1
    layer_bytes = tuple(bytes_by_layer.get(index, 0) for index in range(layer_count))
    return CheckpointLayout(
        path=path,
        layer_count=layer_count,
        layer_bytes=layer_bytes,
        non_layer_bytes=non_layer_bytes,
        total_bytes=total_bytes,
        tower_bytes=tower_bytes,
    )


def estimate_activation_bytes(
    family: str,
    config: Any,
    hint: BudgetHint | None,
    *,
    observed_bytes: int = 0,
    observed_ratio: float = 1.0,
) -> int:
    """Estimate the non-weight VRAM peak of one generation in bytes."""

    from .registry import MODEL_SPECS

    spec = MODEL_SPECS[family]
    defaults = {parameter.name: parameter.default for parameter in spec.param_schema}
    budget_hint = hint or BudgetHint()
    qwen3 = family.startswith("qwen3_")

    def config_value(source: Any, name: str, default: Any = None) -> Any:
        if source is None:
            return default
        if isinstance(source, Mapping):
            value = source.get(name, default)
        else:
            value = getattr(source, name, default)
        return default if value is None else value

    text_config = config_value(config, "text_config", config)
    family_layers = 48 if qwen3 else 28
    family_kv_heads = 4
    family_head_dim = 128
    family_vocab = 152_064

    def positive_int(name: str, default: int) -> int:
        try:
            value = int(config_value(text_config, name, default))
        except (TypeError, ValueError, OverflowError):
            return default
        return value if value > 0 else default

    layers = positive_int("num_hidden_layers", family_layers)
    kv_heads = positive_int("num_key_value_heads", family_kv_heads)
    hidden_size = positive_int("hidden_size", 2_048 if qwen3 else 3_584)
    attention_heads = positive_int("num_attention_heads", 16 if qwen3 else 28)
    raw_head_dim = config_value(text_config, "head_dim", None)
    try:
        head_dim = int(raw_head_dim) if raw_head_dim is not None else hidden_size // attention_heads
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        head_dim = family_head_dim
    if head_dim <= 0:
        head_dim = family_head_dim
    vocab_size = positive_int("vocab_size", family_vocab)

    def hinted(name: str, fallback: int | float) -> int | float:
        value = getattr(budget_hint, name)
        return fallback if value is None else value

    try:
        max_frames = max(0, int(hinted("max_frames", defaults["max_frames"])))
    except (TypeError, ValueError, OverflowError):
        max_frames = max(0, int(defaults["max_frames"]))
    try:
        max_pixels = max(0, int(hinted("max_pixels", defaults["max_pixels"])))
    except (TypeError, ValueError, OverflowError):
        max_pixels = max(0, int(defaults["max_pixels"]))
    try:
        fps = float(hinted("fps", defaults["fps"]))
    except (TypeError, ValueError, OverflowError):
        fps = float(defaults["fps"])
    if not math.isfinite(fps):
        fps = float(defaults["fps"])
    try:
        max_new_tokens = max(0, int(hinted("max_new_tokens", defaults["max_new_tokens"])))
    except (TypeError, ValueError, OverflowError):
        max_new_tokens = max(0, int(defaults["max_new_tokens"]))
    try:
        context_tokens = max(
            0,
            int(
                spec.limits.context_tokens
                if budget_hint.context_tokens is None
                else budget_hint.context_tokens
            ),
        )
    except (TypeError, ValueError, OverflowError):
        context_tokens = max(0, int(spec.limits.context_tokens))

    media_kinds = {str(kind).strip().casefold() for kind in budget_hint.media_kinds}
    capabilities = {str(kind).casefold() for kind in spec.capabilities}
    family_vision = bool(capabilities & {"video", "video_audio", "image"})
    family_audio = bool(capabilities & {"video_audio", "audio"})
    include_vision = family_vision and (not media_kinds or bool(media_kinds & {"video", "image"}))
    include_audio = family_audio and (not media_kinds or bool(media_kinds & {"video", "audio"}))
    token_px = 32 * 32 if qwen3 else 28 * 28
    vision_tokens = (
        math.ceil(max_frames / 2) * math.ceil(max_pixels / token_px)
        if include_vision
        else 0
    )
    vision_patches = vision_tokens * 4
    audio_rate = 13 if qwen3 else 25
    audio_tokens = audio_rate * (max_frames / max(fps, 0.25)) if include_audio else 0.0
    prompt_tokens = vision_tokens + audio_tokens + 256
    seq_tokens = min(context_tokens, prompt_tokens + max_new_tokens)

    gib = 2**30
    base = 0.75 * gib
    kv_bytes = seq_tokens * layers * 2 * kv_heads * head_dim * 2
    # Measured on an RTX 5090 with last-token logits: ~16 KiB/patch (Qwen3-Omni) and
    # ~24 KiB/patch (Qwen2.5-Omni) of vision-tower activation peak; 32 KiB keeps margin.
    vision_peak = vision_patches * (32 * 1024)
    logits_bytes = 8 * vocab_size * 4
    moe_prefill = prompt_tokens * 8 * 2 * 768 * 2 if qwen3 else 0.0
    estimate = base + kv_bytes + vision_peak + logits_bytes + moe_prefill
    try:
        ratio = float(observed_ratio)
    except (TypeError, ValueError, OverflowError):
        ratio = 1.0
    if not math.isfinite(ratio):
        ratio = 1.0
    estimate *= min(OBSERVED_RATIO_MAX, max(OBSERVED_RATIO_MIN, ratio))
    try:
        observed = max(0, int(observed_bytes))
    except (TypeError, ValueError, OverflowError):
        observed = 0
    estimate = max(estimate, observed)
    return min(16 * gib, max(gib, int(math.ceil(estimate))))


# Bounds for the per-variant observed/planned activation ratio applied to fresh estimates.
OBSERVED_RATIO_MIN = 0.75
OBSERVED_RATIO_MAX = 4.0


def observed_activation_ratio(variant_key: str) -> float:
    """Return the recorded observed/planned activation ratio for a variant (1.0 when unknown)."""

    try:
        path = Path(TEMP_DIR) / "vram_activation_cache.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        entry = payload.get(str(variant_key)) if isinstance(payload, dict) else None
        ratio = float(entry.get("ratio", 1.0)) if isinstance(entry, dict) else 1.0
    except (OSError, TypeError, ValueError, OverflowError, json.JSONDecodeError):
        return 1.0
    if not math.isfinite(ratio) or ratio <= 0:
        return 1.0
    return min(OBSERVED_RATIO_MAX, max(OBSERVED_RATIO_MIN, ratio))


# The fixed allocator allowance comes from ``OffloadPlan.plan_slack_mib``. Cached-but-
# unallocated segments also scale with the transient prefill volume: measured at
# 0.6-0.85x the activation peak across the tier matrix on Windows (no expandable segments).
ALLOCATOR_SLACK_RATIO = 0.5


def plan_block_swap(
    plan: OffloadPlan,
    layout: CheckpointLayout,
    *,
    free_vram_bytes: int,
    total_vram_bytes: int,
    activation_bytes: int,
    ram_available_bytes: int | None = None,
    ram_total_bytes: int | None = None,
) -> BlockSwapBudget:
    """Resolve the resident/swapped decoder-layer split for one load."""

    gib = 2**30
    mib = 2**20
    layer_count = max(0, int(layout.layer_count))
    layer_bytes = max(layout.layer_bytes, default=0)
    non_layer_bytes = max(0, int(layout.non_layer_bytes))
    free = max(0, int(free_vram_bytes))
    total = max(0, int(total_vram_bytes))
    activation = max(0, int(activation_bytes))
    slack = int(plan.plan_slack_mib) * mib + int(ALLOCATOR_SLACK_RATIO * activation)
    reserve = max(0, int(float(plan.vram_reserve_gb) * gib))
    tower_bytes = max(0, min(int(getattr(layout, "tower_bytes", 0) or 0), non_layer_bytes))
    # With the towers staged on CPU between prefills, the decode phase no longer holds them;
    # the prefill phase then peaks at towers + encoder activations instead. Encoder
    # activations are bounded by the (vision-dominated) activation estimate.
    phase_staged = max(activation, tower_bytes + min(activation, gib))
    notes: list[str] = []
    stage_towers = False

    def auto_split(dense_bytes: int, phase_bytes: int) -> tuple[int, int, int, str]:
        weights_budget = free - reserve - phase_bytes - slack
        if weights_budget - dense_bytes >= layer_count * layer_bytes:
            return layer_count, 0, 0, "resident"
        slots_ = plan.swap_slots
        if layer_bytes > 0:
            resident_ = (weights_budget - dense_bytes - slots_ * layer_bytes) // layer_bytes
            resident_ = min(layer_count - 1, max(0, int(resident_)))
        else:
            resident_ = max(0, layer_count - 1)
        swapped_ = layer_count - resident_
        slots_ = min(plan.swap_slots, swapped_) if swapped_ > 0 else 0
        return resident_, swapped_, slots_, ("block_swap" if swapped_ > 0 else "resident")

    if plan.uses_legacy_offload:
        resident = layer_count
        swapped = 0
        slots = 0
        mode = "legacy"
    elif plan.gpu_layers == "all":
        resident = layer_count
        swapped = 0
        slots = 0
        resident_weight = non_layer_bytes + resident * layer_bytes
        expected_peak = resident_weight + activation + slack
        mode = "resident" if expected_peak <= free - reserve else "forced_resident"
    elif isinstance(plan.gpu_layers, int):
        resident = min(plan.gpu_layers, layer_count)
        swapped = layer_count - resident
        slots = min(plan.swap_slots, swapped) if swapped > 0 else 0
        mode = "block_swap" if swapped > 0 else "resident"
        if swapped > 0 and tower_bytes > 0:
            plain_peak = non_layer_bytes + (resident + slots) * layer_bytes + activation + slack
            staged_peak = (non_layer_bytes - tower_bytes) + (resident + slots) * layer_bytes + phase_staged + slack
            stage_towers = plain_peak > free - reserve and staged_peak < plain_peak
    else:
        resident, swapped, slots, mode = auto_split(non_layer_bytes, activation)
        if swapped > 0 and tower_bytes > 0:
            staged = auto_split(non_layer_bytes - tower_bytes, phase_staged)
            plain_peak = non_layer_bytes + (resident + slots) * layer_bytes + activation + slack
            staged_peak = (
                (non_layer_bytes - tower_bytes) + (staged[0] + staged[2]) * layer_bytes + phase_staged + slack
            )
            better_layers = staged[0] > resident
            same_layers_lower_peak = staged[0] == resident and staged[1] > 0 and staged_peak < plain_peak
            if better_layers or same_layers_lower_peak:
                resident, swapped, slots, mode = staged
                stage_towers = mode == "block_swap"

    dense_bytes = non_layer_bytes - tower_bytes if stage_towers else non_layer_bytes
    phase_bytes = phase_staged if stage_towers else activation
    resident_weight_bytes = dense_bytes + (resident + slots) * layer_bytes
    expected_peak_bytes = resident_weight_bytes + phase_bytes + slack
    pinned_bytes = swapped * layer_bytes

    if swapped > 0:
        summary = (
            f"Block swap: {resident}/{layer_count} decoder layers resident, {swapped} swapped "
            f"({pinned_bytes / gib:.2f} GiB pinned), {slots} slots x {layer_bytes / mib:.0f} MiB; "
            f"GPU weights {resident_weight_bytes / gib:.1f} GiB; activation estimate "
            f"{activation / gib:.1f} GiB; allocator slack {slack / gib:.1f} GiB; reserve "
            f"{reserve / gib:.1f} GiB; expected peak {expected_peak_bytes / gib:.1f} of "
            f"{free / gib:.1f} GiB free"
            + (
                f"; towers staged on CPU between prefills ({tower_bytes / gib:.1f} GiB)"
                if stage_towers
                else ""
            )
        )
    else:
        label = {
            "forced_resident": "Forced resident",
            "legacy": "Legacy offload",
            "resident": "Resident",
        }[mode]
        summary = (
            f"{label}: {resident}/{layer_count} decoder layers resident; GPU weights "
            f"{resident_weight_bytes / gib:.1f} GiB; activation estimate {activation / gib:.1f} GiB; "
            f"allocator slack {slack / gib:.1f} GiB; reserve {reserve / gib:.1f} GiB; expected peak "
            f"{expected_peak_bytes / gib:.1f} of {free / gib:.1f} GiB free"
        )
    notes.append(summary)

    if mode == "legacy":
        notes.append(
            "Legacy Accelerate offload is enabled by offload_experts or max_memory; block swap is disabled."
        )
    budget_limit = free - reserve
    overflow = expected_peak_bytes - budget_limit
    if mode == "forced_resident":
        notes.append(
            f"expected peak {expected_peak_bytes / gib:.1f} GiB exceeds free VRAM minus reserve "
            f"by {overflow / gib:.1f} GiB; Windows will page into shared memory "
            "\u2014 use auto or a smaller layer count"
        )
    elif isinstance(plan.gpu_layers, int) and not plan.uses_legacy_offload and overflow > 0:
        notes.append(
            f"expected peak {expected_peak_bytes / gib:.1f} GiB exceeds free VRAM minus reserve "
            f"by {overflow / gib:.1f} GiB; respecting the requested resident layer count, but "
            "Windows may page into shared memory"
        )
    elif plan.gpu_layers == "auto" and swapped > 0 and resident == 0 and overflow > 0:
        notes.append(
            f"Even with zero resident decoder layers, the minimum block-swap footprint exceeds "
            f"free VRAM minus reserve by {overflow / gib:.1f} GiB; Windows may page into shared memory"
        )

    if ram_available_bytes is not None:
        configured_limit = pinned_ram_budget_bytes(
            plan,
            ram_total_bytes if ram_total_bytes is not None else ram_available_bytes,
        )
        safe_ram = min(max(0, int(ram_available_bytes)), max(0, configured_limit))
        if pinned_bytes > safe_ram:
            notes.append(
                f"Swapped layers need {pinned_bytes / gib:.2f} GiB of pinned RAM, but only "
                f"{max(0, safe_ram) / gib:.2f} GiB is allowed by the pinned RAM budget and "
                "currently available; "
                "some layers will stay pageable."
            )

    return BlockSwapBudget(
        layer_count=layer_count,
        resident_layers=resident,
        swapped_layers=swapped,
        slots=slots,
        layer_bytes=layer_bytes,
        non_layer_bytes=non_layer_bytes,
        resident_weight_bytes=resident_weight_bytes,
        activation_bytes=activation,
        reserve_bytes=reserve,
        free_vram_bytes=free,
        total_vram_bytes=total,
        expected_peak_bytes=expected_peak_bytes,
        pinned_bytes=pinned_bytes,
        mode=mode,
        notes=tuple(notes),
        allocator_slack_bytes=slack,
        stage_towers=stage_towers,
        tower_bytes=tower_bytes,
    )


def observed_activation_bytes(variant_key: str) -> int:
    """Return the largest observed non-weight peak for a variant, or 0."""

    try:
        path = Path(TEMP_DIR) / "vram_activation_cache.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return 0
        entry = payload.get(str(variant_key), 0)
        if isinstance(entry, dict):
            entry = entry.get("bytes", 0)
        return max(0, int(entry))
    except (OSError, TypeError, ValueError, OverflowError, json.JSONDecodeError):
        return 0


def record_observed_activation_bytes(variant_key: str, value: int, planned_bytes: int = 0) -> None:
    """Persist an observed non-weight peak (and its ratio to the plan) for a variant."""

    try:
        key = str(variant_key)
        observed = max(0, int(value))
        planned = max(0, int(planned_bytes or 0))
        path = Path(TEMP_DIR) / "vram_activation_cache.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                payload = {}
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            payload = {}

        current = payload.get(key, 0)
        if isinstance(current, dict):
            current = current.get("bytes", 0)
        try:
            maximum = max(observed, max(0, int(current)))
        except (TypeError, ValueError, OverflowError):
            maximum = observed
        entry: dict[str, Any] = {"bytes": maximum, "timestamp": time.time()}
        previous = payload.get(key)
        if planned > 0:
            # Blend towards the latest observation so one unusual clip does not dominate.
            fresh = min(OBSERVED_RATIO_MAX, max(OBSERVED_RATIO_MIN, observed / planned))
            try:
                old_ratio = float(previous.get("ratio")) if isinstance(previous, dict) else None
            except (TypeError, ValueError):
                old_ratio = None
            ratio = fresh if old_ratio is None else max(fresh, 0.5 * old_ratio + 0.5 * fresh)
            entry["ratio"] = round(ratio, 4)
            entry["planned_bytes"] = planned
            entry["observed_bytes"] = observed
        elif isinstance(previous, dict) and "ratio" in previous:
            entry["ratio"] = previous["ratio"]
        payload[key] = entry
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except Exception:
        return


__all__ = [
    "BlockSwapBudget",
    "BudgetHint",
    "CheckpointLayout",
    "DeviceMapPlan",
    "OffloadPlan",
    "block_swap_to_gpu_layers",
    "build_device_map",
    "cached_checkpoint_layout",
    "checkpoint_layout",
    "estimate_activation_bytes",
    "estimate_layers_on_gpu",
    "family_layer_count",
    "gpu_layers_to_block_swap",
    "migrate_legacy_gpu_layers",
    "observed_activation_bytes",
    "observed_activation_ratio",
    "plan_block_swap",
    "plan_model_folder",
    "planning_config",
    "pinned_ram_budget_bytes",
    "record_observed_activation_bytes",
    "vram_budget_gb",
]
