"""Local-only thinker model loading, streaming checkpoints, and one-model cache."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import gc
from itertools import chain
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Callable
import weakref

from vcap import TEMP_DIR
from vcap.core.gpu import resource_snapshot, vram_cap_env_disabled
from vcap.core.logs import get_log

from .attention import is_flash_attention_failure, resolve as resolve_attention
from .block_swap import BlockSwapManager
from .offload import (
    BudgetHint,
    OffloadPlan,
    _FAMILY_LAYOUTS,
    build_device_map,
    checkpoint_layout,
    estimate_activation_bytes,
    observed_activation_bytes,
    observed_activation_ratio,
    plan_block_swap,
)
from .registry import MODEL_SPECS, ModelSpec, VariantSpec, get_variant, resolve_model_dir, variant_is_ready, variant_to_family
from .torch_compile import DEFAULT_COMPILE_MODE


ProgressCallback = Callable[..., None]
_MIB = 2**20
_GIB = 2**30


@dataclass(frozen=True)
class LoadReport:
    """Timing, memory, checkpoint, and placement data from model loading."""

    seconds: float
    peak_vram_gb: float
    checkpoint_bytes: int
    attention: str
    device_map: dict[str, Any] | str | None
    quantized_layers: int = 0
    bf16_layers: int = 0
    compile_mode: str = "eager"
    warnings: tuple[str, ...] = ()
    block_swap: dict[str, Any] | None = None
    resident_bytes: int = 0
    activation_estimate_bytes: int = 0
    vram_cap_bytes: int = 0


@dataclass(frozen=True)
class UnloadReport:
    """Observed device-memory change after releasing a model."""

    seconds: float
    vram_before_gb: float
    vram_after_gb: float
    freed_vram_gb: float
    variant_key: str | None = None
    backend: str = ""
    released: bool = True
    host_before_gb: float = 0.0
    host_after_gb: float = 0.0
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe release report."""

        value = asdict(self)
        value["notes"] = list(self.notes)
        return value


@dataclass(eq=False)
class LoadedModel:
    """Resident model, processor, registry metadata, and load diagnostics."""

    model: Any
    processor: Any
    spec: ModelSpec
    variant: VariantSpec
    load_report: LoadReport
    device: str = "cuda:0"
    dtype: Any = None
    attention: str = "sdpa"
    offload: OffloadPlan = field(default_factory=OffloadPlan)
    model_dir: Path | None = None
    gpu_index: int = 0


def _emit(callback: ProgressCallback | None, message: str, fraction: float | None = None) -> None:
    get_log().log(message, scope="models")
    if callback is None:
        return
    payload = {"message": message, "fraction": fraction}
    for args in ((message, fraction), (message,), (payload,)):
        try:
            callback(*args)
            return
        except TypeError:
            continue


def _install_last_token_logits_hook(model: Any, enabled: bool = True) -> Any | None:
    """Slice the language-model head input to the final sequence position."""

    model._vcap_last_token_logits = bool(enabled)
    lm_head = getattr(model, "lm_head", None)
    register = getattr(lm_head, "register_forward_pre_hook", None)
    if not callable(register):
        return None
    existing = getattr(model, "_vcap_last_token_logits_hook", None)
    remove = getattr(existing, "remove", None)
    if callable(remove):
        remove()
    root_ref = weakref.ref(model)

    def slice_last_token(_module: Any, args: tuple[Any, ...]) -> tuple[Any, ...] | None:
        root = root_ref()
        if root is None or not getattr(root, "_vcap_last_token_logits", True) or not args:
            return None
        hidden_states = args[0]
        if getattr(hidden_states, "ndim", 0) == 3 and hidden_states.shape[1] > 1:
            return (hidden_states[:, -1:, :],)
        return None

    handle = register(slice_last_token)
    model._vcap_last_token_logits_hook = handle
    return handle


def _trim_host_working_set() -> None:
    """Return cached checkpoint pages to Windows after a large streamed load.

    Streaming a checkpoint through safetensors leaves tens of GB of file pages in the
    worker's working set next to the page-locked block-swap packs. They are only cache,
    but until Windows trims them a following child process (ffprobe/ffmpeg) can fail to
    start under memory pressure. Locked pages are unaffected; touched private pages are
    soft-faulted back from the standby list.
    """

    if os.name != "nt":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        psapi.EmptyWorkingSet(kernel32.GetCurrentProcess())
    except Exception:
        pass


def _process_private_gb() -> float:
    import psutil

    memory = psutil.Process().memory_info()
    private = getattr(memory, "private", None)
    value = private if private is not None else getattr(memory, "rss", 0)
    return float(value or 0) / _GIB


def _remove_model_runtime(model: Any) -> tuple[str, ...]:
    notes: list[str] = []
    try:
        manager = getattr(model, "_vcap_block_swap_manager", None)
    except Exception as exc:
        manager = None
        notes.append(f"Could not inspect the block-swap manager: {exc}")
    remove_manager = getattr(manager, "remove", None)
    if callable(remove_manager):
        try:
            remove_manager()
        except Exception as exc:
            message = f"Could not remove block-swap manager cleanly: {exc}"
            notes.append(message)
            get_log().warn(message, scope="models")
    try:
        hook = getattr(model, "_vcap_last_token_logits_hook", None)
    except Exception as exc:
        hook = None
        notes.append(f"Could not inspect the last-token hook: {exc}")
    remove_hook = getattr(hook, "remove", None)
    if callable(remove_hook):
        try:
            remove_hook()
        except Exception as exc:
            notes.append(f"Could not remove the last-token hook: {exc}")
    for name in ("_vcap_last_token_logits_hook", "_vcap_last_token_logits"):
        try:
            delattr(model, name)
        except (AttributeError, TypeError):
            pass
        except Exception as exc:
            notes.append(f"Could not clear {name}: {exc}")
    return tuple(notes)


def _block_swap_layers(model: Any, family: str, expected_count: int) -> Any:
    language_model = getattr(model, "model", None)
    layers = getattr(language_model, "layers", None)
    if layers is None:
        raise RuntimeError("Block swap requires decoder layers at model.model.layers")
    if len(layers) != int(expected_count):
        raise RuntimeError(
            f"Block-swap layout expected {expected_count} decoder layers, found {len(layers)}"
        )
    expected_class = _FAMILY_LAYOUTS[family][1][-1]
    mismatch = next(
        (
            (index, layer.__class__.__name__)
            for index, layer in enumerate(layers)
            if layer.__class__.__name__ != expected_class
        ),
        None,
    )
    if mismatch is not None:
        index, actual = mismatch
        raise RuntimeError(
            f"Block-swap decoder layer {index} is {actual}, expected {expected_class}"
        )
    return layers


def _apply_vram_hard_cap(
    torch: Any,
    torch_device: Any,
    *,
    total_vram_bytes: int,
    other_vram_bytes: int,
) -> int:
    if (
        vram_cap_env_disabled()
        or getattr(torch_device, "type", None) != "cuda"
        or not torch.cuda.is_available()
        or total_vram_bytes <= 0
    ):
        return 0
    cap = max(0, int(total_vram_bytes) - int(other_vram_bytes) - 256 * _MIB)
    fraction = min(1.0, max(0.05, cap / int(total_vram_bytes)))
    try:
        torch.cuda.set_per_process_memory_fraction(fraction, torch_device)
    except (AttributeError, RuntimeError) as exc:
        get_log().warn(f"Could not apply the CUDA allocator VRAM cap: {exc}", scope="models")
        return 0
    return int(int(total_vram_bytes) * fraction)


def _checkpoint_bytes(folder: Path) -> int:
    checkpoint = folder / "model.safetensors"
    if checkpoint.is_file():
        return checkpoint.stat().st_size
    return sum(path.stat().st_size for path in folder.glob("*.safetensors") if path.is_file())


def _normalize_device(device: str | int) -> tuple[str, int]:
    """Return a Torch CUDA device string and its process-local device index."""

    value = str(device).strip().casefold()
    if value in {"cuda", ""}:
        return "cuda:0", 0
    if value.isdigit():
        index = int(value)
        return f"cuda:{index}", index
    if value.startswith("cuda:") and value[5:].isdigit():
        index = int(value[5:])
        return f"cuda:{index}", index
    if value == "cpu":
        return "cpu", 0
    raise ValueError(f"Unsupported model device: {device!r}")


def _visible_physical_gpus() -> tuple[int, ...] | None:
    """Return the numeric physical-to-local CUDA visibility mapping, when known."""

    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw:
        return None
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values or any(not value.isdigit() for value in values):
        return None
    return tuple(int(value) for value in values)


def _selected_physical_gpu(explicit: int | None, local_index: int) -> int:
    if explicit is not None:
        return max(0, int(explicit))
    worker_gpu = os.environ.get("VCAP_WORKER_GPU", "").strip()
    if worker_gpu.isdigit():
        return int(worker_gpu)
    visible = _visible_physical_gpus()
    if visible is not None and 0 <= int(local_index) < len(visible):
        return visible[int(local_index)]
    return max(0, int(local_index))


def _resolve_gpu_selection(
    device: str | int,
    gpu_index: int | None,
) -> tuple[str, int, int]:
    """Make an explicit physical GPU selection authoritative over a default device."""

    normalized, local_index = _normalize_device(device)
    physical_index = _selected_physical_gpu(gpu_index, local_index)
    if normalized == "cpu" or gpu_index is None:
        return normalized, local_index, physical_index

    worker_gpu = os.environ.get("VCAP_WORKER_GPU", "").strip()
    if worker_gpu.isdigit() and int(worker_gpu) == physical_index:
        selected_local = 0
    else:
        visible = _visible_physical_gpus()
        if visible is None:
            selected_local = physical_index
        else:
            try:
                selected_local = visible.index(physical_index)
            except ValueError as exc:
                raise RuntimeError(
                    f"Selected physical GPU {physical_index} is not present in "
                    f"CUDA_VISIBLE_DEVICES={','.join(map(str, visible))}"
                ) from exc
    return f"cuda:{selected_local}", selected_local, physical_index


def _enforce_dev_gpu_guard(physical_gpu_index: int) -> None:
    """Optionally pin development runs without constraining shipped builds."""

    forced = os.environ.get("VCAP_DEV_FORCE_GPU", "").strip()
    if not forced:
        return
    try:
        required = int(forced)
    except ValueError as exc:
        raise RuntimeError("VCAP_DEV_FORCE_GPU must be an integer GPU index") from exc
    if int(physical_gpu_index) != required:
        raise RuntimeError(
            f"VCAP_DEV_FORCE_GPU={required} rejects selected physical GPU {physical_gpu_index}"
        )


def _gguf_checkpoint_bytes(folder: Path, variant: VariantSpec) -> int:
    return sum(
        (folder / name).stat().st_size
        for name in (variant.gguf_files or ())
        if (folder / name).is_file()
    )


def _load_llamacpp_model(
    variant_key: str,
    *,
    device: str,
    device_index: int,
    gpu_index: int,
    offload: OffloadPlan | None,
    progress_cb: ProgressCallback | None,
    model_dir: str | os.PathLike[str] | None,
    context_size: int | None = None,
) -> LoadedModel:
    """Start a local llama-server and wrap it in the common load contract."""

    from .llamacpp_backend import LlamaCppCaptioner
    from .llamacpp_install import ensure_llamacpp

    started = time.perf_counter()
    family = variant_to_family(variant_key)
    spec = MODEL_SPECS[family]
    variant = get_variant(variant_key)
    folder = Path(model_dir).expanduser().resolve(strict=False) if model_dir else resolve_model_dir(variant.key)
    if model_dir is None:
        ready, detail = variant_is_ready(variant.key)
        if not ready:
            raise FileNotFoundError(f"{variant.key} is not ready: {detail}")
    elif not folder.is_dir() or any(not (folder / name).is_file() for name in (variant.gguf_files or ())):
        raise FileNotFoundError(f"GGUF model files are incomplete in {folder}")
    _emit(progress_cb, f"Loading {spec.label} / {variant.label} with llama.cpp", 0.0)
    server = ensure_llamacpp(progress_cb)
    plan = offload or OffloadPlan()
    backend = LlamaCppCaptioner(
        family,
        variant_key=variant.key,
        server_path=server,
        model_dir=folder,
        device_index=device_index,
        gpu_index=gpu_index,
        vram_total_gb=float(
            resource_snapshot(gpu_index).get("vram_total_gb", 0.0) or 0.0
        ),
        # The UI's "VRAM to keep free" sets llama.cpp's fit target; the backend's
        # own default must never stand in for the value the user configured.
        vram_reserve_gb=float(plan.vram_reserve_gb),
        # The requested window; the tier plan and llama.cpp's fitter may shrink it.
        **({"context_size": int(context_size)} if context_size else {}),
    )
    backend.start(progress_cb)
    report = LoadReport(
        seconds=time.perf_counter() - started,
        peak_vram_gb=float(backend.load_peak_vram_gb),
        checkpoint_bytes=_gguf_checkpoint_bytes(folder, variant),
        attention="llamacpp",
        device_map={"": str(device)},
        compile_mode="llama.cpp",
        block_swap=_llamacpp_fit_summary(backend),
    )
    loaded = LoadedModel(
        backend,
        backend,
        spec,
        variant,
        report,
        str(device),
        None,
        "llamacpp",
        plan,
        folder,
        gpu_index,
    )
    backend.loaded = loaded
    _SWITCH_REGISTRY.add(loaded)
    _emit(
        progress_cb,
        f"Model ready in {report.seconds:.1f}s; peak GPU {gpu_index} VRAM {report.peak_vram_gb:.2f} GiB",
        1.0,
    )
    return loaded


def _llamacpp_fit_summary(backend: Any) -> dict[str, Any] | None:
    summary = getattr(backend, "block_swap_summary", None)
    if not callable(summary):
        return None
    try:
        result = summary()
    except Exception as exc:
        get_log().warn(f"Could not read the llama.cpp fit report: {exc}", scope="llama.cpp")
        return None
    return dict(result) if isinstance(result, dict) else None


def _model_types(architecture: str) -> tuple[Any, Any]:
    if architecture == "qwen2_5_omni_thinker":
        from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

        return Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
    if architecture == "qwen3_omni_moe_thinker":
        from transformers import Qwen3OmniMoeProcessor, Qwen3OmniMoeThinkerForConditionalGeneration

        return Qwen3OmniMoeThinkerForConditionalGeneration, Qwen3OmniMoeProcessor
    raise ValueError(f"Unsupported model architecture: {architecture}")


def _thinker_config(folder: Path, architecture: str, attention: str, dtype: Any) -> Any:
    from transformers import AutoConfig

    outer = AutoConfig.from_pretrained(folder, local_files_only=True, trust_remote_code=False)
    config = getattr(outer, "thinker_config", outer)
    expected = "qwen2_5_omni_thinker" if architecture == "qwen2_5_omni_thinker" else "qwen3_omni_moe_thinker"
    if getattr(config, "model_type", None) != expected:
        raise RuntimeError(
            f"Checkpoint config did not expose {expected}; got {getattr(config, 'model_type', type(config).__name__)}"
        )
    config._attn_implementation = attention
    config.dtype = dtype
    return config


def _processor(processor_class: Any, folder: Path, spec: ModelSpec) -> Any:
    kwargs: dict[str, Any] = {"local_files_only": True, "trust_remote_code": False}
    if spec.architecture == "qwen3_omni_moe_thinker":
        kwargs.update(min_pixels=spec.limits.min_pixels, max_pixels=1280 * 32 * 32)
    return processor_class.from_pretrained(folder, **kwargs)


def _special_token_id(processor: Any, token: str, fallback: int) -> int:
    tokenizer = getattr(processor, "tokenizer", None)
    for candidate in (tokenizer, processor):
        convert = getattr(candidate, "convert_tokens_to_ids", None)
        if not callable(convert):
            continue
        try:
            value = convert(token)
        except Exception:
            continue
        if isinstance(value, int) and value >= 0:
            unknown_id = getattr(candidate, "unk_token_id", None)
            unknown_token = getattr(candidate, "unk_token", None)
            if unknown_id is not None and value == unknown_id and token != unknown_token:
                continue
            return int(value)
    return int(fallback)


def resolve_stop_token_ids(processor: Any) -> tuple[list[int], int]:
    """Resolve the two Qwen chat terminators and the tokenizer padding id."""

    im_end = _special_token_id(processor, "<|im_end|>", 151_645)
    endoftext = _special_token_id(processor, "<|endoftext|>", 151_643)
    tokenizer = getattr(processor, "tokenizer", None)
    tokenizer_pad = getattr(tokenizer, "pad_token_id", None)
    pad_token_id = int(tokenizer_pad) if isinstance(tokenizer_pad, int) and tokenizer_pad >= 0 else endoftext
    return [im_end, endoftext], pad_token_id


def _load_generation_config(model: Any, folder: Path, processor: Any, spec: ModelSpec) -> None:
    """Install local config plus authoritative family defaults and stop tokens."""

    from transformers import GenerationConfig

    generation_config = getattr(model, "generation_config", None)
    if (folder / "generation_config.json").is_file():
        try:
            generation_config = GenerationConfig.from_pretrained(folder, local_files_only=True)
        except Exception as exc:
            get_log().warn(f"Could not load generation_config.json; using model defaults: {exc}", scope="models")
    if generation_config is None:
        generation_config = GenerationConfig.from_model_config(model.config)

    defaults = {item.name: item.default for item in spec.param_schema}
    for name in (
        "do_sample",
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "max_new_tokens",
    ):
        if name in defaults:
            setattr(generation_config, name, defaults[name])
    eos_token_ids, pad_token_id = resolve_stop_token_ids(processor)
    generation_config.eos_token_id = eos_token_ids
    generation_config.pad_token_id = pad_token_id
    model.generation_config = generation_config

    # Keep model config coherent for helpers that clone GenerationConfig later.
    model.config.eos_token_id = eos_token_ids
    model.config.pad_token_id = pad_token_id


def _stream_single_checkpoint(
    folder: Path,
    model_class: Any,
    config: Any,
    *,
    target_device: str,
    dtype: Any,
    progress_cb: ProgressCallback | None,
    layer_device: Callable[[int], Any] | None = None,
    tower_offload: bool | None = None,
) -> tuple[Any, Any]:
    try:
        from accelerate import init_empty_weights
    except ImportError as exc:
        raise RuntimeError("accelerate is required for streamed single-file model loading") from exc
    try:
        from .quant.convrot import apply_quantized_checkpoint
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "The ConvRot checkpoint loader is unavailable. Finish task T4 or reinstall vcap.models.quant.convrot."
        ) from exc

    with init_empty_weights(include_buffers=True):
        model = model_class(config)
    last_percent = -1
    last_time = 0.0

    def on_tensor(done: int, total: int, name: str) -> None:
        nonlocal last_percent, last_time
        percent = int(done * 100 / max(total, 1))
        now = time.monotonic()
        if percent >= last_percent + 2 or now - last_time >= 2 or done >= total:
            last_percent = percent
            last_time = now
            _emit(progress_cb, f"Loading checkpoint {percent:3d}% - {name}", done / max(total, 1))

    try:
        report = apply_quantized_checkpoint(
            model,
            folder / "model.safetensors",
            device=target_device,
            dtype=dtype,
            progress_cb=on_tensor,
            strip_prefix="thinker.",
            layer_device=layer_device,
            tower_offload=tower_offload,
        )
    except (AttributeError, TypeError) as exc:
        raise RuntimeError(
            "The installed ConvRot loader does not match the required checkpoint API "
            "with per-layer placement support."
        ) from exc
    return model, report


def _plain_sharded_load(
    folder: Path,
    model_class: Any,
    *,
    dtype: Any,
    attention: str,
    placement: Any | None = None,
    device_map: dict[str, Any] | None = None,
) -> Any:
    kwargs: dict[str, Any] = {
        "dtype": dtype,
        "attn_implementation": attention,
        "local_files_only": True,
        "trust_remote_code": False,
        "low_cpu_mem_usage": True,
    }
    if device_map is not None:
        kwargs["device_map"] = device_map
    elif placement is not None:
        kwargs.update(placement.from_pretrained_kwargs())
    else:
        raise ValueError("A placement plan or explicit device map is required")
    return model_class.from_pretrained(folder, **kwargs)


def _pin_cpu_tensors(model: Any, max_fraction: float = 0.40) -> int:
    try:
        import psutil

        budget = int(psutil.virtual_memory().total * max(0.0, min(1.0, max_fraction)))
    except Exception:
        budget = 8 * 1024**3
    pinned = 0
    tensors = chain(model.parameters(), model.buffers())
    for tensor in tensors:
        if tensor.device.type != "cpu" or tensor.is_meta or not tensor.is_contiguous():
            continue
        size = tensor.numel() * tensor.element_size()
        if pinned + size > budget:
            break
        try:
            tensor.data = tensor.data.pin_memory()
            pinned += size
        except (RuntimeError, OSError):
            break
    return pinned


def _dispatch_after_cpu_load(
    model: Any,
    family: str,
    plan: OffloadPlan,
    placement: Any,
    dtype: Any,
    *,
    device_index: int,
) -> dict[str, Any]:
    from accelerate import dispatch_model, infer_auto_device_map

    if plan.gpu_layers == "all" and plan.offload_experts:
        device_map = placement.explicit_device_map
    elif plan.gpu_layers != "all":
        device_map = placement.explicit_device_map
    else:
        device_map = infer_auto_device_map(
            model,
            max_memory=placement.max_memory,
            no_split_module_classes=list(placement.no_split_modules),
            dtype=dtype,
            offload_buffers=False,
            fallback_allocation=True,
        )
    offload_dir = TEMP_DIR / "model_offload" / family
    offload_dir.mkdir(parents=True, exist_ok=True)
    dispatch_model(
        model,
        device_map=device_map,
        main_device=int(device_index),
        offload_dir=offload_dir,
        offload_buffers=False,
        force_hooks=True,
    )
    model.hf_device_map = device_map
    return device_map


def load_model(
    variant_key: str,
    *,
    device: str = "cuda:0",
    gpu_index: int | None = None,
    attention: str = "auto",
    offload: OffloadPlan | None = None,
    dtype: Any = None,
    progress_cb: ProgressCallback | None = None,
    hf_dir: str | os.PathLike[str] | None = None,
    compile_model: bool = False,
    compile_mode: str = DEFAULT_COMPILE_MODE,
    budget_hint: BudgetHint | None = None,
    last_token_logits: bool = True,
) -> LoadedModel:
    """Load one local thinker checkpoint through its registered backend."""

    device, device_index, physical_gpu_index = _resolve_gpu_selection(device, gpu_index)
    _enforce_dev_gpu_guard(physical_gpu_index)
    family = variant_to_family(variant_key)
    spec = MODEL_SPECS[family]
    variant = get_variant(variant_key)
    if variant.scheme == "gguf":
        return _load_llamacpp_model(
            variant.key,
            device=device,
            device_index=device_index,
            gpu_index=physical_gpu_index,
            offload=offload,
            progress_cb=progress_cb,
            model_dir=hf_dir,
            context_size=int(getattr(budget_hint, "context_tokens", 0) or 0) or None,
        )
    import torch

    started = time.perf_counter()
    folder = Path(hf_dir).expanduser().resolve(strict=False) if hf_dir else resolve_model_dir(variant_key)
    if hf_dir is None:
        ready, detail = variant_is_ready(variant_key)
        if not ready:
            raise FileNotFoundError(f"{variant_key} is not ready: {detail}")
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    plan = offload or OffloadPlan()
    dtype = torch.bfloat16 if dtype is None else dtype
    resolved_attention, _ = resolve_attention(attention, family, dtype)
    requested_attention = str(attention or "auto").strip().lower().replace("-", "_")
    if requested_attention != "auto" and resolved_attention != requested_attention:
        _emit(
            progress_cb,
            f"Attention backend {requested_attention} is unavailable; loading with SDPA instead.",
        )
    model_class, processor_class = _model_types(spec.architecture)
    placement = None
    if plan.uses_legacy_offload:
        snapshot = resource_snapshot(physical_gpu_index)
        placement = build_device_map(
            family,
            plan,
            float(snapshot.get("vram_free_gb", 0.0) or 0.0),
            device_index=device_index,
            physical_gpu_index=physical_gpu_index,
        )
    torch_device = torch.device(device)
    cuda_load = bool(torch.cuda.is_available() and torch_device.type == "cuda")
    free_vram_bytes = 0
    total_vram_bytes = 0
    other_vram_bytes = 0
    if cuda_load:
        torch.cuda.reset_peak_memory_stats(torch_device)
        if not plan.uses_legacy_offload:
            free_vram_bytes, total_vram_bytes = (
                int(value) for value in torch.cuda.mem_get_info(torch_device)
            )
            reserved_at_start = int(torch.cuda.memory_reserved(torch_device))
            other_vram_bytes = max(
                0,
                total_vram_bytes - free_vram_bytes - reserved_at_start,
            )
    _emit(progress_cb, f"Loading {spec.label} / {variant.label} from {folder}", 0.0)

    checkpoint = folder / "model.safetensors"
    single_file = checkpoint.is_file()
    use_cpu_staging = single_file and plan.uses_legacy_offload
    planned_attention = resolved_attention
    planned_config = None
    activation_estimate = 0
    ram_available_bytes: int | None = None
    budget = None
    if not plan.uses_legacy_offload:
        planned_config = _thinker_config(
            folder,
            spec.architecture,
            planned_attention,
            dtype,
        )
        activation_estimate = int(
            estimate_activation_bytes(
                family,
                planned_config,
                budget_hint,
                observed_ratio=observed_activation_ratio(variant.key),
            )
        )
        if single_file:
            import psutil

            layout = checkpoint_layout(checkpoint)
            ram_available_bytes = int(psutil.virtual_memory().available)
            budget = plan_block_swap(
                plan,
                layout,
                free_vram_bytes=free_vram_bytes,
                total_vram_bytes=total_vram_bytes,
                activation_bytes=activation_estimate,
                ram_available_bytes=ram_available_bytes,
            )
            if budget.notes:
                _emit(progress_cb, budget.notes[0])
                for note in budget.notes[1:]:
                    get_log().warn(note, scope="models")
        else:
            message = (
                "Block swap requires a single-file model.safetensors checkpoint; "
                "loading this sharded checkpoint fully on the selected device."
            )
            _emit(progress_cb, message)

    attempt_model: Any | None = None

    def load_weights(
        selected_attention: str,
    ) -> tuple[Any, dict[str, Any] | str | None, int, int, Any | None]:
        nonlocal attempt_model
        config = (
            planned_config
            if planned_config is not None and selected_attention == planned_attention
            else _thinker_config(folder, spec.architecture, selected_attention, dtype)
        )
        manager = None
        if single_file:
            target = "cpu" if use_cpu_staging else str(device)
            per_layer = None
            tower_offload = None
            if not plan.uses_legacy_offload:
                assert budget is not None
                per_layer = lambda index: (
                    "cpu" if index >= budget.resident_layers else str(device)
                )
                if budget.stage_towers:
                    tower_offload = True
                else:
                    tower_offload = None if budget.swapped_layers == 0 else False
            loaded_model, stream_report = _stream_single_checkpoint(
                folder,
                model_class,
                config,
                target_device=target,
                dtype=dtype,
                progress_cb=progress_cb,
                layer_device=per_layer,
                tower_offload=tower_offload,
            )
            attempt_model = loaded_model
            selected_map: dict[str, Any] | str | None = {"": target}
            if use_cpu_staging:
                assert placement is not None
                selected_map = _dispatch_after_cpu_load(
                    loaded_model,
                    family,
                    plan,
                    placement,
                    dtype,
                    device_index=device_index,
                )
            elif budget is not None and budget.swapped_layers > 0:
                layers = _block_swap_layers(loaded_model, family, budget.layer_count)
                pin_budget = (
                    max(0, ram_available_bytes - 6 * _GIB)
                    if ram_available_bytes is not None
                    else None
                )
                manager = BlockSwapManager.install(
                    loaded_model,
                    layers,
                    resident=budget.resident_layers,
                    slots=budget.slots,
                    device=torch_device,
                    pin=plan.pin_cpu,
                    pin_budget_bytes=pin_budget,
                    log=lambda message: _emit(progress_cb, message),
                )
                selected_map = {"": str(device)}
            return (
                loaded_model,
                selected_map,
                int(getattr(stream_report, "quantized_layers", 0)),
                int(getattr(stream_report, "bf16_layers", 0)),
                manager,
            )
        if variant.scheme != "bf16":
            raise RuntimeError("Quantized variants must be self-contained single-file checkpoints")
        if plan.uses_legacy_offload:
            assert placement is not None
            loaded_model = _plain_sharded_load(
                folder,
                model_class,
                dtype=dtype,
                attention=selected_attention,
                placement=placement,
            )
            selected_map = getattr(loaded_model, "hf_device_map", placement.device_map)
        else:
            selected_map = {"": str(device)}
            loaded_model = _plain_sharded_load(
                folder,
                model_class,
                dtype=dtype,
                attention=selected_attention,
                device_map=selected_map,
            )
        attempt_model = loaded_model
        return loaded_model, selected_map, 0, 0, None

    try:
        model, device_map, quantized_layers, bf16_layers, swap_manager = load_weights(
            resolved_attention
        )
    except Exception as exc:
        if resolved_attention != "flash_attention_2" or not is_flash_attention_failure(exc):
            raise
        warning = f"FlashAttention 2 load failed ({exc}); retrying with SDPA."
        failed_model = attempt_model
        attempt_model = None
        if failed_model is not None:
            _remove_model_runtime(failed_model)
        failed_model = None
        exc.__traceback__ = None
        get_log().warn(warning, scope="models")
        _emit(progress_cb, warning)
        gc.collect()
        if torch.cuda.is_available() and device.startswith("cuda"):
            torch.cuda.empty_cache()
        resolved_attention = "sdpa"
        model, device_map, quantized_layers, bf16_layers, swap_manager = load_weights(
            resolved_attention
        )

    model.eval()
    if plan.pin_cpu and isinstance(device_map, dict) and any(str(value) == "cpu" for value in device_map.values()):
        pinned = _pin_cpu_tensors(model)
        _emit(progress_cb, f"Pinned {pinned / 1024**3:.2f} GiB of CPU-resident weights")
    vram_cap_bytes = 0
    resident_bytes = int(torch.cuda.memory_allocated(torch_device)) if cuda_load else 0
    block_swap_summary = None
    gc.collect()
    _trim_host_working_set()
    if not plan.uses_legacy_offload:
        vram_cap_bytes = _apply_vram_hard_cap(
            torch,
            torch_device,
            total_vram_bytes=total_vram_bytes,
            other_vram_bytes=other_vram_bytes,
        )
        _install_last_token_logits_hook(model, last_token_logits)
        if budget is not None:
            block_swap_summary = budget.summary()
            if swap_manager is not None:
                block_swap_summary = {
                    **block_swap_summary,
                    **swap_manager.summary(),
                }
    processor = _processor(processor_class, folder, spec)
    _load_generation_config(model, folder, processor, spec)

    compile_name = "eager"
    warnings: list[str] = []
    if compile_model:
        from .torch_compile import apply_compile, prepare_compile_env

        compile_plan = prepare_compile_env(True, mode=compile_mode, family=family)
        warnings.extend(compile_plan.warnings)
        model = apply_compile(model, compile_plan, progress_cb=progress_cb, family=family)
        if getattr(model, "_vcap_compile_plan", None) is not None:
            compile_name = compile_plan.requested_mode

    peak = (
        torch.cuda.max_memory_allocated(torch_device) / 1024**3
        if cuda_load
        else 0.0
    )
    report = LoadReport(
        seconds=time.perf_counter() - started,
        peak_vram_gb=float(peak),
        checkpoint_bytes=_checkpoint_bytes(folder),
        attention=resolved_attention,
        device_map=device_map,
        quantized_layers=quantized_layers,
        bf16_layers=bf16_layers,
        compile_mode=compile_name,
        warnings=tuple(warnings),
        block_swap=block_swap_summary,
        resident_bytes=resident_bytes,
        activation_estimate_bytes=(
            activation_estimate if not plan.uses_legacy_offload else 0
        ),
        vram_cap_bytes=vram_cap_bytes,
    )
    loaded = LoadedModel(
        model,
        processor,
        spec,
        variant,
        report,
        str(device),
        dtype,
        resolved_attention,
        plan,
        folder,
        physical_gpu_index,
    )
    _SWITCH_REGISTRY.add(loaded)
    _emit(
        progress_cb,
        f"Model ready on GPU {physical_gpu_index} in {report.seconds:.1f}s; "
        f"peak VRAM {report.peak_vram_gb:.2f} GiB",
        1.0,
    )
    return loaded


def unload_model(
    loaded: LoadedModel | None,
    *,
    wait_s: float = 5.0,
) -> UnloadReport:
    """Release a loaded model and report whether its resources were reclaimed."""

    if loaded is None:
        return UnloadReport(0.0, 0.0, 0.0, 0.0)

    started = time.perf_counter()
    notes: list[str] = []

    def note(message: str) -> None:
        notes.append(str(message))

    try:
        variant = loaded.variant
        raw_variant_key = getattr(variant, "key", None)
        variant_key = str(raw_variant_key) if raw_variant_key is not None else None
        backend = (
            "llamacpp"
            if str(getattr(variant, "backend", "")).casefold() == "llamacpp"
            else "transformers"
        )
    except Exception as exc:
        variant_key = None
        backend = "transformers"
        note(f"Could not read loaded-model metadata: {exc}")
    try:
        physical_gpu_index = int(getattr(loaded, "gpu_index", 0) or 0)
    except Exception as exc:
        physical_gpu_index = 0
        note(f"Could not read the model GPU index: {exc}")
    try:
        device_name = str(getattr(loaded, "device", "cuda:0") or "cuda:0")
    except Exception as exc:
        device_name = "cuda:0"
        note(f"Could not read the model device: {exc}")

    def vram_used(label: str) -> float | None:
        try:
            snapshot = resource_snapshot(physical_gpu_index)
            return float(snapshot.get("vram_used_gb", 0.0) or 0.0)
        except Exception as exc:
            note(f"Could not read VRAM {label}: {exc}")
            return None

    before = vram_used("before unload")
    before = 0.0 if before is None else before
    try:
        host_before = _process_private_gb()
    except Exception as exc:
        host_before = 0.0
        note(f"Could not read host memory before unload: {exc}")

    try:
        _SWITCH_REGISTRY.discard(loaded)
    except Exception as exc:
        note(f"Could not discard the model from the switch registry: {exc}")
    try:
        model = loaded.model
    except Exception as exc:
        model = None
        note(f"Could not read the loaded model object: {exc}")

    compile_module: Any | None = None
    if backend == "transformers":
        try:
            from . import torch_compile as compile_module

            compile_module.release_compiled_model(model)
        except ImportError as exc:
            note(f"Could not import the compiled-model release helper: {exc}")
        except Exception as exc:
            note(f"Could not release compiled model state: {exc}")
        try:
            notes.extend(_remove_model_runtime(model))
        except Exception as exc:
            note(f"Could not remove model runtime hooks: {exc}")
        try:
            from .quant.convrot import clear_device_caches

            clear_device_caches()
        except ImportError as exc:
            note(f"Could not import the ConvRot cache release helper: {exc}")
        except Exception as exc:
            note(f"Could not clear ConvRot device caches: {exc}")

    try:
        stop = getattr(model, "stop", None)
        if callable(stop):
            stop()
    except Exception as exc:
        note(f"Could not stop the model backend: {exc}")
    finally:
        stop = None

    try:
        loaded.model = None
    except Exception as exc:
        note(f"Could not clear the loaded model reference: {exc}")
    try:
        loaded.processor = None
    except Exception as exc:
        note(f"Could not clear the loaded processor reference: {exc}")

    model_ref: weakref.ReferenceType[Any] | None = None
    try:
        model_ref = weakref.ref(model)
    except (TypeError, AttributeError):
        pass
    except Exception as exc:
        note(f"Could not create a model liveness reference: {exc}")
    try:
        del model
    except UnboundLocalError:
        pass
    try:
        gc.collect()
    except Exception as exc:
        note(f"Could not collect Python objects after unload: {exc}")

    if backend == "transformers":
        if "torch._dynamo" in sys.modules:
            try:
                if compile_module is None:
                    from . import torch_compile as compile_module
                compile_module._reset_compiler_runtime()
            except ImportError as exc:
                note(f"Could not import the compiler runtime reset helper: {exc}")
            except Exception as exc:
                note(f"Could not reset the compiler runtime: {exc}")
        try:
            import torch

            try:
                host_empty_cache = getattr(
                    getattr(torch, "_C", None), "_host_emptyCache", None
                )
                if callable(host_empty_cache):
                    host_empty_cache()
            except Exception as exc:
                note(f"Could not empty the Torch host cache: {exc}")

            cuda_device = device_name.startswith("cuda")
            try:
                cuda_available = bool(torch.cuda.is_available())
            except Exception as exc:
                cuda_available = False
                note(f"Could not query CUDA availability: {exc}")
            if cuda_available and cuda_device:
                try:
                    selected_device = torch.device(device_name)
                except Exception as exc:
                    selected_device = None
                    note(f"Could not select {device_name} for CUDA cleanup: {exc}")
                if selected_device is not None:
                    try:
                        torch.cuda.set_per_process_memory_fraction(
                            1.0, selected_device
                        )
                    except Exception as exc:
                        note(f"Could not reset the CUDA allocator fraction: {exc}")
                    try:
                        torch.cuda.synchronize(selected_device)
                    except Exception as exc:
                        note(f"Could not synchronize CUDA during unload: {exc}")
                    # cuBLAS keeps a 32 MiB workspace per handle inside the caching
                    # allocator; drop it so nothing of the model's session survives.
                    try:
                        clear_workspaces = getattr(
                            getattr(torch, "_C", None), "_cuda_clearCublasWorkspaces", None
                        )
                        if callable(clear_workspaces):
                            clear_workspaces()
                    except Exception as exc:
                        note(f"Could not clear the cuBLAS workspaces: {exc}")
                    try:
                        torch.cuda.empty_cache()
                    except Exception as exc:
                        note(f"Could not empty the CUDA cache: {exc}")
                    try:
                        torch.cuda.ipc_collect()
                    except Exception as exc:
                        note(f"Could not collect CUDA IPC allocations: {exc}")
        except ImportError as exc:
            note(f"Could not import Torch for cache cleanup: {exc}")
        except Exception as exc:
            note(f"Could not complete Torch cache cleanup: {exc}")
        try:
            gc.collect()
        except Exception as exc:
            note(f"Could not collect Python objects after CUDA cleanup: {exc}")

    released = True
    if model_ref is not None:
        referent = model_ref()
        if referent is not None:
            released = False
            try:
                referrer_names = [
                    type(referrer).__name__
                    for referrer in gc.get_referrers(referent)[:8]
                ]
            except Exception as exc:
                referrer_names = [f"referrer inspection failed: {exc}"]
            detail = ", ".join(referrer_names) or "unknown referrer"
            message = (
                f"{variant_key} is still referenced after unload: {detail}"
            )
            note(message)
            try:
                get_log().warn(message, scope="models")
            except Exception:
                pass
        try:
            del referent
        except UnboundLocalError:
            pass

    after = before
    previous = before
    stable_samples = 0
    deadline = time.monotonic() + max(0.0, float(wait_s))
    while True:
        sample = vram_used("while waiting for release")
        if sample is None:
            break
        after = sample
        if abs(previous - sample) < 0.05:
            stable_samples += 1
        else:
            stable_samples = 0
        previous = sample
        if stable_samples >= 4:
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            note(
                f"VRAM settle wait timed out after {max(0.0, float(wait_s)):.1f}s"
            )
            break
        time.sleep(min(0.1, remaining))

    try:
        _trim_host_working_set()
    except Exception as exc:
        note(f"Could not trim the host working set: {exc}")
    try:
        host_after = _process_private_gb()
    except Exception as exc:
        host_after = 0.0
        note(f"Could not read host memory after unload: {exc}")

    seconds = time.perf_counter() - started
    freed = max(0.0, before - after)
    label = variant_key if variant_key is not None else "<unknown>"
    try:
        get_log().log(
            f"Unloaded {label} ({backend}) in {seconds:.1f}s: "
            f"VRAM {before:.2f} -> {after:.2f} GiB (freed {freed:.2f}), "
            f"host {host_before:.2f} -> {host_after:.2f} GB",
            scope="models",
        )
    except Exception as exc:
        note(f"Could not log the model release report: {exc}")
    return UnloadReport(
        seconds,
        before,
        after,
        freed,
        variant_key=variant_key,
        backend=backend,
        released=released,
        host_before_gb=host_before,
        host_after_gb=host_after,
        notes=tuple(notes),
    )


_SWITCH_REGISTRY: "weakref.WeakSet[LoadedModel]" = weakref.WeakSet()


class ModelCache:
    """Thread-safe cache holding at most one strong model reference."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._loaded: LoadedModel | None = None
        self._key: tuple[Any, ...] | None = None

    @property
    def loaded(self) -> LoadedModel | None:
        """Return the currently resident model, if any."""

        with self._lock:
            return self._loaded

    def load(self, variant_key: str, **kwargs: Any) -> LoadedModel:
        """Reuse an identical model or unload before switching variants/options."""

        offload = kwargs.get("offload") or OffloadPlan()
        # A llama-server is started with a fixed context, so a different requested
        # window needs a restart; Transformers apply the window per call instead.
        context_key = 0
        try:
            if get_variant(variant_key).scheme == "gguf":
                context_key = int(getattr(kwargs.get("budget_hint"), "context_tokens", 0) or 0)
        except KeyError:
            context_key = 0
        key = (
            variant_key,
            str(kwargs.get("device", "cuda:0")),
            kwargs.get("gpu_index"),
            str(kwargs.get("attention", "auto")),
            repr(offload),
            str(kwargs.get("dtype")),
            str(kwargs.get("hf_dir")),
            bool(kwargs.get("compile_model", False)),
            str(kwargs.get("compile_mode", DEFAULT_COMPILE_MODE)),
            repr(kwargs.get("last_token_logits", True)),
            context_key,
        )
        with self._lock:
            if self._loaded is not None and self._key == key and self._loaded.model is not None:
                if self._plan_covers(self._loaded, kwargs.get("budget_hint")):
                    return self._loaded
                get_log().log(
                    "Reloading the resident model: the new job needs more activation VRAM "
                    "than the current block-swap plan reserved.",
                    scope="models",
                )
            released_old = False
            if self._loaded is not None:
                old_key = str(getattr(self._loaded.variant, "key", "<unknown>"))
                get_log().log(
                    f"Switching model: unloading {old_key} before loading {variant_key}",
                    scope="models",
                )
                report = unload_model(self._loaded)
                if report is not None and not report.released:
                    get_log().warn(
                        f"The previous model {old_key} is still referenced after the switch",
                        scope="models",
                    )
                self._loaded = None
                self._key = None
                released_old = True
            try:
                self._loaded = load_model(variant_key, **kwargs)
            except Exception:
                self._loaded = None
                self._key = None
                if released_old:
                    gc.collect()
                    try:
                        import torch

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                raise
            self._key = key
            return self._loaded

    @staticmethod
    def _plan_covers(loaded: LoadedModel, hint: Any) -> bool:
        """Return whether the resident plan's activation reserve covers ``hint``."""

        report = getattr(loaded, "load_report", None)
        planned = int(getattr(report, "activation_estimate_bytes", 0) or 0)
        if planned <= 0:
            return True  # legacy, GGUF, or CPU loads carry no block-swap plan
        try:
            needed = int(
                estimate_activation_bytes(
                    loaded.spec.family,
                    getattr(loaded.model, "config", None),
                    hint,
                    observed_ratio=observed_activation_ratio(loaded.variant.key),
                )
            )
        except Exception:
            return True
        return needed <= planned

    def loaded_variant_key(self) -> str | None:
        """Return the resident variant key without exposing the model object."""

        with self._lock:
            if self._loaded is None:
                return None
            try:
                return str(self._loaded.variant.key)
            except Exception:
                return None

    def unload(
        self,
        *,
        unless_variant: str | None = None,
    ) -> UnloadReport | None:
        """Unload the cached model and clear the cache key."""

        with self._lock:
            if self._loaded is None:
                return None
            if (
                unless_variant is not None
                and self.loaded_variant_key() == str(unless_variant)
            ):
                return None
            report = unload_model(self._loaded)
            self._loaded = None
            self._key = None
            return report


MODEL_CACHE = ModelCache()


__all__ = [
    "LoadReport",
    "LoadedModel",
    "MODEL_CACHE",
    "ModelCache",
    "UnloadReport",
    "load_model",
    "unload_model",
]
