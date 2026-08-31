"""Local-only thinker model loading, streaming checkpoints, and one-model cache."""

from __future__ import annotations

from dataclasses import dataclass, field
import gc
from itertools import chain
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable
import weakref

from vcap import TEMP_DIR
from vcap.core.gpu import resource_snapshot
from vcap.core.logs import get_log

from .attention import is_flash_attention_failure, resolve as resolve_attention
from .offload import OffloadPlan, build_device_map
from .registry import MODEL_SPECS, ModelSpec, VariantSpec, get_variant, resolve_model_dir, variant_is_ready, variant_to_family
from .torch_compile import DEFAULT_COMPILE_MODE


ProgressCallback = Callable[..., None]


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


@dataclass(frozen=True)
class UnloadReport:
    """Observed device-memory change after releasing a model."""

    seconds: float
    vram_before_gb: float
    vram_after_gb: float
    freed_vram_gb: float


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
    )
    backend.start(progress_cb)
    report = LoadReport(
        seconds=time.perf_counter() - started,
        peak_vram_gb=float(backend.load_peak_vram_gb),
        checkpoint_bytes=_gguf_checkpoint_bytes(folder, variant),
        attention="llamacpp",
        device_map={"": str(device)},
        compile_mode="llama.cpp",
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
        offload or OffloadPlan(),
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
        )
    except (AttributeError, TypeError) as exc:
        raise RuntimeError(
            "The installed ConvRot loader does not match the required T4 API "
            "apply_quantized_checkpoint(model, path, device, dtype, progress_cb, strip_prefix)."
        ) from exc
    return model, report


def _plain_sharded_load(
    folder: Path,
    model_class: Any,
    *,
    dtype: Any,
    attention: str,
    placement: Any,
) -> Any:
    kwargs: dict[str, Any] = {
        "dtype": dtype,
        "attn_implementation": attention,
        "local_files_only": True,
        "trust_remote_code": False,
        "low_cpu_mem_usage": True,
    }
    kwargs.update(placement.from_pretrained_kwargs())
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
    snapshot = resource_snapshot(physical_gpu_index)
    placement = build_device_map(
        family,
        plan,
        float(snapshot.get("vram_free_gb", 0.0) or 0.0),
        device_index=device_index,
        physical_gpu_index=physical_gpu_index,
    )
    torch_device = torch.device(device)
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch_device)
    _emit(progress_cb, f"Loading {spec.label} / {variant.label} from {folder}", 0.0)

    checkpoint = folder / "model.safetensors"
    single_file = checkpoint.is_file()
    use_cpu_staging = single_file and (
        plan.gpu_layers != "all" or plan.offload_experts or plan.max_memory is not None
    )

    def load_weights(selected_attention: str) -> tuple[Any, dict[str, Any] | str | None, int, int]:
        config = _thinker_config(folder, spec.architecture, selected_attention, dtype)
        if single_file:
            target = "cpu" if use_cpu_staging else str(device)
            loaded_model, stream_report = _stream_single_checkpoint(
                folder,
                model_class,
                config,
                target_device=target,
                dtype=dtype,
                progress_cb=progress_cb,
            )
            selected_map: dict[str, Any] | str | None = {"": target}
            if use_cpu_staging:
                selected_map = _dispatch_after_cpu_load(
                    loaded_model,
                    family,
                    plan,
                    placement,
                    dtype,
                    device_index=device_index,
                )
            return (
                loaded_model,
                selected_map,
                int(getattr(stream_report, "quantized_layers", 0)),
                int(getattr(stream_report, "bf16_layers", 0)),
            )
        if variant.scheme != "bf16":
            raise RuntimeError("Quantized variants must be self-contained single-file checkpoints")
        loaded_model = _plain_sharded_load(
            folder,
            model_class,
            dtype=dtype,
            attention=selected_attention,
            placement=placement,
        )
        return loaded_model, getattr(loaded_model, "hf_device_map", placement.device_map), 0, 0

    try:
        model, device_map, quantized_layers, bf16_layers = load_weights(resolved_attention)
    except Exception as exc:
        if resolved_attention != "flash_attention_2" or not is_flash_attention_failure(exc):
            raise
        warning = f"FlashAttention 2 load failed ({exc}); retrying with SDPA."
        exc.__traceback__ = None
        get_log().warn(warning, scope="models")
        _emit(progress_cb, warning)
        gc.collect()
        if torch.cuda.is_available() and device.startswith("cuda"):
            torch.cuda.empty_cache()
        resolved_attention = "sdpa"
        model, device_map, quantized_layers, bf16_layers = load_weights(resolved_attention)

    model.eval()
    if plan.pin_cpu and isinstance(device_map, dict) and any(str(value) == "cpu" for value in device_map.values()):
        pinned = _pin_cpu_tensors(model)
        _emit(progress_cb, f"Pinned {pinned / 1024**3:.2f} GiB of CPU-resident weights")
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
        if torch.cuda.is_available() and device.startswith("cuda")
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


def unload_model(loaded: LoadedModel | None) -> UnloadReport:
    """Release a loaded model, collect Python objects, and empty CUDA caches."""

    started = time.perf_counter()
    physical_gpu_index = int(getattr(loaded, "gpu_index", 0) or 0) if loaded is not None else 0
    device_name = str(getattr(loaded, "device", "cuda:0") or "cuda:0") if loaded is not None else "cuda:0"
    before = float(
        resource_snapshot(physical_gpu_index).get("vram_used_gb", 0.0) or 0.0
    )
    llama_backend = bool(loaded is not None and loaded.variant.backend == "llamacpp")
    if loaded is not None:
        _SWITCH_REGISTRY.discard(loaded)
        model = loaded.model
        stop = getattr(model, "stop", None)
        if callable(stop):
            stop()
        loaded.model = None
        loaded.processor = None
        try:
            del model
        except UnboundLocalError:
            pass
    gc.collect()
    if not llama_backend:
        try:
            import torch

            if torch.cuda.is_available() and device_name.startswith("cuda"):
                selected_device = torch.device(device_name)
                torch.cuda.synchronize(selected_device)
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except (RuntimeError, AttributeError):
                    pass
        except ImportError:
            pass
    gc.collect()
    after = float(
        resource_snapshot(physical_gpu_index).get("vram_used_gb", 0.0) or 0.0
    )
    return UnloadReport(time.perf_counter() - started, before, after, max(0.0, before - after))


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
        )
        with self._lock:
            if self._loaded is not None and self._key == key and self._loaded.model is not None:
                return self._loaded
            if self._loaded is not None:
                unload_model(self._loaded)
                self._loaded = None
                self._key = None
            self._loaded = load_model(variant_key, **kwargs)
            self._key = key
            return self._loaded

    def unload(self) -> UnloadReport | None:
        """Unload the cached model and clear the cache key."""

        with self._lock:
            if self._loaded is None:
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
