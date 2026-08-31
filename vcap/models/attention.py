"""Attention backend selection without eager Torch or CUDA-extension imports."""

from __future__ import annotations

from contextlib import nullcontext
import importlib
import importlib.util
from typing import Any, ContextManager

from vcap.core.logs import get_log


ATTENTION_CHOICES = ["auto", "flash_attention_2", "sdpa", "sage", "xformers", "eager"]

_PACKAGE_NAMES = {
    "flash_attention_2": "flash-attn",
    "sage": "sageattention",
    "xformers": "xformers",
}
_BACKEND_FAILURES: dict[str, str] = {}
_REGISTERED: set[str] = set()
_WARNED_RUNTIME_FALLBACKS: set[str] = set()


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def probe_available() -> dict[str, bool]:
    """Probe package specs without importing Torch or compiled extensions."""

    torch_present = _has_module("torch")
    return {
        "auto": torch_present,
        "flash_attention_2": _has_module("flash_attn") and "flash_attention_2" not in _BACKEND_FAILURES,
        "sdpa": torch_present,
        "sage": _has_module("sageattention") and "sage" not in _BACKEND_FAILURES,
        "xformers": _has_module("xformers") and "xformers" not in _BACKEND_FAILURES,
        "eager": torch_present,
    }


def describe_available() -> dict[str, str]:
    """Return concise backend availability details for the UI."""

    available = probe_available()
    descriptions: dict[str, str] = {}
    for backend in ATTENTION_CHOICES:
        if available.get(backend, False):
            descriptions[backend] = "available"
        elif backend in _BACKEND_FAILURES:
            descriptions[backend] = f"falls back to SDPA: {_BACKEND_FAILURES[backend]}"
        elif backend in _PACKAGE_NAMES:
            descriptions[backend] = f"unavailable: {_PACKAGE_NAMES[backend]} is not installed"
        else:
            descriptions[backend] = "unavailable: PyTorch is not installed"
    return descriptions


def _dtype_supports_flash(dtype: object | None) -> bool:
    if dtype is None:
        return True
    text = str(dtype).lower()
    return "float16" in text or "bfloat16" in text or text in {"fp16", "bf16"}


def is_flash_attention_failure(exc: BaseException) -> bool:
    """Return whether a load/forward error is suitable for an SDPA retry."""

    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        return True
    message = str(exc).casefold()
    if "out of memory" in message:
        return False
    markers = (
        "flash_attn",
        "flash attention",
        "flashattention",
        "no available kernel",
        "unsupported gpu",
        "unsupported architecture",
        "compute capability",
        "only supports fp16",
        "only supports bf16",
        "must be fp16",
        "must be bf16",
    )
    return any(marker in message for marker in markers)


def _sdpa_forward(
    module: Any,
    query: Any,
    key: Any,
    value: Any,
    attention_mask: Any,
    *,
    scaling: float | None,
    dropout: float,
    **kwargs: Any,
) -> tuple[Any, None]:
    from transformers import AttentionInterface

    implementation = AttentionInterface().get_interface("sdpa", None)
    if implementation is None:
        raise RuntimeError("Transformers does not expose its SDPA attention implementation")
    return implementation(
        module,
        query,
        key,
        value,
        attention_mask,
        scaling=scaling,
        dropout=dropout,
        **kwargs,
    )


def _mask_is_all_ones(attention_mask: Any) -> bool:
    if attention_mask is None:
        return True
    try:
        import torch

        return bool(torch.all(attention_mask == 1).item())
    except Exception:
        return False


def _causal_for_fast_path(module: Any, query: Any, kwargs: dict[str, Any]) -> bool:
    requested = kwargs.get("is_causal")
    causal = bool(getattr(module, "is_causal", True) if requested is None else requested)
    return bool(int(query.shape[2]) > 1 and causal)


def _fast_path_is_safe(
    backend: str,
    module: Any,
    query: Any,
    key: Any,
    value: Any,
    attention_mask: Any,
    dropout: float,
    kwargs: dict[str, Any],
) -> bool:
    if backend in _BACKEND_FAILURES:
        return False
    try:
        import torch

        if any(tensor.ndim != 4 for tensor in (query, key, value)):
            return False
        if not query.is_cuda or query.device != key.device or query.device != value.device:
            return False
        if query.dtype not in {torch.float16, torch.bfloat16}:
            return False
        if key.dtype != query.dtype or value.dtype != query.dtype:
            return False
        head_dim = int(query.shape[-1])
        if head_dim <= 0 or head_dim > 256 or int(key.shape[-1]) != head_dim:
            return False
        query_heads = int(query.shape[1])
        key_heads = int(key.shape[1])
        if key_heads <= 0 or query_heads % key_heads:
            return False
        if float(dropout) != 0.0 or kwargs.get("position_bias") is not None:
            return False
        if any(
            kwargs.get(name) is not None
            for name in ("sliding_window", "softcap", "cu_seq_lens_q", "cu_seq_lens_k")
        ):
            return False
        if kwargs.get("output_attentions", False) or not _mask_is_all_ones(attention_mask):
            return False
        if _causal_for_fast_path(module, query, kwargs) and int(query.shape[2]) != int(key.shape[2]):
            return False
        return True
    except (AttributeError, TypeError, ValueError):
        return False


def _record_runtime_fallback(backend: str, exc: BaseException) -> None:
    message = str(exc).strip().replace("\n", " ") or type(exc).__name__
    reason = message[:240]
    _BACKEND_FAILURES.setdefault(backend, reason)
    if backend not in _WARNED_RUNTIME_FALLBACKS:
        _WARNED_RUNTIME_FALLBACKS.add(backend)
        get_log().warn(
            f"{backend} attention failed ({reason}); this model will fall back to SDPA.",
            scope="attention",
        )


def sage_attention_forward(
    module: Any,
    query: Any,
    key: Any,
    value: Any,
    attention_mask: Any,
    scaling: float | None = None,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[Any, None]:
    """Transformers AttentionInterface adapter for SageAttention."""

    if not _fast_path_is_safe("sage", module, query, key, value, attention_mask, dropout, kwargs):
        return _sdpa_forward(
            module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kwargs
        )
    try:
        sageattention = importlib.import_module("sageattention")
        output = sageattention.sageattn(
            query,
            key,
            value,
            tensor_layout="HND",
            is_causal=_causal_for_fast_path(module, query, kwargs),
            sm_scale=scaling,
        )
        return output.transpose(1, 2).contiguous(), None
    except Exception as exc:
        if "out of memory" in str(exc).casefold():
            raise
        _record_runtime_fallback("sage", exc)
        return _sdpa_forward(
            module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kwargs
        )


def xformers_attention_forward(
    module: Any,
    query: Any,
    key: Any,
    value: Any,
    attention_mask: Any,
    scaling: float | None = None,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[Any, None]:
    """Transformers AttentionInterface adapter for xFormers memory-efficient attention."""

    if not _fast_path_is_safe("xformers", module, query, key, value, attention_mask, dropout, kwargs):
        return _sdpa_forward(
            module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kwargs
        )
    try:
        xops = importlib.import_module("xformers.ops")
        query_nhd = query.transpose(1, 2)
        key_nhd = key.transpose(1, 2)
        value_nhd = value.transpose(1, 2)
        if int(query.shape[1]) != int(key.shape[1]):
            repeats = int(query.shape[1]) // int(key.shape[1])
            key_nhd = key_nhd.repeat_interleave(repeats, dim=2)
            value_nhd = value_nhd.repeat_interleave(repeats, dim=2)
        bias = xops.LowerTriangularMask() if _causal_for_fast_path(module, query, kwargs) else None
        output = xops.memory_efficient_attention(
            query_nhd,
            key_nhd,
            value_nhd,
            attn_bias=bias,
            p=float(dropout),
            scale=scaling,
        )
        return output.contiguous(), None
    except Exception as exc:
        if "out of memory" in str(exc).casefold():
            raise
        _record_runtime_fallback("xformers", exc)
        return _sdpa_forward(
            module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kwargs
        )


def _register_backend(backend: str) -> tuple[bool, str | None]:
    if backend in _REGISTERED:
        return True, None
    try:
        if backend == "sage":
            extension = importlib.import_module("sageattention")
            if not callable(getattr(extension, "sageattn", None)):
                raise RuntimeError("sageattention.sageattn is missing")
            implementation = sage_attention_forward
        elif backend == "xformers":
            extension = importlib.import_module("xformers.ops")
            if not callable(getattr(extension, "memory_efficient_attention", None)):
                raise RuntimeError("xformers.ops.memory_efficient_attention is missing")
            implementation = xformers_attention_forward
        else:
            raise ValueError(f"Cannot register unsupported attention backend {backend!r}")
        from transformers import AttentionInterface

        AttentionInterface.register(backend, implementation)
        _REGISTERED.add(backend)
        return True, None
    except Exception as exc:
        reason = str(exc).strip().replace("\n", " ") or type(exc).__name__
        _BACKEND_FAILURES[backend] = reason[:240]
        return False, _BACKEND_FAILURES[backend]


def resolve(
    backend: str,
    model_family: str,
    dtype: object | None = None,
) -> tuple[str, ContextManager[Any]]:
    """Resolve a UI choice to a registered Transformers implementation."""

    del model_family  # Reserved for model-specific exclusions and UI policy.
    requested = str(backend or "auto").strip().lower().replace("-", "_")
    if requested not in ATTENTION_CHOICES:
        raise ValueError(f"Unknown attention backend {backend!r}; choose from {ATTENTION_CHOICES}")
    available = probe_available()

    if requested == "auto":
        requested = "flash_attention_2" if available["flash_attention_2"] and _dtype_supports_flash(dtype) else "sdpa"

    if requested == "flash_attention_2":
        reason = None
        if not available["flash_attention_2"]:
            reason = _BACKEND_FAILURES.get("flash_attention_2", "flash-attn is not installed")
        elif not _dtype_supports_flash(dtype):
            reason = f"dtype {dtype} is not fp16/bf16"
        if reason:
            get_log().warn(f"FlashAttention 2 unavailable ({reason}); falling back to SDPA.", scope="attention")
            return "sdpa", nullcontext()
        return "flash_attention_2", nullcontext()

    if requested in {"sage", "xformers"}:
        if not available[requested]:
            reason = _BACKEND_FAILURES.get(requested, f"{_PACKAGE_NAMES[requested]} is not installed")
            get_log().warn(f"{requested} unavailable ({reason}); falling back to SDPA.", scope="attention")
            return "sdpa", nullcontext()
        registered, reason = _register_backend(requested)
        if not registered:
            get_log().warn(f"{requested} could not be registered ({reason}); falling back to SDPA.", scope="attention")
            return "sdpa", nullcontext()
        return requested, nullcontext()

    return requested, nullcontext()


__all__ = [
    "ATTENTION_CHOICES",
    "describe_available",
    "is_flash_attention_failure",
    "probe_available",
    "resolve",
    "sage_attention_forward",
    "xformers_attention_forward",
]
