"""Measure dynamic/static KV cache and decoder compilation on application models.

This is an experiment harness, not an application code path.  It deliberately
uses the production loader/profile and the captioner's private preparation
helpers, while keeping all compiler/tuning caches under temp/codex_v14.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
import gc
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from typing import Any, Iterator, Mapping


ROOT = Path(__file__).resolve().parents[2]
WORK_DIR = ROOT / "temp" / "codex_v14"
RESULTS_DIR = ROOT / "tools" / "bench" / "results"
PRIMARY_VIDEO = Path(r"F:\SECourses_Video_Captioner_Pro_TEMP\test_media\lightning_storm_20s.mp4")
PRIMARY_AUDIO = Path(r"F:\SECourses_Video_Captioner_Pro_TEMP\test_media\demon_singer_audio_18_sec.mp3")
ALTERNATE_VIDEO = Path(r"F:\SECourses_Video_Captioner_Pro_TEMP\test_media\voyager_1_launch.mp4")
CONFIG_ORDER = (
    "eager_dynamic",
    "eager_static",
    "compile_default_dynamic",
    "compile_reduce_overhead_static",
    "manual_cudagraph_static",
)


class ForeignGpuContention(RuntimeError):
    """Raised when another process executes kernels on physical GPU 0."""


class _GpuContentionMonitor:
    """Sample per-process utilization out of process to avoid stealing the GIL."""

    def __init__(self) -> None:
        self.events: list[dict[str, int]] = []
        self.error: str | None = None
        self._process: subprocess.Popen[str] | None = None

    def start(self) -> None:
        try:
            self._process = subprocess.Popen(
                ["nvidia-smi", "pmon", "-i", "0", "-d", "1"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=dict(os.environ),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def stop(self) -> None:
        if self._process is None:
            return
        try:
            if self._process.poll() is None:
                self._process.terminate()
            try:
                stdout, stderr = self._process.communicate(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                stdout, stderr = self._process.communicate(timeout=5.0)
            for line_number, line in enumerate(stdout.splitlines(), start=1):
                columns = line.split()
                if len(columns) < 5 or columns[0].startswith("#"):
                    continue
                try:
                    gpu, pid = int(columns[0]), int(columns[1])
                except ValueError:
                    continue
                if gpu != 0 or pid == os.getpid():
                    continue
                sm = int(columns[3]) if columns[3].isdigit() else 0
                memory = int(columns[4]) if columns[4].isdigit() else 0
                if sm or memory:
                    self.events.append(
                        {"pid": pid, "sm_util": sm, "mem_util": memory, "sample": line_number}
                    )
            if self._process.returncode not in (0, 1) and stderr.strip():
                self.error = f"nvidia-smi exit {self._process.returncode}: {stderr.strip()}"
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"


def _configure_process_environment() -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible != "0":
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES must be exactly '0', got {visible!r}")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["VCAP_DEV_FORCE_GPU"] = "0"
    os.environ["VCAP_TEMP_DIR"] = str(WORK_DIR / "runtime")
    os.environ["VCAP_LOGS_DIR"] = str(WORK_DIR / "logs")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(WORK_DIR / "torchinductor")
    os.environ["TRITON_CACHE_DIR"] = str(WORK_DIR / "triton")
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "8"
    os.environ["TORCHINDUCTOR_WORKER_START"] = "spawn"
    for directory in (
        WORK_DIR,
        WORK_DIR / "runtime",
        WORK_DIR / "logs",
        WORK_DIR / "torchinductor",
        WORK_DIR / "triton",
        RESULTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


_configure_process_environment()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--profile-tier", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cache-tag", default="", help=argparse.SUPPRESS)
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=CONFIG_ORDER,
        default=list(CONFIG_ORDER),
        help="Configurations to run in the given order",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def progress(message: object, *args: object, **kwargs: object) -> None:
    del args, kwargs
    if isinstance(message, Mapping):
        message = message.get("message", message)
    text = str(message)
    if text and not text.startswith(("Generating:", "Loading tensor")):
        print(text, flush=True)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(partial, path)


def _counter_snapshot(torch: Any) -> dict[str, dict[str, int]]:
    try:
        counters = torch._dynamo.utils.counters
    except (AttributeError, ImportError):
        return {}
    result: dict[str, dict[str, int]] = {}
    for category, values in counters.items():
        if not isinstance(values, Mapping):
            continue
        converted = {str(key): int(value) for key, value in values.items() if int(value)}
        if converted:
            result[str(category)] = converted
    return result


def _counter_delta(
    before: dict[str, dict[str, int]], after: dict[str, dict[str, int]]
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for category in sorted(set(before) | set(after)):
        values: dict[str, int] = {}
        for key in sorted(set(before.get(category, {})) | set(after.get(category, {}))):
            delta = after.get(category, {}).get(key, 0) - before.get(category, {}).get(key, 0)
            if delta:
                values[key] = delta
        if values:
            result[category] = values
    return result


def _first_difference(reference: list[int], candidate: list[int]) -> int | None:
    for index, (left, right) in enumerate(zip(reference, candidate)):
        if left != right:
            return index
    if len(reference) != len(candidate):
        return min(len(reference), len(candidate))
    return None


def _move_value(value: Any, device: Any, dtype: Any, torch: Any) -> Any:
    if isinstance(value, torch.Tensor):
        moved = value.to(device)
        return moved.to(dtype=dtype) if moved.is_floating_point() else moved
    if isinstance(value, Mapping):
        return {key: _move_value(item, device, dtype, torch) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_value(item, device, dtype, torch) for item in value)
    if isinstance(value, list):
        return [_move_value(item, device, dtype, torch) for item in value]
    return value


def _input_description(inputs: Mapping[str, Any], torch: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            result[key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "bytes": int(value.numel() * value.element_size()),
            }
        else:
            result[key] = {"type": type(value).__name__}
    return result


def _prepare_case(
    captioner: Any,
    media: Any,
    preprocessing: Any,
    generation: Any,
    label: str,
    torch: Any,
) -> dict[str, Any]:
    from vcap.models.base import Callbacks
    from vcap.models.omni_common import _parts

    print(f"preparing {label}: {media.path}", flush=True)
    parts = _parts(media)
    captioner._validate_capabilities(parts)
    prepared = captioner._prepare_media(parts, preprocessing)
    system, user, _preset = captioner._prompt(None, prepared, Callbacks(progress=progress))
    conversation, template_kwargs = captioner._conversation(prepared, system, user, generation)
    processor = captioner.loaded.processor
    rendered = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
        **template_kwargs,
    )
    inputs = dict(captioner._processor_inputs(prepared, rendered, preprocessing))
    input_length = int(inputs["input_ids"].shape[-1])
    prompt_tokens = (
        int(inputs["attention_mask"].sum().item())
        if "attention_mask" in inputs
        else input_length
    )
    return {
        "label": label,
        "source": str(media.path),
        "trim_start_s": media.start,
        "trim_end_s": media.end,
        "kind": media.kind,
        "prepared": prepared,
        "inputs": inputs,
        "input_length": input_length,
        "prompt_tokens": prompt_tokens,
        "use_audio_in_video": bool(prepared.use_audio_in_video),
        "warnings": list(prepared.warnings),
        "input_tensors": _input_description(inputs, torch),
    }


def _stop_ids(model: Any, processor: Any) -> tuple[list[int], int]:
    from vcap.models.loader import resolve_stop_token_ids

    generation_config = getattr(model, "generation_config", None)
    eos_ids = getattr(generation_config, "eos_token_id", None)
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    pad_id = getattr(generation_config, "pad_token_id", None)
    fallback_eos, fallback_pad = resolve_stop_token_ids(processor)
    if not eos_ids:
        eos_ids = fallback_eos
    if not isinstance(pad_id, int) or pad_id < 0:
        pad_id = fallback_pad
    return [int(value) for value in eos_ids], int(pad_id)


def _sampled_logits_processor(torch: Any):
    from transformers import LogitsProcessor

    class SampledFiniteLogits(LogitsProcessor):
        def __init__(self) -> None:
            self.calls = 0
            self.samples: list[tuple[int, Any]] = []
            self.last: tuple[int, Any] | None = None

        def __call__(self, input_ids: Any, scores: Any) -> Any:
            del input_ids
            if self.calls == 0 or self.calls % 32 == 0:
                self.samples.append((self.calls, scores.detach()))
            self.last = (self.calls, scores.detach())
            self.calls += 1
            return scores

        def finish(self) -> tuple[bool, int, list[int]]:
            samples = list(self.samples)
            if self.last is not None and all(index != self.last[0] for index, _ in samples):
                samples.append(self.last)
            bad_positions: list[int] = []
            for index, scores in samples:
                if not bool(torch.isfinite(scores).all().item()):
                    bad_positions.append(int(index))
            return not bad_positions, len(samples), bad_positions

    return SampledFiniteLogits()


def _timing_stopping_criteria(torch: Any, device: Any):
    from transformers import StoppingCriteria

    class TimingCriteria(StoppingCriteria):
        def __init__(self) -> None:
            self.calls = 0
            self.first_event = torch.cuda.Event(enable_timing=True)
            self.stop = torch.zeros(1, dtype=torch.bool, device=device)

        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> Any:
            del input_ids, scores, kwargs
            if self.calls == 0:
                self.first_event.record()
            self.calls += 1
            return self.stop

    return TimingCriteria()


@contextmanager
def _timed_prefill(model: Any, torch: Any) -> Iterator[tuple[Any, Any]]:
    original = model._prefill
    started = torch.cuda.Event(enable_timing=True)
    ended = torch.cuda.Event(enable_timing=True)

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        started.record()
        output = original(*args, **kwargs)
        ended.record()
        return output

    model._prefill = wrapper
    try:
        yield started, ended
    finally:
        model._prefill = original


def _compile_environment() -> dict[str, Any]:
    """Activate MSVC without invoking the app probe that writes under vcap/."""

    vcvars = Path(
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
    )
    details: dict[str, Any] = {
        "vcvars_bat": str(vcvars),
        "vcvars_exists": vcvars.is_file(),
        "activated": False,
        "error": None,
    }
    if vcvars.is_file():
        try:
            completed = subprocess.run(
                ["cmd.exe", "/d", "/s", "/c", "call", str(vcvars), "&&", "set"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
                check=False,
                env=dict(os.environ),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if completed.returncode != 0:
                tail = "\n".join((completed.stdout + "\n" + completed.stderr).splitlines()[-30:])
                raise RuntimeError(f"{vcvars.name} exited with code {completed.returncode}: {tail}")
            captured: dict[str, str] = {}
            for line in completed.stdout.splitlines():
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key and not key.startswith("=") and "\x00" not in key:
                    captured[key] = value
            updates = {key: value for key, value in captured.items() if os.environ.get(key) != value}
            # Preserve the experiment's hard GPU/cache choices over vcvars output.
            protected = {
                key: os.environ[key]
                for key in (
                    "CUDA_VISIBLE_DEVICES",
                    "CUDA_DEVICE_ORDER",
                    "VCAP_DEV_FORCE_GPU",
                    "VCAP_TEMP_DIR",
                    "VCAP_LOGS_DIR",
                    "TORCHINDUCTOR_CACHE_DIR",
                    "TRITON_CACHE_DIR",
                    "TORCHINDUCTOR_COMPILE_THREADS",
                    "TORCHINDUCTOR_WORKER_START",
                )
            }
            os.environ.update(updates)
            os.environ.update(protected)
            details["activated"] = True
            details["cl_path"] = next(
                (
                    str(candidate)
                    for directory in os.environ.get("PATH", "").split(os.pathsep)
                    if directory
                    for candidate in (Path(directory) / "cl.exe",)
                    if candidate.is_file()
                ),
                None,
            )
        except Exception as exc:
            details["error"] = f"{type(exc).__name__}: {exc}"
    return details


def _compile_plan(mode: str) -> Any:
    from vcap.models.torch_compile import CompilePlan

    return CompilePlan(
        mode="full",
        env_updates={
            "TORCHINDUCTOR_CACHE_DIR": os.environ["TORCHINDUCTOR_CACHE_DIR"],
            "TORCHINDUCTOR_COMPILE_THREADS": os.environ["TORCHINDUCTOR_COMPILE_THREADS"],
            "TORCHINDUCTOR_WORKER_START": os.environ["TORCHINDUCTOR_WORKER_START"],
        },
        torch_compile_kwargs={
            "backend": "inductor",
            "mode": mode,
            "dynamic": False if mode == "reduce-overhead" else None,
        },
        warnings=[],
        fallback_modes=("eager",),
        requested_mode=mode,
    )


def _sequence_length(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> int | None:
    for name, axis in (("input_ids", -1), ("inputs_embeds", -2)):
        value = kwargs.get(name)
        if hasattr(value, "shape") and len(value.shape):
            return int(value.shape[axis])
    for value in args:
        if hasattr(value, "shape") and len(value.shape) >= 2:
            return int(value.shape[-1])
    return None


def _install_compile(model: Any, family: str, config_name: str) -> dict[str, Any]:
    from vcap.models.torch_compile import apply_compile

    target = getattr(model, "model", model)
    mode = "default" if config_name == "compile_default_dynamic" else "reduce-overhead"
    plan = _compile_plan(mode)
    setup_started = time.perf_counter()
    apply_compile(model, plan, progress_cb=progress, family=family)
    setup_s = time.perf_counter() - setup_started
    if not bool(getattr(target, "_vcap_compiled", False)):
        raise RuntimeError(f"application compile helper did not wrap {type(target).__name__}")
    selective_decode = config_name == "compile_reduce_overhead_static"
    if selective_decode:
        compiled_forward = target.forward
        eager_forward = target._vcap_original_forward

        def decode_only_forward(*args: Any, **kwargs: Any) -> Any:
            length = _sequence_length(args, kwargs)
            if length == 1 and kwargs.get("past_key_values") is not None:
                return compiled_forward(*args, **kwargs)
            return eager_forward(*args, **kwargs)

        # release_compiled_model still owns the authoritative eager original.
        target.forward = decode_only_forward
    disabled_modules = sum(
        1
        for module in target.modules()
        if callable(getattr(module, "_vcap_original_disabled_forward", None))
    )
    return {
        "setup_s": setup_s,
        "target_class": type(target).__name__,
        "mode": mode,
        "dynamic": plan.torch_compile_kwargs["dynamic"],
        "backend": "inductor",
        "selective_decode_only": selective_decode,
        "convrot_modules_disabled_from_dynamo": disabled_modules,
    }


def _release_compile(model: Any, torch: Any) -> int:
    from vcap.models.torch_compile import release_compiled_model

    restored = int(release_compiled_model(model))
    for name in ("_compiled_call", "_last_compile_config", "_previous_max_cache_length"):
        if name in getattr(model, "__dict__", {}):
            delattr(model, name)
    gc.collect()
    torch.cuda.empty_cache()
    return restored


def _generation_kwargs(
    config_name: str,
    case: Mapping[str, Any],
    max_new_tokens: int,
    eos_ids: list[int],
    pad_id: int,
    timing: Any,
    finite: Any,
) -> dict[str, Any]:
    from transformers import LogitsProcessorList, StoppingCriteriaList

    kwargs: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": False,
        "repetition_penalty": 1.0,
        "use_cache": True,
        "eos_token_id": list(eos_ids),
        "pad_token_id": int(pad_id),
        "stopping_criteria": StoppingCriteriaList([timing]),
        "logits_processor": LogitsProcessorList([finite]),
        # Explicitly prevent Transformers' automatic static-cache compilation;
        # config 4 compiles only the text decoder through the app helper.
        "disable_compile": True,
    }
    if config_name in {"eager_static", "compile_reduce_overhead_static", "manual_cudagraph_static"}:
        kwargs["cache_implementation"] = "static"
        kwargs["max_cache_len"] = int(case["input_length"]) + int(max_new_tokens)
    return kwargs


def _run_generation(
    loaded: Any,
    case: Mapping[str, Any],
    config_name: str,
    run_number: int,
    max_new_tokens: int,
    seed: int,
    reference_ids: list[int] | None,
    torch: Any,
) -> dict[str, Any]:
    from vcap.models.attention import resolve as resolve_attention

    model = loaded.model
    device = torch.device(loaded.device)
    gpu_inputs = _move_value(case["inputs"], device, loaded.dtype, torch)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.synchronize(device)
    gc.collect()
    torch.cuda.reset_peak_memory_stats(device)
    allocated_start = int(torch.cuda.memory_allocated(device))
    reserved_start = int(torch.cuda.memory_reserved(device))
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    timing = _timing_stopping_criteria(torch, device)
    finite = _sampled_logits_processor(torch)
    eos_ids, pad_id = _stop_ids(model, loaded.processor)
    generation = _generation_kwargs(
        config_name,
        case,
        max_new_tokens,
        eos_ids,
        pad_id,
        timing,
        finite,
    )
    _, runtime_context = resolve_attention(loaded.attention, loaded.spec.family, loaded.dtype)
    wall_started = time.perf_counter()
    start_event.record()
    contention = _GpuContentionMonitor()
    contention.start()
    try:
        with _timed_prefill(model, torch) as (prefill_start, prefill_end):
            with torch.inference_mode(), runtime_context:
                output = model.generate(
                    **gpu_inputs,
                    use_audio_in_video=bool(case["use_audio_in_video"]),
                    **generation,
                )
    finally:
        contention.stop()
    end_event.record()
    end_event.synchronize()
    wall_s = time.perf_counter() - wall_started
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    sequences = getattr(output, "sequences", output)
    new_ids_tensor = sequences[:, int(case["input_length"]) :]
    token_ids = [int(value) for value in new_ids_tensor[0].tolist()]
    new_tokens = len(token_ids)
    first_token_ms = (
        float(start_event.elapsed_time(timing.first_event)) if timing.calls else float("nan")
    )
    prefill_s = float(prefill_start.elapsed_time(prefill_end)) / 1000.0
    decode_s = (
        float(timing.first_event.elapsed_time(end_event)) / 1000.0
        if timing.calls
        else 0.0
    )
    total_gpu_s = float(start_event.elapsed_time(end_event)) / 1000.0
    decode_tok_s = max(0, new_tokens - 1) / max(decode_s, 1e-9)
    finite_ok, sampled_logits, nonfinite_positions = finite.finish()
    terminal = token_ids[-1] if token_ids else None
    finish_reason = "eos" if terminal in set(eos_ids) else "length"
    reference_diff = (
        _first_difference(reference_ids, token_ids) if reference_ids is not None else None
    )
    identical = reference_ids is None or reference_diff is None
    decoded = loaded.processor.batch_decode(
        new_ids_tensor,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    del output, sequences, new_ids_tensor, gpu_inputs
    return {
        "run": int(run_number),
        "case": str(case["label"]),
        "source": str(case["source"]),
        "seed": int(seed),
        "prompt_tokens": int(case["prompt_tokens"]),
        "input_length": int(case["input_length"]),
        "static_max_cache_len": generation.get("max_cache_len"),
        "generated_tokens": new_tokens,
        "finish_reason": finish_reason,
        "first_token_ms": first_token_ms,
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "decode_tok_s": decode_tok_s,
        "gpu_total_s": total_gpu_s,
        "wall_s": wall_s,
        "peak_allocated_mib": peak_allocated / 1024**2,
        "peak_reserved_mib": peak_reserved / 1024**2,
        "allocated_at_start_mib": allocated_start / 1024**2,
        "reserved_at_start_mib": reserved_start / 1024**2,
        "token_ids": token_ids,
        "identical_to_eager_dynamic": identical,
        "first_differing_index": reference_diff,
        "finite_logits_sampled": finite_ok,
        "sampled_logits_steps": sampled_logits,
        "nonfinite_sample_positions": nonfinite_positions,
        "caption_preview": " ".join(decoded.split())[:240],
        "gpu_contention_monitor_error": contention.error,
        "foreign_gpu_contention_detected": bool(contention.events),
        "foreign_gpu_contention_samples": contention.events[:32],
    }


def _config_api(config_name: str) -> dict[str, Any]:
    if config_name == "eager_dynamic":
        return {
            "generate_kwargs": {
                "cache_implementation": "omitted (Transformers creates DynamicCache)",
                "disable_compile": True,
            },
            "decoder_compile": None,
        }
    if config_name == "eager_static":
        return {
            "generate_kwargs": {
                "cache_implementation": "static",
                "max_cache_len": "input_ids.shape[-1] + max_new_tokens",
                "disable_compile": True,
            },
            "decoder_compile": None,
        }
    if config_name == "compile_default_dynamic":
        return {
            "generate_kwargs": {
                "cache_implementation": "omitted (Transformers creates DynamicCache)",
                "disable_compile": True,
            },
            "decoder_compile": {
                "helper": "vcap.models.torch_compile.apply_compile",
                "target": "loaded.model.model.forward",
                "torch_compile": {"backend": "inductor", "mode": "default", "dynamic": None},
            },
        }
    if config_name == "compile_reduce_overhead_static":
        return {
            "generate_kwargs": {
                "cache_implementation": "static",
                "max_cache_len": "input_ids.shape[-1] + max_new_tokens",
                "disable_compile": True,
            },
            "decoder_compile": {
                "helper": "vcap.models.torch_compile.apply_compile + decode-only dispatcher",
                "target": "loaded.model.model.forward for one-token calls; eager prefill",
                "torch_compile": {
                    "backend": "inductor",
                    "mode": "reduce-overhead",
                    "dynamic": False,
                },
            },
        }
    return {
        "generate_kwargs": {
            "cache_implementation": "static",
            "max_cache_len": "input_ids.shape[-1] + max_new_tokens",
        },
        "decoder_compile": {"manual_cuda_graph": "not run"},
    }


def _summarize_config(item: dict[str, Any]) -> None:
    runs = item.get("runs", [])
    if not runs:
        item["summary"] = None
        return
    steady = runs[1] if len(runs) > 1 else runs[0]
    first = runs[0]
    alternate = runs[2] if len(runs) > 2 else None
    item["summary"] = {
        "steady_run": int(steady["run"]),
        "decode_tok_s": steady["decode_tok_s"],
        "first_token_ms": steady["first_token_ms"],
        "prefill_s": steady["prefill_s"],
        "peak_allocated_mib": steady["peak_allocated_mib"],
        "peak_reserved_mib": steady["peak_reserved_mib"],
        "identical": all(bool(run["identical_to_eager_dynamic"]) for run in runs),
        "finite_logits_sampled": all(bool(run["finite_logits_sampled"]) for run in runs),
        "foreign_gpu_contention_detected": any(
            bool(run.get("foreign_gpu_contention_detected")) for run in runs
        ),
        "warmup_first_generation_s": first["wall_s"],
        "warmup_overhead_vs_second_s": max(0.0, float(first["wall_s"]) - float(steady["wall_s"])),
        "alternate_wall_s": alternate["wall_s"] if alternate else None,
        "alternate_decode_tok_s": alternate["decode_tok_s"] if alternate else None,
    }


def run(args: argparse.Namespace) -> int:
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")
    for source in (PRIMARY_VIDEO, PRIMARY_AUDIO, ALTERNATE_VIDEO):
        if not source.is_file():
            raise FileNotFoundError(source)

    raw_cache_key = args.variant + (f"_{args.cache_tag}" if args.cache_tag else "")
    cache_key = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in raw_cache_key
    )
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(WORK_DIR / "torchinductor" / cache_key)
    os.environ["TRITON_CACHE_DIR"] = str(WORK_DIR / "triton" / cache_key)
    Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["TRITON_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

    import torch
    import transformers

    from vcap.models import captioner_for_loaded
    from vcap.models.base import GenParams, MediaInput, PreprocessParams
    from vcap.models.loader import load_model, unload_model
    from vcap.models.offload import BudgetHint, OffloadPlan
    from vcap.models.registry import MODEL_SPECS, get_variant, variant_to_family
    from vcap.models.vram_presets import preset_for
    from vcap.models.quant import convrot

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected exactly one visible CUDA device, found {torch.cuda.device_count()}")
    device = torch.device("cuda:0")
    physical_name = torch.cuda.get_device_name(device)
    print(f"GPU guard: visible=0, device_count=1, device={physical_name}", flush=True)

    # ConvRot's offline kernel choice cache normally sits beside convrot.py.
    # Redirect it before any model forward to honor the no-vcap-writes rule.
    convrot._KERNEL_CACHE_PATH = WORK_DIR / f"convrot_kernel_cache_{cache_key}.json"
    convrot._KERNEL_CACHE_MEMORY = None

    output_path = (
        args.output
        if args.output is not None
        else RESULTS_DIR / f"v14_probe_{args.variant}.json"
    ).expanduser().resolve(strict=False)
    registered = get_variant(args.variant)
    family = variant_to_family(registered.key)
    spec = MODEL_SPECS[family]
    profile = preset_for(family, args.profile_tier)
    profile_offload = profile.offload
    offload = OffloadPlan(
        gpu_layers=profile_offload.gpu_layers,
        offload_experts=profile_offload.offload_experts,
        max_memory=profile_offload.max_memory,
        pin_cpu=profile_offload.pin_cpu,
        vram_reserve_gb=profile_offload.vram_reserve_gb,
        swap_slots=profile_offload.swap_slots,
    )
    defaults = {item.name: item.default for item in spec.param_schema}
    generation = GenParams(
        temperature=float(defaults.get("temperature", 0.0)),
        top_p=float(defaults.get("top_p", 1.0)),
        top_k=int(defaults.get("top_k", 0)),
        repetition_penalty=1.0,
        max_new_tokens=int(args.max_new_tokens),
        do_sample=False,
        use_cache=True,
        enable_thinking=bool(defaults.get("enable_thinking", True)),
        seed=int(args.seed),
    )
    preprocessing = PreprocessParams(
        fps=float(profile.fps),
        max_frames=int(profile.max_frames),
        max_pixels=int(profile.max_pixels),
        min_pixels=spec.limits.min_pixels,
        use_audio_in_video="video_audio" in spec.capabilities,
    )
    compile_environment = _compile_environment()
    try:
        import triton

        triton_version = getattr(triton, "__version__", None)
    except Exception as exc:
        triton_version = f"unavailable: {type(exc).__name__}: {exc}"

    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "running",
        "variant": registered.key,
        "family": family,
        "configuration": {
            "profile_tier_gb": int(args.profile_tier),
            "attention_requested": profile.attention,
            "offload": asdict(offload),
            "generation": asdict(generation),
            "preprocessing": asdict(preprocessing),
            "config_order": list(args.configs),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "triton": triton_version,
            "gpu": physical_name,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
            "torch_cuda": torch.version.cuda,
            "compile_environment": compile_environment,
            "torchinductor_cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
            "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
        },
        "transformers_api_facts": {
            "generate_static_kwargs": {
                "cache_implementation": "static",
                "max_cache_len": "prompt input_ids length + max_new_tokens",
            },
            "direct_static_cache_constructor": (
                "transformers.StaticCache(config=model.config.get_text_config(decoder=True), "
                "max_cache_len=prompt_length + max_new_tokens)"
            ),
            "cache_conflict": (
                "Do not pass cache_implementation together with past_key_values; "
                "Transformers 5.16.1 rejects both."
            ),
            "auto_compile_note": (
                "A StaticCache is compileable and generate auto-compiles decode by default; "
                "disable_compile=True is required for a genuinely eager_static control."
            ),
            "hf_compile_config_default": {
                "backend": "inductor",
                "mode": "reduce-overhead",
                "dynamic": None,
                "fullgraph": False,
            },
        },
        "load": None,
        "inputs": {},
        "configs": [],
        "errors": [],
    }
    _write_json(output_path, report)

    loaded = None
    exit_code = 0
    try:
        load_kwargs: dict[str, Any] = {
            "device": "cuda:0",
            "gpu_index": 0,
            "attention": profile.attention,
            "offload": offload,
            "progress_cb": progress,
        }
        if "budget_hint" in inspect.signature(load_model).parameters:
            load_kwargs["budget_hint"] = BudgetHint(
                max_frames=preprocessing.max_frames,
                max_pixels=preprocessing.max_pixels,
                fps=preprocessing.fps,
                max_new_tokens=generation.max_new_tokens,
                context_tokens=spec.limits.context_tokens,
            )
        print(f"loading {registered.key} with production {args.profile_tier} GB plan", flush=True)
        loaded = load_model(registered.key, **load_kwargs)
        report["load"] = asdict(loaded.load_report)
        report["configuration"]["attention_resolved"] = loaded.attention
        swap = loaded.load_report.block_swap or {}
        swapped_layers = int(swap.get("swapped_layers", 0) or 0)
        report["configuration"]["resident"] = swapped_layers == 0
        if swapped_layers:
            print(f"WARNING: loader selected {swapped_layers} swapped decoder layers", flush=True)

        captioner = captioner_for_loaded(loaded)
        if family == "qwen3_omni_captioner":
            primary_media = MediaInput(path=PRIMARY_AUDIO, kind="audio")
            alternate_media = MediaInput(path=ALTERNATE_VIDEO, kind="audio", start=0.0, end=20.0)
        else:
            primary_media = MediaInput(path=PRIMARY_VIDEO, start=0.0, end=20.0)
            alternate_media = MediaInput(path=ALTERNATE_VIDEO, start=0.0, end=20.0)
        primary = _prepare_case(
            captioner, primary_media, preprocessing, generation, "primary", torch
        )
        alternate = _prepare_case(
            captioner, alternate_media, preprocessing, generation, "alternate", torch
        )
        cases = (primary, primary, alternate)
        report["inputs"] = {
            "primary": {key: _json_safe(value) for key, value in primary.items() if key not in {"prepared", "inputs"}},
            "alternate": {
                key: _json_safe(value)
                for key, value in alternate.items()
                if key not in {"prepared", "inputs"}
            },
        }
        _write_json(output_path, report)

        references: dict[str, list[int]] = {}
        for config_index, config_name in enumerate(args.configs, start=1):
            item: dict[str, Any] = {
                "name": config_name,
                "status": "running",
                "api": _config_api(config_name),
                "compile_setup": None,
                "runs": [],
                "dynamo_counters": {},
                "error": None,
            }
            report["configs"].append(item)
            print(f"\n[{config_index}/{len(args.configs)}] {registered.key} / {config_name}", flush=True)
            if config_name == "manual_cudagraph_static":
                item["status"] = "skipped"
                item["error"] = (
                    "Skipped: robust manual replay would need fixed outer-model output/input buffers, "
                    "in-place StaticCache reset semantics, and special handling for ConvRot MoE routing. "
                    "The decode-only reduce-overhead path is the supported graph experiment."
                )
                _summarize_config(item)
                _write_json(output_path, report)
                print(item["error"], flush=True)
                continue
            if swapped_layers and config_name.startswith("compile_"):
                item["status"] = "skipped"
                item["error"] = "Block-swapped decoder parameters change bindings and are not compile/graph safe."
                _summarize_config(item)
                _write_json(output_path, report)
                print(item["error"], flush=True)
                continue

            for attr in ("_previous_max_cache_length", "_compiled_call", "_last_compile_config"):
                if attr in getattr(loaded.model, "__dict__", {}):
                    delattr(loaded.model, attr)
            gc.collect()
            torch.cuda.empty_cache()
            counter_before = _counter_snapshot(torch)
            compiled = False
            try:
                if config_name.startswith("compile_"):
                    item["compile_setup"] = _install_compile(loaded.model, family, config_name)
                    compiled = True
                    print("compile wrapper installed; first execution performs code generation", flush=True)
                for run_index, case in enumerate(cases, start=1):
                    reference = references.get(str(case["label"]))
                    print(
                        f"run {run_index}/3 ({case['label']}, prompt={case['prompt_tokens']} tokens)",
                        flush=True,
                    )
                    run_counter_before = _counter_snapshot(torch)
                    measurement = _run_generation(
                        loaded,
                        case,
                        config_name,
                        run_index,
                        args.max_new_tokens,
                        args.seed,
                        reference,
                        torch,
                    )
                    measurement["dynamo_counters"] = _counter_delta(
                        run_counter_before, _counter_snapshot(torch)
                    )
                    item["runs"].append(measurement)
                    if config_name == "eager_dynamic" and str(case["label"]) not in references:
                        references[str(case["label"])] = list(measurement["token_ids"])
                    print(
                        f"  {measurement['generated_tokens']} tok, "
                        f"decode={measurement['decode_tok_s']:.2f} tok/s, "
                        f"TTFT={measurement['first_token_ms']:.1f} ms, "
                        f"prefill={measurement['prefill_s']:.3f}s, "
                        f"peak={measurement['peak_allocated_mib']:.0f}/"
                        f"{measurement['peak_reserved_mib']:.0f} MiB alloc/reserved, "
                        f"identical={measurement['identical_to_eager_dynamic']}, "
                        f"foreign_gpu={measurement['foreign_gpu_contention_detected']}",
                        flush=True,
                    )
                    _write_json(output_path, report)
                item["status"] = "success"
            except BaseException as exc:
                exit_code = 1
                item["status"] = "failed"
                item["error"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(limit=30),
                }
                report["errors"].append({"config": config_name, **item["error"]})
                print(f"FAILED {config_name}: {type(exc).__name__}: {exc}", flush=True)
                if isinstance(exc, ForeignGpuContention):
                    report["status"] = "contaminated_by_foreign_gpu_process"
                    _write_json(output_path, report)
                    raise
                try:
                    torch.cuda.synchronize(device)
                except Exception:
                    pass
            finally:
                item["dynamo_counters"] = _counter_delta(counter_before, _counter_snapshot(torch))
                if compiled:
                    try:
                        item["compile_restored_modules"] = _release_compile(loaded.model, torch)
                    except Exception as exc:
                        item["compile_release_error"] = f"{type(exc).__name__}: {exc}"
                _summarize_config(item)
                _write_json(output_path, report)

        baseline = next(
            (item for item in report["configs"] if item["name"] == "eager_dynamic" and item.get("summary")),
            None,
        )
        if baseline is not None:
            base_alloc = float(baseline["summary"]["peak_allocated_mib"])
            base_reserved = float(baseline["summary"]["peak_reserved_mib"])
            for item in report["configs"]:
                summary = item.get("summary")
                if summary:
                    summary["peak_allocated_delta_mib_vs_baseline"] = (
                        float(summary["peak_allocated_mib"]) - base_alloc
                    )
                    summary["peak_reserved_delta_mib_vs_baseline"] = (
                        float(summary["peak_reserved_mib"]) - base_reserved
                    )
        report["status"] = "complete_with_failures" if report["errors"] else "complete"
    except BaseException as exc:
        exit_code = 1
        report["status"] = "failed"
        report["errors"].append(
            {
                "config": None,
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(limit=30),
            }
        )
        print(f"FATAL {registered.key}: {type(exc).__name__}: {exc}", flush=True)
    finally:
        if loaded is not None:
            try:
                restored = _release_compile(loaded.model, torch)
                if restored:
                    print(f"restored {restored} compiled module(s) before unload", flush=True)
            except Exception:
                pass
            try:
                unload = unload_model(loaded)
                report["unload"] = asdict(unload)
                print("unload: " + json.dumps(asdict(unload), default=str), flush=True)
            except Exception as exc:
                report["unload_error"] = f"{type(exc).__name__}: {exc}"
        _write_json(output_path, report)
        print(f"raw JSON: {output_path}", flush=True)
    return exit_code


def main(argv: list[str] | None = None) -> int:
    try:
        from vcap.core.logs import setup_utf8_stdio

        setup_utf8_stdio()
    except ImportError:
        pass
    try:
        return run(parse_args(argv))
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr, flush=True)
        return 130
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
