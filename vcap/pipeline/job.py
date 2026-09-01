"""Typed, JSON-serializable job and result contracts for the caption pipeline."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from vcap import OUTPUTS_DIR


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (Path, os.PathLike)):
        return os.fspath(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _load_json_payload(payload: str | bytes | os.PathLike[str]) -> dict[str, Any]:
    if isinstance(payload, bytes):
        text = payload.decode("utf-8")
    else:
        text = os.fspath(payload)
        stripped = text.lstrip()
        if not stripped.startswith(("{", "[")):
            candidate = Path(text)
            if candidate.is_file():
                text = candidate.read_text(encoding="utf-8")
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Serialized job data must be a JSON object")
    return value


def _setting(settings: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in settings:
            return settings[key]
    return default


def _bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().casefold()
    if text in {"1", "true", "yes", "on", "enabled"}:
        return True
    if text in {"0", "false", "no", "off", "disabled", ""}:
        return False
    return default


def _int(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _gpu_layers(value: Any) -> int | str:
    if value is None:
        return "auto"
    normalized = str(value).strip().casefold()
    if normalized in {"auto", "all"}:
        return normalized
    return max(0, _int(value, 0))


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)


def _optional_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if parsed > 0 else None


def _gpu_indices(value: Any, fallback: int) -> tuple[int, ...]:
    if value is None or value == "":
        return (int(fallback),)
    if isinstance(value, str):
        raw: Sequence[Any] = [part for part in re.split(r"[\s,;]+", value.strip()) if part]
    elif isinstance(value, Sequence):
        raw = value
    else:
        raw = [value]
    result: list[int] = []
    for item in raw:
        try:
            index = int(item)
        except (TypeError, ValueError):
            continue
        if index >= 0 and index not in result:
            result.append(index)
    return tuple(result or [int(fallback)])


def _formats(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        values = value.replace(";", ",").split(",")
    elif isinstance(value, Sequence):
        values = value
    else:
        values = ("txt",)
    normalized = [str(item).strip().casefold().lstrip(".") for item in values]
    normalized = [item for item in normalized if item]
    if "txt" not in normalized:
        normalized.insert(0, "txt")
    return tuple(dict.fromkeys(normalized))


def _replace_pairs(value: Any) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        result: list[tuple[str, str]] = []
        for line in value.splitlines():
            for piece in line.split("|"):
                if ";" not in piece:
                    continue
                source, target = piece.split(";", 1)
                if source.strip():
                    result.append((source.strip(), target.strip()))
        return tuple(result)
    result = []
    for item in value:
        if isinstance(item, Sequence) and len(item) >= 2 and str(item[0]).strip():
            result.append((str(item[0]).strip(), str(item[1]).strip()))
    return tuple(result)


@dataclass(frozen=True)
class InputItem:
    """One path, text file, or direct text prompt supplied to a job."""

    path: str | os.PathLike[str] = ""
    kind: str = "auto"
    text_prompt_only: bool = False
    text: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", os.fspath(self.path))
        object.__setattr__(self, "kind", str(self.kind or "auto").casefold())


@dataclass(frozen=True)
class OffloadSpec:
    """Serializable counterpart of :class:`vcap.models.offload.OffloadPlan`."""

    gpu_layers: int | str = "auto"
    offload_experts: bool = False
    max_memory: dict[str, str] | None = None
    pin_cpu: bool = True
    vram_reserve_gb: float = 2.0
    swap_slots: int = 2

    def __post_init__(self) -> None:
        object.__setattr__(self, "gpu_layers", _gpu_layers(self.gpu_layers))
        object.__setattr__(self, "offload_experts", _bool(self.offload_experts, False))
        object.__setattr__(self, "pin_cpu", _bool(self.pin_cpu, True))
        object.__setattr__(self, "vram_reserve_gb", max(0.0, _float(self.vram_reserve_gb, 2.0)))
        object.__setattr__(self, "swap_slots", max(1, min(4, _int(self.swap_slots, 2))))


@dataclass(frozen=True)
class ModelChoice:
    """Selected checkpoint and model-loading policy."""

    variant_key: str = "qwen3_omni_instruct_int8"
    attention: str = "auto"
    vram_preset: str = "auto"
    offload: OffloadSpec = field(default_factory=OffloadSpec)


@dataclass(frozen=True)
class PromptSpec:
    """Prompt preset plus direct prompt overrides and template variables."""

    preset_id: str | None = None
    system_prompt: str | None = None
    user_prompt: str | None = None
    variables: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GenParams:
    """Generation controls shared by every captioner wrapper."""

    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    max_new_tokens: int = 2048
    do_sample: bool = False
    use_cache: bool = True
    enable_thinking: bool = True


@dataclass(frozen=True)
class PreprocessSpec:
    """Trim, frame sampling, pixel, and normalization controls."""

    trim_start_s: float = 0.0
    trim_end_s: float | None = None
    fps: float = 2.0
    max_frames: int = 160
    max_pixels: int = 297_920
    min_pixels: int | None = None
    sampling_strategy: str = "fps"
    normalize_clip: bool = False
    use_audio_in_video: bool = True
    audio_sample_rate: int = 16_000


@dataclass(frozen=True)
class SplitSpec:
    """Scene/fixed/trainer segmentation and optional clip rejection policy."""

    mode: Literal["whole", "scenes", "fixed", "trainer"] = "whole"
    cut_mode: Literal["copy", "precise"] = "copy"
    scene_threshold: float = 27.0
    scene_min_len_s: float = 1.0
    scene_max_len_s: float = 0.0
    merge_short_scenes: bool = True
    merge_below_s: float = 1.0
    fade_detection: bool = False
    scene_detector: Literal["content", "adaptive", "threshold"] = "content"
    scene_downscale: int = 0
    fixed_chunk_s: float = 0.0
    model_max_duration_s: float | None = None
    trainer_target: Any = None
    overlap_s: float = 0.0
    auto_reject: bool = False
    reject_min_duration_s: float = 0.0
    reject_max_black_ratio: float = 1.0
    reject_max_static_score: float = -1.0
    reject_min_sharpness: float = 0.0
    reject_require_audio: bool = False
    reject_max_silence_ratio: float = 1.0


@dataclass(frozen=True)
class PostSpec:
    """Caption cleanup, injection, replacement, and output format controls."""

    prefix: str = ""
    suffix: str = ""
    trigger: str = ""
    trigger_mode: Literal["prefix", "suffix", "none"] = "prefix"
    replace_pairs: tuple[tuple[str, str], ...] = ()
    replace_regex: bool = False
    replace_case_insensitive: bool = True
    replace_whole_words: bool = True
    collapse_whitespace: bool = False
    formats: tuple[str, ...] = ("txt",)
    save_reasoning: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "formats", _formats(self.formats))
        object.__setattr__(self, "replace_pairs", _replace_pairs(self.replace_pairs))


@dataclass(frozen=True)
class OutputSpec:
    """Single-run or stable batch-caption output layout."""

    kind: Literal["single", "batch"] = "single"
    outputs_root: str | os.PathLike[str] = field(default_factory=lambda: str(OUTPUTS_DIR))
    batch_output_dir: str | os.PathLike[str] | None = None
    source_root: str | os.PathLike[str] | None = None
    mirror_names: bool = True
    overwrite: bool = False
    save_processed_files: bool = False
    save_clips: bool = False
    recursive: bool = False
    limit_items: int = 0

    def __post_init__(self) -> None:
        selected = str(self.kind).casefold()
        if selected not in {"single", "batch"}:
            raise ValueError("OutputSpec.kind must be 'single' or 'batch'")
        object.__setattr__(self, "kind", selected)
        object.__setattr__(self, "outputs_root", os.fspath(self.outputs_root))
        if self.batch_output_dir is not None:
            object.__setattr__(self, "batch_output_dir", os.fspath(self.batch_output_dir))
        if self.source_root is not None:
            object.__setattr__(self, "source_root", os.fspath(self.source_root))
        object.__setattr__(self, "limit_items", max(0, int(self.limit_items)))


@dataclass(frozen=True)
class RuntimeSpec:
    """Execution mode, model lifetime, GPU selection, and compile policy."""

    subprocess_mode: bool = True
    keep_model_loaded: bool = True
    idle_unload_minutes: float = 10.0
    gpu_index: int = 0
    gpu_indices: tuple[int, ...] = ()
    compile: bool = False

    def __post_init__(self) -> None:
        primary = max(0, int(self.gpu_index))
        indices = _gpu_indices(self.gpu_indices, primary)
        object.__setattr__(self, "gpu_index", primary)
        object.__setattr__(self, "gpu_indices", indices)


@dataclass(frozen=True)
class JobSpec:
    """Complete immutable input to the unified single/batch pipeline."""

    inputs: list[InputItem] = field(default_factory=list)
    output: OutputSpec = field(default_factory=OutputSpec)
    model: ModelChoice = field(default_factory=ModelChoice)
    prompt: PromptSpec = field(default_factory=PromptSpec)
    generation: GenParams = field(default_factory=GenParams)
    preprocess: PreprocessSpec = field(default_factory=PreprocessSpec)
    split: SplitSpec = field(default_factory=SplitSpec)
    post: PostSpec = field(default_factory=PostSpec)
    runtime: RuntimeSpec = field(default_factory=RuntimeSpec)
    context_carry_over: bool = False
    settings: dict[str, Any] = field(default_factory=dict)
    internal: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def gen(self) -> GenParams:
        """Short compatibility alias matching the model interface."""

        return self.generation

    @property
    def pre(self) -> PreprocessSpec:
        """Short compatibility alias matching the model interface."""

        return self.preprocess

    @classmethod
    def from_settings(
        cls,
        settings: dict[str, Any],
        inputs: list[InputItem],
        output: OutputSpec,
    ) -> "JobSpec":
        """Build typed specs from the UI's flat settings dictionary."""

        source = dict(settings or {})
        variant_key = str(
            _setting(source, "variant_key", "model_key", "model_variant", default=ModelChoice().variant_key)
        )
        family_defaults: dict[str, Any] = {}
        default_preset: str | None = None
        try:
            from vcap.models.registry import MODEL_SPECS, variant_to_family

            model_spec = MODEL_SPECS[variant_to_family(variant_key)]
            family_defaults = {item.name: item.default for item in model_spec.param_schema}
            family_defaults.update(
                fps=model_spec.limits.default_fps,
                max_frames=model_spec.limits.max_frames or 768,
                max_pixels=model_spec.limits.default_max_pixels,
                min_pixels=model_spec.limits.min_pixels,
            )
            default_preset = model_spec.default_prompt_preset
        except (KeyError, ValueError):
            pass

        preset_id_raw = _setting(source, "prompt_preset_id", "preset_id", default=default_preset)
        preset_id = str(preset_id_raw) if preset_id_raw not in {None, ""} else None
        preset_generation: dict[str, Any] = {}
        if preset_id:
            try:
                from vcap.prompts.presets import get_preset

                preset_generation = dict(get_preset(preset_id).generation_overrides)
            except KeyError:
                pass

        raw_offload = _setting(source, "offload", "offload_plan", default={})
        if is_dataclass(raw_offload):
            offload_data = asdict(raw_offload)
        elif isinstance(raw_offload, Mapping):
            offload_data = dict(raw_offload)
        else:
            offload_data = {}
        gpu_layers = _gpu_layers(
            _setting(source, "gpu_layers", "layers_on_gpu", default=offload_data.get("gpu_layers", "auto"))
        )
        raw_max_memory = _setting(source, "offload_max_memory", "max_memory", default=offload_data.get("max_memory"))
        max_memory = (
            {str(key): str(value) for key, value in raw_max_memory.items()}
            if isinstance(raw_max_memory, Mapping)
            else None
        )
        model = ModelChoice(
            variant_key=variant_key,
            attention=str(_setting(source, "attention", "attention_backend", default="auto")),
            vram_preset=str(_setting(source, "vram_preset", default="auto")),
            offload=OffloadSpec(
                gpu_layers=gpu_layers,
                offload_experts=_bool(
                    _setting(source, "offload_experts", default=offload_data.get("offload_experts")),
                    False,
                ),
                max_memory=max_memory,
                pin_cpu=_bool(
                    _setting(source, "pin_cpu", default=offload_data.get("pin_cpu", True)),
                    True,
                ),
                vram_reserve_gb=max(
                    0.0,
                    _float(
                        _setting(
                            source,
                            "vram_reserve_gb",
                            default=offload_data.get("vram_reserve_gb", 2.0),
                        ),
                        2.0,
                    ),
                ),
                swap_slots=max(
                    1,
                    min(
                        4,
                        _int(
                            _setting(
                                source,
                                "swap_slots",
                                default=offload_data.get("swap_slots", 2),
                            ),
                            2,
                        ),
                    ),
                ),
            ),
        )

        variable_values = dict(_setting(source, "prompt_variables", "variables", default={}) or {})
        aliases = {
            "TRIGGER": ("trigger_word", "trigger"),
            "LANGUAGE": ("language",),
            "SOURCE_LANGUAGE": ("source_language",),
            "TARGET_LANGUAGE": ("target_language",),
            "CAPTION_LENGTH": ("caption_length",),
            "AVOID": ("avoid_list", "avoid"),
            "SUBJECT_CLASS": ("subject_class",),
            "EXTRA_INSTRUCTIONS": ("extra_instructions",),
        }
        for variable, keys in aliases.items():
            value = _setting(source, *keys, default=None)
            if value is not None:
                variable_values[variable] = value
        prompt = PromptSpec(
            preset_id=preset_id,
            system_prompt=_setting(source, "system_prompt", default=None),
            user_prompt=_setting(source, "user_prompt", default=None),
            variables=variable_values,
        )

        def generation_value(key: str, fallback: Any) -> Any:
            return _setting(source, key, default=preset_generation.get(key, family_defaults.get(key, fallback)))

        generation = GenParams(
            temperature=_float(generation_value("temperature", 0.0), 0.0),
            top_p=_float(generation_value("top_p", 1.0), 1.0),
            top_k=_int(generation_value("top_k", 0), 0),
            repetition_penalty=_float(generation_value("repetition_penalty", 1.0), 1.0),
            max_new_tokens=max(1, _int(generation_value("max_new_tokens", 2048), 2048)),
            do_sample=_bool(generation_value("do_sample", False), False),
            use_cache=_bool(generation_value("use_cache", True), True),
            enable_thinking=_bool(generation_value("enable_thinking", True), True),
        )
        default_frames = max(1, _int(family_defaults.get("max_frames", 160), 160))
        selected_frames = _int(_setting(source, "max_frames", default=default_frames), default_frames)
        if selected_frames <= 0:
            selected_frames = default_frames
        preprocess = PreprocessSpec(
            trim_start_s=max(0.0, _float(_setting(source, "trim_start_s", "trim_start", default=0.0), 0.0)),
            trim_end_s=_optional_float(_setting(source, "trim_end_s", "trim_end", default=None)),
            fps=max(0.01, _float(_setting(source, "fps", default=family_defaults.get("fps", 2.0)), 2.0)),
            max_frames=selected_frames,
            max_pixels=max(1, _int(_setting(source, "max_pixels", default=family_defaults.get("max_pixels", 297_920)), 297_920)),
            min_pixels=(
                max(1, _int(_setting(source, "min_pixels", default=family_defaults.get("min_pixels")), 1))
                if _setting(source, "min_pixels", default=family_defaults.get("min_pixels")) is not None
                else None
            ),
            sampling_strategy=str(_setting(source, "sampling_strategy", "frame_sampling", default="fps")),
            normalize_clip=_bool(_setting(source, "normalize_clip", default=False), False),
            use_audio_in_video=_bool(
                _setting(source, "use_audio_in_video", default=family_defaults.get("use_audio_in_video", True)),
                True,
            ),
            audio_sample_rate=max(1, _int(_setting(source, "audio_sample_rate", default=16_000), 16_000)),
        )

        raw_segment_mode = str(_setting(source, "segment_mode", "segmentation_mode", default="")).casefold()
        raw_split_mode = str(_setting(source, "split_mode", default="copy")).casefold()
        if raw_segment_mode not in {"whole", "scenes", "fixed", "trainer"}:
            if raw_split_mode in {"whole", "scenes", "fixed", "trainer"}:
                raw_segment_mode = raw_split_mode
            else:
                raw_segment_mode = "scenes" if _bool(_setting(source, "scene_detect_enabled", default=False)) else "whole"
        cut_mode = raw_split_mode if raw_split_mode in {"copy", "precise"} else str(
            _setting(source, "cut_mode", "split_cut_mode", default="copy")
        ).casefold()
        if cut_mode not in {"copy", "precise"}:
            cut_mode = "copy"
        scene_data = _setting(source, "scene_params", default={})
        scene_data = dict(scene_data) if isinstance(scene_data, Mapping) else {}
        split = SplitSpec(
            mode=raw_segment_mode,  # type: ignore[arg-type]
            cut_mode=cut_mode,  # type: ignore[arg-type]
            scene_threshold=_float(_setting(source, "scene_threshold", default=scene_data.get("threshold", 27.0)), 27.0),
            scene_min_len_s=max(0.0, _float(_setting(source, "scene_min_len_s", default=scene_data.get("min_scene_len_s", 1.0)), 1.0)),
            scene_max_len_s=max(0.0, _float(_setting(source, "scene_max_len_s", default=scene_data.get("max_scene_len_s", 0.0)), 0.0)),
            merge_short_scenes=_bool(_setting(source, "merge_short_scenes", default=scene_data.get("merge_short_scenes", True)), True),
            merge_below_s=max(0.0, _float(_setting(source, "merge_below_s", default=scene_data.get("merge_below_s", 1.0)), 1.0)),
            fade_detection=_bool(_setting(source, "fade_detection", default=scene_data.get("fade_detection", False))),
            scene_detector=str(_setting(source, "scene_detector", default=scene_data.get("detector", "content"))),  # type: ignore[arg-type]
            scene_downscale=max(0, _int(_setting(source, "scene_downscale", default=scene_data.get("downscale", 0)), 0)),
            fixed_chunk_s=max(0.0, _float(_setting(source, "fixed_chunk_s", "chunk_s", default=0.0), 0.0)),
            model_max_duration_s=_optional_float(
                _setting(source, "model_max_duration_s", "max_clip_duration_s", default=None)
            ),
            trainer_target=_setting(source, "trainer_target", default=None),
            overlap_s=max(0.0, _float(_setting(source, "sub_split_overlap_s", "overlap_s", default=0.0), 0.0)),
            auto_reject=_bool(_setting(source, "auto_reject", "auto_reject_enabled", default=False)),
            reject_min_duration_s=max(0.0, _float(_setting(source, "reject_min_duration_s", default=0.0), 0.0)),
            reject_max_black_ratio=_float(_setting(source, "reject_max_black_ratio", default=1.0), 1.0),
            reject_max_static_score=_float(_setting(source, "reject_max_static_score", default=-1.0), -1.0),
            reject_min_sharpness=max(0.0, _float(_setting(source, "reject_min_sharpness", default=0.0), 0.0)),
            reject_require_audio=_bool(_setting(source, "reject_require_audio", default=False)),
            reject_max_silence_ratio=_float(_setting(source, "reject_max_silence_ratio", default=1.0), 1.0),
        )
        post = PostSpec(
            prefix=str(_setting(source, "caption_prefix", "prefix", default="") or ""),
            suffix=str(_setting(source, "caption_suffix", "suffix", default="") or ""),
            trigger=str(_setting(source, "trigger_word", "trigger", default="") or ""),
            trigger_mode=str(_setting(source, "trigger_mode", default="prefix")),  # type: ignore[arg-type]
            replace_pairs=_replace_pairs(_setting(source, "replace_pairs", "replace_words", default=())),
            replace_regex=_bool(_setting(source, "replace_regex", "regex_replace", default=False)),
            replace_case_insensitive=_bool(_setting(source, "replace_case_insensitive", default=True), True),
            replace_whole_words=_bool(_setting(source, "replace_whole_words", "replace_whole_words_only", default=True), True),
            collapse_whitespace=_bool(_setting(source, "collapse_whitespace", default=False)),
            formats=_formats(_setting(source, "output_formats", "formats", default=("txt",))),
            save_reasoning=_bool(_setting(source, "save_reasoning", default=False)),
        )
        resolved_output = replace(
            output,
            overwrite=_bool(_setting(source, "overwrite_existing", "overwrite", default=output.overwrite), output.overwrite),
            save_processed_files=_bool(
                _setting(source, "save_processed_files", default=output.save_processed_files),
                output.save_processed_files,
            ),
            save_clips=_bool(_setting(source, "save_clips", default=output.save_clips), output.save_clips),
            mirror_names=_bool(_setting(source, "mirror_names", default=output.mirror_names), output.mirror_names),
            recursive=_bool(_setting(source, "recursive", "batch_recursive", default=output.recursive), output.recursive),
        )
        gpu_index = max(0, _int(_setting(source, "gpu_index", default=0), 0))
        runtime = RuntimeSpec(
            subprocess_mode=_bool(_setting(source, "subprocess_mode", default=True), True),
            keep_model_loaded=_bool(_setting(source, "keep_model_loaded", default=True), True),
            idle_unload_minutes=max(0.0, _float(_setting(source, "idle_unload_minutes", default=10.0), 10.0)),
            gpu_index=gpu_index,
            gpu_indices=_gpu_indices(_setting(source, "gpu_indices", "multi_gpu_indices", default=None), gpu_index),
            compile=_bool(_setting(source, "compile", "torch_compile", default=False)),
        )
        return cls(
            inputs=[item if isinstance(item, InputItem) else InputItem(**item) for item in inputs],
            output=resolved_output,
            model=model,
            prompt=prompt,
            generation=generation,
            preprocess=preprocess,
            split=split,
            post=post,
            runtime=runtime,
            context_carry_over=_bool(_setting(source, "context_carry_over", default=False), False),
            settings=_json_safe(source),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the complete worker-protocol mapping."""

        return {"_schema_version": 1, **_json_safe(asdict(self))}

    def to_json(self, path: str | os.PathLike[str] | None = None, *, indent: int = 2) -> str:
        """Serialize to UTF-8 JSON and optionally write it to ``path``."""

        text = json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
        if path is not None:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text + "\n", encoding="utf-8")
        return text

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "JobSpec":
        """Rehydrate a job sent through the JSON-lines worker protocol."""

        data = dict(value)
        data.pop("_schema_version", None)
        model_data = dict(data.get("model") or {})
        model_data["offload"] = OffloadSpec(**dict(model_data.get("offload") or {}))
        post_data = dict(data.get("post") or {})
        output_data = dict(data.get("output") or {})
        runtime_data = dict(data.get("runtime") or {})
        return cls(
            inputs=[item if isinstance(item, InputItem) else InputItem(**dict(item)) for item in data.get("inputs", [])],
            output=OutputSpec(**output_data),
            model=ModelChoice(**model_data),
            prompt=PromptSpec(**dict(data.get("prompt") or {})),
            generation=GenParams(**dict(data.get("generation") or data.get("gen") or {})),
            preprocess=PreprocessSpec(**dict(data.get("preprocess") or data.get("pre") or {})),
            split=SplitSpec(**dict(data.get("split") or {})),
            post=PostSpec(**post_data),
            runtime=RuntimeSpec(**runtime_data),
            context_carry_over=_bool(data.get("context_carry_over"), False),
            settings=dict(data.get("settings") or {}),
            internal=dict(data.get("internal") or {}),
        )

    @classmethod
    def from_json(cls, payload: str | bytes | os.PathLike[str]) -> "JobSpec":
        """Read a job from a JSON string, bytes, or UTF-8 file path."""

        return cls.from_dict(_load_json_payload(payload))


@dataclass
class ItemResult:
    """Terminal status and artifacts for one resolved input item."""

    index: int
    path: str
    kind: str
    status: str
    message: str = ""
    outputs: dict[str, str] = field(default_factory=dict)
    segments: list[dict[str, Any]] = field(default_factory=list)
    elapsed: float = 0.0
    peak_vram_gb: float = 0.0
    traceback_tail: str = ""
    gpu_index: int | None = None

    def __getitem__(self, key: str) -> Any:
        """Allow lightweight mapping-style access in UI table adapters."""

        return getattr(self, key)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ItemResult":
        return cls(**dict(value))


@dataclass
class JobResult:
    """Merged terminal result returned by in-process and worker execution."""

    items: list[ItemResult]
    counts: dict[str, int]
    run_dir: str
    metadata_path: str
    elapsed: float

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "JobResult":
        data = dict(value)
        data["items"] = [
            item if isinstance(item, ItemResult) else ItemResult.from_dict(item)
            for item in data.get("items", [])
        ]
        data["counts"] = {str(key): int(count) for key, count in dict(data.get("counts") or {}).items()}
        return cls(**data)

    @classmethod
    def from_json(cls, payload: str | bytes | os.PathLike[str]) -> "JobResult":
        return cls.from_dict(_load_json_payload(payload))


__all__ = [
    "GenParams",
    "InputItem",
    "ItemResult",
    "JobResult",
    "JobSpec",
    "ModelChoice",
    "OffloadSpec",
    "OutputSpec",
    "PostSpec",
    "PreprocessSpec",
    "PromptSpec",
    "RuntimeSpec",
    "SplitSpec",
]
