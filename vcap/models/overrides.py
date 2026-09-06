"""Custom checkpoint overrides for the main (video) and audio caption models.

An override replaces the *files* of the selected model variant while the
variant keeps supplying the family behaviour: prompts, media limits,
generation schema, and capabilities. The backend is chosen from the override
itself:

* a ``.gguf`` file, or a folder holding one, runs through the private
  ``llama-server`` (any llama.cpp multimodal GGUF, for example a Q2/Q3
  Qwen3-Omni conversion, an abliterated Qwen3-Omni, or Qwen2.5-VL);
* a Transformers folder (``config.json`` + safetensors) loads through the
  regular Transformers path and must match the selected family's
  architecture;
* a CTranslate2 folder (``model.bin`` + ``config.json``) replaces the Whisper
  speech model and is therefore only meaningful as the audio override.

This module deliberately has no Torch dependency: the UI validates paths
with it before a job is submitted.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Mapping

OverrideKind = Literal["gguf", "transformers", "ctranslate2"]

_GGUF_SHARD_RE = re.compile(r"^(?P<stem>.+)-(?P<index>\d{5})-of-(?P<total>\d{5})\.gguf$", re.IGNORECASE)
_TRANSFORMERS_MODEL_TYPES: dict[str, frozenset[str]] = {
    "qwen2_5_omni_thinker": frozenset({"qwen2_5_omni", "qwen2_5_omni_thinker"}),
    "qwen3_omni_moe_thinker": frozenset({"qwen3_omni_moe", "qwen3_omni_moe_thinker"}),
}
_KIND_LABELS = {
    "gguf": "GGUF (llama.cpp)",
    "transformers": "Transformers checkpoint",
    "ctranslate2": "Whisper CTranslate2 model",
}


@dataclass(frozen=True)
class ModelOverride:
    """One resolved custom checkpoint and, for GGUF, its multimodal projector."""

    kind: OverrideKind
    path: Path
    mmproj: Path | None = None
    size_bytes: int = 0
    model_type: str = ""
    notes: tuple[str, ...] = ()

    @property
    def backend(self) -> str:
        if self.kind == "gguf":
            return "llamacpp"
        if self.kind == "transformers":
            return "transformers"
        return "ctranslate2"

    @property
    def kind_label(self) -> str:
        return _KIND_LABELS[self.kind]

    @property
    def name(self) -> str:
        return self.path.name or str(self.path)

    @property
    def label(self) -> str:
        text = f"{self.name} ({self.kind_label}, {format_bytes(self.size_bytes)})"
        if self.kind == "gguf":
            text += f" + {self.mmproj.name}" if self.mmproj is not None else " without mmproj"
        elif self.model_type:
            text += f" [{self.model_type}]"
        return text

    @property
    def is_sharded_transformers(self) -> bool:
        if self.kind != "transformers":
            return False
        return not (self.path / "model.safetensors").is_file()

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "backend": self.backend,
            "path": str(self.path),
            "mmproj": str(self.mmproj) if self.mmproj is not None else None,
            "size_bytes": int(self.size_bytes),
            "model_type": self.model_type,
            "label": self.label,
            "notes": list(self.notes),
        }


def format_bytes(value: int) -> str:
    """Format a byte count in decimal units, matching the model tables."""

    size = max(0, int(value or 0))
    if size >= 1_000_000_000:
        return f"{size / 1_000_000_000:.1f} GB"
    if size >= 1_000_000:
        return f"{size / 1_000_000:.0f} MB"
    return f"{size} bytes"


def _clean_path(raw: Any) -> Path | None:
    text = str(raw or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()
    if not text:
        return None
    return Path(text).expanduser().resolve(strict=False)


def _file_size(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except OSError:
        return 0


def _gguf_files(folder: Path) -> list[Path]:
    try:
        return sorted(
            (item for item in folder.iterdir() if item.is_file() and item.suffix.casefold() == ".gguf"),
            key=lambda item: item.name.casefold(),
        )
    except OSError:
        return []


def _is_mmproj(path: Path) -> bool:
    return "mmproj" in path.name.casefold()


def _common_prefix_length(left: str, right: str) -> int:
    count = 0
    for a, b in zip(left.casefold(), right.casefold()):
        if a != b:
            break
        count += 1
    return count


def _pick_gguf_model(candidates: list[Path], folder: Path) -> Path:
    if len(candidates) == 1:
        return candidates[0]
    shards: dict[str, list[tuple[int, Path]]] = {}
    for candidate in candidates:
        match = _GGUF_SHARD_RE.match(candidate.name)
        if match is None:
            shards = {}
            break
        shards.setdefault(match.group("stem").casefold(), []).append((int(match.group("index")), candidate))
    if len(shards) == 1:
        first = min(next(iter(shards.values())), key=lambda item: item[0])
        return first[1]
    names = ", ".join(item.name for item in candidates[:6])
    raise ValueError(
        f"Several GGUF model files were found in {folder}: {names}. "
        "Point the override at the exact .gguf file to use."
    )


def _resolve_mmproj(model: Path, mmproj_raw: Any, notes: list[str]) -> Path | None:
    explicit = _clean_path(mmproj_raw)
    if explicit is not None:
        if not explicit.is_file():
            raise ValueError(f"The mmproj file does not exist: {explicit}")
        if explicit.suffix.casefold() != ".gguf":
            raise ValueError(f"The mmproj path must be a .gguf file: {explicit}")
        return explicit
    candidates = [item for item in _gguf_files(model.parent) if _is_mmproj(item)]
    if not candidates:
        notes.append(
            "No multimodal projector (mmproj) was found beside the model; only text prompts will work. "
            "Set the mmproj path to caption images, video, or audio."
        )
        return None
    if len(candidates) == 1:
        chosen = candidates[0]
    else:
        chosen = max(
            candidates,
            key=lambda item: (_common_prefix_length(item.stem, model.stem), -len(item.name)),
        )
        notes.append(
            f"{len(candidates)} projector files were found beside the model; using {chosen.name}. "
            "Set the mmproj path explicitly to pick another."
        )
    return chosen


def _transformers_model_type(folder: Path) -> str:
    try:
        payload = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return ""
    if not isinstance(payload, Mapping):
        return ""
    value = payload.get("model_type")
    if not value:
        thinker = payload.get("thinker_config")
        value = thinker.get("model_type") if isinstance(thinker, Mapping) else ""
    return str(value or "")


def _safetensors_bytes(folder: Path) -> int:
    try:
        return sum(
            _file_size(item)
            for item in folder.iterdir()
            if item.is_file() and item.suffix.casefold() == ".safetensors"
        )
    except OSError:
        return 0


def _has_transformers_weights(folder: Path) -> bool:
    if not (folder / "config.json").is_file():
        return False
    if (folder / "model.safetensors").is_file() or (folder / "model.safetensors.index.json").is_file():
        return True
    try:
        return any(item.is_file() and item.suffix.casefold() == ".safetensors" for item in folder.iterdir())
    except OSError:
        return False


def _is_ctranslate2_folder(folder: Path) -> bool:
    return (folder / "model.bin").is_file() and (folder / "config.json").is_file()


def classify_model_path(raw: Any, mmproj_raw: Any = "") -> ModelOverride | None:
    """Inspect an override path and describe how it can be loaded.

    Returns ``None`` for an empty field. Raises :class:`ValueError` with a
    user-facing message for a missing or unrecognized path.
    """

    path = _clean_path(raw)
    if path is None:
        return None
    if not path.exists():
        raise ValueError(f"The override path does not exist: {path}")
    notes: list[str] = []
    if path.is_file():
        suffix = path.suffix.casefold()
        if suffix == ".gguf":
            if _is_mmproj(path):
                raise ValueError(
                    f"{path.name} is a multimodal projector (mmproj), not a model. Put it in the mmproj "
                    "field and point the model field at the main .gguf file."
                )
            mmproj = _resolve_mmproj(path, mmproj_raw, notes)
            return ModelOverride("gguf", path, mmproj, _file_size(path), "", tuple(notes))
        if suffix == ".safetensors":
            folder = path.parent
            if not _has_transformers_weights(folder):
                raise ValueError(f"{folder} has no config.json beside {path.name}; point at a complete checkpoint folder.")
            return ModelOverride(
                "transformers",
                folder,
                None,
                _safetensors_bytes(folder),
                _transformers_model_type(folder),
                tuple(notes),
            )
        raise ValueError(
            f"Unsupported model file {path.name}: expected a .gguf file, a model.safetensors file, "
            "or a folder containing the checkpoint."
        )
    ggufs = _gguf_files(path)
    if ggufs:
        models = [item for item in ggufs if not _is_mmproj(item)]
        if not models:
            raise ValueError(
                f"{path} only contains a multimodal projector; add the main .gguf model file or point at it directly."
            )
        model = _pick_gguf_model(models, path)
        mmproj = _resolve_mmproj(model, mmproj_raw, notes)
        return ModelOverride("gguf", model, mmproj, _file_size(model), "", tuple(notes))
    if _has_transformers_weights(path):
        return ModelOverride(
            "transformers",
            path,
            None,
            _safetensors_bytes(path),
            _transformers_model_type(path),
            tuple(notes),
        )
    if _is_ctranslate2_folder(path):
        return ModelOverride("ctranslate2", path, None, _file_size(path / "model.bin"), "", tuple(notes))
    raise ValueError(
        f"No model files were found in {path}: expected a .gguf file, config.json with safetensors weights, "
        "or a faster-whisper folder with model.bin and config.json."
    )


def _family_spec(variant_key: str) -> tuple[str, Any]:
    from .registry import MODEL_SPECS, variant_to_family

    family = variant_to_family(str(variant_key))
    return family, MODEL_SPECS[family]


def _check_family_compatibility(override: ModelOverride, variant_key: str, role: str) -> ModelOverride:
    family, spec = _family_spec(variant_key)
    if override.kind == "gguf":
        if not family.startswith("qwen3_omni_"):
            raise ValueError(
                "GGUF overrides run through llama.cpp, which this application supports for the Qwen3-Omni "
                f"families. Select a Qwen3-Omni Instruct, Thinking, or Captioner variant as the base model "
                f"for {override.name} instead of {spec.label}."
            )
        return override
    if override.kind == "transformers":
        accepted = _TRANSFORMERS_MODEL_TYPES.get(spec.architecture, frozenset())
        if override.model_type and override.model_type not in accepted:
            raise ValueError(
                f"The Transformers checkpoint at {override.path} is a '{override.model_type}' model, but the "
                f"selected {spec.label} base expects {spec.architecture}. Pick the matching base model variant, "
                "or use a GGUF conversion to run other architectures through llama.cpp."
            )
        if not override.model_type:
            return replace(
                override,
                notes=override.notes + ("config.json does not declare a model_type; loading will verify the architecture.",),
            )
        return override
    raise ValueError(
        f"{override.path} is a Whisper (CTranslate2) model folder. It can only replace the Whisper speech model: "
        f"put it in the audio caption model override, not the {role} model override."
    )


def resolve_main_override(
    variant_key: str,
    raw: Any,
    mmproj_raw: Any = "",
    *,
    role: str = "main",
) -> ModelOverride | None:
    """Validate a caption-model override against the variant it replaces.

    ``role`` only labels error messages ("main" for the selected caption
    model, "audio" for the sound-caption Captioner session).
    """

    override = classify_model_path(raw, mmproj_raw)
    if override is None:
        return None
    return _check_family_compatibility(override, variant_key, role)


def resolve_audio_override(captioner_variant_key: str, raw: Any, mmproj_raw: Any = "") -> ModelOverride | None:
    """Validate the audio override for the Qwen3-Omni Captioner.

    A CTranslate2 folder targets Whisper instead and yields ``None`` here; use
    :func:`whisper_override_folder` for that side.
    """

    override = classify_model_path(raw, mmproj_raw)
    if override is None or override.kind == "ctranslate2":
        return None
    return _check_family_compatibility(override, captioner_variant_key, "audio")


def whisper_override_folder(raw: Any) -> Path | None:
    """Return the audio override path when it is a faster-whisper CTranslate2 folder."""

    try:
        override = classify_model_path(raw)
    except ValueError:
        return None
    if override is None or override.kind != "ctranslate2":
        return None
    return override.path


def override_variant(base: Any, override: ModelOverride) -> Any:
    """Return a registry-shaped variant describing the override files.

    The key stays the base variant's key so cache, unload, and metadata
    comparisons keep working; the backend, scheme, and file list describe what
    is actually loaded.
    """

    if override.kind == "gguf":
        files: tuple[str, ...] = (override.path.name,)
        if override.mmproj is not None:
            files += (override.mmproj.name,)
        mmproj_bytes = _file_size(override.mmproj) if override.mmproj is not None else 0
        return replace(
            base,
            label=f"{base.label} · override {override.name}",
            scheme="gguf",
            size_gb=(override.size_bytes + mmproj_bytes) / 1_000_000_000,
            folder_name=str(override.path.parent),
            gguf_files=files,
            gguf_repo=None,
            gguf_file_sizes=None,
            gguf_sha256=None,
            backend="llamacpp",
        )
    if override.kind == "transformers":
        scheme = (
            "bf16"
            if override.is_sharded_transformers or getattr(base, "backend", "") != "transformers"
            else base.scheme
        )
        return replace(
            base,
            label=f"{base.label} · override {override.name}",
            scheme=scheme,
            size_gb=override.size_bytes / 1_000_000_000,
            folder_name=str(override.path),
            gguf_files=None,
            gguf_repo=None,
            gguf_file_sizes=None,
            gguf_sha256=None,
            backend="transformers",
        )
    raise ValueError("Whisper CTranslate2 folders cannot replace a caption model variant")


def override_identity(raw: Any, mmproj_raw: Any = "") -> tuple[str, str]:
    """Return a normalized (model, mmproj) pair for cache and session comparisons."""

    model = _clean_path(raw)
    mmproj = _clean_path(mmproj_raw)
    return (str(model) if model is not None else "", str(mmproj) if mmproj is not None else "")


def describe_override(
    role: str,
    raw: Any,
    mmproj_raw: Any,
    variant_key: str,
    *,
    audio_source: str | None = None,
) -> dict[str, Any]:
    """Summarize one override field for status lines: state, text, override."""

    text = str(raw or "").strip()
    if not text:
        return {"state": "off", "text": "", "override": None}
    try:
        if role == "audio":
            classified = classify_model_path(raw, mmproj_raw)
            override = classified
            if classified is not None and classified.kind != "ctranslate2":
                override = _check_family_compatibility(classified, variant_key, "audio")
        else:
            override = resolve_main_override(variant_key, raw, mmproj_raw)
    except ValueError as exc:
        return {"state": "error", "text": str(exc), "override": None}
    if override is None:
        return {"state": "off", "text": "", "override": None}
    notes = list(override.notes)
    state = "ok"
    if role == "audio":
        source = str(audio_source or "none").strip().casefold()
        if override.kind == "ctranslate2":
            if source in {"captioner"}:
                notes.append(
                    "This Whisper folder is used for speech transcripts; the audio caption source is the Captioner, "
                    "so it will not change the sound captions."
                )
                state = "warn"
        elif source in {"none", "whisper"}:
            notes.append(
                "This override replaces the Qwen3-Omni Captioner sound-caption model; it is unused while the audio "
                "caption source is Off or Whisper only."
            )
            state = "warn"
    elif override.kind == "gguf" and override.mmproj is None:
        state = "warn"
    return {"state": state, "text": override.label, "override": override, "notes": notes}


def override_settings_problems(settings: Mapping[str, Any]) -> list[str]:
    """Return blocking validation errors for the override fields of a settings mapping."""

    problems: list[str] = []
    variant_key = str(settings.get("model_key") or settings.get("variant_key") or "")
    try:
        resolve_main_override(
            variant_key,
            settings.get("override_video_model_path", ""),
            settings.get("override_video_mmproj_path", ""),
        )
    except ValueError as exc:
        problems.append(f"Video / main model override: {exc}")
    except KeyError as exc:
        problems.append(f"Video / main model override: {exc}")
    audio_raw = settings.get("override_audio_model_path", "")
    if str(audio_raw or "").strip():
        try:
            classified = classify_model_path(audio_raw, settings.get("override_audio_mmproj_path", ""))
            if classified is not None and classified.kind != "ctranslate2":
                _check_family_compatibility(classified, "qwen3_omni_captioner_int4", "audio")
        except ValueError as exc:
            problems.append(f"Audio caption model override: {exc}")
    return problems


__all__ = [
    "ModelOverride",
    "OverrideKind",
    "classify_model_path",
    "describe_override",
    "format_bytes",
    "override_identity",
    "override_settings_problems",
    "override_variant",
    "resolve_audio_override",
    "resolve_main_override",
    "whisper_override_folder",
]
