"""v1.9.0: txt-only caption output and custom model overrides."""

from __future__ import annotations

import json
import math
import wave
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vcap.core.dataset_captions import (
    DEFAULT_AUDIO_CAPTION_TEMPLATE,
    DEFAULT_CAPTION_MERGE_TEMPLATE,
    caption_unit_paths,
)
from vcap.core.subprocess_runner import CancelToken
from vcap.models import llamacpp_backend
from vcap.models.llamacpp_backend import LlamaCppCaptioner
from vcap.models.overrides import (
    ModelOverride,
    classify_model_path,
    describe_override,
    override_identity,
    override_settings_problems,
    override_variant,
    resolve_audio_override,
    resolve_main_override,
    whisper_override_folder,
)
from vcap.models.registry import get_variant
from vcap.pipeline.job import InputItem, JobSpec, ModelChoice, OutputSpec, PostSpec
from vcap.pipeline.runner import (
    _apply_batch_skip,
    _assign_batch_outputs,
    _resolve_inputs,
    _sound_caption_model_choice,
    _validate_job_overrides,
    run_job,
)
from vcap.whisper.params import WhisperParams

from tests.test_dataset_clips import _fake_transcription


# --------------------------------------------------------------------------- fixtures


def _write_wav(path: Path, seconds: float = 0.4) -> None:
    rate = 8_000
    frames = bytearray()
    for index in range(int(rate * seconds)):
        value = int(1_200 * math.sin(index * 2 * math.pi * 440 / rate))
        frames.extend(value.to_bytes(2, "little", signed=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(frames)


def _gguf_folder(tmp_path: Path, *, mmproj_names: tuple[str, ...] = ("mmproj-Custom-Q8_0.gguf",)) -> Path:
    folder = tmp_path / "custom_gguf"
    folder.mkdir(parents=True)
    (folder / "Custom-Omni-Q3_K_M.gguf").write_bytes(b"g" * 4096)
    for name in mmproj_names:
        (folder / name).write_bytes(b"p" * 512)
    return folder


def _transformers_folder(tmp_path: Path, model_type: str = "qwen3_omni_moe", *, sharded: bool = False) -> Path:
    folder = tmp_path / f"hf_{model_type}{'_sharded' if sharded else ''}"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({"model_type": model_type}), encoding="utf-8")
    if sharded:
        (folder / "model-00001-of-00002.safetensors").write_bytes(b"s" * 64)
        (folder / "model-00002-of-00002.safetensors").write_bytes(b"s" * 64)
        (folder / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    else:
        (folder / "model.safetensors").write_bytes(b"s" * 128)
    return folder


def _ct2_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "whisper_ct2"
    folder.mkdir()
    (folder / "model.bin").write_bytes(b"w" * 256)
    (folder / "config.json").write_text("{}", encoding="utf-8")
    (folder / "vocabulary.json").write_text("[]", encoding="utf-8")
    return folder


def _split_settings(**overrides: Any) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "model_key": "qwen3_omni_instruct_int4",
        "prompt_preset_id": "custom",
        "user_prompt": "Describe this clip.",
        "system_prompt": None,
        "max_frames": 0,
        "use_audio_in_video": True,
        "output_formats": ["txt", "json", "srt"],
        "keep_model_loaded": False,
        "transcript_enabled": True,
        "transcript_formats": ["srt", "txt"],
        "transcript_inject_prompt": False,
        "whisper_device": "cpu",
        "whisper_compute_type": "int8",
        "audio_caption_source": "whisper",
        "video_caption_source": "generate",
        "audio_caption_model_key": "auto",
        "audio_caption_transcript_style": "plain",
        "audio_caption_template": DEFAULT_AUDIO_CAPTION_TEMPLATE,
        "caption_write_merged": True,
        "caption_merge_template": DEFAULT_CAPTION_MERGE_TEMPLATE,
        "audio_caption_empty_policy": "skip",
        "audio_caption_empty_text": "No speech.",
        "caption_txt_only": True,
    }
    settings.update(overrides)
    return settings


# --------------------------------------------------------------------------- job contract


def test_post_spec_txt_only_forces_plain_text_and_drops_reasoning() -> None:
    post = PostSpec(txt_only=True, formats=("txt", "json", "srt", "jsonl"), save_reasoning=True)
    assert post.txt_only is True
    assert post.formats == ("txt",)
    assert post.save_reasoning is False
    # ``replace`` re-runs validation, so a regenerate request cannot re-add formats.
    assert replace(post, formats=("txt", "json")).formats == ("txt",)
    plain = PostSpec(formats=("json",))
    assert plain.txt_only is False and plain.formats == ("txt", "json")


def test_job_spec_maps_txt_only_and_override_keys_and_round_trips(tmp_path: Path) -> None:
    spec = JobSpec.from_settings(
        {
            "model_key": "qwen3_omni_instruct_int4",
            "caption_txt_only": "true",
            "output_formats": ["txt", "json"],
            "save_reasoning": True,
            "override_video_model_path": '  "D:/models/omni-q3.gguf" ',
            "override_video_mmproj_path": "D:/models/mmproj.gguf",
            "override_audio_model_path": "D:/models/whisper-ct2",
            "override_audio_mmproj_path": "",
        },
        [],
        OutputSpec(outputs_root=tmp_path),
    )
    assert spec.post.txt_only is True
    assert spec.post.formats == ("txt",)
    assert spec.post.save_reasoning is False
    assert spec.model.override_model_path == "D:/models/omni-q3.gguf"
    assert spec.model.override_mmproj_path == "D:/models/mmproj.gguf"
    assert spec.model.override_audio_model_path == "D:/models/whisper-ct2"
    assert spec.model.has_override and spec.model.has_audio_override
    restored = JobSpec.from_dict(json.loads(spec.to_json()))
    assert restored.post.txt_only is True
    assert restored.model == spec.model
    plain = JobSpec.from_settings({"model_key": "qwen3_omni_instruct_int4"}, [], OutputSpec(outputs_root=tmp_path))
    assert plain.post.txt_only is False and not plain.model.has_override


# --------------------------------------------------------------------------- path classification


def test_classify_gguf_file_auto_detects_the_sibling_mmproj(tmp_path: Path) -> None:
    folder = _gguf_folder(tmp_path)
    override = classify_model_path(str(folder / "Custom-Omni-Q3_K_M.gguf"))
    assert override is not None
    assert override.kind == "gguf" and override.backend == "llamacpp"
    assert override.path == (folder / "Custom-Omni-Q3_K_M.gguf").resolve()
    assert override.mmproj == (folder / "mmproj-Custom-Q8_0.gguf").resolve()
    assert override.size_bytes == 4096
    assert override.notes == ()
    assert "Custom-Omni-Q3_K_M.gguf" in override.label and "mmproj-Custom-Q8_0.gguf" in override.label


def test_classify_gguf_folder_prefers_the_matching_projector(tmp_path: Path) -> None:
    folder = _gguf_folder(
        tmp_path,
        mmproj_names=("mmproj-Other-Q8_0.gguf", "Custom-Omni-mmproj-F16.gguf"),
    )
    override = classify_model_path(f'"{folder}"')
    assert override is not None
    assert override.path.name == "Custom-Omni-Q3_K_M.gguf"
    assert override.mmproj is not None and override.mmproj.name == "Custom-Omni-mmproj-F16.gguf"
    assert any("2 projector files" in note for note in override.notes)
    explicit = classify_model_path(str(folder), str(folder / "mmproj-Other-Q8_0.gguf"))
    assert explicit is not None and explicit.mmproj is not None
    assert explicit.mmproj.name == "mmproj-Other-Q8_0.gguf" and explicit.notes == ()


def test_classify_gguf_without_projector_notes_text_only_and_shards(tmp_path: Path) -> None:
    folder = _gguf_folder(tmp_path, mmproj_names=())
    override = classify_model_path(str(folder))
    assert override is not None and override.mmproj is None
    assert any("only text prompts" in note for note in override.notes)
    shards = tmp_path / "shards"
    shards.mkdir()
    for index in (2, 1, 3):
        (shards / f"Big-Q8_0-{index:05d}-of-00003.gguf").write_bytes(b"x")
    sharded = classify_model_path(str(shards))
    assert sharded is not None and sharded.path.name == "Big-Q8_0-00001-of-00003.gguf"
    ambiguous = tmp_path / "ambiguous"
    ambiguous.mkdir()
    (ambiguous / "a.gguf").write_bytes(b"x")
    (ambiguous / "b.gguf").write_bytes(b"x")
    with pytest.raises(ValueError, match="Several GGUF model files"):
        classify_model_path(str(ambiguous))


def test_classify_rejects_missing_paths_projectors_and_unknown_folders(tmp_path: Path) -> None:
    assert classify_model_path("") is None
    assert classify_model_path("   ") is None
    with pytest.raises(ValueError, match="does not exist"):
        classify_model_path(str(tmp_path / "missing.gguf"))
    folder = _gguf_folder(tmp_path)
    with pytest.raises(ValueError, match="multimodal projector"):
        classify_model_path(str(folder / "mmproj-Custom-Q8_0.gguf"))
    with pytest.raises(ValueError, match="mmproj file does not exist"):
        classify_model_path(str(folder / "Custom-Omni-Q3_K_M.gguf"), str(tmp_path / "nope.gguf"))
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="No model files were found"):
        classify_model_path(str(empty))
    (tmp_path / "weights.bin").write_bytes(b"x")
    with pytest.raises(ValueError, match="Unsupported model file"):
        classify_model_path(str(tmp_path / "weights.bin"))


def test_classify_transformers_and_ctranslate2_folders(tmp_path: Path) -> None:
    hf = _transformers_folder(tmp_path, "qwen3_omni_moe")
    override = classify_model_path(str(hf))
    assert override is not None and override.kind == "transformers"
    assert override.model_type == "qwen3_omni_moe" and override.size_bytes == 128
    assert not override.is_sharded_transformers
    by_file = classify_model_path(str(hf / "model.safetensors"))
    assert by_file is not None and by_file.path == hf.resolve()
    sharded = classify_model_path(str(_transformers_folder(tmp_path, "qwen2_5_omni", sharded=True)))
    assert sharded is not None and sharded.is_sharded_transformers
    ct2 = _ct2_folder(tmp_path)
    whisper = classify_model_path(str(ct2))
    assert whisper is not None and whisper.kind == "ctranslate2" and whisper.backend == "ctranslate2"
    assert whisper_override_folder(str(ct2)) == ct2.resolve()
    assert whisper_override_folder(str(hf)) is None
    assert whisper_override_folder(str(tmp_path / "missing")) is None


# --------------------------------------------------------------------------- family compatibility


def test_main_override_family_rules(tmp_path: Path) -> None:
    gguf = _gguf_folder(tmp_path) / "Custom-Omni-Q3_K_M.gguf"
    assert resolve_main_override("qwen3_omni_instruct_int4", str(gguf)) is not None
    assert resolve_main_override("qwen3_omni_captioner_gguf_q4", str(gguf)) is not None
    with pytest.raises(ValueError, match="Qwen3-Omni"):
        resolve_main_override("timechat_int4", str(gguf))
    hf_omni = _transformers_folder(tmp_path, "qwen3_omni_moe")
    assert resolve_main_override("qwen3_omni_instruct_bf16", str(hf_omni)) is not None
    with pytest.raises(ValueError, match="qwen3_omni_moe"):
        resolve_main_override("timechat_int4", str(hf_omni))
    hf_vl = _transformers_folder(tmp_path, "qwen2_5_vl")
    with pytest.raises(ValueError, match="GGUF conversion"):
        resolve_main_override("qwen3_omni_instruct_int4", str(hf_vl))
    ct2 = _ct2_folder(tmp_path)
    with pytest.raises(ValueError, match="audio caption model override"):
        resolve_main_override("qwen3_omni_instruct_int4", str(ct2))
    assert resolve_main_override("qwen3_omni_instruct_int4", "") is None


def test_audio_override_targets_captioner_or_whisper(tmp_path: Path) -> None:
    ct2 = _ct2_folder(tmp_path)
    assert resolve_audio_override("qwen3_omni_captioner_int4", str(ct2)) is None
    gguf = _gguf_folder(tmp_path) / "Custom-Omni-Q3_K_M.gguf"
    override = resolve_audio_override("qwen3_omni_captioner_int4", str(gguf))
    assert override is not None and override.kind == "gguf"
    with pytest.raises(ValueError):
        resolve_audio_override("qwen3_omni_captioner_int4", str(_transformers_folder(tmp_path, "whisper")))


def test_override_variant_keeps_key_and_switches_backend(tmp_path: Path) -> None:
    base = get_variant("qwen3_omni_instruct_int4")
    gguf = classify_model_path(str(_gguf_folder(tmp_path) / "Custom-Omni-Q3_K_M.gguf"))
    assert gguf is not None
    synthetic = override_variant(base, gguf)
    assert synthetic.key == base.key
    assert synthetic.backend == "llamacpp" and synthetic.scheme == "gguf"
    assert synthetic.gguf_files == ("Custom-Omni-Q3_K_M.gguf", "mmproj-Custom-Q8_0.gguf")
    assert synthetic.gguf_repo is None and synthetic.gguf_sha256 is None
    assert "Custom-Omni-Q3_K_M.gguf" in synthetic.label
    hf = classify_model_path(str(_transformers_folder(tmp_path, "qwen3_omni_moe", sharded=True)))
    assert hf is not None
    from_gguf_base = override_variant(get_variant("qwen3_omni_instruct_gguf_q4"), hf)
    assert from_gguf_base.backend == "transformers" and from_gguf_base.scheme == "bf16"
    single = classify_model_path(str(_transformers_folder(tmp_path, "qwen3_omni_moe")))
    assert single is not None
    assert override_variant(base, single).scheme == base.scheme
    ct2 = classify_model_path(str(_ct2_folder(tmp_path)))
    assert ct2 is not None
    with pytest.raises(ValueError):
        override_variant(base, ct2)
    assert override_identity(' "D:/a.gguf" ', "") == override_identity("D:/a.gguf", "")


def test_describe_override_and_settings_problems(tmp_path: Path) -> None:
    gguf = _gguf_folder(tmp_path) / "Custom-Omni-Q3_K_M.gguf"
    ct2 = _ct2_folder(tmp_path)
    assert describe_override("main", "", "", "qwen3_omni_instruct_int4")["state"] == "off"
    ok = describe_override("main", str(gguf), "", "qwen3_omni_instruct_int4")
    assert ok["state"] == "ok" and "Custom-Omni-Q3_K_M.gguf" in ok["text"]
    error = describe_override("main", str(gguf), "", "timechat_int4")
    assert error["state"] == "error" and "Qwen3-Omni" in error["text"]
    missing = describe_override("main", str(tmp_path / "nope.gguf"), "", "qwen3_omni_instruct_int4")
    assert missing["state"] == "error" and "does not exist" in missing["text"]
    unused_captioner = describe_override("audio", str(gguf), "", "qwen3_omni_instruct_int4", audio_source="whisper")
    assert unused_captioner["state"] == "warn" and any("unused" in note for note in unused_captioner["notes"])
    used_captioner = describe_override("audio", str(gguf), "", "qwen3_omni_instruct_int4", audio_source="captioner")
    assert used_captioner["state"] == "ok"
    whisper = describe_override("audio", str(ct2), "", "qwen3_omni_instruct_int4", audio_source="whisper")
    assert whisper["state"] == "ok" and "Whisper" in whisper["text"]
    whisper_unused = describe_override("audio", str(ct2), "", "qwen3_omni_instruct_int4", audio_source="captioner")
    assert whisper_unused["state"] == "warn"
    assert override_settings_problems({"model_key": "qwen3_omni_instruct_int4"}) == []
    problems = override_settings_problems(
        {
            "model_key": "timechat_int4",
            "override_video_model_path": str(gguf),
            "override_audio_model_path": str(tmp_path / "missing"),
        }
    )
    assert len(problems) == 2
    assert problems[0].startswith("Video / main model override:")
    assert problems[1].startswith("Audio caption model override:") and "does not exist" in problems[1]


# --------------------------------------------------------------------------- llama.cpp backend


def _override_backend(tmp_path: Path, *, mmproj: bool = True) -> tuple[LlamaCppCaptioner, ModelOverride]:
    folder = _gguf_folder(tmp_path, mmproj_names=("mmproj-Custom-Q8_0.gguf",) if mmproj else ())
    override = classify_model_path(str(folder / "Custom-Omni-Q3_K_M.gguf"))
    assert override is not None
    base = get_variant("qwen3_omni_instruct_int4")
    backend = LlamaCppCaptioner(
        "qwen3_omni_instruct",
        variant=override_variant(base, override),
        server_path=tmp_path / "llama-server.exe",
        model_path=override.path,
        mmproj_path=override.mmproj,
    )
    return backend, override


def test_llamacpp_server_command_uses_override_files(tmp_path: Path) -> None:
    backend, override = _override_backend(tmp_path)
    assert backend.variant.key == "qwen3_omni_instruct_int4"
    assert backend.model_dir == override.path.parent
    assert backend.checkpoint_files() == (override.path, override.mmproj)
    command = backend._server_command(tmp_path / "llama-server.exe", 8080)
    assert command[command.index("--model") + 1] == str(override.path)
    assert command[command.index("--mmproj") + 1] == str(override.mmproj)
    assert backend.supports_audio and backend.supports_vision


def test_llamacpp_text_only_override_omits_mmproj_and_refuses_media(tmp_path: Path) -> None:
    backend, override = _override_backend(tmp_path, mmproj=False)
    assert override.mmproj is None
    command = backend._server_command(tmp_path / "llama-server.exe", 8080)
    assert "--mmproj" not in command
    assert backend.supports_audio is False and backend.supports_vision is False
    (tmp_path / "frame.png").write_bytes(b"not really an image")
    from vcap.models.base import MediaPart, PreprocessParams

    with pytest.raises(ValueError, match="no vision projector"):
        backend._media_content([MediaPart("image", str(tmp_path / "frame.png"))], PreprocessParams(), None)
    with pytest.raises(ValueError, match="no audio encoder"):
        backend._media_content([MediaPart("audio", str(tmp_path / "frame.png"))], PreprocessParams(), None)


def test_llamacpp_start_reports_missing_override_files_before_launch(tmp_path: Path) -> None:
    backend, override = _override_backend(tmp_path)
    override.mmproj.unlink()
    with pytest.raises(FileNotFoundError, match="Custom GGUF override file"):
        backend.start()
    assert not backend.is_running


def test_llamacpp_modalities_gate_audio_and_strip_helper() -> None:
    messages = [
        {"role": "system", "content": "be brief"},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AA=="}},
                {"type": "input_audio", "input_audio": {"data": "AA==", "format": "wav"}},
                {"type": "text", "text": "Describe"},
            ],
        },
    ]
    stripped, removed = llamacpp_backend._strip_audio_parts(messages)
    assert removed == 1
    assert [part["type"] for part in stripped[1]["content"]] == ["image_url", "text"]
    assert stripped[0] == messages[0]
    assert llamacpp_backend._looks_like_audio_rejection(RuntimeError("llama-server HTTP 400: audio input is not supported - hint"))
    assert not llamacpp_backend._looks_like_audio_rejection(RuntimeError("llama-server HTTP 500: context shift"))
    backend = LlamaCppCaptioner("qwen3_omni_instruct", variant_key="qwen3_omni_instruct_gguf_q4", model_dir=Path("."))
    assert backend.supports_audio and backend.supports_vision
    backend.server_modalities = {"vision": True, "audio": False}
    assert backend.supports_vision and not backend.supports_audio


# --------------------------------------------------------------------------- loader cache


class _FakeLoaded:
    def __init__(self, variant_key: str, override: Any) -> None:
        self.variant = SimpleNamespace(key=variant_key, backend="transformers")
        self.model = object()
        self.processor = None
        self.load_report = SimpleNamespace(peak_vram_gb=0.0)
        self.override = override.to_dict() if override is not None else None


def test_model_cache_reloads_when_the_override_changes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from vcap.models import loader

    loads: list[tuple[str, Any]] = []

    def fake_load(variant_key: str, **kwargs: Any) -> _FakeLoaded:
        loads.append((variant_key, kwargs.get("override")))
        return _FakeLoaded(variant_key, kwargs.get("override"))

    monkeypatch.setattr(loader, "load_model", fake_load)
    monkeypatch.setattr(loader, "unload_model", lambda loaded, **_: SimpleNamespace(released=True))
    cache = loader.ModelCache()
    folder = _gguf_folder(tmp_path)
    first = classify_model_path(str(folder / "Custom-Omni-Q3_K_M.gguf"))
    plain = cache.load("qwen3_omni_instruct_int4")
    assert cache.load("qwen3_omni_instruct_int4") is plain
    with_override = cache.load("qwen3_omni_instruct_int4", override=first)
    assert with_override is not plain
    assert cache.load("qwen3_omni_instruct_int4", override=first) is with_override
    assert cache.loaded_variant_key() == "qwen3_omni_instruct_int4"
    other_folder = _gguf_folder(tmp_path / "other")
    second = classify_model_path(str(other_folder / "Custom-Omni-Q3_K_M.gguf"))
    assert cache.load("qwen3_omni_instruct_int4", override=second) is not with_override
    assert len(loads) == 3
    assert cache.unload(unless_variant="qwen3_omni_instruct_int4") is None
    assert cache.unload() is not None


def test_loader_rejects_a_whisper_folder_as_main_model(tmp_path: Path) -> None:
    from vcap.models import loader

    ct2 = classify_model_path(str(_ct2_folder(tmp_path)))
    with pytest.raises(ValueError, match="cannot replace"):
        loader.load_model("qwen3_omni_instruct_int4", device="cpu", override=ct2)


# --------------------------------------------------------------------------- whisper override


def test_whisper_params_pick_up_a_ctranslate2_audio_override(tmp_path: Path) -> None:
    ct2 = _ct2_folder(tmp_path)
    params = WhisperParams.from_settings({"whisper_model": "large-v3", "override_audio_model_path": str(ct2)})
    assert params.model == "large-v3"
    assert params.model_path == str(ct2.resolve())
    assert WhisperParams.from_dict(params.to_dict()).model_path == params.model_path
    gguf = _gguf_folder(tmp_path) / "Custom-Omni-Q3_K_M.gguf"
    assert WhisperParams.from_settings({"override_audio_model_path": str(gguf)}).model_path == ""
    assert WhisperParams.from_settings({"whisper_model_path": "D:/explicit"}).model_path == "D:/explicit"
    assert WhisperParams.from_settings({}).model_path == ""


def test_whisper_engine_uses_the_custom_folder_without_downloading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vcap.whisper import engine as engine_module

    monkeypatch.setattr(
        engine_module,
        "download_model",
        lambda *args, **kwargs: pytest.fail("download_model must not run for a custom folder"),
    )
    ct2 = _ct2_folder(tmp_path)
    messages: list[str] = []
    engine = engine_module.WhisperEngine(
        WhisperParams(model="large-v1", model_path=str(ct2)),
        models_dir=tmp_path / "models",
        log=lambda message, level="info": messages.append(message),
    )
    assert engine.ensure_model() == ct2.resolve()
    assert any("custom Whisper model folder" in message for message in messages)
    broken = engine_module.WhisperEngine(
        WhisperParams(model="large-v1", model_path=str(tmp_path / "missing")),
        models_dir=tmp_path / "models",
    )
    with pytest.raises(FileNotFoundError, match="does not exist"):
        broken.ensure_model()


# --------------------------------------------------------------------------- runner integration


def test_sound_caption_model_choice_applies_only_captioner_overrides(tmp_path: Path) -> None:
    gguf = _gguf_folder(tmp_path) / "Custom-Omni-Q3_K_M.gguf"
    spec = JobSpec.from_settings(
        _split_settings(
            audio_caption_source="captioner",
            override_video_model_path=str(gguf),
            override_audio_model_path=str(gguf),
        ),
        [],
        OutputSpec(outputs_root=tmp_path),
    )
    choice = _sound_caption_model_choice(spec, "qwen3_omni_captioner_int4")
    assert choice.variant_key == "qwen3_omni_captioner_int4"
    assert choice.override_model_path == str(gguf.resolve())
    assert choice.override_mmproj_path.endswith("mmproj-Custom-Q8_0.gguf")
    assert choice.override_audio_model_path == ""
    ct2 = _ct2_folder(tmp_path)
    whisper_spec = replace(spec, model=replace(spec.model, override_audio_model_path=str(ct2)))
    assert _sound_caption_model_choice(whisper_spec, "qwen3_omni_captioner_int4").override_model_path == ""
    notes = _validate_job_overrides(whisper_spec, loads_main_model=True)
    assert any("Main caption model override" in note for note in notes)
    assert any("Whisper" in note for note in notes)
    with pytest.raises(ValueError, match="does not exist"):
        _validate_job_overrides(
            replace(spec, model=replace(spec.model, override_model_path=str(tmp_path / "gone.gguf"))),
            loads_main_model=True,
        )
    # Existing-caption runs never load the main model, so its override is not checked.
    ignored = replace(spec, model=replace(spec.model, override_model_path=str(tmp_path / "gone.gguf")))
    assert all("Main caption model" not in note for note in _validate_job_overrides(ignored, loads_main_model=False))


def test_run_job_fails_fast_on_a_bad_override_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "clip.wav"
    _write_wav(source)
    monkeypatch.setenv("VCAP_FAKE_CAPTIONER", "1")
    spec = JobSpec.from_settings(
        _split_settings(
            audio_caption_source="none",
            caption_txt_only=False,
            transcript_enabled=False,
            override_video_model_path=str(tmp_path / "missing-model.gguf"),
        ),
        [InputItem(source)],
        OutputSpec(outputs_root=tmp_path / "outputs"),
    )
    with pytest.raises(ValueError, match="does not exist"):
        run_job(spec, None, CancelToken())
    assert not (tmp_path / "outputs").exists() or not any((tmp_path / "outputs").iterdir())


def test_fake_session_reloads_when_override_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vcap.pipeline import runner

    monkeypatch.setenv("VCAP_FAKE_CAPTIONER", "1")
    monkeypatch.setattr(runner, "_FAKE_CAPTIONER", None)
    monkeypatch.setattr(runner, "_FAKE_VARIANT", None)
    monkeypatch.setattr(runner, "_FAKE_OVERRIDE", ("", ""))
    gguf = _gguf_folder(tmp_path) / "Custom-Omni-Q3_K_M.gguf"
    logs: list[str] = []

    class _Sink:
        def on_log(self, message: str, level: str = "info", scope: str | None = None) -> None:
            logs.append(message)

        def on_progress(self, event: Any) -> None:
            pass

        def on_item(self, event: Any) -> None:
            pass

    from vcap.core.progress import ProgressTracker

    emitter = runner._Emitter(_Sink(), ProgressTracker(1, ["x"]))
    plain = JobSpec.from_settings({"model_key": "qwen3_omni_instruct_int4"}, [], OutputSpec(outputs_root=tmp_path))
    first = runner._ModelSession(plain, emitter, CancelToken()).ensure()
    with_override = JobSpec.from_settings(
        {"model_key": "qwen3_omni_instruct_int4", "override_video_model_path": str(gguf)},
        [],
        OutputSpec(outputs_root=tmp_path),
    )
    second = runner._ModelSession(with_override, emitter, CancelToken()).ensure()
    assert second is not first
    assert runner._ModelSession(with_override, emitter, CancelToken()).ensure() is second
    assert any("override: Custom-Omni-Q3_K_M.gguf" in message for message in logs)


def test_txt_only_single_run_writes_exactly_one_caption_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "clip ünicode.wav"
    _write_wav(source)
    monkeypatch.setenv("VCAP_FAKE_CAPTIONER", "1")
    monkeypatch.setattr("vcap.whisper.client.run_transcription", _fake_transcription)
    spec = JobSpec.from_settings(
        _split_settings(audio_caption_source="none", summarize_segments=True, save_reasoning=True),
        [InputItem(source)],
        OutputSpec(outputs_root=tmp_path / "outputs"),
    )
    assert spec.post.txt_only and spec.post.formats == ("txt",)
    result = run_job(spec, None, CancelToken())
    assert result.counts["done"] == 1
    run_dir = Path(result.run_dir)
    produced = sorted(path.name for path in run_dir.iterdir() if path.name != ".work")
    assert produced == ["clip ünicode.txt", "metadata.json", "run_log.txt"]
    assert "json" not in result.items[0].outputs
    assert not any(key.startswith("transcript_") for key in result.items[0].outputs)
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["settings"]["caption_txt_only"] is True
    assert "override" not in metadata["model_info"]


def test_txt_only_split_layout_writes_only_the_merged_caption(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "\u043a\u043b\u0438\u043f.wav"
    _write_wav(source)
    monkeypatch.setenv("VCAP_FAKE_CAPTIONER", "1")
    monkeypatch.setattr("vcap.whisper.client.run_transcription", _fake_transcription)
    spec = JobSpec.from_settings(
        _split_settings(caption_prefix="prefix ", caption_write_merged=False),
        [InputItem(source)],
        OutputSpec(outputs_root=tmp_path / "outputs"),
    )
    result = run_job(spec, None, CancelToken())
    assert result.counts["done"] == 1
    assert result.counts["audio_captions"] == 1
    item = result.items[0]
    run_dir = Path(result.run_dir)
    assert sorted(path.name for path in run_dir.iterdir() if path.name != ".work") == ["metadata.json", "run_log.txt", "клип.txt"]
    assert not (run_dir / "video_caption").exists() and not (run_dir / "audio_caption").exists()
    assert not list(run_dir.glob("*_transcript.*"))
    merged = Path(str(item.merged_caption_path))
    assert merged.parent == run_dir
    text = merged.read_text(encoding="utf-8")
    assert text.startswith("prefix ") and text.endswith("\n\nhello world\n")
    assert item.video_caption_path is None and item.audio_caption_path is None
    assert item.outputs["txt"] == str(merged)
    assert "json" not in item.outputs
    assert item.segments[0]["audio_caption"] == "hello world"


def test_txt_only_batch_skip_targets_the_merged_caption(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    clip = root / "clip.wav"
    _write_wav(clip)
    spec = JobSpec.from_settings(
        _split_settings(batch_save_next_to_source=True, caption_write_merged=False),
        [InputItem(clip)],
        OutputSpec(kind="batch", outputs_root=tmp_path / "outputs", source_root=root, save_next_to_source=True),
    )
    resolved = _resolve_inputs(spec)
    _assign_batch_outputs(spec, resolved)
    paths = caption_unit_paths(resolved[0].out_dir, resolved[0].stem)
    paths.video.parent.mkdir(parents=True, exist_ok=True)
    paths.video.write_text("clean\n", encoding="utf-8")
    _apply_batch_skip(spec, resolved)
    assert resolved[0].status == "pending"
    paths.merged.write_text("merged\n", encoding="utf-8")
    _apply_batch_skip(spec, resolved)
    assert resolved[0].status == "skipped"
    assert "clip.txt" in resolved[0].message


# --------------------------------------------------------------------------- UI


def test_build_app_registers_txt_only_and_override_controls(tmp_path: Path) -> None:
    from vcap.ui.app import build_app
    from vcap.ui.tabs.caption_tab import (
        override_status_html,
        txt_only_hint_html,
        whisper_override_note_html,
    )

    demo = build_app()
    registry = demo.settings_registry
    entries = {entry.key: entry for entry in registry.entries()}
    for key in (
        "caption_txt_only",
        "override_video_model_path",
        "override_video_mmproj_path",
        "override_audio_model_path",
        "override_audio_mmproj_path",
    ):
        assert key in entries and entries[key].in_preset and entries[key].in_metadata
    assert entries["caption_txt_only"].kind == "bool"
    context = demo.vcap_context
    assert context.caption_handles.media.txt_only_mirror is not None
    assert context.caption_handles.controls["caption_txt_only"] is not context.caption_handles.media.txt_only_mirror
    assert context.states["whisper_override_note"] is not None
    coerced, warnings = registry.coerce({"caption_txt_only": "yes", "override_video_model_path": None})
    assert coerced["caption_txt_only"] is True and coerced["override_video_model_path"] == ""
    assert not [warning for warning in warnings if "override" in warning or "txt_only" in warning]

    assert "exactly one" in txt_only_hint_html(True)
    assert "Off:" in txt_only_hint_html(False)
    gguf = _gguf_folder(tmp_path) / "Custom-Omni-Q3_K_M.gguf"
    status = override_status_html(str(gguf), "", "", "", "qwen3_omni_instruct_int4", "none")
    assert "vc-ok" in status and "Custom-Omni-Q3_K_M.gguf" in status
    assert "vc-err" in override_status_html(str(gguf), "", "", "", "timechat_int4", "none")
    assert "No override set" in override_status_html("", "", "", "", "qwen3_omni_instruct_int4", "none")
    ct2 = _ct2_folder(tmp_path)
    note = whisper_override_note_html(str(ct2))
    assert note["visible"] is True and "Custom Whisper model folder" in note["value"]
    assert whisper_override_note_html("")["visible"] is False
    assert "is not a faster-whisper folder" in whisper_override_note_html(str(gguf))["value"]


# --------------------------------------------------------------------------- v1.9.1 follow-ups


def test_frames_note_mentions_audio_only_when_it_was_sent() -> None:
    with_audio = llamacpp_backend._frames_note(11, True)
    without_audio = llamacpp_backend._frames_note(11, False)
    assert "11 chronological still frames plus separate audio" in with_audio
    assert "not interleaved" in with_audio
    assert "11 chronological still frames" in without_audio
    assert "no audio input" in without_audio and "plus separate audio" not in without_audio


def test_whisper_loaded_label_names_the_custom_folder(tmp_path: Path) -> None:
    from vcap.whisper import client as client_module
    from vcap.whisper import engine as engine_module

    ct2 = _ct2_folder(tmp_path)
    custom = engine_module.WhisperEngine(
        WhisperParams(model="large-v1", model_path=str(ct2)),
        models_dir=tmp_path / "models",
    )
    label = custom.loaded_model_label(ct2)
    assert label.startswith("whisper_ct2 (custom folder ") and "replaces large-v1" in label
    assert custom.loaded_model_label() == label
    regular = engine_module.WhisperEngine(WhisperParams(model="large-v1"), models_dir=tmp_path / "models")
    assert regular.loaded_model_label(tmp_path / "models" / "large-v1") == "large-v1"

    logged: list[tuple[str, str]] = []

    class _Sink:
        def on_log(self, message: str, level: str = "info") -> None:
            logged.append((message, level))

    client_module._handle_event(
        {"event": "model_loaded", "model": "large-v1", "label": label, "load_s": 2.5},
        _Sink(),
        [],
        {},
    )
    assert logged and logged[-1][0] == f"Whisper model {label} loaded in 2.5s"
    client_module._handle_event({"event": "model_loaded", "model": "large-v1", "load_s": 1.0}, _Sink(), [], {})
    assert logged[-1][0] == "Whisper model large-v1 loaded in 1.0s"


def test_chat_model_note_reflects_the_main_override(tmp_path: Path) -> None:
    from vcap.ui.tabs.chat_tab import (
        chat_model_change_updates,
        chat_override_note_update,
        model_chat_support,
    )

    mode, base = model_chat_support("qwen3_omni_instruct_int4")
    assert mode == "multi" and "override" not in base and "video, audio, image, and text" in base
    gguf = _gguf_folder(tmp_path) / "Custom-Omni-Q3_K_M.gguf"
    mode, note = model_chat_support("qwen3_omni_instruct_int4", str(gguf), "")
    assert mode == "multi" and "vc-ok" in note and "Custom-Omni-Q3_K_M.gguf" in note
    assert "follows the custom model's mmproj" in note
    assert chat_override_note_update("qwen3_omni_instruct_int4", str(gguf), "") == note
    assert chat_model_change_updates("qwen3_omni_instruct_int4", None, str(gguf), "")[0] == note
    assert not isinstance(chat_override_note_update("not-a-variant", str(gguf), ""), str)

    text_only_dir = tmp_path / "text_only"
    text_only_dir.mkdir()
    text_only = text_only_dir / "Text-Only-Q4.gguf"
    text_only.write_bytes(b"g" * 64)
    mode, note = model_chat_support("qwen3_omni_instruct_int4", str(text_only), "")
    assert mode == "multi" and "vc-warn" in note and "text-only" in note

    mode, note = model_chat_support("qwen3_omni_instruct_int4", str(tmp_path / "missing.gguf"), "")
    assert mode == "blocked" and "vc-err" in note and "does not exist" in note
    status = chat_model_change_updates("qwen3_omni_instruct_int4", None, str(tmp_path / "missing.gguf"), "")[7]
    assert status == note
    mode, note = model_chat_support("timechat_int4", str(gguf), "")
    assert mode == "blocked" and "Qwen3-Omni" in note
    hf = _transformers_folder(tmp_path)
    mode, note = model_chat_support("qwen3_omni_instruct_int4", str(hf), "")
    assert mode == "multi" and "Transformers checkpoint" in note and "video, audio, image, and text" in note
    assert model_chat_support("qwen3_omni_captioner_int4", str(gguf), "")[0] == "unsupported"
