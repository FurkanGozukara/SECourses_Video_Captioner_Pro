from __future__ import annotations

import json
from pathlib import Path

import pytest

from vcap.core.presets import PresetStore
from vcap.ui.app import build_app
from vcap.whisper.models import get_model


PRESET_NAMES = (
    "Transcribe - Whisper best quality (large-v1)",
    "Transcribe - Whisper large-v3 turbo (fast)",
    "Caption + Whisper transcript (Qwen3-Omni Instruct)",
)


@pytest.fixture(scope="module")
def registry_defaults() -> tuple[dict, list[str]]:
    app = build_app()
    try:
        return app.settings_registry.defaults(), app.settings_registry.keys()
    finally:
        app.vcap_context.pipeline.shutdown()


def test_shipped_whisper_presets_contain_every_registered_key(
    registry_defaults: tuple[dict, list[str]],
) -> None:
    _defaults, keys = registry_defaults
    root = Path("presets_default")
    for name in PRESET_NAMES:
        payload = json.loads((root / f"{name}.json").read_text(encoding="utf-8"))
        assert payload["_meta"]["format"] == "secourses_vcap_preset"
        assert set(payload["settings"]) == set(keys)


def test_whisper_preset_values_match_large_v1_and_turbo_defaults() -> None:
    root = Path("presets_default")
    best = json.loads(
        (root / "Transcribe - Whisper best quality (large-v1).json").read_text(encoding="utf-8")
    )["settings"]
    turbo = json.loads(
        (root / "Transcribe - Whisper large-v3 turbo (fast).json").read_text(encoding="utf-8")
    )["settings"]
    assert best["whisper_model"] == "large-v1"
    assert best["whisper_compute_type"] == "float16"
    assert best["whisper_beam_size"] == best["whisper_best_of"] == 5
    assert best["whisper_word_timestamps"] is True
    assert best["whisper_normalize_word_timestamps"] is True
    assert best["whisper_vad_filter"] is False
    assert best["whisper_formats"] == ["srt", "vtt", "txt", "lrc", "tsv", "json"]
    assert turbo["whisper_model"] == "large-v3-turbo"
    assert {key: value for key, value in turbo.items() if key != "whisper_model"} == {
        key: value for key, value in best.items() if key != "whisper_model"
    }
    assert get_model("large-v1").note == "Best-quality large-v1 default."


def test_caption_whisper_preset_enables_prompt_injection() -> None:
    settings = json.loads(
        Path("presets_default/Caption + Whisper transcript (Qwen3-Omni Instruct).json").read_text(
            encoding="utf-8"
        )
    )["settings"]
    assert settings["transcript_enabled"] is True
    assert settings["transcript_inject_prompt"] is True
    assert settings["model_key"].startswith("qwen3_omni_instruct")


def test_new_keys_round_trip_through_store_and_registry(
    tmp_path: Path,
    registry_defaults: tuple[dict, list[str]],
) -> None:
    defaults, _keys = registry_defaults
    store = PresetStore(tmp_path / "user", Path("presets_default"))
    values = {
        **defaults,
        "whisper_model": "small.en",
        "whisper_language": "english",
        "whisper_temperature": 0.25,
        "whisper_formats": ["txt", "json"],
        "transcript_enabled": True,
        "transcript_formats": ["vtt", "txt"],
        "transcript_file_suffix": "_speech",
    }
    saved = store.save("Whisper round trip", values)
    loaded = store.load(saved, mark_last_used=False)

    assert loaded == values
