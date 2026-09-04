from __future__ import annotations

from types import SimpleNamespace

import pytest

from vcap.whisper.engine import (
    TranscriptResult,
    TranscriptSegment,
    TranscriptWord,
    resolve_prompt_safe_params,
    text_between,
)
from vcap.whisper.params import (
    LANGUAGE_AUTO,
    LANGUAGE_CHOICES,
    WHISPER_LANGUAGES,
    WHISPER_LANGUAGE_ALIASES,
    TranscriptOutputOptions,
    WhisperParams,
    code_to_language,
    language_to_code,
    parse_suppress_tokens,
)


def test_params_defaults_and_round_trip() -> None:
    params = WhisperParams()
    restored = WhisperParams.from_dict(params.to_dict())

    assert restored == params
    assert params.model == "large-v1"
    assert params.compute_type == "float16"
    assert params.vad.max_speech_duration_s == 9999.0


def test_params_coerce_and_clamp() -> None:
    params = WhisperParams(
        device="bogus",
        compute_type="bogus",
        gpu_index=-5,
        cpu_threads=-1,
        beam_size=99,
        best_of=0,
        temperature=2,
        no_speech_threshold=-2,
        prompt_reset_on_temperature=4,
        chunk_length=0,
        batch_size=1000,
        suppress_tokens="bad",
        vad={"threshold": 4, "max_speech_duration_s": 10_000},
    )

    assert params.device == "auto"
    assert params.compute_type == "float16"
    assert params.gpu_index == 0
    assert params.cpu_threads == 0
    assert params.beam_size == 20
    assert params.best_of == 1
    assert params.temperature == 1.0
    assert params.no_speech_threshold == 0.0
    assert params.prompt_reset_on_temperature == 1.0
    assert params.chunk_length == 1
    assert params.batch_size == 64
    assert params.suppress_tokens == "[-1]"
    assert params.vad.threshold == 1.0
    assert params.vad.max_speech_duration_s == 9999.0


def test_from_settings_maps_registry_keys_and_keeps_missing_defaults() -> None:
    params = WhisperParams.from_settings(
        {
            "whisper_model": "tiny",
            "whisper_language": "tr",
            "whisper_translate": "true",
            "whisper_repeat_initial_prompt": 1,
            "whisper_use_batched_inference": True,
            "whisper_vad_filter": True,
            "whisper_vad_min_speech_ms": "125",
        }
    )

    assert params.model == "tiny"
    assert params.language == "turkish"
    assert params.translate_to_english is True
    assert params.repeat_initial_prompt_every_window is True
    assert params.use_batched_inference is True
    assert params.vad.enabled is True
    assert params.vad.min_speech_duration_ms == 125
    assert params.beam_size == 5


def test_language_table_round_trips_and_aliases() -> None:
    assert LANGUAGE_CHOICES[0] == LANGUAGE_AUTO
    assert LANGUAGE_CHOICES[1:] == sorted(LANGUAGE_CHOICES[1:])
    assert len(WHISPER_LANGUAGES) == 99
    assert len(LANGUAGE_CHOICES) == 100
    for code, language in WHISPER_LANGUAGES.items():
        assert language_to_code(language) == code
        assert language_to_code(code) == code
        assert code_to_language(code) == language
    for alias, code in WHISPER_LANGUAGE_ALIASES.items():
        assert language_to_code(alias) == code
    assert language_to_code(None) is None
    assert language_to_code("") is None
    assert language_to_code(LANGUAGE_AUTO) is None
    assert code_to_language(None) == LANGUAGE_AUTO


def test_suppress_tokens_parser_warns_and_falls_back() -> None:
    warnings: list[str] = []
    assert parse_suppress_tokens("[1, -1, 42]", warning=warnings.append) == [1, -1, 42]
    assert parse_suppress_tokens("{'bad': 1}", warning=warnings.append) == [-1]
    assert parse_suppress_tokens([True], warning=warnings.append) == [-1]
    assert len(warnings) == 2


def test_prompt_safe_max_new_tokens_rules() -> None:
    assert resolve_prompt_safe_params(WhisperParams(max_new_tokens=0)).max_new_tokens is None
    assert (
        resolve_prompt_safe_params(
            WhisperParams(condition_on_previous_text=False, max_new_tokens=999)
        ).max_new_tokens
        == 432
    )
    assert (
        resolve_prompt_safe_params(
            WhisperParams(condition_on_previous_text=True, max_new_tokens=100)
        ).max_new_tokens
        is None
    )


def test_transcript_result_round_trip_and_text_between() -> None:
    result = TranscriptResult(
        segments=[
            TranscriptSegment(
                0,
                0.0,
                2.0,
                "Hello world.",
                [
                    TranscriptWord(0.0, 0.8, " Hello", 0.9),
                    TranscriptWord(0.8, 2.0, " world.", 0.8),
                ],
            ),
            TranscriptSegment(1, 2.0, 4.0, "Fallback text.", []),
        ],
        language="en",
        language_probability=0.99,
        duration_s=4.0,
        elapsed_s=1.5,
        model="tiny",
        compute_type="int8",
        device="cpu",
    )

    assert result.text == "Hello world. Fallback text."
    assert text_between(result, 0.7, 2.5) == "Hello world. Fallback text."
    assert TranscriptResult.from_dict(result.to_dict()) == result


def test_output_options_normalize_webvtt_and_duplicates() -> None:
    options = TranscriptOutputOptions(
        formats=("SRT", ".webvtt", "vtt", "unknown"),
        add_timestamp="yes",
        file_suffix=None,
    )
    assert options.formats == ("srt", "vtt")
    assert options.add_timestamp is True
    assert options.file_suffix == ""
    assert TranscriptOutputOptions.from_dict({}).formats == (
        "srt",
        "vtt",
        "txt",
        "lrc",
        "tsv",
        "json",
    )


@pytest.mark.skipif(
    __import__("os").environ.get("VCAP_WHISPER_GPU_TESTS") != "1",
    reason="real Whisper GPU tests are opt-in",
)
def test_gpu_runtime_opt_in_only() -> None:
    import ctranslate2

    from vcap.whisper.cuda_runtime import enable_cuda_runtime_autodiscovery

    enable_cuda_runtime_autodiscovery()
    assert ctranslate2.get_cuda_device_count() >= 1
