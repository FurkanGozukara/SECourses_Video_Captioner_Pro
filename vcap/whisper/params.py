"""Typed, JSON-safe parameters for the faster-whisper backend."""

from __future__ import annotations

import ast
import math
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Callable, Mapping

LANGUAGE_AUTO = "Automatic Detection"

# Kept in Whisper's canonical order so code/name round trips remain stable.
WHISPER_LANGUAGES: dict[str, str] = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
}

WHISPER_LANGUAGE_ALIASES: dict[str, str] = {
    "burmese": "my",
    "valencian": "ca",
    "flemish": "nl",
    "haitian": "ht",
    "letzeburgesch": "lb",
    "pushto": "ps",
    "panjabi": "pa",
    "moldavian": "ro",
    "moldovan": "ro",
    "sinhalese": "si",
    "castilian": "es",
    "mandarin": "zh",
}

_TO_LANGUAGE_CODE = {name: code for code, name in WHISPER_LANGUAGES.items()}
_TO_LANGUAGE_CODE.update(WHISPER_LANGUAGE_ALIASES)
LANGUAGE_CHOICES: list[str] = [LANGUAGE_AUTO, *sorted(WHISPER_LANGUAGES.values())]

COMPUTE_TYPE_CHOICES = [
    "float16",
    "bfloat16",
    "float32",
    "int8",
    "int8_float16",
    "int8_bfloat16",
]
DEVICE_CHOICES = ["auto", "cuda", "cpu"]


def language_to_code(label: str | None) -> str | None:
    """Translate a language name, alias, or code to its Whisper code."""

    if label is None:
        return None
    value = str(label).strip().casefold()
    if not value or value == LANGUAGE_AUTO.casefold() or value in {"auto", "automatic"}:
        return None
    if value in WHISPER_LANGUAGES:
        return value
    return _TO_LANGUAGE_CODE.get(value)


def code_to_language(code: str | None) -> str:
    """Translate a Whisper language code to its lower-case display name."""

    if code is None or not str(code).strip():
        return LANGUAGE_AUTO
    value = str(code).strip().casefold()
    resolved = language_to_code(value)
    if resolved is None:
        return LANGUAGE_AUTO if value in {"auto", "automatic"} else value
    return WHISPER_LANGUAGES[resolved]


def _bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"1", "true", "yes", "on", "enabled"}:
            return True
        if normalized in {"0", "false", "no", "off", "disabled", ""}:
            return False
        return default
    return bool(value)


def _integer(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(float(value))
    except (TypeError, ValueError, OverflowError):
        number = default
    return max(minimum, min(maximum, number))


def _number(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        number = default
    if not math.isfinite(number):
        number = maximum if number > 0 else minimum
    return max(minimum, min(maximum, number))


def _text(value: Any, default: str = "") -> str:
    return default if value is None else str(value)


def _warn_suppress_tokens(message: str) -> None:
    try:
        from vcap.core.logs import get_log

        get_log().warn(message, scope="whisper")
    except Exception:
        return


def parse_suppress_tokens(
    value: Any,
    *,
    warning: Callable[[str], None] | None = None,
) -> list[int]:
    """Parse the UI suppress-token representation, falling back to ``[-1]``."""

    parsed = value
    try:
        if isinstance(value, str):
            parsed = ast.literal_eval(value.strip())
        if not isinstance(parsed, (list, tuple)) or not all(
            isinstance(item, int) and not isinstance(item, bool) for item in parsed
        ):
            raise ValueError
        return [int(item) for item in parsed]
    except (SyntaxError, ValueError, TypeError):
        message = "Invalid Whisper suppress_tokens value; using [-1]."
        (warning or _warn_suppress_tokens)(message)
        return [-1]


@dataclass
class WhisperVadParams:
    enabled: bool = False
    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    max_speech_duration_s: float = 9999.0
    min_silence_duration_ms: int = 2000
    speech_pad_ms: int = 400

    def __post_init__(self) -> None:
        self.enabled = _bool(self.enabled, False)
        self.threshold = _number(self.threshold, 0.5, 0.0, 1.0)
        self.min_speech_duration_ms = _integer(
            self.min_speech_duration_ms, 250, 0, 60_000
        )
        self.max_speech_duration_s = _number(
            self.max_speech_duration_s, 9999.0, 0.001, 9999.0
        )
        self.min_silence_duration_ms = _integer(
            self.min_silence_duration_ms, 2000, 0, 60_000
        )
        self.speech_pad_ms = _integer(self.speech_pad_ms, 400, 0, 10_000)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "WhisperVadParams":
        source = data if isinstance(data, Mapping) else {}
        known = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in source.items() if key in known})


@dataclass
class WhisperParams:
    model: str = "large-v1"
    language: str = "english"
    translate_to_english: bool = False
    compute_type: str = "float16"
    device: str = "auto"
    gpu_index: int = 0
    cpu_threads: int = 0
    beam_size: int = 5
    best_of: int = 5
    patience: float = 1.0
    temperature: float = 0.0
    length_penalty: float = 1.0
    repetition_penalty: float = 1.2
    no_repeat_ngram_size: int = 0
    compression_ratio_threshold: float = 2.4
    log_prob_threshold: float = -1.0
    no_speech_threshold: float = 0.6
    condition_on_previous_text: bool = True
    prompt_reset_on_temperature: float = 0.5
    initial_prompt: str = ""
    repeat_initial_prompt_every_window: bool = False
    prefix: str = ""
    hotwords: str = ""
    suppress_blank: bool = True
    suppress_tokens: str = "[-1]"
    max_initial_timestamp: float = 1.0
    word_timestamps: bool = True
    normalize_word_timestamps: bool = True
    highlight_words: bool = False
    prepend_punctuations: str = "\"'([{-"
    append_punctuations: str = "\"'.,!?:)]}"
    max_new_tokens: int = 0
    chunk_length: int = 30
    hallucination_silence_threshold: float = 0.0
    language_detection_threshold: float = 0.5
    language_detection_segments: int = 1
    use_batched_inference: bool = False
    batch_size: int = 1
    vad: WhisperVadParams = field(default_factory=WhisperVadParams)

    def __post_init__(self) -> None:
        self.model = _text(self.model, "large-v1").strip() or "large-v1"
        original_language = _text(self.language)
        code = language_to_code(original_language)
        self.language = code_to_language(code) if code else LANGUAGE_AUTO
        if code is None and original_language.strip().casefold() not in {
            "",
            LANGUAGE_AUTO.casefold(),
            "auto",
            "automatic",
        }:
            self.language = "english"
        self.translate_to_english = _bool(self.translate_to_english, False)
        compute = _text(self.compute_type, "float16").strip().casefold()
        self.compute_type = compute if compute in COMPUTE_TYPE_CHOICES else "float16"
        device = _text(self.device, "auto").strip().casefold()
        self.device = device if device in DEVICE_CHOICES else "auto"
        self.gpu_index = _integer(self.gpu_index, 0, 0, 127)
        self.cpu_threads = _integer(self.cpu_threads, 0, 0, 1024)
        self.beam_size = _integer(self.beam_size, 5, 1, 20)
        self.best_of = _integer(self.best_of, 5, 1, 20)
        self.patience = _number(self.patience, 1.0, 0.01, 10.0)
        self.temperature = _number(self.temperature, 0.0, 0.0, 1.0)
        self.length_penalty = _number(self.length_penalty, 1.0, 0.01, 10.0)
        self.repetition_penalty = _number(self.repetition_penalty, 1.2, 0.01, 10.0)
        self.no_repeat_ngram_size = _integer(self.no_repeat_ngram_size, 0, 0, 100)
        self.compression_ratio_threshold = _number(
            self.compression_ratio_threshold, 2.4, 0.01, 100.0
        )
        self.log_prob_threshold = _number(self.log_prob_threshold, -1.0, -100.0, 0.0)
        self.no_speech_threshold = _number(self.no_speech_threshold, 0.6, 0.0, 1.0)
        self.condition_on_previous_text = _bool(self.condition_on_previous_text, True)
        self.prompt_reset_on_temperature = _number(
            self.prompt_reset_on_temperature, 0.5, 0.0, 1.0
        )
        self.initial_prompt = _text(self.initial_prompt)
        self.repeat_initial_prompt_every_window = _bool(
            self.repeat_initial_prompt_every_window, False
        )
        self.prefix = _text(self.prefix)
        self.hotwords = _text(self.hotwords)
        self.suppress_blank = _bool(self.suppress_blank, True)
        self.suppress_tokens = repr(parse_suppress_tokens(self.suppress_tokens))
        self.max_initial_timestamp = _number(
            self.max_initial_timestamp, 1.0, 0.0, 60.0
        )
        self.word_timestamps = _bool(self.word_timestamps, True)
        self.normalize_word_timestamps = _bool(self.normalize_word_timestamps, True)
        self.highlight_words = _bool(self.highlight_words, False)
        self.prepend_punctuations = _text(self.prepend_punctuations, "\"'([{-")
        self.append_punctuations = _text(self.append_punctuations, "\"'.,!?:)]}")
        self.max_new_tokens = _integer(self.max_new_tokens, 0, 0, 65_536)
        self.chunk_length = _integer(self.chunk_length, 30, 1, 30)
        self.hallucination_silence_threshold = _number(
            self.hallucination_silence_threshold, 0.0, 0.0, 60.0
        )
        self.language_detection_threshold = _number(
            self.language_detection_threshold, 0.5, 0.0, 1.0
        )
        self.language_detection_segments = _integer(
            self.language_detection_segments, 1, 1, 100
        )
        self.use_batched_inference = _bool(self.use_batched_inference, False)
        self.batch_size = _integer(self.batch_size, 1, 1, 64)
        if isinstance(self.vad, Mapping):
            self.vad = WhisperVadParams.from_dict(self.vad)
        elif not isinstance(self.vad, WhisperVadParams):
            self.vad = WhisperVadParams()

    @classmethod
    def from_settings(
        cls, settings: Mapping[str, Any], prefix: str = "whisper_"
    ) -> "WhisperParams":
        """Build parameters from the settings registry's flat key namespace."""

        source = settings if isinstance(settings, Mapping) else {}
        key_map = {
            "model": "model",
            "language": "language",
            "translate": "translate_to_english",
            "compute_type": "compute_type",
            "device": "device",
            "gpu_index": "gpu_index",
            "cpu_threads": "cpu_threads",
            "beam_size": "beam_size",
            "best_of": "best_of",
            "patience": "patience",
            "temperature": "temperature",
            "length_penalty": "length_penalty",
            "repetition_penalty": "repetition_penalty",
            "no_repeat_ngram_size": "no_repeat_ngram_size",
            "compression_ratio_threshold": "compression_ratio_threshold",
            "log_prob_threshold": "log_prob_threshold",
            "no_speech_threshold": "no_speech_threshold",
            "condition_on_previous_text": "condition_on_previous_text",
            "prompt_reset_on_temperature": "prompt_reset_on_temperature",
            "initial_prompt": "initial_prompt",
            "repeat_initial_prompt": "repeat_initial_prompt_every_window",
            "prefix": "prefix",
            "hotwords": "hotwords",
            "suppress_blank": "suppress_blank",
            "suppress_tokens": "suppress_tokens",
            "max_initial_timestamp": "max_initial_timestamp",
            "word_timestamps": "word_timestamps",
            "normalize_word_timestamps": "normalize_word_timestamps",
            "highlight_words": "highlight_words",
            "prepend_punctuations": "prepend_punctuations",
            "append_punctuations": "append_punctuations",
            "max_new_tokens": "max_new_tokens",
            "chunk_length": "chunk_length",
            "hallucination_silence_threshold": "hallucination_silence_threshold",
            "language_detection_threshold": "language_detection_threshold",
            "language_detection_segments": "language_detection_segments",
            "use_batched_inference": "use_batched_inference",
            "batch_size": "batch_size",
        }
        values: dict[str, Any] = {}
        for registry_suffix, field_name in key_map.items():
            registry_key = f"{prefix}{registry_suffix}"
            if registry_key in source:
                values[field_name] = source[registry_key]

        vad_map = {
            "vad_filter": "enabled",
            "vad_threshold": "threshold",
            "vad_min_speech_ms": "min_speech_duration_ms",
            "vad_max_speech_s": "max_speech_duration_s",
            "vad_min_silence_ms": "min_silence_duration_ms",
            "vad_speech_pad_ms": "speech_pad_ms",
        }
        vad_values = {
            field_name: source[f"{prefix}{registry_suffix}"]
            for registry_suffix, field_name in vad_map.items()
            if f"{prefix}{registry_suffix}" in source
        }
        if vad_values:
            values["vad"] = WhisperVadParams(**vad_values)
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON request representation."""

        payload = asdict(self)
        payload["vad"] = self.vad.to_dict()
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WhisperParams":
        """Restore parameters from :meth:`to_dict`, ignoring future fields."""

        source = data if isinstance(data, Mapping) else {}
        known = {item.name for item in fields(cls)}
        values = {key: value for key, value in source.items() if key in known}
        values["vad"] = WhisperVadParams.from_dict(source.get("vad"))
        return cls(**values)


@dataclass
class TranscriptOutputOptions:
    formats: tuple[str, ...] = ("srt", "vtt", "txt", "lrc", "tsv", "json")
    add_timestamp: bool = False
    file_suffix: str = ""

    def __post_init__(self) -> None:
        raw_formats = self.formats if not isinstance(self.formats, str) else (self.formats,)
        normalized: list[str] = []
        for item in raw_formats:
            value = str(item).strip().casefold().lstrip(".")
            value = "vtt" if value == "webvtt" else value
            if value in {"srt", "vtt", "txt", "lrc", "tsv", "json"} and value not in normalized:
                normalized.append(value)
        self.formats = tuple(normalized)
        self.add_timestamp = _bool(self.add_timestamp, False)
        self.file_suffix = _text(self.file_suffix)

    def to_dict(self) -> dict[str, Any]:
        return {
            "formats": list(self.formats),
            "add_timestamp": self.add_timestamp,
            "file_suffix": self.file_suffix,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "TranscriptOutputOptions":
        source = data if isinstance(data, Mapping) else {}
        formats = (
            tuple(source.get("formats") or ())
            if "formats" in source
            else ("srt", "vtt", "txt", "lrc", "tsv", "json")
        )
        return cls(
            formats=formats,
            add_timestamp=source.get("add_timestamp", False),
            file_suffix=source.get("file_suffix", ""),
        )


__all__ = [
    "COMPUTE_TYPE_CHOICES",
    "DEVICE_CHOICES",
    "LANGUAGE_AUTO",
    "LANGUAGE_CHOICES",
    "TranscriptOutputOptions",
    "WHISPER_LANGUAGES",
    "WHISPER_LANGUAGE_ALIASES",
    "WhisperParams",
    "WhisperVadParams",
    "code_to_language",
    "language_to_code",
    "parse_suppress_tokens",
]
