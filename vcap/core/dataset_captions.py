"""Helpers for split video/audio dataset-caption outputs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from vcap.models.registry import get_variant, variant_to_family

from .paths import sanitize_filename


DEFAULT_AUDIO_CAPTION_TEMPLATE = "{{TRANSCRIPT}}\n\n{{SOUND_CAPTION}}"
DEFAULT_CAPTION_MERGE_TEMPLATE = "{{VIDEO_CAPTION}}\n\n{{AUDIO_CAPTION}}"

_TOKEN_RE = re.compile(r"\{\{\s*([A-Z_]+)\s*\}\}", re.IGNORECASE)
_BLANK_LINES_RE = re.compile(r"\n[ \t]*\n(?:[ \t]*\n)+")
_EMPTY_TOKEN_MARKER = "\0VCAP_EMPTY_CAPTION_TOKEN\0"


@dataclass(frozen=True)
class CaptionUnitPaths:
    """Canonical split-layout paths for one source or produced clip."""

    video: Path
    audio: Path
    merged: Path


def caption_unit_paths(out_dir: str | Path, stem: str) -> CaptionUnitPaths:
    """Return sanitized video, audio, and merged paths for one caption unit."""

    directory = Path(out_dir)
    safe_stem = sanitize_filename(str(stem) or "caption")
    return CaptionUnitPaths(
        directory / "video_caption" / f"{safe_stem}.txt",
        directory / "audio_caption" / f"{safe_stem}.txt",
        directory / f"{safe_stem}.txt",
    )


def render_caption_template(template: str, values: Mapping[str, Any]) -> str:
    """Render caption tokens while removing whitespace left by empty values.

    Unknown tokens are preserved so a future preset remains forward compatible.
    Empty token-only lines disappear, adjacent blank lines collapse to one, and
    leading/trailing whitespace is removed without flattening meaningful lines.
    """

    normalized = {str(key).upper(): "" if value is None else str(value) for key, value in values.items()}

    def replace_token(match: re.Match[str]) -> str:
        key = match.group(1).upper()
        if key not in normalized:
            return match.group(0)
        value = normalized[key]
        return value if value.strip() else _EMPTY_TOKEN_MARKER

    def collapse_empty_tokens(line: str) -> str:
        while _EMPTY_TOKEN_MARKER in line:
            marker_start = line.index(_EMPTY_TOKEN_MARKER)
            marker_end = marker_start + len(_EMPTY_TOKEN_MARKER)
            gap_start = marker_start
            gap_end = marker_end
            while gap_start > 0 and line[gap_start - 1] in " \t":
                gap_start -= 1
            while gap_end < len(line) and line[gap_end] in " \t":
                gap_end += 1
            left = line[gap_start - 1] if gap_start > 0 else ""
            right = line[gap_end] if gap_end < len(line) else ""
            had_adjacent_space = gap_start < marker_start or gap_end > marker_end
            replacement = " " if left and right and had_adjacent_space else ""
            line = line[:gap_start] + replacement + line[gap_end:]
        return line

    rendered_lines: list[str] = []
    for raw_line in str(template or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        rendered = _TOKEN_RE.sub(replace_token, raw_line)
        had_empty_token = _EMPTY_TOKEN_MARKER in rendered
        rendered = collapse_empty_tokens(rendered)
        if not rendered.strip() and had_empty_token:
            continue
        rendered_lines.append(rendered)
    rendered = "\n".join(rendered_lines).strip()
    rendered = _BLANK_LINES_RE.sub("\n\n", rendered)
    return rendered.strip()


def _segment_value(segment: Any, name: str, default: Any = None) -> Any:
    if isinstance(segment, Mapping):
        return segment.get(name, default)
    return getattr(segment, name, default)


def transcript_segments(
    source: Any,
    start_s: float = 0.0,
    end_s: float | None = None,
) -> list[dict[str, Any]]:
    """Return overlapping Whisper segments with timestamps local to the window."""

    raw_segments: Iterable[Any]
    if source is None:
        raw_segments = ()
    elif isinstance(source, Mapping):
        raw_segments = source.get("segments", ()) or ()
    else:
        raw_segments = getattr(source, "segments", ()) or ()
    window_start = max(0.0, float(start_s or 0.0))
    window_end = float("inf") if end_s is None else max(window_start, float(end_s))
    result: list[dict[str, Any]] = []
    for segment in raw_segments:
        text = str(_segment_value(segment, "text", "") or "").strip()
        if not text:
            continue
        absolute_start = max(0.0, float(_segment_value(segment, "start", 0.0) or 0.0))
        absolute_end = max(absolute_start, float(_segment_value(segment, "end", absolute_start) or absolute_start))
        if absolute_end <= window_start or absolute_start >= window_end:
            continue
        local_start = max(absolute_start, window_start) - window_start
        local_end = min(absolute_end, window_end) - window_start
        result.append({"start": local_start, "end": max(local_start, local_end), "text": text})
    return result


def _transcript_timestamp(seconds: float) -> str:
    tenths = max(0, int(round(float(seconds) * 10.0)))
    minutes, remainder = divmod(tenths, 600)
    whole_seconds, fraction = divmod(remainder, 10)
    return f"{minutes:02d}:{whole_seconds:02d}.{fraction}"


def render_transcript(
    source: Any,
    style: str = "plain",
    *,
    start_s: float = 0.0,
    end_s: float | None = None,
) -> str:
    """Render Whisper segments as a paragraph, lines, or local timestamped lines."""

    segments = transcript_segments(source, start_s, end_s)
    selected = str(style or "plain").strip().casefold()
    if selected == "timestamped":
        return "\n".join(
            f"[{_transcript_timestamp(item['start'])} - {_transcript_timestamp(item['end'])}] {item['text']}"
            for item in segments
        )
    if selected == "lines":
        return "\n".join(item["text"] for item in segments)
    return re.sub(r"\s+", " ", " ".join(item["text"] for item in segments)).strip()


def auto_captioner_variant(
    main_variant_key: str,
    *,
    vram_tier: int | float | str | None = None,
) -> str:
    """Map a main checkpoint to the matching Qwen3-Omni Captioner variant."""

    variant = get_variant(str(main_variant_key))
    family = variant_to_family(str(main_variant_key))
    key = variant.key.casefold()
    if key.startswith("qwen3_omni_captioner_"):
        return variant.key
    if "gguf_q8" in key:
        suffix = "gguf_q8"
    elif "gguf_q4" in key:
        suffix = "gguf_q4"
    elif key.endswith("_int4"):
        suffix = "int4"
    elif key.endswith("_int8"):
        suffix = "int8"
    elif key.endswith("_bf16"):
        tier = 0.0
        try:
            tier = float(vram_tier or 0)
        except (TypeError, ValueError):
            tier = 0.0
        suffix = "bf16" if family.startswith("qwen3_omni_") or tier >= 80.0 else "int8"
    else:
        suffix = "int8"
    return f"qwen3_omni_captioner_{suffix}"


def resolve_captioner_variant(
    selected: str,
    main_variant_key: str,
    *,
    vram_tier: int | float | str | None = None,
) -> str:
    """Validate an explicit Captioner choice or resolve ``auto``."""

    value = str(selected or "auto").strip()
    if value.casefold() == "auto":
        return auto_captioner_variant(main_variant_key, vram_tier=vram_tier)
    if variant_to_family(value) != "qwen3_omni_captioner":
        raise ValueError(f"Audio caption model must be a Qwen3-Omni Captioner variant: {value}")
    return get_variant(value).key


__all__ = [
    "CaptionUnitPaths",
    "DEFAULT_AUDIO_CAPTION_TEMPLATE",
    "DEFAULT_CAPTION_MERGE_TEMPLATE",
    "auto_captioner_variant",
    "caption_unit_paths",
    "render_caption_template",
    "render_transcript",
    "resolve_captioner_variant",
    "transcript_segments",
]
