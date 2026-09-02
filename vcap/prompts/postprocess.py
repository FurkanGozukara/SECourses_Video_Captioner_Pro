"""Prompt-output post-processors.

The model wrappers pass their decoded text through this module rather than
embedding task-specific parsing rules in each backend.  Every processor is
deliberately best-effort: malformed model output is preserved in ``text`` and
represented by an empty/``None`` structured value instead of raising.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any, Callable, Iterable


Segment = tuple[float, float, str]


@dataclass(frozen=True)
class PostResult:
    """Normalized output returned by every post-processor."""

    text: str
    structured: Any = None
    segments: list[Segment] = field(default_factory=list)

    @property
    def reasoning(self) -> str:
        """Return separated reasoning when produced by ``strip_reasoning``."""

        if isinstance(self.structured, dict):
            value = self.structured.get("reasoning", "")
            return value if isinstance(value, str) else ""
        return ""


_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
_FENCE_START_RE = re.compile(r"^\s*```(?:json|javascript|js)?\s*", re.IGNORECASE)
_FENCE_END_RE = re.compile(r"\s*```\s*$")
_TIME_TOKEN = r"(?:\d+:){0,2}\d+(?:\.\d+)?"
_BRACKETED_TIMESTAMP_RE = re.compile(
    rf"^\s*(?:[-*]\s*)?\[\s*(?P<start>{_TIME_TOKEN})\s*"
    rf"(?:-->|[-\u2013\u2014])\s*(?P<end>{_TIME_TOKEN})\s*\]\s*"
    rf"(?P<text>.*)$"
)
_TRAILING_QA_PAIR_RE = re.compile(
    r"(?ims)(?:^|\n)(?P<start>[ \t]*(?:#{1,4}[ \t]*)?"
    r"(?:question|instruction|query|user|human|q)\s*(?:\d+)?\s*[:\uff1a]"
    r".*?\n[ \t]*(?:#{1,4}[ \t]*)?(?:answer|response|assistant|a)\s*[:\uff1a])"
)
_TRAILING_CHAT_TEMPLATE_RE = re.compile(
    r"(?is)(?P<start><\|im_start\|>\s*(?:user|human).*?"
    r"<\|im_(?:end|start)\|>\s*(?:<\|im_start\|>\s*)?(?:assistant|answer))"
)


def _strip_fences(text: str) -> str:
    value = text.strip().lstrip("\ufeff")
    value = _FENCE_START_RE.sub("", value, count=1)
    value = _FENCE_END_RE.sub("", value, count=1)
    return value.strip()


def _loads_repaired(candidate: str) -> Any:
    """Load JSON after removing the common trailing-comma failure mode."""

    try:
        return json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        return json.loads(_TRAILING_COMMA_RE.sub(r"\1", candidate))


def _balanced_json_candidates(text: str) -> Iterable[str]:
    """Yield complete JSON-looking objects/arrays in source order."""

    for start, opening in sorted(
        ((index, text[index]) for index in range(len(text)) if text[index] in "[{"),
        key=lambda item: item[0],
    ):
        closing_for = {"[": "]", "{": "}"}
        stack: list[str] = [closing_for[opening]]
        in_string = False
        escaped = False
        for index in range(start + 1, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char in "[{":
                stack.append(closing_for[char])
            elif char in "]}":
                if not stack or char != stack[-1]:
                    break
                stack.pop()
                if not stack:
                    yield text[start : index + 1]
                    break


def _recover_complete_array_objects(text: str) -> list[dict[str, Any]]:
    """Recover complete top-level objects from a truncated JSON array."""

    array_start = text.find("[")
    if array_start < 0:
        return []

    recovered: list[dict[str, Any]] = []
    object_start: int | None = None
    object_depth = 0
    in_string = False
    escaped = False

    for index in range(array_start + 1, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if object_depth == 0:
                object_start = index
            object_depth += 1
        elif char == "}" and object_depth:
            object_depth -= 1
            if object_depth == 0 and object_start is not None:
                try:
                    item = _loads_repaired(text[object_start : index + 1])
                except (json.JSONDecodeError, TypeError):
                    item = None
                if isinstance(item, dict):
                    recovered.append(item)
                object_start = None
    return recovered


def _timechat_items(raw: str) -> list[dict[str, Any]]:
    cleaned = _strip_fences(raw)
    try:
        value = _loads_repaired(cleaned)
    except (json.JSONDecodeError, TypeError):
        value = None
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]

    complete_values: list[Any] = []
    for candidate in _balanced_json_candidates(cleaned):
        try:
            parsed = _loads_repaired(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        complete_values.append(parsed)
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]

    recovered = _recover_complete_array_objects(cleaned)
    if recovered:
        return recovered
    for parsed in complete_values:
        if isinstance(parsed, dict):
            return [parsed]
    return []


def _time_to_seconds(value: str | int | float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    parts = [float(part) for part in str(value).strip().split(":")]
    if not 1 <= len(parts) <= 3:
        raise ValueError(f"Unsupported time value: {value!r}")
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60.0 + part
    return seconds


def _timestamp_pair(value: Any) -> tuple[float, float] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return _time_to_seconds(value[0]), _time_to_seconds(value[1])
        except (TypeError, ValueError):
            return None
    if not isinstance(value, str):
        return None

    timestamp = value.strip().strip("[]")
    match = re.fullmatch(
        rf"\s*(?P<start>{_TIME_TOKEN})\s*(?:-->|[-\u2013\u2014])\s*"
        rf"(?P<end>{_TIME_TOKEN})\s*",
        timestamp,
    )
    if not match:
        return None
    try:
        return _time_to_seconds(match.group("start")), _time_to_seconds(match.group("end"))
    except ValueError:
        return None


def _clean_piece(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _as_sentence(value: Any) -> str:
    text = _clean_piece(value)
    if text and text[-1] not in ".!?\"'":
        text += "."
    return text


def _segment_text(item: dict[str, Any]) -> str:
    for field_name in ("segment_detail_caption", "storyline", "speech_content"):
        text = _clean_piece(item.get(field_name))
        if text:
            return text
    return ""


def _segments_for_items(
    items: Iterable[dict[str, Any]],
    text_builder: Callable[[dict[str, Any]], str] = _segment_text,
) -> list[Segment]:
    segments: list[Segment] = []
    for item in items:
        pair = _timestamp_pair(item.get("timestamp"))
        if pair is None:
            continue
        start, end = pair
        segments.append((max(0.0, start), max(max(0.0, start), end), text_builder(item)))
    return sorted(segments, key=lambda segment: (segment[0], segment[1]))


def timechat_parse(raw: str, _options: dict | None = None) -> PostResult:
    """Parse TimeChat JSON, repairing fences, commas, and truncated arrays."""

    items = _timechat_items(raw)
    if not items:
        return PostResult(text=raw.strip(), structured=[], segments=[])
    text = json.dumps(items, ensure_ascii=False, indent=2)
    return PostResult(text=text, structured=items, segments=_segments_for_items(items))


def _wan_item_text(item: dict[str, Any]) -> str:
    parts = [
        _as_sentence(item.get("segment_detail_caption")),
        _as_sentence(item.get("camera_state")),
    ]
    return " ".join(part for part in parts if part).strip()


def timechat_flatten_wan(raw: str, _options: dict | None = None) -> PostResult:
    """Flatten TimeChat events and camera motion into one Wan-ready paragraph."""

    parsed = timechat_parse(raw)
    items = parsed.structured if isinstance(parsed.structured, list) else []
    paragraphs = [_wan_item_text(item) for item in items]
    text = " ".join(paragraph for paragraph in paragraphs if paragraph).strip()
    if not items:
        text = raw.strip()
    return PostResult(text=text, structured=items, segments=_segments_for_items(items, _wan_item_text))


def _motion_camera_text(item: dict[str, Any]) -> str:
    return " ".join(
        part
        for part in (
            _as_sentence(item.get("segment_detail_caption")),
            _as_sentence(item.get("camera_state")),
        )
        if part
    ).strip()


def timechat_flatten_motion_camera(raw: str, _options: dict | None = None) -> PostResult:
    """Keep only chronological motion/event detail and camera state."""

    parsed = timechat_parse(raw)
    items = parsed.structured if isinstance(parsed.structured, list) else []
    segments = _segments_for_items(items, _motion_camera_text)
    text = "\n".join(segment[2] for segment in segments if segment[2]).strip()
    return PostResult(text=text if items else raw.strip(), structured=items, segments=segments)


def _audiovisual_text(item: dict[str, Any]) -> str:
    return " ".join(
        part
        for part in (
            _as_sentence(item.get("segment_detail_caption")),
            _as_sentence(item.get("camera_state")),
            _as_sentence(item.get("speech_content")),
            _as_sentence(item.get("acoustics_content")),
        )
        if part
    ).strip()


def timechat_flatten_av(raw: str, _options: dict | None = None) -> PostResult:
    """Keep chronological visual motion, camera, speech, and acoustics."""

    parsed = timechat_parse(raw)
    items = parsed.structured if isinstance(parsed.structured, list) else []
    segments = _segments_for_items(items, _audiovisual_text)
    text = "\n".join(segment[2] for segment in segments if segment[2]).strip()
    return PostResult(text=text if items else raw.strip(), structured=items, segments=segments)


def _speech_text(item: dict[str, Any]) -> str:
    return _clean_piece(item.get("speech_content"))


def timechat_speech_only(raw: str, _options: dict | None = None) -> PostResult:
    """Extract non-empty timestamped speech as a plain transcript and subtitle cues."""

    parsed = timechat_parse(raw)
    items = parsed.structured if isinstance(parsed.structured, list) else []
    segments = [
        segment
        for segment in _segments_for_items(items, _speech_text)
        if segment[2]
    ]
    text = "\n".join(segment[2] for segment in segments)
    return PostResult(text=text if items else raw.strip(), structured=items, segments=segments)


def _chapter_time(seconds: float, include_hours: bool = False) -> str:
    total = max(0, int(round(float(seconds))))
    if include_hours:
        hours, remainder = divmod(total, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    minutes, secs = divmod(total, 60)
    return f"{minutes:02d}:{secs:02d}"


def timechat_chapters(raw: str, _options: dict | None = None) -> PostResult:
    """Render one timestamped storyline chapter per native TimeChat segment."""

    parsed = timechat_parse(raw)
    items = parsed.structured if isinstance(parsed.structured, list) else []
    chapters: list[tuple[float, float, str]] = []
    for item in items:
        pair = _timestamp_pair(item.get("timestamp"))
        storyline = _clean_piece(item.get("storyline"))
        if pair is None or not storyline:
            continue
        chapters.append((pair[0], pair[1], storyline))
    chapters.sort(key=lambda chapter: (chapter[0], chapter[1]))
    include_hours = any(end >= 3600.0 for _, end, _ in chapters)
    lines = [
        f"{_chapter_time(start, include_hours)}-{_chapter_time(end, include_hours)} {storyline}"
        for start, end, storyline in chapters
    ]
    return PostResult(text="\n".join(lines) if items else raw.strip(), structured=items, segments=[])


_FULL_FIELDS: tuple[tuple[str, str], ...] = (
    ("segment_detail_caption", "Visual"),
    ("camera_state", "Camera"),
    ("video_background", "Setting"),
    ("storyline", "Story"),
    ("shooting_style", "Editing"),
    ("speech_content", "Speech"),
    ("acoustics_content", "Audio"),
)


def _full_item_text(item: dict[str, Any]) -> str:
    parts: list[str] = []
    for field_name, label in _FULL_FIELDS:
        value = _clean_piece(item.get(field_name))
        if value:
            parts.append(f"{label}: {_as_sentence(value)}")
    return " ".join(parts)


def timechat_flatten_full(raw: str, _options: dict | None = None) -> PostResult:
    """Render all seven TimeChat caption fields as a paragraph per segment."""

    parsed = timechat_parse(raw)
    items = parsed.structured if isinstance(parsed.structured, list) else []
    paragraphs = [_full_item_text(item) for item in items]
    text = "\n\n".join(paragraph for paragraph in paragraphs if paragraph).strip()
    if not items:
        text = raw.strip()
    return PostResult(text=text, structured=items, segments=_segments_for_items(items, _full_item_text))


def _format_timestamp(seconds: float, decimal_mark: str) -> str:
    total_ms = max(0, round(float(seconds) * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}{decimal_mark}{milliseconds:03d}"


def to_srt(segments: Iterable[Segment]) -> str:
    """Serialize ``(start_s, end_s, text)`` segments as SubRip."""

    blocks: list[str] = []
    for index, (start, end, text) in enumerate(segments, start=1):
        safe_start = max(0.0, float(start))
        safe_end = max(safe_start, float(end))
        blocks.append(
            f"{index}\n"
            f"{_format_timestamp(safe_start, ',')} --> {_format_timestamp(safe_end, ',')}\n"
            f"{str(text).strip()}"
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def to_vtt(segments: Iterable[Segment]) -> str:
    """Serialize ``(start_s, end_s, text)`` segments as WebVTT."""

    blocks = ["WEBVTT"]
    for start, end, text in segments:
        safe_start = max(0.0, float(start))
        safe_end = max(safe_start, float(end))
        blocks.append(
            f"{_format_timestamp(safe_start, '.')} --> {_format_timestamp(safe_end, '.')}\n"
            f"{str(text).strip()}"
        )
    return "\n\n".join(blocks) + ("\n" if len(blocks) > 1 else "\n\n")


def timechat_srt(raw: str, _options: dict | None = None) -> PostResult:
    """Parse TimeChat output and render its detailed event captions as SRT."""

    parsed = timechat_parse(raw)
    return PostResult(text=to_srt(parsed.segments), structured=parsed.structured, segments=parsed.segments)


def strip_reasoning(raw: str, _options: dict | None = None) -> PostResult:
    """Separate Qwen3 Thinking's ``<think>`` block from its final answer."""

    value = raw.strip()
    if "</think>" in value:
        reasoning_text, answer = value.split("</think>", 1)
        reasoning = reasoning_text.replace("<think>", "", 1).strip()
        answer = answer.strip()
    else:
        reasoning = ""
        answer = value
    return PostResult(
        text=answer,
        structured={"reasoning": reasoning, "answer": answer},
        segments=[],
    )


def segments_from_text_timestamps(text: str) -> list[Segment]:
    """Parse ``[MM:SS-MM:SS] text`` or ``[HH:MM:SS-HH:MM:SS]`` lines."""

    parsed: list[Segment] = []
    current: tuple[float, float, list[str]] | None = None

    def flush() -> None:
        nonlocal current
        if current is None:
            return
        start, end, lines = current
        parsed.append((start, max(start, end), "\n".join(lines).strip()))
        current = None

    for line in text.splitlines():
        match = _BRACKETED_TIMESTAMP_RE.match(line)
        if match:
            flush()
            try:
                start = _time_to_seconds(match.group("start"))
                end = _time_to_seconds(match.group("end"))
            except ValueError:
                continue
            current = (max(0.0, start), max(0.0, end), [match.group("text").strip()])
        elif current is not None and line.strip():
            current[2].append(line.strip())
    flush()
    return parsed


def srt_from_bracketed(raw: str, _options: dict | None = None) -> PostResult:
    """Convert bracketed timestamp lines requested by ASR presets to SRT."""

    segments = segments_from_text_timestamps(raw)
    text = to_srt(segments) if segments else raw.strip()
    return PostResult(text=text, structured=None, segments=segments)


def lyrics_lines(raw: str, _options: dict | None = None) -> PostResult:
    """Normalize prompt-compliant lyrics to non-empty unnumbered lines."""

    value = _strip_fences(raw)
    lines: list[str] = []
    for source_line in value.splitlines():
        line = source_line.strip()
        line = re.sub(r"^(?:[-*•]\s+|\d+[.)]\s+)", "", line).strip()
        if line and line.lower() not in {"lyrics:", "lyrics"}:
            lines.append(line)
    return PostResult(text="\n".join(lines), structured=lines, segments=[])


def tags_normalize(raw: str, _options: dict | None = None) -> PostResult:
    """Lowercase and deduplicate comma/newline-separated tags."""

    seen: set[str] = set()
    tags: list[str] = []
    for piece in re.split(r"[,\n]+", _strip_fences(raw)):
        tag = re.sub(r"^(?:[-*•#]\s*)+", "", piece.strip()).strip(" \t\r\n,.;").lower()
        tag = re.sub(r"\s+", " ", tag)
        if tag and tag not in seen:
            seen.add(tag)
            tags.append(tag)
    return PostResult(text=", ".join(tags), structured=tags, segments=[])


def json_extract(raw: str, _options: dict | None = None) -> PostResult:
    """Extract and parse the first complete JSON object or array in text."""

    value = _strip_fences(raw)
    try:
        structured = _loads_repaired(value)
    except (json.JSONDecodeError, TypeError):
        structured = None
    else:
        return PostResult(
            text=json.dumps(structured, ensure_ascii=False, indent=2),
            structured=structured,
            segments=[],
        )

    for candidate in _balanced_json_candidates(value):
        try:
            structured = _loads_repaired(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        return PostResult(
            text=json.dumps(structured, ensure_ascii=False, indent=2),
            structured=structured,
            segments=[],
        )
    return PostResult(text=raw.strip(), structured=None, segments=[])


def plain(raw: str, _options: dict | None = None) -> PostResult:
    """Return decoded text without task-specific transformation."""

    return PostResult(text=raw.strip(), structured=None, segments=[])


def trim_avocado_trailing_qa(raw: str, *, hit_token_cap: bool) -> tuple[str, bool]:
    """Trim unmistakable training QA continuation only on a capped AVoCaDO decode."""

    value = str(raw).strip()
    if not hit_token_cap or len(value) < 160:
        return value, False
    matches = [
        match.start("start")
        for pattern in (_TRAILING_QA_PAIR_RE, _TRAILING_CHAT_TEMPLATE_RE)
        if (match := pattern.search(value)) is not None
    ]
    if not matches:
        return value, False
    boundary = min(matches)
    # Ordinary captions can quote questions. Only remove a continuation that
    # starts after a substantial answer and in the latter part of the decode.
    if boundary < max(120, int(len(value) * 0.35)):
        return value, False
    trimmed = value[:boundary].rstrip(" \t\r\n-#")
    if len(trimmed) < 120:
        return value, False
    return trimmed, True


POST_PROCESSORS: dict[str, Callable[[str, dict], PostResult]] = {
    "timechat_parse": timechat_parse,
    "timechat_flatten_wan": timechat_flatten_wan,
    "timechat_flatten_motion_camera": timechat_flatten_motion_camera,
    "timechat_flatten_av": timechat_flatten_av,
    "timechat_speech_only": timechat_speech_only,
    "timechat_chapters": timechat_chapters,
    "timechat_flatten_full": timechat_flatten_full,
    "timechat_srt": timechat_srt,
    "strip_reasoning": strip_reasoning,
    "srt_from_bracketed": srt_from_bracketed,
    "lyrics_lines": lyrics_lines,
    "tags_normalize": tags_normalize,
    "json_extract": json_extract,
    "plain": plain,
}


__all__ = [
    "POST_PROCESSORS",
    "PostResult",
    "Segment",
    "json_extract",
    "lyrics_lines",
    "plain",
    "segments_from_text_timestamps",
    "srt_from_bracketed",
    "strip_reasoning",
    "tags_normalize",
    "trim_avocado_trailing_qa",
    "timechat_flatten_full",
    "timechat_flatten_motion_camera",
    "timechat_flatten_av",
    "timechat_flatten_wan",
    "timechat_speech_only",
    "timechat_chapters",
    "timechat_parse",
    "timechat_srt",
    "to_srt",
    "to_vtt",
]
