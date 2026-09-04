"""UTF-8 Whisper transcript and subtitle renderers."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable

from vcap.core.paths import normalize_path, sanitize_filename

from .engine import TranscriptResult, TranscriptSegment, TranscriptWord
from .params import TranscriptOutputOptions

FORMAT_EXTENSIONS = {
    "srt": "srt",
    "vtt": "vtt",
    "webvtt": "vtt",
    "txt": "txt",
    "lrc": "lrc",
    "tsv": "tsv",
    "json": "json",
}

SENTENCE_END_RE = re.compile(r"[.!?]+[\"')\]}]*$")
CLAUSE_END_RE = re.compile(r"[,;:]+[\"')\]}]*$")
ABBREVIATION_RE = re.compile(
    r"^(?:[A-Za-z]\.){2,}$|^(?:Mr|Mrs|Ms|Dr|Prof|Sen|Rep|Gov|St|No|Jr|Sr|Inc|Ltd)\.$",
    re.IGNORECASE,
)
NORMALIZED_SUBTITLE_MAX_CHARS = 92
NORMALIZED_SUBTITLE_MAX_WORDS = 24
NORMALIZED_SUBTITLE_MAX_DURATION = 8.0


def format_timestamp(
    seconds: float,
    *,
    always_include_hours: bool = True,
    decimal_marker: str = ",",
) -> str:
    """Format non-negative seconds using Whisper's millisecond rounding."""

    assert seconds is not None and seconds >= 0, "Wrong timestamp provided"
    milliseconds = round(float(seconds) * 1000.0)
    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000
    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000
    whole_seconds = milliseconds // 1_000
    milliseconds -= whole_seconds * 1_000
    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return (
        f"{hours_marker}{minutes:02d}:{whole_seconds:02d}"
        f"{decimal_marker}{milliseconds:03d}"
    )


def _join_word_text(words: list[TranscriptWord]) -> str:
    raw = "".join(word.word for word in words).strip()
    if " " not in raw and len(words) > 1:
        raw = " ".join(word.word.strip() for word in words if word.word.strip())
    return re.sub(r"\s+", " ", raw).replace("-->", "->").strip()


def _is_sentence_end(text: str) -> bool:
    stripped = text.strip()
    return bool(
        stripped
        and not ABBREVIATION_RE.match(stripped)
        and SENTENCE_END_RE.search(stripped)
    )


def _is_clause_end(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped and CLAUSE_END_RE.search(stripped))


def normalize_result_for_segment_subtitles(result: TranscriptResult) -> TranscriptResult:
    """Rebuild word-timestamp output into sentence-aware subtitle segments."""

    if not any(segment.words for segment in result.segments):
        return result

    normalized_segments: list[TranscriptSegment] = []
    current_words: list[TranscriptWord] = []
    current_fallback_start: float | None = None
    current_fallback_end: float | None = None
    last_soft_break_index: int | None = None

    def reset_soft_break() -> None:
        nonlocal last_soft_break_index
        last_soft_break_index = None
        for index, word in enumerate(current_words, start=1):
            if _is_sentence_end(word.word) or _is_clause_end(word.word):
                last_soft_break_index = index

    def append_segment(start: float, end: float, text: str) -> None:
        normalized_segments.append(
            TranscriptSegment(
                id=len(normalized_segments),
                start=float(start),
                end=max(float(start), float(end)),
                text=text,
                words=[],
            )
        )

    def flush(split_at: int | None = None) -> None:
        nonlocal current_words, current_fallback_start, current_fallback_end
        if not current_words:
            return
        boundary = split_at or len(current_words)
        emitting = current_words[:boundary]
        remaining = current_words[boundary:]
        text = _join_word_text(emitting)
        if text:
            start = emitting[0].start if emitting else current_fallback_start
            end = emitting[-1].end if emitting else current_fallback_end
            if start is not None and end is not None:
                append_segment(start, end, text)
        current_words = remaining
        current_fallback_start = current_words[0].start if current_words else None
        current_fallback_end = current_words[-1].end if current_words else None
        reset_soft_break()

    for segment in result.segments:
        if not segment.words:
            flush()
            text = re.sub(r"\s+", " ", segment.text.strip().replace("-->", "->"))
            if text:
                append_segment(segment.start, segment.end, text)
            continue
        for word in segment.words:
            if not word.word.strip():
                continue
            if not current_words:
                current_fallback_start = segment.start
            current_words.append(word)
            current_fallback_end = segment.end
            if _is_sentence_end(word.word):
                flush()
                continue
            if _is_clause_end(word.word):
                last_soft_break_index = len(current_words)
            text = _join_word_text(current_words)
            duration = current_words[-1].end - current_words[0].start
            too_long = (
                len(text) >= NORMALIZED_SUBTITLE_MAX_CHARS
                or len(current_words) >= NORMALIZED_SUBTITLE_MAX_WORDS
                or duration >= NORMALIZED_SUBTITLE_MAX_DURATION
            )
            if too_long:
                split_at = (
                    last_soft_break_index
                    if last_soft_break_index and last_soft_break_index >= 4
                    else len(current_words)
                )
                flush(split_at)
    flush()

    return TranscriptResult(
        segments=normalized_segments,
        language=result.language,
        language_probability=result.language_probability,
        duration_s=result.duration_s,
        elapsed_s=result.elapsed_s,
        model=result.model,
        compute_type=result.compute_type,
        device=result.device,
    )


def _plain_cues(result: TranscriptResult) -> Iterable[tuple[float, float, str]]:
    for segment in result.segments:
        if segment.text is None:
            continue
        text = segment.text.strip().replace("-->", "->")
        if text:
            yield segment.start, segment.end, text


def _word_segment_cues(result: TranscriptResult) -> Iterable[tuple[float, float, str]]:
    for segment in result.segments:
        if segment.words:
            text = "".join(word.word for word in segment.words).strip().replace("-->", "->")
            if text:
                yield segment.words[0].start, segment.words[-1].end, text
        elif segment.text.strip():
            yield segment.start, segment.end, segment.text.strip().replace("-->", "->")


def _highlight_cues(result: TranscriptResult) -> Iterable[tuple[float, float, str]]:
    for segment in result.segments:
        words = segment.words
        if not words:
            if segment.text.strip():
                yield segment.start, segment.end, segment.text.strip().replace("-->", "->")
            continue
        whole = [word.word for word in words]
        last = words[0].start
        plain = "".join(whole).strip().replace("-->", "->")
        for index, word in enumerate(words):
            if last < word.start:
                yield last, word.start, plain
            highlighted = "".join(
                re.sub(r"^(\s*)(.*)$", r"\1<u>\2</u>", value)
                if item_index == index
                else value
                for item_index, value in enumerate(whole)
            ).strip().replace("-->", "->")
            yield word.start, word.end, highlighted
            last = word.end


def _subtitle_cues(
    result: TranscriptResult,
    *,
    highlight_words: bool,
) -> Iterable[tuple[float, float, str]]:
    if highlight_words and any(segment.words for segment in result.segments):
        return _highlight_cues(result)
    if any(segment.words for segment in result.segments):
        return _word_segment_cues(result)
    return _plain_cues(result)


def _render_srt(result: TranscriptResult, *, highlight_words: bool) -> str:
    blocks = []
    for index, (start, end, text) in enumerate(
        _subtitle_cues(result, highlight_words=highlight_words), start=1
    ):
        blocks.append(
            f"{index}\n{format_timestamp(start)} --> {format_timestamp(end)}\n{text}\n"
        )
    return "\n".join(blocks) + ("\n" if blocks else "")


def _render_vtt(result: TranscriptResult, *, highlight_words: bool) -> str:
    blocks: list[str] = []
    for start, end, text in _subtitle_cues(result, highlight_words=highlight_words):
        blocks.append(
            f"{format_timestamp(start, always_include_hours=False, decimal_marker='.')} --> "
            f"{format_timestamp(end, always_include_hours=False, decimal_marker='.')}\n{text}\n"
        )
    return "WEBVTT\n\n" + "\n".join(blocks) + ("\n" if blocks else "")


def _render_lrc(result: TranscriptResult, *, align_words: bool) -> str:
    lines: list[str] = []
    if align_words and any(segment.words for segment in result.segments):
        for segment in result.segments:
            if not segment.words:
                start = format_timestamp(
                    segment.start, always_include_hours=False, decimal_marker="."
                )
                end = format_timestamp(
                    segment.end, always_include_hours=False, decimal_marker="."
                )
                lines.append(f"[{start}]{segment.text.strip()}[{end}]")
                continue
            pieces = []
            for index, word in enumerate(segment.words):
                value = word.word.strip() if index == 0 else word.word
                word_start = format_timestamp(
                    word.start,
                    always_include_hours=False,
                    decimal_marker=".",
                )
                pieces.append(f"[{word_start}]{value}")
            word_end = format_timestamp(
                segment.words[-1].end,
                always_include_hours=False,
                decimal_marker=".",
            )
            pieces[-1] += f"[{word_end}]"
            lines.append(" ".join(pieces))
    else:
        for start_value, end_value, text in _plain_cues(result):
            start = format_timestamp(
                start_value, always_include_hours=False, decimal_marker="."
            )
            end = format_timestamp(
                end_value, always_include_hours=False, decimal_marker="."
            )
            lines.append(f"[{start}]{text}[{end}]")
    return "\n\n".join(lines) + ("\n\n" if lines else "")


def render_transcript(
    result: TranscriptResult,
    fmt: str,
    *,
    normalize: bool,
    highlight_words: bool,
) -> str:
    """Render a transcript in one of the six supported output formats."""

    format_key = str(fmt).strip().casefold().lstrip(".")
    format_key = "vtt" if format_key == "webvtt" else format_key
    if format_key not in FORMAT_EXTENSIONS:
        raise ValueError(f"Unsupported transcript format: {fmt}")
    rendered_result = normalize_result_for_segment_subtitles(result) if normalize else result
    effective_highlight = bool(highlight_words and not normalize)
    if format_key == "srt":
        return _render_srt(rendered_result, highlight_words=effective_highlight)
    if format_key == "vtt":
        return _render_vtt(rendered_result, highlight_words=effective_highlight)
    if format_key == "txt":
        return "".join(f"{segment.text.strip()}\n" for segment in rendered_result.segments)
    if format_key == "lrc":
        return _render_lrc(rendered_result, align_words=effective_highlight)
    if format_key == "tsv":
        lines = ["start\tend\ttext"]
        lines.extend(
            f"{round(1000 * segment.start)}\t{round(1000 * segment.end)}\t"
            f"{segment.text.strip().replace(chr(9), ' ')}"
            for segment in rendered_result.segments
        )
        return "\n".join(lines) + "\n"
    return json.dumps(rendered_result.to_dict(), indent=2, ensure_ascii=False)


def write_transcript_files(
    result: TranscriptResult,
    out_dir: Path,
    stem: str,
    options: TranscriptOutputOptions,
    *,
    normalize: bool,
    highlight_words: bool,
) -> list[Path]:
    """Write the selected transcript formats and return their absolute paths."""

    directory = normalize_path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    output_options = (
        options
        if isinstance(options, TranscriptOutputOptions)
        else TranscriptOutputOptions.from_dict(options)
    )
    base = sanitize_filename(f"{stem}{output_options.file_suffix}")
    timestamp = datetime.now().strftime("%m%d%H%M%S%f") if output_options.add_timestamp else ""
    output_base = f"{base}-{timestamp}" if timestamp else base
    paths: list[Path] = []
    effective_highlight = bool(
        highlight_words and not normalize and any(segment.words for segment in result.segments)
    )
    for format_key in output_options.formats:
        extension = FORMAT_EXTENSIONS[format_key]
        path = normalize_path(directory / f"{output_base}.{extension}")
        content = render_transcript(
            result,
            format_key,
            normalize=normalize,
            highlight_words=effective_highlight,
        )
        path.write_text(content, encoding="utf-8", newline="\n")
        paths.append(path)
    if effective_highlight:
        companion_base = f"{base}_noword_timestamps"
        if timestamp:
            companion_base += f"-{timestamp}"
        companion = normalize_path(directory / f"{companion_base}.srt")
        companion.write_text(
            render_transcript(
                result,
                "srt",
                normalize=False,
                highlight_words=False,
            ),
            encoding="utf-8",
            newline="\n",
        )
        paths.append(companion)
    return paths


__all__ = [
    "FORMAT_EXTENSIONS",
    "format_timestamp",
    "normalize_result_for_segment_subtitles",
    "render_transcript",
    "write_transcript_files",
]
