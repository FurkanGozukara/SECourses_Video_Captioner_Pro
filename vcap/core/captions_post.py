"""Caption cleanup, replacement, subtitle, diff, and output helpers."""

from __future__ import annotations

import difflib
import html
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

from .outputs import OutputWriter

ReplacePair = tuple[str, str]


@dataclass(frozen=True)
class Segment:
    """One timestamped caption segment."""

    start_s: float
    end_s: float
    text: str


def parse_replace_pairs(text: str | Sequence[Sequence[str]] | None) -> list[ReplacePair]:
    """Parse ``find;replace`` pairs separated by newlines or pipe characters."""

    if text is None:
        return []
    if not isinstance(text, str):
        result: list[ReplacePair] = []
        for value in text:
            if len(value) >= 2 and str(value[0]).strip():
                result.append((str(value[0]).strip(), str(value[1]).strip()))
        return result
    result = []
    for line in text.splitlines():
        for raw_pair in line.split("|"):
            pair = raw_pair.strip()
            if not pair or ";" not in pair:
                continue
            find, replacement = pair.split(";", 1)
            find = find.strip()
            if find:
                result.append((find, replacement.strip()))
    return result


def format_replace_pairs(pairs: Iterable[Sequence[str]]) -> str:
    """Serialize replacement pairs as one ``find;replace`` entry per line."""

    normalized = parse_replace_pairs(list(pairs))
    return "\n".join(f"{find};{replacement}" for find, replacement in normalized)


def replace_pairs_to_html_chips(pairs: Iterable[Sequence[str]]) -> str:
    """Render escaped, theme-neutral replacement chips for the UI."""

    chips = []
    for find, replacement in parse_replace_pairs(list(pairs)):
        chips.append(
            '<span class="vcap-replace-chip">'
            f'<span class="vcap-replace-find">{html.escape(find)}</span>'
            '<span class="vcap-replace-arrow" aria-hidden="true">&rarr;</span>'
            f'<span class="vcap-replace-value">{html.escape(replacement)}</span>'
            "</span>"
        )
    return '<div class="vcap-replace-chips">' + "".join(chips) + "</div>"


def apply_replacements(
    text: str,
    pairs: Iterable[Sequence[str]],
    *,
    case_insensitive: bool = True,
    whole_words: bool = True,
    regex: bool = False,
) -> str:
    """Apply all pairs in one regex pass so replacement output never cascades."""

    normalized = parse_replace_pairs(list(pairs))
    if not normalized or not text:
        return str(text)
    unique: list[ReplacePair] = []
    seen: set[str] = set()
    for find, replacement in normalized:
        key = find.casefold() if case_insensitive else find
        if key not in seen:
            seen.add(key)
            unique.append((find, replacement))
    unique.sort(key=lambda pair: len(pair[0]), reverse=True)
    alternatives: list[str] = []
    replacements: dict[str, str] = {}
    for index, (find, replacement) in enumerate(unique):
        expression = find if regex else re.escape(find)
        if whole_words:
            expression = rf"(?<!\w)(?:{expression})(?!\w)"
        name = f"vcap_pair_{index}"
        alternatives.append(rf"(?P<{name}>{expression})")
        replacements[name] = replacement
    flags = re.IGNORECASE if case_insensitive else 0
    compiled = re.compile("|".join(alternatives), flags=flags)

    def replace_match(match: re.Match[str]) -> str:
        group = match.lastgroup
        return replacements.get(group or "", match.group(0))

    return compiled.sub(replace_match, str(text))


_OUTER_FENCE = re.compile(
    r"^\s*```(?:[A-Za-z0-9_.+-]+)?\s*\n?(.*?)\n?```\s*$",
    flags=re.DOTALL,
)


def _strip_outer_fence(text: str) -> str:
    match = _OUTER_FENCE.match(text)
    return match.group(1).strip() if match else text


def finalize_caption(
    text: str,
    *,
    prefix: str = "",
    suffix: str = "",
    trigger: str = "",
    trigger_mode: Literal["prefix", "suffix", "none"] = "prefix",
    replace_pairs: Iterable[Sequence[str]] | str | None = None,
    replace_opts: Mapping[str, Any] | None = None,
    collapse_whitespace: bool = False,
    strip_markdown_fences: bool = True,
    max_length: int | None = None,
) -> str:
    """Finalize a caption, separating injected parts with one ASCII space.

    Cleanup and replacements are applied to model text first. Prefix, trigger,
    and suffix are injected afterwards, so user-supplied replacement rules cannot
    accidentally modify those control words.
    """

    caption = "" if text is None else str(text)
    if strip_markdown_fences:
        caption = _strip_outer_fence(caption)
    if collapse_whitespace:
        caption = " ".join(caption.split())
    pairs = parse_replace_pairs(replace_pairs) if isinstance(replace_pairs, str) else parse_replace_pairs(list(replace_pairs or []))
    if pairs:
        options = dict(replace_opts or {})
        caption = apply_replacements(
            caption,
            pairs,
            case_insensitive=bool(options.get("case_insensitive", True)),
            whole_words=bool(options.get("whole_words", True)),
            regex=bool(options.get("regex", False)),
        )
    mode = str(trigger_mode).casefold()
    if mode not in {"prefix", "suffix", "none"}:
        raise ValueError("trigger_mode must be 'prefix', 'suffix', or 'none'")
    parts: list[str] = []
    if str(prefix).strip():
        parts.append(str(prefix).strip())
    if mode == "prefix" and str(trigger).strip():
        parts.append(str(trigger).strip())
    if caption.strip():
        parts.append(caption.strip())
    if mode == "suffix" and str(trigger).strip():
        parts.append(str(trigger).strip())
    if str(suffix).strip():
        parts.append(str(suffix).strip())
    result = " ".join(parts)
    if max_length is not None:
        limit = int(max_length)
        if limit < 0:
            raise ValueError("max_length cannot be negative")
        result = result[:limit].rstrip()
    return result


def _timestamp(seconds: float, decimal: str) -> str:
    total_ms = max(0, int(round(float(seconds) * 1000)))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{decimal}{milliseconds:03d}"


def _segments(values: Iterable[Segment | Sequence[Any]]) -> list[Segment]:
    result: list[Segment] = []
    for value in values:
        if isinstance(value, Segment):
            segment = value
        else:
            if len(value) < 3:
                continue
            segment = Segment(float(value[0]), float(value[1]), str(value[2]))
        if segment.end_s >= segment.start_s:
            result.append(segment)
    return result


def to_srt(segments: Iterable[Segment | Sequence[Any]]) -> str:
    """Serialize timestamped segments as SubRip UTF-8 text."""

    blocks = []
    for index, segment in enumerate(_segments(segments), start=1):
        blocks.append(
            f"{index}\n{_timestamp(segment.start_s, ',')} --> {_timestamp(segment.end_s, ',')}\n{segment.text.strip()}"
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def to_vtt(segments: Iterable[Segment | Sequence[Any]]) -> str:
    """Serialize timestamped segments as WebVTT UTF-8 text."""

    blocks = []
    for index, segment in enumerate(_segments(segments), start=1):
        blocks.append(
            f"{index}\n{_timestamp(segment.start_s, '.')} --> {_timestamp(segment.end_s, '.')}\n{segment.text.strip()}"
        )
    return "WEBVTT\n\n" + "\n\n".join(blocks) + ("\n" if blocks else "")


def _structured_value(structured: Any, text: str) -> Any:
    if structured is None:
        return {"text": text}
    if isinstance(structured, str):
        try:
            return json.loads(structured)
        except json.JSONDecodeError:
            return {"text": structured}
    return structured


def write_caption_outputs(
    out_dir: str | Path,
    stem: str,
    formats: Iterable[str],
    *,
    text: str,
    structured: Any = None,
    segments: Iterable[Segment | Sequence[Any]] | None = None,
    reasoning: str | None = None,
) -> dict[str, Path]:
    """Write requested caption formats atomically; plain text is always included."""

    requested = [str(value).casefold().lstrip(".") for value in formats]
    if "txt" not in requested:
        requested.insert(0, "txt")
    requested = list(dict.fromkeys(requested))
    writer = OutputWriter()
    paths = writer.caption_output_paths(out_dir, stem, requested)
    normalized_segments = _segments(segments or [])
    value = _structured_value(structured, str(text))
    written: dict[str, Path] = {}
    for output_format in requested:
        path = paths[output_format]
        if output_format in {"srt", "vtt"} and not normalized_segments:
            continue  # no timed cues: never leave an empty subtitle file behind
        if output_format == "reasoning" and not (reasoning or "").strip():
            continue  # non-thinking models produce no reasoning text
        if output_format == "txt":
            payload = str(text)
            written[output_format] = writer.write_text(path, payload + ("\n" if payload and not payload.endswith("\n") else ""))
        elif output_format == "json":
            written[output_format] = writer.write_json(path, value, pretty=True)
        elif output_format == "srt":
            written[output_format] = writer.write_text(path, to_srt(normalized_segments))
        elif output_format == "vtt":
            written[output_format] = writer.write_text(path, to_vtt(normalized_segments))
        elif output_format == "jsonl":
            rows = value if isinstance(value, list) else [value]
            payload = "\n".join(json.dumps(row, ensure_ascii=False, default=str) for row in rows)
            written[output_format] = writer.write_text(path, payload + ("\n" if payload else ""))
        elif output_format == "reasoning":
            payload = reasoning or ""
            written[output_format] = writer.write_text(path, payload + ("\n" if payload and not payload.endswith("\n") else ""))
    return written


def caption_stats(text: str) -> dict[str, int]:
    """Return lightweight, tokenizer-free caption statistics."""

    value = "" if text is None else str(text)
    words = re.findall(r"(?u)\b\w+(?:['’]\w+)*\b", value)
    sentence_marks = re.findall(r"[.!?]+(?=\s|$)", value)
    sentences = len(sentence_marks) if sentence_marks else (1 if value.strip() else 0)
    return {
        "chars": len(value),
        "words": len(words),
        "sentences": sentences,
        "approx_tokens": math.ceil(len(value) / 4) if value else 0,
        "lines": len(value.splitlines()) if value else 0,
    }


def diff_html(old: str, new: str) -> str:
    """Render an escaped word-level diff using theme-neutral insert/delete spans."""

    tokenizer = re.compile(r"(?u)\s+|\w+|[^\w\s]")
    before = tokenizer.findall(str(old))
    after = tokenizer.findall(str(new))
    matcher = difflib.SequenceMatcher(a=before, b=after, autojunk=False)
    output: list[str] = []
    for operation, a0, a1, b0, b1 in matcher.get_opcodes():
        old_part = html.escape("".join(before[a0:a1]))
        new_part = html.escape("".join(after[b0:b1]))
        if operation == "equal":
            output.append(old_part)
        elif operation == "delete":
            output.append(f'<del class="vcap-diff-del">{old_part}</del>')
        elif operation == "insert":
            output.append(f'<ins class="vcap-diff-ins">{new_part}</ins>')
        else:
            output.append(f'<del class="vcap-diff-del">{old_part}</del>')
            output.append(f'<ins class="vcap-diff-ins">{new_part}</ins>')
    return '<div class="vcap-caption-diff">' + "".join(output) + "</div>"


__all__ = [
    "Segment",
    "apply_replacements",
    "caption_stats",
    "diff_html",
    "finalize_caption",
    "format_replace_pairs",
    "parse_replace_pairs",
    "replace_pairs_to_html_chips",
    "to_srt",
    "to_vtt",
    "write_caption_outputs",
]
