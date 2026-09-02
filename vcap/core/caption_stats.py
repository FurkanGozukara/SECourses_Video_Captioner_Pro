"""Pure dataset-level caption statistics used by the Caption Editor."""

from __future__ import annotations

import html
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .captions_post import caption_stats


_WORD_RE = re.compile(r"(?u)\b[^\W\d_][\w'-]*\b")
_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "he",
        "her",
        "his",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "she",
        "that",
        "the",
        "their",
        "there",
        "they",
        "this",
        "to",
        "was",
        "were",
        "with",
        "you",
    }
)


def _item_text(item: Mapping[str, Any]) -> str:
    return str(item.get("caption") or item.get("caption_text") or item.get("text") or "")


def _item_name(item: Mapping[str, Any]) -> str:
    raw = item.get("media_path") or item.get("source_media_path") or item.get("caption_path") or "caption"
    return Path(str(raw)).name


def _distribution(values: Iterable[int]) -> dict[str, float | int]:
    selected = [int(value) for value in values]
    if not selected:
        return {"min": 0, "avg": 0.0, "max": 0}
    return {
        "min": min(selected),
        "avg": sum(selected) / len(selected),
        "max": max(selected),
    }


def duplicate_caption_groups(
    items: Iterable[Mapping[str, Any]],
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Group exact non-empty captions and return at most ``limit`` duplicates."""

    grouped: dict[str, list[str]] = defaultdict(list)
    original: dict[str, str] = {}
    for item in items:
        caption = _item_text(item).strip()
        if not caption:
            continue
        key = re.sub(r"\s+", " ", caption).casefold()
        grouped[key].append(_item_name(item))
        original.setdefault(key, caption)
    duplicates = [
        {"caption": original[key], "files": sorted(names, key=str.casefold)}
        for key, names in grouped.items()
        if len(names) > 1
    ]
    duplicates.sort(key=lambda group: (-len(group["files"]), str(group["caption"]).casefold()))
    return duplicates[: max(0, int(limit))]


def top_caption_words(
    captions: Iterable[str],
    *,
    limit: int = 30,
    stop_words: Iterable[str] | None = None,
) -> list[tuple[str, int]]:
    """Count case-folded caption words, excluding a compact stop-word list."""

    excluded = {str(word).casefold() for word in (stop_words or _STOP_WORDS)}
    counts: Counter[str] = Counter()
    for caption in captions:
        counts.update(
            word.casefold()
            for word in _WORD_RE.findall(str(caption))
            if word.casefold() not in excluded and len(word) > 1
        )
    return counts.most_common(max(0, int(limit)))


def calculate_caption_statistics(
    items: Iterable[Mapping[str, Any]],
    trigger_word: str = "",
    *,
    duplicate_limit: int = 20,
    top_word_limit: int = 30,
) -> dict[str, Any]:
    """Calculate the complete Caption Editor dataset summary."""

    selected = [dict(item) for item in items]
    captions = [_item_text(item) for item in selected]
    non_empty = [caption for caption in captions if caption.strip()]
    char_counts: list[int] = []
    token_counts: list[int] = []
    for item, caption in zip(selected, captions):
        measured = caption_stats(caption)
        char_counts.append(int(item.get("chars", measured["chars"]) or 0))
        token_counts.append(int(item.get("tokens", measured["approx_tokens"]) or 0))

    trigger = str(trigger_word or "").strip()
    trigger_hits = 0
    if trigger:
        trigger_folded = trigger.casefold()
        trigger_hits = sum(trigger_folded in caption.casefold() for caption in non_empty)
    coverage = (100.0 * trigger_hits / len(non_empty)) if non_empty and trigger else 0.0

    return {
        "item_count": len(selected),
        "captioned": len(non_empty),
        "empty": len(selected) - len(non_empty),
        "failed": sum(str(item.get("status") or "").casefold() == "failed" for item in selected),
        "chars": _distribution(char_counts),
        "tokens": _distribution(token_counts),
        "trigger_word": trigger,
        "trigger_hits": trigger_hits,
        "trigger_coverage_pct": coverage,
        "duplicates": duplicate_caption_groups(selected, limit=duplicate_limit),
        "top_words": top_caption_words(non_empty, limit=top_word_limit),
    }


def render_caption_statistics(stats: Mapping[str, Any]) -> str:
    """Render a statistics mapping as compact Markdown with escaped user data."""

    chars = dict(stats.get("chars") or {})
    tokens = dict(stats.get("tokens") or {})
    lines = [
        "### Dataset statistics",
        (
            f"**{int(stats.get('item_count', 0))} items** | "
            f"{int(stats.get('captioned', 0))} captioned | "
            f"{int(stats.get('empty', 0))} empty | "
            f"{int(stats.get('failed', 0))} failed"
        ),
        (
            "**Characters:** "
            f"{int(chars.get('min', 0))} min / {float(chars.get('avg', 0.0)):.1f} avg / "
            f"{int(chars.get('max', 0))} max"
        ),
        (
            "**Approx. tokens:** "
            f"{int(tokens.get('min', 0))} min / {float(tokens.get('avg', 0.0)):.1f} avg / "
            f"{int(tokens.get('max', 0))} max"
        ),
    ]
    trigger = str(stats.get("trigger_word") or "")
    if trigger:
        lines.append(
            f"**Trigger coverage:** {float(stats.get('trigger_coverage_pct', 0.0)):.1f}% "
            f"({int(stats.get('trigger_hits', 0))}/{int(stats.get('captioned', 0))}) for "
            f"`{html.escape(trigger)}`"
        )
    else:
        lines.append("**Trigger coverage:** no trigger word configured")

    duplicates = list(stats.get("duplicates") or [])
    lines.extend(["", "#### Duplicate captions"])
    if duplicates:
        for group in duplicates:
            names = ", ".join(f"`{html.escape(str(name))}`" for name in group.get("files") or [])
            excerpt = re.sub(r"\s+", " ", str(group.get("caption") or "")).strip()
            if len(excerpt) > 140:
                excerpt = excerpt[:137].rstrip() + "..."
            lines.append(f"- {names}: {html.escape(excerpt)}")
    else:
        lines.append("No exact duplicate captions.")

    words = list(stats.get("top_words") or [])
    lines.extend(["", "#### Top words"])
    lines.append(
        ", ".join(f"**{html.escape(str(word))}** {int(count)}" for word, count in words)
        if words
        else "No caption words to count."
    )
    return "\n".join(lines)


# Concise aliases for callers and downstream tests.
dataset_caption_stats = calculate_caption_statistics
render_caption_stats = render_caption_statistics


__all__ = [
    "calculate_caption_statistics",
    "dataset_caption_stats",
    "duplicate_caption_groups",
    "render_caption_statistics",
    "render_caption_stats",
    "top_caption_words",
]
