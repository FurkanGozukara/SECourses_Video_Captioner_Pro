from __future__ import annotations

import json
from pathlib import Path

from vcap.core.captions_post import (
    Segment,
    apply_replacements,
    caption_stats,
    clamp_segments_to_window,
    diff_html,
    finalize_caption,
    format_replace_pairs,
    parse_replace_pairs,
    replace_pairs_to_html_chips,
    to_srt,
    to_vtt,
    write_caption_outputs,
)


def test_replace_parser_single_pass_and_unicode_words() -> None:
    pairs = parse_replace_pairs("a;b | b;c\nkadın;ohwx kadın\n\ninvalid")
    assert pairs == [("a", "b"), ("b", "c"), ("kadın", "ohwx kadın")]
    assert format_replace_pairs(pairs).splitlines()[0] == "a;b"
    assert apply_replacements("a b", pairs[:2], whole_words=False) == "b c"
    assert apply_replacements("kadın kadınlar", pairs[2:]) == "ohwx kadın kadınlar"
    chips = replace_pairs_to_html_chips([("<person>", "ohwx")])
    assert "&lt;person&gt;" in chips and "vcap-replace-chip" in chips


def test_case_insensitive_literal_replacements_preserve_common_word_casing() -> None:
    pairs = [("cars", "vehicles")]
    assert apply_replacements("cars", pairs) == "vehicles"
    assert apply_replacements("Cars", pairs) == "Vehicles"
    assert apply_replacements("CARS", pairs) == "VEHICLES"
    assert apply_replacements("Cars", [("cars", "newVehicles")]) == "newVehicles"

    # Regex replacements retain their existing verbatim behavior.
    assert apply_replacements("Cars CARS", [(r"cars", "vehicles")], regex=True) == (
        "vehicles vehicles"
    )


def test_finalize_order_subtitles_stats_and_diff() -> None:
    result = finalize_caption(
        "```text\nA woman\n```",
        prefix="quality,",
        suffix="outdoors",
        trigger="ohwx",
        trigger_mode="prefix",
        replace_pairs=[("woman", "person")],
        collapse_whitespace=True,
    )
    assert result == "quality, ohwx A person outdoors"
    assert finalize_caption("caption", trigger="ohwx", trigger_mode="suffix") == "caption ohwx"

    segments = [Segment(0, 1.234, "Hello"), Segment(61.5, 62, "World")]
    srt = to_srt(segments)
    assert "00:00:00,000 --> 00:00:01,234" in srt
    assert "00:01:01,500 --> 00:01:02,000" in srt
    assert to_vtt(segments).startswith("WEBVTT\n\n")
    stats = caption_stats("One sentence.\nTwo words!")
    assert stats["sentences"] == 2 and stats["lines"] == 2
    difference = diff_html("old <word>", "new word")
    assert "<del" in difference and "<ins" in difference and "&lt;" in difference


def test_subtitle_cues_are_clamped_to_their_absolute_clip_window() -> None:
    clamped = clamp_segments_to_window(
        [
            Segment(4.0, 7.0, "starts early"),
            Segment(5.0, 5.0, "zero at start"),
            Segment(8.0, 15.0, "ends late"),
            Segment(8.0, 8.0, "zero duration"),
            Segment(9.8, 9.7, "cannot repair"),
            Segment(10.0, 12.0, "starts after clip"),
        ],
        5.0,
        10.0,
    )

    assert clamped == [
        Segment(5.0, 7.0, "starts early"),
        Segment(5.0, 5.5, "zero at start"),
        Segment(8.0, 10.0, "ends late"),
        Segment(8.0, 8.5, "zero duration"),
    ]
    srt = to_srt(clamped)
    vtt = to_vtt(clamped)
    assert "00:00:10,000" in srt
    assert "00:00:10.000" in vtt
    assert "00:00:12" not in srt + vtt


def test_caption_output_writer(tmp_path: Path) -> None:
    written = write_caption_outputs(
        tmp_path,
        "caption",
        ["json", "srt", "vtt", "jsonl", "reasoning"],
        text="kadın",
        structured={"caption": "kadın"},
        segments=[Segment(0, 1, "kadın")],
        reasoning="reason",
    )
    assert set(written) == {"txt", "json", "srt", "vtt", "jsonl", "reasoning"}
    assert "kadın" in written["json"].read_text(encoding="utf-8")
    assert json.loads(written["jsonl"].read_text(encoding="utf-8")) == {"caption": "kadın"}
