from __future__ import annotations

from vcap.core.captions_post import (
    Segment,
    clamp_segments_to_window,
    dedupe_repeated_sentences,
    finalize_caption,
    to_srt,
    to_vtt,
)
from vcap.pipeline.job import JobSpec, PostSpec
from vcap.pipeline.runner import _finalize_structured


def test_repeated_sentence_loop_keeps_the_first_sentence() -> None:
    text, removed = dedupe_repeated_sentences(
        "The man walks. The man walks. The man walks."
    )
    assert text == "The man walks."
    assert removed == 2
    assert finalize_caption("The man walks. The man walks.") == "The man walks."


def test_distinct_and_newline_separated_sentences_preserve_layout() -> None:
    distinct = "The man walks. The woman waves! A car stops?"
    assert dedupe_repeated_sentences(distinct) == (distinct, 0)
    assert dedupe_repeated_sentences("Alpha line\nalpha   line\nBeta line") == (
        "Alpha line\nBeta line",
        1,
    )


def test_comparison_ignores_case_whitespace_and_trailing_punctuation() -> None:
    assert dedupe_repeated_sentences("Hello world!  hello   WORLD?\nHello world…") == (
        "Hello world!",
        2,
    )


def test_structured_json_native_fields_are_not_deduplicated() -> None:
    original = {"caption": "Loop. Loop.", "nested": [{"text": "Again! Again?"}]}
    spec = JobSpec(post=PostSpec(dedupe_repeated_sentences=True))
    assert _finalize_structured(spec, original) == original
    encoded = '{"caption":"Loop. Loop."}'
    assert finalize_caption(encoded) == encoded


def test_subtitle_minimum_duration_and_line_wrapping() -> None:
    cues = clamp_segments_to_window(
        [Segment(0.9, 1.0, "one two three four")],
        0.0,
        1.0,
        min_duration_s=0.5,
    )
    assert cues == [Segment(0.5, 1.0, "one two three four")]
    assert "one two\nthree\nfour" in to_srt(cues, max_line_chars=7)
    assert "one two\nthree\nfour" in to_vtt(cues, max_line_chars=7)
