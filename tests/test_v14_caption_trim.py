from __future__ import annotations

from vcap.core.captions_post import finalize_caption
from vcap.pipeline.job import JobSpec, PostSpec
from vcap.pipeline.runner import _finalize_cue_text, _finalize_structured, _finalize_text


def test_caption_limit_prefers_sentence_then_word_boundary() -> None:
    sentence = finalize_caption(
        "One complete sentence. Another unfinished phrase continues.",
        max_length=38,
        trigger_mode="none",
    )
    assert sentence == "One complete sentence."
    assert len(sentence) <= 38

    words = finalize_caption(
        "alpha beta gamma delta epsilon",
        max_length=18,
        trigger_mode="none",
    )
    assert words == "alpha beta gamma"
    assert len(words) <= 18
    assert finalize_caption("unchanged", max_length=0, trigger_mode="none") == "unchanged"


def test_caption_limit_never_slices_prefix_or_trigger() -> None:
    assert (
        finalize_caption(
            "caption words",
            prefix="PREFIX",
            trigger="two word trigger",
            trigger_mode="prefix",
            max_length=10,
        )
        == ""
    )
    assert (
        finalize_caption(
            "caption words that continue",
            prefix="P",
            trigger="TRIGGER",
            trigger_mode="prefix",
            max_length=12,
        )
        == "P TRIGGER"
    )


def test_limit_is_applied_to_text_cues_and_structured_json() -> None:
    spec = JobSpec(post=PostSpec(max_caption_chars=17))
    long_text = "First sentence. Second sentence is too long."
    assert _finalize_text(spec, long_text) == "First sentence."
    assert _finalize_cue_text(spec, long_text) == "First sentence."
    structured = _finalize_structured(
        spec,
        [
            {"timestamp": "00:00-00:20", "text": long_text},
            {"caption": "alpha beta gamma delta epsilon"},
        ],
    )
    assert structured == [
        {"timestamp": "00:00-00:20", "text": "First sentence."},
        {"caption": "alpha beta gamma"},
    ]
