from __future__ import annotations

from vcap.core.caption_stats import (
    calculate_caption_statistics,
    duplicate_caption_groups,
    render_caption_statistics,
    top_caption_words,
)


def test_caption_statistics_cover_counts_duplicates_trigger_and_words() -> None:
    items = [
        {"media_path": "one.mp4", "caption": "ohwx red car drives fast", "status": "ok"},
        {"media_path": "two.mp4", "caption": "OHWX red car drives fast", "status": "ok"},
        {"media_path": "three.mp4", "caption": "", "status": "failed"},
        {"media_path": "four.mp4", "caption": "blue car stops", "status": "ok"},
    ]
    stats = calculate_caption_statistics(items, "ohwx")
    assert stats["item_count"] == 4
    assert stats["captioned"] == 3
    assert stats["empty"] == 1
    assert stats["failed"] == 1
    assert stats["trigger_hits"] == 2
    assert stats["trigger_coverage_pct"] == 200 / 3
    assert stats["duplicates"][0]["files"] == ["one.mp4", "two.mp4"]
    assert dict(stats["top_words"])["car"] == 3
    rendered = render_caption_statistics(stats)
    assert "66.7%" in rendered
    assert "one.mp4" in rendered and "two.mp4" in rendered


def test_caption_statistics_helpers_limit_and_remove_stop_words() -> None:
    items = [
        {"caption_path": f"{index}.txt", "caption": "same caption"}
        for index in range(3)
    ]
    assert len(duplicate_caption_groups(items, limit=1)) == 1
    words = dict(top_caption_words(["the bright light and the light"]))
    assert "the" not in words and "and" not in words
    assert words == {"light": 2, "bright": 1}

