from __future__ import annotations

import json
import re
from pathlib import Path

from vcap.whisper.engine import TranscriptResult, TranscriptSegment, TranscriptWord
from vcap.whisper.params import TranscriptOutputOptions
from vcap.whisper.writers import (
    format_timestamp,
    normalize_result_for_segment_subtitles,
    render_transcript,
    write_transcript_files,
)


def _result() -> TranscriptResult:
    return TranscriptResult(
        segments=[
            TranscriptSegment(
                id=0,
                start=10.0,
                end=12.0,
                text="Hello world.",
                words=[
                    TranscriptWord(10.0, 10.8, " Hello", 0.9),
                    TranscriptWord(10.8, 12.0, " world.", 0.8),
                ],
                avg_logprob=-0.1,
            ),
            TranscriptSegment(1, 12.0, 14.0, "Second line.", []),
        ],
        language="en",
        language_probability=0.99,
        duration_s=14.0,
        elapsed_s=1.25,
        model="tiny",
        compute_type="int8",
        device="cpu",
    )


def test_format_timestamp_rounding_and_hour_modes() -> None:
    assert format_timestamp(0) == "00:00:00,000"
    assert format_timestamp(61.9996) == "00:01:02,000"
    assert (
        format_timestamp(61.2, always_include_hours=False, decimal_marker=".")
        == "01:01.200"
    )


def test_render_all_formats() -> None:
    result = _result()
    srt = render_transcript(result, "srt", normalize=False, highlight_words=False)
    vtt = render_transcript(result, "webvtt", normalize=False, highlight_words=False)
    txt = render_transcript(result, "txt", normalize=False, highlight_words=False)
    lrc = render_transcript(result, "lrc", normalize=False, highlight_words=False)
    tsv = render_transcript(result, "tsv", normalize=False, highlight_words=False)
    data = json.loads(
        render_transcript(result, "json", normalize=False, highlight_words=False)
    )

    assert "00:00:10,000 --> 00:00:12,000\nHello world." in srt
    assert srt.endswith("\n\n")
    assert vtt.startswith("WEBVTT\n\n")
    assert vtt.endswith("\n\n")
    assert "00:10.000 --> 00:12.000" in vtt
    assert txt == "Hello world.\nSecond line.\n"
    assert "[00:10.000]Hello world.[00:12.000]" in lrc
    assert lrc.endswith("\n\n")
    assert tsv == "start\tend\ttext\n10000\t12000\tHello world.\n12000\t14000\tSecond line.\n"
    assert data == result.to_dict()


def test_normalize_word_stream_uses_sentence_and_abbreviation_rules() -> None:
    words = [
        TranscriptWord(0.0, 0.4, " Dr.", 1.0),
        TranscriptWord(0.4, 0.8, " Jones", 1.0),
        TranscriptWord(0.8, 1.2, " arrived.", 1.0),
        TranscriptWord(1.2, 1.6, " Next", 1.0),
        TranscriptWord(1.6, 2.0, " sentence!", 1.0),
    ]
    source = _result()
    source.segments = [TranscriptSegment(0, 0.0, 2.0, "", words)]
    normalized = normalize_result_for_segment_subtitles(source)

    assert [segment.text for segment in normalized.segments] == [
        "Dr. Jones arrived.",
        "Next sentence!",
    ]
    assert all(not segment.words for segment in normalized.segments)
    assert normalized.model == source.model


def test_normalize_splits_long_word_stream() -> None:
    words = [
        TranscriptWord(index * 0.4, (index + 1) * 0.4, f" word{index}", 1.0)
        for index in range(30)
    ]
    source = _result()
    source.segments = [TranscriptSegment(0, 0.0, 12.0, "", words)]
    normalized = normalize_result_for_segment_subtitles(source)
    assert len(normalized.segments) >= 2
    assert all(len(segment.text) <= 100 for segment in normalized.segments)


def test_highlight_writes_plain_companion(tmp_path: Path) -> None:
    paths = write_transcript_files(
        _result(),
        tmp_path,
        "clip",
        TranscriptOutputOptions(formats=("srt",), file_suffix="_transcript"),
        normalize=False,
        highlight_words=True,
    )

    assert paths == [
        (tmp_path / "clip_transcript.srt").resolve(),
        (tmp_path / "clip_transcript_noword_timestamps.srt").resolve(),
    ]
    assert "<u>Hello</u>" in paths[0].read_text(encoding="utf-8")
    assert "<u>" not in paths[1].read_text(encoding="utf-8")


def test_highlight_companion_is_written_without_srt_selection(tmp_path: Path) -> None:
    paths = write_transcript_files(
        _result(),
        tmp_path,
        "clip",
        TranscriptOutputOptions(formats=("vtt",)),
        normalize=False,
        highlight_words=True,
    )

    assert [path.name for path in paths] == [
        "clip.vtt",
        "clip_noword_timestamps.srt",
    ]
    assert "<u>Hello</u>" in paths[0].read_text(encoding="utf-8")
    assert "<u>" not in paths[1].read_text(encoding="utf-8")


def test_write_all_files_json_is_pretty_and_timestamp_name(tmp_path: Path) -> None:
    paths = write_transcript_files(
        _result(),
        tmp_path,
        "clip",
        TranscriptOutputOptions(add_timestamp=True),
        normalize=True,
        highlight_words=True,
    )

    assert len(paths) == 6
    assert all(re.match(r"clip-\d{16}\.(srt|vtt|txt|lrc|tsv|json)$", path.name) for path in paths)
    json_path = next(path for path in paths if path.suffix == ".json")
    json_text = json_path.read_text(encoding="utf-8")
    assert "\n  \"text\"" in json_text
    assert json.loads(json_text)["segments"][0]["start"] == 10.0
