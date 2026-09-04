from __future__ import annotations

import json
import math
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vcap import PRESETS_DEFAULT_DIR
from vcap.core.dataset_captions import (
    DEFAULT_AUDIO_CAPTION_TEMPLATE,
    DEFAULT_CAPTION_MERGE_TEMPLATE,
    auto_captioner_variant,
    caption_unit_paths,
    render_caption_template,
    render_transcript,
)
from vcap.core.presets import PresetStore
from vcap.core.outputs import list_recent_runs
from vcap.core.subprocess_runner import CancelToken
from vcap.pipeline.job import InputItem, JobSpec, OutputSpec
from vcap.pipeline.runner import (
    _apply_batch_skip,
    _assign_batch_outputs,
    _caption_sound_windows,
    _record_output_location,
    _resolve_inputs,
    run_job,
)
from vcap.ui.app import build_app
from vcap.ui.components import _folder_scan
from vcap.ui.tabs.editor_tab import (
    caption_parts_payload,
    rebuild_caption_parts_after_regeneration,
    scan_folder as scan_editor_folder,
)
from vcap.ui.tabs.caption_tab import _result_summary
from vcap.ui.tabs.recover_tab import present_recovery_settings


def _write_wav(path: Path, seconds: float = 0.4) -> None:
    rate = 8_000
    frames = bytearray()
    for index in range(int(rate * seconds)):
        value = int(1_200 * math.sin(index * 2 * math.pi * 440 / rate))
        frames.extend(value.to_bytes(2, "little", signed=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(frames)


def _settings(**overrides: Any) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "model_key": "qwen3_omni_instruct_int4",
        "prompt_preset_id": "custom",
        "user_prompt": "Describe this clip.",
        "system_prompt": None,
        "max_frames": 0,
        "use_audio_in_video": True,
        "output_formats": ["txt", "json"],
        "keep_model_loaded": False,
        "transcript_enabled": False,
        "transcript_formats": [],
        "transcript_inject_prompt": False,
        "whisper_device": "cpu",
        "whisper_compute_type": "int8",
        "audio_caption_source": "whisper",
        "video_caption_source": "generate",
        "audio_caption_model_key": "auto",
        "audio_caption_transcript_style": "plain",
        "audio_caption_template": DEFAULT_AUDIO_CAPTION_TEMPLATE,
        "caption_write_merged": True,
        "caption_merge_template": DEFAULT_CAPTION_MERGE_TEMPLATE,
        "audio_caption_empty_policy": "skip",
        "audio_caption_empty_text": "No speech.",
    }
    settings.update(overrides)
    return settings


def _fake_transcription(
    request: dict[str, Any],
    **kwargs: Any,
) -> Any:
    from vcap.whisper.client import TranscriptionOutcome
    from vcap.whisper.engine import TranscriptResult, TranscriptSegment

    item = request["items"][0]
    result = TranscriptResult(
        segments=[
            TranscriptSegment(0, 0.0, 0.2, "hello", []),
            TranscriptSegment(1, 0.2, 0.4, "world", []),
        ],
        language="en",
        language_probability=0.99,
        duration_s=0.4,
        elapsed_s=0.01,
        model=request["params"]["model"],
        compute_type=request["params"]["compute_type"],
        device="cpu",
    )
    files: list[str] = []
    for output_format in request["output"]["formats"]:
        path = Path(item["out_dir"]) / (
            f"{item['stem']}{request['output']['file_suffix']}.{output_format}"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(result.text + "\n", encoding="utf-8")
        files.append(str(path))
    sink = kwargs.get("sink")
    if sink is not None:
        sink.on_item_done({"item_index": item["index"], "files": files})
    return TranscriptionOutcome(
        ok=True,
        items=[{"item_index": item["index"], "files": files}],
        results={item["index"]: result},
        elapsed_s=0.01,
        cancelled=False,
        error=None,
    )


def _empty_transcription(request: dict[str, Any], **_kwargs: Any) -> Any:
    from vcap.whisper.client import TranscriptionOutcome
    from vcap.whisper.engine import TranscriptResult

    index = int(request["items"][0]["index"])
    result = TranscriptResult([], "en", 1.0, 0.4, 0.01, "tiny", "int8", "cpu")
    return TranscriptionOutcome(True, [{"item_index": index, "files": []}], {index: result}, 0.01, False, None)


def test_caption_template_rendering_collapses_only_empty_token_gaps() -> None:
    assert render_caption_template(
        DEFAULT_AUDIO_CAPTION_TEMPLATE,
        {"TRANSCRIPT": "speech", "SOUND_CAPTION": ""},
    ) == "speech"
    assert render_caption_template(
        "{{VIDEO_CAPTION}}\n\n{{AUDIO_CAPTION}}\n{{FILENAME}}",
        {
            "VIDEO_CAPTION": "A  video\nsecond line",
            "AUDIO_CAPTION": "sound",
            "FILENAME": "klip_\u0130stanbul",
        },
    ) == "A  video\nsecond line\n\nsound\nklip_\u0130stanbul"
    assert render_caption_template("before {{EMPTY}} after", {"EMPTY": ""}) == "before after"
    assert render_caption_template("({{EMPTY}})", {"EMPTY": ""}) == "()"
    assert render_caption_template("{{VIDEO_CAPTION}}\n\n{{AUDIO_CAPTION}}", {
        "VIDEO_CAPTION": "video", "AUDIO_CAPTION": ""
    }) == "video"


def test_transcript_styles_use_clip_local_timestamps() -> None:
    source = {
        "segments": [
            {"start": 9.5, "end": 10.5, "text": " first   "},
            {"start": 10.5, "end": 11.75, "text": "second"},
        ]
    }
    assert render_transcript(source, "plain", start_s=10.0, end_s=12.0) == "first second"
    assert render_transcript(source, "lines", start_s=10.0, end_s=12.0) == "first\nsecond"
    assert render_transcript(source, "timestamped", start_s=10.0, end_s=12.0) == (
        "[00:00.0 - 00:00.5] first\n[00:00.5 - 00:01.8] second"
    )


@pytest.mark.parametrize(
    ("main", "tier", "expected"),
    [
        ("qwen3_omni_instruct_int4", 24, "qwen3_omni_captioner_int4"),
        ("qwen3_omni_thinking_int8", 48, "qwen3_omni_captioner_int8"),
        ("qwen3_omni_instruct_bf16", 24, "qwen3_omni_captioner_bf16"),
        ("qwen3_omni_instruct_gguf_q4", 24, "qwen3_omni_captioner_gguf_q4"),
        ("qwen3_omni_thinking_gguf_q8", 48, "qwen3_omni_captioner_gguf_q8"),
        ("timechat_bf16", 48, "qwen3_omni_captioner_int8"),
        ("avocado_bf16", 80, "qwen3_omni_captioner_bf16"),
    ],
)
def test_auto_captioner_variant_mapping(main: str, tier: int, expected: str) -> None:
    assert auto_captioner_variant(main, vram_tier=tier) == expected


def test_job_spec_maps_and_round_trips_dataset_caption_settings(tmp_path: Path) -> None:
    spec = JobSpec.from_settings(
        _settings(
            audio_caption_source="both",
            video_caption_source="existing",
            audio_caption_model_key="qwen3_omni_captioner_int8",
            audio_caption_transcript_style="timestamped",
            audio_caption_template="T={{TRANSCRIPT}} S={{SOUND_CAPTION}}",
            caption_write_merged=False,
            caption_merge_template="{{FILENAME}}: {{VIDEO_CAPTION}} / {{AUDIO_CAPTION}}",
            audio_caption_empty_policy="placeholder",
            audio_caption_empty_text="silence",
            batch_save_next_to_source=True,
        ),
        [],
        OutputSpec(kind="batch", outputs_root=tmp_path),
    )
    assert spec.audio_caption.source == "both"
    assert spec.audio_caption.video_source == "existing"
    assert spec.audio_caption.model_key == "qwen3_omni_captioner_int8"
    assert spec.audio_caption.transcript_style == "timestamped"
    assert spec.audio_caption.write_merged is False
    assert spec.audio_caption.empty_policy == "placeholder"
    assert spec.output.save_next_to_source is True
    assert spec.transcript.formats == ()
    assert JobSpec.from_json(spec.to_json()) == spec


def test_job_spec_preserves_explicitly_empty_caption_templates(tmp_path: Path) -> None:
    spec = JobSpec.from_settings(
        _settings(
            audio_caption_template="",
            caption_merge_template="",
            audio_caption_empty_text="",
        ),
        [],
        OutputSpec(outputs_root=tmp_path),
    )

    assert spec.audio_caption.template == ""
    assert spec.audio_caption.merge_template == ""
    assert spec.audio_caption.empty_text == ""


def test_caption_unit_paths_cover_single_mirror_next_to_source_and_segments(tmp_path: Path) -> None:
    single = caption_unit_paths(tmp_path / "outputs" / "0001_model", "\u043a\u043b\u0438\u043f")
    assert single.video == tmp_path / "outputs" / "0001_model" / "video_caption" / "\u043a\u043b\u0438\u043f.txt"
    assert single.audio.parent.name == "audio_caption"
    assert caption_unit_paths(tmp_path, "take.v2").merged.name == "take.v2.txt"
    source = tmp_path / "source"
    media = source / "nested" / "clip.mp4"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"")
    mirror_spec = JobSpec(
        inputs=[InputItem(media)],
        output=OutputSpec(kind="batch", batch_output_dir=tmp_path / "out", source_root=source),
    )
    resolved = _resolve_inputs(mirror_spec)
    _assign_batch_outputs(mirror_spec, resolved)
    assert resolved[0].out_dir == tmp_path / "out" / "nested"
    next_spec = JobSpec(
        inputs=[InputItem(media)],
        output=OutputSpec(
            kind="batch",
            batch_output_dir=tmp_path / "unused",
            source_root=source,
            save_next_to_source=True,
        ),
    )
    next_resolved = _resolve_inputs(next_spec)
    _assign_batch_outputs(next_spec, next_resolved)
    assert next_resolved[0].out_dir == media.parent
    entry = SimpleNamespace(out_dir=tmp_path / "run", stem="movie")
    assert _record_output_location(entry, {"index": 1, "persistent_clip": False}, 2) == (
        tmp_path / "run" / "movie_segments", "clip_0001"
    )
    persistent = tmp_path / "run" / "movie_clips" / "clip_0001.mp4"
    assert _record_output_location(
        entry,
        {"index": 1, "persistent_clip": True, "media_path": str(persistent)},
        1,
    ) == (persistent.parent, "clip_0001")


def test_single_split_run_writes_parts_merged_json_and_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "\u043a\u043b\u0438\u043f.wav"
    _write_wav(source)
    monkeypatch.setenv("VCAP_FAKE_CAPTIONER", "1")
    monkeypatch.setattr("vcap.whisper.client.run_transcription", _fake_transcription)
    spec = JobSpec.from_settings(
        _settings(caption_prefix="prefix ", caption_suffix=" suffix"),
        [InputItem(source)],
        OutputSpec(outputs_root=tmp_path / "outputs"),
    )

    result = run_job(spec, None, CancelToken())

    assert result.counts["done"] == 1
    assert result.counts["audio_captions"] == 1
    item = result.items[0]
    video = Path(str(item.video_caption_path))
    audio = Path(str(item.audio_caption_path))
    merged = Path(str(item.merged_caption_path))
    assert video.parent.name == "video_caption"
    assert audio.parent.name == "audio_caption"
    assert video.read_text(encoding="utf-8").startswith("prefix ")
    assert audio.read_text(encoding="utf-8") == "hello world\n"
    assert merged.read_text(encoding="utf-8") == (
        video.read_text(encoding="utf-8").rstrip("\n") + "\n\nhello world\n"
    )
    assert not list(Path(result.run_dir).glob("*_transcript.*"))
    structured = json.loads(Path(item.outputs["json"]).read_text(encoding="utf-8"))
    assert structured["video_caption"] == video.read_text(encoding="utf-8").rstrip("\n")
    assert structured["audio_caption"] == "hello world"
    assert structured["merged_caption"] == merged.read_text(encoding="utf-8").rstrip("\n")
    assert structured["transcript"]["text"] == "hello world"
    metadata = json.loads(Path(result.metadata_path).read_text(encoding="utf-8"))
    record = metadata["items_results"][0]
    assert record["video_caption_path"] == str(video)
    assert record["audio_caption_path"] == str(audio)
    assert record["merged_caption_path"] == str(merged)
    assert record["audio_caption_source"] == "whisper"


@pytest.mark.parametrize("save_next", [False, True])
def test_batch_split_layout_mirrors_or_saves_next_to_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    save_next: bool,
) -> None:
    source_root = tmp_path / "source"
    source = source_root / "nested" / "clip.wav"
    _write_wav(source)
    output_root = tmp_path / "captions"
    monkeypatch.setenv("VCAP_FAKE_CAPTIONER", "1")
    monkeypatch.setattr("vcap.whisper.client.run_transcription", _fake_transcription)
    spec = JobSpec.from_settings(
        _settings(output_formats=["txt"]),
        [InputItem(source)],
        OutputSpec(
            kind="batch",
            outputs_root=tmp_path / "runs",
            batch_output_dir=output_root,
            source_root=source_root,
            recursive=True,
            overwrite=True,
            save_next_to_source=save_next,
        ),
    )

    result = run_job(spec, None, CancelToken())

    expected = source.parent if save_next else output_root / "nested"
    assert Path(result.items[0].merged_caption_path or "") == expected / "clip.txt"
    assert (expected / "video_caption" / "clip.txt").is_file()
    assert (expected / "audio_caption" / "clip.txt").is_file()
    assert Path(result.metadata_path).parent == Path(result.run_dir)
    assert list_recent_runs(tmp_path / "runs")[0].preview.startswith("A canned caption")


def test_saved_segment_folder_is_a_split_caption_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "movie.wav"
    _write_wav(source, 0.8)
    monkeypatch.setenv("VCAP_FAKE_CAPTIONER", "1")
    monkeypatch.setattr("vcap.whisper.client.run_transcription", _fake_transcription)

    def fake_extract(_source: Any, target: Any, **_kwargs: Any) -> Path:
        output = Path(target)
        _write_wav(output, 0.2)
        return output

    monkeypatch.setattr("vcap.pipeline.runner.extract_audio", fake_extract)
    spec = JobSpec.from_settings(
        _settings(
            segment_mode="fixed",
            fixed_chunk_s=0.25,
            sub_split_overlap_s=0.0,
            output_formats=["txt"],
            save_clips=True,
        ),
        [InputItem(source)],
        OutputSpec(outputs_root=tmp_path / "outputs", save_clips=True),
    )

    result = run_job(spec, None, CancelToken())

    item = result.items[0]
    assert item.status == "done"
    assert len(item.segments) >= 2
    clip_dir = Path(result.run_dir) / "movie_clips"
    for record in item.segments:
        if record.get("status") != "done":
            continue
        stem = f"clip_{int(record['index']):04d}"
        assert (clip_dir / f"{stem}.wav").is_file()
        assert (clip_dir / f"{stem}.txt").is_file()
        assert (clip_dir / "video_caption" / f"{stem}.txt").is_file()
        audio_path = clip_dir / "audio_caption" / f"{stem}.txt"
        assert audio_path.is_file() is (not bool(record.get("no_speech")))
    assert (Path(result.run_dir) / "video_caption" / "movie.txt").is_file()
    assert (Path(result.run_dir) / "audio_caption" / "movie.txt").is_file()


def test_sound_caption_phase_switches_to_captioner_and_records_windows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "sound.wav"
    _write_wav(source)
    monkeypatch.setenv("VCAP_FAKE_CAPTIONER", "1")
    spec = JobSpec.from_settings(
        _settings(
            audio_caption_source="captioner",
            transcript_enabled=False,
            output_formats=["txt", "json"],
        ),
        [InputItem(source)],
        OutputSpec(outputs_root=tmp_path / "outputs"),
    )

    result = run_job(spec, None, CancelToken())

    item = result.items[0]
    assert item.status == "done"
    assert item.sound_caption_model == "qwen3_omni_captioner_int4"
    assert item.audio_windows == 1
    assert "audio_0001" in Path(item.audio_caption_path or "").read_text(encoding="utf-8")
    structured = json.loads(Path(item.outputs["json"]).read_text(encoding="utf-8"))
    assert "audio_0001" in structured["audio_caption"]
    run_log = Path(result.run_dir, "run_log.txt").read_text(encoding="utf-8")
    assert "Phase: sound captions" in run_log
    assert "1 Captioner audio window(s)" in run_log


def test_existing_mode_bootstraps_clean_video_caption_and_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "dataset" / "clip.wav"
    _write_wav(source)
    source.with_suffix(".txt").write_text("clean video\n", encoding="utf-8")
    monkeypatch.setattr("vcap.whisper.client.run_transcription", _fake_transcription)
    load_calls: list[str] = []
    monkeypatch.setattr(
        "vcap.models.loader.load_model",
        lambda key, **_kwargs: load_calls.append(key),
    )
    spec = JobSpec.from_settings(
        _settings(video_caption_source="existing", output_formats=["txt", "json"]),
        [InputItem(source)],
        OutputSpec(
            kind="batch",
            outputs_root=tmp_path / "runs",
            batch_output_dir=tmp_path / "unused",
            source_root=source.parent,
            overwrite=True,
            save_next_to_source=True,
        ),
    )

    first = run_job(spec, None, CancelToken())
    first_text = source.with_suffix(".txt").read_text(encoding="utf-8")
    second = run_job(spec, None, CancelToken())

    assert load_calls == []
    assert first.counts["done"] == second.counts["done"] == 1
    assert (source.parent / "video_caption" / "clip.txt").read_text(encoding="utf-8") == "clean video\n"
    assert first_text == "clean video\n\nhello world\n"
    assert source.with_suffix(".txt").read_text(encoding="utf-8") == first_text
    assert first_text.count("hello world") == 1
    structured = json.loads(Path(second.items[0].outputs["json"]).read_text(encoding="utf-8"))
    assert structured["transcript"]["text"] == "hello world"


def test_existing_mode_without_video_caption_saves_audio_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "dataset" / "orphan.wav"
    _write_wav(source)
    monkeypatch.setattr("vcap.whisper.client.run_transcription", _fake_transcription)
    spec = JobSpec.from_settings(
        _settings(video_caption_source="existing", output_formats=["txt"]),
        [InputItem(source)],
        OutputSpec(
            kind="batch",
            outputs_root=tmp_path / "runs",
            batch_output_dir=tmp_path / "unused",
            source_root=source.parent,
            overwrite=True,
            save_next_to_source=True,
        ),
    )

    result = run_job(spec, None, CancelToken())

    item = result.items[0]
    assert item.status == "done"
    assert "audio caption saved separately" in item.message
    assert (source.parent / "audio_caption" / "orphan.txt").is_file()
    assert not source.with_suffix(".txt").exists()
    assert not (source.parent / "video_caption" / "orphan.txt").exists()


@pytest.mark.parametrize(
    ("policy", "audio_text", "expected_no_speech"),
    [("skip", None, 1), ("placeholder", "No speech.\n", 1)],
)
def test_empty_audio_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    policy: str,
    audio_text: str | None,
    expected_no_speech: int,
) -> None:
    source = tmp_path / f"{policy}.wav"
    _write_wav(source)
    monkeypatch.setenv("VCAP_FAKE_CAPTIONER", "1")
    monkeypatch.setattr("vcap.whisper.client.run_transcription", _empty_transcription)
    spec = JobSpec.from_settings(
        _settings(audio_caption_empty_policy=policy, output_formats=["txt"]),
        [InputItem(source)],
        OutputSpec(outputs_root=tmp_path / "outputs"),
    )

    result = run_job(spec, None, CancelToken())

    item = result.items[0]
    video_text = Path(item.video_caption_path or "").read_text(encoding="utf-8")
    merged_text = Path(item.merged_caption_path or "").read_text(encoding="utf-8")
    if audio_text is None:
        assert merged_text == video_text
        assert item.audio_caption_path is None
    else:
        assert Path(item.audio_caption_path or "").read_text(encoding="utf-8") == audio_text
        assert merged_text == video_text.rstrip("\n") + "\n\n" + audio_text
    assert result.counts["no_speech"] == expected_no_speech


@pytest.mark.parametrize(
    ("settings", "existing_files", "expected"),
    [
        ({"audio_caption_source": "none"}, ("merged",), True),
        ({"audio_caption_source": "whisper", "caption_write_merged": True}, ("merged",), True),
        ({"audio_caption_source": "whisper", "caption_write_merged": False}, ("video", "audio"), True),
        (
            {
                "audio_caption_source": "whisper",
                "video_caption_source": "existing",
                "caption_write_merged": True,
            },
            ("video", "audio", "merged"),
            True,
        ),
        (
            {
                "audio_caption_source": "whisper",
                "video_caption_source": "existing",
                "caption_write_merged": True,
            },
            ("audio",),
            True,
        ),
        ({"audio_caption_source": "whisper", "caption_write_merged": False}, ("video",), False),
    ],
)
def test_batch_skip_rules(
    tmp_path: Path,
    settings: dict[str, Any],
    existing_files: tuple[str, ...],
    expected: bool,
) -> None:
    spec = JobSpec.from_settings(
        _settings(**settings),
        [],
        OutputSpec(kind="batch", batch_output_dir=tmp_path, overwrite=False),
    )
    paths = caption_unit_paths(tmp_path, "clip")
    for name in existing_files:
        path = getattr(paths, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
    entry = SimpleNamespace(
        status="pending", out_dir=tmp_path, stem="clip", message="", path=None
    )

    _apply_batch_skip(spec, [entry])

    assert (entry.status == "skipped") is expected
    if expected:
        assert "already exists" in entry.message


def test_existing_batch_skip_does_not_require_clean_part_once_outputs_exist(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source" / "clip.wav"
    _write_wav(source)
    source.with_suffix(".txt").write_text("legacy video", encoding="utf-8")
    out_dir = tmp_path / "output"
    paths = caption_unit_paths(out_dir, "clip")
    paths.audio.parent.mkdir(parents=True)
    paths.audio.write_text("audio", encoding="utf-8")
    paths.merged.write_text("legacy video\n\naudio", encoding="utf-8")
    spec = JobSpec.from_settings(
        _settings(video_caption_source="existing"),
        [],
        OutputSpec(kind="batch", batch_output_dir=out_dir, overwrite=False),
    )
    entry = SimpleNamespace(
        status="pending", out_dir=out_dir, stem="clip", message="", path=source
    )

    _apply_batch_skip(spec, [entry])

    assert entry.status == "skipped"
    assert str(paths.merged) in entry.message


def test_split_result_summary_reports_zero_audio_captions() -> None:
    result = SimpleNamespace(
        items=[],
        counts={"done": 1, "skipped": 0, "failed": 0, "audio_captions": 0},
        elapsed=0.25,
    )

    _label, message, _style, _eta = _result_summary(result)

    assert "audio captions: 0" in message


def test_captioner_audio_is_split_into_at_most_30_second_windows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media = tmp_path / "long.wav"
    media.write_bytes(b"audio")
    monkeypatch.setattr(
        "vcap.pipeline.runner.probe_media",
        lambda _path: SimpleNamespace(has_audio=True, duration=65.0),
    )
    extracted: list[tuple[float, float]] = []

    def fake_extract(_src: Any, dst: Any, **kwargs: Any) -> Path:
        extracted.append((float(kwargs["start"]), float(kwargs["end"])))
        target = Path(dst)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"wav")
        assert kwargs["sample_rate"] == 16_000
        assert kwargs["mono"] is True
        assert isinstance(kwargs["cancel_token"], CancelToken)
        return target

    monkeypatch.setattr("vcap.pipeline.runner.extract_audio", fake_extract)

    class Captioner:
        def __init__(self) -> None:
            self.calls = 0

        def caption(self, *_args: Any, **_kwargs: Any) -> Any:
            self.calls += 1
            return SimpleNamespace(text=f"window {self.calls}", cancelled=False)

    captioner = Captioner()
    session = SimpleNamespace(ensure=lambda: captioner)
    spec = JobSpec.from_settings(
        _settings(audio_caption_source="captioner"),
        [],
        OutputSpec(outputs_root=tmp_path),
    )
    emitter = SimpleNamespace(log=lambda *_args, **_kwargs: None)

    text, count = _caption_sound_windows(
        spec,
        session,
        media,
        0.0,
        65.0,
        tmp_path / "work",
        emitter,
        CancelToken(),
        unit_label="long.wav",
    )

    assert count == 3
    assert extracted == [(0.0, 30.0), (30.0, 60.0), (60.0, 65.0)]
    assert all(end - start <= 30.0 for start, end in extracted)
    assert text == "window 1 window 2 window 3"


def test_shipped_dataset_presets_load_and_define_new_keys(tmp_path: Path) -> None:
    store = PresetStore(tmp_path / "user", PRESETS_DEFAULT_DIR)
    names = [
        "Dataset clips - video + audio captions (Qwen3-Omni + Whisper)",
        "Dataset clips - add Whisper audio captions to existing captions",
        "Dataset clips - video + sound captions (Qwen3-Omni + Captioner)",
    ]
    required = {
        "audio_caption_source",
        "video_caption_source",
        "audio_caption_model_key",
        "audio_caption_transcript_style",
        "audio_caption_template",
        "caption_write_merged",
        "caption_merge_template",
        "audio_caption_empty_policy",
        "audio_caption_empty_text",
        "batch_save_next_to_source",
    }
    for name in names:
        settings = store.load(name, mark_last_used=False)
        assert required <= set(settings)
        assert settings["batch_save_next_to_source"] is True
        assert settings["transcript_formats"] == []
        assert settings["output_formats"] == ["txt"]


def test_editor_scans_and_rebuilds_caption_parts(tmp_path: Path) -> None:
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"")
    main = tmp_path / "clip.txt"
    main.write_text("old video\n\naudio", encoding="utf-8")
    video = tmp_path / "video_caption" / "clip.txt"
    audio = tmp_path / "audio_caption" / "clip.txt"
    video.parent.mkdir()
    audio.parent.mkdir()
    video.write_text("old video\n", encoding="utf-8")
    audio.write_text("audio\n", encoding="utf-8")

    items = scan_editor_folder(tmp_path)

    assert len(items) == 1
    assert items[0]["video_caption"] == "old video"
    assert items[0]["audio_caption"] == "audio"
    assert caption_parts_payload({"items": items, "selected_index": 0}) == ("old video", "audio")
    merged = rebuild_caption_parts_after_regeneration(
        {"caption_merge_template": "{{VIDEO_CAPTION}}\n\n{{AUDIO_CAPTION}}"},
        items[0],
        "new video",
    )
    assert merged == "new video\n\naudio"
    assert video.read_text(encoding="utf-8") == "new video\n"
    assert main.read_text(encoding="utf-8") == "new video\n\naudio\n"


def test_folder_light_scan_reports_caption_coverage(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    media = source / "nested" / "clip.mp4"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"")
    target = output / "nested"
    (target / "audio_caption").mkdir(parents=True)
    (target / "clip.txt").write_text("merged", encoding="utf-8")
    (target / "audio_caption" / "clip.txt").write_text("audio", encoding="utf-8")

    selected, summary = _folder_scan(
        str(source),
        True,
        str(output),
        False,
        include_kinds=["video"],
        include_caption_coverage=True,
    )

    assert selected == [str(media)]
    assert "1 media files" in summary
    assert "1 already captioned (<stem>.txt)" in summary
    assert "1 with audio captions (audio_caption/)" in summary


def test_build_app_exposes_dataset_caption_controls_and_live_state() -> None:
    demo = build_app()
    try:
        context = demo.vcap_context
        entries = {entry.key: entry for entry in context.settings_registry.entries()}
        for key in (
            "audio_caption_source",
            "video_caption_source",
            "audio_caption_model_key",
            "audio_caption_transcript_style",
            "audio_caption_template",
            "caption_write_merged",
            "caption_merge_template",
            "audio_caption_empty_policy",
            "audio_caption_empty_text",
            "batch_save_next_to_source",
        ):
            assert key in entries
            assert entries[key].description
        config = demo.get_config_file()
        elem_ids = {
            component.get("props", {}).get("elem_id")
            for component in config["components"]
        }
        assert "vc_audio_caption_layout" in elem_ids
        assert "vc_editor_caption_parts" in elem_ids
        assert "vc_caption_result_files" in elem_ids
        handler = context.states["audio_caption_control_handler"]
        updates = handler("both", False, "placeholder")
        assert updates[0]["interactive"] is True
        assert updates[1]["interactive"] is True
        assert updates[2]["interactive"] is True
        assert updates[7]["interactive"] is True
        assert "video_caption/<name>.txt" in updates[-1]
        for entry in context.preset_store.list_presets():
            if not entry.is_default or not entry.name.startswith("Dataset clips -"):
                continue
            loaded = context.preset_store.load(entry.name, mark_last_used=False)
            coerced, warnings = context.settings_registry.coerce(loaded)
            assert not warnings
            assert coerced["audio_caption_source"] in {"whisper", "both"}
            assert coerced["batch_save_next_to_source"] is True
        recovered, warnings = present_recovery_settings(
            {
                "settings": {
                    "audio_caption_source": "both",
                    "video_caption_source": "existing",
                    "batch_save_next_to_source": True,
                }
            },
            context.settings_registry,
            available_gpu_indices=[],
        )
        assert not warnings
        assert recovered["audio_caption_source"] == "both"
        assert recovered["video_caption_source"] == "existing"
        assert recovered["batch_save_next_to_source"] is True
    finally:
        demo.vcap_context.pipeline_client.shutdown()
