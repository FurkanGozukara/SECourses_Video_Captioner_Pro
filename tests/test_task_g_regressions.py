from __future__ import annotations

import json
import math
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vcap.core.outputs import list_recent_runs
from vcap.core.presets import PresetStore, merge_settings
from vcap.core.registry import SettingsRegistry
from vcap.core.subprocess_runner import CancelToken
from vcap.pipeline.job import InputItem, JobSpec, OutputSpec, PromptSpec
from vcap.pipeline.runner import run_job
from vcap.ui.tabs.caption_tab import run_history_records, run_history_rows
from vcap.ui.tabs.editor_tab import _selection_payload, new_editor_state


def _write_wav(path: Path, seconds: float = 0.4) -> None:
    rate = 8_000
    frames = bytearray()
    for index in range(int(rate * seconds)):
        value = int(1_200 * math.sin(index * 2 * math.pi * 440 / rate))
        frames.extend(value.to_bytes(2, "little", signed=True))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(frames)


def test_registry_and_preset_null_string_values_reach_ui_as_empty(tmp_path: Path) -> None:
    registry = SettingsRegistry()
    component = object()
    registry.register("system_prompt", component, "", kind="str")
    registry.register("user_prompt", object(), "default request", kind="str")
    registry.register("nullable_prompt", object(), None, kind="str")

    direct, direct_warnings = registry.coerce(
        {"system_prompt": None, "nullable_prompt": None}
    )
    assert SettingsRegistry._cast(None, "str") == ""
    assert direct["system_prompt"] == ""
    assert direct["nullable_prompt"] == ""
    assert direct_warnings == []

    store = PresetStore(tmp_path / "presets", tmp_path / "defaults")
    store.save(
        "null-system",
        {"system_prompt": None, "user_prompt": "Describe the clip."},
    )
    merged = merge_settings(store.load("null-system"), registry.defaults())
    coerced, warnings = registry.coerce(merged)
    ui_values = registry.dict_to_values(coerced)

    assert warnings == []
    assert ui_values[registry.keys().index("system_prompt")] == ""
    assert ui_values[registry.keys().index("user_prompt")] == "Describe the clip."


@pytest.mark.parametrize("value", [None, "", "None", " none ", "null", "NULL"])
def test_job_spec_normalizes_no_system_prompt_sentinels(
    tmp_path: Path,
    value: Any,
) -> None:
    spec = JobSpec.from_settings(
        {
            "model_key": "qwen3_omni_instruct_int8",
            "prompt_preset_id": "custom",
            "system_prompt": value,
            "user_prompt": "Describe this input.",
            "whisper_device": "cpu",
            "whisper_compute_type": "int8",
        },
        [],
        OutputSpec(outputs_root=tmp_path),
    )

    assert spec.prompt.system_prompt is None
    assert spec.settings["system_prompt"] == (None if value is None else "")
    assert JobSpec.from_dict(spec.to_dict()).prompt.system_prompt is None
    assert PromptSpec(system_prompt=value).system_prompt is None


@pytest.mark.parametrize(
    ("kind", "suffix", "visible_index"),
    [
        ("video", ".mp4", 0),
        ("audio", ".wav", 1),
        ("image", ".png", 2),
    ],
)
def test_editor_preview_hides_placeholder_for_visible_media(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    suffix: str,
    visible_index: int,
) -> None:
    from vcap.ui.tabs import editor_tab

    media = tmp_path / f"preview{suffix}"
    media.write_bytes(b"preview fixture")
    state = new_editor_state(tmp_path)
    state["items"] = [
        {
            "caption_path": str(tmp_path / "preview.txt"),
            "caption": "caption",
            "media_path": str(media),
            "source_media_path": str(media),
            "kind": kind,
        }
    ]
    state["selected_index"] = 0
    monkeypatch.setattr(
        editor_tab,
        "probe_media",
        lambda _path: SimpleNamespace(
            has_video=kind == "video",
            kind=kind,
            duration=1.0,
        ),
    )
    monkeypatch.setattr(editor_tab, "preview_safe_media", lambda path, _cache: Path(path))

    payload = _selection_payload(state, tmp_path / "cache")

    for index, update in enumerate(payload[:3]):
        assert update["visible"] is (index == visible_index)
    assert payload[3]["visible"] is False
    assert payload[3]["value"] == ""


def test_editor_first_frame_preview_replaces_stale_placeholder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vcap.ui.tabs import editor_tab

    media = tmp_path / "unplayable.mkv"
    frame = tmp_path / "unplayable.jpg"
    media.write_bytes(b"video fixture")
    frame.write_bytes(b"frame fixture")
    state = new_editor_state(tmp_path)
    state["items"] = [
        {
            "caption_path": str(tmp_path / "unplayable.txt"),
            "caption": "caption",
            "media_path": str(media),
            "source_media_path": str(media),
            "kind": "video",
        }
    ]
    state["selected_index"] = 0
    monkeypatch.setattr(
        editor_tab,
        "probe_media",
        lambda _path: SimpleNamespace(has_video=True, kind="video", duration=1.0),
    )
    monkeypatch.setattr(editor_tab, "preview_safe_media", lambda _path, _cache: frame)

    payload = _selection_payload(state, tmp_path / "cache")

    assert payload[2]["visible"] is True
    assert payload[3]["visible"] is True
    assert "first frame" in payload[3]["value"]
    assert "No preview selected" not in payload[3]["value"]


def _fake_transcription(request: dict[str, Any], **kwargs: Any) -> Any:
    from vcap.whisper.client import TranscriptionOutcome
    from vcap.whisper.engine import TranscriptResult, TranscriptSegment, TranscriptWord

    item = request["items"][0]
    tokens = [f"word{index}" for index in range(1, 43)]
    words = [
        TranscriptWord(
            index / 105.0,
            (index + 1) / 105.0,
            token if index == 0 else f" {token}",
            0.99,
        )
        for index, token in enumerate(tokens)
    ]
    result = TranscriptResult(
        segments=[TranscriptSegment(0, 0.0, 0.4, " ".join(tokens), words)],
        language="en",
        language_probability=0.98,
        duration_s=0.4,
        elapsed_s=0.12,
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
        path.write_text(result.text, encoding="utf-8")
        files.append(str(path))
    payload = {
        "event": "item_done",
        "item_index": item["index"],
        "files": files,
        "result": result.to_dict(),
        "skipped": False,
    }
    sink = kwargs.get("sink")
    if sink is not None:
        sink.on_item_done(payload)
    return TranscriptionOutcome(
        ok=True,
        items=[payload],
        results={item["index"]: result},
        elapsed_s=0.12,
        cancelled=False,
        error=None,
    )


def test_caption_transcript_injection_is_logged_and_summarized_in_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "speech.wav"
    _write_wav(source)
    monkeypatch.setenv("VCAP_FAKE_CAPTIONER", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr("vcap.whisper.client.run_transcription", _fake_transcription)
    monkeypatch.setattr("vcap.pipeline.runner.gpu.resource_snapshot", lambda _index: {})
    spec = JobSpec.from_settings(
        {
            "model_key": "qwen3_omni_instruct_int8",
            "prompt_preset_id": "custom",
            "system_prompt": "None",
            "user_prompt": "Use the exact dialogue: {{TRANSCRIPT}}",
            "output_formats": ["txt"],
            "max_frames": 0,
            "keep_model_loaded": False,
            "transcript_enabled": True,
            "transcript_formats": ["srt", "txt"],
            "transcript_inject_prompt": True,
            "whisper_model": "large-v1",
            "whisper_device": "cpu",
            "whisper_compute_type": "int8",
        },
        [InputItem(path=str(source))],
        OutputSpec(outputs_root=str(tmp_path / "outputs")),
    )

    result = run_job(spec, None, CancelToken())

    run_dir = Path(result.run_dir)
    log_text = (run_dir / "run_log.txt").read_text(encoding="utf-8")
    assert (
        "Transcript injected into the prompt for clip 1 "
        "(00:00.0-00:00.4): 42 words"
    ) in log_text
    metadata = json.loads(Path(result.metadata_path).read_text(encoding="utf-8"))
    transcript = metadata["extra"]["transcript"]
    assert transcript["model"] == "large-v1"
    assert transcript["language"] == "en"
    assert transcript["probability"] == pytest.approx(0.98)
    assert transcript["duration"] == pytest.approx(0.4)
    assert transcript["elapsed"] == pytest.approx(0.12)
    assert transcript["segment_count"] == 1
    assert transcript["word_count"] == 42
    assert len(transcript["files"]) == 2
    assert transcript["injected"] is True
    assert metadata["settings"]["system_prompt"] == ""


def test_whisper_run_history_is_transcribe_and_recoverable(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    run_dir = outputs / "0097_whisper"
    transcript_dir = tmp_path / "batch transcripts"
    run_dir.mkdir(parents=True)
    transcript_dir.mkdir()
    transcript_path = transcript_dir / "lesson ünicode.txt"
    transcript_path.write_text("A transcript preview from Whisper.", encoding="utf-8")
    metadata_path = run_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "_meta": {"format": "secourses_vcap_metadata", "version": 1},
                "app_version": "test",
                "model_info": {"alias": "large-v1"},
                "settings": {"whisper_model": "large-v1"},
                "items_results": [
                    {
                        "status": "done",
                        "path": "lesson.wav",
                        "files": [str(transcript_path)],
                    }
                ],
                "timings": {"elapsed_s": 1.0},
                "extra": {
                    "kind": "whisper_transcription",
                    "counts": {"done": 1, "failed": 0, "skipped": 0},
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    summaries = list_recent_runs(outputs)
    rows = run_history_rows(summaries)
    records = run_history_records(summaries)

    assert len(summaries) == 1
    assert summaries[0].kind == "transcribe"
    assert summaries[0].model_key == "large-v1"
    assert summaries[0].items == 1
    assert summaries[0].preview == "A transcript preview from Whisper."
    assert rows[0][1:4] == ["transcribe", "large-v1", 1]
    assert Path(records[0]["run_dir"]) == run_dir.resolve()
    assert Path(str(records[0]["metadata_path"])) == metadata_path.resolve()
