from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np

from vcap.core.progress import ProgressEvent
from vcap.core.subprocess_runner import CancelToken
from vcap.pipeline.job import InputItem, JobSpec, OutputSpec
from vcap.pipeline.runner import run_job


class _Sink:
    def __init__(self) -> None:
        self.logs: list[str] = []
        self.progress: list[str] = []

    def on_log(self, message: str, **_kwargs) -> None:
        self.logs.append(str(message))

    def on_progress(self, event: ProgressEvent) -> None:
        self.progress.append(event.message)

    def on_item(self, _event: ProgressEvent) -> None:
        pass


def _silent_wav(path: Path, seconds: float = 1.6) -> None:
    rate = 16_000
    samples = np.zeros(int(rate * seconds), dtype="<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(samples.tobytes())


def test_fake_pipeline_writes_summary_and_json_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "lecture.wav"
    _silent_wav(source)
    monkeypatch.setenv("VCAP_FAKE_CAPTIONER", "1")
    monkeypatch.setenv("VCAP_FAKE_CAPTION_TEXT", "A segment caption.")
    sink = _Sink()
    spec = JobSpec.from_settings(
        {
            "model_key": "qwen3_omni_instruct_int4",
            "segment_mode": "fixed",
            "fixed_chunk_s": 0.6,
            "summarize_segments": True,
            "summary_prompt": "Summarize these captions in {{LANGUAGE}}.",
            "language": "English",
            "output_formats": ["txt", "json"],
            "keep_model_loaded": False,
        },
        [InputItem(source)],
        OutputSpec(outputs_root=tmp_path / "runs"),
    )
    result = run_job(spec, sink, CancelToken())
    item = result.items[0]

    assert item.status == "done"
    assert len([record for record in item.segments if record["status"] == "done"]) >= 2
    assert item.summary
    assert item.summary_usage["new_tokens"] > 0
    assert item.summary_timing["total_s"] >= 0
    summary_path = Path(item.outputs["summary"])
    assert summary_path.name == "lecture_summary.txt"
    assert summary_path.read_text(encoding="utf-8").strip() == item.summary
    structured = json.loads(Path(item.outputs["json"]).read_text(encoding="utf-8"))
    assert structured["summary"] == item.summary
    assert any(message.startswith("Summarizing ") for message in sink.progress)
