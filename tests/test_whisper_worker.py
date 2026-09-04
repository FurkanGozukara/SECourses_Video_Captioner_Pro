from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

from vcap.core.subprocess_runner import CancelToken, CancelledError, build_child_env
from vcap.whisper.client import build_request, run_transcription
from vcap.whisper.engine import TranscriptResult, TranscriptSegment, TranscriptWord
from vcap.whisper.params import TranscriptOutputOptions, WhisperParams


class FakeWhisperEngine:
    """Importable subprocess fixture selected with VCAP_WHISPER_FAKE_ENGINE."""

    def __init__(self, params, *, models_dir=None, log=None, progress=None, cancel_check=None):
        self.params = params
        self.models_dir = Path(models_dir or ".")
        self.log = log or (lambda *_args: None)
        self.progress = progress or (lambda _payload: None)
        self.cancel_check = cancel_check or (lambda: False)

    def ensure_model(self):
        target = self.models_dir / "whisper" / self.params.model
        target.mkdir(parents=True, exist_ok=True)
        self.progress(
            {
                "stage": "download",
                "fraction": 1.0,
                "bytes": 1,
                "total": 1,
                "speed_bps": 1.0,
                "message": "fake model ready",
                "file": "model.bin",
            }
        )
        return target

    def load(self):
        self.progress(
            {
                "stage": "runtime",
                "device": "cpu",
                "compute_type": "int8",
                "cuda_devices": 0,
                "cublas": False,
                "cudnn": False,
                "message": "fake runtime",
            }
        )
        self.progress(
            {
                "stage": "model_loaded",
                "model": self.params.model,
                "load_s": 0.01,
                "path": str(self.models_dir / "whisper" / self.params.model),
            }
        )

    def transcribe(self, media_path, *, on_segment=None):
        name = Path(media_path).name
        if "bad" in name:
            raise RuntimeError("synthetic item failure")
        if "slow" in name:
            for _ in range(200):
                if self.cancel_check():
                    raise CancelledError("synthetic cancellation")
                time.sleep(0.01)
        segment = TranscriptSegment(
            id=0,
            start=0.0,
            end=1.0,
            text=f"transcript for {name}",
            words=[TranscriptWord(0.0, 1.0, f" transcript for {name}", 1.0)],
        )
        self.progress(
            {
                "stage": "transcribe",
                "fraction": 1.0,
                "message": "Transcribing 00:01 / 00:01",
                "segments": 1,
                "elapsed_s": 0.01,
                "eta_s": 0.0,
            }
        )
        if on_segment is not None:
            on_segment(segment)
        return TranscriptResult(
            [segment], "en", 1.0, 1.0, 0.01, self.params.model, "int8", "cpu"
        )

    def unload(self):
        return None


def _worker_env() -> dict[str, str]:
    return build_child_env(
        extra={
            "VCAP_WHISPER_FAKE_ENGINE": "tests.test_whisper_worker:FakeWhisperEngine",
            "CUDA_VISIBLE_DEVICES": "0",
        }
    )


def _request(tmp_path: Path, names: list[str], *, skip_existing: bool = False) -> dict:
    items = []
    output = tmp_path / "output"
    output.mkdir(exist_ok=True)
    for index, name in enumerate(names):
        source = tmp_path / name
        source.write_bytes(b"fixture")
        items.append(
            {
                "index": index,
                "path": str(source),
                "out_dir": str(output),
                "stem": source.stem,
                "trim_start_s": 0.0,
                "trim_end_s": None,
            }
        )
    return build_request(
        WhisperParams(model="tiny", device="cpu", normalize_word_timestamps=False),
        TranscriptOutputOptions(formats=("txt",)),
        items,
        models_dir=tmp_path / "models",
        skip_existing=skip_existing,
    )


def test_worker_protocol_orders_events_and_continues_after_item_error(tmp_path: Path) -> None:
    request = _request(tmp_path, ["first.wav", "bad.wav", "last.wav"])
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    process = subprocess.run(
        [sys.executable, "-m", "vcap.whisper.worker", "--request", str(request_path)],
        cwd=Path(__file__).resolve().parents[1],
        env=_worker_env(),
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    events = [json.loads(line) for line in process.stdout.splitlines() if line.strip()]
    kinds = [event["event"] for event in events]

    assert process.returncode == 1
    assert kinds[:2] == ["runtime", "model_loaded"]
    assert kinds[-1] == "done"
    assert [event["item_index"] for event in events if event["event"] == "item_done"] == [0, 2]
    assert [event["item_index"] for event in events if event["event"] == "item_error"] == [1]
    assert events[-1]["items_done"] == 2
    assert events[-1]["items_failed"] == 1
    assert (tmp_path / "output" / "last.txt").is_file()


def test_worker_ensure_model_action(tmp_path: Path) -> None:
    request = build_request(
        WhisperParams(model="tiny", device="cpu"),
        TranscriptOutputOptions(),
        [],
        models_dir=tmp_path / "models",
        action="ensure_model",
    )
    request_path = tmp_path / "ensure.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    process = subprocess.run(
        [sys.executable, "-m", "vcap.whisper.worker", "--request", str(request_path)],
        cwd=Path(__file__).resolve().parents[1],
        env=_worker_env(),
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    events = [json.loads(line) for line in process.stdout.splitlines() if line.strip()]
    assert process.returncode == 0
    assert [event["event"] for event in events] == ["download", "done"]
    assert events[-1]["ok"] is True


def test_worker_probe_runtime_action_has_stable_payload(tmp_path: Path) -> None:
    request = build_request(
        WhisperParams(model="tiny", device="cpu"),
        TranscriptOutputOptions(),
        [],
        models_dir=tmp_path / "models",
        action="probe_runtime",
    )
    request_path = tmp_path / "probe.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    process = subprocess.run(
        [sys.executable, "-m", "vcap.whisper.worker", "--request", str(request_path)],
        cwd=Path(__file__).resolve().parents[1],
        env=_worker_env(),
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    events = [json.loads(line) for line in process.stdout.splitlines() if line.strip()]

    assert process.returncode == 0
    assert [event["event"] for event in events] == ["runtime", "done"]
    assert {
        "device",
        "compute_type",
        "cuda_devices",
        "cublas",
        "cudnn",
        "message",
    }.issubset(events[0])
    assert events[-1]["ok"] is True


class _Sink:
    def __init__(self) -> None:
        self.kinds: list[str] = []

    def on_log(self, _message, _level="info"):
        self.kinds.append("log")

    def on_download(self, _payload):
        self.kinds.append("download")

    def on_progress(self, _payload):
        self.kinds.append("progress")

    def on_segment(self, _payload):
        self.kinds.append("segment")

    def on_item_done(self, _payload):
        self.kinds.append("item_done")

    def on_item_error(self, _payload):
        self.kinds.append("item_error")


def test_client_assembles_results_and_forwards_events(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv(
        "VCAP_WHISPER_FAKE_ENGINE", "tests.test_whisper_worker:FakeWhisperEngine"
    )
    sink = _Sink()
    outcome = run_transcription(
        _request(tmp_path, ["one.wav", "two.wav"]),
        sink=sink,
        request_dir=tmp_path / "requests",
        timeout_s=30,
    )

    assert outcome.ok is True
    assert [item["item_index"] for item in outcome.items] == [0, 1]
    assert set(outcome.results) == {0, 1}
    assert outcome.results[0].text == "transcript for one.wav"
    assert {"progress", "segment", "item_done"}.issubset(sink.kinds)


def test_skip_existing_emits_skipped_item_done(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(
        "VCAP_WHISPER_FAKE_ENGINE", "tests.test_whisper_worker:FakeWhisperEngine"
    )
    request = _request(tmp_path, ["already.wav"], skip_existing=True)
    existing = tmp_path / "output" / "already.txt"
    existing.write_text("already there", encoding="utf-8")

    outcome = run_transcription(request, request_dir=tmp_path / "requests", timeout_s=30)

    assert outcome.ok is True
    assert outcome.items[0]["skipped"] is True
    assert outcome.results == {}


def test_client_cancel_sends_stdin_cancel(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(
        "VCAP_WHISPER_FAKE_ENGINE", "tests.test_whisper_worker:FakeWhisperEngine"
    )
    cancel = CancelToken()
    threading.Timer(0.15, cancel.cancel).start()

    outcome = run_transcription(
        _request(tmp_path, ["slow.wav"]),
        cancel=cancel,
        request_dir=tmp_path / "requests",
        timeout_s=30,
    )

    assert outcome.cancelled is True
    assert outcome.ok is False


def test_worker_trim_offsets_result_timestamps(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(
        "VCAP_WHISPER_FAKE_ENGINE", "tests.test_whisper_worker:FakeWhisperEngine"
    )
    source = tmp_path / "trim_source.wav"
    with wave.open(str(source), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\x00\x00" * 16_000 * 4)
    out_dir = tmp_path / "output"
    request = build_request(
        WhisperParams(model="tiny", device="cpu", normalize_word_timestamps=False),
        TranscriptOutputOptions(formats=("srt",)),
        [
            {
                "index": 0,
                "path": str(source),
                "out_dir": str(out_dir),
                "stem": "trimmed",
                "trim_start_s": 2.0,
                "trim_end_s": 3.5,
            }
        ],
        models_dir=tmp_path / "models",
    )

    outcome = run_transcription(request, request_dir=tmp_path / "requests", timeout_s=30)

    assert outcome.ok is True
    assert outcome.results[0].segments[0].start == 2.0
    assert outcome.results[0].segments[0].end == 3.0
    assert outcome.results[0].duration_s == 4.0
    assert "00:00:02,000 --> 00:00:03,000" in (out_dir / "trimmed.srt").read_text(
        encoding="utf-8"
    )
