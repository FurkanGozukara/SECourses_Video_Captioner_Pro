from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from vcap.core.scene_split import SceneRange
from vcap.core.subprocess_runner import CancelToken
from vcap.models.base import MediaPart, PreprocessParams
from vcap.models.qwen3_omni import Qwen3OmniInstructCaptioner
from vcap.models.registry import MODEL_SPECS
from vcap.pipeline.job import InputItem, JobSpec, OutputSpec
from vcap.pipeline.runner import (
    _ResolvedInput,
    _log_job_generation_contracts,
    _materialize_segments,
    _required_capability,
)


class _Emitter:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def log(self, message, *_args, **_kwargs) -> None:
        self.messages.append(str(message))


def _spec(tmp_path: Path, model_key: str, max_frames: int) -> JobSpec:
    return JobSpec.from_settings(
        {
            "model_key": model_key,
            "max_frames": max_frames,
            "audio_sample_rate": 48_000,
        },
        [InputItem(tmp_path / "clip.mp4")],
        OutputSpec(outputs_root=tmp_path),
    )


def test_zero_frames_routes_qwen_video_to_extracted_audio(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"video")
    spec = _spec(tmp_path, "qwen3_omni_instruct_int4", 0)
    model = MODEL_SPECS["qwen3_omni_instruct"]
    info = SimpleNamespace(kind="video", has_video=True, has_audio=True)
    assert _required_capability(info, spec.inputs[0], spec, model) == ("audio", "video")

    captured: list[int] = []

    def fake_extract(_source, target, *, sample_rate, **_kwargs):
        captured.append(sample_rate)
        Path(target).write_bytes(b"wav")
        return Path(target)

    monkeypatch.setattr("vcap.pipeline.runner.extract_audio", fake_extract)
    entry = _ResolvedInput(0, 0, spec.inputs[0], source, None, info=info)
    entry.kind = "video"
    entry.capability = "audio"
    sources = _materialize_segments(
        spec,
        entry,
        source,
        info,
        [SceneRange(0.0, 1.0)],
        tmp_path / "work",
        _Emitter(),
        CancelToken(),
    )
    assert captured == [48_000]
    assert sources[0].path is not None and sources[0].path.suffix == ".wav"


def test_frame_clamp_and_zero_semantics_are_logged_once(tmp_path: Path) -> None:
    qwen = _spec(tmp_path, "qwen3_omni_instruct_int4", 0)
    emitter = _Emitter()
    _log_job_generation_contracts(
        qwen,
        MODEL_SPECS["qwen3_omni_instruct"],
        [
            SimpleNamespace(
                status="pending",
                capability="audio",
                info=SimpleNamespace(has_video=True, has_audio=True),
            )
        ],
        emitter,
    )
    assert emitter.messages.count(
        "Visual frames disabled (Maximum frames = 0): captioning the audio track only."
    ) == 1

    timechat = _spec(tmp_path, "timechat_int4", 0)
    assert timechat.preprocess.max_frames == 4
    emitter = _Emitter()
    _log_job_generation_contracts(timechat, MODEL_SPECS["timechat"], [], emitter)
    assert emitter.messages == [
        "Maximum frames 0 is below the TimeChat Captioner GRPO 7B minimum 4; using 4."
    ]

    capped = _spec(tmp_path, "timechat_int4", 999)
    emitter = _Emitter()
    _log_job_generation_contracts(capped, MODEL_SPECS["timechat"], [], emitter)
    assert emitter.messages == [
        "Maximum frames 999 exceeds the TimeChat Captioner GRPO 7B cap 160; using 160."
    ]


def test_total_pixel_cap_reduces_per_frame_pixels_and_warns(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"video")
    info = SimpleNamespace(
        has_video=True,
        has_audio=False,
        duration=50.0,
        nb_frames=None,
    )
    captured: dict[str, int] = {}

    def fake_frames(_path, **kwargs):
        captured["max_pixels"] = kwargs["max_pixels"]
        return SimpleNamespace(
            frames=np.zeros((100, 32, 32, 3), dtype=np.uint8),
            fps_effective=2.0,
        )

    monkeypatch.setattr("vcap.models.omni_common.probe_media", lambda _path: info)
    monkeypatch.setattr("vcap.models.omni_common.read_video_frames", fake_frames)
    captioner = Qwen3OmniInstructCaptioner()
    prepared = captioner._prepare_media(
        [MediaPart("video", source)],
        PreprocessParams(
            fps=2.0,
            max_frames=100,
            max_pixels=600_000,
            min_pixels=4096,
            use_audio_in_video=False,
            total_pixel_cap=10_000_000,
        ),
    )
    assert captured["max_pixels"] == 200_000
    assert any("Total pixel cap 10,000,000 reduced per-frame pixels" in warning for warning in prepared.warnings)
