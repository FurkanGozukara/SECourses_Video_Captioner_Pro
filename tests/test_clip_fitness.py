from __future__ import annotations

import pytest

from vcap.core.clip_fitness import (
    TRAINER_TARGETS,
    evaluate_clip,
    resolution_bucket_preview,
    sub_split_plan,
    suggest_clip_length,
)


def test_target_rules_and_duration_suggestions() -> None:
    assert TRAINER_TARGETS["wan"]["default_frames"] == 81
    assert TRAINER_TARGETS["ltx2"]["frame_multiple"] == 8
    assert TRAINER_TARGETS["minimax_h3"]["resolution_multiple"] == 32
    wan = suggest_clip_length("wan", 16)
    assert (81, pytest.approx(81 / 16)) in wan
    ltx = [frames for frames, _ in suggest_clip_length("ltx2", 25)]
    assert all((frames - 1) % 8 == 0 for frames in ltx)


def test_fitness_warning_buckets_and_subsplit() -> None:
    report = evaluate_clip(
        {"duration": 2.6, "fps": 16, "frames": 41, "width": 1920, "height": 1080},
        "wan",
        "480p",
    )
    assert not report.ok
    assert report.frames_available == 41 and report.frames_needed == 81
    assert report.warnings == ["will be dropped by trainer: only 41 frames, needs 81"]
    assert report.bucket[0] <= 832 and report.bucket[1] <= 480

    out_w, out_h, letterbox = resolution_bucket_preview(1000, 1000, "720p", "letterbox")
    assert (out_w, out_h) == (1280, 720)
    assert sum(letterbox["pad"]) > 0
    crop_w, crop_h, crop = resolution_bucket_preview(1000, 1000, "720p", "crop")
    assert (crop_w, crop_h) == (1280, 720)
    assert sum(crop["crop"]) > 0

    assert sub_split_plan(5, 2, 0.5) == [(0.0, 2.0), (1.5, 3.5), (3.0, 5.0)]

