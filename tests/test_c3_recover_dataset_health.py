from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from vcap.core.registry import SettingsRegistry
from vcap.ui.tabs.dataset_tab import trainer_clip_suggestion
from vcap.ui.tabs.health_tab import verify_local_files
from vcap.ui.tabs.recover_tab import present_recovery_settings, resolve_metadata_path


def _registry() -> SettingsRegistry:
    registry = SettingsRegistry()
    registry.register("theme", object(), "dark", section="settings", kind="str")
    registry.register("outputs_dir", object(), "default-out", section="settings", kind="str")
    registry.register("input_path", object(), "", section="input", kind="str")
    registry.register("model_key", object(), "timechat_int4", section="model", kind="str")
    registry.register("user_prompt", object(), "default prompt", section="prompt", kind="str")
    registry.register("max_frames", object(), 32, section="sampling", kind="int")
    return registry


def test_recover_folder_resolves_and_only_returns_present_keys(tmp_path: Path) -> None:
    run_folder = tmp_path / "outputs" / "batch_0007_unicöde"
    run_folder.mkdir(parents=True)
    metadata_path = run_folder / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "_meta": {"format": "secourses_vcap_metadata", "version": 1},
                "app_version": "test",
                "model_info": {},
                "settings": {"model_key": "qwen", "max_frames": "81"},
                "items_results": [],
                "timings": {},
            }
        ),
        encoding="utf-8",
    )

    assert resolve_metadata_path(run_folder) == metadata_path
    values, warnings = present_recovery_settings(run_folder, _registry())
    assert values == {"model_key": "qwen", "max_frames": 81}
    assert warnings == []
    assert not {"theme", "outputs_dir", "input_path", "user_prompt"} & values.keys()

    model_only, _ = present_recovery_settings(run_folder, _registry(), model_prompt_only=True)
    assert model_only == {"model_key": "qwen"}


def test_recover_maps_runtime_compile_and_recursive_settings_to_ui_controls() -> None:
    registry = SettingsRegistry()
    registry.register(
        "torch_compile_mode",
        object(),
        "default",
        section="runtime",
        kind="str",
        choices=["default", "max-autotune-no-cudagraphs"],
    )
    registry.register(
        "batch_recursive",
        object(),
        False,
        section="input",
        kind="bool",
    )

    values, warnings = present_recovery_settings(
        {
            "settings": {
                "compile_mode": "max-autotune-no-cudagraphs",
                "recursive": True,
            }
        },
        registry,
        available_gpu_indices=[],
    )

    assert values == {
        "torch_compile_mode": "max-autotune-no-cudagraphs",
        "batch_recursive": True,
    }
    assert warnings == []


def test_trainer_clip_suggestion_surfaces_frames_seconds_fps_and_valid_values() -> None:
    text, seconds = trainer_clip_suggestion("wan")

    assert "Suggested clip length: 81 frames = 5.06 s @ 16 fps" in text
    assert "valid:" in text
    assert seconds == pytest.approx(81 / 16)


def test_verify_local_files_checks_known_sha_and_reports_fraction(tmp_path: Path) -> None:
    payload = b"known GGUF fixture\n"
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(payload)
    expected_sha = hashlib.sha256(payload).hexdigest()
    events: list[dict] = []

    ok, detail = verify_local_files(
        [(model_file, len(payload), expected_sha)],
        lambda _message, event: events.append(event),
    )

    assert ok is True
    assert "size and SHA-256 verified" in detail
    assert events[0]["fraction"] == 0.0
    assert events[-1]["fraction"] == 1.0

    bad, mismatch = verify_local_files([(model_file, len(payload), "0" * 64)])
    assert bad is False
    assert "SHA-256 mismatch" in mismatch
