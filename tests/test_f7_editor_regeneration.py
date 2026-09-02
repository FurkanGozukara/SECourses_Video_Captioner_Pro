from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vcap.prompts.presets import default_preset_for
from vcap.ui.app import build_app
from vcap.ui.tabs import editor_tab


def _video_probe() -> SimpleNamespace:
    return SimpleNamespace(
        has_video=True,
        has_audio=True,
        kind="video",
        duration=20.0,
    )


def _make_video_pair(root: Path) -> None:
    (root / "clip.mp4").write_bytes(b"test video placeholder")
    (root / "clip.txt").write_text("Existing caption.\n", encoding="utf-8")


def _choice_values(update: dict[str, Any]) -> set[str]:
    choices = update.get("choices") or []
    assert choices
    values = {
        str(choice[1] if isinstance(choice, (tuple, list)) else choice)
        for choice in choices
    }
    assert str(update.get("value") or "") in values
    return values


def test_scan_and_video_selection_populate_regeneration_prompts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _make_video_pair(tmp_path)
    monkeypatch.setattr(editor_tab, "probe_media", lambda _path: _video_probe())
    monkeypatch.setattr(
        editor_tab,
        "preview_safe_media",
        lambda path, _cache_dir: Path(path),
    )

    demo = build_app()
    try:
        context = demo.vcap_context
        binding = context.states["editor_regeneration_binding"]
        prompt_output_ids = [component._id for component in binding["prompt_outputs"]]
        assert [
            component._id
            for component in context.states["editor_open_binding"]["outputs"][-4:]
        ] == prompt_output_ids
        select_dependency = next(
            dependency
            for dependency in demo.get_config_file()["dependencies"]
            if getattr(demo.fns[dependency["id"]].fn, "__name__", "")
            == "select_handler"
        )
        assert select_dependency["outputs"][-4:] == prompt_output_ids

        for variant in ("qwen3_omni_instruct_int4", "timechat_int4"):
            scan_result = binding["scan_handler"](
                str(tmp_path),
                False,
                25,
                variant,
                None,
                "",
            )
            scanned_state = scan_result[0]
            scan_prompt_update, scan_value, scan_error, _ = scan_result[-4:]
            _choice_values(scan_prompt_update)
            assert scan_error == ""

            select_result = binding["select_handler"](
                scanned_state,
                variant,
                scan_value,
                scan_error,
                SimpleNamespace(index=(0, 0)),
            )
            selection_prompt_update, selection_value, selection_error, _ = (
                select_result[-4:]
            )
            selection_values = _choice_values(selection_prompt_update)
            assert selection_value in selection_values
            assert selection_error == ""
    finally:
        demo.vcap_context.pipeline_client.shutdown()


def test_regenerate_selected_empty_prompt_submits_family_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _make_video_pair(tmp_path)
    monkeypatch.setattr(editor_tab, "probe_media", lambda _path: _video_probe())
    demo = build_app()
    submitted: list[Any] = []

    def fake_run_job(spec: Any, _sink: Any) -> SimpleNamespace:
        submitted.append(spec)
        return SimpleNamespace(counts={"done": 1, "failed": 0})

    try:
        context = demo.vcap_context
        monkeypatch.setattr(context.pipeline, "run_job", fake_run_job)
        binding = context.states["editor_regeneration_binding"]
        scan_result = binding["scan_handler"](
            str(tmp_path),
            False,
            25,
            "qwen3_omni_instruct_int4",
            None,
            "",
        )
        state = scan_result[0]
        runtime_values = [
            entry.default for entry in context.settings_registry.entries()
        ]

        updates = list(
            binding["regenerate_handler"](
                state,
                "qwen3_omni_instruct_int4",
                "",
                "",
                *runtime_values,
            )
        )

        assert len(submitted) == 1
        assert submitted[0].prompt.preset_id == default_preset_for(
            "qwen3_omni_instruct",
            "video_audio",
        ).id
        assert updates[-1][1] == "Regenerated clip.txt. Review the diff, then keep or revert."
    finally:
        demo.vcap_context.pipeline_client.shutdown()
