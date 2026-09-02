from __future__ import annotations

import json
from pathlib import Path

import pytest

from vcap.core.presets import PresetError, PresetStore, merge_settings


def _default_preset(path: Path, settings: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "_meta": {
                    "format": "secourses_vcap_preset",
                    "version": 1,
                    "app_version": "1.0.0",
                    "created_at": "now",
                    "modified_at": "now",
                },
                "settings": settings,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_default_protection_save_load_and_last_used(tmp_path: Path) -> None:
    user_dir = tmp_path / "user"
    default_dir = tmp_path / "defaults"
    _default_preset(default_dir / "Balanced.json", {"fps": 2})
    store = PresetStore(user_dir, default_dir)

    entries = store.list_presets()
    assert entries[0].name == "Balanced"
    assert entries[0].is_default
    with pytest.raises(PresetError):
        store.save("balanced", {"fps": 5})
    with pytest.raises(PresetError):
        store.delete("Balanced")

    saved = store.save("My 日本語 preset", {"fps": 3, "name": "vöyager"})
    assert store.exists(saved)
    assert store.get_last_used() == saved
    assert store.load(saved) == {"fps": 3, "name": "vöyager"}
    assert store.get_last_used() == saved
    payload = json.loads((user_dir / f"{saved}.json").read_text(encoding="utf-8"))
    assert payload["_meta"]["format"] == "secourses_vcap_preset"
    assert payload["_meta"]["version"] == 1


def test_delete_clears_last_used(tmp_path: Path) -> None:
    store = PresetStore(tmp_path / "user", tmp_path / "defaults")
    name = store.save("temporary", {"x": 1})
    assert store.delete(name)
    assert not store.exists(name)
    assert store.get_last_used() is None


def test_explicit_startup_default_is_first_and_used_without_marker(tmp_path: Path) -> None:
    default_dir = tmp_path / "defaults"
    user_dir = tmp_path / "user"
    _default_preset(default_dir / "01 Alphabetical.json", {"fps": 1})
    _default_preset(default_dir / "Default Video.json", {"fps": 2})
    store = PresetStore(
        user_dir,
        default_dir,
        default_preset_name="Default Video",
    )

    assert [entry.name for entry in store.list_presets()][:2] == [
        "Default Video",
        "01 Alphabetical",
    ]
    assert store.get_last_used() is None
    assert store.startup_preset_name() == "Default Video"


def test_startup_ignores_the_marker_and_loading_it_does_not_erase_it(tmp_path: Path) -> None:
    """A launch applies the shipped default; only Load last values reads the marker."""

    default_dir = tmp_path / "defaults"
    user_dir = tmp_path / "user"
    _default_preset(default_dir / "Default Video.json", {"fps": 2})
    store = PresetStore(user_dir, default_dir, default_preset_name="Default Video")
    mine = store.save("Mine", {"fps": 9})

    assert store.get_last_used() == mine
    # Startup picks the shipped default even though a marker exists...
    assert store.default_startup_name() == "Default Video"
    assert store.load("Default Video", mark_last_used=False) == {"fps": 2}
    # ...and reading it that way must leave the marker alone, or the button
    # would have nothing left to restore.
    assert store.get_last_used() == mine
    # The button's lookup still resolves to the remembered preset.
    assert store.startup_preset_name() == mine
    assert store.load(mine) == {"fps": 9}


def test_merge_forward_compatibility() -> None:
    defaults = {"known": 1, "missing": True}
    with pytest.warns(UserWarning, match="unknown"):
        merged = merge_settings({"known": 2, "old_key": "drop"}, defaults)
    assert merged == {"known": 2, "missing": True}
    assert merge_settings(None, defaults) == defaults
