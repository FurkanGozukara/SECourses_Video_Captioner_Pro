from __future__ import annotations

import json
from pathlib import Path

from vcap import PRESETS_DEFAULT_DIR


def test_shipped_presets_make_trigger_injection_explicit() -> None:
    paths = sorted(Path(PRESETS_DEFAULT_DIR).glob("*.json"))
    assert paths
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        settings = payload["settings"]
        is_lora = "character lora" in path.stem.casefold() or "motion lora" in path.stem.casefold()
        assert settings["trigger_mode"] == ("prefix" if is_lora else "none"), path.name
        assert settings["save_reasoning"] is True, path.name
        if "thinking" in str(settings.get("model_key", "")).casefold():
            assert settings["enable_thinking"] is True, path.name

