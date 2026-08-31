from __future__ import annotations

from vcap.core.registry import SettingsRegistry


def test_registration_order_and_positional_conversion() -> None:
    registry = SettingsRegistry()
    first = object()
    assert registry.register("count", first, 2, kind="int") is first
    registry.register("mode", object(), "fast", choices=["fast", "best"])
    assert registry.keys() == ["count", "mode"]
    assert registry.components()[0] is first
    assert registry.defaults() == {"count": 2, "mode": "fast"}
    assert registry.values_to_dict([4, "best"]) == {"count": 4, "mode": "best"}
    assert registry.dict_to_values({"count": 7}) == [7, "fast"]


def test_coerce_cast_clamp_choice_and_unknown() -> None:
    registry = SettingsRegistry()
    registry.register("enabled", object(), False, kind="bool")
    registry.register("frames", object(), 8, kind="int", minimum=2, maximum=32)
    registry.register("temperature", object(), 0.5, kind="float", minimum=0.0, maximum=1.0)
    registry.register("mode", object(), "auto", kind="str", choices=["auto", "fps"])
    registry.register("path", object(), "", kind="str", in_preset=False)
    values, warnings = registry.coerce(
        {
            "enabled": "yes",
            "frames": "100",
            "temperature": "-2",
            "mode": "invalid",
            "unknown": 1,
        }
    )
    assert values == {
        "enabled": True,
        "frames": 32,
        "temperature": 0.0,
        "mode": "auto",
        "path": "",
    }
    assert len(warnings) >= 4
    assert "path" not in registry.preset_subset(values)
    assert registry.metadata_subset(values)["frames"] == 32


def test_gpu_index_list_coercion_drops_invalid_choices_and_deduplicates() -> None:
    registry = SettingsRegistry()
    registry.register(
        "gpu_indices",
        object(),
        [],
        kind="list",
        choices=[0, 1, 2],
        in_preset=False,
        in_metadata=True,
    )

    values, warnings = registry.coerce({"gpu_indices": ["2", 0, "2", 99, "invalid"]})
    assert values["gpu_indices"] == [2, 0]
    assert len(warnings) == 2
    assert "gpu_indices" not in registry.preset_subset(values)
    assert registry.metadata_subset(values)["gpu_indices"] == [2, 0]
