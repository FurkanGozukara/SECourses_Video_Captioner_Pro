from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from tools.bench import benchmark


def test_summary_and_markdown_keep_eos_and_complete_metrics() -> None:
    measurements = [
        {
            "finish_reason": "eos",
            "peak_vram_gib": 19.0 + index,
            "prefill_tok_s": 100.0 + index,
            "decode_tok_s": 20.0 + index,
            "generated_tokens": 200 + index,
            "wall_clock_s": 30.0 + index,
            "caption_preview": "storm | caption",
        }
        for index in range(3)
    ]

    summary = benchmark.summarize("example_int4", 6_500_000_000, 5.25, 18.0, measurements)
    rendered = benchmark.markdown_table(summary)

    assert summary["finish_reason"] == "eos"
    assert summary["generated_tokens_mean"] == 201
    assert summary["wall_clock_s_mean"] == 31
    assert summary["peak_vram_gib"] == 21
    assert "Generated tokens" in rendered
    assert "Wall s/caption" in rendered
    assert "eos" in rendered
    assert "storm \\| caption" in rendered


def test_resolve_variant_supports_legacy_model_dir(tmp_path: Path) -> None:
    folder = tmp_path / "avocado_int4"
    folder.mkdir()
    args = Namespace(variant=None, model_dir=folder)

    variant, resolved = benchmark._resolve_variant(args)

    assert variant == "avocado_int4"
    assert resolved == folder.resolve()
