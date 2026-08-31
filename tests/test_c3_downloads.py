from __future__ import annotations

import pytest

from vcap.models.downloads import _parse_status


def test_parse_status_json_protocol() -> None:
    payload = _parse_status(
        'VCAP_STATUS {"key":"timechat_int4","state":"downloading",'
        '"fraction":0.375,"bytes_done":3,"bytes_total":8,"message":"moving"}'
    )

    assert payload == {
        "key": "timechat_int4",
        "state": "downloading",
        "fraction": 0.375,
        "bytes_done": 3,
        "bytes_total": 8,
        "message": "moving",
    }


def test_parse_status_legacy_text_and_plain_percent() -> None:
    legacy = _parse_status("VCAP_STATUS timechat_int4 verifying Hashing 12.3%")
    plain = _parse_status("Downloading model.safetensors: 87.5% (7/8 GiB)")

    assert legacy is not None
    assert legacy["key"] == "timechat_int4"
    assert legacy["state"] == "verifying"
    assert legacy["fraction"] == pytest.approx(0.123)
    assert plain is not None
    assert plain["fraction"] == pytest.approx(0.875)
