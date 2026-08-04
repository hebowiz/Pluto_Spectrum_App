import json

import pytest

from pluto_sa.vsa.persistence import (
    load_meas_config,
    load_pattern,
    save_meas_config,
    save_pattern,
)


def test_pattern_file_round_trip_is_versioned_and_human_readable(tmp_path) -> None:
    path = tmp_path / "access.vsapattern.json"
    save_pattern(
        path,
        name="Access",
        symbols=[0, 1, 1, 0, 1, 0, 0, 1],
        symbol_format="Binary",
    )

    assert load_pattern(path) == {
        "name": "Access",
        "symbol_format": "Binary",
        "symbols": [0, 1, 1, 0, 1, 0, 0, 1],
    }
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["schema"] == "pluto-vsa-pattern"
    assert document["version"] == 1


def test_meas_config_file_round_trip(tmp_path) -> None:
    path = tmp_path / "measurement.vsaconfig.json"
    settings = {
        "signal_description": {"modulation": "GFSK", "symbol_rate_hz": 1e6},
        "pattern_search": {"symbols": [0, 1, 0, 1]},
    }

    save_meas_config(path, settings)

    assert load_meas_config(path) == settings


def test_wrong_schema_is_rejected(tmp_path) -> None:
    path = tmp_path / "wrong.json"
    path.write_text(
        '{"schema":"pluto-vsa-meas-config","version":1,"settings":{}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not a pluto-vsa-pattern"):
        load_pattern(path)
