import json

import pytest

from pluto_sa.vsa.persistence import (
    load_mode_meas_config,
    load_meas_config,
    load_pattern,
    save_mode_meas_config,
    save_meas_config,
    save_pattern,
)


def test_mode_aware_config_round_trip_and_legacy_generic_compatibility(tmp_path) -> None:
    mode_path = tmp_path / "bluetooth.vsaconfig.json"
    save_mode_meas_config(
        mode_path,
        analysis_mode="bluetooth",
        settings={"protocol": "bluetooth.le", "phy": "LE 2M"},
    )
    assert load_mode_meas_config(mode_path) == (
        "bluetooth",
        {"protocol": "bluetooth.le", "phy": "LE 2M"},
    )

    legacy_path = tmp_path / "legacy.vsaconfig.json"
    save_meas_config(legacy_path, {"signal_description": {"modulation": "GFSK"}})
    assert load_mode_meas_config(legacy_path) == (
        "generic",
        {"signal_description": {"modulation": "GFSK"}},
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
