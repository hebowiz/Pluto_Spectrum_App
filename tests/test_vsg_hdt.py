from dataclasses import replace
from pathlib import Path

import numpy as np

from pluto_protocol.bluetooth.hdt import HDTRate, hdt_definition, map_hdt_symbols
from pluto_vsg.engine import BluetoothHDTWaveformEngine
from pluto_vsg.model import ModulationKind, validate_project
from pluto_vsg.persistence import project_from_dict, project_to_dict
from pluto_vsg.profiles import bluetooth_hdt_fields, bluetooth_hdt_project
from pluto_sa.vsa.sources import FileIQSource


def _project(rate: HDTRate, *, payload_length: int = 16):
    base = bluetooth_hdt_project(rate)
    settings = replace(base.bluetooth_hdt, payload_length_bytes=payload_length)
    return replace(base, bluetooth_hdt=settings, fields=bluetooth_hdt_fields(settings))


def test_hdt_profiles_generate_each_supported_rate() -> None:
    engine = BluetoothHDTWaveformEngine()

    for rate in HDTRate:
        project = _project(rate)
        result = engine.generate(project)
        definition = hdt_definition(rate)

        assert validate_project(project) == ()
        assert project.sample_rate_hz == 16_000_000.0
        assert project.samples_per_symbol == 8
        assert project.center_frequency_hz == 2_440_000_000.0
        assert result.iq.dtype == np.complex64
        assert result.iq.size > 0
        assert np.isfinite(result.iq).all()
        assert np.max(np.abs(result.iq)) <= 1.0 + 1e-6
        assert 0.0 < result.metadata["digital_scale"] <= 1.0
        assert result.metadata["phy"] == rate.value
        assert result.metadata["modulation"] == definition.modulation
        assert result.metadata["payload_code_rate"] == definition.payload_code_rate


def test_hdt6_uses_spec_scaled_16qam_payload_mapping() -> None:
    project = _project(HDTRate.HDT6)

    assert project.fields[-1].modulation.kind == ModulationKind.QAM16
    labels = np.arange(16, dtype=np.uint8)
    bits = ((labels[:, None] >> np.arange(3, -1, -1)) & 1).reshape(-1)
    symbols = map_hdt_symbols(bits, HDTRate.HDT6)
    assert np.unique(np.round(symbols.real, 6)).size == 4
    assert np.unique(np.round(symbols.imag, 6)).size == 4
    assert np.unique(np.round(symbols, 6)).size == 16
    assert np.isclose(np.mean(np.abs(symbols) ** 2), 1.0)


def test_hdt_repetitions_are_complete_and_equally_spaced() -> None:
    project = replace(_project(HDTRate.HDT7_5), repeat_count=3)

    result = BluetoothHDTWaveformEngine().generate(project)

    period = int(result.metadata["period_sample_count"])
    starts = [start for start, _stop in result.metadata["packet_ranges_samples"]]
    assert result.iq.size == 3 * period
    assert np.diff(starts).tolist() == [period, period]


def test_hdt_project_json_round_trip() -> None:
    expected = replace(_project(HDTRate.HDT4), repeat_count=7)

    actual = project_from_dict(project_to_dict(expected))

    assert actual == expected


def test_checked_in_hdt7_5_fixture_matches_generator_and_loads_in_vsa() -> None:
    path = (
        Path(__file__).with_name("fixtures")
        / "bluetooth_hdt7_5_prbs9_16msps.npz"
    )
    project = _project(HDTRate.HDT7_5, payload_length=255)
    expected = BluetoothHDTWaveformEngine().generate(project)

    with np.load(path, allow_pickle=False) as fixture:
        np.testing.assert_array_equal(fixture["iq"], expected.iq)
        np.testing.assert_array_equal(
            fixture["payload_bits"], expected.metadata["payload_bits"]
        )
        np.testing.assert_array_equal(
            fixture["coded_payload_bits"],
            expected.metadata["coded_payload_bits"],
        )
        assert str(fixture["phy"]) == HDTRate.HDT7_5.value
        assert str(fixture["modulation"]) == "16QAM"
        assert int(fixture["payload_length_bytes"]) == 255

    recording = FileIQSource.load(path)
    assert recording.sample_rate_hz == 16_000_000.0
    assert recording.center_frequency_hz == 2_440_000_000.0
    np.testing.assert_array_equal(recording.iq, expected.iq)
