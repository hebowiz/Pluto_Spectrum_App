from pathlib import Path

import numpy as np
import pytest

from pluto_sa.vsa.model import IQRecording, ModulationKind, SignalDescription
from pluto_sa.vsa.pattern import (
    BitOrdering,
    DemodulationSettings,
    KnownPattern,
    PatternAnalyzer,
    PatternSearchMode,
    PatternSearchSettings,
    ResultRangeSettings,
)
from pluto_sa.vsa.sources import FileIQSource, GeneratedIQSource
from pluto_sa.vsa.session import VSASession
from pluto_sa.vsa.profiles.bluetooth_br import access_code_bits


def _pattern_from_generated(recording, start: int, length: int) -> KnownPattern:
    symbols = np.asarray(recording.metadata["generated_symbols"])
    return KnownPattern(tuple(int(value) for value in symbols[start : start + length]))


@pytest.mark.parametrize("gaussian_bt", [None, 0.5])
def test_fsk_pattern_search_decodes_result_range(gaussian_bt):
    recording, signal = GeneratedIQSource.fsk(
        symbol_count=300,
        gaussian_bt=gaussian_bt,
        seed=77,
    )
    expected = np.asarray(recording.metadata["generated_symbols"])
    pattern = _pattern_from_generated(recording, 40, 32)

    result = PatternAnalyzer().search(
        recording,
        signal,
        PatternSearchSettings(
            pattern=pattern,
            mode=PatternSearchMode.ON,
            correlation_threshold_auto=False,
            iq_correlation_threshold=0.7,
        ),
        ResultRangeSettings(result_length=80, offset_symbols=0),
    )

    assert result.pattern_start_symbol == 40
    assert result.pattern_symbol_errors == 0
    np.testing.assert_array_equal(result.decoded_symbols, expected[40:120])


def test_qpsk_pattern_search_handles_carrier_phase_and_frequency_offset():
    recording, signal = GeneratedIQSource.psk(
        modulation=ModulationKind.QPSK,
        symbol_count=240,
        seed=19,
    )
    expected = np.asarray(recording.metadata["generated_symbols"])
    sample_index = np.arange(recording.sample_count)
    frequency_offset_hz = 12_500.0
    rotated = recording.iq * np.exp(
        1j
        * (
            1.1
            + 2.0
            * np.pi
            * frequency_offset_hz
            * sample_index
            / recording.sample_rate_hz
        )
    )
    recording = type(recording)(
        iq=rotated,
        sample_rate_hz=recording.sample_rate_hz,
        source="offset QPSK",
        metadata=recording.metadata,
    )
    pattern = _pattern_from_generated(recording, 35, 24)

    result = PatternAnalyzer().search(
        recording,
        signal,
        PatternSearchSettings(pattern=pattern, mode=PatternSearchMode.ON),
        ResultRangeSettings(result_length=90),
    )

    assert result.pattern_start_symbol == 35
    assert result.pattern_symbol_errors == 0
    assert result.carrier_frequency_offset_hz == pytest.approx(
        frequency_offset_hz, abs=150.0
    )
    np.testing.assert_array_equal(result.decoded_symbols, expected[35:125])


def test_pi4_dqpsk_pattern_search_and_lsb_symbol_bits():
    recording, signal = GeneratedIQSource.psk(
        modulation=ModulationKind.PI4_DQPSK,
        symbol_count=220,
        seed=31,
    )
    expected = np.asarray(recording.metadata["generated_symbols"])
    pattern = _pattern_from_generated(recording, 50, 24)

    result = PatternAnalyzer().search(
        recording,
        signal,
        PatternSearchSettings(pattern=pattern, mode=PatternSearchMode.ON),
        ResultRangeSettings(result_length=60),
        DemodulationSettings(bit_ordering=BitOrdering.LSB),
    )

    assert result.pattern_start_symbol == 50
    np.testing.assert_array_equal(result.decoded_symbols, expected[50:110])
    first_symbol = int(expected[50])
    np.testing.assert_array_equal(
        result.decoded_bits[:2], [first_symbol & 1, (first_symbol >> 1) & 1]
    )


def test_rs_style_pattern_and_result_range_settings_are_independent():
    pattern = KnownPattern((0, 1, 0, 1), name="Sync")
    search = PatternSearchSettings(pattern=pattern)
    result_range = ResultRangeSettings(result_length=100, offset_symbols=8)

    assert search.effective_correlation_threshold == pytest.approx(0.9)
    assert result_range.result_length == 100
    assert not hasattr(pattern, "symbols_after_pattern")


def test_session_publishes_generic_pattern_result():
    recording, signal = GeneratedIQSource.fsk(symbol_count=160, seed=101)
    expected = np.asarray(recording.metadata["generated_symbols"])
    session = VSASession(recording=recording, signal=signal)
    session.update_settings(
        analysis_center_frequency_hz=0.0,
        analysis_bandwidth_hz=1_500_000.0,
    )
    session.configure_pattern_analysis(
        PatternSearchSettings(
            pattern=KnownPattern(tuple(map(int, expected[24:56]))),
            mode=PatternSearchMode.ON,
            correlation_threshold_auto=False,
            iq_correlation_threshold=0.7,
        ),
        ResultRangeSettings(result_length=48),
    )

    session.analyze()

    assert session.result is not None
    assert session.pattern_result is not None
    assert session.pattern_result.metadata["source"].endswith("Analysis channel")
    assert session.pattern_range_result is not None
    assert session.pattern_range_result.iq.size == 48 * 8
    np.testing.assert_array_equal(session.pattern_result.decoded_symbols, expected[24:72])


def test_generic_pattern_session_finds_real_pluto_br_capture():
    fixture = Path(__file__).with_name("fixtures") / "bluetooth_br_prbs9_pluto_16msps.npz"
    with np.load(fixture, allow_pickle=False) as capture:
        recording = IQRecording(
            capture["iq"],
            sample_rate_hz=float(capture["sample_rate_hz"]),
            center_frequency_hz=float(capture["center_frequency_hz"]),
            usable_bandwidth_hz=float(capture["usable_bandwidth_hz"]),
            source="Pluto fixed BR fixture",
        )
    access = access_code_bits(0xC6967E)
    session = VSASession(
        recording=recording,
        signal=SignalDescription(
            modulation=ModulationKind.GFSK,
            symbol_rate_hz=1_000_000.0,
            frequency_deviation_hz=160_000.0,
            tx_filter="Gaussian",
            filter_parameter=0.5,
        ),
    )
    session.update_settings(
        analysis_center_frequency_hz=2_441_000_000.0,
        analysis_bandwidth_hz=1_500_000.0,
    )
    session.configure_pattern_analysis(
        PatternSearchSettings(
            pattern=KnownPattern(tuple(map(int, access))),
            mode=PatternSearchMode.ON,
        ),
        ResultRangeSettings(result_length=126),
    )

    session.analyze()

    assert session.pattern_result is not None
    assert session.pattern_result.correlation > 0.99
    assert session.pattern_result.pattern_symbol_errors == 0
    assert session.pattern_range_result is not None
    assert session.pattern_range_result.iq.size > 0
    np.testing.assert_array_equal(
        session.pattern_result.decoded_symbols[: access.size], access
    )


def test_real_pluto_cfo_stays_anchored_to_known_pattern():
    fixture = Path(__file__).with_name("fixtures") / "bluetooth_br_prbs9_pluto_16msps.npz"
    recording = FileIQSource.load(fixture)
    access = access_code_bits(0xC6967E)
    session = VSASession(
        recording=recording,
        signal=SignalDescription(
            modulation=ModulationKind.GFSK,
            symbol_rate_hz=1_000_000.0,
            frequency_deviation_hz=160_000.0,
            tx_filter="Gaussian",
            filter_parameter=0.5,
        ),
    )
    session.update_settings(
        analysis_center_frequency_hz=2_441_000_000.0,
        analysis_bandwidth_hz=2_000_000.0,
    )
    session.configure_pattern_analysis(
        PatternSearchSettings(
            pattern=KnownPattern(tuple(map(int, access))),
            mode=PatternSearchMode.ON,
        ),
        ResultRangeSettings(result_length=360),
    )

    session.analyze()

    # The known access code measures about +20 kHz.  Packet-wide tentative
    # decisions used to overwrite this with -5.4 kHz and left the corrected
    # instantaneous-frequency trace visibly above zero.
    assert session.pattern_result is not None
    assert session.pattern_result.carrier_frequency_offset_hz == pytest.approx(
        20_000.0, abs=2_000.0
    )


def test_session_builds_sample_level_carrier_corrected_results():
    recording, signal = GeneratedIQSource.fsk(symbol_count=240, seed=211)
    expected = np.asarray(recording.metadata["generated_symbols"])
    carrier_offset_hz = 85_000.0
    sample_index = np.arange(recording.sample_count)
    offset_recording = IQRecording(
        iq=recording.iq
        * np.exp(
            2j
            * np.pi
            * carrier_offset_hz
            * sample_index
            / recording.sample_rate_hz
        ),
        sample_rate_hz=recording.sample_rate_hz,
        metadata=recording.metadata,
    )
    session = VSASession(recording=offset_recording, signal=signal)
    session.update_settings(remove_dc=False)
    session.configure_pattern_analysis(
        PatternSearchSettings(
            pattern=KnownPattern(tuple(map(int, expected[40:72]))),
            mode=PatternSearchMode.ON,
            correlation_threshold_auto=False,
            iq_correlation_threshold=0.7,
        ),
        ResultRangeSettings(result_length=96),
        DemodulationSettings(compensate_carrier_frequency_drift=False),
    )

    session.analyze()

    assert session.pattern_result.carrier_frequency_offset_hz == pytest.approx(
        carrier_offset_hz, abs=2_000.0
    )
    assert session.pattern_range_result is not None
    assert session.carrier_corrected_pattern_range_result is not None
    raw_frequency = session.pattern_range_result.instantaneous_frequency_hz[1:]
    corrected_frequency = (
        session.carrier_corrected_pattern_range_result.instantaneous_frequency_hz[1:]
    )
    np.testing.assert_allclose(
        raw_frequency - corrected_frequency,
        session.pattern_result.carrier_frequency_offset_hz,
        atol=2.0,
    )
