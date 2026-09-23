from dataclasses import replace
import numpy as np
import pytest
import pluto_vsa.session as session_module
from pluto_vsa.model import IQRecording, ModulationKind
from pluto_vsa.pattern import (
    DemodulationSettings,
    KnownPattern,
    MatchSelectionPolicy,
    PatternSearchMode,
    PatternSearchSettings,
    ResultRangeSettings,
)
from pluto_vsa.sources import GeneratedIQSource
from pluto_vsa.session import VSASession
from _vsa_pattern_test_helpers import _pattern_from_generated


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


def test_session_prepares_analysis_channel_once_and_reports_stage_timings(
    monkeypatch,
):
    recording, signal = GeneratedIQSource.fsk(symbol_count=160, seed=102)
    expected = np.asarray(recording.metadata["generated_symbols"])
    original_extract = session_module.extract_analysis_channel
    calls = []

    def counted_extract(*args, **kwargs):
        calls.append((args, kwargs))
        return original_extract(*args, **kwargs)

    monkeypatch.setattr(
        session_module,
        "extract_analysis_channel",
        counted_extract,
    )
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

    assert len(calls) == 1
    assert {
        "preprocess",
        "base_analysis",
        "pattern_search",
        "post_prepare",
        "post_analysis",
        "total_dsp",
    }.issubset(session.analysis_timings_ms)
    assert all(value >= 0.0 for value in session.analysis_timings_ms.values())
    np.testing.assert_array_equal(
        session.pattern_result.decoded_symbols,
        expected[24:72],
    )


def test_packet_snapshots_reuse_preprocessing_and_preserve_analysis(monkeypatch):
    recording, signal = GeneratedIQSource.psk(
        modulation=ModulationKind.PI4_DQPSK,
        symbol_count=160,
        seed=123,
    )
    gap = np.zeros(64, dtype=np.complex64)
    combined = IQRecording(
        iq=np.concatenate((recording.iq, gap, recording.iq)),
        sample_rate_hz=recording.sample_rate_hz,
    )
    pattern = _pattern_from_generated(recording, 20, 16)
    original_extract = session_module.extract_analysis_channel
    calls = []

    def counted_extract(*args, **kwargs):
        calls.append((args, kwargs))
        return original_extract(*args, **kwargs)

    monkeypatch.setattr(session_module, "extract_analysis_channel", counted_extract)
    session = VSASession(recording=combined, signal=signal)
    session.update_settings(
        analysis_center_frequency_hz=0.0,
        analysis_bandwidth_hz=1_500_000.0,
    )
    session.configure_pattern_analysis(
        PatternSearchSettings(
            pattern=pattern,
            mode=PatternSearchMode.ON,
            match_selection=MatchSelectionPolicy.INDEX,
            match_index=1,
        ),
        ResultRangeSettings(result_length=100),
    )
    session.analyze()

    deferred = session.analysis_snapshot()
    deferred.pattern_search = replace(
        session.pattern_search,
        match_selection=MatchSelectionPolicy.INDEX,
        match_index=2,
    )
    deferred.analyze(generate_display_products=False)

    full = session.analysis_snapshot()
    full.pattern_search = deferred.pattern_search
    full.analyze()

    assert len(calls) == 1
    assert deferred.capture_time_s is session.capture_time_s
    assert deferred.capture_power_dbm is session.capture_power_dbm
    assert deferred.pattern_result is not None
    assert full.pattern_result is not None
    assert deferred.pattern_range_result is None
    assert deferred.carrier_corrected_result is None
    assert deferred.pattern_result.pattern_start_sample == (
        full.pattern_result.pattern_start_sample
    )
    assert deferred.pattern_result.pattern_start_sample == pytest.approx(
        recording.sample_count + gap.size + 20 * 8,
        abs=4,
    )
    np.testing.assert_array_equal(
        deferred.pattern_result.decoded_symbols,
        full.pattern_result.decoded_symbols,
    )
    np.testing.assert_allclose(
        deferred.pattern_result.measured_symbols,
        full.pattern_result.measured_symbols,
        rtol=0.0,
        atol=0.0,
    )
    assert deferred.carrier_corrected_pattern_range_result is not None
    assert full.carrier_corrected_pattern_range_result is not None
    np.testing.assert_allclose(
        deferred.carrier_corrected_pattern_range_result.iq,
        full.carrier_corrected_pattern_range_result.iq,
        rtol=0.0,
        atol=0.0,
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
