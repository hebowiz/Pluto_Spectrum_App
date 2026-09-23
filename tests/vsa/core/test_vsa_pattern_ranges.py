import numpy as np
import pytest
from pluto_vsa.model import IQRecording, ModulationKind
from pluto_vsa.pattern import (
    IQPowerTriggerSettings,
    KnownPattern,
    MatchSelectionPolicy,
    PatternAnalyzer,
    PatternSearchMode,
    PatternSearchSettings,
    ResultRangeAlignment,
    ResultRangeSettings,
    detect_iq_power_trigger_events,
)
from pluto_vsa.sources import GeneratedIQSource
from _vsa_pattern_test_helpers import _pattern_from_generated


def test_iq_power_trigger_detects_all_bursts_with_dropout_and_holdoff():
    iq = np.zeros(160, dtype=np.complex64)
    iq[20:60] = 1.0
    iq[35:38] = 0.01  # Short dip must not split the first burst.
    iq[90:140] = 0.5
    recording = IQRecording(iq=iq, sample_rate_hz=8_000_000.0)
    settings = IQPowerTriggerSettings(
        enabled=True,
        level_dbm=-10.0,
        hysteresis_db=3.0,
        dropout_symbols=1.0,
        holdoff_symbols=2.0,
    )

    events = detect_iq_power_trigger_events(
        recording,
        symbol_rate_hz=1_000_000.0,
        settings=settings,
    )

    assert [event.trigger_sample for event in events] == [20, 90]
    assert events[0].active_stop_sample == pytest.approx(60, abs=1)
    assert events[1].active_stop_sample == pytest.approx(140, abs=1)


def test_power_gated_pattern_search_returns_one_match_per_trigger_event():
    recording, signal = GeneratedIQSource.psk(
        modulation=ModulationKind.PI4_DQPSK,
        symbol_count=120,
        seed=932,
    )
    pattern_start = 20
    pattern = _pattern_from_generated(recording, pattern_start, 16)
    silence = np.zeros(80, dtype=np.complex64)
    combined = IQRecording(
        iq=np.concatenate((silence, recording.iq, silence, recording.iq, silence)),
        sample_rate_hz=recording.sample_rate_hz,
    )
    search = PatternSearchSettings(
        pattern=pattern,
        mode=PatternSearchMode.ON,
        match_selection=MatchSelectionPolicy.INDEX,
        match_index=2,
        iq_power_trigger=IQPowerTriggerSettings(
            enabled=True,
            level_dbm=-10.0,
            hysteresis_db=3.0,
            dropout_symbols=2.0,
            search_start_offset_symbols=1.0,
        ),
    )

    result = PatternAnalyzer().search(
        combined,
        signal,
        search,
        ResultRangeSettings(result_length=200),
    )

    expected_second_start = silence.size + recording.sample_count + silence.size
    assert result.pattern_start_sample == expected_second_start + pattern_start * 8
    assert result.metadata["power_trigger_event_count"] == 2
    assert result.metadata["power_trigger_matched_event_count"] == 2
    assert result.metadata["selected_power_trigger_event_index"] == 2
    assert result.metadata["selected_match_index"] == 2
    assert result.metadata["eligible_match_count"] == 2
    assert result.metadata["burst_limited_symbol_count"] < 200
    assert np.sqrt(np.mean(np.abs(result.measured_symbols) ** 2)) == pytest.approx(
        1.0, abs=1e-6
    )
    assert result.evm_rms_percent < 5.0
    assert result.result_stop_sample <= result.metadata[
        "power_trigger_active_stop_sample"
    ]
    assert (
        result.symbol_time_s[-1] + 0.5 / signal.symbol_rate_hz
        <= result.metadata["power_trigger_active_stop_sample"]
        / recording.sample_rate_hz
    )


def test_rs_style_pattern_and_result_range_settings_are_independent():
    pattern = KnownPattern((0, 1, 0, 1), name="Sync")
    search = PatternSearchSettings(pattern=pattern)
    result_range = ResultRangeSettings(result_length=100, offset_symbols=8)

    assert search.effective_correlation_threshold == pytest.approx(0.9)
    assert result_range.result_length == 100
    assert not hasattr(pattern, "symbols_after_pattern")


@pytest.mark.parametrize(
    "modulation",
    (ModulationKind.FSK, ModulationKind.QPSK, ModulationKind.PI4_DQPSK),
)
def test_negative_result_offset_demodulates_symbols_before_the_pattern(
    modulation: ModulationKind,
) -> None:
    if modulation is ModulationKind.FSK:
        recording, signal = GeneratedIQSource.fsk(symbol_count=160, seed=770)
    else:
        recording, signal = GeneratedIQSource.psk(
            modulation=modulation, symbol_count=160, seed=770
        )
    expected = np.asarray(recording.metadata["generated_symbols"])
    pattern_start = 50
    pattern = _pattern_from_generated(recording, pattern_start, 16)

    result = PatternAnalyzer().search(
        recording,
        signal,
        PatternSearchSettings(pattern=pattern),
        ResultRangeSettings(result_length=30, offset_symbols=-10),
    )

    assert result.pattern_start_sample == pattern_start * 8
    assert result.result_start_sample == (pattern_start - 10) * 8
    assert result.decoded_symbols.size == 30
    np.testing.assert_array_equal(
        result.decoded_symbols,
        expected[pattern_start - 10 : pattern_start + 20],
    )
    assert result.metadata["result_offset_symbols"] == -10


@pytest.mark.parametrize(
    ("alignment", "expected_start"),
    (
        (ResultRangeAlignment.CENTER, 43),
        (ResultRangeAlignment.RIGHT, 36),
    ),
)
def test_center_and_right_result_alignment_can_begin_before_the_pattern(
    alignment: ResultRangeAlignment,
    expected_start: int,
) -> None:
    recording, signal = GeneratedIQSource.psk(
        modulation=ModulationKind.QPSK, symbol_count=160, seed=771
    )
    expected = np.asarray(recording.metadata["generated_symbols"])
    pattern_start = 50
    pattern = _pattern_from_generated(recording, pattern_start, 16)

    result = PatternAnalyzer().search(
        recording,
        signal,
        PatternSearchSettings(pattern=pattern),
        ResultRangeSettings(
            result_length=30,
            alignment=alignment,
        ),
    )

    np.testing.assert_array_equal(
        result.decoded_symbols,
        expected[expected_start : expected_start + 30],
    )


def test_negative_result_offset_honors_incomplete_result_setting() -> None:
    recording, signal = GeneratedIQSource.psk(
        modulation=ModulationKind.QPSK, symbol_count=80, seed=772
    )
    expected = np.asarray(recording.metadata["generated_symbols"])
    pattern = _pattern_from_generated(recording, 5, 16)
    settings = ResultRangeSettings(result_length=30, offset_symbols=-10)

    partial = PatternAnalyzer().search(
        recording,
        signal,
        PatternSearchSettings(pattern=pattern),
        settings,
    )

    np.testing.assert_array_equal(partial.decoded_symbols, expected[:25])
    with pytest.raises(ValueError, match="search requirements"):
        PatternAnalyzer().search(
            recording,
            signal,
            PatternSearchSettings(pattern=pattern),
            ResultRangeSettings(
                result_length=30,
                offset_symbols=-10,
                exclude_incomplete_result=True,
            ),
        )
