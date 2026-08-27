from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import shift as fractional_shift

import pluto_sa.vsa.session as session_module
from pluto_sa.vsa.model import IQRecording, ModulationKind, SignalDescription
from pluto_sa.vsa.mapping import reverse_symbol_bits
from pluto_sa.vsa.pattern import (
    BitOrdering,
    DemodulationSettings,
    IQPowerTriggerSettings,
    KnownPattern,
    MatchSelectionPolicy,
    MeasurementFilterMode,
    PatternAnalyzer,
    PatternSearchMode,
    PatternSearchSettings,
    ResultRangeSettings,
    SynchronizationSource,
    detect_iq_power_trigger_events,
    _constellation,
    _fit_differential_psk_phase_model,
)
from pluto_sa.vsa.sources import FileIQSource, GeneratedIQSource
from pluto_sa.vsa.session import VSASession
from pluto_sa.vsa.profiles.bluetooth_br import access_code_bits
from pluto_sa.vsa.profiles.bluetooth_br import (
    build_packet_bits,
    giac_access_code_bits,
    modulate_packet_bits,
)
from pluto_sa.vsa.demod.gfsk import (
    demodulate_gfsk,
    fsk_reference_frequency_levels,
)


def _pattern_from_generated(recording, start: int, length: int) -> KnownPattern:
    symbols = np.asarray(recording.metadata["generated_symbols"])
    maximum = int(np.max(symbols))
    order = 2 if maximum < 2 else (4 if maximum < 4 else 8)
    displayed = reverse_symbol_bits(symbols[start : start + length], order)
    return KnownPattern(tuple(int(value) for value in displayed))


def test_measurement_filter_defaults_to_auto_and_accepts_none() -> None:
    assert DemodulationSettings().measurement_filter is MeasurementFilterMode.AUTO
    assert DemodulationSettings().bit_ordering is BitOrdering.LSB
    assert (
        DemodulationSettings(
            measurement_filter=MeasurementFilterMode.NONE
        ).measurement_filter
        is MeasurementFilterMode.NONE
    )


def test_detected_data_psk_sync_does_not_claim_a_pattern_match() -> None:
    recording, signal = GeneratedIQSource.psk(
        modulation=ModulationKind.DPSK8,
        symbol_count=160,
        seed=20260821,
    )
    session = VSASession(recording=recording, signal=signal)
    session.configure_pattern_analysis(
        PatternSearchSettings(
            pattern=KnownPattern((0,) * 8, name="Not used"),
            mode=PatternSearchMode.ON,
            meas_only_if_pattern_symbols_correct=True,
        ),
        ResultRangeSettings(result_length=120),
        DemodulationSettings(
            coarse_synchronization=SynchronizationSource.DETECTED_DATA,
            measurement_filter=MeasurementFilterMode.NONE,
        ),
    )

    session.analyze()

    result = session.pattern_result
    assert result is not None
    assert result.metadata["synchronization_source"] == "Detected Data"
    assert result.metadata["pattern_match_valid"] is False
    assert result.metadata["pattern_symbol_count"] == 0
    assert result.decoded_symbols.size == 120
    assert result.evm_rms_percent < 10.0
    assert session.pattern_error is None


def test_detected_data_psk_sync_runs_without_pattern_search() -> None:
    recording, signal = GeneratedIQSource.psk(
        modulation=ModulationKind.DPSK8,
        symbol_count=160,
        seed=20260822,
    )
    session = VSASession(recording=recording, signal=signal)
    session.configure_pattern_analysis(
        None,
        ResultRangeSettings(result_length=120),
        DemodulationSettings(
            coarse_synchronization=SynchronizationSource.AUTO,
            measurement_filter=MeasurementFilterMode.NONE,
        ),
        IQPowerTriggerSettings(enabled=False, search_start_offset_symbols=3.0),
    )

    session.analyze()

    result = session.pattern_result
    assert session.pattern_search is None
    assert result is not None
    assert result.metadata["synchronization_source"] == "Detected Data"
    assert result.metadata["pattern_name"] == "Detected Data"
    assert not result.metadata["pattern_match_valid"]
    assert result.decoded_symbols.size == 120
    assert session.pattern_error is None


def test_pattern_only_sync_does_not_run_without_pattern_search() -> None:
    recording, signal = GeneratedIQSource.psk(
        modulation=ModulationKind.PI4_DQPSK,
        symbol_count=80,
        seed=20260823,
    )
    session = VSASession(recording=recording, signal=signal)
    session.configure_pattern_analysis(
        None,
        demodulation=DemodulationSettings(
            coarse_synchronization=SynchronizationSource.PATTERN,
        ),
    )

    session.analyze()

    assert session.pattern_result is None
    assert session.pattern_error is None


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


def test_psk_multiple_matches_support_time_selection_and_incomplete_exclusion():
    recording, signal = GeneratedIQSource.psk(
        modulation=ModulationKind.PI4_DQPSK,
        symbol_count=160,
        seed=123,
    )
    pattern = _pattern_from_generated(recording, 20, 16)
    gap = np.zeros(64, dtype=np.complex64)
    combined_iq = np.concatenate((recording.iq, gap, recording.iq))
    combined = IQRecording(
        iq=combined_iq,
        sample_rate_hz=recording.sample_rate_hz,
    )

    first = PatternAnalyzer().search(
        combined,
        signal,
        PatternSearchSettings(
            pattern=pattern,
            mode=PatternSearchMode.ON,
            match_selection=MatchSelectionPolicy.FIRST,
        ),
        ResultRangeSettings(result_length=100),
    )
    second = PatternAnalyzer().search(
        combined,
        signal,
        PatternSearchSettings(
            pattern=pattern,
            mode=PatternSearchMode.ON,
            match_selection=MatchSelectionPolicy.INDEX,
            match_index=2,
        ),
        ResultRangeSettings(result_length=100),
    )

    assert first.pattern_start_sample == 20 * 8
    assert second.pattern_start_sample == recording.sample_count + 64 + 20 * 8
    assert second.metadata["selected_match_index"] == 2
    assert second.metadata["eligible_match_count"] == 2

    truncated_stop = recording.sample_count + 64 + (20 + 16 + 20) * 8
    truncated = IQRecording(
        iq=combined_iq[:truncated_stop],
        sample_rate_hz=recording.sample_rate_hz,
    )
    allowed = PatternAnalyzer().search(
        truncated,
        signal,
        PatternSearchSettings(
            pattern=pattern,
            mode=PatternSearchMode.ON,
            match_selection=MatchSelectionPolicy.LAST,
        ),
        ResultRangeSettings(result_length=100),
    )
    excluded = PatternAnalyzer().search(
        truncated,
        signal,
        PatternSearchSettings(
            pattern=pattern,
            mode=PatternSearchMode.ON,
            match_selection=MatchSelectionPolicy.LAST,
        ),
        ResultRangeSettings(
            result_length=100,
            exclude_incomplete_result=True,
        ),
    )

    assert allowed.pattern_start_sample == second.pattern_start_sample
    assert allowed.decoded_symbols.size < 100
    assert excluded.pattern_start_sample == first.pattern_start_sample
    assert excluded.decoded_symbols.size == 100
    assert excluded.metadata["detected_match_count"] == 2
    assert excluded.metadata["eligible_match_count"] == 1


def test_psk_symbol_correct_filter_keeps_later_valid_match_navigable():
    recording, signal = GeneratedIQSource.psk(
        modulation=ModulationKind.PI4_DQPSK,
        symbol_count=160,
        seed=912,
    )
    pattern_start = 20
    pattern = _pattern_from_generated(recording, pattern_start, 32)
    corrupted_iq = np.array(recording.iq, copy=True)
    corrupted_symbol = pattern_start + 12
    corrupted_iq[corrupted_symbol * 8 : (corrupted_symbol + 1) * 8] *= 1j
    gap = np.zeros(64, dtype=np.complex64)
    combined = IQRecording(
        iq=np.concatenate((corrupted_iq, gap, recording.iq)),
        sample_rate_hz=recording.sample_rate_hz,
    )
    common = dict(
        pattern=pattern,
        mode=PatternSearchMode.ON,
        correlation_threshold_auto=False,
        iq_correlation_threshold=0.9,
    )

    filtered = PatternAnalyzer().search(
        combined,
        signal,
        PatternSearchSettings(
            **common,
            meas_only_if_pattern_symbols_correct=True,
            match_selection=MatchSelectionPolicy.FIRST,
        ),
        ResultRangeSettings(result_length=100),
    )
    unfiltered_first = PatternAnalyzer().search(
        combined,
        signal,
        PatternSearchSettings(
            **common,
            meas_only_if_pattern_symbols_correct=False,
            match_selection=MatchSelectionPolicy.FIRST,
        ),
        ResultRangeSettings(result_length=100),
    )
    unfiltered_second = PatternAnalyzer().search(
        combined,
        signal,
        PatternSearchSettings(
            **common,
            meas_only_if_pattern_symbols_correct=False,
            match_selection=MatchSelectionPolicy.INDEX,
            match_index=2,
        ),
        ResultRangeSettings(result_length=100),
    )

    later_start = recording.sample_count + gap.size + pattern_start * 8
    assert filtered.pattern_start_sample == later_start
    assert filtered.pattern_symbol_errors == 0
    assert filtered.metadata["eligible_match_count"] == 1
    assert unfiltered_first.pattern_start_sample == pattern_start * 8
    assert unfiltered_first.pattern_symbol_errors > 0
    assert unfiltered_first.metadata["eligible_match_count"] == 2
    assert unfiltered_second.pattern_start_sample == later_start
    assert unfiltered_second.pattern_symbol_errors == 0


def test_fsk_multiple_matches_use_one_physical_candidate_per_packet():
    recording, signal = GeneratedIQSource.fsk(
        symbol_count=180,
        gaussian_bt=0.5,
        seed=321,
    )
    pattern = _pattern_from_generated(recording, 30, 32)
    gap = np.zeros(64, dtype=np.complex64)
    combined = IQRecording(
        iq=np.concatenate((recording.iq, gap, recording.iq)),
        sample_rate_hz=recording.sample_rate_hz,
    )
    result = PatternAnalyzer().search(
        combined,
        signal,
        PatternSearchSettings(
            pattern=pattern,
            mode=PatternSearchMode.ON,
            correlation_threshold_auto=False,
            iq_correlation_threshold=0.7,
            match_selection=MatchSelectionPolicy.LAST,
        ),
        ResultRangeSettings(result_length=100),
    )

    assert result.pattern_start_sample == recording.sample_count + 64 + 30 * 8
    assert result.metadata["selected_match_index"] == 2
    assert result.metadata["eligible_match_count"] == 2
    assert result.metadata["detected_match_count"] == 2


def test_fsk_measurement_filter_none_preserves_tx_reference_shaping() -> None:
    recording, signal = GeneratedIQSource.fsk(
        symbol_count=180,
        gaussian_bt=0.5,
        seed=322,
    )
    expected = np.asarray(recording.metadata["generated_symbols"])
    pattern = _pattern_from_generated(recording, 30, 32)

    result = PatternAnalyzer().search(
        recording,
        signal,
        PatternSearchSettings(pattern=pattern, mode=PatternSearchMode.ON),
        ResultRangeSettings(result_length=100),
        DemodulationSettings(
            measurement_filter=MeasurementFilterMode.NONE
        ),
    )

    assert result.metadata["gaussian_bt"] == pytest.approx(0.5)
    assert result.metadata["fsk_measurement_filter"] == "None"
    assert result.pattern_symbol_errors == 0
    np.testing.assert_array_equal(result.decoded_symbols, expected[30:130])


def test_fsk_natural_mapping_rejects_frequency_inverted_pattern():
    recording, signal = GeneratedIQSource.fsk(
        symbol_count=180,
        gaussian_bt=0.5,
        seed=514,
    )
    pattern = _pattern_from_generated(recording, 30, 32)
    inverted = IQRecording(
        iq=np.conj(recording.iq),
        sample_rate_hz=recording.sample_rate_hz,
    )

    with pytest.raises(ValueError, match="Natural mapping frequency polarity"):
        PatternAnalyzer().search(
            inverted,
            signal,
            PatternSearchSettings(
                pattern=pattern,
                mode=PatternSearchMode.ON,
                correlation_threshold_auto=False,
                iq_correlation_threshold=0.7,
                meas_only_if_pattern_symbols_correct=False,
            ),
            ResultRangeSettings(result_length=80),
        )


def test_fsk_inverted_pattern_match_preserves_natural_mapping_symbols():
    recording, signal = GeneratedIQSource.fsk(
        symbol_count=180,
        gaussian_bt=0.5,
        seed=515,
    )
    expected = np.asarray(recording.metadata["generated_symbols"], dtype=np.uint8)
    pattern_start = 30
    pattern = _pattern_from_generated(recording, pattern_start, 32)
    inverted = IQRecording(
        iq=np.conj(recording.iq),
        sample_rate_hz=recording.sample_rate_hz,
    )

    result = PatternAnalyzer().search(
        inverted,
        signal,
        PatternSearchSettings(
            pattern=pattern,
            mode=PatternSearchMode.ON,
            correlation_threshold_auto=False,
            iq_correlation_threshold=0.7,
            allow_inverted_fsk_pattern=True,
        ),
        ResultRangeSettings(result_length=80),
    )

    assert result.metadata["pattern_match_variant"] == "Inverted"
    assert result.polarity_inverted
    assert result.pattern_symbol_errors == 0
    np.testing.assert_array_equal(
        result.metadata["matched_pattern_symbols"],
        1 - np.asarray(pattern.symbols),
    )
    np.testing.assert_array_equal(
        result.decoded_symbols,
        1 - expected[pattern_start : pattern_start + 80],
    )
    np.testing.assert_array_equal(
        result.decoded_symbols,
        (result.measured_symbols.real >= 0.0).astype(np.int16),
    )


def test_fsk_symbol_correct_filter_keeps_later_valid_match_navigable():
    recording, signal = GeneratedIQSource.fsk(
        symbol_count=180,
        gaussian_bt=None,
        seed=146,
    )
    pattern_start = 30
    pattern = _pattern_from_generated(recording, pattern_start, 32)
    corrupted_symbols = np.array(
        recording.metadata["generated_symbols"], copy=True
    )
    corrupted_symbols[pattern_start + 12] ^= 1
    levels = fsk_reference_frequency_levels(
        corrupted_symbols,
        samples_per_symbol=8,
        transmit_gaussian_bt=None,
    )
    phase = 2.0 * np.pi * np.cumsum(
        signal.frequency_deviation_hz * levels
    ) / recording.sample_rate_hz
    corrupted_iq = np.exp(1j * phase).astype(np.complex64)
    gap = np.zeros(64, dtype=np.complex64)
    combined = IQRecording(
        iq=np.concatenate((corrupted_iq, gap, recording.iq)),
        sample_rate_hz=recording.sample_rate_hz,
    )
    common = dict(
        pattern=pattern,
        mode=PatternSearchMode.ON,
        correlation_threshold_auto=False,
        iq_correlation_threshold=0.9,
    )

    filtered = PatternAnalyzer().search(
        combined,
        signal,
        PatternSearchSettings(
            **common,
            meas_only_if_pattern_symbols_correct=True,
            match_selection=MatchSelectionPolicy.FIRST,
        ),
        ResultRangeSettings(result_length=100),
    )
    unfiltered_first = PatternAnalyzer().search(
        combined,
        signal,
        PatternSearchSettings(
            **common,
            meas_only_if_pattern_symbols_correct=False,
            match_selection=MatchSelectionPolicy.FIRST,
        ),
        ResultRangeSettings(result_length=100),
    )
    unfiltered_second = PatternAnalyzer().search(
        combined,
        signal,
        PatternSearchSettings(
            **common,
            meas_only_if_pattern_symbols_correct=False,
            match_selection=MatchSelectionPolicy.INDEX,
            match_index=2,
        ),
        ResultRangeSettings(result_length=100),
    )

    later_start = recording.sample_count + gap.size + pattern_start * 8
    assert filtered.pattern_start_sample == later_start
    assert filtered.pattern_symbol_errors == 0
    assert filtered.metadata["eligible_match_count"] == 1
    assert unfiltered_first.pattern_start_sample == pattern_start * 8
    assert unfiltered_first.pattern_symbol_errors > 0
    assert unfiltered_first.metadata["eligible_match_count"] == 2
    assert unfiltered_second.pattern_start_sample == later_start
    assert unfiltered_second.pattern_symbol_errors == 0


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


@pytest.mark.parametrize("delay_samples", [0.125, 0.375, 0.625, 0.875])
def test_fsk_frequency_model_applies_fractional_symbol_timing(delay_samples):
    recording, signal = GeneratedIQSource.fsk(
        symbol_count=220,
        gaussian_bt=0.5,
        seed=817,
    )
    expected = np.asarray(recording.metadata["generated_symbols"])
    pattern_start_symbol = 40
    pattern = _pattern_from_generated(recording, pattern_start_symbol, 64)
    delayed = fractional_shift(
        recording.iq.real,
        shift=delay_samples,
        order=3,
        mode="constant",
        cval=0.0,
    ) + 1j * fractional_shift(
        recording.iq.imag,
        shift=delay_samples,
        order=3,
        mode="constant",
        cval=0.0,
    )
    rng = np.random.default_rng(90210)
    signal_power = float(np.mean(np.abs(delayed) ** 2))
    noise_power = signal_power / 100.0  # 20 dB SNR
    noise = np.sqrt(noise_power / 2.0) * (
        rng.standard_normal(delayed.size) + 1j * rng.standard_normal(delayed.size)
    )
    shifted_recording = IQRecording(
        iq=np.asarray(delayed + noise, dtype=np.complex64),
        sample_rate_hz=recording.sample_rate_hz,
    )

    result = PatternAnalyzer().search(
        shifted_recording,
        signal,
        PatternSearchSettings(
            pattern=pattern,
            mode=PatternSearchMode.ON,
            correlation_threshold_auto=False,
            iq_correlation_threshold=0.7,
        ),
        ResultRangeSettings(result_length=100),
    )

    expected_start_sample = pattern_start_symbol * 8 + delay_samples
    measured_start_sample = result.pattern_start_time_s * recording.sample_rate_hz
    assert measured_start_sample == pytest.approx(expected_start_sample, abs=0.08)
    assert abs(result.metadata["fractional_timing_offset_samples"]) > 0.05
    assert result.metadata["frequency_model_residual_rms_hz"] > 0.0
    assert result.pattern_symbol_errors == 0
    np.testing.assert_array_equal(
        result.decoded_symbols,
        expected[pattern_start_symbol : pattern_start_symbol + 100],
    )


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


def test_psk_measurement_filter_none_bypasses_srrc() -> None:
    recording, signal = GeneratedIQSource.psk(
        modulation=ModulationKind.QPSK,
        symbol_count=180,
        seed=20,
    )
    pattern = _pattern_from_generated(recording, 35, 24)

    result = PatternAnalyzer().search(
        recording,
        signal,
        PatternSearchSettings(pattern=pattern, mode=PatternSearchMode.ON),
        ResultRangeSettings(result_length=80),
        DemodulationSettings(
            measurement_filter=MeasurementFilterMode.NONE
        ),
    )

    assert result.metadata["measurement_filter"] == "None"
    assert result.metadata["matched_filter_applied"] is False
    assert result.pattern_symbol_errors == 0


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


@pytest.mark.parametrize(
    "modulation", [ModulationKind.PI4_DQPSK, ModulationKind.DPSK8]
)
def test_differential_psk_short_pattern_uses_joint_result_range_synchronization(
    modulation: ModulationKind,
) -> None:
    recording, signal = GeneratedIQSource.psk(
        modulation=modulation,
        symbol_count=420,
        seed=77,
    )
    expected = np.asarray(recording.metadata["generated_symbols"])
    sample_time_s = np.arange(recording.sample_count) / recording.sample_rate_hz
    initial_cfo_hz = 25_000.0
    drift_hz_per_s = 150.0e6
    phase = 2.0 * np.pi * (
        initial_cfo_hz * sample_time_s
        + 0.5 * drift_hz_per_s * sample_time_s**2
    )
    distorted = IQRecording(
        iq=(recording.iq * np.exp(1j * phase)).astype(np.complex64),
        sample_rate_hz=recording.sample_rate_hz,
        metadata=recording.metadata,
    )
    pattern_start = 50
    pattern = KnownPattern(
        tuple(
            map(
                int,
                reverse_symbol_bits(
                    expected[pattern_start : pattern_start + 10], modulation.order
                ),
            )
        )
    )

    result = PatternAnalyzer().search(
        distorted,
        signal,
        PatternSearchSettings(pattern=pattern, mode=PatternSearchMode.ON),
        ResultRangeSettings(result_length=244),
        DemodulationSettings(compensate_carrier_frequency_drift=True),
    )

    assert result.pattern_start_symbol == pattern_start
    assert result.carrier_frequency_offset_hz == pytest.approx(
        initial_cfo_hz + drift_hz_per_s * pattern_start / signal.symbol_rate_hz,
        abs=200.0,
    )
    assert result.carrier_frequency_drift_hz_per_s == pytest.approx(
        drift_hz_per_s, abs=1_000.0
    )
    assert result.phase_rotation_rad is not None
    assert result.metadata["absolute_reference_waveform_sync"] is True
    assert result.metadata["synchronization_evm_rms"] < 1e-6
    reference = _constellation(modulation)[result.decoded_symbols]
    expected_evm_percent = 100.0 * np.sqrt(
        np.sum(np.abs(result.measured_symbols - reference) ** 2)
        / np.sum(np.abs(reference) ** 2)
    )
    assert result.evm_rms_percent == pytest.approx(expected_evm_percent)
    assert result.evm_rms_percent < 1e-4
    np.testing.assert_array_equal(
        result.decoded_symbols, expected[pattern_start : pattern_start + 244]
    )
    assert (
        result.metadata["phase_estimation_method"]
        == "joint ideal-reference waveform complex-EVM synchronization"
    )


@pytest.mark.parametrize(
    ("modulation", "outlier_seed"),
    [(ModulationKind.PI4_DQPSK, 3), (ModulationKind.DPSK8, 1)],
)
def test_differential_psk_drift_fit_rejects_faded_phase_cycle_slips(
    modulation: ModulationKind,
    outlier_seed: int,
) -> None:
    alphabet = _constellation(modulation)
    symbol_count = 244
    symbol_indices = np.arange(symbol_count, dtype=np.float64)
    intercept_rad = 0.13
    expected_drift_hz_per_s = 6.0e6
    slope_rad_per_symbol = (
        2.0 * np.pi * expected_drift_hz_per_s / 1_000_000.0**2
    )
    rng = np.random.default_rng(11)
    data = rng.integers(alphabet.size, size=symbol_count)
    measured = alphabet[data] * np.exp(
        1j * (intercept_rad + slope_rad_per_symbol * symbol_indices)
    )

    # A short faded/disturbed interval makes ordinary Mth-power phase unwrap
    # acquire a whole-cycle slip, even though the remaining symbols are clean.
    outlier_indices = np.arange(90, 105)
    outlier_rng = np.random.default_rng(outlier_seed)
    measured[outlier_indices] = 0.05 * np.exp(
        1j * outlier_rng.uniform(-np.pi, np.pi, outlier_indices.size)
    )

    _, fitted_slope, _, drift_accepted, _ = _fit_differential_psk_phase_model(
        measured,
        symbol_indices,
        alphabet,
        pattern_phase_anchor_rad=intercept_rad + slope_rad_per_symbol * 4.5,
        pattern_center_symbol=4.5,
    )
    fitted_drift_hz_per_s = fitted_slope * 1_000_000.0**2 / (2.0 * np.pi)

    assert fitted_drift_hz_per_s == pytest.approx(
        expected_drift_hz_per_s, abs=1_000.0
    )
    assert drift_accepted


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


def test_real_pluto_fsk_fractional_timing_is_stable_across_analysis_bandwidth():
    fixture = (
        Path(__file__).with_name("fixtures")
        / "bluetooth_br_prbs9_pluto_16msps.npz"
    )
    recording = FileIQSource.load(fixture)
    access = access_code_bits(0xC6967E)
    signal = SignalDescription(
        modulation=ModulationKind.GFSK,
        symbol_rate_hz=1_000_000.0,
        frequency_deviation_hz=160_000.0,
        tx_filter="Gaussian",
        filter_parameter=0.5,
    )
    start_times: list[float] = []
    for bandwidth_hz in (
        1_200_000.0,
        1_500_000.0,
        2_000_000.0,
        3_000_000.0,
        5_000_000.0,
    ):
        session = VSASession(recording=recording, signal=signal)
        session.update_settings(
            analysis_center_frequency_hz=2_441_000_000.0,
            analysis_bandwidth_hz=bandwidth_hz,
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
        assert session.pattern_result.pattern_symbol_errors == 0
        assert session.pattern_result.metadata["timing_correction_accepted"]
        start_times.append(session.pattern_result.pattern_start_time_s)

    timing_span_analysis_samples = (
        max(start_times) - min(start_times)
    ) * 8_000_000.0
    assert timing_span_analysis_samples < 0.1


def test_le1m_phase_discontinuity_does_not_reverse_symbol_frequency():
    fixture = Path(__file__).with_name("fixtures") / "LE1M_FSK_error.npz"
    recording = FileIQSource.load(fixture)
    access = np.asarray(
        [
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            0, 1, 0, 1, 0, 0, 1, 0, 0, 0,
            0, 0, 1, 0, 0, 1, 1, 0, 1, 1,
            1, 0, 1, 0, 0, 0, 1, 1, 1, 0,
        ],
        dtype=np.uint8,
    )

    result = demodulate_gfsk(
        recording.iq,
        sample_rate_hz=recording.sample_rate_hz,
        access_bits=access,
        symbol_rate_hz=1_000_000.0,
        minimum_correlation=0.9,
        gaussian_bt=0.5,
        apply_measurement_filter=False,
        maximum_symbols=376,
        match_selection="First",
        require_zero_pattern_errors=True,
        allow_complemented_pattern_match=True,
    )

    assert result.bits[60] == 1
    assert result.symbol_frequency_hz[60] == pytest.approx(232_000.0, abs=20_000.0)


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


def test_fsk_pattern_result_honors_carrier_drift_compensation_setting() -> None:
    payload = np.tile(np.asarray([0, 1, 1, 0], dtype=np.uint8), 40)
    packet_bits = build_packet_bits(
        clock_6_1=0x15,
        uap=0x2A,
        payload_bits=payload,
        packet_type=3,
    )
    recording = IQRecording(
        modulate_packet_bits(
            packet_bits,
            sample_rate_hz=8_000_000.0,
            carrier_frequency_offset_hz=45_000.0,
            carrier_frequency_drift_hz_per_s=150.0e6,
            prefix_samples=19,
            suffix_samples=17,
            snr_db=30.0,
            seed=91,
        ),
        sample_rate_hz=8_000_000.0,
    )
    signal = SignalDescription(
        modulation=ModulationKind.GFSK,
        symbol_rate_hz=1_000_000.0,
        frequency_deviation_hz=160_000.0,
        tx_filter="Gaussian",
        filter_parameter=0.5,
    )
    search = PatternSearchSettings(
        pattern=KnownPattern(tuple(map(int, giac_access_code_bits()))),
        mode=PatternSearchMode.ON,
    )
    analyzer = PatternAnalyzer()
    uncompensated = analyzer.search(
        recording,
        signal,
        search,
        ResultRangeSettings(result_length=220),
        DemodulationSettings(compensate_carrier_frequency_drift=False),
    )
    compensated = analyzer.search(
        recording,
        signal,
        search,
        ResultRangeSettings(result_length=220),
        DemodulationSettings(compensate_carrier_frequency_drift=True),
    )

    assert compensated.carrier_frequency_drift_hz_per_s == pytest.approx(
        150.0e6, abs=20.0e6
    )
    np.testing.assert_array_equal(
        uncompensated.decoded_symbols, compensated.decoded_symbols
    )
    difference = (
        uncompensated.measured_symbols.real
        - compensated.measured_symbols.real
    )
    assert np.ptp(difference) > 20_000.0
