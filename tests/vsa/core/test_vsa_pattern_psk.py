import numpy as np
import pytest
from scipy.ndimage import shift as fractional_shift
from pluto_vsa.model import IQRecording, ModulationKind, SignalDescription
from pluto_vsa.mapping import reverse_symbol_bits
from pluto_vsa.pattern import (
    BitOrdering,
    DemodulationSettings,
    KnownPattern,
    MatchSelectionPolicy,
    MeasurementFilterMode,
    PatternAnalyzer,
    PatternSearchMode,
    PatternSearchSettings,
    ResultRangeSettings,
    _constellation,
    _fit_differential_psk_phase_model,
    _root_raised_cosine_taps,
)
from pluto_vsa.sources import GeneratedIQSource
from _vsa_pattern_test_helpers import _pattern_from_generated


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


@pytest.mark.parametrize(
    ("modulation", "delay_samples"),
    (
        (ModulationKind.BPSK, 0.5),
        (ModulationKind.QPSK, 0.1),
        (ModulationKind.QPSK, 0.5),
        (ModulationKind.QPSK, 0.9),
        (ModulationKind.OQPSK, 0.5),
        (ModulationKind.PI4_QPSK, 0.5),
        (ModulationKind.PSK8, 0.5),
    ),
)
def test_known_nondifferential_psk_refines_fractional_symbol_timing(
    modulation: ModulationKind,
    delay_samples: float,
) -> None:
    rng = np.random.default_rng(20260902)
    samples_per_symbol = 8
    symbols = rng.integers(modulation.order, size=260)
    waveform_symbols = _constellation(modulation)[symbols]
    padded = np.pad(waveform_symbols, (10, 10), mode="edge")
    impulses = np.zeros(
        padded.size * samples_per_symbol, dtype=np.complex128
    )
    impulses[
        np.arange(padded.size) * samples_per_symbol
        + samples_per_symbol // 2
    ] = padded
    shaped = np.convolve(
        impulses,
        _root_raised_cosine_taps(samples_per_symbol, 0.4),
        mode="same",
    )
    start = 10 * samples_per_symbol
    shaped = shaped[start : start + symbols.size * samples_per_symbol]
    shaped /= np.sqrt(np.mean(np.abs(shaped) ** 2))
    delayed_iq = fractional_shift(
        shaped.real, delay_samples, order=3, mode="constant"
    ) + 1j * fractional_shift(
        shaped.imag, delay_samples, order=3, mode="constant"
    )
    recording = IQRecording(
        iq=delayed_iq.astype(np.complex64),
        sample_rate_hz=8_000_000.0,
        metadata={"dc_removal_recommended": False},
    )
    signal = SignalDescription(
        modulation=modulation,
        symbol_rate_hz=1_000_000.0,
        tx_filter="Root Raised Cosine",
        filter_parameter=0.4,
    )
    pattern = KnownPattern(
        tuple(
            map(
                int,
                reverse_symbol_bits(symbols[35:59], modulation.order),
            )
        )
    )

    result = PatternAnalyzer().search(
        recording,
        signal,
        PatternSearchSettings(pattern=pattern),
        ResultRangeSettings(result_length=90),
    )

    assert result.pattern_symbol_errors == 0
    assert result.evm_rms_percent < 1.0
    assert result.metadata["fractional_timing_offset_samples"] == pytest.approx(
        delay_samples, abs=0.03
    )
    assert (
        result.metadata["phase_estimation_method"]
        == "known-pattern ambiguity with result-range PSK carrier fit"
    )


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
