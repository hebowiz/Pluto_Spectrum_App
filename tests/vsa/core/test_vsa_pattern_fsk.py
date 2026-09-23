from pathlib import Path
import numpy as np
import pytest
from scipy.ndimage import shift as fractional_shift
from pluto_vsa.model import IQRecording, ModulationKind, SignalDescription
from pluto_vsa.pattern import (
    DemodulationSettings,
    KnownPattern,
    MatchSelectionPolicy,
    MeasurementFilterMode,
    PatternAnalyzer,
    PatternSearchMode,
    PatternSearchSettings,
    ResultRangeSettings,
)
from pluto_vsa.sources import FileIQSource, GeneratedIQSource
from pluto_vsa.session import VSASession
from pluto_vsa.profiles.bluetooth_br import access_code_bits
from pluto_vsa.profiles.bluetooth_br import (
    build_packet_bits,
    giac_access_code_bits,
    modulate_packet_bits,
)
from pluto_vsa.demod.gfsk import demodulate_gfsk, fsk_reference_frequency_levels
from _vsa_pattern_test_helpers import _pattern_from_generated


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


def test_generic_pattern_session_finds_real_pluto_br_capture():
    fixture = Path(__file__).resolve().parents[2] / "data" / "fixtures" / "bluetooth" / "br-edr" / "bluetooth_br_prbs9_pluto_16msps.npz"
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
        Path(__file__).resolve().parents[2] / "data" / "fixtures" / "bluetooth" / "br-edr"
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
    fixture = Path(__file__).resolve().parents[2] / "data" / "fixtures" / "bluetooth" / "le" / "LE1M_FSK_error.npz"
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
    fixture = Path(__file__).resolve().parents[2] / "data" / "fixtures" / "bluetooth" / "br-edr" / "bluetooth_br_prbs9_pluto_16msps.npz"
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
