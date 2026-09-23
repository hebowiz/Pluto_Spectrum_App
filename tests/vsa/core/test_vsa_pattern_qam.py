from pathlib import Path
import numpy as np
import pytest
from scipy.ndimage import shift as fractional_shift
from pluto_vsa.model import IQRecording, ModulationKind, SignalDescription
from pluto_vsa.mapping import BLUETOOTH_HDT_MAPPING, psk_constellation
from pluto_vsa.pattern import (
    BitOrdering,
    DemodulationSettings,
    KnownPattern,
    MeasurementFilterMode,
    PatternAnalyzer,
    PatternSearchSettings,
    ResultRangeSettings,
    carrier_correct_recording,
    prepare_psk_iq,
)
from pluto_vsa.sources import FileIQSource


def test_known_hdt_qam_pattern_uses_the_published_symbol_sample_times() -> None:
    path = Path(__file__).resolve().parents[2] / "data" / "fixtures" / "bluetooth" / "hdt" / "bluetooth_hdt7_5_prbs9_16msps.npz"
    recording = FileIQSource.load(path)
    signal = SignalDescription(
        modulation=ModulationKind.QAM16,
        symbol_rate_hz=2_000_000.0,
        tx_filter="Root Raised Cosine",
        filter_parameter=0.4,
        symbol_mapping=BLUETOOTH_HDT_MAPPING,
    )
    pattern = KnownPattern(
        tuple(int(value, 16) for value in "6 E 1 8 5 B A B 6 3".split())
    )
    demodulation = DemodulationSettings(
        measurement_filter=MeasurementFilterMode.AUTO,
        bit_ordering=BitOrdering.LSB,
    )
    result = PatternAnalyzer().search(
        recording,
        signal,
        PatternSearchSettings(pattern=pattern),
        ResultRangeSettings(result_length=500),
        demodulation,
    )
    corrected_recording = carrier_correct_recording(
        recording, result, compensate_drift=False
    )
    prepared, prepared_rate_hz = prepare_psk_iq(
        corrected_recording.iq,
        sample_rate_hz=corrected_recording.sample_rate_hz,
        symbol_rate_hz=signal.symbol_rate_hz,
        tx_filter=signal.tx_filter,
        filter_parameter=signal.filter_parameter,
    )
    centers = result.symbol_time_s * prepared_rate_hz
    displayed = np.interp(centers, np.arange(prepared.size), prepared.real) + 1j * np.interp(
        centers, np.arange(prepared.size), prepared.imag
    )
    displayed /= np.sqrt(np.mean(np.abs(displayed) ** 2))
    reference = psk_constellation(
        signal.modulation, signal.symbol_mapping
    )[result.decoded_symbols]
    display_evm_percent = 100.0 * np.sqrt(
        np.sum(np.abs(displayed - reference) ** 2)
        / np.sum(np.abs(reference) ** 2)
    )

    np.testing.assert_allclose(displayed, result.measured_symbols, atol=1e-6)
    assert result.evm_rms_percent < 4.0
    assert display_evm_percent < 4.0


def test_known_hdt_qam_pattern_refines_carrier_over_the_result_range() -> None:
    path = Path(__file__).resolve().parents[2] / "data" / "fixtures" / "bluetooth" / "hdt" / "bluetooth_hdt7_5_prbs9_16msps.npz"
    recording = FileIQSource.load(path)
    carrier_offset_hz = 100_000.0
    sample_index = np.arange(recording.sample_count, dtype=np.float64)
    rng = np.random.default_rng(130)
    signal_power = float(np.mean(np.abs(recording.iq) ** 2))
    noise_power = signal_power / 1_000.0
    noise = np.sqrt(noise_power / 2.0) * (
        rng.standard_normal(recording.sample_count)
        + 1j * rng.standard_normal(recording.sample_count)
    )
    impaired = IQRecording(
        iq=(
            recording.iq
            * np.exp(
                1j
                * (
                    0.7
                    + 2.0
                    * np.pi
                    * carrier_offset_hz
                    * sample_index
                    / recording.sample_rate_hz
                )
            )
            + noise
        ).astype(np.complex64),
        sample_rate_hz=recording.sample_rate_hz,
        metadata=recording.metadata,
    )
    signal = SignalDescription(
        modulation=ModulationKind.QAM16,
        symbol_rate_hz=2_000_000.0,
        tx_filter="Root Raised Cosine",
        filter_parameter=0.4,
        symbol_mapping=BLUETOOTH_HDT_MAPPING,
    )
    pattern = KnownPattern(
        tuple(int(value, 16) for value in "6 E 1 8 5 B A B 6 3".split())
    )

    result = PatternAnalyzer().search(
        impaired,
        signal,
        PatternSearchSettings(
            pattern=pattern,
            correlation_threshold_auto=False,
            iq_correlation_threshold=0.5,
        ),
        ResultRangeSettings(result_length=500),
        DemodulationSettings(
            measurement_filter=MeasurementFilterMode.AUTO,
            bit_ordering=BitOrdering.LSB,
        ),
    )

    assert result.pattern_symbol_errors == 0
    assert result.carrier_frequency_offset_hz == pytest.approx(
        carrier_offset_hz, abs=50.0
    )
    assert result.evm_rms_percent < 5.0
    assert (
        result.metadata["phase_estimation_method"]
        == "known-pattern ambiguity with result-range QAM carrier fit"
    )


@pytest.mark.parametrize("delay_samples", (0.1, 0.3, 0.5, 0.7, 0.9))
def test_known_hdt_qam_pattern_refines_fractional_symbol_timing(
    delay_samples: float,
) -> None:
    path = Path(__file__).resolve().parents[2] / "data" / "fixtures" / "bluetooth" / "hdt" / "bluetooth_hdt7_5_prbs9_16msps.npz"
    recording = FileIQSource.load(path)
    delayed_iq = fractional_shift(
        recording.iq.real,
        delay_samples,
        order=3,
        mode="constant",
    ) + 1j * fractional_shift(
        recording.iq.imag,
        delay_samples,
        order=3,
        mode="constant",
    )
    delayed = IQRecording(
        iq=delayed_iq.astype(np.complex64),
        sample_rate_hz=recording.sample_rate_hz,
        metadata=recording.metadata,
    )
    signal = SignalDescription(
        modulation=ModulationKind.QAM16,
        symbol_rate_hz=2_000_000.0,
        tx_filter="Root Raised Cosine",
        filter_parameter=0.4,
        symbol_mapping=BLUETOOTH_HDT_MAPPING,
    )
    pattern = KnownPattern(
        tuple(int(value, 16) for value in "6 E 1 8 5 B A B 6 3".split())
    )

    result = PatternAnalyzer().search(
        delayed,
        signal,
        PatternSearchSettings(pattern=pattern),
        ResultRangeSettings(result_length=500),
        DemodulationSettings(
            measurement_filter=MeasurementFilterMode.AUTO,
            bit_ordering=BitOrdering.LSB,
        ),
    )

    assert result.pattern_symbol_errors == 0
    assert result.evm_rms_percent < 2.0
    assert abs(result.metadata["fractional_timing_offset_samples"]) > 0.05
