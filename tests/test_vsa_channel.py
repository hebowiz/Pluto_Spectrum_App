import numpy as np
import pytest

from pluto_sa.vsa.analysis import VSAAnalyzer
from pluto_sa.vsa.channel import (
    extract_analysis_channel,
    extract_requested_analysis_channel,
    validate_analysis_channel_capture,
)
from pluto_sa.vsa.model import IQRecording, ModulationKind, SignalDescription, VSASettings
from pluto_sa.vsa.profiles.bluetooth_br import (
    BluetoothBRProfile,
    access_code_bits,
    giac_access_code_bits,
    modulate_packet_bits,
)


def _shift(iq: np.ndarray, frequency_hz: float, sample_rate_hz: float) -> np.ndarray:
    samples = np.arange(iq.size, dtype=np.float64)
    return iq * np.exp(2j * np.pi * frequency_hz * samples / sample_rate_hz)


def test_extract_analysis_channel_translates_filters_and_decimates() -> None:
    sample_rate_hz = 16_000_000.0
    sample_count = 16_384
    samples = np.arange(sample_count, dtype=np.float64)
    target = np.exp(2j * np.pi * 3_100_000.0 * samples / sample_rate_hz)
    interferer = 3.0 * np.exp(-2j * np.pi * 2_000_000.0 * samples / sample_rate_hz)
    recording = IQRecording(
        target + interferer,
        sample_rate_hz=sample_rate_hz,
        center_frequency_hz=2_441_000_000.0,
        usable_bandwidth_hz=14_000_000.0,
    )

    selected = extract_analysis_channel(
        recording,
        center_frequency_hz=2_444_000_000.0,
        bandwidth_hz=1_000_000.0,
    )

    assert selected.sample_rate_hz == pytest.approx(4_000_000.0)
    assert selected.center_frequency_hz == pytest.approx(2_444_000_000.0)
    assert selected.usable_bandwidth_hz == pytest.approx(1_000_000.0)
    assert selected.metadata["analysis_decimation"] == 4
    spectrum = np.fft.fftshift(
        np.fft.fft(selected.iq * np.hanning(selected.sample_count))
    )
    frequency = np.fft.fftshift(
        np.fft.fftfreq(selected.sample_count, d=1.0 / selected.sample_rate_hz)
    )
    peak_hz = frequency[int(np.argmax(np.abs(spectrum)))]
    assert peak_hz == pytest.approx(100_000.0, abs=2_000.0)


def test_selected_channel_recovers_giac_with_strong_adjacent_packet() -> None:
    sample_rate_hz = 16_000_000.0
    target_access = giac_access_code_bits(include_trailer=False)
    adjacent_access = access_code_bits(0x123456, include_trailer=False)
    target = modulate_packet_bits(
        target_access,
        sample_rate_hz=sample_rate_hz,
        prefix_samples=513,
        suffix_samples=511,
        snr_db=24.0,
        seed=3,
    )
    adjacent = modulate_packet_bits(
        adjacent_access,
        sample_rate_hz=sample_rate_hz,
        prefix_samples=513,
        suffix_samples=511,
    )
    wideband_iq = _shift(target, 3_050_000.0, sample_rate_hz) + 2.0 * _shift(
        adjacent, -2_000_000.0, sample_rate_hz
    )
    wideband = IQRecording(
        wideband_iq,
        sample_rate_hz=sample_rate_hz,
        center_frequency_hz=2_441_000_000.0,
        usable_bandwidth_hz=14_000_000.0,
    )

    selected = extract_analysis_channel(
        wideband,
        center_frequency_hz=2_444_000_000.0,
        bandwidth_hz=1_500_000.0,
    )
    result = BluetoothBRProfile(access_bits=target_access).analyze(selected)

    assert result.demodulation.access_correlation > 0.95
    assert result.demodulation.access_bit_errors == 0
    np.testing.assert_array_equal(
        result.demodulation.bits[: target_access.size], target_access
    )
    assert result.demodulation.carrier_frequency_offset_hz == pytest.approx(
        50_000.0, abs=10_000.0
    )


def test_vsa_settings_apply_selected_channel_before_analysis() -> None:
    sample_rate_hz = 8_000_000.0
    samples = np.arange(8192, dtype=np.float64)
    iq = np.exp(2j * np.pi * 1_100_000.0 * samples / sample_rate_hz)
    recording = IQRecording(
        iq,
        sample_rate_hz=sample_rate_hz,
        center_frequency_hz=100_000_000.0,
        usable_bandwidth_hz=6_000_000.0,
    )
    signal = SignalDescription(ModulationKind.FSK2, symbol_rate_hz=100_000.0)

    result = VSAAnalyzer().analyze(
        recording,
        signal,
        VSASettings(
            remove_dc=False,
            analysis_center_frequency_hz=101_000_000.0,
            analysis_bandwidth_hz=1_000_000.0,
        ),
    )

    peak_hz = result.spectrum_frequency_hz[int(np.argmax(result.spectrum_dbfs))]
    assert peak_hz == pytest.approx(100_000.0, abs=2_000.0)
    assert result.metadata["analysis_channel_applied"] is True
    assert result.metadata["analysis_center_frequency_hz"] == 101_000_000.0
    assert result.metadata["analysis_sample_rate_hz"] == pytest.approx(4_000_000.0)


def test_extract_analysis_channel_rejects_selection_outside_capture() -> None:
    recording = IQRecording(
        np.ones(1024, dtype=np.complex64),
        sample_rate_hz=8_000_000.0,
        center_frequency_hz=100_000_000.0,
        usable_bandwidth_hz=6_000_000.0,
    )

    with pytest.raises(ValueError, match="exceeds the usable capture bandwidth"):
        extract_analysis_channel(
            recording,
            center_frequency_hz=103_000_000.0,
            bandwidth_hz=1_000_000.0,
        )


def test_offset_lo_requires_filter_and_must_reject_dc() -> None:
    with pytest.raises(ValueError, match="requires Enable Analysis Channel"):
        validate_analysis_channel_capture(
            sample_rate_hz=8_000_000.0,
            usable_bandwidth_hz=6_400_000.0,
            lo_offset_hz=1_500_000.0,
            analysis_bandwidth_hz=None,
        )
    with pytest.raises(ValueError, match="must exceed half"):
        validate_analysis_channel_capture(
            sample_rate_hz=8_000_000.0,
            usable_bandwidth_hz=6_400_000.0,
            lo_offset_hz=700_000.0,
            analysis_bandwidth_hz=1_500_000.0,
        )


def test_offset_lo_and_filter_must_fit_usable_capture_bandwidth() -> None:
    validate_analysis_channel_capture(
        sample_rate_hz=8_000_000.0,
        usable_bandwidth_hz=6_400_000.0,
        lo_offset_hz=1_500_000.0,
        analysis_bandwidth_hz=1_500_000.0,
    )
    with pytest.raises(ValueError, match="exceed the usable Pluto"):
        validate_analysis_channel_capture(
            sample_rate_hz=8_000_000.0,
            usable_bandwidth_hz=6_400_000.0,
            lo_offset_hz=2_500_000.0,
            analysis_bandwidth_hz=1_500_000.0,
        )


def test_requested_live_analysis_channel_returns_to_nominal_center() -> None:
    sample_rate_hz = 8_000_000.0
    samples = np.arange(8192, dtype=np.float64)
    hardware_lo_hz = 2_442_500_000.0
    requested_center_hz = 2_441_000_000.0
    iq = np.exp(-2j * np.pi * 1_500_000.0 * samples / sample_rate_hz)
    recording = IQRecording(
        iq,
        sample_rate_hz=sample_rate_hz,
        center_frequency_hz=hardware_lo_hz,
        usable_bandwidth_hz=6_400_000.0,
        metadata={
            "requested_center_frequency_hz": requested_center_hz,
            "requested_analysis_bandwidth_hz": 1_500_000.0,
        },
    )

    selected = extract_requested_analysis_channel(recording)

    assert selected.center_frequency_hz == requested_center_hz
    assert selected.metadata["analysis_center_offset_hz"] == -1_500_000.0
    assert selected.metadata["analysis_channel_applied"] is True
