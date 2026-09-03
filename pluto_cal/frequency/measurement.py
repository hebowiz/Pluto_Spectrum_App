"""Robust CW frequency measurement away from Pluto's zero-IF region."""

from __future__ import annotations

from collections.abc import Callable
import math

import numpy as np

from pluto_cal.model import (
    CWToneEstimate,
    FrequencyCalibrationConfig,
    FrequencyMeasurement,
)


class CWDetectionError(RuntimeError):
    """Raised when the requested calibration tone is not detectable."""


class MeasurementQualityError(RuntimeError):
    """Raised when repeated tone estimates are not sufficiently stable."""


def estimate_cw_frequency(
    iq: np.ndarray,
    sample_rate_hz: float,
    *,
    expected_frequency_hz: float,
    search_half_width_hz: float = 150_000.0,
    minimum_snr_db: float = 18.0,
) -> CWToneEstimate:
    """Estimate one CW using FFT acquisition and band-limited phase advance.

    The FFT identifies and validates the tone.  The final estimate comes from
    the aggregate IQ phase difference of a narrow, FFT-filtered waveform, so
    its resolution is not limited to one FFT bin.
    """

    values = np.asarray(iq, dtype=np.complex128).reshape(-1)
    rate = float(sample_rate_hz)
    expected = float(expected_frequency_hz)
    half_width = float(search_half_width_hz)
    if values.size < 64:
        raise ValueError("CW estimation requires at least 64 IQ samples")
    if not np.all(np.isfinite(values)):
        raise ValueError("CW estimation requires finite IQ samples")
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError("sample_rate_hz must be positive and finite")
    if half_width <= 0.0 or abs(expected) + half_width >= rate / 2.0:
        raise ValueError("CW search interval must stay inside Nyquist")

    centered = values - np.mean(values)
    window = np.hanning(centered.size)
    windowed = centered * window
    spectrum = np.fft.fftshift(np.fft.fft(windowed))
    frequency = np.fft.fftshift(np.fft.fftfreq(values.size, d=1.0 / rate))
    power = np.abs(spectrum) ** 2
    search = np.flatnonzero(np.abs(frequency - expected) <= half_width)
    if search.size < 12:
        raise ValueError("CW search interval contains too few FFT bins")
    peak_index = int(search[np.argmax(power[search])])
    guard_bins = max(3, int(np.ceil(5_000.0 * values.size / rate)))
    noise_indices = search[np.abs(search - peak_index) > guard_bins]
    if noise_indices.size < 4:
        raise ValueError("CW search interval leaves too few noise bins")
    noise_power = float(np.median(power[noise_indices]))
    peak_power = float(power[peak_index])
    snr_db = 10.0 * math.log10(
        max(peak_power, np.finfo(np.float64).tiny)
        / max(noise_power, np.finfo(np.float64).tiny)
    )
    if not math.isfinite(snr_db) or snr_db < float(minimum_snr_db):
        raise CWDetectionError(
            f"Calibration CW SNR {snr_db:.1f} dB is below "
            f"{float(minimum_snr_db):.1f} dB"
        )

    coarse_hz = float(frequency[peak_index])
    # Keep only a narrow region around the acquired peak before calculating
    # aggregate phase advance.  This rejects broadband noise and DC while the
    # phase estimator retains sub-bin resolution.
    raw_spectrum = np.fft.fft(centered)
    raw_frequency = np.fft.fftfreq(centered.size, d=1.0 / rate)
    bin_width_hz = rate / centered.size
    pass_half_width_hz = max(8.0 * bin_width_hz, 20_000.0)
    passband = np.abs(raw_frequency - coarse_hz) <= pass_half_width_hz
    filtered = np.fft.ifft(np.where(passband, raw_spectrum, 0.0))
    trim = min(filtered.size // 8, max(8, int(rate / pass_half_width_hz)))
    usable = filtered[trim:-trim] if filtered.size > 2 * trim + 2 else filtered
    phase_product = np.vdot(usable[:-1], usable[1:])
    if abs(phase_product) <= np.finfo(np.float64).tiny:
        raise CWDetectionError("Calibration CW phase could not be estimated")
    precise_hz = float(np.angle(phase_product) * rate / (2.0 * np.pi))
    if abs(precise_hz - expected) > half_width:
        raise CWDetectionError("Detected tone is outside the calibration IF window")

    # Complex IQ has only one spectral image, so unlike a real-valued FFT no
    # factor-of-two correction is applied.
    coherent_amplitude = abs(spectrum[peak_index]) / max(
        float(np.sum(window)), 1.0
    )
    peak_dbfs = 20.0 * math.log10(
        max(coherent_amplitude, np.finfo(np.float64).tiny)
    )
    return CWToneEstimate(precise_hz, snr_db, peak_dbfs)


def measure_frequency(
    capture_iq: Callable[[], np.ndarray],
    *,
    xo_correction: int,
    config: FrequencyCalibrationConfig,
    capture_count: int | None = None,
) -> FrequencyMeasurement:
    """Measure repeated captures and return a robust median result."""

    count = int(capture_count or config.captures_per_measurement)
    estimates = tuple(
        estimate_cw_frequency(
            capture_iq(),
            config.sample_rate_hz,
            expected_frequency_hz=config.if_offset_hz,
            search_half_width_hz=config.search_half_width_hz,
            minimum_snr_db=config.minimum_snr_db,
        )
        for _ in range(count)
    )
    frequencies = np.asarray(
        [estimate.frequency_hz for estimate in estimates], dtype=np.float64
    )
    measured_if_hz = float(np.median(frequencies))
    median_absolute_deviation = float(
        np.median(np.abs(frequencies - measured_if_hz))
    )
    spread_hz = 1.4826 * median_absolute_deviation
    if not math.isfinite(spread_hz) or spread_hz > config.maximum_frequency_spread_hz:
        raise MeasurementQualityError(
            f"Repeated CW estimates spread {spread_hz:.2f} Hz exceeds "
            f"{config.maximum_frequency_spread_hz:.2f} Hz"
        )
    measured_frequency_hz = config.rx_lo_hz + measured_if_hz
    error_hz = measured_frequency_hz - config.reference_frequency_hz
    error_ppm = error_hz / config.reference_frequency_hz * 1e6
    return FrequencyMeasurement(
        xo_correction=int(xo_correction),
        measured_if_hz=measured_if_hz,
        measured_frequency_hz=measured_frequency_hz,
        frequency_error_hz=error_hz,
        frequency_error_ppm=error_ppm,
        snr_db=float(np.median([estimate.snr_db for estimate in estimates])),
        spread_hz=spread_hz,
        capture_frequencies_hz=tuple(float(value) for value in frequencies),
    )
