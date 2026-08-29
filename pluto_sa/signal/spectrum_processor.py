"""Spectrum processing."""

from __future__ import annotations

import numpy as np
from scipy import fft as scipy_fft

from pluto_sa.config.spectrum_config import SpectrumConfig
from pluto_sa.signal.fft_filterbank import (
    GaussianFFTFilterBankDesign,
    design_gaussian_fft_filterbank,
)


class SpectrumProcessor:
    """Own FFT-related calculations independent from SDR I/O."""

    def __init__(self, config: SpectrumConfig) -> None:
        self.config = config
        self.window, self.filterbank_design = self.make_analysis_window()
        self._fft_window = np.asarray(self.window, dtype=np.float32)
        self._fft_amplitude_scale = np.float32(1.0 / np.sum(self.window))

        self.freq_axis_hz = np.fft.fftshift(
            np.fft.fftfreq(config.fft_size, d=1.0 / config.sample_rate_hz)
        )

        n = config.fft_size
        guard_bins_each_side = int(round(n * config.guard_ratio))
        self.display_slice = slice(guard_bins_each_side, n - guard_bins_each_side)
        self.update_center_frequency(config.center_freq_hz)

    def compute_filtered_power(self, iq: np.ndarray) -> np.ndarray:
        if self.config.remove_dc_offset:
            iq = iq - np.mean(iq)
        iq_windowed = np.asarray(iq, dtype=np.complex64) * self._fft_window
        spectrum = scipy_fft.fft(iq_windowed, workers=1)
        spectrum *= self._fft_amplitude_scale
        power = spectrum.real * spectrum.real + spectrum.imag * spectrum.imag
        return scipy_fft.fftshift(power)

    def compute_filtered_power_batch(self, iq_frames: np.ndarray) -> np.ndarray:
        """Return linear-power spectra for a batch of contiguous FFT frames."""
        frames = np.asarray(iq_frames)
        if frames.ndim != 2 or frames.shape[1] != int(self.config.fft_size):
            raise ValueError("iq_frames must have shape (frame_count, fft_size)")
        if self.config.remove_dc_offset:
            frames = frames - np.mean(frames, axis=1, keepdims=True)
        windowed = np.asarray(frames, dtype=np.complex64) * self._fft_window[np.newaxis, :]
        spectrum = scipy_fft.fft(windowed, axis=1, workers=1)
        spectrum *= self._fft_amplitude_scale
        power = spectrum.real * spectrum.real + spectrum.imag * spectrum.imag
        return scipy_fft.fftshift(power, axes=1)

    def compute_spectrum(self, iq: np.ndarray) -> np.ndarray:
        filtered_power = self.compute_filtered_power(iq)
        power_db = 10.0 * np.log10(filtered_power + 1e-20)
        return power_db

    def make_analysis_window(self) -> tuple[np.ndarray, GaussianFFTFilterBankDesign]:
        return design_gaussian_fft_filterbank(
            float(self.config.sample_rate_hz),
            int(self.config.fft_size),
            self.config.rbw_hz,
        )

    def extract_display_spectrum(self, power_db_full: np.ndarray) -> np.ndarray:
        return power_db_full[self.display_slice]

    def update_center_frequency(self, center_freq_hz: int) -> None:
        self.config.center_freq_hz = center_freq_hz
        self.freq_axis_abs_ghz = (self.freq_axis_hz + center_freq_hz) / 1e9
        self.freq_axis_display_ghz = self.freq_axis_abs_ghz[self.display_slice]
        self.freq_axis_display_ghz_dec = self.freq_axis_display_ghz[
            :: self.config.waterfall_decimation
        ]

    def update_span_related(self, config: SpectrumConfig) -> None:
        self.config = config
        self.window, self.filterbank_design = self.make_analysis_window()
        self._fft_window = np.asarray(self.window, dtype=np.float32)
        self._fft_amplitude_scale = np.float32(1.0 / np.sum(self.window))
        self.freq_axis_hz = np.fft.fftshift(
            np.fft.fftfreq(config.fft_size, d=1.0 / config.sample_rate_hz)
        )
        guard_bins_each_side = int(round(config.fft_size * config.guard_ratio))
        self.display_slice = slice(guard_bins_each_side, config.fft_size - guard_bins_each_side)
        self.update_center_frequency(config.center_freq_hz)

    def get_display_freq_axis_ghz(self) -> np.ndarray:
        return self.freq_axis_display_ghz

    def get_decimated_display_freq_axis_ghz(self) -> np.ndarray:
        return self.freq_axis_display_ghz_dec

    def detect_peak(self, power_db_display: np.ndarray) -> tuple[float, float]:
        peak_idx = int(np.argmax(power_db_display))
        peak_freq = self.freq_axis_display_ghz[peak_idx]
        peak_val = power_db_display[peak_idx]
        return peak_freq, peak_val
