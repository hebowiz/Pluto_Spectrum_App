"""Spectrum processing."""

from __future__ import annotations

import numpy as np

from pluto_sa.config.spectrum_config import SpectrumConfig


class SpectrumProcessor:
    """Own FFT-related calculations independent from SDR I/O."""

    def __init__(self, config: SpectrumConfig) -> None:
        self.config = config
        self.window = np.hanning(config.fft_size)

        self.freq_axis_hz = np.fft.fftshift(
            np.fft.fftfreq(config.fft_size, d=1.0 / config.sample_rate_hz)
        )
        self.freq_axis_abs_ghz = (self.freq_axis_hz + config.center_freq_hz) / 1e9

        n = config.fft_size
        guard_bins_each_side = int(round(n * config.guard_ratio))
        self.display_slice = slice(guard_bins_each_side, n - guard_bins_each_side)

        self.freq_axis_display_ghz = self.freq_axis_abs_ghz[self.display_slice]
        self.freq_axis_display_ghz_dec = self.freq_axis_display_ghz[
            :: config.waterfall_decimation
        ]

    def compute_spectrum(self, iq: np.ndarray) -> np.ndarray:
        iq = iq - np.mean(iq)
        iq_windowed = iq * self.window
        spectrum = np.fft.fftshift(np.fft.fft(iq_windowed))
        power_db = 20.0 * np.log10(np.abs(spectrum) + 1e-12)
        return power_db

    def extract_display_spectrum(self, power_db_full: np.ndarray) -> np.ndarray:
        return power_db_full[self.display_slice]

    def get_display_freq_axis_ghz(self) -> np.ndarray:
        return self.freq_axis_display_ghz

    def get_decimated_display_freq_axis_ghz(self) -> np.ndarray:
        return self.freq_axis_display_ghz_dec

    def detect_peak(self, power_db_display: np.ndarray) -> tuple[float, float]:
        peak_idx = int(np.argmax(power_db_display))
        peak_freq = self.freq_axis_display_ghz[peak_idx]
        peak_val = power_db_display[peak_idx]
        return peak_freq, peak_val
