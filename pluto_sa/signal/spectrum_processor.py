"""Spectrum processing."""

from __future__ import annotations

import numpy as np

from pluto_sa.config.spectrum_config import SpectrumConfig
from pluto_sa.signal.rbw import (
    apply_rbw_weighting,
    make_gaussian_rbw_kernel,
    resolve_rbw_hz,
)


class SpectrumProcessor:
    """Own FFT-related calculations independent from SDR I/O."""

    def __init__(self, config: SpectrumConfig) -> None:
        self.config = config
        self.window = np.hanning(config.fft_size)
        self.rbw_kernel = self.make_rbw_kernel()

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
        iq_windowed = iq * self.window
        spectrum = np.fft.fftshift(np.fft.fft(iq_windowed))
        n = self.config.fft_size
        coherent_gain = np.sum(self.window) / n
        spectrum = spectrum / n
        spectrum = spectrum / coherent_gain
        power_spectrum = np.abs(spectrum) ** 2
        filtered_power = apply_rbw_weighting(power_spectrum, self.rbw_kernel)
        return filtered_power

    def compute_spectrum(self, iq: np.ndarray) -> np.ndarray:
        filtered_power = self.compute_filtered_power(iq)
        power_db = 10.0 * np.log10(filtered_power + 1e-20)
        return power_db

    def make_rbw_kernel(self) -> np.ndarray:
        rbw_hz = resolve_rbw_hz(self.config.rbw_hz, self.config.bin_width_hz)
        return make_gaussian_rbw_kernel(rbw_hz, self.config.bin_width_hz)

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
        if len(self.window) != config.fft_size:
            self.window = np.hanning(config.fft_size)

        self.rbw_kernel = self.make_rbw_kernel()
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
