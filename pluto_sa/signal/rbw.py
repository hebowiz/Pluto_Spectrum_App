"""Common RBW helpers shared by RT SA and Sweep SA."""

from __future__ import annotations

import numpy as np


def make_gaussian_rbw_kernel(rbw_hz: float, bin_width_hz: float) -> np.ndarray:
    """Build a non-normalized Gaussian RBW kernel from an FWHM RBW definition."""
    sigma_hz = rbw_hz / 2.355
    sigma_bins = sigma_hz / bin_width_hz

    if sigma_bins <= 0.0:
        return np.array([1.0], dtype=np.float64)

    half_width = max(1, int(np.ceil(6.0 * sigma_bins)))
    x = np.arange(-half_width, half_width + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
    return kernel


def resolve_rbw_hz(rbw_hz: float | None, bin_width_hz: float) -> float:
    """Resolve the effective RBW while preserving the current RT SA behavior."""
    if rbw_hz is None:
        return bin_width_hz
    return max(bin_width_hz, rbw_hz)


def apply_rbw_weighting(power_spectrum: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply RBW weighting as frequency-direction energy integration."""
    return np.convolve(power_spectrum, kernel, mode="same")
