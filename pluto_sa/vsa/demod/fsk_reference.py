"""Shared FSK transmit and measurement frequency-pulse filters."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d


def gaussian_bt_sigma_samples(samples_per_symbol: int, bt: float) -> float:
    """Return the ETSI/R&S Gaussian impulse sigma on a sampled symbol grid.

    The analytic Gaussian has a -3 dB amplitude bandwidth ``B`` and uses
    ``BT`` as its dimensionless operating parameter.  R&S FPL K70 Appendix
    F.5 specifies the same definition.
    """
    sps = int(samples_per_symbol)
    value = float(bt)
    if sps < 2:
        raise ValueError("samples_per_symbol must be at least 2")
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("Gaussian BT must be positive")
    return float(np.sqrt(np.log(2.0)) * sps / (2.0 * np.pi * value))


def apply_gaussian_frequency_filter(
    values: np.ndarray,
    *,
    samples_per_symbol: int,
    bt: float,
) -> np.ndarray:
    """Apply the normalized analytic Gaussian filter to frequency samples."""
    samples = np.asarray(values, dtype=np.float64)
    return gaussian_filter1d(
        samples,
        sigma=max(0.5, gaussian_bt_sigma_samples(samples_per_symbol, bt)),
        mode="nearest",
    )


def fsk_reference_frequency_levels(
    symbols: np.ndarray,
    *,
    samples_per_symbol: int,
    transmit_gaussian_bt: float | None,
    measurement_gaussian_bt: float | None = None,
) -> np.ndarray:
    """Build an oversampled ideal FSK instantaneous-frequency waveform.

    Values are normalized to -1/+1 deviation.  The reference path follows the
    R&S model: symbol impulses pass through the transmit filter first and then
    through the same optional measurement filter as the measured frequency.
    """
    values = np.asarray(symbols, dtype=np.uint8)
    if values.ndim != 1 or np.any(values > 1):
        raise ValueError("symbols must be a one-dimensional binary array")
    sps = int(samples_per_symbol)
    levels = np.repeat(2.0 * values.astype(np.float64) - 1.0, sps)
    if transmit_gaussian_bt is not None:
        levels = apply_gaussian_frequency_filter(
            levels,
            samples_per_symbol=sps,
            bt=float(transmit_gaussian_bt),
        )
    if measurement_gaussian_bt is not None:
        levels = apply_gaussian_frequency_filter(
            levels,
            samples_per_symbol=sps,
            bt=float(measurement_gaussian_bt),
        )
    return levels

