"""Bluetooth Lower-Tester-style channel filters for SIG RF measurements."""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache

import numpy as np
from scipy.signal import fftconvolve, firwin2


class BluetoothRFMeasurementFilterProfile(StrEnum):
    BR_1M = "br_1m"
    LE_1M = "le_1m"
    LE_2M = "le_2m"


_PROFILE_SCALE = {
    BluetoothRFMeasurementFilterProfile.BR_1M: 1.0,
    BluetoothRFMeasurementFilterProfile.LE_1M: 1.0,
    BluetoothRFMeasurementFilterProfile.LE_2M: 2.0,
}


@lru_cache(maxsize=16)
def rf_test_channel_filter_taps(
    sample_rate_hz: float,
    profile: BluetoothRFMeasurementFilterProfile | str,
    *,
    tap_count: int = 513,
) -> np.ndarray:
    """Design the measurement filter and preserve its tested RF response.

    The transition anchors represent the RF.TS/RFPHY.TS Lower Tester mask.
    A linear-phase FIR is used offline so no decoder compensation or fitted
    DUT parameters enter this measurement path.
    """

    sample_rate = float(sample_rate_hz)
    selected = BluetoothRFMeasurementFilterProfile(profile)
    count = int(tap_count)
    if count < 33 or count % 2 == 0:
        raise ValueError("Bluetooth RF measurement filter requires an odd tap count >= 33")
    scale = _PROFILE_SCALE[selected]
    required_stop_hz = 2_000_000.0 * scale
    nyquist_hz = sample_rate / 2.0
    if nyquist_hz <= required_stop_hz:
        raise ValueError(
            f"{selected.value} measurement requires sample rate above "
            f"{2.0 * required_stop_hz / 1e6:.3f} MS/s"
        )
    frequencies_hz = np.asarray(
        [
            0.0,
            540_000.0 * scale,
            650_000.0 * scale,
            950_000.0 * scale,
            1_800_000.0 * scale,
            2_000_000.0 * scale,
            nyquist_hz,
        ],
        dtype=np.float64,
    )
    gains_db = np.asarray([0.0, 0.0, -3.0, -14.0, -44.0, -60.0, -100.0])
    taps = firwin2(
        count,
        frequencies_hz,
        np.power(10.0, gains_db / 20.0),
        fs=sample_rate,
    )
    taps = np.asarray(taps, dtype=np.float64)
    taps.setflags(write=False)
    return taps


def apply_rf_test_channel_filter(
    iq: np.ndarray,
    *,
    sample_rate_hz: float,
    profile: BluetoothRFMeasurementFilterProfile | str,
) -> np.ndarray:
    """Apply the zero-delay offline measurement filter to complex IQ."""

    values = np.asarray(iq, dtype=np.complex128)
    if values.ndim != 1:
        raise ValueError("Bluetooth RF measurement IQ must be one-dimensional")
    if values.size == 0:
        return np.empty(0, dtype=np.complex128)
    taps = rf_test_channel_filter_taps(float(sample_rate_hz), profile)
    filtered = fftconvolve(values, taps, mode="same")
    return np.asarray(filtered, dtype=np.complex128)
