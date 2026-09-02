"""Windowed Bluetooth RF power measurements."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RFPowerResult:
    average_dbm: float
    peak_dbm: float
    peak_to_average_db: float
    start_sample: int
    stop_sample: int


def measure_burst_power(
    iq: np.ndarray,
    *,
    full_scale: float,
    dbfs_to_dbm_offset_db: float,
    start_sample: int,
    stop_sample: int,
    central_fraction: float = 0.8,
) -> RFPowerResult:
    """Measure a central RF-test window while excluding burst transients."""

    values = np.asarray(iq, dtype=np.complex128)
    start = max(0, int(start_sample))
    stop = min(values.size, int(stop_sample))
    if stop <= start:
        raise ValueError("RF power measurement range is empty")
    fraction = float(central_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("central_fraction must be in (0, 1]")
    trim = int(np.floor((stop - start) * (1.0 - fraction) / 2.0))
    used_start = start + trim
    used_stop = stop - trim
    normalized_power = np.abs(values[used_start:used_stop] / float(full_scale)) ** 2
    mean_power = float(np.mean(normalized_power))
    peak_power = float(np.max(normalized_power))
    floor = np.finfo(np.float64).tiny
    average_dbm = 10.0 * np.log10(max(mean_power, floor)) + float(dbfs_to_dbm_offset_db)
    peak_dbm = 10.0 * np.log10(max(peak_power, floor)) + float(dbfs_to_dbm_offset_db)
    return RFPowerResult(
        average_dbm=average_dbm,
        peak_dbm=peak_dbm,
        peak_to_average_db=peak_dbm - average_dbm,
        start_sample=used_start,
        stop_sample=used_stop,
    )


def measure_relative_power(reference: RFPowerResult, payload: RFPowerResult) -> float:
    return float(payload.average_dbm - reference.average_dbm)
