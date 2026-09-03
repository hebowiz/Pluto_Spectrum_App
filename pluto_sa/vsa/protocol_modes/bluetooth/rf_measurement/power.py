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


def measure_pre_packet_emissions(
    iq: np.ndarray,
    *,
    packet_start_sample: int,
    packet_stop_sample: int,
    sample_rate_hz: float,
    search_duration_s: float = 8e-6,
) -> float | None:
    """Measure the -35 dB to -1 dB pre-packet rise time in linear power.

    A result is withheld unless the capture contains a below-threshold sample
    before the packet.  This prevents a capture that starts during an already
    active ramp from being reported as a passing measurement.
    """

    values = np.asarray(iq, dtype=np.complex128)
    start = max(0, int(packet_start_sample))
    stop = min(values.size, int(packet_stop_sample))
    rate = float(sample_rate_hz)
    if stop <= start or start < 1 or not np.isfinite(rate) or rate <= 0.0:
        return None
    average_power = float(np.mean(np.abs(values[start:stop]) ** 2))
    if not np.isfinite(average_power) or average_power <= 0.0:
        return None
    low = average_power * 10.0 ** (-35.0 / 10.0)
    high = average_power * 10.0 ** (-1.0 / 10.0)
    search_samples = max(2, int(np.ceil(float(search_duration_s) * rate)))
    search_start = max(0, start - search_samples)
    search_stop = min(values.size, start + search_samples + 1)
    power = np.abs(values[search_start:search_stop]) ** 2
    relative_start = start - search_start
    if relative_start < 1 or not np.any(power[:relative_start] <= low):
        return None
    high_candidates = np.flatnonzero(power[relative_start:] >= high)
    if not high_candidates.size:
        return None
    high_index = relative_start + int(high_candidates[0])
    low_crossings = np.flatnonzero(
        (power[:-1] <= low) & (power[1:] > low)
    ) + 1
    low_crossings = low_crossings[low_crossings <= high_index]
    if not low_crossings.size:
        return None
    low_index = int(low_crossings[-1])
    return float(high_index - low_index) / rate
