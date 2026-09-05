"""DECT FM-discriminator traces and RF-modulation measurement helpers.

This module deliberately keeps synchronization aids, display traces and ETSI
peak measurements as different quantities.  No smoothing or amplitude fitting
is applied to the measured FM waveform.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class DectModulationReference(str, Enum):
    """Frequency references offered by the CTS60 modulation display."""

    MEASURED = "Measured"
    NOMINAL = "Nominal"
    HALF_PEAK = "Half Peak"


@dataclass(frozen=True)
class DectFrequencyReferences:
    measured_hz: float
    nominal_hz: float
    half_peak_hz: float

    def value(self, reference: DectModulationReference | str) -> float:
        selected = DectModulationReference(reference)
        if selected is DectModulationReference.MEASURED:
            return self.measured_hz
        if selected is DectModulationReference.HALF_PEAK:
            return self.half_peak_hz
        return self.nominal_hz


def instantaneous_frequency(
    iq: np.ndarray, sample_rate_hz: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return the unfiltered one-sample phase-difference discriminator output."""

    values = np.asarray(iq, dtype=np.complex128)
    product = values[1:] * np.conj(values[:-1])
    frequency = np.angle(product) * float(sample_rate_hz) / (2.0 * np.pi)
    positions = np.arange(frequency.size, dtype=np.float64) + 0.5
    return frequency, positions


def measurement_bandwidth_hz(
    sample_rate_hz: float, usable_bandwidth_hz: float | None
) -> float:
    """Return the narrowest declared bandwidth in the unfiltered IQ path."""

    declared = (
        0.8 * float(sample_rate_hz)
        if usable_bandwidth_hz is None
        else float(usable_bandwidth_hz)
    )
    return min(float(sample_rate_hz), declared)


def frequency_references(
    frequency_hz: np.ndarray,
    sample_positions: np.ndarray,
    *,
    window_start_sample: float,
    window_stop_sample: float,
) -> DectFrequencyReferences:
    """Calculate Measured/Nominal/Half Peak over one selected time window."""

    selected = np.asarray(frequency_hz, dtype=np.float64)[
        (np.asarray(sample_positions) >= float(window_start_sample))
        & (np.asarray(sample_positions) < float(window_stop_sample))
    ]
    selected = selected[np.isfinite(selected)]
    if not selected.size:
        raise ValueError("DECT modulation reference window contains no samples")
    return DectFrequencyReferences(
        measured_hz=float(np.mean(selected)),
        nominal_hz=0.0,
        half_peak_hz=0.5 * (float(np.max(selected)) + float(np.min(selected))),
    )


def cts60_trace(
    frequency_hz: np.ndarray,
    sample_positions: np.ndarray,
    *,
    first_symbol_sample: float,
    samples_per_symbol: float,
    symbol_count: int,
    first_symbol_number: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate the measured waveform onto the CTS60 six-values/bit grid."""

    point_count = max(0, int(symbol_count)) * 6
    ordinal = np.arange(point_count, dtype=np.int64)
    symbol_ordinals = ordinal // 6
    symbols = symbol_ordinals + int(first_symbol_number)
    fractions = ordinal % 6
    positions = float(first_symbol_sample) + (
        symbol_ordinals.astype(np.float64) + fractions.astype(np.float64) / 6.0
    ) * float(samples_per_symbol)
    values = np.interp(
        positions,
        np.asarray(sample_positions, dtype=np.float64),
        np.asarray(frequency_hz, dtype=np.float64),
        left=np.nan,
        right=np.nan,
    )
    return values, positions, symbols, fractions


def eligible_peak_sample_mask(
    sample_positions: np.ndarray,
    *,
    bits: np.ndarray,
    first_symbol_sample: float,
    samples_per_symbol: float,
    loopback_start: int,
    loopback_stop: int,
    modulation_case: str,
) -> np.ndarray:
    """Build the standards-defined peak search windows on the FM sample grid."""

    positions = np.asarray(sample_positions, dtype=np.float64)
    mask = np.zeros(positions.size, dtype=bool)
    start_bit = max(0, int(loopback_start))
    stop_bit = min(np.asarray(bits).size, int(loopback_stop))
    if stop_bit <= start_bit + 2:
        return mask

    def include(bit_start: float, bit_stop: float) -> None:
        start = float(first_symbol_sample) + bit_start * float(samples_per_symbol)
        stop = float(first_symbol_sample) + bit_stop * float(samples_per_symbol)
        mask[:] |= (positions >= start) & (positions < stop)

    if str(modulation_case).startswith("Case A"):
        run_start = start_bit
        while run_start < stop_bit:
            run_stop = run_start + 1
            while run_stop < stop_bit and bits[run_stop] == bits[run_start]:
                run_stop += 1
            # One complete bit after and one complete bit before transitions
            # are excluded.  A four-bit run therefore retains its middle two.
            if run_stop - run_start >= 4:
                include(run_start + 1, run_stop - 1)
            run_start = run_stop
    elif str(modulation_case).startswith("Case B"):
        # Part 3 defines one continuous window from one bit after the first
        # transition to one bit before the last transition, and applies it to
        # both the first 16 S-field bits and the loopback field.
        include(1, 15)
        include(start_bit + 1, stop_bit - 1)
    else:
        # Arbitrary data are reference information only, not an ETSI verdict.
        include(start_bit, stop_bit)
    return mask


def peak_deviations(
    frequency_hz: np.ndarray,
    sample_positions: np.ndarray,
    eligible_mask: np.ndarray,
    *,
    bits: np.ndarray,
    first_symbol_sample: float,
    samples_per_symbol: float,
    reference_hz: float,
) -> tuple[float, float, np.ndarray]:
    """Return positive/negative extents and per-bit signed eligible peaks."""

    values = np.asarray(frequency_hz, dtype=np.float64)
    positions = np.asarray(sample_positions, dtype=np.float64)
    eligible = np.asarray(eligible_mask, dtype=bool)
    bit_peaks = np.full(np.asarray(bits).size, np.nan, dtype=np.float64)
    for index, bit in enumerate(np.asarray(bits, dtype=np.uint8)):
        start = float(first_symbol_sample) + index * float(samples_per_symbol)
        stop = start + float(samples_per_symbol)
        selected = values[eligible & (positions >= start) & (positions < stop)]
        if selected.size:
            bit_peaks[index] = float(np.max(selected) if bit else np.min(selected))
    positive = bit_peaks[np.asarray(bits, dtype=bool)]
    negative = bit_peaks[~np.asarray(bits, dtype=bool)]
    positive = positive[np.isfinite(positive)]
    negative = negative[np.isfinite(negative)]
    positive_extent = (
        float(np.max(positive) - reference_hz) if positive.size else float("nan")
    )
    negative_extent = (
        float(np.min(negative) - reference_hz) if negative.size else float("nan")
    )
    return positive_extent, negative_extent, bit_peaks
