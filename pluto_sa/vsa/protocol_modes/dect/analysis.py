"""Passive Classic DECT GFSK synchronization and transmitter measurements."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Mapping

import numpy as np
from scipy.optimize import least_squares

from pluto_protocol.dect.classic import DectClassicDecoder
from pluto_protocol.dect.rf_modulation import identify_rf_pattern
from pluto_protocol.model import (
    PacketAnalysisResult,
    PacketDecodeInput,
    PacketField,
    PacketSourceInfo,
)
from pluto_sa.vsa.demod.fsk_reference import fsk_reference_frequency_levels
from pluto_sa.vsa.model import IQRecording
from .generator import (
    DECT_SYMBOL_RATE_HZ,
    PACKET_SYMBOL_COUNTS,
    PP_SYNC_BITS,
    RFP_SYNC_BITS,
)
from .modulation import (
    DectFrequencyReferences,
    DECTDeviationBitResult,
    DectModulationReference,
    cts60_trace,
    deviation_bit_results,
    eligible_peak_sample_mask,
    frequency_references,
    instantaneous_frequency,
    measurement_bandwidth_hz,
    peak_deviations,
)
from .power_time import (
    DectNTPResult,
    DectPowerTimeResult,
    build_dect_power_measurement_paths,
    measure_dect_ntp,
    measure_dect_power_time,
)


_CARRIER_REPETITIONS = {
    "P00": 100,
    "P32": 10,
    "P32Z": 10,
    "P80": 5,
    "P80Z": 5,
}


def carrier_repetition_count(packet_type: str) -> int:
    """Return the RF-test repetition count for a recognized packet type."""

    return _CARRIER_REPETITIONS.get(str(packet_type), 100)


@dataclass(frozen=True)
class DectSummaryRow:
    section: str
    test_item: str
    value: str
    limit: str = "—"
    result: str = "—"


@dataclass(frozen=True)
class DectPacketResult:
    direction: str
    preamble_mode: str
    preamble_correlation: float
    sync_word_correlation: float
    packet_type: str
    nominal_frequency_hz: float
    measured_frequency_hz: float
    carrier_error_hz: float
    carrier_test_eligible: bool
    modulation_case: str
    modulation_test_eligible: bool
    positive_deviation_hz: float
    negative_deviation_hz: float
    symbol_rate_hz: float
    symbol_rate_error_ppm: float
    ntp: DectNTPResult
    attack_time_s: float | None
    release_time_s: float | None
    active_flatness_db: float
    power_time_pass: bool | None
    sync_score: float
    start_sample: int
    stop_sample: int
    p0_sample: float
    power_time_start_sample: float
    packet_end_sample: float
    power_time: DectPowerTimeResult
    packet_analysis: PacketAnalysisResult
    bits: np.ndarray
    symbol_centers: np.ndarray
    symbol_frequency_hz: np.ndarray
    bit_peak_frequency_hz: np.ndarray
    raw_fm_frequency_hz: np.ndarray
    raw_fm_sample: np.ndarray
    measurement_fm_frequency_hz: np.ndarray
    measurement_fm_sample: np.ndarray
    cts60_trace_frequency_hz: np.ndarray
    cts60_trace_sample: np.ndarray
    cts60_trace_symbol: np.ndarray
    cts60_trace_fraction: np.ndarray
    ideal_gfsk_frequency_hz: np.ndarray
    frequency_references: DectFrequencyReferences
    modulation_reference: DectModulationReference
    fitted_deviation_hz: float
    measurement_bandwidth_hz: float
    etsi_eligible_sample_mask: np.ndarray
    instantaneous_frequency_hz: np.ndarray
    instantaneous_frequency_sample: np.ndarray
    bit_measurement_mask: np.ndarray
    deviation_bit_results: tuple[DECTDeviationBitResult, ...]
    power_db: np.ndarray
    raw_measurement_power_db: np.ndarray
    power_time_power_db: np.ndarray
    ntp_power_db: np.ndarray
    summary_rows: tuple[DectSummaryRow, ...] = field(init=False)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "bits",
            "symbol_centers",
            "symbol_frequency_hz",
            "bit_peak_frequency_hz",
            "raw_fm_frequency_hz",
            "raw_fm_sample",
            "measurement_fm_frequency_hz",
            "measurement_fm_sample",
            "cts60_trace_frequency_hz",
            "cts60_trace_sample",
            "cts60_trace_symbol",
            "cts60_trace_fraction",
            "ideal_gfsk_frequency_hz",
            "etsi_eligible_sample_mask",
            "instantaneous_frequency_hz",
            "instantaneous_frequency_sample",
            "bit_measurement_mask",
            "power_db",
            "raw_measurement_power_db",
            "power_time_power_db",
            "ntp_power_db",
        ):
            value = np.asarray(getattr(self, name))
            if value.flags.writeable or not value.flags.owndata:
                value = np.array(value, copy=True)
                value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        object.__setattr__(self, "summary_rows", _summary_rows(self))

    @property
    def fields(self) -> tuple[PacketField, ...]:
        """Compatibility view of the authoritative shared packet field tree."""

        if not self.packet_analysis.root_fields:
            return ()
        return self.packet_analysis.root_fields[0].children

    def modulation_reference_hz(
        self, reference: DectModulationReference | str | None = None
    ) -> float:
        return self.frequency_references.value(reference or self.modulation_reference)


def _contiguous_ranges(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    padded = np.pad(np.asarray(mask, dtype=np.int8), (1, 1))
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return tuple((int(start), int(stop)) for start, stop in zip(starts, stops))


def _burst_ranges(
    iq: np.ndarray,
    samples_per_symbol: float,
    *,
    power: np.ndarray | None = None,
) -> tuple[tuple[int, int], ...]:
    power = (
        np.abs(np.asarray(iq, dtype=np.complex128)) ** 2
        if power is None
        else np.asarray(power, dtype=np.float64)
    )
    window = max(1, int(round(samples_per_symbol / 2.0)))
    if window > 1:
        envelope = np.convolve(power, np.ones(window) / window, mode="same")
    else:
        envelope = power
    # Long P80 bursts may occupy more than 95 % of a short imported fixture.
    # The low tail still represents the captured off-air padding in that case.
    noise = float(np.percentile(envelope, 1.0))
    active = float(np.percentile(envelope, 99.5))
    if not np.isfinite(active) or active <= max(noise * 4.0, np.finfo(float).tiny):
        raise RuntimeError("No DECT RF burst was detected")
    threshold = noise + 0.50 * (active - noise)
    mask = envelope >= threshold
    fill = max(1, int(round(2.0 * samples_per_symbol)))
    false_ranges = _contiguous_ranges(~mask)
    for start, stop in false_ranges:
        if start > 0 and stop < mask.size and stop - start <= fill:
            mask[start:stop] = True
    minimum = int(round(64.0 * samples_per_symbol))
    ranges = tuple(
        (start, stop)
        for start, stop in _contiguous_ranges(mask)
        if stop - start >= minimum
    )
    if not ranges:
        raise RuntimeError("No complete DECT packet-length burst was detected")
    return ranges


def _instantaneous_frequency(iq: np.ndarray, sample_rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    return instantaneous_frequency(iq, sample_rate_hz)


def _sample_frequency(
    frequency: np.ndarray, positions: np.ndarray, centers: np.ndarray
) -> np.ndarray:
    return np.interp(centers, positions, frequency, left=np.nan, right=np.nan)


def _sync_packet(
    frequency: np.ndarray,
    positions: np.ndarray,
    burst_start: int,
    nominal_sps: float,
) -> tuple[str, float, float, float]:
    """Locate p16 from the sync word, then derive the standards-defined p0."""

    best: tuple[float, str, float, float, float] | None = None
    start_min = max(0.0, burst_start - 2.0 * nominal_sps)
    # Normal and prolonged preambles place p16 about 16 or 32 symbols after
    # the RF burst begins.  Leave additional room for the attack threshold.
    start_max = burst_start + 48.0 * nominal_sps
    directions = (
        ("RFP", RFP_SYNC_BITS[16:], RFP_SYNC_BITS[:16]),
        ("PP", PP_SYNC_BITS[16:], PP_SYNC_BITS[:16]),
    )
    p16_values = np.arange(start_min, start_max + 0.01, 0.5)
    symbol_axis = np.arange(16, dtype=np.float64) + 0.5
    for rate_ppm in np.linspace(-500.0, 500.0, 9):
        sps = nominal_sps / (1.0 + rate_ppm * 1e-6)
        centers = p16_values[:, None] + symbol_axis[None, :] * sps
        observed = np.interp(
            centers.ravel(), positions, frequency, left=np.nan, right=np.nan
        ).reshape(centers.shape)
        preamble_centers = p16_values[:, None] + (
            np.arange(-16, 0, dtype=np.float64)[None, :] + 0.5
        ) * sps
        preamble_observed = np.interp(
            preamble_centers.ravel(),
            positions,
            frequency,
            left=np.nan,
            right=np.nan,
        ).reshape(preamble_centers.shape)
        finite = np.all(np.isfinite(observed), axis=1) & np.all(
            np.isfinite(preamble_observed), axis=1
        )
        centered = observed - np.mean(observed, axis=1, keepdims=True)
        norms = np.linalg.norm(centered, axis=1)
        finite &= norms > np.finfo(float).tiny
        preamble_centered = preamble_observed - np.mean(
            preamble_observed, axis=1, keepdims=True
        )
        preamble_norms = np.linalg.norm(preamble_centered, axis=1)
        for direction, bits, preamble_bits in directions:
            reference = 2.0 * bits.astype(np.float64) - 1.0
            preamble_reference = 2.0 * preamble_bits.astype(np.float64) - 1.0
            scores = (centered @ reference) / np.maximum(
                norms * np.linalg.norm(reference), np.finfo(float).tiny
            )
            preamble_scores = (preamble_centered @ preamble_reference) / np.maximum(
                preamble_norms * np.linalg.norm(preamble_reference),
                np.finfo(float).tiny,
            )
            valid = finite & (preamble_scores >= 0.60)
            if not np.any(valid):
                continue
            valid_indices = np.flatnonzero(valid)
            selected = int(valid_indices[np.argmax(scores[valid_indices])])
            separation = float(
                np.mean(observed[selected, bits == 1])
                - np.mean(observed[selected, bits == 0])
            )
            candidate = (
                float(scores[selected]),
                direction,
                float(p16_values[selected]),
                float(sps),
                separation,
            )
            if best is None or candidate[0] > best[0]:
                best = candidate
    if best is None or best[0] < 0.72 or best[4] <= 50_000.0:
        raise RuntimeError("DECT S-field synchronization failed")
    score, direction, p16, sps, separation = best
    return direction, float(p16 - 16.0 * sps), sps, score


def _pattern_correlation(
    observed: np.ndarray,
    bits: np.ndarray,
) -> float:
    values = np.asarray(observed, dtype=np.float64)
    if values.size != bits.size or not np.all(np.isfinite(values)):
        return 0.0
    centered = values - float(np.mean(values))
    reference = 2.0 * np.asarray(bits, dtype=np.float64) - 1.0
    denominator = float(np.linalg.norm(centered) * np.linalg.norm(reference))
    return 0.0 if denominator <= np.finfo(float).tiny else float(
        np.dot(centered, reference) / denominator
    )


def _s_field_correlations(
    frequency: np.ndarray,
    positions: np.ndarray,
    p0: float,
    sps: float,
    direction: str,
) -> tuple[float, float, float]:
    expected = RFP_SYNC_BITS if direction == "RFP" else PP_SYNC_BITS
    centers = p0 + (np.arange(32, dtype=np.float64) + 0.5) * sps
    observed = _sample_frequency(frequency, positions, centers)
    preamble = _pattern_correlation(observed[:16], expected[:16])
    sync_word = _pattern_correlation(observed[16:], expected[16:])
    overall = _pattern_correlation(observed, expected)
    return preamble, sync_word, overall


def _detect_prolonged_preamble(
    frequency: np.ndarray,
    positions: np.ndarray,
    power: np.ndarray,
    p0: float,
    sps: float,
    direction: str,
) -> tuple[bool, float]:
    expected = (RFP_SYNC_BITS if direction == "RFP" else PP_SYNC_BITS)[:16]
    centers = p0 + (np.arange(-16, 0, dtype=np.float64) + 0.5) * sps
    if centers[0] < 0.0 or centers[-1] >= power.size:
        return False, 0.0
    observed = _sample_frequency(frequency, positions, centers)
    correlation = _pattern_correlation(observed, expected)
    normal_centers = p0 + (np.arange(16, dtype=np.float64) + 0.5) * sps
    preceding_power = np.interp(centers, np.arange(power.size), power)
    normal_power = np.interp(normal_centers, np.arange(power.size), power)
    reference = float(np.median(normal_power))
    if not np.isfinite(reference) or reference <= np.finfo(float).tiny:
        return False, correlation
    powered = preceding_power >= reference * 10.0 ** (-3.0 / 10.0)
    return bool(correlation >= 0.85 and np.mean(powered) >= 0.80), correlation


def _refine_p0_crossing(
    frequency: np.ndarray,
    positions: np.ndarray,
    p0: float,
    sps: float,
    direction: str,
) -> float:
    centers = p0 + (np.arange(32, dtype=np.float64) + 0.5) * sps
    sync_bits = RFP_SYNC_BITS if direction == "RFP" else PP_SYNC_BITS
    observed = _sample_frequency(frequency, positions, centers)
    midpoint = 0.5 * (
        float(np.mean(observed[sync_bits == 1]))
        + float(np.mean(observed[sync_bits == 0]))
    )
    expected = p0 + 16.0 * sps
    candidates = np.flatnonzero(
        (positions[:-1] >= expected - 0.75 * sps)
        & (positions[:-1] <= expected + 0.75 * sps)
    )
    if not candidates.size:
        return p0
    centered = frequency - midpoint
    rising = direction == "RFP"
    if rising:
        crossings = candidates[
            (centered[candidates] <= 0.0) & (centered[candidates + 1] > 0.0)
        ]
    else:
        crossings = candidates[
            (centered[candidates] >= 0.0) & (centered[candidates + 1] < 0.0)
        ]
    if not crossings.size:
        return p0
    index = int(crossings[np.argmin(np.abs(positions[crossings] - expected))])
    left, right = centered[index], centered[index + 1]
    fraction = 0.0 if right == left else float(-left / (right - left))
    crossing = positions[index] + np.clip(fraction, 0.0, 1.0)
    return float(crossing - 16.0 * sps)


def _bit_means(
    frequency: np.ndarray,
    positions: np.ndarray,
    p0: float,
    sps: float,
    count: int,
) -> np.ndarray:
    results = np.full(count, np.nan, dtype=np.float64)
    starts = p0 + (np.arange(count, dtype=np.float64) + 0.2) * sps
    stops = p0 + (np.arange(count, dtype=np.float64) + 0.8) * sps
    first = np.searchsorted(positions, starts, side="left")
    last = np.searchsorted(positions, stops, side="left")
    for index, (begin, end) in enumerate(zip(first, last)):
        selected = frequency[int(begin) : int(end)]
        if selected.size:
            results[index] = float(np.mean(selected))
    return results


def _bit_peaks(
    frequency: np.ndarray,
    positions: np.ndarray,
    p0: float,
    sps: float,
    bits: np.ndarray,
) -> np.ndarray:
    """Return the signed peak frequency in each bit measurement interval."""

    results = np.full(bits.size, np.nan, dtype=np.float64)
    for index, bit in enumerate(bits):
        start = p0 + (index + 0.2) * sps
        stop = p0 + (index + 0.8) * sps
        selected = frequency[(positions >= start) & (positions < stop)]
        if selected.size:
            results[index] = float(np.max(selected) if bit else np.min(selected))
    return results


def _two_level_midpoint(values: np.ndarray) -> float:
    """Estimate the GFSK carrier as the midpoint of two level clusters."""

    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return float(np.nanmean(finite))
    low, high = np.percentile(finite, (20.0, 80.0))
    for _ in range(12):
        boundary = 0.5 * (low + high)
        lower = finite[finite <= boundary]
        upper = finite[finite > boundary]
        if not lower.size or not upper.size:
            break
        updated_low = float(np.mean(lower))
        updated_high = float(np.mean(upper))
        if abs(updated_low - low) + abs(updated_high - high) < 1e-6:
            low, high = updated_low, updated_high
            break
        low, high = updated_low, updated_high
    return float(0.5 * (low + high))


def _timing_from_transitions(
    frequency: np.ndarray,
    positions: np.ndarray,
    p0: float,
    sps: float,
    bits: np.ndarray,
    carrier_hz: float,
) -> tuple[float, float]:
    centered = np.asarray(frequency, dtype=np.float64) - carrier_hz
    centered = np.convolve(centered, np.ones(3) / 3.0, mode="same")
    indices: list[int] = []
    crossings: list[float] = []
    for index in np.flatnonzero(bits[1:] != bits[:-1]) + 1:
        expected = p0 + float(index) * sps
        rising = bool(bits[index] > bits[index - 1])
        crossing_mask = (
            (centered[:-1] <= 0.0) & (centered[1:] > 0.0)
            if rising
            else (centered[:-1] >= 0.0) & (centered[1:] < 0.0)
        )
        candidates = np.flatnonzero(
            (positions[:-1] >= expected - 0.45 * sps)
            & (positions[:-1] <= expected + 0.45 * sps)
            & crossing_mask
        )
        if not candidates.size:
            continue
        sample = int(candidates[np.argmin(np.abs(positions[candidates] - expected))])
        left, right = centered[sample], centered[sample + 1]
        fraction = 0.0 if right == left else float(-left / (right - left))
        indices.append(int(index))
        crossings.append(float(positions[sample] + np.clip(fraction, 0.0, 1.0)))
    if len(indices) < 6:
        return sps, p0
    x = np.asarray(indices, dtype=np.float64)
    y = np.asarray(crossings, dtype=np.float64)
    keep = np.ones(x.size, dtype=bool)
    slope, intercept = np.polyfit(x, y, 1)
    for _ in range(2):
        residual = y - (slope * x + intercept)
        median = float(np.median(residual[keep]))
        mad = float(np.median(np.abs(residual[keep] - median)))
        tolerance = max(0.08 * sps, 4.0 * 1.4826 * mad)
        updated = np.abs(residual - median) <= tolerance
        if np.count_nonzero(updated) < 6:
            break
        keep = updated
        slope, intercept = np.polyfit(x[keep], y[keep], 1)
    if not 0.98 * sps <= slope <= 1.02 * sps:
        return sps, p0
    return float(slope), float(intercept)


def _sync_timing(
    frequency: np.ndarray,
    positions: np.ndarray,
    p0: float,
    sps: float,
    direction: str,
) -> tuple[float, float, float]:
    """Refine timing and carrier from the known S-field only.

    Tentatively decoded payload bits must not feed the clock estimate: a bad
    payload decision otherwise moves every later field boundary in the same
    direction and can make an arbitrary packet appear to have a clock error.
    """

    sync_bits = RFP_SYNC_BITS if direction == "RFP" else PP_SYNC_BITS
    centers = p0 + (np.arange(sync_bits.size, dtype=np.float64) + 0.5) * sps
    observed = _sample_frequency(frequency, positions, centers)
    carrier = 0.5 * (
        float(np.nanmean(observed[sync_bits == 1]))
        + float(np.nanmean(observed[sync_bits == 0]))
    )
    refined_sps, refined_p0 = _timing_from_transitions(
        frequency, positions, p0, sps, sync_bits, carrier
    )
    if abs(refined_p0 - p0) <= sps:
        p0, sps = refined_p0, refined_sps
    p0 = _refine_p0_crossing(frequency, positions, p0, sps, direction)

    # Refine against the complete, known GFSK S-field.  Carrier and deviation
    # are nuisance parameters solved linearly for each timing candidate.  This
    # is substantially less sensitive to discriminator noise than locating an
    # individual zero crossing.
    reference_sps = 128
    guarded_bits = np.concatenate((sync_bits[:1], sync_bits, sync_bits[-1:]))
    reference = fsk_reference_frequency_levels(
        guarded_bits,
        samples_per_symbol=reference_sps,
        transmit_gaussian_bt=0.5,
    )
    # Keep the RF attack and the unknown symbol after S out of the fit.
    fit_start = p0 + 0.25 * sps
    fit_stop = p0 + 31.75 * sps
    selected = (positions >= fit_start) & (positions <= fit_stop)
    fit_positions = positions[selected]
    fit_frequency = frequency[selected]

    def residual(parameters: np.ndarray) -> np.ndarray:
        candidate_center, candidate_sps = parameters
        symbol_time = (fit_positions - candidate_center) / candidate_sps + 17.0
        model = np.interp(
            symbol_time * reference_sps,
            np.arange(reference.size, dtype=np.float64),
            reference,
        )
        design = np.column_stack((np.ones(model.size), model))
        coefficients, *_ = np.linalg.lstsq(design, fit_frequency, rcond=None)
        scale = max(abs(float(coefficients[1])), 1.0)
        return (fit_frequency - design @ coefficients) / scale

    if fit_positions.size >= 64:
        initial_center = p0 + 16.0 * sps
        fitted = least_squares(
            residual,
            np.array((initial_center, sps), dtype=np.float64),
            bounds=(
                np.array((initial_center - 0.75 * sps, 0.995 * sps)),
                np.array((initial_center + 0.75 * sps, 1.005 * sps)),
            ),
            x_scale=np.array((max(sps, 1.0), max(0.001 * sps, 1e-3))),
            loss="soft_l1",
            f_scale=0.05,
            max_nfev=100,
        )
        if fitted.success:
            fitted_center, sps = (float(value) for value in fitted.x)
            p0 = fitted_center - 16.0 * sps
    centers = p0 + (np.arange(sync_bits.size, dtype=np.float64) + 0.5) * sps
    observed = _sample_frequency(frequency, positions, centers)
    carrier = 0.5 * (
        float(np.nanmean(observed[sync_bits == 1]))
        + float(np.nanmean(observed[sync_bits == 0]))
    )
    return p0, sps, carrier


def _diagnostic_gfsk_fit(
    frequency: np.ndarray,
    positions: np.ndarray,
    *,
    bits: np.ndarray,
    actual_start: float,
    p0: float,
    sps: float,
) -> tuple[float, np.ndarray]:
    """Fit the known S-field for diagnostics and render an ideal BT=0.5 trace.

    The fit is never fed back into the measured FM waveform or ETSI deviation.
    """

    reference_sps = 128
    levels = fsk_reference_frequency_levels(
        np.asarray(bits, dtype=np.uint8),
        samples_per_symbol=reference_sps,
        transmit_gaussian_bt=0.5,
    )
    reference_axis = np.arange(levels.size, dtype=np.float64)
    symbol_axis = (np.asarray(positions, dtype=np.float64) - actual_start) / sps
    model = np.interp(
        symbol_axis * reference_sps,
        reference_axis,
        levels,
        left=np.nan,
        right=np.nan,
    )
    selected = (
        (positions >= p0 + 0.25 * sps)
        & (positions <= p0 + 31.75 * sps)
        & np.isfinite(model)
    )
    if np.count_nonzero(selected) < 16:
        return float("nan"), np.full(model.shape, np.nan, dtype=np.float64)
    design = np.column_stack((np.ones(np.count_nonzero(selected)), model[selected]))
    coefficients, *_ = np.linalg.lstsq(design, frequency[selected], rcond=None)
    carrier, deviation = (float(value) for value in coefficients)
    return deviation, carrier + deviation * model


def _smoothed_power(power: np.ndarray, samples_per_symbol: float) -> np.ndarray:
    window = max(1, int(round(0.5 * samples_per_symbol)))
    if window == 1:
        return np.asarray(power, dtype=np.float64)
    return np.convolve(
        np.asarray(power, dtype=np.float64),
        np.ones(window, dtype=np.float64) / window,
        mode="same",
    )


def _packet_length_hint_from_envelope(
    power: np.ndarray,
    p0: float,
    sps: float,
    burst_stop: int,
) -> float:
    """Return a coarse length hint; this is not a physical packet-end anchor."""

    envelope = _smoothed_power(power, sps)
    body_start = max(0, int(round(p0 + 8.0 * sps)))
    body_stop = min(envelope.size, max(body_start + 1, int(burst_stop - 4.0 * sps)))
    body = envelope[body_start:body_stop]
    if not body.size:
        return float(burst_stop)
    reference = float(np.median(body))
    high = reference * 10.0 ** (-1.0 / 10.0)
    search_start = max(body_start, int(round(burst_stop - 12.0 * sps)))
    search_stop = min(envelope.size, int(round(burst_stop + 4.0 * sps)))
    above = np.flatnonzero(envelope[search_start:search_stop] >= high)
    if not above.size:
        return float(burst_stop)
    return float(search_start + int(above[-1]) + 1)


def _packet_type(symbol_count: float) -> tuple[str, int]:
    candidates = tuple(PACKET_SYMBOL_COUNTS.items())
    name, count = min(candidates, key=lambda item: abs(symbol_count - item[1]))
    if abs(symbol_count - count) <= 16.0:
        return name, count
    variable = max(100, int(round(symbol_count)))
    return f"P00j ({variable} symbols)", variable


def _loopback_range_with_offset(
    packet_type: str,
    symbol_count: int,
    offset: int,
) -> tuple[int, int]:
    if packet_type == "P00":
        # RF test loopback range a16...a47 inside the 64-bit A-field.
        return offset + 48, min(offset + 80, offset + symbol_count)
    if packet_type in {"P32", "P32Z"}:
        return offset + 96, min(offset + 416, offset + symbol_count)
    if packet_type in {"P80", "P80Z"}:
        return offset + 96, min(offset + 896, offset + symbol_count)
    if symbol_count >= 100:
        return offset + 96, offset + max(96, symbol_count - 4)
    return offset + 32, offset + symbol_count


def _classify_pattern(loopback: np.ndarray, packet_type: str = "") -> tuple[str, bool, object]:
    identification = identify_rf_pattern(loopback, packet_type)
    # Only exact Case A and exact ETSI figure patterns are Part 1/2 eligible.
    eligible = bool(
        identification.case == "A"
        or (identification.case == "B" and identification.exact_etsi_pattern)
    )
    label = "Case A (00001111)" if identification.case == "A" else identification.label
    return label, eligible, identification


def _measurement_mask(
    bits: np.ndarray,
    modulation_case: str,
    loopback_start: int,
    loopback_stop: int,
) -> np.ndarray:
    mask = np.zeros(bits.size, dtype=bool)
    if modulation_case.startswith(("Case A", "Case B")):
        start = loopback_start
        while start < loopback_stop:
            stop = start + 1
            while stop < loopback_stop and bits[stop] == bits[start]:
                stop += 1
            if stop - start >= 4:
                mask[start + 1 : stop - 1] = True
            start = stop
    else:
        mask[loopback_start:loopback_stop] = True
    return mask


def _edge_time(
    power: np.ndarray,
    reference_power: float,
    center: float,
    sample_rate_hz: float,
    *,
    attack: bool,
) -> float | None:
    power = _smoothed_power(power, max(1.0, sample_rate_hz / DECT_SYMBOL_RATE_HZ))
    low = reference_power * 10.0 ** (-35.0 / 10.0)
    high = reference_power * 10.0 ** (-1.0 / 10.0)
    span = int(round(15e-6 * sample_rate_hz))
    if attack:
        lo, hi = max(0, int(center) - span), min(power.size, int(center) + 2)
        low_edges = np.flatnonzero((power[lo : hi - 1] <= low) & (power[lo + 1 : hi] > low)) + lo + 1
        high_edges = np.flatnonzero((power[lo : hi - 1] < high) & (power[lo + 1 : hi] >= high)) + lo + 1
        if not low_edges.size or not high_edges.size:
            return None
        high_index = int(high_edges[-1])
        before = low_edges[low_edges <= high_index]
        return None if not before.size else (high_index - int(before[-1])) / sample_rate_hz
    lo, hi = max(0, int(center) - 1), min(power.size, int(center) + span)
    high_edges = np.flatnonzero((power[lo : hi - 1] >= high) & (power[lo + 1 : hi] < high)) + lo + 1
    low_edges = np.flatnonzero((power[lo : hi - 1] > low) & (power[lo + 1 : hi] <= low)) + lo + 1
    if not high_edges.size or not low_edges.size:
        return None
    high_index = int(high_edges[0])
    after = low_edges[low_edges >= high_index]
    return None if not after.size else (int(after[0]) - high_index) / sample_rate_hz


def _display_signed(value: float, unit: str, scale: float = 1.0) -> str:
    return f"{value / scale:+.3f} {unit}"


def _summary_rows(result: DectPacketResult) -> tuple[DectSummaryRow, ...]:
    # A packet result is only one observation.  The dedicated UI promotes this
    # to PASS/FAIL after the RF-test repetition count has been accumulated.
    carrier_result = "MEASURING" if result.carrier_test_eligible else "N/A"
    carrier_value = _display_signed(result.carrier_error_hz, "kHz", 1e3) if result.carrier_test_eligible else "N/A"
    lower = 259_000.0 if result.modulation_case.startswith("Case A") else 202_000.0
    positive = np.asarray(
        [item.deviation_hz for item in result.deviation_bit_results if item.valid and item.bit_value == 1],
        dtype=np.float64,
    )
    negative = np.asarray(
        [-item.deviation_hz for item in result.deviation_bit_results if item.valid and item.bit_value == 0],
        dtype=np.float64,
    )
    positive_pass = bool(positive.size and np.all((positive > lower) & (positive < 403_000.0)))
    negative_pass = bool(negative.size and np.all((negative > lower) & (negative < 403_000.0)))
    single_pass = positive_pass and negative_pass
    overall_result = "FAIL" if result.modulation_test_eligible and not single_pass else "MEASURING"
    positive_result = "FAIL" if result.modulation_test_eligible and not positive_pass else "MEASURING"
    negative_result = "FAIL" if result.modulation_test_eligible and not negative_pass else "MEASURING"
    if not result.modulation_test_eligible:
        overall_result = positive_result = negative_result = "N/A"
    def range_text(values: np.ndarray, sign: str = "") -> str:
        if not values.size:
            return "N/A"
        return f"{sign}{np.min(values) / 1e3:.1f} to {sign}{np.max(values) / 1e3:.1f} kHz"
    def margin_text(values: np.ndarray) -> str:
        if not values.size:
            return "N/A"
        margin = min(float(np.min(values) - lower), float(403_000.0 - np.max(values)))
        return f"{margin / 1e3:+.1f} kHz"
    timing_limit = 10.0 if result.direction == "RFP" else 25.0
    timing_result = "PASS" if abs(result.symbol_rate_error_ppm) <= timing_limit else "FAIL"
    power_time_value = (
        f"Single packet {result.power_time.overall_status}"
        + (
            f" - {result.power_time.incomplete_reasons[0]}"
            if result.power_time.incomplete_reasons
            and result.power_time.overall_status == "INCOMPLETE"
            else ""
        )
    )
    power_time_result = result.power_time.overall_status
    rows = [
        DectSummaryRow(
            "RF PHY Measurements",
            "NTP",
            (
                f"{result.ntp.power_db:+.3f} {result.ntp.power_unit}"
                if result.ntp.available
                else f"N/A ({result.ntp.power_db:+.3f} {result.ntp.power_unit} reference)"
            ),
            "Measurement BW: 1 MHz",
            "N/A",
        ),
        DectSummaryRow(
            "RF PHY Measurements",
            "Power-Time Template",
            power_time_value,
            "EN 300 176-1 clause 9 / single packet",
            power_time_result,
        ),
        DectSummaryRow(
            "RF PHY Measurements",
            "GFSK Modulation Deviation",
            (
                f"+{result.positive_deviation_hz / 1e3:.1f} / "
                f"{result.negative_deviation_hz / 1e3:.1f} kHz"
                if result.modulation_test_eligible else "N/A"
            ),
            f"{lower / 1e3:.0f} kHz < |Df| < 403 kHz",
            overall_result,
        ),
        DectSummaryRow(
            "RF PHY Measurements",
            "Modulation Speed",
            f"{result.symbol_rate_hz / 1e6:.6f} MSym/s ({result.symbol_rate_error_ppm:+.2f} ppm)",
            f"±{timing_limit:.0f} ppm",
            timing_result,
        ),
        DectSummaryRow("RF PHY Measurements", "Modulation Test Pattern", result.modulation_case, "ETSI Part 1 / Part 2", "PASS" if result.modulation_test_eligible else "N/A"),
        DectSummaryRow("RF PHY Measurements", "Positive Peak Deviation", range_text(positive, "+"), f"{lower / 1e3:.0f} kHz < Df < 403 kHz", positive_result),
        DectSummaryRow("RF PHY Measurements", "Negative Peak Deviation", range_text(negative, "-"), f"{lower / 1e3:.0f} kHz < |Df| < 403 kHz", negative_result),
        DectSummaryRow("RF PHY Measurements", "Positive Worst Margin", margin_text(positive), "> 0 kHz", positive_result),
        DectSummaryRow("RF PHY Measurements", "Negative Worst Margin", margin_text(negative), "> 0 kHz", negative_result),
        DectSummaryRow("RF PHY Measurements", "Pattern Status", str(result.metadata.get("pattern_status", "N/A")), "Exact Air-side pattern", "PASS" if result.modulation_test_eligible else "N/A"),
        DectSummaryRow("RF PHY Measurements", "RF Carrier Frequency Accuracy", carrier_value, "±50 kHz", carrier_result),
        DectSummaryRow("Reference Information", "Direction", result.direction),
        DectSummaryRow("Reference Information", "Preamble Mode", result.preamble_mode),
        DectSummaryRow("Reference Information", "Detected packet type", result.packet_type),
        DectSummaryRow("Reference Information", "Payload pattern", result.modulation_case),
        DectSummaryRow("Reference Information", "Pattern Errors", f"{int(result.metadata.get('pattern_errors', 0))} / {int(result.metadata.get('loopback_bit_count', 0))}"),
        DectSummaryRow("Reference Information", "Maximum |DSV|", str(int(result.metadata.get("pattern_dsv_max", 0)))),
        DectSummaryRow("Reference Information", "Single Packet Modulation", "PASS" if single_pass and result.modulation_test_eligible else "FAIL" if result.modulation_test_eligible else "Not applicable"),
        DectSummaryRow("Reference Information", "Preamble correlation", f"{100.0 * result.preamble_correlation:.2f} %"),
        DectSummaryRow("Reference Information", "Sync Word correlation", f"{100.0 * result.sync_word_correlation:.2f} %"),
        DectSummaryRow("Reference Information", "S-field correlation", f"{100.0 * result.sync_score:.2f} %"),
    ]
    power_time_rows = [
        DectSummaryRow(
            "RF PHY Measurements",
            criterion.name,
            criterion.value,
            criterion.limit,
            criterion.status,
        )
        for criterion in result.power_time.criteria
    ]
    rows[3:3] = power_time_rows
    observed_rows: list[DectSummaryRow] = []
    if not result.carrier_test_eligible:
        observed_rows.append(DectSummaryRow("Reference Information", "Observed carrier frequency error", _display_signed(result.carrier_error_hz, "kHz", 1e3)))
    if not result.modulation_test_eligible:
        observed_rows.append(
            DectSummaryRow(
                "Reference Information",
                "Observed GFSK deviation",
                f"+{result.positive_deviation_hz / 1e3:.1f} / {result.negative_deviation_hz / 1e3:.1f} kHz",
            )
        )
    packet_type_index = next(
        index
        for index, row in enumerate(rows)
        if row.section == "Reference Information"
        and row.test_item == "Detected packet type"
    )
    rows[packet_type_index + 1 : packet_type_index + 1] = observed_rows
    rows.extend(
        (
            DectSummaryRow(
                "Reference Information",
                "Power-Time Reference Power",
                f"{result.power_time.reference_power_db:+.3f} {result.power_time.power_unit}",
            ),
            DectSummaryRow(
                "Reference Information",
                "Power-Time Measurement BW",
                f"{result.power_time.measurement_bandwidth_hz / 1e6:.3f} MHz",
            ),
            DectSummaryRow("Reference Information", "Nominal carrier", f"{result.nominal_frequency_hz / 1e6:.3f} MHz"),
            DectSummaryRow("Reference Information", "p0 sample", f"{result.p0_sample:.3f}"),
            DectSummaryRow(
                "Reference Information",
                "Packet symbols",
                str(result.metadata.get("physical_packet_symbol_count", result.bits.size)),
            ),
        )
    )
    return tuple(rows)


def analyze_dect_recording(
    recording: IQRecording,
    *,
    nominal_frequency_hz: float | None = None,
) -> tuple[DectPacketResult, ...]:
    """Detect and measure Classic DECT GFSK bursts in one IQ recording."""

    sample_rate = float(recording.sample_rate_hz)
    nominal_frequency = float(
        recording.center_frequency_hz
        if nominal_frequency_hz is None
        else nominal_frequency_hz
    )
    nominal_sps = sample_rate / DECT_SYMBOL_RATE_HZ
    if nominal_sps < 3.0:
        raise ValueError("DECT analysis requires at least 3 samples/symbol")
    analysis_bandwidth = measurement_bandwidth_hz(
        sample_rate, recording.usable_bandwidth_hz
    )
    rf_measurement_bandwidth_eligible = analysis_bandwidth >= 3_000_000.0
    frequency, positions = _instantaneous_frequency(recording.iq, sample_rate)
    raw_power = np.abs(np.asarray(recording.iq, dtype=np.complex128)) ** 2
    power_paths = build_dect_power_measurement_paths(recording)
    power_trace_offset = 30.0 if recording.amplitude_calibrated else 0.0
    raw_measurement_power_db = (
        10.0 * np.log10(np.maximum(power_paths.raw_power, np.finfo(float).tiny))
        + power_trace_offset
    )
    power_time_power_db = (
        10.0 * np.log10(np.maximum(power_paths.power_time_power, np.finfo(float).tiny))
        + power_trace_offset
    )
    ntp_power_db = (
        10.0 * np.log10(np.maximum(power_paths.one_mhz_power, np.finfo(float).tiny))
        + power_trace_offset
    )
    power_db = (
        10.0
        * np.log10(
            np.maximum(
                np.abs(recording.iq / recording.full_scale) ** 2,
                np.finfo(float).tiny,
            )
        )
        + recording.dbfs_to_dbm_offset_db
    )
    for shared in (frequency, positions, power_db):
        shared.setflags(write=False)
    results: list[DectPacketResult] = []
    for burst_start, burst_stop in _burst_ranges(
        recording.iq, nominal_sps, power=raw_power
    ):
        try:
            direction, p0, sps, _coarse_sync_word_score = _sync_packet(
                frequency, positions, burst_start, nominal_sps
            )
        except RuntimeError:
            continue
        p0 = _refine_p0_crossing(frequency, positions, p0, sps, direction)
        p0, sps, carrier = _sync_timing(
            frequency, positions, p0, sps, direction
        )
        preamble_score, sync_word_score, sync_score = _s_field_correlations(
            frequency, positions, p0, sps, direction
        )
        if preamble_score < 0.72 or sync_word_score < 0.72:
            continue
        power_length_hint = _packet_length_hint_from_envelope(
            raw_power, p0, sps, burst_stop
        )
        approximate_symbols = (power_length_hint - p0) / sps
        packet_type, symbol_count = _packet_type(approximate_symbols)
        physical_centers = p0 + (
            np.arange(symbol_count, dtype=np.float64) + 0.5
        ) * sps
        physical_frequency = _sample_frequency(
            frequency, positions, physical_centers
        )
        finite = np.isfinite(physical_frequency)
        if np.count_nonzero(finite) < min(64, symbol_count):
            continue
        bit_means = _bit_means(frequency, positions, p0, sps, symbol_count)
        carrier = _two_level_midpoint(bit_means)
        physical_bits = (bit_means > carrier).astype(np.uint8)
        packet_sps, packet_p0 = _timing_from_transitions(
            frequency,
            positions,
            p0,
            sps,
            physical_bits[:-4] if physical_bits.size > 68 else physical_bits,
            carrier,
        )
        if abs(packet_p0 - p0) <= 0.5 * sps and abs(packet_sps - sps) <= 0.001 * sps:
            p0, sps = packet_p0, packet_sps
            bit_means = _bit_means(frequency, positions, p0, sps, symbol_count)
            carrier = _two_level_midpoint(bit_means)
        preamble_score, sync_word_score, sync_score = _s_field_correlations(
            frequency, positions, p0, sps, direction
        )
        if preamble_score < 0.72 or sync_word_score < 0.72:
            continue
        prolonged, prolonged_score = _detect_prolonged_preamble(
            frequency, positions, raw_power, p0, sps, direction
        )
        preamble_mode = "Prolonged" if prolonged else "Normal"
        bit_offset = 16 if prolonged else 0
        actual_start = p0 - bit_offset * sps
        internal_symbol_count = symbol_count + bit_offset
        centers = actual_start + (
            np.arange(internal_symbol_count, dtype=np.float64) + 0.5
        ) * sps
        bit_means = _bit_means(
            frequency,
            positions,
            actual_start,
            sps,
            internal_symbol_count,
        )
        carrier = _two_level_midpoint(bit_means)
        bits = (bit_means > carrier).astype(np.uint8)
        loopback_start, loopback_stop = _loopback_range_with_offset(
            packet_type, symbol_count, bit_offset
        )
        loopback = bits[loopback_start:loopback_stop]
        # EN 300 176-1 clause 7.3(e): carrier frequency is the average of
        # measured absolute frequencies of the loopback bits.  Keep the
        # two-level midpoint above solely as the demodulation threshold.
        loopback_frequencies = bit_means[loopback_start:loopback_stop]
        loopback_frequencies = loopback_frequencies[np.isfinite(loopback_frequencies)]
        measured_carrier = (
            float(np.mean(loopback_frequencies))
            if loopback_frequencies.size
            else float(carrier)
        )
        modulation_case, eligible, pattern_identification = _classify_pattern(
            loopback, packet_type
        )
        measurement_mask = _measurement_mask(
            bits, modulation_case, loopback_start, loopback_stop
        )
        reference_window_start = actual_start + (loopback_start + 1) * sps
        reference_window_stop = actual_start + (loopback_stop - 1) * sps
        references = frequency_references(
            frequency,
            positions,
            window_start_sample=reference_window_start,
            window_stop_sample=reference_window_stop,
        )
        selected_reference = references.value(DectModulationReference.MEASURED)
        eligible_sample_mask = eligible_peak_sample_mask(
            positions,
            bits=bits,
            first_symbol_sample=actual_start,
            samples_per_symbol=sps,
            loopback_start=loopback_start,
            loopback_stop=loopback_stop,
            modulation_case=modulation_case,
        )
        positive_deviation, negative_deviation, bit_peaks = peak_deviations(
            frequency,
            positions,
            eligible_sample_mask,
            bits=bits,
            first_symbol_sample=actual_start,
            samples_per_symbol=sps,
            # EN 300 176-1 parts 1-3 use the carrier measured by the carrier
            # procedure.  The selectable CTS60 display reference is separate.
            reference_hz=measured_carrier,
        )
        per_bit_deviations = deviation_bit_results(
            bit_peaks,
            bits,
            measurement_mask,
            first_symbol_sample=actual_start,
            samples_per_symbol=sps,
            sample_rate_hz=sample_rate,
            reference_hz=measured_carrier,
            modulation_case=modulation_case,
            first_symbol_number=-bit_offset,
        )
        measured_positive = bit_peaks[measurement_mask & (bits == 1)]
        measured_negative = bit_peaks[measurement_mask & (bits == 0)]
        measured_deviations = np.concatenate(
            (measured_positive - measured_carrier, measured_carrier - measured_negative)
        )
        measured_deviations = measured_deviations[np.isfinite(measured_deviations)]
        if not measured_deviations.size:
            measured_deviations = np.abs(
                frequency[eligible_sample_mask] - measured_carrier
            )
        cts_frequency, cts_positions, cts_symbols, cts_fractions = cts60_trace(
            frequency,
            positions,
            first_symbol_sample=actual_start,
            samples_per_symbol=sps,
            symbol_count=internal_symbol_count,
            first_symbol_number=-bit_offset,
        )
        fitted_deviation, ideal_frequency = _diagnostic_gfsk_fit(
            frequency,
            positions,
            bits=bits,
            actual_start=actual_start,
            p0=p0,
            sps=sps,
        )
        packet_end = p0 + symbol_count * sps
        start_sample = max(0, int(np.floor(p0)))
        stop_sample = min(recording.sample_count, int(np.ceil(packet_end)))
        nominal_pluto_power = bool(
            recording.metadata.get("nominal_pluto_amplitude", False)
            or recording.metadata.get("nominal_pluto_amplitude_inferred", False)
        )
        power_time = measure_dect_power_time(
            recording,
            p0_sample=p0,
            power_time_start_sample=actual_start,
            packet_end_sample=packet_end,
            measurement_bandwidth_hz=analysis_bandwidth,
            measurement_paths=power_paths,
        )
        ntp = measure_dect_ntp(
            power_paths,
            p0_sample=p0,
            packet_end_sample=packet_end,
        )
        attack_time = power_time.attack_time_s
        release_time = power_time.release_time_s
        active = _smoothed_power(raw_power, sps)[
            min(stop_sample, start_sample + max(1, int(4 * sps))) :
            max(start_sample + 1, stop_sample - max(1, int(4 * sps)))
        ]
        active_db = 10.0 * np.log10(np.maximum(active, np.finfo(float).tiny))
        flatness = float(np.percentile(active_db, 99.0) - np.percentile(active_db, 1.0))
        power_time_pass = (
            True if power_time.overall_status == "PASS"
            else False if power_time.overall_status == "FAIL"
            else None
        )
        packet_analysis = DectClassicDecoder().decode(
            PacketDecodeInput(
                bits,
                protocol_hint="dect.classic",
                phy_hint="2-level GFSK",
                source=PacketSourceInfo(
                    source_kind="vsa_demodulated",
                    packet_index=len(results),
                    center_frequency_hz=nominal_frequency,
                    start_sample=start_sample,
                    stop_sample=stop_sample,
                ),
                context={
                    "direction": direction,
                    "packet_type": packet_type,
                    "preamble_mode": preamble_mode,
                    "p0_internal_bit": bit_offset,
                },
            )
        )
        results.append(
            DectPacketResult(
                direction=direction,
                preamble_mode=preamble_mode,
                preamble_correlation=preamble_score,
                sync_word_correlation=sync_word_score,
                packet_type=packet_type,
                nominal_frequency_hz=nominal_frequency,
                measured_frequency_hz=nominal_frequency + measured_carrier,
                carrier_error_hz=measured_carrier,
                carrier_test_eligible=modulation_case.startswith("Case A"),
                modulation_case=modulation_case,
                modulation_test_eligible=eligible and rf_measurement_bandwidth_eligible,
                positive_deviation_hz=positive_deviation,
                negative_deviation_hz=negative_deviation,
                symbol_rate_hz=sample_rate / sps,
                symbol_rate_error_ppm=(sample_rate / sps / DECT_SYMBOL_RATE_HZ - 1.0) * 1e6,
                ntp=ntp,
                attack_time_s=attack_time,
                release_time_s=release_time,
                active_flatness_db=flatness,
                power_time_pass=power_time_pass,
                sync_score=sync_score,
                start_sample=start_sample,
                stop_sample=stop_sample,
                p0_sample=p0,
                power_time_start_sample=actual_start,
                packet_end_sample=packet_end,
                power_time=power_time,
                packet_analysis=packet_analysis,
                bits=bits,
                symbol_centers=centers,
                symbol_frequency_hz=bit_means,
                bit_peak_frequency_hz=bit_peaks,
                raw_fm_frequency_hz=frequency,
                raw_fm_sample=positions,
                measurement_fm_frequency_hz=frequency,
                measurement_fm_sample=positions,
                cts60_trace_frequency_hz=cts_frequency,
                cts60_trace_sample=cts_positions,
                cts60_trace_symbol=cts_symbols,
                cts60_trace_fraction=cts_fractions,
                ideal_gfsk_frequency_hz=ideal_frequency,
                frequency_references=references,
                modulation_reference=DectModulationReference.MEASURED,
                fitted_deviation_hz=fitted_deviation,
                measurement_bandwidth_hz=analysis_bandwidth,
                etsi_eligible_sample_mask=eligible_sample_mask,
                instantaneous_frequency_hz=frequency,
                instantaneous_frequency_sample=positions,
                bit_measurement_mask=measurement_mask,
                deviation_bit_results=per_bit_deviations,
                power_db=power_db,
                raw_measurement_power_db=raw_measurement_power_db,
                power_time_power_db=power_time_power_db,
                ntp_power_db=ntp_power_db,
                metadata={
                    "burst_start_sample": burst_start,
                    "burst_stop_sample": burst_stop,
                    "power_length_hint_sample": power_length_hint,
                    "actual_preamble_start_sample": actual_start,
                    "power_time_start_sample": actual_start,
                    "power_time_measurement_bandwidth_hz": power_paths.power_time_measurement_bandwidth_hz,
                    "idle_power_measurement_bandwidth_hz": 1_000_000.0,
                    "power_time_reference_power_dbm": (
                        power_time.reference_power_db if recording.amplitude_calibrated else None
                    ),
                    "power_time_reference_power_w": (
                        power_time.reference_power_linear if recording.amplitude_calibrated else None
                    ),
                    "ntp_power_dbm": ntp.power_db if ntp.available else None,
                    "ntp_power_w": ntp.power_linear if ntp.available else None,
                    "ntp_measurement_bandwidth_hz": power_paths.ntp_measurement_bandwidth_hz,
                    "power_time_filter_type": power_paths.power_time_filter_type,
                    "ntp_filter_type": power_paths.ntp_filter_type,
                    "power_time_filter_group_delay_samples": power_paths.power_time_filter_group_delay_samples,
                    "ntp_filter_group_delay_samples": power_paths.ntp_filter_group_delay_samples,
                    "measurement_filter_delay_compensated": power_paths.filter_delay_compensated,
                    "power_measurement_input_usable_bandwidth_hz": power_paths.input_usable_bandwidth_hz,
                    "power_measurement_input": (
                        "Analysis Channel output"
                        if recording.metadata.get("analysis_channel_applied", False)
                        else "Capture/common DECT IQ"
                    ),
                    "physical_packet_symbol_count": symbol_count,
                    "prolonged_preamble_correlation": prolonged_score,
                    "nominal_pluto_power": nominal_pluto_power,
                    "samples_per_symbol": sps,
                    "capture_sample_rate_hz": sample_rate,
                    "analysis_sample_rate_hz": sample_rate,
                    "pluto_rx_bandwidth_hz": recording.metadata.get(
                        "actual_rf_bandwidth_hz", recording.usable_bandwidth_hz
                    ),
                    "measurement_filter_applied": False,
                    "measurement_filter_name": "None",
                    "transmit_gaussian_bt": 0.5,
                    "drift_compensation_applied": False,
                    "modulation_reference": DectModulationReference.MEASURED.value,
                    "modulation_reference_hz": selected_reference,
                    "clause_7_carrier_reference_hz": measured_carrier,
                    "loopback_bit_range": (loopback_start, loopback_stop),
                    "loopback_bit_count": int(loopback.size),
                    "minimum_measured_deviation_hz": (
                        float(np.min(measured_deviations)) if measured_deviations.size else float("nan")
                    ),
                    "maximum_measured_deviation_hz": (
                        float(np.max(measured_deviations)) if measured_deviations.size else float("nan")
                    ),
                    "pattern_errors": pattern_identification.pattern_errors,
                    "pattern_status": (
                        "Valid"
                        if eligible
                        else "Generic / Part 2 not applicable"
                        if pattern_identification.case == "B"
                        else "Not applicable"
                    ),
                    "pattern_exact_etsi": pattern_identification.exact_etsi_pattern,
                    "pattern_figure": pattern_identification.figure,
                    "pattern_dsv_max": pattern_identification.dsv_max,
                    "rf_measurement_bandwidth_eligible": rf_measurement_bandwidth_eligible,
                },
            )
        )
    if not results:
        raise RuntimeError("No synchronized DECT packet was found")
    # Idle power is defined only between adjacent physical packets.  Resolve it
    # after all synchronized packet starts are known, without using the coarse
    # envelope edge as a substitute for p0/p-16.
    resolved: list[DectPacketResult] = []
    for index, result in enumerate(results):
        next_start = (
            results[index + 1].power_time_start_sample
            if index + 1 < len(results)
            else None
        )
        power_time = measure_dect_power_time(
            recording,
            p0_sample=result.p0_sample,
            power_time_start_sample=result.power_time_start_sample,
            packet_end_sample=result.packet_end_sample,
            next_power_time_start_sample=next_start,
            measurement_bandwidth_hz=analysis_bandwidth,
            measurement_paths=power_paths,
        )
        resolved.append(
            replace(
                result,
                power_time=power_time,
                attack_time_s=power_time.attack_time_s,
                release_time_s=power_time.release_time_s,
                power_time_pass=(
                    True if power_time.overall_status == "PASS"
                    else False if power_time.overall_status == "FAIL"
                    else None
                ),
            )
        )
    return tuple(resolved)
