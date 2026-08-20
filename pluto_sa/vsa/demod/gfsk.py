"""Access-pattern-aided binary GFSK demodulation."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.optimize import minimize_scalar
from scipy.signal import resample_poly

from pluto_sa.vsa.demod.fsk_reference import (
    apply_gaussian_frequency_filter,
    fsk_reference_frequency_levels,
)


@dataclass(frozen=True)
class GFSKDemodulationResult:
    bits: np.ndarray
    symbol_frequency_hz: np.ndarray
    drift_compensated_symbol_frequency_hz: np.ndarray
    symbol_time_s: np.ndarray
    access_start_bit: int
    access_start_sample: int
    access_correlation: float
    access_bit_errors: int
    carrier_frequency_offset_hz: float
    carrier_frequency_drift_hz_per_s: float
    frequency_deviation_hz: float
    timing_phase_samples: int
    analysis_sample_rate_hz: float
    samples_per_symbol: int
    iq_inverted: bool
    complemented_pattern_match: bool
    frequency_model_timing_offset_samples: float
    applied_timing_offset_samples: float
    timing_correction_accepted: bool
    frequency_model_residual_rms_hz: float
    frequency_model_no_drift_residual_rms_hz: float
    drift_model_accepted: bool
    candidate_drift_hz_per_s: float
    drift_model_residual_rms_hz: float
    drift_excursion_hz: float
    drift_bic_improvement: float
    drift_rejection_reason: str
    timing_confidence: float
    estimation_sample_count: int
    burst_ranges: tuple[tuple[int, int], ...]
    selected_match_index: int
    eligible_match_count: int
    detected_match_count: int

    def __post_init__(self) -> None:
        arrays = {
            "bits": (self.bits, np.uint8),
            "symbol_frequency_hz": (self.symbol_frequency_hz, np.float64),
            "drift_compensated_symbol_frequency_hz": (
                self.drift_compensated_symbol_frequency_hz,
                np.float64,
            ),
            "symbol_time_s": (self.symbol_time_s, np.float64),
        }
        for name, (values, dtype) in arrays.items():
            owned = np.array(values, dtype=dtype, copy=True)
            owned.flags.writeable = False
            object.__setattr__(self, name, owned)


def _validate_bits(bits: np.ndarray) -> np.ndarray:
    values = np.asarray(bits, dtype=np.uint8)
    if values.ndim != 1 or values.size < 8:
        raise ValueError("access_bits must be a one-dimensional pattern of at least 8 bits")
    if np.any(values > 1):
        raise ValueError("access_bits must contain only zero and one")
    return values


def _resample_for_symbols(
    iq: np.ndarray,
    sample_rate_hz: float,
    symbol_rate_hz: float,
    samples_per_symbol: int,
) -> tuple[np.ndarray, float]:
    target_rate_hz = float(symbol_rate_hz) * int(samples_per_symbol)
    ratio = Fraction(target_rate_hz / float(sample_rate_hz)).limit_denominator(4096)
    values = resample_poly(iq, ratio.numerator, ratio.denominator)
    actual_rate_hz = float(sample_rate_hz) * ratio.numerator / ratio.denominator
    relative_error = abs(actual_rate_hz - target_rate_hz) / target_rate_hz
    if relative_error > 1e-5:
        raise ValueError("sample rate cannot be resampled accurately enough")
    return np.asarray(values, dtype=np.complex128), actual_rate_hz


def _instantaneous_frequency(iq: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    if iq.size < 2:
        raise ValueError("IQ recording is too short")
    result = np.angle(iq[1:] * np.conj(iq[:-1])) * sample_rate_hz / (2.0 * np.pi)
    return np.concatenate(([result[0]], result))


def prepare_fsk_frequency(
    iq: np.ndarray,
    *,
    sample_rate_hz: float,
    symbol_rate_hz: float,
    gaussian_bt: float | None,
    samples_per_symbol: int = 8,
) -> tuple[np.ndarray, float]:
    """Resample FSK IQ and return the measured instantaneous frequency.

    With ``gaussian_bt`` set, the instantaneous-frequency waveform passes
    through the same Gaussian Auto measurement filter used by the FSK
    demodulator.  ``None`` keeps only the analysis-rate resampling.
    """
    waveform, analysis_rate_hz = _resample_for_symbols(
        np.asarray(iq),
        float(sample_rate_hz),
        float(symbol_rate_hz),
        int(samples_per_symbol),
    )
    frequency_hz = _instantaneous_frequency(waveform, analysis_rate_hz)
    if gaussian_bt is not None:
        frequency_hz = apply_gaussian_frequency_filter(
            frequency_hz,
            samples_per_symbol=int(samples_per_symbol),
            bt=float(gaussian_bt),
        )
    return np.asarray(frequency_hz, dtype=np.float64), analysis_rate_hz


def _detect_bursts(
    iq: np.ndarray,
    *,
    sample_rate_hz: float,
    symbol_rate_hz: float,
    threshold_db: float = 6.0,
    minimum_symbols: int = 16,
) -> tuple[tuple[int, int], ...]:
    window = max(1, int(round(sample_rate_hz / symbol_rate_hz)))
    smoothed = uniform_filter1d(np.abs(iq) ** 2, size=window, mode="nearest")
    noise_power = float(np.percentile(smoothed, 5.0))
    peak_power = float(np.max(smoothed))
    if peak_power <= 0.0:
        return ()
    if noise_power < peak_power * 0.25:
        threshold = max(
            noise_power * 10.0 ** (threshold_db / 10.0), peak_power * 1e-6
        )
    else:
        # A finite recording can be almost entirely occupied by one packet,
        # leaving too few noise-only samples for percentile estimation.
        threshold = peak_power * 0.25
    active = smoothed >= threshold
    edges = np.diff(np.concatenate(([False], active, [False])).astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    minimum_samples = max(1, int(round(minimum_symbols * sample_rate_hz / symbol_rate_hz)))
    padding = window * 2
    return tuple(
        (max(0, int(start) - padding), min(iq.size, int(stop) + padding))
        for start, stop in zip(starts, stops)
        if stop - start >= minimum_samples
    )


def _symbol_values(frequency_hz: np.ndarray, phase: int, samples_per_symbol: int) -> np.ndarray:
    available = (frequency_hz.size - int(phase)) // int(samples_per_symbol)
    if available <= 0:
        return np.empty(0, dtype=np.float64)
    values = frequency_hz[
        int(phase) : int(phase) + available * int(samples_per_symbol)
    ].reshape(available, int(samples_per_symbol))
    edge = max(0, int(samples_per_symbol) // 8)
    if edge * 2 < int(samples_per_symbol):
        values = values[:, edge : int(samples_per_symbol) - edge]
    return np.mean(values, axis=1)


def _fractional_symbol_values(
    frequency_hz: np.ndarray,
    start_sample: float,
    samples_per_symbol: int,
    symbol_count: int,
) -> np.ndarray:
    """Average symbol-frequency windows at a fractional sample boundary.

    The integer case is identical to :func:`_symbol_values`.  Interpolation is
    only used after the all-point frequency-model fit has estimated the timing
    offset, so the final decisions and timestamps use the fitted symbol clock
    rather than the coarse integer phase selected by pattern correlation.
    """
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    sps = int(samples_per_symbol)
    requested = int(symbol_count)
    if frequency.size == 0 or requested <= 0:
        return np.empty(0, dtype=np.float64)
    edge = max(0, sps // 8)
    offsets = np.arange(edge, sps - edge, dtype=np.float64)
    if offsets.size == 0:
        offsets = np.arange(sps, dtype=np.float64)
    available = int(
        np.floor(
            (frequency.size - 1.0 - float(start_sample) - offsets[-1]) / sps
        )
        + 1
    )
    count = min(requested, max(0, available))
    if count <= 0:
        return np.empty(0, dtype=np.float64)
    positions = (
        float(start_sample)
        + np.arange(count, dtype=np.float64)[:, None] * sps
        + offsets[None, :]
    )
    interpolated = np.interp(
        positions.ravel(),
        np.arange(frequency.size, dtype=np.float64),
        frequency,
    ).reshape(count, offsets.size)
    return np.mean(interpolated, axis=1)


def _normalized_sliding_correlation(
    values: np.ndarray, expected_levels: np.ndarray
) -> np.ndarray:
    length = expected_levels.size
    if values.size < length:
        return np.empty(0, dtype=np.float64)
    centered_pattern = expected_levels - np.mean(expected_levels)
    pattern_energy = float(np.sum(centered_pattern**2))
    correlation = np.correlate(values, centered_pattern, mode="valid")
    ones = np.ones(length, dtype=np.float64)
    window_sum = np.convolve(values, ones, mode="valid")
    window_square_sum = np.convolve(values**2, ones, mode="valid")
    window_energy = np.maximum(
        window_square_sum - window_sum**2 / float(length),
        np.finfo(np.float64).tiny,
    )
    return correlation / np.sqrt(window_energy * pattern_energy)


def _local_peak_indices(scores: np.ndarray, threshold: float) -> np.ndarray:
    magnitude = np.abs(np.asarray(scores, dtype=np.float64))
    selected: list[int] = []
    for index in np.flatnonzero(magnitude >= float(threshold)):
        left = magnitude[index - 1] if index > 0 else -np.inf
        right = magnitude[index + 1] if index + 1 < magnitude.size else -np.inf
        if magnitude[index] >= left and magnitude[index] > right:
            selected.append(int(index))
    return np.asarray(selected, dtype=np.int64)


def _expected_fsk_symbol_levels(
    bits: np.ndarray, samples_per_symbol: int, gaussian_bt: float | None
) -> np.ndarray:
    shaped = fsk_reference_frequency_levels(
        bits,
        samples_per_symbol=int(samples_per_symbol),
        transmit_gaussian_bt=gaussian_bt,
        measurement_gaussian_bt=None,
    )
    shaped = uniform_filter1d(
        shaped,
        size=max(1, int(samples_per_symbol) // 2),
        mode="nearest",
    )
    return _symbol_values(shaped, 0, int(samples_per_symbol))


def _expected_fsk_sample_levels(
    bits: np.ndarray, samples_per_symbol: int, gaussian_bt: float | None
) -> np.ndarray:
    """Reconstruct the reference instantaneous-frequency waveform."""
    return fsk_reference_frequency_levels(
        bits,
        samples_per_symbol=int(samples_per_symbol),
        transmit_gaussian_bt=gaussian_bt,
        measurement_gaussian_bt=gaussian_bt,
    )


@dataclass(frozen=True)
class _FrequencyModelFit:
    signed_deviation_hz: float
    cfo_hz: float
    drift_hz_per_symbol: float
    timing_offset_samples: float
    residual_rms_hz: float
    no_drift_residual_rms_hz: float
    drift_accepted: bool
    candidate_drift_hz_per_symbol: float
    drift_residual_rms_hz: float
    drift_excursion_hz: float
    drift_bic_improvement: float
    drift_rejection_reason: str
    timing_confidence: float
    estimation_sample_count: int


def _timing_fit_is_credible(
    fit: _FrequencyModelFit, samples_per_symbol: int
) -> bool:
    deviation = max(abs(fit.signed_deviation_hz), np.finfo(float).tiny)
    return bool(
        abs(fit.timing_offset_samples) <= 0.5 * int(samples_per_symbol)
        and fit.timing_confidence >= 0.01
        and fit.residual_rms_hz <= 0.75 * deviation
    )


def _fit_frequency_distortion_model(
    measured_frequency_hz: np.ndarray,
    bits: np.ndarray,
    *,
    samples_per_symbol: int,
    pattern_symbols: int,
    gaussian_bt: float | None,
    anchored_cfo_hz: float | None = None,
) -> _FrequencyModelFit:
    """Jointly fit deviation, CFO, linear drift, and fractional timing.

    This is the variable-projection form of the R&S FSK frequency model.  For
    each timing offset, deviation scale, CFO, and drift are linear least-squares
    parameters.  A bounded scalar search then selects the timing offset with
    minimum residual energy.
    """
    measured = np.asarray(measured_frequency_hz, dtype=np.float64)
    reference = _expected_fsk_sample_levels(
        bits, int(samples_per_symbol), gaussian_bt
    )
    count = min(measured.size, reference.size)
    if count < max(16, 4 * int(samples_per_symbol)):
        raise ValueError("not enough FSK samples for frequency-model estimation")
    measured = measured[:count]
    reference = reference[:count]
    indices = np.arange(count, dtype=np.float64)
    relative_symbols = (
        (indices + 0.5) / float(samples_per_symbol)
        - (float(pattern_symbols) - 1.0) / 2.0
    )
    # Exclude filter/run-in edge samples when enough packet data is available.
    guard = min(2 * int(samples_per_symbol), max(0, (count - 32) // 4))
    fit_slice = slice(guard, count - guard if guard else count)
    fit_measured = measured[fit_slice]
    fit_relative = relative_symbols[fit_slice]

    def solve(
        timing_offset_samples: float, *, include_drift: bool
    ) -> tuple[float, np.ndarray]:
        shifted_reference = np.interp(
            indices - float(timing_offset_samples),
            indices,
            reference,
            left=reference[0],
            right=reference[-1],
        )[fit_slice]
        if anchored_cfo_hz is None:
            columns = [shifted_reference, np.ones(fit_measured.size)]
            if include_drift:
                columns.append(fit_relative)
            design = np.column_stack(columns)
            parameters = np.linalg.lstsq(design, fit_measured, rcond=None)[0]
            modeled = design @ parameters
            if not include_drift:
                parameters = np.asarray(
                    [parameters[0], parameters[1], 0.0], dtype=np.float64
                )
        else:
            columns = [shifted_reference]
            if include_drift:
                columns.append(fit_relative)
            design = np.column_stack(columns)
            fitted = np.linalg.lstsq(
                design,
                fit_measured - float(anchored_cfo_hz),
                rcond=None,
            )[0]
            parameters = np.asarray(
                [
                    fitted[0],
                    float(anchored_cfo_hz),
                    fitted[1] if include_drift else 0.0,
                ],
                dtype=np.float64,
            )
            modeled = design @ fitted + float(anchored_cfo_hz)
        residual = fit_measured - modeled
        return float(np.mean(residual**2)), parameters

    half_symbol = 0.5 * float(samples_per_symbol)

    def optimize(include_drift: bool) -> tuple[float, float, np.ndarray]:
        coarse_offsets = np.linspace(
            -half_symbol, half_symbol, 2 * int(samples_per_symbol) + 1
        )
        coarse = [
            (solve(offset, include_drift=include_drift)[0], float(offset))
            for offset in coarse_offsets
        ]
        _, best_offset = min(coarse, key=lambda item: item[0])
        lower = max(-half_symbol, best_offset - 1.0)
        upper = min(half_symbol, best_offset + 1.0)
        optimized = minimize_scalar(
            lambda offset: solve(
                float(offset), include_drift=include_drift
            )[0],
            bounds=(lower, upper),
            method="bounded",
            options={"xatol": 0.01},
        )
        timing = float(optimized.x) if optimized.success else best_offset
        residual, parameters = solve(timing, include_drift=include_drift)
        return timing, residual, parameters

    drift_timing, drift_residual, drift_parameters = optimize(True)
    no_drift_timing, no_drift_residual, no_drift_parameters = optimize(False)
    fit_count = int(fit_measured.size)
    drift_bic = fit_count * np.log(max(drift_residual, np.finfo(float).tiny))
    drift_bic += 3.0 * np.log(max(2, fit_count))
    no_drift_bic = fit_count * np.log(
        max(no_drift_residual, np.finfo(float).tiny)
    )
    no_drift_bic += 2.0 * np.log(max(2, fit_count))
    drift_excursion_hz = abs(float(drift_parameters[2])) * float(
        np.ptp(fit_relative)
    )
    excursion_threshold_hz = 0.5 * float(
        np.sqrt(max(no_drift_residual, 0.0))
    )
    drift_bic_improvement = float(no_drift_bic - drift_bic)
    drift_accepted = bool(
        drift_bic_improvement > 0.0
        and drift_excursion_hz > excursion_threshold_hz
    )
    if drift_accepted:
        drift_rejection_reason = "Accepted"
    elif drift_bic_improvement <= 0.0:
        drift_rejection_reason = "BIC did not improve"
    else:
        drift_rejection_reason = "Excursion below residual threshold"
    if drift_accepted:
        timing_offset = drift_timing
        residual_power = drift_residual
        parameters = drift_parameters
    else:
        timing_offset = no_drift_timing
        residual_power = no_drift_residual
        parameters = no_drift_parameters

    timing_probe = 0.25
    neighboring_costs = [
        solve(
            float(np.clip(timing_offset + direction * timing_probe, -half_symbol, half_symbol)),
            include_drift=drift_accepted,
        )[0]
        for direction in (-1.0, 1.0)
    ]
    timing_confidence = max(
        0.0,
        min(neighboring_costs) / max(residual_power, np.finfo(float).tiny) - 1.0,
    )
    signed_deviation_hz, cfo_hz, drift_hz_per_symbol = map(float, parameters)
    return _FrequencyModelFit(
        signed_deviation_hz=signed_deviation_hz,
        cfo_hz=cfo_hz,
        drift_hz_per_symbol=drift_hz_per_symbol,
        timing_offset_samples=timing_offset,
        residual_rms_hz=float(np.sqrt(residual_power)),
        no_drift_residual_rms_hz=float(np.sqrt(no_drift_residual)),
        drift_accepted=drift_accepted,
        candidate_drift_hz_per_symbol=float(drift_parameters[2]),
        drift_residual_rms_hz=float(np.sqrt(drift_residual)),
        drift_excursion_hz=float(drift_excursion_hz),
        drift_bic_improvement=drift_bic_improvement,
        drift_rejection_reason=drift_rejection_reason,
        timing_confidence=float(timing_confidence),
        estimation_sample_count=fit_count,
    )


def demodulate_gfsk(
    iq: np.ndarray,
    *,
    sample_rate_hz: float,
    access_bits: np.ndarray,
    symbol_rate_hz: float = 1_000_000.0,
    analysis_samples_per_symbol: int = 8,
    minimum_correlation: float = 0.65,
    gaussian_bt: float | None = 0.5,
    maximum_symbols: int | None = None,
    match_selection: str = "First",
    match_index: int = 1,
    required_result_symbols: int | None = None,
    exclude_incomplete_result: bool = False,
    require_zero_pattern_errors: bool = False,
    allow_polarity_inversion: bool = True,
    allow_complemented_pattern_match: bool = False,
) -> GFSKDemodulationResult:
    """Recover binary symbols using a known access pattern for timing/CFO.

    The access pattern performs the same logical role as the sliding access-code
    correlator in a Bluetooth receiver. The returned bit zero is the first bit
    of the matched access pattern, not the beginning of the source recording.
    """
    samples = np.asarray(iq)
    if samples.ndim != 1 or not np.issubdtype(samples.dtype, np.complexfloating):
        raise ValueError("iq must be a one-dimensional complex array")
    if samples.size < 2:
        raise ValueError("IQ recording is too short")
    if not np.isfinite(sample_rate_hz) or float(sample_rate_hz) <= 0.0:
        raise ValueError("sample_rate_hz must be positive")
    if not np.isfinite(symbol_rate_hz) or float(symbol_rate_hz) <= 0.0:
        raise ValueError("symbol_rate_hz must be positive")
    if int(analysis_samples_per_symbol) < 4:
        raise ValueError("analysis_samples_per_symbol must be at least 4")
    if maximum_symbols is not None and int(maximum_symbols) <= 0:
        raise ValueError("maximum_symbols must be positive when provided")
    if match_selection not in {"Strongest", "First", "Last", "Match Index"}:
        raise ValueError(f"unsupported match selection: {match_selection}")
    if int(match_index) < 1:
        raise ValueError("match_index must be one-based and positive")
    if required_result_symbols is not None and int(required_result_symbols) <= 0:
        raise ValueError("required_result_symbols must be positive when provided")
    pattern = _validate_bits(access_bits)
    expected_levels = _expected_fsk_symbol_levels(
        pattern,
        int(analysis_samples_per_symbol),
        gaussian_bt,
    )
    resampled, analysis_rate_hz = _resample_for_symbols(
        samples,
        float(sample_rate_hz),
        float(symbol_rate_hz),
        int(analysis_samples_per_symbol),
    )
    raw_frequency_hz = _instantaneous_frequency(resampled, analysis_rate_hz)
    frequency_hz = raw_frequency_hz
    if gaussian_bt is not None:
        frequency_hz = apply_gaussian_frequency_filter(
            frequency_hz,
            samples_per_symbol=int(analysis_samples_per_symbol),
            bt=float(gaussian_bt),
        )
    # Pattern acquisition keeps the robust half-symbol integrate-and-dump
    # correlator.  R&S-style symmetric measurement filtering is used by the
    # subsequent all-point fine estimator and does not redefine coarse search.
    detection_frequency_hz = uniform_filter1d(
        raw_frequency_hz,
        size=max(1, int(analysis_samples_per_symbol) // 2),
        mode="nearest",
    )

    candidates: list[tuple[float, float, int, int, np.ndarray, float]] = []
    observed_correlation = 0.0
    centered_expected = expected_levels - np.mean(expected_levels)
    expected_energy = float(np.sum(centered_expected**2))
    for phase in range(int(analysis_samples_per_symbol)):
        values = _symbol_values(
            detection_frequency_hz,
            phase,
            int(analysis_samples_per_symbol),
        )
        scores = _normalized_sliding_correlation(values, expected_levels)
        if scores.size == 0:
            continue
        observed_correlation = max(
            observed_correlation, float(np.max(np.abs(scores)))
        )
        for index in _local_peak_indices(scores, minimum_correlation):
            matched_values = values[index : index + pattern.size]
            # Normalized correlation deliberately ignores amplitude. With an
            # alternating pattern, use fitted tone separation as the secondary
            # timing metric among detections of the same physical packet.
            eye_opening_hz = abs(
                float(
                    np.dot(
                        matched_values - np.mean(matched_values),
                        centered_expected,
                    )
                    / max(expected_energy, np.finfo(np.float64).tiny)
                )
            )
            candidates.append(
                (
                    float(scores[index]),
                    eye_opening_hz,
                    phase,
                    int(index),
                    values,
                    float(phase + int(index) * int(analysis_samples_per_symbol)),
                )
            )
    if not candidates:
        raise ValueError(
            "access pattern was not found "
            f"(correlation={observed_correlation:.3f})"
        )
    groups: list[list[tuple[float, float, int, int, np.ndarray, float]]] = []
    for candidate in sorted(candidates, key=lambda item: item[5]):
        group = next(
            (
                item
                for item in groups
                if abs(item[0][5] - candidate[5])
                < 0.99 * int(analysis_samples_per_symbol)
            ),
            None,
        )
        if group is None:
            groups.append([candidate])
        else:
            group.append(candidate)

    physical_candidates = []
    for group in groups:
        strongest_correlation = max(abs(candidate[0]) for candidate in group)
        strongest_candidate = max(
            group,
            key=lambda candidate: (
                abs(candidate[0]), candidate[1], -candidate[2]
            ),
        )
        timing_candidates = [
            candidate
            for candidate in group
            if abs(candidate[0]) >= strongest_correlation - 0.01
        ]
        widest_eye_candidate = max(
            timing_candidates,
            key=lambda candidate: (
                candidate[1], abs(candidate[0]), -candidate[2]
            ),
        )
        physical_candidates.append(
            widest_eye_candidate
            if widest_eye_candidate[1] >= strongest_candidate[1] * 1.2
            else strongest_candidate
        )

    bursts = _detect_bursts(
        samples,
        sample_rate_hz=float(sample_rate_hz),
        symbol_rate_hz=float(symbol_rate_hz),
    )

    def available_symbols(candidate: tuple) -> int:
        _, _, phase, index, values, _ = candidate
        available = int(values.size - index)
        start_resampled = phase + index * int(analysis_samples_per_symbol)
        start_source = int(
            round(start_resampled * float(sample_rate_hz) / analysis_rate_hz)
        )
        for _, burst_stop in bursts:
            if burst_stop > start_source:
                burst_symbols = int(
                    np.ceil(
                        (burst_stop - start_source)
                        * float(symbol_rate_hz)
                        / float(sample_rate_hz)
                    )
                )
                return min(available, burst_symbols)
        return available

    def pattern_symbol_errors(candidate: tuple) -> int:
        _, _, _, index, values, _ = candidate
        access_frequency = values[index : index + pattern.size]
        if access_frequency.size != pattern.size:
            return int(pattern.size)
        relative = (
            np.arange(pattern.size, dtype=np.float64)
            - (pattern.size - 1.0) / 2.0
        )
        design = np.column_stack((np.ones(pattern.size), expected_levels, relative))
        cfo_hz, signed_deviation, drift_hz_per_symbol = np.linalg.lstsq(
            design, access_frequency, rcond=None
        )[0]
        if (
            signed_deviation <= 0.0
            and not allow_polarity_inversion
            and not allow_complemented_pattern_match
        ):
            return int(pattern.size)
        center_frequency = cfo_hz + drift_hz_per_symbol * relative
        polarity = -1.0 if signed_deviation < 0.0 else 1.0
        decisions = (
            polarity * (access_frequency - center_frequency) >= 0.0
        ).astype(np.uint8)
        return int(np.count_nonzero(decisions != pattern))

    def natural_mapping_polarity(candidate: tuple) -> bool:
        _, _, _, index, values, _ = candidate
        access_frequency = values[index : index + pattern.size]
        if access_frequency.size != pattern.size:
            return False
        relative = (
            np.arange(pattern.size, dtype=np.float64)
            - (pattern.size - 1.0) / 2.0
        )
        design = np.column_stack((np.ones(pattern.size), expected_levels, relative))
        _cfo_hz, signed_deviation, _drift = np.linalg.lstsq(
            design, access_frequency, rcond=None
        )[0]
        return bool(
            signed_deviation > 0.0
            or allow_polarity_inversion
            or allow_complemented_pattern_match
        )

    detected_match_count = len(physical_candidates)
    mapping_candidates = [
        candidate
        for candidate in physical_candidates
        if natural_mapping_polarity(candidate)
    ]
    eligible_candidates = [
        candidate
        for candidate in mapping_candidates
        if (
            (
                not exclude_incomplete_result
                or required_result_symbols is None
                or available_symbols(candidate) >= int(required_result_symbols)
            )
            and (
                not require_zero_pattern_errors
                or pattern_symbol_errors(candidate) == 0
            )
        )
    ]
    if not eligible_candidates:
        if not mapping_candidates:
            raise ValueError(
                "no pattern match has Natural mapping frequency polarity"
            )
        if require_zero_pattern_errors:
            raise ValueError(
                "no symbol-correct pattern match satisfies the search requirements"
            )
        raise ValueError("no pattern match satisfies the result-range requirements")
    ordered = sorted(eligible_candidates, key=lambda item: item[5])
    if match_selection == "First":
        best = ordered[0]
    elif match_selection == "Last":
        best = ordered[-1]
    elif match_selection == "Match Index":
        if int(match_index) > len(ordered):
            raise ValueError(
                f"Match Index {int(match_index)} is unavailable; "
                f"{len(ordered)} eligible match(es) were found"
            )
        best = ordered[int(match_index) - 1]
    else:
        best = max(ordered, key=lambda item: (abs(item[0]), -item[5]))
    selected_match_index = ordered.index(best) + 1
    eligible_match_count = len(ordered)

    score, _, phase, access_index, all_symbol_frequency, _ = best
    access_frequency = all_symbol_frequency[
        access_index : access_index + pattern.size
    ]
    access_relative_symbols = (
        np.arange(pattern.size, dtype=np.float64) - (pattern.size - 1.0) / 2.0
    )
    design = np.column_stack(
        (np.ones(pattern.size), expected_levels, access_relative_symbols)
    )
    cfo_hz, signed_deviation, drift_hz_per_symbol = np.linalg.lstsq(
        design, access_frequency, rcond=None
    )[0]
    cfo_hz = float(cfo_hz)
    signed_deviation = float(signed_deviation)
    drift_hz_per_symbol = float(drift_hz_per_symbol)
    # CFO is a synchronization measurement anchored to the center of the known
    # pattern.  Keep it separate from the packet-wide, decision-directed model
    # below: tentative payload decisions can improve slicing and the coarse
    # drift estimate, but must not move the reported carrier reference away
    # from the only interval whose transmitted symbols are actually known.
    pattern_cfo_hz = cfo_hz
    initial_center_frequency = cfo_hz + drift_hz_per_symbol * access_relative_symbols
    initial_polarity = -1.0 if signed_deviation < 0.0 else 1.0
    initial_access_bits = (
        initial_polarity * (access_frequency - initial_center_frequency) >= 0.0
    ).astype(np.uint8)
    access_errors = int(np.count_nonzero(initial_access_bits != pattern))
    access_start_resampled = phase + access_index * int(analysis_samples_per_symbol)
    access_start_source = int(
        round(access_start_resampled * float(sample_rate_hz) / analysis_rate_hz)
    )
    training_stop = min(
        frequency_hz.size,
        access_start_resampled
        + pattern.size * int(analysis_samples_per_symbol),
    )
    training_measured_frequency = frequency_hz[
        int(access_start_resampled) : int(training_stop)
    ]
    if training_measured_frequency.size >= pattern.size * 4:
        training_fit = _fit_frequency_distortion_model(
            training_measured_frequency,
            pattern,
            samples_per_symbol=int(analysis_samples_per_symbol),
            pattern_symbols=pattern.size,
            gaussian_bt=gaussian_bt,
            anchored_cfo_hz=None,
        )
        if _timing_fit_is_credible(
            training_fit, int(analysis_samples_per_symbol)
        ):
            pattern_cfo_hz = training_fit.cfo_hz
            cfo_hz = pattern_cfo_hz
            signed_deviation = training_fit.signed_deviation_hz
    packet_observed_frequency = all_symbol_frequency[access_index:]
    desired_packet_symbol_count = int(packet_observed_frequency.size)
    for _, burst_stop in bursts:
        if burst_stop > access_start_source:
            burst_symbols = int(
                np.ceil(
                    (burst_stop - access_start_source)
                    * float(symbol_rate_hz)
                    / float(sample_rate_hz)
                )
            )
            packet_observed_frequency = packet_observed_frequency[:burst_symbols]
            desired_packet_symbol_count = burst_symbols
            break
    if maximum_symbols is not None:
        packet_observed_frequency = packet_observed_frequency[
            : int(maximum_symbols)
        ]
        desired_packet_symbol_count = min(
            desired_packet_symbol_count, int(maximum_symbols)
        )
    packet_symbol_count = int(packet_observed_frequency.size)
    packet_relative_symbols = (
        np.arange(packet_symbol_count, dtype=np.float64)
        - (pattern.size - 1.0) / 2.0
    )
    # Start decisions from the known-pattern estimate, then reconstruct the
    # complete reference frequency waveform.  Unlike the former symbol-rate
    # regression, the joint fit uses every capture-oversampling point and
    # estimates deviation, CFO, drift, and fractional timing together.
    initial_polarity = -1.0 if signed_deviation < 0.0 else 1.0
    initial_packet_frequency = initial_polarity * (
        packet_observed_frequency - pattern_cfo_hz
    )
    bits = (initial_packet_frequency >= 0.0).astype(np.uint8)
    training_size = min(pattern.size, bits.size)
    bits[:training_size] = pattern[:training_size]
    packet_start = int(access_start_resampled)
    packet_sample_count = bits.size * int(analysis_samples_per_symbol)
    packet_sample_stop = min(frequency_hz.size, packet_start + packet_sample_count)
    packet_measured_frequency = frequency_hz[packet_start:packet_sample_stop]
    joint_timing_offset = 0.0
    applied_timing_offset = 0.0
    timing_correction_accepted = True
    joint_residual_rms_hz = 0.0
    no_drift_residual_rms_hz = 0.0
    drift_model_accepted = False
    candidate_drift_hz_per_symbol = 0.0
    drift_model_residual_rms_hz = 0.0
    drift_excursion_hz = 0.0
    drift_bic_improvement = 0.0
    drift_rejection_reason = "Not estimated"
    timing_confidence = 0.0
    estimation_sample_count = 0
    refinement_iterations = 3 if bits.size >= pattern.size + 32 else 0
    cfo_hz = pattern_cfo_hz
    drift_hz_per_symbol = 0.0
    for _ in range(refinement_iterations):
        model_fit = _fit_frequency_distortion_model(
            packet_measured_frequency,
            bits,
            samples_per_symbol=int(analysis_samples_per_symbol),
            pattern_symbols=pattern.size,
            gaussian_bt=gaussian_bt,
            anchored_cfo_hz=pattern_cfo_hz,
        )
        signed_deviation = model_fit.signed_deviation_hz
        cfo_hz = model_fit.cfo_hz
        drift_hz_per_symbol = model_fit.drift_hz_per_symbol
        joint_timing_offset = model_fit.timing_offset_samples
        joint_residual_rms_hz = model_fit.residual_rms_hz
        no_drift_residual_rms_hz = model_fit.no_drift_residual_rms_hz
        drift_model_accepted = model_fit.drift_accepted
        candidate_drift_hz_per_symbol = (
            model_fit.candidate_drift_hz_per_symbol
        )
        drift_model_residual_rms_hz = model_fit.drift_residual_rms_hz
        drift_excursion_hz = model_fit.drift_excursion_hz
        drift_bic_improvement = model_fit.drift_bic_improvement
        drift_rejection_reason = model_fit.drift_rejection_reason
        timing_confidence = model_fit.timing_confidence
        estimation_sample_count = model_fit.estimation_sample_count
        timing_correction_accepted = _timing_fit_is_credible(
            model_fit, int(analysis_samples_per_symbol)
        )
        applied_timing_offset = (
            joint_timing_offset if timing_correction_accepted else 0.0
        )
        packet_observed_frequency = _fractional_symbol_values(
            frequency_hz,
            float(access_start_resampled) + applied_timing_offset,
            int(analysis_samples_per_symbol),
            desired_packet_symbol_count,
        )
        if packet_observed_frequency.size != packet_symbol_count:
            packet_symbol_count = int(packet_observed_frequency.size)
            bits = bits[:packet_symbol_count]
            training_size = min(pattern.size, bits.size)
            packet_relative_symbols = (
                np.arange(packet_symbol_count, dtype=np.float64)
                - (pattern.size - 1.0) / 2.0
            )
        estimated_center_frequency = (
            cfo_hz + drift_hz_per_symbol * packet_relative_symbols
        )
        polarity = -1.0 if signed_deviation < 0.0 else 1.0
        packet_frequency = polarity * (
            packet_observed_frequency - estimated_center_frequency
        )
        bits = (packet_frequency >= 0.0).astype(np.uint8)
        bits[:training_size] = pattern[:training_size]
    if (
        signed_deviation <= 0.0
        and not allow_polarity_inversion
        and not allow_complemented_pattern_match
    ):
        raise ValueError(
            "FSK pattern has inverted frequency polarity for Natural mapping"
        )
    iq_inverted = signed_deviation < 0.0
    complemented_pattern_match = bool(
        iq_inverted and allow_complemented_pattern_match
    )
    # Legacy low-level Bluetooth profiles may intentionally rotate frequency
    # polarity so the returned bits equal the supplied access pattern.  The
    # generic VSA's explicit complemented-pattern search is different: it
    # accepts ~pattern as a search hypothesis while preserving Natural mapping
    # (-deviation=0, +deviation=1) in every displayed/exported decision.
    polarity = (
        1.0
        if complemented_pattern_match
        else (-1.0 if iq_inverted else 1.0)
    )
    deviation_hz = abs(signed_deviation)
    estimated_center_frequency = cfo_hz + drift_hz_per_symbol * packet_relative_symbols
    drift_compensated_packet_frequency = (
        polarity * (packet_observed_frequency - estimated_center_frequency)
    )
    packet_frequency = polarity * (packet_observed_frequency - cfo_hz)
    bits = (drift_compensated_packet_frequency >= 0.0).astype(np.uint8)
    # Downstream fields begin after a known training word.  Preserve that word
    # in the recovered stream; access_errors above retains the raw hard-decision
    # quality observed before decision-directed packet tracking.
    training_size = min(pattern.size, bits.size)
    matched_training = (
        1 - pattern
        if complemented_pattern_match
        else pattern
    )
    bits[:training_size] = matched_training[:training_size]
    refined_access_start_resampled = (
        float(access_start_resampled) + applied_timing_offset
    )
    access_start_source = int(
        round(
            refined_access_start_resampled
            * float(sample_rate_hz)
            / analysis_rate_hz
        )
    )
    symbol_time_s = (
        refined_access_start_resampled
        + (np.arange(bits.size, dtype=np.float64) + 0.5)
        * int(analysis_samples_per_symbol)
    ) / analysis_rate_hz
    return GFSKDemodulationResult(
        bits=bits,
        symbol_frequency_hz=packet_frequency,
        drift_compensated_symbol_frequency_hz=(
            drift_compensated_packet_frequency
        ),
        symbol_time_s=symbol_time_s,
        access_start_bit=access_index,
        access_start_sample=access_start_source,
        access_correlation=abs(score),
        access_bit_errors=access_errors,
        carrier_frequency_offset_hz=cfo_hz,
        carrier_frequency_drift_hz_per_s=(
            drift_hz_per_symbol * float(symbol_rate_hz)
        ),
        frequency_deviation_hz=deviation_hz,
        timing_phase_samples=phase,
        analysis_sample_rate_hz=analysis_rate_hz,
        samples_per_symbol=int(analysis_samples_per_symbol),
        iq_inverted=iq_inverted,
        complemented_pattern_match=complemented_pattern_match,
        frequency_model_timing_offset_samples=joint_timing_offset,
        applied_timing_offset_samples=applied_timing_offset,
        timing_correction_accepted=timing_correction_accepted,
        frequency_model_residual_rms_hz=joint_residual_rms_hz,
        frequency_model_no_drift_residual_rms_hz=(
            no_drift_residual_rms_hz
        ),
        drift_model_accepted=drift_model_accepted,
        candidate_drift_hz_per_s=(
            candidate_drift_hz_per_symbol * float(symbol_rate_hz)
        ),
        drift_model_residual_rms_hz=drift_model_residual_rms_hz,
        drift_excursion_hz=drift_excursion_hz,
        drift_bic_improvement=drift_bic_improvement,
        drift_rejection_reason=drift_rejection_reason,
        timing_confidence=timing_confidence,
        estimation_sample_count=estimation_sample_count,
        burst_ranges=bursts,
        selected_match_index=selected_match_index,
        eligible_match_count=eligible_match_count,
        detected_match_count=detected_match_count,
    )
