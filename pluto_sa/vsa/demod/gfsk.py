"""Access-pattern-aided binary GFSK demodulation."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.optimize import minimize_scalar
from scipy.signal import resample_poly


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
    frequency_model_timing_offset_samples: float
    frequency_model_residual_rms_hz: float
    burst_ranges: tuple[tuple[int, int], ...]

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


def _expected_fsk_symbol_levels(
    bits: np.ndarray, samples_per_symbol: int, gaussian_bt: float | None
) -> np.ndarray:
    levels = np.repeat(
        2.0 * np.asarray(bits, dtype=np.float64) - 1.0,
        int(samples_per_symbol),
    )
    shaped = levels
    if gaussian_bt is not None:
        if float(gaussian_bt) <= 0.0:
            raise ValueError("gaussian_bt must be positive when provided")
        sigma_samples = int(samples_per_symbol) / (
            2.0 * np.pi * float(gaussian_bt)
        )
        shaped = gaussian_filter1d(
            levels, sigma=max(0.5, sigma_samples), mode="nearest"
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
    levels = np.repeat(
        2.0 * np.asarray(bits, dtype=np.float64) - 1.0,
        int(samples_per_symbol),
    )
    if gaussian_bt is not None:
        sigma_samples = int(samples_per_symbol) / (
            2.0 * np.pi * float(gaussian_bt)
        )
        levels = gaussian_filter1d(
            levels, sigma=max(0.5, sigma_samples), mode="nearest"
        )
    return uniform_filter1d(
        levels,
        size=max(1, int(samples_per_symbol) // 2),
        mode="nearest",
    )


def _fit_frequency_distortion_model(
    measured_frequency_hz: np.ndarray,
    bits: np.ndarray,
    *,
    samples_per_symbol: int,
    pattern_symbols: int,
    gaussian_bt: float | None,
    anchored_cfo_hz: float | None = None,
) -> tuple[float, float, float, float, float]:
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

    def solve(timing_offset_samples: float) -> tuple[float, np.ndarray]:
        shifted_reference = np.interp(
            indices - float(timing_offset_samples),
            indices,
            reference,
            left=reference[0],
            right=reference[-1],
        )[fit_slice]
        if anchored_cfo_hz is None:
            design = np.column_stack(
                (shifted_reference, np.ones(fit_measured.size), fit_relative)
            )
            parameters = np.linalg.lstsq(design, fit_measured, rcond=None)[0]
            modeled = design @ parameters
        else:
            design = np.column_stack((shifted_reference, fit_relative))
            fitted = np.linalg.lstsq(
                design,
                fit_measured - float(anchored_cfo_hz),
                rcond=None,
            )[0]
            parameters = np.asarray(
                [fitted[0], float(anchored_cfo_hz), fitted[1]],
                dtype=np.float64,
            )
            modeled = design @ fitted + float(anchored_cfo_hz)
        residual = fit_measured - modeled
        return float(np.mean(residual**2)), parameters

    half_symbol = 0.5 * float(samples_per_symbol)
    coarse_offsets = np.linspace(
        -half_symbol, half_symbol, 2 * int(samples_per_symbol) + 1
    )
    coarse = [(solve(offset)[0], float(offset)) for offset in coarse_offsets]
    _, best_offset = min(coarse, key=lambda item: item[0])
    lower = max(-half_symbol, best_offset - 1.0)
    upper = min(half_symbol, best_offset + 1.0)
    optimized = minimize_scalar(
        lambda offset: solve(float(offset))[0],
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": 0.01},
    )
    timing_offset = float(optimized.x) if optimized.success else best_offset
    residual_power, parameters = solve(timing_offset)
    signed_deviation_hz, cfo_hz, drift_hz_per_symbol = map(float, parameters)
    return (
        signed_deviation_hz,
        cfo_hz,
        drift_hz_per_symbol,
        timing_offset,
        float(np.sqrt(residual_power)),
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
    frequency_hz = _instantaneous_frequency(resampled, analysis_rate_hz)
    frequency_hz = uniform_filter1d(
        frequency_hz,
        size=max(1, int(analysis_samples_per_symbol) // 2),
        mode="nearest",
    )

    candidates: list[tuple[float, float, int, int, np.ndarray]] = []
    centered_expected = expected_levels - np.mean(expected_levels)
    expected_energy = float(np.sum(centered_expected**2))
    for phase in range(int(analysis_samples_per_symbol)):
        values = _symbol_values(frequency_hz, phase, int(analysis_samples_per_symbol))
        scores = _normalized_sliding_correlation(values, expected_levels)
        if scores.size == 0:
            continue
        index = int(np.argmax(np.abs(scores)))
        score = float(scores[index])
        matched_values = values[index : index + pattern.size]
        # Normalized correlation deliberately ignores amplitude.  With an
        # alternating FSK pattern this makes samples near a transition look
        # almost as good as samples at the eye centre, especially after a
        # narrow analysis filter.  Measure the fitted tone separation as a
        # second timing metric so equivalent correlations choose the open eye.
        eye_opening_hz = abs(
            float(
                np.dot(
                    matched_values - np.mean(matched_values),
                    centered_expected,
                )
                / max(expected_energy, np.finfo(np.float64).tiny)
            )
        )
        candidates.append((score, eye_opening_hz, phase, index, values))
    if not candidates:
        observed = 0.0
        raise ValueError(
            f"access pattern was not found (correlation={observed:.3f})"
        )
    strongest_correlation = max(abs(candidate[0]) for candidate in candidates)
    if strongest_correlation < float(minimum_correlation):
        observed = strongest_correlation
        raise ValueError(
            f"access pattern was not found (correlation={observed:.3f})"
        )

    # Correlations within one percentage point are indistinguishable for
    # coarse timing.  Select the candidate with the largest fitted frequency
    # separation, then prefer the stronger correlation and earlier phase.
    strongest_candidate = max(
        candidates,
        key=lambda candidate: (abs(candidate[0]), candidate[1], -candidate[2]),
    )
    timing_candidates = [
        candidate
        for candidate in candidates
        if abs(candidate[0]) >= strongest_correlation - 0.01
    ]
    widest_eye_candidate = max(
        timing_candidates,
        key=lambda candidate: (candidate[1], abs(candidate[0]), -candidate[2]),
    )
    # Preserve the maximum-correlation timing unless its eye is materially
    # closed.  Small opening differences are normal pulse-shape asymmetry and
    # must not move an otherwise well-defined packet timestamp.
    best = (
        widest_eye_candidate
        if widest_eye_candidate[1] >= strongest_candidate[1] * 1.2
        else strongest_candidate
    )

    score, _, phase, access_index, all_symbol_frequency = best
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
    packet_observed_frequency = all_symbol_frequency[access_index:]
    bursts = _detect_bursts(
        samples,
        sample_rate_hz=float(sample_rate_hz),
        symbol_rate_hz=float(symbol_rate_hz),
    )
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
            break
    if maximum_symbols is not None:
        packet_observed_frequency = packet_observed_frequency[
            : int(maximum_symbols)
        ]
    packet_relative_symbols = (
        np.arange(packet_observed_frequency.size, dtype=np.float64)
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
    joint_residual_rms_hz = 0.0
    refinement_iterations = 3 if bits.size >= pattern.size + 32 else 0
    cfo_hz = pattern_cfo_hz
    drift_hz_per_symbol = 0.0
    for _ in range(refinement_iterations):
        (
            signed_deviation,
            cfo_hz,
            drift_hz_per_symbol,
            joint_timing_offset,
            joint_residual_rms_hz,
        ) = _fit_frequency_distortion_model(
            packet_measured_frequency,
            bits,
            samples_per_symbol=int(analysis_samples_per_symbol),
            pattern_symbols=pattern.size,
            gaussian_bt=gaussian_bt,
            anchored_cfo_hz=pattern_cfo_hz,
        )
        polarity = -1.0 if signed_deviation < 0.0 else 1.0
        estimated_center_frequency = (
            cfo_hz + drift_hz_per_symbol * packet_relative_symbols
        )
        packet_frequency = polarity * (
            packet_observed_frequency - estimated_center_frequency
        )
        bits = (packet_frequency >= 0.0).astype(np.uint8)
        bits[:training_size] = pattern[:training_size]
    iq_inverted = signed_deviation < 0.0
    polarity = -1.0 if iq_inverted else 1.0
    deviation_hz = abs(signed_deviation)
    estimated_center_frequency = cfo_hz + drift_hz_per_symbol * packet_relative_symbols
    drift_compensated_packet_frequency = polarity * (
        packet_observed_frequency - estimated_center_frequency
    )
    packet_frequency = polarity * (packet_observed_frequency - cfo_hz)
    bits = (drift_compensated_packet_frequency >= 0.0).astype(np.uint8)
    # Downstream fields begin after a known training word.  Preserve that word
    # in the recovered stream; access_errors above retains the raw hard-decision
    # quality observed before decision-directed packet tracking.
    training_size = min(pattern.size, bits.size)
    bits[:training_size] = pattern[:training_size]
    symbol_time_s = (
        access_start_resampled
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
        frequency_model_timing_offset_samples=joint_timing_offset,
        frequency_model_residual_rms_hz=joint_residual_rms_hz,
        burst_ranges=bursts,
    )
