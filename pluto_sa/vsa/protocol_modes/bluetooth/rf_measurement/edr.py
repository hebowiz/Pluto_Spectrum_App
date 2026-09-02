"""Dedicated EDR RF measurement primitives.

This path intentionally does not consume Generic VSA carrier-corrected IQ.
Only the EDR SRRC filter and per-50-symbol timing/frequency parameters are
optimized, keeping decoder-oriented global fitting out of SIG measurements.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from pluto_sa.vsa.mapping import psk_constellation
from pluto_sa.vsa.model import ModulationKind
from pluto_sa.vsa.pattern import prepare_psk_iq


def _readonly(values: object, dtype: np.dtype | type = np.float64) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _interpolate_complex(values: np.ndarray, positions: np.ndarray) -> np.ndarray:
    axis = np.arange(values.size, dtype=np.float64)
    return np.interp(positions, axis, values.real) + 1j * np.interp(
        positions, axis, values.imag
    )


@dataclass(frozen=True)
class EDRDEVMBlockResult:
    block_index: int
    start_symbol: int
    timing_offset_symbols: float
    residual_frequency_error_hz: float
    rms_devm: float
    peak_devm: float
    symbol_devm: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol_devm", _readonly(self.symbol_devm))


@dataclass(frozen=True)
class EDRDEVMTestResult:
    initial_frequency_error_hz: float
    blocks: tuple[EDRDEVMBlockResult, ...]
    rms_worst: float | None
    peak_worst: float | None
    devm_99_percentile: float | None
    total_symbol_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "blocks", tuple(self.blocks))


@dataclass(frozen=True)
class EDRGuardTimeResult:
    header_end_time_s: float
    reference_symbol_start_time_s: float
    guard_time_s: float


@dataclass(frozen=True)
class EDRConformanceResult:
    sync_symbol_errors: int
    trailer_symbol_errors: int
    evaluated_sync_symbols: int
    evaluated_trailer_symbols: int


def measure_edr_devm(
    iq: np.ndarray,
    *,
    sample_rate_hz: float,
    symbol_rate_hz: float,
    first_symbol_center_sample: float,
    decoded_symbols: np.ndarray,
    modulation: ModulationKind,
    symbol_mapping: str,
    initial_frequency_error_hz: float,
    trailer_symbols: int = 2,
    block_symbols: int = 50,
) -> EDRDEVMTestResult:
    """Measure non-overlapping 50-symbol EDR DEVM blocks.

    The final two trailer symbols are removed before block construction.
    Per block, only timing, residual frequency and a constant complex
    reference coefficient are fitted.
    """

    values = np.asarray(iq, dtype=np.complex128)
    labels = np.asarray(decoded_symbols, dtype=np.int16)
    usable = max(0, labels.size - max(0, int(trailer_symbols)))
    labels = labels[:usable]
    block_size = max(1, int(block_symbols))
    complete = (labels.size // block_size) * block_size
    labels = labels[:complete]
    if labels.size == 0:
        return EDRDEVMTestResult(
            float(initial_frequency_error_hz), (), None, None, None, 0
        )
    filtered, filtered_rate = prepare_psk_iq(
        values,
        sample_rate_hz=float(sample_rate_hz),
        symbol_rate_hz=float(symbol_rate_hz),
        tx_filter="Root Raised Cosine",
        filter_parameter=0.4,
        samples_per_symbol=int(round(float(sample_rate_hz) / float(symbol_rate_hz))),
        apply_measurement_filter=True,
    )
    if not np.isclose(filtered_rate, sample_rate_hz):
        raise ValueError("EDR measurement filter changed the sample rate")
    alphabet = psk_constellation(ModulationKind(modulation), symbol_mapping)
    differential = alphabet[labels]
    absolute_reference = np.cumprod(differential)
    sps = float(sample_rate_hz) / float(symbol_rate_hz)
    blocks: list[EDRDEVMBlockResult] = []
    all_devm: list[np.ndarray] = []

    for block_index, start in enumerate(range(0, labels.size, block_size)):
        reference = absolute_reference[start : start + block_size]
        relative = np.arange(block_size, dtype=np.float64)
        nominal_centers = (
            float(first_symbol_center_sample) + (start + relative) * sps
        )

        def residual(parameters: np.ndarray) -> np.ndarray:
            timing_samples, phase_step = map(float, parameters)
            observed = _interpolate_complex(filtered, nominal_centers + timing_samples)
            rotating_reference = reference * np.exp(1j * phase_step * relative)
            denominator = max(
                float(np.sum(np.abs(rotating_reference) ** 2)),
                np.finfo(np.float64).tiny,
            )
            gain = np.vdot(rotating_reference, observed) / denominator
            error = observed - gain * rotating_reference
            scale = max(abs(gain), np.finfo(np.float64).tiny)
            error /= scale
            return np.concatenate((error.real, error.imag))

        fitted = least_squares(
            residual,
            np.zeros(2, dtype=np.float64),
            bounds=(np.asarray([-0.5 * sps, -0.35]), np.asarray([0.5 * sps, 0.35])),
            x_scale=np.asarray([max(0.25, 0.25 * sps), 0.02]),
            max_nfev=80,
        )
        timing_samples, phase_step = map(float, fitted.x)
        observed = _interpolate_complex(filtered, nominal_centers + timing_samples)
        rotating_reference = reference * np.exp(1j * phase_step * relative)
        gain = np.vdot(rotating_reference, observed) / max(
            float(np.sum(np.abs(rotating_reference) ** 2)),
            np.finfo(np.float64).tiny,
        )
        corrected = observed / max(abs(gain), np.finfo(np.float64).tiny)
        predicted = rotating_reference * np.exp(1j * np.angle(gain))
        symbol_devm = np.abs(corrected - predicted)
        rms = float(np.sqrt(np.mean(symbol_devm**2)))
        peak = float(np.max(symbol_devm))
        blocks.append(
            EDRDEVMBlockResult(
                block_index=block_index,
                start_symbol=start,
                timing_offset_symbols=timing_samples / sps,
                residual_frequency_error_hz=(
                    phase_step * float(symbol_rate_hz) / (2.0 * np.pi)
                ),
                rms_devm=rms,
                peak_devm=peak,
                symbol_devm=symbol_devm,
            )
        )
        all_devm.append(symbol_devm)
    combined = np.concatenate(all_devm)
    return EDRDEVMTestResult(
        initial_frequency_error_hz=float(initial_frequency_error_hz),
        blocks=tuple(blocks),
        rms_worst=max(block.rms_devm for block in blocks),
        peak_worst=max(block.peak_devm for block in blocks),
        devm_99_percentile=float(np.percentile(combined, 99.0)),
        total_symbol_count=int(combined.size),
    )


def measure_edr_guard_time(
    *,
    header_end_sample: float,
    reference_symbol_start_sample: float,
    sample_rate_hz: float,
) -> EDRGuardTimeResult:
    return EDRGuardTimeResult(
        header_end_time_s=float(header_end_sample) / float(sample_rate_hz),
        reference_symbol_start_time_s=(
            float(reference_symbol_start_sample) / float(sample_rate_hz)
        ),
        guard_time_s=(
            float(reference_symbol_start_sample) - float(header_end_sample)
        )
        / float(sample_rate_hz),
    )


def measure_edr_conformance(
    decoded_symbols: np.ndarray,
    expected_sync_symbols: np.ndarray,
    *,
    trailer_symbols: int = 2,
) -> EDRConformanceResult:
    decoded = np.asarray(decoded_symbols, dtype=np.int16)
    sync = np.asarray(expected_sync_symbols, dtype=np.int16)
    sync_count = min(decoded.size, sync.size)
    trailer_count = min(max(0, int(trailer_symbols)), max(0, decoded.size - sync_count))
    trailer = decoded[-trailer_count:] if trailer_count else np.empty(0, dtype=np.int16)
    return EDRConformanceResult(
        sync_symbol_errors=int(np.count_nonzero(decoded[:sync_count] != sync[:sync_count])),
        trailer_symbol_errors=int(np.count_nonzero(trailer)),
        evaluated_sync_symbols=sync_count,
        evaluated_trailer_symbols=trailer_count,
    )
