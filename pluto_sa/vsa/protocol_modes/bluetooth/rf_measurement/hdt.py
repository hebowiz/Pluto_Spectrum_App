"""HDT preamble-referenced RMS EVM measurement path."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar


def _readonly(values: object) -> np.ndarray:
    result = np.asarray(values, dtype=np.complex64).copy()
    result.setflags(write=False)
    return result


def _interpolate_complex(values: np.ndarray, positions: np.ndarray) -> np.ndarray:
    samples = np.arange(values.size, dtype=np.float64)
    return np.interp(positions, samples, values.real) + 1j * np.interp(
        positions, samples, values.imag
    )


@dataclass(frozen=True)
class HDTReferenceEstimate:
    first_symbol_center_sample: float
    amplitude: float
    phase_rad: float
    phase_step_rad_per_symbol: float
    timing_offset_samples: float
    training_correlation: float


@dataclass(frozen=True)
class HDTEVMResult:
    header_rms_percent: float
    payload_rms_percent: float
    reference: HDTReferenceEstimate
    header_measured_symbols: np.ndarray
    header_reference_symbols: np.ndarray
    payload_measured_symbols: np.ndarray
    payload_reference_symbols: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "header_measured_symbols",
            "header_reference_symbols",
            "payload_measured_symbols",
            "payload_reference_symbols",
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name)))


def _fit_reference_at_center(
    filtered_iq: np.ndarray,
    first_center: float,
    samples_per_symbol: float,
    training_reference: np.ndarray,
) -> tuple[float, float, float, float, np.ndarray]:
    axis = np.arange(training_reference.size, dtype=np.float64)
    observed = _interpolate_complex(
        filtered_iq, float(first_center) + axis * float(samples_per_symbol)
    )
    phase_error = np.unwrap(np.angle(observed * np.conj(training_reference)))
    phase_step, phase = np.polyfit(axis, phase_error, 1)
    phase_corrected = observed * np.exp(-1j * (phase + phase_step * axis))
    denominator = float(np.sum(np.abs(training_reference) ** 2))
    amplitude = float(
        np.real(np.vdot(training_reference, phase_corrected))
        / max(denominator, np.finfo(np.float64).tiny)
    )
    amplitude = max(amplitude, np.finfo(np.float64).tiny)
    normalized = phase_corrected / amplitude
    error = float(
        np.sum(np.abs(normalized - training_reference) ** 2)
        / max(denominator, np.finfo(np.float64).tiny)
    )
    correlation = float(
        np.abs(np.vdot(training_reference, observed))
        / max(
            np.linalg.norm(training_reference) * np.linalg.norm(observed),
            np.finfo(np.float64).tiny,
        )
    )
    return error, amplitude, float(phase), float(phase_step), observed


def estimate_hdt_reference(
    filtered_iq: np.ndarray,
    *,
    coarse_first_symbol_center_sample: float,
    samples_per_symbol: float,
    training_reference: np.ndarray,
) -> HDTReferenceEstimate:
    """Estimate gain, phase, CFO and fractional timing from Training only."""

    values = np.asarray(filtered_iq, dtype=np.complex128)
    reference = np.asarray(training_reference, dtype=np.complex128)
    coarse = float(coarse_first_symbol_center_sample)
    sps = float(samples_per_symbol)

    def objective(offset: float) -> float:
        first = coarse + float(offset)
        last = first + (reference.size - 1) * sps
        if first < 0.0 or last > values.size - 1:
            return float("inf")
        return _fit_reference_at_center(values, first, sps, reference)[0]

    refined = minimize_scalar(
        objective,
        bounds=(-1.0, 1.0),
        method="bounded",
        options={"xatol": 1e-4},
    )
    offset = float(refined.x) if np.isfinite(refined.fun) else 0.0
    first_center = coarse + offset
    _error, amplitude, phase, phase_step, observed = _fit_reference_at_center(
        values, first_center, sps, reference
    )
    correlation = float(
        np.abs(np.vdot(reference, observed))
        / max(
            np.linalg.norm(reference) * np.linalg.norm(observed),
            np.finfo(np.float64).tiny,
        )
    )
    return HDTReferenceEstimate(
        first_symbol_center_sample=first_center,
        amplitude=amplitude,
        phase_rad=phase,
        phase_step_rad_per_symbol=phase_step,
        timing_offset_samples=offset,
        training_correlation=correlation,
    )


def apply_hdt_reference(
    filtered_iq: np.ndarray,
    reference: HDTReferenceEstimate,
    *,
    samples_per_symbol: float,
    start_symbol: int,
    symbol_count: int,
) -> np.ndarray:
    axis = int(start_symbol) + np.arange(int(symbol_count), dtype=np.float64)
    positions = (
        reference.first_symbol_center_sample + axis * float(samples_per_symbol)
    )
    observed = _interpolate_complex(
        np.asarray(filtered_iq, dtype=np.complex128), positions
    )
    return observed * np.exp(
        -1j
        * (
            reference.phase_rad
            + reference.phase_step_rad_per_symbol * axis
        )
    ) / max(reference.amplitude, np.finfo(np.float64).tiny)


def rms_evm_percent(measured: np.ndarray, ideal: np.ndarray) -> float:
    actual = np.asarray(measured, dtype=np.complex128)
    reference = np.asarray(ideal, dtype=np.complex128)
    count = min(actual.size, reference.size)
    if count == 0:
        raise ValueError("RMS EVM requires at least one symbol")
    denominator = float(np.sum(np.abs(reference[:count]) ** 2))
    return 100.0 * float(
        np.sqrt(
            np.sum(np.abs(actual[:count] - reference[:count]) ** 2)
            / max(denominator, np.finfo(np.float64).tiny)
        )
    )


def build_hdt_evm_result(
    *,
    reference: HDTReferenceEstimate,
    header_measured_symbols: np.ndarray,
    header_reference_symbols: np.ndarray,
    payload_measured_symbols: np.ndarray,
    payload_reference_symbols: np.ndarray,
) -> HDTEVMResult:
    return HDTEVMResult(
        header_rms_percent=rms_evm_percent(
            header_measured_symbols, header_reference_symbols
        ),
        payload_rms_percent=rms_evm_percent(
            payload_measured_symbols, payload_reference_symbols
        ),
        reference=reference,
        header_measured_symbols=header_measured_symbols,
        header_reference_symbols=header_reference_symbols,
        payload_measured_symbols=payload_measured_symbols,
        payload_reference_symbols=payload_reference_symbols,
    )
