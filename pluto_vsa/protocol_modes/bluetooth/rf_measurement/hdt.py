"""HDT preamble-referenced RMS EVM measurement path."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def _readonly(values: object, dtype: np.dtype | type = np.complex64) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).copy()
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

    @property
    def carrier_error_rad_per_symbol(self) -> float:
        """Return the received carrier error (opposite the correction sign)."""

        return -float(self.phase_step_rad_per_symbol)


@dataclass(frozen=True)
class HDTPayloadEstimate:
    """Appendix-C payload phase/CFO fit with preamble amplitude/timing fixed."""

    phase_rad: float
    phase_step_rad_per_symbol: float

    @property
    def carrier_error_rad_per_symbol(self) -> float:
        return -float(self.phase_step_rad_per_symbol)


@dataclass(frozen=True)
class HDTEVMResult:
    header_rms_percent: float
    payload_rms_percent: float
    reference: HDTReferenceEstimate
    header_measured_symbols: np.ndarray
    header_reference_symbols: np.ndarray
    payload_measured_symbols: np.ndarray
    payload_reference_symbols: np.ndarray
    payload_estimate: HDTPayloadEstimate
    terminating_measured_symbols: np.ndarray
    terminating_reference_symbols: np.ndarray
    header_corrected_waveform: np.ndarray
    payload_corrected_waveform: np.ndarray
    header_symbol_sample_positions: np.ndarray
    payload_symbol_sample_positions: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "header_measured_symbols",
            "header_reference_symbols",
            "payload_measured_symbols",
            "payload_reference_symbols",
            "terminating_measured_symbols",
            "terminating_reference_symbols",
            "header_corrected_waveform",
            "payload_corrected_waveform",
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name)))
        for name in (
            "header_symbol_sample_positions",
            "payload_symbol_sample_positions",
        ):
            object.__setattr__(
                self, name, _readonly(getattr(self, name), np.float64)
            )

    @property
    def header_corrected_symbols(self) -> np.ndarray:
        """Symbols used verbatim by Header EVM and downstream displays."""

        return self.header_measured_symbols

    @property
    def payload_corrected_symbols(self) -> np.ndarray:
        """Symbols used verbatim by Payload EVM and downstream displays."""

        return self.payload_measured_symbols


@dataclass(frozen=True)
class HDTPlotData:
    """HDT display inputs and global sample ranges fixed by analysis."""

    evm: HDTEVMResult
    packet_sample_range: tuple[int, int]
    training_sample_range: tuple[int, int]
    control_header_sample_range: tuple[int, int]
    payload_sample_range: tuple[int, int]
    payload_evm_sample_range: tuple[int, int]

    def __post_init__(self) -> None:
        for name in (
            "packet_sample_range",
            "training_sample_range",
            "control_header_sample_range",
            "payload_sample_range",
            "payload_evm_sample_range",
        ):
            start, stop = getattr(self, name)
            normalized = (int(start), int(stop))
            if normalized[1] <= normalized[0]:
                raise ValueError(f"{name} must be a non-empty sample range")
            object.__setattr__(self, name, normalized)


def _initial_correction(
    observed: np.ndarray,
    reference: np.ndarray,
) -> tuple[float, float, float]:
    """Return an amplitude/phase/phase-step correction seed."""

    axis = np.arange(reference.size, dtype=np.float64)
    phase_error = np.unwrap(np.angle(reference * np.conj(observed)))
    phase_step, phase = np.polyfit(axis, phase_error, 1)
    rotating = observed * np.exp(1j * (phase + phase_step * axis))
    denominator = float(np.sum(np.abs(rotating) ** 2))
    correction = np.vdot(rotating, reference) / max(
        denominator, np.finfo(np.float64).tiny
    )
    amplitude = max(float(np.abs(correction)), np.finfo(np.float64).tiny)
    phase += float(np.angle(correction))
    return amplitude, float(phase), float(phase_step)


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
    amplitude, phase, phase_step = _initial_correction(
        observed, training_reference
    )
    phase_corrected = (
        amplitude * observed * np.exp(1j * (phase + phase_step * axis))
    )
    denominator = float(np.sum(np.abs(training_reference) ** 2))
    error = float(
        np.sum(np.abs(phase_corrected - training_reference) ** 2)
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
    axis = np.arange(reference.size, dtype=np.float64)
    reference_energy = max(
        float(np.sum(np.abs(reference) ** 2)), np.finfo(np.float64).tiny
    )

    def joint_residual(parameters: np.ndarray) -> np.ndarray:
        log_amplitude, current_phase, current_step, current_offset = parameters
        positions = coarse + current_offset + axis * sps
        current = _interpolate_complex(values, positions)
        corrected = np.exp(log_amplitude) * current * np.exp(
            1j * (current_phase + current_step * axis)
        )
        residual = (corrected - reference) / np.sqrt(reference_energy)
        return np.concatenate((residual.real, residual.imag))

    # The scalar timing search supplies a stable seed; this final complex
    # least-squares fit optimizes Appendix C's alpha0, phi0, delta-omega0 and
    # T0 together instead of successively normalizing them.
    joint = least_squares(
        joint_residual,
        np.asarray((np.log(amplitude), phase, phase_step, offset)),
        bounds=(
            np.asarray((-20.0, -np.inf, -np.pi, -1.0)),
            np.asarray((20.0, np.inf, np.pi, 1.0)),
        ),
        x_scale=np.asarray((0.1, 0.1, 0.01, 0.1)),
        max_nfev=400,
        xtol=1e-11,
        ftol=1e-11,
        gtol=1e-11,
    )
    amplitude = float(np.exp(joint.x[0]))
    phase = float(joint.x[1])
    phase_step = float(joint.x[2])
    offset = float(joint.x[3])
    first_center = coarse + offset
    observed = _interpolate_complex(values, first_center + axis * sps)
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
    return reference.amplitude * observed * np.exp(
        1j
        * (
            reference.phase_rad
            + reference.phase_step_rad_per_symbol * axis
        )
    )


def estimate_hdt_payload(
    filtered_iq: np.ndarray,
    preamble: HDTReferenceEstimate,
    *,
    samples_per_symbol: float,
    start_symbol: int,
    payload_reference: np.ndarray,
) -> tuple[HDTPayloadEstimate, np.ndarray]:
    """Fit only payload phi1/delta-omega1 with alpha0/T0 held fixed."""

    values = np.asarray(filtered_iq, dtype=np.complex128)
    ideal = np.asarray(payload_reference, dtype=np.complex128)
    if ideal.size == 0:
        raise ValueError("HDT payload EVM requires at least one reference symbol")
    axis = np.arange(ideal.size, dtype=np.float64)
    positions = preamble.first_symbol_center_sample + (
        float(start_symbol) + axis
    ) * float(samples_per_symbol)
    observed = _interpolate_complex(values, positions)
    scaled = preamble.amplitude * observed
    _unused_amplitude, phase, phase_step = _initial_correction(scaled, ideal)
    reference_energy = max(
        float(np.sum(np.abs(ideal) ** 2)), np.finfo(np.float64).tiny
    )

    def residual(parameters: np.ndarray) -> np.ndarray:
        current_phase, current_step = parameters
        corrected = scaled * np.exp(
            1j * (current_phase + current_step * axis)
        )
        error = (corrected - ideal) / np.sqrt(reference_energy)
        return np.concatenate((error.real, error.imag))

    fitted = least_squares(
        residual,
        np.asarray((phase, phase_step)),
        x_scale=np.asarray((0.1, 0.001)),
        max_nfev=300,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )
    estimate = HDTPayloadEstimate(
        phase_rad=float(fitted.x[0]),
        phase_step_rad_per_symbol=float(fitted.x[1]),
    )
    measured = scaled * np.exp(
        1j
        * (
            estimate.phase_rad
            + estimate.phase_step_rad_per_symbol * axis
        )
    )
    return estimate, measured


def apply_hdt_payload_estimate(
    filtered_iq: np.ndarray,
    preamble: HDTReferenceEstimate,
    payload: HDTPayloadEstimate,
    *,
    samples_per_symbol: float,
    start_symbol: int,
    payload_symbol_offset: int,
    symbol_count: int,
) -> np.ndarray:
    """Apply the fixed alpha0/T0 and fitted phi1/delta-omega1."""

    local_axis = int(payload_symbol_offset) + np.arange(
        int(symbol_count), dtype=np.float64
    )
    positions = preamble.first_symbol_center_sample + (
        int(start_symbol) + local_axis
    ) * float(samples_per_symbol)
    observed = _interpolate_complex(
        np.asarray(filtered_iq, dtype=np.complex128), positions
    )
    return preamble.amplitude * observed * np.exp(
        1j
        * (
            payload.phase_rad
            + payload.phase_step_rad_per_symbol * local_axis
        )
    )


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
    payload_estimate: HDTPayloadEstimate,
    terminating_measured_symbols: np.ndarray,
    terminating_reference_symbols: np.ndarray,
    header_corrected_waveform: np.ndarray,
    payload_corrected_waveform: np.ndarray,
    header_symbol_sample_positions: np.ndarray,
    payload_symbol_sample_positions: np.ndarray,
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
        payload_estimate=payload_estimate,
        terminating_measured_symbols=terminating_measured_symbols,
        terminating_reference_symbols=terminating_reference_symbols,
        header_corrected_waveform=header_corrected_waveform,
        payload_corrected_waveform=payload_corrected_waveform,
        header_symbol_sample_positions=header_symbol_sample_positions,
        payload_symbol_sample_positions=payload_symbol_sample_positions,
    )
