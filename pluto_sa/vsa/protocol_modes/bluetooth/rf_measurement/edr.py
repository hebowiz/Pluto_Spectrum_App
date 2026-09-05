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
    stop_symbol: int
    timing_offset_symbols: float
    residual_frequency_error_hz: float
    rms_devm: float
    peak_devm: float
    symbol_devm: np.ndarray
    physical_symbol_center_samples: np.ndarray
    corrected_received_symbols: np.ndarray
    reference_symbols: np.ndarray
    error_vectors: np.ndarray

    def __post_init__(self) -> None:
        for name, dtype in (
            ("symbol_devm", np.float64),
            ("physical_symbol_center_samples", np.float64),
            ("corrected_received_symbols", np.complex128),
            ("reference_symbols", np.complex128),
            ("error_vectors", np.complex128),
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name), dtype))


@dataclass(frozen=True)
class EDRDEVMTestResult:
    initial_frequency_error_hz: float
    blocks: tuple[EDRDEVMBlockResult, ...]
    rms_worst: float | None
    peak_worst: float | None
    devm_99_percentile: float | None
    total_symbol_count: int
    reference_symbol_center_sample: float | None = None
    sync_first_symbol_center_sample: float | None = None

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
    reference_symbols: np.ndarray | None = None,
    modulation: ModulationKind,
    symbol_mapping: str,
    initial_frequency_error_hz: float,
    trailer_symbols: int = 2,
    block_symbols: int = 50,
) -> EDRDEVMTestResult:
    """Measure non-overlapping 50-symbol Bluetooth EDR DEVM blocks.

    ``first_symbol_center_sample`` is the center of the first EDR Sync
    differential symbol.  The physical Reference Symbol immediately before
    it is included in the first 51-symbol measurement window.  Every block
    minimizes the differential error energy using only sampling phase and
    residual frequency error; no complex gain or absolute phase fit is made.
    """

    values = np.asarray(iq, dtype=np.complex128)
    decoded = np.asarray(decoded_symbols, dtype=np.int16)
    labels = np.asarray(
        decoded if reference_symbols is None else reference_symbols,
        dtype=np.int16,
    )
    if labels.size != decoded.size:
        raise ValueError("EDR DEVM reference must match the decoded symbol count")
    usable = max(0, labels.size - max(0, int(trailer_symbols)))
    labels = labels[:usable]
    block_size = max(1, int(block_symbols))
    complete = (labels.size // block_size) * block_size
    labels = labels[:complete]
    if labels.size == 0:
        return EDRDEVMTestResult(
            float(initial_frequency_error_hz), (), None, None, None, 0
        )
    # Appendix-C DEVM starts after removing omega_i obtained from the BR
    # header.  Leaving that rotation in the samples forced the per-block
    # omega_0 optimizer to acquire the full carrier error from a zero start;
    # on real packets it frequently settled in a data-dependent local minimum
    # and normalized by an almost-zero complex gain, producing 400-900% DEVM.
    sample_axis = np.arange(values.size, dtype=np.float64)
    omega_i_corrected = values * np.exp(
        -2j
        * np.pi
        * float(initial_frequency_error_hz)
        * sample_axis
        / float(sample_rate_hz)
    )
    filtered, filtered_rate = prepare_psk_iq(
        omega_i_corrected,
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
    # S[0] is the EDR Reference Symbol S0.  S[k + 1] is the physical
    # transmitted symbol after applying differential symbol k.
    absolute_reference = np.concatenate(
        (np.ones(1, dtype=np.complex128), np.cumprod(differential))
    )
    sps = float(sample_rate_hz) / float(symbol_rate_hz)
    reference_symbol_center = float(first_symbol_center_sample) - sps
    blocks: list[EDRDEVMBlockResult] = []
    all_devm: list[np.ndarray] = []

    for block_index, start in enumerate(range(0, labels.size, block_size)):
        stop = start + block_size
        # 51 physical symbols produce the 50 differential error vectors.
        reference = absolute_reference[start : stop + 1]
        physical_index = start + np.arange(block_size + 1, dtype=np.float64)
        nominal_centers = (
            reference_symbol_center + physical_index * sps
        )

        def residual(parameters: np.ndarray) -> np.ndarray:
            timing_samples, residual_frequency_hz = map(float, parameters)
            centers = nominal_centers + timing_samples
            observed = _interpolate_complex(filtered, centers)
            observed *= np.exp(
                -2j
                * np.pi
                * residual_frequency_hz
                * (centers - centers[0])
                / float(sample_rate_hz)
            )
            q_symbols = observed * np.conj(reference)
            errors = np.diff(q_symbols)
            denominator = max(
                float(np.sum(np.abs(q_symbols[1:]) ** 2)),
                np.finfo(np.float64).tiny,
            )
            normalized_errors = errors / np.sqrt(denominator)
            return np.concatenate(
                (normalized_errors.real, normalized_errors.imag)
            )

        # A phase-slope estimate from Q gives a stable starting point without
        # introducing an additional fitted parameter into the measurement.
        initial_centers = nominal_centers
        initial_observed = _interpolate_complex(filtered, initial_centers)
        initial_q = initial_observed * np.conj(reference)
        initial_phase_step = float(
            np.median(np.angle(initial_q[1:] * np.conj(initial_q[:-1])))
        )
        initial_omega0_hz = float(
            np.clip(
                initial_phase_step * float(symbol_rate_hz) / (2.0 * np.pi),
                -100_000.0,
                100_000.0,
            )
        )

        fitted = least_squares(
            residual,
            np.asarray([0.0, initial_omega0_hz], dtype=np.float64),
            bounds=(
                np.asarray([-0.5 * sps, -100_000.0]),
                np.asarray([0.5 * sps, 100_000.0]),
            ),
            x_scale=np.asarray([max(0.25, 0.25 * sps), 10_000.0]),
            max_nfev=120,
        )
        timing_samples, residual_frequency_hz = map(float, fitted.x)
        centers = nominal_centers + timing_samples
        corrected = _interpolate_complex(filtered, centers)
        corrected *= np.exp(
            -2j
            * np.pi
            * residual_frequency_hz
            * (centers - centers[0])
            / float(sample_rate_hz)
        )
        q_symbols = corrected * np.conj(reference)
        errors = np.diff(q_symbols)
        rms_amplitude = np.sqrt(
            max(
                float(np.mean(np.abs(q_symbols[1:]) ** 2)),
                np.finfo(np.float64).tiny,
            )
        )
        symbol_devm = np.abs(errors) / rms_amplitude
        rms = float(
            np.sqrt(
                np.sum(np.abs(errors) ** 2)
                / max(
                    float(np.sum(np.abs(q_symbols[1:]) ** 2)),
                    np.finfo(np.float64).tiny,
                )
            )
        )
        peak = float(np.max(symbol_devm))
        blocks.append(
            EDRDEVMBlockResult(
                block_index=block_index,
                start_symbol=start,
                stop_symbol=stop,
                timing_offset_symbols=timing_samples / sps,
                residual_frequency_error_hz=residual_frequency_hz,
                rms_devm=rms,
                peak_devm=peak,
                symbol_devm=symbol_devm,
                physical_symbol_center_samples=centers,
                corrected_received_symbols=corrected,
                reference_symbols=reference,
                error_vectors=errors,
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
        reference_symbol_center_sample=reference_symbol_center,
        sync_first_symbol_center_sample=float(first_symbol_center_sample),
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
