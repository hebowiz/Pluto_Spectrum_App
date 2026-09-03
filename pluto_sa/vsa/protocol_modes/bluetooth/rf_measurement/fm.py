"""Uncompensated FM measurements for Bluetooth BR and uncoded LE PHYs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .filter import BluetoothRFMeasurementFilterProfile, apply_rf_test_channel_filter


def _readonly(values: object, dtype: np.dtype | type) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class BluetoothFMMeasurementTrace:
    time_s: np.ndarray
    frequency_hz: np.ndarray
    p0_sample: float
    sample_rate_hz: float
    symbol_rate_hz: float
    samples_per_symbol: float
    filter_profile: BluetoothRFMeasurementFilterProfile

    def __post_init__(self) -> None:
        time_s = _readonly(self.time_s, np.float64)
        frequency_hz = _readonly(self.frequency_hz, np.float64)
        if time_s.shape != frequency_hz.shape or time_s.ndim != 1:
            raise ValueError("FM trace time and frequency arrays must be one-dimensional peers")
        object.__setattr__(self, "time_s", time_s)
        object.__setattr__(self, "frequency_hz", frequency_hz)
        object.__setattr__(
            self, "filter_profile", BluetoothRFMeasurementFilterProfile(self.filter_profile)
        )


@dataclass(frozen=True)
class FSKModulationCharacteristics:
    delta_f1_avg_hz: float | None
    delta_f2_avg_hz: float | None
    delta_f2_max_hz: np.ndarray
    delta_f2_ratio: float | None
    sample_count: int
    payload_pattern: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "delta_f2_max_hz", _readonly(self.delta_f2_max_hz, np.float64))


@dataclass(frozen=True)
class ObservedFSKDeviation:
    """Pattern-independent payload deviation after removal of carrier offset."""

    mean_abs_hz: float | None
    percentile_99_9_hz: float | None
    max_abs_hz: float | None
    deviations_hz: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "deviations_hz",
            _readonly(self.deviations_hz, np.float64),
        )


@dataclass(frozen=True)
class InitialCarrierFrequencyResult:
    nominal_frequency_hz: float
    f0_hz: float
    error_hz: float
    selected_bit_indices: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "selected_bit_indices",
            _readonly(self.selected_bit_indices, np.int64),
        )


@dataclass(frozen=True)
class CarrierDriftResult:
    f0_hz: float
    fn_hz: np.ndarray
    max_absolute_offset_hz: float
    max_drift_from_f0_hz: float
    max_drift_rate_hz: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "fn_hz", _readonly(self.fn_hz, np.float64))


def build_fm_measurement_trace(
    iq: np.ndarray,
    *,
    sample_rate_hz: float,
    symbol_rate_hz: float,
    p0_sample: float,
    profile: BluetoothRFMeasurementFilterProfile | str,
) -> BluetoothFMMeasurementTrace:
    """Create an RF measurement trace without CFO/drift/deviation fitting."""

    sample_rate = float(sample_rate_hz)
    filtered = apply_rf_test_channel_filter(
        iq, sample_rate_hz=sample_rate, profile=profile
    )
    if filtered.size < 2:
        raise ValueError("Bluetooth RF measurement requires at least two IQ samples")
    frequency_hz = np.angle(filtered[1:] * np.conj(filtered[:-1])) * sample_rate / (
        2.0 * np.pi
    )
    frequency_hz = np.concatenate(([frequency_hz[0]], frequency_hz))
    return BluetoothFMMeasurementTrace(
        time_s=np.arange(filtered.size, dtype=np.float64) / sample_rate,
        frequency_hz=frequency_hz,
        p0_sample=float(p0_sample),
        sample_rate_hz=sample_rate,
        symbol_rate_hz=float(symbol_rate_hz),
        samples_per_symbol=sample_rate / float(symbol_rate_hz),
        filter_profile=BluetoothRFMeasurementFilterProfile(profile),
    )


def _symbol_grid(
    trace: BluetoothFMMeasurementTrace,
    symbol_count: int,
    *,
    start_symbol: int = 0,
    points_per_symbol: int = 32,
    window: tuple[float, float] = (0.25, 0.75),
) -> np.ndarray:
    """Sample every symbol on a >=32-point fractional measurement grid."""

    count = max(0, int(symbol_count))
    points = max(32, int(points_per_symbol))
    if count == 0:
        return np.empty((0, points), dtype=np.float64)
    fractions = np.linspace(float(window[0]), float(window[1]), points)
    positions = trace.p0_sample + (
        int(start_symbol) + np.arange(count, dtype=np.float64)[:, None] + fractions
    ) * trace.samples_per_symbol
    sample_axis = np.arange(trace.frequency_hz.size, dtype=np.float64)
    valid_rows = positions[:, -1] <= sample_axis[-1]
    positions = positions[valid_rows]
    if positions.size == 0:
        return np.empty((0, points), dtype=np.float64)
    return np.interp(positions, sample_axis, trace.frequency_hz)


def _center_from_bits(values_hz: np.ndarray, bits: np.ndarray) -> float:
    decisions = np.asarray(bits, dtype=np.uint8)[: values_hz.shape[0]]
    symbol_values = np.median(values_hz, axis=1)
    ones = symbol_values[decisions == 1]
    zeros = symbol_values[decisions == 0]
    if ones.size and zeros.size:
        return 0.5 * (float(np.median(ones)) + float(np.median(zeros)))
    return float(np.median(symbol_values))


def classify_payload_pattern(bits: np.ndarray) -> str | None:
    values = np.asarray(bits, dtype=np.uint8)
    if values.size < 16:
        return None
    templates = {
        "11110000": np.resize(np.asarray([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.uint8), values.size),
        "10101010": np.resize(np.asarray([1, 0], dtype=np.uint8), values.size),
    }
    errors = {name: float(np.mean(values != template)) for name, template in templates.items()}
    name = min(errors, key=errors.get)
    return name if errors[name] <= 0.05 else None


def measure_modulation_characteristics(
    trace: BluetoothFMMeasurementTrace,
    payload_bits: np.ndarray,
    *,
    payload_start_symbol: int,
) -> FSKModulationCharacteristics:
    """Measure unscaled Δf values in prescribed central symbol windows."""

    bits = np.asarray(payload_bits, dtype=np.uint8)
    grid = _symbol_grid(
        trace,
        bits.size,
        start_symbol=int(payload_start_symbol),
        points_per_symbol=32,
    )
    bits = bits[: grid.shape[0]]
    if grid.size == 0:
        return FSKModulationCharacteristics(None, None, np.empty(0), None, 0, None)
    center_hz = _center_from_bits(grid, bits)
    deviations = np.abs(grid - center_hz)
    symbol_avg = np.mean(deviations, axis=1)
    symbol_max = np.max(deviations, axis=1)
    pattern = classify_payload_pattern(bits)
    delta_f1 = float(np.mean(symbol_avg)) if pattern == "11110000" else None
    delta_f2 = float(np.mean(symbol_avg)) if pattern == "10101010" else None
    delta_f2_max = symbol_max if pattern == "10101010" else np.empty(0)
    ratio = None
    # A single capture normally carries one RF-test pattern.  Preserve the
    # measured mean so an accumulator can combine F1 and F2 packet sets.
    return FSKModulationCharacteristics(
        delta_f1,
        delta_f2,
        delta_f2_max,
        ratio,
        int(bits.size),
        pattern,
    )


def measure_observed_fsk_deviation(
    trace: BluetoothFMMeasurementTrace,
    *,
    payload_start_symbol: int,
    payload_symbol_count: int,
    carrier_frequency_offset_hz: float,
) -> ObservedFSKDeviation:
    """Measure absolute payload deviation without assuming an RF-test pattern.

    The instantaneous-frequency samples use the same central-symbol windows as
    the SIG modulation measurements.  The independently estimated carrier
    offset is removed before taking the absolute value and statistics.
    """

    grid = _symbol_grid(
        trace,
        int(payload_symbol_count),
        start_symbol=int(payload_start_symbol),
        points_per_symbol=32,
    )
    deviations = np.abs(
        np.asarray(grid, dtype=np.float64).reshape(-1)
        - float(carrier_frequency_offset_hz)
    )
    deviations = deviations[np.isfinite(deviations)]
    if deviations.size == 0:
        return ObservedFSKDeviation(None, None, None, np.empty(0))
    return ObservedFSKDeviation(
        float(np.mean(deviations)),
        float(np.percentile(deviations, 99.9)),
        float(np.max(deviations)),
        deviations,
    )


def measure_initial_carrier_frequency(
    trace: BluetoothFMMeasurementTrace,
    bits: np.ndarray,
    *,
    nominal_frequency_hz: float,
    start_symbol: int = 0,
    symbol_count: int | None = None,
) -> InitialCarrierFrequencyResult:
    values = np.asarray(bits, dtype=np.uint8)
    count = values.size if symbol_count is None else min(values.size, int(symbol_count))
    grid = _symbol_grid(trace, count, start_symbol=int(start_symbol))
    values = values[: grid.shape[0]]
    if grid.size == 0:
        raise ValueError("initial carrier frequency window is outside the capture")
    error_hz = _center_from_bits(grid, values)
    return InitialCarrierFrequencyResult(
        nominal_frequency_hz=float(nominal_frequency_hz),
        f0_hz=float(nominal_frequency_hz) + error_hz,
        error_hz=error_hz,
        selected_bit_indices=(
            int(start_symbol) + np.arange(values.size, dtype=np.int64)
        ),
    )


def measure_carrier_drift(
    trace: BluetoothFMMeasurementTrace,
    bits: np.ndarray,
    *,
    nominal_frequency_hz: float,
    start_symbol: int,
    block_symbols: int,
) -> CarrierDriftResult:
    values = np.asarray(bits, dtype=np.uint8)
    block = max(2, int(block_symbols))
    centers: list[float] = []
    for offset in range(0, values.size - block + 1, block):
        block_bits = values[offset : offset + block]
        grid = _symbol_grid(
            trace,
            block,
            start_symbol=int(start_symbol) + offset,
        )
        if grid.shape[0] != block:
            break
        centers.append(_center_from_bits(grid, block_bits))
    if not centers:
        raise ValueError("carrier drift window is outside the capture")
    offsets_hz = np.asarray(centers, dtype=np.float64)
    f0_hz = float(nominal_frequency_hz) + float(offsets_hz[0])
    fn_hz = float(nominal_frequency_hz) + offsets_hz
    drift = fn_hz - f0_hz
    rates = np.diff(fn_hz) if fn_hz.size > 1 else np.zeros(1)
    return CarrierDriftResult(
        f0_hz=f0_hz,
        fn_hz=fn_hz,
        max_absolute_offset_hz=float(np.max(np.abs(offsets_hz))),
        max_drift_from_f0_hz=float(np.max(np.abs(drift))),
        max_drift_rate_hz=float(np.max(np.abs(rates))),
    )
