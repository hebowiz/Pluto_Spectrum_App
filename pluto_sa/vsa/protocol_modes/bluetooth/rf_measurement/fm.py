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
class P0TimingResult:
    """RF.TS/RFPHY.TS p0 estimate from packet frequency zero crossings."""

    p0_sample: float
    coarse_p0_sample: float
    zero_crossing_samples: np.ndarray
    transition_bit_indices: np.ndarray

    def __post_init__(self) -> None:
        crossings = _readonly(self.zero_crossing_samples, np.float64)
        indices = _readonly(self.transition_bit_indices, np.int64)
        if crossings.shape != indices.shape or crossings.ndim != 1:
            raise ValueError(
                "p0 zero crossings and bit indices must be one-dimensional peers"
            )
        object.__setattr__(self, "zero_crossing_samples", crossings)
        object.__setattr__(self, "transition_bit_indices", indices)

    @property
    def correction_samples(self) -> float:
        return float(self.p0_sample) - float(self.coarse_p0_sample)


@dataclass(frozen=True)
class FSKModulationCharacteristics:
    delta_f1_avg_hz: float | None
    delta_f1_max_hz: np.ndarray
    delta_f2_avg_hz: float | None
    delta_f2_max_hz: np.ndarray
    delta_f2_ratio: float | None
    sample_count: int
    payload_pattern: str | None
    delta_f1_bit_indices: np.ndarray
    delta_f2_bit_indices: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "delta_f1_max_hz",
            _readonly(self.delta_f1_max_hz, np.float64),
        )
        object.__setattr__(self, "delta_f2_max_hz", _readonly(self.delta_f2_max_hz, np.float64))
        object.__setattr__(
            self,
            "delta_f1_bit_indices",
            _readonly(self.delta_f1_bit_indices, np.int64),
        )
        object.__setattr__(
            self,
            "delta_f2_bit_indices",
            _readonly(self.delta_f2_bit_indices, np.int64),
        )


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


def estimate_p0_from_zero_crossings(
    trace: BluetoothFMMeasurementTrace,
    bits: np.ndarray,
    *,
    coarse_p0_sample: float | None = None,
) -> P0TimingResult:
    """Calculate p0 using the RF.TS/RFPHY.TS zero-crossing average.

    For every decoded bit transition, the crossing of the nominal channel
    frequency nearest its coarse boundary is interpolated.  If that crossing
    starts bit ``p(i)``, each observation estimates p0 as
    ``t(i) - p(i) * Tbit``; the prescribed p0 is their arithmetic mean.

    Decoder timing is used only to associate a physical crossing with its bit
    index.  Its fitted timing is not itself used as the RF measurement p0.
    """

    values = np.asarray(trace.frequency_hz, dtype=np.float64)
    decoded = np.asarray(bits, dtype=np.uint8)
    if values.ndim != 1 or decoded.ndim != 1:
        raise ValueError("p0 estimation requires one-dimensional trace and bits")
    if values.size < 2 or decoded.size < 2:
        raise ValueError("p0 estimation requires a packet containing bit transitions")

    coarse = float(
        trace.p0_sample if coarse_p0_sample is None else coarse_p0_sample
    )
    sps = float(trace.samples_per_symbol)
    if not np.isfinite(coarse) or not np.isfinite(sps) or sps <= 0.0:
        raise ValueError("p0 estimation requires finite timing coordinates")

    transitions = np.flatnonzero(decoded[1:] != decoded[:-1]).astype(np.int64) + 1
    crossing_samples: list[float] = []
    crossing_bits: list[int] = []
    for bit_index in transitions:
        expected = coarse + float(bit_index) * sps
        # Adjacent bit boundaries are one symbol apart, so a half-symbol
        # association window cannot assign one physical crossing twice.
        start = max(0, int(np.floor(expected - 0.5 * sps)))
        stop = min(values.size - 1, int(np.ceil(expected + 0.5 * sps)))
        if stop <= start:
            continue

        left = values[start:stop]
        right = values[start + 1 : stop + 1]
        finite = np.isfinite(left) & np.isfinite(right)
        brackets = finite & (
            ((left <= 0.0) & (right >= 0.0))
            | ((left >= 0.0) & (right <= 0.0))
        )
        candidate_offsets = np.flatnonzero(brackets)
        if candidate_offsets.size == 0:
            continue

        expected_rising = bool(decoded[bit_index] > decoded[bit_index - 1])
        slopes = right[candidate_offsets] - left[candidate_offsets]
        directed = (
            candidate_offsets[slopes > 0.0]
            if expected_rising
            else candidate_offsets[slopes < 0.0]
        )
        if directed.size:
            candidate_offsets = directed

        candidates: list[float] = []
        for offset in candidate_offsets:
            sample = start + int(offset)
            y0 = float(values[sample])
            y1 = float(values[sample + 1])
            denominator = y1 - y0
            fraction = (
                0.0
                if denominator == 0.0
                else float(np.clip(-y0 / denominator, 0.0, 1.0))
            )
            candidates.append(float(sample) + fraction)
        crossing = min(candidates, key=lambda sample: abs(sample - expected))
        crossing_samples.append(crossing)
        crossing_bits.append(int(bit_index))

    if not crossing_samples:
        raise ValueError("no nominal-frequency zero crossings were found for p0 estimation")

    crossings = np.asarray(crossing_samples, dtype=np.float64)
    indices = np.asarray(crossing_bits, dtype=np.int64)
    p0_observations = crossings - indices.astype(np.float64) * sps
    return P0TimingResult(
        p0_sample=float(np.mean(p0_observations)),
        coarse_p0_sample=coarse,
        zero_crossing_samples=crossings,
        transition_bit_indices=indices,
    )


def _symbol_grid(
    trace: BluetoothFMMeasurementTrace,
    symbol_count: int,
    *,
    start_symbol: int = 0,
    points_per_symbol: int = 32,
    window: tuple[float, float] = (0.25, 0.75),
    sample_bin_centers: bool = False,
) -> np.ndarray:
    """Sample every symbol on a >=32-point fractional measurement grid."""

    count = max(0, int(symbol_count))
    points = max(32, int(points_per_symbol))
    if count == 0:
        return np.empty((0, points), dtype=np.float64)
    if sample_bin_centers:
        width = float(window[1]) - float(window[0])
        fractions = float(window[0]) + (
            np.arange(points, dtype=np.float64) + 0.5
        ) * width / points
    else:
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


def frequency_deviation_yield_floor(
    deviations_hz: np.ndarray,
    *,
    required_fraction: float = 0.999,
) -> float | None:
    """Return the largest observed floor met by the required sample fraction.

    Unlike an interpolated percentile, this is an actual measured deviation and
    guarantees that at least ``required_fraction`` of the finite observations
    are greater than or equal to the returned value.
    """

    values = np.asarray(deviations_hz, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    fraction = float(required_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("required frequency-deviation yield must be in (0, 1]")
    required_count = int(np.ceil(fraction * values.size))
    allowed_below = values.size - required_count
    return float(np.partition(values, allowed_below)[allowed_below])


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
        window=(0.0, 1.0),
        sample_bin_centers=True,
    )
    bits = bits[: grid.shape[0]]
    if grid.size == 0:
        return FSKModulationCharacteristics(
            None,
            np.empty(0),
            None,
            np.empty(0),
            None,
            0,
            None,
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )
    pattern = classify_payload_pattern(bits)
    is_br = trace.filter_profile is BluetoothRFMeasurementFilterProfile.BR_1M

    def complete_sequences(
        start_bit: int, stop_exclusion: int
    ) -> tuple[np.ndarray, np.ndarray]:
        stop = max(int(start_bit), bits.size - int(stop_exclusion))
        sequence_count = max(0, (stop - int(start_bit)) // 8)
        evaluated_count = sequence_count * 8
        if evaluated_count == 0:
            return (
                np.empty((0, 8, grid.shape[1]), dtype=np.float64),
                np.empty((0, 8), dtype=np.int64),
            )
        sequence_grid = grid[
            int(start_bit) : int(start_bit) + evaluated_count
        ].reshape(sequence_count, 8, grid.shape[1])
        bit_indices = np.arange(
            int(start_bit), int(start_bit) + evaluated_count, dtype=np.int64
        ).reshape(sequence_count, 8)
        return sequence_grid, bit_indices

    delta_f1_values = np.empty(0, dtype=np.float64)
    delta_f1_indices = np.empty(0, dtype=np.int64)
    if pattern == "11110000":
        # Both CMW's RF.TS implementation and RFPHY.TS discard the first four
        # and final incomplete four payload bits. This aligns the evaluated
        # groups to 00001111. Only bits 2, 3, 6 and 7 of each such sequence
        # contribute to the Delta-f1 results.
        sequence_grid, sequence_indices = complete_sequences(4, 4)
        if sequence_grid.size:
            sequence_center_hz = np.mean(sequence_grid, axis=(1, 2))
            bit_average_hz = np.mean(sequence_grid, axis=2)
            bit_deviation_hz = np.abs(
                bit_average_hz - sequence_center_hz[:, None]
            )
            selected = np.asarray((1, 2, 5, 6), dtype=np.int64)
            delta_f1_values = bit_deviation_hz[:, selected].reshape(-1)
            delta_f1_indices = sequence_indices[:, selected].reshape(-1)

    delta_f2_max = np.empty(0, dtype=np.float64)
    delta_f2_indices = np.empty(0, dtype=np.int64)
    if pattern == "10101010":
        # RF.TS begins with payload bit two (01010101 sequences). RFPHY.TS
        # discards four bits at each payload edge and starts at bit five.
        sequence_grid, sequence_indices = complete_sequences(
            1 if is_br else 4,
            0 if is_br else 4,
        )
        if sequence_grid.size:
            sequence_center_hz = np.mean(sequence_grid, axis=(1, 2))
            delta_f2_max = np.max(
                np.abs(sequence_grid - sequence_center_hz[:, None, None]),
                axis=2,
            ).reshape(-1)
            delta_f2_indices = sequence_indices.reshape(-1)

    delta_f1 = (
        float(np.mean(delta_f1_values)) if delta_f1_values.size else None
    )
    delta_f2 = float(np.mean(delta_f2_max)) if delta_f2_max.size else None
    # A single capture normally carries one RF-test pattern. Preserve each
    # recorded value so the accumulator can combine the required packet sets.
    return FSKModulationCharacteristics(
        delta_f1,
        delta_f1_values,
        delta_f2,
        delta_f2_max,
        None,
        int(max(delta_f1_values.size, delta_f2_max.size)),
        pattern,
        delta_f1_indices,
        delta_f2_indices,
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
