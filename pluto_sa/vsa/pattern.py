"""Modulation-agnostic known-pattern search and symbol decoding."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from fractions import Fraction
from types import MappingProxyType
from typing import Mapping

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.optimize import least_squares, minimize_scalar
from scipy.signal import resample_poly

from pluto_sa.vsa.demod.gfsk import demodulate_gfsk
from pluto_sa.vsa.mapping import (
    BLUETOOTH_EDR_MAPPING,
    NATURAL_MAPPING,
    psk_constellation,
    reverse_symbol_bits,
)
from pluto_sa.vsa.model import IQRecording, ModulationFamily, ModulationKind, SignalDescription


_EPSILON = np.finfo(np.float64).tiny


def _readonly(values: np.ndarray, dtype: np.dtype | type) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class KnownPattern:
    """R&S-style reusable pattern definition, independent of search settings."""

    symbols: tuple[int, ...]
    name: str = "Known Pattern"
    description: str = ""

    def __post_init__(self) -> None:
        symbols = tuple(int(symbol) for symbol in self.symbols)
        if len(symbols) < 4:
            raise ValueError("known pattern must contain at least four symbols")
        if any(symbol < 0 for symbol in symbols):
            raise ValueError("known pattern symbols must be non-negative")
        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(self, "name", str(self.name).strip() or "Known Pattern")


class PatternSearchMode(str, Enum):
    AUTO = "Auto"
    ON = "On"
    OFF = "Off"


class MatchSelectionPolicy(str, Enum):
    STRONGEST = "Strongest"
    FIRST = "First"
    LAST = "Last"
    INDEX = "Match Index"


@dataclass(frozen=True)
class IQPowerTriggerSettings:
    """Post-capture I/Q power trigger used to gate pattern search.

    The level follows the same calibrated dBm convention as the VSA I/Q Power
    trace.  Timing values are expressed in symbols so one configuration scales
    naturally with the selected signal description.
    """

    enabled: bool = False
    level_dbm: float = -20.0
    hysteresis_db: float = 3.0
    envelope_average_symbols: float = 1.0
    dropout_symbols: float = 8.0
    holdoff_symbols: float = 0.0
    search_start_offset_symbols: float = 0.0
    limit_result_to_active_interval: bool = True

    def __post_init__(self) -> None:
        values = (
            self.level_dbm,
            self.hysteresis_db,
            self.envelope_average_symbols,
            self.dropout_symbols,
            self.holdoff_symbols,
            self.search_start_offset_symbols,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("I/Q power trigger settings must be finite")
        if self.hysteresis_db < 0.0:
            raise ValueError("hysteresis_db must be non-negative")
        if self.envelope_average_symbols < 0.0:
            raise ValueError("envelope_average_symbols must be non-negative")
        if self.dropout_symbols < 0.0:
            raise ValueError("dropout_symbols must be non-negative")
        if self.holdoff_symbols < 0.0:
            raise ValueError("holdoff_symbols must be non-negative")


@dataclass(frozen=True)
class IQPowerTriggerEvent:
    trigger_sample: int
    active_stop_sample: int


@dataclass(frozen=True)
class PatternSearchSettings:
    """Settings corresponding to R&S VSA ``Pattern Search``."""

    pattern: KnownPattern
    mode: PatternSearchMode = PatternSearchMode.AUTO
    iq_correlation_threshold: float = 0.9
    correlation_threshold_auto: bool = True
    meas_only_if_pattern_symbols_correct: bool = True
    match_selection: MatchSelectionPolicy = MatchSelectionPolicy.FIRST
    match_index: int = 1
    iq_power_trigger: IQPowerTriggerSettings = field(
        default_factory=IQPowerTriggerSettings
    )
    allow_inverted_fsk_pattern: bool = False

    def __post_init__(self) -> None:
        threshold = float(self.iq_correlation_threshold)
        if not 0.0 < threshold <= 1.0:
            raise ValueError("iq_correlation_threshold must be in the range (0, 1]")
        if int(self.match_index) < 1:
            raise ValueError("match_index must be one-based and positive")
        object.__setattr__(
            self, "match_selection", MatchSelectionPolicy(self.match_selection)
        )
        object.__setattr__(self, "match_index", int(self.match_index))

    @property
    def effective_correlation_threshold(self) -> float:
        return 0.9 if self.correlation_threshold_auto else float(
            self.iq_correlation_threshold
        )


def detect_iq_power_trigger_events(
    recording: IQRecording,
    *,
    symbol_rate_hz: float,
    settings: IQPowerTriggerSettings,
) -> tuple[IQPowerTriggerEvent, ...]:
    """Return every rising I/Q-power event and its hysteretic active interval."""

    if not np.isfinite(symbol_rate_hz) or float(symbol_rate_hz) <= 0.0:
        raise ValueError("symbol_rate_hz must be positive")
    samples_per_symbol = float(recording.sample_rate_hz) / float(symbol_rate_hz)
    normalized_power = (
        np.abs(np.asarray(recording.iq, dtype=np.complex128))
        / float(recording.full_scale)
    ) ** 2
    average_samples = max(
        1,
        int(round(float(settings.envelope_average_symbols) * samples_per_symbol)),
    )
    envelope_power = normalized_power
    if average_samples > 1:
        envelope_power = uniform_filter1d(
            normalized_power,
            size=average_samples,
            mode="nearest",
        )
    raw_power_dbm = (
        10.0 * np.log10(np.maximum(normalized_power, _EPSILON))
        + recording.dbfs_to_dbm_offset_db
    )
    envelope_power_dbm = (
        10.0 * np.log10(np.maximum(envelope_power, _EPSILON))
        + recording.dbfs_to_dbm_offset_db
    )
    level = float(settings.level_dbm)
    rearm_level = level - float(settings.hysteresis_db)
    dropout_samples = max(
        1, int(round(float(settings.dropout_symbols) * samples_per_symbol))
    )
    holdoff_samples = max(
        0, int(round(float(settings.holdoff_symbols) * samples_per_symbol))
    )

    events: list[IQPowerTriggerEvent] = []
    sample_count = int(raw_power_dbm.size)
    cursor = 0
    last_trigger = -holdoff_samples
    while cursor < sample_count:
        above = np.flatnonzero(raw_power_dbm[cursor:] >= level)
        if above.size == 0:
            break
        trigger = cursor + int(above[0])
        if trigger - last_trigger < holdoff_samples:
            cursor = max(trigger + 1, last_trigger + holdoff_samples)
            continue

        low_run = 0
        active_stop = sample_count
        scan = trigger + 1
        while scan < sample_count:
            if envelope_power_dbm[scan] <= rearm_level:
                low_run += 1
                if low_run >= dropout_samples:
                    active_stop = max(
                        trigger + 1,
                        scan - dropout_samples + 1 - average_samples // 2,
                    )
                    scan += 1
                    break
            else:
                low_run = 0
            scan += 1
        events.append(IQPowerTriggerEvent(trigger, active_stop))
        last_trigger = trigger
        if active_stop >= sample_count:
            break
        cursor = max(scan, trigger + holdoff_samples)
    return tuple(events)

class ResultRangeReference(str, Enum):
    CAPTURE = "Capture"
    BURST = "Burst"
    PATTERN_WAVEFORM = "Pattern Waveform"


class ResultRangeAlignment(str, Enum):
    LEFT = "Left"
    CENTER = "Center"
    RIGHT = "Right"


@dataclass(frozen=True)
class ResultRangeSettings:
    """Settings corresponding to R&S VSA ``Result Range``."""

    result_length: int = 256
    reference: ResultRangeReference = ResultRangeReference.PATTERN_WAVEFORM
    alignment: ResultRangeAlignment = ResultRangeAlignment.LEFT
    offset_symbols: int = 0
    symbol_number_at_reference_start: int = 0
    exclude_incomplete_result: bool = False

    def __post_init__(self) -> None:
        if int(self.result_length) <= 0:
            raise ValueError("result_length must be positive")


class SynchronizationSource(str, Enum):
    AUTO = "Auto"
    DETECTED_DATA = "Detected Data"
    PATTERN = "Pattern"
    KNOWN_DATA = "Known Data"


class BitOrdering(str, Enum):
    MSB = "MSB"
    LSB = "LSB"


class MeasurementFilterMode(str, Enum):
    NONE = "None"
    AUTO = "Auto"


@dataclass(frozen=True)
class DemodulationSettings:
    """Implemented subset of R&S VSA ``Demodulation`` settings."""

    coarse_synchronization: SynchronizationSource = SynchronizationSource.AUTO
    fine_synchronization: SynchronizationSource = SynchronizationSource.AUTO
    measurement_filter: MeasurementFilterMode = MeasurementFilterMode.AUTO
    bit_ordering: BitOrdering = BitOrdering.LSB
    compensate_carrier_frequency_drift: bool = False
    compensate_fsk_deviation_error: bool = True


@dataclass(frozen=True)
class PatternSearchResult:
    modulation: ModulationKind
    pattern_start_sample: int
    pattern_start_time_s: float
    pattern_start_symbol: int
    result_start_sample: int
    result_stop_sample: int
    correlation: float
    pattern_symbol_errors: int
    decoded_symbols: np.ndarray
    decoded_bits: np.ndarray
    measured_symbols: np.ndarray
    symbol_time_s: np.ndarray
    carrier_frequency_offset_hz: float
    carrier_frequency_drift_hz_per_s: float
    frequency_deviation_hz: float | None
    evm_rms_percent: float | None
    polarity_inverted: bool
    phase_rotation_rad: float | None
    timing_phase_samples: int
    analysis_sample_rate_hz: float
    recording_sample_rate_hz: float
    carrier_reference_time_s: float
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "decoded_symbols", _readonly(self.decoded_symbols, np.int16)
        )
        object.__setattr__(self, "decoded_bits", _readonly(self.decoded_bits, np.uint8))
        object.__setattr__(
            self, "measured_symbols", _readonly(self.measured_symbols, np.complex64)
        )
        object.__setattr__(self, "symbol_time_s", _readonly(self.symbol_time_s, np.float64))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        if self.evm_rms_percent is not None and (
            not np.isfinite(self.evm_rms_percent) or self.evm_rms_percent < 0.0
        ):
            raise ValueError("evm_rms_percent must be finite and non-negative")

    @property
    def pattern_stop_symbol(self) -> int:
        return self.pattern_start_symbol + int(self.metadata["pattern_symbol_count"])

    @property
    def result_start_time_s(self) -> float:
        return self.result_start_sample / self.recording_sample_rate_hz

    @property
    def result_stop_time_s(self) -> float:
        return self.result_stop_sample / self.recording_sample_rate_hz

    @property
    def pattern_stop_time_s(self) -> float:
        return self.pattern_start_time_s + (
            int(self.metadata["pattern_symbol_count"])
            / float(self.metadata["symbol_rate_hz"])
        )


def carrier_correct_recording(
    recording: IQRecording,
    result: PatternSearchResult,
    *,
    compensate_drift: bool = True,
) -> IQRecording:
    """Remove the pattern-derived carrier phase model from every IQ sample."""
    if recording.sample_rate_hz != result.recording_sample_rate_hz:
        raise ValueError("recording sample rate does not match the pattern result")
    time_s = np.arange(recording.sample_count, dtype=np.float64) / float(
        recording.sample_rate_hz
    )
    relative_time_s = time_s - float(result.carrier_reference_time_s)
    drift_hz_per_s = (
        float(result.carrier_frequency_drift_hz_per_s)
        if compensate_drift
        else 0.0
    )
    phase_rad = 2.0 * np.pi * (
        float(result.carrier_frequency_offset_hz) * relative_time_s
        + 0.5 * drift_hz_per_s * relative_time_s**2
    )
    if result.phase_rotation_rad is not None:
        phase_rad = phase_rad + float(result.phase_rotation_rad)
    corrected = np.asarray(recording.iq) * np.exp(-1j * phase_rad)
    return replace(
        recording,
        iq=corrected,
        source=f"{recording.source} | Carrier Corrected",
        metadata={
            **dict(recording.metadata),
            "carrier_corrected": True,
            "carrier_frequency_offset_hz": result.carrier_frequency_offset_hz,
            "carrier_frequency_drift_hz_per_s": drift_hz_per_s,
            "carrier_drift_compensated": bool(compensate_drift),
            "carrier_reference_time_s": result.carrier_reference_time_s,
            "carrier_phase_rotation_rad": result.phase_rotation_rad,
        },
    )


def _constellation(
    kind: ModulationKind, mapping: str = NATURAL_MAPPING
) -> np.ndarray:
    """Compatibility wrapper for tests and internal callers."""
    return psk_constellation(kind, mapping)


def _symbols_to_bits(
    symbols: np.ndarray, order: int, bit_ordering: BitOrdering
) -> np.ndarray:
    bit_count = int(round(np.log2(order)))
    shifts = (
        np.arange(bit_count, dtype=np.int16)
        if bit_ordering is BitOrdering.LSB
        else np.arange(bit_count - 1, -1, -1, dtype=np.int16)
    )
    return ((symbols[:, None] >> shifts) & 1).astype(np.uint8).reshape(-1)


def _result_start_symbol(
    pattern_size: int, settings: ResultRangeSettings
) -> int:
    if settings.reference is not ResultRangeReference.PATTERN_WAVEFORM:
        raise NotImplementedError(
            "pattern analysis currently supports Pattern Waveform result reference"
        )
    if settings.alignment is ResultRangeAlignment.LEFT:
        start = int(settings.offset_symbols)
    elif settings.alignment is ResultRangeAlignment.CENTER:
        start = pattern_size // 2 - int(settings.result_length) // 2
        start += int(settings.offset_symbols)
    else:
        start = pattern_size - int(settings.result_length) + int(settings.offset_symbols)
    if start < 0:
        raise NotImplementedError(
            "negative result-range positions require pre-pattern demodulation"
        )
    return start


def _result_is_complete(
    pattern_size: int, available: int, settings: ResultRangeSettings
) -> bool:
    start = _result_start_symbol(pattern_size, settings)
    return int(available) >= start + int(settings.result_length)


def _result_slice(
    pattern_size: int, available: int, settings: ResultRangeSettings
) -> slice:
    start = _result_start_symbol(pattern_size, settings)
    if start >= int(available):
        raise ValueError("result range starts after the available demodulated symbols")
    if settings.exclude_incomplete_result and not _result_is_complete(
        pattern_size, available, settings
    ):
        raise ValueError("complete result range is not available after the pattern")
    stop = min(int(available), start + int(settings.result_length))
    return slice(start, max(start, stop))


def _select_match_candidate(
    candidates: list[object],
    policy: MatchSelectionPolicy,
    match_index: int,
    *,
    time_key,
    score_key,
) -> tuple[object, int, int]:
    """Select one candidate and return it with one-based time index/count."""
    if not candidates:
        raise ValueError("no pattern match satisfies the result-range requirements")
    ordered = sorted(candidates, key=time_key)
    if policy is MatchSelectionPolicy.FIRST:
        selected = ordered[0]
    elif policy is MatchSelectionPolicy.LAST:
        selected = ordered[-1]
    elif policy is MatchSelectionPolicy.INDEX:
        requested = int(match_index)
        if requested > len(ordered):
            raise ValueError(
                f"Match Index {requested} is unavailable; "
                f"{len(ordered)} eligible match(es) were found"
            )
        selected = ordered[requested - 1]
    else:
        selected = max(ordered, key=lambda item: (score_key(item), -time_key(item)))
    return selected, ordered.index(selected) + 1, len(ordered)


def _local_peak_indices(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Return one index for each above-threshold correlation peak."""
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0:
        return np.empty(0, dtype=np.int64)
    magnitude = np.abs(values)
    selected: list[int] = []
    for index in np.flatnonzero(magnitude >= float(threshold)):
        left = magnitude[index - 1] if index > 0 else -np.inf
        right = magnitude[index + 1] if index + 1 < magnitude.size else -np.inf
        if magnitude[index] >= left and magnitude[index] > right:
            selected.append(int(index))
    return np.asarray(selected, dtype=np.int64)


def _resample_for_symbols(
    iq: np.ndarray,
    sample_rate_hz: float,
    symbol_rate_hz: float,
    samples_per_symbol: int = 8,
) -> tuple[np.ndarray, float]:
    target_rate_hz = float(symbol_rate_hz) * int(samples_per_symbol)
    ratio = Fraction(target_rate_hz / float(sample_rate_hz)).limit_denominator(4096)
    result = resample_poly(iq, ratio.numerator, ratio.denominator)
    actual_rate_hz = float(sample_rate_hz) * ratio.numerator / ratio.denominator
    if abs(actual_rate_hz - target_rate_hz) / target_rate_hz > 1e-5:
        raise ValueError("sample rate cannot be resampled accurately enough")
    return np.asarray(result, dtype=np.complex128), actual_rate_hz


def _root_raised_cosine_taps(
    samples_per_symbol: int, beta: float, span_symbols: int = 10
) -> np.ndarray:
    """Return unit-energy SRRC taps for offline matched filtering."""
    sps = int(samples_per_symbol)
    if sps < 2 or not 0.0 < float(beta) <= 1.0:
        raise ValueError("SRRC requires samples_per_symbol >= 2 and 0 < beta <= 1")
    time = np.arange(-span_symbols * sps / 2, span_symbols * sps / 2 + 1) / sps
    taps = np.empty(time.size, dtype=np.float64)
    for index, value in enumerate(time):
        if np.isclose(value, 0.0):
            taps[index] = 1.0 + beta * (4.0 / np.pi - 1.0)
        elif np.isclose(abs(value), 1.0 / (4.0 * beta)):
            taps[index] = beta / np.sqrt(2.0) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
                + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
            )
        else:
            numerator = (
                np.sin(np.pi * value * (1.0 - beta))
                + 4.0 * beta * value * np.cos(np.pi * value * (1.0 + beta))
            )
            denominator = np.pi * value * (1.0 - (4.0 * beta * value) ** 2)
            taps[index] = numerator / denominator
    return taps / np.sqrt(np.sum(taps**2))


def prepare_psk_iq(
    iq: np.ndarray,
    *,
    sample_rate_hz: float,
    symbol_rate_hz: float,
    tx_filter: str,
    filter_parameter: float | None,
    samples_per_symbol: int = 8,
    prefilter_carrier_frequency_offset_hz: float = 0.0,
    apply_measurement_filter: bool = True,
) -> tuple[np.ndarray, float]:
    """Resample PSK IQ and apply carrier centering before the receive filter."""
    waveform, analysis_rate_hz = _resample_for_symbols(
        iq,
        sample_rate_hz,
        symbol_rate_hz,
        samples_per_symbol=samples_per_symbol,
    )
    coarse_cfo_hz = float(prefilter_carrier_frequency_offset_hz)
    if not np.isfinite(coarse_cfo_hz):
        raise ValueError("prefilter carrier frequency offset must be finite")
    if coarse_cfo_hz != 0.0:
        time_s = np.arange(waveform.size, dtype=np.float64) / analysis_rate_hz
        waveform = waveform * np.exp(-2j * np.pi * coarse_cfo_hz * time_s)
    if apply_measurement_filter and tx_filter.lower() in {
        "root raised cosine",
        "root-raised-cosine",
        "rrc",
        "srrc",
    }:
        beta = 0.4 if filter_parameter is None else float(filter_parameter)
        waveform = np.convolve(
            waveform,
            _root_raised_cosine_taps(samples_per_symbol, beta),
            mode="same",
        )
    return np.asarray(waveform, dtype=np.complex128), analysis_rate_hz


def _normalized_complex_correlation(
    values: np.ndarray, pattern: np.ndarray
) -> np.ndarray:
    if values.size < pattern.size:
        return np.empty(0, dtype=np.float64)
    numerator = np.correlate(values, pattern, mode="valid")
    window_energy = np.convolve(
        np.abs(values) ** 2,
        np.ones(pattern.size, dtype=np.float64),
        mode="valid",
    )
    pattern_energy = float(np.sum(np.abs(pattern) ** 2))
    denominator = np.sqrt(np.maximum(window_energy * pattern_energy, _EPSILON))
    return np.abs(numerator) / denominator


def _interpolate_complex(
    waveform: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    """Linearly sample a complex waveform at fractional sample positions."""
    sample_index = np.arange(waveform.size, dtype=np.float64)
    return np.interp(positions, sample_index, waveform.real) + 1j * np.interp(
        positions, sample_index, waveform.imag
    )


def _weighted_phase_line(
    symbol_indices: np.ndarray,
    phase_rad: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    x = np.asarray(symbol_indices, dtype=np.float64)
    y = np.asarray(phase_rad, dtype=np.float64)
    weight = np.maximum(np.asarray(weights, dtype=np.float64), _EPSILON)
    root_weight = np.sqrt(weight / np.mean(weight))
    design = np.column_stack((np.ones(x.size), x))
    parameters = np.linalg.lstsq(
        design * root_weight[:, None],
        y * root_weight,
        rcond=None,
    )[0]
    return float(parameters[0]), float(parameters[1])


def _robust_circular_location(
    phase_rad: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Return a robust mean angle without converting phase to a line.

    Keeping the samples on the unit circle avoids the permanent cycle slips
    that an ordinary ``unwrap`` can create after a faded or disturbed symbol.
    The Tukey reweighting also stops a small group of bad differential symbols
    from steering the carrier-drift estimate.
    """
    phase = np.asarray(phase_rad, dtype=np.float64)
    base_weight = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    if phase.size == 0 or phase.size != base_weight.size:
        raise ValueError("circular location requires equally sized non-empty arrays")
    if not np.any(base_weight > 0.0):
        base_weight = np.ones(phase.size, dtype=np.float64)
    center = float(np.angle(np.sum(base_weight * np.exp(1j * phase))))
    for _ in range(6):
        residual = np.angle(np.exp(1j * (phase - center)))
        scale = 1.4826 * float(np.median(np.abs(residual)))
        cutoff = max(np.deg2rad(2.0), 4.685 * scale)
        ratio = residual / cutoff
        robust_weight = np.where(
            np.abs(ratio) < 1.0,
            (1.0 - ratio**2) ** 2,
            0.0,
        )
        effective_weight = base_weight * robust_weight
        if float(np.sum(effective_weight)) <= _EPSILON:
            break
        correction = float(
            np.sum(effective_weight * residual) / np.sum(effective_weight)
        )
        center += correction
        if abs(correction) < 1e-10:
            break
    return _wrap_phase(center)


def _robust_phase_line_update(
    symbol_indices: np.ndarray,
    wrapped_residual_rad: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    """Fit a small intercept/slope correction while rejecting phase outliers."""
    residual = np.asarray(wrapped_residual_rad, dtype=np.float64)
    base_weight = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    scale = 1.4826 * float(np.median(np.abs(residual - np.median(residual))))
    cutoff = max(np.deg2rad(2.0), 4.685 * scale)
    ratio = residual / cutoff
    robust_weight = np.where(
        np.abs(ratio) < 1.0,
        (1.0 - ratio**2) ** 2,
        0.0,
    )
    effective_weight = base_weight * robust_weight
    if np.count_nonzero(effective_weight > 0.0) < 4:
        effective_weight = base_weight
    return _weighted_phase_line(symbol_indices, residual, effective_weight)


def _fit_differential_psk_phase_model(
    measured_symbols: np.ndarray,
    symbol_indices: np.ndarray,
    alphabet: np.ndarray,
    *,
    pattern_phase_anchor_rad: float,
    pattern_center_symbol: float,
) -> tuple[float, float, float, bool, float]:
    """Estimate differential-PSK carrier offset/drift over Result Range.

    Raising an M-PSK differential symbol to the Mth power removes its data
    phase.  The known pattern resolves the remaining 2*pi/M ambiguity.  Two
    reference-directed iterations then minimize phase error over all detected
    Result Range symbols, matching R&S Auto synchronization's preference for
    detected data when the known pattern is short.
    """
    measured = np.asarray(measured_symbols, dtype=np.complex128)
    x = np.asarray(symbol_indices, dtype=np.float64)
    reference_alphabet = np.asarray(alphabet, dtype=np.complex128)
    if measured.size != x.size or measured.size < 4:
        raise ValueError("differential PSK phase fit requires at least four symbols")
    order = int(reference_alphabet.size)
    invariant = reference_alphabet[0] ** order
    modulation_removed = measured**order / invariant
    weights = np.abs(measured) ** 2

    # The phase change between adjacent modulation-removed symbols is M times
    # the carrier-drift slope.  Estimate it directly on the unit circle.  An
    # unwrap-based line fit can acquire a whole-cycle slip at one weak symbol,
    # which appears as a very large but entirely artificial carrier drift.
    unit = modulation_removed / np.maximum(np.abs(modulation_removed), _EPSILON)
    adjacent_phase = np.angle(unit[1:] * np.conj(unit[:-1]))
    adjacent_weight = np.sqrt(weights[1:] * weights[:-1])
    slope = _robust_circular_location(adjacent_phase, adjacent_weight) / float(order)
    intercept_phase = np.angle(
        unit * np.exp(-1j * float(order) * slope * x)
    )
    intercept = _robust_circular_location(intercept_phase, weights) / float(order)

    ambiguity = 2.0 * np.pi / float(order)
    fitted_at_pattern = intercept + slope * float(pattern_center_symbol)
    intercept += ambiguity * round(
        (float(pattern_phase_anchor_rad) - fitted_at_pattern) / ambiguity
    )

    for _ in range(2):
        predicted = intercept + slope * x
        corrected = measured * np.exp(-1j * predicted)
        rms = float(np.sqrt(np.mean(np.abs(corrected) ** 2)))
        normalized = corrected / max(rms, _EPSILON)
        decoded = np.argmin(
            np.abs(normalized[:, None] - reference_alphabet[None, :]), axis=1
        )
        reference = reference_alphabet[decoded]
        observed_error = np.angle(measured * np.conj(reference))
        residual_error = np.angle(
            np.exp(1j * (observed_error - predicted))
        )
        intercept_update, _ = _robust_phase_line_update(
            x, residual_error, weights
        )
        intercept += intercept_update

    def residual_rms(model_intercept: float, model_slope: float) -> float:
        corrected_model = measured * np.exp(
            -1j * (model_intercept + model_slope * x)
        )
        decoded_model = np.argmin(
            np.abs(corrected_model[:, None] - reference_alphabet[None, :]),
            axis=1,
        )
        residual = np.angle(
            corrected_model * np.conj(reference_alphabet[decoded_model])
        )
        return float(
            np.sqrt(
                np.sum(weights * residual**2)
                / max(float(np.sum(weights)), _EPSILON)
            )
        )

    candidate_residual_rms = residual_rms(intercept, slope)

    # Compare the drift model with a CFO-only model.  Blind detected-data
    # synchronization has periodic false solutions; a reported drift must at
    # least reduce the Result Range phase error.  Otherwise retain the robust
    # CFO estimate and report zero drift rather than applying an unstable fit.
    no_drift_intercept = (
        _robust_circular_location(np.angle(unit), weights) / float(order)
    )
    no_drift_intercept += ambiguity * round(
        (
            float(pattern_phase_anchor_rad)
            - no_drift_intercept
        )
        / ambiguity
    )
    for _ in range(2):
        corrected_no_drift = measured * np.exp(-1j * no_drift_intercept)
        decoded_no_drift = np.argmin(
            np.abs(
                corrected_no_drift[:, None] - reference_alphabet[None, :]
            ),
            axis=1,
        )
        no_drift_error = np.angle(
            corrected_no_drift
            * np.conj(reference_alphabet[decoded_no_drift])
        )
        no_drift_intercept += _robust_circular_location(
            no_drift_error, weights
        )
    no_drift_residual_rms = residual_rms(no_drift_intercept, 0.0)
    # A fitted slope consumes one extra degree of freedom, so tiny numerical
    # differences around the same physical solution are expected.  Permit a
    # 0.1 degree RMS tolerance; reject only a materially worse periodic alias.
    drift_accepted = candidate_residual_rms <= (
        no_drift_residual_rms + np.deg2rad(0.1)
    )
    if not drift_accepted:
        intercept = no_drift_intercept
        slope = 0.0
        candidate_residual_rms = no_drift_residual_rms
    return (
        intercept,
        slope,
        candidate_residual_rms,
        drift_accepted,
        no_drift_residual_rms,
    )


def _wrap_phase(value: float) -> float:
    return float((float(value) + np.pi) % (2.0 * np.pi) - np.pi)


def _detected_psk_decision_interval(
    normalized_symbols: np.ndarray,
    order: int,
) -> tuple[int, int, float]:
    """Return the longest interval with stable M-PSK decision geometry.

    Raising unit phasors to the Mth power removes the transmitted M-PSK data,
    leaving a concentration measure that is invariant to the unknown carrier
    rotation.  This lets a mixed FSK/PSK burst select its PSK portion without
    protocol knowledge or a known symbol pattern.
    """
    values = np.asarray(normalized_symbols, dtype=np.complex128)
    count = int(values.size)
    if count < 32:
        return 0, count, 0.0
    window = min(32, max(16, count // 8))
    mth_power = np.exp(1j * float(order) * np.angle(values))
    concentration = np.abs(
        np.convolve(mth_power, np.ones(window) / float(window), mode="valid")
    )
    # 0.65 tolerates roughly 15-degree RMS phase scatter for 8PSK while
    # rejecting the broad modulo-M distribution produced by the FSK header in
    # the mHDT capture.  Fall back to the complete interval when no sustained
    # PSK-like run exists, preserving generic noisy-signal behavior.
    good = concentration >= 0.65
    runs: list[tuple[int, int]] = []
    run_start: int | None = None
    for index, is_good in enumerate(good):
        if is_good and run_start is None:
            run_start = index
        if run_start is not None and (not is_good or index == good.size - 1):
            run_stop = index if not is_good else index + 1
            runs.append((run_start, min(count, run_stop + window)))
            run_start = None
    if not runs:
        return 0, count, float(np.max(concentration, initial=0.0))
    start, stop = max(runs, key=lambda item: item[1] - item[0])
    minimum_sustained = max(32, int(np.ceil(0.2 * count)))
    if stop - start < minimum_sustained:
        return 0, count, float(np.max(concentration, initial=0.0))
    interval_concentration = float(
        np.mean(concentration[start : max(start + 1, stop - window + 1)])
    )
    return int(start), int(stop), interval_concentration


def _detected_qam_decision_interval(
    symbols: np.ndarray,
) -> tuple[int, int, float]:
    """Locate a sustained square-QAM interval from its amplitude variation."""
    values = np.abs(np.asarray(symbols, dtype=np.complex128))
    count = int(values.size)
    if count < 32:
        return 0, count, 0.0

    window = min(32, max(16, count // 8))
    kernel = np.ones(window, dtype=np.float64) / float(window)
    mean = np.convolve(values, kernel, mode="valid")
    mean_square = np.convolve(values**2, kernel, mode="valid")
    coefficient = np.sqrt(
        np.maximum(0.0, mean_square - mean**2)
    ) / np.maximum(mean, _EPSILON)

    # A square-QAM payload has several sustained amplitude radii.  This
    # separates it from a constant-envelope FSK/PSK header without applying
    # the PSK phase-concentration gate that previously truncated QAM data.
    reference_level = float(np.quantile(mean, 0.9))
    active_level = 0.25 * reference_level
    good = (coefficient >= 0.18) & (mean >= active_level)
    runs: list[tuple[int, int, float]] = []
    run_start: int | None = None
    for index, is_good in enumerate(good):
        if is_good and run_start is None:
            run_start = index
        if run_start is not None and (not is_good or index == good.size - 1):
            run_stop = index if not is_good else index + 1
            start = (
                0
                if run_start == 0 and mean[0] >= 0.6 * reference_level
                else min(count, run_start + 3 * window // 4)
            )
            stop = (
                count
                if (
                    run_stop == good.size
                    and mean[-1] >= 0.6 * reference_level
                )
                else max(start, min(count, run_stop - window))
            )
            if stop - start >= window:
                runs.append(
                    (
                        start,
                        stop,
                        float(np.mean(coefficient[run_start:run_stop])),
                    )
                )
            run_start = None
    if not runs:
        return 0, count, float(np.max(coefficient, initial=0.0))
    start, stop, concentration = max(runs, key=lambda item: item[1] - item[0])
    minimum_sustained = max(32, int(np.ceil(0.2 * count)))
    if stop - start < minimum_sustained:
        return 0, count, float(np.max(coefficient, initial=0.0))
    return int(start), int(stop), concentration


def _psk_carrier_symmetry_order(modulation: ModulationKind) -> int:
    """Return physical carrier-recovery symmetry, independent of decisions."""
    if modulation is ModulationKind.QAM16:
        return 4
    if modulation in {ModulationKind.PI4_DQPSK, ModulationKind.DPSK8}:
        # pi/4-DQPSK has four differential decisions but alternates between
        # two QPSK constellations; its physical IQ therefore has eightfold
        # rotational symmetry, just like 8DPSK.
        return 8
    return int(modulation.order)


def _fit_nondifferential_psk_carrier(
    symbols: np.ndarray,
    alphabet: np.ndarray,
    symmetry_order: int,
) -> tuple[float, float, float]:
    """Fit carrier phase/CFO from the rotational symmetry of absolute PSK."""
    values = np.asarray(symbols, dtype=np.complex128)
    order = int(symmetry_order)
    if values.size < 9 or order < 2:
        return 0.0, 0.0, np.inf
    unit = values / np.maximum(np.abs(values), _EPSILON)
    reference = np.asarray(alphabet, dtype=np.complex128)
    reference_unit = reference / np.maximum(np.abs(reference), _EPSILON)
    reference_moment = np.sum(reference_unit**order)
    powered = unit**order * np.exp(-1j * np.angle(reference_moment))
    axis = np.arange(powered.size, dtype=np.float64)
    initial_slope, initial_intercept = np.polyfit(
        axis, np.unwrap(np.angle(powered)), 1
    )

    def residual(parameters: np.ndarray) -> np.ndarray:
        intercept, slope = parameters
        return np.angle(
            powered * np.exp(-1j * (intercept + slope * axis))
        )

    fitted = least_squares(
        residual,
        np.asarray([initial_intercept, initial_slope]),
        loss="soft_l1",
        f_scale=0.25,
        max_nfev=80,
    )
    powered_intercept, powered_slope = map(float, fitted.x)
    carrier_phase = powered_intercept / order
    carrier_step = powered_slope / order
    rms = float(np.sqrt(np.mean(np.abs(values) ** 2)))
    corrected = values / max(rms, _EPSILON) * np.exp(
        -1j * (carrier_phase + carrier_step * axis)
    )
    decisions = np.argmin(
        np.abs(corrected[:, None] - reference[None, :]), axis=1
    )
    ideal = reference[decisions]
    cost = float(
        np.sqrt(
            np.sum(np.abs(corrected - ideal) ** 2)
            / max(np.sum(np.abs(ideal) ** 2), _EPSILON)
        )
    )
    return (
        _wrap_phase(carrier_phase),
        carrier_step,
        cost,
    )


def _fit_qam_carrier(
    symbols: np.ndarray,
    alphabet: np.ndarray,
) -> tuple[float, float, float]:
    """Fit square-QAM carrier phase/CFO without discarding symbol amplitude."""
    values = np.asarray(symbols, dtype=np.complex128)
    if values.size < 9:
        return 0.0, 0.0, np.inf
    rms = float(np.sqrt(np.mean(np.abs(values) ** 2)))
    normalized = values / max(rms, _EPSILON)
    axis = np.arange(normalized.size, dtype=np.float64)

    # Square QAM has fourfold rotational symmetry and a non-zero fourth
    # moment.  Its spectral peak provides a decision-independent coarse CFO.
    fourth = normalized**4
    fft_size = 1 << int(np.ceil(np.log2(max(64, 4 * fourth.size))))
    spectrum = np.fft.fft(fourth, n=fft_size)
    frequency = np.fft.fftfreq(fft_size)
    carrier_step = float(
        2.0 * np.pi * frequency[int(np.argmax(np.abs(spectrum)))] / 4.0
    )
    derotated_fourth = fourth * np.exp(-4j * carrier_step * axis)
    carrier_phase = float(np.angle(-np.sum(derotated_fourth)) / 4.0)

    # Refine against nearest ideal decisions.  The remaining 90-degree
    # ambiguity only permutes symbol labels and does not affect geometry.
    for _ in range(3):
        corrected = normalized * np.exp(
            -1j * (carrier_phase + carrier_step * axis)
        )
        decisions = np.argmin(
            np.abs(corrected[:, None] - alphabet[None, :]), axis=1
        )
        reference = alphabet[decisions]
        residual_phase = np.unwrap(np.angle(corrected * np.conj(reference)))
        slope_delta, intercept_delta = np.polyfit(axis, residual_phase, 1)
        carrier_phase += float(intercept_delta)
        carrier_step += float(slope_delta)

    corrected = normalized * np.exp(
        -1j * (carrier_phase + carrier_step * axis)
    )
    decisions = np.argmin(
        np.abs(corrected[:, None] - alphabet[None, :]), axis=1
    )
    reference = alphabet[decisions]
    evm = float(
        np.sqrt(
            np.sum(np.abs(corrected - reference) ** 2)
            / max(np.sum(np.abs(reference) ** 2), _EPSILON)
        )
    )
    return _wrap_phase(carrier_phase), carrier_step, evm


class PatternAnalyzer:
    """Search one known symbol pattern and decode a result range from its start."""

    def search(
        self,
        recording: IQRecording,
        signal: SignalDescription,
        search: PatternSearchSettings,
        result_range: ResultRangeSettings | None = None,
        demodulation: DemodulationSettings | None = None,
    ) -> PatternSearchResult:
        pattern = search.pattern
        result_range = result_range or ResultRangeSettings()
        demodulation = demodulation or DemodulationSettings()
        if search.mode is PatternSearchMode.OFF:
            raise RuntimeError("pattern search is disabled")
        if any(symbol >= signal.modulation.order for symbol in pattern.symbols):
            raise ValueError("known pattern contains a symbol outside the modulation order")
        if search.iq_power_trigger.enabled:
            return self._search_power_gated(
                recording, signal, search, result_range, demodulation
            )
        return self._search_ungated(
            recording, signal, search, result_range, demodulation
        )

    def detect_data(
        self,
        recording: IQRecording,
        signal: SignalDescription,
        search: PatternSearchSettings | None = None,
        result_range: ResultRangeSettings | None = None,
        demodulation: DemodulationSettings | None = None,
        iq_power_trigger: IQPowerTriggerSettings | None = None,
    ) -> PatternSearchResult:
        """Synchronize PSK from detected decisions without known symbols.

        This is the R&S-style fallback used when a configured pattern is not
        present.  It deliberately remains separate from ``search`` so a blind
        synchronization result can never be counted as a pattern match.
        """
        if not signal.modulation.family.uses_iq_constellation:
            raise ValueError(
                "detected-data synchronization currently supports PSK/QAM"
            )
        return self._detect_psk_data(
            recording,
            signal,
            search,
            result_range or ResultRangeSettings(),
            demodulation or DemodulationSettings(),
            iq_power_trigger=(
                iq_power_trigger
                if iq_power_trigger is not None
                else (
                    search.iq_power_trigger
                    if search is not None
                    else IQPowerTriggerSettings()
                )
            ),
        )

    def _detect_psk_data(
        self,
        recording: IQRecording,
        signal: SignalDescription,
        search: PatternSearchSettings | None,
        result_range: ResultRangeSettings,
        demodulation: DemodulationSettings,
        iq_power_trigger: IQPowerTriggerSettings,
    ) -> PatternSearchResult:
        samples_per_symbol = 8
        resampled, analysis_rate_hz = prepare_psk_iq(
            recording.iq,
            sample_rate_hz=recording.sample_rate_hz,
            symbol_rate_hz=signal.symbol_rate_hz,
            tx_filter=signal.tx_filter,
            filter_parameter=signal.filter_parameter,
            samples_per_symbol=samples_per_symbol,
            apply_measurement_filter=(
                demodulation.measurement_filter is MeasurementFilterMode.AUTO
            ),
        )
        start_sample = 0
        stop_sample = recording.sample_count
        trigger_event_index: int | None = None
        trigger_event_count = 0
        trigger_sample: int | None = None
        trigger_active_stop_sample: int | None = None
        if iq_power_trigger.enabled:
            events = detect_iq_power_trigger_events(
                recording,
                symbol_rate_hz=signal.symbol_rate_hz,
                settings=iq_power_trigger,
            )
            if not events:
                raise ValueError(
                    "detected-data synchronization found no I/Q power trigger event"
                )
            trigger_event_count = len(events)
            event = events[0]
            trigger_event_index = 1
            trigger_sample = int(event.trigger_sample)
            trigger_active_stop_sample = int(event.active_stop_sample)
            offset = int(
                round(
                    iq_power_trigger.search_start_offset_symbols
                    * recording.sample_rate_hz
                    / signal.symbol_rate_hz
                )
            )
            start_sample = int(np.clip(event.trigger_sample + offset, 0, stop_sample))
            if iq_power_trigger.limit_result_to_active_interval:
                stop_sample = min(stop_sample, int(event.active_stop_sample))

        scale = analysis_rate_hz / recording.sample_rate_hz
        start_resampled = float(start_sample) * scale
        stop_resampled = float(stop_sample) * scale
        alphabet = _constellation(signal.modulation, signal.symbol_mapping)
        order = int(alphabet.size)
        carrier_symmetry_order = _psk_carrier_symmetry_order(signal.modulation)

        def evaluate(offset: float, *, return_values: bool = False):
            centers = np.arange(
                offset + samples_per_symbol / 2.0 - 0.5,
                resampled.size,
                samples_per_symbol,
                dtype=np.float64,
            )
            centers = centers[
                (centers >= start_resampled + samples_per_symbol / 2.0 - 0.5)
                & (centers < stop_resampled - samples_per_symbol / 2.0 + 0.5)
                & (centers <= resampled.size - 1)
            ]
            absolute = _interpolate_complex(resampled, centers)
            if absolute.size < 9:
                return (np.inf, None) if return_values else np.inf
            if signal.modulation.differential:
                observed = absolute[1:] * np.conj(absolute[:-1])
                result_centers = centers[1:]
                decision_absolute = absolute[1:]
            else:
                observed = absolute
                result_centers = centers
                decision_absolute = absolute
            is_qam = signal.modulation is ModulationKind.QAM16
            if is_qam:
                # QAM amplitude is part of the decision geometry.  Do not use
                # the PSK phase-concentration gate, which can mistake valid
                # QAM phase populations for a short PSK-like interval.
                observed_rms = float(np.sqrt(np.mean(np.abs(observed) ** 2)))
                normalized = observed / max(observed_rms, _EPSILON)
                interval_start, interval_stop, interval_concentration = (
                    _detected_qam_decision_interval(observed)
                )
            else:
                # A differential PSK product has magnitude |s[n]||s[n-1]|;
                # remove it so timing is directed by phase geometry.
                normalized = observed / np.maximum(np.abs(observed), _EPSILON)
                interval_start, interval_stop, interval_concentration = (
                    _detected_psk_decision_interval(
                        normalized, carrier_symmetry_order
                    )
                )
            normalized = normalized[interval_start:interval_stop]
            result_centers = result_centers[interval_start:interval_stop]
            decision_absolute = decision_absolute[interval_start:interval_stop]
            if normalized.size < 9:
                return (np.inf, None) if return_values else np.inf

            if is_qam:
                physical_phase, carrier_step, cost = _fit_qam_carrier(
                    decision_absolute, alphabet
                )
                if return_values:
                    return cost, (
                        decision_absolute,
                        normalized,
                        result_centers,
                        physical_phase,
                        carrier_step,
                        interval_start,
                        interval_stop,
                        interval_concentration,
                    )
                return cost

            # Synchronize from the physical IQ symmetry, not from the assumed
            # differential decision alphabet.  In particular pi/4-DQPSK has
            # four phase increments but an eight-state physical constellation.
            # Keeping this fit decision-independent lets a deliberately wrong
            # modulation hypothesis still produce a stable physical plot; its
            # poor decision EVM then exposes the mismatch honestly.
            absolute_unit = decision_absolute / np.maximum(
                np.abs(decision_absolute), _EPSILON
            )
            powered = absolute_unit ** carrier_symmetry_order
            symbol_axis = np.arange(powered.size, dtype=np.float64)
            unwrapped = np.unwrap(np.angle(powered))
            initial_slope, initial_intercept = np.polyfit(
                symbol_axis, unwrapped, 1
            )

            def carrier_residual(parameters: np.ndarray) -> np.ndarray:
                intercept, slope = parameters
                return np.angle(
                    powered
                    * np.exp(-1j * (intercept + slope * symbol_axis))
                )

            fitted = least_squares(
                carrier_residual,
                np.asarray([initial_intercept, initial_slope]),
                loss="soft_l1",
                f_scale=0.25,
                max_nfev=80,
            )
            powered_intercept, powered_slope = map(float, fitted.x)
            residual = carrier_residual(fitted.x)
            physical_phase = powered_intercept / float(carrier_symmetry_order)
            carrier_step = powered_slope / float(carrier_symmetry_order)
            cost = float(
                np.sqrt(np.mean(residual**2))
                / float(carrier_symmetry_order)
            )
            if return_values:
                return cost, (
                    decision_absolute,
                    normalized,
                    result_centers,
                    physical_phase,
                    carrier_step,
                    interval_start,
                    interval_stop,
                    interval_concentration,
                )
            return cost

        integer_costs = np.asarray(
            [evaluate(float(phase)) for phase in range(samples_per_symbol)]
        )
        best_phase = int(np.argmin(integer_costs))
        if not np.isfinite(integer_costs[best_phase]):
            raise ValueError("not enough active PSK symbols for detected-data synchronization")
        fractional_phase = float(best_phase)
        if demodulation.measurement_filter is MeasurementFilterMode.AUTO:
            refined = minimize_scalar(
                lambda value: evaluate(float(value % samples_per_symbol)),
                bounds=(best_phase - 1.0, best_phase + 1.0),
                method="bounded",
                options={"xatol": 1e-4},
            )
            if float(refined.fun) < float(integer_costs[best_phase]):
                fractional_phase = float(refined.x % samples_per_symbol)
        physical_sync_rms_rad, values = evaluate(
            fractional_phase, return_values=True
        )
        assert values is not None
        (
            decision_absolute,
            normalized,
            result_centers,
            physical_phase,
            carrier_step,
            detected_interval_start,
            detected_interval_stop,
            detected_interval_concentration,
        ) = values
        if signal.modulation.differential:
            corrected = normalized * np.exp(-1j * carrier_step)
        else:
            decision_axis = np.arange(normalized.size, dtype=np.float64)
            corrected = normalized * np.exp(
                -1j * (physical_phase + carrier_step * decision_axis)
            )
        decoded_all = np.argmin(
            np.abs(corrected[:, None] - alphabet[None, :]), axis=1
        ).astype(np.int16)

        offset_symbols = max(0, int(result_range.offset_symbols))
        stop_symbols = min(
            decoded_all.size,
            offset_symbols + int(result_range.result_length),
        )
        selection = slice(offset_symbols, stop_symbols)
        corrected = corrected[selection]
        decoded = decoded_all[selection]
        result_centers = result_centers[selection]
        if corrected.size == 0:
            raise ValueError("detected-data Result Range contains no symbols")
        reference = alphabet[decoded]
        evm_rms_percent = 100.0 * float(
            np.sqrt(np.sum(np.abs(corrected - reference) ** 2) / np.sum(np.abs(reference) ** 2))
        )

        carrier_phase = _wrap_phase(
            physical_phase + carrier_step * float(offset_symbols)
        )
        half_symbol_s = 0.5 / signal.symbol_rate_hz
        result_start_sample = max(
            0,
            int(round((result_centers[0] / analysis_rate_hz - half_symbol_s) * recording.sample_rate_hz)),
        )
        result_stop_sample = min(
            recording.sample_count,
            int(round((result_centers[-1] / analysis_rate_hz + half_symbol_s) * recording.sample_rate_hz)),
        )
        cfo_hz = carrier_step * signal.symbol_rate_hz / (2.0 * np.pi)
        return PatternSearchResult(
            modulation=signal.modulation,
            pattern_start_sample=result_start_sample,
            pattern_start_time_s=result_start_sample / recording.sample_rate_hz,
            pattern_start_symbol=0,
            result_start_sample=result_start_sample,
            result_stop_sample=result_stop_sample,
            correlation=0.0,
            pattern_symbol_errors=1,
            decoded_symbols=decoded,
            decoded_bits=_symbols_to_bits(decoded, order, demodulation.bit_ordering),
            measured_symbols=np.asarray(corrected, dtype=np.complex64),
            symbol_time_s=result_centers / analysis_rate_hz,
            carrier_frequency_offset_hz=cfo_hz,
            carrier_frequency_drift_hz_per_s=0.0,
            frequency_deviation_hz=None,
            evm_rms_percent=evm_rms_percent,
            polarity_inverted=False,
            phase_rotation_rad=_wrap_phase(carrier_phase),
            timing_phase_samples=best_phase,
            analysis_sample_rate_hz=analysis_rate_hz,
            recording_sample_rate_hz=recording.sample_rate_hz,
            carrier_reference_time_s=float(result_centers[0]) / analysis_rate_hz,
            metadata={
                "pattern_name": (
                    search.pattern.name if search is not None else "Detected Data"
                ),
                "pattern_symbol_count": 0,
                "symbol_rate_hz": signal.symbol_rate_hz,
                "synchronization_source": SynchronizationSource.DETECTED_DATA.value,
                "pattern_match_valid": False,
                "measurement_filter": demodulation.measurement_filter.value,
                "matched_filter_applied": (
                    demodulation.measurement_filter is MeasurementFilterMode.AUTO
                    and signal.tx_filter.lower() in {
                        "root raised cosine", "root-raised-cosine", "rrc", "srrc"
                    }
                ),
                "fractional_timing_offset_samples": fractional_phase - best_phase,
                "symbol_rate_error_ppm": 0.0,
                "synchronization_evm_rms": evm_rms_percent / 100.0,
                # Absolute IQ has an M-fold phase ambiguity without known
                # symbols.  Do not report a misleading physical EVM; the
                # decision-directed differential EVM remains well-defined.
                "physical_evm_rms_percent": None,
                "differential_symbol_evm_rms_percent": (
                    evm_rms_percent if signal.modulation.differential else None
                ),
                "bluetooth_devm_rms_percent": None,
                "carrier_drift_compensated": False,
                "phase_estimation_method": (
                    "blind physical-symmetry carrier and timing synchronization"
                ),
                "carrier_symmetry_order": carrier_symmetry_order,
                "physical_sync_rms_rad": physical_sync_rms_rad,
                "detected_psk_interval_start_symbol": detected_interval_start,
                "detected_psk_interval_stop_symbol": detected_interval_stop,
                "detected_psk_interval_concentration": (
                    detected_interval_concentration
                ),
                "selected_match_index": 1,
                "eligible_match_count": 1,
                "detected_match_count": 0,
                "power_trigger_enabled": iq_power_trigger.enabled,
                "power_trigger_level_dbm": iq_power_trigger.level_dbm,
                "power_trigger_sample": trigger_sample,
                "power_trigger_active_stop_sample": trigger_active_stop_sample,
                "power_trigger_search_start_sample": (
                    start_sample if iq_power_trigger.enabled else None
                ),
                "power_trigger_search_offset_symbols": (
                    iq_power_trigger.search_start_offset_symbols
                ),
                "power_trigger_envelope_average_symbols": (
                    iq_power_trigger.envelope_average_symbols
                ),
                "power_trigger_limit_result_to_active_interval": (
                    iq_power_trigger.limit_result_to_active_interval
                ),
                "selected_power_trigger_event_index": trigger_event_index,
                "power_trigger_event_count": trigger_event_count,
                "source": recording.source,
            },
        )

    def _search_ungated(
        self,
        recording: IQRecording,
        signal: SignalDescription,
        search: PatternSearchSettings,
        result_range: ResultRangeSettings,
        demodulation: DemodulationSettings,
    ) -> PatternSearchResult:
        if signal.modulation.family is ModulationFamily.FSK:
            return self._search_fsk(
                recording, signal, search, result_range, demodulation
            )
        return self._search_psk(recording, signal, search, result_range, demodulation)

    def _search_power_gated(
        self,
        recording: IQRecording,
        signal: SignalDescription,
        search: PatternSearchSettings,
        result_range: ResultRangeSettings,
        demodulation: DemodulationSettings,
    ) -> PatternSearchResult:
        trigger = search.iq_power_trigger
        events = detect_iq_power_trigger_events(
            recording,
            symbol_rate_hz=signal.symbol_rate_hz,
            settings=trigger,
        )
        if not events:
            raise ValueError(
                f"no I/Q power trigger event exceeded {trigger.level_dbm:.2f} dBm"
            )

        samples_per_symbol = recording.sample_rate_hz / signal.symbol_rate_hz
        offset_samples = int(
            round(trigger.search_start_offset_symbols * samples_per_symbol)
        )
        local_search = replace(
            search,
            match_selection=MatchSelectionPolicy.FIRST,
            match_index=1,
            iq_power_trigger=replace(trigger, enabled=False),
        )
        matches: list[tuple[int, PatternSearchResult]] = []
        for event_index, event in enumerate(events, start=1):
            search_start = int(
                np.clip(event.trigger_sample + offset_samples, 0, recording.sample_count)
            )
            next_trigger = (
                events[event_index].trigger_sample
                if event_index < len(events)
                else recording.sample_count
            )
            search_stop = max(search_start, int(next_trigger))
            if trigger.limit_result_to_active_interval:
                # Limit the waveform before demodulation.  Trimming only the
                # returned arrays afterwards leaves PSK RMS normalization and
                # EVM based on the requested Result Length (including the
                # inactive tail) even though Result Symbols reports the
                # shorter trigger-limited count.
                search_stop = min(search_stop, int(event.active_stop_sample))
            if search_start >= search_stop or search_start >= event.active_stop_sample:
                continue
            local_recording = replace(
                recording,
                iq=recording.iq[search_start:search_stop],
                start_sample_index=recording.start_sample_index + search_start,
                trigger_sample_index=None,
                source=f"{recording.source} | I/Q Power Trigger {event_index}",
            )
            try:
                local_result = self._search_ungated(
                    local_recording,
                    signal,
                    local_search,
                    result_range,
                    demodulation,
                )
            except ValueError:
                continue
            global_pattern_start = search_start + local_result.pattern_start_sample
            if global_pattern_start >= event.active_stop_sample:
                # The first eligible match occurred in the following no-signal
                # interval, so this trigger did not contain a valid pattern.
                continue
            time_offset_s = search_start / recording.sample_rate_hz
            metadata = {
                **dict(local_result.metadata),
                "power_trigger_enabled": True,
                "power_trigger_level_dbm": trigger.level_dbm,
                "power_trigger_event_index": event_index,
                "power_trigger_sample": event.trigger_sample,
                "power_trigger_active_stop_sample": event.active_stop_sample,
                "power_trigger_search_start_sample": search_start,
                "power_trigger_search_offset_symbols": (
                    trigger.search_start_offset_symbols
                ),
                "power_trigger_envelope_average_symbols": (
                    trigger.envelope_average_symbols
                ),
                "power_trigger_limit_result_to_active_interval": (
                    trigger.limit_result_to_active_interval
                ),
            }
            decoded_symbols = local_result.decoded_symbols
            decoded_bits = local_result.decoded_bits
            measured_symbols = local_result.measured_symbols
            symbol_time_s = local_result.symbol_time_s + time_offset_s
            global_result_stop = search_start + local_result.result_stop_sample
            if trigger.limit_result_to_active_interval:
                last_complete_center_s = (
                    event.active_stop_sample / recording.sample_rate_hz
                    - 0.5 / signal.symbol_rate_hz
                )
                complete_count = int(
                    np.searchsorted(
                        symbol_time_s,
                        last_complete_center_s,
                        side="right",
                    )
                )
                if complete_count <= 0:
                    continue
                decoded_symbols = decoded_symbols[:complete_count]
                measured_symbols = measured_symbols[:complete_count]
                symbol_time_s = symbol_time_s[:complete_count]
                decoded_bits = _symbols_to_bits(
                    decoded_symbols,
                    signal.modulation.order,
                    demodulation.bit_ordering,
                )
                global_result_stop = min(
                    global_result_stop,
                    event.active_stop_sample,
                )
                metadata["burst_limited_symbol_count"] = complete_count
            matches.append(
                (
                    event_index,
                    replace(
                        local_result,
                        pattern_start_sample=global_pattern_start,
                        pattern_start_time_s=(
                            local_result.pattern_start_time_s + time_offset_s
                        ),
                        pattern_start_symbol=(
                            local_result.pattern_start_symbol
                            + int(round(search_start / samples_per_symbol))
                        ),
                        result_start_sample=(
                            search_start + local_result.result_start_sample
                        ),
                        result_stop_sample=global_result_stop,
                        decoded_symbols=decoded_symbols,
                        decoded_bits=decoded_bits,
                        measured_symbols=measured_symbols,
                        symbol_time_s=symbol_time_s,
                        carrier_reference_time_s=(
                            local_result.carrier_reference_time_s + time_offset_s
                        ),
                        recording_sample_rate_hz=recording.sample_rate_hz,
                        metadata=metadata,
                    ),
                )
            )

        if not matches:
            raise ValueError(
                f"no valid pattern was found in {len(events)} I/Q power trigger event(s)"
            )
        requested = int(search.match_index)
        if requested > len(matches):
            raise ValueError(
                f"Match Index {requested} is unavailable; "
                f"{len(matches)} triggered match(es) were found"
            )
        event_index, selected = matches[requested - 1]
        return replace(
            selected,
            metadata={
                **dict(selected.metadata),
                "match_selection_policy": search.match_selection.value,
                "selected_match_index": requested,
                "eligible_match_count": len(matches),
                "detected_match_count": len(matches),
                "eligible_match_start_samples": tuple(
                    int(item.pattern_start_sample) for _, item in matches
                ),
                "power_trigger_event_count": len(events),
                "power_trigger_matched_event_count": len(matches),
                "selected_power_trigger_event_index": event_index,
            },
        )

    def _search_fsk(
        self,
        recording: IQRecording,
        signal: SignalDescription,
        search: PatternSearchSettings,
        result_range: ResultRangeSettings,
        demodulation_settings: DemodulationSettings,
    ) -> PatternSearchResult:
        pattern = search.pattern
        if signal.symbol_mapping != "Natural":
            raise ValueError(
                f"unsupported FSK modulation mapping: {signal.symbol_mapping}"
            )
        if len(pattern.symbols) < 8:
            raise ValueError("FSK known pattern must contain at least eight bits")
        bits = np.asarray(pattern.symbols, dtype=np.uint8)
        gaussian_bt = (
            signal.filter_parameter
            if signal.tx_filter.lower() == "gaussian"
            else None
        )
        if result_range.alignment is ResultRangeAlignment.LEFT:
            result_start = int(result_range.offset_symbols)
        elif result_range.alignment is ResultRangeAlignment.CENTER:
            result_start = len(pattern.symbols) // 2 - int(result_range.result_length) // 2
            result_start += int(result_range.offset_symbols)
        else:
            result_start = len(pattern.symbols) - int(result_range.result_length)
            result_start += int(result_range.offset_symbols)
        maximum_symbols = max(
            len(pattern.symbols),
            result_start + int(result_range.result_length),
        )
        demodulation = demodulate_gfsk(
            recording.iq,
            sample_rate_hz=recording.sample_rate_hz,
            access_bits=bits,
            symbol_rate_hz=signal.symbol_rate_hz,
            minimum_correlation=search.effective_correlation_threshold,
            gaussian_bt=gaussian_bt,
            apply_measurement_filter=(
                demodulation_settings.measurement_filter
                is MeasurementFilterMode.AUTO
            ),
            maximum_symbols=maximum_symbols,
            match_selection=search.match_selection.value,
            match_index=search.match_index,
            required_result_symbols=maximum_symbols,
            exclude_incomplete_result=result_range.exclude_incomplete_result,
            require_zero_pattern_errors=(
                search.meas_only_if_pattern_symbols_correct
            ),
            allow_polarity_inversion=False,
            allow_complemented_pattern_match=(
                search.allow_inverted_fsk_pattern
            ),
        )
        selection = _result_slice(len(pattern.symbols), demodulation.bits.size, result_range)
        decoded = demodulation.bits[selection].astype(np.int16)
        measured_frequency = (
            demodulation.drift_compensated_symbol_frequency_hz
            if demodulation_settings.compensate_carrier_frequency_drift
            else demodulation.symbol_frequency_hz
        )
        measured = measured_frequency[selection].astype(np.complex64)
        times = demodulation.symbol_time_s[selection]
        half_symbol_s = 0.5 / signal.symbol_rate_hz
        result_start_sample = max(
            0,
            int(round((float(times[0]) - half_symbol_s) * recording.sample_rate_hz)),
        )
        result_stop_sample = min(
            recording.sample_count,
            int(round((float(times[-1]) + half_symbol_s) * recording.sample_rate_hz)),
        )
        refined_pattern_start_time_s = float(times[0]) - half_symbol_s
        return PatternSearchResult(
            modulation=signal.modulation,
            pattern_start_sample=demodulation.access_start_sample,
            pattern_start_time_s=refined_pattern_start_time_s,
            pattern_start_symbol=demodulation.access_start_bit,
            result_start_sample=result_start_sample,
            result_stop_sample=result_stop_sample,
            correlation=demodulation.access_correlation,
            pattern_symbol_errors=demodulation.access_bit_errors,
            decoded_symbols=decoded,
            decoded_bits=_symbols_to_bits(
                decoded, signal.modulation.order, demodulation_settings.bit_ordering
            ),
            measured_symbols=measured,
            symbol_time_s=times,
            carrier_frequency_offset_hz=demodulation.carrier_frequency_offset_hz,
            carrier_frequency_drift_hz_per_s=(
                demodulation.carrier_frequency_drift_hz_per_s
            ),
            frequency_deviation_hz=demodulation.frequency_deviation_hz,
            evm_rms_percent=None,
            polarity_inverted=demodulation.complemented_pattern_match,
            phase_rotation_rad=None,
            timing_phase_samples=demodulation.timing_phase_samples,
            analysis_sample_rate_hz=demodulation.analysis_sample_rate_hz,
            recording_sample_rate_hz=recording.sample_rate_hz,
            carrier_reference_time_s=(
                refined_pattern_start_time_s
                + len(pattern.symbols) / (2.0 * signal.symbol_rate_hz)
            ),
            metadata={
                "pattern_name": pattern.name,
                "pattern_symbol_count": len(pattern.symbols),
                "pattern_match_variant": (
                    "Inverted"
                    if demodulation.complemented_pattern_match
                    else "Normal"
                ),
                "matched_pattern_symbols": (
                    [1 - int(symbol) for symbol in pattern.symbols]
                    if demodulation.complemented_pattern_match
                    else [int(symbol) for symbol in pattern.symbols]
                ),
                "symbol_rate_hz": signal.symbol_rate_hz,
                "result_length": result_range.result_length,
                "result_offset_symbols": result_range.offset_symbols,
                "gaussian_bt": gaussian_bt,
                "source": recording.source,
                "match_selection_policy": search.match_selection.value,
                "selected_match_index": demodulation.selected_match_index,
                "eligible_match_count": demodulation.eligible_match_count,
                "detected_match_count": demodulation.detected_match_count,
                "eligible_match_start_samples": (
                    demodulation.eligible_match_start_samples
                ),
                "exclude_incomplete_result": (
                    result_range.exclude_incomplete_result
                ),
                "fractional_timing_offset_samples": (
                    demodulation.frequency_model_timing_offset_samples
                ),
                "fractional_timing_offset_symbols": (
                    demodulation.frequency_model_timing_offset_samples
                    / demodulation.samples_per_symbol
                ),
                "applied_timing_offset_samples": (
                    demodulation.applied_timing_offset_samples
                ),
                "timing_correction_accepted": (
                    demodulation.timing_correction_accepted
                ),
                "frequency_model_residual_rms_hz": (
                    demodulation.frequency_model_residual_rms_hz
                ),
                "frequency_model_no_drift_residual_rms_hz": (
                    demodulation.frequency_model_no_drift_residual_rms_hz
                ),
                "drift_model_accepted": demodulation.drift_model_accepted,
                "candidate_drift_hz_per_s": (
                    demodulation.candidate_drift_hz_per_s
                ),
                "drift_model_residual_rms_hz": (
                    demodulation.drift_model_residual_rms_hz
                ),
                "drift_excursion_hz": demodulation.drift_excursion_hz,
                "drift_bic_improvement": (
                    demodulation.drift_bic_improvement
                ),
                "drift_rejection_reason": (
                    demodulation.drift_rejection_reason
                ),
                "timing_confidence": demodulation.timing_confidence,
                "estimation_sample_count": (
                    demodulation.estimation_sample_count
                ),
                "fsk_measurement_filter": (
                    "Gaussian Auto"
                    if gaussian_bt is not None
                    and demodulation_settings.measurement_filter
                    is MeasurementFilterMode.AUTO
                    else "None"
                ),
                "frequency_deviation_error_percent": (
                    100.0
                    * (
                        demodulation.frequency_deviation_hz
                        / signal.frequency_deviation_hz
                        - 1.0
                    )
                    if signal.frequency_deviation_hz is not None
                    and signal.frequency_deviation_hz > 0.0
                    else None
                ),
            },
        )

    def _search_psk(
        self,
        recording: IQRecording,
        signal: SignalDescription,
        search: PatternSearchSettings,
        result_range: ResultRangeSettings,
        demodulation: DemodulationSettings,
    ) -> PatternSearchResult:
        preliminary = self._search_psk_pass(
            recording,
            signal,
            search,
            result_range,
            demodulation,
            prefilter_carrier_frequency_offset_hz=0.0,
        )
        if not preliminary.metadata["matched_filter_applied"]:
            return preliminary

        coarse_cfo_hz = float(preliminary.carrier_frequency_offset_hz)
        if abs(coarse_cfo_hz) <= np.finfo(np.float64).eps:
            return replace(
                preliminary,
                metadata={
                    **dict(preliminary.metadata),
                    "prefilter_cfo_correction_applied": False,
                    "prefilter_coarse_cfo_hz": 0.0,
                    "postfilter_residual_cfo_hz": coarse_cfo_hz,
                },
            )

        refined = self._search_psk_pass(
            recording,
            signal,
            search,
            result_range,
            demodulation,
            prefilter_carrier_frequency_offset_hz=coarse_cfo_hz,
        )
        residual_cfo_hz = float(refined.carrier_frequency_offset_hz)
        total_cfo_hz = coarse_cfo_hz + residual_cfo_hz
        total_phase_rotation = refined.phase_rotation_rad
        if total_phase_rotation is not None:
            total_phase_rotation = _wrap_phase(
                total_phase_rotation
                + 2.0
                * np.pi
                * coarse_cfo_hz
                * refined.carrier_reference_time_s
            )
        return replace(
            refined,
            carrier_frequency_offset_hz=total_cfo_hz,
            phase_rotation_rad=total_phase_rotation,
            metadata={
                **dict(refined.metadata),
                "prefilter_cfo_correction_applied": True,
                "prefilter_coarse_cfo_hz": coarse_cfo_hz,
                "postfilter_residual_cfo_hz": residual_cfo_hz,
                "carrier_recovery_stages": (
                    "coarse CFO -> carrier centering -> matched filter -> fine synchronization"
                ),
            },
        )

    def _search_psk_pass(
        self,
        recording: IQRecording,
        signal: SignalDescription,
        search: PatternSearchSettings,
        result_range: ResultRangeSettings,
        demodulation: DemodulationSettings,
        *,
        prefilter_carrier_frequency_offset_hz: float,
    ) -> PatternSearchResult:
        pattern = search.pattern
        samples_per_symbol = 8
        resampled, analysis_rate_hz = prepare_psk_iq(
            recording.iq,
            sample_rate_hz=recording.sample_rate_hz,
            symbol_rate_hz=signal.symbol_rate_hz,
            tx_filter=signal.tx_filter,
            filter_parameter=signal.filter_parameter,
            samples_per_symbol=samples_per_symbol,
            prefilter_carrier_frequency_offset_hz=(
                prefilter_carrier_frequency_offset_hz
            ),
            apply_measurement_filter=(
                demodulation.measurement_filter is MeasurementFilterMode.AUTO
            ),
        )
        matched_filter_applied = (
            demodulation.measurement_filter is MeasurementFilterMode.AUTO
            and signal.tx_filter.lower() in {
                "root raised cosine",
                "root-raised-cosine",
                "rrc",
                "srrc",
            }
        )
        alphabet = _constellation(signal.modulation, signal.symbol_mapping)
        configured_pattern_symbols = np.asarray(pattern.symbols, dtype=np.int16)
        canonical_pattern_symbols = (
            reverse_symbol_bits(configured_pattern_symbols, signal.modulation.order)
            if demodulation.bit_ordering is BitOrdering.LSB
            else configured_pattern_symbols
        )
        expected = alphabet[canonical_pattern_symbols]
        candidates: list[
            tuple[float, int, int, np.ndarray, np.ndarray, float]
        ] = []
        observed_score = 0.0
        for phase in range(samples_per_symbol):
            centers = np.arange(
                phase + samples_per_symbol / 2.0 - 0.5,
                resampled.size,
                samples_per_symbol,
                dtype=np.float64,
            )
            centers = centers[centers <= resampled.size - 1]
            if centers.size < len(pattern.symbols):
                continue
            if signal.modulation is ModulationKind.QAM16:
                indices = np.floor(centers + 0.5).astype(np.int64)
            else:
                indices = np.rint(centers).astype(np.int64)
            indices = np.clip(indices, 0, resampled.size - 1)
            sampled_centers = (
                centers
                if signal.modulation.differential
                else indices.astype(np.float64)
            )
            waveform_symbols = resampled[indices]
            if signal.modulation.differential:
                observed = waveform_symbols[1:] * np.conj(waveform_symbols[:-1])
                scores = _normalized_complex_correlation(observed, expected)
            else:
                observed_difference = waveform_symbols[1:] * np.conj(
                    waveform_symbols[:-1]
                )
                expected_difference = expected[1:] * np.conj(expected[:-1])
                scores = _normalized_complex_correlation(
                    observed_difference, expected_difference
                )
            if scores.size == 0:
                continue
            observed_score = max(observed_score, float(np.max(scores)))
            for index in _local_peak_indices(
                scores, search.effective_correlation_threshold
            ):
                symbol_offset = 1 if signal.modulation.differential else 0
                start_coordinate = float(
                    sampled_centers[int(index) + symbol_offset]
                )
                candidates.append(
                    (
                        float(scores[index]),
                        phase,
                        int(index),
                        waveform_symbols,
                        sampled_centers,
                        start_coordinate,
                    )
                )
        if not candidates:
            raise ValueError(
                f"known pattern was not found (correlation={observed_score:.3f})"
            )

        # Collapse the same physical packet detected at adjacent timing phases.
        # Retain the best correlation and prefer the earlier phase for a tie.
        physical_candidates: list[
            tuple[float, int, int, np.ndarray, np.ndarray, float]
        ] = []
        for candidate in sorted(candidates, key=lambda item: item[5]):
            duplicate_index = next(
                (
                    position
                    for position, existing in enumerate(physical_candidates)
                    if abs(existing[5] - candidate[5])
                    < 0.99 * samples_per_symbol
                ),
                None,
            )
            if duplicate_index is None:
                physical_candidates.append(candidate)
                continue
            existing = physical_candidates[duplicate_index]
            if (
                candidate[0] > existing[0] + 1e-10
                or (
                    abs(candidate[0] - existing[0]) <= 1e-10
                    and candidate[1] < existing[1]
                )
            ):
                physical_candidates[duplicate_index] = candidate

        detected_match_count = len(physical_candidates)

        def candidate_pattern_symbol_errors(candidate: tuple) -> int:
            _, _, candidate_index, candidate_symbols, _, _ = candidate
            if signal.modulation.differential:
                observed = candidate_symbols[1:] * np.conj(candidate_symbols[:-1])
                training = observed[
                    candidate_index : candidate_index + len(pattern.symbols)
                ]
            else:
                training = candidate_symbols[
                    candidate_index : candidate_index + len(pattern.symbols)
                ]
            if training.size != len(pattern.symbols):
                return len(pattern.symbols)
            relative = np.arange(training.size, dtype=np.float64)
            phase_error = np.unwrap(np.angle(training * np.conj(expected)))
            slope, intercept = np.polyfit(relative, phase_error, 1)
            corrected = training * np.exp(-1j * (intercept + slope * relative))
            rms = float(np.sqrt(np.mean(np.abs(corrected) ** 2)))
            normalized = corrected / max(rms, _EPSILON)
            decisions = np.argmin(
                np.abs(normalized[:, None] - alphabet[None, :]), axis=1
            ).astype(np.int16)
            return int(
                np.count_nonzero(
                    decisions != canonical_pattern_symbols
                )
            )

        eligible_candidates = []
        for candidate in physical_candidates:
            _, _, candidate_index, candidate_symbols, _, _ = candidate
            available = candidate_symbols.size - candidate_index
            if signal.modulation.differential:
                available -= 1
            if (
                not result_range.exclude_incomplete_result
                or _result_is_complete(
                    len(pattern.symbols), available, result_range
                )
            ) and (
                not search.meas_only_if_pattern_symbols_correct
                or candidate_pattern_symbol_errors(candidate) == 0
            ):
                eligible_candidates.append(candidate)
        if (
            search.meas_only_if_pattern_symbols_correct
            and not eligible_candidates
        ):
            raise ValueError(
                "no symbol-correct pattern match satisfies the search requirements"
            )
        best, selected_match_index, eligible_match_count = _select_match_candidate(
            eligible_candidates,
            search.match_selection,
            search.match_index,
            time_key=lambda item: item[5],
            score_key=lambda item: item[0],
        )
        score, phase, index, waveform_symbols, centers, _ = best
        eligible_match_start_samples = tuple(
            int(
                round(
                    float(candidate[5])
                    / analysis_rate_hz
                    * recording.sample_rate_hz
                )
            )
            for candidate in sorted(eligible_candidates, key=lambda item: item[5])
        )
        phase_model_residual_rms_rad: float | None = None
        phase_drift_estimate_accepted: bool | None = None
        phase_no_drift_residual_rms_rad: float | None = None
        fractional_timing_offset_samples = 0.0
        symbol_timing_rate_samples_per_symbol = 0.0
        synchronization_evm_rms: float | None = None
        physical_evm_rms_percent: float | None = None
        bluetooth_devm_rms_percent: float | None = None
        if signal.modulation.differential:
            base_centers = np.asarray(centers, dtype=np.float64)
            base_observed = waveform_symbols[1:] * np.conj(waveform_symbols[:-1])
            base_available = base_observed[index:]
            selection = _result_slice(
                len(pattern.symbols), base_available.size, result_range
            )
            all_relative = np.arange(base_available.size, dtype=np.float64)
            selected_relative = all_relative[selection]
            fit_window = base_available[: len(pattern.symbols)]
            pattern_error = fit_window * np.conj(expected)
            pattern_phase_anchor = float(
                np.angle(np.sum(pattern_error * np.abs(fit_window)))
            )
            phase_model = _fit_differential_psk_phase_model(
                base_available[selection],
                selected_relative,
                alphabet,
                pattern_phase_anchor_rad=pattern_phase_anchor,
                pattern_center_symbol=(len(pattern.symbols) - 1.0) / 2.0,
            )
            (
                intercept,
                slope,
                phase_model_residual_rms_rad,
                phase_drift_estimate_accepted,
                phase_no_drift_residual_rms_rad,
            ) = phase_model

            result_symbol_count = selected_relative.size
            timing_anchor = float(index + 1) + float(
                np.mean(selected_relative)
            )
            timing_axis = np.arange(base_centers.size, dtype=np.float64) - timing_anchor
            maximum_timing_rate = 1.0 / max(result_symbol_count - 1, 1)
            maximum_phase_slope = np.pi / (
                float(alphabet.size) * max(result_symbol_count - 1, 1)
            )
            slope = float(
                np.clip(slope, -maximum_phase_slope, maximum_phase_slope)
            )
            initial_differential = base_available[selection] * np.exp(
                -1j * (intercept + slope * selected_relative)
            )
            initial_differential_rms = float(
                np.sqrt(np.mean(np.abs(initial_differential) ** 2))
            )
            normalized_differential = initial_differential / max(
                initial_differential_rms, _EPSILON
            )
            reference_decisions = alphabet[
                np.argmin(
                    np.abs(normalized_differential[:, None] - alphabet[None, :]),
                    axis=1,
                )
            ]
            pattern_indices = selected_relative.astype(np.int64)
            known_mask = (pattern_indices >= 0) & (
                pattern_indices < len(pattern.symbols)
            )
            reference_decisions[known_mask] = expected[pattern_indices[known_mask]]
            reference_absolute = np.cumprod(reference_decisions)
            base_absolute = waveform_symbols[1 + index :][selection]
            # Once the absolute reference sequence is available, start the
            # fine fit at zero drift.  The absolute waveform objective is
            # smooth in carrier phase/frequency/drift; seeding it with a blind
            # Mth-power slope can otherwise preserve a rare periodic alias.
            fine_slope = 0.0
            initial_dynamic_phase = (
                intercept * selected_relative
                + 0.5
                * fine_slope
                * selected_relative
                * (selected_relative + 1.0)
            )
            carrier_phase = float(
                np.angle(
                    np.sum(
                        base_absolute
                        * np.exp(-1j * initial_dynamic_phase)
                        * np.conj(reference_absolute)
                    )
                )
            )
            initial_parameters = np.asarray(
                [0.0, 0.0, carrier_phase, intercept, fine_slope],
                dtype=np.float64,
            )
            ambiguity = 2.0 * np.pi / float(alphabet.size)
            lower_bounds = np.asarray(
                [
                    -3.5,
                    -maximum_timing_rate,
                    carrier_phase - np.pi,
                    intercept - ambiguity / 2.0,
                    -maximum_phase_slope,
                ]
            )
            upper_bounds = np.asarray(
                [
                    3.5,
                    maximum_timing_rate,
                    carrier_phase + np.pi,
                    intercept + ambiguity / 2.0,
                    maximum_phase_slope,
                ]
            )

            def measured_for_parameters(parameters: np.ndarray) -> tuple[
                np.ndarray, np.ndarray, np.ndarray
            ]:
                (
                    timing_offset,
                    timing_rate,
                    model_phase,
                    model_intercept,
                    model_slope,
                ) = parameters
                candidate_centers = (
                    base_centers
                    + timing_offset
                    + timing_rate * timing_axis
                )
                candidate_symbols = _interpolate_complex(
                    resampled, candidate_centers
                )
                candidate_observed = candidate_symbols[1:] * np.conj(
                    candidate_symbols[:-1]
                )
                candidate_available = candidate_observed[index:]
                candidate_absolute = candidate_symbols[1 + index :][selection]
                absolute_phase = (
                    model_phase
                    + model_intercept * selected_relative
                    + 0.5
                    * model_slope
                    * selected_relative
                    * (selected_relative + 1.0)
                )
                corrected_absolute = candidate_absolute * np.exp(
                    -1j
                    * absolute_phase
                )
                return candidate_centers, candidate_available, corrected_absolute

            def optimize_reference_waveform(
                starting_parameters: np.ndarray,
            ) -> tuple[np.ndarray, np.ndarray, float]:
                parameters = np.asarray(starting_parameters, dtype=np.float64).copy()
                decisions = np.asarray(reference_decisions).copy()
                for _ in range(3):
                    fixed_reference = np.cumprod(decisions)

                    def synchronization_residual(candidate: np.ndarray) -> np.ndarray:
                        _, _, corrected_candidate = measured_for_parameters(candidate)
                        candidate_rms = float(
                            np.sqrt(np.mean(np.abs(corrected_candidate) ** 2))
                        )
                        normalized_candidate = corrected_candidate / max(
                            candidate_rms, _EPSILON
                        )
                        error = normalized_candidate - fixed_reference
                        return np.concatenate((error.real, error.imag))

                    optimized = least_squares(
                        synchronization_residual,
                        parameters,
                        bounds=(lower_bounds, upper_bounds),
                        loss="soft_l1",
                        f_scale=0.05,
                        x_scale=np.asarray(
                            [
                                0.5,
                                maximum_timing_rate,
                                0.1,
                                0.1,
                                maximum_phase_slope,
                            ]
                        ),
                        max_nfev=120,
                    )
                    parameters = optimized.x
                    _, differential_iteration, _ = measured_for_parameters(parameters)
                    iteration_intercept = float(parameters[3])
                    iteration_slope = float(parameters[4])
                    corrected_differential = differential_iteration[
                        selection
                    ] * np.exp(
                        -1j
                        * (
                            iteration_intercept
                            + iteration_slope * selected_relative
                        )
                    )
                    iteration_rms = float(
                        np.sqrt(np.mean(np.abs(corrected_differential) ** 2))
                    )
                    normalized_iteration = corrected_differential / max(
                        iteration_rms, _EPSILON
                    )
                    decisions = alphabet[
                        np.argmin(
                            np.abs(
                                normalized_iteration[:, None] - alphabet[None, :]
                            ),
                            axis=1,
                        )
                    ]
                    decisions[known_mask] = expected[pattern_indices[known_mask]]

                _, _, corrected_result = measured_for_parameters(parameters)
                result_rms = float(np.sqrt(np.mean(np.abs(corrected_result) ** 2)))
                normalized_result = corrected_result / max(result_rms, _EPSILON)
                result_reference = np.cumprod(decisions)
                result_evm = float(
                    np.sqrt(np.mean(np.abs(normalized_result - result_reference) ** 2))
                )
                return parameters, decisions, result_evm

            coarse_dynamic_phase = (
                intercept * selected_relative
                + 0.5 * slope * selected_relative * (selected_relative + 1.0)
            )
            coarse_carrier_phase = float(
                np.angle(
                    np.sum(
                        base_absolute
                        * np.exp(-1j * coarse_dynamic_phase)
                        * np.conj(reference_absolute)
                    )
                )
            )
            coarse_carrier_phase += 2.0 * np.pi * round(
                (carrier_phase - coarse_carrier_phase) / (2.0 * np.pi)
            )
            coarse_parameters = np.asarray(
                [0.0, 0.0, coarse_carrier_phase, intercept, slope],
                dtype=np.float64,
            )
            candidates = (
                optimize_reference_waveform(initial_parameters),
                optimize_reference_waveform(coarse_parameters),
            )
            parameters, reference_decisions, _ = min(
                candidates,
                key=lambda candidate: (
                    candidate[2],
                    abs(float(candidate[0][4])),
                ),
            )

            (
                fractional_timing_offset_samples,
                symbol_timing_rate_samples_per_symbol,
                carrier_phase,
                intercept,
                slope,
            ) = map(float, parameters)
            centers, available, fitted_corrected = measured_for_parameters(parameters)
            fitted_rms = float(np.sqrt(np.mean(np.abs(fitted_corrected) ** 2)))
            fitted_normalized = fitted_corrected / max(fitted_rms, _EPSILON)
            fitted_reference = np.cumprod(reference_decisions)
            fitted_error = fitted_normalized - fitted_reference
            synchronization_evm_rms = float(
                np.sqrt(np.mean(np.abs(fitted_error) ** 2))
            )
            phase_model_residual_rms_rad = float(
                np.sqrt(
                    np.mean(
                        np.angle(
                            fitted_normalized * np.conj(fitted_reference)
                        )
                        ** 2
                    )
                )
            )
            phase_drift_estimate_accepted = True
            final_symbols = _interpolate_complex(resampled, centers)
            final_observed = final_symbols[1:] * np.conj(final_symbols[:-1])
            final_scores = _normalized_complex_correlation(final_observed, expected)
            score = float(final_scores[index])
            applied_slope = (
                slope
                if demodulation.compensate_carrier_frequency_drift
                else 0.0
            )
            metric_parameters = np.asarray(parameters, dtype=np.float64).copy()
            if demodulation.compensate_carrier_frequency_drift:
                metric_parameters[4] = applied_slope
            else:
                # carrier_correct_recording() retains the fitted CFO at the
                # first-symbol reference time when drift correction is off.
                metric_parameters[3] = intercept + 0.5 * slope
                metric_parameters[4] = 0.0
            _, _, metric_absolute = measured_for_parameters(metric_parameters)
            metric_rms = float(np.sqrt(np.mean(np.abs(metric_absolute) ** 2)))
            normalized_absolute = metric_absolute / max(metric_rms, _EPSILON)
            absolute_reference = np.cumprod(reference_decisions)
            physical_evm_rms_percent = 100.0 * float(
                np.sqrt(
                    np.sum(np.abs(normalized_absolute - absolute_reference) ** 2)
                    / max(np.sum(np.abs(absolute_reference) ** 2), _EPSILON)
                )
            )
            if (
                signal.symbol_mapping == BLUETOOTH_EDR_MAPPING
                and normalized_absolute.size > 1
            ):
                reference_removed = normalized_absolute * np.conj(
                    absolute_reference
                )
                bluetooth_devm_rms_percent = 100.0 * float(
                    np.sqrt(
                        np.sum(np.abs(np.diff(reference_removed)) ** 2)
                        / max(np.sum(np.abs(reference_removed) ** 2), _EPSILON)
                    )
                )
            corrected_available = available * np.exp(
                -1j * (intercept + applied_slope * all_relative)
            )
            corrected = corrected_available[selection]
            carrier_offset_hz = (
                (intercept + 0.5 * slope)
                * signal.symbol_rate_hz
                / (2.0 * np.pi)
            )
            drift_hz_per_s = slope * signal.symbol_rate_hz**2 / (2.0 * np.pi)
            result_centers = centers[1 + index :][selection]
            start_center = centers[1 + index]
            phase_rotation = _wrap_phase(carrier_phase)
        else:
            available = waveform_symbols[index:]
            selection = _result_slice(len(pattern.symbols), available.size, result_range)
            base_centers = np.asarray(centers[index:], dtype=np.float64)
            carrier_symmetry_order = _psk_carrier_symmetry_order(
                signal.modulation
            )

            def fit_absolute_carrier(
                candidate: np.ndarray,
            ) -> tuple[float, float, float]:
                if signal.modulation is ModulationKind.QAM16:
                    return _fit_qam_carrier(candidate, alphabet)
                return _fit_nondifferential_psk_carrier(
                    candidate, alphabet, carrier_symmetry_order
                )

            def symbol_timing_cost(timing_offset: float) -> float:
                candidate = _interpolate_complex(
                    resampled, base_centers + float(timing_offset)
                )
                return fit_absolute_carrier(candidate[selection])[2]

            timing_bounds = (
                (-0.6, 0.6)
                if signal.modulation is ModulationKind.QAM16
                else (-1.25, 1.25)
            )
            timing_grid = np.linspace(
                timing_bounds[0],
                timing_bounds[1],
                13 if signal.modulation is ModulationKind.QAM16 else 21,
            )
            timing_costs = np.asarray(
                [symbol_timing_cost(value) for value in timing_grid]
            )
            best_timing_index = int(np.argmin(timing_costs))
            timing_step = float(timing_grid[1] - timing_grid[0])
            local_bounds = (
                max(timing_bounds[0], timing_grid[best_timing_index] - timing_step),
                min(timing_bounds[1], timing_grid[best_timing_index] + timing_step),
            )
            refined_timing = minimize_scalar(
                symbol_timing_cost,
                bounds=local_bounds,
                method="bounded",
                options={"xatol": 1e-4},
            )
            if (
                np.isfinite(refined_timing.fun)
                and float(refined_timing.fun) < symbol_timing_cost(0.0)
            ):
                fractional_timing_offset_samples = float(refined_timing.x)
            centers = base_centers + fractional_timing_offset_samples
            available = _interpolate_complex(resampled, centers)
            fit_window = available[: len(pattern.symbols)]
            phase_error = np.unwrap(np.angle(fit_window * np.conj(expected)))
            relative = np.arange(phase_error.size, dtype=np.float64)
            slope, intercept = np.polyfit(relative, phase_error, 1)
            selected_start = int(selection.start or 0)
            fitted_phase, fitted_step, synchronization_evm_rms = (
                fit_absolute_carrier(available[selection])
            )
            base_intercept = fitted_phase - fitted_step * selected_start
            ambiguity = 2.0 * np.pi / float(carrier_symmetry_order)
            training_axis = np.arange(fit_window.size, dtype=np.float64)

            def training_phase_cost(candidate_intercept: float) -> float:
                residual = np.angle(
                    fit_window
                    * np.conj(expected)
                    * np.exp(
                        -1j
                        * (
                            candidate_intercept
                            + fitted_step * training_axis
                        )
                    )
                )
                return float(np.mean(residual**2))

            intercept = min(
                (
                    base_intercept + ambiguity * rotation
                    for rotation in range(carrier_symmetry_order)
                ),
                key=training_phase_cost,
            )
            slope = fitted_step
            all_relative = np.arange(available.size, dtype=np.float64)
            corrected_available = available * np.exp(
                -1j * (intercept + slope * all_relative)
            )
            measured = available[selection]
            corrected = corrected_available[selection]
            carrier_offset_hz = slope * signal.symbol_rate_hz / (2.0 * np.pi)
            drift_hz_per_s = 0.0
            result_centers = centers[selection]
            start_center = centers[0]
            phase_rotation = _wrap_phase(intercept)

        rms = float(np.sqrt(np.mean(np.abs(corrected) ** 2)))
        normalized = corrected / max(rms, _EPSILON)
        distances = np.abs(normalized[:, None] - alphabet[None, :])
        decoded = np.argmin(distances, axis=1).astype(np.int16)
        training_rms = float(
            np.sqrt(np.mean(np.abs(corrected_available[: len(pattern.symbols)]) ** 2))
        )
        normalized_training = corrected_available[: len(pattern.symbols)] / max(
            training_rms, _EPSILON
        )
        training_decoded = np.argmin(
            np.abs(normalized_training[:, None] - alphabet[None, :]), axis=1
        )
        pattern_errors = int(
            np.count_nonzero(
                training_decoded != canonical_pattern_symbols
            )
        )
        start_boundary_resampled = start_center - samples_per_symbol / 2.0 + 0.5
        start_sample = int(
            round(
                start_boundary_resampled
                * recording.sample_rate_hz
                / analysis_rate_hz
            )
        )
        start_sample = min(recording.sample_count - 1, max(0, start_sample))
        half_symbol_s = 0.5 / signal.symbol_rate_hz
        result_start_sample = max(
            0,
            int(
                round(
                    (float(result_centers[0]) / analysis_rate_hz - half_symbol_s)
                    * recording.sample_rate_hz
                )
            ),
        )
        result_stop_sample = min(
            recording.sample_count,
            int(
                round(
                    (float(result_centers[-1]) / analysis_rate_hz + half_symbol_s)
                    * recording.sample_rate_hz
                )
            ),
        )
        displayed_symbols = np.asarray(normalized, dtype=np.complex64)
        decision_reference = alphabet[decoded]
        evm_rms_percent = (
            100.0
            * float(
                np.sqrt(
                    np.sum(np.abs(displayed_symbols - decision_reference) ** 2)
                    / np.sum(np.abs(decision_reference) ** 2)
                )
            )
            if decision_reference.size
            else None
        )
        if not signal.modulation.differential:
            physical_evm_rms_percent = evm_rms_percent
        differential_symbol_evm_rms_percent = (
            evm_rms_percent if signal.modulation.differential else None
        )
        return PatternSearchResult(
            modulation=signal.modulation,
            pattern_start_sample=start_sample,
            pattern_start_time_s=start_sample / recording.sample_rate_hz,
            pattern_start_symbol=(index + 1 if signal.modulation.differential else index),
            result_start_sample=result_start_sample,
            result_stop_sample=result_stop_sample,
            correlation=score,
            pattern_symbol_errors=pattern_errors,
            decoded_symbols=decoded,
            decoded_bits=_symbols_to_bits(
                decoded, signal.modulation.order, demodulation.bit_ordering
            ),
            measured_symbols=displayed_symbols,
            symbol_time_s=result_centers / analysis_rate_hz,
            carrier_frequency_offset_hz=float(carrier_offset_hz),
            carrier_frequency_drift_hz_per_s=float(drift_hz_per_s),
            frequency_deviation_hz=None,
            evm_rms_percent=evm_rms_percent,
            polarity_inverted=False,
            phase_rotation_rad=phase_rotation,
            timing_phase_samples=phase,
            analysis_sample_rate_hz=analysis_rate_hz,
            recording_sample_rate_hz=recording.sample_rate_hz,
            carrier_reference_time_s=(
                float(result_centers[0]) / analysis_rate_hz
                if signal.modulation.differential
                else float(start_center) / analysis_rate_hz
            ),
            metadata={
                "pattern_name": pattern.name,
                "pattern_symbol_count": len(pattern.symbols),
                "symbol_rate_hz": signal.symbol_rate_hz,
                "result_length": result_range.result_length,
                "result_offset_symbols": result_range.offset_symbols,
                "differential": signal.modulation.differential,
                "matched_filter_applied": matched_filter_applied,
                "measurement_filter": demodulation.measurement_filter.value,
                "prefilter_carrier_frequency_offset_hz": (
                    prefilter_carrier_frequency_offset_hz
                ),
                "carrier_drift_compensated": (
                    demodulation.compensate_carrier_frequency_drift
                ),
                "phase_estimation_method": (
                    "joint ideal-reference waveform complex-EVM synchronization"
                    if signal.modulation.differential
                    else (
                        "known-pattern ambiguity with result-range QAM carrier fit"
                        if signal.modulation is ModulationKind.QAM16
                        else "known-pattern ambiguity with result-range PSK carrier fit"
                    )
                ),
                "phase_model_residual_rms_rad": phase_model_residual_rms_rad,
                "phase_drift_estimate_accepted": phase_drift_estimate_accepted,
                "phase_no_drift_residual_rms_rad": (
                    phase_no_drift_residual_rms_rad
                ),
                "fractional_timing_offset_samples": (
                    fractional_timing_offset_samples
                ),
                "symbol_timing_rate_samples_per_symbol": (
                    symbol_timing_rate_samples_per_symbol
                ),
                "symbol_rate_error_ppm": (
                    -symbol_timing_rate_samples_per_symbol
                    / float(samples_per_symbol)
                    * 1.0e6
                ),
                "synchronization_evm_rms": synchronization_evm_rms,
                "physical_evm_rms_percent": physical_evm_rms_percent,
                "differential_symbol_evm_rms_percent": (
                    differential_symbol_evm_rms_percent
                ),
                "bluetooth_devm_rms_percent": bluetooth_devm_rms_percent,
                "absolute_reference_waveform_sync": (
                    signal.modulation.differential
                ),
                "source": recording.source,
                "match_selection_policy": search.match_selection.value,
                "selected_match_index": selected_match_index,
                "eligible_match_count": eligible_match_count,
                "detected_match_count": detected_match_count,
                "eligible_match_start_samples": eligible_match_start_samples,
                "exclude_incomplete_result": (
                    result_range.exclude_incomplete_result
                ),
            },
        )
