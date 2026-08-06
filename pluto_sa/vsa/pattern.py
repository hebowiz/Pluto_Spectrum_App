"""Modulation-agnostic known-pattern search and symbol decoding."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from fractions import Fraction
from types import MappingProxyType
from typing import Mapping

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import resample_poly

from pluto_sa.vsa.demod.gfsk import demodulate_gfsk
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
class PatternSearchSettings:
    """Settings corresponding to R&S VSA ``Pattern Search``."""

    pattern: KnownPattern
    mode: PatternSearchMode = PatternSearchMode.AUTO
    iq_correlation_threshold: float = 0.9
    correlation_threshold_auto: bool = True
    meas_only_if_pattern_symbols_correct: bool = True
    match_selection: MatchSelectionPolicy = MatchSelectionPolicy.FIRST
    match_index: int = 1

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


@dataclass(frozen=True)
class DemodulationSettings:
    """Implemented subset of R&S VSA ``Demodulation`` settings."""

    coarse_synchronization: SynchronizationSource = SynchronizationSource.AUTO
    fine_synchronization: SynchronizationSource = SynchronizationSource.AUTO
    bit_ordering: BitOrdering = BitOrdering.MSB
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


def _constellation(kind: ModulationKind) -> np.ndarray:
    if kind is ModulationKind.BPSK:
        phases = np.asarray([0.0, np.pi])
    elif kind in {ModulationKind.QPSK, ModulationKind.OQPSK, ModulationKind.PI4_DQPSK}:
        phases = np.pi / 4.0 + np.arange(4) * np.pi / 2.0
    elif kind is ModulationKind.DPSK8:
        phases = np.arange(8) * np.pi / 4.0
    else:
        raise ValueError(f"{kind.value} does not have a PSK constellation")
    return np.exp(1j * phases)


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
) -> tuple[np.ndarray, float]:
    """Resample PSK IQ and apply the configured matched receive filter."""
    waveform, analysis_rate_hz = _resample_for_symbols(
        iq,
        sample_rate_hz,
        symbol_rate_hz,
        samples_per_symbol=samples_per_symbol,
    )
    if tx_filter.lower() in {
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
        if signal.modulation.family is ModulationFamily.FSK:
            return self._search_fsk(
                recording, signal, search, result_range, demodulation
            )
        return self._search_psk(recording, signal, search, result_range, demodulation)

    def _search_fsk(
        self,
        recording: IQRecording,
        signal: SignalDescription,
        search: PatternSearchSettings,
        result_range: ResultRangeSettings,
        demodulation_settings: DemodulationSettings,
    ) -> PatternSearchResult:
        pattern = search.pattern
        if len(pattern.symbols) < 8:
            raise ValueError("FSK known pattern must contain at least eight bits")
        bits = np.asarray(pattern.symbols, dtype=np.uint8)
        gaussian_bt = (
            signal.filter_parameter
            if signal.modulation is ModulationKind.GFSK
            and signal.tx_filter.lower() == "gaussian"
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
            maximum_symbols=maximum_symbols,
            match_selection=search.match_selection.value,
            match_index=search.match_index,
            required_result_symbols=maximum_symbols,
            exclude_incomplete_result=result_range.exclude_incomplete_result,
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
        return PatternSearchResult(
            modulation=signal.modulation,
            pattern_start_sample=demodulation.access_start_sample,
            pattern_start_time_s=(
                demodulation.access_start_sample / recording.sample_rate_hz
            ),
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
            polarity_inverted=demodulation.iq_inverted,
            phase_rotation_rad=None,
            timing_phase_samples=demodulation.timing_phase_samples,
            analysis_sample_rate_hz=demodulation.analysis_sample_rate_hz,
            recording_sample_rate_hz=recording.sample_rate_hz,
            carrier_reference_time_s=(
                demodulation.access_start_sample / recording.sample_rate_hz
                + len(pattern.symbols) / (2.0 * signal.symbol_rate_hz)
            ),
            metadata={
                "pattern_name": pattern.name,
                "pattern_symbol_count": len(pattern.symbols),
                "symbol_rate_hz": signal.symbol_rate_hz,
                "result_length": result_range.result_length,
                "result_offset_symbols": result_range.offset_symbols,
                "gaussian_bt": gaussian_bt,
                "source": recording.source,
                "match_selection_policy": search.match_selection.value,
                "selected_match_index": demodulation.selected_match_index,
                "eligible_match_count": demodulation.eligible_match_count,
                "detected_match_count": demodulation.detected_match_count,
                "exclude_incomplete_result": (
                    result_range.exclude_incomplete_result
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
        pattern = search.pattern
        samples_per_symbol = 8
        resampled, analysis_rate_hz = prepare_psk_iq(
            recording.iq,
            sample_rate_hz=recording.sample_rate_hz,
            symbol_rate_hz=signal.symbol_rate_hz,
            tx_filter=signal.tx_filter,
            filter_parameter=signal.filter_parameter,
            samples_per_symbol=samples_per_symbol,
        )
        matched_filter_applied = signal.tx_filter.lower() in {
            "root raised cosine",
            "root-raised-cosine",
            "rrc",
            "srrc",
        }
        alphabet = _constellation(signal.modulation)
        expected = alphabet[np.asarray(pattern.symbols, dtype=np.int16)]
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
            indices = np.clip(np.rint(centers).astype(np.int64), 0, resampled.size - 1)
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
                    centers[int(index) + symbol_offset]
                )
                candidates.append(
                    (
                        float(scores[index]),
                        phase,
                        int(index),
                        waveform_symbols,
                        centers,
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
            ):
                eligible_candidates.append(candidate)
        best, selected_match_index, eligible_match_count = _select_match_candidate(
            eligible_candidates,
            search.match_selection,
            search.match_index,
            time_key=lambda item: item[5],
            score_key=lambda item: item[0],
        )
        score, phase, index, waveform_symbols, centers, _ = best
        phase_model_residual_rms_rad: float | None = None
        phase_drift_estimate_accepted: bool | None = None
        phase_no_drift_residual_rms_rad: float | None = None
        fractional_timing_offset_samples = 0.0
        symbol_timing_rate_samples_per_symbol = 0.0
        synchronization_evm_rms: float | None = None
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
            fit_window = available[: len(pattern.symbols)]
            phase_error = np.unwrap(np.angle(fit_window * np.conj(expected)))
            relative = np.arange(phase_error.size, dtype=np.float64)
            slope, intercept = np.polyfit(relative, phase_error, 1)
            all_relative = np.arange(available.size, dtype=np.float64)
            corrected_available = available * np.exp(
                -1j * (intercept + slope * all_relative)
            )
            measured = available[selection]
            corrected = corrected_available[selection]
            carrier_offset_hz = slope * signal.symbol_rate_hz / (2.0 * np.pi)
            drift_hz_per_s = 0.0
            result_centers = centers[index:][selection]
            start_center = centers[index]
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
                training_decoded
                != np.asarray(pattern.symbols, dtype=np.int16)
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
            measured_symbols=normalized,
            symbol_time_s=result_centers / analysis_rate_hz,
            carrier_frequency_offset_hz=float(carrier_offset_hz),
            carrier_frequency_drift_hz_per_s=float(drift_hz_per_s),
            frequency_deviation_hz=None,
            polarity_inverted=False,
            phase_rotation_rad=phase_rotation,
            timing_phase_samples=phase,
            analysis_sample_rate_hz=analysis_rate_hz,
            recording_sample_rate_hz=recording.sample_rate_hz,
            carrier_reference_time_s=(
                float(result_centers[0]) / analysis_rate_hz
                if signal.modulation.differential
                else (
                    start_sample / recording.sample_rate_hz
                    + 0.5 / signal.symbol_rate_hz
                )
            ),
            metadata={
                "pattern_name": pattern.name,
                "pattern_symbol_count": len(pattern.symbols),
                "symbol_rate_hz": signal.symbol_rate_hz,
                "result_length": result_range.result_length,
                "result_offset_symbols": result_range.offset_symbols,
                "differential": signal.modulation.differential,
                "matched_filter_applied": matched_filter_applied,
                "carrier_drift_compensated": (
                    demodulation.compensate_carrier_frequency_drift
                ),
                "phase_estimation_method": (
                    "joint ideal-reference waveform complex-EVM synchronization"
                    if signal.modulation.differential
                    else "known-pattern phase fit"
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
                "absolute_reference_waveform_sync": (
                    signal.modulation.differential
                ),
                "source": recording.source,
                "match_selection_policy": search.match_selection.value,
                "selected_match_index": selected_match_index,
                "eligible_match_count": eligible_match_count,
                "detected_match_count": detected_match_count,
                "exclude_incomplete_result": (
                    result_range.exclude_incomplete_result
                ),
            },
        )
