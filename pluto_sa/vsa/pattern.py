"""Modulation-agnostic known-pattern search and symbol decoding."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from types import MappingProxyType
from typing import Mapping

import numpy as np
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


@dataclass(frozen=True)
class PatternSearchSettings:
    """Settings corresponding to R&S VSA ``Pattern Search``."""

    pattern: KnownPattern
    mode: PatternSearchMode = PatternSearchMode.AUTO
    iq_correlation_threshold: float = 0.9
    correlation_threshold_auto: bool = True
    meas_only_if_pattern_symbols_correct: bool = True

    def __post_init__(self) -> None:
        threshold = float(self.iq_correlation_threshold)
        if not 0.0 < threshold <= 1.0:
            raise ValueError("iq_correlation_threshold must be in the range (0, 1]")

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
    compensate_carrier_frequency_drift: bool = True
    compensate_fsk_deviation_error: bool = True


@dataclass(frozen=True)
class PatternSearchResult:
    modulation: ModulationKind
    pattern_start_sample: int
    pattern_start_time_s: float
    pattern_start_symbol: int
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


def _result_slice(
    pattern_size: int, available: int, settings: ResultRangeSettings
) -> slice:
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
    stop = min(int(available), start + int(settings.result_length))
    return slice(start, max(start, stop))


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
        demodulation = demodulate_gfsk(
            recording.iq,
            sample_rate_hz=recording.sample_rate_hz,
            access_bits=bits,
            symbol_rate_hz=signal.symbol_rate_hz,
            minimum_correlation=search.effective_correlation_threshold,
            gaussian_bt=gaussian_bt,
        )
        selection = _result_slice(len(pattern.symbols), demodulation.bits.size, result_range)
        decoded = demodulation.bits[selection].astype(np.int16)
        measured = demodulation.symbol_frequency_hz[selection].astype(np.complex64)
        times = demodulation.symbol_time_s[selection]
        return PatternSearchResult(
            modulation=signal.modulation,
            pattern_start_sample=demodulation.access_start_sample,
            pattern_start_time_s=(
                demodulation.access_start_sample / recording.sample_rate_hz
            ),
            pattern_start_symbol=demodulation.access_start_bit,
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
            metadata={
                "pattern_name": pattern.name,
                "pattern_symbol_count": len(pattern.symbols),
                "result_length": result_range.result_length,
                "result_offset_symbols": result_range.offset_symbols,
                "gaussian_bt": gaussian_bt,
                "source": recording.source,
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
        resampled, analysis_rate_hz = _resample_for_symbols(
            recording.iq,
            recording.sample_rate_hz,
            signal.symbol_rate_hz,
        )
        samples_per_symbol = 8
        alphabet = _constellation(signal.modulation)
        expected = alphabet[np.asarray(pattern.symbols, dtype=np.int16)]
        best: tuple[float, int, int, np.ndarray, np.ndarray] | None = None
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
            index = int(np.argmax(scores))
            score = float(scores[index])
            # Rectangular synthetic waveforms can produce numerically equal
            # correlations at a symbol centre and at a transition. Prefer the
            # earlier timing phase for ties; this keeps the reported symbol
            # index aligned with the capture rather than one symbol early.
            if best is None or score > best[0] + 1e-10:
                best = (score, phase, index, waveform_symbols, centers)
        if best is None or best[0] < search.effective_correlation_threshold:
            observed_score = 0.0 if best is None else best[0]
            raise ValueError(
                f"known pattern was not found (correlation={observed_score:.3f})"
            )

        score, phase, index, waveform_symbols, centers = best
        if signal.modulation.differential:
            observed = waveform_symbols[1:] * np.conj(waveform_symbols[:-1])
            available = observed[index:]
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
            carrier_offset_hz = intercept * signal.symbol_rate_hz / (2.0 * np.pi)
            drift_hz_per_s = slope * signal.symbol_rate_hz**2 / (2.0 * np.pi)
            result_centers = centers[1 + index :][selection]
            start_center = centers[1 + index]
            phase_rotation: float | None = None
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
        return PatternSearchResult(
            modulation=signal.modulation,
            pattern_start_sample=start_sample,
            pattern_start_time_s=start_sample / recording.sample_rate_hz,
            pattern_start_symbol=(index + 1 if signal.modulation.differential else index),
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
            metadata={
                "pattern_name": pattern.name,
                "pattern_symbol_count": len(pattern.symbols),
                "result_length": result_range.result_length,
                "result_offset_symbols": result_range.offset_symbols,
                "differential": signal.modulation.differential,
                "source": recording.source,
            },
        )
