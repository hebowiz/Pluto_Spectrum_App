"""Source-independent data contracts for vector signal analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Mapping

import numpy as np


class ModulationFamily(str, Enum):
    FSK = "FSK"
    PSK = "PSK"
    QAM = "QAM"


class ModulationKind(str, Enum):
    FSK2 = "2-FSK"
    GFSK = "GFSK"
    BPSK = "BPSK"
    QPSK = "QPSK"
    OQPSK = "OQPSK"
    PI4_DQPSK = "pi/4-DQPSK"
    DPSK8 = "8DPSK"

    @property
    def family(self) -> ModulationFamily:
        if self in {ModulationKind.FSK2, ModulationKind.GFSK}:
            return ModulationFamily.FSK
        return ModulationFamily.PSK

    @property
    def order(self) -> int:
        return {
            ModulationKind.FSK2: 2,
            ModulationKind.GFSK: 2,
            ModulationKind.BPSK: 2,
            ModulationKind.QPSK: 4,
            ModulationKind.OQPSK: 4,
            ModulationKind.PI4_DQPSK: 4,
            ModulationKind.DPSK8: 8,
        }[self]

    @property
    def differential(self) -> bool:
        return self in {ModulationKind.PI4_DQPSK, ModulationKind.DPSK8}


def _readonly_complex64(values: np.ndarray) -> np.ndarray:
    array = np.array(values, dtype=np.complex64, copy=True)
    if array.ndim != 1:
        raise ValueError("IQ samples must be one-dimensional")
    array.flags.writeable = False
    return array


def _readonly_array(values: np.ndarray, dtype: np.dtype | type) -> np.ndarray:
    array = np.array(values, dtype=dtype, copy=True)
    array.flags.writeable = False
    return array


@dataclass(frozen=True)
class IQRecording:
    """Immutable IQ capture plus the metadata required to interpret it."""

    iq: np.ndarray
    sample_rate_hz: float
    center_frequency_hz: float = 0.0
    usable_bandwidth_hz: float | None = None
    source: str = "unknown"
    full_scale: float = 1.0
    calibration_offset_db: float = 0.0
    frequency_dependent_offset_db: float = 0.0
    input_correction_db: float = 0.0
    amplitude_calibrated: bool = False
    start_sample_index: int = 0
    trigger_sample_index: int | None = None
    discontinuity_reason: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not np.isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz must be positive")
        if not np.isfinite(self.center_frequency_hz):
            raise ValueError("center_frequency_hz must be finite")
        if self.usable_bandwidth_hz is not None and self.usable_bandwidth_hz <= 0.0:
            raise ValueError("usable_bandwidth_hz must be positive when provided")
        if not np.isfinite(self.full_scale) or self.full_scale <= 0.0:
            raise ValueError("full_scale must be positive")
        corrections = (
            self.calibration_offset_db,
            self.frequency_dependent_offset_db,
            self.input_correction_db,
        )
        if not all(np.isfinite(value) for value in corrections):
            raise ValueError("amplitude correction values must be finite")
        owned = _readonly_complex64(self.iq)
        if owned.size == 0:
            raise ValueError("IQ recording must contain at least one sample")
        if self.trigger_sample_index is not None:
            if not self.start_sample_index <= self.trigger_sample_index < self.end_sample_index:
                raise ValueError("trigger_sample_index must be inside the recording")
        object.__setattr__(self, "iq", owned)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def sample_count(self) -> int:
        return int(self.iq.size)

    @property
    def end_sample_index(self) -> int:
        return int(self.start_sample_index) + self.sample_count

    @property
    def duration_s(self) -> float:
        return self.sample_count / float(self.sample_rate_hz)

    @property
    def is_contiguous(self) -> bool:
        return self.discontinuity_reason is None

    @property
    def dbfs_to_dbm_offset_db(self) -> float:
        """Offset matching the common TA/Power Trigger amplitude convention."""
        return float(
            20.0 * np.log10(self.full_scale)
            + self.calibration_offset_db
            + self.frequency_dependent_offset_db
            + self.input_correction_db
        )


@dataclass(frozen=True)
class SignalDescription:
    modulation: ModulationKind
    symbol_rate_hz: float
    frequency_deviation_hz: float | None = None
    tx_filter: str = "None"
    filter_parameter: float | None = None
    symbol_mapping: str = "Natural"
    name: str = "Manual"

    def __post_init__(self) -> None:
        if not np.isfinite(self.symbol_rate_hz) or self.symbol_rate_hz <= 0.0:
            raise ValueError("symbol_rate_hz must be positive")
        if self.modulation.family is ModulationFamily.FSK:
            if self.frequency_deviation_hz is not None and self.frequency_deviation_hz <= 0.0:
                raise ValueError("frequency_deviation_hz must be positive")
        normalized_filter = str(self.tx_filter).strip() or "None"
        object.__setattr__(self, "tx_filter", normalized_filter)
        from pluto_sa.vsa.mapping import (
            BLUETOOTH_EDR_MAPPING,
            NATURAL_MAPPING,
            normalize_symbol_mapping,
        )

        normalized_mapping = normalize_symbol_mapping(self.symbol_mapping)
        if (
            self.modulation.family is ModulationFamily.FSK
            and normalized_mapping != NATURAL_MAPPING
        ):
            raise ValueError("FSK modulation mapping must be Natural")
        if normalized_mapping == BLUETOOTH_EDR_MAPPING and self.modulation not in {
            ModulationKind.PI4_DQPSK,
            ModulationKind.DPSK8,
        }:
            raise ValueError(
                "Bluetooth EDR mapping requires pi/4-DQPSK or 8DPSK"
            )
        object.__setattr__(self, "symbol_mapping", normalized_mapping)


@dataclass(frozen=True)
class ModulationSegment:
    """One modulation region on the authoritative recording timeline."""

    start_sample: int
    stop_sample: int
    signal: SignalDescription
    name: str = "Segment"
    evaluation_start_symbol: int = 0
    evaluation_stop_symbol: int | None = None

    def __post_init__(self) -> None:
        if int(self.start_sample) < 0:
            raise ValueError("start_sample must be non-negative")
        if int(self.stop_sample) <= int(self.start_sample):
            raise ValueError("stop_sample must be greater than start_sample")
        if int(self.evaluation_start_symbol) < 0:
            raise ValueError("evaluation_start_symbol must be non-negative")
        if (
            self.evaluation_stop_symbol is not None
            and int(self.evaluation_stop_symbol) <= int(self.evaluation_start_symbol)
        ):
            raise ValueError("evaluation_stop_symbol must exceed evaluation_start_symbol")


@dataclass(frozen=True)
class CompositeSignalDescription:
    segments: tuple[ModulationSegment, ...]
    profile_name: str = "Manual Composite"

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("a composite signal requires at least one segment")
        ordered = tuple(sorted(self.segments, key=lambda segment: segment.start_sample))
        for previous, current in zip(ordered, ordered[1:]):
            if current.start_sample < previous.stop_sample:
                raise ValueError("modulation segments must not overlap")
        object.__setattr__(self, "segments", ordered)


@dataclass(frozen=True)
class VSASettings:
    fft_size: int = 4096
    timing_offset_samples: float = 0.0
    remove_dc: bool = True
    analysis_center_frequency_hz: float | None = None
    analysis_bandwidth_hz: float | None = None

    def __post_init__(self) -> None:
        if int(self.fft_size) < 16:
            raise ValueError("fft_size must be at least 16")
        if not np.isfinite(self.timing_offset_samples):
            raise ValueError("timing_offset_samples must be finite")
        if (
            self.analysis_center_frequency_hz is not None
            and not np.isfinite(self.analysis_center_frequency_hz)
        ):
            raise ValueError("analysis_center_frequency_hz must be finite")
        if (
            self.analysis_bandwidth_hz is not None
            and (
                not np.isfinite(self.analysis_bandwidth_hz)
                or self.analysis_bandwidth_hz <= 0.0
            )
        ):
            raise ValueError("analysis_bandwidth_hz must be positive when provided")


@dataclass(frozen=True)
class VSAAnalysisResult:
    """One immutable analysis snapshot shared by all result windows."""

    time_s: np.ndarray
    iq: np.ndarray
    power_dbfs: np.ndarray
    power_dbm: np.ndarray
    spectrum_frequency_hz: np.ndarray
    spectrum_dbfs: np.ndarray
    spectrum_dbm: np.ndarray
    instantaneous_frequency_hz: np.ndarray
    symbol_time_s: np.ndarray
    measured_symbols: np.ndarray
    reference_symbols: np.ndarray
    decoded_symbols: np.ndarray
    decoded_bits: np.ndarray
    evm_rms_percent: float | None
    frequency_error_hz: float | None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        array_fields = {
            "time_s": (self.time_s, np.float64),
            "iq": (self.iq, np.complex64),
            "power_dbfs": (self.power_dbfs, np.float64),
            "power_dbm": (self.power_dbm, np.float64),
            "spectrum_frequency_hz": (self.spectrum_frequency_hz, np.float64),
            "spectrum_dbfs": (self.spectrum_dbfs, np.float64),
            "spectrum_dbm": (self.spectrum_dbm, np.float64),
            "instantaneous_frequency_hz": (self.instantaneous_frequency_hz, np.float64),
            "symbol_time_s": (self.symbol_time_s, np.float64),
            "measured_symbols": (self.measured_symbols, np.complex64),
            "reference_symbols": (self.reference_symbols, np.complex64),
            "decoded_symbols": (self.decoded_symbols, np.int16),
            "decoded_bits": (self.decoded_bits, np.uint8),
        }
        for name, (values, dtype) in array_fields.items():
            object.__setattr__(self, name, _readonly_array(values, dtype))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class VSASegmentAnalysis:
    segment: ModulationSegment
    result: VSAAnalysisResult


@dataclass(frozen=True)
class CompositeVSAAnalysisResult:
    """Results from every modulation region in one capture/result range."""

    profile_name: str
    segments: tuple[VSASegmentAnalysis, ...]
    decoded_bits: np.ndarray
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("composite analysis requires at least one segment result")
        object.__setattr__(self, "segments", tuple(self.segments))
        object.__setattr__(self, "decoded_bits", _readonly_array(self.decoded_bits, np.uint8))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
