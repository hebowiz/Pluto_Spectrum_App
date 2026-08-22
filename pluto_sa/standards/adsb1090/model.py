"""Immutable contracts for 1090 MHz Mode S / ADS-B results."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class ADSB1090Settings:
    """Detector and decoder settings independent of the IQ source."""

    minimum_preamble_snr_db: float = 8.0
    minimum_preamble_correlation: float = 0.72
    require_valid_crc: bool = False
    maximum_messages: int = 4096

    def __post_init__(self) -> None:
        if not np.isfinite(self.minimum_preamble_snr_db):
            raise ValueError("minimum_preamble_snr_db must be finite")
        if not 0.0 <= float(self.minimum_preamble_correlation) <= 1.0:
            raise ValueError("minimum_preamble_correlation must be between 0 and 1")
        if int(self.maximum_messages) <= 0:
            raise ValueError("maximum_messages must be positive")


@dataclass(frozen=True)
class ADSB1090Message:
    start_sample: int
    sample_rate_hz: float
    raw_hex: str
    bits: np.ndarray
    downlink_format: int
    capability: int
    icao_address: str | None
    type_code: int | None
    crc_remainder: int
    crc_ok: bool
    preamble_snr_db: float
    preamble_correlation: float
    fields: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.asarray(self.bits, dtype=np.uint8).reshape(-1).copy()
        if values.size not in {56, 112}:
            raise ValueError("a Mode S message must contain 56 or 112 bits")
        if np.any(values > 1):
            raise ValueError("message bits must be binary")
        values.flags.writeable = False
        object.__setattr__(self, "bits", values)
        object.__setattr__(self, "fields", MappingProxyType(dict(self.fields)))

    @property
    def bit_length(self) -> int:
        return int(self.bits.size)

    @property
    def start_time_s(self) -> float:
        return float(self.start_sample) / float(self.sample_rate_hz)

    @property
    def is_adsb_extended_squitter(self) -> bool:
        return self.downlink_format in {17, 18}


@dataclass(frozen=True)
class ADSB1090AnalysisResult:
    time_s: np.ndarray
    power_dbfs: np.ndarray
    messages: tuple[ADSB1090Message, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        time = np.asarray(self.time_s, dtype=np.float64).reshape(-1).copy()
        power = np.asarray(self.power_dbfs, dtype=np.float64).reshape(-1).copy()
        if time.size != power.size:
            raise ValueError("time_s and power_dbfs must have equal length")
        time.flags.writeable = False
        power.flags.writeable = False
        object.__setattr__(self, "time_s", time)
        object.__setattr__(self, "power_dbfs", power)
        object.__setattr__(self, "messages", tuple(self.messages))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
