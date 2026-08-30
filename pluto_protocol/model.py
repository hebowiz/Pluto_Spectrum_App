"""UI- and instrument-independent protocol analysis data model."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


class BitRepresentation(StrEnum):
    AIR = "air"
    LOGICAL = "logical"


class FieldStatus(StrEnum):
    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"
    UNKNOWN = "unknown"
    INFO = "info"


class IssueSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


def _readonly_bits(bits: Any) -> np.ndarray:
    values = np.asarray(bits, dtype=np.uint8)
    if values.ndim != 1 or np.any(values > 1):
        raise ValueError("bits must be a one-dimensional binary array")
    owned = np.array(values, dtype=np.uint8, copy=True)
    owned.flags.writeable = False
    return owned


@dataclass(frozen=True)
class PacketSourceInfo:
    source_kind: str = "unknown"
    packet_index: int | None = None
    timestamp_s: float | None = None
    center_frequency_hz: float | None = None
    start_sample: int | None = None
    stop_sample: int | None = None


@dataclass(frozen=True)
class PacketDecodeInput:
    bits: np.ndarray
    representation: BitRepresentation = BitRepresentation.AIR
    protocol_hint: str | None = None
    phy_hint: str | None = None
    source: PacketSourceInfo = field(default_factory=PacketSourceInfo)
    context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "bits", _readonly_bits(self.bits))
        object.__setattr__(self, "representation", BitRepresentation(self.representation))
        object.__setattr__(self, "context", MappingProxyType(dict(self.context)))


@dataclass(frozen=True)
class PacketField:
    field_id: str
    name: str
    start_bit: int
    stop_bit: int
    raw_bits: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.uint8))
    value: Any = None
    meaning: str = ""
    status: FieldStatus = FieldStatus.INFO
    children: tuple["PacketField", ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_bits", _readonly_bits(self.raw_bits))
        object.__setattr__(self, "status", FieldStatus(self.status))
        object.__setattr__(self, "children", tuple(self.children))


@dataclass(frozen=True)
class PacketSummaryItem:
    key: str
    label: str
    value: Any
    display: str
    status: FieldStatus = FieldStatus.INFO


@dataclass(frozen=True)
class PacketIssue:
    code: str
    message: str
    severity: IssueSeverity = IssueSeverity.WARNING
    start_bit: int | None = None
    stop_bit: int | None = None


@dataclass(frozen=True)
class PacketIntegritySummary:
    hec_valid: bool | None = None
    crc_valid: bool | None = None
    complete: bool = True


@dataclass(frozen=True)
class DecodeProbeResult:
    protocol_id: str
    confidence: float
    reason: str = ""


@dataclass(frozen=True)
class PacketAnalysisResult:
    schema_version: str
    protocol_id: str
    protocol_name: str
    phy_name: str | None
    packet_type: str | None
    summary: tuple[PacketSummaryItem, ...]
    root_fields: tuple[PacketField, ...]
    issues: tuple[PacketIssue, ...]
    integrity: PacketIntegritySummary
    source: PacketSourceInfo
    raw_bits: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "summary", tuple(self.summary))
        object.__setattr__(self, "root_fields", tuple(self.root_fields))
        object.__setattr__(self, "issues", tuple(self.issues))
        object.__setattr__(self, "raw_bits", _readonly_bits(self.raw_bits))


@dataclass(frozen=True)
class GeneratedPacketBits:
    """Exact transmitted packet bits produced by a waveform generator."""

    bits: np.ndarray
    protocol_id: str
    phy_name: str
    representation: BitRepresentation = BitRepresentation.AIR
    context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "bits", _readonly_bits(self.bits))
        object.__setattr__(self, "representation", BitRepresentation(self.representation))
        object.__setattr__(self, "context", MappingProxyType(dict(self.context)))
