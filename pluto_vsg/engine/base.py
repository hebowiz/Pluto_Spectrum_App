"""Interfaces shared by waveform engines and the GUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import numpy.typing as npt

from pluto_vsg.model import WaveformProject


@dataclass(frozen=True)
class FieldBoundary:
    name: str
    start_sample: int
    stop_sample: int
    start_symbol: int = 0
    stop_symbol: int = 0
    logical_bit_count: int | None = None
    level: int = 0
    parent_name: str | None = None


@dataclass(frozen=True)
class GenerationResult:
    iq: npt.NDArray[np.complex64]
    sample_rate_hz: float
    field_boundaries: tuple[FieldBoundary, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)


class WaveformEngine(Protocol):
    """Generate normalized complex IQ without knowing the output device."""

    def generate(self, project: WaveformProject) -> GenerationResult:
        ...
