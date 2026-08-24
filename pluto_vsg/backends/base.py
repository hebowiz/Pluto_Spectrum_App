"""Output backend contract kept separate from waveform generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pluto_vsg.engine import GenerationResult


@dataclass(frozen=True)
class BackendCapabilities:
    backend_name: str
    supports_rf_output: bool
    supports_hardware_trigger: bool = False
    maximum_sample_rate_hz: float | None = None


class OutputBackend(Protocol):
    @property
    def capabilities(self) -> BackendCapabilities:
        ...

    def transfer(self, result: GenerationResult) -> None:
        ...

    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...
