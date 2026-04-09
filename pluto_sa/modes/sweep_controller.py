"""Sweep SA execution controller skeleton."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pluto_sa.config.spectrum_config import SpectrumConfig
from pluto_sa.sdr.pluto_receiver import PlutoReceiver


@dataclass
class SweepFrameResult:
    """Latest sweep frame snapshot passed from controller to UI."""

    freq_axis_hz: np.ndarray | None = None
    display_db: np.ndarray | None = None
    completed_points: int = 0
    sweep_complete: bool = False


class SweepController:
    """Own Sweep SA run-state without owning SDR lifetime management."""

    def __init__(self, config: SpectrumConfig, receiver: PlutoReceiver) -> None:
        self.config = config
        self.receiver = receiver
        self._run_requested = False
        self._single_requested = False
        self._current_point_index = 0
        self._latest_result = SweepFrameResult()

    def reset(self) -> None:
        """Reset all sweep-progress state for a future run."""
        self._run_requested = False
        self._single_requested = False
        self._current_point_index = 0
        self._latest_result = SweepFrameResult()

    def stop(self) -> None:
        """Stop any future sweep progress requests."""
        self._run_requested = False
        self._single_requested = False

    def request_continuous(self) -> None:
        """Request continuous sweep execution."""
        self._run_requested = True
        self._single_requested = False

    def request_single(self) -> None:
        """Request one complete sweep execution."""
        self._run_requested = True
        self._single_requested = True

    def is_running(self) -> bool:
        """Return whether sweep execution is currently requested."""
        return self._run_requested

    def get_latest_result(self) -> SweepFrameResult:
        """Return the latest published sweep result snapshot."""
        return self._latest_result
