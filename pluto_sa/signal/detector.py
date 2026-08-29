"""Sweep SA detector definitions."""

from __future__ import annotations

from enum import Enum

import numpy as np


class DetectorMode(str, Enum):
    """Representative-value detector modes for sweep points."""

    SAMPLE = "Sample"
    PEAK = "Peak"
    NEGATIVE_PEAK = "Negative Peak"
    AVERAGE = "Average"
    RMS = "RMS"


def apply_detector(values: np.ndarray, mode: DetectorMode | str) -> float:
    """Reduce a linear-power series to one representative detector value."""
    if values.size == 0:
        raise ValueError("detector input must not be empty")

    resolved_mode = DetectorMode(mode)

    if resolved_mode is DetectorMode.SAMPLE:
        return float(values[-1])
    if resolved_mode is DetectorMode.PEAK:
        return float(np.max(values))
    if resolved_mode is DetectorMode.NEGATIVE_PEAK:
        return float(np.min(values))
    if resolved_mode in (DetectorMode.AVERAGE, DetectorMode.RMS):
        # Input is already squared voltage (linear power). An RMS voltage
        # detector therefore reports its arithmetic mean in power units.
        return float(np.mean(values))

    raise ValueError(f"unsupported detector mode: {mode}")
