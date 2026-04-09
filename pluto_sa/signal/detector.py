"""Sweep SA detector definitions."""

from __future__ import annotations

from enum import Enum

import numpy as np


class DetectorMode(str, Enum):
    """Representative-value detector modes for sweep points."""

    SAMPLE = "Sample"
    PEAK = "Peak"
    RMS = "RMS"


def apply_detector(values: np.ndarray, mode: DetectorMode | str) -> float:
    """Reduce a data series to one representative detector value."""
    if values.size == 0:
        raise ValueError("detector input must not be empty")

    resolved_mode = DetectorMode(mode)

    if resolved_mode is DetectorMode.SAMPLE:
        return float(values[-1])
    if resolved_mode is DetectorMode.PEAK:
        return float(np.max(values))
    if resolved_mode is DetectorMode.RMS:
        return float(np.sqrt(np.mean(np.square(values))))

    raise ValueError(f"unsupported detector mode: {mode}")
