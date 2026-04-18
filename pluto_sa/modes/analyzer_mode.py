"""Analyzer mode definitions."""

from __future__ import annotations

from enum import Enum


class AnalyzerMode(str, Enum):
    """Supported analyzer operation modes."""

    REALTIME_SA = "RealTime SA"
    WIDEBAND_REALTIME_SA = "WideBand RT SA"
    SWEEP_SA = "Sweep SA"
