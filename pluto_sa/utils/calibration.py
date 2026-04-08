"""Calibration utility."""

from __future__ import annotations

import numpy as np


def apply_offset_db(values: np.ndarray, offset_db: float) -> np.ndarray:
    """Apply the configured dB offset to a spectrum trace."""
    return values + offset_db
