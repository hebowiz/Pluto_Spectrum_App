"""Calibration utility."""

from __future__ import annotations

import numpy as np


def apply_offset_db(values: np.ndarray, offset_db: float) -> np.ndarray:
    """Apply the configured dB offset to a spectrum trace."""
    return values + offset_db


def apply_display_power_correction(
    values: np.ndarray,
    calibration_offset_db: float,
    input_correction_db: float,
) -> np.ndarray:
    """Apply final display-stage power corrections after measured power is computed."""
    return values + calibration_offset_db + input_correction_db
