"""Nominal sample-rate classification for Wi-Fi Non-HT IQ."""
from __future__ import annotations

from dataclasses import dataclass
import math


# Pluto metadata can differ slightly from the requested hardware rate because
# of integer quantization or rounding (for example, 39_999_999 Hz for 40 MS/s).
NON_HT_SAMPLE_RATE_TOLERANCE_HZ = 100.0


@dataclass(frozen=True)
class NonHTSampleRate:
    nominal_sample_rate_hz: float
    decimation_factor: int


def resolve_non_ht_sample_rate(sample_rate_hz: float) -> NonHTSampleRate | None:
    """Classify a measured rate as nominal 20 or 40 MS/s, if supported."""
    sample_rate_hz = float(sample_rate_hz)
    if not math.isfinite(sample_rate_hz):
        return None
    for nominal_sample_rate_hz, decimation_factor in (
        (20_000_000.0, 1),
        (40_000_000.0, 2),
    ):
        if math.isclose(
            sample_rate_hz,
            nominal_sample_rate_hz,
            rel_tol=0.0,
            abs_tol=NON_HT_SAMPLE_RATE_TOLERANCE_HZ,
        ):
            return NonHTSampleRate(nominal_sample_rate_hz, decimation_factor)
    return None
