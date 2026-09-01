"""Waveform-level measurements used by Pluto RF-level control."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pluto_vsg.engine.base import GenerationResult


@dataclass(frozen=True)
class IQLevelMetrics:
    """Peak and active-interval RMS values relative to complex full scale."""

    peak: float
    peak_dbfs: float
    active_rms: float
    active_rms_dbfs: float
    crest_factor_db: float
    active_sample_count: int


def _amplitude_dbfs(value: float) -> float:
    return 20.0 * math.log10(value) if value > 0.0 else float("-inf")


def measure_iq_levels(
    iq: np.ndarray,
    active_ranges_samples: Iterable[tuple[int, int]] | None = None,
) -> IQLevelMetrics:
    """Measure complex magnitude without including scheduled idle samples."""

    samples = np.asarray(iq, dtype=np.complex128).reshape(-1)
    if active_ranges_samples is None:
        ranges = ((0, int(samples.size)),)
    else:
        ranges = tuple((int(start), int(stop)) for start, stop in active_ranges_samples)

    power_sum = 0.0
    peak = 0.0
    count = 0
    for start, stop in ranges:
        if start < 0 or stop < start or stop > samples.size:
            raise ValueError(
                f"Active IQ range [{start}, {stop}) is outside {samples.size} samples"
            )
        if start == stop:
            continue
        magnitude = np.abs(samples[start:stop])
        power_sum += float(np.sum(magnitude * magnitude, dtype=np.float64))
        peak = max(peak, float(np.max(magnitude)))
        count += int(stop - start)
    if count == 0:
        return IQLevelMetrics(0.0, float("-inf"), 0.0, float("-inf"), 0.0, 0)

    active_rms = math.sqrt(power_sum / count)
    peak_dbfs = _amplitude_dbfs(peak)
    active_rms_dbfs = _amplitude_dbfs(active_rms)
    return IQLevelMetrics(
        peak=peak,
        peak_dbfs=peak_dbfs,
        active_rms=active_rms,
        active_rms_dbfs=active_rms_dbfs,
        crest_factor_db=peak_dbfs - active_rms_dbfs,
        active_sample_count=count,
    )


def generation_result_iq_levels(result: GenerationResult) -> IQLevelMetrics:
    """Measure a result using generator-declared active intervals when present."""

    ranges = result.metadata.get("active_ranges_samples")
    if ranges is None:
        # Compatibility for older/external engines. Packet ranges are explicit
        # protocol intervals and are preferable to a magnitude threshold.
        ranges = result.metadata.get("packet_ranges_samples")
    return measure_iq_levels(result.iq, ranges)


def iq_level_metadata(metrics: IQLevelMetrics) -> dict[str, float | int]:
    """Return standardized JSON-safe metadata keys for a generated waveform."""

    return {
        "iq_peak": metrics.peak,
        "iq_peak_dbfs": metrics.peak_dbfs,
        "active_rms": metrics.active_rms,
        "active_rms_dbfs": metrics.active_rms_dbfs,
        "crest_factor_db": metrics.crest_factor_db,
        "active_sample_count": metrics.active_sample_count,
    }


__all__ = [
    "IQLevelMetrics",
    "generation_result_iq_levels",
    "iq_level_metadata",
    "measure_iq_levels",
]
