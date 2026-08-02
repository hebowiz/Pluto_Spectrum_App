"""Stateful complex-IQ measurement filters used by swept and time modes."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi, sosfreqz

from pluto_sa.signal.detector import DetectorMode


DEFAULT_IQ_FILTER_ORDER = 4
_MAX_TWO_SIDED_RBW_RATIO = 0.98


@dataclass(frozen=True)
class IQFilterDesign:
    """Resolved characteristics of a complex-baseband RBW filter.

    ``effective_rbw_hz`` is the full two-sided 3 dB bandwidth. The SciPy
    low-pass cutoff is consequently half that value on either side of DC.
    """

    sample_rate_hz: float
    requested_rbw_hz: float
    effective_rbw_hz: float
    cutoff_hz: float
    order: int
    noise_equivalent_bandwidth_hz: float
    settling_samples: int


@lru_cache(maxsize=128)
def _design_cached(
    sample_rate_hz: float,
    requested_rbw_hz: float,
    order: int,
) -> tuple[np.ndarray, IQFilterDesign]:
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive")
    if not np.isfinite(requested_rbw_hz) or requested_rbw_hz <= 0.0:
        raise ValueError("rbw_hz must be positive")
    if int(order) <= 0:
        raise ValueError("order must be positive")

    effective_rbw_hz = min(
        float(requested_rbw_hz),
        float(sample_rate_hz) * _MAX_TWO_SIDED_RBW_RATIO,
    )
    cutoff_hz = effective_rbw_hz / 2.0
    sos = butter(
        int(order),
        cutoff_hz,
        btype="lowpass",
        fs=float(sample_rate_hz),
        output="sos",
    )

    # Integrate the complete 0..Fs response. Its upper half represents the
    # negative-frequency side of this complex-baseband low-pass.
    _, response = sosfreqz(sos, worN=32_768, whole=True, fs=float(sample_rate_hz))
    enbw_hz = float(np.mean(np.abs(response) ** 2) * float(sample_rate_hz))

    # Eight time constants is a conservative state/warm-up indicator. It is
    # metadata, not an automatic sample discard performed by the filter.
    settling_samples = max(
        int(order) * 4,
        int(np.ceil(8.0 * float(sample_rate_hz) / (np.pi * effective_rbw_hz))),
    )
    design = IQFilterDesign(
        sample_rate_hz=float(sample_rate_hz),
        requested_rbw_hz=float(requested_rbw_hz),
        effective_rbw_hz=effective_rbw_hz,
        cutoff_hz=cutoff_hz,
        order=int(order),
        noise_equivalent_bandwidth_hz=enbw_hz,
        settling_samples=settling_samples,
    )
    sos.setflags(write=False)
    return sos, design


def design_iq_rbw_filter(
    sample_rate_hz: float,
    rbw_hz: float,
    order: int = DEFAULT_IQ_FILTER_ORDER,
) -> tuple[np.ndarray, IQFilterDesign]:
    """Return SOS coefficients and metadata for a centered IQ RBW filter."""
    sos, design = _design_cached(float(sample_rate_hz), float(rbw_hz), int(order))
    return np.array(sos, copy=True), design


class StatefulIQMeasurementFilter:
    """Complex Butterworth low-pass whose state survives input block boundaries."""

    def __init__(
        self,
        sample_rate_hz: float,
        rbw_hz: float,
        order: int = DEFAULT_IQ_FILTER_ORDER,
    ) -> None:
        self.sos, self.design = design_iq_rbw_filter(sample_rate_hz, rbw_hz, order)
        self._steady_state_zi = sosfilt_zi(self.sos).astype(np.complex128)
        self._zi = np.zeros_like(self._steady_state_zi)

    def reset(self, initial_sample: complex | None = None) -> None:
        """Clear history, optionally assuming a steady input before this record."""
        if initial_sample is None:
            self._zi = np.zeros_like(self._steady_state_zi)
        else:
            self._zi = self._steady_state_zi * complex(initial_sample)

    def process(self, iq: np.ndarray) -> np.ndarray:
        """Filter one block and retain final state for the next block."""
        values = np.asarray(iq)
        if values.ndim != 1:
            raise ValueError("iq must be one-dimensional")
        if not np.issubdtype(values.dtype, np.complexfloating):
            raise ValueError("iq must contain complex samples")
        if values.size == 0:
            return np.empty(0, dtype=np.complex128)
        output, self._zi = sosfilt(self.sos, values, zi=self._zi)
        return np.asarray(output, dtype=np.complex128)


def reduce_filtered_iq_power(
    filtered_iq: np.ndarray,
    mode: DetectorMode | str,
    *,
    axis: int | None = None,
) -> np.ndarray | float:
    """Detect filtered IQ using spectrum-analyzer power semantics.

    RMS means mean-square voltage, so in linear power units it is the mean of
    ``abs(iq)**2`` rather than the RMS of already-squared power values.
    """
    values = np.asarray(filtered_iq)
    if values.size == 0:
        raise ValueError("detector input must not be empty")
    power = np.abs(values) ** 2
    resolved_mode = DetectorMode(mode)
    if resolved_mode is DetectorMode.SAMPLE:
        result = np.take(power, indices=-1, axis=axis)
    elif resolved_mode is DetectorMode.PEAK:
        result = np.max(power, axis=axis)
    else:
        result = np.mean(power, axis=axis)
    if np.ndim(result) == 0:
        return float(result)
    return np.asarray(result, dtype=np.float64)
