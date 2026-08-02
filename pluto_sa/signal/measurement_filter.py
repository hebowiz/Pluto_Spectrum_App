"""Stateful complex-IQ measurement filters used by swept and time modes."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.signal import butter, fftconvolve, lfilter, sosfilt, sosfilt_zi, sosfreqz

from pluto_sa.signal.detector import DetectorMode


DEFAULT_IQ_FILTER_ORDER = 4
DEFAULT_IQ_FILTER_SHAPE = "gaussian"
_MAX_TWO_SIDED_RBW_RATIO = 0.98
_GAUSSIAN_TRUNCATION_SIGMA = 4.0
_DIRECT_FIR_MAX_TAPS = 256


@dataclass(frozen=True)
class IQFilterDesign:
    """Resolved characteristics of a complex-baseband RBW filter.

    ``effective_rbw_hz`` is the full two-sided 3 dB bandwidth. The low-pass
    cutoff is consequently half that value on either side of DC. IIR group
    delay is frequency-dependent and is reported as ``None``.
    """

    sample_rate_hz: float
    requested_rbw_hz: float
    effective_rbw_hz: float
    cutoff_hz: float
    order: int
    noise_equivalent_bandwidth_hz: float
    settling_samples: int
    filter_shape: str
    tap_count: int
    group_delay_samples: float | None


@lru_cache(maxsize=128)
def _design_cached(
    sample_rate_hz: float,
    requested_rbw_hz: float,
    order: int,
    shape: str,
) -> tuple[np.ndarray, IQFilterDesign]:
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive")
    if not np.isfinite(requested_rbw_hz) or requested_rbw_hz <= 0.0:
        raise ValueError("rbw_hz must be positive")
    if int(order) <= 0:
        raise ValueError("order must be positive")
    if shape not in {"gaussian", "butterworth"}:
        raise ValueError("shape must be 'gaussian' or 'butterworth'")

    effective_rbw_hz = min(
        float(requested_rbw_hz),
        float(sample_rate_hz) * _MAX_TWO_SIDED_RBW_RATIO,
    )
    cutoff_hz = effective_rbw_hz / 2.0
    if shape == "gaussian":
        # A Gaussian impulse with this standard deviation has a power response
        # of -3.0103 dB at +/- RBW/2. Truncating at +/-4 sigma keeps the error
        # negligible while bounding processing cost.
        sigma_samples = (
            np.sqrt(np.log(2.0))
            * float(sample_rate_hz)
            / (np.pi * effective_rbw_hz)
        )
        half_width = max(
            1,
            int(np.ceil(_GAUSSIAN_TRUNCATION_SIGMA * sigma_samples)),
        )
        sample_indices = np.arange(-half_width, half_width + 1, dtype=np.float64)
        coefficients = np.exp(-0.5 * (sample_indices / sigma_samples) ** 2)
        coefficients /= np.sum(coefficients)
        enbw_hz = float(np.sum(coefficients**2) * float(sample_rate_hz))
        settling_samples = int(coefficients.size - 1)
        tap_count = int(coefficients.size)
        group_delay_samples = float(half_width)
    else:
        coefficients = butter(
            int(order),
            cutoff_hz,
            btype="lowpass",
            fs=float(sample_rate_hz),
            output="sos",
        )

        # Integrate the complete 0..Fs response. Its upper half represents the
        # negative-frequency side of this complex-baseband low-pass.
        _, response = sosfreqz(
            coefficients,
            worN=32_768,
            whole=True,
            fs=float(sample_rate_hz),
        )
        enbw_hz = float(np.mean(np.abs(response) ** 2) * float(sample_rate_hz))

        # Eight time constants is a conservative state/warm-up indicator. It is
        # metadata, not an automatic sample discard performed by the filter.
        settling_samples = max(
            int(order) * 4,
            int(np.ceil(8.0 * float(sample_rate_hz) / (np.pi * effective_rbw_hz))),
        )
        tap_count = 0
        group_delay_samples = None
    design = IQFilterDesign(
        sample_rate_hz=float(sample_rate_hz),
        requested_rbw_hz=float(requested_rbw_hz),
        effective_rbw_hz=effective_rbw_hz,
        cutoff_hz=cutoff_hz,
        order=int(order),
        noise_equivalent_bandwidth_hz=enbw_hz,
        settling_samples=settling_samples,
        filter_shape=shape,
        tap_count=tap_count,
        group_delay_samples=group_delay_samples,
    )
    coefficients.setflags(write=False)
    return coefficients, design


def design_iq_rbw_filter(
    sample_rate_hz: float,
    rbw_hz: float,
    order: int = DEFAULT_IQ_FILTER_ORDER,
    shape: str = DEFAULT_IQ_FILTER_SHAPE,
) -> tuple[np.ndarray, IQFilterDesign]:
    """Return coefficients and metadata for a centered IQ RBW filter.

    Gaussian coefficients are one-dimensional FIR taps. Butterworth
    coefficients use SciPy's second-order-section representation.
    """
    normalized_shape = str(shape).strip().lower()
    coefficients, design = _design_cached(
        float(sample_rate_hz),
        float(rbw_hz),
        int(order),
        normalized_shape,
    )
    return np.array(coefficients, copy=True), design


class StatefulIQMeasurementFilter:
    """Complex IQ low-pass whose state survives input block boundaries."""

    def __init__(
        self,
        sample_rate_hz: float,
        rbw_hz: float,
        order: int = DEFAULT_IQ_FILTER_ORDER,
        shape: str = DEFAULT_IQ_FILTER_SHAPE,
    ) -> None:
        self.coefficients, self.design = design_iq_rbw_filter(
            sample_rate_hz,
            rbw_hz,
            order,
            shape,
        )
        self.sos: np.ndarray | None = None
        self.taps: np.ndarray | None = None
        self._history: np.ndarray | None = None
        if self.design.filter_shape == "butterworth":
            self.sos = self.coefficients
            self._steady_state_zi = sosfilt_zi(self.sos).astype(np.complex128)
        else:
            self.taps = self.coefficients
            # lfilter uses a transposed direct-form state. For a constant input,
            # zi[k] is the sum of all taps following tap k.
            self._steady_state_zi = np.cumsum(self.taps[:0:-1])[::-1].astype(
                np.complex128
            )
            if self.taps.size > _DIRECT_FIR_MAX_TAPS:
                self._history = np.zeros(self.taps.size - 1, dtype=np.complex128)
        self._zi = np.zeros_like(self._steady_state_zi)

    def reset(self, initial_sample: complex | None = None) -> None:
        """Clear history, optionally assuming a steady input before this record."""
        if initial_sample is None:
            self._zi = np.zeros_like(self._steady_state_zi)
        else:
            self._zi = self._steady_state_zi * complex(initial_sample)
        if self._history is not None:
            fill = 0.0j if initial_sample is None else complex(initial_sample)
            self._history.fill(fill)

    def process(self, iq: np.ndarray) -> np.ndarray:
        """Filter one block and retain final state for the next block."""
        values = np.asarray(iq)
        if values.ndim != 1:
            raise ValueError("iq must be one-dimensional")
        if not np.issubdtype(values.dtype, np.complexfloating):
            raise ValueError("iq must contain complex samples")
        if values.size == 0:
            return np.empty(0, dtype=np.complex128)
        if self.sos is not None:
            output, self._zi = sosfilt(self.sos, values, zi=self._zi)
        elif self._history is None:
            output, self._zi = lfilter(
                self.taps,
                np.asarray([1.0]),
                values,
                zi=self._zi,
            )
        else:
            extended = np.concatenate((self._history, values.astype(np.complex128)))
            convolution = fftconvolve(extended, self.taps, mode="full")
            start = int(self.taps.size - 1)
            output = convolution[start : start + values.size]
            self._history = np.asarray(extended[-start:], dtype=np.complex128)
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


def reduce_filtered_iq_power_buckets(
    filtered_iq: np.ndarray,
    mode: DetectorMode | str,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce a filtered IQ record into nearly equal contiguous time buckets.

    Returns linear-power detector values and the fractional center sample index
    of every bucket. Every input sample belongs to exactly one bucket.
    """
    values = np.asarray(filtered_iq)
    if values.ndim != 1:
        raise ValueError("filtered_iq must be one-dimensional")
    if values.size == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    if int(max_points) <= 0:
        raise ValueError("max_points must be positive")

    sample_count = int(values.size)
    bucket_count = min(sample_count, int(max_points))
    edges = np.floor(
        np.arange(bucket_count + 1, dtype=np.float64)
        * float(sample_count)
        / float(bucket_count)
    ).astype(np.int64)
    edges[-1] = sample_count
    starts = edges[:-1]
    ends = edges[1:]
    lengths = ends - starts
    power = np.asarray(np.abs(values) ** 2, dtype=np.float64)

    resolved_mode = DetectorMode(mode)
    if resolved_mode is DetectorMode.SAMPLE:
        detector_values = power[ends - 1]
    elif resolved_mode is DetectorMode.PEAK:
        detector_values = np.maximum.reduceat(power, starts)
    else:
        detector_values = np.add.reduceat(power, starts) / lengths

    center_sample_indices = (starts.astype(np.float64) + ends.astype(np.float64) - 1.0) / 2.0
    return np.asarray(detector_values), center_sample_indices
