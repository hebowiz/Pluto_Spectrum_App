"""Continuous overlap-FFT and display-frame detector processing for RTSA."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from pluto_sa.signal.detector import DetectorMode
from pluto_sa.signal.spectrum_processor import SpectrumProcessor


@dataclass(frozen=True)
class RealtimeFFTPlan:
    fft_size: int
    window_length_samples: int
    hop_samples: int
    target_overlap_ratio: float
    actual_overlap_ratio: float
    target_fft_rate_hz: float
    actual_fft_rate_hz: float
    analysis_coverage_ratio: float
    quality: str


@dataclass(frozen=True)
class RealtimeDetectorFrame:
    power_linear: np.ndarray
    fft_frames: int
    input_samples: int
    discontinuities: int
    plan: RealtimeFFTPlan


def build_realtime_fft_plan(
    *,
    sample_rate_hz: float,
    fft_size: int,
    window_length_samples: int | None = None,
    target_overlap_ratio: float = 0.8,
    max_fft_rate_hz: float = 10_000.0,
) -> RealtimeFFTPlan:
    """Resolve a best-effort hop for an RTSA analysis filter bank.

    ``fft_size`` determines the frequency grid. ``window_length_samples`` is
    the non-zero time support of the RBW analysis filter and therefore the
    quantity that determines temporal overlap and observation coverage.
    Keeping these concepts separate is essential when a short RBW window is
    zero-padded to a larger FFT for display-bin density.
    """
    rate = float(sample_rate_hz)
    size = int(fft_size)
    window_length = size if window_length_samples is None else int(window_length_samples)
    overlap = float(target_overlap_ratio)
    if rate <= 0.0 or size <= 0 or window_length <= 0:
        raise ValueError("sample_rate_hz, fft_size, and window_length_samples must be positive")
    if window_length > size:
        raise ValueError("window_length_samples must not exceed fft_size")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("target_overlap_ratio must be in [0, 1)")
    if float(max_fft_rate_hz) <= 0.0:
        raise ValueError("max_fft_rate_hz must be positive")

    target_hop = max(1, int(round(window_length * (1.0 - overlap))))
    rate_limited_hop = max(1, int(math.ceil(rate / float(max_fft_rate_hz))))
    hop = max(target_hop, rate_limited_hop)
    actual_overlap = max(0.0, 1.0 - (float(hop) / float(window_length)))
    coverage = min(1.0, float(window_length) / float(hop))
    if hop <= target_hop:
        quality = "Real-time"
    elif hop <= window_length:
        quality = "Reduced overlap"
    else:
        quality = "Analysis gaps"
    return RealtimeFFTPlan(
        fft_size=size,
        window_length_samples=window_length,
        hop_samples=hop,
        target_overlap_ratio=overlap,
        actual_overlap_ratio=actual_overlap,
        target_fft_rate_hz=rate / float(target_hop),
        actual_fft_rate_hz=rate / float(hop),
        analysis_coverage_ratio=coverage,
        quality=quality,
    )


class RealtimeFFTAccumulator:
    """Turn ordered IQ blocks into overlap FFTs and one detector display frame."""

    def __init__(
        self,
        processor: SpectrumProcessor,
        detector_mode: DetectorMode | str,
        *,
        target_overlap_ratio: float = 0.8,
        max_fft_rate_hz: float = 10_000.0,
        max_batch_frames: int = 256,
    ) -> None:
        self.processor = processor
        self.plan = build_realtime_fft_plan(
            sample_rate_hz=float(processor.config.sample_rate_hz),
            fft_size=int(processor.config.fft_size),
            window_length_samples=int(processor.filterbank_design.support_samples),
            target_overlap_ratio=target_overlap_ratio,
            max_fft_rate_hz=max_fft_rate_hz,
        )
        self.detector_mode = DetectorMode(detector_mode)
        self.max_batch_frames = max(1, int(max_batch_frames))
        self._pending = np.empty(0, dtype=np.complex64)
        self._skip_samples = 0
        self._aggregate: np.ndarray | None = None
        self._aggregate_count = 0
        self._input_samples = 0
        self._discontinuities = 0
        self._has_seen_input = False
        self._expected_sample_index: int | None = None

    @property
    def has_seen_input(self) -> bool:
        return self._has_seen_input

    @property
    def expected_sample_index(self) -> int | None:
        """Next stream sample index expected by this consumer, when known."""
        return self._expected_sample_index

    def set_detector_mode(self, detector_mode: DetectorMode | str) -> None:
        resolved = DetectorMode(detector_mode)
        if resolved == self.detector_mode:
            return
        self.detector_mode = resolved
        self._aggregate = None
        self._aggregate_count = 0

    def mark_discontinuity(self) -> None:
        if self._has_seen_input:
            self._discontinuities += 1
        self._pending = np.empty(0, dtype=np.complex64)
        self._skip_samples = 0
        self._expected_sample_index = None

    def seal_contiguous_region(self) -> None:
        """Prevent a future FFT from crossing the current input boundary.

        Pluto/libiio does not provide a hardware sample timestamp for every
        returned RX buffer. A software-contiguous sample index proves ordering,
        but cannot prove that no DMA/USB samples were lost between two refills.
        Discard the partial FFT tail at that boundary so a possible phase jump
        cannot be converted into a broadband spectrum.

        This is a conservative analysis boundary rather than evidence of a
        confirmed loss, so it does not increment the discontinuity counter.
        """
        self._pending = np.empty(0, dtype=np.complex64)
        self._skip_samples = 0
        self._expected_sample_index = None

    def process(
        self,
        iq: np.ndarray,
        *,
        discontinuity_before: bool = False,
        start_sample_index: int | None = None,
    ) -> int:
        """Consume one contiguous IQ block.

        A producer discontinuity flag or a jump in ``start_sample_index``
        discards all partial FFT/overlap state.  Consequently, no FFT window
        can contain samples from both sides of a known time gap.
        """
        samples = np.asarray(iq)
        if samples.ndim != 1 or not np.issubdtype(samples.dtype, np.complexfloating):
            raise ValueError("iq must be one-dimensional complex samples")
        resolved_start = (
            None if start_sample_index is None else int(start_sample_index)
        )
        index_discontinuity = bool(
            resolved_start is not None
            and self._expected_sample_index is not None
            and resolved_start != self._expected_sample_index
        )
        if discontinuity_before or index_discontinuity:
            self.mark_discontinuity()
        self._has_seen_input = True
        self._input_samples += int(len(samples))
        if resolved_start is not None:
            self._expected_sample_index = resolved_start + int(len(samples))
        if len(samples) == 0:
            return 0
        samples = samples.astype(np.complex64, copy=False)
        if self._skip_samples > 0:
            skipped = min(self._skip_samples, len(samples))
            self._skip_samples -= skipped
            samples = samples[skipped:]
            if len(samples) == 0:
                return 0
        combined = samples if len(self._pending) == 0 else np.concatenate((self._pending, samples))
        size = self.plan.fft_size
        hop = self.plan.hop_samples
        if len(combined) < size:
            self._pending = combined.copy()
            return 0

        frame_count = 1 + (len(combined) - size) // hop
        processed = 0
        starts = np.arange(frame_count, dtype=np.int64) * hop
        for batch_start in range(0, frame_count, self.max_batch_frames):
            batch_starts = starts[batch_start : batch_start + self.max_batch_frames]
            frames = np.stack([combined[start : start + size] for start in batch_starts])
            powers = self.processor.compute_filtered_power_batch(frames)
            self._accumulate(powers)
            processed += int(len(powers))
        next_start = frame_count * hop
        if next_start <= len(combined):
            self._pending = combined[next_start:].copy()
        else:
            self._pending = np.empty(0, dtype=np.complex64)
            self._skip_samples = next_start - len(combined)
        return processed

    def _accumulate(self, powers: np.ndarray) -> None:
        if len(powers) == 0:
            return
        mode = self.detector_mode
        if mode is DetectorMode.SAMPLE:
            batch_value = powers[-1].copy()
        elif mode is DetectorMode.PEAK:
            batch_value = np.max(powers, axis=0)
        elif mode is DetectorMode.NEGATIVE_PEAK:
            batch_value = np.min(powers, axis=0)
        else:
            batch_value = np.sum(powers, axis=0)

        if self._aggregate is None:
            self._aggregate = batch_value
        elif mode is DetectorMode.SAMPLE:
            self._aggregate = batch_value
        elif mode is DetectorMode.PEAK:
            np.maximum(self._aggregate, batch_value, out=self._aggregate)
        elif mode is DetectorMode.NEGATIVE_PEAK:
            np.minimum(self._aggregate, batch_value, out=self._aggregate)
        else:
            self._aggregate += batch_value
        self._aggregate_count += int(len(powers))

    def take_frame(self) -> RealtimeDetectorFrame | None:
        if self._aggregate is None or self._aggregate_count <= 0:
            return None
        power = self._aggregate
        if self.detector_mode in (DetectorMode.AVERAGE, DetectorMode.RMS):
            power = power / float(self._aggregate_count)
        result = RealtimeDetectorFrame(
            power_linear=np.asarray(power).copy(),
            fft_frames=int(self._aggregate_count),
            input_samples=int(self._input_samples),
            discontinuities=int(self._discontinuities),
            plan=self.plan,
        )
        self._aggregate = None
        self._aggregate_count = 0
        self._input_samples = 0
        self._discontinuities = 0
        return result
