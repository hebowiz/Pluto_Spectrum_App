"""Exact sample-count windows assembled from common IQ stream blocks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pluto_sa.sdr.iq_stream import IQBlock


def resolve_fft_aligned_window_samples(
    time_span_s: float,
    sample_rate_hz: float,
    fft_size: int,
) -> int:
    """Return the smallest whole-FFT record covering the requested duration."""
    if float(time_span_s) <= 0.0:
        raise ValueError("time_span_s must be positive")
    if float(sample_rate_hz) <= 0.0:
        raise ValueError("sample_rate_hz must be positive")
    if int(fft_size) <= 0:
        raise ValueError("fft_size must be positive")
    requested = max(1, int(round(float(time_span_s) * float(sample_rate_hz))))
    fft_n = int(fft_size)
    return ((requested + fft_n - 1) // fft_n) * fft_n


@dataclass(frozen=True)
class IQWindow:
    """One contiguous, exact-length IQ analysis window."""

    stream_id: int
    start_sample_index: int
    iq: np.ndarray
    discontinuity_before: bool
    first_sequence: int
    last_sequence: int
    source: str

    @property
    def sample_count(self) -> int:
        return int(len(self.iq))

    @property
    def end_sample_index(self) -> int:
        return self.start_sample_index + self.sample_count


class IQWindowAssembler:
    """Split ordered IQ blocks into exact windows while carrying block tails."""

    def __init__(self, window_samples: int) -> None:
        if int(window_samples) <= 0:
            raise ValueError("window_samples must be positive")
        self.window_samples = int(window_samples)
        self.discarded_partial_samples = 0
        self._parts: list[np.ndarray] = []
        self._sample_count = 0
        self._stream_id: int | None = None
        self._start_sample_index = 0
        self._expected_sample_index: int | None = None
        self._first_sequence = 0
        self._last_sequence = 0
        self._source = ""
        self._discontinuity_before = True

    @property
    def pending_samples(self) -> int:
        return self._sample_count

    def reset(self) -> None:
        """Discard a partial window and require a new contiguous boundary."""
        self.discarded_partial_samples += self._sample_count
        self._parts = []
        self._sample_count = 0
        self._stream_id = None
        self._expected_sample_index = None
        self._discontinuity_before = True

    def feed(self, block: IQBlock) -> tuple[IQWindow, ...]:
        """Consume one block and return every exact window completed by it."""
        contiguous = (
            self._stream_id == block.stream_id
            and self._expected_sample_index == block.start_sample_index
            and not block.discontinuity_before
            and (self._sample_count == 0 or self._source == block.source)
        )
        if not contiguous:
            if self._sample_count > 0:
                self.reset()
            self._stream_id = block.stream_id
            self._expected_sample_index = block.start_sample_index
            self._discontinuity_before = True

        windows: list[IQWindow] = []
        offset = 0
        while offset < block.sample_count:
            if self._sample_count == 0:
                self._start_sample_index = block.start_sample_index + offset
                self._first_sequence = block.sequence
                self._source = block.source

            take = min(
                self.window_samples - self._sample_count,
                block.sample_count - offset,
            )
            self._parts.append(block.iq[offset : offset + take])
            self._sample_count += take
            offset += take
            self._expected_sample_index = block.start_sample_index + offset
            self._last_sequence = block.sequence

            if self._sample_count != self.window_samples:
                continue

            if len(self._parts) == 1:
                iq = self._parts[0].copy()
            else:
                iq = np.concatenate(self._parts)
            windows.append(
                IQWindow(
                    stream_id=int(self._stream_id),
                    start_sample_index=self._start_sample_index,
                    iq=iq,
                    discontinuity_before=self._discontinuity_before,
                    first_sequence=self._first_sequence,
                    last_sequence=self._last_sequence,
                    source=self._source,
                )
            )
            self._parts = []
            self._sample_count = 0
            self._discontinuity_before = False

        return tuple(windows)
