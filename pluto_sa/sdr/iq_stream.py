"""Thread-safe IQ block stream primitives shared by all analyzer modes."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
import time

import numpy as np


@dataclass(frozen=True)
class IQBlock:
    """One ordered block published by an IQ producer.

    The producer transfers ownership of ``iq`` to the stream. Consumers must
    treat the array as read-only.
    """

    sequence: int
    stream_id: int
    block_index: int
    start_sample_index: int
    iq: np.ndarray
    timestamp_s: float
    discontinuity_before: bool
    source: str
    capture_elapsed_s: float = 0.0

    @property
    def sample_count(self) -> int:
        return int(len(self.iq))

    @property
    def end_sample_index(self) -> int:
        return self.start_sample_index + self.sample_count


@dataclass(frozen=True)
class IQStreamCursor:
    """Immutable position of one independent stream consumer."""

    next_sequence: int


@dataclass(frozen=True)
class IQReadResult:
    """Blocks returned to a consumer together with overrun information."""

    blocks: tuple[IQBlock, ...]
    cursor: IQStreamCursor
    overrun: bool = False
    missed_blocks: int = 0


@dataclass(frozen=True)
class IQStreamStats:
    """Snapshot of producer-side stream counters."""

    stream_id: int
    published_blocks: int
    published_samples: int
    buffered_blocks: int
    overwritten_blocks: int


class IQStreamBuffer:
    """Bounded multi-consumer IQ block history.

    The buffer never silently hides a consumer overrun: if a cursor points to
    data that has already been overwritten, ``read`` reports the exact number
    of missed blocks still inferable from the global block sequence.
    """

    def __init__(self, capacity_blocks: int) -> None:
        if int(capacity_blocks) <= 0:
            raise ValueError("capacity_blocks must be positive")
        self.capacity_blocks = int(capacity_blocks)
        self._blocks: deque[IQBlock] = deque(maxlen=self.capacity_blocks)
        self._lock = threading.Lock()
        self._stream_id = 0
        self._next_sequence = 0
        self._next_block_index = 0
        self._next_sample_index = 0
        self._pending_discontinuity = True
        self._published_blocks = 0
        self._published_samples = 0
        self._overwritten_blocks = 0

    def begin_stream(self, *, clear: bool = False) -> int:
        """Start a new stream epoch and return its ID."""
        with self._lock:
            self._stream_id += 1
            self._next_block_index = 0
            self._next_sample_index = 0
            self._pending_discontinuity = True
            if clear:
                self._blocks.clear()
            return self._stream_id

    def mark_discontinuity(self) -> None:
        """Mark the next published block as discontinuous from its predecessor."""
        with self._lock:
            self._pending_discontinuity = True

    def publish(
        self,
        iq: np.ndarray,
        *,
        timestamp_s: float | None = None,
        source: str = "continuous",
        discontinuity_before: bool = False,
        capture_elapsed_s: float = 0.0,
    ) -> IQBlock:
        """Publish an owned one-dimensional complex IQ array."""
        iq_array = np.asarray(iq)
        if iq_array.ndim != 1:
            raise ValueError("iq must be a one-dimensional array")
        if not np.issubdtype(iq_array.dtype, np.complexfloating):
            raise TypeError("iq must use a complex dtype")
        if not iq_array.flags.c_contiguous:
            iq_array = np.ascontiguousarray(iq_array)
        resolved_timestamp = time.perf_counter() if timestamp_s is None else float(timestamp_s)

        with self._lock:
            if len(self._blocks) == self.capacity_blocks:
                self._overwritten_blocks += 1
            block = IQBlock(
                sequence=self._next_sequence,
                stream_id=self._stream_id,
                block_index=self._next_block_index,
                start_sample_index=self._next_sample_index,
                iq=iq_array,
                timestamp_s=resolved_timestamp,
                discontinuity_before=bool(
                    discontinuity_before or self._pending_discontinuity
                ),
                source=str(source),
                capture_elapsed_s=max(0.0, float(capture_elapsed_s)),
            )
            self._blocks.append(block)
            self._next_sequence += 1
            self._next_block_index += 1
            self._next_sample_index += block.sample_count
            self._pending_discontinuity = False
            self._published_blocks += 1
            self._published_samples += block.sample_count
            return block

    def create_cursor(self, *, start: str = "latest") -> IQStreamCursor:
        """Create a cursor at new, newest retained, or oldest retained data."""
        with self._lock:
            if start == "latest":
                next_sequence = self._next_sequence
            elif start == "newest":
                next_sequence = (
                    self._blocks[-1].sequence if self._blocks else self._next_sequence
                )
            elif start == "oldest":
                next_sequence = (
                    self._blocks[0].sequence if self._blocks else self._next_sequence
                )
            else:
                raise ValueError("start must be 'latest', 'newest', or 'oldest'")
            return IQStreamCursor(next_sequence=next_sequence)

    def read(
        self,
        cursor: IQStreamCursor,
        *,
        max_blocks: int | None = None,
    ) -> IQReadResult:
        """Read retained blocks at or after ``cursor`` without removing them."""
        if max_blocks is not None and int(max_blocks) <= 0:
            raise ValueError("max_blocks must be positive when provided")

        with self._lock:
            oldest_sequence = (
                self._blocks[0].sequence if self._blocks else self._next_sequence
            )
            requested_sequence = int(cursor.next_sequence)
            missed_blocks = max(0, oldest_sequence - requested_sequence)
            effective_sequence = max(requested_sequence, oldest_sequence)
            blocks = [
                block for block in self._blocks if block.sequence >= effective_sequence
            ]
            if max_blocks is not None:
                blocks = blocks[: int(max_blocks)]
            next_sequence = blocks[-1].sequence + 1 if blocks else effective_sequence
            return IQReadResult(
                blocks=tuple(blocks),
                cursor=IQStreamCursor(next_sequence=next_sequence),
                overrun=missed_blocks > 0,
                missed_blocks=missed_blocks,
            )

    def latest_samples(self, sample_count: int) -> np.ndarray | None:
        """Return the latest contiguous samples from the current stream epoch."""
        requested = int(sample_count)
        if requested <= 0:
            raise ValueError("sample_count must be positive")

        with self._lock:
            selected: list[np.ndarray] = []
            available = 0
            expected_start = self._next_sample_index
            for block in reversed(self._blocks):
                if block.stream_id != self._stream_id:
                    break
                if block.end_sample_index != expected_start:
                    break
                selected.append(block.iq)
                available += block.sample_count
                expected_start = block.start_sample_index
                if available >= requested:
                    break
                if block.discontinuity_before:
                    break
            if available < requested:
                return None
            selected.reverse()
            if len(selected) == 1:
                return np.asarray(selected[0][-requested:]).copy()
            return np.concatenate(selected)[-requested:].copy()

    def stats(self) -> IQStreamStats:
        with self._lock:
            return IQStreamStats(
                stream_id=self._stream_id,
                published_blocks=self._published_blocks,
                published_samples=self._published_samples,
                buffered_blocks=len(self._blocks),
                overwritten_blocks=self._overwritten_blocks,
            )
