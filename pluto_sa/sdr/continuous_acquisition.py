"""Shared lifecycle for gap-aware continuous IQ acquisition.

The receiver owns the libiio producer thread and :class:`IQStreamBuffer`.
This class gives RTSA, HSTA, and VSA one common owner for starting that
producer, creating independent consumer cursors, and stopping/reconfiguring
it without mode-specific buffer handling.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading

from pluto_sa.sdr.iq_stream import IQReadResult, IQStreamCursor


def resolve_record_stream_block_samples(
    record_samples: int,
    *,
    base_block_samples: int = 65_536,
    island_max_samples: int = 262_144,
    max_records_per_block: int | None = None,
) -> int:
    """Resolve a large RX block containing an integer number of records.

    A pyadi/libiio v0 receive loop refills one userspace buffer at a time.
    Keeping a short burst train inside a larger buffer island materially lowers
    its exposure to a host refill boundary.  Integer record multiples also
    avoid adding an artificial boundary inside a nominal finite record.
    """

    record = max(1, int(record_samples))
    base = max(1, int(base_block_samples))
    island_max = max(base, int(island_max_samples))
    if record > island_max:
        return record
    if max_records_per_block is not None:
        records = min(
            max(1, int(max_records_per_block)),
            max(1, (base + record - 1) // record),
        )
        return record * records
    records = max(1, island_max // record)
    block = record * records
    if block >= base:
        return block
    records = max(1, (base + record - 1) // record)
    return record * records


@dataclass(frozen=True)
class ContinuousIQStreamPlan:
    block_size: int
    source: str
    max_blocks: int | None = None

    def __post_init__(self) -> None:
        if int(self.block_size) <= 0:
            raise ValueError("block_size must be positive")
        if self.max_blocks is not None and int(self.max_blocks) <= 0:
            raise ValueError("max_blocks must be positive when provided")


class ContinuousIQAcquisition:
    """Coordinate one reusable receiver stream for independent consumers.

    ``fresh`` only applies when a producer is actually started.  Asking for a
    new cursor on an already compatible stream never destroys the IIO buffer;
    this is what lets a VSA re-arm without introducing a receive blind time.
    """

    def __init__(self, receiver) -> None:
        self.receiver = receiver
        self._plan: ContinuousIQStreamPlan | None = None
        self._running = False
        self._lock = threading.RLock()

    @property
    def plan(self) -> ContinuousIQStreamPlan | None:
        with self._lock:
            return self._plan

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._running and self._receiver_is_streaming()

    def start(
        self,
        *,
        block_size: int,
        source: str = "continuous",
        max_blocks: int | None = None,
        fresh: bool = False,
        cursor_start: str = "latest",
    ) -> IQStreamCursor:
        plan = ContinuousIQStreamPlan(
            block_size=max(1, int(block_size)),
            source=str(source),
            max_blocks=None if max_blocks is None else int(max_blocks),
        )
        with self._lock:
            if self._running and not self._receiver_is_streaming():
                self._running = False
                self._plan = None
            if self._running and self._plan == plan:
                return self.receiver.create_iq_stream_cursor(start=cursor_start)
            if self._running and not self._stop_locked():
                raise RuntimeError(
                    "continuous IQ acquisition did not stop before restart"
                )
            initial_cursor = self.receiver.start(
                block_size=plan.block_size,
                source=plan.source,
                max_blocks=plan.max_blocks,
                fresh=bool(fresh),
            )
            self._plan = plan
            self._running = True
            if cursor_start == "latest":
                return initial_cursor
            return self.receiver.create_iq_stream_cursor(start=cursor_start)

    def cursor(self, *, start: str = "latest") -> IQStreamCursor:
        with self._lock:
            if not self._running:
                raise RuntimeError("continuous IQ acquisition is not running")
            return self.receiver.create_iq_stream_cursor(start=start)

    def read(
        self,
        cursor: IQStreamCursor,
        *,
        max_blocks: int | None = None,
    ) -> IQReadResult:
        return self.receiver.read_iq_stream(cursor, max_blocks=max_blocks)

    def latest_samples(self, sample_count: int):
        if int(sample_count) <= 0:
            raise ValueError("sample_count must be positive")
        return self.receiver.iq_stream.latest_samples(int(sample_count))

    def reconfigure(self, config, *, restart: bool = True, fresh: bool = True) -> None:
        with self._lock:
            plan = self._plan if self._running and restart else None
            if self._running and not self._stop_locked():
                raise RuntimeError(
                    "continuous IQ acquisition did not stop before reconfigure"
                )
            self.receiver.reconfigure(config)
            if plan is not None:
                self.receiver.start(
                    block_size=plan.block_size,
                    source=plan.source,
                    max_blocks=plan.max_blocks,
                    fresh=bool(fresh),
                )
                self._plan = plan
                self._running = True

    def stop(self) -> bool:
        with self._lock:
            return self._stop_locked()

    def _stop_locked(self) -> bool:
        stopped = bool(self.receiver.stop())
        if stopped:
            self._running = False
            self._plan = None
        return stopped

    def _receiver_is_streaming(self) -> bool:
        checker = getattr(self.receiver, "is_streaming", None)
        if checker is None:
            return self._running
        return bool(checker())
