"""Circular prestore and poststore assembly for triggered IQ records."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pluto_sa.sdr.iq_stream import IQBlock
from pluto_sa.sdr.trigger import (
    AcquisitionMetadata,
    IQAcquisitionRecord,
    TriggerConfig,
    TriggerEvent,
    TriggerRearmMode,
    TriggerRunMode,
)


@dataclass(frozen=True)
class _PendingRecord:
    event: TriggerEvent
    start_sample_index: int
    end_sample_index: int


class TriggeredIQRecorder:
    """Build exact immutable records around trigger events from contiguous IQ."""

    def __init__(
        self,
        config: TriggerConfig,
        metadata: AcquisitionMetadata,
        *,
        event_latency_samples: int | None = None,
    ) -> None:
        self.config = config
        self.metadata = metadata
        resolved_latency = (
            int(config.minimum_duration_samples) - 1
            if event_latency_samples is None
            else int(event_latency_samples)
        )
        if resolved_latency < 0:
            raise ValueError("event_latency_samples must be non-negative")
        self.event_latency_samples = resolved_latency
        self.rejected_prestore_events = 0
        self.dropped_pending_records = 0
        self.stopped = False
        self._stream_id: int | None = None
        self._expected_sample_index: int | None = None
        self._buffer_start_sample_index = 0
        self._buffer = np.empty(0, dtype=np.complex64)
        self._pending: list[_PendingRecord] = []

    @property
    def pending_records(self) -> int:
        return len(self._pending)

    def reset(self) -> None:
        self.stopped = False
        self._clear_timeline()

    def feed(
        self,
        block: IQBlock,
        events: tuple[TriggerEvent, ...] | list[TriggerEvent] = (),
    ) -> tuple[IQAcquisitionRecord, ...]:
        if self.stopped:
            return ()
        if block.source != self.metadata.source:
            raise ValueError("block source must match acquisition metadata")

        contiguous = (
            self._stream_id == block.stream_id
            and self._expected_sample_index == block.start_sample_index
            and not block.discontinuity_before
        )
        if not contiguous:
            self.dropped_pending_records += len(self._pending)
            self._clear_timeline()
            self._stream_id = block.stream_id
            self._buffer_start_sample_index = block.start_sample_index

        block_iq = np.asarray(block.iq, dtype=np.complex64)
        if len(self._buffer) == 0:
            self._buffer = block_iq.copy()
            self._buffer_start_sample_index = block.start_sample_index
        else:
            self._buffer = np.concatenate((self._buffer, block_iq))
        self._stream_id = block.stream_id
        self._expected_sample_index = block.end_sample_index

        for event in sorted(events, key=lambda item: item.sample_index):
            self._accept_event(block, event)

        records: list[IQAcquisitionRecord] = []
        buffer_end = self._buffer_start_sample_index + len(self._buffer)
        remaining: list[_PendingRecord] = []
        for pending in self._pending:
            if pending.end_sample_index > buffer_end:
                remaining.append(pending)
                continue
            start_offset = pending.start_sample_index - self._buffer_start_sample_index
            stop_offset = pending.end_sample_index - self._buffer_start_sample_index
            record_iq = self._buffer[start_offset:stop_offset].copy()
            records.append(
                IQAcquisitionRecord(
                    stream_id=block.stream_id,
                    start_sample_index=pending.start_sample_index,
                    trigger_sample_index=pending.event.sample_index,
                    iq=record_iq,
                    trigger=pending.event,
                    config=self.config,
                    metadata=self.metadata,
                )
            )
        self._pending = remaining

        if records and (
            self.config.run_mode == TriggerRunMode.SINGLE
            or self.config.rearm_mode == TriggerRearmMode.STOP_ON_TRIGGER
        ):
            self.stopped = True
            self._pending = []

        self._trim_buffer()
        return tuple(records[:1] if self.stopped else records)

    def _accept_event(self, block: IQBlock, event: TriggerEvent) -> None:
        if event.stream_id != block.stream_id:
            raise ValueError("event and block stream_id must match")
        buffer_end = self._buffer_start_sample_index + len(self._buffer)
        if event.sample_index < self._buffer_start_sample_index or event.sample_index >= buffer_end:
            raise ValueError("event must be inside retained IQ data")

        start = event.sample_index - int(self.config.pretrigger_samples)
        end = event.sample_index + int(self.config.posttrigger_samples) + 1
        if start < self._buffer_start_sample_index:
            self.rejected_prestore_events += 1
            return
        self._pending.append(
            _PendingRecord(
                event=event,
                start_sample_index=start,
                end_sample_index=end,
            )
        )

    def _trim_buffer(self) -> None:
        buffer_end = self._buffer_start_sample_index + len(self._buffer)
        if self.stopped:
            keep_from = buffer_end
        elif self._pending:
            keep_from = min(item.start_sample_index for item in self._pending)
        else:
            keep_from = max(
                self._buffer_start_sample_index,
                buffer_end
                - int(self.config.pretrigger_samples)
                - self.event_latency_samples,
            )
        trim = max(0, keep_from - self._buffer_start_sample_index)
        if trim > 0:
            self._buffer = self._buffer[trim:].copy()
            self._buffer_start_sample_index += trim

    def _clear_timeline(self) -> None:
        self._stream_id = None
        self._expected_sample_index = None
        self._buffer_start_sample_index = 0
        self._buffer = np.empty(0, dtype=np.complex64)
        self._pending = []
