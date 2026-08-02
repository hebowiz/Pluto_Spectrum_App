"""Trigger acquisition controller for continuous IQ consumers."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from pluto_sa.sdr.iq_stream import IQBlock
from pluto_sa.sdr.trigger import (
    AcquisitionMetadata,
    IQAcquisitionRecord,
    TriggerConfig,
    TriggerEvent,
    TriggerKind,
    TriggerRunMode,
)
from pluto_sa.sdr.trigger_detector import PowerLevelTriggerDetector
from pluto_sa.sdr.trigger_recorder import TriggeredIQRecorder


class TriggerAcquisitionController:
    """Arm, detect, and assemble immutable IQ records on one sample timeline."""

    def __init__(
        self,
        config: TriggerConfig,
        metadata: AcquisitionMetadata,
    ) -> None:
        if config.kind == TriggerKind.FREQUENCY_MASK:
            raise NotImplementedError("frequency-mask trigger is not implemented")
        self.config = config
        self.metadata = metadata
        self.detector = (
            PowerLevelTriggerDetector(
                replace(
                    config,
                    holdoff_samples=(
                        int(config.holdoff_samples)
                        + int(config.posttrigger_samples)
                    ),
                ),
                full_scale=metadata.iq_full_scale,
            )
            if config.kind == TriggerKind.POWER_LEVEL
            else None
        )
        self.recorder = TriggeredIQRecorder(config, metadata)
        self._stream_id: int | None = None
        self._expected_sample_index: int | None = None
        self._next_free_event_index: int | None = None
        self._arm_ready_index: int | None = None
        self._auto_deadline_index: int | None = None

    @property
    def stopped(self) -> bool:
        return self.recorder.stopped

    def reset(self) -> None:
        if self.detector is not None:
            self.detector.reset()
        self.recorder.reset()
        self._stream_id = None
        self._expected_sample_index = None
        self._next_free_event_index = None
        self._arm_ready_index = None
        self._auto_deadline_index = None

    def feed(
        self,
        block: IQBlock,
        *,
        forced_sample_index: int | None = None,
        trigger_iq: np.ndarray | None = None,
    ) -> tuple[IQAcquisitionRecord, ...]:
        """Consume one ordered block and return any records completed by it."""
        if self.stopped:
            return ()
        contiguous = (
            self._stream_id == block.stream_id
            and self._expected_sample_index == block.start_sample_index
            and not block.discontinuity_before
        )
        if not contiguous:
            self._reset_timeline(block)

        if forced_sample_index is not None:
            if self.config.kind != TriggerKind.POWER_LEVEL:
                raise ValueError("forced_sample_index requires a power trigger")
            events = (
                self._event_at(block, int(forced_sample_index), forced=True),
            )
        elif self.config.kind == TriggerKind.FREE_RUN:
            events = self._free_run_events(block)
        else:
            assert self.detector is not None
            detector_block = block
            if trigger_iq is not None:
                detector_iq = np.asarray(trigger_iq)
                if detector_iq.shape != block.iq.shape:
                    raise ValueError("trigger_iq must match the raw IQ block shape")
                if not np.issubdtype(detector_iq.dtype, np.complexfloating):
                    raise TypeError("trigger_iq must use a complex dtype")
                detector_block = replace(block, iq=detector_iq)
            detected = self._detect_power_events(detector_block)
            events = self._select_power_events(block, detected)

        records = self.recorder.feed(block, events)
        self._stream_id = block.stream_id
        self._expected_sample_index = block.end_sample_index
        return records

    def _reset_timeline(self, block: IQBlock) -> None:
        if self.detector is not None:
            self.detector.reset()
        self.recorder.reset()
        self._stream_id = block.stream_id
        self._expected_sample_index = block.start_sample_index
        first_eligible = block.start_sample_index + int(self.config.pretrigger_samples)
        self._next_free_event_index = first_eligible
        self._arm_ready_index = first_eligible
        timeout = self.config.auto_timeout_samples
        self._auto_deadline_index = (
            None
            if timeout is None
            else max(first_eligible, block.start_sample_index + int(timeout))
        )

    def _free_run_events(self, block: IQBlock) -> tuple[TriggerEvent, ...]:
        next_index = self._next_free_event_index
        if next_index is None:
            return ()
        events: list[TriggerEvent] = []
        step = max(1, int(self.config.record_samples))
        while next_index < block.end_sample_index:
            if next_index >= block.start_sample_index:
                events.append(self._event_at(block, next_index, forced=False))
            next_index += step
            if self.config.run_mode == TriggerRunMode.SINGLE:
                break
        self._next_free_event_index = next_index
        return tuple(events)

    def _detect_power_events(self, block: IQBlock) -> tuple[TriggerEvent, ...]:
        """Do not arm the detector until a full pretrigger history exists."""
        assert self.detector is not None
        eligible = int(self._arm_ready_index or block.start_sample_index)
        if block.end_sample_index <= eligible:
            return ()
        if block.start_sample_index >= eligible:
            return self.detector.process(block)
        offset = eligible - block.start_sample_index
        sliced = replace(
            block,
            start_sample_index=eligible,
            iq=block.iq[offset:],
            discontinuity_before=True,
        )
        return tuple(
            replace(
                event,
                offset_in_block=event.sample_index - block.start_sample_index,
            )
            for event in self.detector.process(sliced)
        )

    def _select_power_events(
        self,
        block: IQBlock,
        detected: tuple[TriggerEvent, ...],
    ) -> tuple[TriggerEvent, ...]:
        eligible = int(self._arm_ready_index or block.start_sample_index)
        deadline = self._auto_deadline_index
        timeout = self.config.auto_timeout_samples
        selected: list[TriggerEvent] = []

        for event in detected:
            if event.sample_index < eligible:
                continue
            while (
                self.config.run_mode == TriggerRunMode.AUTO
                and timeout is not None
                and deadline is not None
                and deadline <= event.sample_index
            ):
                forced = self._event_at(block, deadline, forced=True)
                selected.append(forced)
                eligible = self._next_arm_index(forced.sample_index)
                deadline = eligible + int(timeout)
                if event.sample_index < eligible:
                    continue
            selected.append(event)
            eligible = self._next_arm_index(event.sample_index)
            deadline = None if timeout is None else eligible + int(timeout)

        while (
            self.config.run_mode == TriggerRunMode.AUTO
            and timeout is not None
            and deadline is not None
            and block.start_sample_index <= deadline < block.end_sample_index
            and deadline >= eligible
        ):
            forced = self._event_at(block, deadline, forced=True)
            selected.append(forced)
            eligible = self._next_arm_index(forced.sample_index)
            deadline = eligible + int(timeout)

        self._arm_ready_index = eligible
        self._auto_deadline_index = deadline
        return tuple(selected)

    def _next_arm_index(self, event_index: int) -> int:
        return (
            int(event_index)
            + int(self.config.posttrigger_samples)
            + 1
            + int(self.config.holdoff_samples)
        )

    def _event_at(self, block: IQBlock, sample_index: int, *, forced: bool) -> TriggerEvent:
        if sample_index < block.start_sample_index or sample_index >= block.end_sample_index:
            raise ValueError("trigger event must be inside the current block")
        return TriggerEvent(
            stream_id=block.stream_id,
            sample_index=int(sample_index),
            sequence=block.sequence,
            offset_in_block=int(sample_index - block.start_sample_index),
            kind=self.config.kind,
            measured_value=None,
            forced=forced,
        )
