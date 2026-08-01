"""Streaming software-trigger detectors operating on IQ sample indices."""

from __future__ import annotations

import numpy as np

from pluto_sa.sdr.iq_stream import IQBlock
from pluto_sa.sdr.trigger import TriggerConfig, TriggerEvent, TriggerKind, TriggerSlope


class PowerLevelTriggerDetector:
    """Detect qualified IQ magnitude crossings in dBFS without block gaps."""

    def __init__(self, config: TriggerConfig, *, full_scale: float) -> None:
        if config.kind != TriggerKind.POWER_LEVEL:
            raise ValueError("PowerLevelTriggerDetector requires POWER_LEVEL config")
        if float(full_scale) <= 0.0:
            raise ValueError("full_scale must be positive")
        self.config = config
        self.full_scale = float(full_scale)
        self._stream_id: int | None = None
        self._expected_sample_index: int | None = None
        self._armed = True
        self._holdoff_until = -1
        self._candidate_start: int | None = None
        self._candidate_value: float | None = None
        self._candidate_slope: TriggerSlope | None = None
        self._candidate_sequence = 0
        self._candidate_offset = 0
        self._stable_side = 0

    def reset(self) -> None:
        self._stream_id = None
        self._expected_sample_index = None
        self._armed = True
        self._holdoff_until = -1
        self._candidate_start = None
        self._candidate_value = None
        self._candidate_slope = None
        self._candidate_sequence = 0
        self._candidate_offset = 0
        self._stable_side = 0

    def process(self, block: IQBlock) -> tuple[TriggerEvent, ...]:
        """Evaluate one ordered block and return sample-accurate trigger events."""
        contiguous = (
            self._stream_id == block.stream_id
            and self._expected_sample_index == block.start_sample_index
            and not block.discontinuity_before
        )
        if not contiguous:
            self.reset()
            self._stream_id = block.stream_id

        magnitude = np.abs(block.iq).astype(np.float64, copy=False)
        metric_dbfs = 20.0 * np.log10(
            np.maximum(magnitude / self.full_scale, np.finfo(np.float64).tiny)
        )
        events: list[TriggerEvent] = []
        for offset, metric in enumerate(metric_dbfs):
            sample_index = block.start_sample_index + offset
            event = self._process_sample(
                float(metric),
                sample_index,
                block.sequence,
                offset,
            )
            if event is None:
                continue
            events.append(
                TriggerEvent(
                    stream_id=block.stream_id,
                    sample_index=event[0],
                    sequence=event[2],
                    offset_in_block=event[3],
                    kind=TriggerKind.POWER_LEVEL,
                    measured_value=event[1],
                )
            )

        self._stream_id = block.stream_id
        self._expected_sample_index = block.end_sample_index
        return tuple(events)

    def _process_sample(
        self,
        metric: float,
        sample_index: int,
        sequence: int,
        offset_in_block: int,
    ) -> tuple[int, float, int, int] | None:
        if not self._armed:
            if sample_index <= self._holdoff_until:
                return None
            if not self._can_rearm(metric):
                return None
            self._armed = True
            self._candidate_start = None
            self._candidate_slope = None

        slope = self._qualifying_slope(metric)
        if slope is None:
            self._candidate_start = None
            self._candidate_value = None
            self._candidate_slope = None
            self._update_stable_side(metric)
            return None

        if self._candidate_start is None or self._candidate_slope != slope:
            self._candidate_start = sample_index
            self._candidate_value = metric
            self._candidate_slope = slope
            self._candidate_sequence = sequence
            self._candidate_offset = offset_in_block

        qualified_samples = sample_index - self._candidate_start + 1
        if qualified_samples < int(self.config.minimum_duration_samples):
            return None

        event_index = int(self._candidate_start)
        event_value = float(self._candidate_value)
        event_sequence = self._candidate_sequence
        event_offset = self._candidate_offset
        self._armed = False
        self._holdoff_until = sample_index + int(self.config.holdoff_samples)
        self._candidate_start = None
        self._candidate_value = None
        self._candidate_slope = None
        self._stable_side = 1 if slope == TriggerSlope.RISING else -1
        return event_index, event_value, event_sequence, event_offset

    def _qualifying_slope(self, metric: float) -> TriggerSlope | None:
        threshold = float(self.config.level_dbfs)
        if self.config.slope == TriggerSlope.RISING:
            return TriggerSlope.RISING if metric >= threshold else None
        if self.config.slope == TriggerSlope.FALLING:
            return TriggerSlope.FALLING if metric <= threshold else None

        if self._stable_side < 0 and metric >= threshold:
            return TriggerSlope.RISING
        if self._stable_side > 0 and metric <= threshold:
            return TriggerSlope.FALLING
        return None

    def _can_rearm(self, metric: float) -> bool:
        threshold = float(self.config.level_dbfs)
        hysteresis = float(self.config.hysteresis_db)
        if self.config.slope == TriggerSlope.RISING:
            return metric <= threshold - hysteresis
        if self.config.slope == TriggerSlope.FALLING:
            return metric >= threshold + hysteresis
        self._update_stable_side(metric)
        return self._stable_side != 0

    def _update_stable_side(self, metric: float) -> None:
        threshold = float(self.config.level_dbfs)
        hysteresis = float(self.config.hysteresis_db)
        if metric <= threshold - hysteresis:
            self._stable_side = -1
        elif metric >= threshold + hysteresis:
            self._stable_side = 1
