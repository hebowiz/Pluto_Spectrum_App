from __future__ import annotations

import numpy as np

from pluto_sa.sdr.iq_stream import IQStreamBuffer
from pluto_sa.sdr.trigger import (
    AcquisitionMetadata,
    TriggerConfig,
    TriggerEvent,
    TriggerKind,
    TriggerRearmMode,
    TriggerSlope,
)
from pluto_sa.sdr.trigger_detector import PowerLevelTriggerDetector
from pluto_sa.sdr.trigger_recorder import TriggeredIQRecorder


def iq_at_dbfs(values_dbfs: list[float]) -> np.ndarray:
    return np.asarray([10.0 ** (value / 20.0) for value in values_dbfs], dtype=np.complex64)


def metadata() -> AcquisitionMetadata:
    return AcquisitionMetadata(
        sample_rate_hz=1_000_000,
        center_freq_hz=100_000_000,
        rf_bandwidth_hz=1_000_000,
        gain_db=20,
        source="trigger_test",
    )


def publish(stream: IQStreamBuffer, values: np.ndarray):
    return stream.publish(values, source="trigger_test")


def event_for(block, offset: int) -> TriggerEvent:
    return TriggerEvent(
        stream_id=block.stream_id,
        sample_index=block.start_sample_index + offset,
        sequence=block.sequence,
        offset_in_block=offset,
        kind=TriggerKind.POWER_LEVEL,
        measured_value=-5.0,
    )


def test_power_trigger_qualification_crosses_block_boundary() -> None:
    config = TriggerConfig(
        kind=TriggerKind.POWER_LEVEL,
        slope=TriggerSlope.RISING,
        level_dbfs=-10.0,
        minimum_duration_samples=2,
    )
    detector = PowerLevelTriggerDetector(config, full_scale=1.0)
    stream = IQStreamBuffer(capacity_blocks=8)
    stream.begin_stream()
    first = publish(stream, iq_at_dbfs([-20.0, -5.0]))
    second = publish(stream, iq_at_dbfs([-5.0, -20.0]))

    assert detector.process(first) == ()
    events = detector.process(second)

    assert len(events) == 1
    assert events[0].sample_index == 1
    assert events[0].sequence == first.sequence
    assert events[0].offset_in_block == 1


def test_power_trigger_requires_hysteresis_rearm_and_holdoff() -> None:
    config = TriggerConfig(
        kind=TriggerKind.POWER_LEVEL,
        slope=TriggerSlope.RISING,
        level_dbfs=-10.0,
        hysteresis_db=3.0,
        holdoff_samples=2,
    )
    detector = PowerLevelTriggerDetector(config, full_scale=1.0)
    stream = IQStreamBuffer(capacity_blocks=8)
    stream.begin_stream()
    block = publish(stream, iq_at_dbfs([-20, -5, -11, -14, -14, -5]))

    events = detector.process(block)

    assert [event.sample_index for event in events] == [1, 5]


def test_power_trigger_discards_partial_qualification_at_epoch_boundary() -> None:
    config = TriggerConfig(
        kind=TriggerKind.POWER_LEVEL,
        level_dbfs=-10.0,
        minimum_duration_samples=2,
    )
    detector = PowerLevelTriggerDetector(config, full_scale=1.0)
    stream = IQStreamBuffer(capacity_blocks=8)
    stream.begin_stream()
    assert detector.process(publish(stream, iq_at_dbfs([-5.0]))) == ()

    stream.begin_stream()
    first = publish(stream, iq_at_dbfs([-5.0]))
    second = publish(stream, iq_at_dbfs([-5.0]))
    assert detector.process(first) == ()
    events = detector.process(second)

    assert len(events) == 1
    assert events[0].stream_id == first.stream_id
    assert events[0].sample_index == 0


def test_trigger_recorder_builds_exact_record_across_blocks() -> None:
    config = TriggerConfig(
        kind=TriggerKind.POWER_LEVEL,
        pretrigger_samples=2,
        posttrigger_samples=3,
    )
    recorder = TriggeredIQRecorder(config, metadata())
    stream = IQStreamBuffer(capacity_blocks=8)
    stream.begin_stream()
    first = publish(stream, np.arange(4, dtype=np.complex64))
    second = publish(stream, np.arange(4, 8, dtype=np.complex64))

    assert recorder.feed(first, [event_for(first, 2)]) == ()
    records = recorder.feed(second)

    assert len(records) == 1
    record = records[0]
    np.testing.assert_array_equal(record.iq, np.arange(6, dtype=np.complex64))
    assert record.start_sample_index == 0
    assert record.trigger_sample_offset == 2
    assert record.end_sample_index == 6


def test_qualified_trigger_keeps_prestore_until_delayed_event_is_reported() -> None:
    config = TriggerConfig(
        kind=TriggerKind.POWER_LEVEL,
        level_dbfs=-10.0,
        minimum_duration_samples=2,
        pretrigger_samples=2,
        posttrigger_samples=1,
    )
    detector = PowerLevelTriggerDetector(config, full_scale=1.0)
    recorder = TriggeredIQRecorder(config, metadata())
    stream = IQStreamBuffer(capacity_blocks=8)
    stream.begin_stream()
    first = publish(stream, iq_at_dbfs([-30.0, -25.0, -20.0, -5.0]))
    second = publish(stream, iq_at_dbfs([-5.0, -20.0]))

    assert recorder.feed(first, detector.process(first)) == ()
    records = recorder.feed(second, detector.process(second))

    assert len(records) == 1
    assert records[0].start_sample_index == 1
    assert records[0].trigger_sample_index == 3
    assert records[0].sample_count == 4


def test_trigger_recorder_rejects_event_without_full_prestore() -> None:
    config = TriggerConfig(
        kind=TriggerKind.POWER_LEVEL,
        pretrigger_samples=3,
        posttrigger_samples=0,
    )
    recorder = TriggeredIQRecorder(config, metadata())
    stream = IQStreamBuffer(capacity_blocks=8)
    stream.begin_stream()
    block = publish(stream, np.arange(4, dtype=np.complex64))

    assert recorder.feed(block, [event_for(block, 1)]) == ()
    assert recorder.rejected_prestore_events == 1


def test_trigger_recorder_drops_pending_record_on_discontinuity() -> None:
    config = TriggerConfig(
        kind=TriggerKind.POWER_LEVEL,
        pretrigger_samples=1,
        posttrigger_samples=5,
    )
    recorder = TriggeredIQRecorder(config, metadata())
    stream = IQStreamBuffer(capacity_blocks=8)
    stream.begin_stream()
    first = publish(stream, np.arange(4, dtype=np.complex64))
    assert recorder.feed(first, [event_for(first, 2)]) == ()
    assert recorder.pending_records == 1

    stream.begin_stream()
    discontinuous = publish(stream, np.arange(10, 14, dtype=np.complex64))
    assert recorder.feed(discontinuous) == ()

    assert recorder.pending_records == 0
    assert recorder.dropped_pending_records == 1


def test_stop_on_trigger_returns_one_record_and_stops() -> None:
    config = TriggerConfig(
        kind=TriggerKind.POWER_LEVEL,
        rearm_mode=TriggerRearmMode.STOP_ON_TRIGGER,
        pretrigger_samples=0,
        posttrigger_samples=0,
    )
    recorder = TriggeredIQRecorder(config, metadata())
    stream = IQStreamBuffer(capacity_blocks=8)
    stream.begin_stream()
    block = publish(stream, np.arange(4, dtype=np.complex64))

    records = recorder.feed(block, [event_for(block, 1), event_for(block, 2)])

    assert len(records) == 1
    assert recorder.stopped is True
    assert recorder.feed(publish(stream, np.arange(4, 8, dtype=np.complex64))) == ()
