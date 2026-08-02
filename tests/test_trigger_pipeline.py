from __future__ import annotations

import numpy as np
import pytest

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


@pytest.mark.parametrize(
    "slope",
    (TriggerSlope.RISING, TriggerSlope.FALLING, TriggerSlope.EITHER),
)
def test_vectorized_single_sample_trigger_matches_sample_state_machine(
    slope: TriggerSlope,
) -> None:
    rng = np.random.default_rng(20260802)
    values_dbfs = rng.uniform(-30.0, -2.0, size=2000)
    config = TriggerConfig(
        kind=TriggerKind.POWER_LEVEL,
        slope=slope,
        level_dbfs=-12.0,
        hysteresis_db=2.5,
        holdoff_samples=7,
    )
    stream = IQStreamBuffer(capacity_blocks=8)
    stream.begin_stream()
    blocks = (
        publish(stream, iq_at_dbfs(values_dbfs[:731].tolist())),
        publish(stream, iq_at_dbfs(values_dbfs[731:].tolist())),
    )
    vectorized = PowerLevelTriggerDetector(config, full_scale=1.0)
    reference = PowerLevelTriggerDetector(config, full_scale=1.0)

    actual = [event for block in blocks for event in vectorized.process(block)]
    expected: list[tuple[int, float]] = []
    for block in blocks:
        metric = 20.0 * np.log10(np.maximum(np.abs(block.iq), np.finfo(float).tiny))
        for offset, value in enumerate(metric):
            event = reference._process_sample(
                float(value),
                block.start_sample_index + offset,
                block.sequence,
                offset,
            )
            if event is not None:
                expected.append((event[0], event[1]))

    assert [event.sample_index for event in actual] == [item[0] for item in expected]
    np.testing.assert_allclose(
        [event.measured_value for event in actual],
        [item[1] for item in expected],
        rtol=0.0,
        atol=1e-6,
    )


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
