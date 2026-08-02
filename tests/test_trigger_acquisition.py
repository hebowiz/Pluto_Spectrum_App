from __future__ import annotations

import numpy as np
import pytest

from pluto_sa.sdr.iq_stream import IQStreamBuffer
from pluto_sa.sdr.trigger import (
    AcquisitionMetadata,
    TriggerConfig,
    TriggerKind,
    TriggerRunMode,
    TriggerSlope,
)
from pluto_sa.sdr.trigger_acquisition import TriggerAcquisitionController


def metadata() -> AcquisitionMetadata:
    return AcquisitionMetadata(
        sample_rate_hz=1_000_000,
        center_freq_hz=100_000_000,
        rf_bandwidth_hz=1_000_000,
        gain_db=20,
        source="trigger_test",
        iq_full_scale=1.0,
    )


def publish(stream: IQStreamBuffer, values: list[float]):
    iq = np.asarray(values, dtype=np.complex64)
    return stream.publish(iq, source="trigger_test")


def test_free_run_builds_nonoverlapping_exact_records_across_blocks() -> None:
    config = TriggerConfig(
        kind=TriggerKind.FREE_RUN,
        pretrigger_samples=2,
        posttrigger_samples=3,
    )
    controller = TriggerAcquisitionController(config, metadata())
    stream = IQStreamBuffer(capacity_blocks=8)
    stream.begin_stream()

    assert controller.feed(publish(stream, [0, 1, 2, 3])) == ()
    records = controller.feed(publish(stream, [4, 5, 6, 7, 8, 9, 10, 11]))

    assert len(records) == 2
    np.testing.assert_array_equal(records[0].iq, np.arange(6, dtype=np.complex64))
    np.testing.assert_array_equal(records[1].iq, np.arange(6, 12, dtype=np.complex64))
    assert [record.trigger_sample_offset for record in records] == [2, 2]


def test_power_normal_waits_for_real_event_and_keeps_prestore() -> None:
    config = TriggerConfig(
        kind=TriggerKind.POWER_LEVEL,
        run_mode=TriggerRunMode.NORMAL,
        slope=TriggerSlope.RISING,
        level_dbfs=-10.0,
        pretrigger_samples=2,
        posttrigger_samples=2,
    )
    controller = TriggerAcquisitionController(config, metadata())
    stream = IQStreamBuffer(capacity_blocks=8)
    stream.begin_stream()

    quiet = [10 ** (-30 / 20)] * 4
    crossing = [10 ** (-30 / 20), 10 ** (-5 / 20), 10 ** (-5 / 20), 10 ** (-30 / 20)]
    assert controller.feed(publish(stream, quiet)) == ()
    records = controller.feed(publish(stream, crossing))

    assert len(records) == 1
    assert records[0].trigger.forced is False
    assert records[0].trigger_sample_index == 5
    assert records[0].sample_count == 5


def test_power_detector_does_not_arm_before_prestore_is_full() -> None:
    config = TriggerConfig(
        kind=TriggerKind.POWER_LEVEL,
        run_mode=TriggerRunMode.NORMAL,
        slope=TriggerSlope.RISING,
        level_dbfs=-10.0,
        pretrigger_samples=4,
        posttrigger_samples=1,
    )
    controller = TriggerAcquisitionController(config, metadata())
    stream = IQStreamBuffer(capacity_blocks=8)
    stream.begin_stream()

    high = 10 ** (-5 / 20)
    low = 10 ** (-30 / 20)
    assert controller.feed(publish(stream, [high, high, low, low])) == ()
    records = controller.feed(publish(stream, [low, high, high]))

    assert len(records) == 1
    assert records[0].trigger_sample_index == 5
    assert records[0].trigger.offset_in_block == 1


def test_power_auto_forces_record_at_sample_timeout() -> None:
    config = TriggerConfig(
        kind=TriggerKind.POWER_LEVEL,
        run_mode=TriggerRunMode.AUTO,
        level_dbfs=-3.0,
        pretrigger_samples=2,
        posttrigger_samples=1,
        auto_timeout_samples=5,
    )
    controller = TriggerAcquisitionController(config, metadata())
    stream = IQStreamBuffer(capacity_blocks=8)
    stream.begin_stream()

    records = controller.feed(publish(stream, [0.01] * 8))

    assert len(records) == 1
    assert records[0].trigger.forced is True
    assert records[0].trigger_sample_index == 5
    assert records[0].trigger_sample_offset == 2


def test_power_trigger_accepts_host_timed_forced_event_inside_island() -> None:
    config = TriggerConfig(
        kind=TriggerKind.POWER_LEVEL,
        run_mode=TriggerRunMode.AUTO,
        level_dbfs=-3.0,
        pretrigger_samples=2,
        posttrigger_samples=3,
        auto_timeout_samples=None,
    )
    controller = TriggerAcquisitionController(config, metadata())
    stream = IQStreamBuffer(capacity_blocks=8)
    stream.begin_stream()
    block = publish(stream, [0.01] * 12)

    records = controller.feed(
        block,
        forced_sample_index=block.start_sample_index + 2,
    )

    assert len(records) == 1
    assert records[0].trigger.forced is True
    assert records[0].trigger_sample_offset == 2
    assert records[0].sample_count == 6


def test_power_trigger_detects_filtered_iq_but_records_raw_iq() -> None:
    config = TriggerConfig(
        kind=TriggerKind.POWER_LEVEL,
        run_mode=TriggerRunMode.NORMAL,
        level_dbfs=-10.0,
        pretrigger_samples=0,
        posttrigger_samples=0,
    )
    controller = TriggerAcquisitionController(config, metadata())
    stream = IQStreamBuffer(capacity_blocks=8)
    stream.begin_stream()
    raw_iq = np.asarray([0.01, 0.01], dtype=np.complex64)
    block = stream.publish(raw_iq, source="trigger_test")
    filtered_iq = np.asarray([0.01, 0.5], dtype=np.complex64)

    records = controller.feed(block, trigger_iq=filtered_iq)

    assert len(records) == 1
    assert records[0].trigger_sample_index == 1
    assert records[0].trigger.measured_value == pytest.approx(
        20.0 * np.log10(0.5)
    )
    np.testing.assert_array_equal(records[0].iq, raw_iq[1:])


def test_power_trigger_rejects_mismatched_filtered_iq_shape() -> None:
    config = TriggerConfig(kind=TriggerKind.POWER_LEVEL)
    controller = TriggerAcquisitionController(config, metadata())
    stream = IQStreamBuffer(capacity_blocks=8)
    stream.begin_stream()
    block = stream.publish(np.ones(4, dtype=np.complex64), source="trigger_test")

    with pytest.raises(ValueError, match="shape"):
        controller.feed(block, trigger_iq=np.ones(3, dtype=np.complex64))


def test_power_trigger_edge_record_is_pending_until_island_reset() -> None:
    config = TriggerConfig(
        kind=TriggerKind.POWER_LEVEL,
        run_mode=TriggerRunMode.NORMAL,
        slope=TriggerSlope.RISING,
        level_dbfs=-10.0,
        pretrigger_samples=2,
        posttrigger_samples=3,
    )
    controller = TriggerAcquisitionController(config, metadata())
    stream = IQStreamBuffer(capacity_blocks=8)
    stream.begin_stream()
    low = 10 ** (-30 / 20)
    high = 10 ** (-5 / 20)

    assert controller.feed(publish(stream, [low] * 6 + [high, high])) == ()
    assert controller.recorder.pending_records == 1

    controller.reset()
    assert controller.recorder.pending_records == 0


def test_power_auto_can_rearm_multiple_times_inside_one_large_block() -> None:
    config = TriggerConfig(
        kind=TriggerKind.POWER_LEVEL,
        run_mode=TriggerRunMode.AUTO,
        level_dbfs=-3.0,
        pretrigger_samples=1,
        posttrigger_samples=1,
        auto_timeout_samples=3,
    )
    controller = TriggerAcquisitionController(config, metadata())
    stream = IQStreamBuffer(capacity_blocks=8)
    stream.begin_stream()

    records = controller.feed(publish(stream, [0.01] * 12))

    assert [record.trigger_sample_index for record in records] == [3, 8]
    assert all(record.trigger.forced for record in records)


def test_power_detector_does_not_rearm_inside_posttrigger_record() -> None:
    config = TriggerConfig(
        kind=TriggerKind.POWER_LEVEL,
        run_mode=TriggerRunMode.NORMAL,
        level_dbfs=-10.0,
        hysteresis_db=1.0,
        pretrigger_samples=0,
        posttrigger_samples=3,
    )
    controller = TriggerAcquisitionController(config, metadata())
    stream = IQStreamBuffer(capacity_blocks=8)
    stream.begin_stream()
    low = 10 ** (-30 / 20)
    high = 10 ** (-5 / 20)

    records = controller.feed(
        publish(stream, [low, high, low, high, low, low, high, low, low, low])
    )

    assert [record.trigger_sample_index for record in records] == [1, 6]


def test_single_stops_after_first_complete_record() -> None:
    config = TriggerConfig(
        kind=TriggerKind.FREE_RUN,
        run_mode=TriggerRunMode.SINGLE,
        pretrigger_samples=0,
        posttrigger_samples=3,
    )
    controller = TriggerAcquisitionController(config, metadata())
    stream = IQStreamBuffer(capacity_blocks=8)
    stream.begin_stream()

    records = controller.feed(publish(stream, list(range(8))))

    assert len(records) == 1
    assert controller.stopped is True
    assert controller.feed(publish(stream, list(range(8, 12)))) == ()


def test_discontinuity_drops_partial_record_and_rearms_new_epoch() -> None:
    config = TriggerConfig(
        kind=TriggerKind.FREE_RUN,
        pretrigger_samples=1,
        posttrigger_samples=4,
    )
    controller = TriggerAcquisitionController(config, metadata())
    stream = IQStreamBuffer(capacity_blocks=8)
    stream.begin_stream()
    assert controller.feed(publish(stream, [0, 1, 2])) == ()

    stream.begin_stream()
    assert controller.feed(publish(stream, [10, 11, 12])) == ()
    records = controller.feed(publish(stream, [13, 14, 15]))

    assert len(records) == 1
    np.testing.assert_array_equal(records[0].iq, [10, 11, 12, 13, 14, 15])
