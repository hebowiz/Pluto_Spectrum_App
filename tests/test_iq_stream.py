from __future__ import annotations

import numpy as np
import pytest

from pluto_sa.sdr.iq_stream import IQStreamBuffer


def make_iq(start: int, count: int) -> np.ndarray:
    values = np.arange(start, start + count, dtype=np.float32)
    return (values + 1j * -values).astype(np.complex64)


def test_publish_assigns_contiguous_block_and_sample_indices() -> None:
    stream = IQStreamBuffer(capacity_blocks=4)
    stream.begin_stream()

    first = stream.publish(make_iq(0, 3), timestamp_s=1.0)
    second = stream.publish(make_iq(3, 2), timestamp_s=2.0)

    assert first.stream_id == second.stream_id == 1
    assert first.block_index == 0
    assert second.block_index == 1
    assert first.start_sample_index == 0
    assert second.start_sample_index == 3
    assert second.end_sample_index == 5
    assert first.discontinuity_before is True
    assert second.discontinuity_before is False


def test_independent_consumers_do_not_remove_each_others_blocks() -> None:
    stream = IQStreamBuffer(capacity_blocks=4)
    stream.begin_stream()
    cursor_a = stream.create_cursor(start="latest")
    cursor_b = stream.create_cursor(start="latest")
    stream.publish(make_iq(0, 2))

    result_a = stream.read(cursor_a)
    result_b = stream.read(cursor_b)

    assert len(result_a.blocks) == 1
    assert len(result_b.blocks) == 1
    assert result_a.blocks[0].sequence == result_b.blocks[0].sequence


def test_newest_cursor_reuses_latest_retained_block_without_replaying_backlog() -> None:
    stream = IQStreamBuffer(capacity_blocks=4)
    stream.begin_stream()
    stream.publish(make_iq(0, 2))
    newest = stream.publish(make_iq(2, 2))

    result = stream.read(stream.create_cursor(start="newest"))

    assert result.overrun is False
    assert [block.sequence for block in result.blocks] == [newest.sequence]
    np.testing.assert_array_equal(result.blocks[0].iq, make_iq(2, 2))


def test_consumer_overrun_is_reported_with_missed_block_count() -> None:
    stream = IQStreamBuffer(capacity_blocks=2)
    stream.begin_stream()
    cursor = stream.create_cursor(start="latest")
    for index in range(4):
        stream.publish(make_iq(index, 1))

    result = stream.read(cursor)

    assert result.overrun is True
    assert result.missed_blocks == 2
    assert [block.sequence for block in result.blocks] == [2, 3]
    assert stream.stats().overwritten_blocks == 2


def test_begin_stream_resets_local_indices_and_marks_discontinuity() -> None:
    stream = IQStreamBuffer(capacity_blocks=4)
    stream.begin_stream()
    old = stream.publish(make_iq(0, 2))
    stream.begin_stream()
    new = stream.publish(make_iq(10, 3))

    assert new.stream_id == old.stream_id + 1
    assert new.sequence == old.sequence + 1
    assert new.block_index == 0
    assert new.start_sample_index == 0
    assert new.discontinuity_before is True


def test_latest_samples_can_span_multiple_blocks() -> None:
    stream = IQStreamBuffer(capacity_blocks=4)
    stream.begin_stream()
    stream.publish(make_iq(0, 3))
    stream.publish(make_iq(3, 3))

    latest = stream.latest_samples(4)

    np.testing.assert_array_equal(latest, make_iq(2, 4))


def test_latest_samples_does_not_cross_stream_epoch() -> None:
    stream = IQStreamBuffer(capacity_blocks=4)
    stream.begin_stream()
    stream.publish(make_iq(0, 3))
    stream.begin_stream()
    stream.publish(make_iq(3, 2))

    assert stream.latest_samples(3) is None


@pytest.mark.parametrize("capacity", [0, -1])
def test_capacity_must_be_positive(capacity: int) -> None:
    with pytest.raises(ValueError):
        IQStreamBuffer(capacity_blocks=capacity)
