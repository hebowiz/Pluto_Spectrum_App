from __future__ import annotations

import numpy as np

from pluto_sa.sdr.iq_stream import IQStreamBuffer
from pluto_sa.sdr.iq_window import (
    IQWindowAssembler,
    resolve_fft_aligned_window_samples,
    resolve_time_window_samples,
)


def publish(stream: IQStreamBuffer, values: list[int]):
    return stream.publish(np.asarray(values, dtype=np.complex64), source="test")


def test_assembler_carries_tail_across_blocks() -> None:
    stream = IQStreamBuffer(capacity_blocks=8)
    stream.begin_stream()
    assembler = IQWindowAssembler(window_samples=5)

    assert assembler.feed(publish(stream, [0, 1, 2])) == ()
    windows = assembler.feed(publish(stream, [3, 4, 5, 6]))

    assert len(windows) == 1
    np.testing.assert_array_equal(windows[0].iq, [0, 1, 2, 3, 4])
    assert windows[0].start_sample_index == 0
    assert windows[0].end_sample_index == 5
    assert assembler.pending_samples == 2


def test_assembler_emits_multiple_exact_windows_from_one_block() -> None:
    stream = IQStreamBuffer(capacity_blocks=8)
    stream.begin_stream()
    assembler = IQWindowAssembler(window_samples=3)

    windows = assembler.feed(publish(stream, list(range(8))))

    assert [window.sample_count for window in windows] == [3, 3]
    assert [window.start_sample_index for window in windows] == [0, 3]
    np.testing.assert_array_equal(windows[1].iq, [3, 4, 5])
    assert assembler.pending_samples == 2


def test_discontinuity_discards_partial_window() -> None:
    stream = IQStreamBuffer(capacity_blocks=8)
    stream.begin_stream()
    assembler = IQWindowAssembler(window_samples=4)
    assembler.feed(publish(stream, [0, 1]))

    stream.begin_stream()
    windows = assembler.feed(publish(stream, [10, 11, 12, 13]))

    assert assembler.discarded_partial_samples == 2
    assert len(windows) == 1
    np.testing.assert_array_equal(windows[0].iq, [10, 11, 12, 13])
    assert windows[0].discontinuity_before is True
    assert windows[0].start_sample_index == 0


def test_sample_index_gap_discards_partial_window() -> None:
    stream = IQStreamBuffer(capacity_blocks=8)
    stream.begin_stream()
    assembler = IQWindowAssembler(window_samples=4)
    first = publish(stream, [0, 1])
    assembler.feed(first)
    second = publish(stream, [2, 3, 4, 5])
    broken = type(second)(
        **{**second.__dict__, "start_sample_index": second.start_sample_index + 10}
    )

    windows = assembler.feed(broken)

    assert assembler.discarded_partial_samples == 2
    assert len(windows) == 1
    assert windows[0].start_sample_index == 12
    assert windows[0].discontinuity_before is True


def test_fft_aligned_window_covers_requested_time_without_partial_frame() -> None:
    samples = resolve_fft_aligned_window_samples(
        time_span_s=0.010,
        sample_rate_hz=1_000_000,
        fft_size=4096,
    )

    assert samples == 12_288
    assert samples % 4096 == 0
    assert samples >= 10_000


def test_time_window_uses_requested_sample_count_without_fft_alignment() -> None:
    assert resolve_time_window_samples(0.010, 4_000_000) == 40_000
