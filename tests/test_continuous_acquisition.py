from __future__ import annotations

import numpy as np
import pytest

from pluto_sa.sdr.continuous_acquisition import (
    ContinuousIQAcquisition,
    resolve_record_stream_block_samples,
)
from pluto_sa.sdr.iq_stream import IQStreamBuffer


class _Receiver:
    def __init__(self) -> None:
        self.iq_stream = IQStreamBuffer(capacity_blocks=8)
        self.running = False
        self.starts: list[tuple[int, str, int | None, bool]] = []
        self.stop_calls = 0
        self.reconfigured: list[object] = []

    def start(self, *, block_size, source, max_blocks, fresh):
        self.starts.append((block_size, source, max_blocks, fresh))
        self.running = True
        self.iq_stream.begin_stream(clear=True)
        return self.iq_stream.create_cursor(start="latest")

    def stop(self) -> bool:
        self.stop_calls += 1
        self.running = False
        return True

    def is_streaming(self) -> bool:
        return self.running

    def create_iq_stream_cursor(self, *, start="latest"):
        return self.iq_stream.create_cursor(start=start)

    def read_iq_stream(self, cursor, *, max_blocks=None):
        return self.iq_stream.read(cursor, max_blocks=max_blocks)

    def reconfigure(self, config) -> None:
        self.reconfigured.append(config)


def test_compatible_rearm_reuses_one_continuous_producer() -> None:
    receiver = _Receiver()
    acquisition = ContinuousIQAcquisition(receiver)

    first = acquisition.start(block_size=65_536, source="vsa", fresh=True)
    receiver.iq_stream.publish(np.ones(8, dtype=np.complex64), source="vsa")
    second = acquisition.start(block_size=65_536, source="vsa", fresh=True)

    assert second.next_sequence >= first.next_sequence
    assert receiver.starts == [(65_536, "vsa", None, True)]
    assert receiver.stop_calls == 0


def test_incompatible_plan_restarts_the_producer() -> None:
    receiver = _Receiver()
    acquisition = ContinuousIQAcquisition(receiver)
    acquisition.start(block_size=1024, source="rtsa")

    acquisition.start(block_size=2048, source="hsta", max_blocks=4)

    assert receiver.stop_calls == 1
    assert receiver.starts == [
        (1024, "rtsa", None, False),
        (2048, "hsta", 4, False),
    ]


def test_reconfigure_restores_the_exact_stream_plan() -> None:
    receiver = _Receiver()
    acquisition = ContinuousIQAcquisition(receiver)
    acquisition.start(block_size=4096, source="continuous", fresh=True)

    config = object()
    acquisition.reconfigure(config)

    assert receiver.reconfigured == [config]
    assert receiver.stop_calls == 1
    assert receiver.starts[-1] == (4096, "continuous", None, True)
    assert acquisition.is_running


def test_finished_finite_producer_is_restarted_on_next_arm() -> None:
    receiver = _Receiver()
    acquisition = ContinuousIQAcquisition(receiver)
    acquisition.start(block_size=2048, source="snapshot", max_blocks=2)
    receiver.running = False

    acquisition.start(block_size=2048, source="snapshot", max_blocks=2)

    assert len(receiver.starts) == 2


def test_incompatible_plan_does_not_restart_when_stop_fails() -> None:
    receiver = _Receiver()
    acquisition = ContinuousIQAcquisition(receiver)
    acquisition.start(block_size=1024, source="rtsa")
    receiver.stop = lambda: False

    with pytest.raises(RuntimeError, match="did not stop before restart"):
        acquisition.start(block_size=2048, source="hsta")

    assert receiver.starts == [(1024, "rtsa", None, False)]


def test_reconfigure_is_not_applied_when_stop_fails() -> None:
    receiver = _Receiver()
    acquisition = ContinuousIQAcquisition(receiver)
    acquisition.start(block_size=1024, source="rtsa")
    receiver.stop = lambda: False

    with pytest.raises(RuntimeError, match="did not stop before reconfigure"):
        acquisition.reconfigure(object())

    assert receiver.reconfigured == []


@pytest.mark.parametrize(
    ("record_samples", "max_records", "expected"),
    [
        (80_000, None, 240_000),
        (24_000, None, 240_000),
        (24_000, 4, 72_000),
        (300_000, None, 300_000),
    ],
)
def test_record_stream_block_policy_uses_integer_record_islands(
    record_samples: int,
    max_records: int | None,
    expected: int,
) -> None:
    assert (
        resolve_record_stream_block_samples(
            record_samples,
            max_records_per_block=max_records,
        )
        == expected
    )
