from __future__ import annotations

from collections import deque
from queue import Queue
from types import MethodType, SimpleNamespace

from pluto_sa.ui.main_window import RealtimeSpectrumWindow


def build_queue_owner(*, capacity: int = 1):
    owner = SimpleNamespace(
        _high_speed_ta_analysis_jobs=Queue(maxsize=capacity),
        _high_speed_ta_analysis_results=Queue(maxsize=capacity),
        _high_speed_ta_pending_analysis_jobs=deque(),
        _high_speed_ta_generation=2,
        config=SimpleNamespace(sweep_profile_logging=False),
    )
    owner._submit_high_speed_ta_analysis_job = MethodType(
        RealtimeSpectrumWindow._submit_high_speed_ta_analysis_job,
        owner,
    )
    return owner


def test_hsta_pending_jobs_apply_fifo_backpressure_without_drop() -> None:
    owner = build_queue_owner(capacity=1)
    first = SimpleNamespace(name="first")
    second = SimpleNamespace(name="second")
    owner._high_speed_ta_pending_analysis_jobs.extend((first, second))

    assert RealtimeSpectrumWindow._flush_high_speed_ta_pending_jobs(owner) is False
    assert owner._high_speed_ta_analysis_jobs.get_nowait() is first
    assert list(owner._high_speed_ta_pending_analysis_jobs) == [second]

    assert RealtimeSpectrumWindow._flush_high_speed_ta_pending_jobs(owner) is True
    assert owner._high_speed_ta_analysis_jobs.get_nowait() is second
    assert not owner._high_speed_ta_pending_analysis_jobs


def test_hsta_result_queue_ignores_previous_generation() -> None:
    owner = build_queue_owner(capacity=2)
    stale = SimpleNamespace(generation=1)
    current = SimpleNamespace(generation=2)
    owner._high_speed_ta_analysis_results.put_nowait(stale)
    owner._high_speed_ta_analysis_results.put_nowait(current)

    result = RealtimeSpectrumWindow._take_high_speed_ta_analysis_result(owner)

    assert result is current
