from __future__ import annotations

from collections import deque
from queue import Queue
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import pluto_sa.ui.main_window as main_window_module

from pluto_sa.ui.main_window import (
    HIGH_SPEED_TA_CAPTURE_BLOCK_SAMPLES,
    HIGH_SPEED_TA_SNAPSHOT_MAX_SAMPLES,
    HighSpeedTAAnalysisJob,
    RealtimeSpectrumWindow,
    SWEEP_STATE_SINGLE,
    SWEEP_STATE_RUNNING,
    format_hsta_sampling_status,
)


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


def test_hsta_continuous_restart_invalidates_existing_stream_cursor() -> None:
    calls: list[object] = []
    owner = SimpleNamespace(
        sweep_state="single",
        sweep_controller=SimpleNamespace(stop=lambda: calls.append("sweep_stop")),
        _high_speed_ta_single_waiting_result=True,
        _hsta_debug_log=lambda *args, **kwargs: None,
        _prepare_sweep_like_continuous_entry_state=lambda: calls.append("prepare"),
        _stop_high_speed_ta_stream=lambda **kwargs: calls.append(("stop", kwargs)),
        _reset_high_speed_time_analyzer_capture_window=lambda **kwargs: calls.append(
            ("reset", kwargs)
        ),
        _start_high_speed_ta_stream=lambda: calls.append("start"),
        _restart_timer_for_current_mode=lambda: calls.append("timer"),
        _update_continuous_button=lambda: calls.append("button"),
    )

    RealtimeSpectrumWindow._start_high_speed_time_analyzer_continuous(owner)

    assert ("stop", {"stop_analysis_thread": False}) in calls
    assert calls.index(("stop", {"stop_analysis_thread": False})) < calls.index("start")
    assert owner._high_speed_ta_single_waiting_result is False


def test_hsta_single_restart_invalidates_existing_stream_cursor() -> None:
    calls: list[object] = []
    owner = SimpleNamespace(
        sweep_state="running",
        _high_speed_ta_single_waiting_result=True,
        _hsta_debug_log=lambda *args, **kwargs: None,
        _stop_high_speed_ta_stream=lambda **kwargs: calls.append(("stop", kwargs)),
        _reset_high_speed_time_analyzer_capture_window=lambda **kwargs: calls.append(
            ("reset", kwargs)
        ),
        _start_high_speed_ta_stream=lambda: calls.append("start"),
    )

    RealtimeSpectrumWindow._enter_single_high_speed_time_analyzer_mode(owner)

    assert calls[0] == ("stop", {"stop_analysis_thread": False})
    assert calls[-1] == "start"
    assert owner._high_speed_ta_single_waiting_result is False


def _snapshot_owner(
    *,
    trigger_kind: str = "free_run",
    time_span_s: float = 0.01,
    sample_rate_hz: int = 12_000_000,
    sweep_state: str = SWEEP_STATE_SINGLE,
):
    config = SimpleNamespace(
        sample_rate_hz=sample_rate_hz,
        hsta_trigger_kind=trigger_kind,
    )
    owner = SimpleNamespace(
        config=config,
        sweep_state=sweep_state,
        _time_analyzer_time_span_s=lambda: time_span_s,
        _hsta_debug_log=lambda *args, **kwargs: None,
    )
    owner._use_high_speed_ta_snapshot = MethodType(
        RealtimeSpectrumWindow._use_high_speed_ta_snapshot,
        owner,
    )
    owner._use_high_speed_ta_buffer_islands = MethodType(
        RealtimeSpectrumWindow._use_high_speed_ta_buffer_islands,
        owner,
    )
    return owner


def test_hsta_single_free_run_uses_one_exact_snapshot_buffer() -> None:
    owner = _snapshot_owner(time_span_s=0.01)

    samples = RealtimeSpectrumWindow._resolve_high_speed_ta_capture_block_samples(owner)

    assert samples == 120_000


def test_hsta_high_rate_power_trigger_uses_large_record_aligned_island() -> None:
    owner = _snapshot_owner(trigger_kind="power_level")

    samples = RealtimeSpectrumWindow._resolve_high_speed_ta_capture_block_samples(owner)

    assert samples == 240_000


@pytest.mark.parametrize("sweep_state", (SWEEP_STATE_SINGLE, SWEEP_STATE_RUNNING))
def test_16msps_2ms_power_trigger_uses_256k_island(sweep_state: str) -> None:
    owner = _snapshot_owner(
        trigger_kind="power_level",
        time_span_s=0.002,
        sample_rate_hz=16_000_000,
        sweep_state=sweep_state,
    )

    assert RealtimeSpectrumWindow._use_high_speed_ta_buffer_islands(owner)
    assert (
        RealtimeSpectrumWindow._resolve_high_speed_ta_capture_block_samples(owner)
        == 256_000
    )


def test_hsta_oversized_single_record_falls_back_to_stream_blocks() -> None:
    owner = _snapshot_owner(
        time_span_s=(HIGH_SPEED_TA_SNAPSHOT_MAX_SAMPLES + 1) / 12_000_000.0
    )

    samples = RealtimeSpectrumWindow._resolve_high_speed_ta_capture_block_samples(owner)

    assert samples == HIGH_SPEED_TA_CAPTURE_BLOCK_SAMPLES


def test_hsta_time_span_limits_follow_sample_rate_and_allow_100us() -> None:
    owner = _snapshot_owner(sample_rate_hz=16_000_000)

    minimum_s, maximum_s = RealtimeSpectrumWindow._high_speed_ta_time_span_limits_s(
        owner
    )

    assert minimum_s == pytest.approx(100e-6)
    assert maximum_s == pytest.approx(HIGH_SPEED_TA_SNAPSHOT_MAX_SAMPLES / 16_000_000)


def test_hsta_100us_snapshot_at_16msps_uses_1600_samples() -> None:
    owner = _snapshot_owner(time_span_s=100e-6, sample_rate_hz=16_000_000)

    samples = RealtimeSpectrumWindow._resolve_high_speed_ta_capture_block_samples(owner)

    assert samples == 1_600


def test_four_mhz_rbw_resolves_to_16msps() -> None:
    owner = SimpleNamespace()
    owner._clip_sweep_rbw = MethodType(RealtimeSpectrumWindow._clip_sweep_rbw, owner)
    owner._clamp_int = RealtimeSpectrumWindow._clamp_int

    rbw_hz, sample_rate_hz, _, _ = (
        RealtimeSpectrumWindow._resolve_sweep_like_capture_from_rbw(owner, 4_000_000.0)
    )

    assert rbw_hz == 4_000_000.0
    assert sample_rate_hz == 16_000_000


@pytest.mark.parametrize(
    ("time_span_s", "record_samples", "records_per_block"),
    ((0.002, 32_000, 3), (0.003, 48_000, 2)),
)
def test_16msps_continuous_uses_record_aligned_iq_islands(
    time_span_s: float,
    record_samples: int,
    records_per_block: int,
) -> None:
    owner = _snapshot_owner(
        time_span_s=time_span_s,
        sample_rate_hz=16_000_000,
        sweep_state=SWEEP_STATE_RUNNING,
    )

    assert RealtimeSpectrumWindow._use_high_speed_ta_buffer_islands(owner)
    block_samples = RealtimeSpectrumWindow._resolve_high_speed_ta_capture_block_samples(
        owner
    )

    assert block_samples == record_samples * records_per_block == 96_000


def test_hsta_power_trigger_builds_exact_time_positioned_record() -> None:
    config = SimpleNamespace(
        sample_rate_hz=1_000_000,
        fft_size=4096,
        time_analyzer_time_span_s=0.01,
        hsta_trigger_kind="power_level",
        hsta_trigger_run_mode="auto",
        hsta_trigger_slope="rising",
        hsta_trigger_level_dbm=-20.0,
        hsta_trigger_hysteresis_db=1.0,
        hsta_trigger_position_percent=25.0,
        hsta_trigger_auto_timeout_s=0.2,
        center_freq_hz=100_000_000,
        rx_bandwidth_hz=1_000_000,
        rx_gain_db=20,
    )
    owner = SimpleNamespace(
        config=config,
        sweep_state=SWEEP_STATE_RUNNING,
        _high_speed_ta_trigger_acquisition=None,
        _high_speed_ta_generation=0,
        _time_analyzer_time_span_s=lambda: config.time_analyzer_time_span_s,
        _clear_high_speed_ta_analysis_queues=lambda: None,
        _use_high_speed_ta_buffer_islands=lambda: False,
        _hsta_trigger_level_dbfs=lambda: -24.0,
    )

    acquisition = RealtimeSpectrumWindow._ensure_high_speed_ta_trigger_acquisition(owner)

    assert acquisition.config.record_samples == 10_000
    assert acquisition.config.pretrigger_samples == 2_500
    assert acquisition.config.posttrigger_samples == 7_499
    assert acquisition.config.auto_timeout_samples == 200_000
    assert acquisition.config.level_dbfs == -24.0


def _analysis_job(iq: np.ndarray) -> HighSpeedTAAnalysisJob:
    return HighSpeedTAAnalysisJob(
        iq_blocks=[iq],
        block_timestamps_s=[len(iq) / 1_000_000.0],
        gap_times_s=[],
        capture_total_s=len(iq) / 1_000_000.0,
        capture_call_count=1,
        capture_call_total_s=0.0,
        capture_total_samples=len(iq),
        block_sample_counts=[len(iq)],
        gap_count=0,
        gap_ratio_sum=0.0,
        max_gap_ratio=0.0,
        sample_rate_hz=1_000_000.0,
        display_points=16,
        rbw_hz=100_000.0,
        detector_mode="RMS",
        calibration_offset_db=0.0,
        frequency_dependent_offset_db=0.0,
        input_correction_db=0.0,
        remove_dc_offset=False,
        single_shot=False,
        sweep_id=0,
    )


def test_hsta_analysis_uses_iq_filter_before_power_detection() -> None:
    n = np.arange(4096, dtype=np.float64)
    center_iq = np.ones(4096, dtype=np.complex64)
    rejected_iq = np.exp(2j * np.pi * 200_000.0 * n / 1_000_000.0).astype(np.complex64)

    center = RealtimeSpectrumWindow._run_high_speed_ta_analysis_job(
        None,
        _analysis_job(center_iq),
    )
    rejected = RealtimeSpectrumWindow._run_high_speed_ta_analysis_job(
        None,
        _analysis_job(rejected_iq),
    )

    assert len(center.sweep_y_db) == 16
    assert center.sweep_y_db[-1] == pytest.approx(0.0, abs=1e-6)
    assert rejected.sweep_y_db[-1] < -45.0


def test_hsta_status_distinguishes_iq_samples_and_plot_points() -> None:
    status = format_hsta_sampling_status(40_000, 1000, 10e-6)
    assert status == "IQ Samples: 40000   Plot Points: 1000   Plot dt: 0.010 ms"


def test_hsta_power_trigger_line_uses_configured_display_dbm() -> None:
    class FakeLine:
        def __init__(self) -> None:
            self.position = None
            self.visible = None

        def setPos(self, value: float) -> None:
            self.position = value

        def setVisible(self, visible: bool) -> None:
            self.visible = visible

    line = FakeLine()
    owner = SimpleNamespace(
        config=SimpleNamespace(
            hsta_trigger_kind="power_level",
            hsta_trigger_level_dbm=-37.5,
        ),
        high_speed_ta_trigger_level_line=line,
        _is_high_speed_time_analyzer_mode=lambda: True,
    )

    RealtimeSpectrumWindow._update_hsta_trigger_level_line(owner)

    assert line.position == -37.5
    assert line.visible is True


def test_trigger_level_dialog_pauses_and_resumes_continuous_capture(
    monkeypatch,
) -> None:
    calls: list[str] = []

    class FakeTimer:
        def stop(self) -> None:
            calls.append("timer_stop")

    owner = SimpleNamespace(
        sweep_state=SWEEP_STATE_RUNNING,
        _high_speed_ta_stream_cursor=object(),
        _high_speed_ta_generation=4,
        timer=FakeTimer(),
        config=SimpleNamespace(hsta_trigger_level_dbm=-20.0),
        _stop_high_speed_ta_stream=lambda **kwargs: calls.append("stream_stop"),
        _clear_high_speed_ta_analysis_queues=lambda: calls.append("queues_clear"),
        _start_high_speed_time_analyzer_continuous=lambda: calls.append("resume"),
    )

    def cancel_dialog(*args, **kwargs):
        assert calls == ["timer_stop", "stream_stop", "queues_clear"]
        return -20.0, False

    monkeypatch.setattr(
        main_window_module.QtWidgets.QInputDialog,
        "getDouble",
        cancel_dialog,
    )

    RealtimeSpectrumWindow._on_hsta_trigger_level_clicked(owner)

    assert calls[-1] == "resume"
    assert owner.config.hsta_trigger_level_dbm == -20.0
