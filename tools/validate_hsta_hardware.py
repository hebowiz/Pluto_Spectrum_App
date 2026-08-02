"""Offscreen, receive-only HighSpeed TA Single integration validation."""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pyqtgraph as pg

from pluto_sa.config.spectrum_config import SpectrumConfig
from pluto_sa.modes.analyzer_mode import AnalyzerMode
from pluto_sa.modes.sweep_controller import SweepController
from pluto_sa.sdr.pluto_receiver import PlutoReceiver
from pluto_sa.signal.spectrum_processor import SpectrumProcessor
from pluto_sa.ui.main_window import (
    RealtimeSpectrumWindow,
    SWEEP_STATE_SINGLE,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default=None)
    parser.add_argument("--sample-rate", type=int, default=4_000_000)
    parser.add_argument("--rbw", type=float, default=1_000_000.0)
    parser.add_argument("--time-span", type=float, default=0.1)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--continuous-duration", type=float, default=0.0)
    parser.add_argument("--exercise-transitions", action="store_true")
    parser.add_argument("--trigger-kind", choices=("free_run", "power_level"), default="free_run")
    parser.add_argument("--trigger-run-mode", choices=("auto", "normal"), default="auto")
    parser.add_argument("--trigger-level-dbm", type=float, default=-20.0)
    parser.add_argument("--trigger-position-percent", type=float, default=50.0)
    parser.add_argument("--trigger-auto-timeout", type=float, default=1.0)
    args = parser.parse_args()

    app = pg.mkQApp("Pluto HSTA hardware validation")
    config = SpectrumConfig(
        analyzer_mode=AnalyzerMode.HIGH_SPEED_TIME_ANALYZER,
        time_analyzer_sample_rate_hz=args.sample_rate,
        time_analyzer_rf_bandwidth_hz=min(args.sample_rate, 20_000_000),
        time_analyzer_time_span_s=args.time_span,
        rbw_hz=args.rbw,
        fft_size=4096,
        sdr_uri=args.uri,
        hsta_trigger_kind=args.trigger_kind,
        hsta_trigger_run_mode=args.trigger_run_mode,
        hsta_trigger_level_dbm=args.trigger_level_dbm,
        hsta_trigger_position_percent=args.trigger_position_percent,
        hsta_trigger_auto_timeout_s=args.trigger_auto_timeout,
    )
    receiver = PlutoReceiver(config)
    processor = SpectrumProcessor(config)
    sweep_controller = SweepController(config, receiver)
    window = RealtimeSpectrumWindow(
        config,
        receiver,
        processor,
        sweep_controller,
        calibration_offset_db=config.calibration_offset_db,
    )
    window.timer.stop()
    window.sweep_state = SWEEP_STATE_SINGLE

    started = time.perf_counter()
    completed = False
    error = None
    publish_count = 0
    max_job_queue_size = 0
    max_result_queue_size = 0
    transition_publish_counts: dict[str, int] = {}
    forced_trigger_count = 0
    natural_trigger_count = 0
    natural_trigger_measured_dbfs: list[float] = []
    published_display_min_dbm: list[float] = []
    published_display_max_dbm: list[float] = []
    update_call_durations_ms: list[float] = []
    original_publish = window._publish_high_speed_ta_analysis_result

    def counted_publish(result) -> None:
        nonlocal publish_count, forced_trigger_count, natural_trigger_count
        publish_count += 1
        finite_display = np.asarray(result.sweep_y_db, dtype=float)
        finite_display = finite_display[np.isfinite(finite_display)]
        if len(finite_display) > 0:
            published_display_min_dbm.append(float(np.min(finite_display)))
            published_display_max_dbm.append(float(np.max(finite_display)))
        if result.trigger_kind == "power_level":
            if result.trigger_forced:
                forced_trigger_count += 1
            else:
                natural_trigger_count += 1
                if result.trigger_measured_value is not None:
                    natural_trigger_measured_dbfs.append(
                        float(result.trigger_measured_value)
                    )
        original_publish(result)

    window._publish_high_speed_ta_analysis_result = counted_publish

    def update_hsta() -> None:
        update_started = time.perf_counter()
        window._update_high_speed_time_analyzer_spectrum()
        update_call_durations_ms.append(
            (time.perf_counter() - update_started) * 1000.0
        )

    try:
        if args.exercise_transitions:
            window._enter_single_high_speed_time_analyzer_mode()
            single_started = time.perf_counter()
            while (time.perf_counter() - single_started) < min(0.03, args.time_span * 0.25):
                app.processEvents()
                update_hsta()
                time.sleep(0.001)

            before_continuous = publish_count
            window._start_high_speed_time_analyzer_continuous()
            continuous_deadline = time.perf_counter() + max(0.75, args.time_span * 4.0)
            while time.perf_counter() < continuous_deadline:
                app.processEvents()
                update_hsta()
                time.sleep(0.001)
            transition_publish_counts["single_to_continuous"] = (
                publish_count - before_continuous
            )

            window.config.time_analyzer_time_span_s = max(0.001, args.time_span * 0.5)
            before_sweep_time_restart = publish_count
            window._start_high_speed_time_analyzer_continuous()
            restart_deadline = time.perf_counter() + max(0.75, args.time_span * 4.0)
            while time.perf_counter() < restart_deadline:
                app.processEvents()
                update_hsta()
                time.sleep(0.001)
            transition_publish_counts["sweep_time_restart"] = (
                publish_count - before_sweep_time_restart
            )
            completed = all(value > 0 for value in transition_publish_counts.values())
        else:
            continuous = args.continuous_duration > 0.0
            if continuous:
                window._start_high_speed_time_analyzer_continuous()
            else:
                window._enter_single_high_speed_time_analyzer_mode()
            while (time.perf_counter() - started) < args.timeout:
                app.processEvents()
                update_hsta()
                max_job_queue_size = max(
                    max_job_queue_size,
                    window._high_speed_ta_analysis_jobs.qsize(),
                )
                max_result_queue_size = max(
                    max_result_queue_size,
                    window._high_speed_ta_analysis_results.qsize(),
                )
                if continuous and (time.perf_counter() - started) >= args.continuous_duration:
                    completed = publish_count > 0
                    break
                display = window._last_current_display_db
                if (
                    not continuous
                    and display is not None
                    and len(display) > 0
                    and not window._high_speed_ta_single_waiting_result
                ):
                    completed = True
                    break
                time.sleep(0.001)
    except Exception as exc:
        error = repr(exc)
        raise
    finally:
        elapsed_s = time.perf_counter() - started
        stats = receiver.get_iq_stream_stats()
        window._stop_high_speed_ta_stream(stop_analysis_thread=True)
        receiver.close()
        result = {
            "connection_uri": receiver.connection_uri,
            "completed": completed,
            "error": error,
            "elapsed_s": elapsed_s,
            "requested_sample_rate_hz": args.sample_rate,
            "requested_rbw_hz": args.rbw,
            "requested_time_span_s": args.time_span,
            "continuous_duration_s": args.continuous_duration,
            "trigger_kind": args.trigger_kind,
            "trigger_run_mode": args.trigger_run_mode,
            "trigger_level_dbm": args.trigger_level_dbm,
            "trigger_level_internal_dbfs": window._hsta_trigger_level_dbfs(),
            "trigger_line_dbm": (
                None
                if window.high_speed_ta_trigger_level_line is None
                else float(window.high_speed_ta_trigger_level_line.value())
            ),
            "trigger_line_visible": (
                False
                if window.high_speed_ta_trigger_level_line is None
                else bool(window.high_speed_ta_trigger_level_line.isVisible())
            ),
            "trigger_position_percent": args.trigger_position_percent,
            "publish_count": publish_count,
            "forced_trigger_count": forced_trigger_count,
            "natural_trigger_count": natural_trigger_count,
            "natural_trigger_measured_dbfs_min": (
                None
                if not natural_trigger_measured_dbfs
                else min(natural_trigger_measured_dbfs)
            ),
            "natural_trigger_measured_dbfs_max": (
                None
                if not natural_trigger_measured_dbfs
                else max(natural_trigger_measured_dbfs)
            ),
            "published_display_dbm_min": (
                None if not published_display_min_dbm else min(published_display_min_dbm)
            ),
            "published_display_dbm_max": (
                None if not published_display_max_dbm else max(published_display_max_dbm)
            ),
            "display_points": (
                0
                if window._last_current_display_db is None
                else len(window._last_current_display_db)
            ),
            "published_blocks": stats.published_blocks,
            "published_samples": stats.published_samples,
            "overwritten_blocks": stats.overwritten_blocks,
            "job_queue_size": window._high_speed_ta_analysis_jobs.qsize(),
            "result_queue_size": window._high_speed_ta_analysis_results.qsize(),
            "pending_jobs": len(window._high_speed_ta_pending_analysis_jobs),
            "max_job_queue_size": max_job_queue_size,
            "max_result_queue_size": max_result_queue_size,
            "gui_update_p95_ms": (
                0.0
                if not update_call_durations_ms
                else float(np.percentile(update_call_durations_ms, 95.0))
            ),
            "gui_update_max_ms": (
                0.0
                if not update_call_durations_ms
                else float(max(update_call_durations_ms))
            ),
            "island_blocks": window._high_speed_time_analyzer.island_blocks,
            "island_records": window._high_speed_time_analyzer.island_records,
            "island_edge_rejections": (
                window._high_speed_time_analyzer.island_edge_rejections
            ),
            "island_blind_time_s": (
                window._high_speed_time_analyzer.island_blind_time_s
            ),
            "transition_publish_counts": transition_publish_counts,
        }
        print(json.dumps(result, indent=2))
    if not completed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
