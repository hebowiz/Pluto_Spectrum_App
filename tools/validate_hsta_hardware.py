"""Offscreen, receive-only HighSpeed TA Single integration validation."""

from __future__ import annotations

import argparse
import json
import os
import time

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
    parser.add_argument("--time-span", type=float, default=0.1)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--continuous-duration", type=float, default=0.0)
    parser.add_argument("--exercise-transitions", action="store_true")
    args = parser.parse_args()

    app = pg.mkQApp("Pluto HSTA hardware validation")
    config = SpectrumConfig(
        analyzer_mode=AnalyzerMode.HIGH_SPEED_TIME_ANALYZER,
        time_analyzer_sample_rate_hz=args.sample_rate,
        time_analyzer_rf_bandwidth_hz=min(args.sample_rate, 20_000_000),
        time_analyzer_time_span_s=args.time_span,
        fft_size=4096,
        sdr_uri=args.uri,
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
    original_publish = window._publish_high_speed_ta_analysis_result

    def counted_publish(result) -> None:
        nonlocal publish_count
        publish_count += 1
        original_publish(result)

    window._publish_high_speed_ta_analysis_result = counted_publish
    try:
        if args.exercise_transitions:
            window._enter_single_high_speed_time_analyzer_mode()
            single_started = time.perf_counter()
            while (time.perf_counter() - single_started) < min(0.03, args.time_span * 0.25):
                app.processEvents()
                window._update_high_speed_time_analyzer_spectrum()
                time.sleep(0.001)

            before_continuous = publish_count
            window._start_high_speed_time_analyzer_continuous()
            continuous_deadline = time.perf_counter() + max(0.75, args.time_span * 4.0)
            while time.perf_counter() < continuous_deadline:
                app.processEvents()
                window._update_high_speed_time_analyzer_spectrum()
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
                window._update_high_speed_time_analyzer_spectrum()
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
                window._update_high_speed_time_analyzer_spectrum()
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
            "requested_time_span_s": args.time_span,
            "continuous_duration_s": args.continuous_duration,
            "publish_count": publish_count,
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
            "transition_publish_counts": transition_publish_counts,
        }
        print(json.dumps(result, indent=2))
    if not completed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
