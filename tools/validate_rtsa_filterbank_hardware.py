"""Offscreen Pluto validation for RTSA/WB Gaussian FFT filter-bank paths."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyqtgraph as pg

from pluto_sa.config.spectrum_config import SpectrumConfig
from pluto_sa.modes.analyzer_mode import AnalyzerMode
from pluto_sa.modes.sweep_controller import SweepController
from pluto_sa.sdr.pluto_receiver import PlutoReceiver
from pluto_sa.signal.spectrum_processor import SpectrumProcessor
from pluto_sa.ui.main_window import (
    RealtimeSpectrumWindow,
    resolve_wideband_chunk_capture_span_hz,
)


def build_window(config: SpectrumConfig) -> tuple[RealtimeSpectrumWindow, PlutoReceiver]:
    receiver_config = config
    if config.analyzer_mode == AnalyzerMode.WIDEBAND_REALTIME_SA:
        receiver_config = deepcopy(config)
        receiver_config.analyzer_mode = AnalyzerMode.REALTIME_SA
        receiver_config.display_span_hz = resolve_wideband_chunk_capture_span_hz(
            config.wideband_chunk_width_hz
        )
    receiver = PlutoReceiver(receiver_config)
    processor = SpectrumProcessor(config)
    window = RealtimeSpectrumWindow(
        config,
        receiver,
        processor,
        SweepController(config, receiver),
        calibration_offset_db=config.calibration_offset_db,
    )
    window.timer.stop()
    return window, receiver


def summarize_display(window: RealtimeSpectrumWindow) -> dict[str, float | int | None]:
    axis = window._last_display_freq_axis_ghz
    display = window._last_current_display_db
    if axis is None or display is None or len(axis) != len(display):
        return {"points": 0, "peak_frequency_hz": None, "peak_dbm": None}
    valid = np.isfinite(axis) & np.isfinite(display)
    if not np.any(valid):
        return {"points": int(len(display)), "peak_frequency_hz": None, "peak_dbm": None}
    valid_indices = np.flatnonzero(valid)
    peak_local = int(np.argmax(display[valid]))
    peak_index = int(valid_indices[peak_local])
    return {
        "points": int(len(display)),
        "peak_frequency_hz": float(axis[peak_index] * 1e9),
        "peak_dbm": float(display[peak_index]),
    }


def run_realtime(app, args) -> dict:
    config = SpectrumConfig(
        analyzer_mode=AnalyzerMode.REALTIME_SA,
        center_freq_hz=args.center_frequency,
        display_span_hz=args.span,
        fft_size=args.fft_size,
        rbw_hz=args.rbw,
        sdr_uri=args.uri,
    )
    window, receiver = build_window(config)
    try:
        receiver.start()
        deadline = time.perf_counter() + args.timeout
        while time.perf_counter() < deadline:
            app.processEvents()
            window.update_spectrum()
            if window._last_current_display_db is not None:
                break
            time.sleep(0.001)
        stats = receiver.get_iq_stream_stats()
        design = window.processor.filterbank_design
        return {
            "mode": "rtsa",
            "completed": window._last_current_display_db is not None,
            "connection_uri": receiver.connection_uri,
            "effective_rbw_hz": design.effective_rbw_hz,
            "enbw_hz": design.noise_equivalent_bandwidth_hz,
            "support_samples": design.support_samples,
            "rbw_limited": design.rbw_limited_by_fft_size,
            "published_blocks": stats.published_blocks,
            "overwritten_blocks": stats.overwritten_blocks,
            **summarize_display(window),
        }
    finally:
        window.timer.stop()
        receiver.close()


def run_wideband(app, args) -> dict:
    config = SpectrumConfig(
        analyzer_mode=AnalyzerMode.WIDEBAND_REALTIME_SA,
        center_freq_hz=args.center_frequency,
        display_span_hz=args.span,
        fft_size=args.fft_size,
        rbw_hz=args.rbw,
        sdr_uri=args.uri,
        wideband_chunk_width_hz=args.chunk_width,
    )
    window, receiver = build_window(config)
    try:
        receiver.stop()
        window._invalidate_wideband_runtime()
        deadline = time.perf_counter() + args.timeout
        while time.perf_counter() < deadline:
            app.processEvents()
            window._update_wideband_spectrum()
            if window._last_current_display_db is not None:
                break
        chunk_processor = window._wideband_chunk_processor
        design = None if chunk_processor is None else chunk_processor.filterbank_design
        runtime = window._wideband_runtime_state
        return {
            "mode": "wideband_rtsa",
            "completed": window._last_current_display_db is not None,
            "connection_uri": receiver.connection_uri,
            "chunk_count": 0 if runtime is None else int(len(runtime.chunk_centers_hz)),
            "chunk_width_hz": int(config.wideband_chunk_width_hz),
            "sample_rate_hz": receiver.get_current_sample_rate_hz(),
            "rf_bandwidth_hz": receiver.get_current_rf_bandwidth_hz(),
            "effective_rbw_hz": None if design is None else design.effective_rbw_hz,
            "enbw_hz": None if design is None else design.noise_equivalent_bandwidth_hz,
            "support_samples": None if design is None else design.support_samples,
            "rbw_limited": None if design is None else design.rbw_limited_by_fft_size,
            **summarize_display(window),
        }
    finally:
        window.timer.stop()
        receiver.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default=None)
    parser.add_argument("--mode", choices=("rtsa", "wideband", "both"), default="both")
    parser.add_argument("--center-frequency", type=int, default=2_440_000_000)
    parser.add_argument("--span", type=int, default=20_000_000)
    parser.add_argument("--fft-size", type=int, default=4096)
    parser.add_argument("--rbw", type=float, default=1_000_000.0)
    parser.add_argument(
        "--chunk-width",
        type=int,
        choices=(10_000_000, 20_000_000, 30_000_000, 40_000_000),
        default=10_000_000,
    )
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()

    app = pg.mkQApp("RTSA Gaussian filter-bank hardware validation")
    if args.mode == "rtsa":
        results = [run_realtime(app, args)]
    elif args.mode == "wideband":
        results = [run_wideband(app, args)]
    else:
        results = [run_realtime(app, args)]
        # An automatic second connection may fall back to Pluto's IP context
        # on Windows while the first Qt/libiio context is being released.
        time.sleep(1.0)
        results.append(run_wideband(app, args))
    print(json.dumps(results, indent=2))
    if not all(bool(result["completed"]) for result in results):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
