"""Sweep SA execution controller skeleton."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

import numpy as np

from pluto_sa.config.spectrum_config import SpectrumConfig
from pluto_sa.sdr.pluto_receiver import PlutoReceiver
from pluto_sa.signal.detector import apply_detector
from pluto_sa.signal.rbw import (
    apply_rbw_weighting,
    make_gaussian_rbw_kernel,
    resolve_rbw_hz,
)


@dataclass
class SweepFrameResult:
    """Latest sweep frame snapshot passed from controller to UI."""

    freq_axis_hz: np.ndarray | None = None
    display_db: np.ndarray | None = None
    completed_points: int = 0
    sweep_complete: bool = False


@dataclass
class SweepPointResult:
    """One measured sweep point after RBW and detector processing."""

    frequency_hz: int
    measured_power_linear: float
    measured_power_db: float
    capture_samples: int
    effective_rbw_hz: float
    bin_width_hz: float
    peak_bin_index_unshifted: int
    peak_bin_index_shifted: int
    peak_frequency_relative_hz: float
    rbw_center_bin_index: int
    rbw_center_frequency_hz: float
    detector_input_values: np.ndarray
    retune_ms: float = 0.0
    settle_wait_ms: float = 0.0
    flush_ms: float = 0.0
    capture_ms: float = 0.0
    process_ms: float = 0.0
    step_total_ms: float = 0.0


class SweepController:
    """Own Sweep SA run-state without owning SDR lifetime management."""

    def __init__(self, config: SpectrumConfig, receiver: PlutoReceiver) -> None:
        self.config = config
        self.receiver = receiver
        self._run_requested = False
        self._single_requested = False
        self._restart_pending = False
        self._current_point_index = 0
        self._latest_result = SweepFrameResult()
        self._latest_point_result: SweepPointResult | None = None
        self._sweep_complete_callback: Callable[[SweepFrameResult], None] | None = None
        self._sweep_freq_axis_hz = self._build_sweep_frequency_axis_hz()
        self._partial_power_db = np.full(config.sweep_points, np.nan, dtype=np.float64)
        self.reset()

    def reset(self) -> None:
        """Reset all sweep-progress state for a future run."""
        self._run_requested = False
        self._single_requested = False
        self._restart_pending = False
        self._prepare_next_sweep_cycle()

    def stop(self) -> None:
        """Stop any future sweep progress requests."""
        self._run_requested = False
        self._single_requested = False
        self._restart_pending = False

    def request_continuous(self) -> None:
        """Request continuous sweep execution."""
        if self._latest_result.sweep_complete or self._restart_pending:
            self._prepare_next_sweep_cycle()
        self._run_requested = True
        self._single_requested = False
        self._restart_pending = False

    def request_single(self) -> None:
        """Request one complete sweep execution."""
        if self._latest_result.sweep_complete or self._restart_pending:
            self._prepare_next_sweep_cycle()
        self._run_requested = True
        self._single_requested = True
        self._restart_pending = False

    def is_running(self) -> bool:
        """Return whether sweep execution is currently requested."""
        return self._run_requested

    def get_latest_result(self) -> SweepFrameResult:
        """Return the latest published sweep result snapshot."""
        return self._latest_result

    def get_latest_point_result(self) -> SweepPointResult | None:
        """Return the latest measured single-point result."""
        return self._latest_point_result

    def set_sweep_complete_callback(
        self,
        callback: Callable[[SweepFrameResult], None] | None,
    ) -> None:
        """Set an optional callback fired when one sweep completes."""
        self._sweep_complete_callback = callback

    def measure_point(self, frequency_hz: int) -> SweepPointResult:
        """Measure one sweep point using the fixed Sweep SA signal path."""
        self.receiver.configure_for_sweep(self.config)
        point_start = time.perf_counter()

        retune_start = point_start
        self.receiver.retune_lo(int(frequency_hz), update_config=False)
        retune_ms = (time.perf_counter() - retune_start) * 1000.0

        settle_start = time.perf_counter()
        self._wait_for_lo_settle()
        settle_wait_ms = (time.perf_counter() - settle_start) * 1000.0

        capture_samples = self._resolve_capture_samples()
        flush_start = time.perf_counter()
        self._flush_post_retune_buffers(capture_samples)
        flush_ms = (time.perf_counter() - flush_start) * 1000.0

        capture_start = time.perf_counter()
        iq = self.receiver.capture_block(capture_samples)
        capture_ms = (time.perf_counter() - capture_start) * 1000.0

        process_start = time.perf_counter()
        (
            measured_power_linear,
            effective_rbw_hz,
            bin_width_hz,
            peak_bin_index_unshifted,
            peak_bin_index_shifted,
            peak_frequency_relative_hz,
            rbw_center_bin_index,
            rbw_center_frequency_hz,
            detector_input_values,
        ) = self._measure_point_power(iq)
        process_ms = (time.perf_counter() - process_start) * 1000.0
        measured_power_db = 10.0 * np.log10(measured_power_linear + 1e-20)
        step_total_ms = (time.perf_counter() - point_start) * 1000.0

        point_result = SweepPointResult(
            frequency_hz=int(frequency_hz),
            measured_power_linear=measured_power_linear,
            measured_power_db=measured_power_db,
            capture_samples=capture_samples,
            effective_rbw_hz=effective_rbw_hz,
            bin_width_hz=bin_width_hz,
            peak_bin_index_unshifted=peak_bin_index_unshifted,
            peak_bin_index_shifted=peak_bin_index_shifted,
            peak_frequency_relative_hz=peak_frequency_relative_hz,
            rbw_center_bin_index=rbw_center_bin_index,
            rbw_center_frequency_hz=rbw_center_frequency_hz,
            detector_input_values=detector_input_values.copy(),
            retune_ms=retune_ms,
            settle_wait_ms=settle_wait_ms,
            flush_ms=flush_ms,
            capture_ms=capture_ms,
            process_ms=process_ms,
            step_total_ms=step_total_ms,
        )
        self._latest_point_result = point_result
        return point_result

    def measure_points(self, frequencies_hz: np.ndarray) -> list[SweepPointResult]:
        """Measure a series of independent sweep points for debug or probing use."""
        return [self.measure_point(int(frequency_hz)) for frequency_hz in frequencies_hz]

    def step_sweep(self) -> SweepFrameResult:
        """Measure and publish the next sweep point in left-to-right order."""
        if not self._run_requested:
            return self._latest_result

        if self._restart_pending:
            self._prepare_next_sweep_cycle()
            self._restart_pending = False

        if self._current_point_index >= len(self._sweep_freq_axis_hz):
            if self._single_requested:
                self.stop()
                return self._latest_result
            self._prepare_next_sweep_cycle()

        frequency_hz = int(self._sweep_freq_axis_hz[self._current_point_index])
        point_result = self.measure_point(frequency_hz)
        self._partial_power_db[self._current_point_index] = point_result.measured_power_db
        self._current_point_index += 1

        sweep_complete = self._current_point_index >= len(self._sweep_freq_axis_hz)
        self._publish_partial_result(sweep_complete=sweep_complete)

        if sweep_complete:
            self._handle_sweep_complete()

        return self._latest_result

    def run_single_sweep(self) -> SweepFrameResult:
        """Run one full sweep internally and retain the completed result."""
        self.request_single()
        while self._run_requested:
            self.step_sweep()
        return self._latest_result

    def get_sweep_frequency_axis_hz(self) -> np.ndarray:
        """Return the configured sweep frequency axis."""
        return self._sweep_freq_axis_hz.copy()

    def _build_sweep_frequency_axis_hz(self) -> np.ndarray:
        return np.linspace(
            self.config.sweep_start_freq_hz,
            self.config.sweep_stop_freq_hz,
            self.config.sweep_points,
            dtype=np.float64,
        )

    def _prepare_next_sweep_cycle(self) -> None:
        self._current_point_index = 0
        self._latest_point_result = None
        self._sweep_freq_axis_hz = self._build_sweep_frequency_axis_hz()
        self._partial_power_db = np.full(self.config.sweep_points, np.nan, dtype=np.float64)
        self._latest_result = SweepFrameResult(
            freq_axis_hz=self._sweep_freq_axis_hz.copy(),
            display_db=self._partial_power_db.copy(),
            completed_points=0,
            sweep_complete=False,
        )

    def _publish_partial_result(self, sweep_complete: bool) -> None:
        self._latest_result = SweepFrameResult(
            freq_axis_hz=self._sweep_freq_axis_hz.copy(),
            display_db=self._partial_power_db.copy(),
            completed_points=self._current_point_index,
            sweep_complete=sweep_complete,
        )

    def _handle_sweep_complete(self) -> None:
        if self._sweep_complete_callback is not None:
            self._sweep_complete_callback(self._latest_result)

        if self._single_requested:
            self.stop()
        else:
            self._restart_pending = True

    def _wait_for_lo_settle(self) -> None:
        time.sleep(max(0.0, self.config.sweep_lo_settle_us) / 1_000_000.0)

    def _flush_post_retune_buffers(self, capture_samples: int) -> None:
        flush_reads = max(0, int(self.config.sweep_retune_flush_reads))
        for _ in range(flush_reads):
            self.receiver.capture_block(capture_samples)

    def _resolve_capture_samples(self) -> int:
        if self.config.sweep_capture_samples_override is not None:
            return max(1, int(self.config.sweep_capture_samples_override))

        effective_rbw_hz = self._resolve_effective_rbw_hz()
        observation_ratio = self.config.sweep_sample_rate_hz / effective_rbw_hz
        min_samples = max(256, int(np.ceil(observation_ratio * 8.0)))
        capture_samples = 1 << int(np.ceil(np.log2(min_samples)))
        return int(capture_samples)

    def _build_signed_frequency_axis_hz(self, sample_count: int) -> np.ndarray:
        bin_indices = np.arange(sample_count, dtype=np.float64) - (sample_count // 2)
        return bin_indices * (self.config.sweep_sample_rate_hz / sample_count)

    def _resolve_effective_rbw_hz(self) -> float:
        min_bin_width_hz = self.config.sweep_sample_rate_hz / 4096.0
        return resolve_rbw_hz(self.config.rbw_hz, min_bin_width_hz)

    def _measure_point_power(
        self,
        iq: np.ndarray,
    ) -> tuple[float, float, float, int, int, float, int, float, np.ndarray]:
        iq = iq - np.mean(iq)
        window = np.hanning(len(iq))
        iq_windowed = iq * window
        spectrum_unshifted = np.fft.fft(iq_windowed)
        spectrum = np.fft.fftshift(spectrum_unshifted)

        n = len(iq)
        coherent_gain = np.sum(window) / n
        spectrum_unshifted = spectrum_unshifted / n
        spectrum_unshifted = spectrum_unshifted / coherent_gain
        spectrum = spectrum / n
        spectrum = spectrum / coherent_gain

        power_spectrum = np.abs(spectrum) ** 2
        bin_width_hz = self.config.sweep_sample_rate_hz / n
        freq_axis_hz = self._build_signed_frequency_axis_hz(n)
        power_spectrum_unshifted = np.abs(spectrum_unshifted) ** 2
        effective_rbw_hz = resolve_rbw_hz(self.config.rbw_hz, bin_width_hz)
        rbw_kernel = make_gaussian_rbw_kernel(effective_rbw_hz, bin_width_hz)
        filtered_power = apply_rbw_weighting(power_spectrum, rbw_kernel)

        center_idx = len(filtered_power) // 2
        detector_input = np.array([filtered_power[center_idx]], dtype=np.float64)
        measured_power_linear = apply_detector(detector_input, self.config.sweep_detector_mode)
        peak_bin_index_unshifted = int(np.argmax(power_spectrum_unshifted))
        peak_bin_index_shifted = int(np.argmax(power_spectrum))
        peak_frequency_relative_hz = float(freq_axis_hz[peak_bin_index_shifted])
        rbw_center_bin_index = center_idx
        rbw_center_frequency_hz = float(freq_axis_hz[center_idx])
        return (
            measured_power_linear,
            effective_rbw_hz,
            bin_width_hz,
            peak_bin_index_unshifted,
            peak_bin_index_shifted,
            peak_frequency_relative_hz,
            rbw_center_bin_index,
            rbw_center_frequency_hz,
            detector_input,
        )
