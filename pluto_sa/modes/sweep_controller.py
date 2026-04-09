"""Sweep SA execution controller skeleton."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

import numpy as np

from pluto_sa.config.spectrum_config import SpectrumConfig
from pluto_sa.sdr.pluto_receiver import PlutoReceiver
from pluto_sa.signal.detector import DetectorMode, apply_detector
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
    active_point_index: int | None = None


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
    flush_reads: int = 0
    flush_samples: int = 0
    flush_actual_samples_total: int = 0
    flush_actual_samples_per_read: int = 0
    configure_ms: float = 0.0
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
        self._current_sweep_started_at: float | None = None
        self._next_sweep_start_time: float | None = None
        self.reset()

    def reset(self) -> None:
        """Reset all sweep-progress state for a future run."""
        self._run_requested = False
        self._single_requested = False
        self._restart_pending = False
        self._current_sweep_started_at = None
        self._next_sweep_start_time = None
        self._partial_power_db = np.full(self.config.sweep_points, np.nan, dtype=np.float64)
        self._prepare_next_sweep_cycle()

    def stop(self) -> None:
        """Stop any future sweep progress requests."""
        self._run_requested = False
        self._single_requested = False
        self._restart_pending = False
        self._current_sweep_started_at = None
        self._next_sweep_start_time = None

    def request_continuous(self) -> None:
        """Request continuous sweep execution."""
        if self._latest_result.sweep_complete or self._restart_pending:
            self._prepare_next_sweep_cycle()
        self._run_requested = True
        self._single_requested = False
        self._restart_pending = False
        self._current_sweep_started_at = None
        self._next_sweep_start_time = None

    def request_single(self) -> None:
        """Request one complete sweep execution."""
        if self._latest_result.sweep_complete or self._restart_pending:
            self._prepare_next_sweep_cycle()
        self._run_requested = True
        self._single_requested = True
        self._restart_pending = False
        self._current_sweep_started_at = None
        self._next_sweep_start_time = None

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
        point_start = time.perf_counter()

        configure_start = point_start
        self.receiver.configure_for_sweep(self.config)
        configure_ms = (time.perf_counter() - configure_start) * 1000.0

        retune_start = time.perf_counter()
        self.receiver.retune_lo(int(frequency_hz), update_config=False)
        retune_ms = (time.perf_counter() - retune_start) * 1000.0

        settle_start = time.perf_counter()
        self._wait_for_lo_settle()
        settle_wait_ms = (time.perf_counter() - settle_start) * 1000.0

        capture_samples = self._resolve_capture_samples()
        flush_samples = min(self.config.sweep_flush_samples, capture_samples)
        flush_start = time.perf_counter()
        (
            flush_reads,
            flush_actual_samples_total,
            flush_actual_samples_per_read,
        ) = self._flush_post_retune_buffers(capture_samples)
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
            flush_reads=flush_reads,
            flush_samples=int(flush_samples),
            flush_actual_samples_total=flush_actual_samples_total,
            flush_actual_samples_per_read=flush_actual_samples_per_read,
            configure_ms=configure_ms,
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
            self._current_sweep_started_at = time.perf_counter()
            self._next_sweep_start_time = None

        if self._current_point_index >= len(self._sweep_freq_axis_hz):
            if self._single_requested:
                self.stop()
                return self._latest_result
            self._prepare_next_sweep_cycle()

        if self._current_point_index == 0 and self._current_sweep_started_at is None:
            self._current_sweep_started_at = time.perf_counter()

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

    def estimate_capture_samples(self) -> int:
        """Return the current capture-size estimate used by Sweep SA."""
        return self._resolve_capture_samples()

    def estimate_point_time_s(self) -> float:
        """Estimate one sweep-point duration from the current configuration."""
        capture_samples = self._resolve_capture_samples()
        flush_samples = max(1, int(min(self.config.sweep_flush_samples, capture_samples)))
        discard_samples_per_read = max(flush_samples, int(self.config.fft_size))
        retune_s = 0.0015
        settle_s = max(0.0, self.config.sweep_lo_settle_us) / 1_000_000.0
        flush_s = (
            self.config.sweep_retune_flush_reads
            * discard_samples_per_read
            / self.config.sweep_sample_rate_hz
        )
        capture_s = capture_samples / self.config.sweep_sample_rate_hz
        process_s = 0.001
        return retune_s + settle_s + flush_s + capture_s + process_s

    def estimate_minimum_sweep_time_s(self) -> float:
        """Estimate the minimum realizable sweep time for the current configuration."""
        return self.config.sweep_points * self.estimate_point_time_s()

    def get_actual_sweep_time_s(self) -> float:
        """Return the current minimum realizable sweep time used for UI display."""
        return self.estimate_minimum_sweep_time_s()

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
        if len(self._partial_power_db) != self.config.sweep_points:
            self._partial_power_db = np.full(self.config.sweep_points, np.nan, dtype=np.float64)
        self._latest_result = SweepFrameResult(
            freq_axis_hz=self._sweep_freq_axis_hz.copy(),
            display_db=self._partial_power_db.copy(),
            completed_points=0,
            sweep_complete=False,
            active_point_index=None,
        )

    def _publish_partial_result(self, sweep_complete: bool) -> None:
        self._latest_result = SweepFrameResult(
            freq_axis_hz=self._sweep_freq_axis_hz.copy(),
            display_db=self._partial_power_db.copy(),
            completed_points=self._current_point_index,
            sweep_complete=sweep_complete,
            active_point_index=None if sweep_complete else max(0, self._current_point_index - 1),
        )

    def _handle_sweep_complete(self) -> None:
        if self._sweep_complete_callback is not None:
            self._sweep_complete_callback(self._latest_result)

        if self._single_requested:
            self.stop()
        else:
            self._restart_pending = True
            self._next_sweep_start_time = None
            self._current_sweep_started_at = None

    def _wait_for_lo_settle(self) -> None:
        time.sleep(max(0.0, self.config.sweep_lo_settle_us) / 1_000_000.0)

    def _flush_post_retune_buffers(self, capture_samples: int) -> tuple[int, int, int]:
        flush_reads = max(0, int(self.config.sweep_retune_flush_reads))
        flush_samples = max(1, int(min(self.config.sweep_flush_samples, capture_samples)))
        flushed_total = 0
        actual_per_read = 0
        for _ in range(flush_reads):
            actual_per_read = self.receiver.discard_block(flush_samples)
            flushed_total += actual_per_read
        return flush_reads, flushed_total, actual_per_read

    def _resolve_capture_samples(self) -> int:
        effective_rbw_hz = self._resolve_effective_rbw_hz()
        observation_ratio = self.config.sweep_sample_rate_hz / effective_rbw_hz
        min_samples = max(256, int(np.ceil(observation_ratio * 8.0)))
        if self.config.sweep_capture_samples_override is not None:
            min_samples = max(min_samples, int(self.config.sweep_capture_samples_override))
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
        n = len(iq)

        window = np.hanning(n)
        iq_windowed = iq * window
        spectrum_unshifted = np.fft.fft(iq_windowed)
        spectrum = np.fft.fftshift(spectrum_unshifted)

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
        detector_input = self._build_detector_observation_series(iq, effective_rbw_hz)
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

    def _build_detector_observation_series(
        self,
        iq: np.ndarray,
        effective_rbw_hz: float,
    ) -> np.ndarray:
        """Build a time-direction observation series for detector reduction."""
        total_samples = len(iq)
        if total_samples < 2:
            return np.asarray([0.0], dtype=np.float64)

        segment_length = 1 << int(np.floor(np.log2(max(2, total_samples // 2))))
        segment_length = max(256, segment_length)
        segment_length = min(segment_length, total_samples)
        if segment_length < 2:
            segment_length = total_samples

        step = max(1, segment_length // 2)
        starts = list(range(0, max(1, total_samples - segment_length + 1), step))
        if not starts:
            starts = [0]
        if starts[-1] != total_samples - segment_length:
            starts.append(total_samples - segment_length)

        observation_values: list[float] = []
        for start in starts:
            segment = iq[start : start + segment_length]
            if len(segment) < segment_length:
                continue

            segment = segment - np.mean(segment)
            window = np.hanning(segment_length)
            coherent_gain = np.sum(window) / segment_length
            spectrum = np.fft.fftshift(np.fft.fft(segment * window))
            spectrum = spectrum / segment_length
            spectrum = spectrum / coherent_gain

            power_spectrum = np.abs(spectrum) ** 2
            segment_bin_width_hz = self.config.sweep_sample_rate_hz / segment_length
            rbw_kernel = make_gaussian_rbw_kernel(effective_rbw_hz, segment_bin_width_hz)
            filtered_power = apply_rbw_weighting(power_spectrum, rbw_kernel)
            center_idx = len(filtered_power) // 2
            observation_values.append(float(filtered_power[center_idx]))

        if not observation_values:
            observation_values.append(0.0)
        return np.asarray(observation_values, dtype=np.float64)
