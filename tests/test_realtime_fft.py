from __future__ import annotations

import numpy as np
import pytest

from pluto_sa.config.spectrum_config import SpectrumConfig
from pluto_sa.modes.analyzer_mode import AnalyzerMode
from pluto_sa.signal.realtime_fft import (
    RealtimeFFTAccumulator,
    build_realtime_fft_plan,
)
from pluto_sa.signal.spectrum_processor import SpectrumProcessor


def _processor(fft_size: int = 64) -> SpectrumProcessor:
    return SpectrumProcessor(
        SpectrumConfig(
            analyzer_mode=AnalyzerMode.REALTIME_SA,
            display_span_hz=3_680_000,
            fft_size=fft_size,
            rbw_hz=400_000.0,
        )
    )


def test_realtime_fft_plan_uses_requested_overlap_when_load_allows() -> None:
    plan = build_realtime_fft_plan(
        sample_rate_hz=4_000_000.0,
        fft_size=4096,
        target_overlap_ratio=0.8,
        max_fft_rate_hz=25_000.0,
    )

    assert plan.hop_samples == 819
    assert plan.actual_overlap_ratio == pytest.approx(0.80005, abs=1e-4)
    assert plan.analysis_coverage_ratio == 1.0
    assert plan.quality == "Real-time"


def test_realtime_fft_plan_reports_analysis_gaps_instead_of_changing_fft() -> None:
    plan = build_realtime_fft_plan(
        sample_rate_hz=56_000_000.0,
        fft_size=64,
        target_overlap_ratio=0.8,
        max_fft_rate_hz=10_000.0,
    )

    assert plan.hop_samples == 5600
    assert plan.actual_overlap_ratio == 0.0
    assert plan.analysis_coverage_ratio == pytest.approx(64.0 / 5600.0)
    assert plan.quality == "Analysis gaps"


def test_realtime_fft_plan_uses_rbw_window_support_for_time_coverage() -> None:
    plan = build_realtime_fft_plan(
        sample_rate_hz=21_739_130.0,
        fft_size=4096,
        window_length_samples=49,
        target_overlap_ratio=0.8,
        max_fft_rate_hz=10_000.0,
    )

    assert plan.hop_samples == 2174
    assert plan.window_length_samples == 49
    assert plan.actual_overlap_ratio == 0.0
    assert plan.analysis_coverage_ratio == pytest.approx(49.0 / 2174.0)
    assert plan.quality == "Analysis gaps"


def test_overlap_accumulator_preserves_frame_grid_across_iq_blocks() -> None:
    processor = _processor()
    accumulator = RealtimeFFTAccumulator(
        processor,
        "Peak",
        target_overlap_ratio=0.5,
        max_fft_rate_hz=1_000_000.0,
    )
    rng = np.random.default_rng(1234)
    iq = (
        rng.normal(size=160) + 1j * rng.normal(size=160)
    ).astype(np.complex64)

    assert accumulator.process(iq[:50]) == 0
    expected_frame_count = (
        (len(iq) - processor.config.fft_size) // accumulator.plan.hop_samples
    ) + 1
    assert accumulator.process(iq[50:]) == expected_frame_count
    result = accumulator.take_frame()
    assert result is not None

    hop = accumulator.plan.hop_samples
    starts = range(0, len(iq) - 64 + 1, hop)
    expected_frames = np.stack([iq[start : start + 64] for start in starts])
    expected = np.max(
        processor.compute_filtered_power_batch(expected_frames),
        axis=0,
    )
    np.testing.assert_allclose(result.power_linear, expected)
    assert result.fft_frames == expected_frame_count
    assert result.input_samples == 160


def test_gap_limited_accumulator_keeps_hop_phase_across_blocks() -> None:
    processor = _processor()
    accumulator = RealtimeFFTAccumulator(
        processor,
        "Sample",
        target_overlap_ratio=0.0,
        max_fft_rate_hz=1_000.0,
    )
    first = np.ones(100, dtype=np.complex64)
    second = np.full(64, 2.0 + 0.0j, dtype=np.complex64)

    assert accumulator.process(first) == 1
    first_result = accumulator.take_frame()
    assert first_result is not None
    assert accumulator.process(np.zeros(3900, dtype=np.complex64)) == 0
    assert accumulator.process(second) == 1
    second_result = accumulator.take_frame()
    assert second_result is not None

    expected = processor.compute_filtered_power(second)
    np.testing.assert_allclose(second_result.power_linear, expected)


def test_discontinuity_discards_partial_overlap_history() -> None:
    processor = _processor()
    accumulator = RealtimeFFTAccumulator(
        processor,
        "Sample",
        target_overlap_ratio=0.5,
        max_fft_rate_hz=1_000_000.0,
    )

    assert accumulator.process(np.ones(40, dtype=np.complex64)) == 0
    assert accumulator.process(
        np.full(64, 2.0 + 0.0j, dtype=np.complex64),
        discontinuity_before=True,
    ) == 1
    result = accumulator.take_frame()
    assert result is not None
    assert result.discontinuities == 1
    np.testing.assert_allclose(
        result.power_linear,
        processor.compute_filtered_power(np.full(64, 2.0 + 0.0j, dtype=np.complex64)),
    )


def test_sample_index_gap_discards_partial_overlap_history() -> None:
    processor = _processor()
    accumulator = RealtimeFFTAccumulator(
        processor,
        "Sample",
        target_overlap_ratio=0.5,
        max_fft_rate_hz=1_000_000.0,
    )

    assert accumulator.process(
        np.ones(40, dtype=np.complex64),
        start_sample_index=0,
    ) == 0
    after_gap = np.full(64, 2.0 + 0.0j, dtype=np.complex64)
    assert accumulator.process(
        after_gap,
        # 40 was expected.  An unflagged jump must still form a hard FFT
        # boundary rather than concatenating the two waveform islands.
        start_sample_index=100,
    ) == 1
    result = accumulator.take_frame()
    assert result is not None
    assert result.discontinuities == 1
    np.testing.assert_allclose(
        result.power_linear,
        processor.compute_filtered_power(after_gap),
    )


def test_contiguous_sample_indices_keep_overlap_history() -> None:
    processor = _processor()
    accumulator = RealtimeFFTAccumulator(
        processor,
        "Sample",
        target_overlap_ratio=0.5,
        max_fft_rate_hz=1_000_000.0,
    )

    first = np.ones(40, dtype=np.complex64)
    second = np.full(24, 2.0 + 0.0j, dtype=np.complex64)
    assert accumulator.process(first, start_sample_index=200) == 0
    assert accumulator.process(second, start_sample_index=240) == 1
    result = accumulator.take_frame()
    assert result is not None
    assert result.discontinuities == 0
    np.testing.assert_allclose(
        result.power_linear,
        processor.compute_filtered_power(np.concatenate((first, second))),
    )
