from __future__ import annotations

import numpy as np
import pytest

from pluto_sa.config.spectrum_config import SpectrumConfig
from pluto_sa.modes.analyzer_mode import AnalyzerMode
from pluto_sa.signal.fft_filterbank import (
    design_gaussian_fft_filterbank,
    required_gaussian_fft_size,
    resolve_automatic_rtsa_fft_design,
)
from pluto_sa.signal.measurement_filter import design_iq_rbw_filter
from pluto_sa.signal.spectrum_processor import SpectrumProcessor
from pluto_sa.ui.main_window import (
    MAX_REALTIME_FFT_SIZE,
    RealtimeSpectrumWindow,
    make_waterfall_lookup_table,
    resolve_waterfall_color_levels,
)


def _tone(sample_count: int, frequency_hz: float, sample_rate_hz: float) -> np.ndarray:
    indices = np.arange(sample_count, dtype=np.float64)
    return np.exp(2j * np.pi * frequency_hz * indices / sample_rate_hz).astype(
        np.complex64
    )


def test_gaussian_fft_filterbank_matches_iq_filter_metadata() -> None:
    sample_rate_hz = 4_000_000.0
    rbw_hz = 1_000_000.0

    window, design = design_gaussian_fft_filterbank(
        sample_rate_hz,
        4096,
        rbw_hz,
    )
    coefficients, iq_design = design_iq_rbw_filter(sample_rate_hz, rbw_hz)

    assert design.effective_rbw_hz == iq_design.effective_rbw_hz
    assert design.noise_equivalent_bandwidth_hz == pytest.approx(
        iq_design.noise_equivalent_bandwidth_hz
    )
    assert design.support_samples == len(coefficients)
    np.testing.assert_allclose(window[window != 0.0], coefficients)


def test_gaussian_fft_filterbank_has_two_sided_3db_rbw() -> None:
    sample_rate_hz = 4_000_000.0
    rbw_hz = 400_000.0
    fft_size = 4096
    window, design = design_gaussian_fft_filterbank(
        sample_rate_hz,
        fft_size,
        rbw_hz,
    )

    center_gain = abs(np.sum(window)) ** 2
    offset_tone = _tone(fft_size, rbw_hz / 2.0, sample_rate_hz)
    offset_gain = abs(np.sum(window * offset_tone)) ** 2

    assert design.effective_rbw_hz == rbw_hz
    assert offset_gain / center_gain == pytest.approx(0.5, abs=0.003)


def test_spectrum_processor_preserves_bin_center_cw_amplitude() -> None:
    config = SpectrumConfig(
        analyzer_mode=AnalyzerMode.REALTIME_SA,
        display_span_hz=4_000_000,
        fft_size=4096,
        rbw_hz=400_000.0,
    )
    processor = SpectrumProcessor(config)
    iq = np.ones(config.fft_size, dtype=np.complex64)

    power = processor.compute_filtered_power(iq)

    center = config.fft_size // 2
    assert power[center] == pytest.approx(1.0, abs=1e-12)
    half_rbw_bins = int(round((config.rbw_hz / 2.0) / config.bin_width_hz))
    assert power[center + half_rbw_bins] == pytest.approx(0.5, abs=0.01)


def test_spectrum_processor_batch_matches_individual_frames() -> None:
    config = SpectrumConfig(
        analyzer_mode=AnalyzerMode.REALTIME_SA,
        display_span_hz=4_000_000,
        fft_size=64,
        rbw_hz=400_000.0,
    )
    processor = SpectrumProcessor(config)
    frames = np.stack(
        (
            _tone(config.fft_size, 0.0, config.sample_rate_hz),
            _tone(config.fft_size, config.bin_width_hz, config.sample_rate_hz),
        )
    )

    batch = processor.compute_filtered_power_batch(frames)
    scalar = np.stack(
        [processor.compute_filtered_power(frame) for frame in frames]
    )

    np.testing.assert_allclose(batch, scalar, rtol=1e-6, atol=1e-12)


def test_too_narrow_rbw_is_limited_to_window_that_fits_fft() -> None:
    window, design = design_gaussian_fft_filterbank(
        4_000_000.0,
        64,
        100.0,
    )

    assert design.rbw_limited_by_fft_size is True
    assert design.effective_rbw_hz > design.requested_rbw_hz
    assert design.support_samples <= len(window)


def test_required_fft_size_contains_requested_gaussian_support() -> None:
    required_size = required_gaussian_fft_size(4_000_000.0, 10_000.0)
    window, design = design_gaussian_fft_filterbank(
        4_000_000.0,
        required_size,
        10_000.0,
    )

    assert required_size == 1024
    assert design.rbw_limited_by_fft_size is False
    assert design.support_samples <= len(window)


def test_automatic_rtsa_fft_separates_rbw_window_from_frequency_grid() -> None:
    design = resolve_automatic_rtsa_fft_design(
        sample_rate_hz=21_739_130.0,
        rbw_hz=1_000_000.0,
        guard_ratio=0.04,
        minimum_display_bins=1024,
    )

    assert design.window_length_samples == 49
    assert design.fft_size == 2048
    assert design.available_display_bins >= design.requested_display_bins
    assert design.limited_by_fft_size is False


def test_automatic_rtsa_fft_grows_for_narrow_rbw_support() -> None:
    design = resolve_automatic_rtsa_fft_design(
        sample_rate_hz=21_739_130.0,
        rbw_hz=10_000.0,
        guard_ratio=0.04,
        minimum_display_bins=1024,
    )

    assert design.window_length_samples == 4611
    assert design.fft_size == 8192
    assert design.limited_by_fft_size is False


def test_wideband_chunk_uses_same_gaussian_filterbank_processor() -> None:
    config = SpectrumConfig(
        analyzer_mode=AnalyzerMode.REALTIME_SA,
        display_span_hz=20_000_000,
        fft_size=4096,
        rbw_hz=1_000_000.0,
    )

    rtsa = SpectrumProcessor(config)
    wideband_chunk = SpectrumProcessor(config)

    assert wideband_chunk.filterbank_design == rtsa.filterbank_design
    np.testing.assert_array_equal(wideband_chunk.window, rtsa.window)


def test_realtime_rbw_expands_fft_to_fit_gaussian_window() -> None:
    config = SpectrumConfig(
        analyzer_mode=AnalyzerMode.REALTIME_SA,
        display_span_hz=20_000_000,
        fft_size=4096,
        rbw_hz=10_000.0,
    )
    updates: list[int] = []
    owner = type(
        "Owner",
        (),
        {
            "config": config,
            "_update_fft_menu_controls": lambda self: updates.append(
                int(self.config.fft_size)
            ),
        },
    )()

    changed = RealtimeSpectrumWindow._expand_realtime_fft_for_rbw(owner)

    assert changed is True
    assert config.fft_size == 8192
    assert updates == [8192]


def test_realtime_rbw_expansion_stops_at_supported_fft_limit() -> None:
    config = SpectrumConfig(
        analyzer_mode=AnalyzerMode.REALTIME_SA,
        display_span_hz=20_000_000,
        fft_size=4096,
        rbw_hz=100.0,
    )
    owner = type(
        "Owner",
        (),
        {
            "config": config,
            "_update_fft_menu_controls": lambda self: None,
        },
    )()

    assert RealtimeSpectrumWindow._expand_realtime_fft_for_rbw(owner) is True
    assert config.fft_size == MAX_REALTIME_FFT_SIZE
    processor = SpectrumProcessor(config)
    assert processor.filterbank_design.rbw_limited_by_fft_size is True


def test_advanced_realtime_fft_does_not_shrink_manual_fft_size() -> None:
    config = SpectrumConfig(
        analyzer_mode=AnalyzerMode.REALTIME_SA,
        display_span_hz=20_000_000,
        fft_size=16_384,
        rbw_hz=1_000_000.0,
        realtime_fft_parameter_mode="Advanced",
    )
    owner = type(
        "Owner",
        (),
        {
            "config": config,
            "_update_fft_menu_controls": lambda self: None,
        },
    )()

    assert RealtimeSpectrumWindow._expand_realtime_fft_for_rbw(owner) is False
    assert config.fft_size == 16_384


def test_waterfall_reaches_red_at_80_percent_of_measurement_range() -> None:
    assert resolve_waterfall_color_levels(-100.0, 0.0) == (-100.0, -20.0)


def test_waterfall_keeps_lower_15_percent_dark_navy() -> None:
    lookup_table = make_waterfall_lookup_table()
    navy_end_index = int(np.floor((0.15 / 0.80) * 255))

    np.testing.assert_array_equal(lookup_table[0], [0, 0, 128])
    np.testing.assert_array_equal(
        lookup_table[: navy_end_index + 1],
        np.tile(lookup_table[0], (navy_end_index + 1, 1)),
    )
    np.testing.assert_array_equal(lookup_table[-1], [255, 0, 0])
