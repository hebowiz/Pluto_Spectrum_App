from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pluto_common.config.spectrum_config import SpectrumConfig
from pluto_common.config.analyzer_mode import AnalyzerMode
from pluto_rtsa.signal.spectrum_processor import SpectrumProcessor
from pluto_rtsa.ui.main_window import (
    RealtimeSpectrumWindow,
    WidebandRuntimeState,
    plan_wideband_chunks,
    resolve_wideband_chunk_capture_span_hz,
)


@pytest.mark.parametrize(
    ("chunk_width_hz", "capture_span_hz", "sample_rate_hz"),
    [
        (10_000_000, 20_000_000, 21_739_130),
        (20_000_000, 30_000_000, 32_608_696),
        (30_000_000, 40_000_000, 43_478_261),
        (40_000_000, 50_000_000, 54_347_826),
    ],
)
def test_wideband_chunk_capture_keeps_fixed_and_four_percent_guards(
    chunk_width_hz: int,
    capture_span_hz: int,
    sample_rate_hz: int,
) -> None:
    assert resolve_wideband_chunk_capture_span_hz(chunk_width_hz) == capture_span_hz
    config = SpectrumConfig(
        analyzer_mode=AnalyzerMode.WIDEBAND_REALTIME_SA,
        wideband_chunk_width_hz=chunk_width_hz,
    )
    owner = type(
        "Owner",
        (),
        {
            "config": config,
            "_get_wideband_start_stop_hz": lambda self: (
                100_000_000,
                200_000_000,
            ),
        },
    )()

    chunk_config = RealtimeSpectrumWindow._build_wideband_chunk_config(owner)

    assert chunk_config.display_span_hz == capture_span_hz
    assert chunk_config.sample_rate_hz == sample_rate_hz
    assert chunk_config.rx_bandwidth_hz == sample_rate_hz
    assert chunk_config.guard_ratio == pytest.approx(0.04)
    assert chunk_config.center_freq_hz == 100_000_000 + chunk_width_hz // 2


def test_wideband_chunks_start_at_lower_edge_and_clip_only_final_chunk() -> None:
    starts, centers, stops = plan_wideband_chunks(
        100_000_000,
        175_000_000,
        30_000_000,
    )

    np.testing.assert_array_equal(starts, [100_000_000, 130_000_000, 160_000_000])
    np.testing.assert_array_equal(centers, [115_000_000, 145_000_000, 175_000_000])
    np.testing.assert_array_equal(stops, [130_000_000, 160_000_000, 175_000_000])


def test_invalid_wideband_chunk_width_falls_back_to_10mhz() -> None:
    config = SpectrumConfig(wideband_chunk_width_hz=15_000_000)

    assert config.wideband_chunk_width_hz == 10_000_000


def test_wideband_capture_recreates_iio_buffer_after_each_retune() -> None:
    config = SpectrumConfig(
        analyzer_mode=AnalyzerMode.WIDEBAND_REALTIME_SA,
        fft_size=64,
        wideband_chunk_width_hz=10_000_000,
    )
    chunk_config = SpectrumConfig(
        analyzer_mode=AnalyzerMode.REALTIME_SA,
        center_freq_hz=105_000_000,
        display_span_hz=20_000_000,
        fft_size=64,
    )
    processor = SpectrumProcessor(chunk_config)
    display_bin_count = len(processor.get_display_freq_axis_ghz())

    class Receiver:
        def __init__(self) -> None:
            self.capture_calls: list[tuple[int, str, bool]] = []

        def retune_lo(self, center_hz: int, *, update_config: bool) -> None:
            assert center_hz == 105_000_000
            assert update_config is False

        def capture_iq_block(
            self,
            sample_count: int,
            *,
            source: str,
            fresh: bool,
        ) -> SimpleNamespace:
            self.capture_calls.append((sample_count, source, fresh))
            return SimpleNamespace(iq=np.zeros(sample_count, dtype=np.complex64))

    receiver = Receiver()
    owner = SimpleNamespace(
        config=config,
        receiver=receiver,
        _wideband_chunk_config=chunk_config,
        _wideband_chunk_processor=processor,
        _wideband_runtime_state=WidebandRuntimeState(
            start_hz=100_000_000,
            stop_hz=120_000_000,
            chunk_centers_hz=np.array([105_000_000, 115_000_000]),
            chunk_freq_ranges_hz=[
                (100_000_000, 110_000_000),
                (110_000_000, 120_000_000),
            ],
            chunk_source_ranges=[(0, display_bin_count), (0, display_bin_count)],
            chunk_slice_ranges=[
                (0, display_bin_count),
                (display_bin_count, 2 * display_bin_count),
            ],
            composite_freq_axis_ghz=np.zeros(2 * display_bin_count),
            composite_display_db=np.zeros(2 * display_bin_count),
        ),
        _apply_display_power_correction_with_frequency=lambda power, **_kwargs: power,
        _invalidate_wideband_runtime=lambda: None,
    )

    RealtimeSpectrumWindow._update_wideband_spectrum(owner)

    assert receiver.capture_calls == [(64, "wideband", True)]
