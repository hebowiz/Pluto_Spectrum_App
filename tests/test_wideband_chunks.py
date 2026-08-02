from __future__ import annotations

import numpy as np
import pytest

from pluto_sa.config.spectrum_config import SpectrumConfig
from pluto_sa.modes.analyzer_mode import AnalyzerMode
from pluto_sa.ui.main_window import (
    RealtimeSpectrumWindow,
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
