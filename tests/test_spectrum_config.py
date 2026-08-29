import pytest

from pluto_sa.config.spectrum_config import SpectrumConfig


def test_time_analyzer_defaults_to_ten_milliseconds() -> None:
    config = SpectrumConfig()
    assert config.time_analyzer_time_span_s == pytest.approx(0.010)
    assert config.time_analyzer_display_points == 1000


def test_realtime_fft_defaults_to_automatic_span_rbw_design() -> None:
    config = SpectrumConfig()

    assert config.realtime_fft_parameter_mode == "Auto"
    assert config.realtime_min_display_bins == 1024


def test_realtime_fft_parameter_mode_is_normalized() -> None:
    assert SpectrumConfig(realtime_fft_parameter_mode="advanced").realtime_fft_parameter_mode == "Advanced"
    assert SpectrumConfig(realtime_fft_parameter_mode="unexpected").realtime_fft_parameter_mode == "Auto"
