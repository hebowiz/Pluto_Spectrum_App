import pytest

from pluto_sa.config.spectrum_config import SpectrumConfig


def test_time_analyzer_defaults_to_ten_milliseconds() -> None:
    config = SpectrumConfig()
    assert config.time_analyzer_time_span_s == pytest.approx(0.010)
    assert config.time_analyzer_display_points == 1000
