import numpy as np
import pytest

from pluto_sa.signal.detector import apply_detector


def test_rms_detector_averages_linear_power() -> None:
    power = np.asarray([1.0, 4.0], dtype=np.float64)
    assert apply_detector(power, "RMS") == pytest.approx(2.5)


def test_sample_and_peak_detector_semantics() -> None:
    power = np.asarray([4.0, 1.0, 2.0], dtype=np.float64)
    assert apply_detector(power, "Sample") == pytest.approx(2.0)
    assert apply_detector(power, "Peak") == pytest.approx(4.0)
    assert apply_detector(power, "Negative Peak") == pytest.approx(1.0)


def test_average_detector_averages_linear_power() -> None:
    power = np.asarray([1.0, 4.0], dtype=np.float64)
    assert apply_detector(power, "Average") == pytest.approx(2.5)
