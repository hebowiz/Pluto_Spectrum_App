from types import SimpleNamespace

import numpy as np
import pytest

from pluto_sa.modes.sweep_controller import SweepController


def _controller(*, detector: str = "RMS") -> SweepController:
    controller = SweepController.__new__(SweepController)
    controller.config = SimpleNamespace(
        fft_size=1024,
        remove_dc_offset=False,
        rbw_hz=100_000.0,
        sweep_sample_rate_hz=1_000_000,
        sweep_detector_mode=detector,
        sweep_capture_samples_override=1024,
    )
    return controller


def _tone(frequency_hz: float, samples: int = 4096) -> np.ndarray:
    n = np.arange(samples, dtype=np.float64)
    return np.exp(2j * np.pi * frequency_hz * n / 1_000_000.0).astype(np.complex64)


def test_sweep_measurement_uses_iq_filter_3db_bandwidth() -> None:
    controller = _controller()
    center = controller._measure_point_power(_tone(0.0))[0]
    edge = controller._measure_point_power(_tone(50_000.0))[0]

    assert center == pytest.approx(1.0, abs=1e-6)
    assert 10.0 * np.log10(edge / center) == pytest.approx(-3.0103, abs=0.05)


def test_sweep_capture_includes_filter_settling_and_observation() -> None:
    controller = _controller()
    controller.config.rbw_hz = 100.0
    controller.config.sweep_sample_rate_hz = 521_000
    controller.config.sweep_capture_samples_override = 16_384

    # The shared Gaussian RBW filter now retains +/-6 sigma so its far-out
    # coherent sidelobes remain below the Pluto dynamic range.  Sweep capture
    # must include the correspondingly longer settling interval.
    assert controller._resolve_capture_samples() == 65_536
