import numpy as np
import pytest

from pluto_sa.signal.measurement_filter import (
    StatefulIQMeasurementFilter,
    design_iq_rbw_filter,
    reduce_filtered_iq_power,
    reduce_filtered_iq_power_buckets,
)


def _tone(sample_rate_hz: float, frequency_hz: float, samples: int) -> np.ndarray:
    n = np.arange(samples, dtype=np.float64)
    return np.exp(2j * np.pi * frequency_hz * n / sample_rate_hz).astype(np.complex64)


def test_filter_state_is_continuous_across_blocks() -> None:
    sample_rate_hz = 1_000_000.0
    iq = _tone(sample_rate_hz, 20_000.0, 8192)

    whole = StatefulIQMeasurementFilter(sample_rate_hz, 100_000.0)
    split = StatefulIQMeasurementFilter(sample_rate_hz, 100_000.0)

    expected = whole.process(iq)
    actual = np.concatenate((split.process(iq[:1234]), split.process(iq[1234:])))

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_rbw_is_full_two_sided_3db_bandwidth() -> None:
    sample_rate_hz = 1_000_000.0
    rbw_hz = 100_000.0
    samples = 40_000

    center_filter = StatefulIQMeasurementFilter(sample_rate_hz, rbw_hz)
    edge_filter = StatefulIQMeasurementFilter(sample_rate_hz, rbw_hz)
    center_filter.reset(1.0 + 0.0j)
    center = center_filter.process(_tone(sample_rate_hz, 0.0, samples))
    edge = edge_filter.process(_tone(sample_rate_hz, rbw_hz / 2.0, samples))

    center_power = np.mean(np.abs(center[-4096:]) ** 2)
    edge_power = np.mean(np.abs(edge[-4096:]) ** 2)
    assert 10.0 * np.log10(edge_power / center_power) == pytest.approx(-3.0103, abs=0.05)


def test_out_of_band_tone_is_rejected() -> None:
    sample_rate_hz = 1_000_000.0
    measurement_filter = StatefulIQMeasurementFilter(sample_rate_hz, 100_000.0)
    output = measurement_filter.process(_tone(sample_rate_hz, 200_000.0, 40_000))
    power_db = 10.0 * np.log10(np.mean(np.abs(output[-4096:]) ** 2))
    assert power_db < -45.0


def test_design_reports_enbw_and_settling() -> None:
    taps, design = design_iq_rbw_filter(1_000_000.0, 100_000.0)
    assert design.effective_rbw_hz == 100_000.0
    assert design.cutoff_hz == 50_000.0
    assert design.filter_shape == "gaussian"
    assert design.tap_count == taps.size
    assert design.group_delay_samples == (taps.size - 1) / 2
    assert design.noise_equivalent_bandwidth_hz / design.effective_rbw_hz == pytest.approx(
        1.0645,
        rel=2e-3,
    )
    assert design.settling_samples >= 16


def test_default_gaussian_filter_has_symmetric_unit_gain_taps() -> None:
    taps, design = design_iq_rbw_filter(4_000_000.0, 1_000_000.0)

    np.testing.assert_allclose(taps, taps[::-1], rtol=0.0, atol=0.0)
    assert np.sum(taps) == pytest.approx(1.0)
    assert design.tap_count == 11


def test_narrow_gaussian_fft_filter_is_continuous_across_blocks() -> None:
    sample_rate_hz = 521_000.0
    iq = _tone(sample_rate_hz, 200.0, 12_000)
    whole = StatefulIQMeasurementFilter(sample_rate_hz, 1_000.0)
    split = StatefulIQMeasurementFilter(sample_rate_hz, 1_000.0)

    expected = whole.process(iq)
    actual = np.concatenate(
        (split.process(iq[:317]), split.process(iq[317:4099]), split.process(iq[4099:]))
    )

    assert whole.design.tap_count > 256
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)


def test_butterworth_remains_available_as_an_explicit_shape() -> None:
    coefficients, design = design_iq_rbw_filter(
        1_000_000.0,
        100_000.0,
        shape="butterworth",
    )
    measurement_filter = StatefulIQMeasurementFilter(
        1_000_000.0,
        100_000.0,
        shape="butterworth",
    )

    assert coefficients.ndim == 2
    assert design.filter_shape == "butterworth"
    assert measurement_filter.sos is not None


def test_rms_detector_returns_mean_complex_power() -> None:
    iq = np.asarray([1.0 + 0.0j, 2.0 + 0.0j], dtype=np.complex64)
    assert reduce_filtered_iq_power(iq, "RMS") == pytest.approx(2.5)
    assert reduce_filtered_iq_power(iq, "Peak") == pytest.approx(4.0)
    assert reduce_filtered_iq_power(iq, "Sample") == pytest.approx(4.0)


def test_bucket_detector_covers_every_sample_once() -> None:
    iq = np.arange(1, 11, dtype=np.float64).astype(np.complex64)

    rms, centers = reduce_filtered_iq_power_buckets(iq, "RMS", max_points=4)
    peak, _ = reduce_filtered_iq_power_buckets(iq, "Peak", max_points=4)
    sample, _ = reduce_filtered_iq_power_buckets(iq, "Sample", max_points=4)

    np.testing.assert_allclose(rms, [2.5, 50.0 / 3.0, 85.0 / 2.0, 245.0 / 3.0])
    np.testing.assert_allclose(peak, [4.0, 25.0, 49.0, 100.0])
    np.testing.assert_allclose(sample, [4.0, 25.0, 49.0, 100.0])
    np.testing.assert_allclose(centers, [0.5, 3.0, 5.5, 8.0])
