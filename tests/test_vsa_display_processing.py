import numpy as np

from pluto_sa.vsa.ui.display_processing import (
    build_fsk_display_data,
    fit_binary_fsk_display_drift,
    sample_fsk_display_trace,
)


def test_fsk_display_symbols_are_interpolated_from_the_displayed_trace() -> None:
    time_s = np.array([0.0, 1.0, 2.0, 3.0]) * 1e-6
    frequency_hz = np.array([-100_000.0, 100_000.0, 200_000.0, -200_000.0])
    symbol_time_s = np.array([0.5, 1.5, 2.25]) * 1e-6

    sampled = sample_fsk_display_trace(
        frequency_hz, time_s, symbol_time_s
    )

    np.testing.assert_allclose(sampled, (0.0, 150_000.0, 100_000.0))


def test_fsk_display_data_removes_frequency_offset_and_drift_once() -> None:
    time_s = np.linspace(0.0, 1e-3, 101)
    offset_hz = 31_000.0
    drift_hz_per_s = 8_000_000.0
    modulation_hz = 150_000.0 * np.sin(2.0 * np.pi * time_s / 100e-6)
    frequency_hz = modulation_hz + offset_hz + drift_hz_per_s * time_s
    symbol_time_s = np.array([0.1e-3, 0.4e-3, 0.9e-3])

    display = build_fsk_display_data(
        frequency_hz,
        time_s,
        symbol_time_s,
        frequency_offset_hz=offset_hz,
        frequency_drift_hz_per_s=drift_hz_per_s,
    )

    np.testing.assert_allclose(
        display.corrected_frequency_hz, modulation_hz, atol=1e-9
    )
    np.testing.assert_allclose(
        display.symbol_frequency_hz,
        np.interp(symbol_time_s, time_s, modulation_hz),
        atol=1e-9,
    )


def test_binary_fsk_display_drift_fit_separates_levels_from_drift() -> None:
    time_s = np.arange(100, dtype=np.float64) * 1e-6
    bits = np.tile((0, 1, 1, 0), 25)
    drift_hz_per_s = 12_000_000.0
    reference_time_s = float(np.mean(time_s))
    frequency_hz = (
        17_000.0
        + (2.0 * bits - 1.0) * 160_000.0
        + drift_hz_per_s * (time_s - reference_time_s)
    )

    fitted_drift, fitted_reference = fit_binary_fsk_display_drift(
        time_s, frequency_hz, bits
    )

    np.testing.assert_allclose(fitted_drift, drift_hz_per_s, rtol=1e-12)
    assert fitted_reference == reference_time_s
