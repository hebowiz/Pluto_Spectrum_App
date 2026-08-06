import numpy as np
import pytest

from pluto_sa.vsa.demod.fsk_reference import (
    apply_gaussian_frequency_filter,
    fsk_reference_frequency_levels,
)


def test_gaussian_bt_filter_has_analytic_minus_3_db_bandwidth() -> None:
    samples_per_symbol = 64
    bt = 0.5
    impulse = np.zeros(4097, dtype=np.float64)
    impulse[impulse.size // 2] = 1.0
    filtered = apply_gaussian_frequency_filter(
        impulse,
        samples_per_symbol=samples_per_symbol,
        bt=bt,
    )
    response = np.abs(np.fft.rfft(np.fft.ifftshift(filtered)))
    frequency = np.fft.rfftfreq(filtered.size)
    index = int(np.argmin(np.abs(frequency - bt / samples_per_symbol)))

    assert response[index] / response[0] == pytest.approx(
        1.0 / np.sqrt(2.0), abs=0.015
    )


def test_fsk_reference_applies_transmit_then_measurement_filter() -> None:
    symbols = np.asarray([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
    transmit_only = fsk_reference_frequency_levels(
        symbols,
        samples_per_symbol=8,
        transmit_gaussian_bt=0.5,
    )
    expected = apply_gaussian_frequency_filter(
        transmit_only,
        samples_per_symbol=8,
        bt=0.5,
    )
    combined = fsk_reference_frequency_levels(
        symbols,
        samples_per_symbol=8,
        transmit_gaussian_bt=0.5,
        measurement_gaussian_bt=0.5,
    )

    np.testing.assert_allclose(combined, expected, rtol=0.0, atol=1e-12)

