import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from types import SimpleNamespace
import numpy as np
import pyqtgraph as pg
import pytest
from pyqtgraph.Qt import QtCore
from pluto_vsa.ui.main_window import (
    VSAWindow,
    _FixedInteractionViewBox,
    _constellation_density,
    _constellation_density_color_levels,
    _constellation_density_extent,
    _constellation_display_symbols,
    _frequency_constellation_density,
    _initial_result_time_range_ms,
    _decimation_indices_with_required_times,
    _fsk_phase_difference_symbols,
    _format_evm,
    _limit_iq_power_display_dbm,
    _peak_decimate_xy,
    _prepare_fsk_display_frequency,
    _prepare_psk_display_waveform,
)
from pluto_vsa.demod.gfsk import prepare_fsk_frequency
from pluto_vsa.model import ModulationKind
from pluto_vsa.pattern import prepare_psk_iq
from pluto_vsa.ui.measurement_chrome import (
    SymbolDensitySpread,
    make_measurement_plot,
    plot_complex_symbol_distribution,
    plot_frequency_symbol_distribution,
    set_iq_power_default_y_range,
    symbol_density_sigma_bins,
)


def test_peak_decimation_keeps_bucket_extrema() -> None:
    x = np.arange(1000, dtype=np.float64)
    y = np.sin(x / 13.0)
    y[123] = -20.0
    y[456] = 30.0

    plotted_x, plotted_y = _peak_decimate_xy(x, y, maximum=100)

    assert plotted_x.size <= 102
    assert -20.0 in plotted_y
    assert 30.0 in plotted_y


def test_peak_decimation_ignores_invalid_power_samples_for_extrema() -> None:
    x = np.arange(1_000, dtype=np.float64)
    y = np.linspace(-70.0, -20.0, x.size)
    y[::7] = np.nan

    _, plotted_y = _peak_decimate_xy(x, y, maximum=100)

    finite = plotted_y[np.isfinite(plotted_y)]
    assert finite.size > 0
    assert float(np.min(finite)) >= -70.0
    assert float(np.max(finite)) <= -20.0


def test_iq_power_display_floor_replaces_invalid_and_extreme_values() -> None:
    limited = _limit_iq_power_display_dbm(
        np.asarray([-np.inf, np.nan, -1_000.0, -119.0, -20.0])
    )

    np.testing.assert_allclose(limited, [-120.0, -120.0, -120.0, -119.0, -20.0])


def test_iq_power_default_range_is_50_db_below_peak_and_keeps_upper() -> None:
    pg.mkQApp("IQ power default Y range test")
    plot = make_measurement_plot("IQ Power (dBm)", "Time (ms)")
    try:
        values = np.asarray([-92.0, -18.0, -20.0], dtype=np.float64)
        applied = set_iq_power_default_y_range(
            plot, values, upper_dbm=-15.5
        )

        assert applied == pytest.approx((-68.0, -15.5))
        np.testing.assert_allclose(plot.viewRange()[1], [-68.0, -15.5])
    finally:
        plot.close()


def test_burst_search_initial_time_range_starts_before_trigger() -> None:
    pattern = SimpleNamespace(
        result_start_time_s=200e-6,
        result_stop_time_s=500e-6,
        recording_sample_rate_hz=8_000_000.0,
        metadata={
            "power_trigger_enabled": True,
            "power_trigger_sample": 800,
        },
    )

    start_ms, stop_ms = _initial_result_time_range_ms(pattern)

    assert start_ms == pytest.approx(0.07)
    assert stop_ms == pytest.approx(0.53)


def test_non_burst_initial_time_range_remains_result_centered() -> None:
    pattern = SimpleNamespace(
        result_start_time_s=200e-6,
        result_stop_time_s=500e-6,
        recording_sample_rate_hz=8_000_000.0,
        metadata={"power_trigger_enabled": False},
    )

    start_ms, stop_ms = _initial_result_time_range_ms(pattern)

    assert start_ms == pytest.approx(0.17)
    assert stop_ms == pytest.approx(0.53)


def test_frequency_constellation_density_is_vertical_and_count_weighted() -> None:
    values = np.asarray([-160.0] * 4 + [160.0] * 12)

    density = _frequency_constellation_density(values, limit_khz=240.0)

    assert density.shape == (96, 16)
    assert np.count_nonzero(density) > 0
    assert float(np.max(density[48:])) > float(np.max(density[:48]))


def test_peak_decimation_includes_required_symbol_coordinates() -> None:
    x = np.arange(1000, dtype=np.float64)
    y = np.sin(x / 13.0)
    required_x = np.asarray([123.25, 456.75, 789.5])

    plotted_x, plotted_y = _peak_decimate_xy(
        x,
        y,
        maximum=100,
        required_x_values=required_x,
    )

    for value in required_x:
        matches = np.flatnonzero(plotted_x == value)
        assert matches.size == 1
        assert plotted_y[matches[0]] == pytest.approx(np.interp(value, x, y))


def test_trajectory_decimation_brackets_every_required_symbol_time() -> None:
    time_s = np.arange(100, dtype=np.float64)
    required_time_s = np.asarray([12.25, 55.75])

    indices = _decimation_indices_with_required_times(
        time_s,
        required_time_s,
        maximum=10,
    )

    assert {12, 13, 55, 56}.issubset(set(indices))


def test_trace_symbol_plot_keeps_all_symbols_above_previous_limit() -> None:
    pg.mkQApp("VSA symbol point decimation test")
    plot = pg.PlotWidget()
    x = np.arange(2747, dtype=np.float64)
    y = np.sin(x)
    try:
        VSAWindow._plot_symbol_points(plot, x, y)
        plotted_x, plotted_y = plot.listDataItems()[-1].getData()
        assert np.asarray(plotted_x) == pytest.approx(x)
        assert np.asarray(plotted_y) == pytest.approx(y)
    finally:
        plot.close()


def test_evm_formatter_shows_percent_and_amplitude_ratio_db() -> None:
    assert _format_evm(5.0) == "5.00 % / -26.0 dB"
    assert _format_evm(100.0) == "100.00 % / 0.0 dB"
    assert _format_evm(0.0) == "0.00 % / -inf dB"
    assert _format_evm(float("nan")) == "—"


def test_psk_display_preparation_limits_work_to_result_range() -> None:
    sample_rate_hz = 8e6
    symbol_rate_hz = 1e6
    iq = np.exp(1j * np.arange(800_000, dtype=np.float64) * 0.01)

    prepared, time_s = _prepare_psk_display_waveform(
        iq,
        sample_rate_hz=sample_rate_hz,
        symbol_rate_hz=symbol_rate_hz,
        tx_filter="Root Raised Cosine",
        filter_parameter=0.4,
        result_start_time_s=0.050,
        result_stop_time_s=0.051,
    )

    assert prepared.size < 12_000
    assert time_s[0] < 0.050
    assert time_s[-1] > 0.051

    full, full_rate_hz = prepare_psk_iq(
        iq,
        sample_rate_hz=sample_rate_hz,
        symbol_rate_hz=symbol_rate_hz,
        tx_filter="Root Raised Cosine",
        filter_parameter=0.4,
    )
    visible = (time_s >= 0.050) & (time_s < 0.051)
    full_index = np.rint(time_s[visible] * full_rate_hz).astype(np.int64)
    assert prepared[visible] == pytest.approx(full[full_index], abs=1e-10)


def test_fsk_display_preparation_uses_demodulator_measurement_filter() -> None:
    sample_rate_hz = 8e6
    symbol_rate_hz = 1e6
    levels = np.repeat(np.tile([-160_000.0, 160_000.0], 200), 8)
    phase = np.cumsum(2.0 * np.pi * levels / sample_rate_hz)
    iq = np.exp(1j * phase)

    measured_hz, time_s = _prepare_fsk_display_frequency(
        iq,
        sample_rate_hz=sample_rate_hz,
        symbol_rate_hz=symbol_rate_hz,
        gaussian_bt=0.5,
        result_start_time_s=100e-6,
        result_stop_time_s=200e-6,
    )

    full_hz, full_rate_hz = prepare_fsk_frequency(
        iq,
        sample_rate_hz=sample_rate_hz,
        symbol_rate_hz=symbol_rate_hz,
        gaussian_bt=0.5,
    )
    visible = (time_s >= 100e-6) & (time_s < 200e-6)
    full_index = np.rint(time_s[visible] * full_rate_hz).astype(np.int64)
    assert measured_hz[visible] == pytest.approx(full_hz[full_index], abs=1e-8)
    assert time_s[0] < 100e-6
    assert time_s[-1] > 200e-6


def test_fixed_plot_interaction_uses_middle_drag_for_pan(monkeypatch) -> None:
    pg.mkQApp("VSA fixed mouse interaction test")
    observed_modes: list[int] = []

    def observe_drag(view_box, _event, axis=None) -> None:
        observed_modes.append(view_box.state["mouseMode"])

    monkeypatch.setattr(pg.ViewBox, "mouseDragEvent", observe_drag)
    view_box = _FixedInteractionViewBox()

    class Event:
        def __init__(self, button: QtCore.Qt.MouseButton) -> None:
            self._button = button

        def button(self) -> QtCore.Qt.MouseButton:
            return self._button

    view_box.mouseDragEvent(Event(QtCore.Qt.MouseButton.LeftButton))
    view_box.mouseDragEvent(Event(QtCore.Qt.MouseButton.MiddleButton))

    assert observed_modes == [pg.ViewBox.RectMode, pg.ViewBox.PanMode]
    assert view_box.state["mouseMode"] == pg.ViewBox.RectMode
    view_box.setMouseMode(pg.ViewBox.PanMode)
    assert view_box.state["mouseMode"] == pg.ViewBox.RectMode


def test_fsk_phase_difference_preserves_rms_normalized_symbol_amplitude() -> None:
    time_s = np.arange(32, dtype=np.float64) / 8_000_000.0
    symbol_time_s = np.asarray([0.5, 1.5, 2.5, 3.5]) / 1_000_000.0
    amplitude = np.asarray([0.5, 1.0, 1.5, 2.0])
    iq = np.interp(time_s, symbol_time_s, amplitude).astype(np.complex128)
    frequency_hz = np.asarray([-160_000.0, 160_000.0, -160_000.0, 160_000.0])

    symbols = _fsk_phase_difference_symbols(
        iq,
        time_s,
        symbol_time_s,
        frequency_hz,
        1_000_000.0,
    )

    expected_magnitude = amplitude / np.sqrt(np.mean(amplitude**2))
    np.testing.assert_allclose(np.abs(symbols), expected_magnitude, atol=1e-12)
    np.testing.assert_allclose(
        np.angle(symbols),
        2.0 * np.pi * frequency_hz / 1_000_000.0,
        atol=1e-12,
    )


def test_constellation_display_rotation_is_qpsk_family_only() -> None:
    diagonal = np.exp(1j * (np.pi / 4.0 + np.arange(4) * np.pi / 2.0))
    qpsk_display = _constellation_display_symbols(ModulationKind.QPSK, diagonal)
    pi4_display = _constellation_display_symbols(
        ModulationKind.PI4_DQPSK, diagonal
    )
    d8psk = np.exp(1j * np.arange(8) * np.pi / 4.0)

    np.testing.assert_allclose(qpsk_display, [1.0, 1j, -1.0, -1j], atol=1e-12)
    np.testing.assert_allclose(pi4_display, qpsk_display, atol=1e-12)
    np.testing.assert_allclose(
        _constellation_display_symbols(ModulationKind.DPSK8, d8psk),
        d8psk,
        atol=1e-12,
    )


def test_constellation_density_encodes_occurrence_count() -> None:
    symbols = np.asarray([-0.75 + 0.0j] * 8 + [0.75 + 0.0j])

    density = _constellation_density(symbols, bins=40)

    assert density.shape == (40, 40)
    assert np.count_nonzero(density) > 2
    assert np.all(np.isfinite(density))
    # Gaussian spreading must preserve the stronger occurrence cluster.
    assert float(np.max(density[:, :20])) > float(np.max(density[:, 20:]))


def test_constellation_density_can_disable_smoothing() -> None:
    symbols = np.asarray([0.0 + 0.0j] * 8 + [1.0 + 0.0j])

    density = _constellation_density(
        symbols, bins=20, smoothing_sigma_bins=0.0
    )

    nonzero = density[density > 0.0]
    assert nonzero.size == 2
    assert float(np.max(nonzero)) == pytest.approx(np.log1p(8.0))
    assert float(np.min(nonzero)) == pytest.approx(np.log1p(1.0))


def test_symbol_density_spread_has_three_shared_kernel_widths() -> None:
    assert symbol_density_sigma_bins(SymbolDensitySpread.NONE) == 0.0
    assert 0.0 < symbol_density_sigma_bins(SymbolDensitySpread.MEDIUM) < 0.7
    assert symbol_density_sigma_bins(SymbolDensitySpread.MAXIMUM) == 0.7


def test_symbol_density_spread_applies_to_complex_and_frequency_plots() -> None:
    pg.mkQApp("VSA shared density spread test")
    complex_plot = pg.PlotWidget()
    frequency_plot = pg.PlotWidget()
    symbols = np.asarray([0.25 + 0.25j] * 8)
    frequencies = np.asarray([100.0] * 8)
    complex_counts = []
    frequency_counts = []
    try:
        for spread in SymbolDensitySpread:
            complex_plot.clear()
            frequency_plot.clear()
            complex_item = plot_complex_symbol_distribution(
                complex_plot,
                symbols,
                density=True,
                density_spread=spread,
            )
            frequency_item = plot_frequency_symbol_distribution(
                frequency_plot,
                frequencies,
                y_limit_khz=200.0,
                density=True,
                density_spread=spread,
            )
            assert complex_item is not None
            assert frequency_item is not None
            complex_counts.append(np.count_nonzero(complex_item.image))
            frequency_counts.append(np.count_nonzero(frequency_item.image))
    finally:
        complex_plot.close()
        frequency_plot.close()

    assert complex_counts[0] < complex_counts[1] < complex_counts[2]
    assert frequency_counts[0] < frequency_counts[1] < frequency_counts[2]


def test_constellation_density_extent_includes_symbols_outside_nominal_plane() -> None:
    symbols = np.asarray([-1.6 + 0.2j, 0.1 + 1.4j, np.nan + 0.0j])

    limit = _constellation_density_extent(symbols)
    density = _constellation_density(symbols, limit=limit, bins=40)

    assert limit > 1.6
    assert np.count_nonzero(density) > 0


def test_constellation_density_saturates_high_density_region_to_red() -> None:
    density = np.asarray([[0.0, 1.0], [2.0, 4.0]])

    levels = _constellation_density_color_levels(density)

    assert levels == pytest.approx((0.0, 3.0))
