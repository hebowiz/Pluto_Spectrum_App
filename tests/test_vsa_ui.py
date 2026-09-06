import os
import json
import threading
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyqtgraph as pg
import pytest
from pyqtgraph.Qt import QtCore, QtWidgets

from pluto_sa.vsa.ui.main_window import (
    VSAWindow,
    _FixedInteractionViewBox,
    _constellation_density,
    _constellation_density_color_levels,
    _constellation_density_extent,
    _constellation_display_symbols,
    _frequency_constellation_density,
    _initial_result_time_range_ms,
    _physical_constellation_display_symbols,
    _decimation_indices_with_required_times,
    _fsk_phase_difference_symbols,
    _format_evm,
    _limit_iq_power_display_dbm,
    _peak_decimate_xy,
    _prepare_fsk_display_frequency,
    _prepare_psk_display_waveform,
)
from pluto_sa.vsa.demod.gfsk import prepare_fsk_frequency
from pluto_sa.vsa.pluto_source import CaptureCancelledError
from pluto_sa.vsa.model import IQRecording, ModulationKind, SignalDescription
from pluto_sa.vsa.mapping import (
    BLUETOOTH_HDT_MAPPING,
    psk_constellation,
    reverse_symbol_bits,
)
from pluto_sa.vsa.pattern import (
    BitOrdering,
    DemodulationSettings,
    KnownPattern,
    MeasurementFilterMode,
    PatternSearchSettings,
    ResultRangeSettings,
    SynchronizationSource,
    prepare_psk_iq,
)
from pluto_sa.vsa.session import VSASession
from pluto_sa.vsa.sources import FileIQSource, GeneratedIQSource
from pluto_sa.vsa.ui.measurement_chrome import (
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


def test_large_symbol_result_limits_table_display_but_not_export(tmp_path) -> None:
    pg.mkQApp("VSA bounded symbol table test")
    recording, signal = GeneratedIQSource.psk(
        modulation=ModulationKind.PI4_DQPSK,
        symbol_count=1500,
        seed=91,
    )
    session = VSASession()
    session.set_recording(recording)
    session.set_signal(signal)
    # This test exercises bounded rendering of the unsynchronized/base result,
    # rather than the detected-data Result Range.
    session.configure_pattern_analysis(
        None,
        demodulation=DemodulationSettings(
            coarse_synchronization=SynchronizationSource.PATTERN,
        ),
    )
    session.analyze()
    window = VSAWindow(
        preferences=_isolated_preferences(tmp_path, "bounded-symbol-table")
    )
    try:
        window.session = session
        window._update_summary()
        window._update_plots(reset_ranges=True)

        assert window.symbol_plot_dock.windowTitle() == "Symbol Plot (Physical)"
        window.differential_iq_symbol_plot_action.trigger()
        assert window.symbol_plot_dock.windowTitle() == "Symbol Plot (Differential)"
        assert window.symbol_table.rowCount() == 100
        result_symbol_count = session.result.decoded_symbols.size
        assert f"Showing 1000 of {result_symbol_count}" in window.symbol_table.toolTip()
        assert (
            len(window._symbol_table_export_document()["rows"])
            == result_symbol_count
        )
    finally:
        window.close()


def _isolated_preferences(tmp_path: Path, name: str) -> QtCore.QSettings:
    return QtCore.QSettings(
        str(tmp_path / f"{name}.ini"), QtCore.QSettings.Format.IniFormat
    )


def _wait_for_background_analysis(window: VSAWindow) -> None:
    for _index in range(500):
        QtWidgets.QApplication.processEvents()
        thread = window._analysis_thread
        if thread is None and window._pending_analysis is None:
            QtWidgets.QApplication.processEvents()
            return
        if thread is not None:
            thread.wait(10)
    raise AssertionError("background VSA analysis did not finish")


def test_iq_power_signal_switch_selects_raw_or_measured_trace(tmp_path) -> None:
    pg.mkQApp("VSA IQ power signal selection test")
    recording, signal = GeneratedIQSource.fsk(symbol_count=96, seed=442)
    session = VSASession()
    session.set_recording(recording)
    session.set_signal(signal)
    session.analyze()
    session.capture_power_dbm = np.full(
        session.capture_time_s.shape, -11.0, dtype=np.float64
    )
    session.capture_power_dbm[0] = -1_000.0
    measured_power_dbm = np.full(
        session.result.time_s.shape, -22.0, dtype=np.float64
    )
    measured_power_dbm[0] = np.nan
    session.result = replace(
        session.result,
        power_dbm=measured_power_dbm,
    )
    window = VSAWindow(
        preferences=_isolated_preferences(tmp_path, "iq-power-signal")
    )
    try:
        window.session = session
        window._update_summary()
        window._update_plots(reset_ranges=True)

        assert window.raw_iq_power_action.isChecked()
        _, raw_y = window.zero_span_plot.listDataItems()[0].getData()
        assert np.all(np.isfinite(raw_y))
        assert float(np.min(raw_y)) == pytest.approx(-120.0)
        np.testing.assert_allclose(raw_y[1:], -11.0)

        window.measured_iq_power_action.setChecked(True)
        window._refresh_display_only()
        _, measured_y = window.zero_span_plot.listDataItems()[0].getData()
        assert np.all(np.isfinite(measured_y))
        assert float(np.min(measured_y)) == pytest.approx(-120.0)
        np.testing.assert_allclose(measured_y[1:], -22.0)

        window.zero_span_plot.setYRange(-1_000.0, -900.0, padding=0.0)
        assert window.zero_span_plot.viewRange()[1][0] >= -120.0
    finally:
        window.close()
        window.deleteLater()
        QtWidgets.QApplication.processEvents()


def test_result_range_arrow_actions_select_adjacent_packet(tmp_path) -> None:
    pg.mkQApp("VSA result navigation test")
    window = VSAWindow(
        preferences=_isolated_preferences(tmp_path, "result-navigation")
    )
    try:
        assert not hasattr(window, "pattern_match_selection_combo")
        assert not hasattr(window, "pattern_match_index_spin")
        recording, signal = GeneratedIQSource.psk(
            modulation=ModulationKind.PI4_DQPSK,
            symbol_count=160,
            seed=123,
        )
        generated = np.asarray(recording.metadata["generated_symbols"])
        gap = np.zeros(64, dtype=np.complex64)
        combined = IQRecording(
            iq=np.concatenate((recording.iq, gap, recording.iq)),
            sample_rate_hz=recording.sample_rate_hz,
        )
        window.load_recording(combined, signal)
        _wait_for_background_analysis(window)
        window.pattern_search_check.setChecked(True)
        window._set_pattern_symbols(
            reverse_symbol_bits(generated[20:36], signal.modulation.order)
        )
        window.result_length_spin.setValue(100)

        assert window._selected_match_index == 1
        assert window._analyze()
        window._reset_all_packet_statistics()
        assert window._request_analysis(
            analysis_context={
                "continuous": True,
                "collect_all_packets": True,
            }
        )
        _wait_for_background_analysis(window)
        assert window._all_packet_statistics.packet_count == 2
        assert window._selected_match_index == 1
        assert window.session.pattern_result.pattern_start_sample == 20 * 8
        summary_labels = {
            window.result_summary.item(row, 0).text()
            for row in range(window.result_summary.rowCount())
        }
        assert {
            "EVM RMS",
            "Differential Symbol EVM RMS",
            "Bluetooth DEVM RMS",
            "Symbol Rate Error",
        }.issubset(summary_labels)
        assert "FSK Deviation Error" not in summary_labels
        pattern_result = window.session.pattern_result
        reference = np.exp(
            1j * (np.pi / 4.0 + pattern_result.decoded_symbols * np.pi / 2.0)
        )
        plotted_symbol_evm = 100.0 * np.sqrt(
            np.sum(np.abs(pattern_result.measured_symbols - reference) ** 2)
            / np.sum(np.abs(reference) ** 2)
        )
        summary_values = {
            window.result_summary.item(row, 0).text(): window.result_summary.item(
                row, 1
            ).text()
            for row in range(window.result_summary.rowCount())
        }
        assert float(
            summary_values["Differential Symbol EVM RMS"].split()[0]
        ) == pytest.approx(
            plotted_symbol_evm, abs=0.005
        )
        assert float(summary_values["EVM RMS"].split()[0]) == pytest.approx(
            float(pattern_result.metadata["physical_evm_rms_percent"]), abs=0.005
        )
        assert summary_values["Bluetooth DEVM RMS"] == "—"
        first_start = window.session.pattern_result.pattern_start_sample
        assert first_start == 20 * 8
        assert not window.previous_result_action.isEnabled()
        assert window.next_result_action.isEnabled()
        assert window.previous_result_action.shortcut().toString() == "Left"
        assert window.next_result_action.shortcut().toString() == "Right"

        window.next_result_action.trigger()
        _wait_for_background_analysis(window)
        second = window.session.pattern_result
        assert second.pattern_start_sample == recording.sample_count + 64 + 20 * 8
        assert window._selected_match_index == 2
        assert window.previous_result_action.isEnabled()
        assert not window.next_result_action.isEnabled()

        # Refreshing analysis of the same IQ keeps the selected packet.
        assert window._analyze()
        assert window._selected_match_index == 2
        assert (
            window.session.pattern_result.pattern_start_sample
            == recording.sample_count + 64 + 20 * 8
        )

        window.previous_result_action.trigger()
        _wait_for_background_analysis(window)
        assert window.session.pattern_result.pattern_start_sample == first_start
        assert window._selected_match_index == 1

        window.next_result_action.trigger()
        _wait_for_background_analysis(window)
        assert window._selected_match_index == 2
        # Loading new IQ always returns focus to the first eligible packet.
        window.load_recording(combined, signal)
        _wait_for_background_analysis(window)
        assert window._selected_match_index == 1
        assert window.session.pattern_result.pattern_start_sample == first_start
    finally:
        window._meas_config_dialog.close()
        window.close()
        window.deleteLater()
        QtWidgets.QApplication.processEvents()


def test_pattern_result_uses_table_and_fitted_plot_ranges(tmp_path) -> None:
    pg.mkQApp("VSA UI test")
    window = VSAWindow(
        preferences=_isolated_preferences(tmp_path, "pattern-result")
    )
    try:
        window._load_generated(ModulationKind.GFSK)
        _wait_for_background_analysis(window)
        expected = np.asarray(window.session.recording.metadata["generated_symbols"])
        window.pattern_search_check.setChecked(True)
        window.pattern_symbols_edit.setText(
            "".join(str(int(value)) for value in expected[20:52])
        )
        window.result_length_spin.setValue(64)

        window._analyze()

        assert isinstance(window.symbol_table, QtWidgets.QTableWidget)
        assert not window.symbol_table.alternatingRowColors()
        assert not window.result_summary.alternatingRowColors()
        assert window.symbol_table.columnCount() == 10
        assert window.symbol_table.rowCount() == 7
        assert window.symbol_table.item(0, 0).text() == str(int(expected[20]))
        assert window.symbol_table.item(0, 0).textAlignment() == int(
            QtCore.Qt.AlignmentFlag.AlignCenter
        )
        green_cells = [
            window.symbol_table.item(index // 10, index % 10)
            for index in range(window.session.pattern_result.decoded_symbols.size)
            if window.symbol_table.item(index // 10, index % 10).background().color().green() > 80
        ]
        assert len(green_cells) == 32
        configured_pattern = list(window._parse_pattern_symbols(2))
        configured_pattern[7] = 1 - configured_pattern[7]
        window._set_pattern_symbols(configured_pattern)
        window._update_plots(reset_ranges=False)
        green_cells = [
            window.symbol_table.item(index // 10, index % 10)
            for index in range(window.session.pattern_result.decoded_symbols.size)
            if window.symbol_table.item(index // 10, index % 10).background().color().green() > 80
        ]
        assert len(green_cells) == 31
        assert window.modulation_plot.viewRange()[1] == pytest.approx([-375.0, 375.0])
        assert window.zero_span_plot.viewRange()[0] == pytest.approx(
            window.modulation_plot.viewRange()[0]
        )
        result = window.session.pattern_result
        duration_ms = (result.result_stop_time_s - result.result_start_time_s) * 1e3
        expected_x = [
            result.result_start_time_s * 1e3 - 0.1 * duration_ms,
            result.result_stop_time_s * 1e3 + 0.1 * duration_ms,
        ]
        assert window.zero_span_plot.viewRange()[0] == pytest.approx(expected_x)
        assert not window.symbol_display_action.isChecked()
        assert window.symbol_display_action.shortcut().toString() == "S"
        assert len(window.zero_span_plot.listDataItems()) == 1
        assert len(window.modulation_plot.listDataItems()) == 1
        for plot in (
            window.zero_span_plot,
            window.spectrum_plot,
            window.modulation_plot,
        ):
            trace_color = plot.listDataItems()[0].opts["pen"].color()
            assert trace_color.getRgb()[:3] == (255, 255, 0)
        _spectrum_x, spectrum_y = window.spectrum_plot.listDataItems()[0].getData()
        np.testing.assert_allclose(
            spectrum_y,
            window.session.pattern_range_result.spectrum_dbm,
        )
        window.symbol_display_action.trigger()
        assert window.symbol_display_action.isChecked()
        assert len(window.zero_span_plot.listDataItems()) == 2
        assert len(window.modulation_plot.listDataItems()) == 2
        power_marker = window.zero_span_plot.listDataItems()[1]
        marker_time_ms, marker_power_dbm = power_marker.getData()
        assert marker_time_ms.size == marker_power_dbm.size == 64
        marker_color = power_marker.opts["symbolBrush"].color()
        assert marker_color.green() > marker_color.red()
        assert marker_color.green() > marker_color.blue()
        assert power_marker.opts["symbolSize"] == pytest.approx(5.5)
        assert not hasattr(window, "pan_action")
        assert not hasattr(window, "rect_zoom_action")
        assert "Mouse Interaction" not in {
            action.text() for action in window._display_menu.actions()
        }
        assert all(
            isinstance(plot.getViewBox(), _FixedInteractionViewBox)
            and plot.getViewBox().state["mouseMode"] == pg.ViewBox.RectMode
            for _name, plot in window._plot_widgets()
        )
        for name, plot in window._plot_widgets():
            menu = plot.getViewBox().getMenu(None)
            menu_labels = [action.text() for action in menu.actions()]
            assert menu_labels[:3] == [
                "Reset",
                "",
                "View All",
            ]
            assert {"X axis", "Y axis"}.issubset(menu_labels)
            assert "Mouse Mode" not in menu_labels
            assert window._plot_context_actions[name]["view_all"] is menu.viewAll
        initial_ranges = {
            name: (list(ranges[0]), list(ranges[1]))
            for name, ranges in window._analysis_plot_ranges.items()
        }
        _, initial_power = window.zero_span_plot.listDataItems()[0].getData()
        assert initial_ranges["iq_power"][1][0] == pytest.approx(
            float(np.max(initial_power)) - 50.0
        )
        for _name, plot in window._plot_widgets():
            plot.setRange(xRange=(-99.0, -98.0), yRange=(-77.0, -76.0))
        window.reset_graph_scales_action.trigger()
        for name, plot in window._plot_widgets():
            assert plot.viewRange()[0] == pytest.approx(initial_ranges[name][0])
            assert plot.viewRange()[1] == pytest.approx(initial_ranges[name][1])
        window.zero_span_plot.setRange(
            xRange=(-99.0, -98.0), yRange=(-77.0, -76.0), padding=0.0
        )
        window._plot_context_actions["iq_power"]["reset"].trigger()
        assert window.zero_span_plot.viewRange()[0] == pytest.approx(
            initial_ranges["iq_power"][0]
        )
        assert window.zero_span_plot.viewRange()[1] == pytest.approx(
            initial_ranges["iq_power"][1]
        )

        # View All must fit every finite trace point without allowing a distant
        # overlay line to inflate the range.
        far_overlay = pg.InfiniteLine(pos=1e9, angle=90)
        window.zero_span_plot.addItem(far_overlay)
        trace_x, trace_y = window.zero_span_plot.listDataItems()[0].getData()
        window.zero_span_plot.setRange(
            xRange=(-99.0, -98.0), yRange=(-77.0, -76.0), padding=0.0
        )
        window._plot_context_actions["iq_power"]["view_all"].trigger()
        view_x, view_y = window.zero_span_plot.viewRange()
        assert view_x[0] <= float(np.min(trace_x))
        assert view_x[1] >= float(np.max(trace_x))
        assert view_y[0] <= float(np.min(trace_y))
        assert view_y[1] >= float(np.max(trace_y))
        assert view_x[1] < 1e6

        symbol_x, symbol_y = (
            window.symbol_plot.listDataItems()[0].getOriginalDataset()
        )
        window.symbol_plot.setRange(
            xRange=(-0.1, 0.1), yRange=(-0.1, 0.1), padding=0.0
        )
        window._plot_context_actions["symbol_plot"]["view_all"].trigger()
        symbol_view_x, symbol_view_y = window.symbol_plot.viewRange()
        assert symbol_view_x[0] <= float(np.min(symbol_x))
        assert symbol_view_x[1] >= float(np.max(symbol_x))
        assert symbol_view_y[0] <= float(np.min(symbol_y))
        assert symbol_view_y[1] >= float(np.max(symbol_y))
        assert window.symbol_plot.getViewBox().state["aspectLocked"] == 1.0
        assert window._meas_config_dialog.isModal()
        assert window._meas_config_dialog.windowModality() != (
            QtCore.Qt.WindowModality.NonModal
        )
        assert window._config_stack.currentIndex() == 0
        assert set(window._config_top_buttons) == {
            "Input / Frontend",
            "Signal Description",
            "Signal Capture",
            "Trigger",
            "Pattern Search",
            "Result Range",
            "Demodulation",
            "Result Summary",
            "Sweep / Run",
        }
        assert all(
            button.font().pointSizeF() >= 18.0
            and button.minimumHeight() >= 84
            for button in window._config_top_buttons.values()
        )
        assert window._config_top_title.font().pointSizeF() >= 16.0
        assert not hasattr(window, "_config_load_button")
        assert not hasattr(window, "_config_save_button")
        window._config_top_buttons["Signal Description"].click()
        assert window._config_stack.currentIndex() == 2
        assert window._config_back_button.isVisibleTo(window._meas_config_dialog)
        window._config_back_button.click()
        assert window._config_stack.currentIndex() == 0
        active_modal_widgets = []

        def inspect_modality() -> None:
            active_modal_widgets.append(QtWidgets.QApplication.activeModalWidget())
            window._meas_config_dialog.reject()

        window.show()
        QtWidgets.QApplication.processEvents()
        window._equalize_result_docks()
        QtWidgets.QApplication.processEvents()
        for _name, plot in window._plot_widgets():
            vertical_axis = plot.getAxis("left")
            vertical_label_bounds = vertical_axis.label.mapRectToParent(
                vertical_axis.label.boundingRect()
            )
            assert vertical_label_bounds.center().y() == pytest.approx(
                vertical_axis.size().height() / 2.0, abs=1.0
            )
            horizontal_axis = plot.getAxis("bottom")
            horizontal_label_bounds = horizontal_axis.label.mapRectToParent(
                horizontal_axis.label.boundingRect()
            )
            assert horizontal_label_bounds.center().x() == pytest.approx(
                horizontal_axis.size().width() / 2.0, abs=1.0
            )
        docks = (
            window.zero_span_dock,
            window.spectrum_dock,
            window.result_summary_dock,
            window.modulation_dock,
            window.reserved_dock,
            window.symbol_dock,
        )
        assert all(isinstance(dock, QtWidgets.QDockWidget) for dock in docks)
        assert all(dock.font().bold() for dock in docks)
        assert all(
            dock.font().pointSizeF() >= window.result_summary.font().pointSizeF() * 1.25
            for dock in docks
        )
        assert not window.result_summary.font().bold()
        assert not window.symbol_table.font().bold()
        assert all(
            not plot.getPlotItem().titleLabel.text
            for _name, plot in window._plot_widgets()
        )
        assert window.spectrum_plot.getAxis("left").labelText == "Magnitude (dBm)"
        assert window.symbol_plot_dock.windowTitle() == (
            "Symbol Plot (Phase Difference)"
        )
        phase_items = window.symbol_plot.listDataItems()
        assert len(phase_items) == 2
        assert phase_items[0].opts["symbolPen"].color().getRgb()[:3] == (
            255,
            255,
            0,
        )
        phase_i, phase_q = phase_items[0].getData()
        assert phase_i.size == phase_q.size
        assert phase_i.size == window.session.pattern_result.measured_symbols.size
        phase_magnitude = np.hypot(phase_i, phase_q)
        assert np.sqrt(np.mean(phase_magnitude**2)) == pytest.approx(
            1.0, abs=1e-6
        )
        decoded = window.session.pattern_result.decoded_symbols
        assert np.mean(phase_q[decoded == 1]) > 0.25
        assert np.mean(phase_q[decoded == 0]) < -0.25

        window.fsk_constellation_frequency_action.trigger()
        assert window.symbol_plot_dock.windowTitle() == (
            "Symbol Plot (Constellation Frequency)"
        )
        assert not window.symbol_plot.getAxis("bottom").isVisible()
        assert window.symbol_plot.getAxis("left").labelText == "Frequency (kHz)"
        assert window.symbol_plot.getViewBox().state["aspectLocked"] is False
        frequency_view_box = window.symbol_plot.getViewBox()
        assert frequency_view_box.state["mouseEnabled"] == [False, True]
        assert frequency_view_box.state["limits"]["xLimits"] == [-1.0, 1.0]
        assert frequency_view_box.state["limits"]["xRange"] == [2.0, 2.0]
        window.symbol_plot.setXRange(-0.25, 0.25, padding=0.0)
        assert window.symbol_plot.viewRange()[0] == pytest.approx([-1.0, 1.0])
        assert window.symbol_plot.viewRange()[1] == pytest.approx(
            window.modulation_plot.viewRange()[1]
        )
        frequency_items = window.symbol_plot.listDataItems()
        assert len(frequency_items) == 1
        frequency_x, frequency_y = frequency_items[0].getData()
        _modulation_symbol_x, modulation_symbol_y = (
            window.modulation_plot.listDataItems()[1].getData()
        )
        np.testing.assert_allclose(frequency_x, 0.0)
        np.testing.assert_array_equal(frequency_y, modulation_symbol_y)
        window.constellation_density_action.trigger()
        assert window._constellation_density_item is not None
        assert window._constellation_density_item.image.shape == (96, 16)
        window.constellation_flat_action.trigger()
        window.fsk_phase_difference_action.trigger()
        assert window.symbol_plot.getAxis("bottom").isVisible()
        assert window.symbol_plot.getViewBox().state["mouseEnabled"] == [True, True]
        assert window.symbol_plot.getViewBox().state["limits"]["xLimits"] == [
            None,
            None,
        ]
        assert window.symbol_plot.viewRange()[1] == pytest.approx([-1.25, 1.25])
        assert window.centralWidget() is None
        assert not (
            window.dockOptions()
            & QtWidgets.QMainWindow.DockOption.AnimatedDocks
        )
        for plot in (
            window.zero_span_plot,
            window.spectrum_plot,
            window.modulation_plot,
        ):
            options = plot.getPlotItem().ctrl
            assert options.downsampleCheck.isChecked()
            assert options.autoDownsampleCheck.isChecked()
            assert options.peakRadio.isChecked()
            assert options.clipToViewCheck.isChecked()
        assert max(dock.width() for dock in docks) - min(
            dock.width() for dock in docks
        ) <= 2
        assert max(dock.height() for dock in docks) - min(
            dock.height() for dock in docks
        ) <= 2
        QtCore.QTimer.singleShot(0, inspect_modality)
        window._open_meas_config()
        assert active_modal_widgets == [window._meas_config_dialog]
        assert window.measured_modulation_signal_action.isChecked()
        assert not hasattr(window, "raw_carrier_action")
        assert not hasattr(window, "corrected_carrier_action")
        default_summary = {
            window.result_summary.item(row, 0).text(): window.result_summary.item(
                row, 1
            ).text()
            for row in range(window.result_summary.rowCount())
        }
        assert "Power" in default_summary
        assert "Carrier Frequency Error" in default_summary
        assert default_summary["FSK Deviation Error"].endswith("Hz")
        assert default_summary["Carrier Frequency Drift"].endswith("Hz/Sym")
        assert "Frequency Fit RMS" not in default_summary

        context_menu = window._create_result_summary_context_menu()
        category_labels = {
            action.text() for action in context_menu.actions() if action.menu() is not None
        }
        assert category_labels == {
            "Common Measurement Results",
            "PSK Measurement Results",
            "FSK Measurement Results",
            "Synchronization Diagnostics",
        }
        psk_menu = next(
            submenu
            for submenu in context_menu.findChildren(QtWidgets.QMenu)
            if submenu.title() == "PSK Measurement Results"
        )
        evm_peak_action = next(
            action for action in psk_menu.actions() if action.text().startswith("EVM Peak")
        )
        assert not evm_peak_action.isEnabled()
        assert "Not implemented" in evm_peak_action.text()

        window._apply_result_summary_preset("all")
        context_menu = window._create_result_summary_context_menu()
        diagnostics_menu = next(
            submenu
            for submenu in context_menu.findChildren(QtWidgets.QMenu)
            if submenu.title() == "Synchronization Diagnostics"
        )
        frequency_fit_action = next(
            action
            for action in diagnostics_menu.actions()
            if action.text() == "Frequency Fit RMS"
        )
        assert frequency_fit_action.isChecked()
        frequency_fit_action.trigger()
        assert "frequency_fit_rms" not in window._selected_result_summary_ids
        assert window._result_summary_tree_items[
            "frequency_fit_rms"
        ].checkState(0) == QtCore.Qt.CheckState.Unchecked
        frequency_fit_action.trigger()
        assert "frequency_fit_rms" in window._selected_result_summary_ids
        summary = {
            window.result_summary.item(row, 0).text(): window.result_summary.item(
                row, 1
            ).text()
            for row in range(window.result_summary.rowCount())
        }
        assert "Carrier Frequency Error" in summary
        assert "Fractional Timing" in summary
        assert "Timing Confidence" in summary
        assert "Deviation Error (%)" in summary
        assert summary["Drift Model"].startswith(("Accepted", "Rejected"))
        assert "Applied Drift" in summary
        assert summary["Display"] == "Measured"

        window.raw_modulation_signal_action.trigger()

        summary = {
            window.result_summary.item(row, 0).text(): window.result_summary.item(
                row, 1
            ).text()
            for row in range(window.result_summary.rowCount())
        }
        assert summary["Display"] == "Raw IQ"
    finally:
        window._meas_config_dialog.close()
        window.close()
        window.deleteLater()
        QtWidgets.QApplication.processEvents()


def test_symbol_table_defaults_to_hex_and_switches_to_decimal(tmp_path) -> None:
    pg.mkQApp("VSA Symbol Table format test")
    window = VSAWindow(
        preferences=_isolated_preferences(tmp_path, "symbol-table-format")
    )
    try:
        window._load_generated(ModulationKind.QAM16)
        _wait_for_background_analysis(window)
        document = window._symbol_table_export_document()
        displayed_values = [int(row[1]) for row in document["rows"]]

        assert window.symbol_table_hex_action.isChecked()
        assert any(value >= 10 for value in displayed_values)
        hex_text = [
            window.symbol_table.item(index // 10, index % 10).text()
            for index in range(len(displayed_values))
        ]
        assert hex_text == [format(value, "X") for value in displayed_values]

        window.symbol_table_decimal_action.trigger()
        decimal_text = [
            window.symbol_table.item(index // 10, index % 10).text()
            for index in range(len(displayed_values))
        ]
        assert decimal_text == [str(value) for value in displayed_values]
        assert window._symbol_table_export_document()["rows"] == document["rows"]
        assert (
            window._meas_config_values()["display_config"]["symbol_table_format"]
            == "Decimal"
        )
    finally:
        window._meas_config_dialog.close()
        window.close()
        window.deleteLater()
        QtWidgets.QApplication.processEvents()


def test_symbol_table_click_places_and_toggles_fsk_plot_markers(tmp_path) -> None:
    pg.mkQApp("VSA FSK symbol marker test")
    window = VSAWindow(
        preferences=_isolated_preferences(tmp_path, "fsk-symbol-marker")
    )
    try:
        window._load_generated(ModulationKind.GFSK)
        _wait_for_background_analysis(window)
        expected = np.asarray(window.session.recording.metadata["generated_symbols"])
        window.pattern_search_check.setChecked(True)
        window._set_pattern_symbols(expected[20:52])
        window.result_length_spin.setValue(64)
        assert window._analyze()
        assert window._last_analysis_timings_ms["total_dsp"] >= 0.0
        assert window._last_analysis_timings_ms["display"] >= 0.0
        assert "DSP" in window.statusBar().currentMessage()
        assert "Display" in window.statusBar().currentMessage()

        symbol_index = 7
        window._symbol_table_cell_clicked(0, symbol_index)

        assert window._selected_symbol_marker_index == symbol_index
        assert set(window._symbol_marker_items) == {
            "iq_power",
            "modulation",
            "symbol_plot",
        }
        power_point, power_label = window._symbol_marker_items["iq_power"]
        modulation_point, modulation_label = window._symbol_marker_items[
            "modulation"
        ]
        symbol_point, symbol_label = window._symbol_marker_items["symbol_plot"]
        assert power_point.opts["symbolSize"] == pytest.approx(18.0)
        assert modulation_point.opts["symbolSize"] == pytest.approx(18.0)
        assert symbol_point.opts["symbolSize"] == pytest.approx(18.0)
        assert power_point.opts["symbolBrush"].color().getRgb()[:3] == (
            0,
            255,
            255,
        )
        power_text = power_label.textItem.toPlainText()
        modulation_text = modulation_label.textItem.toPlainText()
        symbol_text = symbol_label.textItem.toPlainText()
        assert f"Symbol: {symbol_index}" in power_text
        assert "Power:" in power_text and "dBm" in power_text
        assert f"Symbol: {symbol_index}" in modulation_text
        assert "Frequency:" in modulation_text and "kHz" in modulation_text
        assert f"Symbol: {symbol_index}" in symbol_text
        assert "Amplitude:" in symbol_text
        assert "Phase:" in symbol_text and "degree" in symbol_text
        assert "Frequency:" not in symbol_text

        marker_x, marker_y = power_point.getData()
        result = window.session.result
        pattern_result = window.session.pattern_result
        expected_time_s = float(pattern_result.symbol_time_s[symbol_index])
        expected_power = float(
            np.interp(
                expected_time_s,
                window.session.capture_time_s,
                window.session.capture_power_dbm,
            )
        )
        assert marker_x[0] == pytest.approx(expected_time_s * 1e3)
        assert marker_y[0] == pytest.approx(expected_power)

        _modulation_x, modulation_y = modulation_point.getData()
        trace_x, trace_y = window.modulation_plot.listDataItems()[0].getData()
        symbol_i, symbol_q = symbol_point.getData()
        symbol_plot_frequency_hz = (
            np.angle(complex(symbol_i[0], symbol_q[0]))
            * window.session.signal.symbol_rate_hz
            / (2.0 * np.pi)
        )
        assert modulation_y[0] == pytest.approx(
            float(np.interp(expected_time_s * 1e3, trace_x, trace_y))
        )
        window.raw_modulation_signal_action.trigger()
        raw_modulation_point, _raw_modulation_label = (
            window._symbol_marker_items["modulation"]
        )
        _raw_modulation_x, raw_modulation_y = raw_modulation_point.getData()
        raw_trace_x, raw_trace_y = window.modulation_plot.listDataItems()[0].getData()
        assert raw_modulation_y[0] == pytest.approx(
            float(np.interp(expected_time_s * 1e3, raw_trace_x, raw_trace_y))
        )
        assert symbol_plot_frequency_hz == pytest.approx(modulation_y[0] * 1e3)

        window._symbol_table_cell_clicked(0, symbol_index)
        assert window._selected_symbol_marker_index is None
        assert window._symbol_marker_items == {}
    finally:
        window._meas_config_dialog.close()
        window.close()
        window.deleteLater()
        QtWidgets.QApplication.processEvents()


def test_symbol_table_click_places_psk_amplitude_phase_and_evm_markers(
    tmp_path,
) -> None:
    pg.mkQApp("VSA PSK symbol marker test")
    window = VSAWindow(
        preferences=_isolated_preferences(tmp_path, "psk-symbol-marker")
    )
    try:
        window._load_generated(ModulationKind.PI4_DQPSK)
        _wait_for_background_analysis(window)
        expected = np.asarray(window.session.recording.metadata["generated_symbols"])
        window.pattern_search_check.setChecked(True)
        window._set_pattern_symbols(expected[20:36])
        window.result_length_spin.setValue(80)
        assert window._analyze()

        symbol_index = 9
        window._symbol_table_cell_clicked(0, symbol_index)

        assert set(window._symbol_marker_items) == {
            "iq_power",
            "modulation",
            "symbol_plot",
        }
        modulation_text = window._symbol_marker_items[
            "modulation"
        ][1].textItem.toPlainText()
        symbol_text = window._symbol_marker_items[
            "symbol_plot"
        ][1].textItem.toPlainText()
        assert f"Symbol: {symbol_index}" in modulation_text
        assert "Amplitude:" in modulation_text
        assert "Phase:" in modulation_text and "degree" in modulation_text
        assert f"Symbol: {symbol_index}" in symbol_text
        assert "Amplitude:" in symbol_text
        assert "Phase:" in symbol_text and "degree" in symbol_text
        assert "EVM:" in symbol_text and "%" in symbol_text
        assert "Frequency:" not in symbol_text
    finally:
        window._meas_config_dialog.close()
        window.close()
        window.deleteLater()
        QtWidgets.QApplication.processEvents()


def test_qam_display_options_and_iq_markers_are_independent_from_psk(
    tmp_path,
) -> None:
    pg.mkQApp("VSA QAM display separation test")
    window = VSAWindow(
        preferences=_isolated_preferences(tmp_path, "qam-display-separation")
    )
    try:
        window.raw_modulation_signal_action.setChecked(True)
        window.differential_iq_symbol_plot_action.setChecked(True)
        window._load_generated(ModulationKind.QAM16)
        _wait_for_background_analysis(window)

        assert window.qam_modulation_signal_menu.isEnabled()
        assert not window.psk_fsk_modulation_signal_menu.isEnabled()
        assert not window.psk_symbol_plot_menu.isEnabled()
        assert window.qam_measured_modulation_signal_action.isChecked()
        assert window._measured_modulation_signal_selected(ModulationKind.QAM16)
        assert window.symbol_plot_dock.windowTitle() == "Symbol Plot (Physical)"

        symbol_index = 9
        window._symbol_table_cell_clicked(0, symbol_index)
        modulation_text = window._symbol_marker_items[
            "modulation"
        ][1].textItem.toPlainText()
        symbol_text = window._symbol_marker_items[
            "symbol_plot"
        ][1].textItem.toPlainText()
        for text in (modulation_text, symbol_text):
            assert f"Symbol: {symbol_index}" in text
            assert "I:" in text
            assert "Q:" in text
            assert "Amplitude:" not in text
            assert "Phase:" not in text
        assert "EVM:" in symbol_text

        before_i, before_q = window.symbol_plot.listDataItems()[0].getData()
        window.differential_iq_symbol_plot_action.trigger()
        assert window.symbol_plot_dock.windowTitle() == "Symbol Plot (Physical)"
        after_i, after_q = window.symbol_plot.listDataItems()[0].getData()
        np.testing.assert_allclose(after_i, before_i)
        np.testing.assert_allclose(after_q, before_q)
    finally:
        window._meas_config_dialog.close()
        window.close()
        window.deleteLater()
        QtWidgets.QApplication.processEvents()


def test_qam_physical_symbol_plot_stays_carrier_corrected_when_raw_trace_is_selected(
    tmp_path,
) -> None:
    pg.mkQApp("VSA QAM physical carrier correction test")
    fixture = (
        Path(__file__).with_name("fixtures")
        / "bluetooth_hdt7_5_prbs9_16msps.npz"
    )
    recording = FileIQSource.load(fixture)
    sample_index = np.arange(recording.sample_count, dtype=np.float64)
    carrier_offset_hz = 100_000.0
    recording = replace(
        recording,
        iq=(
            recording.iq
            * np.exp(
                1j
                * 2.0
                * np.pi
                * carrier_offset_hz
                * sample_index
                / recording.sample_rate_hz
            )
        ).astype(np.complex64),
    )
    signal = SignalDescription(
        modulation=ModulationKind.QAM16,
        symbol_rate_hz=2_000_000.0,
        tx_filter="Root Raised Cosine",
        filter_parameter=0.4,
        symbol_mapping=BLUETOOTH_HDT_MAPPING,
    )
    session = VSASession(recording=recording, signal=signal)
    session.configure_pattern_analysis(
        PatternSearchSettings(
            pattern=KnownPattern(
                tuple(int(value, 16) for value in "3 E D E 5 0 F 4 7 E".split())
            )
        ),
        ResultRangeSettings(result_length=500),
        DemodulationSettings(
            measurement_filter=MeasurementFilterMode.AUTO,
            bit_ordering=BitOrdering.LSB,
        ),
    )
    session.analyze()

    window = VSAWindow(
        preferences=_isolated_preferences(tmp_path, "qam-raw-trace-symbol-plot")
    )
    try:
        window.session = session
        window.qam_raw_modulation_signal_action.setChecked(True)
        window._update_summary()
        window._update_plots(reset_ranges=True)

        symbol_i, symbol_q = window.symbol_plot.listDataItems()[0].getData()
        displayed = symbol_i + 1j * symbol_q
        alphabet = psk_constellation(ModulationKind.QAM16, BLUETOOTH_HDT_MAPPING)
        reference = alphabet[
            np.argmin(np.abs(displayed[:, None] - alphabet[None, :]), axis=1)
        ]
        display_evm_percent = 100.0 * np.sqrt(
            np.sum(np.abs(displayed - reference) ** 2)
            / np.sum(np.abs(reference) ** 2)
        )

        assert window.qam_raw_modulation_signal_action.isChecked()
        assert display_evm_percent < 4.0
    finally:
        window._meas_config_dialog.close()
        window.close()
        window.deleteLater()
        QtWidgets.QApplication.processEvents()


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


def test_symbol_correct_search_failure_clears_previous_match_display(tmp_path) -> None:
    pg.mkQApp("VSA exact pattern filter UI test")
    window = VSAWindow(
        preferences=_isolated_preferences(tmp_path, "exact-pattern-filter")
    )
    try:
        window._load_generated(ModulationKind.GFSK)
        _wait_for_background_analysis(window)
        generated = np.asarray(
            window.session.recording.metadata["generated_symbols"]
        )
        exact_pattern = [int(value) for value in generated[20:52]]
        window.pattern_search_check.setChecked(True)
        window._set_pattern_symbols(exact_pattern)
        window.result_length_spin.setValue(64)
        assert window._analyze()
        assert window.session.pattern_result is not None

        incorrect_pattern = list(exact_pattern)
        incorrect_pattern[11] = 1 - incorrect_pattern[11]
        window._set_pattern_symbols(incorrect_pattern)
        window.pattern_threshold_auto.setChecked(False)
        window.pattern_threshold_spin.setValue(80.0)
        window.pattern_meas_only_check.setChecked(True)

        assert not window._analyze()
        assert window.session.pattern_result is None
        assert "no symbol-correct pattern match" in window.session.pattern_error
        summary = {
            window.result_summary.item(row, 0).text(): window.result_summary.item(
                row, 1
            ).text()
            for row in range(window.result_summary.rowCount())
        }
        assert "Pattern Error" in summary
        assert len(window.zero_span_plot.listDataItems()) == 1
        assert not window.previous_result_action.isEnabled()
        assert not window.next_result_action.isEnabled()
    finally:
        window._meas_config_dialog.close()
        window.close()
        window.deleteLater()
        QtWidgets.QApplication.processEvents()


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


def test_fsk_symbol_plot_supports_density_trace(tmp_path) -> None:
    pg.mkQApp("VSA FSK density UI test")
    window = VSAWindow(
        preferences=_isolated_preferences(tmp_path, "fsk-density")
    )
    try:
        window._load_generated(ModulationKind.GFSK)
        _wait_for_background_analysis(window)
        assert window._constellation_density_item is None

        window.constellation_density_action.trigger()

        density_item = window._constellation_density_item
        assert density_item is not None
        assert density_item.image.shape == (96, 96)
        assert np.count_nonzero(density_item.image) > 0
        assert density_item.lut[0, 3] == 0
        assert window.session.signal.modulation.family.value == "FSK"
        assert window.symbol_plot.viewRange()[1] == pytest.approx([-1.25, 1.25])
        assert window.symbol_plot.viewRange()[0][0] <= -1.25
        assert window.symbol_plot.viewRange()[0][1] >= 1.25
        # Unit-circle reference plus the hidden finite-data trace remain.
        assert len(window.symbol_plot.listDataItems()) == 2
    finally:
        window._meas_config_dialog.close()
        window.close()
        window.deleteLater()
        QtWidgets.QApplication.processEvents()


def test_psk_constellation_uses_normalized_pattern_result_only(tmp_path) -> None:
    pg.mkQApp("VSA PSK UI test")
    window = VSAWindow(
        preferences=_isolated_preferences(tmp_path, "psk-constellation")
    )
    try:
        fixture = (
            Path(__file__).with_name("fixtures")
            / "bluetooth_2dh1_prbs9_16msps.npz"
        )
        with np.load(fixture, allow_pickle=False) as values:
            pattern = " ".join(
                str(int(value))
                for value in reverse_symbol_bits(
                    values["differential_phase_indices"][:32], 4
                )
            )
        window.load_recording(
            FileIQSource.load(fixture),
            SignalDescription(
                modulation=ModulationKind.PI4_DQPSK,
                symbol_rate_hz=1_000_000.0,
                tx_filter="Root Raised Cosine",
                filter_parameter=0.4,
            ),
        )
        _wait_for_background_analysis(window)
        window.pattern_search_check.setChecked(True)
        window.pattern_format_combo.setCurrentText("Decimal")
        window.pattern_symbols_edit.setText(pattern)
        window.result_length_spin.setValue(244)
        window.channel_filter_check.setChecked(True)
        window.analysis_center_spin.setValue(2441.0)
        window.analysis_bandwidth_spin.setValue(1.5)
        window._analyze()

        trajectory_items = window.modulation_plot.listDataItems()
        assert len(trajectory_items) == 1
        assert trajectory_items[0].opts["pen"].color().getRgb()[:3] == (
            255,
            255,
            0,
        )
        trajectory_i, trajectory_q = trajectory_items[0].getData()
        assert trajectory_i.size == trajectory_q.size
        assert 1_000 < trajectory_i.size < 3_000

        plot_items = window.symbol_plot.listDataItems()
        assert len(plot_items) == 2
        assert plot_items[0].opts["symbolBrush"].color().getRgb()[:3] == (
            255,
            255,
            0,
        )
        assert plot_items[0].opts["symbolPen"].color().getRgb()[:3] == (
            255,
            255,
            0,
        )
        i_values, q_values = plot_items[0].getData()
        magnitude = np.hypot(i_values, q_values)
        assert magnitude.size == 244
        assert np.median(magnitude) == pytest.approx(1.0, abs=0.03)
        assert np.min(magnitude) > 0.85
        assert np.max(magnitude) < 1.10
        # R&S-style QPSK-family display compensates the pi/4 rotation, placing
        # decision points on the I/Q axes while leaving decoded symbols intact.
        distance_from_nearest_axis = np.minimum(np.abs(i_values), np.abs(q_values))
        assert np.percentile(distance_from_nearest_axis, 95) < 0.08
        circle_i, circle_q = plot_items[1].getData()
        np.testing.assert_allclose(
            np.hypot(circle_i, circle_q), 1.0, atol=1e-12
        )
        x_range, y_range = window.symbol_plot.viewRange()
        assert x_range[0] <= -1.0 and x_range[1] >= 1.0
        assert y_range[0] <= -1.0 and y_range[1] >= 1.0
        assert x_range[1] - x_range[0] < 4.0
        assert y_range[1] - y_range[0] < 4.0
        assert y_range == pytest.approx([-1.25, 1.25])
        trajectory_x_range, trajectory_y_range = window.modulation_plot.viewRange()
        assert trajectory_x_range[0] <= float(np.min(trajectory_i))
        assert trajectory_x_range[1] >= float(np.max(trajectory_i))
        assert trajectory_y_range[0] <= float(np.min(trajectory_q))
        assert trajectory_y_range[1] >= float(np.max(trajectory_q))
        assert window.modulation_plot.getViewBox().state["aspectLocked"] == pytest.approx(
            1.0
        )

        assert window.constellation_flat_action.isChecked()
        assert window._constellation_density_item is None
        pattern_result = window.session.pattern_result
        evm_before_display_change = pattern_result.evm_rms_percent
        flat_view_range = window.symbol_plot.viewRange()
        window.constellation_density_action.trigger()
        assert window.constellation_density_action.isChecked()
        assert window.session.pattern_result is pattern_result
        assert window.session.pattern_result.evm_rms_percent == evm_before_display_change
        density_item = window._constellation_density_item
        assert density_item is not None
        assert density_item.image.shape == (96, 96)
        assert np.count_nonzero(density_item.image) > 0
        assert float(np.max(density_item.image)) > 0.0
        assert density_item.lut[0, 3] == 0
        assert window.symbol_plot.viewRange()[0] == pytest.approx(flat_view_range[0])
        assert window.symbol_plot.viewRange()[1] == pytest.approx(flat_view_range[1])
        window.constellation_flat_action.trigger()
        assert window.constellation_flat_action.isChecked()
        assert window._constellation_density_item is None

        window.symbol_display_action.trigger()
        trajectory_items = window.modulation_plot.listDataItems()
        assert len(trajectory_items) == 2
        marker_i, marker_q = trajectory_items[1].getData()
        assert marker_i.size == marker_q.size == 244
        assert trajectory_items[1].opts["symbolSize"] == pytest.approx(5.5)
        marker_magnitude = np.hypot(marker_i, marker_q)
        assert np.median(marker_magnitude) == pytest.approx(1.0, abs=0.02)
        assert np.std(marker_magnitude) < 0.08
        physical_markers = _physical_constellation_display_symbols(
            ModulationKind.PI4_DQPSK, marker_i + 1j * marker_q
        )
        np.testing.assert_allclose(i_values + 1j * q_values, physical_markers)

        window.differential_iq_symbol_plot_action.trigger()
        differential_i, differential_q = window.symbol_plot.listDataItems()[0].getData()
        expected_differential = _constellation_display_symbols(
            ModulationKind.PI4_DQPSK, pattern_result.measured_symbols
        )
        np.testing.assert_allclose(
            differential_i + 1j * differential_q,
            expected_differential,
        )
    finally:
        window._meas_config_dialog.close()
        window.close()
        window.deleteLater()
        QtWidgets.QApplication.processEvents()


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


def test_pattern_table_config_round_trip_and_directory_preferences(tmp_path) -> None:
    pg.mkQApp("VSA config UI test")
    preferences = QtCore.QSettings(
        str(tmp_path / "preferences.ini"), QtCore.QSettings.Format.IniFormat
    )
    window = VSAWindow(preferences=preferences)
    try:
        window.pattern_format_combo.setCurrentText("Decimal")
        window._set_pattern_symbols([0, 1, 1, 0, 1, 0])
        window.pattern_name_edit.setText("Saved Pattern")
        window.pattern_allow_inverted_fsk_check.setChecked(True)
        window.result_length_spin.setValue(73)
        window.result_offset_spin.setValue(-12)
        window.exclude_incomplete_result_check.setChecked(True)
        window.bit_order_combo.setCurrentText("LSB")
        window.capture_length_spin.setValue(3.0)
        window.capture_length_unit_combo.setCurrentText("ms")
        window.capture_oversampling_combo.setCurrentIndex(
            window.capture_oversampling_combo.findData(8)
        )
        window.capture_center_spin.setValue(2441.0)
        window.capture_rf_bandwidth_spin.setValue(8.0)
        window.channel_filter_check.setChecked(True)
        window.analysis_center_spin.setValue(2441.0)
        window.analysis_bandwidth_spin.setValue(1.5)
        window.lo_offset_check.setChecked(True)
        window.lo_offset_spin.setValue(1.25)
        window.internal_gain_spin.setValue(12)
        window.external_attenuation_spin.setValue(30.0)
        window.external_gain_spin.setValue(3.0)
        window.acquisition_trigger_source_combo.setCurrentIndex(
            window.acquisition_trigger_source_combo.findData("power_level")
        )
        window.acquisition_trigger_level_spin.setValue(-25.0)
        window.acquisition_trigger_slope_combo.setCurrentIndex(
            window.acquisition_trigger_slope_combo.findData("falling")
        )
        window.acquisition_trigger_offset_spin.setValue(-12.5)
        window.acquisition_trigger_hysteresis_spin.setValue(5.0)
        window.iq_power_trigger_check.setChecked(True)
        window.iq_power_trigger_level_spin.setValue(-18.5)
        window.iq_power_trigger_hysteresis_spin.setValue(4.0)
        window.iq_power_trigger_average_spin.setValue(1.5)
        window.iq_power_trigger_dropout_spin.setValue(12.0)
        window.iq_power_trigger_holdoff_spin.setValue(20.0)
        window.iq_power_trigger_offset_spin.setValue(1.5)
        window.iq_power_trigger_limit_result_check.setChecked(False)
        window._apply_result_summary_preset("diagnostics")
        window.symbol_display_action.setChecked(True)
        window.symbol_table_decimal_action.setChecked(True)
        window.measured_iq_power_action.setChecked(True)
        window.analysis_spectrum_display_check.setChecked(False)
        window.raw_modulation_signal_action.setChecked(True)
        window.qam_raw_modulation_signal_action.setChecked(True)
        window.constellation_density_action.setChecked(True)
        window.constellation_density_spread_actions[
            SymbolDensitySpread.MEDIUM
        ].setChecked(True)
        window.differential_iq_symbol_plot_action.setChecked(True)
        window.fsk_constellation_frequency_action.setChecked(True)
        window.measurement_filter_combo.setCurrentText("None")
        window._set_selected_pluto_target("serial:rx-a")
        selected_summary_items = set(window._selected_result_summary_ids)
        saved = window._meas_config_values()
        assert "pluto_uri" not in saved["input_frontend"]
        assert "match_selection" not in saved["pattern_search"]
        assert "match_index" not in saved["pattern_search"]
        assert saved["pattern_search"]["allow_inverted_fsk_pattern"] is True
        assert saved["result_range"]["offset_symbols"] == -12
        assert saved["input_frontend"]["apply_analysis_bandwidth_to_power"] is True
        assert saved["input_frontend"]["apply_analysis_bandwidth_to_spectrum"] is False

        window._set_pattern_symbols([1, 1, 1, 1])
        window.pattern_allow_inverted_fsk_check.setChecked(False)
        window.result_length_spin.setValue(12)
        window.result_offset_spin.setValue(0)
        window.exclude_incomplete_result_check.setChecked(False)
        window.bit_order_combo.setCurrentText("MSB")
        window.capture_oversampling_combo.setCurrentIndex(
            window.capture_oversampling_combo.findData(16)
        )
        window.internal_gain_spin.setValue(0)
        window.acquisition_trigger_source_combo.setCurrentIndex(
            window.acquisition_trigger_source_combo.findData("free_run")
        )
        window.iq_power_trigger_check.setChecked(False)
        window.iq_power_trigger_average_spin.setValue(0.0)
        window.iq_power_trigger_limit_result_check.setChecked(True)
        window._apply_result_summary_preset("defaults")
        window.symbol_display_action.setChecked(False)
        window.symbol_table_hex_action.setChecked(True)
        window.raw_iq_power_action.setChecked(True)
        window.analysis_spectrum_display_check.setChecked(True)
        window.measured_modulation_signal_action.setChecked(True)
        window.qam_measured_modulation_signal_action.setChecked(True)
        window.constellation_flat_action.setChecked(True)
        window.constellation_density_spread_actions[
            SymbolDensitySpread.MAXIMUM
        ].setChecked(True)
        window.physical_iq_symbol_plot_action.setChecked(True)
        window.fsk_phase_difference_action.setChecked(True)
        window.measurement_filter_combo.setCurrentText("Auto")
        window._set_selected_pluto_target("serial:rx-b")
        saved["input_frontend"]["pluto_uri"] = "serial:legacy-config-value"
        window._apply_meas_config_values(saved)

        assert window._parse_pattern_symbols(2) == (0, 1, 1, 0, 1, 0)
        assert window.pattern_name_edit.text() == "Saved Pattern"
        assert window.pattern_allow_inverted_fsk_check.isChecked()
        assert window.result_length_spin.value() == 73
        assert window.result_offset_spin.value() == -12
        assert window.exclude_incomplete_result_check.isChecked()
        assert window.bit_order_combo.currentText() == "LSB"
        assert window.measurement_filter_combo.currentText() == "None"
        assert window._selected_pluto_target() == "serial:rx-b"
        assert window.capture_oversampling_combo.currentData() == 8
        assert window.capture_sample_rate_label.text() == "8.000 MS/s"
        assert window.capture_samples_label.text() == "24,000 samples"
        assert window.capture_usable_bandwidth_label.text() == "6.400 MHz"
        assert window.lo_offset_check.isChecked()
        assert window.lo_offset_spin.value() == pytest.approx(1.25)
        assert window.lo_offset_status_label.text().startswith("2442.250000 MHz")
        assert saved["input_frontend"]["lo_offset_enabled"] is True
        assert saved["input_frontend"]["lo_offset_mhz"] == pytest.approx(1.25)
        assert window.internal_gain_spin.value() == 12
        assert window.capture_correction_label.text().startswith("+15.0 dB")
        assert window.acquisition_trigger_source_combo.currentData() == "power_level"
        assert window.acquisition_trigger_level_spin.value() == pytest.approx(-25.0)
        assert window.acquisition_trigger_slope_combo.currentData() == "falling"
        assert window.acquisition_trigger_offset_spin.value() == pytest.approx(-12.5)
        assert window.acquisition_trigger_hysteresis_spin.value() == pytest.approx(5.0)
        assert window.iq_power_trigger_check.isChecked()
        assert window.iq_power_trigger_level_spin.value() == pytest.approx(-18.5)
        assert window.iq_power_trigger_hysteresis_spin.value() == pytest.approx(4.0)
        assert window.iq_power_trigger_average_spin.value() == pytest.approx(1.5)
        assert window.iq_power_trigger_dropout_spin.value() == pytest.approx(12.0)
        assert window.iq_power_trigger_holdoff_spin.value() == pytest.approx(20.0)
        assert window.iq_power_trigger_offset_spin.value() == pytest.approx(1.5)
        assert not window.iq_power_trigger_limit_result_check.isChecked()
        assert window._selected_result_summary_ids == selected_summary_items
        assert set(saved["result_summary"]["visible_items"]) == selected_summary_items
        assert saved["display_config"]["show_symbol_points"] is True
        assert saved["display_config"]["symbol_table_format"] == "Decimal"
        assert saved["display_config"]["iq_power_signal"] == "Measured"
        assert saved["display_config"]["modulation_signal"] == "Raw IQ"
        assert saved["display_config"]["qam_modulation_signal"] == "Raw IQ"
        assert "carrier_display" not in saved["display_config"]
        assert saved["display_config"]["constellation_trace_mode"] == "Density"
        assert saved["display_config"]["constellation_density_spread"] == "Medium"
        assert saved["display_config"]["psk_symbol_plot_mode"] == "Differential IQ"
        assert saved["display_config"]["fsk_symbol_plot_mode"] == (
            "Constellation Frequency"
        )
        assert window.symbol_display_action.isChecked()
        assert window.symbol_table_decimal_action.isChecked()
        assert window.measured_iq_power_action.isChecked()
        assert window.analysis_power_display_check.isChecked()
        assert not window.analysis_spectrum_display_check.isChecked()
        assert window.raw_modulation_signal_action.isChecked()
        assert window.qam_raw_modulation_signal_action.isChecked()
        assert window.constellation_density_action.isChecked()
        assert window._symbol_density_spread() is SymbolDensitySpread.MEDIUM
        assert window.differential_iq_symbol_plot_action.isChecked()
        assert window.fsk_constellation_frequency_action.isChecked()
        assert window.pattern_symbol_table.item(0, 1).text() == "1"
        new_item = QtWidgets.QTableWidgetItem("1")
        window.pattern_symbol_table.setItem(0, 6, new_item)
        assert new_item.textAlignment() == int(
            QtCore.Qt.AlignmentFlag.AlignCenter
        )

        legacy = json.loads(json.dumps(saved))
        legacy["demodulation"].pop("measurement_filter")
        window._apply_meas_config_values(legacy)
        assert window.measurement_filter_combo.currentText() == "Auto"
        legacy["display_config"].pop("modulation_signal")
        legacy["display_config"]["carrier_display"] = "Carrier Corrected"
        window._apply_meas_config_values(legacy)
        assert window.measured_modulation_signal_action.isChecked()
        legacy["display_config"]["carrier_display"] = "Raw IQ"
        window._apply_meas_config_values(legacy)
        assert window.raw_modulation_signal_action.isChecked()
        legacy["display_config"].pop("qam_modulation_signal")
        window._apply_meas_config_values(legacy)
        assert window.qam_measured_modulation_signal_action.isChecked()
        legacy["display_config"].pop("symbol_table_format")
        window._apply_meas_config_values(legacy)
        assert window.symbol_table_hex_action.isChecked()
        legacy["display_config"].pop("iq_power_signal")
        window._apply_meas_config_values(legacy)
        assert window.raw_iq_power_action.isChecked()
        legacy["display_config"].pop("constellation_density_spread")
        window._apply_meas_config_values(legacy)
        assert window._symbol_density_spread() is SymbolDensitySpread.MAXIMUM

        iq_path = tmp_path / "captures" / "sample.npz"
        pattern_path = tmp_path / "patterns" / "access.vsapattern.json"
        config_path = tmp_path / "configs" / "measurement.vsaconfig.json"
        symbol_path = tmp_path / "exports" / "symbols.vsasymbols.json"
        iq_path.parent.mkdir()
        pattern_path.parent.mkdir()
        config_path.parent.mkdir()
        symbol_path.parent.mkdir()
        window._remember_directory("iq", iq_path)
        window._remember_directory("pattern", pattern_path)
        window._remember_directory("config", config_path)
        window._remember_directory("symbol_table", symbol_path)
        assert window._last_directory("iq") == str(iq_path.parent.resolve())
        assert window._last_directory("pattern") == str(pattern_path.parent.resolve())
        assert window._last_directory("config") == str(config_path.parent.resolve())
        assert window._last_directory("symbol_table") == str(
            symbol_path.parent.resolve()
        )
        assert len(
            {
                window._last_directory("iq"),
                window._last_directory("pattern"),
                window._last_directory("config"),
                window._last_directory("symbol_table"),
            }
        ) == 4
    finally:
        window._meas_config_dialog.close()
        window.close()
        window.deleteLater()
        QtWidgets.QApplication.processEvents()


def test_bit_ordering_defaults_and_missing_config_migrate_to_lsb(tmp_path) -> None:
    pg.mkQApp("VSA LSB default test")
    window = VSAWindow(
        preferences=_isolated_preferences(tmp_path, "lsb-default")
    )
    try:
        assert window.bit_order_combo.currentText() == "LSB"
        saved = window._meas_config_values()
        saved["demodulation"].pop("bit_ordering")
        window.bit_order_combo.setCurrentText("MSB")

        window._apply_meas_config_values(saved)

        assert window.bit_order_combo.currentText() == "LSB"
    finally:
        window._meas_config_dialog.close()
        window.close()
        window.deleteLater()
        QtWidgets.QApplication.processEvents()


def test_symbol_table_json_export_document_contains_machine_readable_context(
    tmp_path, monkeypatch,
) -> None:
    pg.mkQApp("VSA Symbol Table export test")
    window = VSAWindow(
        preferences=_isolated_preferences(tmp_path, "symbol-export")
    )
    try:
        window._load_generated(ModulationKind.GFSK)
        _wait_for_background_analysis(window)
        expected = np.asarray(window.session.recording.metadata["generated_symbols"])
        window.pattern_search_check.setChecked(True)
        window._set_pattern_symbols(expected[20:52])
        window.result_length_spin.setValue(64)
        assert window._analyze()

        document = window._symbol_table_export_document()

        assert document["schema"] == "pluto-vsa-symbol-table"
        assert document["version"] == 1
        assert document["metadata"]["modulation"] == "FSK"
        assert document["metadata"]["symbol_mapping"] == "Natural"
        assert document["metadata"]["pattern"]["match_variant"] == "Normal"
        assert document["columns"] == [
            "index",
            "symbol",
            "bits",
            "time_s",
            "pattern_index",
            "pattern_status",
        ]
        assert len(document["rows"]) == 64
        assert document["rows"][0][0:3] == [0, int(expected[20]), [int(expected[20])]]
        assert document["rows"][0][4:] == [0, "matched"]
        assert all(row[5] == "matched" for row in document["rows"][:32])
        assert all(row[5] == "outside" for row in document["rows"][32:])
        export_stem = tmp_path / "symbol-exports" / "capture-symbols"
        export_stem.parent.mkdir()
        monkeypatch.setattr(
            QtWidgets.QFileDialog,
            "getSaveFileName",
            lambda *args, **kwargs: (str(export_stem), ""),
        )
        window._export_symbol_table()
        export_path = export_stem.with_suffix(".vsasymbols.json")
        written = json.loads(export_path.read_text(encoding="utf-8"))
        assert written == document
        assert window._last_directory("symbol_table") == str(
            export_stem.parent.resolve()
        )
    finally:
        window._meas_config_dialog.close()
        window.close()
        window.deleteLater()
        QtWidgets.QApplication.processEvents()


def test_iq_export_can_save_raw_or_software_dc_removed_capture(
    tmp_path, monkeypatch,
) -> None:
    pg.mkQApp("VSA IQ export test")
    generated, signal = GeneratedIQSource.fsk(symbol_count=96, seed=908)
    recording = replace(
        generated,
        iq=(np.asarray(generated.iq) + (0.27 - 0.14j)).astype(np.complex64),
        source="VSA Pluto Single",
        metadata={
            **dict(generated.metadata),
            "dc_removal_recommended": True,
        },
    )
    window = VSAWindow(
        preferences=_isolated_preferences(tmp_path, "iq-export")
    )
    try:
        window.load_recording(recording, signal)
        _wait_for_background_analysis(window)
        assert window.export_iq_action.isEnabled()

        export_dir = tmp_path / "iq-exports"
        export_dir.mkdir()
        raw_stem = export_dir / "raw-capture"
        monkeypatch.setattr(
            QtWidgets.QInputDialog,
            "getItem",
            lambda *args, **kwargs: ("Raw capture", True),
        )
        monkeypatch.setattr(
            QtWidgets.QFileDialog,
            "getSaveFileName",
            lambda *args, **kwargs: (str(raw_stem), ""),
        )
        window._export_iq_recording()
        raw = FileIQSource.load(raw_stem.with_suffix(".npz"))
        np.testing.assert_array_equal(raw.iq, recording.iq)
        assert raw.metadata["dc_removal_recommended"] is True

        corrected_stem = export_dir / "dc-removed"
        monkeypatch.setattr(
            QtWidgets.QInputDialog,
            "getItem",
            lambda *args, **kwargs: (
                "Software DC removed (full-rate capture)",
                True,
            ),
        )
        monkeypatch.setattr(
            QtWidgets.QFileDialog,
            "getSaveFileName",
            lambda *args, **kwargs: (str(corrected_stem), ""),
        )
        window._export_iq_recording()
        corrected = FileIQSource.load(corrected_stem.with_suffix(".npz"))
        offset = complex(
            corrected.metadata["software_dc_offset_real"],
            corrected.metadata["software_dc_offset_imag"],
        )
        np.testing.assert_allclose(
            corrected.iq,
            np.asarray(recording.iq) - offset,
            atol=2e-7,
        )
        assert corrected.sample_count == recording.sample_count
        assert corrected.sample_rate_hz == recording.sample_rate_hz
        assert corrected.metadata["software_dc_removal_applied"] is True
        assert corrected.metadata["dc_removal_recommended"] is False
        assert window._last_directory("iq") == str(export_dir.resolve())
    finally:
        window._meas_config_dialog.close()
        window.close()
        window.deleteLater()
        QtWidgets.QApplication.processEvents()


def test_lsb_bit_ordering_applies_to_psk_pattern_table_and_export(tmp_path) -> None:
    pg.mkQApp("VSA LSB Symbol Table test")
    window = VSAWindow(
        preferences=_isolated_preferences(tmp_path, "lsb-symbol-table")
    )
    try:
        recording, signal = GeneratedIQSource.psk(
            modulation=ModulationKind.PI4_DQPSK,
            symbol_count=180,
            seed=822,
        )
        expected = np.asarray(
            recording.metadata["generated_symbols"], dtype=np.int16
        )
        displayed = np.asarray([0, 2, 1, 3], dtype=np.int16)[expected]
        window.load_recording(recording, signal)
        _wait_for_background_analysis(window)
        window.pattern_search_check.setChecked(True)
        window.bit_order_combo.setCurrentText("LSB")
        window._set_pattern_symbols(displayed[30:54])
        window.result_length_spin.setValue(64)
        assert window._analyze()

        result = window.session.pattern_result
        assert result is not None
        np.testing.assert_array_equal(result.decoded_symbols, expected[30:94])
        table_values = np.asarray(
            [
                int(window.symbol_table.item(index // 10, index % 10).text())
                for index in range(result.decoded_symbols.size)
            ]
        )
        np.testing.assert_array_equal(table_values, displayed[30:94])
        document = window._symbol_table_export_document()
        assert document["metadata"]["bit_ordering"] == "LSB"
        assert document["rows"][0][1] == int(displayed[30])
        assert document["rows"][0][2] == [
            (int(displayed[30]) >> 1) & 1,
            int(displayed[30]) & 1,
        ]
        assert all(row[5] == "matched" for row in document["rows"][:24])
    finally:
        window._meas_config_dialog.close()
        window.close()
        window.deleteLater()
        QtWidgets.QApplication.processEvents()


def test_inverted_fsk_match_ui_keeps_observed_symbols_and_reports_variant(
    tmp_path,
) -> None:
    pg.mkQApp("VSA inverted FSK pattern UI test")
    window = VSAWindow(
        preferences=_isolated_preferences(tmp_path, "inverted-fsk-ui")
    )
    try:
        recording, signal = GeneratedIQSource.fsk(
            symbol_count=180,
            gaussian_bt=0.5,
            seed=616,
        )
        expected = np.asarray(recording.metadata["generated_symbols"], dtype=np.uint8)
        inverted = IQRecording(
            iq=np.conj(recording.iq),
            sample_rate_hz=recording.sample_rate_hz,
            source="conjugated test capture",
        )
        window.load_recording(inverted, signal)
        _wait_for_background_analysis(window)
        window.pattern_search_check.setChecked(True)
        window._set_pattern_symbols(expected[30:62])
        window.pattern_allow_inverted_fsk_check.setChecked(True)
        window.result_length_spin.setValue(80)
        assert window._analyze()

        result = window.session.pattern_result
        assert result is not None
        np.testing.assert_array_equal(
            result.decoded_symbols,
            1 - expected[30:110],
        )
        displayed = [
            int(window.symbol_table.item(index // 10, index % 10).text())
            for index in range(result.decoded_symbols.size)
        ]
        np.testing.assert_array_equal(displayed, result.decoded_symbols)
        green_cells = [
            window.symbol_table.item(index // 10, index % 10)
            for index in range(result.decoded_symbols.size)
            if window.symbol_table.item(
                index // 10, index % 10
            ).background().color().green() > 80
        ]
        assert len(green_cells) == 32
        summary = {
            window.result_summary.item(row, 0).text():
            window.result_summary.item(row, 1).text()
            for row in range(window.result_summary.rowCount())
        }
        assert summary["Pattern Match"] == "Inverted"
        document = window._symbol_table_export_document()
        assert document["metadata"]["pattern"]["match_variant"] == "Inverted"
        assert document["rows"][0][1] == int(1 - expected[30])
    finally:
        window._meas_config_dialog.close()
        window.close()
        window.deleteLater()
        QtWidgets.QApplication.processEvents()


def test_startup_restores_meas_config_without_restoring_iq(tmp_path) -> None:
    pg.mkQApp("VSA startup persistence test")
    preferences_path = tmp_path / "startup-preferences.ini"
    preferences = QtCore.QSettings(
        str(preferences_path), QtCore.QSettings.Format.IniFormat
    )
    first = VSAWindow(preferences=preferences)
    try:
        assert first.session.recording is None
        assert first.summary_label.text() == "No capture"
        recording, signal = GeneratedIQSource.fsk(symbol_count=96, seed=412)
        first.load_recording(recording, signal)
        _wait_for_background_analysis(first)
        first.input_source_combo.setCurrentText("IQ File")
        first.capture_center_spin.setValue(2450.5)
        first.internal_gain_spin.setValue(17)
        first.external_attenuation_spin.setValue(24.0)
        first.result_length_spin.setValue(91)
        first.pattern_name_edit.setText("Restored startup pattern")
        first._set_pattern_symbols([1, 0, 1, 1, 0, 0, 1, 0])
        first._apply_result_summary_preset("measurement")
        first.constellation_density_action.setChecked(True)
        first._set_selected_pluto_target("serial:test-pluto")
        assert preferences.value("pluto/selected_target", "", type=str) == (
            "serial:test-pluto"
        )
        assert first._pluto_capture_settings().sdr_uri == "serial:test-pluto"
        expected_summary_items = set(first._selected_result_summary_ids)
    finally:
        first._meas_config_dialog.close()
        first.close()
        first.deleteLater()
        QtWidgets.QApplication.processEvents()

    serialized = preferences.value("startup/measurement_config", "", type=str)
    document = json.loads(serialized)
    assert document["schema"] == "pluto-vsa-startup-config"
    assert not {
        "iq",
        "iq_path",
        "recording",
        "recording_path",
    }.intersection(document["settings"])
    assert "pluto_uri" not in document["settings"]["input_frontend"]

    restored_preferences = QtCore.QSettings(
        str(preferences_path), QtCore.QSettings.Format.IniFormat
    )
    second = VSAWindow(preferences=restored_preferences)
    try:
        assert second.session.recording is None
        assert second.session.result is None
        assert second.summary_label.text() == "No capture"
        for plot in (second.modulation_plot, second.symbol_plot):
            _x_range, y_range = plot.viewRange()
            assert y_range == pytest.approx([-1.25, 1.25])
            assert plot.getViewBox().state["aspectLocked"] == pytest.approx(1.0)
        assert second.input_source_combo.currentText() == "IQ File"
        assert second.capture_center_spin.value() == pytest.approx(2450.5)
        assert second.internal_gain_spin.value() == 17
        assert second.external_attenuation_spin.value() == pytest.approx(24.0)
        assert second.result_length_spin.value() == 91
        assert second.pattern_name_edit.text() == "Restored startup pattern"
        assert second._parse_pattern_symbols(2) == (1, 0, 1, 1, 0, 0, 1, 0)
        assert second._selected_result_summary_ids == expected_summary_items
        assert second.constellation_density_action.isChecked()
        assert second._selected_pluto_target() == "serial:test-pluto"
        assert not second._analyze()
        assert "configuration restored" in second.statusBar().currentMessage()
    finally:
        second._meas_config_dialog.close()
        second.close()
        second.deleteLater()
        QtWidgets.QApplication.processEvents()


def test_invalid_startup_config_falls_back_to_empty_session(tmp_path) -> None:
    pg.mkQApp("VSA invalid startup persistence test")
    preferences = QtCore.QSettings(
        str(tmp_path / "invalid-preferences.ini"),
        QtCore.QSettings.Format.IniFormat,
    )
    preferences.setValue("startup/measurement_config", "{invalid json")
    preferences.sync()

    window = VSAWindow(preferences=preferences)
    try:
        assert window.session.recording is None
        assert window.summary_label.text() == "No capture"
        assert not preferences.contains("startup/measurement_config")
    finally:
        window._meas_config_dialog.close()
        window.close()
        window.deleteLater()
        QtWidgets.QApplication.processEvents()


def test_pluto_run_single_uses_async_capture_and_updates_session(tmp_path) -> None:
    pg.mkQApp("VSA Pluto UI test")

    class FakePlutoSource:
        def __init__(self) -> None:
            self.settings = None
            self.closed = False

        def capture_single(self, settings, *, cancelled=None, armed=None):
            self.settings = settings
            if armed is not None:
                armed()
            recording, _signal = GeneratedIQSource.fsk(
                symbol_count=64,
                symbol_rate_hz=settings.symbol_rate_hz,
                samples_per_symbol=settings.samples_per_symbol,
            )
            return recording

        def close(self) -> None:
            self.closed = True

    source = FakePlutoSource()
    window = VSAWindow(
        preferences=_isolated_preferences(tmp_path, "pluto-run-single"),
        pluto_source=source,
    )
    try:
        window._run_pluto_single()
        for _index in range(200):
            QtWidgets.QApplication.processEvents()
            thread = window._pluto_capture_thread
            if thread is None:
                break
            thread.wait(10)

        _wait_for_background_analysis(window)

        assert window._pluto_capture_thread is None
        assert source.settings is not None
        assert source.settings.samples_per_symbol == 8
        assert source.settings.requested_sample_rate_hz == 8_000_000
        assert source.settings.capture_samples == 24_000
        assert window.input_source_combo.currentText() == "Pluto"
        assert window.run_single_action.isEnabled()
        assert window.session.recording.sample_rate_hz == 8_000_000.0
        status = window.statusBar().currentMessage()
        assert "Capture" in status
        assert "DSP" in status
        assert "Display" in status
        assert "Total" in status
    finally:
        window._meas_config_dialog.close()
        window.close()
        window.deleteLater()
        QtWidgets.QApplication.processEvents()
    assert source.closed


def test_pluto_continuous_applies_backpressure_and_updates_all_packets(
    tmp_path,
) -> None:
    pg.mkQApp("VSA Pluto Continuous UI test")

    class FakePlutoSource:
        def __init__(self) -> None:
            self.capture_count = 0

        def capture_single(
            self,
            settings,
            *,
            cancelled=None,
            armed=None,
            prefer_buffered=False,
        ):
            assert prefer_buffered
            self.capture_count += 1
            if armed is not None:
                armed()
            recording, _signal = GeneratedIQSource.fsk(
                symbol_count=64,
                symbol_rate_hz=settings.symbol_rate_hz,
                samples_per_symbol=settings.samples_per_symbol,
                seed=self.capture_count,
            )
            return recording

        def close(self) -> None:
            pass

    source = FakePlutoSource()
    window = VSAWindow(
        preferences=_isolated_preferences(tmp_path, "pluto-continuous"),
        pluto_source=source,
    )
    window.analysis_published.connect(window._toggle_pluto_continuous)
    try:
        window._toggle_pluto_continuous()
        for _index in range(500):
            QtWidgets.QApplication.processEvents()
            capture = window._pluto_capture_thread
            analysis = window._analysis_thread
            if capture is not None:
                capture.wait(10)
            if analysis is not None:
                analysis.wait(10)
            if (
                not window._continuous_run_requested
                and window._pluto_capture_thread is None
                and window._analysis_thread is None
                and window.run_continuous_action.isEnabled()
            ):
                break
        else:
            raise AssertionError("Continuous did not stop after the active analysis")

        assert source.capture_count == 1
        assert window._continuous_sweep_count == 1
        assert window._all_packet_statistics.packet_count == 1
        assert window.result_summary.columnCount() == 3
        assert window.result_summary.horizontalHeaderItem(2).text() == "All Packets"
        assert window._all_packet_summary_values["match_selection"] == "1 packet(s)"
        assert window.run_single_action.isEnabled()
        assert window.run_single_action.text() == "Run Single"
        assert window.run_single_button.text() == "Run Single (Pluto)"
        assert window.open_config_action.isEnabled()
    finally:
        window._meas_config_dialog.close()
        window.close()
        window.deleteLater()
        QtWidgets.QApplication.processEvents()


def test_pluto_continuous_keeps_running_after_transient_failures(
    tmp_path, monkeypatch
) -> None:
    pg.mkQApp("VSA Pluto Continuous transient failure test")
    window = VSAWindow(
        preferences=_isolated_preferences(tmp_path, "pluto-continuous-retry")
    )
    dialogs: list[str] = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "critical",
        lambda _parent, _title, message: dialogs.append(str(message)),
    )
    try:
        window._continuous_run_requested = True
        window._active_analysis_context = {"continuous": True}
        window._pluto_capture_failed("temporary USB timeout")
        assert window._continuous_run_requested
        assert dialogs == []

        generation = window._analysis_generation
        window._analysis_failed(generation, None, "packet not found")
        assert window._continuous_run_requested
        assert "retrying Continuous" in window.statusBar().currentMessage()
    finally:
        window._continuous_run_requested = False
        window._meas_config_dialog.close()
        window.close()
        window.deleteLater()


def test_run_single_action_stops_pending_power_trigger_wait(tmp_path) -> None:
    pg.mkQApp("VSA Pluto cancellation test")
    started = threading.Event()

    class WaitingPlutoSource:
        def capture_single(self, settings, *, cancelled=None, armed=None):
            if armed is not None:
                armed()
            started.set()
            while cancelled is None or not cancelled():
                threading.Event().wait(0.002)
            raise CaptureCancelledError("Pluto capture cancelled")

        def close(self) -> None:
            pass

    window = VSAWindow(
        preferences=_isolated_preferences(tmp_path, "pluto-cancel-single"),
        pluto_source=WaitingPlutoSource(),
    )
    try:
        window.acquisition_trigger_source_combo.setCurrentIndex(
            window.acquisition_trigger_source_combo.findData("power_level")
        )
        window._run_pluto_single()
        assert started.wait(timeout=2.0)
        assert window.run_single_action.text() == "Stop Single"
        assert window.run_single_action.isEnabled()

        window._run_pluto_single()
        for _index in range(200):
            QtWidgets.QApplication.processEvents()
            thread = window._pluto_capture_thread
            if thread is None:
                break
            thread.wait(10)

        assert window._pluto_capture_thread is None
        assert window.run_single_action.text() == "Run Single"
        assert window.run_single_action.isEnabled()
        assert "cancelled" in window.statusBar().currentMessage().lower()
    finally:
        window._meas_config_dialog.close()
        window.close()
        window.deleteLater()
        QtWidgets.QApplication.processEvents()


def test_analysis_runs_outside_gui_thread_and_latest_request_wins(
    tmp_path, monkeypatch
) -> None:
    pg.mkQApp("VSA background analysis test")
    window = VSAWindow(
        preferences=_isolated_preferences(tmp_path, "background-analysis")
    )
    entered = threading.Event()
    release = threading.Event()
    observed_threads: list[QtCore.QThread] = []
    original_analyze = VSASession.analyze

    def delayed_analyze(session):
        observed_threads.append(QtCore.QThread.currentThread())
        if not entered.is_set():
            entered.set()
            assert release.wait(timeout=5.0)
        return original_analyze(session)

    monkeypatch.setattr(VSASession, "analyze", delayed_analyze)
    try:
        recording, signal = GeneratedIQSource.fsk(symbol_count=64, seed=919)
        window.load_recording(recording, signal)
        assert entered.wait(timeout=2.0)

        window.symbol_rate_spin.setValue(900_000.0)
        assert window._request_analysis()
        window.symbol_rate_spin.setValue(800_000.0)
        assert window._request_analysis()
        assert window._pending_analysis is not None

        release.set()
        _wait_for_background_analysis(window)

        assert observed_threads
        assert all(
            thread is not QtWidgets.QApplication.instance().thread()
            for thread in observed_threads
        )
        assert len(observed_threads) == 2
        assert window.session.signal.symbol_rate_hz == pytest.approx(800_000.0)
        assert window.session.result is not None
        assert "Analysis complete" in window.statusBar().currentMessage()
    finally:
        release.set()
        _wait_for_background_analysis(window)
        window._meas_config_dialog.close()
        window.close()
        window.deleteLater()
        QtWidgets.QApplication.processEvents()
