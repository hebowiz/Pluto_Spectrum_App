import os
import json
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
    _constellation_display_symbols,
    _fsk_phase_difference_symbols,
)
from pluto_sa.vsa.model import IQRecording, ModulationKind, SignalDescription
from pluto_sa.vsa.sources import FileIQSource, GeneratedIQSource


def _isolated_preferences(tmp_path: Path, name: str) -> QtCore.QSettings:
    return QtCore.QSettings(
        str(tmp_path / f"{name}.ini"), QtCore.QSettings.Format.IniFormat
    )


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
        window.pattern_search_check.setChecked(True)
        window._set_pattern_symbols(generated[20:36])
        window.result_length_spin.setValue(100)

        assert window._selected_match_index == 1
        assert window._analyze()
        summary_labels = {
            window.result_summary.item(row, 0).text()
            for row in range(window.result_summary.rowCount())
        }
        assert {"EVM RMS", "Symbol Rate Error"}.issubset(summary_labels)
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
        assert float(summary_values["EVM RMS"].split()[0]) == pytest.approx(
            plotted_symbol_evm, abs=0.005
        )
        first_start = window.session.pattern_result.pattern_start_sample
        assert first_start == 20 * 8
        assert not window.previous_result_action.isEnabled()
        assert window.next_result_action.isEnabled()
        assert window.previous_result_action.shortcut().toString() == "Left"
        assert window.next_result_action.shortcut().toString() == "Right"

        window.next_result_action.trigger()
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
        assert window.session.pattern_result.pattern_start_sample == first_start
        assert window._selected_match_index == 1

        window.next_result_action.trigger()
        assert window._selected_match_index == 2
        # Loading new IQ always returns focus to the first eligible packet.
        window.load_recording(combined, signal)
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
            window.session.carrier_corrected_pattern_range_result.spectrum_dbm,
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
        assert window.symbol_plot_dock.windowTitle() == "Symbol Plot"
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
        assert window.corrected_carrier_action.isChecked()
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
        assert summary["Display"] == "Carrier Corrected"

        window.raw_carrier_action.trigger()

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
        assert window._constellation_density_item is None

        window.constellation_density_action.trigger()

        density_item = window._constellation_density_item
        assert density_item is not None
        assert density_item.image.shape == (96, 96)
        assert np.count_nonzero(density_item.image) > 0
        assert density_item.lut[0, 3] == 0
        assert window.session.signal.modulation.family.value == "FSK"
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
                for value in values["differential_phase_indices"][:32]
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
        assert len(plot_items) == 1
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
        window.result_length_spin.setValue(73)
        window.exclude_incomplete_result_check.setChecked(True)
        window.bit_order_combo.setCurrentText("LSB")
        window.capture_length_spin.setValue(3.0)
        window.capture_length_unit_combo.setCurrentText("ms")
        window.capture_oversampling_combo.setCurrentIndex(
            window.capture_oversampling_combo.findData(8)
        )
        window.capture_center_spin.setValue(2441.0)
        window.capture_rf_bandwidth_spin.setValue(8.0)
        window.internal_gain_spin.setValue(12)
        window.external_attenuation_spin.setValue(30.0)
        window.external_gain_spin.setValue(3.0)
        window._apply_result_summary_preset("diagnostics")
        window.constellation_density_action.setChecked(True)
        selected_summary_items = set(window._selected_result_summary_ids)
        saved = window._meas_config_values()
        assert "match_selection" not in saved["pattern_search"]
        assert "match_index" not in saved["pattern_search"]

        window._set_pattern_symbols([1, 1, 1, 1])
        window.result_length_spin.setValue(12)
        window.exclude_incomplete_result_check.setChecked(False)
        window.bit_order_combo.setCurrentText("MSB")
        window.capture_oversampling_combo.setCurrentIndex(
            window.capture_oversampling_combo.findData(16)
        )
        window.internal_gain_spin.setValue(0)
        window._apply_result_summary_preset("defaults")
        window.constellation_flat_action.setChecked(True)
        window._apply_meas_config_values(saved)

        assert window._parse_pattern_symbols(2) == (0, 1, 1, 0, 1, 0)
        assert window.pattern_name_edit.text() == "Saved Pattern"
        assert window.result_length_spin.value() == 73
        assert window.exclude_incomplete_result_check.isChecked()
        assert window.bit_order_combo.currentText() == "LSB"
        assert window.capture_oversampling_combo.currentData() == 8
        assert window.capture_sample_rate_label.text() == "8.000 MS/s"
        assert window.capture_samples_label.text() == "24,000 samples"
        assert window.capture_usable_bandwidth_label.text() == "6.400 MHz"
        assert window.internal_gain_spin.value() == 12
        assert window.capture_correction_label.text().startswith("+15.0 dB")
        assert window._selected_result_summary_ids == selected_summary_items
        assert set(saved["result_summary"]["visible_items"]) == selected_summary_items
        assert saved["display_config"]["constellation_trace_mode"] == "Density"
        assert window.constellation_density_action.isChecked()
        assert window.pattern_symbol_table.item(0, 1).text() == "1"
        new_item = QtWidgets.QTableWidgetItem("1")
        window.pattern_symbol_table.setItem(0, 6, new_item)
        assert new_item.textAlignment() == int(
            QtCore.Qt.AlignmentFlag.AlignCenter
        )

        iq_path = tmp_path / "captures" / "sample.npz"
        pattern_path = tmp_path / "patterns" / "access.vsapattern.json"
        config_path = tmp_path / "configs" / "measurement.vsaconfig.json"
        iq_path.parent.mkdir()
        pattern_path.parent.mkdir()
        config_path.parent.mkdir()
        window._remember_directory("iq", iq_path)
        window._remember_directory("pattern", pattern_path)
        window._remember_directory("config", config_path)
        assert window._last_directory("iq") == str(iq_path.parent.resolve())
        assert window._last_directory("pattern") == str(pattern_path.parent.resolve())
        assert window._last_directory("config") == str(config_path.parent.resolve())
        assert len(
            {
                window._last_directory("iq"),
                window._last_directory("pattern"),
                window._last_directory("config"),
            }
        ) == 3
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
        first.input_source_combo.setCurrentText("IQ File")
        first.capture_center_spin.setValue(2450.5)
        first.internal_gain_spin.setValue(17)
        first.external_attenuation_spin.setValue(24.0)
        first.result_length_spin.setValue(91)
        first.pattern_name_edit.setText("Restored startup pattern")
        first._set_pattern_symbols([1, 0, 1, 1, 0, 0, 1, 0])
        first._apply_result_summary_preset("measurement")
        first.constellation_density_action.setChecked(True)
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

        def capture_single(self, settings):
            self.settings = settings
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

        assert window._pluto_capture_thread is None
        assert source.settings is not None
        assert source.settings.samples_per_symbol == 8
        assert source.settings.requested_sample_rate_hz == 8_000_000
        assert source.settings.capture_samples == 24_000
        assert window.input_source_combo.currentText() == "Pluto"
        assert window.run_single_action.isEnabled()
        assert window.session.recording.sample_rate_hz == 8_000_000.0
    finally:
        window._meas_config_dialog.close()
        window.close()
        window.deleteLater()
        QtWidgets.QApplication.processEvents()
    assert source.closed
