import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace
import numpy as np
import pyqtgraph as pg
import pytest
from pyqtgraph.Qt import QtCore, QtWidgets
from pluto_vsa.ui.main_window import VSAWindow, _FixedInteractionViewBox
from pluto_vsa.model import IQRecording, ModulationKind
from pluto_vsa.mapping import reverse_symbol_bits
from pluto_vsa.session import VSASession
from pluto_vsa.sources import GeneratedIQSource
from _vsa_ui_test_helpers import _isolated_preferences, _wait_for_background_analysis


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
            "Display",
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
        assert window._config_stack.currentIndex() == 1
        assert window._config_back_button.isVisibleTo(window._meas_config_dialog)
        window._config_back_button.click()
        assert window._config_stack.currentIndex() == 0
        active_modal_widgets = []

        def inspect_modality() -> None:
            active_modal_widgets.append(QtWidgets.QApplication.activeModalWidget()._is_draft)
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
        assert active_modal_widgets == [True]
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
