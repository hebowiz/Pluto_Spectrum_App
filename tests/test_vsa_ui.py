import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyqtgraph as pg
import pytest
from pyqtgraph.Qt import QtCore, QtWidgets

from pluto_sa.vsa.ui.main_window import VSAWindow, _constellation_display_symbols
from pluto_sa.vsa.model import ModulationKind, SignalDescription
from pluto_sa.vsa.sources import FileIQSource


def test_pattern_result_uses_table_and_fitted_plot_ranges() -> None:
    pg.mkQApp("VSA UI test")
    window = VSAWindow()
    try:
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
        assert window._meas_config_dialog.isModal()
        assert window._meas_config_dialog.windowModality() != (
            QtCore.Qt.WindowModality.NonModal
        )
        assert window._config_stack.currentIndex() == 0
        assert set(window._config_top_buttons) == {
            "Input / Frontend",
            "Signal Description",
            "Pattern Search",
            "Result Range",
            "Demodulation",
            "Sweep / Run",
        }
        assert all(
            button.font().pointSizeF() >= 18.0
            and button.minimumHeight() >= 84
            for button in window._config_top_buttons.values()
        )
        assert window._config_top_title.font().pointSizeF() >= 16.0
        assert window._config_load_button.text() == "Load Config..."
        assert window._config_save_button.text() == "Save Config As..."
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
        docks = (
            window.zero_span_dock,
            window.spectrum_dock,
            window.result_summary_dock,
            window.modulation_dock,
            window.reserved_dock,
            window.symbol_dock,
        )
        assert all(isinstance(dock, QtWidgets.QDockWidget) for dock in docks)
        assert window.reserved_dock.windowTitle() == "IQ Trajectory"
        trajectory_items = window.iq_trajectory_plot.listDataItems()
        assert len(trajectory_items) == 1
        trajectory_i, trajectory_q = trajectory_items[0].getData()
        assert trajectory_i.size == trajectory_q.size
        assert trajectory_i.size > 0
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
        assert "Carrier Corrected" in window.spectrum_plot.getPlotItem().titleLabel.text
        summary = {
            window.result_summary.item(row, 0).text(): window.result_summary.item(
                row, 1
            ).text()
            for row in range(window.result_summary.rowCount())
        }
        assert "CFO" in summary
        assert summary["Display"] == "Carrier Corrected"

        window.raw_carrier_action.trigger()

        assert "Raw IQ" in window.spectrum_plot.getPlotItem().titleLabel.text
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


def test_psk_constellation_uses_normalized_pattern_result_only() -> None:
    pg.mkQApp("VSA PSK UI test")
    window = VSAWindow()
    try:
        fixture = (
            Path(__file__).with_name("fixtures")
            / "bluetooth_2dh1_prbs9_16msps.npz"
        )
        with np.load(fixture, allow_pickle=False) as values:
            pattern = " ".join(
                str(int(value))
                for value in values["differential_phase_indices"][:10]
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

        plot_items = window.modulation_plot.listDataItems()
        assert len(plot_items) == 1
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
        x_range, y_range = window.modulation_plot.viewRange()
        assert x_range[0] <= -1.0 and x_range[1] >= 1.0
        assert y_range[0] <= -1.0 and y_range[1] >= 1.0
        assert x_range[1] - x_range[0] < 4.0
        assert y_range[1] - y_range[0] < 4.0
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
        window.bit_order_combo.setCurrentText("LSB")
        saved = window._meas_config_values()

        window._set_pattern_symbols([1, 1, 1, 1])
        window.result_length_spin.setValue(12)
        window.bit_order_combo.setCurrentText("MSB")
        window._apply_meas_config_values(saved)

        assert window._parse_pattern_symbols(2) == (0, 1, 1, 0, 1, 0)
        assert window.pattern_name_edit.text() == "Saved Pattern"
        assert window.result_length_spin.value() == 73
        assert window.bit_order_combo.currentText() == "LSB"
        assert window.pattern_symbol_table.item(0, 1).text() == "1"

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
