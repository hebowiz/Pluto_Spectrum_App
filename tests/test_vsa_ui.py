import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyqtgraph as pg
import pytest
from pyqtgraph.Qt import QtCore, QtWidgets

from pluto_sa.vsa.ui.main_window import (
    VSAWindow,
    _constellation_display_symbols,
    _fsk_phase_difference_symbols,
)
from pluto_sa.vsa.model import IQRecording, ModulationKind, SignalDescription
from pluto_sa.vsa.sources import FileIQSource, GeneratedIQSource


def test_result_range_arrow_actions_select_adjacent_packet() -> None:
    pg.mkQApp("VSA result navigation test")
    window = VSAWindow()
    try:
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

        assert window.pattern_match_selection_combo.currentText() == "First"
        assert window._analyze()
        first_start = window.session.pattern_result.pattern_start_sample
        assert first_start == 20 * 8
        assert not window.previous_result_action.isEnabled()
        assert window.next_result_action.isEnabled()
        assert window.previous_result_action.shortcut().toString() == "Left"
        assert window.next_result_action.shortcut().toString() == "Right"

        window.next_result_action.trigger()
        second = window.session.pattern_result
        assert second.pattern_start_sample == recording.sample_count + 64 + 20 * 8
        assert window.pattern_match_selection_combo.currentText() == "Match Index"
        assert window.pattern_match_index_spin.value() == 2
        assert window.previous_result_action.isEnabled()
        assert not window.next_result_action.isEnabled()

        window.previous_result_action.trigger()
        assert window.session.pattern_result.pattern_start_sample == first_start
        assert window.pattern_match_index_spin.value() == 1
    finally:
        window._meas_config_dialog.close()
        window.close()
        window.deleteLater()
        QtWidgets.QApplication.processEvents()


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
        assert not window.symbol_display_action.isChecked()
        assert len(window.zero_span_plot.listDataItems()) == 1
        assert len(window.modulation_plot.listDataItems()) == 1
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
        assert window.rect_zoom_action.isChecked()
        assert all(
            plot.getViewBox().state["mouseMode"] == pg.ViewBox.RectMode
            for _name, plot in window._plot_widgets()
        )
        window.pan_action.trigger()
        assert all(
            plot.getViewBox().state["mouseMode"] == pg.ViewBox.PanMode
            for _name, plot in window._plot_widgets()
        )
        window.rect_zoom_action.trigger()
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
        docks = (
            window.zero_span_dock,
            window.spectrum_dock,
            window.result_summary_dock,
            window.modulation_dock,
            window.reserved_dock,
            window.symbol_dock,
        )
        assert all(isinstance(dock, QtWidgets.QDockWidget) for dock in docks)
        assert window.symbol_plot_dock.windowTitle() == "Symbol Plot"
        phase_items = window.symbol_plot.listDataItems()
        assert len(phase_items) == 2
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
        assert "Carrier Corrected" in window.spectrum_plot.getPlotItem().titleLabel.text
        summary = {
            window.result_summary.item(row, 0).text(): window.result_summary.item(
                row, 1
            ).text()
            for row in range(window.result_summary.rowCount())
        }
        assert "CFO" in summary
        assert "Fractional Timing" in summary
        assert "Timing Confidence" in summary
        assert "Deviation Error" in summary
        assert summary["Drift Model"].startswith(("Accepted", "Rejected"))
        assert "Applied Drift" in summary
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
        # This test validates the known high-correlation packet fixture rather
        # than the application's time-first default. A short ten-symbol word
        # also has an earlier 90%-correlation occurrence in this capture.
        window.pattern_match_selection_combo.setCurrentText("Strongest")
        window.channel_filter_check.setChecked(True)
        window.analysis_center_spin.setValue(2441.0)
        window.analysis_bandwidth_spin.setValue(1.5)
        window._analyze()

        assert "IQ Trajectory" in window.modulation_plot.getPlotItem().titleLabel.text
        trajectory_items = window.modulation_plot.listDataItems()
        assert len(trajectory_items) == 1
        trajectory_i, trajectory_q = trajectory_items[0].getData()
        assert trajectory_i.size == trajectory_q.size
        assert 1_000 < trajectory_i.size < 3_000

        plot_items = window.symbol_plot.listDataItems()
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
        x_range, y_range = window.symbol_plot.viewRange()
        assert x_range[0] <= -1.0 and x_range[1] >= 1.0
        assert y_range[0] <= -1.0 and y_range[1] >= 1.0
        assert x_range[1] - x_range[0] < 4.0
        assert y_range[1] - y_range[0] < 4.0

        window.symbol_display_action.trigger()
        trajectory_items = window.modulation_plot.listDataItems()
        assert len(trajectory_items) == 2
        marker_i, marker_q = trajectory_items[1].getData()
        assert marker_i.size == marker_q.size == 244
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
        window.pattern_match_selection_combo.setCurrentText("Match Index")
        window.pattern_match_index_spin.setValue(3)
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
        saved = window._meas_config_values()

        window._set_pattern_symbols([1, 1, 1, 1])
        window.result_length_spin.setValue(12)
        window.pattern_match_selection_combo.setCurrentText("First")
        window.exclude_incomplete_result_check.setChecked(False)
        window.bit_order_combo.setCurrentText("MSB")
        window.capture_oversampling_combo.setCurrentIndex(
            window.capture_oversampling_combo.findData(16)
        )
        window.internal_gain_spin.setValue(0)
        window._apply_meas_config_values(saved)

        assert window._parse_pattern_symbols(2) == (0, 1, 1, 0, 1, 0)
        assert window.pattern_name_edit.text() == "Saved Pattern"
        assert window.result_length_spin.value() == 73
        assert window.pattern_match_selection_combo.currentText() == "Match Index"
        assert window.pattern_match_index_spin.value() == 3
        assert window.pattern_match_index_spin.isEnabled()
        assert window.exclude_incomplete_result_check.isChecked()
        assert window.bit_order_combo.currentText() == "LSB"
        assert window.capture_oversampling_combo.currentData() == 8
        assert window.capture_sample_rate_label.text() == "8.000 MS/s"
        assert window.capture_samples_label.text() == "24,000 samples"
        assert window.capture_usable_bandwidth_label.text() == "6.400 MHz"
        assert window.internal_gain_spin.value() == 12
        assert window.capture_correction_label.text().startswith("+15.0 dB")
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


def test_pluto_run_single_uses_async_capture_and_updates_session() -> None:
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
    window = VSAWindow(pluto_source=source)
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
