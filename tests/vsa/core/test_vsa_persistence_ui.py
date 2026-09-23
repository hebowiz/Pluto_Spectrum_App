import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
import pyqtgraph as pg
import pytest
from pyqtgraph.Qt import QtCore, QtWidgets
from pluto_vsa.ui.main_window import VSAWindow
from pluto_vsa.sources import GeneratedIQSource
from pluto_vsa.ui.measurement_chrome import SymbolDensitySpread
from _vsa_ui_test_helpers import _isolated_preferences, _wait_for_background_analysis


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
            # The common right-side VSA UI fixes live input to Pluto. IQ files
            # are opened explicitly from System > File and are not persisted
            # as an input-source mode.
            assert second.input_source_combo.currentText() == "Pluto"
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
