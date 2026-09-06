import os
from dataclasses import replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyqtgraph as pg
import pytest
from pyqtgraph.Qt import QtCore, QtWidgets

from pluto_sa.vsa.protocol_modes.dect import (
    DectModulationReference,
    DectAnalyzerWindow,
    analyze_dect_recording,
    generate_dect_packet,
)
from pluto_sa.vsa.session import VSASession
from pluto_sa.vsa.ui.display_processing import fit_binary_fsk_display_drift


class _Source:
    def close(self) -> None:
        pass


def _window(tmp_path) -> DectAnalyzerWindow:
    pg.mkQApp("DECT dedicated VSA test")
    settings = QtCore.QSettings(
        str(tmp_path / "dect.ini"), QtCore.QSettings.Format.IniFormat
    )
    return DectAnalyzerWindow(pluto_source=_Source(), preferences=settings)


def test_dect_dedicated_exposes_common_iq_export_action(tmp_path) -> None:
    window = _window(tmp_path)
    recording = generate_dect_packet(
        center_frequency_hz=window._nominal_frequency_hz()
    )
    try:
        assert not window.export_iq_action.isEnabled()
        window.stage_session(VSASession(recording=recording))
        assert window.export_iq_action.isEnabled()
        assert window.export_iq_action.text() == "Export IQ Recording..."
    finally:
        window._config_dialog.close()
        window.close()
        window.deleteLater()


def test_dect_workspace_has_independent_carrier_list_and_capture_settings(tmp_path) -> None:
    window = _window(tmp_path)
    try:
        assert window.plan_combo.count() >= 5
        assert window.carrier_combo.count() == 10
        assert window._preferences.applicationName() != "PlutoVSA-Bluetooth"
        capture = window._capture_settings()
        assert capture.center_frequency_hz == window.carrier_combo.currentData()
        assert capture.symbol_rate_hz == 1_152_000.0
        assert capture.requested_sample_rate_hz == 9_216_000
        assert capture.rf_bandwidth_hz >= 3_000_000.0
        assert capture.trigger_source.value == "power_level"
        assert capture.analysis_bandwidth_hz is None
        assert capture.lo_offset_hz == 0.0
        assert window.analysis_power_display_check.isChecked()
        assert not window.analysis_spectrum_display_check.isChecked()
        saved = window._config_values()
        assert saved["apply_analysis_bandwidth_to_power"] is True
        assert saved["apply_analysis_bandwidth_to_spectrum"] is False
        window.lo_offset_check.setChecked(True)
        capture = window._capture_settings()
        assert window.channel_filter_check.isChecked()
        assert capture.analysis_bandwidth_hz == 3_000_000.0
        assert capture.lo_offset_hz == 2_000_000.0
    finally:
        window._config_dialog.close()
        window.close()
        window.deleteLater()


def test_dect_analysis_bandwidth_accepts_and_restores_sub_3_mhz(tmp_path) -> None:
    window = _window(tmp_path)
    preferences = window._preferences
    try:
        window.analysis_bandwidth_spin.setValue(1.5)
        assert window.analysis_bandwidth_spin.value() == pytest.approx(1.5)
        window._save_config()
    finally:
        window._config_dialog.close()
        window.close()
        window.deleteLater()

    restored = DectAnalyzerWindow(pluto_source=_Source(), preferences=preferences)
    try:
        assert restored.analysis_bandwidth_spin.value() == pytest.approx(1.5)
    finally:
        restored._config_dialog.close()
        restored.close()
        restored.deleteLater()


def test_jp_dect_plan_populates_all_named_carriers(tmp_path) -> None:
    window = _window(tmp_path)
    try:
        window.plan_combo.setCurrentIndex(window.plan_combo.findData("j_dect"))
        assert window.carrier_combo.count() == 12
        assert window.carrier_combo.itemText(0) == "F7  1885.248 MHz"
        assert window.carrier_combo.itemText(5) == "F0  1893.888 MHz"
        assert window.carrier_combo.itemText(11) == "F6  1904.256 MHz"
        assert window.carrier_combo.itemData(0) == 1_885_248_000.0
        assert window.carrier_combo.itemData(11) == 1_904_256_000.0
    finally:
        window._config_dialog.close()
        window.close()
        window.deleteLater()


def test_dect_workspace_renders_measurement_and_packet_views(tmp_path) -> None:
    window = _window(tmp_path)
    recording = generate_dect_packet(
        center_frequency_hz=window._nominal_frequency_hz(),
        frequency_error_hz=7_500.0,
    )
    result = analyze_dect_recording(recording)[0]
    try:
        window._recording = recording
        window._results = (result,)
        window._result = result
        window._render(result)
        assert window.summary_table.rowCount() >= 10
        assert window.power_plot.listDataItems()
        assert window.spectrum_plot.listDataItems()
        assert window.deviation_plot.listDataItems()
        assert window.symbol_plot.listDataItems()
        assert window.decode_tree.topLevelItemCount() == 1
        assert window.packet_table.rowCount() == 1
        assert "First transmitted bit" in window.air_bits_text.toPlainText()
        power_names = {
            item.name() for item in window.power_plot.listDataItems()
        }
        assert "Power-Time upper limit" in power_names
        assert "Power-Time lower limit" in power_names
        assert window.power_plot.getViewBox().state["limits"]["yLimits"][0] == -120.0
        power_y = window.power_plot.listDataItems()[0].getData()[1]
        assert np.nanmin(power_y) == -120.0
        assert window.decode_tree.textElideMode() == QtCore.Qt.TextElideMode.ElideNone
        assert window.packet_table.textElideMode() == QtCore.Qt.TextElideMode.ElideNone
        packet_root = window.decode_tree.topLevelItem(0)
        assert packet_root.text(0) == "DECT Packet"
        assert packet_root.child(0).childCount() == 2
        assert "RFP" in packet_root.child(0).text(1)
        assert "320 bits" in packet_root.child(2).text(1)
        raw_b_field = packet_root.child(2).child(2)
        raw_value_index = window.decode_tree.indexFromItem(raw_b_field, 1)
        raw_option = QtWidgets.QStyleOptionViewItem()
        raw_option.initFrom(window.decode_tree)
        raw_hint = window.decode_tree.itemDelegate().sizeHint(
            raw_option, raw_value_index
        )
        assert raw_hint.height() > 2 * raw_option.fontMetrics.height()
        assert window.decode_tree.visualItemRect(raw_b_field).height() > (
            2 * raw_option.fontMetrics.height()
        )
        spectrum_trace = window.spectrum_plot.listDataItems()[0]
        assert spectrum_trace.opts["pen"].color().name() == "#ffff00"
        assert window.spectrum_plot.plotItem.legend is None
        assert window.modulation_reference_actions[
            DectModulationReference.MEASURED
        ].isChecked()
        fm_trace = window.deviation_plot.listDataItems()[0]
        _fm_x, fm_y = fm_trace.getData()
        fm_markers = window.deviation_plot.listDataItems()[1]
        np.testing.assert_allclose(
            fm_markers.yData,
            np.interp(fm_markers.xData, fm_trace.xData, fm_trace.yData),
            atol=1e-9,
        )
        symbol_values = window.symbol_plot.listDataItems()[0].yData
        np.testing.assert_array_equal(symbol_values, fm_markers.yData)
        fm_mask = (
            (result.measurement_fm_sample >= result.metadata["actual_preamble_start_sample"])
            & (result.measurement_fm_sample <= result.packet_end_sample)
        )
        display_drift, display_reference_time = fit_binary_fsk_display_drift(
            result.symbol_centers / recording.sample_rate_hz,
            result.symbol_frequency_hz,
            result.bits,
        )
        fm_time_s = result.measurement_fm_sample[fm_mask] / recording.sample_rate_hz
        assert fm_y == pytest.approx(
            (
                result.measurement_fm_frequency_hz[fm_mask]
                - result.frequency_references.measured_hz
                - display_drift * (fm_time_s - display_reference_time)
            )
            / 1e3
        )
        summary_items = {
            window.summary_table.item(row, 0).text(): row
            for row in range(window.summary_table.rowCount())
            if window.summary_table.item(row, 0) is not None
        }
        gfsk_row = summary_items["GFSK Modulation Deviation"]
        assert window.summary_table.item(gfsk_row, 3).foreground().color().name() == "#43f5a5"
    finally:
        window._config_dialog.close()
        window.close()
        window.deleteLater()


def test_dect_power_and_spectrum_select_capture_or_analysis_plane(tmp_path) -> None:
    window = _window(tmp_path)
    analysis_recording = generate_dect_packet(
        center_frequency_hz=window._nominal_frequency_hz()
    )
    capture_recording = replace(
        analysis_recording,
        iq=np.asarray(0.5 * analysis_recording.iq, dtype=np.complex64),
        center_frequency_hz=analysis_recording.center_frequency_hz + 2_000_000.0,
    )
    result = analyze_dect_recording(analysis_recording)[0]
    try:
        window._capture_recording = capture_recording
        window._recording = analysis_recording
        window._results = (result,)
        window._result = result
        window.channel_filter_check.setChecked(True)
        window.analysis_power_display_check.setChecked(False)
        window.analysis_spectrum_display_check.setChecked(False)
        window._render(result)
        raw_power = window.power_plot.listDataItems()[0].yData
        raw_spectrum_x = window.spectrum_plot.listDataItems()[0].xData

        window.analysis_power_display_check.setChecked(True)
        window.analysis_spectrum_display_check.setChecked(True)
        window._render(result)
        analysis_power = window.power_plot.listDataItems()[0].yData
        analysis_spectrum_x = window.spectrum_plot.listDataItems()[0].xData

        assert np.nanmedian(analysis_power - raw_power) == pytest.approx(
            20.0 * np.log10(2.0), abs=1e-3
        )
        assert np.mean(raw_spectrum_x) == pytest.approx(
            capture_recording.center_frequency_hz / 1e6, abs=0.01
        )
        assert np.mean(analysis_spectrum_x) == pytest.approx(
            analysis_recording.center_frequency_hz / 1e6, abs=0.01
        )
    finally:
        window._config_dialog.close()
        window.close()
        window.deleteLater()


def test_dect_fsk_symbol_plot_uses_common_bt_display_options(tmp_path) -> None:
    window = _window(tmp_path)
    recording = generate_dect_packet(center_frequency_hz=window._nominal_frequency_hz())
    result = analyze_dect_recording(recording)[0]
    try:
        window._recording = recording
        window._results = (result,)
        window._result = result
        window._render(result)
        assert window.fsk_frequency_action.isChecked()
        assert window.symbol_plot.getAxis("bottom").labelText == ""
        window._set_symbol_density(True)
        assert window.density_action.isChecked()
        assert any(
            isinstance(item, pg.ImageItem)
            for item in window.symbol_plot.getPlotItem().items
        )
        window._set_symbol_density_spread("Medium")
        assert window.density_spread_actions[window._symbol_density_spread].isChecked()
        window._set_fsk_symbol_plot_mode("Phase Difference")
        assert window.fsk_phase_action.isChecked()
        assert window.symbol_plot.getAxis("bottom").labelText == "I"
        assert window.symbol_plot.getViewBox().state["aspectLocked"] is not False
    finally:
        window._config_dialog.close()
        window.close()
        window.deleteLater()


def test_dect_modulation_reference_and_debug_export(tmp_path, monkeypatch) -> None:
    window = _window(tmp_path)
    recording = generate_dect_packet(
        center_frequency_hz=window._nominal_frequency_hz(),
        prolonged_preamble=True,
    )
    result = analyze_dect_recording(recording)[0]
    export_path = tmp_path / "dect_modulation.csv"
    try:
        window._recording = recording
        window._results = (result,)
        window._result = result
        window.export_modulation_action.setEnabled(True)
        window._set_modulation_reference(DectModulationReference.NOMINAL)
        assert window.modulation_reference_actions[
            DectModulationReference.NOMINAL
        ].isChecked()
        monkeypatch.setattr(
            QtWidgets.QFileDialog,
            "getSaveFileName",
            lambda *_args, **_kwargs: (str(export_path), "CSV files (*.csv)"),
        )
        window._export_modulation_debug_csv()
        exported = export_path.read_text(encoding="utf-8")
        assert "Raw phase-difference frequency" in exported
        assert "Measurement frequency (no filter)" in exported
        assert "CTS60-compatible 6 SPS,p-16,0/6" in exported
        assert "Symbol decision frequency" in exported
        assert "Ideal BT=0.5 diagnostic fit" in exported
        assert ",Nominal" in exported
    finally:
        window._config_dialog.close()
        window.close()
        window.deleteLater()


def test_iq_power_initial_range_is_packet_plus_margin_not_full_capture(tmp_path) -> None:
    window = _window(tmp_path)
    recording = generate_dect_packet(
        center_frequency_hz=window._nominal_frequency_hz(),
        padding_symbols=200,
    )
    result = analyze_dect_recording(recording)[0]
    try:
        window._recording = recording
        window._results = (result,)
        window._result = result
        window._render(result)
        lower, upper = window.power_plot.viewRange()[0]
        packet_start = result.p0_sample / recording.sample_rate_hz * 1e3
        packet_stop = result.packet_end_sample / recording.sample_rate_hz * 1e3
        assert lower < packet_start
        assert upper > packet_stop
        assert upper - lower < recording.duration_s * 1e3
    finally:
        window._config_dialog.close()
        window.close()
        window.deleteLater()


def test_prolonged_preamble_decode_shows_bit_and_dect_symbol_ranges(tmp_path) -> None:
    window = _window(tmp_path)
    recording = generate_dect_packet(
        direction="PP",
        prolonged_preamble=True,
        center_frequency_hz=window._nominal_frequency_hz(),
    )
    result = analyze_dect_recording(recording)[0]
    try:
        window._recording = recording
        window._results = (result,)
        window._result = result
        window._render(result)
        assert window.decode_tree.columnCount() == 5
        packet_root = window.decode_tree.topLevelItem(0)
        prolonged = packet_root.child(0)
        s_field = packet_root.child(1)
        assert prolonged.text(0) == "Prolonged Preamble"
        assert prolonged.text(2) == "0–15"
        assert prolonged.text(3) == "p-16–p-1"
        assert s_field.text(2) == "16–47"
        assert s_field.text(3) == "p0–p31"
        assert s_field.child(0).text(2) == "16–31"
        assert s_field.child(0).text(3) == "p0–p15"
        b_field = packet_root.child(3)
        value_index = window.decode_tree.indexFromItem(b_field, 1)
        option = QtWidgets.QStyleOptionViewItem()
        option.initFrom(window.decode_tree)
        wrapped_hint = window.decode_tree.itemDelegate().sizeHint(
            option, value_index
        )
        assert wrapped_hint.height() > option.fontMetrics.height() + 6
        rows = {
            window.summary_table.item(row, 0).text(): window.summary_table.item(
                row, 1
            ).text()
            for row in range(window.summary_table.rowCount())
            if window.summary_table.item(row, 0) is not None
            and window.summary_table.item(row, 1) is not None
        }
        assert rows["Preamble Mode"] == "Prolonged"
        assert "Sync Word correlation" in rows
    finally:
        window._config_dialog.close()
        window.close()
        window.deleteLater()


def test_plot_context_reset_only_restores_target_plot(tmp_path) -> None:
    window = _window(tmp_path)
    recording = generate_dect_packet(center_frequency_hz=window._nominal_frequency_hz())
    result = analyze_dect_recording(recording)[0]
    try:
        window._recording = recording
        window._results = (result,)
        window._render(result)
        expected_spectrum = np.asarray(window._analysis_plot_ranges["spectrum"][0])
        np.testing.assert_allclose(
            window._analysis_plot_ranges["gfsk_modulation"][1],
            [-500.0, 500.0],
        )
        window.power_plot.setXRange(10.0, 20.0, padding=0.0)
        window.spectrum_plot.setXRange(30.0, 40.0, padding=0.0)
        window._plot_context_actions["spectrum"]["reset"].trigger()
        np.testing.assert_allclose(
            window.spectrum_plot.viewRange()[0], expected_spectrum, atol=1e-9
        )
        np.testing.assert_allclose(window.power_plot.viewRange()[0], [10.0, 20.0])
        window.deviation_plot.setYRange(-100.0, 100.0, padding=0.0)
        window._plot_context_actions["gfsk_modulation"]["reset"].trigger()
        np.testing.assert_allclose(
            window.deviation_plot.viewRange()[1],
            [-500.0, 500.0],
        )
    finally:
        window._config_dialog.close()
        window.close()
        window.deleteLater()


def test_carrier_verdict_waits_for_required_packet_accumulation(tmp_path) -> None:
    window = _window(tmp_path)
    recording = generate_dect_packet(
        center_frequency_hz=window._nominal_frequency_hz(),
        frequency_error_hz=7_500.0,
    )
    result = analyze_dect_recording(recording)[0]
    try:
        for capture_index in range(10):
            window._recording = recording
            window._recording_revision += 1
            window._analysis_ready((result,))
        items = {
            window.summary_table.item(row, 0).text(): (
                window.summary_table.item(row, 1).text(),
                window.summary_table.item(row, 3).text(),
            )
            for row in range(window.summary_table.rowCount())
            if window.summary_table.item(row, 0) is not None
            and window.summary_table.item(row, 1) is not None
            and window.summary_table.item(row, 3) is not None
        }
        assert items["RF Carrier Frequency Accuracy"][1] == "PASS"
        assert items["Carrier Packets Evaluated"][0] == "10 / 10"
    finally:
        window._config_dialog.close()
        window.close()
        window.deleteLater()
