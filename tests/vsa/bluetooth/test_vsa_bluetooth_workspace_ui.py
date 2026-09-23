import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace
from pathlib import Path
import numpy as np
import pyqtgraph as pg
import pytest
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from pluto_vsa.model import IQRecording, ModulationKind
from pluto_vsa.protocol_modes.bluetooth.model import (
    BluetoothAnalysisProfile,
    analyze_bluetooth_classic_recording,
    analyze_bluetooth_classic_recordings,
)
import pluto_vsa.protocol_modes.bluetooth.ui as bluetooth_ui
from pluto_vsa.protocol_modes.bluetooth.ui import (
    BluetoothAnalyzerWindow,
    format_air_bits,
    infer_le_channel,
)
from pluto_common.sdr.trigger import TriggerKind, TriggerSlope
from pluto_vsa.session import VSASession
from pluto_vsa.sources import FileIQSource
from pluto_vsa.ui.measurement_config_dialog import HierarchicalMeasConfigDialog
from pluto_vsa.ui.display_processing import physical_constellation_display_symbols
from pluto_vsa.ui.measurement_chrome import SymbolDensitySpread
from pluto_vsg.engine import BluetoothBRWaveformEngine
from pluto_vsg.model import BluetoothPacketKind
from pluto_vsg.profiles import bluetooth_br_edr_project, bluetooth_br_fields
from _bluetooth_dedicated_test_helpers import _session_with_le_bits


def test_bluetooth_dedicated_exposes_common_iq_export_action(tmp_path) -> None:
    pg.mkQApp("Bluetooth dedicated IQ export action test")
    recording = IQRecording(
        iq=np.ones(128, dtype=np.complex64),
        sample_rate_hz=8_000_000.0,
        center_frequency_hz=2_440_000_000.0,
    )
    window = BluetoothAnalyzerWindow(
        preferences=QtCore.QSettings(
            str(tmp_path / "bluetooth-export.ini"),
            QtCore.QSettings.Format.IniFormat,
        )
    )
    try:
        assert not window.export_iq_action.isEnabled()
        window.stage_session(VSASession(recording=recording))
        assert window.export_iq_action.isEnabled()
        assert window.export_iq_action.text() == "Export IQ Recording..."
    finally:
        window._meas_config_dialog.close()
        window.close()
        window.deleteLater()


def test_bluetooth_workspace_opens_iq_file_directly(
    tmp_path, monkeypatch
) -> None:
    pg.mkQApp("Bluetooth dedicated IQ file test")
    iq_path = Path(__file__).resolve().parents[2] / "data" / "fixtures" / "bluetooth" / "hdt" / "RT_HDT7_5.npz"
    preferences = QtCore.QSettings(
        str(tmp_path / "bluetooth-open-iq.ini"),
        QtCore.QSettings.Format.IniFormat,
    )
    window = BluetoothAnalyzerWindow(preferences=preferences)
    refresh_calls: list[bool] = []
    try:
        monkeypatch.setattr(
            QtWidgets.QFileDialog,
            "getOpenFileName",
            lambda *_args, **_kwargs: (str(iq_path), ""),
        )
        monkeypatch.setattr(window, "refresh", lambda: refresh_calls.append(True))
        window._open_iq()
        assert window._recording is not None
        assert window._recording.source == "File: RT_HDT7_5.npz"
        assert window._session is None
        assert window.center_spin.value() == pytest.approx(2440.0)
        assert refresh_calls == [True]
        assert preferences.value("directories/iq", "", type=str) == str(
            iq_path.resolve().parent
        )
    finally:
        window.close()
        window.deleteLater()


def test_bluetooth_workspace_restores_saved_offset_lo_analysis_channel(
    tmp_path, monkeypatch
) -> None:
    pg.mkQApp("Bluetooth dedicated Offset LO IQ file test")
    iq_path = tmp_path / "offset-lo.npz"
    capture = IQRecording(
        iq=np.ones(4096, dtype=np.complex64),
        sample_rate_hz=8_000_000.0,
        center_frequency_hz=2_405_500_000.0,
        usable_bandwidth_hz=8_000_000.0,
        metadata={
            "requested_center_frequency_hz": 2_404_000_000.0,
            "hardware_lo_frequency_hz": 2_405_500_000.0,
            "lo_offset_hz": 1_500_000.0,
            "experimental_lo_offset": True,
            "requested_analysis_bandwidth_hz": 1_500_000.0,
        },
    )
    FileIQSource.save_npz(iq_path, capture)
    window = BluetoothAnalyzerWindow(
        preferences=QtCore.QSettings(
            str(tmp_path / "bluetooth-offset-open.ini"),
            QtCore.QSettings.Format.IniFormat,
        )
    )
    try:
        monkeypatch.setattr(
            QtWidgets.QFileDialog,
            "getOpenFileName",
            lambda *_args, **_kwargs: (str(iq_path), ""),
        )
        monkeypatch.setattr(window, "refresh", lambda: None)
        window._open_iq()

        assert window._capture_recording is not None
        assert window._recording is not None
        assert window._capture_recording.center_frequency_hz == pytest.approx(
            2_405_500_000.0
        )
        assert window._recording.center_frequency_hz == pytest.approx(
            2_404_000_000.0
        )
        assert window._recording.metadata["analysis_channel_applied"] is True
        assert window.center_spin.value() == pytest.approx(2404.0)
    finally:
        window.close()
        window.deleteLater()


def test_bluetooth_workspace_renders_decode_payload_and_air_bits(tmp_path) -> None:
    pg.mkQApp("Bluetooth dedicated VSA test")
    session, _context = _session_with_le_bits()
    preferences = QtCore.QSettings(
        str(tmp_path / "bluetooth-render.ini"), QtCore.QSettings.Format.IniFormat
    )
    window = BluetoothAnalyzerWindow(preferences=preferences)
    try:
        window.profile_combo.setCurrentIndex(1)
        window.protocol_combo.setCurrentIndex(1)
        window.whitening_check.setChecked(True)
        window.set_session(session)
        while window._analysis_thread is not None:
            QtWidgets.QApplication.processEvents()
        assert window.packet_tabs.count() == 5
        assert window.decode_tree.topLevelItemCount() > 0
        assert "Air bits" in window.air_bits_text.toPlainText()
        assert window.summary_table.rowCount() > 0
        summary_labels = {
            window.summary_table.item(row, 0).text()
            for row in range(window.summary_table.rowCount())
        }
        assert "Access Address" not in summary_labels
        assert "Analysis Profile" not in summary_labels
    finally:
        window.close()
        window.deleteLater()


def test_bluetooth_display_helpers_are_deterministic() -> None:
    assert infer_le_channel(2_402e6) == 37
    assert infer_le_channel(2_440e6) == 17
    assert "55" in format_air_bits(np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8))


def test_bluetooth_multi_packet_ui_preserves_tabs_and_tracks_selected_fsk_range(tmp_path) -> None:
    pg.mkQApp("Bluetooth dedicated multi-packet UI test")
    base = bluetooth_br_edr_project()
    settings = replace(
        base.bluetooth_br,
        packet_kind=BluetoothPacketKind.DH1_2,
        payload_length_bytes=54,
    )
    generated = BluetoothBRWaveformEngine().generate(
        replace(base, bluetooth_br=settings, fields=bluetooth_br_fields(settings))
    )
    spacer = np.zeros(256, dtype=np.complex64)
    recording = IQRecording(
        iq=np.concatenate((spacer, generated.iq, spacer, generated.iq, spacer)),
        sample_rate_hz=generated.sample_rate_hz,
        center_frequency_hz=base.center_frequency_hz,
        source="two generated 2-DH1 UI packets",
    )
    results = analyze_bluetooth_classic_recordings(
        recording,
        profile=BluetoothAnalysisProfile.GENERAL_PACKET,
        lap=settings.lap,
        uap=settings.uap,
        clock_6_1=settings.clock_6_1,
        whitening_enabled=settings.whitening_enabled,
        result_length=1024,
    )
    preferences = QtCore.QSettings(
        str(tmp_path / "bluetooth-multi-ui.ini"), QtCore.QSettings.Format.IniFormat
    )
    window = BluetoothAnalyzerWindow(preferences=preferences)
    try:
        window._recording = recording
        window._classic_analysis_ready(results)
        window.modulation_tabs.setCurrentIndex(1)
        window.symbol_tabs.setCurrentIndex(1)
        window._select_result(1)
        assert window.modulation_tabs.currentIndex() == 1
        assert window.symbol_tabs.currentIndex() == 1
        fsk_trace = window.fsk_modulation_plot.listDataItems()[0]
        x_min = float(np.min(fsk_trace.xData))
        x_max = float(np.max(fsk_trace.xData))
        view_min, view_max = window.fsk_modulation_plot.viewRange()[0]
        assert view_min <= x_max and x_min <= view_max
        first_pattern_ms = (
            results[0].metadata["packet_start_sample"]
            / recording.sample_rate_hz
            * 1e3
        )
        assert view_min > first_pattern_ms
        window._set_symbol_density(not window._symbol_density)
        assert window.modulation_tabs.currentIndex() == 1
        assert window.symbol_tabs.currentIndex() == 1
    finally:
        window.close()
        window.deleteLater()


def test_bluetooth_workspace_uses_generic_run_config_and_edr_tabs(
    tmp_path, monkeypatch
) -> None:
    pg.mkQApp("Bluetooth dedicated EDR UI test")
    base = bluetooth_br_edr_project()
    settings = replace(
        base.bluetooth_br,
        packet_kind=BluetoothPacketKind.DH1_2,
        payload_length_bytes=54,
    )
    generated = BluetoothBRWaveformEngine().generate(
        replace(base, bluetooth_br=settings, fields=bluetooth_br_fields(settings))
    )
    result = analyze_bluetooth_classic_recording(
        IQRecording(
            iq=generated.iq,
            sample_rate_hz=generated.sample_rate_hz,
            center_frequency_hz=base.center_frequency_hz,
            source="generated 2-DH1 UI",
        ),
        profile=BluetoothAnalysisProfile.GENERAL_PACKET,
        lap=settings.lap,
        uap=settings.uap,
        clock_6_1=settings.clock_6_1,
        whitening_enabled=settings.whitening_enabled,
        result_length=1024,
    )
    preferences = QtCore.QSettings(
        str(tmp_path / "bluetooth-edr-ui.ini"), QtCore.QSettings.Format.IniFormat
    )
    window = BluetoothAnalyzerWindow(preferences=preferences)
    try:
        assert [action.text() for action in window.menuBar().actions()] == [
            "File",
            "Sweep / Run",
            "Display",
            "Meas Config",
            "Analysis Mode",
        ]
        assert window.run_action.shortcut().toString() == "F6"
        assert window.clear_measurement_history_action.text() == (
            "Clear Measurement History"
        )
        assert window.open_iq_action.text() == "Open IQ..."
        assert window.open_iq_action.shortcut() == QtGui.QKeySequence(
            QtGui.QKeySequence.StandardKey.Open
        )
        window._build_meas_config_dialog()
        assert isinstance(window._meas_config_dialog, HierarchicalMeasConfigDialog)
        assert not hasattr(window, "derived_modulation")
        assert not hasattr(window, "derived_symbol_rate")
        window.show()
        window._meas_config_dialog.show()
        window._config_top_buttons["Signal Description"].click()
        QtWidgets.QApplication.processEvents()
        assert window.profile_combo.isVisibleTo(window._meas_config_dialog)
        assert window.protocol_combo.isVisibleTo(window._meas_config_dialog)
        assert window.phy_combo.isVisibleTo(window._meas_config_dialog)
        window._config_top_buttons["Input / Frontend"].click()
        QtWidgets.QApplication.processEvents()
        assert window.center_spin.isVisibleTo(window._meas_config_dialog)
        assert not window.capture_length_spin.isVisibleTo(window._meas_config_dialog)
        assert window.internal_gain_spin.isVisibleTo(window._meas_config_dialog)
        window._config_top_buttons["Signal Capture"].click()
        assert window._common_setup.length.isVisibleTo(window._meas_config_dialog)
        assert window.oversampling_combo.isVisibleTo(window._meas_config_dialog)
        window._config_top_buttons["Trigger"].click()
        QtWidgets.QApplication.processEvents()
        assert window.acquisition_trigger_source_combo.isVisibleTo(
            window._meas_config_dialog
        )
        assert window.iq_power_trigger_check.isVisibleTo(
            window._meas_config_dialog
        )
        window.acquisition_trigger_source_combo.setCurrentIndex(1)
        window.acquisition_trigger_level_spin.setValue(-31.5)
        window.acquisition_trigger_slope_combo.setCurrentIndex(1)
        window.acquisition_trigger_offset_spin.setValue(-12.0)
        capture = window._capture_settings()
        assert capture.trigger_source is TriggerKind.POWER_LEVEL
        assert capture.trigger_slope is TriggerSlope.FALLING
        assert capture.trigger_level_dbm == -31.5
        assert capture.trigger_offset_s == -12e-6
        window._meas_config_dialog.hide()

        def reject_duplicate_fsk_filter(*_args, **_kwargs):
            raise AssertionError(
                "Bluetooth Dedicated must reuse the RF measurement trace"
            )

        monkeypatch.setattr(
            bluetooth_ui,
            "prepare_fsk_display_frequency",
            reject_duplicate_fsk_filter,
        )
        window._recording = result.metadata["analysis_session"].recording
        window._classic_analysis_ready((result,))
        assert window._psk_symbol_plot_mode == "Physical IQ"
        assert window.psk_symbol_plot.getAxis("bottom").labelText == "I"
        assert window.psk_symbol_plot.getAxis("left").labelText == "Q"
        physical_markers = window.psk_modulation_plot.listDataItems()[1]
        physical_marker_iq = np.asarray(physical_markers.xData) + 1j * np.asarray(
            physical_markers.yData
        )
        plotted_physical = window.psk_symbol_plot.listDataItems()[0]
        plotted_physical_iq = np.asarray(plotted_physical.xData) + 1j * np.asarray(
            plotted_physical.yData
        )
        np.testing.assert_allclose(
            plotted_physical_iq,
            physical_constellation_display_symbols(
                ModulationKind.PI4_DQPSK, physical_marker_iq
            ),
            atol=1e-12,
        )
        distance_from_iq_axes = np.minimum(
            np.abs(plotted_physical_iq.real),
            np.abs(plotted_physical_iq.imag),
        )
        assert np.percentile(distance_from_iq_axes, 95) < 0.08
        fsk_trace, fsk_markers = window.fsk_modulation_plot.listDataItems()[:2]
        np.testing.assert_allclose(
            fsk_markers.yData,
            np.interp(fsk_markers.xData, fsk_trace.xData, fsk_trace.yData),
            atol=1e-9,
        )
        measurement_trace = result.metadata["fsk_measurement_trace"]
        expected_fsk_marker_time_ms = (
            measurement_trace.p0_sample
            + (np.arange(fsk_markers.xData.size, dtype=np.float64) + 0.5)
            * measurement_trace.samples_per_symbol
        ) / measurement_trace.sample_rate_hz * 1e3
        np.testing.assert_allclose(
            fsk_markers.xData,
            expected_fsk_marker_time_ms,
            atol=1e-12,
        )
        fsk_symbol_values = window.fsk_symbol_plot.listDataItems()[0].yData
        np.testing.assert_array_equal(fsk_symbol_values, fsk_markers.yData)
        window._set_fsk_symbol_plot_mode("Phase Difference")
        np.testing.assert_allclose(
            window.fsk_modulation_plot.viewRange()[1],
            [-240.0, 240.0],
        )
        assert window.modulation_tabs.isTabVisible(1)
        assert window.modulation_tabs.isTabVisible(2)
        assert window.modulation_tabs.isTabVisible(3)
        assert window.modulation_tabs.tabText(2) == "PSK - Phase Difference"
        assert window.modulation_tabs.tabText(3) == "PSK - DEVM"
        assert window.symbol_tabs.isTabVisible(1)
        assert (
            window.psk_phase_difference_plot.getAxis("bottom").labelText
            == "Time (ms)"
        )
        assert window.psk_devm_plot.getAxis("bottom").labelText == "Time (ms)"
        measurement = result.metadata["rf_measurements"][0]
        block_centers = measurement.arrays[
            "block_physical_symbol_center_samples"
        ]
        block_received = measurement.arrays[
            "block_corrected_received_symbols"
        ]
        block_reference = measurement.arrays["block_reference_symbols"]
        expected_time_ms = (
            block_centers[0, 1:] / window._recording.sample_rate_hz * 1e3
        )
        expected_measured_phase = np.angle(
            block_received[0, 1:] * np.conj(block_received[0, :-1])
        ) / np.pi
        expected_reference_phase = np.angle(
            block_reference[0, 1:] * np.conj(block_reference[0, :-1])
        ) / np.pi
        expected_phase_error = np.angle(
            (
                block_received[0, 1:]
                * np.conj(block_received[0, :-1])
                * np.conj(
                    block_reference[0, 1:]
                    * np.conj(block_reference[0, :-1])
                )
            )
        ) / np.pi
        phase_measured, phase_reference, phase_error = (
            window.psk_phase_difference_plot.listDataItems()[:3]
        )
        np.testing.assert_allclose(
            phase_measured.xData[:50], expected_time_ms
        )
        np.testing.assert_allclose(
            phase_measured.yData[:50], expected_measured_phase
        )
        np.testing.assert_allclose(
            phase_reference.yData[:50], expected_reference_phase
        )
        np.testing.assert_allclose(
            phase_error.yData[:50], expected_phase_error
        )
        np.testing.assert_allclose(
            window.psk_devm_plot.listDataItems()[0].xData[:50],
            expected_time_ms,
        )
        np.testing.assert_allclose(
            window.psk_devm_plot.listDataItems()[0].yData[:50],
            100.0 * measurement.arrays["symbol_devm"][:50],
        )
        assert len(window.spectrum_plot.listDataItems()) == 2
        assert len(window.spectrum_legend.items) == 2
        analysis_recording = window._recording
        assert analysis_recording is not None
        capture_recording = replace(
            analysis_recording,
            center_frequency_hz=analysis_recording.center_frequency_hz
            + 2_000_000.0,
        )
        window._capture_recording = capture_recording
        window.channel_filter_check.setChecked(True)
        window.analysis_spectrum_display_check.setChecked(False)
        window._render(result)
        capture_spectra = window.spectrum_plot.listDataItems()
        assert len(capture_spectra) == 2
        assert all(
            np.mean(trace.xData)
            == pytest.approx(capture_recording.center_frequency_hz / 1e6, abs=0.01)
            for trace in capture_spectra
        )
        window.analysis_spectrum_display_check.setChecked(True)
        window._render(result)
        analysis_spectra = window.spectrum_plot.listDataItems()
        assert len(analysis_spectra) == 2
        assert all(
            np.mean(trace.xData)
            == pytest.approx(analysis_recording.center_frequency_hz / 1e6, abs=0.01)
            for trace in analysis_spectra
        )
        assert {item[1].text for item in window.spectrum_legend.items} == {
            "FSK",
            "PSK",
        }
        window.channel_filter_check.setChecked(False)
        window._capture_recording = None
        window._render(result)
        assert set(window._plot_context_actions) == {
            "iq_power",
            "spectrum",
            "fsk_modulation",
            "psk_modulation",
            "psk_phase_difference",
            "psk_devm",
            "fsk_symbol",
            "psk_symbol",
        }
        assert window.packet_table.rowCount() == 1
        assert window.packet_table.item(0, 2).text() == "2-DH1"
        pending = [
            window.decode_tree.topLevelItem(index)
            for index in range(window.decode_tree.topLevelItemCount())
        ]
        payload_body = None
        while pending:
            item = pending.pop()
            if item.text(0) == "Payload Body":
                payload_body = item
                break
            pending.extend(item.child(index) for index in range(item.childCount()))
        assert payload_body is not None
        assert "\n" not in payload_body.text(1)
        assert " " in payload_body.text(1)
        assert window.decode_tree.textElideMode() is QtCore.Qt.TextElideMode.ElideNone
        psk_trajectory = window.psk_modulation_plot.listDataItems()[0]
        assert psk_trajectory.xData.size > result.vsa_result.measured_symbols.size
        symbol_plot_items_with_overlay = len(window.psk_symbol_plot.listDataItems())
        iq_items_with_overlay = len(window.power_plot.listDataItems())
        window._set_show_symbol_points(False)
        # Match Generic VSA: this option controls synchronized points on the
        # time-domain traces, not the Symbol Plot measurement itself.
        assert len(window.psk_symbol_plot.listDataItems()) == symbol_plot_items_with_overlay
        assert len(window.power_plot.listDataItems()) < iq_items_with_overlay
        window._set_fsk_symbol_plot_mode("Constellation Frequency")
        fsk_view = window.fsk_symbol_plot.getViewBox()
        assert fsk_view.state["mouseEnabled"][0] is False
        # Dedicated and Generic VSA share the same non-semantic horizontal
        # constellation-frequency axis.
        assert np.allclose(window.fsk_symbol_plot.viewRange()[0], (-1.0, 1.0))
    finally:
        window.close()
        window.deleteLater()


def test_bluetooth_config_is_separate_and_restored(tmp_path) -> None:
    pg.mkQApp("Bluetooth dedicated config persistence test")
    settings_path = str(tmp_path / "bluetooth-persistence.ini")
    preferences = QtCore.QSettings(
        settings_path, QtCore.QSettings.Format.IniFormat
    )
    preferences.clear()
    first = BluetoothAnalyzerWindow(preferences=preferences)
    try:
        assert first.center_spin.value() == 2440.0
        assert first.analysis_power_display_check.isChecked()
        assert not first.analysis_spectrum_display_check.isChecked()
        first.center_spin.setValue(2426.0)
        first.protocol_combo.setCurrentIndex(1)
        first.phy_combo.setCurrentText("LE 2M")
        first.channel_spin.setValue(38)
        first._set_symbol_density(True)
        first._set_symbol_density_spread(SymbolDensitySpread.MEDIUM)
        first._set_fsk_symbol_plot_mode("Phase Difference")
        first.channel_filter_check.setChecked(True)
        first.analysis_bandwidth_spin.setValue(1.75)
        first.analysis_power_display_check.setChecked(True)
        first.analysis_spectrum_display_check.setChecked(False)
        first.lo_offset_check.setChecked(True)
        first.lo_offset_spin.setValue(1.6)
        first._save_startup_meas_config()
    finally:
        first.close()
        first.deleteLater()

    restored_preferences = QtCore.QSettings(
        settings_path, QtCore.QSettings.Format.IniFormat
    )
    second = BluetoothAnalyzerWindow(preferences=restored_preferences)
    try:
        assert second.center_spin.value() == 2426.0
        assert second.protocol_combo.currentData() == "bluetooth.le"
        assert second.phy_combo.currentText() == "LE 2M"
        assert second.channel_spin.value() == 38
        assert second._symbol_density is True
        assert second._symbol_density_spread is SymbolDensitySpread.MEDIUM
        assert second.config_density_spread.currentText() == "Medium"
        assert second._fsk_symbol_plot_mode == "Phase Difference"
        assert second.channel_filter_check.isChecked()
        assert second.analysis_bandwidth_spin.value() == 1.75
        assert second.analysis_power_display_check.isChecked()
        assert not second.analysis_spectrum_display_check.isChecked()
        assert second.lo_offset_check.isChecked()
        assert second.lo_offset_spin.value() == 1.6
        capture = second._capture_settings()
        assert capture.analysis_bandwidth_hz == 1_750_000.0
        assert capture.lo_offset_hz == 1_600_000.0
    finally:
        second.close()
        second.deleteLater()


def test_bluetooth_config_accept_does_not_start_analysis(
    tmp_path, monkeypatch
) -> None:
    pg.mkQApp("Bluetooth config close does not analyze")
    refresh_calls: list[bool] = []
    monkeypatch.setattr(
        BluetoothAnalyzerWindow,
        "refresh",
        lambda _self: refresh_calls.append(True),
    )
    preferences = QtCore.QSettings(
        str(tmp_path / "bluetooth-config-close.ini"),
        QtCore.QSettings.Format.IniFormat,
    )
    window = BluetoothAnalyzerWindow(preferences=preferences)
    try:
        window._meas_config_dialog.accept()
        QtWidgets.QApplication.processEvents()
        assert refresh_calls == []
    finally:
        window.close()
        window.deleteLater()


def test_bluetooth_continuous_repeats_until_user_stop(tmp_path, monkeypatch) -> None:
    pg.mkQApp("Bluetooth continuous capture loop test")

    class Source:
        def __init__(self) -> None:
            self.capture_count = 0
            self.buffered: list[bool] = []
            self.stop_count = 0

        def capture_single(
            self, settings, *, cancelled=None, armed=None, prefer_buffered=False
        ):
            self.capture_count += 1
            self.buffered.append(bool(prefer_buffered))
            if armed is not None:
                armed()
            return IQRecording(
                np.ones(256, dtype=np.complex64),
                sample_rate_hz=settings.requested_sample_rate_hz,
                center_frequency_hz=settings.center_frequency_hz,
            )

        def stop_stream(self) -> None:
            self.stop_count += 1

        def close(self) -> None:
            pass

    source = Source()
    window = BluetoothAnalyzerWindow(
        pluto_source=source,
        preferences=QtCore.QSettings(
            str(tmp_path / "bluetooth-continuous.ini"),
            QtCore.QSettings.Format.IniFormat,
        ),
    )

    accepted_recordings: list[object] = []

    def accept_recording(_recording, *, capture_recording=None) -> None:
        accepted_recordings.append(_recording)
        if len(accepted_recordings) == 2:
            window._toggle_continuous_capture()

    monkeypatch.setattr(window, "load_recording", accept_recording)
    try:
        window._toggle_continuous_capture()
        for _index in range(500):
            QtWidgets.QApplication.processEvents()
            thread = window._capture_thread
            if thread is not None:
                thread.wait(10)
            QtCore.QThread.msleep(2)
            if (
                not window._continuous_run_requested
                and window._capture_thread is None
                and window.run_continuous_action.isEnabled()
            ):
                break
        else:
            raise AssertionError("Bluetooth Continuous did not stop")
        assert source.capture_count == 2
        assert len(accepted_recordings) == 2
        assert window._continuous_capture_count == 2
        assert source.buffered == [True, True]
        assert source.stop_count == 1
        assert window.run_action.text() == "Run Single"
        assert window.capture_button.text() == "Single Capture"
    finally:
        window.close()
        window.deleteLater()
