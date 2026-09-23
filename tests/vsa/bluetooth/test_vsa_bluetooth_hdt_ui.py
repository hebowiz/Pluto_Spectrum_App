import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace
import numpy as np
import pyqtgraph as pg
import pytest
from pyqtgraph.Qt import QtCore, QtWidgets
from pluto_protocol.bluetooth.hdt import HDTRate, hdt_definition
from pluto_protocol.model import PacketIssue
from pluto_vsa.model import IQRecording
from pluto_vsa.protocol_modes.bluetooth.model import (
    BluetoothAnalysisProfile,
    analyze_bluetooth_hdt_recording,
    analyze_bluetooth_le_recording,
)
from pluto_vsa.protocol_modes.bluetooth.ui import BluetoothAnalyzerWindow
from pluto_vsa.ui.measurement_chrome import (
    CenteredDedicatedTableDelegate,
    DEDICATED_TABLE_GRID_COLOR,
)
from pluto_vsg.engine import BluetoothLEWaveformEngine
from pluto_vsg.model import BluetoothLEPhy
from pluto_vsg.profiles import bluetooth_le_project
from _bluetooth_dedicated_test_helpers import _hdt_recording


@pytest.mark.parametrize("rate", tuple(HDTRate))
def test_hdt_plots_use_analysis_ranges_and_evm_symbols_for_every_rate(
    rate: HDTRate, tmp_path
) -> None:
    pg.mkQApp(f"Bluetooth dedicated {rate.value} plot test")
    recording, _generated, _project = _hdt_recording(rate, payload_length=73)
    result = analyze_bluetooth_hdt_recording(
        recording, profile=BluetoothAnalysisProfile.RF_PHY_TEST
    )
    window = BluetoothAnalyzerWindow(
        preferences=QtCore.QSettings(
            str(tmp_path / f"bluetooth-{rate.value}-plot.ini"),
            QtCore.QSettings.Format.IniFormat,
        )
    )
    try:
        window._recording = recording
        window._classic_analysis_ready((result,))
        hdt_evm = result.metadata["hdt_evm_result"]
        plot_data = result.metadata["hdt_plot_data"]

        def plotted_symbols(item) -> np.ndarray:
            return np.asarray(item.xData) + 1j * np.asarray(item.yData)

        def pi4_display(values: np.ndarray) -> np.ndarray:
            symbols = np.asarray(values, dtype=np.complex128)
            return symbols * np.exp(
                -1j
                * (np.arange(symbols.size, dtype=np.float64) + 1.0)
                * np.pi
                / 4.0
            )

        payload_is_qpsk = rate in {HDTRate.HDT2, HDTRate.HDT3}
        expected_header_symbols = pi4_display(hdt_evm.header_corrected_symbols)
        expected_payload_symbols = (
            pi4_display(hdt_evm.payload_corrected_symbols)
            if payload_is_qpsk
            else hdt_evm.payload_corrected_symbols
        )
        expected_header_reference = pi4_display(hdt_evm.header_reference_symbols)
        expected_payload_reference = (
            pi4_display(hdt_evm.payload_reference_symbols)
            if payload_is_qpsk
            else hdt_evm.payload_reference_symbols
        )

        header_vector_points = plotted_symbols(
            window.fsk_modulation_plot.listDataItems()[1]
        )
        header_symbol_points = plotted_symbols(
            window.fsk_symbol_plot.listDataItems()[0]
        )
        payload_vector_points = plotted_symbols(
            window.psk_modulation_plot.listDataItems()[1]
        )
        payload_symbol_points = plotted_symbols(
            window.psk_symbol_plot.listDataItems()[0]
        )
        np.testing.assert_array_equal(
            plotted_symbols(window.fsk_modulation_plot.listDataItems()[0]),
            hdt_evm.header_corrected_waveform,
        )
        np.testing.assert_array_equal(
            plotted_symbols(window.psk_modulation_plot.listDataItems()[0]),
            hdt_evm.payload_corrected_waveform,
        )
        np.testing.assert_array_equal(
            header_vector_points, hdt_evm.header_corrected_symbols
        )
        np.testing.assert_allclose(
            header_symbol_points, expected_header_symbols, atol=1e-12
        )
        np.testing.assert_array_equal(
            payload_vector_points, hdt_evm.payload_corrected_symbols
        )
        np.testing.assert_allclose(
            payload_symbol_points, expected_payload_symbols, atol=1e-12
        )

        def unique_constellation_points(values: np.ndarray) -> int:
            points = np.column_stack((values.real, values.imag))
            return np.unique(np.round(points, decimals=6), axis=0).shape[0]

        assert unique_constellation_points(expected_header_reference) == 4
        if payload_is_qpsk:
            assert unique_constellation_points(expected_payload_reference) == 4

        def rms_evm_percent(measured: np.ndarray, reference: np.ndarray) -> float:
            return 100.0 * float(
                np.sqrt(
                    np.sum(np.abs(measured - reference) ** 2)
                    / np.sum(np.abs(reference) ** 2)
                )
            )

        assert rms_evm_percent(
            header_symbol_points, expected_header_reference
        ) == pytest.approx(hdt_evm.header_rms_percent, abs=1e-6)
        assert rms_evm_percent(
            payload_symbol_points, expected_payload_reference
        ) == pytest.approx(hdt_evm.payload_rms_percent, abs=1e-6)

        regions = [
            tuple(item.getRegion())
            for item in window.power_plot.getPlotItem().items
            if isinstance(item, pg.LinearRegionItem)
        ]
        expected_result_ms = tuple(
            sample / recording.sample_rate_hz * 1e3
            for sample in plot_data.payload_evm_sample_range
        )
        expected_training_ms = tuple(
            sample / recording.sample_rate_hz * 1e3
            for sample in plot_data.training_sample_range
        )
        assert len(regions) == 2
        assert any(np.allclose(region, expected_result_ms) for region in regions)
        assert any(np.allclose(region, expected_training_ms) for region in regions)
        payload_label = (
            "QPSK"
            if payload_is_qpsk
            else hdt_definition(rate).modulation
        )
        assert window.modulation_tabs.tabText(0) == "QPSK Header"
        assert window.modulation_tabs.tabText(1) == f"{payload_label} Payload"
        assert window.symbol_tabs.tabText(0) == "QPSK Header"
        assert window.symbol_tabs.tabText(1) == f"{payload_label} Payload"
        legend_labels = {
            item[1].text for item in window.spectrum_legend.items
        }
        assert legend_labels == {"QPSK Header", f"{payload_label} Payload"}
        header_spectrum, payload_spectrum = window.spectrum_plot.listDataItems()
        np.testing.assert_array_equal(
            header_spectrum.xData,
            result.metadata["hdt_header_spectrum_frequency_hz"] / 1e6,
        )
        np.testing.assert_array_equal(
            header_spectrum.yData, result.metadata["hdt_header_spectrum_dbm"]
        )
        np.testing.assert_array_equal(
            payload_spectrum.xData,
            result.metadata["hdt_payload_spectrum_frequency_hz"] / 1e6,
        )
        np.testing.assert_array_equal(
            payload_spectrum.yData, result.metadata["hdt_payload_spectrum_dbm"]
        )
        assert result.metadata["hdt_header_spectrum_sample_range"] == (
            plot_data.control_header_sample_range
        )
        assert result.metadata["hdt_payload_spectrum_sample_range"] == (
            plot_data.payload_sample_range
        )
        expected_power_marker_ms = (
            np.concatenate(
                (
                    hdt_evm.header_symbol_sample_positions,
                    hdt_evm.payload_symbol_sample_positions,
                )
            )
            / recording.sample_rate_hz
            * 1e3
        )
        np.testing.assert_allclose(
            window.power_plot.listDataItems()[1].xData,
            expected_power_marker_ms,
        )
    finally:
        window.close()
        window.deleteLater()


def test_bluetooth_workspace_renders_hdt_header_payload_and_fields(tmp_path) -> None:
    pg.mkQApp("Bluetooth dedicated HDT UI test")
    recording, _generated, _project = _hdt_recording(HDTRate.HDT7_5, 48)
    result = analyze_bluetooth_hdt_recording(
        recording,
        profile=BluetoothAnalysisProfile.RF_PHY_TEST,
    )
    preferences = QtCore.QSettings(
        str(tmp_path / "bluetooth-hdt-render.ini"),
        QtCore.QSettings.Format.IniFormat,
    )
    window = BluetoothAnalyzerWindow(preferences=preferences)
    try:
        window._recording = recording
        window._classic_analysis_ready((result,))
        assert window.modulation_tabs.tabText(0) == "QPSK Header"
        assert window.modulation_tabs.tabText(1) == "16QAM Payload"
        assert window.modulation_tabs.isTabVisible(1)
        assert not window.modulation_tabs.isTabVisible(2)
        assert not window.modulation_tabs.isTabVisible(3)
        assert window.symbol_tabs.isTabVisible(1)
        assert len(window.fsk_symbol_plot.listDataItems()) > 0
        assert len(window.psk_symbol_plot.listDataItems()) > 0
        header_vector_trace = window.fsk_modulation_plot.listDataItems()[0]
        assert header_vector_trace.opts["pen"] is not None
        assert (
            header_vector_trace.xData.size
            > result.metadata["hdt_header_vector_symbols"].size
        )
        assert np.quantile(
            np.hypot(header_vector_trace.xData, header_vector_trace.yData), 0.95
        ) < 1.5
        assert window.decode_tree.topLevelItemCount() == 3
        summary_labels = {
            window.summary_table.item(row, 0).text()
            for row in range(window.summary_table.rowCount())
        }
        assert summary_labels == {
            "RF PHY Measurements",
            "Output power",
            "Control Header RMS EVM",
            "PDU Header and payload RMS EVM",
            "Center frequency deviation",
            "Center frequency offset change between the preamble and the payload",
            "Symbol timing accuracy",
            "Pre-packet emissions",
            "Reference Information",
            "Detected PHY",
            "RF Test Eligibility",
            "Preamble Carrier Frequency Error",
            "Payload Carrier Frequency Error",
            "Control Header Average Power",
            "PDU Header and Payload Average Power",
            "Relative Power (Payload - Header)",
            "Preamble Correlation",
            "RMS EVM Packets Evaluated",
        }
        assert window.summary_table.columnCount() == 4
        assert tuple(
            window.summary_table.horizontalHeaderItem(column).text()
            for column in range(window.summary_table.columnCount())
        ) == ("Test Item", "Value", "Limit", "Result")
        dedicated_tables = (
            window.summary_table,
            window.decode_tree,
            window.packet_table,
            window.issues_table,
        )
        assert all(
            isinstance(table.itemDelegate(), CenteredDedicatedTableDelegate)
            for table in dedicated_tables
        )
        assert all(
            DEDICATED_TABLE_GRID_COLOR in table.styleSheet()
            for table in dedicated_tables
        )
        assert (
            window.summary_table.horizontalHeader().defaultAlignment()
            == QtCore.Qt.AlignmentFlag.AlignCenter
        )
        assert (
            window.decode_tree.header().defaultAlignment()
            == QtCore.Qt.AlignmentFlag.AlignCenter
        )
        assert (
            window.issues_table.horizontalHeader().sectionResizeMode(2)
            == QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        issue_packet = replace(
            result.packet,
            issues=(
                PacketIssue(
                    "truncated_pdu",
                    "PDU declares 173 payload bytes, but the captured packet "
                    "is incomplete and cannot be validated.",
                ),
            ),
        )
        window._render_packet(replace(result, packet=issue_packet))
        QtWidgets.QApplication.processEvents()
        assert (
            window.issues_table.verticalHeader().sectionResizeMode(0)
            == QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        assert window.issues_table.rowHeight(0) > (
            2 * window.issues_table.fontMetrics().height()
        )
        assert (
            window.summary_table.textElideMode()
            == QtCore.Qt.TextElideMode.ElideNone
        )
        assert all(
            window.summary_table.columnWidth(column) >= 40
            for column in range(window.summary_table.columnCount())
        )
        assert sum(
            window.summary_table.columnWidth(column)
            for column in range(window.summary_table.columnCount())
        ) <= window.summary_table.viewport().width() + 1
        summary_rows = {
            window.summary_table.item(row, 0).text(): row
            for row in range(window.summary_table.rowCount())
        }
        header_evm_row = summary_rows["Control Header RMS EVM"]
        payload_evm_row = summary_rows["PDU Header and payload RMS EVM"]
        output_power_row = summary_rows["Output power"]
        measurement_group_row = summary_rows["RF PHY Measurements"]
        assert output_power_row == measurement_group_row + 1
        assert (
            window.summary_table.item(output_power_row, 2).text()
            == "Power Class dependent"
        )
        assert window.summary_table.item(output_power_row, 3).text() == "N/A"
        assert window.summary_table.item(header_evm_row, 2).text() == "≤ -10 dB"
        assert window.summary_table.item(header_evm_row, 3).text() == "PASS"
        assert window.summary_table.item(payload_evm_row, 2).text() == "≤ -22 dB"
        assert window.summary_table.item(payload_evm_row, 3).text() == "PASS"
        timing_row = summary_rows["Symbol timing accuracy"]
        assert window.summary_table.item(timing_row, 1).text() == "N/A"
        assert window.summary_table.item(timing_row, 3).text() == "N/A"
        assert "Payload Length" not in summary_labels
        assert "PDU Control Length" not in summary_labels
        assert "HEC-C" not in summary_labels
        assert "CRC-32" not in summary_labels
        assert window.decode_tree.headerItem().text(2) == "Stream"
        training = window.decode_tree.topLevelItem(0)
        control = window.decode_tree.topLevelItem(1)
        payload = window.decode_tree.topLevelItem(2)
        assert (training.text(2), training.text(3)) == (
            "Training symbols",
            "N/A",
        )
        assert (control.text(2), control.text(3)) == (
            "Control Header",
            "0\N{EN DASH}56",
        )
        assert control.child(2).text(3) == "19"
        assert (payload.text(1), payload.text(2), payload.text(3)) == (
            "\N{EM DASH}",
            "PDU+Payload",
            "0\N{EN DASH}423",
        )
        assert payload.child(0).text(3) == "0\N{EN DASH}7"
        assert payload.child(1).text(0) == "Payload"
        assert payload.child(1).text(3) == "8\N{EN DASH}391"
        assert payload.child(2).text(3) == "392\N{EN DASH}423"
        assert len(window.spectrum_plot.listDataItems()) == 2
        assert len(window.spectrum_legend.items) == 2
        assert {item[1].text for item in window.spectrum_legend.items} == {
            "QPSK Header",
            "16QAM Payload",
        }
        assert (
            window.fsk_modulation_plot.listDataItems()[0].xData.size
            > result.metadata["hdt_header_symbols"].size
        )
        assert result.metadata["hdt_header_symbols"].size == 62
        assert result.metadata["hdt_header_vector_symbols"].size == 62
        hdt_evm = result.metadata["hdt_evm_result"]

        def plotted_symbols(item) -> np.ndarray:
            return np.asarray(item.xData) + 1j * np.asarray(item.yData)

        header_vector_points = plotted_symbols(
            window.fsk_modulation_plot.listDataItems()[1]
        )
        header_symbol_points = plotted_symbols(
            window.fsk_symbol_plot.listDataItems()[0]
        )
        payload_vector_points = plotted_symbols(
            window.psk_modulation_plot.listDataItems()[1]
        )
        payload_symbol_points = plotted_symbols(
            window.psk_symbol_plot.listDataItems()[0]
        )
        header_rotation = np.exp(
            -1j
            * (
                np.arange(hdt_evm.header_corrected_symbols.size) + 1.0
            )
            * np.pi
            / 4.0
        )
        header_reference = hdt_evm.header_reference_symbols * header_rotation
        np.testing.assert_array_equal(
            header_vector_points, hdt_evm.header_corrected_symbols
        )
        np.testing.assert_allclose(
            header_symbol_points,
            hdt_evm.header_corrected_symbols * header_rotation,
            atol=1e-12,
        )
        np.testing.assert_array_equal(
            payload_vector_points, hdt_evm.payload_corrected_symbols
        )
        np.testing.assert_array_equal(payload_symbol_points, payload_vector_points)

        def rms_evm_percent(measured: np.ndarray, reference: np.ndarray) -> float:
            return 100.0 * float(
                np.sqrt(
                    np.sum(np.abs(measured - reference) ** 2)
                    / np.sum(np.abs(reference) ** 2)
                )
            )

        assert rms_evm_percent(
            header_symbol_points, header_reference
        ) == pytest.approx(hdt_evm.header_rms_percent, abs=1e-6)
        assert rms_evm_percent(
            payload_symbol_points, hdt_evm.payload_reference_symbols
        ) == pytest.approx(hdt_evm.payload_rms_percent, abs=1e-6)
        assert window.summary_table.item(header_evm_row, 1).text().startswith(
            f"{hdt_evm.header_rms_percent:.2f} %"
        )
        assert window.summary_table.item(payload_evm_row, 1).text().startswith(
            f"{hdt_evm.payload_rms_percent:.2f} %"
        )
        assert (
            window.power_plot.listDataItems()[0].xData.size
            == recording.sample_count
        )
        assert window.packet_table.item(0, 1).text() == "HDT7.5"
    finally:
        window.close()
        window.deleteLater()


def test_hdt_iq_power_reset_restores_both_axes(tmp_path) -> None:
    pg.mkQApp("Bluetooth HDT IQ power reset regression")
    recording, _, _ = _hdt_recording(HDTRate.HDT3, 32)
    result = analyze_bluetooth_hdt_recording(
        recording,
        profile=BluetoothAnalysisProfile.RF_PHY_TEST,
    )
    window = BluetoothAnalyzerWindow(
        preferences=QtCore.QSettings(
            str(tmp_path / "hdt-reset.ini"), QtCore.QSettings.Format.IniFormat
        )
    )
    try:
        window._recording = recording
        window._classic_analysis_ready((result,))
        expected_x, expected_y = window._analysis_plot_ranges["iq_power"]
        _, displayed_power = window.power_plot.listDataItems()[0].getData()
        assert expected_y[0] == pytest.approx(float(np.max(displayed_power)) - 50.0)
        window.power_plot.setRange(xRange=[0.0, 0.1], yRange=[-2.0, 2.0])
        window._reset_plot_scale("iq_power", window.power_plot)
        actual_x, actual_y = window.power_plot.viewRange()
        np.testing.assert_allclose(actual_x, expected_x)
        np.testing.assert_allclose(actual_y, expected_y)
    finally:
        window.close()
        window.deleteLater()


def test_hdt_to_le_restores_fsk_modulation_axes(tmp_path) -> None:
    pg.mkQApp("Bluetooth HDT to LE plot-axis regression")
    hdt_recording, _, _ = _hdt_recording(HDTRate.HDT3, 32)
    hdt_result = analyze_bluetooth_hdt_recording(
        hdt_recording,
        profile=BluetoothAnalysisProfile.RF_PHY_TEST,
    )
    generated = BluetoothLEWaveformEngine().generate(
        bluetooth_le_project(BluetoothLEPhy.LE_1M)
    )
    le_recording = IQRecording(
        iq=generated.iq,
        sample_rate_hz=generated.sample_rate_hz,
        center_frequency_hz=2_440e6,
    )
    le_result = analyze_bluetooth_le_recording(
        le_recording,
        profile=BluetoothAnalysisProfile.GENERAL_PACKET,
        phy="LE 1M",
        access_address=0x8E89BED6,
        channel_index=37,
        crc_init=0x555555,
        whitening_enabled=True,
        result_length=512,
    )
    window = BluetoothAnalyzerWindow(
        preferences=QtCore.QSettings(
            str(tmp_path / "hdt-to-le-axes.ini"),
            QtCore.QSettings.Format.IniFormat,
        )
    )
    try:
        window._recording = hdt_recording
        window._classic_analysis_ready((hdt_result,))
        assert window.fsk_modulation_plot.getAxis("bottom").labelText == "I"
        assert window.fsk_modulation_plot.getAxis("left").labelText == "Q"
        assert window.fsk_modulation_plot.getViewBox().state["aspectLocked"] == 1.0

        window._recording = le_recording
        window._classic_analysis_ready((le_result,))
        assert (
            window.fsk_modulation_plot.getAxis("bottom").labelText
            == "Time (ms)"
        )
        assert (
            window.fsk_modulation_plot.getAxis("left").labelText
            == "Frequency (kHz)"
        )
        assert window.fsk_modulation_plot.getViewBox().state["aspectLocked"] is False
    finally:
        window.close()
        window.deleteLater()
