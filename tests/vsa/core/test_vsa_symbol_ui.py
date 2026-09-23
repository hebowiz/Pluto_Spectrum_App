import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace
from pathlib import Path
import numpy as np
import pyqtgraph as pg
import pytest
from pyqtgraph.Qt import QtWidgets
from pluto_vsa.ui.main_window import (
    VSAWindow,
    _constellation_display_symbols,
    _physical_constellation_display_symbols,
)
from pluto_vsa.model import IQRecording, ModulationKind, SignalDescription
from pluto_vsa.mapping import (
    BLUETOOTH_HDT_MAPPING,
    psk_constellation,
    reverse_symbol_bits,
)
from pluto_vsa.pattern import (
    BitOrdering,
    DemodulationSettings,
    KnownPattern,
    MeasurementFilterMode,
    PatternSearchSettings,
    ResultRangeSettings,
)
from pluto_vsa.session import VSASession
from pluto_vsa.sources import FileIQSource, GeneratedIQSource
from _vsa_ui_test_helpers import _isolated_preferences, _wait_for_background_analysis


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
        Path(__file__).resolve().parents[2] / "data" / "fixtures" / "bluetooth" / "hdt"
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
            Path(__file__).resolve().parents[2] / "data" / "fixtures" / "bluetooth" / "br-edr"
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
