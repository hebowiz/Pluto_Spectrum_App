import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import pytest
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from pluto_protocol.bluetooth.hdt import HDTRate, hdt_definition
from pluto_sa.vsa.model import IQRecording, ModulationKind, SignalDescription, VSAAnalysisResult
from pluto_sa.vsa.pattern import IQPowerTriggerSettings, MeasurementFilterMode
from pluto_sa.vsa.protocol_modes.bluetooth.model import (
    BluetoothAnalysisProfile,
    analyze_bluetooth_classic_recording,
    analyze_bluetooth_classic_recordings,
    analyze_bluetooth_hdt_recording,
    analyze_bluetooth_hdt_recordings,
    analyze_bluetooth_le_recording,
    analyze_bluetooth_le_recordings,
    analyze_bluetooth_session,
)
import pluto_sa.vsa.protocol_modes.bluetooth.model as bluetooth_model
from pluto_sa.vsa.profiles.bluetooth_br import access_code_bits
from pluto_sa.vsa.protocol_modes.bluetooth.rf_measurement import (
    BluetoothRFTestAccumulator,
)
from pluto_sa.vsa.protocol_modes.bluetooth.ui import BluetoothAnalyzerWindow, format_air_bits, infer_le_channel
from pluto_sa.sdr.trigger import TriggerKind, TriggerSlope
from pluto_sa.vsa.session import VSASession
from pluto_sa.vsa.sources import FileIQSource
from pluto_sa.vsa.ui.measurement_config_dialog import HierarchicalMeasConfigDialog
from pluto_sa.vsa.ui.measurement_chrome import (
    CenteredDedicatedTableDelegate,
    DEDICATED_TABLE_GRID_COLOR,
    SymbolDensitySpread,
)
from pluto_vsg.engine import (
    BluetoothBRWaveformEngine,
    BluetoothHDTWaveformEngine,
    BluetoothLEWaveformEngine,
)
from pluto_vsg.model import BluetoothLEPhy, BluetoothPacketKind
from pluto_vsg.model import PayloadSourceKind
from pluto_vsg.profiles import (
    bluetooth_br_edr_project,
    bluetooth_br_fields,
    bluetooth_hdt_fields,
    bluetooth_hdt_project,
    bluetooth_le_project,
    bluetooth_le_test_project,
)


def _hdt_recording(rate: HDTRate, payload_length: int = 64):
    base = bluetooth_hdt_project(rate)
    settings = replace(
        base.bluetooth_hdt, payload_length_bytes=payload_length
    )
    project = replace(
        base,
        bluetooth_hdt=settings,
        fields=bluetooth_hdt_fields(settings),
    )
    generated = BluetoothHDTWaveformEngine().generate(project)
    return IQRecording(
        iq=generated.iq,
        sample_rate_hz=generated.sample_rate_hz,
        center_frequency_hz=project.center_frequency_hz,
        source=f"generated {rate.value}",
    ), generated, project


def _session_with_le_bits() -> tuple[VSASession, dict[str, object]]:
    generated = BluetoothLEWaveformEngine().generate(bluetooth_le_project(BluetoothLEPhy.LE_1M))
    artifact = generated.packet_bits
    assert artifact is not None
    count = generated.iq.size
    time_s = np.arange(count, dtype=np.float64) / generated.sample_rate_hz
    spectrum_frequency_hz = np.linspace(-2e6, 2e6, 256)
    result = VSAAnalysisResult(
        time_s=time_s,
        iq=generated.iq,
        power_dbfs=np.full(count, -12.0),
        power_dbm=np.full(count, -22.0),
        spectrum_frequency_hz=spectrum_frequency_hz,
        spectrum_dbfs=np.full(256, -70.0),
        spectrum_dbm=np.full(256, -80.0),
        instantaneous_frequency_hz=np.zeros(count),
        symbol_time_s=np.arange(artifact.bits.size) / 1e6,
        measured_symbols=np.ones(artifact.bits.size, dtype=np.complex64),
        reference_symbols=np.ones(artifact.bits.size, dtype=np.complex64),
        decoded_symbols=artifact.bits.astype(np.int16),
        decoded_bits=artifact.bits,
        evm_rms_percent=1.0,
        frequency_error_hz=2_500.0,
        metadata={"symbol_rate_error_ppm": 1.25},
    )
    session = VSASession(
        recording=IQRecording(
            iq=generated.iq,
            sample_rate_hz=generated.sample_rate_hz,
            center_frequency_hz=2_402e6,
            source="test",
        ),
        signal=SignalDescription(ModulationKind.FSK, 1e6, 250e3),
        result=result,
    )
    return session, dict(artifact.context)


def test_dedicated_model_combines_rf_metrics_and_shared_le_decode() -> None:
    session, context = _session_with_le_bits()
    result = analyze_bluetooth_session(
        session,
        profile=BluetoothAnalysisProfile.GENERAL_PACKET,
        protocol_id="bluetooth.le",
        phy_name="LE 1M",
        context=context,
    )
    assert result.packet.protocol_id == "bluetooth.le"
    assert result.packet.integrity.crc_valid is True
    assert any(metric.metric_id == "packet_power" for metric in result.metrics)


@pytest.mark.parametrize("rate", tuple(HDTRate))
def test_dedicated_hdt_auto_detects_rate_length_and_exact_payload_range(
    rate: HDTRate,
) -> None:
    recording, _generated, project = _hdt_recording(rate, payload_length=73)

    result = analyze_bluetooth_hdt_recording(
        recording,
        profile=BluetoothAnalysisProfile.RF_PHY_TEST,
    )

    expected_payload_symbols = next(
        field.symbol_count
        for field in project.fields
        if field.name == "Coded PDU Header / Payload / CRC"
    )
    metrics = {metric.metric_id: metric.display for metric in result.metrics}
    assert result.packet.protocol_id == "bluetooth.hdt"
    assert result.packet.phy_name == rate.value
    assert result.packet.packet_type == rate.value
    assert result.packet.integrity.complete is True
    assert metrics["payload_length"] == "73 byte(s)"
    assert metrics["automatic_result_range"] == (
        f"{expected_payload_symbols} symbol(s) (automatic)"
    )
    assert result.metadata["hdt_payload_symbol_count"] == expected_payload_symbols
    assert result.metadata["hdt_header_evm_rms_percent"] < 1.0
    assert result.metadata["hdt_payload_evm_rms_percent"] < 2.0
    hdt_evm = result.metadata["hdt_evm_result"]
    samples_per_symbol = recording.sample_rate_hz / 2_000_000.0
    assert hdt_evm.header_corrected_symbols is hdt_evm.header_measured_symbols
    assert hdt_evm.payload_corrected_symbols is hdt_evm.payload_measured_symbols
    assert hdt_evm.header_corrected_waveform.size > (
        hdt_evm.header_corrected_symbols.size
    )
    assert hdt_evm.payload_corrected_waveform.size > (
        hdt_evm.payload_corrected_symbols.size
    )
    np.testing.assert_allclose(
        np.diff(hdt_evm.header_symbol_sample_positions), samples_per_symbol
    )
    np.testing.assert_allclose(
        np.diff(hdt_evm.payload_symbol_sample_positions), samples_per_symbol
    )
    np.testing.assert_array_equal(
        result.metadata["hdt_header_symbols"], hdt_evm.header_corrected_symbols
    )
    np.testing.assert_array_equal(
        result.metadata["hdt_payload_symbols"], hdt_evm.payload_corrected_symbols
    )
    plot_data = result.metadata["hdt_plot_data"]
    assert plot_data.evm is hdt_evm
    assert plot_data.payload_sample_range == (
        result.metadata["hdt_payload_start_sample"],
        result.metadata["hdt_payload_stop_sample"],
    )
    assert plot_data.payload_evm_sample_range == (
        result.metadata["hdt_payload_start_sample"],
        result.metadata["hdt_payload_evm_stop_sample"],
    )
    assert plot_data.packet_sample_range == (
        result.metadata["packet_start_sample"],
        result.metadata["packet_stop_sample"],
    )
    assert [row.metric_id for row in result.summary_rows] == [
        "sig_hdt_output_power",
        "sig_hdt_header_evm_rms",
        "sig_hdt_payload_evm_rms",
        "sig_hdt_center_frequency_deviation",
        "sig_hdt_frequency_offset_change",
        "sig_hdt_symbol_timing_accuracy",
        "sig_hdt_pre_packet_emissions",
        "detected_phy",
        "sig_eligibility",
        "sig_hdt_preamble_carrier_error",
        "sig_hdt_payload_carrier_error",
        "sig_hdt_header_average_power",
        "sig_hdt_payload_average_power",
        "sig_hdt_relative_power",
        "sig_hdt_training_correlation",
        "sig_hdt_evm_packets_evaluated",
    ]
    payload_evm_row = next(
        row
        for row in result.summary_rows
        if row.metric_id == "sig_hdt_payload_evm_rms"
    )
    expected_limit_db = {
        HDTRate.HDT2: -10,
        HDTRate.HDT3: -13,
        HDTRate.HDT4: -16,
        HDTRate.HDT6: -19,
        HDTRate.HDT7_5: -22,
    }[rate]
    assert payload_evm_row.limit == f"≤ {expected_limit_db} dB"
    assert all(
        row.limit == "—" and row.result == "—"
        for row in result.summary_rows
        if row.section == "Reference Information"
    )
    field_ids = {field.field_id for field in result.packet.root_fields}
    assert field_ids == {"training", "control_header", "payload"}
    control = next(
        field
        for field in result.packet.root_fields
        if field.field_id == "control_header"
    )
    children = {field.field_id: field for field in control.children}
    assert children["rate_indicator"].meaning.startswith(rate.value)
    assert children["rate_indicator"].value == (
        f"{rate.value} (0b{result.metadata['hdt_rate_indicator']:03b})"
    )
    assert children["pdu_control"].value == 74
    assert children["hec_c"].status.value == "valid"
    assert result.packet.integrity.hec_valid is True
    assert result.packet.integrity.crc_valid is True


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


@pytest.mark.parametrize("rate", (HDTRate.HDT2, HDTRate.HDT3))
def test_hdt_qpsk_long_packet_uses_1000_symbol_range_and_marks_boundary(
    rate: HDTRate, tmp_path
) -> None:
    pg.mkQApp(f"Bluetooth dedicated {rate.value} long range test")
    recording, _generated, _project = _hdt_recording(rate, payload_length=255)
    result = analyze_bluetooth_hdt_recording(
        recording, profile=BluetoothAnalysisProfile.RF_PHY_TEST
    )
    plot_data = result.metadata["hdt_plot_data"]
    assert plot_data.packet_sample_range[1] > plot_data.payload_evm_sample_range[1]
    window = BluetoothAnalyzerWindow(
        preferences=QtCore.QSettings(
            str(tmp_path / f"bluetooth-{rate.value}-long-range.ini"),
            QtCore.QSettings.Format.IniFormat,
        )
    )
    try:
        window._recording = recording
        window._classic_analysis_ready((result,))
        regions = [
            tuple(item.getRegion())
            for item in window.power_plot.getPlotItem().items
            if isinstance(item, pg.LinearRegionItem)
        ]
        expected_result_ms = tuple(
            sample / recording.sample_rate_hz * 1e3
            for sample in plot_data.payload_evm_sample_range
        )
        assert any(np.allclose(region, expected_result_ms) for region in regions)
        packet_stop_ms = (
            plot_data.packet_sample_range[1] / recording.sample_rate_hz * 1e3
        )
        boundary_lines = [
            item
            for item in window.power_plot.getPlotItem().items
            if isinstance(item, pg.InfiniteLine)
            and np.isclose(float(item.value()), packet_stop_ms)
        ]
        assert len(boundary_lines) == 1
        assert boundary_lines[0].label.format == "Packet End"
        assert boundary_lines[0].pen.style() == QtCore.Qt.PenStyle.SolidLine
        assert boundary_lines[0].pen.width() == 1
        assert window.power_plot.viewRange()[0][1] > packet_stop_ms
    finally:
        window.close()
        window.deleteLater()


def test_dedicated_hdt_returns_every_packet_in_capture() -> None:
    recording, generated, project = _hdt_recording(HDTRate.HDT7_5, 32)
    spacer = np.zeros(128, dtype=np.complex64)
    repeated = replace(
        recording,
        iq=np.concatenate((spacer, generated.iq, spacer, generated.iq, spacer)),
        source="two generated HDT7.5 packets",
    )

    results = analyze_bluetooth_hdt_recordings(
        repeated,
        profile=BluetoothAnalysisProfile.RF_PHY_TEST,
    )

    assert len(results) == 2
    assert all(result.packet.phy_name == HDTRate.HDT7_5.value for result in results)
    expected_payload_symbols = next(
        field.symbol_count
        for field in project.fields
        if field.name == "Coded PDU Header / Payload / CRC"
    )
    assert all(
        result.metadata["hdt_payload_symbol_count"]
        == expected_payload_symbols
        for result in results
    )
    first_plot = results[0].metadata["hdt_plot_data"]
    second_plot = results[1].metadata["hdt_plot_data"]
    assert first_plot.packet_sample_range[1] < second_plot.packet_sample_range[0]
    assert second_plot.packet_sample_range[0] == results[1].metadata[
        "packet_start_sample"
    ]
    assert second_plot.payload_evm_sample_range == (
        results[1].metadata["hdt_payload_start_sample"],
        results[1].metadata["hdt_payload_evm_stop_sample"],
    )
    assert all(
        result.metadata["hdt_rms_evm_aggregate_status"]
        == "MEASURING 2 / 1500"
        for result in results
    )
    for result in results:
        evaluated = next(
            metric
            for metric in result.metrics
            if metric.metric_id == "sig_hdt_evm_packets_evaluated"
        )
        assert evaluated.display == "2 / 1500"


def test_dedicated_hdt_decodes_real_hdt7_5_and_identifies_legacy_crc_init() -> None:
    recording = FileIQSource.load(
        Path(__file__).with_name("fixtures") / "RT_HDT7_5.npz"
    )

    result = analyze_bluetooth_hdt_recording(
        recording,
        profile=BluetoothAnalysisProfile.RF_PHY_TEST,
    )

    assert result.packet.phy_name == HDTRate.HDT7_5.value
    assert result.packet.integrity.hec_valid is True
    assert result.packet.integrity.crc_valid is False
    assert result.metadata["hdt_pca_a"] == 0x9F15
    assert result.metadata["hdt_nesn"] == 1
    assert result.metadata["hdt_packet_format_indicator"] == 0
    assert result.metadata["hdt_rate_indicator"] == 0b101
    assert result.metadata["hdt_pdu_control_octets"] == 510
    assert result.metadata["hdt_received_hec_c"] == 0x13FB5A
    assert result.metadata["hdt_control_path_errors"] == 0
    assert result.metadata["hdt_payload_path_errors"] == 0
    assert result.metadata["hdt_received_crc32"] == 0xBB166D73
    assert result.metadata["hdt_calculated_crc32"] == 0xCDCA2EBD
    assert result.metadata["hdt_legacy_init_crc32_match"] is True
    assert result.metadata["hdt_header_evm_rms_percent"] == pytest.approx(
        5.47, abs=0.25
    )
    assert result.metadata["hdt_payload_evm_rms_percent"] == pytest.approx(
        4.01, abs=0.25
    )
    assert result.metadata["hdt_payload_evm_symbol_count"] == 1000
    assert result.metadata["hdt_payload_reference_source"] == (
        "decoded_reencoded_bits"
    )
    assert (
        result.metadata["hdt_preamble_carrier_error_hz"]
        - result.metadata["hdt_payload_carrier_error_hz"]
    ) == pytest.approx(189.0, abs=25.0)
    for key in (
        "hdt_alpha0",
        "hdt_phi0_rad",
        "hdt_delta_omega0_rad_per_symbol",
        "hdt_t0_sample",
        "hdt_phi1_rad",
        "hdt_delta_omega1_rad_per_symbol",
    ):
        assert np.isfinite(result.metadata[key])
    metrics = {metric.label: metric for metric in result.metrics if metric.group}
    assert metrics["Output power"].display.endswith(" dBm")
    assert metrics["Output power"].limit == "Power Class dependent"
    assert metrics["Output power"].result == "N/A"
    assert result.metadata["hdt_output_power_measurement_status"] == "provisional"
    assert result.metadata["hdt_output_power_window_start_sample"] < (
        result.metadata["hdt_output_power_window_stop_sample"]
    )
    header_trajectory = np.asarray(result.metadata["hdt_header_trajectory"])
    assert header_trajectory.size > result.metadata["hdt_header_symbols"].size
    assert np.all(np.isfinite(header_trajectory))
    assert np.quantile(np.abs(header_trajectory), 0.95) < 1.5
    assert metrics["Control Header RMS EVM"].result == "PASS"
    assert metrics["Control Header RMS EVM"].limit == "≤ -10 dB"
    assert metrics["PDU Header and payload RMS EVM"].result == "PASS"
    assert metrics["PDU Header and payload RMS EVM"].limit == "≤ -22 dB"
    assert metrics["Center frequency deviation"].display == "+14.103 kHz"
    assert metrics["Center frequency deviation"].result == "PASS"
    assert metrics[
        "Center frequency offset change between the preamble and the payload"
    ].display == "0.189 kHz"
    assert metrics["Symbol timing accuracy"].result == "N/A"
    assert metrics["Pre-packet emissions"].result == "N/A"


@pytest.mark.parametrize(
    ("filename", "whitening", "expected_phy", "expected_packet", "expected_start"),
    (
        ("DH1_test.npz", False, "BR", "DH1", 96),
        ("bluetooth_br_prbs9_pluto_16msps.npz", False, "BR", "DH1", 15206),
        ("bluetooth_2dh1_prbs9_16msps.npz", True, "EDR 2M", "2-DH1", 32001),
        ("bluetooth_3dh1_prbs9_16msps.npz", True, "EDR 3M", "3-DH1", 32001),
        ("PLUTO_VSG_SMCV100B_2DH1.npz", False, "EDR 2M", "2-DH1", 2074),
    ),
)
def test_real_classic_fixtures_preserve_sync_decode_and_symbol_products(
    filename: str,
    whitening: bool,
    expected_phy: str,
    expected_packet: str,
    expected_start: int,
) -> None:
    recording = FileIQSource.load(Path(__file__).with_name("fixtures") / filename)
    result = analyze_bluetooth_classic_recording(
        recording,
        profile=BluetoothAnalysisProfile.RF_PHY_TEST,
        lap=0xC6967E,
        uap=0x6B,
        clock_6_1=0x2B,
        whitening_enabled=whitening,
        result_length=4096,
    )
    pattern = result.metadata["br_analysis_session"].pattern_result
    assert pattern is not None
    assert pattern.correlation > 0.98
    assert pattern.pattern_start_sample == pytest.approx(expected_start, abs=2)
    assert pattern.result_start_sample == pattern.pattern_start_sample
    assert pattern.metadata["eligible_match_count"] >= 1
    assert pattern.decoded_bits.size == 126
    np.testing.assert_array_equal(pattern.decoded_bits[:72], access_code_bits(0xC6967E))
    first_center = pattern.symbol_time_s[0] * recording.sample_rate_hz
    samples_per_symbol = recording.sample_rate_hz / 1_000_000.0
    assert first_center == pytest.approx(
        pattern.pattern_start_sample + 0.5 * samples_per_symbol, abs=1.0
    )
    assert result.packet.phy_name == expected_phy
    assert result.packet.packet_type == expected_packet
    assert result.packet.integrity.crc_valid is True
    assert result.packet.integrity.complete is True
    assert result.vsa_result.measured_symbols.size > 0
    if expected_phy.startswith("EDR"):
        assert result.metadata["analysis_session"].pattern_result.decoded_symbols.size > 10
        measurement = result.metadata["rf_measurements"][0]
        assert measurement.metrics["rms_devm_worst"] < 0.10
        assert measurement.metrics["peak_devm_worst"] < 0.15
        assert measurement.metrics["omega0_abs_worst_hz"] < 10_000.0


@pytest.mark.parametrize("filename", ("LE1M_FSK_error_raw.npz", "LE1M_FSK_error.npz"))
def test_real_le_rf_test_fixtures_preserve_sync_and_symbol_products(filename: str) -> None:
    recording = FileIQSource.load(Path(__file__).with_name("fixtures") / filename)
    result = analyze_bluetooth_le_recording(
        recording,
        profile=BluetoothAnalysisProfile.RF_PHY_TEST,
        phy="LE 1M",
        access_address=0x71764129,
        channel_index=18,
        crc_init=0x555555,
        whitening_enabled=False,
        result_length=4096,
    )
    pattern = result.metadata["analysis_session"].pattern_result
    expected_access = np.unpackbits(
        np.frombuffer((0x71764129).to_bytes(4, "little"), dtype=np.uint8),
        bitorder="little",
    )
    expected_sync = np.concatenate(
        (np.resize(np.asarray([1, 0], dtype=np.uint8), 8), expected_access)
    )
    assert pattern is not None
    assert pattern.correlation > 0.99
    assert pattern.pattern_start_sample == pytest.approx(106, abs=2)
    assert pattern.result_start_sample == pattern.pattern_start_sample
    assert pattern.metadata["eligible_match_count"] >= 1
    np.testing.assert_array_equal(pattern.decoded_bits[:40], expected_sync)
    assert pattern.decoded_bits.size >= 408
    assert result.packet.phy_name == "LE 1M"
    assert result.vsa_result.measured_symbols.size > 0


def test_sync_is_independent_of_rf_measurement_profile_and_failure(monkeypatch) -> None:
    recording = FileIQSource.load(
        Path(__file__).with_name("fixtures") / "bluetooth_br_prbs9_pluto_16msps.npz"
    )
    options = {
        "lap": 0xC6967E,
        "uap": 0x6B,
        "clock_6_1": 0x2B,
        "whitening_enabled": False,
        "result_length": 4096,
    }
    general = analyze_bluetooth_classic_recording(
        recording, profile=BluetoothAnalysisProfile.GENERAL_PACKET, **options
    )
    general_pattern = general.metadata["br_analysis_session"].pattern_result

    def fail_measurement(*_args, **_kwargs):
        raise RuntimeError("injected RF measurement failure")

    monkeypatch.setattr(bluetooth_model, "build_fm_measurement_trace", fail_measurement)
    rf_test = analyze_bluetooth_classic_recording(
        recording, profile=BluetoothAnalysisProfile.RF_PHY_TEST, **options
    )
    rf_pattern = rf_test.metadata["br_analysis_session"].pattern_result
    assert rf_pattern.pattern_start_sample == general_pattern.pattern_start_sample
    assert rf_pattern.result_start_sample == general_pattern.result_start_sample
    np.testing.assert_array_equal(rf_pattern.decoded_bits, general_pattern.decoded_bits)
    assert rf_test.packet.packet_type == general.packet.packet_type == "DH1"
    assert rf_test.packet.integrity.crc_valid is True
    measurement = rf_test.metadata["rf_measurements"][0]
    assert measurement.eligibility.eligible is False
    assert "injected RF measurement failure" in measurement.metadata["reason"]


def test_bluetooth_workspace_opens_iq_file_directly(
    tmp_path, monkeypatch
) -> None:
    pg.mkQApp("Bluetooth dedicated IQ file test")
    iq_path = Path(__file__).with_name("fixtures") / "RT_HDT7_5.npz"
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
        QtWidgets.QApplication.processEvents()
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


def test_le_rf_profile_uses_test_sync_word_and_general_preserves_user_config(
    tmp_path,
) -> None:
    pg.mkQApp("Bluetooth LE profile config test")
    preferences = QtCore.QSettings(
        str(tmp_path / "bluetooth-le-profile.ini"),
        QtCore.QSettings.Format.IniFormat,
    )
    window = BluetoothAnalyzerWindow(preferences=preferences)
    try:
        window.protocol_combo.setCurrentIndex(
            window.protocol_combo.findData("bluetooth.le")
        )
        assert window.profile_combo.currentData() == BluetoothAnalysisProfile.RF_PHY_TEST
        assert window.access_address_edit.text() == "71764129"
        assert window.crc_init_edit.text() == "555555"
        assert window.whitening_check.isChecked() is False
        assert window.access_address_edit.isEnabled() is False
        assert window.crc_init_edit.isEnabled() is False
        assert window.whitening_check.isEnabled() is False
        rf_options = window._le_options()
        assert rf_options["access_address"] == 0x71764129
        assert rf_options["crc_init"] == 0x555555
        assert rf_options["whitening_enabled"] is False

        window.profile_combo.setCurrentIndex(
            window.profile_combo.findData(BluetoothAnalysisProfile.GENERAL_PACKET)
        )
        assert window.access_address_edit.isEnabled() is True
        assert window.crc_init_edit.isEnabled() is True
        assert window.whitening_check.isEnabled() is True
        window.access_address_edit.setText("8E89BED6")
        window.crc_init_edit.setText("123456")
        window.whitening_check.setChecked(True)
        general_options = window._le_options()
        assert general_options["access_address"] == 0x8E89BED6
        assert general_options["crc_init"] == 0x123456
        assert general_options["whitening_enabled"] is True
    finally:
        window.close()
        window.deleteLater()


def test_dedicated_le_analyzer_synchronizes_generated_iq_and_shared_crc() -> None:
    generated = BluetoothLEWaveformEngine().generate(bluetooth_le_project(BluetoothLEPhy.LE_1M))
    recording = IQRecording(
        iq=generated.iq,
        sample_rate_hz=generated.sample_rate_hz,
        center_frequency_hz=2_440e6,
        source="generated LE packet",
    )
    result = analyze_bluetooth_le_recording(
        recording,
        profile=BluetoothAnalysisProfile.GENERAL_PACKET,
        phy="LE 1M",
        access_address=0x8E89BED6,
        channel_index=37,
        crc_init=0x555555,
        whitening_enabled=True,
        result_length=512,
    )
    assert result.packet.protocol_id == "bluetooth.le"
    assert result.packet.integrity.crc_valid is True
    assert result.packet.raw_bits.size == generated.packet_bits.bits.size


@pytest.mark.parametrize(
    ("phy", "expected_deviation_hz"),
    ((BluetoothLEPhy.LE_1M, 250_000.0), (BluetoothLEPhy.LE_2M, 500_000.0)),
)
def test_le_rf_test_packet_produces_eligible_raw_sig_measurements(
    phy: BluetoothLEPhy, expected_deviation_hz: float
) -> None:
    project = bluetooth_le_test_project(phy)
    generated = BluetoothLEWaveformEngine().generate(project)
    result = analyze_bluetooth_le_recording(
        IQRecording(
            iq=generated.iq,
            sample_rate_hz=generated.sample_rate_hz,
            center_frequency_hz=project.center_frequency_hz,
            source=f"generated {phy.value} RF test packet",
        ),
        profile=BluetoothAnalysisProfile.RF_PHY_TEST,
        phy=phy.value,
        access_address=0x71764129,
        channel_index=0,
        crc_init=0x555555,
        whitening_enabled=False,
        result_length=512,
    )

    measurement = result.metadata["rf_measurements"][0]
    assert measurement.eligibility.eligible is True
    assert measurement.metadata["payload_pattern"] == "10101010"
    assert measurement.metrics["delta_f2_avg_hz"] is not None
    assert result.metadata["analysis_session"].signal.frequency_deviation_hz == (
        expected_deviation_hz
    )
    assert [row.metric_id for row in result.summary_rows] == [
        "output_power",
        "delta_f1_avg",
        "delta_f2_p999",
        "delta_f2_ratio",
        "initial_carrier_frequency",
        "carrier_frequency_drift",
        "carrier_frequency_drift_rate",
        "detected_phy",
        "rf_test_eligibility",
        "payload_pattern",
        "sync_correlation",
        "packets_evaluated",
        "peak_power",
        "mean_abs_fsk_deviation",
        "p999_fsk_deviation",
        "max_fsk_deviation",
    ]
    summary = {row.metric_id: row for row in result.summary_rows}
    expected_f1_limit = {
        BluetoothLEPhy.LE_1M: "225 kHz ≤ Δf1avg ≤ 275 kHz",
        BluetoothLEPhy.LE_2M: "450 kHz ≤ Δf1avg ≤ 550 kHz",
    }[phy]
    assert summary["delta_f1_avg"].limit == expected_f1_limit
    assert summary["delta_f1_avg"].result == "MEASURING"
    assert summary["delta_f2_p999"].value != "N/A"
    assert summary["delta_f2_p999"].limit == {
        BluetoothLEPhy.LE_1M: "≥ 185 kHz",
        BluetoothLEPhy.LE_2M: "≥ 370 kHz",
    }[phy]
    assert summary["delta_f2_p999"].result == "MEASURING"
    assert summary["delta_f2_ratio"].result == "MEASURING"
    assert summary["initial_carrier_frequency"].limit == "±150 kHz"
    assert summary["carrier_frequency_drift"].limit == "< 50 kHz"
    assert summary["carrier_frequency_drift_rate"].limit == "≤ 20 kHz / 50 µs"
    for metric_id in (
        "mean_abs_fsk_deviation",
        "p999_fsk_deviation",
        "max_fsk_deviation",
    ):
        assert summary[metric_id].value.endswith(" kHz")
        assert summary[metric_id].limit == "—"
        assert summary[metric_id].result == "—"
    assert all(
        row.limit == "—" and row.result == "—"
        for row in result.summary_rows
        if row.section == "Reference Information"
    )


def test_br_rf_test_packet_produces_eligible_raw_sig_measurements() -> None:
    base = bluetooth_br_edr_project()
    settings = replace(
        base.bluetooth_br,
        packet_kind=BluetoothPacketKind.DH1,
        payload_length_bytes=27,
        payload_source=PayloadSourceKind.PATTERN,
        payload_pattern="10101010",
        whitening_enabled=False,
    )
    generated = BluetoothBRWaveformEngine().generate(
        replace(base, bluetooth_br=settings, fields=bluetooth_br_fields(settings))
    )
    result = analyze_bluetooth_classic_recording(
        IQRecording(
            iq=generated.iq,
            sample_rate_hz=generated.sample_rate_hz,
            center_frequency_hz=base.center_frequency_hz,
            source="generated BR RF test packet",
        ),
        profile=BluetoothAnalysisProfile.RF_PHY_TEST,
        lap=settings.lap,
        uap=settings.uap,
        clock_6_1=settings.clock_6_1,
        whitening_enabled=False,
    )

    measurement = result.metadata["rf_measurements"][0]
    assert measurement.eligibility.eligible is True
    assert measurement.metadata["payload_pattern"] == "10101010"
    assert measurement.metrics["delta_f2_avg_hz"] is not None
    assert [row.metric_id for row in result.summary_rows] == [
        "output_power",
        "delta_f1_avg",
        "delta_f2_p999",
        "delta_f2_ratio",
        "initial_carrier_frequency",
        "carrier_frequency_drift",
        "carrier_frequency_drift_rate",
        "detected_phy",
        "rf_test_eligibility",
        "packet_type",
        "payload_pattern",
        "access_code_correlation",
        "packets_evaluated",
        "peak_power",
        "mean_abs_fsk_deviation",
        "p999_fsk_deviation",
        "max_fsk_deviation",
    ]
    summary = {row.metric_id: row for row in result.summary_rows}
    assert summary["delta_f1_avg"].result == "MEASURING"
    assert summary["delta_f2_p999"].value != "N/A"
    assert summary["delta_f2_p999"].limit == "≥ 115 kHz"
    assert summary["delta_f2_p999"].result == "MEASURING"
    assert summary["delta_f2_ratio"].result == "MEASURING"
    assert summary["initial_carrier_frequency"].limit == "±75 kHz"
    assert summary["carrier_frequency_drift_rate"].limit == "≤ 20 kHz / 50 µs"
    for metric_id in (
        "mean_abs_fsk_deviation",
        "p999_fsk_deviation",
        "max_fsk_deviation",
    ):
        assert summary[metric_id].value.endswith(" kHz")
        assert summary[metric_id].limit == "—"
        assert summary[metric_id].result == "—"
    assert all(
        row.limit == "—" and row.result == "—"
        for row in result.summary_rows
        if row.section == "Reference Information"
    )


def test_br_arbitrary_payload_keeps_reference_fsk_deviation_metrics() -> None:
    base = bluetooth_br_edr_project()
    settings = replace(
        base.bluetooth_br,
        packet_kind=BluetoothPacketKind.DH1,
        payload_length_bytes=27,
        payload_source=PayloadSourceKind.PRBS9,
        whitening_enabled=False,
    )
    generated = BluetoothBRWaveformEngine().generate(
        replace(base, bluetooth_br=settings, fields=bluetooth_br_fields(settings))
    )
    result = analyze_bluetooth_classic_recording(
        IQRecording(
            iq=generated.iq,
            sample_rate_hz=generated.sample_rate_hz,
            center_frequency_hz=base.center_frequency_hz,
        ),
        profile=BluetoothAnalysisProfile.RF_PHY_TEST,
        lap=settings.lap,
        uap=settings.uap,
        clock_6_1=settings.clock_6_1,
        whitening_enabled=False,
    )

    measurement = result.metadata["rf_measurements"][0]
    assert measurement.metadata["payload_pattern"] is None
    assert measurement.eligibility.eligible is False
    summary = {row.metric_id: row for row in result.summary_rows}
    for metric_id in (
        "mean_abs_fsk_deviation",
        "p999_fsk_deviation",
        "max_fsk_deviation",
    ):
        assert summary[metric_id].section == "Reference Information"
        assert summary[metric_id].value.endswith(" kHz")
        assert summary[metric_id].limit == "—"
        assert summary[metric_id].result == "—"


def test_le_sig_initial_carrier_tracks_injected_cfo() -> None:
    project = bluetooth_le_test_project(BluetoothLEPhy.LE_1M)
    generated = BluetoothLEWaveformEngine().generate(project)
    injected_cfo_hz = 50_000.0
    axis = np.arange(generated.iq.size, dtype=np.float64)
    shifted = generated.iq * np.exp(
        2j * np.pi * injected_cfo_hz * axis / generated.sample_rate_hz
    )
    result = analyze_bluetooth_le_recording(
        IQRecording(
            iq=shifted,
            sample_rate_hz=generated.sample_rate_hz,
            center_frequency_hz=project.center_frequency_hz,
        ),
        profile=BluetoothAnalysisProfile.RF_PHY_TEST,
        phy="LE 1M",
        access_address=0x71764129,
        channel_index=0,
        crc_init=0x555555,
        whitening_enabled=False,
        result_length=512,
    )

    measurement = result.metadata["rf_measurements"][0]
    assert measurement.metrics["initial_carrier_error_hz"] == pytest.approx(
        injected_cfo_hz, abs=2_000.0
    )


def test_br_sig_deviation_is_not_normalized_to_nominal() -> None:
    measured: list[float] = []
    for deviation_hz in (130_000.0, 180_000.0):
        base = bluetooth_br_edr_project()
        settings = replace(
            base.bluetooth_br,
            packet_kind=BluetoothPacketKind.DH1,
            payload_length_bytes=27,
            payload_source=PayloadSourceKind.PATTERN,
            payload_pattern="10101010",
            whitening_enabled=False,
            frequency_deviation_hz=deviation_hz,
        )
        generated = BluetoothBRWaveformEngine().generate(
            replace(base, bluetooth_br=settings, fields=bluetooth_br_fields(settings))
        )
        result = analyze_bluetooth_classic_recording(
            IQRecording(
                iq=generated.iq,
                sample_rate_hz=generated.sample_rate_hz,
                center_frequency_hz=base.center_frequency_hz,
            ),
            profile=BluetoothAnalysisProfile.RF_PHY_TEST,
            lap=settings.lap,
            uap=settings.uap,
            clock_6_1=settings.clock_6_1,
            whitening_enabled=False,
        )
        measured.append(
            float(
                result.metadata["rf_measurements"][0].metrics[
                    "delta_f2_avg_hz"
                ]
            )
        )

    assert measured[1] / measured[0] == pytest.approx(180.0 / 130.0, rel=0.08)


def test_dedicated_edr_length_crc_and_type_meaning_use_air_bit_order() -> None:
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
            source="generated 2-DH1",
        ),
        profile=BluetoothAnalysisProfile.GENERAL_PACKET,
        lap=settings.lap,
        uap=settings.uap,
        clock_6_1=settings.clock_6_1,
        whitening_enabled=settings.whitening_enabled,
        result_length=1024,
    )
    assert result.packet.packet_type == "2-DH1"
    assert result.packet.integrity.crc_valid is True
    header = next(field for field in result.packet.root_fields if field.field_id == "header")
    type_field = next(field for field in header.children if field.field_id == "type")
    assert type_field.meaning == "2-DH1"
    payload = next(field for field in result.packet.root_fields if field.field_id == "payload")
    payload_header = next(field for field in payload.children if field.field_id == "payload_header")
    length = next(field for field in payload_header.children if field.field_id == "length")
    assert int(length.value) == 54
    edr_session = result.metadata["analysis_session"]
    # 10 sync + (16-bit enhanced header + 54-byte payload + 16-bit CRC) / 2
    # + 2 trailer symbols.  The configured discovery range was 1024 symbols,
    # but the result must end at the decoded packet boundary.
    assert edr_session.pattern_result.decoded_symbols.size == 244
    assert (
        edr_session.pattern_result.result_stop_sample
        - edr_session.pattern_result.result_start_sample
        == 244 * 8
    )
    assert result.vsa_result.iq.size == 244 * 8
    assert result.vsa_result.measured_symbols.size <= 244
    br_session = result.metadata["br_analysis_session"]
    expected_edr_search_start = int(
        br_session.pattern_result.pattern_start_sample
        + round(131.0 * generated.sample_rate_hz / 1_000_000.0)
    )
    search_guard = max(
        round(2.0 * generated.sample_rate_hz / 1_000_000.0),
        # EDR 2M is 2 bits/symbol at the common 1-Msym/s symbol clock.
        round(8.0 * generated.sample_rate_hz / 1_000_000.0),
    )
    assert result.metadata["analysis_sample_offset"] <= expected_edr_search_start
    assert (
        expected_edr_search_start - result.metadata["analysis_sample_offset"]
        <= search_guard
    )
    metrics = {metric.metric_id: metric.display for metric in result.metrics}
    assert metrics["bluetooth_devm_rms"] != "--"
    assert "evm_rms" not in metrics
    assert "differential_symbol_evm_rms" not in metrics
    assert (
        result.metadata["analysis_session"].demodulation.measurement_filter
        is MeasurementFilterMode.AUTO
    )
    assert (
        result.metadata["br_analysis_session"].demodulation.measurement_filter
        is MeasurementFilterMode.NONE
    )


def test_edr_sig_measurement_uses_five_us_guard_and_excludes_trailer() -> None:
    base = bluetooth_br_edr_project()
    settings = replace(
        base.bluetooth_br,
        packet_kind=BluetoothPacketKind.DH3_2,
        payload_length_bytes=300,
        whitening_enabled=False,
    )
    generated = BluetoothBRWaveformEngine().generate(
        replace(base, bluetooth_br=settings, fields=bluetooth_br_fields(settings))
    )
    result = analyze_bluetooth_classic_recording(
        IQRecording(
            iq=generated.iq,
            sample_rate_hz=generated.sample_rate_hz,
            center_frequency_hz=base.center_frequency_hz,
            source="generated 2-DH3 RF test packet",
        ),
        profile=BluetoothAnalysisProfile.RF_PHY_TEST,
        lap=settings.lap,
        uap=settings.uap,
        clock_6_1=settings.clock_6_1,
        whitening_enabled=False,
        result_length=4096,
    )

    measurement = result.metadata["rf_measurements"][0]
    assert measurement.eligibility.eligible is True
    assert measurement.metrics["guard_time_s"] == pytest.approx(5.0e-6, abs=0.2e-6)
    assert measurement.metrics["sync_symbol_errors"] == 0
    assert measurement.metrics["trailer_symbol_errors"] == 0
    assert measurement.metrics["block_count"] > 0
    assert measurement.metadata["trailer_excluded_from_devm"] is True
    assert measurement.metrics["rms_devm_worst"] < 0.05
    assert measurement.metrics["output_power_dbm"] is not None
    assert measurement.metadata["output_power_window_start_sample"] > (
        result.metadata["packet_start_sample"]
    )
    assert measurement.metadata["output_power_window_stop_sample"] < (
        result.metadata["packet_stop_sample"]
    )
    assert measurement.metrics["payload_bit_errors"] == 0
    summary = {row.metric_id: row for row in result.summary_rows}
    assert summary["output_power"].value.endswith(" dBm")
    assert summary["output_power"].limit == "Power Class dependent"
    assert summary["output_power"].result == "N/A"
    assert summary["omega_i"].result == "PASS"
    assert summary["omega_0"].result == "PASS"
    assert summary["omega_i_plus_omega_0"].result == "PASS"
    assert summary["rms_devm"].value != "N/A"
    assert summary["rms_devm"].result == "MEASURING"
    assert summary["p99_devm"].value != "N/A"
    assert summary["p99_devm"].result == "MEASURING"
    assert summary["peak_devm"].value != "N/A"
    assert summary["peak_devm"].result == "MEASURING"
    assert summary["guard_time"].limit == "4.60–5.40 µs"
    assert summary["guard_time"].result == "MEASURING"
    assert summary["differential_phase_encoding"].result == "MEASURING"
    assert summary["synchronization_sequence"].result == "MEASURING"
    assert summary["trailer"].limit == "≤ 1 bit error / 50 packets"
    assert summary["trailer"].result == "MEASURING"

    accumulator = BluetoothRFTestAccumulator()
    for _ in range(100):
        accumulator.add(measurement)
    aggregate = accumulator.aggregate_edr()
    aggregated = replace(
        result,
        metadata={**result.metadata, "rf_capture_aggregate": aggregate},
    )
    aggregated_summary = {row.metric_id: row for row in aggregated.summary_rows}
    assert aggregated_summary["rms_devm"].result == "PASS"
    assert aggregated_summary["p99_devm"].result == "PASS"
    assert aggregated_summary["peak_devm"].result == "PASS"
    assert aggregated_summary["guard_time"].result == "PASS"
    assert aggregated_summary["differential_phase_encoding"].result == "PASS"
    assert aggregated_summary["synchronization_sequence"].result == "PASS"
    assert aggregated_summary["trailer"].result == "PASS"
    assert aggregated_summary["devm_blocks_evaluated"].value == "200 / 200"
    assert aggregated_summary["guard_time_packets_evaluated"].value == "100 / 100"
    assert aggregated_summary["guard_time_valid_packets"].value == "100 / 100"

    bad_metrics = dict(measurement.metrics)
    bad_metrics.update(
        guard_time_s=6.0e-6,
        payload_bit_errors=1,
        sync_symbol_errors=1,
        sync_bit_errors=1,
        trailer_symbol_errors=1,
        trailer_bit_errors=1,
    )
    bad_measurement = replace(measurement, metrics=bad_metrics)
    failing_accumulator = BluetoothRFTestAccumulator()
    for _ in range(6):
        failing_accumulator.add(bad_measurement)
    failing = replace(
        result,
        metadata={
            **result.metadata,
            "rf_capture_aggregate": failing_accumulator.aggregate_edr(),
        },
    )
    failing_summary = {row.metric_id: row for row in failing.summary_rows}
    assert failing_summary["guard_time"].result == "FAIL"
    assert failing_summary["differential_phase_encoding"].result == "FAIL"
    assert failing_summary["synchronization_sequence"].result == "FAIL"
    assert failing_summary["trailer"].result == "FAIL"


def test_edr_sig_devm_retains_symbol_dependent_phase_and_amplitude_error() -> None:
    base = bluetooth_br_edr_project()
    settings = replace(
        base.bluetooth_br,
        packet_kind=BluetoothPacketKind.DH3_2,
        payload_length_bytes=300,
        whitening_enabled=False,
    )
    generated = BluetoothBRWaveformEngine().generate(
        replace(base, bluetooth_br=settings, fields=bluetooth_br_fields(settings))
    )

    def analyze(iq: np.ndarray):
        return analyze_bluetooth_classic_recording(
            IQRecording(
                iq=iq,
                sample_rate_hz=generated.sample_rate_hz,
                center_frequency_hz=base.center_frequency_hz,
            ),
            profile=BluetoothAnalysisProfile.RF_PHY_TEST,
            lap=settings.lap,
            uap=settings.uap,
            clock_6_1=settings.clock_6_1,
            whitening_enabled=False,
            result_length=4096,
        ).metadata["rf_measurements"][0]

    baseline = analyze(generated.iq)
    impaired_iq = np.array(generated.iq, copy=True)
    start = int(generated.metadata["edr_start_sample"])
    stop = int(generated.metadata["data_stop_sample"]) - 2 * 8
    sample_axis = np.arange(stop - start, dtype=np.float64)
    symbol_index = (sample_axis // 8).astype(np.int64)
    amplitude = np.where((symbol_index & 1) == 0, 0.82, 1.0)
    phase_error = 0.16 * np.sin(2.0 * np.pi * sample_axis / (3.0 * 8.0))
    impaired_iq[start:stop] *= amplitude * np.exp(1j * phase_error)
    impaired = analyze(impaired_iq)

    assert impaired.metrics["rms_devm_worst"] > (
        baseline.metrics["rms_devm_worst"] + 0.08
    )
    assert impaired.metrics["peak_devm_worst"] > (
        baseline.metrics["peak_devm_worst"] + 0.12
    )


@pytest.mark.parametrize(
    ("packet_kind", "expected_phy", "expected_result_symbols"),
    (
        # BR: Access + Header + ACL header + 12-byte body + CRC.
        (BluetoothPacketKind.DH1, "BR", 72 + 54 + 8 + 12 * 8 + 16),
        (BluetoothPacketKind.DH3, "BR", 72 + 54 + 16 + 12 * 8 + 16),
        (BluetoothPacketKind.DH5, "BR", 72 + 54 + 16 + 12 * 8 + 16),
        # EDR: Sync + ceil((enhanced header + body + CRC) / modulation width)
        # + two trailer symbols.
        (BluetoothPacketKind.DH1_2, "EDR 2M", 10 + (16 + 12 * 8 + 16) // 2 + 2),
        (BluetoothPacketKind.DH3_2, "EDR 2M", 10 + (16 + 12 * 8 + 16) // 2 + 2),
        (BluetoothPacketKind.DH5_2, "EDR 2M", 10 + (16 + 12 * 8 + 16) // 2 + 2),
        (BluetoothPacketKind.DH1_3, "EDR 3M", 10 + 43 + 2),
        (BluetoothPacketKind.DH3_3, "EDR 3M", 10 + 43 + 2),
        (BluetoothPacketKind.DH5_3, "EDR 3M", 10 + 43 + 2),
    ),
)
def test_classic_type_is_only_a_phy_candidate_and_length_sets_result_range(
    packet_kind: BluetoothPacketKind,
    expected_phy: str,
    expected_result_symbols: int,
) -> None:
    base = bluetooth_br_edr_project()
    settings = replace(
        base.bluetooth_br,
        packet_kind=packet_kind,
        payload_length_bytes=12,
    )
    generated = BluetoothBRWaveformEngine().generate(
        replace(base, bluetooth_br=settings, fields=bluetooth_br_fields(settings))
    )
    result = analyze_bluetooth_classic_recording(
        IQRecording(
            iq=generated.iq,
            sample_rate_hz=generated.sample_rate_hz,
            center_frequency_hz=base.center_frequency_hz,
            source=f"generated {packet_kind.value}",
        ),
        profile=BluetoothAnalysisProfile.GENERAL_PACKET,
        lap=settings.lap,
        uap=settings.uap,
        clock_6_1=settings.clock_6_1,
        whitening_enabled=settings.whitening_enabled,
        result_length=4096,
    )

    metrics = {metric.metric_id: metric.display for metric in result.metrics}
    assert metrics["detected_phy"] == expected_phy
    assert result.packet.packet_type == packet_kind.value
    assert result.packet.integrity.crc_valid is True
    assert result.packet.integrity.complete is True
    assert (
        result.metadata["analysis_session"].pattern_result.decoded_symbols.size
        == expected_result_symbols
    )
    if expected_phy.startswith("EDR"):
        assert [row.metric_id for row in result.summary_rows[:12]] == [
            "output_power",
            "relative_transmit_power",
            "omega_i",
            "omega_0",
            "omega_i_plus_omega_0",
            "rms_devm",
            "p99_devm",
            "peak_devm",
            "guard_time",
            "differential_phase_encoding",
            "synchronization_sequence",
            "trailer",
        ]
        summary = {row.metric_id: row for row in result.summary_rows}
        expected_devm_limits = (
            ("≤ 20 %", "≤ 30 %", "≤ 35 %")
            if expected_phy == "EDR 2M"
            else ("≤ 13 %", "≤ 20 %", "≤ 25 %")
        )
        assert (
            summary["rms_devm"].limit,
            summary["p99_devm"].limit,
            summary["peak_devm"].limit,
        ) == expected_devm_limits
        assert summary["relative_transmit_power"].limit == (
            "-4 dB < value < +1 dB"
        )
        assert summary["omega_i"].limit == "-75 kHz < ωi < +75 kHz"
        assert summary["omega_0"].limit == "-10 kHz < ω0 < +10 kHz"
        assert summary["omega_i_plus_omega_0"].limit == (
            "-75 kHz < value < +75 kHz"
        )
        assert summary["rms_devm"].result == "N/A"
        assert all(
            row.limit == "—" and row.result == "—"
            for row in result.summary_rows
            if row.section == "Reference Information"
        )


def test_br_packet_is_not_promoted_by_a_later_edr_sync() -> None:
    base = bluetooth_br_edr_project()
    br_settings = replace(
        base.bluetooth_br,
        packet_kind=BluetoothPacketKind.DH1,
        payload_length_bytes=12,
    )
    edr_settings = replace(
        base.bluetooth_br,
        packet_kind=BluetoothPacketKind.DH1_2,
        payload_length_bytes=12,
    )
    br = BluetoothBRWaveformEngine().generate(
        replace(base, bluetooth_br=br_settings, fields=bluetooth_br_fields(br_settings))
    )
    edr = BluetoothBRWaveformEngine().generate(
        replace(base, bluetooth_br=edr_settings, fields=bluetooth_br_fields(edr_settings))
    )
    spacer = np.zeros(256, dtype=np.complex64)
    recording = IQRecording(
        iq=np.concatenate((spacer, br.iq, spacer, edr.iq, spacer)),
        sample_rate_hz=br.sample_rate_hz,
        center_frequency_hz=base.center_frequency_hz,
        source="generated DH1 followed by 2-DH1",
    )

    results = analyze_bluetooth_classic_recordings(
        recording,
        profile=BluetoothAnalysisProfile.GENERAL_PACKET,
        lap=br_settings.lap,
        uap=br_settings.uap,
        clock_6_1=br_settings.clock_6_1,
        whitening_enabled=br_settings.whitening_enabled,
        result_length=4096,
    )

    assert [result.packet.packet_type for result in results] == ["DH1", "2-DH1"]
    assert [
        {metric.metric_id: metric.display for metric in result.metrics}["detected_phy"]
        for result in results
    ] == ["BR", "EDR 2M"]
    assert all(result.packet.integrity.crc_valid is True for result in results)


@pytest.mark.parametrize(
    "packet_kind",
    (BluetoothPacketKind.DH3_3, BluetoothPacketKind.DH5_3),
)
def test_3m_edr_phy_detection_tolerates_realistic_sync_boundary_delay(
    packet_kind: BluetoothPacketKind,
) -> None:
    """A guard/ramp timing offset must not turn 3-DHx into BR DHx."""

    base = bluetooth_br_edr_project()
    settings = replace(
        base.bluetooth_br,
        packet_kind=packet_kind,
        payload_length_bytes=12,
    )
    generated = BluetoothBRWaveformEngine().generate(
        replace(base, bluetooth_br=settings, fields=bluetooth_br_fields(settings))
    )
    recording = IQRecording(
        iq=generated.iq,
        sample_rate_hz=generated.sample_rate_hz,
        center_frequency_hz=base.center_frequency_hz,
        source=f"delayed {packet_kind.value}",
    )
    initial = analyze_bluetooth_classic_recording(
        recording,
        profile=BluetoothAnalysisProfile.GENERAL_PACKET,
        lap=settings.lap,
        uap=settings.uap,
        clock_6_1=settings.clock_6_1,
        whitening_enabled=settings.whitening_enabled,
        result_length=4096,
    )
    access_start = int(
        initial.metadata["br_analysis_session"].pattern_result.pattern_start_sample
    )
    switch_boundary = int(
        round(
            access_start
            + 131.0 * generated.sample_rate_hz / 1_000_000.0
        )
    )
    delay_samples = int(round(4.0e-6 * generated.sample_rate_hz))
    delayed = replace(
        recording,
        iq=np.concatenate(
            (
                generated.iq[:switch_boundary],
                np.zeros(delay_samples, dtype=np.complex64),
                generated.iq[switch_boundary:],
            )
        ),
    )

    result = analyze_bluetooth_classic_recording(
        delayed,
        profile=BluetoothAnalysisProfile.GENERAL_PACKET,
        lap=settings.lap,
        uap=settings.uap,
        clock_6_1=settings.clock_6_1,
        whitening_enabled=settings.whitening_enabled,
        result_length=4096,
    )

    assert result.packet.packet_type == packet_kind.value
    assert result.packet.integrity.complete is True
    assert result.packet.integrity.crc_valid is True
    assert result.metadata["edr_candidate_error"] is None


def test_dedicated_edr_multi_packet_analysis_uses_local_ranges_and_reports_relative_power() -> None:
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
        source="two generated 2-DH1 packets",
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
    assert len(results) == 2
    assert all(result.packet.packet_type == "2-DH1" for result in results)
    assert all(result.packet.integrity.crc_valid is True for result in results)
    assert results[1].metadata["recording_sample_offset"] > 0
    assert results[1].metadata["analysis_sample_offset"] > results[0].metadata["analysis_sample_offset"]
    assert results[1].metadata["packet_start_sample"] > results[0].metadata["packet_stop_sample"]
    metrics = {metric.metric_id: metric.display for metric in results[1].metrics}
    assert metrics["fsk_average_power"] != "--"
    assert metrics["psk_average_power"] != "--"
    assert metrics["psk_relative_power"] != "--"
    assert abs(float(metrics["psk_relative_power"].split()[0])) < 0.5
    aggregate = results[1].metadata["rf_capture_aggregate"]
    assert aggregate.metrics["packet_count"] == 2
    assert aggregate.metrics["block_count"] == sum(
        result.metadata["rf_measurements"][0].metrics["block_count"]
        for result in results
    )


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


def test_bluetooth_workspace_uses_generic_run_config_and_edr_tabs(tmp_path) -> None:
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
            "Display Config",
            "Meas Config",
            "Analysis Mode",
        ]
        assert window.run_action.shortcut().toString() == "F6"
        assert window.open_iq_action.text() == "Open IQ..."
        assert window.open_iq_action.shortcut() == QtGui.QKeySequence(
            QtGui.QKeySequence.StandardKey.Open
        )
        window._build_meas_config_dialog()
        assert isinstance(window._meas_config_dialog, HierarchicalMeasConfigDialog)
        assert not window.derived_modulation.isEnabled()
        assert not window.derived_symbol_rate.isEnabled()
        window.show()
        window._meas_config_dialog.show()
        window._config_top_buttons["Bluetooth Analysis"].click()
        QtWidgets.QApplication.processEvents()
        assert window.profile_combo.isVisibleTo(window._meas_config_dialog)
        assert window.protocol_combo.isVisibleTo(window._meas_config_dialog)
        assert window.phy_combo.isVisibleTo(window._meas_config_dialog)
        window._config_top_buttons["Input / Frontend"].click()
        QtWidgets.QApplication.processEvents()
        assert window.center_spin.isVisibleTo(window._meas_config_dialog)
        assert window.capture_length_spin.isVisibleTo(window._meas_config_dialog)
        assert window.internal_gain_spin.isVisibleTo(window._meas_config_dialog)
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
        window._recording = result.metadata["analysis_session"].recording
        window._classic_analysis_ready((result,))
        assert window.modulation_tabs.isTabVisible(1)
        assert window.symbol_tabs.isTabVisible(1)
        assert len(window.spectrum_plot.listDataItems()) == 2
        assert len(window.spectrum_legend.items) == 2
        assert set(window._plot_context_actions) == {
            "iq_power",
            "spectrum",
            "fsk_modulation",
            "psk_modulation",
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
        assert "\n" in payload_body.text(1)
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
        first.center_spin.setValue(2426.0)
        first.protocol_combo.setCurrentIndex(1)
        first.phy_combo.setCurrentText("LE 2M")
        first.channel_spin.setValue(38)
        first._set_symbol_density(True)
        first._set_symbol_density_spread(SymbolDensitySpread.MEDIUM)
        first._set_fsk_symbol_plot_mode("Phase Difference")
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
    finally:
        second.close()
        second.deleteLater()


def test_dedicated_le_analyzer_returns_every_packet_in_capture() -> None:
    generated = BluetoothLEWaveformEngine().generate(bluetooth_le_project(BluetoothLEPhy.LE_1M))
    spacer = np.zeros(128, dtype=np.complex64)
    iq = np.concatenate((generated.iq, spacer, generated.iq))
    results = analyze_bluetooth_le_recordings(
        IQRecording(
            iq=iq,
            sample_rate_hz=generated.sample_rate_hz,
            center_frequency_hz=2_440e6,
            source="two generated LE packets",
        ),
        profile=BluetoothAnalysisProfile.GENERAL_PACKET,
        phy="LE 1M",
        access_address=0x8E89BED6,
        channel_index=37,
        crc_init=0x555555,
        whitening_enabled=True,
        result_length=512,
    )
    assert len(results) == 2
    assert all(result.packet.integrity.crc_valid is True for result in results)


def test_dedicated_le_burst_search_gates_pattern_candidates() -> None:
    generated = BluetoothLEWaveformEngine().generate(
        bluetooth_le_project(BluetoothLEPhy.LE_1M)
    )
    spacer = np.zeros(256, dtype=np.complex64)
    recording = IQRecording(
        iq=np.concatenate((spacer, generated.iq, spacer, generated.iq, spacer)),
        sample_rate_hz=generated.sample_rate_hz,
        center_frequency_hz=2_440e6,
        source="trigger-gated LE packets",
    )
    results = analyze_bluetooth_le_recordings(
        recording,
        profile=BluetoothAnalysisProfile.GENERAL_PACKET,
        phy="LE 1M",
        access_address=0x8E89BED6,
        channel_index=37,
        crc_init=0x555555,
        whitening_enabled=True,
        result_length=512,
        iq_power_trigger=IQPowerTriggerSettings(
            enabled=True,
            level_dbm=-100.0,
            hysteresis_db=3.0,
            dropout_symbols=8.0,
        ),
    )
    assert len(results) == 2
    assert all(result.packet.integrity.crc_valid is True for result in results)
