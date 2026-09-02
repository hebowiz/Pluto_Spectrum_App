import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace

import numpy as np
import pyqtgraph as pg
import pytest
from pyqtgraph.Qt import QtCore, QtWidgets

from pluto_protocol.bluetooth.hdt import HDTRate
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
from pluto_sa.vsa.protocol_modes.bluetooth.ui import BluetoothAnalyzerWindow, format_air_bits, infer_le_channel
from pluto_sa.sdr.trigger import TriggerKind, TriggerSlope
from pluto_sa.vsa.session import VSASession
from pluto_sa.vsa.ui.measurement_config_dialog import HierarchicalMeasConfigDialog
from pluto_sa.vsa.ui.measurement_chrome import SymbolDensitySpread
from pluto_vsg.engine import (
    BluetoothBRWaveformEngine,
    BluetoothHDTWaveformEngine,
    BluetoothLEWaveformEngine,
)
from pluto_vsg.model import BluetoothLEPhy, BluetoothPacketKind
from pluto_vsg.profiles import (
    bluetooth_br_edr_project,
    bluetooth_br_fields,
    bluetooth_hdt_fields,
    bluetooth_hdt_project,
    bluetooth_le_project,
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


@pytest.mark.parametrize("rate", (HDTRate.HDT4, HDTRate.HDT7_5))
def test_dedicated_hdt_auto_detects_rate_length_and_exact_payload_range(
    rate: HDTRate,
) -> None:
    recording, _generated, project = _hdt_recording(rate, payload_length=73)

    result = analyze_bluetooth_hdt_recording(
        recording,
        profile=BluetoothAnalysisProfile.RF_PHY_TEST,
    )

    expected_payload_symbols = project.fields[-1].symbol_count
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
    field_ids = {field.field_id for field in result.packet.root_fields}
    assert field_ids == {"training", "control_header", "payload"}
    control = next(
        field
        for field in result.packet.root_fields
        if field.field_id == "control_header"
    )
    children = {field.field_id: field for field in control.children}
    assert children["rate_indicator"].meaning.startswith(rate.value)
    assert children["payload_length"].value == 73
    assert children["payload_length"].meaning == "73 byte(s) before channel coding"


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
        profile=BluetoothAnalysisProfile.GENERAL_PACKET,
    )

    assert len(results) == 2
    assert all(result.packet.phy_name == HDTRate.HDT7_5.value for result in results)
    assert all(
        result.metadata["hdt_payload_symbol_count"]
        == project.fields[-1].symbol_count
        for result in results
    )


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
        assert window.modulation_tabs.tabText(0) == "QPSK Header - Vector"
        assert window.modulation_tabs.tabText(1) == "16QAM Payload - Vector"
        assert window.modulation_tabs.isTabVisible(1)
        assert window.symbol_tabs.isTabVisible(1)
        assert len(window.fsk_symbol_plot.listDataItems()) > 0
        assert len(window.psk_symbol_plot.listDataItems()) > 0
        assert window.decode_tree.topLevelItemCount() == 3
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
        window.protocol_combo.setCurrentIndex(1)
        window.whitening_check.setChecked(True)
        window.set_session(session)
        while window._analysis_thread is not None:
            QtWidgets.QApplication.processEvents()
        assert window.packet_tabs.count() == 5
        assert window.decode_tree.topLevelItemCount() > 0
        assert "Air bits" in window.air_bits_text.toPlainText()
        assert window.summary_table.rowCount() > 0
    finally:
        window.close()
        window.deleteLater()


def test_bluetooth_display_helpers_are_deterministic() -> None:
    assert infer_le_channel(2_402e6) == 37
    assert infer_le_channel(2_440e6) == 17
    assert "55" in format_air_bits(np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8))


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
