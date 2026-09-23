import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace

import pyqtgraph as pg
import pytest
from pyqtgraph.Qt import QtCore, QtWidgets

from pluto_vsg.model import BluetoothPacketKind, BluetoothLEPhy
from pluto_vsg.profiles import (
    bluetooth_br_edr_project, bluetooth_br_fields, bluetooth_le_project,
    bluetooth_hdt_project, dect_project, wifi_project,
)
from pluto_vsg.ui.main_window import PlutoVSGWindow
from pluto_vsa.ui.packet_decode import PacketDecodeTabs


def edr_project(kind):
    base = bluetooth_br_edr_project()
    settings = replace(base.bluetooth_br, packet_kind=kind)
    return replace(base, bluetooth_br=settings, fields=bluetooth_br_fields(settings))


@pytest.mark.parametrize("project,protocol", [
    (bluetooth_br_edr_project(), "bluetooth.br_edr"),
    (edr_project(BluetoothPacketKind.DH1_2), "bluetooth.br_edr"),
    (edr_project(BluetoothPacketKind.DH1_3), "bluetooth.br_edr"),
    (bluetooth_le_project(), "bluetooth.le"),
    (bluetooth_hdt_project(), "bluetooth.hdt"),
    (dect_project(), "dect.classic"),
])
def test_vsg_decodes_only_on_request_and_clears_on_generation(project, protocol, monkeypatch):
    import pluto_vsg.ui.main_window as module
    from pluto_protocol.registry import ProtocolRegistry

    def reject_autodetection(*args, **kwargs):
        raise AssertionError("VSG must select the generated packet's decoder, not auto-probe")

    monkeypatch.setattr(ProtocolRegistry, "probe", reject_autodetection)
    calls = []
    decode = module.analyze_generation_result
    monkeypatch.setattr(module, "analyze_generation_result",
                        lambda result: calls.append(result) or decode(result))
    pg.mkQApp("VSG Packet Decode")
    window = PlutoVSGWindow(project)
    try:
        tabs = window.packet_decode
        assert isinstance(tabs, PacketDecodeTabs)
        assert [tabs.tabText(i) for i in range(tabs.count())] == ["Decode", "Payload Hex", "Issues"]
        assert not calls
        assert tabs.decode_tree.topLevelItemCount() == 0
        assert window.verify_packet_button.isEnabled()
        window.verify_packet_button.click()
        assert calls == [window.result]
        assert window._verified_packet.protocol_id == protocol
        if protocol == "bluetooth.br_edr":
            assert window._verified_packet.packet_type == project.bluetooth_br.packet_kind.value
        elif protocol == "bluetooth.le":
            assert window._verified_packet.phy_name == project.bluetooth_le.phy.value
        elif protocol == "dect.classic":
            assert window._verified_packet.packet_type == project.dect.packet_type.value
        assert tabs.decode_tree.topLevelItemCount() > 0
        assert tabs.payload_text.toPlainText()
        assert tabs.issues_table.rowCount() == len(window._verified_packet.issues)
        window.generate_waveform()
        assert len(calls) == 1
        assert window._verified_packet is None
        assert tabs.decode_tree.topLevelItemCount() == 0
        assert tabs.payload_text.toPlainText() == ""
        assert tabs.issues_table.rowCount() == 0
    finally:
        window.close()


def test_vsg_layout_fonts_and_control_names():
    pg.mkQApp("VSG Packet Decode layout")
    window = PlutoVSGWindow()
    try:
        window.show()
        QtWidgets.QApplication.processEvents()
        assert not hasattr(window, "generate_button")
        assert window.instrument_settings_button.text() == "Device"
        expected = QtWidgets.QApplication.font().pointSizeF()
        for widget in (window.inspector, window.field_table, window.packet_decode,
                       window.field_table.parentWidget(), window.iq_waveform_plot.parentWidget()):
            assert widget.font().pointSizeF() == expected
        inspector_panel = window.inspector.parentWidget().parentWidget()
        decode_panel = window.packet_decode.parentWidget()
        assert inspector_panel.width() == decode_panel.width()
        assert inspector_panel.parentWidget() is decode_panel.parentWidget()
        # The buttons belong to different groups, so compare window coordinates.
        verify_position = window.verify_packet_button.mapTo(window, QtCore.QPoint())
        device_position = window.instrument_settings_button.mapTo(window, QtCore.QPoint())
        assert verify_position.y() < device_position.y()
    finally:
        window.close()


def test_vsg_unsupported_packet_decode_is_disabled():
    pg.mkQApp("VSG unsupported Packet Decode")
    window = PlutoVSGWindow(wifi_project())
    try:
        assert not window.verify_packet_button.isEnabled()
        assert "not available" in window.verify_packet_button.toolTip()
        window._verify_packet()
        assert window.packet_decode.decode_tree.topLevelItemCount() == 0
    finally:
        window.close()


def test_vsa_and_vsg_use_the_same_packet_content_widgets(tmp_path):
    from pyqtgraph.Qt import QtCore
    from pluto_vsa.protocol_modes.bluetooth.ui import BluetoothAnalyzerWindow
    from pluto_vsa.protocol_modes.dect.ui import DectAnalyzerWindow

    pg.mkQApp("Shared packet content views")
    prefs = QtCore.QSettings(str(tmp_path / "shared.ini"), QtCore.QSettings.Format.IniFormat)
    windows = [BluetoothAnalyzerWindow(preferences=prefs), DectAnalyzerWindow(preferences=prefs)]
    try:
        for window in windows:
            assert isinstance(window.packet_tabs, PacketDecodeTabs)
            assert window.decode_tree is window.packet_tabs.decode_tree
    finally:
        for window in windows:
            window.close()


def test_verify_reports_corrupted_generated_packet():
    pg.mkQApp("Corrupted generated packet")
    window = PlutoVSGWindow(bluetooth_le_project())
    try:
        artifact = window.result.packet_bits
        bits = artifact.bits.copy()
        bits[-1] ^= 1
        window.result = replace(window.result, packet_bits=replace(artifact, bits=bits))
        window.verify_packet_button.click()
        assert window._verified_packet.integrity.crc_valid is False
        assert window.packet_decode.issues_table.rowCount() > 0
        messages = [window.packet_decode.issues_table.item(row, 1).text()
                    for row in range(window.packet_decode.issues_table.rowCount())]
        assert any("crc" in code for code in messages)
    finally:
        window.close()


def test_le_2m_verify_uses_the_generated_phy_and_whitening_conditions():
    from pluto_vsg.profiles import bluetooth_le_fields

    pg.mkQApp("LE 2M Packet Decode")
    base = bluetooth_le_project(BluetoothLEPhy.LE_2M)
    settings = replace(base.bluetooth_le, phy=BluetoothLEPhy.LE_2M,
                       whitening_enabled=True, whitening_channel_index=12, crc_init=0x123456)
    project = replace(base, bluetooth_le=settings, fields=bluetooth_le_fields(settings))
    window = PlutoVSGWindow(project)
    try:
        window.verify_packet_button.click()
        assert window._verified_packet.phy_name == settings.phy.value
        assert window._verified_packet.integrity.crc_valid is True
    finally:
        window.close()
