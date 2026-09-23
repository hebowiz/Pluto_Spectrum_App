import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from dataclasses import replace

import numpy as np
import pyqtgraph as pg
import pytest
from pyqtgraph.Qt import QtWidgets

from pluto_protocol import PacketDecodeInput, analyze_packet
from pluto_vsg.packet_import import project_from_packet, packet_export_error
from pluto_vsg.persistence import project_from_dict, project_to_dict
from pluto_vsg.protocol import analyze_generation_result
from pluto_vsg.profiles import (
    bluetooth_br_edr_project, bluetooth_br_fields, bluetooth_le_project,
    bluetooth_le_fields, bluetooth_hdt_project, dect_project, dect_fields,
)
from pluto_vsg.engine import (
    BluetoothBRWaveformEngine, BluetoothLEWaveformEngine,
    BluetoothHDTWaveformEngine, DectWaveformEngine,
)
from pluto_vsg.model import BluetoothPacketKind, BluetoothLEPhy, DectPacketType


def cases():
    br = bluetooth_br_edr_project()
    for kind in BluetoothPacketKind:
        settings = replace(br.bluetooth_br, packet_kind=kind, payload_length_bytes=17, whitening_enabled=True)
        yield replace(br, bluetooth_br=settings, fields=bluetooth_br_fields(settings)), BluetoothBRWaveformEngine
    le = bluetooth_le_project()
    for phy in BluetoothLEPhy:
        settings = replace(le.bluetooth_le, phy=phy, payload_length_bytes=17,
                           preamble_bits="10101010" if phy == BluetoothLEPhy.LE_1M else "1010101010101010")
        yield replace(le, bluetooth_le=settings, sample_rate_hz=8e6 if phy == BluetoothLEPhy.LE_1M else 16e6,
                      fields=bluetooth_le_fields(settings)), BluetoothLEWaveformEngine
    yield bluetooth_hdt_project(), BluetoothHDTWaveformEngine
    dect = dect_project()
    for kind in DectPacketType:
        settings = replace(dect.dect, packet_type=kind)
        yield replace(dect, dect=settings, fields=dect_fields(settings),
                      period_symbols=None if kind in {DectPacketType.P80, DectPacketType.P80Z} else dect.period_symbols), DectWaveformEngine


@pytest.mark.parametrize("original,engine", list(cases()))
def test_export_preserves_exact_packet_bits_and_defaults_unknown_settings(original, engine):
    generated = engine().generate(replace(original, repeat_count=3))
    packet = analyze_generation_result(generated)
    project = project_from_packet(packet)
    assert project.repeat_count == 1
    assert project.period_symbols == (
        960.0 if original.dect and original.dect.packet_type in {DectPacketType.P80, DectPacketType.P80Z}
        else 480.0 if original.dect else None
    )
    assert project.power_envelope == original.power_envelope
    if original.bluetooth_le:
        assert project.bluetooth_le.frequency_deviation_hz == (
            250e3 if original.bluetooth_le.phy == BluetoothLEPhy.LE_1M else 500e3
        )
    assert project.manual_packet_fields
    restored = project_from_dict(project_to_dict(project))
    result = engine().generate(restored)
    np.testing.assert_array_equal(result.packet_bits.bits, generated.packet_bits.bits)
    assert analyze_generation_result(result).integrity.complete


@pytest.mark.parametrize("original,engine", list(cases()))
def test_export_allows_crc_failures(original, engine):
    generated = engine().generate(original)
    artifact = generated.packet_bits
    bits = artifact.bits.copy()
    if original.bluetooth_br:
        # Last payload CRC bit precedes padding and the two EDR trailer symbols.
        bps = 3 if original.bluetooth_br.packet_kind.value.startswith("3-") else 2 if original.bluetooth_br.packet_kind.value.startswith("2-") else 1
        padding = (-((original.bluetooth_br.payload_length_bytes+4)*8)) % bps if bps > 1 else 0
        bits[-1-(2*bps+padding if bps > 1 else 0)] ^= 1
    elif original.dect:
        bits[32+63] ^= 1  # R-CRC
    else:
        bits[-1] ^= 1
    packet = analyze_packet(PacketDecodeInput(
        bits, representation=artifact.representation, protocol_hint=artifact.protocol_id,
        phy_hint=artifact.phy_name, context=artifact.context,
    ))
    assert packet.integrity.complete and packet.integrity.crc_valid is False
    assert packet_export_error(packet) is None
    result = engine().generate(project_from_packet(packet))
    np.testing.assert_array_equal(result.packet_bits.bits, bits)
    assert analyze_generation_result(result).integrity.crc_valid is False


def test_export_rejects_incomplete_packet():
    generated = BluetoothLEWaveformEngine().generate(bluetooth_le_project())
    artifact = generated.packet_bits
    packet = analyze_packet(PacketDecodeInput(
        artifact.bits[:-30], protocol_hint=artifact.protocol_id, phy_hint=artifact.phy_name,
        context=artifact.context,
    ))
    assert packet_export_error(packet)
    with pytest.raises(ValueError):
        project_from_packet(packet)
    assert packet_export_error(None)


def test_export_rejects_undecoded_trailing_data():
    generated = BluetoothLEWaveformEngine().generate(bluetooth_le_project())
    artifact = generated.packet_bits
    packet = analyze_packet(PacketDecodeInput(
        np.concatenate((artifact.bits, np.zeros(8, dtype=np.uint8))),
        protocol_hint=artifact.protocol_id, phy_hint=artifact.phy_name, context=artifact.context,
    ))
    assert packet.integrity.complete
    assert packet_export_error(packet)


def test_manual_length_remains_received_when_payload_changes():
    generated = BluetoothLEWaveformEngine().generate(bluetooth_le_project())
    project = project_from_packet(analyze_generation_result(generated))
    length = project.manual_packet_fields["le_length"]
    settings = replace(project.bluetooth_le, payload_length_bytes=100)
    project = replace(project, bluetooth_le=settings, fields=bluetooth_le_fields(settings))
    result = BluetoothLEWaveformEngine().generate(project)
    assert project.manual_packet_fields["le_length"] == length
    assert result.metadata["pdu_length_bits"].tolist() == [int(b) for b in length]


def test_repeat_button_updates_project_and_is_disabled_during_transmission(monkeypatch):
    from pluto_vsg.ui.main_window import PlutoVSGWindow
    import pluto_vsg.ui.main_window as module
    pg.mkQApp()
    window = PlutoVSGWindow(bluetooth_le_project())
    try:
        assert window.repetitions_button.text() == "Repeat Count\n1"
        assert "Repeat Count" not in dict(window._project_inspector_parameters)
        monkeypatch.setattr(module, "get_deferred_int", lambda *args: (5, True))
        window.repetitions_button.click()
        assert window.project.repeat_count == 5
        assert window.repetitions_button.text() == "Repeat Count\n5"
        assert "Repeat Count" not in dict(window._project_inspector_parameters)
        window._set_pluto_busy(preparing=False, transmitting=True)
        assert not window.repetitions_button.isEnabled()
    finally:
        window.close()


def test_manual_field_editor_auto_and_cancel():
    from pluto_vsg.ui.packet_fields import PacketFieldsDialog
    pg.mkQApp()
    result = BluetoothLEWaveformEngine().generate(bluetooth_le_project())
    project = project_from_packet(analyze_generation_result(result))
    dialog = PacketFieldsDialog(project)
    try:
        dialog.controls["le_crc"][0].setChecked(True)
        dialog._accept_fields()
        assert "le_crc" not in dialog.manual_fields
        assert "le_length" in dialog.manual_fields
        assert "le_crc" in project.manual_packet_fields
    finally:
        dialog.close()


@pytest.mark.parametrize("kind", (BluetoothPacketKind.DH1, BluetoothPacketKind.DH1_2, BluetoothPacketKind.DH1_3))
def test_actual_classic_vsa_packet_exports(kind):
    from pluto_vsa.model import IQRecording
    from pluto_vsa.protocol_modes.bluetooth.model import analyze_bluetooth_classic_recording, BluetoothAnalysisProfile
    base = bluetooth_br_edr_project()
    settings = replace(base.bluetooth_br, packet_kind=kind,
                       payload_length_bytes=16 if kind == BluetoothPacketKind.DH1_3 else 17)
    project = replace(base, bluetooth_br=settings, fields=bluetooth_br_fields(settings))
    generated = BluetoothBRWaveformEngine().generate(project)
    recording = IQRecording(generated.iq, generated.sample_rate_hz, 2440e6)
    result = analyze_bluetooth_classic_recording(
        recording, profile=BluetoothAnalysisProfile.GENERAL_PACKET, lap=settings.lap,
        uap=settings.uap, clock_6_1=settings.clock_6_1, whitening_enabled=False,
        _generate_display_products=False,
    )
    restored = project_from_packet(result.packet)
    actual = BluetoothBRWaveformEngine().generate(restored)
    np.testing.assert_array_equal(actual.packet_bits.bits, generated.packet_bits.bits)


def test_actual_le_vsa_packet_exports():
    from pluto_vsa.model import IQRecording
    from pluto_vsa.protocol_modes.bluetooth.model import analyze_bluetooth_le_recording, BluetoothAnalysisProfile
    generated = BluetoothLEWaveformEngine().generate(bluetooth_le_project())
    result = analyze_bluetooth_le_recording(
        IQRecording(generated.iq, generated.sample_rate_hz, 2440e6),
        profile=BluetoothAnalysisProfile.GENERAL_PACKET, phy="LE 1M",
        _generate_display_products=False,
    )
    restored = project_from_packet(result.packet)
    actual = BluetoothLEWaveformEngine().generate(restored)
    np.testing.assert_array_equal(actual.packet_bits.bits, generated.packet_bits.bits)


def test_actual_dect_vsa_packet_exports():
    from pluto_vsa.model import IQRecording
    from pluto_vsa.protocol_modes.dect import analyze_dect_recording
    original = dect_project()
    generated = DectWaveformEngine().generate(original)
    result = analyze_dect_recording(
        IQRecording(generated.iq, generated.sample_rate_hz, original.center_frequency_hz),
    )[0]
    actual = DectWaveformEngine().generate(project_from_packet(result.packet_analysis))
    np.testing.assert_array_equal(actual.packet_bits.bits, generated.packet_bits.bits)


def test_actual_hdt_vsa_packet_exports():
    from pluto_vsa.model import IQRecording
    from pluto_vsa.protocol_modes.bluetooth.model import analyze_bluetooth_hdt_recording, BluetoothAnalysisProfile
    generated = BluetoothHDTWaveformEngine().generate(bluetooth_hdt_project())
    result = analyze_bluetooth_hdt_recording(
        IQRecording(generated.iq, generated.sample_rate_hz, 2440e6),
        profile=BluetoothAnalysisProfile.RF_PHY_TEST, _include_generic_visualization=False,
    )
    actual = BluetoothHDTWaveformEngine().generate(project_from_packet(result.packet))
    np.testing.assert_array_equal(actual.packet_bits.bits, generated.packet_bits.bits)


def test_capture_rf_settings_are_not_invented_as_transmitter_settings():
    generated = BluetoothLEWaveformEngine().generate(bluetooth_le_project())
    packet = analyze_generation_result(generated)
    packet = replace(packet, source=replace(packet.source, center_frequency_hz=2402e6))
    restored = project_from_packet(packet)
    assert restored.center_frequency_hz == bluetooth_le_project().center_frequency_hz


def test_vsa_file_export_and_structure_based_enablement(tmp_path, monkeypatch):
    from pluto_vsa.ui.packet_export import export_packet_project, update_export_action
    from pyqtgraph.Qt import QtGui
    pg.mkQApp()
    parent = QtWidgets.QMainWindow()
    action = QtGui.QAction(parent)
    generated = BluetoothLEWaveformEngine().generate(bluetooth_le_project())
    packet = analyze_generation_result(generated)
    path = tmp_path / "received.pvsg.json"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *args: (str(path), ""))
    try:
        update_export_action(action, None)
        assert not action.isEnabled()
        update_export_action(action, packet)
        assert action.isEnabled()
        export_packet_project(parent, packet)
        from pluto_vsg.persistence import load_project
        actual = BluetoothLEWaveformEngine().generate(load_project(path))
        np.testing.assert_array_equal(actual.packet_bits.bits, generated.packet_bits.bits)
    finally:
        parent.close()
