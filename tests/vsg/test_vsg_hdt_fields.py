import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace

import numpy as np
import pyqtgraph as pg
import pytest
from pyqtgraph.Qt import QtWidgets

from pluto_protocol.bluetooth.hdt import HDTRate, hdt_crc24, hdt_crc32
from pluto_vsg.engine import BluetoothHDTWaveformEngine
from pluto_vsg.model import HDTPayloadSourceKind, hdt_is_rf_test_configuration, validate_project
from pluto_vsg.persistence import project_from_dict, project_to_dict
from pluto_vsg.profiles import bluetooth_hdt_fields, bluetooth_hdt_project
from pluto_vsg.protocol import analyze_generation_result
from pluto_vsg.ui.main_window import _BluetoothHDTSettingsDialog


def project_with(**changes):
    base = bluetooth_hdt_project()
    settings = replace(base.bluetooth_hdt, **changes)
    return replace(base, bluetooth_hdt=settings, fields=bluetooth_hdt_fields(settings))


def lsb(bits):
    return sum(int(bit) << i for i, bit in enumerate(bits))


def msb(bits):
    return sum(int(bit) << (bits.size - 1 - i) for i, bit in enumerate(bits))


@pytest.mark.parametrize("rate", tuple(HDTRate))
def test_hdt_generated_fields_and_shared_decode(rate):
    project = project_with(
        rate=rate, payload_length_bytes=29, pca=0x123456789A,
        nesn=6, md=1, sn=5, llid=2, crc_init=0x1234ABCD,
    )
    result = BluetoothHDTWaveformEngine().generate(project)
    control = result.metadata["control_header_bits"]
    pdu = result.metadata["format0_bits"]
    assert lsb(control[:16]) == 0x1234
    assert lsb(control[16:19]) == 6
    assert lsb(control[24:33]) == 30
    assert msb(control[-24:]) == hdt_crc24(control[:33], init=0x56789A)
    assert lsb(pdu[:8]) == (1 << 2) | (5 << 3) | (2 << 6)
    assert msb(pdu[-32:]) == hdt_crc32(pdu[:-32], init=0x1234ABCD)
    decoded = analyze_generation_result(result)
    assert decoded.integrity.hec_valid and decoded.integrity.crc_valid
    header = {f.field_id: f for f in decoded.root_fields[1].children}
    assert header["nesn"].value == 6
    initial = {f.field_id: f for f in decoded.root_fields[2].children[0].children}
    assert [initial[key].value for key in ("md", "sn", "llid")] == [1, 5, 2]
    np.testing.assert_array_equal(decoded.root_fields[2].children[1].raw_bits,
                                  result.metadata["payload_bits"])
    assert not hdt_is_rf_test_configuration(project.bluetooth_hdt)
    assert "Custom Format 0" in result.metadata["packet_name"]
    assert project_from_dict(project_to_dict(project)) == project


def test_manual_integrity_values_are_transmitted_and_reported_invalid():
    project = project_with(hec_auto=False, hec_manual=0, crc_auto=False, crc_manual=0)
    result = BluetoothHDTWaveformEngine().generate(project)
    assert msb(result.metadata["control_header_bits"][-24:]) == 0
    assert msb(result.metadata["format0_bits"][-32:]) == 0
    decoded = analyze_generation_result(result)
    assert decoded.integrity.hec_valid is False
    assert decoded.integrity.crc_valid is False
    assert {i.code for i in decoded.issues} == {"invalid_hec_c", "invalid_crc32"}


def test_legacy_project_without_new_fields_retains_defaults():
    document = project_to_dict(bluetooth_hdt_project())
    for key in ("pca", "nesn", "md", "sn", "llid", "hec_auto", "hec_manual",
                "crc_init", "crc_auto", "crc_manual"):
        del document["project"]["bluetooth_hdt"][key]
    assert project_from_dict(document) == bluetooth_hdt_project()


@pytest.mark.parametrize("name,value", (
    ("pca", 1 << 40), ("pca", -1), ("nesn", 8), ("md", 2),
    ("sn", 8), ("llid", 4), ("hec_manual", 1 << 24),
    ("crc_init", 1 << 32), ("crc_manual", -1),
))
def test_field_ranges_block_invalid_configuration(name, value):
    project = project_with(**{name: value})
    assert any(i.path == f"bluetooth_hdt.{name}" for i in validate_project(project))
    with pytest.raises(ValueError):
        BluetoothHDTWaveformEngine().generate(project)


def test_dialog_fields_are_transactional_and_generate_after_ok():
    pg.mkQApp()
    original = bluetooth_hdt_project()
    dialog = _BluetoothHDTSettingsDialog(original)
    try:
        dialog.pca_edit.setText("123456789A")
        dialog.nesn_combo.setCurrentIndex(dialog.nesn_combo.findData(3))
        dialog.md_combo.setCurrentIndex(dialog.md_combo.findData(1))
        dialog.sn_combo.setCurrentIndex(dialog.sn_combo.findData(7))
        dialog.llid_combo.setCurrentIndex(dialog.llid_combo.findData(3))
        dialog.crc_init_edit.setText("1234ABCD")
        assert dialog.project == original
        assert "PCA-A=0x1234" in dialog.address_readback.text()
        assert "Custom" in dialog.packet_profile_label.text()
        assert "HEC-C=0x" in dialog.integrity_readback.text()
        dialog._accept_settings()
        assert dialog.result() == QtWidgets.QDialog.DialogCode.Accepted
        settings = dialog.project.bluetooth_hdt
        assert (settings.pca, settings.nesn, settings.md, settings.sn, settings.llid) == (
            0x123456789A, 3, 1, 7, 3,
        )
        assert settings.crc_init == 0x1234ABCD
        result = BluetoothHDTWaveformEngine().generate(dialog.project)
        assert analyze_generation_result(result).integrity.crc_valid
    finally:
        dialog.close()


def test_dialog_invalid_hex_blocks_ok_and_cancel_keeps_original(monkeypatch):
    pg.mkQApp()
    original = bluetooth_hdt_project()
    dialog = _BluetoothHDTSettingsDialog(original)
    notices = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args: notices.append(args))
    try:
        dialog.pca_edit.setText("12")
        dialog._accept_settings()
        assert notices and dialog.project == original
        assert dialog.result() != QtWidgets.QDialog.DialogCode.Accepted
        dialog.reject()
        assert dialog.project == original
    finally:
        dialog.close()


def test_dialog_manual_integrity_modes_and_readback():
    pg.mkQApp()
    dialog = _BluetoothHDTSettingsDialog(bluetooth_hdt_project())
    try:
        assert not dialog.hec_edit.isEnabled() and not dialog.crc_edit.isEnabled()
        dialog.hec_auto_check.setChecked(False)
        dialog.crc_auto_check.setChecked(False)
        assert dialog.hec_edit.isEnabled() and dialog.crc_edit.isEnabled()
        dialog.hec_edit.setText("ABCDEF")
        dialog.crc_edit.setText("DEADBEEF")
        assert "HEC-C=0xABCDEF; CRC-32=0xDEADBEEF" == dialog.integrity_readback.text()
        dialog._accept_settings()
        settings = dialog.project.bluetooth_hdt
        assert not settings.hec_auto and not settings.crc_auto
        assert settings.hec_manual == 0xABCDEF and settings.crc_manual == 0xDEADBEEF
    finally:
        dialog.close()


def test_incomplete_payload_edit_does_not_crash_or_commit(monkeypatch):
    pg.mkQApp()
    original = bluetooth_hdt_project()
    dialog = _BluetoothHDTSettingsDialog(original)
    notices = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args: notices.append(args))
    try:
        dialog.source_combo.setCurrentIndex(dialog.source_combo.findData(HDTPayloadSourceKind.PATTERN))
        dialog.pattern_edit.clear()
        assert "invalid" in dialog.integrity_readback.text()
        dialog._accept_settings()
        assert notices and dialog.project == original
        dialog.pattern_edit.setText("1010")
        assert "CRC-32=0x" in dialog.integrity_readback.text()
    finally:
        dialog.close()
