import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pyqtgraph as pg
import pytest
from pyqtgraph.Qt import QtWidgets
from pluto_vsg.model import (
    BluetoothLEPayloadSourceKind,
    BluetoothLEPhy,
    BluetoothPacketKind,
    PayloadSourceKind,
)
from pluto_vsg.profiles import (
    bluetooth_br_edr_project,
    bluetooth_hdt_project,
    bluetooth_le_project,
)
from pluto_vsg.ui.main_window import (
    PlutoVSGWindow,
    _BluetoothHDTSettingsDialog,
    _BluetoothLESettingsDialog,
    _BluetoothSettingsDialog,
)
from pluto_vsg.ui.frequency_settings import (
    FrequencySelection,
    FrequencySettingsDialog,
    effective_rf_frequency_hz,
    with_manual_rf_frequency,
)


def _tab_names(dialog) -> list[str]:
    return [dialog.tabs.tabText(index) for index in range(dialog.tabs.count())]


def test_bluetooth_packet_settings_share_two_tab_structure() -> None:
    pg.mkQApp("Bluetooth VSG common packet settings tabs")
    parent = QtWidgets.QWidget()
    dialogs = (
        _BluetoothSettingsDialog(bluetooth_br_edr_project(), parent),
        _BluetoothLESettingsDialog(bluetooth_le_project(), parent),
        _BluetoothHDTSettingsDialog(bluetooth_hdt_project(), parent),
    )
    try:
        assert all(_tab_names(dialog) == ["RF / Timing", "Fields"] for dialog in dialogs)
        assert all(
            "us" in dialog._timing_controls[0].time_label.text()
            for dialog in dialogs
        )
    finally:
        for dialog in dialogs:
            dialog.close()
        parent.close()


def test_frequency_settings_uses_protocol_carrier_and_offset() -> None:
    pg.mkQApp("Pluto VSG frequency settings")
    project = bluetooth_br_edr_project()
    dialog = FrequencySettingsDialog(project)
    try:
        dialog.carrier_combo.setCurrentIndex(
            dialog.carrier_combo.findData(2_402_000_000.0)
        )
        dialog.offset_spin.setValue(125.0)
        updated = dialog.project
        assert updated.center_frequency_hz == 2_402_000_000.0
        assert effective_rf_frequency_hz(updated) == 2_402_125_000.0
    finally:
        dialog.close()


def test_frequency_settings_reopens_saved_selection_not_manual_frequency() -> None:
    pg.mkQApp("Pluto VSG independent frequency selection")
    project = with_manual_rf_frequency(bluetooth_br_edr_project(), 2_475_123_456.0)
    previous = FrequencySelection(
        "bluetooth_classic", "0", 2_402_000_000.0, 125_000.0
    )
    dialog = FrequencySettingsDialog(project, selection=previous)
    try:
        assert dialog.carrier_combo.currentData() == 2_402_000_000.0
        assert dialog.offset_spin.value() == pytest.approx(125.0)
        assert effective_rf_frequency_hz(dialog.project) == 2_402_125_000.0
    finally:
        dialog.close()


def test_vsg_payload_source_labels_explain_generation_behavior() -> None:
    pg.mkQApp("Pluto VSG payload source label test")
    parent = PlutoVSGWindow()
    dialog = _BluetoothSettingsDialog(parent.project, parent)
    try:
        labels = [
            dialog.payload_source_combo.itemText(index)
            for index in range(dialog.payload_source_combo.count())
        ]
        assert labels == [
            "Constant (All 0 / All 1)",
            "Repeating Bit Pattern",
            "PRBS-9",
        ]
        assert "PRBS-9 sequence" in dialog.payload_source_help.text()
        ok_button = dialog.findChild(QtWidgets.QDialogButtonBox).button(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
        )
        assert ok_button.text() == "Apply and Generate"
    finally:
        dialog.close()
        parent.close()


def test_vsg_settings_keep_composer_payload_source_in_sync() -> None:
    pg.mkQApp("Pluto VSG composer source sync test")
    parent = PlutoVSGWindow()
    dialog = _BluetoothSettingsDialog(parent.project, parent)
    try:
        dialog.payload_source_combo.setCurrentIndex(
            dialog.payload_source_combo.findData(PayloadSourceKind.PATTERN)
        )
        dialog.pattern_edit.setText("1100")
        dialog._accept_settings()

        payload_field = next(
            packet_field
            for packet_field in dialog.project.fields
            if packet_field.name == "Payload"
        )
        assert payload_field.data_source.value == "Pattern"
        assert payload_field.data == "1100"
    finally:
        dialog.close()
        parent.close()


def test_vsg_settings_edit_packet_header_and_recalculate_hec() -> None:
    pg.mkQApp("Pluto VSG packet header settings test")
    parent = PlutoVSGWindow()
    dialog = _BluetoothSettingsDialog(parent.project, parent)
    try:
        initial_hec = dialog.hec_value.text()
        dialog.lt_addr_spin.setValue(5)
        dialog.flow_combo.setCurrentIndex(dialog.flow_combo.findData(0))
        dialog.arqn_combo.setCurrentIndex(dialog.arqn_combo.findData(1))
        dialog.seqn_combo.setCurrentIndex(dialog.seqn_combo.findData(1))

        assert dialog.hec_mode_combo.currentData() is True
        assert dialog.hec_value.text().startswith("0x")
        assert not dialog.hec_value.isEnabled()
        assert dialog.hec_value.text() != initial_hec

        dialog._accept_settings()
        settings = dialog.project.bluetooth_br
        assert settings is not None
        assert settings.lt_addr == 5
        assert settings.flow == 0
        assert settings.arqn == 1
        assert settings.seqn == 1
    finally:
        dialog.close()
        parent.close()


def test_classic_rf_test_preset_populates_existing_payload_controls() -> None:
    pg.mkQApp("Pluto VSG Classic RF preset test")
    parent = PlutoVSGWindow()
    dialog = _BluetoothSettingsDialog(parent.project, parent)
    try:
        dialog.rf_test_payload_combo.setCurrentIndex(
            dialog.rf_test_payload_combo.findData("11110000")
        )
        dialog._apply_rf_test_preset()

        assert dialog.payload_source_combo.currentData() == PayloadSourceKind.PATTERN
        assert dialog.pattern_edit.text() == "11110000"
        assert dialog.whitening_check.isChecked() is False
    finally:
        dialog.close()
        parent.close()


def test_le_rf_test_preset_populates_editable_packet_controls() -> None:
    pg.mkQApp("Pluto VSG LE RF preset test")
    project = bluetooth_le_project(BluetoothLEPhy.LE_2M)
    parent = PlutoVSGWindow(project)
    dialog = _BluetoothLESettingsDialog(project, parent)
    try:
        dialog._apply_rf_test_preset()

        assert dialog.sync_edit.text() == "10010100100000100110111010001110"
        assert dialog.payload_source_combo.currentData() == (
            BluetoothLEPayloadSourceKind.PATTERN
        )
        assert dialog.payload_pattern_edit.text() == "10101010"
        assert dialog.crc_init_edit.text() == "555555"
        assert dialog.whitening_check.isChecked() is False
    finally:
        dialog.close()
        parent.close()


def test_vsg_settings_dialog_preserves_edr_guard_relative_power() -> None:
    pg.mkQApp("Pluto VSG EDR guard power settings test")
    parent = PlutoVSGWindow()
    dialog = _BluetoothSettingsDialog(parent.project, parent)
    try:
        dialog.packet_type_combo.setCurrentIndex(
            dialog.packet_type_combo.findData(BluetoothPacketKind.DH1_2)
        )
        dialog.edr_guard_power_spin.setValue(-18.5)
        dialog.edr_guard_ramp_in_spin.setValue(0.75)
        dialog.edr_guard_ramp_out_spin.setValue(0.5)
        dialog.edr_guard_ramp_shape_combo.setCurrentText("Linear")
        dialog._accept_settings()

        settings = dialog.project.bluetooth_br
        assert settings is not None
        assert settings.edr_guard_relative_power_db == -18.5
        assert settings.edr_guard_ramp_in_symbols == 0.75
        assert settings.edr_guard_ramp_out_symbols == 0.5
        assert settings.edr_guard_ramp_shape == "Linear"
        assert dialog.project.fields[2].relative_power_db == -18.5
    finally:
        dialog.close()
        parent.close()


def test_vsg_packet_type_dialog_updates_edr_project() -> None:
    pg.mkQApp("Pluto VSG EDR settings test")
    parent = PlutoVSGWindow()
    dialog = _BluetoothSettingsDialog(parent.project, parent)
    try:
        dialog.packet_type_combo.setCurrentIndex(
            dialog.packet_type_combo.findData(BluetoothPacketKind.DH1_3)
        )
        dialog.payload_length_spin.setValue(83)
        dialog._accept_settings()

        settings = dialog.project.bluetooth_br
        assert settings is not None
        assert settings.packet_kind == BluetoothPacketKind.DH1_3
        assert dialog.project.fields[-1].name == "EDR Data"
        assert dialog.project.fields[-1].modulation.kind.value == "8DPSK"
    finally:
        dialog.close()
        parent.close()


def test_vsg_dialog_preserves_maximum_edr_payload_when_reopened() -> None:
    pg.mkQApp("Pluto VSG payload persistence test")
    parent = PlutoVSGWindow()
    first = _BluetoothSettingsDialog(parent.project, parent)
    second = None
    try:
        first.packet_type_combo.setCurrentIndex(
            first.packet_type_combo.findData(BluetoothPacketKind.DH5_3)
        )
        first.payload_length_spin.setValue(1021)
        first._accept_settings()

        second = _BluetoothSettingsDialog(first.project, parent)

        assert second.payload_length_spin.maximum() == 1021
        assert second.payload_length_spin.value() == 1021
    finally:
        if second is not None:
            second.close()
        first.close()
        parent.close()


@pytest.mark.parametrize(
    ("packet_kind", "payload_max"),
    (
        (BluetoothPacketKind.DH1, 27),
        (BluetoothPacketKind.DH3, 183),
        (BluetoothPacketKind.DH5, 339),
        (BluetoothPacketKind.DH1_2, 54),
        (BluetoothPacketKind.DH3_2, 367),
        (BluetoothPacketKind.DH5_2, 679),
        (BluetoothPacketKind.DH1_3, 83),
        (BluetoothPacketKind.DH3_3, 552),
        (BluetoothPacketKind.DH5_3, 1021),
    ),
)
def test_vsg_dialog_packet_type_change_selects_maximum_payload(
    packet_kind, payload_max
) -> None:
    parent = QtWidgets.QWidget()
    dialog = _BluetoothSettingsDialog(bluetooth_br_edr_project(), parent)
    try:
        # Move away first so the DH1 case also emits currentIndexChanged.
        dialog.packet_type_combo.setCurrentIndex(
            dialog.packet_type_combo.findData(BluetoothPacketKind.DH5_3)
        )
        dialog.packet_type_combo.setCurrentIndex(
            dialog.packet_type_combo.findData(packet_kind)
        )
        assert dialog.payload_length_spin.maximum() == payload_max
        assert dialog.payload_length_spin.value() == payload_max
    finally:
        dialog.close()
        parent.close()
