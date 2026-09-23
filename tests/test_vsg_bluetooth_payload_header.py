import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace

import numpy as np
import pyqtgraph as pg
import pytest
from pyqtgraph.Qt import QtWidgets

from pluto_vsa.profiles.bluetooth_br import (
    fec13_encode,
    header_error_check,
    payload_crc_bytes,
    whitening_sequence,
)
from pluto_vsa.profiles.bluetooth_edr import (
    EDR_SYNC_BITS_2MBPS,
    EDR_SYNC_BITS_3MBPS,
)
from pluto_vsg.engine import BluetoothBRWaveformEngine
from pluto_vsg.model import (
    BluetoothBRSettings,
    BluetoothPacketKind,
    bluetooth_packet_properties,
    validate_project,
)
from pluto_vsg.persistence import project_from_dict, project_to_dict
from pluto_vsg.profiles import bluetooth_br_edr_project, bluetooth_br_fields
from pluto_vsg.ui.main_window import _BluetoothSettingsDialog


def _project(
    packet_kind: BluetoothPacketKind = BluetoothPacketKind.DH1,
    *,
    payload_llid: int = 2,
    packet_flow: int = 1,
    payload_flow: int = 1,
    whitening_enabled: bool = False,
    hec_auto: bool = True,
    hec_manual: int = 0,
):
    base = bluetooth_br_edr_project()
    settings = replace(
        base.bluetooth_br,
        packet_kind=packet_kind,
        payload_length_bytes=min(11, bluetooth_packet_properties(packet_kind)[0]),
        payload_llid=payload_llid,
        flow=packet_flow,
        payload_flow=payload_flow,
        whitening_enabled=whitening_enabled,
        hec_auto=hec_auto,
        hec_manual=hec_manual,
    )
    return replace(
        base,
        bluetooth_br=settings,
        fields=bluetooth_br_fields(settings),
    )


def _generated(project):
    return BluetoothBRWaveformEngine().generate(project)


def _packed(bits: np.ndarray) -> int:
    return sum(int(bit) << index for index, bit in enumerate(bits))


def _packet_header_logical_bits(result, project) -> np.ndarray:
    settings = project.bluetooth_br
    encoded = np.asarray(result.metadata["packet_bits"])[72:126]
    fec_bits = encoded.reshape(-1, 3)[:, 0]
    if not settings.whitening_enabled:
        return fec_bits
    return fec_bits ^ whitening_sequence(settings.clock_6_1, 18)


def _hec_from_header_bits(header: np.ndarray) -> int:
    return sum(int(bit) << (7 - index) for index, bit in enumerate(header[10:18]))


def _unwhitened_payload(result, project) -> np.ndarray:
    settings = project.bluetooth_br
    payload = np.concatenate(
        (
            result.metadata["payload_header_bits"],
            result.metadata["payload_body_bits"],
            result.metadata["payload_crc_bits"],
        )
    )
    _maximum, _packet_type, bits_per_symbol, _slots = bluetooth_packet_properties(
        settings.packet_kind
    )
    packet_bits = np.asarray(result.metadata["packet_bits"])
    if bits_per_symbol == 1:
        transmitted = packet_bits[126 : 126 + payload.size]
    else:
        sync = (
            EDR_SYNC_BITS_2MBPS
            if bits_per_symbol == 2
            else EDR_SYNC_BITS_3MBPS
        )
        transmitted = packet_bits[126 + sync.size : 126 + sync.size + payload.size]
    if not settings.whitening_enabled:
        return transmitted
    sequence = whitening_sequence(settings.clock_6_1, 18 + payload.size)
    return transmitted ^ sequence[18:]


def test_payload_header_defaults_preserve_previous_bit_pattern() -> None:
    settings = BluetoothBRSettings()
    result = _generated(_project())
    header = np.asarray(result.metadata["payload_header_bits"])

    assert settings.payload_llid == 2
    assert settings.payload_flow == 1
    assert _packed(header) == 2 | (1 << 2) | (11 << 3)


@pytest.mark.parametrize("whitening_enabled", (False, True))
@pytest.mark.parametrize(
    "packet_kind", (BluetoothPacketKind.DH1, BluetoothPacketKind.DH1_2)
)
def test_auto_hec_preserves_calculated_header_and_normal_fec(
    packet_kind: BluetoothPacketKind, whitening_enabled: bool
) -> None:
    project = _project(packet_kind, whitening_enabled=whitening_enabled)
    result = _generated(project)
    logical = _packet_header_logical_bits(result, project)
    expected_hec = header_error_check(logical[:10], project.bluetooth_br.uap)
    transmitted_logical = logical
    if whitening_enabled:
        transmitted_logical = logical ^ whitening_sequence(
            project.bluetooth_br.clock_6_1, 18
        )

    assert _hec_from_header_bits(logical) == expected_hec
    assert result.metadata["packet_header_hec"] == expected_hec
    assert result.metadata["packet_header_hec_mode"] == "Auto"
    np.testing.assert_array_equal(
        np.asarray(result.metadata["packet_bits"])[72:126],
        fec13_encode(transmitted_logical),
    )


@pytest.mark.parametrize("manual_hec", (0x00, 0x01, 0xA7, 0xFF))
@pytest.mark.parametrize("whitening_enabled", (False, True))
def test_manual_hec_is_transmitted_before_normal_fec_and_whitening(
    manual_hec: int, whitening_enabled: bool
) -> None:
    project = _project(
        BluetoothPacketKind.DH1_3,
        whitening_enabled=whitening_enabled,
        hec_auto=False,
        hec_manual=manual_hec,
    )
    result = _generated(project)
    logical = _packet_header_logical_bits(result, project)
    transmitted_logical = logical
    if whitening_enabled:
        transmitted_logical = logical ^ whitening_sequence(
            project.bluetooth_br.clock_6_1, 18
        )

    assert _hec_from_header_bits(logical) == manual_hec
    assert result.metadata["packet_header_hec"] == manual_hec
    assert result.metadata["packet_header_hec_mode"] == "Manual"
    np.testing.assert_array_equal(
        np.asarray(result.metadata["packet_bits"])[72:126],
        fec13_encode(transmitted_logical),
    )


@pytest.mark.parametrize(
    "changes",
    (
        {"lt_addr": 5},
        {"packet_kind": BluetoothPacketKind.DH3},
        {"flow": 0},
        {"arqn": 1},
        {"seqn": 1},
        {"uap": 0x47},
    ),
)
def test_packet_header_changes_update_only_auto_hec(changes) -> None:
    base = _project()
    auto_settings = replace(base.bluetooth_br, **changes)
    auto_project = replace(
        base,
        bluetooth_br=auto_settings,
        fields=bluetooth_br_fields(auto_settings),
    )
    auto_result = _generated(auto_project)
    auto_header = _packet_header_logical_bits(auto_result, auto_project)
    assert _hec_from_header_bits(auto_header) == header_error_check(
        auto_header[:10], auto_settings.uap
    )

    manual_settings = replace(auto_settings, hec_auto=False, hec_manual=0x5A)
    manual_project = replace(
        auto_project,
        bluetooth_br=manual_settings,
        fields=bluetooth_br_fields(manual_settings),
    )
    manual_header = _packet_header_logical_bits(
        _generated(manual_project), manual_project
    )
    assert _hec_from_header_bits(manual_header) == 0x5A


@pytest.mark.parametrize("llid", range(4))
def test_payload_llid_changes_only_header_bits_1_to_0(llid: int) -> None:
    reference = np.asarray(
        _generated(_project(payload_llid=2)).metadata["payload_header_bits"]
    )
    actual = np.asarray(
        _generated(_project(payload_llid=llid)).metadata["payload_header_bits"]
    )

    np.testing.assert_array_equal(actual[2:], reference[2:])
    assert _packed(actual[:2]) == llid


@pytest.mark.parametrize("payload_flow", (0, 1))
def test_payload_flow_changes_only_header_bit_2(payload_flow: int) -> None:
    reference = np.asarray(
        _generated(_project(payload_flow=1)).metadata["payload_header_bits"]
    )
    actual = np.asarray(
        _generated(_project(payload_flow=payload_flow)).metadata[
            "payload_header_bits"
        ]
    )

    np.testing.assert_array_equal(actual[:2], reference[:2])
    np.testing.assert_array_equal(actual[3:], reference[3:])
    assert int(actual[2]) == payload_flow


def test_packet_header_flow_and_payload_header_flow_are_independent() -> None:
    project = _project(packet_flow=0, payload_flow=1)
    result = _generated(project)
    header_air = np.asarray(result.metadata["packet_bits"])[72:126]
    packet_header_data = header_air.reshape(-1, 3)[:, 0][:10]

    assert int(packet_header_data[7]) == 0
    assert int(result.metadata["payload_header_bits"][2]) == 1

    inverse = _generated(_project(packet_flow=1, payload_flow=0))
    inverse_header_air = np.asarray(inverse.metadata["packet_bits"])[72:126]
    inverse_packet_header = inverse_header_air.reshape(-1, 3)[:, 0][:10]
    assert int(inverse_packet_header[7]) == 1
    assert int(inverse.metadata["payload_header_bits"][2]) == 0


def test_payload_crc_is_recomputed_from_configured_header() -> None:
    default = _generated(_project(payload_llid=2, payload_flow=1))
    changed = _generated(_project(payload_llid=3, payload_flow=0))

    for result in (default, changed):
        expected = payload_crc_bytes(
            np.concatenate(
                (
                    result.metadata["payload_header_bits"],
                    result.metadata["payload_body_bits"],
                )
            ),
            0x6B,
        )
        expected_bits = np.asarray(
            [(byte >> bit) & 1 for byte in expected for bit in range(8)],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(
            result.metadata["payload_crc_bits"], expected_bits
        )
    assert not np.array_equal(
        default.metadata["payload_crc_bits"], changed.metadata["payload_crc_bits"]
    )


@pytest.mark.parametrize("packet_kind", list(BluetoothPacketKind))
@pytest.mark.parametrize("whitening_enabled", (False, True))
def test_configured_payload_header_survives_all_dhx_and_whitening_modes(
    packet_kind: BluetoothPacketKind, whitening_enabled: bool
) -> None:
    project = _project(
        packet_kind,
        payload_llid=3,
        payload_flow=0,
        whitening_enabled=whitening_enabled,
    )
    assert validate_project(project) == ()

    result = _generated(project)
    header = np.asarray(result.metadata["payload_header_bits"])
    expected_width = 8 if packet_kind is BluetoothPacketKind.DH1 else 16
    expected_payload = np.concatenate(
        (
            header,
            result.metadata["payload_body_bits"],
            result.metadata["payload_crc_bits"],
        )
    )

    assert header.size == expected_width
    assert _packed(header) == 3 | (11 << 3)
    np.testing.assert_array_equal(_unwhitened_payload(result, project), expected_payload)


def test_payload_header_values_round_trip_and_old_json_uses_defaults() -> None:
    expected = _project(
        payload_llid=1,
        payload_flow=0,
        hec_auto=False,
        hec_manual=0xA7,
    )
    restored = project_from_dict(project_to_dict(expected))
    assert restored.bluetooth_br.payload_llid == 1
    assert restored.bluetooth_br.payload_flow == 0
    assert restored.bluetooth_br.hec_auto is False
    assert restored.bluetooth_br.hec_manual == 0xA7

    old_document = project_to_dict(expected)
    old_settings = old_document["project"]["bluetooth_br"]
    old_settings.pop("payload_llid")
    old_settings.pop("payload_flow")
    old_settings.pop("hec_auto")
    old_settings.pop("hec_manual")
    old_restored = project_from_dict(old_document)
    assert old_restored.bluetooth_br.payload_llid == 2
    assert old_restored.bluetooth_br.payload_flow == 1
    assert old_restored.bluetooth_br.hec_auto is True
    assert old_restored.bluetooth_br.hec_manual == 0x00


def test_hec_dialog_auto_manual_switch_and_saved_preview() -> None:
    pg.mkQApp("Bluetooth manual HEC settings test")
    parent = QtWidgets.QWidget()
    dialog = _BluetoothSettingsDialog(_project(), parent)
    reopened = None
    try:
        automatic = dialog.hec_value.text()
        assert dialog.hec_mode_combo.currentData() is True
        assert not dialog.hec_value.isEnabled()

        dialog.hec_mode_combo.setCurrentIndex(
            dialog.hec_mode_combo.findData(False)
        )
        assert dialog.hec_value.isEnabled()
        assert dialog.hec_value.text() == automatic
        dialog.hec_value.setText("0xA6")
        dialog.lt_addr_spin.setValue(5)
        dialog.uap_edit.setText("47")
        assert dialog.hec_value.text() == "0xA6"

        dialog.hec_mode_combo.setCurrentIndex(
            dialog.hec_mode_combo.findData(True)
        )
        assert not dialog.hec_value.isEnabled()
        assert dialog.hec_value.text() != "0xA6"
        dialog.hec_mode_combo.setCurrentIndex(
            dialog.hec_mode_combo.findData(False)
        )
        dialog.hec_value.setText("0x5A")
        dialog._accept_settings()

        assert dialog.project.bluetooth_br.hec_auto is False
        assert dialog.project.bluetooth_br.hec_manual == 0x5A
        hec_field = next(
            child
            for field in dialog.project.fields
            for child in field.children
            if child.name == "HEC"
        )
        assert hec_field.data == "0x5A (manual) + 1/3 FEC"

        reopened = _BluetoothSettingsDialog(dialog.project, parent)
        assert reopened.hec_mode_combo.currentData() is False
        assert reopened.hec_value.isEnabled()
        assert reopened.hec_value.text() == "0x5A"
    finally:
        if reopened is not None:
            reopened.close()
        dialog.close()
        parent.close()


def test_payload_header_field_and_dialog_show_and_restore_independent_values() -> None:
    pg.mkQApp("Bluetooth payload header settings test")
    parent = QtWidgets.QWidget()
    project = _project(payload_llid=1, packet_flow=1, payload_flow=0)
    dialog = _BluetoothSettingsDialog(project, parent)
    reopened = None
    try:
        assert dialog.flow_combo.currentData() == 1
        assert dialog.payload_llid_spin.value() == 1
        assert dialog.payload_flow_combo.currentData() == 0
        assert dialog.payload_header_length_value.text() == "11 byte (auto)"

        dialog.payload_llid_spin.setValue(3)
        dialog.payload_flow_combo.setCurrentIndex(
            dialog.payload_flow_combo.findData(1)
        )
        dialog.payload_length_spin.setValue(7)
        assert dialog.payload_header_length_value.text() == "7 byte (auto)"
        dialog._accept_settings()

        settings = dialog.project.bluetooth_br
        assert settings.flow == 1
        assert settings.payload_llid == 3
        assert settings.payload_flow == 1
        payload_header_field = next(
            child
            for field in dialog.project.fields
            for child in field.children
            if child.name == "Payload Header"
        )
        assert payload_header_field.data == "LLID=3, FLOW=1, LENGTH=7"

        reopened = _BluetoothSettingsDialog(dialog.project, parent)
        assert reopened.payload_llid_spin.value() == 3
        assert reopened.payload_flow_combo.currentData() == 1
        assert reopened.payload_header_length_value.text() == "7 byte (auto)"
    finally:
        if reopened is not None:
            reopened.close()
        dialog.close()
        parent.close()


@pytest.mark.parametrize(
    ("changes", "path"),
    (
        ({"payload_llid": -1}, "bluetooth_br.payload_llid"),
        ({"payload_llid": 4}, "bluetooth_br.payload_llid"),
        ({"payload_flow": -1}, "bluetooth_br.payload_flow"),
        ({"payload_flow": 2}, "bluetooth_br.payload_flow"),
        ({"hec_manual": -1}, "bluetooth_br.hec_manual"),
        ({"hec_manual": 256}, "bluetooth_br.hec_manual"),
        ({"hec_auto": 1}, "bluetooth_br.hec_auto"),
    ),
)
def test_payload_header_settings_are_validated(changes, path: str) -> None:
    base = bluetooth_br_edr_project()
    settings = replace(base.bluetooth_br, **changes)
    project = replace(
        base,
        bluetooth_br=settings,
        fields=bluetooth_br_fields(settings),
    )

    assert path in {issue.path for issue in validate_project(project)}
