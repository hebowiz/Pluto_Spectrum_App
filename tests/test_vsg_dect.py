from dataclasses import replace
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

from pluto_protocol.dect import DECT_CARRIER_PLANS
from pluto_sa.vsa.model import IQRecording
from pluto_sa.vsa.protocol_modes.dect.carriers import (
    DECT_CARRIER_PLANS as VSA_DECT_CARRIER_PLANS,
)
from pluto_sa.vsa.protocol_modes.dect import analyze_dect_recording
from pluto_vsg.engine import DectWaveformEngine
from pluto_vsg.model import (
    DectDirection,
    DectPacketType,
    PayloadSourceKind,
    StandardProfile,
    validate_project,
)
from pluto_vsg.persistence import project_from_dict, project_to_dict
from pluto_vsg.profiles import dect_fields, dect_project
from pluto_vsg.protocol import analyze_generation_result
from pluto_vsg.ui.dect_settings import DectSettingsDialog
from pluto_vsg.ui.main_window import PlutoVSGWindow


def _text(bits: np.ndarray) -> str:
    return "".join(str(int(bit)) for bit in bits)


def _updated_project(project, settings):
    packet_type = DectPacketType(settings.packet_type)
    period = 960.0 if packet_type in {DectPacketType.P80, DectPacketType.P80Z} else 480.0
    return replace(
        project,
        dect=settings,
        fields=dect_fields(settings),
        period_symbols=period,
    )


def test_vsa_and_vsg_share_one_dect_carrier_plan_definition() -> None:
    assert VSA_DECT_CARRIER_PLANS is DECT_CARRIER_PLANS
    assert [carrier.channel for carrier in DECT_CARRIER_PLANS[2].carriers] == [
        "F7", "F8", "F9", "Fa", "Fb", "F0", "F1", "F2", "F3", "F4", "F5", "F6"
    ]


def test_default_dect_project_exposes_complete_field_hierarchy() -> None:
    project = dect_project()
    assert project.standard is StandardProfile.DECT
    assert project.sample_rate_hz == 9_216_000.0
    assert project.center_frequency_hz == 1_897_344_000.0
    assert validate_project(project) == ()
    assert [field.name for field in project.fields] == [
        "S-field", "A-field", "B-field", "X-field"
    ]
    assert [child.name for child in project.fields[0].children] == [
        "Preamble", "Packet Synchronization Word"
    ]
    assert [child.name for child in project.fields[1].children] == [
        "Header", "Tail", "R-CRC"
    ]


def test_all_dect_packet_types_generate_and_decode_with_valid_checks() -> None:
    expected_lengths = {
        DectPacketType.P00: 96,
        DectPacketType.P32: 420,
        DectPacketType.P32Z: 424,
        DectPacketType.P80: 900,
        DectPacketType.P80Z: 904,
    }
    engine = DectWaveformEngine()
    for packet_type, expected_length in expected_lengths.items():
        base = dect_project()
        settings = replace(
            base.dect,
            packet_type=packet_type,
            b_field_source=PayloadSourceKind.PRBS9,
        )
        result = engine.generate(_updated_project(base, settings))
        decoded = analyze_generation_result(result)
        summary = {item.key: item for item in decoded.summary}
        assert result.packet_bits.bits.size == expected_length
        assert decoded.packet_type == packet_type.value
        assert decoded.integrity.complete
        assert summary["r_crc"].value is True
        assert summary["x_crc"].value is (None if packet_type is DectPacketType.P00 else True)


def test_direction_derived_sync_and_manual_crc_fields_are_transmitted() -> None:
    base = dect_project()
    settings = replace(
        base.dect,
        preamble_bits="0101010101010101",
        sync_word_bits="0000111100001111",
        a_header_bits="10110110",
        a_tail_bits="10" * 20,
        r_crc_auto=False,
        r_crc_bits="1010010110100101",
        b_field_source=PayloadSourceKind.PATTERN,
        b_field_pattern="0011",
        x_crc_auto=False,
        x_field_bits="1010",
    )
    result = DectWaveformEngine().generate(_updated_project(base, settings))
    assert _text(result.metadata["preamble_bits"]) == "1010101010101010"
    assert _text(result.metadata["sync_word_bits"]) == "1110100110001010"
    assert _text(result.metadata["a_header_bits"]) == settings.a_header_bits
    assert _text(result.metadata["a_tail_bits"]) == settings.a_tail_bits
    assert _text(result.metadata["r_crc_bits"]) == settings.r_crc_bits
    assert _text(result.metadata["b_field_bits"][:12]) == "001100110011"
    assert _text(result.metadata["x_field_bits"]) == "1010"


def test_frequency_offset_is_a_baseband_impairment_from_selected_carrier() -> None:
    base = dect_project()
    offset_settings = replace(base.dect, carrier_frequency_offset_hz=37_500.0)
    offset_project = _updated_project(base, offset_settings)
    nominal = DectWaveformEngine().generate(base)
    offset = DectWaveformEngine().generate(offset_project)
    start, stop = nominal.metadata["packet_ranges_samples"][0]
    ratio = offset.iq[start + 1 : stop] * np.conj(nominal.iq[start + 1 : stop])
    increment = ratio[1:] * np.conj(ratio[:-1])
    measured = np.mean(np.angle(increment)) * nominal.sample_rate_hz / (2.0 * np.pi)
    np.testing.assert_allclose(measured, 37_500.0, atol=1.0)
    assert offset.metadata["center_frequency_hz"] == base.center_frequency_hz
    assert offset.metadata["actual_rf_frequency_hz"] == base.center_frequency_hz + 37_500.0


def test_prolonged_preamble_and_z_repeat_are_explicit() -> None:
    base = dect_project()
    settings = replace(
        base.dect,
        packet_type=DectPacketType.P32Z,
        prolonged_preamble=True,
    )
    result = DectWaveformEngine().generate(_updated_project(base, settings))
    assert result.packet_bits.bits.size == 440
    assert _text(result.packet_bits.bits[:16]) == settings.preamble_bits
    assert _text(result.metadata["z_field_bits"]) == _text(result.metadata["x_field_bits"])
    assert result.packet_bits.context["p0_internal_bit"] == 16


def test_generated_waveform_is_detected_by_dect_vsa() -> None:
    project = dect_project()
    generated = DectWaveformEngine().generate(project)
    recording = IQRecording(
        iq=generated.iq,
        sample_rate_hz=generated.sample_rate_hz,
        center_frequency_hz=project.center_frequency_hz,
        usable_bandwidth_hz=0.8 * generated.sample_rate_hz,
        source="DECT VSG integration test",
        full_scale=1.0,
        amplitude_calibrated=False,
        metadata={"dc_removal_recommended": False},
    )

    analyzed = analyze_dect_recording(recording)[0]

    assert analyzed.direction == "RFP"
    assert analyzed.packet_type == "P32"
    assert analyzed.modulation_case == "Case A (00001111)"
    assert analyzed.sync_score > 0.99


def test_dect_project_json_round_trip() -> None:
    base = dect_project()
    settings = replace(
        base.dect,
        direction=DectDirection.PP,
        packet_type=DectPacketType.P80Z,
        carrier_plan_id="j_dect",
        carrier_channel="F0",
        carrier_frequency_offset_hz=-12_500.0,
        b_field_source=PayloadSourceKind.PRBS9,
    )
    project = replace(
        _updated_project(base, settings),
        center_frequency_hz=1_893_888_000.0,
    )
    assert project_from_dict(project_to_dict(project)) == project


def test_dect_settings_dialog_uses_carrier_list_and_updates_fields() -> None:
    pg.mkQApp("DECT VSG settings test")
    dialog = DectSettingsDialog(dect_project())
    try:
        dialog.plan_combo.setCurrentIndex(dialog.plan_combo.findData("j_dect"))
        dialog.carrier_combo.setCurrentIndex(dialog.carrier_combo.findData("F0"))
        dialog.offset_spin.setValue(12.5)
        dialog.packet_type_combo.setCurrentIndex(
            dialog.packet_type_combo.findData(DectPacketType.P32Z)
        )
        dialog.ta_combo.setCurrentIndex(dialog.ta_combo.findData(0b110))
        dialog.q1_combo.setCurrentIndex(dialog.q1_combo.findData(1))
        dialog.ba_combo.setCurrentIndex(dialog.ba_combo.findData(0b011))
        dialog.q2_combo.setCurrentIndex(dialog.q2_combo.findData(0))
        dialog.period_spin.setValue(600.0)
        dialog.rise_spin.setValue(2.5)
        dialog.rise_delay_spin.setValue(-2.5)
        dialog.fall_spin.setValue(3.5)
        dialog.fall_delay_spin.setValue(0.5)
        project = dialog.project
        assert project.center_frequency_hz == 1_893_888_000.0
        assert project.dect.carrier_frequency_offset_hz == 12_500.0
        assert project.dect.a_header_bits == "11010110"
        assert project.dect.post_idle_symbols == 0
        assert project.period_symbols == 600.0
        assert project.power_envelope.rise_symbols == 2.5
        assert project.power_envelope.rise_delay_symbols == -2.5
        assert project.power_envelope.fall_symbols == 3.5
        assert project.power_envelope.fall_delay_symbols == 0.5
        assert [field.name for field in project.fields][-2:] == ["X-field", "Z-field"]
        assert "1893.900500 MHz" in dialog.actual_frequency_label.text()
    finally:
        dialog.close()


def test_sync_is_direction_derived_and_a_field_uses_choice_controls() -> None:
    pg.mkQApp("DECT VSG structured field controls test")
    dialog = DectSettingsDialog(dect_project())
    try:
        assert isinstance(dialog.preamble_value, QtWidgets.QLabel)
        assert isinstance(dialog.sync_value, QtWidgets.QLabel)
        assert isinstance(dialog.ta_combo, QtWidgets.QComboBox)
        assert isinstance(dialog.ba_combo, QtWidgets.QComboBox)
        assert isinstance(dialog.a_tail_combo, QtWidgets.QComboBox)
        assert not hasattr(dialog, "preamble_edit")
        assert not hasattr(dialog, "sync_edit")
        assert not hasattr(dialog, "a_tail_edit")
        assert not hasattr(dialog, "r_crc_edit")

        dialog.direction_combo.setCurrentIndex(
            dialog.direction_combo.findData(DectDirection.PP)
        )
        assert dialog.preamble_value.text() == "0101010101010101"
        assert dialog.sync_value.text() == "0001011001110101"
        assert dialog.project.dect.direction is DectDirection.PP
    finally:
        dialog.close()


def test_main_window_dispatches_dect_engine_and_uses_three_mhz_tx_bandwidth() -> None:
    pg.mkQApp("DECT VSG main window test")
    window = PlutoVSGWindow()
    try:
        window._new_dect_project()
        assert window.project.standard is StandardProfile.DECT
        assert window.result.packet_bits.protocol_id == "dect.classic"
        assert window.settings_action.text() == "DECT Packet / Waveform Settings..."
        assert window._current_pluto_settings().rf_bandwidth_hz >= 3_000_000.0
        assert window.new_dect_action in window.findChildren(type(window.new_dect_action))
    finally:
        window.close()
