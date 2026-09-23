from dataclasses import replace

import numpy as np
import pytest

from pluto_protocol import PacketDecodeInput, analyze_packet
from pluto_protocol.model import BitRepresentation
from pluto_protocol.bluetooth.hdt import HDTRate, map_hdt_symbols
from pluto_vsa.protocol import analyze_demodulated_packet_bits
from pluto_vsa.model import IQRecording
from pluto_vsa.protocol_modes.bluetooth.model import (
    BluetoothAnalysisProfile, analyze_bluetooth_hdt_recording,
)
from pluto_vsg.engine.bluetooth_hdt import BluetoothHDTWaveformEngine
from pluto_vsg.model import HDTPayloadSourceKind, validate_project
from pluto_vsg.persistence import project_from_dict, project_to_dict
from pluto_vsg.profiles import bluetooth_hdt_fields, bluetooth_hdt_project
from pluto_vsg.protocol import analyze_generation_result


def generated(rate, length=16, source=HDTPayloadSourceKind.PRBS9):
    base = bluetooth_hdt_project(rate)
    settings = replace(base.bluetooth_hdt, payload_length_bytes=length, payload_source=source)
    project = replace(base, bluetooth_hdt=settings, fields=bluetooth_hdt_fields(settings))
    return project, BluetoothHDTWaveformEngine().generate(project)


@pytest.mark.parametrize("rate", tuple(HDTRate))
@pytest.mark.parametrize("length", (1, 2, 510))
def test_vsa_vsg_share_hdt_logical_decoder(rate, length):
    project, result = generated(rate, length)
    vsg = analyze_generation_result(result)
    artifact = result.packet_bits
    vsa = analyze_demodulated_packet_bits(
        artifact.bits, protocol_id=artifact.protocol_id, phy_name=artifact.phy_name,
        representation=artifact.representation, context=dict(artifact.context),
    )
    assert artifact.representation == BitRepresentation.LOGICAL
    assert vsg.phy_name == vsa.phy_name == rate.value
    assert vsg.integrity.hec_valid is vsa.integrity.hec_valid is True
    assert vsg.integrity.crc_valid is vsa.integrity.crc_valid is True
    assert vsg.integrity.complete and not vsg.issues
    assert [(f.field_id, f.value) for f in vsg.root_fields] == [
        (f.field_id, f.value) for f in vsa.root_fields
    ]
    payload = vsg.root_fields[-1].children[1]
    np.testing.assert_array_equal(payload.raw_bits, result.metadata["payload_bits"])
    assert payload.raw_bits.size == length * 8


@pytest.mark.parametrize("position,code", ((33, "invalid_hec_c"), (-1, "invalid_crc32")))
def test_hdt_corruption_is_not_reported_valid(position, code):
    _, result = generated(HDTRate.HDT2)
    artifact = result.packet_bits
    bits = artifact.bits.copy()
    bits[position] ^= 1
    decoded = analyze_packet(PacketDecodeInput(
        bits, representation=artifact.representation, protocol_hint=artifact.protocol_id,
    ))
    assert code in {issue.code for issue in decoded.issues}


@pytest.mark.parametrize("count", (0, 20, 56, 57, 100))
def test_hdt_truncation_is_incomplete(count):
    _, result = generated(HDTRate.HDT3)
    decoded = analyze_packet(PacketDecodeInput(
        result.packet_bits.bits[:count], representation=BitRepresentation.LOGICAL,
        protocol_hint="bluetooth.hdt",
    ))
    assert not decoded.integrity.complete
    assert decoded.issues


@pytest.mark.parametrize("start,stop,code", (
    (20, 23, "unsupported_rate_indicator"),
    (19, 20, "unsupported_packet_format"),
))
def test_hdt_unsupported_headers_are_not_guessed_from_phy_hint(start, stop, code):
    _, result = generated(HDTRate.HDT3)
    bits = result.packet_bits.bits.copy()
    bits[start:stop] = 1
    decoded = analyze_packet(PacketDecodeInput(
        bits, representation=BitRepresentation.LOGICAL,
        protocol_hint="bluetooth.hdt", phy_hint="HDT3",
    ))
    assert not decoded.integrity.complete
    assert code in {issue.code for issue in decoded.issues}


def test_hdt_does_not_confuse_coded_air_bits_with_logical_packet_bits():
    _, result = generated(HDTRate.HDT6)
    decoded = analyze_packet(PacketDecodeInput(
        result.metadata["coded_payload_bits"], protocol_hint="bluetooth.hdt",
    ))
    assert not decoded.integrity.complete
    assert decoded.issues[0].code == "unsupported_bit_representation"


@pytest.mark.parametrize("rate", (HDTRate.HDT2, HDTRate.HDT3))
def test_qpsk_termination_continues_symbol_parity(rate):
    _, result = generated(rate, 2)
    trace = result.constellation_traces[-1]
    preceding = trace.symbols.size - 2
    expected = map_hdt_symbols(np.zeros(4, dtype=np.uint8), rate, symbol_offset=preceding)
    np.testing.assert_array_equal(trace.symbols[-2:], expected)


def test_hdt_prbs15_and_configuration_round_trip():
    project, result = generated(HDTRate.HDT6, 64, HDTPayloadSourceKind.PRBS15)
    assert project_from_dict(project_to_dict(project)) == project
    decoded = analyze_generation_result(result)
    assert decoded.integrity.crc_valid
    assert result.metadata["payload_bits"][:15].tolist() == [1] * 15
    assert "RF Test" in result.metadata["packet_name"]


def test_custom_hdt_configuration_is_not_labelled_rf_test():
    _, result = generated(HDTRate.HDT6, 16, HDTPayloadSourceKind.PATTERN)
    assert "Custom Format 0" in result.metadata["packet_name"]
    assert analyze_generation_result(result).integrity.crc_valid


def test_hdt_frequency_plan_uses_le_two_mhz_channels():
    from pluto_vsg.ui.frequency_settings import default_frequency_selection, FrequencySettingsDialog
    import pyqtgraph as pg
    pg.mkQApp()
    project = bluetooth_hdt_project()
    state = default_frequency_selection(project)
    assert state.plan_id == "bluetooth_le"
    dialog = FrequencySettingsDialog(project, selection=state)
    try:
        assert dialog.carrier_combo.count() == 40
        frequencies = sorted(dialog.carrier_combo.itemData(i) for i in range(40))
        np.testing.assert_array_equal(np.diff(frequencies), np.full(39, 2e6))
    finally:
        dialog.close()


def test_hdt_settings_shows_format_scope_and_prbs15():
    from pluto_vsg.ui.main_window import _BluetoothHDTSettingsDialog
    import pyqtgraph as pg
    pg.mkQApp()
    dialog = _BluetoothHDTSettingsDialog(bluetooth_hdt_project())
    try:
        assert dialog.length_spin.minimum() == 1
        assert dialog.length_spin.maximum() == 510
        index = dialog.source_combo.findData(HDTPayloadSourceKind.PRBS15)
        assert index >= 0
        dialog.source_combo.setCurrentIndex(index)
        assert not dialog.pattern_edit.isEnabled()
        assert "RF Test" in dialog.packet_profile_label.text()
        dialog.rolloff_spin.setValue(0.5)
        assert "Custom Format 0" in dialog.packet_profile_label.text()
    finally:
        dialog.close()


@pytest.mark.parametrize("length", (0, 511))
def test_hdt_format0_rf_test_rejects_out_of_range(length):
    base = bluetooth_hdt_project()
    settings = replace(base.bluetooth_hdt, payload_length_bytes=length)
    assert any(issue.path == "bluetooth_hdt.payload_length_bytes"
               for issue in validate_project(replace(base, bluetooth_hdt=settings)))


@pytest.mark.parametrize("rate", tuple(HDTRate))
@pytest.mark.parametrize("source", (HDTPayloadSourceKind.PRBS9, HDTPayloadSourceKind.PRBS15))
def test_generated_hdt_iq_and_vsg_decode_agree(rate, source):
    _, result = generated(rate, 73, source)
    recording = IQRecording(
        iq=result.iq, sample_rate_hz=result.sample_rate_hz,
        center_frequency_hz=2_440_000_000.0, source="HDT shared decode test",
    )
    vsa = analyze_bluetooth_hdt_recording(
        recording, profile=BluetoothAnalysisProfile.RF_PHY_TEST,
        _include_generic_visualization=False,
    ).packet
    vsg = analyze_generation_result(result)
    assert vsa.integrity.hec_valid and vsa.integrity.crc_valid
    np.testing.assert_array_equal(vsa.raw_bits, vsg.raw_bits)
