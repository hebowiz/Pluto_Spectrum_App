from dataclasses import replace

import numpy as np

from pluto_protocol import (
    PacketDecodeInput, PacketSourceInfo, analyze_packet, packet_table_rows,
)
from pluto_sa.vsa.protocol import analyze_demodulated_packet_bits
from pluto_vsg.engine import BluetoothBRWaveformEngine, BluetoothLEWaveformEngine
from pluto_vsg.model import BluetoothLEPhy, BluetoothPacketKind
from pluto_vsg.profiles import (
    bluetooth_br_edr_project,
    bluetooth_br_fields,
    bluetooth_le_project,
)
from pluto_vsg.protocol import analyze_generation_result


def test_br_generator_exposes_and_decodes_exact_air_bits() -> None:
    generated = BluetoothBRWaveformEngine().generate(bluetooth_br_edr_project())

    assert generated.packet_bits is not None
    np.testing.assert_array_equal(generated.packet_bits.bits, generated.metadata["packet_bits"])
    result = analyze_generation_result(generated)

    assert result.protocol_id == "bluetooth.br_edr"
    assert result.packet_type == "DH1"
    assert result.integrity.hec_valid is True
    assert result.integrity.crc_valid is True
    assert result.integrity.complete is True
    assert result.source.source_kind == "vsg_generated"
    assert any(field.field_id == "payload" for field in result.root_fields)
    rows = packet_table_rows(result)
    assert any(row.path == "header.arqn" and row.display_value == "0" for row in rows)
    assert any(row.path == "payload.payload_body" for row in rows)


def test_edr_generator_uses_the_same_shared_decoder() -> None:
    base = bluetooth_br_edr_project()
    settings = replace(
        base.bluetooth_br,
        packet_kind=BluetoothPacketKind.DH1_2,
        payload_length_bytes=54,
    )
    project = replace(base, bluetooth_br=settings, fields=bluetooth_br_fields(settings))
    generated = BluetoothBRWaveformEngine().generate(project)

    result = analyze_generation_result(generated)

    assert result.packet_type == "2-DH1"
    assert result.phy_name == "EDR 2M"
    assert result.integrity.hec_valid is True
    assert result.integrity.crc_valid is True


def test_every_supported_br_edr_dhx_kind_round_trips() -> None:
    base = bluetooth_br_edr_project()
    engine = BluetoothBRWaveformEngine()

    for packet_kind in BluetoothPacketKind:
        settings = replace(
            base.bluetooth_br,
            packet_kind=packet_kind,
            payload_length_bytes=min(
                base.bluetooth_br.payload_length_bytes,
                27 if packet_kind == BluetoothPacketKind.DH1 else 31,
            ),
        )
        project = replace(
            base,
            bluetooth_br=settings,
            fields=bluetooth_br_fields(settings),
        )

        result = analyze_generation_result(engine.generate(project))

        assert result.packet_type == packet_kind.value
        assert result.integrity.hec_valid is True
        assert result.integrity.crc_valid is True


def test_le_generator_exposes_and_decodes_exact_air_bits() -> None:
    generated = BluetoothLEWaveformEngine().generate(
        bluetooth_le_project(BluetoothLEPhy.LE_2M)
    )

    result = analyze_generation_result(generated)

    assert result.protocol_id == "bluetooth.le"
    assert result.phy_name == "LE 2M"
    assert result.integrity.crc_valid is True
    assert result.integrity.complete is True


def test_truncated_packet_is_preserved_as_partial_result() -> None:
    generated = BluetoothLEWaveformEngine().generate(
        bluetooth_le_project(BluetoothLEPhy.LE_1M)
    )
    artifact = generated.packet_bits
    assert artifact is not None

    result = analyze_packet(
        PacketDecodeInput(
            artifact.bits[:-11],
            protocol_hint=artifact.protocol_id,
            phy_hint=artifact.phy_name,
            source=PacketSourceInfo(source_kind="test_truncated"),
            context=artifact.context,
        )
    )

    assert result.integrity.complete is False
    assert result.integrity.crc_valid is None
    assert any(issue.code == "truncated_pdu" for issue in result.issues)


def test_vsa_adapter_returns_same_semantic_result_as_vsg_adapter() -> None:
    generated = BluetoothBRWaveformEngine().generate(bluetooth_br_edr_project())
    artifact = generated.packet_bits
    assert artifact is not None

    result = analyze_demodulated_packet_bits(
        artifact.bits,
        protocol_id=artifact.protocol_id,
        phy_name=artifact.phy_name,
        context=dict(artifact.context),
        packet_index=3,
        center_frequency_hz=2_440_000_000.0,
    )

    assert result.packet_type == "DH1"
    assert result.integrity.crc_valid is True
    assert result.source.source_kind == "vsa_demodulated"
    assert result.source.packet_index == 3
