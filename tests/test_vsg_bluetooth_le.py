from dataclasses import replace

import numpy as np

from pluto_vsg.engine import BluetoothLEWaveformEngine
from pluto_vsg.engine.bluetooth_le import le_crc24_bits, le_test_payload_bits
from pluto_vsg.model import (
    BluetoothLEPayloadType,
    BluetoothLEPayloadSourceKind,
    BluetoothLEPhy,
    bluetooth_le_payload_code,
    validate_project,
)
from pluto_vsg.persistence import project_from_dict, project_to_dict
from pluto_vsg.profiles import (
    apply_bluetooth_le_rf_test_preset,
    bluetooth_le_fields,
    bluetooth_le_project,
    bluetooth_le_test_project,
)


def _text(bits: np.ndarray) -> str:
    return "".join(str(int(bit)) for bit in bits)


def test_le_profiles_have_specified_rates_and_fields() -> None:
    le1m = bluetooth_le_test_project(BluetoothLEPhy.LE_1M)
    le2m = bluetooth_le_test_project(BluetoothLEPhy.LE_2M)

    assert validate_project(le1m) == ()
    assert validate_project(le2m) == ()
    assert le1m.sample_rate_hz == 8_000_000.0
    assert le2m.sample_rate_hz == 16_000_000.0
    assert le1m.samples_per_symbol == le2m.samples_per_symbol == 8
    assert le1m.center_frequency_hz == le2m.center_frequency_hz == 2_440_000_000.0
    assert [field.name for field in le1m.fields] == [
        "Preamble",
        "Access Address / Sync Word",
        "PDU Header",
        "PDU Length",
        "PDU Payload",
        "CRC",
    ]


def test_le_test_packet_known_air_order_and_lengths() -> None:
    engine = BluetoothLEWaveformEngine()
    le1m = engine.generate(bluetooth_le_test_project(BluetoothLEPhy.LE_1M))
    le2m = engine.generate(bluetooth_le_test_project(BluetoothLEPhy.LE_2M))

    assert _text(le1m.metadata["preamble_bits"]) == "10101010"
    assert _text(le2m.metadata["preamble_bits"]) == "1010101010101010"
    assert _text(le1m.metadata["sync_word_bits"]) == (
        "10010100100000100110111010001110"
    )
    assert len(le1m.metadata["packet_bits"]) == 80 + 37 * 8
    assert len(le2m.metadata["packet_bits"]) == 88 + 37 * 8
    assert le1m.metadata["whitening_enabled"] is False
    assert le1m.metadata["crc_init"] == 0x555555


def test_le_header_example_and_payload_type_codes() -> None:
    base = bluetooth_le_test_project()
    settings = apply_bluetooth_le_rf_test_preset(
        base.bluetooth_le,
        payload_type=BluetoothLEPayloadType.F0,
        payload_length_bytes=37,
    )
    project = replace(base, bluetooth_le=settings, fields=bluetooth_le_fields(settings))
    result = BluetoothLEWaveformEngine().generate(project)

    # Bluetooth Core RF Test Modes example, both fields in transmission order.
    assert _text(result.metadata["pdu_header_bits"]) == "10000000"
    assert _text(result.metadata["pdu_length_bits"]) == "10100100"
    assert [
        bluetooth_le_payload_code(payload_type) for payload_type in BluetoothLEPayloadType
    ] == list(range(8))


def test_le_payload_sources_match_defined_sequences() -> None:
    assert _text(le_test_payload_bits(BluetoothLEPayloadType.F0, 2)) == "1111000011110000"
    assert _text(le_test_payload_bits(BluetoothLEPayloadType.AA, 2)) == "1010101010101010"
    assert _text(le_test_payload_bits(BluetoothLEPayloadType.OF, 1)) == "00001111"
    assert _text(le_test_payload_bits(BluetoothLEPayloadType.FIVE, 1)) == "01010101"
    assert _text(le_test_payload_bits(BluetoothLEPayloadType.PRBS9, 3)[:20]) == (
        "11111111100000111101"
    )
    assert _text(le_test_payload_bits(BluetoothLEPayloadType.PRBS15, 3)[:20]) == (
        "11111111111111100000"
    )


def test_le_crc_matches_direct_test_mode_packet() -> None:
    base = bluetooth_le_test_project(BluetoothLEPhy.LE_1M)
    settings = apply_bluetooth_le_rf_test_preset(
        base.bluetooth_le,
        payload_type=BluetoothLEPayloadType.F0,
        payload_length_bytes=37,
    )
    project = replace(base, bluetooth_le=settings, fields=bluetooth_le_fields(settings))

    result = BluetoothLEWaveformEngine().generate(project)

    assert _text(result.metadata["pdu_header_bits"]) == "10000000"
    assert _text(result.metadata["pdu_length_bits"]) == "10100100"
    assert _text(result.metadata["crc_bits"]) == "001001010011101001000101"


def test_le_crc_matches_bluetooth_sig_reference_packet() -> None:
    # Core Vol 6, Part C sample packet. Bytes are listed in transmission
    # order and bits inside each byte are transmitted LSB first.
    pdu = bytes.fromhex("0003424c45")
    pdu_bits = np.asarray(
        [(byte >> bit) & 1 for byte in pdu for bit in range(8)],
        dtype=np.uint8,
    )

    crc_bits = le_crc24_bits(pdu_bits, init=0x555555)
    crc_bytes = bytes(
        sum(int(crc_bits[start + bit]) << bit for bit in range(8))
        for start in range(0, 24, 8)
    )

    assert crc_bytes == bytes.fromhex("290ace")


def test_le1m_and_le2m_use_the_same_crc_for_the_same_pdu() -> None:
    engine = BluetoothLEWaveformEngine()

    le1m = engine.generate(bluetooth_le_test_project(BluetoothLEPhy.LE_1M))
    le2m = engine.generate(bluetooth_le_test_project(BluetoothLEPhy.LE_2M))

    np.testing.assert_array_equal(le1m.metadata["crc_bits"], le2m.metadata["crc_bits"])


def test_le_crc_rejects_invalid_inputs() -> None:
    with np.testing.assert_raises(ValueError):
        le_crc24_bits(np.asarray([[0, 1]], dtype=np.uint8))
    with np.testing.assert_raises(ValueError):
        le_crc24_bits(np.asarray([0, 2], dtype=np.uint8))
    with np.testing.assert_raises(ValueError):
        le_crc24_bits(np.asarray([0, 1], dtype=np.uint8), init=0x1000000)


def test_le_packet_interval_and_repeat_layout() -> None:
    base = bluetooth_le_test_project(BluetoothLEPhy.LE_1M)
    project = replace(base, repeat_count=3)
    result = BluetoothLEWaveformEngine().generate(project)

    assert result.metadata["test_packet_interval_us"] == 625.0
    assert result.iq.size == 3 * 5000
    starts = [item[0] for item in result.metadata["packet_ranges_samples"]]
    assert starts[1] - starts[0] == starts[2] - starts[1] == 5000


def test_zero_length_le_payload_is_valid() -> None:
    base = bluetooth_le_test_project()
    settings = replace(base.bluetooth_le, payload_length_bytes=0)
    project = replace(base, bluetooth_le=settings, fields=bluetooth_le_fields(settings))

    assert validate_project(project) == ()
    assert "PDU Payload" not in [field.name for field in project.fields]
    result = BluetoothLEWaveformEngine().generate(project)
    assert result.metadata["payload_bits"].size == 0


def test_le_project_json_round_trip() -> None:
    expected = bluetooth_le_test_project(BluetoothLEPhy.LE_2M)

    actual = project_from_dict(project_to_dict(expected))

    assert actual == expected


def test_generic_le_project_keeps_all_packet_fields_editable() -> None:
    base = bluetooth_le_project(BluetoothLEPhy.LE_2M)
    settings = replace(
        base.bluetooth_le,
        preamble_bits="0101010101010101",
        sync_word_bits="10110011100011110000111101010101",
        pdu_header_bits="11001010",
        payload_source=BluetoothLEPayloadSourceKind.PATTERN,
        payload_pattern="11001",
        payload_length_bytes=3,
        crc_init=0x123456,
        whitening_enabled=False,
        rf_test_interval_enabled=False,
    )
    project = replace(base, bluetooth_le=settings, fields=bluetooth_le_fields(settings))

    result = BluetoothLEWaveformEngine().generate(project)

    assert _text(result.metadata["preamble_bits"]) == settings.preamble_bits
    assert _text(result.metadata["sync_word_bits"]) == settings.sync_word_bits
    assert _text(result.metadata["pdu_header_bits"]) == settings.pdu_header_bits
    assert _text(result.metadata["payload_bits"]) == "110011100111001110011100"
    assert result.metadata["crc_init"] == 0x123456
    assert result.metadata["test_packet_interval_us"] is None


def test_le_rf_test_preset_populates_generic_settings() -> None:
    generic = bluetooth_le_project(BluetoothLEPhy.LE_2M)

    settings = apply_bluetooth_le_rf_test_preset(
        generic.bluetooth_le,
        payload_type=BluetoothLEPayloadType.PRBS15,
        payload_length_bytes=255,
    )

    assert settings.sync_word_bits == "10010100100000100110111010001110"
    assert settings.pdu_header_bits == "11000000"
    assert settings.payload_source == BluetoothLEPayloadSourceKind.PRBS15
    assert settings.crc_init == 0x555555
    assert settings.whitening_enabled is False
    assert settings.rf_test_interval_enabled is True
