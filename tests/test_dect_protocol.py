import numpy as np
import pytest

from pluto_protocol import FieldStatus, PacketDecodeInput, analyze_packet
from pluto_protocol.dect import dect_p_range, r_crc_bits, r_crc_valid
from pluto_protocol.dect.classic import PP_S_FIELD, RFP_S_FIELD


def _bits(value: int, width: int) -> np.ndarray:
    return np.asarray([int(bit) for bit in f"{value:0{width}b}"], dtype=np.uint8)


def _packet(*, direction="RFP", ta=0b000, ba=0b000, tail=None, prolonged=False, packet_type="P00", bearer_request=False):
    sync = RFP_S_FIELD if direction == "RFP" else PP_S_FIELD
    header = np.concatenate((_bits(ta, 3), _bits(0, 1), _bits(ba, 3), _bits(0, 1)))
    tail_bits = np.zeros(40, dtype=np.uint8) if tail is None else np.asarray(tail, dtype=np.uint8)
    data = np.concatenate((header, tail_bits))
    physical = np.concatenate((sync, data, r_crc_bits(data)))
    if prolonged:
        physical = np.concatenate((sync[:16], physical))
    context = {
        "direction": direction,
        "packet_type": packet_type,
        "preamble_mode": "Prolonged" if prolonged else "Normal",
        "p0_internal_bit": 16 if prolonged else 0,
        "bearer_request": bearer_request,
    }
    return analyze_packet(PacketDecodeInput(physical, protocol_hint="dect.classic", context=context))


def _find(result, field_id):
    pending = list(result.root_fields)
    while pending:
        field = pending.pop(0)
        if field.field_id == field_id:
            return field
        pending[0:0] = list(field.children)
    raise AssertionError(f"field not found: {field_id}")


def test_normal_and_prolonged_internal_ranges_map_to_fixed_p_symbols():
    normal = _packet()
    prolonged = _packet(prolonged=True)
    assert (_find(normal, "preamble").start_bit, _find(normal, "a_field").start_bit) == (0, 32)
    assert (_find(prolonged, "preamble").start_bit, _find(prolonged, "a_field").start_bit) == (16, 48)
    assert dect_p_range(0, 16, 16) == (-16, 0)
    assert dect_p_range(16, 32, 16) == (0, 16)
    assert dect_p_range(48, 112, 16) == (32, 96)


@pytest.mark.parametrize("direction,expected", (("RFP", "1110100110001010"), ("PP", "0001011001110101")))
def test_direction_specific_sync_word_decode(direction, expected):
    result = _packet(direction=direction)
    sync = _find(result, "sync_word")
    assert sync.value == expected
    assert sync.status == FieldStatus.VALID


@pytest.mark.parametrize("ta,tail_id", ((0b011, "rfpi"), (0b100, "qh"), (0b110, "mt_family")))
def test_tail_identification_dispatches_to_decoder(ta, tail_id):
    assert _find(_packet(ta=ta), tail_id)


def test_conditional_ta_010_is_not_over_interpreted():
    ta = _find(_packet(direction="PP", ta=0b010, ba=0b000), "ta")
    assert ta.status == FieldStatus.UNKNOWN
    assert "Conditional" in ta.value


@pytest.mark.parametrize(
    "direction,ba,expected",
    (
        ("RFP", 0b111, "DummyPointer"),
        ("RFP", 0b000, "Connectionless"),
        ("PP", 0b111, "ULE NT"),
    ),
)
def test_ta_010_uses_direction_and_ba_when_the_meaning_is_defined(direction, ba, expected):
    ta = _find(_packet(direction=direction, ta=0b010, ba=ba), "ta")
    assert ta.status == FieldStatus.INFO
    assert expected in ta.value


def test_combined_ta_101_uses_one_a3_a7_field():
    result = _packet(ta=0b101)
    combined = _find(result, "combined_a3_a7")
    assert combined.value.endswith("Escape")
    with pytest.raises(AssertionError):
        _find(result, "q1_bck")


def test_ta_111_is_paging_for_rfp_and_mt_for_pp():
    assert "PT" in _find(_packet(direction="RFP", ta=0b111), "tail").value
    assert _find(_packet(direction="PP", ta=0b111), "mt_family")


def test_ba_uses_normal_or_bearer_request_context():
    assert "E-type" in _find(_packet(ba=0b101), "ba").value
    assert "Long slot" in _find(_packet(ba=0b101, bearer_request=True), "ba").value


def test_class_a_rfpi_is_decoded_msb_first():
    tail = np.concatenate((_bits(1, 1), _bits(0, 3), _bits(0x9F15, 16), _bits(0x12345, 17), _bits(5, 3)))
    result = _packet(ta=0b011, tail=tail)
    assert _find(result, "ari_class").value == "A"
    assert _find(result, "emc").value == 0x9F15
    assert _find(result, "fpn").value == 0x12345
    assert _find(result, "rpn").value == 5


def test_qt_static_system_information_subfields_are_exposed():
    tail = np.zeros(40, dtype=np.uint8)
    tail[0:4] = _bits(0, 4)
    tail[4:8] = _bits(7, 4)
    tail[8:10] = _bits(2, 2)
    tail[11] = 1
    tail[14] = 1
    tail[26:32] = _bits(9, 6)
    tail[34:40] = _bits(3, 6)
    result = _packet(ta=0b100, tail=tail)
    assert "Static System Information" in _find(result, "qh").value
    assert _find(result, "slot_number").value == 7
    assert _find(result, "start_position").value == 2
    assert _find(result, "carrier_number").value == 9
    assert _find(result, "pscn").value == 3


def test_advanced_access_request_decodes_command_fmid_and_pmid():
    tail = np.concatenate((_bits(1, 4), _bits(0, 4), _bits(0xABC, 12), _bits(0x54321, 20)))
    result = _packet(ta=0b110, tail=tail)
    assert "ACCESS_REQUEST" in _find(result, "mt_command").value
    assert _find(result, "fmid").value == 0xABC
    assert _find(result, "pmid").value == 0x54321


# Captured from an independently generated real DECT RFP P32Z transmission.
# Its A-field and format-selected B/X bits provide external known-good vectors.
REAL_RFP_P32Z = np.asarray([int(bit) for bit in (
    "1010101010101010"
    + "10101010101010101110100110001010"
    + "0110000101110000011100110110111001100011011000110011010011110000"
    + "10100101" * 40
    + "00000000"
)], dtype=np.uint8)


def test_real_capture_r_crc_known_good_and_one_bit_error_fails():
    codeword = REAL_RFP_P32Z[48:112]
    assert "".join(map(str, codeword[48:].tolist())) == "0011010011110000"
    assert r_crc_valid(codeword)
    broken = codeword.copy()
    broken[9] ^= 1
    assert not r_crc_valid(broken)


def test_real_capture_x_crc_and_z_repeat_checks():
    result = analyze_packet(PacketDecodeInput(REAL_RFP_P32Z, protocol_hint="dect.classic", context={"direction": "RFP", "packet_type": "P32Z", "p0_internal_bit": 16, "preamble_mode": "Prolonged"}))
    assert _find(result, "x_field").status == FieldStatus.VALID
    assert _find(result, "z_field").status == FieldStatus.VALID
    broken = REAL_RFP_P32Z.copy()
    broken[-1] ^= 1
    failed = analyze_packet(PacketDecodeInput(broken, protocol_hint="dect.classic", context={"direction": "RFP", "packet_type": "P32Z", "p0_internal_bit": 16, "preamble_mode": "Prolonged"}))
    assert _find(failed, "z_field").status == FieldStatus.INVALID


def test_truncated_and_reserved_packets_return_partial_results():
    truncated = _packet().raw_bits[:-9]
    result = analyze_packet(PacketDecodeInput(truncated, protocol_hint="dect.classic", context={"direction": "RFP", "packet_type": "P00"}))
    assert result.integrity.complete is False
    assert _find(result, "r_crc").status == FieldStatus.WARNING
    reserved_qh = np.concatenate((_bits(0xF, 4), np.zeros(36, dtype=np.uint8)))
    assert _find(_packet(ta=0b100, tail=reserved_qh), "qh").status == FieldStatus.WARNING
