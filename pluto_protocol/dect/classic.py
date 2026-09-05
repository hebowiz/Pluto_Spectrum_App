"""UI-independent Classic DECT packet decoder.

The decoder consumes the air bits produced by the dedicated demodulator.  Bit
zero is the first actually detected preamble bit; ``p0_internal_bit`` carries
the Normal/Prolonged conversion to ETSI p-symbol numbering.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pluto_protocol.bitops import bits_to_int_msb
from pluto_protocol.dect.common import r_crc_bits, r_crc_valid, x_crc_bits, x_crc_valid
from pluto_protocol.model import (
    DecodeProbeResult,
    FieldStatus,
    IssueSeverity,
    PacketAnalysisResult,
    PacketDecodeInput,
    PacketField,
    PacketIntegritySummary,
    PacketIssue,
    PacketSummaryItem,
)


PROTOCOL_ID = "dect.classic"
RFP_S_FIELD = np.asarray([int(bit) for bit in "10101010101010101110100110001010"], dtype=np.uint8)
PP_S_FIELD = 1 - RFP_S_FIELD

TA_NAMES = {
    0b000: "CT0 / CT data packet number 0",
    0b001: "CT1 / CT data packet number 1",
    0b011: "NT / Identities Information",
    0b100: "QT / Multiframe Synchronization and System Information",
    0b101: "Combined a3-a7 coding",
    0b110: "MT / MAC Layer Control",
}

BA_NAMES = {
    0b000: "U-type: IN/SIN/IP packet 0, or no valid IP error-detect data",
    0b001: "U-type: IP packet 1/SIP, or no valid IN data",
    0b010: "E-type: all CF/CLF, packet 0",
    0b011: "E-type: all CF, packet 1",
    0b100: "E-type: not all CF/CLF; CF packet 0",
    0b101: "E-type: not all CF; CF packet 1",
    0b110: "E+U packet 0, or E-type MAC signalling (service dependent)",
    0b111: "E+U packet 1 or no B-field (service dependent)",
}

BEARER_REQUEST_BA = {
    0b010: "Double slot required",
    0b100: "Half slot required",
    0b101: "Long slot (j=640) required",
    0b110: "Long slot (j=672) required",
}

QH_NAMES = {
    0x0: "Static System Information",
    0x1: "Static System Information",
    0x2: "Extended RF Carriers Part 1",
    0x3: "Fixed Part Capabilities",
    0x4: "Extended Fixed Part Capabilities",
    0x5: "SARI List Contents",
    0x6: "Multi-frame Number",
    0x7: "Escape",
    0x8: "Obsolete",
    0x9: "Extended RF Carriers Part 2",
    0xB: "Transmit Information",
    0xC: "Extended Fixed Part Capabilities Part 2",
    0xD: "Extended Static System Information",
    0xE: "Extended Fixed Part Capabilities Part 3",
    0xF: "Reserved",
}

MT_FAMILIES = {
    0x0: "Basic connection control",
    0x1: "Advanced connection control",
    0x2: "MAC layer test messages",
    0x3: "Quality control",
    0x4: "Broadcast and connectionless services",
    0x5: "Encryption control",
    0x6: "First transmission of a B-field BEARER_REQUEST",
    0x7: "Escape",
    0x8: "TARI message",
    0x9: "REP connection control",
    0xA: "Advanced connection control part 2",
}

CONNECTION_COMMANDS = {
    0x0: "ACCESS_REQUEST",
    0x1: "BEARER_HANDOVER_REQUEST",
    0x2: "CONNECTION_HANDOVER_REQUEST",
    0x3: "UNCONFIRMED_ACCESS_REQUEST",
    0x4: "BEARER_CONFIRM",
    0x5: "WAIT",
    0x6: "ATTRIBUTES_T_REQUEST",
    0x7: "ATTRIBUTES_T_CONFIRM",
    0xF: "RELEASE",
}

ADVANCED_EXTRA_COMMANDS = {
    0x8: "BANDWIDTH_T_REQUEST",
    0x9: "BANDWIDTH_T_CONFIRM",
    0xA: "CHANNEL_LIST",
    0xB: "UNCONFIRMED_DUMMY",
    0xC: "UNCONFIRMED_HANDOVER",
}

PAGE_LENGTHS = {
    0b000: "Zero length page",
    0b001: "Short page",
    0b010: "Full page",
    0b011: "MAC resume and control page",
    0b100: "Long page - intermediate",
    0b101: "Long page - first",
    0b110: "Long page - last",
    0b111: "Long page - first and last",
}


def _text(bits: np.ndarray) -> str:
    return "".join(str(int(bit)) for bit in np.asarray(bits, dtype=np.uint8))


def _hex(bits: np.ndarray) -> str:
    values = np.asarray(bits, dtype=np.uint8)
    if not values.size:
        return ""
    width = (values.size + 3) // 4
    return f"0x{bits_to_int_msb(values):0{width}X}"


def _slice(bits: np.ndarray, start: int, stop: int) -> np.ndarray:
    return np.asarray(bits[max(0, start):max(0, min(stop, bits.size))], dtype=np.uint8)


def _field(
    bits: np.ndarray,
    field_id: str,
    name: str,
    start: int,
    stop: int,
    value=None,
    meaning: str = "",
    status: FieldStatus = FieldStatus.INFO,
    children: tuple[PacketField, ...] = (),
) -> PacketField:
    raw = _slice(bits, start, stop)
    effective_stop = start + raw.size
    if effective_stop < stop and status not in {FieldStatus.INVALID, FieldStatus.WARNING}:
        status = FieldStatus.WARNING
        meaning = f"{meaning}; truncated" if meaning else "Truncated"
    return PacketField(field_id, name, start, effective_stop, raw, value, meaning, status, children)


def _numeric_field(bits, field_id, name, start, stop, meaning="", status=FieldStatus.INFO):
    raw = _slice(bits, start, stop)
    value = bits_to_int_msb(raw) if raw.size == stop - start else _text(raw)
    return _field(bits, field_id, name, start, stop, value, meaning, status)


def _decode_rfpi(bits: np.ndarray, start: int) -> PacketField:
    raw = _slice(bits, start, start + 40)
    if raw.size < 40:
        return _field(bits, "rfpi", "RFPI", start, start + 40, _hex(raw), "Truncated RFPI", FieldStatus.WARNING)
    arc = bits_to_int_msb(raw[1:4])
    class_name = chr(ord("A") + arc) if arc <= 4 else f"Reserved ({arc:03b})"
    class_status = FieldStatus.INFO if arc <= 4 else FieldStatus.WARNING
    rpn_start = {0: 37, 1: 32, 2: 32, 3: 32, 4: 32}.get(arc, 40)
    children: list[PacketField] = [
        _numeric_field(bits, "rfpi_e", "E", start, start + 1, "Secondary ARIs available flag"),
        _field(bits, "ari_class", "ARI Class", start + 1, start + 4, class_name, f"ARC {arc:03b}", class_status),
        _field(bits, "pari", "PARI", start + 1, start + rpn_start, _hex(_slice(bits, start + 1, start + rpn_start)), "Primary Access Rights Identifier"),
    ]
    if arc == 0:
        layout = (("emc", "EMC", 4, 20, "Equipment Manufacturer's Code"), ("fpn", "FPN", 20, 37, "Fixed Part Number"), ("rpn", "RPN", 37, 40, "Radio Fixed Part Number"))
    elif arc == 1:
        layout = (("eic", "EIC", 4, 20, "Equipment Installer's Code"), ("fpn_fps", "FPN + FPS", 20, 32, "Fixed Part Number and sub-number"), ("rpn", "RPN", 32, 40, "Radio Fixed Part Number"))
    elif arc == 2:
        layout = (("poc", "POC", 4, 20, "Public Operator Code"), ("fpn_fps", "FPN + FPS", 20, 32, "Fixed Part Number and sub-number"), ("rpn", "RPN", 32, 40, "Radio Fixed Part Number"))
    elif arc == 3:
        layout = (("gop", "GOP", 4, 24, "GSM/UMTS operator code"), ("fpn", "FPN", 24, 32, "Fixed Part Number"), ("rpn", "RPN", 32, 40, "Radio Fixed Part Number"))
    elif arc == 4:
        layout = (("fill", "FIL", 4, 20, "Fixed 0101 fill pattern"), ("fpn", "FPN", 20, 32, "Fixed Part Number"), ("rpn", "RPN", 32, 40, "Radio Fixed Part Number"))
    else:
        layout = ()
    for field_id, name, lo, hi, meaning in layout:
        field_start = start + lo
        if field_id == "fill":
            fill = _slice(bits, field_start, start + hi)
            valid = _text(fill) == "0101010101010101"
            children.append(_field(bits, field_id, name, field_start, start + hi, _text(fill), meaning, FieldStatus.VALID if valid else FieldStatus.INVALID))
        else:
            children.append(_numeric_field(bits, field_id, name, field_start, start + hi, meaning))
    return _field(bits, "rfpi", "RFPI", start, start + 40, _hex(raw), f"ARI class {class_name}", class_status, tuple(children))


def _decode_static_system_info(bits: np.ndarray, a_start: int) -> tuple[PacketField, ...]:
    def n(field_id, name, lo, hi, meaning="", status=FieldStatus.INFO):
        return _numeric_field(bits, field_id, name, a_start + lo, a_start + hi, meaning, status)

    carriers = _slice(bits, a_start + 22, a_start + 32)
    available = [str(index) for index, value in enumerate(carriers) if value]
    return (
        n("normal_reverse", "Normal / Reverse", 11, 12, "0: normal RFP transmit half-frame; 1: normal PP transmit half-frame"),
        n("slot_number", "Slot Number", 12, 16, "Slot pair number"),
        n("start_position", "Start Position", 16, 18, "00: f0; 10: f240; other values reserved"),
        n("escape_available", "ESC", 18, 19, "Escape QT message available"),
        n("transceiver_count", "Number of Transceivers", 19, 21, "Encoded as 1, 2, 3, or 4 or more"),
        n("extended_rf_available", "Extended RF Carrier Information Available", 21, 22),
        _field(bits, "rf_carriers", "RF Carriers Available", a_start + 22, a_start + 32, ", ".join(available) if available else "None", "Carrier numbers 0 through 9"),
        n("static_spare", "Spare", 32, 34, "Shall be 0", FieldStatus.VALID if not np.any(_slice(bits, a_start + 32, a_start + 34)) else FieldStatus.WARNING),
        n("carrier_number", "Carrier Number", 34, 40),
        n("extended_static_available", "Extended Static System Information Available", 40, 41),
        n("static_spare_2", "Spare", 41, 42, "Shall be 0", FieldStatus.VALID if not np.any(_slice(bits, a_start + 41, a_start + 42)) else FieldStatus.WARNING),
        n("pscn", "PSCN", 42, 48, "Primary receiver scan carrier number"),
    )


def _decode_qt(bits: np.ndarray, a_start: int) -> PacketField:
    tail_start = a_start + 8
    qh_bits = _slice(bits, tail_start, tail_start + 4)
    if qh_bits.size < 4:
        return _field(bits, "tail", "Tail", tail_start, a_start + 48, _text(qh_bits), "Truncated QT", FieldStatus.WARNING)
    qh = bits_to_int_msb(qh_bits)
    name = QH_NAMES.get(qh, "Reserved")
    qh_status = FieldStatus.WARNING if qh in {0x8, 0xA, 0xF} else FieldStatus.INFO
    children: list[PacketField] = [
        _field(bits, "qh", "QH", tail_start, tail_start + 4, f"0x{qh:X} / {name}", name, qh_status)
    ]
    if qh in {0x0, 0x1}:
        children.extend(_decode_static_system_info(bits, a_start))
    elif qh == 0x6:
        children.append(_numeric_field(bits, "multiframe_number", "Multi-frame Number", a_start + 24, a_start + 48, "Modulo 2^24"))
    elif qh == 0xB:
        children.extend((
            _numeric_field(bits, "tx_type", "TX Type", a_start + 12, a_start + 16, "0000 maximum transmit power; 0101 bandwidth-dependent maximum"),
            _numeric_field(bits, "power_level", "Power Level", a_start + 16, a_start + 24, "Binary encoded mW"),
            _field(bits, "tx_spare", "Spare", a_start + 24, a_start + 48, _hex(_slice(bits, a_start + 24, a_start + 48)), "Shall be 0", FieldStatus.VALID if not np.any(_slice(bits, a_start + 24, a_start + 48)) else FieldStatus.WARNING),
        ))
    elif qh == 0xD:
        children.extend((
            _field(bits, "extended_static_spare", "Spare", a_start + 12, a_start + 44, _hex(_slice(bits, a_start + 12, a_start + 44)), "Shall be 0", FieldStatus.VALID if not np.any(_slice(bits, a_start + 12, a_start + 44)) else FieldStatus.WARNING),
            _numeric_field(bits, "rfp_slot_scheme", "RFP Slot Scheme", a_start + 44, a_start + 48),
        ))
    else:
        children.append(_field(bits, "system_information", "System Information", a_start + 12, a_start + 48, _hex(_slice(bits, a_start + 12, a_start + 48)), "Subtype fields not decoded", FieldStatus.UNKNOWN))
    return _field(bits, "tail", "Tail", tail_start, a_start + 48, f"QT / {name}", name, qh_status, tuple(children))


def _decode_mt(bits: np.ndarray, a_start: int) -> tuple[PacketField, bool]:
    tail_start = a_start + 8
    family_bits = _slice(bits, tail_start, tail_start + 4)
    if family_bits.size < 4:
        return _field(bits, "tail", "Tail", tail_start, a_start + 48, _text(family_bits), "Truncated MT", FieldStatus.WARNING), False
    family = bits_to_int_msb(family_bits)
    family_name = MT_FAMILIES.get(family, "Reserved")
    family_status = FieldStatus.WARNING if family > 0xA else FieldStatus.INFO
    children: list[PacketField] = [
        _field(bits, "mt_family", "Message Family", tail_start, tail_start + 4, f"0x{family:X} / {family_name}", family_name, family_status)
    ]
    bearer_request = family == 0x6
    command_name = "Not decoded"
    if family in {0x0, 0x1} and bits.size >= tail_start + 8:
        command_bits = _slice(bits, tail_start + 4, tail_start + 8)
        command = bits_to_int_msb(command_bits)
        commands = dict(CONNECTION_COMMANDS)
        if family == 0x1:
            commands.update(ADVANCED_EXTRA_COMMANDS)
        command_name = commands.get(command, "Reserved")
        command_status = FieldStatus.WARNING if command_name == "Reserved" else FieldStatus.INFO
        children.append(_field(bits, "mt_command", "Command", tail_start + 4, tail_start + 8, f"0x{command:X} / {command_name}", command_name, command_status))
        if command_name in {"ACCESS_REQUEST", "BEARER_HANDOVER_REQUEST", "CONNECTION_HANDOVER_REQUEST", "UNCONFIRMED_ACCESS_REQUEST", "BEARER_CONFIRM", "WAIT"}:
            children.extend((
                _numeric_field(bits, "fmid", "FMID", a_start + 16, a_start + 28, "Fixed MAC Identity"),
                _numeric_field(bits, "pmid", "PMID", a_start + 28, a_start + 48, "Portable MAC Identity"),
            ))
    else:
        children.append(_field(bits, "mt_parameters", "Raw Parameters", tail_start + 4, a_start + 48, _hex(_slice(bits, tail_start + 4, a_start + 48)), "Message-specific decode not implemented", FieldStatus.UNKNOWN))
    return _field(bits, "tail", "Tail", tail_start, a_start + 48, f"MT / {family_name} / {command_name}", family_name, family_status, tuple(children)), bearer_request


def _decode_pt(bits: np.ndarray, a_start: int) -> PacketField:
    start = a_start + 8
    length_bits = _slice(bits, start + 1, start + 4)
    length_code = bits_to_int_msb(length_bits) if length_bits.size == 3 else None
    page_name = PAGE_LENGTHS.get(length_code, "Truncated")
    children: list[PacketField] = [
        _numeric_field(bits, "page_extend", "Extend Flag", start, start + 1),
        _field(bits, "page_length", "BS SDU Length", start + 1, start + 4, page_name, f"Code {_text(length_bits)}", FieldStatus.INFO if length_code is not None else FieldStatus.WARNING),
    ]
    if length_code == 0b000:
        children.append(_field(bits, "rfpi_low", "Least-significant RFPI", a_start + 12, a_start + 32, _hex(_slice(bits, a_start + 12, a_start + 32)), "20 least-significant RFPI bits"))
    elif length_code == 0b001:
        children.extend((
            _field(bits, "bs_data", "BS Channel Data", a_start + 12, a_start + 32, _hex(_slice(bits, a_start + 12, a_start + 32))),
            _numeric_field(bits, "page_info_type", "Information Type", a_start + 32, a_start + 36),
            _field(bits, "page_information", "MAC Layer Information", a_start + 36, a_start + 48, _hex(_slice(bits, a_start + 36, a_start + 48))),
        ))
    elif length_code == 0b011:
        children.extend((
            _numeric_field(bits, "page_pmid", "PMID", a_start + 12, a_start + 32, "Portable MAC Identity"),
            _numeric_field(bits, "page_ecn", "ECN / Info 3", a_start + 32, a_start + 36),
            _numeric_field(bits, "page_command", "Command", a_start + 36, a_start + 38),
            _numeric_field(bits, "page_info_1", "Info 1", a_start + 38, a_start + 42),
            _numeric_field(bits, "page_info_2", "Info 2", a_start + 42, a_start + 48),
        ))
    else:
        children.append(_field(bits, "page_data", "Page Data", a_start + 12, a_start + 48, _hex(_slice(bits, a_start + 12, a_start + 48)), "Format-specific payload"))
    return _field(bits, "tail", "Tail", start, a_start + 48, f"PT / {page_name}", "Paging tail", children=tuple(children))


def _decode_tail(bits: np.ndarray, a_start: int, ta: int, direction: str) -> tuple[PacketField, bool]:
    start = a_start + 8
    if ta in {0b000, 0b001}:
        name = "CT0" if ta == 0 else "CT1"
        return _field(bits, "tail", "Tail", start, a_start + 48, name, "Encrypted/context dependent; raw tail retained", FieldStatus.UNKNOWN, (
            _field(bits, "encrypted", "Encrypted?", start, start, "Unknown", "Connection and cipher context required", FieldStatus.UNKNOWN),
            _field(bits, "raw_tail", "Raw Tail", start, a_start + 48, _hex(_slice(bits, start, a_start + 48)), "Not decoded", FieldStatus.UNKNOWN),
        )), False
    if ta == 0b010:
        ba_bits = _slice(bits, a_start + 4, a_start + 7)
        ba = bits_to_int_msb(ba_bits) if ba_bits.size == 3 else None
        if direction == "RFP" and ba == 0b111:
            meaning, status = "Nt / Identity Information on DummyPointer bearer", FieldStatus.INFO
        elif direction == "RFP" and ba is not None:
            meaning, status = "NT / Identities Information on connectionless bearer", FieldStatus.INFO
        elif direction == "PP" and ba == 0b111:
            meaning, status = "ULE NT / ULE DummyRequest identity information", FieldStatus.INFO
        else:
            meaning, status = "Conditional NT / ULE NT; BA/bearer context required", FieldStatus.UNKNOWN
        rfpi = _decode_rfpi(bits, start)
        return _field(bits, "tail", "Tail", start, a_start + 48, meaning, meaning, status, (rfpi,)), False
    if ta == 0b011:
        rfpi = _decode_rfpi(bits, start)
        meaning = "NT / Identities Information"
        status = FieldStatus.INFO
        return _field(bits, "tail", "Tail", start, a_start + 48, meaning, meaning, status, (rfpi,)), False
    if ta == 0b100:
        return _decode_qt(bits, a_start), False
    if ta == 0b101:
        return _field(bits, "tail", "Tail", start, a_start + 48, _hex(_slice(bits, start, a_start + 48)), "Combined frame content; proprietary/reserved tail not decoded", FieldStatus.UNKNOWN), False
    if ta == 0b110 or (ta == 0b111 and direction == "PP"):
        return _decode_mt(bits, a_start)
    if ta == 0b111 and direction == "RFP":
        return _decode_pt(bits, a_start), False
    return _field(bits, "tail", "Tail", start, a_start + 48, _hex(_slice(bits, start, a_start + 48)), "Unknown", FieldStatus.UNKNOWN), False


def _header(bits: np.ndarray, a_start: int, ta: int, bearer_request: bool, direction: str) -> PacketField:
    ta_bits = _slice(bits, a_start, a_start + 3)
    if ta == 0b010:
        ba_bits = _slice(bits, a_start + 4, a_start + 7)
        ba = bits_to_int_msb(ba_bits) if ba_bits.size == 3 else None
        if direction == "RFP" and ba == 0b111:
            ta_name, ta_status = "Nt / DummyPointer identity information", FieldStatus.INFO
        elif direction == "RFP" and ba is not None:
            ta_name, ta_status = "NT / Connectionless bearer identities", FieldStatus.INFO
        elif direction == "PP" and ba == 0b111:
            ta_name, ta_status = "ULE NT", FieldStatus.INFO
        else:
            ta_name, ta_status = "Conditional NT / ULE NT (context required)", FieldStatus.UNKNOWN
    elif ta == 0b111:
        ta_name = "PT / Paging Tail" if direction == "RFP" else "MT / First PP Transmission"
        ta_status = FieldStatus.INFO
    else:
        ta_name = TA_NAMES.get(ta, "Reserved")
        ta_status = FieldStatus.WARNING if ta_name == "Reserved" else FieldStatus.INFO
    children: list[PacketField] = [
        _field(bits, "ta", "TA", a_start, a_start + 3, f"{_text(ta_bits)} / {ta_name}", ta_name, ta_status)
    ]
    if ta == 0b101:
        combined_bits = _slice(bits, a_start + 3, a_start + 8)
        combined = bits_to_int_msb(combined_bits) if combined_bits.size == 5 else -1
        meaning = "Escape" if combined == 0 else "Mesh dummy bearer" if combined == 1 else "Reserved"
        children.append(_field(bits, "combined_a3_a7", "Combined a3-a7", a_start + 3, a_start + 8, f"{_text(combined_bits)} / {meaning}", meaning, FieldStatus.WARNING if combined > 1 else FieldStatus.INFO))
    else:
        ba_bits = _slice(bits, a_start + 4, a_start + 7)
        ba = bits_to_int_msb(ba_bits) if ba_bits.size == 3 else 0
        ba_name = BEARER_REQUEST_BA.get(ba, "Full slot required") if bearer_request else BA_NAMES.get(ba, "Unknown")
        children.extend((
            _numeric_field(bits, "q1_bck", "Q1 / BCK", a_start + 3, a_start + 4, "Q1 / BCK (bearer and MAC-service context dependent)", FieldStatus.UNKNOWN),
            _field(bits, "ba", "BA", a_start + 4, a_start + 7, f"{_text(ba_bits)} / {ba_name}", "BEARER_REQUEST slot requirement" if bearer_request else "B-field identification"),
            _numeric_field(bits, "q2", "Q2", a_start + 7, a_start + 8, "Bearer-quality / C-channel flow-control bit; context dependent", FieldStatus.UNKNOWN),
        ))
    return _field(bits, "header", "Header", a_start, a_start + 8, _hex(_slice(bits, a_start, a_start + 8)), "A-field header", children=tuple(children))


def _x_test_data(packet_type: str, b_bits: np.ndarray) -> np.ndarray | None:
    if packet_type in {"P32", "P32Z"} and b_bits.size >= 320:
        indices = [index + 48 * (1 + index // 16) for index in range(80)]
    elif packet_type in {"P80", "P80Z"} and b_bits.size >= 800:
        indices = [index + 64 * (1 + index // 16) for index in range(160)]
    else:
        return None
    return np.asarray(b_bits[indices], dtype=np.uint8)


@dataclass(frozen=True)
class DectClassicDecoder:
    protocol_id: str = PROTOCOL_ID
    protocol_name: str = "Classic DECT"

    def probe(self, packet: PacketDecodeInput) -> DecodeProbeResult:
        confidence = 0.98 if packet.protocol_hint == self.protocol_id else 0.1
        return DecodeProbeResult(self.protocol_id, confidence, "Classic DECT S/A/B/X/Z air-bit layout")

    def decode(self, packet: PacketDecodeInput) -> PacketAnalysisResult:
        bits = packet.bits
        context = packet.context
        direction = str(context.get("direction", "Unknown")).upper()
        p0_internal_bit = int(context.get("p0_internal_bit", 16 if str(context.get("preamble_mode", "Normal")) == "Prolonged" else 0))
        packet_type = str(context.get("packet_type", "Unknown"))
        issues: list[PacketIssue] = []
        children: list[PacketField] = []

        expected = RFP_S_FIELD if direction == "RFP" else PP_S_FIELD if direction == "PP" else None
        if p0_internal_bit:
            raw = _slice(bits, 0, p0_internal_bit)
            valid = expected is not None and raw.size == 16 and np.array_equal(raw, expected[:16])
            children.append(_field(bits, "prolonged_preamble", "Prolonged Preamble", 0, p0_internal_bit, _text(raw), f"{direction} prolonged preamble", FieldStatus.VALID if valid else FieldStatus.INVALID if raw.size == 16 else FieldStatus.WARNING))
        s_start = p0_internal_bit
        preamble = _slice(bits, s_start, s_start + 16)
        sync = _slice(bits, s_start + 16, s_start + 32)
        preamble_valid = expected is not None and preamble.size == 16 and np.array_equal(preamble, expected[:16])
        sync_valid = expected is not None and sync.size == 16 and np.array_equal(sync, expected[16:])
        s_status = FieldStatus.VALID if preamble_valid and sync_valid else FieldStatus.INVALID if preamble.size == sync.size == 16 else FieldStatus.WARNING
        children.append(_field(bits, "s_field", "S-field", s_start, s_start + 32, f"{direction} synchronization", "Direction-specific synchronization field", s_status, (
            _field(bits, "preamble", "Preamble", s_start, s_start + 16, _text(preamble), f"{direction} Preamble", FieldStatus.VALID if preamble_valid else FieldStatus.INVALID if preamble.size == 16 else FieldStatus.WARNING),
            _field(bits, "sync_word", "Packet Synchronization Word", s_start + 16, s_start + 32, _text(sync), f"{direction} Sync Word", FieldStatus.VALID if sync_valid else FieldStatus.INVALID if sync.size == 16 else FieldStatus.WARNING),
        )))

        a_start = s_start + 32
        a_raw = _slice(bits, a_start, a_start + 64)
        crc_valid: bool | None = None
        if a_raw.size:
            ta_bits = a_raw[:min(3, a_raw.size)]
            ta = bits_to_int_msb(ta_bits) if ta_bits.size == 3 else -1
            tail, bearer_request = _decode_tail(bits, a_start, ta, direction)
            bearer_request = bearer_request or bool(context.get("bearer_request", False))
            header = _header(bits, a_start, ta, bearer_request, direction)
            crc_raw = _slice(bits, a_start + 48, a_start + 64)
            if a_raw.size == 64:
                calculated = r_crc_bits(a_raw[:48])
                crc_valid = r_crc_valid(a_raw)
                crc_status = FieldStatus.VALID if crc_valid else FieldStatus.INVALID
                crc_meaning = f"Calculated {_hex(calculated)}; {'PASS' if crc_valid else 'FAIL'}"
                if not crc_valid:
                    issues.append(PacketIssue("r_crc_mismatch", "A-field R-CRC does not match", IssueSeverity.WARNING, a_start + 48, a_start + 64))
            else:
                calculated = np.empty(0, dtype=np.uint8)
                crc_status = FieldStatus.WARNING
                crc_meaning = "Not checked - truncated A-field"
                issues.append(PacketIssue("truncated_a_field", "Packet ends inside the A-field", IssueSeverity.WARNING, a_start, a_start + 64))
            crc_field = _field(bits, "r_crc", "R-CRC", a_start + 48, a_start + 64, _hex(crc_raw), crc_meaning, crc_status, (
                _field(bits, "r_crc_received", "Received", a_start + 48, a_start + 64, _hex(crc_raw)),
                _field(bits, "r_crc_calculated", "Calculated", a_start + 48, a_start + 48, _hex(calculated) if calculated.size else "Not checked", "Expected check bits", FieldStatus.INFO if calculated.size else FieldStatus.UNKNOWN),
                _field(bits, "r_crc_result", "Result", a_start + 48, a_start + 48, "PASS" if crc_valid else "FAIL" if crc_valid is False else "Not checked", "R-CRC validation", crc_status),
            ))
            a_status = FieldStatus.WARNING if a_raw.size < 64 else FieldStatus.VALID if crc_valid else FieldStatus.INVALID
            children.append(_field(bits, "a_field", "A-field", a_start, a_start + 64, "64-bit signalling field", "H (8) + Tail (40) + R-CRC (16)", a_status, (header, tail, crc_field)))

        b_start = a_start + 64
        layout = {"P32": (320, False), "P32Z": (320, True), "P80": (800, False), "P80Z": (800, True)}.get(packet_type)
        x_valid: bool | None = None
        if layout is not None:
            b_size, has_z = layout
            b_raw = _slice(bits, b_start, b_start + b_size)
            ba = bits_to_int_msb(_slice(bits, a_start + 4, a_start + 7)) if bits.size >= a_start + 7 else None
            ba_name = BA_NAMES.get(ba, "Unknown") if ba is not None else "Unknown"
            children.append(_field(bits, "b_field", "B-field", b_start, b_start + b_size, f"{b_raw.size} bits", "Scrambled air bits; payload not decoded", FieldStatus.INFO if b_raw.size == b_size else FieldStatus.WARNING, (
                _field(bits, "b_length", "Length", b_start, b_start, f"{b_raw.size} bits", "Physical B-field length"),
                _field(bits, "b_ba_type", "BA Type", b_start, b_start, ba_name, "Interpretation from A-field BA"),
                _field(bits, "b_raw", "Raw", b_start, b_start + b_size, _hex(b_raw), "Scrambled air bits"),
                _field(bits, "b_decode", "Decode", b_start, b_start, "Not decoded", "B-field descrambling/higher layers are outside this analyzer", FieldStatus.UNKNOWN),
            )))
            x_start = b_start + b_size
            x_raw = _slice(bits, x_start, x_start + 4)
            test_data = _x_test_data(packet_type, b_raw)
            if test_data is not None and x_raw.size == 4:
                calculated_x = x_crc_bits(test_data)
                x_valid = x_crc_valid(np.concatenate((test_data, x_raw)))
                x_status = FieldStatus.VALID if x_valid else FieldStatus.INVALID
                x_meaning = f"Calculated {_text(calculated_x)}; {'PASS' if x_valid else 'FAIL'}"
                if not x_valid:
                    issues.append(PacketIssue("x_crc_mismatch", "X-CRC does not match the format-specific test bits", IssueSeverity.WARNING, x_start, x_start + 4))
            else:
                x_status = FieldStatus.UNKNOWN if x_raw.size == 4 else FieldStatus.WARNING
                x_meaning = "Not checked - packet format or complete test-bit set unavailable"
            children.append(_field(bits, "x_field", "X-field", x_start, x_start + 4, _text(x_raw), x_meaning, x_status, (
                _field(bits, "x_crc_received", "Received", x_start, x_start + 4, _text(x_raw)),
                _field(bits, "x_crc_calculated", "Calculated", x_start, x_start, _text(calculated_x) if test_data is not None and x_raw.size == 4 else "Not checked", "Expected X-CRC", FieldStatus.INFO if test_data is not None and x_raw.size == 4 else FieldStatus.UNKNOWN),
                _field(bits, "x_crc_result", "Result", x_start, x_start, "PASS" if x_valid else "FAIL" if x_valid is False else "Not checked", "Format-specific X-CRC validation", x_status),
            )))
            if has_z:
                z_start = x_start + 4
                z_raw = _slice(bits, z_start, z_start + 4)
                z_valid = z_raw.size == 4 and x_raw.size == 4 and np.array_equal(z_raw, x_raw)
                z_status = FieldStatus.VALID if z_valid else FieldStatus.INVALID if z_raw.size == 4 else FieldStatus.WARNING
                children.append(_field(bits, "z_field", "Z-field", z_start, z_start + 4, _text(z_raw), f"Expected {_text(x_raw)}; X Repeat Check {'PASS' if z_valid else 'FAIL' if z_raw.size == 4 else 'Not checked'}", z_status, (
                    _field(bits, "z_value", "Value", z_start, z_start + 4, _text(z_raw)),
                    _field(bits, "z_expected", "Expected", z_start, z_start, _text(x_raw), "X-field repeat value"),
                    _field(bits, "z_repeat_result", "X Repeat Check", z_start, z_start, "PASS" if z_valid else "FAIL" if z_raw.size == 4 else "Not checked", status=z_status),
                )))
                if z_raw.size == 4 and not z_valid:
                    issues.append(PacketIssue("z_repeat_mismatch", "Z-field is not equal to X-field", IssueSeverity.WARNING, z_start, z_start + 4))
        elif packet_type != "P00" and bits.size > b_start:
            x_start = max(b_start, bits.size - 4)
            b_raw = _slice(bits, b_start, x_start)
            children.extend((
                _field(bits, "b_field", "B-field", b_start, x_start, f"{b_raw.size} bits", "Variable scrambled bearer field; payload not decoded", FieldStatus.UNKNOWN),
                _field(bits, "x_field", "X-field", x_start, bits.size, _text(_slice(bits, x_start, bits.size)), "Not checked - packet format uncertain", FieldStatus.UNKNOWN),
            ))

        complete = bool(packet_type == "P00" and bits.size >= a_start + 64 or layout is not None and bits.size >= b_start + layout[0] + 4 + (4 if layout[1] else 0))
        root_status = FieldStatus.WARNING if not complete else FieldStatus.INVALID if crc_valid is False or x_valid is False else FieldStatus.VALID
        root = _field(bits, "dect_packet", "DECT Packet", 0, bits.size, packet_type, f"{direction}; p0 internal bit {p0_internal_bit}", root_status, tuple(children))
        summary = (
            PacketSummaryItem("protocol", "Protocol", self.protocol_name, self.protocol_name),
            PacketSummaryItem("direction", "Direction", direction, direction),
            PacketSummaryItem("packet_type", "Packet Type", packet_type, packet_type),
            PacketSummaryItem("r_crc", "R-CRC", crc_valid, "Not checked" if crc_valid is None else "PASS" if crc_valid else "FAIL", FieldStatus.UNKNOWN if crc_valid is None else FieldStatus.VALID if crc_valid else FieldStatus.INVALID),
            PacketSummaryItem("x_crc", "X-CRC", x_valid, "Not checked" if x_valid is None else "PASS" if x_valid else "FAIL", FieldStatus.UNKNOWN if x_valid is None else FieldStatus.VALID if x_valid else FieldStatus.INVALID),
        )
        return PacketAnalysisResult("1.0", self.protocol_id, self.protocol_name, "2-level GFSK", packet_type, summary, (root,), tuple(issues), PacketIntegritySummary(None, crc_valid, complete), packet.source, bits)
