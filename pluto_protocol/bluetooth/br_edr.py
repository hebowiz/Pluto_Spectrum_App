"""Bluetooth BR/EDR packet decoder for canonical over-the-air bits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pluto_protocol.bitops import bits_hex_lsb, bits_to_bytes_lsb, bits_to_int_lsb, bits_to_int_msb
from pluto_protocol.bluetooth.common import (
    br_whitening_sequence,
    decode_acl_header,
    fec13_decode,
    header_error_check,
    payload_crc_bytes,
)
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


PROTOCOL_ID = "bluetooth.br_edr"
ACCESS_BITS = 72
HEADER_AIR_BITS = 54

_TYPE_NAMES = {
    (1, 0x4): "DH1", (1, 0xB): "DH3", (1, 0xF): "DH5",
    (2, 0x4): "2-DH1", (2, 0xA): "2-DH3", (2, 0xE): "2-DH5",
    (3, 0x8): "3-DH1", (3, 0xB): "3-DH3", (3, 0xF): "3-DH5",
}


def _field(field_id: str, name: str, start: int, bits: np.ndarray, value=None, meaning: str = "", status=FieldStatus.INFO, children=()) -> PacketField:
    return PacketField(field_id, name, start, start + int(bits.size), bits, value, meaning, status, tuple(children))


def _issue(code: str, message: str, start: int | None = None) -> PacketIssue:
    return PacketIssue(code, message, IssueSeverity.WARNING, start, None)


@dataclass(frozen=True)
class BluetoothBREDRDecoder:
    protocol_id: str = PROTOCOL_ID
    protocol_name: str = "Bluetooth BR / EDR"

    def probe(self, packet: PacketDecodeInput) -> DecodeProbeResult:
        confidence = 0.95 if packet.protocol_hint == self.protocol_id else 0.25
        if packet.bits.size >= ACCESS_BITS + HEADER_AIR_BITS:
            confidence += 0.03
        return DecodeProbeResult(self.protocol_id, min(confidence, 1.0), "BR/EDR access-code and header layout")

    def decode(self, packet: PacketDecodeInput) -> PacketAnalysisResult:
        bits = packet.bits
        context = packet.context
        issues: list[PacketIssue] = []
        fields: list[PacketField] = []
        complete = True

        access = bits[: min(bits.size, ACCESS_BITS)]
        fields.append(_field("access_code", "Access Code", 0, access, bits_hex_lsb(access), "Preamble, sync word and trailer"))
        if access.size < ACCESS_BITS:
            issues.append(_issue("truncated_access_code", "Packet ends inside the 72-bit access code", access.size))
            return self._result(packet, None, None, fields, issues, None, None, False)

        header_air = bits[ACCESS_BITS:min(bits.size, ACCESS_BITS + HEADER_AIR_BITS)]
        if header_air.size < HEADER_AIR_BITS:
            fields.append(_field("header", "Header", ACCESS_BITS, header_air, status=FieldStatus.WARNING))
            issues.append(_issue("truncated_header", "Packet ends inside the 54-bit FEC header", ACCESS_BITS + header_air.size))
            return self._result(packet, None, None, fields, issues, None, None, False)

        header_fec, corrected = fec13_decode(header_air)
        whitening_enabled = bool(context.get("whitening_enabled", True))
        clock = context.get("clock_6_1")
        if whitening_enabled and clock is None:
            issues.append(_issue("missing_clock", "CLK_6-1 is required to dewhiten BR/EDR headers"))
            header = header_fec
        elif whitening_enabled:
            header = header_fec ^ br_whitening_sequence(int(clock), header_fec.size)
        else:
            header = header_fec

        data = header[:10]
        packed = bits_to_int_lsb(data)
        lt_addr, packet_type = packed & 0x7, (packed >> 3) & 0xF
        flow, arqn, seqn = (packed >> 7) & 1, (packed >> 8) & 1, (packed >> 9) & 1
        hec = bits_to_int_msb(header[10:18])
        uap = context.get("uap")
        hec_valid = None if uap is None else header_error_check(data, int(uap)) == hec
        header_children = (
            _field("lt_addr", "LT_ADDR", ACCESS_BITS, data[:3], lt_addr),
            _field("type", "TYPE", ACCESS_BITS + 3, data[3:7], packet_type),
            _field("flow", "FLOW", ACCESS_BITS + 7, data[7:8], flow),
            _field("arqn", "ARQN", ACCESS_BITS + 8, data[8:9], arqn),
            _field("seqn", "SEQN", ACCESS_BITS + 9, data[9:10], seqn),
            _field("hec", "HEC", ACCESS_BITS + 10, header[10:18], f"0x{hec:02X}", status=(FieldStatus.UNKNOWN if hec_valid is None else FieldStatus.VALID if hec_valid else FieldStatus.INVALID)),
        )
        fields.append(_field("header", "Header", ACCESS_BITS, header_air, meaning=f"1/3 FEC; {corrected} corrected triplet(s)", children=header_children))
        if hec_valid is False:
            issues.append(_issue("hec_mismatch", "Header Error Check does not match", ACCESS_BITS + 30))

        phy = packet.phy_hint or str(context.get("phy", "BR"))
        packet_kind = context.get("packet_kind")
        bits_per_symbol = 3 if "3" in phy.upper() or str(packet_kind).startswith("3-") else 2 if "2" in phy.upper() or str(packet_kind).startswith("2-") else 1
        packet_name = str(packet_kind) if packet_kind else _TYPE_NAMES.get((bits_per_symbol, packet_type), f"TYPE 0x{packet_type:X}")

        payload_start = ACCESS_BITS + HEADER_AIR_BITS
        if bits_per_symbol > 1:
            sync_count = 10 * bits_per_symbol
            sync = bits[payload_start:min(bits.size, payload_start + sync_count)]
            fields.append(_field("edr_sync", "EDR Synchronization", payload_start, sync))
            payload_start += sync_count

        payload_air = bits[payload_start:]
        trailer_count = 2 * bits_per_symbol if bits_per_symbol > 1 else 0
        padding_count = int(context.get("edr_padding_bits", 0)) if bits_per_symbol > 1 else 0
        payload_stop = max(payload_start, bits.size - trailer_count - padding_count)
        payload_air = bits[payload_start:payload_stop]
        if whitening_enabled and clock is not None:
            whitening = br_whitening_sequence(int(clock), 18 + payload_air.size)
            payload = payload_air ^ whitening[18:]
        else:
            payload = payload_air

        # BR DH1 alone uses the one-byte ACL payload header.  EDR 2-DH1 and
        # 3-DH1 use the enhanced two-byte length field despite sharing the
        # one-slot suffix.
        header_width = 8 if packet_name == "DH1" else 16
        if payload.size < header_width:
            fields.append(_field("payload", "Payload", payload_start, payload, status=FieldStatus.WARNING))
            issues.append(_issue("truncated_payload_header", "Packet ends inside the ACL payload header", payload_start + payload.size))
            return self._result(packet, phy, packet_name, fields, issues, hec_valid, None, False)

        acl_header = payload[:header_width]
        llid, payload_flow, length_bytes = decode_acl_header(acl_header)
        body_stop = header_width + length_bytes * 8
        crc_stop = body_stop + 16
        complete = payload.size >= crc_stop
        body = payload[header_width:min(payload.size, body_stop)]
        crc_bits = payload[body_stop:min(payload.size, crc_stop)] if payload.size > body_stop else np.empty(0, dtype=np.uint8)
        crc_valid: bool | None = None
        received_crc = b""
        expected_crc = b""
        if complete and uap is not None:
            received_crc = bits_to_bytes_lsb(crc_bits)
            expected_crc = payload_crc_bytes(payload[:body_stop], int(uap))
            crc_valid = received_crc == expected_crc
        elif not complete:
            issues.append(_issue("truncated_payload", f"Payload declares {length_bytes} byte(s), but the captured packet is incomplete", payload_start + payload.size))
        elif uap is None:
            issues.append(_issue("crc_not_checked", "UAP is unavailable; payload CRC was not checked"))

        payload_children = (
            _field("payload_header", "Payload Header", payload_start, acl_header, children=(
                _field("llid", "LLID", payload_start, acl_header[:2], llid),
                _field("payload_flow", "FLOW", payload_start + 2, acl_header[2:3], payload_flow),
                _field("length", "Length", payload_start + 3, acl_header[3:], length_bytes, f"{length_bytes} byte(s)"),
            )),
            _field("payload_body", "Payload Body", payload_start + header_width, body, bits_hex_lsb(body), f"{body.size // 8} complete byte(s)"),
            _field("payload_crc", "Payload CRC", payload_start + body_stop, crc_bits, received_crc.hex().upper() if received_crc else None, (f"Expected {expected_crc.hex().upper()}" if expected_crc else "Not checked"), status=(FieldStatus.UNKNOWN if crc_valid is None else FieldStatus.VALID if crc_valid else FieldStatus.INVALID)),
        )
        fields.append(_field("payload", "ACL Payload", payload_start, payload[:min(payload.size, crc_stop)], meaning=f"{length_bytes} byte payload", children=payload_children))
        if crc_valid is False:
            issues.append(_issue("crc_mismatch", "Payload CRC does not match", payload_start + body_stop))
        if padding_count:
            fields.append(_field("edr_padding", "EDR Padding", payload_stop, bits[payload_stop:payload_stop + padding_count]))
        if trailer_count:
            fields.append(_field("edr_trailer", "EDR Trailer", bits.size - trailer_count, bits[-trailer_count:]))
        return self._result(packet, phy, packet_name, fields, issues, hec_valid, crc_valid, complete)

    def _result(self, packet, phy, packet_name, fields, issues, hec_valid, crc_valid, complete):
        integrity = PacketIntegritySummary(hec_valid, crc_valid, complete)
        summary = (
            PacketSummaryItem("protocol", "Protocol", self.protocol_name, self.protocol_name),
            PacketSummaryItem("phy", "PHY", phy, phy or "Unknown"),
            PacketSummaryItem("packet_type", "Packet Type", packet_name, packet_name or "Unknown"),
            PacketSummaryItem("hec", "HEC", hec_valid, "Not checked" if hec_valid is None else "Valid" if hec_valid else "Invalid", FieldStatus.UNKNOWN if hec_valid is None else FieldStatus.VALID if hec_valid else FieldStatus.INVALID),
            PacketSummaryItem("crc", "CRC", crc_valid, "Not checked" if crc_valid is None else "Valid" if crc_valid else "Invalid", FieldStatus.UNKNOWN if crc_valid is None else FieldStatus.VALID if crc_valid else FieldStatus.INVALID),
        )
        return PacketAnalysisResult("1.0", self.protocol_id, self.protocol_name, phy, packet_name, summary, tuple(fields), tuple(issues), integrity, packet.source, packet.bits)
