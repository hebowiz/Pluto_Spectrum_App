"""Bluetooth LE 1M/2M uncoded packet decoder."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pluto_protocol.bitops import bits_hex_lsb, bits_to_int_lsb
from pluto_protocol.bluetooth.common import le_crc24_bits, le_whitening_sequence
from pluto_protocol.model import (
    DecodeProbeResult, FieldStatus, IssueSeverity, PacketAnalysisResult,
    PacketDecodeInput, PacketField, PacketIntegritySummary, PacketIssue,
    PacketSummaryItem,
)

PROTOCOL_ID = "bluetooth.le"


def _field(field_id, name, start, bits, value=None, meaning="", status=FieldStatus.INFO, children=()):
    return PacketField(field_id, name, start, start + int(bits.size), bits, value, meaning, status, tuple(children))


@dataclass(frozen=True)
class BluetoothLEDecoder:
    protocol_id: str = PROTOCOL_ID
    protocol_name: str = "Bluetooth Low Energy"

    def probe(self, packet: PacketDecodeInput) -> DecodeProbeResult:
        confidence = 0.95 if packet.protocol_hint == self.protocol_id else 0.2
        return DecodeProbeResult(self.protocol_id, confidence, "LE preamble, access address and PDU layout")

    def decode(self, packet: PacketDecodeInput) -> PacketAnalysisResult:
        bits, context = packet.bits, packet.context
        phy = packet.phy_hint or str(context.get("phy", "LE 1M"))
        preamble_count = 16 if "2M" in phy.upper().replace(" ", "") else 8
        issues: list[PacketIssue] = []
        fields: list[PacketField] = []
        minimum = preamble_count + 32
        preamble = bits[:min(bits.size, preamble_count)]
        fields.append(_field("preamble", "Preamble", 0, preamble, bits_hex_lsb(preamble)))
        access = bits[preamble_count:min(bits.size, minimum)] if bits.size > preamble_count else np.empty(0, dtype=np.uint8)
        fields.append(_field("access_address", "Access Address", preamble_count, access, bits_hex_lsb(access)))
        if bits.size < minimum:
            issues.append(PacketIssue("truncated_access_address", "Packet ends before the access address is complete", IssueSeverity.WARNING))
            return self._result(packet, phy, fields, issues, None, False)

        encoded = bits[minimum:]
        whitening_enabled = bool(context.get("whitening_enabled", True))
        channel = context.get("whitening_channel_index")
        if whitening_enabled and channel is None:
            logical = encoded
            issues.append(PacketIssue("missing_channel", "LE channel index is required to dewhiten the PDU", IssueSeverity.WARNING))
        elif whitening_enabled:
            logical = encoded ^ le_whitening_sequence(int(channel), encoded.size)
        else:
            logical = encoded
        if logical.size < 16:
            fields.append(_field("pdu", "PDU", minimum, logical, status=FieldStatus.WARNING))
            issues.append(PacketIssue("truncated_pdu_header", "Packet ends inside the PDU header", IssueSeverity.WARNING))
            return self._result(packet, phy, fields, issues, None, False)

        header, length_bits = logical[:8], logical[8:16]
        length_bytes = bits_to_int_lsb(length_bits)
        body_stop = 16 + length_bytes * 8
        crc_stop = body_stop + 24
        complete = logical.size >= crc_stop
        body = logical[16:min(logical.size, body_stop)]
        crc = logical[body_stop:min(logical.size, crc_stop)] if logical.size > body_stop else np.empty(0, dtype=np.uint8)
        crc_enabled = bool(context.get("crc_enabled", True))
        crc_valid: bool | None = None
        if complete and crc_enabled:
            expected = le_crc24_bits(logical[:body_stop], int(context.get("crc_init", 0x555555)))
            crc_valid = bool(np.array_equal(crc, expected))
        if not complete:
            issues.append(PacketIssue("truncated_pdu", f"PDU declares {length_bytes} payload byte(s), but the captured packet is incomplete", IssueSeverity.WARNING))
        elif crc_valid is False:
            issues.append(PacketIssue("crc_mismatch", "LE CRC does not match", IssueSeverity.WARNING, minimum + body_stop, minimum + crc_stop))

        pdu_type = bits_to_int_lsb(header[:4])
        pdu_children = (
            _field("pdu_header", "PDU Header", minimum, header, f"0x{bits_to_int_lsb(header):02X}", children=(
                _field("pdu_type", "PDU Type", minimum, header[:4], pdu_type),
                _field("pdu_length", "Length", minimum + 8, length_bits, length_bytes, f"{length_bytes} byte(s)"),
            )),
            _field("payload", "Payload", minimum + 16, body, bits_hex_lsb(body), f"{body.size // 8} complete byte(s)"),
            _field("crc", "CRC", minimum + body_stop, crc, bits_hex_lsb(crc), status=FieldStatus.UNKNOWN if crc_valid is None else FieldStatus.VALID if crc_valid else FieldStatus.INVALID),
        )
        fields.append(_field("pdu", "PDU", minimum, logical[:min(logical.size, crc_stop)], meaning=f"Type {pdu_type}; {length_bytes} byte payload", children=pdu_children))
        return self._result(packet, phy, fields, issues, crc_valid, complete)

    def _result(self, packet, phy, fields, issues, crc_valid, complete):
        summary = (
            PacketSummaryItem("protocol", "Protocol", self.protocol_name, self.protocol_name),
            PacketSummaryItem("phy", "PHY", phy, phy),
            PacketSummaryItem("crc", "CRC", crc_valid, "Not checked" if crc_valid is None else "Valid" if crc_valid else "Invalid", FieldStatus.UNKNOWN if crc_valid is None else FieldStatus.VALID if crc_valid else FieldStatus.INVALID),
        )
        return PacketAnalysisResult("1.0", self.protocol_id, self.protocol_name, phy, None, summary, tuple(fields), tuple(issues), PacketIntegritySummary(None, crc_valid, complete), packet.source, packet.bits)
