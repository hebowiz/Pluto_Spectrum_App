"""Bluetooth HDT PHY definitions shared by VSG generation and VSA analysis."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from pluto_protocol.bitops import bits_hex_octets_lsb
from pluto_protocol.model import (
    BitRepresentation,
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


PROTOCOL_ID = "bluetooth.hdt"


class HDTRate(StrEnum):
    HDT2 = "HDT2"
    HDT3 = "HDT3"
    HDT4 = "HDT4"
    HDT6 = "HDT6"
    HDT7_5 = "HDT7.5"


@dataclass(frozen=True)
class HDTDefinition:
    rate: HDTRate
    rate_indicator: int
    modulation: str
    bits_per_symbol: int
    payload_code_rate: str


HDT_DEFINITIONS = {
    HDTRate.HDT2: HDTDefinition(HDTRate.HDT2, 0b001, "pi/4-QPSK", 2, "1/2"),
    HDTRate.HDT3: HDTDefinition(HDTRate.HDT3, 0b010, "pi/4-QPSK", 2, "3/4"),
    HDTRate.HDT4: HDTDefinition(HDTRate.HDT4, 0b011, "8PSK", 3, "2/3"),
    HDTRate.HDT6: HDTDefinition(HDTRate.HDT6, 0b100, "16QAM", 4, "3/4"),
    HDTRate.HDT7_5: HDTDefinition(HDTRate.HDT7_5, 0b101, "16QAM", 4, "15/16"),
}

HDT_RF_TEST_PCA = 0x9F_1555_5555
HDT_RF_TEST_CRC32_INIT = 0xAA55_5555
HDT_RF_TEST_LTS_ROOT = 7
HDT_RF_TEST_LTS_PHASE = 8


def hdt_definition(rate: HDTRate | str) -> HDTDefinition:
    return HDT_DEFINITIONS[HDTRate(rate)]


_PUNCTURE_MASKS = {
    "1/2": (1, 1),
    "2/3": (1, 1, 0, 1),
    "3/4": (1, 1, 0, 1, 0, 1),
    "15/16": (
        1, 1, 0, 1, 1, 0, 1, 0, 1, 0,
        0, 1, 0, 1, 0, 1, 1, 0, 1, 0,
        0, 1, 0, 1, 0, 1, 1, 0, 0, 1,
    ),
}


def convolutional_encode(bits: np.ndarray, *, terminate: bool = True) -> np.ndarray:
    """Encode with HDT K=6, G0=1+x^2+x^4+x^5 and G1=1+x+x^2+x^3+x^5."""

    values = np.asarray(bits, dtype=np.uint8)
    if values.ndim != 1 or np.any(values > 1):
        raise ValueError("bits must be a one-dimensional binary array")
    if terminate:
        values = np.concatenate((values, np.zeros(5, dtype=np.uint8)))
    history = np.zeros(5, dtype=np.uint8)
    encoded = np.empty(values.size * 2, dtype=np.uint8)
    for index, bit in enumerate(values):
        taps = np.concatenate(([bit], history))
        encoded[2 * index] = taps[0] ^ taps[2] ^ taps[4] ^ taps[5]
        encoded[2 * index + 1] = taps[0] ^ taps[1] ^ taps[2] ^ taps[3] ^ taps[5]
        history[1:] = history[:-1]
        history[0] = bit
    return encoded


def puncture(bits: np.ndarray, code_rate: str) -> np.ndarray:
    values = np.asarray(bits, dtype=np.uint8)
    try:
        mask = np.asarray(_PUNCTURE_MASKS[str(code_rate)], dtype=bool)
    except KeyError as error:
        raise ValueError(f"Unsupported HDT code rate: {code_rate}") from error
    return values[np.resize(mask, values.size)]


def _lsb_bits(value: int, width: int) -> np.ndarray:
    return np.asarray([(int(value) >> index) & 1 for index in range(width)], dtype=np.uint8)


def _msb_bits(value: int, width: int) -> np.ndarray:
    return np.asarray(
        [(int(value) >> (width - 1 - index)) & 1 for index in range(width)],
        dtype=np.uint8,
    )


def hdt_crc24(bits: np.ndarray, *, init: int) -> int:
    """Return the Vol 6 Part B 24-bit CRC register after transmitted-order bits."""

    state = int(init) & 0xFF_FFFF
    for bit in np.asarray(bits, dtype=np.uint8):
        feedback = ((state >> 23) & 1) ^ int(bit)
        state = (state << 1) & 0xFF_FFFF
        if feedback:
            state ^= 0x00065B
    return state


def hdt_crc32(bits: np.ndarray, *, init: int = HDT_RF_TEST_CRC32_INIT) -> int:
    """Return the Vol 6 Part B HDT CRC-32 register after transmitted-order bits."""

    state = int(init) & 0xFFFF_FFFF
    for bit in np.asarray(bits, dtype=np.uint8):
        feedback = ((state >> 31) & 1) ^ int(bit)
        state = (state << 1) & 0xFFFF_FFFF
        if feedback:
            state ^= 0x04C11DB7
    return state


def hdt_rf_test_training_symbols() -> np.ndarray:
    """Return the standard 74-symbol RF PHY test preamble (STS x9, GI, LTS x2)."""

    short = np.tile(np.asarray([-1.0, -1.0j, 1.0j, 1.0]), 9)
    index = np.arange(17, dtype=np.float64)
    long = np.exp(
        -1j * np.pi * HDT_RF_TEST_LTS_ROOT * index * (index + 1.0) / 17.0
    ) * np.exp(1j * 2.0 * np.pi * HDT_RF_TEST_LTS_PHASE / 17.0)
    return np.asarray(np.concatenate((short, long[-4:], long, long)), dtype=np.complex64)


def hdt_rf_test_control_bits(
    rate: HDTRate | str,
    payload_length_bytes: int,
) -> np.ndarray:
    """Build the 57 logical Control Header bits for an RF PHY format-0 packet."""

    payload_length = int(payload_length_bytes)
    if not 1 <= payload_length <= 510:
        raise ValueError("HDT RF test format-0 payload length must be between 1 and 510 bytes")
    definition = hdt_definition(rate)
    pdu_length = payload_length + 1  # one-octet ACL Initial Portion
    header = np.concatenate(
        (
            _lsb_bits((HDT_RF_TEST_PCA >> 24) & 0xFFFF, 16),
            _lsb_bits(1, 3),
            _lsb_bits(0, 1),
            _lsb_bits(definition.rate_indicator, 3),
            _lsb_bits(0, 1),
            _lsb_bits(pdu_length, 9),
        )
    )
    hec = hdt_crc24(header, init=HDT_RF_TEST_PCA & 0xFF_FFFF)
    return np.concatenate((header, _msb_bits(hec, 24)))


def hdt_rf_test_format0_bits(payload_bits: np.ndarray) -> np.ndarray:
    """Build PDU Header + payload + CRC-32 bits for an RF PHY format-0 packet."""

    payload = np.asarray(payload_bits, dtype=np.uint8)
    if payload.ndim != 1 or np.any(payload > 1) or payload.size % 8:
        raise ValueError("HDT payload must be a whole number of binary octets")
    if payload.size > 510 * 8:
        raise ValueError("HDT RF test format-0 payload cannot exceed 510 bytes")
    pdu = np.concatenate((np.zeros(8, dtype=np.uint8), payload))
    crc = hdt_crc32(pdu)
    return np.concatenate((pdu, _msb_bits(crc, 32)))


def hdt_coded_payload_bit_count(
    rate: HDTRate | str, payload_length_bytes: int
) -> int:
    """Return format-0 PDU/payload/CRC transmitted bits after coding/puncturing."""

    payload = np.zeros(max(0, int(payload_length_bytes)) * 8, dtype=np.uint8)
    logical = hdt_rf_test_format0_bits(payload)
    return int(
        puncture(
            convolutional_encode(logical),
            hdt_definition(rate).payload_code_rate,
        ).size
    )


def map_hdt_symbols(bits: np.ndarray, rate: HDTRate | str, *, symbol_offset: int = 0) -> np.ndarray:
    """Map coded MSB-first bits to the HDT air-interface constellation."""

    definition = hdt_definition(rate)
    values = np.asarray(bits, dtype=np.uint8)
    width = definition.bits_per_symbol
    if values.ndim != 1 or np.any(values > 1):
        raise ValueError("bits must be a one-dimensional binary array")
    if values.size % width:
        values = np.pad(values, (0, width - values.size % width))
    grouped = values.reshape(-1, width)
    labels = grouped.dot(1 << np.arange(width - 1, -1, -1))
    if definition.modulation == "pi/4-QPSK":
        even_phases = np.asarray(
            [np.pi / 4.0, 3.0 * np.pi / 4.0, -np.pi / 4.0, -3.0 * np.pi / 4.0]
        )
        odd_phases = np.asarray([np.pi / 2.0, np.pi, 0.0, -np.pi / 2.0])
        phases = np.where(
            (np.arange(labels.size) + symbol_offset) % 2 == 0,
            even_phases[labels],
            odd_phases[labels],
        )
        return np.exp(1j * phases).astype(np.complex64)
    if definition.modulation == "8PSK":
        phases = np.asarray(
            [0.0, np.pi / 4.0, 3.0 * np.pi / 4.0, np.pi / 2.0,
             -np.pi / 4.0, -np.pi / 2.0, -np.pi, -3.0 * np.pi / 4.0]
        )
        return np.exp(1j * phases[labels]).astype(np.complex64)
    levels = np.asarray([-3.0, -1.0, 3.0, 1.0])
    points = levels[labels >> 2] + 1j * levels[labels & 0x3]
    # HDT_VSr03_PR tabulates S_k x sqrt(10), so recover S_k with the
    # conventional unit-mean-power 16QAM normalization.
    return (points / np.sqrt(10.0)).astype(np.complex64)


def _field(
    field_id: str,
    name: str,
    start: int,
    bits: np.ndarray,
    value: object = None,
    meaning: str = "",
    status: FieldStatus = FieldStatus.INFO,
    children: tuple[PacketField, ...] = (),
) -> PacketField:
    return PacketField(
        field_id,
        name,
        int(start),
        int(start) + int(bits.size),
        bits,
        value,
        meaning,
        status,
        children,
    )


@dataclass(frozen=True)
class BluetoothHDTDecoder:
    """Decode the shared FEC-decoded HDT Control Header / PDU bitstream."""

    protocol_id: str = PROTOCOL_ID
    protocol_name: str = "Bluetooth HDT"

    def probe(self, packet: PacketDecodeInput) -> DecodeProbeResult:
        confidence = 0.98 if packet.protocol_hint == self.protocol_id else 0.15
        return DecodeProbeResult(
            self.protocol_id,
            confidence,
            "HDT protocol hint; logical Control Header and PDU layout",
        )

    def decode(self, packet: PacketDecodeInput) -> PacketAnalysisResult:
        """Interpret FEC-decoded Control Header + Format-0 PDU bits.

        Training, convolutional tails and symbol padding are PHY quantities,
        not logical packet fields. Both VSA and VSG supply this same layout.
        """
        bits = packet.bits
        def unsupported(code: str, message: str) -> PacketAnalysisResult:
            return PacketAnalysisResult(
                "1.0", self.protocol_id, self.protocol_name, packet.phy_hint,
                None, (), (), (PacketIssue(code, message, IssueSeverity.ERROR),),
                PacketIntegritySummary(None, None, False), packet.source, bits,
            )
        if packet.representation != BitRepresentation.LOGICAL:
            return unsupported("unsupported_bit_representation",
                               "HDT requires FEC-decoded logical bits")
        if bits.size < 57:
            return unsupported("truncated_control_header",
                               "HDT Control Header requires 57 logical bits")
        control_data = bits[:57]
        lsb = lambda values: sum(int(bit) << i for i, bit in enumerate(values))
        msb = lambda values: sum(int(bit) << (len(values)-1-i) for i, bit in enumerate(values))
        pca = int(packet.context.get("pca", HDT_RF_TEST_PCA))
        pca_a = lsb(control_data[:16])
        nesn = lsb(control_data[16:19])
        packet_format_indicator = lsb(control_data[19:20])
        rate_indicator = lsb(control_data[20:23])
        control_rfu = lsb(control_data[23:24])
        pdu_octets = lsb(control_data[24:33])
        received_hec = msb(control_data[33:57])
        calculated_hec = hdt_crc24(control_data[:33], init=pca & 0xFFFFFF)
        hec_valid = received_hec == calculated_hec
        rate = next((r for r, d in HDT_DEFINITIONS.items()
                     if d.rate_indicator == rate_indicator), None)
        if rate is None:
            return unsupported("unsupported_rate_indicator",
                               f"Unsupported HDT rate indicator {rate_indicator}")
        if packet_format_indicator != 0:
            return unsupported("unsupported_packet_format",
                               "HDT Format 1 decoding is not implemented")
        if pdu_octets < 1:
            return unsupported("invalid_pdu_length", "Format 0 requires an Initial Portion")
        expected = 57 + pdu_octets * 8 + 32
        if bits.size < expected:
            return unsupported("truncated_payload",
                               f"Expected {expected} logical bits, received {bits.size}")
        definition = hdt_definition(rate)
        payload_length = pdu_octets - 1
        format0_bits = bits[57:expected]
        pdu_bits = format0_bits[:-32]
        payload_bits = pdu_bits[8:]
        if pdu_bits[0] or pdu_bits[1]:
            return unsupported("unsupported_pdu_header",
                               "Extended header / Rx power header decoding is not implemented")
        received_crc = msb(format0_bits[-32:])
        calculated_crc = hdt_crc32(pdu_bits, init=int(packet.context.get("crc_init", HDT_RF_TEST_CRC32_INIT)))
        crc_valid = received_crc == calculated_crc
        legacy_crc_match = received_crc == hdt_crc32(pdu_bits, init=0x00555555)
        control_children = (
            PacketField("pca_a", "PCA-A", 0, 16, control_data[:16], f"0x{pca_a:04X}"),
            PacketField("nesn", "NESN", 16, 19, control_data[16:19], nesn),
            PacketField("pfi", "PFI", 19, 20, control_data[19:20], packet_format_indicator, "Packet format 0"),
            PacketField("rate_indicator", "Rate Indicator", 20, 23, control_data[20:23], f"{rate.value} (0b{rate_indicator:03b})", f"{rate.value}: {definition.modulation}, code rate {definition.payload_code_rate}", FieldStatus.VALID),
            PacketField("rfu", "RFU", 23, 24, control_data[23:24], control_rfu),
            PacketField("pdu_control", "PDU Control", 24, 33, control_data[24:33], pdu_octets, f"{pdu_octets} octet(s), excluding CRC"),
            PacketField("hec_c", "HEC-C", 33, 57, control_data[33:57], f"0x{received_hec:06X}", f"Calculated 0x{calculated_hec:06X}", FieldStatus.VALID if hec_valid else FieldStatus.INVALID),
        )
        payload_offset = 57
        pdu_stop = payload_offset + pdu_bits.size
        payload_children = (
            PacketField(
                "pdu_header", "PDU Header", payload_offset, payload_offset + 8,
                pdu_bits[:8], f"0x{lsb(pdu_bits[:8]):02X}",
                "ACL Format 0 Initial Portion", children=(
                    _field("xhp", "XHP", 57, pdu_bits[:1], lsb(pdu_bits[:1])),
                    _field("rx_pp", "RxPP", 58, pdu_bits[1:2], lsb(pdu_bits[1:2])),
                    _field("md", "MD", 59, pdu_bits[2:3], lsb(pdu_bits[2:3])),
                    _field("sn", "SN", 60, pdu_bits[3:6], lsb(pdu_bits[3:6])),
                    _field("llid", "LLID", 63, pdu_bits[6:8], lsb(pdu_bits[6:8])),
                ),
            ),
            PacketField(
                "payload_body",
                "Payload",
                payload_offset + 8,
                pdu_stop,
                payload_bits,
                bits_hex_octets_lsb(payload_bits),
                f"{payload_bits.size // 8} byte(s)",
            ),
            PacketField("crc32", "CRC-32", pdu_stop, pdu_stop + 32, format0_bits[-32:], f"0x{received_crc:08X}", f"Calculated 0x{calculated_crc:08X}", FieldStatus.VALID if crc_valid else FieldStatus.INVALID),
        )
        issues: tuple[PacketIssue, ...] = tuple(
            issue
            for issue in (
                None if hec_valid else PacketIssue("invalid_hec_c", "HEC-C does not match the Control Header", IssueSeverity.ERROR, 33, 57),
                None if crc_valid else PacketIssue("invalid_crc32", "CRC-32 does not match the PDU Header and Payload" + ("; received value matches legacy 0x00555555 initialization" if legacy_crc_match else ""), IssueSeverity.ERROR, pdu_stop, pdu_stop + 32),
                None if bits.size == expected else PacketIssue("trailing_logical_bits", "Logical stream contains bits after the declared Format 0 PDU / CRC", IssueSeverity.WARNING, expected, bits.size),
            )
            if issue is not None
        )
        return PacketAnalysisResult(
            "1.0",
            "bluetooth.hdt",
            "Bluetooth HDT",
            rate.value,
            rate.value,
            (
                PacketSummaryItem("protocol", "Protocol", "Bluetooth HDT", "Bluetooth HDT"),
                PacketSummaryItem("phy", "Detected PHY", rate.value, rate.value),
                PacketSummaryItem("payload_length", "Payload Length", payload_length, f"{payload_length} byte(s)"),
                PacketSummaryItem("hec_c", "HEC-C", hec_valid, "Pass" if hec_valid else "Fail", FieldStatus.VALID if hec_valid else FieldStatus.INVALID),
                PacketSummaryItem("crc32", "CRC-32", crc_valid, "Pass" if crc_valid else "Fail", FieldStatus.VALID if crc_valid else FieldStatus.INVALID),
            ),
            (
                PacketField("training", "Training / Preamble", 0, 0, np.empty(0, dtype=np.uint8), "74 symbols", "STS x9 + GI + LTS x2", FieldStatus.VALID),
                PacketField("control_header", "Control Header", 0, 57, control_data, f"RI=0b{rate_indicator:03b}, PDU={pdu_octets} octets", "RF PHY test Control Header", FieldStatus.VALID if hec_valid else FieldStatus.INVALID, control_children),
                PacketField("payload", "PDU Header / Payload / CRC", payload_offset, payload_offset + format0_bits.size, format0_bits, f"{payload_length} payload byte(s)", "Packet format 0 decoded bitstream", FieldStatus.VALID if crc_valid else FieldStatus.INVALID, payload_children),
            ),
            issues,
            PacketIntegritySummary(hec_valid, crc_valid, True),
            packet.source,
            packet.bits,
        )


__all__ = [
    "BluetoothHDTDecoder", "HDTDefinition", "HDT_DEFINITIONS", "HDTRate",
    "HDT_RF_TEST_CRC32_INIT", "HDT_RF_TEST_LTS_PHASE", "HDT_RF_TEST_LTS_ROOT",
    "HDT_RF_TEST_PCA", "convolutional_encode", "hdt_coded_payload_bit_count",
    "hdt_crc24", "hdt_crc32", "hdt_definition", "hdt_rf_test_control_bits",
    "hdt_rf_test_format0_bits", "hdt_rf_test_training_symbols", "map_hdt_symbols",
    "puncture",
]
