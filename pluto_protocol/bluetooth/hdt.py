"""Bluetooth HDT PHY definitions shared by VSG generation and VSA analysis."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from pluto_protocol.bitops import bits_hex_lsb
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


def hdt_coded_payload_bit_count(
    rate: HDTRate | str, payload_length_bytes: int
) -> int:
    """Return the transmitted payload bit count after coding/puncturing."""

    logical = np.zeros(max(0, int(payload_length_bytes)) * 8, dtype=np.uint8)
    return int(
        puncture(
            convolutional_encode(logical),
            hdt_definition(rate).payload_code_rate,
        ).size
    )


def map_hdt_symbols(bits: np.ndarray, rate: HDTRate | str) -> np.ndarray:
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
            np.arange(labels.size) % 2 == 0,
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
    """Decode the HDT training/control/coded-payload stream used by the PHY."""

    protocol_id: str = PROTOCOL_ID
    protocol_name: str = "Bluetooth HDT"

    def probe(self, packet: PacketDecodeInput) -> DecodeProbeResult:
        confidence = 0.98 if packet.protocol_hint == self.protocol_id else 0.15
        return DecodeProbeResult(
            self.protocol_id,
            confidence,
            "HDT QPSK training, control header and coded payload layout",
        )

    def decode(self, packet: PacketDecodeInput) -> PacketAnalysisResult:
        bits = packet.bits
        context = packet.context
        training_count = int(context.get("training_bit_count", 148))
        control_count = int(context.get("control_bit_count", 20))
        training = bits[: min(bits.size, training_count)]
        control_start = training_count
        control = bits[
            control_start : min(bits.size, control_start + control_count)
        ]
        issues: list[PacketIssue] = []
        fields: list[PacketField] = [
            _field(
                "training",
                "Training / Preamble",
                0,
                training,
                f"{training.size // 2} QPSK symbol(s)",
                "Packet synchronization and carrier/timing estimation",
                FieldStatus.VALID if training.size == training_count else FieldStatus.WARNING,
            )
        ]
        if control.size < 15:
            issues.append(
                PacketIssue(
                    "truncated_control_header",
                    "Packet ends before the HDT rate and payload length are complete",
                    IssueSeverity.WARNING,
                    control_start,
                    control_start + control.size,
                )
            )
            fields.append(
                _field(
                    "control_header",
                    "Control Header",
                    control_start,
                    control,
                    status=FieldStatus.WARNING,
                )
            )
            return self._result(packet, None, None, fields, issues, False)

        rate_indicator = int(
            sum(int(control[index]) << (2 - index) for index in range(3))
        )
        rate = next(
            (
                candidate
                for candidate, definition in HDT_DEFINITIONS.items()
                if definition.rate_indicator == rate_indicator
            ),
            None,
        )
        length_bits = control[3:15]
        payload_length = int(
            sum(int(length_bits[index]) << index for index in range(12))
        )
        tail = control[15:20]
        ri_status = FieldStatus.VALID if rate is not None else FieldStatus.INVALID
        if rate is None:
            issues.append(
                PacketIssue(
                    "unsupported_rate_indicator",
                    f"Unsupported HDT rate indicator 0b{rate_indicator:03b}",
                    IssueSeverity.ERROR,
                    control_start,
                    control_start + 3,
                )
            )
        rate_meaning = (
            "Unknown/reserved HDT rate"
            if rate is None
            else (
                f"{rate.value}: {hdt_definition(rate).modulation}, "
                f"code rate {hdt_definition(rate).payload_code_rate}"
            )
        )
        control_children = (
            _field(
                "rate_indicator",
                "Rate Indicator",
                control_start,
                control[:3],
                f"0b{rate_indicator:03b}",
                rate_meaning,
                ri_status,
            ),
            _field(
                "payload_length",
                "Payload Length",
                control_start + 3,
                length_bits,
                payload_length,
                f"{payload_length} byte(s) before channel coding",
                FieldStatus.VALID,
            ),
            _field(
                "encoder_tail",
                "Encoder Tail",
                control_start + 15,
                tail,
                "".join(str(int(value)) for value in tail),
                "Terminates the K=6 convolutional encoder",
                FieldStatus.VALID if tail.size == 5 and not np.any(tail) else FieldStatus.WARNING,
            ),
        )
        fields.append(
            _field(
                "control_header",
                "Control Header",
                control_start,
                control,
                f"RI=0b{rate_indicator:03b}, Length={payload_length}",
                "Convolutionally decoded PHY control information",
                FieldStatus.VALID if rate is not None else FieldStatus.INVALID,
                control_children,
            )
        )

        payload_start = control_start + control_count
        payload = bits[payload_start:]
        expected_payload_bits = (
            hdt_coded_payload_bit_count(rate, payload_length)
            if rate is not None
            else int(context.get("expected_payload_bit_count", payload.size))
        )
        complete = payload.size >= expected_payload_bits
        payload = payload[:expected_payload_bits]
        if not complete:
            issues.append(
                PacketIssue(
                    "truncated_payload",
                    f"Expected {expected_payload_bits} coded payload bits, received {payload.size}",
                    IssueSeverity.WARNING,
                    payload_start,
                    payload_start + payload.size,
                )
            )
        definition = hdt_definition(rate) if rate is not None else None
        fields.append(
            _field(
                "payload",
                "Coded Payload",
                payload_start,
                payload,
                bits_hex_lsb(payload),
                (
                    f"{payload_length} logical byte(s), {payload.size} transmitted bit(s)"
                    + (
                        ""
                        if definition is None
                        else f"; {definition.modulation}, code rate {definition.payload_code_rate}"
                    )
                ),
                FieldStatus.VALID if complete else FieldStatus.WARNING,
            )
        )
        return self._result(packet, rate, payload_length, fields, issues, complete)

    def _result(
        self,
        packet: PacketDecodeInput,
        rate: HDTRate | None,
        payload_length: int | None,
        fields: list[PacketField],
        issues: list[PacketIssue],
        complete: bool,
    ) -> PacketAnalysisResult:
        phy = rate.value if rate is not None else packet.phy_hint
        definition = hdt_definition(rate) if rate is not None else None
        summary = (
            PacketSummaryItem("protocol", "Protocol", self.protocol_name, self.protocol_name),
            PacketSummaryItem("phy", "Detected PHY", phy, phy or "Unknown"),
            PacketSummaryItem(
                "payload_modulation",
                "Payload Modulation",
                None if definition is None else definition.modulation,
                "Unknown" if definition is None else definition.modulation,
            ),
            PacketSummaryItem(
                "payload_length",
                "Payload Length",
                payload_length,
                "Unknown" if payload_length is None else f"{payload_length} byte(s)",
            ),
        )
        return PacketAnalysisResult(
            "1.0",
            self.protocol_id,
            self.protocol_name,
            phy,
            phy,
            summary,
            tuple(fields),
            tuple(issues),
            PacketIntegritySummary(None, None, complete),
            packet.source,
            packet.bits,
        )


__all__ = [
    "BluetoothHDTDecoder", "HDTDefinition", "HDT_DEFINITIONS", "HDTRate",
    "convolutional_encode", "hdt_coded_payload_bit_count", "hdt_definition",
    "map_hdt_symbols", "puncture",
]
