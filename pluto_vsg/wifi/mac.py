"""Small Wi-Fi PSDU and Beacon builder."""

from __future__ import annotations

import binascii

import numpy as np

from pluto_vsg.model import WiFiPSDUSource, WiFiSettings


def _mac(value: str) -> bytes:
    return bytes(int(part, 16) for part in value.split(":"))


def append_fcs(frame: bytes) -> bytes:
    return frame + binascii.crc32(frame).to_bytes(4, "little")


def build_beacon_psdu(settings: WiFiSettings) -> bytes:
    bssid = _mac(settings.bssid)
    sequence_control = (int(settings.sequence_number) << 4).to_bytes(2, "little")
    header = (
        b"\x80\x00" + b"\x00\x00" + b"\xff" * 6 + bssid + bssid + sequence_control
    )
    fixed = (
        b"\x00" * 8
        + int(settings.beacon_interval_tu).to_bytes(2, "little")
        + b"\x01\x04"  # ESS + short-slot-time, open ERP BSS
    )
    ssid = settings.ssid.encode("utf-8")
    rates = bytes((0x8C, 0x12, 0x98, 0x24, 0xB0, 0x48, 0x60, 0x6C))
    ies = (
        bytes((0, len(ssid))) + ssid
        + bytes((1, len(rates))) + rates
        + bytes((3, 1, int(settings.channel)))
        # TIM: DTIM count 0, period 1, no buffered unicast/broadcast traffic.
        + bytes((5, 4, 0, 1, 0, 0))
        # ERP Information: no non-ERP stations and no protection required.
        + bytes((42, 1, 0))
    )
    frame = header + fixed + ies
    return append_fcs(frame) if settings.fcs_auto else frame


def _hex_bytes(text: str) -> bytes:
    compact = "".join(text.replace("0x", "").replace("0X", "").split())
    if len(compact) % 2:
        raise ValueError("Raw PSDU hex must contain complete octets")
    try:
        return bytes.fromhex(compact)
    except ValueError as error:
        raise ValueError("Raw PSDU must contain hexadecimal octets") from error


def build_psdu(settings: WiFiSettings) -> bytes:
    source = WiFiPSDUSource(settings.psdu_source)
    if source == WiFiPSDUSource.BEACON:
        return build_beacon_psdu(settings)
    if source == WiFiPSDUSource.RAW_HEX:
        return _hex_bytes(settings.raw_psdu_hex)
    count = int(settings.payload_length_bytes)
    if source == WiFiPSDUSource.PATTERN:
        pattern = _hex_bytes(settings.payload_pattern_hex)
        if not pattern:
            raise ValueError("Wi-Fi payload pattern must not be empty")
        return bytes(pattern[index % len(pattern)] for index in range(count))
    # Deterministic PRBS-9, packed LSB first into octets.
    register = 0x1FF
    bits: list[int] = []
    for _ in range(count * 8):
        bits.append(register & 1)
        feedback = ((register >> 4) ^ (register >> 8)) & 1
        register = (register >> 1) | (feedback << 8)
    return bytes(sum(bits[offset + bit] << bit for bit in range(8)) for offset in range(0, len(bits), 8))


def bytes_to_air_bits(value: bytes) -> np.ndarray:
    return np.asarray([(octet >> bit) & 1 for octet in value for bit in range(8)], dtype=np.uint8)
