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
    sequence_control = ((int(settings.sequence_number) << 4) | int(settings.fragment_number)).to_bytes(2, "little")
    header = (
        int(settings.frame_control).to_bytes(2, "little")
        + int(settings.duration_id).to_bytes(2, "little")
        + _mac(settings.destination_address)
        + _mac(settings.source_address or settings.bssid) + bssid + sequence_control
    )
    fixed = (
        int(settings.timestamp).to_bytes(8, "little")
        + int(settings.beacon_interval_tu).to_bytes(2, "little")
        + int(settings.capability_information).to_bytes(2, "little")
    )
    ssid = settings.ssid.encode("utf-8")
    rates = _hex_bytes(settings.supported_rates_hex)
    tim = _hex_bytes(settings.tim_hex)
    channel = settings.channel if settings.ds_channel_auto else settings.ds_channel
    ies = (
        bytes((0, len(ssid))) + ssid
        + bytes((1, len(rates))) + rates
        + bytes((3, 1, int(channel)))
        # TIM: DTIM count 0, period 1, no buffered unicast/broadcast traffic.
        + bytes((5, len(tim))) + tim
        # ERP Information: no non-ERP stations and no protection required.
        + bytes((42, 1, int(settings.erp_information)))
    )
    frame = header + fixed + ies
    return _with_fcs(frame, settings)


def _with_fcs(frame: bytes, settings: WiFiSettings) -> bytes:
    if settings.fcs_auto:
        return append_fcs(frame)
    fcs = _hex_bytes(settings.manual_fcs_hex)
    if len(fcs) != 4:
        raise ValueError("Manual FCS must contain four octets in transmitted order")
    return frame + fcs


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
        raw = _hex_bytes(settings.raw_psdu_hex)
        return raw if settings.raw_includes_fcs else _with_fcs(raw, settings)
    count = int(settings.payload_length_bytes)
    if source == WiFiPSDUSource.PATTERN:
        pattern = _hex_bytes(settings.payload_pattern_hex)
        if not pattern:
            raise ValueError("Wi-Fi payload pattern must not be empty")
        return bytes(pattern[index % len(pattern)] for index in range(count))
    # Deterministic PRBS-9 (x^9 + x^5 + 1), MSB output, packed LSB first
    # into octets. Shift direction must agree with taps; the old right shift
    # with these taps collapsed to a 21-bit cycle instead of 511 bits.
    register = 0x1FF
    bits: list[int] = []
    for _ in range(count * 8):
        bits.append((register >> 8) & 1)
        feedback = ((register >> 4) ^ (register >> 8)) & 1
        register = ((register << 1) | feedback) & 0x1FF
    return bytes(sum(bits[offset + bit] << bit for bit in range(8)) for offset in range(0, len(bits), 8))


def bytes_to_air_bits(value: bytes) -> np.ndarray:
    return np.asarray([(octet >> bit) & 1 for octet in value for bit in range(8)], dtype=np.uint8)
