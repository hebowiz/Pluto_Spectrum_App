"""Wi-Fi PSDU and non-HT management frame builders."""

from __future__ import annotations

import binascii

import numpy as np

from pluto_vsg.model import WiFiPSDUSource, WiFiSettings


def _mac(value: str) -> bytes:
    return bytes(int(part, 16) for part in value.split(":"))


def append_fcs(frame: bytes) -> bytes:
    return frame + binascii.crc32(frame).to_bytes(4, "little")


MANAGEMENT_FRAME_CONTROLS = {
    WiFiPSDUSource.BEACON: 0x0080,
    WiFiPSDUSource.PROBE_REQUEST: 0x0040,
    WiFiPSDUSource.PROBE_RESPONSE: 0x0050,
}


def effective_frame_control(settings: WiFiSettings, source=None) -> int:
    return (MANAGEMENT_FRAME_CONTROLS[WiFiPSDUSource(source or settings.psdu_source)]
            if settings.frame_control_auto else int(settings.frame_control))


def build_management_header(settings: WiFiSettings, source: WiFiPSDUSource) -> bytes:
    sequence = (int(settings.sequence_number) << 4) | int(settings.fragment_number)
    return (
        effective_frame_control(settings, source).to_bytes(2, "little")
        + int(settings.duration_id).to_bytes(2, "little")
        + _mac(settings.destination_address)
        + _mac(settings.source_address or settings.bssid) + _mac(settings.bssid)
        + sequence.to_bytes(2, "little")
    )


def build_information_element(ident: int, value: bytes) -> bytes:
    return bytes((ident, len(value))) + value


def _common_ies(settings: WiFiSettings) -> bytes:
    return (build_information_element(0, settings.ssid.encode("utf-8"))
            + build_information_element(1, _hex_bytes(settings.supported_rates_hex)))


def _extended_rates(settings: WiFiSettings) -> bytes:
    rates = _hex_bytes(settings.extended_supported_rates_hex)
    return build_information_element(50, rates) if rates else b""


def _fixed_parameters(settings: WiFiSettings) -> bytes:
    return (int(settings.timestamp).to_bytes(8, "little")
            + int(settings.beacon_interval_tu).to_bytes(2, "little")
            + int(settings.capability_information).to_bytes(2, "little"))


def _ds_parameter(settings: WiFiSettings) -> bytes:
    channel = settings.channel if settings.ds_channel_auto else settings.ds_channel
    return build_information_element(3, bytes((int(channel),)))


def _erp(settings: WiFiSettings) -> bytes:
    return build_information_element(42, bytes((int(settings.erp_information),)))


def _additional_ies(settings: WiFiSettings) -> bytes:
    data = _hex_bytes(settings.additional_ies_hex)
    cursor = 0
    while cursor < len(data):
        if cursor + 2 > len(data) or cursor + 2 + data[cursor + 1] > len(data):
            raise ValueError("Additional IEs must contain complete Element ID / Length / Value records")
        cursor += 2 + data[cursor + 1]
    return data


def build_beacon_psdu(settings: WiFiSettings) -> bytes:
    # IEEE 802.11-2024 Table 9-62; retain the existing Beacon octet order.
    frame = (build_management_header(settings, WiFiPSDUSource.BEACON)
             + _fixed_parameters(settings) + _common_ies(settings)
             + _ds_parameter(settings)
             + build_information_element(5, _hex_bytes(settings.tim_hex))
             + _erp(settings) + _extended_rates(settings) + _additional_ies(settings))
    return _with_fcs(frame, settings)


def build_probe_request_psdu(settings: WiFiSettings) -> bytes:
    # Table 9-68: no fixed parameters; DSSS is optional for this non-RM profile.
    frame = (build_management_header(settings, WiFiPSDUSource.PROBE_REQUEST)
             + _common_ies(settings) + _extended_rates(settings) + _additional_ies(settings))
    return _with_fcs(frame, settings)


def build_probe_response_psdu(settings: WiFiSettings) -> bytes:
    # Table 9-69: ERP response fixed fields and IEs, without Beacon-only TIM.
    frame = (build_management_header(settings, WiFiPSDUSource.PROBE_RESPONSE)
             + _fixed_parameters(settings) + _common_ies(settings)
             + _ds_parameter(settings) + _erp(settings) + _extended_rates(settings) + _additional_ies(settings))
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
    if source == WiFiPSDUSource.PROBE_REQUEST:
        return build_probe_request_psdu(settings)
    if source == WiFiPSDUSource.PROBE_RESPONSE:
        return build_probe_response_psdu(settings)
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
