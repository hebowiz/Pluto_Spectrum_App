"""Mode S parity and ADS-B field decoding.

Bit positions in this module follow the ICAO convention: the first transmitted
bit is bit 1.  Python slices are therefore deliberately documented where used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


MODE_S_GENERATOR = 0x1FFF409

ADDRESS_PARITY_DFS = frozenset({0, 4, 5, 16, 20, 21, *range(24, 32)})
CRC_PARITY_DFS = frozenset({17, 18, 19})


@dataclass(frozen=True)
class ModeSParity:
    """DF-aware interpretation of the 24-bit Mode S parity field."""

    kind: str
    remainder: int
    valid: bool | None
    recovered_icao: str | None = None
    interrogator_identifier: int | None = None


def bits_to_int(bits: Iterable[int]) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | (int(bit) & 1)
    return value


def bits_to_hex(bits: np.ndarray) -> str:
    values = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if values.size % 4:
        raise ValueError("bit count must be a multiple of four")
    return f"{bits_to_int(values):0{values.size // 4}X}"


def mode_s_crc_remainder(bits: np.ndarray) -> int:
    """Return the 24-bit Mode S polynomial remainder over the full message."""
    values = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if values.size not in {56, 112}:
        raise ValueError("Mode S CRC accepts 56 or 112 bits")
    work = bits_to_int(values)
    for position in range(values.size - 1, 23, -1):
        if work & (1 << position):
            work ^= MODE_S_GENERATOR << (position - 24)
    return int(work & 0xFFFFFF)


def classify_mode_s_parity(bits: np.ndarray) -> ModeSParity:
    """Interpret the polynomial syndrome according to the downlink format.

    Address-parity replies do not carry a stand-alone CRC.  Their syndrome is
    the aircraft address when the message is error-free, so validity remains
    unknown until that address can be corroborated by another message.
    """

    values = np.asarray(bits, dtype=np.uint8).reshape(-1)
    remainder = mode_s_crc_remainder(values)
    downlink_format = bits_to_int(values[0:5])
    if downlink_format in ADDRESS_PARITY_DFS:
        return ModeSParity(
            kind="address",
            remainder=remainder,
            valid=None,
            recovered_icao=f"{remainder:06X}",
        )
    if downlink_format == 11:
        # The all-call parity/interrogator field overlays at most seven low
        # syndrome bits.  Non-zero upper bits indicate an invalid reply.
        return ModeSParity(
            kind="interrogator",
            remainder=remainder,
            valid=(remainder & 0xFFFF80) == 0,
            interrogator_identifier=remainder & 0x7F,
        )
    if downlink_format in CRC_PARITY_DFS:
        return ModeSParity(
            kind="crc",
            remainder=remainder,
            valid=remainder == 0,
        )
    return ModeSParity(kind="unsupported", remainder=remainder, valid=None)


def decode_mode_s_header_fields(bits: np.ndarray) -> dict[str, object]:
    """Decode DF-dependent fields immediately following the five-bit DF."""

    values = np.asarray(bits, dtype=np.uint8).reshape(-1)
    downlink_format = bits_to_int(values[0:5])
    if downlink_format in {4, 5, 20, 21}:
        return {"flight_status": bits_to_int(values[5:8])}
    if downlink_format in {11, 17}:
        return {"capability": bits_to_int(values[5:8])}
    if downlink_format == 18:
        return {"control_field": bits_to_int(values[5:8])}
    if downlink_format in {0, 16}:
        return {
            "vertical_status": int(values[5]),
            "cross_link_capability": int(values[6]),
            "sensitivity_level": bits_to_int(values[7:10]),
        }
    return {}


def _decode_callsign(me: np.ndarray) -> str:
    charset = "#ABCDEFGHIJKLMNOPQRSTUVWXYZ#####_###############0123456789######"
    characters = []
    for start in range(8, 56, 6):
        code = bits_to_int(me[start : start + 6])
        characters.append(charset[code] if code < len(charset) else "#")
    return "".join(characters).replace("_", " ").replace("#", "").strip()


def _decode_altitude(ac12: int) -> int | None:
    # Q=1 is the common 25-foot encoding. Gillham/Q=0 can be added without
    # changing the public result contract.
    if ac12 & 0x10:
        n = ((ac12 & 0xFE0) >> 1) | (ac12 & 0xF)
        return int(n * 25 - 1000)
    return None


def _cpr_nl(latitude_deg: float) -> int:
    """Return the number of CPR longitude zones at *latitude_deg*."""

    latitude = abs(float(latitude_deg))
    if latitude >= 87.0:
        return 1
    if latitude < 1e-12:
        return 59
    nz = 15.0
    denominator = np.cos(np.radians(latitude)) ** 2
    argument = 1.0 - (1.0 - np.cos(np.pi / (2.0 * nz))) / denominator
    argument = float(np.clip(argument, -1.0, 1.0))
    return int(np.floor(2.0 * np.pi / np.arccos(argument)))


def decode_global_airborne_cpr(
    even_latitude: int,
    even_longitude: int,
    odd_latitude: int,
    odd_longitude: int,
    *,
    use_odd: bool,
) -> tuple[float, float] | None:
    """Decode a globally unambiguous airborne CPR even/odd frame pair.

    The caller owns temporal and ICAO pairing. ``use_odd`` selects the most
    recently received frame, as required when resolving longitude.
    """

    scale = 131_072.0
    yz_even = float(even_latitude) / scale
    xz_even = float(even_longitude) / scale
    yz_odd = float(odd_latitude) / scale
    xz_odd = float(odd_longitude) / scale
    latitude_index = int(np.floor(59.0 * yz_even - 60.0 * yz_odd + 0.5))
    latitude_even = 6.0 * ((latitude_index % 60) + yz_even)
    latitude_odd = (360.0 / 59.0) * ((latitude_index % 59) + yz_odd)
    if latitude_even >= 270.0:
        latitude_even -= 360.0
    if latitude_odd >= 270.0:
        latitude_odd -= 360.0
    if not (-90.0 <= latitude_even <= 90.0 and -90.0 <= latitude_odd <= 90.0):
        return None
    nl_even = _cpr_nl(latitude_even)
    nl_odd = _cpr_nl(latitude_odd)
    if nl_even != nl_odd:
        return None
    longitude_index = int(
        np.floor(xz_even * (nl_even - 1) - xz_odd * nl_even + 0.5)
    )
    if use_odd:
        latitude = latitude_odd
        longitude_zones = max(nl_odd - 1, 1)
        longitude = (360.0 / longitude_zones) * (
            (longitude_index % longitude_zones) + xz_odd
        )
    else:
        latitude = latitude_even
        longitude_zones = max(nl_even, 1)
        longitude = (360.0 / longitude_zones) * (
            (longitude_index % longitude_zones) + xz_even
        )
    if longitude >= 180.0:
        longitude -= 360.0
    return float(latitude), float(longitude)


def decode_adsb_fields(bits: np.ndarray) -> dict[str, object]:
    """Decode stable DF17/18 fields without maintaining cross-frame state."""
    values = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if values.size != 112:
        return {}
    downlink_format = bits_to_int(values[0:5])
    if downlink_format not in {17, 18}:
        return {}
    me = values[32:88]
    type_code = bits_to_int(me[0:5])
    fields: dict[str, object] = {
        "type_code": type_code,
        "icao_address": f"{bits_to_int(values[8:32]):06X}",
    }
    if 1 <= type_code <= 4:
        fields["emitter_category"] = bits_to_int(me[5:8])
        fields["callsign"] = _decode_callsign(me)
    elif 5 <= type_code <= 8:
        fields.update(
            {
                "position_type": "surface",
                "air_ground": "ground",
                "cpr_format": "odd" if int(me[21]) else "even",
                "cpr_latitude": bits_to_int(me[22:39]),
                "cpr_longitude": bits_to_int(me[39:56]),
            }
        )
    elif 9 <= type_code <= 18 or 20 <= type_code <= 22:
        altitude = _decode_altitude(bits_to_int(me[8:20]))
        fields.update(
            {
                "position_type": "airborne",
                "air_ground": "airborne",
                "altitude_ft": altitude,
                "time_flag": int(me[20]),
                "cpr_format": "odd" if int(me[21]) else "even",
                "cpr_latitude": bits_to_int(me[22:39]),
                "cpr_longitude": bits_to_int(me[39:56]),
            }
        )
    elif type_code == 19:
        subtype = bits_to_int(me[5:8])
        fields["velocity_subtype"] = subtype
        fields["air_ground"] = "airborne"
        if subtype in {1, 2}:
            ew_raw = bits_to_int(me[14:24])
            ns_raw = bits_to_int(me[25:35])
            ew = None if ew_raw == 0 else ew_raw - 1
            ns = None if ns_raw == 0 else ns_raw - 1
            if ew is not None and int(me[13]):
                ew = -ew
            if ns is not None and int(me[24]):
                ns = -ns
            fields["east_west_velocity_kt"] = ew
            fields["north_south_velocity_kt"] = ns
            if ew is not None and ns is not None:
                fields["ground_speed_kt"] = float(np.hypot(ew, ns))
                fields["track_deg"] = float((np.degrees(np.arctan2(ew, ns)) + 360.0) % 360.0)
        vertical_rate_raw = bits_to_int(me[37:46])
        if vertical_rate_raw:
            vertical_rate_fpm = float((vertical_rate_raw - 1) * 64)
            if int(me[36]):
                vertical_rate_fpm = -vertical_rate_fpm
            fields["vertical_rate_fpm"] = vertical_rate_fpm
            fields["vertical_rate_source"] = (
                "GNSS" if int(me[35]) else "barometric"
            )
    return fields
