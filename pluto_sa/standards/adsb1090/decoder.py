"""Mode S parity and ADS-B field decoding.

Bit positions in this module follow the ICAO convention: the first transmitted
bit is bit 1.  Python slices are therefore deliberately documented where used.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


MODE_S_GENERATOR = 0x1FFF409


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
    return fields
