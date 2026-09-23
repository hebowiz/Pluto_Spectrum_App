"""Wi-Fi settings validation shared by project loading, editor and generator."""
from __future__ import annotations
import math
from .common import LEGACY_RATES
from .mac import _hex_bytes, build_psdu


def validate_wifi_settings(s):
    issues = []
    for name, high in (("frame_control",65535),("duration_id",65535),("fragment_number",15),
                        ("timestamp",2**64-1),("capability_information",65535),("erp_information",255)):
        if not 0 <= getattr(s,name) <= high:
            issues.append((name,f"{name} must be between 0 and {high}."))
    for name in ("destination_address","source_address"):
        value = getattr(s,name)
        if name == "source_address" and not value:
            continue
        try:
            parts = value.split(":")
            if len(parts) != 6 or not all(len(p) == 2 and 0 <= int(p,16) <= 255 for p in parts):
                raise ValueError("Invalid MAC address")
        except ValueError:
            issues.append((name,f"{name} must use XX:XX:XX:XX:XX:XX notation."))
    if not 1 <= s.ds_channel <= 13:
        issues.append(("ds_channel","DS channel must be 1-13."))
    try:
        rates = _hex_bytes(s.supported_rates_hex)
        tim = _hex_bytes(s.tim_hex)
        if not 1<=len(rates)<=8 or any((v & 127)==0 for v in rates):
            issues.append(("supported_rates_hex","Supported Rates requires 1-8 nonzero rate octets."))
        if not 4<=len(tim)<=254 or tim[1]==0 or tim[0]>=tim[1]:
            issues.append(("tim_hex","TIM requires count < nonzero DTIM period, bitmap control and 1-251 bitmap bytes."))
        psdu = build_psdu(s)
        if not 1<=len(psdu)<=4095:
            issues.append(("psdu","Final PSDU length must be 1-4095 bytes, including FCS when supplied."))
        if s.legacy_rate_mbps in LEGACY_RATES:
            count = math.ceil((16+8*len(psdu)+6)/LEGACY_RATES[s.legacy_rate_mbps].n_dbps)
            minimum = 20+4*count+6
            if not math.isfinite(s.packet_period_us) or s.packet_period_us < minimum:
                issues.append(("packet_period_us",f"ERP packet period must be at least {minimum} us (PPDU + 6 us signal extension)."))
    except (ValueError,OverflowError) as error:
        issues.append(("fields",str(error)))
    return issues
