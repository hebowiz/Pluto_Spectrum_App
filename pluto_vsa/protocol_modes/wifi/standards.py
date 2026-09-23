"""IEEE Std 802.11-2024 Non-HT / ERP limits, checked against the supplied PDF.

See docs/verification/vsa/wifi/measurement-review.md for pages and conditions.
These constants do not establish receiver accuracy or regulatory compliance.
"""
from dataclasses import dataclass

CURRENT_REVISION = "IEEE Std 802.11-2024"
CURRENT_MEASUREMENT_CLAUSES_VERIFIED = True
REFERENCE_REVISION = CURRENT_REVISION

# Table 17-20, 17.3.9.7.4; ERP inherits via 18.4.7.1.
REFERENCE_RCE_LIMIT_DB = {6:-5.,9:-8.,12:-10.,18:-13.,24:-16.,36:-19.,48:-22.,54:-25.}
# 17.3.9.8, Eq. (17-28): 52 tones, >=20 frames, >=16 DATA
# symbols/frame and random data. Arithmetic mean of packet RMS in printed eq.
REFERENCE_MIN_PACKETS = 20
REFERENCE_MIN_SYMBOLS = 16
# 17.3.9.3, Figure 17-13: 100 kHz RBW, 30 kHz VBW.
REFERENCE_MASK_OFFSETS_HZ = (9e6,11e6,20e6,30e6)
REFERENCE_MASK_LEVELS_DBR = (0.,-20.,-28.,-40.)
REFERENCE_MASK_ABSOLUTE_DBM_MHZ = -39.
REFERENCE_RBW_HZ = 100_000.
REFERENCE_VBW_HZ = 30_000.
# 17.3.9.7.2 and .3. Leakage: max(P - 15, -20) dBm for non-VHT STAs.
REFERENCE_LEAKAGE_LIMIT_DB = -15.
REFERENCE_LEAKAGE_ABSOLUTE_DBM = -20.
REFERENCE_FLATNESS_INNER_DB = (-4.,4.)
REFERENCE_FLATNESS_EDGE_DB = (-6.,4.)


@dataclass(frozen=True)
class ReferenceProfile:
    phy: str
    band: str
    frequency_tolerance_ppm: float
    frequency_clause: str
    clock_clause: str
    inheritance_clause: str = ""


def reference_profile(center_hz):
    # Selection describes the reviewed reference, not band recognition from bits.
    if 2.4e9 <= center_hz < 2.5e9:
        return ReferenceProfile("ERP-OFDM","2.4 GHz",25.,"18.4.7.4","18.4.7.5","18.4.7.1")
    if 5e9 <= center_hz < 6e9:
        return ReferenceProfile("Non-HT OFDM","5 GHz",20.,"17.3.9.5","17.3.9.6")
    return None


def reference(clause):
    return f"{REFERENCE_REVISION} {clause}"
