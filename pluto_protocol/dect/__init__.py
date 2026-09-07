"""Classic DECT protocol decoding for the shared packet analyzer."""

from pluto_protocol.dect.classic import DectClassicDecoder
from pluto_protocol.dect.carriers import (
    DECT_CARRIER_PLANS,
    DectCarrier,
    DectCarrierPlan,
    carrier_by_identity,
    carrier_plan,
)
from pluto_protocol.dect.common import (
    dect_p_range,
    r_crc_bits,
    r_crc_valid,
    x_crc_bits,
    x_crc_valid,
)
from pluto_protocol.dect.rf_modulation import (
    CASE_B_DEFINITIONS,
    DectCaseBFormat,
    DectRFPattern,
    DectScramblingMode,
    DECT_RF_MODULATION_LIMITS_HZ,
    case_a_bits,
    case_b_bits,
    identify_rf_pattern,
    deviation_in_limits,
    maximum_dsv,
)

__all__ = [
    "DECT_CARRIER_PLANS",
    "DectClassicDecoder",
    "DectCarrier",
    "DectCarrierPlan",
    "carrier_by_identity",
    "carrier_plan",
    "dect_p_range",
    "r_crc_bits",
    "r_crc_valid",
    "x_crc_bits",
    "x_crc_valid",
    "CASE_B_DEFINITIONS",
    "DectCaseBFormat",
    "DectRFPattern",
    "DectScramblingMode",
    "DECT_RF_MODULATION_LIMITS_HZ",
    "case_a_bits",
    "case_b_bits",
    "identify_rf_pattern",
    "deviation_in_limits",
    "maximum_dsv",
]
