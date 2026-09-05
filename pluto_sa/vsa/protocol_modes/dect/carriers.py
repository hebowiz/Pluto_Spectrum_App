"""Compatibility re-export of shared Classic DECT carrier plans."""

from pluto_protocol.dect.carriers import (
    DECT_CARRIER_PLANS,
    DectCarrier,
    DectCarrierPlan,
    carrier_by_identity,
    carrier_plan,
)

__all__ = [
    "DECT_CARRIER_PLANS",
    "DectCarrier",
    "DectCarrierPlan",
    "carrier_by_identity",
    "carrier_plan",
]
