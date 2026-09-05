"""Regional Classic DECT carrier plans shared by VSA and VSG."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DectCarrier:
    channel: int | str
    center_frequency_hz: float

    @property
    def label(self) -> str:
        prefix = str(self.channel) if isinstance(self.channel, str) else f"c={self.channel}"
        return f"{prefix}  {self.center_frequency_hz / 1e6:.3f} MHz"


@dataclass(frozen=True)
class DectCarrierPlan:
    plan_id: str
    label: str
    carriers: tuple[DectCarrier, ...]


def _plan(
    plan_id: str,
    label: str,
    frequencies_mhz: tuple[float, ...],
    *,
    first_channel: int = 0,
    channels: tuple[int | str, ...] | None = None,
) -> DectCarrierPlan:
    identifiers = channels or tuple(
        first_channel + index for index in range(len(frequencies_mhz))
    )
    if len(identifiers) != len(frequencies_mhz):
        raise ValueError("carrier identifiers and frequencies must have equal length")
    return DectCarrierPlan(
        plan_id,
        label,
        tuple(
            DectCarrier(channel, frequency * 1e6)
            for channel, frequency in zip(identifiers, frequencies_mhz)
        ),
    )


DECT_CARRIER_PLANS = (
    _plan(
        "etsi_1880", "ETSI / Europe 1880–1900 MHz",
        tuple(1897.344 - 1.728 * channel for channel in range(10)),
    ),
    _plan(
        "dect_6_us", "DECT 6.0 / US 1920–1930 MHz",
        tuple(1921.536 + 1.728 * channel for channel in range(5)),
    ),
    _plan(
        "j_dect", "JP-DECT / Japan 1885–1905 MHz",
        (
            1885.248, 1886.976, 1888.704, 1890.432, 1892.160, 1893.888,
            1895.616, 1897.344, 1899.072, 1900.800, 1902.528, 1904.256,
        ),
        channels=("F7", "F8", "F9", "Fa", "Fb", "F0", "F1", "F2", "F3", "F4", "F5", "F6"),
    ),
    _plan(
        "etsi_ext_1935", "ETSI extended 1935–1960 MHz",
        tuple(1937.088 + 1.728 * channel for channel in range(14)),
        first_channel=10,
    ),
    _plan(
        "etsi_ext_2010", "ETSI extended 2010–2025 MHz",
        tuple(2011.392 + 1.728 * channel for channel in range(8)),
        first_channel=25,
    ),
)


def carrier_plan(plan_id: str) -> DectCarrierPlan:
    return next(plan for plan in DECT_CARRIER_PLANS if plan.plan_id == plan_id)


def carrier_by_identity(plan_id: str, channel: int | str) -> DectCarrier:
    target = str(channel)
    return next(
        carrier
        for carrier in carrier_plan(plan_id).carriers
        if str(carrier.channel) == target
    )
