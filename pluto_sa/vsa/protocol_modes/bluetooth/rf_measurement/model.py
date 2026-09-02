"""Shared immutable result contracts for Bluetooth RF measurements."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping

import numpy as np


class RFTestVerdict(StrEnum):
    PASS = "Pass"
    FAIL = "Fail"
    NOT_APPLICABLE = "N/A"


@dataclass(frozen=True)
class RFTestEligibility:
    eligible: bool
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        reasons = tuple(str(reason) for reason in self.reasons if str(reason))
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(self, "eligible", bool(self.eligible and not reasons))

    @classmethod
    def from_reasons(cls, reasons: tuple[str, ...] | list[str]) -> "RFTestEligibility":
        values = tuple(str(reason) for reason in reasons if str(reason))
        return cls(not values, values)


def _readonly_array(values: object) -> np.ndarray:
    result = np.array(values, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class BluetoothRFMeasurementResult:
    test_case_id: str
    eligibility: RFTestEligibility
    verdict: RFTestVerdict = RFTestVerdict.NOT_APPLICABLE
    metrics: Mapping[str, float | int | str | bool | None] = field(default_factory=dict)
    arrays: Mapping[str, np.ndarray] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "verdict", RFTestVerdict(self.verdict))
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        object.__setattr__(
            self,
            "arrays",
            MappingProxyType(
                {name: _readonly_array(values) for name, values in self.arrays.items()}
            ),
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        if not self.eligibility.eligible and self.verdict is not RFTestVerdict.NOT_APPLICABLE:
            raise ValueError("an ineligible RF measurement must have an N/A verdict")
