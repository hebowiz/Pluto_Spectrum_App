"""Typed measurement values, references and independent decision conditions."""
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class MeasurementKind(StrEnum):
    STANDARD = "RF / PHY Measurement"
    PHY_DECODE = "PHY Decode"
    MAC_DECODE = "Packet / MAC Decode"
    DIAGNOSTIC = "Diagnostics"


class MeasurementStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    INFO = "Info"
    NOT_MEASURED = "Not Measured"
    INSUFFICIENT_DATA = "Insufficient Data"
    NOT_AVAILABLE = "Not Available"


@dataclass(frozen=True)
class MeasurementResult:
    measurement_id: str
    name: str
    measurement_kind: MeasurementKind
    value: float | str | None
    unit: str = ""
    limit: str = "—"
    status: MeasurementStatus = MeasurementStatus.INFO
    standard_reference: str | None = None
    measurement_conditions_satisfied: bool = False
    canonical_id: str | None = None
    default_visible: bool = True
    conditions: tuple[str,...] = ()
    metadata: dict[str,Any] = field(default_factory=dict)

    def row(self):
        value = "N/A" if self.value is None else (f"{self.value:.5g}" if isinstance(self.value,float) else str(self.value))
        return (self.name,value+(f" {self.unit}" if self.unit and self.value is not None else ""),self.limit,self.status.value)

    def tooltip(self):
        parts = [self.measurement_kind.value]
        if self.standard_reference:
            parts.append(self.standard_reference)
        for key,confirmed in self.metadata.get("confirmed_setup",{}).items():
            if confirmed:
                parts.append("User confirmed: " + key.replace("_"," "))
        parts.extend(self.conditions)
        return "\n".join(parts)
