"""Device-independent project model for generated IQ waveforms."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class StandardProfile(StrEnum):
    USER = "User"
    BLUETOOTH_BR_EDR = "Bluetooth BR / EDR"
    BLUETOOTH_LE = "Bluetooth LE"


class DataSourceKind(StrEnum):
    FIXED = "Fixed"
    PATTERN = "Pattern"
    PRBS = "PRBS"
    COMPUTED = "Computed"


class ModulationKind(StrEnum):
    GFSK = "GFSK"
    PI_4_DQPSK = "pi/4-DQPSK"
    DPSK8 = "8DPSK"


class FilterKind(StrEnum):
    NONE = "None"
    GAUSSIAN = "Gaussian"
    ROOT_RAISED_COSINE = "Root Raised Cosine"


@dataclass(frozen=True)
class ModulationDefinition:
    kind: ModulationKind = ModulationKind.GFSK
    symbol_rate_hz: float = 1_000_000.0
    filter_kind: FilterKind = FilterKind.GAUSSIAN
    filter_parameter: float = 0.5


@dataclass(frozen=True)
class FieldDefinition:
    name: str
    symbol_count: int
    data_source: DataSourceKind = DataSourceKind.FIXED
    data: str = "0"
    modulation: ModulationDefinition = field(default_factory=ModulationDefinition)


@dataclass(frozen=True)
class PowerEnvelopeDefinition:
    enabled: bool = True
    on_level_db: float = 0.0
    idle_level_db: float = -120.0
    rise_symbols: float = 1.0
    fall_symbols: float = 1.0
    shape: str = "Cosine"


@dataclass(frozen=True)
class WaveformProject:
    name: str = "Untitled Waveform"
    standard: StandardProfile = StandardProfile.USER
    sample_rate_hz: float = 8_000_000.0
    samples_per_symbol: int = 8
    repeat_count: int = 1
    fields: tuple[FieldDefinition, ...] = ()
    power_envelope: PowerEnvelopeDefinition = field(
        default_factory=PowerEnvelopeDefinition
    )


@dataclass(frozen=True)
class ValidationIssue:
    path: str
    message: str


def validate_project(project: WaveformProject) -> tuple[ValidationIssue, ...]:
    issues: list[ValidationIssue] = []
    if not project.name.strip():
        issues.append(ValidationIssue("name", "Project name must not be empty."))
    if project.sample_rate_hz <= 0.0:
        issues.append(
            ValidationIssue("sample_rate_hz", "Sample rate must be positive.")
        )
    if project.samples_per_symbol < 2:
        issues.append(
            ValidationIssue(
                "samples_per_symbol", "Samples per symbol must be at least 2."
            )
        )
    if project.repeat_count < 1:
        issues.append(ValidationIssue("repeat_count", "Repeat count must be positive."))
    for index, packet_field in enumerate(project.fields):
        path = f"fields[{index}]"
        if not packet_field.name.strip():
            issues.append(ValidationIssue(f"{path}.name", "Field name is required."))
        if packet_field.symbol_count < 1:
            issues.append(
                ValidationIssue(
                    f"{path}.symbol_count", "Field must contain at least one symbol."
                )
            )
        if packet_field.modulation.symbol_rate_hz <= 0.0:
            issues.append(
                ValidationIssue(
                    f"{path}.modulation.symbol_rate_hz",
                    "Symbol rate must be positive.",
                )
            )
    return tuple(issues)


def create_default_project() -> WaveformProject:
    return WaveformProject(
        fields=(
            FieldDefinition(
                name="Packet Field",
                symbol_count=64,
                data_source=DataSourceKind.PATTERN,
                data="10",
            ),
        )
    )
