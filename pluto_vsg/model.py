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


class PayloadSourceKind(StrEnum):
    FIXED = "Fixed"
    PATTERN = "Pattern"
    PRBS9 = "PRBS-9"


class BluetoothPacketKind(StrEnum):
    DH1 = "DH1"
    DH1_2 = "2-DH1"
    DH1_3 = "3-DH1"


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
    logical_bit_count: int | None = None
    data_source: DataSourceKind = DataSourceKind.FIXED
    data: str = "0"
    modulation: ModulationDefinition = field(default_factory=ModulationDefinition)
    children: tuple[FieldDefinition, ...] = ()


@dataclass(frozen=True)
class PowerEnvelopeDefinition:
    enabled: bool = True
    on_level_db: float = 0.0
    idle_level_db: float = -120.0
    rise_symbols: float = 1.0
    fall_symbols: float = 1.0
    rise_delay_symbols: float = 0.0
    fall_delay_symbols: float = -1.0
    shape: str = "Cosine"


@dataclass(frozen=True)
class BluetoothBRSettings:
    """Packet and RF-independent baseband settings for a BR/EDR DH1 packet."""

    packet_kind: BluetoothPacketKind = BluetoothPacketKind.DH1
    lap: int = 0xC6967E
    uap: int = 0x6B
    clock_6_1: int = 0x2B
    lt_addr: int = 1
    flow: int = 1
    arqn: int = 0
    seqn: int = 0
    whitening_enabled: bool = True
    payload_length_bytes: int = 27
    payload_source: PayloadSourceKind = PayloadSourceKind.PRBS9
    payload_pattern: str = "10101010"
    frequency_deviation_hz: float = 160_000.0
    carrier_frequency_offset_hz: float = 0.0
    gaussian_bt: float = 0.5
    edr_guard_symbols: int = 5
    edr_rolloff: float = 0.4
    edr_relative_power_db: float = 0.0
    pre_idle_symbols: int = 8
    post_idle_symbols: int = 8


@dataclass(frozen=True)
class WaveformProject:
    name: str = "Untitled Waveform"
    standard: StandardProfile = StandardProfile.USER
    sample_rate_hz: float = 8_000_000.0
    samples_per_symbol: int = 8
    repeat_count: int = 1
    center_frequency_hz: float = 2_441_000_000.0
    fields: tuple[FieldDefinition, ...] = ()
    power_envelope: PowerEnvelopeDefinition = field(
        default_factory=PowerEnvelopeDefinition
    )
    bluetooth_br: BluetoothBRSettings | None = None


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
    if project.center_frequency_hz < 0.0:
        issues.append(
            ValidationIssue(
                "center_frequency_hz", "Center frequency must not be negative."
            )
        )
    envelope = project.power_envelope
    if envelope.rise_symbols < 0.0 or envelope.fall_symbols < 0.0:
        issues.append(
            ValidationIssue(
                "power_envelope.ramp_symbols",
                "Ramp durations must not be negative.",
            )
        )
    if envelope.shape not in {"Cosine", "Linear"}:
        issues.append(
            ValidationIssue(
                "power_envelope.shape", "Ramp shape must be Cosine or Linear."
            )
        )

    def validate_field(packet_field: FieldDefinition, path: str) -> None:
        if not packet_field.name.strip():
            issues.append(ValidationIssue(f"{path}.name", "Field name is required."))
        if packet_field.symbol_count < 1:
            issues.append(
                ValidationIssue(
                    f"{path}.symbol_count", "Field must contain at least one symbol."
                )
            )
        if (
            packet_field.logical_bit_count is not None
            and packet_field.logical_bit_count < 1
        ):
            issues.append(
                ValidationIssue(
                    f"{path}.logical_bit_count",
                    "Logical bit count must be positive when specified.",
                )
            )
        if packet_field.modulation.symbol_rate_hz <= 0.0:
            issues.append(
                ValidationIssue(
                    f"{path}.modulation.symbol_rate_hz",
                    "Symbol rate must be positive.",
                )
            )
        if packet_field.children:
            child_symbols = sum(child.symbol_count for child in packet_field.children)
            if child_symbols != packet_field.symbol_count:
                issues.append(
                    ValidationIssue(
                        f"{path}.children",
                        "Child transmitted-symbol counts must equal the parent field.",
                    )
                )
            if packet_field.logical_bit_count is not None and all(
                child.logical_bit_count is not None
                for child in packet_field.children
            ):
                child_bits = sum(
                    int(child.logical_bit_count) for child in packet_field.children
                )
                if child_bits != packet_field.logical_bit_count:
                    issues.append(
                        ValidationIssue(
                            f"{path}.children",
                            "Child logical-bit counts must equal the parent field.",
                        )
                    )
            for child_index, child in enumerate(packet_field.children):
                validate_field(child, f"{path}.children[{child_index}]")

    for index, packet_field in enumerate(project.fields):
        validate_field(packet_field, f"fields[{index}]")

    settings = project.bluetooth_br
    if settings is not None:
        packet_kind = BluetoothPacketKind(settings.packet_kind)
        payload_max = {
            BluetoothPacketKind.DH1: 27,
            BluetoothPacketKind.DH1_2: 54,
            BluetoothPacketKind.DH1_3: 83,
        }[packet_kind]
        integer_ranges = (
            ("lap", settings.lap, 0, 0xFFFFFF),
            ("uap", settings.uap, 0, 0xFF),
            ("clock_6_1", settings.clock_6_1, 0, 0x3F),
            ("lt_addr", settings.lt_addr, 0, 7),
            ("payload_length_bytes", settings.payload_length_bytes, 0, payload_max),
        )
        for name, value, lower, upper in integer_ranges:
            if not lower <= int(value) <= upper:
                issues.append(
                    ValidationIssue(
                        f"bluetooth_br.{name}",
                        f"Value must be between {lower} and {upper}.",
                    )
                )
        for name, value in (
            ("flow", settings.flow),
            ("arqn", settings.arqn),
            ("seqn", settings.seqn),
        ):
            if int(value) not in (0, 1):
                issues.append(
                    ValidationIssue(f"bluetooth_br.{name}", "Value must be 0 or 1.")
                )
        if settings.frequency_deviation_hz <= 0.0:
            issues.append(
                ValidationIssue(
                    "bluetooth_br.frequency_deviation_hz",
                    "Frequency deviation must be positive.",
                )
            )
        if settings.gaussian_bt <= 0.0:
            issues.append(
                ValidationIssue(
                    "bluetooth_br.gaussian_bt", "Gaussian B*T must be positive."
                )
            )
        if settings.edr_guard_symbols < 0:
            issues.append(
                ValidationIssue(
                    "bluetooth_br.edr_guard_symbols",
                    "EDR guard duration must not be negative.",
                )
            )
        if not 0.0 < settings.edr_rolloff <= 1.0:
            issues.append(
                ValidationIssue(
                    "bluetooth_br.edr_rolloff",
                    "EDR SRRC roll-off must be greater than 0 and at most 1.",
                )
            )
        if settings.pre_idle_symbols < 0 or settings.post_idle_symbols < 0:
            issues.append(
                ValidationIssue(
                    "bluetooth_br.idle_symbols", "Idle symbol counts must not be negative."
                )
            )
        if settings.payload_source in {
            PayloadSourceKind.FIXED,
            PayloadSourceKind.PATTERN,
        }:
            pattern = settings.payload_pattern.strip().replace(" ", "")
            if not pattern or any(character not in "01" for character in pattern):
                issues.append(
                    ValidationIssue(
                        "bluetooth_br.payload_pattern",
                        "Fixed and pattern payloads require a binary pattern.",
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
