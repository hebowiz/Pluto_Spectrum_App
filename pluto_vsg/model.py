"""Device-independent project model for generated IQ waveforms."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from pluto_protocol.bluetooth.hdt import HDTRate


class StandardProfile(StrEnum):
    USER = "User"
    BLUETOOTH_BR_EDR = "Bluetooth BR / EDR"
    BLUETOOTH_LE = "Bluetooth LE"
    BLUETOOTH_HDT = "Bluetooth HDT"


class DataSourceKind(StrEnum):
    FIXED = "Fixed"
    PATTERN = "Pattern"
    PRBS = "PRBS"
    COMPUTED = "Computed"


class ModulationKind(StrEnum):
    GFSK = "GFSK"
    PI_4_DQPSK = "pi/4-DQPSK"
    DPSK8 = "8DPSK"
    PI_4_QPSK = "pi/4-QPSK"
    PSK8 = "8PSK"
    QAM16 = "16QAM"


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
    DH3 = "DH3"
    DH5 = "DH5"
    DH1_2 = "2-DH1"
    DH3_2 = "2-DH3"
    DH5_2 = "2-DH5"
    DH1_3 = "3-DH1"
    DH3_3 = "3-DH3"
    DH5_3 = "3-DH5"


class BluetoothLEPhy(StrEnum):
    LE_1M = "LE 1M"
    LE_2M = "LE 2M"


class BluetoothLEPayloadType(StrEnum):
    PRBS9 = "PRBS9"
    F0 = "11110000"
    AA = "10101010"
    PRBS15 = "PRBS15"
    FF = "11111111"
    ZERO = "00000000"
    OF = "00001111"
    FIVE = "01010101"


class BluetoothLEPayloadSourceKind(StrEnum):
    FIXED = "Fixed"
    PATTERN = "Pattern"
    PRBS9 = "PRBS9"
    PRBS15 = "PRBS15"


_BLUETOOTH_LE_PAYLOAD_CODES = {
    BluetoothLEPayloadType.PRBS9: 0x0,
    BluetoothLEPayloadType.F0: 0x1,
    BluetoothLEPayloadType.AA: 0x2,
    BluetoothLEPayloadType.PRBS15: 0x3,
    BluetoothLEPayloadType.FF: 0x4,
    BluetoothLEPayloadType.ZERO: 0x5,
    BluetoothLEPayloadType.OF: 0x6,
    BluetoothLEPayloadType.FIVE: 0x7,
}


def bluetooth_le_payload_code(payload_type: BluetoothLEPayloadType | str) -> int:
    return _BLUETOOTH_LE_PAYLOAD_CODES[BluetoothLEPayloadType(payload_type)]


_BLUETOOTH_PACKET_PROPERTIES = {
    BluetoothPacketKind.DH1: (27, 0x4, 1, 1),
    BluetoothPacketKind.DH3: (183, 0xB, 1, 3),
    BluetoothPacketKind.DH5: (339, 0xF, 1, 5),
    BluetoothPacketKind.DH1_2: (54, 0x4, 2, 1),
    BluetoothPacketKind.DH3_2: (367, 0xA, 2, 3),
    BluetoothPacketKind.DH5_2: (679, 0xE, 2, 5),
    BluetoothPacketKind.DH1_3: (83, 0x8, 3, 1),
    BluetoothPacketKind.DH3_3: (552, 0xB, 3, 3),
    BluetoothPacketKind.DH5_3: (1021, 0xF, 3, 5),
}


def bluetooth_packet_properties(
    packet_kind: BluetoothPacketKind | str,
) -> tuple[int, int, int, int]:
    """Return max payload, TYPE, bits/symbol and occupied slots."""

    return _BLUETOOTH_PACKET_PROPERTIES[BluetoothPacketKind(packet_kind)]


def bluetooth_packet_is_edr(packet_kind: BluetoothPacketKind | str) -> bool:
    return bluetooth_packet_properties(packet_kind)[2] > 1


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
    relative_power_db: float = 0.0
    children: tuple[FieldDefinition, ...] = ()


@dataclass(frozen=True)
class PowerEnvelopeDefinition:
    enabled: bool = True
    on_level_db: float = 0.0
    idle_level_db: float = -120.0
    rise_symbols: float = 1.0
    fall_symbols: float = 1.0
    rise_delay_symbols: float = -1.0
    fall_delay_symbols: float = 1.0
    shape: str = "Cosine"


@dataclass(frozen=True)
class BluetoothBRSettings:
    """Packet and RF-independent baseband settings for a BR/EDR DHx packet."""

    packet_kind: BluetoothPacketKind = BluetoothPacketKind.DH1
    lap: int = 0xC6967E
    uap: int = 0x6B
    clock_6_1: int = 0x2B
    lt_addr: int = 1
    flow: int = 1
    arqn: int = 0
    seqn: int = 0
    whitening_enabled: bool = False
    payload_length_bytes: int = 27
    payload_source: PayloadSourceKind = PayloadSourceKind.PRBS9
    payload_pattern: str = "10101010"
    frequency_deviation_hz: float = 160_000.0
    carrier_frequency_offset_hz: float = 0.0
    gaussian_bt: float = 0.5
    edr_guard_symbols: int = 5
    edr_guard_relative_power_db: float = 0.0
    edr_guard_ramp_in_symbols: float = 1.0
    edr_guard_ramp_out_symbols: float = 1.0
    edr_guard_ramp_shape: str = "Cosine"
    edr_rolloff: float = 0.4
    edr_relative_power_db: float = 0.0
    pre_idle_symbols: int = 8
    post_idle_symbols: int = 8


@dataclass(frozen=True)
class BluetoothLESettings:
    """Editable Bluetooth LE uncoded packet-bit and modulation settings."""

    phy: BluetoothLEPhy = BluetoothLEPhy.LE_1M
    preamble_bits: str = "10101010"
    sync_word_bits: str = "01101011011111011001000101110001"
    pdu_header_bits: str = "00000000"
    payload_type: BluetoothLEPayloadType = BluetoothLEPayloadType.AA
    payload_source: BluetoothLEPayloadSourceKind = (
        BluetoothLEPayloadSourceKind.PATTERN
    )
    payload_pattern: str = "10101010"
    payload_length_bytes: int = 37
    crc_enabled: bool = True
    crc_init: int = 0x555555
    whitening_enabled: bool = True
    whitening_channel_index: int = 37
    rf_test_interval_enabled: bool = False
    frequency_deviation_hz: float = 250_000.0
    gaussian_bt: float = 0.5
    pre_idle_symbols: int = 8
    post_idle_symbols: int = 8


@dataclass(frozen=True)
class BluetoothHDTSettings:
    """Editable Bluetooth-derived HDT RF test waveform settings."""

    rate: HDTRate = HDTRate.HDT6
    payload_length_bytes: int = 255
    payload_source: PayloadSourceKind = PayloadSourceKind.PRBS9
    payload_pattern: str = "10101010"
    training_enabled: bool = True
    rrc_rolloff: float = 0.4
    pre_idle_symbols: int = 16
    post_idle_symbols: int = 16


@dataclass(frozen=True)
class WaveformProject:
    name: str = "Untitled Waveform"
    standard: StandardProfile = StandardProfile.USER
    sample_rate_hz: float = 8_000_000.0
    samples_per_symbol: int = 8
    repeat_count: int = 1
    # Complete repetition period. ``None`` keeps version-1 project files
    # compatible by deriving the period from Pre/Post Idle and the envelope.
    period_symbols: float | None = None
    center_frequency_hz: float = 2_441_000_000.0
    fields: tuple[FieldDefinition, ...] = ()
    power_envelope: PowerEnvelopeDefinition = field(
        default_factory=PowerEnvelopeDefinition
    )
    bluetooth_br: BluetoothBRSettings | None = None
    bluetooth_le: BluetoothLESettings | None = None
    bluetooth_hdt: BluetoothHDTSettings | None = None


@dataclass(frozen=True)
class ValidationIssue:
    path: str
    message: str


def packet_symbol_count(project: WaveformProject) -> int:
    return int(sum(packet_field.symbol_count for packet_field in project.fields))


def waveform_timing_samples(project: WaveformProject) -> tuple[int, int, int, int]:
    """Return active start/stop, minimum period and effective period in samples."""

    sps = int(project.samples_per_symbol)
    packet_samples = packet_symbol_count(project) * sps
    envelope = project.power_envelope
    if envelope.enabled:
        rise_start = round(envelope.rise_delay_symbols * sps)
        rise_count = round(envelope.rise_symbols * sps)
        fall_start = packet_samples + round(envelope.fall_delay_symbols * sps)
        fall_count = round(envelope.fall_symbols * sps)
        active_start = min(0, rise_start)
        active_stop = max(packet_samples, fall_start + fall_count)
    else:
        active_start, active_stop = 0, packet_samples
    settings = project.bluetooth_br or project.bluetooth_le or project.bluetooth_hdt
    pre_idle = int(getattr(settings, "pre_idle_symbols", 0)) * sps
    post_idle = int(getattr(settings, "post_idle_symbols", 0)) * sps
    minimum_period = pre_idle + active_stop - active_start
    if project.period_symbols is None:
        period = minimum_period + post_idle
    else:
        period = round(float(project.period_symbols) * sps)
    return active_start, active_stop, minimum_period, period


def minimum_period_symbols(project: WaveformProject) -> float:
    return waveform_timing_samples(project)[2] / float(project.samples_per_symbol)


def effective_period_symbols(project: WaveformProject) -> float:
    return waveform_timing_samples(project)[3] / float(project.samples_per_symbol)


def effective_post_idle_symbols(project: WaveformProject) -> float:
    _, _, minimum, period = waveform_timing_samples(project)
    return max(0, period - minimum) / float(project.samples_per_symbol)


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
    elif project.repeat_count > 1000:
        issues.append(
            ValidationIssue(
                "repeat_count", "Pluto VSG supports at most 1000 packet repetitions."
            )
        )
    if project.period_symbols is not None:
        if project.period_symbols <= 0.0:
            issues.append(
                ValidationIssue("period_symbols", "Packet period must be positive.")
            )
        elif waveform_timing_samples(project)[3] < waveform_timing_samples(project)[2]:
            issues.append(
                ValidationIssue(
                    "period_symbols",
                    "Packet period is shorter than Pre Idle plus the complete "
                    "Ramp Up/packet/Ramp Down interval.",
                )
            )
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
        payload_max = bluetooth_packet_properties(packet_kind)[0]
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
        if not -120.0 <= settings.edr_guard_relative_power_db <= 20.0:
            issues.append(
                ValidationIssue(
                    "bluetooth_br.edr_guard_relative_power_db",
                    "EDR guard relative power must be between -120 and +20 dB.",
                )
            )
        if (
            settings.edr_guard_ramp_in_symbols < 0.0
            or settings.edr_guard_ramp_out_symbols < 0.0
        ):
            issues.append(
                ValidationIssue(
                    "bluetooth_br.edr_guard_ramp_symbols",
                    "EDR guard ramp durations must not be negative.",
                )
            )
        if (
            settings.edr_guard_ramp_in_symbols
            + settings.edr_guard_ramp_out_symbols
            > settings.edr_guard_symbols
        ):
            issues.append(
                ValidationIssue(
                    "bluetooth_br.edr_guard_ramp_symbols",
                    "EDR guard Ramp In plus Ramp Out must not exceed Guard duration.",
                )
            )
        if settings.edr_guard_ramp_shape not in {"Cosine", "Linear"}:
            issues.append(
                ValidationIssue(
                    "bluetooth_br.edr_guard_ramp_shape",
                    "EDR guard ramp shape must be Cosine or Linear.",
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

    le_settings = project.bluetooth_le
    if le_settings is not None:
        if not 0 <= int(le_settings.payload_length_bytes) <= 255:
            issues.append(
                ValidationIssue(
                    "bluetooth_le.payload_length_bytes",
                    "LE test payload length must be between 0 and 255 bytes.",
                )
            )
        if le_settings.frequency_deviation_hz <= 0.0:
            issues.append(
                ValidationIssue(
                    "bluetooth_le.frequency_deviation_hz",
                    "LE frequency deviation must be positive.",
                )
            )
        if le_settings.gaussian_bt <= 0.0:
            issues.append(
                ValidationIssue(
                    "bluetooth_le.gaussian_bt",
                    "LE Gaussian B*T must be positive.",
                )
            )
        if le_settings.pre_idle_symbols < 0 or le_settings.post_idle_symbols < 0:
            issues.append(
                ValidationIssue(
                    "bluetooth_le.idle_symbols",
                    "LE idle symbol counts must not be negative.",
                )
            )
        bit_fields = (
            ("preamble_bits", le_settings.preamble_bits, None),
            ("sync_word_bits", le_settings.sync_word_bits, 32),
            ("pdu_header_bits", le_settings.pdu_header_bits, 8),
        )
        for name, value, required_length in bit_fields:
            bits = str(value).strip().replace(" ", "")
            if not bits or any(bit not in "01" for bit in bits):
                issues.append(
                    ValidationIssue(
                        f"bluetooth_le.{name}", "Value must be a binary bit string."
                    )
                )
            elif required_length is not None and len(bits) != required_length:
                issues.append(
                    ValidationIssue(
                        f"bluetooth_le.{name}",
                        f"Value must contain exactly {required_length} bits.",
                    )
                )
        if not 0 <= int(le_settings.crc_init) <= 0xFFFFFF:
            issues.append(
                ValidationIssue(
                    "bluetooth_le.crc_init", "CRCInit must be a 24-bit value."
                )
            )
        if not 0 <= int(le_settings.whitening_channel_index) <= 39:
            issues.append(
                ValidationIssue(
                    "bluetooth_le.whitening_channel_index",
                    "Whitening channel index must be between 0 and 39.",
                )
            )
        if le_settings.payload_source in {
            BluetoothLEPayloadSourceKind.FIXED,
            BluetoothLEPayloadSourceKind.PATTERN,
        }:
            pattern = le_settings.payload_pattern.strip().replace(" ", "")
            if not pattern or any(bit not in "01" for bit in pattern):
                issues.append(
                    ValidationIssue(
                        "bluetooth_le.payload_pattern",
                        "Fixed and pattern payloads require a binary pattern.",
                    )
                )
    hdt_settings = project.bluetooth_hdt
    if hdt_settings is not None:
        if not 0 <= int(hdt_settings.payload_length_bytes) <= 4095:
            issues.append(
                ValidationIssue(
                    "bluetooth_hdt.payload_length_bytes",
                    "HDT payload length must be between 0 and 4095 bytes.",
                )
            )
        if not 0.0 < float(hdt_settings.rrc_rolloff) <= 1.0:
            issues.append(
                ValidationIssue(
                    "bluetooth_hdt.rrc_rolloff",
                    "HDT SRRC roll-off must be greater than 0 and at most 1.",
                )
            )
        if hdt_settings.pre_idle_symbols < 0 or hdt_settings.post_idle_symbols < 0:
            issues.append(
                ValidationIssue(
                    "bluetooth_hdt.idle_symbols",
                    "HDT idle symbol counts must not be negative.",
                )
            )
        if hdt_settings.payload_source in {
            PayloadSourceKind.FIXED,
            PayloadSourceKind.PATTERN,
        }:
            pattern = hdt_settings.payload_pattern.strip().replace(" ", "")
            if not pattern or any(bit not in "01" for bit in pattern):
                issues.append(
                    ValidationIssue(
                        "bluetooth_hdt.payload_pattern",
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
