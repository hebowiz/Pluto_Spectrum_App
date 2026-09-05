"""Classic DECT packet templates for the common VSG project model."""

from __future__ import annotations

from pluto_protocol.dect.carriers import carrier_by_identity
from pluto_protocol.dect.classic import PP_S_FIELD, RFP_S_FIELD
from pluto_vsg.model import (
    DataSourceKind,
    DectDirection,
    DectPacketType,
    DectSettings,
    FieldDefinition,
    FilterKind,
    ModulationDefinition,
    ModulationKind,
    PayloadSourceKind,
    StandardProfile,
    WaveformProject,
)


DECT_SYMBOL_RATE_HZ = 1_152_000.0
DECT_PACKET_SYMBOL_COUNTS = {
    DectPacketType.P00: 96,
    DectPacketType.P32: 420,
    DectPacketType.P32Z: 424,
    DectPacketType.P80: 900,
    DectPacketType.P80Z: 904,
}
DECT_B_FIELD_BITS = {
    DectPacketType.P32: 320,
    DectPacketType.P32Z: 320,
    DectPacketType.P80: 800,
    DectPacketType.P80Z: 800,
}


def dect_s_field_text(settings: DectSettings) -> str:
    bits = (
        RFP_S_FIELD
        if DectDirection(settings.direction) is DectDirection.RFP
        else PP_S_FIELD
    )
    return "".join(str(int(bit)) for bit in bits)


def dect_fields(settings: DectSettings) -> tuple[FieldDefinition, ...]:
    """Build the editable transmitted-field hierarchy in exact air order."""

    packet_type = DectPacketType(settings.packet_type)
    modulation = ModulationDefinition(
        kind=ModulationKind.GFSK,
        symbol_rate_hz=DECT_SYMBOL_RATE_HZ,
        filter_kind=FilterKind.GAUSSIAN,
        filter_parameter=float(settings.gaussian_bt),
    )

    def field(
        name: str,
        bits: int,
        data: str,
        source: DataSourceKind = DataSourceKind.FIXED,
        children: tuple[FieldDefinition, ...] = (),
    ) -> FieldDefinition:
        return FieldDefinition(
            name=name,
            symbol_count=bits,
            logical_bit_count=bits,
            data_source=source,
            data=data,
            modulation=modulation,
            children=children,
        )

    fields: list[FieldDefinition] = []
    s_field = dect_s_field_text(settings)
    preamble_bits = s_field[:16]
    sync_word_bits = s_field[16:]
    if settings.prolonged_preamble:
        fields.append(field("Prolonged Preamble", 16, preamble_bits))
    fields.extend(
        (
            field(
                "S-field",
                32,
                f"{settings.direction.value} synchronization",
                DataSourceKind.COMPUTED,
                (
                    field("Preamble", 16, preamble_bits),
                    field("Packet Synchronization Word", 16, sync_word_bits),
                ),
            ),
            field(
                "A-field",
                64,
                "H (8) + Tail (40) + R-CRC (16)",
                DataSourceKind.COMPUTED,
                (
                    field("Header", 8, settings.a_header_bits),
                    field("Tail", 40, settings.a_tail_bits),
                    field(
                        "R-CRC",
                        16,
                        "Auto" if settings.r_crc_auto else settings.r_crc_bits,
                        DataSourceKind.COMPUTED if settings.r_crc_auto else DataSourceKind.FIXED,
                    ),
                ),
            ),
        )
    )
    b_count = DECT_B_FIELD_BITS.get(packet_type)
    if b_count is not None:
        source = PayloadSourceKind(settings.b_field_source)
        fields.append(
            field(
                "B-field",
                b_count,
                "PRBS-9" if source is PayloadSourceKind.PRBS9 else settings.b_field_pattern,
                (
                    DataSourceKind.PRBS
                    if source is PayloadSourceKind.PRBS9
                    else DataSourceKind.FIXED
                    if source is PayloadSourceKind.FIXED
                    else DataSourceKind.PATTERN
                ),
            )
        )
        fields.append(
            field(
                "X-field",
                4,
                "Auto X-CRC" if settings.x_crc_auto else settings.x_field_bits,
                DataSourceKind.COMPUTED if settings.x_crc_auto else DataSourceKind.FIXED,
            )
        )
        if packet_type in {DectPacketType.P32Z, DectPacketType.P80Z}:
            fields.append(
                field(
                    "Z-field",
                    4,
                    "Repeat X" if settings.z_repeat_auto else settings.z_field_bits,
                    DataSourceKind.COMPUTED if settings.z_repeat_auto else DataSourceKind.FIXED,
                )
            )
    return tuple(fields)


def dect_project(settings: DectSettings | None = None) -> WaveformProject:
    selected = settings or DectSettings()
    carrier = carrier_by_identity(
        selected.carrier_plan_id, selected.carrier_channel
    )
    packet_type = DectPacketType(selected.packet_type)
    period_symbols = 960.0 if packet_type in {DectPacketType.P80, DectPacketType.P80Z} else 480.0
    return WaveformProject(
        name=f"DECT {packet_type.value} Packet",
        standard=StandardProfile.DECT,
        sample_rate_hz=DECT_SYMBOL_RATE_HZ * 8,
        samples_per_symbol=8,
        period_symbols=period_symbols,
        center_frequency_hz=carrier.center_frequency_hz,
        fields=dect_fields(selected),
        dect=selected,
    )
