"""Initial Bluetooth project templates.

The template deliberately creates common model objects. Protocol-specific waveform
generation will be added to the engine without introducing a second UI model.
"""

from __future__ import annotations

from pluto_vsg.model import (
    BluetoothBRSettings,
    BluetoothPacketKind,
    DataSourceKind,
    FieldDefinition,
    PayloadSourceKind,
    StandardProfile,
    WaveformProject,
    ModulationDefinition,
    ModulationKind,
    FilterKind,
    bluetooth_packet_is_edr,
    bluetooth_packet_properties,
)


def _computed_field(
    name: str,
    *,
    logical_bits: int,
    transmitted_symbols: int | None = None,
    data: str = "Computed",
    modulation: ModulationDefinition | None = None,
    relative_power_db: float = 0.0,
) -> FieldDefinition:
    return FieldDefinition(
        name=name,
        symbol_count=(
            int(logical_bits)
            if transmitted_symbols is None
            else int(transmitted_symbols)
        ),
        logical_bit_count=int(logical_bits),
        data_source=DataSourceKind.COMPUTED,
        data=data,
        modulation=modulation or ModulationDefinition(),
        relative_power_db=float(relative_power_db),
    )


def bluetooth_br_fields(settings: BluetoothBRSettings) -> tuple[FieldDefinition, ...]:
    """Return the logical/transmitted hierarchy for a BR/EDR DH1 packet."""

    packet_kind = BluetoothPacketKind(settings.packet_kind)
    payload_body_bits = int(settings.payload_length_bytes) * 8
    is_edr = bluetooth_packet_is_edr(packet_kind)
    _, _, bits_per_symbol, _ = bluetooth_packet_properties(packet_kind)
    payload_header_bits = 8 if packet_kind == BluetoothPacketKind.DH1 else 16
    payload_modulation = (
        ModulationDefinition()
        if not is_edr
        else ModulationDefinition(
            kind=(
                ModulationKind.PI_4_DQPSK
                if bits_per_symbol == 2
                else ModulationKind.DPSK8
            ),
            filter_kind=FilterKind.ROOT_RAISED_COSINE,
            filter_parameter=float(settings.edr_rolloff),
        )
    )
    payload_bit_count = payload_header_bits + payload_body_bits + 16
    payload_symbols = (payload_bit_count + bits_per_symbol - 1) // bits_per_symbol
    header_symbols = (payload_header_bits + bits_per_symbol - 1) // bits_per_symbol
    body_stop_symbols = (
        payload_header_bits + payload_body_bits + bits_per_symbol - 1
    ) // bits_per_symbol
    body_symbols = body_stop_symbols - header_symbols
    crc_symbols = payload_symbols - body_stop_symbols
    source = PayloadSourceKind(settings.payload_source)
    payload_source = {
        PayloadSourceKind.FIXED: DataSourceKind.FIXED,
        PayloadSourceKind.PATTERN: DataSourceKind.PATTERN,
        PayloadSourceKind.PRBS9: DataSourceKind.PRBS,
    }[source]
    payload_data = (
        "PRBS9"
        if source == PayloadSourceKind.PRBS9
        else settings.payload_pattern
    )
    payload_relative_power_db = (
        float(settings.edr_relative_power_db) if is_edr else 0.0
    )
    payload_children = [
        _computed_field(
            "Payload Header",
            logical_bits=payload_header_bits,
            transmitted_symbols=header_symbols,
            data="LLID + FLOW + LENGTH",
            modulation=payload_modulation,
            relative_power_db=payload_relative_power_db,
        )
    ]
    if payload_body_bits:
        payload_children.append(
            FieldDefinition(
                name="Payload Body",
                symbol_count=body_symbols,
                logical_bit_count=payload_body_bits,
                data_source=payload_source,
                data=payload_data,
                modulation=payload_modulation,
                relative_power_db=payload_relative_power_db,
            )
        )
    payload_children.append(
        _computed_field(
            "Payload CRC",
            logical_bits=16,
            transmitted_symbols=crc_symbols,
            data="CRC-16",
            modulation=payload_modulation,
            relative_power_db=payload_relative_power_db,
        )
    )
    common = (
        FieldDefinition(
            name="Access Code",
            symbol_count=72,
            logical_bit_count=72,
            data_source=DataSourceKind.COMPUTED,
            data="BD_ADDR",
            children=(
                _computed_field("Preamble", logical_bits=4),
                _computed_field("Sync Word", logical_bits=64, data="LAP + BCH"),
                _computed_field("Trailer", logical_bits=4),
            ),
        ),
        FieldDefinition(
            name="Header",
            symbol_count=54,
            logical_bit_count=18,
            data_source=DataSourceKind.COMPUTED,
            data="Header + HEC + 1/3 FEC",
            children=(
                _computed_field("LT_ADDR", logical_bits=3, transmitted_symbols=9),
                _computed_field("TYPE", logical_bits=4, transmitted_symbols=12),
                _computed_field("FLOW", logical_bits=1, transmitted_symbols=3),
                _computed_field("ARQN", logical_bits=1, transmitted_symbols=3),
                _computed_field("SEQN", logical_bits=1, transmitted_symbols=3),
                _computed_field(
                    "HEC",
                    logical_bits=8,
                    transmitted_symbols=24,
                    data="HEC + 1/3 FEC",
                ),
            ),
        ),
        FieldDefinition(
            name="Payload" if not is_edr else "EDR Payload",
            symbol_count=payload_symbols,
            logical_bit_count=payload_bit_count,
            data_source=payload_source,
            data=payload_data,
            modulation=payload_modulation,
            relative_power_db=payload_relative_power_db,
            children=tuple(payload_children),
        ),
    )
    if not is_edr:
        return common
    edr_modulation = common[-1].modulation
    sync_bits = 20 if bits_per_symbol == 2 else 30
    return (
        *common[:2],
        FieldDefinition(
            name="Guard",
            symbol_count=int(settings.edr_guard_symbols),
            logical_bit_count=None,
            data_source=DataSourceKind.COMPUTED,
            data="Hold GFSK phase",
            relative_power_db=float(settings.edr_guard_relative_power_db),
        ),
        FieldDefinition(
            name="EDR Data",
            symbol_count=1 + sync_bits // bits_per_symbol + payload_symbols + 2,
            logical_bit_count=None,
            data_source=DataSourceKind.COMPUTED,
            data=packet_kind.value,
            modulation=edr_modulation,
            relative_power_db=float(settings.edr_relative_power_db),
            children=(
                FieldDefinition(
                    name="EDR Sync",
                    symbol_count=1 + sync_bits // bits_per_symbol,
                    logical_bit_count=sync_bits,
                    data_source=DataSourceKind.COMPUTED,
                    data="Reference + Sync Word",
                    modulation=edr_modulation,
                    relative_power_db=float(settings.edr_relative_power_db),
                ),
                common[-1],
                FieldDefinition(
                    name="EDR Trailer",
                    symbol_count=2,
                    logical_bit_count=2 * bits_per_symbol,
                    data_source=DataSourceKind.COMPUTED,
                    data="Trailer",
                    modulation=edr_modulation,
                    relative_power_db=float(settings.edr_relative_power_db),
                ),
            ),
        ),
    )


def bluetooth_br_edr_project() -> WaveformProject:
    settings = BluetoothBRSettings()
    return WaveformProject(
        name="Bluetooth BR / EDR Packet",
        standard=StandardProfile.BLUETOOTH_BR_EDR,
        sample_rate_hz=8_000_000.0,
        samples_per_symbol=8,
        center_frequency_hz=2_440_000_000.0,
        bluetooth_br=settings,
        fields=bluetooth_br_fields(settings),
    )
