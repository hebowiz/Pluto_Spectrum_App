"""Initial Bluetooth project templates.

The template deliberately creates common model objects. Protocol-specific waveform
generation will be added to the engine without introducing a second UI model.
"""

from __future__ import annotations

from pluto_vsg.model import (
    BluetoothBRSettings,
    DataSourceKind,
    FieldDefinition,
    PayloadSourceKind,
    StandardProfile,
    WaveformProject,
)


def _computed_field(
    name: str,
    *,
    logical_bits: int,
    transmitted_symbols: int | None = None,
    data: str = "Computed",
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
    )


def bluetooth_br_fields(settings: BluetoothBRSettings) -> tuple[FieldDefinition, ...]:
    """Return the logical/transmitted hierarchy for the implemented DH1 slice."""

    payload_body_bits = int(settings.payload_length_bytes) * 8
    payload_symbols = 8 + payload_body_bits + 16
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
    payload_children = [
        _computed_field(
            "Payload Header", logical_bits=8, data="LLID + FLOW + LENGTH"
        )
    ]
    if payload_body_bits:
        payload_children.append(
            FieldDefinition(
                name="Payload Body",
                symbol_count=payload_body_bits,
                logical_bit_count=payload_body_bits,
                data_source=payload_source,
                data=payload_data,
            )
        )
    payload_children.append(
        _computed_field("Payload CRC", logical_bits=16, data="CRC-16")
    )
    return (
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
            name="Payload",
            symbol_count=payload_symbols,
            logical_bit_count=payload_symbols,
            data_source=payload_source,
            data=payload_data,
            children=tuple(payload_children),
        ),
    )


def bluetooth_br_edr_project() -> WaveformProject:
    settings = BluetoothBRSettings()
    return WaveformProject(
        name="Bluetooth BR / EDR Packet",
        standard=StandardProfile.BLUETOOTH_BR_EDR,
        sample_rate_hz=8_000_000.0,
        samples_per_symbol=8,
        bluetooth_br=settings,
        fields=bluetooth_br_fields(settings),
    )
