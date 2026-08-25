"""Bluetooth LE uncoded Direct Test Mode packet templates."""

from __future__ import annotations

from dataclasses import replace

from pluto_vsg.model import (
    BluetoothLEPayloadType,
    BluetoothLEPayloadSourceKind,
    BluetoothLEPhy,
    BluetoothLESettings,
    DataSourceKind,
    FieldDefinition,
    FilterKind,
    ModulationDefinition,
    ModulationKind,
    StandardProfile,
    WaveformProject,
    bluetooth_le_payload_code,
)


def _payload_preset_source(
    payload_type: BluetoothLEPayloadType,
) -> tuple[BluetoothLEPayloadSourceKind, str]:
    if payload_type == BluetoothLEPayloadType.PRBS9:
        return BluetoothLEPayloadSourceKind.PRBS9, ""
    if payload_type == BluetoothLEPayloadType.PRBS15:
        return BluetoothLEPayloadSourceKind.PRBS15, ""
    return BluetoothLEPayloadSourceKind.PATTERN, payload_type.value


def apply_bluetooth_le_rf_test_preset(
    settings: BluetoothLESettings,
    *,
    phy: BluetoothLEPhy | None = None,
    payload_type: BluetoothLEPayloadType = BluetoothLEPayloadType.AA,
    payload_length_bytes: int = 37,
) -> BluetoothLESettings:
    """Populate editable LE settings with the Core RF Test Packet fields."""

    selected_phy = BluetoothLEPhy(settings.phy if phy is None else phy)
    selected_type = BluetoothLEPayloadType(payload_type)
    payload_source, payload_pattern = _payload_preset_source(selected_type)
    type_code = bluetooth_le_payload_code(selected_type)
    return replace(
        settings,
        phy=selected_phy,
        preamble_bits=(
            "10101010"
            if selected_phy == BluetoothLEPhy.LE_1M
            else "1010101010101010"
        ),
        sync_word_bits="10010100100000100110111010001110",
        pdu_header_bits="".join(str((type_code >> index) & 1) for index in range(8)),
        payload_type=selected_type,
        payload_source=payload_source,
        payload_pattern=payload_pattern,
        payload_length_bytes=int(payload_length_bytes),
        crc_enabled=True,
        crc_init=0x555555,
        whitening_enabled=False,
        rf_test_interval_enabled=True,
        frequency_deviation_hz=(
            250_000.0 if selected_phy == BluetoothLEPhy.LE_1M else 500_000.0
        ),
        gaussian_bt=0.5,
    )


def bluetooth_le_fields(
    settings: BluetoothLESettings,
) -> tuple[FieldDefinition, ...]:
    phy = BluetoothLEPhy(settings.phy)
    symbol_rate_hz = 1_000_000.0 if phy == BluetoothLEPhy.LE_1M else 2_000_000.0
    modulation = ModulationDefinition(
        kind=ModulationKind.GFSK,
        symbol_rate_hz=symbol_rate_hz,
        filter_kind=FilterKind.GAUSSIAN,
        filter_parameter=float(settings.gaussian_bt),
    )

    def field(
        name: str,
        bits: int,
        data: str,
        source: DataSourceKind = DataSourceKind.COMPUTED,
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

    payload_bits = int(settings.payload_length_bytes) * 8
    payload_source_kind = BluetoothLEPayloadSourceKind(settings.payload_source)
    payload_source = (
        DataSourceKind.PRBS
        if payload_source_kind
        in {BluetoothLEPayloadSourceKind.PRBS9, BluetoothLEPayloadSourceKind.PRBS15}
        else DataSourceKind.PATTERN
    )
    fields = [
        field("Preamble", len(settings.preamble_bits.replace(" ", "")), settings.preamble_bits),
        field("Access Address / Sync Word", 32, settings.sync_word_bits),
        field("PDU Header", 8, settings.pdu_header_bits),
        field("PDU Length", 8, f"{settings.payload_length_bytes} byte"),
    ]
    if payload_bits:
        fields.append(
            field(
                "PDU Payload",
                payload_bits,
                (
                    payload_source_kind.value
                    if payload_source_kind
                    in {
                        BluetoothLEPayloadSourceKind.PRBS9,
                        BluetoothLEPayloadSourceKind.PRBS15,
                    }
                    else settings.payload_pattern
                ),
                payload_source,
            )
        )
    if settings.crc_enabled:
        fields.append(field("CRC", 24, f"CRC-24 / Init 0x{settings.crc_init:06X}"))
    return tuple(fields)


def bluetooth_le_test_project(
    phy: BluetoothLEPhy = BluetoothLEPhy.LE_1M,
) -> WaveformProject:
    phy = BluetoothLEPhy(phy)
    settings = apply_bluetooth_le_rf_test_preset(
        BluetoothLESettings(phy=phy),
        phy=phy,
    )
    return WaveformProject(
        name=f"Bluetooth {phy.value} RF Test Packet",
        standard=StandardProfile.BLUETOOTH_LE,
        sample_rate_hz=(
            8_000_000.0 if phy == BluetoothLEPhy.LE_1M else 16_000_000.0
        ),
        samples_per_symbol=8,
        center_frequency_hz=2_440_000_000.0,
        fields=bluetooth_le_fields(settings),
        bluetooth_le=settings,
    )


def bluetooth_le_project(
    phy: BluetoothLEPhy = BluetoothLEPhy.LE_1M,
) -> WaveformProject:
    """Return an editable LE packet project without applying the RF test preset."""

    phy = BluetoothLEPhy(phy)
    settings = BluetoothLESettings(
        phy=phy,
        preamble_bits=("10101010" if phy == BluetoothLEPhy.LE_1M else "1010101010101010"),
        frequency_deviation_hz=(250_000.0 if phy == BluetoothLEPhy.LE_1M else 500_000.0),
    )
    return WaveformProject(
        name=f"Bluetooth {phy.value} Packet",
        standard=StandardProfile.BLUETOOTH_LE,
        sample_rate_hz=(8_000_000.0 if phy == BluetoothLEPhy.LE_1M else 16_000_000.0),
        samples_per_symbol=8,
        center_frequency_hz=2_440_000_000.0,
        fields=bluetooth_le_fields(settings),
        bluetooth_le=settings,
    )
