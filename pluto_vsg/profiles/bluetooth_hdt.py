"""Editable Bluetooth-derived HDT RF-test waveform profile."""

from __future__ import annotations

import math

from pluto_protocol.bluetooth.hdt import (
    HDTRate,
    hdt_coded_payload_bit_count as hdt_coded_payload_bit_count_for_rate,
    hdt_definition,
)
from pluto_vsg.model import (
    BluetoothHDTSettings, DataSourceKind, FieldDefinition, FilterKind,
    ModulationDefinition, ModulationKind, StandardProfile, WaveformProject,
)


_MODULATION_KIND = {
    "pi/4-QPSK": ModulationKind.PI_4_QPSK,
    "8PSK": ModulationKind.PSK8,
    "16QAM": ModulationKind.QAM16,
}


def hdt_coded_payload_bit_count(settings: BluetoothHDTSettings) -> int:
    return hdt_coded_payload_bit_count_for_rate(
        settings.rate, settings.payload_length_bytes
    )


def bluetooth_hdt_fields(settings: BluetoothHDTSettings) -> tuple[FieldDefinition, ...]:
    definition = hdt_definition(settings.rate)
    symbol_rate = 2_000_000.0
    qpsk = ModulationDefinition(ModulationKind.PI_4_QPSK, symbol_rate, FilterKind.ROOT_RAISED_COSINE, settings.rrc_rolloff)
    payload_mod = ModulationDefinition(_MODULATION_KIND[definition.modulation], symbol_rate, FilterKind.ROOT_RAISED_COSINE, settings.rrc_rolloff)
    coded_bits = hdt_coded_payload_bit_count(settings)
    fields = [
        FieldDefinition("Training / Preamble", 74, 148, DataSourceKind.COMPUTED, "STS x9 + GI + LTS x2", qpsk),
        FieldDefinition("Control Header", 62, 31, DataSourceKind.COMPUTED, f"RI={definition.rate_indicator:03b}", qpsk),
    ]
    if coded_bits:
        fields.append(FieldDefinition(
            "Coded Payload", math.ceil(coded_bits / definition.bits_per_symbol),
            int(settings.payload_length_bytes) * 8,
            (DataSourceKind.PRBS if settings.payload_source.value == "PRBS-9" else DataSourceKind(settings.payload_source.value)),
            settings.payload_source.value, payload_mod,
        ))
    return tuple(fields)


def bluetooth_hdt_project(rate: HDTRate = HDTRate.HDT6) -> WaveformProject:
    settings = BluetoothHDTSettings(rate=HDTRate(rate))
    return WaveformProject(
        name=f"Bluetooth {settings.rate.value} RF Test Packet",
        standard=StandardProfile.BLUETOOTH_HDT,
        sample_rate_hz=16_000_000.0,
        samples_per_symbol=8,
        center_frequency_hz=2_440_000_000.0,
        fields=bluetooth_hdt_fields(settings),
        bluetooth_hdt=settings,
    )
