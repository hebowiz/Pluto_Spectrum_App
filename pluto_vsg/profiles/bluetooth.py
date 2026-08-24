"""Initial Bluetooth project templates.

The template deliberately creates common model objects. Protocol-specific waveform
generation will be added to the engine without introducing a second UI model.
"""

from __future__ import annotations

from pluto_vsg.model import (
    DataSourceKind,
    FieldDefinition,
    StandardProfile,
    WaveformProject,
)


def bluetooth_br_edr_project() -> WaveformProject:
    return WaveformProject(
        name="Bluetooth BR / EDR Packet",
        standard=StandardProfile.BLUETOOTH_BR_EDR,
        sample_rate_hz=8_000_000.0,
        samples_per_symbol=8,
        fields=(
            FieldDefinition(
                name="Access Code",
                symbol_count=72,
                data_source=DataSourceKind.COMPUTED,
                data="BD_ADDR",
            ),
            FieldDefinition(
                name="Header",
                symbol_count=54,
                data_source=DataSourceKind.COMPUTED,
                data="Header + HEC",
            ),
            FieldDefinition(
                name="Payload",
                symbol_count=64,
                data_source=DataSourceKind.PRBS,
                data="PRBS9",
            ),
        ),
    )
