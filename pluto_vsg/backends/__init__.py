"""Output backends for files and RF signal generators."""

from pluto_vsg.backends.base import BackendCapabilities, OutputBackend
from pluto_vsg.backends.pluto import (
    PlutoOutputBackend,
    PlutoPlaybackMode,
    PlutoTransmitSettings,
    estimate_pluto_output_power_dbm,
    pluto_hardware_gain_for_output_power_dbm,
    pluto_output_power_range_dbm,
)

__all__ = [
    "BackendCapabilities",
    "OutputBackend",
    "PlutoOutputBackend",
    "PlutoPlaybackMode",
    "PlutoTransmitSettings",
    "estimate_pluto_output_power_dbm",
    "pluto_hardware_gain_for_output_power_dbm",
    "pluto_output_power_range_dbm",
]
