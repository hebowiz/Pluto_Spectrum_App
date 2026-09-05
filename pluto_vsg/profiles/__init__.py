"""Standard profiles which expand into the common project model."""

from pluto_vsg.profiles.bluetooth import bluetooth_br_edr_project, bluetooth_br_fields
from pluto_vsg.profiles.bluetooth_le import (
    apply_bluetooth_le_rf_test_preset,
    bluetooth_le_fields,
    bluetooth_le_project,
    bluetooth_le_test_project,
)
from pluto_vsg.profiles.bluetooth_hdt import bluetooth_hdt_fields, bluetooth_hdt_project
from pluto_vsg.profiles.wifi import wifi_beacon_project, wifi_fields, wifi_project
from pluto_vsg.profiles.dect import dect_fields, dect_project

__all__ = [
    "bluetooth_br_edr_project",
    "bluetooth_br_fields",
    "bluetooth_le_fields",
    "bluetooth_le_project",
    "bluetooth_le_test_project",
    "apply_bluetooth_le_rf_test_preset",
    "bluetooth_hdt_fields",
    "bluetooth_hdt_project",
    "wifi_beacon_project",
    "wifi_fields",
    "wifi_project",
    "dect_fields",
    "dect_project",
]
