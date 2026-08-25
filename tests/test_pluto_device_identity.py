from __future__ import annotations

import pytest

from pluto_common import discover_pluto_devices, resolve_pluto_uri


SERIAL_A = "1044730c370e00100400120023338fb325"
SERIAL_B = "1044730c370e001004001200abcdef0123"


def _contexts() -> dict[str, str]:
    return {
        "ip:pluto.local": f"Analog Devices PlutoSDR, serial={SERIAL_A}",
        "usb:1.26.5": f"Analog Devices Inc. PlutoSDR, serial={SERIAL_A}",
        "usb:2.4.5": f"Analog Devices Inc. PlutoSDR, serial={SERIAL_B}",
        "ip:unrelated": "Unrelated network IIO device",
    }


def test_discovery_identifies_two_physical_plutos_and_prefers_usb() -> None:
    devices = discover_pluto_devices(_contexts())

    assert len(devices) == 2
    assert [(device.serial, device.uri) for device in devices] == [
        (SERIAL_A, "usb:1.26.5"),
        (SERIAL_B, "usb:2.4.5"),
    ]
    assert devices[0].selector == f"serial:{SERIAL_A}"
    assert SERIAL_A in devices[0].label


def test_serial_selector_resolves_to_current_usb_uri() -> None:
    assert resolve_pluto_uri(f"serial:{SERIAL_B}", _contexts()) == "usb:2.4.5"


def test_serial_selector_does_not_fall_back_to_the_wrong_pluto() -> None:
    with pytest.raises(RuntimeError, match="is not connected"):
        resolve_pluto_uri("serial:missing", _contexts())


def test_auto_and_explicit_uri_remain_backward_compatible() -> None:
    assert resolve_pluto_uri(None, _contexts()) == "usb:1.26.5"
    assert resolve_pluto_uri("ip:192.168.2.1", _contexts()) == "ip:192.168.2.1"
