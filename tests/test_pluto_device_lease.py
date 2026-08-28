from __future__ import annotations

import pytest

from pluto_common import (
    PlutoDeviceBusyError,
    PlutoDeviceLease,
    pluto_identity,
    short_pluto_identity,
)


_CONTEXTS = {
    "usb:1.2.3": "ADALM-PLUTO SDR (serial=104473ABCDEF1234)",
}


def test_pluto_identity_normalizes_serial_selector() -> None:
    identity, serial = pluto_identity(
        "serial:104473ABCDEF1234", "usb:1.2.3", _CONTEXTS
    )

    assert identity == "serial:104473abcdef1234"
    assert serial == "104473ABCDEF1234"
    assert short_pluto_identity(f"serial:{serial}") == "…1234"


def test_pluto_identity_recovers_serial_from_resolved_uri() -> None:
    identity, serial = pluto_identity(None, "usb:1.2.3", _CONTEXTS)

    assert identity == "serial:104473abcdef1234"
    assert serial == "104473ABCDEF1234"


def test_device_lease_rejects_a_second_owner_and_reports_it(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    first = PlutoDeviceLease.acquire(
        "serial:104473ABCDEF1234",
        "usb:1.2.3",
        _CONTEXTS,
        application="Pluto RTSA",
        role="RX",
    )
    try:
        with pytest.raises(PlutoDeviceBusyError) as raised:
            PlutoDeviceLease.acquire(
                "serial:104473ABCDEF1234",
                "usb:1.2.3",
                _CONTEXTS,
                application="Pluto VSG",
                role="TX",
            )
        assert raised.value.owner is not None
        assert raised.value.owner.application == "Pluto RTSA"
        assert raised.value.owner.role == "RX"
        assert "already in use" in str(raised.value)
    finally:
        first.release()


def test_device_lease_can_be_reacquired_after_release(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    first = PlutoDeviceLease.acquire(
        None,
        "usb:1.2.3",
        _CONTEXTS,
        application="Pluto VSA",
        role="RX",
    )
    first.release()

    second = PlutoDeviceLease.acquire(
        None,
        "usb:1.2.3",
        _CONTEXTS,
        application="Pluto VSG",
        role="TX",
    )
    second.release()
