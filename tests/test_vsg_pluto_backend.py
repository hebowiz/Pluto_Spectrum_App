from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from pluto_vsg.backends import PlutoOutputBackend, PlutoTransmitSettings
from pluto_vsg.engine import GenerationResult
from pluto_vsg.model import validate_project
from pluto_vsg.profiles import bluetooth_br_edr_project


class _FakePluto:
    instances: list["_FakePluto"] = []

    def __init__(self, uri=None) -> None:
        self.uri = uri
        self.transmitted: np.ndarray | None = None
        self.destroy_count = 0
        self.zero_source_count = 0
        self.tx_enabled_history: list[list[int]] = []
        self.gain_history: list[float] = []
        self.__class__.instances.append(self)

    @property
    def tx_enabled_channels(self) -> list[int]:
        return self.tx_enabled_history[-1]

    @tx_enabled_channels.setter
    def tx_enabled_channels(self, value) -> None:
        self.tx_enabled_history.append(list(value))

    @property
    def tx_hardwaregain_chan0(self) -> float:
        return self.gain_history[-1]

    @tx_hardwaregain_chan0.setter
    def tx_hardwaregain_chan0(self, value) -> None:
        self.gain_history.append(float(value))

    def tx(self, values=None) -> None:
        if values is None:
            self.zero_source_count += 1
            return
        self.transmitted = np.asarray(values).copy()

    def tx_destroy_buffer(self) -> None:
        self.destroy_count += 1


def _settings(**changes) -> PlutoTransmitSettings:
    values = {
        "center_frequency_hz": 2_441_000_000.0,
        "sample_rate_hz": 8_000_000.0,
        "rf_bandwidth_hz": 8_000_000.0,
        "hardware_gain_db": -30.0,
        "connection_uri": "usb:test",
        "lead_in_guard_s": 0.001,
        "stop_guard_s": 0.010,
    }
    values.update(changes)
    return PlutoTransmitSettings(**values)


def test_pluto_backend_uses_guarded_cyclic_superframe_and_mutes(monkeypatch) -> None:
    import pluto_vsg.backends.pluto as module

    _FakePluto.instances.clear()
    monkeypatch.setattr(module.adi, "Pluto", _FakePluto)
    iq = np.asarray([0.0, 1.0 + 0.0j, 0.0 + 1.0j, -0.5 - 0.25j], dtype=np.complex64)
    result = GenerationResult(iq=iq, sample_rate_hz=8_000_000.0)
    backend = PlutoOutputBackend(_settings())

    backend.transfer(result)
    backend.start()

    device = _FakePluto.instances[-1]
    assert device.uri == "usb:test"
    assert device.tx_enabled_history[0] == [0]
    assert device.sample_rate == 8_000_000
    assert device.tx_lo == 2_441_000_000
    assert device.tx_rf_bandwidth == 8_000_000
    assert device.gain_history[0] == -30.0
    assert device.tx_cyclic_buffer is True
    lead_in_samples = 8_000
    stop_guard_samples = 80_000
    assert device.transmitted.size == lead_in_samples + iq.size + stop_guard_samples
    np.testing.assert_array_equal(device.transmitted[:lead_in_samples], 0.0)
    np.testing.assert_allclose(
        device.transmitted[lead_in_samples : lead_in_samples + iq.size],
        iq * (2**14 - 1),
        rtol=0.0,
        atol=1e-3,
    )
    np.testing.assert_array_equal(
        device.transmitted[lead_in_samples + iq.size :], 0.0
    )
    assert device.destroy_count == 1
    assert device.tx_hardwaregain_chan0 == -89.75
    assert device.gain_history == [-30.0, -89.75]
    assert device.tx_enabled_channels == []
    assert device.zero_source_count == 1


def test_pluto_backend_honors_stop_requested_before_tx(monkeypatch) -> None:
    import pluto_vsg.backends.pluto as module

    _FakePluto.instances.clear()
    monkeypatch.setattr(module.adi, "Pluto", _FakePluto)
    result = GenerationResult(
        iq=np.ones(8, dtype=np.complex64), sample_rate_hz=8_000_000.0
    )
    backend = PlutoOutputBackend(_settings())
    backend.transfer(result)

    backend.stop()
    backend.start()

    device = _FakePluto.instances[-1]
    assert device.transmitted is None
    assert device.destroy_count == 1
    assert device.tx_hardwaregain_chan0 == -89.75
    assert device.zero_source_count == 1


def test_pluto_backend_resolves_saved_serial_selector(monkeypatch) -> None:
    import pluto_vsg.backends.pluto as module

    serial = "1044730c370e001004001200abcdef0123"
    monkeypatch.setattr(
        PlutoOutputBackend,
        "discover",
        staticmethod(
            lambda: {
                "usb:1.26.5": "Analog Devices PlutoSDR, serial=other",
                "usb:2.4.5": f"Analog Devices PlutoSDR, serial={serial}",
            }
        ),
    )
    _FakePluto.instances.clear()
    monkeypatch.setattr(module.adi, "Pluto", _FakePluto)
    backend = PlutoOutputBackend(_settings(connection_uri=f"serial:{serial}"))
    backend.transfer(
        GenerationResult(
            iq=np.ones(8, dtype=np.complex64), sample_rate_hz=8_000_000.0
        )
    )

    backend.start()

    assert _FakePluto.instances[-1].uri == "usb:2.4.5"


@pytest.mark.parametrize(
    "change",
    (
        {"sample_rate_hz": 100_000.0},
        {"rf_bandwidth_hz": 100_000.0},
        {"hardware_gain_db": 1.0},
        {"lead_in_guard_s": -0.001},
        {"stop_guard_s": 0.001},
    ),
)
def test_pluto_backend_rejects_unsupported_output_settings(change) -> None:
    with pytest.raises(ValueError):
        PlutoOutputBackend(_settings(**change))


def test_vsg_packet_repetition_is_limited_to_one_thousand() -> None:
    project = replace(bluetooth_br_edr_project(), repeat_count=1001)

    issues = validate_project(project)

    assert any(issue.path == "repeat_count" for issue in issues)
