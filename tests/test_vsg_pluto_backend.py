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
        self.__class__.instances.append(self)

    def tx(self, values) -> None:
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
    }
    values.update(changes)
    return PlutoTransmitSettings(**values)


def test_pluto_backend_configures_finite_tx_and_destroys_buffer(monkeypatch) -> None:
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
    assert device.tx_enabled_channels == [0]
    assert device.sample_rate == 8_000_000
    assert device.tx_lo == 2_441_000_000
    assert device.tx_rf_bandwidth == 8_000_000
    assert device.tx_hardwaregain_chan0 == -30.0
    assert device.tx_cyclic_buffer is False
    np.testing.assert_allclose(
        device.transmitted,
        iq * (2**14 - 1),
        rtol=0.0,
        atol=1e-3,
    )
    assert device.destroy_count == 1


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


@pytest.mark.parametrize(
    "change",
    (
        {"sample_rate_hz": 100_000.0},
        {"rf_bandwidth_hz": 100_000.0},
        {"hardware_gain_db": 1.0},
    ),
)
def test_pluto_backend_rejects_unsupported_output_settings(change) -> None:
    with pytest.raises(ValueError):
        PlutoOutputBackend(_settings(**change))


def test_vsg_packet_repetition_is_limited_to_one_thousand() -> None:
    project = replace(bluetooth_br_edr_project(), repeat_count=1001)

    issues = validate_project(project)

    assert any(issue.path == "repeat_count" for issue in issues)
