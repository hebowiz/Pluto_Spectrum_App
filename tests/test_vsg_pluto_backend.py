from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

from pluto_vsg.backends import (
    PlutoOutputBackend,
    PlutoTransmitSettings,
    estimate_pluto_output_power_dbm,
    pluto_hardware_gain_for_output_power_dbm,
    pluto_output_power_range_dbm,
)
from pluto_vsg.engine import GenerationResult
from pluto_vsg.model import validate_project
from pluto_vsg.profiles import bluetooth_br_edr_project


class _FakeAttr:
    def __init__(self, value: str, on_write=None) -> None:
        self._value = value
        self._on_write = on_write

    @property
    def value(self) -> str:
        return self._value

    @value.setter
    def value(self, value: str) -> None:
        self._value = str(value)
        if self._on_write is not None:
            self._on_write(str(value))


class _FakeGpioChannel:
    def __init__(self) -> None:
        self.attrs = {
            "label": _FakeAttr("PHASER_ENABLE"),
            "raw": _FakeAttr("0"),
        }


class _FakeGpioDevice:
    name = "one-bit-adc-dac"
    attrs = {}

    def __init__(self) -> None:
        self.channels = [_FakeGpioChannel()]


class _FakeContext:
    attrs = {"fw_version": "v0.39"}

    def __init__(self) -> None:
        self.phy = type("FakePhy", (), {})()
        self.phy.name = "ad9361-phy"
        self.phy.channels = []
        self.phy.attrs = {
            "calib_mode": _FakeAttr(
                _FakePluto.hardware["calib_mode"],
                self._calibration_mode_written,
            ),
            "calib_mode_available": _FakeAttr(
                "auto manual manual_tx_quad tx_quad rf_dc_offs rssi_gain_step"
            ),
        }
        self.devices = [self.phy, _FakeGpioDevice()]

    def _calibration_mode_written(self, value: str) -> None:
        _FakePluto.events.append(f"calib:{value}")
        # tx_quad is a command. The driver policy is restored explicitly by
        # the backend immediately afterward.
        _FakePluto.hardware["calib_mode"] = value

    def find_device(self, name: str):
        return self.phy if name == "ad9361-phy" else None


class _FakePluto:
    instances: list["_FakePluto"] = []
    events: list[str] = []
    hardware = {
        "sample_rate": 4_000_000,
        "tx_lo": 2_440_000_000,
        "tx_rf_bandwidth": 4_000_000,
        "calib_mode": "auto",
    }

    def __init__(self, uri=None) -> None:
        self.uri = uri
        self._ctx = _FakeContext()
        self.transmitted: np.ndarray | None = None
        self.destroy_count = 0
        self.zero_source_count = 0
        self.tx_enabled_history: list[list[int]] = []
        self.gain_history: list[float] = []
        self.__class__.instances.append(self)

    @property
    def sample_rate(self) -> int:
        return int(self.hardware["sample_rate"])

    @sample_rate.setter
    def sample_rate(self, value) -> None:
        self.hardware["sample_rate"] = int(value)
        self.events.append(f"sample_rate:{int(value)}")

    @property
    def tx_lo(self) -> int:
        return int(self.hardware["tx_lo"])

    @tx_lo.setter
    def tx_lo(self, value) -> None:
        self.hardware["tx_lo"] = int(value)
        self.events.append(f"tx_lo:{int(value)}")

    @property
    def tx_rf_bandwidth(self) -> int:
        return int(self.hardware["tx_rf_bandwidth"])

    @tx_rf_bandwidth.setter
    def tx_rf_bandwidth(self, value) -> None:
        self.hardware["tx_rf_bandwidth"] = int(value)
        self.events.append(f"tx_rf_bandwidth:{int(value)}")

    @property
    def tx_enabled_channels(self) -> list[int]:
        return self.tx_enabled_history[-1]

    @tx_enabled_channels.setter
    def tx_enabled_channels(self, value) -> None:
        self.tx_enabled_history.append(list(value))
        self.events.append(f"tx_channels:{list(value)}")

    @property
    def tx_hardwaregain_chan0(self) -> float:
        return self.gain_history[-1]

    @tx_hardwaregain_chan0.setter
    def tx_hardwaregain_chan0(self, value) -> None:
        self.gain_history.append(float(value))
        self.events.append(f"gain:{float(value)}")

    def tx(self, values=None) -> None:
        if values is None:
            self.zero_source_count += 1
            self.events.append("dac_zero")
            return
        self.transmitted = np.asarray(values).copy()
        self.events.append("buffer_transferred")

    def tx_destroy_buffer(self) -> None:
        self.destroy_count += 1
        self.events.append("buffer_destroyed")


class _FakeTddChannel:
    def __init__(self) -> None:
        self.enable = False
        self.on_raw = 0
        self.off_raw = 0
        self.polarity = 0


class _FakeTddn:
    instances: list["_FakeTddn"] = []
    channel: list[_FakeTddChannel] = []

    def __init__(self, uri="") -> None:
        self.uri = uri
        self.channel.extend(_FakeTddChannel() for _ in range(3))
        self.enable = False
        self.sync_external = False
        self.sync_internal = False
        self.sync_reset = False
        self.startup_delay_ms = 0.0
        self.frame_length_ms = 0.0
        self.frame_length_raw = 7999
        self.burst_count = 0
        self._sync_soft = 0
        self.__class__.instances.append(self)

    @property
    def sync_soft(self) -> int:
        return self._sync_soft

    @sync_soft.setter
    def sync_soft(self, value) -> None:
        self._sync_soft = int(value)
        _FakePluto.events.append(f"sync_soft:{int(value)}")


def _settings(**changes) -> PlutoTransmitSettings:
    values = {
        "center_frequency_hz": 2_441_000_000.0,
        "sample_rate_hz": 8_000_000.0,
        "rf_bandwidth_hz": 8_000_000.0,
        "hardware_gain_db": -30.0,
        "digital_backoff_db": 0.0,
        "connection_uri": "usb:test",
        "lead_in_guard_s": 0.0,
        "dma_preroll_s": 0.002,
        "stop_guard_s": 0.010,
        "burst_count": 3,
    }
    values.update(changes)
    return PlutoTransmitSettings(**values)


def _install_fakes(monkeypatch) -> None:
    import pluto_vsg.backends.pluto as module

    _FakePluto.instances.clear()
    _FakePluto.events.clear()
    _FakePluto.hardware = {
        "sample_rate": 4_000_000,
        "tx_lo": 2_440_000_000,
        "tx_rf_bandwidth": 4_000_000,
        "calib_mode": "auto",
    }
    _FakeTddn.instances.clear()
    _FakeTddn.channel = []
    monkeypatch.setattr(module.adi, "Pluto", _FakePluto)
    monkeypatch.setattr(module.adi, "tddn", _FakeTddn)

    def set_tx_lo_powerdown(device, powerdown):
        device.tx_lo_powerdown = bool(powerdown)
        _FakePluto.events.append(f"tx_lo_powerdown:{bool(powerdown)}")

    monkeypatch.setattr(
        module.PlutoOutputBackend,
        "_set_tx_lo_powerdown",
        staticmethod(set_tx_lo_powerdown),
    )


def test_pluto_backend_transfers_complete_schedule_once_with_noncyclic_dma(monkeypatch) -> None:
    _install_fakes(monkeypatch)
    frame = np.asarray([0.0, 1.0 + 0.0j, 0.0 + 1.0j, -0.5 - 0.25j])
    result = GenerationResult(iq=np.tile(frame, 3), sample_rate_hz=8_000_000.0)
    backend = PlutoOutputBackend(_settings())

    backend.prepare()
    backend.transfer(result)
    backend.start()

    device = _FakePluto.instances[-1]
    assert device.uri == "usb:test"
    assert _FakeTddn.instances == []
    assert device.sample_rate == 8_000_000
    assert device.tx_lo == 2_441_000_000
    assert device.tx_rf_bandwidth == 8_000_000
    assert device.tx_cyclic_buffer is False
    prefix_count = round(0.002 * 8_000_000)
    suffix_count = round(0.002 * 8_000_000)
    assert device.transmitted.size == prefix_count + frame.size * 3 + suffix_count
    assert np.count_nonzero(device.transmitted[:prefix_count]) == 0
    np.testing.assert_allclose(
        device.transmitted[prefix_count : prefix_count + frame.size * 3],
        np.tile(frame, 3) * (2**15 - 1),
        rtol=0.0,
        atol=1e-3,
    )
    assert np.count_nonzero(device.transmitted[-suffix_count:]) == 0
    assert device.gain_history == [-89.75, -89.75, -30.0, -89.75]
    assert _FakePluto.events.index("tx_lo_powerdown:False") < _FakePluto.events.index(
        "gain:-30.0"
    )
    assert _FakePluto.events.index("gain:-30.0") < _FakePluto.events.index(
        "buffer_transferred"
    )
    assert device.destroy_count == 1
    assert device.tx_enabled_channels == []
    assert device.zero_source_count == 1
    assert device.tx_lo_powerdown is True


def test_pluto_backend_honors_stop_requested_during_buffer_transfer(monkeypatch) -> None:
    _install_fakes(monkeypatch)
    original_tx = _FakePluto.tx

    def stop_during_transfer(device, values=None):
        original_tx(device, values)
        if values is not None:
            backend.stop()

    monkeypatch.setattr(_FakePluto, "tx", stop_during_transfer)
    result = GenerationResult(
        iq=np.tile(np.ones(8, dtype=np.complex64), 3), sample_rate_hz=8_000_000.0
    )
    backend = PlutoOutputBackend(_settings())
    backend.prepare()
    backend.transfer(result)

    backend.start()

    device = _FakePluto.instances[-1]
    assert device.transmitted is not None
    assert _FakeTddn.instances == []
    assert device.gain_history == [-89.75, -89.75, -30.0, -89.75]
    assert device.destroy_count == 1


def test_pluto_backend_diagnostic_report_is_json_safe(monkeypatch) -> None:
    _install_fakes(monkeypatch)
    backend = PlutoOutputBackend(_settings(burst_count=1))
    backend.prepare()
    backend.transfer(
        GenerationResult(iq=np.ones(8, dtype=np.complex64), sample_rate_hz=8_000_000.0)
    )

    backend.start()

    report = backend.diagnostic_report()
    json.dumps(report)
    assert report["tx_dma_mode"] == "non-cyclic finite buffer"
    assert report["tdd_policy"].startswith("not accessed")
    assert isinstance(report["libiio_version"], str)
    assert isinstance(report["pyadi_iio_version"], str)
    names = [event["name"] for event in report["events"]]
    assert "noncyclic_push_started" in names
    assert "noncyclic_push_completed" in names
    assert "packet_schedule_submitted" in names
    observations = {
        item["stage"]: item for item in report["hardware_observations"]
    }
    assert observations["before_noncyclic_push"]["tx_gain_db"] == -30.0
    assert report["superframe_sample_count"] > report["sample_count"]
    assert report["settings"]["digital_backoff_db"] == 0.0
    assert report["dac_full_scale"] == 2**15 - 1
    assert report["dac_peak_code"] == pytest.approx(2**15 - 1)
    assert report["dma_buffer_duration_ms"] > report["waveform_duration_ms"]
    assert "after_requested_gain" not in observations


def test_pluto_backend_skips_equivalent_rf_parameter_rewrites() -> None:
    backend = PlutoOutputBackend(_settings(burst_count=1))

    class Device:
        sample_rate = 7_999_999

    device = Device()
    changed = backend._set_numeric_if_changed(
        device,
        "sample_rate",
        8_000_000,
        tolerance=8.0,
        event_prefix="sample_rate",
    )

    assert changed is False
    assert device.sample_rate == 7_999_999
    assert backend.event_log[-1][0] == "sample_rate_unchanged"


def test_pluto_backend_writes_materially_different_rf_parameter() -> None:
    backend = PlutoOutputBackend(_settings(burst_count=1))

    class Device:
        sample_rate = 4_000_000

    device = Device()
    changed = backend._set_numeric_if_changed(
        device,
        "sample_rate",
        8_000_000,
        tolerance=8.0,
        event_prefix="sample_rate",
    )

    assert changed is True
    assert device.sample_rate == 8_000_000
    assert backend.event_log[-1][0] == "sample_rate_configured"


def test_pluto_backend_commits_first_libiio_v1_noncyclic_block() -> None:
    class Stream:
        def __init__(self) -> None:
            self.advance_count = 0

        def __iter__(self):
            return self

        def __next__(self):
            self.advance_count += 1
            return object()

    class Device:
        _tx_stream = Stream()

    device = Device()

    assert PlutoOutputBackend._commit_noncyclic_stream(device) is True
    assert device._tx_stream.advance_count == 1


def test_pluto_backend_does_not_commit_libiio_v0_buffer() -> None:
    class Device:
        _tx_stream = None

    assert PlutoOutputBackend._commit_noncyclic_stream(Device()) is False


def test_pluto_backend_honors_stop_requested_before_start(monkeypatch) -> None:
    _install_fakes(monkeypatch)
    backend = PlutoOutputBackend(_settings(burst_count=1))
    backend.prepare()
    backend.transfer(
        GenerationResult(iq=np.ones(8, dtype=np.complex64), sample_rate_hz=8_000_000.0)
    )

    backend.stop()
    backend.start()

    device = _FakePluto.instances[-1]
    assert device.transmitted is None
    assert _FakeTddn.instances == []
    assert device.gain_history == [-89.75, -89.75]
    assert device.destroy_count == 1


def test_pluto_backend_accepts_old_firmware_without_tddn(monkeypatch) -> None:
    _install_fakes(monkeypatch)
    monkeypatch.setattr(_FakeContext, "attrs", {"fw_version": "v0.38"})
    backend = PlutoOutputBackend(_settings(burst_count=1))
    backend.prepare()
    backend.transfer(
        GenerationResult(iq=np.ones(8, dtype=np.complex64), sample_rate_hz=8_000_000.0)
    )

    backend.start()

    device = _FakePluto.instances[-1]
    assert device.transmitted is not None
    assert device.gain_history == [-89.75, -89.75, -30.0, -89.75]
    assert device.destroy_count == 1


def test_pluto_backend_preserves_nonidentical_packet_schedule() -> None:
    backend = PlutoOutputBackend(_settings(burst_count=2))
    result = GenerationResult(
        iq=np.asarray([1, 0, 1, 0.5], dtype=np.complex64),
        sample_rate_hz=8_000_000.0,
    )

    backend.transfer(result)

    prefix_count = round(0.002 * 8_000_000)
    np.testing.assert_allclose(
        backend._superframe[prefix_count : prefix_count + 4],
        result.iq * (2**15 - 1),
        rtol=0.0,
        atol=1e-3,
    )


def test_pluto_backend_applies_configured_digital_backoff() -> None:
    backend = PlutoOutputBackend(
        _settings(burst_count=1, digital_backoff_db=-6.0)
    )
    result = GenerationResult(
        iq=np.ones(4, dtype=np.complex64), sample_rate_hz=8_000_000.0
    )

    backend.transfer(result)

    prefix_count = round(0.002 * 8_000_000)
    expected = (2**15 - 1) * 10.0 ** (-6.0 / 20.0)
    np.testing.assert_allclose(
        backend._superframe[prefix_count : prefix_count + 4],
        expected,
        rtol=0.0,
        atol=1e-3,
    )


@pytest.mark.parametrize(
    ("hardware_gain_db", "digital_backoff_db", "measured_power_dbm"),
    (
        (0.0, 0.0, -0.2),
        (-5.0, 0.0, -4.8),
        (-10.0, 0.0, -9.4),
        (-20.0, 0.0, -19.0),
        (0.0, -3.0, -3.1),
        (-10.0, -3.0, -12.4),
        (0.0, -6.0, -6.1),
        (-10.0, -6.0, -15.4),
    ),
)
def test_provisional_pluto_level_calibration_matches_measured_fsk_power(
    hardware_gain_db: float,
    digital_backoff_db: float,
    measured_power_dbm: float,
) -> None:
    estimated_dbm = estimate_pluto_output_power_dbm(
        hardware_gain_db,
        digital_backoff_db,
    )

    assert estimated_dbm == pytest.approx(measured_power_dbm, abs=0.11)


@pytest.mark.parametrize("backoff_db", (0.0, -3.0, -6.0))
def test_provisional_pluto_level_conversion_round_trips(backoff_db: float) -> None:
    for gain_db in (-89.75, -30.0, -20.0, -10.0, -5.0, 0.0):
        output_dbm = estimate_pluto_output_power_dbm(gain_db, backoff_db)
        recovered_gain_db = pluto_hardware_gain_for_output_power_dbm(
            output_dbm,
            backoff_db,
        )
        assert recovered_gain_db == pytest.approx(gain_db)


def test_pluto_backend_converts_requested_dbm_to_hardware_gain(monkeypatch) -> None:
    _install_fakes(monkeypatch)
    backend = PlutoOutputBackend(
        _settings(
            burst_count=1,
            hardware_gain_db=-30.0,
            digital_backoff_db=-3.0,
            output_power_dbm=-12.4,
        )
    )
    backend.prepare()
    backend.transfer(
        GenerationResult(iq=np.ones(8, dtype=np.complex64), sample_rate_hz=8_000_000.0)
    )

    backend.start()

    device = _FakePluto.instances[-1]
    assert device.gain_history == [-89.75, -89.75, -10.0, -89.75]
    report = backend.diagnostic_report()
    assert report["settings"]["output_power_dbm"] == -12.4
    assert report["resolved_hardware_gain_db"] == pytest.approx(-10.0)


def test_provisional_pluto_output_range_respects_backoff() -> None:
    zero_backoff = pluto_output_power_range_dbm(0.0)
    six_db_backoff = pluto_output_power_range_dbm(-6.0)

    assert six_db_backoff[0] == pytest.approx(zero_backoff[0] - 6.0)
    assert six_db_backoff[1] == pytest.approx(zero_backoff[1] - 6.0)


def test_pluto_backend_resolves_saved_serial_selector(monkeypatch) -> None:
    _install_fakes(monkeypatch)
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
    backend = PlutoOutputBackend(
        _settings(connection_uri=f"serial:{serial}", burst_count=1)
    )
    backend.prepare()
    backend.transfer(
        GenerationResult(iq=np.ones(8, dtype=np.complex64), sample_rate_hz=8_000_000.0)
    )

    backend.start()

    assert _FakePluto.instances[-1].uri == "usb:2.4.5"
    assert _FakeTddn.instances == []


def test_pluto_backend_falls_back_to_same_serial_network_uri(monkeypatch) -> None:
    import pluto_vsg.backends.pluto as module

    _install_fakes(monkeypatch)
    serial = "10447318ac0f00050a001600356a18eee6"
    monkeypatch.setattr(
        PlutoOutputBackend,
        "discover",
        staticmethod(
            lambda: {
                "usb:1.30.5": f"Analog Devices PlutoSDR, serial={serial}",
                "ip:pluto.local": f"Analog Devices PlutoSDR, serial={serial}",
            }
        ),
    )

    def open_pluto(uri=None):
        if uri == "usb:1.30.5":
            raise RuntimeError("USB interface unavailable")
        return _FakePluto(uri=uri)

    monkeypatch.setattr(module.adi, "Pluto", open_pluto)
    backend = PlutoOutputBackend(
        _settings(connection_uri=f"serial:{serial}", burst_count=1)
    )
    backend.prepare()
    backend.transfer(
        GenerationResult(iq=np.ones(8, dtype=np.complex64), sample_rate_hz=8_000_000.0)
    )

    backend.start()

    assert _FakePluto.instances[-1].uri == "ip:pluto.local"
    assert _FakeTddn.instances == []


def test_pluto_prepare_disables_auto_before_config_and_runs_explicit_calibration(
    monkeypatch,
) -> None:
    _install_fakes(monkeypatch)
    backend = PlutoOutputBackend(_settings(burst_count=1))

    backend.prepare()

    events = _FakePluto.events
    manual_index = events.index("calib:manual_tx_quad")
    assert manual_index < events.index("sample_rate:8000000")
    assert manual_index < events.index("tx_lo:2441000000")
    assert manual_index < events.index("tx_rf_bandwidth:8000000")
    tx_quad_index = events.index("calib:tx_quad")
    assert tx_quad_index > events.index("tx_rf_bandwidth:8000000")
    assert events[tx_quad_index + 1] == "calib:manual_tx_quad"
    assert backend.diagnostic_report()["state"] == "READY"


def test_pluto_calibration_mode_accepts_firmware_status_suffix(monkeypatch) -> None:
    _install_fakes(monkeypatch)
    _FakePluto.hardware.update(
        sample_rate=8_000_000,
        tx_lo=2_441_000_000,
        tx_rf_bandwidth=8_000_000,
    )
    device = _FakePluto(uri="usb:test")
    device._ctx.phy.attrs["calib_mode"]._value = "manual_tx_quad 21"
    backend = PlutoOutputBackend(_settings(burst_count=1))

    backend._verify_prepared_configuration(device)

    report = backend.diagnostic_report()
    assert report["calibration_mode"] == "manual_tx_quad"
    assert report["calibration_mode_raw"] == "manual_tx_quad 21"


def test_pluto_transmit_rejects_unprepared_device_without_reconfiguring(
    monkeypatch,
) -> None:
    _install_fakes(monkeypatch)
    backend = PlutoOutputBackend(_settings(burst_count=1))
    backend.transfer(
        GenerationResult(iq=np.ones(8, dtype=np.complex64), sample_rate_hz=8_000_000.0)
    )

    with pytest.raises(RuntimeError, match="not READY"):
        backend.start()

    assert not any(event.startswith("sample_rate:") for event in _FakePluto.events)
    assert not any(event.startswith("tx_lo:") for event in _FakePluto.events)
    assert not any(event.startswith("tx_rf_bandwidth:") for event in _FakePluto.events)
    assert "buffer_transferred" not in _FakePluto.events


@pytest.mark.parametrize(
    "change",
    (
        {"sample_rate_hz": 100_000.0},
        {"rf_bandwidth_hz": 100_000.0},
        {"hardware_gain_db": 1.0},
        {"digital_backoff_db": 0.1},
        {"digital_backoff_db": -60.1},
        {"digital_backoff_db": -6.0, "output_power_dbm": -1.0},
        {"output_power_dbm": float("nan")},
        {"lead_in_guard_s": -0.001},
        {"dma_preroll_s": -0.001},
        {"stop_guard_s": 0.001},
        {"burst_count": 0},
        {"burst_count": 1001},
    ),
)
def test_pluto_backend_rejects_unsupported_output_settings(change) -> None:
    with pytest.raises(ValueError):
        PlutoOutputBackend(_settings(**change))


def test_vsg_packet_repetition_is_limited_to_one_thousand() -> None:
    project = replace(bluetooth_br_edr_project(), repeat_count=1001)

    issues = validate_project(project)

    assert any(issue.path == "repeat_count" for issue in issues)
