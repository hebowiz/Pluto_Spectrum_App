from __future__ import annotations

import pluto_sa.main as main_module


class _FakeSettings:
    values: dict[str, str] = {}

    def __init__(self, *_args) -> None:
        pass

    def value(self, key: str, default=""):
        return self.values.get(key, default)

    def setValue(self, key: str, value) -> None:
        self.values[key] = str(value)

    def sync(self) -> None:
        pass


def test_environment_target_skips_device_prompt(monkeypatch) -> None:
    monkeypatch.setenv("PLUTO_SDR_URI", "serial:chosen")
    monkeypatch.setattr(
        main_module.iio,
        "scan_contexts",
        lambda: (_ for _ in ()).throw(AssertionError("scan must not run")),
    )

    accepted, selector = main_module._choose_pluto_target()

    assert accepted is True
    assert selector == "serial:chosen"


def test_two_devices_prompt_for_and_remember_receiver_serial(monkeypatch) -> None:
    serial_a = "1044730c370e00100400120023338fb325"
    serial_b = "10447318ac0f00050a001600356a18eee6"
    monkeypatch.delenv("PLUTO_SDR_URI", raising=False)
    monkeypatch.setattr(
        main_module.iio,
        "scan_contexts",
        lambda: {
            "usb:1.31.5": f"Analog Devices PlutoSDR, serial={serial_a}",
            "usb:1.32.5": f"Analog Devices PlutoSDR, serial={serial_b}",
        },
    )
    _FakeSettings.values = {}
    monkeypatch.setattr(main_module.QtCore, "QSettings", _FakeSettings)

    def choose(_parent, _title, _prompt, labels, _selected, _editable):
        return labels[1], True

    monkeypatch.setattr(main_module.QtWidgets.QInputDialog, "getItem", choose)

    accepted, selector = main_module._choose_pluto_target()

    assert accepted is True
    assert selector == f"serial:{serial_b}"
    assert _FakeSettings.values[main_module.RTSA_DEVICE_KEY] == selector


def test_force_prompt_allows_reselecting_the_only_receiver(monkeypatch) -> None:
    serial = "1044730c370e00100400120023338fb325"
    monkeypatch.delenv("PLUTO_SDR_URI", raising=False)
    monkeypatch.setattr(
        main_module.iio,
        "scan_contexts",
        lambda: {"usb:1.31.5": f"Analog Devices PlutoSDR, serial={serial}"},
    )
    _FakeSettings.values = {main_module.RTSA_DEVICE_KEY: "serial:old"}
    monkeypatch.setattr(main_module.QtCore, "QSettings", _FakeSettings)
    prompts = []

    def choose(_parent, title, prompt, labels, selected, _editable):
        prompts.append((title, prompt, tuple(labels), selected))
        return labels[0], True

    monkeypatch.setattr(main_module.QtWidgets.QInputDialog, "getItem", choose)

    accepted, selector = main_module._choose_pluto_target(force_prompt=True)

    assert accepted is True
    assert selector == f"serial:{serial}"
    assert prompts and prompts[0][3] == 0
    assert _FakeSettings.values[main_module.RTSA_DEVICE_KEY] == selector
