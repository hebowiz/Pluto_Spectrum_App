"""Application entry point."""

from __future__ import annotations

import os

import iio
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from pluto_common import discover_pluto_devices
from pluto_sa.config.spectrum_config import SpectrumConfig
from pluto_sa.modes.sweep_controller import SweepController
from pluto_sa.sdr.pluto_receiver import PlutoReceiver
from pluto_sa.signal.spectrum_processor import SpectrumProcessor
from pluto_sa.ui.main_window import RealtimeSpectrumWindow


_RTSA_ORGANIZATION = "PlutoSpectrumApp"
_RTSA_APPLICATION = "PlutoRTSA"
_RTSA_DEVICE_KEY = "pluto_rx/selector"


def _choose_pluto_target(parent=None) -> tuple[bool, str | None]:
    """Choose one physical receiver when more than one Pluto is connected."""

    environment_target = os.environ.get("PLUTO_SDR_URI", "").strip()
    if environment_target:
        return True, environment_target
    try:
        devices = discover_pluto_devices(iio.scan_contexts())
    except Exception:
        devices = ()
    if not devices:
        return True, None
    if len(devices) == 1:
        return True, devices[0].selector

    settings = QtCore.QSettings(_RTSA_ORGANIZATION, _RTSA_APPLICATION)
    saved = str(settings.value(_RTSA_DEVICE_KEY, ""))
    labels = [device.label for device in devices]
    selected_index = next(
        (index for index, device in enumerate(devices) if device.selector == saved),
        0,
    )
    label, accepted = QtWidgets.QInputDialog.getItem(
        parent,
        "Select ADALM-Pluto Receiver",
        "Two or more Pluto devices are connected. Select the RTSA receiver:",
        labels,
        selected_index,
        False,
    )
    if not accepted:
        return False, None
    index = labels.index(label)
    selector = devices[index].selector
    settings.setValue(_RTSA_DEVICE_KEY, selector)
    return True, selector


def build_app_components(sdr_uri: str | None = None) -> tuple[
    SpectrumConfig,
    PlutoReceiver,
    SpectrumProcessor,
    SweepController,
    RealtimeSpectrumWindow,
]:
    config = SpectrumConfig(sdr_uri=sdr_uri)
    receiver = PlutoReceiver(config)
    processor = SpectrumProcessor(config)
    sweep_controller = SweepController(config, receiver)
    window = RealtimeSpectrumWindow(
        config,
        receiver,
        processor,
        sweep_controller,
        calibration_offset_db=config.calibration_offset_db,
    )
    return config, receiver, processor, sweep_controller, window


def main() -> int:
    app = pg.mkQApp("PlutoSDR Real-Time Spectrum Prototype")
    accepted, sdr_uri = _choose_pluto_target()
    if not accepted:
        return 0
    try:
        _, receiver, _, _, window = build_app_components(sdr_uri=sdr_uri)
    except Exception as error:
        QtWidgets.QMessageBox.critical(
            None,
            "ADALM-Pluto Connection Error",
            f"Could not open the selected receiver.\n\n{error}",
        )
        return 1
    receiver.start()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
