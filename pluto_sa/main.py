"""Application entry point."""

from __future__ import annotations

import os

import iio
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from pluto_common import discover_pluto_devices
from pluto_sa.config.session_state import (
    RTSA_APPLICATION,
    RTSA_DEVICE_KEY,
    RTSA_ORGANIZATION,
)
from pluto_sa.config.spectrum_config import SpectrumConfig
from pluto_sa.modes.sweep_controller import SweepController
from pluto_sa.sdr.pluto_receiver import PlutoReceiver
from pluto_sa.signal.spectrum_processor import SpectrumProcessor
from pluto_sa.ui.session_window import SessionRealtimeSpectrumWindow


def _choose_pluto_target(parent=None) -> tuple[bool, str | None]:
    """Choose a physical receiver, reusing the previous choice when possible."""

    environment_target = os.environ.get("PLUTO_SDR_URI", "").strip()
    if environment_target:
        return True, environment_target
    try:
        devices = discover_pluto_devices(iio.scan_contexts())
    except Exception:
        devices = ()
    if not devices:
        return True, None

    settings = QtCore.QSettings(RTSA_ORGANIZATION, RTSA_APPLICATION)
    saved = str(settings.value(RTSA_DEVICE_KEY, "")).strip()
    if saved:
        saved_key = saved.casefold()
        for device in devices:
            if device.selector.casefold() == saved_key:
                return True, device.selector

    if len(devices) == 1:
        selector = devices[0].selector
        settings.setValue(RTSA_DEVICE_KEY, selector)
        settings.sync()
        return True, selector

    labels = [device.label for device in devices]
    label, accepted = QtWidgets.QInputDialog.getItem(
        parent,
        "Select ADALM-Pluto Receiver",
        "Two or more Pluto devices are connected. Select the RTSA receiver:",
        labels,
        0,
        False,
    )
    if not accepted:
        return False, None
    index = labels.index(label)
    selector = devices[index].selector
    settings.setValue(RTSA_DEVICE_KEY, selector)
    settings.sync()
    return True, selector


def build_app_components(sdr_uri: str | None = None) -> tuple[
    SpectrumConfig,
    PlutoReceiver,
    SpectrumProcessor,
    SweepController,
    SessionRealtimeSpectrumWindow,
]:
    config = SpectrumConfig(sdr_uri=sdr_uri)
    receiver = PlutoReceiver(config)
    processor = SpectrumProcessor(config)
    sweep_controller = SweepController(config, receiver)
    window = SessionRealtimeSpectrumWindow(
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
        _, _, _, _, window = build_app_components(sdr_uri=sdr_uri)
        window.start_initial_acquisition()
    except Exception as error:
        QtWidgets.QMessageBox.critical(
            None,
            "ADALM-Pluto Connection Error",
            f"Could not open the selected receiver.\n\n{error}",
        )
        return 1
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
