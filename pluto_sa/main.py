"""Application entry point."""

import pyqtgraph as pg

from pluto_sa.config.spectrum_config import SpectrumConfig
from pluto_sa.sdr.pluto_receiver import PlutoReceiver
from pluto_sa.signal.spectrum_processor import SpectrumProcessor
from pluto_sa.ui.main_window import RealtimeSpectrumWindow


def build_app_components() -> tuple[
    SpectrumConfig,
    PlutoReceiver,
    SpectrumProcessor,
    RealtimeSpectrumWindow,
]:
    config = SpectrumConfig()
    receiver = PlutoReceiver(config)
    processor = SpectrumProcessor(config)
    window = RealtimeSpectrumWindow(
        config,
        receiver,
        processor,
        calibration_offset_db=config.calibration_offset_db,
    )
    return config, receiver, processor, window


def main() -> int:
    app = pg.mkQApp("PlutoSDR Real-Time Spectrum Prototype")
    _, receiver, _, window = build_app_components()
    receiver.start()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
