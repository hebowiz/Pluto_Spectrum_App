"""Generate annotated UI screenshots used by the three user manuals.

The capture runs with Qt's offscreen platform and test doubles, so updating the
manual images never requires an attached ADALM-Pluto or changes RF state.
"""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from pluto_sa.config.spectrum_config import SpectrumConfig
from pluto_sa.signal.spectrum_processor import SpectrumProcessor
from pluto_sa.ui.session_window import SessionRealtimeSpectrumWindow
from pluto_sa.vsa.ui.application_window import PlutoAnalysisWindow
from pluto_vsg.ui.main_window import PlutoVSGWindow


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "images" / "user-manual"


class _OfflinePlutoSource:
    def close(self) -> None:
        pass

    def stop_stream(self) -> None:
        pass

    def capture_single(self, *_args, **_kwargs):
        raise RuntimeError("Manual screenshots do not use live capture")


def _widget_rect(window: QtWidgets.QWidget, widget: QtWidgets.QWidget) -> QtCore.QRect:
    origin = widget.mapTo(window, QtCore.QPoint(0, 0))
    return QtCore.QRect(origin, widget.size())


def _save_annotated(
    window: QtWidgets.QWidget,
    filename: str,
    widgets: list[QtWidgets.QWidget],
) -> None:
    pixmap = window.grab()
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    outline = QtGui.QPen(QtGui.QColor("#ff4d4d"), 3)
    painter.setPen(outline)
    font = QtGui.QFont("Segoe UI", 13)
    font.setBold(True)
    painter.setFont(font)
    for number, widget in enumerate(widgets, start=1):
        rect = _widget_rect(window, widget).adjusted(2, 2, -3, -3)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(rect, 4, 4)
        center = rect.topLeft() + QtCore.QPoint(18, 18)
        painter.setBrush(QtGui.QColor("#ff4d4d"))
        painter.drawEllipse(center, 15, 15)
        painter.setPen(QtGui.QPen(QtGui.QColor("white"), 1))
        painter.drawText(
            QtCore.QRect(center.x() - 15, center.y() - 15, 30, 30),
            QtCore.Qt.AlignmentFlag.AlignCenter,
            str(number),
        )
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.setPen(outline)
    painter.end()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not pixmap.save(str(OUTPUT_DIR / filename), "PNG"):
        raise RuntimeError(f"Could not save {filename}")


def _capture_rtsa(app: QtWidgets.QApplication) -> None:
    config = SpectrumConfig()
    receiver = MagicMock()
    receiver.device_selector = "manual-preview"
    receiver.get_received_sample_count.return_value = 0
    sweep_controller = MagicMock()
    sweep_controller.estimate_sweep_time_seconds.return_value = 0.1
    window = SessionRealtimeSpectrumWindow(
        config,
        receiver,
        SpectrumProcessor(config),
        sweep_controller,
        calibration_offset_db=0.0,
    )
    window.timer.stop()
    window.calibration_controller.set_correction_enabled(False)
    window._refresh_status_label()
    window.resize(1600, 960)
    window.show()
    app.processEvents()
    _save_annotated(
        window,
        "pluto-rtsa-overview.png",
        [
            window.status_label,
            window.waterfall_plot,
            window.spectrum_plot,
            window.control_panel,
            window.statusBar(),
        ],
    )
    window.close()
    window.deleteLater()


def _capture_vsa(app: QtWidgets.QApplication, settings_path: str) -> None:
    source = _OfflinePlutoSource()
    settings = QtCore.QSettings(settings_path, QtCore.QSettings.Format.IniFormat)
    window = PlutoAnalysisWindow(pluto_source=source, preferences=settings)
    window.resize(1600, 960)
    window.show()
    app.processEvents()
    generic = window.generic_workspace
    _save_annotated(
        window,
        "pluto-vsa-generic-overview.png",
        [
            generic.zero_span_dock,
            generic.spectrum_dock,
            generic.result_summary_dock,
            generic.modulation_dock,
            generic.symbol_plot_dock,
            window.control_panel,
        ],
    )
    window.set_analysis_mode("bluetooth")
    app.processEvents()
    bluetooth = window.bluetooth_workspace
    _save_annotated(
        window,
        "pluto-vsa-bluetooth-overview.png",
        [
            bluetooth.power_dock,
            bluetooth.spectrum_dock,
            bluetooth.summary_dock,
            bluetooth.modulation_dock,
            bluetooth.packet_dock,
            window.control_panel,
        ],
    )
    window.close()
    window.deleteLater()


def _capture_vsg(app: QtWidgets.QApplication) -> None:
    window = PlutoVSGWindow()
    window._pluto_output_power_dbm = -20.0
    window._power_step_db = 1.0
    window._update_vsg_control_labels()
    window.resize(1500, 900)
    window.show()
    app.processEvents()
    _save_annotated(
        window,
        "pluto-vsg-overview.png",
        [
            window.block_library,
            window.composer_view,
            window.inspector,
            window.iq_waveform_plot,
            window.rf_button.parentWidget(),
        ],
    )
    window.close()
    window.deleteLater()


def main() -> int:
    app = pg.mkQApp("Pluto manuals screenshot generator")
    with TemporaryDirectory(prefix="pluto-manual-") as temp_dir:
        _capture_rtsa(app)
        _capture_vsa(app, str(Path(temp_dir) / "vsa.ini"))
        _capture_vsg(app)
    app.processEvents()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
