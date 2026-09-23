"""Generate annotated UI screenshots used by the three user manuals.

Render recorded IQ and generated waveforms in the actual application widgets.
Device discovery and acquisition are replaced; no Pluto or RF transmission is
used. On Windows, use the native Qt platform so installed fonts render correctly.
"""

from __future__ import annotations

import os
import json
import time
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "windows" if os.name == "nt" else "offscreen")

import pyqtgraph as pg
import iio
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from pluto_common.config.spectrum_config import SpectrumConfig
from pluto_rtsa.signal.spectrum_processor import SpectrumProcessor
from pluto_rtsa.ui.session_window import SessionRealtimeSpectrumWindow
from pluto_vsa.ui.application_window import PlutoAnalysisWindow
from pluto_vsg.ui.main_window import PlutoVSGWindow
from pluto_vsa.sources import FileIQSource
from pluto_vsa.model import SignalDescription, ModulationKind


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "images" / "user-manual"
ROOT = OUTPUT_DIR.parents[2]
INVENTORY = {}


def _settle(app, workspace=None):
    deadline = time.monotonic() + 90
    while True:
        app.processEvents()
        busy = workspace is not None and any(
            getattr(workspace, name, None) is not None
            for name in ("_analysis_thread", "_capture_thread")
        )
        if not busy:
            break
        if time.monotonic() > deadline:
            raise RuntimeError("Manual analysis did not finish")
        time.sleep(0.01)
    for _ in range(5):
        app.processEvents()


def _inventory(name, widget):
    rows = []
    for form in widget.findChildren(QtWidgets.QFormLayout):
        for index in range(form.rowCount()):
            label = form.itemAt(index, QtWidgets.QFormLayout.ItemRole.LabelRole)
            field = form.itemAt(index, QtWidgets.QFormLayout.ItemRole.FieldRole)
            label_widget = label.widget() if label else None
            title = label_widget.text() if isinstance(label_widget, QtWidgets.QLabel) else ""
            controls = []
            if field:
                root = field.widget()
                controls = ([root] + root.findChildren(QtWidgets.QWidget)) if root else []
                if field.layout():
                    controls = [field.layout().itemAt(i).widget() for i in range(field.layout().count())]
            values = []
            for control in controls:
                if isinstance(control, QtWidgets.QComboBox):
                    values.append({"choices": [control.itemText(i) for i in range(control.count())], "value": control.currentText(), "enabled": control.isEnabled()})
                elif isinstance(control, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
                    values.append({"range": [control.minimum(), control.maximum()], "unit": control.suffix(), "value": control.value(), "enabled": control.isEnabled()})
                elif isinstance(control, (QtWidgets.QCheckBox, QtWidgets.QLabel, QtWidgets.QLineEdit)):
                    values.append({"text": control.text(), "enabled": control.isEnabled()})
            rows.append({"label": title, "controls": values})
    INVENTORY[name] = rows


def _config_images(app, mode, workspace):
    dialog = (getattr(workspace, "_meas_config_dialog", None)
              or getattr(workspace, "_config_dialog", None) or workspace._setup_dialog)
    for index, page in enumerate(dialog.page_names):
        if page in ("Config Top Menu", "Sweep / Run"):
            continue
        dialog.show_page(index)
        dialog.back_button.hide()
        dialog.resize(1000, 940)
        dialog.show()
        _settle(app)
        slug = page.lower().replace(" / ", "-").replace(" ", "-")
        _save_annotated(dialog, f"pluto-vsa-{mode}-{slug}.png", [])
        _inventory(f"vsa/{mode}/{page}", dialog.stack.widget(index))
    dialog.hide()


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


def _capture_rtsa(app: QtWidgets.QApplication, settings_path: str) -> None:
    config = SpectrumConfig()
    recording = FileIQSource.load(ROOT / "tests/fixtures/bluetooth_br_prbs9_pluto_16msps.npz")
    config.center_freq_hz = int(recording.center_frequency_hz)
    config.display_span_hz = int(recording.sample_rate_hz * (1 - 2 * config.guard_ratio))
    config.rx_gain_db = 0
    config.ext_att_db = recording.input_correction_db
    config.rbw_hz = 100_000.0
    config.ref_level_dbm = 0.0
    config.display_range_db = 80.0
    config.waterfall_history = 30
    config.waterfall_decimation = 1
    receiver = MagicMock()
    receiver.device_selector = "manual-preview"
    receiver.get_received_sample_count.return_value = 0
    sweep_controller = MagicMock()
    sweep_controller.estimate_sweep_time_seconds.return_value = 0.1
    with patch.object(SessionRealtimeSpectrumWindow, "_load_app_settings", lambda self: None):
        window = SessionRealtimeSpectrumWindow(
            config, receiver, SpectrumProcessor(config), sweep_controller,
            calibration_offset_db=recording.calibration_offset_db,
            preferences=QtCore.QSettings(settings_path, QtCore.QSettings.Format.IniFormat),
        )
    window.timer.stop()
    window.calibration_controller.set_correction_enabled(False)
    window._refresh_status_label()
    window.status_label.setMinimumHeight(window.status_label.sizeHint().height())
    window.resize(1600, 960)
    window.show()
    app.processEvents()
    for start in range(0, recording.sample_count - config.fft_size, config.fft_size):
        frame = recording.iq[start:start + config.fft_size]
        power = window.processor.compute_filtered_power(frame)
        window._read_realtime_detector_frame = lambda p=power: SimpleNamespace(power_linear=p, input_samples=config.fft_size)
        window.update_spectrum()
    window.statusBar().showMessage("Recorded Pluto IQ replay - no live receiver; nominal amplitude reference")
    _settle(app)
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
    for name, page, title in (
        ("frequency", window.freq_channel_page, "Frequency"),
        ("amplitude", window.amptd_y_scale_page, "Amplitude"),
        ("input", window.input_page, "Input"),
        ("bw", window.bw_page, "BW"),
        ("display", window.display_page, "Display"),
        ("trace", window.trace_detail_pages[0], "Trace 1"),
        ("fft", window.fft_parameter_mode_page, "FFT Parameters"),
        ("sweep", window.sweep_page, "Sweep"),
        ("trigger", window.trigger_page, "Trigger"),
        ("calibration", window.calibration_menu_page, "Calibration"),
    ):
        window._show_control_page(title, page)
        _settle(app)
        _save_annotated(window.control_panel, f"pluto-rtsa-{name}.png", [])
    window.control_stack.setCurrentWidget(window.main_menu_page)
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
    from pluto_vsa.profiles.bluetooth_br import access_code_bits
    generic.pattern_search_check.setChecked(True)
    generic.pattern_name_edit.setText("BR access code C6967E")
    generic._set_pattern_symbols(tuple(int(bit) for bit in access_code_bits(0xC6967E)))
    generic.result_length_spin.setValue(256)
    recording = FileIQSource.load(ROOT / "tests/fixtures/bluetooth_br_prbs9_pluto_16msps.npz")
    generic.load_recording(recording, SignalDescription(modulation=ModulationKind.FSK, symbol_rate_hz=1e6, frequency_deviation_hz=160e3, tx_filter="Gaussian", filter_parameter=0.5))
    _settle(app, generic)
    if generic.session.result is None:
        raise RuntimeError("General VSA screenshot has no analysis result")
    if generic.session.pattern_result is None:
        raise RuntimeError("General VSA screenshot has no pattern synchronization")
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
    _config_images(app, "general", generic)
    window.set_analysis_mode("bluetooth")
    app.processEvents()
    bluetooth = window.bluetooth_workspace
    bluetooth.profile_combo.setCurrentIndex(bluetooth.profile_combo.findText("General Packet"))
    with patch.object(QtWidgets.QFileDialog, "getOpenFileName", return_value=(str(ROOT / "tests/fixtures/RT_Packet_TX_2DH1.npz"), "")):
        bluetooth._open_iq()
    _settle(app, bluetooth)
    if bluetooth._result is None:
        raise RuntimeError("Bluetooth screenshot has no analysis result")
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
    _config_images(app, "bluetooth", bluetooth)
    window.set_analysis_mode("dect")
    dect = window.dect_workspace
    dect.plan_combo.setCurrentIndex(dect.plan_combo.findData("j_dect"))
    dect.carrier_combo.setCurrentIndex(dect.carrier_combo.findData(1902528000.0))
    dect.load_recording(FileIQSource.load(ROOT / "tests/fixtures/DECT_PP_A5_OK.npz"))
    _settle(app, dect)
    if dect._result is None:
        raise RuntimeError("DECT screenshot has no analysis result")
    _save_annotated(window, "pluto-vsa-dect-overview.png", [])
    _config_images(app, "dect", dect)
    window.set_analysis_mode("adsb1090")
    adsb = window.adsb1090_workspace
    adsb.analyze_recording(FileIQSource.load(ROOT / "tests/fixtures/adsb1090_multi_8msps.npz"))
    _settle(app, adsb)
    _save_annotated(window, "pluto-vsa-adsb-overview.png", [])
    _config_images(app, "adsb", adsb)
    window.close()
    window.deleteLater()


def _capture_vsg(app: QtWidgets.QApplication, settings_path: str) -> None:
    from pluto_vsg.ui.main_window import _BluetoothSettingsDialog, _BluetoothLESettingsDialog, _BluetoothHDTSettingsDialog, _WiFiSettingsDialog
    from pluto_vsg.ui.dect_settings import DectSettingsDialog
    from pluto_vsg.profiles import bluetooth_br_edr_project, bluetooth_le_project, bluetooth_hdt_project, wifi_project, dect_project
    window = PlutoVSGWindow(preferences=QtCore.QSettings(settings_path, QtCore.QSettings.Format.IniFormat))
    window._pluto_output_power_dbm = -20.0
    window._power_step_db = 1.0
    window._update_vsg_control_labels()
    window.resize(1500, 900)
    window.show()
    app.processEvents()
    window._verify_packet()
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
    for name, cls, project in (
        ("classic", _BluetoothSettingsDialog, bluetooth_br_edr_project()),
        ("le", _BluetoothLESettingsDialog, bluetooth_le_project()),
        ("hdt", _BluetoothHDTSettingsDialog, bluetooth_hdt_project()),
        ("wifi", _WiFiSettingsDialog, wifi_project()),
        ("dect", DectSettingsDialog, dect_project()),
    ):
        dialog = cls(project, window)
        dialog.resize(1050, 940)
        dialog.show()
        for index in range(dialog.tabs.count()):
            dialog.tabs.setCurrentIndex(index)
            _settle(app)
            _save_annotated(dialog, f"pluto-vsg-{name}-settings-{index}.png", [])
            _inventory(f"vsg/{name}/{dialog.tabs.tabText(index)}", dialog.tabs.widget(index))
        dialog.close()
    window.close()
    window.deleteLater()


def main() -> int:
    iio.scan_contexts = lambda: {}
    app = pg.mkQApp("Pluto manuals screenshot generator")
    app.setFont(QtGui.QFont("Segoe UI", 10))
    def fail_dialog(*args):
        raise RuntimeError(str(args[1:]))
    QtWidgets.QMessageBox.critical = fail_dialog
    QtWidgets.QMessageBox.warning = fail_dialog
    with TemporaryDirectory(prefix="pluto-manual-") as temp_dir:
        _capture_rtsa(app, str(Path(temp_dir) / "rtsa.ini"))
        _capture_vsa(app, str(Path(temp_dir) / "vsa.ini"))
        _capture_vsg(app, str(Path(temp_dir) / "vsg.ini"))
    inventory_path = ROOT / "tmp/manual-ui-inventory.json"
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(json.dumps(INVENTORY, ensure_ascii=False, indent=2), encoding="utf-8")
    app.processEvents()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
