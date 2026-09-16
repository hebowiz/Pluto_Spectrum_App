import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from pluto_sa.vsa.ui.application_window import PlutoAnalysisWindow


class _SharedPlutoSource:
    def __init__(self) -> None:
        self.close_count = 0
        self.stop_stream_count = 0
        self.capture_count = 0

    def close(self) -> None:
        self.close_count += 1

    def stop_stream(self) -> None:
        self.stop_stream_count += 1

    def capture_single(self, *_args, **_kwargs):
        self.capture_count += 1
        raise AssertionError("capture must be started only by an explicit Run action")


class _CancellableCapture(QtCore.QObject):
    finished = QtCore.Signal()

    def __init__(self) -> None:
        super().__init__()
        self.running = True
        self.cancel_count = 0

    def isRunning(self) -> bool:
        return self.running

    def cancel(self) -> None:
        self.cancel_count += 1


def test_single_window_switches_complete_workspaces_and_shares_pluto(tmp_path) -> None:
    pg.mkQApp("Pluto analysis shell test")
    preferences = QtCore.QSettings(
        str(tmp_path / "analysis-shell.ini"),
        QtCore.QSettings.Format.IniFormat,
    )
    source = _SharedPlutoSource()
    window = PlutoAnalysisWindow(pluto_source=source, preferences=preferences)
    try:
        assert window._stack.currentWidget() is window.generic_workspace
        assert not window.generic_workspace.menuBar().isVisible()
        assert window.control_panel.width() == 240
        assert not window.generic_workspace.open_config_action.isEnabled()
        assert all(
            not bool(
                dock.features()
                & QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
            )
            for dock in window.generic_workspace.findChildren(QtWidgets.QDockWidget)
        )
        assert {
            "Analyzer Mode",
            "Input / Frontend",
            "Signal Description",
            "Signal Capture",
            "Trigger",
            "Pattern Search",
            "Result Range",
            "Demodulation",
            "Result Summary",
            "Display",
            "Continuous",
            "Single",
            "Refresh Analysis",
            "Reset",
            "Device",
            "State",
            "File",
        }.issubset(window.control_panel.buttons)
        assert window.control_panel.buttons["Analyzer Mode"].text() == (
            "Analyzer Mode\nGeneric VSA"
        )
        single_action = window.generic_workspace.run_single_action
        continuous_action = window.generic_workspace.run_continuous_action
        single_action.setText("Stop Single")
        assert window.control_panel.buttons["Single"].isChecked()
        assert not window.control_panel.buttons["Continuous"].isEnabled()
        single_action.setText("Run Single")
        assert not window.control_panel.buttons["Single"].isChecked()
        assert window.control_panel.buttons["Continuous"].isEnabled()
        continuous_action.setText("Stop Continuous")
        assert window.control_panel.buttons["Continuous"].isChecked()
        single_action.setEnabled(False)
        single_action.setText("Stop")
        assert window.control_panel.buttons["Continuous"].isEnabled()
        assert window.control_panel.buttons["Continuous"].isChecked()
        assert window.control_panel.buttons["Single"].text() == "Single"
        assert not window.control_panel.buttons["Single"].isChecked()
        single_action.setText("Run Single")
        single_action.setEnabled(True)
        continuous_action.setText("Run Continuous")
        assert not window.control_panel.buttons["Continuous"].isChecked()
        QtCore.QTimer.singleShot(0, window.generic_workspace._meas_config_dialog.reject)
        window.control_panel.buttons["Signal Description"].click()
        assert window.generic_workspace._config_page_title.text() == "Signal Description"
        assert not window.generic_workspace._config_back_button.isVisible()
        assert "Generic" in window.windowTitle()
        assert window.generic_workspace._pluto_source is source
        assert window.adsb1090_workspace._pluto_source is source
        assert window.bluetooth_workspace._pluto_source is source
        assert window.dect_workspace._pluto_source is source
        assert window.bluetooth_workspace._recording is None
        assert window.dect_workspace._recording is None
        assert window.adsb1090_workspace.recording is None
        window.generic_workspace.analysis_published.emit(
            window.generic_workspace.session
        )
        assert window.bluetooth_workspace._recording is None
        assert window.dect_workspace._recording is None
        assert window.adsb1090_workspace.recording is None

        window.set_analysis_mode("bluetooth")
        assert window._stack.currentWidget() is window.bluetooth_workspace
        assert "Bluetooth Dedicated" in window.windowTitle()
        assert window.control_panel.buttons["Analyzer Mode"].text() == (
            "Analyzer Mode\nBluetooth"
        )
        assert "Signal Description" in window.control_panel.buttons
        window.bluetooth_workspace.center_spin.setValue(2420.0)

        window.set_analysis_mode("dect")
        assert window.control_panel.buttons["Analyzer Mode"].text() == (
            "Analyzer Mode\nDECT"
        )
        assert window._stack.currentWidget() is window.dect_workspace
        assert "DECT Dedicated" in window.windowTitle()
        assert "Signal Description" in window.control_panel.buttons
        window.dect_workspace.capture_length_spin.setValue(3.0)

        window.set_analysis_mode("adsb1090")
        assert window._stack.currentWidget() is window.adsb1090_workspace
        assert "ADS-B 1090ES" in window.windowTitle()
        assert "Signal Description" in window.control_panel.buttons
        assert "Display" not in window.control_panel.buttons
        window.adsb1090_workspace.capture_length_spin.setValue(300.0)

        window.set_analysis_mode("generic")
        assert window._stack.currentWidget() is window.generic_workspace
        assert source.capture_count == 0
        assert source.stop_stream_count >= 4
    finally:
        window.generic_workspace._meas_config_dialog.close()
        window.close()
        window.deleteLater()
    assert source.close_count == 1


def test_close_requests_capture_stop_then_closes_shared_source(tmp_path) -> None:
    app = pg.mkQApp("Pluto analysis graceful-close test")
    preferences = QtCore.QSettings(
        str(tmp_path / "analysis-close.ini"),
        QtCore.QSettings.Format.IniFormat,
    )
    source = _SharedPlutoSource()
    window = PlutoAnalysisWindow(pluto_source=source, preferences=preferences)
    capture = _CancellableCapture()
    window.generic_workspace._pluto_capture_thread = capture
    window.show()

    assert window.close() is False
    assert capture.cancel_count == 1
    assert source.close_count == 0

    capture.running = False
    window._continue_shutdown()
    app.processEvents()

    assert source.close_count == 1


def test_mode_aware_recall_switches_workspace_without_capture(tmp_path) -> None:
    pg.mkQApp("Pluto analysis mode-aware config test")
    preferences = QtCore.QSettings(
        str(tmp_path / "analysis-recall.ini"),
        QtCore.QSettings.Format.IniFormat,
    )
    source = _SharedPlutoSource()
    window = PlutoAnalysisWindow(pluto_source=source, preferences=preferences)
    path = tmp_path / "bluetooth.vsaconfig.json"
    try:
        window.set_analysis_mode("bluetooth")
        window.bluetooth_workspace.protocol_combo.setCurrentIndex(
            window.bluetooth_workspace.protocol_combo.findData("bluetooth.le")
        )
        window.bluetooth_workspace.phy_combo.setCurrentText("LE 2M")
        window.bluetooth_workspace.center_spin.setValue(2442.0)
        window.save_meas_config_path(path)

        window.set_analysis_mode("generic")
        assert window.recall_meas_config_path(path)
        assert window._active_mode() == "bluetooth"
        assert window.bluetooth_workspace.protocol_combo.currentData() == "bluetooth.le"
        assert window.bluetooth_workspace.phy_combo.currentText() == "LE 2M"
        assert window.bluetooth_workspace.center_spin.value() == 2442.0
        assert window.control_panel.stack.currentWidget() is window.control_panel.main_page
        assert source.capture_count == 0
    finally:
        window.close()
        window.deleteLater()


def test_each_mode_has_one_default_preset_without_capture(tmp_path, monkeypatch) -> None:
    pg.mkQApp("Pluto analysis default preset test")
    preferences = QtCore.QSettings(
        str(tmp_path / "analysis-preset.ini"),
        QtCore.QSettings.Format.IniFormat,
    )
    source = _SharedPlutoSource()
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *_args, **_kwargs: QtWidgets.QMessageBox.StandardButton.Ok,
    )
    window = PlutoAnalysisWindow(pluto_source=source, preferences=preferences)
    try:
        for mode in ("generic", "bluetooth", "dect", "adsb1090"):
            window.set_analysis_mode(mode)
            assert "Preset" in window.control_panel.buttons
            assert "Default" not in window.control_panel.buttons
            preset_buttons = [
                key for key in window.control_panel.buttons if key == "Preset"
            ]
            assert preset_buttons == ["Preset"]

        window.set_analysis_mode("bluetooth")
        window.bluetooth_workspace.center_spin.setValue(2420.0)
        window._apply_default_preset()
        assert window.bluetooth_workspace.center_spin.value() == 2440.0
        assert source.capture_count == 0
    finally:
        window.close()
        window.deleteLater()
