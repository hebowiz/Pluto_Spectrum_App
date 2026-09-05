import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

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
        window.bluetooth_workspace.center_spin.setValue(2420.0)

        window.set_analysis_mode("dect")
        assert window._stack.currentWidget() is window.dect_workspace
        assert "DECT Dedicated" in window.windowTitle()
        window.dect_workspace.capture_length_spin.setValue(3.0)

        window.set_analysis_mode("adsb1090")
        assert window._stack.currentWidget() is window.adsb1090_workspace
        assert "ADS-B 1090ES" in window.windowTitle()
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
