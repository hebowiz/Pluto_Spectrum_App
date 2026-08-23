"""Single-window shell for generic and standard-specific analyzers."""

from __future__ import annotations

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from pluto_sa.standards.adsb1090.ui import ADSB1090Window
from pluto_sa.vsa.pluto_source import PlutoLiveSource
from pluto_sa.vsa.ui.main_window import VSAWindow


class PlutoAnalysisWindow(QtWidgets.QMainWindow):
    """Own one Pluto connection and switch complete measurement workspaces."""

    def __init__(
        self,
        pluto_source: PlutoLiveSource | None = None,
        preferences: QtCore.QSettings | None = None,
    ) -> None:
        super().__init__()
        self._pluto_source = pluto_source or PlutoLiveSource()
        self._stack = QtWidgets.QStackedWidget()
        self.setCentralWidget(self._stack)
        self.generic_workspace = VSAWindow(
            preferences=preferences,
            pluto_source=self._pluto_source,
            owns_pluto_source=False,
        )
        self.adsb1090_workspace = ADSB1090Window(
            pluto_source=self._pluto_source,
            owns_pluto_source=False,
        )
        for workspace in (self.generic_workspace, self.adsb1090_workspace):
            workspace.setWindowFlags(QtCore.Qt.WindowType.Widget)
            self._stack.addWidget(workspace)
            workspace.analysis_mode_requested.connect(self.set_analysis_mode)
            workspace.application_close_requested.connect(self.close)
        self.resize(1600, 960)
        self.set_analysis_mode("generic")

    def _busy_reason(self) -> str | None:
        capture = self.generic_workspace._pluto_capture_thread
        if capture is not None and capture.isRunning():
            return "Generic VSA Pluto capture is running"
        analysis = self.generic_workspace._analysis_thread
        if analysis is not None and analysis.isRunning():
            return "Generic VSA analysis is running"
        adsb_capture = self.adsb1090_workspace._capture_thread
        if adsb_capture is not None and adsb_capture.isRunning():
            return "ADS-B Pluto capture is running"
        adsb_analysis = self.adsb1090_workspace._analysis_stream_thread
        if adsb_analysis is not None and adsb_analysis.isRunning():
            return "ADS-B stream analysis is running"
        return None

    @QtCore.Slot(str)
    def set_analysis_mode(self, mode: str) -> None:
        target = {
            "generic": self.generic_workspace,
            "adsb1090": self.adsb1090_workspace,
        }.get(str(mode))
        if target is None:
            raise ValueError(f"unsupported analysis mode: {mode}")
        if target is self._stack.currentWidget():
            self._update_window_title(target)
            return
        busy = self._busy_reason()
        if busy is not None:
            QtWidgets.QMessageBox.information(
                self,
                "Analysis Mode",
                f"{busy}. Stop it before changing modes.",
            )
            return
        if target is self.adsb1090_workspace:
            recording = self.generic_workspace.session.recording
            if recording is not None and recording is not target.recording:
                target.analyze_recording(recording)
        self._stack.setCurrentWidget(target)
        self._update_window_title(target)

    def _update_window_title(self, target: QtWidgets.QWidget) -> None:
        if target is self.generic_workspace:
            self.setWindowTitle("Pluto VSA - Generic FSK / PSK")
        else:
            self.setWindowTitle("Pluto VSA - ADS-B 1090ES")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        busy = self._busy_reason()
        if busy is not None:
            self.statusBar().showMessage(f"{busy}; stop it before closing.")
            event.ignore()
            return
        self.adsb1090_workspace.prepare_for_shutdown()
        self.generic_workspace._save_startup_meas_config()
        self._pluto_source.close()
        event.accept()
        super().closeEvent(event)
