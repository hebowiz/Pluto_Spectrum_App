"""Single-window shell for generic and standard-specific analyzers."""

from __future__ import annotations

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from pluto_common import short_pluto_identity

from pluto_sa.standards.adsb1090.ui import ADSB1090Window
from pluto_sa.vsa.pluto_source import PlutoLiveSource
from pluto_sa.vsa.protocol_modes.bluetooth import BluetoothAnalyzerWindow
from pluto_sa.vsa.protocol_modes.dect import DectAnalyzerWindow
from pluto_sa.vsa.ui.main_window import VSAWindow


class PlutoAnalysisWindow(QtWidgets.QMainWindow):
    """Own one Pluto connection and switch complete measurement workspaces."""

    def __init__(
        self,
        pluto_source: PlutoLiveSource | None = None,
        preferences: QtCore.QSettings | None = None,
    ) -> None:
        super().__init__()
        self._shutdown_requested = False
        self._shutdown_finalized = False
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
        self.bluetooth_workspace = BluetoothAnalyzerWindow(
            pluto_source=self._pluto_source,
        )
        self.dect_workspace = DectAnalyzerWindow(pluto_source=self._pluto_source)
        for workspace in (
            self.generic_workspace,
            self.bluetooth_workspace,
            self.dect_workspace,
            self.adsb1090_workspace,
        ):
            workspace.setWindowFlags(QtCore.Qt.WindowType.Widget)
            self._stack.addWidget(workspace)
            workspace.analysis_mode_requested.connect(self.set_analysis_mode)
            workspace.application_close_requested.connect(self.close)
            if hasattr(workspace, "shutdown_ready"):
                workspace.shutdown_ready.connect(self._continue_shutdown)
        self.generic_workspace.pluto_uri_edit.currentTextChanged.connect(
            self._pluto_target_changed
        )
        self._pluto_target_changed(self.generic_workspace._selected_pluto_target())
        self.resize(1600, 960)
        self.set_analysis_mode("generic")

    def _busy_reason(self) -> str | None:
        generic = self.generic_workspace.shutdown_busy_reason()
        if generic is not None:
            return generic
        adsb = self.adsb1090_workspace.shutdown_busy_reason()
        if adsb is not None:
            return adsb
        bluetooth = self.bluetooth_workspace.shutdown_busy_reason()
        if bluetooth is not None:
            return bluetooth
        dect = self.dect_workspace.shutdown_busy_reason()
        if dect is not None:
            return dect
        return None

    @QtCore.Slot(str)
    def set_analysis_mode(self, mode: str) -> None:
        target = {
            "generic": self.generic_workspace,
            "bluetooth": self.bluetooth_workspace,
            "dect": self.dect_workspace,
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
        stop_stream = getattr(self._pluto_source, "stop_stream", None)
        if callable(stop_stream):
            stop_stream()
        self._stack.setCurrentWidget(target)
        self._update_window_title(target)

    def _update_window_title(self, target: QtWidgets.QWidget) -> None:
        identity = short_pluto_identity(
            self.generic_workspace._selected_pluto_target()
        )
        if target is self.generic_workspace:
            self.setWindowTitle(f"Pluto VSA - Generic FSK / PSK [RX: {identity}]")
        elif target is self.bluetooth_workspace:
            self.setWindowTitle(
                f"Pluto VSA - Bluetooth Dedicated Analyzer [RX: {identity}]"
            )
        elif target is self.dect_workspace:
            self.setWindowTitle(
                f"Pluto VSA - DECT Dedicated Analyzer [RX: {identity}]"
            )
        else:
            self.setWindowTitle(f"Pluto VSA - ADS-B 1090ES [RX: {identity}]")

    @QtCore.Slot(str)
    def _pluto_target_changed(self, _text: str) -> None:
        target = self.generic_workspace._selected_pluto_target()
        self.adsb1090_workspace.set_pluto_target(target)
        self.bluetooth_workspace.set_pluto_target(target)
        self.dect_workspace.set_pluto_target(target)
        current = self._stack.currentWidget()
        if current is not None:
            self._update_window_title(current)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._shutdown_requested = True
        self.generic_workspace.request_shutdown()
        self.bluetooth_workspace.request_shutdown()
        self.dect_workspace.request_shutdown()
        self.adsb1090_workspace.request_shutdown()
        busy = self._busy_reason()
        if busy is not None:
            self.statusBar().showMessage(f"Stopping {busy} before closing...")
            event.ignore()
            return
        if not self._shutdown_finalized:
            self._shutdown_finalized = True
            self.adsb1090_workspace.finalize_shutdown()
            self.bluetooth_workspace.finalize_shutdown()
            self.dect_workspace.finalize_shutdown()
            self.generic_workspace.finalize_shutdown()
            self._pluto_source.close()
        event.accept()
        super().closeEvent(event)

    @QtCore.Slot()
    def _continue_shutdown(self) -> None:
        if self._shutdown_requested:
            self.close()
