"""Single-window shell for generic and standard-specific analyzers."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from pluto_common import short_pluto_identity

from pluto_vsa.standards.adsb1090.ui import ADSB1090Window
from pluto_vsa.pluto_source import PlutoLiveSource
from pluto_vsa.session import VSASession
from pluto_vsa.protocol_modes.bluetooth import BluetoothAnalyzerWindow
from pluto_vsa.protocol_modes.dect import DectAnalyzerWindow
from pluto_vsa.persistence import load_mode_meas_config, save_mode_meas_config
from pluto_vsa.ui.control_panel import (
    PanelCommand,
    VSAControlPanel,
    WorkspacePanelSpec,
)
from pluto_vsa.ui.main_window import VSAWindow


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
        self._preferences = preferences or QtCore.QSettings("PlutoSA", "PlutoVSA")
        self._stack = QtWidgets.QStackedWidget()
        central = QtWidgets.QWidget()
        central_layout = QtWidgets.QHBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(6)
        central_layout.addWidget(self._stack, 1)
        self.control_panel = VSAControlPanel()
        central_layout.addWidget(self.control_panel)
        self.setCentralWidget(central)
        self.generic_workspace = VSAWindow(
            preferences=self._preferences,
            pluto_source=self._pluto_source,
            owns_pluto_source=False,
        )
        self.adsb1090_workspace = ADSB1090Window(
            pluto_source=self._pluto_source,
            owns_pluto_source=False,
            preferences=self._preferences,
        )
        self.bluetooth_workspace = BluetoothAnalyzerWindow(
            pluto_source=self._pluto_source,
            preferences=self._preferences,
        )
        self.dect_workspace = DectAnalyzerWindow(
            pluto_source=self._pluto_source,
            preferences=self._preferences,
        )
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
            workspace.menuBar().hide()
            if hasattr(workspace, "open_config_action"):
                workspace.open_config_action.setEnabled(False)
                workspace.open_config_action.setShortcut(QtGui.QKeySequence())
            for dock in workspace.findChildren(QtWidgets.QDockWidget):
                dock.show()
                dock.setFeatures(
                    dock.features()
                    & ~QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
                )
        if hasattr(self.adsb1090_workspace, "capture_toolbar"):
            self.adsb1090_workspace.capture_toolbar.hide()
        self.generic_workspace.pluto_uri_edit.currentTextChanged.connect(
            self._pluto_target_changed
        )
        self.control_panel.mode_requested.connect(self.set_analysis_mode)
        self.control_panel.preset_requested.connect(self._apply_default_preset)
        self.control_panel.device_requested.connect(self._show_device_dialog)
        self.control_panel.recall_requested.connect(self._recall_meas_config)
        self.control_panel.save_requested.connect(self._save_meas_config)
        self._pluto_target_changed(self.generic_workspace._selected_pluto_target())
        self.resize(1600, 960)
        self.set_analysis_mode("generic")

    @staticmethod
    def _command(
        label: str,
        callback: Callable[[], None],
        action: QtGui.QAction | None = None,
    ) -> PanelCommand:
        return PanelCommand(label=label, callback=callback, action=action)

    def _panel_spec(self, mode: str, workspace: QtWidgets.QWidget) -> WorkspacePanelSpec:
        command = self._command
        if mode == "generic":
            setup = tuple(
                command(label, lambda page=page: workspace.open_config_page(page))
                for label, page in (
                    ("Signal Description", "Signal Description"),
                    ("Input / Frontend", "Input / Frontend"),
                    ("Signal Capture", "Signal Capture"),
                    ("Trigger", "Trigger"),
                    ("Pattern Search", "Pattern Search"),
                    ("Result Range", "Result Range"),
                    ("Demodulation", "Demodulation"),
                    ("Result Summary", "Result Summary"),
                    ("Display", "Display"),
                )
            )
            files = (
                command("Import IQ", workspace._open_iq, workspace.open_iq_action),
                command("Export IQ", workspace._export_iq_recording, workspace.export_iq_action),
                command(
                    "Export Symbol Table",
                    workspace._export_symbol_table,
                    workspace.export_symbol_table_action,
                ),
            )
            return WorkspacePanelSpec(
                mode,
                "Generic VSA",
                setup,
                command("Single", workspace._run_pluto_single, workspace.run_single_action),
                command(
                    "Continuous",
                    workspace._toggle_pluto_continuous,
                    workspace.run_continuous_action,
                ),
                command("Refresh Analysis", workspace._request_analysis, workspace.refresh_analysis_action),
                command("Reset", lambda: self._reset_workspace("generic", workspace), workspace.reset_all_packets_action),
                files,
            )
        if mode == "bluetooth":
            setup = tuple(
                command(label, lambda page=page: workspace.open_config_page(page))
                for label, page in (
                    ("Signal Description", "Signal Description"),
                    ("Input / Frontend", "Input / Frontend"),
                    ("Signal Capture", "Signal Capture"),
                    ("Trigger", "Trigger"),
                    ("Display", "Display"),
                )
            )
            files = (
                command("Import IQ", workspace._open_iq, workspace.open_iq_action),
                command("Export IQ", workspace._export_iq_recording, workspace.export_iq_action),
                command("Export VSG Project", workspace._export_vsg_project, workspace.export_vsg_action),
            )
            return WorkspacePanelSpec(
                mode,
                "Bluetooth",
                setup,
                command("Single", workspace._toggle_capture, workspace.run_action),
                command("Continuous", workspace._toggle_continuous_capture, workspace.run_continuous_action),
                command("Refresh Analysis", workspace.refresh, workspace.refresh_analysis_action),
                command("Reset", lambda: self._reset_workspace("bluetooth", workspace), workspace.clear_measurement_history_action),
                files,
            )
        if mode == "dect":
            setup = tuple(
                command(label, lambda page=page: workspace.open_config_page(page))
                for label, page in (
                    ("Signal Description", "Signal Description"),
                    ("Input / Frontend", "Input / Frontend"),
                    ("Signal Capture", "Signal Capture"),
                    ("Trigger", "Trigger"),
                    ("Display", "Display"),
                )
            )
            files = (
                command("Import IQ", workspace._open_iq, workspace.open_iq_action),
                command("Export IQ", workspace._export_iq_recording, workspace.export_iq_action),
                command("Export VSG Project", workspace._export_vsg_project, workspace.export_vsg_action),
            )
            return WorkspacePanelSpec(
                mode,
                "DECT",
                setup,
                command("Single", workspace._toggle_capture, workspace.run_action),
                command("Continuous", workspace._toggle_continuous_capture, workspace.run_continuous_action),
                command("Refresh Analysis", workspace.refresh, workspace.refresh_analysis_action),
                command("Reset", lambda: self._reset_workspace("dect", workspace), workspace.clear_measurement_history_action),
                files,
            )
        setup = (
            command("Signal Description", workspace.open_analysis_settings),
            command("Input / Frontend", workspace.open_frontend_settings),
            command("Signal Capture", workspace.open_capture_settings),
            command("Trigger", workspace.open_trigger_settings),
        )
        files = (
            command("Import IQ", workspace._open_iq, workspace.open_iq_action),
            command("Export IQ", workspace._export_iq_recording, workspace.export_iq_action),
            command("Export Packet List", workspace._export_packet_list, workspace.export_packet_list_action),
            command(
                "Import OpenSky CSV",
                workspace._import_aircraft_database,
                workspace.import_aircraft_database_action,
            ),
            command(
                "Update Database",
                workspace._download_aircraft_database,
                workspace.update_aircraft_database_action,
            ),
        )
        return WorkspacePanelSpec(
            mode,
            "ADS-B 1090ES",
            setup,
            command("Single", workspace._run_pluto_single, workspace.run_single_action),
            command("Continuous", workspace._run_pluto_continuous, workspace.run_continuous_action),
            command("Refresh Analysis", workspace._refresh, workspace.refresh_analysis_action),
            command("Reset", lambda: self._reset_workspace("adsb1090", workspace), workspace.clear_measurement_history_action),
            files,
        )

    def _reset_workspace(self, mode: str, workspace) -> None:
        """Clear all acquired products while retaining mode settings/device."""
        if self._busy_reason() is not None:
            return
        if mode == "generic":
            workspace._reset_all_packet_statistics()
            previous = workspace.session
            workspace.session = VSASession(
                name=previous.name, signal=previous.signal, settings=previous.settings,
                pattern_search=previous.pattern_search, iq_power_trigger=previous.iq_power_trigger,
                result_range=previous.result_range, demodulation=previous.demodulation,
            )
            workspace._result_summary_values.clear()
            workspace._selected_symbol_marker_index = None
            workspace._selected_match_index = 1
            workspace._pending_analysis = None
            workspace._analysis_generation += 1
            workspace._continuous_sweep_count = 0
        elif mode == "adsb1090":
            workspace._stream_display_timer.stop()
            workspace._pending_stream_views.clear()
            workspace._clear_packet_history()
            workspace._analyzer = type(workspace._analyzer)()
            workspace.recording = None
            workspace.result = None
            workspace._reset_stream_state(float(workspace.sample_rate_combo.currentData()) * 1e6)
        else:
            workspace._reset_measurement_statistics()
            workspace._result = None
            workspace._results = ()
            workspace._recording = None
            workspace._capture_recording = None
            workspace._session = None
            workspace._selected_result_index = 0
            workspace._continuous_capture_count = 0
        import pyqtgraph as pg
        for plot in workspace.findChildren(pg.PlotWidget):
            plot.clear()
        for name in ("symbol_table", "result_summary", "summary_table", "packet_table", "issues_table"):
            widget = getattr(workspace, name, None)
            if widget is not None:
                widget.setRowCount(0)
        for name in ("decode_tree", "payload_text", "payload_hex_text", "air_bits_text"):
            widget = getattr(workspace, name, None)
            if widget is not None:
                widget.clear()
        for name in ("export_iq_action", "export_symbol_table_action", "refresh_analysis_action",
                     "previous_result_action", "next_result_action", "export_modulation_action",
                     "export_power_action", "export_vsg_action"):
            action = getattr(workspace, name, None)
            if action is not None:
                action.setEnabled(False)
        workspace.statusBar().showMessage("All results, history and plots cleared; settings retained")

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
            self.control_panel.set_workspace(self._panel_spec(str(mode), target))
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
        self.control_panel.set_workspace(self._panel_spec(str(mode), target))

    def _active_mode(self) -> str:
        current = self._stack.currentWidget()
        if current is self.bluetooth_workspace:
            return "bluetooth"
        if current is self.dect_workspace:
            return "dect"
        if current is self.adsb1090_workspace:
            return "adsb1090"
        return "generic"

    def _active_workspace(self) -> QtWidgets.QWidget:
        workspace = self._stack.currentWidget()
        if workspace is None:
            raise RuntimeError("no active VSA workspace")
        return workspace

    def _collect_meas_config(self, mode: str) -> dict[str, object]:
        workspace = {
            "generic": self.generic_workspace,
            "bluetooth": self.bluetooth_workspace,
            "dect": self.dect_workspace,
            "adsb1090": self.adsb1090_workspace,
        }[mode]
        if mode == "dect":
            return workspace._config_values()
        return workspace._meas_config_values()

    def _apply_meas_config(self, mode: str, settings: dict[str, object]) -> None:
        workspace = {
            "generic": self.generic_workspace,
            "bluetooth": self.bluetooth_workspace,
            "dect": self.dect_workspace,
            "adsb1090": self.adsb1090_workspace,
        }[mode]
        if mode == "dect":
            workspace._apply_config_values(settings)
        else:
            workspace._apply_meas_config_values(settings)

    def _apply_default_preset(self) -> None:
        if self._busy_reason() is not None:
            return
        mode = self._active_mode()
        answer = QtWidgets.QMessageBox.question(
            self,
            "Default Preset",
            f"Restore the {self.control_panel._spec.mode_label} Default settings?",
            QtWidgets.QMessageBox.StandardButton.Ok | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Cancel,
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Ok:
            return
        workspace = self._active_workspace()
        defaults = deepcopy(workspace._default_meas_config)
        self._apply_meas_config(mode, defaults)
        self.statusBar().showMessage(f"{self.control_panel._spec.mode_label} Default applied")
        self.control_panel.show_main_menu()

    def _config_directory(self) -> str:
        stored = self._preferences.value("directories/config", "", type=str)
        return stored if stored and Path(stored).is_dir() else str(Path.cwd())

    def _remember_config_directory(self, path: str) -> None:
        self._preferences.setValue("directories/config", str(Path(path).parent))

    def _save_meas_config(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Measurement Configuration",
            self._config_directory(),
            "VSA configuration (*.vsaconfig.json);;JSON files (*.json)",
        )
        if not path:
            return
        if not path.lower().endswith(".vsaconfig.json"):
            path += ".vsaconfig.json"
        try:
            self.save_meas_config_path(path)
        except ValueError as error:
            QtWidgets.QMessageBox.critical(self, "Config Save Error", str(error))
            return
        self._remember_config_directory(path)
        self.statusBar().showMessage(f"Configuration saved - {Path(path).name}")

    def save_meas_config_path(self, path: str | Path) -> None:
        mode = self._active_mode()
        save_mode_meas_config(
            path,
            analysis_mode=mode,
            settings=self._collect_meas_config(mode),
        )

    def _recall_meas_config(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Recall Measurement Configuration",
            self._config_directory(),
            "VSA configuration (*.vsaconfig.json *.json);;All files (*)",
        )
        if not path:
            return
        try:
            if not self.recall_meas_config_path(path):
                return
        except (KeyError, TypeError, ValueError) as error:
            QtWidgets.QMessageBox.critical(self, "Config Recall Error", str(error))
            return
        self._remember_config_directory(path)
        self.control_panel.show_main_menu()
        self.statusBar().showMessage(f"Configuration recalled - {Path(path).name}")

    def recall_meas_config_path(self, path: str | Path) -> bool:
        mode, settings = load_mode_meas_config(path)
        previous_mode = self._active_mode()
        self.set_analysis_mode(mode)
        if self._active_mode() != mode:
            return False
        previous_settings = deepcopy(self._collect_meas_config(mode))
        try:
            self._apply_meas_config(mode, settings)
        except (KeyError, TypeError, ValueError):
            self._apply_meas_config(mode, previous_settings)
            if previous_mode != mode:
                self.set_analysis_mode(previous_mode)
            raise
        self.control_panel.show_main_menu()
        return True

    def _show_device_dialog(self) -> None:
        busy = self._busy_reason()
        if busy is not None:
            QtWidgets.QMessageBox.information(
                self, "Device", f"{busy}. Stop it before changing devices."
            )
            return
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("ADALM-Pluto Device")
        layout = QtWidgets.QVBoxLayout(dialog)
        form = QtWidgets.QFormLayout()
        target_combo = QtWidgets.QComboBox()
        target_combo.setEditable(True)

        def sync_targets() -> None:
            selected = self.generic_workspace._selected_pluto_target()
            target_combo.blockSignals(True)
            target_combo.clear()
            for index in range(self.generic_workspace.pluto_uri_edit.count()):
                target_combo.addItem(
                    self.generic_workspace.pluto_uri_edit.itemText(index),
                    self.generic_workspace.pluto_uri_edit.itemData(index),
                )
            selected_index = target_combo.findData(selected)
            if selected_index >= 0:
                target_combo.setCurrentIndex(selected_index)
            else:
                target_combo.setCurrentText(selected)
            target_combo.blockSignals(False)

        sync_targets()
        form.addRow("Connection URI", target_combo)
        layout.addLayout(form)
        refresh = QtWidgets.QPushButton("Refresh Devices")
        refresh_timer = QtCore.QTimer(dialog)
        refresh_timer.setInterval(150)

        def wait_for_refresh() -> None:
            if self.generic_workspace._pluto_discovery_thread is not None:
                return
            refresh_timer.stop()
            sync_targets()
            refresh.setEnabled(True)

        refresh_timer.timeout.connect(wait_for_refresh)

        def refresh_devices() -> None:
            refresh.setEnabled(False)
            self.generic_workspace._refresh_pluto_devices()
            refresh_timer.start()

        refresh.clicked.connect(refresh_devices)
        layout.addWidget(refresh)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        selected_data = target_combo.currentData()
        selected_target = (
            target_combo.currentText()
            if selected_data is None
            else str(selected_data)
        )
        self.generic_workspace._set_selected_pluto_target(selected_target)
        self._pluto_target_changed(selected_target)

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
