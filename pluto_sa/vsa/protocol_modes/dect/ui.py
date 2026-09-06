"""Classic DECT dedicated transmitter-analysis workspace."""

from __future__ import annotations

from dataclasses import replace
import csv
import json
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from pluto_protocol.dect.common import dect_p_range
from pluto_protocol.model import PacketField
from pluto_sa.config.input_frontend import InputPowerCorrection
from pluto_sa.sdr.trigger import TriggerKind, TriggerSlope
from pluto_sa.vsa.analysis import capture_power_traces, recording_spectrum_trace
from pluto_sa.vsa.model import IQRecording
from pluto_sa.vsa.channel import (
    AnalysisDisplayRecordings,
    extract_requested_analysis_channel,
    validate_analysis_channel_capture,
)
from pluto_sa.vsa.pluto_source import PlutoCaptureSettings, PlutoLiveSource
from pluto_sa.vsa.session import VSASession
from pluto_sa.vsa.sources import FileIQSource
from pluto_sa.vsa.ui.capture_thread import PlutoSingleCaptureThread
from pluto_sa.vsa.ui.measurement_chrome import (
    DedicatedPacketAnalysisTree,
    DedicatedSummaryTable,
    SymbolDensitySpread,
    add_fsk_symbol_plot_menu,
    add_result_range_overlay,
    add_symbol_density_menu,
    apply_dedicated_table_style,
    configure_iq_power_plot,
    dedicated_status_color,
    install_measurement_plot_menu,
    limit_iq_power_display_dbm,
    make_analysis_bandwidth_display_controls,
    make_measurement_dock,
    make_measurement_plot,
    packet_time_view_range_ms,
    plot_complex_symbol_distribution,
    plot_frequency_symbol_distribution,
    plot_trace_symbol_points,
    plot_unit_circle,
    set_frequency_constellation_x_lock,
    set_iq_plane_range,
    view_all_traces,
)
from pluto_sa.vsa.ui.measurement_config_dialog import HierarchicalMeasConfigDialog
from pluto_sa.vsa.ui.iq_export import export_iq_recording
from pluto_sa.vsa.ui.display_processing import (
    FSKDisplayData,
    build_fsk_display_data,
    fit_binary_fsk_display_drift,
)

from .analysis import (
    DectPacketResult,
    DectSummaryRow,
    analyze_dect_recording,
    carrier_repetition_count,
)
from .carriers import DECT_CARRIER_PLANS, DectCarrierPlan
from .generator import DECT_SYMBOL_RATE_HZ
from .modulation import DectModulationReference


_CONFIG_KEY = "dect_dedicated/startup_meas_config"
_CONFIG_SCHEMA = "pluto-vsa-dect-dedicated-config"
_CONFIG_VERSION = 1
_DECT_FSK_MODULATION_Y_LIMIT_KHZ = 500.0


class _DectAnalysisThread(QtCore.QThread):
    analysis_ready = QtCore.Signal(object)
    analysis_failed = QtCore.Signal(str)

    def __init__(self, recording: IQRecording, nominal_frequency_hz: float, parent=None):
        super().__init__(parent)
        self._recording = recording
        self._nominal_frequency_hz = nominal_frequency_hz

    def run(self) -> None:
        try:
            results = analyze_dect_recording(
                self._recording,
                nominal_frequency_hz=self._nominal_frequency_hz,
            )
            if not self.isInterruptionRequested():
                self.analysis_ready.emit(results)
        except Exception as error:
            self.analysis_failed.emit(str(error))


class _SummaryTable(DedicatedSummaryTable):
    pass


class DectAnalyzerWindow(QtWidgets.QMainWindow):
    """Six-pane DECT analyzer with an independent measurement configuration."""

    analysis_mode_requested = QtCore.Signal(str)
    application_close_requested = QtCore.Signal()
    shutdown_ready = QtCore.Signal()

    def __init__(
        self,
        pluto_source: PlutoLiveSource | None = None,
        preferences: QtCore.QSettings | None = None,
    ) -> None:
        super().__init__()
        self._pluto_source = pluto_source or PlutoLiveSource()
        self._owns_pluto_source = pluto_source is None
        self._preferences = preferences or QtCore.QSettings("PlutoSA", "PlutoVSA-DECT")
        self._pluto_target = ""
        self._recording: IQRecording | None = None
        self._capture_recording: IQRecording | None = None
        self._recording_revision = 0
        self._results: tuple[DectPacketResult, ...] = ()
        self._result: DectPacketResult | None = None
        self._selected_result_index = 0
        self._show_symbol_points = True
        self._symbol_density = False
        self._symbol_density_spread = SymbolDensitySpread.MAXIMUM
        self._fsk_symbol_plot_mode = "Constellation Frequency"
        self._modulation_reference = DectModulationReference.MEASURED
        self._carrier_history: dict[tuple[float, str, str], list[float]] = {}
        self._accumulated_packet_tokens: set[tuple[int, int, int]] = set()
        self._capture_thread: PlutoSingleCaptureThread | None = None
        self._analysis_thread: _DectAnalysisThread | None = None
        self._analysis_plot_ranges: dict[
            str, tuple[list[float], list[float]]
        ] = {}
        self._plot_context_actions: dict[str, dict[str, QtGui.QAction]] = {}
        self._shutdown_requested = False
        self.setDockOptions(
            QtWidgets.QMainWindow.DockOption.AllowNestedDocks
            | QtWidgets.QMainWindow.DockOption.AllowTabbedDocks
        )
        self._build_menu()
        self._build_controls()
        self._build_config_dialog()
        self._build_results()
        restored = self._restore_config()
        self.statusBar().showMessage(
            "Ready - DECT configuration restored"
            if restored
            else "Ready - select a regional carrier and capture a DECT burst"
        )

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        open_action = file_menu.addAction("Open IQ...")
        open_action.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._open_iq)
        self.export_iq_action = file_menu.addAction("Export IQ Recording...")
        self.export_iq_action.setEnabled(False)
        self.export_iq_action.triggered.connect(self._export_iq_recording)
        self.export_modulation_action = file_menu.addAction(
            "Export GFSK Modulation Debug CSV..."
        )
        self.export_modulation_action.setEnabled(False)
        self.export_modulation_action.triggered.connect(
            self._export_modulation_debug_csv
        )
        file_menu.addSeparator()
        file_menu.addAction("Close").triggered.connect(
            self.application_close_requested.emit
        )

        run_menu = self.menuBar().addMenu("Sweep / Run")
        self.run_action = run_menu.addAction("Run Single")
        self.run_action.setShortcut(QtGui.QKeySequence("F6"))
        self.run_action.triggered.connect(self._toggle_capture)
        refresh = run_menu.addAction("Refresh Analysis")
        refresh.setShortcut(QtGui.QKeySequence("F5"))
        refresh.triggered.connect(self.refresh)
        run_menu.addSeparator()
        run_menu.addAction("Previous Packet").triggered.connect(
            lambda: self._select_result(-1)
        )
        run_menu.addAction("Next Packet").triggered.connect(
            lambda: self._select_result(1)
        )
        run_menu.addSeparator()
        run_menu.addAction("Reset Measurement Statistics").triggered.connect(
            self._reset_measurement_statistics
        )

        display_menu = self.menuBar().addMenu("Display Config")
        self.symbols_action = display_menu.addAction("Show Symbol Points")
        self.symbols_action.setCheckable(True)
        self.symbols_action.setChecked(self._show_symbol_points)
        self.symbols_action.setShortcut(QtGui.QKeySequence("S"))
        self.symbols_action.toggled.connect(self._set_show_symbol_points)
        (
            self.density_action,
            self.density_spread_group,
            self.density_spread_actions,
        ) = add_symbol_density_menu(
            display_menu,
            self,
            enabled=self._symbol_density,
            spread=self._symbol_density_spread,
            on_enabled=self._set_symbol_density,
            on_spread=self._set_symbol_density_spread,
        )
        (
            self.fsk_frequency_action,
            self.fsk_phase_action,
            self.fsk_plot_group,
        ) = add_fsk_symbol_plot_menu(
            display_menu,
            self,
            mode=self._fsk_symbol_plot_mode,
            on_mode=self._set_fsk_symbol_plot_mode,
        )
        reference_menu = display_menu.addMenu("GFSK Modulation Reference")
        self.modulation_reference_group = QtGui.QActionGroup(self)
        self.modulation_reference_group.setExclusive(True)
        self.modulation_reference_actions: dict[
            DectModulationReference, QtGui.QAction
        ] = {}
        for reference in DectModulationReference:
            action = reference_menu.addAction(reference.value)
            action.setCheckable(True)
            action.setChecked(reference is self._modulation_reference)
            action.triggered.connect(
                lambda _checked=False, selected=reference: (
                    self._set_modulation_reference(selected)
                )
            )
            self.modulation_reference_group.addAction(action)
            self.modulation_reference_actions[reference] = action
        reset = display_menu.addAction("Reset Plot Scales")
        reset.setShortcut(QtGui.QKeySequence("Home"))
        reset.triggered.connect(self._reset_plot_scales)

        config_menu = self.menuBar().addMenu("Meas Config")
        config = config_menu.addAction("Open Meas Config...")
        config.setShortcut(QtGui.QKeySequence("Ctrl+M"))
        config.triggered.connect(self._show_config)

        mode_menu = self.menuBar().addMenu("Analysis Mode")
        mode_menu.addAction("Generic FSK / PSK VSA...").triggered.connect(
            lambda: self.analysis_mode_requested.emit("generic")
        )
        mode_menu.addAction("Bluetooth Dedicated Analyzer...").triggered.connect(
            lambda: self.analysis_mode_requested.emit("bluetooth")
        )
        current = mode_menu.addAction("DECT Dedicated Analyzer")
        current.setCheckable(True)
        current.setChecked(True)
        current.setEnabled(False)
        mode_menu.addAction("ADS-B 1090ES...").triggered.connect(
            lambda: self.analysis_mode_requested.emit("adsb1090")
        )

    def _build_controls(self) -> None:
        self.plan_combo = QtWidgets.QComboBox()
        for plan in DECT_CARRIER_PLANS:
            self.plan_combo.addItem(plan.label, plan.plan_id)
        self.carrier_combo = QtWidgets.QComboBox()
        self.capture_length_spin = QtWidgets.QDoubleSpinBox()
        self.capture_length_spin.setRange(1.0, 100.0)
        self.capture_length_spin.setValue(2.0)
        self.capture_length_spin.setSuffix(" ms")
        self.oversampling_combo = QtWidgets.QComboBox()
        for sps in (4, 8, 16, 32):
            self.oversampling_combo.addItem(f"{sps} S/sym", sps)
        self.oversampling_combo.setCurrentIndex(1)
        self.rf_bandwidth_spin = QtWidgets.QDoubleSpinBox()
        self.rf_bandwidth_spin.setRange(3.0, 20.0)
        self.rf_bandwidth_spin.setValue(6.0)
        self.rf_bandwidth_spin.setSuffix(" MHz")
        self.channel_filter_check = QtWidgets.QCheckBox("Enable Analysis Channel")
        self.analysis_bandwidth_spin = QtWidgets.QDoubleSpinBox()
        self.analysis_bandwidth_spin.setRange(0.000001, 100.0)
        self.analysis_bandwidth_spin.setDecimals(6)
        self.analysis_bandwidth_spin.setValue(3.0)
        self.analysis_bandwidth_spin.setSuffix(" MHz")
        (
            self.analysis_power_display_check,
            self.analysis_spectrum_display_check,
        ) = make_analysis_bandwidth_display_controls()
        self.lo_offset_check = QtWidgets.QCheckBox("Enable")
        self.lo_offset_check.setToolTip(
            "Tune the Pluto LO away from the selected DECT carrier. "
            "Requires the Analysis Channel filter."
        )
        self.lo_offset_spin = QtWidgets.QDoubleSpinBox()
        self.lo_offset_spin.setRange(-50.0, 50.0)
        self.lo_offset_spin.setDecimals(6)
        self.lo_offset_spin.setValue(2.0)
        self.lo_offset_spin.setSuffix(" MHz")
        self.resolved_lo_label = QtWidgets.QLabel()
        self.internal_gain_spin = QtWidgets.QDoubleSpinBox()
        self.internal_gain_spin.setRange(0.0, 70.0)
        self.internal_gain_spin.setValue(30.0)
        self.internal_gain_spin.setSuffix(" dB")
        self.external_att_spin = QtWidgets.QDoubleSpinBox()
        self.external_att_spin.setRange(-100.0, 100.0)
        self.external_att_spin.setValue(30.0)
        self.external_att_spin.setSuffix(" dB")
        self.trigger_level_spin = QtWidgets.QDoubleSpinBox()
        self.trigger_level_spin.setRange(-150.0, 30.0)
        self.trigger_level_spin.setValue(-25.0)
        self.trigger_level_spin.setSuffix(" dBm")
        self.device_label = QtWidgets.QLabel("Pluto: Auto")
        self.capture_button = QtWidgets.QPushButton("Single Capture")
        self.refresh_button = QtWidgets.QPushButton("Refresh Result")
        self.plan_combo.currentIndexChanged.connect(self._plan_changed)
        self.carrier_combo.currentIndexChanged.connect(
            self._sync_analysis_channel_controls
        )
        self.channel_filter_check.toggled.connect(
            self._sync_analysis_channel_controls
        )
        self.lo_offset_check.toggled.connect(self._sync_analysis_channel_controls)
        self.lo_offset_spin.valueChanged.connect(self._sync_analysis_channel_controls)
        self.analysis_power_display_check.toggled.connect(
            self._display_source_changed
        )
        self.analysis_spectrum_display_check.toggled.connect(
            self._display_source_changed
        )
        self.capture_button.clicked.connect(self._toggle_capture)
        self.refresh_button.clicked.connect(self.refresh)
        self._plan_changed()
        self._sync_analysis_channel_controls()

    def _sync_analysis_channel_controls(self, _value: object = None) -> None:
        filter_enabled = self.channel_filter_check.isChecked()
        if self.sender() is self.lo_offset_check and self.lo_offset_check.isChecked():
            self.channel_filter_check.setChecked(True)
            filter_enabled = True
        elif not filter_enabled and self.lo_offset_check.isChecked():
            self.lo_offset_check.setChecked(False)
        self.analysis_bandwidth_spin.setEnabled(filter_enabled)
        self.analysis_power_display_check.setEnabled(filter_enabled)
        self.analysis_spectrum_display_check.setEnabled(filter_enabled)
        offset_enabled = self.lo_offset_check.isChecked()
        self.lo_offset_spin.setEnabled(offset_enabled)
        offset_mhz = self.lo_offset_spin.value() if offset_enabled else 0.0
        center_mhz = (
            self._nominal_frequency_hz() / 1e6
            if self.carrier_combo.currentData() is not None
            else 0.0
        )
        self.resolved_lo_label.setText(
            f"{center_mhz + offset_mhz:.6f} MHz"
            + (" (offset on)" if offset_enabled else " (offset off)")
        )

    @QtCore.Slot(bool)
    def _display_source_changed(self, _enabled: bool) -> None:
        if self._result is not None:
            self._render(self._result)

    @QtCore.Slot(bool)
    def _set_show_symbol_points(self, enabled: bool) -> None:
        self._show_symbol_points = bool(enabled)
        self.symbols_action.blockSignals(True)
        self.symbols_action.setChecked(self._show_symbol_points)
        self.symbols_action.blockSignals(False)
        if hasattr(self, "config_show_symbols"):
            self.config_show_symbols.blockSignals(True)
            self.config_show_symbols.setChecked(self._show_symbol_points)
            self.config_show_symbols.blockSignals(False)
        if self._result is not None:
            self._render(self._result)

    @QtCore.Slot(bool)
    def _set_symbol_density(self, enabled: bool) -> None:
        self._symbol_density = bool(enabled)
        self.density_action.blockSignals(True)
        self.density_action.setChecked(self._symbol_density)
        self.density_action.blockSignals(False)
        if hasattr(self, "config_density"):
            self.config_density.blockSignals(True)
            self.config_density.setChecked(self._symbol_density)
            self.config_density.blockSignals(False)
        if self._result is not None:
            self._render(self._result)

    @QtCore.Slot(str)
    def _set_symbol_density_spread(
        self, spread: SymbolDensitySpread | str
    ) -> None:
        try:
            resolved = SymbolDensitySpread(spread)
        except ValueError:
            return
        self._symbol_density_spread = resolved
        self.density_spread_actions[resolved].setChecked(True)
        if hasattr(self, "config_density_spread"):
            self.config_density_spread.blockSignals(True)
            self.config_density_spread.setCurrentText(resolved.value)
            self.config_density_spread.blockSignals(False)
        if self._result is not None:
            self._render(self._result)

    @QtCore.Slot(str)
    def _set_modulation_reference(
        self, reference: DectModulationReference | str
    ) -> None:
        try:
            selected = DectModulationReference(reference)
        except ValueError:
            selected = DectModulationReference.MEASURED
        self._modulation_reference = selected
        if hasattr(self, "modulation_reference_actions"):
            self.modulation_reference_actions[selected].setChecked(True)
        if self._result is not None:
            self._render(self._result)

    @QtCore.Slot(str)
    def _set_fsk_symbol_plot_mode(self, mode: str) -> None:
        if mode not in {"Constellation Frequency", "Phase Difference"}:
            return
        self._fsk_symbol_plot_mode = mode
        self.fsk_frequency_action.setChecked(mode == "Constellation Frequency")
        self.fsk_phase_action.setChecked(mode == "Phase Difference")
        if hasattr(self, "config_fsk_mode"):
            self.config_fsk_mode.blockSignals(True)
            self.config_fsk_mode.setCurrentText(mode)
            self.config_fsk_mode.blockSignals(False)
        if hasattr(self, "symbol_plot"):
            set_frequency_constellation_x_lock(
                self.symbol_plot, mode == "Constellation Frequency"
            )
        if self._result is not None:
            self._render(self._result)

    def _build_config_dialog(self) -> None:
        dect_page = QtWidgets.QWidget()
        dect_form = QtWidgets.QFormLayout(dect_page)
        dect_form.addRow("Regional Carrier Plan", self.plan_combo)
        dect_form.addRow("RF Carrier", self.carrier_combo)
        dect_form.addRow("Modulation", QtWidgets.QLabel("GFSK, BT = 0.5"))
        dect_form.addRow("Symbol Rate", QtWidgets.QLabel("1.152 MSym/s"))

        input_page = QtWidgets.QWidget()
        input_form = QtWidgets.QFormLayout(input_page)
        for label, widget in (
            ("Capture Length", self.capture_length_spin),
            ("Samples / Symbol", self.oversampling_combo),
            ("RF Bandwidth", self.rf_bandwidth_spin),
            ("Analysis Channel", self.channel_filter_check),
            ("Analysis Bandwidth", self.analysis_bandwidth_spin),
            (
                "Apply Analysis Bandwidth to Power",
                self.analysis_power_display_check,
            ),
            (
                "Apply Analysis Bandwidth to Spectrum",
                self.analysis_spectrum_display_check,
            ),
            ("LO Offset", self.lo_offset_check),
            ("Offset Frequency", self.lo_offset_spin),
            ("Resolved LO", self.resolved_lo_label),
            ("Internal Gain", self.internal_gain_spin),
            ("External ATT", self.external_att_spin),
            ("I/Q Power Trigger", self.trigger_level_spin),
            ("Input Device", self.device_label),
        ):
            input_form.addRow(label, widget)

        run_page = QtWidgets.QWidget()
        run_layout = QtWidgets.QVBoxLayout(run_page)
        run_layout.addWidget(self.capture_button)
        run_layout.addWidget(self.refresh_button)
        run_layout.addStretch(1)

        display_page = QtWidgets.QWidget()
        display_layout = QtWidgets.QVBoxLayout(display_page)
        self.config_show_symbols = QtWidgets.QCheckBox("Show Symbol Points")
        self.config_show_symbols.setChecked(self._show_symbol_points)
        self.config_show_symbols.toggled.connect(self._set_show_symbol_points)
        self.config_density = QtWidgets.QCheckBox("Symbol Plot Density")
        self.config_density.setChecked(self._symbol_density)
        self.config_density.toggled.connect(self._set_symbol_density)
        self.config_density_spread = QtWidgets.QComboBox()
        self.config_density_spread.addItems(
            tuple(spread.value for spread in SymbolDensitySpread)
        )
        self.config_density_spread.setCurrentText(self._symbol_density_spread.value)
        self.config_density_spread.currentTextChanged.connect(
            self._set_symbol_density_spread
        )
        self.config_fsk_mode = QtWidgets.QComboBox()
        self.config_fsk_mode.addItems(
            ("Constellation Frequency", "Phase Difference")
        )
        self.config_fsk_mode.setCurrentText(self._fsk_symbol_plot_mode)
        self.config_fsk_mode.currentTextChanged.connect(
            self._set_fsk_symbol_plot_mode
        )
        display_layout.addWidget(self.config_show_symbols)
        display_layout.addWidget(self.config_density)
        display_layout.addWidget(QtWidgets.QLabel("Density Spread (all modulations)"))
        display_layout.addWidget(self.config_density_spread)
        display_layout.addWidget(QtWidgets.QLabel("FSK Symbol Plot"))
        display_layout.addWidget(self.config_fsk_mode)
        display_layout.addStretch(1)

        self._config_dialog = HierarchicalMeasConfigDialog(
            self,
            (
                ("DECT Analysis", dect_page),
                ("Input / Frontend", input_page),
                ("Display Config", display_page),
                ("Sweep / Run", run_page),
            ),
            window_title="DECT Meas Config",
            size=(760, 520),
            standard_buttons=(
                QtWidgets.QDialogButtonBox.StandardButton.Ok
                | QtWidgets.QDialogButtonBox.StandardButton.Cancel
            ),
        )
        self._config_dialog.accepted.connect(self._save_config)

    def _dock(self, title: str, widget: QtWidgets.QWidget) -> QtWidgets.QDockWidget:
        return make_measurement_dock(
            title, widget, self, object_prefix="vsa-dect", closable=False
        )

    def _build_results(self) -> None:
        self.power_plot = make_measurement_plot("IQ Power (dBm)", "Time (ms)")
        configure_iq_power_plot(self.power_plot)
        self.power_dock = self._dock("IQ Power", self.power_plot)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.power_dock)

        self.spectrum_plot = make_measurement_plot("Magnitude (dBm)", "Frequency (MHz)")
        self.spectrum_dock = self._dock("Spectrum", self.spectrum_plot)
        self.splitDockWidget(
            self.power_dock, self.spectrum_dock, QtCore.Qt.Orientation.Horizontal
        )

        self.summary_table = _SummaryTable()
        self.summary_dock = self._dock("Result Summary", self.summary_table)
        self.splitDockWidget(
            self.spectrum_dock, self.summary_dock, QtCore.Qt.Orientation.Horizontal
        )

        self.deviation_plot = make_measurement_plot(
            "Frequency Deviation (kHz)", "Time (ms)"
        )
        self.deviation_dock = self._dock("GFSK Modulation", self.deviation_plot)
        self.splitDockWidget(
            self.power_dock, self.deviation_dock, QtCore.Qt.Orientation.Vertical
        )

        self.symbol_plot = make_measurement_plot("Frequency (kHz)", "Symbol Index")
        self.symbol_dock = self._dock("Symbol Plot", self.symbol_plot)
        self.splitDockWidget(
            self.spectrum_dock, self.symbol_dock, QtCore.Qt.Orientation.Vertical
        )

        self.packet_tabs = QtWidgets.QTabWidget()
        self.decode_tree = DedicatedPacketAnalysisTree(
            ("Field", "Value", "Bit Range", "DECT Symbols", "Status"),
            (125, 115, 68, 92, 55),
            expand_columns=(0, 1),
        )
        self.packet_table = QtWidgets.QTableWidget(0, 5)
        self.packet_table.setHorizontalHeaderLabels(
            ("#", "Direction", "Packet", "Pattern", "Symbols")
        )
        apply_dedicated_table_style(self.packet_table)
        self.packet_table.setTextElideMode(QtCore.Qt.TextElideMode.ElideNone)
        self.packet_table.setWordWrap(True)
        self.packet_table.verticalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        self.packet_table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self.packet_table.cellClicked.connect(
            lambda row, _column: self._select_result_index(row)
        )
        self.air_bits_text = QtWidgets.QPlainTextEdit(readOnly=True)
        self.packet_tabs.addTab(self.decode_tree, "Decode")
        self.packet_tabs.addTab(self.packet_table, "Packet List")
        self.packet_tabs.addTab(self.air_bits_text, "Air Bits")
        self.packet_dock = self._dock("Packet Analysis", self.packet_tabs)
        self.splitDockWidget(
            self.summary_dock, self.packet_dock, QtCore.Qt.Orientation.Vertical
        )
        QtCore.QTimer.singleShot(0, self._equalize_docks)
        for name, plot in self._plot_widgets():
            self._plot_context_actions[name] = install_measurement_plot_menu(
                plot,
                reset=lambda plot_name=name, target=plot: self._reset_plot_scale(
                    plot_name, target
                ),
                view_all=lambda plot_name=name, selected=plot: self._view_all_plot(
                    plot_name, selected
                ),
            )
        set_frequency_constellation_x_lock(self.symbol_plot, True)

    def _plots(self) -> tuple[pg.PlotWidget, ...]:
        return tuple(plot for _name, plot in self._plot_widgets())

    def _plot_widgets(self) -> tuple[tuple[str, pg.PlotWidget], ...]:
        return (
            ("iq_power", self.power_plot),
            ("spectrum", self.spectrum_plot),
            ("gfsk_modulation", self.deviation_plot),
            ("fsk_symbol", self.symbol_plot),
        )

    def _equalize_docks(self) -> None:
        self.resizeDocks(
            [self.power_dock, self.spectrum_dock, self.summary_dock],
            [500, 500, 500],
            QtCore.Qt.Orientation.Horizontal,
        )
        self.resizeDocks(
            [self.deviation_dock, self.symbol_dock, self.packet_dock],
            [500, 500, 500],
            QtCore.Qt.Orientation.Horizontal,
        )

    def _plan_changed(self) -> None:
        plan = self._current_plan()
        previous = self.carrier_combo.currentData()
        self.carrier_combo.clear()
        for carrier in plan.carriers:
            self.carrier_combo.addItem(carrier.label, carrier.center_frequency_hz)
        index = self.carrier_combo.findData(previous)
        self.carrier_combo.setCurrentIndex(max(0, index))

    def _current_plan(self) -> DectCarrierPlan:
        plan_id = self.plan_combo.currentData()
        return next(plan for plan in DECT_CARRIER_PLANS if plan.plan_id == plan_id)

    def _nominal_frequency_hz(self) -> float:
        return float(self.carrier_combo.currentData())

    def _config_values(self) -> dict[str, object]:
        return {
            "plan": self.plan_combo.currentData(),
            "carrier_hz": self.carrier_combo.currentData(),
            "capture_ms": self.capture_length_spin.value(),
            "samples_per_symbol": self.oversampling_combo.currentData(),
            "rf_bandwidth_mhz": self.rf_bandwidth_spin.value(),
            "analysis_channel_enabled": self.channel_filter_check.isChecked(),
            "analysis_bandwidth_mhz": self.analysis_bandwidth_spin.value(),
            "apply_analysis_bandwidth_to_power": (
                self.analysis_power_display_check.isChecked()
            ),
            "apply_analysis_bandwidth_to_spectrum": (
                self.analysis_spectrum_display_check.isChecked()
            ),
            "lo_offset_enabled": self.lo_offset_check.isChecked(),
            "lo_offset_mhz": self.lo_offset_spin.value(),
            "internal_gain_db": self.internal_gain_spin.value(),
            "external_att_db": self.external_att_spin.value(),
            "trigger_level_dbm": self.trigger_level_spin.value(),
            "show_symbol_points": self._show_symbol_points,
            "symbol_density": self._symbol_density,
            "symbol_density_spread": self._symbol_density_spread.value,
            "fsk_symbol_plot": self._fsk_symbol_plot_mode,
            "modulation_reference": self._modulation_reference.value,
        }

    def _save_config(self) -> None:
        payload = {
            "schema": _CONFIG_SCHEMA,
            "version": _CONFIG_VERSION,
            "settings": self._config_values(),
        }
        self._preferences.setValue(
            _CONFIG_KEY, json.dumps(payload, separators=(",", ":"))
        )
        self._preferences.sync()

    def _restore_config(self) -> bool:
        raw = self._preferences.value(_CONFIG_KEY, "", type=str)
        if not raw:
            return False
        try:
            payload = json.loads(raw)
            if payload.get("schema") != _CONFIG_SCHEMA:
                return False
            settings = payload["settings"]
            plan_index = self.plan_combo.findData(settings["plan"])
            if plan_index >= 0:
                self.plan_combo.setCurrentIndex(plan_index)
            carrier_index = self.carrier_combo.findData(settings["carrier_hz"])
            if carrier_index >= 0:
                self.carrier_combo.setCurrentIndex(carrier_index)
            self.capture_length_spin.setValue(float(settings["capture_ms"]))
            sps_index = self.oversampling_combo.findData(settings["samples_per_symbol"])
            if sps_index >= 0:
                self.oversampling_combo.setCurrentIndex(sps_index)
            self.rf_bandwidth_spin.setValue(float(settings["rf_bandwidth_mhz"]))
            self.analysis_bandwidth_spin.setValue(
                float(settings.get("analysis_bandwidth_mhz", 3.0))
            )
            self.lo_offset_spin.setValue(float(settings.get("lo_offset_mhz", 2.0)))
            self.channel_filter_check.setChecked(
                bool(settings.get("analysis_channel_enabled", False))
            )
            self.analysis_power_display_check.setChecked(
                bool(settings.get("apply_analysis_bandwidth_to_power", True))
            )
            self.analysis_spectrum_display_check.setChecked(
                bool(settings.get("apply_analysis_bandwidth_to_spectrum", False))
            )
            self.lo_offset_check.setChecked(
                bool(settings.get("lo_offset_enabled", False))
            )
            self.internal_gain_spin.setValue(float(settings["internal_gain_db"]))
            self.external_att_spin.setValue(float(settings["external_att_db"]))
            self.trigger_level_spin.setValue(float(settings["trigger_level_dbm"]))
            self._set_show_symbol_points(bool(settings.get("show_symbol_points", True)))
            self._set_symbol_density(bool(settings.get("symbol_density", False)))
            self._set_symbol_density_spread(
                str(
                    settings.get(
                        "symbol_density_spread",
                        SymbolDensitySpread.MAXIMUM.value,
                    )
                )
            )
            self._set_fsk_symbol_plot_mode(
                str(settings.get("fsk_symbol_plot", "Constellation Frequency"))
            )
            self._set_modulation_reference(
                str(
                    settings.get(
                        "modulation_reference",
                        DectModulationReference.MEASURED.value,
                    )
                )
            )
            self._sync_analysis_channel_controls()
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return False
        return True

    def _show_config(self) -> None:
        self._config_dialog.open_top()

    def set_pluto_target(self, target: str | None) -> None:
        self._pluto_target = str(target or "")
        self.device_label.setText(f"Pluto: {self._pluto_target or 'Auto'}")

    def set_session(self, session: VSASession) -> None:
        self.stage_session(session)
        self.refresh()

    def stage_session(self, session: VSASession) -> None:
        if self._recording is not session.recording:
            self._capture_recording = session.recording
            self._recording = session.recording
            self._recording_revision += 1
        self.export_iq_action.setEnabled(self._recording is not None)
        self.export_modulation_action.setEnabled(False)

    def load_recording(
        self,
        recording: IQRecording,
        *,
        capture_recording: IQRecording | None = None,
    ) -> None:
        self._capture_recording = capture_recording or recording
        self._recording = recording
        self._recording_revision += 1
        self.export_iq_action.setEnabled(True)
        self.export_modulation_action.setEnabled(False)
        self.refresh()

    def _last_directory(self) -> str:
        stored = self._preferences.value("directories/iq", "", type=str)
        return stored if stored and Path(stored).is_dir() else str(Path.cwd())

    def _open_iq(self) -> None:
        if self.shutdown_busy_reason() is not None:
            self.statusBar().showMessage("Stop DECT capture or analysis before opening IQ")
            return
        path, _selected = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open DECT IQ Recording",
            self._last_directory(),
            "IQ recordings (*.iq.tar *.npz *.npy *.cf32 *.bin);;All files (*)",
        )
        if not path:
            return
        try:
            suffix = Path(path).suffix.lower()
            if suffix in {".cf32", ".bin", ".npy"}:
                recording = FileIQSource.load(
                    path,
                    sample_rate_hz=DECT_SYMBOL_RATE_HZ
                    * int(self.oversampling_combo.currentData()),
                    center_frequency_hz=self._nominal_frequency_hz(),
                )
            else:
                recording = FileIQSource.load(path)
            if recording.center_frequency_hz == 0.0:
                recording = replace(
                    recording, center_frequency_hz=self._nominal_frequency_hz()
                )
        except Exception as error:
            QtWidgets.QMessageBox.critical(self, "Open DECT IQ", str(error))
            return
        self._preferences.setValue("directories/iq", str(Path(path).resolve().parent))
        self._preferences.sync()
        self.load_recording(recording)

    def _export_iq_recording(self) -> None:
        export_iq_recording(
            self,
            self._capture_recording or self._recording,
            self._preferences,
        )

    def _export_modulation_debug_csv(self) -> None:
        result = self._result
        recording = self._recording
        if result is None or recording is None:
            return
        path_text, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export DECT GFSK Modulation Debug",
            str(Path(self._last_directory()) / "dect_gfsk_modulation.csv"),
            "CSV files (*.csv);;All files (*)",
        )
        if not path_text:
            return
        path = Path(path_text)
        if path.suffix.lower() != ".csv":
            path = path.with_suffix(".csv")
        reference_hz = result.modulation_reference_hz(self._modulation_reference)
        actual_start = float(
            result.metadata.get("actual_preamble_start_sample", result.p0_sample)
        )
        sps = float(result.metadata["samples_per_symbol"])
        try:
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(
                    (
                        "Trace",
                        "DECT Symbol",
                        "Fraction",
                        "Time [s]",
                        "Frequency [Hz]",
                        "Frequency Deviation [Hz]",
                        "ETSI Eligible",
                        "Reference",
                    )
                )

                def write_trace(  # noqa: PLR0913 - flat CSV trace schema
                    name: str,
                    positions: np.ndarray,
                    values: np.ndarray,
                    eligible=None,
                    symbols: np.ndarray | None = None,
                    fractions: np.ndarray | None = None,
                ) -> None:
                    for index, (sample, value) in enumerate(zip(positions, values)):
                        symbol_time = (float(sample) - result.p0_sample) / sps
                        symbol = (
                            int(symbols[index])
                            if symbols is not None
                            else int(np.floor(symbol_time))
                        )
                        fraction = (
                            f"{int(fractions[index])}/6"
                            if fractions is not None
                            else f"{symbol_time - np.floor(symbol_time):.9f}"
                        )
                        writer.writerow(
                            (
                                name,
                                f"p{symbol}",
                                fraction,
                                f"{float(sample) / recording.sample_rate_hz:.12g}",
                                f"{float(value):.12g}",
                                f"{float(value) - reference_hz:.12g}",
                                (
                                    int(bool(eligible[index]))
                                    if eligible is not None
                                    else ""
                                ),
                                self._modulation_reference.value,
                            )
                        )

                packet = (
                    (result.raw_fm_sample >= actual_start)
                    & (result.raw_fm_sample <= result.packet_end_sample)
                )
                write_trace(
                    "Raw phase-difference frequency",
                    result.raw_fm_sample[packet],
                    result.raw_fm_frequency_hz[packet],
                    result.etsi_eligible_sample_mask[packet],
                )
                write_trace(
                    "Measurement frequency (no filter)",
                    result.measurement_fm_sample[packet],
                    result.measurement_fm_frequency_hz[packet],
                    result.etsi_eligible_sample_mask[packet],
                )
                write_trace(
                    "CTS60-compatible 6 SPS",
                    result.cts60_trace_sample,
                    result.cts60_trace_frequency_hz,
                    symbols=result.cts60_trace_symbol,
                    fractions=result.cts60_trace_fraction,
                )
                write_trace(
                    "Symbol decision frequency",
                    result.symbol_centers,
                    result.symbol_frequency_hz,
                )
                write_trace(
                    "Ideal BT=0.5 diagnostic fit",
                    result.raw_fm_sample[packet],
                    result.ideal_gfsk_frequency_hz[packet],
                )
        except OSError as error:
            QtWidgets.QMessageBox.critical(self, "Export GFSK Modulation", str(error))
            return
        self._preferences.setValue("directories/iq", str(path.resolve().parent))
        self._preferences.sync()
        self.statusBar().showMessage(f"Exported DECT modulation debug: {path}")

    def _capture_settings(self) -> PlutoCaptureSettings:
        return PlutoCaptureSettings(
            center_frequency_hz=self._nominal_frequency_hz(),
            symbol_rate_hz=DECT_SYMBOL_RATE_HZ,
            samples_per_symbol=int(self.oversampling_combo.currentData()),
            capture_length_s=self.capture_length_spin.value() * 1e-3,
            rf_bandwidth_hz=self.rf_bandwidth_spin.value() * 1e6,
            lo_offset_hz=(
                self.lo_offset_spin.value() * 1e6
                if self.lo_offset_check.isChecked()
                else 0.0
            ),
            analysis_bandwidth_hz=(
                self.analysis_bandwidth_spin.value() * 1e6
                if self.channel_filter_check.isChecked()
                else None
            ),
            sdr_uri=self._pluto_target or None,
            power_correction=InputPowerCorrection(
                internal_gain_db=self.internal_gain_spin.value(),
                external_attenuation_db=self.external_att_spin.value(),
            ),
            trigger_source=TriggerKind.POWER_LEVEL,
            trigger_level_dbm=self.trigger_level_spin.value(),
            trigger_slope=TriggerSlope.RISING,
            trigger_hysteresis_db=3.0,
        )

    def _toggle_capture(self) -> None:
        if self._analysis_thread is not None and self._analysis_thread.isRunning():
            self._analysis_thread.requestInterruption()
            self.statusBar().showMessage("Stopping DECT analysis...")
            return
        if self._capture_thread is not None and self._capture_thread.isRunning():
            self._capture_thread.cancel()
            self.statusBar().showMessage("Stopping DECT capture...")
            return
        settings = self._capture_settings()
        try:
            validate_analysis_channel_capture(
                sample_rate_hz=settings.requested_sample_rate_hz,
                usable_bandwidth_hz=settings.nominal_usable_bandwidth_hz,
                lo_offset_hz=settings.lo_offset_hz,
                analysis_bandwidth_hz=settings.analysis_bandwidth_hz,
            )
        except ValueError as error:
            QtWidgets.QMessageBox.critical(self, "DECT Capture", str(error))
            return
        thread = PlutoSingleCaptureThread(
            self._pluto_source,
            settings,
            f"Waiting for DECT burst at {settings.center_frequency_hz / 1e6:.3f} MHz",
            self,
        )
        thread.capture_armed.connect(self.statusBar().showMessage)
        thread.capture_ready.connect(self._capture_ready)
        thread.capture_failed.connect(self._capture_failed)
        thread.capture_cancelled.connect(
            lambda: self.statusBar().showMessage("DECT capture cancelled")
        )
        thread.finished.connect(self._capture_stopped)
        thread.finished.connect(thread.deleteLater)
        self._capture_thread = thread
        self.capture_button.setText("Stop Capture")
        self.run_action.setText("Stop")
        self.statusBar().showMessage("Preparing Pluto for DECT IQ capture...")
        thread.start()

    @QtCore.Slot(object)
    def _capture_ready(self, recording: object) -> None:
        if not isinstance(recording, IQRecording):
            self._capture_failed("capture returned an invalid IQ recording")
            return
        capture_recording = recording
        try:
            recording = extract_requested_analysis_channel(capture_recording)
        except ValueError as error:
            self._capture_failed(str(error))
            return
        self.load_recording(recording, capture_recording=capture_recording)

    @QtCore.Slot(str)
    def _capture_failed(self, message: str) -> None:
        self.statusBar().showMessage(f"DECT capture failed: {message}")
        if not self._shutdown_requested:
            QtWidgets.QMessageBox.critical(self, "DECT Capture", message)

    @QtCore.Slot()
    def _capture_stopped(self) -> None:
        self._capture_thread = None
        stop_stream = getattr(self._pluto_source, "stop_stream", None)
        if callable(stop_stream):
            stop_stream()
        if self._analysis_thread is None or not self._analysis_thread.isRunning():
            self.capture_button.setText("Single Capture")
            self.run_action.setText("Run Single")
        if self._shutdown_requested and self.shutdown_busy_reason() is None:
            self.shutdown_ready.emit()

    @QtCore.Slot()
    def refresh(self) -> None:
        if self._recording is None:
            self.statusBar().showMessage("Capture or open a DECT IQ recording first")
            return
        if self._analysis_thread is not None and self._analysis_thread.isRunning():
            self.statusBar().showMessage("DECT analysis is already running")
            return
        thread = _DectAnalysisThread(
            self._recording, self._nominal_frequency_hz(), self
        )
        thread.analysis_ready.connect(self._analysis_ready)
        thread.analysis_failed.connect(self._analysis_failed)
        thread.finished.connect(self._analysis_stopped)
        thread.finished.connect(thread.deleteLater)
        self._analysis_thread = thread
        self.capture_button.setText("Stop Analysis")
        self.run_action.setText("Stop")
        self.statusBar().showMessage("Synchronizing and measuring DECT bursts...")
        thread.start()

    @QtCore.Slot(object)
    def _analysis_ready(self, payload: object) -> None:
        if not isinstance(payload, tuple) or not payload or not all(
            isinstance(item, DectPacketResult) for item in payload
        ):
            self._analysis_failed("invalid DECT analysis result")
            return
        self._results = payload
        recording_token = self._recording_revision
        for result in payload:
            token = (recording_token, result.start_sample, result.stop_sample)
            if token in self._accumulated_packet_tokens:
                continue
            self._accumulated_packet_tokens.add(token)
            if result.carrier_test_eligible:
                key = self._carrier_key(result)
                self._carrier_history.setdefault(key, []).append(
                    result.carrier_error_hz
                )
        self._selected_result_index = 0
        self._result = payload[0]
        self.export_modulation_action.setEnabled(True)
        self._render(self._result)
        self.statusBar().showMessage(
            f"DECT analysis complete - {len(payload)} packet(s), "
            f"{self._result.direction} {self._result.packet_type}"
        )

    @QtCore.Slot(str)
    def _analysis_failed(self, message: str) -> None:
        self.statusBar().showMessage(f"DECT analysis failed: {message}")

    @QtCore.Slot()
    def _analysis_stopped(self) -> None:
        self._analysis_thread = None
        self.capture_button.setText("Single Capture")
        self.run_action.setText("Run Single")
        if self._shutdown_requested and self.shutdown_busy_reason() is None:
            self.shutdown_ready.emit()

    def _select_result(self, step: int) -> None:
        self._select_result_index(self._selected_result_index + int(step))

    def _select_result_index(self, index: int) -> None:
        if not 0 <= int(index) < len(self._results):
            return
        self._selected_result_index = int(index)
        self._result = self._results[self._selected_result_index]
        self._render(self._result)
        self.statusBar().showMessage(
            f"Selected DECT packet {self._selected_result_index + 1}/{len(self._results)}"
        )

    def _render(self, result: DectPacketResult) -> None:
        recording = self._recording
        if recording is None:
            return
        display_recordings = AnalysisDisplayRecordings(
            capture=self._capture_recording or recording,
            analysis=recording,
        )
        for plot in self._plots():
            plot.clear()
        apply_analysis_to_power = (
            self.channel_filter_check.isChecked()
            and self.analysis_power_display_check.isChecked()
        )
        power_recording = display_recordings.power(apply_analysis_to_power)
        power_time_s, _power_dbfs, power_dbm = capture_power_traces(
            power_recording
        )
        time_ms = power_time_s * 1e3
        self.power_plot.plot(
            time_ms,
            limit_iq_power_display_dbm(power_dbm),
            pen=pg.mkPen("y", width=1),
        )
        start_ms = result.p0_sample / recording.sample_rate_hz * 1e3
        stop_ms = result.packet_end_sample / recording.sample_rate_hz * 1e3
        add_result_range_overlay(
            self.power_plot,
            result_start_ms=start_ms,
            result_stop_ms=stop_ms,
            pattern_start_ms=start_ms,
            label="p0",
        )
        self.power_plot.addItem(
            pg.InfiniteLine(
                pos=stop_ms,
                angle=90,
                pen=pg.mkPen(90, 100, 110, 200, width=1),
                label="Packet End",
                labelOpts={"position": 0.9, "color": (130, 140, 150)},
            )
        )
        self._render_power_time_template(result, recording)

        apply_analysis_to_spectrum = (
            self.channel_filter_check.isChecked()
            and self.analysis_spectrum_display_check.isChecked()
        )
        spectrum_recording = display_recordings.spectrum(
            apply_analysis_to_spectrum
        )
        frequency_hz, spectrum = recording_spectrum_trace(
            spectrum_recording,
            fft_size=65_536,
            start_time_s=result.start_sample / recording.sample_rate_hz,
            stop_time_s=result.stop_sample / recording.sample_rate_hz,
        )
        frequency_mhz = frequency_hz / 1e6
        self.spectrum_plot.plot(
            frequency_mhz, spectrum, pen=pg.mkPen("y", width=1)
        )

        packet_mask = (
            (
                result.measurement_fm_sample
                >= float(
                    result.metadata.get(
                        "actual_preamble_start_sample", result.p0_sample
                    )
                )
            )
            & (result.measurement_fm_sample <= result.packet_end_sample)
        )
        reference_hz = result.modulation_reference_hz(self._modulation_reference)
        symbol_time_s = result.symbol_centers / recording.sample_rate_hz
        display_drift_hz_per_s, display_reference_time_s = (
            fit_binary_fsk_display_drift(
                symbol_time_s,
                result.symbol_frequency_hz,
                result.bits,
            )
        )
        display_data = build_fsk_display_data(
            result.measurement_fm_frequency_hz[packet_mask],
            result.measurement_fm_sample[packet_mask]
            / recording.sample_rate_hz,
            symbol_time_s,
            frequency_offset_hz=reference_hz,
            frequency_drift_hz_per_s=display_drift_hz_per_s,
            reference_time_s=display_reference_time_s,
        )
        self.deviation_plot.plot(
            display_data.time_s * 1e3,
            display_data.corrected_frequency_hz / 1e3,
            pen=pg.mkPen("y", width=1),
        )
        if self._show_symbol_points:
            plot_trace_symbol_points(
                self.deviation_plot,
                display_data.symbol_time_s * 1e3,
                display_data.symbol_frequency_hz / 1e3,
            )
        for level in (-403.0, -259.0, -202.0, 202.0, 259.0, 403.0):
            self.deviation_plot.addItem(
                pg.InfiniteLine(
                    pos=level,
                    angle=0,
                    pen=pg.mkPen(100, 110, 120, 130, style=QtCore.Qt.PenStyle.DashLine),
                )
            )

        self._render_symbol_plot(result, display_data)
        self._render_summary(result)
        self._render_packet_analysis(result)
        for name, plot in self._plot_widgets():
            if name == "fsk_symbol":
                continue
            if (
                name == "gfsk_modulation"
                and self._fsk_symbol_plot_mode == "Constellation Frequency"
            ):
                continue
            view_all_traces(plot)
        self.deviation_plot.setYRange(
            -_DECT_FSK_MODULATION_Y_LIMIT_KHZ,
            _DECT_FSK_MODULATION_Y_LIMIT_KHZ,
            padding=0.0,
        )
        actual_start_ms = float(
            result.metadata.get("actual_preamble_start_sample", result.p0_sample)
        ) / recording.sample_rate_hz * 1e3
        power_start_ms, power_stop_ms = packet_time_view_range_ms(
            packet_start_ms=actual_start_ms,
            packet_stop_ms=stop_ms,
            capture_stop_ms=recording.duration_s * 1e3,
            minimum_margin_ms=16.0 / DECT_SYMBOL_RATE_HZ * 1e3,
        )
        self.power_plot.setXRange(
            power_start_ms, power_stop_ms, padding=0.0
        )
        self._capture_analysis_plot_ranges()

    def _render_symbol_plot(
        self, result: DectPacketResult, display_data: FSKDisplayData
    ) -> None:
        measured_frequency_hz = display_data.symbol_frequency_hz
        if self._fsk_symbol_plot_mode == "Constellation Frequency":
            self.symbol_plot.setAspectLocked(False)
            self.symbol_plot.setYLink(self.deviation_plot)
            set_frequency_constellation_x_lock(self.symbol_plot, True)
            self.symbol_plot.showAxis("bottom", False)
            self.symbol_plot.setLabel("bottom", "")
            self.symbol_plot.setLabel("left", "Frequency (kHz)")
            limit_khz = 1.5 * 288.0
            plot_frequency_symbol_distribution(
                self.symbol_plot,
                measured_frequency_hz / 1e3,
                y_limit_khz=limit_khz,
                density=self._symbol_density,
                density_spread=self._symbol_density_spread,
            )
            self.symbol_plot.setYRange(-limit_khz, limit_khz, padding=0.0)
            return
        self.symbol_plot.setYLink(None)
        set_frequency_constellation_x_lock(self.symbol_plot, False)
        self.symbol_plot.showAxis("bottom", True)
        self.symbol_plot.setLabel("bottom", "I")
        self.symbol_plot.setLabel("left", "Q")
        self.symbol_plot.setAspectLocked(True, 1.0)
        phase_symbols = np.exp(
            2j * np.pi * measured_frequency_hz / max(result.symbol_rate_hz, 1.0)
        )
        plot_complex_symbol_distribution(
            self.symbol_plot,
            phase_symbols,
            density=self._symbol_density,
            density_spread=self._symbol_density_spread,
        )
        plot_unit_circle(self.symbol_plot)
        set_iq_plane_range(self.symbol_plot)

    def _render_power_time_template(
        self,
        result: DectPacketResult,
        recording: IQRecording,
    ) -> None:
        """Overlay the representative DECT active/edge power limits."""

        burst_start_ms = float(
            result.metadata.get("actual_preamble_start_sample", result.p0_sample)
        ) / recording.sample_rate_hz * 1e3
        end_ms = result.packet_end_sample / recording.sample_rate_hz * 1e3
        edge_ms = 10e-3
        quiet = max(-120.0, result.output_power - 35.0)
        upper = result.output_power + 1.0
        lower = result.output_power - 1.0
        upper_x = np.array(
            (
                burst_start_ms - edge_ms,
                burst_start_ms,
                end_ms,
                end_ms + edge_ms,
            )
        )
        lower_x = upper_x.copy()
        self.power_plot.plot(
            upper_x,
            np.array((quiet, upper, upper, quiet)),
            pen=pg.mkPen(170, 105, 55, 210, width=1),
            name="Power-Time upper limit",
        )
        self.power_plot.plot(
            lower_x,
            np.array((quiet, lower, lower, quiet)),
            pen=pg.mkPen(115, 90, 70, 190, width=1),
            name="Power-Time lower limit",
        )

    def _render_summary(self, result: DectPacketResult) -> None:
        self.summary_table.clearSpans()
        self.summary_table.setRowCount(0)
        last_section = None
        rows = list(result.summary_rows)
        if result.carrier_test_eligible:
            history = self._carrier_history.get(self._carrier_key(result), [])
            required = carrier_repetition_count(result.packet_type)
            if history:
                carrier_error = float(np.mean(history))
                verdict = (
                    ("PASS" if abs(carrier_error) <= 50_000.0 else "FAIL")
                    if len(history) >= required
                    else "MEASURING"
                )
                for index, row in enumerate(rows):
                    if row.test_item == "RF Carrier Frequency Accuracy":
                        rows[index] = replace(
                            row,
                            value=f"{carrier_error / 1e3:+.3f} kHz",
                            result=verdict,
                        )
                        break
            rows.append(
                DectSummaryRow(
                    "Reference Information",
                    "Carrier Packets Evaluated",
                    f"{len(history)} / {required}",
                )
            )
        for row in rows:
            if row.section != last_section:
                section_row = self.summary_table.rowCount()
                self.summary_table.insertRow(section_row)
                item = QtWidgets.QTableWidgetItem(row.section)
                font = item.font()
                font.setBold(True)
                item.setFont(font)
                item.setBackground(QtGui.QColor("#353535"))
                self.summary_table.setItem(section_row, 0, item)
                self.summary_table.setSpan(section_row, 0, 1, 4)
                last_section = row.section
            table_row = self.summary_table.rowCount()
            self.summary_table.insertRow(table_row)
            for column, text in enumerate(
                (row.test_item, row.value, row.limit, row.result)
            ):
                item = QtWidgets.QTableWidgetItem(text)
                if column == 3:
                    color = dedicated_status_color(text)
                    if color is not None:
                        item.setForeground(QtGui.QBrush(color))
                self.summary_table.setItem(table_row, column, item)
        self.summary_table.resizeRowsToContents()

    @staticmethod
    def _carrier_key(result: DectPacketResult) -> tuple[float, str, str]:
        return (
            result.nominal_frequency_hz,
            result.direction,
            result.packet_type,
        )

    def _reset_measurement_statistics(self) -> None:
        self._carrier_history.clear()
        self._accumulated_packet_tokens.clear()
        if self._result is not None:
            self._render_summary(self._result)
        self.statusBar().showMessage("DECT measurement statistics reset")

    def _render_packet_analysis(self, result: DectPacketResult) -> None:
        self.decode_tree.clear()
        p0_internal_bit = 16 if result.preamble_mode == "Prolonged" else 0
        for field in result.packet_analysis.root_fields:
            item = self._packet_field_item(field, result.bits, p0_internal_bit)
            self.decode_tree.addTopLevelItem(item)
            item.setExpanded(True)
        self.decode_tree.expandToDepth(2)
        grouped = " ".join(
            "".join(str(int(bit)) for bit in result.bits[start : start + 8])
            for start in range(0, result.bits.size, 8)
        )
        self.air_bits_text.setPlainText(
            f"First transmitted bit at left\n\n{grouped}"
        )
        self.packet_table.setRowCount(len(self._results))
        for row, packet in enumerate(self._results):
            for column, text in enumerate(
                (
                    str(row + 1),
                    packet.direction,
                    packet.packet_type,
                    packet.modulation_case,
                    str(
                        packet.metadata.get(
                            "physical_packet_symbol_count", packet.bits.size
                        )
                    ),
                )
            ):
                self.packet_table.setItem(row, column, QtWidgets.QTableWidgetItem(text))
        self.packet_table.selectRow(self._selected_result_index)

    def _packet_field_item(
        self,
        field: PacketField,
        bits: np.ndarray,
        p0_internal_bit: int,
    ) -> QtWidgets.QTreeWidgetItem:
        stop = min(field.stop_bit, bits.size)
        value = "" if field.value is None else str(field.value)
        bit_range = (
            "N/A"
            if stop <= field.start_bit
            else str(field.start_bit)
            if stop == field.start_bit + 1
            else f"{field.start_bit}–{stop - 1}"
        )
        dect_start, dect_stop = dect_p_range(
            field.start_bit, stop, p0_internal_bit
        )
        dect_symbols = (
            "N/A"
            if dect_start is None or dect_stop is None or dect_stop <= dect_start
            else f"p{dect_start}"
            if dect_stop == dect_start + 1
            else f"p{dect_start}–p{dect_stop - 1}"
        )
        item = QtWidgets.QTreeWidgetItem(
            (field.name, value, bit_range, dect_symbols, str(field.status))
        )
        color = dedicated_status_color(field.status)
        if color is not None:
            item.setForeground(4, QtGui.QBrush(color))
        for column, text in enumerate(
            (field.name, value, bit_range, dect_symbols, str(field.status))
        ):
            tooltip = str(text)
            if field.meaning:
                tooltip = f"{tooltip}\n{field.meaning}" if tooltip else field.meaning
            item.setToolTip(column, tooltip)
        for child in field.children:
            item.addChild(
                self._packet_field_item(child, bits, p0_internal_bit)
            )
        return item

    def _reset_plot_scales(self) -> None:
        for name, plot in self._plot_widgets():
            self._reset_plot_scale(name, plot)

    def _reset_plot_scale(self, name: str, plot: pg.PlotWidget) -> None:
        ranges = self._analysis_plot_ranges.get(name)
        if ranges is None:
            return
        x_range, y_range = ranges
        plot.setRange(xRange=x_range, yRange=y_range, padding=0.0)

    def _view_all_plot(self, name: str, plot: pg.PlotWidget) -> None:
        if name != "fsk_symbol":
            view_all_traces(plot)
            return
        if self._fsk_symbol_plot_mode == "Constellation Frequency":
            set_frequency_constellation_x_lock(plot, True)
            plot.setYRange(-432.0, 432.0, padding=0.0)
            return
        set_iq_plane_range(plot)

    def _capture_analysis_plot_ranges(self) -> None:
        self._analysis_plot_ranges = {
            name: (list(plot.viewRange()[0]), list(plot.viewRange()[1]))
            for name, plot in self._plot_widgets()
        }

    def shutdown_busy_reason(self) -> str | None:
        if self._capture_thread is not None and self._capture_thread.isRunning():
            return "DECT IQ capture is running"
        if self._analysis_thread is not None and self._analysis_thread.isRunning():
            return "DECT analysis is running"
        return None

    def request_shutdown(self) -> None:
        self._shutdown_requested = True
        if self._capture_thread is not None and self._capture_thread.isRunning():
            self._capture_thread.cancel()
        if self._analysis_thread is not None and self._analysis_thread.isRunning():
            self._analysis_thread.requestInterruption()

    def finalize_shutdown(self) -> None:
        self._save_config()
        if self._owns_pluto_source:
            self._pluto_source.close()
