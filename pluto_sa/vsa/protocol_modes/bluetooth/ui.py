"""Bluetooth dedicated-analysis workspace embedded in Pluto VSA."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from pluto_protocol.model import FieldStatus, PacketField
from pluto_sa.sdr.trigger import TriggerKind, TriggerSlope
from pluto_sa.vsa.model import IQRecording, ModulationFamily, VSAAnalysisResult
from pluto_sa.vsa.analysis import capture_power_traces
from pluto_sa.vsa.channel import (
    extract_requested_analysis_channel,
    validate_analysis_channel_capture,
)
from pluto_sa.config.input_frontend import InputPowerCorrection
from pluto_sa.vsa.pluto_source import PlutoCaptureSettings, PlutoLiveSource
from pluto_sa.vsa.session import VSASession
from pluto_sa.vsa.sources import FileIQSource
from pluto_sa.vsa.pattern import IQPowerTriggerSettings, MeasurementFilterMode
from pluto_sa.vsa.ui.display_processing import (
    FSKDisplayData,
    build_fsk_display_data,
    normalized_psk_display,
    prepare_fsk_display_frequency,
    prepare_psk_display_waveform,
)
from pluto_sa.vsa.ui.capture_thread import PlutoSingleCaptureThread
from pluto_sa.vsa.ui.measurement_chrome import (
    DedicatedPacketAnalysisTree,
    DedicatedSummaryTable,
    IQ_PLANE_LIMIT,
    FREQUENCY_CONSTELLATION_X_LIMIT,
    SymbolDensitySpread,
    add_fsk_symbol_plot_menu,
    add_result_range_overlay,
    add_symbol_density_menu,
    apply_dedicated_table_style,
    configure_iq_power_plot,
    dedicated_status_color,
    install_measurement_plot_menu,
    limit_iq_power_display_dbm,
    make_measurement_dock,
    make_measurement_plot,
    plot_complex_symbol_distribution,
    plot_frequency_symbol_distribution,
    plot_trace_symbol_points,
    plot_unit_circle,
    set_iq_plane_range,
    set_frequency_constellation_x_lock,
    trace_bounds,
    view_all_traces,
)
from pluto_sa.vsa.ui.measurement_config_dialog import HierarchicalMeasConfigDialog
from pluto_sa.vsa.ui.iq_export import export_iq_recording

from .model import (
    BluetoothAnalysisProfile,
    BluetoothDedicatedResult,
    analyze_bluetooth_classic_recordings,
    analyze_bluetooth_hdt_recordings,
    analyze_bluetooth_le_recordings,
    analyze_bluetooth_session,
)
from .rf_measurement import BluetoothFMMeasurementTrace, HDTEVMResult, HDTPlotData


_TRACE = "#ffff00"
_STARTUP_CONFIG_KEY = "bluetooth_dedicated/startup_meas_config"
_STARTUP_CONFIG_SCHEMA = "pluto-vsa-bluetooth-dedicated-config"
_STARTUP_CONFIG_VERSION = 1
_FREQUENCY_CONSTELLATION_X_LIMIT = FREQUENCY_CONSTELLATION_X_LIMIT


def _hdt_modulation_name(name: str) -> str:
    """Return the modulation label shared by HDT plot tabs and legends."""

    return "QPSK" if name.strip().lower() == "pi/4-qpsk" else name.strip()


def _hdt_pi4_qpsk_display_symbols(symbols: np.ndarray) -> np.ndarray:
    """Collapse alternating pi/4-QPSK symbol sets onto the four I/Q axes.

    The unit-magnitude, symbol-index-dependent rotation is display-only.  If
    it is applied to both measured and reference symbols, their RMS EVM is
    unchanged.
    """

    values = np.asarray(symbols, dtype=np.complex128)
    axis = np.arange(values.size, dtype=np.float64)
    return values * np.exp(-1j * (axis + 1.0) * np.pi / 4.0)


def format_air_bits(bits: np.ndarray, group: int = 8) -> str:
    values = np.asarray(bits, dtype=np.uint8)
    binary = " ".join(
        "".join(str(int(value)) for value in values[start : start + group])
        for start in range(0, values.size, group)
    )
    octets = np.packbits(np.pad(values, (0, (-values.size) % 8)), bitorder="little")
    hexadecimal = " ".join(f"{int(value):02X}" for value in octets)
    return f"Air bits (first transmitted bit at left)\n{binary}\n\nOctets (LSB-first)\n{hexadecimal}"


def payload_field(fields: Iterable[PacketField]) -> PacketField | None:
    for field in fields:
        if field.field_id in {"payload", "payload_body"}:
            return field
        found = payload_field(field.children)
        if found is not None:
            return found
    return None


def infer_le_channel(center_frequency_hz: float) -> int:
    mhz = int(round(center_frequency_hz / 1e6))
    if mhz in {2402, 2426, 2480}:
        return {2402: 37, 2426: 38, 2480: 39}[mhz]
    if 2404 <= mhz <= 2424 and mhz % 2 == 0:
        return (mhz - 2404) // 2
    if 2428 <= mhz <= 2478 and mhz % 2 == 0:
        return 11 + (mhz - 2428) // 2
    return 37


class _PacketAnalysisTree(DedicatedPacketAnalysisTree):
    def __init__(self, parent=None) -> None:
        super().__init__(
            ("Field", "Value", "Stream", "Bit Range", "Status"),
            (120, 120, 105, 72, 55),
            expand_columns=(0, 1),
            parent=parent,
        )


class _SummaryTable(DedicatedSummaryTable):
    pass


class _BluetoothClassicAnalysisThread(QtCore.QThread):
    analysis_ready = QtCore.Signal(object)
    analysis_failed = QtCore.Signal(str)

    def __init__(self, recording: IQRecording, options: dict[str, object], parent=None) -> None:
        super().__init__(parent)
        self._recording = recording
        self._options = dict(options)

    def run(self) -> None:
        try:
            results = analyze_bluetooth_classic_recordings(
                self._recording,
                cancelled=self.isInterruptionRequested,
                **self._options,
            )
            if not self.isInterruptionRequested():
                self.analysis_ready.emit(results)
        except Exception as error:
            self.analysis_failed.emit(str(error))


class _BluetoothLEAnalysisThread(QtCore.QThread):
    analysis_ready = QtCore.Signal(object)
    analysis_failed = QtCore.Signal(str)

    def __init__(self, recording: IQRecording, options: dict[str, object], parent=None) -> None:
        super().__init__(parent)
        self._recording = recording
        self._options = dict(options)

    def run(self) -> None:
        try:
            results = analyze_bluetooth_le_recordings(
                self._recording,
                cancelled=self.isInterruptionRequested,
                **self._options,
            )
            if not self.isInterruptionRequested():
                self.analysis_ready.emit(results)
        except Exception as error:
            self.analysis_failed.emit(str(error))


class _BluetoothHDTAnalysisThread(QtCore.QThread):
    analysis_ready = QtCore.Signal(object)
    analysis_failed = QtCore.Signal(str)

    def __init__(self, recording: IQRecording, options: dict[str, object], parent=None) -> None:
        super().__init__(parent)
        self._recording = recording
        self._options = dict(options)

    def run(self) -> None:
        try:
            results = analyze_bluetooth_hdt_recordings(
                self._recording,
                cancelled=self.isInterruptionRequested,
                **self._options,
            )
            if not self.isInterruptionRequested():
                self.analysis_ready.emit(results)
        except Exception as error:
            self.analysis_failed.emit(str(error))


class BluetoothAnalyzerWindow(QtWidgets.QMainWindow):
    """Six-pane Bluetooth analyzer using capture or reusable Generic VSA IQ."""

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
        # Dedicated analysis intentionally owns a separate configuration
        # namespace from Generic VSA.
        self._preferences = preferences or QtCore.QSettings(
            "PlutoSA", "PlutoVSA-Bluetooth"
        )
        self._owns_pluto_source = pluto_source is None
        self._pluto_target = ""
        self._capture_thread: PlutoSingleCaptureThread | None = None
        self._analysis_thread: (
            _BluetoothClassicAnalysisThread
            | _BluetoothLEAnalysisThread
            | _BluetoothHDTAnalysisThread
            | None
        ) = None
        self._shutdown_requested = False
        self._session: VSASession | None = None
        self._recording: IQRecording | None = None
        self._result: BluetoothDedicatedResult | None = None
        self._results: tuple[BluetoothDedicatedResult, ...] = ()
        self._selected_result_index = 0
        self._show_symbol_points = True
        self._symbol_density = False
        self._symbol_density_spread = SymbolDensitySpread.MAXIMUM
        self._fsk_symbol_plot_mode = "Constellation Frequency"
        self._psk_symbol_plot_mode = "Physical IQ"
        self._analysis_plot_ranges: dict[
            str, tuple[list[float], list[float]]
        ] = {}
        self._plot_context_actions: dict[
            str, dict[str, QtGui.QAction]
        ] = {}
        self.setDockOptions(
            QtWidgets.QMainWindow.DockOption.AllowNestedDocks
            | QtWidgets.QMainWindow.DockOption.AllowTabbedDocks
        )
        self._build_menu()
        self._build_controls()
        # Trigger controls affect capture/analysis even before the user opens
        # Meas Config, so construct their shared dialog eagerly.
        self._build_meas_config_dialog()
        self._build_results()
        self._configure_plot_context_menus()
        restored = self._restore_startup_meas_config()
        self.statusBar().showMessage(
            "Ready - Bluetooth configuration restored"
            if restored
            else "Ready - capture IQ or reuse the current Generic VSA recording"
        )

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        self.open_iq_action = file_menu.addAction("Open IQ...")
        self.open_iq_action.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        self.open_iq_action.triggered.connect(self._open_iq)
        self.export_iq_action = file_menu.addAction("Export IQ Recording...")
        self.export_iq_action.setEnabled(False)
        self.export_iq_action.triggered.connect(self._export_iq_recording)
        file_menu.addSeparator()
        close_action = file_menu.addAction("Close")
        close_action.triggered.connect(self.application_close_requested.emit)
        run_menu = self.menuBar().addMenu("Sweep / Run")
        self.run_action = run_menu.addAction("Run Single")
        self.run_action.setShortcut(QtGui.QKeySequence("F6"))
        self.run_action.triggered.connect(self._toggle_capture)
        refresh_action = run_menu.addAction("Refresh Analysis")
        refresh_action.setShortcut(QtGui.QKeySequence("F5"))
        refresh_action.triggered.connect(self.refresh)
        run_menu.addSeparator()
        previous_action = run_menu.addAction("Previous Packet")
        previous_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Left))
        previous_action.triggered.connect(lambda: self._select_result(-1))
        next_action = run_menu.addAction("Next Packet")
        next_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Right))
        next_action.triggered.connect(lambda: self._select_result(1))

        display_menu = self.menuBar().addMenu("Display Config")
        self.symbols_action = display_menu.addAction("Show Symbol Points")
        self.symbols_action.setCheckable(True)
        self.symbols_action.setChecked(True)
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
        psk_plot_menu = display_menu.addMenu("PSK Symbol Plot")
        psk_plot_group = QtGui.QActionGroup(self)
        psk_plot_group.setExclusive(True)
        self.psk_physical_action = psk_plot_menu.addAction("Physical IQ")
        self.psk_differential_action = psk_plot_menu.addAction("Differential IQ")
        for action in (self.psk_physical_action, self.psk_differential_action):
            action.setCheckable(True)
            psk_plot_group.addAction(action)
        self.psk_physical_action.setChecked(True)
        self.psk_physical_action.triggered.connect(
            lambda: self._set_psk_symbol_plot_mode("Physical IQ")
        )
        self.psk_differential_action.triggered.connect(
            lambda: self._set_psk_symbol_plot_mode("Differential IQ")
        )
        reset_action = display_menu.addAction("Reset Plot Scales")
        reset_action.setShortcut(QtGui.QKeySequence("Home"))
        reset_action.triggered.connect(self._reset_plot_scales)

        config_menu = self.menuBar().addMenu("Meas Config")
        open_config = config_menu.addAction("Open Meas Config...")
        open_config.setShortcut(QtGui.QKeySequence("Ctrl+M"))
        open_config.triggered.connect(self._show_meas_config)

        menu = self.menuBar().addMenu("Analysis Mode")
        generic = menu.addAction("Generic FSK / PSK VSA...")
        generic.triggered.connect(lambda: self.analysis_mode_requested.emit("generic"))
        current = menu.addAction("Bluetooth Dedicated Analyzer")
        current.setCheckable(True)
        current.setChecked(True)
        current.setEnabled(False)
        dect = menu.addAction("DECT Dedicated Analyzer...")
        dect.triggered.connect(lambda: self.analysis_mode_requested.emit("dect"))
        adsb = menu.addAction("ADS-B 1090ES...")
        adsb.triggered.connect(lambda: self.analysis_mode_requested.emit("adsb1090"))

    def _build_controls(self) -> None:
        # These widgets live exclusively in the modal Meas Config dialog.
        # Registering them with a hidden QToolBar first creates QWidgetActions
        # which continue to control their visibility even after a layout
        # reparents them.  That left only the form labels visible in Config.
        self.profile_combo = QtWidgets.QComboBox()
        self.profile_combo.addItem("RF / PHY Test", BluetoothAnalysisProfile.RF_PHY_TEST)
        self.profile_combo.addItem("General Packet", BluetoothAnalysisProfile.GENERAL_PACKET)
        self.protocol_combo = QtWidgets.QComboBox()
        self.protocol_combo.addItem("Bluetooth BR / EDR", "bluetooth.br_edr")
        self.protocol_combo.addItem("Bluetooth LE", "bluetooth.le")
        self.protocol_combo.addItem("Bluetooth HDT", "bluetooth.hdt")
        self.phy_combo = QtWidgets.QComboBox()
        self.lap_edit = QtWidgets.QLineEdit("C6967E")
        self.lap_edit.setMaximumWidth(72)
        self.uap_edit = QtWidgets.QLineEdit("6B")
        self.uap_edit.setMaximumWidth(50)
        self.clock_spin = QtWidgets.QSpinBox()
        self.clock_spin.setRange(0, 63)
        self.clock_spin.setValue(0x2B)
        self.channel_spin = QtWidgets.QSpinBox()
        self.channel_spin.setRange(0, 39)
        self.channel_spin.setValue(37)
        self.access_address_edit = QtWidgets.QLineEdit("8E89BED6")
        self.access_address_edit.setMaximumWidth(82)
        self.crc_init_edit = QtWidgets.QLineEdit("555555")
        self.crc_init_edit.setMaximumWidth(72)
        self.whitening_check = QtWidgets.QCheckBox("Whitening")
        self.refresh_button = QtWidgets.QPushButton("Refresh Result")
        self.context_label = QtWidgets.QLabel()
        self.capture_button = QtWidgets.QPushButton("Single Capture")
        self.center_spin = QtWidgets.QDoubleSpinBox()
        self.center_spin.setRange(70.0, 6000.0)
        self.center_spin.setDecimals(6)
        self.center_spin.setValue(2440.0)
        self.capture_length_spin = QtWidgets.QDoubleSpinBox()
        self.capture_length_spin.setRange(0.1, 1000.0)
        self.capture_length_spin.setValue(10.0)
        self.oversampling_combo = QtWidgets.QComboBox()
        for value in (4, 8, 16, 32):
            self.oversampling_combo.addItem(f"{value} S/sym", value)
        self.oversampling_combo.setCurrentIndex(1)
        self.rf_bandwidth_spin = QtWidgets.QDoubleSpinBox()
        self.rf_bandwidth_spin.setRange(0.2, 56.0)
        self.rf_bandwidth_spin.setValue(8.0)
        self.channel_filter_check = QtWidgets.QCheckBox("Enable Analysis Channel")
        self.analysis_bandwidth_spin = QtWidgets.QDoubleSpinBox()
        self.analysis_bandwidth_spin.setRange(0.000001, 100.0)
        self.analysis_bandwidth_spin.setDecimals(6)
        self.analysis_bandwidth_spin.setValue(1.5)
        self.analysis_bandwidth_spin.setSuffix(" MHz")
        self.lo_offset_check = QtWidgets.QCheckBox("Enable")
        self.lo_offset_check.setToolTip(
            "Tune the Pluto LO away from the selected Bluetooth channel. "
            "Requires the Analysis Channel filter."
        )
        self.lo_offset_spin = QtWidgets.QDoubleSpinBox()
        self.lo_offset_spin.setRange(-50.0, 50.0)
        self.lo_offset_spin.setDecimals(6)
        self.lo_offset_spin.setValue(1.5)
        self.lo_offset_spin.setSuffix(" MHz")
        self.resolved_lo_label = QtWidgets.QLabel()
        self.internal_gain_spin = QtWidgets.QDoubleSpinBox()
        self.internal_gain_spin.setRange(0.0, 70.0)
        self.internal_gain_spin.setValue(30.0)
        self.external_att_spin = QtWidgets.QDoubleSpinBox()
        self.external_att_spin.setRange(-100.0, 100.0)
        self.external_att_spin.setValue(30.0)
        self.device_label = QtWidgets.QLabel("Pluto: Auto")
        self.protocol_combo.currentIndexChanged.connect(self._protocol_changed)
        self.profile_combo.currentIndexChanged.connect(self._profile_changed)
        self.phy_combo.currentIndexChanged.connect(self._update_derived_config)
        self.refresh_button.clicked.connect(self.refresh)
        self.capture_button.clicked.connect(self._toggle_capture)
        self.center_spin.valueChanged.connect(self._sync_analysis_channel_controls)
        self.channel_filter_check.toggled.connect(
            self._sync_analysis_channel_controls
        )
        self.lo_offset_check.toggled.connect(self._sync_analysis_channel_controls)
        self.lo_offset_spin.valueChanged.connect(self._sync_analysis_channel_controls)
        self._sync_analysis_channel_controls()
        self._protocol_changed()

    def _sync_analysis_channel_controls(self, _value: object = None) -> None:
        filter_enabled = self.channel_filter_check.isChecked()
        if self.sender() is self.lo_offset_check and self.lo_offset_check.isChecked():
            self.channel_filter_check.setChecked(True)
            filter_enabled = True
        elif not filter_enabled and self.lo_offset_check.isChecked():
            self.lo_offset_check.setChecked(False)
        self.analysis_bandwidth_spin.setEnabled(filter_enabled)
        offset_enabled = self.lo_offset_check.isChecked()
        self.lo_offset_spin.setEnabled(offset_enabled)
        offset_mhz = self.lo_offset_spin.value() if offset_enabled else 0.0
        self.resolved_lo_label.setText(
            f"{self.center_spin.value() + offset_mhz:.6f} MHz"
            + (" (offset on)" if offset_enabled else " (offset off)")
        )

    def _build_trigger_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)

        acquisition_heading = QtWidgets.QLabel("Acquisition Trigger")
        acquisition_heading.setStyleSheet("font-weight: bold;")
        self.acquisition_trigger_source_combo = QtWidgets.QComboBox()
        self.acquisition_trigger_source_combo.addItem(
            "Free Run", TriggerKind.FREE_RUN.value
        )
        self.acquisition_trigger_source_combo.addItem(
            "I/Q Power", TriggerKind.POWER_LEVEL.value
        )
        self.acquisition_trigger_level_spin = QtWidgets.QDoubleSpinBox()
        self.acquisition_trigger_level_spin.setRange(-200.0, 100.0)
        self.acquisition_trigger_level_spin.setDecimals(2)
        self.acquisition_trigger_level_spin.setValue(-20.0)
        self.acquisition_trigger_level_spin.setSuffix(" dBm")
        self.acquisition_trigger_slope_combo = QtWidgets.QComboBox()
        for slope in TriggerSlope:
            self.acquisition_trigger_slope_combo.addItem(
                slope.value.capitalize(), slope.value
            )
        self.acquisition_trigger_offset_spin = QtWidgets.QDoubleSpinBox()
        self.acquisition_trigger_offset_spin.setRange(-1_000_000.0, 1_000_000.0)
        self.acquisition_trigger_offset_spin.setDecimals(3)
        self.acquisition_trigger_offset_spin.setSuffix(" sym")
        self.acquisition_trigger_hysteresis_spin = QtWidgets.QDoubleSpinBox()
        self.acquisition_trigger_hysteresis_spin.setRange(0.0, 50.0)
        self.acquisition_trigger_hysteresis_spin.setDecimals(1)
        self.acquisition_trigger_hysteresis_spin.setValue(3.0)
        self.acquisition_trigger_hysteresis_spin.setSuffix(" dB")
        form.addRow(acquisition_heading)
        form.addRow("Trigger Source", self.acquisition_trigger_source_combo)
        form.addRow("Level", self.acquisition_trigger_level_spin)
        form.addRow("Slope", self.acquisition_trigger_slope_combo)
        form.addRow("Trigger Offset", self.acquisition_trigger_offset_spin)
        form.addRow("Hysteresis", self.acquisition_trigger_hysteresis_spin)

        burst_heading = QtWidgets.QLabel("Post-capture Burst Search")
        burst_heading.setStyleSheet("font-weight: bold;")
        self.iq_power_trigger_check = QtWidgets.QCheckBox("Burst Search On")
        self.iq_power_trigger_check.setChecked(True)
        self.iq_power_trigger_level_spin = QtWidgets.QDoubleSpinBox()
        self.iq_power_trigger_level_spin.setRange(-200.0, 100.0)
        self.iq_power_trigger_level_spin.setDecimals(2)
        self.iq_power_trigger_level_spin.setValue(-20.0)
        self.iq_power_trigger_level_spin.setSuffix(" dBm")
        self.iq_power_trigger_hysteresis_spin = QtWidgets.QDoubleSpinBox()
        self.iq_power_trigger_hysteresis_spin.setRange(0.0, 60.0)
        self.iq_power_trigger_hysteresis_spin.setDecimals(2)
        self.iq_power_trigger_hysteresis_spin.setValue(3.0)
        self.iq_power_trigger_hysteresis_spin.setSuffix(" dB")
        self.iq_power_trigger_average_spin = QtWidgets.QDoubleSpinBox()
        self.iq_power_trigger_average_spin.setRange(0.0, 1_000.0)
        self.iq_power_trigger_average_spin.setDecimals(2)
        self.iq_power_trigger_average_spin.setValue(1.0)
        self.iq_power_trigger_average_spin.setSuffix(" sym")
        self.iq_power_trigger_dropout_spin = QtWidgets.QDoubleSpinBox()
        self.iq_power_trigger_dropout_spin.setRange(0.0, 1_000_000.0)
        self.iq_power_trigger_dropout_spin.setDecimals(2)
        self.iq_power_trigger_dropout_spin.setValue(8.0)
        self.iq_power_trigger_dropout_spin.setSuffix(" sym")
        self.iq_power_trigger_holdoff_spin = QtWidgets.QDoubleSpinBox()
        self.iq_power_trigger_holdoff_spin.setRange(0.0, 1_000_000.0)
        self.iq_power_trigger_holdoff_spin.setDecimals(2)
        self.iq_power_trigger_holdoff_spin.setSuffix(" sym")
        self.iq_power_trigger_offset_spin = QtWidgets.QDoubleSpinBox()
        self.iq_power_trigger_offset_spin.setRange(-1_000_000.0, 1_000_000.0)
        self.iq_power_trigger_offset_spin.setDecimals(3)
        self.iq_power_trigger_offset_spin.setSuffix(" sym")
        self.iq_power_trigger_limit_result_check = QtWidgets.QCheckBox(
            "Limit Result Range to Active Interval"
        )
        self.iq_power_trigger_limit_result_check.setChecked(True)
        form.addRow(burst_heading)
        form.addRow(self.iq_power_trigger_check)
        form.addRow("Level", self.iq_power_trigger_level_spin)
        form.addRow("Hysteresis", self.iq_power_trigger_hysteresis_spin)
        form.addRow("Envelope Average", self.iq_power_trigger_average_spin)
        form.addRow("Drop-Out Time", self.iq_power_trigger_dropout_spin)
        form.addRow("Holdoff", self.iq_power_trigger_holdoff_spin)
        form.addRow("Search Start Offset", self.iq_power_trigger_offset_spin)
        form.addRow(self.iq_power_trigger_limit_result_check)
        self.acquisition_trigger_source_combo.currentIndexChanged.connect(
            self._sync_acquisition_trigger_controls
        )
        self._sync_acquisition_trigger_controls()
        return page

    def _build_meas_config_dialog(self) -> None:
        if hasattr(self, "_meas_config_dialog"):
            return
        pages: list[tuple[str, QtWidgets.QWidget]] = []

        def add_page(title: str, page: QtWidgets.QWidget, row: int, column: int) -> None:
            del row, column
            pages.append((title, page))

        bt_page = QtWidgets.QWidget()
        bt_form = QtWidgets.QFormLayout(bt_page)
        for label, widget in (
            ("Profile", self.profile_combo), ("Protocol", self.protocol_combo),
            ("PHY", self.phy_combo), ("LAP", self.lap_edit), ("UAP", self.uap_edit),
            ("CLK6-1", self.clock_spin), ("Access Address", self.access_address_edit),
            ("LE Channel", self.channel_spin), ("CRC Init", self.crc_init_edit),
            ("Whitening", self.whitening_check),
        ):
            bt_form.addRow(label, widget)

        input_page = QtWidgets.QWidget()
        input_form = QtWidgets.QFormLayout(input_page)
        for label, widget in (
            ("Center Frequency (MHz)", self.center_spin),
            ("Capture Length (ms)", self.capture_length_spin),
            ("Samples / Symbol", self.oversampling_combo),
            ("RF Bandwidth (MHz)", self.rf_bandwidth_spin),
            ("Analysis Channel", self.channel_filter_check),
            ("Analysis Bandwidth", self.analysis_bandwidth_spin),
            ("LO Offset", self.lo_offset_check),
            ("Offset Frequency", self.lo_offset_spin),
            ("Resolved LO", self.resolved_lo_label),
            ("Internal Gain (dB)", self.internal_gain_spin),
            ("External ATT (dB)", self.external_att_spin),
            ("Input Device", self.device_label),
        ):
            input_form.addRow(label, widget)

        signal_page = QtWidgets.QWidget()
        signal_form = QtWidgets.QFormLayout(signal_page)
        self.derived_modulation = QtWidgets.QLineEdit(readOnly=True)
        self.derived_symbol_rate = QtWidgets.QLineEdit(readOnly=True)
        self.derived_tx_filter = QtWidgets.QLineEdit(readOnly=True)
        self.derived_result_range = QtWidgets.QLineEdit(readOnly=True)
        for widget in (self.derived_modulation, self.derived_symbol_rate, self.derived_tx_filter, self.derived_result_range):
            widget.setEnabled(False)
        signal_form.addRow("Modulation (from PHY)", self.derived_modulation)
        signal_form.addRow("Symbol Rate (from PHY)", self.derived_symbol_rate)
        signal_form.addRow("TX Filter (from PHY)", self.derived_tx_filter)
        signal_form.addRow("Result Range", self.derived_result_range)
        signal_form.addRow(QtWidgets.QLabel("PHY-derived parameters are intentionally read-only."))

        display_page = QtWidgets.QWidget()
        display_layout = QtWidgets.QVBoxLayout(display_page)
        self.config_show_symbols = QtWidgets.QCheckBox("Show Symbol Points")
        self.config_show_symbols.setChecked(self._show_symbol_points)
        self.config_show_symbols.toggled.connect(self._set_show_symbol_points)
        self.config_density = QtWidgets.QCheckBox("Symbol Plot Density")
        self.config_density.setChecked(self._symbol_density)
        self.config_density.toggled.connect(self._set_symbol_density)
        display_layout.addWidget(self.config_show_symbols)
        display_layout.addWidget(self.config_density)
        self.config_density_spread = QtWidgets.QComboBox()
        self.config_density_spread.addItems(
            tuple(spread.value for spread in SymbolDensitySpread)
        )
        self.config_density_spread.setCurrentText(self._symbol_density_spread.value)
        self.config_density_spread.currentTextChanged.connect(
            self._set_symbol_density_spread
        )
        display_layout.addWidget(QtWidgets.QLabel("Density Spread (all modulations)"))
        display_layout.addWidget(self.config_density_spread)
        self.config_fsk_mode = QtWidgets.QComboBox()
        self.config_fsk_mode.addItems(("Constellation Frequency", "Phase Difference"))
        self.config_fsk_mode.setCurrentText(self._fsk_symbol_plot_mode)
        self.config_fsk_mode.currentTextChanged.connect(self._set_fsk_symbol_plot_mode)
        self.config_psk_mode = QtWidgets.QComboBox()
        self.config_psk_mode.addItems(("Physical IQ", "Differential IQ"))
        self.config_psk_mode.setCurrentText(self._psk_symbol_plot_mode)
        self.config_psk_mode.currentTextChanged.connect(self._set_psk_symbol_plot_mode)
        display_layout.addWidget(QtWidgets.QLabel("FSK Symbol Plot"))
        display_layout.addWidget(self.config_fsk_mode)
        display_layout.addWidget(QtWidgets.QLabel("PSK Symbol Plot"))
        display_layout.addWidget(self.config_psk_mode)
        display_layout.addStretch(1)

        run_page = QtWidgets.QWidget()
        run_layout = QtWidgets.QVBoxLayout(run_page)
        run_layout.addWidget(self.capture_button)
        run_layout.addWidget(self.refresh_button)
        run_layout.addStretch(1)
        add_page("Bluetooth Analysis", bt_page, 0, 0)
        add_page("Input / Frontend", input_page, 0, 1)
        add_page("Signal Description", signal_page, 1, 0)
        add_page("Display Config", display_page, 1, 1)
        add_page("Trigger", self._build_trigger_page(), 2, 0)
        add_page("Sweep / Run", run_page, 2, 1)

        dialog = HierarchicalMeasConfigDialog(
            self,
            pages,
            window_title="Bluetooth Meas Config",
            size=(820, 620),
            standard_buttons=(
                QtWidgets.QDialogButtonBox.StandardButton.Ok
                | QtWidgets.QDialogButtonBox.StandardButton.Cancel
            ),
        )
        dialog.accepted.connect(self._save_startup_meas_config)
        dialog.accepted.connect(self.refresh)
        self._meas_config_dialog = dialog
        self._config_stack = dialog.stack
        self._config_top_buttons = dialog.top_buttons
        self._config_back_button = dialog.back_button
        self._update_derived_config()

    @QtCore.Slot()
    def _show_meas_config(self) -> None:
        if not hasattr(self, "_meas_config_dialog"):
            self._build_meas_config_dialog()
        for control, value in (
            (self.config_show_symbols, self._show_symbol_points),
            (self.config_density, self._symbol_density),
        ):
            control.blockSignals(True)
            control.setChecked(value)
            control.blockSignals(False)
        for control, value in (
            (self.config_fsk_mode, self._fsk_symbol_plot_mode),
            (self.config_psk_mode, self._psk_symbol_plot_mode),
            (self.config_density_spread, self._symbol_density_spread.value),
        ):
            control.blockSignals(True)
            control.setCurrentText(value)
            control.blockSignals(False)
        self._update_derived_config()
        self._meas_config_dialog.open_top()

    @QtCore.Slot()
    def _update_derived_config(self) -> None:
        if not hasattr(self, "derived_modulation"):
            return
        phy = self.phy_combo.currentText()
        if phy == "LE 2M":
            values = ("GFSK", "2.000 MSym/s", "Gaussian")
        elif phy.startswith("LE"):
            values = ("GFSK", "1.000 MSym/s", "Gaussian")
        elif self.protocol_combo.currentData() == "bluetooth.hdt":
            values = (
                "Auto: pi/4-QPSK + 8PSK / 16QAM",
                "2.000 MSym/s",
                "Root Raised Cosine",
            )
        else:
            values = ("Auto: GFSK / DPSK", "PHY-derived", "PHY-defined")
        self.derived_modulation.setText(values[0])
        self.derived_symbol_rate.setText(values[1])
        self.derived_tx_filter.setText(values[2])
        self.derived_result_range.setText("Automatic packet extent")

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
        if hasattr(self, "density_spread_actions"):
            self.density_spread_actions[resolved].setChecked(True)
        if hasattr(self, "config_density_spread"):
            self.config_density_spread.blockSignals(True)
            self.config_density_spread.setCurrentText(resolved.value)
            self.config_density_spread.blockSignals(False)
        if self._result is not None:
            self._render(self._result)

    @QtCore.Slot(str)
    def _set_fsk_symbol_plot_mode(self, mode: str) -> None:
        if mode not in {"Constellation Frequency", "Phase Difference"}:
            return
        self._fsk_symbol_plot_mode = mode
        if hasattr(self, "fsk_frequency_action"):
            self.fsk_frequency_action.setChecked(mode == "Constellation Frequency")
            self.fsk_phase_action.setChecked(mode == "Phase Difference")
        if hasattr(self, "config_fsk_mode"):
            self.config_fsk_mode.blockSignals(True)
            self.config_fsk_mode.setCurrentText(mode)
            self.config_fsk_mode.blockSignals(False)
        if hasattr(self, "fsk_symbol_plot"):
            self._set_frequency_constellation_x_lock(
                mode == "Constellation Frequency"
            )
        if self._result is not None:
            self._render(self._result)

    def _set_frequency_constellation_x_lock(self, locked: bool) -> None:
        """Keep Constellation Frequency's display-only X axis immutable."""
        set_frequency_constellation_x_lock(self.fsk_symbol_plot, locked)

    @QtCore.Slot(str)
    def _set_psk_symbol_plot_mode(self, mode: str) -> None:
        if mode not in {"Physical IQ", "Differential IQ"}:
            return
        self._psk_symbol_plot_mode = mode
        if hasattr(self, "psk_physical_action"):
            self.psk_physical_action.setChecked(mode == "Physical IQ")
            self.psk_differential_action.setChecked(mode == "Differential IQ")
        if hasattr(self, "config_psk_mode"):
            self.config_psk_mode.blockSignals(True)
            self.config_psk_mode.setCurrentText(mode)
            self.config_psk_mode.blockSignals(False)
        if self._result is not None:
            self._render(self._result)

    @QtCore.Slot()
    def _reset_plot_scales(self) -> None:
        for name, plot in self._plot_widgets():
            self._reset_plot_scale(name, plot)

    def _plot_widgets(self) -> tuple[tuple[str, pg.PlotWidget], ...]:
        return (
            ("iq_power", self.power_plot),
            ("spectrum", self.spectrum_plot),
            ("fsk_modulation", self.fsk_modulation_plot),
            ("psk_modulation", self.psk_modulation_plot),
            ("fsk_symbol", self.fsk_symbol_plot),
            ("psk_symbol", self.psk_symbol_plot),
        )

    def _configure_plot_context_menus(self) -> None:
        """Use the same fixed interaction and scale menu as Generic VSA."""

        self._plot_context_actions.clear()
        for name, plot in self._plot_widgets():
            actions = install_measurement_plot_menu(
                plot,
                reset=lambda plot_name=name, target=plot: self._reset_plot_scale(
                    plot_name, target
                ),
                view_all=lambda target=plot: self._view_all_plot(target),
            )
            if actions:
                actions["reset"].setToolTip(
                    "Restore this plot's analysis-complete scale"
                )
                self._plot_context_actions[name] = actions

    def _view_all_plot(self, plot: pg.PlotWidget) -> None:
        bounds = trace_bounds(plot)
        if bounds is None:
            return
        x_min, x_max, y_min, y_max = bounds
        if (
            plot is self.fsk_symbol_plot
            and self._fsk_symbol_plot_mode == "Constellation Frequency"
        ):
            y_padding = max(1.0, 0.05 * max(y_max - y_min, 1.0))
            plot.setYRange(y_min - y_padding, y_max + y_padding, padding=0.0)
            self._set_frequency_constellation_x_lock(True)
            return
        if plot.getViewBox().state.get("aspectLocked", False) is not False:
            limit = max(
                IQ_PLANE_LIMIT,
                1.05 * max(abs(x_min), abs(x_max), abs(y_min), abs(y_max)),
            )
            plot.setRange(
                xRange=[-limit, limit],
                yRange=[-limit, limit],
                padding=0.0,
            )
            return
        view_all_traces(plot)

    def _reset_plot_scale(self, name: str, plot: pg.PlotWidget) -> None:
        ranges = self._analysis_plot_ranges.get(name)
        if ranges is None:
            return
        x_range, y_range = ranges
        plot.setRange(xRange=x_range, yRange=y_range, padding=0.0)
        if name == "fsk_symbol" and self._fsk_symbol_plot_mode == "Constellation Frequency":
            self._set_frequency_constellation_x_lock(True)

    def _capture_analysis_plot_ranges(self) -> None:
        self._analysis_plot_ranges = {
            name: (list(plot.viewRange()[0]), list(plot.viewRange()[1]))
            for name, plot in self._plot_widgets()
        }

    def _dock(self, title: str, widget: QtWidgets.QWidget) -> QtWidgets.QDockWidget:
        return make_measurement_dock(title, widget, self, object_prefix="vsa-bluetooth", closable=False)

    def _build_results(self) -> None:
        self.power_plot = make_measurement_plot("IQ Power (dBm)", "Time (ms)")
        configure_iq_power_plot(self.power_plot)
        self.power_dock = self._dock("IQ Power", self.power_plot)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.power_dock)
        self.spectrum_plot = make_measurement_plot("Magnitude (dBm)", "Frequency (MHz)")
        self.spectrum_legend = self.spectrum_plot.addLegend(offset=(10, 10))
        self.spectrum_dock = self._dock("Spectrum", self.spectrum_plot)
        self.splitDockWidget(self.power_dock, self.spectrum_dock, QtCore.Qt.Orientation.Horizontal)

        self.summary_table = _SummaryTable()
        self.summary_dock = self._dock("Result Summary", self.summary_table)
        self.splitDockWidget(self.spectrum_dock, self.summary_dock, QtCore.Qt.Orientation.Horizontal)

        self.modulation_tabs = QtWidgets.QTabWidget()
        self.fsk_modulation_plot = make_measurement_plot("Frequency (kHz)", "Time (ms)")
        self.psk_modulation_plot = make_measurement_plot("Q", "I")
        self.modulation_tabs.addTab(self.fsk_modulation_plot, "FSK - Instantaneous Frequency")
        self.modulation_tabs.addTab(self.psk_modulation_plot, "PSK - Vector")
        self.modulation_plot = self.fsk_modulation_plot
        self.modulation_dock = self._dock("Modulation", self.modulation_tabs)
        self.splitDockWidget(self.power_dock, self.modulation_dock, QtCore.Qt.Orientation.Vertical)
        self.symbol_tabs = QtWidgets.QTabWidget()
        self.fsk_symbol_plot = make_measurement_plot("Frequency (kHz)", "Symbol Index")
        self.psk_symbol_plot = make_measurement_plot("Q", "I")
        self.symbol_tabs.addTab(self.fsk_symbol_plot, "FSK")
        self.symbol_tabs.addTab(self.psk_symbol_plot, "PSK")
        self.symbol_plot = self.fsk_symbol_plot
        self.symbol_dock = self._dock("Symbol Plot", self.symbol_tabs)
        self.splitDockWidget(self.spectrum_dock, self.symbol_dock, QtCore.Qt.Orientation.Vertical)

        self.packet_tabs = QtWidgets.QTabWidget()
        self.decode_tree = _PacketAnalysisTree()
        self.decode_tree.setHeaderLabels(("Field", "Value", "Stream", "Bit Range", "Status"))
        self.decode_tree.setWordWrap(True)
        self.payload_text = QtWidgets.QPlainTextEdit(readOnly=True)
        self.packet_table = QtWidgets.QTableWidget(0, 5)
        self.packet_table.setHorizontalHeaderLabels(("#", "PHY", "Type", "Integrity", "Bits"))
        apply_dedicated_table_style(self.packet_table)
        self.packet_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.packet_table.cellClicked.connect(self._packet_row_clicked)
        self.issues_table = QtWidgets.QTableWidget(0, 4)
        self.issues_table.setHorizontalHeaderLabels(("Severity", "Code", "Message", "Bit Range"))
        apply_dedicated_table_style(self.issues_table)
        self.issues_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.air_bits_text = QtWidgets.QPlainTextEdit(readOnly=True)
        for label, widget in (("Decode", self.decode_tree), ("Payload Hex", self.payload_text), ("Packet List", self.packet_table), ("Issues", self.issues_table), ("Air Bits", self.air_bits_text)):
            self.packet_tabs.addTab(widget, label)
        self.packet_dock = self._dock("Packet Analysis", self.packet_tabs)
        self.splitDockWidget(self.summary_dock, self.packet_dock, QtCore.Qt.Orientation.Vertical)
        QtCore.QTimer.singleShot(0, self._equalize_docks)

    def _equalize_docks(self) -> None:
        self.resizeDocks([self.power_dock, self.spectrum_dock, self.summary_dock], [500] * 3, QtCore.Qt.Orientation.Horizontal)
        self.resizeDocks([self.modulation_dock, self.symbol_dock, self.packet_dock], [500] * 3, QtCore.Qt.Orientation.Horizontal)
        for upper, lower in ((self.power_dock, self.modulation_dock), (self.spectrum_dock, self.symbol_dock), (self.summary_dock, self.packet_dock)):
            self.resizeDocks([upper, lower], [450, 450], QtCore.Qt.Orientation.Vertical)

    @QtCore.Slot()
    def _protocol_changed(self) -> None:
        protocol = self.protocol_combo.currentData()
        is_le = protocol == "bluetooth.le"
        is_hdt = protocol == "bluetooth.hdt"
        self.phy_combo.blockSignals(True)
        self.phy_combo.clear()
        if is_le:
            self.phy_combo.addItems(("LE 1M", "LE 2M"))
        elif is_hdt:
            self.phy_combo.addItem("Auto (HDT2 / HDT3 / HDT4 / HDT6 / HDT7.5)")
        else:
            self.phy_combo.addItem("Auto (BR / EDR 2M / EDR 3M)")
        self.phy_combo.blockSignals(False)
        self.context_label.setText(
            "No manual packet parameters (RI and Length are decoded):"
            if is_hdt
            else "Access Address / Channel / CRC Init:"
            if is_le
            else "LAP / UAP / CLK6-1:"
        )
        for widget in (self.access_address_edit, self.channel_spin, self.crc_init_edit):
            widget.setVisible(is_le)
        for widget in (self.lap_edit, self.uap_edit, self.clock_spin):
            widget.setVisible(not is_le and not is_hdt)
        self._sync_le_profile_controls()
        self._update_derived_config()

    @QtCore.Slot()
    def _profile_changed(self) -> None:
        self._sync_le_profile_controls()
        self._update_derived_config()

    def _sync_le_profile_controls(self) -> None:
        if self.protocol_combo.currentData() != "bluetooth.le":
            for widget in (
                self.access_address_edit,
                self.crc_init_edit,
                self.whitening_check,
            ):
                widget.setEnabled(True)
            return
        rf_test = (
            self.profile_combo.currentData()
            == BluetoothAnalysisProfile.RF_PHY_TEST
        )
        if rf_test:
            self.access_address_edit.setText("71764129")
            self.crc_init_edit.setText("555555")
            self.whitening_check.setChecked(False)
        for widget in (
            self.access_address_edit,
            self.crc_init_edit,
            self.whitening_check,
        ):
            widget.setEnabled(not rf_test)

    def set_session(self, session: VSASession) -> None:
        self.stage_session(session)
        self.refresh()

    def stage_session(self, session: VSASession) -> None:
        """Remember a Generic VSA result without repainting a hidden workspace."""
        self._session = session
        self._recording = session.recording
        self.export_iq_action.setEnabled(self._recording is not None)
        if session.recording is not None and self.protocol_combo.currentData() == "bluetooth.le":
            channel = infer_le_channel(session.recording.center_frequency_hz)
            previous = self.channel_spin.blockSignals(True)
            self.channel_spin.setValue(channel)
            self.channel_spin.blockSignals(previous)

    def _last_directory(self, file_kind: str) -> str:
        stored = self._preferences.value(f"directories/{file_kind}", "", type=str)
        return stored if stored and Path(stored).is_dir() else str(Path.cwd())

    def _remember_directory(self, file_kind: str, path: str | Path) -> None:
        self._preferences.setValue(
            f"directories/{file_kind}", str(Path(path).resolve().parent)
        )
        self._preferences.sync()

    @QtCore.Slot()
    def _open_iq(self) -> None:
        if self.shutdown_busy_reason() is not None:
            self.statusBar().showMessage(
                "Stop the active Bluetooth capture or analysis before opening IQ"
            )
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open IQ Recording",
            self._last_directory("iq"),
            "IQ recordings (*.iq.tar *.npz *.npy *.cf32 *.bin);;All files (*)",
        )
        if not path:
            return
        self._remember_directory("iq", path)
        try:
            try:
                recording = FileIQSource.load(path)
            except ValueError as error:
                if "sample_rate_hz is required" not in str(error):
                    raise
                symbol_rate_hz = (
                    2_000_000.0
                    if self.protocol_combo.currentData() == "bluetooth.hdt"
                    or (
                        self.protocol_combo.currentData() == "bluetooth.le"
                        and self.phy_combo.currentText() == "LE 2M"
                    )
                    else 1_000_000.0
                )
                sample_rate_hz, accepted = QtWidgets.QInputDialog.getDouble(
                    self,
                    "IQ Sample Rate",
                    "Sample Rate (Hz)",
                    symbol_rate_hz * int(self.oversampling_combo.currentData()),
                    1.0,
                    100_000_000.0,
                    0,
                )
                if not accepted:
                    return
                recording = FileIQSource.load(
                    path,
                    sample_rate_hz=sample_rate_hz,
                    center_frequency_hz=self.center_spin.value() * 1e6,
                )
            if recording.center_frequency_hz <= 0.0:
                recording = replace(
                    recording,
                    center_frequency_hz=self.center_spin.value() * 1e6,
                )
            self.load_recording(recording)
        except Exception as error:
            QtWidgets.QMessageBox.critical(self, "IQ Import Error", str(error))

    @QtCore.Slot()
    def _export_iq_recording(self) -> None:
        export_iq_recording(self, self._recording, self._preferences)

    def load_recording(self, recording: IQRecording) -> None:
        """Load one file/capture recording and start dedicated analysis."""

        self._recording = recording
        self.export_iq_action.setEnabled(True)
        self._session = None
        self.center_spin.setValue(recording.center_frequency_hz / 1e6)
        if self.protocol_combo.currentData() == "bluetooth.le":
            previous = self.channel_spin.blockSignals(True)
            self.channel_spin.setValue(infer_le_channel(recording.center_frequency_hz))
            self.channel_spin.blockSignals(previous)
        self.statusBar().showMessage(f"IQ loaded - {recording.source}")
        self.refresh()

    def _context(self) -> dict[str, object]:
        if self.protocol_combo.currentData() == "bluetooth.le":
            try:
                crc_init = int(self.crc_init_edit.text().strip(), 16)
            except ValueError:
                crc_init = 0x555555
            return {
                "whitening_channel_index": self.channel_spin.value(),
                "crc_init": crc_init,
                "crc_enabled": True,
                "whitening_enabled": self.whitening_check.isChecked(),
            }
        try:
            uap = int(self.uap_edit.text().strip(), 16)
        except ValueError:
            uap = 0
        return {"uap": uap & 0xFF, "clock_6_1": self.clock_spin.value(), "whitening_enabled": self.whitening_check.isChecked()}

    def _meas_config_values(self) -> dict[str, object]:
        """Return the Bluetooth workspace state without Generic VSA state."""

        return {
            "profile": str(self.profile_combo.currentData()),
            "protocol": str(self.protocol_combo.currentData()),
            "phy": self.phy_combo.currentText(),
            "lap": self.lap_edit.text().strip(),
            "uap": self.uap_edit.text().strip(),
            "clock_6_1": self.clock_spin.value(),
            "le_channel": self.channel_spin.value(),
            "access_address": self.access_address_edit.text().strip(),
            "crc_init": self.crc_init_edit.text().strip(),
            "whitening": self.whitening_check.isChecked(),
            "center_mhz": self.center_spin.value(),
            "capture_ms": self.capture_length_spin.value(),
            "samples_per_symbol": int(self.oversampling_combo.currentData()),
            "rf_bandwidth_mhz": self.rf_bandwidth_spin.value(),
            "analysis_channel_enabled": self.channel_filter_check.isChecked(),
            "analysis_bandwidth_mhz": self.analysis_bandwidth_spin.value(),
            "lo_offset_enabled": self.lo_offset_check.isChecked(),
            "lo_offset_mhz": self.lo_offset_spin.value(),
            "internal_gain_db": self.internal_gain_spin.value(),
            "external_att_db": self.external_att_spin.value(),
            "acquisition_trigger_source": str(
                self.acquisition_trigger_source_combo.currentData()
            ),
            "acquisition_trigger_level_dbm": self.acquisition_trigger_level_spin.value(),
            "acquisition_trigger_slope": str(
                self.acquisition_trigger_slope_combo.currentData()
            ),
            "acquisition_trigger_offset_symbols": self.acquisition_trigger_offset_spin.value(),
            "acquisition_trigger_hysteresis_db": self.acquisition_trigger_hysteresis_spin.value(),
            "burst_search": self.iq_power_trigger_check.isChecked(),
            "burst_level_dbm": self.iq_power_trigger_level_spin.value(),
            "burst_hysteresis_db": self.iq_power_trigger_hysteresis_spin.value(),
            "burst_average_symbols": self.iq_power_trigger_average_spin.value(),
            "burst_dropout_symbols": self.iq_power_trigger_dropout_spin.value(),
            "burst_holdoff_symbols": self.iq_power_trigger_holdoff_spin.value(),
            "burst_offset_symbols": self.iq_power_trigger_offset_spin.value(),
            "burst_limit_result": self.iq_power_trigger_limit_result_check.isChecked(),
            "show_symbol_points": self._show_symbol_points,
            "symbol_density": self._symbol_density,
            "symbol_density_spread": self._symbol_density_spread.value,
            "fsk_symbol_plot": self._fsk_symbol_plot_mode,
            "psk_symbol_plot": self._psk_symbol_plot_mode,
        }

    @staticmethod
    def _set_combo_data(combo: QtWidgets.QComboBox, value: object) -> None:
        index = combo.findData(value)
        if index < 0:
            index = combo.findData(str(value))
        if index >= 0:
            combo.setCurrentIndex(index)

    def _apply_meas_config_values(self, values: dict[str, object]) -> None:
        self._set_combo_data(self.profile_combo, values.get("profile", "rf_phy_test"))
        self._set_combo_data(self.protocol_combo, values.get("protocol", "bluetooth.br_edr"))
        self._protocol_changed()
        phy = str(values.get("phy", self.phy_combo.currentText()))
        phy_index = self.phy_combo.findText(phy)
        if phy_index >= 0:
            self.phy_combo.setCurrentIndex(phy_index)

        text_controls = (
            (self.lap_edit, "lap"),
            (self.uap_edit, "uap"),
            (self.access_address_edit, "access_address"),
            (self.crc_init_edit, "crc_init"),
        )
        for control, key in text_controls:
            if key in values:
                control.setText(str(values[key]))

        numeric_controls = (
            (self.clock_spin, "clock_6_1"),
            (self.channel_spin, "le_channel"),
            (self.center_spin, "center_mhz"),
            (self.capture_length_spin, "capture_ms"),
            (self.rf_bandwidth_spin, "rf_bandwidth_mhz"),
            (self.analysis_bandwidth_spin, "analysis_bandwidth_mhz"),
            (self.lo_offset_spin, "lo_offset_mhz"),
            (self.internal_gain_spin, "internal_gain_db"),
            (self.external_att_spin, "external_att_db"),
            (self.acquisition_trigger_level_spin, "acquisition_trigger_level_dbm"),
            (self.acquisition_trigger_offset_spin, "acquisition_trigger_offset_symbols"),
            (self.acquisition_trigger_hysteresis_spin, "acquisition_trigger_hysteresis_db"),
            (self.iq_power_trigger_level_spin, "burst_level_dbm"),
            (self.iq_power_trigger_hysteresis_spin, "burst_hysteresis_db"),
            (self.iq_power_trigger_average_spin, "burst_average_symbols"),
            (self.iq_power_trigger_dropout_spin, "burst_dropout_symbols"),
            (self.iq_power_trigger_holdoff_spin, "burst_holdoff_symbols"),
            (self.iq_power_trigger_offset_spin, "burst_offset_symbols"),
        )
        for control, key in numeric_controls:
            if key in values:
                control.setValue(float(values[key]))

        self._set_combo_data(
            self.oversampling_combo, values.get("samples_per_symbol", 8)
        )
        self._set_combo_data(
            self.acquisition_trigger_source_combo,
            values.get("acquisition_trigger_source", TriggerKind.FREE_RUN.value),
        )
        self._set_combo_data(
            self.acquisition_trigger_slope_combo,
            values.get("acquisition_trigger_slope", TriggerSlope.RISING.value),
        )
        self.whitening_check.setChecked(bool(values.get("whitening", False)))
        self.iq_power_trigger_check.setChecked(bool(values.get("burst_search", True)))
        self.iq_power_trigger_limit_result_check.setChecked(
            bool(values.get("burst_limit_result", True))
        )
        self.channel_filter_check.setChecked(
            bool(values.get("analysis_channel_enabled", False))
        )
        self.lo_offset_check.setChecked(bool(values.get("lo_offset_enabled", False)))
        self._set_show_symbol_points(bool(values.get("show_symbol_points", True)))
        self._set_symbol_density(bool(values.get("symbol_density", False)))
        self._set_symbol_density_spread(
            str(
                values.get(
                    "symbol_density_spread",
                    SymbolDensitySpread.MAXIMUM.value,
                )
            )
        )
        self._set_fsk_symbol_plot_mode(
            str(values.get("fsk_symbol_plot", "Constellation Frequency"))
        )
        self._set_psk_symbol_plot_mode(
            str(values.get("psk_symbol_plot", "Physical IQ"))
        )
        self._sync_le_profile_controls()
        self._sync_acquisition_trigger_controls()
        self._sync_analysis_channel_controls()
        self._update_derived_config()

    def _save_startup_meas_config(self) -> None:
        payload = {
            "schema": _STARTUP_CONFIG_SCHEMA,
            "version": _STARTUP_CONFIG_VERSION,
            "settings": self._meas_config_values(),
        }
        self._preferences.setValue(
            _STARTUP_CONFIG_KEY,
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        )
        self._preferences.sync()

    def _restore_startup_meas_config(self) -> bool:
        raw = self._preferences.value(_STARTUP_CONFIG_KEY, "", type=str)
        if not raw:
            return False
        try:
            payload = json.loads(raw)
            if (
                payload.get("schema") != _STARTUP_CONFIG_SCHEMA
                or int(payload.get("version", 0)) != _STARTUP_CONFIG_VERSION
                or not isinstance(payload.get("settings"), dict)
            ):
                return False
            self._apply_meas_config_values(payload["settings"])
        except (TypeError, ValueError, json.JSONDecodeError):
            return False
        return True

    def set_pluto_target(self, target: str | None) -> None:
        self._pluto_target = str(target or "")
        self.device_label.setText(f"Pluto: {self._pluto_target or 'Auto'}")

    def _sync_acquisition_trigger_controls(self, _value: object = None) -> None:
        enabled = (
            self.acquisition_trigger_source_combo.currentData()
            == TriggerKind.POWER_LEVEL.value
        )
        for control in (
            self.acquisition_trigger_level_spin,
            self.acquisition_trigger_slope_combo,
            self.acquisition_trigger_offset_spin,
            self.acquisition_trigger_hysteresis_spin,
        ):
            control.setEnabled(enabled)

    def _iq_power_trigger_settings(self) -> IQPowerTriggerSettings:
        return IQPowerTriggerSettings(
            enabled=self.iq_power_trigger_check.isChecked(),
            level_dbm=self.iq_power_trigger_level_spin.value(),
            hysteresis_db=self.iq_power_trigger_hysteresis_spin.value(),
            envelope_average_symbols=self.iq_power_trigger_average_spin.value(),
            dropout_symbols=self.iq_power_trigger_dropout_spin.value(),
            holdoff_symbols=self.iq_power_trigger_holdoff_spin.value(),
            search_start_offset_symbols=self.iq_power_trigger_offset_spin.value(),
            limit_result_to_active_interval=(
                self.iq_power_trigger_limit_result_check.isChecked()
            ),
        )

    def _capture_settings(self) -> PlutoCaptureSettings:
        symbol_rate_hz = (
            2_000_000.0
            if (
                self.protocol_combo.currentData() == "bluetooth.hdt"
                or (
                    self.protocol_combo.currentData() == "bluetooth.le"
                    and self.phy_combo.currentText() == "LE 2M"
                )
            )
            else 1_000_000.0
        )
        return PlutoCaptureSettings(
            center_frequency_hz=self.center_spin.value() * 1e6,
            symbol_rate_hz=symbol_rate_hz,
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
            trigger_source=TriggerKind(
                self.acquisition_trigger_source_combo.currentData()
            ),
            trigger_level_dbm=self.acquisition_trigger_level_spin.value(),
            trigger_slope=TriggerSlope(
                self.acquisition_trigger_slope_combo.currentData()
            ),
            trigger_offset_s=(
                self.acquisition_trigger_offset_spin.value() / symbol_rate_hz
            ),
            trigger_hysteresis_db=self.acquisition_trigger_hysteresis_spin.value(),
        )

    @QtCore.Slot()
    def _toggle_capture(self) -> None:
        if self._analysis_thread is not None and self._analysis_thread.isRunning():
            self._analysis_thread.requestInterruption()
            self.capture_button.setEnabled(False)
            self.run_action.setEnabled(False)
            self.statusBar().showMessage("Stopping Bluetooth analysis...")
            return
        if self._capture_thread is not None and self._capture_thread.isRunning():
            self._capture_thread.cancel()
            self.capture_button.setEnabled(False)
            self.statusBar().showMessage("Stopping Bluetooth IQ capture...")
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
            QtWidgets.QMessageBox.critical(self, "Bluetooth Capture", str(error))
            return
        if settings.trigger_source is TriggerKind.POWER_LEVEL:
            armed_message = (
                "Waiting for Bluetooth I/Q Power trigger - "
                f"{settings.trigger_slope.value}, "
                f"{settings.trigger_level_dbm:.2f} dBm"
            )
        else:
            armed_message = (
                "Bluetooth capture armed - "
                f"{settings.requested_sample_rate_hz / 1e6:.3f} MS/s"
            )
        thread = PlutoSingleCaptureThread(
            self._pluto_source,
            settings,
            armed_message,
            self,
        )
        thread.capture_armed.connect(self.statusBar().showMessage)
        thread.capture_ready.connect(self._capture_ready)
        thread.capture_failed.connect(self._capture_failed)
        thread.capture_cancelled.connect(lambda: self.statusBar().showMessage("Bluetooth IQ capture cancelled"))
        thread.finished.connect(self._capture_stopped)
        thread.finished.connect(thread.deleteLater)
        self._capture_thread = thread
        self.capture_button.setText("Stop Capture")
        self.run_action.setText("Stop")
        self.statusBar().showMessage("Preparing Pluto for Bluetooth IQ capture...")
        thread.start()

    @QtCore.Slot(object)
    def _capture_ready(self, recording: object) -> None:
        if not isinstance(recording, IQRecording):
            self._capture_failed("capture returned an invalid IQ recording")
            return
        try:
            recording = extract_requested_analysis_channel(recording)
        except ValueError as error:
            self._capture_failed(str(error))
            return
        self.load_recording(recording)

    @QtCore.Slot(str)
    def _capture_failed(self, message: str) -> None:
        self.statusBar().showMessage(f"Bluetooth capture failed: {message}")
        if not self._shutdown_requested:
            QtWidgets.QMessageBox.critical(self, "Bluetooth Capture", message)

    @QtCore.Slot()
    def _capture_stopped(self) -> None:
        self._capture_thread = None
        stop_stream = getattr(self._pluto_source, "stop_stream", None)
        if callable(stop_stream):
            stop_stream()
        if self._analysis_thread is None or not self._analysis_thread.isRunning():
            self.capture_button.setText("Single Capture")
            self.run_action.setText("Run Single")
        self.capture_button.setEnabled(True)
        if self._shutdown_requested and self.shutdown_busy_reason() is None:
            self.shutdown_ready.emit()

    def _classic_options(self) -> dict[str, object]:
        try:
            lap = int(self.lap_edit.text().strip(), 16)
            uap = int(self.uap_edit.text().strip(), 16)
        except ValueError as error:
            raise ValueError("LAP and UAP must be hexadecimal values") from error
        return {
            "profile": self.profile_combo.currentData(),
            "lap": lap,
            "uap": uap,
            "clock_6_1": self.clock_spin.value(),
            "whitening_enabled": self.whitening_check.isChecked(),
            "result_length": max(256, int(self.capture_length_spin.value() * 1000.0)),
            "iq_power_trigger": self._iq_power_trigger_settings(),
        }

    def _le_options(self) -> dict[str, object]:
        try:
            access_address = int(self.access_address_edit.text().strip(), 16)
            crc_init = int(self.crc_init_edit.text().strip(), 16)
        except ValueError as error:
            raise ValueError("Access Address and CRC Init must be hexadecimal values") from error
        return {
            "profile": self.profile_combo.currentData(),
            "phy": self.phy_combo.currentText(),
            "access_address": access_address,
            "channel_index": self.channel_spin.value(),
            "crc_init": crc_init,
            "whitening_enabled": self.whitening_check.isChecked(),
            "result_length": max(256, int(self.capture_length_spin.value() * 2000.0)),
            "iq_power_trigger": self._iq_power_trigger_settings(),
        }

    def _hdt_options(self) -> dict[str, object]:
        return {"profile": self.profile_combo.currentData()}

    @QtCore.Slot()
    def refresh(self) -> None:
        if self._recording is None:
            self.statusBar().showMessage("Capture IQ in Bluetooth mode or load it in Generic VSA first")
            return
        if self._analysis_thread is not None and self._analysis_thread.isRunning():
            self.statusBar().showMessage("Bluetooth analysis is already running")
            return
        if self.protocol_combo.currentData() == "bluetooth.br_edr":
            try:
                options = self._classic_options()
            except ValueError as error:
                self.statusBar().showMessage(str(error))
                return
            thread = _BluetoothClassicAnalysisThread(self._recording, options, self)
            thread.analysis_ready.connect(self._classic_analysis_ready)
            thread.analysis_failed.connect(self._classic_analysis_failed)
            thread.finished.connect(self._analysis_stopped)
            thread.finished.connect(thread.deleteLater)
            self._analysis_thread = thread
            self.capture_button.setText("Stop Analysis")
            self.run_action.setText("Stop")
            self.statusBar().showMessage("Analyzing Classic header and detecting BR / EDR PHY...")
            thread.start()
            return
        if self.protocol_combo.currentData() == "bluetooth.hdt":
            thread = _BluetoothHDTAnalysisThread(
                self._recording, self._hdt_options(), self
            )
            thread.analysis_ready.connect(self._classic_analysis_ready)
            thread.analysis_failed.connect(self._classic_analysis_failed)
            thread.finished.connect(self._analysis_stopped)
            thread.finished.connect(thread.deleteLater)
            self._analysis_thread = thread
            self.capture_button.setText("Stop Analysis")
            self.run_action.setText("Stop")
            self.statusBar().showMessage(
                "Synchronizing HDT and decoding Rate Indicator / Payload Length..."
            )
            thread.start()
            return
        try:
            options = self._le_options()
        except ValueError as error:
            self.statusBar().showMessage(str(error))
            return
        thread = _BluetoothLEAnalysisThread(self._recording, options, self)
        thread.analysis_ready.connect(self._classic_analysis_ready)
        thread.analysis_failed.connect(self._classic_analysis_failed)
        thread.finished.connect(self._analysis_stopped)
        thread.finished.connect(thread.deleteLater)
        self._analysis_thread = thread
        self.capture_button.setText("Stop Analysis")
        self.run_action.setText("Stop")
        self.statusBar().showMessage(f"Synchronizing and analyzing {self.phy_combo.currentText()} packet...")
        thread.start()

    @QtCore.Slot(object)
    def _classic_analysis_ready(self, result: object) -> None:
        if isinstance(result, BluetoothDedicatedResult):
            results = (result,)
        elif isinstance(result, tuple) and result and all(
            isinstance(item, BluetoothDedicatedResult) for item in result
        ):
            results = result
        else:
            self._classic_analysis_failed("invalid Bluetooth analysis result")
            return
        self._results = results
        self._selected_result_index = 0
        result = results[0]
        self._result = result
        session = result.metadata.get("analysis_session")
        if isinstance(session, VSASession):
            self._session = session
        self._render(result)
        aggregate_status = result.metadata.get("hdt_rms_evm_aggregate_status")
        aggregate_suffix = (
            f" - {aggregate_status}" if aggregate_status is not None else ""
        )
        self.statusBar().showMessage(
            f"Bluetooth analysis complete - {result.packet.phy_name} / "
            f"{result.packet.packet_type or 'packet'} - {len(results)} packet(s)"
            f"{aggregate_suffix}"
        )

    @QtCore.Slot(int)
    def _select_result(self, step: int) -> None:
        if not self._results:
            return
        target = self._selected_result_index + int(step)
        if not 0 <= target < len(self._results):
            return
        self._selected_result_index = target
        selected = self._results[target]
        sessions: list[VSASession] = []
        for key in ("br_analysis_session", "analysis_session"):
            candidate = selected.metadata.get(key)
            if isinstance(candidate, VSASession) and all(
                candidate is not current for current in sessions
            ):
                sessions.append(candidate)
        for session in sessions:
            session.generate_display_products()
        primary = selected.metadata.get("analysis_session")
        if isinstance(primary, VSASession):
            display_result = (
                primary.carrier_corrected_pattern_range_result
                or primary.pattern_range_result
                or primary.result
            )
            if display_result is not None and display_result is not selected.vsa_result:
                selected = replace(selected, vsa_result=display_result)
                mutable_results = list(self._results)
                mutable_results[target] = selected
                self._results = tuple(mutable_results)
        self._result = selected
        session = self._result.metadata.get("analysis_session")
        if isinstance(session, VSASession):
            self._session = session
        self._render(self._result)
        aggregate_status = self._result.metadata.get(
            "hdt_rms_evm_aggregate_status"
        )
        aggregate_suffix = (
            f" - {aggregate_status}" if aggregate_status is not None else ""
        )
        self.statusBar().showMessage(
            f"Selected Bluetooth packet {target + 1}/{len(self._results)}"
            f"{aggregate_suffix}"
        )

    @QtCore.Slot(str)
    def _classic_analysis_failed(self, message: str) -> None:
        self.statusBar().showMessage(f"Bluetooth analysis failed: {message}")

    @QtCore.Slot()
    def _analysis_stopped(self) -> None:
        self._analysis_thread = None
        self.capture_button.setEnabled(True)
        self.run_action.setEnabled(True)
        self.capture_button.setText("Single Capture")
        self.run_action.setText("Run Single")
        if self._shutdown_requested and self.shutdown_busy_reason() is None:
            self.shutdown_ready.emit()

    def shutdown_busy_reason(self) -> str | None:
        if self._capture_thread is not None and self._capture_thread.isRunning():
            return "Bluetooth IQ capture is running"
        if self._analysis_thread is not None and self._analysis_thread.isRunning():
            return "Bluetooth analysis is running"
        return None

    def request_shutdown(self) -> None:
        self._shutdown_requested = True
        if self._capture_thread is not None and self._capture_thread.isRunning():
            self._capture_thread.cancel()
        if self._analysis_thread is not None and self._analysis_thread.isRunning():
            self._analysis_thread.requestInterruption()

    def finalize_shutdown(self) -> None:
        self._save_startup_meas_config()
        if self._owns_pluto_source:
            self._pluto_source.close()

    def _render(self, result: BluetoothDedicatedResult) -> None:
        modulation_tab_index = self.modulation_tabs.currentIndex()
        symbol_tab_index = self.symbol_tabs.currentIndex()
        vsa = result.vsa_result
        recording = self._recording
        if recording is None:
            return
        session = result.metadata.get("analysis_session")
        br_session = result.metadata.get("br_analysis_session")
        recording_sample_offset = int(
            result.metadata.get("recording_sample_offset", 0)
        )
        analysis_sample_offset = int(
            result.metadata.get("analysis_sample_offset", recording_sample_offset)
        )
        is_hdt = result.packet.protocol_id == "bluetooth.hdt"
        hdt_plot_data = result.metadata.get("hdt_plot_data")
        if not isinstance(hdt_plot_data, HDTPlotData):
            hdt_plot_data = None
        hdt_evm = (
            hdt_plot_data.evm
            if hdt_plot_data is not None
            else result.metadata.get("hdt_evm_result")
        )
        if not isinstance(hdt_evm, HDTEVMResult):
            hdt_evm = None
        is_psk = (
            isinstance(session, VSASession)
            and session.signal is not None
            and session.signal.modulation.family.uses_iq_constellation
        )
        if is_hdt:
            payload_modulation = str(
                next(
                    (
                        item.display
                        for item in result.metrics
                        if item.metric_id == "payload_modulation"
                    ),
                    "Payload",
                )
            )
            payload_name = _hdt_modulation_name(payload_modulation)
            payload_is_qpsk = payload_name == "QPSK"
            self.modulation_tabs.setTabText(0, "QPSK Header")
            self.modulation_tabs.setTabText(1, f"{payload_name} Payload")
            self.symbol_tabs.setTabText(0, "QPSK Header")
            self.symbol_tabs.setTabText(1, f"{payload_name} Payload")
        else:
            payload_name = "Payload"
            payload_is_qpsk = False
            self.modulation_tabs.setTabText(0, "FSK - Instantaneous Frequency")
            self.modulation_tabs.setTabText(1, "PSK - Vector")
            self.symbol_tabs.setTabText(0, "FSK")
            self.symbol_tabs.setTabText(1, "PSK")
        full_result = result.metadata.get("capture_result")
        if not isinstance(full_result, VSAAnalysisResult):
            full_result = session.result if isinstance(session, VSASession) else None
            if (
                is_psk
                and isinstance(br_session, VSASession)
                and br_session.result is not None
            ):
                full_result = br_session.result
        self.power_plot.clear()
        power_result = full_result or vsa
        if is_hdt:
            power_time_s, _power_dbfs, power_dbm = capture_power_traces(recording)
            power_time_offset_s = 0.0
        else:
            power_time_s = np.asarray(power_result.time_s, dtype=np.float64)
            power_dbm = np.asarray(power_result.power_dbm, dtype=np.float64)
            power_time_offset_s = 0.0
        self.power_plot.plot(
            (power_time_s + power_time_offset_s) * 1e3,
            limit_iq_power_display_dbm(power_dbm),
            pen=_TRACE,
        )
        displayed_power_dbm = limit_iq_power_display_dbm(power_dbm)
        finite_power_dbm = displayed_power_dbm[np.isfinite(displayed_power_dbm)]
        if finite_power_dbm.size:
            power_min = float(np.min(finite_power_dbm))
            power_max = float(np.max(finite_power_dbm))
            power_padding = max(1.0, 0.05 * max(power_max - power_min, 1.0))
            self.power_plot.setYRange(
                power_min - power_padding,
                power_max + power_padding,
                padding=0.0,
            )
        selected_ranges: list[tuple[float, float]] = []
        power_symbol_times_ms: list[np.ndarray] = []
        if is_hdt and hdt_plot_data is not None:
            packet_start_sample, packet_stop_sample = (
                hdt_plot_data.packet_sample_range
            )
            result_start_sample, result_stop_sample = (
                hdt_plot_data.payload_evm_sample_range
            )
            training_start_sample, training_stop_sample = (
                hdt_plot_data.training_sample_range
            )
            selected_ranges.append(
                (
                    packet_start_sample / recording.sample_rate_hz * 1e3,
                    packet_stop_sample / recording.sample_rate_hz * 1e3,
                )
            )
            add_result_range_overlay(
                self.power_plot,
                result_start_ms=(
                    result_start_sample / recording.sample_rate_hz * 1e3
                ),
                result_stop_ms=(
                    result_stop_sample / recording.sample_rate_hz * 1e3
                ),
                pattern_start_ms=(
                    training_start_sample / recording.sample_rate_hz * 1e3
                ),
                pattern_stop_ms=(
                    training_stop_sample / recording.sample_rate_hz * 1e3
                ),
                label="Packet Start",
            )
            packet_end_line = pg.InfiniteLine(
                pos=packet_stop_sample / recording.sample_rate_hz * 1e3,
                angle=90,
                movable=False,
                pen=pg.mkPen(95, 100, 108, 150),
                label="Packet End",
                labelOpts={"position": 0.92, "color": (115, 120, 128)},
            )
            self.power_plot.addItem(packet_end_line)
            if hdt_evm is not None:
                power_symbol_times_ms.append(
                    np.concatenate(
                        (
                            hdt_evm.header_symbol_sample_positions,
                            hdt_evm.payload_symbol_sample_positions,
                        )
                    )
                    / recording.sample_rate_hz
                    * 1e3
                )
        analyses: list[VSASession] = []
        for candidate in (() if is_hdt else (br_session, session)):
            if isinstance(candidate, VSASession) and all(
                candidate is not existing for existing in analyses
            ):
                analyses.append(candidate)
        for analysis in analyses:
            if not isinstance(analysis, VSASession) or analysis.pattern_result is None:
                continue
            pattern = analysis.pattern_result
            offset = (
                analysis_sample_offset
                if analysis is session
                else recording_sample_offset
            )
            if (
                result.packet.protocol_id == "bluetooth.le"
                and analysis is session
            ):
                start_ms = (
                    float(result.metadata["packet_start_sample"])
                    / recording.sample_rate_hz
                    * 1e3
                )
                stop_ms = (
                    float(result.metadata["packet_stop_sample"])
                    / recording.sample_rate_hz
                    * 1e3
                )
            else:
                start_ms = (
                    (offset + pattern.result_start_sample)
                    / recording.sample_rate_hz
                    * 1e3
                )
                stop_ms = (
                    (offset + pattern.result_stop_sample)
                    / recording.sample_rate_hz
                    * 1e3
                )
            selected_ranges.append((start_ms, stop_ms))
            pattern_start_ms = (
                offset + pattern.pattern_start_sample
            ) / recording.sample_rate_hz * 1e3
            pattern_stop_ms = (
                offset / recording.sample_rate_hz + pattern.pattern_stop_time_s
            ) * 1e3
            add_result_range_overlay(
                self.power_plot,
                result_start_ms=start_ms,
                result_stop_ms=stop_ms,
                pattern_start_ms=pattern_start_ms,
                pattern_stop_ms=pattern_stop_ms,
            )
            symbol_time_s = np.asarray(pattern.symbol_time_s, dtype=np.float64)
            if result.packet.protocol_id == "bluetooth.le" and analysis is session:
                symbol_time_s = symbol_time_s[
                    : int(result.metadata.get("packet_symbol_count", symbol_time_s.size))
                ]
            if analysis is session and is_psk and not is_hdt:
                devm_centers = np.asarray(
                    result.metadata.get("edr_devm_symbol_center_samples", ()),
                    dtype=np.float64,
                )
                if devm_centers.size and symbol_time_s.size:
                    symbol_time_s = np.array(symbol_time_s, copy=True)
                    coordinate_count = min(
                        symbol_time_s.size, devm_centers.size
                    )
                    symbol_time_s[:coordinate_count] = (
                        devm_centers[:coordinate_count] - offset
                    ) / recording.sample_rate_hz
                reference_center = result.metadata.get(
                    "edr_reference_symbol_center_sample"
                )
                if reference_center is not None:
                    symbol_time_s = np.concatenate(
                        (
                            np.asarray(
                                [
                                    (
                                        float(reference_center) - offset
                                    )
                                    / recording.sample_rate_hz
                                ]
                            ),
                            symbol_time_s,
                        )
                    )
            if symbol_time_s.size:
                power_symbol_times_ms.append(
                    (symbol_time_s + offset / recording.sample_rate_hz) * 1e3
                )
        if self._show_symbol_points and power_symbol_times_ms:
            marker_time_ms = np.concatenate(power_symbol_times_ms)
            power_time_ms = (power_time_s + power_time_offset_s) * 1e3
            count = min(power_time_ms.size, power_dbm.size)
            if count:
                valid = (
                    (marker_time_ms >= power_time_ms[0])
                    & (marker_time_ms <= power_time_ms[count - 1])
                )
                marker_time_ms = marker_time_ms[valid]
                plot_trace_symbol_points(
                    self.power_plot,
                    marker_time_ms,
                    np.interp(
                        marker_time_ms,
                        power_time_ms[:count],
                        power_dbm[:count],
                    ),
                )
        if selected_ranges:
            start_ms = min(value[0] for value in selected_ranges)
            stop_ms = max(value[1] for value in selected_ranges)
            margin = max((stop_ms - start_ms) * 0.10, 1e-6)
            self.power_plot.setXRange(start_ms - margin, stop_ms + margin, padding=0.0)
        self.spectrum_plot.clear()
        br_vsa = None
        if isinstance(br_session, VSASession):
            br_vsa = br_session.carrier_corrected_pattern_range_result or br_session.pattern_range_result
        if br_vsa is not None:
            self.spectrum_plot.plot((br_vsa.spectrum_frequency_hz + recording.center_frequency_hz) / 1e6, br_vsa.spectrum_dbm, pen=_TRACE, name="FSK")
        if is_hdt:
            header_frequency_hz = np.asarray(
                result.metadata.get("hdt_header_spectrum_frequency_hz", ()),
                dtype=np.float64,
            )
            header_spectrum_dbm = np.asarray(
                result.metadata.get("hdt_header_spectrum_dbm", ()),
                dtype=np.float64,
            )
            if header_frequency_hz.size and header_spectrum_dbm.size:
                self.spectrum_plot.plot(
                    header_frequency_hz / 1e6,
                    header_spectrum_dbm,
                    pen=_TRACE,
                    name="QPSK Header",
                )
            payload_frequency_hz = np.asarray(
                result.metadata.get("hdt_payload_spectrum_frequency_hz", ()),
                dtype=np.float64,
            )
            payload_spectrum_dbm = np.asarray(
                result.metadata.get("hdt_payload_spectrum_dbm", ()),
                dtype=np.float64,
            )
            if payload_frequency_hz.size and payload_spectrum_dbm.size:
                self.spectrum_plot.plot(
                    payload_frequency_hz / 1e6,
                    payload_spectrum_dbm,
                    pen="#00ffff",
                    name=f"{payload_name} Payload",
                )
        elif is_psk or br_vsa is None:
            vector_name = (
                "PSK" if is_psk else "FSK"
            )
            self.spectrum_plot.plot((vsa.spectrum_frequency_hz + recording.center_frequency_hz) / 1e6, vsa.spectrum_dbm, pen="#00ffff" if is_psk else _TRACE, name=vector_name)

        self.fsk_modulation_plot.clear()
        self.psk_modulation_plot.clear()
        self.fsk_symbol_plot.clear()
        self.psk_symbol_plot.clear()
        self._configure_fsk_modulation_plot(iq_plane=is_hdt)
        fsk_session = br_session if is_psk else session
        fsk_vsa = br_vsa if is_psk and br_vsa is not None else vsa
        fsk_pattern = (
            fsk_session.pattern_result
            if isinstance(fsk_session, VSASession)
            else None
        )
        fsk_display_result = (
            fsk_session.result
            if isinstance(fsk_session, VSASession)
            else None
        )
        fsk_signal = (
            fsk_session.signal
            if isinstance(fsk_session, VSASession)
            else None
        )
        fsk_display_data: FSKDisplayData | None = None
        if fsk_display_result is not None and fsk_signal is not None:
            fsk_analysis_rate_hz = float(
                fsk_display_result.metadata.get(
                    "analysis_sample_rate_hz",
                    fsk_session.recording.sample_rate_hz,
                )
            )
            display_start_time_s = (
                fsk_pattern.result_start_time_s
                if fsk_pattern is not None
                else None
            )
            display_stop_time_s = (
                fsk_pattern.result_stop_time_s
                if fsk_pattern is not None
                else None
            )
            if (
                result.packet.protocol_id == "bluetooth.le"
                and fsk_session is session
            ):
                display_start_time_s = (
                    float(result.metadata["packet_start_sample"])
                    - recording_sample_offset
                ) / recording.sample_rate_hz
                display_stop_time_s = (
                    float(result.metadata["packet_stop_sample"])
                    - recording_sample_offset
                ) / recording.sample_rate_hz
            measurement_trace = result.metadata.get("fsk_measurement_trace")
            if isinstance(measurement_trace, BluetoothFMMeasurementTrace):
                trace_time_s = measurement_trace.time_s
                trace_frequency_hz = measurement_trace.frequency_hz
                if display_start_time_s is not None and display_stop_time_s is not None:
                    guard_s = 16.0 / float(fsk_signal.symbol_rate_hz)
                    trace_mask = (
                        (trace_time_s >= float(display_start_time_s) - guard_s)
                        & (trace_time_s <= float(display_stop_time_s) + guard_s)
                    )
                    fsk_time_s = trace_time_s[trace_mask]
                    fsk_frequency_hz = trace_frequency_hz[trace_mask]
                else:
                    fsk_time_s = trace_time_s
                    fsk_frequency_hz = trace_frequency_hz
            else:
                # Compatibility fallback for results created before the RF
                # measurement trace became part of the dedicated result model.
                fsk_frequency_hz, fsk_time_s = prepare_fsk_display_frequency(
                    fsk_display_result.iq,
                    sample_rate_hz=fsk_analysis_rate_hz,
                    symbol_rate_hz=fsk_signal.symbol_rate_hz,
                    gaussian_bt=(
                        fsk_signal.filter_parameter
                        if fsk_session.demodulation.measurement_filter
                        is MeasurementFilterMode.AUTO
                        else None
                    ),
                    result_start_time_s=display_start_time_s,
                    result_stop_time_s=display_stop_time_s,
                )
            symbol_time_s = np.asarray(
                fsk_pattern.symbol_time_s if fsk_pattern is not None else (),
                dtype=np.float64,
            )
            if (
                result.packet.protocol_id == "bluetooth.le"
                and fsk_session is session
            ):
                symbol_time_s = symbol_time_s[
                    : int(result.metadata.get("packet_symbol_count", symbol_time_s.size))
                ]
            fsk_display_data = build_fsk_display_data(
                fsk_frequency_hz,
                fsk_time_s,
                symbol_time_s,
                frequency_offset_hz=(
                    fsk_pattern.carrier_frequency_offset_hz
                    if fsk_pattern is not None
                    else 0.0
                ),
                frequency_drift_hz_per_s=(
                    fsk_pattern.carrier_frequency_drift_hz_per_s
                    if fsk_pattern is not None
                    else 0.0
                ),
                reference_time_s=(
                    fsk_pattern.carrier_reference_time_s
                    if fsk_pattern is not None
                    else 0.0
                ),
            )
            self.fsk_modulation_plot.plot(
                (
                    fsk_display_data.time_s
                    + recording_sample_offset / recording.sample_rate_hz
                )
                * 1e3,
                fsk_display_data.corrected_frequency_hz / 1e3,
                pen=_TRACE,
            )
            if self._show_symbol_points and fsk_pattern is not None:
                symbol_count = min(
                    fsk_display_data.symbol_time_s.size,
                    fsk_display_data.symbol_frequency_hz.size,
                )
                plot_trace_symbol_points(
                    self.fsk_modulation_plot,
                    (
                        fsk_display_data.symbol_time_s[:symbol_count]
                        + recording_sample_offset / recording.sample_rate_hz
                    )
                    * 1e3,
                    fsk_display_data.symbol_frequency_hz[:symbol_count] / 1e3,
                )
        else:
            fsk_count = min(
                fsk_vsa.time_s.size,
                fsk_vsa.instantaneous_frequency_hz.size,
            )
            self.fsk_modulation_plot.plot(
                (
                    fsk_vsa.time_s[:fsk_count]
                    + recording_sample_offset / recording.sample_rate_hz
                )
                * 1e3,
                fsk_vsa.instantaneous_frequency_hz[:fsk_count] / 1e3,
                pen=_TRACE,
            )
        if fsk_pattern is not None:
            fsk_start_ms = (
                recording_sample_offset + fsk_pattern.result_start_sample
            ) / recording.sample_rate_hz * 1e3
            fsk_stop_ms = (
                recording_sample_offset + fsk_pattern.result_stop_sample
            ) / recording.sample_rate_hz * 1e3
            fsk_margin_ms = max((fsk_stop_ms - fsk_start_ms) * 0.10, 1e-6)
            self.fsk_modulation_plot.setXRange(
                fsk_start_ms - fsk_margin_ms,
                fsk_stop_ms + fsk_margin_ms,
                padding=0.0,
            )
        measured_frequency_hz = (
            fsk_display_data.symbol_frequency_hz
            if fsk_display_data is not None
            else np.real(np.asarray(fsk_vsa.measured_symbols, dtype=np.complex128))
        )
        if is_hdt:
            header_symbols = np.asarray(
                hdt_evm.header_corrected_symbols if hdt_evm is not None else (),
                dtype=np.complex128,
            )
            header_trajectory = np.asarray(
                hdt_evm.header_corrected_waveform if hdt_evm is not None else (),
                dtype=np.complex128,
            )
            self.fsk_modulation_plot.clear()
            self.fsk_modulation_plot.plot(
                header_trajectory.real, header_trajectory.imag, pen=_TRACE
            )
            if self._show_symbol_points and header_symbols.size:
                plot_trace_symbol_points(
                    self.fsk_modulation_plot,
                    header_symbols.real,
                    header_symbols.imag,
                )
            self.fsk_symbol_plot.setYLink(None)
            self._set_frequency_constellation_x_lock(False)
            self.fsk_symbol_plot.showAxis("bottom", True)
            self.fsk_symbol_plot.setLabel("bottom", "I")
            self.fsk_symbol_plot.setLabel("left", "Q")
            self.fsk_symbol_plot.setAspectLocked(True, 1.0)
            self._plot_symbol_vectors(
                self.fsk_symbol_plot,
                _hdt_pi4_qpsk_display_symbols(header_symbols),
            )
            self._plot_unit_circle(self.fsk_symbol_plot)
            self._set_iq_plane_range(self.fsk_symbol_plot)
            self._set_iq_plane_range(self.fsk_modulation_plot)
        elif self._fsk_symbol_plot_mode == "Constellation Frequency":
            self.fsk_symbol_plot.setAspectLocked(False)
            self.fsk_symbol_plot.setYLink(self.fsk_modulation_plot)
            self._set_frequency_constellation_x_lock(True)
            self.fsk_symbol_plot.showAxis("bottom", False)
            self.fsk_symbol_plot.setLabel("bottom", "")
            self.fsk_symbol_plot.setLabel("left", "Frequency (kHz)")
            limit_khz = max(
                1.0,
                1.5 * float(fsk_signal.frequency_deviation_hz or 0.0) / 1e3,
            )
            plot_frequency_symbol_distribution(
                self.fsk_symbol_plot,
                measured_frequency_hz / 1e3,
                y_limit_khz=limit_khz,
                density=self._symbol_density,
                density_spread=self._symbol_density_spread,
            )
            self.fsk_symbol_plot.setXRange(
                -_FREQUENCY_CONSTELLATION_X_LIMIT,
                _FREQUENCY_CONSTELLATION_X_LIMIT,
                padding=0.0,
            )
            self.fsk_symbol_plot.setYRange(-limit_khz, limit_khz, padding=0.0)
        else:
            self.fsk_symbol_plot.setYLink(None)
            self._set_frequency_constellation_x_lock(False)
            self.fsk_symbol_plot.showAxis("bottom", True)
            self.fsk_symbol_plot.setLabel("bottom", "I")
            self.fsk_symbol_plot.setLabel("left", "Q")
            self.fsk_symbol_plot.setAspectLocked(True, 1.0)
            symbol_rate_hz = (
                float(br_session.signal.symbol_rate_hz)
                if isinstance(br_session, VSASession) and br_session.signal is not None
                else 1_000_000.0
            )
            phase_symbols = np.exp(
                2j * np.pi * measured_frequency_hz / max(symbol_rate_hz, 1.0)
            )
            self._plot_symbol_vectors(self.fsk_symbol_plot, phase_symbols)
            self._plot_unit_circle(self.fsk_symbol_plot)
            self._set_iq_plane_range(self.fsk_symbol_plot)

        if not is_hdt and fsk_signal is not None:
            # The display FM demodulator deliberately includes guard samples
            # around the result range.  Phase differences in a low-power guard
            # can be arbitrarily large and must not control the normal Y scale.
            # Match Generic VSA's nominal-deviation scale; View All remains
            # available when inspection of the complete trace is wanted.
            deviation_hz = abs(float(fsk_signal.frequency_deviation_hz or 0.0))
            if deviation_hz > 0.0:
                limit_khz = 1.5 * deviation_hz / 1e3
                self.fsk_modulation_plot.setYRange(
                    -limit_khz,
                    limit_khz,
                    padding=0.0,
                )

        if is_psk and is_hdt and hdt_evm is not None:
            trajectory = np.asarray(
                hdt_evm.payload_corrected_waveform, dtype=np.complex128
            )
            symbols = np.asarray(
                hdt_evm.payload_corrected_symbols, dtype=np.complex128
            )
            self.psk_modulation_plot.setAspectLocked(True, 1.0)
            self.psk_modulation_plot.setDownsampling(auto=False)
            self.psk_modulation_plot.setClipToView(False)
            self.psk_modulation_plot.plot(
                trajectory.real, trajectory.imag, pen=_TRACE
            )
            if self._show_symbol_points and symbols.size:
                plot_trace_symbol_points(
                    self.psk_modulation_plot, symbols.real, symbols.imag
                )
            self.psk_symbol_plot.setAspectLocked(True, 1.0)
            self._plot_symbol_vectors(
                self.psk_symbol_plot,
                (
                    _hdt_pi4_qpsk_display_symbols(symbols)
                    if payload_is_qpsk
                    else symbols
                ),
            )
            self._plot_unit_circle(self.psk_symbol_plot)
            self._set_iq_plane_range(self.psk_symbol_plot)
        elif is_psk:
            psk_pattern = session.pattern_result
            display_result = session.carrier_corrected_result or session.result
            if display_result is None or psk_pattern is None or session.signal is None:
                return
            analysis_sample_rate_hz = float(
                display_result.metadata.get(
                    "analysis_sample_rate_hz", session.recording.sample_rate_hz
                )
            )
            processed_iq, processed_time_s = prepare_psk_display_waveform(
                display_result.iq,
                sample_rate_hz=analysis_sample_rate_hz,
                symbol_rate_hz=session.signal.symbol_rate_hz,
                tx_filter=session.signal.tx_filter,
                filter_parameter=session.signal.filter_parameter,
                apply_measurement_filter=(
                    session.demodulation.measurement_filter
                    is MeasurementFilterMode.AUTO
                ),
                result_start_time_s=psk_pattern.result_start_time_s,
                result_stop_time_s=psk_pattern.result_stop_time_s,
            )
            psk_symbol_time_s = np.asarray(
                psk_pattern.symbol_time_s, dtype=np.float64
            )
            devm_centers = np.asarray(
                result.metadata.get("edr_devm_symbol_center_samples", ()),
                dtype=np.float64,
            )
            if devm_centers.size and psk_symbol_time_s.size:
                psk_symbol_time_s = np.array(psk_symbol_time_s, copy=True)
                coordinate_count = min(
                    psk_symbol_time_s.size, devm_centers.size
                )
                psk_symbol_time_s[:coordinate_count] = (
                    devm_centers[:coordinate_count] - analysis_sample_offset
                ) / recording.sample_rate_hz
            reference_center = result.metadata.get(
                "edr_reference_symbol_center_sample"
            )
            if reference_center is not None:
                psk_symbol_time_s = np.concatenate(
                    (
                        np.asarray(
                            [
                                (
                                    float(reference_center)
                                    - analysis_sample_offset
                                )
                                / recording.sample_rate_hz
                            ]
                        ),
                        psk_symbol_time_s,
                    )
                )
            trajectory, physical_symbol_iq, symbols = normalized_psk_display(
                processed_iq,
                processed_time_s,
                psk_symbol_time_s,
                modulation=session.signal.modulation,
                differential_symbols=psk_pattern.measured_symbols,
                physical=self._psk_symbol_plot_mode == "Physical IQ",
            )
            in_result_range = (
                (processed_time_s >= psk_pattern.result_start_time_s)
                & (processed_time_s < psk_pattern.result_stop_time_s)
            )
            trajectory = trajectory[in_result_range]
            self.psk_modulation_plot.setAspectLocked(True, 1.0)
            # A dedicated packet result is bounded to one decoded packet and
            # is small enough to draw sample-for-sample.  pyqtgraph's generic
            # auto-downsampling can skip the RRC transition samples and turn
            # the vector trajectory into misleading straight chords.
            self.psk_modulation_plot.setDownsampling(auto=False)
            self.psk_modulation_plot.setClipToView(False)
            self.psk_modulation_plot.plot(np.real(trajectory), np.imag(trajectory), pen=_TRACE)
            if self._show_symbol_points and physical_symbol_iq.size:
                plot_trace_symbol_points(
                    self.psk_modulation_plot,
                    physical_symbol_iq.real,
                    physical_symbol_iq.imag,
                )
            self.psk_symbol_plot.setAspectLocked(True, 1.0)
            self._plot_symbol_vectors(self.psk_symbol_plot, symbols)
            self._plot_unit_circle(self.psk_symbol_plot)
            self._set_iq_plane_range(self.psk_symbol_plot)
        self.modulation_tabs.setTabVisible(1, is_psk)
        self.symbol_tabs.setTabVisible(1, is_psk)
        self.modulation_tabs.setCurrentIndex(
            modulation_tab_index if is_psk and modulation_tab_index == 1 else 0
        )
        self.symbol_tabs.setCurrentIndex(
            symbol_tab_index if is_psk and symbol_tab_index == 1 else 0
        )
        self._render_summary(result)
        self._render_packet(result)
        self._capture_analysis_plot_ranges()

    def _configure_fsk_modulation_plot(self, *, iq_plane: bool) -> None:
        """Restore the axis contract when the first tab changes PHY role."""

        if iq_plane:
            self.fsk_modulation_plot.setLabel("bottom", "I")
            self.fsk_modulation_plot.setLabel("left", "Q")
            self.fsk_modulation_plot.setAspectLocked(True, 1.0)
            self.fsk_modulation_plot.setDownsampling(auto=False)
            self.fsk_modulation_plot.setClipToView(False)
            return
        self.fsk_modulation_plot.setAspectLocked(False)
        self.fsk_modulation_plot.setLabel("bottom", "Time (ms)")
        self.fsk_modulation_plot.setLabel("left", "Frequency (kHz)")
        self.fsk_modulation_plot.setDownsampling(auto=True, mode="peak")
        self.fsk_modulation_plot.setClipToView(True)

    @staticmethod
    def _set_iq_plane_range(plot: pg.PlotWidget) -> None:
        set_iq_plane_range(plot)

    @staticmethod
    def _plot_unit_circle(plot: pg.PlotWidget) -> None:
        plot_unit_circle(plot)

    def _plot_frequency_symbols(
        self, plot: pg.PlotWidget, frequency_khz: np.ndarray
    ) -> None:
        values = np.asarray(frequency_khz, dtype=np.float64)
        finite = values[np.isfinite(values)]
        y_limit_khz = max(
            1.0,
            1.2 * float(np.max(np.abs(finite))) if finite.size else 1.0,
        )
        plot_frequency_symbol_distribution(
            plot,
            values,
            y_limit_khz=y_limit_khz,
            density=self._symbol_density,
            density_spread=self._symbol_density_spread,
        )

    def _plot_symbol_vectors(
        self, plot: pg.PlotWidget, symbols: np.ndarray
    ) -> None:
        plot_complex_symbol_distribution(
            plot,
            symbols,
            density=self._symbol_density,
            density_spread=self._symbol_density_spread,
        )

    def _render_summary(self, result: BluetoothDedicatedResult) -> None:
        rows: list[tuple[str | None, object | None]] = []
        previous_group: str | None = None
        for summary_row in result.summary_rows:
            if summary_row.section != previous_group:
                rows.append((summary_row.section, None))
                previous_group = summary_row.section
            rows.append((None, summary_row))
        self.summary_table.clearSpans()
        self.summary_table.setRowCount(len(rows))
        for row, (group, metric) in enumerate(rows):
            if group is not None:
                item = QtWidgets.QTableWidgetItem(group)
                font = item.font()
                font.setBold(True)
                item.setFont(font)
                item.setBackground(QtGui.QColor("#353535"))
                self.summary_table.setItem(row, 0, item)
                self.summary_table.setSpan(row, 0, 1, 4)
                continue
            if metric is None:
                continue
            values = (metric.test_item, metric.value, metric.limit, metric.result)
            for column, value in enumerate(values):
                self.summary_table.setItem(
                    row, column, QtWidgets.QTableWidgetItem(str(value))
                )
            result_color = dedicated_status_color(metric.result)
            if result_color is not None:
                self.summary_table.item(row, 3).setForeground(
                    QtGui.QBrush(result_color)
                )
        self.summary_table.resizeRowsToContents()

    @staticmethod
    def _field_bit_range(field: PacketField, bit_offset: int) -> str:
        bit_count = int(field.stop_bit) - int(field.start_bit)
        if bit_count <= 0:
            return "N/A"
        start = int(field.start_bit) - int(bit_offset)
        stop = int(field.stop_bit) - int(bit_offset) - 1
        return str(start) if start == stop else f"{start}\N{EN DASH}{stop}"

    def _tree_item(
        self,
        field: PacketField,
        *,
        stream: str,
        bit_offset: int,
    ) -> QtWidgets.QTreeWidgetItem:
        value = str(field.value)
        if field.field_id in {"payload", "payload_body"}:
            compact = "".join(value.split())
            value = "\n".join(
                compact[start : start + 20]
                for start in range(0, len(compact), 20)
            )
        if field.field_id == "payload" and field.children:
            value = "\N{EM DASH}"
        bit_range = self._field_bit_range(field, bit_offset)
        item = QtWidgets.QTreeWidgetItem(
            (field.name, value, stream, bit_range, field.status.value)
        )
        item.setTextAlignment(
            1,
            QtCore.Qt.AlignmentFlag.AlignLeft
            | QtCore.Qt.AlignmentFlag.AlignVCenter,
        )
        color = dedicated_status_color(field.status)
        if color:
            item.setForeground(4, QtGui.QBrush(color))
        for column, text in enumerate(
            (
                field.name,
                str(field.value),
                f"{stream}: {field.meaning}" if field.meaning else stream,
                bit_range,
                field.status.value,
            )
        ):
            item.setToolTip(column, text)
        for child in field.children:
            item.addChild(
                self._tree_item(
                    child,
                    stream=stream,
                    bit_offset=bit_offset,
                )
            )
        return item

    def _render_packet(self, result: BluetoothDedicatedResult) -> None:
        packet = result.packet
        self.decode_tree.clear()
        for field in packet.root_fields:
            if packet.protocol_id == "bluetooth.hdt":
                stream = {
                    "training": "Training symbols",
                    "control_header": "Control Header",
                    "payload": "PDU+Payload",
                }.get(field.field_id, "Packet")
                bit_offset = field.start_bit if field.field_id == "payload" else 0
            else:
                stream = "Packet"
                bit_offset = 0
            self.decode_tree.addTopLevelItem(
                self._tree_item(
                    field,
                    stream=stream,
                    bit_offset=bit_offset,
                )
            )
        self.decode_tree.expandToDepth(1)
        QtCore.QTimer.singleShot(0, self.decode_tree._fit_columns)
        payload = payload_field(packet.root_fields)
        if payload is None:
            self.payload_text.setPlainText("Payload field was not decoded")
        else:
            values = np.packbits(np.pad(payload.raw_bits, (0, (-payload.raw_bits.size) % 8)), bitorder="little")
            lines = [f"{offset:04X}: " + " ".join(f"{int(value):02X}" for value in values[offset : offset + 16]) for offset in range(0, values.size, 16)]
            self.payload_text.setPlainText("\n".join(lines) or "(empty payload)")
        listed = self._results or (result,)
        self.packet_table.setRowCount(len(listed))
        for row, listed_result in enumerate(listed):
            listed_packet = listed_result.packet
            integrity = listed_packet.integrity
            checks = [name for name, value in (("HEC OK", integrity.hec_valid), ("CRC OK", integrity.crc_valid)) if value]
            status = "/".join(checks) or ("Incomplete" if not integrity.complete else "Not evaluated")
            for column, value in enumerate((row + 1, listed_packet.phy_name or "--", listed_packet.packet_type or "--", status, listed_packet.raw_bits.size)):
                self.packet_table.setItem(row, column, QtWidgets.QTableWidgetItem(str(value)))
        if listed:
            self.packet_table.selectRow(self._selected_result_index)
        self.issues_table.setRowCount(len(packet.issues))
        for row, issue in enumerate(packet.issues):
            bit_range = "--" if issue.start_bit is None else f"{issue.start_bit}:{issue.stop_bit}"
            for column, value in enumerate((issue.severity.value, issue.code, issue.message, bit_range)):
                self.issues_table.setItem(row, column, QtWidgets.QTableWidgetItem(str(value)))
        self.air_bits_text.setPlainText(format_air_bits(packet.raw_bits))

    @QtCore.Slot(int, int)
    def _packet_row_clicked(self, row: int, _column: int) -> None:
        if not 0 <= int(row) < len(self._results):
            return
        self._select_result(int(row) - self._selected_result_index)
