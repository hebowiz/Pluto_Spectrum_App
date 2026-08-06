"""R&S-inspired multi-window shell for the first offline VSA milestone."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from pluto_sa.config.input_frontend import InputPowerCorrection
from pluto_sa.vsa.model import IQRecording, ModulationFamily, ModulationKind, SignalDescription
from pluto_sa.vsa.pattern import (
    BitOrdering,
    DemodulationSettings,
    KnownPattern,
    MatchSelectionPolicy,
    PatternSearchMode,
    PatternSearchSettings,
    ResultRangeAlignment,
    ResultRangeReference,
    ResultRangeSettings,
    SynchronizationSource,
    prepare_psk_iq,
)
from pluto_sa.vsa.persistence import (
    load_meas_config,
    load_pattern,
    save_meas_config,
    save_pattern,
)
from pluto_sa.vsa.session import VSASession
from pluto_sa.vsa.pluto_source import PlutoCaptureSettings, PlutoLiveSource
from pluto_sa.vsa.sources import FileIQSource, GeneratedIQSource


_MODULATIONS = (
    ModulationKind.GFSK,
    ModulationKind.FSK2,
    ModulationKind.BPSK,
    ModulationKind.QPSK,
    ModulationKind.OQPSK,
    ModulationKind.PI4_DQPSK,
    ModulationKind.DPSK8,
)
_MAX_DISPLAY_POINTS = 100_000


def _decimation_indices(count: int, maximum: int = _MAX_DISPLAY_POINTS) -> slice:
    step = max(1, int(np.ceil(int(count) / int(maximum))))
    return slice(None, None, step)


def _constellation_display_symbols(
    modulation: ModulationKind, symbols: np.ndarray
) -> np.ndarray:
    """Apply the R&S-style display reference without changing decisions."""
    values = np.asarray(symbols, dtype=np.complex128)
    if modulation in {
        ModulationKind.QPSK,
        ModulationKind.OQPSK,
        ModulationKind.PI4_DQPSK,
    }:
        values = values * np.exp(-1j * np.pi / 4.0)
    return values


def _fsk_phase_difference_symbols(
    iq: np.ndarray,
    time_s: np.ndarray,
    symbol_time_s: np.ndarray,
    symbol_frequency_hz: np.ndarray,
    symbol_rate_hz: float,
) -> np.ndarray:
    """Build RMS-normalized FSK phase vectors without discarding amplitude."""
    symbol_times = np.asarray(symbol_time_s, dtype=np.float64)
    frequencies = np.asarray(symbol_frequency_hz, dtype=np.float64)
    count = min(symbol_times.size, frequencies.size)
    if count == 0:
        return np.empty(0, dtype=np.complex128)
    symbol_times = symbol_times[:count]
    frequencies = frequencies[:count]
    samples = np.asarray(iq, dtype=np.complex128)
    sample_times = np.asarray(time_s, dtype=np.float64)
    sampled_iq = np.interp(symbol_times, sample_times, samples.real) + 1j * np.interp(
        symbol_times, sample_times, samples.imag
    )
    rms = float(np.sqrt(np.mean(np.abs(sampled_iq) ** 2)))
    normalized_magnitude = np.abs(sampled_iq) / max(
        rms, np.finfo(np.float64).tiny
    )
    phase_rad = 2.0 * np.pi * frequencies / float(symbol_rate_hz)
    return normalized_magnitude * np.exp(1j * phase_rad)


class _PlutoSingleCaptureThread(QtCore.QThread):
    capture_ready = QtCore.Signal(object)
    capture_failed = QtCore.Signal(str)

    def __init__(
        self,
        source: PlutoLiveSource,
        settings: PlutoCaptureSettings,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._source = source
        self._settings = settings

    def run(self) -> None:
        try:
            self.capture_ready.emit(self._source.capture_single(self._settings))
        except Exception as error:
            self.capture_failed.emit(str(error))


class VSAWindow(QtWidgets.QMainWindow):
    """One VSA measurement session with detachable result windows."""

    def __init__(
        self,
        session: VSASession | None = None,
        preferences: QtCore.QSettings | None = None,
        pluto_source: PlutoLiveSource | None = None,
    ) -> None:
        super().__init__()
        self.session = session or VSASession()
        self._preferences = preferences or QtCore.QSettings("PlutoSA", "PlutoVSA")
        self._pluto_source = pluto_source or PlutoLiveSource()
        self._pluto_capture_thread: _PlutoSingleCaptureThread | None = None
        self._updating_pattern_table = False
        self._pattern_values: list[int] = []
        self._analysis_plot_ranges: dict[str, tuple[list[float], list[float]]] = {}
        self.setWindowTitle("Pluto VSA - FSK / PSK")
        self.resize(1600, 960)
        self.setDockOptions(
            QtWidgets.QMainWindow.DockOption.AllowNestedDocks
            | QtWidgets.QMainWindow.DockOption.AllowTabbedDocks
        )
        self._build_menu()
        self._build_summary_bar()
        self._build_results()
        self._build_configuration()
        self.statusBar().showMessage("Ready - generate or load an IQ recording")
        self._load_generated(ModulationKind.GFSK)

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        open_action = QtGui.QAction("Open IQ...", self)
        open_action.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._open_iq)
        file_menu.addAction(open_action)
        file_menu.addSeparator()
        close_action = QtGui.QAction("Close", self)
        close_action.triggered.connect(self.close)
        file_menu.addAction(close_action)

        run_menu = self.menuBar().addMenu("Sweep / Run")
        self.run_single_action = QtGui.QAction("Run Single", self)
        self.run_single_action.setShortcut("F6")
        self.run_single_action.triggered.connect(self._run_pluto_single)
        run_menu.addAction(self.run_single_action)
        analyze_action = QtGui.QAction("Refresh Analysis", self)
        analyze_action.setShortcut("F5")
        analyze_action.triggered.connect(self._analyze)
        run_menu.addAction(analyze_action)
        run_menu.addSeparator()
        self.previous_result_action = QtGui.QAction(
            "Previous Result Range", self
        )
        self.previous_result_action.setShortcut("Left")
        self.previous_result_action.setEnabled(False)
        self.previous_result_action.triggered.connect(
            lambda: self._select_adjacent_match(-1)
        )
        run_menu.addAction(self.previous_result_action)
        self.next_result_action = QtGui.QAction("Next Result Range", self)
        self.next_result_action.setShortcut("Right")
        self.next_result_action.setEnabled(False)
        self.next_result_action.triggered.connect(
            lambda: self._select_adjacent_match(1)
        )
        run_menu.addAction(self.next_result_action)

        display_menu = self.menuBar().addMenu("Display Config")
        self._display_menu = display_menu
        self.symbol_display_action = QtGui.QAction(
            "Show Symbol Points", self, checkable=True
        )
        self.symbol_display_action.setChecked(False)
        self.symbol_display_action.triggered.connect(self._refresh_display_only)
        display_menu.addAction(self.symbol_display_action)
        self.reset_graph_scales_action = QtGui.QAction(
            "Reset Graph Scales", self
        )
        self.reset_graph_scales_action.setShortcut("Home")
        self.reset_graph_scales_action.triggered.connect(self._reset_graph_scales)
        display_menu.addAction(self.reset_graph_scales_action)
        mouse_menu = display_menu.addMenu("Mouse Interaction")
        self.rect_zoom_action = QtGui.QAction("Rect Zoom", self, checkable=True)
        self.pan_action = QtGui.QAction("Pan", self, checkable=True)
        mouse_group = QtGui.QActionGroup(self)
        mouse_group.setExclusive(True)
        mouse_group.addAction(self.rect_zoom_action)
        mouse_group.addAction(self.pan_action)
        self.rect_zoom_action.setChecked(True)
        self.rect_zoom_action.triggered.connect(self._apply_mouse_interaction_mode)
        self.pan_action.triggered.connect(self._apply_mouse_interaction_mode)
        mouse_menu.addActions(mouse_group.actions())
        display_menu.addSeparator()
        carrier_menu = display_menu.addMenu("Carrier Display")
        self.raw_carrier_action = QtGui.QAction("Raw IQ", self, checkable=True)
        self.corrected_carrier_action = QtGui.QAction(
            "Carrier Corrected", self, checkable=True
        )
        carrier_group = QtGui.QActionGroup(self)
        carrier_group.setExclusive(True)
        carrier_group.addAction(self.raw_carrier_action)
        carrier_group.addAction(self.corrected_carrier_action)
        self.corrected_carrier_action.setChecked(True)
        self.raw_carrier_action.triggered.connect(self._refresh_display_only)
        self.corrected_carrier_action.triggered.connect(self._refresh_display_only)
        carrier_menu.addActions(carrier_group.actions())

        meas_config_menu = self.menuBar().addMenu("Meas Config")
        open_config_action = QtGui.QAction("Open Meas Config...", self)
        open_config_action.setShortcut("Ctrl+M")
        open_config_action.triggered.connect(self._open_meas_config)
        meas_config_menu.addAction(open_config_action)
        meas_config_menu.addSeparator()
        load_config_action = QtGui.QAction("Load Meas Config...", self)
        load_config_action.triggered.connect(self._load_meas_config_file)
        meas_config_menu.addAction(load_config_action)
        save_config_action = QtGui.QAction("Save Meas Config As...", self)
        save_config_action.triggered.connect(self._save_meas_config_file)
        meas_config_menu.addAction(save_config_action)

    def _build_summary_bar(self) -> None:
        toolbar = QtWidgets.QToolBar("Session Summary", self)
        toolbar.setMovable(False)
        toolbar.setObjectName("vsa-session-summary")
        self.summary_label = QtWidgets.QLabel("No capture")
        self.summary_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        toolbar.addWidget(self.summary_label)
        self.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, toolbar)

    def _make_plot(self, title: str, left: str, bottom: str) -> pg.PlotWidget:
        plot = pg.PlotWidget(title=title)
        plot.showGrid(x=True, y=True, alpha=0.25)
        plot.setLabel("left", left)
        plot.setLabel("bottom", bottom)
        # Long IQ traces are expensive to repaint while Windows is moving or
        # exposing the top-level window. Let pyqtgraph retain extrema while
        # reducing the curve to the available horizontal pixels, and avoid
        # painting samples outside the current result-range view.
        plot.setDownsampling(auto=True, mode="peak")
        plot.setClipToView(True)
        return plot

    def _dock(self, title: str, widget: QtWidgets.QWidget) -> QtWidgets.QDockWidget:
        dock = QtWidgets.QDockWidget(title, self)
        dock.setObjectName(f"vsa-{title.lower().replace(' ', '-')}")
        dock.setWidget(widget)
        dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
            | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        action = dock.toggleViewAction()
        self._display_menu.addAction(action)
        return dock

    def _build_results(self) -> None:
        self.zero_span_plot = self._make_plot("Capture Power", "IQ Power (dBm)", "Time (ms)")
        self.zero_span_dock = self._dock("IQ Power", self.zero_span_plot)
        self.addDockWidget(
            QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.zero_span_dock
        )

        self.spectrum_plot = self._make_plot("Spectrum", "Magnitude (dBFS)", "Relative Frequency (MHz)")
        self.spectrum_dock = self._dock("Spectrum", self.spectrum_plot)
        self.splitDockWidget(
            self.zero_span_dock,
            self.spectrum_dock,
            QtCore.Qt.Orientation.Horizontal,
        )

        self.result_summary = QtWidgets.QTableWidget(0, 2)
        self.result_summary.setHorizontalHeaderLabels(("Parameter", "Current"))
        self.result_summary.verticalHeader().setVisible(False)
        self.result_summary.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.result_summary.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.NoSelection
        )
        self.result_summary.setAlternatingRowColors(False)
        self.result_summary.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self.result_summary_dock = self._dock("Result Summary", self.result_summary)
        self.splitDockWidget(
            self.spectrum_dock,
            self.result_summary_dock,
            QtCore.Qt.Orientation.Horizontal,
        )

        self.modulation_plot = self._make_plot("Modulation", "Q", "I")
        self.modulation_dock = self._dock("Modulation", self.modulation_plot)
        self.splitDockWidget(
            self.zero_span_dock,
            self.modulation_dock,
            QtCore.Qt.Orientation.Vertical,
        )

        self.symbol_plot = self._make_plot("Symbol Plot", "Q", "I")
        # I is not monotonic in either constellation or phase-difference
        # views, so time-series clipping/downsampling does not apply here.
        self.symbol_plot.setDownsampling(auto=False)
        self.symbol_plot.setClipToView(False)
        self.symbol_plot.setAspectLocked(True, ratio=1.0)
        self.symbol_plot_dock = self._dock("Symbol Plot", self.symbol_plot)
        # Compatibility alias for the original empty Reserved dock name.
        self.reserved_dock = self.symbol_plot_dock
        self.splitDockWidget(
            self.spectrum_dock,
            self.symbol_plot_dock,
            QtCore.Qt.Orientation.Vertical,
        )

        symbol_container = QtWidgets.QWidget()
        symbol_layout = QtWidgets.QVBoxLayout(symbol_container)
        symbol_layout.setContentsMargins(6, 6, 6, 6)
        self.symbol_table = QtWidgets.QTableWidget(0, 10)
        self.symbol_table.setHorizontalHeaderLabels([str(index) for index in range(10)])
        self.symbol_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.symbol_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectItems
        )
        self.symbol_table.setAlternatingRowColors(False)
        self.symbol_table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self.symbol_table.verticalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        symbol_layout.addWidget(self.symbol_table, 1)
        self.symbol_dock = self._dock("Symbol Table", symbol_container)
        self.splitDockWidget(
            self.result_summary_dock,
            self.symbol_dock,
            QtCore.Qt.Orientation.Vertical,
        )
        self._apply_mouse_interaction_mode()
        QtCore.QTimer.singleShot(0, self._equalize_result_docks)

    def _equalize_result_docks(self) -> None:
        top_row = (
            self.zero_span_dock,
            self.spectrum_dock,
            self.result_summary_dock,
        )
        bottom_row = (
            self.modulation_dock,
            self.reserved_dock,
            self.symbol_dock,
        )
        self.resizeDocks(list(top_row), [500, 500, 500], QtCore.Qt.Orientation.Horizontal)
        self.resizeDocks(
            list(bottom_row), [500, 500, 500], QtCore.Qt.Orientation.Horizontal
        )
        for upper, lower in zip(top_row, bottom_row):
            self.resizeDocks(
                [upper, lower], [400, 400], QtCore.Qt.Orientation.Vertical
            )

    def _build_configuration(self) -> None:
        config_pages: list[tuple[str, QtWidgets.QWidget]] = []

        source_page = QtWidgets.QWidget()
        source_layout = QtWidgets.QVBoxLayout(source_page)
        self.input_source_combo = QtWidgets.QComboBox()
        self.input_source_combo.addItems(("Generated", "IQ File", "Pluto"))
        source_layout.addWidget(QtWidgets.QLabel("Input Source"))
        source_layout.addWidget(self.input_source_combo)
        gfsk_button = QtWidgets.QPushButton("Generate GFSK")
        qpsk_button = QtWidgets.QPushButton("Generate QPSK")
        edr_button = QtWidgets.QPushButton("Generate pi/4-DQPSK")
        open_button = QtWidgets.QPushButton("Open IQ File...")
        gfsk_button.clicked.connect(lambda: self._load_generated(ModulationKind.GFSK))
        qpsk_button.clicked.connect(lambda: self._load_generated(ModulationKind.QPSK))
        edr_button.clicked.connect(lambda: self._load_generated(ModulationKind.PI4_DQPSK))
        open_button.clicked.connect(self._open_iq)
        source_layout.addWidget(gfsk_button)
        source_layout.addWidget(qpsk_button)
        source_layout.addWidget(edr_button)
        source_layout.addWidget(open_button)
        source_layout.addSpacing(12)
        pluto_form = QtWidgets.QFormLayout()
        self.pluto_uri_edit = QtWidgets.QLineEdit()
        self.pluto_uri_edit.setPlaceholderText("Auto (direct USB preferred)")
        self.capture_center_spin = QtWidgets.QDoubleSpinBox()
        self.capture_center_spin.setRange(70.0, 6000.0)
        self.capture_center_spin.setDecimals(6)
        self.capture_center_spin.setValue(2441.0)
        self.capture_center_spin.setSuffix(" MHz")
        self.capture_rf_bandwidth_spin = QtWidgets.QDoubleSpinBox()
        self.capture_rf_bandwidth_spin.setRange(0.2, 56.0)
        self.capture_rf_bandwidth_spin.setDecimals(3)
        self.capture_rf_bandwidth_spin.setValue(8.0)
        self.capture_rf_bandwidth_spin.setSuffix(" MHz")
        self.internal_gain_spin = QtWidgets.QSpinBox()
        self.internal_gain_spin.setRange(0, 40)
        self.internal_gain_spin.setValue(30)
        self.internal_gain_spin.setSuffix(" dB")
        self.external_attenuation_spin = QtWidgets.QDoubleSpinBox()
        self.external_attenuation_spin.setRange(-200.0, 200.0)
        self.external_attenuation_spin.setDecimals(1)
        self.external_attenuation_spin.setValue(30.0)
        self.external_attenuation_spin.setSuffix(" dB")
        self.external_gain_spin = QtWidgets.QDoubleSpinBox()
        self.external_gain_spin.setRange(-200.0, 200.0)
        self.external_gain_spin.setDecimals(1)
        self.external_gain_spin.setValue(0.0)
        self.external_gain_spin.setSuffix(" dB")
        self.capture_correction_label = QtWidgets.QLabel()
        pluto_form.addRow("Pluto URI", self.pluto_uri_edit)
        pluto_form.addRow("Center Frequency", self.capture_center_spin)
        pluto_form.addRow("RF Bandwidth", self.capture_rf_bandwidth_spin)
        pluto_form.addRow("Internal Gain", self.internal_gain_spin)
        pluto_form.addRow("External ATT", self.external_attenuation_spin)
        pluto_form.addRow("External Gain", self.external_gain_spin)
        pluto_form.addRow("Input Correction", self.capture_correction_label)
        source_layout.addLayout(pluto_form)
        for control in (
            self.internal_gain_spin,
            self.external_attenuation_spin,
            self.external_gain_spin,
        ):
            control.valueChanged.connect(self._sync_capture_settings)
        self.channel_filter_check = QtWidgets.QCheckBox("Enable Analysis Channel")
        self.analysis_center_spin = QtWidgets.QDoubleSpinBox()
        self.analysis_center_spin.setRange(-100_000.0, 100_000.0)
        self.analysis_center_spin.setDecimals(6)
        self.analysis_center_spin.setSuffix(" MHz")
        self.analysis_bandwidth_spin = QtWidgets.QDoubleSpinBox()
        self.analysis_bandwidth_spin.setRange(0.000001, 100.0)
        self.analysis_bandwidth_spin.setDecimals(6)
        self.analysis_bandwidth_spin.setValue(1.5)
        self.analysis_bandwidth_spin.setSuffix(" MHz")
        channel_form = QtWidgets.QFormLayout()
        channel_form.addRow(self.channel_filter_check)
        channel_form.addRow("Analysis Center", self.analysis_center_spin)
        channel_form.addRow("Analysis Bandwidth", self.analysis_bandwidth_spin)
        source_layout.addLayout(channel_form)
        self.channel_filter_check.toggled.connect(self._sync_analysis_controls)
        self._sync_analysis_controls()
        source_layout.addStretch(1)
        config_pages.append(("Input / Frontend", source_page))

        signal_page = QtWidgets.QWidget()
        signal_form = QtWidgets.QFormLayout(signal_page)
        self.modulation_combo = QtWidgets.QComboBox()
        for modulation in _MODULATIONS:
            self.modulation_combo.addItem(modulation.value, modulation.value)
        self.symbol_rate_spin = QtWidgets.QDoubleSpinBox()
        self.symbol_rate_spin.setRange(1.0, 100_000_000.0)
        self.symbol_rate_spin.setDecimals(0)
        self.symbol_rate_spin.setValue(1_000_000.0)
        self.symbol_rate_spin.setSuffix(" Sym/s")
        self.deviation_spin = QtWidgets.QDoubleSpinBox()
        self.deviation_spin.setRange(1.0, 50_000_000.0)
        self.deviation_spin.setDecimals(0)
        self.deviation_spin.setValue(250_000.0)
        self.deviation_spin.setSuffix(" Hz")
        self.mapping_combo = QtWidgets.QComboBox()
        self.mapping_combo.addItem("Natural")
        self.tx_filter_combo = QtWidgets.QComboBox()
        self.tx_filter_combo.addItems(("None", "Gaussian", "Root Raised Cosine"))
        self.filter_parameter_spin = QtWidgets.QDoubleSpinBox()
        self.filter_parameter_spin.setRange(0.01, 2.0)
        self.filter_parameter_spin.setDecimals(3)
        self.filter_parameter_spin.setValue(0.5)
        signal_form.addRow("Modulation Type / Order", self.modulation_combo)
        signal_form.addRow("Symbol Rate", self.symbol_rate_spin)
        signal_form.addRow("FSK Ref Deviation", self.deviation_spin)
        signal_form.addRow("Modulation Mapping", self.mapping_combo)
        signal_form.addRow("Transmit Filter Type", self.tx_filter_combo)
        signal_form.addRow("Alpha / BT", self.filter_parameter_spin)
        self.modulation_combo.currentIndexChanged.connect(self._sync_signal_controls)
        self.tx_filter_combo.currentTextChanged.connect(
            lambda value: self.filter_parameter_spin.setEnabled(value != "None")
        )
        self.symbol_rate_spin.valueChanged.connect(self._sync_capture_settings)
        config_pages.append(("Signal Description", signal_page))

        capture_page = QtWidgets.QWidget()
        capture_form = QtWidgets.QFormLayout(capture_page)
        self.capture_length_spin = QtWidgets.QDoubleSpinBox()
        self.capture_length_spin.setRange(0.001, 1_000_000.0)
        self.capture_length_spin.setDecimals(3)
        self.capture_length_spin.setValue(3.0)
        self.capture_length_unit_combo = QtWidgets.QComboBox()
        self.capture_length_unit_combo.addItems(("ms", "Symbols"))
        capture_length_row = QtWidgets.QHBoxLayout()
        capture_length_row.addWidget(self.capture_length_spin)
        capture_length_row.addWidget(self.capture_length_unit_combo)
        self.capture_oversampling_combo = QtWidgets.QComboBox()
        for value in (2, 4, 8, 16, 32, 64, 128):
            self.capture_oversampling_combo.addItem(
                f"{value} samples/symbol", value
            )
        self.capture_oversampling_combo.setCurrentIndex(
            self.capture_oversampling_combo.findData(8)
        )
        self.capture_sample_rate_label = QtWidgets.QLabel()
        self.capture_samples_label = QtWidgets.QLabel()
        self.capture_usable_bandwidth_label = QtWidgets.QLabel()
        self.swap_iq_check = QtWidgets.QCheckBox("Swap I/Q")
        capture_form.addRow("Capture Length", capture_length_row)
        capture_form.addRow("Sample Rate", self.capture_oversampling_combo)
        capture_form.addRow("Resulting Sample Rate", self.capture_sample_rate_label)
        capture_form.addRow("Record Length", self.capture_samples_label)
        capture_form.addRow("Usable I/Q Bandwidth", self.capture_usable_bandwidth_label)
        capture_form.addRow(self.swap_iq_check)
        for control in (
            self.capture_length_spin,
            self.capture_length_unit_combo,
            self.capture_oversampling_combo,
            self.capture_rf_bandwidth_spin,
        ):
            if isinstance(control, QtWidgets.QComboBox):
                control.currentIndexChanged.connect(self._sync_capture_settings)
            else:
                control.valueChanged.connect(self._sync_capture_settings)
        self._sync_capture_settings()
        config_pages.append(("Signal Capture", capture_page))

        pattern_page = QtWidgets.QWidget()
        pattern_layout = QtWidgets.QVBoxLayout(pattern_page)
        pattern_form = QtWidgets.QFormLayout()
        self.pattern_search_check = QtWidgets.QCheckBox("Pattern Search On")
        self.pattern_name_edit = QtWidgets.QLineEdit("Known Pattern")
        self.pattern_format_combo = QtWidgets.QComboBox()
        self.pattern_format_combo.addItems(("Binary", "Decimal", "Hexadecimal"))
        # Kept as a compatibility input for callers that previously populated
        # the one-line editor. The visible editor is now pattern_symbol_table.
        self.pattern_symbols_edit = QtWidgets.QLineEdit("01010101")
        self.pattern_symbols_edit.setVisible(False)
        self.pattern_symbol_table = QtWidgets.QTableWidget(1, 10)
        self.pattern_symbol_table.setHorizontalHeaderLabels(
            [str(index) for index in range(10)]
        )
        self.pattern_symbol_table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self.pattern_symbol_table.verticalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        self.pattern_symbol_table.setMinimumHeight(190)
        pattern_table_buttons = QtWidgets.QHBoxLayout()
        add_pattern_row = QtWidgets.QPushButton("Add Row")
        remove_pattern_row = QtWidgets.QPushButton("Remove Last Row")
        load_pattern_button = QtWidgets.QPushButton("Load Pattern...")
        save_pattern_button = QtWidgets.QPushButton("Save Pattern As...")
        add_pattern_row.clicked.connect(self._add_pattern_row)
        remove_pattern_row.clicked.connect(self._remove_pattern_row)
        load_pattern_button.clicked.connect(self._load_pattern_file)
        save_pattern_button.clicked.connect(self._save_pattern_file)
        for button in (
            add_pattern_row,
            remove_pattern_row,
            load_pattern_button,
            save_pattern_button,
        ):
            pattern_table_buttons.addWidget(button)
        self.pattern_threshold_auto = QtWidgets.QCheckBox("Auto (90%)")
        self.pattern_threshold_auto.setChecked(True)
        self.pattern_threshold_spin = QtWidgets.QDoubleSpinBox()
        self.pattern_threshold_spin.setRange(0.1, 100.0)
        self.pattern_threshold_spin.setValue(90.0)
        self.pattern_threshold_spin.setSuffix(" %")
        self.pattern_threshold_spin.setEnabled(False)
        self.pattern_meas_only_check = QtWidgets.QCheckBox(
            "Meas only if Pattern Symbols Correct"
        )
        self.pattern_meas_only_check.setChecked(True)
        self.pattern_match_selection_combo = QtWidgets.QComboBox()
        for policy in MatchSelectionPolicy:
            self.pattern_match_selection_combo.addItem(policy.value, policy.value)
        self.pattern_match_selection_combo.setCurrentText(
            MatchSelectionPolicy.FIRST.value
        )
        self.pattern_match_index_spin = QtWidgets.QSpinBox()
        self.pattern_match_index_spin.setRange(1, 1_000_000)
        self.pattern_match_index_spin.setValue(1)
        self.pattern_match_index_spin.setEnabled(False)
        pattern_form.addRow(self.pattern_search_check)
        pattern_form.addRow("Name", self.pattern_name_edit)
        pattern_form.addRow("Symbol Format", self.pattern_format_combo)
        pattern_form.addRow("I/Q Correlation Threshold", self.pattern_threshold_spin)
        pattern_form.addRow(self.pattern_threshold_auto)
        pattern_form.addRow(self.pattern_meas_only_check)
        pattern_form.addRow(
            "Match Selection", self.pattern_match_selection_combo
        )
        pattern_form.addRow("Match Index", self.pattern_match_index_spin)
        pattern_layout.addLayout(pattern_form)
        pattern_layout.addWidget(QtWidgets.QLabel("Pattern Symbols"))
        pattern_layout.addWidget(self.pattern_symbol_table, 1)
        pattern_layout.addLayout(pattern_table_buttons)
        self.pattern_threshold_auto.toggled.connect(
            lambda checked: self.pattern_threshold_spin.setEnabled(not checked)
        )
        self.pattern_match_selection_combo.currentTextChanged.connect(
            lambda value: self.pattern_match_index_spin.setEnabled(
                value == MatchSelectionPolicy.INDEX.value
            )
        )
        self.pattern_format_combo.currentTextChanged.connect(
            self._refresh_pattern_table_format
        )
        self.pattern_symbols_edit.textChanged.connect(
            self._load_pattern_compatibility_text
        )
        self.pattern_symbol_table.cellChanged.connect(
            self._pattern_table_cell_changed
        )
        self._load_pattern_compatibility_text(self.pattern_symbols_edit.text())
        config_pages.append(("Pattern Search", pattern_page))

        range_page = QtWidgets.QWidget()
        range_form = QtWidgets.QFormLayout(range_page)
        self.result_length_spin = QtWidgets.QSpinBox()
        self.result_length_spin.setRange(1, 1_000_000)
        self.result_length_spin.setValue(256)
        self.result_reference_combo = QtWidgets.QComboBox()
        self.result_reference_combo.addItem(
            ResultRangeReference.PATTERN_WAVEFORM.value,
            ResultRangeReference.PATTERN_WAVEFORM.value,
        )
        self.result_alignment_combo = QtWidgets.QComboBox()
        for alignment in ResultRangeAlignment:
            self.result_alignment_combo.addItem(alignment.value, alignment.value)
        self.result_offset_spin = QtWidgets.QSpinBox()
        self.result_offset_spin.setRange(-1_000_000, 1_000_000)
        self.reference_symbol_number_spin = QtWidgets.QSpinBox()
        self.reference_symbol_number_spin.setRange(-1_000_000, 1_000_000)
        self.reference_symbol_number_spin.setEnabled(False)
        self.reference_symbol_number_spin.setToolTip(
            "Display-axis numbering is planned; it does not change DSP yet."
        )
        self.exclude_incomplete_result_check = QtWidgets.QCheckBox(
            "Exclude incomplete Result Range"
        )
        self.exclude_incomplete_result_check.setChecked(False)
        range_form.addRow("Result Length (Symbols)", self.result_length_spin)
        range_form.addRow("Reference", self.result_reference_combo)
        range_form.addRow("Alignment", self.result_alignment_combo)
        range_form.addRow("Offset (Symbols)", self.result_offset_spin)
        range_form.addRow(
            "Symbol Number at Pattern Start", self.reference_symbol_number_spin
        )
        range_form.addRow(self.exclude_incomplete_result_check)
        config_pages.append(("Result Range", range_page))

        demod_page = QtWidgets.QWidget()
        demod_form = QtWidgets.QFormLayout(demod_page)
        self.coarse_sync_combo = QtWidgets.QComboBox()
        self.coarse_sync_combo.addItems(("Auto", "Detected Data", "Pattern"))
        self.fine_sync_combo = QtWidgets.QComboBox()
        self.fine_sync_combo.addItems(("Auto", "Detected Data", "Pattern"))
        self.bit_order_combo = QtWidgets.QComboBox()
        self.bit_order_combo.addItems(("MSB", "LSB"))
        self.compensate_drift_check = QtWidgets.QCheckBox("Carrier Frequency Drift")
        self.compensate_drift_check.setChecked(False)
        self.compensate_drift_check.setToolTip(
            "Experimental linear-drift compensation; CFO compensation is always applied."
        )
        self.compensate_deviation_check = QtWidgets.QCheckBox("FSK Deviation Error")
        self.compensate_deviation_check.setChecked(True)
        for control in (
            self.coarse_sync_combo,
            self.fine_sync_combo,
            self.compensate_deviation_check,
        ):
            control.setEnabled(False)
            control.setToolTip("R&S-compatible setting contract; DSP connection is planned.")
        demod_form.addRow("Coarse Synchronization", self.coarse_sync_combo)
        demod_form.addRow("Fine Synchronization", self.fine_sync_combo)
        demod_form.addRow("Bit Ordering", self.bit_order_combo)
        demod_form.addRow("Compensate for", self.compensate_drift_check)
        demod_form.addRow("", self.compensate_deviation_check)
        config_pages.append(("Demodulation", demod_page))

        run_page = QtWidgets.QWidget()
        run_layout = QtWidgets.QVBoxLayout(run_page)
        self.run_single_button = QtWidgets.QPushButton("Run Single (Pluto)")
        self.run_single_button.clicked.connect(self._run_pluto_single)
        run_layout.addWidget(self.run_single_button)
        refresh_button = QtWidgets.QPushButton("Refresh Analysis")
        refresh_button.clicked.connect(self._analyze)
        run_layout.addWidget(refresh_button)
        run_layout.addWidget(
            QtWidgets.QLabel(
                "Run Single captures new Pluto IQ. Refresh reuses the current capture."
            )
        )
        run_layout.addStretch(1)
        config_pages.append(("Sweep / Run", run_page))

        self._meas_config_dialog = QtWidgets.QDialog(self)
        self._meas_config_dialog.setWindowTitle("Meas Config")
        self._meas_config_dialog.setModal(True)
        self._meas_config_dialog.setWindowModality(
            QtCore.Qt.WindowModality.WindowModal
        )
        self._meas_config_dialog.resize(620, 520)
        dialog_layout = QtWidgets.QVBoxLayout(self._meas_config_dialog)

        navigation_layout = QtWidgets.QHBoxLayout()
        self._config_back_button = QtWidgets.QPushButton("< Config Top")
        self._config_back_button.clicked.connect(self._show_config_top)
        self._config_page_title = QtWidgets.QLabel()
        title_font = self._config_page_title.font()
        title_font.setBold(True)
        title_font.setPointSize(title_font.pointSize() + 2)
        self._config_page_title.setFont(title_font)
        navigation_layout.addWidget(self._config_back_button)
        navigation_layout.addWidget(self._config_page_title)
        navigation_layout.addStretch(1)
        dialog_layout.addLayout(navigation_layout)

        self._config_stack = QtWidgets.QStackedWidget()
        config_top = QtWidgets.QWidget()
        config_top_layout = QtWidgets.QVBoxLayout(config_top)
        self._config_top_title = QtWidgets.QLabel("Config Top Menu")
        top_title_font = self._config_top_title.font()
        top_title_font.setBold(True)
        top_title_font.setPointSizeF(max(16.0, top_title_font.pointSizeF() + 6.0))
        self._config_top_title.setFont(top_title_font)
        config_top_layout.addWidget(self._config_top_title)
        config_button_grid = QtWidgets.QGridLayout()
        config_button_grid.setHorizontalSpacing(14)
        config_button_grid.setVerticalSpacing(14)
        self._config_top_buttons: dict[str, QtWidgets.QPushButton] = {}
        for index, (name, page) in enumerate(config_pages, start=1):
            button = QtWidgets.QPushButton(name)
            button_font = button.font()
            button_font.setPointSizeF(max(18.0, button_font.pointSizeF() * 2.0))
            button_font.setBold(True)
            button.setFont(button_font)
            button.setMinimumHeight(84)
            button.setProperty("configPageIndex", index)
            button.clicked.connect(self._show_selected_config_page)
            config_button_grid.addWidget(button, (index - 1) // 2, (index - 1) % 2)
            self._config_top_buttons[name] = button
            self._config_stack.addWidget(page)
        config_top_layout.addLayout(config_button_grid)
        config_top_layout.addStretch(1)
        self._config_stack.insertWidget(0, config_top)
        self._config_page_names = ("Config Top Menu",) + tuple(
            name for name, _page in config_pages
        )
        dialog_layout.addWidget(self._config_stack, 1)
        close_buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Close
        )
        close_buttons.rejected.connect(self._meas_config_dialog.reject)
        dialog_layout.addWidget(close_buttons)
        self._show_config_page(0)

    def _show_config_top(self) -> None:
        self._show_config_page(0)

    def _show_selected_config_page(self) -> None:
        button = self.sender()
        if not isinstance(button, QtWidgets.QPushButton):
            return
        self._show_config_page(int(button.property("configPageIndex")))

    def _show_config_page(self, index: int) -> None:
        self._config_stack.setCurrentIndex(index)
        is_top = index == 0
        self._config_back_button.setVisible(not is_top)
        self._config_page_title.setText(
            "" if is_top else self._config_page_names[index]
        )

    def _open_meas_config(self) -> None:
        self._show_config_page(0)
        self._meas_config_dialog.exec()

    def _last_directory(self, file_kind: str) -> str:
        stored = self._preferences.value(f"directories/{file_kind}", "", type=str)
        # Never pass an empty path to the native Windows dialog. An empty path
        # makes Qt reuse the process-wide native-dialog history, which makes
        # the Pattern and Config histories appear to be shared.
        return stored if stored and Path(stored).is_dir() else str(Path.cwd())

    def _remember_directory(self, file_kind: str, path: str | Path) -> None:
        directory = str(Path(path).resolve().parent)
        self._preferences.setValue(f"directories/{file_kind}", directory)
        self._preferences.sync()

    @staticmethod
    def _with_suffix(path: str, suffix: str) -> str:
        candidate = Path(path)
        return str(candidate if candidate.suffix else candidate.with_suffix(suffix))

    def _format_pattern_symbol(self, symbol: int) -> str:
        symbol_format = self.pattern_format_combo.currentText()
        if symbol_format == "Binary":
            width = int(round(np.log2(self._selected_modulation().order)))
            return format(int(symbol), f"0{width}b")
        if symbol_format == "Hexadecimal":
            return format(int(symbol), "X")
        return str(int(symbol))

    def _set_pattern_symbols(self, symbols: list[int] | tuple[int, ...]) -> None:
        self._pattern_values = [int(symbol) for symbol in symbols]
        row_count = max(1, int(np.ceil(len(self._pattern_values) / 10.0)))
        self._updating_pattern_table = True
        try:
            self.pattern_symbol_table.clearContents()
            self.pattern_symbol_table.setRowCount(row_count)
            self.pattern_symbol_table.setVerticalHeaderLabels(
                [str(row * 10) for row in range(row_count)]
            )
            for index, symbol in enumerate(self._pattern_values):
                item = QtWidgets.QTableWidgetItem(
                    self._format_pattern_symbol(symbol)
                )
                item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                self.pattern_symbol_table.setItem(index // 10, index % 10, item)
        finally:
            self._updating_pattern_table = False

    def _pattern_symbols_from_table(self, order: int) -> tuple[int, ...]:
        values: list[int] = []
        found_empty = False
        symbol_format = self.pattern_format_combo.currentText()
        base = 2 if symbol_format == "Binary" else (16 if symbol_format == "Hexadecimal" else 10)
        for index in range(self.pattern_symbol_table.rowCount() * 10):
            item = self.pattern_symbol_table.item(index // 10, index % 10)
            text = "" if item is None else item.text().strip()
            if not text:
                found_empty = True
                continue
            if found_empty:
                raise ValueError("Pattern Symbol table may only have empty cells at the end")
            try:
                value = int(text, base)
            except ValueError as error:
                raise ValueError(
                    f"Invalid {symbol_format} symbol at index {index}: {text!r}"
                ) from error
            if value < 0 or value >= order:
                raise ValueError(f"Pattern symbol must be between 0 and {order - 1}")
            values.append(value)
        if len(values) < 4:
            raise ValueError("known pattern must contain at least four symbols")
        return tuple(values)

    def _pattern_table_cell_changed(self, _row: int, _column: int) -> None:
        if self._updating_pattern_table:
            return
        item = self.pattern_symbol_table.item(_row, _column)
        if item is not None:
            self._updating_pattern_table = True
            try:
                item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            finally:
                self._updating_pattern_table = False
        try:
            self._pattern_values = list(
                self._pattern_symbols_from_table(self._selected_modulation().order)
            )
        except ValueError:
            # Keep the user's in-progress edit visible. Validation is reported
            # on save or analysis rather than interrupting every keystroke.
            pass

    def _refresh_pattern_table_format(self, _value: str = "") -> None:
        self._set_pattern_symbols(self._pattern_values)

    def _load_pattern_compatibility_text(self, text: str) -> None:
        if self._updating_pattern_table or not text.strip():
            return
        try:
            symbol_format = self.pattern_format_combo.currentText()
            if symbol_format == "Binary":
                compact = "".join(text.replace(",", " ").split())
                width = int(round(np.log2(self._selected_modulation().order)))
                if any(character not in "01" for character in compact) or len(compact) % width:
                    return
                values = [
                    int(compact[index : index + width], 2)
                    for index in range(0, len(compact), width)
                ]
            else:
                base = 16 if symbol_format == "Hexadecimal" else 10
                values = [int(token, base) for token in text.replace(",", " ").split()]
            if values:
                self._set_pattern_symbols(values)
        except ValueError:
            pass

    def _add_pattern_row(self) -> None:
        row = self.pattern_symbol_table.rowCount()
        self.pattern_symbol_table.insertRow(row)
        self.pattern_symbol_table.setVerticalHeaderItem(
            row, QtWidgets.QTableWidgetItem(str(row * 10))
        )

    def _remove_pattern_row(self) -> None:
        if self.pattern_symbol_table.rowCount() > 1:
            self.pattern_symbol_table.removeRow(
                self.pattern_symbol_table.rowCount() - 1
            )
            self._pattern_table_cell_changed(0, 0)

    def _save_pattern_file(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Known Pattern",
            self._last_directory("pattern"),
            "VSA pattern (*.vsapattern.json);;JSON files (*.json)",
        )
        if not path:
            return
        path = self._with_suffix(path, ".vsapattern.json")
        self._remember_directory("pattern", path)
        try:
            save_pattern(
                path,
                name=self.pattern_name_edit.text(),
                symbols=self._pattern_symbols_from_table(
                    self._selected_modulation().order
                ),
                symbol_format=self.pattern_format_combo.currentText(),
            )
            self.statusBar().showMessage(f"Pattern saved - {Path(path).name}")
        except ValueError as error:
            QtWidgets.QMessageBox.critical(self, "Pattern Save Error", str(error))

    def _load_pattern_file(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Known Pattern",
            self._last_directory("pattern"),
            "VSA pattern (*.vsapattern.json *.json);;All files (*)",
        )
        if not path:
            return
        self._remember_directory("pattern", path)
        try:
            document = load_pattern(path)
            order = self._selected_modulation().order
            if any(int(symbol) >= order for symbol in document["symbols"]):
                raise ValueError(
                    f"pattern contains symbols outside the current modulation order {order}"
                )
            self.pattern_name_edit.setText(document["name"])
            self.pattern_format_combo.setCurrentText(document["symbol_format"])
            self._set_pattern_symbols(document["symbols"])
            self.statusBar().showMessage(f"Pattern loaded - {Path(path).name}")
        except ValueError as error:
            QtWidgets.QMessageBox.critical(self, "Pattern Load Error", str(error))

    def _meas_config_values(self) -> dict[str, object]:
        return {
            "input_frontend": {
                "input_source": self.input_source_combo.currentText(),
                "pluto_uri": self.pluto_uri_edit.text().strip(),
                "center_frequency_mhz": self.capture_center_spin.value(),
                "rf_bandwidth_mhz": self.capture_rf_bandwidth_spin.value(),
                "internal_gain_db": self.internal_gain_spin.value(),
                "external_attenuation_db": self.external_attenuation_spin.value(),
                "external_gain_db": self.external_gain_spin.value(),
                "analysis_channel_enabled": self.channel_filter_check.isChecked(),
                "analysis_center_mhz": self.analysis_center_spin.value(),
                "analysis_bandwidth_mhz": self.analysis_bandwidth_spin.value(),
            },
            "signal_capture": {
                "capture_length": self.capture_length_spin.value(),
                "capture_length_unit": self.capture_length_unit_combo.currentText(),
                "samples_per_symbol": int(
                    self.capture_oversampling_combo.currentData()
                ),
                "swap_iq": self.swap_iq_check.isChecked(),
            },
            "signal_description": {
                "modulation": self._selected_modulation().value,
                "symbol_rate_hz": self.symbol_rate_spin.value(),
                "frequency_deviation_hz": self.deviation_spin.value(),
                "symbol_mapping": self.mapping_combo.currentText(),
                "tx_filter": self.tx_filter_combo.currentText(),
                "filter_parameter": self.filter_parameter_spin.value(),
            },
            "pattern_search": {
                "enabled": self.pattern_search_check.isChecked(),
                "name": self.pattern_name_edit.text(),
                "symbol_format": self.pattern_format_combo.currentText(),
                "symbols": list(
                    self._pattern_symbols_from_table(
                        self._selected_modulation().order
                    )
                ),
                "threshold_auto": self.pattern_threshold_auto.isChecked(),
                "threshold_percent": self.pattern_threshold_spin.value(),
                "meas_only_if_correct": self.pattern_meas_only_check.isChecked(),
                "match_selection": self.pattern_match_selection_combo.currentText(),
                "match_index": self.pattern_match_index_spin.value(),
            },
            "result_range": {
                "length_symbols": self.result_length_spin.value(),
                "reference": self.result_reference_combo.currentText(),
                "alignment": self.result_alignment_combo.currentText(),
                "offset_symbols": self.result_offset_spin.value(),
                "symbol_number_at_pattern_start": self.reference_symbol_number_spin.value(),
                "exclude_incomplete_result": (
                    self.exclude_incomplete_result_check.isChecked()
                ),
            },
            "demodulation": {
                "coarse_synchronization": self.coarse_sync_combo.currentText(),
                "fine_synchronization": self.fine_sync_combo.currentText(),
                "bit_ordering": self.bit_order_combo.currentText(),
                "compensate_carrier_frequency_drift": self.compensate_drift_check.isChecked(),
                "compensate_fsk_deviation_error": self.compensate_deviation_check.isChecked(),
            },
        }

    @staticmethod
    def _set_combo_text(combo: QtWidgets.QComboBox, value: object, name: str) -> None:
        index = combo.findText(str(value))
        if index < 0:
            raise ValueError(f"unsupported {name}: {value!r}")
        combo.setCurrentIndex(index)

    def _apply_meas_config_values(self, settings: dict[str, object]) -> None:
        try:
            source = settings["input_frontend"]
            signal = settings["signal_description"]
            pattern = settings["pattern_search"]
            result_range = settings["result_range"]
            demodulation = settings["demodulation"]
            signal_capture = settings.get("signal_capture", {})
            if not all(isinstance(section, dict) for section in (
                source, signal, pattern, result_range, demodulation, signal_capture
            )):
                raise TypeError("configuration sections must be objects")
            self._set_combo_text(self.modulation_combo, signal["modulation"], "modulation")
            self.symbol_rate_spin.setValue(float(signal["symbol_rate_hz"]))
            self.deviation_spin.setValue(float(signal["frequency_deviation_hz"]))
            self._set_combo_text(self.mapping_combo, signal["symbol_mapping"], "symbol mapping")
            self._set_combo_text(self.tx_filter_combo, signal["tx_filter"], "TX filter")
            self.filter_parameter_spin.setValue(float(signal["filter_parameter"]))
            if "input_source" in source:
                self._set_combo_text(
                    self.input_source_combo, source["input_source"], "input source"
                )
            self.pluto_uri_edit.setText(str(source.get("pluto_uri", "")))
            self.capture_center_spin.setValue(
                float(source.get("center_frequency_mhz", 2441.0))
            )
            self.capture_rf_bandwidth_spin.setValue(
                float(source.get("rf_bandwidth_mhz", 8.0))
            )
            self.internal_gain_spin.setValue(
                int(round(float(source.get("internal_gain_db", 30.0))))
            )
            self.external_attenuation_spin.setValue(
                float(source.get("external_attenuation_db", 30.0))
            )
            self.external_gain_spin.setValue(
                float(source.get("external_gain_db", 0.0))
            )
            self.channel_filter_check.setChecked(bool(source["analysis_channel_enabled"]))
            self.analysis_center_spin.setValue(float(source["analysis_center_mhz"]))
            self.analysis_bandwidth_spin.setValue(float(source["analysis_bandwidth_mhz"]))
            self.pattern_search_check.setChecked(bool(pattern["enabled"]))
            self.pattern_name_edit.setText(str(pattern["name"]))
            self._set_combo_text(self.pattern_format_combo, pattern["symbol_format"], "symbol format")
            pattern_symbols = pattern["symbols"]
            if not isinstance(pattern_symbols, list):
                raise TypeError("pattern symbols must be an array")
            self._set_pattern_symbols([int(value) for value in pattern_symbols])
            self.pattern_threshold_auto.setChecked(bool(pattern["threshold_auto"]))
            self.pattern_threshold_spin.setValue(float(pattern["threshold_percent"]))
            self.pattern_meas_only_check.setChecked(bool(pattern["meas_only_if_correct"]))
            self._set_combo_text(
                self.pattern_match_selection_combo,
                pattern.get("match_selection", MatchSelectionPolicy.FIRST.value),
                "match selection",
            )
            self.pattern_match_index_spin.setValue(
                int(pattern.get("match_index", 1))
            )
            self.result_length_spin.setValue(int(result_range["length_symbols"]))
            self._set_combo_text(self.result_reference_combo, result_range["reference"], "result reference")
            self._set_combo_text(self.result_alignment_combo, result_range["alignment"], "result alignment")
            self.result_offset_spin.setValue(int(result_range["offset_symbols"]))
            self.reference_symbol_number_spin.setValue(int(result_range["symbol_number_at_pattern_start"]))
            self.exclude_incomplete_result_check.setChecked(
                bool(result_range.get("exclude_incomplete_result", False))
            )
            self._set_combo_text(self.coarse_sync_combo, demodulation["coarse_synchronization"], "coarse synchronization")
            self._set_combo_text(self.fine_sync_combo, demodulation["fine_synchronization"], "fine synchronization")
            self._set_combo_text(self.bit_order_combo, demodulation["bit_ordering"], "bit ordering")
            self.compensate_drift_check.setChecked(bool(demodulation["compensate_carrier_frequency_drift"]))
            self.compensate_deviation_check.setChecked(bool(demodulation["compensate_fsk_deviation_error"]))
            if signal_capture:
                self.capture_length_spin.setValue(
                    float(signal_capture.get("capture_length", 3.0))
                )
                self._set_combo_text(
                    self.capture_length_unit_combo,
                    signal_capture.get("capture_length_unit", "ms"),
                    "capture length unit",
                )
                oversampling = int(signal_capture.get("samples_per_symbol", 8))
                oversampling_index = self.capture_oversampling_combo.findData(
                    oversampling
                )
                if oversampling_index < 0:
                    raise ValueError(
                        f"unsupported capture oversampling: {oversampling}"
                    )
                self.capture_oversampling_combo.setCurrentIndex(
                    oversampling_index
                )
                self.swap_iq_check.setChecked(
                    bool(signal_capture.get("swap_iq", False))
                )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid measurement configuration: {error}") from error
        self._sync_signal_controls()
        self._sync_analysis_controls()
        self._sync_capture_settings()

    def _save_meas_config_file(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Measurement Configuration",
            self._last_directory("config"),
            "VSA configuration (*.vsaconfig.json);;JSON files (*.json)",
        )
        if not path:
            return
        path = self._with_suffix(path, ".vsaconfig.json")
        self._remember_directory("config", path)
        try:
            save_meas_config(path, self._meas_config_values())
            self.statusBar().showMessage(f"Configuration saved - {Path(path).name}")
        except ValueError as error:
            QtWidgets.QMessageBox.critical(self, "Config Save Error", str(error))

    def _load_meas_config_file(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Measurement Configuration",
            self._last_directory("config"),
            "VSA configuration (*.vsaconfig.json *.json);;All files (*)",
        )
        if not path:
            return
        self._remember_directory("config", path)
        try:
            self._apply_meas_config_values(load_meas_config(path))
            if self._analyze():
                self.statusBar().showMessage(
                    f"Configuration loaded - {Path(path).name}"
                )
        except ValueError as error:
            QtWidgets.QMessageBox.critical(self, "Config Load Error", str(error))

    def _refresh_display_only(self) -> None:
        if self.session.result is None:
            return
        self._update_summary()
        self._update_plots(reset_ranges=False)

    def _plot_widgets(self) -> tuple[tuple[str, pg.PlotWidget], ...]:
        if not hasattr(self, "zero_span_plot"):
            return ()
        return (
            ("iq_power", self.zero_span_plot),
            ("spectrum", self.spectrum_plot),
            ("modulation", self.modulation_plot),
            ("symbol_plot", self.symbol_plot),
        )

    def _apply_mouse_interaction_mode(self) -> None:
        mode = (
            pg.ViewBox.RectMode
            if self.rect_zoom_action.isChecked()
            else pg.ViewBox.PanMode
        )
        for _name, plot in self._plot_widgets():
            plot.getViewBox().setMouseMode(mode)

    def _capture_analysis_plot_ranges(self) -> None:
        captured: dict[str, tuple[list[float], list[float]]] = {}
        for name, plot in self._plot_widgets():
            plot.getViewBox().updateAutoRange()
            x_range, y_range = plot.viewRange()
            captured[name] = (list(x_range), list(y_range))
        self._analysis_plot_ranges = captured

    def _reset_graph_scales(self) -> None:
        for name, plot in self._plot_widgets():
            ranges = self._analysis_plot_ranges.get(name)
            if ranges is None:
                continue
            x_range, y_range = ranges
            plot.setRange(xRange=x_range, yRange=y_range, padding=0.0)

    def _selected_modulation(self) -> ModulationKind:
        return ModulationKind(str(self.modulation_combo.currentData()))

    def _sync_signal_controls(self) -> None:
        modulation = self._selected_modulation()
        self.deviation_spin.setEnabled(modulation.family is ModulationFamily.FSK)
        if modulation is ModulationKind.GFSK:
            self.tx_filter_combo.setCurrentText("Gaussian")
        if hasattr(self, "pattern_symbol_table"):
            self._refresh_pattern_table_format()

    def _sync_analysis_controls(self) -> None:
        enabled = self.channel_filter_check.isChecked()
        self.analysis_center_spin.setEnabled(enabled)
        self.analysis_bandwidth_spin.setEnabled(enabled)

    def _capture_length_s(self) -> float:
        value = float(self.capture_length_spin.value())
        if self.capture_length_unit_combo.currentText() == "Symbols":
            return value / float(self.symbol_rate_spin.value())
        return value / 1e3

    def _input_power_correction(self) -> InputPowerCorrection:
        return InputPowerCorrection(
            calibration_offset_db=-62.0,
            internal_gain_db=self.internal_gain_spin.value(),
            external_attenuation_db=self.external_attenuation_spin.value(),
            external_gain_db=self.external_gain_spin.value(),
        )

    def _pluto_capture_settings(self) -> PlutoCaptureSettings:
        return PlutoCaptureSettings(
            center_frequency_hz=self.capture_center_spin.value() * 1e6,
            symbol_rate_hz=self.symbol_rate_spin.value(),
            samples_per_symbol=int(self.capture_oversampling_combo.currentData()),
            capture_length_s=self._capture_length_s(),
            rf_bandwidth_hz=self.capture_rf_bandwidth_spin.value() * 1e6,
            sdr_uri=self.pluto_uri_edit.text().strip() or None,
            swap_iq=self.swap_iq_check.isChecked(),
            power_correction=self._input_power_correction(),
        )

    def _sync_capture_settings(self, _value: object = None) -> None:
        if not hasattr(self, "capture_oversampling_combo"):
            return
        settings = self._pluto_capture_settings()
        self.capture_sample_rate_label.setText(
            f"{settings.requested_sample_rate_hz / 1e6:.3f} MS/s"
        )
        self.capture_samples_label.setText(f"{settings.capture_samples:,} samples")
        self.capture_usable_bandwidth_label.setText(
            f"{settings.nominal_usable_bandwidth_hz / 1e6:.3f} MHz"
        )
        correction = settings.power_correction
        self.capture_correction_label.setText(
            f"{correction.input_correction_db:+.1f} dB "
            "(Ext ATT - Internal Gain - Ext Gain)"
        )

    def _run_pluto_single(self) -> None:
        if (
            self._pluto_capture_thread is not None
            and self._pluto_capture_thread.isRunning()
        ):
            return
        try:
            settings = self._pluto_capture_settings()
        except ValueError as error:
            QtWidgets.QMessageBox.critical(self, "Pluto Capture Error", str(error))
            return
        self.input_source_combo.setCurrentText("Pluto")
        self.run_single_action.setEnabled(False)
        self.run_single_button.setEnabled(False)
        self.statusBar().showMessage(
            "Capturing Pluto IQ - "
            f"{settings.requested_sample_rate_hz / 1e6:.3f} MS/s, "
            f"{settings.capture_samples:,} samples"
        )
        thread = _PlutoSingleCaptureThread(
            self._pluto_source,
            settings,
            self,
        )
        thread.capture_ready.connect(self._pluto_capture_ready)
        thread.capture_failed.connect(self._pluto_capture_failed)
        thread.finished.connect(self._pluto_capture_stopped)
        thread.finished.connect(thread.deleteLater)
        self._pluto_capture_thread = thread
        thread.start()

    def _pluto_capture_ready(self, recording: object) -> None:
        if not isinstance(recording, IQRecording):
            self._pluto_capture_failed("capture returned an invalid IQ record")
            return
        self.load_recording(recording, self._signal_from_controls())
        self.statusBar().showMessage(
            "Pluto Single complete - "
            f"{recording.sample_count:,} samples, "
            f"{recording.sample_rate_hz / 1e6:.3f} MS/s"
        )

    def _pluto_capture_failed(self, message: str) -> None:
        self.statusBar().showMessage(f"Pluto capture failed: {message}")
        QtWidgets.QMessageBox.critical(self, "Pluto Capture Error", message)

    def _pluto_capture_stopped(self) -> None:
        self._pluto_capture_thread = None
        self.run_single_action.setEnabled(True)
        self.run_single_button.setEnabled(True)

    def _set_analysis_controls_from_recording(self, recording: IQRecording) -> None:
        self.analysis_center_spin.setValue(recording.center_frequency_hz / 1e6)
        usable_hz = min(
            recording.sample_rate_hz,
            recording.usable_bandwidth_hz or recording.sample_rate_hz,
        )
        self.analysis_bandwidth_spin.setMaximum(
            max(0.000001, usable_hz / 1e6 * 0.999)
        )
        self.analysis_bandwidth_spin.setValue(min(1.5, usable_hz / 1e6 * 0.8))

    def _update_analysis_settings(self) -> None:
        enabled = self.channel_filter_check.isChecked()
        self.session.update_settings(
            analysis_center_frequency_hz=(
                self.analysis_center_spin.value() * 1e6 if enabled else None
            ),
            analysis_bandwidth_hz=(
                self.analysis_bandwidth_spin.value() * 1e6 if enabled else None
            ),
        )

    def _signal_from_controls(self) -> SignalDescription:
        modulation = self._selected_modulation()
        return SignalDescription(
            modulation=modulation,
            symbol_rate_hz=self.symbol_rate_spin.value(),
            frequency_deviation_hz=(
                self.deviation_spin.value()
                if modulation.family is ModulationFamily.FSK
                else None
            ),
            tx_filter=self.tx_filter_combo.currentText(),
            filter_parameter=(
                self.filter_parameter_spin.value()
                if self.tx_filter_combo.currentText() != "None"
                else None
            ),
            symbol_mapping=self.mapping_combo.currentText(),
        )

    def _parse_pattern_symbols(self, order: int) -> tuple[int, ...]:
        return self._pattern_symbols_from_table(order)

    def _configure_pattern_analysis(self, signal: SignalDescription) -> None:
        if not self.pattern_search_check.isChecked():
            self.session.configure_pattern_analysis(None)
            return
        search = PatternSearchSettings(
            pattern=KnownPattern(
                symbols=self._parse_pattern_symbols(signal.modulation.order),
                name=self.pattern_name_edit.text(),
            ),
            mode=PatternSearchMode.ON,
            iq_correlation_threshold=self.pattern_threshold_spin.value() / 100.0,
            correlation_threshold_auto=self.pattern_threshold_auto.isChecked(),
            meas_only_if_pattern_symbols_correct=(
                self.pattern_meas_only_check.isChecked()
            ),
            match_selection=MatchSelectionPolicy(
                self.pattern_match_selection_combo.currentData()
            ),
            match_index=self.pattern_match_index_spin.value(),
        )
        result_range = ResultRangeSettings(
            result_length=self.result_length_spin.value(),
            reference=ResultRangeReference(self.result_reference_combo.currentData()),
            alignment=ResultRangeAlignment(self.result_alignment_combo.currentData()),
            offset_symbols=self.result_offset_spin.value(),
            symbol_number_at_reference_start=(
                self.reference_symbol_number_spin.value()
            ),
            exclude_incomplete_result=(
                self.exclude_incomplete_result_check.isChecked()
            ),
        )
        demodulation = DemodulationSettings(
            coarse_synchronization=SynchronizationSource(
                self.coarse_sync_combo.currentText()
            ),
            fine_synchronization=SynchronizationSource(
                self.fine_sync_combo.currentText()
            ),
            bit_ordering=BitOrdering(self.bit_order_combo.currentText()),
            compensate_carrier_frequency_drift=(
                self.compensate_drift_check.isChecked()
            ),
            compensate_fsk_deviation_error=(
                self.compensate_deviation_check.isChecked()
            ),
        )
        self.session.configure_pattern_analysis(search, result_range, demodulation)

    def _set_controls_from_signal(self, signal: SignalDescription) -> None:
        index = self.modulation_combo.findData(signal.modulation.value)
        if index >= 0:
            self.modulation_combo.setCurrentIndex(index)
        self.symbol_rate_spin.setValue(signal.symbol_rate_hz)
        if signal.frequency_deviation_hz is not None:
            self.deviation_spin.setValue(signal.frequency_deviation_hz)
        self.tx_filter_combo.setCurrentText(signal.tx_filter)
        if signal.filter_parameter is not None:
            self.filter_parameter_spin.setValue(signal.filter_parameter)
        self._sync_signal_controls()

    def _load_generated(self, modulation: ModulationKind) -> None:
        self.input_source_combo.setCurrentText("Generated")
        if modulation.family is ModulationFamily.FSK:
            recording, signal = GeneratedIQSource.fsk(
                gaussian_bt=0.5 if modulation is ModulationKind.GFSK else None
            )
        else:
            recording, signal = GeneratedIQSource.psk(modulation=modulation)
        self.session.set_recording(recording)
        self.session.set_signal(signal)
        self._set_analysis_controls_from_recording(recording)
        self._set_controls_from_signal(signal)
        self._analyze()

    def load_recording(self, recording: IQRecording, signal: SignalDescription | None = None) -> None:
        self.session.set_recording(recording)
        self._set_analysis_controls_from_recording(recording)
        if signal is not None:
            self.session.set_signal(signal)
            self._set_controls_from_signal(signal)
        self._analyze()

    def _open_iq(self) -> None:
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
                sample_rate_hz, accepted = QtWidgets.QInputDialog.getDouble(
                    self,
                    "IQ Sample Rate",
                    "Sample Rate (Hz)",
                    self.symbol_rate_spin.value() * 8.0,
                    1.0,
                    100_000_000.0,
                    0,
                )
                if not accepted:
                    return
                recording = FileIQSource.load(path, sample_rate_hz=sample_rate_hz)
            self.input_source_combo.setCurrentText("IQ File")
            self.load_recording(recording, self._signal_from_controls())
        except Exception as error:
            QtWidgets.QMessageBox.critical(self, "IQ Import Error", str(error))

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if (
            self._pluto_capture_thread is not None
            and self._pluto_capture_thread.isRunning()
        ):
            self.statusBar().showMessage(
                "Pluto capture is still running; close again after it completes."
            )
            event.ignore()
            return
        self._pluto_source.close()
        super().closeEvent(event)

    def _analyze(self) -> bool:
        if self.session.recording is None:
            return False
        try:
            signal = self._signal_from_controls()
            self.session.set_signal(signal)
            self._update_analysis_settings()
            self._configure_pattern_analysis(signal)
            result = self.session.analyze()
        except Exception as error:
            self.statusBar().showMessage(f"Analysis failed: {error}")
            return False
        self._update_summary()
        self._update_plots(reset_ranges=True)
        self._update_match_navigation_actions()
        self.statusBar().showMessage(
            f"Analysis complete - {self.session.recording.sample_count:,} samples"
        )
        return True

    def _update_match_navigation_actions(self) -> None:
        pattern_result = self.session.pattern_result
        if pattern_result is None:
            self.previous_result_action.setEnabled(False)
            self.next_result_action.setEnabled(False)
            return
        current = int(pattern_result.metadata.get("selected_match_index", 1))
        count = int(pattern_result.metadata.get("eligible_match_count", 1))
        self.previous_result_action.setEnabled(current > 1)
        self.next_result_action.setEnabled(current < count)

    def _select_adjacent_match(self, direction: int) -> bool:
        pattern_result = self.session.pattern_result
        if pattern_result is None or int(direction) == 0:
            return False
        current = int(pattern_result.metadata.get("selected_match_index", 1))
        count = int(pattern_result.metadata.get("eligible_match_count", 1))
        target = current + (1 if int(direction) > 0 else -1)
        if target < 1 or target > count:
            self._update_match_navigation_actions()
            return False
        self.pattern_match_selection_combo.setCurrentText(
            MatchSelectionPolicy.INDEX.value
        )
        self.pattern_match_index_spin.setValue(target)
        if not self._analyze():
            return False
        selected = self.session.pattern_result
        if selected is not None:
            selected_index = int(
                selected.metadata.get("selected_match_index", target)
            )
            selected_count = int(
                selected.metadata.get("eligible_match_count", count)
            )
            self.statusBar().showMessage(
                f"Selected Result Range {selected_index}/{selected_count}"
            )
        return True

    def _update_summary(self) -> None:
        recording = self.session.recording
        signal = self.session.signal
        if recording is None or signal is None:
            self.summary_label.setText("No capture")
            return
        self.summary_label.setText(
            "  |  ".join(
                (
                    f"Input: {recording.source}",
                    f"Capture: {recording.duration_s * 1e3:.3f} ms",
                    f"Fs: {recording.sample_rate_hz / 1e6:.3f} MS/s",
                    (
                        f"Center: {recording.center_frequency_hz / 1e6:.6f} MHz"
                        if recording.center_frequency_hz
                        else "Center: Baseband"
                    ),
                    f"Mod: {signal.modulation.value}",
                    f"Symbol Rate: {signal.symbol_rate_hz / 1e6:.3f} MSym/s",
                    f"TX Filter: {signal.tx_filter}",
                    *(
                        (
                            "Analysis: "
                            f"{(self.session.settings.analysis_center_frequency_hz or recording.center_frequency_hz) / 1e6:.6f} MHz / "
                            f"{self.session.settings.analysis_bandwidth_hz / 1e6:.3f} MHz",
                        )
                        if self.session.settings.analysis_bandwidth_hz is not None
                        else ()
                    ),
                    (
                        "Amplitude: Cal"
                        if recording.amplitude_calibrated
                        else (
                            "Amplitude: Nominal Pluto"
                            if (
                                recording.metadata.get(
                                    "nominal_pluto_amplitude_inferred", False
                                )
                                or recording.metadata.get("amplitude_reference")
                            )
                            else "Amplitude: Uncal"
                        )
                    ),
                    "SGL",
                    *(
                        (
                            f"CFO: {self.session.pattern_result.carrier_frequency_offset_hz / 1e3:+.3f} kHz",
                            (
                                "Carrier: "
                                f"{((self.session.settings.analysis_center_frequency_hz or recording.center_frequency_hz) + self.session.pattern_result.carrier_frequency_offset_hz) / 1e6:.6f} MHz"
                            ),
                        )
                        if self.session.pattern_result is not None
                        else ()
                    ),
                )
            )
        )

    def _update_plots(self, *, reset_ranges: bool = False) -> None:
        result = self.session.result
        signal = self.session.signal
        if result is None or signal is None:
            return
        if reset_ranges:
            for _name, plot in self._plot_widgets():
                plot.enableAutoRange(enable=True)
        show_corrected = (
            self.corrected_carrier_action.isChecked()
            and self.session.carrier_corrected_result is not None
        )
        display_result = (
            self.session.carrier_corrected_result if show_corrected else result
        )
        capture_slice = _decimation_indices(result.time_s.size)
        self.zero_span_plot.clear()
        self.zero_span_plot.plot(
            result.time_s[capture_slice] * 1e3,
            result.power_dbm[capture_slice],
            pen=pg.mkPen("y", width=1),
        )
        symbol_times_s = (
            self.session.pattern_result.symbol_time_s
            if self.session.pattern_result is not None
            else result.symbol_time_s
        )
        if self.symbol_display_action.isChecked() and symbol_times_s.size:
            symbol_power_dbm = np.interp(
                symbol_times_s, result.time_s, result.power_dbm
            )
            self._plot_symbol_points(
                self.zero_span_plot,
                symbol_times_s * 1e3,
                symbol_power_dbm,
            )
        self._add_pattern_range_overlay(
            self.zero_span_plot, fit_range=reset_ranges
        )
        self.spectrum_plot.clear()
        spectrum_result = (
            self.session.carrier_corrected_pattern_range_result
            if show_corrected
            else self.session.pattern_range_result
        ) or display_result
        self.spectrum_plot.setTitle(
            (
                "Spectrum (Result Range, Carrier Corrected)"
                if show_corrected
                else "Spectrum (Result Range, Raw IQ)"
            )
            if self.session.pattern_range_result is not None
            else "Spectrum"
        )
        analysis_center_hz = float(
            spectrum_result.metadata.get("analysis_center_frequency_hz", 0.0) or 0.0
        )
        if analysis_center_hz:
            spectrum_x = (
                spectrum_result.spectrum_frequency_hz + analysis_center_hz
            ) / 1e6
            self.spectrum_plot.setLabel("bottom", "Frequency (MHz)")
        else:
            spectrum_x = spectrum_result.spectrum_frequency_hz / 1e6
            self.spectrum_plot.setLabel("bottom", "Relative Frequency (MHz)")
        self.spectrum_plot.plot(
            spectrum_x,
            spectrum_result.spectrum_dbfs,
            pen=pg.mkPen("c", width=1),
        )
        self.modulation_plot.clear()
        self.symbol_plot.clear()
        if signal.modulation.family is ModulationFamily.FSK:
            self.modulation_plot.setDownsampling(auto=True, mode="peak")
            self.modulation_plot.setClipToView(True)
            self.modulation_plot.setTitle("Instantaneous Frequency")
            self.modulation_plot.getAxis("left").enableAutoSIPrefix(True)
            self.modulation_plot.getAxis("bottom").enableAutoSIPrefix(True)
            self.modulation_plot.setLabel("left", "Frequency (kHz)")
            self.modulation_plot.setLabel("bottom", "Time (ms)")
            self.modulation_plot.setAspectLocked(False)
            self.modulation_plot.plot(
                display_result.time_s[capture_slice] * 1e3,
                display_result.instantaneous_frequency_hz[capture_slice] / 1e3,
                pen=pg.mkPen("m", width=1),
            )
            if self.symbol_display_action.isChecked() and symbol_times_s.size:
                symbol_frequency_khz = np.interp(
                    symbol_times_s,
                    display_result.time_s,
                    display_result.instantaneous_frequency_hz,
                ) / 1e3
                self._plot_symbol_points(
                    self.modulation_plot,
                    symbol_times_s * 1e3,
                    symbol_frequency_khz,
                )
            if reset_ranges and signal.frequency_deviation_hz is not None:
                y_limit_khz = 1.5 * signal.frequency_deviation_hz / 1e3
                self.modulation_plot.setYRange(
                    -y_limit_khz, y_limit_khz, padding=0.0
                )
            self._add_pattern_range_overlay(
                self.modulation_plot, fit_range=reset_ranges
            )

            self.symbol_plot.setTitle("FSK Symbol Phase Difference")
            self.symbol_plot.getAxis("left").enableAutoSIPrefix(False)
            self.symbol_plot.getAxis("bottom").enableAutoSIPrefix(False)
            self.symbol_plot.setLabel("left", "Q")
            self.symbol_plot.setLabel("bottom", "I")
            self.symbol_plot.setAspectLocked(True, ratio=1.0)
            measured_frequency_hz = np.real(
                self.session.pattern_result.measured_symbols
                if self.session.pattern_result is not None
                else display_result.measured_symbols
            )
            phase_difference = _fsk_phase_difference_symbols(
                display_result.iq,
                display_result.time_s,
                symbol_times_s,
                measured_frequency_hz,
                signal.symbol_rate_hz,
            )
            phase_slice = _decimation_indices(
                phase_difference.size, maximum=20_000
            )
            self.symbol_plot.plot(
                phase_difference.real[phase_slice],
                phase_difference.imag[phase_slice],
                pen=None,
                symbol="o",
                symbolSize=6,
                symbolBrush=pg.mkBrush("y"),
            )
            unit_angle = np.linspace(0.0, 2.0 * np.pi, 361)
            self.symbol_plot.plot(
                np.cos(unit_angle),
                np.sin(unit_angle),
                pen=pg.mkPen((120, 120, 120, 110), width=1),
            )
            if reset_ranges:
                symbol_limit = (
                    max(
                        1.25,
                        1.15
                        * float(np.percentile(np.abs(phase_difference), 99.5)),
                    )
                    if phase_difference.size
                    else 1.25
                )
                self.symbol_plot.setXRange(-symbol_limit, symbol_limit, padding=0.0)
                self.symbol_plot.setYRange(-symbol_limit, symbol_limit, padding=0.0)
            summary = f"Frequency Error: {result.frequency_error_hz or 0.0:.1f} Hz"
        else:
            self.modulation_plot.setDownsampling(auto=False)
            self.modulation_plot.setClipToView(False)
            self.modulation_plot.setTitle("IQ Trajectory")
            self.modulation_plot.getAxis("left").enableAutoSIPrefix(False)
            self.modulation_plot.getAxis("bottom").enableAutoSIPrefix(False)
            self.modulation_plot.setLabel("left", "Q")
            self.modulation_plot.setLabel("bottom", "I")
            self.modulation_plot.setAspectLocked(True, ratio=1.0)
            processed_iq, processed_rate_hz = prepare_psk_iq(
                display_result.iq,
                sample_rate_hz=float(
                    display_result.metadata.get(
                        "analysis_sample_rate_hz",
                        self.session.recording.sample_rate_hz,
                    )
                ),
                symbol_rate_hz=signal.symbol_rate_hz,
                tx_filter=signal.tx_filter,
                filter_parameter=signal.filter_parameter,
            )
            processed_time_s = (
                np.arange(processed_iq.size, dtype=np.float64)
                / processed_rate_hz
            )
            symbol_iq = np.interp(
                symbol_times_s,
                processed_time_s,
                np.real(processed_iq),
            ) + 1j * np.interp(
                symbol_times_s,
                processed_time_s,
                np.imag(processed_iq),
            )
            trajectory_rms = (
                float(np.sqrt(np.mean(np.abs(symbol_iq) ** 2)))
                if symbol_iq.size
                else 1.0
            )
            if np.isfinite(trajectory_rms) and trajectory_rms > 0.0:
                processed_iq = processed_iq / trajectory_rms
                symbol_iq = symbol_iq / trajectory_rms
            pattern_result = self.session.pattern_result
            if pattern_result is not None:
                in_result_range = (
                    (processed_time_s >= pattern_result.result_start_time_s)
                    & (processed_time_s < pattern_result.result_stop_time_s)
                )
                trajectory_iq = processed_iq[in_result_range]
            else:
                trajectory_iq = processed_iq
            trajectory_slice = _decimation_indices(
                trajectory_iq.size, maximum=20_000
            )
            trajectory_iq = trajectory_iq[trajectory_slice]
            self.modulation_plot.plot(
                trajectory_iq.real,
                trajectory_iq.imag,
                pen=pg.mkPen((255, 210, 40, 170), width=1),
            )
            if self.symbol_display_action.isChecked() and symbol_times_s.size:
                self._plot_symbol_points(
                    self.modulation_plot,
                    symbol_iq.real,
                    symbol_iq.imag,
                )
            trajectory_limit = (
                max(
                    1.25,
                    1.15 * float(np.percentile(np.abs(trajectory_iq), 99.5)),
                )
                if trajectory_iq.size
                else 1.25
            )
            if reset_ranges:
                self.modulation_plot.setXRange(
                    -trajectory_limit, trajectory_limit, padding=0.0
                )
                self.modulation_plot.setYRange(
                    -trajectory_limit, trajectory_limit, padding=0.0
                )

            self.symbol_plot.setTitle("Constellation")
            self.symbol_plot.getAxis("left").enableAutoSIPrefix(False)
            self.symbol_plot.getAxis("bottom").enableAutoSIPrefix(False)
            self.symbol_plot.setLabel("left", "Q")
            self.symbol_plot.setLabel("bottom", "I")
            self.symbol_plot.setAspectLocked(True, ratio=1.0)
            constellation_symbols = (
                self.session.pattern_result.measured_symbols
                if self.session.pattern_result is not None
                else display_result.measured_symbols
            )
            constellation_symbols = _constellation_display_symbols(
                signal.modulation, constellation_symbols
            )
            self.symbol_plot.plot(
                constellation_symbols.real,
                constellation_symbols.imag,
                pen=None,
                symbol="o",
                symbolSize=6,
                symbolBrush=pg.mkBrush("y"),
            )
            # clear() retains the previous ViewBox range.  Explicitly reset
            # both axes because an FSK frequency range or an earlier malformed
            # constellation can otherwise leave all unit-circle symbols offscreen.
            if reset_ranges:
                self.symbol_plot.setXRange(-1.25, 1.25, padding=0.0)
                self.symbol_plot.setYRange(-1.25, 1.25, padding=0.0)
            summary = f"EVM RMS: {result.evm_rms_percent or 0.0:.4f} %"
        pattern_result = self.session.pattern_result
        symbols = (
            pattern_result.decoded_symbols
            if pattern_result is not None
            else result.decoded_symbols
        )
        if pattern_result is not None:
            display_name = "Carrier Corrected" if show_corrected else "Raw IQ"
            recording = self.session.recording
            analysis_center_hz = (
                self.session.settings.analysis_center_frequency_hz
                if self.session.settings.analysis_center_frequency_hz is not None
                else (recording.center_frequency_hz if recording is not None else 0.0)
            )
            reported_drift_hz_per_s = (
                float(
                    pattern_result.metadata.get(
                        "candidate_drift_hz_per_s",
                        pattern_result.carrier_frequency_drift_hz_per_s,
                    )
                )
                if signal.modulation.family is ModulationFamily.FSK
                else pattern_result.carrier_frequency_drift_hz_per_s
            )
            summary_rows = [
                    ("Modulation", signal.modulation.value),
                    (
                        "Pattern Symbols Correct",
                        "Yes" if pattern_result.pattern_symbol_errors == 0 else "No",
                    ),
                    ("I/Q Correlation", f"{pattern_result.correlation * 100.0:.2f} %"),
                    ("CFO", f"{pattern_result.carrier_frequency_offset_hz / 1e3:+.3f} kHz"),
                    (
                        "Estimated Carrier",
                        f"{(analysis_center_hz + pattern_result.carrier_frequency_offset_hz) / 1e6:.6f} MHz",
                    ),
                    (
                        "Carrier Drift",
                        f"{reported_drift_hz_per_s / 1e6:+.3f} kHz/ms",
                    ),
            ]
            if signal.modulation.family is ModulationFamily.PSK:
                rate_error_ppm = pattern_result.metadata.get("symbol_rate_error_ppm")
                sync_evm = pattern_result.metadata.get("synchronization_evm_rms")
                if rate_error_ppm is not None:
                    summary_rows.append(
                        ("Symbol Rate Error", f"{float(rate_error_ppm):+.2f} ppm")
                    )
                if sync_evm is not None:
                    summary_rows.append(
                        ("Sync EVM RMS", f"{float(sync_evm) * 100.0:.2f} %")
                    )
            elif signal.modulation.family is ModulationFamily.FSK:
                timing_offset = pattern_result.metadata.get(
                    "fractional_timing_offset_samples"
                )
                timing_symbols = pattern_result.metadata.get(
                    "fractional_timing_offset_symbols"
                )
                applied_timing = pattern_result.metadata.get(
                    "applied_timing_offset_samples"
                )
                timing_accepted = pattern_result.metadata.get(
                    "timing_correction_accepted"
                )
                frequency_residual = pattern_result.metadata.get(
                    "frequency_model_residual_rms_hz"
                )
                no_drift_residual = pattern_result.metadata.get(
                    "frequency_model_no_drift_residual_rms_hz"
                )
                timing_confidence = pattern_result.metadata.get(
                    "timing_confidence"
                )
                deviation_error = pattern_result.metadata.get(
                    "frequency_deviation_error_percent"
                )
                drift_accepted = pattern_result.metadata.get(
                    "drift_model_accepted"
                )
                candidate_drift = pattern_result.metadata.get(
                    "candidate_drift_hz_per_s"
                )
                drift_reason = pattern_result.metadata.get(
                    "drift_rejection_reason"
                )
                if timing_offset is not None and timing_symbols is not None:
                    timing_status = (
                        ""
                        if timing_accepted is not False
                        else f" (rejected; applied {float(applied_timing or 0.0):+.3f})"
                    )
                    summary_rows.append(
                        (
                            "Fractional Timing",
                            f"{float(timing_offset):+.3f} sample "
                            f"({float(timing_symbols) * 100.0:+.2f} % sym)"
                            f"{timing_status}",
                        )
                    )
                if frequency_residual is not None:
                    residual_status = (
                        ""
                        if no_drift_residual is None
                        else f" / no drift {float(no_drift_residual) / 1e3:.3f}"
                    )
                    summary_rows.append(
                        (
                            "Frequency Fit RMS",
                            f"{float(frequency_residual) / 1e3:.3f}"
                            f"{residual_status} kHz",
                        )
                    )
                if timing_confidence is not None:
                    summary_rows.append(
                        ("Timing Confidence", f"{float(timing_confidence):.3f}")
                    )
                if deviation_error is not None:
                    summary_rows.append(
                        ("Deviation Error", f"{float(deviation_error):+.2f} %")
                    )
                summary_rows.append(
                    (
                        "Drift Model",
                        (
                            "Accepted"
                            if drift_accepted
                            else f"Rejected ({drift_reason or 'quality gate'})"
                        ),
                    )
                )
                if candidate_drift is not None:
                    summary_rows.append(
                        (
                            "Applied Drift",
                            f"{pattern_result.carrier_frequency_drift_hz_per_s / 1e6:+.3f} kHz/ms",
                        )
                    )
            summary_rows.extend(
                (
                    ("Display", display_name),
                    (
                        "Match Selection",
                        f"{pattern_result.metadata.get('match_selection_policy', 'First')} "
                        f"({pattern_result.metadata.get('selected_match_index', 1)}/"
                        f"{pattern_result.metadata.get('eligible_match_count', 1)})",
                    ),
                    ("Result Symbols", str(symbols.size)),
                )
            )
            self._set_result_summary(tuple(summary_rows))
        elif self.session.pattern_error:
            self._set_result_summary(
                (
                    ("Modulation", signal.modulation.value),
                    ("Pattern Symbols Correct", "No"),
                    ("Pattern Error", self.session.pattern_error),
                    ("Result Symbols", str(symbols.size)),
                )
            )
        else:
            self._set_result_summary(
                (
                    ("Modulation", signal.modulation.value),
                    ("Result Symbols", str(result.decoded_symbols.size)),
                    (
                        "Frequency Error"
                        if signal.modulation.family is ModulationFamily.FSK
                        else "EVM RMS",
                        summary.split(":", 1)[-1].strip(),
                    ),
                )
            )
        shown = symbols[:2048]
        row_count = int(np.ceil(shown.size / 10.0))
        self.symbol_table.clearContents()
        self.symbol_table.setRowCount(row_count)
        self.symbol_table.setVerticalHeaderLabels(
            [str(row * 10) for row in range(row_count)]
        )
        for index, symbol in enumerate(shown):
            item = QtWidgets.QTableWidgetItem(str(int(symbol)))
            item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            if pattern_result is not None and index < pattern_result.symbol_time_s.size:
                symbol_time_s = float(pattern_result.symbol_time_s[index])
                if (
                    pattern_result.pattern_start_time_s
                    <= symbol_time_s
                    < pattern_result.pattern_stop_time_s
                ):
                    item.setBackground(QtGui.QColor(24, 112, 55))
                    item.setForeground(QtGui.QColor(255, 255, 255))
            self.symbol_table.setItem(index // 10, index % 10, item)
        self.symbol_table.setToolTip(
            f"Showing {shown.size} of {symbols.size} result-range symbols"
        )
        if reset_ranges:
            self._capture_analysis_plot_ranges()

    @staticmethod
    def _plot_symbol_points(
        plot: pg.PlotWidget, x_values: np.ndarray, y_values: np.ndarray
    ) -> None:
        selection = _decimation_indices(len(x_values), maximum=20_000)
        plot.plot(
            np.asarray(x_values)[selection],
            np.asarray(y_values)[selection],
            pen=None,
            symbol="o",
            symbolSize=7,
            symbolBrush=pg.mkBrush(70, 255, 145, 230),
            symbolPen=pg.mkPen(10, 35, 20, 230, width=1),
        )

    def _set_result_summary(self, rows: tuple[tuple[str, str], ...]) -> None:
        self.result_summary.clearContents()
        self.result_summary.setRowCount(len(rows))
        for row, (name, value) in enumerate(rows):
            name_item = QtWidgets.QTableWidgetItem(name)
            value_item = QtWidgets.QTableWidgetItem(value)
            name_item.setTextAlignment(
                QtCore.Qt.AlignmentFlag.AlignLeft
                | QtCore.Qt.AlignmentFlag.AlignVCenter
            )
            value_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.result_summary.setItem(row, 0, name_item)
            self.result_summary.setItem(row, 1, value_item)

    def _add_pattern_range_overlay(
        self, plot: pg.PlotWidget, *, fit_range: bool = True
    ) -> None:
        pattern = self.session.pattern_result
        if pattern is None:
            return
        result_region = pg.LinearRegionItem(
            values=(
                pattern.result_start_time_s * 1e3,
                pattern.result_stop_time_s * 1e3,
            ),
            movable=False,
            brush=pg.mkBrush(60, 130, 255, 35),
            pen=pg.mkPen(80, 150, 255, 150),
        )
        result_region.setZValue(-5)
        plot.addItem(result_region)
        pattern_region = pg.LinearRegionItem(
            values=(
                pattern.pattern_start_time_s * 1e3,
                pattern.pattern_stop_time_s * 1e3,
            ),
            movable=False,
            brush=pg.mkBrush(40, 220, 100, 65),
            pen=pg.mkPen(40, 240, 120, 190),
        )
        pattern_region.setZValue(-4)
        plot.addItem(pattern_region)
        marker = pg.InfiniteLine(
            pos=pattern.pattern_start_time_s * 1e3,
            angle=90,
            movable=False,
            pen=pg.mkPen(80, 255, 130, 220, width=2),
            label="Pattern Start",
            labelOpts={"position": 0.92, "color": (120, 255, 160)},
        )
        plot.addItem(marker)
        duration_ms = (
            pattern.result_stop_time_s - pattern.result_start_time_s
        ) * 1e3
        margin_ms = max(duration_ms * 0.1, 1e-9)
        if fit_range:
            plot.setXRange(
                pattern.result_start_time_s * 1e3 - margin_ms,
                pattern.result_stop_time_s * 1e3 + margin_ms,
                padding=0.0,
            )
