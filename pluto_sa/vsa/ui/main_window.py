"""R&S-inspired multi-window shell for the first offline VSA milestone."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from pluto_sa.vsa.model import IQRecording, ModulationFamily, ModulationKind, SignalDescription
from pluto_sa.vsa.pattern import (
    BitOrdering,
    DemodulationSettings,
    KnownPattern,
    PatternSearchMode,
    PatternSearchSettings,
    ResultRangeAlignment,
    ResultRangeReference,
    ResultRangeSettings,
    SynchronizationSource,
)
from pluto_sa.vsa.session import VSASession
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


class VSAWindow(QtWidgets.QMainWindow):
    """One VSA measurement session with detachable result windows."""

    def __init__(self, session: VSASession | None = None) -> None:
        super().__init__()
        self.session = session or VSASession()
        self.setWindowTitle("Pluto VSA - Offline FSK / PSK")
        self.resize(1600, 960)
        self.setDockOptions(
            QtWidgets.QMainWindow.DockOption.AllowNestedDocks
            | QtWidgets.QMainWindow.DockOption.AllowTabbedDocks
            | QtWidgets.QMainWindow.DockOption.AnimatedDocks
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
        analyze_action = QtGui.QAction("Refresh Analysis", self)
        analyze_action.setShortcut("F5")
        analyze_action.triggered.connect(self._analyze)
        run_menu.addAction(analyze_action)

        display_menu = self.menuBar().addMenu("Display Config")
        self._display_menu = display_menu

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
        self.setCentralWidget(self.zero_span_plot)

        self.spectrum_plot = self._make_plot("Spectrum", "Magnitude (dBFS)", "Relative Frequency (MHz)")
        self.spectrum_dock = self._dock("Spectrum", self.spectrum_plot)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.spectrum_dock)

        self.modulation_plot = self._make_plot("Modulation", "Q", "I")
        self.modulation_dock = self._dock("Modulation", self.modulation_plot)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.modulation_dock)

        symbol_container = QtWidgets.QWidget()
        symbol_layout = QtWidgets.QVBoxLayout(symbol_container)
        symbol_layout.setContentsMargins(6, 6, 6, 6)
        self.result_summary = QtWidgets.QLabel("No result")
        self.symbol_table = QtWidgets.QTableWidget(0, 10)
        self.symbol_table.setHorizontalHeaderLabels([str(index) for index in range(10)])
        self.symbol_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.symbol_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectItems
        )
        self.symbol_table.setAlternatingRowColors(True)
        self.symbol_table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self.symbol_table.verticalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        symbol_layout.addWidget(self.result_summary)
        symbol_layout.addWidget(self.symbol_table, 1)
        self.symbol_dock = self._dock("Symbol Table", symbol_container)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.symbol_dock)
        self.splitDockWidget(
            self.modulation_dock,
            self.symbol_dock,
            QtCore.Qt.Orientation.Vertical,
        )

    def _build_configuration(self) -> None:
        toolbox = QtWidgets.QToolBox()

        source_page = QtWidgets.QWidget()
        source_layout = QtWidgets.QVBoxLayout(source_page)
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
        toolbox.addItem(source_page, "Input / Frontend")

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
        toolbox.addItem(signal_page, "Signal Description")

        pattern_page = QtWidgets.QWidget()
        pattern_form = QtWidgets.QFormLayout(pattern_page)
        self.pattern_search_check = QtWidgets.QCheckBox("Pattern Search On")
        self.pattern_name_edit = QtWidgets.QLineEdit("Known Pattern")
        self.pattern_format_combo = QtWidgets.QComboBox()
        self.pattern_format_combo.addItems(("Binary", "Decimal", "Hexadecimal"))
        self.pattern_symbols_edit = QtWidgets.QLineEdit("01010101")
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
        pattern_form.addRow(self.pattern_search_check)
        pattern_form.addRow("Name", self.pattern_name_edit)
        pattern_form.addRow("Symbol Format", self.pattern_format_combo)
        pattern_form.addRow("Symbols", self.pattern_symbols_edit)
        pattern_form.addRow("I/Q Correlation Threshold", self.pattern_threshold_spin)
        pattern_form.addRow(self.pattern_threshold_auto)
        pattern_form.addRow(self.pattern_meas_only_check)
        self.pattern_threshold_auto.toggled.connect(
            lambda checked: self.pattern_threshold_spin.setEnabled(not checked)
        )
        toolbox.addItem(pattern_page, "Pattern Search")

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
        range_form.addRow("Result Length (Symbols)", self.result_length_spin)
        range_form.addRow("Reference", self.result_reference_combo)
        range_form.addRow("Alignment", self.result_alignment_combo)
        range_form.addRow("Offset (Symbols)", self.result_offset_spin)
        range_form.addRow(
            "Symbol Number at Pattern Start", self.reference_symbol_number_spin
        )
        toolbox.addItem(range_page, "Result Range")

        demod_page = QtWidgets.QWidget()
        demod_form = QtWidgets.QFormLayout(demod_page)
        self.coarse_sync_combo = QtWidgets.QComboBox()
        self.coarse_sync_combo.addItems(("Auto", "Detected Data", "Pattern"))
        self.fine_sync_combo = QtWidgets.QComboBox()
        self.fine_sync_combo.addItems(("Auto", "Detected Data", "Pattern"))
        self.bit_order_combo = QtWidgets.QComboBox()
        self.bit_order_combo.addItems(("MSB", "LSB"))
        self.compensate_drift_check = QtWidgets.QCheckBox("Carrier Frequency Drift")
        self.compensate_drift_check.setChecked(True)
        self.compensate_deviation_check = QtWidgets.QCheckBox("FSK Deviation Error")
        self.compensate_deviation_check.setChecked(True)
        for control in (
            self.coarse_sync_combo,
            self.fine_sync_combo,
            self.compensate_drift_check,
            self.compensate_deviation_check,
        ):
            control.setEnabled(False)
            control.setToolTip("R&S-compatible setting contract; DSP connection is planned.")
        demod_form.addRow("Coarse Synchronization", self.coarse_sync_combo)
        demod_form.addRow("Fine Synchronization", self.fine_sync_combo)
        demod_form.addRow("Bit Ordering", self.bit_order_combo)
        demod_form.addRow("Compensate for", self.compensate_drift_check)
        demod_form.addRow("", self.compensate_deviation_check)
        toolbox.addItem(demod_page, "Demodulation")

        run_page = QtWidgets.QWidget()
        run_layout = QtWidgets.QVBoxLayout(run_page)
        refresh_button = QtWidgets.QPushButton("Refresh Analysis")
        refresh_button.clicked.connect(self._analyze)
        run_layout.addWidget(refresh_button)
        run_layout.addWidget(QtWidgets.QLabel("Offline milestone: Refresh reuses the current capture."))
        run_layout.addStretch(1)
        toolbox.addItem(run_page, "Sweep / Run")

        config_dock = QtWidgets.QDockWidget("Meas Config", self)
        config_dock.setObjectName("vsa-meas-config")
        config_dock.setWidget(toolbox)
        config_dock.setMinimumWidth(280)
        config_dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, config_dock)
        self._display_menu.addSeparator()
        self._display_menu.addAction(config_dock.toggleViewAction())

    def _selected_modulation(self) -> ModulationKind:
        return ModulationKind(str(self.modulation_combo.currentData()))

    def _sync_signal_controls(self) -> None:
        modulation = self._selected_modulation()
        self.deviation_spin.setEnabled(modulation.family is ModulationFamily.FSK)
        if modulation is ModulationKind.GFSK:
            self.tx_filter_combo.setCurrentText("Gaussian")

    def _sync_analysis_controls(self) -> None:
        enabled = self.channel_filter_check.isChecked()
        self.analysis_center_spin.setEnabled(enabled)
        self.analysis_bandwidth_spin.setEnabled(enabled)

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
        text = self.pattern_symbols_edit.text().strip()
        if not text:
            raise ValueError("Pattern Symbols is empty")
        symbol_format = self.pattern_format_combo.currentText()
        if symbol_format == "Binary":
            compact = "".join(text.replace(",", " ").split())
            if any(character not in "01" for character in compact):
                raise ValueError("Binary pattern may contain only 0 and 1")
            width = int(round(np.log2(order)))
            if len(compact) % width:
                raise ValueError(f"Binary pattern length must be a multiple of {width}")
            values = tuple(
                int(compact[index : index + width], 2)
                for index in range(0, len(compact), width)
            )
        else:
            tokens = text.replace(",", " ").split()
            base = 16 if symbol_format == "Hexadecimal" else 10
            values = tuple(int(token, base) for token in tokens)
        if any(value < 0 or value >= order for value in values):
            raise ValueError(f"Pattern symbol must be between 0 and {order - 1}")
        return values

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
        )
        result_range = ResultRangeSettings(
            result_length=self.result_length_spin.value(),
            reference=ResultRangeReference(self.result_reference_combo.currentData()),
            alignment=ResultRangeAlignment(self.result_alignment_combo.currentData()),
            offset_symbols=self.result_offset_spin.value(),
            symbol_number_at_reference_start=(
                self.reference_symbol_number_spin.value()
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
            "",
            "IQ recordings (*.npz *.npy *.cf32 *.bin);;All files (*)",
        )
        if not path:
            return
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
            self.load_recording(recording, self._signal_from_controls())
        except Exception as error:
            QtWidgets.QMessageBox.critical(self, "IQ Import Error", str(error))

    def _analyze(self) -> None:
        if self.session.recording is None:
            return
        try:
            signal = self._signal_from_controls()
            self.session.set_signal(signal)
            self._update_analysis_settings()
            self._configure_pattern_analysis(signal)
            result = self.session.analyze()
        except Exception as error:
            self.statusBar().showMessage(f"Analysis failed: {error}")
            return
        self._update_summary()
        self._update_plots()
        self.statusBar().showMessage(
            f"Analysis complete - {self.session.recording.sample_count:,} samples"
        )

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
                )
            )
        )

    def _update_plots(self) -> None:
        result = self.session.result
        signal = self.session.signal
        if result is None or signal is None:
            return
        capture_slice = _decimation_indices(result.time_s.size)
        self.zero_span_plot.clear()
        self.zero_span_plot.enableAutoRange(axis=pg.ViewBox.XAxis, enable=True)
        self.zero_span_plot.plot(
            result.time_s[capture_slice] * 1e3,
            result.power_dbm[capture_slice],
            pen=pg.mkPen("y", width=1),
        )
        self._add_pattern_range_overlay(self.zero_span_plot)
        self.spectrum_plot.clear()
        spectrum_result = self.session.pattern_range_result or result
        self.spectrum_plot.setTitle(
            "Spectrum (Result Range)"
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
        self.modulation_plot.enableAutoRange(axis=pg.ViewBox.XAxis, enable=True)
        if signal.modulation.family is ModulationFamily.FSK:
            self.modulation_plot.setTitle("Instantaneous Frequency")
            self.modulation_plot.setLabel("left", "Frequency (kHz)")
            self.modulation_plot.setLabel("bottom", "Time (ms)")
            self.modulation_plot.setAspectLocked(False)
            self.modulation_plot.plot(
                result.time_s[capture_slice] * 1e3,
                result.instantaneous_frequency_hz[capture_slice] / 1e3,
                pen=pg.mkPen("m", width=1),
            )
            if signal.frequency_deviation_hz is not None:
                y_limit_khz = 1.5 * signal.frequency_deviation_hz / 1e3
                self.modulation_plot.setYRange(
                    -y_limit_khz, y_limit_khz, padding=0.0
                )
            self._add_pattern_range_overlay(self.modulation_plot)
            summary = f"Frequency Error: {result.frequency_error_hz or 0.0:.1f} Hz"
        else:
            self.modulation_plot.setTitle("Constellation")
            self.modulation_plot.setLabel("left", "Q")
            self.modulation_plot.setLabel("bottom", "I")
            self.modulation_plot.setAspectLocked(True, ratio=1.0)
            self.modulation_plot.plot(
                result.measured_symbols.real,
                result.measured_symbols.imag,
                pen=None,
                symbol="o",
                symbolSize=6,
                symbolBrush=pg.mkBrush("y"),
            )
            summary = f"EVM RMS: {result.evm_rms_percent or 0.0:.4f} %"
        self.result_summary.setText(
            f"{signal.modulation.value} | Symbols: {result.decoded_symbols.size} | {summary}"
        )
        pattern_result = self.session.pattern_result
        symbols = (
            pattern_result.decoded_symbols
            if pattern_result is not None
            else result.decoded_symbols
        )
        if pattern_result is not None:
            self.result_summary.setText(
                f"{signal.modulation.value} | Pattern Symbols Correct: "
                f"{'Yes' if pattern_result.pattern_symbol_errors == 0 else 'No'} | "
                f"I/Q Correlation: {pattern_result.correlation * 100.0:.2f}% | "
                "Match: Strongest | "
                f"Result Symbols: {symbols.size}"
            )
        elif self.session.pattern_error:
            self.result_summary.setText(
                f"{signal.modulation.value} | Pattern Symbols Correct: No | "
                f"{self.session.pattern_error}"
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
            self.symbol_table.setItem(index // 10, index % 10, item)
        self.symbol_table.setToolTip(
            f"Showing {shown.size} of {symbols.size} result-range symbols"
        )

    def _add_pattern_range_overlay(self, plot: pg.PlotWidget) -> None:
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
        plot.setXRange(
            pattern.result_start_time_s * 1e3 - margin_ms,
            pattern.result_stop_time_s * 1e3 + margin_ms,
            padding=0.0,
        )
