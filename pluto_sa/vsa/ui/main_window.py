"""R&S-inspired multi-window shell for the first offline VSA milestone."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from pluto_sa.vsa.model import IQRecording, ModulationFamily, ModulationKind, SignalDescription
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
        self.symbol_table = QtWidgets.QPlainTextEdit()
        self.symbol_table.setReadOnly(True)
        self.symbol_table.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
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
        signal_form.addRow("Modulation", self.modulation_combo)
        signal_form.addRow("Symbol Rate", self.symbol_rate_spin)
        signal_form.addRow("FSK Deviation", self.deviation_spin)
        self.modulation_combo.currentIndexChanged.connect(self._sync_signal_controls)
        toolbox.addItem(signal_page, "Signal Description")

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
            tx_filter="Gaussian" if modulation is ModulationKind.GFSK else "None",
            filter_parameter=0.5 if modulation is ModulationKind.GFSK else None,
        )

    def _set_controls_from_signal(self, signal: SignalDescription) -> None:
        index = self.modulation_combo.findData(signal.modulation.value)
        if index >= 0:
            self.modulation_combo.setCurrentIndex(index)
        self.symbol_rate_spin.setValue(signal.symbol_rate_hz)
        if signal.frequency_deviation_hz is not None:
            self.deviation_spin.setValue(signal.frequency_deviation_hz)
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
        self._set_controls_from_signal(signal)
        self._analyze()

    def load_recording(self, recording: IQRecording, signal: SignalDescription | None = None) -> None:
        self.session.set_recording(recording)
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
            self.session.set_signal(self._signal_from_controls())
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
                    f"Mod: {signal.modulation.value}",
                    f"Symbol Rate: {signal.symbol_rate_hz / 1e6:.3f} MSym/s",
                    f"TX Filter: {signal.tx_filter}",
                    "Amplitude: Cal" if recording.amplitude_calibrated else "Amplitude: Uncal",
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
        self.zero_span_plot.plot(
            result.time_s[capture_slice] * 1e3,
            result.power_dbm[capture_slice],
            pen=pg.mkPen("y", width=1),
        )
        self.spectrum_plot.clear()
        self.spectrum_plot.plot(
            result.spectrum_frequency_hz / 1e6,
            result.spectrum_dbfs,
            pen=pg.mkPen("c", width=1),
        )
        self.modulation_plot.clear()
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
        shown = result.decoded_symbols[:2048]
        lines = [f"{index:6d}  {int(symbol):3d}" for index, symbol in enumerate(shown)]
        if result.decoded_symbols.size > shown.size:
            lines.append(f"... {result.decoded_symbols.size - shown.size} more symbols")
        self.symbol_table.setPlainText("Index  Symbol\n" + "\n".join(lines))
