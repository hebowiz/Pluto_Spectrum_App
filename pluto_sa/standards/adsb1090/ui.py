"""Dedicated 1090ES result workspace using the shared IQ recording contract."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from pluto_sa.config.input_frontend import InputPowerCorrection
from pluto_sa.standards.adsb1090.analysis import ADSB1090Analyzer
from pluto_sa.standards.adsb1090.model import ADSB1090AnalysisResult, ADSB1090Message
from pluto_sa.vsa.model import IQRecording
from pluto_sa.vsa.pluto_source import (
    CaptureCancelledError,
    PlutoCaptureSettings,
    PlutoLiveSource,
)
from pluto_sa.vsa.sources import FileIQSource
from pluto_sa.vsa.ui.measurement_chrome import (
    install_measurement_plot_menu,
    make_measurement_dock,
    make_measurement_plot,
)


_TRACE_COLOR = "y"
_BIT_COLOR = (0, 255, 160)
_ZERO_BIT_COLOR = (0, 210, 255)
_PACKET_GROUP_GAP_S = 0.020
_STREAM_BLOCK_DURATION_S = 0.050
_STREAM_OVERLAP_S = 160e-6
_SINGLE_PRETRIGGER_S = 1e-3
_IQ_POWER_DISPLAY_FLOOR_DBM = -140.0
_MODE_S_HEADER_FIELDS = (
    "flight_status",
    "capability",
    "control_field",
    "vertical_status",
    "cross_link_capability",
    "sensitivity_level",
)
_FLIGHT_STATUS_DESCRIPTIONS = {
    0: "No alert, no SPI, airborne",
    1: "No alert, no SPI, on ground",
    2: "Alert, no SPI, airborne",
    3: "Alert, no SPI, on ground",
    4: "Alert, SPI",
    5: "No alert, SPI",
    6: "Reserved",
    7: "Not assigned",
}


@dataclass(frozen=True)
class _ADSBCaptureBatch:
    recording: IQRecording


@dataclass(frozen=True)
class _ADSBPacketEntry:
    message: ADSB1090Message
    result: ADSB1090AnalysisResult
    recording: IQRecording
    elapsed_s: float
    wall_time: datetime
    on_pulse_power_dbm: float


class _ADSBPlutoCaptureThread(QtCore.QThread):
    capture_ready = QtCore.Signal(object)
    capture_failed = QtCore.Signal(str)
    capture_cancelled = QtCore.Signal()

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
        fresh = True
        stream_settings = replace(
            self._settings,
            capture_length_s=_STREAM_BLOCK_DURATION_S,
        )
        while not self.isInterruptionRequested():
            try:
                recording = self._source.capture_single(
                    stream_settings,
                    cancelled=self.isInterruptionRequested,
                    fresh=fresh,
                )
            except CaptureCancelledError:
                self.capture_cancelled.emit()
                break
            except Exception as error:
                self.capture_failed.emit(str(error))
                break
            self.capture_ready.emit(
                _ADSBCaptureBatch(recording=recording)
            )
            fresh = False


class ADSB1090Window(QtWidgets.QMainWindow):
    """Protocol workspace kept separate from generic modulation analysis."""

    analysis_mode_requested = QtCore.Signal(str)
    application_close_requested = QtCore.Signal()

    def __init__(
        self,
        recording: IQRecording | None = None,
        pluto_source: PlutoLiveSource | None = None,
        owns_pluto_source: bool = True,
    ) -> None:
        super().__init__()
        self.setWindowTitle("Pluto VSA - ADS-B 1090ES")
        self.resize(1400, 850)
        self.recording = recording
        self.result: ADSB1090AnalysisResult | None = None
        self._analyzer = ADSB1090Analyzer()
        self._pluto_source = pluto_source or PlutoLiveSource()
        self._owns_pluto_source = bool(owns_pluto_source)
        self._capture_thread: _ADSBPlutoCaptureThread | None = None
        self._packet_history: list[_ADSBPacketEntry] = []
        self._continuous_scan = False
        self._scan_started_wall_time: datetime | None = None
        self._stream_sample_rate_hz: float | None = None
        self._stream_total_samples = 0
        self._stream_ring_start_sample = 0
        self._stream_ring_iq = np.empty(0, dtype=np.complex64)
        self._stream_tail_start_sample = 0
        self._stream_tail_iq = np.empty(0, dtype=np.complex64)
        self._last_reported_start_sample = -1
        self._single_trigger_sample: int | None = None
        self._single_messages: list[tuple[int, ADSB1090Message]] = []
        self._single_complete = False
        self._plot_initial_ranges: dict[
            str, tuple[list[float], list[float]]
        ] = {}
        self._plot_context_actions: dict[str, dict[str, QtGui.QAction]] = {}
        self._closing = False
        self._packet_selection_connected = False
        self._build_menu()
        self._build_ui()
        if recording is not None:
            self.analyze_recording(recording)

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        open_action = file_menu.addAction("Open IQ...")
        open_action.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._open_iq)
        file_menu.addSeparator()
        close_action = file_menu.addAction("Close")
        close_action.triggered.connect(self.application_close_requested.emit)
        run_menu = self.menuBar().addMenu("Sweep / Run")
        self.run_single_action = run_menu.addAction("Run Single (Pluto)")
        self.run_single_action.setShortcut("F6")
        self.run_single_action.triggered.connect(self._run_pluto_single)
        self.run_continuous_action = run_menu.addAction("Run Continuous (Pluto)")
        self.run_continuous_action.setShortcut("F7")
        self.run_continuous_action.triggered.connect(self._run_pluto_continuous)
        refresh = run_menu.addAction("Refresh Analysis")
        refresh.setShortcut("F5")
        refresh.triggered.connect(self._refresh)
        mode_menu = self.menuBar().addMenu("Analysis Mode")
        generic_action = mode_menu.addAction("Generic FSK / PSK VSA")
        generic_action.triggered.connect(
            lambda: self.analysis_mode_requested.emit("generic")
        )
        mode_menu.addSeparator()
        adsb_action = mode_menu.addAction("ADS-B 1090ES")
        adsb_action.setCheckable(True)
        adsb_action.setChecked(True)
        adsb_action.setEnabled(False)

    def _dock(self, title: str, widget: QtWidgets.QWidget) -> QtWidgets.QDockWidget:
        return make_measurement_dock(
            title,
            widget,
            self,
            object_prefix="adsb",
            closable=False,
        )

    def _build_ui(self) -> None:
        self.setCentralWidget(QtWidgets.QWidget())
        toolbar = QtWidgets.QToolBar("1090ES Capture", self)
        toolbar.setMovable(False)
        toolbar.addWidget(QtWidgets.QLabel("Center: 1090 MHz   Fs:"))
        self.sample_rate_combo = QtWidgets.QComboBox()
        self.sample_rate_combo.addItem("8 MS/s", 8)
        self.sample_rate_combo.addItem("16 MS/s", 16)
        toolbar.addWidget(self.sample_rate_combo)
        toolbar.addWidget(QtWidgets.QLabel("   Capture:"))
        self.capture_length_spin = QtWidgets.QDoubleSpinBox()
        self.capture_length_spin.setRange(1.0, 2000.0)
        self.capture_length_spin.setValue(250.0)
        self.capture_length_spin.setSuffix(" ms")
        toolbar.addWidget(self.capture_length_spin)
        toolbar.addWidget(QtWidgets.QLabel("   Internal Gain:"))
        self.internal_gain_spin = QtWidgets.QDoubleSpinBox()
        self.internal_gain_spin.setRange(0.0, 70.0)
        self.internal_gain_spin.setValue(50.0)
        self.internal_gain_spin.setSuffix(" dB")
        toolbar.addWidget(self.internal_gain_spin)
        self.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, toolbar)
        self.power_plot = make_measurement_plot(
            "IQ Power (dBm)", "Measurement Elapsed Time (ms)"
        )
        self.ppm_plot = make_measurement_plot(
            "First / Second Chip Power (dB)", "Data Bit Index"
        )
        self.packet_table = QtWidgets.QTableWidget(0, 11)
        self.packet_table.setHorizontalHeaderLabels(
            [
                "#",
                "Elapsed (s)",
                "OS Time",
                "Capture (ms)",
                "DF",
                "ICAO",
                "TC",
                "Parity / CRC",
                "ON Power (dBm)",
                "SNR (dB)",
                "Decoded",
            ]
        )
        self.packet_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.packet_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.packet_table.verticalHeader().setVisible(False)
        packet_header = self.packet_table.horizontalHeader()
        for column in range(10):
            packet_header.setSectionResizeMode(
                column,
                QtWidgets.QHeaderView.ResizeMode.ResizeToContents,
            )
        packet_header.setSectionResizeMode(
            10,
            QtWidgets.QHeaderView.ResizeMode.Stretch,
        )
        self.packet_table.itemSelectionChanged.connect(self._selected_packet_changed)
        self._packet_selection_connected = True
        self.summary_table = QtWidgets.QTableWidget(0, 2)
        self.summary_table.setHorizontalHeaderLabels(["Parameter", "Current"])
        summary_header = self.summary_table.horizontalHeader()
        summary_header.setSectionResizeMode(
            0,
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents,
        )
        summary_header.setSectionResizeMode(
            1,
            QtWidgets.QHeaderView.ResizeMode.Stretch,
        )
        self.summary_table.verticalHeader().setVisible(False)
        self.summary_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        # Keep Python references as well as Qt parentage.  PySide can otherwise
        # collect a locally-created QDockWidget and delete its child ViewBox while
        # the PlotWidget wrapper is still reachable from this window.
        self.power_dock = self._dock("IQ Power", self.power_plot)
        self.ppm_dock = self._dock("PPM Demodulation", self.ppm_plot)
        self.packet_dock = self._dock("Packet List", self.packet_table)
        self.summary_dock = self._dock("Message Summary", self.summary_table)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.power_dock)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.packet_dock)
        self.splitDockWidget(
            self.power_dock,
            self.ppm_dock,
            QtCore.Qt.Orientation.Vertical,
        )
        self.splitDockWidget(
            self.packet_dock,
            self.summary_dock,
            QtCore.Qt.Orientation.Vertical,
        )
        self._configure_plot_context_menus()
        self.statusBar().showMessage("Ready - load 1090 MHz IQ or pass the current VSA capture")

    def _open_iq(self) -> None:
        settings = QtCore.QSettings()
        directory = str(settings.value("adsb1090/iq_directory", ""))
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open 1090 MHz IQ",
            directory,
            "IQ Recordings (*.iq.tar *.npz);;All Files (*)",
        )
        if not path:
            return
        try:
            recording = FileIQSource.load(path)
            self.analyze_recording(recording)
        except Exception as error:
            QtWidgets.QMessageBox.critical(self, "ADS-B 1090ES", str(error))
            return
        settings.setValue("adsb1090/iq_directory", str(Path(path).resolve().parent))

    def _refresh(self) -> None:
        if self.recording is not None:
            self.analyze_recording(self.recording)

    def _pluto_settings(self) -> PlutoCaptureSettings:
        sample_rate_msps = int(self.sample_rate_combo.currentData())
        return PlutoCaptureSettings(
            center_frequency_hz=1_090_000_000.0,
            symbol_rate_hz=1_000_000.0,
            samples_per_symbol=sample_rate_msps,
            capture_length_s=self.capture_length_spin.value() / 1e3,
            rf_bandwidth_hz=4_000_000.0,
            power_correction=InputPowerCorrection(
                internal_gain_db=self.internal_gain_spin.value(),
                external_attenuation_db=0.0,
                external_gain_db=0.0,
            ),
        )

    def _run_pluto_single(self) -> None:
        self._run_pluto(continuous=False)

    def _run_pluto_continuous(self) -> None:
        self._run_pluto(continuous=True)

    def _run_pluto(self, *, continuous: bool) -> None:
        if self._capture_thread is not None and self._capture_thread.isRunning():
            self._capture_thread.requestInterruption()
            self.run_single_action.setEnabled(False)
            self.run_continuous_action.setEnabled(False)
            self.statusBar().showMessage("Stopping Pluto capture...")
            return
        try:
            settings = self._pluto_settings()
        except ValueError as error:
            QtWidgets.QMessageBox.critical(self, "Pluto Capture", str(error))
            return
        self._continuous_scan = bool(continuous)
        self._scan_started_wall_time = datetime.now().astimezone()
        self._clear_packet_history()
        self._reset_stream_state(settings.requested_sample_rate_hz)
        if continuous:
            self.run_continuous_action.setText("Stop Continuous")
            self.run_single_action.setEnabled(False)
        else:
            self.run_single_action.setText("Stop Single")
            self.run_continuous_action.setEnabled(False)
        self.statusBar().showMessage(
            f"{'Continuously scanning' if continuous else 'Capturing'} 1090 MHz - "
            f"{settings.requested_sample_rate_hz / 1e6:.0f} MS/s, "
            f"{settings.capture_length_s * 1e3:.0f} ms"
        )
        thread = _ADSBPlutoCaptureThread(
            self._pluto_source,
            settings,
            parent=self,
        )
        thread.capture_ready.connect(self._pluto_capture_ready)
        thread.capture_failed.connect(self._pluto_capture_failed)
        thread.capture_cancelled.connect(
            lambda: self.statusBar().showMessage("Pluto capture cancelled")
        )
        thread.finished.connect(self._pluto_capture_stopped)
        thread.finished.connect(thread.deleteLater)
        self._capture_thread = thread
        thread.start()

    def _reset_stream_state(self, sample_rate_hz: float) -> None:
        self._stream_sample_rate_hz = float(sample_rate_hz)
        self._stream_total_samples = 0
        self._stream_ring_start_sample = 0
        self._stream_ring_iq = np.empty(0, dtype=np.complex64)
        self._stream_tail_start_sample = 0
        self._stream_tail_iq = np.empty(0, dtype=np.complex64)
        self._last_reported_start_sample = -1
        self._single_trigger_sample = None
        self._single_messages.clear()
        self._single_complete = False

    def _pluto_capture_ready(self, payload: object) -> None:
        if not isinstance(payload, _ADSBCaptureBatch):
            self._pluto_capture_failed("capture returned an invalid IQ recording")
            return
        self._process_stream_block(payload.recording)

    def _process_stream_block(self, recording: IQRecording) -> None:
        if self._single_complete:
            return
        sample_rate_hz = float(recording.sample_rate_hz)
        if (
            self._stream_sample_rate_hz is None
            or not np.isclose(self._stream_sample_rate_hz, sample_rate_hz)
        ):
            self._reset_stream_state(sample_rate_hz)
        block_iq = np.asarray(recording.iq, dtype=np.complex64)
        block_start = self._stream_total_samples
        self._stream_total_samples += block_iq.size
        self._append_stream_ring(block_iq)

        if self._stream_tail_iq.size:
            analysis_iq = np.concatenate((self._stream_tail_iq, block_iq))
            analysis_start = self._stream_tail_start_sample
        else:
            analysis_iq = block_iq
            analysis_start = block_start
        analysis_recording = replace(
            recording,
            iq=analysis_iq,
            start_sample_index=analysis_start,
            trigger_sample_index=None,
            source="VSA Pluto ADS-B Stream",
        )
        try:
            analysis_result = self._analyzer.analyze(analysis_recording)
        except Exception as error:
            self._pluto_capture_failed(str(error))
            if self._capture_thread is not None:
                self._capture_thread.requestInterruption()
            return

        new_messages: list[tuple[int, ADSB1090Message]] = []
        for message in analysis_result.messages:
            absolute_start = analysis_start + message.start_sample
            if absolute_start <= self._last_reported_start_sample:
                continue
            new_messages.append((absolute_start, message))
        if new_messages:
            self._last_reported_start_sample = max(
                absolute_start for absolute_start, _message in new_messages
            )
            if self._continuous_scan:
                display_samples = self._pluto_settings().capture_samples
                view_start = max(
                    self._stream_ring_start_sample,
                    self._stream_total_samples - display_samples,
                )
                self._display_stream_view(
                    recording,
                    view_start,
                    self._stream_total_samples,
                    new_messages,
                    append=True,
                )
            else:
                self._single_messages.extend(new_messages)
                if self._single_trigger_sample is None:
                    self._single_trigger_sample = new_messages[0][0]

        if not self._continuous_scan and self._single_trigger_sample is not None:
            target_stop = (
                self._single_trigger_sample + self._pluto_settings().capture_samples
            )
            if self._stream_total_samples >= target_stop:
                pretrigger = int(round(_SINGLE_PRETRIGGER_S * sample_rate_hz))
                view_start = max(
                    self._stream_ring_start_sample,
                    self._single_trigger_sample - pretrigger,
                )
                self._display_stream_view(
                    recording,
                    view_start,
                    target_stop,
                    self._single_messages,
                    append=False,
                )
                self._single_complete = True
                if self._capture_thread is not None:
                    self._capture_thread.requestInterruption()

        overlap_samples = max(1, int(round(_STREAM_OVERLAP_S * sample_rate_hz)))
        keep = min(overlap_samples, analysis_iq.size)
        self._stream_tail_iq = analysis_iq[-keep:].copy()
        self._stream_tail_start_sample = self._stream_total_samples - keep

    def _append_stream_ring(self, block_iq: np.ndarray) -> None:
        if self._stream_ring_iq.size:
            self._stream_ring_iq = np.concatenate((self._stream_ring_iq, block_iq))
        else:
            self._stream_ring_iq = block_iq.copy()
        settings = self._pluto_settings()
        pretrigger = int(round(_SINGLE_PRETRIGGER_S * settings.requested_sample_rate_hz))
        stream_block = int(
            round(_STREAM_BLOCK_DURATION_S * settings.requested_sample_rate_hz)
        )
        maximum = settings.capture_samples + pretrigger + 2 * stream_block
        excess = self._stream_ring_iq.size - maximum
        if excess > 0:
            self._stream_ring_iq = self._stream_ring_iq[excess:].copy()
            self._stream_ring_start_sample += excess

    def _display_stream_view(
        self,
        template: IQRecording,
        start_sample: int,
        stop_sample: int,
        messages: list[tuple[int, ADSB1090Message]],
        *,
        append: bool,
    ) -> None:
        start_sample = max(start_sample, self._stream_ring_start_sample)
        stop_sample = min(stop_sample, self._stream_total_samples)
        lo = start_sample - self._stream_ring_start_sample
        hi = stop_sample - self._stream_ring_start_sample
        if hi <= lo:
            return
        view_recording = replace(
            template,
            iq=self._stream_ring_iq[lo:hi],
            start_sample_index=start_sample,
            trigger_sample_index=None,
            source="VSA Pluto ADS-B Stream",
        )
        relative_messages = tuple(
            replace(message, start_sample=absolute_start - start_sample)
            for absolute_start, message in messages
            if start_sample <= absolute_start < stop_sample
        )
        full_scale = float(view_recording.full_scale)
        linear_power = (np.abs(view_recording.iq) / full_scale) ** 2
        power_dbfs = 10.0 * np.log10(
            np.maximum(linear_power, np.finfo(np.float64).tiny)
        )
        result = ADSB1090AnalysisResult(
            time_s=np.arange(view_recording.sample_count, dtype=np.float64)
            / view_recording.sample_rate_hz,
            power_dbfs=power_dbfs,
            messages=relative_messages,
            metadata={
                "source": view_recording.source,
                "stream_start_sample": start_sample,
            },
        )
        self.recording = view_recording
        self.result = result
        scan_wall = self._scan_started_wall_time or datetime.now().astimezone()
        elapsed_base_s = start_sample / view_recording.sample_rate_hz
        self._display_result(
            result,
            view_recording,
            append=append,
            capture_started_at=scan_wall + timedelta(seconds=elapsed_base_s),
            elapsed_base_s=elapsed_base_s,
            fit_latest_group=False,
        )
        valid = sum(message.parity_ok is True for message in relative_messages)
        self.statusBar().showMessage(
            f"{'Continuous scan' if append else 'Single complete'} - "
            f"{len(relative_messages)} new messages, {valid} parity verified, "
            f"{len(self._packet_history)} total"
        )

    def _pluto_capture_failed(self, message: str) -> None:
        self.statusBar().showMessage(f"Pluto capture failed: {message}")
        QtWidgets.QMessageBox.critical(self, "Pluto Capture", message)

    def _pluto_capture_stopped(self) -> None:
        self._capture_thread = None
        self.run_single_action.setText("Run Single (Pluto)")
        self.run_continuous_action.setText("Run Continuous (Pluto)")
        self.run_single_action.setEnabled(True)
        self.run_continuous_action.setEnabled(True)
        self._continuous_scan = False

    def analyze_recording(
        self,
        recording: IQRecording,
        *,
        append: bool = False,
        capture_started_at: datetime | None = None,
        elapsed_base_s: float = 0.0,
    ) -> None:
        self.statusBar().showMessage("Analyzing 1090 MHz capture...")
        QtWidgets.QApplication.processEvents()
        try:
            result = self._analyzer.analyze(recording)
        except Exception as error:
            self.statusBar().showMessage(f"Analysis failed: {error}")
            QtWidgets.QMessageBox.critical(self, "ADS-B 1090ES", str(error))
            return
        self.recording = recording
        self.result = result
        started_at = capture_started_at or datetime.now().astimezone()
        self._display_result(
            result,
            recording,
            append=append,
            capture_started_at=started_at,
            elapsed_base_s=float(elapsed_base_s),
        )
        valid = sum(message.parity_ok is True for message in result.messages)
        self.statusBar().showMessage(
            f"{'Continuous scan' if append else 'Analysis complete'} - "
            f"{len(result.messages)} new messages, {valid} parity verified, "
            f"{len(self._packet_history)} total"
        )

    def _clear_packet_history(self) -> None:
        self._packet_history.clear()
        blocker = QtCore.QSignalBlocker(self.packet_table)
        self.packet_table.clearSelection()
        self.packet_table.setRowCount(0)
        del blocker
        self.power_plot.clear()
        self.ppm_plot.clear()
        self._plot_initial_ranges.clear()
        self._show_summary(None)

    def _display_result(
        self,
        result: ADSB1090AnalysisResult,
        recording: IQRecording,
        *,
        append: bool,
        capture_started_at: datetime,
        elapsed_base_s: float,
        fit_latest_group: bool = True,
    ) -> None:
        if not append:
            self._clear_packet_history()
        self.power_plot.clear()
        time_ms = (elapsed_base_s + result.time_s) * 1e3
        display_power = np.maximum(
            result.power_dbfs + recording.dbfs_to_dbm_offset_db,
            _IQ_POWER_DISPLAY_FLOOR_DBM,
        )
        self.power_plot.plot(time_ms, display_power, pen=_TRACE_COLOR)
        new_entries: list[_ADSBPacketEntry] = []
        for message in result.messages:
            elapsed_s = elapsed_base_s + message.start_time_s
            entry = _ADSBPacketEntry(
                message=message,
                result=result,
                recording=recording,
                elapsed_s=elapsed_s,
                wall_time=capture_started_at + timedelta(seconds=message.start_time_s),
                on_pulse_power_dbm=self._on_pulse_power_dbm(
                    result,
                    recording,
                    message,
                ),
            )
            new_entries.append(entry)
            line = pg.InfiniteLine(
                pos=elapsed_s * 1e3,
                angle=90,
                pen=pg.mkPen(_BIT_COLOR, width=1),
            )
            self.power_plot.addItem(line)
        first_new_row = len(self._packet_history)
        self._packet_history.extend(new_entries)
        selection_blocker = QtCore.QSignalBlocker(self.packet_table)
        if not append or new_entries:
            self.packet_table.clearSelection()
        self.packet_table.setRowCount(len(self._packet_history))
        for offset, entry in enumerate(new_entries):
            row = first_new_row + offset
            message = entry.message
            decoded = message.fields.get("callsign")
            if not decoded and message.fields.get("altitude_ft") is not None:
                decoded = f"{message.fields['altitude_ft']} ft"
            values = (
                str(row + 1),
                f"{entry.elapsed_s:.6f}",
                entry.wall_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
                f"{message.start_time_s * 1e3:.6f}",
                str(message.downlink_format),
                message.icao_address or "-",
                "-" if message.type_code is None else str(message.type_code),
                message.parity_display,
                f"{entry.on_pulse_power_dbm:+.2f}",
                f"{message.preamble_snr_db:.1f}",
                str(decoded or ""),
            )
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                self.packet_table.setItem(row, column, item)
        selected_entry = new_entries[-1] if append and new_entries else None
        selected_row = len(self._packet_history) - 1 if selected_entry is not None else -1
        if not append and new_entries:
            selected_entry = new_entries[0]
            selected_row = 0
        if selected_row >= 0:
            self.packet_table.selectRow(selected_row)
        del selection_blocker
        if fit_latest_group:
            self._set_latest_group_power_range(result, elapsed_base_s)
        elif result.time_s.size:
            self.power_plot.setXRange(
                elapsed_base_s * 1e3,
                (elapsed_base_s + float(result.time_s[-1])) * 1e3,
                padding=0.0,
            )
        self._remember_plot_range("power", self.power_plot)
        if selected_entry is not None:
            self._show_message_plot(selected_entry)
            self._show_summary(selected_entry)
        elif not self._packet_history:
            self._show_summary(None)

    def _selected_packet_changed(self) -> None:
        if self._closing:
            return
        row = self.packet_table.currentRow()
        if not 0 <= row < len(self._packet_history):
            return
        entry = self._packet_history[row]
        self._show_message_plot(entry)
        self._show_summary(entry)

    def _show_message_plot(self, entry: _ADSBPacketEntry) -> None:
        message = entry.message
        result = entry.result
        self.ppm_plot.clear()
        linear_power = np.power(10.0, result.power_dbfs / 10.0)
        samples_per_us = message.sample_rate_hz * 1e-6
        data_start = message.start_sample + 8.0 * samples_per_us
        first_chip = np.empty(message.bit_length, dtype=np.float64)
        second_chip = np.empty(message.bit_length, dtype=np.float64)
        for bit_index in range(message.bit_length):
            symbol_start = data_start + bit_index * samples_per_us
            midpoint = symbol_start + 0.5 * samples_per_us
            symbol_stop = symbol_start + samples_per_us
            first_chip[bit_index] = self._fractional_window_mean(
                linear_power, symbol_start, midpoint
            )
            second_chip[bit_index] = self._fractional_window_mean(
                linear_power, midpoint, symbol_stop
            )
        epsilon = np.finfo(np.float64).tiny
        chip_ratio_db = 10.0 * np.log10(
            np.maximum(first_chip, epsilon) / np.maximum(second_chip, epsilon)
        )
        bit_index = np.arange(message.bit_length, dtype=np.float64)
        self.ppm_plot.addItem(
            pg.InfiniteLine(pos=0.0, angle=0, pen=pg.mkPen((160, 160, 160), width=1))
        )
        self.ppm_plot.plot(
            bit_index,
            chip_ratio_db,
            pen=pg.mkPen(_TRACE_COLOR, width=1),
        )
        one = message.bits == 1
        zero = ~one
        self.ppm_plot.plot(
            bit_index[one],
            chip_ratio_db[one],
            pen=None,
            symbol="o",
            symbolSize=6,
            symbolPen=None,
            symbolBrush=_BIT_COLOR,
        )
        self.ppm_plot.plot(
            bit_index[zero],
            chip_ratio_db[zero],
            pen=None,
            symbol="o",
            symbolSize=6,
            symbolPen=None,
            symbolBrush=_ZERO_BIT_COLOR,
        )
        self.ppm_plot.setXRange(-1.0, float(message.bit_length), padding=0.0)
        limit = max(3.0, 1.1 * float(np.max(np.abs(chip_ratio_db))))
        self.ppm_plot.setYRange(-limit, limit, padding=0.0)
        self._remember_plot_range("ppm", self.ppm_plot)

    @staticmethod
    def _fractional_window_mean(
        values: np.ndarray, start: float, stop: float
    ) -> float:
        lo = max(0, int(np.floor(start)))
        hi = min(values.size, max(lo + 1, int(np.ceil(stop))))
        return float(np.mean(values[lo:hi]))

    def _show_summary(self, entry: _ADSBPacketEntry | None) -> None:
        if entry is None:
            rows: list[tuple[str, object]] = []
        else:
            message = entry.message
            rows = [
                ("Measurement Elapsed", f"{entry.elapsed_s:.6f} s"),
                ("OS Time", entry.wall_time.strftime("%Y-%m-%d %H:%M:%S.%f")),
                ("Raw Message", message.raw_hex),
                ("Length", f"{message.bit_length} bit"),
                ("Downlink Format", message.downlink_format),
            ]
            rows.extend(
                (
                    key.replace("_", " ").title(),
                    self._format_mode_s_header_field(key, message.fields[key]),
                )
                for key in _MODE_S_HEADER_FIELDS
                if key in message.fields
            )
            rows.extend([
                ("ICAO Address", message.icao_address or "-"),
                (
                    "ICAO Address Source",
                    (message.icao_address_source or "-").replace("_", " ").title(),
                ),
                ("Type Code", message.type_code if message.type_code is not None else "-"),
                ("Parity / CRC", message.parity_display),
                ("Mean ON Pulse Power", f"{entry.on_pulse_power_dbm:+.2f} dBm"),
                ("Preamble SNR", f"{message.preamble_snr_db:.2f} dB"),
                ("Preamble Correlation", f"{message.preamble_correlation:.3f}"),
            ])
            fixed_fields = {"type_code", "icao_address", *_MODE_S_HEADER_FIELDS}
            rows.extend(
                (key.replace("_", " ").title(), value)
                for key, value in message.fields.items()
                if key not in fixed_fields
            )
        self.summary_table.setRowCount(len(rows))
        for row, (name, value) in enumerate(rows):
            self.summary_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(name)))
            self.summary_table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(value)))

    @staticmethod
    def _format_mode_s_header_field(key: str, value: object) -> object:
        if key != "flight_status":
            return value
        numeric = int(value)
        description = _FLIGHT_STATUS_DESCRIPTIONS.get(numeric, "Unknown")
        return f"{numeric} ({description})"

    def _on_pulse_power_dbm(
        self,
        result: ADSB1090AnalysisResult,
        recording: IQRecording,
        message: ADSB1090Message,
    ) -> float:
        linear_power = np.power(10.0, result.power_dbfs / 10.0)
        samples_per_us = message.sample_rate_hz * 1e-6
        data_start = message.start_sample + 8.0 * samples_per_us
        pulse_powers = np.empty(message.bit_length, dtype=np.float64)
        for bit_index, bit in enumerate(message.bits):
            pulse_start = data_start + bit_index * samples_per_us
            if bit == 0:
                pulse_start += 0.5 * samples_per_us
            pulse_powers[bit_index] = self._fractional_window_mean(
                linear_power,
                pulse_start,
                pulse_start + 0.5 * samples_per_us,
            )
        mean_dbfs = 10.0 * np.log10(
            max(float(np.mean(pulse_powers)), np.finfo(np.float64).tiny)
        )
        return mean_dbfs + recording.dbfs_to_dbm_offset_db

    def _set_latest_group_power_range(
        self,
        result: ADSB1090AnalysisResult,
        elapsed_base_s: float,
    ) -> None:
        if result.time_s.size == 0:
            return
        capture_start_s = elapsed_base_s + float(result.time_s[0])
        capture_stop_s = elapsed_base_s + float(result.time_s[-1])
        messages = result.messages
        if not messages:
            lower_s, upper_s = capture_start_s, capture_stop_s
        else:
            group_start = len(messages) - 1
            while group_start > 0:
                gap = (
                    messages[group_start].start_time_s
                    - messages[group_start - 1].start_time_s
                )
                if gap > _PACKET_GROUP_GAP_S:
                    break
                group_start -= 1
            first_s = elapsed_base_s + messages[group_start].start_time_s
            last = messages[-1]
            last_s = (
                elapsed_base_s
                + last.start_time_s
                + (8.0 + last.bit_length) * 1e-6
            )
            span_s = max(last_s - first_s, 1e-3)
            margin_s = max(1e-3, 0.1 * span_s)
            lower_s = max(capture_start_s, first_s - margin_s)
            upper_s = min(capture_stop_s, last_s + margin_s)
        if upper_s <= lower_s:
            upper_s = lower_s + 1e-3
        self.power_plot.setXRange(lower_s * 1e3, upper_s * 1e3, padding=0.0)

    def _configure_plot_context_menus(self) -> None:
        for name, plot in (("power", self.power_plot), ("ppm", self.ppm_plot)):
            self._plot_context_actions[name] = install_measurement_plot_menu(
                plot,
                reset=lambda plot_name=name, target=plot: self._reset_plot(
                    plot_name, target
                ),
            )

    def _remember_plot_range(self, name: str, plot: pg.PlotWidget) -> None:
        plot.getViewBox().updateAutoRange()
        x_range, y_range = plot.viewRange()
        self._plot_initial_ranges[name] = (list(x_range), list(y_range))

    def _reset_plot(self, name: str, plot: pg.PlotWidget) -> None:
        ranges = self._plot_initial_ranges.get(name)
        if ranges is None:
            return
        x_range, y_range = ranges
        plot.setRange(xRange=x_range, yRange=y_range, padding=0.0)

    def prepare_for_shutdown(self) -> None:
        """Stop UI callbacks before Qt starts deleting child dock widgets."""

        if self._closing:
            return
        self._closing = True
        if self._packet_selection_connected:
            try:
                self.packet_table.itemSelectionChanged.disconnect(
                    self._selected_packet_changed
                )
            except (RuntimeError, TypeError):
                pass
            self._packet_selection_connected = False
        try:
            self.packet_table.blockSignals(True)
        except RuntimeError:
            pass

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.prepare_for_shutdown()
        if self._capture_thread is not None and self._capture_thread.isRunning():
            self._capture_thread.requestInterruption()
            self._capture_thread.wait(1500)
        if self._owns_pluto_source:
            self._pluto_source.close()
        super().closeEvent(event)
