"""R&S-inspired multi-window shell for the first offline VSA milestone."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from scipy.ndimage import gaussian_filter

from pluto_sa.config.input_frontend import InputPowerCorrection
from pluto_sa.sdr.trigger import TriggerKind, TriggerSlope
from pluto_sa.vsa.mapping import (
    BLUETOOTH_EDR_MAPPING,
    GRAY_MAPPING,
    NATURAL_MAPPING,
    psk_constellation,
)
from pluto_sa.vsa.model import IQRecording, ModulationFamily, ModulationKind, SignalDescription
from pluto_sa.vsa.pattern import (
    BitOrdering,
    DemodulationSettings,
    IQPowerTriggerSettings,
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
from pluto_sa.vsa.result_summary import (
    DEFAULT_RESULT_SUMMARY_IDS,
    RESULT_SUMMARY_BY_ID,
    RESULT_SUMMARY_ITEMS,
    ResultSummaryCategory,
    normalize_result_summary_ids,
)
from pluto_sa.vsa.session import VSASession
from pluto_sa.vsa.pluto_source import (
    CaptureCancelledError,
    PlutoCaptureSettings,
    PlutoLiveSource,
)
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
_MAX_DISPLAY_POINTS = 30_000
_MAX_TRACE_SYMBOL_POINTS = 2_000
_MAX_IQ_TRAJECTORY_POINTS = 10_000
_MAX_SYMBOL_TABLE_DISPLAY_SYMBOLS = 1_000
_STARTUP_CONFIG_KEY = "startup/measurement_config"
_STARTUP_CONFIG_SCHEMA = "pluto-vsa-startup-config"
_STARTUP_CONFIG_VERSION = 1
_SYMBOL_TABLE_EXPORT_SCHEMA = "pluto-vsa-symbol-table"
_SYMBOL_TABLE_EXPORT_VERSION = 1
_TRACE_COLOR = "y"
_IQ_PLANE_LIMIT = 1.25
_TRACE_SYMBOL_SIZE = 5.5
_SYMBOL_PLOT_FLAT_SIZE = 6.0
_SELECTED_MARKER_SIZE = 18.0
_SELECTED_MARKER_COLOR = (0, 255, 255)
_CONSTELLATION_DENSITY_BINS = 96
_CONSTELLATION_DENSITY_SIGMA_BINS = 0.7
_CONSTELLATION_DENSITY_RED_LEVEL = 0.75


class _CenteredLabelAxisItem(pg.AxisItem):
    """Keep horizontal and rotated vertical labels visually centered."""

    def resizeEvent(self, event=None) -> None:
        super().resizeEvent(event)
        if (
            not hasattr(self, "_linkedView")
            or self.label is None
        ):
            return
        label_bounds = self.label.mapRectToParent(self.label.boundingRect())
        axis_center = QtCore.QPointF(
            self.size().width() / 2.0,
            self.size().height() / 2.0,
        )
        if self.orientation in {"left", "right"}:
            self.label.setY(
                self.label.y() + axis_center.y() - label_bounds.center().y()
            )
        else:
            self.label.setX(
                self.label.x() + axis_center.x() - label_bounds.center().x()
            )


class _FixedInteractionViewBox(pg.ViewBox):
    """Left-drag rectangle zoom with middle-drag pan, without mode switching."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        super().setMouseMode(pg.ViewBox.RectMode)

    def setMouseMode(self, _mode: int) -> None:
        """Keep left-button interaction fixed to rectangular zoom."""
        super().setMouseMode(pg.ViewBox.RectMode)

    def mouseDragEvent(self, event: object, axis: int | None = None) -> None:
        if event.button() != QtCore.Qt.MouseButton.MiddleButton:
            super().mouseDragEvent(event, axis=axis)
            return
        # pyqtgraph implements middle-button panning in its three-button
        # PanMode. Use that path for this event only, while keeping left drag
        # permanently assigned to RectMode.
        self.state["mouseMode"] = pg.ViewBox.PanMode
        try:
            super().mouseDragEvent(event, axis=axis)
        finally:
            self.state["mouseMode"] = pg.ViewBox.RectMode


def _decimation_indices(count: int, maximum: int = _MAX_DISPLAY_POINTS) -> slice:
    step = max(1, int(np.ceil(int(count) / int(maximum))))
    return slice(None, None, step)


def _peak_decimate_xy(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    maximum: int = _MAX_DISPLAY_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Bound plot data while retaining each time bucket's min/max excursion."""
    x = np.asarray(x_values)
    y = np.asarray(y_values)
    count = min(x.size, y.size)
    if count <= int(maximum):
        return x[:count], y[:count]
    bucket_count = max(1, int(maximum) // 2)
    step = max(1, int(np.ceil(count / bucket_count)))
    full_count = (count // step) * step
    grouped = y[:full_count].reshape(-1, step)
    group_offset = np.arange(grouped.shape[0], dtype=np.int64) * step
    minimum = group_offset + np.argmin(grouped, axis=1)
    maximum_index = group_offset + np.argmax(grouped, axis=1)
    paired = np.column_stack((minimum, maximum_index))
    paired.sort(axis=1)
    indices = paired.reshape(-1)
    if full_count < count:
        tail = y[full_count:count]
        tail_indices = np.asarray(
            [full_count + int(np.argmin(tail)), full_count + int(np.argmax(tail))],
            dtype=np.int64,
        )
        indices = np.concatenate((indices, np.sort(tail_indices)))
    return x[indices], y[indices]


def _prepare_psk_display_waveform(
    iq: np.ndarray,
    *,
    sample_rate_hz: float,
    symbol_rate_hz: float,
    tx_filter: str,
    filter_parameter: float | None,
    result_start_time_s: float | None = None,
    result_stop_time_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare only the visible PSK range, including receive-filter guards."""
    values = np.asarray(iq)
    start_sample = 0
    stop_sample = values.size
    if result_start_time_s is not None and result_stop_time_s is not None:
        # The SRRC filter spans ten symbols. Sixteen symbols on either side
        # also cover scipy's polyphase-resampler transient for normal VSA
        # sample-rate ratios, keeping the visible Result Range unchanged.
        guard_s = 16.0 / float(symbol_rate_hz)
        start_sample = max(
            0,
            int(np.floor((float(result_start_time_s) - guard_s) * sample_rate_hz)),
        )
        stop_sample = min(
            values.size,
            int(np.ceil((float(result_stop_time_s) + guard_s) * sample_rate_hz)),
        )
    prepared, prepared_rate_hz = prepare_psk_iq(
        values[start_sample:stop_sample],
        sample_rate_hz=sample_rate_hz,
        symbol_rate_hz=symbol_rate_hz,
        tx_filter=tx_filter,
        filter_parameter=filter_parameter,
    )
    time_offset_s = start_sample / float(sample_rate_hz)
    time_s = time_offset_s + np.arange(prepared.size, dtype=np.float64) / float(
        prepared_rate_hz
    )
    return prepared, time_s


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


def _physical_constellation_display_symbols(
    modulation: ModulationKind, symbols: np.ndarray
) -> np.ndarray:
    """Apply R&S physical-constellation derotation to absolute IQ symbols."""
    values = np.asarray(symbols, dtype=np.complex128)
    if modulation is ModulationKind.PI4_DQPSK:
        rotation = (np.arange(values.size, dtype=np.float64) + 1.0) * np.pi / 4.0
        return values * np.exp(-1j * rotation)
    if modulation in {ModulationKind.QPSK, ModulationKind.OQPSK}:
        return values * np.exp(-1j * np.pi / 4.0)
    return values


def _constellation_density(
    symbols: np.ndarray,
    *,
    limit: float = _IQ_PLANE_LIMIT,
    bins: int = _CONSTELLATION_DENSITY_BINS,
    smoothing_sigma_bins: float = _CONSTELLATION_DENSITY_SIGMA_BINS,
) -> np.ndarray:
    """Return a smoothed row-major log-density image over the fixed I/Q plane."""
    values = np.asarray(symbols, dtype=np.complex128)
    finite = np.isfinite(values.real) & np.isfinite(values.imag)
    values = values[finite]
    if values.size == 0:
        return np.zeros((int(bins), int(bins)), dtype=np.float64)
    histogram, _i_edges, _q_edges = np.histogram2d(
        values.real,
        values.imag,
        bins=int(bins),
        range=((-float(limit), float(limit)), (-float(limit), float(limit))),
    )
    # ImageItem row-major data is indexed [Q, I].  A small Gaussian kernel
    # turns each hard histogram cell into a continuous density contribution,
    # which better represents repeated observations than a grid of enlarged
    # dots.  The kernel is normalized, so relative occurrence remains intact.
    density = histogram.T
    sigma = max(0.0, float(smoothing_sigma_bins))
    if sigma > 0.0:
        density = gaussian_filter(
            density,
            sigma=sigma,
            mode="constant",
            cval=0.0,
            truncate=3.0,
        )
    # log1p keeps low-occurrence regions visible without allowing the most
    # common decision cell to hide all surrounding distribution detail.
    return np.log1p(density)


def _constellation_density_color_levels(
    density: np.ndarray,
    *,
    red_level: float = _CONSTELLATION_DENSITY_RED_LEVEL,
) -> tuple[float, float]:
    """Map the upper density region to saturated red without widening it."""
    values = np.asarray(density, dtype=np.float64)
    finite = values[np.isfinite(values)]
    peak = float(np.max(finite)) if finite.size else 0.0
    if peak <= 0.0:
        return 0.0, 1.0
    saturation = float(np.clip(red_level, 0.01, 1.0))
    return 0.0, peak * saturation


def _constellation_density_extent(
    symbols: np.ndarray,
    *,
    minimum: float = _IQ_PLANE_LIMIT,
) -> float:
    """Cover every finite I/Q point while retaining the nominal minimum view."""
    values = np.asarray(symbols, dtype=np.complex128)
    finite = np.isfinite(values.real) & np.isfinite(values.imag)
    values = values[finite]
    if values.size == 0:
        return max(float(minimum), np.finfo(np.float64).eps)
    component_peak = float(
        max(np.max(np.abs(values.real)), np.max(np.abs(values.imag)))
    )
    # Keep the outermost point away from the histogram boundary.  Apart from
    # making it visible, this avoids relying on the last-bin inclusive edge.
    return max(float(minimum), component_peak * 1.02, np.finfo(np.float64).eps)


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
        try:
            recording = self._source.capture_single(
                self._settings,
                cancelled=self.isInterruptionRequested,
            )
            if self.isInterruptionRequested():
                self.capture_cancelled.emit()
            else:
                self.capture_ready.emit(recording)
        except CaptureCancelledError:
            self.capture_cancelled.emit()
        except Exception as error:
            self.capture_failed.emit(str(error))

    def cancel(self) -> None:
        self.requestInterruption()


class _AnalysisThread(QtCore.QThread):
    """Run one immutable VSA session snapshot outside the GUI thread."""

    analysis_ready = QtCore.Signal(int, object)
    analysis_failed = QtCore.Signal(int, object, str)

    def __init__(
        self,
        generation: int,
        session: VSASession,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.generation = int(generation)
        self.session = session

    def run(self) -> None:
        try:
            self.session.analyze()
        except Exception as error:
            self.analysis_failed.emit(self.generation, self.session, str(error))
            return
        self.analysis_ready.emit(self.generation, self.session)


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
        self._analysis_thread: _AnalysisThread | None = None
        self._analysis_generation = 0
        self._pending_analysis: tuple[int, VSASession, dict[str, float]] | None = None
        self._active_analysis_context: dict[str, float] = {}
        self._updating_pattern_table = False
        self._pattern_values: list[int] = []
        self._analysis_plot_ranges: dict[str, tuple[list[float], list[float]]] = {}
        self._plot_context_actions: dict[str, dict[str, QtGui.QAction]] = {}
        self._selected_result_summary_ids = set(DEFAULT_RESULT_SUMMARY_IDS)
        self._result_summary_values: dict[str, str] = {}
        self._updating_result_summary_selection = False
        self._selected_match_index = 1
        self._selected_symbol_marker_index: int | None = None
        self._last_analysis_timings_ms: dict[str, float] = {}
        self._pluto_capture_started_at: float | None = None
        self._symbol_marker_items: dict[
            str, tuple[pg.PlotDataItem, pg.TextItem]
        ] = {}
        self._constellation_density_item: pg.ImageItem | None = None
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
        restored = self._restore_startup_meas_config()
        self._update_summary()
        if self.session.recording is None:
            self.statusBar().showMessage(
                "Ready - configuration restored; load or capture IQ"
                if restored
                else "Ready - load or capture an IQ recording"
            )

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        open_action = QtGui.QAction("Open IQ...", self)
        open_action.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._open_iq)
        file_menu.addAction(open_action)
        self.export_symbol_table_action = QtGui.QAction(
            "Export Symbol Table...", self
        )
        self.export_symbol_table_action.triggered.connect(
            self._export_symbol_table
        )
        self.export_symbol_table_action.setEnabled(False)
        file_menu.addAction(self.export_symbol_table_action)
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
        analyze_action.triggered.connect(self._request_analysis)
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
        self.symbol_display_action.setShortcut("S")
        self.symbol_display_action.setChecked(False)
        self.symbol_display_action.triggered.connect(self._refresh_display_only)
        display_menu.addAction(self.symbol_display_action)
        constellation_trace_menu = display_menu.addMenu("Symbol Plot Trace")
        self.constellation_flat_action = QtGui.QAction(
            "Flat", self, checkable=True
        )
        self.constellation_density_action = QtGui.QAction(
            "Density", self, checkable=True
        )
        constellation_trace_group = QtGui.QActionGroup(self)
        constellation_trace_group.setExclusive(True)
        constellation_trace_group.addAction(self.constellation_flat_action)
        constellation_trace_group.addAction(self.constellation_density_action)
        self.constellation_flat_action.setChecked(True)
        self.constellation_flat_action.triggered.connect(
            self._refresh_display_only
        )
        self.constellation_density_action.triggered.connect(
            self._refresh_display_only
        )
        constellation_trace_menu.addActions(constellation_trace_group.actions())
        psk_symbol_plot_menu = display_menu.addMenu("PSK Symbol Plot")
        self.physical_iq_symbol_plot_action = QtGui.QAction(
            "Absolute IQ (Physical)", self, checkable=True
        )
        self.differential_iq_symbol_plot_action = QtGui.QAction(
            "Differential IQ", self, checkable=True
        )
        psk_symbol_plot_group = QtGui.QActionGroup(self)
        psk_symbol_plot_group.setExclusive(True)
        psk_symbol_plot_group.addAction(self.physical_iq_symbol_plot_action)
        psk_symbol_plot_group.addAction(self.differential_iq_symbol_plot_action)
        self.physical_iq_symbol_plot_action.setChecked(True)
        self.physical_iq_symbol_plot_action.triggered.connect(
            self._refresh_display_only
        )
        self.differential_iq_symbol_plot_action.triggered.connect(
            self._refresh_display_only
        )
        psk_symbol_plot_menu.addActions(psk_symbol_plot_group.actions())
        self.reset_graph_scales_action = QtGui.QAction(
            "Reset Graph Scales", self
        )
        self.reset_graph_scales_action.setShortcut("Home")
        self.reset_graph_scales_action.triggered.connect(self._reset_graph_scales)
        display_menu.addAction(self.reset_graph_scales_action)
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
        # The surrounding dock already owns the visible title. Keeping a
        # second title inside the plot wastes vertical graph area.
        plot = pg.PlotWidget(
            viewBox=_FixedInteractionViewBox(),
            axisItems={
                "left": _CenteredLabelAxisItem(orientation="left"),
                "bottom": _CenteredLabelAxisItem(orientation="bottom"),
            }
        )
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
        content_font = QtGui.QFont(widget.font())
        content_point_size = content_font.pointSizeF()
        content_font.setBold(False)
        if content_point_size > 0.0:
            # Explicitly resolve the original point size so the enlarged dock
            # title font is not inherited by the dock contents.
            content_font.setPointSizeF(content_point_size)
        title_font = QtGui.QFont(dock.font())
        title_font.setBold(True)
        if title_font.pointSizeF() > 0.0:
            title_font.setPointSizeF(title_font.pointSizeF() * 1.3)
        elif title_font.pixelSize() > 0:
            title_font.setPixelSize(max(1, round(title_font.pixelSize() * 1.3)))
        dock.setFont(title_font)
        dock.setWidget(widget)
        # QDockWidget's native Windows title renderer uses the dock font.
        # Keep that bold font from propagating into plots and result tables.
        widget.setFont(content_font)
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

        self.spectrum_plot = self._make_plot("Spectrum", "Magnitude (dBm)", "Relative Frequency (MHz)")
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
        self.result_summary.setContextMenuPolicy(
            QtCore.Qt.ContextMenuPolicy.CustomContextMenu
        )
        self.result_summary.customContextMenuRequested.connect(
            self._show_result_summary_context_menu
        )
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
        self.modulation_plot.setAspectLocked(True, ratio=1.0)
        self.modulation_plot.setXRange(
            -_IQ_PLANE_LIMIT, _IQ_PLANE_LIMIT, padding=0.0
        )
        self.modulation_plot.setYRange(
            -_IQ_PLANE_LIMIT, _IQ_PLANE_LIMIT, padding=0.0
        )
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
        self.symbol_plot.setXRange(
            -_IQ_PLANE_LIMIT, _IQ_PLANE_LIMIT, padding=0.0
        )
        self.symbol_plot.setYRange(
            -_IQ_PLANE_LIMIT, _IQ_PLANE_LIMIT, padding=0.0
        )
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
        self.symbol_table.setContextMenuPolicy(
            QtCore.Qt.ContextMenuPolicy.CustomContextMenu
        )
        self.symbol_table.customContextMenuRequested.connect(
            self._show_symbol_table_context_menu
        )
        self.symbol_table.cellClicked.connect(
            self._symbol_table_cell_clicked
        )
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
        self._configure_plot_context_menus()
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
        self.mapping_combo.addItems(
            (NATURAL_MAPPING, GRAY_MAPPING, BLUETOOTH_EDR_MAPPING)
        )
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
        config_pages.append(("Signal Capture", capture_page))

        trigger_page = QtWidgets.QWidget()
        trigger_form = QtWidgets.QFormLayout(trigger_page)
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
        self.acquisition_trigger_offset_spin.setToolTip(
            "R&S Trigger Offset: positive starts the record after the crossing; "
            "negative retains pretrigger samples."
        )
        self.acquisition_trigger_hysteresis_spin = QtWidgets.QDoubleSpinBox()
        self.acquisition_trigger_hysteresis_spin.setRange(0.0, 50.0)
        self.acquisition_trigger_hysteresis_spin.setDecimals(1)
        self.acquisition_trigger_hysteresis_spin.setValue(3.0)
        self.acquisition_trigger_hysteresis_spin.setSuffix(" dB")
        trigger_form.addRow(acquisition_heading)
        trigger_form.addRow("Trigger Source", self.acquisition_trigger_source_combo)
        trigger_form.addRow("Level", self.acquisition_trigger_level_spin)
        trigger_form.addRow("Slope", self.acquisition_trigger_slope_combo)
        trigger_form.addRow("Trigger Offset", self.acquisition_trigger_offset_spin)
        trigger_form.addRow("Hysteresis", self.acquisition_trigger_hysteresis_spin)

        burst_heading = QtWidgets.QLabel("Post-capture Burst Search")
        burst_heading.setStyleSheet("font-weight: bold;")
        trigger_form.addRow(burst_heading)
        self.iq_power_trigger_check = QtWidgets.QCheckBox("Burst Search On")
        self.iq_power_trigger_check.setToolTip(
            "Detect every rising power event in the current I/Q capture and "
            "run pattern search once inside each active interval."
        )
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
        self.iq_power_trigger_average_spin.setToolTip(
            "Moving average applied to linear I/Q envelope power before "
            "threshold comparison."
        )
        self.iq_power_trigger_dropout_spin = QtWidgets.QDoubleSpinBox()
        self.iq_power_trigger_dropout_spin.setRange(0.0, 1_000_000.0)
        self.iq_power_trigger_dropout_spin.setDecimals(2)
        self.iq_power_trigger_dropout_spin.setValue(8.0)
        self.iq_power_trigger_dropout_spin.setSuffix(" sym")
        self.iq_power_trigger_dropout_spin.setToolTip(
            "Power must remain below Level - Hysteresis for this duration "
            "before another trigger can be detected."
        )
        self.iq_power_trigger_holdoff_spin = QtWidgets.QDoubleSpinBox()
        self.iq_power_trigger_holdoff_spin.setRange(0.0, 1_000_000.0)
        self.iq_power_trigger_holdoff_spin.setDecimals(2)
        self.iq_power_trigger_holdoff_spin.setValue(0.0)
        self.iq_power_trigger_holdoff_spin.setSuffix(" sym")
        self.iq_power_trigger_offset_spin = QtWidgets.QDoubleSpinBox()
        self.iq_power_trigger_offset_spin.setRange(-1_000_000.0, 1_000_000.0)
        self.iq_power_trigger_offset_spin.setDecimals(3)
        self.iq_power_trigger_offset_spin.setValue(0.0)
        self.iq_power_trigger_offset_spin.setSuffix(" sym")
        self.iq_power_trigger_offset_spin.setToolTip(
            "Signed offset from each trigger event to the pattern-search start; "
            "positive values delay the search and negative values include pre-trigger data."
        )
        self.iq_power_trigger_limit_result_check = QtWidgets.QCheckBox(
            "Limit Result Range to Active Interval"
        )
        self.iq_power_trigger_limit_result_check.setChecked(True)
        self.iq_power_trigger_limit_result_check.setToolTip(
            "Keep only complete symbols ending before the hysteretic burst stop. "
            "Disable for OOK or signals whose valid data contains long power gaps."
        )
        trigger_form.addRow(self.iq_power_trigger_check)
        trigger_form.addRow("Level", self.iq_power_trigger_level_spin)
        trigger_form.addRow("Hysteresis", self.iq_power_trigger_hysteresis_spin)
        trigger_form.addRow("Envelope Average", self.iq_power_trigger_average_spin)
        trigger_form.addRow("Drop-Out Time", self.iq_power_trigger_dropout_spin)
        trigger_form.addRow("Holdoff", self.iq_power_trigger_holdoff_spin)
        trigger_form.addRow("Search Start Offset", self.iq_power_trigger_offset_spin)
        trigger_form.addRow(self.iq_power_trigger_limit_result_check)
        self.acquisition_trigger_source_combo.currentIndexChanged.connect(
            self._sync_acquisition_trigger_controls
        )
        self._sync_acquisition_trigger_controls()
        self._sync_capture_settings()
        config_pages.append(("Trigger", trigger_page))

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
        self.pattern_allow_inverted_fsk_check = QtWidgets.QCheckBox(
            "Allow Inverted Pattern Match (FSK only)"
        )
        self.pattern_allow_inverted_fsk_check.setChecked(False)
        self.pattern_allow_inverted_fsk_check.setToolTip(
            "Also search the bitwise complement of the configured binary FSK "
            "pattern. Decoded symbols remain in Natural mapping."
        )
        pattern_form.addRow(self.pattern_search_check)
        pattern_form.addRow("Name", self.pattern_name_edit)
        pattern_form.addRow("Symbol Format", self.pattern_format_combo)
        pattern_form.addRow("I/Q Correlation Threshold", self.pattern_threshold_spin)
        pattern_form.addRow(self.pattern_threshold_auto)
        pattern_form.addRow(self.pattern_meas_only_check)
        pattern_form.addRow(self.pattern_allow_inverted_fsk_check)
        pattern_layout.addLayout(pattern_form)
        pattern_layout.addWidget(QtWidgets.QLabel("Pattern Symbols"))
        pattern_layout.addWidget(self.pattern_symbol_table, 1)
        pattern_layout.addLayout(pattern_table_buttons)
        self.pattern_threshold_auto.toggled.connect(
            lambda checked: self.pattern_threshold_spin.setEnabled(not checked)
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

        summary_page = QtWidgets.QWidget()
        summary_layout = QtWidgets.QVBoxLayout(summary_page)
        summary_layout.addWidget(
            QtWidgets.QLabel(
                "Choose Result Summary rows. Planned R&S items remain visible "
                "but cannot be selected yet."
            )
        )
        self.result_summary_item_tree = QtWidgets.QTreeWidget()
        self.result_summary_item_tree.setColumnCount(2)
        self.result_summary_item_tree.setHeaderLabels(("Result", "Status"))
        self.result_summary_item_tree.header().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self.result_summary_item_tree.header().setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        self.result_summary_item_tree.itemChanged.connect(
            self._result_summary_tree_item_changed
        )
        summary_layout.addWidget(self.result_summary_item_tree, 1)
        summary_presets = QtWidgets.QHBoxLayout()
        for label, preset in (
            ("Show All", "all"),
            ("Measurement Only", "measurement"),
            ("Diagnostics Only", "diagnostics"),
            ("Restore Defaults", "defaults"),
        ):
            button = QtWidgets.QPushButton(label)
            button.clicked.connect(
                lambda _checked=False, selected=preset: (
                    self._apply_result_summary_preset(selected)
                )
            )
            summary_presets.addWidget(button)
        summary_layout.addLayout(summary_presets)
        self._populate_result_summary_item_tree()
        config_pages.append(("Result Summary", summary_page))

        run_page = QtWidgets.QWidget()
        run_layout = QtWidgets.QVBoxLayout(run_page)
        self.run_single_button = QtWidgets.QPushButton("Run Single (Pluto)")
        self.run_single_button.clicked.connect(self._run_pluto_single)
        run_layout.addWidget(self.run_single_button)
        refresh_button = QtWidgets.QPushButton("Refresh Analysis")
        refresh_button.clicked.connect(self._request_analysis)
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

    def _populate_result_summary_item_tree(self) -> None:
        self._updating_result_summary_selection = True
        try:
            self.result_summary_item_tree.clear()
            self._result_summary_tree_items: dict[str, QtWidgets.QTreeWidgetItem] = {}
            for category in ResultSummaryCategory:
                parent = QtWidgets.QTreeWidgetItem((category.value, ""))
                parent.setFirstColumnSpanned(True)
                category_font = parent.font(0)
                category_font.setBold(True)
                parent.setFont(0, category_font)
                self.result_summary_item_tree.addTopLevelItem(parent)
                for definition in RESULT_SUMMARY_ITEMS:
                    if definition.category is not category:
                        continue
                    status = "Available" if definition.implemented else "Not implemented"
                    child = QtWidgets.QTreeWidgetItem((definition.label, status))
                    child.setData(0, QtCore.Qt.ItemDataRole.UserRole, definition.item_id)
                    child.setToolTip(0, definition.description)
                    child.setToolTip(1, status)
                    flags = child.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable
                    if not definition.implemented:
                        flags &= ~QtCore.Qt.ItemFlag.ItemIsEnabled
                    child.setFlags(flags)
                    child.setCheckState(
                        0,
                        QtCore.Qt.CheckState.Checked
                        if definition.item_id in self._selected_result_summary_ids
                        else QtCore.Qt.CheckState.Unchecked,
                    )
                    parent.addChild(child)
                    self._result_summary_tree_items[definition.item_id] = child
                parent.setExpanded(True)
        finally:
            self._updating_result_summary_selection = False

    def _sync_result_summary_item_tree(self) -> None:
        if not hasattr(self, "_result_summary_tree_items"):
            return
        self._updating_result_summary_selection = True
        try:
            for item_id, child in self._result_summary_tree_items.items():
                child.setCheckState(
                    0,
                    QtCore.Qt.CheckState.Checked
                    if item_id in self._selected_result_summary_ids
                    else QtCore.Qt.CheckState.Unchecked,
                )
        finally:
            self._updating_result_summary_selection = False

    def _result_summary_tree_item_changed(
        self, item: QtWidgets.QTreeWidgetItem, column: int
    ) -> None:
        if self._updating_result_summary_selection or column != 0:
            return
        item_id = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if not item_id or item_id not in RESULT_SUMMARY_BY_ID:
            return
        self._set_result_summary_item_visible(
            str(item_id), item.checkState(0) == QtCore.Qt.CheckState.Checked
        )

    def _set_result_summary_item_visible(self, item_id: str, visible: bool) -> None:
        definition = RESULT_SUMMARY_BY_ID.get(item_id)
        if definition is None or not definition.implemented:
            return
        if visible:
            self._selected_result_summary_ids.add(item_id)
        else:
            self._selected_result_summary_ids.discard(item_id)
        self._sync_result_summary_item_tree()
        self._render_result_summary()

    def _apply_result_summary_preset(self, preset: str) -> None:
        if preset == "all":
            selected = {
                item.item_id for item in RESULT_SUMMARY_ITEMS if item.implemented
            }
        elif preset == "measurement":
            selected = {
                item.item_id
                for item in RESULT_SUMMARY_ITEMS
                if item.implemented
                and item.category is not ResultSummaryCategory.DIAGNOSTICS
            }
        elif preset == "diagnostics":
            selected = {
                item.item_id
                for item in RESULT_SUMMARY_ITEMS
                if item.implemented
                and item.category is ResultSummaryCategory.DIAGNOSTICS
            }
        elif preset == "defaults":
            selected = set(DEFAULT_RESULT_SUMMARY_IDS)
        else:
            raise ValueError(f"unsupported Result Summary preset: {preset}")
        self._selected_result_summary_ids = selected
        self._sync_result_summary_item_tree()
        self._render_result_summary()

    def _create_result_summary_context_menu(self) -> QtWidgets.QMenu:
        menu = QtWidgets.QMenu(self.result_summary)
        for category in ResultSummaryCategory:
            category_menu = menu.addMenu(category.value)
            for definition in RESULT_SUMMARY_ITEMS:
                if definition.category is not category:
                    continue
                label = definition.label
                if not definition.implemented:
                    label += " (Not implemented)"
                action = category_menu.addAction(label)
                action.setCheckable(True)
                action.setChecked(
                    definition.item_id in self._selected_result_summary_ids
                )
                action.setEnabled(definition.implemented)
                action.setToolTip(definition.description)
                action.toggled.connect(
                    lambda checked, item_id=definition.item_id: (
                        self._set_result_summary_item_visible(item_id, checked)
                    )
                )
        menu.addSeparator()
        for label, preset in (
            ("Show All", "all"),
            ("Measurement Results Only", "measurement"),
            ("Diagnostics Only", "diagnostics"),
            ("Restore Defaults", "defaults"),
        ):
            action = menu.addAction(label)
            action.triggered.connect(
                lambda _checked=False, selected=preset: (
                    self._apply_result_summary_preset(selected)
                )
            )
        return menu

    def _show_result_summary_context_menu(self, position: QtCore.QPoint) -> None:
        menu = self._create_result_summary_context_menu()
        menu.exec(self.result_summary.mapToGlobal(position))

    def _create_symbol_table_context_menu(self) -> QtWidgets.QMenu:
        menu = QtWidgets.QMenu(self.symbol_table)
        export_action = menu.addAction("Export Symbol Table...")
        export_action.setEnabled(self.session.result is not None)
        export_action.triggered.connect(self._export_symbol_table)
        return menu

    def _show_symbol_table_context_menu(self, position: QtCore.QPoint) -> None:
        menu = self._create_symbol_table_context_menu()
        menu.exec(self.symbol_table.viewport().mapToGlobal(position))

    def _symbol_table_cell_clicked(self, row: int, column: int) -> None:
        item = self.symbol_table.item(int(row), int(column))
        if item is None:
            return
        symbol_index = int(row) * self.symbol_table.columnCount() + int(column)
        self._selected_symbol_marker_index = (
            None
            if self._selected_symbol_marker_index == symbol_index
            else symbol_index
        )
        self._update_plots(reset_ranges=False)

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

    def _symbol_table_export_document(self) -> dict[str, object]:
        result = self.session.result
        signal = self.session.signal
        recording = self.session.recording
        if result is None or signal is None or recording is None:
            raise RuntimeError("no analyzed Symbol Table is available")
        pattern_result = self.session.pattern_result
        symbols = np.asarray(
            pattern_result.decoded_symbols
            if pattern_result is not None
            else result.decoded_symbols,
            dtype=np.int16,
        )
        symbol_times = np.asarray(
            pattern_result.symbol_time_s
            if pattern_result is not None
            else result.symbol_time_s,
            dtype=np.float64,
        )
        bit_width = int(round(np.log2(signal.modulation.order)))
        lsb_first = self.bit_order_combo.currentText() == BitOrdering.LSB.value
        matched_pattern_symbols: tuple[int, ...] = ()
        pattern_metadata: dict[str, object] | None = None
        if pattern_result is not None:
            matched_pattern_symbols = tuple(
                int(value)
                for value in pattern_result.metadata.get(
                    "matched_pattern_symbols",
                    self._parse_pattern_symbols(pattern_result.modulation.order),
                )
            )
            pattern_metadata = {
                "name": str(pattern_result.metadata.get("pattern_name", "")),
                "match_variant": str(
                    pattern_result.metadata.get("pattern_match_variant", "Normal")
                ),
                "configured_symbols": list(
                    self._parse_pattern_symbols(pattern_result.modulation.order)
                ),
                "matched_symbols": list(matched_pattern_symbols),
                "start_sample": int(pattern_result.pattern_start_sample),
                "start_time_s": float(pattern_result.pattern_start_time_s),
                "symbol_count": len(matched_pattern_symbols),
                "symbol_errors": int(pattern_result.pattern_symbol_errors),
                "correlation": float(pattern_result.correlation),
            }

        rows: list[list[object]] = []
        for index, symbol_value in enumerate(symbols):
            symbol = int(symbol_value)
            ordered_bits = [
                (symbol >> shift) & 1
                for shift in (
                    range(bit_width)
                    if lsb_first
                    else range(bit_width - 1, -1, -1)
                )
            ]
            time_s = (
                float(symbol_times[index])
                if index < symbol_times.size
                else None
            )
            pattern_index: int | None = None
            pattern_status = "outside"
            if pattern_result is not None and time_s is not None:
                candidate_index = int(
                    np.floor(
                        (time_s - pattern_result.pattern_start_time_s)
                        * float(pattern_result.metadata["symbol_rate_hz"])
                        + 1e-6
                    )
                )
                if 0 <= candidate_index < len(matched_pattern_symbols):
                    pattern_index = candidate_index
                    pattern_status = (
                        "matched"
                        if symbol == matched_pattern_symbols[candidate_index]
                        else "mismatch"
                    )
            rows.append(
                [index, symbol, ordered_bits, time_s, pattern_index, pattern_status]
            )

        return {
            "schema": _SYMBOL_TABLE_EXPORT_SCHEMA,
            "version": _SYMBOL_TABLE_EXPORT_VERSION,
            "metadata": {
                "source": recording.source,
                "center_frequency_hz": float(recording.center_frequency_hz),
                "sample_rate_hz": float(recording.sample_rate_hz),
                "modulation": signal.modulation.value,
                "modulation_order": int(signal.modulation.order),
                "symbol_rate_hz": float(signal.symbol_rate_hz),
                "symbol_mapping": signal.symbol_mapping,
                "bit_ordering": self.bit_order_combo.currentText(),
                "result_symbol_count": int(symbols.size),
                "pattern": pattern_metadata,
            },
            "columns": [
                "index",
                "symbol",
                "bits",
                "time_s",
                "pattern_index",
                "pattern_status",
            ],
            "rows": rows,
        }

    def _export_symbol_table(self) -> None:
        try:
            document = self._symbol_table_export_document()
        except RuntimeError as error:
            self.statusBar().showMessage(str(error))
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Symbol Table",
            self._last_directory("symbol_table"),
            "VSA Symbol Table (*.vsasymbols.json);;JSON files (*.json)",
        )
        if not path:
            return
        path = self._with_suffix(path, ".vsasymbols.json")
        self._remember_directory("symbol_table", path)
        try:
            Path(path).write_text(
                json.dumps(document, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            self.statusBar().showMessage(
                f"Symbol Table exported - {Path(path).name}"
            )
        except OSError as error:
            QtWidgets.QMessageBox.critical(
                self, "Symbol Table Export Error", str(error)
            )

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
            "acquisition_trigger": {
                "source": self.acquisition_trigger_source_combo.currentData(),
                "level_dbm": self.acquisition_trigger_level_spin.value(),
                "slope": self.acquisition_trigger_slope_combo.currentData(),
                "offset_symbols": self.acquisition_trigger_offset_spin.value(),
                "hysteresis_db": self.acquisition_trigger_hysteresis_spin.value(),
            },
            "burst_search": {
                "enabled": self.iq_power_trigger_check.isChecked(),
                "level_dbm": self.iq_power_trigger_level_spin.value(),
                "hysteresis_db": self.iq_power_trigger_hysteresis_spin.value(),
                "envelope_average_symbols": (
                    self.iq_power_trigger_average_spin.value()
                ),
                "dropout_symbols": self.iq_power_trigger_dropout_spin.value(),
                "holdoff_symbols": self.iq_power_trigger_holdoff_spin.value(),
                "search_start_offset_symbols": (
                    self.iq_power_trigger_offset_spin.value()
                ),
                "limit_result_to_active_interval": (
                    self.iq_power_trigger_limit_result_check.isChecked()
                ),
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
                "allow_inverted_fsk_pattern": (
                    self.pattern_allow_inverted_fsk_check.isChecked()
                ),
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
            "result_summary": {
                "visible_items": [
                    item.item_id
                    for item in RESULT_SUMMARY_ITEMS
                    if item.item_id in self._selected_result_summary_ids
                ],
            },
            "display_config": {
                "constellation_trace_mode": (
                    "Density"
                    if self.constellation_density_action.isChecked()
                    else "Flat"
                ),
                "psk_symbol_plot_mode": (
                    "Differential IQ"
                    if self.differential_iq_symbol_plot_action.isChecked()
                    else "Physical IQ"
                ),
            },
        }

    def _save_startup_meas_config(self) -> None:
        document = {
            "schema": _STARTUP_CONFIG_SCHEMA,
            "version": _STARTUP_CONFIG_VERSION,
            "settings": self._meas_config_values(),
        }
        self._preferences.setValue(
            _STARTUP_CONFIG_KEY,
            json.dumps(document, ensure_ascii=False, separators=(",", ":")),
        )
        self._preferences.sync()

    def _restore_startup_meas_config(self) -> bool:
        serialized = self._preferences.value(
            _STARTUP_CONFIG_KEY, "", type=str
        )
        if not serialized:
            return False
        try:
            document = json.loads(serialized)
            if not isinstance(document, dict):
                raise ValueError("startup configuration root must be an object")
            if document.get("schema") != _STARTUP_CONFIG_SCHEMA:
                raise ValueError("startup configuration schema is invalid")
            if document.get("version") != _STARTUP_CONFIG_VERSION:
                raise ValueError("startup configuration version is unsupported")
            settings = document.get("settings")
            if not isinstance(settings, dict):
                raise ValueError("startup configuration settings must be an object")
            self._apply_meas_config_values(settings)
        except (TypeError, ValueError, json.JSONDecodeError):
            # A stale or partially written preference must never prevent the
            # analyzer from opening. Fall back to widget defaults and replace
            # the invalid document on the next clean close.
            self._preferences.remove(_STARTUP_CONFIG_KEY)
            self._preferences.sync()
            return False
        return True

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
            acquisition_trigger = settings.get("acquisition_trigger", {})
            iq_power_trigger = settings.get(
                "burst_search", settings.get("iq_power_trigger", {})
            )
            result_range = settings["result_range"]
            demodulation = settings["demodulation"]
            signal_capture = settings.get("signal_capture", {})
            result_summary = settings.get("result_summary", {})
            display_config = settings.get("display_config", {})
            if not all(isinstance(section, dict) for section in (
                source,
                signal,
                pattern,
                acquisition_trigger,
                iq_power_trigger,
                result_range,
                demodulation,
                signal_capture,
                result_summary,
                display_config,
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
            self.pattern_allow_inverted_fsk_check.setChecked(
                bool(pattern.get("allow_inverted_fsk_pattern", False))
            )
            if acquisition_trigger:
                source_value = str(
                    acquisition_trigger.get("source", TriggerKind.FREE_RUN.value)
                )
                source_index = self.acquisition_trigger_source_combo.findData(
                    source_value
                )
                if source_index < 0:
                    raise ValueError(
                        f"unsupported acquisition trigger source: {source_value!r}"
                    )
                self.acquisition_trigger_source_combo.setCurrentIndex(source_index)
                self.acquisition_trigger_level_spin.setValue(
                    float(acquisition_trigger.get("level_dbm", -20.0))
                )
                slope_value = str(
                    acquisition_trigger.get("slope", TriggerSlope.RISING.value)
                )
                slope_index = self.acquisition_trigger_slope_combo.findData(
                    slope_value
                )
                if slope_index < 0:
                    raise ValueError(
                        f"unsupported acquisition trigger slope: {slope_value!r}"
                    )
                self.acquisition_trigger_slope_combo.setCurrentIndex(slope_index)
                self.acquisition_trigger_offset_spin.setValue(
                    float(acquisition_trigger.get("offset_symbols", 0.0))
                )
                self.acquisition_trigger_hysteresis_spin.setValue(
                    float(acquisition_trigger.get("hysteresis_db", 3.0))
                )
            if iq_power_trigger:
                self.iq_power_trigger_check.setChecked(
                    bool(iq_power_trigger.get("enabled", False))
                )
                self.iq_power_trigger_level_spin.setValue(
                    float(iq_power_trigger.get("level_dbm", -20.0))
                )
                self.iq_power_trigger_hysteresis_spin.setValue(
                    float(iq_power_trigger.get("hysteresis_db", 3.0))
                )
                self.iq_power_trigger_average_spin.setValue(
                    float(iq_power_trigger.get("envelope_average_symbols", 1.0))
                )
                self.iq_power_trigger_dropout_spin.setValue(
                    float(iq_power_trigger.get("dropout_symbols", 8.0))
                )
                self.iq_power_trigger_holdoff_spin.setValue(
                    float(iq_power_trigger.get("holdoff_symbols", 0.0))
                )
                self.iq_power_trigger_offset_spin.setValue(
                    float(
                        iq_power_trigger.get(
                            "search_start_offset_symbols", 0.0
                        )
                    )
                )
                self.iq_power_trigger_limit_result_check.setChecked(
                    bool(
                        iq_power_trigger.get(
                            "limit_result_to_active_interval", True
                        )
                    )
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
            self._selected_result_summary_ids = normalize_result_summary_ids(
                result_summary.get("visible_items")
            )
            constellation_trace_mode = str(
                display_config.get("constellation_trace_mode", "Flat")
            )
            if constellation_trace_mode not in {"Flat", "Density"}:
                raise ValueError(
                    "constellation trace mode must be Flat or Density"
                )
            self.constellation_flat_action.setChecked(
                constellation_trace_mode == "Flat"
            )
            self.constellation_density_action.setChecked(
                constellation_trace_mode == "Density"
            )
            psk_symbol_plot_mode = str(
                display_config.get("psk_symbol_plot_mode", "Physical IQ")
            )
            if psk_symbol_plot_mode not in {"Physical IQ", "Differential IQ"}:
                raise ValueError(
                    "PSK symbol plot mode must be Physical IQ or Differential IQ"
                )
            self.physical_iq_symbol_plot_action.setChecked(
                psk_symbol_plot_mode == "Physical IQ"
            )
            self.differential_iq_symbol_plot_action.setChecked(
                psk_symbol_plot_mode == "Differential IQ"
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid measurement configuration: {error}") from error
        self._sync_signal_controls()
        self._sync_analysis_controls()
        self._sync_capture_settings()
        self._sync_result_summary_item_tree()
        self._render_result_summary()

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
            if self._request_analysis():
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

    def _configure_plot_context_menus(self) -> None:
        """Add VSA scale actions while preserving pyqtgraph's menu."""

        self._plot_context_actions.clear()
        for name, plot in self._plot_widgets():
            menu = plot.getViewBox().getMenu(None)
            if menu is None:
                continue
            reset_action = QtGui.QAction("Reset", menu)
            reset_action.setToolTip("Restore this plot's analysis-complete scale")
            reset_action.triggered.connect(
                lambda _checked=False, plot_name=name, target=plot: (
                    self._reset_plot_scale(plot_name, target)
                )
            )
            menu.insertAction(menu.viewAll, reset_action)
            menu.insertSeparator(menu.viewAll)

            # ViewBox.autoRange also considers overlay graphics such as result
            # regions and infinite boundary lines.  Those items can dominate
            # the bounds and make the actual traces appear tiny.  Reuse the
            # standard action label/location but give it trace-only semantics.
            menu.viewAll.triggered.disconnect(menu.autoRange)
            menu.viewAll.triggered.connect(
                lambda _checked=False, target=plot: self._view_all_plot(target)
            )
            for action in tuple(menu.actions()):
                if action.text() == "Mouse Mode":
                    menu.removeAction(action)
            self._plot_context_actions[name] = {
                "reset": reset_action,
                "view_all": menu.viewAll,
            }

    @staticmethod
    def _trace_bounds(
        plot: pg.PlotWidget,
    ) -> tuple[float, float, float, float] | None:
        """Return finite bounds of visible data traces, excluding overlays."""

        x_min = y_min = np.inf
        x_max = y_max = -np.inf
        found = False
        for item in plot.listDataItems():
            if not item.isVisible():
                continue
            # getData() returns the transformed display dataset and therefore
            # may contain only the current ViewBox when clipToView is enabled.
            # View All must inspect the complete source trace instead.
            x_values, y_values = item.getOriginalDataset()
            if x_values is None or y_values is None:
                continue
            x_values = np.asarray(x_values)
            y_values = np.asarray(y_values)
            count = min(x_values.size, y_values.size)
            if count == 0:
                continue
            x_values = x_values[:count]
            y_values = y_values[:count]
            finite = np.isfinite(x_values) & np.isfinite(y_values)
            if not np.any(finite):
                continue
            x_finite = x_values[finite]
            y_finite = y_values[finite]
            x_min = min(x_min, float(np.min(x_finite)))
            x_max = max(x_max, float(np.max(x_finite)))
            y_min = min(y_min, float(np.min(y_finite)))
            y_max = max(y_max, float(np.max(y_finite)))
            found = True
        if not found:
            return None
        return x_min, x_max, y_min, y_max

    @staticmethod
    def _padded_range(lower: float, upper: float) -> list[float]:
        span = upper - lower
        if span <= np.finfo(float).eps:
            span = max(abs(lower), abs(upper), 1.0) * 0.1
        margin = 0.05 * span
        return [lower - margin, upper + margin]

    def _view_all_plot(self, plot: pg.PlotWidget) -> None:
        bounds = self._trace_bounds(plot)
        if bounds is None:
            return
        x_min, x_max, y_min, y_max = bounds
        view_box = plot.getViewBox()
        if view_box.state.get("aspectLocked", False) is not False:
            limit = max(
                _IQ_PLANE_LIMIT,
                1.05 * max(abs(x_min), abs(x_max), abs(y_min), abs(y_max)),
            )
            x_range = y_range = [-limit, limit]
        else:
            x_range = self._padded_range(x_min, x_max)
            y_range = self._padded_range(y_min, y_max)
        plot.setRange(xRange=x_range, yRange=y_range, padding=0.0)

    def _reset_plot_scale(self, name: str, plot: pg.PlotWidget) -> None:
        ranges = self._analysis_plot_ranges.get(name)
        if ranges is None:
            return
        x_range, y_range = ranges
        plot.setRange(xRange=x_range, yRange=y_range, padding=0.0)

    def _capture_analysis_plot_ranges(self) -> None:
        captured: dict[str, tuple[list[float], list[float]]] = {}
        for name, plot in self._plot_widgets():
            plot.getViewBox().updateAutoRange()
            x_range, y_range = plot.viewRange()
            captured[name] = (list(x_range), list(y_range))
        self._analysis_plot_ranges = captured

    def _reset_graph_scales(self) -> None:
        for name, plot in self._plot_widgets():
            self._reset_plot_scale(name, plot)

    def _selected_modulation(self) -> ModulationKind:
        return ModulationKind(str(self.modulation_combo.currentData()))

    def _sync_signal_controls(self) -> None:
        modulation = self._selected_modulation()
        self.deviation_spin.setEnabled(modulation.family is ModulationFamily.FSK)
        mapping_enabled = modulation.family is ModulationFamily.PSK
        self.mapping_combo.setEnabled(mapping_enabled)
        if not mapping_enabled:
            self.mapping_combo.setCurrentText(NATURAL_MAPPING)
        elif (
            self.mapping_combo.currentText() == BLUETOOTH_EDR_MAPPING
            and modulation not in {ModulationKind.PI4_DQPSK, ModulationKind.DPSK8}
        ):
            self.mapping_combo.setCurrentText(NATURAL_MAPPING)
        if hasattr(self, "pattern_allow_inverted_fsk_check"):
            self.pattern_allow_inverted_fsk_check.setEnabled(
                modulation.family is ModulationFamily.FSK
            )
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
            trigger_source=TriggerKind(
                self.acquisition_trigger_source_combo.currentData()
            ),
            trigger_level_dbm=self.acquisition_trigger_level_spin.value(),
            trigger_slope=TriggerSlope(
                self.acquisition_trigger_slope_combo.currentData()
            ),
            trigger_offset_s=(
                self.acquisition_trigger_offset_spin.value()
                / self.symbol_rate_spin.value()
            ),
            trigger_hysteresis_db=self.acquisition_trigger_hysteresis_spin.value(),
        )

    def _sync_acquisition_trigger_controls(self, _value: object = None) -> None:
        if not hasattr(self, "acquisition_trigger_source_combo"):
            return
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
            self._pluto_capture_thread.cancel()
            self.run_single_action.setEnabled(False)
            self.run_single_button.setEnabled(False)
            self.statusBar().showMessage("Stopping Pluto IQ capture...")
            return
        try:
            settings = self._pluto_capture_settings()
        except ValueError as error:
            QtWidgets.QMessageBox.critical(self, "Pluto Capture Error", str(error))
            return
        self.input_source_combo.setCurrentText("Pluto")
        self.run_single_action.setText("Stop Single")
        self.run_single_button.setText("Stop Single (Pluto)")
        if settings.trigger_source is TriggerKind.POWER_LEVEL:
            capture_status = (
                "Waiting for Pluto I/Q Power trigger - "
                f"{settings.trigger_slope.value}, "
                f"{settings.trigger_level_dbm:.2f} dBm, "
                f"offset {self.acquisition_trigger_offset_spin.value():+.3f} sym"
            )
        else:
            capture_status = (
                "Capturing Pluto IQ - "
                f"{settings.requested_sample_rate_hz / 1e6:.3f} MS/s, "
                f"{settings.capture_samples:,} samples"
            )
        self.statusBar().showMessage(capture_status)
        thread = _PlutoSingleCaptureThread(
            self._pluto_source,
            settings,
            self,
        )
        thread.capture_ready.connect(self._pluto_capture_ready)
        thread.capture_failed.connect(self._pluto_capture_failed)
        thread.capture_cancelled.connect(self._pluto_capture_cancelled)
        thread.finished.connect(self._pluto_capture_stopped)
        thread.finished.connect(thread.deleteLater)
        self._pluto_capture_thread = thread
        self._pluto_capture_started_at = perf_counter()
        thread.start()

    def _pluto_capture_ready(self, recording: object) -> None:
        if not isinstance(recording, IQRecording):
            self._pluto_capture_failed("capture returned an invalid IQ record")
            return
        capture_finished = perf_counter()
        capture_ms = (
            (capture_finished - self._pluto_capture_started_at) * 1e3
            if self._pluto_capture_started_at is not None
            else 0.0
        )
        self.load_recording(
            recording,
            self._signal_from_controls(),
            analysis_context={"capture_ms": capture_ms},
        )

    def _pluto_capture_failed(self, message: str) -> None:
        self._pluto_capture_started_at = None
        self.statusBar().showMessage(f"Pluto capture failed: {message}")
        QtWidgets.QMessageBox.critical(self, "Pluto Capture Error", message)

    def _pluto_capture_cancelled(self) -> None:
        self._pluto_capture_started_at = None
        self.statusBar().showMessage("Pluto IQ capture cancelled")

    def _pluto_capture_stopped(self) -> None:
        self._pluto_capture_started_at = None
        self._pluto_capture_thread = None
        self.run_single_action.setText("Run Single")
        self.run_single_button.setText("Run Single (Pluto)")
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
            allow_inverted_fsk_pattern=(
                self.pattern_allow_inverted_fsk_check.isChecked()
                and signal.modulation.family is ModulationFamily.FSK
            ),
            match_selection=MatchSelectionPolicy.INDEX,
            match_index=self._selected_match_index,
            iq_power_trigger=IQPowerTriggerSettings(
                enabled=self.iq_power_trigger_check.isChecked(),
                level_dbm=self.iq_power_trigger_level_spin.value(),
                hysteresis_db=self.iq_power_trigger_hysteresis_spin.value(),
                envelope_average_symbols=(
                    self.iq_power_trigger_average_spin.value()
                ),
                dropout_symbols=self.iq_power_trigger_dropout_spin.value(),
                holdoff_symbols=self.iq_power_trigger_holdoff_spin.value(),
                search_start_offset_symbols=(
                    self.iq_power_trigger_offset_spin.value()
                ),
                limit_result_to_active_interval=(
                    self.iq_power_trigger_limit_result_check.isChecked()
                ),
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
        self._selected_match_index = 1
        self.session.set_recording(recording)
        self.session.set_signal(signal)
        self._set_analysis_controls_from_recording(recording)
        self._set_controls_from_signal(signal)
        self._request_analysis()

    def load_recording(
        self,
        recording: IQRecording,
        signal: SignalDescription | None = None,
        *,
        analysis_context: dict[str, float] | None = None,
    ) -> None:
        self._selected_match_index = 1
        self.session.set_recording(recording)
        self._set_analysis_controls_from_recording(recording)
        if signal is not None:
            self.session.set_signal(signal)
            self._set_controls_from_signal(signal)
        self._request_analysis(analysis_context=analysis_context)

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
        if self._analysis_thread is not None:
            self.statusBar().showMessage(
                "Analysis is still running; close again after it completes."
            )
            event.ignore()
            return
        self._save_startup_meas_config()
        self._pluto_source.close()
        super().closeEvent(event)

    def _request_analysis(
        self,
        _checked: bool = False,
        *,
        analysis_context: dict[str, float] | None = None,
    ) -> bool:
        """Queue the newest configured analysis without blocking the GUI."""
        if self.session.recording is None:
            return False
        self._selected_symbol_marker_index = None
        try:
            signal = self._signal_from_controls()
            self.session.set_signal(signal)
            self._update_analysis_settings()
            self._configure_pattern_analysis(signal)
            snapshot = self.session.analysis_snapshot()
        except Exception as error:
            self.statusBar().showMessage(f"Analysis setup failed: {error}")
            return False

        self._analysis_generation += 1
        request = (
            self._analysis_generation,
            snapshot,
            dict(analysis_context or {}),
        )
        if self._analysis_thread is not None:
            self._pending_analysis = request
            self.statusBar().showMessage("Analysis queued - waiting for current DSP")
            return True
        self._start_analysis_request(request)
        return True

    def _start_analysis_request(
        self,
        request: tuple[int, VSASession, dict[str, float]],
    ) -> None:
        generation, snapshot, context = request
        thread = _AnalysisThread(generation, snapshot, self)
        thread.analysis_ready.connect(self._analysis_ready)
        thread.analysis_failed.connect(self._analysis_failed)
        thread.finished.connect(self._analysis_stopped)
        thread.finished.connect(thread.deleteLater)
        self._analysis_thread = thread
        self._active_analysis_context = context
        self.statusBar().showMessage("Analyzing I/Q data...")
        thread.start()

    def _analysis_ready(self, generation: int, completed: object) -> None:
        if not isinstance(completed, VSASession):
            self._analysis_failed(generation, completed, "invalid analysis result")
            return
        if generation != self._analysis_generation:
            return
        try:
            self.session.adopt_analysis_results(completed)
        except ValueError:
            return
        display_started = perf_counter()
        self._update_summary()
        self._update_plots(reset_ranges=True)
        display_ms = (perf_counter() - display_started) * 1e3
        self._last_analysis_timings_ms = {
            **self.session.analysis_timings_ms,
            "display": display_ms,
            "total_ui": (
                self.session.analysis_timings_ms.get("total_dsp", 0.0) + display_ms
            ),
        }
        if self.session.pattern_result is not None:
            self._selected_match_index = int(
                self.session.pattern_result.metadata.get("selected_match_index", 1)
            )
        self._update_match_navigation_actions()
        capture_ms = self._active_analysis_context.get("capture_ms")
        if capture_ms is None:
            self.statusBar().showMessage(
                f"Analysis complete - {self.session.recording.sample_count:,} samples | "
                f"DSP {self._last_analysis_timings_ms.get('total_dsp', 0.0):.0f} ms | "
                f"Display {display_ms:.0f} ms"
            )
            return
        dsp_ms = self._last_analysis_timings_ms.get("total_dsp", 0.0)
        total_ms = float(capture_ms) + dsp_ms + display_ms
        self.statusBar().showMessage(
            "Pluto Single complete - "
            f"{self.session.recording.sample_count:,} samples, "
            f"{self.session.recording.sample_rate_hz / 1e6:.3f} MS/s | "
            f"Capture {capture_ms:.0f} ms | DSP {dsp_ms:.0f} ms | "
            f"Display {display_ms:.0f} ms | Total {total_ms:.0f} ms"
        )

    def _analysis_failed(
        self,
        generation: int,
        completed: object,
        message: str,
    ) -> None:
        if generation != self._analysis_generation:
            return
        if isinstance(completed, VSASession):
            try:
                self.session.adopt_analysis_results(completed)
            except ValueError:
                pass
        if self.session.pattern_error is not None and self.session.result is not None:
            self._update_summary()
            self._update_plots(reset_ranges=True)
            self._update_match_navigation_actions()
        self.statusBar().showMessage(f"Analysis failed: {message}")

    def _analysis_stopped(self) -> None:
        self._analysis_thread = None
        self._active_analysis_context = {}
        if self._pending_analysis is None:
            return
        request = self._pending_analysis
        self._pending_analysis = None
        self._start_analysis_request(request)

    def _analyze(self) -> bool:
        if self.session.recording is None:
            return False
        analysis_started = perf_counter()
        self._selected_symbol_marker_index = None
        try:
            signal = self._signal_from_controls()
            self.session.set_signal(signal)
            self._update_analysis_settings()
            self._configure_pattern_analysis(signal)
            result = self.session.analyze()
        except Exception as error:
            if self.session.pattern_error is not None and self.session.result is not None:
                # The core has already invalidated the former Pattern Result.
                # Repaint that state so a previous successful match is not
                # mistaken for the result of this failed search.
                self._update_summary()
                self._update_plots(reset_ranges=True)
                self._update_match_navigation_actions()
            self.statusBar().showMessage(f"Analysis failed: {error}")
            return False
        display_started = perf_counter()
        self._update_summary()
        self._update_plots(reset_ranges=True)
        display_ms = (perf_counter() - display_started) * 1e3
        self._last_analysis_timings_ms = {
            **self.session.analysis_timings_ms,
            "display": display_ms,
            "total_ui": (perf_counter() - analysis_started) * 1e3,
        }
        if self.session.pattern_result is not None:
            self._selected_match_index = int(
                self.session.pattern_result.metadata.get("selected_match_index", 1)
            )
        self._update_match_navigation_actions()
        self.statusBar().showMessage(
            f"Analysis complete - {self.session.recording.sample_count:,} samples | "
            f"DSP {self._last_analysis_timings_ms.get('total_dsp', 0.0):.0f} ms | "
            f"Display {display_ms:.0f} ms"
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
        previous_index = self._selected_match_index
        self._selected_match_index = target
        if not self._request_analysis():
            self._selected_match_index = previous_index
            return False
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

    def _add_selected_symbol_marker(
        self,
        name: str,
        plot: pg.PlotWidget,
        x_value: float,
        y_value: float,
        text: str,
    ) -> None:
        if not np.isfinite(x_value) or not np.isfinite(y_value):
            return
        x_range, y_range = plot.viewRange()
        anchor_x = 0.0 if x_value <= float(np.mean(x_range)) else 1.0
        anchor_y = 1.0 if y_value <= float(np.mean(y_range)) else 0.0
        point = plot.plot(
            [x_value],
            [y_value],
            pen=None,
            symbol="d",
            symbolSize=_SELECTED_MARKER_SIZE,
            symbolBrush=pg.mkBrush(*_SELECTED_MARKER_COLOR),
            symbolPen=pg.mkPen(0, 0, 0, 255, width=2.0),
        )
        point.setZValue(1000.0)
        label = pg.TextItem(
            text=text,
            color=_SELECTED_MARKER_COLOR,
            fill=pg.mkBrush(0, 0, 0, 210),
            border=pg.mkPen(*_SELECTED_MARKER_COLOR, 190, width=1),
            anchor=(anchor_x, anchor_y),
        )
        label.setPos(float(x_value), float(y_value))
        label.setZValue(1001.0)
        plot.addItem(label)
        self._symbol_marker_items[name] = (point, label)

    def _draw_selected_symbol_markers(
        self,
        context: dict[str, object],
    ) -> None:
        symbol_index = self._selected_symbol_marker_index
        if symbol_index is None:
            return
        symbol_times_s = np.asarray(
            context.get("symbol_times_s", ()), dtype=np.float64
        )
        symbol_power_dbm = np.asarray(
            context.get("symbol_power_dbm", ()), dtype=np.float64
        )
        if (
            symbol_index < 0
            or symbol_index >= symbol_times_s.size
            or symbol_index >= symbol_power_dbm.size
        ):
            self._selected_symbol_marker_index = None
            return
        symbol_time_ms = float(symbol_times_s[symbol_index]) * 1e3
        power_dbm = float(symbol_power_dbm[symbol_index])
        self._add_selected_symbol_marker(
            "iq_power",
            self.zero_span_plot,
            symbol_time_ms,
            power_dbm,
            f"Symbol: {symbol_index}\nPower: {power_dbm:+.2f} dBm",
        )

        signal = self.session.signal
        if signal is None:
            return
        if signal.modulation.family is ModulationFamily.FSK:
            modulation_frequency_hz = np.asarray(
                context.get("modulation_frequency_hz", ()), dtype=np.float64
            )
            symbol_frequency_hz = np.asarray(
                context.get("symbol_frequency_hz", ()), dtype=np.float64
            )
            phase_difference = np.asarray(
                context.get("symbol_plot_vectors", ()), dtype=np.complex128
            )
            if symbol_index < modulation_frequency_hz.size:
                frequency_khz = float(modulation_frequency_hz[symbol_index]) / 1e3
                self._add_selected_symbol_marker(
                    "modulation",
                    self.modulation_plot,
                    symbol_time_ms,
                    frequency_khz,
                    f"Symbol: {symbol_index}\nFrequency: {frequency_khz:+.3f} kHz",
                )
            if (
                symbol_index < phase_difference.size
                and symbol_index < symbol_frequency_hz.size
            ):
                vector = complex(phase_difference[symbol_index])
                amplitude = abs(vector)
                phase_degree = float(np.degrees(np.angle(vector)))
                self._add_selected_symbol_marker(
                    "symbol_plot",
                    self.symbol_plot,
                    vector.real,
                    vector.imag,
                    (
                        f"Symbol: {symbol_index}\n"
                        f"Amplitude: {amplitude:.4f}\n"
                        f"Phase: {phase_degree:+.2f} degree"
                    ),
                )
            return

        modulation_vectors = np.asarray(
            context.get("modulation_vectors", ()), dtype=np.complex128
        )
        symbol_plot_vectors = np.asarray(
            context.get("symbol_plot_vectors", ()), dtype=np.complex128
        )
        raw_symbol_vectors = np.asarray(
            context.get("raw_symbol_vectors", ()), dtype=np.complex128
        )
        symbol_plot_reference_vectors = np.asarray(
            context.get("symbol_plot_reference_vectors", ()),
            dtype=np.complex128,
        )
        decoded_symbols = np.asarray(
            context.get("decoded_symbols", ()), dtype=np.int16
        )
        if symbol_index < modulation_vectors.size:
            vector = complex(modulation_vectors[symbol_index])
            self._add_selected_symbol_marker(
                "modulation",
                self.modulation_plot,
                vector.real,
                vector.imag,
                (
                    f"Symbol: {symbol_index}\n"
                    f"Amplitude: {abs(vector):.4f}\n"
                    f"Phase: {np.degrees(np.angle(vector)):+.2f} degree"
                ),
            )
        if (
            symbol_index < symbol_plot_vectors.size
            and symbol_index < raw_symbol_vectors.size
            and symbol_index < symbol_plot_reference_vectors.size
        ):
            vector = complex(symbol_plot_vectors[symbol_index])
            reference = complex(symbol_plot_reference_vectors[symbol_index])
            evm_percent = (
                100.0
                * abs(vector - reference)
                / max(abs(reference), np.finfo(np.float64).tiny)
            )
            self._add_selected_symbol_marker(
                "symbol_plot",
                self.symbol_plot,
                vector.real,
                vector.imag,
                (
                    f"Symbol: {symbol_index}\n"
                    f"Amplitude: {abs(vector):.4f}\n"
                    f"Phase: {np.degrees(np.angle(vector)):+.2f} degree\n"
                    f"EVM: {evm_percent:.2f} %"
                ),
            )

    def _update_plots(self, *, reset_ranges: bool = False) -> None:
        result = self.session.result
        signal = self.session.signal
        if result is None or signal is None:
            return
        self._symbol_marker_items = {}
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
        capture_time_ms, capture_power_dbm = _peak_decimate_xy(
            result.time_s * 1e3,
            result.power_dbm,
        )
        self.zero_span_plot.clear()
        self.zero_span_plot.plot(
            capture_time_ms,
            capture_power_dbm,
            pen=pg.mkPen(_TRACE_COLOR, width=1),
        )
        symbol_times_s = (
            self.session.pattern_result.symbol_time_s
            if self.session.pattern_result is not None
            else result.symbol_time_s
        )
        symbol_power_dbm = (
            np.interp(symbol_times_s, result.time_s, result.power_dbm)
            if symbol_times_s.size
            else np.empty(0, dtype=np.float64)
        )
        marker_context: dict[str, object] = {
            "symbol_times_s": symbol_times_s,
            "symbol_power_dbm": symbol_power_dbm,
        }
        if self.symbol_display_action.isChecked() and symbol_times_s.size:
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
            spectrum_result.spectrum_dbm,
            pen=pg.mkPen(_TRACE_COLOR, width=1),
        )
        self.modulation_plot.clear()
        self.symbol_plot.clear()
        self._constellation_density_item = None
        if signal.modulation.family is ModulationFamily.FSK:
            self.modulation_plot.setDownsampling(auto=True, mode="peak")
            self.modulation_plot.setClipToView(True)
            self.modulation_plot.getAxis("left").enableAutoSIPrefix(True)
            self.modulation_plot.getAxis("bottom").enableAutoSIPrefix(True)
            self.modulation_plot.setLabel("left", "Frequency (kHz)")
            self.modulation_plot.setLabel("bottom", "Time (ms)")
            self.modulation_plot.setAspectLocked(False)
            frequency_time_ms, display_frequency_khz = _peak_decimate_xy(
                display_result.time_s * 1e3,
                display_result.instantaneous_frequency_hz / 1e3,
            )
            self.modulation_plot.plot(
                frequency_time_ms,
                display_frequency_khz,
                pen=pg.mkPen(_TRACE_COLOR, width=1),
            )
            modulation_frequency_hz = (
                np.interp(
                    symbol_times_s,
                    display_result.time_s,
                    display_result.instantaneous_frequency_hz,
                )
                if symbol_times_s.size
                else np.empty(0, dtype=np.float64)
            )
            marker_context["modulation_frequency_hz"] = (
                modulation_frequency_hz
            )
            if (
                self.symbol_display_action.isChecked()
                and symbol_times_s.size
                and modulation_frequency_hz.size
            ):
                symbol_count = min(
                    symbol_times_s.size, modulation_frequency_hz.size
                )
                self._plot_symbol_points(
                    self.modulation_plot,
                    symbol_times_s[:symbol_count] * 1e3,
                    modulation_frequency_hz[:symbol_count] / 1e3,
                )
            if reset_ranges and signal.frequency_deviation_hz is not None:
                y_limit_khz = 1.5 * signal.frequency_deviation_hz / 1e3
                self.modulation_plot.setYRange(
                    -y_limit_khz, y_limit_khz, padding=0.0
                )
            self._add_pattern_range_overlay(
                self.modulation_plot, fit_range=reset_ranges
            )

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
            marker_context["symbol_frequency_hz"] = measured_frequency_hz
            marker_context["symbol_plot_vectors"] = phase_difference
            phase_slice = _decimation_indices(
                phase_difference.size, maximum=20_000
            )
            self._plot_symbol_distribution(
                phase_difference[phase_slice],
                density_limit=_IQ_PLANE_LIMIT,
            )
            self._plot_unit_circle(self.symbol_plot)
            if reset_ranges:
                self.symbol_plot.setXRange(
                    -_IQ_PLANE_LIMIT, _IQ_PLANE_LIMIT, padding=0.0
                )
                self.symbol_plot.setYRange(
                    -_IQ_PLANE_LIMIT, _IQ_PLANE_LIMIT, padding=0.0
                )
        else:
            self.modulation_plot.setDownsampling(auto=False)
            self.modulation_plot.setClipToView(False)
            self.modulation_plot.getAxis("left").enableAutoSIPrefix(False)
            self.modulation_plot.getAxis("bottom").enableAutoSIPrefix(False)
            self.modulation_plot.setLabel("left", "Q")
            self.modulation_plot.setLabel("bottom", "I")
            self.modulation_plot.setAspectLocked(True, ratio=1.0)
            pattern_result = self.session.pattern_result
            analysis_sample_rate_hz = float(
                display_result.metadata.get(
                    "analysis_sample_rate_hz",
                    self.session.recording.sample_rate_hz,
                )
            )
            processed_iq, processed_time_s = _prepare_psk_display_waveform(
                display_result.iq,
                sample_rate_hz=analysis_sample_rate_hz,
                symbol_rate_hz=signal.symbol_rate_hz,
                tx_filter=signal.tx_filter,
                filter_parameter=signal.filter_parameter,
                result_start_time_s=(
                    pattern_result.result_start_time_s
                    if pattern_result is not None
                    else None
                ),
                result_stop_time_s=(
                    pattern_result.result_stop_time_s
                    if pattern_result is not None
                    else None
                ),
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
            marker_context["modulation_vectors"] = symbol_iq
            if pattern_result is not None:
                in_result_range = (
                    (processed_time_s >= pattern_result.result_start_time_s)
                    & (processed_time_s < pattern_result.result_stop_time_s)
                )
                trajectory_iq = processed_iq[in_result_range]
            else:
                trajectory_iq = processed_iq
            trajectory_slice = _decimation_indices(
                trajectory_iq.size, maximum=_MAX_IQ_TRAJECTORY_POINTS
            )
            trajectory_iq = trajectory_iq[trajectory_slice]
            self.modulation_plot.plot(
                trajectory_iq.real,
                trajectory_iq.imag,
                pen=pg.mkPen(_TRACE_COLOR, width=1),
            )
            if self.symbol_display_action.isChecked() and symbol_times_s.size:
                self._plot_symbol_points(
                    self.modulation_plot,
                    symbol_iq.real,
                    symbol_iq.imag,
                )
            if reset_ranges:
                finite_iq = trajectory_iq[
                    np.isfinite(trajectory_iq.real) & np.isfinite(trajectory_iq.imag)
                ]
                maximum_component = (
                    max(
                        float(np.max(np.abs(finite_iq.real))),
                        float(np.max(np.abs(finite_iq.imag))),
                    )
                    if finite_iq.size
                    else 0.0
                )
                trajectory_limit = max(
                    _IQ_PLANE_LIMIT,
                    1.05 * maximum_component,
                )
                self.modulation_plot.setXRange(
                    -trajectory_limit, trajectory_limit, padding=0.0
                )
                self.modulation_plot.setYRange(
                    -trajectory_limit, trajectory_limit, padding=0.0
                )

            self.symbol_plot.getAxis("left").enableAutoSIPrefix(False)
            self.symbol_plot.getAxis("bottom").enableAutoSIPrefix(False)
            self.symbol_plot.setLabel("left", "Q")
            self.symbol_plot.setLabel("bottom", "I")
            self.symbol_plot.setAspectLocked(True, ratio=1.0)
            decoded_symbols = np.asarray(
                self.session.pattern_result.decoded_symbols
                if self.session.pattern_result is not None
                else display_result.decoded_symbols,
                dtype=np.int16,
            )
            reference_symbols = psk_constellation(
                signal.modulation, signal.symbol_mapping
            )[decoded_symbols]
            use_physical_iq = self.physical_iq_symbol_plot_action.isChecked()
            if use_physical_iq:
                raw_constellation_symbols = np.asarray(
                    symbol_iq, dtype=np.complex128
                )
                if signal.modulation.differential:
                    reference_symbols = np.cumprod(reference_symbols)
                constellation_symbols = _physical_constellation_display_symbols(
                    signal.modulation, raw_constellation_symbols
                )
                display_reference_symbols = _physical_constellation_display_symbols(
                    signal.modulation, reference_symbols
                )
            else:
                raw_constellation_symbols = np.asarray(
                    self.session.pattern_result.measured_symbols
                    if self.session.pattern_result is not None
                    else display_result.measured_symbols,
                    dtype=np.complex128,
                )
                constellation_symbols = _constellation_display_symbols(
                    signal.modulation, raw_constellation_symbols
                )
                display_reference_symbols = _constellation_display_symbols(
                    signal.modulation, reference_symbols
                )
            marker_context["raw_symbol_vectors"] = raw_constellation_symbols
            marker_context["symbol_plot_vectors"] = constellation_symbols
            marker_context["symbol_plot_reference_vectors"] = (
                display_reference_symbols
            )
            marker_context["decoded_symbols"] = decoded_symbols
            self._plot_symbol_distribution(
                constellation_symbols,
                density_limit=_IQ_PLANE_LIMIT,
            )
            self._plot_unit_circle(self.symbol_plot)
            # clear() retains the previous ViewBox range.  Explicitly reset
            # both axes because an FSK frequency range or an earlier malformed
            # constellation can otherwise leave all unit-circle symbols offscreen.
            if reset_ranges:
                self.symbol_plot.setXRange(
                    -_IQ_PLANE_LIMIT, _IQ_PLANE_LIMIT, padding=0.0
                )
                self.symbol_plot.setYRange(
                    -_IQ_PLANE_LIMIT, _IQ_PLANE_LIMIT, padding=0.0
                )
        self._draw_selected_symbol_markers(marker_context)
        pattern_result = self.session.pattern_result
        symbols = (
            pattern_result.decoded_symbols
            if pattern_result is not None
            else result.decoded_symbols
        )
        summary_values: dict[str, str] = {
            "modulation": signal.modulation.value,
            "result_symbols": str(symbols.size),
        }
        self.export_symbol_table_action.setEnabled(True)
        finite_power_dbm = spectrum_result.power_dbm[
            np.isfinite(spectrum_result.power_dbm)
        ]
        if finite_power_dbm.size:
            mean_mw = float(np.mean(10.0 ** (finite_power_dbm / 10.0)))
            if mean_mw > 0.0:
                summary_values["power"] = f"{10.0 * np.log10(mean_mw):+.2f} dBm"
        if signal.modulation.family is ModulationFamily.PSK:
            if pattern_result is not None:
                physical_evm = pattern_result.metadata.get(
                    "physical_evm_rms_percent"
                )
                differential_evm = pattern_result.metadata.get(
                    "differential_symbol_evm_rms_percent"
                )
                bluetooth_devm = pattern_result.metadata.get(
                    "bluetooth_devm_rms_percent"
                )
                if physical_evm is not None:
                    summary_values["evm_rms"] = f"{float(physical_evm):.2f} %"
                if differential_evm is not None:
                    summary_values["differential_symbol_evm_rms"] = (
                        f"{float(differential_evm):.2f} %"
                    )
                if bluetooth_devm is not None:
                    summary_values["bluetooth_devm_rms"] = (
                        f"{float(bluetooth_devm):.2f} %"
                    )
            elif spectrum_result.evm_rms_percent is not None:
                item_id = (
                    "differential_symbol_evm_rms"
                    if signal.modulation.differential
                    else "evm_rms"
                )
                summary_values[item_id] = (
                    f"{float(spectrum_result.evm_rms_percent):.2f} %"
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
            selected_result_text = (
                f"{pattern_result.metadata.get('selected_match_index', 1)} / "
                f"{pattern_result.metadata.get('eligible_match_count', 1)}"
            )
            if pattern_result.metadata.get("power_trigger_enabled", False):
                selected_result_text += (
                    " (Trigger "
                    f"{pattern_result.metadata.get('selected_power_trigger_event_index', 1)} / "
                    f"{pattern_result.metadata.get('power_trigger_event_count', 1)})"
                )
            summary_values.update(
                {
                    "pattern_symbols_correct": (
                        "Yes" if pattern_result.pattern_symbol_errors == 0 else "No"
                    ),
                    "pattern_match_variant": str(
                        pattern_result.metadata.get("pattern_match_variant", "Normal")
                    ),
                    "iq_correlation": f"{pattern_result.correlation * 100.0:.2f} %",
                    "carrier_frequency_error": (
                        f"{pattern_result.carrier_frequency_offset_hz / 1e3:+.3f} kHz"
                    ),
                    "estimated_carrier": (
                        f"{(analysis_center_hz + pattern_result.carrier_frequency_offset_hz) / 1e6:.6f} MHz"
                    ),
                    "display": display_name,
                    "match_selection": selected_result_text,
                }
            )
            if signal.modulation.family is ModulationFamily.PSK:
                rate_error_ppm = pattern_result.metadata.get("symbol_rate_error_ppm")
                sync_evm = pattern_result.metadata.get("synchronization_evm_rms")
                if rate_error_ppm is not None:
                    summary_values["symbol_rate_error"] = (
                        f"{float(rate_error_ppm):+.2f} ppm"
                    )
                if sync_evm is not None:
                    summary_values["sync_evm_rms"] = (
                        f"{float(sync_evm) * 100.0:.2f} %"
                    )
                summary_values["psk_carrier_drift"] = (
                    f"{reported_drift_hz_per_s / 1e6:+.3f} kHz/ms"
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
                measured_deviation = pattern_result.frequency_deviation_hz
                reference_deviation = signal.frequency_deviation_hz
                if measured_deviation is not None:
                    summary_values["fsk_measured_deviation"] = (
                        f"{float(measured_deviation) / 1e3:.3f} kHz"
                    )
                if reference_deviation is not None:
                    summary_values["fsk_reference_deviation"] = (
                        f"{float(reference_deviation) / 1e3:.3f} kHz"
                    )
                if measured_deviation is not None and reference_deviation is not None:
                    summary_values["fsk_deviation_error"] = (
                        f"{float(measured_deviation) - float(reference_deviation):+.0f} Hz"
                    )
                summary_values["carrier_frequency_drift"] = (
                    f"{reported_drift_hz_per_s / signal.symbol_rate_hz:+.3f} Hz/Sym"
                )
                if timing_offset is not None and timing_symbols is not None:
                    timing_status = (
                        ""
                        if timing_accepted is not False
                        else f" (rejected; applied {float(applied_timing or 0.0):+.3f})"
                    )
                    summary_values["fractional_timing"] = (
                        f"{float(timing_offset):+.3f} sample "
                        f"({float(timing_symbols) * 100.0:+.2f} % sym)"
                        f"{timing_status}"
                    )
                if frequency_residual is not None:
                    residual_status = (
                        ""
                        if no_drift_residual is None
                        else f" / no drift {float(no_drift_residual) / 1e3:.3f}"
                    )
                    summary_values["frequency_fit_rms"] = (
                        f"{float(frequency_residual) / 1e3:.3f}"
                        f"{residual_status} kHz"
                    )
                    if measured_deviation is not None and measured_deviation > 0.0:
                        summary_values["frequency_error_rms"] = (
                            f"{100.0 * float(frequency_residual) / float(measured_deviation):.2f} %"
                        )
                if timing_confidence is not None:
                    summary_values["timing_confidence"] = (
                        f"{float(timing_confidence):.3f}"
                    )
                if deviation_error is not None:
                    summary_values["deviation_error_percent"] = (
                        f"{float(deviation_error):+.2f} %"
                    )
                summary_values["drift_model"] = (
                    "Accepted"
                    if drift_accepted
                    else f"Rejected ({drift_reason or 'quality gate'})"
                )
                if candidate_drift is not None:
                    summary_values["applied_drift"] = (
                        f"{pattern_result.carrier_frequency_drift_hz_per_s / 1e6:+.3f} kHz/ms"
                    )
        elif self.session.pattern_error:
            summary_values["pattern_symbols_correct"] = "No"
            summary_values["pattern_error"] = self.session.pattern_error
        else:
            if result.frequency_error_hz is not None:
                summary_values["carrier_frequency_error"] = (
                    f"{float(result.frequency_error_hz) / 1e3:+.3f} kHz"
                )
        self._set_result_summary(summary_values)
        shown = symbols[:_MAX_SYMBOL_TABLE_DISPLAY_SYMBOLS]
        row_count = int(np.ceil(shown.size / 10.0))
        self.symbol_table.setUpdatesEnabled(False)
        try:
            self.symbol_table.clearContents()
            self.symbol_table.setRowCount(row_count)
            self.symbol_table.setVerticalHeaderLabels(
                [str(row * 10) for row in range(row_count)]
            )
            matched_pattern_symbols = ()
            if pattern_result is not None:
                configured_symbols = self._parse_pattern_symbols(
                    pattern_result.modulation.order
                )
                matched_pattern_symbols = (
                    tuple(1 - int(symbol) for symbol in configured_symbols)
                    if pattern_result.metadata.get("pattern_match_variant") == "Inverted"
                    else configured_symbols
                )
            for index, symbol in enumerate(shown):
                item = QtWidgets.QTableWidgetItem(str(int(symbol)))
                item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                if pattern_result is not None and index < pattern_result.symbol_time_s.size:
                    symbol_time_s = float(pattern_result.symbol_time_s[index])
                    pattern_index = int(
                        np.floor(
                            (symbol_time_s - pattern_result.pattern_start_time_s)
                            * float(pattern_result.metadata["symbol_rate_hz"])
                            + 1e-6
                        )
                    )
                    if (
                        0 <= pattern_index < len(matched_pattern_symbols)
                        and int(symbol) == int(matched_pattern_symbols[pattern_index])
                    ):
                        item.setBackground(QtGui.QColor(24, 112, 55))
                        item.setForeground(QtGui.QColor(255, 255, 255))
                self.symbol_table.setItem(index // 10, index % 10, item)
        finally:
            self.symbol_table.setUpdatesEnabled(True)
        if (
            self._selected_symbol_marker_index is not None
            and self._selected_symbol_marker_index < shown.size
        ):
            marker_index = self._selected_symbol_marker_index
            self.symbol_table.setCurrentCell(marker_index // 10, marker_index % 10)
        else:
            self.symbol_table.clearSelection()
        self.symbol_table.setToolTip(
            f"Showing {shown.size} of {symbols.size} result-range symbols"
        )
        if reset_ranges:
            self._capture_analysis_plot_ranges()

    @staticmethod
    def _plot_symbol_points(
        plot: pg.PlotWidget, x_values: np.ndarray, y_values: np.ndarray
    ) -> None:
        selection = _decimation_indices(
            len(x_values), maximum=_MAX_TRACE_SYMBOL_POINTS
        )
        plot.plot(
            np.asarray(x_values)[selection],
            np.asarray(y_values)[selection],
            pen=None,
            symbol="o",
            symbolSize=_TRACE_SYMBOL_SIZE,
            symbolBrush=pg.mkBrush(70, 255, 145, 230),
            symbolPen=pg.mkPen(10, 35, 20, 230, width=1),
        )

    @staticmethod
    def _plot_unit_circle(plot: pg.PlotWidget) -> None:
        unit_angle = np.linspace(0.0, 2.0 * np.pi, 361)
        plot.plot(
            np.cos(unit_angle),
            np.sin(unit_angle),
            pen=pg.mkPen((120, 120, 120, 110), width=1),
        )

    def _plot_symbol_distribution(
        self,
        symbols: np.ndarray,
        *,
        density_limit: float,
    ) -> None:
        """Draw the current PSK or FSK symbol vectors as flat points or density."""
        values = np.asarray(symbols, dtype=np.complex128)
        if not self.constellation_density_action.isChecked():
            self.symbol_plot.plot(
                values.real,
                values.imag,
                pen=None,
                symbol="o",
                symbolSize=_SYMBOL_PLOT_FLAT_SIZE,
                symbolBrush=pg.mkBrush(_TRACE_COLOR),
                symbolPen=pg.mkPen(_TRACE_COLOR),
            )
            return

        limit = _constellation_density_extent(values, minimum=density_limit)
        density = _constellation_density(values, limit=limit)
        density_item = pg.ImageItem(axisOrder="row-major")
        lookup_table = np.array(
            pg.colormap.get("turbo").getLookupTable(nPts=256, alpha=True),
            copy=True,
        )
        lookup_table[0, 3] = 0
        density_item.setLookupTable(lookup_table)
        density_item.setImage(
            density,
            levels=_constellation_density_color_levels(density),
        )
        density_item.setRect(
            QtCore.QRectF(-limit, -limit, 2.0 * limit, 2.0 * limit)
        )
        self.symbol_plot.addItem(density_item)
        self._constellation_density_item = density_item
        # Keep a non-rendering data trace so View All uses the finite symbol
        # bounds instead of the density image rectangle.
        self.symbol_plot.plot(
            values.real,
            values.imag,
            pen=None,
            symbol=None,
        )

    def _set_result_summary(self, values: dict[str, str]) -> None:
        self._result_summary_values = dict(values)
        self._render_result_summary()

    def _render_result_summary(self) -> None:
        signal = self.session.signal
        if signal is None:
            rows: list[tuple[str, str]] = []
        else:
            rows = [
                (definition.label, self._result_summary_values.get(definition.item_id, "—"))
                for definition in RESULT_SUMMARY_ITEMS
                if definition.implemented
                and definition.item_id in self._selected_result_summary_ids
                and definition.applies_to(signal.modulation.family)
                and not (
                    definition.item_id == "pattern_error"
                    and definition.item_id not in self._result_summary_values
                )
            ]
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
