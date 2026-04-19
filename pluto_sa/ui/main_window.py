"""Main window."""

from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from pluto_sa.config.spectrum_config import (
    MAX_DISPLAY_SPAN_HZ,
    MAX_INTERNAL_GAIN_DB,
    MIN_INTERNAL_GAIN_DB,
    SpectrumConfig,
)
from pluto_sa.modes.analyzer_mode import AnalyzerMode
from pluto_sa.modes.sweep_controller import SweepController
from pluto_sa.sdr.pluto_receiver import PlutoReceiver
from pluto_sa.signal.detector import DetectorMode, apply_detector
from pluto_sa.signal.rbw import resolve_rbw_hz
from pluto_sa.signal.spectrum_processor import SpectrumProcessor
from pluto_sa.utils.calibration import apply_display_power_correction

X_AXIS_PADDING = 0.02
Y_AXIS_PADDING = 0.03
LEFT_AXIS_WIDTH = 72
BOTTOM_AXIS_HEIGHT = 42
PLUTO_MIN_CENTER_FREQ_MHZ = 70.0
PLUTO_MAX_CENTER_FREQ_MHZ = 6000.0
MIN_SPAN_MHZ = 0.001
MIN_RBW_KHZ = 0.0
MIN_SWEEP_RBW_HZ = 100.0
MAX_SWEEP_RBW_HZ = 3_000_000.0
MAX_REALTIME_RBW_HZ = 55_000_000.0
MAX_REALTIME_FFT_SIZE = 16_384
MIN_TIME_ANALYZER_SAMPLE_RATE_HZ = 521_000
MIN_TIME_ANALYZER_FFT_SIZE = 64
MAX_TIME_ANALYZER_FFT_SIZE = 16_384
SWEEP_LIKE_FFT_GUARD = 14.0
SWEEP_LIKE_GUARD_OFFSET_HZ = 300_000.0
MIN_WIDEBAND_SPAN_HZ = 10_000_000
MAX_WIDEBAND_SPAN_HZ = 6_000_000_000
MIN_WIDEBAND_START_HZ = 80_000_000
MAX_WIDEBAND_STOP_HZ = 5_990_000_000
WIDEBAND_EFFECTIVE_SPAN_HZ = 10_000_000
WIDEBAND_CHUNK_CAPTURE_SPAN_HZ = 20_000_000
WIDEBAND_CHUNK_STEP_HZ = 10_000_000
WIDEBAND_EDGE_OFFSET_HZ = 5_000_000
WIDEBAND_LO_SETTLE_US = 200
WIDEBAND_FLUSH_READS = 5
TIME_ANALYZER_BUFFER_POINTS = 1000
TIME_ANALYZER_WARMUP_DISCARD_COUNT = 5
MIN_TIME_ANALYZER_TIME_SPAN_S = 0.01
MAX_TIME_ANALYZER_TIME_SPAN_S = 10_000.0
MIN_CENTER_FREQ_STEP_MHZ = 0.001
MAX_CENTER_FREQ_STEP_MHZ = 1_000.0
MIN_REF_LEVEL_DBM = -100.0
MAX_REF_LEVEL_DBM = 100.0
MIN_DISPLAY_RANGE_DB = 1.0
MAX_DISPLAY_RANGE_DB = 200.0
MIN_EXT_ATT_DB = 0.0
MAX_EXT_ATT_DB = 100.0
MIN_EXT_GAIN_DB = 0.0
MAX_EXT_GAIN_DB = 100.0
MIN_SWEEP_TIME_MS = 0.001
MAX_SWEEP_TIME_MS = 10_000.0
MIN_SWEEP_POINTS = 11
MAX_SWEEP_POINTS = 1001
MIN_WATERFALL_HISTORY = 1
MAX_WATERFALL_HISTORY = 1000
MIN_TRACE_AVERAGE_COUNT = 1
MAX_TRACE_AVERAGE_COUNT = 1000
MIN_MARKER_FREQUENCY_HZ = 70
MAX_MARKER_FREQUENCY_HZ = 6_000_000_000
MIN_MARKER_STEP_HZ = 1
MAX_MARKER_STEP_HZ = 1_000_000_000
UNBOUNDED_DOUBLE_MIN = -1_000_000_000_000.0
UNBOUNDED_DOUBLE_MAX = 1_000_000_000_000.0
UNBOUNDED_INT_MIN = -2_147_483_648
UNBOUNDED_INT_MAX = 2_147_483_647
PLOT_SPACING = 12
CONTROL_PANEL_WIDTH = 240
WINDOW_WIDTH = 1664
WINDOW_HEIGHT = 980
OUTER_MARGIN_TOTAL = 24
OUTER_SPACING_TOTAL = 12
SIDE_PANEL_HEIGHT = WINDOW_HEIGHT - 24
STATUS_PANEL_HEIGHT = 108
PLOT_WIDTH = WINDOW_WIDTH - CONTROL_PANEL_WIDTH - OUTER_MARGIN_TOTAL - OUTER_SPACING_TOTAL
PLOT_HEIGHT = (SIDE_PANEL_HEIGHT - STATUS_PANEL_HEIGHT - (PLOT_SPACING * 2)) // 2
DUAL_PLOT_TOTAL_HEIGHT = PLOT_HEIGHT * 2 + PLOT_SPACING
GRAPH_VIEW_BOTH = "both"
GRAPH_VIEW_WATERFALL_ONLY = "waterfall_only"
GRAPH_VIEW_SPECTRUM_ONLY = "spectrum_only"
GRAPH_VIEW_LABELS = {
    GRAPH_VIEW_BOTH: "Both",
    GRAPH_VIEW_WATERFALL_ONLY: "Waterfall Only",
    GRAPH_VIEW_SPECTRUM_ONLY: "Spectrum Only",
}
GRAPH_VIEW_OPTIONS = [
    GRAPH_VIEW_BOTH,
    GRAPH_VIEW_WATERFALL_ONLY,
    GRAPH_VIEW_SPECTRUM_ONLY,
]
SWEEP_STATE_RUNNING = "running"
SWEEP_STATE_SINGLE = "single"
SWEEP_STATE_STOPPED = "stopped"
TRACE_TYPE_LIVE = "Live"
TRACE_TYPE_MAX_HOLD = "Max Hold"
TRACE_TYPE_AVERAGE = "Average"
TRACE_TYPE_OPTIONS = [TRACE_TYPE_LIVE, TRACE_TYPE_MAX_HOLD, TRACE_TYPE_AVERAGE]
TRACE_COLORS = ["#FFFC12", "#78FFEC", "#FC05FF", "#00FF07"]
FFT_SIZE_OPTIONS = [str(2**power) for power in range(6, 15)]
SWEEP_DETECTOR_OPTIONS = [
    DetectorMode.SAMPLE,
    DetectorMode.PEAK,
    DetectorMode.RMS,
]
PERSISTENCE_AMPLITUDE_BINS = 256
PERSISTENCE_HIT_INCREMENT = 5.0
PERSISTENCE_DECAY_VALUES = {
    "Fast": 0.96,
    "Medium": 0.985,
    "Slow": 0.995,
}
SELECTED_BUTTON_PREFIX = "> "
UNSELECTED_BUTTON_PREFIX = "  "


@dataclass
class MarkerState:
    """Persisted marker configuration for future multi-marker expansion."""

    name: str
    is_enabled: bool = False
    trace_name: str = "Trace1"
    frequency_hz: int = 2_440_000_000
    step_hz: int = 1_000_000
    time_sec: float = 0.0
    time_step_sec: float = 1.0
    continuous_peak_enabled: bool = False
    sweep_snapshot_power_db: float | None = None


@dataclass
class TraceState:
    """Persisted trace configuration and runtime buffers."""

    name: str
    color_hex: str
    is_visible: bool = True
    trace_type: str = TRACE_TYPE_LIVE
    hold_enabled: bool = False
    average_count: int = 10
    display_db: np.ndarray | None = None
    max_hold_power: np.ndarray | None = None
    average_power: np.ndarray | None = None


@dataclass
class WidebandRuntimeState:
    start_hz: int
    stop_hz: int
    chunk_centers_hz: np.ndarray
    chunk_freq_ranges_hz: list[tuple[int, int]]
    chunk_source_ranges: list[tuple[int, int]]
    chunk_slice_ranges: list[tuple[int, int]]
    composite_freq_axis_ghz: np.ndarray
    composite_display_db: np.ndarray
    current_chunk_index: int = 0


class FrequencyAxisItem(pg.AxisItem):
    """Format frequency ticks with a fixed GHz precision."""

    def tickStrings(self, values, scale, spacing):
        return [f"{value:.3f}" for value in values]


class RealtimeSpectrumWindow(QtWidgets.QMainWindow):
    """Own GUI construction and rendering only."""

    def __init__(
        self,
        config: SpectrumConfig,
        receiver: PlutoReceiver,
        processor: SpectrumProcessor,
        sweep_controller: SweepController,
        calibration_offset_db: float,
    ) -> None:
        super().__init__()
        self.config = config
        self.receiver = receiver
        self.processor = processor
        self.sweep_controller = sweep_controller
        self.calibration_offset_db = calibration_offset_db
        self.graph_view_mode = GRAPH_VIEW_BOTH
        self._saved_realtime_graph_view_mode = GRAPH_VIEW_BOTH
        self.sweep_state = SWEEP_STATE_RUNNING
        self.actual_sweep_time_s = 0.0
        self._pending_sweep_marker_update = False
        self._last_sweep_drawn_completed_points = -1
        self._last_sweep_callback_time: float | None = None
        self._sweep_callback_sequence = 0
        self._sweep_like_suppress_progress_until_first_complete = False

        self.setWindowTitle("PlutoSDR Real-Time Spectrum Prototype")
        self.setFixedSize(WINDOW_WIDTH, WINDOW_HEIGHT)

        self.frame_count_total = 0
        self.frame_count_interval = 0
        self.start_time = time.perf_counter()
        self.last_report_time = self.start_time

        self.prev_frame_time = None
        self.frame_dt_history: list[float] = []
        self.drop_count = 0

        self._last_received_samples_total = 0
        self.received_samples_interval = 0
        self._page_history: list[tuple[str, QtWidgets.QWidget]] = []
        self.trace_states = [
            TraceState(name=f"Trace{index + 1}", color_hex=TRACE_COLORS[index])
            for index in range(4)
        ]
        self.trace_states[0].is_visible = True
        self.trace_states[1].is_visible = False
        self.trace_states[2].is_visible = False
        self.trace_states[3].is_visible = False
        self.marker_states = [
            MarkerState(
                name="Marker1",
                trace_name="Trace1",
                frequency_hz=self.config.center_freq_hz,
            ),
            MarkerState(
                name="Marker2",
                trace_name="Trace1",
                frequency_hz=self.config.center_freq_hz,
            ),
            MarkerState(
                name="Marker3",
                trace_name="Trace1",
                frequency_hz=self.config.center_freq_hz,
            ),
            MarkerState(
                name="Marker4",
                trace_name="Trace1",
                frequency_hz=self.config.center_freq_hz,
            ),
        ]
        self.marker_trace_options = [trace_state.name for trace_state in self.trace_states]
        self.marker_items: list[tuple[pg.ScatterPlotItem, pg.TextItem]] = []
        self.sweep_progress_item: pg.ScatterPlotItem | None = None
        self.time_analyzer_progress_item: pg.ScatterPlotItem | None = None
        self.marker_controls: list[dict[str, QtWidgets.QPushButton]] = []
        self.trace_controls: list[dict[str, QtWidgets.QPushButton]] = []
        self.trace_menu_buttons: list[QtWidgets.QPushButton] = []
        self.spectrum_curves: list[pg.PlotCurveItem] = []
        self.marker_menu_buttons: list[QtWidgets.QPushButton] = []
        self.page_title_colors: dict[QtWidgets.QWidget, str] = {}
        self.analyzer_mode_option_buttons: dict[AnalyzerMode, QtWidgets.QPushButton] = {}
        self.graph_view_option_buttons: dict[str, QtWidgets.QPushButton] = {}
        self.trace_type_option_buttons: list[dict[str, QtWidgets.QPushButton]] = []
        self.marker_trace_option_buttons: list[dict[str, QtWidgets.QPushButton]] = []
        self.fft_size_option_buttons: dict[str, QtWidgets.QPushButton] = {}
        self.sweep_detector_option_buttons: dict[DetectorMode, QtWidgets.QPushButton] = {}
        self.persistence_decay_option_buttons: dict[str, QtWidgets.QPushButton] = {}
        self._last_display_freq_axis_ghz: np.ndarray | None = None
        self._last_waterfall_freq_axis_ghz: np.ndarray | None = None
        self._last_display_power_db: np.ndarray | None = None
        self._last_current_display_db: np.ndarray | None = None
        self._last_completed_sweep_freq_axis_ghz: np.ndarray | None = None
        self._last_completed_sweep_trace_db: dict[str, np.ndarray] = {}
        self._wideband_chunk_config: SpectrumConfig | None = None
        self._wideband_chunk_processor: SpectrumProcessor | None = None
        self._wideband_runtime_state: WidebandRuntimeState | None = None
        self._time_analyzer_time_axis_s: np.ndarray | None = None
        self._time_analyzer_sample_elapsed_s: np.ndarray | None = None
        self._time_analyzer_trace_db: np.ndarray | None = None
        self._time_analyzer_valid_mask: np.ndarray | None = None
        self._time_analyzer_write_index: int = 0
        self._time_analyzer_sweep_start_timestamp: float | None = None
        self._time_analyzer_sweep_sample_count: int = 0
        self._time_analyzer_sweep_first_timestamp: float | None = None
        self._time_analyzer_sweep_last_timestamp: float | None = None
        self._time_analyzer_last_sweep_samples: int = 0
        self._time_analyzer_last_sweep_avg_dt_s: float | None = None
        self._time_analyzer_discard_samples_remaining: int = 0
        # Temporary debug switch for TA marker trace-source isolation.
        self._ta_marker_debug_force_trace1: bool = False
        self.persistence_enabled = False
        self.persistence_image: pg.ImageItem | None = None
        self.persistence_histogram = np.zeros((PERSISTENCE_AMPLITUDE_BINS, 0), dtype=np.float32)

        self._sync_amplitude_scale_from_config()

        wf_width = len(self.processor.get_decimated_display_freq_axis_ghz())
        self.waterfall_buffer = np.full(
            (self.config.waterfall_history, wf_width),
            self.y_min,
            dtype=np.float32,
        )

        self.frame_period_s = (
            self.config.update_interval_ms / 1000.0
            if self.config.update_interval_ms > 0
            else 1.0 / 60.0
        )
        self._build_ui()
        self._build_timer()
        self.sweep_controller.set_sweep_complete_callback(self._on_sweep_complete)
        self._refresh_sweep_time_estimate()
        self._apply_analyzer_mode_ui_constraints()
        self._update_continuous_button()
        self._update_trace_menu_buttons()
        self._update_marker_menu_buttons()
        self._refresh_status_label()

    def _sync_amplitude_scale_from_config(self) -> None:
        self.y_max = self.config.y_max_dbm
        self.y_min = self.config.y_min_dbm

    def _is_wideband_mode(self) -> bool:
        return self.config.analyzer_mode == AnalyzerMode.WIDEBAND_REALTIME_SA

    def _is_time_analyzer_mode(self) -> bool:
        return self.config.analyzer_mode == AnalyzerMode.TIME_ANALYZER

    def _initialize_time_analyzer_runtime(self) -> None:
        point_count = max(64, int(TIME_ANALYZER_BUFFER_POINTS))
        time_span_s = self._clamp_float(
            float(self.config.time_analyzer_time_span_s),
            MIN_TIME_ANALYZER_TIME_SPAN_S,
            MAX_TIME_ANALYZER_TIME_SPAN_S,
        )
        self.config.time_analyzer_time_span_s = float(time_span_s)
        self._time_analyzer_time_axis_s = np.linspace(
            0.0,
            time_span_s,
            point_count,
            endpoint=True,
            dtype=float,
        )
        self._time_analyzer_sample_elapsed_s = self._time_analyzer_time_axis_s.copy()
        self._time_analyzer_trace_db = np.full(point_count, self.y_min, dtype=float)
        self._time_analyzer_valid_mask = np.zeros(point_count, dtype=bool)
        self._time_analyzer_write_index = 0
        self._time_analyzer_sweep_start_timestamp = None
        self._time_analyzer_sweep_sample_count = 0
        self._time_analyzer_sweep_first_timestamp = None
        self._time_analyzer_sweep_last_timestamp = None
        self._time_analyzer_last_sweep_samples = 0
        self._time_analyzer_last_sweep_avg_dt_s = None
        self._time_analyzer_discard_samples_remaining = int(TIME_ANALYZER_WARMUP_DISCARD_COUNT)

    def _time_analyzer_time_span_s(self) -> float:
        if self._time_analyzer_time_axis_s is None or len(self._time_analyzer_time_axis_s) == 0:
            return 1.0
        return max(1e-6, float(self._time_analyzer_time_axis_s[-1]))

    def _reset_time_analyzer_time_window(self, *, start_timestamp: float | None = None) -> None:
        # Time Analyzer uses fixed-window sweep semantics:
        # each new sweep redraws a fresh 0..TimeSpan window.
        if self._time_analyzer_trace_db is not None:
            self._time_analyzer_trace_db.fill(self.y_min)
        if self._time_analyzer_valid_mask is not None:
            self._time_analyzer_valid_mask.fill(False)
        if (
            self._time_analyzer_time_axis_s is not None
            and self._time_analyzer_sample_elapsed_s is not None
            and len(self._time_analyzer_time_axis_s) == len(self._time_analyzer_sample_elapsed_s)
        ):
            np.copyto(self._time_analyzer_sample_elapsed_s, self._time_analyzer_time_axis_s)
        self._time_analyzer_write_index = 0
        self._time_analyzer_sweep_start_timestamp = start_timestamp
        self._time_analyzer_sweep_sample_count = 0
        self._time_analyzer_sweep_first_timestamp = None
        self._time_analyzer_sweep_last_timestamp = None
        self._hide_time_analyzer_progress_symbol()

    def _finalize_time_analyzer_sweep_stats(self) -> None:
        count = int(self._time_analyzer_sweep_sample_count)
        if count <= 0:
            # Do not overwrite last valid stats with an empty sweep.
            return
        self._time_analyzer_last_sweep_samples = count
        if (
            count >= 2
            and self._time_analyzer_sweep_first_timestamp is not None
            and self._time_analyzer_sweep_last_timestamp is not None
        ):
            elapsed = self._time_analyzer_sweep_last_timestamp - self._time_analyzer_sweep_first_timestamp
            self._time_analyzer_last_sweep_avg_dt_s = max(0.0, elapsed / float(count - 1))
        else:
            self._time_analyzer_last_sweep_avg_dt_s = None
        self._refresh_status_label()

    def _compose_time_analyzer_plot_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if (
            self._time_analyzer_sample_elapsed_s is None
            or self._time_analyzer_trace_db is None
            or self._time_analyzer_valid_mask is None
        ):
            return np.empty(0, dtype=float), np.empty(0, dtype=float)
        if len(self._time_analyzer_sample_elapsed_s) != len(self._time_analyzer_trace_db):
            return np.empty(0, dtype=float), np.empty(0, dtype=float)
        if len(self._time_analyzer_valid_mask) != len(self._time_analyzer_trace_db):
            return np.empty(0, dtype=float), np.empty(0, dtype=float)

        valid_mask = (
            self._time_analyzer_valid_mask
            & np.isfinite(self._time_analyzer_sample_elapsed_s)
            & np.isfinite(self._time_analyzer_trace_db)
        )
        if not np.any(valid_mask):
            return np.empty(0, dtype=float), np.empty(0, dtype=float)

        x_valid = self._time_analyzer_sample_elapsed_s[valid_mask]
        y_valid = self._time_analyzer_trace_db[valid_mask]
        order = np.argsort(x_valid, kind="stable")
        return x_valid[order], y_valid[order]

    def _publish_time_analyzer_accumulated_sweep(self) -> None:
        ta_plot_x, ta_plot_y = self._compose_time_analyzer_plot_arrays()
        self._last_display_freq_axis_ghz = ta_plot_x
        self._last_current_display_db = ta_plot_y
        self._last_display_power_db = ta_plot_y
        self._update_wideband_traces_from_display_db(ta_plot_y)
        self._update_trace_curves()
        self._update_marker_items()
        self._hide_time_analyzer_progress_symbol()
        if self.config.sweep_profile_logging:
            if len(ta_plot_x) == 0 or len(ta_plot_y) == 0:
                print("TAPlot valid_count=0")
            else:
                print(
                    "TAPlot "
                    f"valid_count={len(ta_plot_x)} "
                    f"x_min={float(np.min(ta_plot_x)):.6f} "
                    f"x_max={float(np.max(ta_plot_x)):.6f} "
                    f"y_min={float(np.min(ta_plot_y)):.6f} "
                    f"y_max={float(np.max(ta_plot_y)):.6f}"
                )

    def _resolve_sweep_like_capture_from_rbw(self, rbw_hz: float | None) -> tuple[float, int, int, float]:
        resolved_rbw_hz = self._clip_sweep_rbw(rbw_hz)
        target_bw_hz = int(max(4.0 * resolved_rbw_hz, float(MIN_TIME_ANALYZER_SAMPLE_RATE_HZ)))
        fft_rbw = (8.0 * target_bw_hz) / resolved_rbw_hz
        fft_guard = (SWEEP_LIKE_FFT_GUARD * target_bw_hz) / SWEEP_LIKE_GUARD_OFFSET_HZ
        required_fft = int(np.ceil(max(fft_rbw, fft_guard)))
        fft_size = MIN_TIME_ANALYZER_FFT_SIZE
        while fft_size < required_fft and fft_size < MAX_TIME_ANALYZER_FFT_SIZE:
            fft_size *= 2
        fft_size = self._clamp_int(
            int(fft_size),
            MIN_TIME_ANALYZER_FFT_SIZE,
            MAX_TIME_ANALYZER_FFT_SIZE,
        )
        bin_width_hz = target_bw_hz / float(fft_size)
        return resolved_rbw_hz, target_bw_hz, fft_size, bin_width_hz

    def _resolve_time_analyzer_capture_from_rbw(self) -> tuple[float, int, int, float]:
        return self._resolve_sweep_like_capture_from_rbw(self.config.rbw_hz)

    def _apply_sweep_rbw_driven_capture_settings(self) -> None:
        resolved_rbw_hz, target_bw_hz, fft_size, bin_width_hz = (
            self._resolve_sweep_like_capture_from_rbw(self.config.rbw_hz)
        )
        self.config.rbw_hz = float(resolved_rbw_hz)
        self.config.sweep_sample_rate_hz = int(target_bw_hz)
        self.config.sweep_rf_bandwidth_hz = int(target_bw_hz)
        self.config.fft_size = int(fft_size)
        self.config.sweep_capture_samples_override = int(fft_size)
        self.config.__post_init__()
        if self.config.sweep_profile_logging:
            print(
                "SweepConfig "
                f"RBW={resolved_rbw_hz:.1f}Hz "
                f"BW={target_bw_hz}Hz "
                f"SR={target_bw_hz}Hz "
                f"FFT={fft_size} "
                f"bin_width={bin_width_hz:.1f}Hz"
            )

    def _apply_time_analyzer_rbw_driven_capture_settings(self) -> None:
        resolved_rbw_hz, target_bw_hz, fft_size, bin_width_hz = (
            self._resolve_time_analyzer_capture_from_rbw()
        )
        self.config.rbw_hz = float(resolved_rbw_hz)
        self.config.time_analyzer_sample_rate_hz = int(target_bw_hz)
        self.config.time_analyzer_rf_bandwidth_hz = int(target_bw_hz)
        self.config.fft_size = int(fft_size)
        self._time_analyzer_discard_samples_remaining = int(TIME_ANALYZER_WARMUP_DISCARD_COUNT)
        self.config.__post_init__()
        self.receiver.reconfigure_span(self.config)
        self.processor = SpectrumProcessor(self.config)
        self.processor.update_center_frequency(self.config.center_freq_hz)
        if self.config.sweep_profile_logging:
            print(
                "TimeAnalyzerConfig "
                f"RBW={resolved_rbw_hz:.1f}Hz "
                f"BW={target_bw_hz}Hz "
                f"SR={target_bw_hz}Hz "
                f"FFT={fft_size} "
                f"bin_width={bin_width_hz:.1f}Hz"
            )

    def _get_wideband_start_stop_hz(self) -> tuple[int, int]:
        if (
            self.config.use_start_stop_freq
            and self.config.display_start_freq_hz is not None
            and self.config.display_stop_freq_hz is not None
        ):
            start_hz = self.config.display_start_freq_hz
            stop_hz = self.config.display_stop_freq_hz
        else:
            half_span_hz = self.config.display_span_hz // 2
            start_hz = self.config.center_freq_hz - half_span_hz
            stop_hz = self.config.center_freq_hz + half_span_hz

        start_hz = self._clamp_int(
            int(start_hz),
            MIN_WIDEBAND_START_HZ,
            MAX_WIDEBAND_STOP_HZ,
        )
        stop_hz = self._clamp_int(
            int(stop_hz),
            MIN_WIDEBAND_START_HZ,
            MAX_WIDEBAND_STOP_HZ,
        )
        if stop_hz - start_hz < MIN_WIDEBAND_SPAN_HZ:
            stop_hz = min(
                start_hz + MIN_WIDEBAND_SPAN_HZ,
                MAX_WIDEBAND_STOP_HZ,
            )
        return start_hz, stop_hz

    def _initialize_wideband_runtime(self) -> None:
        start_hz, stop_hz = self._get_wideband_start_stop_hz()
        chunk_start_hz_values = np.arange(
            start_hz,
            stop_hz,
            WIDEBAND_CHUNK_STEP_HZ,
            dtype=np.int64,
        )
        if chunk_start_hz_values.size == 0:
            chunk_start_hz_values = np.array([int(start_hz)], dtype=np.int64)
        chunk_centers_hz = chunk_start_hz_values + WIDEBAND_EDGE_OFFSET_HZ

        chunk_config = deepcopy(self.config)
        chunk_config.analyzer_mode = AnalyzerMode.REALTIME_SA
        chunk_config.display_span_hz = WIDEBAND_CHUNK_CAPTURE_SPAN_HZ
        chunk_config.center_freq_hz = int(chunk_centers_hz[0])
        chunk_config.use_start_stop_freq = False
        chunk_config.display_start_freq_hz = None
        chunk_config.display_stop_freq_hz = None
        chunk_config.rbw_hz = self.config.rbw_hz
        chunk_config.__post_init__()

        self._wideband_chunk_config = chunk_config
        self._wideband_chunk_processor = SpectrumProcessor(chunk_config)
        self.receiver.reconfigure_span(chunk_config)
        self.receiver.retune_lo(int(chunk_centers_hz[0]), update_config=False)

        self._wideband_chunk_processor.update_center_frequency(int(chunk_centers_hz[0]))
        first_chunk_axis_ghz = self._wideband_chunk_processor.get_display_freq_axis_ghz().copy()
        full_chunk_bin_count = len(first_chunk_axis_ghz)
        if full_chunk_bin_count > 1:
            bin_step_ghz = float(first_chunk_axis_ghz[1] - first_chunk_axis_ghz[0])
        else:
            bin_step_ghz = WIDEBAND_CHUNK_SPAN_HZ / 1e9

        chunk_slice_ranges: list[tuple[int, int]] = []
        chunk_freq_ranges_hz: list[tuple[int, int]] = []
        chunk_source_ranges: list[tuple[int, int]] = []
        global_axis_start_ghz = float(first_chunk_axis_ghz[0])
        max_end_index = 0
        for index, chunk_start_hz in enumerate(chunk_start_hz_values):
            chunk_stop_hz = (
                stop_hz
                if index == len(chunk_start_hz_values) - 1
                else min(int(chunk_start_hz) + WIDEBAND_EFFECTIVE_SPAN_HZ, stop_hz)
            )
            self._wideband_chunk_processor.update_center_frequency(int(chunk_centers_hz[index]))
            chunk_axis_ghz = self._wideband_chunk_processor.get_display_freq_axis_ghz().copy()
            chunk_axis_hz = chunk_axis_ghz * 1e9

            source_start_idx = int(np.searchsorted(chunk_axis_hz, int(chunk_start_hz), side="left"))
            source_end_idx = int(np.searchsorted(chunk_axis_hz, int(chunk_stop_hz), side="left"))
            if source_start_idx >= len(chunk_axis_hz):
                source_start_idx = len(chunk_axis_hz) - 1
            if source_end_idx <= source_start_idx:
                source_end_idx = min(len(chunk_axis_hz), source_start_idx + 1)

            dest_start_idx = int(
                round(((int(chunk_start_hz) / 1e9) - global_axis_start_ghz) / bin_step_ghz)
            )
            use_count = source_end_idx - source_start_idx
            dest_end_idx = dest_start_idx + use_count

            chunk_freq_ranges_hz.append((chunk_start_hz, chunk_stop_hz))
            chunk_source_ranges.append((source_start_idx, source_end_idx))
            chunk_slice_ranges.append((dest_start_idx, dest_end_idx))
            max_end_index = max(max_end_index, dest_end_idx)

        composite_axis_ghz = global_axis_start_ghz + np.arange(max_end_index, dtype=float) * bin_step_ghz
        composite_display_db = np.full(len(composite_axis_ghz), np.nan, dtype=float)
        self._wideband_runtime_state = WidebandRuntimeState(
            start_hz=start_hz,
            stop_hz=stop_hz,
            chunk_centers_hz=chunk_centers_hz,
            chunk_freq_ranges_hz=chunk_freq_ranges_hz,
            chunk_source_ranges=chunk_source_ranges,
            chunk_slice_ranges=chunk_slice_ranges,
            composite_freq_axis_ghz=composite_axis_ghz,
            composite_display_db=composite_display_db,
            current_chunk_index=0,
        )

    def _invalidate_wideband_runtime(self) -> None:
        self._wideband_runtime_state = None
        self._wideband_chunk_config = None
        self._wideband_chunk_processor = None

    def _publish_wideband_composite(self) -> None:
        if self._wideband_runtime_state is None:
            return
        runtime = self._wideband_runtime_state
        composite_axis_ghz = self._wideband_runtime_state.composite_freq_axis_ghz
        composite_display_db = self._wideband_runtime_state.composite_display_db
        if len(composite_axis_ghz) == 0:
            return

        self._last_display_freq_axis_ghz = composite_axis_ghz.copy()
        self._last_current_display_db = composite_display_db.copy()
        self._last_display_power_db = composite_display_db.copy()
        self.spectrum_plot.setXRange(
            runtime.start_hz / 1e9,
            runtime.stop_hz / 1e9,
            padding=X_AXIS_PADDING,
        )
        self._update_fixed_ticks()
        self._accumulate_persistence(composite_display_db)
        self._update_wideband_traces_from_display_db(composite_display_db)
        self._last_display_power_db = (
            self.trace_states[0].display_db
            if self.trace_states[0].display_db is not None
            else composite_display_db
        )
        self._append_waterfall_line(composite_display_db)
        waterfall_x_min = runtime.start_hz / 1e9
        waterfall_x_max = runtime.stop_hz / 1e9
        self.waterfall_img.setRect(
            QtCore.QRectF(
                waterfall_x_min,
                0.0,
                waterfall_x_max - waterfall_x_min,
                float(self.config.waterfall_history),
            )
        )
        self.waterfall_plot.setXRange(
            waterfall_x_min,
            waterfall_x_max,
            padding=X_AXIS_PADDING,
        )
        self._update_trace_curves()
        self._update_marker_items()

    def _update_wideband_spectrum(self) -> None:
        if self._wideband_runtime_state is None or self._wideband_chunk_processor is None or self._wideband_chunk_config is None:
            self._initialize_wideband_runtime()
        if self._wideband_runtime_state is None or self._wideband_chunk_processor is None or self._wideband_chunk_config is None:
            return

        runtime = self._wideband_runtime_state
        chunk_count = len(runtime.chunk_centers_hz)
        if chunk_count == 0:
            runtime.current_chunk_index = 0
            return
        if runtime.current_chunk_index >= chunk_count:
            runtime.current_chunk_index = 0
        chunk_index = runtime.current_chunk_index
        center_hz = int(runtime.chunk_centers_hz[chunk_index])
        actual_lo_before_retune_hz = self.receiver.get_current_lo_hz()
        self.receiver.retune_lo(center_hz, update_config=False)
        actual_lo_after_retune_hz = self.receiver.get_current_lo_hz()
        if WIDEBAND_LO_SETTLE_US > 0:
            time.sleep(WIDEBAND_LO_SETTLE_US / 1_000_000.0)
        flush_actual_total = 0
        capture_size = int(self.config.fft_size)
        for _ in range(WIDEBAND_FLUSH_READS):
            flush_actual_total += self.receiver.discard_block(capture_size)
        actual_lo_before_capture_hz = self.receiver.get_current_lo_hz()
        iq = self.receiver.capture_block(capture_size)
        actual_lo_during_capture_hz = self.receiver.get_current_lo_hz()
        processor_window_len = len(self._wideband_chunk_processor.window)
        if len(iq) != processor_window_len:
            self._invalidate_wideband_runtime()
            return
        self._wideband_chunk_processor.update_center_frequency(center_hz)
        power_linear_full = self._wideband_chunk_processor.compute_filtered_power(iq)
        power_linear_display = self._wideband_chunk_processor.extract_display_spectrum(power_linear_full)
        current_power_db_display = 10.0 * np.log10(power_linear_display + 1e-20)
        current_display_db = apply_display_power_correction(
            current_power_db_display,
            self.calibration_offset_db,
            self.config.input_correction_db,
        )

        chunk_axis_ghz = self._wideband_chunk_processor.get_display_freq_axis_ghz()
        source_start_idx, source_end_idx = runtime.chunk_source_ranges[chunk_index]
        start_idx, end_idx = runtime.chunk_slice_ranges[chunk_index]
        runtime.composite_display_db[start_idx:end_idx] = current_display_db[source_start_idx:source_end_idx]
        runtime.current_chunk_index += 1

        if runtime.current_chunk_index >= len(runtime.chunk_centers_hz):
            self._publish_wideband_composite()
            if self.sweep_state == SWEEP_STATE_SINGLE:
                self.timer.stop()
                self.sweep_state = SWEEP_STATE_STOPPED
                self._update_continuous_button()
            else:
                runtime.current_chunk_index = 0

    def _reset_all_measurement_state(
        self,
        *,
        stop_receiver: bool = False,
        stop_sweep: bool = True,
        reset_markers: bool = False,
    ) -> None:
        if stop_receiver and hasattr(self, "timer"):
            self.timer.stop()
            self.receiver.stop()

        if stop_sweep:
            self.sweep_controller.stop()
            self.sweep_controller.reset()
            self.receiver.invalidate_sweep_configuration()
            self._invalidate_wideband_runtime()
            self._pending_sweep_marker_update = False
            self._last_sweep_drawn_completed_points = -1
            self._last_sweep_callback_time = None
            self._sweep_callback_sequence = 0
            self._last_display_freq_axis_ghz = None
            self._last_display_power_db = None
            self._last_current_display_db = None
            self._last_completed_sweep_freq_axis_ghz = None
            self._last_completed_sweep_trace_db = {}
            self._hide_sweep_progress_symbol()

        self.frame_count_interval = 0
        self._last_received_samples_total = 0
        self.received_samples_interval = 0
        self.prev_frame_time = None
        self.frame_dt_history.clear()
        self._reset_trace_runtime_buffers()

        if reset_markers:
            for marker_state in self.marker_states:
                marker_state.is_enabled = False
                marker_state.continuous_peak_enabled = False
            self._update_all_marker_control_states()

        if hasattr(self, "waterfall_plot"):
            self._reset_plot_state()

    def _reset_measurement_state_for_mode_change(self) -> None:
        self._reset_all_measurement_state(
            stop_receiver=True,
            stop_sweep=True,
            reset_markers=True,
        )

    def _apply_analyzer_mode_ui_constraints(self) -> None:
        if self.config.analyzer_mode in (AnalyzerMode.SWEEP_SA, AnalyzerMode.TIME_ANALYZER):
            self._saved_realtime_graph_view_mode = self.graph_view_mode
            self.graph_view_mode = GRAPH_VIEW_SPECTRUM_ONLY
        elif self.graph_view_mode == GRAPH_VIEW_SPECTRUM_ONLY:
            self.graph_view_mode = self._saved_realtime_graph_view_mode

        if hasattr(self, "waterfall_plot"):
            self._apply_display_mode()
        if hasattr(self, "graph_view_button"):
            self._update_graph_view_controls()
        if hasattr(self, "persistence_button"):
            self._update_persistence_controls()
        if hasattr(self, "persistence_decay_button"):
            self._update_realtime_sa_controls()
        if hasattr(self, "analyzer_mode_button"):
            self._update_analyzer_mode_controls()
        if hasattr(self, "freq_span_button") and hasattr(self, "time_span_button"):
            self._update_span_x_scale_controls()
        if hasattr(self, "mode_specific_menu_stack"):
            is_sweep_mode = self.config.analyzer_mode == AnalyzerMode.SWEEP_SA
            self.mode_specific_menu_stack.setCurrentIndex(1 if is_sweep_mode else 0)

    def _change_analyzer_mode(self, analyzer_mode: AnalyzerMode) -> None:
        if self.config.analyzer_mode == analyzer_mode:
            return

        freeze_overlays = self._freeze_mode_switch_panels()
        self._set_left_display_updates_enabled(False)
        try:
            previous_mode = self.config.analyzer_mode
            self.config.analyzer_mode = analyzer_mode
            self._sweep_like_suppress_progress_until_first_complete = analyzer_mode in (
                AnalyzerMode.SWEEP_SA,
                AnalyzerMode.TIME_ANALYZER,
            )
            if analyzer_mode == AnalyzerMode.SWEEP_SA:
                self.config.rbw_hz = self._clip_sweep_rbw(self.config.rbw_hz)
                self._apply_sweep_rbw_driven_capture_settings()
                self.sweep_state = SWEEP_STATE_RUNNING
            elif analyzer_mode == AnalyzerMode.WIDEBAND_REALTIME_SA:
                self.config.fft_size = min(int(self.config.fft_size), MAX_REALTIME_FFT_SIZE)
                if self.config.display_span_hz < MIN_WIDEBAND_SPAN_HZ:
                    self.config.display_span_hz = MIN_WIDEBAND_SPAN_HZ
                self.sweep_state = SWEEP_STATE_RUNNING
            elif analyzer_mode == AnalyzerMode.TIME_ANALYZER:
                self.sweep_state = SWEEP_STATE_RUNNING
            elif previous_mode in (
                AnalyzerMode.SWEEP_SA,
                AnalyzerMode.WIDEBAND_REALTIME_SA,
                AnalyzerMode.TIME_ANALYZER,
            ):
                self._apply_realtime_span_limit()
            if analyzer_mode == AnalyzerMode.REALTIME_SA:
                self.config.fft_size = min(int(self.config.fft_size), MAX_REALTIME_FFT_SIZE)
            self._reset_measurement_state_for_mode_change()
            if analyzer_mode == AnalyzerMode.REALTIME_SA:
                self._rebuild_realtime_runtime_after_mode_change()
            elif analyzer_mode == AnalyzerMode.TIME_ANALYZER:
                self._rebuild_time_analyzer_runtime_after_mode_change()
            self._refresh_sweep_time_estimate()
            self._apply_analyzer_mode_ui_constraints()
            self._refresh_status_label()
            self._page_history.clear()
            self._show_control_page("Main Menu", self.main_menu_page, push_history=False)
            if analyzer_mode == AnalyzerMode.SWEEP_SA:
                self.receiver.invalidate_sweep_configuration()
                self.sweep_controller.reset()
                self._last_display_freq_axis_ghz = None
                self._last_display_power_db = None
                self._last_current_display_db = None
                self._apply_span_update()
                self._start_sweep_continuous()
            elif analyzer_mode == AnalyzerMode.WIDEBAND_REALTIME_SA:
                self._invalidate_wideband_runtime()
                self._reset_plot_state()
                self._start_wideband_continuous()
            elif analyzer_mode == AnalyzerMode.TIME_ANALYZER:
                self._start_time_analyzer_continuous()
            else:
                self._start_realtime_continuous()
        finally:
            self._set_left_display_updates_enabled(True)
            self._thaw_mode_switch_panels(freeze_overlays)

    def _set_left_display_updates_enabled(self, enabled: bool) -> None:
        if hasattr(self, "left_panel"):
            self.left_panel.setUpdatesEnabled(enabled)
        if hasattr(self, "status_label"):
            self.status_label.setUpdatesEnabled(enabled)
        if hasattr(self, "waterfall_plot"):
            self.waterfall_plot.setUpdatesEnabled(enabled)
        if hasattr(self, "spectrum_plot"):
            self.spectrum_plot.setUpdatesEnabled(enabled)

    def _freeze_mode_switch_panels(self) -> list[QtWidgets.QLabel]:
        overlays: list[QtWidgets.QLabel] = []
        for widget in (
            getattr(self, "left_panel", None),
            getattr(self, "control_panel", None),
        ):
            if widget is None or not widget.isVisible():
                continue
            overlay = QtWidgets.QLabel(widget)
            overlay.setPixmap(widget.grab())
            overlay.setGeometry(widget.rect())
            overlay.show()
            overlay.raise_()
            overlays.append(overlay)
        return overlays

    def _thaw_mode_switch_panels(self, overlays: list[QtWidgets.QLabel]) -> None:
        if hasattr(self, "left_panel"):
            self.left_panel.updateGeometry()
            self.left_panel.repaint()
        if hasattr(self, "control_panel"):
            self.control_panel.updateGeometry()
            self.control_panel.repaint()
        QtWidgets.QApplication.processEvents(
            QtCore.QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents
        )
        for overlay in overlays:
            overlay.hide()
            overlay.deleteLater()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer_layout = QtWidgets.QHBoxLayout(central)
        outer_layout.setContentsMargins(12, 12, 12, 12)
        outer_layout.setSpacing(12)

        left_panel = QtWidgets.QWidget()
        left_panel.setFixedWidth(PLOT_WIDTH)
        left_panel.setFixedHeight(SIDE_PANEL_HEIGHT)
        self.left_panel = left_panel
        layout = QtWidgets.QVBoxLayout(left_panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(PLOT_SPACING)

        self.status_label = QtWidgets.QLabel()
        self.status_label.setStyleSheet(
            "color: white; background-color: black; padding-left: 7px; padding-right: 7px; padding-top: 7px;"
        )
        self.status_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        status_font = QtGui.QFont(self.status_label.font())
        status_font.setPointSizeF(status_font.pointSizeF() * 1.4)
        status_font.setBold(True)
        self.status_label.setFont(status_font)
        self.status_label.setText(
            self._make_status_text(
                interval_fps=0.0,
                avg_fps=0.0,
                median_dt_ms=0.0,
                interval_capture_ratio=0.0,
                avg_capture_ratio=0.0,
            )
        )
        self.status_label.setFixedWidth(PLOT_WIDTH)
        self.status_label.setFixedHeight(STATUS_PANEL_HEIGHT)
        self.status_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        layout.addWidget(self.status_label)

        pg.setConfigOptions(antialias=False)

        self.waterfall_plot = pg.PlotWidget(
            axisItems={
                "bottom": FrequencyAxisItem(orientation="bottom"),
                "left": pg.AxisItem(orientation="left"),
            }
        )
        self.waterfall_plot.setBackground("k")
        self._configure_plot_chrome(self.waterfall_plot)
        self.waterfall_plot.setLabel("bottom", "Frequency [GHz]")
        self.waterfall_plot.setLabel("left", "History")
        self.waterfall_plot.getViewBox().invertY(False)
        self.waterfall_plot.getAxis("left").setPen("w")
        self.waterfall_plot.getAxis("bottom").setPen("w")
        self.waterfall_plot.setFixedSize(PLOT_WIDTH, PLOT_HEIGHT)

        self.waterfall_img = pg.ImageItem()
        self.waterfall_plot.addItem(self.waterfall_img)

        self.waterfall_img.setImage(
            np.flipud(self.waterfall_buffer),
            autoLevels=False,
            axisOrder="row-major",
        )

        freq_axis_display_ghz_dec = self.processor.get_decimated_display_freq_axis_ghz()
        x_min = freq_axis_display_ghz_dec[0]
        x_max = freq_axis_display_ghz_dec[-1]

        self.waterfall_img.setRect(
            QtCore.QRectF(
                x_min,
                0.0,
                x_max - x_min,
                float(self.config.waterfall_history),
            )
        )

        self.waterfall_plot.setXRange(x_min, x_max, padding=X_AXIS_PADDING)
        self.waterfall_plot.setYRange(0.0, float(self.config.waterfall_history), padding=0.0)
        self._update_waterfall_ticks()

        lut = pg.ColorMap(
            pos=np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
            color=np.array(
                [
                    [0, 0, 128, 255],
                    [0, 255, 255, 255],
                    [0, 255, 0, 255],
                    [255, 255, 0, 255],
                    [255, 0, 0, 255],
                ],
                dtype=np.ubyte,
            ),
        ).getLookupTable(0.0, 1.0, 256)
        self.waterfall_img.setLookupTable(lut)
        self.waterfall_img.setLevels((self.y_min, self.y_max))

        layout.addWidget(self.waterfall_plot)

        self.spectrum_plot = pg.PlotWidget(
            axisItems={"bottom": FrequencyAxisItem(orientation="bottom")}
        )
        self.spectrum_plot.setBackground("k")
        self._configure_plot_chrome(self.spectrum_plot)
        self.spectrum_plot.showGrid(x=True, y=True, alpha=0.2)

        self.spectrum_plot.getAxis("left").setPen("w")
        self.spectrum_plot.getAxis("bottom").setPen("w")

        self.spectrum_plot.setLabel("bottom", "Frequency [GHz]")
        self.spectrum_plot.setLabel("left", "Amplitude [dBm]")
        self.spectrum_plot.setFixedSize(PLOT_WIDTH, PLOT_HEIGHT)

        freq_axis_display_ghz = self.processor.get_display_freq_axis_ghz()
        self.spectrum_plot.setXRange(
            freq_axis_display_ghz[0],
            freq_axis_display_ghz[-1],
            padding=X_AXIS_PADDING,
        )
        self.spectrum_plot.setYRange(self.y_min, self.y_max, padding=Y_AXIS_PADDING)
        self._update_fixed_ticks()

        self.persistence_image = pg.ImageItem()
        self.persistence_image.setZValue(-20)
        self.persistence_image.setVisible(False)
        self.persistence_image.setLookupTable(self._build_persistence_lut())
        self.spectrum_plot.addItem(self.persistence_image)

        initial_trace = np.full(len(freq_axis_display_ghz), self.y_min, dtype=float)
        for trace_state in self.trace_states:
            trace_state.display_db = initial_trace.copy()
            curve = self.spectrum_plot.plot(
                freq_axis_display_ghz,
                trace_state.display_db,
                pen=self._make_trace_pen(trace_state),
            )
            self.spectrum_curves.append(curve)

        self.sweep_progress_item = pg.ScatterPlotItem(
            size=10,
            pen=pg.mkPen("w"),
            brush=pg.mkBrush("w"),
        )
        self.sweep_progress_item.setVisible(False)
        self.spectrum_plot.addItem(self.sweep_progress_item)
        self.time_analyzer_progress_item = pg.ScatterPlotItem(
            symbol="o",
            size=9,
            pen=pg.mkPen("w"),
            brush=pg.mkBrush("w"),
        )
        self.time_analyzer_progress_item.setVisible(False)
        self.spectrum_plot.addItem(self.time_analyzer_progress_item)

        marker_colors = ["r", "c", "m", "g"]
        for index, color in enumerate(marker_colors):
            marker_item = pg.ScatterPlotItem(size=8, brush=pg.mkBrush(color))
            text_item = pg.TextItem(color="w", anchor=(0, 1))
            text_item.setText(f"M{index + 1}")
            self.spectrum_plot.addItem(marker_item)
            self.spectrum_plot.addItem(text_item)
            self.marker_items.append((marker_item, text_item))

        layout.addWidget(self.spectrum_plot)
        layout.setStretch(0, 0)
        layout.setStretch(1, 1)
        layout.setStretch(2, 1)
        self._apply_display_mode()
        self.installEventFilter(self)
        self.spectrum_plot.installEventFilter(self)
        self.spectrum_plot.viewport().installEventFilter(self)
        self.waterfall_plot.installEventFilter(self)
        self.waterfall_plot.viewport().installEventFilter(self)

        outer_layout.addWidget(left_panel, stretch=1)
        outer_layout.addWidget(self._build_control_panel())
        outer_layout.setStretch(0, 1)
        outer_layout.setStretch(1, 0)

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.Type.MouseButtonPress and isinstance(event, QtGui.QMouseEvent):
            if (
                event.button() == QtCore.Qt.MouseButton.RightButton
                and self._is_control_panel_target(watched)
            ):
                if self._can_navigate_back():
                    self._navigate_back()
                event.accept()
                return True
        if event.type() == QtCore.QEvent.Type.Wheel and isinstance(event, QtGui.QWheelEvent):
            if self._handle_active_marker_wheel(event):
                return True
        return super().eventFilter(watched, event)

    def _is_control_panel_target(self, watched: QtCore.QObject) -> bool:
        if not hasattr(self, "control_panel"):
            return False
        if not isinstance(watched, QtWidgets.QWidget):
            return False
        return watched is self.control_panel or self.control_panel.isAncestorOf(watched)

    def _can_navigate_back(self) -> bool:
        if not hasattr(self, "control_stack") or not hasattr(self, "main_menu_page"):
            return False
        return (
            self.control_stack.currentWidget() is not self.main_menu_page
            and len(self._page_history) > 0
        )

    def _active_marker_index_for_wheel(self) -> int | None:
        if not hasattr(self, "control_stack"):
            return None
        current_page = self.control_stack.currentWidget()
        if current_page in self.marker_detail_pages:
            return self.marker_detail_pages.index(current_page)
        if current_page in self.marker_trace_pages:
            return self.marker_trace_pages.index(current_page)
        return None

    def _marker_wheel_step_hz(self, *, coarse: bool) -> int:
        start_hz, stop_hz = self._resolve_display_start_stop_hz()
        span_hz = max(1.0, stop_hz - start_hz)
        divisor = 100.0 if coarse else 1000.0
        step_hz = max(1, int(round(span_hz / divisor)))
        return step_hz

    def _time_marker_bounds_sec(self) -> tuple[float, float]:
        if self._time_analyzer_time_axis_s is None or len(self._time_analyzer_time_axis_s) == 0:
            return 0.0, 1.0
        return 0.0, float(self._time_analyzer_time_axis_s[-1])

    def _marker_wheel_step_sec(self, *, coarse: bool) -> float:
        start_s, stop_s = self._time_marker_bounds_sec()
        span_s = max(1e-6, stop_s - start_s)
        divisor = 100.0 if coarse else 1000.0
        return max(1e-6, span_s / divisor)

    def _shift_marker_frequency_hz(self, marker_index: int, delta_hz: int) -> bool:
        marker_state = self._marker_state(marker_index)
        if marker_state.continuous_peak_enabled:
            return False
        display_start_hz, display_stop_hz = self._resolve_display_start_stop_hz()
        target_frequency_hz = self._clamp_int(
            marker_state.frequency_hz + int(delta_hz),
            int(display_start_hz),
            int(display_stop_hz),
        )
        marker_state.frequency_hz = target_frequency_hz
        marker_state.is_enabled = True
        marker_state.continuous_peak_enabled = False
        self._clear_marker_sweep_snapshot(marker_state)
        self._update_marker_control_state(marker_index)
        self._update_marker_items()
        return True

    def _shift_marker_time_sec(self, marker_index: int, delta_sec: float) -> bool:
        marker_state = self._marker_state(marker_index)
        if marker_state.continuous_peak_enabled:
            return False
        start_s, stop_s = self._time_marker_bounds_sec()
        marker_state.time_sec = self._clamp_float(
            marker_state.time_sec + float(delta_sec),
            start_s,
            stop_s,
        )
        marker_state.is_enabled = True
        marker_state.continuous_peak_enabled = False
        self._clear_marker_sweep_snapshot(marker_state)
        self._update_marker_control_state(marker_index)
        self._update_marker_items()
        return True

    def _handle_active_marker_wheel(self, event: QtGui.QWheelEvent) -> bool:
        marker_index = self._active_marker_index_for_wheel()
        if marker_index is None:
            return False

        marker_state = self._marker_state(marker_index)
        if marker_state.continuous_peak_enabled:
            event.accept()
            return True

        wheel_delta = event.angleDelta().y()
        if wheel_delta == 0:
            event.accept()
            return True

        coarse = bool(event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier)
        direction = 1 if wheel_delta > 0 else -1
        if self._is_time_analyzer_mode():
            step_sec = self._marker_wheel_step_sec(coarse=coarse)
            moved = self._shift_marker_time_sec(marker_index, direction * step_sec)
        else:
            step_hz = self._marker_wheel_step_hz(coarse=coarse)
            moved = self._shift_marker_frequency_hz(marker_index, direction * step_hz)
        if moved:
            event.accept()
        return True

    def _build_control_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QFrame()
        self.control_panel = panel
        panel.setFixedWidth(CONTROL_PANEL_WIDTH)
        panel.setFixedHeight(WINDOW_HEIGHT - 24)
        panel.setStyleSheet(
            "QFrame { background-color: #1c1c1c; }"
            "QGroupBox { color: white; border: 1px solid #555; margin-top: 12px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }"
            "QPushButton { background-color: #303030; color: white; border: 1px solid #666; padding: 8px; }"
            "QPushButton:hover { background-color: #3c3c3c; }"
        )
        panel_layout = QtWidgets.QVBoxLayout(panel)
        panel_layout.setContentsMargins(10, 10, 10, 10)
        panel_layout.setSpacing(10)

        self.control_title_label = QtWidgets.QLabel("Main Menu")
        title_font = QtGui.QFont(self.control_title_label.font())
        title_font.setPointSizeF(title_font.pointSizeF() * 1.4)
        title_font.setBold(True)
        self.control_title_label.setFont(title_font)
        self.control_title_label.setStyleSheet("color: white; padding: 4px 2px;")
        panel_layout.addWidget(self.control_title_label)

        self.control_stack = QtWidgets.QStackedWidget()
        panel_layout.addWidget(self.control_stack, stretch=1)

        footer_layout = QtWidgets.QHBoxLayout()
        footer_layout.addStretch(1)
        self.back_button = self._make_control_button("Back")
        footer_layout.addWidget(self.back_button)
        panel_layout.addLayout(footer_layout)

        self.main_menu_page = self._build_main_menu_page()
        self.analyzer_mode_page = self._build_analyzer_mode_page()
        self.freq_channel_page = self._build_freq_channel_page()
        self.span_x_scale_page = self._build_span_x_scale_page()
        self.amptd_y_scale_page = self._build_amptd_y_scale_page()
        self.input_page = self._build_input_page()
        self.bw_page = self._build_bw_page()
        self.display_page = self._build_display_page()
        self.graph_view_select_page = self._build_graph_view_select_page()
        self.trace_detector_menu_page = self._build_trace_detector_menu_page()
        self.trace_detail_pages = [
            self._build_trace_detail_page(index) for index in range(len(self.trace_states))
        ]
        self.trace_type_pages = [
            self._build_trace_type_page(index) for index in range(len(self.trace_states))
        ]
        self.detector_page = self._build_detector_page()
        self.marker_menu_page = self._build_marker_menu_page()
        self.marker_detail_pages = [
            self._build_marker_detail_page(index) for index in range(len(self.marker_states))
        ]
        self.fft_size_page = self._build_fft_size_page()
        self.marker_trace_pages = [
            self._build_marker_trace_page(index) for index in range(len(self.marker_states))
        ]
        self.realtime_sa_page = self._build_realtime_sa_page()
        self.persistence_decay_page = self._build_persistence_decay_page()
        self.sweep_page = self._build_sweep_page()
        self.control_stack.addWidget(self.main_menu_page)
        self.control_stack.addWidget(self.analyzer_mode_page)
        self.control_stack.addWidget(self.freq_channel_page)
        self.control_stack.addWidget(self.span_x_scale_page)
        self.control_stack.addWidget(self.amptd_y_scale_page)
        self.control_stack.addWidget(self.input_page)
        self.control_stack.addWidget(self.bw_page)
        self.control_stack.addWidget(self.display_page)
        self.control_stack.addWidget(self.graph_view_select_page)
        self.control_stack.addWidget(self.trace_detector_menu_page)
        for page in self.trace_detail_pages:
            self.control_stack.addWidget(page)
        for page in self.trace_type_pages:
            self.control_stack.addWidget(page)
        self.control_stack.addWidget(self.detector_page)
        self.control_stack.addWidget(self.marker_menu_page)
        for page in self.marker_detail_pages:
            self.control_stack.addWidget(page)
        self.control_stack.addWidget(self.fft_size_page)
        for page in self.marker_trace_pages:
            self.control_stack.addWidget(page)
        self.control_stack.addWidget(self.realtime_sa_page)
        self.control_stack.addWidget(self.persistence_decay_page)
        self.control_stack.addWidget(self.sweep_page)
        self.back_button.clicked.connect(
            self._navigate_back
        )
        for index, page in enumerate(self.trace_detail_pages):
            self.page_title_colors[page] = TRACE_COLORS[index]
        self._show_control_page("Main Menu", self.main_menu_page)
        self._install_control_panel_event_filters()

        return panel

    def _install_control_panel_event_filters(self) -> None:
        if not hasattr(self, "control_panel"):
            return
        self.control_panel.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.NoContextMenu)
        self.control_panel.installEventFilter(self)
        for widget in self.control_panel.findChildren(QtWidgets.QWidget):
            widget.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.NoContextMenu)
            widget.installEventFilter(self)

    def _build_main_menu_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        panel_layout = QtWidgets.QVBoxLayout(page)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(10)

        analyzer_group = QtWidgets.QGroupBox("ANALYZER SETUP")
        self._apply_groupbox_title_font(analyzer_group)
        analyzer_layout = QtWidgets.QVBoxLayout(analyzer_group)
        self.analyzer_mode_button = self._make_value_control_button("Analyzer Mode")
        self.freq_menu_button = self._make_control_button("FREQ Channel")
        self.span_menu_button = self._make_control_button("SPAN X Scale")
        self.amplitude_menu_button = self._make_control_button("AMPTD Y Scale")
        self.input_menu_button = self._make_control_button("Input")
        self.bw_menu_button = self._make_control_button("BW")
        self.display_menu_button = self._make_control_button("Display")
        self.trace_detector_button = self._make_control_button("Trace/Detector")
        self.realtime_sa_menu_button = self._make_control_button("RealTime SA")
        self.sweep_menu_button = self._make_control_button("Sweep")
        analyzer_layout.addWidget(self.analyzer_mode_button)
        analyzer_layout.addWidget(self.freq_menu_button)
        analyzer_layout.addWidget(self.span_menu_button)
        analyzer_layout.addWidget(self.amplitude_menu_button)
        analyzer_layout.addWidget(self.input_menu_button)
        analyzer_layout.addWidget(self.bw_menu_button)
        analyzer_layout.addWidget(self.display_menu_button)
        analyzer_layout.addWidget(self.trace_detector_button)
        self.mode_specific_menu_stack = QtWidgets.QStackedWidget()
        self.mode_specific_menu_stack.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        realtime_sa_menu_page = QtWidgets.QWidget()
        realtime_sa_menu_layout = QtWidgets.QVBoxLayout(realtime_sa_menu_page)
        realtime_sa_menu_layout.setContentsMargins(0, 0, 0, 0)
        realtime_sa_menu_layout.setSpacing(0)
        realtime_sa_menu_layout.addWidget(self.realtime_sa_menu_button)
        sweep_menu_page = QtWidgets.QWidget()
        sweep_menu_layout = QtWidgets.QVBoxLayout(sweep_menu_page)
        sweep_menu_layout.setContentsMargins(0, 0, 0, 0)
        sweep_menu_layout.setSpacing(0)
        sweep_menu_layout.addWidget(self.sweep_menu_button)
        self.mode_specific_menu_stack.addWidget(realtime_sa_menu_page)
        self.mode_specific_menu_stack.addWidget(sweep_menu_page)
        analyzer_layout.addWidget(self.mode_specific_menu_stack)

        sweep_group = QtWidgets.QGroupBox("SWEEP CONTROL")
        self._apply_groupbox_title_font(sweep_group)
        sweep_layout = QtWidgets.QVBoxLayout(sweep_group)
        self.cont_button = self._make_control_button("Continuous")
        self.single_button = self._make_control_button("Single")
        self.reset_button = self._make_control_button("Reset")
        sweep_layout.addWidget(self.cont_button)
        sweep_layout.addWidget(self.single_button)
        sweep_layout.addWidget(self.reset_button)

        trigger_group = QtWidgets.QGroupBox("TRIGGER / MARKER")
        self._apply_groupbox_title_font(trigger_group)
        trigger_layout = QtWidgets.QVBoxLayout(trigger_group)
        self.trigger_button = self._make_control_button("Trigger")
        self.marker_button = self._make_control_button("Marker")
        trigger_layout.addWidget(self.trigger_button)
        trigger_layout.addWidget(self.marker_button)

        panel_layout.addWidget(analyzer_group)
        panel_layout.addWidget(sweep_group)
        panel_layout.addWidget(trigger_group)
        panel_layout.addStretch(1)

        self.analyzer_mode_button.clicked.connect(
            lambda: self._show_control_page("Analyzer Mode", self.analyzer_mode_page)
        )
        self.freq_menu_button.clicked.connect(
            lambda: self._show_control_page("FREQ Channel", self.freq_channel_page)
        )
        self.span_menu_button.clicked.connect(
            lambda: self._show_control_page("SPAN X Scale", self.span_x_scale_page)
        )
        self.amplitude_menu_button.clicked.connect(
            lambda: self._show_control_page("AMPTD Y Scale", self.amptd_y_scale_page)
        )
        self.input_menu_button.clicked.connect(
            lambda: self._show_control_page("Input", self.input_page)
        )
        self.bw_menu_button.clicked.connect(
            lambda: self._show_control_page("BW", self.bw_page)
        )
        self.display_menu_button.clicked.connect(
            lambda: self._show_control_page("Display", self.display_page)
        )
        self.trace_detector_button.clicked.connect(
            lambda: self._show_control_page("Trace/Detector", self.trace_detector_menu_page)
        )
        self.realtime_sa_menu_button.clicked.connect(
            lambda: self._show_control_page("RealTime SA", self.realtime_sa_page)
        )
        self.sweep_menu_button.clicked.connect(
            lambda: self._show_control_page("Sweep", self.sweep_page)
        )
        self.cont_button.clicked.connect(self._on_cont_clicked)
        self.single_button.clicked.connect(self._on_single_clicked)
        self.reset_button.clicked.connect(self._on_reset_clicked)
        self.trigger_button.clicked.connect(lambda: self._show_not_implemented("Trigger"))
        self.marker_button.clicked.connect(
            lambda: self._show_control_page("Marker", self.marker_menu_page)
        )
        self._update_analyzer_mode_controls()
        return page

    def _build_analyzer_mode_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        page_layout = QtWidgets.QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(10)

        for mode in (
            AnalyzerMode.REALTIME_SA,
            AnalyzerMode.WIDEBAND_REALTIME_SA,
            AnalyzerMode.SWEEP_SA,
            AnalyzerMode.TIME_ANALYZER,
        ):
            button = self._make_control_button(mode.value)
            button.clicked.connect(
                lambda _checked=False, selected_mode=mode: self._change_analyzer_mode(
                    selected_mode
                )
            )
            self.analyzer_mode_option_buttons[mode] = button
            page_layout.addWidget(button)

        page_layout.addStretch(1)
        self._update_analyzer_mode_selection_page()
        return page

    def _build_freq_channel_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        page_layout = QtWidgets.QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(10)

        self.freq_center_button = self._make_control_button("Center")
        self.cf_step_button = self._make_control_button("CF Step")
        self.freq_center_left_button = self._make_control_button("<-")
        self.freq_center_right_button = self._make_control_button("->")
        self.freq_start_stop_button = self._make_control_button("Start/Stop")

        center_nudge_layout = QtWidgets.QHBoxLayout()
        center_nudge_layout.setContentsMargins(0, 0, 0, 0)
        center_nudge_layout.setSpacing(10)
        center_nudge_layout.addWidget(self.freq_center_left_button)
        center_nudge_layout.addWidget(self.freq_center_right_button)

        page_layout.addWidget(self.freq_center_button)
        page_layout.addWidget(self.cf_step_button)
        page_layout.addLayout(center_nudge_layout)
        page_layout.addWidget(self.freq_start_stop_button)
        page_layout.addStretch(1)

        self.freq_center_button.clicked.connect(self._on_freq_channel_clicked)
        self.cf_step_button.clicked.connect(self._on_cf_step_clicked)
        self.freq_center_left_button.clicked.connect(
            lambda _checked=False: self._nudge_center_frequency(-1)
        )
        self.freq_center_right_button.clicked.connect(
            lambda _checked=False: self._nudge_center_frequency(1)
        )
        self.freq_start_stop_button.clicked.connect(self._on_freq_start_stop_clicked)
        return page

    def _build_span_x_scale_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        page_layout = QtWidgets.QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(10)

        self.freq_span_button = self._make_control_button("Freq Span")
        self.time_span_button = self._make_control_button("Time Span")

        page_layout.addWidget(self.freq_span_button)
        page_layout.addWidget(self.time_span_button)
        page_layout.addStretch(1)

        self.freq_span_button.clicked.connect(self._on_span_x_scale_clicked)
        self.time_span_button.clicked.connect(self._on_time_span_clicked)
        return page

    def _build_amptd_y_scale_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        page_layout = QtWidgets.QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(10)

        self.ref_level_button = self._make_control_button("Ref Level")
        self.range_button = self._make_control_button("Range")

        page_layout.addWidget(self.ref_level_button)
        page_layout.addWidget(self.range_button)
        page_layout.addStretch(1)

        self.ref_level_button.clicked.connect(self._on_ref_level_clicked)
        self.range_button.clicked.connect(self._on_range_clicked)
        return page

    def _build_input_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        page_layout = QtWidgets.QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(10)

        self.int_gain_button = self._make_control_button("Int Gain")
        self.ext_att_button = self._make_control_button("Ext ATT")
        self.ext_gain_button = self._make_control_button("Ext Gain")

        page_layout.addWidget(self.int_gain_button)
        page_layout.addWidget(self.ext_att_button)
        page_layout.addWidget(self.ext_gain_button)
        page_layout.addStretch(1)

        self.int_gain_button.clicked.connect(self._on_int_gain_clicked)
        self.ext_att_button.clicked.connect(self._on_ext_att_clicked)
        self.ext_gain_button.clicked.connect(self._on_ext_gain_clicked)
        return page

    def _build_bw_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        page_layout = QtWidgets.QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(10)

        self.rbw_button = self._make_control_button("RBW")
        self.vbw_button = self._make_control_button("VBW")

        page_layout.addWidget(self.rbw_button)
        page_layout.addWidget(self.vbw_button)
        page_layout.addStretch(1)

        self.rbw_button.clicked.connect(self._on_rbw_clicked)
        self.vbw_button.clicked.connect(lambda: self._show_not_implemented("VBW"))
        return page

    def _build_display_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        page_layout = QtWidgets.QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(10)

        self.graph_view_button = self._make_value_control_button("Graph View")
        self.persistence_button = self._make_control_button("Persistence OFF")

        page_layout.addWidget(self.graph_view_button)
        page_layout.addWidget(self.persistence_button)
        page_layout.addStretch(1)

        self.graph_view_button.clicked.connect(
            lambda: self._show_control_page("Graph View", self.graph_view_select_page)
        )
        self.persistence_button.clicked.connect(self._toggle_persistence_enabled)
        self._update_graph_view_controls()
        self._update_persistence_controls()
        return page

    def _build_graph_view_select_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        page_layout = QtWidgets.QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(10)

        for mode in GRAPH_VIEW_OPTIONS:
            button = self._make_control_button(GRAPH_VIEW_LABELS[mode])
            button.clicked.connect(
                lambda _checked=False, selected_mode=mode: self._select_graph_view(
                    selected_mode
                )
            )
            self.graph_view_option_buttons[mode] = button
            page_layout.addWidget(button)

        page_layout.addStretch(1)
        self._update_graph_view_selection_page()
        return page

    def _build_trace_detector_menu_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        page_layout = QtWidgets.QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(10)

        for index, trace_state in enumerate(self.trace_states):
            button = self._make_control_button(trace_state.name)
            button.setStyleSheet(f"color: {trace_state.color_hex};")
            button.clicked.connect(
                lambda _checked=False, idx=index: self._show_control_page(
                    self.trace_states[idx].name,
                    self.trace_detail_pages[idx],
                )
            )
            self.trace_menu_buttons.append(button)
            page_layout.addWidget(button)

        detector_button = self._make_control_button("Detector")
        detector_button.clicked.connect(
            lambda: self._show_control_page("Detector", self.detector_page)
        )
        page_layout.addWidget(detector_button)
        page_layout.addStretch(1)
        return page

    def _build_trace_detail_page(self, trace_index: int) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        page_layout = QtWidgets.QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(10)

        visible_button = self._make_control_button("Trace ON")
        type_button = self._make_control_button("Type")
        hold_button = self._make_control_button("Hold")
        average_count_button = self._make_control_button("Average Count")

        page_layout.addWidget(visible_button)
        page_layout.addWidget(type_button)
        page_layout.addWidget(hold_button)
        page_layout.addWidget(average_count_button)
        page_layout.addStretch(1)

        visible_button.clicked.connect(
            lambda _checked=False, idx=trace_index: self._on_trace_visible_clicked(idx)
        )
        type_button.clicked.connect(
            lambda _checked=False, idx=trace_index: self._show_control_page(
                f"{self.trace_states[idx].name} Type",
                self.trace_type_pages[idx],
            )
        )
        hold_button.clicked.connect(
            lambda _checked=False, idx=trace_index: self._on_trace_hold_clicked(idx)
        )
        average_count_button.clicked.connect(
            lambda _checked=False, idx=trace_index: self._on_trace_average_count_clicked(idx)
        )

        self.trace_controls.append(
            {
                "visible": visible_button,
                "type": type_button,
                "hold": hold_button,
                "average_count": average_count_button,
            }
        )
        self._update_trace_control_state(trace_index)
        return page

    def _build_trace_type_page(self, trace_index: int) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        page_layout = QtWidgets.QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(10)

        option_buttons: dict[str, QtWidgets.QPushButton] = {}
        for trace_type in TRACE_TYPE_OPTIONS:
            button = self._make_control_button(trace_type)
            button.clicked.connect(
                lambda _checked=False, idx=trace_index, selected_type=trace_type: (
                    self._select_trace_type(idx, selected_type)
                )
            )
            option_buttons[trace_type] = button
            page_layout.addWidget(button)

        page_layout.addStretch(1)
        self.trace_type_option_buttons.append(option_buttons)
        self.page_title_colors[page] = self.trace_states[trace_index].color_hex
        self._update_trace_type_selection_page(trace_index)
        return page

    def _build_detector_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        page_layout = QtWidgets.QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(10)

        for detector_mode in SWEEP_DETECTOR_OPTIONS:
            button = self._make_control_button(detector_mode.value)
            button.clicked.connect(
                lambda _checked=False, selected_mode=detector_mode: self._select_sweep_detector(
                    selected_mode
                )
            )
            self.sweep_detector_option_buttons[detector_mode] = button
            page_layout.addWidget(button)

        page_layout.addStretch(1)
        self._update_sweep_detector_selection_page()
        return page

    def _build_marker_menu_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        page_layout = QtWidgets.QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(10)

        for index, marker_state in enumerate(self.marker_states):
            button = self._make_control_button(marker_state.name)
            button.clicked.connect(
                lambda _checked=False, idx=index: self._show_control_page(
                    self.marker_states[idx].name,
                    self.marker_detail_pages[idx],
                )
            )
            self.marker_menu_buttons.append(button)
            page_layout.addWidget(button)

        page_layout.addStretch(1)
        return page

    def _build_marker_detail_page(self, marker_index: int) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        page_layout = QtWidgets.QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(10)

        toggle_button = self._make_control_button("ON / OFF")
        trace_button = self._make_control_button("Trace")
        frequency_button = self._make_control_button("Frequency")
        step_button = self._make_control_button("Step")
        left_button = self._make_control_button("<-")
        right_button = self._make_control_button("->")
        peak_search_button = self._make_control_button("Peak Search")
        continuous_peak_button = self._make_control_button("Continuous Peak")
        marker_to_center_button = self._make_control_button("Mkr->CF")

        nudge_layout = QtWidgets.QHBoxLayout()
        nudge_layout.setContentsMargins(0, 0, 0, 0)
        nudge_layout.setSpacing(10)
        nudge_layout.addWidget(left_button)
        nudge_layout.addWidget(right_button)

        for button in (
            toggle_button,
            trace_button,
            frequency_button,
            step_button,
        ):
            page_layout.addWidget(button)

        page_layout.addLayout(nudge_layout)

        for button in (
            peak_search_button,
            continuous_peak_button,
            marker_to_center_button,
        ):
            page_layout.addWidget(button)

        page_layout.addStretch(1)

        toggle_button.clicked.connect(
            lambda _checked=False, idx=marker_index: self._on_marker_toggle_clicked(idx)
        )
        trace_button.clicked.connect(
            lambda _checked=False, idx=marker_index: self._show_control_page(
                f"{self.marker_states[idx].name} Trace",
                self.marker_trace_pages[idx],
            )
        )
        frequency_button.clicked.connect(
            lambda _checked=False, idx=marker_index: self._on_marker_frequency_clicked(idx)
        )
        step_button.clicked.connect(
            lambda _checked=False, idx=marker_index: self._on_marker_step_clicked(idx)
        )
        left_button.clicked.connect(
            lambda _checked=False, idx=marker_index: self._nudge_marker_frequency(idx, -1)
        )
        right_button.clicked.connect(
            lambda _checked=False, idx=marker_index: self._nudge_marker_frequency(idx, 1)
        )
        peak_search_button.clicked.connect(
            lambda _checked=False, idx=marker_index: self._on_marker_peak_search_clicked(idx)
        )
        continuous_peak_button.clicked.connect(
            lambda _checked=False, idx=marker_index: self._on_marker_continuous_peak_clicked(
                idx
            )
        )
        marker_to_center_button.clicked.connect(
            lambda _checked=False, idx=marker_index: self._on_marker_to_center_clicked(idx)
        )

        self.marker_controls.append(
            {
                "toggle": toggle_button,
                "trace": trace_button,
                "frequency": frequency_button,
                "step": step_button,
                "left": left_button,
                "right": right_button,
                "peak_search": peak_search_button,
                "continuous_peak": continuous_peak_button,
                "to_center": marker_to_center_button,
            }
        )
        self._update_marker_control_state(marker_index)

        return page

    def _build_marker_trace_page(self, marker_index: int) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        page_layout = QtWidgets.QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(10)

        option_buttons: dict[str, QtWidgets.QPushButton] = {}
        for trace_name in self.marker_trace_options:
            button = self._make_control_button(trace_name)
            trace_state = self._trace_state_by_name(trace_name)
            if trace_state is not None:
                button.setStyleSheet(f"color: {trace_state.color_hex};")
            button.clicked.connect(
                lambda _checked=False, idx=marker_index, selected_trace=trace_name: (
                    self._select_marker_trace(idx, selected_trace)
                )
            )
            option_buttons[trace_name] = button
            page_layout.addWidget(button)

        page_layout.addStretch(1)
        self.marker_trace_option_buttons.append(option_buttons)
        self._update_marker_trace_selection_page(marker_index)
        return page

    def _build_realtime_sa_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        page_layout = QtWidgets.QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(10)

        self.fft_size_button = self._make_control_button("FFT size")
        self.history_button = self._make_control_button("History")
        self.persistence_decay_button = self._make_value_control_button("Persistence Decay")

        page_layout.addWidget(self.fft_size_button)
        page_layout.addWidget(self.history_button)
        page_layout.addWidget(self.persistence_decay_button)
        page_layout.addStretch(1)

        self.fft_size_button.clicked.connect(
            lambda: self._show_control_page("FFT size", self.fft_size_page)
        )
        self.history_button.clicked.connect(self._on_history_clicked)
        self.persistence_decay_button.clicked.connect(
            lambda: self._show_control_page("Persistence Decay", self.persistence_decay_page)
        )
        self._update_realtime_sa_controls()
        return page

    def _build_persistence_decay_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        page_layout = QtWidgets.QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(10)

        for decay_mode in PERSISTENCE_DECAY_VALUES:
            button = self._make_control_button(decay_mode)
            button.clicked.connect(
                lambda _checked=False, selected_mode=decay_mode: self._select_persistence_decay_mode(
                    selected_mode
                )
            )
            self.persistence_decay_option_buttons[decay_mode] = button
            page_layout.addWidget(button)

        page_layout.addStretch(1)
        self._update_persistence_decay_selection_page()
        return page

    def _build_sweep_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        page_layout = QtWidgets.QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(10)

        self.sweep_time_button = self._make_value_control_button("Swp Time")
        self.sweep_points_button = self._make_value_control_button("Swp Pts")

        page_layout.addWidget(self.sweep_time_button)
        page_layout.addWidget(self.sweep_points_button)
        page_layout.addStretch(1)

        self.sweep_time_button.clicked.connect(self._on_sweep_time_clicked)
        self.sweep_points_button.clicked.connect(self._on_sweep_points_clicked)
        self._update_sweep_controls()
        return page

    def _build_fft_size_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        page_layout = QtWidgets.QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(10)

        for fft_size in FFT_SIZE_OPTIONS:
            button = self._make_control_button(fft_size)
            button.clicked.connect(
                lambda _checked=False, selected_fft_size=fft_size: self._select_fft_size(
                    int(selected_fft_size)
                )
            )
            self.fft_size_option_buttons[fft_size] = button
            page_layout.addWidget(button)

        page_layout.addStretch(1)
        self._update_fft_size_selection_page()
        return page

    def _show_control_page(
        self,
        title: str,
        page: QtWidgets.QWidget,
        *,
        push_history: bool = True,
    ) -> None:
        current_page = self.control_stack.currentWidget()
        if push_history and current_page is not None and current_page is not page:
            self._page_history.append((self.control_title_label.text(), current_page))
        self.control_title_label.setText(title)
        title_color = self.page_title_colors.get(page, "white")
        self.control_title_label.setStyleSheet(
            f"color: {title_color}; padding: 4px 2px;"
        )
        self.control_stack.setCurrentWidget(page)
        self._update_back_button_visibility()

    def _navigate_back(self) -> None:
        if not self._page_history:
            return

        title, page = self._page_history.pop()
        self._show_control_page(title, page, push_history=False)

    def _update_back_button_visibility(self) -> None:
        current_page = self.control_stack.currentWidget()
        show_back = current_page is not self.main_menu_page and len(self._page_history) > 0
        self.back_button.setEnabled(show_back)
        self.back_button.setVisible(show_back)

    def _apply_groupbox_title_font(self, group_box: QtWidgets.QGroupBox) -> None:
        group_font = QtGui.QFont(group_box.font())
        group_font.setPointSizeF(group_font.pointSizeF() * 1.6)
        group_font.setBold(True)
        group_box.setFont(group_font)

    def _make_control_button(self, text: str) -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton(text)
        button_font = QtGui.QFont(button.font())
        button_font.setPointSizeF(button_font.pointSizeF() * 1.6)
        button_font.setBold(True)
        button.setFont(button_font)
        button.setMinimumHeight(50)
        return button

    def _make_value_control_button(self, label_text: str) -> QtWidgets.QPushButton:
        button = self._make_control_button(label_text)
        button.setMinimumHeight(72)
        button.setStyleSheet(
            "QPushButton {"
            " background-color: #303030;"
            " color: white;"
            " border: 1px solid #666;"
            " padding-top: 10px;"
            " padding-bottom: 10px;"
            "}"
            "QPushButton:hover { background-color: #3c3c3c; }"
        )
        return button

    def _configure_plot_chrome(self, plot_widget: pg.PlotWidget) -> None:
        plot_item = plot_widget.getPlotItem()
        view_box = plot_widget.getViewBox()
        plot_item.layout.setContentsMargins(12, 8, 12, 8)
        view_box.setDefaultPadding(0.0)
        plot_item.setMouseEnabled(x=False, y=False)
        plot_item.enableAutoRange(False)
        view_box.setMouseEnabled(x=False, y=False)
        view_box.setMenuEnabled(False)
        plot_item.getAxis("bottom").setStyle(autoExpandTextSpace=True, tickTextOffset=8)
        plot_item.getAxis("left").setStyle(autoExpandTextSpace=True, tickTextOffset=8)
        plot_item.getAxis("left").setWidth(LEFT_AXIS_WIDTH)
        plot_item.getAxis("bottom").setHeight(BOTTOM_AXIS_HEIGHT)

    def _resolve_display_start_stop_hz(self) -> tuple[float, float]:
        if self._is_wideband_mode():
            start_hz, stop_hz = self._get_wideband_start_stop_hz()
            return float(start_hz), float(stop_hz)
        if (
            self.config.display_start_freq_hz is not None
            and self.config.display_stop_freq_hz is not None
            and self.config.display_stop_freq_hz > self.config.display_start_freq_hz
        ):
            return float(self.config.display_start_freq_hz), float(self.config.display_stop_freq_hz)
        freq_axis_display_ghz = self._get_active_spectrum_freq_axis_ghz()
        return float(freq_axis_display_ghz[0] * 1e9), float(freq_axis_display_ghz[-1] * 1e9)

    def _frequency_tick_precision(self, span_hz: float) -> int:
        if span_hz >= 100_000_000.0:
            return 2
        if span_hz >= 10_000_000.0:
            return 3
        if span_hz >= 1_000_000.0:
            return 4
        return 6

    def _build_frequency_ticks_ghz(self, start_hz: float, stop_hz: float) -> list[tuple[float, str]]:
        start_ghz = start_hz / 1e9
        stop_ghz = stop_hz / 1e9
        span_hz = max(1.0, stop_hz - start_hz)
        precision = self._frequency_tick_precision(span_hz)
        tick_values = np.linspace(start_ghz, stop_ghz, 11)
        return [(value, f"{value:.{precision}f}") for value in tick_values]

    def _update_fixed_ticks(self) -> None:
        if self._is_time_analyzer_mode():
            axis = self._time_analyzer_time_axis_s
            if axis is None or len(axis) == 0:
                axis = np.array([0.0, 1.0], dtype=float)
            x_values = np.linspace(float(axis[0]), float(axis[-1]), 11)
            x_ticks = [(value, f"{value:.2f}") for value in x_values]
            y_ticks = [
                (value, f"{value:.0f}") for value in np.linspace(self.y_min, self.y_max, 11)
            ]
            self.spectrum_plot.getAxis("bottom").setTicks([x_ticks])
            self.spectrum_plot.getAxis("left").setTicks([y_ticks])
            self.waterfall_plot.getAxis("bottom").setTicks([x_ticks])
            return

        start_hz, stop_hz = self._resolve_display_start_stop_hz()
        x_ticks = self._build_frequency_ticks_ghz(start_hz, stop_hz)
        y_ticks = [
            (value, f"{value:.0f}") for value in np.linspace(self.y_min, self.y_max, 11)
        ]

        self.spectrum_plot.getAxis("bottom").setTicks([x_ticks])
        self.spectrum_plot.getAxis("left").setTicks([y_ticks])
        self.waterfall_plot.getAxis("bottom").setTicks([x_ticks])

    def _update_waterfall_ticks(self) -> None:
        start_hz, stop_hz = self._resolve_display_start_stop_hz()
        x_ticks = self._build_frequency_ticks_ghz(start_hz, stop_hz)
        history_max = float(self.config.waterfall_history)
        y_values = np.linspace(0.0, history_max, 11)
        y_ticks = [(value, f"{int(round(value))}") for value in y_values]
        self.waterfall_plot.getAxis("bottom").setTicks([x_ticks])
        self.waterfall_plot.getAxis("left").setTicks([y_ticks])

    def _get_active_waterfall_freq_axis_ghz(self) -> np.ndarray:
        if self._is_time_analyzer_mode():
            if self._time_analyzer_time_axis_s is not None:
                return self._time_analyzer_time_axis_s
            return np.array([0.0, 1.0], dtype=float)
        if self._is_wideband_mode():
            return self._wideband_waterfall_display_axis_ghz()
        return self.processor.get_decimated_display_freq_axis_ghz()

    def _wideband_waterfall_x_bounds_ghz(self) -> tuple[float, float] | None:
        if not self._is_wideband_mode() or self._wideband_runtime_state is None:
            return None
        return (
            self._wideband_runtime_state.start_hz / 1e9,
            self._wideband_runtime_state.stop_hz / 1e9,
        )

    def _wideband_waterfall_display_axis_ghz(self, line_length: int | None = None) -> np.ndarray:
        bounds = self._wideband_waterfall_x_bounds_ghz()
        if bounds is None:
            start_hz, stop_hz = self._get_wideband_start_stop_hz()
            bounds = (start_hz / 1e9, stop_hz / 1e9)
        start_ghz, stop_ghz = bounds
        if line_length is None:
            if self._wideband_runtime_state is not None:
                line_length = len(self._wideband_runtime_state.composite_display_db)
            elif self._last_display_freq_axis_ghz is not None:
                line_length = len(self._last_display_freq_axis_ghz)
            else:
                line_length = 2
        if line_length <= 1:
            return np.array([start_ghz, stop_ghz], dtype=float)
        return np.linspace(start_ghz, stop_ghz, int(line_length), endpoint=False, dtype=float)

    def _rebuild_waterfall_buffer_for_axis(
        self,
        freq_axis_display_ghz_dec: np.ndarray,
        *,
        x_min: float | None = None,
        x_max: float | None = None,
    ) -> None:
        wf_width = len(freq_axis_display_ghz_dec)
        self.waterfall_buffer = np.full(
            (self.config.waterfall_history, wf_width),
            self.y_min,
            dtype=np.float32,
        )
        self.waterfall_img.setImage(
            np.flipud(self.waterfall_buffer),
            autoLevels=False,
            axisOrder="row-major",
        )
        wideband_bounds = self._wideband_waterfall_x_bounds_ghz()
        if wideband_bounds is not None and x_min is None and x_max is None:
            rect_x_min, rect_x_max = wideband_bounds
        else:
            rect_x_min = freq_axis_display_ghz_dec[0] if x_min is None else x_min
            rect_x_max = freq_axis_display_ghz_dec[-1] if x_max is None else x_max
        self.waterfall_img.setRect(
            QtCore.QRectF(
                rect_x_min,
                0.0,
                rect_x_max - rect_x_min,
                float(self.config.waterfall_history),
            )
        )
        self.waterfall_plot.setXRange(
            rect_x_min,
            rect_x_max,
            padding=X_AXIS_PADDING,
        )
        self.waterfall_plot.setYRange(0.0, float(self.config.waterfall_history), padding=0.0)
        self._last_waterfall_freq_axis_ghz = freq_axis_display_ghz_dec.copy()
        self._update_waterfall_ticks()

    def _append_waterfall_line(self, power_db_line: np.ndarray) -> None:
        if power_db_line.size == 0:
            return
        if self._is_wideband_mode():
            source_axis_ghz = (
                self._wideband_runtime_state.composite_freq_axis_ghz
                if self._wideband_runtime_state is not None
                else self._last_display_freq_axis_ghz
            )
            target_axis_ghz = self._wideband_waterfall_display_axis_ghz(len(power_db_line))
            if (
                source_axis_ghz is not None
                and len(source_axis_ghz) == len(power_db_line)
                and len(target_axis_ghz) == len(power_db_line)
            ):
                power_db_dec = np.interp(
                    target_axis_ghz,
                    source_axis_ghz,
                    power_db_line,
                ).astype(np.float32, copy=False)
            else:
                power_db_dec = power_db_line.astype(np.float32, copy=False)
            freq_axis_display_ghz_dec = target_axis_ghz
        else:
            power_db_dec = power_db_line[:: self.config.waterfall_decimation]
            freq_axis_display_ghz_dec = self._get_active_waterfall_freq_axis_ghz()
        axis_mismatch = (
            self._last_waterfall_freq_axis_ghz is None
            or len(self._last_waterfall_freq_axis_ghz) != len(freq_axis_display_ghz_dec)
            or not np.array_equal(self._last_waterfall_freq_axis_ghz, freq_axis_display_ghz_dec)
        )
        if self.waterfall_buffer.shape[1] != len(power_db_dec) or axis_mismatch:
            wideband_bounds = self._wideband_waterfall_x_bounds_ghz()
            if wideband_bounds is not None:
                self._rebuild_waterfall_buffer_for_axis(
                    freq_axis_display_ghz_dec,
                    x_min=wideband_bounds[0],
                    x_max=wideband_bounds[1],
                )
            else:
                self._rebuild_waterfall_buffer_for_axis(freq_axis_display_ghz_dec)
        self.waterfall_buffer[:-1] = self.waterfall_buffer[1:]
        self.waterfall_buffer[-1] = power_db_dec.astype(np.float32)
        display_image = np.flipud(self.waterfall_buffer)
        self.waterfall_img.setImage(
            display_image,
            autoLevels=False,
            axisOrder="row-major",
        )

    def _current_sweep_state(self) -> str:
        return self.sweep_state

    def _restore_sweep_state(self, previous_state: str) -> None:
        if self.config.analyzer_mode == AnalyzerMode.WIDEBAND_REALTIME_SA:
            if previous_state == SWEEP_STATE_RUNNING:
                self._start_wideband_continuous()
            elif previous_state == SWEEP_STATE_SINGLE:
                self.sweep_state = SWEEP_STATE_SINGLE
                self.receiver.stop()
                self._restart_timer_for_current_mode()
            else:
                self.timer.stop()
                self.sweep_state = SWEEP_STATE_STOPPED
            self._update_continuous_button()
            return
        if self.config.analyzer_mode != AnalyzerMode.SWEEP_SA:
            self._update_continuous_button()
            return
        if previous_state == SWEEP_STATE_RUNNING:
            self._start_sweep_continuous()
        elif previous_state == SWEEP_STATE_SINGLE:
            self.sweep_state = SWEEP_STATE_SINGLE
            self.receiver.stop()
            self.sweep_controller.request_single()
            self._restart_timer_for_current_mode()
        else:
            self.timer.stop()
            self.sweep_controller.stop()
            self.sweep_state = SWEEP_STATE_STOPPED
        self._update_continuous_button()

    def _reset_sweep_display_and_restore_state(self, previous_sweep_state: str) -> None:
        """Reset sweep runtime/display state and resume the prior sweep mode."""
        self.sweep_controller.reset()
        # Clear cached axes/data so tick regeneration uses the new sweep axis immediately.
        self._last_display_freq_axis_ghz = None
        self._last_display_power_db = None
        self._last_current_display_db = None
        self._apply_span_update()
        self._restore_sweep_state(previous_sweep_state)

    def _start_realtime_continuous(self) -> None:
        self.sweep_state = SWEEP_STATE_RUNNING
        self.sweep_controller.stop()
        self.receiver.start()
        self._restart_timer_for_current_mode()
        self._update_continuous_button()

    def _start_time_analyzer_continuous(self) -> None:
        self.sweep_state = SWEEP_STATE_RUNNING
        self.sweep_controller.stop()
        self.receiver.stop()
        self._reset_time_analyzer_time_window(start_timestamp=time.perf_counter())
        self._restart_timer_for_current_mode()
        self._update_continuous_button()

    def _start_wideband_continuous(self) -> None:
        self.sweep_state = SWEEP_STATE_RUNNING
        self.sweep_controller.stop()
        self.receiver.stop()
        self._invalidate_wideband_runtime()
        self._restart_timer_for_current_mode()
        self._update_continuous_button()

    def _apply_realtime_span_limit(self) -> None:
        if self.config.display_span_hz <= MAX_DISPLAY_SPAN_HZ:
            return

        half_span_hz = MAX_DISPLAY_SPAN_HZ // 2
        start_freq_hz = int(self.config.center_freq_hz) - half_span_hz
        stop_freq_hz = int(self.config.center_freq_hz) + half_span_hz
        self._apply_frequency_window(start_freq_hz, stop_freq_hz, use_start_stop_display=self.config.use_start_stop_freq)

    def _rebuild_realtime_runtime_after_mode_change(self) -> None:
        self.config.__post_init__()
        self.receiver.reconfigure_span(self.config)
        self.receiver.retune_lo(self.config.center_freq_hz)
        self.processor.update_span_related(self.config)
        self.processor.update_center_frequency(self.config.center_freq_hz)
        self._reset_plot_state()

    def _rebuild_time_analyzer_runtime_after_mode_change(self) -> None:
        self._apply_time_analyzer_rbw_driven_capture_settings()
        self.receiver.retune_lo(self.config.center_freq_hz)
        self._initialize_time_analyzer_runtime()
        self._reset_plot_state()

    def _update_plot_axis_labels_for_mode(self) -> None:
        if self._is_time_analyzer_mode():
            self.spectrum_plot.setLabel("bottom", "Time [s]")
            self.spectrum_plot.setLabel("left", "Amplitude [dBm]")
        else:
            self.spectrum_plot.setLabel("bottom", "Frequency [GHz]")
            self.spectrum_plot.setLabel("left", "Amplitude [dBm]")

    def _clear_start_stop_display_mode(self) -> None:
        self.config.use_start_stop_freq = False

    def _set_start_stop_display_mode(self, start_freq_hz: int, stop_freq_hz: int) -> None:
        self._apply_frequency_window(start_freq_hz, stop_freq_hz, use_start_stop_display=True)

    def _frequency_bounds_hz(self) -> tuple[int, int]:
        if self._is_wideband_mode():
            return MIN_WIDEBAND_START_HZ, MAX_WIDEBAND_STOP_HZ
        return int(round(PLUTO_MIN_CENTER_FREQ_MHZ * 1e6)), int(round(PLUTO_MAX_CENTER_FREQ_MHZ * 1e6))

    def _minimum_display_span_hz(self) -> int:
        return MIN_WIDEBAND_SPAN_HZ if self._is_wideband_mode() else int(round(MIN_SPAN_MHZ * 1e6))

    def _maximum_display_span_hz(self) -> int:
        if self.config.analyzer_mode == AnalyzerMode.REALTIME_SA:
            return MAX_DISPLAY_SPAN_HZ
        lower_hz, upper_hz = self._frequency_bounds_hz()
        return upper_hz - lower_hz

    def _apply_frequency_window(
        self,
        start_freq_hz: int,
        stop_freq_hz: int,
        *,
        use_start_stop_display: bool,
    ) -> None:
        lower_bound_hz, upper_bound_hz = self._frequency_bounds_hz()
        minimum_span_hz = self._minimum_display_span_hz()
        maximum_span_hz = self._maximum_display_span_hz()

        start_freq_hz = int(start_freq_hz)
        stop_freq_hz = int(stop_freq_hz)
        span_hz = max(stop_freq_hz - start_freq_hz, minimum_span_hz)
        span_hz = min(span_hz, maximum_span_hz)

        start_freq_hz = self._clamp_int(start_freq_hz, lower_bound_hz, upper_bound_hz)
        stop_freq_hz = start_freq_hz + span_hz
        if stop_freq_hz > upper_bound_hz:
            stop_freq_hz = upper_bound_hz
            start_freq_hz = max(lower_bound_hz, stop_freq_hz - span_hz)

        if stop_freq_hz - start_freq_hz < minimum_span_hz:
            stop_freq_hz = min(upper_bound_hz, start_freq_hz + minimum_span_hz)
            start_freq_hz = max(lower_bound_hz, stop_freq_hz - minimum_span_hz)

        self.config.display_start_freq_hz = int(start_freq_hz)
        self.config.display_stop_freq_hz = int(stop_freq_hz)
        self.config.center_freq_hz = (self.config.display_start_freq_hz + self.config.display_stop_freq_hz) // 2
        self.config.display_span_hz = self.config.display_stop_freq_hz - self.config.display_start_freq_hz
        self.config.use_start_stop_freq = use_start_stop_display

    @staticmethod
    def _clamp_float(value: float, minimum: float, maximum: float) -> float:
        return min(max(value, minimum), maximum)

    @staticmethod
    def _clamp_int(value: int, minimum: int, maximum: int) -> int:
        return min(max(value, minimum), maximum)

    def _show_center_frequency_dialog(self) -> float | None:
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Center")

        layout = QtWidgets.QVBoxLayout(dialog)

        form_layout = QtWidgets.QFormLayout()
        spin_box = QtWidgets.QDoubleSpinBox(dialog)
        spin_box.setRange(UNBOUNDED_DOUBLE_MIN, UNBOUNDED_DOUBLE_MAX)
        spin_box.setDecimals(3)
        spin_box.setSingleStep(self.config.center_freq_step_mhz)
        spin_box.setValue(self.config.center_freq_hz / 1e6)
        spin_box.setSuffix(" MHz")
        form_layout.addRow("Center Frequency", spin_box)
        layout.addLayout(form_layout)

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            parent=dialog,
        )
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return None

        return spin_box.value()

    def _on_freq_channel_clicked(self) -> None:
        value = self._show_center_frequency_dialog()
        if value is None:
            return

        center_freq_hz = int(round(value * 1e6))
        self._apply_center_frequency_change(center_freq_hz)

    def _on_cf_step_clicked(self) -> None:
        value, accepted = QtWidgets.QInputDialog.getDouble(
            self,
            "CF Step",
            "Center Step [MHz]",
            value=self.config.center_freq_step_mhz,
            min=UNBOUNDED_DOUBLE_MIN,
            max=UNBOUNDED_DOUBLE_MAX,
            decimals=3,
        )
        if not accepted:
            return

        self.config.center_freq_step_mhz = self._clamp_float(
            value,
            MIN_CENTER_FREQ_STEP_MHZ,
            MAX_CENTER_FREQ_STEP_MHZ,
        )
        self._refresh_status_label()

    def _nudge_center_frequency(self, direction: int) -> None:
        step_hz = int(round(self.config.center_freq_step_mhz * 1e6))
        if step_hz <= 0:
            return
        center_freq_hz = self.config.center_freq_hz + (direction * step_hz)
        self._apply_center_frequency_change(center_freq_hz)

    def _apply_center_frequency_change(self, center_freq_hz: int) -> None:
        half_span_hz = int(round(self.config.display_span_hz / 2.0))
        self._apply_frequency_window(
            int(center_freq_hz) - half_span_hz,
            int(center_freq_hz) + half_span_hz,
            use_start_stop_display=False,
        )
        if self.config.center_freq_hz <= 0:
            return

        if self.config.analyzer_mode in (AnalyzerMode.SWEEP_SA, AnalyzerMode.WIDEBAND_REALTIME_SA):
            previous_sweep_state = self._current_sweep_state()
            if self._is_wideband_mode():
                self._invalidate_wideband_runtime()
            self._reset_sweep_display_and_restore_state(previous_sweep_state)
        elif self._is_time_analyzer_mode():
            self.receiver.retune_lo(self.config.center_freq_hz)
            self._reset_plot_state()
            if self.sweep_state == SWEEP_STATE_RUNNING:
                self._start_time_analyzer_continuous()
        else:
            self.receiver.retune_lo(self.config.center_freq_hz)
            self.processor.update_center_frequency(self.config.center_freq_hz)
            self._apply_center_frequency_update()
        self._refresh_status_label()

    def _on_freq_start_stop_clicked(self) -> None:
        if self._is_time_analyzer_mode():
            QtWidgets.QMessageBox.information(
                self,
                "Start/Stop",
                "Time Analyzer mode does not use Start/Stop.",
            )
            return
        previous_sweep_state = self._current_sweep_state()
        if (
            self.config.use_start_stop_freq
            and self.config.display_start_freq_hz is not None
            and self.config.display_stop_freq_hz is not None
        ):
            current_start_mhz = self.config.display_start_freq_hz / 1e6
            current_stop_mhz = self.config.display_stop_freq_hz / 1e6
        else:
            current_start_mhz = (
                self.config.center_freq_hz - self.config.display_span_hz // 2
            ) / 1e6
            current_stop_mhz = (
                self.config.center_freq_hz + self.config.display_span_hz // 2
            ) / 1e6

        start_value, accepted = QtWidgets.QInputDialog.getDouble(
            self,
            "Start/Stop",
            "Start Frequency [MHz]",
            value=current_start_mhz,
            min=UNBOUNDED_DOUBLE_MIN,
            max=UNBOUNDED_DOUBLE_MAX,
            decimals=3,
        )
        if not accepted:
            return
        start_value = self._clamp_float(
            start_value,
            (MIN_WIDEBAND_START_HZ / 1e6) if self._is_wideband_mode() else PLUTO_MIN_CENTER_FREQ_MHZ,
            (MAX_WIDEBAND_STOP_HZ / 1e6) if self._is_wideband_mode() else PLUTO_MAX_CENTER_FREQ_MHZ,
        )

        stop_value, accepted = QtWidgets.QInputDialog.getDouble(
            self,
            "Start/Stop",
            "Stop Frequency [MHz]",
            value=current_stop_mhz,
            min=UNBOUNDED_DOUBLE_MIN,
            max=UNBOUNDED_DOUBLE_MAX,
            decimals=3,
        )
        if not accepted:
            return
        stop_value = self._clamp_float(
            stop_value,
            (MIN_WIDEBAND_START_HZ / 1e6) if self._is_wideband_mode() else PLUTO_MIN_CENTER_FREQ_MHZ,
            (MAX_WIDEBAND_STOP_HZ / 1e6) if self._is_wideband_mode() else PLUTO_MAX_CENTER_FREQ_MHZ,
        )

        if stop_value <= start_value:
            QtWidgets.QMessageBox.warning(
                self,
                "Start/Stop",
                "Stop frequency must be greater than start frequency.",
            )
            return

        start_freq_hz = int(round(start_value * 1e6))
        stop_freq_hz = int(round(stop_value * 1e6))
        if self.config.analyzer_mode == AnalyzerMode.REALTIME_SA and stop_freq_hz - start_freq_hz > MAX_DISPLAY_SPAN_HZ:
            QtWidgets.QMessageBox.warning(
                self,
                "Start/Stop",
                f"Display span must be {MAX_DISPLAY_SPAN_HZ / 1e6:.1f} MHz or less.",
            )
            return
        self._apply_frequency_window(start_freq_hz, stop_freq_hz, use_start_stop_display=True)
        if self.config.analyzer_mode in (AnalyzerMode.SWEEP_SA, AnalyzerMode.WIDEBAND_REALTIME_SA):
            if self._is_wideband_mode():
                self._invalidate_wideband_runtime()
            self._reset_sweep_display_and_restore_state(previous_sweep_state)
        else:
            self.receiver.reconfigure_span(self.config)
            self.receiver.retune_lo(self.config.center_freq_hz)
            self.processor.update_span_related(self.config)
            self._apply_span_update()
        self._refresh_status_label()

    def _on_span_x_scale_clicked(self) -> None:
        if self._is_time_analyzer_mode():
            return
        previous_sweep_state = self._current_sweep_state()
        max_span_mhz = (
            MAX_DISPLAY_SPAN_HZ / 1e6
            if self.config.analyzer_mode == AnalyzerMode.REALTIME_SA
            else ((MAX_WIDEBAND_STOP_HZ - MIN_WIDEBAND_START_HZ) / 1e6)
        )
        value, accepted = QtWidgets.QInputDialog.getDouble(
            self,
            "Freq Span",
            "Display Span [MHz]",
            value=self.config.display_span_hz / 1e6,
            min=(MIN_WIDEBAND_SPAN_HZ / 1e6) if self._is_wideband_mode() else MIN_SPAN_MHZ,
            max=max_span_mhz,
            decimals=3,
        )
        if not accepted:
            return

        display_span_hz = int(round(value * 1e6))
        if display_span_hz <= 0:
            return

        half_span_hz = int(round(display_span_hz / 2.0))
        self._apply_frequency_window(
            self.config.center_freq_hz - half_span_hz,
            self.config.center_freq_hz + half_span_hz,
            use_start_stop_display=False,
        )
        if self.config.analyzer_mode in (AnalyzerMode.SWEEP_SA, AnalyzerMode.WIDEBAND_REALTIME_SA):
            if self._is_wideband_mode():
                self._invalidate_wideband_runtime()
            self._reset_sweep_display_and_restore_state(previous_sweep_state)
        else:
            self.receiver.reconfigure_span(self.config)
            self.processor.update_span_related(self.config)
            self._apply_span_update()
        self._refresh_sweep_time_estimate()
        self._refresh_status_label()

    def _on_ref_level_clicked(self) -> None:
        value, accepted = QtWidgets.QInputDialog.getDouble(
            self,
            "Ref Level",
            "Ref Level [dBm]",
            value=self.config.ref_level_dbm,
            min=UNBOUNDED_DOUBLE_MIN,
            max=UNBOUNDED_DOUBLE_MAX,
            decimals=1,
        )
        if not accepted:
            return

        self.config.ref_level_dbm = self._clamp_float(
            value,
            MIN_REF_LEVEL_DBM,
            MAX_REF_LEVEL_DBM,
        )
        self._sync_amplitude_scale_from_config()
        self._apply_display_scale()
        self._refresh_status_label()

    def _on_range_clicked(self) -> None:
        value, accepted = QtWidgets.QInputDialog.getDouble(
            self,
            "Range",
            "Display Range [dB]",
            value=self.config.display_range_db,
            min=UNBOUNDED_DOUBLE_MIN,
            max=UNBOUNDED_DOUBLE_MAX,
            decimals=1,
        )
        if not accepted:
            return

        self.config.display_range_db = self._clamp_float(
            value,
            MIN_DISPLAY_RANGE_DB,
            MAX_DISPLAY_RANGE_DB,
        )
        self._sync_amplitude_scale_from_config()
        self._apply_display_scale()
        self._refresh_status_label()

    def _on_int_gain_clicked(self) -> None:
        value, accepted = QtWidgets.QInputDialog.getInt(
            self,
            "Int Gain",
            "Internal Gain [dB]",
            value=self.config.rx_gain_db,
            min=MIN_INTERNAL_GAIN_DB,
            max=MAX_INTERNAL_GAIN_DB,
        )
        if not accepted:
            return

        self.config.rx_gain_db = value
        self.config.__post_init__()
        self.receiver.set_gain_db(self.config.rx_gain_db)
        self._refresh_status_label()

    def _on_ext_att_clicked(self) -> None:
        value, accepted = QtWidgets.QInputDialog.getDouble(
            self,
            "Ext ATT",
            "External ATT [dB]",
            value=self.config.ext_att_db,
            min=UNBOUNDED_DOUBLE_MIN,
            max=UNBOUNDED_DOUBLE_MAX,
            decimals=1,
        )
        if not accepted:
            return

        self.config.ext_att_db = self._clamp_float(
            value,
            MIN_EXT_ATT_DB,
            MAX_EXT_ATT_DB,
        )
        self._refresh_status_label()

    def _on_ext_gain_clicked(self) -> None:
        value, accepted = QtWidgets.QInputDialog.getDouble(
            self,
            "Ext Gain",
            "External Gain [dB]",
            value=self.config.ext_gain_db,
            min=UNBOUNDED_DOUBLE_MIN,
            max=UNBOUNDED_DOUBLE_MAX,
            decimals=1,
        )
        if not accepted:
            return

        self.config.ext_gain_db = self._clamp_float(
            value,
            MIN_EXT_GAIN_DB,
            MAX_EXT_GAIN_DB,
        )
        self._refresh_status_label()

    def _on_rbw_clicked(self) -> None:
        previous_state = self._current_sweep_state()
        is_wideband_mode = self._is_wideband_mode()
        is_time_analyzer_mode = self._is_time_analyzer_mode()
        is_sweep_mode = self.config.analyzer_mode == AnalyzerMode.SWEEP_SA
        current_value = 0.0 if self.config.rbw_hz is None else self.config.rbw_hz / 1e3
        value, accepted = QtWidgets.QInputDialog.getDouble(
            self,
            "RBW",
            "RBW [kHz] (0 = None)",
            value=current_value,
            min=UNBOUNDED_DOUBLE_MIN,
            max=UNBOUNDED_DOUBLE_MAX,
            decimals=3,
        )
        if not accepted:
            return

        rbw_hz = None if value <= 0.0 else float(value * 1e3)
        if self.config.analyzer_mode in (AnalyzerMode.SWEEP_SA, AnalyzerMode.TIME_ANALYZER):
            rbw_hz = self._clip_sweep_rbw(rbw_hz)
        else:
            rbw_hz = self._clip_realtime_rbw(rbw_hz)
        self.config.rbw_hz = rbw_hz
        if is_time_analyzer_mode:
            self._apply_time_analyzer_rbw_driven_capture_settings()
        elif is_sweep_mode:
            self._apply_sweep_rbw_driven_capture_settings()
            self.receiver.invalidate_sweep_configuration()
            self._reset_sweep_display_and_restore_state(previous_state)
        else:
            self._rebuild_processor_only()
        if is_wideband_mode:
            self.timer.stop()
            self._invalidate_wideband_runtime()
            self._apply_span_update()
            self._restore_sweep_state(previous_state)
        self._refresh_sweep_time_estimate()
        self._refresh_status_label()

    def _on_time_span_clicked(self) -> None:
        if not self._is_time_analyzer_mode():
            return
        previous_state = self._current_sweep_state()
        value, accepted = QtWidgets.QInputDialog.getDouble(
            self,
            "Time Span",
            "Time Span [s]",
            value=float(self.config.time_analyzer_time_span_s),
            min=UNBOUNDED_DOUBLE_MIN,
            max=UNBOUNDED_DOUBLE_MAX,
            decimals=3,
        )
        if not accepted:
            return

        self.config.time_analyzer_time_span_s = self._clamp_float(
            float(value),
            MIN_TIME_ANALYZER_TIME_SPAN_S,
            MAX_TIME_ANALYZER_TIME_SPAN_S,
        )
        self._reset_plot_state()
        if previous_state == SWEEP_STATE_RUNNING:
            self._start_time_analyzer_continuous()
        elif previous_state == SWEEP_STATE_SINGLE:
            self.sweep_state = SWEEP_STATE_STOPPED
            self.timer.stop()
            self._update_continuous_button()
        self._refresh_status_label()

    def _clip_sweep_rbw(self, rbw_hz: float | None) -> float:
        if rbw_hz is None or rbw_hz < MIN_SWEEP_RBW_HZ:
            return MIN_SWEEP_RBW_HZ
        return min(rbw_hz, MAX_SWEEP_RBW_HZ)

    def _clip_realtime_rbw(self, rbw_hz: float | None) -> float | None:
        if rbw_hz is None:
            return None
        return min(max(rbw_hz, 0.0), MAX_REALTIME_RBW_HZ)

    def _select_fft_size(self, fft_size: int) -> None:
        if self._is_time_analyzer_mode():
            self._refresh_status_label()
            return
        previous_state = self._current_sweep_state()
        is_wideband_mode = self._is_wideband_mode()
        if self.config.analyzer_mode in (
            AnalyzerMode.REALTIME_SA,
            AnalyzerMode.WIDEBAND_REALTIME_SA,
        ):
            fft_size = min(int(fft_size), MAX_REALTIME_FFT_SIZE)
        should_resume_wideband = (
            is_wideband_mode and previous_state == SWEEP_STATE_RUNNING and self.timer.isActive()
        )
        if is_wideband_mode:
            self.timer.stop()
        self.config.fft_size = int(fft_size)
        if is_wideband_mode:
            self._invalidate_wideband_runtime()
        self.receiver.reconfigure_span(self.config)
        self.processor.update_span_related(self.config)
        self._apply_span_update()
        self._update_realtime_sa_controls()
        self._refresh_sweep_time_estimate()
        self._refresh_status_label()
        if is_wideband_mode:
            if should_resume_wideband:
                self._start_wideband_continuous()
            else:
                self.sweep_state = SWEEP_STATE_STOPPED
                self._update_continuous_button()

    def _on_sweep_time_clicked(self) -> None:
        previous_state = self._current_sweep_state()
        value, accepted = QtWidgets.QInputDialog.getDouble(
            self,
            "Swp Time",
            "Sweep Time [ms]",
            value=self.config.sweep_time_ms,
            min=UNBOUNDED_DOUBLE_MIN,
            max=UNBOUNDED_DOUBLE_MAX,
            decimals=3,
        )
        if not accepted:
            return

        self.config.sweep_time_ms = self._clamp_float(
            float(value),
            MIN_SWEEP_TIME_MS,
            MAX_SWEEP_TIME_MS,
        )
        self._refresh_sweep_time_estimate()
        self._update_sweep_controls()
        self._refresh_status_label()
        self._restore_sweep_state(previous_state)

    def _on_sweep_points_clicked(self) -> None:
        previous_state = self._current_sweep_state()
        value, accepted = QtWidgets.QInputDialog.getInt(
            self,
            "Swp Pts",
            "Sweep Points",
            value=self.config.sweep_points,
            min=UNBOUNDED_INT_MIN,
            max=UNBOUNDED_INT_MAX,
        )
        if not accepted:
            return

        self.config.sweep_points = self._clamp_int(
            int(value),
            MIN_SWEEP_POINTS,
            MAX_SWEEP_POINTS,
        )
        self.sweep_controller.reset()
        self._reset_plot_state()
        self._refresh_sweep_time_estimate()
        self._update_sweep_controls()
        self._refresh_status_label()
        self._restore_sweep_state(previous_state)

    def _select_sweep_detector(self, detector_mode: DetectorMode | str) -> None:
        resolved_mode = DetectorMode(detector_mode)
        if self.config.sweep_detector_mode == resolved_mode.value:
            return

        self.config.sweep_detector_mode = resolved_mode.value
        self._update_sweep_controls()
        self._update_sweep_detector_selection_page()
        self._refresh_status_label()

    def _on_history_clicked(self) -> None:
        value, accepted = QtWidgets.QInputDialog.getInt(
            self,
            "History",
            "Waterfall History",
            value=self.config.waterfall_history,
            min=UNBOUNDED_INT_MIN,
            max=UNBOUNDED_INT_MAX,
        )
        if not accepted:
            return

        self.config.waterfall_history = self._clamp_int(
            int(value),
            MIN_WATERFALL_HISTORY,
            MAX_WATERFALL_HISTORY,
        )
        self._reset_plot_state()
        self._refresh_status_label()

    def _on_cont_clicked(self) -> None:
        if self.config.analyzer_mode == AnalyzerMode.SWEEP_SA:
            self._start_sweep_continuous()
            return
        if self._is_wideband_mode():
            self._start_wideband_continuous()
            return
        if self._is_time_analyzer_mode():
            self._start_time_analyzer_continuous()
            return

        self.sweep_state = SWEEP_STATE_RUNNING
        self.receiver.start()
        self._restart_timer_for_current_mode()
        self._update_continuous_button()

    def _on_single_clicked(self) -> None:
        previous_state = self.sweep_state
        self.sweep_state = SWEEP_STATE_SINGLE
        if self.config.analyzer_mode == AnalyzerMode.SWEEP_SA:
            self.receiver.stop()
            self.sweep_controller.request_single()
        elif self._is_wideband_mode():
            self.receiver.stop()
            self._invalidate_wideband_runtime()
        elif self._is_time_analyzer_mode():
            self.receiver.stop()
            self._reset_time_analyzer_time_window(start_timestamp=time.perf_counter())
        else:
            self.receiver.start()
        self._restart_timer_for_current_mode()
        self._update_continuous_button()

    def _start_sweep_continuous(self) -> None:
        self.sweep_state = SWEEP_STATE_RUNNING
        self.receiver.stop()
        self.sweep_controller.request_continuous()
        self._restart_timer_for_current_mode()
        self._update_continuous_button()

    def _on_reset_clicked(self) -> None:
        previous_state = self._current_sweep_state()
        self._reset_all_measurement_state(
            stop_receiver=False,
            stop_sweep=True,
            reset_markers=False,
        )
        self._clear_persistence()
        if self.config.analyzer_mode in (AnalyzerMode.SWEEP_SA, AnalyzerMode.WIDEBAND_REALTIME_SA):
            if self.config.analyzer_mode == AnalyzerMode.SWEEP_SA:
                self._sweep_like_suppress_progress_until_first_complete = True
            self._restore_sweep_state(previous_state)
        elif self._is_time_analyzer_mode():
            self._sweep_like_suppress_progress_until_first_complete = True
            if previous_state == SWEEP_STATE_RUNNING:
                self._start_time_analyzer_continuous()
            elif previous_state == SWEEP_STATE_SINGLE:
                self.sweep_state = SWEEP_STATE_STOPPED
                self.timer.stop()
                self._update_continuous_button()
        self._update_continuous_button()
        self._refresh_status_label()

    def _select_graph_view(self, mode: str) -> None:
        freeze_overlays = self._freeze_mode_switch_panels()
        self._set_left_display_updates_enabled(False)
        try:
            self.graph_view_mode = mode
            self._update_graph_view_controls()
            self._apply_display_mode()
        finally:
            self._set_left_display_updates_enabled(True)
            self._thaw_mode_switch_panels(freeze_overlays)

    def _on_trace_visible_clicked(self, trace_index: int) -> None:
        trace_state = self._trace_state(trace_index)
        trace_state.is_visible = not trace_state.is_visible
        self._update_trace_control_state(trace_index)
        self._set_trace_curve_visibility(trace_index)
        self._update_all_marker_control_states()
        self._update_marker_items()

    def _select_trace_type(self, trace_index: int, trace_type: str) -> None:
        trace_state = self._trace_state(trace_index)
        trace_state.trace_type = trace_type
        trace_state.max_hold_power = None
        trace_state.average_power = None
        self._update_trace_control_state(trace_index)
        self._update_trace_type_selection_page(trace_index)

    def _on_trace_hold_clicked(self, trace_index: int) -> None:
        trace_state = self._trace_state(trace_index)
        trace_state.hold_enabled = not trace_state.hold_enabled
        self._update_trace_control_state(trace_index)
        self._set_trace_curve_visibility(trace_index)

    def _on_trace_average_count_clicked(self, trace_index: int) -> None:
        trace_state = self._trace_state(trace_index)
        value, accepted = QtWidgets.QInputDialog.getInt(
            self,
            trace_state.name,
            "Average Count",
            value=trace_state.average_count,
            min=UNBOUNDED_INT_MIN,
            max=UNBOUNDED_INT_MAX,
        )
        if not accepted:
            return

        trace_state.average_count = self._clamp_int(
            int(value),
            MIN_TRACE_AVERAGE_COUNT,
            MAX_TRACE_AVERAGE_COUNT,
        )
        trace_state.average_power = None
        self._update_trace_control_state(trace_index)

    def _marker_state(self, marker_index: int) -> MarkerState:
        return self.marker_states[marker_index]

    def _trace_state(self, trace_index: int) -> TraceState:
        return self.trace_states[trace_index]

    def _trace_state_by_name(self, trace_name: str) -> TraceState | None:
        for trace_state in self.trace_states:
            if trace_state.name == trace_name:
                return trace_state
        return None

    def _make_trace_pen(self, trace_state: TraceState) -> pg.Pen:
        color = QtGui.QColor(trace_state.color_hex)
        if trace_state.hold_enabled:
            color = color.darker(165)
        return pg.mkPen(color, width=1.5)

    def _set_trace_curve_visibility(self, trace_index: int) -> None:
        trace_state = self._trace_state(trace_index)
        curve = self.spectrum_curves[trace_index]
        curve.setPen(self._make_trace_pen(trace_state))
        curve.setVisible(trace_state.is_visible)

    def _update_trace_control_state(self, trace_index: int) -> None:
        if trace_index >= len(self.trace_controls):
            return

        trace_state = self._trace_state(trace_index)
        controls = self.trace_controls[trace_index]
        controls["visible"].setText(
            "Trace ON" if trace_state.is_visible else "Trace OFF"
        )
        controls["type"].setText(f"Type: {trace_state.trace_type}")
        controls["hold"].setText(
            "Hold ON" if trace_state.hold_enabled else "Hold OFF"
        )
        controls["average_count"].setText(
            f"Average Count: {trace_state.average_count}"
        )
        controls["average_count"].setEnabled(trace_state.trace_type == TRACE_TYPE_AVERAGE)
        self._update_trace_menu_button(trace_index)

    def _update_trace_menu_button(self, trace_index: int) -> None:
        if trace_index >= len(self.trace_menu_buttons):
            return
        trace_state = self._trace_state(trace_index)
        prefix = SELECTED_BUTTON_PREFIX if trace_state.is_visible else ""
        self.trace_menu_buttons[trace_index].setText(f"{prefix}{trace_state.name}")

    def _update_trace_menu_buttons(self) -> None:
        for trace_index in range(len(self.trace_states)):
            self._update_trace_menu_button(trace_index)

    def _update_graph_view_controls(self) -> None:
        if self.config.analyzer_mode in (AnalyzerMode.SWEEP_SA, AnalyzerMode.TIME_ANALYZER):
            self.graph_view_button.setText("Graph View\nSpectrum Only")
            self.graph_view_button.setEnabled(False)
        else:
            self.graph_view_button.setText(
                f"Graph View\n{GRAPH_VIEW_LABELS[self.graph_view_mode]}"
            )
            self.graph_view_button.setEnabled(True)
        self._update_graph_view_selection_page()

    def _update_graph_view_selection_page(self) -> None:
        for mode, button in self.graph_view_option_buttons.items():
            prefix = (
                SELECTED_BUTTON_PREFIX
                if mode == self.graph_view_mode
                else UNSELECTED_BUTTON_PREFIX
            )
            button.setText(f"{prefix}{GRAPH_VIEW_LABELS[mode]}")
            if self.config.analyzer_mode == AnalyzerMode.SWEEP_SA:
                button.setEnabled(mode == GRAPH_VIEW_SPECTRUM_ONLY)
            else:
                button.setEnabled(True)

    def _update_persistence_controls(self) -> None:
        persistence_supported = self.config.analyzer_mode in (
            AnalyzerMode.REALTIME_SA,
            AnalyzerMode.WIDEBAND_REALTIME_SA,
        )
        if not persistence_supported:
            self.persistence_button.setText("Persistence OFF")
            self.persistence_button.setEnabled(False)
        else:
            state_label = "ON" if self.persistence_enabled else "OFF"
            self.persistence_button.setText(f"Persistence {state_label}")
            self.persistence_button.setEnabled(True)
        self._apply_persistence_visibility()

    def _toggle_persistence_enabled(self) -> None:
        if self.config.analyzer_mode not in (
            AnalyzerMode.REALTIME_SA,
            AnalyzerMode.WIDEBAND_REALTIME_SA,
        ):
            return
        self.persistence_enabled = not self.persistence_enabled
        if self.persistence_enabled:
            self._initialize_persistence_buffer()
        else:
            self._apply_persistence_visibility()
        self._update_persistence_controls()

    def _current_persistence_decay(self) -> float:
        return PERSISTENCE_DECAY_VALUES.get(
            self.config.persistence_decay_mode,
            PERSISTENCE_DECAY_VALUES["Medium"],
        )

    def _update_persistence_decay_selection_page(self) -> None:
        for decay_mode, button in self.persistence_decay_option_buttons.items():
            prefix = (
                SELECTED_BUTTON_PREFIX
                if decay_mode == self.config.persistence_decay_mode
                else UNSELECTED_BUTTON_PREFIX
            )
            button.setText(f"{prefix}{decay_mode}")
            button.setEnabled(
                self.config.analyzer_mode
                in (AnalyzerMode.REALTIME_SA, AnalyzerMode.WIDEBAND_REALTIME_SA)
            )

    def _select_persistence_decay_mode(self, decay_mode: str) -> None:
        if self.config.analyzer_mode not in (
            AnalyzerMode.REALTIME_SA,
            AnalyzerMode.WIDEBAND_REALTIME_SA,
        ):
            return
        if decay_mode not in PERSISTENCE_DECAY_VALUES:
            return
        self.config.persistence_decay_mode = decay_mode
        self._update_realtime_sa_controls()

    def _build_persistence_lut(self) -> np.ndarray:
        color_map = pg.ColorMap(
            pos=np.array([0.0, 0.2, 0.45, 0.7, 0.88, 1.0]),
            color=np.array(
                [
                    [0, 0, 0, 0],
                    [0, 0, 120, 110],
                    [0, 180, 255, 150],
                    [0, 255, 96, 180],
                    [255, 220, 0, 220],
                    [255, 48, 0, 255],
                ],
                dtype=np.ubyte,
            ),
        )
        return color_map.getLookupTable(nPts=256, alpha=True)

    def _initialize_persistence_buffer(self) -> None:
        axis = self._get_active_spectrum_freq_axis_ghz()
        width = len(axis) if axis is not None else 0
        self.persistence_histogram = np.zeros(
            (PERSISTENCE_AMPLITUDE_BINS, width),
            dtype=np.float32,
        )
        self._update_persistence_image()
        self._update_persistence_rect()

    def _clear_persistence(self) -> None:
        self._initialize_persistence_buffer()

    def _update_persistence_rect(self) -> None:
        if self.persistence_image is None or self._last_display_freq_axis_ghz is None:
            return
        if (
            self.persistence_image.image is None
            or self.persistence_image.width() is None
            or self.persistence_image.height() is None
        ):
            return
        if len(self._last_display_freq_axis_ghz) == 0:
            return
        x_min = float(self._last_display_freq_axis_ghz[0])
        x_max = float(self._last_display_freq_axis_ghz[-1])
        self.persistence_image.setRect(
            QtCore.QRectF(
                x_min,
                self.y_min,
                x_max - x_min,
                self.y_max - self.y_min,
            )
        )

    def _update_persistence_image(self) -> None:
        if self.persistence_image is None:
            return
        display_image = np.log1p(self.persistence_histogram)
        self.persistence_image.setImage(
            display_image,
            autoLevels=False,
            axisOrder="row-major",
        )
        if display_image.size:
            high_level = float(np.percentile(display_image, 99.5))
            max_level = float(np.max(display_image))
            high_level = min(max_level, max(1.5, high_level))
            low_level = 0.15
            if high_level <= low_level:
                high_level = low_level + 1.0
        else:
            low_level = 0.0
            high_level = 1.0
        self.persistence_image.setLevels((low_level, high_level))
        self._update_persistence_rect()
        self._apply_persistence_visibility()

    def _apply_persistence_visibility(self) -> None:
        if self.persistence_image is None:
            return
        visible = self.persistence_enabled and self.config.analyzer_mode in (
            AnalyzerMode.REALTIME_SA,
            AnalyzerMode.WIDEBAND_REALTIME_SA,
        )
        self.persistence_image.setVisible(visible)

    def _accumulate_persistence(self, current_display_db: np.ndarray) -> None:
        if (
            not self.persistence_enabled
            or self.config.analyzer_mode
            not in (AnalyzerMode.REALTIME_SA, AnalyzerMode.WIDEBAND_REALTIME_SA)
            or self.persistence_image is None
            or self._last_display_freq_axis_ghz is None
        ):
            return

        if self.persistence_histogram.shape[1] != len(current_display_db):
            self._initialize_persistence_buffer()
        if self.persistence_histogram.shape[1] != len(current_display_db):
            return

        self.persistence_histogram *= self._current_persistence_decay()

        valid_mask = np.isfinite(current_display_db)
        if not np.any(valid_mask):
            self._update_persistence_image()
            return

        clipped_db = np.clip(current_display_db[valid_mask], self.y_min, self.y_max)
        y_span = max(1e-9, self.y_max - self.y_min)
        normalized = (clipped_db - self.y_min) / y_span
        row_indices = np.clip(
            np.rint(normalized * (PERSISTENCE_AMPLITUDE_BINS - 1)).astype(np.int32),
            0,
            PERSISTENCE_AMPLITUDE_BINS - 1,
        )
        column_indices = np.flatnonzero(valid_mask)
        np.add.at(
            self.persistence_histogram,
            (row_indices, column_indices),
            PERSISTENCE_HIT_INCREMENT,
        )
        self._update_persistence_image()

    def _update_analyzer_mode_controls(self) -> None:
        self.analyzer_mode_button.setText(
            f"Analyzer Mode\n{self.config.analyzer_mode.value}"
        )
        self.analyzer_mode_button.setMinimumHeight(68)
        self._update_analyzer_mode_selection_page()

    def _update_analyzer_mode_selection_page(self) -> None:
        for mode, button in self.analyzer_mode_option_buttons.items():
            prefix = (
                SELECTED_BUTTON_PREFIX
                if mode == self.config.analyzer_mode
                else UNSELECTED_BUTTON_PREFIX
            )
            button.setText(f"{prefix}{mode.value}")

    def _update_trace_type_selection_page(self, trace_index: int) -> None:
        if trace_index >= len(self.trace_type_option_buttons):
            return

        trace_state = self._trace_state(trace_index)
        option_buttons = self.trace_type_option_buttons[trace_index]
        for trace_type, button in option_buttons.items():
            prefix = (
                SELECTED_BUTTON_PREFIX
                if trace_type == trace_state.trace_type
                else UNSELECTED_BUTTON_PREFIX
            )
            button.setText(f"{prefix}{trace_type}")

    def _update_realtime_sa_controls(self) -> None:
        self.fft_size_button.setText(f"FFT size: {self.config.fft_size}")
        self.fft_size_button.setEnabled(not self._is_time_analyzer_mode())
        if self.config.analyzer_mode not in (
            AnalyzerMode.REALTIME_SA,
            AnalyzerMode.WIDEBAND_REALTIME_SA,
        ):
            self.persistence_decay_button.setText(
                f"Persistence Decay\n{self.config.persistence_decay_mode}"
            )
            self.persistence_decay_button.setEnabled(False)
        else:
            self.persistence_decay_button.setText(
                f"Persistence Decay\n{self.config.persistence_decay_mode}"
            )
            self.persistence_decay_button.setEnabled(True)
        self._update_fft_size_selection_page()
        self._update_persistence_decay_selection_page()
        self._update_span_x_scale_controls()

    def _update_span_x_scale_controls(self) -> None:
        if not hasattr(self, "freq_span_button") or not hasattr(self, "time_span_button"):
            return
        is_time_analyzer = self._is_time_analyzer_mode()
        self.freq_span_button.setEnabled(not is_time_analyzer)
        if is_time_analyzer:
            self.time_span_button.setEnabled(True)
            self.time_span_button.setText(
                f"Time Span\n{float(self.config.time_analyzer_time_span_s):.3f} s"
            )
        else:
            self.time_span_button.setEnabled(False)
            self.time_span_button.setText("Time Span")

    def _update_sweep_controls(self) -> None:
        self.sweep_time_button.setText(f"Swp Time\n{self.config.sweep_time_ms:.0f} ms")
        self.sweep_points_button.setText(f"Swp Pts\n{self.config.sweep_points}")

    def _update_sweep_detector_selection_page(self) -> None:
        for detector_mode, button in self.sweep_detector_option_buttons.items():
            prefix = "笨・" if detector_mode.value == self.config.sweep_detector_mode else "  "
            button.setText(f"{prefix}{detector_mode.value}")

    def _update_continuous_button(self) -> None:
        label = "Continuous"
        if self.sweep_state == SWEEP_STATE_RUNNING:
            label = f"{SELECTED_BUTTON_PREFIX}{label}"
        self.cont_button.setText(label)

    def _update_sweep_detector_selection_page(self) -> None:
        for detector_mode, button in self.sweep_detector_option_buttons.items():
            prefix = "> " if detector_mode.value == self.config.sweep_detector_mode else "  "
            button.setText(f"{prefix}{detector_mode.value}")

    def _refresh_sweep_time_estimate(self) -> None:
        self.actual_sweep_time_s = self.sweep_controller.get_actual_sweep_time_s()
        if hasattr(self, "sweep_time_button") and hasattr(self, "sweep_points_button"):
            self._update_sweep_controls()

    def _update_fft_size_selection_page(self) -> None:
        current_fft_size = str(self.config.fft_size)
        for fft_size, button in self.fft_size_option_buttons.items():
            prefix = (
                SELECTED_BUTTON_PREFIX
                if fft_size == current_fft_size
                else UNSELECTED_BUTTON_PREFIX
            )
            button.setText(f"{prefix}{fft_size}")

    def _reset_trace_runtime_buffers(self) -> None:
        for trace_state in self.trace_states:
            trace_state.max_hold_power = None
            trace_state.average_power = None

    def _get_active_spectrum_freq_axis_ghz(self) -> np.ndarray:
        if self._is_time_analyzer_mode():
            if self._time_analyzer_time_axis_s is not None:
                return self._time_analyzer_time_axis_s
            return np.array([0.0, 1.0], dtype=float)
        if self.config.analyzer_mode == AnalyzerMode.SWEEP_SA:
            if self._last_display_freq_axis_ghz is not None:
                return self._last_display_freq_axis_ghz
            return self.sweep_controller.get_sweep_frequency_axis_hz() / 1e9
        if self._is_wideband_mode():
            if self._last_display_freq_axis_ghz is not None:
                return self._last_display_freq_axis_ghz
            start_hz, stop_hz = self._get_wideband_start_stop_hz()
            return np.array([start_hz / 1e9, stop_hz / 1e9], dtype=float)

        return self.processor.get_display_freq_axis_ghz()

    def _update_trace_curves(self) -> None:
        if self._last_display_freq_axis_ghz is None:
            return

        for index, trace_state in enumerate(self.trace_states):
            if trace_state.display_db is None:
                trace_state.display_db = np.full(
                    len(self._last_display_freq_axis_ghz),
                    self.y_min,
                    dtype=float,
                )
            self.spectrum_curves[index].setData(
                self._last_display_freq_axis_ghz,
                trace_state.display_db,
            )
            self._set_trace_curve_visibility(index)

    def _active_sweep_progress_trace_state(self) -> TraceState | None:
        for trace_state in self.trace_states:
            if (
                trace_state.is_visible
                and not trace_state.hold_enabled
                and trace_state.trace_type == TRACE_TYPE_LIVE
            ):
                return trace_state
        return None

    def _update_sweep_like_write_index(
        self,
        current_index: int,
        point_count: int,
        *,
        is_single: bool,
    ) -> tuple[int, bool]:
        if point_count <= 0:
            return 0, True
        reached_right_edge = current_index >= (point_count - 1)
        if is_single:
            next_index = current_index if reached_right_edge else current_index + 1
        else:
            next_index = 0 if reached_right_edge else current_index + 1
        return int(next_index), bool(reached_right_edge)

    def _finish_single_sweep_like(self) -> None:
        self.timer.stop()
        self.sweep_state = SWEEP_STATE_STOPPED
        self._update_continuous_button()

    def _set_sweep_like_progress_symbol(
        self,
        item: pg.ScatterPlotItem | None,
        point_index: int | None,
    ) -> None:
        if item is None or self._last_display_freq_axis_ghz is None or point_index is None:
            if item is not None:
                item.setData([], [])
                item.setVisible(False)
            return

        trace_state = self._active_sweep_progress_trace_state()
        if trace_state is None or trace_state.display_db is None:
            item.setData([], [])
            item.setVisible(False)
            return

        point_index = int(point_index)
        if point_index < 0 or point_index >= len(self._last_display_freq_axis_ghz):
            item.setData([], [])
            item.setVisible(False)
            return
        if point_index >= len(trace_state.display_db):
            item.setData([], [])
            item.setVisible(False)
            return

        marker_y = float(trace_state.display_db[point_index])
        if not np.isfinite(marker_y):
            item.setData([], [])
            item.setVisible(False)
            return

        color = QtGui.QColor(trace_state.color_hex)
        item.setPen(pg.mkPen(color))
        item.setBrush(pg.mkBrush(color))
        item.setData(
            [self._last_display_freq_axis_ghz[point_index]],
            [marker_y],
        )
        item.setVisible(True)

    def _hide_sweep_progress_symbol(self) -> None:
        self._set_sweep_like_progress_symbol(self.sweep_progress_item, None)

    def _hide_time_analyzer_progress_symbol(self) -> None:
        self._set_sweep_like_progress_symbol(self.time_analyzer_progress_item, None)

    def _update_time_analyzer_progress_symbol(self, point_index: int | None = None) -> None:
        if (
            self.time_analyzer_progress_item is None
            or not self._is_time_analyzer_mode()
            or self._time_analyzer_sample_elapsed_s is None
            or self._time_analyzer_trace_db is None
        ):
            self._hide_time_analyzer_progress_symbol()
            return

        if point_index is None:
            point_index = max(0, int(self._time_analyzer_write_index) - 1)
        point_index = int(point_index)
        if point_index < 0 or point_index >= len(self._time_analyzer_sample_elapsed_s):
            self._hide_time_analyzer_progress_symbol()
            return
        if point_index >= len(self._time_analyzer_trace_db):
            self._hide_time_analyzer_progress_symbol()
            return

        marker_x = float(self._time_analyzer_sample_elapsed_s[point_index])
        marker_y = float(self._time_analyzer_trace_db[point_index])
        if not np.isfinite(marker_x) or not np.isfinite(marker_y):
            self._hide_time_analyzer_progress_symbol()
            return

        trace_state = self._active_sweep_progress_trace_state()
        if trace_state is None:
            self._hide_time_analyzer_progress_symbol()
            return

        color = QtGui.QColor(trace_state.color_hex)
        self.time_analyzer_progress_item.setPen(pg.mkPen(color))
        self.time_analyzer_progress_item.setBrush(pg.mkBrush(color))
        self.time_analyzer_progress_item.setData([marker_x], [marker_y])
        self.time_analyzer_progress_item.setVisible(True)

    def _update_sweep_progress_symbol(self, frame_result) -> None:
        if (
            self.sweep_progress_item is None
            or self._last_display_freq_axis_ghz is None
            or frame_result.sweep_complete
            or frame_result.active_point_index is None
            or self._sweep_like_suppress_progress_until_first_complete
        ):
            self._hide_sweep_progress_symbol()
            return
        self._set_sweep_like_progress_symbol(
            self.sweep_progress_item,
            int(frame_result.active_point_index),
        )

    def _select_marker_trace_db(self, marker_state: MarkerState) -> np.ndarray | None:
        for trace_state in self.trace_states:
            if trace_state.name == marker_state.trace_name:
                if not trace_state.is_visible:
                    return None
                return trace_state.display_db
        return None

    @staticmethod
    def _clear_marker_sweep_snapshot(marker_state: MarkerState) -> None:
        marker_state.sweep_snapshot_power_db = None

    def _get_marker_peak_source(
        self,
        marker_state: MarkerState,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if self.config.analyzer_mode == AnalyzerMode.SWEEP_SA:
            if self._last_completed_sweep_freq_axis_ghz is None:
                return None
            trace_db = self._last_completed_sweep_trace_db.get(marker_state.trace_name)
            if trace_db is None:
                return None
            return self._last_completed_sweep_freq_axis_ghz, trace_db

        trace_db = self._select_marker_trace_db(marker_state)
        if trace_db is None or self._last_display_freq_axis_ghz is None:
            return None
        return self._last_display_freq_axis_ghz, trace_db

    def _update_marker_control_state(self, marker_index: int) -> None:
        if marker_index >= len(self.marker_controls):
            return

        marker_state = self._marker_state(marker_index)
        controls = self.marker_controls[marker_index]
        selected_trace_state = self._trace_state_by_name(marker_state.trace_name)

        controls["toggle"].setText(
            "Marker ON" if marker_state.is_enabled else "Marker OFF"
        )
        controls["trace"].setText(marker_state.trace_name)
        if selected_trace_state is not None:
            controls["trace"].setStyleSheet(f"color: {selected_trace_state.color_hex};")
        controls["continuous_peak"].setText(
            "Continuous Peak ON"
            if marker_state.continuous_peak_enabled
            else "Continuous Peak OFF"
        )
        if self._is_time_analyzer_mode():
            controls["frequency"].setText(f"Time\n{marker_state.time_sec:.3f} s")
            controls["step"].setText(f"Step\n{marker_state.time_step_sec:.3f} s")
        else:
            controls["frequency"].setText(
                f"Frequency\n{(marker_state.frequency_hz / 1e6):.3f} MHz"
            )
            controls["step"].setText(
                f"Step\n{(marker_state.step_hz / 1e6):.3f} MHz"
            )

        manual_move_enabled = not marker_state.continuous_peak_enabled
        selected_trace_db = self._select_marker_trace_db(marker_state)
        trace_available = selected_trace_db is not None
        controls["frequency"].setEnabled(manual_move_enabled)
        controls["left"].setEnabled(manual_move_enabled)
        controls["right"].setEnabled(manual_move_enabled)
        controls["peak_search"].setEnabled(
            manual_move_enabled and trace_available and (not self._is_time_analyzer_mode())
        )
        controls["continuous_peak"].setEnabled(not self._is_time_analyzer_mode())
        controls["to_center"].setEnabled(not self._is_time_analyzer_mode())
        self._update_marker_menu_button(marker_index)

    def _update_all_marker_control_states(self) -> None:
        for marker_index in range(len(self.marker_states)):
            self._update_marker_control_state(marker_index)

    def _update_marker_menu_button(self, marker_index: int) -> None:
        if marker_index >= len(self.marker_menu_buttons):
            return
        marker_state = self._marker_state(marker_index)
        prefix = SELECTED_BUTTON_PREFIX if marker_state.is_enabled else ""
        self.marker_menu_buttons[marker_index].setText(f"{prefix}{marker_state.name}")

    def _update_marker_menu_buttons(self) -> None:
        for marker_index in range(len(self.marker_states)):
            self._update_marker_menu_button(marker_index)

    def _on_marker_toggle_clicked(self, marker_index: int) -> None:
        marker_state = self._marker_state(marker_index)
        marker_state.is_enabled = not marker_state.is_enabled
        self._update_marker_control_state(marker_index)
        self._update_marker_items()

    def _select_marker_trace(self, marker_index: int, trace_name: str) -> None:
        marker_state = self._marker_state(marker_index)
        marker_state.trace_name = trace_name
        self._clear_marker_sweep_snapshot(marker_state)
        self._update_marker_control_state(marker_index)
        self._update_marker_trace_selection_page(marker_index)
        self._update_marker_items()

    def _update_marker_trace_selection_page(self, marker_index: int) -> None:
        if marker_index >= len(self.marker_trace_option_buttons):
            return

        marker_state = self._marker_state(marker_index)
        option_buttons = self.marker_trace_option_buttons[marker_index]
        for trace_name, button in option_buttons.items():
            prefix = (
                SELECTED_BUTTON_PREFIX
                if trace_name == marker_state.trace_name
                else UNSELECTED_BUTTON_PREFIX
            )
            button.setText(f"{prefix}{trace_name}")

    def _on_marker_frequency_clicked(self, marker_index: int) -> None:
        marker_state = self._marker_state(marker_index)
        if marker_state.continuous_peak_enabled:
            return
        if self._is_time_analyzer_mode():
            start_s, stop_s = self._time_marker_bounds_sec()
            value, accepted = QtWidgets.QInputDialog.getDouble(
                self,
                marker_state.name,
                "Time [s]",
                value=marker_state.time_sec,
                min=UNBOUNDED_DOUBLE_MIN,
                max=UNBOUNDED_DOUBLE_MAX,
                decimals=3,
            )
            if not accepted:
                return
            marker_state.time_sec = self._clamp_float(float(value), start_s, stop_s)
            marker_state.is_enabled = True
            marker_state.continuous_peak_enabled = False
            self._clear_marker_sweep_snapshot(marker_state)
            self._update_marker_control_state(marker_index)
            self._update_marker_items()
            return
        value, accepted = QtWidgets.QInputDialog.getDouble(
            self,
            marker_state.name,
            "Frequency [MHz]",
            value=marker_state.frequency_hz / 1e6,
            min=UNBOUNDED_DOUBLE_MIN,
            max=UNBOUNDED_DOUBLE_MAX,
            decimals=3,
        )
        if not accepted:
            return

        marker_state.frequency_hz = self._clamp_int(
            int(round(value * 1e6)),
            MIN_MARKER_FREQUENCY_HZ,
            MAX_MARKER_FREQUENCY_HZ,
        )
        marker_state.is_enabled = True
        marker_state.continuous_peak_enabled = False
        self._clear_marker_sweep_snapshot(marker_state)
        self._update_marker_control_state(marker_index)
        self._update_marker_items()

    def _on_marker_step_clicked(self, marker_index: int) -> None:
        marker_state = self._marker_state(marker_index)
        if self._is_time_analyzer_mode():
            value, accepted = QtWidgets.QInputDialog.getDouble(
                self,
                marker_state.name,
                "Step [s]",
                value=marker_state.time_step_sec,
                min=UNBOUNDED_DOUBLE_MIN,
                max=UNBOUNDED_DOUBLE_MAX,
                decimals=3,
            )
            if not accepted:
                return
            marker_state.time_step_sec = self._clamp_float(float(value), 0.001, 1_000.0)
            self._update_marker_control_state(marker_index)
            return
        value, accepted = QtWidgets.QInputDialog.getDouble(
            self,
            marker_state.name,
            "Step [kHz]",
            value=marker_state.step_hz / 1e3,
            min=UNBOUNDED_DOUBLE_MIN,
            max=UNBOUNDED_DOUBLE_MAX,
            decimals=3,
        )
        if not accepted:
            return

        marker_state.step_hz = self._clamp_int(
            int(round(value * 1e3)),
            MIN_MARKER_STEP_HZ,
            MAX_MARKER_STEP_HZ,
        )
        self._update_marker_control_state(marker_index)

    def _nudge_marker_frequency(self, marker_index: int, direction: int) -> None:
        marker_state = self._marker_state(marker_index)
        if marker_state.continuous_peak_enabled:
            return
        if self._is_time_analyzer_mode():
            start_s, stop_s = self._time_marker_bounds_sec()
            marker_state.time_sec = self._clamp_float(
                marker_state.time_sec + (direction * marker_state.time_step_sec),
                start_s,
                stop_s,
            )
            marker_state.is_enabled = True
            marker_state.continuous_peak_enabled = False
            self._clear_marker_sweep_snapshot(marker_state)
            self._update_marker_control_state(marker_index)
            self._update_marker_items()
            return
        marker_state.frequency_hz = self._clamp_int(
            marker_state.frequency_hz + direction * marker_state.step_hz,
            MIN_MARKER_FREQUENCY_HZ,
            MAX_MARKER_FREQUENCY_HZ,
        )
        marker_state.is_enabled = True
        marker_state.continuous_peak_enabled = False
        self._clear_marker_sweep_snapshot(marker_state)
        self._update_marker_control_state(marker_index)
        self._update_marker_items()

    def _on_marker_peak_search_clicked(self, marker_index: int) -> None:
        if self._is_time_analyzer_mode():
            return
        marker_state = self._marker_state(marker_index)
        if marker_state.continuous_peak_enabled:
            return
        peak_source = self._get_marker_peak_source(marker_state)
        if peak_source is None:
            return

        peak_result = self._detect_peak_on_axis(*peak_source)
        if peak_result is None:
            return

        peak_freq_ghz, peak_val = peak_result
        marker_state.frequency_hz = int(round(peak_freq_ghz * 1e9))
        if self.config.analyzer_mode == AnalyzerMode.SWEEP_SA:
            marker_state.sweep_snapshot_power_db = float(peak_val)
        else:
            self._clear_marker_sweep_snapshot(marker_state)
        marker_state.is_enabled = True
        marker_state.continuous_peak_enabled = False
        self._update_marker_control_state(marker_index)
        self._update_marker_items()

    def _on_marker_continuous_peak_clicked(self, marker_index: int) -> None:
        if self._is_time_analyzer_mode():
            return
        marker_state = self._marker_state(marker_index)
        marker_state.continuous_peak_enabled = not marker_state.continuous_peak_enabled
        if marker_state.continuous_peak_enabled:
            marker_state.is_enabled = True
        else:
            self._clear_marker_sweep_snapshot(marker_state)
        self._update_marker_control_state(marker_index)
        self._update_marker_items()

    def _on_marker_to_center_clicked(self, marker_index: int) -> None:
        if self._is_time_analyzer_mode():
            return
        marker_state = self._marker_state(marker_index)
        self._apply_center_frequency_change(int(marker_state.frequency_hz))

    def _update_marker_items(self) -> None:
        if self._last_display_freq_axis_ghz is None or self._last_display_power_db is None:
            for marker_item, text_item in self.marker_items:
                marker_item.setData([], [])
                text_item.setText("")
            return

        if len(self._last_display_freq_axis_ghz) == 0:
            return

        for index, marker_state in enumerate(self.marker_states):
            marker_item, text_item = self.marker_items[index]
            if not marker_state.is_enabled:
                marker_item.setData([], [])
                text_item.setText("")
                continue

            trace_db = self._select_marker_trace_db(marker_state)
            if trace_db is None:
                marker_item.setData([], [])
                trace_suffix = marker_state.trace_name.replace("Trace", "Tr")
                text_item.setText(f"M{index + 1} {trace_suffix}")
                continue

            if self._is_time_analyzer_mode():
                marker_x = float(marker_state.time_sec)
                try:
                    trace_index = int(marker_state.trace_name.replace("Trace", "")) - 1
                except ValueError:
                    marker_item.setData([], [])
                    text_item.setText("")
                    continue
                if self._ta_marker_debug_force_trace1:
                    trace_index = 0
                if trace_index < 0 or trace_index >= len(self.spectrum_curves):
                    marker_item.setData([], [])
                    text_item.setText("")
                    continue

                # Use the exact x/y arrays that are actually plotted in TA.
                ta_x_raw, ta_y_raw = self.spectrum_curves[trace_index].getData()
                if ta_x_raw is None or ta_y_raw is None:
                    if self.config.sweep_profile_logging:
                        print(
                            "TAMarker "
                            f"trace_name={marker_state.trace_name} "
                            f"forced_trace=Trace{trace_index + 1} "
                            f"curve_is_visible={self.spectrum_curves[trace_index].isVisible()} "
                            "curve_point_count=0 "
                            f"marker_time_sec={marker_x:.6f} "
                            "x_min=-- x_max=-- interp_y=--"
                        )
                    marker_item.setData([], [])
                    text_item.setText("")
                    continue
                ta_x = np.asarray(ta_x_raw, dtype=float)
                ta_y = np.asarray(ta_y_raw, dtype=float)
                if len(ta_x) == 0 or len(ta_y) == 0:
                    if self.config.sweep_profile_logging:
                        print(
                            "TAMarker "
                            f"trace_name={marker_state.trace_name} "
                            f"forced_trace=Trace{trace_index + 1} "
                            f"curve_is_visible={self.spectrum_curves[trace_index].isVisible()} "
                            "curve_point_count=0 "
                            f"marker_time_sec={marker_x:.6f} "
                            "x_min=-- x_max=-- interp_y=--"
                        )
                    marker_item.setData([], [])
                    text_item.setText("")
                    continue
                if len(ta_x) != len(ta_y):
                    marker_item.setData([], [])
                    text_item.setText("")
                    continue
                valid_mask = np.isfinite(ta_x) & np.isfinite(ta_y)
                if not np.any(valid_mask):
                    marker_item.setData([], [])
                    text_item.setText("")
                    continue
                ta_x_valid = ta_x[valid_mask]
                ta_y_valid = ta_y[valid_mask]
                sort_order = np.argsort(ta_x_valid, kind="stable")
                ta_x_sorted = ta_x_valid[sort_order]
                ta_y_sorted = ta_y_valid[sort_order]
                x_min = float(ta_x_sorted[0])
                x_max = float(ta_x_sorted[-1])
                # If the new sweep has not reached this time yet, hide marker.
                if marker_x < x_min or marker_x > x_max:
                    if self.config.sweep_profile_logging:
                        print(
                            "TAMarker "
                            f"trace_name={marker_state.trace_name} "
                            f"forced_trace=Trace{trace_index + 1} "
                            f"curve_is_visible={self.spectrum_curves[trace_index].isVisible()} "
                            f"curve_point_count={len(ta_x_sorted)} "
                            f"marker_time_sec={marker_x:.6f} "
                            f"x_min={x_min:.6f} x_max={x_max:.6f} interp_y=OUT_OF_RANGE"
                        )
                    marker_item.setData([], [])
                    text_item.setText("")
                    continue
                nearest_index = int(np.argmin(np.abs(ta_x_sorted - marker_x)))
                marker_x = float(ta_x_sorted[nearest_index])
                marker_val = float(ta_y_sorted[nearest_index])
                if self.config.sweep_profile_logging:
                    print(
                        "TAMarker "
                        f"trace_name={marker_state.trace_name} "
                        f"forced_trace=Trace{trace_index + 1} "
                        f"curve_is_visible={self.spectrum_curves[trace_index].isVisible()} "
                        f"curve_point_count={len(ta_x_sorted)} "
                        f"marker_time_sec={marker_x:.6f} "
                        f"x_min={x_min:.6f} x_max={x_max:.6f} "
                        f"interp_y={marker_val:.6f}"
                    )
            else:
                marker_x = marker_state.frequency_hz / 1e9
                nearest_index = int(
                    np.argmin(np.abs(self._last_display_freq_axis_ghz - marker_x))
                )
                marker_x = float(self._last_display_freq_axis_ghz[nearest_index])
                if (
                    self.config.analyzer_mode == AnalyzerMode.SWEEP_SA
                    and marker_state.sweep_snapshot_power_db is not None
                ):
                    marker_val = float(marker_state.sweep_snapshot_power_db)
                else:
                    marker_val = float(trace_db[nearest_index])
            trace_suffix = marker_state.trace_name.replace("Trace", "Tr")

            marker_item.setData([marker_x], [marker_val])
            if self._is_time_analyzer_mode() and self.config.sweep_profile_logging:
                print(
                    "TAMarkerFinal "
                    f"marker_time_sec={marker_x:.6f} marker_y={marker_val:.6f}"
                )
            if self._is_time_analyzer_mode():
                text_item.setText(
                    f"M{index + 1} {trace_suffix}\n"
                    f"{marker_x:.3f} s\n"
                    f"{marker_val:.2f} dBm"
                )
            else:
                text_item.setText(
                    f"M{index + 1} {trace_suffix}\n"
                    f"{marker_x:.6f} GHz\n"
                    f"{marker_val:.2f} dBm"
                )
            text_item.setPos(marker_x, marker_val)

    def _show_not_implemented(self, feature_name: str) -> None:
        QtWidgets.QMessageBox.information(
            self,
            feature_name,
            f"{feature_name} is reserved for a future implementation.",
        )

    def _rebuild_processor_only(self) -> None:
        self.processor = SpectrumProcessor(self.config)
        self._refresh_sweep_time_estimate()

    def _on_sweep_complete(self, frame_result) -> None:
        self._pending_sweep_marker_update = True

    def _detect_peak_on_axis(
        self,
        freq_axis_ghz: np.ndarray,
        power_db: np.ndarray,
    ) -> tuple[float, float] | None:
        if len(freq_axis_ghz) == 0 or len(power_db) == 0:
            return None

        valid_mask = np.isfinite(power_db)
        if not np.any(valid_mask):
            return None

        valid_indices = np.flatnonzero(valid_mask)
        peak_local_index = int(np.argmax(power_db[valid_mask]))
        peak_index = int(valid_indices[peak_local_index])
        return float(freq_axis_ghz[peak_index]), float(power_db[peak_index])

    def _update_markers_for_completed_sweep(self) -> None:
        for marker_state in self.marker_states:
            if marker_state.is_enabled and marker_state.continuous_peak_enabled:
                peak_source = self._get_marker_peak_source(marker_state)
                if peak_source is None:
                    continue
                peak_result = self._detect_peak_on_axis(*peak_source)
                if peak_result is None:
                    continue
                marker_peak_freq_ghz, marker_peak_val = peak_result
                marker_state.frequency_hz = int(round(marker_peak_freq_ghz * 1e9))
                marker_state.sweep_snapshot_power_db = float(marker_peak_val)

        self._update_marker_items()

    def _capture_completed_sweep_snapshot(self) -> None:
        if self._last_display_freq_axis_ghz is None:
            return
        self._last_completed_sweep_freq_axis_ghz = self._last_display_freq_axis_ghz.copy()
        self._last_completed_sweep_trace_db = {}
        for trace_state in self.trace_states:
            if trace_state.display_db is not None:
                self._last_completed_sweep_trace_db[trace_state.name] = trace_state.display_db.copy()

    def _update_traces_from_power_linear(self, power_linear_display: np.ndarray) -> None:
        current_power_db_display = np.full(len(power_linear_display), np.nan, dtype=float)
        valid_mask = np.isfinite(power_linear_display)
        current_power_db_display[valid_mask] = 10.0 * np.log10(
            power_linear_display[valid_mask] + 1e-20
        )
        current_display_db = apply_display_power_correction(
            current_power_db_display,
            self.calibration_offset_db,
            self.config.input_correction_db,
        )
        self._last_current_display_db = current_display_db

        for trace_state in self.trace_states:
            if trace_state.hold_enabled:
                continue

            if trace_state.trace_type == TRACE_TYPE_MAX_HOLD:
                if (
                    trace_state.max_hold_power is None
                    or len(trace_state.max_hold_power) != len(power_linear_display)
                ):
                    trace_state.max_hold_power = power_linear_display.copy()
                else:
                    update_mask = np.isfinite(power_linear_display)
                    replace_mask = update_mask & ~np.isfinite(trace_state.max_hold_power)
                    max_mask = (
                        update_mask
                        & np.isfinite(trace_state.max_hold_power)
                        & (power_linear_display > trace_state.max_hold_power)
                    )
                    trace_state.max_hold_power[replace_mask] = power_linear_display[replace_mask]
                    trace_state.max_hold_power[max_mask] = power_linear_display[max_mask]
                trace_linear = trace_state.max_hold_power
            elif trace_state.trace_type == TRACE_TYPE_AVERAGE:
                alpha = 1.0 / max(1, trace_state.average_count)
                if (
                    trace_state.average_power is None
                    or len(trace_state.average_power) != len(power_linear_display)
                ):
                    trace_state.average_power = power_linear_display.copy()
                else:
                    update_mask = np.isfinite(power_linear_display)
                    init_mask = update_mask & ~np.isfinite(trace_state.average_power)
                    trace_state.average_power[init_mask] = power_linear_display[init_mask]
                    avg_mask = update_mask & np.isfinite(trace_state.average_power)
                    trace_state.average_power[avg_mask] = (
                        alpha * power_linear_display[avg_mask]
                        + (1.0 - alpha) * trace_state.average_power[avg_mask]
                    )
                trace_linear = trace_state.average_power
            else:
                trace_linear = power_linear_display

            trace_db = np.full(len(trace_linear), np.nan, dtype=float)
            trace_valid_mask = np.isfinite(trace_linear)
            trace_db[trace_valid_mask] = 10.0 * np.log10(trace_linear[trace_valid_mask] + 1e-20)
            trace_state.display_db = apply_display_power_correction(
                trace_db,
                self.calibration_offset_db,
                self.config.input_correction_db,
            )

        self._last_display_power_db = (
            self.trace_states[0].display_db
            if self.trace_states[0].display_db is not None
            else current_display_db
        )

    def _update_wideband_traces_from_display_db(self, composite_display_db: np.ndarray) -> None:
        self._last_current_display_db = composite_display_db.copy()

        for trace_state in self.trace_states:
            if trace_state.hold_enabled:
                continue

            if trace_state.trace_type == TRACE_TYPE_MAX_HOLD:
                if (
                    trace_state.max_hold_power is None
                    or len(trace_state.max_hold_power) != len(composite_display_db)
                ):
                    trace_state.max_hold_power = composite_display_db.copy()
                else:
                    update_mask = np.isfinite(composite_display_db)
                    replace_mask = update_mask & ~np.isfinite(trace_state.max_hold_power)
                    max_mask = (
                        update_mask
                        & np.isfinite(trace_state.max_hold_power)
                        & (composite_display_db > trace_state.max_hold_power)
                    )
                    trace_state.max_hold_power[replace_mask] = composite_display_db[replace_mask]
                    trace_state.max_hold_power[max_mask] = composite_display_db[max_mask]
                trace_state.display_db = trace_state.max_hold_power.copy()
            elif trace_state.trace_type == TRACE_TYPE_AVERAGE:
                alpha = 1.0 / max(1, trace_state.average_count)
                if (
                    trace_state.average_power is None
                    or len(trace_state.average_power) != len(composite_display_db)
                ):
                    trace_state.average_power = composite_display_db.copy()
                else:
                    update_mask = np.isfinite(composite_display_db)
                    init_mask = update_mask & ~np.isfinite(trace_state.average_power)
                    trace_state.average_power[init_mask] = composite_display_db[init_mask]
                    avg_mask = update_mask & np.isfinite(trace_state.average_power)
                    trace_state.average_power[avg_mask] = (
                        alpha * composite_display_db[avg_mask]
                        + (1.0 - alpha) * trace_state.average_power[avg_mask]
                    )
                trace_state.display_db = trace_state.average_power.copy()
            else:
                trace_state.display_db = composite_display_db.copy()

    def _reset_plot_state(self) -> None:
        self.frame_period_s = (
            self.config.update_interval_ms / 1000.0
            if self.config.update_interval_ms > 0
            else 1.0 / 60.0
        )
        if self._is_time_analyzer_mode():
            self._initialize_time_analyzer_runtime()
            if self._time_analyzer_time_axis_s is None:
                freq_axis_display_ghz = np.array([0.0, 1.0], dtype=float)
            else:
                freq_axis_display_ghz = self._time_analyzer_time_axis_s
        elif self.config.analyzer_mode == AnalyzerMode.SWEEP_SA:
            freq_axis_display_ghz = self.sweep_controller.get_sweep_frequency_axis_hz() / 1e9
        elif self._is_wideband_mode():
            if self._wideband_runtime_state is not None:
                freq_axis_display_ghz = self._wideband_runtime_state.composite_freq_axis_ghz
            else:
                start_hz, stop_hz = self._get_wideband_start_stop_hz()
                freq_axis_display_ghz = np.array(
                    [start_hz / 1e9, stop_hz / 1e9],
                    dtype=float,
                )
        else:
            freq_axis_display_ghz = self.processor.get_display_freq_axis_ghz()
        freq_axis_display_ghz_dec = self._get_active_waterfall_freq_axis_ghz()
        self._rebuild_waterfall_buffer_for_axis(freq_axis_display_ghz_dec)
        self.spectrum_plot.setXRange(
            freq_axis_display_ghz[0],
            freq_axis_display_ghz[-1],
            padding=X_AXIS_PADDING,
        )
        # Store the active axis before tick generation so Sweep SA reticks immediately.
        self._last_display_freq_axis_ghz = freq_axis_display_ghz
        self._update_plot_axis_labels_for_mode()
        self._apply_display_scale()
        self._last_display_power_db = np.full(len(freq_axis_display_ghz), self.y_min, dtype=float)
        self._last_current_display_db = self._last_display_power_db.copy()
        self._initialize_persistence_buffer()
        self._reset_trace_runtime_buffers()
        for trace_state in self.trace_states:
            trace_state.display_db = np.full(len(freq_axis_display_ghz), self.y_min, dtype=float)
        self._update_trace_curves()
        self._hide_sweep_progress_symbol()
        self._hide_time_analyzer_progress_symbol()
        self._update_all_marker_control_states()
        self._update_marker_items()

    def _apply_display_scale(self) -> None:
        self.spectrum_plot.setYRange(self.y_min, self.y_max, padding=Y_AXIS_PADDING)
        self.waterfall_img.setLevels((self.y_min, self.y_max))
        self._clear_persistence()
        self._update_fixed_ticks()

    def _apply_display_mode(self) -> None:
        if self.graph_view_mode == GRAPH_VIEW_WATERFALL_ONLY:
            self.waterfall_plot.setFixedSize(PLOT_WIDTH, DUAL_PLOT_TOTAL_HEIGHT)
            self.waterfall_plot.show()
            self.spectrum_plot.hide()
        elif self.graph_view_mode == GRAPH_VIEW_SPECTRUM_ONLY:
            self.spectrum_plot.setFixedSize(PLOT_WIDTH, DUAL_PLOT_TOTAL_HEIGHT)
            self.spectrum_plot.show()
            self.waterfall_plot.hide()
        else:
            self.waterfall_plot.setFixedSize(PLOT_WIDTH, PLOT_HEIGHT)
            self.spectrum_plot.setFixedSize(PLOT_WIDTH, PLOT_HEIGHT)
            self.waterfall_plot.show()
            self.spectrum_plot.show()
            return

        if self.graph_view_mode != GRAPH_VIEW_WATERFALL_ONLY:
            self.waterfall_plot.setFixedSize(PLOT_WIDTH, PLOT_HEIGHT)
        if self.graph_view_mode != GRAPH_VIEW_SPECTRUM_ONLY:
            self.spectrum_plot.setFixedSize(PLOT_WIDTH, PLOT_HEIGHT)

    def _apply_center_frequency_update(self) -> None:
        freq_axis_display_ghz = self.processor.get_display_freq_axis_ghz()
        freq_axis_display_ghz_dec = self.processor.get_decimated_display_freq_axis_ghz()

        self._last_display_freq_axis_ghz = freq_axis_display_ghz

        self.waterfall_img.setRect(
            QtCore.QRectF(
                freq_axis_display_ghz_dec[0],
                0.0,
                freq_axis_display_ghz_dec[-1] - freq_axis_display_ghz_dec[0],
                float(self.config.waterfall_history),
            )
        )
        self.waterfall_plot.setXRange(
            freq_axis_display_ghz_dec[0],
            freq_axis_display_ghz_dec[-1],
            padding=X_AXIS_PADDING,
        )
        self.waterfall_plot.setYRange(0.0, float(self.config.waterfall_history), padding=0.0)
        self._update_waterfall_ticks()
        self.spectrum_plot.setXRange(
            freq_axis_display_ghz[0],
            freq_axis_display_ghz[-1],
            padding=X_AXIS_PADDING,
        )
        self._clear_persistence()
        self._update_fixed_ticks()
        self._update_trace_curves()
        self._update_marker_items()

    def _apply_span_update(self) -> None:
        self._last_received_samples_total = 0
        self.received_samples_interval = 0
        self._reset_plot_state()

    def _format_sweep_point_profile(
        self,
        point_result,
        *,
        callback_sequence: int,
        callback_start_time: float,
        ui_draw_ms: float,
        marker_update_ms: float,
        redraw_performed: bool,
        step_sweep_total_ms: float,
        callback_gap_ms: float,
        callback_total_ms: float,
    ) -> str:
        return (
            f"SweepPoint "
            f"cb={callback_sequence} "
            f"t={callback_start_time:.6f}s "
            f"idx={self.sweep_controller.get_latest_result().completed_points} "
            f"f={point_result.frequency_hz / 1e6:.6f}MHz "
            f"config={point_result.configure_ms:.2f}ms "
            f"retune={point_result.retune_ms:.2f}ms "
            f"settle={point_result.settle_wait_ms:.2f}ms "
            f"flush={point_result.flush_ms:.2f}ms("
            f"r={point_result.flush_reads} "
            f"req={point_result.flush_samples}s "
            f"act={point_result.flush_actual_samples_per_read}s "
            f"total={point_result.flush_actual_samples_total}s) "
            f"capture={point_result.capture_ms:.2f}ms "
            f"process={point_result.process_ms:.2f}ms "
            f"step={point_result.step_total_ms:.2f}ms "
            f"step+ctrl={step_sweep_total_ms:.2f}ms "
            f"gap={callback_gap_ms:.2f}ms "
            f"timer={self._current_timer_interval_ms()}ms "
            f"ui={ui_draw_ms:.2f}ms "
            f"marker={marker_update_ms:.2f}ms "
            f"cb_total={callback_total_ms:.2f}ms "
            f"redraw={'Y' if redraw_performed else 'N'}"
        )

    def _build_timer(self) -> None:
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_spectrum)
        self._restart_timer_for_current_mode()

    def _current_timer_interval_ms(self) -> int:
        if self.config.analyzer_mode in (
            AnalyzerMode.SWEEP_SA,
            AnalyzerMode.WIDEBAND_REALTIME_SA,
        ):
            return self.config.sweep_update_interval_ms
        return self.config.update_interval_ms

    def _restart_timer_for_current_mode(self) -> None:
        self.timer.start(self._current_timer_interval_ms())

    def _make_status_text(
        self,
        interval_fps: float,
        avg_fps: float,
        median_dt_ms: float,
        interval_capture_ratio: float,
        avg_capture_ratio: float,
    ) -> str:
        return self._make_header_status_text()

    def _refresh_status_label(self) -> None:
        self._refresh_sweep_time_estimate()
        self._update_span_x_scale_controls()
        self.status_label.setText(self._make_header_status_text())

    def _get_header_frequency_fields(self) -> tuple[str, str, str, str]:
        if self._is_time_analyzer_mode():
            return (
                "Center",
                f"{self.config.center_freq_hz / 1e6:.3f} MHz",
                "Time Span",
                f"{float(self.config.time_analyzer_time_span_s):.3f} s",
            )
        if (
            self.config.use_start_stop_freq
            and self.config.display_start_freq_hz is not None
            and self.config.display_stop_freq_hz is not None
        ):
            return (
                "Start Freq",
                f"{self.config.display_start_freq_hz / 1e6:.3f} MHz",
                "Stop Freq",
                f"{self.config.display_stop_freq_hz / 1e6:.3f} MHz",
            )

        return (
            "Center",
            f"{self.config.center_freq_hz / 1e6:.3f} MHz",
            "Span",
            f"{self.config.display_span_hz / 1e6:.3f} MHz",
        )

    def _make_header_status_text(self) -> str:
        freq_label_1, freq_value_1, freq_label_2, freq_value_2 = (
            self._get_header_frequency_fields()
        )
        rbw_text = self._format_rbw_text()
        if self._is_time_analyzer_mode():
            avg_dt_text = (
                f"{self._time_analyzer_last_sweep_avg_dt_s:.4f} s"
                if self._time_analyzer_last_sweep_avg_dt_s is not None
                else "--"
            )
            line1 = (
                f"{freq_label_1}: {freq_value_1}   "
                f"{freq_label_2}: {freq_value_2}   "
                f"RBW: {rbw_text}   "
                f"Samples: {self._time_analyzer_last_sweep_samples}   "
                f"Avg dt: {avg_dt_text}"
            )
        elif self.config.analyzer_mode == AnalyzerMode.SWEEP_SA:
            line1 = (
                f"{freq_label_1}: {freq_value_1}   "
                f"{freq_label_2}: {freq_value_2}   "
                f"RBW: {rbw_text}   "
                f"Actual Swp: {self.actual_sweep_time_s:.1f} s"
            )
        else:
            line1 = (
                f"{freq_label_1}: {freq_value_1}   "
                f"{freq_label_2}: {freq_value_2}   "
                f"RBW: {rbw_text}"
            )
        line2 = (
            f"Ref Level: {self.config.ref_level_dbm:.0f} dBm   "
            f"Int Gain: {self.config.rx_gain_db} dB   "
            f"Ext ATT: {self.config.ext_att_db:.0f} dB   "
            f"Ext Gain: {self.config.ext_gain_db:.0f} dB"
        )
        line3 = self._make_fft_info_status_line()
        line4 = f"Detector: {self.config.sweep_detector_mode}"
        return f"{line1}\n{line2}\n{line3}\n{line4}"

    def _make_fft_info_status_line(self) -> str:
        if self.config.analyzer_mode == AnalyzerMode.SWEEP_SA:
            sample_rate_hz = float(self.config.sweep_sample_rate_hz)
            rf_bandwidth_hz = float(self.config.sweep_rf_bandwidth_hz)
            bin_width_hz = sample_rate_hz / max(1, int(self.config.fft_size))
        else:
            sample_rate_hz = float(self.config.sample_rate_hz)
            rf_bandwidth_hz = float(self.config.rx_bandwidth_hz)
            bin_width_hz = float(self.config.bin_width_hz)

        base_text = (
            f"FFT: {self.config.fft_size}   "
            f"Bin Width: {self._format_frequency_value_hz_compact(bin_width_hz)}   "
            f"RF BW: {self._format_frequency_value_hz_compact(rf_bandwidth_hz)}   "
            f"Samp Rate: {self._format_frequency_value_hz_compact(sample_rate_hz)}"
        )

        if self.config.analyzer_mode not in (
            AnalyzerMode.REALTIME_SA,
            AnalyzerMode.WIDEBAND_REALTIME_SA,
        ):
            return base_text

        fps_value = self._estimate_realtime_fps()
        if fps_value > 0.0:
            fps_int = int(round(fps_value))
            estimated_time_sec = int(round(self.config.waterfall_history / fps_value))
            history_text = f"History: {self.config.waterfall_history} ({estimated_time_sec}s)"
            fps_text = f"FPS: {fps_int}"
        else:
            fps_text = "FPS: --"
            history_text = f"History: {self.config.waterfall_history}"

        return f"{base_text}   {fps_text}   {history_text}"

    def _make_realtime_status_suffix(self) -> str:
        if self.config.analyzer_mode != AnalyzerMode.REALTIME_SA:
            return ""

        fps_value = self._estimate_realtime_fps()
        history_frames = self.config.waterfall_history
        if fps_value <= 0.0:
            time_text = "--"
            fps_text = "0"
        else:
            time_text = f"{history_frames / fps_value:.1f}"
            fps_text = f"{int(round(fps_value))}"

        return (
            f"   FPS: {fps_text}"
            f"   History: {history_frames} (≈ {time_text} s)"
        )

    def _estimate_realtime_fps(self) -> float:
        if not self.frame_dt_history:
            return 0.0

        window_size = min(30, len(self.frame_dt_history))
        recent_dts = self.frame_dt_history[-window_size:]
        avg_dt = float(np.mean(recent_dts)) if recent_dts else 0.0
        if avg_dt <= 0.0:
            return 0.0
        return 1.0 / avg_dt

    def _make_console_status_text(
        self,
        interval_fps: float,
        avg_fps: float,
        median_dt_ms: float,
        interval_capture_ratio: float,
        avg_capture_ratio: float,
    ) -> str:
        return (
            f"Perf    "
            f"Drops: {self.drop_count}    "
            f"MedianDt: {median_dt_ms:.2f} ms    "
            f"Capture(interval): {interval_capture_ratio * 100:.2f}%    "
            f"Capture(avg): {avg_capture_ratio * 100:.2f}%    "
            f"FPS(interval): {interval_fps:.2f}    "
            f"FPS(avg): {avg_fps:.2f}"
        )

    def _format_rbw_text(self) -> str:
        if self.config.rbw_hz is None:
            return "N/A"
        return self._format_frequency_value_hz(self.config.rbw_hz)

    def _format_frequency_value_hz(self, value_hz: float) -> str:
        if value_hz >= 1_000_000:
            return f"{value_hz / 1_000_000:.1f} MHz"
        if value_hz >= 1_000:
            return f"{value_hz / 1_000:.1f} kHz"
        return f"{value_hz:.0f} Hz"

    def _format_frequency_value_hz_compact(self, value_hz: float) -> str:
        if value_hz >= 1_000_000:
            return f"{value_hz / 1_000_000:.2f} MHz"
        if value_hz >= 1_000:
            return f"{value_hz / 1_000:.2f} kHz"
        return f"{value_hz:.0f} Hz"

    def _log_compare_sa(self, mode: str, **values: object) -> None:
        if not self.config.sweep_profile_logging:
            return
        ordered_keys = [
            "center_freq_hz",
            "point_freq_hz",
            "point_index",
            "center_idx",
            "rbw_hz",
            "fft_size",
            "sample_rate_hz",
            "rf_bandwidth_hz",
            "dc_removal",
            "window_len",
            "capture_len",
            "raw_point_power",
            "filtered_point_power",
            "detector_value",
            "detector_equivalent_sweep",
            "display_value",
            "display_value_sweep_equivalent",
            "ta_center_filtered",
            "ta_detector_equivalent",
            "ta_display_direct",
            "ta_display_detector_equivalent",
            "calibration_offset",
            "ext_att_db",
            "int_gain_db",
            "ext_gain_db",
        ]
        items: list[str] = [f"mode={mode}"]
        for key in ordered_keys:
            if key in values:
                items.append(f"{key}={values[key]}")
        print("CompareSA " + " ".join(items))

    def update_spectrum(self) -> None:
        if self._is_time_analyzer_mode():
            self._update_time_analyzer_spectrum()
            return
        self._hide_time_analyzer_progress_symbol()
        if self.config.analyzer_mode == AnalyzerMode.SWEEP_SA:
            self._update_sweep_spectrum()
            return
        if self._is_wideband_mode():
            self._update_wideband_spectrum()
            return

        now_frame = time.perf_counter()

        if self.prev_frame_time is not None:
            dt = now_frame - self.prev_frame_time
            self.frame_dt_history.append(dt)

            if len(self.frame_dt_history) >= self.config.drop_judge_window:
                recent = self.frame_dt_history[-self.config.drop_judge_window :]
                median_dt = float(np.median(recent))

                if dt > median_dt * self.config.drop_threshold_factor:
                    self.drop_count += 1

        self.prev_frame_time = now_frame

        iq = self.receiver.get_latest_block()
        if iq is None:
            return

        power_linear_full = self.processor.compute_filtered_power(iq)
        power_linear_display = self.processor.extract_display_spectrum(power_linear_full)
        current_power_db_display = 10.0 * np.log10(power_linear_display + 1e-20)
        current_display_db = apply_display_power_correction(
            current_power_db_display,
            self.calibration_offset_db,
            self.config.input_correction_db,
        )

        current_received_total = self.receiver.get_received_sample_count()
        self.received_samples_interval = (
            current_received_total - self._last_received_samples_total
        )
        self._last_received_samples_total = current_received_total

        self._append_waterfall_line(current_display_db)

        self._last_display_freq_axis_ghz = self.processor.get_display_freq_axis_ghz()
        self._last_current_display_db = current_display_db
        self._accumulate_persistence(current_display_db)

        self._update_traces_from_power_linear(power_linear_display)

        self._last_display_power_db = (
            self.trace_states[0].display_db
            if self.trace_states[0].display_db is not None
            else current_display_db
        )
        for marker_state in self.marker_states:
            if marker_state.is_enabled and marker_state.continuous_peak_enabled:
                trace_db = self._select_marker_trace_db(marker_state)
                if trace_db is None:
                    continue
                marker_peak_freq, _marker_peak_val = self.processor.detect_peak(trace_db)
                marker_state.frequency_hz = int(round(marker_peak_freq * 1e9))

        self._update_trace_curves()
        self._update_marker_items()

        self.frame_count_total += 1
        self.frame_count_interval += 1

        if self.sweep_state == SWEEP_STATE_SINGLE:
            self.timer.stop()
            self.receiver.stop()
            self.sweep_state = SWEEP_STATE_STOPPED
            self._update_continuous_button()

        now = time.perf_counter()
        if now - self.last_report_time >= 1.0:
            interval_elapsed = now - self.last_report_time
            total_elapsed = now - self.start_time

            interval_fps = self.frame_count_interval / interval_elapsed
            avg_fps = self.frame_count_total / total_elapsed

            interval_capture_ratio = (
                self.received_samples_interval
                / (self.config.sample_rate_hz * interval_elapsed)
            )

            avg_capture_ratio = (
                self.receiver.get_received_sample_count()
                / (self.config.sample_rate_hz * total_elapsed)
            )

            if len(self.frame_dt_history) >= self.config.drop_judge_window:
                recent = self.frame_dt_history[-self.config.drop_judge_window :]
                median_dt_ms = float(np.median(recent) * 1000.0)
            else:
                median_dt_ms = 0.0

            header_text = self._make_header_status_text()
            self.status_label.setText(header_text)

            self.frame_count_interval = 0
            self.last_report_time = now

    def _update_time_analyzer_spectrum(self) -> None:
        if (
            self._time_analyzer_time_axis_s is None
            or self._time_analyzer_sample_elapsed_s is None
            or self._time_analyzer_trace_db is None
            or self._time_analyzer_valid_mask is None
        ):
            self._initialize_time_analyzer_runtime()
            self._reset_plot_state()
        if (
            self._time_analyzer_time_axis_s is None
            or self._time_analyzer_sample_elapsed_s is None
            or self._time_analyzer_trace_db is None
            or self._time_analyzer_valid_mask is None
        ):
            return

        clipped_rbw = self._clip_sweep_rbw(self.config.rbw_hz)
        if self.config.rbw_hz != clipped_rbw:
            self.config.rbw_hz = clipped_rbw
            self._apply_time_analyzer_rbw_driven_capture_settings()

        capture_size = int(self.config.fft_size)
        iq = self.receiver.capture_block(capture_size)
        if iq is None or len(iq) == 0:
            return
        sample_timestamp = time.perf_counter()
        self._hide_time_analyzer_progress_symbol()
        if self._time_analyzer_discard_samples_remaining > 0:
            self._time_analyzer_discard_samples_remaining -= 1
            self._time_analyzer_sweep_start_timestamp = sample_timestamp
            self._hide_sweep_progress_symbol()
            return

        time_span_s = self._time_analyzer_time_span_s()
        wrapped = False
        if self._time_analyzer_sweep_sample_count == 0:
            # First valid sample defines time origin (t0).
            self._time_analyzer_sweep_start_timestamp = sample_timestamp
            elapsed_sec = 0.0
        else:
            if self._time_analyzer_sweep_start_timestamp is None:
                self._time_analyzer_sweep_start_timestamp = sample_timestamp
            elapsed_sec = max(0.0, sample_timestamp - self._time_analyzer_sweep_start_timestamp)
            if elapsed_sec > time_span_s:
                # Finalize and render the completed sweep once.
                self._finalize_time_analyzer_sweep_stats()
                self._publish_time_analyzer_accumulated_sweep()
                if self.sweep_state == SWEEP_STATE_SINGLE:
                    if self._sweep_like_suppress_progress_until_first_complete:
                        self._sweep_like_suppress_progress_until_first_complete = False
                    self._hide_sweep_progress_symbol()
                    self._finish_single_sweep_like()
                    return
                # Continuous mode uses fixed time window; restart from 0 s.
                self._reset_time_analyzer_time_window(start_timestamp=sample_timestamp)
                elapsed_sec = 0.0
                wrapped = True

        n = len(iq)
        window = self.processor.window
        iq_zero_mean = iq - np.mean(iq)
        iq_windowed = iq_zero_mean * window
        spectrum = np.fft.fftshift(np.fft.fft(iq_windowed))
        coherent_gain = np.sum(window) / n
        spectrum = spectrum / n
        spectrum = spectrum / coherent_gain
        raw_power_full = np.abs(spectrum) ** 2

        power_linear_full = self.processor.compute_filtered_power(iq)
        if len(power_linear_full) == 0:
            return
        center_index = max(0, int(self.config.fft_size) // 2)
        if center_index >= len(power_linear_full):
            center_index = len(power_linear_full) // 2
        raw_center_power_linear = float(raw_power_full[center_index])
        center_power_linear = float(power_linear_full[center_index])
        center_power_db = 10.0 * np.log10(center_power_linear + 1e-20)
        corrected_center_db = float(
            apply_display_power_correction(
                np.asarray([center_power_db], dtype=float),
                self.calibration_offset_db,
                self.config.input_correction_db,
            )[0]
        )
        ta_bin_width_hz = float(self.config.sample_rate_hz) / max(1, int(self.config.fft_size))
        ta_effective_rbw_hz = resolve_rbw_hz(self.config.rbw_hz, ta_bin_width_hz)
        ta_detector_series = SweepController._build_detector_observation_series_core(
            iq=iq,
            effective_rbw_hz=ta_effective_rbw_hz,
            sample_rate_hz=float(self.config.sample_rate_hz),
        )
        ta_detector_equivalent_linear = float(
            apply_detector(ta_detector_series, self.config.sweep_detector_mode)
        )
        ta_detector_equivalent_db = 10.0 * np.log10(ta_detector_equivalent_linear + 1e-20)
        ta_display_detector_equivalent_db = float(
            apply_display_power_correction(
                np.asarray([ta_detector_equivalent_db], dtype=float),
                self.calibration_offset_db,
                self.config.input_correction_db,
            )[0]
        )
        if self.config.sweep_profile_logging:
            print(
                "TimeAnalyzer "
                f"value_raw={center_power_db:.6f} "
                f"value_display={ta_display_detector_equivalent_db:.6f} "
                f"value_display_direct={corrected_center_db:.6f} "
                f"center_freq={int(self.config.center_freq_hz)}Hz "
                f"rbw={float(self.config.rbw_hz):.1f}Hz"
            )
        self._log_compare_sa(
            "TIME",
            center_freq_hz=int(self.config.center_freq_hz),
            point_freq_hz=int(self.config.center_freq_hz),
            point_index=center_index,
            center_idx=max(0, int(self.config.fft_size) // 2),
            rbw_hz=float(self.config.rbw_hz) if self.config.rbw_hz is not None else 0.0,
            fft_size=int(self.config.fft_size),
            sample_rate_hz=int(self.config.sample_rate_hz),
            rf_bandwidth_hz=int(self.config.rx_bandwidth_hz),
            dc_removal="ON",
            window_len=len(window),
            capture_len=len(iq),
            raw_point_power=f"{raw_center_power_linear:.6e}",
            filtered_point_power=f"{center_power_linear:.6e}",
            detector_value=f"{ta_detector_equivalent_linear:.6e}",
            detector_equivalent_sweep=f"{ta_detector_equivalent_linear:.6e}",
            display_value=f"{ta_display_detector_equivalent_db:.6f}",
            display_value_sweep_equivalent=f"{ta_display_detector_equivalent_db:.6f}",
            ta_center_filtered=f"{center_power_linear:.6e}",
            ta_detector_equivalent=f"{ta_detector_equivalent_linear:.6e}",
            ta_display_direct=f"{corrected_center_db:.6f}",
            ta_display_detector_equivalent=f"{ta_display_detector_equivalent_db:.6f}",
            calibration_offset=f"{self.calibration_offset_db:.2f}",
            ext_att_db=f"{self.config.ext_att_db:.2f}",
            int_gain_db=int(self.config.rx_gain_db),
            ext_gain_db=f"{self.config.ext_gain_db:.2f}",
        )

        idx = int(self._time_analyzer_write_index)
        if idx < 0:
            idx = 0
        if idx >= len(self._time_analyzer_trace_db):
            idx = len(self._time_analyzer_trace_db) - 1
        self._time_analyzer_sample_elapsed_s[idx] = elapsed_sec
        self._time_analyzer_trace_db[idx] = ta_display_detector_equivalent_db
        self._time_analyzer_valid_mask[idx] = True
        if self._time_analyzer_sweep_sample_count == 0:
            self._time_analyzer_sweep_first_timestamp = sample_timestamp
        self._time_analyzer_sweep_last_timestamp = sample_timestamp
        self._time_analyzer_sweep_sample_count += 1
        if idx < (len(self._time_analyzer_trace_db) - 1):
            self._time_analyzer_write_index = idx + 1
        self._hide_sweep_progress_symbol()
        # No per-sample drawing in TA; render only when one sweep is complete.
        if wrapped and self._sweep_like_suppress_progress_until_first_complete:
            self._sweep_like_suppress_progress_until_first_complete = False

    def _update_sweep_spectrum(self) -> None:
        callback_start = time.perf_counter()
        self._sweep_callback_sequence += 1
        if self._last_sweep_callback_time is None:
            callback_gap_ms = 0.0
        else:
            callback_gap_ms = (callback_start - self._last_sweep_callback_time) * 1000.0
        self._last_sweep_callback_time = callback_start

        step_start = time.perf_counter()
        frame_result = self.sweep_controller.step_sweep()
        step_sweep_total_ms = (time.perf_counter() - step_start) * 1000.0
        if frame_result.freq_axis_hz is None or frame_result.display_db is None:
            return

        sweep_freq_axis_ghz = frame_result.freq_axis_hz / 1e9
        axis_changed = (
            self._last_display_freq_axis_ghz is None
            or len(self._last_display_freq_axis_ghz) != len(sweep_freq_axis_ghz)
            or not np.array_equal(self._last_display_freq_axis_ghz, sweep_freq_axis_ghz)
        )
        if axis_changed:
            self._last_display_freq_axis_ghz = sweep_freq_axis_ghz
            self.spectrum_plot.setXRange(
                sweep_freq_axis_ghz[0],
                sweep_freq_axis_ghz[-1],
                padding=X_AXIS_PADDING,
            )
            self._update_fixed_ticks()

        completed_points = frame_result.completed_points
        redraw_interval = max(1, int(self.config.sweep_ui_update_interval_points))
        redraw_performed = (
            completed_points != self._last_sweep_drawn_completed_points
            and (
                completed_points <= 1
                or frame_result.sweep_complete
                or (completed_points % redraw_interval == 0)
            )
        )

        ui_draw_ms = 0.0
        marker_update_ms = 0.0
        if redraw_performed:
            ui_start = time.perf_counter()
            self._last_display_freq_axis_ghz = sweep_freq_axis_ghz

            power_linear_display = np.full(len(frame_result.display_db), np.nan, dtype=float)
            valid_mask = np.isfinite(frame_result.display_db)
            power_linear_display[valid_mask] = 10.0 ** (
                frame_result.display_db[valid_mask] / 10.0
            )
            self._update_traces_from_power_linear(power_linear_display)
            if frame_result.sweep_complete:
                self._capture_completed_sweep_snapshot()
            self._update_trace_curves()
            self._last_sweep_drawn_completed_points = completed_points
            ui_draw_ms = (time.perf_counter() - ui_start) * 1000.0

        self._update_sweep_progress_symbol(frame_result)

        if frame_result.sweep_complete and self._pending_sweep_marker_update:
            marker_start = time.perf_counter()
            self._pending_sweep_marker_update = False
            self._update_markers_for_completed_sweep()
            marker_update_ms = (time.perf_counter() - marker_start) * 1000.0

        if frame_result.sweep_complete and self._sweep_like_suppress_progress_until_first_complete:
            self._sweep_like_suppress_progress_until_first_complete = False

        if self.sweep_state == SWEEP_STATE_SINGLE and not self.sweep_controller.is_running():
            self._finish_single_sweep_like()

        callback_total_ms = (time.perf_counter() - callback_start) * 1000.0
        latest_point_result = self.sweep_controller.get_latest_point_result()
        if latest_point_result is not None:
            center_match_tolerance_hz = max(1.0, float(self.config.sweep_step_hz) * 0.5)
            if abs(float(latest_point_result.frequency_hz) - float(self.config.center_freq_hz)) <= center_match_tolerance_hz:
                display_value = float(
                    apply_display_power_correction(
                        np.asarray([latest_point_result.measured_power_db], dtype=float),
                        self.calibration_offset_db,
                        self.config.input_correction_db,
                    )[0]
                )
                self._log_compare_sa(
                    "SWEEP",
                    center_freq_hz=int(self.config.center_freq_hz),
                    point_freq_hz=int(latest_point_result.frequency_hz),
                    point_index=int(latest_point_result.rbw_center_bin_index),
                    center_idx=max(0, int(self.config.fft_size) // 2),
                    rbw_hz=float(latest_point_result.effective_rbw_hz),
                    fft_size=int(self.config.fft_size),
                    sample_rate_hz=int(self.config.sweep_sample_rate_hz),
                    rf_bandwidth_hz=int(self.config.sweep_rf_bandwidth_hz),
                    dc_removal="ON" if latest_point_result.dc_removal_applied else "OFF",
                    window_len=int(latest_point_result.window_len),
                    capture_len=int(latest_point_result.capture_samples),
                    raw_point_power=f"{latest_point_result.raw_center_power_linear:.6e}",
                    filtered_point_power=f"{latest_point_result.filtered_center_power_linear:.6e}",
                    detector_value=f"{latest_point_result.measured_power_linear:.6e}",
                    display_value=f"{display_value:.6f}",
                    calibration_offset=f"{self.calibration_offset_db:.2f}",
                    ext_att_db=f"{self.config.ext_att_db:.2f}",
                    int_gain_db=int(self.config.rx_gain_db),
                    ext_gain_db=f"{self.config.ext_gain_db:.2f}",
                )

    def closeEvent(self, event) -> None:
        self.timer.stop()
        self.receiver.close()
        super().closeEvent(event)
