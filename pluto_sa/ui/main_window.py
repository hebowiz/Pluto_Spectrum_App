"""Main window."""

from __future__ import annotations

import time

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from pluto_sa.config.spectrum_config import SpectrumConfig
from pluto_sa.sdr.pluto_receiver import PlutoReceiver
from pluto_sa.signal.spectrum_processor import SpectrumProcessor
from pluto_sa.utils.calibration import apply_offset_db

X_AXIS_PADDING = 0.02
Y_AXIS_PADDING = 0.03
LEFT_AXIS_WIDTH = 72
BOTTOM_AXIS_HEIGHT = 42
PLUTO_MIN_CENTER_FREQ_MHZ = 325.0
PLUTO_MAX_CENTER_FREQ_MHZ = 3800.0
MIN_SPAN_MHZ = 0.001
MIN_RBW_KHZ = 0.0
PLOT_WIDTH = 920
PLOT_HEIGHT = 320
PLOT_SPACING = 12
DUAL_PLOT_TOTAL_HEIGHT = PLOT_HEIGHT * 2 + PLOT_SPACING
CONTROL_PANEL_WIDTH = 240
WINDOW_WIDTH = 1216
WINDOW_HEIGHT = 762
DISPLAY_MODE_WATERFALL_SPECTRUM = "Waterfall + Spectrum"
DISPLAY_MODE_WATERFALL_ONLY = "Waterfall only"
DISPLAY_MODE_SPECTRUM_ONLY = "Spectrum only"
DISPLAY_MODE_OPTIONS = [
    DISPLAY_MODE_WATERFALL_SPECTRUM,
    DISPLAY_MODE_WATERFALL_ONLY,
    DISPLAY_MODE_SPECTRUM_ONLY,
]


class FrequencyAxisItem(pg.AxisItem):
    """Format frequency ticks with a fixed GHz precision."""

    def tickStrings(self, values, scale, spacing):
        return [f"{value:.3f}" for value in values]


class WaterfallTimeAxisItem(pg.AxisItem):
    """Show elapsed time labels with 0 at the bottom of the waterfall."""

    def __init__(self, time_span_fn, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._time_span_fn = time_span_fn

    def tickStrings(self, values, scale, spacing):
        time_span_s = self._time_span_fn()
        return [f"{max(0.0, time_span_s - value):.1f}" for value in values]


class RealtimeSpectrumWindow(QtWidgets.QMainWindow):
    """Own GUI construction and rendering only."""

    def __init__(
        self,
        config: SpectrumConfig,
        receiver: PlutoReceiver,
        processor: SpectrumProcessor,
        calibration_offset_db: float,
    ) -> None:
        super().__init__()
        self.config = config
        self.receiver = receiver
        self.processor = processor
        self.calibration_offset_db = calibration_offset_db
        self.display_mode = DISPLAY_MODE_WATERFALL_SPECTRUM

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

        self.y_min = -100
        self.y_max = 0

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
        self.waterfall_time_span_s = self.config.waterfall_history * self.frame_period_s

        self._build_ui()
        self._build_timer()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer_layout = QtWidgets.QHBoxLayout(central)
        outer_layout.setContentsMargins(12, 12, 12, 12)
        outer_layout.setSpacing(12)

        left_panel = QtWidgets.QWidget()
        left_panel.setFixedWidth(PLOT_WIDTH)
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
        layout.addWidget(self.status_label)

        pg.setConfigOptions(antialias=False)

        self.waterfall_plot = pg.PlotWidget(
            axisItems={
                "bottom": FrequencyAxisItem(orientation="bottom"),
                "left": WaterfallTimeAxisItem(
                    lambda: self.waterfall_time_span_s,
                    orientation="left",
                ),
            }
        )
        self.waterfall_plot.setBackground("k")
        self._configure_plot_chrome(self.waterfall_plot)
        self.waterfall_plot.setLabel("bottom", "Frequency [GHz]")
        self.waterfall_plot.setLabel("left", "Time [s]")
        self.waterfall_plot.getViewBox().invertY(True)
        self.waterfall_plot.getAxis("left").setPen("w")
        self.waterfall_plot.getAxis("bottom").setPen("w")
        self.waterfall_plot.setFixedSize(PLOT_WIDTH, PLOT_HEIGHT)

        self.waterfall_img = pg.ImageItem()
        self.waterfall_plot.addItem(self.waterfall_img)

        self.waterfall_img.setImage(
            self.waterfall_buffer,
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
                self.waterfall_time_span_s,
            )
        )

        self.waterfall_plot.setXRange(x_min, x_max, padding=X_AXIS_PADDING)
        self.waterfall_plot.setYRange(0.0, self.waterfall_time_span_s, padding=0.0)

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

        self.spectrum_curve = self.spectrum_plot.plot(
            freq_axis_display_ghz,
            np.full(len(freq_axis_display_ghz), self.y_min, dtype=float),
            pen=pg.mkPen("y", width=1.5),
        )

        self.peak_marker = pg.ScatterPlotItem(size=8, brush=pg.mkBrush("r"))
        self.spectrum_plot.addItem(self.peak_marker)

        self.peak_text = pg.TextItem(color="w", anchor=(0, 1))
        self.spectrum_plot.addItem(self.peak_text)

        layout.addWidget(self.spectrum_plot)
        self._apply_display_mode()

        outer_layout.addWidget(left_panel, stretch=1)
        outer_layout.addWidget(self._build_control_panel())

    def _build_control_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QFrame()
        panel.setFixedWidth(CONTROL_PANEL_WIDTH)
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
        self.freq_channel_page = self._build_freq_channel_page()
        self.control_stack.addWidget(self.main_menu_page)
        self.control_stack.addWidget(self.freq_channel_page)
        self.back_button.clicked.connect(
            lambda: self._show_control_page("Main Menu", self.main_menu_page)
        )
        self._show_control_page("Main Menu", self.main_menu_page)

        return panel

    def _build_main_menu_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        panel_layout = QtWidgets.QVBoxLayout(page)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(10)

        analyzer_group = QtWidgets.QGroupBox("Analyzer Setup")
        self._apply_groupbox_title_font(analyzer_group)
        analyzer_layout = QtWidgets.QVBoxLayout(analyzer_group)
        self.freq_menu_button = self._make_control_button("FREQ Channel")
        self.span_button = self._make_control_button("SPAN X Scale")
        self.rbw_button = self._make_control_button("RBW")
        self.amplitude_button = self._make_control_button("AMPTD Y Scale")
        analyzer_layout.addWidget(self.freq_menu_button)
        analyzer_layout.addWidget(self.span_button)
        analyzer_layout.addWidget(self.rbw_button)
        analyzer_layout.addWidget(self.amplitude_button)

        sweep_group = QtWidgets.QGroupBox("Sweep Control")
        self._apply_groupbox_title_font(sweep_group)
        sweep_layout = QtWidgets.QVBoxLayout(sweep_group)
        self.start_button = self._make_control_button("Start")
        self.stop_button = self._make_control_button("Stop")
        sweep_layout.addWidget(self.start_button)
        sweep_layout.addWidget(self.stop_button)

        display_group = QtWidgets.QGroupBox("Display")
        self._apply_groupbox_title_font(display_group)
        display_layout = QtWidgets.QVBoxLayout(display_group)
        self.display_button = self._make_control_button("View / Display")
        display_layout.addWidget(self.display_button)

        panel_layout.addWidget(analyzer_group)
        panel_layout.addWidget(sweep_group)
        panel_layout.addWidget(display_group)
        panel_layout.addStretch(1)

        self.freq_menu_button.clicked.connect(
            lambda: self._show_control_page("FREQ Channel", self.freq_channel_page)
        )
        self.span_button.clicked.connect(self._on_span_x_scale_clicked)
        self.rbw_button.clicked.connect(self._on_rbw_clicked)
        self.amplitude_button.clicked.connect(self._on_amptd_y_scale_clicked)
        self.start_button.clicked.connect(self._on_start_clicked)
        self.stop_button.clicked.connect(self._on_stop_clicked)
        self.display_button.clicked.connect(self._on_view_display_clicked)
        return page

    def _build_freq_channel_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        page_layout = QtWidgets.QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(10)

        self.freq_center_button = self._make_control_button("Center")
        self.freq_span_button = self._make_control_button("Span")
        self.freq_start_button = self._make_control_button("Start")
        self.freq_stop_button = self._make_control_button("Stop")

        page_layout.addWidget(self.freq_center_button)
        page_layout.addWidget(self.freq_span_button)
        page_layout.addWidget(self.freq_start_button)
        page_layout.addWidget(self.freq_stop_button)
        page_layout.addStretch(1)

        self.freq_center_button.clicked.connect(self._on_freq_channel_clicked)
        self.freq_span_button.clicked.connect(self._on_span_x_scale_clicked)
        self.freq_start_button.clicked.connect(self._on_start_clicked)
        self.freq_stop_button.clicked.connect(self._on_stop_clicked)
        return page

    def _show_control_page(self, title: str, page: QtWidgets.QWidget) -> None:
        self.control_title_label.setText(title)
        self.control_stack.setCurrentWidget(page)
        self.back_button.setEnabled(page is not self.main_menu_page)
        self.back_button.setVisible(page is not self.main_menu_page)

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

    def _configure_plot_chrome(self, plot_widget: pg.PlotWidget) -> None:
        plot_item = plot_widget.getPlotItem()
        plot_item.layout.setContentsMargins(12, 8, 12, 8)
        plot_widget.getViewBox().setDefaultPadding(0.0)
        plot_item.getAxis("bottom").setStyle(autoExpandTextSpace=True, tickTextOffset=8)
        plot_item.getAxis("left").setStyle(autoExpandTextSpace=True, tickTextOffset=8)
        plot_item.getAxis("left").setWidth(LEFT_AXIS_WIDTH)
        plot_item.getAxis("bottom").setHeight(BOTTOM_AXIS_HEIGHT)

    def _update_fixed_ticks(self) -> None:
        freq_axis_display_ghz = self.processor.get_display_freq_axis_ghz()
        x_min = freq_axis_display_ghz[0]
        x_max = freq_axis_display_ghz[-1]

        x_ticks = [(value, f"{value:.3f}") for value in np.linspace(x_min, x_max, 11)]
        y_ticks = [
            (value, f"{value:.0f}") for value in np.linspace(self.y_min, self.y_max, 11)
        ]

        self.spectrum_plot.getAxis("bottom").setTicks([x_ticks])
        self.spectrum_plot.getAxis("left").setTicks([y_ticks])

    def _on_freq_channel_clicked(self) -> None:
        value, accepted = QtWidgets.QInputDialog.getDouble(
            self,
            "FREQ Channel",
            "Center Frequency [MHz]",
            value=self.config.center_freq_hz / 1e6,
            min=PLUTO_MIN_CENTER_FREQ_MHZ,
            max=PLUTO_MAX_CENTER_FREQ_MHZ,
            decimals=3,
        )
        if not accepted:
            return

        center_freq_hz = int(round(value * 1e6))
        if center_freq_hz <= 0:
            return

        self.config.center_freq_hz = center_freq_hz
        self.receiver.retune_lo(center_freq_hz)
        self.processor.update_center_frequency(center_freq_hz)
        self._apply_center_frequency_update()
        self._refresh_status_label()

    def _on_span_x_scale_clicked(self) -> None:
        value, accepted = QtWidgets.QInputDialog.getDouble(
            self,
            "SPAN X Scale",
            "Display Span [MHz]",
            value=self.config.display_span_hz / 1e6,
            min=MIN_SPAN_MHZ,
            decimals=3,
        )
        if not accepted:
            return

        display_span_hz = int(round(value * 1e6))
        if display_span_hz <= 0:
            return

        self.config.display_span_hz = display_span_hz
        self.config.__post_init__()
        self.receiver.reconfigure_span(self.config)
        self.processor.update_span_related(self.config)
        self._apply_span_update()
        self._refresh_status_label()

    def _on_rbw_clicked(self) -> None:
        current_value = 0.0 if self.config.rbw_hz is None else self.config.rbw_hz / 1e3
        value, accepted = QtWidgets.QInputDialog.getDouble(
            self,
            "RBW",
            "RBW [kHz] (0 = Auto/None)",
            value=current_value,
            min=MIN_RBW_KHZ,
            decimals=3,
        )
        if not accepted:
            return

        self.config.rbw_hz = None if value <= 0.0 else float(value * 1e3)
        self._rebuild_processor_only()
        self._refresh_status_label()

    def _on_amptd_y_scale_clicked(self) -> None:
        min_value, accepted = QtWidgets.QInputDialog.getDouble(
            self,
            "AMPTD Y Scale",
            "Y Min [dBm]",
            value=float(self.y_min),
            decimals=1,
        )
        if not accepted:
            return

        max_value, accepted = QtWidgets.QInputDialog.getDouble(
            self,
            "AMPTD Y Scale",
            "Y Max [dBm]",
            value=float(self.y_max),
            decimals=1,
        )
        if not accepted or max_value <= min_value:
            return

        self.y_min = min_value
        self.y_max = max_value
        self._apply_display_scale()
        self._refresh_status_label()

    def _on_start_clicked(self) -> None:
        self.receiver.start()
        self.timer.start(self.config.update_interval_ms)

    def _on_stop_clicked(self) -> None:
        self.timer.stop()
        self.receiver.stop()

    def _on_view_display_clicked(self) -> None:
        mode, accepted = QtWidgets.QInputDialog.getItem(
            self,
            "View / Display",
            "Display Mode",
            DISPLAY_MODE_OPTIONS,
            current=DISPLAY_MODE_OPTIONS.index(self.display_mode),
            editable=False,
        )
        if not accepted:
            return

        self.display_mode = mode
        self._apply_display_mode()

    def _rebuild_receiver_and_processor(self) -> None:
        self.timer.stop()
        self.receiver.reconfigure(self.config)
        self.processor = SpectrumProcessor(self.config)
        self._last_received_samples_total = 0
        self.received_samples_interval = 0
        self._reset_plot_state()
        self.timer.start(self.config.update_interval_ms)

    def _rebuild_processor_only(self) -> None:
        self.processor = SpectrumProcessor(self.config)
        self._reset_plot_state()

    def _reset_plot_state(self) -> None:
        self.frame_period_s = (
            self.config.update_interval_ms / 1000.0
            if self.config.update_interval_ms > 0
            else 1.0 / 60.0
        )
        self.waterfall_time_span_s = self.config.waterfall_history * self.frame_period_s

        freq_axis_display_ghz = self.processor.get_display_freq_axis_ghz()
        freq_axis_display_ghz_dec = self.processor.get_decimated_display_freq_axis_ghz()
        wf_width = len(freq_axis_display_ghz_dec)
        self.waterfall_buffer = np.full(
            (self.config.waterfall_history, wf_width),
            self.y_min,
            dtype=np.float32,
        )

        self.waterfall_img.setImage(
            self.waterfall_buffer,
            autoLevels=False,
            axisOrder="row-major",
        )
        self.waterfall_img.setRect(
            QtCore.QRectF(
                freq_axis_display_ghz_dec[0],
                0.0,
                freq_axis_display_ghz_dec[-1] - freq_axis_display_ghz_dec[0],
                self.waterfall_time_span_s,
            )
        )
        self.waterfall_plot.setXRange(
            freq_axis_display_ghz_dec[0],
            freq_axis_display_ghz_dec[-1],
            padding=X_AXIS_PADDING,
        )
        self.waterfall_plot.setYRange(0.0, self.waterfall_time_span_s, padding=0.0)

        self.spectrum_curve.setData(
            freq_axis_display_ghz,
            np.full(len(freq_axis_display_ghz), self.y_min, dtype=float),
        )
        self.spectrum_plot.setXRange(
            freq_axis_display_ghz[0],
            freq_axis_display_ghz[-1],
            padding=X_AXIS_PADDING,
        )
        self._apply_display_scale()
        self.peak_marker.setData([], [])
        self.peak_text.setText("")

    def _apply_display_scale(self) -> None:
        self.spectrum_plot.setYRange(self.y_min, self.y_max, padding=Y_AXIS_PADDING)
        self.waterfall_img.setLevels((self.y_min, self.y_max))
        self._update_fixed_ticks()

    def _apply_display_mode(self) -> None:
        if self.display_mode == DISPLAY_MODE_WATERFALL_ONLY:
            self.waterfall_plot.setFixedSize(PLOT_WIDTH, DUAL_PLOT_TOTAL_HEIGHT)
            self.waterfall_plot.show()
            self.spectrum_plot.hide()
        elif self.display_mode == DISPLAY_MODE_SPECTRUM_ONLY:
            self.spectrum_plot.setFixedSize(PLOT_WIDTH, DUAL_PLOT_TOTAL_HEIGHT)
            self.spectrum_plot.show()
            self.waterfall_plot.hide()
        else:
            self.waterfall_plot.setFixedSize(PLOT_WIDTH, PLOT_HEIGHT)
            self.spectrum_plot.setFixedSize(PLOT_WIDTH, PLOT_HEIGHT)
            self.waterfall_plot.show()
            self.spectrum_plot.show()
            return

        if self.display_mode != DISPLAY_MODE_WATERFALL_ONLY:
            self.waterfall_plot.setFixedSize(PLOT_WIDTH, PLOT_HEIGHT)
        if self.display_mode != DISPLAY_MODE_SPECTRUM_ONLY:
            self.spectrum_plot.setFixedSize(PLOT_WIDTH, PLOT_HEIGHT)

    def _apply_center_frequency_update(self) -> None:
        freq_axis_display_ghz = self.processor.get_display_freq_axis_ghz()
        freq_axis_display_ghz_dec = self.processor.get_decimated_display_freq_axis_ghz()

        self.waterfall_img.setRect(
            QtCore.QRectF(
                freq_axis_display_ghz_dec[0],
                0.0,
                freq_axis_display_ghz_dec[-1] - freq_axis_display_ghz_dec[0],
                self.waterfall_time_span_s,
            )
        )
        self.waterfall_plot.setXRange(
            freq_axis_display_ghz_dec[0],
            freq_axis_display_ghz_dec[-1],
            padding=X_AXIS_PADDING,
        )
        self.spectrum_plot.setXRange(
            freq_axis_display_ghz[0],
            freq_axis_display_ghz[-1],
            padding=X_AXIS_PADDING,
        )
        self._update_fixed_ticks()
        self.peak_marker.setData([], [])
        self.peak_text.setText("")

    def _apply_span_update(self) -> None:
        self._last_received_samples_total = 0
        self.received_samples_interval = 0
        self._reset_plot_state()

    def _build_timer(self) -> None:
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_spectrum)
        self.timer.start(self.config.update_interval_ms)

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
        self.status_label.setText(self._make_header_status_text())

    def _make_header_status_text(self) -> str:
        rbw_text = self._format_rbw_text()
        line1 = (
            f"Center: {self.config.center_freq_hz / 1e6:.3f} MHz   "
            f"Span: {self.config.display_span_hz / 1e6:.3f} MHz   "
            f"RBW: {rbw_text}   "
            f"FFT: {self.config.fft_size}   "
            f"Bin Width: {self.config.sample_rate_hz / self.config.rx_buffer_size:.1f} Hz"
        )
        line2 = f"Gain: {self.config.rx_gain_db} dB"
        return f"{line1}\n{line2}"

    def _make_console_status_text(
        self,
        interval_fps: float,
        avg_fps: float,
        median_dt_ms: float,
        interval_capture_ratio: float,
        avg_capture_ratio: float,
    ) -> str:
        rbw_text = self._format_rbw_text()
        return (
            f"Center: {self.config.center_freq_hz / 1e6:.3f} MHz    "
            f"Span(display): {self.config.display_span_hz / 1e6:.3f} MHz    "
            f"Span(rx): {self.config.sample_rate_hz / 1e6:.3f} MHz    "
            f"RBW: {rbw_text}    "
            f"FFT: {self.config.fft_size}    "
            f"CaptureBuf: {self.config.capture_buffer_blocks} blk    "
            f"Bin Width: {self.config.sample_rate_hz / self.config.rx_buffer_size:.1f} Hz    "
            f"Gain: {self.config.rx_gain_db} dB    "
            f"Drops: {self.drop_count}    "
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

    def update_spectrum(self) -> None:
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

        power_db_full = self.processor.compute_spectrum(iq)
        power_db_display = self.processor.extract_display_spectrum(power_db_full)
        power_db_display = apply_offset_db(power_db_display, self.calibration_offset_db)

        current_received_total = self.receiver.get_received_sample_count()
        self.received_samples_interval = (
            current_received_total - self._last_received_samples_total
        )
        self._last_received_samples_total = current_received_total

        power_db_dec = power_db_display[:: self.config.waterfall_decimation]

        self.waterfall_buffer[:-1] = self.waterfall_buffer[1:]
        self.waterfall_buffer[-1] = power_db_dec.astype(np.float32)

        self.waterfall_img.setImage(
            self.waterfall_buffer,
            autoLevels=False,
            axisOrder="row-major",
        )

        self.spectrum_curve.setData(
            self.processor.get_display_freq_axis_ghz(),
            power_db_display,
        )

        peak_freq, peak_val = self.processor.detect_peak(power_db_display)

        self.peak_marker.setData([peak_freq], [peak_val])
        self.peak_text.setText(f"{peak_freq:.6f} GHz\n{peak_val:.2f} dBm")
        self.peak_text.setPos(peak_freq, peak_val)

        self.frame_count_total += 1
        self.frame_count_interval += 1

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
            console_text = self._make_console_status_text(
                interval_fps=interval_fps,
                avg_fps=avg_fps,
                median_dt_ms=median_dt_ms,
                interval_capture_ratio=interval_capture_ratio,
                avg_capture_ratio=avg_capture_ratio,
            )

            self.status_label.setText(header_text)
            print(console_text)

            self.frame_count_interval = 0
            self.last_report_time = now

    def closeEvent(self, event) -> None:
        self.timer.stop()
        self.receiver.close()
        super().closeEvent(event)
