"""Main window."""

from __future__ import annotations

import time

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from pluto_sa.config.spectrum_config import SpectrumConfig
from pluto_sa.sdr.pluto_receiver import PlutoReceiver
from pluto_sa.signal.spectrum_processor import SpectrumProcessor
from pluto_sa.utils.calibration import apply_offset_db

X_AXIS_PADDING = 0.02
Y_AXIS_PADDING = 0.03
LEFT_AXIS_WIDTH = 72
BOTTOM_AXIS_HEIGHT = 42


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

        self.setWindowTitle("PlutoSDR Real-Time Spectrum Prototype")
        self.resize(1280, 800)

        self.frame_count_total = 0
        self.frame_count_interval = 0
        self.start_time = time.perf_counter()
        self.last_report_time = self.start_time

        self.prev_frame_time = None
        self.frame_dt_history: list[float] = []
        self.drop_count = 0

        self._last_received_samples_total = 0
        self.received_samples_interval = 0

        wf_width = len(self.processor.get_decimated_display_freq_axis_ghz())
        self.waterfall_buffer = np.full(
            (self.config.waterfall_history, wf_width),
            0.0,
            dtype=np.float32,
        )

        self.y_min = -100
        self.y_max = 0

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
        layout = QtWidgets.QVBoxLayout(central)

        self.status_label = QtWidgets.QLabel()
        self.status_label.setStyleSheet("color: white; background-color: black;")
        self.status_label.setText(
            self._make_status_text(
                interval_fps=0.0,
                avg_fps=0.0,
                median_dt_ms=0.0,
                interval_capture_ratio=0.0,
                avg_capture_ratio=0.0,
            )
        )
        layout.addWidget(self.status_label)

        pg.setConfigOptions(antialias=False)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        layout.addWidget(splitter)

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

        splitter.addWidget(self.waterfall_plot)

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
            np.zeros(len(freq_axis_display_ghz)),
            pen=pg.mkPen("y", width=1.5),
        )

        self.peak_marker = pg.ScatterPlotItem(size=8, brush=pg.mkBrush("r"))
        self.spectrum_plot.addItem(self.peak_marker)

        self.peak_text = pg.TextItem(color="w", anchor=(0, 1))
        self.spectrum_plot.addItem(self.peak_text)

        splitter.addWidget(self.spectrum_plot)
        splitter.setSizes([500, 300])

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
        return (
            f"Center: {self.config.center_freq_hz / 1e6:.3f} MHz    "
            f"Span(display): {self.config.display_span_hz / 1e6:.3f} MHz    "
            f"Span(rx): {self.config.sample_rate_hz / 1e6:.3f} MHz    "
            f"FFT: {self.config.fft_size}    "
            f"CaptureBuf: {self.config.capture_buffer_blocks} blk    "
            f"RBW(base): {self.config.sample_rate_hz / self.config.rx_buffer_size:.1f} Hz/bin    "
            f"Gain: {self.config.rx_gain_db} dB    "
            f"Drops: {self.drop_count}    "
            f"Capture(interval): {interval_capture_ratio * 100:.2f}%    "
            f"Capture(avg): {avg_capture_ratio * 100:.2f}%    "
            f"FPS(interval): {interval_fps:.2f}    "
            f"FPS(avg): {avg_fps:.2f}"
        )

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

            text = self._make_status_text(
                interval_fps=interval_fps,
                avg_fps=avg_fps,
                median_dt_ms=median_dt_ms,
                interval_capture_ratio=interval_capture_ratio,
                avg_capture_ratio=avg_capture_ratio,
            )

            self.status_label.setText(text)
            print(text)

            self.frame_count_interval = 0
            self.last_report_time = now

    def closeEvent(self, event) -> None:
        self.timer.stop()
        self.receiver.close()
        super().closeEvent(event)
