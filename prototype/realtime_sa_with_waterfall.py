import sys
import time
from dataclasses import dataclass

import adi
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

MAX_DISPLAY_SPAN_HZ = 55_000_000

@dataclass
class SpectrumConfig:
    # ===== ユーザー設定値（初期値）=====
    center_freq_hz: int = 2_440_000_000

    # ユーザーが指定するのは表示帯域幅
    display_span_hz: int = 55_000_000

    # 左右それぞれのガード比率
    guard_ratio: float = 0.04

    # FFTサイズ（= rx_buffer_size）
    fft_size: int = 4096
    
    rx_gain_db: int = 40

    # 欠落限界確認のため、デフォルトは最速更新
    update_interval_ms: int = 0

    waterfall_history: int = 300
    waterfall_decimation: int = 4  # 周波数方向のみ間引き

    # 欠落候補判定用
    drop_threshold_factor: float = 2.5
    drop_judge_window: int = 30
    
    def __post_init__(self):
        if self.display_span_hz > MAX_DISPLAY_SPAN_HZ:
            print(f"[WARN] display_span clipped to {MAX_DISPLAY_SPAN_HZ / 1e6:.1f} MHz")
            self.display_span_hz = MAX_DISPLAY_SPAN_HZ

    # ===== 内部計算値 =====
    @property
    def sample_rate_hz(self) -> int:
        """表示帯域幅から内部受信帯域幅を計算する。"""
        return int(round(self.display_span_hz / (1.0 - 2.0 * self.guard_ratio)))

    @property
    def rx_bandwidth_hz(self) -> int:
        """現時点では sample_rate と同じ値を使う。"""
        return self.sample_rate_hz
    
    @property
    def rx_buffer_size(self) -> int:
        """現時点ではFFTサイズと同一とする。"""
        return self.fft_size


class PlutoSpectrumAnalyzer:
    def __init__(self, config: SpectrumConfig) -> None:
        self.config = config
        self.sdr = adi.Pluto()

        self.sdr.rx_lo = config.center_freq_hz
        self.sdr.sample_rate = config.sample_rate_hz
        self.sdr.rx_rf_bandwidth = config.rx_bandwidth_hz
        self.sdr.rx_buffer_size = config.rx_buffer_size
        self.sdr.gain_control_mode_chan0 = "manual"
        self.sdr.rx_hardwaregain_chan0 = config.rx_gain_db

        self.window = np.hanning(config.fft_size)

        # 内部受信帯域全体の周波数軸
        self.freq_axis_hz = np.fft.fftshift(
            np.fft.fftfreq(config.fft_size, d=1.0 / config.sample_rate_hz)
        )
        self.freq_axis_abs_ghz = (
            self.freq_axis_hz + config.center_freq_hz
        ) / 1e9

        # 表示有効帯域（中央部）を切り出すためのbin範囲
        n = config.fft_size
        guard_bins_each_side = int(round(n * config.guard_ratio))
        self.display_slice = slice(guard_bins_each_side, n - guard_bins_each_side)

        self.freq_axis_display_ghz = self.freq_axis_abs_ghz[self.display_slice]
        self.freq_axis_display_ghz_dec = self.freq_axis_display_ghz[
            ::config.waterfall_decimation
        ]

    def acquire_spectrum_full(self) -> np.ndarray:
        """内部受信帯域全体のFFTスペクトルを返す。"""
        iq = self.sdr.rx()
        iq = iq - np.mean(iq)
        iq_windowed = iq * self.window
        spectrum = np.fft.fftshift(np.fft.fft(iq_windowed))
        power_db = 20.0 * np.log10(np.abs(spectrum) + 1e-12)
        return power_db

    def acquire_spectrum_display(self) -> np.ndarray:
        """表示有効帯域のみ切り出したスペクトルを返す。"""
        power_db_full = self.acquire_spectrum_full()
        power_db_display = power_db_full[self.display_slice]
        return power_db_display

    def close(self) -> None:
        del self.sdr


class RealtimeSpectrumWindow(QtWidgets.QMainWindow):
    def __init__(self, analyzer: PlutoSpectrumAnalyzer) -> None:
        super().__init__()
        self.analyzer = analyzer
        self.config = analyzer.config

        self.setWindowTitle("PlutoSDR Real-Time Spectrum Prototype")
        self.resize(1280, 800)

        # FPS計測
        self.frame_count_total = 0
        self.frame_count_interval = 0
        self.start_time = time.perf_counter()
        self.last_report_time = self.start_time

        # 欠落候補検出用
        self.prev_frame_time = None
        self.frame_dt_history: list[float] = []
        self.drop_count = 0

        # 受信率計算用
        self.received_samples_total = 0
        self.received_samples_interval = 0

        wf_width = len(self.analyzer.freq_axis_display_ghz_dec)
        self.waterfall_buffer = np.full(
            (self.config.waterfall_history, wf_width),
            0.0,
            dtype=np.float32,
        )

        self.y_min = 0.0
        self.y_max = 140.0

        # update_interval_ms=0 のときは実時間軸が決めにくいので、
        # ここでは暫定的に 1/60 s を初期値にしておく。
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

        # ===== ウォーターフォール（上）=====
        self.waterfall_plot = pg.PlotWidget()
        self.waterfall_plot.setBackground("k")
        self.waterfall_plot.setLabel("bottom", "Frequency [GHz]")
        self.waterfall_plot.setLabel("left", "Time [s]")
        self.waterfall_plot.getViewBox().invertY(True)

        self.waterfall_img = pg.ImageItem()
        self.waterfall_plot.addItem(self.waterfall_img)

        self.waterfall_img.setImage(
            self.waterfall_buffer,
            autoLevels=False,
            axisOrder="row-major",
        )

        x_min = self.analyzer.freq_axis_display_ghz_dec[0]
        x_max = self.analyzer.freq_axis_display_ghz_dec[-1]

        self.waterfall_img.setRect(
            QtCore.QRectF(
                x_min,
                0.0,
                x_max - x_min,
                self.waterfall_time_span_s,
            )
        )

        self.waterfall_plot.setXRange(x_min, x_max, padding=0.0)
        self.waterfall_plot.setYRange(0.0, self.waterfall_time_span_s, padding=0.0)

        lut = pg.colormap.get("viridis").getLookupTable()
        self.waterfall_img.setLookupTable(lut)
        self.waterfall_img.setLevels((self.y_min, self.y_max))

        splitter.addWidget(self.waterfall_plot)

        # ===== スペクトル（下）=====
        self.spectrum_plot = pg.PlotWidget()
        self.spectrum_plot.setBackground("k")
        self.spectrum_plot.showGrid(x=True, y=True, alpha=0.2)

        self.spectrum_plot.getAxis("left").setPen("w")
        self.spectrum_plot.getAxis("bottom").setPen("w")

        self.spectrum_plot.setLabel("bottom", "Frequency [GHz]")
        self.spectrum_plot.setLabel("left", "Amplitude [dB]")

        self.spectrum_plot.setXRange(
            self.analyzer.freq_axis_display_ghz[0],
            self.analyzer.freq_axis_display_ghz[-1],
            padding=0.0,
        )
        self.spectrum_plot.setYRange(self.y_min, self.y_max)

        self.spectrum_curve = self.spectrum_plot.plot(
            self.analyzer.freq_axis_display_ghz,
            np.zeros(len(self.analyzer.freq_axis_display_ghz)),
            pen=pg.mkPen("y", width=1.5),
        )

        splitter.addWidget(self.spectrum_plot)
        splitter.setSizes([500, 300])

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
            f"RBW(base): {self.config.sample_rate_hz / self.config.rx_buffer_size:.1f} Hz/bin    "
            f"Guard: {self.config.guard_ratio * 100:.1f}% each side    "
            f"Gain: {self.config.rx_gain_db} dB    "
            f"Drops: {self.drop_count}    "
            f"Frame dt(median): {median_dt_ms:.2f} ms    "
            f"Capture(interval): {interval_capture_ratio * 100:.2f}%    "
            f"Capture(avg): {avg_capture_ratio * 100:.2f}%    "
            f"FPS(interval): {interval_fps:.2f}    "
            f"FPS(avg): {avg_fps:.2f}"
        )

    def update_spectrum(self) -> None:
        # ===== 欠落候補検出用のフレーム時間計測 =====
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

        # ===== スペクトル取得 =====
        power_db_display = self.analyzer.acquire_spectrum_display()
        
        self.received_samples_total += self.config.rx_buffer_size
        self.received_samples_interval += self.config.rx_buffer_size

        # ウォーターフォール用に周波数方向だけ間引く
        power_db_dec = power_db_display[::self.config.waterfall_decimation]

        self.waterfall_buffer[:-1] = self.waterfall_buffer[1:]
        self.waterfall_buffer[-1] = power_db_dec.astype(np.float32)

        self.waterfall_img.setImage(
            self.waterfall_buffer,
            autoLevels=False,
            axisOrder="row-major",
        )

        self.spectrum_curve.setData(
            self.analyzer.freq_axis_display_ghz,
            power_db_display,
        )

        # ===== FPS表示更新 =====
        self.frame_count_total += 1
        self.frame_count_interval += 1

        now = time.perf_counter()
        if now - self.last_report_time >= 1.0:
            interval_elapsed = now - self.last_report_time
            total_elapsed = now - self.start_time

            interval_fps = self.frame_count_interval / interval_elapsed
            avg_fps = self.frame_count_total / total_elapsed

            # ★ここ追加（受信率計算）
            interval_capture_ratio = (
                self.received_samples_interval
                / (self.config.sample_rate_hz * interval_elapsed)
            )

            avg_capture_ratio = (
                self.received_samples_total
                / (self.config.sample_rate_hz * total_elapsed)
            )

            if len(self.frame_dt_history) >= self.config.drop_judge_window:
                recent = self.frame_dt_history[-self.config.drop_judge_window :]
                median_dt_ms = float(np.median(recent) * 1000.0)
            else:
                median_dt_ms = 0.0

            # ★ここ変更（引数追加）
            text = self._make_status_text(
                interval_fps=interval_fps,
                avg_fps=avg_fps,
                median_dt_ms=median_dt_ms,
                interval_capture_ratio=interval_capture_ratio,
                avg_capture_ratio=avg_capture_ratio,
            )

            self.status_label.setText(text)
            print(text)

            # ★ここ追加（intervalリセット）
            self.received_samples_interval = 0

            self.frame_count_interval = 0
            self.last_report_time = now

    def closeEvent(self, event) -> None:
        self.timer.stop()
        self.analyzer.close()
        super().closeEvent(event)


def main() -> int:
    app = pg.mkQApp("PlutoSDR Real-Time Spectrum Prototype")
    config = SpectrumConfig()
    analyzer = PlutoSpectrumAnalyzer(config)
    window = RealtimeSpectrumWindow(analyzer)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())