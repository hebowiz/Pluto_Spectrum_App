# PlutoSDR + pyqtgraph による最小リアルタイムSAプロトタイプ
#
# 目的:
# - PlutoSDRからIQデータを連続取得する
# - FFTしてスペクトルをリアルタイム表示する
# - 現時点ではDCマスクなし、振幅単位は未正規化の相対dBのまま
# - フレームレートを確認する
#
# 前提:
# - pyadi-iio, numpy, pyqtgraph がインストール済み
# - PlutoSDR が接続済み
#
# メモ:
# - 今は「まず気持ちよく表示できるか」の確認が主目的
# - dBFS化、校正、複数トレース、ウォーターフォールは後段で追加する

import sys
import time
from dataclasses import dataclass

import adi
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets


@dataclass
class SpectrumConfig:
    center_freq_hz: int = 2_440_000_000
    sample_rate_hz: int = 10_000_000
    rx_bandwidth_hz: int = 10_000_000
    rx_buffer_size: int = 4096
    rx_gain_db: int = 40
    update_interval_ms: int = 0  # 0で可能な限り速く回す


class PlutoSpectrumAnalyzer:
    """PlutoSDRからIQを取得してFFTスペクトルを返す簡易アナライザ。"""

    def __init__(self, config: SpectrumConfig) -> None:
        self.config = config
        self.sdr = adi.Pluto()

        # Pluto設定
        self.sdr.rx_lo = config.center_freq_hz
        self.sdr.sample_rate = config.sample_rate_hz
        self.sdr.rx_rf_bandwidth = config.rx_bandwidth_hz
        self.sdr.rx_buffer_size = config.rx_buffer_size
        self.sdr.gain_control_mode_chan0 = "manual"
        self.sdr.rx_hardwaregain_chan0 = config.rx_gain_db

        # ループ外に出せるものは先に作っておく
        self.window = np.hanning(config.rx_buffer_size)
        self.freq_axis_hz = np.fft.fftshift(
            np.fft.fftfreq(config.rx_buffer_size, d=1.0 / config.sample_rate_hz)
        )
        self.freq_axis_abs_mhz = (
            self.freq_axis_hz + config.center_freq_hz
        ) / 1e6

    def acquire_spectrum(self) -> np.ndarray:
        """IQを1フレーム取得し、相対dBスペクトルを返す。"""
        iq = self.sdr.rx()

        # ダイレクトコンバージョン由来のDCオフセットを少し抑える
        iq = iq - np.mean(iq)

        # 窓関数
        iq_windowed = iq * self.window

        # FFT
        spectrum = np.fft.fftshift(np.fft.fft(iq_windowed))

        # 未正規化の相対dB
        power_db = 20.0 * np.log10(np.abs(spectrum) + 1e-12)
        return power_db

    def close(self) -> None:
        """明示的に破棄する。"""
        if hasattr(self, "sdr"):
            del self.sdr


class RealtimeSpectrumWindow(QtWidgets.QMainWindow):
    """リアルタイムスペクトル表示ウィンドウ。"""

    def __init__(self, analyzer: PlutoSpectrumAnalyzer) -> None:
        super().__init__()
        self.analyzer = analyzer
        self.config = analyzer.config

        self.setWindowTitle("PlutoSDR Real-Time Spectrum Prototype")
        self.resize(1200, 700)

        # FPS計測用
        self.frame_count_total = 0
        self.frame_count_interval = 0
        self.start_time = time.perf_counter()
        self.last_report_time = self.start_time

        self._build_ui()
        self._build_timer()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        layout = QtWidgets.QVBoxLayout(central)

        # 上部ステータス表示
        self.status_label = QtWidgets.QLabel()
        self.status_label.setText(
            self._make_status_text(interval_fps=0.0, avg_fps=0.0)
        )
        layout.addWidget(self.status_label)

        # pyqtgraph設定
        pg.setConfigOptions(antialias=False)

        # プロット
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("w")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel("bottom", "Frequency", units="MHz")
        self.plot_widget.setLabel("left", "Amplitude", units="dB")
        self.plot_widget.setTitle("PlutoSDR Real-Time FFT Spectrum")
        self.plot_widget.setXRange(
            self.analyzer.freq_axis_abs_mhz[0],
            self.analyzer.freq_axis_abs_mhz[-1],
            padding=0.0,
        )
        self.plot_widget.setYRange(0, 140)

        self.curve = self.plot_widget.plot(
            self.analyzer.freq_axis_abs_mhz,
            np.zeros(self.config.rx_buffer_size),
            pen=pg.mkPen(width=1),
        )

        layout.addWidget(self.plot_widget)

    def _build_timer(self) -> None:
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_spectrum)
        self.timer.start(self.config.update_interval_ms)

    def _make_status_text(self, interval_fps: float, avg_fps: float) -> str:
        return (
            f"Center: {self.config.center_freq_hz / 1e6:.3f} MHz    "
            f"Span: {self.config.sample_rate_hz / 1e6:.3f} MHz    "
            f"RBW(base): {self.config.sample_rate_hz / self.config.rx_buffer_size:.1f} Hz/bin    "
            f"Gain: {self.config.rx_gain_db} dB    "
            f"FPS(interval): {interval_fps:.2f}    "
            f"FPS(avg): {avg_fps:.2f}"
        )

    def update_spectrum(self) -> None:
        power_db = self.analyzer.acquire_spectrum()
        self.curve.setData(self.analyzer.freq_axis_abs_mhz, power_db)

        # FPS更新
        self.frame_count_total += 1
        self.frame_count_interval += 1

        now = time.perf_counter()
        elapsed_since_last = now - self.last_report_time

        if elapsed_since_last >= 1.0:
            interval_fps = self.frame_count_interval / elapsed_since_last
            avg_fps = self.frame_count_total / (now - self.start_time)

            self.status_label.setText(
                self._make_status_text(interval_fps=interval_fps, avg_fps=avg_fps)
            )
            print(
                f"FPS(interval): {interval_fps:.2f}, "
                f"FPS(avg): {avg_fps:.2f}"
            )

            self.frame_count_interval = 0
            self.last_report_time = now

    def closeEvent(self, event) -> None:
        """終了時にPlutoを明示的に破棄する。"""
        self.timer.stop()
        self.analyzer.close()
        super().closeEvent(event)


def main() -> int:
    config = SpectrumConfig(
        center_freq_hz=2_440_000_000,
        sample_rate_hz=10_000_000,
        rx_bandwidth_hz=10_000_000,
        rx_buffer_size=4096,
        rx_gain_db=40,
        update_interval_ms=0,
    )

    app = pg.mkQApp("PlutoSDR Real-Time Spectrum Prototype")
    analyzer = PlutoSpectrumAnalyzer(config)
    window = RealtimeSpectrumWindow(analyzer)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())