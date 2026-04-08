# PlutoからIQデータを取得し、FFTしてスペクトルを表示するテストコード

import adi
import numpy as np
import matplotlib.pyplot as plt

# ===== 設定 =====
CENTER_FREQ_HZ = 2_440_000_000
SAMPLE_RATE_HZ = 10_000_000
RX_BANDWIDTH_HZ = 10_000_000
RX_BUFFER_SIZE = 4096
RX_GAIN_DB = 40

# ===== Pluto接続 =====
sdr = adi.Pluto()

try:
    sdr.rx_lo = CENTER_FREQ_HZ
    sdr.sample_rate = SAMPLE_RATE_HZ
    sdr.rx_rf_bandwidth = RX_BANDWIDTH_HZ
    sdr.rx_buffer_size = RX_BUFFER_SIZE
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = RX_GAIN_DB

    # IQデータ取得
    iq = sdr.rx()

    # DCオフセットを少し抑えるため平均値を引く
    iq = iq - np.mean(iq)

    # 窓関数
    window = np.hanning(len(iq))
    iq_windowed = iq * window

    # FFT
    spectrum = np.fft.fftshift(np.fft.fft(iq_windowed))

    # 振幅をdB化
    power_db = 20 * np.log10(np.abs(spectrum) + 1e-12)

    # ===== 表示用DCマスク（DC±2bin）=====
    # 元データは残し、表示用だけ編集する
    power_db_display = power_db.copy()

    center = len(power_db_display) // 2
    mask_half_width = 2  # DC±2bin → 合計5bin

    left_src = center - mask_half_width - 1
    right_src = center + mask_half_width + 1

    # 左右の隣接点の平均値で埋める
    fill_value = (power_db_display[left_src] + power_db_display[right_src]) / 2.0
    power_db_display[center - mask_half_width : center + mask_half_width + 1] = fill_value

    # 周波数軸作成（LO中心の相対周波数 → 絶対周波数）
    freq_axis_hz = np.fft.fftshift(np.fft.fftfreq(len(iq), d=1 / SAMPLE_RATE_HZ))
    freq_axis_abs_hz = freq_axis_hz + CENTER_FREQ_HZ
    freq_axis_abs_mhz = freq_axis_abs_hz / 1e6

    # プロット
    plt.figure(figsize=(12, 6))
    plt.plot(freq_axis_abs_mhz, power_db, alpha=0.5, label="Original")
    plt.plot(freq_axis_abs_mhz, power_db_display, label="Display spectrum (DC masked)")
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Amplitude [dBFS-like]")
    plt.title("PlutoSDR FFT Spectrum")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

finally:
    del sdr