# PlutoからIQデータを連続取得し、FFTスペクトルをリアルタイム更新するテストコード
# まずはDCマスクなし、振幅単位も未正規化の相対dBのままで動作確認する

import time

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

    # 周波数軸は固定なので先に作っておく
    freq_axis_hz = np.fft.fftshift(np.fft.fftfreq(RX_BUFFER_SIZE, d=1 / SAMPLE_RATE_HZ))
    freq_axis_abs_mhz = (freq_axis_hz + CENTER_FREQ_HZ) / 1e6

    # 描画初期化
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 6))
    line, = ax.plot(freq_axis_abs_mhz, np.zeros(RX_BUFFER_SIZE))

    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel("Amplitude [relative dB]")
    ax.set_title("PlutoSDR Real-Time FFT Test")
    ax.grid(True)
    ax.set_xlim(freq_axis_abs_mhz[0], freq_axis_abs_mhz[-1])
    ax.set_ylim(0, 140)  # とりあえず仮固定。必要に応じて調整

    fig.tight_layout()
    fig.show()

    frame_count = 0
    start_time = time.perf_counter()
    fps_report_time = start_time
    
    window = np.hanning(RX_BUFFER_SIZE)

    while plt.fignum_exists(fig.number):
        # IQ取得
        iq = sdr.rx()

        # 平均値を引いて軽くDCオフセットを抑える
        iq = iq - np.mean(iq)

        # 窓関数
        
        iq_windowed = iq * window

        # FFT
        spectrum = np.fft.fftshift(np.fft.fft(iq_windowed))
        power_db = 20 * np.log10(np.abs(spectrum) + 1e-12)

        # グラフ更新
        line.set_ydata(power_db)
        fig.canvas.draw()
        fig.canvas.flush_events()

        # FPS計測
        frame_count += 1
        now = time.perf_counter()
        elapsed_report = now - fps_report_time

        if elapsed_report >= 1.0:
            fps = frame_count / (now - start_time)
            print(f"Average FPS: {fps:.2f}")
            fps_report_time = now

except KeyboardInterrupt:
    print("Stopped by user.")

finally:
    plt.ioff()
    del sdr