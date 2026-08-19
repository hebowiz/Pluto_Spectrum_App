# IQ Waveform Generator / Bluetooth EDR Support
## 仮設計仕様（Codex向け）

## 1. 目的

R&S SMCV100B と ADALM-Pluto で共用できる、PythonベースのIQ波形生成GUIツールを作成する。

初期ターゲットは Bluetooth BR/EDR。
特に、1パケット内で GFSK と π/4-DQPSK / 8DPSK が切り替わる EDR パケットを自前生成し、IQ波形として可視化・保存・送信できることを目標とする。

Bluetooth専用オプションを使用しない R&S SMCV100B でも、ARB用IQ波形を入力することでEDRパケットを模擬できる構成を想定する。

---

## 2. 基本方針

- GUI: PySide6
- 波形表示: pyqtgraph
- 数値処理: NumPy / SciPy
- 内部IQ表現: `numpy.ndarray` の complex 系
- 波形生成エンジンとGUIは分離する
- 出力先依存処理はBackendとして分離する
- 同一IQ波形を Pluto / R&S の両方へ出力可能にする
- GUI表示は、設定値から疑似表示するのではなく「実際に生成したIQ配列」を解析して表示する
- 色使い・最終デザインは現時点では固定しない

---

## 3. 全体アーキテクチャ

```text
GUI
 │
 └─ WaveformParameters
        │
        v
Waveform Engine
 ├─ Packet Builder
 ├─ GFSK Modulator
 ├─ EDR DPSK Modulator
 ├─ Filter Processing
 ├─ Guard / Power Transition
 └─ Impairment Processing
        │
        v
complex IQ waveform
        │
        ├─ Preview / Analysis
        │   ├─ I/Q vs Time
        │   ├─ Envelope / Power
        │   ├─ Instantaneous Frequency
        │   ├─ Phase
        │   ├─ Spectrum
        │   └─ Constellation
        │
        └─ Output Backend
            ├─ Pluto
            └─ R&S SMCV100B
```

---

## 4. 初期対応波形

### 4.1 Bluetooth BR

- GFSK
- Gaussian pulse shaping
- Bluetooth相当の変調指数を指定可能
- Access Code / Header を含むパケット波形生成

### 4.2 Bluetooth EDR

初期対応対象:

- 2-DH1
  - BR部: GFSK
  - EDR部: π/4-DQPSK
- 3-DH1
  - BR部: GFSK
  - EDR部: 8DPSK

将来的に DH3 / DH5 へ拡張する。

EDRパケット全体を、変調方式ごとに別送信するのではなく、1本の連続したIQ時系列として生成する。

概念構成:

```text
Access Code / Header
        |
      GFSK
        |
      Guard
        |
   EDR Sync
        |
π/4-DQPSK または 8DPSK
        |
     Trailer
```

---

## 5. サンプルレート

Bluetooth BR/EDR のシンボルレートは基本 1 Msymbol/s とする。

初期デフォルト:

- SPS: 8 samples/symbol
- Sample Rate: 8 MS/s

GUI上で変更可能とする。

想定候補:

- 4 SPS
- 8 SPS
- 16 SPS

3 Mbps EDRでもシンボルレートは1 Msymbol/sのため、8 SPSなら8 MS/sで扱う。

---

## 6. GFSK生成

処理イメージ:

```text
bit sequence
  ↓
NRZ (+1 / -1)
  ↓
Gaussian Filter
  ↓
instantaneous frequency
  ↓
phase integration
  ↓
complex IQ
```

代表パラメータ:

- Gaussian BT
  - 初期値: 0.5
- Modulation Index `h`
  - 初期値候補: 0.32
- GFSK Power
  - 基準 0 dB として扱う

GFSKも最終的には複素ベースバンドIQとして生成する。

---

## 7. EDR PSK生成

### 2 Mbps

- π/4-DQPSK
- Differential phase mapping
- RRC / SRRC pulse shaping

### 3 Mbps

- 8DPSK
- Differential phase mapping
- RRC / SRRC pulse shaping

代表パラメータ:

- RRC roll-off
  - 初期値: 0.4
- Filter span
  - 可変
- DPSK Power
  - GFSK部に対する相対dBで設定可能

---

## 8. Guard処理

GFSK部とDPSK部の間にGuard期間を設ける。

初期値:

- Guard Duration: 約5 us

Guardは単純に「RF OFF」と固定せず、振幅エンベロープとして独立制御可能にする。

例:

- Power保持
- 一定レベルまで低下
- Power dip
- 線形遷移
- 任意エンベロープへの将来拡張

GUI上では Guard Power / Guard Depth などを調整可能にする。

注意:
Gaussian filterおよびRRC filterの過渡応答を考慮し、各区間を単純なぶつ切り連結にしない。

---

## 9. GFSK / DPSK パワー差

GFSK部とDPSK部で独立に振幅を設定可能にする。

パラメータ例:

```text
GFSK Power : 0.0 dB
DPSK Power : -2.0 dB
```

相対dB値をIQ振幅へ変換して適用する。

```python
amplitude_ratio = 10 ** (power_db / 20)
```

---

## 10. Impairment機能

理想波形生成とは別レイヤーとして実装する。

初期実装必須ではないが、拡張しやすい構造とする。

将来候補:

- Carrier Frequency Offset
- Phase Offset
- IQ Gain Imbalance
- IQ Phase Imbalance
- DC Offset
- AWGN
- Amplitude Error
- Phase Error
- EVM相当の擾乱

構造:

```text
Ideal waveform
      ↓
Impairment Processor
      ↓
Final IQ
```

---

## 11. GUIレイアウト案

### 左側

設定パネル。

想定カテゴリ:

- Packet
- Sampling
- GFSK
- Guard
- DPSK
- Impairment
- Output

### 右側

pyqtgraphによるリアルタイム波形表示。

主要表示:

1. I / Q vs Time
2. Envelope / Power vs Time
3. Instantaneous Frequency vs Time
4. Phase vs Time
5. Spectrum
6. Constellation

主要グラフは可能な範囲で同時表示する。

---

## 12. グラフ連動

時間軸を持つグラフはX軸を同期する。

対象:

- I/Q
- Envelope / Power
- Instantaneous Frequency
- Phase

例:

PowerグラフでGuard付近を拡大した場合、他の時間波形も同じ範囲へ連動する。

pyqtgraphのViewBox / XLink等を利用する。

---

## 13. パケット区間表示

時間波形上にパケット構造をオーバーレイ表示する。

例:

```text
| Access Code | Header | Guard | Sync | Payload | Trailer |
|     GFSK     | GFSK   |       |        DPSK           |
```

実装候補:

- `LinearRegionItem`
- `InfiniteLine`
- Text label

区間表示は薄い背景や境界線などで表現し、色仕様は後で決める。

---

## 14. 選択・ズーム機能

波形の任意区間を選択・拡大可能にする。

特に以下を確認しやすくする:

- GFSK終端
- Guard
- GFSK → DPSK遷移
- DPSK開始点
- 特定シンボル周辺

可能であれば選択領域に応じて、Constellation等の表示対象も連動させる。

---

## 15. Constellation表示

パケット全体を無条件に表示するとGFSK部が混在するため、表示対象区間を限定できるようにする。

例:

- EDR Sync
- EDR Payload
- 任意選択区間

表示対象となるPSK部だけを抽出し、適切なシンボルタイミングで表示する。

---

## 16. リアルタイム更新

GUIパラメータ変更時に波形を自動再生成する。

処理例:

```text
parameter changed
      ↓
debounce
      ↓
waveform regeneration
      ↓
analysis
      ↓
pyqtgraph update
```

目安:

- debounce: 100～200 ms程度
- 重い処理は必要に応じてWorker Threadへ分離

GUI操作をブロックしないこと。

---

## 17. ステータス表示

以下を常時表示可能にする。

- Sample Rate
- SPS
- Waveform Length
- Sample Count
- Peak Level
- RMS Level
- PAPR

例:

```text
Fs      : 8.000 MS/s
SPS     : 8
Length  : 0.384 ms
Samples : 3072
Peak    : -1.2 dBFS
RMS     : -4.8 dBFS
PAPR    : 3.6 dB
```

---

## 18. IQ内部データ形式

基本形式:

```python
waveform: np.ndarray
dtype: np.complex64
sample_rate_hz: float
```

必要に応じて内部計算は complex128 でもよいが、出力時は complex64 を基本とする。

波形本体とは別にmetadataを保持する。

例:

```python
WaveformResult:
    iq
    sample_rate_hz
    packet_regions
    symbol_boundaries
    metadata
```

---

## 19. ファイル入出力

初期候補:

- NumPy `.npy`
- raw complex IQ
- CSV（デバッグ用途）
- R&S向けARB形式への変換

将来的にロードしたIQファイルの可視化にも対応できる構造が望ましい。

---

## 20. Pluto Backend

Pluto出力はBackendとして分離する。

想定:

- pyadi-iio
- Center Frequency設定
- Sample Rate設定
- TX attenuation / gain設定
- Cyclic buffer
- Single / Continuous TX

例:

```python
pluto_backend.transmit(
    iq=waveform,
    sample_rate_hz=8e6,
    center_frequency_hz=2.441e9
)
```

---

## 21. R&S SMCV100B Backend

Bluetooth専用オプションは使用しない前提。

自前生成したIQ波形をARBへ読み込み、RF出力する。

Backendで以下を担当する:

- IQ / ARB waveform file生成
- SMCV100Bへの転送
- ARB sample rate設定
- RF center frequency設定
- RF level設定
- ARB再生開始 / 停止

可能であればSCPI制御対応。

R&S依存形式・通信処理はWaveform Engineから完全に分離する。

---

## 22. クラス構成案

```text
core/
  waveform_params.py
  packet_builder.py
  gfsk.py
  dpsk.py
  filters.py
  guard.py
  impairment.py
  waveform_engine.py
  analysis.py

backends/
  base.py
  pluto.py
  rohde_schwarz.py
  file_export.py

gui/
  main_window.py
  parameter_panel.py
  waveform_view.py
  spectrum_view.py
  constellation_view.py
  status_bar.py
```

---

## 23. コアAPI案

```python
params = WaveformParameters(
    packet_type="2-DH1",
    sample_rate_hz=8e6,
    gfsk_bt=0.5,
    gfsk_mod_index=0.32,
    guard_duration_s=5e-6,
    gfsk_power_db=0.0,
    dpsk_power_db=-2.0,
    rrc_rolloff=0.4,
)

result = generate_waveform(params)

iq = result.iq
```

GUI、Pluto、R&S、ファイル保存はすべてこの結果を利用する。

---

## 24. MVP

まず以下までを実装する。

### Waveform Engine

- BR GFSK生成
- 2-DH1生成
- 3-DH1生成
- Gaussian filter
- Guard
- RRC filter
- GFSK / DPSK power差
- 8 SPS default
- complex64 IQ出力

### GUI

- PySide6
- パラメータ編集
- pyqtgraph
- I/Q表示
- Power / Envelope表示
- Instantaneous Frequency表示
- Spectrum表示
- Constellation表示
- 時間軸連動
- パケット区間表示
- パラメータ変更時の自動再生成

### Output

- IQファイル保存
- Pluto送信

R&S SMCV100B BackendはMVP後に追加してもよい。

---

## 25. 設計上の重要事項

1. GUIと波形生成ロジックを分離する。
2. ハードウェア固有処理をBackendへ隔離する。
3. PlutoとR&Sで同一のIQデータを使用可能にする。
4. 可視化対象は実際に生成されたIQデータとする。
5. フィルタ過渡応答を考慮し、GFSK / Guard / DPSKを単純連結しない。
6. Bluetooth専用ロジックを過度にGUIへ埋め込まない。
7. 将来的にBluetooth以外の任意デジタル変調へ拡張可能な設計にする。
8. 最終的な色・テーマ・細かなUIデザインは後で決定する。
