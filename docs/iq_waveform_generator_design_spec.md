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
- 送信側SRRC pulse shaping

### 3 Mbps

- 8DPSK
- Differential phase mapping
- 送信側SRRC pulse shaping

代表パラメータ:

- SRRC roll-off
  - 初期値: 0.4
- Filter span
  - 可変
- DPSK Power
  - GFSK部に対する相対dBで設定可能

用語は次のように統一する。

- 送信波形に適用するパルス整形フィルタはSRRC（Square Root Raised Cosine）
- 送信側SRRCと受信側SRRCを合成した総合特性がRC（Raised Cosine）
- GUIで単に「RRC」と表示して実装上の意味を曖昧にしない

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
Gaussian filterおよびSRRC filterの過渡応答を考慮し、各区間を単純なぶつ切り連結にしない。

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
    srrc_rolloff=0.4,
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
- SRRC filter
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

---

## 26. 既存VSA資産の再利用方針

現行VSAには、Bluetooth BRおよび2-DH1 / 3-DH1の解析試験用IQ生成処理が存在する。
パケット構築、GFSK、DPSK mapping、SRRC、whitening、CRCなどはWaveform Engineの初期実装へ再利用できる。

ただし、既存処理は解析テスト用fixtureとして以下を含むため、そのまま送信Backendへ接続しない。

- 固定長・固定PRBS-9 payload
- 記録区間内のpacket start位置
- CFO
- AWGN
- 固定振幅
- 解析用metadata

再利用時は次のレイヤーへ分離する。

1. Packet Builder
2. Ideal Modulator
3. Filter / Guard / Power Transition
4. Impairment
5. Recording Layout（packet前後のidleや反復間隔）
6. Hardware Backend

理想波形生成時の既定値はCFO、AWGN、IQ imbalanceなどを無効とし、Impairmentを明示的に有効化した場合だけ付加する。

---

## 27. IQ振幅・スケーリング契約

Waveform Engineが返すIQは、機器固有のDAC codeではなく、正規化された複素ベースバンド波形とする。

基本契約:

- dtype: `complex64`
- 無次元のnormalized IQ
- `max(abs(iq)) <= 1.0`
- 推奨既定peak: -3 dBFS程度
- clippingは暗黙に行わず、生成エラーまたは警告として扱う
- peak、RMS、PAPR、適用backoffをmetadataに保存する

「PlutoとR&Sで同一IQを使用する」とは、同じ正規化IQ配列とsample rateを入力に使うことを意味する。
実際の転送時には各Backendが必要な量子化、整数化、byte order、container化を行うため、機器へ送るbyte列まで同一である必要はない。

絶対RF出力レベルはIQ振幅だけから保証しない。
PlutoではTX hardware gain、周波数依存損失、外部ATT/Gain、個体差を含む校正が必要であり、設定値と実測dBmを区別する。

---

## 28. Bluetooth準拠Profileと実験Profile

規格相当の波形と、研究・デバッグ用に任意変更した波形を混同しない。

- Standard Profile
  - packet構造、symbol rate、guard、mapping、filter、power transitionなどを規定値へ固定または制約する
- Experimental Profile
  - symbol rate、SPS、filter、guard、power差、独自packet typeなどを変更可能にする

GUIとmetadataには、どちらのProfileで生成したかを必ず表示・記録する。
規定値から変更された波形をBluetooth準拠波形として表示しない。

---

## 29. 送信シーケンスとcyclic境界

送信対象はpacket本体だけでなく、次の構造を持つRecording Layoutとして生成する。

```text
Pre Idle -> Packet -> Post Idle / Inter-Packet Gap
```

Plutoのcyclic送信では配列末尾から先頭へ連続して反復されるため、境界で以下を満たすこと。

- packet同士が意図せず直結しない
- 設定したinter-packet gapが含まれる
- 境界で不必要な振幅・位相ジャンプを発生させない
- RF OFF相当区間はゼロIQまたは定義済みenvelopeで表現する

PCからのAPI呼び出し時刻を基準にした単発バースト開始は厳密なリアルタイムタイミングを保証しない。
初期MVPでは「idleを含む完成波形のcyclic再生」を優先し、厳密なsingle-shot timingは別機能として扱う。

---

## 30. Backend capabilityと設定検証

Backendは共通APIを持つが、すべての機器が同じ能力を持つとは仮定しない。

各Backendは少なくとも次を公開する。

- 対応sample rate範囲
- 対応RF frequency範囲
- 最大waveform sample数またはARB memory
- IQ量子化形式
- waveform長・alignment制約
- cyclic / single-shot対応可否
- RF levelまたはhardware gainの設定範囲
- 接続状態と現在設定のreadback可否

生成前または送信前にcapability validationを行い、暗黙の丸めを避ける。
丸めやresamplingが必要な場合は、実際に適用された値をWaveformResultまたは送信結果へ記録する。

---

## 31. 送信状態管理とRF安全設計

GUIとBackendの間に、少なくとも次の状態を持つ送信state machineを設ける。

```text
Disconnected -> Connected -> Ready -> Transmitting
                                  -> Error
```

安全上の要件:

- 起動時はRF出力OFF
- 接続直後は十分低いTX gain / levelを初期値とする
- Transmit開始前にcenter frequency、sample rate、gain / level、cyclic設定を確認表示する
- Stop、ウィンドウ終了、例外、接続断でbuffer破棄とRF停止を試みる
- 出力中の波形更新は、一旦停止してbufferを破棄してから再転送する
- 現在送信中であることをGUI上で明確に表示する

外部ATT/Gainは送信設定metadataへ保存するが、Backendのhardware gainとは別項目として扱う。

---

## 32. Pluto Backend補足

Pluto MVPはpyadi-iioのTX bufferを使用する。

- 反復パケットはcyclic bufferを基本とする
- 更新時は既存TX bufferを明示的にdestroyしてから再送する
- Backendでnormalized IQをPluto向けDAC scaleへ変換する
- peak超過、NaN、Inf、空波形を送信前に拒否する
- 実際に設定されたsample rate、RF bandwidth、LO、hardware gainをreadbackして記録する

cyclic bufferでは初回転送後の反復に継続的なUSB転送を必要としないため、反復波形ではUSB throughputを主制約としない。
一方、異なるpacketを連続してリアルタイム供給する用途はUSB転送、host scheduling、buffer underrunの影響を受けるため、MVP後の別課題とする。

---

## 33. R&S SMCV100B Backend補足

SMCV100B Backendは、共通のnormalized IQを機器対応ARB形式へ変換し、転送と再生設定を担当する。

実装前に対象実機について次を確認する。

- ARB機能および必要option / license
- 対応waveform file形式とtag / header仕様
- I/Qのbit depth、byte order、full-scale定義
- ARB sample clock範囲
- waveform memoryとminimum / alignment制約
- SCPIによるupload、select、play、stop、RF ON/OFF手順

機器内蔵Bluetooth generatorの再現を前提にせず、ARBへ自前IQを入力する方式とする。
SCPI error queueを各主要操作後に確認し、転送成功と再生開始を分けて報告する。

---

## 34. 検証方針と完了条件

送信機能は、見た目だけでなく段階的に検証する。

### 34.1 Unit Test

- packet bit列、air order、CRC、whitening
- GFSK modulation indexと周波数偏移
- DPSK differential mapping
- SRRC impulse responseとgroup delay
- guard長、region境界、sample数
- peak / RMS / PAPRとclipping検出

### 34.2 Offline Round Trip

生成したWaveformResultを現行Pluto VSAへ直接入力し、次を確認する。

- pattern match
- symbol countとsymbol列
- mapping
- CFOが無効ならほぼ0 Hz
- constellation / EVM / deviation
- packet regionと解析Result Rangeの整合

### 34.3 Pluto実機Loopback

Pluto TXから有線ATTを介して受信し、offline結果との差を確認する。

- spectrum
- power
- CFO / drift
- symbol rate error
- deviationまたはEVM
- cyclic境界とinter-packet gap

### 34.4 R&S比較

同一のnormalized IQをPlutoとSMCV100Bから送信し、同一受信系で比較する。
Backend固有の量子化・level差を分離し、symbol列、spectrum、EVM、deviationが許容範囲内で一致することを確認する。

MVPの完了条件は、2-DH1および3-DH1をPlutoからcyclic送信し、現行VSAで安定してpattern matchとsymbol復調ができることとする。

---

## 35. 将来対応: 高速化BT-like波形とBluetooth HDT

初期MVP完了後、次の2系統へ拡張する。

### 35.1 BT-like Rate-Scaled Profile

Bluetooth BR/EDRのpacket構造、mapping、filter、guardなどを基礎とし、symbol rateだけを任意倍率へ変更できる実験Profileを用意する。

用途:

- 既知Bluetooth波形を高速化した独自信号の生成・解析
- symbol rate変更に対する帯域、EVM、timing recoveryの評価
- 受信VSAの高速変調解析試験

注意事項:

- Bluetooth BR/EDRの規格上のsymbol rateは1 MSym/sであり、本ProfileはBluetooth準拠とは表示しない
- symbol rate変更時はsample rate、SPS、Gaussian / SRRC filter帯域、guard sample数、analysis bandwidthを連動させる
- 時間で規定する値とsymbol数で規定する値を区別し、倍率変更時の挙動をmetadataへ記録する
- packet bit列を維持した単純時間圧縮と、filter・guardを再設計した波形を区別できるようにする

### 35.2 Bluetooth High Data Throughput Profile

Bluetooth HDTは、既存BR/EDRを単純に高速再生する機能として扱わず、Bluetooth LEの新しいPHY / protocol拡張として独立実装する。

設計方針:

- `BluetoothHDTProfile`などの独立ProfileとPacket Builderを設ける
- 公開仕様のrevisionをProfile metadataへ保存する
- PHY、packet format、coding、modulation、symbol mapping、filter、channel bandwidthを仕様revisionごとに定義する
- Draft中の値を共通Engineへ固定値として埋め込まない
- 仕様改訂に備え、versioned parameter setとgolden vectorを用意する
- HDT対応VSA解析も送信Profileと同じrevision定義を参照する

2026年8月時点でBluetooth SIGが公開しているHDT文書はDraftであり、Core Specification 6.2を基礎とする変更仕様として公開されている。
正式採用まで仕様変更の可能性があるため、初期実装はExperimental / Draftと明示し、標準準拠判定には使用しない。

### 35.3 共通化する範囲

高速化BT-like波形とHDTで共通利用するもの:

- normalized IQ / WaveformResult
- sample rate / SPS管理
- filter基盤
- impairment
- Recording Layout
- preview / analysis表示
- Pluto / R&S Backend

個別Profileへ隔離するもの:

- packet構造
- bit order / whitening / CRC / FEC
- modulationとmapping
- symbol rateとchannel定義
- 規格固有の測定条件および合否判定
