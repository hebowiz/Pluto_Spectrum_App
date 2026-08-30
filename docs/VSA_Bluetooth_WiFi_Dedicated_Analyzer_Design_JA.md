# Pluto VSA Bluetooth / Wi-Fi 専用解析モード設計案

作成日: 2026-08-29  
対象: **Pluto VSA**  
実装優先順: **Bluetooth / BLE → Wi-Fi**  
ステータス: Bluetooth初期実装中

## 0. 結論

Pluto VSA の現行 Generic VSA を置き換えるのではなく、その上に **Protocol-Specific Analyzer Mode** を追加する。

専用解析モードは、1 回の IQ capture から以下をまとめて実施する。

1. packet detection / synchronization
2. packet / PHY 判別
3. RF / modulation quality measurement
4. packet bit recovery
5. `Shared Protocol Packet Analyzer` による semantic decode
6. 必要に応じた規格値との比較
7. 複数 packet の統計処理

基本構成:

```text
IQRecording
   ↓
Protocol Packet Detector / Synchronizer
   ↓
Packet Region + PHY Classification
   ↓
┌────────────────────────────┬────────────────────────────┐
│ RF / Modulation Analyzer   │ Demodulator / PHY Decoder │
│ Power / CFO / EVM / DEVM   │ air bits / PSDU           │
└────────────────────────────┴────────────────────────────┘
                  ↓                         ↓
         RF Metric Result      Shared Protocol Packet Analyzer
                  ↓                         ↓
                  └──────────┬──────────────┘
                             ↓
                 ProtocolAnalysisResult
                             ↓
              Summary / Plot / Decode / Stats
```

最初に Bluetooth 専用モードでこの共通 framework を確立し、その後 Wi-Fi へ拡張する。

---

## 1. 既存資産との関係

本設計は、リポジトリ内の以下の既存設計・実装を前提とする。

- `docs/Bluetooth_RF_Test_Packet_Design_Memo_JA.md`
  - BR / EDR / LE / HDT の test packet、変調、DEVM / EVM 等の基礎資料
- `docs/WiFi_RF_Test_Packet_Design_Memo_JA.md`
  - 802.11a/g Non-HT OFDM、802.11b DSSS/CCK、将来 HT の PHY / packet 生成資料
- `docs/shared_protocol_packet_analyzer_design.md`
  - VSG / VSA 共通 semantic packet analyzer
- `pluto_sa/vsa/model.py`
  - `IQRecording`、`VSAAnalysisResult`、`CompositeVSAAnalysisResult` 等
- `pluto_sa/vsa/profiles/bluetooth_br.py`
- `pluto_sa/vsa/profiles/bluetooth_edr.py`

重要な設計方針として、`Shared Protocol Packet Analyzer` は **packet detection、同期、復調、EVM 計算を担当しない**。

専用 VSA mode が RF / PHY 解析を担当し、得られた bit 列または PSDU を semantic analyzer へ渡す。

```text
RF / PHY analysis != Protocol semantic decode
```

この責務分離は維持する。

---

## 2. 専用解析モード共通アーキテクチャ

### 2.1 Analyzer Mode

VSA の上位モードとして以下を想定する。

```text
VSA Mode
├─ Generic VSA
├─ Bluetooth Analyzer
└─ Wi-Fi Analyzer
```

Generic VSA は従来どおり、ユーザー指定の modulation / symbol rate を解析する汎用モードとして残す。

Bluetooth / Wi-Fi mode は packet structure を理解し、同期・評価区間・decode を自動化する。

### 2.2 1 packet と複数 packet の両方を扱う

専用 mode では、1 packet の詳細解析だけでなく、連続 packet の統計評価を前提とする。

```text
Capture / Stream
    ↓
Packet 0
Packet 1
Packet 2
...
    ↓
Per-Packet Result
    ↓
Batch / Statistical Result
```

特に Bluetooth RF test では同一 test packet が繰り返されるため、複数 packet から DEVM block、Power、CFO 等を集計できる構造にする。

### 2.3 推奨共通 result model

概念モデル:

```python
@dataclass(frozen=True)
class ProtocolAnalysisResult:
    protocol: str
    phy: str | None
    packets: tuple[PacketMeasurementResult, ...]
    batch_metrics: tuple[MetricResult, ...]
    metadata: Mapping[str, object]

@dataclass(frozen=True)
class PacketMeasurementResult:
    packet_index: int
    start_sample: int
    stop_sample: int
    phy: str | None
    packet_type: str | None
    metrics: tuple[MetricResult, ...]
    plots: Mapping[str, object]
    protocol_result: PacketAnalysisResult | None
    issues: tuple[str, ...]

@dataclass(frozen=True)
class MetricResult:
    metric_id: str
    display_name: str
    value: float | None
    unit: str | None
    status: str       # INFO / PASS / FAIL / UNKNOWN
    limit_low: float | None = None
    limit_high: float | None = None
```

`PacketMeasurementResult` は RF test packet だけでなく任意の実 packet を保持できなければならない。既知 payload pattern との一致を result 成立条件にせず、復調できたデータを少なくとも以下の段階に分けて保持する。

```text
Air Bits
  復調直後の無線上の bit 列

Recovered Bytes
  byte 境界へ整列した byte 列

Payload Bytes / Payload Hex
  packet header 等を除いた payload 本体

Decoded Fields
  Shared Protocol Packet Analyzer による意味解析結果
```

暗号化、接続 context 不足、未知 packet type 等により semantic decode できない場合でも、復元済み bit / byte / payload は破棄しない。packet が capture 端で途切れた場合も取得できた範囲を保持し、欠損 byte は valid mask または `??` 表示で明示する。

`MetricResult` の limit は解析アルゴリズムに埋め込まず、後述する versioned rule profile から与える。

---

## 3. 共通 UI 案

専用解析 mode 起動時は、**現行 Generic VSA の 6 分割 workspace をベースに画面レイアウトを専用 mode 向けへ切り替える**。

画面全体を別 window に置き換えるのではなく、既存の `3 列 × 2 行` の 6 sub-window 構成を維持し、各 sub-window の役割・タイトル・tab 構成を protocol / PHY に応じて切り替える。

表示項目が 6 面に収まらない場合は sub-window を追加せず、**1 つの sub-window 内に tab を設けて表示を切り替える**。これにより画面密度を維持しつつ、Bluetooth / Wi-Fi それぞれで必要となる多数の result を扱う。

### 3.1 基本 workspace

基本配置:

```text
┌──────────────────────┬──────────────────────┬──────────────────────┐
│ 1 IQ Power           │ 2 Spectrum           │ 3 Result Summary     │
│                      │                      │ [Current][Statistics]│
├──────────────────────┼──────────────────────┼──────────────────────┤
│ 4 Modulation Results │ 5 Symbol / PHY Plot  │ 6 Packet Analysis    │
│ [Tab A][Tab B]...    │ [Tab A][Tab B]...    │ [Decode][List]...   │
└──────────────────────┴──────────────────────┴──────────────────────┘
```

Pane 1 / 2 は原則として Generic VSA と同じ **IQ Power / Spectrum** を維持する。

Pane 3～6 は専用 mode に応じて内容を切り替える。

推奨 role:

| Pane | 基本 role | tab 化候補 |
|---:|---|---|
| 1 | IQ Power / packet timing | field overlay、packet region overlay |
| 2 | Spectrum | Spectrum、必要に応じ spectral measurement |
| 3 | Result Summary | Current、Statistics、Limits / Pass-Fail |
| 4 | Modulation / RF quality | FSK deviation、DEVM/EVM、power transition、EVM vs symbol/subcarrier 等 |
| 5 | Symbol / PHY visualization | Vector、Constellation、Channel Response、Pilot 等 |
| 6 | Packet / protocol analysis | Decode、Packet List、Issues、Raw Bits/Bytes |

固定すべきなのは pane 番号そのものではなく、**ユーザーが mode を切り替えても主要情報の位置関係が大きく変わらないこと**とする。

### 3.2 mode 起動時の layout 切り替え

Generic VSA から Bluetooth / Wi-Fi 専用 mode へ入った時点で、その mode 用 layout preset を適用する。

例:

```text
Generic VSA
  IQ Power | Spectrum | Result Summary
  Mod      | Symbol   | Symbol Table

Bluetooth Analyzer
  IQ Power | Spectrum | Result Summary
  RF/Mod   | Symbol   | Packet Analysis

Wi-Fi Analyzer
  IQ Power | Spectrum | Result Summary
  EVM/PHY  | Symbol   | Packet Analysis
```

専用 mode から Generic VSA へ戻る場合は Generic VSA の layout preset を復元する。

各 mode で最後に選択していた tab を保持してよいが、初回起動時は代表的な tab を選択する。

### 3.3 Bluetooth Analyzer の tab 例

```text
Pane 3 Result Summary
  [Current] [Statistics] [Limits]

Pane 4 RF / Modulation
  BR / LE:
    [Freq Deviation] [Timing] [Power]

  EDR:
    [DEVM vs Symbol] [FSK-PSK Power] [Timing]

  HDT:
    [EVM vs Symbol] [Power] [Timing]

Pane 5 Symbol / PHY Plot
    [Vector] [Constellation]

Pane 6 Packet Analysis
    [Decode] [Payload Hex] [Packet List] [Issues] [Air Bits]
```

PHY に存在しない tab は表示しない。

例えば BR / LE GFSK で Constellation が意味を持たない場合は、Pane 5 を frequency-domain / symbol-domain の別 plot に差し替えてよい。

### 3.4 Wi-Fi Analyzer の tab 例

802.11a/g Non-HT OFDM:

```text
Pane 3 Result Summary
  [Current] [Statistics] [Limits]

Pane 4 EVM / PHY
  [EVM vs Symbol] [EVM vs Subcarrier] [Timing/CFO]

Pane 5 Symbol / Channel
  [Constellation] [Channel Response] [Pilot/CPE]

Pane 6 Packet Analysis
  [Decode] [Packet List] [Issues] [Raw PSDU]
```

802.11b DSSS / CCK では Pane 4 / 5 の tab を PHY に合わせて差し替える。

### 3.5 Packet List

capture 中に複数 packet がある場合は、Pane 6 の `Packet List` tab を使用する。

例:

| # | Time | PHY | Type | Power | CFO | Mod Quality | Integrity |
|---:|---:|---|---|---:|---:|---:|---|
| 0 | ... | EDR 3M | 3-DH1 | ... | ... | ... | CRC OK |
| 1 | ... | EDR 3M | 3-DH1 | ... | ... | ... | CRC OK |

行を選択すると、6 pane 全体の result を選択 packet に同期して更新する。

### 3.6 IQ Power への field overlay

IQ Power は単なる time trace ではなく、packet field / modulation region を overlay できるようにする。

Bluetooth EDR 例:

```text
| Access + Header | Guard | Sync | EDR Payload | Trailer |
|------ GFSK -----|       |--------- PSK -------------|
```

Wi-Fi OFDM 例:

```text
| L-STF | L-LTF | L-SIG | DATA0 | DATA1 | DATA2 | ... |
```

Protocol logical field と waveform region は必ずしも 1:1 ではないため、Wi-Fi では **PHY time region** と **MAC field tree** を別表示とする。

### 3.7 UI 実装方針

- 6 pane の親 layout は Generic VSA と共有する。
- 各 pane の content は mode-specific widget / tab set として差し替える。
- result 種別追加のたびに main window の分割数を増やさない。
- plot / table は必要に応じ tab 内で遅延生成し、非表示 tab の再描画負荷を抑える。
- packet selection、Single/Continuous capture、selected result index は全 tab で共有する。
- 1 capture に対して複数 tab が別々に DSP を再実行せず、共通 analysis result を参照する。

---

# Part A. Bluetooth / BLE Analyzer

## 4. 目的

Bluetooth 専用 mode は、BR / EDR / LE の RF test packet を主対象として、認証試験や送信品質評価で見る代表的な項目を **1 回の capture からまとめて確認できる mode** とする。

初期目的は Bluetooth 認証 tester の完全代替ではない。

まずは以下を短時間で確認できることを価値とする。

- packet power
- carrier frequency offset / drift
- modulation quality
- FSK frequency deviation
- EDR GFSK / PSK relative power
- symbol timing / rate error
- packet structure / payload / integrity
- packet 間ばらつき

---

## 5. 対象 PHY と実装順

### Phase BT-1

- BR 1M
- EDR 2M
- EDR 3M
- LE 1M
- LE 2M

### Phase BT-2

- LE HDT

### Phase BT-3

- より一般的な OTA packet 解析
- connection context を利用した decode
- Coded PHY 等の追加 PHY

最初は **RF Test preset** を中心に実装し、一般 packet analyzer は後から拡張する。

---

## 6. Bluetooth Analyzer 設定

推奨 UI parameter:

```text
Protocol        Bluetooth
Analysis Profile RF / PHY Test / General Packet
PHY             Auto / BR / EDR 2M / EDR 3M / LE 1M / LE 2M / HDT
Channel/Freq    RF channel または Center Frequency
Packet Type     Auto / optional hint
Expected Payload None / Auto Detect / PRBS9 / 1010 / 11110000 / Custom
Payload Display Hex / Bits
Packet Count    Single / N / Continuous
Spec Profile    None / Bluetooth RF Test <version>
```

MVP では PHY の完全自動識別を必須としない。

まず PHY hint を与えたうえで packet synchronization と packet type decode を自動化する方が堅牢。

### 6.1 Analysis Profile

Bluetooth 専用 Analyzer の内部に、用途に応じた 2 種類の analysis profile を設ける。

```text
Bluetooth Analyzer
├─ RF / PHY Test
└─ General Packet
```

両 profile は別々の復調器を持たず、packet detection、同期、PHY 判別、field segmentation、RF / modulation measurement、bit recovery までを共通化する。その後の payload 解釈、規格集計、limit 判定、表示上の重点だけを切り替える。

```text
Packet Detection / Segmentation
              ↓
Common RF / PHY Measurement
Power / CFO / Drift / Timing / Deviation / EVM / DEVM / Relative Power
              ↓
┌────────────────────────┬────────────────────────┐
│ RF / PHY Test          │ General Packet         │
│ pattern comparison     │ payload hex / decode   │
│ standard aggregation   │ general statistics     │
│ rule-based limits      │ informational metrics  │
└────────────────────────┴────────────────────────┘
```

#### RF / PHY Test

- PRBS9 等の既知 payload pattern の選択または自動推定
- 規格で定義された evaluation interval
- 規格固有の複数 packet aggregation
- versioned rule profile による PASS / FAIL
- expected data と recovered data の bit comparison

#### General Packet

- test packet に限定しない任意の実 packet
- payload pattern 一致を解析成立条件にしない
- Payload Hex / Air Bits / semantic field decode
- 暗号化 payload も復元済み byte 列として表示
- 未知 packet type や不完全 packet も解析可能範囲を保持
- measurement result は原則 INFO とし、正式な規格 limit 判定を強制しない

Analysis Profile は RF / modulation measurement の ON / OFF ではない。General Packet でも packet structure と PHY が判別できる限り、test packet と同じ基礎測定を行う。

| Measurement / Function | RF / PHY Test | General Packet |
|---|---:|---:|
| Packet / Peak Power | Yes | Yes |
| CFO / Carrier Drift | Yes | Yes |
| Symbol Rate Error / Timing | Yes | Yes |
| FSK Frequency Deviation | Yes | Yes |
| EVM / DEVM | Yes | Yes |
| EDR GFSK / PSK Relative Power | Yes | Yes |
| Guard / transition timing | Yes | Yes |
| Payload Hex / Air Bits | Yes | Yes |
| Known pattern comparison | Primary | Optional |
| Standard-defined aggregation | Yes | No（一般統計） |
| Rule Profile PASS / FAIL | Optional | 原則 No |

例えば通常の EDR 実 packet でも、GFSK Access / Header 区間の平均 power、PSK Payload 区間の平均 power、両者の Relative Power、Payload DEVM を測定する。payload が PRBS9 等であることをこれらの測定成立条件にしない。

測定区間を確定できない、packet が途中で欠けている、PHY 判別に失敗した等の場合は値を推測して出さず、metric を `UNKNOWN` とし、`PSK payload boundary unknown` のような測定不能理由を `issues` に記録する。

### 6.2 Payload 表示と pattern 判定の分離

`Expected Payload` は RF test pattern の照合・推定に使う optional hint であり、表示対象 payload を選別する設定ではない。`None` の場合も packet detection、同期、復調、RF measurement、Payload Hex 表示、可能な範囲の semantic decode を実行する。

Pane 6 の Packet Analysis は以下を基本 tab とする。

```text
Packet Analysis
├─ Decode
├─ Payload Hex
├─ Packet List
├─ Issues
└─ Air Bits
```

Payload Hex は byte offset とともに表示する。Bluetooth の LSB-first air bit order と通常の byte-oriented hex 表記を混同しないよう、bit order、whitening / dewhitening、復号等の適用状態を metadata として併記する。表示・export は実 packet と RF test packet で同一形式を使用する。

---

## 7. Bluetooth 解析 pipeline

```text
IQRecording
  ↓
Power-based candidate detection
  ↓
Protocol-specific correlation / timing acquisition
  ↓
Coarse CFO correction
  ↓
PHY-specific synchronization
  ↓
Packet field segmentation
  ↓
RF / modulation measurements
  ↓
Hard decision bits
  ↓
Shared Protocol Packet Analyzer
```

### 7.1 BR / LE GFSK

主処理:

1. packet candidate detection
2. preamble / access sequence based timing
3. CFO estimation
4. instantaneous frequency / GFSK demodulation
5. symbol timing recovery
6. hard decision
7. FSK modulation metrics
8. semantic decode

### 7.2 EDR

EDR は composite packet として扱う。

```text
GFSK Access + Header
        ↓
Guard
        ↓
EDR Sync
        ↓
PSK Payload
        ↓
Trailer
```

処理:

1. GFSK 部で packet start / coarse CFO / timing を取得
2. Header decode から packet type を判断
3. Guard / EDR Sync を検出
4. PSK 部で carrier / timing を fine adjustment
5. π/4-DQPSK または 8DPSK を復調
6. DEVM を計算
7. GFSK と PSK の power を個別計測
8. bit stream を semantic analyzer へ渡す

上記 RF / modulation measurement は `Analysis Profile` にかかわらず実施する。RF / PHY Test では規格指定の評価区間と aggregation を適用し、General Packet では decode された packet structure から同等の評価区間を自動生成する。

現行 Generic VSA の `CompositeSignalDescription` / segment analysis の考え方を再利用する。

---

## 8. Bluetooth 測定項目

### 8.1 共通 packet metrics

全 PHY で可能な限り共通表示する。

| Metric | 内容 |
|---|---|
| Packet Average Power | packet 評価区間の平均 power |
| Peak Power | packet 内 peak |
| Carrier Frequency Offset | nominal center からの周波数誤差 |
| Carrier Drift | packet 内の周波数変化 |
| Symbol Rate Error | 推定 symbol rate の誤差 |
| Packet Duration | packet start - stop |
| Decode Integrity | HEC / CRC / FCS 等 |
| Packet Type | decode 結果 |
| Payload Length / Hex | 復元できた payload 長と byte 列。実 packet でも常時保持 |
| Expected Payload Match | RF / PHY Test 時の既知 pattern 照合結果。General Packet では optional |
| Decode Status | semantic decode、暗号化、context 不足、不完全 capture 等の状態 |

### 8.2 BR / LE GFSK metrics

候補:

- average positive / negative frequency deviation
- `Δf1` 系 metric
- `Δf2` 系 metric
- modulation index 相当値
- zero-crossing / symbol timing error
- frequency deviation vs symbol/time

正確な metric 名、評価区間、統計方法、Pass/Fail threshold は対象 Bluetooth RF Test Specification revision に合わせて rule profile 化する。

### 8.3 EDR metrics

必須候補:

- GFSK section average power
- PSK section average power
- **Relative Power = PSK Power - GFSK Power**
- Guard duration
- EDR Sync detection quality
- RMS DEVM
- 99% DEVM
- Peak DEVM
- DEVM vs symbol
- Carrier Frequency Offset
- Symbol Rate Error

既存 Bluetooth memo では EDR DEVM が sync + payload を対象とし、trailer を評価対象から外す構成となっているため、VSA 側も packet segmentation から評価区間を自動生成する。

50-symbol block 等の規格由来 aggregation は、per-packet result と分離した `BatchMetricAggregator` で扱う。

### 8.4 LE HDT metrics

HDT 対応時は EDR DEVM を流用せず、HDT 向け RMS EVM pipeline を持つ。

候補:

- Control Header EVM
- Payload EVM
- overall / section RMS EVM
- carrier frequency error
- timing error
- constellation
- decoded packet integrity

---

## 9. Bluetooth 用 plot

### 常時表示候補

- IQ Power vs Time
- Spectrum
- Instantaneous Frequency / Frequency Deviation vs Time

### GFSK

- Frequency Deviation vs Symbol
- Histogram / distribution of deviation

### EDR / HDT

- Constellation
- Vector Plot
- DEVM / EVM vs Symbol
- segment power plot

### Packet decode

`Shared Protocol Packet Analyzer` の field tree / table を表示する。

例:

```text
Bluetooth EDR Packet
├─ Access Code
├─ Header
│  ├─ LT_ADDR
│  ├─ TYPE
│  ├─ FLOW
│  ├─ ARQN
│  ├─ SEQN
│  └─ HEC
└─ Payload
   ├─ Payload Header
   ├─ Payload Body
   └─ CRC
```

RF / PHY Test では expected payload pattern と一致率を summary に表示する。General Packet では payload を既知 pattern に分類できなくても、復元済み Payload Hex、Air Bits、decode 済み field、および RF / modulation measurement を表示する。

---

## 10. Bluetooth Batch / Certification-oriented Result

単一 packet の詳細解析とは別に複数 packet の統計 result を持ち、原則として Pane 3 `Result Summary` の `Statistics` tab に表示する。独立した追加 window は作らない。

例:

```text
Packets analyzed : 200
CRC OK           : 200 / 200
Power Mean       : ... dBm
Power Min/Max    : ... / ... dBm
CFO Mean         : ... kHz
CFO Min/Max      : ... / ... kHz
RMS DEVM Mean    : ... %
99% DEVM         : ... %
Peak DEVM        : ... %
Relative Power   : ... dB
```

必要な統計量は metric ごとに aggregator policy を持たせる。

```python
MetricDefinition(
    id="edr_devm_rms",
    aggregation="standard_defined",
)
```

単純な平均で規格定義を置き換えない。

---

## 11. Pass / Fail 判定

専用 analyzer の DSP と規格値を分離する。

```text
Measurement Algorithm
        ↓
Raw MetricResult
        ↓
Versioned Rule Profile
        ↓
PASS / FAIL / UNKNOWN
```

想定:

```text
rules/
└─ bluetooth/
   ├─ core63_rf_test.yaml
   └─ hdt_vsr03.yaml
```

Rule profile が持つもの:

- applicable PHY
- metric ID
- evaluation region / aggregation policy ID
- lower / upper limit
- source document / revision
- notes

これにより Bluetooth specification revision が変わっても DSP core を変更せずに済む。

また、Pluto VSA を正式認証 tester と誤認しないよう、UI では `Certification-oriented` または `RF Test Analysis` と表現し、正式な qualification 判定とは区別する。

---

## 12. Bluetooth MVP 完成条件

### BT MVP-1: EDR 中心

現状資産との距離が近い EDR から専用 mode を成立させる。

- 2-DH1 / 3-DH1 を検出
- packet segmentation
- GFSK / PSK power
- relative power
- CFO
- RMS / 99% / Peak DEVM
- IQ Power / Spectrum / DEVM / Constellation
- demod bits → Shared Protocol Packet Analyzer
- field tree / CRC status

### BT MVP-2: BR / LE 1M / LE 2M

- GFSK packet synchronization
- frequency deviation metrics
- power / CFO / timing
- RF test packet decode
- multi-packet statistics

### BT MVP-3: Rule Profile

- Bluetooth RF Test specification に合わせた metric 定義の再確認
- versioned limits
- PASS / FAIL 表示

### BT MVP-4: HDT

- HDT packet detection
- QPSK / 8PSK / 16QAM section analysis
- RMS EVM
- packet decode

---

# Part B. Wi-Fi Analyzer

## 13. 目的

Wi-Fi mode は Generic VSA の IQ Power / Spectrum を維持しつつ、packet-based PHY synchronization、modulation analysis、PHY decode、MAC decode を追加する。

対象順:

1. **802.11a / 802.11g Non-HT OFDM 20 MHz**
2. **802.11b DSSS / CCK**
3. 将来: 802.11n HT20 / HT40

Wi-Fi VSG memo と同じく、最初から全 Wi-Fi PHY を扱わない。

---

## 14. Wi-Fi 共通 pipeline

```text
IQRecording
  ↓
Packet Detection
  ↓
PHY Synchronization
  ↓
PHY Header Decode
  ↓
Rate / Length / Modulation Decision
  ↓
Payload Demodulation
  ├─ RF / Modulation Metrics
  └─ PHY bits / PSDU
             ↓
    Shared Protocol Packet Analyzer
             ↓
          MAC Decode
```

Wi-Fi では PHY header の decode 結果によって DATA の modulation / coding が決まるため、Generic VSA のように modulation をユーザーが固定指定する構造にはしない。

---

## 15. 802.11a/g Non-HT OFDM Analyzer

### 15.1 packet acquisition

推奨処理:

1. L-STF periodic correlation による packet detection
2. L-STF による coarse CFO estimation
3. L-LTF による fine timing
4. L-LTF による fine CFO
5. L-LTF による channel estimation
6. L-SIG decode
7. RATE / LENGTH 取得
8. DATA symbol 数を決定
9. DATA OFDM demodulation

### 15.2 DATA decode

```text
Time-domain OFDM symbol
  ↓ GI removal
FFT
  ↓
Pilot phase correction
  ↓
Channel equalization
  ↓
BPSK / QPSK / 16QAM / 64QAM demap
  ↓
Deinterleave
  ↓
BCC / puncturing decode
  ↓
Descramble
  ↓
SERVICE / PSDU / TAIL / PAD
  ↓
PSDU
  ↓
MAC / FCS decode
```

L-SIG の RATE field によって data rate を判別する。

### 15.3 OFDM modulation metrics

候補:

- Packet Average Power
- Carrier Frequency Offset
  - coarse
  - fine
  - residual
- Symbol Timing / Sample Clock Error
- RMS EVM overall
- EVM per OFDM symbol
- EVM per subcarrier
- pilot phase / common phase error
- channel magnitude / phase response
- L-SIG decode status
- FCS status

将来、802.11 RF test / measurement requirement と紐付ける場合は Bluetooth と同じ rule-profile architecture を使う。

### 15.4 OFDM plot

標準表示候補は 6 pane layout に割り当て、項目数が多い部分は tab で切り替える。

- Pane 1: IQ Power vs Time
  - L-STF / L-LTF / L-SIG / DATA overlay
- Pane 2: Spectrum
- Pane 3: Result Summary
- Pane 4 tabs:
  - EVM vs OFDM Symbol
  - EVM vs Subcarrier
  - Timing / CFO
- Pane 5 tabs:
  - Constellation
  - Channel Response
  - Pilot / CPE
- Pane 6 tabs:
  - Packet Decode
  - Packet List
  - Issues
  - Raw PSDU

Result summary:

```text
PHY              802.11g Non-HT OFDM
Rate             54 Mbps
Length           128 bytes
Power            ... dBm
CFO              ... kHz
EVM RMS           ... % / dB
L-SIG             Valid
FCS               Valid
MAC Type/Subtype  Beacon
```

---

## 16. Wi-Fi semantic decode

Wi-Fi OFDM では MAC field の bit が FEC / interleave / subcarrier mapping により time-domain IQ 上で連続しない。

したがって、Bluetooth のように `air bits` と packet field を直接 1:1 対応させない。

専用 PHY decoder が PSDU を復元した後、`Shared Protocol Packet Analyzer` へ **LOGICAL representation** として渡す構成を推奨する。

```text
OFDM IQ
 ↓
Wi-Fi PHY Decoder
 ↓
PSDU bytes / logical bits
 ↓
Shared Protocol Packet Analyzer
 ↓
MAC Frame
  Frame Control
  Type / Subtype
  Address fields
  Sequence Control
  Frame Body
  FCS
```

一方、IQ waveform 上の L-STF / L-LTF / L-SIG / DATA region は VSA 側で管理する。

```text
Protocol logical tree != Waveform time-region tree
```

この分離は `WiFi_RF_Test_Packet_Design_Memo_JA.md` の VSG 側設計とも整合させる。

---

## 17. 802.11b DSSS / CCK Analyzer

11a/g OFDM と decoder chain が大きく異なるため、無理に OFDM analyzer と内部実装を共通化しない。

共通化するのは上位 interface / result / UI / protocol decode とする。

### 17.1 packet acquisition

対象:

- Long PLCP preamble
- Short PLCP preamble

処理候補:

1. preamble correlation / packet detection
2. carrier frequency correction
3. chip / symbol timing
4. PLCP header decode
5. SIGNAL から rate 判定
6. PSDU decode
7. FCS check

### 17.2 rate-specific demodulation

- 1 Mbps: DBPSK + Barker
- 2 Mbps: DQPSK + Barker
- 5.5 Mbps: CCK
- 11 Mbps: CCK

### 17.3 初期 measurement

MVP では以下を優先する。

- IQ Power
- Spectrum
- Packet Power
- Carrier Frequency Offset
- clock / timing error
- PLCP decode
- selected data rate
- PSDU / MAC decode
- FCS

DSSS / CCK 固有の modulation accuracy 指標については、実装前に IEEE 802.11 および reference VSA の評価定義を確認し、独自定義を certification metric として表示しない。

内部 debug metric として correlation quality / codeword error 等を持つことは可能。

---

## 18. Wi-Fi MVP 完成条件

### Wi-Fi MVP-1: 802.11a/g Non-HT OFDM

- L-STF detection
- L-LTF fine sync / channel estimate
- L-SIG decode
- RATE / LENGTH auto detection
- DATA OFDM decode
- BPSK / QPSK / 16QAM / 64QAM
- BCC / deinterleave / descramble
- PSDU recovery
- FCS
- IQ Power / Spectrum / Constellation
- EVM overall / symbol / subcarrier
- MAC decode through Shared Protocol Packet Analyzer

### Wi-Fi MVP-2: 802.11b

- Long / Short PLCP detection
- 1 / 2 Mbps Barker decode
- 5.5 / 11 Mbps CCK decode
- PLCP / rate / length decode
- PSDU / FCS / MAC decode
- packet power / CFO / timing

### Wi-Fi MVP-3: HT

- HT-Mixed detection
- HT-SIG decode
- HT20 first
- 1 spatial stream / MCS 0-7
- HT-specific EVM / pilot / channel metrics

---

# Part C. 実装構成案

## 19. package 構成

既存 `pluto_sa.vsa` の Generic VSA core を維持し、protocol-specific layer を追加する。

```text
pluto_sa/vsa/
├─ model.py
├─ analysis.py
├─ demod/
├─ profiles/
│
└─ protocol_modes/
   ├─ __init__.py
   ├─ model.py
   ├─ metrics.py
   ├─ rule_engine.py
   ├─ batch.py
   │
   ├─ bluetooth/
   │  ├─ detector.py
   │  ├─ synchronizer.py
   │  ├─ br.py
   │  ├─ edr.py
   │  ├─ le.py
   │  ├─ hdt.py
   │  └─ metrics.py
   │
   └─ wifi/
      ├─ detector.py
      ├─ legacy_ofdm.py
      ├─ dsss_cck.py
      ├─ ofdm_metrics.py
      └─ model.py
```

Protocol semantic decode は別 package:

```text
pluto_protocol/
├─ bluetooth/
└─ wifi/
```

依存方向:

```text
pluto_protocol
      ↑
protocol_modes
      ↑
VSA UI
```

`pluto_protocol` が Qt、Pluto device、VSA DSP に依存してはいけない。

---

## 20. Generic VSA とのコード共有

専用 mode で再利用する候補:

- `IQRecording`
- amplitude calibration
- Spectrum calculation
- IQ Power calculation
- generic PSK / FSK demod primitives
- EVM primitives
- symbol mapping
- `CompositeSignalDescription`
- existing Bluetooth profile constants / primitives

ただし以下は protocol-specific layer 側に置く。

- packet detection
- protocol synchronization
- packet field segmentation
- PHY classification
- standard-specific evaluation interval
- packet-specific metric aggregation
- rule evaluation

Generic VSA の `VSAAnalysisResult` を無理に巨大化せず、専用 mode result が Generic result を内包できる構造が望ましい。

---

## 21. Validation 方針

専用 mode は「decode できた」だけでは完成としない。

### 21.1 VSG loop validation

```text
Pluto VSG known packet
  ↓
IQ file / RF loop
  ↓
Pluto VSA dedicated analyzer
  ↓
expected semantic result と比較
```

### 21.2 Commercial VSA correlation

R&S FPL1014 等と同じ RF signal を分配入力し、以下を比較する。

- Power
- CFO
- DEVM / EVM
- symbol count / evaluation range
- relative power
- decode result

### 21.3 Independent reference vector

Generator と Analyzer が同じ誤りを持つことを防ぐため、VSG の内部関数だけを oracle にしない。

- Bluetooth specification reference
- Wi-Fi independent reference vector
- commercial VSA
- independent OSS / capture where appropriate

を組み合わせる。

### 21.4 Regression data

IQ capture と expected result を固定 test asset として保存する。

```text
tests/data/protocol_vsa/
├─ bluetooth/
│  ├─ br/
│  ├─ edr/
│  └─ le/
└─ wifi/
   ├─ nonht_ofdm/
   └─ dsss_cck/
```

期待値には tolerance を設定する。

---

## 22. 実装優先順位

推奨順序:

```text
1. Bluetooth Dedicated Analyzer shell と Analysis Profile 切り替え
2. Shared Protocol Packet Analyzer の Bluetooth MVP
3. Bluetooth EDR Dedicated Analyzer（RF / PHY Test + General Packet）
4. Bluetooth BR / LE Dedicated Analyzer（RF / PHY Test + General Packet）
5. Payload Hex / Air Bits / partial packet 共通 result と export
6. Bluetooth multi-packet statistics
7. Bluetooth versioned RF Test rule profile
8. HDT extension
9. Wi-Fi Non-HT OFDM packet detector / synchronizer
10. L-SIG + DATA decoder
11. OFDM EVM / subcarrier analysis
12. Wi-Fi MAC semantic decode
13. 802.11b DSSS / CCK
14. HT20
```

最初に EDR を選ぶ理由は、現行 VSA に BR / EDR profile、composite modulation、PSK VSA、DEVM に近い解析資産がすでにあり、専用 mode framework の検証対象として最も距離が近いため。

Wi-Fi は Bluetooth framework の上位 result / UI / packet list / semantic analyzer integration を再利用し、PHY detector / decoder のみ大きく追加する。

---

## 23. 最重要設計原則

1. **Generic VSA と protocol-specific analysis を分離する**
2. **RF / PHY analysis と semantic packet decode を分離する**
3. **単一 packet と複数 packet 統計を別レイヤーにする**
4. **規格値を DSP 実装に埋め込まない**
5. **Bluetooth で framework を固めてから Wi-Fi へ広げる**
6. **Wi-Fi の logical field と waveform time region を混同しない**
7. **VSG と VSA だけで閉じた自己検証にせず、独立 reference で相関確認する**
8. **既知 payload pattern の一致を RF / modulation measurement の成立条件にしない**
9. **semantic decode 不能でも復元済み bit / byte と測定値を失わない**

この構成なら、Bluetooth の RF / PHY test と実 packet 解析を同一の専用 analyzer framework 上で両立し、その後 Wi-Fi、将来は ADS-B 等へ拡張できる。

## 24. Bluetooth専用解析モード 初期実装（2026-08-30）

Bluetooth専用解析モードの最初のUI統合を実施した。

- `Analysis Mode > Bluetooth Dedicated Analyzer` を追加
- Generic VSAと同一トップレベルウィンドウ内でワークスペースを切り替える
- Pluto接続はトップレベルアプリが一つだけ所有し、Generic / Bluetooth / ADS-Bで共有する
- 6等分レイアウトを採用
  - 左上: IQ Power
  - 中上: Spectrum
  - 右上: Result Summary
  - 左下: RF / Modulation
  - 中下: Symbol
  - 右下: Packet Analysis
- Analysis Profileとして`RF / PHY Test`と`General Packet`を用意
- Packet AnalysisにDecode / Payload Hex / Packet List / Issues / Air Bitsタブを実装
- BR/EDRとLE、各PHY、whitening、UAP/CLK6-1、LE Channel/CRC Initを指定可能
- Semantic decodeにはVSA/VSG共通の`pluto_protocol`を使用
- Generic VSAの新しい解析結果はsignalでBluetoothワークスペースへ公開する

初期MVPのGeneric VSA結果再利用に加え、Bluetooth画面からの直接IQ取得と専用解析を
実装した。Classic BR/EDRはaccess code同期後にBR/EDR PHYを自動評価し、LE 1M/2Mは
Access Addressを使って同期、whitening解除、PDU長によるpacket切り出し、CRC検証まで
行う。LE preambleはAccess Addressの先頭air bitから正しい極性を決め、復調bitを自動反転
しない。未知packetの完全なPHY分類、BRからEDRへの複合区間抽出、規格判定値、
multi-packet統計は引き続き次段階とする。

---

## 25. Generic VSA準拠UI・複数パケット表示（2026-08-30）

Bluetooth専用解析画面は、解析方式だけを規格専用とし、操作体系と測定表示はGeneric VSAへ揃える。

- メインウィンドウ上部には解析設定ツールバーを置かない。
- `Meas Config` はGeneric VSAと同じモーダル・階層型の操作とし、Bluetooth Analysis / Input / Signal Description / Display / Sweepのページを持つ。
- PHYから一意に決まる変調方式、Symbol Rate、TX Filter、Result Rangeは読み取り専用とする。
- 測定開始・停止は `Sweep / Run` メニューおよびF6で行う。
- 同一IQ内で検出したpacketはすべて解析し、Packet Listと左右矢印キーで選択する。
- PlotのRect Zoom、3-button mouse、Reset/View All等は共通measurement chromeを使用する。

BR + EDR PSKでは、IQ PowerをBR部からEDR部まで同一時間軸で表示し、SpectrumへFSKを黄色、PSKをシアンで重ねる。ModulationはFSK Instantaneous Frequency / PSK Vector、Symbol PlotはFSK / PSKをタブで切り替える。FSKはConstellation Frequency / Phase Difference、PSKはPhysical IQ / Differential IQを切替可能とし、Flat / Densityとsymbol-point表示もGeneric VSAへ揃える。

共有decoder検証では生成2-DH1について、Header TYPEのMeaning=`2-DH1`、EDR Length=54 byte、CRC=Validを確認した。TYPEはraw nibbleだけでなく実際のpacket形式をMeaningへ表示する。

---

## 26. Bluetooth Triggerと解析停止条件（2026-08-30）

Bluetooth専用解析モードの`Meas Config > Trigger`はGeneric VSAと同じ二段構成とする。

- Acquisition Trigger: Free RunまたはI/Q Power、Level、Slope、Offset、Hysteresisを設定する。
- Post-capture Burst Search: 取得済みIQ内の複数バーストを抽出し、各バースト直後から同期パターンを探索する。
- Burst SearchにはLevel、Hysteresis、Envelope Average、Drop-Out、Holdoff、Search Start Offset、active interval制限を持たせる。

I/Q Power acquisition triggerはPluto取得の開始位置を揃える機能であり、取得済みIQ内に含まれる複数packetの探索はPost-capture Burst Searchが担当する。両者は独立してON/OFFできる。

無信号や設定不一致時に弱い相関候補を大量に再解析してUI上終了不能に見えることを防ぐため、複数packet解析には以下の停止条件を設ける。

- F6または同じRun操作で解析停止を要求できる。
- 候補間の再解析ごとに停止要求を確認する。
- 1 captureあたりの候補再解析数に安全上限を設ける。
- Burst Search有効時はtrigger-gated candidateを同期・header decode双方の基準とし、無信号区間の候補をheader decodeへ渡さない。

---

## 27. 測定プロット共通化とEDR PHY境界（2026-08-30）

Bluetooth専用解析画面のプロット操作・配色・密度表示をGeneric VSAと同じ共有部品へ統合した。

- 全プロットでRect Zoom、中央ボタンドラッグPan、Reset、View Allを共通化する。
- IQ PowerのResult RangeはGeneric VSAと同じ青、Pattern Rangeは緑で表示する。
- FSK Constellation FrequencyとPSK Symbol Plotは、Flat/DensityともGeneric VSAと同一の描画関数を使う。
- SpectrumはFSKを黄色、PSKをシアンで重ね、凡例を表示する。
- Packet Analysisの長いPayload Hexは32文字単位で折り返し、値・Meaning列をウィンドウ幅へ追従させる。

BR/EDRのPHY切替位置は、Access Code開始位置を基準に次式で決定する。

```text
EDR sync search origin = Access Code start
                       + 72 BR symbols (Access Code)
                       + 54 BR symbols (Header)
                       + 5 BR-symbol periods (Guard)
                       = Access Code start + 131 BR symbols
```

EDR同期語の探索はこの理論位置近傍だけに制限し、後続packetのPSK部を誤って現在packetへ結合しない。PSKのIQ Power、Spectrum、Vector、Symbol Plotは、検出したEDR同期位置を含む同一の局所解析範囲から生成する。

---

## 28. EDR Length終端と専用Config永続化（2026-08-30）

EDR同期の初回解析では、ユーザーのResult Rangeをpacket探索用の十分広い上限として使う。Enhanced Payload Headerをdecodeできた後は、そのLengthとPayload CRC位置からpacketの正確なPSK symbol数を計算し、同じEDR同期位置を基準に再解析する。2-DH1では次の構成になる。

```text
PSK Result Symbols = 10 symbol EDR Sync
                   + ceil((Enhanced Header + Payload + Payload CRC) / 2 bit)
                   + 2 symbol Trailer
```

これにより、Post Idle、後続packet、ユーザーが余裕を持って指定したResult RangeをPSK Vector、Constellation、EVM、Spectrumへ混入させない。表示側も再解析フィルタの端部ではなく、pattern同期で確定したsymbol列と時刻を基準にする。

FSK Constellation FrequencyはGeneric VSAと同じ横軸`-1.0..+1.0`へ固定し、横方向のPan/Zoomを無効にする。中心周波数の初期値は2440 MHzとする。

Bluetooth専用モードのMeas ConfigはGeneric VSAと共通ファイルへ保存しない。Bluetooth専用のQSettings名前空間とschema/version付きJSONへ、Bluetooth Analysis、Input / Frontend、Trigger、Burst Search、Display設定を自動保存し、次回起動時に復元する。設定ダイアログのOK時にも保存し、アプリ終了時に最終状態を再保存する。

---

## 29. Generic VSA共通解析・表示パイプライン（2026-08-30）

Bluetooth専用モードは別の変調解析器を持たない。専用モード固有の責務は、Protocol / PHY / packet typeの判定、PHYから一意に決まるSignal Descriptionの設定、packet境界の決定、およびprotocol fieldのdecodeに限定する。境界が決まったFSK部とPSK部はGeneric VSAと同じ`VSASession` / pattern解析器へ入力する。

次の処理はGeneric VSAと専用モードで共通モジュールを使用する。

- FSKのRaw / Measured表示、Gaussian measurement filter、復元symbol周波数
- PSKのTX filter / measurement filter、symbol timing、carrier補正、振幅正規化、pi/4-DQPSKの表示基準
- Physical / Differential constellation、EVM / Differential Symbol EVM / Bluetooth DEVM
- Symbol PlotのFlat / Density表示、FSK周波数プロットの固定横軸
- IQ Power / Modulation上のsymbol点、Result Range / Pattern Range表示
- 軸、Rect Zoom、middle-button Pan、Reset / View All等のplot操作

`Show Symbol Points`は時間軸trace上の同期symbol点だけをON/OFFする。Symbol Plotそのものを非表示にはしない。

EDR 2M / 3Mのsymbol rateはいずれも1 MSym/sであり、2 Mbit/s / 3 Mbit/sは1 symbolあたりのbit数で決まる。専用モードのPSK範囲計算と表示フィルタもこのsymbol rateを使う。EDRのDEVMは専用UIで再計算せず、Generic VSA解析結果のmetadataを表示する。
