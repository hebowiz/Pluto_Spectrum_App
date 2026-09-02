# Classic / BLE 専用VSA：SIG RF測定整合性レビューとCodex修正指示

## 1. 目的

現行のBluetooth Classic BR/EDRおよびLE 1M/2M専用VSAについて、

- 現在の復調・解析処理のうち正しい部分
- Bluetooth SIGのRF測定に対して不足している測定項目
- Generic VSA由来の補正をそのまま規格測定へ流用してはいけない部分
- EDR DEVMの修正方針
- BR / LE GFSKのModulation Characteristics、Carrier Frequency、Drift等の実装方針
- Power / Spectrum系でVSA単体ではなくSA機能へ委譲すべき項目
- Codexが実装する際の優先順位と回帰試験

を整理する。

本書は、HDT専用VSAの仕様整合性レビューと同様に、**既存のpacket decode能力を壊さず、SIG測定用のmeasurement pathを独立して追加する**ことを基本方針とする。

---

# 2. 対象

主な現行コード：

```text
pluto_sa/vsa/protocol_modes/bluetooth/model.py
pluto_sa/vsa/demod/gfsk.py
pluto_sa/vsa/pattern.py
pluto_sa/vsa/profiles/bluetooth_br.py
pluto_sa/vsa/profiles/bluetooth_edr.py
pluto_sa/vsa/protocol.py
```

関連テスト：

```text
tests/test_bluetooth_br_profile.py
tests/test_bluetooth_edr.py
tests/test_vsa_bluetooth_dedicated.py
```

仕様メモ：

```text
docs/Bluetooth_RF_Test_Packet_Design_Memo_JA.md
```

参照するBluetooth SIG文書：

```text
BR/EDR Radio Physical Layer (RF) Test Suite
RF.TS.p36
Revision: 2025-11

LE Radio Physical Layer (RFPHY) Test Suite
RFPHY.TS.p24ed2
Revision: 2025-11

Bluetooth Core Specification v6.3
Vol 2 Part A / B
Vol 6 Part A / B / F
```

> 注意
> 本実装はBluetooth認証テスターそのものを代替するものではない。
> Pass/Failを「SIG準拠測定」として扱うには、DSPアルゴリズムだけでなく、Plutoの周波数基準、絶対レベル校正、受信帯域、measurement filterの実測応答、測定不確かさを別途characterizeする必要がある。

---

# 3. 結論

現行Classic/BLE専用VSAは、**packet detector / demodulator / decoderとしてはかなりよくできている**。

特に以下は有効な共通基盤として残す。

- BR Access Code detection
- BR packet header decode
- HEC / CRC
- BR/EDR自動判別
- EDR Sync detection
- LE Preamble + Access Address synchronization
- LE packet length trim
- LE CRC / whitening decode
- fractional timingを含むGFSK frequency model
- CFO / drift / deviation推定
- multi-packet detection
- Packet Decode UI

ただしSIGのRF測定として見ると、現状は**復調を安定させるための推定・補正**と**DUT誤差を測定するための処理**が混在している。

最重要方針は次の通り。

```text
Decoder / Diagnostic path
    → 復調成功率を高めるためのCFO補正、drift補正、deviation fit、
       fractional timing fit、decision-directed trackingを許可する

SIG Measurement path
    → 各Test Suiteが明示的に許可・要求した補正だけを使う
    → DUTのdeviation error / drift / phase errorを勝手にfitして消さない
```

この2経路をコード上でも明確に分離すること。

---

# 4. 現行実装の評価サマリー

| 項目 | 現行 | 評価 |
|---|---|---|
| BR Access Code同期 | 実装済み | 維持 |
| BR Header / HEC / Payload decode | 実装済み | 維持 |
| EDR PHY自動判別 | 実装済み | 維持 |
| EDR Sync検出 | 実装済み | 維持 |
| LE 1M/2M packet sync / decode | 実装済み | 維持 |
| Generic GFSK CFO推定 | 実装済み | Diagnosticとして有用 |
| Generic GFSK drift推定 | 実装済み | Diagnosticとして有用 |
| Generic GFSK deviation fit | 実装済み | SIG Modulation判定にはそのまま使わない |
| GFSK fractional timing | 実装済み | 同期基盤として有用 |
| BR Δf1 / Δf2測定 | 未実装 | 追加必須 |
| LE Δf1 / Δf2測定 | 未実装 | 追加必須 |
| BR f0 / fk方式のdrift | 未実装 | 追加必須 |
| LE f0 / fn方式のdrift | 未実装 | 追加必須 |
| EDR Relative Power | 概略値あり | SIG windowへ修正 |
| EDR DEVM RMS | Global metricあり | SIG測定としては未完成 |
| EDR DEVM 50-symbol block | 未実装 | 追加必須 |
| EDR 200-block aggregation | 未実装 | 追加必須 |
| EDR Peak / 99% DEVM | 未実装 | 追加必須 |
| EDR Guard Time | 判別用timingのみ | 測定値として追加 |
| EDR Sync / Trailer conformance | 未実装 | 追加 |
| LE Output Power PAVG / PPK | Generic powerのみ | 測定windowを追加 |
| LE In-band Emission | 未実装 | SA機能へ委譲 |
| RF Test Case aggregation | 未実装 | 追加 |
| Low/Mid/High自動評価 | 未実装 | 将来automation |
| LE Coded / CTE / CS | 対象外 | Unsupportedを明示 |

---

# 5. 最重要：復調補正と規格測定補正を分離する

## 5.1 現行GFSK demodulator

`pluto_sa/vsa/demod/gfsk.py`は、既知bit列と実測instantaneous frequencyから、

- deviation
- CFO
- linear drift
- fractional timing

をjoint fitできる。

これはdecoderとして非常に有用であり、削除しない。

また最終bit判定では、CFOやdriftを除去したfrequency traceを使用している。

これは**decode用として正しい**。

---

## 5.2 SIG測定では同じ補正をそのまま使わない

たとえばCarrier Drift試験で、

```text
実測frequency
→ linear drift fit
→ driftを除去
→ 残差からCarrier Driftを算出
```

としてはいけない。

測りたいDUT誤差そのものを先に消してしまうからである。

同様にModulation Characteristicsで、

```text
実測 deviation
→ nominal deviationへscale fit
→ Δf1 / Δf2を算出
```

としてはいけない。

SIG measurement pathでは、tester channel filterと規定measurement windowを通した後の**実測frequency deviationそのもの**を使う。

---

## 5.3 推奨データモデル

```python
class BluetoothMeasurementIntent(StrEnum):
    DIAGNOSTIC = "diagnostic"
    SIG_RF_TEST = "sig_rf_test"
```

または内部的に、

```text
DemodulationTrace
SIGMeasurementTrace
```

を完全に分ける。

推奨：

```python
@dataclass(frozen=True)
class BluetoothFMMeasurementTrace:
    time_s: np.ndarray
    frequency_hz: np.ndarray
    p0_sample: float
    sample_rate_hz: float
    samples_per_symbol: float
    filter_profile: str
```

`frequency_hz`には、**SIG測定用channel filter + FM demodulationを通した値を、CFO/drift/deviation scale補正なしで保存**する。

---

# 6. SIG measurement filterを新設する

## 6.1 現行の問題

現在Bluetooth FSK pathでは、

```python
MeasurementFilterMode.NONE
```

を使用している。

コメントにある、

```text
Bluetooth GFSKにはTX Gaussian shapingが既に含まれているため、
受信側で同じGaussian BT=0.5をもう一度かけるとdeviationを過小評価する
```

という判断自体は正しい。

しかし、

```text
Gaussian matched filterをかけない
```

ことと、

```text
SIG Lower Testerのmeasurement channel filterを使う
```

ことは別である。

したがって、

```text
Gaussian BT=0.5 measurement filter
```

を追加するのではなく、**RF.TS / RFPHY.TSで定義されるtester channel filter**を別実装する。

---

## 6.2 BR / LE 1M measurement channel filter

目標応答：

```text
passband ripple: <= 0.5 dB within ±550 kHz
attenuation:
    ±650 kHz : approximately -3 dB
    ±1 MHz   : at least -14 dB
    ±2 MHz   : at least -44 dB
```

BRとLE 1MのModulation / Carrier系測定で共通利用できる構成にする。

---

## 6.3 LE 2M measurement channel filter

1Mの約2倍の周波数スケール。

```text
passband ripple: <= 0.5 dB within ±1.1 MHz
attenuation:
    ±1.3 MHz : approximately -3 dB
    ±2 MHz   : at least -14 dB
    ±4 MHz   : at least -44 dB
```

FM demodulator帯域についても、RFPHY Test SuiteのLower Tester要件を満たすこと。

---

## 6.4 実装方針

新規module例：

```text
pluto_sa/vsa/protocol_modes/bluetooth/rf_measurement_filter.py
```

API例：

```python
class BluetoothRFMeasurementFilterProfile(StrEnum):
    BR_1M = "br_1m"
    LE_1M = "le_1m"
    LE_2M = "le_2m"
```

```python
def apply_rf_test_channel_filter(
    iq: np.ndarray,
    *,
    sample_rate_hz: float,
    profile: BluetoothRFMeasurementFilterProfile,
) -> np.ndarray:
    ...
```

フィルタは単に「それらしいlow-pass」にせず、unit testでfrequency responseを検証する。

---

# 7. LE 2M nominal deviationの修正

## 現行

`_le_signal()`はLE 1M / LE 2Mの双方で、

```python
frequency_deviation_hz=250_000.0
```

となっている。

## 問題

LE 2Mのnominal GFSK deviationは約500 kHzである。

したがって少なくともSignalDescriptionのnominal/reference値としては、

```python
LE 1M: 250_000 Hz
LE 2M: 500_000 Hz
```

とする。

Generic GFSK demodulatorが実測deviationをfitできるため、現状でもdecodeできる可能性は高いが、設定値として誤っている。

## 修正

```python
def _le_signal(phy):
    if phy is LE_1M:
        rate = 1_000_000.0
        deviation = 250_000.0
    else:
        rate = 2_000_000.0
        deviation = 500_000.0
```

優先度Aで修正する。

---

# 8. RF PHY Test profileでtest packet条件を明示する

現行APIではClassic / LEとも、通常packet用のdefaultが残っている。

特にLE analyzerはGeneral Packetを考慮して、

```python
whitening_enabled=True
```

をdefaultとしている。

これはGeneral Packetでは正しいが、RF PHY Testでは異なる。

## 方針

`BluetoothAnalysisProfile.RF_PHY_TEST`では、少なくとも以下をvalidateする。

```text
BR/EDR RF Test:
    whitening OFF

LE RF Test:
    whitening OFF
    test packet format
    expected Access Address / Sync Word
    CRCInit = RF test value
```

General Packet profileでは現在の柔軟性を維持する。

---

# 9. BR：Modulation Characteristicsを実装する

## 9.1 現行

GFSK modelからglobalな、

```text
frequency_deviation_hz
```

を推定できる。

これは「このpacketのおおよそのdeviation」を見るDiagnosticとして有用。

しかしSIGのModulation Characteristicsとは異なる。

---

## 9.2 SIG測定用に追加する値

BRでは少なくとも、

```text
Δf1avg
Δf2max distribution
Δf2avg
Δf2avg / Δf1avg
```

を出す。

RF Test payload patternに応じて、

```text
11110000...
10101010...
```

をそれぞれ使用する。

`Δf1avg`、`Δf2max`はTest Suiteで指定されたbit位置とsample windowから求める。

Global deviation fitを代用しない。

---

## 9.3 BR pass criteria

現行Core/Test Suiteベース：

```text
140 kHz <= Δf1avg <= 175 kHz
99.9% of Δf2max >= 115 kHz
Δf2avg / Δf1avg >= 0.8
```

1 packetだけではなく、RF Test Caseとして必要なpacket数・周波数条件を満たしたaggregate resultを別に持つ。

---

## 9.4 実装API例

```python
@dataclass(frozen=True)
class FSKModulationCharacteristics:
    delta_f1_avg_hz: float
    delta_f2_avg_hz: float
    delta_f2_max_hz: np.ndarray
    delta_f2_ratio: float
    sample_count: int
```

```python
def measure_br_modulation_characteristics(
    trace: BluetoothFMMeasurementTrace,
    decoded_packet: PacketAnalysisResult,
    payload_pattern: str,
) -> FSKModulationCharacteristics:
    ...
```

---

# 10. BR：Initial Carrier Frequencyを追加する

Generic GFSKのCFO fitを、そのままRF/TRM/CA/BV-08の`f0`として表示しない。

SIG measurementではpreamble位置を基準とした規定integration windowから`f0`を得る。

実装：

```python
@dataclass(frozen=True)
class InitialCarrierFrequencyResult:
    nominal_frequency_hz: float
    f0_hz: float
    error_hz: float
```

BRでは、

```text
f0 error within ±75 kHz
```

を判定可能にする。

Packet synchronizationから得たp0を使うが、p0はinteger sampleではなく`float sample position`として保持する。

---

# 11. BR：Carrier Frequency Driftを追加する

## 11.1 現行

GFSK demodulatorにはlinear drift modelが既に存在する。

これはDiagnostic値として残す。

例：

```text
Estimated Linear Drift: +xxx Hz/s
```

ただしSIG verdictは別計算にする。

---

## 11.2 SIG measurement

規定payload patternから、

```text
f0
f1
f2
...
fk
```

をTest Suite指定のbit integration windowで求める。

判定に使う値：

```text
max |fk - f0|
max drift-rate between specified fk pairs
```

BRのpacket slot数に応じたdrift limitを適用する。

```text
1-slot : ±25 kHz
3-slot : ±40 kHz
5-slot : ±40 kHz
```

また50 µs相当のintervalに対するdrift-rate条件も別metricとして出す。

## 11.3 禁止事項

```text
linear driftをfit
→ subtract
→ drift test
```

は禁止。

SIG drift resultはraw measurement traceから算出する。

---

# 12. EDR：Relative Transmit PowerをSIG windowへ修正する

## 12.1 現行

現行は概ね、

```text
FSK Average Power
PSK Average Power
Relative Power = PSK - FSK
```

を出している。

考え方は正しい。

---

## 12.2 修正

RF.TSのRelative Transmit Powerでは、

```text
PGFSK:
    Access Code + HeaderのGFSK portionから十分な中央区間

PDPSK:
    Sync + EDR PayloadのDPSK portionから十分な中央区間
```

を測る。

少なくとも各portionの80%以上を使い、guardやPHY切替transientを混ぜない。

Test Suiteとしては複数trace averaging、low/mid/high、必要に応じmax/min power条件もある。

したがって現在の「result range全体の単純平均」をそのままSIG値としない。

結果：

```text
PGFSK
PDPSK
PDPSK - PGFSK
```

判定：

```text
-4 dB < (PDPSK - PGFSK) < +1 dB
```

---

# 13. EDR：現行Bluetooth DEVM RMSはDiagnostic扱いへ変更する

## 13.1 現行

Generic VSAのdifferential PSK pathでは、

- measured absolute symbolsをRMS normalization
- ideal differential sequenceからabsolute referenceを生成
- referenceを除去
- adjacent symbol間の差分から`bluetooth_devm_rms_percent`

を生成している。

これはDEVMに近い品質指標として有用。

しかし、**RF.TSのDEVM Test Caseを再現していない**。

---

## 13.2 不足

SIG RF measurementには少なくとも、

```text
BR headerからinitial frequency error ωiを求める
EDR portionへωi correction
SRRC measurement filter β=0.4
50-symbol non-overlapping block
blockごとのtiming phase ε0
blockごとのresidual frequency error ω0
blockごとのRMS DEVM
symbolごとのPeak DEVM
全symbolから99% DEVM
total 200 blocks
trailer excluded
```

が必要。

現行のglobal DEVM scalarでは不足。

---

# 14. EDR：ωiはBR Headerから規定方法で求める

Current Generic CFOやAccess Code全体fitを、そのまま`ωi`に使わない。

RF.TSではBR packet headerのfrequency deviationを使い、ISIの影響が小さいbitを選択してinitial center frequency errorを求める。

概念：

```text
selected '1' header bits -> mean Δω1
selected '0' header bits -> mean Δω2

ωi = center estimate derived from Δω1 and Δω2
```

実装時はRF.TSの定義に合わせる。

結果として以下を保持する。

```python
initial_frequency_error_hz: float
selected_header_bit_indices: np.ndarray
```

---

# 15. EDR：50-symbol block DEVM engineを新設する

新規module例：

```text
pluto_sa/vsa/protocol_modes/bluetooth/edr_rf_measurement.py
```

データモデル：

```python
@dataclass(frozen=True)
class EDRDEVMBlockResult:
    block_index: int
    start_symbol: int
    timing_offset_symbols: float
    residual_frequency_error_hz: float
    rms_devm: float
    peak_devm: float
    symbol_devm: np.ndarray
```

aggregate：

```python
@dataclass(frozen=True)
class EDRDEVMTestResult:
    initial_frequency_error_hz: float
    blocks: tuple[EDRDEVMBlockResult, ...]
    rms_worst: float
    peak_worst: float
    devm_99_percentile: float
    total_symbol_count: int
```

---

# 16. EDR DEVM measurement範囲

測定対象：

```text
Synchronization sequence
+
EDR Payload
```

測定対象外：

```text
Trailer 2 symbols
```

packetごとの端数blockはRF.TSに従って扱い、複数packetから50-symbol non-overlapping blocksを収集し、最終的に200 blocksを構成する。

1 packetのDEVM表示はDiagnosticとして許可するが、

```text
SIG Test Verdict
```

は200 blocks揃うまで出さない。

---

# 17. EDR DEVM limits

現行仕様メモとRF.TSに基づく代表値：

| Modulation | RMS DEVM | 99% DEVM | Peak DEVM |
|---|---:|---:|---:|
| π/4-DQPSK | <= 0.20 | <= 0.30 | <= 0.35 |
| 8DPSK | <= 0.13 | <= 0.20 | <= 0.25 |

またfrequency stabilityについて、

```text
ωi
ω0
ωi + ω0
```

もblock単位で記録し、RF.TS limitsと比較できるようにする。

---

# 18. EDR DEVMでGeneric VSAの補正を流用しない

Generic VSAの、

```text
best timing
best carrier
best phase
global RMS normalization
decision-directed correction
```

は、display / diagnosticには使ってよい。

ただしSIG DEVM engineではAppendix Cに規定された、

```text
ε0
ω0
DEVM
```

のoptimizationのみを使用する。

特に、

```python
bluetooth_devm_rms_percent
```

は当面、

```text
Diagnostic DEVM
```

としてUI上も区別する。

SIG engine完成後に、

```text
SIG RMS DEVM
SIG 99% DEVM
SIG Peak DEVM
```

を別metricとして追加する。

---

# 19. EDR：Differential Phase Encodingを追加する

Packet decoderは既にPRBS9 / payload bit comparisonを実装可能な基盤を持っている。

RF Test profileでは複数packetのpayload demodulation結果を集約し、

```text
packet count
bit error count
packets with zero errors
```

を表示する。

これはDEVMとは独立したtest resultとして扱う。

---

# 20. EDR：Guard Timeを追加する

現行EDR detectorは、

```text
BR header後のexpected PHY boundary
```

を使ってEDR Syncを探索しており、PHY判定として良い。

しかしGuard Time測定値は出していない。

追加：

```python
@dataclass(frozen=True)
class EDRGuardTimeResult:
    header_end_time_s: float
    reference_symbol_start_time_s: float
    guard_time_s: float
```

Guard Timeは、単純に「検出sync index - nominal header end」とするだけでなく、RF.TSのreference symbol timing定義に合わせて専用estimateを作る。

複数packetを集約してdistribution / pass rateを出せるようにする。

---

# 21. EDR：Synchronization Sequence / Trailerチェック

現行ではSync sequenceをEDR検出に使っている。

これをRF Test resultとしても保存する。

```text
Sync symbol / bit errors
Trailer symbol / bit errors
packet count
```

TrailerはDEVMから除外するが、**Trailer correctness testでは解析対象**である。

DEVM対象外だからdecodeまで捨てないこと。

---

# 22. LE 1M / 2M：Modulation Characteristicsを追加する

## 22.1 現行

Generic GFSK engineはnominal waveformとのfitからdeviationを推定する。

これをDiagnosticに残す。

---

## 22.2 SIG measurement

RF Test Packetのpayload patternを判定し、

```text
11110000...
10101010...
```

の測定を分ける。

出力：

```text
Δf1avg
Δf2avg
Δf2max samples
Δf2avg / Δf1avg
```

uncoded PHYではpayload先頭/末尾の規定除外範囲を守る。

各bitのfrequency traceは内部的に十分細かくsampleする。

---

## 22.3 sampling grid

RFPHY Test SuiteではModulation Characteristicsで各bitを少なくとも32 pointsで評価する。

したがってmeasurement engineの内部sample gridは、

```text
>= 32 samples / symbol
```

とする。

Pluto captureが、

```text
16 MS/s
LE 1M -> 16 native samples/symbol
LE 2M -> 8 native samples/symbol
```

の場合でも、帯域条件を満たしたIQからfractional resamplingして32-point gridを生成することは可能。

ただし、interpolationでpoint数を増やしただけでLower Tester equivalentを名乗らず、reference instrumentとの相関確認を必須とする。

---

# 23. LE modulation limits

通常modulation index：

```text
LE 1M:
    225 kHz <= Δf1avg <= 275 kHz

LE 2M:
    450 kHz <= Δf1avg <= 550 kHz
```

Stable Modulation Index対応時：

```text
LE 1M:
    247.5 kHz <= Δf1avg <= 252.5 kHz

LE 2M:
    495 kHz <= Δf1avg <= 505 kHz
```

またuncoded PHYでは、

```text
99.9% Δf2max criterion
Δf2avg / Δf1avg >= 0.8
```

をTest Suiteに従って判定する。

---

# 24. LE：Carrier Frequency Offset / Driftを追加する

## 24.1 現行

Generic GFSK global modelの、

```text
CFO
linear drift
```

はDiagnosticとして表示してよい。

## 24.2 SIG measurement

RF Test payload `10101010...`を使い、

```text
f0
f1
f2
...
fn
```

を規定windowで求める。

LE 1MとLE 2Mではpreamble lengthとpayload integration block lengthが異なるので、同じ単純関数でsymbol countだけ固定しない。

output例：

```python
@dataclass(frozen=True)
class CarrierDriftResult:
    f0_hz: float
    fn_hz: np.ndarray
    max_absolute_offset_hz: float
    max_drift_from_f0_hz: float
    max_drift_rate_hz: float
```

判定には、

```text
nominal carrierに対する各frequency
f0との差
指定interval間のdrift rate
```

を個別に使う。

---

# 25. LE：Output PowerをSIG windowへ変更する

現行の、

```text
Packet Average Power
Peak Power
```

は一般VSAとして有用。

ただしSIG Output Power Test Caseとは別metricとする。

LE RF Testでは、

```text
PPK
PAVG
```

をTest Suite指定のRBW/VBW/detector条件とburst timing windowで測る。

少なくともPAVGは、

```text
burst中央の十分な区間
```

から求め、burst start/end transientを含む単純packet全体平均にしない。

UI：

```text
Packet Average Power        ← Diagnostic
SIG PAVG                    ← RF Test
SIG PPK                     ← RF Test
PPK - PAVG
```

と分離する。

絶対dBm Pass/FailはPluto個体のpower calibrationが有効な場合のみenableする。

---

# 26. LE：In-band EmissionsはSpectrum Analyzerへ委譲する

RFPHY transmitter testにはIn-band Emissionsがある。

これはVSAのsymbol demodulation処理ではなく、規定RBW/VBW / detector / frequency-bin power integrationを使うSpectrum Analyzer系測定。

したがって専用VSA内へ無理に実装せず、

```text
Bluetooth Analyzer
    ↓
SIG Test
    ↓
In-band Emissions
    ↓
SA measurement engine
```

と共通SA処理へ委譲する。

Plutoの1 capture bandwidthで全offsetを覆えない場合は、

```text
Swept SA
または
multiple center-frequency captures
```

を使う。

---

# 27. ClassicでVSA外へ委譲すべきSIG transmitter項目

BR/EDR RF Test Suiteには、VSA modulation paneだけでは完結しない項目もある。

少なくとも以下はSA / automation側の担当とする。

```text
BR:
- Output Power
- Power Density
- Power Control
- TX Output Spectrum Frequency Range
- 20 dB Bandwidth
- Adjacent Channel Power

EDR:
- In-band Spurious Emission
```

Bluetooth専用VSAから呼び出して同じResult Summaryへ結果を統合するのは可。

---

# 28. 現行専用VSAに不足しているSIG transmitter coverage

## 28.1 BR

| Test area | 現状 | 対応 |
|---|---|---|
| Packet decode | あり | 維持 |
| Output Power | Diagnosticのみ | SA測定profile追加 |
| Power Density | なし | SA |
| Power Control | なし | Automation + SA |
| Spectrum range | なし | Swept SA |
| 20 dB BW | なし | SA |
| Adjacent Channel Power | なし | SA |
| Modulation Characteristics | なし | VSA追加 |
| Initial Carrier Frequency | Generic CFOのみ | SIG f0追加 |
| Carrier Frequency Drift | Generic linear fitのみ | SIG fk追加 |

---

## 28.2 EDR

| Test area | 現状 | 対応 |
|---|---|---|
| BR/EDR auto detect | あり | 維持 |
| Relative Transmit Power | 概略値あり | SIG window / aggregation |
| Carrier Stability | Generic CFOのみ | ωi / ω0 |
| RMS DEVM | global diagnostic | 50-symbol block engine |
| 99% DEVM | なし | 追加 |
| Peak DEVM | なし | 追加 |
| 200-block aggregate | なし | 追加 |
| Differential Phase Encoding | decode可能 | aggregate test追加 |
| In-band Spurious | なし | SA |
| Guard Time | detector内部のみ | metric追加 |
| Sync Sequence correctness | detectorのみ | aggregate test追加 |
| Trailer correctness | decode testなし | 追加 |

---

## 28.3 LE 1M / 2M

| Test area | 現状 | 対応 |
|---|---|---|
| Packet sync / CRC | あり | 維持 |
| Output Power | Diagnosticのみ | SIG power profile |
| In-band Emission | なし | SA |
| Modulation Characteristics | global deviationのみ | Δf1 / Δf2 engine |
| Stable Modulation Index | なし | option追加 |
| Carrier Offset | Generic CFOのみ | f0 / fn |
| Carrier Drift | global linear fitのみ | windowed drift |
| LE Coded | 未対応 | Scope外を明示 |
| CTE | 未対応 | Scope外を明示 |
| AoD Power Stability | 未対応 | 将来 |
| Channel Sounding | 未対応 | 将来 |

---

# 29. RF Test Case eligibilityを導入する

任意のBluetooth packetから数値を出すことと、SIG Test CaseとしてPass/Failすることを分ける。

例：

```python
@dataclass(frozen=True)
class RFTestEligibility:
    eligible: bool
    reasons: tuple[str, ...]
```

判定例：

```text
wrong payload pattern
whitening enabled
wrong packet type
capture too short
insufficient packet count
unsupported PHY
insufficient measurement bandwidth
power calibration unavailable
```

条件未成立時：

```text
Diagnostic value: 表示可
SIG verdict: N/A
```

とする。

---

# 30. Multi-packet / Multi-condition accumulatorを追加する

SIG Test Suiteは1 packetだけで完結しない項目が多い。

新規：

```python
class BluetoothRFTestAccumulator:
    ...
```

保持するdimension例：

```text
PHY
test case
RF frequency
TX power setting
packet type
packet index
```

用途：

```text
BR modulation: multiple packets
EDR DEVM: 200 blocks
EDR Guard: many packets
EDR Sync/Trailer: many packets
LE modulation: multiple packets
low / mid / high frequency sweep
max / min output power
```

---

# 31. 推奨module構成

```text
pluto_sa/vsa/protocol_modes/bluetooth/
    model.py
    ui.py

    rf_measurement/
        __init__.py
        filter.py
        fm_trace.py
        eligibility.py
        power.py

        br.py
        edr.py
        le.py

        accumulator.py
        limits.py
```

Generic GFSK decoderは、

```text
pluto_sa/vsa/demod/gfsk.py
```

に残す。

SIG measurementロジックを`gfsk.py`へ詰め込まない。

---

# 32. 推奨結果モデル

```python
@dataclass(frozen=True)
class BluetoothRFMeasurementResult:
    test_case_id: str
    eligible: bool
    verdict: str
    metrics: Mapping[str, float]
    arrays: Mapping[str, np.ndarray]
    metadata: Mapping[str, object]
```

arraysには可能な限り根拠データを残す。

例：

```text
BR:
    delta_f1_values
    delta_f2_max_values
    fk_values

EDR:
    block_rms_devm
    symbol_devm
    omega0
    timing_offsets

LE:
    delta_f1_values
    delta_f2_max_values
    fn_values
```

単なるPass/Failだけにしない。

---

# 33. UI方針

現行6分割レイアウトを維持する。

推奨例：

```text
Result Summary
    [Current Packet]
    [SIG Test]
    [Statistics]

Modulation
    BR/LE:
        [FM Trace]
        [Δf1 / Δf2]
        [Carrier Drift]

    EDR:
        [Vector]
        [DEVM vs Symbol]
        [DEVM Blocks]
        [Carrier Stability]

Packet Analysis
    [Decode]
    [Packet List]
    [Issues]
    [Raw]
```

Current PacketにはDiagnosticを表示してよい。

SIG Test tabにはTest Suite semanticsを満たした結果のみ表示する。

---

# 34. 表示名を明確に分ける

曖昧な、

```text
Carrier Frequency Offset
Frequency Deviation
Bluetooth DEVM RMS
```

だけでは、Generic値とSIG値の区別がつかない。

例：

```text
Diagnostic CFO
Diagnostic Deviation
Diagnostic DEVM RMS

SIG Initial Carrier Error f0
SIG Δf1avg
SIG Δf2avg
SIG Max Drift
SIG RMS DEVM
SIG 99% DEVM
SIG Peak DEVM
```

とする。

---

# 35. テスト追加：measurement filter response

FIR/IIR implementationに対してfrequency response unit testを作る。

BR / LE1M：

```text
±550 kHz passband ripple
±650 kHz
±1 MHz
±2 MHz
```

LE2M：

```text
±1.1 MHz passband ripple
±1.3 MHz
±2 MHz
±4 MHz
```

を検証する。

---

# 36. テスト追加：FSK deviation injection

理想packetに既知deviationを与える。

BR例：

```text
130 kHz
150 kHz
160 kHz
180 kHz
```

LE 1M例：

```text
200 kHz
250 kHz
300 kHz
```

LE 2M例：

```text
400 kHz
500 kHz
600 kHz
```

期待：

- Diagnostic fitは入力値を追従
- SIG Δf1 / Δf2も規定windowで期待値を再現
- nominal値へ勝手に正規化しない

---

# 37. テスト追加：CFO injection

```text
-100 kHz
-50 kHz
0
+50 kHz
+100 kHz
```

を付加。

期待：

```text
Diagnostic CFO
SIG f0
EDR ωi
```

がそれぞれ定義通り動く。

Generic CFOとSIG f0が必ず同じになることをテスト条件にしない。

---

# 38. テスト追加：nonlinear drift

linear driftだけではなく、packet途中で変化するdriftを入れる。

例：

```text
0 -> +20 kHz -> -10 kHz
```

期待：

- Generic linear fitは平均trendとして表示
- SIG fk / fnは局所変化を検出
- Generic drift subtractionでSIG値が消えない

---

# 39. テスト追加：fractional timing

```text
+0.25 sample
+0.5 sample
```

などを与える。

Decoderがpacketを維持できることに加え、

```text
p0
bit center
DEVM block timing
```

が正しく追従することを確認する。

---

# 40. テスト追加：EDR phase / amplitude impairment

EDR PSK portionだけに、

```text
phase noise
phase ramp
symbol-dependent phase error
amplitude error
```

を追加する。

期待：

```text
Generic VSAで再同期しても
SIG DEVM側では規定以上の誤差が残る
```

こと。

---

# 41. テスト追加：EDR Guard / Trailer

VSG側で、

```text
guard time shift
wrong sync symbol
wrong trailer
```

を注入できるtest helperを作る。

期待：

```text
Guard Time metric
Sync errors
Trailer errors
```

が独立して検出する。

---

# 42. テスト追加：Power window

burst edgeだけにrise/fall distortionを追加する。

期待：

```text
Generic whole-packet average
SIG PAVG
```

が異なることを確認する。

これによりSIG PAVGがburst edgeを誤って含めていないことを保証する。

---

# 43. Reference instrument comparison

self-generated VSG loopbackだけでは不十分。

最低限、

```text
BR:
    Δf1avg
    Δf2avg
    f0
    drift

EDR:
    PGFSK / PDPSK
    ωi
    RMS / 99% / Peak DEVM

LE 1M / 2M:
    PAVG
    Δf1avg
    Δf2avg
    f0
    drift
```

についてR&S等の既知reference measurementと比較するfixtureを残す。

可能なら同一RF信号をdividerで同時測定する。

---

# 44. Pluto固有補正

SIG measurement engine内でPluto固有誤差をDUT誤差と混ぜない。

別layerとして、

```text
Amplitude calibration
Receiver frequency reference correction
I/Q imbalance calibration
Frequency-dependent response correction
```

を適用する。

ただし補正値は、

```text
instrument calibration
```

として明示し、

```text
DUT signal fitting
```

と混同しない。

特にCFO測定ではPluto XO errorが直接結果へ入るので、RF Test用途ではfrequency reference calibrationが重要。

---

# 45. 補正ルールまとめ

| 補正 | Decoder | SIG Measurement |
|---|---|---|
| Instrument amplitude calibration | 可 | 可 / 必須 |
| Instrument frequency reference correction | 可 | 可 / 必須 |
| Fixed I/Q calibration | 可 | 可 |
| Packet CFO correction | 可 | Test Caseが要求する範囲のみ |
| Linear drift subtraction | 可 | 原則不可 |
| Deviation scale normalization | 可 | 不可 |
| Fractional timing estimation | 可 | Test Caseのtiming定義に合わせて可 |
| Decision-directed carrier correction | 可 | 原則不可 |
| Generic constellation best-fit | 可 | EDR Appendix C指定以外不可 |
| Gaussian BT matched filter | Decoder用途次第 | SIG tester filterの代用不可 |
| SIG tester channel filter | 不要な場合あり | 必須 |

---

# 46. 優先順位

## Priority A

1. LE2M nominal deviationを500 kHzへ修正
2. Diagnostic pathとSIG Measurement pathを分離
3. SIG GFSK measurement channel filterを実装
4. BR Δf1 / Δf2 Modulation Characteristics
5. LE 1M/2M Δf1 / Δf2 Modulation Characteristics
6. BR f0 / fk Carrier Frequency measurement
7. LE f0 / fn Carrier Frequency measurement
8. EDR DEVM global scalarをDiagnostic扱いへ変更
9. EDR 50-symbol block DEVM engineを新設

## Priority B

10. EDR ωi / ω0
11. EDR 200-block accumulator
12. EDR 99% / Peak DEVM
13. EDR Relative Power window修正
14. EDR Guard Time
15. EDR Sync / Trailer conformance
16. LE / BR SIG Power window
17. RF Test eligibility判定

## Priority C

18. Low/Mid/High automation
19. Max/Min power automation
20. SA側SIG spectrum measurementsとの統合
21. LE Coded
22. CTE
23. AoD
24. Channel Sounding

---

# 47. 既存機能で壊してはいけないもの

以下は回帰させない。

```text
BR Access Code detection
BR HEC
BR/EDR CRC
BR/EDR whitening
BR/EDR automatic PHY detection
EDR Sync detection
2-DHx / 3-DHx packet decode
LE 1M / 2M sync
LE Access Address
LE whitening
LE CRC
multi-packet detection
Packet List
Packet Decode tree
Raw/Air bits
Generic VSA Vector / Spectrum / IQ Power
Generic GFSK Diagnostic CFO / drift / deviation
```

SIG measurementを追加するためにdecoderを作り直さない。

---

# 48. 完了条件

Classic / BLE専用VSAのSIG Measurement基盤について、以下を満たしたら第一段階完了とする。

```text
[ ] Decoder pathとSIG Measurement pathが分離されている
[ ] LE2M nominal deviation = 500 kHz
[ ] RF Test profileでwhitening/test packet条件をvalidate
[ ] BR/LE用SIG measurement channel filterを実装
[ ] filter response unit testあり
[ ] BR Δf1avg / Δf2avg / Δf2max / ratio
[ ] LE1M/2M Δf1avg / Δf2avg / Δf2max / ratio
[ ] BR f0 / fk
[ ] LE f0 / fn
[ ] Generic drift/deviation補正がSIG値を消さない
[ ] EDR Relative Powerを規定portionで測定
[ ] EDR ωi
[ ] EDR 50-symbol block DEVM
[ ] EDR 200-block aggregation
[ ] EDR RMS / 99% / Peak DEVM
[ ] trailerをDEVMから除外
[ ] EDR Guard Time
[ ] EDR Sync / Trailer error result
[ ] insufficient packet/test conditionではSIG verdict=N/A
[ ] reference instrument comparison fixtureあり
```

---

# 49. Codexへの実装指示

既存`gfsk.py`やGeneric VSAを大規模に書き換えないこと。

実装順序：

```text
1. 現行decoder / Generic VSAの回帰テストを固定
2. LE2M nominal deviationを修正
3. rf_measurement package新設
4. SIG measurement filter実装 + response tests
5. filtered raw FM trace + float p0を生成
6. BR modulation / carrier metrics追加
7. LE modulation / carrier metrics追加
8. UIにDiagnosticとSIG Testを分離表示
9. EDR Relative Powerを規定window化
10. EDR ωi estimator
11. EDR Appendix-C専用DEVM engine
12. 50-symbol blocks
13. 200-block accumulator
14. Peak / 99% / RMS verdict
15. Guard / Sync / Trailer
16. SA engineとのPower/Spectrum test統合
17. R&S comparison fixtureを追加
```

重要：

```text
「復調できるように誤差を補正すること」と
「DUTの誤差を規格通りに測ること」は別問題である。
```

Generic VSAは前者として高機能なまま維持し、Bluetooth RF Test measurementは後者として独立実装すること。

---

# 50. 最終的な製品上の位置づけ

専用Bluetooth Analyzerには、以下の2レベルを共存させる。

```text
General / Diagnostic Analysis
    任意packet
    aggressive synchronization
    decode重視
    Generic CFO / drift / deviation / DEVM
    packet inspection

RF PHY Test Analysis
    RF Test packet eligibility
    SIG tester filter
    prescribed measurement windows
    prescribed aggregation
    SIG-style metrics
    Pass / Fail / N/A
```

この分離を徹底すれば、通常の「Bluetooth packet analyzer」と、
開発・pre-compliance用途の「RF PHY measurement tool」を同じUI内で両立できる。
