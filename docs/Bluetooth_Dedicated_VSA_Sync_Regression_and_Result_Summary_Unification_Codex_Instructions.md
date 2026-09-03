# Bluetooth Dedicated VSA：同期回帰修正とResult Summary統一指示

## 目的

HDT / Bluetooth Classic BR/EDR / LE 1M/2M のDedicated VSAについて、Result Summaryの考え方を統一する。

ただし、**Classic / LEは直近の変更後にシンボル同期・捕捉へ回帰が発生しているため、Result Summary改修より先に必ず同期回帰を修正すること**。

作業順序は以下を厳守する。

```text
1. Classic / LE同期回帰の原因特定・修正
2. 同期回帰テスト追加
3. Result Summary共通モデル化
4. HDT全rateへ統一展開
5. BR / EDRへ展開
6. LE 1M / LE 2Mへ展開
```

現在のworking treeを正とする。GitHubの`feature/bluetooth-vsa-compliance`（`cd5b383`）は設計確認用の基準として参照するが、その後ローカルで実装済みのHDT Appendix C対応や4列Result Summaryを巻き戻さないこと。

---

# 1. 最優先：Classic / LE同期回帰の修正

## 1.1 現状認識

直近のSIG RF measurement追加では、主に以下が追加・変更されている。

- `pluto_sa/vsa/protocol_modes/bluetooth/model.py`
- `pluto_sa/vsa/protocol_modes/bluetooth/rf_measurement/*`
- `pluto_sa/vsa/protocol_modes/bluetooth/ui.py`

一方、既存のGeneric GFSK demodulator / Pattern Analyzer本体はこの変更の中心ではない。

したがって、まずDedicated Analyzer側の

- sync pattern生成
- profile/config値
- packet crop / sample offset
- analysis session選択
- RF measurement追加との責務混在

を優先して疑うこと。

推測でGeneric demodulatorを書き換えない。

---

## 1.2 必須の切り分け

現在のworking treeと、同期が正常だった直前revisionを比較する。

最低限、以下の実IQ fixtureでBefore/Afterを確認する。

```text
tests/fixtures/DH1_test.npz
tests/fixtures/bluetooth_br_prbs9_pluto_16msps.npz
tests/fixtures/bluetooth_2dh1_prbs9_16msps.npz
tests/fixtures/bluetooth_3dh1_prbs9_16msps.npz
tests/fixtures/PLUTO_VSG_SMCV100B_2DH1.npz
tests/fixtures/LE1M_FSK_error_raw.npz
tests/fixtures/LE1M_FSK_error.npz
```

各fixtureについて以下を記録する。

```text
profile / PHY
sync pattern
correlation
pattern_start_sample
result_start_sample
first symbol timing
eligible_match_count
decoded bit count
CRC / HEC result
失敗したstage
```

「Bluetooth synchronization pattern was not found」等の最終例外だけで判断せず、どのstageで差が生じたかを特定する。

---

## 1.3 RF measurement pathを同期経路から分離する

`rf_measurement/filter.py`、`fm.py`等のSIG measurement filterは、**packet/symbol synchronization完了後の測定専用**とする。

禁止：

```text
SIG measurement filter済みIQをAccess Code / Preamble同期へ入力する
SIG用FM traceをdecoderのbit decisionへ戻す
SIG測定失敗をpacket synchronization失敗として扱う
```

正しい流れ：

```text
Raw IQ
  ↓
既存Dedicated/Generic同期・decode
  ↓
packet start / symbol timing / decoded packet確定
  ├─ Packet Decode / Display
  └─ SIG RF Measurement path
```

SIG measurement側がN/Aになっても、同期済みpacketやsymbol plotは正常表示を維持すること。

---

## 1.4 LEで重点確認する箇所

現在はRF Test Packet用として`0x71764129`を特別扱いしている。

以下を確認する。

- RF / PHY Test profileの場合だけRF Test Sync Wordを使用しているか
- General Packetではユーザー指定Access Addressをそのまま使用しているか
- Preambleのair orderが実際のRF Test Packetと一致するか
- LE 1M / LE 2MでPreamble長が正しいか
- UI設定値とanalyzerへ渡る`access_address`が一致するか
- RF Test preset適用前後で意図せずAccess Addressだけ変更されていないか

`LE 2M frequency_deviation_hz = 500 kHz`への修正は正しいため、同期回帰の原因と確認できない限り250 kHzへ戻さない。

---

## 1.5 Classic / EDRで重点確認する箇所

- LAPから生成するAccess Codeが変更前と同じか
- BR Access Code / Headerのcrop範囲がずれていないか
- `_recording_sample_offset`の二重加算がないか
- multi-packet crop後のlocal sample座標とglobal sample座標を混同していないか
- EDR PHY判別用Sync searchがBR Header直後の正しいboundaryにanchorされているか
- BR prefix sessionとEDR payload sessionの結果を混同していないか

EDR SIG measurementはBR/EDR PHY判別が完了した後に実行する。

---

## 1.6 同期修正の完了条件

Result Summary改修へ進む前に以下を満たすこと。

```text
[ ] BR実IQ fixtureでAccess Code / Headerを再び捕捉できる
[ ] 2-DHx / 3-DHx実IQ fixtureでEDR SyncとPSK symbolsを捕捉できる
[ ] LE実IQ fixtureでPreamble / Sync Word / payload symbolsを捕捉できる
[ ] Packet Decodeが以前と同じ結果になる
[ ] Symbol Plotへsymbol pointsが再表示される
[ ] SIG measurementを無効化しても有効化しても同期結果が変化しない
[ ] RF measurement失敗時もpacket decodeは維持される
```

この回帰テストを固定してから次へ進む。

---

# 2. Result Summary共通方針

HDTで現在採用している形式へ全PHYを統一する。

列：

```text
Test Item | Value | Limit | Result
```

section：

```text
RF PHY Measurements
Reference Information
```

基本ルール：

- 認証/RF PHY試験項目を上段へ表示する。
- 規格判定に直接使わない参考値は下段へ表示する。
- Generic VSA的なDiagnostic値はDedicated VSAへ重複表示しない。
- `SIG` prefixは表示名から外す。
- Test Item名、Limit、判定式はBluetooth公式仕様/Test Suiteの表記を優先する。
- Test Suiteで確認できない値を推測でPASS/FAILしない。
- 未測定・必要packet不足は行自体を消さず`N/A`または`MEASURING`にする。

Result：

```text
PASS
FAIL
MEASURING
N/A
—
```

Reference Informationは原則：

```text
Limit  = —
Result = —
```

---

# 3. Summary用View Modelを共通化する

現状のUIは`BluetoothMetric`と`metric_id.startswith("sig_")`に依存しているため、表示順・Limit・ResultをPHYごとに統一しにくい。

SIG measurementの生データである`BluetoothRFMeasurementResult`は維持し、UI用に別のsummary rowを作る。

例：

```python
@dataclass(frozen=True)
class BluetoothSummaryRow:
    section: str
    test_item: str
    value: str
    limit: str = "—"
    result: str = "—"
    metric_id: str = ""
```

PHYごとに、

```python
build_hdt_summary(...)
build_br_summary(...)
build_edr_summary(...)
build_le_summary(...)
```

のようなbuilderで、**表示順・公式名称・Limit・Resultを一箇所で確定**する。

UI側で`sig_`文字列から自動生成しない。

---

# 4. HDT：全rateを現在のHDT7.5形式へ統一

HDT2 / HDT3 / HDT4 / HDT6 / HDT7.5でTest Itemの並びは共通とする。

```text
RF PHY Measurements
Output power
Control Header RMS EVM
PDU Header and payload RMS EVM
Center frequency deviation
Center frequency offset change between the preamble and the payload
Symbol timing accuracy
Pre-packet emissions

Reference Information
Detected PHY
RF Test Eligibility
Preamble Carrier Frequency Error
Payload Carrier Frequency Error
Control Header Average Power
PDU Header and Payload Average Power
Relative Power (Payload - Header)
Preamble Correlation
RMS EVM Packets Evaluated
```

## 4.1 rate別Limit

Control Header RMS EVM：全rate

```text
≤ -10 dB
```

PDU Header and payload RMS EVM：

```text
HDT2   ≤ -10 dB
HDT3   ≤ -13 dB
HDT4   ≤ -16 dB
HDT6   ≤ -19 dB
HDT7.5 ≤ -22 dB
```

Center frequency deviation：

```text
±125 kHz
```

Center frequency offset change between the preamble and the payload：

```text
HDT2   ≤ 3.0 kHz
HDT3   ≤ 3.0 kHz
HDT4   ≤ 1.8 kHz
HDT6   ≤ 1.8 kHz
HDT7.5 ≤ 1.2 kHz
```

Symbol timing accuracy：

```text
< ±50 ppm
```

ただし認証測定方法を正しく実装できていない間は`N/A`。

Pre-packet emissions：

```text
≤ 4 us
```

Output powerはRF PHY Testの正式なmeasurement condition / limitが確認できるまでは、値を表示しても

```text
Limit  = TBD
Result = N/A
```

とする。

## 4.2 EVM処理

現在ローカルで修正済みのAppendix C準拠処理を維持する。

- Preamble：`α0, φ0, Δω0, T0` joint optimization
- Control Header：Preamble parameter固定
- Payload：`α0, T0`固定、`φ1, Δω1`のみ最適化
- nearest-decision reference禁止
- fixed reference使用
- Appendix Cの`Npld=1000`を使用

過去のfeature branch実装へ戻さない。

---

# 5. BR Result Summary

BR専用表示も同じ4列形式にする。

RF PHY Measurementsの候補：

```text
Output power
Δf1avg
99.9% Δf2max
Δf2avg / Δf1avg
Initial carrier frequency
Carrier frequency drift
Carrier frequency drift rate
```

表示名称・Limitは使用中のBluetooth RF Test Suiteの正式表記へ最終的に合わせる。

既にある以下のmeasurement primitiveを利用する。

- `measure_burst_power()`
- `measure_modulation_characteristics()`
- `measure_initial_carrier_frequency()`
- `measure_carrier_drift()`

重要：

`11110000` packetと`10101010` packetの両方が必要な評価は、1packetのResultから無理に判定しない。

`BluetoothRFTestAccumulator.aggregate_fsk()`を拡張し、少なくとも以下をaggregateする。

```text
Δf1avg
Δf2avg
全Δf2max samples
99.9% criterion
Δf2avg / Δf1avg
```

現在の`aggregate_fsk()`はF1/F2平均とratioまでで、最終Verdictを出していないため、PHY別Limitを適用できるよう拡張する。

必要packet/patternが揃っていない場合：

```text
Value  = available value or N/A
Result = MEASURING
```

Reference Information例：

```text
Detected PHY
RF Test Eligibility
Packet Type
Payload Pattern
Access Code Correlation
Packets Evaluated
Peak Power（規格Test Itemとして使わない場合）
```

---

# 6. EDR 2M / 3M Result Summary

EDRは現在のSIG measurement primitivesを活かし、Result Summaryを公式試験項目中心へ整理する。

RF PHY Measurements候補：

```text
Output power
Relative transmit power
Initial frequency error ωi
Residual frequency error ω0
RMS DEVM
99% DEVM
Peak DEVM
Guard time
Differential phase encoding
Synchronization sequence
Trailer
```

現在実装済みの情報：

```text
PGFSK
PDPSK
PDPSK - PGFSK
ωi
ω0
50-symbol block RMS DEVM
99% DEVM
Peak DEVM
Guard Time
Sync symbol errors
Trailer symbol errors
```

## 6.1 DEVM

200 non-overlapping 50-symbol blocksを使用する現在のAccumulator方針を維持する。

```text
< 200 blocks → Result = MEASURING
200 blocks到達 → PASS / FAIL
```

2M / 3MでRMS / 99% / Peak limitを切り替える。

`DEVM Blocks Evaluated`はReference Informationへ置く。

現在保持している`omega0_hz`も、Test Suiteで規定されたfrequency-stability判定へ利用する。単に表示するだけでなく、公式Limitを確認してResultへ反映する。

## 6.2 Relative transmit power

表示値は原則：

```text
PDPSK - PGFSK
```

PGFSK / PDPSK個別値はReference Informationへ移してよい。

## 6.3 Sync / Trailer

公式Test Suite上で独立したPASS/FAIL対象となる条件を確認し、それに従って判定する。

単なるdecoder error countを公式判定へ読み替えない。

---

# 7. LE 1M / LE 2M Result Summary

LEもBRと同じ思想で統一する。

RF PHY Measurements候補：

```text
Output power
Δf1avg
99.9% Δf2max
Δf2avg / Δf1avg
Carrier frequency offset
Carrier frequency drift
Carrier frequency drift rate
```

Stable Modulation Indexを評価するprofile/feature条件では、通常limitと混ぜず、対応する公式limitへ切り替える。

LE 1M / LE 2Mで

- symbol rate
- nominal deviation
- measurement filter
- modulation limits
- carrier/drift window

を明示的に切り替える。

現在の`LE 2M = 500 kHz nominal deviation`は維持する。

BRと同様、F1/F2の異なるpayload patternを跨ぐ評価はAccumulatorで完結させる。

Reference Information例：

```text
Detected PHY
RF Test Eligibility
Payload Pattern
Sync Correlation
Packets Evaluated
PAVG / PPKの補助値（Test Itemと重複しないもののみ）
```

---

# 8. Dedicated VSAとSAの責務

Dedicated VSAへ入れる：

```text
packet/burst power
FSK modulation characteristics
carrier frequency / drift
EDR DEVM
HDT RMS EVM
Guard / timing等のpacket-domain測定
Packet Decode
```

Spectrum Analyzer側へ委譲：

```text
In-band emissions
Adjacent Channel Power
20 dB / 6 dB bandwidth
Out-of-band spurious
広帯域spectrum mask系
```

Dedicated VSAのResult Summaryへ、SA未実装項目を無理に追加しない。

---

# 9. Limit / Resultの中央管理

Limit文字列と判定式をUIへ直書きしない。

例：

```text
bluetooth/rf_measurement/limits.py
```

へ集約する。

key例：

```text
PHY
Test Item ID
feature / stable modulation flag
packet family
```

Test Suite revisionをmetadataとして保持できる構造にする。

公式仕様で確認できないLimitは推測せず、

```text
Limit = TBD
Result = N/A
```

とする。

---

# 10. Summary表示の固定性

同一PHYでは測定途中でも行順を変えない。

悪い例：

```text
11110000 packetではΔf1だけ行が出る
10101010 packetではΔf2だけ行が出る
```

正しい例：

```text
Δf1avg             160 kHz      ...    PASS
99.9% Δf2max       N/A          ...    MEASURING
Δf2avg/Δf1avg      N/A          ...    MEASURING
```

次のpattern取得後に同じ行が更新される。

HDT / BR / EDR / LEすべてこの考え方へ統一する。

---

# 11. Regression Test

## 同期回帰

最優先で実IQ fixture testを追加する。

```text
BR Access Code / Header捕捉
EDR BR-prefix + PSK Sync捕捉
LE Preamble / Sync Word捕捉
Symbol Plot用symbols生成
CRC / HEC / PHY判定
```

## Summary

PHYごとに以下をテストする。

```text
[ ] Header = Test Item / Value / Limit / Result
[ ] RF PHY Measurementsが先頭
[ ] Reference Informationが下段
[ ] 行順が固定
[ ] PASS / FAIL / MEASURING / N/Aが正しい
[ ] HDT rate変更でlimitのみ適切に変わる
[ ] EDR 2M / 3MでDEVM limitが変わる
[ ] LE 1M / 2Mでlimit/filter/rateが変わる
[ ] FSK pattern不足時にMEASURINGになる
[ ] Reference InformationへPASS/FAILを付けない
```

---

# 12. 最終方針

Dedicated Bluetooth VSAのResult Summaryは全PHYで、

```text
Test Item | Value | Limit | Result
```

を共通フォーマットとし、

```text
RF PHY Measurements
Reference Information
```

の2階層へ統一する。

Dedicated VSAでは認証/RF PHY評価とPacket Decodeを担当し、Generic DiagnosticはGeneric VSAへ任せる。

ただし、**Classic / LE同期回帰を解消して既存packet decodeを完全に復旧させるまでは、Result Summaryの大規模変更へ進まないこと。**
