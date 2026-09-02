# HDT専用VSA：同期・EVM処理の仕様整合性レビューと修正指示

## 1. 目的

現行のBluetooth HDT専用VSA実装について、HDT仕様で定義されている同期・RMS EVM測定フローとの整合性を整理し、必要な修正方針を示す。

本書はCodex向けの実装指示として使用する。

対象の主な実装：

- `pluto_sa/vsa/protocol_modes/bluetooth/model.py`
  - `_hdt_training_matches()`
  - `analyze_bluetooth_hdt_recording()`
  - `_hdt_qpsk_constellation()`
  - `_hdt_sample_symbols()`
- `pluto_protocol/bluetooth/hdt.py`
  - `hdt_rf_test_training_symbols()`
  - `map_hdt_symbols()`
  - HDT rate / FEC / puncturing / CRC定義
- `pluto_vsg/engine/bluetooth_hdt.py`
  - HDT RF Test Packet生成
- `tests/test_vsa_bluetooth_dedicated.py`

仕様参照元として、リポジトリ内の以下のメモも使用する。

- `docs/Bluetooth_RF_Test_Packet_Design_Memo_JA.md`
  - HDT_VSr03_PR, Vol 6, Part A, Sections 3.6, 7
  - HDT_VSr03_PR, Vol 6, Part B, Sections 2.7.1, 3.4
  - HDT_VSr03_PR, Vol 6, Part F, Sections 4.2, 5.3
  - Appendix C

> 注意
> 現時点では、Appendix Cの数値最適化式そのものとコードを1対1で照合した最終監査までは行っていない。
> 本書で「仕様上確認が必要」とした項目は、Appendix C原文との直接照合を前提とする。

---

# 2. 結論

現行HDT VSAは、

- HDT Training検出
- 2 Msym/s前提
- STS / GI / LTS構成
- RF PHY Test用LTS
- Trainingを使ったphase/CFO推定
- Control Header decode
- RI / PDU length decode
- FEC / puncturing decode
- HEC-C / CRC-32検証
- HDT rate自動判別

まで含め、**HDT packet analyzer / decoderとしてはかなり正しい**。

一方で、**Bluetooth規定のRMS EVM測定器としては未完成**。

特に以下は修正対象とする。

1. Control Header自身のRMSで振幅再正規化している
2. Payload自身のRMSで振幅再正規化している
3. HDT4/HDT6/HDT7.5 PayloadでGeneric VSAに再同期させている
4. Payload末尾の2 terminating symbolsがEVM対象から外れている
5. Timing推定がinteger sample phase中心で、fractional timingが不足している
6. Appendix Cの数値最適化手順との厳密一致が未確認

最終方針としては、Generic VSAの「最もきれいに同期する処理」をそのまま流用するのではなく、**HDT専用のReference Estimator / EVM Evaluatorを独立させる**。

---

# 3. 現行設計で正しい部分

## 3.1 HDT symbol rate

現行実装：

```python
_HDT_SYMBOL_RATE_HZ = 2_000_000.0
```

HDT PHYは2 Msym/sであり、ここは正しい。

---

## 3.2 HDT Training sequence

現行実装：

```python
_HDT_TRAINING_SYMBOLS = 74
```

`hdt_rf_test_training_symbols()`では、

```text
STS x9 + GI + LTS x2
```

を生成している。

STS：

```text
[-1, -j, +j, +1]
```

RF PHY Test LTS root：

```text
u = 7
```

これらは現行仕様メモと一致している。

この部分は変更不要。

---

## 3.3 Training detection

`_hdt_training_matches()`は、

1. RRC系measurement filter適用
2. SPS phaseごとにsymbol sampling
3. STS前半36 symbolsでcoarse correlation
4. 74-symbol Training全体でphase/CFO補正
5. full training correlationで最終候補判定

という流れになっている。

Packet detectionの考え方として妥当。

現行のcandidate search / duplicate suppressionも残してよい。

---

## 3.4 Trainingからのphase / CFO推定

現行：

```python
training_phase_error = np.unwrap(
    np.angle(
        header_observed[:_HDT_TRAINING_SYMBOLS]
        * np.conj(training_reference)
    )
)

phase_step, phase_intercept = np.polyfit(
    np.arange(_HDT_TRAINING_SYMBOLS),
    training_phase_error,
    1,
)
```

さらに、

```python
cfo_hz = phase_step * _HDT_SYMBOL_RATE_HZ / (2.0 * np.pi)
```

としている。

これは、

```text
received × conj(reference)
→ phase error
→ phase slope
→ CFO
```

という自然な推定方法であり、HDT preambleからcarrier frequency error / phaseを求める方針と合っている。

Appendix Cの厳密な推定式と完全一致しているかは別途確認が必要だが、現行処理の方向性は正しい。

---

## 3.5 Control Headerの位置

現行：

```python
_HDT_TRAINING_SYMBOLS = 74
_HDT_CONTROL_SYMBOLS = 62
_HDT_TERMINATING_SYMBOLS = 2

_HDT_PAYLOAD_START_SYMBOL = (
    _HDT_TRAINING_SYMBOLS
    + _HDT_CONTROL_SYMBOLS
    + _HDT_TERMINATING_SYMBOLS
)
```

Control Headerは、

- 57 logical bits
- K=6 convolutional encoding
- 5 termination bits
- rate 1/2
- 合計124 coded bits
- π/4 QPSKなので62 symbols

となり、Control Header後のterminating 2 symbolsも含めたpayload開始位置の考え方は正しい。

---

## 3.6 Control Header decode

以下は現行のまま基本維持してよい。

- PCA-A
- NESN
- PFI
- RI
- RFU
- PDU Control
- HEC-C
- 5 termination bits
- K=6 / 32-state Viterbi decode

Generator polynomialも現行仕様メモと一致している。

---

## 3.7 Payload modulation / code rate

現行のrate定義：

```text
HDT2   pi/4-QPSK  1/2
HDT3   pi/4-QPSK  3/4
HDT4   8PSK       2/3
HDT6   16QAM      3/4
HDT7.5 16QAM      15/16
```

正しい。

Symbol mappingも現行仕様メモと一致しているため、基本変更不要。

---

## 3.8 HDT FEC / puncturing

以下も現行のまま基本維持してよい。

- non-systematic convolutional code
- constraint length 6
- 32 states
- 5 zero termination bits
- G0 / G1
- 1/2, 2/3, 3/4, 15/16 puncturing

---

## 3.9 HEC-C / CRC-32 / Length decode

現行ではControl HeaderからRI / PDU Controlをdecodeし、それを使ってPayload rangeを自動決定している。

また、

- HEC-C
- CRC-32
- legacy CRC initとの差分診断

まで実装されている。

Packet analyzerとして重要な部分なので維持する。

---

# 4. 修正必須項目

# 4.1 Control Headerのself-normalizationを廃止する

## 現行

`analyze_bluetooth_hdt_recording()`内：

```python
control_rms = float(
    np.sqrt(np.mean(np.abs(control_observed) ** 2))
)

control_observed = control_observed / max(
    control_rms,
    np.finfo(np.float64).tiny
)
```

その後、この再正規化済みControl HeaderからEVMを計算している。

## 問題

HDTの測定フローは、

```text
Preamble processing
  ↓
amplitude / phase / carrier frequency error / timing 推定
  ↓
Control Header processing
```

である。

したがって、Control Header自身のRMSを使って振幅を再推定すると、Preambleに対するControl Headerのamplitude errorをEVMから除去してしまう。

例：

```text
Preamble amplitude = 1.00
Control Header amplitude = 0.95
```

の場合、本来5%程度のamplitude error成分がEVMに入るべきだが、現行ではControl Headerを再び1.0へ正規化してしまう。

## 修正

Preambleから得たamplitude estimateを保存し、Control Headerへそのまま適用する。

例：

```python
@dataclass(frozen=True)
class HDTReferenceEstimate:
    amplitude: float
    phase_rad: float
    phase_step_rad_per_symbol: float
    timing_offset_samples: float
```

Preambleから：

```python
reference = estimate_hdt_reference(...)
```

Control Headerは：

```python
corrected_control = apply_hdt_reference(
    observed_control,
    reference,
    symbol_indices=...
)
```

とし、Control Header自身からamplitudeを再推定しない。

---

# 4.2 Payloadのself-normalizationを見直す

## 現行

```python
payload_rms = float(
    np.sqrt(np.mean(np.abs(payload_corrected) ** 2))
)

payload_corrected /= max(
    payload_rms,
    np.finfo(np.float64).tiny
)
```

## 問題

Payload自身から振幅を再推定しているため、

- Preamble → Payloadのabsolute amplitude error
- Control Header → Payloadのrelative power error

の一部がEVMから消える可能性がある。

## 修正方針

原則として、Preamble derived amplitude referenceを使用する。

ただし、Appendix CでPayloadに対する追加のgain optimizationが明示的に許可されている場合は、その範囲のみ再推定してよい。

したがって実装は、

```text
デフォルト:
Preamble estimateをPayloadまで保持

Appendix Cが別gain optimizationを要求:
仕様通りの式でのみ再推定
```

とする。

現在の単純な、

```python
payload /= rms(payload)
```

は廃止する。

---

# 4.3 PayloadでGeneric VSAによる独立再同期をしない

## 現行

HDT4/HDT6/HDT7.5ではPayload先頭のdecisionからseedを作り、

```python
payload_session = _analyze_known_pattern(...)
```

へ渡してGeneric VSAで再解析している。

Generic VSAはPayload自身を使って、

- carrier
- timing
- phase
- constellation alignment

を再度最適化できる。

## 問題

復調器としては有効だが、規格測定器としては「補正しすぎ」になる可能性がある。

HDT RMS EVMは、Preamble processingで得たreference receiver parametersを基準に測るべきであり、Payloadのdecisionを使って都合よく再同期すると、DUT側の誤差を測定器側で吸収してしまう。

## 修正

HDT EVM計算ではGeneric VSAの再同期結果を使わない。

Generic VSAは、

- 表示
- diagnostic
- unknown waveform inspection

用として残してよい。

規格EVMは別経路にする。

推奨構成：

```text
HDT packet synchronization
        |
        v
HDTReferenceEstimate
        |
        +---- Control Header EVM
        |
        +---- Payload EVM
        |
        +---- CFO / timing metrics
```

---

# 4.4 Payload terminating symbolsをEVMへ含める

## 現行

現行の`payload_symbol_count`は、

```text
coded PDU Header
+ coded Payload
+ CRC
```

から計算している。

そのため、

```python
payload_stop = payload_start + payload_symbol_count * sps
```

で測定区間を終了している。

一方、VSGではPayload後に、

```python
payload_termination = map_hdt_symbols(
    np.zeros(2 * definition.bits_per_symbol),
    settings.rate
)
```

として2 terminating symbolsを付与している。

## 問題

現行仕様メモではPayload EVM対象を、

```text
PDU Header / Payload / Terminating symbol
```

としている。

現行VSAではPayload terminating 2 symbolsがEVM範囲外。

## 修正

EVM measurement rangeを、

```text
coded PDU Header
+ coded Payload
+ CRC
+ 2 terminating symbols
```

まで拡張する。

例：

```python
payload_evm_symbol_count = payload_symbol_count + 2
```

ただしprotocol decode用のpayload symbol countと、EVM measurement用のsymbol countは別変数にする。

推奨：

```python
coded_payload_symbol_count
payload_evm_symbol_count
```

---

# 4.5 fractional timing estimatorを追加する

## 現行

`_hdt_training_matches()`では、

```python
for phase in range(integer_sps):
```

として、integer sample phaseを探索している。

16 MS/s / 2 Msym/sでは8 SPSなので、

```text
1 sample = 1/8 symbol
```

のタイミング分解能になる。

## 問題

HDT7.5 16QAMで-22 dBクラスのRMS EVMを評価するには、integer sample固定では粗い可能性がある。

また仕様上、Preamble processingの推定対象にtimingが含まれている。

## 修正

coarse timingは現行integer phase searchを残してよい。

その後、LTSまたはTraining全体を使ってfractional timing refinementを行う。

例：

```text
integer SPS phase search
        ↓
coarse timing
        ↓
fractional delay search / interpolation
        ↓
minimum EVM or maximum reference correlation
```

実装方法は以下のいずれかでよい。

- parabolic interpolation
- fractional-delay interpolation
- oversampled correlation
- local scalar optimization

ただし最終的にはAppendix Cの指定方法に合わせる。

---

# 5. Appendix Cとの直接照合が必要な項目

以下は現行コードの方向性は理解できるが、「Bluetooth準拠EVM」と断定するには原文確認が必要。

## 5.1 Preamble amplitude estimator

現在はHeader / Payload自身のRMSで正規化しているため、まずこれを削除する。

その後、Preamble amplitudeをどのsymbol範囲・どの式で推定するかをAppendix Cと合わせる。

---

## 5.2 Carrier phase / CFO estimator

現行：

```python
phase_error = unwrap(angle(rx * conj(reference)))
polyfit(...)
```

は合理的。

ただし、

- 対象symbol
- weighting
- outlier handling
- phase fitting range
- drift項の有無

をAppendix C原文と照合する。

---

## 5.3 Timing estimator

fractional timingの推定式・使用範囲をAppendix Cと合わせる。

---

## 5.4 Payloadで許可される追加補正

特に確認する。

- Payload gain再推定の可否
- Payload phase再推定の可否
- Payload CFO再推定の可否
- decision-directed optimizationの可否
- modulation別の処理差

Generic VSAのblind / decision-directed synchronizationをそのまま規格EVMへ流用しないこと。

---

# 6. 推奨リファクタ構成

HDT EVM計算を`analyze_bluetooth_hdt_recording()`から分離する。

例：

```text
pluto_sa/vsa/protocol_modes/bluetooth/hdt_measurement.py
```

を新設。

推奨API：

```python
@dataclass(frozen=True)
class HDTReferenceEstimate:
    first_symbol_center_sample: float
    amplitude: float
    phase_rad: float
    phase_step_rad_per_symbol: float
    timing_offset_samples: float
    training_correlation: float


@dataclass(frozen=True)
class HDTEVMResult:
    header_rms_percent: float
    payload_rms_percent: float
    reference: HDTReferenceEstimate
    header_measured_symbols: np.ndarray
    header_reference_symbols: np.ndarray
    payload_measured_symbols: np.ndarray
    payload_reference_symbols: np.ndarray
```

関数例：

```python
estimate_hdt_reference(
    recording,
    training_start,
) -> HDTReferenceEstimate
```

```python
measure_hdt_control_header_evm(
    recording,
    reference,
) -> ...
```

```python
measure_hdt_payload_evm(
    recording,
    reference,
    rate,
    coded_payload_bits,
    include_terminating_symbols=True,
) -> ...
```

---

# 7. EVM計算式

RMS EVM自体は現行の形でよい。

```python
evm = sqrt(
    sum(abs(measured - reference) ** 2)
    / sum(abs(reference) ** 2)
)
```

percent：

```python
evm_percent = 100 * evm
```

dB：

```python
evm_db = 20 * log10(evm)
```

重要なのは式そのものではなく、

```text
measured symbolを作る前に
どのparameterを推定・補正してよいか
```

である。

---

# 8. EVM limit

現行仕様メモに基づくlimit：

| Rate | Control Header | PDU Header / Payload |
|---|---:|---:|
| HDT2 | -10 dB | -10 dB |
| HDT3 | -10 dB | -13 dB |
| HDT4 | -10 dB | -16 dB |
| HDT6 | -10 dB | -19 dB |
| HDT7.5 | -10 dB | -22 dB |

将来的には測定結果にPass/Failも追加してよい。

---

# 9. テスト修正

現行テストは、

```text
自作VSG → 自作VSA
```

のloopbackを中心としており、VSGとVSAが同じ定義を共有するため、規格EVM処理の誤りを検出しにくい。

以下を追加する。

## 9.1 Control Header amplitude error test

理想HDT waveform生成後、Control Headerだけgainを変更する。

例：

```text
Training = 1.0
Control Header = 0.90
Payload = 1.0
```

期待：

```text
Header EVMが明確に悪化する
```

self-normalizationが残っている場合はこのテストで検出できる。

---

## 9.2 Payload amplitude error test

```text
Training = 1.0
Control Header = 1.0
Payload = 0.90
```

期待：

```text
Payload EVMが悪化する
```

---

## 9.3 CFO test

既知のCFOを付加：

```text
+10 kHz
-20 kHz
```

期待：

- TrainingからCFOを推定
- Header / Payloadに同じreferenceを適用
- CFO metricが入力値と一致
- residual EVMが十分低い

---

## 9.4 fractional timing test

IQをfractional sample shiftする。

例：

```text
+0.25 sample
+0.50 sample
```

期待：

- fractional timing estimatorが追従
- integer phaseのみの場合よりEVMが改善

---

## 9.5 Payload terminating symbol test

Payload terminating symbolsだけ意図的に変形する。

期待：

```text
Payload RMS EVMが悪化する
```

これによりterminating symbolsが測定範囲へ含まれていることを確認する。

---

## 9.6 Generic VSA再同期禁止テスト

Payload部にphase ramp / amplitude stepを意図的に入れる。

期待：

HDT規格EVMでは、その誤差がPayload EVMに反映される。

Generic VSA側が独立再同期して誤差を消していないことを確認する。

---

# 10. 既存機能で壊してはいけないもの

以下は回帰させないこと。

- HDT2 / HDT3 / HDT4 / HDT6 / HDT7.5自動判別
- Training detection
- multiple packet detection
- RI decode
- PDU Control decode
- HEC-C
- CRC-32
- payload length自動決定
- FEC / puncturing
- symbol mapping
- QPSK Header vector表示
- Payload vector表示
- packet tree / air bit表示
- spectrum / power表示
- 実IQ `RT_HDT7_5.npz` のdecode

---

# 11. 実装優先順位

優先度A：

1. Control Header self-normalization廃止
2. Payload self-normalization廃止または仕様準拠のgain estimatorへ置換
3. Payload Generic VSA再同期を規格EVM経路から外す
4. Payload terminating 2 symbolsをEVMへ追加

優先度B：

5. fractional timing estimator追加
6. HDT専用Reference Estimatorへリファクタ
7. amplitude / phase / CFO / timingの推定値をmetadataへ保存

優先度C：

8. Appendix C原文との式単位監査
9. Appendix Cに合わせてoptimization処理を微調整
10. rate別Pass/Fail判定追加

---

# 12. 完了条件

以下を満たしたら、HDT RMS EVM実装を「仕様準拠候補」と判断する。

- Trainingから得たreference parametersがHeader / Payloadに一貫して使用される
- Header自身のRMS normalizationをしない
- Payload自身の単純RMS normalizationをしない
- Payload decisionを使った独立carrier/timing再同期を規格EVM経路で行わない
- Payload terminating 2 symbolsをEVMへ含める
- fractional timing correctionを持つ
- Appendix C原文とparameter estimation手順を照合済み
- amplitude / CFO / timing / phase error injection testを通る
- 現行HDT packet decode regression testをすべて維持する

---

# 13. Codexへの実装方針

既存HDT decoder全体を書き換えないこと。

まずHDT EVM処理だけを独立させる。

実装順：

```text
1. 既存analyze_bluetooth_hdt_recording()の同期・decode経路を維持
2. HDTReferenceEstimate追加
3. Preambleからreference estimate生成
4. Header EVMを新reference経路へ変更
5. Payload EVMを新reference経路へ変更
6. terminating symbolsを含める
7. Generic VSA payload_sessionは表示/diagnostic用途のみに限定
8. error injection regression tests追加
9. Appendix Cと照合後にestimatorを最終調整
```

既存のHDT packet decoding correctnessを維持しながら、EVM測定部分のみを規格測定器として独立・厳密化すること。
