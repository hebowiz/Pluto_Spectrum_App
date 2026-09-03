# HDT専用VSA Result Summary 改修指示

## 目的

HDT専用VSAの `Result Summary` を、Bluetooth HDT仕様に基づく送信系RF評価を優先した表示へ改修する。

参照仕様：

- `docs/HDT_VSr03_PR.pdf`
- Vol 6, Part A, Section 3.3 `Radio frequency tolerance`
- Vol 6, Part A, Section 3.6 `LE HDT PHY`
- Vol 6, Part A, Section 3.6.1 `Pre-packet emissions`
- Vol 6, Part A, Appendix C `Transmitter RMS EVM levels`

方針：

- 認証・規格評価に直接対応する項目を上段へ配置する。
- 項目名は仕様書の用語に合わせる。
- 参考情報は下段へ分離する。
- Result Summaryの右端に判定列を追加する。
- 規格測定値と参考値を混同しない。
- 算出方法が仕様書で明記されていない項目について、Generic VSAの値を勝手に「SIG値」として流用しない。

---

# 1. Result Summaryの列構成

現在：

```text
Parameter | Current
```

を以下へ変更する。

```text
Parameter | Current | Limit | Result
```

`Result` は以下のいずれか。

```text
PASS
FAIL
MEASURING
N/A
—
```

色付けする場合：

- PASS: green
- FAIL: red
- MEASURING: yellow
- N/A / —: 通常色またはgray

認証項目ではない参考情報は、

```text
Limit  = —
Result = —
```

とする。

---

# 2. 表示順

Result Summaryの先頭に規格評価項目を並べ、その後にReference Informationを配置する。

推奨順：

```text
Control Header RMS EVM
PDU Header and payload RMS EVM
Center frequency deviation
Center frequency offset change between the preamble and the payload
Symbol timing accuracy
Pre-packet emissions

--- Reference Information ---

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

section header行を実装できる場合は、

```text
RF PHY Measurements
Reference Information
```

の2グループに分ける。

---

# 3. 項目名の変更

現行：

```text
SIG QPSK Header EVM RMS
SIG 16QAM Payload EVM RMS
SIG QPSK Header Average Power
SIG 16QAM Payload Average Power
SIG Relative Power (Payload - Header)
SIG Carrier Frequency Error
SIG Training Correlation
SIG RF Test Eligibility
```

は、規格項目名と参考情報名を明確に分離する。

変更：

```text
SIG QPSK Header EVM RMS
→ Control Header RMS EVM

SIG 16QAM Payload EVM RMS
→ PDU Header and payload RMS EVM

SIG QPSK Header Average Power
→ Control Header Average Power

SIG 16QAM Payload Average Power
→ PDU Header and Payload Average Power

SIG Relative Power (Payload - Header)
→ Relative Power (Payload - Header)

SIG Carrier Frequency Error
→ 廃止
   下記2つの規格値と2つのreference値へ分離

SIG Training Correlation
→ Preamble Correlation

SIG RF Test Eligibility
→ RF Test Eligibility
```

公式規格項目に不要な `SIG` prefixを付けない。

---

# 4. Control Header RMS EVM

仕様：

- Vol 6, Part A, Section 3.6
- Appendix C

表示：

```text
Parameter: Control Header RMS EVM
Current:   5.47 % / -25.2 dB
Limit:     ≤ -10 dB
Result:    PASS
```

全HDT rateでControl Header limitは：

```text
-10 dB
```

## 算出方法

Appendix Cに従う。

Preamble：

```text
STS + GI + LTS
```

を使用し、

```text
α0
φ0
Δω0
T0
```

をjoint least-squaresで推定する。

Preambleで得た、

```text
α0, φ0, Δω0, T0
```

をControl Headerへそのまま適用する。

Control Header側では、

```text
gain再推定
phase再推定
CFO再推定
timing再推定
```

を行わない。

EVM：

```text
sqrt(
    sum(|Qctrl - Sctrl|^2)
    /
    sum(|Sctrl|^2)
)
```

dB：

```text
20 * log10(EVM)
```

Pass条件：

```python
evm_db <= -10.0
```

---

# 5. PDU Header and payload RMS EVM

仕様：

- Vol 6, Part A, Section 3.6
- Appendix C

表示例：

```text
Parameter: PDU Header and payload RMS EVM
Current:   4.01 % / -27.9 dB
Limit:     ≤ -22 dB
Result:    PASS
```

rate別limit：

| Rate | Limit |
|---|---:|
| HDT2 | -10 dB |
| HDT3 | -13 dB |
| HDT4 | -16 dB |
| HDT6 | -19 dB |
| HDT7.5 | -22 dB |

## 算出方法

Appendix Cに従う。

Preambleで求めた、

```text
α0
T0
```

はPayload測定でも固定する。

Payload全体についてのみ、

```text
φ1
Δω1
```

をjoint least-squaresで最適化する。

Payload側で以下は行わない。

```text
gain再最適化
timing再最適化
block単位phase/CFO補正
decision-directed reference
nearest-point reference
Generic VSAによる再同期
```

Reference symbolsは、

```text
RF Test Packetの既知bit系列
または
decode後に再符号化した正しい送信bit系列
```

から生成した固定referenceを使用する。

`measured symbol → nearest constellation point`

でreferenceを生成しない。

## Measurement range

Appendix Cに従い、

```text
Npld = first 1000 Payload symbols
```

をSIG RMS EVM measurement rangeとする。

現在の、

```text
1098 payload symbols + 2 terminating symbols
```

全体をそのままSIG Payload RMS EVMへ入れない。

Terminating symbolsは別途保持可能とするが、Appendix Cで `Npld=1000` とのindex関係が明示されていないため、勝手に1000 symbolsへ追加しない。

---

# 6. Center frequency deviation

仕様：

- Vol 6, Part A, Section 3.3 `Radio frequency tolerance`

仕様文：

```text
For the LE HDT PHY, the deviation of the center frequency
during the packet shall not exceed ±125 kHz.
```

表示例：

```text
Parameter: Center frequency deviation
Current:   +14.103 kHz
Limit:     ±125 kHz
Result:    PASS
```

## 算出

Appendix Cのcarrier-frequency parameterを使用する。

```python
f_preamble = delta_omega_0 / (2*pi)
f_payload  = delta_omega_1 / (2*pi)
```

packet内の評価値として、

```python
center_frequency_deviation =
    value with the greater abs(f_preamble), abs(f_payload)
```

を使用する。

Currentには符号付きで表示してよい。

例：

```text
+14.103 kHz
```

Pass：

```python
abs(center_frequency_deviation_hz) <= 125_000
```

同時にreference情報として、

```text
Preamble Carrier Frequency Error
Payload Carrier Frequency Error
```

も下段へ表示する。

---

# 7. Center frequency offset change between the preamble and the payload

仕様：

- Vol 6, Part A, Section 3.3
- Table 3.5

項目名は仕様文に合わせる。

表示例：

```text
Parameter:
Center frequency offset change between the preamble and the payload

Current:
0.192 kHz

Limit:
≤ 1.2 kHz

Result:
PASS
```

算出：

```python
frequency_offset_change_hz =
    abs(f_payload - f_preamble)
```

rate別limit：

| Rate | Maximum offset change |
|---|---:|
| HDT2 | 3000 Hz |
| HDT3 | 3000 Hz |
| HDT4 | 1800 Hz |
| HDT6 | 1800 Hz |
| HDT7.5 | 1200 Hz |

今回のHDT7.5例なら、

```text
Preamble ≈ +14.103 kHz
Payload  ≈ +13.911 kHz

Offset change ≈ 0.192 kHz
```

となる。

---

# 8. Symbol timing accuracy

仕様：

- Vol 6, Part A, Section 3.6

仕様値：

```text
symbol rate = 2 Msym/s
symbol period = 0.5 us
symbol timing accuracy < ±50 ppm
```

表示：

```text
Parameter: Symbol timing accuracy
Current:   +x.xxx ppm
Limit:     < ±50 ppm
Result:    PASS / FAIL
```

## 注意

Appendix Cの `T0` はsynchronization timing pointであり、symbol-rate errorそのものではない。

したがって、

```text
T0をそのままppmへ変換
Generic VSAのSymbol Rate ErrorをそのままSIG値として流用
```

しない。

`HDT_VSr03_PR.pdf` 内で規定されている測定方法から直接算出方法を確定できない場合は、

```text
Current = N/A
Result  = N/A
```

としてよい。

仕様書または資格試験Test Specificationで測定アルゴリズムを確認してから有効化する。

推測実装でPASS/FAILしない。

---

# 9. Pre-packet emissions

仕様：

- Vol 6, Part A, Section 3.6.1 `Pre-packet emissions`

項目名：

```text
Pre-packet emissions
```

仕様：

```text
平均packet output powerから -35 dBのlevelを最初に超えた時点
↓
平均packet output powerから -1 dBのlevelへ最初に到達した時点

この時間 <= 4 us
```

表示例：

```text
Parameter: Pre-packet emissions
Current:   1.82 us
Limit:     ≤ 4 us
Result:    PASS
```

## 算出

IQ Power traceからlinear powerでaverage packet output powerを求める。

```python
threshold_low  = average_packet_power_dbm - 35.0
threshold_high = average_packet_power_dbm - 1.0
```

packet開始前から探索し、

```text
t_low  = first crossing above threshold_low
t_high = first crossing/reach of threshold_high
```

とする。

```python
pre_packet_emission_time = t_high - t_low
```

Pass：

```python
pre_packet_emission_time <= 4e-6
```

capture内に十分なpre-trigger / idle区間がなく、最初のcrossingが判断できない場合：

```text
Result = N/A
```

とする。

---

# 10. Reference Information

以下は有用だが、HDTの規格Pass/Fail項目として扱わない。

```text
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

Limit：

```text
—
```

Result：

```text
—
```

とする。

`Header / Payload Relative Power`はEDRのようなHDT固有の規格limitがSection 3.6には定義されていないため、reference information扱いとする。

---

# 11. EVM 1500 packets aggregation

Vol 6, Part A, Section 3.6では、

```text
RMS EVM shall be measured over 1500 packets.
```

かつ、

```text
Control Header RMS EVM
payload RMS EVM
```

は各packetでlimit未満である必要がある。

そのため既存multi-packet analyzerを利用し、

```text
eligible packet count
header pass count
payload pass count
first failure
worst header EVM
worst payload EVM
```

を保持する。

Reference Informationに：

```text
RMS EVM Packets Evaluated: 19 / 1500
```

を表示する。

1500 packet未満ではoverall qualification verdictを `PASS` と断定しない。

必要ならstatus barまたは別Statistics tabで、

```text
MEASURING 19 / 1500
```

と表示する。

selected/current packetのResult Summaryでは個々のpacketについてPASS/FAILを表示してよい。

---

# 12. VSAから外す測定

以下はSpectrum Analyzer / RF power measurement側の責務とし、HDT VSA Result Summaryへ無理に入れない。

```text
Modulation spectrum / 6 dB bandwidth
In-band spurious emission
Out-of-band spurious emission
Adjacent channel power
```

Vol 6, Part A, Section 3.2の測定はSA側で実装する。

Output Power / Pmaxについても、現在のHeader/Payload Average Powerをそのまま認証用Output Powerとして扱わない。

将来、正確なRFPHY Test SpecificationのOutput Power test条件とPluto amplitude calibrationを満たした時点で別測定として追加する。

---

# 13. UI例

最終イメージ：

```text
Parameter                                               Current               Limit       Result
------------------------------------------------------------------------------------------------
Control Header RMS EVM                                  5.47 % / -25.2 dB     ≤ -10 dB    PASS
PDU Header and payload RMS EVM                          4.01 % / -27.9 dB     ≤ -22 dB    PASS
Center frequency deviation                              +14.103 kHz           ±125 kHz    PASS
Center frequency offset change between the
  preamble and the payload                              0.192 kHz             ≤ 1.2 kHz   PASS
Symbol timing accuracy                                  N/A                   ±50 ppm     N/A
Pre-packet emissions                                    1.82 us               ≤ 4 us      PASS

Reference Information
------------------------------------------------------------------------------------------------
Detected PHY                                            HDT7.5                —           —
RF Test Eligibility                                     Eligible              —           —
Preamble Carrier Frequency Error                        +14.103 kHz           —           —
Payload Carrier Frequency Error                         +13.911 kHz           —           —
Control Header Average Power                            +1.788 dBm            —           —
PDU Header and Payload Average Power                    +1.739 dBm            —           —
Relative Power (Payload - Header)                       -0.048 dB             —           —
Preamble Correlation                                    60.83 %               —           —
RMS EVM Packets Evaluated                               19 / 1500             —           —
```

---

# 14. 回帰条件

以下を壊さないこと。

```text
HDT2 / HDT3 / HDT4 / HDT6 / HDT7.5 rate判定
Preamble detection
Control Header decode
HEC-C
PDU Header decode
Payload decode
CRC-32
FEC / puncturing
multi-packet detection
Packet List
Packet Analysis tree
IQ Power
Spectrum
Vector / Symbol Plot
```

`RT_HDT7_5.npz` では少なくとも、

```text
Control Header RMS EVM ≈ 5%級
PDU Header and payload RMS EVM ≈ 4%級
Preamble/Payload frequency difference ≈ 0.19 kHz
```

となることを回帰確認する。

---

# 15. 重要

Result Summaryの目的は、

```text
Generic VSA的な解析値を多数並べること
```

ではなく、

```text
HDT PHYのRF規格評価項目を規定方法で測定し、
現在値・limit・PASS/FAILを即座に確認できること
```

とする。

Genericな解析・diagnosticはGeneric VSAへ任せる。

Bluetooth Dedicated Analyzerには、

```text
規格測定
Packet Decode
RF Test statistics
```

のみを残す。
