# Classic DECT PHY / RF Test Specification
## 自作VSA / DECTテスター実装ガイド

更新: 2026-09-04

---

## 0. この文書の目的とスコープ

本書は、Bluetooth専用VSAと同様の考え方で **Classic DECTのPHY解析器 / RFテスターを自作する**ために、以下のETSI規格を実装視点で整理したものです。

主対象:

- Classic DECT Common InterfaceのPhysical Layer
- RF送信試験
- RF受信試験
- バースト / タイミング測定
- GFSK変調解析
- Higher Level Modulation (HLM) のEVM
- 自作VSAに必要な同期・周波数推定・シンボルタイミング処理
- 将来のActive Lower Tester（信号生成＋BER試験）への拡張

対象外:

- GAP等の相互接続Profile
- アプリケーション層の相互接続性
- 製品認証手続きそのもの

> **重要**
>
> DECT規格は、シンボル同期や変調解析の「測定対象」「測定区間」「基準となるタイミングや周波数」をかなり具体的に規定しています。
>
> 一方で、Gardner、Mueller & Müller、Early-Late、特定PLL、特定の相関器構成など、**DSPアルゴリズムそのものを一意に指定しているわけではありません**。
>
> したがって自作VSAでは、
>
> 1. 規格が定める測定量と時間基準を厳密に再現する
> 2. その測定量を得るための同期・補間・推定アルゴリズムは独自実装する
>
> という切り分けが適切です。

---

# 1. 主要規格

## 1.1 Physical Layer

**ETSI EN 300 175-2 V2.9.1 (2022-03)**
Digital Enhanced Cordless Telecommunications (DECT);
Common Interface (CI); Part 2: Physical Layer (PHL)

主に規定されるもの:

- RFキャリア
- スロット / フレーム構造
- パケット構造
- Synchronization field
- GFSK変調
- 送信電力
- 周波数精度
- タイミング精度
- 受信性能
- Higher Level Modulation

Official:
https://www.etsi.org/deliver/etsi_en/300100_300199/30017502/02.09.01_60/en_30017502v020901p.pdf

---

## 1.2 RF Test Specification

**ETSI EN 300 176-1 V2.4.1 (2022-11)**
Digital Enhanced Cordless Telecommunications (DECT);
Test specification; Part 1: Radio

主に規定されるもの:

- RF周波数測定方法
- タイミング測定方法
- Power-Time Template
- 送信電力
- GFSK変調測定
- 不要輻射
- 感度
- BER / FER
- 干渉耐性
- Blocking
- Receiver Intermodulation
- Lower Tester用テスト信号
- 測定不確かさ

Official:
https://www.etsi.org/deliver/etsi_en/300100_300199/30017601/02.04.01_60/en_30017601v020401p.pdf

---

## 1.3 Active Testerを作る場合に追加で必要になる規格

**ETSI EN 300 175-3**
Common Interface; Part 3: Medium Access Control (MAC) layer

Passive VSAで波形を解析するだけなら、MAC全体を実装する必要はありません。

ただし、規格試験のようにDUTを

- 特定チャネルに固定
- 特定slotで送受信
- 規定bit patternをloopback
- BER測定
- handover / diversityを抑制

させる **Active Lower Tester** を作る場合は、PHYだけでは不十分で、MACのTest Message / loopback制御まで扱う必要があります。

---

# 2. PHYクイックリファレンス

## 2.1 基本シンボルレート

基本シンボルレート:

```text
Rs = 1.152 Msymbol/s
```

シンボル時間:

```text
Ts = 1 / 1.152e6
   = 0.8680556 µs
```

基本GFSKでは1 symbol = 1 bitなので、

```text
Rb = 1.152 Mbit/s
```

です。

---

## 2.2 フレーム / slot

1 frame:

```text
10 ms
11520 symbols
```

1 frame = 24 full slots。

| Slot type | Symbols | Time |
|---|---:|---:|
| Half slot | 240 | 208.333 µs |
| Full slot | 480 | 416.667 µs |
| Double slot | 960 | 833.333 µs |

通常のTDD構成では:

```text
K = 0 ... 11   RFP -> PP
K = 12 ... 23  PP  -> RFP
```

- RFP: Radio Fixed Part（一般には親機側）
- PP: Portable Part（一般には子機側）

---

# 3. RFキャリア

## 3.1 ETSI main DECT band

欧州系Classic DECTのmain band:

```text
1880 MHz ～ 1900 MHz
```

main carrierは10波です。

```text
Fc = F0 - c × 1.728 MHz

F0 = 1897.344 MHz
c  = 0 ... 9
```

| c | Center frequency |
|---:|---:|
| 0 | 1897.344 MHz |
| 1 | 1895.616 MHz |
| 2 | 1893.888 MHz |
| 3 | 1892.160 MHz |
| 4 | 1890.432 MHz |
| 5 | 1888.704 MHz |
| 6 | 1886.976 MHz |
| 7 | 1885.248 MHz |
| 8 | 1883.520 MHz |
| 9 | 1881.792 MHz |

Nominal RF channel widthは1.728 MHzです。

> **日本向け注意**
>
> 日本国内のDECT系製品を対象にする場合、この1880–1900 MHz / 10 carrierを固定値として実装しないこと。
>
> 日本ではARIB STD-T101が関係し、周波数配置・法規要求は地域依存です。
>
> PHY解析エンジンとcarrier tableを分離し、carrier planを設定ファイル化する設計を推奨します。

## 3.2 JP-DECT carrier plan

JP-DECTテスターの周波数選択には、以下の12 carrierを使用します。carrier識別子は
`F7`～`Fb`、`F0`～`F6`の表記を維持し、単純な連番へ置き換えません。

| JP-DECT carrier | Center frequency |
|---:|---:|
| F7 | 1885.248 MHz |
| F8 | 1886.976 MHz |
| F9 | 1888.704 MHz |
| Fa | 1890.432 MHz |
| Fb | 1892.160 MHz |
| F0 | 1893.888 MHz |
| F1 | 1895.616 MHz |
| F2 | 1897.344 MHz |
| F3 | 1899.072 MHz |
| F4 | 1900.800 MHz |
| F5 | 1902.528 MHz |
| F6 | 1904.256 MHz |

実装ではPHY解析処理から独立したregional carrier planとして保持し、DECT専用VSAの
周波数選択リストはこの表をsource of truthとして生成します。

---

# 4. 周波数精度

PHY要求の代表値:

### RFP

```text
Carrier frequency error:
±50 kHz
```

### PP

通常:

```text
±50 kHz
```

非送信状態から送信状態へ移行した直後1秒以内には緩和条件があり:

```text
±100 kHz
```

となる場合があります。

さらにslot間のcenter frequency変動について:

```text
maximum change:
15 kHz / slot
```

が規定されています。

---

# 5. 基本GFSK

## 5.1 変調方式

Classic DECTのbasic modulation:

```text
GFSK
Gaussian BT = 0.5 nominal
```

bit mapping:

```text
1 -> positive frequency deviation
0 -> negative frequency deviation
```

nominal peak frequency deviation:

```text
Δf = ±288 kHz
```

したがって、通常のCPFSKとして変調指数を計算すると:

```text
h = 2Δf / Rs
  = 2 × 288 kHz / 1.152 MHz
  = 0.5
```

これは規格値から導出した値です。

---

# 6. GFSKの周波数偏移要求

規格ではbit patternにより2ケースに分けています。

## Case A

対象:

```text
0000 1111 0000 1111 ...
```

Peak frequency deviation:

```text
259 kHz < |Δf| < 403 kHz
```

---

## Case B

Case A以外で、Digital Sum Variation (DSV) が規定範囲内となるsequence。

```text
202 kHz < |Δf| < 403 kHz
```

DSVは概念的には、

```text
bit 1 -> +1
bit 0 -> -1
```

を累積した値です。

規格では対象sequenceについて:

```text
|DSV|max <= 64
```

を使用します。

`101010...` のようなalternating patternはCase Bです。

---

# 7. DECT packetのSynchronization field

## 7.1 S-field

Synchronization fieldは常に:

```text
32 bits
p0 ... p31
```

です。

構成:

```text
p0  ... p15 : preamble
p16 ... p31 : packet synchronization word
```

S-fieldは **常に2-level modulation** を使用します。

---

## 7.2 RFP sync pattern

```text
1010 1010 1010 1010 1110 1001 1000 1010
```

---

## 7.3 PP sync pattern

```text
0101 0101 0101 0101 0001 0110 0111 0101
```

RFPとPPは互いにbit inverseです。

これは自作VSAにとって非常に便利です。

例えばfrequency discriminator出力に対して同じreference waveformを使い、

- correlation正方向 -> RFP
- correlation負方向 -> PP

のように識別する実装が可能です。

ただし、これは **推奨実装例でありETSI指定アルゴリズムではありません**。

---

# 8. パケット種別

代表的なphysical packet:

| Packet | 概要 | 主なsymbols |
|---|---|---:|
| P00 | Short physical packet | 96 |
| P32 | Basic physical packet | 420 / optional Z込み424 |
| P00j | Variable length packet | 100+j / optional Z |
| P80 | Double-slot physical packet | 900 / optional Z込み904 |

P32はClassic DECTで最も基本的なtelephony packetです。

---

# 9. シンボル同期について何が「規定」されているか

ここが自作VSA設計で特に重要です。

## 9.1 PHYが要求するもの

EN 300 175-2ではreceiverはSynchronization fieldを利用して

- slot synchronizationをacquire
- slot synchronizationをconfirm

することが要求されています。

また規格中には、

- sync fieldをclock synchronizationに使用できる
- 一つのcorrelatorでRFP / PP sync wordを検出・識別可能

という趣旨の記述があります。

ただし、これらは **実装方法を一意に強制するものではありません**。

---

## 9.2 規格で指定されていないもの

少なくともbasic GFSKについて、以下のようなDSP実装は指定されていません。

- Gardner Timing Error Detector
- Mueller & Müller
- Early-Late Gate
- specific matched-filter taps
- PLL loop bandwidth
- timing recovery loop order
- interpolation filter
- correlation threshold
- fractional timing estimator
- CFO estimatorの具体式
- instantaneous-frequency estimatorの具体式

つまり、

> **「どのアルゴリズムでsymbol timingを得るか」は実装自由**

です。

---

# 10. ただし「測定上のbit位置」は自由ではない

EN 300 176-1のsampling methodでは、取り込んだRF signalのsample列から、

```text
known received bit pattern
```

を利用してpacket内のbit positionを計算し、そのbit positionを

- RF frequency
- RF phase
- RF power

測定の時間基準として使用します。

さらにp0は、多数のsampleを利用して高精度に計算することを想定しています。

したがって、

```text
burst envelopeを検出したsample = p0
```

としてはいけません。

---

# 11. p0の厳密な定義

EN 300 176-1では、試験測定に用いるp0がかなり厳密に定義されています。

概念的には:

1. Synchronization fieldの後半16 bit、すなわちpacket sync wordを見る
2. その最初のbitへ遷移する直前に、FM信号がnominal channel frequencyをcrossする時刻を求める
3. その時刻から **16 bit periods前** をp0とする

つまりp0は、

```text
receiverが「packetを検出した瞬間」
```

ではありません。

また、

```text
power envelopeがthresholdを超えた瞬間
```

でもありません。

### 実装上の意味

RFP preamble:

```text
1010101010101010
```

PP preamble:

```text
0101010101010101
```

はいずれもalternatingなので、preamble終了からpacket sync wordへ入る境界近傍のfrequency crossingを精密に推定できます。

自作VSAでは、この規格上のp0を内部のmaster timing referenceにするのが適切です。

---

# 12. 自作VSA向けsymbol synchronization案
## 非規定・推奨実装

以下はETSI指定アルゴリズムではなく、自作VSA向けの実装案です。

---

## 12.1 RF burst detection

まず:

```text
P[n] = I[n]^2 + Q[n]^2
```

からburst envelopeを検出します。

これはあくまでcoarse acquisitionに使用します。

ここで得た時刻をp0としてはいけません。

---

## 12.2 Frequency discriminator

complex IQ:

```text
x[n] = I[n] + jQ[n]
```

に対して:

```text
dphi[n] = angle(x[n] * conj(x[n-1]))
```

instantaneous frequency:

```text
f_inst[n] = Fs / (2π) * dphi[n]
```

とします。

GFSKなので、これは非常に扱いやすい観測量です。

noise耐性を改善する場合は、

- phase unwrap
- multi-sample phase difference
- local polynomial phase fit
- low-pass filtering

などを追加できます。

これらもETSI規定ではありません。

---

## 12.3 Coarse sync

S-fieldの既知32 bitをGFSK変調したreference waveform:

```text
m_ref(t)
```

を生成します。

そのうえで、

```text
corr(τ) = Σ f_inst(t) * m_ref(t - τ)
```

を最大化するτを探索します。

RFP / PPがinverseなのでcorrelationの符号も識別に利用できます。

---

## 12.4 Fractional symbol timing

sample boundaryだけではなくfractional timingを推定します。

一例:

```text
J(τ, Δf0, A)
 = Σ w[n] * (
     f_inst[n]
     - Δf0
     - A * m_ref(t[n] - τ)
   )^2
```

を最小化します。

推定対象:

- τ: fractional timing
- Δf0: frequency offset
- A: modulation amplitude

ただし注意点があります。

このleast-squaresで求めたΔf0を、そのまま **EN 300 176-1のcarrier-frequency test値** に使用するのは避けます。

carrier frequencyの規格試験には別途、規格指定のbit patternとaveraging procedureがあります。

このfitはあくまで同期推定用です。

---

## 12.5 p0 refinement

coarse correlationでpacket位置を得た後、

- S-field全体
- 特にpreamble / sync境界付近
- interpolationしたfrequency crossing

を用いて、規格定義のp0へ補正します。

推奨:

```text
coarse burst detect
    ↓
S-field correlation
    ↓
fractional timing fit
    ↓
nominal-frequency crossing interpolation
    ↓
exact p0
```

です。

---

# 13. Carrier Frequency試験
## EN 300 176-1 clause 7

ここは測定方法がかなり具体的です。

## 13.1 test pattern

EUTにloopbackさせるbit pattern:

```text
0000 1111 0000 1111 ...
```

を使用します。

---

## 13.2 基本手順

概略:

1. 初期carrierはc=5
2. DUTをactive_lockedへ
3. active_locked後、原則1秒以上経過した信号をcapture
4. loopback bitのabsolute frequencyを測定
5. 各bitで得たfrequencyを平均
6. 1 packetのcarrier-frequency measurementとする
7. packet typeに応じて複数回repeat
8. 全測定値のmeanをcenter frequencyとする
9. extreme conditionでもrepeat
10. c=0 / c=9でもrepeat

PPについてはnon-transmit -> transmit移行後1秒以内の条件も別途確認します。

---

## 13.3 規定repeat数

| Burst / slot type | Repetitions |
|---|---:|
| A-field only | 100 |
| Half slot | 40 |
| Full slot | 10 |
| Long slot | 5 |
| Double slot | 5 |

---

## 13.4 自作VSAへの意味

FFT peakだけで:

```text
carrier = max(|FFT|)
```

とするのは、規格reference procedureとは一致しません。

規格試験相当を目指すなら:

```text
known bit timing
 -> bit-aligned absolute-frequency measurement
 -> packet内average
 -> repeated packet average
```

とする必要があります。

ただし、

- phase差からfrequencyを求めるか
- polynomial phase fitを使うか
- zero crossingをinterpolateするか

といった **各bitのfrequency estimator自体は一意に規定されていません**。

---

# 14. GFSK Modulation試験
## EN 300 176-1 clause 11

Basic modulationの解析は大きく4 partに分かれます。

測定系のbandwidth要求:

```text
>= 3 MHz
```

です。

これは自作SDR/VSAのcapture bandwidth設計上かなり重要です。

---

# 15. Modulation Test Part 1
## 00001111 pattern

test pattern:

```text
0000 1111 0000 1111 ...
```

を使用します。

### Peak deviationを測定する有効区間

0 -> 1 または 1 -> 0 transitionの直後はGaussian shapingによるtransition regionです。

規格では、同一bit runについて概念的に:

```text
transition
    ↓
1 bit time待つ
    ↓
[ measurement interval ]
    ↓
次のtransitionの1 bit time前で終了
```

とします。

したがって0000 runなら中央側bit区間のみを利用し、transition直近をpeak deviation判定から外します。

これは非常に重要です。

単純にpacket全体の:

```text
max(f_inst)
min(f_inst)
```

を取る実装は規格procedureと一致しません。

### 判定

Case A:

```text
259 kHz < |Δf| < 403 kHz
```

---

# 16. Modulation Test Part 2
## Other bit patterns

規格のFigure 27～31で定められたtest packet patternを使用します。

packet typeに応じて、

- alternating bits
- consecutive 1
- consecutive 0

を組み合わせ、Case Bのdeviationを評価します。

measurement intervalの考え方はPart 1と同じで、

```text
transitionから1 bit後
～
次transitionの1 bit前
```

です。

判定:

```text
202 kHz < |Δf| < 403 kHz
```

### 実装上の推奨

内部的には各eligible bitについて:

```text
bit index
expected bit
positive/negative
peak deviation
mean deviation
timing offset
```

を保存しておくと、規格判定だけでなくdebug用VSAとして非常に有用です。

---

# 17. Modulation Test Part 3
## 0101 pattern

test pattern:

```text
0101 0101 0101 ...
```

を使用します。

測定window:

```text
first transitionから1 bit後
～
last transitionの1 bit前
```

規格は、

- first 16 sync bits
- loopback field

内のbit periodを対象としてpeak deviationを求めます。

これはCase Bに該当します。

---

# 18. Modulation Test Part 4
## Frequency drift

test pattern:

```text
0101 0101 0101 ...
```

を使用。

active_lockedから1秒以上経過後に測定します。

analysis bandwidth:

```text
>= 3 MHz
```

### frequency measurement window

packet先頭のSynchronization field:

```text
最初の16 bits
```

のうち、

```text
last 14 bits
```

の平均周波数を求めます。

packet末尾側loopback fieldについては、

```text
last 16 bits
```

のうち、

```text
first 14 bits
```

の平均周波数を求めます。

両者の差からdriftを求めます。

repeat:

```text
200 measurements
```

### 判定

PHY上の要求は:

```text
maximum center-frequency change:
15 kHz / slot
```

Test Specificationでは測定不確かさを加味したverdict rangeが設定されます。

実装上は少なくとも:

```text
drift [Hz]
drift [Hz/slot]
start-window mean frequency
end-window mean frequency
```

を個別に保持することを推奨します。

---

# 19. 「変調解析アルゴリズム」は規定されているか

## Basic GFSK

結論:

```text
No
```

ただし、**測定procedureはかなり具体的**です。

規定されるもの:

- test pattern
- measurement bandwidth
- bit timing reference
- measurement interval
- transitionから除外する期間
- carrier-frequency reference
- repeat回数
- limit

規定されないもの:

- IQからinstantaneous frequencyを求める具体式
- phase unwrap方式
- filtering
- timing interpolator
- peak search interpolation
- sample rate
- samples/symbol
- ML estimator
- least-squares estimator

つまり、

> **算法ではなく、測定量と測定条件の互換性を規定する**

という構造です。

EN 300 176-1自体も、規格記載方法と同等の結果を得られるalternative test methodを許容しています。

---

# 20. 推奨sample rate
## 非規定

DECT basic rate:

```text
1.152 Msymbol/s
```

なので整数samples/symbolにすると実装しやすくなります。

例:

```text
Fs = 9.216 MS/s  -> 8 samples/symbol
Fs = 18.432 MS/s -> 16 samples/symbol
```

規格上の指定値ではありません。

BT専用VSA同様、内部でresampleしてinteger SPSへ揃える設計が扱いやすいです。

### 個人的に推奨する構成

初期実装:

```text
8 SPS
```

精密timing / frequency crossing解析:

```text
16 SPS + fractional interpolation
```

程度が扱いやすいでしょう。

重要なのはnative sample rateそのものより、

```text
analysis bandwidth >= 3 MHz
```

とmeasurement uncertaintyを満たせることです。

---

# 21. Packet timing

PHYではRFP / PPそれぞれのpacket timing accuracyが規定されています。

代表値:

### RFP

packet transmission jitter:

```text
< ±1 µs
```

p0からpacket内の他symbol位置:

```text
±0.1 µs
```

程度の精度要求があります。

### PP

ideal timingに対するp0:

```text
±2 µs
```

packet内部のrelative timing:

```text
±0.1 µs
```

---

# 22. Reference timer accuracy

代表要求:

### PP

```text
better than 25 ppm
```

### Multi-channel RFP

nominal:

```text
5 ppm
```

extreme:

```text
10 ppm
```

### Single-channel RFP

extreme:

```text
10 ppm
```

---

# 23. Power-Time Template

DECTはTDMA burstなので、Power-Time解析は重要なPHY試験です。

代表値:

```text
attack time  < 10 µs
release time < 10 µs
```

packet期間中のpower flatness、burst直前・直後のpower、idle期間のemissionが規定されています。

代表的には:

- p0からpacket endまで nominal transmit powerに対し -1 dB以上
- packet主要区間は nominal +1 dBを超えない
- p0近傍のovershootにもupper limit
- packet end後のpower decay
- idle slot中の残留power

などを評価します。

自作VSAでは以下の表示が有効です。

```text
Power vs Time
p0 marker
packet-end marker
upper/lower mask
attack time
release time
overshoot
idle leakage
```

---

# 24. Transmit power

Classic DECT PHYではnominal transmit powerについて最大値が規定されます。

代表的なupper limit:

```text
250 mW
≈ 24 dBm
```

ただし、実際の製品に適用される出力上限・EIRP等は地域法規と組み合わせて考える必要があります。

自作テスターでは:

```text
RF path loss
external attenuator
coupler loss
cable loss
calibration offset
```

を必ず独立parameter化することを推奨します。

---

# 25. Unwanted emissions due to modulation

DECT band内の他channelへのemissionも規定されています。

wanted channelをMとした代表limit:

| Channel offset | Maximum power |
|---|---:|
| M ± 1 | 160 µW |
| M ± 2 | 1 µW |
| M ± 3 | 80 nW |
| Other DECT channels | 40 nW |

測定は1 MHz帯域内のpower integrationを基本とします。

Test Specificationでは実装上、

- 100 kHz RBW
- 1 MHz span/integration
- burst同期
- peak hold

等を用いるprocedureが記述されています。

### 自作SDR上の注意

wideband IQを一括captureしてdigital channel powerで算出する方式でも、reference methodと等価な結果およびuncertaintyを示せるならalternative methodとして成立し得ます。

ただし、

```text
1 MHz integration bandwidth
burst内measurement window
```

は規格条件に合わせる必要があります。

---

# 26. Transmitter transient emissions

TX burstの立上り / 立下りに伴う隣接channel emissionも別途規定されます。

代表limit:

| Channel offset | Maximum |
|---|---:|
| M ± 1 | 250 µW |
| M ± 2 | 40 µW |
| M ± 3 | 4 µW |
| Other | 1 µW |

measurement bandwidthは100 kHzを用い、1 MHz範囲を評価するreference procedureがあります。

---

# 27. Spurious emissions

allocated transmitterのout-of-band spuriousには広い周波数範囲の要求があります。

代表limit:

```text
< 1 GHz     : 250 nW
> 1 GHz     : 1 µW
```

測定帯域はband edgeからのoffsetにより変わります。

例:

| Offset from band edge | Measurement BW |
|---|---:|
| 0–2 MHz | 30 kHz |
| 2–5 MHz | 30 kHz |
| 5–10 MHz | 100 kHz |
| 10–20 MHz | 300 kHz |
| 20–30 MHz | 1 MHz |
| 30 MHz以上 | 3 MHz |

要求周波数域は最大12.75 GHzまで及びます。

### Pluto等を使う場合

Pluto単体では全spurious conformity testを完結できません。

したがって自作システムは、

```text
DECT VSA
```

と

```text
Full RF conformity tester
```

を分けて考えるのがよいです。

前者ならPlutoクラスでも十分実用的です。

---

# 28. Receiver sensitivity

代表的なbasic sensitivity requirement:

```text
BER = 0.001
```

となるwanted signal levelが:

```text
<= -83 dBm
```

であること。

reference DECT signalには:

```text
frequency error ±50 kHz
```

を与える条件があります。

Test Specificationでは、

- center carrier付近
- band edge carrier
- frequency offset

を組み合わせて評価します。

---

# 29. Reference BER / FER

receiver input:

```text
>= -73 dBm
```

において代表要求:

```text
BER < 1e-5
FER < 5e-4
```

があります。

Active Testerを作る場合、DUTからloopbackされたbit列を利用する構成になります。

---

# 30. Interference performance

wanted:

```text
-73 dBm
```

BER requirement:

```text
BER < 1e-3
```

代表interferer limits:

| Interferer channel | Level |
|---|---:|
| Same channel M | -84 dBm |
| M ± 1 | -60 dBm |
| M ± 2 | -39 dBm |
| Other DECT channel | -33 dBm |

DECT channel直外側のnominal carrier positionsも試験対象に含まれます。

---

# 31. Blocking
## Same-time blocking

wanted:

```text
-80 dBm
```

BER:

```text
< 1e-3
```

CW blockerを広い周波数範囲で掃引します。

代表level:

| Frequency region | Blocker |
|---|---:|
| far, 25 MHz～band-100 MHz | -23 dBm |
| band edgeから5～100 MHz | -33 dBm |
| close-in, wantedから十分離れた近傍 | -43 dBm |
| upper side 5～100 MHz | -33 dBm |
| far upper～12.75 GHz | -23 dBm |

full complianceを目指す場合、広帯域signal sourceが必要です。

---

# 32. Different-time blocking

DECT特有で面白い試験です。

強いinterfering DECT burstとwanted burstを **異なるslot** に配置し、receiverが強入力後に正常復帰できるかを評価します。

代表条件:

```text
interfering burst: -14 dBm
wanted signal   : -83 dBm
BER             : < 1e-3
```

強いburstの後に受信slotが来るため、

- AGC recovery
- LNA/VGA saturation recovery
- DC transient
- baseband recovery

などの実力が効きます。

自作テスターとして非常に面白い項目です。

---

# 33. Receiver intermodulation

代表条件:

```text
wanted            = -80 dBm
DECT-like interferer = -48 dBm
CW interferer        = -48 dBm
BER                < 1e-3
```

非隣接carrierの組み合わせで評価します。

例:

```text
wanted 5 / interferers 7,9
wanted 5 / interferers 3,1
wanted 0 / interferers 2,4
wanted 9 / interferers 7,5
```

Active Lower Tester化する場合は、最低でも:

```text
wanted DECT VSG
DECT-like interferer VSG
CW signal generator
```

相当の3 signalを合成できる構成が必要になります。

---

# 34. Lower Tester用DECT-like signal

EN 300 176-1では試験器側の信号にも要件があります。

DECT-like modulation:

```text
GFSK
BT = 0.5
1.152 Mbit/s
```

dataは最低:

```text
2^9 - 1 = 511 bits
```

程度のPRBSを用います。

wanted D-M2 test signalではITU-T O.153に基づく511-bit以上のPRBSを連続反復する構成が定義されています。

signal generator carrier accuracyの目安:

```text
±5 kHz
```

です。

---

# 35. Active BER Testerの構成

Active Lower Testerを作るなら:

```text
        +---------------------+
        | DECT protocol / MAC |
        +----------+----------+
                   |
        test / loopback control
                   |
      +------------v------------+
      |      DECT waveform      |
      |       generator         |
      +------------+------------+
                   |
                  RF
                   |
                 DUT
                   |
                  RF
                   |
      +------------v------------+
      |      DECT VSA / Rx      |
      | sync / demod / decoder  |
      +------------+------------+
                   |
             BER / FER
```

が基本構成です。

---

# 36. Loopback対象bit

Radio Test Specificationではpacket typeごとのloopback bit範囲も規定されています。

代表例:

| Packet | Loopback field |
|---|---|
| A-field only | a16 ... a47 |
| Half slot | b0 ... b79 |
| Full slot | b0 ... b319 |
| Long slot | b0 ... b639 |
| Double slot | b0 ... b799 |

BER testerでは、この対応をpacket parserと共通定義にしておくとよいでしょう。

---

# 37. Higher Level Modulation
## Optional scope

Classic DECTにはbasic GFSKだけでなくHigher Level Modulationも規定されています。

代表modulation:

| Modulation | Bits/symbol相当 | Data rate |
|---|---:|---:|
| GFSK | 1 | 1.152 Mbit/s |
| π/2-DBPSK | 1 | 1.152 Mbit/s |
| π/4-DQPSK | 2 | 2.304 Mbit/s |
| π/8-D8PSK | 3 | 3.456 Mbit/s |
| 16-QAM | 4 | 4.608 Mbit/s |
| 64-QAM | 6 | 6.912 Mbit/s |

初期の「コードレス電話用Classic DECT VSA」なら、まずGFSKだけで十分実用的です。

---

# 38. HLM pulse shaping

HLMではreference receiver / transmitter系としてRoot Raised Cosine shapingが規定されます。

```text
RRC roll-off α = 0.5
symbol rate = 1.152 Msymbol/s
```

です。

---

# 39. HLM EVM

代表EVM limit:

| Modulation | RMS EVM limit |
|---|---:|
| π/2-DBPSK | < 0.14 |
| π/4-DQPSK | < 0.14 |
| π/8-D8PSK | < 0.09 |
| 16-QAM | < 0.047 |
| 64-QAM | < 0.026 |

EVMは1000 symbolsを対象として評価します。

error vector powerを、constellationで最も外側にあるsymbol vector magnitude:

```text
Smax
```

でnormalizeしたRMS量です。

---

# 40. HLMでは同期アルゴリズムが規定されるか

これはbasic GFSKと同様、重要な点です。

規格はreference receiverについて、

- carrier lock
- symbol timing recovery
- amplitude adjustment

を行ったうえでEVMを測定することを要求します。

しかし、

```text
どのcarrier recovery algorithmを使うか
どのtiming recovery algorithmを使うか
どのloop bandwidthを使うか
```

までは指定していません。

つまり:

> **「carrier / timing / amplitudeを回復したreference receiver出力で測れ」は規定されるが、その回復器の内部アルゴリズムは規定されない**

という理解が正確です。

---

# 41. HLM EVM向け推奨実装
## 非規定

自作VSAなら例えば:

1. RRC matched filtering
2. coarse CFO estimation
3. known preamble / sync correlation
4. fractional timing optimization
5. residual CFO / phase fit
6. complex amplitude normalization
7. differential symbol mapping
8. EVM calculation

とできます。

さらにoffline VSAなら、

```text
minimize EVM over:
  timing τ
  residual CFO Δf
  common phase φ
  gain A
```

というleast-squares / maximum-likelihood寄りの実装が可能です。

ただし、過剰な補正によりDUTの実エラーを消してしまわないよう、

```text
規格reference receiverが補正を許している量
```

と

```text
DUT impairmentとして残すべき量
```

を明確に分離する必要があります。

---

# 42. Test Specificationの主要RF試験一覧

相互接続関連を除いた、自作PHY/RF testerで重要な項目です。

| # | Test | EN 300 176-1 |
|---:|---|---|
| 1 | RF carrier accuracy / stability | §7 |
| 2 | Timing jitter | §8.3 |
| 3 | RFP reference timing accuracy | §8.4 |
| 4 | PP packet timing accuracy | §8.5 |
| 5 | Transmission burst | §9 |
| 6 | Transmitted power | §10 |
| 7 | RF carrier modulation | §11 |
| 8 | Modulation emissions | §12.2 |
| 9 | Transmitter transients | §12.3 |
| 10 | Transmitter intermodulation | §12.4 |
| 11 | TX spurious | §12.5 |
| 12 | Sensitivity | §13.1 |
| 13 | Reference BER / FER | §13.2 |
| 14 | Interference performance | §13.3 |
| 15 | Blocking Case 1 | §13.4 |
| 16 | Blocking Case 2 | §13.5 |
| 17 | Receiver intermodulation | §13.6 |
| 18 | RX / idle spurious | §13.7 |
| 19 | Higher Level Modulation tests | §21 |

---

# 43. Measurement uncertainty

EN 300 176-1 Annex Gにはrecommended maximum uncertaintyが示されています。

自作テスターで特に重要なもの:

| Quantity | Recommended max uncertainty |
|---|---:|
| Relative RF frequency drift | ±1 kHz |
| Absolute RF frequency | ±10 kHz |
| Conducted emission power | ±1 dB |
| Radiated emission power | ±3 dB |
| Relative RF power | ±1 dB |
| Relative packet timing | ±0.1 µs |
| Absolute packet timing | ±1 µs |
| Timing stability | 1 ppm |
| Peak frequency deviation | ±10 kHz |

これはかなり重要です。

たとえば1.9 GHzにおいて:

```text
1 ppm ≈ 1.9 kHz
```

なので、PC clockやSDR内蔵XOを無条件に信用してcarrier accuracyを測るのは危険です。

---

# 44. 自作VSAの推奨構成

```text
IQ Input
  |
  +-- RF calibration
  |
  +-- channel filter / resampler
  |
  +-- burst detector
  |
  +-- coarse CFO
  |
  +-- GFSK discriminator
  |
  +-- S-field correlator
  |
  +-- fractional timing estimator
  |
  +-- exact p0 estimator
  |
  +-- packet type / direction detector
  |
  +-- bit decoder
  |
  +------------------------------------+
  |                                    |
  v                                    v
Timing Analysis                 Modulation Analysis
  |                                    |
  |                                    +-- freq deviation
  |                                    +-- drift
  |                                    +-- eye / trajectory
  |                                    +-- per-bit deviation
  |
  +-- slot timing
  +-- jitter
  +-- symbol timing

  +------------------------------------+
  |
  v
Power Analysis
  |
  +-- Power-Time Template
  +-- attack / release
  +-- burst flatness
  +-- idle power
```

---

# 45. VSA画面として有用な表示

BT専用VSAの経験をそのまま生かせます。

## Summary

```text
Direction       : RFP / PP
Carrier         : xxx MHz
Carrier Error   : xx.x kHz
Symbol Rate     : x.x ppm
Packet Type     : P32
p0              : ...
Peak Dev (+)    : ...
Peak Dev (-)    : ...
Freq Drift      : ...
Burst Power     : ...
Attack Time     : ...
Release Time    : ...
```

---

## Frequency Deviation vs Time

GFSK discriminator waveform上に:

- ideal bit centers
- exact p0
- bit boundaries
- excluded transition regions
- valid deviation measurement windows
- +259 / +403 kHz等のlimit
- -259 / -403 kHz等のlimit

をoverlayすると、規格判定と波形の因果が非常に見やすくなります。

---

## Sync Correlation

```text
correlation vs time
```

と、

```text
RFP score
PP score
fractional timing
```

を表示。

これはdebug用途としてかなり有効です。

---

## GFSK Eye

frequency discriminatorを1 symbolまたは2 symbolでfoldして、

```text
frequency deviation eye
```

として表示できます。

規格必須ではありませんが、

- BT
- DECT

を同じ思想で比較できる便利な解析表示になります。

---

# 46. 推奨内部データモデル

各packetについて:

```text
PacketResult
  timestamp
  direction
  carrier_index
  nominal_frequency
  measured_frequency
  carrier_error
  p0_time
  symbol_period
  symbol_rate_error_ppm
  packet_type
  raw_bits
  sync_score
  power
  attack_time
  release_time
  freq_drift
  modulation:
      positive_peak_deviation
      negative_peak_deviation
      per_bit_results[]
      case_a_pass
      case_b_pass
```

各bit:

```text
BitMeasurement
  index
  expected_bit
  decoded_bit
  start_time
  center_time
  valid_measurement_start
  valid_measurement_end
  mean_frequency
  peak_frequency
  deviation
```

としておくと、規格testとVSA可視化を共通engineで扱えます。

---

# 47. 実装Phase案

## Phase 1: Passive DECT VSA

まずここがおすすめです。

実装:

- burst detect
- RFP / PP sync
- exact p0
- symbol timing
- carrier frequency
- GFSK bit decode
- frequency deviation
- frequency drift
- Power-Time
- packet type identification
- GFSK eye / analysis plots

これだけでもかなり実用的なDECT専用VSAになります。

---

## Phase 2: Test-pattern aware VSA

DUT側を既存のDECT test modeで制御できるなら:

- `00001111`
- `0101`
- EN 300 176-1 Figure 27～31 patterns

を認識し、

```text
Carrier Accuracy Test
Modulation Part 1
Modulation Part 2
Modulation Part 3
Modulation Part 4
```

を自動判定できるようにします。

この段階で送信系PHY testerとしてかなり完成度が高くなります。

---

## Phase 3: VSG / Lower Tester

追加:

- DECT GFSK packet generator
- exact slot timing
- frequency offset injection
- power control
- PRBS generator
- MAC test-mode control
- loopback
- BER / FER

これにより:

- Sensitivity
- Reference BER
- Interference

まで可能になります。

---

## Phase 4: Multi-signal RF test

さらに:

- 2nd DECT-like interferer
- CW interferer
- calibrated combiner
- wide-range signal source

を用意すれば:

- Blocking
- Different-time Blocking
- Receiver Intermodulation

へ進めます。

---

## Phase 5: HLM

必要になったら:

- π/2-DBPSK
- π/4-DQPSK
- π/8-D8PSK
- 16-QAM
- 64-QAM
- RRC α=0.5
- EVM

を追加します。

Classic cordless phone用途だけなら優先度は低めです。

---

# 48. 最初のMVPとして実装すべき項目

BT専用VSAと同じ思想なら、まず以下で十分です。

```text
[Acquisition]
- IQ capture
- center frequency / Fs
- calibration

[Sync]
- burst detect
- S-field correlation
- RFP / PP detection
- fractional timing
- exact p0

[Demod]
- GFSK discriminator
- bit decode
- packet extraction

[Measurement]
- carrier frequency
- carrier error
- symbol-rate error
- timing error
- positive/negative peak deviation
- frequency drift
- burst power
- attack/release

[Display]
- IQ Power
- Frequency Deviation
- Sync correlation
- GFSK Eye
- Summary
```

このMVPならPassive VSAとして非常に扱いやすく、その後のEN 300 176-1自動試験へそのまま発展できます。

---

# 49. 実装時に特に間違えやすい点

## 49.1 Burst edgeをp0にしない

NG:

```text
power > threshold -> p0
```

Correct concept:

```text
known sync waveform
 -> precise timing
 -> prescribed nominal-frequency crossing
 -> p0
```

---

## 49.2 CFO推定値をすべての試験で使い回さない

同期用coarse CFOと、EN 300 176-1 carrier-frequency test resultは別物として扱います。

```text
sync_cfo
measurement_carrier_frequency
```

を別parameterにすることを推奨します。

---

## 49.3 GFSK deviationをpacket全体のmax/minで測らない

transition regionを除外する規定があります。

必ず規格指定bit windowで測定します。

---

## 49.4 288 kHzだけをlimitと思わない

```text
288 kHz
```

はnominalです。

pass/fail limitはCase A / Bで異なります。

---

## 49.5 Sampling algorithmを「規格指定」と表現しない

ETSIが規定するのは主に:

```text
何を
どのbitで
どの時間区間で
何回測るか
```

です。

具体的DSP estimatorは実装自由です。

---

## 49.6 3 MHz analysis bandwidthを忘れない

GFSK modulation測定では:

```text
>= 3 MHz
```

を満たすmeasurement pathを用意します。

---

## 49.7 SDRの周波数基準精度

frequency accuracy / drift試験ではXO精度がそのまま測定誤差になります。

実機tester化するなら:

- TCXO
- OCXO
- external 10 MHz reference
- GPSDO

などを検討すべきです。

---

# 50. 日本向け製品を対象にする場合

日本では:

**ARIB STD-T101**
Radio Equipment Used for TDMA Digital Enhanced Cordless Telecommunications

が関連します。

2026-09-04時点でARIB Webサイト上ではVersion 2.2が公開されています。

Official:
https://www.arib.or.jp/english/std_tr/telecommunications/std-t101.html

ARIB STD-T101はETSI DECTの技術を一部参照していますが、

- 周波数
- channel plan
- 電波法上の送信条件
- 不要発射

等は日本側の要求確認が必要です。

したがってコード設計は:

```text
DECT PHY engine
        +
Regional RF profile
```

に分離するのがおすすめです。

例:

```text
profiles/
  etsi_eu.yaml
  arib_jp.yaml
```

---

# 51. 結論

自作DECTテスターを作るうえで最も重要な整理は以下です。

### 1. DECT PHYは十分詳細に規定されている

- 1.152 Msymbol/s
- TDMA slot
- 32-bit synchronization field
- GFSK BT=0.5
- nominal ±288 kHz
- packet timing
- carrier accuracy
- Power-Time
- receiver性能

まで明確です。

### 2. RF Test Specificationも存在する

EN 300 176-1により、

- どのpatternを使うか
- どのbit intervalを測るか
- carrier frequencyをどう評価するか
- repeat数
- bandwidth
- pass/fail limit

まで具体的に規定されます。

### 3. ただしsymbol synchronization DSPは規定されない

規格はSynchronization fieldを用いた同期獲得を要求しますが、

```text
Gardnerを使え
correlator tapはこれ
PLL bandwidthはこれ
```

とは定めていません。

### 4. p0の測定定義は厳密

receiverのpacket detection pointではなく、

```text
packet sync word直前のnominal-frequency crossing
```

を基準に計算するため、自作VSAもこの定義へ合わせる必要があります。

### 5. GFSK変調解析はmeasurement windowが重要

transition直後 / 直前を除外し、規定bit区間だけでfrequency deviationを評価します。

### 6. Passive VSAから始めるのが合理的

最初は:

```text
Sync
Carrier Frequency
GFSK Deviation
Frequency Drift
Timing
Power-Time
```

を実装。

その後、

```text
VSG
MAC test control
Loopback
BER
Interference / Blocking
```

へ拡張すると、コード資産をそのまま活用できます。

---

# 52. References

1. ETSI EN 300 175-2 V2.9.1 (2022-03)
   Digital Enhanced Cordless Telecommunications (DECT); Common Interface (CI); Part 2: Physical Layer (PHL)
   https://www.etsi.org/deliver/etsi_en/300100_300199/30017502/02.09.01_60/en_30017502v020901p.pdf

2. ETSI EN 300 176-1 V2.4.1 (2022-11)
   Digital Enhanced Cordless Telecommunications (DECT); Test specification; Part 1: Radio
   https://www.etsi.org/deliver/etsi_en/300100_300199/30017601/02.04.01_60/en_30017601v020401p.pdf

3. ETSI EN 300 175-3
   Digital Enhanced Cordless Telecommunications (DECT); Common Interface (CI); Part 3: Medium Access Control (MAC) layer

4. ARIB STD-T101
   Radio Equipment Used for TDMA Digital Enhanced Cordless Telecommunications
   https://www.arib.or.jp/english/std_tr/telecommunications/std-t101.html

5. ITU-T O.153
   Basic parameters for the measurement of error performance at bit rates below the primary rate

---

## 実装上の表記ルール

本書中の:

- **規格値**
- **規格procedure**

はETSI / ARIBの規定を要約したものです。

一方、

- 「推奨」
- 「実装案」
- 「非規定」

と明記したDSP処理は、自作VSA向けのengineering proposalであり、規格指定アルゴリズムではありません。

規格適合性を正式に判定する場合は、必ず原文の最新版・適用地域の法規・測定不確かさを含めて確認してください。
