# Wi-Fi RF パケット／変調波形設計メモ

対象: **IEEE 802.11 Non-HT OFDM (802.11a / 802.11g) / DSSS・CCK (802.11b) / HT-Mixed (802.11n)**  
作成日: 2026-08-29  
目的: Pluto VSG で Wi-Fi として成立する複素 IQ 波形を生成し、実機 Wi-Fi 受信機・Wireshark・将来の Pluto VSA / Validator で検証できるようにするための実装メモ。  
主な参照仕様: **IEEE Std 802.11-2024**。実装時の clause 番号は、legacy PHY の参照性が高い **IEEE Std 802.11-2020** の Clause 16 / 17 / 18 / 19 も併記する。

> このメモは、Pluto VSG に Wi-Fi 波形生成機能を追加するための設計資料です。認証試験器や適合性試験仕様の代替ではありません。特に spectral mask、EVM、送信電力、地域ごとの使用可能 channel などは、最終的には対象地域・対象版の IEEE 802.11 規格および電波法規で再確認します。仕様書から直接固定値を転記する必要がある training sequence 等は、実装時に一次資料と独立した reference vector で照合します。

---

## 0. 結論と推奨実装順

Pluto VSG の Wi-Fi 対応は、最初から全 Wi-Fi PHY を実装するのではなく、以下の順序を推奨します。

1. **Non-HT OFDM 20 MHz (802.11a/g) を最初に完成させる**
   - 6 / 9 / 12 / 18 / 24 / 36 / 48 / 54 Mbps
   - L-STF / L-LTF / L-SIG / DATA
   - Scrambler / BCC / puncturing / interleaver / BPSK-QPSK-16QAM-64QAM / pilot / IFFT / GI
   - Raw PSDU と MAC frame builder の両方を扱える構成
2. **Beacon preset を追加する**
   - Android / PC の Wi-Fi scan で SSID が見えることを interoperability check に使う
3. **最低限の Wi-Fi Packet Validator を別実装する**
   - L-STF/L-LTF 検出、L-SIG decode、DATA decode、FCS check
   - VSG の generator と同じ内部関数をそのまま逆向きに使わず、generator の誤りを検出できる独立性を保つ
4. **802.11b DSSS / CCK を追加する**
   - 1 / 2 / 5.5 / 11 Mbps
5. **802.11n HT20 / HT40、1 spatial stream を追加する**
   - MCS 0–7 を第一段階とする

Pluto の帯域・sample-rate 制約を考えると、**20 MHz Wi-Fi と 40 MHz Wi-Fi は実装対象として現実的**です。一方、80 MHz 以上の VHT/HE 波形を Pluto 単体で標準帯域のまま生成する設計は、このプロジェクトの第一対象から外します。

---

## 1. 参照箇所マップ

### 1.1 IEEE 802.11-2020

Legacy PHY の実装では以下を主に参照します。

- **Clause 9**
  - MAC frame format
  - Management / Control / Data frame
  - Beacon frame
  - FCS
- **Clause 16: DSSS PHY**
  - 1 Mbps DBPSK
  - 2 Mbps DQPSK
  - 5.5 / 11 Mbps CCK
  - PLCP long / short preamble
- **Clause 17: OFDM PHY**
  - 17.3.2 PPDU format
  - 17.3.3 PHY preamble
  - 17.3.4 SIGNAL field
  - 17.3.5 DATA field
  - scrambler / convolutional encoder / interleaver / constellation mapping / pilots / OFDM modulation
- **Clause 18: ERP**
  - 2.4 GHz 802.11g compatibility and ERP behavior
- **Clause 19: HT PHY**
  - HT-Mixed PPDU
  - HT-SIG / HT-STF / HT-LTF
  - MCS / HT20 / HT40

IEEE Std 802.11-2024 が現行 revision です。802.11-2020 の clause 番号を本メモで使うのは、legacy PHY の実装資料や計測器資料との照合を容易にするためです。2024 revision で実装適合性を確認する際は clause の対応を再確認します。

### 1.2 外部の実装・計測リファレンス

- MathWorks WLAN Toolbox: Non-HT PPDU Structure
  - L-STF / L-LTF / L-SIG / DATA の構造確認
- MathWorks WLAN Toolbox: HT PPDU Structure
  - HT-Mixed field structure の確認
- Keysight WLAN: OFDM Signal Structure
  - legacy OFDM の rate / modulation / coding parameter の確認
- Keysight WLAN: DSSS Frame Structure
  - DSSS/CCK frame structure の確認
- `bastibl/gr-ieee802-11`
  - 802.11a/g/p の OTA interoperability 実績がある独立実装として参考にする
- `cloud9477/gr-ieee80211`
  - IEEE 802.11-2020 に沿った Python packet generator を含むため、golden vector 比較の参考にできる

外部 OSS は **reference implementation / test oracle として利用し、コードをそのまま Pluto VSG へ取り込まない**方針とします。ライセンスと独立実装性を保ちます。

---

## 2. Wi-Fi 波形で Bluetooth と設計思想が異なる点

Bluetooth BR/EDR/LE は基本的に、時間方向へ並ぶ bit / symbol と IQ sample の境界を比較的直接対応させられます。

Wi-Fi OFDM は異なります。

```text
logical bits
  ↓ Scrambler
  ↓ FEC / Puncturing
  ↓ Interleaver
  ↓ Constellation mapping
  ↓ 48 data subcarriersへ並列配置
  + 4 pilot subcarriers
  ↓ IFFT
  ↓ Guard Interval
complex IQ samples
```

このため、例えば MAC Header の 1 field が時間軸上の 1 つの連続した IQ 区間になるとは限りません。FEC と interleaving によって、その field の bit は複数 subcarrier / OFDM symbol へ分散します。

### 2.1 Pluto VSG の field hierarchy に必要な拡張思想

現行 `FieldDefinition` / `FieldBoundary` は、logical bit count、transmitted symbol count、IQ sample range を持つ再帰構造であり、Bluetooth の serial modulation には非常に相性が良いです。

Wi-Fi では、以下の **2層を分離**して扱うことを推奨します。

#### A. Protocol / Logical tree

```text
PSDU
  MAC Header
    Frame Control
    Duration
    Address 1
    Address 2
    Address 3
    Sequence Control
  Frame Body
  FCS
```

さらに PHY DATA input として:

```text
DATA logical bits
  SERVICE
  PSDU
  TAIL
  PAD
```

この tree は **bit / byte span を表す**ものであり、必ずしも連続 IQ sample range を持ちません。

#### B. Waveform / Time-region tree

```text
PPDU
  L-STF       8 us
  L-LTF       8 us
  L-SIG       4 us
  DATA
    OFDM Symbol 0   4 us
    OFDM Symbol 1   4 us
    ...
```

こちらは IQ waveform 上で連続する **sample range** を表します。

将来詳細表示が必要なら、logical bit から以下への mapping metadata を保持します。

```text
logical bit index
 → scrambled bit index
 → coded bit index
 → interleaved bit index
 → OFDM symbol index
 → subcarrier index
 → constellation bit position
```

これを `OFDMResourceMapping` のような debug metadata として持たせると、VSG の visual composer と将来の VSA/Validator の両方に使えます。

### 2.2 現行 model へ無理に serial symbol の意味を押し込まない

Wi-Fi では `symbol` という語が複数の意味を持ちます。

- coded bit
- QAM constellation symbol
- OFDM symbol
- DSSS chip

Wi-Fi profile では UI と metadata で単位を明示します。

Non-HT OFDM の top-level timing では、`1 OFDM symbol = 4 us` を時間単位として扱えます。L-STF と L-LTF はそれぞれ 8 us なので、top-level 表示上は 2 OFDM-symbol duration と表現できますが、内部 training structure は通常の DATA OFDM symbol と同一ではありません。

---

# Part A. Non-HT OFDM 20 MHz (802.11a / 802.11g)

## A.1 第一実装対象

Pluto VSG の Wi-Fi MVP は **Non-HT OFDM 20 MHz** とします。

理由:

- PHY 構造が明確
- 20 MHz native complex sample rate で生成可能
- 802.11a と 802.11g OFDM で同じ基本 PHY chain を再利用可能
- 2.4 GHz の 802.11g Beacon を生成すれば、市販 Wi-Fi device で認識確認しやすい
- HT/VHT/HE でも legacy preamble の L-STF / L-LTF / L-SIG は再利用される

802.11g で Legacy OFDM を使う場合、VSG UI では `MCS` ではなく **Data Rate [Mbps]** と表現します。MCS index は HT 以降で導入します。

## A.2 PPDU structure

```text
Non-HT OFDM PPDU

L-STF       8 us
L-LTF       8 us
L-SIG       4 us
DATA        N_SYM × 4 us
```

DATA logical structure:

```text
SERVICE     16 bits
PSDU        LENGTH × 8 bits
TAIL        6 bits
PAD         N_PAD bits
```

20 MHz の packet duration:

```text
T_PPDU_us = 20 + 4 * N_SYM
```

20 MS/s native sampling の sample count:

```text
L-STF = 160 samples
L-LTF = 160 samples
L-SIG =  80 samples
DATA  =  80 * N_SYM samples

N_PPDU_samples = 400 + 80 * N_SYM
```

40 MS/s へ 2× oversample する場合はすべて 2 倍です。

## A.3 OFDM basic parameters

20 MHz Non-HT OFDM:

| Parameter | Value |
|---|---:|
| Nominal channel bandwidth | 20 MHz |
| FFT size | 64 |
| Subcarrier spacing | 312.5 kHz |
| Useful symbol time | 3.2 us |
| Guard interval | 0.8 us |
| OFDM symbol time | 4.0 us |
| Used subcarriers | 52 |
| Data subcarriers | 48 |
| Pilot subcarriers | 4 |
| DC | 0 |

Pilot subcarrier index:

```text
-21, -7, +7, +21
```

DATA subcarrier は `-26 ... -1, +1 ... +26` のうち pilot 4 本を除いた 48 本です。

`numpy.fft.ifft()` の natural bin indexing を使う場合:

```text
k >= 0 : fft_bin = k
k <  0 : fft_bin = N_FFT + k
```

DC bin 0 と unused bins は 0 にします。

## A.4 Legacy rate table

| Rate [Mbps] | RATE bits | Modulation | Coding rate | N_BPSC | N_CBPS | N_DBPS |
|---:|---|---|---:|---:|---:|---:|
| 6  | 1101 | BPSK   | 1/2 | 1 | 48  | 24  |
| 9  | 1111 | BPSK   | 3/4 | 1 | 48  | 36  |
| 12 | 0101 | QPSK   | 1/2 | 2 | 96  | 48  |
| 18 | 0111 | QPSK   | 3/4 | 2 | 96  | 72  |
| 24 | 1001 | 16QAM  | 1/2 | 4 | 192 | 96  |
| 36 | 1011 | 16QAM  | 3/4 | 4 | 192 | 144 |
| 48 | 0001 | 64QAM  | 2/3 | 6 | 288 | 192 |
| 54 | 0011 | 64QAM  | 3/4 | 6 | 288 | 216 |

`RATE bits` は L-SIG の transmission-order bit representation として扱い、実装時に Annex/reference vector で byte/bit ordering を必ず照合します。

## A.5 L-STF

L-STF は packet detection、AGC、coarse timing、coarse CFO 推定に使われます。

20 MHz では:

```text
10 short repetitions × 0.8 us = 8 us
```

実装方針:

1. IEEE 802.11 Clause 17.3.3 で定義された frequency-domain short-training sequence を constant table として保持
2. 指定された subcarrier のみへ配置
3. standard normalization を適用
4. IFFT で time-domain short sequence を生成
5. 0.8 us periodicity を満たす 8 us field を構成

training sequence の全数値列を別の推測実装から再生成せず、**IEEE reference sequence と独立した golden IQ vector の双方で確認**します。

## A.6 L-LTF

L-LTF は fine timing、fine CFO、channel estimation に使われます。

20 MHz では:

```text
GI2       1.6 us
Long #1   3.2 us
Long #2   3.2 us
----------------
Total     8.0 us
```

実装方針:

1. Clause 17.3.3 の long-training frequency sequence を保持
2. 64-point IFFT
3. long symbol の後半 1.6 us を GI2 として前置
4. long symbol を 2 回連結

```text
L-LTF = GI2 + LTF + LTF
```

## A.7 L-SIG

L-SIG は 24 logical bits です。

```text
RATE       4 bits
Reserved   1 bit = 0
LENGTH    12 bits
Parity     1 bit
Tail       6 bits = 0
-------------------
Total     24 bits
```

- LENGTH は PSDU length [octets]
- LENGTH は LSB-first の bit field として構成
- Parity は先行 field に対する even parity
- L-SIG 自体は **scramble しない**
- coding = BCC rate 1/2
- modulation = BPSK
- 1 OFDM symbol = 4 us
- pilot polarity は L-SIG を index 0 として開始する

L-SIG encoder chain:

```text
24 bits
 ↓ BCC 1/2
48 coded bits
 ↓ interleave (N_CBPS=48)
48 BPSK symbols
 ↓ 48 data subcarriers + 4 pilots
64 FFT bins
 ↓ IFFT
64 samples
 + 16-sample GI
80 samples @ 20 MS/s
```

## A.8 DATA field size calculation

`LENGTH` を PSDU octet count とします。

```text
N_SYM  = ceil((16 + 8*LENGTH + 6) / N_DBPS)
N_DATA = N_SYM * N_DBPS
N_PAD  = N_DATA - (16 + 8*LENGTH + 6)
```

DATA logical bit stream:

```text
SERVICE(16)
+ PSDU(8*LENGTH)
+ TAIL(6)
+ PAD(N_PAD)
```

SERVICE は scramble 前に 0 で初期化します。PSDU の各 octet は **bit 0 を先に air-order bit stream へ入れます**。

## A.9 Scrambler

Legacy OFDM DATA scrambler:

```text
length = 127
polynomial = x^7 + x^4 + 1
initial state = non-zero 7-bit value
```

VSG では repeatability のため、以下を選択可能にすることを推奨します。

- Auto / pseudo-random non-zero seed
- Fixed seed

Test / golden vector では fixed seed を使用します。

重要な処理順:

1. SERVICE + PSDU + six zero TAIL + PAD zeros を作る
2. DATA bit string を scramble
3. **scramble 後の TAIL 位置 6 bits を nonscrambled zero に置換**
4. BCC encoder へ入力

PAD bits は scramble 対象です。

## A.10 BCC convolutional encoder

Mother code:

```text
constraint length K = 7
rate = 1/2
g0 = 133 octal
g1 = 171 octal
```

同一 input bit に対して g0 output を先、g1 output を後に出します。

Higher coding rate は puncturing で作ります。

実装時には puncturing mask を `coding_rate -> mask` の immutable table とし、mask を rate 名から推測生成しない方が安全です。

概念:

```text
1/2 : all mother-code outputsを使用
2/3 : 一部を周期的に puncture
3/4 : より多くを周期的に puncture
```

unit test は IEEE Annex の既知 bit stream または独立 reference implementation と完全一致させます。

## A.11 Interleaver

1 OFDM symbol ごとに `N_CBPS` coded bits を interleave します。

第一 permutation:

```text
i = (N_CBPS / 16) * (k mod 16) + floor(k / 16)
```

第二 permutation:

```text
s = max(N_BPSC / 2, 1)

j = s * floor(i / s)
    + (i + N_CBPS - floor(16*i / N_CBPS)) mod s
```

目的:

- 隣接 coded bit を離れた subcarrier へ分散
- QAM symbol 内で bit significance が偏らないようにする

L-SIG も BPSK / `N_CBPS=48` の同じ interleaver rule を使います。

## A.12 Constellation mapping

Gray mapping と平均電力 normalization を使用します。

Normalization:

| Modulation | Scale |
|---|---:|
| BPSK | 1 |
| QPSK | 1/sqrt(2) |
| 16QAM | 1/sqrt(10) |
| 64QAM | 1/sqrt(42) |

推奨 unit test:

- すべての constellation bit combination を列挙
- expected I/Q coordinate と完全一致
- bit tuple の先頭 bit を勝手に MSB 扱いしない

Wi-Fi では「byte の bit order」「interleaver output order」「QAM label order」が別概念なので、各 API の引数名に `air_bits`, `coded_bits`, `interleaved_bits` のような段階名を付けます。

## A.13 Pilot insertion

Pilot carrier:

```text
k = -21, -7, +7, +21
```

基本 pilot sign pattern に 127-symbol 周期の polarity sequence `p[n]` を掛けます。

- `p[0]`: L-SIG
- `p[1]`: DATA OFDM symbol 0
- `p[2]`: DATA OFDM symbol 1
- ...

127-element sequence は Clause 17.3.5.9 の定義を immutable constant として実装し、**長い数値列を手入力するだけで終わらせず unit test で sequence generator と照合**します。

## A.14 IFFT and Guard Interval

DATA / L-SIG の 20 MHz native generation:

```text
48 data constellation values
+ 4 BPSK pilots
+ DC / unused = 0
      ↓
64-bin frequency vector
      ↓ 64-point IFFT
64 complex samples = 3.2 us
      ↓ prepend last 16 samples
80 complex samples = 4.0 us
```

`np.fft.ifft()` の scale convention は IEEE 数式の scale と異なり得るため、最終 IQ を単純 peak normalize する前に、**L-STF / L-LTF / L-SIG / DATA 間の相対 RMS level** が reference waveform と一致することを確認します。

## A.15 20 MS/s と 40 MS/s

### Native mode

```text
sample rate = 20 MS/s
FFT = 64
GI = 16 samples
```

標準 parameter と直接対応し、golden-vector test が最も単純です。

### 2× oversampled mode

```text
sample rate = 40 MS/s
FFT-equivalent = 128
GI = 32 samples
```

推奨方法は FFT-based oversampling です。

- subcarrier spacing 312.5 kHz を維持
- occupied subcarrier を 128-bin grid の同じ周波数位置へ配置
- unused high-frequency bins を zero padding
- IFFT
- amplitude normalization を補正

別案として 20 MS/s canonical waveform を生成後、band-limited resampler で 40 MS/s に変換しても構いません。

**設計推奨:**

- 内部 correctness reference: 20 MS/s
- Pluto RF output default: 40 MS/s selectable

とすると、standard-native test と DAC/analog reconstruction の余裕を両立しやすくなります。

## A.16 Symbol windowing / packet edge ramp

Wi-Fi OFDM では Bluetooth のように DATA 全体へ Gaussian / RRC pulse shaping を掛けません。

基本は:

```text
subcarrier mapping -> IFFT -> GI -> symbol concatenation
```

spectral splatter を抑えるための OFDM symbol transition windowing は、Clause 17 の discrete-time implementation に合わせて別処理として設計します。

重要:

- 各 OFDM symbol へ独立した arbitrary cosine ramp を掛けない
- GI と cyclic extension の関係を壊さない
- Pluto VSG 共通の `PowerEnvelopeDefinition` と Wi-Fi OFDM symbol windowing を混同しない

推奨:

```text
Wi-Fi PHY windowing
    = standard waveform constructionの一部

VSG hardware outer ramp
    = PPDUのさらに外側のRF ON/OFF補助
```

MVP ではまず decode 可能な native PPDU を作り、その後 spectral mask を見ながら standard-compatible windowing を追加します。

---

# Part B. MAC / PSDU 生成

## B.1 PHY generator と MAC builder を分離する

Wi-Fi PHY engine の入力は、基本的に完成した `PSDU: bytes` とします。

```text
MAC builder
   ↓ PSDU bytes
PHY encoder
   ↓ PPDU IQ
Pluto backend
```

これにより:

- Raw PSDU test
- Beacon
- Data frame
- Probe Response 等の将来追加

を同じ PHY encoder へ渡せます。

PHY encoder が SSID や MAC address を理解する必要はありません。

## B.2 Management frame の基本 MAC Header

典型的な Management frame:

```text
Frame Control       2 octets
Duration / ID       2 octets
Address 1           6 octets
Address 2           6 octets
Address 3           6 octets
Sequence Control    2 octets
Frame Body          variable
FCS                  4 octets
```

Beacon では通常:

```text
Address 1 = FF:FF:FF:FF:FF:FF   # broadcast DA
Address 2 = transmitter MAC / BSSID
Address 3 = BSSID
```

VSG 用の既定 BSSID は、実ネットワークと衝突しにくい **locally administered unicast address** を使います。

例:

```text
02:11:22:33:44:55
```

## B.3 Beacon preset

Beacon は Wi-Fi VSG の interoperability check に非常に有用です。

推奨 MVP preset:

```text
PHY              802.11g Non-HT OFDM
Channel          6
Center           2437 MHz
Data Rate        6 Mbps
SSID             Pluto_Test_AP
BSSID            02:11:22:33:44:55
Security         Open
Beacon Interval  100 TU
```

1 TU:

```text
1 TU = 1024 us
100 TU = 102.4 ms
```

Beacon Frame Body の最小構成候補:

```text
Timestamp              8 octets
Beacon Interval         2 octets
Capability Information 2 octets
SSID IE                 variable
Supported Rates IE      variable
DS Parameter Set IE     channel information (2.4 GHz)
TIM IE                  APらしいBeaconとして必要に応じ追加
```

Android に SSID が表示されることは「Wi-Fi packet として一定の互換性がある」ことを示す強い smoke test ですが、**AP として association/authentication が成立することは意味しません**。

## B.4 Beacon の動的 field

実 AP の Beacon では Timestamp と Sequence Number が進行します。

Pluto VSG で同一 IQ buffer を周期 repeat すると、これらも同じ値で繰り返されます。

第一段階:

- static Timestamp
- static Sequence Number
- valid FCS

でも scan recognition の試験は可能と考えられますが、これは実機互換性で確認します。

第二段階:

```text
packet scheduler
  ↓ packetごとに
Timestamp update
Sequence Number increment
FCS regenerate
PHY regenerate
```

を実装すると、より AP に近い Beacon source になります。

## B.5 FCS

MAC FCS は 32-bit CRC です。

実装では byte reflection / transmission bit order の取り違えが起きやすいため、独自式を一度書いて終わりにせず:

1. known Beacon / Data frame の Wireshark capture
2. Python 側 FCS calculation
3. Wireshark の `FCS good`

で検証します。

Raw PSDU mode では:

- `FCS Auto`
- `FCS Supplied`
- `FCS Disabled`（意図的 invalid packet test）

を分けると便利です。

---

# Part C. 802.11b DSSS / CCK

## C.1 対象 rate

| Data Rate | Chip Rate | Spreading / Coding | Modulation |
|---:|---:|---|---|
| 1 Mbps | 11 Mcps | 11-chip Barker | DBPSK |
| 2 Mbps | 11 Mcps | 11-chip Barker | DQPSK |
| 5.5 Mbps | 11 Mcps | 8-chip CCK | differential phase + CCK |
| 11 Mbps | 11 Mcps | 8-chip CCK | differential phase + CCK |

802.11b は OFDM engine と別 engine にします。

```text
WifiLegacyOfdmEngine
WifiDsssCckEngine
```

共通化するのは MAC/PSDU builder、project persistence、packet scheduler、Pluto backend までとし、PHY transform を無理に共通化しません。

## C.2 Long PLCP preamble

概略:

```text
SYNC        128 bits
SFD          16 bits
-------------------
Preamble    144 bits @ 1 Mbps

SIGNAL        8 bits
SERVICE       8 bits
LENGTH       16 bits
CRC          16 bits
-------------------
Header       48 bits @ 1 Mbps

Total PLCP preamble+header = 192 us
```

DATA は 1 / 2 / 5.5 / 11 Mbps の選択 rate で送ります。

## C.3 Short PLCP preamble

概略:

```text
Short SYNC   56 bits
Short SFD    16 bits
-------------------
Preamble     72 us @ 1 Mbps

PLCP Header  48 bits @ 2 Mbps
             24 us
-------------------
Total        96 us
```

Short preamble は 1 Mbps payload 用には使用しません。

## C.4 Sampling

chip rate:

```text
11 Mcps
```

Pluto VSG では整数 samples/chip が扱いやすいです。

```text
22 MS/s = 2 samples/chip
44 MS/s = 4 samples/chip
```

推奨 default:

```text
44 MS/s
```

理由:

- chip waveform の時間分解能が高い
- filter / transition の設計余裕
- Pluto の実用 sample-rate 範囲内

## C.5 CCK 実装方針

CCK は単純 QPSK mapper ではありません。

5.5 / 11 Mbps では、input bits から differential phase と CCK codeword phase を生成して 8-chip complex codeword を作ります。

この部分は bit mapping の取り違えが起きやすいため、実装前に IEEE 802.11-2020 Clause 16.3.6.6 の 5.5 / 11 Mbps mapping table と known vector を別途抽出します。

**このメモでは CCK codeword table を推測して固定しません。**

802.11b 実装 phase では以下を追加調査します。

- Long / Short SFD exact transmission sequence
- PLCP scrambler initialization
- SIGNAL field values
- LENGTH field rounding / length extension rule
- 5.5 Mbps CCK mapping
- 11 Mbps CCK mapping
- differential phase continuity
- transmit filter / spectral shaping

---

# Part D. 802.11n HT20 / HT40

## D.1 実装対象

legacy OFDM 完成後の次候補として:

```text
HT-Mixed
1 spatial stream
BCC
MCS 0–7
HT20 / HT40
Long GI first
```

を推奨します。

## D.2 HT-Mixed PPDU structure

```text
L-STF       8 us
L-LTF       8 us
L-SIG       4 us
HT-SIG      8 us  (2 OFDM symbols)
HT-STF      4 us
HT-LTF      4 us × required count
DATA        variable
```

1 spatial stream の第一実装では HT-LTF は 1 field から始めます。

HT-SIG には MCS、HT LENGTH、coding、GI、aggregation 等の情報が入ります。

## D.3 1SS MCS 0–7

20 MHz / Long GI の代表 rate:

| MCS | Modulation | Coding | 20 MHz rate [Mbps] |
|---:|---|---:|---:|
| 0 | BPSK  | 1/2 | 6.5 |
| 1 | QPSK  | 1/2 | 13.0 |
| 2 | QPSK  | 3/4 | 19.5 |
| 3 | 16QAM | 1/2 | 26.0 |
| 4 | 16QAM | 3/4 | 39.0 |
| 5 | 64QAM | 2/3 | 52.0 |
| 6 | 64QAM | 3/4 | 58.5 |
| 7 | 64QAM | 5/6 | 65.0 |

HT40 では FFT size が 128 になり、native complex sample rate は 40 MS/s です。

Pluto では HT40 native generation は現実的ですが、2× oversampling の 80 MS/s は対象外です。したがって HT40 はまず native 40 MS/s を基本とします。

## D.4 Legacy engine から再利用できるもの

- L-STF
- L-LTF
- L-SIG building block
- convolutional encoder
- puncturing primitive
- constellation mapper
- OFDM FFT-bin mapping primitive
- pilot utility の考え方

一方、HT では subcarrier allocation、pilot sequence、HT-SIG、HT training、data interleaving 等が変わるため、`Legacy mode の parameter を少し変えるだけ` という実装にはしません。

---

# Part E. Pluto VSG への組込み設計

## E.1 現行 VSG との対応

現行 Pluto VSG は:

```text
WaveformProject
  ↓ Profile
WaveformEngine
  ↓ GenerationResult
normalized complex IQ
  ↓ Backend / Export
Pluto TX / file
```

という device-independent な構造です。

Wi-Fi もこの境界を維持します。

推奨追加:

```text
pluto_vsg/
  wifi/
    mac.py
    crc.py
    common.py
    legacy_ofdm.py
    dsss_cck.py        # phase 2
    ht.py              # phase 3

  engine/
    wifi_legacy_ofdm.py
    wifi_dsss_cck.py
    wifi_ht.py

  profiles/
    wifi.py
```

実際の package 分割は既存命名規則に合わせて調整します。

## E.2 Model proposal

`StandardProfile` に Wi-Fi を追加します。

例:

```text
StandardProfile.WIFI
```

Wi-Fi 専用 settings は serial-modulation settings と分離します。

```text
WiFiSettings
  phy_format
    NON_HT_OFDM
    DSSS_CCK
    HT_MIXED

  channel_bandwidth_mhz
  legacy_rate_mbps
  ht_mcs
  scrambler_seed_mode
  scrambler_seed

  psdu_source
    RAW_HEX
    PATTERN
    PRBS
    MAC_FRAME
    BEACON

  mac settings
  beacon settings

  packet_period_us
  oversample_factor
  symbol_window_enabled
```

`N_SYM`, `N_PAD`, L-SIG parity、FCS、coded bits 等の **derived value は project file の source-of-truth とせず再計算**します。

## E.3 ModulationKind の扱い

現行 `ModulationKind` は GFSK / pi/4-DQPSK / 8DPSK のような serial modulation を表します。

Wi-Fi を追加する際は、単純に `OFDM` だけを 1 modulation kind として終わらせるより:

```text
waveform family = OFDM
subcarrier modulation = BPSK / QPSK / 16QAM / 64QAM
```

を metadata として分けた方が将来 VSA と整合します。

Visual Composer の Modulation track では例えば:

```text
L-STF      Training
L-LTF      Training
L-SIG      OFDM / BPSK / R=1/2
DATA       OFDM / 64QAM / R=3/4
```

のように表示します。

## E.4 GenerationResult metadata

Wi-Fi engine は debug / validator 用として以下を `metadata` に入れることを推奨します。

```text
phy_format
channel_bandwidth_hz
legacy_rate_mbps / mcs
sample_rate_hz
oversample_factor

psdu_length_bytes
fcs
scrambler_seed

n_bpsc
n_cbps
n_dbps
n_sym
n_pad

sample ranges:
  l_stf
  l_ltf
  l_sig
  data
  each_data_ofdm_symbol

optional debug:
  l_sig_bits
  scrambled_data_bits
  punctured_bits per symbol
  interleaved_bits per symbol
  data constellation per symbol
  frequency-domain bins per symbol
```

large debug array は通常 generation では保持せず、`diagnostic=True` のときだけ生成してもよいです。

## E.5 Visual Composer

時間軸上の major field:

```text
L-STF | L-LTF | L-SIG | DATA
```

DATA の下は waveform view では:

```text
OFDM #0 | OFDM #1 | ...
```

とするのが正確です。

一方、MAC Header / Payload / FCS を `DATA` の時間サブフィールドとして等幅に描画するのは誤解を招きます。これらは logical tree / Inspector で別表示します。

将来 `Resource Map` view を追加するなら:

```text
vertical   = subcarrier
horizontal = OFDM symbol
cell       = data / pilot / null + constellation value
```

が Wi-Fi には非常に有効です。

## E.6 Packet period / repeat

VSG の packet repeat は **start-to-start period** で定義します。

```text
period >= PPDU duration + required idle
```

UI は Wi-Fi では symbol count より:

- Packet Duration [us]
- Packet Period [us/ms]
- Duty Cycle [%]
- Repeat Count / Continuous

の表示が分かりやすいです。

Beacon preset では default period を 100 TU = 102.4 ms にできます。

## E.7 Center frequency preset

2.4 GHz 20 MHz channel の代表値:

| Channel | Center [MHz] |
|---:|---:|
| 1 | 2412 |
| 6 | 2437 |
| 11 | 2462 |
| 13 | 2472 |
| 14 | 2484 |

Channel 1–13 は基本的に:

```text
f_center_MHz = 2407 + 5 * channel
```

Channel 14 は 2484 MHz で別扱いです。日本の channel 14 は legacy DSSS/802.11b 用として扱い、OFDM preset の選択肢には入れません。

VSG では regulatory enforcement を暗黙に行うより、channel preset と direct frequency input を分け、実 RF 出力時はユーザーが使用地域・設備条件を確認できる設計がよいです。

---

# Part F. Generator implementation pipeline

## F.1 Non-HT OFDM

推奨 function boundary:

```text
build_psdu(settings) -> bytes

build_l_sig(rate, length) -> bits[24]
encode_l_sig(bits) -> freq_bins / time_iq

build_data_bits(psdu) -> SERVICE + PSDU + TAIL + PAD
scramble_data(...)
bcc_encode(...)
puncture(...)
interleave_symbol(...)
map_constellation(...)
insert_pilots(...)
ofdm_modulate(...)

build_l_stf(...)
build_l_ltf(...)

assemble_ppdu(...) -> IQ
apply_ppdu_window(...)
apply_outer_envelope_and_idle(...)
```

`generate()` 1 関数にすべて書かず、各段を pure function にすると unit test が容易です。

## F.2 型の分離

Python 上で全部 `np.ndarray[int]` にすると stage 取り違えが起きやすいので、少なくとも命名で厳格に区別します。

```text
psdu_bytes
uncoded_bits
scrambled_bits
encoded_bits
punctured_bits
interleaved_bits
qam_symbols
freq_bins
time_samples
```

必要なら small dataclass を使います。

## F.3 Randomness

以下は reproducibility のため seed を管理します。

- scrambler seed
- PRBS/random payload
- dynamic MAC sequence number

`Randomize each generation` と `Fixed for test` を分けます。

## F.4 Normalization / RF Level

Wi-Fi OFDM は PAPR が大きいため、Bluetooth と同じ peak headroom ではなく waveform-specific crest factor を意識します。

VSG engine output は device-independent normalized IQ とし:

```text
active RMS dBFS
peak dBFS
crest factor = peak - RMS
```

を metadata に出します。

Pluto backend の RF Level calibration は既存の RMS 基準設計を再利用しますが、Wi-Fi は peak clipping が EVM と spectral regrowth に直結するため、default digital backoff を十分に確保します。

**RF Level は active PPDU の RMS を基準**とし、packet idle zero を含む長時間 RMS にしません。

---

# Part G. Validation strategy

## G.1 Generator 自身と Validator を同じロジックにしない

例えば generator の interleaver と validator の deinterleaver を、同じ index table を単純 inverse して作るだけでは共通 bug を見逃す可能性があります。

可能な部分は:

- independent formula
- known standard vector
- third-party implementation

のいずれかで cross-check します。

## G.2 Unit test 層

### Layer 1: MAC

- Beacon byte sequence
- Frame Control
- Address order
- Sequence Control
- Information Element TLV
- FCS known vector

### Layer 2: PHY bit processing

- L-SIG 24 bits
- L-SIG parity
- DATA length / N_SYM / N_PAD
- scrambler known sequence
- BCC known vector
- puncturing
- interleaving index permutation
- constellation map
- pilot polarity index

### Layer 3: OFDM resource grid

各 OFDM symbol について:

- 48 data carrier が正しい index に入る
- pilot 4 本が正しい
- DC = 0
- guard/unused carrier = 0

### Layer 4: IQ waveform

- L-STF known waveform correlation
- L-LTF known waveform correlation
- L-SIG reference IQ
- complete PPDU reference IQ

absolute scale を除いて complex correlation がほぼ 1 になることを確認します。

## G.3 Digital round-trip Validator

最低限:

```text
generated IQ
  ↓ Packet detect
L-STF
  ↓ coarse CFO / timing
L-LTF
  ↓ fine timing / channel estimate
L-SIG decode
  ↓ RATE / LENGTH / parity
DATA FFT
  ↓ pilot/CPE correction
subcarrier equalization
  ↓ demap / deinterleave / Viterbi
Descramble
  ↓
PSDU
  ↓
FCS check
```

表示例:

```text
Preamble detected       OK
L-SIG parity            OK
Rate                     6 Mbps
Length                  128 bytes
PSDU decode              OK
FCS                      OK
```

この段階では EVM measurement を実装しなくても、VSG の packet correctness 検証として十分価値があります。

## G.4 RF round-trip

```text
Pluto VSG
  ↓ RF
Pluto / R&S capture
  ↓ IQ
Wi-Fi Validator
```

確認:

- preamble detect
- L-SIG decode
- payload一致
- FCS OK

## G.5 Third-party interoperability

### Monitor mode + Wireshark

対応 Wi-Fi adapter がある場合:

- generated frame が packet として表示される
- RATE / channel / MAC frame fields が一致
- FCS good

を確認します。

### Beacon + Android / PC

- SSID `Pluto_Test_AP` が scan list に表示される
- channel が期待どおり

を smoke test とします。

ただし scan 表示は full AP functionality の保証ではありません。

### R&S / commercial VSA

可能なら:

- EVM
- carrier frequency error
- symbol clock error
- spectral flatness
- power

を比較し、Pluto VSG 自体の analog impairment と generator algorithm の誤りを分離します。

## G.6 OSS reference

`gr-ieee802-11` 等で同じ PSDU / rate / scrambler seed の baseband を作り、以下の各 stage を比較できると強力です。

```text
L-SIG bits
encoded bits
interleaved bits
constellation
frequency-domain OFDM symbols
time-domain IQ
```

第三者コードは test oracle として使用し、Pluto VSG 本体は独立実装を維持します。

---

# Part H. UI proposal

## H.1 New project

```text
File > New > Wi-Fi
```

初期 preset:

```text
PHY Format       Non-HT OFDM
Band             2.4 GHz
Channel          6
Center           2437 MHz
Bandwidth        20 MHz
Data Rate        6 Mbps
Frame Source     Raw PSDU / Beacon
Sample Rate      40 MS/s
Packet Period    user selectable
```

## H.2 Wi-Fi Settings

第一段階:

```text
PHY
  Format              Non-HT OFDM
  Bandwidth            20 MHz
  Data Rate            6..54 Mbps
  Scrambler Seed       Auto / Fixed
  Sample Rate          20 / 40 MS/s

RF
  Channel / Center Frequency
  RF Level

Packet
  Source               Raw / Pattern / PRBS / MAC / Beacon
  PSDU Length
  Packet Period
  Repeat

MAC / Beacon
  SSID
  BSSID
  Sequence Number
  Beacon Interval
  Supported Rates
  Channel IE
  FCS Auto
```

Derived result を read-only 表示:

```text
N_BPSC
Coding Rate
N_DBPS
N_SYM
N_PAD
PPDU Duration
Active RMS
Peak
Crest Factor
```

## H.3 Debug view

開発時に有用:

- PPDU field tree
- MAC logical field tree
- OFDM resource grid
- Constellation per DATA symbol
- Spectrum
- IQ waveform

一般利用 UI では resource grid / coding debug を隠し、Advanced/Diagnostics で開く構成がよいです。

---

# Part I. 実装時の注意事項

## I.1 最も壊れやすい箇所

- PSDU byte 内 bit order
- L-SIG RATE bit order
- L-SIG LENGTH LSB-first
- parity range
- scrambler register direction
- TAIL の scramble 後 zero replacement
- convolutional encoder g0/g1 output order
- puncturing phase
- interleaver permutation direction
- QAM bit tuple order
- negative subcarrier の FFT bin index
- pilot polarity start index
- IFFT scale
- training field relative amplitude
- FCS byte/reflection order

このため、完成 IQ だけを見てデバッグせず、各 stage の intermediate data を diagnostic mode で取り出せるようにします。

## I.2 OFDM は「サブキャリアごとの独立 narrowband signal」を足す実装にしない

理論上は各 subcarrier の和ですが、実装では standard の FFT bin mapping と IFFT を source of truth とします。

個別 oscillator を 52 本生成して加算する方式は:

- phase origin
- sample boundary
- normalization
- orthogonality

の管理が難しく、reference vector と比較しにくくなります。

## I.3 L-STF / L-LTF を DATA mapper から作らない

training fields は固有 frequency sequence / time repetition を持つため、DATA OFDM symbol の generic mapper へ無理に通しません。

共通にするのは `frequency bins -> IFFT` primitive までとします。

## I.4 Packet repetition と Wi-Fi MAC timing は別

Pluto VSG の repeat は waveform generator の timing 機能です。

実際の Wi-Fi MAC の:

- CSMA/CA
- DIFS/SIFS
- ACK
- backoff
- NAV
- association state

を実装するものではありません。

したがって本機能は **Wi-Fi packet waveform generator / interference source** であり、完全な Wi-Fi MAC transceiver ではありません。

---

# Part J. 推奨 MVP acceptance criteria

Non-HT OFDM 第一版の完成条件:

1. 20 MHz / 20 MS/s で 6–54 Mbps 全 rate を生成できる
2. L-STF / L-LTF が independent reference と一致する
3. L-SIG の RATE / LENGTH / parity を独立 decoder が正しく読める
4. arbitrary PSDU が FCS を含めて digital round-trip で一致する
5. 40 MS/s 2× oversampled waveform でも同じ packet を decode できる
6. Pluto から RF 送信し、別 receiver で packet decode + FCS OK
7. 6 Mbps Beacon preset を Pluto から送信し、市販 Wi-Fi receiver / monitor mode で Wi-Fi frame として認識できる
8. Android / PC scan で SSID が確認できれば interoperability smoke test 合格
9. R&S 等で legacy OFDM として解析でき、重大な constellation / spectrum 異常がない

ここまで到達した時点で、Pluto VSG の Wi-Fi 機能は「単なる OFDM-like interference」ではなく、**他の Wi-Fi 実装が packet として認識可能な 802.11 waveform generator** と呼べる状態になります。

---

# Part K. 実装 phase proposal

## Phase 1: Non-HT OFDM core

- WiFiSettings
- Legacy rate table
- L-STF / L-LTF
- L-SIG
- scrambler
- BCC / puncturing
- interleaver
- QAM mapper
- pilot
- OFDM modulator
- raw PSDU
- 20 MS/s

## Phase 1.1: Pluto-friendly output

- 40 MS/s oversampling
- packet outer ramp / idle
- active RMS / crest factor metadata
- repeated packet period
- export / Pluto backend integration

## Phase 1.2: MAC / Beacon

- MAC Header builder
- FCS
- Beacon builder
- SSID / BSSID / channel preset
- 100 TU repeat preset
- Wireshark / Android interoperability test

## Phase 1.3: Validator

- L-STF/L-LTF detection
- L-SIG decode
- DATA decode
- FCS
- diagnostic compare

## Phase 2: DSSS / CCK

- Long / Short PLCP
- 1 / 2 Mbps Barker
- 5.5 / 11 Mbps CCK
- 44 MS/s default

## Phase 3: HT

- HT-Mixed
- HT20 1SS MCS0–7
- HT40 1SS MCS0–7
- HT-SIG / HT-STF / HT-LTF
- short GI は後段

---

# Part L. References

Primary standard:

- IEEE 802.11 Working Group: https://www.ieee802.org/11/
- IEEE Std 802.11-2024 — current base standard
- IEEE Std 802.11-2020 — Clause 9, 16, 17, 18, 19 used for implementation cross-reference in this memo
- IEEE Std 802.11a-1999 — legacy OFDM Annex/reference vectors remain useful for algorithm verification

Vendor / educational references:

- MathWorks, Non-HT PPDU Structure: https://www.mathworks.com/help/wlan/gs/non-ht-ppdu-structure.html
- MathWorks, HT PPDU Structure: https://www.mathworks.com/help/wlan/gs/ht-ppdu-structure.html
- Keysight, OFDM Signal Structure: https://helpfiles.keysight.com/csg/n7617b/Content/Main/ofdm_signal_structure.htm
- Keysight, DSSS Frame Structure: https://helpfiles.keysight.com/csg/n7617/Content/Main/dsss_frame_structure.htm

Independent implementations for comparison only:

- https://github.com/bastibl/gr-ieee802-11
- https://github.com/cloud9477/gr-ieee80211

---

## 最終設計判断

Pluto VSG では Wi-Fi を既存 Bluetooth engine の単なる `ModulationKind` 追加として扱わず、**Wi-Fi 専用 PHY engine** として実装します。

ただし以下は既存 VSG infrastructure を再利用します。

```text
Project persistence
UI shell
RF / Pluto backend
RF Level / active RMS concept
Packet scheduler / repeat
Export
Preview plots
```

Wi-Fi 固有部分は:

```text
MAC/PSDU builder
      ↓
Wi-Fi PHY encoder
      ↓
OFDM or DSSS/CCK waveform
```

として分離します。

特に OFDM では、**protocol logical structure と time-domain waveform structure を同じ field boundary tree に無理に押し込まない**ことを設計原則とします。これは Wi-Fi VSG を将来 HT/VHT 系へ拡張し、さらに Pluto VSA の Wi-Fi analyzer と共通概念で扱うために重要です。
