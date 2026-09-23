# Bluetooth RF テストパケット設計メモ

対象: **BR / EDR 2M / EDR 3M / LE 1M / LE 2M / LE HDT**  
作成日: 2026-08-25  
目的: VSG によるテストパケット生成、および VSA による復調・変調解析システムを設計するための実装メモ。  
主な参照仕様: **Bluetooth Core Specification v6.3** および **High Data Throughput HDT_VSr03_PR**。

> このメモは、BR、EDR 2M、EDR 3M、LE 1M、LE 2M、LE HDT の RF テストパケット生成・解析に必要な packet format、変調方式、payload pattern、EVM/DEVM などをまとめたものです。Bluetooth RF Test Specification や認証用テスターの代替ではありません。参照仕様から特定できない詳細は **implementation specific** として明記します。

---

## 0. 参照箇所マップ

### BR / EDR

- **Core Specification v6.3, Vol 2, Part A, Section 1**  
  BR/EDR radio overview、symbol rate。
- **Core Specification v6.3, Vol 2, Part A, Section 3.1**  
  BR GFSK modulation characteristics。
- **Core Specification v6.3, Vol 2, Part A, Section 3.2.1**  
  EDR modulation、pulse shaping、DEVM limits。
- **Core Specification v6.3, Vol 2, Part A, Appendix C.1**  
  EDR DEVM definition。
- **Core Specification v6.3, Vol 2, Part B, Sections 6.1–6.7**  
  BR/EDR packet formats、bit ordering、payload format、ACL packet summary。
- **Core Specification v6.3, Vol 3, Part D, Section 1.1.2**  
  BR/EDR transmitter test packet format、payload patterns、whitening disabled。

### LE 1M / LE 2M

- **Core Specification v6.3, Vol 6, Part A, Sections 1, 3.1, 4.6**  
  LE PHY overview、GFSK modulation、reference signal。
- **Core Specification v6.3, Vol 6, Part B, Section 2.1**  
  LE Uncoded PHY packet format。
- **Core Specification v6.3, Vol 6, Part F, Sections 3–4**  
  LE Direct Test Mode および LE test packet definition。

### LE HDT

- **High Data Throughput HDT_VSr03_PR, Vol 6, Part A, Sections 3.6 and 7**  
  HDT transmitter characteristics、rate table、symbol mappings、training sequences。
- **High Data Throughput HDT_VSr03_PR, Vol 6, Part A, Appendix C**  
  HDT RMS EVM。
- **High Data Throughput HDT_VSr03_PR, Vol 6, Part B, Sections 2.7, 3.1, 3.4**  
  HDT packet format、HEC/CRC、whitening/FEC/puncturing。
- **High Data Throughput HDT_VSr03_PR, Vol 6, Part F, Sections 4.2 and 5.3**  
  HDT RF test packet および test controls。

---

## 1. 共通実装ルール

### 1.1 Bit order

BR/EDR Baseband の bit order は little-endian 形式です。

- LSB が `b0` に対応します。
- LSB が air interface 上で最初に送信されます。
- 図示上も LSB が左側に置かれることがあります。

LE/HDT では field ごとの bit order に従います。LE test packet では多くの sequence が **transmission order** で明示されます。HDT の symbol mapping では、16QAM の `(b4k+0, b4k+1, b4k+2, b4k+3)` のように、仕様書の tuple order をそのまま実装します。symbol 内で MSB/LSB を反転しないことが重要です。

### 1.2 PHY family summary

| Family | Test packet basis | PHY/rate | Modulation | Symbol rate | Test whitening |
|---|---|---:|---|---:|---|
| BR DHx | BR/EDR Transmitter Test normal packet | BR 1M | GFSK, BT=0.5 | 1 Msym/s | Disabled |
| EDR 2-DHx | BR/EDR Transmitter Test normal packet | EDR 2M | GFSK access/header + π/4-DQPSK payload | 1 Msym/s | Disabled |
| EDR 3-DHx | BR/EDR Transmitter Test normal packet | EDR 3M | GFSK access/header + 8DPSK payload | 1 Msym/s | Disabled |
| LE 1M | LE Direct Test Mode packet | LE 1M | GFSK, BT=0.5 | 1 Msym/s | Disabled |
| LE 2M | LE Direct Test Mode packet | LE 2M | GFSK, BT=0.5 | 2 Msym/s | Disabled |
| LE HDT | LE HDT RF PHY Test packet | HDT2/3/4/6/7.5 | π/4 QPSK / 8PSK / 16QAM | 2 Msym/s | Disabled |

---

# Part A. BR / EDR テストパケット

## A.1 BR/EDR transmitter test mode

参照: **Core Specification v6.3, Vol 3, Part D, Section 1.1.2**

BR/EDR transmitter test では以下のように扱います。

- 同じ test packet が各 transmission で繰り返し送信されます。
- Tester が packet type と payload length を定義します。
- Payload length の制限には **Vol 2, Part B, Section 6.5** の Baseband 仕様が適用されます。
- ACL/SCO/eSCO payload structure は Baseband で定義された構造を維持します。
- FEC なしの packet のみを使います。
  - `HV3`
  - `EV3`, `EV5`
  - `DH1`, `DH3`, `DH5`
  - `2-EV3`, `2-EV5`
  - `3-EV3`, `3-EV5`
  - `2-DH1`, `2-DH3`, `2-DH5`
  - `3-DH1`, `3-DH3`, `3-DH5`
  - `AUX1`
- Transmitter test mode では whitening は off です。

Payload pattern:

- constant zero
- constant one
- alternating `1010...`
- alternating `11110000 11110000...`
- PRBS9
- transmission off

PRBS9:

- 9-stage LFSR
- feedback = 5th stage output と 9th stage output の XOR
- 初期値 = nine ones
- period = `2^9 - 1 = 511 bits`

## A.2 BR packet structure and modulation

参照: **Core Specification v6.3, Vol 2, Part B, Section 6.1.1**

```text
BR packet:
  Access Code   68 or 72 bits
  Header        54 bits
  Payload       0 to 2790 bits
```

packet header が続く normal packet では:

```text
Access Code = 72 bits
Header      = 54 bits
```

参照: **Core Specification v6.3, Vol 2, Part A, Section 3.1.1**

BR modulation:

- GFSK
- BT = 0.5
- modulation index = 0.28 to 0.35
- binary `1` = positive frequency deviation
- binary `0` = negative frequency deviation
- symbol rate = 1 Msym/s
- gross air data rate = 1 Mb/s

## A.3 BR DHx ACL test packet lengths

参照: **Core Specification v6.3, Vol 2, Part B, Section 6.7, Table 6.8**

| Packet type | Payload header | User payload range | FEC | CRC |
|---|---:|---:|---|---|
| DH1 | 1 octet | 0–27 octets | No | Yes |
| DH3 | 2 octets | 0–183 octets | No | Yes |
| DH5 | 2 octets | 0–339 octets | No | Yes |

前提:

- 72-bit access code と 54-bit header を持つ normal BR packet
- AES-CCM MIC なし
- CRC = 16 bits
- `U` = user payload octets
- `H` = payload header octets

```text
BR_DHx_total_bits = 72 + 54 + 8*H + 8*U + 16
BR_DHx_duration_us = BR_DHx_total_bits
```

最大長の例:

| Packet | H | U max | Total bits/symbols | Duration |
|---|---:|---:|---:|---:|
| DH1 | 1 | 27 | 366 | 366 µs |
| DH3 | 2 | 183 | 1622 | 1622 µs |
| DH5 | 2 | 339 | 2870 | 2870 µs |

## A.4 EDR packet structure

参照: **Core Specification v6.3, Vol 2, Part B, Section 6.1.2**

```text
EDR packet:
  Access Code
  Header
  Guard
  Sync
  Enhanced Data Rate Payload
  Trailer
```

- Access Code と Header は Basic Rate packet と同じ format および modulation です。
- Access Code と Header は GFSK です。
- Sync、EDR Payload、Trailer は EDR PSK modulation です。
- Trailer は 2 symbols です。

EDR modulation overview:

| Mode | Modulation | Symbol rate | Gross rate |
|---|---|---:|---:|
| EDR 2M | π/4-DQPSK | 1 Msym/s | 2 Mb/s |
| EDR 3M | 8DPSK | 1 Msym/s | 3 Mb/s |

Pulse shaping:

- square-root raised cosine
- symbol period `T = 1 µs`
- roll-off factor `β = 0.4`

## A.5 EDR guard, synchronization sequence, trailer

参照: **Core Specification v6.3, Vol 2, Part B, Section 6.6.1**

Guard time:

```text
4.75 µs to 5.25 µs
```

Synchronization sequence:

```text
duration  = 11 µs
length    = 11 DPSK symbols
structure = reference symbol + 10 DPSK symbols
```

Phase changes:

```text
+3π/4, -3π/4, +3π/4, -3π/4, +3π/4,
-3π/4, -3π/4, +3π/4, +3π/4, +3π/4
```

2 Mbps EDR π/4-DQPSK の generating bit sequence:

```text
01 11 01 11 01 11 11 01 01 01
```

Trailer:

```text
2 DPSK symbols
```

Trailer symbols は DEVM 測定対象に含めません。

## A.6 EDR ACL packet lengths

参照: **Core Specification v6.3, Vol 2, Part B, Section 6.7, Table 6.8**

| Packet type | Payload header | User payload range | EDR modulation | FEC | CRC |
|---|---:|---:|---|---|---|
| 2-DH1 | 2 octets | 0–54 octets | π/4-DQPSK | No | Yes |
| 2-DH3 | 2 octets | 0–367 octets | π/4-DQPSK | No | Yes |
| 2-DH5 | 2 octets | 0–679 octets | π/4-DQPSK | No | Yes |
| 3-DH1 | 2 octets | 0–83 octets | 8DPSK | No | Yes |
| 3-DH3 | 2 octets | 0–552 octets | 8DPSK | No | Yes |
| 3-DH5 | 2 octets | 0–1021 octets | 8DPSK | No | Yes |

前提:

- access code = 72 bits
- GFSK packet header = 54 bits
- nominal guard = 5 µs
- EDR sync = 11 symbols
- EDR trailer = 2 symbols
- EDR payload header = 2 octets
- CRC = 16 bits
- AES-CCM MIC なし
- `U` = user payload octets

2-DHx:

```text
EDR_payload_bits    = 8*(2 + U) + 16
EDR_payload_symbols = EDR_payload_bits / 2
2-DHx_total_us ≈ 126 + 5 + 11 + EDR_payload_symbols + 2
```

最大 2-DHx の例:

| Packet | U max | EDR payload bits | EDR payload symbols | Nominal total duration |
|---|---:|---:|---:|---:|
| 2-DH1 | 54 | 464 | 232 | 376 µs |
| 2-DH3 | 367 | 2968 | 1484 | 1628 µs |
| 2-DH5 | 679 | 5464 | 2732 | 2876 µs |

3-DHx:

```text
EDR_payload_bits    = 8*(2 + U) + 16
EDR_payload_symbols = EDR_payload_bits / 3
3-DHx_total_us ≈ 126 + 5 + 11 + EDR_payload_symbols + 2
```

注意:

- 8DPSK symbol count が整数にならない場合の padding 規則を、このメモでは推測しません。
- 任意の 3-DHx payload を生成する前に、対象 packet section で有効な payload length と packing/padding behavior を確認します。

## A.7 EDR DEVM limits

参照: **Core Specification v6.3, Vol 2, Part A, Section 3.2.1.4**

DEVM measurement:

- synchronization sequence と payload portions に対して測定
- trailer symbols は対象外
- 200 non-overlapping blocks
- each block = 50 symbols
- measurement filter = square-root raised cosine, roll-off 0.4, 3 dB bandwidth ±500 kHz

| EDR modulation | RMS DEVM | 99% DEVM | Peak DEVM |
|---|---:|---:|---:|
| π/4-DQPSK | ≤ 0.20 | ≤ 0.30 | ≤ 0.35 |
| 8DPSK | ≤ 0.13 | ≤ 0.20 | ≤ 0.25 |

Implementation specific:

- 50-symbol block を構成できない端数 symbols の扱い
- packet-to-block collection policy
- Appendix C.1 の数値最適化アルゴリズム

---

# Part B. LE 1M / LE 2M テストパケット

## B.1 LE Direct Test Mode basics

参照: **Core Specification v6.3, Vol 6, Part F, Section 3**

Commands:

- `LE_Test_Setup`
- `LE_Receiver_Test`
- `LE_Transmitter_Test`
- `LE_Test_End`

2-wire UART command fields:

- Frequency: `N` は `(2N + 2402) MHz` を表す。`N = 0x00..0x27`
- Length: 下位 6 bits が packet payload length。上位 2 bits は `LE_Test_Setup` Control `0x01` で設定
- PKT:
  - `00`: PRBS9
  - `01`: `11110000`
  - `10`: `10101010`
  - `11`: LE Uncoded PHY では vendor-specific、LE Coded PHY では `11111111`

PHY selection via `LE_Test_Setup` Control `0x02`:

- `0x04..0x07`: LE 1M
- `0x08..0x0B`: LE 2M
- `0x0C..0x0F`: LE Coded S=8
- `0x10..0x13`: LE Coded S=2

## B.2 LE Uncoded PHY test packet format

参照: **Core Specification v6.3, Vol 6, Part F, Section 4.1**

```text
LE Uncoded PHY Test Packet:
  Preamble
  Sync Word
  PDU Header
  PDU Length
  [CTEInfo]
  PDU Payload
  CRC
  [Constant Tone Extension]
```

Fields:

- Preamble: LE 1M は 8 bits、LE 2M は 16 bits
- Sync Word: 32 bits
- PDU Header: 8 bits
- PDU Length: 8 bits
- optional CTEInfo: 0 or 8 bits
- CRC: 24 bits

Whitening:

```text
LE test packets shall not use whitening.
```

Sync word in transmission order:

```text
10010100100000100110111010001110
```

Preamble:

```text
LE 1M: 10101010
LE 2M: 1010101010101010
```

CRC:

```text
CRCInit = 0x555555 for every LE test packet.
```

PDU Header:

```text
Payload Type    4 bits
RFU             2 bits
CP              1 bit  // CTEInfo Present
RFU             1 bit
```

PDU Length:

```text
Payload length in bytes
```

## B.3 LE test payload types

参照: **Core Specification v6.3, Vol 6, Part F, Section 4.1.4, Table 4.1**

| Payload type | Description |
|---:|---|
| `0b0000` | PRBS9 `11111111100000111101...` |
| `0b0001` | repeated `11110000` |
| `0b0010` | repeated `10101010` |
| `0b0011` | PRBS15 |
| `0b0100` | repeated `11111111` |
| `0b0101` | repeated `00000000` |
| `0b0110` | repeated `00001111` |
| `0b0111` | repeated `01010101` |

PRBS9:

- 9-bit LFSR
- taps: 5th and 9th stage XOR
- 初期値 = nine ones
- period = 511 bits

PRBS15:

- 15-bit LFSR
- taps: 14th and 15th stage XOR
- 初期値 = fifteen ones
- period = 32767 bits

## B.4 LE 1M / LE 2M modulation

参照: **Core Specification v6.3, Vol 6, Part A, Section 3.1**

| PHY | Modulation | Symbol rate | Data rate | BT | Modulation index |
|---|---|---:|---:|---:|---:|
| LE 1M | GFSK | 1 Msym/s | 1 Mb/s | 0.5 | 0.45–0.55 |
| LE 2M | GFSK | 2 Msym/s | 2 Mb/s | 0.5 | 0.45–0.55 |

その他:

- binary `1`: positive frequency deviation
- binary `0`: negative frequency deviation
- minimum frequency deviation:
  - 1 Msym/s で ≥185 kHz
  - 2 Msym/s で ≥370 kHz
- symbol timing accuracy は ±50 ppm より良いこと
- zero crossing error < ±1/8 symbol period

## B.5 LE 1M / LE 2M duration formulas

前提:

- CTE なし
- payload length = `L` octets
- PDU header = 1 octet
- PDU length = 1 octet
- CRC = 3 octets

LE 1M:

```text
LE1M_total_bits = 8 + 32 + 8 + 8 + 8*L + 24 = 80 + 8*L
LE1M_duration_us = 80 + 8*L
```

LE 2M:

```text
LE2M_total_bits = 16 + 32 + 8 + 8 + 8*L + 24 = 88 + 8*L
LE2M_duration_us = (88 + 8*L) / 2
```

Example `L=37`:

```text
LE1M = 376 µs
LE2M = 192 µs
```

CTE がある場合は CTE duration を追加します。CTE は unwhitened で、CRC/MIC calculation には含めません。

---

# Part C. LE HDT テストパケット

## C.1 HDT rate table

参照: **HDT_VSr03_PR, Vol 6, Part A, Sections 3.6 and 7**、**Vol 6, Part B, Section 3.4**

LE HDT PHY:

```text
symbol rate   = 2 Msym/s
symbol period = 0.5 µs
```

| HDT rate | RI | Effective bit rate | Modulation | Bits/symbol | Payload coding rate | PDU Header coding rate in packet format 1 | Control Header RMS EVM | PDU Header / payload RMS EVM |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| HDT2 | `0b001` | 2 Mb/s | π/4 QPSK | 2 | 1/2 | 1/2 | -10 dB | -10 dB |
| HDT3 | `0b010` | 3 Mb/s | π/4 QPSK | 2 | 3/4 | 1/2 | -10 dB | -13 dB |
| HDT4 | `0b011` | 4 Mb/s | 8PSK | 3 | 2/3 | 1/2 | -10 dB | -16 dB |
| HDT6 | `0b100` | 6 Mb/s | 16QAM | 4 | 3/4 | 1/2 | -10 dB | -19 dB |
| HDT7.5 | `0b101` | 7.5 Mb/s | 16QAM | 4 | 15/16 | 1/2 | -10 dB | -22 dB |

## C.2 HDT RF PHY Test Mode controls

参照: **HDT_VSr03_PR, Vol 6, Part F, Section 5.3**

PHY value:

```text
0x05 = LE HDT PHY
```

Rate indicator:

| Value | Meaning |
|---:|---|
| `0x00` | short format |
| `0x01` | HDT2 |
| `0x02` | HDT3 |
| `0x03` | HDT4 |
| `0x04` | HDT6 |
| `0x05` | HDT7.5 |

Packet format indicator:

| Value | Meaning |
|---:|---|
| `0x00` | PFI = 0: short format or packet format 0 |
| `0x01` | PFI = 1: packet format 1 |

Payload length:

- short format: `0x0000`
- packet format 0: `0x0001` to `0x01FE`
- packet format 1: unused

PHY Interval:

| Value | PHY INT | Payload symbols per interval |
|---:|---:|---:|
| `0x00` | `0b00` | 128 |
| `0x01` | `0b01` | 256 |
| `0x02` | `0b10` | 384 |
| `0x03` | `0b11` | 512 |

## C.3 HDT test packet format

参照: **HDT_VSr03_PR, Vol 6, Part F, Section 4.2**

Short format / packet format 0:

```text
Preamble              37 µs
Control Header        31 µs
[PDU Header]          8 to 504 bits
[Payload Zone]        0 to 4120 bits
[Terminating symbols] 1 µs
[CTE]                 24 to 160 µs
```

Packet format 1:

```text
Preamble              37 µs
Control Header        31 µs
PDU Header            8 to 504 bits
Terminating symbols   1 µs

Payload Zone:
  PHY Interval        128 to 512 symbols
  Terminating symbols 1 µs
  PITS between intervals
  Final PHY Interval  3 to 512 symbols
  Terminating symbols 1 µs

[CTE]                 24 to 160 µs
```

HDT RF PHY test packets では whitening は使用しません。

## C.4 HDT Control Header

参照: **HDT_VSr03_PR, Vol 6, Part B, Section 2.7.1**

```text
Control Header:
  PCA-A         16 bits
  NESN           3 bits
  PFI            1 bit
  RI             3 bits
  RFU            1 bit
  PDU Control    9 bits
  HEC-C         24 bits
```

RF PHY test packet では:

- PCA = `0x9F_1555_5555`
- NESN = 1
- PFI、RI、PDU Control は test configuration に依存
- HEC-C は Control Header fields を保護します。

## C.5 Packet format 0 と packet format 1

Packet format 0:

```text
1 packet -> 1 payload -> 1 block
```

- 32-bit CRC は PDU Header Zone と Payload Zone の combined に対して計算されます。

Packet format 1:

```text
1 packet -> up to 4 payloads
each payload -> 1..16 blocks
each block -> own 32-bit CRC
```

Payload blocks は複数 packet にまたがる場合があります。また、1 packet に複数 payload 由来の blocks が含まれる場合があります。

## C.6 HDT training sequences

参照: **HDT_VSr03_PR, Vol 6, Part A, Section 7.4**

Preamble:

```text
Preamble = STS × 9 + GI + LTS × 2
Duration = 18 µs + 2 µs + 17 µs = 37 µs
```

STS:

```text
[-1, -j, j, 1]
```

GI:

```text
GI = xu(13), xu(14), xu(15), xu(16)
```

LTS:

```text
x_u(k) = exp(-jπu k(k+1) / 17) * exp(j2πp / 17)
```

RF PHY test packets では:

```text
u = 7
```

Terminating symbols:

- 2 symbols
- 終了対象の stream の modulation において all-zero bits を表す symbols。

PITS:

```text
[-1, +1, +1, +1, -1, +1]
```

## C.7 HDT FEC and puncturing

参照: **HDT_VSr03_PR, Vol 6, Part B, Section 3.4**

FEC:

- non-systematic, non-recursive convolutional code
- rate 1/2
- constraint length 6
- 32 states
- initial state all zeros
- 末尾に five zero termination bits を追加

Generator polynomials:

```text
G0(x) = 1 + x^2 + x^4 + x^5
G1(x) = 1 + x + x^2 + x^3 + x^5
```

Puncturing:

| Coding rate | Pattern |
|---:|---|
| 1/2 | `[1 1]` |
| 2/3 | `[1 1 0 1]` |
| 3/4 | `[1 1 0 1 0 1]` |
| 15/16 | `[1 1 0 1 1 0 1 0 1 0 0 1 0 1 0 1 1 0 1 0 0 1 0 1 0 1 1 0 0 1]` |

## C.8 HDT symbol mapping

参照: **HDT_VSr03_PR, Vol 6, Part A, Section 7.3**

bitstream length が `log2(M)` の倍数でない場合、最後の symbol を作るため zero padding bits を追加します。

### π/4 QPSK: HDT2/HDT3

| Bits for `S2k` | `S2k` | Bits for `S2k+1` | `S2k+1` |
|---|---|---|---|
| `(0,0)` | `e^(jπ/4)` | `(0,0)` | `e^(jπ/2)` |
| `(0,1)` | `e^(j3π/4)` | `(0,1)` | `e^(jπ)` |
| `(1,0)` | `e^(-jπ/4)` | `(1,0)` | `e^0` |
| `(1,1)` | `e^(-j3π/4)` | `(1,1)` | `e^(-jπ/2)` |

### 8PSK: HDT4

| Bits | Symbol |
|---|---|
| `(0,0,0)` | `e^0` |
| `(0,0,1)` | `e^(jπ/4)` |
| `(0,1,0)` | `e^(j3π/4)` |
| `(0,1,1)` | `e^(jπ/2)` |
| `(1,0,0)` | `e^(-jπ/4)` |
| `(1,0,1)` | `e^(-jπ/2)` |
| `(1,1,0)` | `e^(-jπ)` |
| `(1,1,1)` | `e^(-j3π/4)` |

### 16QAM: HDT6/HDT7.5

仕様表は `Sk × √10` を与えるため、実際の `Sk` は √10 で割ります。

| Bits | `Sk` |
|---|---|
| `(0,0,0,0)` | `(-3 - 3j)/√10` |
| `(0,0,0,1)` | `(-3 - 1j)/√10` |
| `(0,0,1,0)` | `(-3 + 3j)/√10` |
| `(0,0,1,1)` | `(-3 + 1j)/√10` |
| `(0,1,0,0)` | `(-1 - 3j)/√10` |
| `(0,1,0,1)` | `(-1 - 1j)/√10` |
| `(0,1,1,0)` | `(-1 + 3j)/√10` |
| `(0,1,1,1)` | `(-1 + 1j)/√10` |
| `(1,0,0,0)` | `(3 - 3j)/√10` |
| `(1,0,0,1)` | `(3 - 1j)/√10` |
| `(1,0,1,0)` | `(3 + 3j)/√10` |
| `(1,0,1,1)` | `(3 + 1j)/√10` |
| `(1,1,0,0)` | `(1 - 3j)/√10` |
| `(1,1,0,1)` | `(1 - 1j)/√10` |
| `(1,1,1,0)` | `(1 + 3j)/√10` |
| `(1,1,1,1)` | `(1 + 1j)/√10` |

## C.9 HDT RMS EVM

参照: **HDT_VSr03_PR, Vol 6, Part A, Section 3.6 and Appendix C**

HDT は EDR DEVM ではなく **RMS EVM** を使います。

Procedure:

1. **Preamble processing**  
   amplitude、phase、carrier frequency error、timing を推定します。
2. **Control Header processing**  
   preamble estimate を使って Control Header RMS EVM を計算します。
3. **Payload processing**  
   PDU Header / Payload / Terminating symbol RMS EVM を計算します。

EVM limits:

| Rate | Control Header RMS EVM | PDU Header / payload RMS EVM |
|---|---:|---:|
| HDT2 | -10 dB | -10 dB |
| HDT3 | -10 dB | -13 dB |
| HDT4 | -10 dB | -16 dB |
| HDT6 | -10 dB | -19 dB |
| HDT7.5 | -10 dB | -22 dB |

---

# Part D. VSG/VSA implementation checklist

## D.1 BR

VSG:

- access code
- 54-bit packet header
- DH1/DH3/DH5 payload header、payload body、16-bit CRC
- test mode では whitening disabled
- GFSK BT=0.5、1 Msym/s
- PRBS9 を含む payload patterns

VSA:

- access-code detection
- GFSK demodulation
- header/payload/CRC validation
- EVM ではなく GFSK metrics を評価

## D.2 EDR 2M/3M

VSG:

- BR GFSK access/header
- guard 4.75–5.25 µs
- EDR sync sequence
- 2-DHx or 3-DHx payload
- trailer
- transmitter test では whitening disabled
- PSK payload portion に RRC β=0.4

VSA:

- GFSK prefix を検出
- GFSK-to-DPSK boundary を合わせる
- sync sequence を使う
- DEVM を計算
- trailer は DEVM から除外

## D.3 LE 1M/2M

VSG:

- preamble
- sync word
- PDU header/length/payload
- CRCInit 0x555555
- whitening disabled
- optional CTE

VSA:

- preamble + sync word を検出
- GFSK demodulate
- payload pattern と CRC を検証
- GFSK metrics を評価

### D.3.1 Pluto VSG 実装状況（2026-08-25）

LE 1M/2Mの編集可能なpacket generatorを実装し、Direct Test Mode用RF Test Packetは
専用generatorではなく、同じfield/settingsへ規定値を入力するpresetとして扱う。

- PHY: LE 1M / LE 2M
- Samples/Symbol初期値: 8（8 MS/s / 16 MS/s）
- Preamble、Access Address/Sync Word、PDU Headerをair-order bit列で直接編集可能
- Payload source: Fixed、Pattern、PRBS9、PRBS15
- Payload Length: 0～255 byte、初期値37 byte
- CRC-24: 有効/無効と24-bit CRCInitを編集可能
- Whitening: 有効/無効とChannel Index 0～39を編集可能
- GFSK: BT=0.5、初期DeviationはLE 1M=250 kHz、LE 2M=500 kHz
- Power ramp、pre/post idle、1～1000回のrepeat、Preview、NPZ/IQ TAR/WV exportを共通利用

RF Test Packet presetを適用すると以下を同じ編集欄へロードする。

- Preamble: LE 1M=8 bit、LE 2M=16 bit
- Sync Word: `10010100100000100110111010001110`（transmission order）
- PDU Header: Payload Type code + RFU/CP=0
- Payload Type: PRBS9、`11110000`、`10101010`、PRBS15、`11111111`、
  `00000000`、`00001111`、`01010101`
- CRCInit=`0x555555`、Whitening OFF
- Packet interval: `ceil((packet_duration_us + 249) / 625) * 625 us`

BR/EDR SettingsにもRF test payload presetを設け、PRBS9、固定0/1、交互`1010`、
反復`11110000`をPayload Source/Dataへロードし、WhiteningをOFFにする。メインメニューの
default presetはBR/EDRではPRBS9、LEでは現在選択中のRF test payload typeを適用する。

CTE/CTEInfo、LE Coded PHY、HCI command生成は未実装である。CRCとWhiteningはPDU
（Header、Length、Payload）およびCRCの正しい処理範囲を分離し、CRCはCore仕様のLFSR
とBluetooth SIG公式sample packetの既知CRCを照合する。

### C.2.1 LE CRC-24修正記録（2026-08-27）

初期実装の`le_crc24_bits()`はLFSRのfeedback tap位置を誤っており、LE 1M/2Mの
両方で不正なCRCを生成していた。実装とテストが同じ誤ったregister更新式を複製していたため、
自己整合テストだけでは検出できなかった。

修正後は、PDU bitを送信順に処理し、次の生成多項式を使用する。

```text
x^24 + x^10 + x^9 + x^6 + x^4 + x^3 + x + 1
lower polynomial mask = 0x00065B
CRC output order = register position 23 down to 0
```

Bluetooth Core Vol 6, Part Cのsample packetを固定回帰vectorとして追加した。

```text
PDU bytes (transmission order): 00 03 42 4C 45
CRC bytes (transmission order): 29 0A CE
CRCInit: 0x555555
```

同一PDUならLE 1M/2MでCRC値は同一であり、PHYによる違いはpreamble長とsymbol rateである。
BR/EDRは別のCRC-16実装を使用する。Bluetooth SIG sample vector
`4E 01 02 03 04 05 06 07 08 09`、UAP `0x47`に対するCRC `6D D2`との一致を再確認し、
DH1/DH3/DH5、2-DHx、3-DHxの全生成経路がpayload headerとbody全体から同じCRC-16を
構成する回帰テストを追加した。したがって今回の計算修正対象はLE 1M/2Mのみである。

## D.4 HDT

VSG:

- STS/GI/LTS preamble
- Control Header
- まず packet format 0、その後 packet format 1
- FEC + puncturing
- π/4 QPSK / 8PSK / 16QAM mapping
- terminating symbols、PITS
- test packet では whitening disabled
- RRC pulse shaping

VSA:

- preamble detection and parameter estimation
- Control Header PFI/RI decode
- known test packet から reference symbols を生成
- Control Header RMS EVM を計算
- PDU Header/Payload RMS EVM を計算
- rate-specific limits と比較

---

# Part E. 未解決・実装確認が必要な項目

## BR/EDR

- 各 payload pattern / packet type に対する `LMP_TEST_CONTROL` parameter encoding
- 8DPSK symbol grouping を満たすための有効な 3-DHx arbitrary payload length
- relevant test scenario で encryption が有効な場合の AES-CCM MIC handling
- 選択した piconet/test setup に対する access code/header generation

## LE

- HCI を使う場合の正確な HCI command variant
- Controller-supported maximum payload length
- CTE を含める場合の CTEInfo と antenna switching details

## HDT

- HEC-C、HEC-P、32-bit CRC の bit-level implementation
- PDU Header Zone の完全な construction
- Packet format 1 scheduling と per-block CRC/reference-symbol generation
- Appendix C の numerical optimization details
- unknown-payload VSA decode flow

---

# 付録: 設計上の注意

## 1. DEVM と EVM を混同しない

- EDR は **DEVM**。
- HDT は **RMS EVM**。
- LE 1M/2M と BR は GFSK であり、EDR/HDT のような EVM/DEVM 規格ではなく GFSK modulation characteristics を評価する。

## 2. Whitening disabled の扱い

- BR/EDR transmitter test: whitening disabled。
- LE Direct Test Mode packet: whitening disabled。
- HDT RF PHY test packet: whitening disabled。
- 通常動作 packet では whitening が有効な場合があるため、test packet generator と normal packet analyzer を分ける。

## 3. Bit order の取り違え

EDR の DPSK、HDT の π/4 QPSK / 8PSK / 16QAM では、symbol mapping の tuple order を誤ると reference symbol 系列が変わり、EVM/DEVM が大きく悪化する。仕様書の bit tuple order をそのまま使う。

## 4. RF 実装と解析実装を分けて検証する

推奨する検証順序:

```text
1. ideal baseband waveform 生成
2. ideal waveform を VSA に入力して EVM/DEVM がほぼ 0 になるか確認
3. VSG から有線で DUT/VSA に入力
4. OTA 測定
5. unknown packet decode
```
