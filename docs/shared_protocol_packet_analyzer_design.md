# VSG / VSA 共通 Protocol Packet Analyzer 設計

作成日: 2026-08-29  
対象: **Pluto VSG / Pluto VSA 共通の packet semantic analyzer**  
初期対象: **Bluetooth BR / EDR / LE**  
将来対象: Wi-Fi、ADS-B、その他 bit-oriented protocol

> 本資料は、復調または生成済みの packet bit 列から protocol field を抽出し、raw bit/value だけでなく「その値が何を意味するか」まで共通形式で解釈するための設計案です。Ellisys の packet view に近い役割を想定しますが、現時点では VSG / VSA の具体的な画面レイアウトへ組み込まず、**解析 core と UI-independent result model を先に定義する**ことを目的とします。

---

## 1. 目的

現在の Pluto VSG は Bluetooth packet を生成する際に、Packet Type、LT_ADDR、FLOW、ARQN、SEQN、payload length、payload source などの protocol parameter を設定し、それらから packet bit 列と IQ を生成しています。

一方、Pluto VSA は受信 IQ から symbol / bit 列を復調できますが、bit 列そのものを表示するだけでは packet の意味を読み取るために Bluetooth Core Specification と照合する必要があります。

本機能ではこの間に共通の semantic layer を置きます。

```text
VSG
WaveformProject / Generator
        ↓
実際に生成された air-order bits
        ↓
        ┌───────────────────────────────┐
        │ Shared Protocol Packet Analyzer│
        └───────────────────────────────┘
                    ↓
            PacketAnalysisResult
                    ↑
        ┌───────────────────────────────┐
        │ Shared Protocol Packet Analyzer│
        └───────────────────────────────┘
        ↑
VSA demodulated air-order bits
```

これにより、VSG と VSA が同じ packet を扱った場合に、**同じ field 名、同じ value 表記、同じ meaning、同じ validation status** で結果を表示できます。

最終的には以下の round-trip validation が可能になります。

```text
VSG packet settings
        ↓
VSG generates air bits / IQ
        ↓
Protocol Packet Analyzer ①
        ↓
RF transmit
        ↓
VSA demodulation
        ↓
Protocol Packet Analyzer ②
        ↓
① と ② の semantic result を比較
```

---

## 2. 非目標

初期実装では以下を本機能の責務にしません。

- IQ からの packet detection
- carrier / timing synchronization
- GFSK / PSK / QAM demodulation
- EVM / DEVM 計算
- VSG waveform generation
- VSA / VSG の具体的な window layout
- Bluetooth 全上位 protocol の完全 decode
- Ellisys と同等の connection state tracking

本 Analyzer の基本責務は、**入力された packet bit 列を protocol structure と semantic value に変換すること**です。

VSA の RF / modulation analysis と protocol analysis は明確に分離します。

```text
IQ
 ↓
VSA demodulation
 ↓
bit stream
 ↓
Protocol Packet Analyzer
 ↓
semantic packet result
```

---

## 3. 設計上の最重要方針

### 3.1 VSG と VSA は同じ Analyzer core を使う

VSG 用 decoder と VSA 用 decoder を別々に実装しません。

VSG は generator の設定値そのものを Analyzer に渡すのではなく、**generator が実際に生成した bit 列**を Analyzer へ入力します。

これは generator の誤りを Analyzer が検出できるようにするためです。

悪い例:

```text
VSG Settings
 ├─ Generator
 └─ Analyzer
```

この構造では両方が同じ設定値を読むため、generator が誤った bit packing をしても Analyzer result は正しく見える可能性があります。

推奨:

```text
VSG Settings
     ↓
 Generator
     ↓
generated air bits
     ↓
 Analyzer
```

### 3.2 canonical input は transmission-order bit stream

共通 Analyzer の第一入力形式は、原則として **air interface 上の transmission order に並んだ demodulated bits** とします。

Bluetooth の場合、これは modulation symbol から hard decision された bit 列であり、必要に応じて parser 内で以下を処理します。

- FEC decode
- dewhitening
- HEC / CRC check
- field extraction
- enum / flag interpretation

VSG も同じ air-order bits を渡します。

これにより、VSG と VSA の入力条件を揃えられます。

### 3.3 raw value と meaning を分離する

Analyzer result は単に数値を返すだけでなく、少なくとも以下を分離して保持します。

```text
Raw Bits
Raw Value
Decoded Value
Meaning
Validation Status
```

例:

```text
Field      Raw     Value    Meaning
TYPE       0100    0x4      DH1
ARQN       1       1        ACK
CRC        xxxx    0x....   Valid
```

UI は `Value` のみ、`Value + Meaning`、あるいは全列を表示できます。

### 3.4 parser は malformed packet を捨てない

RF capture では bit error、truncation、unknown context が普通に発生します。

したがって protocol parser は、CRC mismatch や reserved value を理由に packet 全体を例外終了させません。

可能な範囲まで field を decode し、問題を status / issue として返します。

```text
Packet
 ├─ Header            Valid
 ├─ Length            27 bytes
 ├─ Payload           Parsed
 └─ CRC               Invalid
```

例外は API misuse や内部不整合など、programming error に限定します。

---

## 4. 推奨 package 構成

既存の `pluto_common` は device discovery / ownership を中心としており、protocol decode は今後かなり大きくなる可能性があります。

そのため、共有 protocol layer は独立 package とすることを推奨します。

```text
pluto_protocol/
├─ __init__.py
├─ model.py
├─ registry.py
├─ bitops.py
├─ formatter.py
├─ compare.py
│
├─ bluetooth/
│   ├─ __init__.py
│   ├─ common.py
│   ├─ br_edr.py
│   ├─ le.py
│   └─ hdt.py
│
├─ wifi/                 # future
│   ├─ __init__.py
│   ├─ mac.py
│   └─ legacy.py
│
└─ adsb/                 # future
    ├─ __init__.py
    └─ mode_s.py
```

重要な依存方向:

```text
pluto_protocol
     ↑      ↑
     │      │
pluto_vsg  pluto_sa.vsa
```

`pluto_protocol` から VSG / VSA / Qt へ依存してはいけません。

---

## 5. 現行コードとの関係

現在、Bluetooth protocol primitive の一部は VSA profile にあり、VSG engine がそれを import しています。

例:

```text
pluto_vsg.engine.bluetooth_br
        ↓ imports
pluto_sa.vsa.profiles.bluetooth_br
```

共有 Analyzer 導入時には、この依存を解消します。

以下のような純粋 protocol primitive は `pluto_protocol.bluetooth` へ移す候補です。

- access code bit generation / interpretation
- BR/EDR whitening sequence
- BR/EDR HEC
- BR/EDR payload CRC
- rate 1/3 FEC encode / decode
- packet TYPE table
- payload header field definition
- PRBS utilities
- LE whitening
- LE CRC-24
- LE header field interpretation

一方、以下は移しません。

### VSA に残す

- GFSK demodulator
- PSK demodulator
- timing / carrier synchronization
- IQ / EVM calculation
- packet detection

### VSG に残す

- GFSK modulator
- SRRC shaping
- power ramp
- IQ generation
- repetition / packet interval
- Pluto TX backend

この境界により、protocol definition を一か所にできます。

---

## 6. Input data contract

### 6.1 PacketDecodeInput

概念モデル:

```python
@dataclass(frozen=True)
class PacketDecodeInput:
    bits: np.ndarray
    representation: BitRepresentation
    protocol_hint: str | None
    phy_hint: str | None
    source: PacketSourceInfo
    context: Mapping[str, object]
```

### 6.2 BitRepresentation

初期候補:

```text
AIR
LOGICAL
```

`AIR`:

- demodulator hard decision 後
- transmission order
- whitening / FEC 等が air 上の状態のまま
- VSG / VSA 共通の推奨入力

`LOGICAL`:

- protocol-specific transform 後の bit 列
- unit test や低レベル debugging 用
- 通常 UI では使用しなくてもよい

将来必要になれば `DEWHITENED`、`FEC_DECODED` 等を追加できますが、最初から過剰に bit-domain を増やさない方がよいです。

### 6.3 PacketSourceInfo

protocol interpretation 自体とは分離した source metadata を持ちます。

```python
@dataclass(frozen=True)
class PacketSourceInfo:
    source_kind: str       # "VSG", "VSA", "File", "Test"
    packet_index: int | None = None
    timestamp_s: float | None = None
    center_frequency_hz: float | None = None
    start_sample: int | None = None
    stop_sample: int | None = None
```

Analyzer は source metadata の有無に依存して decode しません。

### 6.4 context

protocol decode に必要だが bit 列単体から決定できない情報を optional context として渡します。

Bluetooth BR / EDR の例:

```text
uap
clock_6_1
lap
whitening assumption
packet type hint
```

Bluetooth LE の例:

```text
channel index
CRCInit
Access Address / connection context
PHY
```

VSG は既知の context を渡せます。

VSA では context が不明な場合があります。その場合 parser は可能な範囲で candidate search / inference を行い、決定できなければ `Unknown` として返します。

**context 不足は parse failure ではありません。**

---

## 7. Output data contract

### 7.1 PacketAnalysisResult

```python
@dataclass(frozen=True)
class PacketAnalysisResult:
    schema_version: int
    protocol_id: str
    protocol_name: str
    phy_name: str | None
    packet_type: str | None
    summary: tuple[PacketSummaryItem, ...]
    root_fields: tuple[PacketField, ...]
    issues: tuple[PacketIssue, ...]
    integrity: PacketIntegritySummary
    source: PacketSourceInfo
    raw_bits: np.ndarray
```

この object を VSG / VSA 共通の唯一の semantic result とします。

### 7.2 PacketField

```python
@dataclass(frozen=True)
class PacketField:
    field_id: str
    name: str
    display_name: str
    bit_start: int | None
    bit_stop: int | None
    raw_bits: np.ndarray
    raw_value: int | bytes | str | None
    value: object
    meaning: str | None
    status: FieldStatus
    description: str | None
    children: tuple["PacketField", ...]
```

`bit_start / bit_stop` は canonical input bit stream 上の位置です。

直接 bit を持たない derived field は `None` を許可します。

例:

```text
Payload
 ├─ Header
 │   ├─ LLID
 │   ├─ FLOW
 │   └─ LENGTH
 ├─ Body
 └─ CRC
     ├─ Received CRC
     ├─ Calculated CRC
     └─ CRC Status      # derived
```

### 7.3 FieldStatus

最低限:

```text
VALID
INVALID
WARNING
UNKNOWN
INFO
```

用途例:

- CRC OK -> VALID
- CRC mismatch -> INVALID
- reserved bit is non-zero -> WARNING または INVALID
- UAP unknown -> UNKNOWN
- ordinary field -> INFO

### 7.4 PacketIssue

packet 全体に関わる anomaly を格納します。

```python
@dataclass(frozen=True)
class PacketIssue:
    severity: str
    code: str
    message: str
    field_id: str | None = None
```

例:

```text
TRUNCATED_PAYLOAD
CRC_MISMATCH
HEC_MISMATCH
UNKNOWN_PACKET_TYPE
MISSING_CONTEXT
RESERVED_VALUE
LENGTH_MISMATCH
```

---

## 8. semantic hierarchy

Analyzer は flat list ではなく、階層構造を authoritative model とします。

例: BR DH1

```text
Bluetooth BR Packet
├─ Access Code
│   ├─ Preamble
│   ├─ Sync Word
│   └─ Trailer
├─ Header
│   ├─ LT_ADDR
│   ├─ TYPE
│   ├─ FLOW
│   ├─ ARQN
│   ├─ SEQN
│   └─ HEC
└─ Payload
    ├─ Payload Header
    │   ├─ LLID
    │   ├─ FLOW
    │   └─ LENGTH
    ├─ Payload Body
    └─ CRC
```

EDR:

```text
Bluetooth EDR Packet
├─ Access Code
├─ Header
├─ EDR Sync
├─ EDR Payload
│   ├─ Payload Header
│   ├─ Payload Body
│   └─ CRC
└─ Trailer
```

Guard は RF waveform 上には存在しますが bit field ではありません。

Protocol Packet Analyzer の field tree に含める場合は `bit_start=None` の derived / waveform annotation とするか、初期実装では semantic tree から外します。

LE:

```text
Bluetooth LE Packet
├─ Preamble
├─ Access Address
├─ PDU
│   ├─ Header
│   │   ├─ PDU Type / LL-specific fields
│   │   ├─ RFU
│   │   └─ other flags
│   ├─ Length
│   └─ Payload
└─ CRC
```

実際の LE header semantics は Advertising / Data Channel / Direct Test Mode 等の context によって異なるため、同じ raw header bits を context-specific dissector へ渡します。

---

## 9. Parser / Dissector architecture

### 9.1 ProtocolDecoder interface

```python
class ProtocolDecoder(Protocol):
    protocol_id: str
    display_name: str

    def probe(self, packet: PacketDecodeInput) -> DecodeProbeResult:
        ...

    def decode(self, packet: PacketDecodeInput) -> PacketAnalysisResult:
        ...
```

### 9.2 registry

```text
ProtocolRegistry
 ├─ bluetooth.br_edr
 ├─ bluetooth.le
 ├─ bluetooth.hdt       future
 ├─ wifi.legacy         future
 └─ adsb.mode_s         future
```

通常、VSG / VSA profile が `protocol_hint` を渡すため auto detect は必須ではありません。

Auto detection を実装する場合も、confidence が低い packet を強制的に一つへ分類しないようにします。

```text
Bluetooth BR    confidence 0.92
Bluetooth LE    confidence 0.31
```

のような candidate result を返せる設計が望ましいです。

---

## 10. Layered dissector

Ellisys に近づける場合、将来は一つの packet parser がすべてを直接解釈するのではなく、layer ごとの dissector を連結できる構造にします。

例: Bluetooth BR/EDR ACL

```text
Baseband Packet
      ↓
ACL Payload
      ↓
L2CAP
      ↓
ATT / SMP / SDP / AVDTP ...
```

例: LE

```text
LE Link Layer
      ↓
L2CAP
      ↓
ATT / SMP / ISO adaptation ...
```

初期実装は **PHY/Baseband/Link Layer の packet structure 解釈まで**でよいですが、`PacketField.children` により後から higher-layer result を同じ tree に追加できるようにします。

上位 layer decoder は payload byte sequence と context を入力として受けます。

```python
class PayloadDissector(Protocol):
    def accepts(self, parent: PacketField, context: Mapping[str, object]) -> bool:
        ...

    def dissect(self, payload: bytes, context: Mapping[str, object]) -> tuple[PacketField, ...]:
        ...
```

---

## 11. Bluetooth 初期対応範囲

### Phase 1: BR / EDR DHx

対象:

```text
DH1 / DH3 / DH5
2-DH1 / 2-DH3 / 2-DH5
3-DH1 / 3-DH3 / 3-DH5
```

表示候補:

```text
PHY
Packet Type
LAP / Access Code
LT_ADDR
TYPE raw value / packet name
FLOW
ARQN
SEQN
HEC received / expected / status
Whitening state
Clock candidate
Payload LLID
Payload FLOW
Payload Length
Payload Body length
Payload preview (Hex)
CRC received / expected / status
```

EDR additionally:

```text
EDR modulation
EDR Sync status
EDR payload bit count
Trailer
```

### Phase 2: LE 1M / LE 2M

最初は現行 VSG が扱っている RF Test Packet / editable uncoded packet を対象とします。

表示候補:

```text
PHY
Preamble
Access Address
PDU Header
Payload Type / header meaning
Length
Payload Pattern / Hex preview
CRCInit
CRC received / expected / status
Whitening
Channel Index
```

その後 Advertising PDU / Data Channel PDU の semantic decode を追加します。

### Phase 3: HDT

HDT 実装時には既存 `PacketAnalysisResult` を再利用し、HEC、CRC、FEC、puncturing、rate、modulation などを同じ model に載せます。

---

## 12. unknown context / candidate resolution

Bluetooth BR/EDR では UAP や native clock が不明だと HEC / whitening の完全 decode ができない場合があります。

現行 VSA には HEC-valid candidate を clock / UAP に対して探索するロジックが存在するため、この機能は shared protocol layer へ整理できます。

結果は一つに決め打ちせず、必要であれば candidate を保持します。

```python
@dataclass(frozen=True)
class DecodeCandidate:
    confidence: float
    context: Mapping[str, object]
    result: PacketAnalysisResult
```

例:

```text
Candidate 1
  UAP       0x6B
  CLK_6_1   0x2B
  HEC       Valid
  CRC       Valid
  Confidence 1.0
```

CRC まで一致した candidate は非常に強い判断材料になります。

---

## 13. VSG integration contract

現行 VSG は `GenerationResult.metadata` に protocol-specific bit arrays を保持している箇所があります。

長期的には metadata の暗黙 key に依存せず、生成結果から protocol analyzer へ渡す packet bit artifact を明示化することを推奨します。

例:

```python
@dataclass(frozen=True)
class GeneratedPacketBits:
    protocol_id: str
    phy_name: str
    bits: np.ndarray
    context: Mapping[str, object]
    packet_index: int
```

```python
@dataclass(frozen=True)
class GenerationResult:
    iq: np.ndarray
    sample_rate_hz: float
    field_boundaries: tuple[FieldBoundary, ...]
    packet_bits: tuple[GeneratedPacketBits, ...]
    metadata: Mapping[str, object]
```

VSG Analyzer はこの `packet_bits` だけを見ることを原則とします。

`WaveformProject.bluetooth_br.lt_addr` 等から直接 result table を作りません。

---

## 14. VSA integration contract

VSA 側は modulation decoder から得られた bit 列を `PacketDecodeInput` へ変換します。

```text
CompositeVSAAnalysisResult
        ↓
decoded bits
        ↓
VSA Packet Source Adapter
        ↓
PacketDecodeInput
        ↓
Shared Protocol Analyzer
```

VSA-specific metadata:

```text
packet index
capture timestamp
sample range
RF center frequency
frequency error
EVM / DEVM
```

は protocol result の `source` または別 measurement result として紐づけます。

Protocol Analyzer 自体には EVM を計算させません。

将来 UI で一つの packet を選択したとき、

```text
RF / Modulation Results
Protocol Packet Results
```

を同じ packet ID で関連付けられる構造が望ましいです。

---

## 15. 共通 table representation

VSA の現在の Result Summary と同様、UI widget そのものではなく、まず canonical row data を定義します。

### 15.1 PacketTableRow

```python
@dataclass(frozen=True)
class PacketTableRow:
    row_id: str
    depth: int
    layer: str
    field: str
    bit_range: str
    raw: str
    value: str
    meaning: str
    status: str
```

`PacketAnalysisResult` の tree から共通 formatter が生成します。

```text
PacketAnalysisResult
        ↓
flatten_packet_rows()
        ↓
PacketTableRow[]
       ↙        ↘
  VSG table    VSA table
```

これにより、具体的な Qt widget や window layout を後から決めても表示内容の定義は変わりません。

### 15.2 推奨 table columns

現時点の候補:

| Column | 内容 |
|---|---|
| Field | protocol field 名。階層は indent 表示可能 |
| Value | decode 後の値 |
| Meaning | 値の意味 |
| Raw | bit / hex raw value |
| Bits | canonical input 上の bit range |
| Status | Valid / Invalid / Warning / Unknown |

VSA Result Summary のような簡潔な表示を優先する場合、初期表示は例えば以下でもよいです。

```text
Field                  Value                 Status
PHY                    BR                    -
Packet Type            DH1                   -
LT_ADDR                 1                     -
TYPE                    0x4 (DH1)            -
ARQN                    1 (ACK)              -
Payload Length          27 bytes              -
HEC                     0xA3                  Valid
CRC                     0x1234                Valid
```

`Raw`、`Bits`、詳細 field hierarchy は optional column / expanded view にできます。

**具体的な VSG / VSA UI 配置は本設計では決定しません。**

---

## 16. compact summary と detail tree

一つの packet について、用途の異なる二つの presentation を同じ result から生成できるようにします。

### Compact Summary

測定時に重要な代表値のみ。

```text
Protocol
PHY
Packet Type
Length
Address / Access Address
HEC / Header integrity
CRC / FCS
Payload type
```

### Detailed Packet Fields

packet field を全階層表示。

```text
Header
  LT_ADDR
  TYPE
  FLOW
  ARQN
  SEQN
  HEC
Payload Header
  LLID
  FLOW
  LENGTH
...
```

summary 用情報を parser 内で別ロジックとして重複生成せず、`PacketAnalysisResult.root_fields` を参照して summary item を構成することを推奨します。

---

## 17. value formatting rules

VSG / VSA 間で表示差が出ないよう、formatting も shared layer に置きます。

例:

```text
Address        0x8E89BED6
TYPE           0x4 (DH1)
Length         27 bytes
Boolean        1 (Enabled)
CRC            0x12AB (Valid)
Bit sequence   10101010
Byte payload   01 02 03 04 ...
```

基本原則:

- enum は numeric raw value と semantic name を両方保持
- address / CRC / HEC は field width に合わせた固定桁 Hex
- transmission-order bits と numeric representation を混同しない
- payload は巨大な一行文字列にせず preview + length を基本とする
- full payload は別 formatter で hex dump 可能にする

---

## 18. protocol-independent extensibility

この model は Bluetooth 固有名を core dataclass に入れません。

悪い例:

```python
PacketAnalysisResult(
    bluetooth_uap=...,
    bluetooth_clock=...,
)
```

推奨:

```python
PacketField(...)
PacketSummaryItem(...)
context={...}
```

これにより Wi-Fi では、

```text
Frame Control
Type / Subtype
To DS / From DS
Duration
Address 1/2/3/4
Sequence Control
QoS Control
FCS
```

を同じ field tree に載せられます。

ADS-B では、

```text
DF
CA
ICAO
ME
Parity
Type Code
Altitude
Callsign
Position
```

を同じ table model に載せられます。

Protocol ごとに異なるのは decoder / dissector だけです。

---

## 19. comparison support

VSG direct result と VSA receive result の比較用に semantic comparison を追加できる設計にします。

```python
@dataclass(frozen=True)
class PacketFieldDifference:
    field_id: str
    expected: object
    actual: object
    equal: bool
```

比較時に無視可能な field も定義します。

例:

- source timestamp
- packet index
- VSA RF measurement values
- auto-inferred context diagnostics

比較対象:

- packet type
- header fields
- payload length
- payload bytes
- HEC / CRC calculated values

VSG -> RF -> VSA round trip では、payload と semantic field が一致することを強い validation として使えます。

---

## 20. serialization

`PacketAnalysisResult` は JSON-compatible mapping へ変換可能にします。

用途:

- unit test fixture
- VSG / VSA regression comparison
- packet analysis result export
- future CLI
- user bug report

schema version を必ず持ちます。

```json
{
  "schema_version": 1,
  "protocol": "bluetooth.br_edr",
  "phy": "BR",
  "packet_type": "DH1",
  "summary": [],
  "fields": [],
  "issues": []
}
```

raw NumPy objectsをそのまま JSON に埋めず、bit string / integer / hex string / byte array などへ canonicalize します。

---

## 21. test strategy

### 21.1 protocol unit tests

仕様から既知の bit vector を与え、field interpretation を検証します。

```text
input bits
 ↓
Analyzer
 ↓
expected field tree
```

最低限:

- valid packet
- CRC mismatch
- HEC mismatch
- truncated packet
- unknown/reserved field value
- missing decode context
- whitening ON/OFF

### 21.2 VSG direct test

```text
VSG generator
 ↓
generated air bits
 ↓
Analyzer
 ↓
expected semantic result
```

VSG settings と result の一致だけでなく、actual generated bits の decode を確認します。

### 21.3 VSG -> VSA offline round trip

```text
VSG IQ
 ↓
VSA demod
 ↓
Analyzer
 ↓
VSG direct semantic result と比較
```

RF hardware なしで実行可能な最重要 regression test とします。

### 21.4 RF round trip

```text
Pluto VSG
 ↓ RF
Pluto VSA
 ↓
Protocol Analyzer
```

CRC / HEC / payload一致まで確認します。

---

## 22. implementation phases

### Phase 0: shared model

実装:

```text
pluto_protocol.model
pluto_protocol.registry
pluto_protocol.formatter
```

この段階では既存 UI を変更しません。

### Phase 1: Bluetooth common primitives 移設

VSA profile / VSG engine に分散している以下を shared layer へ移します。

```text
HEC
CRC
whitening
FEC
bit packing
packet type tables
LE CRC / whitening
```

既存 VSG / VSA の挙動を regression test で維持します。

### Phase 2: BR / EDR semantic decoder

`PacketAnalysisResult` を生成します。

まず DHx 系を対象とします。

### Phase 3: LE semantic decoder

LE 1M / 2M RF Test Packet から開始します。

### Phase 4: source adapters

```text
VSG GenerationResult -> PacketDecodeInput
VSA decoded_bits      -> PacketDecodeInput
```

を実装します。

### Phase 5: common table formatter

Result Summary に近い table row model を作成します。

まだ VSA / VSG window への具体的な配置は行いません。

### Phase 6: UI integration

VSG / VSA それぞれの最適な表示場所、packet selection、window/panel layout を別途検討します。

### Phase 7: higher layer dissectors

必要に応じて L2CAP / ATT / SMP / AVDTP 等へ拡張します。

### Phase 8: non-Bluetooth protocol

同じ core model / table formatter を使用して Wi-Fi、ADS-B 等へ拡張します。

---

## 23. 推奨 MVP

最初の実装範囲としては、**BR/EDR DHx + LE RF Test Packet の field interpretation と table row generation** までを推奨します。

MVP の完成条件:

```text
1. VSG が生成した packet の air bits を Analyzer が直接 decode できる
2. VSA demod bits を同じ Analyzer が decode できる
3. 両者が同じ PacketAnalysisResult schema を返す
4. Header field の raw value と meaning が表示できる
5. HEC / CRC validity が表示できる
6. payload length と payload Hex preview が表示できる
7. partial / invalid packet でも可能な範囲の結果を返す
8. PacketAnalysisResult から共通 table rows を生成できる
```

UIへの組み込みはこのMVP完成後に検討します。

---

## 24. 今後の設計判断ポイント

実装開始時に最終決定する項目:

- `pluto_protocol` package 名
- raw bit array を NumPy immutable array とするか tuple / bytes とするか
- Access Code / preamble を semantic tree にどこまで含めるか
- VSA で unknown UAP / clock candidate search を自動実行する範囲
- BR/EDR whitening unknown 時の candidate ranking
- compact summary の default field
- payload hex dump の最大 preview length
- higher-layer decode をいつ導入するか
- packet auto-detection の必要性

これらは core result schema と独立して後から調整できるようにします。

---

## 25. まとめ

この機能は VSG / VSA のどちらかに属する機能ではなく、**Pluto RF toolset 共通の protocol interpretation layer** として作るのが適切です。

```text
              pluto_protocol
        ┌──────────┴──────────┐
        │                     │
      VSG                   VSA
  generated bits       demodulated bits
        │                     │
        └──── same result ────┘
```

中心となる設計原則は以下です。

- IQ処理とprotocol解釈を分離する
- VSG / VSAで同じair-order bits入力契約を使う
- VSG設定値ではなく実際に生成されたbit列を解析する
- raw value と semantic meaning を両方保持する
- malformed / partial packet も best-effort で解析する
- result は階層構造をauthoritativeとする
- table表示は共通formatterから生成する
- UI配置は後から決める
- Bluetooth固有情報をcore modelへ埋め込まない
- 将来Wi-Fi / ADS-B等へ同じframeworkを拡張できるようにする

この構造にしておけば、最初は Bluetooth の「bit列を読めるpacket viewer」から始めながら、将来的には複数protocolを同じ表示体系で扱う汎用 packet analyzer へ拡張できます。
