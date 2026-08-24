# Pluto VSG: R&S VSG / WinIQSIM2 UI調査

更新日: 2026-08-24

## 1. 目的と参照範囲

R&S SMCV100Bのローカルマニュアルと、R&S公式のWinIQSIM2マニュアル、仕様書、Bluetooth関連ドキュメントを参照し、Pluto VSGでパケットを設計する際に必要な設定項目とUI階層を整理する。

参照資料:

- `docs/SMCV100B_UserManual_en_10.pdf`
  - Custom Digital Modulation: おおむねp.106–123
  - ARB / waveform再生と転送: おおむねp.138–160
- R&S WinIQSIM2 User Manual（公式オンライン版）
- R&S WinIQSIM2 Specifications（公式オンライン版）
- R&S Bluetooth BR/EDR / LE waveform generation manuals（公式オンライン版）

R&Sの画面を複製することが目的ではない。設定の責務分離、標準プリセット、依存項目の表示制御、Data/Control List Editor、グラフィック表示をPluto VSGのVisual Composerへ取り込むための調査である。

## 2. R&S SMCV100Bの基本構造

### 2.1 Custom Digital Modulation

主要アクセスパスは次の構成である。

```text
Baseband
└─ Custom Digital Mod
   ├─ General
   ├─ Trigger
   ├─ Marker
   ├─ Clock
   ├─ Data Source
   ├─ Modulation
   └─ Filter
```

`Power Ramp Control`はGeneralから開く下位設定である。Custom Digital Modulationはリアルタイム生成系であり、設定変更が出力へ直接反映される。一方、WinIQSIM2で事前計算したIQはARBへ転送して再生する。

### 2.2 General

- State
- Set To Default
- Save / Recall
- Set according to Standard
- Symbol Rate
- Coding
- Power Ramp Control

標準を選択すると変調方式、symbol rate、filter、codingが一括設定される。その後いずれかを変更すると設定種別は`User`になる。これはPluto VSGでも採用する。

### 2.3 Data Source

- All 0 / All 1
- PRBSおよびPRBS type
- Pattern（SMCV100Bでは最大64 bit）
- Data List (`*.dm_iqd`)
- Data Listの新規作成、編集、ファイル選択
- Control List (`*.dm_iqc`)
- Connector Settings

Data List Editorの主な操作:

- BIN / HEX表示
- cursor position、list length、Go To
- 範囲選択、Copy / Cut / Paste
- Replace / Insert
- Save / Save As

Pluto VSGでは固定データ、PRBS、ファイル、計算フィールド、CRC/HEC、whiteningをField Blockごとに選択できるよう拡張する。

### 2.4 Modulation

- Modulation Type
- theoretical constellation
- User Mapping (`*.vam`)
- ASK Depth
- FSK Deviation
- Angle Alpha
- Variable FSK（4/8/16FSK）とシンボル別deviation
- APSKのGamma / Gamma1
- Modulation CW switching

設定可能項目はmodulation typeに応じて切り替わる。常に全入力欄を並べるのではなく、選択方式に有効な設定だけをInspectorに表示する。

### 2.5 Filter

- Filter Type
- Roll Off FactorまたはB×T
- Cut Off Frequency Factor
- Gaussian / Lowpass用cutoff
- Raised Cosine用bandwidth
- User Filter (`*.vaf`)

WinIQSIM2側ではoversamplingのAuto/Manualも波形計算条件として扱われる。Pluto VSGでは`Samples per Symbol`、sample rate、filter span/group delay、出力帯域見積りを同じページで確認可能にする。

### 2.6 Power Ramp ControlとControl List

- Enable / State
- Source（Control List）
- Ramp Function: Linear / Cosine
- Ramp Time [symbols]
- Rise Delay [symbols]
- Fall Delay [symbols]
- Lev_Att時のattenuation
- In Baseband Only

Control Data EditorではMarker、CW、Hop、Burst Gate、Lev_Attを色分けされたレーンとして編集する。Ramp Up/Down等のpreset、cursor、transition position/state table、zoom、saveを備える。

SMCV100Bのpower rampにはsymbol rate上限などの実機制約があるが、これはR&S Backendのcapabilityとして扱い、Waveform Engine全体の制約にはしない。

### 2.7 Trigger / Marker / Clock

共通baseband設定として次を持つ。

- Trigger source: internal / external
- Trigger mode: Auto、Single、Retrigger、Armed Auto、Armed Retrigger
- Trigger delay / inhibit
- Marker signal / mapping / connector
- Clock source: internal / external

Pluto VSGでは外部triggerが未対応でも、model上は`TriggerPolicy`として保持し、Backend capabilityによってUIの有効/無効を切り替える。

## 3. WinIQSIM2で追加される設計概念

WinIQSIM2のCustom Digital ModulationはSMCV100Bと概ね同じ分類を持つが、オフライン波形生成のため次が追加される。

- Generate Waveform File
- Sequence Length
- Samples / Sample Rate
- Oversampling Auto / Manual
- 生成波形の機器転送
- IQ import、AWGN、multicarrier、multisegment waveform
- 最大3個のconfigurable graphic
  - I/Q vs Time
  - Magnitude / Phase vs Time
  - Constellation
  - Spectrum
  - Eye Diagram
  - CCDF

Pluto VSGではgraphic数を3個へ制限せず、Dock Widgetとして追加・削除、配置保存を可能にする。previewは設定値から描くのではなく、生成済みIQを再解析した結果を表示する。

## 4. Bluetooth packet generationで必要な設定

### 4.1 BR / EDR

- Bluetooth version / profile preset
- Transport Mode
- Packet Type
- Sequence Length / packet count
- Slot Timing / packet interval
- Packet Configuration / Packet Editor
  - BD_ADDRからAccess Code生成
  - header各bit
  - SEQN toggle
  - HEC自動計算
  - payload length / payload data source
  - payload CRC自動計算
  - whitening
- BR GFSK parameters
  - symbol rate
  - modulation index / deviation
  - Gaussian filter / B×T
- EDR parameters
  - π/4-DQPSK / 8DPSK
  - mapping / bit ordering
  - root raised cosine filter / roll-off
  - guard、sync、trailer、modulation transition
- Power Ramp
- Dirty Transmitter / impairments
  - carrier frequency offset
  - drift rate / drift deviation
  - start phase
  - timing error
  - modulation index error

### 4.2 Bluetooth LE

- PHY: LE 1M / LE 2M / LE Coded
- RF channel / advertising or data channel
- Access Address
- preamble polarity derived from Access Address
- CRCInit
- whitening enable / channel index
- PDU / payload
- coding and pattern mapper for LE Coded
- packet interval / event sequence
- power ramp and impairments

Bluetooth専用画面は汎用変調器の設定値を裏で上書きする別エンジンにしない。Packet TemplateがField Block、Modulation Segment、Filter、Power Envelope、Control Trackを生成する構造にする。

## 5. Pluto VSGへ採用するメニュー構造

```text
File
├─ New Project
├─ Open / Recent
├─ Save / Save As
├─ Export IQ / Export Waveform Package
└─ Exit

Edit
├─ Undo / Redo
├─ Cut / Copy / Paste
├─ Duplicate / Delete
└─ Preferences

Waveform
├─ Profile / Standard
├─ Packet Composer
├─ Data Sources and Lists
├─ Modulation Profiles
├─ TX / Measurement Filters
├─ Power Envelope and Control Tracks
├─ Impairments / Dirty Transmitter
└─ Recording Layout / Sequence

Graphics
├─ Add Graphic
│  ├─ IQ / Magnitude / Phase / Power
│  ├─ Instantaneous Frequency
│  ├─ Spectrum / Time-Frequency
│  ├─ Constellation / Eye
│  └─ CCDF
├─ Remove Graphic
├─ Link Time Axes
└─ Save / Restore Layout

Output
├─ Offline / File
├─ Device Manager
├─ ADALM-Pluto
├─ R&S VSG
├─ RF Frequency / Level / Calibration
├─ Transfer / Generate
└─ Start / Stop

Tools
├─ Validate Project
├─ Inspect Generated IQ
├─ Device Capabilities
└─ Calibration
```

## 6. Composer内の設定階層

R&Sのタブ分類は右側Inspectorへ取り込み、中央はGNU Radio風の直列Field Block Composerとする。

1. Project / General
   - Standard/User、sequence length、sample rate、SPS、metadata
2. Packet / Frame
   - transport、packet type、slot timing、field sequence
3. Selected Field / Data Source
   - fixed、pattern、PRBS、file、computed、CRC、whitening
4. Modulation
   - type、mapping、coding、symbol rate、deviation/phase alphabet
5. Filter
   - type、B×T、roll-off、cutoff、span、custom coefficients
6. Power Envelope / Control Track
   - level、ramp shape/time/offset、gate、idle/guard level
7. Impairments
   - CFO、drift、timing、phase、IQ imbalance、noise
8. Sequence / Output
   - repetition、period、trigger、marker、output device、RF level

Field Blockを選択すると該当Inspectorへ移動し、上位Packet/Project設定はbreadcrumbから開く。R&Sの「タブ名に現在値を表示する」考え方を取り入れ、例えば`Filter: RRC α=0.4`のように現在値を要約する。

## 7. UI動作ルール

### 7.1 Standard presetとUser化

- standard presetは関連設定を一括投入する。
- preset値から編集された場合は`Standard (Modified)`または`User`と表示する。
- 変更項目とstandard値の差分を表示し、preset再適用時の上書き範囲を確認可能にする。

### 7.2 Context-sensitive settings

- modulation/filter/packet typeに無関係な項目は非表示またはdisabledにする。
- Backend固有制約はWaveform Engineの制約と分離する。
- 無効な組合せは画面内に理由を表示し、曖昧な状態で生成しない。
- sample count、sample rate、duration、bandwidth、memory estimateを常時表示する。

### 7.3 Data / Control editor

R&SのList Editorを参考に、次を採用する。

- BIN / HEX / symbol表示
- field boundaryとsymbol index
- Insert / Replace、範囲選択、Go To、Copy/Cut/Paste
- CRC/HEC/whitening等のcomputed fieldはlock表示し、入力依存関係を示す
- Control Trackは色分けレーン、draggable point、symbol/field snap、presetを持つ
- 平坦なdata/control fileだけを正本にせず、project内の構造化modelを正本とする

### 7.4 Graphics

- 設定変更はdebounceしてpreviewを再生成する。
- graphは生成IQ、field boundary、symbol point、power envelope、marker/control eventを同じ時刻基準で表示する。
- 選択Fieldへzoom、全体表示、X軸link、cursor/markerを提供する。
- 重い波形はworkerで計算し、GUI threadには表示用decimated dataだけを渡す。

## 8. R&Sから踏襲する点・変更する点

踏襲する点:

- General / Data / Modulation / Filter / Power / Triggerの責務分離
- standard presetとUser化
- context-sensitive parameter
- frame structureの可視化
- Data List / Control Listの編集思想
- 波形生成、graphics、機器転送が一つのworkflowにあること

Pluto VSG向けに変更する点:

- 汎用Custom Digital ModとBluetooth専用Packet Editorを同じComposer modelへ統合する。
- R&Sのbinary list中心ではなく、field/segment/control trackを正本にする。
- realtime generatorとARBを別アプリにせず、Output Backendの違いとして扱う。
- SMCV100B固有のsymbol rateやmemory制約を全Backendへ波及させない。
- 無効な組合せを警告だけで生成せず、validation errorとして明確化する。
- graphic数を固定せず、複数Dockとlayout保存へ対応する。

## 9. 実装優先順位

### MVP

1. Project / Standard preset
2. Field Block Composer
3. Fixed / PRBS / computed data source
4. GFSK、π/4-DQPSK、8DPSK
5. Gaussian / RRC filter
6. Power Envelope / guard / repetition
7. IQ Power、Spectrum、Instantaneous Frequency、Constellation preview
8. IQ file exportとPluto Backend

### 次段階

- Bluetooth Packet Editor、HEC/CRC/whitening
- Control Track / marker
- impairments / dirty transmitter
- eye / CCDF / time-frequency
- R&S transfer Backend
- multisegment / mixed modulation packet

## 10. 調査上の注意

- WinIQSIM2のstandalone PDFは現時点でリポジトリへ保存していない。公式オンライン版を参照した。
- R&SのBluetooth固有項目はoption/versionで差があるため、実装時は特定機種のUI名ではなく内部modelの意味を基準にする。
- 本資料はUI/設定構造の参考資料であり、Bluetooth規格値の一次資料ではない。規格依存値はBluetooth Core Specificationと照合する。

## 11. 実装状況（2026-08-24）

最初のvertical sliceとして、Bluetooth BR DH1を対象に次を実装した。

- `pluto_vsg.model`
  - device-independentなProject、Field、Modulation、Power Envelope、Bluetooth BR設定
  - payload source: Fixed / Pattern / PRBS-9
  - 設定値validation
- `pluto_vsg.engine.bluetooth_br`
  - Access Code、header、HEC、whitening、DH1 payload header、payload CRC
  - Gaussian GFSK、CFO、cosine/linear power ramp、pre/post idle、repeat
  - Access Code / Header / Payloadのsample境界metadata
  - Ramp開始をpacket先頭/末尾からの相対symbol位置で定義
  - packet外へRampが張り出す区間は先頭/最終symbolの周波数偏移を保持し、位相を連続化
- `pluto_vsg.persistence`
  - version付き`.pvsg.json`保存・復元
- `pluto_vsg.export`
  - VSAから読めるNPZ
  - R&S互換IQ TAR (`complex float32`)
  - R&S信号発生器のARBへ直接ロードするSMU-WV (`.wv`)
- `pluto_vsg.ui`
  - Block Library / Packet Composer / Inspector
  - Bluetooth BR / DH1設定dialog
  - I/Q time waveform、IQ Power、Instantaneous Frequency、Spectrum、Constellation preview
  - Project open/save、NPZ/IQ TAR export

現時点のComposerはBluetooth templateの構造を表示し、packet fieldの任意追加・削除・並べ替えはまだ無効である。これは、構造変更が生成結果へ反映されない見かけだけの編集UIを避けるためである。次段階でField Block modelを生成graphへ直接接続してから有効化する。

生成IQは既存VSAのBluetooth BR復調器へ戻すround-trip testを行い、DH1のHEC、payload length、payload CRCが成立することを確認する。Project JSON、NPZ、IQ TARにもread-back testを設ける。

次のvertical sliceとして、同じProject/Composer/Engineへ2-DH1と3-DH1を追加した。

- Packet TypeをDH1 / 2-DH1 / 3-DH1から選択
- packet typeごとのpayload上限（27 / 54 / 83 byte）をvalidationへ反映
- BR Access Code/HeaderからGuardを経てEDRへ連続する単一IQを生成
- 2-DH1: pi/4-DQPSK、3-DH1: 8DPSK（Bluetooth EDR mapping）
- SRRC roll-off、Guard長、GFSKに対するEDR相対powerを設定可能
- ComposerへGuard / EDR Sync / EDR Payload / EDR Trailer階層を表示
- Constellation previewはEDR区間のsymbol判定点を抽出して表示
- 既存VSAの検証済みEDR fixtureとdifferential phase列が一致することをtestで確認
- 生成IQを汎用VSAへ直接渡し、両packetでsync相関99%以上、symbol error 0、
  全244 symbol一致、差動EVM 5%未満となるoffline round-trip testを追加

次の優先項目は、ComposerのField Block編集を生成graphへ接続することと、Pluto Backendの
安全なcyclic送信を追加して実機loopback検証へ進むことである。

## 12. R&S WV export仕様（2026-08-24）

R&S SMCV100B User Manual revision 10、4.6.7「Tags for waveforms, data and
control lists」（pp.175–184）を一次資料として実装した。`.wv`はraw IQではなく、ASCII tag
headerとbinary IQを一つのfileに格納するARB waveform形式である。

- 先頭tagは`{TYPE: SMU-WV,<checksum>}`。
- `CLOCK`へsample rate [Hz]、`SAMPLES`へ複素sample数を格納する。
- `LEVEL OFFS`には量子化後IQのRMS/Peakが16-bit full scaleより何dB低いかを格納する。
- `EMPTYTAG`で`WAVEFORM` tag先頭をhex address `0x4000`へ揃える。
- `WAVEFORM-Length`は`1 + 4 * complex sample count`。先頭1 byteは`#`。
- binaryはI、Qの順に交互配置したsigned 16-bit two's complement、little-endian。
- checksumは`0xA50F74FF`を初期値とし、`#`直後から全binary dataをlittle-endian
  32-bit word単位でXORする。TYPE tagには結果を10進ASCIIで格納する。

export時に波形単独の再正規化は行わない。Project/engineが生成したnormalized complex IQを
32767 full scaleへ量子化し、実際に書き込んだ量子化IQから`LEVEL OFFS`を求める。これにより
field間の相対levelとpower envelopeを保持する。空、非finite、全zero、またはvector magnitude
が1.0を超えるIQは、暗黙のclippingやlevel metadata不整合を避けるためexport errorとする。

UIは`File > Export R&S WV...`から保存する。read-back regression testでは、0x4000 alignment、
tag値、checksum、I/Q ordering、量子化誤差、level offsetをbinaryから独立に再計算して検証する。
中心周波数やRF出力levelはbaseband WVの属性ではないため格納せず、R&S VSG側または将来の
Output Backendで設定する。marker/control listおよびmulti-segment WVは現時点では未実装。

## 13. ADALM-Pluto finite transmit backend（2026-08-24）

`Output > ADALM-Pluto Settings...`で接続URI、TX RF bandwidth、TX hardware gainを設定し、
`Transmit with ADALM-Pluto`（`Ctrl+T`）で現在の生成IQを直接送信する。中心周波数、sample
rate、packet repetition countはProjectを正本とする。Pluto固有設定はVSA/SAの受信設定や
Project JSONへ混在させず、Pluto VSG専用`QSettings`へ保存する。

- Projectの`Repeat Count`は1～1000 packetに制限する。
- packetごとのpre/post idleとpower rampを含む有限IQ列をengineで生成する。
- Plutoへはnormalized IQを`2^14-1` scaleへ変換して渡す。
- `tx_cyclic_buffer=False`固定とし、IQ列を一度だけ送信する。cyclic bufferによる意図しない
  無限RF出力は使用しない。
- 接続、転送、air time待機、buffer破棄はGUI thread外で実行する。送信中もGUIのStop操作を
  受け付ける。
- Stopまたはwindow closeはcancelを要求し、TX owner threadが`tx_destroy_buffer()`を実行する。
- TX hardware gain初期値は`-30 dB`。これはPlutoの相対hardware settingであり、dBm校正値では
  ない。絶対出力levelは外部ATTを含め実測し、将来のOutput calibrationで管理する。

現時点ではsoftware triggerによる有限送信であり、hardware trigger、時刻指定、連続cyclic
送信、R&S VSGへのSCPI転送は未実装。unit testはPluto deviceをmockし、LO/sample rate/RF
bandwidth/gain、非cyclic設定、DAC scale、停止前要求、buffer cleanupを検証する。実機では
スペアナまたは十分なATTを介したloopbackで、packet数、level、packet間隔を確認する。
