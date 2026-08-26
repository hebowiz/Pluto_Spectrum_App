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

## 13. ADALM-Pluto finite transmit backend（2026-08-24、2026-08-25改訂）

`Output > ADALM-Pluto Settings...`で接続URI、TX RF bandwidth、TX hardware gainを設定し、
`Transmit with ADALM-Pluto`（`Ctrl+T`）で現在の生成IQを直接送信する。中心周波数、sample
rate、packet repetition countはProjectを正本とする。Pluto固有設定はVSA/SAの受信設定や
Project JSONへ混在させず、Pluto VSG専用`QSettings`へ保存する。

- Projectの`Repeat Count`は1～1000 packetに制限する。
- packetごとのpre/post idleとpower rampを含む有限IQ列をengineで生成する。
- Plutoへはnormalized IQを`2^14-1` scaleへ変換して渡す。
- 2026-08-25の実機検証では、短い非cyclic bufferを送信すると、1 packetのDH1（既定設定で
  3056 sample / 8 MS/s = 382 us）の予定に対して約660 msの間欠出力が観測され、FSKも正常に
  復調できなかった。一方、同じgeneratorがexportしたDH1/2DH1 WVはSMCV100Bから正常送信
  でき、offline HEC/CRC testも通る。このためgeneratorではなく、stock Pluto/libiioの短い
  non-cyclic DMA転送とbuffer teardownの境界を主要因と判断した。
- 直接送信は`[Lead-in zero guard][生成済み有限IQ列][Stop zero guard]`という1周期の
  superframeへ変更する。既定guardは10 ms / 100 msで、`tx_cyclic_buffer=True`によりDMA
  underflowを避ける。packet repetitionは従来どおり生成済みIQ列内に1～1000 packetを配置し、
  cyclic周回でrepeat countを作らない。
- backendはStop guardへ入ってから、TX gainを最小値`-89.75 dB`へ下げ、bufferを破棄し、
  TX channelを無効化してDAC zero sourceへ切り替える。この一連の停止をsuperframeが先頭へ
  周回する前に完了させる。
- 接続、転送、guard待機、buffer破棄はGUI thread外で実行する。送信中もGUIのStop操作を
  受け付ける。Stopまたはwindow closeは同じmute/teardown sequenceを実行する。
- Lead-in/Stop guardはPluto VSG専用Output Settings/QSettingsで調整でき、Projectの波形内容、
  SA/VSAの設定とは共有しない。
- 2台構成ではIIOの一時的なUSB URIではなくhardware serialを個体IDとして選択・保存する。
  Output Settingsは`serial:<id>`を保持し、送信開始時に現在のdirect USB URIへ解決する。同じ
  個体がUSB/IPの両contextで見える場合は1台へまとめてUSBを優先する。選択個体が不在なら
  他のPlutoへfallbackせず送信を中止する。VSA側のRX個体選択とは独立して保存する。
- TX hardware gain初期値は`-30 dB`。これはPlutoの相対hardware settingであり、dBm校正値では
  ない。絶対出力levelは外部ATTを含め実測し、将来のOutput calibrationで管理する。

現時点ではsoftware timingによる有限送信であり、hardware trigger、時刻指定、連続ユーザー
波形送信、R&S VSGへのSCPI転送は未実装。zero IQは変調成分を止めるがRF LO leakageを物理的に
遮断するものではない。unit testはPluto deviceをmockし、LO/sample rate/RF bandwidth/gain、
guarded cyclic superframe、DAC scale、停止前要求、mute、buffer cleanup、DAC zero source切替を
検証する。実機ではスペアナまたは十分なATTを介したloopbackで、packet数、level、packet間隔、
先頭への再周回がないことを確認する。

### 13.1 先行異常バースト再調査とTDDN移行方針（2026-08-26）

guarded cyclic superframe版を実機再検証した結果、送信操作から約2秒後に、指定Gainへ追従しない
約650 msの強い異常バーストが先行し、その後に指定packetが出力された。現行backendを再確認
すると、`transfer()`はdevice転送ではなくhost上でsuperframeを作るだけであり、実際のPluto
buffer転送は`start()`内の`tx(superframe)`で初めて行われる。その直前にTX channel、sample
rate、LO、RF bandwidth、指定Gainを有効化している。したがって、TX chainを開いた状態で大きな
cyclic bufferをUSB転送している時間が先行異常バーストの観測区間と整合する。約650 msが
`tx()`のblocking transfer時間そのものかは、API区間timestampとRF観測を対応させる次回試験で
確定する。

ADI公式のPluto TDDN例は、firmware v0.39以降、pyadi-iio v0.18以降を前提に、cyclic TX bufferを
software syncより先に転送する。generic AXI TDD coreはsoftware sync、frame length、startup
delay、有限`burst_count`を持ち、Pluto v0.39ではTDD channel 2がTX DMA専用sync portへ接続
されている。従って、1 frameを`packet + packet interval/zero guard`とし、1 frame分だけをcyclic
bufferへ保持したまま、TDDNのframe countで反復数を制御する構成が可能性の高い本命である。

ただしPlutoのTDDN channel 2で公式に確認できるのはTX DMA開始同期であり、RF port/VCOの完全な
物理muteとは区別する。移行時は次の順序を採る。

1. 接続直後にTX Gainを`-89.75 dB`へ設定し、TDDNをdisableする。
2. mute中にsample rate、LO、RF bandwidthを設定する。sample rateはframe timingとbufferの前提
   なので、文字どおり全RF設定より先にbufferを転送する順序にはしない。
3. TDDNをdisableしたままframe length、TX DMA sync channel、有限burst countを設定する。
4. 1 frame分のcyclic IQ bufferをPlutoへ転送する。この時点ではTX DMAをsoftware sync待ちにする。
5. 設定値をread-backし、最後に指定Gainを適用してTDDNをarmする。
6. `sync_soft=1`だけを送信開始操作とし、指定frame数の完了をTDD stateで監視する。
7. 完了またはCancel時は、Gain最小化、TDD disable、buffer破棄、DAC zero sourceの順で停止する。

generic coreのburst counterは合成時のbit幅に依存する。pyadi-iioの`tddn` API自体には旧`tdd`
APIの255 frame制限は記述されていないため、1000 packetをhost IQへ複製せず実行できる可能性が
ある。実装時はIIO device `iio-axi-tdd-0`の存在、firmware version、burst counter幅または1000の
write/read-backをruntime capabilityとして確認し、非対応個体では安全に送信を拒否する。

次回実機試験では、各attribute write、`tx()`開始/完了、Gain read-back、software sync、TDD終了を
monotonic timestampで記録する。Repeat 1を基準に、(a) sync前の変調出力、(b) LO leakage、
(c) packet先頭欠落、(d) packet数、(e) Gain追従、(f) buffer再周回の有無を確認してから1000 frameへ
拡張する。

### 13.2 Pluto firmware v0.39更新記録（2026-08-26）

TDDN実機検証の前提を整えるため、接続中のADALM-Pluto Rev.CをADI公式firmware `v0.32`から
`v0.39`へ更新した。対象serialは`1044730c370e00100400120023338fb325`である。公式GitHub releaseの
`plutosdr-fw-v0.39.zip`を取得し、ZIPのSHA-256
`6542A4A9AAFE7D51239DC896B93E1C56C301FAB526E8A78472F7F13B6F27DFEC`と、`pluto.frm`の
SHA-256 `6506786F9055282ACF031A2AF057E563DEB6BDB4ABED6820DDCC67845312E69F`を記録した。
さらに`pluto.frm`末尾の内蔵MD5
`c83fa190266f616029b0c449f379626f`が、それより前のfirmware本体から再計算したMD5と一致することを
確認した。

更新はPluto mass-storage volumeへ検証済み`pluto.frm`だけをコピーし、コピー元・コピー先の
SHA-256一致を確認してからWindowsの安全な取り外しを実行した。`boot.frm`と`uboot-env.dfu`は
使用していない。再接続後の実機read-back結果は次のとおり。

- firmware: `v0.39`
- kernel: `6.1.0-gf3da30df6004`
- hardware model: `Analog Devices PlutoSDR Rev.C (Z7010-AD9364)`
- AD936x model: `ad9364`（AD9364化設定を維持）
- XO correction: `39999996`（更新前後で一致）
- serial: `1044730c370e00100400120023338fb325`（更新前後で一致）
- TDD IIO control device: `adi-iio-fakedev`
- pyadi-iio `adi.tddn`接続: 成功、3 channel、初期`enable=False`

このfirmwareではTDD coreが必ずしも`iio-axi-tdd-0`というdevice名で公開されず、pyadi-iioからは
`adi-iio-fakedev`をcontrol deviceとして解決する。従ってruntime capability判定は固定device名だけに
依存せず、`adi.tddn`の生成成功、`burst_count`、`frame_length_ms`、`sync_soft`、3 channelの存在を
確認する。

### 13.3 2台目Pluto firmware v0.39更新記録（2026-08-26）

2台構成のもう一方のADALM-Pluto Rev.Cも、公式firmware `v0.32`から`v0.39`へ更新した。対象serialは
`10447318ac0f00050a001600356a18eee6`である。13.2節と同じ公式配布物を再取得し、ZIP SHA-256、
`pluto.frm` SHA-256、内蔵MD5のすべてが13.2節の記録値と一致することを確認した。更新方法も同様に
検証済み`pluto.frm`だけをmass-storage volumeへコピーし、コピー元・コピー先のSHA-256一致後に
安全な取り外しを実行した。`boot.frm`と`uboot-env.dfu`は使用していない。

再接続後の実機read-back結果は次のとおり。

- firmware: `v0.39`
- kernel: `6.1.0-gf3da30df6004`
- hardware model: `Analog Devices PlutoSDR Rev.C (Z7010-AD9364)`
- AD936x model: `ad9364`（AD9364化設定を維持）
- XO correction: `40000061`（更新前後で一致）
- serial: `10447318ac0f00050a001600356a18eee6`（更新前後で一致）
- TDD IIO control device: `adi-iio-fakedev`
- pyadi-iio `adi.tddn`接続: 成功、3 channel、初期`enable=False`

### 13.4 TDDN有限バースト送信の実装（2026-08-26）

13節のguarded cyclic superframe方式は、実機で確認された先行異常バーストを解消できなかったため、
Pluto firmware v0.39のTDDNを使う方式へ置き換えた。新しいbackendはfirmware version、
`adi.tddn`生成、3 channel以上、frame lengthおよびburst countのread-backをruntimeで検証し、
要件を満たさない個体ではTX Gainを最小値にしたまま送信を拒否する。

送信sequenceは次のとおりである。

1. 接続直後にTX Gainを`-89.75 dB`へ下げる。
2. TDDNをdisableし、mute中にsample rate、LO、RF bandwidthを設定する。
3. TDDNのstartup delay、frame length、有限burst countを設定する。channel 2をPluto v0.39の
   TX DMA sync channelとして`on_raw=0`、`off_raw=10`、`polarity=0`で有効化する。
4. TDDNをarmしてsoftware sync待ちにしてから、1 frame分だけをcyclic TX bufferへ転送する。
5. buffer転送完了後に初めて指定TX Gainを適用し、`sync_soft=1`で有限送信を開始する。
6. 指定frame数の経過後、またはユーザーCancel時に、Gain最小化、TDD disable、buffer破棄、
   TX channel無効化、DAC zero source切替の順でcleanupする。

Project/Preview/WV exportでは従来どおり`Repeat Count`回を連結したIQを保持するが、Plutoへ転送する
際は完全に同一なrepeatへ分割できることを確認し、先頭の1 frameだけを転送する。1～1000回の
反復はTDDNの`burst_count`で実行するため、repeat数に比例してUSB転送量やcyclic bufferが増えない。
repeat間で異なるIQが含まれる場合は、異なるframeを誤って同一反復することを避けるため送信を拒否
する。

Output Settingsの旧`Lead-in Zero Guard`は`TDD Startup Delay`、旧`Stop Zero Guard`は
`Completion Timeout Margin`として表示を変更した。既存QSettingsとの互換性のため内部key名は
維持している。TX APIの主要境界はmonotonic timestampのevent logへ記録し、次回のスペアナ実機
検証でsoftware sync前の出力、packet先頭、packet数、Gain追従と停止時刻をRF観測と照合できる。

mock unit testでは、1 frameだけが転送されること、TDD channel 2とburst count、
`mute -> buffer transfer -> requested gain -> software sync`の順序、転送中Cancel、旧firmware拒否、
cleanupを検証した。serial指定時は優先transportが開けなくても、同じserialを示す別transportだけを
試し、別個体へはfallbackしない。VSG関連testは81件すべて合格した。実機ではRF出力を開始せずcapability probeのみ
行い、接続可能なv0.39個体で3 TDD channel、初期`enable=False`、初期`burst_count=0`を確認した。
実RF検証が完了するまでは、本方式を安全sequenceの実装完了・送信波形未確認として扱う。

初版ではPluto本体と`adi.tddn`が同一USB URIへ別々のlibiio contextを開いていたため、Windowsの
direct USB接続で2個目のcontextが`No device found`となった。IP transportでは再現しないがUSB 2台の
両方で再現したため、TDDN wrapperをPluto本体が既に開いたIIO context上へ構築する方式へ変更した。
修正後は両方のdirect USB URIで、同じcontextからFW v0.39、3 TDD channel、`enable=False`、
`burst_count=0`をread-only確認できた。

### 13.5 TDD channel配線の訂正とactive-low TX window（2026-08-26）

TDDN版の再実機観測でも、指定packetとは異なる約430 msの波形がbuffer転送時に先行した。
`PHASER_ENABLE`と公式例のchannel設定を追加しても波形が変化しなかったため、Pluto v0.39が基にする
ADI HDL `hdl_2023_r2/projects/pluto/system_bd.tcl`まで追跡した。その結果、pyadi-iio公式例の
`TX DMA SYNC`というコメントを、DMA transfer start gateと解釈したことが誤りだった。

Plutoの実際の配線は次のとおりである。

- channel 0: 外部`txdata_o`端子。内部RF/DMA gateではない。
- channel 1: RX DMAの`fifo_wr_sync`。
- channel 2: `logic_or(axi_ad9361/rst, channel 2)`を介した`tx_upack/reset`。AXI-DMACの
  transfer start入力ではない。

したがって、公式例のchannel 2設定（`polarity=0, on_raw=0, off_raw=10`）はframe先頭で
`tx_upack`へ短いreset pulseを与えてTX/RXを整列させるが、software sync前のcyclic buffer転送を
遮断しない。`PHASER_ENABLE`もPhaser用外部配線を有効化するGPIOであり、内部TX gateではない。

修正版ではchannel 0/1を無効のままとし、channel 2をactive-low transmission windowとして使う。
channel 2は`polarity=1`によりTDD ARMED状態で`tx_upack/reset`をassertし、buffer transfer中の
upack出力を停止する。software sync後のframe counter 0でLowへ遷移してresetを解除し、frame末尾で
Highへ戻して再びresetする。また、外部入力による意図しない開始を防ぐため`sync_external=False`、
`sync_internal=False`、`sync_reset=False`とし、開始源を`sync_soft`だけに限定する。

TX GainはRF/TDD設定後かつbuffer転送直前にも`-89.75 dB`を再適用してread-backする。cleanupでは
channel 2のreset保持とGain muteを維持したままbuffer破棄とDAC zero source切替を先に行い、その後
TDD channelをneutral値へ戻す。mock testを含むbackend test 16件が合格した。active-low windowで
先行異常バーストが消え、packet先頭が欠落しないことは次回実機試験で確認する。

### 13.6 Pluto TX timing diagnostic log（2026-08-26）

active-low channel 2へ変更後も、送信前の不明バーストは形状が若干変化しただけで残存した。
そこでホスト側の推測だけでTDD設定を変更し続けず、異常RF出力とAPI処理の時間関係を実測する。
VSGは各送信試行をリポジトリ直下の`pluto_vsg_tx_trace.log`へJSON Lines形式で記録する。
このログはGit管理外であり、次の境界時刻をミリ秒単位で保存する。

- RF設定完了
- TDD arm完了
- cyclic buffer転送開始・完了
- 指定TX Gain適用
- software sync
- finite burst終了およびcleanup

各段階でTX Gain、LO、sample rate、RF bandwidth、TDD state、channel 2の極性・ON/OFF値も
可能な範囲でread-backする。RTSAの時間波形と対応付け、異常RF出力がbuffer push中、
software sync前、または同期後のどこで生じるかを切り分けてから次のgate方式を決定する。

最初の診断結果では、3168 sample（0.396 ms）のcyclic buffer転送はログ分解能上0 msで完了し、
channel 2はbuffer転送前後ともARMED、`polarity=1`、`on_raw=0`、`off_raw=3167`を保持した。
buffer転送からsoftware syncまでも16 msであり、約430 msの先行バーストとは一致しない。一方、
Pluto接続開始から最初のGain muteまで約2969 ms、mute後のRF設定に625 msを要した。RTSAで観測した
異常バースト終了から本来の送信までの間隔とも概ね整合するため、準備中のRF出力が主因と判断した。

Gain `-89.75 dB`だけをmuteとして使う設計を改め、AD9361 TX LOの`powerdown`属性をhard muteとして
追加した。接続直後にTX LOをOFFとし、RF設定、TDD arm、buffer preloadをLO OFFのまま実行する。
指定Gainを設定した後にTX LOを起動し、software syncで有限送信を開始する。cleanupではGain muteに
加えてTX LOを先にOFFへ戻す。以後の送信では前回cleanupのLO OFF状態もハードウェアに保持される。

TX LO hard mute適用後の実機ログでは、準備中のLO powerdown read-backは一貫してtrueだったが、
先行バーストは残った。詳細時刻を追加した結果、buffer transferは15 msである一方、既に
7,999,999 S/sで動作中のPlutoへ`sample_rate=8,000,000`を再設定する処理だけが625 msを占めた。
この長さは先行バーストと概ね一致する。AD9361のsample rate変更はキャリブレーションを伴うため、
同値の再書込みが不要な内部TX動作を起こしている可能性が高い。

sample rate、TX LO周波数、RF bandwidthは、read-backと要求値が実用上同じ場合には書き直さない。
8 MS/sに対する7,999,999 S/s、2440 MHzに対する2 Hz差などは同値として扱う。設定が本当に変わる
場合だけpropertyを書き込み、診断ログにも`*_configured`または`*_unchanged`を区別して残す。

不要な再設定を避けた実機試験では大きなキャリブレーション波形は消えたが、0.396 ms packet後に
約40 msの一定レベル出力が残った。これはTX LO ONからcleanupまでの47 msと一致する。stock Plutoの
TDD channel 2は`tx_upack/reset`でありRF gateではないため、frame終了後も指定GainのTX LO leakageが
残る。channel 0もAD9361 ENABLE pinではなくPhaser向け外部PL_GPIO0へ接続されている。したがって
stock v0.39のTDDNだけではAD9361 RF出力をsub-msで直接ON/OFFできない。

ホスト制御で残留時間を最小化するため、LO起動・安定待ちはGain `-89.75 dB`のまま行う。安定後に
指定Gainを適用して直ちにsoftware syncし、packet durationだけ`perf_counter`ベースで高精度待機して
即cleanupへ進む。RF unmute中のIIO read-backも廃止した。WindowsのEvent/Sleepによる約15.6 ms単位の
overshootは避けられるが、cleanupのUSB Gain/LO書込み遅延は残る。完全なsub-ms RF gateには、TDD
出力をAD9361 ENABLEへ接続するcustom HDL/firmwareまたは外部RF switchが必要になる。

### 13.7 5 ms観測によるcyclic反復確定とnon-cyclic TXへの転換（2026-08-26）

RTSAの時間軸を5 msへ拡大し、VSAでも同時観測した結果、1 packet指定に対して完全なpacketが
4回、その後に途中までのpacketが1回送信されていた。3168 sample / 8 MS/s = 0.396 msの
短いcyclic bufferが、USB cleanup完了まで周回していた結果と一致する。先頭の不完全burstは、
TDD reset解除時点のDMA read pointerがbuffer先頭に揃わないことによるものと判断した。

stock Pluto HDLではTDD channel 2は`tx_upack/reset`へ接続され、TX DMAのread pointer、有限転送回数、
AD9361 RF gateを直接制御しない。このためTDD software syncとburst countによる有限packet送信案を
廃止した。標準HDLでTDD出力だけを使い、buffer preload後にRFを指定回数だけ正確に送信することは
できない。必要ならTDD出力をDMA syncまたはAD9361 ENABLEへ接続するcustom HDLが必要である。

標準Pluto向けbackendは`tx_cyclic_buffer=False`へ変更した。指定回数分のIQは従来どおりgeneration
結果に含め、次の単一DMA bufferとして一度だけpushする。

1. TX LO powerdownとGain -89.75 dBでRF設定を行う。
2. LOを起動し、Gain最小のまま設定されたsettling timeを待つ。
3. 指定Gainを適用する。
4. `短いzero prefix + 指定回数分のIQ + trailing zero guard`をnon-cyclic bufferとしてpushする。
5. buffer全長相当時間を待ち、Gain mute、LO powerdown、buffer破棄、DAC zero source切替の順で停止する。

`tx()`はbuffer転送と再生開始を分離できないため、TDD案のような「mute中にpreloadし、その後software
trigger」は行わない。`tx()`呼び出し自体をsoftware start eventとして扱う。zero prefixはDAC source
切替の過渡がpacket先頭へ重なることを避け、trailing zero guardは最終変調sampleの保持を防ぐ。
non-cyclic DMAなのでhost cleanupの遅延がpacket反復回数を増やすことはない。実機ではpacket数、先頭
欠落、末尾の余分なpacket、LO leakageを改めて確認する。

初回実機確認ではpacketが全く観測されなかった。診断ログ上はLO ON、指定Gain、non-cyclic `tx()`
まで成功していた。使用中のpyadi-iio/libiio v1 compatibility layerを確認すると、最初の`tx(data)`は
stream blockを取得してwriteするだけであり、そのblockは次のstream advanceで初めてenqueueされる。
連続streamingでは次回`tx()`が暗黙にadvanceするが、one-shotでは次回呼び出しがないため、未送信の
blockをcleanupで破棄していた。non-cyclic push直後にTX streamを1回明示的にadvanceし、最初のblockを
DMAへcommitする処理を追加した。libiio v0系では`_tx_stream`が存在しないため、この追加処理は行わない。

再試験でもpacketは観測されず、実行環境を直接確認した結果はlibiio `0.24` / pyadi-iio `0.0.20`で
あった。この環境はlibiio v0 compatibility pathを使うため、上記stream commitは実行されず、原因でも
なかった。診断ログでは0.396 ms packetに対して100 ms trailing zeroを含む819,168 sampleのbufferを
作成し、non-cyclic `Buffer.push()`が約140 msを要していた。one-shot DMAとして過大であり、USB転送・
再生開始条件を不安定にするため、DMA内部のprefix/suffixは各2 msへ固定した。既存Completion Marginは
DMA sampleへ追加せず、push後にcleanupを遅延するhost waitとして扱う。1 packet時のDMA bufferは
約4.396 msまで短縮される。

### 13.8 non-cyclic有限送信の実機確定（2026-08-26）

TDD実験後にTX DMAが無音となった個体では、TDD engineをdisableし、全channelをneutralへ戻す
公式例相当のcleanupを実行しても送信は復旧しなかった。物理的にPlutoの電源を再投入し、以後TDDへ
一切アクセスしない状態にするとTX DMAが復旧した。従ってTDDのread-backが`enable=False`、
`state=0`であっても、有限TXを安全に実行できる状態を保証しない。通常のPluto VSG backendはTDD
deviceを生成・参照・変更せず、過去にTDD実験を行った個体は送信前に電源再投入することを運用条件と
する。

復旧後の実行環境はlibiio `0.24`（commit `c4498c2`）、pyadi-iio `0.0.20`であった。このlibiio
0.x経路のnon-cyclic `sdr.tx(data)`は`Buffer.write()`後に`Buffer.push()`を実行するため、1回の
呼出しだけで有限DMA転送が開始される。libiio 1.x用の明示stream advance処理は、`_tx_stream`が
存在する場合だけ互換処理として実行する。診断reportには両library version、non-cyclic DMA mode、
TDD非使用policyを記録する。

実機では次の2段階で確認した。

1. cyclic bufferへ500 ms toneを設定した基準試験では、観測時間も正確に500 msだった。
2. non-cyclic bufferへ`20 ms zero + 10 ms tone + 20 ms zero + 20 ms tone + 30 ms zero`を格納して
   1回だけ送信した結果、RTSAには10 msと20 msのtoneが指定間隔で各1回だけ観測された。先頭欠落、
   末尾の途中packet、cyclic再周回はなかった。

さらにVSGの短packet相当bufferでも、指定した有限burstだけが1回観測された。以上によりstock
Plutoの正確なpacket回数制御には、送信対象の全packetを1本のnon-cyclic bufferへ格納して1回push
する方式を採用する。TDDNによるpreload/trigger/finite-count方式は採用しない。RF mute精度とLO
leakageはDMA回数制御とは別課題であり、必要ならcustom HDLまたは外部RF switchで扱う。

Repeat Countが複数の場合も、送信およびfile export用の生成結果には指定packet数をすべて保持する。
一方、GUI Previewは描画負荷をpacket数に比例させないため、先頭の1 packet intervalだけを表示する。
IQ Waveform、IQ Power、Instantaneous Frequency、Spectrum、Constellationおよびfield guideは同じ
先頭intervalを参照し、Packet Endも1本だけ表示する。この表示制限は送信packet数やexport内容を
変更しない。

## 14. Bluetooth settings / preview補正（2026-08-25）

- 新規Bluetooth BR/EDR ProjectのWhitening初期値はOFFとする。既存Projectは保存済みの
  `whitening_enabled`を維持し、規格波形が必要な場合はユーザーがONを選択する。
- Packet TypeごとのPayload最大長はDH1=27 byte、2-DH1=54 byte、3-DH1=83 byteである。
  Settings dialogは保存値をspin boxへ設定する前に全体上限83 byteを確保し、その後選択packet
  typeの上限へ絞る。これにより2-DH1/3-DH1の27 byte超の値がdialog再表示時にclampされない。
- generation metadataへrepeatごとの`packet_ranges_samples`を格納する。IQ Waveform、IQ Power、
  Instantaneous Frequencyの各Previewは全field startに加え、packet data最終sampleの直後へ
  `Packet End` guideを表示する。Power RampやPost Idleの終端ではなく、論理packet境界を示す。

## 15. LE 1M / LE 2M RF Test Packet（2026-08-25）

Visual Packet Composerへ進む前の規格profileとして、編集可能なLE 1M/LE 2M packet
generatorを追加した。Preamble、Access Address/Sync、Header、Payload、CRC、Whiteningを
同じSettingsで編集できる。Direct Test ModeのRF Test Packetは別形式にせず、Core規定値を
これらの編集欄へロードするpresetとした。BR/EDRにも同じ考え方でRF test payload presetを
追加した。

生成結果は既存PreviewおよびNPZ/IQ TAR/WV/Pluto Outputの共通経路を通る。次のComposer
段階ではpreset適用後のfield graphも通常packetと同じblockとして見え、ユーザーが内容を確認・
変更できることを維持する。
