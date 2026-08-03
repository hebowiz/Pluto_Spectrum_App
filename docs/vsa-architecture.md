# VSAアプリケーション設計方針

最終更新: 2026-08-03

参照モデル: `FPL_K70_VSA_UserManual_en_12.pdf`（R&S FPL1-K70 VSA User Manual、551 pages）

実装済み範囲と既知の制約: [vsa-implementation.md](vsa-implementation.md)

## 1. 基本判断

VSAは現行Spectrum Analyzerの単純な追加モードではなく、同じrepository内の別application shellとして実装します。取得、IQ record、trigger、calibration、共通plot部品は共有し、VSA session、demodulation、result model、multi-window UIはVSA側で所有します。

R&S FPL VSAの用語、設定順序、result分類をUXの参照モデルとします。ただしPluto、SCPI instrument、保存IQという異なるsourceを同じ解析器へ接続できるよう、hardware固有設定は`IQSource` adapterへ分離します。

### 1.1 当面の対象信号

R&S VSAの全機能・全standard presetの再現は目標にしません。初期開発は次へ集中します。

- FSK family: 2-FSK、GFSKおよび連続位相FSKを扱える共通demodulator contract。
- PSK family: BPSK、QPSK、OQPSK、差動PSK、pi/4-DQPSK、8DPSK。
- 想定用途: DECT、Bluetooth BR/EDRの観測、復調、symbol/packet解析。
- 将来拡張: QAM familyを同じsymbol/result contractへ追加。

DECT/Bluetoothは固定値をDSPへ埋め込まず、symbol rate、modulation、mapping、BT/Alpha、preamble/sync word、packet structure等をまとめた`AnalysisProfile`として実装します。profile値は対象規格・modeごとに定義し、manual設定で上書きできるようにします。

Bluetooth EDRのように1 packet内で変調方式が切り替わる信号を最終到達点とします。このため、1 captureまたは1 result rangeにつきmodulationは1種類、という制約をarchitectureへ持ち込みません。

## 2. R&Sから採用する主要モデル

### 2.0 操作互換性の目標

VSAの標準UIはR&S FPL-K70に慣れた利用者が説明なしでも辿れる使用感を目標とします。単なる配色や外観の模倣ではなく、menu名、設定の所在、操作順、run state、window追加方法、測定範囲の意味を優先して合わせます。

基本操作は次の対応を目標にします。

| R&Sの操作概念 | 本VSAでの扱い |
|---|---|
| MODE > VSA | VSA専用entry point、またはSAから`Open VSA Workspace` |
| Measurement Channel | `VSASession` tab |
| Channel bar | 常時表示するSession summary bar |
| Meas Config > Overview | 処理順に並ぶVSA Overview |
| Signal Description | 同名のmodulation/signal structure設定 |
| Input/Frontend | Source選択とsource固有設定 |
| Signal Capture | Capture Length、Oversampling、usable IQ BW、Trigger |
| Burst/Pattern Search | 同名のpost-capture search設定 |
| Result Range / Evaluation Range | 同じ三段階range model |
| Demod/Meas Filter | Demodulation、Equalizer、Measurement Filter |
| Display Config | Signal Sourceをgridへ追加する画面 |
| Window Config | Result Type、Transformation、Points/Symbol |
| Run Continuous / Run Single | 同じrun stateとbutton semantics |
| Refresh | captureを更新せず現在recordだけ再解析 |
| Auto Level / Auto Scale | source capabilityに応じて同じ位置へ配置 |
| Standard files | Measurement ProfileのLoad/Save |

`Meas Config`はmain result workspace内の常設dockではなく、menuから開く独立したWindow Modal dialogとする。設定中は背後のplot、dock、menu操作を受け付けず、測定設定と結果window操作が同時進行して不整合になることを防ぐ。dialog内のRefresh/Apply後も、dialogを閉じるまではmain workspaceをmodal blockする。

Session summary barにはmanual pp.19-20を参考に、少なくともRef Level、Capture Length、Profile/Modulation、Result Length、Center、Symbol Rate、TX Filter、Input、Burst、Pattern、Equalizer、Single状態を表示します。sourceがfileの場合など変更不能な項目は非表示にせず、値を表示したままdisabledとし、metadata由来であることを示します。

右側の測定器風menuは現行SAの操作感を継承しつつ、R&Sに近い入口名へ整理します。

- Frequency
- Amplitude
- Input / Frontend
- Meas Config
- Trigger
- Sweep / Run
- Trace
- Marker
- Display Config
- Auto Set
- Save / Recall

独自機能は標準workflowへ混在させず、原則として`Extensions`、追加のResult Type、追加Source、追加Profileのいずれかへ登録します。R&S互換に近い基本画面を維持したまま機能を増やせるplugin pointを用意します。

### 2.1 Measurement Channel / VSA Session

R&SはVSA applicationを開くと独立したmeasurement channelを作り、同じapplicationを異なる設定で複数channelとして保持します（manual pp.17-20）。本アプリではこれを`VSASession`として実装します。

1 sessionは次を所有します。

- input sourceとsource capability
- immutable IQ capture/recording
- signal description
- capture、trigger、burst/pattern設定
- demodulation、filter、equalizer設定
- result rangeとevaluation range
- analysis result snapshot
- window layout
- run state、status、warning

初期段階は1 sessionを実装し、data contractを複数session対応にしておきます。後からtabまたはsession listを追加できる構造にします。

### 2.2 Overviewの設定順序

R&SのOverviewは信号処理順に重要設定を並べています（manual pp.158-161）。VSAの主設定画面も次の順序にします。

1. Signal Description
2. Input / Frontend
3. Signal Capture / Trigger
4. Burst / Pattern Search
5. Result Range
6. Demodulation / Equalizer
7. Measurement Filter
8. Evaluation Range
9. Display Configuration
10. Analysis

すべてを同時に平坦なmenuへ置かず、Overviewから各設定dialogへ遷移できるようにします。設定変更時は依存する下流stageだけをinvalid化して再計算します。

### 2.3 三段階の測定範囲

```text
Capture Buffer
  └─ Result Range
       └─ Evaluation Range
```

- Capture Buffer: sourceから取得またはfileから読んだ位相連続IQの正本。
- Result Range: capture、burst、またはpattern waveformへalignし、指定symbol数を切り出す解析record。
- Evaluation Range: Result Rangeの一部または全部。EVM、MER、phase/magnitude error、power等を集計する範囲。

複数変調packetでは、Result Range内に複数の`ModulationSegment`を持ちます。各segmentはsample範囲、変調設定、同期条件、reference、evaluation rangeを個別に所有し、packet全体の時間軸とsample indexは共通に保ちます。

```text
Capture Buffer
  └─ Result Range / Packet
       ├─ Modulation Segment 0: FSK settings + Evaluation Range
       ├─ Modulation Segment 1: PSK settings + Evaluation Range
       └─ Packet-level decoded fields / status
```

R&SのResult Rangeはcapture/burst/patternへのreference、alignment、offset、symbol numberを持ち（manual pp.215-217）、Evaluation Rangeはsymbol start/stopを持ちます（manual pp.227-228）。この区別を採用し、画面のzoom範囲と測定範囲を混同しません。

## 3. Input Source

```text
IQSource
├─ PlutoLiveSource
├─ ScpiInstrumentSource
└─ FileIQSource
       ↓
IQRecording / IQAcquisitionRecord
```

共通source contractは概ね次を提供します。

- `capabilities()`
- `configure()`
- `arm()` / `capture()` / `stop()`
- continuous block stream（対応sourceのみ）
- finite recording取得
- source metadataとstatus

共通recordにはIQ samples、datatype、sample rate、center frequency、usable IQ bandwidth、scale/unit、impedance、timestamp/sample index、calibration状態、overload/gap、trigger位置、source設定snapshotを保存します。

### 3.1 Pluto

現行`PlutoReceiver`、`IQBlock`、`IQAcquisitionRecord`、Power Trigger、Single Snapshotを再利用します。高sample rateではContinuousの無欠落を保証しないため、source capabilityへcontinuous/snapshot制約を明示します。

### 3.2 R&S等のSCPI instrument

transportと機種driverを分離します。汎用SCPI adapterへ機種別のcommand set、binary block parser、scaling、trigger/capture capabilityをpluginします。instrument側で取得済みのIQも同じ`IQRecording`へ変換し、解析DSPはsource機種を条件分岐しません。

### 3.3 保存IQ

R&SはI/Q file input時にcenter frequency、sample rate、measurement bandwidth等をfile metadataから固定します（manual pp.185-186）。本アプリもfile metadataを正本とし、欠落項目だけimport dialogで指定します。

標準保存形式の第一候補はSigMFです。加えてraw `cf32/ci16`、NumPy、R&S `.iq.tar`等をimport adapterで扱います。元fileは改変せず、必要なら内部recordへ変換します。

## 4. Signal Description

R&SのSignal Description（manual pp.164-181）に合わせ、次を独立設定にします。

- Signal Type: Continuous / Burst
- Modulation familyとorder
- Symbol Rate
- Symbol Mapping
- TX Filter Type
- Alpha / BT
- signal structure、burst length、gap
- pattern name、symbols、offset
- frame/subframe structure（将来）
- known data / PRBS（将来）

最初の対応modulation familyはFSKとPSKです。FSKは周波数偏移、modulation index、Gaussian BT、連続位相を設定可能にし、PSKはabsolute/differential mappingとphase ambiguityを明示的に扱います。QAMは同じsymbol/reference/result contractへ将来追加します。

単一変調の`SignalDescription`に加え、複数のdescriptionと時間区間を束ねる`CompositeSignalDescription`を定義します。規格profileはpacket detector、segment boundary、各segmentのSignal Description、既知pattern、field decoderを提供します。

## 5. Sample Rateとbandwidth

source sample rateとanalysis sample rateを区別します。

```text
Source Fs
  → channel selection / resampling
Analysis Fs = Symbol Rate × Capture Oversampling
  → VSA demodulation
```

R&SはSample Rate設定をsamples/symbol（Capture Oversampling）として扱い、usable IQ bandwidthを別表示します（manual pp.69-78、199-200）。同じUI概念を採用します。

- Source Fs: Pluto、instrument、fileが提供する実sample rate。
- Capture Oversampling: 2/4/8/16/32/64/128 samples per symbol等。
- Analysis Fs: resampler後のrate。
- Usable IQ BW: source/front-endで有効な帯域。
- Demodulation BW: channel/measurement filter後に評価する帯域。
- Display Points/Symbol: 表示密度であり推定点数とは別。
- Estimation Points/Symbol: synchronization parameter推定へ使う点数。

### 5.1 Manual analysis channel selection

通常のsingle-channel VSAと同様に、capture内の全信号を自動復調対象にはしません。
ユーザーが`Analysis Center`と`Analysis Bandwidth`を指定し、対象信号をDDC、
complex FIR low-pass、integer decimationでbasebandへ切り出してから共通解析へ渡します。

```text
Wideband IQ recording
  → user-selected Analysis CenterへDDC
  → Analysis Bandwidthのcomplex LPF
  → 約4 × Analysis Bandwidthを目安にdecimation
  → FSK / PSK / profile demodulation
```

このstageはBluetooth固有ではなく、すべてのVSA modulation familyとinput sourceで
共用します。複数channelが見える場合はfilter OFFのSpectrumで探索し、対象周波数を
手動設定して解析します。全channel自動channelizerやhopping追従は、必要になった場合の
追加機能とし、固定周波数test signalの復調を妨げないよう当面の必須要件から外します。

## 6. Demodulation pipeline

R&Sの処理順（manual pp.112-124）を参照し、次のstageへ分けます。

```text
Capture Buffer
  → integrity / overload check
  → burst search
  → I/Q pattern waveform search
  → result range / packet extraction
  → packet structure detection
  → ModulationSegment[] creation
      → frequency shift / resampling
      → family-specific synchronization and demodulation
      → symbol decisions
      → pattern symbol check / ambiguity resolution
      → ideal reference generation
      → measurement filtering (Meas and Ref)
      → fine synchronization
      → optional equalizer
      → segment error/result calculation
  → decoded fields and packet-level result composition
```

各stageの入力・出力・設定・statusを型として分離し、中間結果をpytestで検証できるようにします。

family-specific stageでは共通の入力/output contractを使い、FSKはinstantaneous frequency、frequency/timing recovery、frequency decisionを中心に処理し、PSKはcarrier phase/timing recovery、complex symbol decisionを中心に処理します。表示側は両者を共通のsymbol table、decoded bits、error trace、summaryとして扱えます。

segment boundaryは段階的に、manual指定、known patternからの相対位置、profile detectorによる自動判定へ拡張します。境界付近のfilter transientを評価範囲へ含めるかどうかもsegment metadataへ残します。

## 7. TX/RX/Measurement/Reference Filter

R&Sと同じく役割を分離します（manual pp.71-75、225-226）。

- TX Filter: DUTが使用した送信filterのモデル。
- RX/ISI Filter: symbol decision用。TX filterと組み合わせてISI-free pointを作る内部filter。
- Measurement Filter: measurement signalとreference signalの両方へ適用し、error/EVMの帯域重みも決める。
- Reference Filter:原則として`TX Filter * Measurement Filter`。

初期対応はNone、RC、RRC、Gaussian、user-defined coefficientsです。Alpha/BT、filter span、normalization、group delay、settlingをmetadata化します。Measurement FilterをOFFにした結果とONにした結果は別measurement conditionとして扱います。

## 8. Trigger、Burst Search、Pattern Search

これらを同じ機能として扱いません。

### Acquisition Trigger

- Free Run
- Power Trigger
- pre/post-trigger offset
- rising/falling slope
- hysteresis
- dropout time
- holdoff
- Auto / Normal / Single

Power Triggerはcapture開始位置を決める前段機能です。Plutoではhost software trigger、対応instrumentではinstrument trigger capabilityを利用します。

### Post-capture Search

- Burst Search: power envelopeからburst候補を抽出。
- I/Q Pattern Search: modulationとTX filterから生成した既知waveformを、time/frequency仮説を変えてcorrelation検索。
- Pattern Symbol Check: 仮復調symbolとpatternを比較し、PSKのphase ambiguityも解消。
- Result Gating: burst/patternが見つかったrecordだけを表示・平均へ採用。

R&SもI/Q correlation thresholdでpattern候補を検出し、その後symbol一致を検査します（manual pp.113-120、205-214）。ユーザー向けにはPattern Triggerと表現できても、内部ではacquisition triggerではなくsearch/gating stageとして実装します。

## 9. Result model

R&SのEvaluation Data Source分類（manual pp.21-24）を採用します。

- Capture Buffer
- Measurement & Reference Signal
- Symbols
- Error Vector
- Modulation Errors
- Modulation Accuracy
- Equalizer
- Multi Source

各windowはまずSignal Sourceを選び、次に対応するResult Typeを選びます。代表resultは次のとおりです。

- Magnitude absolute/relative
- Phase wrapped/unwrapped
- Real/Imag I/Q
- Spectrum
- Spectrogram（本アプリ拡張）
- Constellation I/Q
- Vector I/Q
- Eye I / Eye Q
- Symbol table（binary/decimal/hex）
- EVM / MER
- Magnitude Error / Phase Error
- Carrier Frequency Error / Symbol Rate Error
- Result Summary
- Equalizer impulse/frequency response、group delay
- histogram/statistics

測定signal、reference、errorを同じwindowへ重ねられるMulti Source表示も用意します。

## 10. Multi-window UI

R&SはSignal Sourceを配置した後、windowごとにResult TypeとNormal/Spectrum/Statistics transformationを選び、最大16 result windowsを同時表示します（manual pp.247-251）。本アプリではQt dock widgetを基本とします。

- gridへ追加
- tab化
- dragによる配置変更
- detachして独立OS window化
- close/duplicate
- session全体のlayout保存・復元
- window固有scale、unit、trace、marker

解析はwindowごとにraw IQから再実行せず、`VSAAnalysisSnapshot`の共有resultを各viewが購読します。Display Points/Symbolは描画設定、Estimation Points/Symbolは解析設定として分離します。

Predefined Display Configurationとして最低限次を用意します。

- Overview: Capture Power、Spectrum、Spectrogram、Vector I/Q
- Typical PSK: Constellation、Symbol Table、EVM vs Symbol、Result Summary
- Sync Debug: Capture、correlation、carrier/timing estimate、symbol decision
- Filter/Equalizer: Meas/Ref Spectrum、Error Spectrum、channel/equalizer response
- FSK Analysis: Instantaneous Frequency、FSK Eye、Symbol/Bit Table、Frequency Error
- Packet Overview: packet全体のPower/Frequency、segment境界、segment別summary、decoded fields

## 11. Demodulation / compensation properties

R&Sの設定（manual pp.217-224）を参照し、段階的に次を扱います。

- compensate I/Q offset
- I/Q gain imbalance
- quadrature error / I/Q skew
- amplitude droop
- carrier frequency and phase error
- symbol rate error
- channel compensation
- EVM normalization: max/mean reference、max/mean constellation power
- optimization: minimize RMS error / minimize EVM
- coarse sync: detected data / pattern
- fine sync: detected data / known data / pattern
- bit ordering: MSB / LSB first
- phase rotation / PSK ambiguity
- equalizer: Off / Normal / Tracking / Freeze / User / Averaging
- equalizer length、reset、save/load

補正ONの値だけを出さず、可能な範囲で推定されたraw impairmentと、どの補正をEVMから除外したかをresult metadataへ残します。

## 12. 実装段階

### Phase 0: 分離準備

- VSA packageと別entry pointを作成。
- source、record、session、settings、result contractを定義。
- `CompositeSignalDescription`と`ModulationSegment`のcontractをこの段階で定義。
- HighSpeed TAからVSAへ再利用する取得処理をUIから分離。

### Phase 1: Offline FSK/PSK VSA

- generated/file IQ source。
- Capture/Result/Evaluation Range。
- Zero Span、Spectrum、Spectrogram、Vector/Constellation。
- 2-FSK/GFSK、BPSK/QPSKと差動PSK、Gaussian/RRC、manual symbol rate。
- instantaneous frequency、symbol/bit table、PSK basic EVM、FSK error metrics。

### Phase 2: Synchronizationとpattern

- carrier/timing recovery（pattern-based timing/CFO/phase推定は実装、symbol-rate error追従は未実装）。
- I/Q waveform correlation（FSK/GFSK/PSK/DPSKの任意symbol patternを実装）。
- pattern symbol checkとphase ambiguity解消（実装）。
- burst search、result gating（pattern result gatingのみ実装）。
- DECT/Bluetooth向け`AnalysisProfile`の基礎。

汎用Pattern Searchはprotocol decoderより下位の共通機能とする。Bluetooth Access Code、DECT sync word、将来のEDR sync blockはいずれも`KnownPattern` presetとして利用できるが、検索結果はprotocol fieldへ固定せず、R&Sと同様にResult Rangeのsymbol/vectorデータとして公開する。

### Phase 3: Live source

- Pluto finite snapshotとcontinuous record。
- Power Trigger、pre/post-trigger。
- analysis workerとUIの非同期化。

### Phase 4: Instrument source

- SCPI transport。
- R&S model driverとbinary IQ scaling。
- instrument-side capture/trigger capability。

### Phase 5: Measurement accuracy

- Meas/Ref/error filter chain。
- EVM normalization variants。
- compensation、equalizer、limit check、statistics。
- 既知vectorと実機によるcross-validation。

### Phase 6: Composite / packet analysis

- `CompositeSignalDescription`と`ModulationSegment[]`。
- manualおよびprofile-driven segment boundary。
- Bluetooth EDRを想定したFSK/PSK区間の一括解析。
- segment別同期・復調結果とpacket-level decoded fieldの統合表示。
- capture全体、segment別、packet全体の測定結果を同じsession snapshotへ保持。

## 13. 検証方針

- 理想symbol列からIQ waveformを生成し、TX/RX filter、timing/carrier error、AWGN、IQ imbalance、droop、multipathを個別注入する。
- FSKはfrequency offset/deviation、BT、modulation index、連続位相、symbol timing errorを個別に注入する。
- 各stageの推定誤差、decoded symbol、EVMをpytestで固定する。
- FSK→PSKの合成waveformを生成し、segment境界、各blockのdecoded bits、sample index対応が保たれることを固定testにする。
- source adapterごとに同じrecordを入力し、解析結果が一致することを確認する。
- R&Sから同じIQ dataと設定で得たResult Summary、symbol table、EVM traceと比較する。
- 補正、filter、normalization、evaluation rangeを一致させずにEVM値だけを比較しない。

## 14. 当面の対象外

- 全R&S standard presetの再現。
- QAM/APSKと高度なmulti-carrier modulationの初期実装。
- hardware external trigger。
- R&S固有file/commandの全機種共通化。
- multi-channel/MIMO。
- RTSA overlap/POIとの統合。

これらを後から追加できるcontractにはしますが、最初のFSK/PSK VSA完成を妨げないよう段階化します。
