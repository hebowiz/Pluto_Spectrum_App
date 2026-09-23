# VSAアプリケーション設計方針

> 参照範囲: session・record・解析段階の共通概念と拡張方針を担当します。現在の実装と将来構想を節ごとに区別します。操作の入口は [右側操作UI設計](VSA_UI.md)、位置・サイズ・内部配置の要件は [共通ウィンドウ仕様](../../spec/common/window-layout.md)、主題ごとの参照先は [設計索引](README.md) にあります。

本文改訂: 2026-09-23（現行実装と初期構想の区分）

参照モデル: `FPL_K70_VSA_UserManual_en_12.pdf`（R&S FPL1-K70 VSA User Manual、551 pages）

現行の操作・制約は [ユーザーマニュアル](../../user-manual/Pluto_VSA_User_Manual_JA.md)、確認した実装との差分は [照合記録](../../verification/vsa/README.md) を参照してください。[初期実装メモ](../../work-notes/vsa-implementation.md) は当時の進行記録です。

## 1. 基本判断

VSAは現行Spectrum Analyzerの単純な追加モードではなく、同じrepository内の別application shellとして実装します。取得、IQ record、trigger、calibration、共通plot部品は共有し、VSA session、demodulation、result model、multi-window UIはVSA側で所有します。

統合VSAは単一`PlutoAnalysisWindow`内でworkspace全体を切り替えます。現在の選択肢はGeneral VSA、Bluetooth、DECT、ADS-B 1090ESです。Pluto接続・所有権は外枠に1つだけ置き、取得・解析中はモード切替を禁止します。Wi-Fi専用workspaceとSCPI instrument sourceは拡張構想であり、この選択肢には含みません。

R&S FPL VSAの用語、設定順序、result分類をUXの参照モデルとします。ただしPluto、SCPI instrument、保存IQという異なるsourceを同じ解析器へ接続できるよう、hardware固有設定は`IQSource` adapterへ分離します。

### 1.1 General VSAの対象信号と専用モード

R&S VSAの全機能・全standard presetの再現は目標にしません。現在のmodulation定義は [model.py](../../../pluto_vsa/model.py) で管理します。

- FSK family: Signal Description上は`FSK`へ統一し、Gaussian等のpulse shapingはTX Filterで表す。現行のbinary FSKに加えて将来の多値FSKへ拡張できる共通demodulator contractとする。
- PSK family: BPSK、QPSK、OQPSK、pi/4-DQPSK、8DPSK、pi/4-QPSK、8PSK。
- 想定用途: DECT、Bluetooth BR/EDRの観測、復調、symbol/packet解析。
- QAM family: 16QAMを実装済み。同じsymbol/result contractを使いますが、振幅を保持した判定・同期を行い、差動PSKには変換しません。任意のQAM次数への対応を意味しません。

DECT/Bluetoothは固定値をDSPへ埋め込まず、symbol rate、modulation、mapping、BT/Alpha、preamble/sync word、packet structure等をまとめた`AnalysisProfile`として実装します。profile値は対象規格・modeごとに定義し、manual設定で上書きできるようにします。

Bluetooth専用解析ではEDRやHDTの区間別解析を実装しています。このため、1 captureまたは1 result rangeにつきmodulationは1種類、という制約をarchitectureへ持ち込みません。

## 2. R&Sから採用する主要モデル

### 2.0 操作モデルと現行UI

R&S FPL-K70の用語と処理順を参照しますが、全メニューの互換再現は目標にしません。現在は右側の共通パネルから設定ページを直接開き、SWEEP CONTROLで取得・再解析を操作します。Config Top / Overviewを経由する初期案は通常操作には使いません。メニュー一覧は [右側操作UI設計](VSA_UI.md) に集約します。

設定はWindow Modal dialogで編集し、設定変更だけではCapture / Analysisを開始しません。新規取得はSingle / Continuous、保持recordの再解析はRefresh Analysisで明示的に要求します。Display設定と測定条件、Plotのzoom範囲と測定範囲を分離します。

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
- run state、status、warning

これはsessionの概念的な責務です。UIの配置・復元は外枠とworkspaceが管理します（§10）。複数sessionを切り替えるtab / session listは拡張方針であり、現在のAnalyzer Modeによるworkspace切替とは別です。

### 2.2 Overviewの設定順序

R&SのOverviewは信号処理順に重要設定を並べています（manual pp.158-161）。VSAの解析条件も次の依存関係で整理します。

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

これは解析上の依存関係を示す順序です。UIは右パネルから各設定dialogを直接開きます。条件の変更を次の取得・再解析に反映し、dialogを閉じるだけでは再計算を開始しません。

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
├─ ScpiInstrumentSource（将来構想、統合VSAの実機入力ではない）
└─ FileIQSource
       ↓
IQRecording / IQAcquisitionRecord
```

以下はsourceを拡張する際の概念的なcontractです。全sourceが同じ名前のメソッドを実装しているという意味ではありません。

- `capabilities()`
- `configure()`
- `arm()` / `capture()` / `stop()`
- continuous block stream（対応sourceのみ）
- finite recording取得
- source metadataとstatus

共通recordにはIQ samples、datatype、sample rate、center frequency、usable IQ bandwidth、scale/unit、impedance、timestamp/sample index、calibration状態、overload/gap、trigger位置、source設定snapshotを保存します。

### 3.1 Pluto

現行`PlutoReceiver`、`IQBlock`、`IQAcquisitionRecord`、Power Trigger、Single Snapshotを再利用します。高sample rateではContinuousの無欠落を保証しないため、source capabilityへcontinuous/snapshot制約を明示します。

### 3.2 R&S等のSCPI instrument（将来構想）

transportと機種driverを分離します。汎用SCPI adapterへ機種別のcommand set、binary block parser、scaling、trigger/capture capabilityをpluginします。instrument側で取得済みのIQも同じ`IQRecording`へ変換し、解析DSPはsource機種を条件分岐しません。

### 3.3 保存IQ

R&SはI/Q file input時にcenter frequency、sample rate、measurement bandwidth等をfile metadataから固定します（manual pp.185-186）。本アプリもfile metadataを正本とし、欠落項目だけimport dialogで指定します。

現行の [FileIQSource](../../../pluto_vsa/sources.py) はR&S IQ-TAR、NPY、NPZを形式別に読み、それ以外をraw complex IQとして扱います。元fileは改変しません。SigMFを標準形式とする案は将来構想であり、SigMFメタデータの専用読込は実装されていません。対応拡張子・保存内容は [ファイル操作仕様](../../spec/vsa/general/vsa-file-workflows.md) を参照してください。

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

現在のmodulation familyはFSK、PSK、16QAMです。FSKは周波数偏移、modulation index、Gaussian BT、連続位相を設定可能にし、PSKはabsolute/differential mappingとphase ambiguityを明示的に扱います。16QAMも同じsymbol/reference/result contractを使い、振幅を保持した同期・判定を行います。

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

実装済みのPluto acquisition I/Q Power Trigger、post-capture Burst Search、Pattern Search gateは
[vsa-iq-power-trigger.md](vsa-iq-power-trigger.md)を参照。取得triggerはSingle / Continuousの各recordの位置を決め、Burst Searchは取得済みbuffer内の全power eventを列挙し、各active intervalの最初の有効patternをResult Range候補にする。両者は別contractとして維持する。

これらを同じ機能として扱いません。

### Acquisition Trigger

現行Pluto sourceはFree RunとI/Q Powerに対応します。I/Q PowerはLevel、Rising / Falling / Either、Hysteresis、符号付きTrigger Offsetを使い、固定長recordの位置を決めます。SingleとContinuousは同じ連続producerを利用し、recordごとにcursorを作ります。取得と再アームの寿命は [連続IQ取得設計](../acquisition/continuous-iq-acquisition.md) を参照してください。

Drop-Out / Holdoffは取得後のBurst Searchに属します。外部hardware triggerやinstrument側trigger capabilityは将来拡張です。Continuousが実装されたことを理由に、検索設定を取得triggerへ移しません。

### Post-capture Search

- Burst Search: power envelopeからburst候補を抽出。
- I/Q Pattern Search: modulationとTX filterから生成した既知waveformを、time/frequency仮説を変えてcorrelation検索。
- Pattern Symbol Check: 仮復調symbolとpatternを比較し、PSKのphase ambiguityも解消。
- Result Gating: burst/patternが見つかったrecordだけを表示・平均へ採用。

R&SもI/Q correlation thresholdでpattern候補を検出し、その後symbol一致を検査します（manual pp.113-120、205-214）。ユーザー向けにはPattern Triggerと表現できても、内部ではacquisition triggerではなくsearch/gating stageとして実装します。

## 9. Result model

R&SのEvaluation Data Source分類（manual pp.21-24）を参照した拡張モデルです。以下は設計上の分類・候補を含み、全項目が現在のDisplay設定で選べるという意味ではありません。

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

IQ Powerを含む全result blockを同格のDock Widgetとして扱う。初期workspaceは3列×2行の均等gridとし、上段をIQ Power / Spectrum / Result Summary、下段をModulation / Symbol Plot / Symbol Tableとする。PSKではModulationにIQ軌跡、Symbol PlotにConstellationを表示する。FSKではModulationにInstantaneous Frequency、Symbol Plotに1 symbol期間の位相差分を表示する。Result Summaryは単一行labelではなく測定項目を縦に列挙する独立result windowとする。

統合VSAの現行動作は次のとおりです。

- 既存Dockのdrag、tab化、別窓化に対応します。任意のwindow追加・close/duplicateは提供しません。
- メインウィンドウの位置・サイズは再起動時に復元し、最小サイズは960×640です。
- Dock配置・分割比率・フローティング状態はモード別に実行中だけ保存・復元します。再起動時は初期配置です。
- 選択タブは永続保存しません。リサイズ時の明示的な再均等化も行いません。
- 詳細と初期配分は [共通ウィンドウ仕様](../../spec/common/window-layout.md) を正本とします。

解析はwindowごとにraw IQから再実行せず、`VSAAnalysisSnapshot`の共有resultを各viewが購読します。Display Points/Symbolは描画設定、Estimation Points/Symbolは解析設定として分離します。

以下のPredefined Display Configurationは拡張候補です。現在の選択可能なpreset一覧として案内しません。

- Overview: Capture Power、Spectrum、Spectrogram、Vector I/Q
- Typical PSK: Constellation、Symbol Table、EVM vs Symbol、Result Summary
- Sync Debug: Capture、correlation、carrier/timing estimate、symbol decision
- Filter/Equalizer: Meas/Ref Spectrum、Error Spectrum、channel/equalizer response
- FSK Analysis: Instantaneous Frequency、FSK Eye、Symbol/Bit Table、Frequency Error
- Packet Overview: packet全体のPower/Frequency、segment境界、segment別summary、decoded fields

## 11. Demodulation / compensation properties（拡張方針）

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

## 12. 初期ロードマップの記録

この節は導入時の段階分けです。下記のPhaseや当時の「未実装」を現在の進捗一覧として使いません。現在は別entry point、PlutoのSingle / Continuous取得、Burst / Pattern Search、16QAM、Bluetooth / DECT / ADS-B専用workspaceが存在します。SCPI sourceは将来構想です。補正・測定精度の個別状況は [同期設計](vsa-carrier-synchronization.md)、[Bluetooth解析補足](bluetooth/bluetooth_dedicated_analysis_pipeline_ja.md) と各テストを参照してください。

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

Binary FSKでは設定でbitwise-complement patternも探索候補にできる。ただし候補生成とsymbol mappingは分離し、decisionは常に設定されたNatural mappingの物理周波数極性を保持する。PSKのambiguityはbit反転では一般化せず、将来のphase/conjugate/mapping仮説として別設計にする。

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

## 14. 対象範囲の限界

- 全R&S standard presetの再現。
- 16QAM以外のQAM全般、APSK、高度なmulti-carrier modulationの網羅的対応。
- hardware external trigger。
- R&S固有file/commandの全機種共通化。
- multi-channel/MIMO。
- RTSA overlap/POIとの統合。

これらは拡張可能な構造を保ちますが、実装済みの16QAMや専用packet解析まで対象外として扱いません。
