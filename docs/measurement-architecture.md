# リアルタイムSA・VSA準拠の計測アーキテクチャ

この文書は、一般的なリアルタイムSpectrum Analyzer（RTSA）とVector Signal Analyzer（VSA）の計測原理を、本アプリの設計判断へ落とし込むための基準資料です。将来のTrigger、VSA、記録・再解析機能はこのモデルに沿って追加します。

## 1. 計測器の基本モデル

### Swept Spectrum Analyzer

従来型SAはLOまたは解析帯域を周波数方向へ移動し、RBW filter出力をdetectorで代表値へ変換してtraceを作ります。広いSpanを測れる一方、異なる周波数点は異なる時刻の測定であり、短いeventを掃引間で見逃す可能性があります。

本アプリのSweep SAとWideBand RT SAはこの性質を持ちます。名称にRTを含むWideBandも、瞬時帯域を超える範囲では時間・位相連続なsnapshotではありません。

### Real-Time Spectrum Analyzer

RTSAは瞬時解析帯域内のADC/DDC後IQを連続処理し、overlapを持つFFT、detector、spectrogram、persistence、frequency mask判定を並列に進めます。Rohde & Schwarzの公開構成図でも、DDC後のデータがIQ memoryとFFT経路へ分岐し、表示処理とFMT trigger controlが並列です。

重要なのは画面更新速度ではなく、解析帯域内で入力sampleを途切れさせず、eventを監視できることです。100% POIとなる最短event時間はFFT長、Sample Rate、overlapに依存します。したがって、本アプリでもGUI FPSとFFT処理率、event POIを別の指標として扱います。

### Vector Signal Analyzer

VSAは位相を保持したIQ time recordを解析対象にします。一般的な処理は次の順序です。

1. RF front-end、zero-IFまたはIF downconversion、ADC
2. DDC、channel filter、resampling/decimation
3. trigger基準のtime record切出し
4. carrier frequency/phase recovery、symbol timing recovery
5. modulation固有の同期、equalization、reference symbol生成
6. constellation、EVM、frequency error、IQ imbalance、CCDF等の算出

解析・描画より先に連続time recordを保存することで、同じIQを異なる設定で再解析できます。Keysightの資料も、処理間にgapがあるblock-mode VSAと、処理前にgapのないwaveformを保存して再生解析するtime captureを区別しています。

## 1.1 実機メーカー資料から整理したZero SpanとVSAの違い

### Zero Span / Time Domain Power

一般的なZero SpanはLOを1周波数へ固定し、選択したRBWまたはchannel filterを通過した信号の電力を時間に対して表示します。基本経路は次のとおりで、各時間点にFFTは必須ではありません。

```text
RF/IFまたはDDC後IQ
    → RBW / channel filter
    → envelope or power (I² + Q²)
    → detector (sample / peak / RMS等)
    → optional VBW / averaging
    → time bucket / display trace
```

- RBWは測定対象帯域とnoise bandwidthを決め、filterのimpulse responseが立上がり時間と実効時間分解能を制限する。
- Sample Rateは内部時刻量子化を決めるが、独立した電力測定点の時間分解能が必ず`1 / Sample Rate`になるわけではない。
- Sweep Timeは観測record長、trace pointsは表示またはdetector bucket数であり、ADC/IQ sample数と同じとは限らない。
- 現代のRTSAではRBW/channel filter、power trigger、time traceをDDC後IQからデジタル実装できる。FFT/DPX/Frequency Maskは同じIQから分岐する別処理である。

Rohde & SchwarzはZero Spanを「固定したRBW filterでpower versus timeを表示する測定」と説明し、time-domain powerではRBWが測定帯域を決め、Gaussian RBWだけでなくchannel filterも使用できるとしています。TektronixのRTSA構成もADC→correction→DDC/decimation→IQを基点に、filtered power level trigger、FMT/DPX、capture、post-acquisition analysisへ分岐しています。

### VSA

VSAは電力包絡線ではなく、位相を保持したIQ time recordを正本にします。

```text
RF front end / LO / ADC
    → DDC / decimation / calibrated complex IQ
    → trigger + raw/search time record
    → frequency shift / resampling / measurement filter
    → burst/frame search
    → carrier and symbol timing lock
    → measured IQ and ideal reference IQ
    → EVM / magnitude error / phase error / constellation / spectrum
```

- Raw Main Time、Search Time、解析対象Time recordを区別する。
- Hardware取得点数はresampling filterのsettling分だけ解析recordより多い場合がある。
- Digital demodulationではsymbol rateとpoints/symbolへresampleし、carrier lock、symbol lock、IQ offset補償、measurement filterを適用する。
- 理想symbolからreference IQを再構成し、measured IQとの差をEVM等として評価する。
- Spectrum traceが必要な場合は選択windowをtime recordへ適用してFFTするが、demodulationの全処理が「短いFFT frameごとの電力値」ではない。
- Digital demodulationのResBWは独立したZero Span RBW knobとは異なり、time record長とwindow ENBWから決まる場合がある。

### 本アプリへの判断

HighSpeed TA、RTSA、VSAは共通IQ Producerとsample timelineを共有しつつ、解析分岐を分けます。

```text
continuous calibrated IQ
    ├─ raw magnitude → software Power Trigger
    ├─ Zero Span TA
    │    ├─ Fast Envelope: RF/acquisition BW → I²+Q² → detector/bucket
    │    └─ Filtered Power: digital measurement filter (known ENBW) → detector/bucket
    ├─ RTSA: overlap FFT → spectrum/density/Frequency Mask
    └─ VSA: trigger record → DDC/resample/filter/sync/demod/EVM
```

Plutoの`rx_rf_bandwidth`はfront-end/acquisition bandwidthとして使用できますが、filter shape、ENBW、settling、校正を定義しない限り、従来の測定器RBWと同一とは呼びません。HighSpeed TAにはFFT方式に加えてFast EnvelopeとFiltered Powerを設け、画面には全IQ sampleを直接渡さず、pixel/time bucket単位のPeak/RMS等で短いeventを保持する方針が妥当です。

## 2. 本アプリで採用する信号経路

```text
Pluto RF/ADC/DDC
    ↓ libiio / USB（単一Producer）
IQStreamBuffer（連続IQ、sample連番、epoch、overrun）
    ├─ Fast monitor path
    │    ├─ power/edge trigger
    │    └─ overlap FFT → frequency-mask trigger / density（将来）
    ├─ Circular prestore + poststore
    │    └─ immutable IQAcquisitionRecord
    ├─ RTSA display path
    │    └─ spectrum / detector / spectrogram / persistence
    └─ Analysis/replay path
         ├─ HighSpeed TA
         ├─ VSA demodulation
         └─ file recording / offline re-analysis
```

原則は「取得と解析を分離し、TriggerはGUIより前段でsample timeline上に置く」です。表示が遅れてもProducerとTrigger monitorを停止しません。

## 3. Triggerモデル

### Sample基準

Trigger位置はホスト時刻ではなく、`stream_id + sample_index`を正本とします。ホスト側でTrigger判定が遅れても、循環prestoreにsampleが残っていればevent以前のIQを正確に切り出せます。

```text
oldest                                              newest
──────── pre-trigger ────────┼──────── post-trigger ────────
                             ↑ trigger sample / time=0
```

完成recordには次を保存します。

- trigger source/type
- trigger sample indexとrecord内offset
- pre/post-trigger sample数
- 判定値、threshold、slope、hysteresis、minimum duration
- Center、Sample Rate、bandwidth、gain等の取得設定snapshot
- stream epochとdiscontinuity/overrun状態

不連続を含むrecordは正常recordとして解析しません。破棄するか、invalid理由付きで保存します。

### Trigger source

段階的に次を実装します。

1. Free Run / Immediate
2. IQ magnitudeによるPower Level（rising/falling/either、hysteresis）
3. minimum duration、holdoff、delay
4. Frequency Mask Trigger（overlap FFTを全frame評価）
5. 将来: density、runt、protocol/pattern trigger

External hardware triggerは当面の対象外です。Triggerはすべて、Plutoから連続受信したIQをホスト側で評価します。

### Run/rearm state

- Auto: timeoutまでeventがなければforced trigger
- Normal: 条件成立まで待機
- Single: 1 record完成後に停止
- Auto Rearm: poststore完成後、holdoffを経て再arm
- Stop on Trigger: 1 recordを保持して取得consumerを停止。ただしProducer停止とは分離する

Trigger判定とrecord生成は状態機械として実装し、UIボタンの分岐へ埋め込みません。

Power Level Triggerの初期実装は各complex IQ sampleのmagnitudeを明示的な`iq_full_scale`で正規化したdBFSを使用します。校正済みdBm triggerとは区別します。Pluto実機でraw code scaleを確認し、取得metadataへ保存したfull-scale値から判定を再現できるようにします。

## 4. RTSA表示処理

- FFT frameは連続sampleから生成し、必要なPOIに応じてoverlap率を設定します。
- RBWは解析window/FFT bin幅/デジタルfilterの関係を明示します。
- Spectrogramの1行へ複数FFTを集約する場合、Sample/Positive Peak/Negative Peak/Average等のdetectorを適用します。
- Persistenceはtrace holdではなく、frequency-amplitude cellの発生頻度を蓄積するdensity表示として将来分離します。
- producer overrun、consumer lag、FFT処理率、最大見逃しevent時間を計測指標にします。

### USB帯域を超えるSingle Snapshot（TA Free Run実装済み）

12 MSPS等、持続USB throughputを超えるsample rateはContinuous modeとして無欠落にできません。一方、HighSpeed TAのSingle Free Runでは1 record全体を単一のdevice/DMA bufferへ収め、取得完了後にUSB転送する有限長Snapshot経路を実装しました。capture後にoffline trigger searchするVSA用途へ再利用できます。

- `record samples = round(Time Span × Fs)`と同じ長さのbufferを先にarmする。Gaussian IQ Filterはrecord間stateを使わず、先頭sampleを定常初期値として解析する。
- warm-up 5 buffersと本取得1 bufferの計6回に制限したSingle専用Producerとし、取得後にoffline解析する。最大record長は4,194,304 samplesで、超過時は従来streamへfallbackする。
- 12 MSPSでは10 msが120,000 complex samples（wire上のI/Q int16で約480 kB、host complex64で約960 kB）、100 msが1,200,000 samples（約4.8 MB / 9.6 MB）となる。
- HighSpeed TAのTime Span下限は100 µs、上限は`4,194,304 / Fs`を基本とする。RBW上限を5 MHzへ広げ、既存の4×規則により16 MSPS（RBW 4 MHz）と20 MSPS（RBW 5 MHz）をUIから選択できる。
- 単一buffer内部の連続性、最大buffer長、再arm間のblind timeをcounter/PRBSまたは位相連続CWで実機検証する。配列長とhost sample indexだけを連続性の証明に使わない。
- host software triggerをリアルタイム評価するには全sampleの連続転送が必要である。USB帯域超過時の任意event pre/post-triggerを保証するには、Pluto側の循環buffer/FPGA trigger、またはcapture-then-searchが必要となる。

このSnapshot経路は現在TA Free RunのSingle record向けであり、Continuousの欠落を解消するものではありません。Power Triggerはevent監視に連続streamが必要なため対象外です。Sweep SAはLO pointごとの有限長bufferが連続なら掃引全体を連続転送する必要がなく、主な影響は各pointの転送待ちによるSweep Time増加です。

現在のPersistenceはtrace残光表現であり、一般的なRTSAのdensity/probability表示と同一ではありません。

## 5. VSA実装への要件

- 解析入力はUI表示配列ではなく`IQAcquisitionRecord`とする。
- acquisition設定と解析設定を分離し、保存IQを再解析可能にする。
- raw IQを可能な限り保持し、DC除去、周波数補正、resampling等は再現可能なprocessing stageとして記録する。
- trigger前後の位相連続性を保持する。
- time/frequency synchronizationの結果と品質指標をrecord metadataへ保存する。
- 最初の対象変調方式は別途決め、共通同期器へ過剰な方式依存を持ち込まない。

## 6. Pluto固有の制約

- Plutoはdirect-conversion AD9363系receiverであり、DC offset、LO leakage、IQ imbalanceを考慮する必要があります。
- USB 2.0/libiio転送が連続帯域の上限になります。
- host software triggerは判定latencyを持ちますが、prestore方式ならrecord内trigger位置には影響させずに済みます。
- アプリが付けるsample連番は、Pluto/libiio内部のdropを直接証明できません。既知の連続信号による相関検証が必要です。
- External hardware triggerは当面使用しない前提です。共通API、UI、検証項目にも含めません。将来必要になった場合はsoftware trigger sourceの一種として混在させず、transport/hardware capabilityとして別途設計します。

## 7. 実装状況

2026-08-02時点:

- 共通`IQStreamBuffer`、epoch、cursor、overrun検出: 実装済み
- exact-length `IQWindowAssembler`: 実装済み
- `TriggerConfig`、`TriggerEvent`、`AcquisitionMetadata`、`IQAcquisitionRecord`: data contract実装済み
- Power Level detector: 実装済み（rising/falling/either、hysteresis、minimum duration、holdoff）
- pre/poststore state machine: 実装済み（block境界、判定遅延、不連続、Single/Stop on Trigger）
- 共通Trigger acquisition controller: 実装済み（Free Run、Power Auto/Normal/Single、forced timeout、rearm）
- HighSpeed TAのtrigger-aware record consumer化: 実装済み
- HighSpeed TA Trigger UI: 基本設定を実装済み（Source、Mode、Level、Slope、Position、Auto Timeout）
- FMT detector: 未実装
- VSA demodulation: 未実装

## 参考資料

- [Rohde & Schwarz: Understanding Zero Span](https://www.rohde-schwarz.com/ca/knowledge-center/videos/understanding-zero-span_251220-1614806.html)
- [Rohde & Schwarz: Speeding up Spectrum Analyzer Measurements](https://scdn.rohde-schwarz.com/ur/pws/dl_downloads/dl_application/application_notes/1ef90_speeding_up_sa_measurements/1EF90_2e_Speeding_up_SA_Measurements.pdf)
- [Tektronix: Fundamentals of Real-Time Spectrum Analysis](https://download.tek.com/document/37W_17249_6_Fundamentals_of_RealTime_Spectrum_Analysis_0.pdf)
- [Keysight 89600 VSA: Understanding Time and Frequency Parameters](https://helpfiles.keysight.com/csg/89600B/Webhelp/Subsystems/gui/content/understandingtimeandfreqparameters.htm)
- [Keysight 89600 VSA: IQ Meas Time and IQ Ref Time](https://helpfiles.keysight.com/csg/89600B/Webhelp/Subsystems/customIq/content/mnu_trdata_iqmeastimiqreftim_custiq.htm)
- [Rohde & Schwarz: Implementation of Real-Time Spectrum Analysis](https://scdn.rohde-schwarz.com/ur/pws/dl_downloads/dl_application/application_notes/1ef77/1EF77_3e_Real-time_Spectrum_Analysis.pdf)
- [Tektronix: Real-Time Spectrum Analysis for EMI Diagnostics](https://www.tek.com/en/documents/application-note/real-time-spectrum-analysis-emi-diagnostics)
- [Keysight: Capturing Signals for Measurement](https://www.keysight.com/bw/en/assets/9018-02562/user-manuals/9018-02562.pdf)
- [NI: Vector Signal Transceiver Hardware Architecture](https://www.ni.com/en/support/documentation/supplemental/12/the-ni-vector-signal-transceiver-hardware-architecture.html)
- [Analog Devices: ADALM-PLUTO Receive Architecture](https://wiki.analog.com/university/tools/pluto/users/receive)
- [Analog Devices: ADALM-PLUTO Performance Metrics](https://wiki.analog.com/university/tools/pluto/devs/performance)
