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

## 4. RTSA表示処理

- FFT frameは連続sampleから生成し、必要なPOIに応じてoverlap率を設定します。
- RBWは解析window/FFT bin幅/デジタルfilterの関係を明示します。
- Spectrogramの1行へ複数FFTを集約する場合、Sample/Positive Peak/Negative Peak/Average等のdetectorを適用します。
- Persistenceはtrace holdではなく、frequency-amplitude cellの発生頻度を蓄積するdensity表示として将来分離します。
- producer overrun、consumer lag、FFT処理率、最大見逃しevent時間を計測指標にします。

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
- Power/FMT detector、pre/poststore state machine: 未実装
- HighSpeed TAのtrigger-aware record consumer化: 未実装
- VSA demodulation: 未実装

## 参考資料

- [Rohde & Schwarz: Implementation of Real-Time Spectrum Analysis](https://scdn.rohde-schwarz.com/ur/pws/dl_downloads/dl_application/application_notes/1ef77/1EF77_3e_Real-time_Spectrum_Analysis.pdf)
- [Tektronix: Real-Time Spectrum Analysis for EMI Diagnostics](https://www.tek.com/en/documents/application-note/real-time-spectrum-analysis-emi-diagnostics)
- [Keysight: Capturing Signals for Measurement](https://www.keysight.com/bw/en/assets/9018-02562/user-manuals/9018-02562.pdf)
- [NI: Vector Signal Transceiver Hardware Architecture](https://www.ni.com/en/support/documentation/supplemental/12/the-ni-vector-signal-transceiver-hardware-architecture.html)
- [Analog Devices: ADALM-PLUTO Receive Architecture](https://wiki.analog.com/university/tools/pluto/users/receive)
- [Analog Devices: ADALM-PLUTO Performance Metrics](https://wiki.analog.com/university/tools/pluto/devs/performance)
