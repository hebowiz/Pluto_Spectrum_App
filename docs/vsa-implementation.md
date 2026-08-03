# VSA実装状況・引き継ぎノート

最終更新: 2026-08-02

設計上の判断は[vsa-architecture.md](vsa-architecture.md)を参照してください。この文書は実際に動作する範囲、既知の制約、次の実装順を第三者が把握するための記録です。

Bluetooth BR復調の詳細は[vsa-bluetooth-br.md](vsa-bluetooth-br.md)を参照してください。

## 1. 現在の到達点

独立したVSA application shellと、hardware/UIに依存しないoffline解析coreを追加しました。

起動方法:

```powershell
python -m pluto_sa.vsa.main
```

起動時はPlutoへ接続せず、生成GFSKを自動解析します。画面から次を実行できます。

- GFSK、QPSK、pi/4-DQPSKのtest waveform生成。
- NumPy `.npy` / `.npz`およびraw complex IQの読込み。
- IQ Power（Zero Span、dBm）、Spectrum表示。
- FSKのinstantaneous frequency表示。
- PSKのConstellation表示。
- symbol tableとbasic EVMまたはfrequency errorのsummary表示。
- result dockの移動、tab化、detach、表示/非表示。
- 現captureを更新せずに再解析する`Refresh Analysis`。

## 2. 実装済みcontract

### IQRecording

`pluto_sa/vsa/model.py`にsource-independentなimmutable IQ recordを追加しました。IQ、sample rate、center frequency、usable bandwidth、full scale、source、sample index、trigger位置、gap理由、metadataを保持します。

`recording_from_acquisition()`により、既存HighSpeed TA/Power Triggerの`IQAcquisitionRecord`をVSA recordへ変換できます。Pluto固有型を解析DSPへ渡さない境界です。

振幅換算条件として`full_scale`、base calibration offset、frequency-dependent offset、input correction、校正済みflagも保持します。Zero SpanのIQ Powerは既存TA/Power Triggerと同じ規約で次のようにdBmへ換算します。

```text
power_dbfs = 20 log10(|IQ| / full_scale)
power_dbm  = power_dbfs
           + 20 log10(full_scale)
           + calibration_offset_db
           + frequency_dependent_offset_db
           + input_correction_db
```

生成IQは0 dBm基準の校正済みtest sourceとして扱います。校正metadataを持たないfile IQも数値配列としては`power_dbm`を生成しますが、correctionが0 dBの仮値であるためUIに`Amplitude: Uncal`と表示します。絶対電力として使う前に校正条件を設定する必要があります。

### Signal Description

現在定義済みのmodulation kind:

- 2-FSK / GFSK
- BPSK / QPSK / OQPSK
- pi/4-DQPSK / 8DPSK

symbol rate、FSK deviation、TX filter、BT/Alpha相当parameter、mapping名を保持します。QAMはenum/analysisへまだ追加していません。

### Composite signal

`CompositeSignalDescription`と`ModulationSegment`を実装しました。1 recording内に重ならない複数変調区間を定義できます。

`VSAAnalyzer.analyze_composite()`は各segmentを対応するFSK/PSK analyzerで処理し、結果のtime/symbol timeを元captureの共通時間軸へ戻します。FSKからPSKへ切り替わる合成recordの固定testがあります。

現段階のsegment境界はmanual指定です。Bluetooth EDRのpacket detectorとprofile-driven境界判定は未実装です。

## 3. 現在のDSP

共通処理:

1. optional DC除去。
2. time/power trace生成。
3. Hann window FFTとrelative frequency spectrum生成。
4. instantaneous frequency生成。
5. manual symbol rateとtiming offsetからsymbol center生成。

FSK:

- symbol中央付近のinstantaneous frequencyを平均。
- 平均frequencyをcenter errorとして除去。
- 正負2値のsymbol/bit decision。
- expectedまたはestimated deviationをresult metadataへ保存。

PSK:

- symbol centerをlinear interpolation。
- ideal constellationへのnearest decision。
- pi/4-DQPSK/8DPSKは隣接symbol間のphase differenceをdecision。
- ideal referenceとの差からbasic RMS EVMを計算。

## 4. Source

- `GeneratedIQSource`: FSK/GFSK/PSK test waveform。seed固定に対応。
- `FileIQSource`: `.npy`、`.npz`、raw complex file。
- `recording_from_acquisition`: 共通Pluto acquisition record adapter。

`.npz`は`iq`、`sample_rate_hz`、`center_frequency_hz`を保存/復元できます。`.npy`とraw IQはUIでsample rateを指定します。SigMF、R&S `.iq.tar`、SCPI instrumentは未実装です。

## 5. Test

VSA unit test:

```powershell
python -m pytest tests/test_vsa_core.py -q
```

検証済み項目:

- IQ recordの所有権/read-only性。
- modulation segmentの順序とoverlap拒否。
- generated GFSKのbit decode。
- ideal QPSKとpi/4-DQPSKのsymbol decode/basic EVM。
- session invalidation。
- `.npz` round trip。
- Zero Span IQ PowerのdBm換算と既存TA補正規約との一致。
- spectrum frequency axis。
- 1 capture内のFSK/PSK segment一括解析と共通時間軸。

Qtは`QT_QPA_PLATFORM=offscreen`でwindow生成、初期GFSK解析、closeまでsmoke test済みです。

## 6. 重要な未実装・制約

- timing recovery、carrier recovery、frequency drift trackingは未実装。現在はmanual timingです。
- TX/RX/Measurement/Reference filter chainは未実装。
- generated Gaussian waveformは開発用近似であり、規格reference/EVM用filterではありません。
- PSK EVMはbasic decision-directed値。R&S相当のnormalization、同期、evaluation range、measurement filter条件をまだ満たしません。
- FSK error metricsはfrequency error/deviationの基礎のみです。
- burst/pattern search、packet field decode、DECT/Bluetooth profileは未実装。
- VSA UIはoffline Single/Refreshのみ。処理はUI thread上で同期実行します。
- Pluto live source、Power Trigger接続、SCPI sourceは未実装。
- Composite解析coreは動作しますがUIからsegment設定・表示はできません。

Bluetooth BRについてはAccess Code相関、GFSK timing/CFO/drift補正、Header rate 1/3 FEC、whitening、HEC、field抽出までcore実装済みです。任意LAPのAccess Codeを生成でき、保存IQ解析CLIとPluto finite capture CLIがあります。ただし実Bluetooth送信機IQでは未検証です。

現在の数値を規格適合判定やR&SとのEVM比較へ使用してはいけません。

## 7. 次の推奨実装順

1. PlutoでGIAC Inquiry IQを捕捉し、実電波のAccess Code/bit timingを検証。
2. 1 MHz Bluetooth channelizerと複数burst候補解析を追加。
3. FHS decodeからLAP/UAP/clockを取得し、CAC packet trackingへ接続。
4. packet TYPE別Payload length/FEC/CRCを追加。
5. pulse shaping/matched filter contractとPSK carrier/timing recoveryを追加。
6. VSA analysis workerを追加しUI threadからDSPを分離。
7. EDR guard/sync検出、Composite segment UI、EDR profileへ拡張。
8. SigMF、R&S IQ file、SCPI sourceを追加。

実機接続前に、生成waveformへCFO、timing offset、AWGNを注入したpytestを追加し、推定器の許容誤差を固定してください。
