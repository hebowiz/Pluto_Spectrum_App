# VSA実装状況・引き継ぎノート

最終更新: 2026-08-03

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
- Analysis Center/Bandwidthによる手動single-channel選択。
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

設定名と区分は投入済みのR&S FPL1-K70 VSA User Manual rev.12に準拠する。現在のUIでは`Modulation Type / Order`、`Symbol Rate`、`FSK Ref Deviation`、`Modulation Mapping`、`Transmit Filter Type`、`Alpha / BT`を同じページに配置した。内部の`SignalDescription`はsourceやBluetooth profileに依存しない。

### Pattern Search / Result Range / Demodulation

2026-08-03に一般VSA用の既知パターン解析を追加した。Bluetooth Access Code専用処理とは別の`pluto_sa/vsa/pattern.py`で、任意のFSK/GFSK/BPSK/QPSK/pi/4-DQPSK/8DPSK symbol列を検索し、patternを基準に指定範囲をsymbol単位で復調する。

設定責務はmanual pp.164-170、208-224に従い、次のように分離した。

- `KnownPattern`: Name、Description、Symbols。Result Lengthや検索しきい値は持たない。
- `PatternSearchSettings`: Pattern Search Auto/On/Off、I/Q Correlation Threshold（AutoはR&Sと同じ90%）、`Meas only if Pattern Symbols Correct`。
- `ResultRangeSettings`: Result Length、Reference、Alignment、Offset、`Symbol Number at Pattern Start`。
- `DemodulationSettings`: Coarse/Fine Synchronization、Bit Ordering、FSKのCarrier Frequency Drift/Deviation Error補償選択。

UIにも`Pattern Search`、`Result Range`、`Demodulation`を独立ページとして追加した。Pattern SymbolsはBinary、Decimal、Hexadecimalを入力できる。現在実際にDSPへ反映されるのはpattern、correlation threshold、Pattern Waveform/Leftを起点とした非負offset、Result Length、Bit Orderingである。その他はR&S互換の設定contractを先に固定した段階で、未実装項目を有効に見せないため今後段階的に接続する。

PSK検索はpattern symbol間の差分相関により一定phase回転とCFOに耐え、patternでcarrier phase/CFOを推定してResult Rangeをdecisionする。Differential PSKはphase incrementを直接検索する。FSK/GFSKは既存のCFO、deviation、drift推定器を任意patternへ一般化した。pattern後の出力はprotocol fieldではなく、symbol番号、symbol値、bit列、symbol時刻、測定vector/frequencyからなる汎用結果である。

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

1. optional Analysis CenterへのDDC、FIR low-pass、integer decimation。
2. optional DC除去。
3. time/power trace生成。
4. Hann window FFTとrelative/absolute frequency spectrum生成。
5. instantaneous frequency生成。
6. manual symbol rateとtiming offsetからsymbol center生成。

Analysis channel処理は`pluto_sa/vsa/channel.py`にsource/modulation非依存で実装済みです。
出力sample rateはAnalysis Bandwidthの約4倍を目安に、input rateの整数分周から選びます。
filter未選択時は元recordingをそのまま解析します。

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

`.npz`は`iq`、`sample_rate_hz`、`center_frequency_hz`、`usable_bandwidth_hz`、振幅補正条件を保存/復元できます。旧NPZにusable bandwidthがない場合は`0.8 * sample rate`をfallbackにします。`.npy`とraw IQはUIでsample rateを指定します。SigMF、R&S `.iq.tar`、SCPI instrumentは未実装です。

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
- DDC/FIR/decimation後の周波数軸とmetadata。
- 強い隣接GFSK packetを含む16 MSPS IQから手動選択したGIACの0-error復元。
- Pluto実測Inquiry IQをAnalysis Center/Bandwidth指定後も0-error復元。
- Pluto実測固定BR波形から通常Access Code、無誤りHeader FEC、DH1 27-byte bodyを復元。
- 任意FSK/GFSK patternの検索とResult Range復調。
- QPSKの任意phase/CFO下でのpattern検索、carrier補正、symbol復調。
- pi/4-DQPSKの差動pattern検索とLSB/MSB Bit Ordering。
- 16 MSPS Pluto実測BR captureを汎用Pattern Searchへ通し、手動Analysis Center/Bandwidth後に任意72-symbol patternを相関99%以上、0 symbol errorで検出。
- 実測DH1 body 216 bitとPRBS-9を0 bit errorで照合。
- Bluetooth SIG公式vectorに対するPayload CRCとcomplete DH1 payload decode。

Qtは`QT_QPA_PLATFORM=offscreen`でwindow生成、初期GFSK解析、closeまでsmoke test済みです。

## 6. 重要な未実装・制約

- 通常解析pipelineのtiming/carrier recoveryは未実装。Pattern Search側には8 samples/symbolへのresample、timing phase探索、patternを用いたcarrier phase/CFO推定があるが、symbol-rate error追従は未実装。
- TX/RX/Measurement/Reference filter chainは未実装。
- generated Gaussian waveformは開発用近似であり、規格reference/EVM用filterではありません。
- PSK EVMはbasic decision-directed値。R&S相当のnormalization、同期、evaluation range、measurement filter条件をまだ満たしません。
- FSK error metricsはfrequency error/deviationの基礎のみです。
- 一般VSA用pattern searchは実装済み。Burst Search、DECT profile、negative Result Range offsetによるpattern前symbol復調は未実装。
- VSA UIはoffline Single/Refreshのみ。処理はUI thread上で同期実行します。
- Pluto live source、Power Trigger接続、SCPI sourceは未実装。
- Composite解析coreは動作しますがUIからsegment設定・表示はできません。

Bluetooth BRについてはAccess Code相関、GFSK timing/CFO/drift補正、Header rate 1/3 FEC、whitening、HEC、field抽出、DH1 Payload/CRC、PRBS-9照合までcore実装済みです。任意LAPのAccess Codeを生成でき、保存IQ解析CLIとPluto finite capture CLIがあります。2026-08-03にスマートフォンのInquiryをPlutoで実測し、4 MSPS狭帯域captureからGIAC 68 bitを相関0.9979、0 bit errorで復元しました。さらに固定2441 MHzのBR test waveformを16 MSPSで取得し、通常Access Code、Header FEC、DH1 27-byte body、PRBS-9 216 bitを0 bit errorで復元しました。このtest waveformはUAP `0x6B`のHECとPayload CRCが一致せず、Whitening OFFかつcheck初期値が別設定の可能性があります。16 MSPS全帯域への直接相関は行わず、ユーザー指定Analysis Center/Bandwidthで1 channelを抽出してから復調します。詳細値は[vsa-bluetooth-br.md](vsa-bluetooth-br.md)を参照してください。

現在の数値を規格適合判定やR&SとのEVM比較へ使用してはいけません。

## 7. 次の推奨実装順

1. 現在の固定BR信号で、Access Codeの一部など任意patternをUI入力し、Result Rangeのsymbol列を実機確認。
2. Burst Searchとpattern前を含むnegative Result Range offsetを実装。
3. pulse shaping/matched/measurement filter contractとPSK symbol-rate recoveryを追加。
4. pattern検索結果からFSK→PSKのsegment boundaryを相対指定し、EDRを汎用Composite解析する。
5. EVM用Fine Synchronization、compensation、Evaluation Rangeを接続。
6. VSA analysis workerを追加しUI threadからDSPを分離。
7. SigMF、R&S IQ file、SCPI sourceを追加。

実機接続前に、生成waveformへCFO、timing offset、AWGNを注入したpytestを追加し、推定器の許容誤差を固定してください。
