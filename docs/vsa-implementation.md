# VSA実装状況・引き継ぎノート

最終更新: 2026-08-03

設計上の判断は[vsa-architecture.md](vsa-architecture.md)を参照してください。この文書は実際に動作する範囲、既知の制約、次の実装順を第三者が把握するための記録です。

Bluetooth BR復調の詳細は[vsa-bluetooth-br.md](vsa-bluetooth-br.md)を参照してください。

CFO、carrier phase、linear driftの計算式とsample単位補正は[vsa-carrier-synchronization.md](vsa-carrier-synchronization.md)を参照してください。

## 1. 現在の到達点

独立したVSA application shellと、hardware/UIに依存しないoffline解析coreを追加しました。

起動方法:

```powershell
python -m pluto_sa.vsa.main
```

起動時はPlutoへ接続せず、生成GFSKを自動解析します。画面から次を実行できます。

- GFSK、QPSK、pi/4-DQPSKのtest waveform生成。
- NumPy `.npy` / `.npz`およびraw complex IQの読込み。
- Plutoからの非同期finite Run Single capture。
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

生成IQは0 dBm基準の校正済みtest sourceとして扱います。新規NPZは`full_scale`を含む全振幅metadataを保存する。旧Pluto NPZのうち生ADC値と判定できるものは`full_scale=2048`、公称変換`-62 dB`をfallbackし、校正値とは区別してUIに`Amplitude: Nominal Pluto`と表示する。外部ATT、RX gain、外部gainはIQ値だけから推定できないため、既知の取得条件は同名の`.npz.json` sidecarで上書きできる。

実測BR fixtureには、取得時のRX gain 0 dB、外部ATT 30 dBをsidecarへ記録した。これにより表示基準はPluto入力端ではなくATT手前のsource planeとなり、capture peakは従来の誤った約+35 dBmから公称約+3 dBmへ修正される。`amplitude_calibrated=false`なので、この値は絶対確度を保証する校正値ではない。

### Signal Description

設定名と区分は投入済みのR&S FPL1-K70 VSA User Manual rev.12に準拠する。現在のUIでは`Modulation Type / Order`、`Symbol Rate`、`FSK Ref Deviation`、`Modulation Mapping`、`Transmit Filter Type`、`Alpha / BT`を同じページに配置した。内部の`SignalDescription`はsourceやBluetooth profileに依存しない。

Meas Configは縦並びのaccordionではなく、`Config Top Menu`にカテゴリボタンを2列で配置する。ボタンから個別設定ページへ移動し、`< Config Top`でトップへ戻る2階層構造とする。トップのカテゴリボタンは18 pt以上の太字・高さ84 px以上とし、タイトルも個別設定画面より大きく表示する。ダイアログを開くたびにトップを表示し、Window Modalによるメイン画面の操作抑止は維持する。

### Pattern Search / Result Range / Demodulation

2026-08-03に一般VSA用の既知パターン解析を追加した。Bluetooth Access Code専用処理とは別の`pluto_sa/vsa/pattern.py`で、任意のFSK/GFSK/BPSK/QPSK/pi/4-DQPSK/8DPSK symbol列を検索し、patternを基準に指定範囲をsymbol単位で復調する。

設定責務はmanual pp.164-170、208-224に従い、次のように分離した。

- `KnownPattern`: Name、Description、Symbols。Result Lengthや検索しきい値は持たない。
- `PatternSearchSettings`: Pattern Search Auto/On/Off、I/Q Correlation Threshold（AutoはR&Sと同じ90%）、`Meas only if Pattern Symbols Correct`。
- `ResultRangeSettings`: Result Length、Reference、Alignment、Offset、`Symbol Number at Pattern Start`。
- `DemodulationSettings`: Coarse/Fine Synchronization、Bit Ordering、FSKのCarrier Frequency Drift/Deviation Error補償選択。

UIにも`Pattern Search`、`Result Range`、`Demodulation`を独立ページとして追加した。Pattern SymbolsはBinary、Decimal、Hexadecimalを入力できる。現在実際にDSPへ反映されるのはpattern、correlation threshold、Pattern Waveform/Leftを起点とした非負offset、Result Length、Bit Orderingである。その他はR&S互換の設定contractを先に固定した段階で、未実装項目を有効に見せないため今後段階的に接続する。

PSK検索はpattern symbol間の差分相関により一定phase回転とCFOに耐え、patternでcarrier phase/CFOを推定してResult Rangeをdecisionする。Differential PSKはphase incrementを直接検索する。FSK/GFSKは既存のCFO、deviation、drift推定器を任意patternへ一般化した。pattern後の出力はprotocol fieldではなく、symbol番号、symbol値、bit列、symbol時刻、測定vector/frequencyからなる汎用結果である。

R&SのResult Range表示に合わせ、Pattern Search成立時は次の表示規則とする。

- IQ PowerとInstantaneous Frequencyはcapture全体のdataを保持したまま、表示X軸を選択中Result Rangeの前後10%へfitする。Pattern Waveformを緑、Result Rangeを青の領域、Pattern Startを縦線で示す。zoom/panでcapture内の他部分も確認できる。
- Spectrumはcapture全体ではなく、検出されたResult RangeのIQだけから再計算し、titleを`Spectrum (Result Range)`とする。
- FSK Instantaneous Frequencyの初期Y軸は`FSK Ref Deviation`の±150%とする。
- Symbol Tableは`QTableWidget`を使い、Result Rangeの復調symbolを中央揃えの10列へ配置する。列headerは0～9、行headerはその行の先頭symbol indexとする。

Symbol Tableは現在decimal symbol値のみを表示する。将来の4FSK/8FSK、PSK、QAMを想定し、Binary/Hexadecimal/Decimal表示切替を追加する。表示formatはdecision結果を変更せず、R&Sの`Symbol Format`と同様にview設定として扱う。

### Carrier周波数測定・補正の実装境界

- 通常FSK解析はResult Range内symbol frequencyの平均を`frequency_error_hz`として算出する基礎処理がある。
- GFSK/FSK Pattern Searchはpatternからcarrier frequency offset、frequency deviation、linear frequency driftを同時推定し、offset/drift/polarityを除去してsymbol decisionする。
- PSK Pattern Searchはpatternからcarrier phaseとcarrier frequency offsetを推定し、Result Rangeのsymbol decisionへ補正する。Differential PSKはphase increment上で補正する。
- これらはPattern Searchを基準にしたcoarse synchronizationである。
- CFO、推定carrier frequency、linear driftをResult Summaryへ表示する。
- `Display Config > Carrier Display`で`Raw IQ`と`Carrier Corrected`を切り替える。補正表示ではsample単位の位相補正後IQからInstantaneous FrequencyとResult Range Spectrumを再計算する。既定はCarrier Corrected。
- `Display Config > Show Symbol Points`をONにすると、IQ PowerとModulationのtrace上へ復調symbol中心位置を明るい緑の点で重ねる。FSKではtime/frequency座標、PSKではIQ軌跡座標を使用する。既定はOFF。
- PSKのIQ軌跡は8 samples/symbolへresampleし、TX FilterがRoot Raised Cosineなら同じalphaのSRRC matched receive filterを通した連続IQを使用する。軌跡とsymbol markerは同一のfilter outputとsymbol時刻RMS正規化を共有する。
- 解析完了時に全plotの初期X/Y rangeをsnapshotし、`Display Config > Reset Graph Scales`または`Home`で復元する。mouse modeは全plot共通で、既定の`Rect Zoom`は左drag矩形拡大、`Pan`は左drag移動とする。表示のみの更新では現在rangeを維持する。
- Carrier Frequency Drift補正はDemodulation設定から切り替えられるが、実測cross-validation未完了のため既定OFF。CFO補正はCarrier Corrected表示で常に適用する。
- R&S相当のFine Synchronization、残留CFO評価、estimator confidenceは未実装。

### Meas Config window

従来のmain window右側dockを廃止し、menu barの`Meas Config > Open Meas Config...`または`Ctrl+M`から独立dialogを開く。dialogはWindow Modalであり、開いている間はplot、dock、menuを含むmain windowを操作できない。設定dialog内の`Refresh Analysis`で現在captureを再解析し、`Close`でmain windowへ戻る。

### Default result window layout

main workspaceの結果blockはすべて同格の`QDockWidget`とし、central widgetを使用しない。起動時は同一幅・同一高さの3列×2行へ配置する。

```text
IQ Power    | Spectrum | Result Summary
Modulation  | Symbol Plot | Symbol Table
```

- IQ Powerも他resultと同様に移動、float、close、再表示できる。
- Result SummaryはSymbol Tableから分離し、`Parameter`と`Current`の2列へ項目を縦方向に並べる。
- Result SummaryとSymbol Tableのdata cell背景は交互色を使わず、単一色で統一する。
- Symbol PlotはPSK時にConstellation、FSK時に1 symbol期間の位相差分をI/Q平面へ表示するDock Widget。
- 初期geometryは各列幅と各行高を均等化する。ユーザーが移動・resizeした後はQt dock layoutに従う。

### Capture内の複数pattern

現実装はcapture全体のcorrelation最大値を1件だけ採用し、`PatternSearchResult`とResult Rangeも1件だけ保持する。同じpatternを含むpacketが複数ある場合、時刻順の最初ではなく相関が最も高いpacketが表示される。したがって現段階では他候補をNext/Previousで選べない。

R&S manual pp.116-117、143-145では、Burst Search有効時は各burst内の最初のpatternを検索し、複数の離散Result Rangeをcapture buffer上に持ち、`Select Result Rng`で現在範囲を切り替える。今後は単純なcorrelation local peak列挙ではなく、Burst Searchでpacket候補を分離してから各burstの最初のpatternを対応づけ、available Result Rangesとselected Result Rangeを別管理する。この段階でNext/Previousまたはcapture上のrange選択UIを追加する。

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
- `FileIQSource`: R&S `.iq.tar`、`.npy`、`.npz`、raw complex file。
- `recording_from_acquisition`: 共通Pluto acquisition record adapter。
- `PlutoLiveSource`: 共通`PlutoReceiver`とtrigger/acquisition contractを使うfinite live source。

`.npz`は`iq`、`sample_rate_hz`、`center_frequency_hz`、`usable_bandwidth_hz`、振幅補正条件を保存/復元できます。旧NPZにusable bandwidthがない場合は`0.8 * sample rate`をfallbackにします。`.npy`とraw IQはUIでsample rateを指定します。SigMFとSCPI instrumentは未実装です。

### Pluto live Run Single

R&S VSAの`Input / Frontend`、`Signal Capture > Data Acquisition`、`Sweep / Run`の区分に合わせ、Pluto finite captureを追加した。`Run Single`（F6）はGUI threadとは別のcapture threadでPluto接続、設定、IQ取得を行い、完了したimmutable `IQRecording`だけをmain threadへ返す。`Refresh Analysis`（F5）は従来どおり現在のcaptureを再解析し、再取得しない。

初期設定はsymbol rate 1 Msym/s、capture oversampling 8 samples/symbol、source sample rate 8 MS/s、RF bandwidth 8 MHz、capture length 3 ms、record length 24,000 samples、nominal usable I/Q bandwidth 6.4 MHz。Capture Lengthはmsまたはsymbolsで指定し、実sample countへ変換する。Plutoからread backした実sample rateとRF bandwidthをrecord metadataの正本とする。

振幅補正はSA/VSAで別実装にしない。`pluto_sa.config.input_frontend.InputPowerCorrection`を共通contractとし、次式を`SpectrumConfig.input_correction_db`とVSA live captureの両方で使う。

```text
input_correction_db = external_attenuation_db
                    - internal_gain_db
                    - external_gain_db

power_dbm = 20 log10(|IQ|)
          + Pluto calibration offset (-62 dB nominal)
          + frequency-dependent offset
          + input_correction_db
```

Internal GainはPluto hardwareへ設定し、External ATT/GainはDUT側reference planeへの表示補正としてのみ適用する。初期値はPluto SAと共通でInternal Gain 30 dB、External ATT 30 dB、External Gain 0 dB。現段階の`-62 dB`は公称換算でありtraceable calibrationではないため、表示単位はdBmだがmetadataの`amplitude_calibrated`はfalseを維持する。

`Swap I/Q`はR&Sと同じ`Q + jI`変換をcapture後に適用する。現在はFree Run / Singleのみ。I/Q Power Trigger、negative Trigger Offset（pretrigger）、Continuous、Stopは次段階で同じ共通streamへ接続する。

### R&S iq-tar import

Rohde & Schwarzの公式[iq-tar File Format Specification](https://scdn.rohde-schwarz.com/ur/pws/dl_downloads/dl_common_library/dl_manuals/dl_manual/RS_iq-tar_FileFormatSpecification_en_02.pdf) version 2に基づき、`.iq.tar`を展開せず直接読み込む。archive内の唯一のparameter XMLから次を復元する。

- `Samples`、`Clock`（sample rate）、`Format`、`DataType`
- `ScalingFactor`（省略時1 V）、`NumberOfChannels`（省略時1）
- `DataFilename`、任意の`Name` / `Comment` / `DateTime`
- R&S固有`UserData`内に`CenterFrequency`がある場合のcenter frequency

対応形式は`complex`（I/Q）、`real`（Q=0として格納）、`polar`（magnitude/phase）で、data typeは`int8`、`int16`、`int32`、`float32`、`float64`。polarは仕様どおりfloatのみとする。complex/polarの成分、さらにmulti-channel sampleは仕様の順序どおりinterleaveを解除し、`channel_index`（既定0）で1 channelを選択して共通`IQRecording`へ変換する。`ScalingFactor`はcomplex/realの全成分、polarのmagnitudeだけに適用し、IQ値をVoltにする。

現行VSA UIはsingle-channel解析のため、ファイルダイアログからの読込みではchannel 0を使う。core loaderはmulti-channel archiveから任意channelを選択可能なので、将来のchannel selector追加時もDSP contractは変えない。

振幅値は仕様上Voltへ変換されるが、iq-tar core metadataだけでは測定系のimpedance、peak/RMS定義、外部補正を一意に決められない。そのため`amplitude_calibrated=False`を維持し、Volt値をdBm校正済みとは扱わない。必要な補正は取得条件またはinstrument固有`UserData`の仕様が判明した段階で追加する。

安全性と誤読防止のため、tarをfilesystemへextractせず、parameter XMLの個数、`DataFilename`、regular file属性、path traversal、XML DTD/entity、binary byte数を検証する。現在のR&S/Windows出力に合わせてmulti-byte binaryはlittle-endianとして読む。公式仕様はbyte orderを明記していないため、異なるendianのproducerが必要になった場合は明示的な選択肢を追加する。

手動確認用fixtureは`tests/fixtures/rs_sample_gfsk_8msps.iq.tar`。中心周波数2441 MHz、sample rate 8 MS/s、symbol rate 1 Msym/s、deviation 250 kHz、BT 0.5のdeterministic GFSKで、先頭16 symbolsは`1010...`、以降240 symbolsはPRBS9とする。`tools/generate_rs_iqtar_fixture.py`で同一内容を再生成できる。

Analysis BandwidthのFIR適用後は、FSKの交互patternに対して複数のsymbol timing phaseがほぼ同じ正規化相関になる場合がある。正規化相関は振幅を捨てるため、transition付近の小さなtone separationを誤って選び得る。timing recoveryでは最大相関との差が1 percentage point以内の候補を比較し、既知patternへfitしたfrequency separation（eye opening）が最大相関候補より20%以上広い場合に限り、その候補を選ぶ。小さなpulse-shape非対称では従来の最大相関時刻を維持し、channel filterでeyeが明確に閉じた場合だけsymbol centerへ補正する。

また、短い交互patternだけから得たlinear driftはAnalysis Bandwidth FIRのcapture端過渡と強く結合する。誤ったdriftを初回payload decisionへ適用するとdecision-directed fitが自己強化し、瞬時周波数が同じでもFSK Symbol Phase Differenceだけが大きく変形する。patternはCFO、polarity、deviationの初期推定に使うが、十分長いResult Rangeがある場合のpacket-wide drift refinementは0 driftから開始し、全symbolのdecisionを使って再推定する。Symbol Plotとcarrier correctionへ同じ安定した推定値を渡す。

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

- 通常解析pipeline単独のtiming/carrier recoveryは未実装。Pattern Search成立後の差動PSKでは、判定差動symbolを累積しTX Filter設定に対応する絶対referenceを生成する。8 samples/symbolへのresampleとmatched filter後、Result Range全体の測定絶対IQとreferenceの複素EVMを目的関数としてfractional timing、symbol-rate error、carrier phase/CFO、linear driftを同時推定する。既知patternを固定reference、残りをdecision-directed referenceとして反復し、zero-drift開始とM乗coarse開始の低EVM側を採用する。
- TX/RX/Measurement/Reference filter chainは未実装。
- generated Gaussian waveformは開発用近似であり、規格reference/EVM用filterではありません。
- PSK EVMはbasic decision-directed値。R&S相当のnormalization、同期、evaluation range、measurement filter条件をまだ満たしません。
- FSK error metricsはfrequency error/deviationの基礎のみです。
- 一般VSA用pattern searchは実装済み。Burst Search、DECT profile、negative Result Range offsetによるpattern前symbol復調は未実装。
- VSA UIはPluto Free Run / Run Singleとoffline Refreshに対応。Pluto取得は非同期だが、取得後のDSP解析はUI thread上で同期実行します。
- Pluto Power Trigger、Continuous、SCPI sourceは未実装。
- Composite解析coreは動作しますがUIからsegment設定・表示はできません。

Bluetooth BRについてはAccess Code相関、GFSK timing/CFO/drift補正、Header rate 1/3 FEC、whitening、HEC、field抽出、DH1 Payload/CRC、PRBS-9照合までcore実装済みです。任意LAPのAccess Codeを生成でき、保存IQ解析CLIとPluto finite capture CLIがあります。2026-08-03にスマートフォンのInquiryをPlutoで実測し、4 MSPS狭帯域captureからGIAC 68 bitを相関0.9979、0 bit errorで復元しました。さらに固定2441 MHzのBR test waveformを16 MSPSで取得し、通常Access Code、Header FEC、DH1 27-byte body、PRBS-9 216 bitを0 bit errorで復元しました。このtest waveformはUAP `0x6B`のHECとPayload CRCが一致せず、Whitening OFFかつcheck初期値が別設定の可能性があります。16 MSPS全帯域への直接相関は行わず、ユーザー指定Analysis Center/Bandwidthで1 channelを抽出してから復調します。詳細値は[vsa-bluetooth-br.md](vsa-bluetooth-br.md)を参照してください。

現在の数値を規格適合判定やR&SとのEVM比較へ使用してはいけません。

## 7. 次の推奨実装順

1. Pluto I/Q Power Trigger、Trigger Offset、Continuous/Stopを接続。
2. pattern前を含むnegative Result Range offsetを実装。
3. pulse shaping/matched/measurement filter contractとPSK symbol-rate recoveryを追加。
4. pattern検索結果からFSK→PSKのsegment boundaryを相対指定し、EDRを汎用Composite解析する。
5. EVM用Fine Synchronization、compensation、Evaluation Rangeを接続。
6. VSA analysis workerを追加しUI threadからDSPを分離。
7. SigMF、SCPI sourceとiq-tarのUI channel selectorを追加。

実機接続前に、生成waveformへCFO、timing offset、AWGNを注入したpytestを追加し、推定器の許容誤差を固定してください。
