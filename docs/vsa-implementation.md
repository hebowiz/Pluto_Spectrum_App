# VSA実装状況・引き継ぎノート

最終更新: 2026-08-07

設計上の判断は[vsa-architecture.md](vsa-architecture.md)を参照してください。この文書は実際に動作する範囲、既知の制約、次の実装順を第三者が把握するための記録です。

Bluetooth BR復調の詳細は[vsa-bluetooth-br.md](vsa-bluetooth-br.md)を参照してください。

CFO、carrier phase、linear driftの計算式とsample単位補正は[vsa-carrier-synchronization.md](vsa-carrier-synchronization.md)を参照してください。

## 1. 現在の到達点

独立したVSA application shellと、hardware/UIに依存しないoffline解析coreを追加しました。

起動方法:

```powershell
python -m pluto_sa.vsa.main
```

起動時はPlutoへ接続せず、前回終了時のMeas Configだけを復元した`No capture`状態で
開始します。IQ sample、IQ file path、直前の解析結果は保存・自動読込みしません。
画面から次を実行できます。

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

正常終了時は`_meas_config_values()`のJSON互換値をQt `QSettings`の
`startup/measurement_config`へ保存する。schema/version付きdocumentとし、次回起動時は
controlへ適用するだけで解析を開始しない。破損・旧version・不正schemaは削除してwidget
defaultへfallbackし、起動を妨げない。手動Config fileと同じ測定項目を対象とするが、
IQ dataとrecording metadataはstartup documentへ含めない。

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

2026-08-19にPSKの`Modulation Mapping`へ`Gray`と`Bluetooth EDR`を追加した。
`Gray`はR&Sの汎用QPSK/D8PSK Gray table、`Bluetooth EDR`はBluetooth Coreの
pi/4-DQPSK/8DPSK bit-to-differential-phase tableを使う。両者は8DPSKの上半分が
一致しないため別設定とする。constellation alphabetを論理symbol順に並べることで、
pattern search、symbol decision、Symbol Table、bit conversionが同じmappingを共有する。
FSKは従来どおり`Natural`固定で、`Bluetooth EDR`はpi/4-DQPSK/8DPSK以外では無効。

同日にPSK Symbol Plotへ`Absolute IQ (Physical)`と`Differential IQ`の切替を
追加した。前者を既定とし、R&S同様にmatched-filter後の絶対IQ決定点を表示する。
pi/4-DQPSKはsymbolごとのpi/4回転を表示時に補償し、8DPSKは絶対点をそのまま使う。
Result Summaryは通常の絶対IQ `EVM RMS`、従来の`Differential Symbol EVM RMS`、
Bluetooth Appendix C式の`Bluetooth DEVM RMS`を独立項目として選択可能にした。

Meas Configは縦並びのaccordionではなく、`Config Top Menu`にカテゴリボタンを2列で配置する。ボタンから個別設定ページへ移動し、`< Config Top`でトップへ戻る2階層構造とする。トップのカテゴリボタンは18 pt以上の太字・高さ84 px以上とし、タイトルも個別設定画面より大きく表示する。ダイアログを開くたびにトップを表示し、Window Modalによるメイン画面の操作抑止は維持する。

### Pattern Search / Result Range / Demodulation

2026-08-03に一般VSA用の既知パターン解析を追加した。Bluetooth Access Code専用処理とは別の`pluto_sa/vsa/pattern.py`で、任意のFSK/BPSK/QPSK/pi/4-DQPSK/8DPSK symbol列を検索し、patternを基準に指定範囲をsymbol単位で復調する。

設定責務はmanual pp.164-170、208-224に従い、次のように分離した。

- `KnownPattern`: Name、Description、Symbols。Result Lengthや検索しきい値は持たない。
- `PatternSearchSettings`: Pattern Search Auto/On/Off、I/Q Correlation Threshold（AutoはR&Sと同じ90%）、`Meas only if Pattern Symbols Correct`、FSK専用の`Allow Inverted Pattern Match`。
- `ResultRangeSettings`: Result Length、Reference、Alignment、Offset、`Symbol Number at Pattern Start`。
- `DemodulationSettings`: Coarse/Fine Synchronization、Bit Ordering、FSKのCarrier Frequency Drift/Deviation Error補償選択。

UIにも`Pattern Search`、`Result Range`、`Demodulation`を独立ページとして追加した。Pattern SymbolsはBinary、Decimal、Hexadecimalを入力できる。現在実際にDSPへ反映されるのはpattern、correlation threshold、Pattern Waveform/Leftを起点とした非負offset、Result Length、Bit Orderingである。その他はR&S互換の設定contractを先に固定した段階で、未実装項目を有効に見せないため今後段階的に接続する。

PSK検索はpattern symbol間の差分相関により一定phase回転とCFOに耐え、patternでcarrier phase/CFOを推定してResult Rangeをdecisionする。Differential PSKはphase incrementを直接検索する。FSK/GFSKは既存のCFO、deviation、drift推定器を任意patternへ一般化した。pattern後の出力はprotocol fieldではなく、symbol番号、symbol値、bit列、symbol時刻、測定vector/frequencyからなる汎用結果である。

R&SのResult Range表示に合わせ、Pattern Search成立時は次の表示規則とする。

- IQ PowerとInstantaneous Frequencyはcapture全体のdataを保持したまま、表示X軸を選択中Result Rangeの前後10%へfitする。Pattern Waveformを緑、Result Rangeを青の領域、Pattern Startを縦線で示す。zoom/panでcapture内の他部分も確認できる。
- Spectrumはcapture全体ではなく、検出されたResult RangeのIQだけから再計算する。現在は中央の最大`fft_size` samplesへHann windowを掛け、`|FFT| / sum(window)`のcoherent amplitudeを`20log10`した後、IQ Powerと同じfull-scaleおよびfrontend補正を適用してdBm表示する。bin中心CWのSpectrum peakは同じIQ振幅のZero Span powerと一致する。これはdBm/Hzのpower spectral densityではなく、Hann相当の各resolution filter出力のdBmである。broadband/modulated signalはpowerが複数filterへ分散するため各点がtotal powerより低くなる。実効noise bandwidthはzero padding後の表示bin間隔ではなく、使用record長`L`に対して概ね`1.5 * Fs / L`である。現時点ではwindow/RBW/ENBW normalizationを選択するSpectrum measurement設定がなく、R&S VSAのresult transformationと同じ高水準構成ではあるが測定値を完全再現する仕様ではない。
- FSK Instantaneous Frequencyの初期Y軸は`FSK Ref Deviation`の±150%とする。
- Symbol Tableは`QTableWidget`を使い、Result Rangeの復調symbolを中央揃えの10列へ配置する。列headerは0～9、行headerはその行の先頭symbol indexとする。pattern範囲内でも、設定patternと実測decisionが一致したsymbolだけを緑背景にし、不一致symbolは通常背景のまま表示する。
- Symbol Tableは`File > Export Symbol Table...`または右クリックから、schema/version付きUTF-8 JSON（`.vsasymbols.json`）へ全Result Rangeをexportできる。source、signal/mapping/bit-ordering、pattern variant、列定義、symbol/time/pattern statusを保持し、UIの2048 symbol表示上限には切り詰めない。
- Symbol Tableの入力済みcellをクリックすると、同じResult Range symbolをIQ Power、Modulation、Symbol Plotへcyan diamondで連動表示する。選択は1 symbolのみで、同じcellの再クリックにより解除する。IQ Powerはsymbol番号/dBm、FSK Modulationはsymbol番号/frequency、PSK Modulationはsymbol番号/normalized amplitude/phase、PSK Symbol Plotはsymbol番号/normalized amplitude/phase/point EVMをplot内labelへ表示する。FSK Modulationはtraceを復元symbol中心で補間した瞬時周波数を使用する。FSK Symbol PlotのPhase Differenceは復調用symbol区間平均から位相差vectorを生成し、瞬時値と混同しないようFrequencyを表示しない。Constellation Frequencyでは同じ復調symbol周波数を縦軸へ描き、markerにもFrequencyを表示する。選択markerは通常のFlat Symbol Plot point（6 px）とは独立した18 pxとし、黒輪郭付きで強調する。新規解析開始時には選択をclearする。

Symbol Tableは現在decimal symbol値のみを表示する。将来の4FSK/8FSK、PSK、QAMを想定し、Binary/Hexadecimal/Decimal表示切替を追加する。表示formatはdecision結果を変更せず、R&Sの`Symbol Format`と同様にview設定として扱う。

### Carrier周波数測定・補正の実装境界

- 通常FSK解析はResult Range内symbol frequencyの平均を`frequency_error_hz`として算出する基礎処理がある。
- GFSK/FSK Pattern Searchはpatternからcarrier frequency offset、frequency deviation、linear frequency driftを同時推定し、offset/driftを除去してsymbol decisionする。VSAのNatural mappingではpolarityを自動反転しない。
- PSK Pattern Searchはpatternからcarrier phaseとcarrier frequency offsetを推定し、Result Rangeのsymbol decisionへ補正する。Differential PSKはphase increment上で補正する。
- Root Raised Cosineを選んだPSKは二段carrier recoveryとする。第1 passでcoarse CFOを
  推定し、resample後のIQをNCOで中心へ移してからSRRC matched/measurement filterを
  適用し、第2 passでresidual CFO、timing、phase、driftをfine推定する。外部へ返すCFOは
  coarse+residual、phaseは原recordingのtime originへ再合成する。これはBluetooth専用
  profileではなくPSK共通処理である。metadataには
  `prefilter_cfo_correction_applied`、`prefilter_coarse_cfo_hz`、
  `postfilter_residual_cfo_hz`を保存する。
- これらはPattern Searchを基準にしたcoarse synchronizationである。
- CFO、推定carrier frequency、linear driftをResult Summaryへ表示する。
- `Display Config > Modulation Signal`はFSK/PSK共通で`Raw IQ`と`Measured`を切り替える。既定は`Measured`。Raw IQはcarrier補正とMeasurement Filterの双方を適用せず、PSKは取得IQ軌跡、FSKは取得IQ由来の瞬時周波数を表示する。Measuredはsample単位のCFO（設定時はdriftも）を補正したIQを入力とし、PSKでは8 samples/symbolへresample後、TX FilterがRRCなら同じalphaのSRRC matched/measurement filterを適用する。FSKでは同じく8 samples/symbolへresampleした瞬時周波数へ、TX FilterがGaussianなら同じBTのGaussian Auto Measurement Filterを適用する。切替は解析結果、symbol decision、EVMを変更しない。Spectrumは常にcarrier未補正のResult Range IQを表示する。
- `Display Config > Show Symbol Points`をONにすると、IQ PowerとModulationのtrace上へ復調symbol中心位置を明るい緑の点で重ねる。FSKではtime/frequency座標、PSKではIQ軌跡座標を使用する。既定はOFF。
- measurement configの`display_config`には`Show Symbol Points`、共通`Modulation Signal`、Symbol PlotのFlat/Density、PSKのAbsolute/Differential IQ、FSKのPhase Difference/Constellation Frequencyを保存し、config読込時と次回起動時に復元する。旧configに`modulation_signal`がない場合は旧`carrier_display`が`Raw IQ`ならRaw IQ、それ以外はMeasuredへ移行する。
- IQ Power/Modulation上の`Pattern Start`ラベルはtraceとの重なりを抑えるため、縦markerの下端寄りに配置する。
- Signal DescriptionのFSK系modulation名は`FSK`へ統一する。Gaussian shapingは独立したTransmit FilterとBTで定義し、symbol同期もmodulation名ではなくこのfilter設定を参照する。旧`2-FSK`/`GFSK` configは読込時に`FSK`へ移行する。
- EVM系Result SummaryとSymbol Plotの単一symbol EVM markerは、linear percentageと振幅比`20 log10(EVM/100)`を`x.xx % / y.y dB`形式で併記する。
- PSKのSymbol Plot dock titleは表示方式と連動し、`Symbol Plot (Physical)`または`Symbol Plot (Differential)`と表示する。FSKでは従来どおり`Symbol Plot`とする。
- PSKのIQ軌跡は8 samples/symbolへresampleし、TX FilterがRoot Raised Cosineなら同じalphaのSRRC matched receive filterを通した連続IQを使用する。軌跡とsymbol markerは同一のfilter outputとsymbol時刻RMS正規化を共有する。
- 解析完了時に全plotの初期X/Y rangeをsnapshotし、`Display Config > Reset Graph Scales`または`Home`で全plotを復元する。各plotの既存右クリックメニュー先頭には、そのplotだけを復元する`Reset`と、有限な表示traceだけを5%余白付きで収める`View All`を置く。後者はPattern/Result Rangeの帯・境界線等のoverlayを範囲計算から除外し、IQ平面の1:1 scaleを維持する。全plotは専用ViewBoxで左drag=`Rect Zoom`、middle button（wheel押込み）drag=`Pan`、right click=context menuへ固定する。Display ConfigのMouse Interaction menuと右クリックmenuの`Mouse Mode`は設けず、外部からPanModeを指定しても左dragはRect Zoomへ戻す。right dragのpyqtgraph軸scaleは維持する。表示のみの更新では現在rangeを維持する。
- 主traceはIQ Powerと同じyellowへ統一し、Power/Modulation上のsymbol markerは5.5 pxとする。Symbol Plotのsymbolは塗りと輪郭を同じyellowとする。空data時、およびFSK/PSK Symbol PlotはQ軸`-1.25..+1.25`を初期rangeとし、packetごとの振幅percentileでは変更しない。I軸はwidgetの縦横比と1:1単位scaleを維持するため、横長widgetでは±1.25より広く見える。FSK Phase DifferenceとPSK Constellationの双方へ半径1のgray reference circleを表示する。PSK IQ Trajectoryは解析完了時の全有限trace sampleが収まる最大I/Q成分へ5%の余白を加え、最小rangeを±1.25として自動設定する。両IQ平面ともI/Qの単位scaleは1:1に固定する。全plotの縦・横軸labelは実際の軸size中央へ合わせる。plot内titleはdock titleとの重複を避けるため表示しない。各result dockのtitleはboldかつ通常UI fontの130%とし、dock内容のfontは通常size/weightを維持する。`Display Config > Show Symbol Points`は単キー`S`でもON/OFFできる。
- `Display Config > Symbol Plot Trace`でPSK/FSK Symbol Plotを`Flat`（既定のyellow
  scatter）または`Density`へ切り替える。Densityはcurrent Result Rangeの最終
  symbol vectorを96×96 I/Q histogramへ集計する。2026-08-18以降はPSK/FSKとも
  全有限symbolのI/Q成分最大値に2%の余白を加えた範囲を使用し、最小範囲だけを
  `-1.25..+1.25`とする。したがって±1.25外の点もhistogramから欠落しない。
  PSKの解析完了時の初期viewは従来どおり±1.25を維持し、外側はzoomまたは
  `View All`で確認する。
  各観測を標準偏差0.7 binのGaussian kernelで平滑化してから`log(1+density)`を
  turbo color mapで表示し、peak densityの75%以上はredで飽和させる。kernel外の
  density 0は透明。表示切替はDSP結果と
  EVMを変更せず、manual Configと終了時Configへ保存する。R&S FPL K70 pp.31-34の
  density traceと同じ「出現頻度を色で示す」表示契約だが、bin数・log scale・color
  map・Gaussian kernelはPluto VSA固有とする。
- Carrier Frequency Drift補正はDemodulation設定から切り替えられるが、実測cross-validation未完了のため既定OFF。CFO補正はModulation SignalのMeasured表示で常に適用する。
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
- Result Summary項目は`pluto_sa/vsa/result_summary.py`の安定した内部ID、表示名、Common/PSK/FSK/Diagnostics分類、対応modulation family、実装状態、既定表示を唯一の定義元とする。Result Summary右クリックの階層check menuと`Meas Config > Result Summary`のcheck treeは同じ選択setを共有する。`Show All`、`Measurement Results Only`、`Diagnostics Only`、`Restore Defaults`を備え、選択IDは手動Configと終了時Configへ保存する。旧Configのsection欠落時は既定へ戻し、未知の将来IDは無視する。R&S項目の未実装分は`Not implemented`として表示するが選択不可とし、同期用`Sync EVM RMS`/`Frequency Fit RMS`を正式な`EVM RMS`/`Frequency Error RMS`と混同しない。
- 既定表示はCommonのModulation/Power/Carrier Frequency Error、PSKのEVM RMS/Symbol Rate Error、FSKのFrequency Error RMS/FSK Meas Deviation/FSK Deviation Error/Carrier Frequency Drift、DiagnosticsのPattern Symbols Correct/IQ Correlation/Selected Result/Result Symbols/Pattern Errorとする。PowerはResult Range解析dataのdBmをlinear powerへ戻して平均する。FSK Deviation Errorは`measured-reference`をHz、Carrier Frequency DriftはHz/Symで表示する。Frequency Error RMSは現行FSK frequency-model residualをmeasured deviationで正規化した開発値であり、規格適合値ではない。
- Symbol PlotはPSK時にConstellationを表示する。FSK時は1 symbol期間の位相差分をI/Q平面へ表示する既存方式と、復調器のsymbol-frequencyをR&S `Constellation Frequency`同様に縦一列へ表示する方式を切り替えるDock Widget。後者の縦軸はModulationと同じReference Deviationの±150%としFlat/Densityの両traceに対応する。
- 初期geometryは各列幅と各行高を均等化する。ユーザーが移動・resizeした後はQt dock layoutに従う。

### Capture内の複数pattern

2026-08-06にFSK/PSK共通の複数候補列挙を実装した。threshold以上のcorrelation local peakを列挙し、隣接timing phaseで同一packetを重複検出したものは物理候補1件へ統合する。2026-08-07にUIの`First`/`Strongest`/`Last`/`Match Index`選択を廃止し、eligible候補を常にcapture時刻順・1始まりで管理する方式へ統一した。ファイルload、generated recording、Pluto新規取得では必ずindex 1へ戻り、同じIQに対するRefreshや設定変更後の再解析では現在indexを維持する。Result Summaryの`Selected Result`は`selected / eligible`件数を表示し、metadataには検出件数も残す。`Sweep / Run > Previous/Next Result Range`または左右矢印で、IQを再取得せず同じcapture内の前後候補へ切り替える。端では停止する。解析結果そのものは引き続き選択した1件だけを保持する。選択indexはcapture固有の一時状態でConfigへ保存せず、旧Configの`match_selection`/`match_index`は読み込み時に無視する。

`Meas only if Pattern Symbols Correct`がONの場合、全相関候補をsymbol判定してからpattern symbol errorが0の候補だけをeligible候補にする。先行する誤相関候補があっても、後続の最初の完全一致候補がindex 1となる。OFFの場合は誤りを含む候補も時間順の一覧へ残すが、その候補で探索を打ち切らず、後続の完全一致候補も左右矢印で選択できる。`detected_match_count`は相関候補総数、`eligible_match_count`はResult Range条件とsymbol correctness条件を適用した後の候補数を表す。ONで完全一致候補が0件の場合は以前のPattern Resultとrange overlayを画面から消し、`Pattern Error`を表示する。

`Result Range > Exclude incomplete Result Range`をONにすると、capture端またはFSK burst端までに指定Result Lengthを確保できない候補を選択前に除外する。OFFは従来互換で、選択候補の取得可能なsymbolだけを返す。除外後に候補がない場合はpattern analysis error、選択indexは除外後のeligible候補に対する番号とする。

R&S manual pp.116-117、143-145では、Burst Search有効時は各burst内の最初のpatternを検索し、複数の離散Result Rangeをcapture buffer上に持ち、`Select Result Rng`で現在範囲を切り替える。今後は単純なcorrelation local peak列挙ではなく、Burst Searchでpacket候補を分離してから各burstの最初のpatternを対応づけ、available Result Rangesとselected Result Rangeを別管理する。この段階でNext/Previousまたはcapture上のrange選択UIを追加する。

現在定義済みのmodulation kind:

- FSK（現時点の復調orderは2。将来の多値FSK拡張を想定して名称にはorderを含めない）
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
- Pattern Search成立時のPSK EVM RMSは、Constellationへ渡す最終
  `PatternSearchResult.measured_symbols`と、decode結果に対応するideal alphabetを
  同じResult Range・1 point/symbolで比較する。式は
  `100 * sqrt(sum(|measured-reference|^2) / sum(|reference|^2))`で、R&Sの
  Mean Reference Power正規化に対応する。同期最適化用の`Sync EVM RMS`とは別値。

## 4. Source

- `GeneratedIQSource`: FSK/GFSK/PSK test waveform。seed固定に対応。
- `FileIQSource`: R&S `.iq.tar`、`.npy`、`.npz`、raw complex file。
- `recording_from_acquisition`: 共通Pluto acquisition record adapter。
- `PlutoLiveSource`: 共通`PlutoReceiver`とtrigger/acquisition contractを使うfinite live source。

`.npz`は`iq`、`sample_rate_hz`、`center_frequency_hz`、`usable_bandwidth_hz`、振幅補正条件を保存/復元できます。旧NPZにusable bandwidthがない場合は`0.8 * sample rate`をfallbackにします。`.npy`とraw IQはUIでsample rateを指定します。SigMFとSCPI instrumentは未実装です。

### Pluto live Run Single

R&S VSAの`Input / Frontend`、`Signal Capture > Data Acquisition`、`Sweep / Run`の区分に合わせ、Pluto finite captureを追加した。`Run Single`（F6）はGUI threadとは別のcapture threadでPluto接続、設定、IQ取得を行い、完了したimmutable `IQRecording`だけをmain threadへ返す。`Refresh Analysis`（F5）は従来どおり現在のcaptureを再解析し、再取得しない。

### Analysis latency optimization

2026-08-07に、測定定義を変えない第1段のlatency最適化を追加した。Analysis Center/BandwidthのDDC/FIR/decimationとoptional DC removalは、従来のbase analysis用とpattern analysis用の二重実行を廃止し、1回だけ生成した同一`IQRecording`を両処理へ渡す。実測Pluto fixtureではcore全体が概ね50 msから40 msへ短縮した。

I/Q Power Triggerで得た複数burstは、timing、CFO、drift、pattern correlationをburstごとに従来どおり独立推定する。別burstの推定値やsymbol clockを共有せず、候補の時系列順序、selected index、Result Range制限も変更しない。FSK 9 burstの実測相当benchmarkではPython worker並列がNumPy内部処理と競合して直列より遅く、Pattern成立後の3結果解析もthread生成costを回収できなかったため、どちらの並列化も採用していない。

`VSASession.analysis_timings_ms`は`preprocess`、`base_analysis`、`pattern_search`、`post_prepare`、`post_analysis`、`total_dsp`を保持する。GUIはoffline解析完了時に`DSP`/`Display`、Pluto Run Single完了時に`Capture`/`DSP`/`Display`/`Total`の経過時間をstatus barへ表示する。これは性能診断値でありResult Summaryの測定結果ではない。Capture時間は別threadで計測しDSP値へ含めない。

2026-08-08にDSP解析をGUI threadから専用`_AnalysisThread`へ分離した。GUI threadはcontrol値からrevision付き`VSASession.analysis_snapshot()`を作るだけで、channel extraction、pattern search、復調、各result解析はworker内で実行する。完了時にrevisionが一致する結果だけをmain sessionへpublishし、plot/table生成だけをGUI threadで行う。解析中にRefresh、Result Range移動、新規IQ load/capture等が複数要求された場合、現在のworkerを並列実行せず、途中の待機要求を置換して最新1件だけを次に実行する。古いgenerationの完了結果は描画しない。このため操作受付をDSP時間から分離しつつ、異なる設定の結果が新しい画面へ巻き戻る競合を防ぐ。アプリ終了時は実行中workerを破棄せず、解析完了後の終了を要求する。

2026-08-18に長時間captureのDisplay処理をbounded化した。PSK IQ Trajectoryの表示用resample/RX filterはcapture全体ではなく、選択Result Rangeの前後16 symbolsをguardとして加えた区間だけへ適用する。guardは現行10-symbol SRRCとpolyphase resamplerの過渡領域をResult Range外へ置くためで、表示区間内のfilter出力とsymbol振幅正規化は全capture処理時と同じに保つ。IQ PowerとFSK Instantaneous Frequencyは最大約30,000 plot pointsへ制限するが、単純strideではなく各時間bucketのmin/maxを時系列順に残すpeak-preserving decimationを使い、短いpower dipやfrequency excursionを表示から落としにくくする。Pattern Search失敗時など全captureのsymbolがfallback表示される場合に備え、Power/Modulation上のsymbol pointsは最大2,000点、PSK IQ Trajectoryは最大10,000点、画面上のSymbol Tableは先頭1,000 symbolsへ制限し、table更新中の再描画も停止する。測定演算、Result SummaryおよびSymbol Table export dataは間引かず全数を保持する。

Pluto Run Singleはreceiver/USB contextをsource lifetime中再利用する。2026-08-07に、取得設定が前回と完全に同一ならhardware reconfigureを省略した。従来は各Runで標準bufferへ戻した直後にrecord lengthへ再変更しており、同一設定の反復測定でも不要な二重設定が発生していた。Center、sample rate、RF bandwidth、gain、record length等の`SpectrumConfig`が変われば従来どおり再設定する。初回だけはUSB context生成とPluto属性設定を含むため、後続RunよりCapture時間が長い。この初期化時間は測定DSPの差ではない。

2026-08-18に、同一pyadi RX bufferをRun Single間で維持すると、DSP/display中にIIO/kernel側へ滞留した過去sampleを次回Runで先に読み出すことを実機で確認した。finite acquisitionの時刻基準を「Run要求後」に戻すため、VSA Pluto Singleだけは各read直前に`rx_destroy_buffer()`を行い、同じsample countであっても新しいIIO RX bufferから取得する。receiver/USB contextと変更のないhardware属性は引き続き再利用する。これはcapture連続性より鮮度を優先するSingleの契約であり、共通streamやSweepの既定captureには適用しない。buffer再生成分だけCapture時間が増える可能性がある。

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

`Swap I/Q`はR&Sと同じ`Q + jI`変換をcapture後に適用する。Free Run / Singleに加えて、Pluto Run SingleのI/Q Power Triggerと符号付きTrigger Offset（negativeはpretrigger、positiveはtrigger後から開始）を共通trigger recorderへ接続した。Continuous、Stop、trigger rearmは次段階で同じ共通streamへ接続する。

### VSA Power Trigger検討

`PlutoLiveSource.capture_single()`は共通`TriggerAcquisitionController`を経由し、Free RunまたはI/Q Powerを選択できる。Levelは最終表示と同じdBm基準から共通補正を逆算してdBFS detectorへ渡し、Slope、Hysteresis、符号付きTrigger OffsetをVSA UIに公開した。返却record長はTrigger Offsetによらず一定とする。R&S K70と同様にtrigger timeoutは設けず、待機中に同じRun Single操作を再度実行するとoperator cancellationを要求する。

Power TriggerをRising edge、適切なlevel、固定pre-trigger positionで使用すれば、各captureの時間原点をpacket立ち上がりのlevel crossingへ合わせられるため、Free Runよりpacket開始時刻を大幅に揃えられる。ただし同期するのはprotocol上のpacket先頭ではなく、RBW/IQ filter後のpower envelopeがthresholdを横切ったsampleである。送信ramp、packetごとのpower差、noise、interference、filter group delayにより数sample程度以上のjitterは残り得る。正確なsymbol境界とpattern startは引き続きPattern Search/Symbol Synchronizationで決定し、Power Triggerはcapture内のおおまかな位置合わせと必要pre-trigger量の削減に使う設計とする。

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

### Post-capture I/Q Power Trigger / Pattern Search Gate

2026-08-07 に、Pluto入力とファイル入力に共通のpost-capture I/Q Power Triggerを実装した。キャプチャ全体から全てのrising power eventを検出し、各active interval内の最初の有効patternを時系列Result Range候補にする。LevelはIQ Power traceと同じdBm換算、再trigger制御はHysteresis、Drop-Out、Holdoff、検索開始位置は符号付きSearch Start Offset（symbols）で設定する。新規IQでは先頭候補、Refreshでは現在Indexを維持し、既存の左右キーで候補を切り替える。詳細な演算・既定値・R&Sとの差分は[vsa-iq-power-trigger.md](vsa-iq-power-trigger.md)を参照。

Burst終端制限ではlinear envelope powerを既定1 symbolで平均し、Hysteresis/Drop-Out成立位置をfilter delay補正してfalling edgeとする。`Limit Result Range to Active Interval`がONなら、pattern search用local waveformを復調前にそのedgeで制限し、復調・PSK振幅正規化・EVM/frequency error・Symbol Plot/Tableの母集団を同じactive symbol列へ統一する。そのうえでedgeより後まで続く不完全symbolをResultから除外する。2026-08-18以前は設定Result Lengthで正規化/EVMを計算した後に表示配列だけをburst長へ切り詰めていたため、3500 symbols指定を699 symbolsへtrigger制限した場合などにPSKクラスタが過大表示される不具合があった。OOKはvalid zero runと無信号をpowerだけで一意に区別できないため、最大zero runより長いDrop-Outを設定するか終端制限をOFFにする。

Plutoのacquisition I/Q Power Triggerは別途実装済みであり、本機能は引き続き1キャプチャ内の複数packetを全件評価するpost-capture Burst Searchである。両者を同じ設定・同じ処理として扱わない。

### FSK Natural Mapping polarity

2026-08-07以降、VSA Pattern Searchの`Natural` mappingは`0 = negative frequency deviation`、`1 = positive frequency deviation`を固定する。既知patternとの相関が逆極性で成立しても自動反転せず、候補から除外する。これによりpattern内容の誤りやI/Q inversionがSymbol Table上で暗黙に隠れない。低レベルBluetooth BR protocol profileには明示的なlegacy polarity探索を残すが、汎用VSAの`PatternAnalyzer`からは無効化している。保存patternはAdvertising Access Address `0x8E89BED6`用を`LE1M_ADV`/`LE2M_ADV`、Access Address `0x71764129`用を`LE1M`/`LE2M`とする。いずれもPreambleとAccess AddressのLSB-first OTA sequenceを保持する。保存済みLE ConfigはAdvertising用sequenceを内包するため、pattern名を`LE1M_ADV`/`LE2M_ADV`とする。

2026-08-07にFSK専用の明示的な`Allow Inverted Pattern Match`を追加した。ONでは設定patternとbitwise complementを同一correlation探索の正負仮説として許可するが、復調decisionやInstantaneous Frequencyの極性は反転しない。反転一致時もSymbol TableはNatural mappingの実測bitを表示し、metadata/Result Summaryへ`Pattern Match = Inverted`を記録する。これは以前の暗黙的なmapping反転を復活させるものではない。PSKは位相回転・複素共役とbit反転が変調次数/mappingごとに異なるため対象外とする。

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
- PSK EVMは最終同期・補正後のConstellation点と同じsymbol列を使用し、表示との
  内部整合性を保証する。現状はcurrent Result Range、1 point/symbol、Mean Reference
  Power正規化に固定される。R&S相当の独立Evaluation Range、Display Points/Symbol、
  Optimization、Measurement Filter、`Normalize EVM to`選択は未実装のため、規格適合値
  またはR&Sとの数値同等性はまだ保証しない。
- 現PSK SRRCは8 samples/symbol、betaはSignal DescriptionのRolloff（未指定時0.4）、
  10 symbol spanの有限FIRである。coarse CFO補正はSRRC前に適用するが、resample_polyの
  anti-alias filterとAnalysis Bandwidth channel filterよりは後段である。
- FSK error metricsはfrequency error/deviationの基礎のみです。
- 一般VSA用pattern searchとpost-capture I/Q Power Trigger gateは実装済み。自動thresholdを持つ独立Burst Search、DECT profile、negative Result Range offsetによるpattern前symbol復調は未実装。
- VSA UIはPluto Free Run / Run Singleとoffline Refreshに対応。Pluto取得とDSP解析はそれぞれ専用threadで実行し、GUI threadは結果の描画だけを行います。
- Pluto acquisition Power Trigger（Run Single）は実装済み。Continuous/rearm、SCPI sourceは未実装。取得済みIQ内のmulti-event Burst Searchも実装済み。
- Composite解析coreは動作しますがUIからsegment設定・表示はできません。

Bluetooth BRについてはAccess Code相関、GFSK timing/CFO/drift補正、Header rate 1/3 FEC、whitening、HEC、field抽出、DH1 Payload/CRC、PRBS-9照合までcore実装済みです。任意LAPのAccess Codeを生成でき、保存IQ解析CLIとPluto finite capture CLIがあります。2026-08-03にスマートフォンのInquiryをPlutoで実測し、4 MSPS狭帯域captureからGIAC 68 bitを相関0.9979、0 bit errorで復元しました。さらに固定2441 MHzのBR test waveformを16 MSPSで取得し、通常Access Code、Header FEC、DH1 27-byte body、PRBS-9 216 bitを0 bit errorで復元しました。このtest waveformはUAP `0x6B`のHECとPayload CRCが一致せず、Whitening OFFかつcheck初期値が別設定の可能性があります。16 MSPS全帯域への直接相関は行わず、ユーザー指定Analysis Center/Bandwidthで1 channelを抽出してから復調します。詳細値は[vsa-bluetooth-br.md](vsa-bluetooth-br.md)を参照してください。

現在の数値を規格適合判定やR&SとのEVM比較へ使用してはいけません。

## 7. 次の推奨実装順

1. Pluto acquisition I/Q Power TriggerをContinuous/Stopとtrigger rearmへ拡張。
2. pattern前を含むnegative Result Range offsetを実装。
3. pulse shaping/matched/measurement filter contractとPSK symbol-rate recoveryを追加。
4. pattern検索結果からFSK→PSKのsegment boundaryを相対指定し、EDRを汎用Composite解析する。
5. EVM用の独立Evaluation Range、Display Points/Symbol、Optimization、Measurement
   Filter、normalization選択を接続。
6. SigMF、SCPI sourceとiq-tarのUI channel selectorを追加。

実機接続前に、生成waveformへCFO、timing offset、AWGNを注入したpytestを追加し、推定器の許容誤差を固定してください。

## 8. 将来構想: 未知信号のフリーラン解析

既知patternなしでも、ユーザー指定または候補探索したmodulation、symbol rate、TX
filterが信号に合えばsymbol同期と波形復調は可能。ただし現行の高精度Fine Syncは
既知patternがpacket位置、timing、CFO/phase、FSK polarityまたはPSK rotationを拘束
しているため、通常解析の固定Timing Offsetだけでは安定しない。

段階的な実装方針:

1. Pluto Power Triggerとpre-triggerを使ってburstを取り込み、capture内ではpowerの
   hysteresis/holdoffを持つBurst Searchで実際の開始・終了とResult Rangeを決める。
2. modulation、symbol rate、TX/RX filterをユーザーが指定する`Free Run Demod`を追加。
   PSKはGardner等のdata-aidedでないtiming recoveryとCostas/Viterbi-Viterbi系carrier
   recovery、FSKは瞬時周波数波形のtransition/eye metricとdecision-directed frequency
   modelでfractional timing、CFO、deviationを推定する。
3. 設定候補をgrid searchし、eye opening、timing confidence、cluster separation、
   PSK EVM/DEVM、FSK frequency residualなどを共通Quality Scoreとして順位表示する。
4. symbol rateのcyclostationary/スペクトル推定、帯域・center推定、modulation classifier
   を加え、候補生成を自動化する。自動判定値は断定せずconfidenceと代替候補を表示する。

Power TriggerはADC captureの開始条件でありsymbol boundaryを保証しない。trigger latencyを
吸収するpre-triggerとcapture後Burst Searchは別に必要。既知patternがないPSKでは絶対
carrier phase、constellation rotation、bit mapping、差動列の先頭symbolが不定になり、
FSKではmark/space polarityが不定になり得る。したがって未知信号でも波形品質評価と
相対symbol列の復調は可能だが、絶対bit値やpacket field解釈にはpreamble、codingまたは
protocol情報が別途必要。
