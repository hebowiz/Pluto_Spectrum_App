# Wi-Fi Dedicated Analyzer 現行仕様

## 対象と入力

Analysis Mode `wifi`。Non-HT OFDM / ERP-OFDM、20 MHz、6/9/12/18/24/36/48/54 Mbps。
保存IQは20/40 MS/s。Rate・Length・scrambler stateはIQから自動取得し、VSG metadataを使用しない。
VSAからVSGへの依存は設けず、`pluto_protocol/wifi`のPHY・MAC解析を共有する。

ライブ取得の標準は40 MS/s、RF bandwidth 30 MHz、Channel 6 / 2437 MHz、10 ms。
Plutoの既存有効帯域モデル`min(0.8 Fs, RF BW)`が16.25 MHzを下回る設定は拒否する。
20 MS/sはcanonical保存IQの解析用であり、この有効帯域条件ではライブ取得に使わない。
Analysis ChannelとLO Offsetは共通DDC/LPFと共通帯域検証に従う。
LO OffsetのDC回避条件を満たせない組合せを黙って許可しない。
現行の最大40 MS/s・20 MHz解析帯域ではDC回避用LO Offsetを確保できないため、ライブ取得はOffset=0を使用する。

## 取得と表示

Singleは有限取得1回、Continuousは有限取得・解析の反復。解析中は次の取得を開始しない。
連続した取りこぼしなしのRF監視や、capture境界をまたぐpacket再結合は保証しない。
Refresh Analysisは保持する元IQへ現設定を適用して再解析する。ResetはIQ・結果・表示を消し、設定を維持する。
停止・終了は取得threadと解析threadへ中断要求を送り、終了後に接続を解放する。
他モードとPluto接続を共有し、busy時はモード変更を禁止する。
Meas Configは既存の下書き編集・OK/Cancel、起動時保存、StateのSave/Recallに対応する。

上段はIQ Power / Spectrum / Result Summary、下段はModulation / Symbol Plot / Packet Analysis。
初回だけ均等化し、移動・別窓化が可能。モード間の配置保持はアプリ動作中のみ。
ウィンドウリサイズで再均等化せず、再起動時は内部配置・選択タブを初期化する。
表・dock・フォント・ステータス色は共通部品を使用する。

| 領域 | 内容 |
| --- | --- |
| IQ Power | captureのdBm対ms。初期範囲は選択packetの前後に各10%の余白を加えた範囲（capture端で制限）。未検出時はcapture全体。表示範囲内の最小・最大を残す間引きに加え、検出packet区間へ優先的に表示点を割り当てる。ズーム・パン時は元の電力配列から再選択する。選択packetのSTF/LTF/SIG/DATAを色分け |
| Spectrum | 従来のFFT振幅表示（dBm）を維持。内部Mask tabは解析IQの等価デジタルPSD（dBm/MHz）とIEEE 802.11-2024の上限線。全域判定には帯域・VBWが不足 |
| Result Summary | RF/PHY測定・PHY Decode・MAC Decode・Diagnosticsを内部modelで分類。観測不足や校正条件を測定値と独立したstatusで表示 |
| Modulation | L-SIG / DATA別tab。横軸Subcarrier Index、縦軸OFDM Symbol Index、色はEVM %。DC・pilot・nullは空白 |
| Modulation追加tab | DATA EVM / Carrier、LTF channel相対振幅 / 位相、Spectral Flatnessは横軸Subcarrier Index。DATA EVM / Symbolのみ横軸OFDM Symbol Indexを維持。Flatnessの上下線はIEEE 802.11-2024のLimit |
| Symbol Plot | L-SIG / DATA別tab。等化・CPE補正後の測定点を他モードと共通のFlat / Densityで表示。点の色・サイズ、密度処理、単位円、初期IQ範囲（±1.25）を共通化。Density SpreadはNone / Medium / Maximum |
| Packet Analysis | 共通Decode / Payload Hex / IssuesとPacket List。選択すると他5領域も追従 |

Packet Listの列順は`# | Rate | Type | SSID | Length | FCS | Power`。
BeaconのSSIDは既存のdecoded summaryから表示し、SSIDなしは`—`、空文字SSIDは`(empty)`で区別する。
SSID列を主な可変幅とし、空白のない長いSSIDもQtの列幅依存の折り返しと行高再計算で表示する。
元のSSID文字列は変更せず、Tooltipで全文を確認できる。

IQ Powerの背景は最大2048 bucket、可視packetへ合計32768 bucket（1 packet最大8192）、
選択packetには16384 bucketを割り当て、各bucketの最小・最大の実サンプルと区間境界を残す。
サンプル数が割当点数以内のpacketは全点表示する。末尾の端数bucketも保持する。
非等間隔の表示点を描画側で再間引きせず、View Allは拡大後もcapture全体へ戻す。
初期範囲は`measurement_chrome.packet_time_view_range_ms`で計算し、DECT・Bluetoothと計算責務を共有する。
規格側はpacket区間と必要な最小余白、capture端で制限するかを渡し、10%の計算を重複実装しない。
DECTの最小余白、Bluetoothのcapture外余白を含む既存動作は維持する。
Wi-Fiの手動ズームは共通`PersistentPlotRanges`でpacket先頭からの相対範囲として保持し、
packet選択・再描画でも追従する。PlotのResetは選択packetの初期範囲へ戻す。
未検出時のcapture表示と検出時のpacket表示は別contextとして扱う。
未検出packetも背景のピークを保持し、拡大すると元配列から細部を復元する。
これは表示専用の処理で、取得IQ・電力測定・EVMには影響しない。

Symbol PlotのFlatマーカーは共通サイズ6 px。順序が単調でないI/Q点を失わないよう、
時系列用の自動間引き・データ範囲省略は無効にし、共通の点・密度描画を使用する。

PowerとSpectrumのApply Analysis Bandwidthは個別にcapture / analysis-channel面を選ぶ。
packet powerはactive PPDU内の線形電力平均、peakは同区間の最大値。ERPの無送信6 µsは含めない。
IQRecordingのfull scale・calibration・input correctionを使用する。未校正入力はその旨を表示する。
内部解析結果のpowerはanalysis面、画面Summaryのpowerは選択した表示面の値とする。

## 受信・測定

16 sample遅延のSTF自己相関を96 sample窓で評価し、しきい値0.75以上が24 sample以上続く場所を候補とする（20 MS/s座標）。
LTFの2回の相関で確認し、L-SIG validityで復調条件を確定する。定常単一トーンや雑音だけをpacketとしない。
正常開始を確認できた候補は、L-SIG異常やDATA欠落でも解析可能な結果とIssuesを残す。
1 captureの解析上限は128 packet。先行packetの区間内にある重複候補は除く。

STF反復の位相差からcoarse CFO、LTF反復からfine CFOを推定する。
LTF2本のFFT平均を既知系列で割り、52本のH[k]を得てzero-forcing等化する。
SIGNAL/DATAごとに4 pilotからCPEを推定して補正し、残差を別に保持する。
独立したdemap/deinterleave/depuncture/Viterbi/descrambleでPSDUを復元する。
Symbol Clock Frequency Errorは未実装につきValue=N/A、Result=Not Measuredとし、CFO等から推測しない。

### 診断用data-tone EVMの定義

L-SIGとDATAを分け、各OFDM symbolの48 data subcarrierを測る。
`error = measured_equalized - nearest_ideal`、全理想constellationの平均電力を1に正規化する。
`EVM RMS = 100 sqrt(mean(|error|²))`、`EVM Peak = 100 max(|error|)`。
packetごとの追加gain fitはしない。pilotは別の既知BPSK基準で補正後のRMS residualを%表示する。
carrier別・symbol別EVMとCPE配列を保持する。FCS不良とPHY測定可能性は区別する。

これは単一packet・data-toneの診断測定。既存plotの意味は変更しない。

### RF/PHY measurementと判定

`MeasurementResult`はID、kind、value/unit、limit/status、standard_reference、
measurement_conditions_satisfied、canonical_id、default_visible、conditions/metadataを保持する。
kindはRF/PHY Measurement、PHY Decode、Packet/MAC Decode、Diagnosticsの4種。
statusはPASS / FAIL / Info / Not Measured / Insufficient Data / Not Available。
Summaryのtooltipで分類・参照版・判定できない理由を確認できる。

IEEE Std 802.11-2024本文の§17.3.9、§18.4.7を照合済み。
2.4 GHz ERP-OFDM / 5 GHz Non-HT OFDMを分離し、未知bandへ値を推測しない。
測定可能で、必要な観測数・帯域・測定系条件が満たされた項目だけPASS/FAILを表示する。
Meas ConfigのMeasurement Conditionsで、受信周波数基準、受信精度と有線経路、random test data、
non-VHT DUTを個別に確認する。初期値はすべて未確認。通常設定と同様にOK/Cancel・起動時保存・State Save/Recallへ対応する。
これらは使用者による測定系条件の申告であり、IQの自動校正やDUT能力の自動認識ではない。
測定系・信号源を変えた場合は設定を見直す。確認済み条件と未成立理由はResultのmetadata/tooltipへ記録する。

- Relative Constellation Error: DATAの48 data tone誤差と4 pilot誤差から、52 toneのpacket RMSを算出。
  同一capture・同一decode rate・16 DATA symbols以上のPHY測定可能packetで集計する。
  Eq.(17-28)の印刷式に従いpacket RMSを等重み平均。長いpacketを点数で重くしない。
  20 packet未満はInsufficient Data。ランダムpayload・受信系条件が未確認ならNot Measured。
  Table 17-20のLimitは6/9/12/18/24/36/48/54 Mbpsに対して−5/−8/−10/−13/−16/−19/−22/−25 dB。
  対象packetがないときは選択packetの値を表示し、scopeをmetadataに保持する。
  dB=`20 log10(rms)`、EVM RMS %=`100 rms`で同じcanonical値を変換し、EVM行のResultはInfo。
- Carrier Frequency Error: Wi-Fi STF/LTFのcoarse+fine CFOをHzで保持。ppmで比較し、5 GHzは±20 ppm、ERPは±25 ppm。
  受信周波数基準の確認とL-LTF同期成立が必要。
- Spectral Flatness: CFO補正後の2 LTF FFTのtone別平均energy。等化前のinner tone平均に対する偏差。
  outer toneも同じinner平均を基準にする。inner ±1…16は±4 dB、outer ±17…26は−6/+4 dB。
  tone別energy・上下Limit・最小marginを保持。受信応答と有線経路の確認、52 toneを覆う帯域が必要。
- Center Frequency Leakage: 2 LTFのDC平均energyを、active 52 tone + DCの合計energyに対してdB化。
  任意のcapture FFTのDCを読む測定ではない。§17.3.9.7.2の上限は`max(P−15, −20) dBm`。
  Pはchannel-training区間の合計電力を既存振幅補正で送信基準面へ換算する。
  校正済み振幅なら相対Limitへ変換して両分岐を評価する。未校正では−15 dB以下を確認できるが、
  それを超えた値は−20 dBm例外を評価できないためFAILとせずNot Measured。
  受信DC・伝搬路条件とnon-VHT DUTの確認が必要。VHT STAはNon-HT送信でも§21.3.17.4.2が適用され、
  そのRF LO位置/RBW測定は対象外のため、この確認なしに旧来Limitを適用しない。
- Transmit Spectrum Mask: active PPDUをCFO補正し、Hann Welch（ENBW 100 kHz、50% overlap、線形power平均）でPSDを取得。
  30 kHz instrument VBW/detectorは未再現。20/40 MS/sでは±30 MHz全域も不足するためInsufficient Data。
  最小margin・違反位置・観測域・違反binを保持。正負両側ともupper emission limitであり、lower maskは設けない。
  ±9/11/20/30 MHzで0/−20/−28/−40 dBr。±30 MHz以遠は校正済み振幅に限り−39 dBm/MHzとの大きい方を適用する。
  絶対値例外を20–30 MHzの傾斜へ拡張しない。地域規制の追加maskは評価せず、正式認証判定ではない。
- Packet Powerは既存calibrationを使うactive PPDU平均、Limit=— / Info。
  Peak Power、EVM Peak、Pilot Error RMS、48-tone EVM、CPE、同期指標はDiagnosticsで、通常非表示。
  Displayの追加結果チェックで詳細DecodeとDiagnosticsを表示する。

Continuousはcaptureをまたいで測定統計を累積しない。Refreshで同じ録音を重複加算しない。
FCS不良とRF精度不良を区別し、FCS良否でPHY測定を一律に除外しない。
各Resultの条項・試験・既知制限は[測定検証報告](../../../verification/vsa/wifi/measurement-review.md)を参照。

## Packetと統計

Preamble Detected / L-SIG Complete / Parity / DATA Complete / PSDU Complete / FCS Validを個別に保持する。
detected / complete / measurement eligible / decode success / FCS validの件数を区別する。
Frame Controlのversion/type/subtype/flags、Duration、Address、Sequence/Fragment、Frame Body、FCSを解析する。
Beacon固定値と既知IEを表示し、未知IEはID・Length・Raw Valueを残す。
Bit RangeはL-SIG logical / DATA logical / PSDU logicalの座標であり、IQの連続sample区間へ変換しない。

## 制限

DSSS/CCK、HT/VHT/HE、MIMO、暗号復号・MAC再構成、sample clock追従は対象外。
40 MS/sは2:1でnative OFDM座標へ変換する。帯域外妨害波・強いDC・大きなSFO・長いdelay spreadなどへの耐性は未保証。
DATA欠落時はPHYまでの結果を返し、途中PSDUの部分Viterbi復元は行わない。
実RF受信・SMCV・市販APの受信品質は[実機手順](../../../verification/vsa/wifi/non-ht-hardware.md)で別に確認する。
