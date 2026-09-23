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
| IQ Power | capture全体のdBm対ms。表示範囲内の最小・最大を残す間引きに加え、検出packet区間へ優先的に表示点を割り当てる。ズーム・パン時は元の電力配列から再選択する。選択packetのSTF/LTF/SIG/DATAを色分け |
| Spectrum | 選択packetのactive区間。絶対RF MHz対dBmのFFT振幅表示。PSDではない |
| Result Summary | rate/coding、power/peak、CFO、L-SIG、EVM、pilot error、MAC種別、FCS。測定値はInfo、parity/FCSはPASS/FAIL |
| Modulation | L-SIG / DATA別tab。横軸Subcarrier Index、縦軸OFDM Symbol Index、色はEVM %。DC・pilot・nullは空白 |
| Modulation追加tab | DATA EVM / CarrierとLTF channel相対振幅 / 位相は横軸Subcarrier Index。DATA EVM / Symbolのみ時間方向の評価のため横軸OFDM Symbol Indexを維持 |
| Symbol Plot | L-SIG / DATA別tab。等化・CPE補正後の測定点を他モードと共通のFlat / Densityで表示。点の色・サイズ、密度処理、単位円、初期IQ範囲（±1.25）を共通化。Density SpreadはNone / Medium / Maximum |
| Packet Analysis | 共通Decode / Payload Hex / IssuesとPacket List。選択すると他5領域も追従 |

IQ Powerの背景は最大2048 bucket、可視packetへ合計32768 bucket（1 packet最大8192）、
選択packetには16384 bucketを割り当て、各bucketの最小・最大の実サンプルと区間境界を残す。
サンプル数が割当点数以内のpacketは全点表示する。末尾の端数bucketも保持する。
非等間隔の表示点を描画側で再間引きせず、View Allは拡大後もcapture全体へ戻す。
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
Symbol Clock Errorは未実装につきNot Availableとし、推測値を出さない。

### EVMの定義

L-SIGとDATAを分け、各OFDM symbolの48 data subcarrierを測る。
`error = measured_equalized - nearest_ideal`、全理想constellationの平均電力を1に正規化する。
`EVM RMS = 100 sqrt(mean(|error|²))`、`EVM Peak = 100 max(|error|)`。
packetごとの追加gain fitはしない。pilotは別の既知BPSK基準で補正後のRMS residualを%表示する。
carrier別・symbol別EVMとCPE配列を保持する。FCS不良とPHY測定可能性は区別する。

これは単一packet・data-toneの診断測定であり、52 toneや複数frameの条件を含むIEEE RF適合試験の代替ではない。
根拠のないEVM/CFO/power Limitを設定しない。規格との関係は[検証報告](../../../verification/vsa/wifi/non-ht-review.md)を参照。

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
