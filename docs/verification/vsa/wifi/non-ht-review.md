# Wi-Fi Dedicated Analyzer 実装・検証報告

RF/PHY Result追加後の仕様・IEEE 802.11-2024本文の照合結果・測定別の制限は
[測定追加の検証報告](measurement-review.md)を参照。以下の実装初期レビューと区別する。

## Architectureと共有化

既存working treeのVSG独立受信器`pluto_protocol/wifi/non_ht.py`を再利用した。
VSAはIQRecordingのみを入力とし、`pluto_vsg`をimportしない。
共通PHYへ開始候補指定と任意の測定配列取得を追加し、VSGの既存IQ Verify APIは維持した。
新規`pluto_protocol/wifi/detection.py`がcapture全域からSTF候補を探す。
MACのflags・Sequence子要素・未知IEの構造は共有decoderへ追加した。

VSA固有処理は`pluto_vsa/protocol_modes/wifi/`の以下へ分離した。

| ファイル | 責務 |
| --- | --- |
| `model.py` | capture/packet/region結果、個別integrity、各種件数 |
| `analysis.py` | 複数packet検出、共通PHY呼出し、測定結果構築 |
| `measurement.py` | 等化後のEVM、packet power |
| `acquisition.py` | 中断可能な解析worker。取得は既存PlutoSingleCaptureThread |
| `summary.py` | 共通Summary用の値・判定。RF Limitは推測しない |
| `configuration.py` | ReceiverSetupControls / TriggerControls / HierarchicalMeasConfigDialog |
| `ui.py` | 6 dock、packet選択、取得・停止・終了、plot更新 |

Shell、control panel、全mode menu、設定persistenceに`wifi`を追加した。
共通PacketDecodeTabsはWi-Fiのlogical stream列だけを分岐し、他規格の表示は維持する。
共通plot/range/table/fontを再利用し、Bluetoothの巨大workspaceをコピーしていない。

## 同期・復調・測定方式

STFの16 sample周期を正規化自己相関で探索し、LTFペアの相関で時刻を精密化する。
STF位相差でcoarse CFO、LTFペアでfine CFO。H[k]は2本のLTF FFT平均と既知系列の比。
52 subcarrierをZF等化し、4 pilotの平均位相で各symbolのCPEを補正する。
pilot residualは補正後の値であり、補正前のCPE角度とは別配列にする。
L-SIGからrate/lengthを決め、独立Viterbi等を経てPSDU/FCSを得る。

EVMは48 data subcarrierを対象とするnearest-point / decision-directed測定。
BPSK/QPSK/16QAM/64QAMそれぞれを公称平均constellation power=1へ正規化し、RMS/Peak、carrier別、symbol別を計算する。
L-SIGとDATAは分離し、pilot residualは別集計。SFO推定は未提供でNot Available。

参照した[IEEE 802.11a-1999 §17.3.9.7](https://pdos.csail.mit.edu/archive/decouto/papers/802.11a.pdf)は、
同期・周波数補正・channel推定・pilot補正後に最近傍点との誤差を集計する測定順序を記載している。
同節の52 toneと複数frameを含む適合試験条件は今回の48 data-tone単一packet診断とは異なる。
[KeysightのEVM解説](https://www.keysight.com/blogs/en/tech/rfmw/2022/09/29/take-charge-of-your-evm-measurements)でも正規化基準の区別を確認した。
[MathWorks Non-HT受信API](https://www.mathworks.com/help/wlan/ref/wlannonhtdatarecover.html)の等化symbol/CPE出力も設計比較に使用した。
これら外部製品との実行時相互比較を実施したという意味ではない。

## 表示設計

既存Bluetooth/GeneralのModulationは時間波形のI/Q軌跡、Symbol Plotは判定時刻の点・密度表示。
OFDMの時間IQは複数subcarrierの合成なので、その軌跡をQAM constellationとして使えない。

- Modulation: 横軸Subcarrier Index、縦軸OFDM Symbol IndexのEVM resource表示。DATA EVM / Symbolのみ横軸OFDM Symbol Indexを維持。
- Symbol Plot: FFT/ZF/CPE後の測定点を共通描画処理でFlat / Density表示。点の色・サイズ、Density Spread、単位円、初期IQ範囲を他モードと統一。
- 両方にL-SIG - BPSK / DATA - 実際のmodulationのtab。
- 追加表示はDATA EVMのcarrier別・symbol別、Channel Magnitude/Phase。Modulation内部に置き6 dockを維持。
- raw time-domain IQのQ/I散布図はWi-Fi constellationとして描画しない。

Packet AnalysisはL-SIG各bit、DATA SERVICE/TAIL/PAD、PSDU/MAC/Beacon/FCSを共通treeで表示する。
L-SIG logical、DATA logical、PSDU logicalのbit座標を明示し、time sample座標と混同しない。

## 自動検証

`tests/vsa/wifi/test_wifi_analysis.py`:

- 全8 rate×20/40 MS/s。PSDU全byte一致、L-SIG parity、FCS、理想IQの低EVM。
- 4変調×2sample rateで、amplitude/initial phase/85 kHz CFO/fractional delay/AWGN/短いmultipathを合成し2 packetを復元。
- known +10% gain errorのEVM=10%、pilot residual=3%。録音のpower補正値の反映。
- L-SIG parity異常、未定義RATE、不正FCS、STF/LTF/SIG/DATA途中欠落、noise/zero/toneの誤検出抑制。
- 未知IE、MAC flags、sequence/fragment、VSA→VSG import禁止。

`tests/vsa/wifi/test_wifi_workspace.py`:

- 6 dockの順序・共通table/font、L-SIG/DATA tab、複数packet選択、plot range維持。
- ModulationへのEVM/Channel表示配置、Symbol Plotへの共通コンスタレーション配置、点の色・サイズ・全測定点、Flat/Density切替後のrange維持。
- 旧Density設定の読込とDensity Spreadの起動時復元。
- Single/Continuous、停止・終了、共有source、busy mode guard。
- draft Cancel、起動時設定復元、State Save/Recall、Pluto帯域制約。
- EVM画像の軸と実データ配置、Flatマーカー実サイズ6とズーム後の全測定点保持。
- 低Duty packetの全サンプル保持、異なるcapture/analysis sample rateの座標換算、ズーム再描画、View All、Reset。

`tests/vsa/wifi/test_wifi_display.py`:

- 約200万サンプル中の1000サンプルpacketを全点保持し、残りの背景を含め表示6000点未満。
- 未検出区間もズームで全サンプルを復元。区間端点・末尾bucketのピークと元の時刻を保持。
- 128個の長いpacketでも表示105000点未満。空配列・NaN・範囲外を処理。

共通window layout testへWi-Fiを加え、リサイズ再均等化なし・モード別比率復元を確認した。
VSGのIEEE固定値・IQ Verifyも再実行した。全体回帰は **1,256 passed（234.12秒）**。
実行コマンドは`QT_QPA_PLATFORM=offscreen`で`.venv/Scripts/python.exe -m pytest -q`。
General / Bluetooth / DECT / ADS-B / VSGを含め、テスト上の回帰は検出されなかった。
最終画面確認ではMeas ConfigにOK/Cancelを明示し、IQ Powerの時間軸をms表示に固定した。
設定編集のaccept/reject/close/Escapeを共通transactionテストへ追加した。
最終UI変更後のworkspace・設定関連テストは33件すべて成功した。
最終のMAC表示階層調整後、Wi-Fi解析・workspace・VSG MAC/Verifyの関連70件も成功した。
Modulation / Symbol Plotの内容入替・共通描画への変更後、`tests/vsa/wifi`、
`tests/vsa/core/test_vsa_setup_controls.py`、`tests/common/test_window_layout.py`の90件が成功（28.47秒）。
Windows GUIでマニュアルのWi-Fi画面を再撮影し、6ペインの名称・配置、EVM表示とコンスタレーション表示を確認した。

その後のModulation軸変更・IQ Powerのpacket優先間引き／ズーム再描画・Flat表示の
時系列間引き無効化について、同じ3対象の関連テストは **95 passed（29.00秒）**。
マニュアル本文・画像・PDFは今回更新していない。次回の明示的な改訂依頼時に軸と表示動作を反映する。

## IQ Power初期範囲の共通化

Wi-Fiの初期表示を、選択packetの前後各10%の余白付き範囲へ変更した。
DECTで使用している`packet_time_view_range_ms`を共通計算元とし、BluetoothのIQ Power/FSK表示の重複計算も集約した。
Bluetoothの最小余白とcapture外まで含む既存範囲、DECTのcapture端制限と最小余白は維持する。
Wi-Fiはcapture端で制限し、手動ズームのpacket相対保持は共通`PersistentPlotRanges`を使用する。

Wi-Fi workspace、共通plot range、Bluetooth workspace、DECT Dedicatedの関連 **33件が成功（12.39秒）**。
初期範囲、packet切替、手動ズームの相対追従とReset、未検出から再検出への遷移、
capture/analysisのsample rateが異なる場合、View Allでの全体表示を確認した。
マニュアルは変更していない。

## 残る制限とRF確認

実RF試験は未実施。自動検証は生成IQと合成劣化のみで、市販AP/Pluto/SMCVの受信成功を保証する証拠ではない。
誤差の大きな捕捉、強いDC/帯域外波、SFO追従、GIを越すmultipath、capture境界再結合は未対応または未検証。
Continuousは有限captureの反復で、1 capture最大128 packet。統計はそのcapture内の件数。
高次PHYのlegacy互換preambleを検出する可能性があるが、HT/VHT/HEの復調はサポートしない。

今後もMAC/PacketAnalysisResultとworkspace chromeを共有できる。HTのtraining/DATA、DSSS/CCKの同期・復調は別PHYへ分ける。
現行仕様は[Wi-Fi仕様](../../../spec/vsa/wifi/non-ht-analyzer.md)、実機の残項目は[実機手順](non-ht-hardware.md)を参照。
