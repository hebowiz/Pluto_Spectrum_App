# VSA モード別メニュー／設定項目比較（現行実装）

確認日: 2026-09-16  
対象コード: `d3e8371`  
目的: メニュー再編のための現状棚卸し。ユーザーマニュアルや新仕様の提案ではない。

## 1. 対象と表記

対象は Pluto VSA 共通シェルの右側操作パネルから利用できる以下4モード。

- General VSA
- Bluetooth（BR / EDR、LE、HDT はこのモード内で選択）
- DECT
- ADS-B 1090ES

各ワークスペースの旧メニューバーは共通シェル内では非表示。
以下は右側ボタンを起点とした一覧であり、プロットタブ／Decodeフィールド／測定結果行そのものはメニューボタンとして数えない。
必要に応じて「表示のみ」「無効／未接続」「条件付き」を明記する。

## 2. ANALYZER SETUP ボタン比較

表示順は各列の上から下。

| General VSA | Bluetooth | DECT | ADS-B 1090ES |
|---|---|---|---|
| Analyzer Mode | Analyzer Mode | Analyzer Mode | Analyzer Mode |
| Input / Frontend | Bluetooth Analysis | DECT Analysis | ADS-B Analysis |
| Signal Description | Input / Frontend | Input / Frontend | Receiver Location |
| Signal Capture | Signal Description | Signal Description | Display |
| Trigger | Trigger | Trigger | — |
| Pattern Search | Display | Display | — |
| Result Range | — | — | — |
| Demodulation | — | — | — |
| Result Summary | — | — | — |
| Display | — | — | — |

Analyzer Mode は2行表示で、2行目に現在モードを表示。
その中の選択ボタンは全モード共通で `General VSA / Bluetooth / DECT / ADS-B 1090ES`。

## 3. 設定機能の配置比較

「なし」は独立した設定UIがないという意味であり、内部DSPがないという意味ではない。

| 機能／項目 | General VSA | Bluetooth | DECT | ADS-B 1090ES |
|---|---|---|---|---|
| Protocol / PHY / 解析Profile | Signal Descriptionで変調を指定 | Bluetooth Analysis | DECT固定 | ADS-B固定 |
| Carrier plan / RF channel | なし、数値周波数入力 | LE ChannelはBluetooth Analysisに条件付き表示 | DECT Analysis | 1090 MHz固定 |
| Capture center | Input / Frontend | Input / Frontend | DECT AnalysisのCarrierで決定 | 固定 |
| Capture length | Signal Capture | Input / Frontend | Input / Frontend | ADS-B Analysis |
| Samples / symbol | Signal Capture | Input / Frontend | Input / Frontend | ADS-B AnalysisのSample Rate |
| Capture RF bandwidth | Input / Frontend | Input / Frontend | Input / Frontend | 4 MHz固定 |
| Analysis Channel ON/OFF | Input / Frontend | Input / Frontend | Input / Frontend | 設定UIなし |
| Analysis centerの独立入力 | Input / Frontend | なし、選択centerに従う | なし、選択carrierに従う | なし |
| Analysis bandwidth | Input / Frontend | Input / Frontend | Input / Frontend | 設定UIなし |
| Analysis BWのPower/Spectrum適用選択 | Input / Frontend | Input / Frontend | Input / Frontend | なし |
| Offset LO | Input / Frontend | Input / Frontend | Input / Frontend | なし |
| Internal gain / External ATT | Input / Frontend | Input / Frontend | Input / Frontend | ADS-B Analysis |
| External gain | Input / Frontend | 設定UIなし | 設定UIなし | ADS-B Analysis |
| 変調／symbol rate／TX filter | Signal Descriptionで編集 | Signal DescriptionでPHY由来値を確認 | DECT固定値を確認 | 固定、独立確認画面なし |
| Acquisition trigger | Triggerで詳細設定 | Triggerで詳細設定 | TriggerでLevelのみ | 独立設定UIなし |
| Post-capture burst search | Triggerで詳細設定 | Triggerで詳細設定 | 内部自動処理、設定UIなし | 内部自動処理、SNR thresholdのみ設定 |
| 既知Patternの入力 | Pattern Search | RF / PHY Test設定とProtocolに従う | 内部pattern識別、任意入力UIなし | 不要 |
| Result rangeの指定 | Result Range | packet extentから自動 | packet extentから自動 | packetから自動 |
| Sync / Measurement filter設定 | Demodulation | Protocol固有処理、設定UIなし | Protocol固有処理、設定UIなし | Protocol固有処理、設定UIなし |
| Result Summary行の選択 | Result Summary | 設定UIなし | 設定UIなし | 設定UIなし |
| Receiver location | なし | なし | なし | Receiver Location |

## 4. General VSA：各設定ボタンの内容

### Input / Frontend

| 項目 | 内容／状態 |
|---|---|
| Center Frequency | Capture要求中心周波数、MHz |
| RF Bandwidth | Pluto RF帯域幅、MHz |
| LO Offset | Enable (Experimental) |
| Offset Frequency | Hardware LOへのoffset、MHz |
| Resolved LO | 実際のLO周波数、表示のみ |
| Internal Gain | dB |
| External ATT | dB |
| External Gain | dB |
| Input Correction | 補正値、表示のみ |
| Enable Analysis Channel | ON/OFF |
| Analysis Center | 解析中心周波数、MHz、独立指定可能 |
| Analysis Bandwidth | MHz |
| Apply Analysis Bandwidth to Power | ON/OFF、初期ON |
| Apply Analysis Bandwidth to Spectrum | ON/OFF、初期OFF |

Analysis ChannelとProtocol固有Measurement Filterは別の処理。
Device選択は共通 `System > Device` がユーザー向けの経路。

### Signal Description

| 項目 | 選択肢／内容 |
|---|---|
| Modulation Type / Order | FSK、BPSK、QPSK、OQPSK、pi/4-DQPSK、8DPSK、pi/4-QPSK、8PSK、16QAM |
| Symbol Rate | Sym/s |
| FSK Ref Deviation | Hz、変調に応じて有効化 |
| Modulation Mapping | Natural、Gray、Bluetooth EDR、Bluetooth HDTのmapping |
| Transmit Filter Type | None / Gaussian / Root Raised Cosine |
| Alpha / BT | filter parameter、None選択時は無効 |

### Signal Capture

| 項目 | 内容／状態 |
|---|---|
| Capture Length | 数値と単位（ms / Symbols） |
| Sample Rate | 2 / 4 / 8 / 16 / 32 / 64 / 128 samples/symbol |
| Resulting Sample Rate | 計算値、表示のみ |
| Record Length | sample数、表示のみ |
| Usable I/Q Bandwidth | 計算値、表示のみ |
| Swap I/Q | ON/OFF |

### Trigger

| 区分 | 項目 |
|---|---|
| Acquisition Trigger | Trigger Source（Free Run / I/Q Power）、Level（dBm）、Slope（Rising / Falling）、Trigger Offset（symbol）、Hysteresis（dB） |
| Post-capture Burst Search | Burst Search On、Level（dBm）、Hysteresis（dB）、Envelope Average（symbol）、Drop-Out Time（symbol）、Holdoff（symbol）、Search Start Offset（symbol）、Limit Result Range to Active Interval |

### Pattern Search

| 項目 | 内容 |
|---|---|
| Pattern Search On | ON/OFF |
| Name | Pattern名 |
| Symbol Format | Binary / Decimal / Hexadecimal |
| I/Q Correlation Threshold | % |
| Auto (90%) | threshold自動設定 |
| Meas only if Pattern Symbols Correct | ON/OFF |
| Allow Inverted Pattern Match (FSK only) | ON/OFF |
| Pattern Symbols | symbol入力table |
| Add Row / Remove Last Row | table行編集操作 |
| Load Pattern... / Save Pattern As... | Patternファイルの読込／保存操作 |

### Result Range

| 項目 | 内容／状態 |
|---|---|
| Result Length (Symbols) | symbol数 |
| Reference | 現UIの選択肢はPattern Waveformのみ |
| Alignment | Left / Center / Right |
| Offset (Symbols) | 正負symbol offset |
| Symbol Number at Pattern Start | 無効、表示軸番号の機能は未接続 |
| Exclude incomplete Result Range | ON/OFF |

### Demodulation

| 項目 | 内容／状態 |
|---|---|
| Coarse Synchronization | Auto / Detected Data / Pattern |
| Fine Synchronization | Auto / Detected Data / Patternを定義しているがUI無効。現状はCoarseのsourceに従う |
| Measurement Filter | Auto / None |
| Bit Ordering | MSB / LSB |
| Compensate for: Carrier Frequency Drift | ON/OFF、experimental、初期OFF |
| Compensate for: FSK Deviation Error | 無効、DSP未接続 |

### Result Summary

表示するResult行のcheckbox選択。変調familyに応じて対象が異なる。
treeには未実装項目も表示されるが選択不可。

| 分類 | 選択可能な項目 |
|---|---|
| Common Measurement Results | Modulation、Power、Carrier Frequency Error |
| PSK Measurement Results（QAMにも適用） | EVM RMS、Differential Symbol EVM RMS、Bluetooth DEVM RMS、Symbol Rate Error |
| FSK Measurement Results | Frequency Error RMS、FSK Deviation Error、FSK Meas Deviation、FSK Ref Deviation、Carrier Frequency Drift |
| Synchronization Diagnostics | Pattern Symbols Correct、Pattern Match、I/Q Correlation、Selected Result、Result Symbols、Pattern Error、Estimated Carrier、Display、PSK Carrier Drift、Sync EVM RMS、Fractional Timing、Frequency Fit RMS、Timing Confidence、Deviation Error (%)、Drift Model、Applied Drift |

未実装／選択不可: EVM Peak、MER RMS / Peak、Phase Error RMS / Peak、Magnitude Error RMS / Peak（PSK／FSKそれぞれ）、I/Q Skew、Rho、I/Q Offset、I/Q Imbalance、Gain Imbalance、Quadrature Error、Amplitude Droop、Frequency Error Peak。

ページ内操作ボタン: Show All / Measurement Only / Diagnostics Only / Restore Defaults。

### Display

| 項目 | 選択肢／内容 |
|---|---|
| Show Symbol Points | ON/OFF |
| Symbol Table Format | Hexadecimal / Decimal |
| IQ Power Signal | Raw Capture / Measured |
| Modulation Signal | Raw IQ / Measured |
| QAM Modulation Signal | Raw IQ / Measured、独立設定 |
| Symbol Plot Trace | Flat / Density |
| Density Spread | None / Medium / Maximum |
| PSK Symbol Plot | Physical IQ / Differential IQ |
| FSK Symbol Plot | Phase Difference / Constellation Frequency |
| Reset Plot Scales | プロットrangeを既定値へ戻す操作 |

## 5. Bluetooth：各設定ボタンの内容

### Bluetooth Analysis

| 項目 | 選択肢／条件 |
|---|---|
| Profile | RF / PHY Test / General Packet |
| Protocol | Bluetooth BR / EDR、Bluetooth LE、Bluetooth HDT |
| PHY：BR / EDR | Auto (BR / EDR 2M / EDR 3M)、BR、EDR 2M、EDR 3M |
| PHY：LE | LE 1M / LE 2M |
| PHY：HDT | Auto (HDT2 / HDT3 / HDT4 / HDT6 / HDT7.5) |
| LAP / UAP / CLK6-1 | BR / EDRのRF / PHY Test時のみ表示・入力 |
| Access Address | LEのRF / PHY Test時のみ表示。71764129固定、編集不可 |
| LE Channel | LEのRF / PHY Test時のみ表示・入力 |
| CRC Init | LEのRF / PHY Test時のみ表示。555555固定、編集不可 |
| Expected EDR RF Test Packet | BR / EDRのRF / PHY Test時のみ表示。Not configured、2-DH1 / 2-EV3 / 2-DH3 / 2-EV5 / 2-DH5、3-DH1 / 3-EV3 / 3-DH3 / 3-EV5 / 3-DH5。BR指定時は無効 |
| Whitening | BR / EDRまたはLEのRF / PHY Test時のみ表示。LE RF / PHY TestではOFF固定・編集不可 |

General Packetでは手動identity項目を隠し、自動取得する。
HDTのidentityはtraining sequenceから取得する。

### Input / Frontend

| 項目 | 内容／状態 |
|---|---|
| Center Frequency (MHz) | Capture要求中心周波数 |
| Capture Length (ms) | 取得時間 |
| Samples / Symbol | 4 / 8 / 16 / 32 |
| RF Bandwidth (MHz) | Pluto RF帯域幅 |
| Analysis Channel | Enable Analysis Channel |
| Analysis Bandwidth | MHz |
| Apply Analysis Bandwidth to Power | ON/OFF、初期ON |
| Apply Analysis Bandwidth to Spectrum | ON/OFF、初期OFF |
| LO Offset / Offset Frequency | 有効選択とMHz入力 |
| Resolved LO | 表示のみ |
| Internal Gain (dB) / External ATT (dB) | 電力補正関連入力 |

### Signal Description

Modulation (from PHY)、Symbol Rate (from PHY)、TX Filter (from PHY)、Result Range。
すべて表示のみ。PHYに従って決まり、Result RangeはAutomatic packet extent。

### Trigger

Genericと同じ2区分／同じ設定項目。

- Acquisition Trigger: Trigger Source、Level、Slope、Trigger Offset、Hysteresis。
- Post-capture Burst Search: Burst Search On、Level、Hysteresis、Envelope Average、Drop-Out Time、Holdoff、Search Start Offset、Limit Result Range to Active Interval。

### Display

Show Symbol Points、Symbol Plot Density、Density Spread（None / Medium / Maximum）、FSK Symbol Plot（Constellation Frequency / Phase Difference）、PSK Symbol Plot（Physical IQ / Differential IQ）。

独立したResult Summary行選択、Raw / Measured Modulation選択、Reset Plot Scalesボタンはこのページにはない。

## 6. DECT：各設定ボタンの内容

### DECT Analysis

Regional Carrier Plan、RF Carrier。
Modulation（GFSK, BT = 0.5）、Symbol Rate（1.152 MSym/s）は表示のみ。

### Input / Frontend

| 項目 | 内容／状態 |
|---|---|
| Capture Length | 取得時間、ms |
| Samples / Symbol | 4 / 8 / 16 / 32 |
| RF Bandwidth | MHz |
| Analysis Channel / Analysis Bandwidth | 有効選択とMHz入力 |
| Apply Analysis Bandwidth to Power | ON/OFF、初期ON |
| Apply Analysis Bandwidth to Spectrum | ON/OFF、初期OFF |
| LO Offset / Offset Frequency | 有効選択とMHz入力 |
| Resolved LO | 表示のみ |
| Internal Gain / External ATT | dB |

Center Frequencyの独立数値入力はなく、DECT AnalysisのRF Carrierで決定。

### Signal Description

Modulation（GFSK, BT = 0.5）、Symbol Rate（1.152 MSym/s）、Modulation reference（DECT measurement trace / selected display reference）。
すべて固定説明／表示のみ。ここではreferenceを変更できない。

### Trigger

I/Q Power Trigger（Level、dBm）のみ。
Source / Slope / Trigger Offset / Hysteresis / Post-capture Burst Searchの詳細設定UIはない。
取得経路ではI/Q Power、Rising、Hysteresis 3 dBを指定する。

### Display

Show Symbol Points、Symbol Plot Density、Density Spread（None / Medium / Maximum）、FSK Symbol Plot（Constellation Frequency / Phase Difference）、GFSK Modulation Reference（Measured / Window Mean / Nominal / Half Peak）。

- Measured: 観測carrierを引く。Generic packetではS-field fit、Case Aでは規格carrier測定値。
- Window Mean: 従来のpayload測定区間の平均周波数を引く。
- Nominal: 基準0 Hz。
- Half Peak: 選択区間の最大／最小周波数の中点。

独立したResult Summary行選択とReset Plot Scalesボタンはこのページにはない。

## 7. ADS-B 1090ES：各設定ボタンの内容

### ADS-B Analysis

Sample Rate（8 / 16 MS/s）、Capture Time（ms）、External ATT（dB）、Internal Gain（dB）、External Gain（dB）、Preamble SNR Threshold（dB）。

Capture centerは1090 MHz、RF Bandwidthは4 MHz固定。
Input / Frontend、Signal Description、Triggerを独立した設定ページとして持たない。

### Receiver Location

Latitude (degree)、Longitude (degree)。
ページ内操作: Select on Map... / Clear。
Local CPRのreference位置として使用する。

### Display

6つのresult paneを常時表示する旨の説明と、Reset Plot Scales操作（IQ Power / PPM）。
Symbol、Density、Raw / Measured等の選択設定はない。

## 8. SWEEP CONTROL：共通ボタンと実際の意味

| ボタン | General VSA | Bluetooth | DECT | ADS-B 1090ES |
|---|---|---|---|---|
| Continuous | Capture → Analysisを停止操作まで反復 | 同左 | 同左 | 連続capture / scan |
| Single | 1回capture / analysis | 同左 | 同左 | 同左 |
| Refresh Analysis | 現IQを再解析 | 同左 | 同左 | 同左 |
| Reset | All Packets統計をクリア、現解析を保持 | 測定履歴をクリア、現packetを保持 | 同左 | packet / aircraft履歴、table、power / PPM plot、summaryをクリア |

実行中の操作表示はStop、停止処理中はStopping...、実行中のボタンは青色。
Single実行中はContinuousを無効化。
Refresh Analysis / Resetは実行中に無効化。
Continuousが所有する内部解析中はSingleを実行中と見なさず、Continuous Stopを有効に保つ。

注意: 上記Resetはプロットのrange resetではない。

## 9. SYSTEM：共通構成

| 経路 | 項目／操作 | モード差 |
|---|---|---|
| System > Preset > Default | 既定の測定設定へ戻す | 各モードにDefaultを1つずつ保持 |
| System > Device | Connection URI、Refresh Devices | Pluto接続先を全モードで共有 |
| System > Recall | 測定設定ファイルの読込 | ファイルに記録されたmodeへ切替して復元 |
| System > Save | 現modeの測定設定ファイル保存 | 現modeのsettingsを保存 |
| System > File | 次節参照 | mode別に操作が異なる |

設定ファイル: `*.vsaconfig.json`。
Backは階層移動用の共通ボタン。設定項目ではない。

### System > File 比較

| ボタン | General VSA | Bluetooth | DECT | ADS-B 1090ES |
|---|---|---|---|---|
| Open IQ | あり | あり | あり | あり |
| Export IQ | あり | あり | あり | あり |
| Export Symbol Table | あり | なし | なし | なし |
| Export Packet List | なし | なし | なし | あり |
| Import OpenSky CSV | なし | なし | なし | あり |
| Download / Update from OpenSky | なし | なし | なし | あり |

## 10. 整理時に注意する現行差分

1. Capture設定の配置: GenericはSignal Capture独立、BT / DECTはInput / Frontendへ混在、ADS-BはADS-B Analysisへ混在。
2. 周波数設定の配置: Generic / BTはInput / Frontend、DECTはDECT Analysis、ADS-Bは固定。
3. Signal Descriptionの役割: Genericは編集、BT / DECTは確認だけ。DECTはDECT Analysisと固定説明が重複。
4. Triggerの粒度: Generic / BTは取得と後処理の2系統、DECTは取得Levelだけ、ADS-Bは独立設定なし。
5. Analysis Channel: Generic / BT / DECTでは共有項目が多いが、独立Analysis CenterはGenericだけ。ADS-Bに同じ設定経路はない。
6. Power correction: External GainはGeneric / ADS-Bだけ。Input Correction計算値の表示はGenericだけ。
7. Displayの名称: 共通パネルはDisplayだが、BT / DECTの内部ページ名はDisplay Config。
8. Plot resetの入口: Generic / ADS-BのDisplayにはReset Plot Scalesボタンがある。BT / DECTの同ページにはない（plot右クリックResetは別経路）。
9. Resetの意味: 統計／測定履歴／全履歴と表示のクリアが同一ラベルへ割り当てられている。
10. 無効なGeneric項目: Fine Synchronization、FSK Deviation Error compensation、Symbol Number at Pattern Start、および未実装Result Summary行。
11. 旧設定ページのSweep / Run: Generic / BT / DECTの内部設定dialogには残っているが、共通パネルのANALYZER SETUPには独立ボタンがない。同等操作はSWEEP CONTROLにある。
12. DECT Debug export: Export GFSK Modulation Debug CSV / Export DECT Power Debug CSVは旧File actionに実装されているが、共通パネルSystem > Fileには登録されていない。
13. Generic input source: Generated / IQ File / Plutoのcomboと個別Pluto selectorはコード上で作成されるが、現Input / Frontendフォームへ配置されていない。入力経路としてはRunとSystem > Fileを区別する必要がある。

この節は現状の差分であり、すべてを不具合や統一対象と断定するものではない。
Protocol固有の自動設定と、メニュー配置だけの不統一は分けて検討する。

## 11. 確認元コード

- `pluto_rtsa/vsa/ui/application_window.py`: 現共通シェルのボタン登録、各modeのpanel spec、System操作。
- `pluto_rtsa/vsa/ui/control_panel.py`: 共通メニュー階層、SWEEP CONTROL、Analyzer Mode。
- `pluto_rtsa/vsa/ui/main_window.py`: Generic設定ページと操作。
- `pluto_rtsa/vsa/result_summary.py`: Generic Result Summary項目と実装状態。
- `pluto_rtsa/vsa/protocol_modes/bluetooth/ui.py`: BT設定、Profile / Protocolによる条件付き表示。
- `pluto_rtsa/vsa/protocol_modes/dect/ui.py`: DECT設定と旧Debug export action。
- `pluto_rtsa/vsa/protocol_modes/dect/analysis.py`、`modulation.py`: DECT表示referenceの定義。
- `pluto_rtsa/standards/adsb1090/ui.py`: ADS-B Analysis / Receiver Location / Display設定。
- `pluto_rtsa/vsa/ui/measurement_chrome.py`: Power / Spectrumの適用初期値とDensity Spread。
