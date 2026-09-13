# Pluto VSA ユーザーマニュアル

文書版: 1.0
対象: Pluto VSA（Generic / Bluetooth / DECT / ADS-B Analyzer）

## 1. はじめに

Pluto VSAは、ADALM-Plutoで取得したIQまたは保存済みIQを解析するベクトル信号解析アプリケーションです。汎用FSK/PSK解析に加え、Bluetooth、DECT、ADS-B 1090ESの専用解析ワークスペースを備えます。

規格測定用Measurement Filterと、受信チャネルを切り出すAnalysis Bandwidthは別の処理です。設定を解釈するときは両者を混同しないでください。

## 2. 測定上の注意

- Result SummaryのPASS/FAILは、選択した条件と実装済みLimitに対する判定です。N/Aは必ずしも異常を意味しません。
- 規格適合性を判断するときは、対象規格、測定回数、送信電力クラス、テストパターン、校正状態を確認してください。
- PlutoのDC spurを避ける場合はOffset LOとAnalysis Channelを使用できます。
- 入力飽和やサンプル欠落があるIQでは、同期、PHY判定、Decode、EVM/DEVMが不正確になります。

## 3. 起動と基本手順

1. ADALM-Plutoを接続します。
2. `Pluto_VSA.bat`を起動します。
3. `Analyzer Mode`から解析モードを選びます。
4. `Input / Frontend`でPluto、中心周波数、Gain、Analysis Bandwidthを設定します。
5. `Signal Description`または専用Analysis設定でPHYと変調条件を設定します。
6. `Single`または`Continuous`を押してキャプチャします。
7. Result Summary、各プロット、Decode結果を確認します。

設定ウィンドウを閉じただけではキャプチャを開始しません。キャプチャはユーザーがRun操作を行ったときだけ開始します。

## 4. Generic VSA画面

![Generic VSA画面構成](../images/user-manual/pluto-vsa-generic-overview.png)

1. **IQ Power** — キャプチャ電力対時間。Trigger、Pattern、Result範囲も確認できます。
2. **Spectrum** — Raw CaptureまたはAnalysis Channel後IQの周波数表示。
3. **Result Summary** — 現在の解析値と複数パケット集計。
4. **Modulation** — FSK瞬時周波数、PSK Vector、Phase Difference、DEVMなど。
5. **Symbol Plot** — 復元されたシンボル表現。下段のSymbol Tableで数値も確認できます。
6. **操作パネル** — Analyzer Setup、Sweep Control、System、Fileを操作します。

## 5. Bluetooth専用画面

![Bluetooth Dedicated Analyzer画面構成](../images/user-manual/pluto-vsa-bluetooth-overview.png)

1. **IQ Power** — バースト包絡と解析範囲。
2. **Spectrum** — FSK/PSKを含むパケットのスペクトラム。
3. **Result Summary** — RF PHY測定値、Limit、判定、Reference Information。
4. **Modulation** — FSK瞬時周波数またはEDR PSK解析タブ。
5. **Packet Analysis** — Decode、Payload Hex、Packet List、Issues、Air Bits。
6. **操作パネル** — Bluetooth Analysisを含む専用設定とRun操作。

## 6. 共通操作パネル

### 6.1 ANALYZER SETUP

| ボタン | 内容 |
|---|---|
| Analyzer Mode | Generic、Bluetooth、DECT、ADS-Bを切替 |
| Input / Frontend | Pluto、周波数、Gain、Analysis Channelを設定 |
| Signal Description | 変調方式、Symbol Rate、Deviation等を設定 |
| Signal Capture | Genericの取得長などを設定 |
| Trigger | Power Triggerと位置を設定 |
| Pattern Search | 同期パターンと検索条件を設定 |
| Result Range | 解析対象範囲を設定 |
| Demodulation | 復調フィルタや補正条件を設定 |
| Result Summary | 表示項目や集計条件を設定 |
| Display | Power/Spectrumの入力系列などを設定 |

モードに不要な項目は表示されません。設定画面内の値は、Apply/OKで確定するまで編集途中の値として扱われます。不正値が残っている場合は確定できません。

### 6.2 SWEEP CONTROL

- **Single**: 1回キャプチャして解析。
- **Continuous**: Stopするまでキャプチャと解析を繰り返す。
- **Refresh Analysis**: 保存済みの同一IQを再解析。新規キャプチャは行いません。
- **Reset**: 現在の測定履歴と統計をクリア。

Continuous後もSingleの表示と動作は一致します。停止処理中は完了を待ってから次の操作を行ってください。

### 6.3 SYSTEM / FILE

- **Preset > Default**: 現在モードの既定設定を適用します。各モードにDefaultは1つです。
- **Device**: 使用するPlutoを選択します。
- **Recall / Save**: モード情報を含むMeasurement Configを読込／保存します。
- **Open IQ**: 保存済みIQを読込みます。
- **Export IQ**: 現在のIQを保存します。
- GenericではSymbol TableのExportも使用できます。

## 7. Generic VSA

### 7.1 Signal Description

FSKまたはPSKのSymbol Rate、Deviation、変調次数など、既知の信号条件を設定します。Auto設定がある項目は推定結果を使用しますが、信号仕様が既知なら明示指定の方が安定する場合があります。

### 7.2 同期と解析範囲

処理の概略は次のとおりです。

```text
Raw Capture IQ
  -> Analysis Channel（DDC / LPF / decimation）
  -> Trigger / Pattern Search
  -> Result Range
  -> Demodulation / Symbol recovery
  -> Result / Plot
```

Pattern Searchを使用する場合は、期待するbit/symbol列と信号のbit orderが一致していることを確認します。Result RangeはTriggerまたはPatternを基準にOffsetとLengthを指定できます。

### 7.3 Analysis Bandwidthの表示適用

Analysis Channel後IQは、Decode、同期、Modulation、Symbol、Resultで常に使用されます。PowerとSpectrumだけは適用先を個別に選べます。

| 設定 | ON | OFF |
|---|---|---|
| Apply Analysis Bandwidth to Power | Analysis IQの電力 | Raw Capture IQの電力 |
| Apply Analysis Bandwidth to Spectrum | Analysis IQ、Requested Center基準 | Raw IQ、Hardware LO基準 |

既定値はPower ON、Spectrum OFFです。これはMeasurement FilterのON/OFFではありません。

## 8. Bluetooth Dedicated Analyzer

対応範囲はBR、EDR 2M/3M、LE 1M/2M、HDTです。

1. `Bluetooth Analysis`でClassic/LE/HDT、PHY、テスト条件を選択します。
2. 既知のRFテストパターンを使用する場合はPattern条件を一致させます。
3. SingleまたはContinuousを実行します。
4. `Detected PHY`、Decode、CRC/HEC、Issuesを確認します。
5. 有効なパケットについてRF PHY測定値と判定を確認します。

FSK表示では、補正済み連続瞬時周波数、同じトレースをsymbol centerで読んだ緑点、同じ値のSymbol Plotを使用します。Bluetooth専用FSK表示はRF測定と同じBluetooth Measurement Filter後のtraceを基準にします。

EDRではFSK部とPSK部を分けて確認できます。PSKタブにはVector、Phase Difference、DEVMがあります。DEVMは規格用のブロック測定であり、Generic VSAの自由同期結果をそのまま使用しません。

## 9. DECT Dedicated Analyzer

1. `DECT Analysis`でDirection、Packet Type、Modulation Caseなどを設定します。
2. Center、Analysis Bandwidth、Triggerを設定します。
3. RunしてPacket ListとPattern Statusを確認します。
4. Power、Power-Time Template、GFSK Deviation、Modulation Speed等を確認します。

Current packetの判定と履歴集計は分離されています。過去のNGを含む統計を消す場合は`Reset`を使用します。DECT FSK Modulationの既定縦軸は±500 kHzです。

## 10. ADS-B 1090ES Analyzer

ADS-Bモードは1090 MHz IQからMode S / ADS-Bメッセージを検出・解析します。

- Input、Trigger、ADS-B Analysis条件を設定します。
- Packet Listから対象メッセージを選択します。
- DecodeされたDF、ICAO、各フィールドを確認します。
- 受信局位置を必要とする結果は、位置設定が正しい場合だけ使用してください。

## 11. プロット操作

- ホイールやドラッグで表示範囲を変更できます。
- ユーザーが変更した範囲は、次のsweepでも維持されます。
- Power/Modulationの時間軸はパケット基準の相対位置を保持します。
- 右クリックの`Reset`で、そのプロットの既定範囲へ戻ります。
- Modulation上のSymbol Pointは連続波形上に存在します。

Plotの拡大表示は測定範囲そのものを変更しません。測定範囲を変更する場合はResult Rangeまたは専用Analysis設定を使用します。

## 12. 保存ファイル

| 種類 | 用途 |
|---|---|
| Measurement Config JSON | Analyzer Modeと測定設定 |
| NPZ / IQ recording | IQとメタデータ |
| IQ TAR | 対応形式のIQ交換 |
| CSV等 | Symbol Tableやデバッグ出力（対応モードのみ） |

ConfigのRecallは保存元モードへ自動的に切り替えますが、自動キャプチャは行いません。Plutoの物理的な接続先はConfig読込だけでは変更されません。

## 13. トラブルシューティング

| 症状 | 確認事項 |
|---|---|
| 全ResultがN/A | パケット適格性、Pattern、Decode、必要パケット数、Limit定義を確認 |
| PHY誤判定 | サンプル欠落、SNR、中心周波数、Analysis Bandwidth、packet末尾を確認 |
| EVM/DEVMが大きい | 入力飽和、Offset LO、CFO、timing、対象PHY、Measurement Filterを確認 |
| FSK点が波形とずれる | 最新コードか、対象がDedicated/Genericのどちらかを確認 |
| Spectrum中心が想定外 | Analysis Bandwidth適用OFF時はHardware LO基準 |
| Configを閉じると解析される | 現仕様ではRun操作なしにキャプチャしない。再現時は操作順を記録 |
| Device busy | 同じPlutoを使用中の別アプリを停止または終了 |

## 14. 用語

- **Analysis Bandwidth**: DDC、channel LPF、decimationで解析チャネルを切り出すVSA共通処理。
- **Measurement Filter**: Bluetooth、EDR、HDTなどの規格測定に固有の受信フィルタ。
- **CFO**: Carrier Frequency Offset。
- **EVM / DEVM**: 理想シンボルに対する誤差／差動符号化信号に対する誤差。
- **Symbol Point**: recovered symbol center時刻で表示トレースをサンプリングした点。
