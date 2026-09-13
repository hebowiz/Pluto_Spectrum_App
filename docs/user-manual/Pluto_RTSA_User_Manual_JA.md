# Pluto RTSA ユーザーマニュアル

文書版: 1.0
対象: Pluto RTSA（Spectrum Analyzer）

## 1. はじめに

Pluto RTSAは、ADALM-Plutoから取得したIQデータをスペクトラム、ウォーターフォール、または時間領域で観測するアプリケーションです。リアルタイム帯域内観測、広帯域の分割取得、掃引測定、高速時間解析、振幅校正を1つの画面から切り替えられます。

本書は日常操作を対象とします。DSP方式やフィルタ設計の詳細は、上位の技術文書を参照してください。

## 2. 安全上の注意

> **注意:** PlutoのRX入力へ大電力を直接加えないでください。送信機をケーブル接続する場合は、十分な減衰量とDCブロックの要否を確認してください。

- 表示電力は、校正値、外部ATT、外部Gain、Pluto内部Gainの設定に依存します。
- CalibrationがOFF、または校正範囲外の場合、絶対電力値には追加誤差が含まれます。
- WideBand RT SAおよびSweep SAは周波数を切り替えて取得します。同時刻の全帯域信号ではありません。

## 3. 起動と接続

1. ADALM-PlutoをUSBでPCへ接続します。
2. `Pluto_RTSA.bat`を起動します。
3. 複数台が検出された場合は、使用するシリアル番号を選択します。
4. タイトルバーの`[RX: …xxxx]`が対象機器と一致することを確認します。

アプリ起動後の機器変更は、`SYSTEM > System > Device`から行います。変更時は現在の取得を停止し、新しいPlutoを初期化して現在のAnalyzer Modeを再開します。

## 4. 画面構成

![Pluto RTSA画面構成](../images/user-manual/pluto-rtsa-overview.png)

1. **状態表示** — Center/Span、RBW、Reference Level、Gain、FFT条件、Detectorなどを表示します。
2. **Waterfall** — 周波数対時間の履歴表示です。新しいデータが連続的に追加されます。
3. **Spectrum / Time Trace** — 通常は振幅対周波数、Time Analyzer系では電力対時間を表示します。
4. **操作パネル** — 測定設定、Sweep Control、Trigger/Marker、Systemを階層式に操作します。高さが不足するとスクロールできます。
5. **メッセージ領域** — 観測ギャップ、RX discontinuityなど、注意が必要な状態を表示します。

右クリックまたは`Back`で1つ前の操作ページへ戻ります。ウィンドウをリサイズしても上下プロットの配置関係は維持されます。

## 5. クイックスタート

### 5.1 信号を観測する

1. `Analyzer Mode`で`RealTime SA`を選択します。
2. `Frequency`でCenterとSpanを設定します。
3. `Amplitude`でReference Levelと表示Rangeを設定します。
4. `Input`でInternal Gain、External ATT/Gainを設定します。
5. `BW`でRBWを設定します。
6. `Continuous`を押して連続取得します。
7. 停止する場合は、実行中表示のボタンを再度押します。

### 5.2 単発測定する

`Single`を押すと、現在のモードで1回分を取得して停止します。設定確認や画面保存前の静止表示に適しています。

### 5.3 初期状態へ戻す

- `Reset`: 現在のトレース蓄積や進行状態をリセットします。
- `SYSTEM > System > Preset`: 接続先Plutoを保持したまま、現在のコードで定義された既定設定へ戻します。

## 6. Analyzer Mode

| モード | 主な用途 | 注意点 |
|---|---|---|
| RealTime SA | Plutoの瞬時帯域内を連続観測 | 最大表示Spanと時間カバレッジを確認 |
| WideBand RT SA | 瞬時帯域より広い周波数範囲をチャンク合成 | 各チャンクは異なる時刻に取得 |
| Sweep SA | Start/Stop範囲を周波数点ごとに掃引 | Sweep Timeは設定と実測値を確認 |
| High Speed TA | バーストや電力変化を高速な時間軸で観測 | Triggerと取得長を適切に設定 |
| Calibration | 周波数別振幅補正データを作成 | 既知レベルの信号源が必要 |

旧Time Analyzer経路が表示される構成では、通常はHigh Speed TAを優先してください。

## 7. 主な設定

### 7.1 Frequency

- **Center**: 表示中心周波数。
- **Span**: 表示周波数幅。
- **Start/Stop**: 掃引系モードで使用する範囲。
- **CF Step**: Centerを増減するときの刻み。
- **Chunk Width**: WideBand RT SAの1回の取得幅。

### 7.2 Amplitude / Input

- **Reference Level**: 画面上端の基準レベル。
- **Range**: 縦軸表示幅。
- **Internal Gain**: Pluto RX gain。
- **External ATT / Gain**: 外部回路を含めた表示補正。
- **Correction**: 周波数別校正CSVによる補正。

外部ATTを挿入したときは、その減衰量を正値で入力します。プリアンプを使用したときはExternal Gainへ入力します。

### 7.3 BW / FFT / Detector

- RBWは分離能力、ノイズ表示、更新速度に影響します。
- FFT Autoは要求RBWと表示点数から解析条件を選択します。
- DetectorはPeak、Sampleなどから用途に合う方式を選択します。
- RealTime SAのTime Coverageが100%未満の場合、見逃し時間が発生し得ます。警告は画面下部へ表示されます。

## 8. Trace、表示、Marker

### 8.1 Trace

最大4本のTraceを使用できます。

- **Live**: 最新値。
- **Max Hold**: 各周波数binの最大値。
- **Average**: 指定回数に基づく平均。
- **Hold**: 現在のTraceを保持。

### 8.2 Display

`Both`、`Waterfall Only`、`Spectrum Only`を選択できます。Persistence対応モードでは残光表示と減衰速度を設定できます。

### 8.3 Marker

Markerは最大4個です。対象Trace、周波数または時間位置、移動Step、Peak Search、Continuous Peakを設定できます。Marker設定ページではマウスホイールで位置を微調整できます。

## 9. High Speed TAとTrigger

High Speed TAでは、連続IQから指定時間窓を構成して電力対時間を表示します。

1. Analyzer Modeを`High Speed TA`にします。
2. Time SpanとRBWを設定します。
3. `Trigger`でFree RunまたはPower Levelを選択します。
4. Power LevelではLevel、Slope、Position、Auto Timeoutを設定します。
5. `Single`または`Continuous`を実行します。

Trigger Levelは表示と同じdBm基準です。外部ATT/Gainを変更した場合はTrigger Levelも再確認してください。

## 10. Calibration

Calibrationは既知レベルのCWを用いて周波数別補正CSVを作成します。校正中は通常測定設定とは独立した固定条件が使用されます。

- 信号源レベルと接続損失を確認してから開始してください。
- 完了後のCSVは`data/calibration`配下へ保存されます。
- 最後に使用したCSVは次回起動時に自動読込されます。
- CSVが見つからない場合はCorrection OFFで起動します。

## 11. セッションと機器共有

Analyzer Modeごとに主要設定が保存され、再度そのモードへ入ると復元されます。Center Frequencyはモード間で共有されます。Trace蓄積、Waterfall履歴、掃引途中の位置、測定データは保存されません。

同じPlutoをRTSA、VSA、VSGから同時に開くと排他エラーになります。使用中のアプリで停止し、必要ならアプリを終了してから別アプリで選択してください。

## 12. トラブルシューティング

| 症状 | 確認事項 |
|---|---|
| Plutoが一覧にない | USB接続、給電、libiio、Device再検索を確認 |
| 電力値が合わない | Calibration、Internal Gain、External ATT/Gain、入力飽和を確認 |
| 信号が見えない | Center/Span、Reference Level、RF bandwidth、接続先Plutoを確認 |
| 更新が遅い | RBW、FFT Size、Span、Waterfall History、対象モードを確認 |
| 画面下部に観測ギャップ警告 | FFT条件を緩和するか、表示Span/RBWを見直す |
| Device busy | 同じPlutoを使用中のRTSA/VSA/VSGを停止または終了 |

## 13. 用語

- **RBW**: 近接信号の分離とノイズ帯域を決める分解能帯域幅。
- **FFT Size**: 1回のFFT点数。
- **Time Coverage**: 取得時間のうちFFT解析窓で評価できた割合。
- **Persistence**: 過去のスペクトラム発生頻度を色で重ねる表示。
- **Reference Level**: 振幅軸上端の基準値。
