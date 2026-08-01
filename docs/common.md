# 共通仕様

## 1. 初期設定

| 項目 | 初期値 |
|---|---:|
| モード | RealTime SA |
| Center Frequency | 2.440 GHz |
| Span | 20 MHz |
| RBW | 1 MHz |
| FFT Size | 4096 |
| Internal Gain | 30 dB |
| Reference Level | 20 dBm |
| Display Range | 100 dB |
| External Attenuation | 30 dB |
| External Gain | 0 dB |
| DC Offset Removal | OFF |

周波数の基本入力範囲は70 MHz～6 GHzです。ただしWideBand RT SAは80 MHz～5.990 GHzへ制限されます。

## 2. 振幅処理

FFT前にHann窓を適用し、FFTサイズと窓のcoherent gainで正規化します。RBWはGaussianカーネルを電力スペクトルへ畳み込む方式です。カーネルはFWHMとして指定されたRBWから生成され、総和を1へ正規化しないため、周波数方向のエネルギー積算として働きます。

表示値には最終段で次の補正を加算します。

```text
表示値 [dBm]
  = 測定値 [dB]
  + 固定 Calibration Offset
  + External Attenuation
  - Internal Gain
  - External Gain
  + 周波数別 Calibration Offset（有効時）
```

初期値では固定Calibration Offsetが-62 dB、入力補正が `30 - 30 - 0 = 0 dB` です。

周波数別補正はCSVの測定点間を線形補間します。範囲外では端点の値を使用します。

## 3. 表示

- ウィンドウサイズは1664×980固定です。
- 左側にステータス、ウォーターフォール、スペクトラムを配置します。
- 右側に測定器風の階層メニューを配置します。
- 右クリックで右側メニューの前ページへ戻ります。
- 周波数軸はGHz、振幅軸はdBmです。
- Time Analyzer系では横軸が時間へ切り替わります。

表示モードは次の3種類です。ただしSweep SAとTime Analyzer系はSpectrum Onlyへ固定されます。

- Both
- Waterfall Only
- Spectrum Only

RealTime SAとWideBand RT SAではPersistence表示を利用できます。減衰設定はFast、Medium、Slowです。

## 4. トレース

トレースは4本です。初期状態ではTrace1だけが表示されます。

| 種類 | 動作 |
|---|---|
| Live | 最新値を表示 |
| Max Hold | 各点の最大値を保持 |
| Average | `alpha = 1 / Average Count` の指数移動平均 |

各トレースには表示ON/OFFとHoldがあります。Hold中はトレースの内部値を更新しません。Average Countの入力範囲は1～1000です。

## 5. マーカー

マーカーは4個あり、それぞれ対象トレースを選択できます。

- ON/OFF
- 周波数または時間の指定
- Step指定と増減
- Peak Search
- Continuous Peak
- Marker to Center（周波数モード）
- マウスホイールによる位置調整

Sweep系では完了した掃引データを基準にマーカーを更新します。

## 6. 測定制御

| 操作 | 動作 |
|---|---|
| Continuous | 現在のモードを連続実行 |
| Single | 1フレーム、1掃引、または1時間窓を取得して停止 |
| Reset | トレースや進行状態を初期化し、実行状態を可能な範囲で復元 |

## 7. 設定と校正ファイル

- アプリ設定: `data/settings.json`
- 校正CSVの既定保存先: `data/calibration/`
- `settings.json`には最後に使用した校正CSVの絶対パスを保存します。
- 起動時にそのCSVが存在すれば自動読込してCalibrationをONにします。
- パスが存在しない場合はエラーをコンソールへ表示し、Calibration OFFで起動します。

## 8. 現在の制限

- Triggerボタンは未実装です。
- 設定の多くは終了時に永続化されません。永続化対象は最後の校正CSVパスだけです。
- SDRなしで実行できるIQストリーム単体テストがあります。GUI全体の自動テストはまだありません。
- PlutoSDRなしで動かすモックまたはデモモードはありません。
- GUIとモード統合処理の大部分が`RealtimeSpectrumWindow`へ集中しています。

## 9. 共通IQ取得層

作業ブランチ`feature/continuous-iq-stream`では、すべての取得経路が`PlutoReceiver`所有の`IQStreamBuffer`へ`IQBlock`を発行します。RealTime SAとHighSpeed TAは単一の連続RX workerを利用し、Sweep/WideBand/Calibration/旧Time Analyzerの同期取得も同じブロック形式を経由します。

- `sequence`: アプリ起動中の全ブロック連番
- `stream_id`: retune、設定変更、再開などで切り替わるepoch
- `block_index` / `start_sample_index`: epoch内の位置
- `discontinuity_before`: 直前との不連続
- `source`: 取得を要求したモード

consumerは独立cursorを持ちます。保持容量を超えた場合は`overrun`と推定可能な欠落ブロック数を返し、黙って上書きを隠しません。詳細と未完了項目は[IQストリーム改善計画](iq-streaming.md)を参照してください。

## 10. PlutoSDR接続

接続URIの優先順は、`SpectrumConfig.sdr_uri`、環境変数`PLUTO_SDR_URI`、列挙されたdirect USB、PlutoのIP contextです。direct USBとRNDISが同時に見える場合にpyadi-iioの自動選択へ依存しないようにしています。

現在選択されたURIは`PlutoReceiver.connection_uri`で参照できます。実機の短時間測定結果と再現手順は[PlutoSDR実機検証記録](hardware-validation.md)を参照してください。
