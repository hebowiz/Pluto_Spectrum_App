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

WideBand RT SAのFrequencyメニューではChunk Widthを10／20／30／40 MHzから選択できます。初期値10 MHzはPluto瞬時帯域の中央部を優先し、広い設定は速度と引き換えにノイズフロアの盛り上がり、mirror、帯域端特性の影響を受けやすくなります。

## 2. 振幅処理

RBW処理は全モードで同じGaussian complex IQ FIR特性を共有します。Sweep SAとTime Analyzer系はstateful FIRを電力化前に直接適用し、RealTime SA、WideBand RT SA、Calibrationは同じFIR係数を解析窓とするGaussian FFT filter bankを使います。指定RBWは両側3 dB bandwidth、ENBWは約1.0645倍です。RTSA系のoverlap FFTは次段階です。

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

- ウィンドウサイズは1664×1060固定です。SYSTEMフレーム追加前の980 pxから80 px拡大しています。
- 左側にステータス、ウォーターフォール、スペクトラムを配置します。
- 右側に測定器風の階層メニューを配置します。
- Main Menuの`TRIGGER / MARKER`下に`SYSTEM`フレームを配置し、`System`を収容します。
- 右クリックで右側メニューの前ページへ戻ります。
- 周波数軸はGHz、振幅軸はdBmです。
- Time Analyzer系では横軸が時間へ切り替わります。

表示モードは次の3種類です。ただしSweep SAとTime Analyzer系はSpectrum Onlyへ固定されます。

- Both
- Waterfall Only
- Spectrum Only

RealTime SAとWideBand RT SAではPersistence表示を利用できます。減衰設定はFast、Medium、Slowです。

ウォーターフォールの色スケールは測定レンジ下端から15%までを暗いNavyへ固定し、ノイズフロアが明るい色で表示を占有しないようにします。15～80%をBlue→Cyan→Green→Yellow→Redへ展開し、80～100%はRedへ飽和表示します。スペクトラムのY軸測定レンジ自体は変更しません。

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
| SYSTEM > System > Preset | 接続先Plutoを維持し、それ以外の保存対象設定を現行コードの初期値へ戻す |
| SYSTEM > System > Device | 接続先Plutoを選択し、初期設定後に現在のAnalyzer Modeを再始動する |

Systemページは他の右ペイン設定ページと同じstack/historyへ入り、Backまたは右クリックで前ページへ戻ります。Preset実行時は確認ダイアログを表示します。

## 7. 設定と校正ファイル

- 校正関連アプリ設定: `data/settings.json`
- 校正CSVの既定保存先: `data/calibration/`
- `settings.json`には最後に使用した校正CSVの絶対パスを保存します。
- 起動時にそのCSVが存在すれば自動読込してCalibrationをONにします。
- パスが存在しない場合はエラーをコンソールへ表示し、Calibration OFFで起動します。
- RTSAの接続先Pluto selectorと前回セッションは、PCローカルの`QSettings(PlutoSpectrumApp, PlutoRTSA)`へ保存します。
- セッション保存対象はAnalyzer Mode、Frequency/Span、Amplitude/Input、RBW、FFT/Waterfall/Persistence、Sweep、High Speed TA/Trigger、Trace 1～4、Marker 1～4です。
- TraceのAverage/Max Hold蓄積値、Waterfall履歴、Sweep進行位置、Trigger待機中のruntime状態、IQ/測定結果は保存しません。
- Calibration中に終了した場合はCalibrationへ入る前の通常設定を保存し、次回起動時にCalibration固定profileを再適用します。
- Preset実行時は保存済みセッションも初期状態へ即時更新しますが、接続先Pluto selectorは削除・変更しません。

## 8. 現在の制限

- TriggerボタンはHighSpeed TAでFree Run/Power Level設定を開きます。Power Levelは最終表示と同じdBmで指定し、選択中はグラフへ水平Trigger lineを表示します。他モードのTriggerは未実装です。
- SDRなしで実行できるIQストリーム単体テストがあります。GUI全体の自動テストはまだありません。
- PlutoSDRなしで動かすモックまたはデモモードはありません。
- GUIとモード統合処理の大部分が`RealtimeSpectrumWindow`へ集中しています。セッション永続化とSYSTEM/Preset UIは`SessionRealtimeSpectrumWindow`で拡張します。

## 9. 共通IQ取得層

作業ブランチ`feature/continuous-iq-stream`では、すべての取得経路が`PlutoReceiver`所有の`IQStreamBuffer`へ`IQBlock`を発行します。RealTime SAとHighSpeed TAは単一の連続RX workerを利用し、Sweep/WideBand/Calibration/旧Time Analyzerの同期取得も同じブロック形式を経由します。

- `sequence`: アプリ起動中の全ブロック連番
- `stream_id`: retune、設定変更、再開などで切り替わるepoch
- `block_index` / `start_sample_index`: epoch内の位置
- `discontinuity_before`: 直前との不連続
- `source`: 取得を要求したモード

consumerは独立cursorを持ちます。保持容量を超えた場合は`overrun`と推定可能な欠落ブロック数を返し、黙って上書きを隠しません。詳細と未完了項目は[IQストリーム改善計画](iq-streaming.md)を参照してください。

## 10. PlutoSDR接続

接続targetの優先順は、`SpectrumConfig.sdr_uri`、環境変数`PLUTO_SDR_URI`、列挙されたdirect USB、PlutoのIP contextです。2026-08-25以降、targetには一時的なIIO URIだけでなく`serial:<hardware serial>`を指定できます。接続時にserialから現在のURIを解決するため、USB portや列挙順が変わっても同じ個体へ接続できます。選択したserialが見つからない場合は別個体へ自動接続せずerrorとします。

RTSAは最後に選択したselectorをQSettingsへ保存し、次回起動時に同じ個体が列挙されていれば選択ダイアログを省略します。前回個体が存在せず1台だけ接続されている場合はその1台を選択・保存し、複数台なら選択ダイアログを表示します。`PLUTO_SDR_URI`の明示指定はこの記憶より優先します。

VSAのInput/Frontendでは`Refresh Devices`で個体一覧を非同期更新します。VSGのADALM-Pluto Settingsは直前の列挙cacheを即時表示し、明示的に`Refresh`を押したときだけ非同期更新します。設定dialogを開くだけではUSB scanを始めないため、表示を待たせず、閉じる際のnative thread競合も避けます。同じPlutoがdirect USBとRNDISの両方で見える場合はserialで1台にまとめ、direct USBを優先します。VSA/VSGはそれぞれ専用`QSettings`へ独立したserial selectorを保存するため、2台構成ではRX用とTX用を明示的に分離できます。VSAのPluto selectorはMeas Configの一部ではなく、Config loadで現在のRX個体を変更しません。serialを取得できないcontextは従来どおり明示URIで選択します。

解決後に実際に使われたURIは`PlutoReceiver.connection_uri`で参照できます。実機の短時間測定結果と再現手順は[PlutoSDR実機検証記録](hardware-validation.md)を参照してください。
