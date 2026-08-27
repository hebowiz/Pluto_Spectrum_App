# RealTime SA 仕様

## 目的

PlutoSDRからIQを連続取得し、現在の受信帯域をFFTスペクトラムとウォーターフォールでリアルタイム表示します。

## Pluto受信個体の選択

2台以上のADALM-Plutoが接続されている場合、RTSA起動時にhardware serial付きの選択dialogを
表示します。選択した`serial:<id>`はRTSA専用QSettingsへ保存し、次回dialogの初期選択に使います。
起動後も`SYSTEM > System > Device`から選択dialogを開けます。選択を変更すると現在の受信処理を
停止し、新しいPlutoへ中心周波数、sample rate、RF bandwidth、manual gain等の初期設定を適用した後、
現在のAnalyzer ModeをContinuousで再始動します。初期化に失敗した場合は旧Receiverへrollbackします。
VSG/VSAのdevice設定とは共有しないため、送信機と受信機へ同じ個体を誤って割り当てることを
避けられます。`PLUTO_SDR_URI`環境変数が指定されている場合はそれを最優先し、dialogを表示しません。
接続個体が1台だけの場合はそのserialを自動選択します。

2026-08-26の2台構成検証では、RTSAにdevice selectorがなく、自動選択された個体がVSGの送信個体と
重複したことで、RX workerのbuffer refillが`OSError: [Errno 110] host unreachable`で停止した
可能性が高いと判断しました。この競合を避けるため、上記の起動時選択を追加しました。

## 周波数・取得条件

- 最大表示Span: 55 MHz
- 初期Span: 20 MHz
- サンプルレート: `display_span_hz / (1 - 2 × guard_ratio)`
- 初期guard ratio: 0.04（FFT両端を各4%非表示）
- RF bandwidth: サンプルレートと同値
- RX buffer size: FFT size
- FFT size選択肢: 64～16384の2のべき乗

PlutoReceiverの共通RX workerがメタデータ付きIQブロックをリングへ連続発行します。GUIタイマーは現在のepochに属する最新のFFTサイズ分をコピーして描画します。初期保持容量は512ブロックです。

## 信号処理

1. 必要に応じてIQ平均値を減算
2. Sweep SA／TAと同じGaussian complex IQ FIR係数をFFT長へゼロ埋めした解析窓を適用
3. FFTとfftshift
4. FFTサイズおよびcoherent gainで正規化
5. 絶対値二乗で電力化
6. FFT両端のguard領域を除外
7. dB変換と表示補正

各FFT binは、中心周波数だけが異なる同一Gaussian複素フィルターの出力に相当します。RBWは両側3 dB bandwidth、ENBWは約`1.0645 × RBW`です。従来のFFT電力化後のGaussian畳み込みはRBW経路から除外しました。

指定RBWのGaussian係数が現在のFFT長へ収まらない場合は、FFT Sizeを最大16384まで自動拡張します。それでも収まらない場合は収まる最小RBWへ制限し、ステータスの`Eff RBW`へ実効値と`limited`を表示します。

## 表示

- Spectrum、Waterfall、または両方を選択できます。
- Waterfallの初期履歴は300行、入力範囲は1～1000行です。
- Waterfallは周波数方向を4点ごとに間引きます。
- Waterfallは測定レンジの下端15%を暗いNavyへ固定し、15～80%を多色表示、80%以上をRedへ飽和表示します。
- Persistenceを利用でき、Fast／Medium／Slowの減衰を選択できます。
- Live、Max Hold、Averageの4トレースと4マーカーを利用できます。
- 約1秒ごとにステータス表示を更新します。

## Continuous / Single

- Continuous: RXスレッドとGUIタイマーを継続動作させます。
- Single: 最新FFTブロックを1回描画した後、タイマーとRXスレッドを停止します。

## 初期値

| 項目 | 値 |
|---|---:|
| Center | 2.440 GHz |
| Span | 20 MHz |
| FFT Size | 4096 |
| RBW | 1 MHz |
| Update Interval | 0 ms（Qtタイマー最短間隔） |
| Waterfall History | 300 |
| Persistence Decay | Medium |

## 制限・注意事項

- 55 MHzを超えるSpanはモード移行時または設定反映時に55 MHzへ制限されます。
- フレーム落ち判定用の統計はありますが、取得データを再送・補間する機能ではありません。
- 現段階ではGUI更新ごとに最新のFFTサイズ分を解析し、overlap STFTは未実装です。したがってGaussian filter bankの周波数特性は適用済みですが、一般的RTSA相当のPOIや全sampleの時間被覆はまだ保証しません。
- 表示値は校正された相対的なFFT電力をdBmとして扱う実装で、絶対精度は校正条件に依存します。
