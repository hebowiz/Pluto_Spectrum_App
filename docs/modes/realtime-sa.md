# RealTime SA 仕様

## 目的

PlutoSDRからIQを連続取得し、現在の受信帯域をFFTスペクトラムとウォーターフォールでリアルタイム表示します。

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
2. Hann窓を適用
3. FFTとfftshift
4. FFTサイズおよびcoherent gainで正規化
5. 絶対値二乗で電力化
6. Gaussian RBWカーネルを畳み込み
7. FFT両端のguard領域を除外
8. dB変換と表示補正

## 表示

- Spectrum、Waterfall、または両方を選択できます。
- Waterfallの初期履歴は300行、入力範囲は1～1000行です。
- Waterfallは周波数方向を4点ごとに間引きます。
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
- 表示値は校正された相対的なFFT電力をdBmとして扱う実装で、絶対精度は校正条件に依存します。
