# Sweep SA 仕様

## 目的

Start～Stop間の各周波数へLOを順次設定し、周波数点ごとの代表電力を測定して掃引トレースを作成します。

## 初期設定

| 項目 | 値 |
|---|---:|
| Sweep Points | 201 |
| Detector | Sample |
| RBW | 1 MHz |
| UI Timer Interval | 1 ms |
| LO Settle | 200 µs |
| Retune Flush Reads | 4 |
| Flush Samples | 256 |
| UI再描画間隔 | 4 points |

Sweep Pointsの入力範囲は11～1001です。周波数点はStartとStopを含む等間隔の`linspace`で生成します。

## RBW連動取得設定

RBWは100 Hz～3 MHzへ制限されます。RBW変更時はサンプルレート、RF bandwidth、FFT size、取得サンプル数を自動決定します。

```text
target bandwidth = max(4 × RBW, 521 kHz)
required FFT = max(8 × target bandwidth / RBW,
                   14 × target bandwidth / 300 kHz)
FFT size = required FFT以上となる64～16384の2のべき乗
```

FFT sizeは周波数ピーク診断用です。実際の取得サンプル数はIQ filterのsettlingとdetector観測時間を確保するため、次の必要数以上となる2のべき乗へ拡張します。このため狭RBWでは従来よりSweep Timeが長くなります。

```text
minimum capture samples
  = IQ filter settling samples + ceil(4 × Sample Rate / RBW)
capture samples
  = max(FFT size, minimum capture samples)以上の2のべき乗
```

## 1点の測定フロー

1. Sweep用sample rate、RF bandwidth、manual gainをPlutoSDRへ設定
2. 測定周波数へLOをリチューン
3. LO settle時間だけ待機
4. 指定回数だけ受信データを読み捨て
5. IQブロックを取得
6. 必要に応じてDC平均値を除去
7. LO pointごとに初期化したGaussian complex IQ FIRを適用
8. filterのsettling区間後に`I² + Q²`の時間系列を生成
9. Detectorで代表値へ集約
10. dB変換後、掃引結果へ格納

RBWはcomplex basebandの両側3 dB bandwidthです。Gaussian FIRはDC gain 1、ENBW約`1.0645 × RBW`で、linear-phase群遅延と全tap分のsettling区間を除いてからDetectorへ渡します。FFTはピーク周波数などのdebug情報には残していますが、掃引点の測定電力には使用しません。

## Detector

| Detector | 代表値 |
|---|---|
| Sample | 観測系列の最後の値 |
| Peak | 観測系列の最大値 |
| RMS | IQの平均二乗、すなわちlinear power系列の算術平均 |

## Sweep進行

- 1回のGUIタイマー呼び出しにつき原則1点を測定します。
- 最初の1点、指定再描画間隔、掃引完了時にトレースを更新します。
- Singleは全点完了後に停止します。
- Continuousは全点完了後に次の掃引を開始します。
- マーカー更新と完了スナップショットは掃引完了時に行います。

## Sweep Time

設定値としてSweep Timeを保持しますが、実際の最小掃引時間は次の見積りを基準に表示されます。

```text
1点時間 = 想定retune時間 + settle時間 + flush取得時間 + 本取得時間 + 想定処理時間
最小掃引時間 = Sweep Points × 1点時間
```

## 制限・注意事項

- 各点は異なる時刻に測定されるため、掃引中に変化した信号は1本のトレース内で時間差を含みます。
- 各点の同期取得結果は共通`IQBlock`として発行され、LO切替は新しい`stream_id`になります。flushで読み捨てた区間は不連続として扱います。
- `SweepController`の先頭docstringには「skeleton」と残っていますが、点測定、連続掃引、Detector処理は実装済みです。
- Sweep Timeはハードウェア処理時間を完全に保証する制御値ではありません。
