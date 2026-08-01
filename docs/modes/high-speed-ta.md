# HighSpeed TA 仕様

## 目的

指定したCenter Frequencyの電力を連続取得し、時間軸上の振幅変化として表示します。GUIタイマーから受信処理を分離し、通常のTime Analyzerより高い連続取得性を目指したモードです。

## 表示

- 横軸: Time [ms]
- 縦軸: Amplitude [dBm]
- 表示はSpectrum Onlyへ固定
- Time Span入力範囲: 0.01～10000秒
- 4トレース、4マーカーを利用可能
- マーカー位置は時間として扱います

## RBW連動設定

RBWは100 Hz～3 MHzです。Sample Rate、RF bandwidth、FFT sizeはSweep SAと共通の式で自動決定します。

```text
target bandwidth = max(4 × RBW, 521 kHz)
FFT size = RBW条件とguard条件を満たす64～16384の2のべき乗
```

共通IQ Producerの1ブロックは現在65536 samplesです。FFT sizeとは独立しており、解析時にFFT size単位へ分割します。

## スレッド構成

```text
PlutoReceiver共通RX worker（全モード共通の単一Producer）
  └─ IQBlock発行 → IQStreamBuffer（初期512ブロック）

GUIスレッド
  └─ 時間窓完成の判定、解析ジョブ投入、結果描画

解析スレッド
  └─ IQブロック群のFFT/RBW/Detector処理
```

HighSpeed TA consumerは独立cursorでブロックを読みます。受信開始直後または設定変更後は5ブロックをwarm-upとして解析対象から除外します。

## 取得バックエンド

`PlutoReceiver`の共通RX workerがpyadi-iioの公開`rx()`をブロッキング呼出しし、取得結果をcomplex64の`IQBlock`として発行します。HighSpeed TA固有のprivate API探索や別リングは廃止しました。SDR transportへアクセスする連続Producerは1つだけです。

`start()`は既存workerとblock size/sourceが異なる二重起動を拒否します。`stop()`がタイムアウトしても生存workerの参照を保持し、別workerを重ねて開始しません。

## 時間窓と解析

- 指定Time Span以上のサンプル数になるまでブロックを蓄積して1掃引とします。
- ブロック間隔が期待時間の1.2倍を超えた場合、gapとして統計へ記録します。
- 1つの時間窓が完成すると取得データのスナップショットを解析スレッドへ渡します。
- Continuousでは解析・描画中も共通Producerを停止せず、後続IQをリングに蓄積します。
- consumerが512ブロックより遅れた場合はoverrunとして明示し、不連続をまたいだ時間窓を破棄します。
- Singleは1時間窓の解析・表示完了後に停止します。

## 振幅処理

各ブロックに対してFFT/RBW処理を行い、中心周波数の電力またはSweep相当のDetector値を表示値へ変換します。固定補正、入出力補正、Center Frequencyにおける周波数別校正を適用します。

## 制限・注意事項

- gap検出用データは保持しますが、現時点ではgapマーカー表示を無効化しています。
- ピークログも初期状態では無効です。
- 大きなTime SpanではIQブロックと解析負荷が増加します。
- 現段階の時間窓はブロック境界まで含むため、指定Time Spanを最大1ブロック弱超過する場合があります。厳密なsample単位window切出しは未実装です。
- GUIは解析中にconsumer読出しを止め、解析完了後に1ブロックずつ追従します。長い解析ではリングoverrunの可能性があります。
- 512×65536 samplesのcomplex64保持は最大約256 MiBです。
- USB/libiio内部の欠落はアプリ側連番だけでは検出できないため、実機の既知信号による連続性検証が必要です。
