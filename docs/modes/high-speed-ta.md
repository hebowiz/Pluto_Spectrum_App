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
  └─ TriggerAcquisitionController → IQAcquisitionRecord → bounded解析queue → 結果描画

解析スレッド
  └─ job queueから連続windowを取得 → FFT/RBW/Detector処理 → result queue
```

HighSpeed TA consumerは独立cursorでブロックを読みます。受信開始直後または設定変更後は5ブロックをwarm-upとして解析対象から除外します。

## 取得バックエンド

`PlutoReceiver`の共通RX workerがpyadi-iioの公開`rx()`をブロッキング呼出しし、取得結果をcomplex64の`IQBlock`として発行します。HighSpeed TA固有のprivate API探索や別リングは廃止しました。SDR transportへアクセスする連続Producerは1つだけです。

`start()`は既存workerとblock size/sourceが異なる二重起動を拒否します。`stop()`がタイムアウトしても生存workerの参照を保持し、別workerを重ねて開始しません。

## Trigger

Main Menuの`Trigger`から次を設定できます。現時点ではHighSpeed TAだけが対象です。

| 設定 | 内容 |
|---|---|
| Source | Free Run / Power Level |
| Mode | Auto / Normal |
| Level | complex IQ magnitudeのdBFS、初期-20 dBFS |
| Slope | Rising / Falling / Either |
| Position | record内のpre-trigger比率、初期50% |
| Auto Timeout | eventがない場合にforced recordを作る時間、初期1000 ms |

- Free Runは従来どおり隙間のない連続recordを生成します。
- Power LevelのAutoは条件成立時にnatural record、timeout時にforced recordを生成して再armします。
- Power LevelのNormalは条件成立まで表示recordを更新しません。
- Main MenuのSingleはTrigger条件成立後にpre/postを含む1 recordを完成して停止します。SingleではAuto timeoutを使用しません。
- PositionからFFT frame整数倍のrecord長をpre/post sample数へ分配します。
- Levelは校正済みdBmではありません。`iq_full_scale=2048`で正規化したsample magnitudeのdBFSです。

## 時間recordと解析

- `TriggerAcquisitionController`がブロック境界をまたぐpre/post-trigger recordを作ります。
- window長は指定Time Spanを覆う最小のFFT frame整数倍です。これにより解析時に端数sampleを捨てません。
- ブロック間隔が期待時間の1.2倍を超えた場合、gapとして統計へ記録します。
- 1つの時間窓が完成すると取得データのスナップショットを最大4件の解析job queueへ渡します。
- 解析結果も最大4件のFIFO queueでGUIへ渡します。旧実装の単一pending/latest slotによる上書きはありません。
- Continuousでは解析・描画中も共通Producerを停止せず、後続IQをリングに蓄積します。
- GUI consumerは1回のtimer callbackで最大8 IQ blockまで追従します。
- job queueが満杯の場合は完成windowを保持して新しいIQを読まず、backpressureを共通リングへ伝えます。
- consumerが512ブロックより遅れた場合はoverrunとして明示し、不連続をまたいだ時間窓を破棄します。
- Singleは正確な1 recordが完成した時点でProducerを停止し、解析・表示完了後に測定を終了します。

## 振幅処理

各ブロックに対してFFT/RBW処理を行い、中心周波数の電力またはSweep相当のDetector値を表示値へ変換します。固定補正、入出力補正、Center Frequencyにおける周波数別校正を適用します。

## 制限・注意事項

- gap検出用データは保持しますが、現時点ではgapマーカー表示を無効化しています。
- ピークログも初期状態では無効です。
- 大きなTime SpanではIQブロックと解析負荷が増加します。
- 実効Time SpanはFFT frame単位へ切り上げるため、指定値より最大`(FFT size - 1) / Sample Rate`だけ長くなります。
- 解析が取得より継続的に遅い場合、job/result queueから共通リングへbackpressureが伝わり、最終的にring overrunとなる可能性があります。この場合は不連続を明示してpartial windowを破棄します。
- Trigger位置の縦線表示、minimum duration/holdoff/hysteresisのUI設定、Frequency Mask Triggerは未実装です。
- 512×65536 samplesのcomplex64保持は最大約256 MiBです。
- USB/libiio内部の欠落はアプリ側連番だけでは検出できないため、実機の既知信号による連続性検証が必要です。
- Single/Continuous切替やSweep Time変更時は、既存RX workerとstream cursorを同時に無効化して新しいepochで再開します。
