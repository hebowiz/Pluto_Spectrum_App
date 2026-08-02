# HighSpeed TA 仕様

## 目的

指定したCenter Frequencyの電力を連続取得し、時間軸上の振幅変化として表示します。GUIタイマーから受信処理を分離し、通常のTime Analyzerより高い連続取得性を目指したモードです。

## 表示

- 横軸: Time [ms]
- 縦軸: Amplitude [dBm]
- 表示はSpectrum Onlyへ固定
- Time Span下限: 100 µs
- Time Span上限: `min(10000 s, 4,194,304 / Sample Rate)`
- Time Span初期値: 10 ms
- 表示点数: 最大1000 points
- 4トレース、4マーカーを利用可能
- マーカー位置は時間として扱います

## RBW連動設定

RBWは100 Hz～5 MHzです。Sample Rate、RF bandwidth、FFT sizeはSweep SAと共通の式で自動決定します。RBW 4 MHzで16 MSPS、RBW 5 MHzで20 MSPSとなります。

```text
target bandwidth = max(4 × RBW, 521 kHz)
FFT size = RBW条件とguard条件を満たす64～16384の2のべき乗
```

共通IQ Producerの1ブロックは現在65536 samplesです。取得record長、RBW filter、表示bucketはいずれもFFT sizeから分離されています。

Time Span上限はrecord保持量を4,194,304 IQ samples以内へ制限するためSample Rateに反比例します。代表値は521 kSPSで約8.051秒、12 MSPSで約349.5 ms、16 MSPSで262.144 ms、20 MSPSで約209.7 msです。高速sample rateで非現実的な長時間raw IQを確保しません。

例外として、SingleかつFree Runでは有限長Snapshotを使用します。`round(Time Span × Sample Rate)`が4,194,304 samples以下なら、1 record全体と同じ長さのRX bufferを使用し、warm-up 5 buffersと本取得1 bufferの計6 buffersでProducer自身が終了します。これによりUSBの持続throughputを超えるsample rateでも、本取得recordが単一device/DMA buffer内で連続している可能性を利用できます。Single上限超過時は65536-sample streamへ戻ります。短時間のFree Run ContinuousとPower Triggerは次のBuffer Island方式を使用します。

6 MSPSを超え、record長が262,144 samples以下の場合にBuffer Island方式を使用します。Free Run Continuousでは約65,536 samples以上となる最小のrecord整数倍をRX buffer長とし、最大4 recordsを1 bufferへ格納します。16 MSPS・2 msは32,000 samples×3、3 msは48,000 samples×2で、どちらも96,000-sample bufferです。

Power TriggerのSingle/Continuousでは、262,144 samples以下に収まる最大のrecord整数倍をRX buffer長とします。16 MSPS・2 msは32,000 samples×8＝256,000 samples、3 msは48,000 samples×5＝240,000 samplesです。各bufferを取得後に全sampleからtrigger edgeを探索し、pre/post sampleが同じbuffer内で完成するeventだけを公開します。buffer端で未完成のeventは棄却し、次buffer先頭でtrigger recorderとGaussian filter stateをresetします。buffer間にはUSB転送中のblind timeがありますが、不連続な境界をrecordへ混入させません。

## スレッド構成

```text
PlutoReceiver共通RX worker（全モード共通の単一Producer）
  └─ IQBlock発行 → IQStreamBuffer（初期512ブロック）

GUIスレッド
  └─ TriggerAcquisitionController → IQAcquisitionRecord → bounded解析queue → 結果描画

解析スレッド
  └─ job queueから連続windowを取得 → stateful IQ Filter → Power/Detector → result queue
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
- Positionから指定Time Spanに対応するrecord長をpre/post sample数へ分配します。
- Levelは校正済みdBmではありません。`iq_full_scale=2048`で正規化したsample magnitudeのdBFSです。
- 6 MSPS超のPower Islandでは、Auto Timeoutをsample数ではなくhostの単調時計で評価します。timeout時は取得済みbuffer内の安全な位置へforced eventを置き、完全なpre/post recordを生成します。
- Power Islandが監視できるのは各RX buffer内部だけです。buffer間blind timeに発生したevent、およびbuffer端でpre/postが完成しないeventは捕捉できません。周期信号や再試行可能なevent向けであり、希少なone-shot eventの捕捉は保証しません。

## 時間recordと解析

- 通常streamでは`TriggerAcquisitionController`がブロック境界をまたぐpre/post-trigger recordを作ります。Buffer Islandでは境界をまたがず、同一buffer内で完成するrecordだけを作ります。
- window長は`round(Time Span × Sample Rate)`です。FFT frameへの切り上げと解析時の端数sample破棄はありません。
- ブロック間隔が期待時間の1.2倍を超えた場合、gapとして統計へ記録します。
- 1つの時間窓が完成すると取得データのスナップショットを最大4件の解析job queueへ渡します。
- 解析結果も最大4件のFIFO queueでGUIへ渡します。旧実装の単一pending/latest slotによる上書きはありません。
- Continuousでは解析・描画中も共通Producerを停止せず、後続IQをリングに蓄積します。
- GUI consumerは1回のtimer callbackで最大8 IQ blockまで追従します。
- job queueが満杯の場合は完成windowを保持して新しいIQを読まず、backpressureを共通リングへ伝えます。
- consumerが512ブロックより遅れた場合はoverrunとして明示し、不連続をまたいだ時間窓を破棄します。
- Singleは正確な1 recordが完成した時点でProducerを停止し、解析・表示完了後に測定を終了します。
- Single Free Run SnapshotのProducerは最大6 buffersで自動停止するため、GUI停止時に巨大bufferがring容量まで増え続けません。
- Buffer Islandでは同一buffer内のrecord間だけIQ Filter stateを継続し、次buffer先頭recordへ`rx_buffer_island_boundary`を付けてresetします。

## 振幅処理

record全体へGaussian complex IQ FIR low-passを適用してから`I² + Q²`へ変換します。指定RBWは両側3 dB bandwidthであり、中心から±`RBW / 2`で電力が約-3.0103 dB、ENBWは約`1.0645 × RBW`です。FIRはlinear phaseで、群遅延は`(tap数 - 1) / 2 samples`です。Free Run等で前recordの終了sampleと次recordの開始sampleが連続している場合はfilter stateを引き継ぎ、overrun、設定変更、trigger recordの重複・空白がある場合はresetします。

filter後の全sampleを最大1000個の連続時間bucketへ重複・欠落なく分配し、各bucketへDetectorを適用します。Sampleは最後のpower、Peakは最大power、RMSはIQの平均二乗powerです。bucketの時間位置は中央sampleです。

表示間隔は概ね`Time Span / display points`です。初期10 ms、1000 pointsでは約10 µsとなり、FFT sizeを変えても表示間隔は変化しません。recordが1000 samples未満の場合は1 sampleを1 pointとして表示します。

40 samples/bucketは4 MSPS、10 ms、1000 pointsから得られる初期条件であり、固定値でも理論上の必須値でもありません。実装上の最小bucketは1 raw IQ sampleです。ただし意味のある時間分解能は`1 / Sample Rate`だけでなくIQ RBW filterの過渡応答にも制限されます。4 MSPSにおける現行Gaussian FIRの電力step応答では、10–90% rise timeはRBW 1 MHzで約0.5 µs、100 kHzで約5.5 µs、10 kHzで約56 µsです。

1000 pointsは描画負荷を一定にする表示上限です。Peak Detectorはbucket内の短いeventの最大値を保持しますが時刻はbucket幅へ量子化され、RMSは平均化します。raw IQおよびtriggerのsample timeline自体が10 µsへ間引かれるわけではありません。

ヘッダーは`IQ Samples`、`Plot Points`、`Plot dt`を別々に表示します。横軸の内部単位は秒ですが、HighSpeed TAの固定tick labelは1000倍してms表示します。

最後に固定補正、入出力補正、Center Frequencyにおける周波数別校正を適用します。Power Triggerは引き続きfilter前のraw IQ magnitudeを評価するため、表示RBWとTrigger bandwidthはまだ独立です。

## 制限・注意事項

- gap検出用データは保持しますが、現時点ではgapマーカー表示を無効化しています。
- ピークログも初期状態では無効です。
- 大きなTime SpanではIQブロックと解析負荷が増加します。
- 実効Time Spanはsample整数化のため、指定値に対して最大約0.5 sample周期の丸め差を持ちます。
- 解析が取得より継続的に遅い場合、job/result queueから共通リングへbackpressureが伝わり、最終的にring overrunとなる可能性があります。この場合は不連続を明示してpartial windowを破棄します。
- Trigger位置の縦線表示、minimum duration/holdoff/hysteresisのUI設定、Frequency Mask Triggerは未実装です。
- Gaussian FIRの群遅延はmetadata化済みですが、表示時刻およびTrigger markerの補正は未実装です。raw trigger判定は入力sample timeline、表示powerは遅延したfilter出力である点に注意が必要です。
- 512×65536 samplesのcomplex64保持は最大約256 MiBです。
- USB/libiio内部の欠落はアプリ側連番だけでは検出できないため、実機の既知信号による連続性検証が必要です。
- Snapshotは単一buffer長、record長、解析完了を保証しますが、buffer内部の物理的なsample連続性はまだ保証しません。counter/PRBSまたは位相連続な既知CWによる検証が必要です。
- Buffer Islandは連続した時間軸を保証する方式ではなく、連続bufferから得た個別のcontiguous record列です。ヘッダーへ`Islands`、`Island Records`、`Edge Reject`、推定`Blind ms`を表示します。blind timeはhost受領間隔から引いた標本時間の概算であり、Pluto内部の正確な欠落位置を示すものではありません。
- Single/Continuous切替やSweep Time変更時は、既存RX workerとstream cursorを同時に無効化して新しいepochで再開します。
