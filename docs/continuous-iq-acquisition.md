# 連続IQ受信 共通ライフサイクル

更新日: 2026-08-28

## 目的

RTSA、HighSpeed TA、VSAが個別にPlutoのRXワーカーとlibiioバッファを開始・停止すると、再アームのたびに受信不能時間が生じる。`ContinuousIQAcquisition`を共通の受信ライフサイクルとして使用し、同一RF設定中は1本のRX producerを維持する。

## 構成

```text
ADALM-Pluto / libiio
  -> PlutoReceiver RX worker（1本だけ）
  -> IQStreamBuffer（sequence / stream_id / sample index付きring）
       -> RTSA: cursorで順次読む overlap-FFT consumer
       -> HighSpeed TA: cursorで順次読む trigger/window consumer
       -> VSA: 解析開始ごとに新しいcursorを作る finite-record consumer
```

- `pluto_sa/sdr/pluto_receiver.py`: ハードウェア、RXワーカー、IQ block発行を所有する。
- `pluto_sa/sdr/continuous_acquisition.py`: start/re-arm/stop/reconfigureとstream planを共通管理する。
- `IQStreamBuffer`: consumerごとに独立cursorを持ち、読み出しても他consumerのデータを削除しない。

## モード別の扱い

### RTSA

Continuous中はproducerを動かし続け、consumer cursorでIQ blockを順次読む。FFT開始sample位置はblock境界をまたいで維持し、既定80% overlapの全FFTをDetectorへ渡す。GUIは約60 FPS以下で更新し、各表示frameにはその間に処理した複数FFTを集約する。consumerがring保持時間を超えて遅れた場合は古い画面を再生せず最古の保持blockへ追従するが、overrunを欠落として表示し、連続しないsampleを同じFFT窓へ混ぜない。

### HighSpeed TA

producerのblockをcursorで順次読み、Power TriggerまたはFree Runの有限時間recordを作る。ring overrunは欠落として明示し、欠落を跨いだrecordは正常な連続recordとして扱わない。

### VSA

VSA UI readiness is split into `Preparing` and `Armed`. Starting the Python RX
thread is not proof that libiio has queued a DMA buffer, so `Armed` is reported
only after the first actual IQ block has been published. This avoids telling
the operator to transmit during the initial RX blind interval.

Free RunとI/Q Power Triggerの両方を同じ`vsa_continuous` producerから取得する。解析完了・キャンセル・再アームではproducerを停止せず、新しいcursorだけを現在位置に作る。中心周波数、sample rate、RF bandwidth、gain、接続先などハードウェア設定が変わった場合だけ停止・再設定・再始動する。

これにより、従来の `capture_iq_block()` および取得ごとの `receiver.stop()` による受信空白を除去する。VSAの有限recordはblock境界を跨いでsample index基準で組み立てる。

## stream plan

producer互換性は次の組で判定する。

- RX block size
- source（producer用途名）
- finite producerの場合の最大block数

同じplanでの`start()`はハードウェアを再始動せず、新しいconsumer cursorを返す。異なるplanまたは終了済みfinite producerの場合のみ安全に停止して再始動する。

## libiioブロック境界への対策

共通ライフサイクル化だけでは、pyadi/libiio v0の同期的なbuffer refill境界に短いパケット列が重なった場合の欠落確率は下がらない。VSAが従来使っていた65,536 sampleブロックは8 MS/sで8.192 msであり、10 msのPower Trigger recordや複数パケット列が境界を跨ぎやすかった。一方、HighSpeed TAのPower Triggerは最大262,144 sampleのbuffer islandを使用していた。

`resolve_record_stream_block_samples()`を共通方針とし、有限recordの整数倍かつ最大約262k sampleとなるRXブロックを選ぶ。例えば80,000 sample recordは240,000 sample、24,000 sample recordも240,000 sampleとなる。VSAとHighSpeed TAはこの同じ計算を使用する。

さらに`PlutoReceiver`生成時、最初のRX bufferを作る前に対応バックエンドへkernel DMA buffer数8を要求する。Pythonが直前のブロックを変換・発行している間もPluto側の受信キューを維持するためである。未対応バックエンドでは例外にせずドライバ初期値を使用する。

VSA recording metadataには次の診断値を保存する。

- `continuous_rx_block_samples`
- `trigger_offset_in_rx_block`
- `trigger_samples_to_rx_block_end`

末尾欠落が再発した場合、トリガーから欠落位置までのsample数と`trigger_samples_to_rx_block_end`を比較し、RX block refill境界との相関を確認する。

## Pluto USBの連続受信レート上限

共通receiverはアプリ側のstop/start空白を除去するが、PlutoからUSBを渡って
こなかったsampleを復元することはできない。Plutoのcomplex sampleはI/Q各16 bit
の計4 byteであり、8 MS/sはプロトコルoverheadを除いても約32 MB/sを必要とする。
ADIが示すstock Pluto/USB 2.0の連続受信目安は約6 MS/sである。この上限を超える
場合、kernel/application bufferの増量は一時的なjitterを吸収するだけで、生成と
転送の継続的な速度差により最終的にqueueが満杯となりsampleが破棄される。

VSAは6 MS/s超を`GAP RISK`として表示する。また、最も遅いlibiio refill時間を
対応するIQ block時間で割った`continuous_rx_max_refill_ratio`を記録する。1.0以上は
host経路が実時間でその区間をdrainできなかった直接的な証拠である。1.0未満でも、
それ以前のkernel/DMA overflowがなかったことまでは保証しない。

HSTA/VSA比較ではsample rateも一致させる必要がある。LE1Mの切り分け推奨値は
4 samples/symbol（4 MS/s）とする。8/16 MS/sのburstを確実に無欠落取得するには、
Pluto local RAMへの有限capture後に転送する方式、またはより高速なhost interfaceを
持つhardwareが必要となる。

## 現時点の制限

- RF設定変更時にはAD936x/libiioの再設定が必要なので、その区間は連続ではない。
- ring保持時間を超えてconsumerが遅れるとoverrunになる。VSA/HSTAは欠落を黙って連結しない。
- USB転送能力を超えるsample rateでは、アプリ側の共通化だけで無欠落を保証できない。
- 1台のPlutoを別プロセスのRTSA/VSA/VSGから同時利用する設計ではない。既存のdevice leaseにより競合を拒否する。

## 検証

- `tests/test_continuous_acquisition.py`: 同一plan再アーム、plan変更、再設定、finite producer終了後の再始動。
- `tests/test_vsa_pluto_source.py`: VSA Free Run/Power Trigger、block跨ぎ、再利用、キャンセル。
- `tests/test_hsta_analysis_queue.py`: HSTAのcursor consumerと解析queue。

実機では、既知数の短いパケット列を入力し、VSAの取得窓内にある完全パケット数、先頭・末尾欠け、stream overrun、受信sample総数をHSTAと比較する。
