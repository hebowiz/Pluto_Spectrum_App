# Pluto VSG Continuous Transmission Design

## 1. Purpose

Pluto VSGへ、現在の有限長non-cyclic送信を維持したまま、1周期の波形を停止操作まで反復するContinuous送信を追加する。

Continuous送信は固定波形のgapless反復を目的とする。周期ごとに内容を変更するstreaming送信、packet境界に同期した厳密な停止、外部trigger同期は本設計の対象外とする。

### CW output added 2026-08-30

The Output menu and toolbar provide `Start CW with ADALM-Pluto (Current
Frequency / Level)` for RTSA/VSA verification. It sends a normalized constant
`1+0j` period through the existing cyclic-DMA continuous path. The RF carrier
is therefore at the configured TX LO/center frequency and uses the current
target output dBm. Device leasing, READY/configuration verification, Stop, and
safe application shutdown are shared with packet transmission. Starting CW
does not alter RF/baseband properties or launch calibration; after changing
frequency, sample rate, RF bandwidth, or device, Prepare must be run first.

Because this is a zero-IF CW, receiver-side LO leakage/DC effects can overlap
the wanted signal if both instruments use the same center frequency. For that
test, offset the receiver center while leaving the VSG CW frequency unchanged.

## 2. Existing behavior that must remain unchanged

- Finite送信は、指定packet数を含む全scheduleを1本のnon-cyclic DMA bufferとして1回だけ送信する。
- Finite送信のpacket数はProjectの`Repeat Count`で決まる。
- Previewは送信packet数にかかわらず先頭1周期だけを描画する。
- RF/baseband設定とTX quadrature calibrationは`Prepare / Calibrate ADALM-Pluto`で完了させる。
- Transmit操作ではsample rate、LO、RF bandwidth、calibration modeを変更しない。
- Stopまたはアプリ終了時は、gain mute、TX LO powerdown、DMA buffer破棄の順で安全停止する。
- stock PlutoのTDD engineは使用しない。

過去にcyclic bufferを有限packet数制御へ使った際、host cleanupまでbufferが周回し、余分な完全packetと途中packetが送信された。これは有限送信では不具合だが、停止まで同一周期を反復するContinuous送信では意図した動作になる。

## 3. User-facing model

Pluto Output Settingsへ次を追加する。

- `Playback Mode`
  - `Finite`（初期値）
  - `Continuous`

`Playback Mode`はPluto backend固有の再生方針であり、波形Projectの内容ではない。`QSettings`のPluto TX設定として保存し、`.pvsg` Project、NPZ、IQ TAR、WV exportの内容には影響させない。

### Finite

- `Repeat Count`回分をnon-cyclic bufferへ格納する。
- 送信完了後、自動的にmuteしてREADYへ戻る。
- 現行動作をそのまま使用する。

### Continuous

- Projectから生成済みの先頭1周期だけをcyclic bufferへ格納する。
- 1周期には`Pre Idle + Ramp Up + Packet + Ramp Down + effective Post Idle`を含める。
- Projectの`Repeat Count`はPluto送信には使用しない。ファイルexportでは従来どおり使用する。
- Transmit actionは開始後、Stop actionまたはアプリ終了まで実行中のままとする。
- UIには`1 period, continuous`と周期時間を表示する。

設定dialogではContinuous選択時に`Packets per transmission`を無効表示し、`One complete packet period repeats until Stop`と説明する。Project editorのRepeat Count自体は、export用途があるため変更・無効化しない。

## 4. Data model

backend設定へenumを追加する。

```python
class PlutoPlaybackMode(str, Enum):
    FINITE = "finite"
    CONTINUOUS = "continuous"
```

`PlutoTransmitSettings`へ`playback_mode`を追加し、初期値を`FINITE`とする。

`GenerationResult.iq`は現在どおりProjectのRepeat Count回を含んでよい。backendの`transfer()`は`iq.size / burst_count`を1周期のsample数として検証し、次のbufferを保持する。

- Finite: 全`result.iq`
- Continuous: `result.iq[:frame_sample_count]`

Continuous用cycleへFinite専用のDMA Pre-rollとtrailing zero guardを追加してはならない。追加するとそれらが毎周期反復され、Projectで指定したperiodと異なるためである。Continuousの周期境界保護はProject自身のPre/Post IdleとRampで定義する。

## 5. Backend separation

公開classは当面`PlutoOutputBackend`を維持し、内部処理を次の単位へ分ける。

- `_start_finite_noncyclic(sdr)`
- `_start_continuous_cyclic(sdr)`
- 共通の`_enter_safe_tx_state(sdr)`
- 共通の`_mute_and_stop(sdr)`

Finite経路へContinuous条件を混在させず、現在実機で確定しているnon-cyclic手順を変更しない。

### Continuous start sequence

1. 対象Plutoのexclusive leaseを取得する。
2. TX LO powerdownとminimum gainを適用する。
3. Prepare済みconfiguration signatureと実機read-backを照合する。
4. `tx_enabled_channels = [0]`を設定する。
5. `tx_cyclic_buffer = True`を設定する。
6. TX LOをONにし、minimum gainのまま既存のLO settling timeを待つ。
7. requested gainを適用する。
8. 先頭1周期のcycle bufferを`tx()`へ1回だけ渡す。
9. backend owner threadは`stop_event`を待つ。hostから追加bufferを送らない。
10. Stop後、共通cleanupを実行してleaseを解放する。

この手順でもhost commandとRF packet境界の厳密な同期は保証しない。最初のpacketを完全に観測する必要がある場合、ProjectのPre Idleを十分に確保する。専用のstartup guardをcycleへ暗黙追加しない。

### Continuous stop sequence

1. `stop_event`をsetする。
2. owner threadがTX hardware gainをminimumへ設定する。
3. TX LOをpowerdownする。
4. cyclic DMA bufferをdestroyする。
5. TX channelをdisableし、DAC zero sourceへ戻す。
6. stateをREADYへ戻し、leaseを解放する。

Stopは安全な即時停止を優先するため、packet途中で停止する場合がある。`Stop after current period`はstock Plutoからperiod完了feedbackを得られないため、別の将来機能とする。

## 6. State machine

```text
MUTED
  -> CONFIGURE -> CALIBRATING -> READY       (Prepare)
READY
  -> FINITE_TX -> STOPPING -> READY          (Finite)
READY
  -> CONTINUOUS_TX -> STOPPING -> READY      (Continuous)
any active state
  -> ERROR -> safe cleanup -> READY or MUTED
```

- `Prepare`、Pluto selector変更、RF/baseband設定変更は送信中に禁止する。
- waveform編集は送信中に反映しない。変更波形を送る場合はStop後に再生成・再転送する。
- window closeは既存のgraceful shutdownを使い、ContinuousでもStop完了後に終了する。
- device leaseはContinuous実行中ずっと保持し、RTSA/VSA/VSG間の同一Pluto競合を防ぐ。

## 7. Diagnostics

既存の`pluto_vsg_tx_trace.log`へ最低限次を記録する。

- `playback_mode`
- `tx_dma_mode`: `non-cyclic finite buffer`または`cyclic continuous buffer`
- `period_samples` / `period_duration_s`
- `finite_packet_count`（Finiteのみ）
- `continuous_started`
- `continuous_stop_requested`
- cleanup開始・完了
- libiio / pyadi-iio version
- selected Pluto selector / serial suffix

Continuousで`tx()`を複数回呼び出していないこともtestと診断logで確認できるようにする。

## 8. UI behavior

- `Transmit with ADALM-Pluto`
  - Finite: 現行どおり送信開始
  - Continuous: cyclic送信開始
- `Stop Pluto Transmission`
  - Finite/Continuous共通
  - Continuous中は常に有効
- status bar
  - Finite: packet数と全schedule時間
  - Continuous: `Continuous TX running: period ... ms`
- Continuous終了messageは`Pluto continuous transmission stopped`とする。
- Continuousは自動完了しないため、正常終了はStop要求またはgraceful application shutdownによってのみ発生する。

## 9. Validation

### Automated tests

1. 既定modeがFiniteである。
2. Finiteでは`tx_cyclic_buffer=False`で、全repeatを1回だけ送る。
3. Continuousでは`tx_cyclic_buffer=True`で、先頭1周期だけを1回`tx()`へ渡す。
4. Continuous cycle sample数がProjectのeffective periodと一致する。
5. Continuous cycleへDMA Pre-rollやFinite trailing guardが混入しない。
6. Stop時の順序がgain mute -> LO powerdown -> buffer destroyである。
7. Continuous実行中、device leaseが解放されない。
8. close requestがStopを要求し、worker完了後にwindowを閉じる。
9. Playback ModeのQSettings保存・復元がProject persistenceへ混入しない。
10. Finite既存testがすべてpassし、送信順序に差分がない。

### Pluto hardware tests

1. 1 ms以上の観測窓で、指定periodのpacketが停止まで反復される。
2. 1分以上の実行でperiod欠落・USB起因gapがない。
3. packet間隔がsample丸め後のperiodと一致する。
4. Stop後に変調packetが再送されない。
5. Continuous開始時にTX calibration burstが発生しない。
6. Finiteへ戻した際、指定packet数と先頭/末尾が従来どおり正しい。
7. Pre Idleが短い場合と十分長い場合を比較し、初回packetの完全性を記録する。

## 10. Known limitations

- zero IQは物理的なRF OFFではなく、LO leakageが残り得る。
- Continuous開始時刻とpacket境界を外部機器へsub-ms精度で同期できない。
- Stopはpacket境界を保証しない。
- cycle内容は実行中に変更できない。
- 異なるpacketを逐次供給する用途は、non-cyclic streaming、double buffering、custom HDLの別設計が必要である。

## 11. Implementation order

1. enum、settings、diagnostic reportを追加する。
2. `transfer()`でfinite scheduleとcontinuous cycleを明示的に分ける。
3. backend start経路をFinite/Continuousへ分割する。
4. mock backend testでbuffer長、cyclic flag、cleanup順を固定する。
5. Pluto Output SettingsとQSettings persistenceを追加する。
6. worker/status/close semanticsをContinuousへ対応する。
7. 全test後、十分な外部ATTを入れて実機検証する。

## 12. Implementation status (2026-08-29)

実装済み。

- `PlutoPlaybackMode`と`PlutoTransmitSettings.playback_mode`を追加した。初期値は`Finite`である。
- Finiteは従来の全schedule、DMA Pre-roll、trailing guard、non-cyclic commit処理を維持した。
- Continuousは生成結果から先頭1周期だけを抽出し、追加guardを混入させずcyclic DMAへ1回だけ渡す。
- Continuous実行中のowner threadはStop Eventを待ち、hostから周期ごとの再送を行わない。
- Stopおよびwindow closeは既存の安全停止処理を通り、gain mute、TX LO powerdown、buffer destroyを行う。
- Pluto Output SettingsへPlayback Modeを追加した。Continuous時はFinite専用設定とpacket数表示を無効化する。
- Playback ModeはQSettingsへ保存し、Projectおよびwaveform exportへは保存しない。
- diagnostic reportへplayback mode、DMA mode、period sample数・時間、開始・停止eventを追加した。
- mock PlutoによるContinuous buffer内容、cyclic flag、Stop cleanup、JSON reportとFinite回帰を自動test化した。
- VSG backend/UI関連testは97件passした。

未実施なのはPluto実機と外部ATTを用いた連続時間、周期欠落、開始packet、停止後RFの検証である。
