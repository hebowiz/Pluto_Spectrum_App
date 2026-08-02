# IQストリーム改善計画・実装状況

この文書は「取りこぼしのないIQデータ受信」を目指す改善作業の設計判断、実装状況、検証結果を記録する引継ぎ資料です。第三者または別のAIが作業を再開する場合は、最初にこの文書と各モード仕様を確認してください。

## 作業情報

- 作業ブランチ: `feature/continuous-iq-stream`
- 開始日: 2026-08-02
- 基準コミット: `b33f42e`
- 状態: Phase 1～3の初期統合完了、短時間の実機smoke test完了

## 目標

1. SDRからのIQ取得をGUI、FFT、モード固有処理から分離する。
2. 受信可能なサンプルレート範囲では、解析や描画中もIQ取得を停止しない。
3. 欠落やバッファ超過を黙って処理せず、サンプル連番と統計で観測可能にする。
4. RealTime SA、WideBand RT SA、Sweep SA、Time Analyzer、HighSpeed TA、Calibration、将来のVSAが共通のIQブロック形式を使用する。
5. ハードウェアなしでストリーム境界、consumer遅延、overrunをpytestできるようにする。
6. 将来のTrigger/VSAが同じsample timelineとtime recordを利用できる構造にする。

「無欠落」は無制限のSample Rateで保証する意味ではありません。実機ベンチマークで定める動作範囲内において、アプリケーション側の未報告欠落がなく、既知の不連続がすべて通知される状態を指します。

## 改善前の実装で確認した問題

- HighSpeed TAは時間窓完成時に受信スレッドとlibiioバッファを停止・破棄する。
- HighSpeed TAは解析と描画の完了後に受信を再開するため、各時間窓の間に必ず空白が生じる。
- HighSpeed TAの再開時にprime用の1ブロックを読み捨てる。
- HighSpeed TAの`deque(maxlen=4096)`は満杯時に古いブロックを通知なしで上書きする。
- ブロック完了後のホスト時刻だけでは、実際の欠落サンプル数を確定できない。
- RealTime SA、HighSpeed TA、同期`capture_block()`が別々のバッファ経路を持つ。
- 現在の環境はpyadi-iio 0.0.20、ネイティブlibiio 0.24であり、libiio 1.xの複数Block Stream APIを使用していない。

## 設計方針

### 共通IQブロック

すべての取得データを次のメタデータ付きブロックとして扱います。

```text
IQBlock
  stream_id             設定変更や明示的再開で変わるepoch ID
  block_index           stream内のブロック連番
  start_sample_index    stream内の先頭サンプル連番
  sample_count          ブロック内サンプル数
  iq                    complex64 IQ配列
  timestamp_s           ホスト上の取得完了時刻
  discontinuity_before  このブロック直前が連続でないことを示す
  source                 continuous / sweep / wideband / calibration等
```

LO、Sample Rate、RF bandwidth、buffer sizeの変更、明示的なバッファflushは新しい`stream_id`として表現します。

### Producer / Consumer

```text
PlutoSDR/libiio
    ↓ 単一Producerが連続取得
IQStreamBuffer
    ├─ RealTime SA: latest-only consumer
    ├─ HighSpeed TA: lossless window consumer
    ├─ Time Analyzer: window consumer
    ├─ VSA: 将来のlossless/overlap consumer
    └─ Sweep/WideBand/Calibration: retuneごとの明示的epoch
```

- SDR transportを操作できるProducerは常に1つだけにします。
- consumerごとに読み取りcursorを持たせます。
- consumerがリング容量より遅れた場合は`overrun`を返し、黙って古いデータへ飛びません。
- 解析用windowはホスト時刻ではなくサンプル数で区切ります。
- Trigger eventも`stream_id + sample_index`で表し、循環prestoreからpre/post-trigger recordを生成します。
- 表示用consumerは古いフレームを捨てられますが、HighSpeed TA/VSAでは欠落を明示します。

### モード別の連続性

| モード | 連続性の扱い |
|---|---|
| RealTime SA | 同一設定中は連続ストリーム。GUIは最新windowを表示可能 |
| HighSpeed TA | 解析中もProducerを停止せず、連続サンプルから時間窓を作る |
| Time Analyzer | HighSpeed TAと同じProducerから低頻度表示を作る |
| Sweep SA | LO変更ごとに新しいepoch。点内部のIQは連続 |
| WideBand RT SA | チャンクLO変更ごとに新しいepoch。チャンク内部のIQは連続 |
| Calibration | 校正周波数変更ごとに新しいepoch。各測定フレームを共通ブロック化 |

## 実装フェーズ

### Phase 1: テスト可能な共通ストリーム層

- [x] `IQBlock`とcursor結果型を追加
- [x] 容量制限付き`IQStreamBuffer`を追加
- [x] stream epoch、block/sample連番を追加
- [x] consumer overrun検出を追加
- [x] Fake IQによるpytestを追加
- [x] sample単位の厳密なwindow assemblerを追加

### Phase 2: PlutoReceiver統合

- [x] 連続RX workerの出力を`IQStreamBuffer`へ発行
- [x] すべての同期取得結果も共通IQブロックとして発行
- [x] 設定変更・retune・flush時の不連続を記録
- [x] 発行block/sample数と上書きblock数の統計を共通化
- [x] 生存workerを見失わず、二重Producerを拒否するlife-cycle制御
- [ ] RX worker例外と実効sample/sをUIへ通知する診断統計

### Phase 3: モード移行

- [x] RealTime SAをlatest-only consumerへ移行
- [x] HighSpeed TAを解析中も停止しないconsumerへ移行
- [x] 旧Time Analyzerの同期取得を共通ブロックAPIへ移行
- [x] Sweep/WideBand/Calibrationの同期取得を共通ブロックAPIへ移行
- [x] HighSpeed TAをFFT整列した厳密sample数windowと連続解析job queueへ移行
- [ ] 旧Time Analyzerを同期取得から共通連続Producer consumerへ移行

### Phase 4: transport最適化

- [ ] ブロックサイズ16,384 / 32,768 / 65,536 / 131,072を実測比較
- [ ] libiio kernel buffer数を実測比較
- [x] direct USBとRNDISの短時間スループットを比較
- [ ] 生int16 I/Q保持によるコピー削減を評価
- [ ] libiio 1.x Stream APIを評価
- [ ] 必要な場合のみCythonまたは専用受信プロセスを評価

### Phase 5: Trigger / VSA acquisition

- [x] Triggerと取得recordの共通data contractを追加
- [x] Power Level trigger detectorを追加
- [x] pre/post-trigger circular recorderを追加
- [x] 共通acquisition controllerとAuto timeoutを追加
- [ ] overlap FFT監視によるFrequency Mask Triggerを追加
- [x] HighSpeed TAをtrigger-aware record consumerへ移行
- [ ] IQ保存・offline replay APIを追加
- [ ] VSAのDDC/resampling/synchronization/demodulation pipelineを追加

External hardware triggerは当面の対象外とし、連続IQに対するhost software triggerへ集中します。

## テスト方針

### 実機不要

- 複数ブロックをまたぐサンプル連番
- epoch切替
- 複数consumerの独立cursor
- リング上書き時のoverrun検出
- 任意長windowの切り出し
- 解析が遅い場合もProducerが停止しないこと

現時点では`pytest.ini`で`tests/`だけを収集します。開発環境は`requirements-dev.txt`から構築できます。

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

2026-08-02時点: 93 tests passed。従来対象に加え、Gaussian complex IQ filterのblock境界state、狭RBW FFT convolutionの分割同値性、両側3 dB RBW、帯域外抑圧、ENBW/settling/group-delay metadata、Butterworth明示選択、linear-power detector定義、Sweep/HSTA統合、FFT非依存record長、display bucketの全sample被覆、IQ sample数とplot統計の表示分離、Single Snapshotのrecord長選択・fallback・有限block Producer、100 µs下限、Sample Rate依存上限、RBW 4 MHz→16 MSPS設定、Free Run/Power Buffer Islandのrecord整列、host-timed forced event、buffer端未完成eventのreset、dBm/dBFS往復変換、Trigger line表示条件、Power Triggerベクトル化と逐次状態機械の同値性、filtered trigger/raw record分離、Trigger Level modal中の停止・再開、CW位相jump/slip候補検出を検証しています。

## 現在の実装構成

- `pluto_sa/sdr/iq_stream.py`: ハードウェア非依存のブロック、cursor、リング、統計
- `pluto_sa/sdr/iq_window.py`: block境界をまたぐ厳密sample数window assembler
- `pluto_sa/sdr/trigger.py`: Trigger設定/eventとpre/post-trigger取得recordの共通contract
- `pluto_sa/sdr/trigger_detector.py`: sample-domain Power Level Trigger状態機械
- `pluto_sa/sdr/trigger_recorder.py`: circular prestore/poststoreとrecord生成
- `pluto_sa/sdr/trigger_acquisition.py`: arm/detect/forced timeout/rearmを統括する共通controller
- `pluto_sa/sdr/pluto_receiver.py`: SDRの単一所有者、連続Producer、同期取得、epoch発行
- `pluto_sa/ui/main_window.py`: RealTime latest consumer、HighSpeed TA loss-aware consumer
- `pluto_sa/modes/sweep_controller.py`: Sweep同期IQBlock consumer
- `tests/test_iq_stream.py`: 純粋ストリームテスト
- `tests/test_iq_window.py`: window分割、tail carry、不連続テスト
- `tests/test_pluto_receiver_stream.py`: Fake Pluto統合テスト
- `tests/test_hsta_analysis_queue.py`: HighSpeed TA FIFO backpressureとstale result分離

### 現時点で保証できること

- HighSpeed TAは時間窓完成時および解析中にRX workerを意図的に停止しない。
- HighSpeed TA consumerは解析と並行してIQを読み、指定Time Spanに対応する正確なsample数の連続windowをFIFO解析queueへ投入する。
- job/result queue満杯時は無通知上書きせず、backpressureを共通IQ ringへ伝える。
- 1つのプロセス内で発行済みブロックの順序とepoch内sample位置を追跡できる。
- consumer遅延でリング上書きが起きた場合、欠落を黙って隠さない。
- stop timeout時に生存workerを見失って二重Producerを開始しない。
- Trigger/VSA recordがhost時刻ではなくsample timelineで位置を共有できる。
- direct USBの短時間試験では4 MSPSおよび6 MSPSで共通stream/HSTA Continuousがring上書き0で完走する。

### まだ保証できないこと

- Pluto/libiio/USBより前または内部で発生した欠落。アプリ連番は取得成功したblockに付ける番号であり、ハードウェア連続性の証明ではない。
- 長時間相関試験で保証できるSample Rate上限。短時間の転送試験ではdirect USB 6 MSPS、RNDIS 5 MSPSが境界だった。
- Time Spanはsample整数へ丸めるため最大約0.5 sample周期の差を持つが、FFT frame整数倍への切り上げは廃止済み。
- 解析がリング保持時間より長い場合の無欠落。現在はoverrunを検出してwindowを破棄する。
- Sweep/WideBandはLO切替を伴うため、帯域全体の時間・位相連続性は設計上存在しない。

## 次に行う作業

1. Trigger位置の表示と、minimum duration/holdoff/hysteresisのUI設定を追加する。
2. RX refill時間、実効sample/s、ring使用量、consumer lag、job/result queue使用量、overrunをUI/ログへ公開する。
3. 実機でblock sizeを掃引し、既知信号の長時間相関で欠落を検証する。
4. 結果に基づいて安全な最大Sample Rate、block size、kernel buffer数を決める。

### PlutoSDR実機

- Sample Rateごとの受信sample/s、refill時間、CPU、メモリ
- block sizeとkernel buffer数の組み合わせ
- 10分以上の連続受信でアプリ側overrunが0であること
- 既知の周期波形またはPRBS/QPSK信号を使った相関による欠落検出
- 設定上限を超えた場合に警告または測定停止できること

## 参考実装・資料

- UniversalRadioHackerはPluto受信をCython/libiioの専用プロセスへ分離し、65,536 samples単位で連続refillする。
- libiio 1.xには複数Blockを使うStream APIがある。
- ADI公開値では、Plutoの転送量はdirect IIO USBで約26.1 MiB/s、RNDISで約20.4 MiB/s。
- complex I/Qを各16 bitで運ぶ場合、理論換算はそれぞれ約6.8 MSPS、約5.3 MSPS。実用上限は実機測定と安全率から決定する。

## 変更履歴

### 2026-08-02

- 改善用ブランチを作成。
- `git fetch --prune`後に`main...origin/main`が0 ahead / 0 behind（`b33f42e`）であることを確認。
- 現行HighSpeed TAの意図的な受信停止が最大の欠落原因であることを確認。
- 全モード共通のIQブロック、epoch、Producer/Consumer設計を決定。
- 実装フェーズと検証方針を作成。
- `IQStreamBuffer`、`IQWindowAssembler`、Fake IQ/Fake Pluto pytestを追加（17件合格）。
- `PlutoReceiver`の連続・同期取得を共通`IQBlock`発行へ統合。
- RealTime SAをlatest consumer、HighSpeed TAを独立cursor consumerへ移行。
- HighSpeed TA固有のprivate API探索、受信スレッド、無通知dequeを廃止。
- HighSpeed TAは解析中も共通RX workerを止めない構成へ変更。
- Sweep/WideBand/Calibration/旧Time Analyzerの同期取得を共通APIへ移行。
- stop timeout後の二重Producer起動を防止。
- 一般的なRTSA/VSAの信号経路を調査し、Trigger/VSA共通architectureとdata contractを追加。
- External hardware triggerを当面の対象外とし、software triggerへ範囲を限定。
- Power Level Triggerとpre/post-trigger recorderを追加し、判定遅延分を含むprestoreを実装。
- HighSpeed TAをFFT整列exact window、bounded FIFO job/result queue、明示的backpressureへ移行。
- 実機のdirect USB/RNDISを短時間比較し、direct USBは6 MSPS、RNDISは5 MSPSまで要求転送量へ追従することを確認。
- 接続先の暗黙なRNDIS選択を防ぐため、明示URI、環境変数、direct USB、Pluto IPの優先順を追加。
- 共通streamを4/6 MSPSで停止・再開し、overrun/連番/sample index errorが0であることを確認。
- HighSpeed TA SingleおよびContinuousを4/6 MSPSで実機検証し、ring上書き0、解析queue最大1を確認。
- 終了時のpyadi-iio buffer access violationを検出し、worker停止後の明示的RX buffer破棄で再発しないことを確認。
- HSTA状態変更時に古いstream cursorが残る不具合を修正し、Single→ContinuousとSweep Time変更後の表示再開を実機確認。
- Free Run/Power Levelのarm、Auto forced timeout、Normal、Single、rearmを統括する共通acquisition controllerを追加。
- HighSpeed TAの窓生成を`IQAcquisitionRecord`へ移行し、Trigger基本設定UIを追加。
- Power Auto forced recordとPower Normal natural recordをdirect USB実機で確認。
- 現行RBWがFFT後のpower spectrumへ非正規化Gaussian convolutionを行う方式であることを監査し、TA/Sweepはstateful IQ-domain measurement filterへ分離する方針を決定。
- SciPy SOSによる共通4次Butterworth complex IQ filterを追加し、Sweep SA、HighSpeed TA、旧Time Analyzerをpower化前のRBW処理へ移行。
- RBWを両側3 dB bandwidthとして定義し、ENBWとsettling samplesをfilter metadata化。RMS Detectorはlinear powerの平均へ修正。
- HighSpeed TAは連続するrecord間でfilter stateを保持。不連続・設定変更・record重複時はresetする。
- HighSpeed TAのrecord長とdisplay bucketをFFT sizeから分離。Time Span初期値を10 ms、表示上限を1000 pointsとし、各IQ sampleをDetector bucketへ一度ずつ割り当てる。
- HSTAヘッダーの曖昧な`Samples`/`Avg dt`を`IQ Samples`、`Plot Points`、`Plot dt`へ分離。内部秒座標とms tick変換が既に整合していることを再確認。
- Sweep/TA共通IQ Filterのデフォルトを4次Butterworthからlinear-phase Gaussian FIRへ変更。両側3 dB RBW、ENBW約1.0645倍、tap数、群遅延をmetadata化し、狭RBWではstateful FFT convolutionへ自動切替。Butterworthは明示指定時の比較用として維持。
- HighSpeed TAのSingle Free Runへexact-record RX Snapshotを追加。4,194,304 samples以下を単一bufferとし、warm-up 5＋本取得1 blockでProducerを自動終了。12 MSPSの10/100 ms recordを実機で解析・表示まで確認。
- LiteVNA 2441 MHz CWをPluto Center 2440 MHzで12 MSPS Snapshot取得し、10 ms/100 ms双方の単一buffer内で位相jump 0を確認。CWの2π ambiguityを残すため最終判定はPRBS/counter待ち。
- HighSpeed TAのTime Span下限を100 µs、上限を4,194,304 samples相当へ変更し、RBW上限5 MHzにより16/20 MSPSを選択可能化。LiteVNA CWで16/20 MSPSの10/100 ms、30/40 MSPSの10 msを検証し、全単一bufferでsample slip候補0を確認。
- 6 MSPS超のFree Run Continuousへbuffer-isolated record生成を追加。16 MSPS・2 msは3 records、3 msは2 recordsを各96,000-sample buffer内だけから作り、buffer境界でtrigger/filter stateをreset。実機でring上書き・queue滞留0とCW slip候補0を確認。
- 6 MSPS超のPower Trigger Single/ContinuousへBuffer Islandを追加。262,144 samples以下の最大record整数倍をoffline走査し、同一bufferでpre/postが完成するeventだけを採用。端eventを棄却し、Auto timeoutはhost時刻から安全なforced eventを生成する。16 MSPSの2 ms Autoと3 ms Normal Singleを実機確認。
- Power Trigger LevelをdBFS UIから最終表示と同じdBmへ変更。固定補正、Center Frequencyの周波数別補正、入出力補正、IQ full scaleを逆算して内部dBFS detectorへ渡し、HighSpeed TAのPower Level選択中はグラフへ黄色破線のTrigger levelを表示。
- GUIスレッドを長時間占有していたPower Triggerのsample逐次走査を、minimum duration 1 sample時のNumPy threshold検索へ変更。256,000-sample判定を約88.5 msから約4.1 msへ短縮し、Rising/Falling/Eitherのevent同値性と16 MSPS実機GUI callback時間を確認。
- Power Trigger判定をraw瞬時IQから表示と同じGaussian IQ RBW filter後へ変更し、raw IQ record保持と分離。Trigger Level dialog中はProducer/timerを停止してqueueを破棄し、未確定入力中の古いrecord表示を防止。
- 詳細な条件・数値・限界を[PlutoSDR実機検証記録](hardware-validation.md)へ記録。
