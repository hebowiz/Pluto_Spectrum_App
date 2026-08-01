# IQストリーム改善計画・実装状況

この文書は「取りこぼしのないIQデータ受信」を目指す改善作業の設計判断、実装状況、検証結果を記録する引継ぎ資料です。第三者または別のAIが作業を再開する場合は、最初にこの文書と各モード仕様を確認してください。

## 作業情報

- 作業ブランチ: `feature/continuous-iq-stream`
- 開始日: 2026-08-02
- 基準コミット: `b33f42e`
- 状態: Phase 1～3の初期統合完了、実機検証前

## 目標

1. SDRからのIQ取得をGUI、FFT、モード固有処理から分離する。
2. 受信可能なサンプルレート範囲では、解析や描画中もIQ取得を停止しない。
3. 欠落やバッファ超過を黙って処理せず、サンプル連番と統計で観測可能にする。
4. RealTime SA、WideBand RT SA、Sweep SA、Time Analyzer、HighSpeed TA、Calibration、将来のVSAが共通のIQブロック形式を使用する。
5. ハードウェアなしでストリーム境界、consumer遅延、overrunをpytestできるようにする。

「無欠落」は無制限のSample Rateで保証する意味ではありません。実機ベンチマークで定める動作範囲内において、アプリケーション側の未報告欠落がなく、既知の不連続がすべて通知される状態を指します。

## 現行実装で確認した問題

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
- [ ] HighSpeed TAを厳密なsample数windowと連続解析job queueへ移行
- [ ] 旧Time Analyzerを同期取得から共通連続Producer consumerへ移行

### Phase 4: transport最適化

- [ ] ブロックサイズ16,384 / 32,768 / 65,536 / 131,072を実測比較
- [ ] libiio kernel buffer数を実測比較
- [ ] direct USBとRNDISを比較
- [ ] 生int16 I/Q保持によるコピー削減を評価
- [ ] libiio 1.x Stream APIを評価
- [ ] 必要な場合のみCythonまたは専用受信プロセスを評価

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

2026-08-02時点: 17 tests passed。対象はリング、epoch、複数cursor、overrun、latest window、任意長windowと余剰sample carry、不連続時のpartial破棄、Fake Plutoによる同期/連続発行、互換しない二重startの拒否、停止待ちworkerの保持です。

## 現在の実装構成

- `pluto_sa/sdr/iq_stream.py`: ハードウェア非依存のブロック、cursor、リング、統計
- `pluto_sa/sdr/iq_window.py`: block境界をまたぐ厳密sample数window assembler
- `pluto_sa/sdr/pluto_receiver.py`: SDRの単一所有者、連続Producer、同期取得、epoch発行
- `pluto_sa/ui/main_window.py`: RealTime latest consumer、HighSpeed TA loss-aware consumer
- `pluto_sa/modes/sweep_controller.py`: Sweep同期IQBlock consumer
- `tests/test_iq_stream.py`: 純粋ストリームテスト
- `tests/test_iq_window.py`: window分割、tail carry、不連続テスト
- `tests/test_pluto_receiver_stream.py`: Fake Pluto統合テスト

### 現時点で保証できること

- HighSpeed TAは時間窓完成時および解析中にRX workerを意図的に停止しない。
- 1つのプロセス内で発行済みブロックの順序とepoch内sample位置を追跡できる。
- consumer遅延でリング上書きが起きた場合、欠落を黙って隠さない。
- stop timeout時に生存workerを見失って二重Producerを開始しない。

### まだ保証できないこと

- Pluto/libiio/USBより前または内部で発生した欠落。アプリ連番は取得成功したblockに付ける番号であり、ハードウェア連続性の証明ではない。
- どのSample Rateまで実機で無欠落か。
- 任意Time Spanぴったりのsample数window。現在は最後のblock分だけ超過し得る。
- 解析がリング保持時間より長い場合の無欠落。現在はoverrunを検出してwindowを破棄する。
- Sweep/WideBandはLO切替を伴うため、帯域全体の時間・位相連続性は設計上存在しない。

## 次に行う作業

1. HighSpeed TAを`IQWindowAssembler`へ接続する。
2. 1件だけの解析pending slotを、容量とbackpressure方針を持つ連続window queueへ移行する。
3. RX refill時間、実効sample/s、ring使用量、consumer lag、overrunをUI/ログへ公開する。
4. 実機でblock sizeとSample Rateを掃引し、既知信号の相関で欠落を検証する。
5. 結果に基づいて安全な最大Sample Rate、block size、kernel buffer数を決める。

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
