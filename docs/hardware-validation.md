# PlutoSDR実機検証記録

この文書は共通IQストリームとHighSpeed TAの実機検証条件、結果、未検証事項を記録します。短時間のホスト側試験であり、Pluto/libiio内部を含む無欠落の最終保証ではありません。

## 2026-08-02 検証環境

| 項目 | 内容 |
|---|---|
| Device | ADALM-Pluto Rev.C、Z7010-AD9364 |
| Serial | `1044730c370e00100400120023338fb325` |
| pyadi-iio | 0.0.20 |
| native libiio | 0.24 |
| direct USB URI | `usb:1.54.5`（列挙時の値であり固定値ではない） |
| RNDIS URI | `ip:pluto.local` / `192.168.2.1` |
| RX block | 65,536 complex samples |

検証はRXのみで実施し、送信機能は使用していません。

## 接続経路の問題と対策

修正前の`adi.Pluto()`自動接続は、USBとRNDISの両方が見える環境でRNDIS (`ctx_name=network`) を選択しました。同じ物理Plutoでも接続経路により持続可能な転送速度が異なるため、測定条件が暗黙に変わる問題でした。

`PlutoReceiver`は現在、次の優先順で接続します。

1. `SpectrumConfig.sdr_uri`で明示したURI
2. 環境変数`PLUTO_SDR_URI`
3. `iio.scan_contexts()`で列挙したdirect USB
4. Plutoと識別できるIP context
5. 列挙不能時はpyadi-iioの既定探索

USB URIは起動時に列挙して選ぶため、`usb:1.54.5`をコードへ固定していません。RNDISを明示的に使う例は次のとおりです。

```powershell
$env:PLUTO_SDR_URI = 'ip:pluto.local'
python -m pluto_sa.main
```

## 短時間RXスループット

各sample rateを約1.5秒間測定しました。`sustainable`は取得sample数/経過時間が要求値の98%以上という、この診断ツール内の判定です。

| Transport | Request | 実効値の概況 | 判定 |
|---|---:|---:|---|
| direct USB | 1 / 2 / 4 / 6 MSPS | 要求値とほぼ一致 | sustainable |
| direct USB | 6.5 MSPS | 約6.04 MSPS | saturated |
| direct USB | 7 MSPS | 約6.01 MSPS | saturated |
| direct USB | 8 MSPS | 約6.07 MSPS | saturated |
| direct USB | 12 MSPS | 約6.02 MSPS | saturated（要求の50.2%） |
| RNDIS | 1 / 4 / 5 MSPS | 要求値とほぼ一致 | sustainable |
| RNDIS | 6 MSPS | 約5.37 MSPS | saturated |

この個体・PC・libiio構成における短時間試験上の境界は、direct USBが6 MSPS、RNDISが5 MSPSです。安全な製品上限として確定するまでは余裕を見てdirect USB 5 MSPS以下を推奨します。

12 MSPS、65536 samples/block、2秒の追加試験では184回、12,058,624 samplesを取得し、実効6.0215 MSPSでした。1 blockの標本時間は5.461 msですが、`rx()` refillは平均10.878 ms、p95 11.102 ms、最大11.351 msで、全184回が期待時間の1.2倍を超えました。pyadi-iioは要求長の配列を返すため、現行アプリのsequence/sample indexだけではPluto/DMAC/libiio/USB段の欠落位置を識別できません。

この条件では各DMA buffer内部が連続で、buffer間に約1 block相当のblind timeが生じる可能性が高いものの、既知counter/PRBSなしには保証できません。HighSpeed TAはblock受領後にアプリ連番を付けるため、物理的に欠落したblock間も論理上連続に見える可能性があります。timestamp gapは検出できますが、現在はrecordを無効化していません。

### 12 MSPS Single Snapshot（2026-08-02）

HighSpeed TAのSingle Free Runでrecord全体を単一RX bufferへ収める有限長Snapshotを実装し、direct USBで検証しました。Producerは120,000または1,200,000 samplesのbufferを5回warm-upとして破棄し、6回目を本recordとして発行した時点で自動終了します。

| 条件 | 結果 |
|---|---|
| 12 MSPS、10 ms、120,000 samples/record | 約0.148秒、1 record/1000 plot points、6 blocks/720,000受信samples |
| 12 MSPS、100 ms、1,200,000 samples/record | 約1.354秒、1 record/1000 plot points、6 blocks/7,200,000受信samples |
| 共通 | ring上書き0、analysis queue最大1、終了時queue滞留0、例外なし |

両条件ともrecord長とRX buffer長が一致し、IQ Filter、Detector、表示まで完了しました。USB転送時間が標本時間を超えても、buffer間blind timeをrecord内へ連結しない構造です。ただしこの試験は単一buffer内部の無欠落を証明しません。既知counter/PRBSまたは位相連続CWによるsample slip検証が残っています。

再実行例:

```powershell
python -m tools.benchmark_pluto_rx --uri usb:1.54.5 --rates 1000000,2000000,4000000,6000000,8000000
```

## 共通IQストリーム検証

`PlutoReceiver`の連続Producerをdirect USBで開始・停止・再開し、consumer cursorで全ブロックを検査しました。

| 条件 | 結果 |
|---|---|
| 4 MSPS、3秒 + 再開 | overrun 0、missed block 0、sequence error 0、sample index error 0、正常停止、stream ID更新 |
| 6 MSPS、5秒 + 再開 | overrun 0、missed block 0、sequence error 0、sample index error 0、正常停止、stream ID更新 |

```powershell
python -m tools.validate_common_iq_stream --sample-rate 6000000 --duration 5
```

## HighSpeed TA統合検証

### IQ Filter統合後（2026-08-02）

4次Butterworth complex IQ filterへ移行後、direct USB `usb:1.54.5`、4 MSPS、RBW 1 MHz、Free Runで再検証しました。

- Time Span 100 ms、Continuous 2秒: 18 records、各98 display points、7,929,856 samples受信
- ring overwritten blocks: 0
- analysis job queue最大使用量: 1、終了時pending/job/result: 0
- Single途中からContinuousへ変更: 6 records更新
- Time Span 100 msから50 msへ変更して再開始: 13 records更新、各49 display points
- 変更後もring overwritten blocks: 0、例外なし
- host上の合成complex64 4,000,000 samplesに対するIQ filter単体処理: 約45 ms

Sweep SAは2.440 GHz、RBW 1 MHz、RMS Detectorで1 point取得を実施し、256 samplesの自動取得、IQ filter後detector系列生成、結果出力まで例外なく完了しました。入力信号が既知ではないため、振幅精度と3 dB帯域の実機検証には含めません。

この結果は処理統合と短時間の追従性確認です。既知CW/noise/burstによるRBW shape、ENBW、rise/fall time、旧校正との差は未検証です。

### FFT非依存display bucket（2026-08-02）

Time Span初期値を10 msへ変更し、record長とdisplay bucketをFFT sizeから分離した後、direct USB、4 MSPS、RBW 1 MHz、Free Run Continuousを2秒実行しました。

- record長: 40,000 samples（FFT整列なし）
- display: 各record 1000 points、約10 µs/point
- publish: 188 records
- RX: 121 blocks、7,929,856 samples
- ring overwritten blocks: 0
- analysis job queue最大使用量: 2
- 終了時pending/job/result queue: 0
- 例外なし
- Single途中からContinuousへの変更後: 65 records更新
- Continuous中のTime Span再設定後: 65 records更新
- Power Level Auto、level 0 dBFS、timeout 50 ms、1秒: forced 10 records、ring上書き0、queue滞留0

100 records/s、各1000 pointsに近い条件でも短時間は受信へ追従しました。長時間動作、画面表示環境でのCPU/GPU負荷、既知burstのPeak保持は未検証です。

ヘッダーを`IQ Samples / Plot Points / Plot dt`へ分離後、同じ4 MSPS、10 ms、1000 pointsで0.3秒のoffscreen GUI統合試験を実施しました。19 recordsを更新し、1,114,112 samples受信、ring上書き0、queue最大2、例外なしでした。

### Gaussian IQ Filterデフォルト化後（2026-08-02）

Sweep/TA共通IQ Filterを4次ButterworthからGaussian FIRへ変更後、direct USB `usb:1.54.5`、4 MSPS、RBW 1 MHz、Time Span 10 ms、Free Run Continuousを2秒実行しました。

- Gaussian FIR: 11 taps、群遅延5 samples、ENBW約1.0645 MHz
- publish: 188 records、各1000 points
- RX: 121 blocks、7,929,856 samples
- ring overwritten blocks: 0
- analysis job queue最大使用量: 2
- 終了時pending/job/result queue: 0
- 例外なし

同条件のButterworth時と同じrecord数・受信sample数・queue最大値で、短時間の処理追従性低下は見られませんでした。pytestではCWの両側3 dB幅、ENBW、tap対称性/DC gain、direct FIRと狭RBW FFT convolution双方のblock分割同値性を検証しています。既知CW/noiseを用いた実機shape/ENBW測定は未実施です。

Qtをoffscreenで動かし、実際の`RealtimeSpectrumWindow`、共通Producer、window assembler、解析job/result queue、描画publishまでを通しました。Time Spanは0.1秒です。

| Mode | 条件 | 主な結果 |
|---|---|---|
| Single | 4 MSPS | 約0.257秒で完了、13 blocks、表示98 points、ring上書き0 |
| Single | 6 MSPS | 約0.239秒で完了、16 blocks、表示147 points、ring上書き0 |
| Continuous | 4 MSPS、3秒 | 28 windows、182 blocks、11,927,552 samples、ring上書き0、job queue最大1 |
| Continuous | 6 MSPS、5秒 | 46 windows、434 blocks、28,442,624 samples、ring上書き0、job queue最大1 |

初回の4 MSPS Continuous終了時に、pyadi-iioの`Buffer.__del__`からaccess violationが1回発生しました。受信worker停止後に`rx_destroy_buffer()`を明示実行するよう`PlutoReceiver.close()`を修正し、4 MSPS・6 MSPSの再試験では再発していません。

### 実操作で判明した状態遷移不具合

SingleからContinuousへの切替、およびContinuous中のSweep Time変更後に表示が停止する不具合がありました。再開始処理がRX workerだけを停止してHSTA stream cursorを残していたため、次の開始を重複開始と誤認していたことが原因です。

状態変更時は`_stop_high_speed_ta_stream()`を通し、worker停止とcursor無効化を一体で行うよう修正しました。direct USB、4 MSPSで次の連続シナリオを再試験しています。

| 操作 | 修正後の結果 |
|---|---|
| Single取得中 → Continuous | 0.75秒間に6 windows更新 |
| Sweep Time 100 ms → 50 ms | 続く0.75秒間に12 windows更新 |
| 全シナリオ | ring上書き0、終了時例外なし |

### Power Trigger統合

共通`TriggerAcquisitionController`とHighSpeed TAのrecord consumerをdirect USB、4 MSPS、Time Span 100 ms、Position 50%で検証しました。

| 条件 | 結果 |
|---|---|
| Power Auto、到達不能な+20 dBFS、timeout 200 ms、1.5秒 | 3 records、forced 3、natural 0、ring上書き0 |
| Power Normal、-100 dBFS、Single | 1 record、forced 0、natural 1、ring上書き0 |

前者は診断CLIからUI範囲外の+20 dBFSを指定し、自然eventが絶対に成立しない条件でforced timeoutだけを検証しています。通常UIのLevel上限は0 dBFSです。

```powershell
$env:QT_QPA_PLATFORM = 'offscreen'
python -m tools.validate_hsta_hardware --sample-rate 6000000 --time-span 0.1 --continuous-duration 5 --timeout 7
python -m tools.validate_hsta_hardware --sample-rate 4000000 --time-span 0.1 --exercise-transitions
python -m tools.validate_hsta_hardware --sample-rate 4000000 --time-span 0.1 --continuous-duration 1.5 --trigger-kind power_level --trigger-run-mode auto --trigger-level-dbfs 20 --trigger-auto-timeout 0.2
```

## 解釈上の注意と次の検証

- アプリのsequence/sample indexは、libiioから正常に返ったブロックへ付ける連番です。Pluto、USB、libiio内部でデータが抜けても、常に検出できるわけではありません。
- 実効転送速度が要求値に一致することは必要条件ですが、無欠落の十分条件ではありません。
- 6 MSPSは短時間試験の境界値であり、温度、USB controller負荷、OS schedulingの変動余裕がありません。
- USBとRNDISは同じ物理デバイスなので、並列に試験してはいけません。

次は既知のPRBS/QPSKまたは連続カウンタ相当信号を入力し、10分以上の相関検証でsample slipを数えます。その後、block sizeとkernel buffer数を比較し、安全な最大sample rateを決定します。
