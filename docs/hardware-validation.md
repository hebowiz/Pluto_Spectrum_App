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

#### LiteVNA 2441 MHz CW位相連続性

LiteVNAから2441 MHz CWを入力し、Pluto Center 2440 MHz、12 MSPS、RF BW 12 MHz、manual gain 30 dBで+約1.036 MHz IFとして取得しました。PlutoとLiteVNAの基準clockは非同期のため、公称1 MHzとの差と緩やかなdriftは許容し、各sampleの複素積`x[n] × conj(x[n-1])`から平均位相進行を除いた残差を評価しました。

| Snapshot | 結果 |
|---|---|
| 10 ms、120,000 samples × 6 buffers | 各buffer内outlier 0、本取得buffer最大残差3.86°、ADC clip 0%、FFT peak SNR約95 dB |
| 100 ms、1,200,000 samples × 6 buffers | 各buffer内outlier 0、全buffer最大残差7.17°、本取得buffer最大5.72°、ADC clip 0%、FFT peak SNR約103 dB |

平均位相進行は約31.07°/sampleであり、単発1 sample slipなら同程度の位相ジャンプになる条件です。noise MADの6倍と位相進行75%の大きい方をsample slip候補閾値とし、検出は0件でした。10 ms試験では5→6 buffer境界に155.9°の不連続が1回あり、100 ms試験の境界は最大2.75°でした。Single Snapshotはbuffer境界をrecord内へ含めないため、前者も本record内部には影響しません。

この結果は12 MSPSで10/100 msの単一buffer内部が位相連続であることを強く支持します。ただしCW位相は2π周期なので、位相回転が偶然整数周期に近くなる複数sample欠落を原理的に見逃す可能性があります。最終的な無欠落保証にはPRBS/counterが必要です。再現用に`python -m tools.validate_snapshot_phase`を追加しました。

#### 16～40 MSPS追加試験

4 Msymbol/s解析を想定して16 MSPSを本命条件とし、20 MSPSまで10/100 ms、余裕確認として30/40 MSPSを10 msで試験しました。30/40 MSPSでは1 sample slipの位相を大きくするためPluto Centerを2438 MHzとし、約+3.04 MHz IFで評価しました。

| Sample Rate | Snapshot | buffer内結果 | 本取得buffer最大残差 |
|---:|---:|---|---:|
| 16 MSPS | 10 ms / 160,000 samples | 6 buffers全てslip候補0 | 3.21°（1 sample進行23.28°） |
| 16 MSPS | 100 ms / 1,600,000 samples | 6 buffers全てslip候補0 | 6.89° |
| 20 MSPS | 10 ms / 200,000 samples | 6 buffers全てslip候補0 | 3.16°（1 sample進行18.62°） |
| 20 MSPS | 100 ms / 2,000,000 samples | 6 buffers全てslip候補0 | 12.08°未満、slip閾値約13.97° |
| 30 MSPS | 10 ms / 300,000 samples | 6 buffers全てslip候補0 | 5.55°（1 sample進行36.42°） |
| 40 MSPS | 10 ms / 400,000 samples | 6 buffers全てslip候補0 | 3.44°（1 sample進行27.30°） |

ADC clipは全条件0、FFT peak SNRは概ね91～104 dBでした。16/20/30 MSPSでは後半buffer境界に最大約120～180°の位相飛びがあり、40 MSPSでも最大約46°でした。高sample rateでも単一buffer内部は良好ですが、buffer連結を無欠落recordとして扱わない制約は維持します。

実アプリ統合経路では16 MSPS、RBW 4 MHzについて、100 µs（1,600 IQ samples、約0.021秒）と10 ms（160,000 samples、約0.189秒）をSingle表示まで確認しました。いずれも6 blocks、ring上書き0、analysis queue最大1、終了時滞留0です。

#### 16 MSPS Continuous Island（2～3 ms）

16 MSPS、RBW 4 MHz、Free Run Continuousで、96,000-sample RX buffer内部だけからrecordを生成しました。

| Time Span | buffer構成 | 実機結果 |
|---:|---|---|
| 2 ms | 32,000 samples × 3 records | 1秒で60 buffers受信、warm-up後164 records表示、ring上書き0、analysis queue最大3 |
| 3 ms | 48,000 samples × 2 records | 2秒で122 buffers受信、warm-up後234 records表示、ring上書き0、analysis queue最大2 |

両条件とも終了時job/result/pending queueは0、例外なしでした。record数は2 msで`(60 - 5 warm-up) × 3 = 165`に対し停止時刻の関係で164、3 msでは`(122 - 5) × 2 = 234`と一致しています。

同じ96,000-sample bufferを20回連続取得したLiteVNA CW試験では、全20 buffersでsample slip候補0、buffer内最大位相残差7.54°、1 sample進行約23.29°でした。一方、19個のbuffer境界では最大約164°の位相飛びを確認しました。この結果はbuffer内部recordだけを採用し、境界でtimeline/filterをresetする設計を支持します。

#### 16 MSPS Power Trigger Buffer Island（2～3 ms）

Power Triggerではevent探索範囲を広げるため、262,144 samples以下に収まる最大のrecord整数倍を1 RX bufferとしました。

| 条件 | buffer構成 | 実機結果 |
|---|---:|---|
| Auto Continuous、2 ms、Level +20 dBFS、timeout 200 ms | 32,000 samples × 8＝256,000 | 1.55秒で24 buffers、forced 3、natural 0、6,144,000 samples、ring上書き0、job/result queue最大1、終了時滞留0 |
| Normal Single、3 ms、Level -100 dBFS、Position 50% | 48,000 samples × 5＝240,000 | 0.327秒で7 buffers、natural 1、forced 0、1,680,000 samples、ring上書き0、job queue最大1、終了時滞留0 |

Autoは到達不能なlevelでもhost時刻timeoutにより完全な2 ms recordを生成し、Normal Singleは自然edgeからpre/postを同一buffer内で完成して停止しました。この試験はPower Islandの制御・解析経路を確認するもので、buffer間blind time中のevent捕捉は保証しません。

統計出力追加後の再試験でも、Auto Continuousは23 islands、forced 3、ring上書き0、edge棄却0、推定blind time 1.219秒、Normal Singleは1 measurement island、natural 1、ring上書き0、edge棄却0で完了しました。Autoの推定blind timeは、各bufferのhost受領間隔からbuffer標本時間を差し引いた累積値です。

#### dBm Trigger Level・グラフ線移行後（2026-08-02）

周波数別校正CSVを自動loadした状態で、dBm設定から内部dBFSへの変換と水平線を含むoffscreen GUI経路を再検証しました。

| 条件 | 換算・表示 | 実機結果 |
|---|---|---|
| 16 MSPS、2 ms、Auto Continuous、100 dBm、timeout 200 ms | 内部94.650 dBFS、Trigger line 100.0 dBm、visible | 0.8秒、forced 1、natural 0、ring上書き0 |
| 16 MSPS、3 ms、Normal Single、-100 dBm | 内部-105.350 dBFS、Trigger line -100.0 dBm、visible | 0.386秒、natural 1、forced 0、ring上書き0 |

100 dBmは到達不能levelとしてAuto timeoutを、-100 dBmは自然triggerを確認する診断条件です。両条件とも校正・入出力補正を含む変換値がcontrollerへ渡り、グラフ線はユーザー指定dBmを保持しました。

#### Power Trigger GUI処理時間短縮（2026-08-02）

Power Trigger detectorがGUIスレッド上でcomplex IQを1 sampleずつPython判定しており、合成bufferの単体測定で96,000 samplesは29.8 ms、240,000 samplesは81.5 ms、256,000 samplesは88.5 msを占有していました。minimum duration 1 sampleの判定をNumPy threshold検索へ置き換えた結果、同条件はそれぞれ1.45 ms、3.70 ms、4.12 msとなりました。

Rising/Falling/Eitherについて、ランダム2,000 samplesを2 blocksへ分割し、hysteresis 2.5 dB、holdoff 7 samplesのベクトル化結果を旧sample状態機械と比較してevent位置・測定値が一致することをpytestで確認しました。

16 MSPS、2 ms、Power Normal Continuous、-100 dBmの実機負荷試験では、1秒にnatural 186 recordsを描画する意図的な高負荷条件でもGUI update callbackはp95 2.38 ms、最大16.70 ms、ring上書き0でした。Auto 100 dBmのtimeout試験ではp95 0.25 ms、最大0.76 msでした。

#### RBW後Power Triggerへの修正（2026-08-02）

Power Normal/Rising、Level -25 dBmで、RBW後traceが約-40～-31 dBmにもかかわらずnatural triggerする現象を再現しました。修正前は内部threshold -30.350 dBFSに対しraw発火sampleが-30.099～-27.116 dBFS（校正後約-24.75～-21.77 dBm）まで瞬間的に上昇する一方、同recordのRBW後表示最大は-30.735 dBmでした。dBm換算ではなく、filter前triggerとfilter後表示の経路差が原因です。

Power Triggerへ表示と同じGaussian complex IQ RBW filterを適用し、判定用filtered IQと保存用raw IQを分離しました。16 MSPS、RBW 4 MHz、2 msで再試験した結果は次のとおりです。

- Level -20 dBm、Normal Continuous、1秒: natural/forcedとも0件、23 buffers、ring上書き0。
- Level -25 dBm、Normal Continuous、0.8秒: natural 8件。filter後発火sampleは-29.797～-27.518 dBFS（校正後約-24.45～-22.17 dBm）で全件threshold以上。record表示最大は-20.537 dBm。
- 4 MSPS、RBW 1 MHz、10 ms、Level -25 dBm、Normal Continuous、0.8秒: natural/forcedとも0件、47 buffers、3,080,192 samples、ring上書き0。GUI callbackはp95 0.42 ms、最大4.37 ms。
- 修正後のGUI callbackは-25 dBm試験でp95 0.29 ms、最大2.10 ms、-20 dBm試験でp95 0.31 ms、最大13.00 ms。

TriggerはRBW filter後の各sampleを判定します。表示DetectorがSampleの場合はbucket末尾sampleだけを描くため短い超過が見えないことがあり、目視検証はPeak Detectorで行います。Trigger Level modalを開く前にProducer/timerを停止して解析queueを破棄する変更も加え、`-`等の未確定入力中に測定結果が進まないようにしました。

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

前者はdBm UI移行前の診断CLIから+20 dBFSを指定し、自然eventが絶対に成立しない条件でforced timeoutだけを検証した履歴です。現在のUI/CLIは補正済みdBmで指定し、内部で同等のdBFSへ変換します。

```powershell
$env:QT_QPA_PLATFORM = 'offscreen'
python -m tools.validate_hsta_hardware --sample-rate 6000000 --time-span 0.1 --continuous-duration 5 --timeout 7
python -m tools.validate_hsta_hardware --sample-rate 4000000 --time-span 0.1 --exercise-transitions
python -m tools.validate_hsta_hardware --sample-rate 4000000 --time-span 0.1 --continuous-duration 1.5 --trigger-kind power_level --trigger-run-mode auto --trigger-level-dbm 100 --trigger-auto-timeout 0.2
```

### RTSA/WideBand Gaussian FFT filter bank（2026-08-02）

Sweep/TAと同じGaussian FIR係数を用いるFFT filter bankへ移行後、direct USB `usb:1.54.5`、Center 2.440 GHz、Span 20 MHz、FFT 4096、RBW 1 MHzで各モードを別processから確認しました。

| モード | 結果 |
|---|---|
| RealTime SA | 完了、8 blocks、ring上書き0、3768 points、実効RBW 1 MHz、ENBW 1.0645109 MHz、support 49 samples、FFT制限なし |
| WideBand RT SA | 完了、2 chunks、3770 points、実効RBW 1 MHz、ENBW 1.0645109 MHz、support 49 samples、FFT制限なし |

Windows上では同一process内でdirect USB contextを閉じて直ちに再openするとlibiioの解放問題があるため、検証toolは`--mode rtsa`と`--mode wideband`を別processで実行できます。今回の入力レベルは管理していないため、これは取得・解析・合成経路とmetadataの統合確認です。Gaussianの3 dB shape、ENBW、絶対振幅、旧校正との差は既知CW/noiseによる追加検証が必要です。

### WideBand Chunk Width 40 MHz（2026-08-02）

direct USB `usb:1.54.5`、Center 2.440 GHz、測定Span 80 MHz、Chunk Width 40 MHz、FFT 4096、RBW 1 MHzを検証しました。下端から2 chunksで合成を完了し、6030 pointsを出力しました。Pluto実設定はSR 54,347,825 Hz、RF BW 54,347,826 Hz、実効RBW 1 MHz、ENBW約1.06457 MHz、Gaussian support 117 samples、FFT制限なしです。

これは最大Chunk Widthにおける設定受付・リチューン・2 chunk合成の確認です。入力信号を管理していないため、広幅時のノイズフロア、mirror、roll-off、chunk境界振幅差の定量評価ではありません。

### VSA Pluto Run Single（2026-08-05）

R&S風`Input / Frontend`および`Signal Capture`設定から、共通`PlutoReceiver`と`TriggerAcquisitionController`を通してVSA finite captureを取得した。direct USB接続のPlutoへCenter 2441 MHz、Capture Oversampling 8 samples/symbol、Symbol Rate 1 Msym/s、要求SR 8 MS/s、RF BW 8 MHz、Capture Length 3 ms、Internal Gain 30 dBを設定した。

| 項目 | 実機結果 |
|---|---:|
| Record length | 24,000 samples |
| Duration | 3.000000 ms |
| Pluto sample-rate readback | 7.999999 MS/s |
| Nominal usable I/Q bandwidth | 6.399999 MHz |
| Center | 2441.000000 MHz |
| Discontinuity reason | None |
| External ATT / Gain | 30 dB / 0 dB |
| Input correction | 0 dB（30 dB ATT - 30 dB internal gain） |
| Median / peak power | -62.000 / -50.861 dBm |

電力はPluto SAと共通の`InputPowerCorrection`を用い、`20log10(|IQ|) - 62 dB + Ext ATT - Internal Gain - Ext Gain`でDUT側reference planeのdBmへ変換した。今回の値は既知CWによる確度検証ではなく、接続・readback・record length・換算経路の統合確認である。frequency-dependent calibrationは未適用。既知レベル信号による絶対振幅検証を別途行う。

## 解釈上の注意と次の検証

- アプリのsequence/sample indexは、libiioから正常に返ったブロックへ付ける連番です。Pluto、USB、libiio内部でデータが抜けても、常に検出できるわけではありません。
- 実効転送速度が要求値に一致することは必要条件ですが、無欠落の十分条件ではありません。
- 6 MSPSは短時間試験の境界値であり、温度、USB controller負荷、OS schedulingの変動余裕がありません。
- USBとRNDISは同じ物理デバイスなので、並列に試験してはいけません。

次は既知のPRBS/QPSKまたは連続カウンタ相当信号を入力し、10分以上の相関検証でsample slipを数えます。その後、block sizeとkernel buffer数を比較し、安全な最大sample rateを決定します。

# 2026-08-05: Bluetooth BR FSK symbol-plot asymmetry

An 8 MS/s, 1 ms Pluto VSA Single capture at 2441 MHz was compared with the
`FSK Symbol Phase Difference` display path.  Its angle is calculated from each
demodulated symbol-frequency value using `exp(j*2*pi*f_symbol/R_symbol)`.
The radius now retains the symbol-instant IQ magnitude with one global RMS
normalization, so angular asymmetry and radial amplitude spread are separate.

One directly captured packet matched the configured 32-symbol access pattern
at 99.63 %.  Its zero/one cluster medians were -148.17 kHz and +149.11 kHz,
with a midpoint of +0.47 kHz.  This confirms that neither Pluto nor the basic
phase-difference display equation inherently produces the asymmetry.

Repeated captures exposed an intermittent carrier-drift estimation problem.
Nominal cases estimated about +40 kHz/ms and left the cluster midpoint near
-3 kHz.  Other captures of the same transmitted waveform estimated
+218...+305 kHz/ms and left the midpoint about -11...-17 kHz.  In one
+280 kHz/ms case, reconstructing the pre-compensation symbol frequencies
showed the local upper/lower midpoint changing only from about +65 kHz to
+43 kHz over the packet.  Applying the estimated positive drift instead moved
the corrected midpoint from about +36 kHz to -57 kHz.  The large fitted drift
therefore did not represent the observed common movement of the two FSK tones
and made the symbol plot less symmetric.

The implementation at the time also applied the decision-directed drift estimate
to `PatternSearchResult.measured_symbols` unconditionally.  This means the FSK
Symbol Plot receives drift-compensated values even when Demodulation >
Compensate for > Carrier Frequency Drift is disabled (the default).  The
sample-level carrier-corrected IQ path does honor that checkbox, so the two
result paths are inconsistent.

The subsequently implemented correction:

- retain the pattern-anchored CFO measurement;
- keep CFO-only and CFO-plus-drift symbol-frequency results separate;
- select between them using `compensate_carrier_frequency_drift` for every
  plot/result path;
- replace the symbol-rate decision-directed drift fit with the sample-rate
  reconstructed-reference model described below;
- add regression coverage for payload sequences whose symbol values correlate
  with time, because those sequences currently allow deviation-model mismatch
  to leak into the linear drift coefficient.

The IQ DC magnitude in the symmetric direct capture was about 3.35 % of RMS.
The normal VSA session removes the complex mean before pattern analysis when
`remove_dc` is enabled, and the direct unremoved capture was nevertheless
symmetric, so DC offset was not the primary cause in this observation.  Tuning
the wanted carrier exactly to the Pluto LO still deserves separate DC-notch
and image-rejection characterization.

## 2026-08-05: sample-rate joint-fit drift validation

The FSK estimator was changed from symbol-rate decision-directed regression to
a capture-oversampling reference-frequency fit modeled after R&S VSA.  Initial
3 ms tests exposed a second issue: multiple transmitted packets can occur in
one capture, and the burst detector can combine adjacent activity.  Fitting the
entire detected range produced occasional +120...+200 kHz/ms drift and
126...155 kHz deviation estimates even when the requested result was only one
366-symbol DH1 packet.

The estimation range is now capped at the configured Result Range, matching the
R&S burst/result-range rule.  Eight subsequent 3 ms captures, each analyzed as
one complete 366-symbol result, produced:

- I/Q correlation: 99.53...99.67 %
- CFO: +18.92...+19.19 kHz
- carrier drift: +12.05...+13.31 kHz/ms
- measured deviation: 167.90...168.72 kHz

The earlier bimodal estimates did not recur.  Capture Length must still be long
enough to contain a complete packet; for this asynchronous Free Run BR test,
3 ms is preferred over 1 ms.  Result Length then limits parameter estimation to
the intended packet rather than every packet present in the capture.

## 2026-08-05: live 2DH1 phase stability

Ten 3 ms Pluto captures of the fixed 2DH1 signal were evaluated independently
of the current ten-symbol PSK drift fit.  The application estimator varied by
roughly -814...+1774 kHz/ms.  A data-removing pi/4-DQPSK fourth-power estimate
over the complete 244-symbol Result Range measured only -0.72...+1.03 kHz/ms.
After removing that small linear component, residual phase RMS was
2.26...6.45 degrees.

This separates a modest real receive/transmit phase spread from the dominant
analysis artifact: the large visible capture-to-capture constellation changes
were caused mainly by fitting drift to only ten known symbols.

After implementing full-Result-Range Mth-power/detected-data synchronization,
ten further live captures measured CFO +21.10...+21.37 kHz, drift
-1.72...+0.80 kHz/ms, and phase RMS 2.32...4.86 degrees.  Drift compensation
ON and OFF produced effectively the same phase RMS because the physical drift
was small.  The previous hundreds-to-thousands of kHz/ms variation no longer
occurred.

### Rare 8DPSK false-drift reproduction and final guard

With the fixed 2441 MHz 3DH1 signal, Pluto gain 0 dB, external attenuation
30 dB, 8 MS/s and a 3 ms capture, the issue was reproduced independently of
the GUI. Two normal captures reported -1.1 and +3.6 kHz/ms. A third capture
still decoded its ten-symbol pattern with 99.24% correlation and zero errors,
but the old fit reported -736.5 kHz/ms and moved CFO from its normal +21 kHz to
+44.3 kHz. This demonstrates an estimator alias rather than receiver overload
or ambient 2.4 GHz ingress.

After circular robust drift estimation, CFO-only model fallback, and fractional
symbol timing were added, twenty consecutive live 8DPSK captures produced:

- pattern correlation: 98.49...99.91%;
- pattern errors: zero in all captures;
- CFO: +20.61...+21.02 kHz;
- accepted/reported drift: 0 kHz/ms because none of the candidate slopes
  improved the CFO-only weighted phase residual;
- fractional timing correction: -0.50...+0.40 analysis samples;
- weighted phase residual RMS: 2.52...7.67 degrees.

No false high-drift solution occurred. Reporting zero here means drift was not
supported by the Result Range error comparison; it does not assert that the
physical oscillator has mathematically exact zero drift.

### Joint-EVM 2DH1 synchronization validation

The preceding CFO-only fallback was superseded because it also rejected valid
small drift and left Carrier Drift fixed at zero. A joint complex-EVM fit of
fractional timing, timing rate, CFO, and drift was then tested with twenty
fixed-channel 2DH1 captures under the same 2441 MHz, 8 MS/s, 3 ms, Pluto gain
0 dB and external attenuation 30 dB conditions.

The initial +/-0.75-sample timing bound was reached by several captures, proving
that short-pattern coarse correlation could be more than one sample from the
eye center. Expanding fine timing to the non-ambiguous part of one 8-sample
symbol removed the boundary hits. A final twenty-capture run produced:

- pattern errors: zero in all captures;
- CFO: +20.64...+21.47 kHz;
- Carrier Drift: -3.25...+3.02 kHz/ms, no zero locking;
- phase residual RMS: 2.42...3.17 degrees;
- complex synchronization EVM RMS: 5.58...7.41%;
- fractional timing offset: -0.40...+1.47 analysis samples;
- symbol-rate error: -17.8...+56.0 ppm.

The earlier outlier captures near 6 degrees phase residual and 14% EVM did not
recur in this run. These values characterize synchronization repeatability,
not yet standards-compliant Bluetooth EVM accuracy.

### 3DH1 absolute-vector versus differential-phase validation

The IQ Trajectory symbol-point spread was independently calculated through two
paths: directly from matched-filtered samples at the optimized demodulator symbol
times, and through the carrier-corrected IQ path used by the GUI. Across twelve
live captures the two absolute phase-spread results agreed within 0.14 degree,
excluding stale timing or a separate UI interpolation bug.

Across ten subsequent captures, differential 8DPSK phase error remained about
2.8...3.4 degrees RMS while absolute phase spread varied 3.35...10.82 degrees.
The cumulative sum of differential phase error reproduced absolute trajectory
phase error with 0.000-degree RMS mismatch for every capture. The varying spread
is therefore the accumulated differential phase-error random walk. It can contain
real transmitter/receiver phase noise plus residual synchronization error; it is
not evidence by itself that symbol timing selected a wrong sample.

### 3DH1 absolute-reference waveform synchronization

After replacing differential-decision-point synchronization with the reconstructed
absolute-reference waveform fit, an initial twenty-capture run reduced IQ
Trajectory phase spread from the earlier 3.35...10.82 degrees to
1.95...2.56 degrees and Sync EVM to 4.46...5.86%. One run still retained a
-21.19 kHz/ms local drift solution even though its trajectory looked good.

The optimizer was then changed to solve from both zero drift and the Mth-power
coarse drift, selecting the lower absolute-reference EVM result. A final twenty
fixed-channel 3DH1 captures at 2441 MHz, 8 MS/s and 3 ms produced:

- pattern errors: zero in every capture;
- I/Q correlation: 99.61...99.94%;
- CFO: +21.594...+21.654 kHz;
- Carrier Drift: -0.26...+0.23 kHz/ms;
- Symbol Rate Error: -22.7...+37.6 ppm;
- absolute IQ Trajectory phase spread: 1.97...2.61 degrees;
- absolute-reference Sync EVM RMS: 4.64...6.06%.

The prior tight/dispersed two-state trajectory behavior and high-drift outlier did
not recur. These are implementation repeatability figures for the connected test
source, not a standards-conformance EVM specification.
