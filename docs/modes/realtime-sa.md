# RealTime SA 仕様

## 目的

PlutoSDRからIQを連続取得し、現在の受信帯域をFFTスペクトラムとウォーターフォールでリアルタイム表示します。

## Pluto受信個体の選択

2台以上のADALM-Plutoが接続されている場合、RTSA起動時にhardware serial付きの選択dialogを
表示します。選択した`serial:<id>`はRTSA専用QSettingsへ保存し、次回dialogの初期選択に使います。
起動後も`SYSTEM > System > Device`から選択dialogを開けます。選択を変更すると現在の受信処理を
停止し、新しいPlutoへ中心周波数、sample rate、RF bandwidth、manual gain等の初期設定を適用した後、
現在のAnalyzer ModeをContinuousで再始動します。初期化に失敗した場合は旧Receiverへrollbackします。
VSG/VSAのdevice設定とは共有しないため、送信機と受信機へ同じ個体を誤って割り当てることを
避けられます。`PLUTO_SDR_URI`環境変数が指定されている場合はそれを最優先し、dialogを表示しません。
接続個体が1台だけの場合はそのserialを自動選択します。

2026-08-26の2台構成検証では、RTSAにdevice selectorがなく、自動選択された個体がVSGの送信個体と
重複したことで、RX workerのbuffer refillが`OSError: [Errno 110] host unreachable`で停止した
可能性が高いと判断しました。この競合を避けるため、上記の起動時選択を追加しました。

## 周波数・取得条件

- 最大表示Span: 55 MHz
- 初期Span: 20 MHz
- サンプルレート: `display_span_hz / (1 - 2 × guard_ratio)`
- 初期guard ratio: 0.04（FFT両端を各4%非表示）
- RF bandwidth: サンプルレートと同値
- 連続RX block size: 65536 samples以上（FFT sizeとは独立）
- FFT size選択肢: 64～16384の2のべき乗

PlutoReceiverの共通RX workerがメタデータ付きIQブロックをリングへ連続発行します。RTSA consumerは独立cursorで全ブロックを順次読み、ブロック境界をまたいでoverlap FFTの位相（開始sample位置）を維持します。初期保持容量は512ブロックです。ring overrunまたはproducer discontinuityを検出した場合は、連続しないIQを同じFFT窓へ混ぜず、ステータスへ`RX Discontinuity`を表示します。

## 信号処理

1. 必要に応じてIQ平均値を減算
2. 既定80% overlap（hopはRBWから決まるWindow Lengthの約20%）で連続IQを解析窓へ分割
3. Sweep SA／TAと同じGaussian complex IQ FIR係数をFFT長へゼロ埋めした解析窓を適用
4. FFTとfftshift
5. FFTサイズおよびcoherent gainで正規化
6. 絶対値二乗で電力化
7. GUI更新間に生成された全FFTをDetectorで周波数binごとに集約
8. FFT両端のguard領域を除外
9. dB変換、表示補正、既存Trace Mode処理

各FFT binは、中心周波数だけが異なる同一Gaussian複素フィルターの出力に相当します。RBWは両側3 dB bandwidth、ENBWは約`1.0645 × RBW`です。従来のFFT電力化後のGaussian畳み込みはRBW経路から除外しました。

Autoでは、指定RBWのGaussian係数が収まるWindow Lengthとguard除外後の表示bin密度からNFFTを最大16384まで自動決定します。それでもWindow Lengthが収まらない場合は収まる最小RBWへ制限し、ステータスの`Eff RBW`へ実効値と`limited`を表示します。

既定の目標overlapは80%、host側の処理量上限は10000 FFT/sです。目標を処理できない設定では、NFFT、Span、Sample Rate、RBWを強制変更せずhopを広げます。ステータスへWindow Length、実overlap、FFT rate、表示1 frame当たりのFFT数を表示し、overlap低下時は`Reduced overlap`、hopがWindow Lengthを超えて未解析区間が生じる場合は`Analysis gaps`、時間被覆率と警告を表示します。この上限は2026-08-29時点の開発PCでcomplex64 batched FFTが約17000 FFT/sだった実測に対し、GUI処理の余力を残した値です。

DetectorはGUI更新間に得たFFT群へ適用します。

- Sample: 最後のFFT
- Peak: binごとの最大linear power
- Negative Peak: binごとの最小linear power
- Average: binごとのlinear power平均
- RMS: IQ電圧のRMS二乗に相当するlinear power平均

AverageとRMSは現段階では入力がFFT後のlinear powerなので同値です。名称は将来の測定量定義と一般的な計測器UIとの対応のため分けています。Detector出力後のLive／Max Hold／Average等のTrace Modeは既存実装を変更していません。

## 表示

- Spectrum、Waterfall、または両方を選択できます。
- Waterfallの初期履歴は300行、入力範囲は1～1000行です。
- Waterfallは周波数方向を4点ごとに間引きます。
- Waterfallは測定レンジの下端15%を暗いNavyへ固定し、15～80%を多色表示、80%以上をRedへ飽和表示します。
- Persistenceを利用でき、Fast／Medium／Slowの減衰を選択できます。
- Live、Max Hold、Averageの4トレースと4マーカーを利用できます。
- 約1秒ごとにステータス表示を更新します。

## Continuous / Single

- Continuous: RXスレッドとGUIタイマーを継続動作させます。
- Single: 開始後に到着した連続IQからoverlap FFT群を生成し、最初の表示frameを1回描画した後、タイマーとRXスレッドを停止します。

## 初期値

| 項目 | 値 |
|---|---:|
| Center | 2.440 GHz |
| Span | 20 MHz |
| FFT Parameters | Auto（既定Span/RBWではNFFT 2048） |
| RBW | 1 MHz |
| Target Overlap | 80% |
| FFT processing limit | 10000 FFT/s |
| Update Interval | 0 ms（RTSAではGUIを最大約60 FPSへ制限） |
| Waterfall History | 300 |
| Persistence Decay | Medium |

## 制限・注意事項

- 55 MHzを超えるSpanはモード移行時または設定反映時に55 MHzへ制限されます。
- RX ring overrunまたはUSB/DMA側の欠落データを再送・補間する機能はありません。
- overlap FFTを実装しましたが、処理上限により`Analysis gaps`と表示された設定では全sampleを解析しません。`Real-time`または`Reduced overlap`かつ`Time Coverage: 100%`であっても、USB/DMA欠落がないことは別途`RX Discontinuity`と実機条件で確認します。
- WB RTSAは今回のoverlap consumerの対象外で、従来のchunk取得方式を維持します。
- 表示値は校正された相対的なFFT電力をdBmとして扱う実装で、絶対精度は校正条件に依存します。

# 2026-08-29: Span / RBW 主体の自動FFT設計

通常のRealTime SAでは、ユーザーが指定する主要な測定条件を`Span`と`RBW`に整理し、FFT内部パラメータの既定モードを`Auto`としました。WideBand RT SAのchunk処理は今回の対象外で、従来仕様を維持します。

- RBWからGaussian解析窓の非ゼロ長（Window Length）を決定します。
- NFFTはWindow Lengthとは独立に、解析窓が収まることと、guard除外後に最低1024表示binを確保することの両方から2のべき乗へ自動決定します。短い窓はzero paddingされます。
- Hop SizeはWindow Lengthに対する目標80% overlapから求めます。FFT処理上限を超える場合はHopを広げ、Span/RBWを勝手に変更しません。
- 時間被覆率は`min(1, Window Length / Hop Size)`で算出します。100%未満ではステータスに`WARNING: time-domain observation gaps`を表示します。
- 連続IQからのFFT生成はGUI表示FPSと独立して進み、表示期間中に得られた複数FFTをDetectorで集約します。
- `FFT Parameters > Advanced`を選ぶと従来のFFT Size個別設定を利用できます。手動NFFTが指定RBWの解析窓を収容できない場合だけ、安全のためNFFTを拡張します。

ここでWindow Lengthは時間分解能・POI・overlapの基準、NFFTは周波数表示gridの基準です。zero paddingされたNFFT全長を時間観測窓として扱わないことが重要です。
