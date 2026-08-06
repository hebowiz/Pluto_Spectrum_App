# VSA FSKシンボル同期監査

最終更新: 2026-08-07

## 結論

監査で見つかったP0のfine timing未適用は解消した。8通りの整数timing phaseから
coarse時刻を選んだ後、Result Range全体から推定したfractional timing offset `tau`を
最終symbol-frequency、symbol time、Result Range境界、表示markerへ反映する。

P0では短いtraining patternで得た`-1.50 analysis sample`を固定幅によりrejectしていた。
P1でTransmit/Measurement Filterを共通化して再検証すると、この値はcoarse detectorの
開始位置を実際のpacket boundaryへ戻す有効な補正だった。現在は固定幅ではなく、半symbol
Estimation Range、timing cost曲率、frequency residual/deviation比で採否を決める。

端数遅延を注入した生成GFSKでは、既存の`tau`推定値だけで20 dB SNR時の平均
timing errorを0.269 analysis sampleから0.016 sampleへ低減できた。このため、
2FSK/GFSKについてPSKと同等クラスの実用的なsymbol timing精度を狙うことは可能と
判断する。ただしFSKとPSKではR&Sも推定モデルが異なり、同一アルゴリズムにはしない。

## R&S FPL1-K70の参照仕様

参照文書は`FPL_K70_VSA_UserManual_en_12.pdf` rev.12である。

- pp.112-114: Result Range抽出後にcoarse timing/scaling/frequency/phase recovery、
  symbol decision、reference生成、measurement filtering、fine synchronizationを行う。
- pp.121-123: fine synchronizationはmeasurement signalとideal reference signalの
  相関・誤差最小化で行う。Detected Data、Known Data、Patternをreference sourceに
  使用できる。短いpatternだけでは同期確度が下がる。
- pp.127-128: synchronizationはestimation range内でmeasurement/reference間の
  mean-square errorを最小化する。burst時はburstとResult Rangeの重複部から
  Run-In/Run-Outを除く。
- pp.139-141: FSKはreference instantaneous-frequency waveformに対してdeviation
  scale `B`、CFO `f0`、linear drift `fd`、fractional timing offset `tau`を同時推定する。
  timingはfrequency waveformだけから推定する。
- pp.141, 222-223: FSK/MSKの自動Estimation Points/SymbolはCapture Oversampling、
  すなわちsymbol centerだけでなく取得した全sample pointを使用する。
- pp.130, 219-220: FSKにはSymbol Rate Error補償とequalizerがない。FSK固有の補償は
  Carrier Frequency DriftとFSK Deviation Errorである。
- pp.297-298: short patternや短いestimation rangeは同期を不安定にする。Pattern同期が
  不適切ならDetected Dataを用いるかpattern/Result Rangeを長くする。

R&SのFSK frequency modelは概念的に次式である。

```text
f_meas(t) = B * f_ref(t - tau) + f0 + fd * t + noise
```

`f_ref(t)`はsymbol列、modulation mapping、TX frequency-pulse filterから生成される
連続時間の基準瞬時周波数であり、単純なsymbol centerの2値列ではない。

## 現行実装

対象は`pluto_sa/vsa/demod/gfsk.py`である。

1. IQを8 samples/symbolへresampleする。
2. 隣接IQのphase differenceからinstantaneous frequencyを生成する。
3. coarse pattern acquisition用に半symbolのmoving averageを適用する。
4. 8個の整数timing phaseでsymbol-frequency平均値を作る。
5. known patternとのnormalized correlationとeye openingでcoarse phaseを選ぶ。
6. patternからCFO、signed deviation、coarse driftを推定する。
7. tentative payload decisionを作り、sample-rate reference frequency waveformを再構成する。
8. 測定瞬時周波数とreference瞬時周波数へ同じMeasurement Filterを適用し、Result Range
   全sampleを使ってdeviation、drift、fractional `tau`を最小二乗fitする。
9. 採用された`tau`でsymbol-frequencyを補間し直し、decision、`symbol_time_s`、pattern
   start、Result Range、plot markerを同じfractional clockへ揃える。
10. 半symbol Range、cost曲率、residual/deviation比を満たさないtiming補正はrejectする。

PSK側はideal absolute reference waveformに対するcomplex-EVM objectiveでfractional
timing offsetとsymbol timing rateを同時最適化し、最適化後のcenterでIQを補間し直す。
この「最適化結果を最終measurementへ戻す」段階がFSKにはない。

## 定量検証

### Fractional timing Monte Carlo

条件:

- deterministic GFSK、BT=0.5、1 MSym/s、8 MS/s
- 68-symbol known pattern、200-symbol Result Range
- source IQへ0から1 sampleの一様な端数遅延を注入
- 各SNR 80 captures
- coarse errorは現在返されるpattern startと真値の差
- joint errorはcoarse errorへ既存`frequency_model_timing_offset_samples`を加えた値

| SNR | captures/fail | coarse MAE | coarse max | tau適用後MAE | tau適用後max | bit errors |
|---:|---:|---:|---:|---:|---:|---:|
| 30 dB | 80 / 0 | 0.2421 sample | 0.5355 | 0.0044 | 0.0146 | 0 |
| 20 dB | 80 / 0 | 0.2687 sample | 0.5900 | 0.0159 | 0.0439 | 0 |
| 15 dB | 80 / 0 | 0.2908 sample | 0.7496 | 0.0239 | 0.0732 | 0 |
| 10 dB | 80 / 0 | 0.4106 sample | 1.2613 | 0.0459 | 0.2043 | 0 |

8 MS/sでは1 sample=125 nsである。20 dB時の0.0159 sampleは約2.0 ns、symbol
periodの約0.20%に相当する。これは生成波形とreference modelが一致した条件であり、
実機のTX filter mismatch、clock error、multipathを含む保証値ではないが、既存joint
estimatorに十分な分解能があることを示す。

### 保存済みPluto BR fixture

`bluetooth_br_prbs9_pluto_16msps.npz`の68-bit access codeをAnalysis Bandwidth
off/1.2/1.5/2/3/5 MHzで解析した。全条件でcorrelation 0.996以上、pattern error 0。
推定fractional `tau`はanalysis grid上+0.005から+0.079 sampleで、帯域変更に対して
小さく安定していた。したがってこのfixtureでは大きなtimingずれは再現せず、UI上の
甘さは主として整数phase markerと、実captureごとのfilter/model mismatchを疑う。

## R&Sとの差分と改善順

### P0: 推定済みtauを最終結果へ適用（2026-08-06 実装済み）

- fractional centerでinstantaneous frequencyを再積分/補間してsymbol-frequencyを再計算する。
- `symbol_time_s`、Result Range sample境界、Power/Modulation symbol markerを同じcenterへ揃える。
- `pattern_start_sample`は互換用整数値を維持し、fractional startをmetadataへ追加する。
- tau適用後にreference fitを再実行し、decisionとtauを2～3回反復して収束させる。
- Result SummaryへFractional Timing、reject状態、Frequency Fit RMSを表示する。

実装後の自動回帰では、20 dB SNRで0.125/0.375/0.625/0.875 sampleの端数遅延を
与えた4条件すべてでpattern start誤差0.08 sample未満、100 symbolsの復号error 0を
確認した。保存済みPluto BR fixtureもAnalysis Bandwidth 1.2/1.5/2/3/5 MHzの全条件で
timing補正が採用され、pattern error 0、開始時刻の幅0.1 analysis sample未満を維持した。
P0時点では短い32-symbol DH1の`-1.50 sample`を誤推定としてrejectする回帰を置いたが、
P1の対称filter化後は高confidenceでpacket boundaryへ近づくことが確認できたため、固定幅
rejectを廃止した。現在の回帰は補正後startが既知packet boundaryの±1 source sample以内、
全packet symbols一致、無drift波形でDrift model棄却を確認する。

### P1: R&S型のmeasurement/reference waveform fitへ統一（2026-08-06 実装済み）

- TX filterからFSK reference frequency pulseを一つの共通実装で生成する。
- measurement signalとreference signalへ同じMeasurement Filterを適用する。
- symbol平均値ではなくCapture Oversamplingの全frequency sampleをobjectiveへ使う。
- `B / f0 / fd / tau`のjoint fitとno-drift fitを比較し、driftの過適合を拒否する。
- known patternはtraining symbolsとして固定し、pattern外はDetected Dataでreferenceを作る。

実装内容:

- `demod/fsk_reference.py`へETSI/R&S Appendix F.5のGaussian BT impulseを共通化した。
- 生成FSK/BR/EDRのGFSK部、coarse reference、fine referenceが同じTransmit Filter定義を使う。
- Auto Measurement FilterはTransmit BTと同値とし、測定瞬時周波数へ1回、referenceへ
  Transmit Filter後に1回適用する。
- known pattern CFO/Deviationとpacket-wide `B/f0/fd/tau`をCapture Oversampling全点でfitする。
- Driftあり／なしを個別最適化し、BIC改善に加えて推定区間のdrift excursionが
  no-drift residual RMSの50%を超える場合だけ採用する。
- BR/2-DH1/3-DH1生成fixtureは新しい解析Gaussian定義で再生成した。
- Result SummaryへDeviation Error、Timing Confidence、Drift Model、drift有無双方のRMSを追加した。

P1回帰結果:

- 自動テスト200件が成功した。
- 無drift生成DH1ではDrift modelを棄却して`0 kHz/ms`、既知の`150 kHz/ms`を
  注入した反転IQではDrift modelを採用し、許容差`25 kHz/ms`以内で復元した。
- 保存済みPluto BR fixtureのAnalysis Bandwidth 1.2/1.5/2/3/5 MHzではpattern error 0、
  `tau=+0.180..+0.220 sample`、Timing Confidence `0.215..0.328`となった。
- 同fixtureで一時的に得られた約`+22 kHz/ms`は、126 us区間の総変化が約2.8 kHzで
  no-drift residual RMS約9 kHzに対して小さいため棄却した。最終条件では全帯域0 kHz/ms。

### Pluto実機DH1反復確認（2026-08-07）

アプリのDH1設定を再現し、8 MS/s、3 ms capture、Analysis BW 2 MHz、32-symbol
pattern、366-symbol Result Range、Internal Gain 30 dBで30回＋診断追加20回を取得した。

- 最初の30回は全captureでmatch、pattern error 0、correlation 0.9902～0.9971。
- Timing Confidenceは0.117～0.704で、fine timingは全captureで有効だった。
- 診断追加20回のcandidate driftは中央値`+4.049 kHz/ms`、範囲
  `+0.728..+5.424 kHz/ms`で、符号と大きさは比較的一貫していた。
- 366 us区間のdrift excursion中央値は1.47 kHz、no-drift residual RMSは概ね
  5.4～9.9 kHzであり、全20回が品質gateでRejectedとなった。
- 18/20回はBICが改善したが、excursionがresidual RMSの50%未満だった。小さな実driftの
  可能性はあるものの、このpacket長で補正を適用するには効果が小さく、Rejectedは妥当。

表示はR&Sに近づけ、`Carrier Drift`へ候補推定値、`Applied Drift`へ品質gate通過後の
補正値を示す。Rejectedでも推定値を0へ潰さず、棄却理由、BIC差、excursion、双方の
residualをmetadataへ保持する。

### P2: estimation rangeと品質指標

- Burst Search結果とResult Rangeの重複部からramp/filter settlingを除外する。
- tau objectiveの曲率または近傍cost差からtiming confidenceを算出する。
- Result SummaryへFractional Timing Offset、FSK Sync Residual、使用symbol数を追加する。
- FSKにはR&SどおりSymbol Rate Errorを表示せず、必要なら別機能としてclock trackingを設計する。

## 受入基準案

- model一致のGFSK、8 samples/symbol、SNR 20 dBでtiming MAE < 0.05 sample。
- SNR 15 dBでtiming max < 0.15 sample、pattern bit error 0。
- Analysis Bandwidth ON/OFFで同一fixtureのsymbol time差 < 0.1 analysis sample。
- tau適用前後でpattern correlationを悪化させず、FSK reference-frequency RMS residualを低減する。
- 連続同一symbol、短pattern、BT mismatchではconfidence低下を報告し、過大な補正を適用しない。
- 保存済みBR fixtureおよびPluto live反復captureでsymbol marker位置とphase-difference clusterの
  sweep間分散が改善する。

## 判断

PSKと同じコードを流用するのではなく、R&SどおりFSK専用frequency-waveform modelを使う。
P0/P1によりR&S FSK frequency modelの中核と結果contractへの反映は実装された。次は
Measurement Filterの手動設定、Estimation Range、confidence threshold設定、FSK error/EVM
相当結果を追加する。実機captureではTX filter mismatchとclock/reference誤差を含むため、
BR連続測定でTiming Confidence、Drift採否、symbol cluster分散を継続評価する。
