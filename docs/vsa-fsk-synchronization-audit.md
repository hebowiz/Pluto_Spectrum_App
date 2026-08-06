# VSA FSKシンボル同期監査

最終更新: 2026-08-06

## 結論

監査で見つかったP0のfine timing未適用は解消した。8通りの整数timing phaseから
coarse時刻を選んだ後、Result Range全体から推定したfractional timing offset `tau`を
最終symbol-frequency、symbol time、Result Range境界、表示markerへ反映する。

短いtraining patternとTX/reference filter mismatchがある生成DH1では、joint fitが
`-1.50 analysis sample`の誤った隣接解を選ぶケースも確認した。整数phaseを全探索した
fine correctionとして`|tau| > 0.75 sample`はrejectし、推定値は診断用に残しながら
適用値を0とする安全策を追加した。

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
3. 半symbolのmoving averageを適用する。
4. 8個の整数timing phaseでsymbol-frequency平均値を作る。
5. known patternとのnormalized correlationとeye openingでcoarse phaseを選ぶ。
6. patternからCFO、signed deviation、coarse driftを推定する。
7. tentative payload decisionを作り、sample-rate reference frequency waveformを再構成する。
8. Result Range全sampleを使い、deviation、drift、fractional `tau`を最小二乗fitする。
9. 採用された`tau`でsymbol-frequencyを補間し直し、decision、`symbol_time_s`、pattern
   start、Result Range、plot markerを同じfractional clockへ揃える。
10. `|tau| > 0.75 sample`は推定値を保持したまま適用をrejectする。

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
短い32-symbol DH1で発生する`-1.50 sample`の誤推定はrejectされ、正しいcoarse startと
全packet symbolsが維持されることを回帰テストへ固定した。

### P1: R&S型のmeasurement/reference waveform fitへ統一

- TX filterからFSK reference frequency pulseを一つの共通実装で生成する。
- measurement signalとreference signalへ同じMeasurement Filterを適用する。
- symbol平均値ではなくCapture Oversamplingの全frequency sampleをobjectiveへ使う。
- `B / f0 / fd / tau`のjoint fitとno-drift fitを比較し、driftの過適合を拒否する。
- known patternはtraining symbolsとして固定し、pattern外はDetected Dataでreferenceを作る。

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
現行joint estimatorはその中核に近く、P0は比較的小さい変更で効果が見込める。P1以降は
FSK error/EVM相当結果の基礎にもなるため、表示だけを補正する暫定対応ではなくDSP result
contractから一貫して実装する。
