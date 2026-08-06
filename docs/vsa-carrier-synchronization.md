# VSA Carrier周波数推定・補正仕様

最終更新: 2026-08-07

この文書は、Pattern SearchにおけるCarrier Frequency Offset（CFO）、carrier phase、linear frequency driftの計算方法と、表示・復調へ適用する補正の境界を記録する。R&S FPL1-K70 VSA User Manual rev.12のdemodulation process（pp.112-124）とDemodulation設定（pp.217-224）を用語・処理段階の参照モデルとする。

## 1. 周波数基準

CFOはAnalysis CenterへDDCした後のbaseband残差周波数である。

```text
estimated_carrier_frequency_hz = analysis_center_frequency_hz + CFO_hz
```

Analysis Channelを使わない場合はrecording centerを基準とする。CFOはRF frontendやreference clockの絶対確度を保証する値ではなく、選択されたpatternとcaptureに対するcoarse synchronization推定値である。

## 2. FSK / GFSK

### 2.1 瞬時周波数

隣接IQ sampleの位相差から瞬時周波数を計算する。

```text
f[n] = Fs / (2 pi) * arg(x[n] * conj(x[n-1]))
```

8 samples/symbolへresampleし、半symbol幅の移動平均を適用する。全timing phaseを試し、既知patternから生成したFSK levelとのnormalized correlationとeye openingからcoarse phaseと開始位置を採用する。Result Range全sampleを使うjoint fitでfractional timing offsetも推定し、採用されたoffsetは最終symbol-frequency、symbol time、Result Range境界、表示markerへ一貫して反映する。

coarse pattern acquisitionは従来の半symbol integrate-and-dump相関を維持し、fine estimatorはR&SどおりMeasurement Filter通過後の全瞬時周波数sampleを使う。Gaussian時は測定瞬時周波数へMeasurement Filterを1回、ideal referenceへTransmit Filterと同じMeasurement Filterを順番に適用する。Auto時のMeasurement Filter BTはTransmit Filter BTと同値である。

Gaussian impulseはR&S Appendix F.5と同じ解析定義を使う。symbol periodを`T`とすると、sample grid上の標準偏差は次のとおりであり、周波数`B=BT/T`で振幅応答が-3 dBとなる。

```text
sigma_samples = sqrt(ln(2)) * samples_per_symbol / (2 pi BT)
```

fine timingの採否は固定±0.75 sampleではなく、半symbolのEstimation Range、最適点から±0.25 sampleにおけるcost上昇率、frequency fit residual/deviation比で判定する。Result Summaryの`Fractional Timing`は推定値とreject状態、`Timing Confidence`はcost上昇率、`Frequency Fit RMS`は採用modelとno-drift modelの残差を示す。R&Sとの差分と定量評価は[vsa-fsk-synchronization-audit.md](vsa-fsk-synchronization-audit.md)を参照。

### 2.2 CFO、Deviation、Driftの同時推定

pattern内のsymbol frequencyを次の線形modelへ最小二乗fitする。

```text
f_k = CFO + signed_deviation * m_k + drift_per_symbol * k
```

- `m_k`: 既知symbolを-1/+1へmappingした期待FSK level。
- GFSKではSignal DescriptionのGaussian BTで期待levelを整形する。
- `k`: pattern中央を0とした相対symbol番号。
- `CFO`: pattern中央時刻におけるresidual carrier frequency。
- `signed_deviation`: 実測周波数偏移。負の場合はIQ polarity inversionとして扱う。
- `drift_per_symbol`: 1 symbol当たりのlinear frequency変化。

十分なpacket後続部がある場合は、known patternを固定training dataとし、後続のtentative decisionから最大3回再fitする。このpacket-wide fitはsymbol判定とcoarse drift推定にだけ用いる。Result Summaryへ表示しsample単位補正へ用いるCFOは、送信symbolが確定しているknown pattern単独のfit値に固定する。tentative decisionでCFO基準を上書きしない。

P1ではknown patternのCFO/Deviationもsymbol平均だけでなく、Transmit FilterとMeasurement Filterを通したideal frequency waveformとの全sample最小二乗fitで更新する。packet-wide fitではDriftあり／なしを別々に最適化し、BICが改善し、かつ推定区間内のdrift excursionがno-drift residual RMSの50%を超える場合だけDrift modelを採用する。棄却時のApplied Driftは0だが、推定したCarrier Drift候補値は診断結果として保持・表示する。

```text
drift_hz_per_s = drift_per_symbol * symbol_rate_hz
```

symbol decisionでは次を差し引く。

```text
f_corrected(k) = polarity * (f_measured(k) - CFO - drift_per_symbol * k)
```

## 3. PSK

### 3.1 Non-differential PSK

pattern symbolに対する位相誤差をunwrapし、symbol番号に対して直線fitする。

```text
phase_error(k) = unwrap(arg(r_k * conj(s_k)))
phase_error(k) = phase_rotation + phase_slope * k
CFO_hz = phase_slope * symbol_rate_hz / (2 pi)
```

Result Range symbolは次で補正する。

```text
r_corrected(k) = r_k * exp(-j * (phase_rotation + phase_slope * k))
```

### 3.2 Differential PSK

隣接symbol間のphase incrementを用いる。

```text
d_k = r_k * conj(r_(k-1))
```

期待incrementとの位相誤差を直線fitする。increment上の定数項は1 symbol当たりのcarrier phase advanceなのでCFOとなり、傾きはfrequency driftとなる。

```text
CFO_hz = intercept * symbol_rate_hz / (2 pi)
drift_hz_per_s = slope * symbol_rate_hz^2 / (2 pi)
```

## 4. Sample単位のCarrier Corrected IQ

表示用IQは、推定基準時刻`t_ref`から各sample時刻までcarrier frequency modelを積分して補正位相を作る。

```text
dt = t - t_ref
phase_correction(t) = 2 pi * (CFO * dt + 0.5 * drift_hz_per_s * dt^2)
x_corrected(t) = x(t) * exp(-j * phase_correction(t))
```

Non-differential PSKでは推定したconstant phase rotationも`phase_correction`へ加える。FSKの`t_ref`はpattern中央、non-differential PSKはpattern先頭symbol center、differential PSKは最初のphase-increment区間のcenterとする。

UIの`Display Config > Carrier Display`で次を切り替える。

- `Raw IQ`: Result Rangeの取得IQからSpectrumとInstantaneous Frequencyを生成。
- `Carrier Corrected`: sample単位でCFOを除去したIQから再生成。既定値。

Powerは位相回転で変化しないためRaw IQを使用する。CFO、推定carrier frequency、driftはResult Summaryへ表示する。FSKでは`Carrier Drift`がjoint fitの推定候補値、`Applied Drift`が品質gate通過後に補正へ使用する値である。

## 5. Drift補正の扱い

`Demodulation > Compensate for > Carrier Frequency Drift`でsample単位補正へlinear driftを含める。CFO補正はCarrier Corrected表示で常に適用する。

現行drift推定はdecision-directedな一次modelであり、測定確度のcross-validationが未完了である。2026-08-04の固定BR fixture（Analysis BW 2 MHz）では、known pattern単独のCFOは約+20.0 kHzだった一方、旧実装のpacket-wide fitはCFOを-5.37 kHzへ移動させ、Carrier Corrected瞬時周波数に約25 kHzの中心ずれを残していた。現在はCFOをknown patternへ固定し、packet-wide fitからはcoarse driftだけを採用する。長いResult Rangeへのdrift外挿はなお過補正の可能性があるため、drift補正の既定値はOFFとする。推定値はOFFでも表示し、検証可能にする。

## 6. 制約と次段階

- Pattern Searchが成立しないcaptureではpattern-derived CFO補正を生成しない。
- 短いpattern、同一symbolの連続、低SNR、TX filter不一致ではCFOとDeviationの分離が不安定になる。
- FSKのfractional timing、measurement/reference filterの対称処理、offset探索costに基づくconfidenceは実装済み。Measurement Filterの手動Type/BT設定と設定可能なEstimation Rangeは未実装。R&S仕様ではFSKにsymbol-rate error補償とequalizerは存在しないため、これらをPSKから機械的に移植しない。
- 複数packet時は現在選択されたResult RangeのCFO modelだけを表示へ適用する。
- CFO/drift estimatorの不確かさ、timing confidence、残留CFOをResult Summaryへ追加する必要がある。
