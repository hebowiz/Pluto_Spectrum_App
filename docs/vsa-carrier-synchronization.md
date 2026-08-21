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

UIの`Display Config > Modulation Signal`で次を切り替える。

- `Raw IQ`: carrier補正もMeasurement Filterも適用せず、取得IQからModulation表示を生成。
- `Measured`: sample単位でCFOを除去し、その後にmodulationごとのMeasurement Filterを適用してModulation表示を生成。既定値。

Powerは位相回転で変化しないためRaw IQを使用する。Spectrumもcarrier未補正IQへ固定する。CFO、推定carrier frequency、driftはResult Summaryへ表示する。FSKでは`Carrier Drift`がjoint fitの推定候補値、`Applied Drift`が品質gate通過後に補正へ使用する値である。

## 5. Drift補正の扱い

`Demodulation > Compensate for > Carrier Frequency Drift`でsample単位補正へlinear driftを含める。CFO補正はModulation SignalのMeasured表示で常に適用する。

現行drift推定はdecision-directedな一次modelであり、測定確度のcross-validationが未完了である。2026-08-04の固定BR fixture（Analysis BW 2 MHz）では、known pattern単独のCFOは約+20.0 kHzだった一方、旧実装のpacket-wide fitはCFOを-5.37 kHzへ移動させ、Carrier Corrected瞬時周波数に約25 kHzの中心ずれを残していた。現在はCFOをknown patternへ固定し、packet-wide fitからはcoarse driftだけを採用する。長いResult Rangeへのdrift外挿はなお過補正の可能性があるため、drift補正の既定値はOFFとする。推定値はOFFでも表示し、検証可能にする。

## 6. 制約と次段階

- Pattern Searchが成立しないPSK captureでは、Auto設定時にDetected Data同期から
  decision-directed CFO補正を生成する。FSKはまだknown patternを必要とする。
- 短いpattern、同一symbolの連続、低SNR、TX filter不一致ではCFOとDeviationの分離が不安定になる。
- FSKのfractional timing、measurement/reference filterの対称処理、offset探索costに基づくconfidenceは実装済み。Measurement Filterの手動Type/BT設定と設定可能なEstimation Rangeは未実装。R&S仕様ではFSKにsymbol-rate error補償とequalizerは存在しないため、これらをPSKから機械的に移植しない。
- 複数packet時は現在選択されたResult RangeのCFO modelだけを表示へ適用する。
- CFO/drift estimatorの不確かさ、timing confidence、残留CFOをResult Summaryへ追加する必要がある。

## 7. PSK Detected Data synchronization (2026-08-21)

PSK波形ではPattern Searchの有効/無効とDetected Data同期を独立させる。
Pattern Searchが有効で既知パターン検索に失敗した場合、
`Coarse Synchronization = Auto`ではDetected Data同期へフォールバックする。
Pattern Searchが無効でも`Auto`または`Detected Data`なら、最初から既知パターンを
使用せずDetected Data同期を実行する。`Pattern`を選ぶと既知パターン以外へ
フォールバックしない。FSKは今回の対象外で、従来経路を維持する。

Result Range、Demodulation、I/Q Power TriggerはPattern Searchを無効にしても保持する。
したがってPower Triggerで候補burstを絞ったうえで、既知symbolなしのPSK同期を行える。

処理順は次のとおり。

1. I/Q Power Triggerが有効なら、最初のtrigger active intervalだけを候補区間にする。
2. 8 samples/symbolへresampleし、Measurement FilterがAutoならSignal Descriptionの
   SRRC（既定roll-off 0.4）を適用する。
3. 各timing候補で差動判定点をM乗し、data phaseを除去した円周上のconcentrationを
   32-symbol窓で測る。Power Trigger区間内にFSKなどが混在する場合は、concentrationが
   継続して高い最長区間だけをPSK segmentとして選ぶ。十分な区間がなければcapture
   全体へ戻し、低品質なPSKを一律に切り捨てない。
4. 1 symbol内の8 timing phaseを走査し、選択PSK segmentにおける80% trimmed decision
   errorが最小となる位相を選ぶ。その近傍はfractional sampleで再探索する。
5. Physical IQの回転対称性からdecision-independent carrier stepを推定し、CFOへ換算する。
6. Result Rangeは検出PSK segmentと設定Result Lengthの短い方に制限する。

2026-08-21の追加修正で、blind carrier/timing同期は仮定した差動decision alphabetから
分離した。π/4-DQPSKは差動symbolが4種類でも、絶対IQは2つのQPSK集合を交互に使うため
8-fold symmetryを持つ。8DPSKも同じ8-fold symmetryである。このため両仮定では絶対IQを
8乗してdata phaseを除き、その位相直線からtiming、constant phase、CFOを推定する。
その後にだけ仮定modulationの差動alphabetでsymbol decisionとEVMを計算する。

この分離により、実信号が8DPSKなのにπ/4-DQPSKを仮定した場合でもPhysical IQの8点は
収束し、同じPSK segmentとCFOを得る。一方、π/4-DQPSKの4 decisionに存在しない8DPSK
位相差があるためDifferential Symbol EVMは大きくなり、変調仮定の不一致を判別できる。
同期が崩れることと、仮定modulationの復調品質が悪いことを混同しない設計である。

差動積`d[k] = r[k] conj(r[k-1])`の振幅は`|r[k]||r[k-1]|`となる。AM rippleや
burst rampをtiming評価へ混入させないため、Detected Dataのtiming costと差動Symbol
Plotは位相単位円へ正規化する。既知データがないため絶対PSK位相にはM-fold ambiguity
が残る。この経路ではPattern Symbols Correctを`No`、I/Q Correlationを未定義として
表示し、blind resultをpattern matchとして扱わない。絶対IQ EVMも誤解を避けて報告せず、
decision-directed differential EVMを使用する。

回帰fixture `tests/fixtures/bt_mHDT4_capture.iq.tar`（R&S取得、8DPSK 2 Msym/s）は、
Power Trigger後の先頭約246 symbolsがFSKで、その後に約448 symbolsの8DPSKが続く。
segment分離前はFSKにCFO推定を引かれて+16.4 kHzだったが、分離後は+9.23 kHzとなり、
比較画像のR&S結果+8.821 kHzに近づく。選択PSK区間のdecision-directed differential
EVMは約7.64%である。R&S画面のEVM 17.31%とは評価区間が異なるため直接比較せず、
R&S内部のequalizer・EVM定義との完全同一性も保証しない。

### mHDT4 capture内のPSK方式切替

同fixtureのDetected Data 448 symbolsを差動位相で再評価すると、単一方式ではなく
途中でπ/4-DQPSKから8DPSKへ切り替わる構造が観測された。Detected Data symbol 32～250
付近は差動位相がπ/4-DQPSKの4点（±45°、±135°）へ収束する。symbol 251
（capture時刻約267.2 us）以降では0°、±90°、180°付近の8DPSK固有点が継続して現れる。

この構造では8DPSK仮定が区間全体に対して一見良好になる。π/4-DQPSKの4位相差は
8DPSK alphabetの部分集合なので、8DPSK decoderは前半も低いdecision errorで受理できる。
逆にπ/4-DQPSK decoderは後半の8DPSK固有位相を表現できず、fixtureではDifferential
Symbol EVMが約35.8%となる。Physical IQはπ/4-DQPSKでも8状態を取り得るため、Physical
Symbol Plotだけでは方式切替を識別できない。識別には差動位相の時間推移、窓単位EVM、
またはDifferential Symbol Plotを用いる必要がある。

将来のmixed-modulation解析では、Detected Data区間を短い窓で各candidate alphabetへ
当てはめ、方式ごとのdecision errorに持続時間とhysteresisを加えてsegment境界を決める。
各segmentは共通のcarrier/timing modelを引き継ぎつつ、個別のmapping、symbol decision、
EVMを持つ設計が適する。
