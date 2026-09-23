# Bluetooth EDR実信号デバッグ計画

最終更新: 2026-08-04

## 目的

固定channelの実EDR packetをPlutoで保存し、Bluetooth専用decoderへ閉じない汎用VSA処理としてPSK区間のpattern search、symbol復調、constellation、carrier補正を検証する。

複数packetのResult Range分割やfrequency hopping追従はこの段階では扱わない。1 captureから人が選んだ1 packetを解析できればよい。

## 最初の対象

最初は2-DH1相当のpi/4-DQPSKを対象とする。これが成立した後に3-DH1相当の8DPSKへ進む。payloadはPRBS-9を推奨するが、先頭の既知patternと一部symbol列が確認できれば、Whitening、HEC、CRCの条件が不明でも初期VSA検証は可能である。

想定する実験条件:

- RF channel: 2441 MHz固定。
- Source sample rate / RF bandwidth: 16 MS/s / 16 MHz。
- Analysis Center: 2441 MHz。
- Analysis Bandwidth: 初期値3 MHz。
- Capture time: 3 ms。
- Pluto RX gain: 0 dB。
- 外部ATT: 30 dB。
- BD_ADDR: `00006BC6967E`、LAP: `0xC6967E`。

## 合成IQ fixture（2026-08-04）

実送信commandが利用できるまでの開発用として、Bluetooth Core仕様に基づく最大長2-DH1/3-DH1 IQを追加した。

| file | modulation | payload body | PSK symbols |
|---|---|---:|---:|
| `tests/fixtures/bluetooth_2dh1_prbs9_16msps.npz` | pi/4-DQPSK | 54 bytes (`0x36`) | 245 |
| `tests/fixtures/bluetooth_3dh1_prbs9_16msps.npz` | 8DPSK | 83 bytes (`0x53`) | 245 |

共通条件:

- 16 MS/s、center 2441 MHz、capture 3 ms、packet start 2.000 ms。
- Access Code 72 symbolとHeader 54 air bitはBT=0.5 GFSK、1 Msym/s。
- Header後に5 us Guard、11-symbol EDR Sync、Payload Header、PRBS-9 body、CRC、2-symbol Trailerを配置。
- EDR symbol rateは1 MSym/s。TX filterはSRRC、roll-off 0.4。
- CFOは+20 kHz、SNRは35 dB、振幅は校正済み合成基準として記録。
- UAP `0x6B`、CLK_6-1 `0x2B`。Header/Payload whiteningとHEC/CRCを適用。

NPZにはIQだけでなく、`access_bits`、`header_air_bits`、`sync_bits`、payload各field、`differential_phase_indices`、各segmentのsample indexを格納する。再生成は次で行う。

```powershell
python -m tools.generate_bluetooth_edr_iq
```

現行の汎用VSA Pattern Searchで、次のBluetooth EDR論理symbolをDecimal指定すると両fixtureとも相関98%以上、0 symbol errorで検出できる。

- 2-DH1（R&S LSB表示）: `2 3 2 3 2 3 3 2 2 2`
- 3-DH1: `2 7 2 7 2 7 7 2 2 2`

Signal DescriptionはSymbol Rate 1 MSym/s、Transmit Filter `Root Raised Cosine`、Alpha 0.4、Result Length 244とする。検出されるPattern Startはsample 34112（2.132 ms）で、EDR reference symbolに続く最初のdifferential symbolを指す。保存fixtureのCFO推定は2-DH1で約+19.5 kHz、3-DH1で約+18.4 kHz。

変調mapping、SRRC、Guard、Sync、TrailerはBluetooth Core SpecificationのBaseband Specification 6.6およびRadio Physical Layer Specification 3.2に従う。合成波形はreceiver開発用であり、Bluetooth RF-PHY conformance test sourceを称するものではない。

### PSK Symbol Mapping（2026-08-19）

`Modulation Mapping`は`Natural`、R&S汎用の`Gray`、Bluetooth規格専用の
`Bluetooth EDR`を区別する。汎用8DPSK GrayとBluetooth EDR 8DPSKは一部の
bit-to-phase割当が異なるため、EDRを単なるGrayとして扱わない。

Bluetooth EDRのOTA bit（LSB first）と差動位相は次のとおり。内部DSPはMSB番号のcanonical symbolを維持し、R&S互換のLSB表示境界でsymbol内のbit順を反転する。

- pi/4-DQPSK: `00, 01, 11, 10` -> `+pi/4, +3pi/4, -3pi/4, -pi/4`
- 8DPSK: `000, 001, 011, 010, 110, 111, 101, 100` ->
  `0, +pi/4, +pi/2, +3pi/4, pi, -3pi/4, -pi/2, -pi/4`

Symbol Table、pattern file、symbol exportは物理位相Indexではなく論理symbol値を
保持する。既存のEDR sync patternはこの契約へ移行した。

- 2DH系（R&S LSB表示）: `2 3 2 3 2 3 3 2 2 2`
- 3DH系: `2 7 2 7 2 7 7 2 2 2`

生成IQ fixture内の`differential_phase_indices`は波形生成の再現性確認用として
物理位相Indexのまま保持し、`BluetoothEDRWaveform.logical_symbols`を解析期待値に
使用する。

### Constellation表示修正（2026-08-04）

初回実装ではPSK Pattern Searchが成立していても、UIはcapture全体を固定timingでsampleした`VSAAnalysisResult.measured_symbols`を描画していた。このため2 msの無信号、GFSK Access/Header、Guardがconstellationへ混入し、無信号を含むRMS normalizationによって有効信号が約5～10倍へ拡大されていた。原点・I軸付近の点と大きな45度方向の広がりは生成IQのphase noiseではなく、この表示経路の誤りだった。

修正後はPattern Search成立時に`PatternSearchResult.measured_symbols`だけを描画する。TX filterがRoot Raised Cosineの場合は、8 samples/symbolへresampleした後に同じroll-offのSRRC matched filterを適用し、Result RangeだけでRMS振幅を1へnormalizeする。また`Compensate for Carrier Frequency Drift`がOFFでもPSK symbol補正へdriftを適用していた漏れを修正した。

修正後の合成fixture結果:

- 2-DH1: Sync correlation 0.99998、244 symbol error 0、median magnitude 1.000、簡易EVM約0.80%。
- 3-DH1: Sync correlation 0.99997、244 symbol error 0、median magnitude 1.000、簡易EVM約0.82%。

Analysis Bandwidth 1.5 MHzを有効にした2-DH1でも244 symbol error 0、magnitude範囲は概ね0.91～1.06となる。これらのEVM値は同期・表示経路の回帰確認用であり、規格適合値ではない。

さらに、`PlotWidget.clear()`が直前のFSKまたは旧constellationのViewBox rangeを保持するため、unit circle上の正常な点が表示範囲外になる問題を修正した。Constellation更新時はI/Q軸のSI prefixを無効化し、両軸を±1.25へ明示的にresetする。aspect ratio 1:1は維持する。

R&S FPL1-K70 VSA User Manual rev.12 pp.86-88では、differential PSKの物理constellationをISI-free demodulation後のdecision pointとして表示し、pi/4-DQPSKはpi/4 rotation compensatedで表示する。これに合わせ、QPSK、OQPSK、pi/4-DQPSKのConstellation表示vectorだけを-45 degree回転し、4点を+I、+Q、-I、-Q軸上へ置く。内部のcarrier correction、symbol decision、mapping、Symbol Tableは回転前の論理vectorを使い続ける。8DPSKは回転せず、I/Q軸を含む45 degree間隔の8点表示とする。

## 解析順序

1. BR/GFSKのAccess Codeを既存復調器で検出し、packet時刻、CFO、symbol timingを得る。
2. BR Headerからpacket typeを確認し、EDR modulation区間の概略開始位置を決める。
3. EDR同期区間を用いてPSKの位相、CFO、symbol timing、symbol mappingを推定する。
4. pi/4-DQPSKまたは8DPSKの差動symbolを復元する。
5. IQ PowerとPSK表示へ、Access Code、EDR同期区間、解析Result Rangeを重ねる。
6. 既知payload patternに対してsymbol errorを確認する。

BR側のpacket検出は時刻アンカーとして用いるが、PSK解析器そのものは任意のKnown PatternとSignal Descriptionで動作する設計を維持する。

## 最初のcapture手順

送信開始後、既存のcapture toolでAccess Code一致を条件にwideband IQを保存する。

```powershell
python -m tools.capture_bluetooth_br_iq `
  --center-frequency 2441000000 `
  --sample-rate 16000000 `
  --rf-bandwidth 16000000 `
  --analysis-center-frequency 2441000000 `
  --analysis-bandwidth 3000000 `
  --duration-ms 3 `
  --gain 0 `
  --attempts 100 `
  --lap 0xC6967E `
  --output bluetooth_edr_2dh1_pluto_16msps.npz
```

EDR payloadを現行GFSK decoderが正しく解釈できないこと自体はcapture失敗条件にしない。Access Codeの相関、開始sample、overloadの有無を先に確認し、保存IQからPSK区間を段階的に切り出す。

## 完了条件

- 実EDR capture内でBR Access Code位置が安定して求まる。
- PSK同期patternの一致位置がIQ Power上で確認できる。
- 2-DH系でpi/4-DQPSK symbol列の一部が既知patternと一致する。
- carrier-corrected constellationとsymbol tableが同じResult Rangeを表す。
- 推定条件と未確定条件をfixture sidecarおよび本書へ残す。

# 2026-08-05: live 2DH1 phase-stability diagnosis

A fixed 2441 MHz 2DH1 transmission was captured ten times with Pluto at
8 MS/s, 3 ms Capture Length, 1.5 MHz Analysis Bandwidth, pi/4-DQPSK,
1 MSym/s, and RRC alpha 0.4.  Pattern Search used the existing ten-symbol
sequence `1 2 1 2 1 2 2 1 1 1` and Result Length 244 symbols.

The existing PSK estimator fits a straight line to phase error over only those
ten known symbols.  Its reported Carrier Drift varied from approximately
-814 to +1774 kHz/ms in this run.  Enabling that correction increased the
244-symbol constellation phase RMS from 2.3...8.0 degrees to approximately
5.5...29.0 degrees in a preceding ten-capture comparison.

An independent pi/4-DQPSK fourth-power estimate was then applied to all 244
differential symbols.  Because every ideal differential symbol is at
+/-pi/4 or +/-3pi/4, `-z[k]**4` removes the transmitted data phase.  A linear
fit to its unwrapped phase measures the common physical phase slope without
knowing the payload.  Across ten captures it measured only
-0.72...+1.03 kHz/ms, with detrended residual phase RMS 2.26...6.45 degrees.

Conclusion: the Pluto/input signal has several degrees of real residual phase
variation, but the large capture-to-capture constellation change and the
hundreds-to-thousands of kHz/ms Drift values are primarily artifacts of the
short-pattern estimator.  PSK drift correction should remain off until the
estimator uses the complete Result Range.  For pi/4-DQPSK/8DPSK, a modulation-
removing M-th-power coarse estimate followed by reference-directed refinement
is the preferred next implementation.  The ten known pattern symbols should
anchor phase ambiguity and symbol mapping, not independently determine a
packet-wide drift from such a short baseline.

## Result-Range PSK synchronization implementation

The estimator was subsequently changed to follow the R&S Auto synchronization
principle: PSK uses one estimation point per symbol and detected data over the
complete Result Range when the known pattern is short.  For differential M-PSK,
raising each measured differential symbol to the Mth power removes its data
phase.  A weighted phase line over all Result Range symbols supplies coarse CFO
and drift.  The ten-symbol pattern resolves the remaining `2*pi/M` ambiguity.
Two detected-data reference iterations then minimize phase error over the full
Result Range.  The pattern remains responsible for search, timing, and absolute
symbol mapping, but no longer determines packet-wide drift by itself.

A generated pi/4-DQPSK waveform with a ten-symbol pattern, 244-symbol Result
Range, and known +/-150 kHz/ms drift recovered the drift to within 1 Hz/ms and
decoded all 244 symbols without error.

Ten new live 2DH1 captures produced:

- CFO: +21.10...+21.37 kHz
- Carrier Drift: -1.72...+0.80 kHz/ms
- Result-Range phase RMS: 2.32...4.86 degrees
- I/Q correlation: 99.43...99.82 %

Enabling drift compensation no longer increases phase RMS; ON and OFF are
numerically almost identical for this source because its measured drift is
small.  The remaining 2...5 degree spread is therefore attributable to the
received/transmitted waveform, noise, filtering, and timing residual rather
than the former short-pattern drift artifact.

## Robust drift acceptance and fractional timing (2026-08-05)

Further 8DPSK live testing exposed rare false estimates despite successful
pattern decoding. One capture had 99.24% pattern correlation and zero known
symbol errors but reported -736.5 kHz/ms instead of the normal few kHz/ms;
another UI capture reported -1537.832 kHz/ms. Changing external attenuation or
Pluto gain cannot explain a carrier-model failure with a correct pattern.

Differential QPSK/8PSK synchronization now:

1. estimates adjacent Mth-power phase on the unit circle with amplitude and
   Tukey robust weighting, without a global phase unwrap;
2. uses detected data to refine carrier-phase intercept but does not allow the
   decision-directed stage to pull drift slope into a periodic alias;
3. compares the drift candidate with a separately fitted CFO-only model and
   accepts drift only when it reduces weighted Result Range phase error;
4. otherwise reports zero drift and uses the robust CFO-only solution;
5. searches +/-0.5 analysis samples around coarse 8-sample/symbol timing and
   selects the fractional timing with minimum Result Range phase error while
   retaining the pattern correlation threshold.

This applies the R&S principle of minimizing symbol error over the estimation
range while explicitly guarding against differential-PSK periodic aliases.
Fractional timing is necessary because Pluto and the transmitter do not start
each capture at the same sub-sample symbol-clock phase.

## Joint complex-EVM synchronization (supersedes the fallback above)

The CFO-only fallback proved too conservative in the GUI: real small drift was
also rejected, so Carrier Drift remained exactly zero. The fixed fractional
timing search also minimized phase-only error and did not fully stabilize
symbol amplitude. It has therefore been replaced, not retained as the final
PSK synchronization design.

The differential-PSK fine synchronizer now jointly optimizes four parameters
over the complete Result Range:

- fractional symbol timing offset;
- symbol timing rate (sample-clock/symbol-rate error);
- carrier phase increment, reported as CFO;
- linear change of carrier phase increment, reported as Carrier Drift.

Its residual is the real and imaginary complex error vector between normalized
measured symbols and the known/detected reference symbols. Robust `soft_l1`
least squares minimizes this complex EVM. Pattern symbols are fixed references;
the remaining Result Range references are updated decision-directed over three
iterations. The Mth-power estimator supplies only the bounded initial carrier
solution. Timing offset is allowed to move within one symbol eye, and timing
rate can move by one analysis sample across the Result Range. Carrier-drift
slope is bounded to the unambiguous decision region to prevent periodic false
solutions without forcing valid estimates to zero.

`fractional_timing_offset_samples`, `symbol_rate_error_ppm`, and
`synchronization_evm_rms` are retained in Pattern Result metadata. Symbol Rate
Error and Sync EVM RMS are also shown in Result Summary for PSK.

## Absolute IQ trajectory spread investigation (2026-08-05)

Repeated live 2DH1 and 3DH1 measurements can show either tight or dispersed
symbol points in IQ Trajectory while the differential constellation remains
tight. This is not a separate UI resampling path error. In twelve 3DH1 captures,
symbol phase spread computed directly from the demodulator's optimized timing
points and spread reproduced through the UI carrier-corrected trajectory path
agreed within 0.14 degree in every capture.

For ten additional 3DH1 captures, differential phase error was stable at about
2.8...3.4 degrees RMS, but absolute symbol phase spread varied from 3.35 to
10.82 degrees. Reconstructing absolute phase error by cumulatively summing the
differential-symbol phase errors matched the measured IQ trajectory with
0.000-degree RMS mismatch in every capture. Thus small differential phase errors
form a capture-dependent random walk in absolute IQ phase. A tight differential
constellation and a dispersed absolute trajectory are mathematically compatible.

The R&S FPL1-K70 manual, sections 4.4.5 and 4.5.1.2, describes a more complete
fine-synchronization stage: detected/known symbols and the configured transmit
filter generate an ideal continuous reference signal, then measurement and
reference waveforms are correlated by minimizing mean-square error-vector
magnitude. The current Pluto VSA joint synchronizer minimizes complex EVM at
differential decision symbols, not against a reconstructed absolute reference
waveform. This is the remaining architectural difference.

The next correction should therefore reconstruct the ideal absolute PSK waveform
from decoded differential symbols and TX-filter settings, then jointly fit timing,
symbol-rate error, carrier phase, CFO, and drift against measurement samples.
Residual trajectory spread after that fit should remain visible as modulation
error/phase noise; symbol points must not be forced onto ideal decisions merely
to make the plot look tighter.

## Absolute-reference waveform fine synchronization implementation

The correction proposed above is now implemented for differential PSK. Detected
differential symbols are accumulated into an ideal absolute symbol sequence. The
configured matched-filter output is sampled at continuously adjustable symbol
times and compared with that absolute reference. A robust complex least-squares
fit jointly estimates:

- absolute carrier phase;
- carrier phase increment (CFO);
- quadratic absolute phase term (linear Carrier Drift);
- fractional timing offset;
- timing-rate/Symbol Rate Error.

Known pattern symbols replace detected decisions in the pattern interval. The
remaining Result Range is updated decision-directed for three iterations. Both a
zero-drift start and the modulation-removed coarse-drift start are optimized; the
solution with lower absolute-reference EVM is selected, which avoids retaining a
periodic Mth-power local solution. Carrier phase is stored in `phase_rotation_rad`
and is applied to carrier-corrected IQ, so IQ Trajectory and demodulation use the
same fine-synchronization solution.

No per-symbol decision forcing is applied to the displayed measurement waveform.
Residual symbol spread therefore remains a genuine error result rather than being
hidden by snapping points to ideal locations.
