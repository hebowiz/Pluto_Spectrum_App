# RBW演算監査と改善方針

最終更新: 2026-08-29

## 実装状況（2026-08-02）

| 経路 | 現在のRBW方式 |
|---|---|
| Sweep SA | Gaussian FIR stateful complex IQ filterへ移行済み |
| HighSpeed TA | Gaussian FIR stateful complex IQ filterへ移行済み |
| 旧Time Analyzer | 同じIQ filterへ移行済み（UI非表示） |
| RealTime SA | 共通Gaussian FIR係数によるFFT filter bank、連続cursor、既定80% overlap、時間方向Detectorへ移行済み |
| WideBand RT SA | 共通Gaussian FIR係数によるFFT filter bank。chunk取得と非overlap処理は現状維持 |
| Calibration | RealTime SAと同じGaussian FFT filter bankへ移行済み |

実装は`pluto_sa/signal/measurement_filter.py`へ集約しました。指定RBWはcomplex basebandの負周波数側から正周波数側までを含む「両側3 dB bandwidth」と定義し、内部low-pass cutoffは`RBW / 2`です。

デフォルトshapeは掃引型SAに近い裾の滑らかな選択度を優先し、linear-phase Gaussian FIRとしました。Gaussian impulseの標準偏差を`√ln(2) × Fs / (π × RBW)`とし、±4σで打ち切ってDC gainを1へ正規化します。この定義ではCWが中心から±RBW/2で約-3.0103 dB、ENBWが約`1.0645 × RBW`です。tap数、群遅延`(tap数 - 1) / 2`、有限インパルス応答全長をmetadataとして保持します。通常RBWの短いFIRはdirect filter、狭RBWで256 tapsを超える場合はFFT convolutionを使い、いずれもblock境界stateを維持します。

従来の4次Butterworth IIRは比較・将来のchannel filter候補として`shape="butterworth"`を明示した場合のみ利用できます。UIのfilter shape選択はまだなく、Sweep SA/TAはGaussian固定です。Gaussianは一般的なSAらしい公称shapeを意図しますが、特定メーカーの実機RBW filterを完全再現するものではありません。

RealTime SA/WideBand RT SA/Calibrationは`pluto_sa/signal/fft_filterbank.py`で、同じFIR係数をFFT長へ中央配置してゼロ埋めした解析窓を生成します。窓をIQ frameへ乗算してFFTすることで、各binは中心周波数だけが異なる同一Gaussian複素filterの出力になります。tone coherent gainを補正し、電力化後のGaussian convolutionは行いません。

狭いRBWではfilter supportが収まる最小の2のべき乗へFFT Sizeを自動拡張します。上限16384でも不足する場合は収まるRBWへ制限し、requested/effective RBW、ENBW、support samples、制限状態をmetadataと画面へ出します。通常RTSAは連続sampleをhopごとに解析するoverlap STFTへ移行済みです。WB RTSAとCalibrationはこの変更の対象外です。

以下は移行理由を残すための「旧実装」監査記録です。現行の測定経路では使いません。

## 旧実装

移行前のPluto Spectrum Appは、IQへ時間領域filterを適用せず、基本的に次の演算でした。

```text
IQ block
  → optional block平均DC除去
  → Hann window
  → FFT
  → coherent gain補正
  → |FFT|²（周波数binごとのpower）
  → 周波数bin方向に非正規化Gaussian kernelをconvolution
  → detector / log / calibration
```

- `make_gaussian_rbw_kernel()`は指定RBWをpower-domain GaussianのFWHMとして`σ = RBW / 2.355`へ変換する。
- kernel総和は1へ正規化しない。RBW内のbin powerを加算するenergy integrationとして振る舞う。
- `apply_rbw_weighting()`は`np.convolve(power_spectrum, kernel, mode="same")`であり、complex spectrumまたはIQをfilterしていない。
- Sweep/TA detectorは時間blockをoverlap segmentへ分け、segmentごとに同じFFT→power→Gaussian積算を行ったcenter bin系列へSample/Peak/RMS detectorを適用する。
- Power TriggerだけはRBW経路を通らず、raw complex IQ magnitudeをsample単位で評価する。

## 実RBW filterとの相違

現在のpower-domain convolutionは、定常noiseの平均powerや単一toneの帯域積算にはfilter-bank的な近似として利用できます。しかし、次の点で「IQへmeasurement filterを掛けてから電力化」と一般には等価ではありません。

1. `|X(f)|²`を作った時点でbin間の位相関係を失う。
2. 複数tone、変調波、burstの過渡応答にcomplex filterのcross termとimpulse responseが現れない。
3. blockごとにwindow/FFTをやり直すため、RBW filter stateが時間record間で連続しない。
4. kernelは非正規化で、Gaussian FWHM、ENBW、noise power校正の関係を明示していない。
5. Hannのcoherent gainは補正するが、noise ENBWを含む測定量の定義は別途必要。

FFT analyzerでFFTを使うこと自体は実機RTSAとの差ではありません。適切なwindow、FFT長、overlapによるSTFTはfilter bankと等価に解釈できます。相違点は、現在の実装がFFT filter bankのbin出力そのものではなく、power化した後から任意幅のGaussian積算を追加してRBWを表現していることです。

## モード別の改善方針

### Zero Span / HighSpeed TA

測定器に近い経路は次です。

```text
continuous IQ
  → complex digital measurement filter（既知の振幅応答・ENBW）
  → group delay / settlingを考慮
  → I² + Q²
  → Sample / Peak / RMS detector
  → display time bucket
```

Plutoの`rx_rf_bandwidth`はalias防止と粗いacquisition bandwidthに使用し、測定RBWはhost側のstateful FIR/IIR filterで定義します。filter stateはIQ block境界をまたいで保持します。

### Sweep SA

各LO pointでcentered IQへ同じmeasurement filterを適用し、settling後の包絡線をdetectorへ渡す方式が従来Zero Span/Swept SAに近い実装です。LO retuneごとにfilter stateをresetし、RF/LO settleとは別にdigital filterのgroup delayとsettling samplesを破棄します。

### RealTime SA / WideBand RT SA

周波数軸全体を同時表示するため、単一のlow-pass IQ filterへ置換しません。Gaussian FFT filter bankを採用し、通常RTSAはoverlap STFTへ拡張済みです。WB RTSAは現状維持です。将来の選択肢は次のとおりです。

- overlap STFTのwindow shape・FFT長・ENBWで定義する。
- 必要な場合はpolyphase filter bankを使用する。

power-domain smoothingは表示平均またはchannel-power integrationとしてRBWと別名称へ分離します。

#### 実機RTSAで公開されているRBWモデル

TektronixのRTSA資料では、連続digitizeしたIQ streamを選択RBWに応じた長さのtime recordへ分割し、各recordへwindow付きDFTを連続実行します。このDFTは入力をFFT bin中心周波数に並ぶband-pass filter bankへ通し、各filter出力のmagnitude/phaseをsampleする処理と数学的に等価です。

```text
continuous IQ at Fs
  → L samplesのanalysis record（RBWで決定）
  → window w[n]（filter shapeを決定）
  → optional zero padding to Nfft（trace gridを決定）
  → FFT = parallel complex filter-bank outputs
  → |X[k]|² / detector / density / FMT
  → H samples進めて次record（overlap = 1 - H/L）
```

- RBWは概ね`window bandwidth coefficient × Fs / L`で決まり、bin spacingだけではなくwindowの3 dB bandwidthまたはENBW定義を伴う。
- trace points要求で`Nfft`を大きくしても、zero paddingは表示frequency gridを細かくするだけで、実分解能はrecord長`L`とwindowで決まる。
- 狭いRBWは長いrecord/time constantを必要とし、transform rateと最短eventのfull-amplitude測定能力を下げる。
- overlapはwindow端でeventが減衰または見逃される問題を抑える。50%固定ではなく、window、必要POI、full-amplitude条件からhopを決める。
- DPX/densityは各FFT traceをfrequency-amplitude cellへ蓄積し、Frequency Mask Triggerは表示とは独立にoverlap FFTを全frame評価する。

KeysightもRTSAのPOIへsampling bandwidth、連続処理、FFT overlapが影響すると説明しています。NI RFmxはGaussian/Flat RBWをデジタルでemulateでき、速度面ではFFT-based RBWを推奨しています。このため商用機の内部実装は機種ごとに、windowed FFT、zero padding、digital RBW emulation、filter bankを組み合わせると考えるべきで、単一アルゴリズムへ一般化しません。

#### Pluto向けRTSA実装

第一候補のFFT-based RBWを実装しました。通常RTSAでは1～5と旧power convolution除去が完了しています。

1. `analysis_record_samples L`をrequested RBWとwindow ENBW/3 dB係数から決める。
2. trace point数に必要な`Nfft >= L`を独立に決め、必要分をzero padする。
3. Hann等のwindowを掛け、complex FFT後にtone coherent gainとnoise ENBWを別々に補正する。
4. 既定80% overlapと処理量上限からhopを決め、連続IQをframe化する。処理不能設定は強制変更せずoverlap低下または解析gapを警告する。
5. Sample／Peak／Negative Peak／Average／RMS detectorを同一frequency binの時間frame方向へ適用する。
6. 現行Gaussian power convolutionはRBWから外し、必要なら`Frequency Smoothing`または`Channel Integration BW`として別設定にする。

第二候補のpolyphase filter bankはfilter shape、channel isolation、decimationを明示しやすい一方、実装と計算量が大きいためFMT/POI要件でFFT方式が不足した場合に評価します。

参考:

- [Tektronix: Fundamentals of Real-Time Spectrum Analysis](https://download.tek.com/document/37W_17249_6_Fundamentals_of_RealTime_Spectrum_Analysis_0.pdf)
- [Tektronix RSA5100B Help](https://download.tek.com/manual/RSA5100B-Real-Time-Spectrum-Analyzer-Help_EN-US_077-0899-07_077089907.pdf)
- [Tektronix: Understanding FFT Overlap Processing](https://download.tek.com/document/37W_18839_1.pdf)
- [Keysight: X-Series RTSA Technical Overview](https://www.keysight.com/us/en/assets/7018-03791/technical-overviews/5991-1748.pdf)
- [NI RFmx SpecAn Spectrum / Zero Span](https://www.ni.com/docs/en-US/bundle/rfmx-specan/page/spectrum.html/)

### VSA

raw trigger recordを正本として保持し、解析段でchannel selection filter、resampling、measurement filter、carrier/symbol synchronizationを適用します。RBWはSpectrum traceのwindow/record長またはdemodulation measurement filterとして扱い、Zero Span RBWと混同しません。

## 実装時の必須事項

- filterの振幅応答、-3 dB/FWHM、shape factor、ENBWを区別して表示する。
- DC gainを正規化したFIRなら、ENBWを概ね`Fs × Σ|h[n]|² / |Σh[n]|²`で算出しmetadataへ保存する。
- filter group delay、startup/retune settling、block境界stateをテストする。
- CW振幅、二信号分離、white-noise power、burst rise/fall、block境界連続性をpytestと実機で検証する。
- FFT経路とIQ-filter経路で既存calibration offsetを無条件に共用しない。

## 推奨する次の段階

1. 既知CW/noise/burstを実機入力し、3 dB bandwidth、ENBW、rise/fall time、校正差を検証する。
2. TA UIの`RBW`と`RF BW`を`Measurement BW`と`Acquisition BW`へ整理する。
3. 実装済みoverlap/hop処理について、既知burstで時間被覆率、POI、各Detectorと商用RTSAの差を実機検証する。
