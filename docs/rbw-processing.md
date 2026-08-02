# RBW演算監査と改善方針

最終更新: 2026-08-02

## 現行実装

Pluto Spectrum AppのRBWは、IQへ時間領域filterを適用する実装ではありません。RealTime SA、WideBand RT SA、Sweep SA、Time Analyzer、HighSpeed TAは基本的に次の演算です。

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

周波数軸全体を同時表示するため、単一のlow-pass IQ filterへ置換しません。RBWは次のどちらかで定義します。

- overlap STFTのwindow shape・FFT長・ENBWで定義する。
- 必要な場合はpolyphase filter bankを使用する。

power-domain smoothingは表示平均またはchannel-power integrationとしてRBWと別名称へ分離します。

### VSA

raw trigger recordを正本として保持し、解析段でchannel selection filter、resampling、measurement filter、carrier/symbol synchronizationを適用します。RBWはSpectrum traceのwindow/record長またはdemodulation measurement filterとして扱い、Zero Span RBWと混同しません。

## 実装時の必須事項

- filterの振幅応答、-3 dB/FWHM、shape factor、ENBWを区別して表示する。
- DC gainを正規化したFIRなら、ENBWを概ね`Fs × Σ|h[n]|² / |Σh[n]|²`で算出しmetadataへ保存する。
- filter group delay、startup/retune settling、block境界stateをテストする。
- CW振幅、二信号分離、white-noise power、burst rise/fall、block境界連続性をpytestと実機で検証する。
- FFT経路とIQ-filter経路で既存calibration offsetを無条件に共用しない。

## 推奨する次の段階

1. HighSpeed TAへ`FFT Power`と`Filtered Envelope`の処理方式を選択できる内部APIを追加する。
2. stateful complex FIRとENBW metadataを追加する。
3. 表示は全sample描画ではなく、時間pixel bucketごとのPeak/RMSを選べるようにする。
4. 既知CW/noise/burstで旧FFT方式と比較する。
5. 結果確定後、TA UIの`RBW`と`RF BW`を`Measurement BW`と`Acquisition BW`へ整理する。
