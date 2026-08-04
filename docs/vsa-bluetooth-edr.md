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

現行の汎用VSA Pattern Searchで、次のSync phase indexをDecimal指定すると両fixtureとも相関98%以上、0 symbol errorで検出できる。

- 2-DH1: `1 2 1 2 1 2 2 1 1 1`
- 3-DH1: `3 5 3 5 3 5 5 3 3 3`

Signal DescriptionはSymbol Rate 1 MSym/s、Transmit Filter `Root Raised Cosine`、Alpha 0.4、Result Length 244とする。検出されるPattern Startはsample 34112（2.132 ms）で、EDR reference symbolに続く最初のdifferential symbolを指す。保存fixtureのCFO推定は2-DH1で約+19.5 kHz、3-DH1で約+18.4 kHz。

変調mapping、SRRC、Guard、Sync、TrailerはBluetooth Core SpecificationのBaseband Specification 6.6およびRadio Physical Layer Specification 3.2に従う。合成波形はreceiver開発用であり、Bluetooth RF-PHY conformance test sourceを称するものではない。

### Constellation表示修正（2026-08-04）

初回実装ではPSK Pattern Searchが成立していても、UIはcapture全体を固定timingでsampleした`VSAAnalysisResult.measured_symbols`を描画していた。このため2 msの無信号、GFSK Access/Header、Guardがconstellationへ混入し、無信号を含むRMS normalizationによって有効信号が約5～10倍へ拡大されていた。原点・I軸付近の点と大きな45度方向の広がりは生成IQのphase noiseではなく、この表示経路の誤りだった。

修正後はPattern Search成立時に`PatternSearchResult.measured_symbols`だけを描画する。TX filterがRoot Raised Cosineの場合は、8 samples/symbolへresampleした後に同じroll-offのSRRC matched filterを適用し、Result RangeだけでRMS振幅を1へnormalizeする。また`Compensate for Carrier Frequency Drift`がOFFでもPSK symbol補正へdriftを適用していた漏れを修正した。

修正後の合成fixture結果:

- 2-DH1: Sync correlation 0.99998、244 symbol error 0、median magnitude 1.000、簡易EVM約0.80%。
- 3-DH1: Sync correlation 0.99997、244 symbol error 0、median magnitude 1.000、簡易EVM約0.82%。

Analysis Bandwidth 1.5 MHzを有効にした2-DH1でも244 symbol error 0、magnitude範囲は概ね0.91～1.06となる。これらのEVM値は同期・表示経路の回帰確認用であり、規格適合値ではない。

さらに、`PlotWidget.clear()`が直前のFSKまたは旧constellationのViewBox rangeを保持するため、unit circle上の正常な点が表示範囲外になる問題を修正した。Constellation更新時はI/Q軸のSI prefixを無効化し、両軸を±1.25へ明示的にresetする。aspect ratio 1:1は維持する。

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
