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

