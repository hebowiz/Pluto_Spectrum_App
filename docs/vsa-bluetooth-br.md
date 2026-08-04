# Bluetooth BRパケット復調 実装・検証ノート

最終更新: 2026-08-03

## 1. 対象と参照仕様

最初の実パケット復調対象はBluetooth Basic RateのGFSKです。Enhanced Data RateでもAccess CodeとPacket HeaderはBasic Rate GFSKなので、この処理を後のEDR segment検出へ再利用します。

参照したBluetooth SIG公式仕様:

- [BR/EDR Radio Physical Layer Specification](https://www.bluetooth.com/wp-content/uploads/Files/Specification/HTML/Core_v6.3/out/en/br-edr-controller/radio-physical-layer-specification.html)
- [BR/EDR Baseband Specification](https://www.bluetooth.com/wp-content/uploads/Files/Specification/HTML/Core-61/out/en/br-edr-controller/baseband-specification.html)
- [BR/EDR Sample Data](https://www.bluetooth.com/wp-content/uploads/Files/Specification/HTML/Core-54/out/en/br-edr-controller/sample-data.html)

実装へ固定した主要条件:

- Basic Rateは1 Msym/s、GFSK、BT=0.5。
- bit 1はpositive frequency deviation、bit 0はnegative deviation。
- Access Codeは4-bit preamble、64-bit sync word、headerが続く場合は4-bit trailer。
- 64-bit sync wordは24-bit LAPから(64,30) expurgated block codeとPN overlayで生成。
- Packet Headerは18 bitをrate 1/3 FECで54 air bitsへ符号化。
- Header/PayloadはFEC前にwhiteningされ、HeaderとPayloadの間でLFSR stateを継続。
- Header fieldsはLT_ADDR 3 bit、TYPE 4 bit、FLOW、ARQN、SEQN、HEC 8 bit。

## 2. 実装済みDSP

`pluto_sa/vsa/demod/gfsk.py`:

1. 任意source sample rateから8 samples/symbolへpolyphase resampling。
2. IQ位相差からinstantaneous frequencyを計算。
3. power envelopeからburst range候補を検出。
4. 全timing phaseで既知Access Codeとのnormalized sliding correlationを実行。
5. Access Codeからsymbol timing、IQ inversion、CFO、deviation、coarse driftを推定。
6. decision-directed refinementでpacket内frequency driftを補正。
7. binary symbol/bit streamを復元。

coarse drift値は復調補正用であり、Bluetooth RF conformance measurementとして使える精度ではありません。

## 3. Bluetooth profile

`pluto_sa/vsa/profiles/bluetooth_br.py`:

- 任意24-bit LAPから68/72-bit Access Codeを生成。
- GIAC LAP `0x9E8B33`をpreset。
- Bluetooth whitening LFSR。
- Header rate 1/3 FEC majority decode。
- HEC生成/検証。
- LT_ADDR、TYPE、FLOW、ARQN、SEQN抽出。
- unknown CLK_6-1の64候補とWhiteningなしのHeader候補探索。
- DH1 1-byte Payload Header、最大27-byte body、CRC-CCITT検証。
- PRBS-9のphase、polarity、time direction、bit error探索。
- known clock/UAPを使ったHeader dewhiteningとHEC validation。
- test packet bitstream/GFSK waveform generator。

Access Code相関とraw air bit復元にはLAPだけが必要です。Header内容を得るには`CLK_6-1`が必要で、HECを正しく検証するにはUAPも必要です。

## 4. 検証済み条件

Bluetooth SIG Sample Dataとの一致:

- GIACを含む複数LAPのAccess Code。
- whitening LFSR出力。
- 複数のHeader/UAPに対するHEC。
- rate 1/3 FEC。

合成packet IQへ次の障害を加えたpytestでAccess Code、Header、Payload bitを復元しています。

- 8/16 MSPS。
- symbol境界と一致しないcapture start。
- +55 kHz CFO。
- AWGN。
- IQ conjugation/inversion。
- packet内frequency drift。

### Pluto + smartphone Inquiry実機検証（2026-08-03）

スマートフォンのBluetooth機器検索を送信源として、Pluto Rev.C
（`usb:1.60.5`）でGIAC Inquiry ID packetを捕捉しました。

- 4 MSPS、RF BW 3 MHz、center 2460 MHz、6 ms snapshot。
- 49回目のsnapshotで、約2461 MHzのGIACを検出。
- shortened GIAC 68 bitを0 bit errorで復元。
- normalized correlation: 0.9978998。
- 推定carrier位置: centerから+1.037494 MHz（約2461.037494 MHz）。
- 推定GFSK frequency deviation: 164.778 kHz。
- capture内のAccess Code開始: 2611 sample（0.652750 ms）。
- 回帰fixture: `tests/fixtures/bluetooth_giac_inquiry_pluto_4msps.npz`。

これにより、少なくとも実電波のInquiry ID packetについて、burst捕捉、GIAC相関、
symbol timing、CFO/deviation推定、68-bit同期word復元まで動作することを確認しました。
Header/Payloadを持つ接続packetの復元は未検証です。

最初に行った16 MSPS全帯域への直接相関では、相関0.888、2 bit error、偏移約
1.82 MHzという不整合な候補を検出しました。複数信号を含み得るwideband IQへ
single-channel GFSK復調器を直接適用すべきではありません。通常のVSAと同じく、
ユーザー指定のAnalysis Center/Bandwidthで対象channelをDDC/FIR抽出してから
復調する設計へ変更しました。全channelの自動channelizerは当面実装しません。

保存した実測IQをAnalysis Center 2460.750 MHz、Bandwidth 1.5 MHzで再解析し、
相関0.99645、0 bit errorで同じGIACを復元できることも確認済みです。

また、shortened Access Codeしかないpacketへdecision-directed drift refinementを
適用すると、burst paddingを未知dataと誤認して既知bitを崩す問題を実測で確認しました。
既知Access Codeをtraining symbolとして固定し、十分なpost-access fieldがある場合だけ
drift refinementするよう修正済みです。

### 固定周波数BR/PRBS-9実機検証（2026-08-03）

固定2441 MHzのBR test waveformを30 dB外部ATT経由でPlutoへ入力し、通常Access
Code、Header、Payloadを解析しました。送信出力は0～10 dBmの範囲、Pluto RX gainは
0 dBです。

- Source Fs / RF BW: 16 MSPS / 16 MHz。
- Analysis Center / Bandwidth: 2441 MHz / 1.5 MHz。
- input peak: -31.27 dBFS。overloadなし。
- LAP: `0xC6967E`、72-bit Access Code相関0.99731、0 bit error。
- CFO: -4.39 kHz、推定GFSK deviation: 147.09 kHz。
- 54 air-bit Headerは全18 FEC tripletが一致し、FEC correction 0。
- Whitening OFF候補でTYPE=4（DH1）、Payload length=27 bytes。
- Payload body 216 bitはPRBS-9 phase 0と完全一致し、0 bit error。
- 回帰fixture: `tests/fixtures/bluetooth_br_prbs9_pluto_16msps.npz`。

一方、入力されたBD_ADDR `00006BC6967E`から期待するUAP `0x6B`ではHeader HECが
一致しません。Whitening OFFのHeader bit列から逆算するとHEC初期値は`0x5D`相当です。
Payload CRCもreceived `72b4`、expected `df5d`で不一致でした。このため、このtest
waveformはWhitening/CRC/HECの一部がdisabled、またはAccess Codeとは別のcheck初期値を
使っている可能性があります。今回の結果はRF復調、Header FEC、DH1 length、PRBS-9
Payload復元の実機検証として有効ですが、標準準拠packetのHEC/CRC成功例ではありません。

### 合成DH1検証IQ（2026-08-04）

実測fixtureとは別に、全fieldの期待値が既知でHEC/CRCも有効な最大長DH1を追加しました。

- file: `tests/fixtures/bluetooth_dh1_prbs9_16msps.npz`
- Fs / center / capture: 16 MS/s / 2441 MHz / 3 ms
- packet start/stop: sample 32000 / 37856（2.000 ms / 2.366 ms）
- modulation: GFSK、1 MSym/s、BT 0.5、deviation 160 kHz
- CFO / SNR: +20 kHz / 35 dB
- LAP / UAP / CLK6-1: `0xC6967E` / `0x6B` / `0x2B`
- TYPE: DH1 (`0x4`)、Payload body: 27 bytes (`0x1B`) PRBS-9
- Header whitening、rate 1/3 FEC、HEC、Payload whitening、CRCを適用

NPZにはIQに加えて`packet_bits`、`access_bits`、`header_air_bits`、Payload Header、
PRBS-9 body、CRC、air bits、packet sample範囲、各RF/DSP条件を格納しています。再生成:

```powershell
python -m tools.generate_bluetooth_br_iq
```

VSAの初期確認設定はGFSK、1 MSym/s、Deviation 160 kHz、Gaussian BT 0.5です。
Pattern SearchにはAccess Code先頭32 symbol
`10101011011111001100001011011001`をBinaryで指定し、Result Lengthを366 symbolsとします。
回帰試験ではcorrelation 0.9837、pattern error 0、全366 symbol error 0、CFO約
+20.45 kHzを確認しています。

## 5. 保存IQの解析

```powershell
python -m tools.analyze_bluetooth_br_iq capture.npz `
  --analysis-center-frequency 2441000000 `
  --analysis-bandwidth 1500000 `
  --lap 0x9E8B33 `
  --clock 0x2B `
  --uap 0x47
```

送信器のUAP/Clock/Whiteningが不明で、DH1/PRBS-9候補を診断する場合:

```powershell
python -m tools.analyze_bluetooth_br_iq capture.npz `
  --lap 0xC6967E `
  --analysis-center-frequency 2441000000 `
  --analysis-bandwidth 1500000 `
  --search-all-uap `
  --payload-pattern prbs9
```

Inquiry/Page IDの68-bit shortened Access Codeを検索する場合:

```powershell
python -m tools.analyze_bluetooth_br_iq capture.npz `
  --lap 0x9E8B33 `
  --shortened-access-code
```

## 6. Plutoでの捕捉

```powershell
python -m tools.capture_bluetooth_br_iq `
  --center-frequency 2441000000 `
  --sample-rate 16000000 `
  --analysis-center-frequency 2443000000 `
  --analysis-bandwidth 1500000 `
  --duration-ms 3 `
  --attempts 50 `
  --lap 0x9E8B33 `
  --shortened-access-code `
  --output bluetooth_capture.npz
```

`center-frequency`はPlutoのcapture center、`analysis-center-frequency`はそのIQ内で復調する信号の絶対周波数です。後者を中心に`analysis-bandwidth`のDDC/FIR処理を適用します。固定周波数test signalでは対象周波数を直接指定してください。Bluetooth BRの実trafficはfrequency hoppingするため、単一Analysis Centerでの捕捉は確率的です。捕捉成功時は再解析可能な元のwideband IQを保存します。

接続中packetを狙う場合はCentral BD_ADDRのLAP、Header復元にはclock、HEC検証にはUAPが必要です。まずは既知GIACを使えるInquiry trafficでRF捕捉とbit timingを検証するのが安全です。

## 7. 未実装

- 複数channel自動探索とfrequency hopping追従（手動single-channel選択は実装済み）。
- unknown LAP discovery。
- clock/UAPのcandidate searchと連続packet tracking。
- actual packet TYPEに応じたPayload length判定。
- DM系Payloadのrate 2/3 FEC。
- DH1以外のPayload decode、暗号化状態、再送判定（DH1 CRCは実装済み）。
- FHS payloadからBD_ADDR/clockを取得しCAC trackingへ移行する処理。
- EDR guard/sync検出とpi/4-DQPSK/8DPSK segment解析。
- VSA GUIからのBluetooth profile設定・packet result表示。

現在の`payload_bits`はHeader後のbit streamをwhitening解除した結果であり、すべてのBluetooth packet typeについて最終payload dataを意味するものではありません。
