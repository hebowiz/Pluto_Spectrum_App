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

実Bluetooth送信機から取得したIQによる検証はまだ行っていません。

## 5. 保存IQの解析

```powershell
python -m tools.analyze_bluetooth_br_iq capture.npz `
  --lap 0x9E8B33 `
  --clock 0x2B `
  --uap 0x47
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
  --duration-ms 3 `
  --attempts 50 `
  --lap 0x9E8B33 `
  --shortened-access-code `
  --output bluetooth_capture.npz
```

Bluetooth BRはfrequency hoppingするため、単一center frequencyでの捕捉は確率的です。16 MHz capture帯域内の複数channelを同時に受信できますが、現在のprofileはrecord全体を1 complex basebandとして検索し、channelizerはまだありません。捕捉成功時だけIQを保存します。

接続中packetを狙う場合はCentral BD_ADDRのLAP、Header復元にはclock、HEC検証にはUAPが必要です。まずは既知GIACを使えるInquiry trafficでRF捕捉とbit timingを検証するのが安全です。

## 7. 未実装

- 16 MHz内を1 MHz channelごとに分離するBluetooth channelizer。
- unknown LAP discovery。
- clock/UAPのcandidate searchと連続packet tracking。
- actual packet TYPEに応じたPayload length判定。
- DM系Payloadのrate 2/3 FEC。
- Payload CRC、暗号化状態、再送判定。
- FHS payloadからBD_ADDR/clockを取得しCAC trackingへ移行する処理。
- EDR guard/sync検出とpi/4-DQPSK/8DPSK segment解析。
- VSA GUIからのBluetooth profile設定・packet result表示。

現在の`payload_bits`はHeader後のbit streamをwhitening解除した結果であり、すべてのBluetooth packet typeについて最終payload dataを意味するものではありません。
