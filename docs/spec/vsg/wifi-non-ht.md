# Wi-Fi Non-HT OFDM生成とPacket Verify

## 対象

Non-HT OFDM、20 MHz、6/9/12/18/24/36/48/54 Mbps、20/40 MS/s。
RFチャンネルは2.4 GHzの1〜13、周波数Offsetはチャンネル中心からのRF周波数変更。
IQへの人工的なCFO付与とは別で、出力バックエンドの既存周波数制約に従う。

## パケットと周期

active PPDUはL-STF / L-LTF / L-SIG / DATAで構成する。
PSDU長はFCSを含め1〜4095 byte。L-SIG LENGTH / parityはAuto、Reserved / TAILはzero。
SERVICE + PSDU + TAIL + PADをscrambleし、DATAのTAILだけを符号化前にzeroへ戻す。
Auto scrambler seedは生成ごとに非zeroの7-bit値を選び、Fixedでは1〜127を指定する。

`N_SYM = ceil((16 + 8×PSDU bytes + 6) / N_DBPS)`、active PPDU長は`20 + 4×N_SYM` µs。
ERP Signal Extensionの6 µsは無送信時間とし、DATA symbolやactive RMS範囲へ加えない。
Packet PeriodはPPDU長+6 µs以上。UIと生成器の双方で検証する。
繰り返しは同一IQの再生であり、Timestamp / Sequence Number / seed / FCSを再生成しない。

OFDM境界は現在、CP付きsymbolの矩形連結。共通Power EnvelopeとOFDMの任意のwindow/overlap処理を混同しない。
独自rampでdecodeを補う処理は追加しない。スペクトルマスクは実機測定が必要。

## PSDUとFCS

| Source | 入力の意味 |
| --- | --- |
| Raw PSDU including FCS | hex octetをそのまま使う。FCSを追加・修正しない。旧Rawプロジェクトもこの意味を保持 |
| Raw MAC frame without FCS | hex octetにAuto CRC-32またはManual 4 octetを末尾付加 |
| Pattern / PRBS-9 | 指定長の合成PSDU。MAC headerやFCSは追加しない。PHY decode成功とMAC妥当性は別 |
| Beacon | MAC header、固定パラメータ、IEを構築しAuto / Manual FCSを付加 |

Manual FCSは「送信順の4 octet」で入力する。CRC整数を記入する欄ではない。
PRBS-9はx^9+x^5+1、初期状態all-ones、MSB出力、511 bit周期をoctet内LSB順に詰める。

BeaconではFrame Control、Duration/ID、Destination、Source、BSSID、Sequence/Fragment、Timestamp、
Beacon Interval、Capability、SSID、Supported Rates、DS channel、TIM、ERP Informationを編集できる。
Source空欄はBSSIDと同じ。DS channelはAutoでRFチャンネルに追従し、Manualも可能。
Beacon IntervalはTU、周期はµs。`Use Beacon interval × 1024 us`で周期へコピーでき、異なる場合は表示で知らせる。
Timestamp / Sequenceはstaticと表示する。Manual編集は異常系の作成も許すため、任意の組合せでBeacon検出を保証しない。

Default BeaconはChannel 6 / 2437 MHz / 6 Mbps、SSID `Pluto_Test_AP`、BSSID `02:11:22:33:44:55`、
宛先broadcast、100 TU / 102.4 ms、ESS + short-slot + open、FCS Auto。
SSID / Supported Rates / DS Parameter Set / TIM / ERP Informationを含める。

## Verifyの独立性と表示

Wi-FiのVerify Packetは`GenerationResult.iq`とsample rateを使い、最初のpacketを検証する。
rate・length・seed・PSDU・packet境界をgenerator metadataや`packet_bits`から取得しない。
`pluto_protocol/wifi/non_ht.py`がSTF検出、CFO、LTF同期・channel推定、pilot補正、demap、独立した
deinterleave / depuncture / Viterbi / descrambleを行う。MAC解釈は同packageの`mac.py`が担う。
`wifi.non_ht`をshared registryへ登録し、結果は既存の`PacketAnalysisResult`で返す。
bit入力のMAC解析を行うAPIと、IQからPHYを検証するAPIは区別する。

画面は既存のDecode / Payload Hex / Issuesを使う。L-SIG RATE / LENGTH / parity、modulation、coding、
N_SYM、SERVICE / TAIL / PAD、PSDU completeness、MAC/Beaconの各field、FCSを表示する。
truncated packetや異常signalは取得できたPHY情報と理由を返し、生成元データで補完しない。
Waveform側はPPDUの時間領域、Decode側はPSDU bit/byte領域とし、SSIDなどへ偽の連続sample範囲を割り当てない。

通常metadataにはPSDU・短いL-SIG等を残す。大きな中間bit配列は`generate(..., diagnostics=True)`の場合のみ保持する。
Bluetooth / DECTのVerifyは従来のbit解析を維持する。

## 検証・制約

[検証記録](../../verification/vsg/wifi-non-ht-review.md) と [実機確認手順](../../verification/vsg/wifi-non-ht-hardware.md) を参照。
RF相互接続、spectral mask、continuous受信、ACK、association、CSMA/CA、HT/DSSS-CCKは自動Verifyの対象外。
今回のdecoderは生成済みの理想IQを主対象とし、任意の雑音・multipath・sample clock errorへの性能を保証しない。
