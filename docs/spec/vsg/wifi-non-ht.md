# Wi-Fi Non-HT OFDM生成とPacket Verify

## 対象

Non-HT OFDM、20 MHz、6/9/12/18/24/36/48/54 Mbps、20/40 MS/s。
RFチャンネルは2.4 GHzの1〜13、周波数Offsetはチャンネル中心からのRF周波数変更。
IQへの人工的なCFO付与とは別で、出力バックエンドの既存周波数制約に従う。

パケット設定は右側UIの`PACKET > Packet Settings`から開く。Inspector内には重複する設定ボタンを置かない。

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
| Probe Request | MAC header + IE。固定パラメータを含めずAuto / Manual FCSを付加 |
| Probe Response | MAC header + 固定パラメータ + IE。TIMを含めずAuto / Manual FCSを付加 |

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

## Management Frame設定

共通の`RF / Timing`と`Fields`を維持する。FieldsはSource / Payload、Management MAC Header、
Common Management IEs、Beacon / Probe Response Fixed Fields、Beacon-only IEs、FCSの6群。
選択したSourceで使わない群・入力欄は無効にし、Validationからも除く。
Probe RequestではTimestamp / Beacon Interval / Capability / DS / ERP / TIMを使用せず、
Probe ResponseではTIMを使用しない。周期をBeacon Intervalから設定する補助ボタンはBeacon専用。
Probe ResponseのBeacon Intervalは広告するBSSの値であり、static responseの再生周期とは独立する。

Frame ControlはAuto / Manual。AutoはBeacon `0x0080`、Probe Request `0x0040`、Probe Response `0x0050`。
実効値を読み取り専用表示し、Manualの16-bit入力値はAutoへの切り替え中も保持する。
旧JSONにFC値があり新しいmodeがない場合はManualとして読み込み、既存byte列を維持する。
FC自体もない旧JSONと新規設定はAuto。ManualでSubtype/flagsを変更してもbody形式は選択Sourceのままで、
任意の組合せの規格適合性は保証しない（+HTC等の追加headerを自動挿入しない）。

全3種でDuration / Address 1=Destination / Address 2=Source / Address 3=BSSID / Sequence / Fragment、
SSID、Supported Rates、Extended Supported Rates、Additional IEs、FCSを編集可能。
Probe RequestのSourceは送信元STA自身のMAC、RatesはSTAの対応レート、SSIDは検索対象である。
SSIDはUTF-8で最大32 byte。Requestの空文字はWildcard、非空文字はSpecificとして扱う。
空文字を既定のSSIDへ補完しない。Capability Information固定fieldはRequestには存在しない。
HT CapabilitiesやVendor Specific等の送信元情報はAdditional IEsで指定できる。

| Frame | 固定field | 標準生成IE順（ID） |
| --- | --- | --- |
| Beacon | Timestamp 8 / Interval 2 / Capability 2 byte | SSID(0)、Supported Rates(1)、DS(3)、TIM(5)、ERP(42)、Extended Rates(50、非空時) |
| Probe Request | なし | SSID(0)、Supported Rates(1)、Extended Rates(50、非空時) |
| Probe Response | Timestamp 8 / Interval 2 / Capability 2 byte | SSID(0)、Supported Rates(1)、DS(3)、ERP(42)、Extended Rates(50、非空時) |

MAC headerは24 byte、FCSは4 byte。Supported Ratesは従来どおり1〜8 octet、
追加の`extended_supported_rates_hex`は空欄なら省略、非空なら1〜255 octetをID 50として出力する。
Ratesを9個以上指定する場合は両欄へ分割する。自動ソートやBasic bitの書き換えはしない。
`additional_ies_hex`は完成したID / Length / Value列。TLVの長さを検証し、標準IEの後に入力順で追加する。
内容・重複・追加IE間の順序や機能ごとの条件までは自動補正しない。任意位置への挿入はRaw PSDUを使う。
Probe Requestの標準生成ではRM非対応を前提に任意のDS IEを省略する。

### 明示的な初期値適用

Source切り替えだけではアドレスやSSIDなどの入力値を上書きしない。
`Apply Beacon / Probe Request / Probe Response Default`でManagement fieldとData Rateを初期化する。
Project Name、RF channel、周波数Offset、sample rate、Packet Period、Repeat、Scramblerは維持する。

| Default | Destination | Source（実効値） | BSSID | SSID | Duration |
| --- | --- | --- | --- | --- | --- |
| Beacon | FF:FF:FF:FF:FF:FF | 02:11:22:33:44:55 | 02:11:22:33:44:55 | Pluto_Test_AP | 0 |
| Probe Request | FF:FF:FF:FF:FF:FF | 02:11:22:33:44:66 | FF:FF:FF:FF:FF:FF | 空文字 / Wildcard | 0 |
| Probe Response | 02:11:22:33:44:66（試験STA） | 02:11:22:33:44:55 | 02:11:22:33:44:55 | Pluto_Test_AP | 60 µs |

Probe両presetは2.4 GHz ERP試験用に1/2/5.5/11/6/9/12/18/24/36/48/54 Mbpsを広告する。
Requestは`02040B160C121824` + `3048606C`、Responseは`82848B968C129824` + `B048606C`。
ResponseのBasic ratesは1/2/5.5/11/6/12/24 Mbps、送信レート6 Mbpsを含む。
6 MbpsはERP必須レートの一つ。Beaconの従来rate octet列は変更しない。
広告するDSSS/CCKレートは模擬STA/BSSの能力情報であり、本VSGにDSSS/CCK PHYを追加するものではない。
Response Duration=60 µsは6 Mbps ERP ACKの50 µs + SIFS 10 µsを想定した試験値。
レート・宛先・fragment等を任意編集してもDurationは自動再計算しない。
Response固定fieldはstatic Timestamp=0、Interval=100 TU、Capability=`0x0401`、DSは現在channel、ERP=0。
全presetはFrame Control Auto / FCS Auto / Sequence=0 / Fragment=0 / Additional IEs空欄。

新規project名は`Wi-Fi Beacon` / `Wi-Fi Probe Request` / `Wi-Fi Probe Response`。
保存済みの任意Project Nameは維持する。InspectorにはSourceに応じたSSID・送信元・宛先・BSSIDを表示し、
Raw / Pattern / PRBSにはManagement値を適用しない。

## Verifyの独立性と表示

Wi-FiのVerify Packetは`GenerationResult.iq`とsample rateを使い、最初のpacketを検証する。
rate・length・seed・PSDU・packet境界をgenerator metadataや`packet_bits`から取得しない。
`pluto_protocol/wifi/non_ht.py`がSTF検出、CFO、LTF同期・channel推定、pilot補正、demap、独立した
deinterleave / depuncture / Viterbi / descrambleを行う。MAC解釈は同packageの`mac.py`が担う。
`wifi.non_ht`をshared registryへ登録し、結果は既存の`PacketAnalysisResult`で返す。
bit入力のMAC解析を行うAPIと、IQからPHYを検証するAPIは区別する。

画面は既存のDecode / Payload Hex / Issuesを使う。L-SIG RATE / LENGTH / parity、modulation、coding、
N_SYM、SERVICE / TAIL / PAD、PSDU completeness、MAC/Managementの各field、FCSを表示する。
共有MAC decoderはSubtype 4 / 5 / 8を認識し、共通IE parserでRequest / Response / Beaconを解析する。
Requestの空SSIDは値を空文字のまま保持し、意味を`Wildcard / empty`と表示する。
Extended Ratesを解釈し、未知IEはID / Length / Raw Valueを保持する。必須IE警告はframe種別ごとに行う。
truncated packetや異常signalは取得できたPHY情報と理由を返し、生成元データで補完しない。
Waveform側はPPDUの時間領域、Decode側はPSDU bit/byte領域とし、SSIDなどへ偽の連続sample範囲を割り当てない。

通常metadataにはPSDU・短いL-SIG等を残す。大きな中間bit配列は`generate(..., diagnostics=True)`の場合のみ保持する。
Bluetooth / DECTのVerifyは従来のbit解析を維持する。

## 検証・制約

[検証記録](../../verification/vsg/wifi-non-ht-review.md) と [実機確認手順](../../verification/vsg/wifi-non-ht-hardware.md) を参照。
RF相互接続、spectral mask、continuous受信、ACK、association、CSMA/CA、HT/DSSS-CCKは自動Verifyの対象外。
Reactive Probe Responseは対象外。受信STA情報・送信タイミングの動的取得は行わず、指定宛先へstatic waveformを生成する。
今回のdecoderは生成済みの理想IQを主対象とし、任意の雑音・multipath・sample clock errorへの性能を保証しない。
