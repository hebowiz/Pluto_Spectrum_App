# Wi-Fi Non-HT OFDM 適合性レビューと検証

## 変更前レビュー

以下は初回実装時の記録。OFDM境界の2024版本文による再確認は末尾の「OFDM Symbol Boundary / Windowing再確認」を参照。

生成器を変更する前に、IEEE Std 802.11a-1999 Clause 17とAnnex Gの数値例を照合した。
36 Mbps / 100 byte / seed 1011101の固定値テストを先に追加した。
出典は [IEEE規格のMITミラー](https://pdos.csail.mit.edu/archive/decouto/papers/802.11a.pdf)。
2.4 GHz ERPは [IEEE 802.11-2012](https://openofdm.readthedocs.io/en/latest/_downloads/802.11-2012.pdf) §19.3.2.4（印刷頁1638）の6 µs無送信区間を参照する。旧版・新版でClause番号が変わるため、設計メモの番号を無条件には流用しない。

| 項目 | 変更前の判定 | 根拠・対応 |
| --- | --- | --- |
| L-STF周波数系列、10回の0.8 µs反復 | Correct | Clause 17.3.3 Eq.(6)。Annex G.2の誤った系列には合わせない |
| L-LTF系列、GI2と2回のlong symbol | Correct | Eq.(8)、1.6 + 3.2 + 3.2 µs |
| L-SIG RATE、Reserved、LENGTHのLSB順、parity範囲、TAIL | Correct | G.7。parityは先頭17 bitのeven parity、TAILは6 zero |
| L-SIG coding | Correct | G.8/G.9/G.11/G.12の固定値と一致 |
| SERVICE、PSDU octet内LSB順、DATA TAIL、PAD | Correct | G.13/G.14/G.16/G.17。全体をscramble後、TAILのみ0に上書き |
| scrambler多項式、state方向、初期状態 | Correct | x^7+x^4+1、G.15の127 bitと一致。非zeroの7-bit seed |
| BCC generator、shift方向、g0/g1順序 | Correct | newestをLSBに置くため133/171の反転mask155/117。G.8/G.18と一致 |
| puncture maskと位相、interleave方向 | Correct | G.18/G.21。DATA全体でencoder stateを継続 |
| BPSK/QPSK/16QAM/64QAM mappingと正規化 | Correct | Tables 82〜85。16QAMはG.22でも照合 |
| data/pilot配置、polarityと開始index、FFT bins | Correct | Eq.(24)/(25)、G.11/G.22。SIGNAL=p0、DATA=p1以降 |
| IFFT係数、GI、各field相対レベル | Correct | 共通係数64/√52によるスケール差を除きSIGNAL時系列が一致。STF/LTF平均電力一致 |
| OFDM境界 | Known limitation | 現行は矩形CP連結。Annexの半値1 sample overlapは非規範例。spectral maskの実測は別途必要 |
| PPDU長 | Correct | 20 + 4×ceil((16+8×LENGTH+6)/N_DBPS) µs |
| ERP Signal Extension | 修正対象 | 現行周期検証がPPDU長のみ。6 µsの無送信時間を最小周期へ追加する |
| MAC byte order、FCS | Correct / 拡張対象 | little endianとCRC-32は既存Beaconで正しい。Raw入力のFCS有無とManualを明示する |
| Beacon field/UI | 拡張対象 | 固定値の編集、可視のChannel/Offset、Repeat、階層化、static TSF/Sequenceの表示を追加する |
| Verify | 未実装 | `packet_bits`を使わず、IQから独立復調してshared Packet Decodeへ接続する |
| PRBS-9 payload | 修正対象 | 右shiftとtap方向が不一致で21 bit周期。MSB出力・左shiftに統一し511 bit周期へ修正 |

## Reference vectorの制約

IEEEの[2000年7月会議録](https://grouper.ieee.org/groups/802/11/Minutes/Cons_Minutes_July-2000.pdf)にはAnnex G.2/G.3/G.24の数値誤りが記録されている。
G.24のDATA時系列もG.22のIFFTと一致しないため、完成波形全体のgolden値には採用しない。
固定値はSIGNAL時系列G.12とDATA周波数bins G.22までを直接比較し、DATA時系列は公表binsをIFFTして比較する。
元PDFのSHA-256、rate、PSDU、seed、抽出方法は [fixture](../../../tests/data/fixtures/wifi/README.md) に記録する。
誤った参考例への一致を目的とした生成器変更は行わない。

現時点の外部実装照合は [gr-ieee802-11の定数](https://github.com/bastibl/gr-ieee802-11/blob/maint-3.10/examples/wifi_phy_hier.grc) によるSTFの確認。GNU Radioとの実行時相互復調や実RF相互接続を実施したという意味ではない。

## 実装結果

Fixed: ERPの最小周期へ6 µsのsilenceを追加。PRBS-9の周期を修正。Channel / Offsetの既存widgetがフォームへ未配置だった点を修正。
FEC・interleave・training・constellationは外部固定値と一致したため維持した。
RawはFCS込みとFCSなしを明示し、Beacon / Raw FCSなしにAuto / Manual付加を実装した。
Beacon全主要field、DS Auto / Manual、static Timestamp / Sequence、周期とBeacon Intervalの比較を追加した。
UIは共通2 tabを維持し、Fields内のグループ切替を共通helperに分離した。

Independent Verify: IQだけからSTF/LTF、CFO、channel、SIGNAL、DATAを復調する。
MSB-newest 133/171 trellisによるViterbiと規格の逆置換式を受信側で独立実装し、送信側関数をimportしない。
生成metadataのPSDU/seed/rate/境界を偽装しても、IQから取得した結果を返すことをテストする。
L-SIG破損、DATA破損、truncation、Manual不正FCSはIssueとなる。

| テストカテゴリ | モジュール |
| --- | --- |
| MAC / Beacon / FCS / persistence | `tests/vsg/test_wifi_mac.py` |
| PHY primitive / mapping / pilot / ERP / diagnostics | `tests/vsg/test_wifi_phy_primitives.py` |
| IEEE固定値 / 送受信primitive | `tests/vsg/test_wifi_ieee_reference.py` |
| IQ round-trip / seed / 最長PSDU / 異常系 | `tests/vsg/test_wifi_iq_verify.py` |
| UI設定・可視性・Cancel | `tests/vsg/test_wifi_settings_ui.py` |
| 共通Packet Verify UI統合 | `tests/vsg/test_vsg_packet_decode.py` |

Digital round-tripは全8 rate×20/40 MS/sでPSDU全byte一致、L-SIG parity、FCSを確認。
追加でseed方向、最大4095 byte、pilot周期越え、任意Raw PSDU、75 kHz CFO、開始sample遅延、複素gainを検証する。
浮動小数点の丸め以外に、送信側の保持bitを受信結果の代わりに使わない。

全自動テストの実行結果: **1,211 passed（246.02秒）**。
`QT_QPA_PLATFORM=offscreen` で `.venv/Scripts/python.exe -m pytest -q` を実行した。
Bluetooth / DECT、バックエンド、出力、共有UIを含む既存テストでも失敗は検出されなかった。
マニュアル画像はWindows Qtの実画面を生成IQで撮影したもので、実RF受信画像ではない。
最終表示レビュー後、Frame Type/Subtypeの表示とFrame Body配下の固定パラメータ・IE階層を整え、
MAC・IQ復調・設定UI・共通Verify統合の関連70件を再実行してすべて成功した。

## 主な変更ファイル

| 責務 | ファイル |
| --- | --- |
| 設定・最小周期 | `pluto_vsg/model.py`、`pluto_vsg/wifi/validation.py` |
| Beacon / Raw / PRBS / FCS | `pluto_vsg/wifi/mac.py` |
| ERP・診断配列の任意保持 | `pluto_vsg/engine/wifi_legacy_ofdm.py` |
| 独立IQ復調・MAC解析 | `pluto_protocol/wifi/{__init__,non_ht,mac}.py`、`pluto_protocol/registry.py` |
| Verify接続 | `pluto_vsg/protocol.py`、`pluto_vsg/ui/main_window.py` |
| 設定UIと共有グループ | `pluto_vsg/ui/wifi_settings.py`、`pluto_vsg/ui/packet_settings.py` |
| 固定値・自動テスト | `tests/data/fixtures/wifi/`、上表の6テストモジュール |
| 操作資料 | `docs/user-manual/Pluto_VSG_User_Manual_JA.md`、画面索引・検証記録、`docs/images/user-manual/pluto-vsg-wifi-*.png` |
| 仕様・設計・検証 | `docs/spec/vsg/wifi-non-ht.md`、本書、実機手順、既存Wi-Fi設計メモ・共有解析設計 |
| 画面再生成 | `tools/generate_user_manual_screenshots.py` |

## 残る制限と実機確認

Known limitations: first-packet offline decode、20/40 MS/sのみ。雑音中の継続検出・一般multipathやsample clock追従は対象外。
既知の誤りがあるAnnex G.24の全packet時系列へ完全一致したとはしない。
矩形CP連結のspectral maskと実機相互接続は未確認。[実機手順](wifi-non-ht-hardware.md)に分けて記録する。
PHYが有効でも任意Pattern/PRBSや不整合Manual fieldが有効なMAC/Beaconになるとは限らない。

## HT / DSSS-CCKへ再利用する部分

共通のPacketAnalysisResult、registry、Decode/Payload Hex/Issues、MAC parser、Beacon builder、FCS設定、
RF/Timing・Fieldsの枠とfield group、保存・出力・バックエンドは再利用可能。
HTではlegacy preamble部分を再利用候補とし、HT-SIG・HT DATA等は別PHYにする。
DSSS/CCKへOFDM同期・interleave・trellisを無理に共有せず、IQからPSDUへの専用経路を追加する。
現行の動作要件は [仕様](../../spec/vsg/wifi-non-ht.md) を参照。

## Probe Request / Probe Response追加の検証

ローカルのIEEE Std 802.11-2024を一次資料として照合した。元PDF・本文抜粋・規格ページ画像は配布物へ含めない。
今回PHY engine、RF level、backend、WV exportは変更せず、MAC生成・設定・共通MAC decodeを拡張した。

### 参照箇所

| Clause / Table（2024版） | 照合内容・実装への反映 |
| --- | --- |
| 9.2.4.1.3 / Table 9-1 | Type=Management、Subtype=4 Request / 5 Response / 8 Beacon |
| 9.2.4.3、9.2.4.4.1 | Addressとwildcard BSSID、Sequence上位12 bit / Fragment下位4 bit |
| 9.3.3.1 / Figure 9-128 | 24-byte Management header、Address 1=DA/RA・2=SA/TA・3=BSSID、Duration |
| 9.3.3.2 / Table 9-62 | Beacon固定fieldとIE順を維持 |
| 9.3.3.9 / Table 9-68 | Requestは固定fieldなし、SSID・Rates・Extended Rates、非RMではDS省略可能 |
| 9.3.3.10 / Table 9-69 | Responseの12-byte固定field、ERPでのDS・ERP IE、TIMなし |
| 9.4.2.2 / Figure 9-209 | SSID 0〜32 octet、Requestの空SSIDはwildcard |
| 9.4.2.3、9.4.2.11 / Figures 9-210、9-233 | Rates 1〜8 / Extended 1〜255 octet、500 kbps単位、RequestのBasic bitは受信側で無視 |
| 9.4.2.4、9.4.2.10 | DSSS Parameter Setのchannel、ERP Informationのbit構造 |
| 10.6.5.1、10.6.5.4、10.6.5.8 | Beacon/group/unicastの送信レートとBasic/相手対応レートの条件 |
| 11.1.4.3.2、11.1.4.6 | active scanning、8個を超える広告rateとExtended IE |
| 17.4.2 / Table 17-23、18.1.2 | 6/12/24 MbpsがOFDM必須、6 MbpsがERP必須に含まれること |
| 18.1.3、18.5.3.2 | ERP SIFS=10 µs、TXTIMEに6 µs Signal Extensionを含むこと |

実装対象はopen infrastructure / non-RM / non-HTのstatic試験frame。
セキュリティ・HT/VHT/HE・Country等の条件付きIEを持つ全BSS構成を自動構築するものではない。
Additional IEsはユーザー指定の完成TLVを追加できるが、機能の有効化条件・重複・順序の意味検証は行わない。

### 実装完了項目（依頼の20項目）

| No. | 項目 | 結果 |
| --- | --- | --- |
| 1 | 再利用 | Non-HT PHY全段、ERP 6 µs、IQ Verify経路、共通2 tab / field group、FCS、保存・出力 |
| 2 | 新Settings | `frame_control_auto=True`、`extended_supported_rates_hex=""`、`additional_ies_hex=""` |
| 3 | Source | `PROBE_REQUEST="Probe Request"`、`PROBE_RESPONSE="Probe Response"` |
| 4 | frame構造 | Beacon/Response=header 24 + fixed 12 + IE + FCS 4、Request=header 24 + IE + FCS 4 |
| 5 | Address初期値 | Request DA/BSSID=broadcast・SA末尾66、Response DA末尾66・SA/BSSID末尾55。全値とBeaconは[仕様](../../spec/vsg/wifi-non-ht.md#明示的な初期値適用)参照 |
| 6 | IE | Request 0/1/50、Response 0/1/3/42/50、Beacon 0/1/3/5/42/50。50は非空時、追加TLVは末尾 |
| 7 | IEEE参照 | 上表。ローカル2024版に基づく |
| 8 | FC | Auto=0080/0040/0050、Manualは任意16-bit。Source切替でmanual値を消さない |
| 9 | Request SSID | 検索対象。空文字=Wildcard、非空=Specific。送信元情報はSA・Rates・追加Capability等のIEで設定 |
| 10 | Response固定field | static Timestamp、100 TU等のInterval、Capabilityを編集可能。TIM標準生成なし |
| 11 | Rates | 既存1〜8 octetを維持しExtendedを別欄に追加。Probe presetはERP 12 ratesを8+4に分割 |
| 12 | FCS | 3種ともAuto CRC-32 little endian / Manual送信順4 octet。不正CRCも試験用に許可 |
| 13 | UI切替 | 6群をSourceに応じて無効化。Source変更は入力値を保持、Defaultsは明示操作。Inspectorは適用値のみ表示 |
| 14 | Validation | 非適用fieldを除外。SSID UTF-8 byte数、Rates長、追加TLV長、適用field範囲、最終PSDU長・最小周期を確認 |
| 15 | 共通decoder | Subtype 4/5の追加、共通IE parser、Extended Rates、wildcard意味表示、未知IE保持、種別別必須IE警告 |
| 16 | IQ Verify | 3種×20/40 MS/s、生成metadataのPSDU/rateを偽装してもIQからPSDU一致・正しいpacket_type・FCS Validを確認 |
| 17 | テスト | 新規Managementテスト32件、UI追加4件、Inspector/Verify追加4件。既存Beacon/全8 rate×2 Fs・VSA Wi-Fi回帰を併用 |
| 18 | 旧project | 新fieldなしJSONのliteral FCはManualへ移行、FC自体なしならAuto。従来Beacon byte順と任意project名を維持 |
| 19 | 実機残項目 | [手順](wifi-non-ht-hardware.md)に3種のWireshark field/FCS比較、AP応答、OS scanを記載。すべて未実施 |
| 20 | 制限 | static replay、CSMA/CA・ACK・reactive responseなし。6 MbpsはERP向けでDSSS-only受信機との相互接続を保証しない |

### 自動確認と表示確認

`QT_QPA_PLATFORM=offscreen`で`.venv/Scripts/python.exe -m pytest tests/vsg tests/vsa/wifi -q`を実行し、
**486 passed（106.26秒）**。VSGの他規格・backend・出力・UIとVSA Wi-Fiを含む関連範囲で、全体suiteの再実行ではない。
初回のbackend 17件はsandboxのAppDataデバイスロック書き込み制限で失敗したため、同コマンドを権限付きで再実行した。

`tests/vsg/test_wifi_management.py`の3つの固定byte列はheader・fixed・IEを明示し、
CRCは反射多項式`0xEDB88320`のbit計算で独立に求めた末尾定数と比較する。
builder/decoder相互一致だけを根拠にbyte orderを判定していない。
既存VSAの任意FC試験は新設Manual modeを明示し、Retry/More Data等の従来assertionを維持した。

Qt offscreenの実widgetを描画してRequestのMAC Header / Common IEs、Response固定fieldを確認。
無効群、Auto FC値、送信元・宛先欄、Wildcard説明、追加IE欄が表示され、内容の欠けは認めなかった。
今回ユーザーマニュアル・マニュアル画像・PDFは改訂しない。Probe操作説明のマニュアル追記は次回の明示依頼時に行う。

## OFDM Symbol Boundary / Windowing再確認

### 結論と一次資料

**Case A：規格の標準波形はrectangular。現行boundary constructionは一致しており、IQ生成処理の修正は不要。**
IEEE Std 802.11-2024のローカル本文を確認し、数式・図のページも描画して照合した。
旧Working Group資料や1999版Annexだけからwindowの要否を判断していない。
元PDFおよび描画した規格ページをリポジトリや配布物へ含めない。

| Clause / Subclause | 確認事項 |
| --- | --- |
| 17.3.2.4 / Table 17-5（印刷頁3347） | useful=3.2 µs、GI=0.8 µs、GI2=1.6 µs、STF/LTF各8 µs、SIGNAL/DATA各4 µs |
| 17.3.2.5 / (17-2)〜(17-4)（3347〜3348） | subfieldを矩形窓付き逆フーリエ和で定義し、guard分の時間シフトでcyclic prefixを構成。SIGNAL開始16 µs、DATA開始20 µs |
| 17.3.2.5 / Figure 17-2（3348〜3349） | 非zero T_TRの窓・overlapは平滑化の実装例。約100 nsは固定必須値ではない。windowing以外のfiltering等も認める |
| 17.3.2.6 / (17-5)（3349） | discrete implementationの説明はinformationalと明記。20 MS/sでn=0,80を半値にする例をmandatory処理としない |
| 17.3.3 / (17-6)〜(17-10)（3350〜3351） | STF系列と10反復、LTF系列とGI2 + 2周期、両fieldの連結 |
| 17.3.4.1（3351） | SIGNALはpreamble直後のBPSK 1/2 OFDM symbol |
| 17.3.5.10 / (17-22)〜(17-26)（3365〜3367） | DATAのguard・pilotを含むOFDM和、symbolごとの時間シフトによる連結 |
| 17.3.9.3（3370〜3371） | 20 MHzの送信Spectrum Maskと100 kHz RBW / 30 kHz VBW条件 |
| 17.3.9.7〜17.3.9.8（3372〜3374） | leakage・flatness・rate依存constellation errorが拘束要件。受信機相当のFFT/channel/phase補正を用いた測定 |
| 18.3.2.4（3392〜3393） | ERP-OFDMは17.3.2〜17.3.5のformatを利用し、6 µsの無送信Signal Extensionを別に設ける |

17.3.2.5にはsidelobe低減に平滑化が必要との説明もあるが、拘束要件として指定するのはSpectrum MaskとModulation Accuracy。
rectangularの参照波形を生成できることと、DAC・再構成filter・RFを含む送信機がmaskに適合することは別である。
今回、cosine窓や100 ns overlap、外側RF envelopeを追加して適合を主張することはしない。

### 実装判断と独立検証

現行処理で正しいのは、L-SIG/DATAの16+64 sample、L-LTFの32+64+64 sample、STFの16×10 sample、
および20 MS/sで160 / 320 / 400を境界とする連結。40 MS/sではすべて2倍になる。
CPはuseful末尾のcyclic copyであり、boundary sampleの重複・欠落はなかった。
矩形subfield間の振幅差は誤りではない。隣接sampleを平滑化して一致させるassertionは置かない。

変更はmetadata表現・IEEE参照の追加、実装コメント、設定画面の読み取り専用表示、および回帰テスト。
MAC/Probe/Beacon、bit生成・FEC・mapping・pilot、RF Level、ERP、backend、WV export、受信FFT位置は変更していない。

`tests/vsg/test_wifi_ofdm_boundaries.py`に以下の独立確認を追加した。

- 全8 rate×20/40 MS/sのfield長、CP全sample、LTF GI2/反復、STF反復、ERP無送信区間。
- 本文の複素指数関数和を時間軸上で直接評価し、PPDU全sampleと全join前後を比較。
  参照計算は生成器のIFFT/CP/training/pilot helperを使用しない。許容する補正は全体に共通の実数gainのみ。
  DATA constellationは入力として使い、bit/mappingの正しさは既存の外部固定vectorテストで別途確認する。
- 40 MS/sの偶数sampleと20 MS/sが共通gainを除いて一致。奇数sampleも直接フーリエ和との比較対象。
- 既存IEEE Annex G.11/G.22の独立したSIGNAL/DATA binsから20/40 MS/sの時間波形を計算し、
  最初のsampleを含めた全sampleを比較。旧informative例の半値端点は標準矩形波形へ強制しない。
- L-SIG/DATAのCPだけを意図的にゼロへ壊しても、独立decoderの抽出constellationと復元PSDUが変わらないことを確認。
  FFTがCPを含まず、20 MS/s換算でSIGNAL `[336,400)`、DATA `[416+80n,480+80n)`を使うことの回帰検証。
- 直接フーリエ和と生成IQのnormalized FFT power spectrum、および観測可能帯域内の|f|≥9 MHzの積分電力比を比較。
  デジタルspectrum regressionであり、規定RBW/VBWによるRF mask測定ではない。
- Boundary metadata、Outer RF Envelopeとの独立性、設定画面の固定表示を確認。

20/40 MS/sのNyquist範囲はそれぞれ±10/±20 MHzであり、20 MHz maskの±30 MHz以上は直接確認できない。
有限sample rateのbaseband FFTとRF測定を同一視しない。今回filterやwindow shapeを変更する根拠はなく、
RF mask・hardware reconstruction filtering・EVMの実測は[実機手順](wifi-non-ht-hardware.md)で別途行う。

### Known limitationsと報告範囲

検証結果：**関連306件PASS（43.12秒、Qt offscreen）**。
対象はVSGの境界・IEEE参照・PHY primitives・IQ Verify・MAC/Management・設定UI・Packet Decode統合と、
VSA Wi-Fi全テスト。今回追加したboundaryテストは59件。全リポジトリsuiteの再実行ではない。
全8 rate×20/40 MS/sでL-SIG・PSDU復元・FCS成功を確認し、VSAのDATA EVM最大値は
RMS **0.0000036161 %以下**、peak **0.000010455 %以下**（理想IQ、sample数やsample rateによる浮動小数点誤差を含む）。
変更前後に同一条件（default Beacon、Fixed seed、1 ms period）で生成した16波形のcomplex64 IQは
SHA-256が全件一致した。今回の変更によるspectrum・EVM・RF levelへの波形上の差はない。

非矩形windowはnormativeな固定処理ではなく、今回追加していない。overlap/addも必須ではない。
理想IQのdecode/EVM確認は実機のmodulation accuracy適合判定ではない。
17.3.9.8の20 PPDU以上・DATA 16 symbol以上・random data等を満たすRF試験や、実RF spectrum maskは未実施。
境界確認後の明示依頼に基づき、UI表示をユーザーマニュアル・画像・PDFへ反映済み。
改訂内容とPDFの確認結果は[マニュアル確認記録](../../user-manual/manual-validation.md)を参照。
