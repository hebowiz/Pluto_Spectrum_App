# Wi-Fi Non-HT OFDM 適合性レビューと検証

## 変更前レビュー

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
