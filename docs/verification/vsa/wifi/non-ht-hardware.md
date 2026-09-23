# Wi-Fi VSA 手動RF受入手順

**状態: 未実施。以下にPASSの実測結果はない。**

## 記録する条件

送受Plutoの個体/firmware、ケーブル・外部ATT、内部gain、sample rate、RF bandwidth、LO/analysis filter、
送信power、waveformのPSDU/seed/rate、capture長とtrigger、使用版、室温・測定時刻を記録する。
power/EVMは入力飽和と雑音床の影響を受けるので、ATTを変えた再測定で安定域を確認する。

## 有線RF round-trip

1. 送信Pluto VSG → ケーブル/適切なATT → 別の受信Pluto VSAを接続する。
2. VSGはChannel 6 / 2437 MHz / Non-HT / 6 MbpsのDefault Beacon、100 TU周期。
3. VSAはWi-Fi、40 MS/s、RF BW 30 MHz、Rate Auto、同じ中心周波数。
4. 100 TU周期を含めるにはcaptureを150 ms程度へ広げるかIQ Power triggerを使う。
5. Singleで検出、Rate=6、Length=送信PSDU長、modulation=BPSK、PSDU全byte、FCSを比較する。
6. Packet List選択と6領域の追従、Spectrum、packet/peak power、CFO、L-SIG/DATA EVMを記録する。
7. 9/12/18/24/36/48/54 Mbpsと任意の正常Raw MAC frameへ切り替えて反復する。
8. Continuous開始・停止、停止後のmode切替、取得中の終了を確認する。
9. input ATT/gainを変えてPower補正とEVMの変化を確認する。未校正の絶対powerを校正済みとして扱わない。

## 独立信号源と外部packet

- 同じIQをSMCV100Bから送信し、Pluto VSAでRate/Length/PSDU/FCSを比較する。
- 可能なら独立信号源のNon-HT波形も使い、VSGだけに成立する条件がないことを確認する。
- 市販AP等の20 MHz Non-HT packetを受信し、monitor-mode受信機のpcapとAddress/Sequence/Length/PSDUを照合する。
- HT/VHT/HEやDSSS/CCKをNon-HT成功と誤報しないことを確認する。
- CFOとEVMの数値比較には独立校正機器を使い、測定範囲・等化・pilot補正・正規化の条件をそろえる。

## 結果記録欄

| 試験 | 状態 | capture / pcap / 条件・観察 |
| --- | --- | --- |
| Pluto→Pluto 全8 rate | 未実施 | — |
| CFO / EVM / power比較 | 未実施 | — |
| SMCV→Pluto | 未実施 | — |
| 市販機器Non-HT | 未実施 | — |
| Continuous / 停止 / 終了 | 未実施 | — |

収録データは`tests/data/fixtures/wifi/`、人が読む比較画像・測定証跡は`docs/verification/assets/`へ配置する。
