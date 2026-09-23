# Wi-Fi Non-HT実機相互接続確認

これは未実施の手順書。自動IQ round-tripの成功を実RF受信成功とは記録しない。

## Beacon

1. VSGでNew Wi-Fi Packetを選び、Non-HT OFDM / Channel 6 / 2437 MHz / 6 Mbpsに設定する。
2. SSID `Pluto_Test_AP`、BSSID `02:11:22:33:44:55`、Destination broadcast、FCS Auto、
   Beacon Interval 100 TU、Packet Period 102400 µs、Frequency Offset 0を確認する。
3. Generateし、Verify PacketでL-SIG parity、PSDU completeness、FCSがValidとなることを確認する。
4. 検証用の減衰・接続条件と出力レベルを記録し、Plutoから繰り返し送信する。
5. monitor-mode adapterをChannel 6へ固定しcaptureする。WiresharkでBeaconを絞り込む。
6. SSID、BSSID、DS Channel=6、Beacon Interval=100 TU、Supported Rates、TIM、ERP IEを確認する。
7. adapterがFCSを保持・報告できる場合はFCS goodを確認する。FCSを除去するadapterで
   「エラー表示がない」ことだけをFCS検証済みとはしない。
8. PC / AndroidのscanでSSIDを確認する。各OS・adapter・driver・scan条件と検出率を記録する。

Timestamp / Sequenceはstaticなcyclic replayであり、通常APの継続動作を模擬していない。
これによるscan側の扱い、周波数誤差、受信帯域、送信レベル、I/Q/DC不平衡などを分けて調査する。
association / ACK応答は本ツールの対象外。

## 任意のMAC frameとレート

- 既知のMAC frameをRaw without FCS / Autoで生成するか、既知FCS付きPSDUをRaw including FCSで投入する。
- 全8レート、20/40 MS/sを同一PSDUで比較し、monitor receiverでTYPE、Length、内容、FCSを確認する。
- 最小周期ではactive PPDU終了後の6 µsを含む無送信時間があることをRF captureで確認する。
- FCS Manualで1 bit変えたframeを送信し、受信側がbad FCSとして扱うことを確認する（bad frameを破棄する設定も記録）。
- 必要なRF評価は別途、EVM、周波数誤差、spectral mask、symbol境界、Plutoの補間フィルタを測定する。

## 記録欄

| 項目 | 記録 |
| --- | --- |
| アプリrevision / project file | 未実施 |
| Pluto firmware / sample rate / RF bandwidth / Tx gain / attenuation | 未実施 |
| Receiver / driver / OS / monitor channel | 未実施 |
| Wireshark capture / FCS保持有無 / Beacon field一致 | 未実施 |
| PC / Android SSID検出 / 試行回数 | 未実施 |
| RF波形 / EVM / spectral mask | 未実施 |

結果は同じ検証フォルダに追記し、captureをテストfixture化する場合は`tests/data/fixtures/wifi/`へ出典・条件を添える。
