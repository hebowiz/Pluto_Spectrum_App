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

## Probe Request（未実施）

1. SourceをProbe Requestに変更し`Apply Probe Request Default`。6 Mbps、Channel 6、
   Address 1 / 3=broadcast、Address 2=`02:11:22:33:44:66`、SSID空欄、FCS Autoを確認する。
2. Generate → VerifyでSubtype 4、Wildcard / empty、Supported / Extended Rates、FCS Validを確認する。
3. Beaconと同じ接続・減衰・受信条件を記録して送信し、monitor-mode adapter / Wiresharkで
   `wlan.fc.type_subtype == 0x04`を確認する。送信元MAC、空SSID、rates、FCS保持時のGoodを確認する。
4. 送信元を別のlocal unicast MACに変更し、Specific SSIDを設定したpacketでも一致を確認する。
   directed BSSIDを使う場合はAddress 1 / 3を対象BSSIDへ合わせる。
5. Additional IEsに試験対象のCapability / Vendor Specificを指定した場合はID・Length・Valueも比較する。
6. 市販APのProbe Responseを捕捉できるかを追加試験とする。AP設定・対応レート・SSIDと試行回数を記録する。
   応答がないことだけではformat不正と断定しない。CSMA/CA未実装、static再送、scan条件等も分けて確認する。

## Probe Response（未実施）

1. SourceをProbe Responseに変更し`Apply Probe Response Default`。宛先は試験STAの実MACへ置き換える。
   初期値`02:11:22:33:44:66`は仮の試験用。Source/BSSID=`02:11:22:33:44:55`、SSID=`Pluto_Test_AP`、
   DS=現在channel、6 Mbps、Timestamp=0、Interval=100 TU、Capability=`0x0401`、FCS Autoを確認する。
2. Generate → VerifyでSubtype 5、固定パラメータ、IE、TIMなし、FCS Validを確認する。
3. 記録したRF条件で送信し、Wiresharkの`wlan.fc.type_subtype == 0x05`で宛先・BSSID・SSID・channel・
   Supported / Extended Rates・FCS保持時のGoodを比較する。
4. PC / Android scan list表示は追加試験。static responseだけで必ず表示されるとはしない。
   受信Probe Requestに同期して返答する機能、ACK処理、associationは実装していない。

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
| Probe Request field一致 / Wildcard・Specific / AP応答 | 未実施 |
| Probe Response field一致 / 宛先 / TIMなし | 未実施 |
| PC / Android SSID検出 / 試行回数 | 未実施 |
| RF波形 / EVM / spectral mask | 未実施 |

結果は同じ検証フォルダに追記し、captureをテストfixture化する場合は`tests/data/fixtures/wifi/`へ出典・条件を添える。
