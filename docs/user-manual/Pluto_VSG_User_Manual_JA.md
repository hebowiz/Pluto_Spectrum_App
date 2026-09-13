# Pluto VSG ユーザーマニュアル

文書版: 1.0
対象: Pluto VSG（IQ Waveform Generator / ADALM-Pluto TX）

## 1. はじめに

Pluto VSGは、規格別パケットを編集してIQ波形を生成し、ファイルへ保存、またはADALM-Plutoから送信するアプリケーションです。Bluetooth BR/EDR、Bluetooth LE、Bluetooth HDT、Wi-Fi、DECTのプロジェクトを作成できます。

## 2. RF安全と法令

> **警告:** RF ONは実際の送信を開始します。アンテナ接続時は使用地域の法令と免許条件を必ず確認してください。

- 初回確認は、シールド環境または十分なATTを入れたケーブル接続で行ってください。
- 受信計測器の最大入力を超えないよう、VSG出力と外部ATTを確認してください。
- RF OFFでもハードウェア状態やLO leakageを含む完全な理想遮断を前提にしないでください。
- 同じPlutoを別アプリのRX/TXで同時使用することはできません。

## 3. 起動

1. ADALM-PlutoをPCへ接続します。
2. `Pluto_VSG.bat`を起動します。
3. 必要なら`Inst Settings`で対象Plutoを選択します。
4. タイトルバーの`[TX: …xxxx]`を確認します。

起動直後は`RF OFF`、`Mod ON`、`Continuous ON`です。アプリ起動だけでは送信しません。

## 4. 画面構成

![Pluto VSG画面構成](../images/user-manual/pluto-vsg-overview.png)

1. **Block Library** — Fixed Data、Pattern、PRBS、Guard/Idle、Power Rampなどの構成要素。現行版では参照中心の項目があります。
2. **Packet Composer / Field Tree** — パケット、変調、電力制御の時間配置とフィールド構造。
3. **Inspector** — Standard、Center、Sample Rate、Packet Type、Periodなど現在値の一覧。設定編集と波形生成を実行できます。
4. **Generated IQ Preview** — IQ Waveform、IQ Power、Instantaneous Frequency、Spectrum、Constellationを切替表示。
5. **VSG Control** — RF、変調、連続送信、Power、Frequency、Pluto設定を操作。

画面下部のステータスバーには生成sample数、時間、sample rate、送信状態、エラーが表示されます。

## 5. クイックスタート

### 5.1 パケットを作る

1. `File > New`から規格を選択します。
2. Inspector下の`Edit ... Settings`を押します。
3. `RF / Timing`タブでPHY、sample rate、packet period、ramp等を設定します。
4. `Fields`タブでAccess Address、Payload、Header等を選択または入力します。
5. `Apply and Generate`または`Generate Waveform (F5)`を実行します。
6. PreviewとInspectorで内容を確認します。

### 5.2 Plutoから送信する

1. `Inst Settings`でConnection URIとDigital Backoffを確認します。
2. `Freq Settings`または`Frequency`で送信周波数を確認します。
3. `Power`で目標出力を設定します。
4. `Mod`と`Continuous`を目的に合わせます。
5. `RF`を押します。
6. Calibrationが必要と表示された場合は実行します。Calibration完了後も自動送信されないため、内容を再確認してからもう一度`RF`を押します。
7. Continuous送信は`RF`を再度押して停止します。

## 6. プロジェクト種類

| Newメニュー | 主な設定 |
|---|---|
| Bluetooth BR / EDR Project | DH系packet、BR GFSK、EDR 2M/3M DPSK |
| Bluetooth LE Packet | LE 1M/2M、Access Address、PDU、CRC等 |
| Bluetooth HDT Packet | HDT rate、Control Header、PDU |
| Wi-Fi Packet | 対応するPHY/field構成 |
| DECT Packet | Carrier Plan、Direction、Packet Type、Case A/B、A/B/X/Z field |

パケット種別を変更すると、その規格の既定Frequency Selectionが設定されます。`Freq Settings`を再度開いた場合は、Frequency数値から逆算せず、前回選択したCarrierとOffsetを表示します。

## 7. パケット設定ダイアログ

設定は原則として2タブに分かれています。

### 7.1 RF / Timing

- Packet Type / ModulationまたはPHY
- Samples / Symbolと自動計算されるSample Rate
- Repeat Count
- Deviation、Gaussian BT、SRRC roll-off等
- Pre Idle、Packet Period、Derived Post Idle
- Ramp Up/Downの時間、開始位置、Shape
- 規格固有のProlonged Preamble、Guard、Relative Power等

symbol単位の項目には、可能な場合us換算も併記されます。末尾のblank時間はPacket Periodから自動計算され、負になる設定は確定できません。

### 7.2 Fields

フィールド値は、定義済みの選択肢がある場合はCombo Boxから選択します。自由入力が必要なPayload、Address、Tail等はbit数または桁数を確認してください。

- Hex入力はフィールド幅に一致させます。
- 桁不足や範囲外の値がある場合、確定操作は拒否されます。
- DECT A-field Tailでは任意値に加え、`Test Burst Tx`プリセット（`0x70736E6363`）を選択できます。

数値欄は編集途中の一時的な範囲超過を許容しますが、不正値を残したままApply/OKはできません。赤色表示された欄を修正してください。

## 8. VSG Control

### 8.1 RF

- **RF OFF**: 送信停止状態。
- **RF ON操作、Mod ON、Continuous OFF**: ProjectのRepeat Count回を送信し、自動でOFFへ戻ります。
- **RF ON操作、Mod ON、Continuous ON**: 1 packet periodをcyclic DMAで、再度押すまで反復します。
- **Mod OFF**: Continuous設定に関係なくCWを連続送信します。

Stopは安全停止を優先し、Gain mute、LO powerdown、DMA buffer解放を行います。USB応答待ちで表示がStoppingになる場合は、ケーブルを抜かず完了を待ってください。

### 8.2 Mod

- **ON**: 生成したIQ波形で変調。
- **OFF**: 現在のFrequencyとPowerでCW送信。

CWはzero-IFのため、同じ中心周波数で受信すると受信側DC/LO leakageと重なることがあります。評価時は受信側でOffset LOを使用してください。

### 8.3 Continuous

- **ON**: Stopまで継続。
- **OFF**: 設定回数を有限送信。

ContinuousではProjectの先頭1周期だけを反復します。周期にはPre Idle、Ramp、Packet、Derived Post Idleが含まれます。

### 8.4 Power / Power Step

- `Power`を押すと目標RF Output LevelをdBmで入力できます。
- 上下矢印は`Power Step`分だけ増減します。
- 範囲外になるStep操作は適用されません。
- 送信中もPower変更が可能です。
- `Estimated Peak Power`は、Digital Backoff等を含む参考値です。

表示値はPluto個体差、周波数、温度、外部回路で変化します。精密なレベル設定には外部Power Meterまたは校正済み受信機を使用してください。

### 8.5 Frequency / Freq Settings

- `Frequency`: 任意周波数をMHz、小数点以下6桁まで入力します。送信中は変更できません。
- `Freq Settings`: 規格別Carrier Plan/ChannelとOffsetを選び、結果をFrequencyへ反映します。

Frequencyを直接編集しても、保存されたCarrier/Offset選択は上書きされません。

### 8.6 Inst Settings

- **Connection URI**: 使用するPluto。Refreshで再検索。
- **Digital Backoff**: IQ full scaleからのデジタル減衰。
- **LO Stabilization Wait (Muted)**: LOを有効にしてからGainを上げるまでの待ち時間。
- **Finite TX Lead-in (Zero IQ)**: 有限送信前にDMAへ先行配置する無信号時間。
- **Finite TX Minimum Hold**: DMA投入後、mute/cleanupまで送信状態を最低限保持する時間。

Frequency、Sample Rate、TX RF Bandwidth、Power、Playback Modeはメイン画面またはProjectから管理されるため、このダイアログには表示しません。TX RF BandwidthはSample Rateと同じ値へ自動設定されます。

## 9. Preview

| タブ | 内容 |
|---|---|
| IQ Waveform | I/Qの時間波形とfield境界 |
| IQ Power | dBFS電力包絡、ramp、idle |
| Instantaneous Frequency | FSK/FMの瞬時周波数 |
| Spectrum | 生成IQのベースバンドスペクトラム |
| Constellation | 変調区間ごとに分離したsymbol constellation |

時間軸の初期表示はActive Windowに約10%以下の余白を加えた範囲です。Packet後の長いIdleは初期表示から除外されますが、パンまたはズームアウトすると確認できます。

複数変調を含むEDR等では、Constellationを変調区間ごとに区別して表示します。GFSK区間とDPSK区間を同じsymbol判定として解釈しないでください。

ユーザーが変更したPlot範囲は再生成後も維持されます。右クリックの`Reset`で波形に基づく既定範囲へ戻ります。

## 10. DECT固有事項

- CarrierはCarrier Planと番号から選び、Offsetを追加できます。
- Prolonged Preambleはパケット長・Timing側で設定します。
- Modulation Case A/Bは規格テストパターンとfield値を連動させます。
- Ramp Up中の変調はETSI規定に従ってPreamble patternを延長します。
- Packet末尾も規格fieldとramp位置に従って生成されます。
- Packet Periodを設定するとPost Idleが自動決定されます。

## 11. 保存、読込、Export

| 操作 | 内容 |
|---|---|
| Save / Save As | `.pvsg.json` Projectを保存 |
| Open | Projectを読込 |
| Export NPZ | IQとメタデータをNumPy形式で保存 |
| Export R&S IQ TAR | R&S互換IQアーカイブを保存 |
| Export R&S WV | R&S waveform形式を保存 |
| Validate Project | field、timing、rangeの整合を確認 |

送信機のConnection URI、Power、Digital Backoff、Continuous等はローカル機器設定で、波形ProjectやExport IQへ含まれない項目があります。別PCで開く場合は送信設定を再確認してください。

## 12. Calibrationと送信状態

周波数、sample rate、bandwidth、接続Plutoなど、校正に影響する条件が変わるとPrepared状態は無効になります。RF ON時にCalibrationが必要なら確認ダイアログが表示されます。

```text
RF OFF
  -> Calibration確認
  -> Calibration実行
  -> READY（自動送信しない）
  -> ユーザーが再度RF ON
  -> TX
```

これにより、Calibration直後の意図しないRF送信を防止します。

## 13. トラブルシューティング

| 症状 | 確認事項 |
|---|---|
| RF ONできない | Calibration、Project validation、接続Pluto、Frequency/Power範囲を確認 |
| 設定欄が赤い | 空欄、範囲外、Hex桁数、Period不足を修正 |
| 送信周波数が違う | Frequency表示、Freq SettingsのCarrier/Offset、直接編集履歴を確認 |
| 出力が想定より低い | Power、Digital Backoff、Estimated Peak、外部ATTを確認 |
| Continuousが止まらない | RFボタンでStopし、Stopping完了を待つ |
| Stopが長い | USB/libiio応答、DMA cleanupを待つ。強制切断は避ける |
| Constellationが不自然 | 対象変調区間、PHY、Samples/Symbol、生成更新を確認 |
| Device busy | 同じPlutoを使用中のRTSA/VSA/VSGを停止または終了 |

## 14. 用語

- **Digital Backoff**: 量子化full scaleに対するIQ振幅の余裕。
- **Active Window**: RampとPacketを含む、送信波形として有効な時間範囲。
- **Packet Period**: 反復開始点から次周期開始点までの時間。
- **Derived Post Idle**: Packet/Ramp終了から周期末尾まで自動計算されるIdle。
- **Finite / Continuous**: 有限回のnon-cyclic DMA送信／1周期のcyclic DMA反復。
