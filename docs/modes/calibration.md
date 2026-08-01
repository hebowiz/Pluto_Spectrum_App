# Calibration 仕様

## 目的

既知の基準信号レベルとPluto Spectrum Appの測定値を比較し、周波数別の振幅補正テーブルを作成します。

## モード移行

- Calibrationへ入る前の設定をコピーして保存します。
- Calibration中は固定測定プロファイルへ変更します。
- 完了またはReturn時に移行前のモードと設定を復元します。

## 固定測定プロファイル

| 項目 | 値 |
|---|---:|
| Span | 10 MHz |
| RBW | 1 MHz |
| FFT Size | 4096 |
| Reference Level | 20 dBm |
| Internal Gain | 30 dB |
| 信号位置 | Centerを測定周波数より2 MHz低く設定 |
| Peak検索幅 | 測定周波数 ±1 MHz |
| 取得回数 | 5 frames / point |
| 最低有効Peak | -40 dBm |

## 基準CSV

測定開始前にReference CSVの読込が必要です。

```csv
frequency_hz,reference_power_dbm
100000000,-6.4
200000000,-4.2
```

- 空行と`#`から始まるコメント行は無視します。
- 同一周波数が複数ある場合は最後の値を採用します。
- 周波数の昇順へ並べ替えます。

コードには100 MHz～5.9 GHzを100 MHz刻みとする既定シーケンスもありますが、現行UIの測定開始処理はReference CSVが未読込の場合に開始しません。

## 1点の校正フロー

1. Reference CSVの現在点を取得
2. 対象信号がDC位置へ重ならないようCenter Frequencyを調整
3. 5フレーム取得
4. 対象周波数±1 MHzの最大値を各フレームから取得
5. 固定Calibration Offsetと入力補正を加えた表示相当値へ変換
6. 5フレームの平均値を測定値とする
7. 平均値が-40 dBm未満ならエラーとして同一点を再試行可能にする
8. 次式で周波数別補正値を算出

```text
calibration_offset_db = reference_power_dbm - measured_power_dbm
```

## 結果CSV

全点完了後、保存ダイアログを表示します。既定ファイル名は`pluto_cal_YYYYMMDD_HHMMSS.csv`です。

```csv
# Calibration Result File
# Int Gain [dB]: 30
# RBW [Hz]: 1000000
# Ext ATT [dB]: 30.0
# Ext Gain [dB]: 0.0
# Date: YYYY-MM-DD HH:MM:SS
# Note: Pluto Spectrum App Calibration
frequency_hz,measured_power_dbm,reference_power_dbm,calibration_offset_db
```

保存成功後は結果を現在の周波数別補正テーブルとして読み込み、CalibrationをONにし、パスを`data/settings.json`へ保存します。

## 補正CSVの読込

補正CSVには次の列が必要です。

```csv
frequency_hz,calibration_offset_db
```

読込後は周波数間を線形補間します。最低周波数未満と最高周波数超過では最寄りの端点補正値を使用します。

## 制限・注意事項

- Reference CSVがない場合、現行UIから測定シーケンスを開始できません。
- 保存先パスを絶対パスで記録するため、プロジェクト移動後は起動時の自動読込に失敗する場合があります。
- 校正精度は基準信号源、外部ATT/Gain設定、PlutoSDR個体差、温度、配線条件に依存します。
- 各フレームの同期取得結果は`source=calibration`の共通`IQBlock`として発行されます。
