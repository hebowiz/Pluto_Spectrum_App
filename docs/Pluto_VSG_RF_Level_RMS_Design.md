# Pluto VSG 振幅設定仕様メモ
## R&S VSGの考え方に合わせたRF Level定義

## 1. 目的

PlutoSDRをVSG（Vector Signal Generator）として使用する際の振幅設定について、
R&S SMCV100Bなどの一般的なVSGに近い考え方を採用する。

Pluto/AD936xはハードウェアとしてはDAC Full Scale（0 dBFS）を基準に扱うのが自然だが、
VSGアプリケーションのユーザーインターフェースでは、**RF LevelをIQ波形の平均電力（RMS Power）として定義する**。

これにより、EDRのように複数の変調ブロックが共存し、
各ブロック間に電力差がある信号でも、R&S VSGとの比較を行いやすくする。

---

## 2. 基本方針

### Pluto/SDRとして自然な考え方

PlutoではIQデータの最大振幅をDAC Full Scaleに対応させる。

```text
|IQ| = 1.0
→ 0 dBFS
→ DAC Full Scale
```

この場合、IQ波形の平均振幅が小さいほど、平均RF出力電力も小さくなる。

つまり、

```text
0 dBFS出力電力 = RF出力の最大基準
```

という考え方になる。

これはSDR/DACとしては自然だが、VSGのRF Level設定としては扱いにくい。

---

## 3. VSGとして採用する考え方

Pluto VSGでは、R&S VSGに近い考え方として、

```text
RF Level = IQ信号の平均RF電力
```

と定義する。

例：

```text
RF Level = -10 dBm
```

と設定した場合、

```text
対象IQ信号の平均RF電力 = -10 dBm
```

となるようにPlutoのTx attenuationを調整する。

IQ波形のPeak値やCrest Factorが変化しても、
平均RF Levelは設定値に維持する。

---

## 4. IQ振幅の定義

IQ複素振幅を

```text
x[n] = I[n] + jQ[n]
```

とする。

瞬時振幅：

```text
A[n] = sqrt(I[n]^2 + Q[n]^2)
```

Peak振幅：

```text
A_peak = max(A[n])
```

RMS振幅：

```text
A_rms = sqrt(mean(I[n]^2 + Q[n]^2))
```

---

## 5. dBFS換算

Peak dBFS：

```text
Peak_dBFS = 20 * log10(A_peak)
```

RMS dBFS：

```text
RMS_dBFS = 20 * log10(A_rms)
```

Crest Factor：

```text
Crest_Factor_dB = Peak_dBFS - RMS_dBFS
```

通常、

```text
A_peak <= 1.0
```

となるようにIQを正規化する。

---

## 6. RF出力電力との関係

周波数ごとのPlutoの0 dBFS RF出力校正値を

```text
P_FS(f) [dBm]
```

とする。

Tx attenuationを

```text
ATT_TX [dB]
```

とすると、IQ信号の平均RF出力は概念的に

```text
P_RF_avg
= P_FS(f)
+ RMS_dBFS
- ATT_TX
```

となる。

ユーザーが指定したRF Levelを

```text
P_SET [dBm]
```

とした場合、

```text
P_RF_avg = P_SET
```

となるように、

```text
ATT_TX
= P_FS(f)
+ RMS_dBFS
- P_SET
```

を設定する。

---

## 7. 推奨実装

### IQ側

IQデータは可能な限りFull Scale付近まで使用する。

```text
max(sqrt(I^2 + Q^2)) ≈ 1.0
```

ただしクリッピングは禁止する。

IQをRF Level設定のために毎回縮小するのではなく、
基本的にはIQ波形の相対振幅関係を維持する。

### RF Level制御側

RF Levelの調整は主にPluto / AD936xのTx attenuationで行う。

つまり、

```text
IQデータ
    ↓
PeakがFull Scale付近になるよう正規化
    ↓
RMS_dBFSを計算
    ↓
希望RF Levelとの差分をTx attenuationへ反映
```

という構成とする。

---

## 8. EDRパケットでの考え方

Bluetooth EDRでは1パケット内に、

```text
GFSK block
Guard
DPSK block
```

が存在し、GFSKとDPSKの間に電力差を持たせる場合がある。

例：

```text
GFSK : 0 dB relative
DPSK : +3 dB relative
```

とする。

GFSKとDPSKの時間が同じ場合、相対電力は

```text
GFSK = 1
DPSK = 2
```

なので全体平均は

```text
(1 + 2) / 2 = 1.5
```

となる。

GFSK基準では

```text
10 * log10(1.5)
≈ +1.76 dB
```

である。

RF Levelを0 dBmに設定した場合、

```text
GFSK ≈ -1.76 dBm
DPSK ≈ +1.24 dBm
Packet Active Average = 0 dBm
```

となる。

重要なのは、

```text
最大振幅ブロック = RF Level
```

ではなく、

```text
信号全体の平均電力 = RF Level
```

とすることである。

---

## 9. Pluto従来方式との差

### Peak / Full Scale基準

例えばPSK側を0 dBFSとし、

```text
PSK = 0 dBm
GFSK = -3 dBm
```

とした場合、

全体平均は約

```text
-1.24 dBm
```

となる。

したがって、

```text
RF Level = 0 dBm
```

という設定値と実際の平均RF電力が一致しない。

EDRなど、振幅差を持つ複合波形ではこの差が特に目立つ。

---

## 10. 平均電力を計算する区間

ここは重要な仕様項目とする。

パケット波形が、

```text
Ramp Up
GFSK
Guard
DPSK
Ramp Down
Packet Interval
```

という構成の場合、

Packet Intervalまで含めてIQ配列全体のRMSを計算すると、
パケットDutyによってRF Levelが変化してしまう。

例えば同じBluetoothパケットでも、

```text
Packet Interval = 1 ms
```

と

```text
Packet Interval = 10 ms
```

でRMSが変化する。

これは通常のVSGとして扱いにくい。

そのためPluto VSGでは原則として、

```text
RF Level = Active Signal RMS
```

とする。

つまり、

```text
Packet Interval
Idle
Zero IQ区間
```

はRF Level計算対象から除外する。

---

## 11. Active Signalの定義

実装上は、波形ジェネレータ側でActive区間を明示的に保持することを推奨する。

例：

```python
active_start_sample
active_end_sample
```

または、

```python
active_mask[n]
```

を持たせる。

単純な振幅ThresholdによるActive判定は、
Ramp Up / Ramp Downや低レベル信号で誤判定する可能性があるため、
波形生成時のメタデータを利用する方が望ましい。

---

## 12. Bluetooth EDRのブロック別Power

EDR波形については、全体平均とは別にブロック単位でも電力を計算可能とする。

例：

```text
Active RF Level : -10.00 dBm
GFSK Power      : -11.76 dBm
DPSK Power      :  -8.76 dBm
Delta Power     :  +3.00 dB
```

これにより、

```text
DPSK Power - GFSK Power
```

が設定した相対電力差と一致していることを容易に確認できる。

---

## 13. UI表示候補

VSGのAmplitude関連表示として以下を検討する。

```text
RF Level          -10.00 dBm
IQ RMS             -3.42 dBFS
IQ Peak            -0.50 dBFS
Crest Factor        2.92 dB
Peak RF Level       -7.08 dBm
```

Bluetooth EDRの場合は追加で、

```text
GFSK Power         -11.20 dBm
DPSK Power          -8.20 dBm
Delta Power         +3.00 dB
```

などを表示できると、R&S VSG / VSAとの比較に有用。

---

## 14. Peak RF Level

平均RF LevelとCrest Factorから、

```text
Peak_RF_Level
= RF_Level
+ Crest_Factor
```

として概算できる。

例：

```text
RF Level       = -10 dBm
Crest Factor   = +4 dB
```

なら、

```text
Peak RF Level ≈ -6 dBm
```

となる。

ただし実際のPluto RF出力では、
DAC、Tx chain、RF amplifier等の非線形性や飽和により
理想値と一致しない場合がある。

Peak側に十分なHeadroomが存在することを確認する必要がある。

---

## 15. Calibrationとの関係

Plutoの実際の0 dBFS出力電力は、

- RF frequency
- Tx channel
- AD936x gain / attenuation
- 個体差
- 温度
- RF回路

などに依存する。

そのため、

```text
P_FS(f)
```

は固定値ではなく、周波数依存のCalibration Tableとして保持することを推奨する。

例：

```text
Frequency      0 dBFS RF Power
--------------------------------
2402 MHz       +6.7 dBm
2441 MHz       +6.9 dBm
2480 MHz       +6.5 dBm
```

必要に応じて補間して使用する。

---

## 16. Tx Attenuationの制約

AD936xのTx attenuationには設定範囲および分解能がある。

希望RF LevelがTx attenuationだけでは実現できない場合、

1. IQ amplitude scaling
2. Tx attenuation
3. 外部ATT / AMP

を組み合わせる必要がある。

ただし通常動作範囲では、

```text
IQ amplitude ≈ Full Scale
```

を維持し、

```text
Tx attenuation
```

を優先的に使用する。

---

## 17. 推奨内部パラメータ

少なくとも以下を保持する。

```text
rf_level_dbm
iq_peak_dbfs
iq_rms_dbfs
crest_factor_db
active_rms_dbfs

calibrated_fullscale_power_dbm
tx_attenuation_db
```

Bluetooth EDR波形の場合は追加で、

```text
gfsk_rms_dbfs
dpsk_rms_dbfs

gfsk_power_dbm
dpsk_power_dbm
delta_power_db
```

を保持するとよい。

---

## 18. 設計上の分離

Pluto VSGでは、以下を明確に分離する。

### Waveform Level

IQデータ内部の相対振幅。

例：

```text
GFSK : 0 dB
DPSK : +3 dB
```

### Generator RF Level

VSGとしての絶対RF出力。

例：

```text
RF Level : -10 dBm
```

この2つを混在させない。

Waveform Levelは波形の特性であり、
RF LevelはVSG出力設定である。

---

## 19. 最終方針

Pluto VSGでは以下を基本仕様とする。

```text
1. IQ Peakは原則としてFull Scale付近へ正規化する

2. IQのActive Signal RMSを計算する

3. RF LevelはActive Signalの平均RF電力として定義する

4. RF Levelの調整は主にTx attenuationで行う

5. Packet Interval / Idle区間はRMS計算から除外する

6. EDRなど複数ブロックを持つ波形では、
   各ブロックの相対電力差をIQ内部で維持する

7. Peak / RMS / Crest Factorを内部で管理する

8. RF周波数ごとの0 dBFS出力校正値を使用する

9. 必要に応じてブロック別RF Powerも表示する
```

---

## 20. 設計思想

Pluto内部では、

```text
0 dBFS
```

をDAC / RFチェーンの物理的な基準として使用する。

一方、ユーザーに対するVSGインターフェースでは、

```text
RF Level = Active Signal RMS Power
```

を使用する。

つまり、

```text
SDR内部
    0 dBFS基準
        ↓
IQ RMS / Calibration
        ↓
VSG UI
    Average RF Power基準
```

という二層構造とする。

この構成により、
Plutoのハードウェア特性を維持しつつ、
R&S SMCV100B等の計測器に近い操作感・出力定義を実現する。
