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

---

## 21. Pluto送信デジタルバックオフ（2026-08-27実装）

### 背景

従来のPluto送信バックエンドは、正規化IQの`|IQ| = 1.0`を
16 bit複素サンプルの`16383`へ変換していた。
これは符号付き16 bitの最大コード`32767`に対して、振幅で約-6.02 dBとなる。

GFSKのような定包絡波形では生成IQのpeakとactive RMSがともにほぼ1であるため、
この固定スケールがそのまま約6 dBの出力低下として現れる。

### 実装仕様

- DAC/DMA変換基準を`32767`（signed 16 bit full scale）へ変更する。
- Pluto Output設定に`Digital Backoff`を追加する。
- 選択肢は`0 dB (Full Scale)`、`-3 dB`、`-6 dB`とする。
- 初期値は`0 dB`とする。
- 設定は`pluto_tx/digital_backoff_db`として次回起動時も保持する。
- UI上の`TX Hardware Gain`表記は、AD936xの実態に合わせて
  `TX Attenuation`へ変更する（内部設定名は互換性維持のため当面そのまま）。

変換式は次のとおり。

```text
DAC code = clip(IQ component, -1, +1)
           * 32767
           * 10^(Digital Backoff [dB] / 20)
```

送信診断ログには、正規化IQ peak、DAC full scale、実際のDAC peak code、
Digital Backoff設定値を記録する。

### 期待値と注意点

`Digital Backoff = 0 dB`では、従来実装に対してRF出力が理論上約+6.02 dBとなる。
ただし`TX Attenuation = 0 dB`は「Plutoの最大送信設定」であって、
校正済みの0 dBmを意味しない。周波数、個体差、TX FIR、負荷、外部損失により
実RF電力は変化する。

Full Scale駆動ではDAC/アナログチェーンの非線形性やスペクトル再成長が
増える可能性がある。実機では`0/-3/-6 dB`を同一条件で比較し、
出力電力だけでなくEVM、イメージ、スプリアスも確認する。

この実装は相対デジタル駆動量の明示化であり、将来予定している
周波数別0 dBFS校正やRF Level（dBm）閉ループ設定の代替ではない。

---

## 22. 暫定dBm出力設定（2026-08-27実装）

### 目的と適用範囲

Pluto Outputダイアログの主設定を`TX Attenuation`から`RF Output Level [dBm]`へ
変更する。これは現時点でトレーサブルな絶対電力校正ではなく、次の条件で得た
実測値を使う暫定推定である。

- 周波数: 2440 MHz
- 波形: 定包絡FSKパケット
- 対象: 測定に使用したPluto個体
- 基準面: 実測時のPluto RF出力端

UIには暫定校正である旨を常時表示し、周波数特性、個体差、温度、変調波形の
PAPR、残留非線形性をまだ補正していないことを明示する。

### 暫定校正モデル

Backoff 0 dBで得た次の4点を、Tx Gainに対する区分線形校正曲線として用いる。

| Tx Gain | 実測FSK電力 |
|---:|---:|
| 0 dB | -0.2 dBm |
| -5 dB | -4.8 dBm |
| -10 dB | -9.4 dBm |
| -20 dB | -19.0 dBm |

Digital Backoffはこの曲線から独立したデジタル振幅差として加算する。
実測したBackoff -3/-6 dBの4点を含め、現モデルとの残差は最大0.1 dBである。
校正点の間は線形補間し、-20 dB未満は末端勾配による外挿とするため、低出力側は
特に未検証の推定値として扱う。

```text
Pout_est = calibration(Tx Gain) + Digital Backoff
Tx Gain  = inverse_calibration(Pout_target - Digital Backoff)
```

Backoffを変えても、Tx Gainの調整範囲内であれば希望RF Output Levelを維持する。
Backoffによって最大到達電力は下がり、暫定値では0/-3/-6 dB時にそれぞれ
約-0.2/-3.2/-6.2 dBmとなる。到達不能な組み合わせはUI範囲で制限し、
バックエンドでも送信前に検証する。

### 保存・互換性・診断

- 希望値は`pluto_tx/output_power_dbm`へdBmで保存する。
- 旧設定しかない場合は`hardware_gain_db`と`digital_backoff_db`からdBmへ移行する。
- 旧版との互換用に、逆算した`hardware_gain_db`も引き続き保存する。
- 送信診断には希望dBmと実際に適用したTx Gainの両方を記録する。
- 出力設定の変更だけではRF/baseband再設定やTX Quad Calibrationを要求しない。

将来はこの暫定関数を、Pluto serial、周波数、sample rate、RF bandwidth、温度、
変調方式、Digital Backoffを軸にした校正データへ差し替える。UIと送信状態機械は
そのまま維持し、校正層だけを交換できる構造とする。

---

## 23. Active RMS連動（2026-09-01 実装）

暫定dBm校正を、生成波形が明示する有効区間のRMSへ連動させた。Packet Period内のidle/zero IQは平均に含めない。各波形生成器は`active_ranges_samples`を出力し、バックエンドは送信直前にも同じ区間から実測値を再計算する。

```text
Pout_est = calibration(Tx Gain)
           + Digital Backoff
           + Active IQ RMS [dBFS]

Tx Gain = inverse_calibration(
              Pout_target
              - Digital Backoff
              - Active IQ RMS [dBFS]
          )
```

UIには`IQ Active RMS`、`IQ Peak`、`Crest Factor`、`Estimated Peak RF Level`を表示する。EDRはGFSK、Guard、DPSKのブロック別RMSも生成結果とInspectorへ表示する。診断ログには希望RF Level、適用Tx Gain、Active RMS、Peak、Crest Factor、推定平均/ピークRF Levelを記録する。

この補正はデジタル波形のRMS差を補うものであり、周波数応答、個体差、温度差、OFDMのPAPRに依存するアナログ圧縮は引き続き未校正である。
