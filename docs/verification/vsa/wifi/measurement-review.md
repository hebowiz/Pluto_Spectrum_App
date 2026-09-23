# Wi-Fi RF / PHY Measurement 実装・照合報告

## 規格と照合範囲

Non-HT OFDM / ERP-OFDM、20 MHz、6/9/12/18/24/36/48/54 Mbpsを対象とする。
ユーザー提供の `references/standards/Wi-Fi/IEEE_Std_802.11-2024.pdf` 本文を照合した。
SHA-256: `652e169be943f13c567752583536bb62a376511b20187e0999d0a1a86a51e9b8`。
印刷頁3370–3374（§17.3.9）、3395（§18.4.7）、3642（§21.3.17.4.2）を確認。
漏洩の数式とEq.(17-28)はページ描画でも確認した。PDF原本は変更せずGit対象外とし、転載画像も成果物に含めない。

現行条項に基づくLimitと条件付き判定を実装した。旧版からの主要な差分は以下。

- 20 MHzのdefault maskは、±30 MHz以遠で `max(-40 dBr, -39 dBm/MHz)`。
- non-VHT STAの中心漏洩は `max(P−15, −20) dBm`。絶対値側の例外を省略してFAILにしない。
- ERPの継承は§18.4.7.1、maskは§18.4.7.3、CFOは§18.4.7.4、clockは§18.4.7.5。
- RCEはTable 17-20 / Eq.(17-28)。52 tone、20 PPDU以上、各16 DATA symbols以上、random data。
- Non-HT PPDUであってもVHT STAの漏洩は§21.3.17.4.2。波形形式だけからnon-VHT DUTとは判定しない。

これはIEEE 802.11 based measurementであり、認証試験器の代替や地域規制を含む適合証明ではない。

## Result別報告

参照版はすべてIEEE Std 802.11-2024。ERPのmodulation accuracyは§18.4.7.1から§17.3.9を継承する。
テスト名は主に[測定テスト](../../../../tests/vsa/wifi/test_wifi_measurements.py)の `test_` 以下。

| Measurement | IEEE Standard Reference | Algorithm | Unit | Limit | PASS / FAIL対応 | Required Observation Condition | Automated Test | Known Limitation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Packet Power | §17.3.9.2 / ERP §18.4.7.2 | active PPDUの線形平均、既存振幅補正 | dBm | 普遍Limitなし | Info | active PPDU範囲 | 既存 `power_uses_active_packet_and_recording_corrections` | 地域別出力上限は評価しない。絶対値は校正に依存 |
| Carrier Frequency Error | §17.3.9.5 / ERP §18.4.7.4 | STF coarse + LTF fine CFO、中心周波数からppm換算 | Hz / metadata ppm | 5 GHz ±20 ppm / ERP ±25 ppm | 条件成立時PASS/FAIL | L-LTF同期成立、受信周波数基準確認、既知band | `current_cfo_limits_apply_only_with_frequency_reference`、`gain_phase_cfo_multipath_correction_and_deliberate_distortion` | 受信基準未確認なら送受信誤差を分離できずNot Measured |
| Symbol Clock Frequency Error | §17.3.9.6 / ERP §18.4.7.5 | OFDM timing推定器は未実装。CFOを代用しない | ppm | ±20 / ±25 ppm | N/A / Not Measured | timing由来推定器と精度検証が必要 | `known_sample_clock_offset_is_explicitly_not_measured` | +100 ppm合成IQでも値を捏造しない。sample-clock追従なし |
| Relative Constellation Error | §17.3.9.7.4、Table 17-20、§17.3.9.8 Eq.(17-28) | 同期/CFO補正→FFT→LTF等化→pilot CPE→最近傍誤差。DATA 48 + pilot 4のpacket RMSを等重み平均 | dB | 6/9/12/18/24/36/48/54 Mbps: −5/−8/−10/−13/−16/−19/−22/−25 dB | 条件成立時PASS/FAIL | 同一capture・rateで20 PPDU以上、各16 DATA symbols以上、random data、受信精度・経路確認、52 toneを覆う帯域 | `current_rce_boundary_and_confirmation_gates`、`twenty_real_ppdus_qualify_without_fcs_and_refresh_does_not_accumulate`、`canonical_error_includes_pilots_but_not_lsig_and_uses_nominal_power` | 観測数不足はInsufficient Data、測定系未確認はNot Measured。capture間累積なし |
| EVM RMS | RCEと同じcanonical result | linear RMS ×100 | % | 独立Limitなし | Info | canonical値取得 | `all_rates_choose_current_limits_but_require_observation_conditions` | RCEの表現変換であり独立した規格判定ではない |
| Transmit Center Frequency Leakage | §17.3.9.7.2（non-VHT STA） | CFO補正後の2 LTFのDC平均energy / active 52 tone + DC合計。合計電力Pを送信基準面へ換算 | dB / metadata dBm | max(P−15, −20) dBmを相対Limitへ換算 | 条件成立時PASS/FAIL | non-VHT DUT、受信DC/応答/有線経路確認、全tone帯域。絶対値例外には校正済み振幅 | `current_leakage_absolute_exception_and_dut_gate`、`training_leakage_relative_to_total_power`、`waveform_leakage_without_reading_an_arbitrary_dc_fft_bin` | 未校正時は相対≤−15 dBをPASSにできるが超過はNot Measured。VHT STA/不明DUTへこのLimitを適用しない |
| Spectral Flatness | §17.3.9.7.3 | 等化前LTFのtone平均energyをinner平均で規格化、上下Limitまでの最小margin | dB margin / tone別dB | ±1…16は±4 dB、±17…26は−6/+4 dB | 条件成立時PASS/FAIL | LTF、受信応答/有線経路確認、全tone帯域 | `training_flatness_inner_edge_boundaries`、`flatness_decision_requires_characterized_path_and_full_active_band` | OTA fadingや受信filterを自動で除去しない |
| Transmit Spectrum Mask | §17.3.9.3、Fig.17-13 / ERP §18.4.7.3 | active PPDU Hann Welch、100 kHz ENBW、50% overlap、線形PSD平均、上限比較 | dBm/MHz / dB margin / Hz | ±9/11/20/30 MHzで0/−20/−28/−40 dBr。±30 MHz以遠は校正時のみ−39 dBm/MHzとの大きい方 | 20/40 MS/sはInsufficient Data | 規定100 kHz RBW / 30 kHz VBW、±30 MHz以遠までの取得・受信帯域 | `reference_mask_known_spectrum_violation_location_and_absolute_floor`、`mask_current_absolute_floor_is_not_the_old_revision_limit`、`equivalent_digital_mask_detects_added_out_of_band_tone` | Equivalent Digital Measurement。VBW/detector未再現、全域帯域不足、地域mask未対応。部分比較から規格PASS/FAILを出さない |

Eq.(17-28)の印刷式はpacketごとのRMSを加算してNfで割るため、packet長による重み付けをしない。
DATA constellationの公称平均電力P0=1、pilotは既知BPSK点を用いる。
DATA/Pilotを別々に振幅fitして誤差を消さず、L-SIGはRCE集計へ含めない。
共通gain・phase・CFO・短いmultipathをtrainingで補正し、DATAだけの歪みは誤差として残す。

漏洩のPはchannel-estimation区間の52 active tone + DC energyをFFT長64の二乗で割った線形平均電力。
IQRecordingのfull scaleと既存校正/input correctionに従ってdBmへ換算する。
相対Limitは `max(-15, -20-P)` dB。送信基準面への校正がなければ絶対値側の例外は使用しない。
規格の+2 dB/tone表現を別の判定として二重適用せず、明示された絶対電力式を採用した。
VHT STAはRF LO位置と312.5 kHz RBWを使う別手順が必要で、この実装ではNot Measured。

## 測定条件の設定と表示

Meas Config → Measurement Conditionsに4つの確認欄を設ける。
受信周波数基準、受信IQ/DC/phase-noise/flatness精度と有線経路、random test source、non-VHT DUTを独立に確認する。
初期値はすべてFalse。使用者の確認であり、自動校正や規格条件の自動認識ではない。
通常の設定と同様にOK/Cancel、起動時保存、State Save/Recallへ対応し、測定系・信号源変更時は見直す。
設定変更後は保持IQの結果を再判定し、再取得は不要。確認済み条件と不足理由はtooltip/metadataへ記録する。
未知band、未取得値、観測数不足、帯域不足はチェックを入れてもPASSにならない。

Continuousはcaptureをまたぐ統計を累積せず、Refreshも同じ録音を二重加算しない。
RCEの同rate集計値はpacket選択にかかわらずcaptureを表す。対象packetがない場合だけ選択packet値を参考表示する。
FCS失敗でもPHY測定条件が成立すればRF測定を行い、FCS PASSだけでは測定系の適格性を認めない。

6 dockの名称・配置を維持し、FlatnessはModulation内部、maskはSpectrum内部tabに表示する。
FFT振幅(dBm)とmask用PSD(dBm/MHz)を分離する。正負offsetのmask線はいずれも上限であり、最低PSD要求は設けない。
maskの最小marginと違反周波数は観測できた範囲の情報として保持する。

## Decode / Diagnosticの分離

| 分類 | Result | 判定 |
| --- | --- | --- |
| PHY Decode | PHY Format、Channel Bandwidth、Data Rate、Modulation、Coding Rate、L-SIG Rate/Length、DATA OFDM Symbol Count | Info、未取得値はN/A |
| PHY Decode | L-SIG Parity | Valid=PASS、Invalid=FAIL、未取得=Not Available |
| Packet / MAC Decode | Frame Type、Frame Subtype、PSDU Length | Info。PSDU未完了ではLength=N/A |
| Packet / MAC Decode | FCS | Valid=PASS、Invalid=FAIL、欠落=Not Available。RF判定と独立 |
| Diagnostics | Peak Power、L-SIG/DATA 48-tone EVM RMS、EVM Peak、Pilot Error RMS、STF/LTF correlation、coarse/fine CFO、CPE RMS、timing offset、detection metric、power calibration | Info、Limitなし。通常非表示 |
| Diagnostics | Residual CFO、Channel Estimate Quality | N/A / Not Measured |

ResultはID/kind/reference/conditions/limit/value/unit/status/canonical_idを保持する。
Displayの追加結果チェックで詳細Decode・Diagnosticsを表示する。

## 検証記録

測定テストは全8Rate、gain/phase/CFO/multipath、DATA歪み、pilot寄与、rate混在、短packet、20 PPDU集計、
CFOの正負境界、全RateのRCE境界、漏洩の相対/絶対境界、inner/edge flatness境界、mask既知PSDと帯域外toneの位置を含む。
受信系・random data・DUT種別・帯域・観測数の不足でPASSしないことを検証する。
maskの広帯域試験だけは80 MS/s合成IQを用いる。製品UIの取得範囲を拡張したものではない。
UIテストは6 dock、既存plot、設定Cancelの隔離、起動時復元、保持結果の再判定を確認する。

全体回帰テストは **1323件成功（414.01秒）**。
デバイスリース領域への書込権限を付けて実行し、前回のsandbox PermissionErrorも含め失敗なし。
Limit表示の記号修正後もWi-Fi関連106件が成功（10.14秒）。Windows GUIでLimit・判定表示とplotを確認した。

[Result Summary](../../assets/wifi/measurement-summary.png)と
[Mask / Flatness](../../assets/wifi/measurement-reference-plots.png)はWindows GUIの合成IQ検証画像。
PRBS-9の600-byte PSDUであり、正常MAC frameを構成したBeaconではないため、FCS FAILとRF測定値が共存する。
合成入力について受信基準・応答・non-VHT DUT条件を確認した設定で撮影し、RCEはpacket数不足を示す。

## 残る測定上の制限

- Symbol Clock推定器・追従と誤差検証は未実装。依頼の完成条件に従いNot Measuredとする。
- Maskは40 MS/s取得帯域で全域を覆えず、30 kHz VBW/detectorも未再現。Equivalent Digital Measurementとして扱う。
- VHT STAのRF LO漏洩手順は対象外。Non-HT形式という理由だけで従来の漏洩判定を適用しない。
- 実RF・独立測定器との突合せ、PlutoのDC/周波数/応答/位相雑音の校正精度検証は未実施。

マニュアル本文・画像・PDFは未編集。次回の明示的な改訂依頼時に反映する。
