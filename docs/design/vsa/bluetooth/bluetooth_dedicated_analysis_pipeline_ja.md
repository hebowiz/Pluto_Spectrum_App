# Bluetooth専用解析パイプライン補足

> 参照範囲: Bluetoothの汎用解析・表示処理の再利用、専用RF測定との境界、packet座標系を説明します。利用者向けの流れは [解析補足](../../../user-manual/Pluto_VSA_Analysis_Guide_JA.md)、資料の分担は [設計索引](../README.md) を参照してください。

## General VSAとの共通化と専用処理

Bluetooth専用解析は、PHY・packet境界・既知同期列から解析条件を決め、
`VSASession`、pattern解析、表示DSPを再利用します。一方、規格別RF測定は
専用の参照信号・評価区間・補正条件を持ち、汎用EVMをそのまま測定結果として
採用する構成ではありません。

```text
IQ → packet検出・境界決定
     +-- 汎用session / pattern解析 → FSK・PSK等の表示用結果
     +-- 専用RF測定 → EDR DEVM / HDT EVM等 → RF結果・集計
     +-- bit recovery → pluto_protocol → fieldの意味・payload表示
```

この図は責務の分担を示します。独立した3回の取得を行う意味ではなく、実際の
処理ではdecodeした長さやbit列を測定区間・参照信号へ渡します。

| 担当 | 現行の責務 |
| --- | --- |
| [Bluetooth model](../../../../pluto_vsa/protocol_modes/bluetooth/model.py) | packet検出、PHY判別、区間と参照データの準備、汎用解析・専用測定の呼出し |
| 汎用session / 表示DSP | pattern同期、FSK/PSK/QAMの表示用解析、共通Plot操作 |
| [rf_measurement](../../../../pluto_vsa/protocol_modes/bluetooth/rf_measurement/) | EDRの`measure_edr_devm()`、HDTの`build_hdt_evm_result()`等による規格別測定 |
| [summary](../../../../pluto_vsa/protocol_modes/bluetooth/summary.py) | 専用測定結果から集計・limit・必要データ量に応じた表示を構成 |
| `pluto_protocol` | 復調済みbit列のsemantic decode。IQ同期・EVM測定は担当しない |

HDTでは専用経路でHeader / Payloadの同期・参照生成・EVM評価を行います。
汎用表示側の同期条件を変えて専用RF測定値まで変える構成にしません。
General VSAのEVMと専用RF測定値を比較するときは、参照信号、フィルタ、補正、
評価区間が一致するかを確認します。

## FSK Measurement Filter

BR/LEのGFSKは送信信号自体にGaussian shapingが含まれる。専用解析で同じ
Gaussian特性をMeasurement Filterとして再適用すると、特に01交互列の
シンボル判定周波数偏移が過小になる。そのため専用Bluetooth FSK解析は
Measurement Filterを`None`とし、wide discriminator出力からシンボル判定を
行う。連続瞬時周波数トレースとConstellation Frequencyは同じ復調結果を
参照する。

EDR PSK部はPHYで規定したRoot Raised Cosine TX filterに対応するmatched
receive filterを`Auto`で適用する。

## EDR Vector / Symbol Plot

EDR部はdecoded packet境界に切り詰めた局所IQをGeneral VSA共通のPSK表示DSPへ
渡す。専用packetのシンボル数は表示負荷上十分小さいため、PSK Vectorの
サンプル間引きは行わない。pyqtgraphのauto downsamplingとclip-to-viewも
無効にし、filter通過後の全サンプル軌跡を描く。Symbol Plotは共通の正規化・
pi/4-DQPSK/8DPSK差動処理を使用する。

## EDR品質指標

汎用表示結果には`Bluetooth DEVM RMS`等の診断metadataが残っていますが、
専用RF Summaryは`rf_measurements`を参照し、RMS DEVM、99% DEVM、Peak DEVM等を
表示します。「Bluetooth DEVM RMSだけを表示する」という初期実装時の説明は
現在の専用RF Summaryには適用しません。集計block数、参照データの成立条件、
limit判定は専用測定とsummaryの責務であり、汎用EVMやDifferential Symbol EVMで
代用しません。

FSK部とPSK部の平均電力は、各部のdBm値を直接算術平均せず、いったんmWへ
戻して線形領域で平均した後にdBmへ変換する。Result Summaryには
`FSK Average Power`、`PSK Average Power`、および
`Relative Power (PSK - FSK)`を表示する。Relative Powerの正符号はPSK部が
FSK部より高いことを示す。

## 複数パケット解析と表示状態

初回のFSK同期探索で得た全候補のsample位置を保持し、2件目以降は候補近傍の
局所IQだけを解析する。全キャプチャをpacket indexごとに繰り返し探索しない。
局所IQ上のsample番号は`recording_sample_offset`と
`analysis_sample_offset`で元キャプチャの絶対sample番号へ戻し、Power、FSK
Modulation、Result Rangeを同じ時間軸に表示する。

左右キーによるpacket移動、およびDisplay Config変更に伴う再描画では、
Modulation/Symbol PlotのFSK/PSK tab選択を実行中は保持する。これは選択タブの再起動時復元を意味しない。選択packetを変更した際は
FSK ModulationのX rangeだけを選択packetへ追従させる。

10個の内部生成2-DH1を使った開発時回帰では、10件すべてのCRCを確認し、候補
局所解析化後の処理時間は約1.4秒（開発機上、UI描画を除く）だった。実機IQの
処理時間はcapture長、SNR、候補数、PC性能に依存する。

## Packet Analysis表示

Decode treeは省略記号を使用しない。Payload/Meaningはセル内で折り返し、
ウィンドウ幅に応じてValue/Meaning列へ余白を配分する。全値はtooltipでも
確認できる。

## 回帰確認

- 生成LE 1M IQ: 同期、Length、CRC、FSK filter mode
- 生成2-DH1 IQ: BR header、EDR Length/CRC、PSK packet境界、DEVM
- UI: 全PSK trajectory sampleの描画、Decode tree非省略表示
- 複数生成2-DH1: 全packet CRC、絶対sample offset、FSK/PSK個別電力
- UI: packet移動後のFSK表示範囲、FSK/PSK tab選択保持
- [HDT RF測定と汎用表示の独立性](../../../../tests/vsa/bluetooth/test_bluetooth_rf_measurement.py): `test_hdt_payload_phase_and_cfo_fit_is_independent_of_generic_display`
