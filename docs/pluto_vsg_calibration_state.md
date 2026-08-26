# Pluto VSG RF設定・校正・送信状態

更新日: 2026-08-26

## 目的

sample rate等の変更時に観測された約600 msの異常RF出力は、AD936x driverの自動TX
quadrature calibrationが生成する内部test toneと整合する。有限長non-cyclic DMA送信とは
別課題とし、RF/baseband設定・明示校正・送信を分離する。

## 状態遷移

`MUTED -> CONFIGURE -> CALIBRATING -> READY -> TRANSMITTING -> READY`

### Prepare / Calibrate ADALM-Pluto

1. TX LO powerdownとGain -89.75 dBを先に適用する。
2. AD936xの`calib_mode`を`manual_tx_quad`へ変更し、read-backする。
3. sample rate、TX LO、TX RF bandwidthを一括設定する。
4. `tx_quad`を一度だけ明示実行する。
5. `manual_tx_quad`へ戻し、全設定をread-backしてREADYへ移る。

Pluto firmware v0.39の実機では`calib_mode`のread-backが`manual_tx_quad 21`のように
付加数値を伴う。このため先頭tokenをmode名、残りを診断情報として扱い、完全一致では
判定しない。raw値は診断ログへ残す。

校正自体は内部toneをRF端子へ出す可能性がある。このためPrepareを独立操作として表示し、
Output Settings確定時にはその場で自動実行する。

### Transmit

Transmit経路はsample rate、TX LO、TX RF bandwidth、`calib_mode`へ書き込まない。
開始時に4項目をread-backし、Prepare済み値と違えばGain/LOをmuteしたまま送信を拒否する。
従ってTransmit操作が暗黙の再校正を発生させることはない。

Center、sample rate、RF bandwidth、接続個体が変わるとREADYを失効する。既に一度
Prepare済みのsessionでは変更直後に自動Prepareする。TX Gain、packet count、lead-in、
completion marginだけの変更は再校正条件に含めない。

## 運用と制約

先にRTSA等を起動してからVSGのOutput SettingsまたはPrepareを実行し、校正波が終了して
READY表示になった後でTransmitする。stock Plutoでは校正toneの物理的遮断をsoftwareだけで
保証できない。これを許容できない用途は、外部RF switchまたはAD9361 ENABLEを制御する
custom HDLの課題とする。

`pluto_vsg_tx_trace.log`にはPrepare/Transmitのoperation、state、calib_mode、library version、
各eventをJSON Linesで残す。

## 実機確認項目

- `calib_mode_available`と現在値のread-back
- `manual_tx_quad`中のsample rate変更で約600 ms波形が抑制されること
- 明示`tx_quad`実行時に同種の校正波が現れること
- Prepare後のEVM、LO leakage、image rejection、出力level
- READY後のTransmit操作では校正波が再発しないこと

## 2026-08-26 実機確認

Pluto firmware v0.39実機でPrepareを実行した。`calib_mode` read-backの付加数値に
対応した修正後はREADYまで完了し、RTSA上で校正由来と考えられるRF burstは1回だけ
観測された。従来のようにRF/baseband propertyの書込みごとに複数回発生する状態から、
最後の明示`tx_quad` 1回へ集約できた結果と整合する。

観測波形にはburst内でlevelが段階的に変化する区間があるが、これは1回のTX quadrature
calibration内部処理として扱う。次の確認ではREADY後にTransmitだけを実行し、この校正波が
再発せず、指定した有限packetだけが出力されることを確認する。
