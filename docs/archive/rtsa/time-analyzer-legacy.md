# 旧 Time Analyzer 仕様（UI非表示）

## 位置づけ

`AnalyzerMode.TIME_ANALYZER`としてコードに残っている旧実装です。現在のAnalyzer Modeメニューには表示されず、このモードへの直接または内部的な切替要求はHighSpeed TAへ転送されます。

このページは保守・整理時に既存コードの意図を確認できるよう、残存実装を記録するものです。ユーザー向けの現行機能ではありません。

## 動作

- 横軸: Time [s]
- 縦軸: Amplitude [dBm]
- Spectrum Only表示
- 固定長1000点の作業バッファを使用
- Time Span: 0.0001～10000秒
- GUIタイマーごとにFFTサイズ分のIQを共通`IQBlock` APIから同期取得
- 取得開始または設定変更後に5回分をwarm-upとして破棄
- Time Span到達時に完成した時間窓をまとめて描画
- Continuousでは0秒から次の時間窓を開始
- Singleでは1時間窓の完成後に停止

## 信号処理

各IQブロックについて次を実施します。

1. 必要に応じてDC平均値を除去
2. block境界をまたいで状態を保持するGaussian complex IQ FIRを適用
3. filter出力を`I² + Q²`へ変換
4. Sample/Peak/RMS Detectorで代表値を生成
5. Detector代表値へ固定補正、入力補正、周波数別補正を適用
6. 取得時刻を時間軸位置として作業バッファへ格納

RBW変更時のSample Rate、RF bandwidth、FFT size決定式はSweep SAおよびHighSpeed TAと共通です。

## HighSpeed TAとの違い

| 項目 | 旧Time Analyzer | HighSpeed TA |
|---|---|---|
| IQ取得 | GUIタイマーから同期取得 | 専用受信スレッド |
| 解析 | GUIスレッド | 専用解析スレッド |
| 横軸表示単位 | 秒 | ミリ秒 |
| UI選択 | 不可 | 可能 |
| gap統計 | なし | あり |

## 保守上の注意

旧Time Analyzer用の状態、更新処理、分岐が`RealtimeSpectrumWindow`内に残っています。削除または復活させる場合は、HighSpeed TAと共有しているRBW設定、時間軸マーカー、Sweep-like進行状態への影響を確認する必要があります。
