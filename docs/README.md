# Pluto Spectrum App 現行仕様

このディレクトリは、Pluto Spectrum App の現行実装、改善方針、検証状況を記録する引継ぎ資料です。モード仕様はコード上で現在動作する内容を、改善資料は未完了項目を含む作業状態を基準にしています。

- 現在の作業ブランチ: `feature/continuous-iq-stream`
- 基準コミット: `b33f42e`
- 最終更新日: 2026-08-02
- 対象ハードウェア: ADALM-Pluto（PlutoSDR）

## ドキュメント一覧

- [共通仕様](common.md)
- [RealTime SA](modes/realtime-sa.md)
- [WideBand RT SA](modes/wideband-rt-sa.md)
- [Sweep SA](modes/sweep-sa.md)
- [HighSpeed TA](modes/high-speed-ta.md)
- [Calibration](modes/calibration.md)
- [旧 Time Analyzer（UI非表示）](modes/time-analyzer-legacy.md)
- [IQストリーム改善計画・実装状況](iq-streaming.md)
- [リアルタイムSA・VSA準拠の計測アーキテクチャ](measurement-architecture.md)
- [R&S FPL-K70を参照したVSAアプリケーション設計方針](vsa-architecture.md)
- [RBW演算監査と改善方針](rbw-processing.md)
- [PlutoSDR実機検証記録](hardware-validation.md)

## モード一覧

| モード | 主な用途 | 横軸 | 取得方式 | UI表示 |
|---|---|---|---|---|
| RealTime SA | 最大55 MHz幅のリアルタイム観測 | 周波数 | 連続RXスレッド | あり |
| WideBand RT SA | 広帯域の分割取得・合成表示 | 周波数 | LO切替によるチャンク取得 | あり |
| Sweep SA | 周波数点ごとの掃引測定 | 周波数 | LO切替による点測定 | あり |
| HighSpeed TA | 電力の時間変化を高速観測 | 時間 | 共通IQ Producer＋専用解析スレッド | あり |
| Calibration | 周波数別振幅補正値の作成 | 周波数 | RealTime SA系のFFT測定 | あり |
| Time Analyzer | 電力の時間変化を通常経路で観測 | 時間 | GUIタイマーごとのブロック取得 | なし（HighSpeed TAへ転送） |

## 起動構成

起動点は `pluto_sa/main.py` です。

```powershell
python -m pluto_sa.main
```

起動時に次のコンポーネントを生成し、PlutoSDRの連続受信を開始してからウィンドウを表示します。

1. `SpectrumConfig`: 全モード共通の設定状態
2. `PlutoReceiver`: PlutoSDR接続、設定、IQ取得
3. `SpectrumProcessor`: FFT、RBW処理、表示帯域抽出
4. `SweepController`: Sweep SAの点測定と進行管理
5. `RealtimeSpectrumWindow`: UI、モード切替、描画、測定制御

PlutoSDRへの接続は `PlutoReceiver` の生成時に行われます。URIを明示しない場合はdirect USBを優先し、`SpectrumConfig.sdr_uri`または環境変数`PLUTO_SDR_URI`で上書きできます。現時点では、実機なしでGUIだけを起動するシミュレーションモードはありません。

## 更新方針

モードの挙動を変更した場合は、対応する仕様書と共通仕様を同じコミットで更新してください。将来仕様や改善案は現行仕様と混在させず、「検討事項」または別ドキュメントとして記録します。
