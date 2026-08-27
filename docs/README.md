# Pluto Spectrum App 現行仕様

このディレクトリは、Pluto Spectrum App の現行実装、改善方針、検証状況を記録する引継ぎ資料です。モード仕様はコード上で現在動作する内容を、改善資料は未完了項目を含む作業状態を基準にしています。

- 現在の作業ブランチ: `main`
- 基準コミット: `701865e`
- 最終更新日: 2026-08-22
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
- [VSA Carrier周波数推定・補正仕様](vsa-carrier-synchronization.md)
- [VSA FSKシンボル同期監査](vsa-fsk-synchronization-audit.md)
- [RBW演算監査と改善方針](rbw-processing.md)
- [PlutoSDR実機検証記録](hardware-validation.md)
- [ADS-B 1090ES専用解析モード](adsb1090.md)

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

Windowsではリポジトリ直下のBATをダブルクリックして各アプリを起動できる。いずれも
リポジトリ直下をworking directoryとし、`.venv\Scripts\python.exe`を明示的に使用する。

| BAT | アプリ | Python entry point |
|---|---|---|
| `Pluto_RTSA.bat` | RTSA / Spectrum Analyzer | `python -m pluto_sa.main` |
| `Pluto_VSA.bat` | Generic VSA / ADS-B切替 | `python -m pluto_sa.vsa.main` |
| `Pluto_VSG.bat` | IQ Waveform Generator / Pluto TX | `python -m pluto_vsg` |

`.venv`が存在しない場合、またはアプリが非zero codeで終了した場合は、原因を確認できるよう
consoleを閉じずにメッセージを表示する。正常終了時はpauseしない。

起動時に次のコンポーネントを生成し、選択・復元されたAnalyzer Modeに対応する取得経路を開始してからウィンドウを表示します。

1. `SpectrumConfig`: 全モード共通の設定状態
2. `PlutoReceiver`: PlutoSDR接続、設定、IQ取得
3. `SpectrumProcessor`: FFT、RBW処理、表示帯域抽出
4. `SweepController`: Sweep SAの点測定と進行管理
5. `SessionRealtimeSpectrumWindow`: UI、モード切替、描画、測定制御、前回セッションの保存・復元、Preset

PlutoSDRへの接続は `PlutoReceiver` の生成時に行われます。URIを明示しない場合はdirect USBを優先し、`SpectrumConfig.sdr_uri`または環境変数`PLUTO_SDR_URI`で上書きできます。現時点では、実機なしでGUIだけを起動するシミュレーションモードはありません。

RTSAは前回使用したPlutoのselectorを`QSettings`へ保存します。次回起動時に同じPlutoが検出できれば、複数台接続されていても選択ダイアログを表示せず自動接続します。前回のPlutoが見つからず、接続中のPlutoが1台だけならその1台を自動選択して保存し、複数台なら選択ダイアログを表示します。シリアル番号を取得できる個体は`serial:<id>`を保存するため、USB URIが変化しても同じ個体を再選択できます。環境変数`PLUTO_SDR_URI`が指定されている場合は従来どおりその指定を優先します。

RTSA終了時には、Analyzer Mode、Frequency/Span、Amplitude/Input、RBW、FFT/Waterfall/Persistence、Sweep、High Speed TA/Triggerのユーザー設定に加え、Trace 1～4とMarker 1～4の設定を`QSettings`へ保存します。次回起動時は保存状態を復元して対応モードで取得を開始します。測定データ、TraceのAverage/Max Hold蓄積値、Waterfall履歴、Sweep進行位置などのruntimeデータは保存しません。Calibration中に終了した場合はCalibrationへ入る前の通常設定を保存し、次回起動時にその設定を基準としてCalibration固定profileを再適用します。

Main Menuの`TRIGGER / MARKER`下には`SYSTEM`フレームがあり、`System`ページから`Preset`または`Device`を実行できます。Presetは確認後、接続先Plutoのselectorを維持したまま、Analyzer Mode・各ユーザー設定・Trace・Marker・Display/Persistenceを現在のコードで定義された初期状態へ戻します。Deviceは接続中のPluto一覧を表示し、選択変更後に旧受信を停止、新Plutoの初期設定、現在Analyzer ModeのContinuous再始動を行います。新Plutoを初期化できなかった場合は旧selectorと旧Receiverへ戻して測定を再開します。SYSTEM配下も他の設定ページと同じstack/historyルールでBack操作します。SYSTEMフレーム追加に伴い、RTSAの固定ウィンドウ高さは従来より80 px拡大します。

## 更新方針

モードの挙動を変更した場合は、対応する仕様書と共通仕様を同じコミットで更新してください。将来仕様や改善案は現行仕様と混在させず、「検討事項」または別ドキュメントとして記録します。
