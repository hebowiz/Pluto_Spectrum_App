# 起動時のウィンドウ・内部レイアウトの現状

調査対象: RTSA / VSA / VSG、コミット `1f5c57a` の実装。
次の設計統一に向けた現状整理であり、アプリの動作変更は含まない。

この文書は統一前の調査記録。統一後の現行仕様は
[ウィンドウ位置・サイズとペイン配置](../spec/common/window-layout.md)を参照。

数値はコードがQtに指定する論理ピクセル。タイトルバー等の外枠を含む実寸や、
画面上の最終座標を保証する値ではない。OS、表示倍率、利用可能な画面領域、
各部品の最小寸法によって実際の表示は変わる。

## 1. メインウィンドウ

| 項目 | RTSA | VSA | VSG |
|---|---|---|---|
| 通常の起動クラス | `SessionRealtimeSpectrumWindow` | `PlutoAnalysisWindow` | `PlutoVSGWindow(restore_startup_state=True)` |
| 保存情報がない場合の指定サイズ | 1600 × 960 | 1600 × 960 | 1600 × 960 |
| 出現位置の明示指定 | なし。OS / Qtに委ねる | なし。OS / Qtに委ねる | 初回はなし。保存情報があれば`restoreGeometry()`に委ねる |
| 再起動時の位置・サイズ復元 | なし | なし | あり。正常な終了処理で保存したgeometryを復元 |
| 最大化状態の明示的な扱い | 保存・復元コードなし | 保存・復元コードなし | 専用フラグはなく、geometryの保存・復元に委ねる |
| 明示的なメイン最小サイズ | 960 × 640 | 指定なし。内部レイアウトの制約はある | 指定なし。内部レイアウトの制約はある |
| ウィンドウのリサイズ | 可 | 可 | 可 |
| 画面サイズに合わせた初期サイズ計算 | なし | なし | なし |
| 独自のモニター選択・画面内補正 | なし | なし | なし。復元時の補正はQtに委ねる |
| 起動時の表示モード | 保存済みのAnalyzer Mode / Graph Viewを復元 | 毎回Generic。前回の解析モードは復元しない | 保存済みプロジェクトを復元。内部タブの選択状態は保存しない |

3アプリとも通常の入口では`window.show()`を呼ぶ。起動コードに固定座標への
`move()`、明示的な中央配置、`showMaximized()`はない。

RTSA / VSAは、測定設定を復元する機能があってもウィンドウgeometryの復元は行わない。
VSGはクラス単体のコンストラクタでは保存・復元が既定で無効だが、
通常のアプリ入口が`restore_startup_state=True`を渡す。

参照:
[RTSA基本画面](../../pluto_rtsa/ui/main_window.py) の定数とコンストラクタ、
[RTSAセッション画面](../../pluto_rtsa/ui/session_window.py) の`__init__`、
[VSA外枠](../../pluto_vsa/ui/application_window.py) の`__init__`、
[VSG入口](../../pluto_vsg/main.py)、
[VSG画面](../../pluto_vsg/ui/main_window.py) の`__init__` / `_save_startup_state` / `closeEvent`。

## 2. 内部ペインの構成・操作・保存

| 項目 | RTSA | VSA | VSG |
|---|---|---|---|
| 構成方式 | 通常の水平・垂直レイアウト | 共通外枠 + モード別`QMainWindow`のドック | 入れ子の`QSplitter` + タブ |
| 右操作パネル | 幅240固定 | 幅240固定 | 幅240固定 |
| 外側の構成 | 余白12、左右間隔12 | 余白0、左右間隔6 | ワークスペースと操作パネルも水平splitterで接続 |
| ペイン境界のドラッグ | なし | ドック境界で可能 | splitter境界で可能 |
| ペインの並べ替え・別ウィンドウ化 | なし | ドックの移動・フローティングが可能 | 通常のペインは不可 |
| ペインを閉じる操作 | なし。Graph Viewで表示を切り替える | 通常の統合VSAでは全モードのドックを閉じる操作を無効化 | 通常のペインに閉じる操作なし |
| ペイン位置・分割比率の再起動時復元 | なし | なし | なし |

共通の幅240は[共通操作パネル定義](../../pluto_common/control_panel.py)から参照する。
同じ幅でも周囲の余白・分割機構は異なる。

### RTSA

左側は「ステータス表示 → Waterfall → Spectrum」の縦配置、右側が操作パネル。
ステータス表示は高さ108固定。両グラフを表示する場合は伸縮係数1:1で高さを分ける。

`Graph View`はBoth / Waterfall Only / Spectrum Onlyの3種類で、保存対象。
片方だけを表示する場合は表示中のグラフが空いた領域を使う。
Sweep / Time Analyzer系のモードではSpectrum側だけを表示する制約がある。
前回のモードや表示選択によって起動時に見えるグラフは変わるが、
ユーザーがドラッグして決めた分割比率を復元する仕組みではない。

参照: [RTSA画面](../../pluto_rtsa/ui/main_window.py) の`_build_ui` /
`_apply_display_mode` / `_apply_analyzer_mode_ui_constraints`、
[保存データ定義](../../pluto_rtsa/config/session_state.py) の`RTSASessionState`。

### VSA

1つの外枠の中でGeneric / Bluetooth / DECT / ADS-Bを切り替える。
各モードの画面は起動時に生成され、`QStackedWidget`で表示を切り替える。
通常のモード切替では画面を作り直さないので、実行中の配置変更は基本的に残る。
ただしADS-Bはリサイズ時の均等化があるため、手動の分割比率が再調整される。
終了・再起動をまたぐドック配置の保存はない。

各モードの初期配置は次の3列×2段。

| モード・段 | 左 | 中央 | 右 |
|---|---|---|---|
| Generic 上 | IQ Power | Spectrum | Result Summary |
| Generic 下 | Modulation | Symbol Plot | Symbol Table |
| Bluetooth 上 | IQ Power | Spectrum | Result Summary |
| Bluetooth 下 | Modulation（内部タブあり） | Symbol Plot（内部タブあり） | Packet Analysis（内部タブあり） |
| DECT 上 | IQ Power | Spectrum | Result Summary |
| DECT 下 | GFSK Modulation | Symbol Plot | Packet Analysis（内部タブあり） |
| ADS-B 上 | IQ Power | Packet List | Detected Aircraft |
| ADS-B 下 | PPM波形 | Message Summary | Aircraft Details（内部タブあり） |

均等化の実装にも差がある。

| モード | 初期の列幅指定 | 初期の上下高さ指定 | モード画面のリサイズ時 |
|---|---|---|---|
| Generic | 500:500:500 | 各列400:400 | 独自の再均等化なし |
| Bluetooth | 500:500:500 | 各列450:450 | 独自の再均等化なし |
| DECT | 500:500:500 | 明示的な上下均等化なし | 独自の再均等化なし |
| ADS-B | 現在の画面幅÷3で均等化 | 各列400:400 | 再度均等化 |

これらの`resizeDocks()`値はQtへのサイズ配分要求で、固定寸法ではない。
初期均等化は`QTimer.singleShot(0, ...)`で遅延実行する。
ADS-B単体クラスには1400×850の指定があるが、通常のVSA起動では埋め込み画面となるため、
外枠の1600×960とは区別する必要がある。

参照:
[外枠](../../pluto_vsa/ui/application_window.py) の`__init__` / `set_analysis_mode`、
[Generic](../../pluto_vsa/ui/main_window.py) の`_build_results` / `_equalize_result_docks`、
[Bluetooth](../../pluto_vsa/protocol_modes/bluetooth/ui.py) の`_build_results` / `_equalize_docks`、
[DECT](../../pluto_vsa/protocol_modes/dect/ui.py) の`_build_results` / `_equalize_docks`、
[ADS-B](../../pluto_vsa/standards/adsb1090/ui.py) の`_build_ui` / `resizeEvent`、
[ドック共通処理](../../pluto_vsa/ui/measurement_chrome.py) の`make_measurement_dock`。

### VSG

右端の操作パネルを除いたワークスペースは、左領域:右領域を820:410で初期指定する。

| 領域 | 上段 | 下段 | 初期の上下配分 |
|---|---|---|---|
| 左領域 | Block Library + Packet Composer | Generated IQ Preview | 450:450 |
| 右領域 | Inspector | Packet Decode | 450:450 |

左上のBlock Library:Packet Composerには伸縮係数1:3を指定する。
これは厳密な初期幅比の指定ではなく、部品のsize hint等にも依存する。
Packet ComposerはVisual Composer / Field Tree、PreviewはIQ Waveform等のタブを持つ。

**メインウィンドウが復元されても、splitterの分割比率は復元されない。**
`startup/window_state`には`QMainWindow.saveState()`を保存するが、
この画面の中央ウィジェット内にある`QSplitter`の`saveState()`は呼んでいない。
タブの選択状態も保存対象に含まれていない。

参照: [VSG画面](../../pluto_vsg/ui/main_window.py) の`_build_workspace` /
`_save_startup_state`。

## 3. 設定用の別ダイアログ

起動時の結果ペインとは別物。通常は操作した時点で表示する。
これらの位置・サイズを次回アプリ起動まで保存するコードはない。

| アプリ | 主な設定ダイアログの指定サイズ |
|---|---|
| RTSA | 主に右パネル内のページ切替と小さな入力ダイアログ。入力ダイアログに共通の固定幅・高さ指定なし |
| VSA | Generic / Bluetooth / DECT / ADS-BのMeas Configはすべて820×620、親ウィンドウに対するモーダル表示 |
| VSG | Wi-Fi / Bluetooth BR・EDR / LE / HDT / DECT設定は860×760、Pluto出力設定660×360、Packet Fields 820×520、周波数設定は幅610・高さsizeHint |

VSAのMeas Configは編集用の一時ダイアログを作り、元ダイアログのサイズを渡す実装。
これもウィンドウgeometryの永続保存とは別の処理。

参照: [VSA共通設定ダイアログ](../../pluto_vsa/ui/measurement_config_dialog.py)、
[共通数値入力](../../pluto_common/numeric_input.py)、
[VSG設定群](../../pluto_vsg/ui/main_window.py)、
[DECT設定](../../pluto_vsg/ui/dect_settings.py)、
[Packet Fields](../../pluto_vsg/ui/packet_fields.py)、
[周波数設定](../../pluto_vsg/ui/frequency_settings.py)。

## 4. 保存領域

| アプリ | QSettingsのorganization / application | ウィンドウに関係する保存内容 |
|---|---|---|
| RTSA | `PlutoSpectrumApp` / `PlutoRTSA` | `session/state_json`にモード・Graph View等。geometryなし |
| VSA | `PlutoSA` / `PlutoVSA`（通常の統合起動） | モード別の測定設定等。外枠geometry・ドックstateなし |
| VSG | `PlutoSpectrumApp` / `PlutoVSG` | `startup/geometry`、`startup/window_state`、`startup/state`。splitter stateなし |

## 5. 次の設計で決める項目

1. 初回の位置・サイズ: 中央配置かOS任せか、1600×960固定指定か画面比率か。
2. 再起動時の復元範囲: 位置・サイズ・最大化・使用モニター。
3. 画面環境の変更: 小さい画面、表示倍率変更、モニター取り外し時の補正。
4. 内部レイアウトの自由度: ドック移動・フローティングを許可するか、分割幅の変更だけか。
5. 内部状態の保存範囲: 分割比率、タブ、VSAの最後のモード、モード別の配置。
6. 初期レイアウトへ戻す操作と測定Presetの関係。
7. VSA全モードの上下配分と、ADS-Bだけにあるリサイズ時の均等化の扱い。
8. QSettingsの命名・キー・既存設定の移行方法、設定ダイアログの共通方針。

## 調査範囲

起動入口、画面生成、保存・復元、終了処理をソースで確認した。
VSGについては一時INI設定とoffscreen表示で補助確認し、splitter操作の前後で
`QMainWindow.saveState()`が変化せず、再生成時に分割比率が維持されないことを確認した。
利用中の設定は変更していない。実デスクトップ上の出現座標、複数モニター、
DPI変更時の挙動については実画面での確認は行っていない。
