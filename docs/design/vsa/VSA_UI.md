# VSA 右側操作UI 設計

統合VSAの外枠、共通操作パネルとworkspaceの分担、設定Widgetの所有権を説明します。個々の設定値・操作例は [ユーザーマニュアル](../../user-manual/Pluto_VSA_User_Manual_JA.md)、配置の要件は [共通ウィンドウ仕様](../../spec/common/window-layout.md) を参照してください。

## 1. 目的

- General VSA / Bluetooth / DECT / ADS-B 1090ESに共通の右側操作パネルを提供する。
- 設定、取得、再解析、ファイル操作を分離し、設定変更だけで取得・解析を開始しない。
- 測定設定ファイルに解析モードを含め、異なるモードからRecallしても保存元モードへ設定を適用する。

## 2. 画面全体

```text
PlutoAnalysisWindow
  +-- workspace stack（アクティブな1モードを表示）
  |     +-- 各モードの6つの解析Dock
  +-- 右操作パネル（幅240 px）
        +-- ANALYZER SETUP
        +-- SWEEP CONTROL
        +-- SYSTEM: State / File / Device
```

メインウィンドウの初期指定サイズは1600×960、最小サイズは960×640です。終了時の位置・サイズを次回起動時に復元します。右側のMain Menuは必要に応じて縦スクロールできます。上部メニューバーは非表示にし、既存のショートカットはウィンドウActionとして接続します。

Dockは移動・タブ化・別窓化できますが、閉じる操作と表示ON/OFFは提供しません。モードを離れる際に配置・分割サイズ・フローティング状態をメモリーに保存し、実行中のモード復帰時に復元します。再起動時は初期配置に戻り、選択タブも永続保存しません。リサイズ時や同じモードの再選択で明示的な再均等化は行いません。

## 3. 共通操作パネル

### 3.1 所有関係

`PlutoAnalysisWindow`が共通の`VSAControlPanel`と、4つのworkspaceを所有します。Pluto接続は共有の`PlutoLiveSource`に集約します。

共通パネルへ各workspaceのUIを複製せず、外枠の`_panel_spec()`が`WorkspacePanelSpec`を作って渡します。specにはmode ID・表示名と、設定／Single／Continuous／Refresh／Reset／Fileの`PanelCommand`を格納します。各commandは表示名、callback、必要に応じて`QAction`を持ち、既存Actionの有効状態をボタンへ反映します。設定・Capture・Analysis・Exportの処理は各workspaceへ委譲します。

モード切替時はspecを差し替え、Main Menuへ戻ります。実装は [application_window.py](../../../pluto_vsa/ui/application_window.py) と [control_panel.py](../../../pluto_vsa/ui/control_panel.py) を参照してください。

### 3.2 外観とナビゲーション

共通の操作パネル部品を用い、通常ボタンは最小高さ50 pxを基準にします。下位ページは右下の`Back`または右クリックで1階層戻ります。Main MenuではBackを隠します。`Esc`は設定ダイアログのCancelに使い、パネルのBackには割り当てません。

### 3.3 階層

```text
Main Menu
  +-- Analyzer Mode → モード選択ページ
  +-- 各測定設定 → 指定ページを直接開くモーダルダイアログ
  +-- Continuous / Single / Refresh Analysis / Reset
  +-- State → Recall / Save / Preset
  +-- File → モード別のファイル操作
  +-- Device → 接続先ダイアログ
```

`SYSTEM`はMain Menu内のグループ名です。`System`という中間ページはありません。Presetは現在モードのDefaultを適用する確認ダイアログを開きます。統合VSAの通常操作はConfig Topを経由しません。

## 4. ANALYZER SETUP

各モードの先頭は`Analyzer Mode`です。その後に以下の設定を、この順序で並べます。

| モード | 設定ボタンの順序 |
| --- | --- |
| General VSA | Signal Description → Input / Frontend → Signal Capture → Trigger → Pattern Search → Result Range → Demodulation → Result Summary → Display |
| Bluetooth | Signal Description → Input / Frontend → Signal Capture → Trigger → Display |
| DECT | Signal Description → Input / Frontend → Signal Capture → Trigger → Display |
| ADS-B 1090ES | Signal Description → Input / Frontend → Signal Capture → Trigger |

BluetoothのProtocol / PHY / Access情報やDECT固有条件は`Signal Description`で扱います。旧`Bluetooth Analysis`、`DECT Analysis`、`ADS-B Analysis`を別の入口として案内しません。ADS-Bにも共通の4設定入口を用意し、Receiver LocationやDisplayという独立ボタンはこの一覧には置きません。

通常の実機入力はPlutoです。IQファイルは`File > Import IQ`、接続先選択は`Device`で扱います。周波数、Gain、External ATT/Gain、Analysis Channel等の測定条件を接続先ダイアログへ重複配置しません。General VSAのPrevious / Next Result RangeはResult Rangeページとショートカットから操作します。

## 5. Analyzer Mode

選択肢はGeneral VSA、Bluetooth、DECT、ADS-B 1090ESです。内部IDはそれぞれ`generic`、`bluetooth`、`dect`、`adsb1090`であり、表示名のGeneral VSAとは区別します。Wi-Fiは統合VSAの選択肢に登録されていません。

現在モードをcheckedで示します。Capture・Continuous・Analysis等でworkspaceがbusyの場合はモード変更を拒否します。切替だけではCapture / Analysisを開始しません。同じモードを選び直した場合はMain Menuへ戻り、配置は維持します。

## 6. SWEEP CONTROL

共通の表示順は`Continuous`、`Single`、`Refresh Analysis`、`Reset`です。

- Continuousは取得と解析を繰り返し、実行中は停止操作へ切り替わります。
- Singleは1回の取得・解析を開始します。
- Refresh Analysisは保持IQを再解析し、Plutoからの取得は開始しません。
- Resetは現在IQ・解析結果・履歴・プロットをクリアし、測定設定とDeviceを保持します。Plotの表示範囲だけを戻す右クリックResetとは別の操作です。

ボタンの有効状態・実行中表示はworkspaceのActionとbusy状態へ同期します。取得中は設定・Device・Preset・Recall・モード選択もロックします。各モードの停止・キャンセル処理の詳細は既存workspaceに委ね、共通パネルから別の取得状態機械を作りません。

## 7. SYSTEM

### 7.1 State

`Recall`、`Save`、`Preset`をこの順序で表示します。

Recall / Saveは共通の`.vsaconfig.json`を扱い、version 2には`analysis_mode`とそのモードの`settings`を保存します。旧General VSAのversion 1も読めます。各モードの起動時QSettingsとは別の外部ファイル操作です。

Recallは保存元モードへ切り替えて設定を適用します。設定適用に失敗した場合は元の設定とモードへ戻します。未知のmode/versionを部分的に適用しません。保存にはDevice URI、IQ、統計履歴、Plotのzoom/pan、Dock配置を含めず、Recall後にCapture / Analysisを自動実行しません。詳細は [ファイル操作仕様](../../spec/vsa/general/vsa-file-workflows.md) を参照してください。

Presetは各workspaceが持つ`_default_meas_config`を用い、現在モードのDefault適用前に確認を表示します。測定設定だけを初期化し、Device URI、現在IQ、解析履歴、Plotの手動レンジを変更しません。適用後もCapture / Analysisを自動実行しません。

### 7.2 Device

全モード共通のダイアログで、編集可能なConnection URI／検出済みデバイス一覧とRefresh Devicesを提供します。選択した接続先を共有Pluto sourceへ反映します。測定中の接続先変更は拒否します。

### 7.3 File

| モード | 操作の順序 |
| --- | --- |
| General VSA | Import IQ → Export IQ → Export Symbol Table |
| Bluetooth / DECT | Import IQ → Export IQ → Export VSG Project |
| ADS-B 1090ES | Import IQ → Export IQ → Export Packet List → Import OpenSky CSV → Update Database |

Import IQはファイルの表示・解析を行いますが、Pluto Captureは開始しません。出力に必要なIQ・結果がない場合は対応するExportを無効にします。フォルダ履歴は [共通仕様](../../spec/common/file-dialog-folders.md) に従ってファイル種別ごとに記憶します。Windowを閉じる操作は標準Closeボタンを使います。

## 8. 設定ダイアログ

共通の`HierarchicalMeasConfigDialog.open_page()`は指定された設定ページを直接開き、TopとBackを隠します。内部部品には`open_top()`も残っていますが、統合VSAの通常操作経路には使いません。

- 設定Widgetをパネルとダイアログへ二重登録しない。
- 既存のvalidationを用い、無効値を確定させない。
- Cancelでは未確定の編集値を破棄する。
- 設定確定時は依存項目やderived valueを更新し、Capture / Analysisを開始しない。保持IQを更新後の条件で解析する操作はRefresh Analysisとする。

### 設定Widgetの所有権

旧共通Config UI文書のうち、画面遷移の変更後も必要な注意点をここで管理する。

Config内の入力Widgetを非表示Toolbarや`QWidgetAction`へ重複登録しない。Widgetの所有先はConfigページに一本化する。

Qtでは、非表示Toolbarが保持する`QWidgetAction`へ登録したWidgetを別レイアウトへ移しても、Action側の可視状態に影響されて入力欄が非表示になることがある。このため、設定値を操作するWidgetとメイン画面上の操作Widgetを共用しない。


## 9. 既存操作の到達先

| 操作 | 現在の到達先 |
| --- | --- |
| 解析モード選択 | Analyzer Mode |
| 測定条件の編集 | 各設定ボタンから指定ページを直接開く |
| 取得・再解析・結果クリア | SWEEP CONTROL |
| 測定設定の読込・保存・初期化 | State > Recall / Save / Preset |
| IQ等の読込・出力 | File |
| Pluto接続先選択・再検出 | Device |
| Plot scale初期化 | 各Plotの右クリックReset |
| Previous / Next Result Range | General VSAのResult Rangeページとショートカット |
| 終了 | Window標準Close |

## 10. 検証の担当

- [設定経路テスト](../../../tests/vsa/core/test_vsa_setup_controls.py): 設定ボタンの順序、設定編集と取得・解析の分離。
- [外枠テスト](../../../tests/vsa/core/test_analysis_application.py): モード切替と共有操作。
- [レイアウトテスト](../../../tests/common/test_window_layout.py): geometry、モード別配置、別窓、再起動時の初期化。

本書の更新は既存の操作経路を記述するもので、解析処理や測定条件を変更しません。過去の移行計画・旧メニューはGit履歴、照合範囲は [検証記録](../../verification/vsa/README.md) を参照してください。
