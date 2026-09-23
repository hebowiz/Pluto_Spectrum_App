# VSA 右側操作UI 設計

更新日: 2026-09-13  
対象: General VSA / Bluetooth Dedicated Analyzer / DECT Dedicated Analyzer / ADS-B 1090ES

## 1. 目的

- 4つのVSAモードへ、Pluto SA / VSGと同じ操作感の右側操作パネルを設ける。
- 測定条件、Sweep操作、ファイル操作を右側パネルから呼び出せるようにし、上部メニューバーを廃止する。
- 現在の解析処理、ユーザー操作時だけCaptureする原則、プロットのレンジ保持を変えない。
- 測定設定ファイルに解析モードを含め、どのモードからRecallしても保存元モードへ遷移できるようにする。

## 2. 画面全体

```text
+------------------------------------------------------+------------------+
|                                                      | Main Menu        |
|                                                      |                  |
|          現在の6つの解析サブウィンドウ              | ANALYZER SETUP   |
|          （配置・内容は維持）                        | [Analyzer Mode]  |
|                                                      | [mode settings]  |
|          現在より約15%狭い表示領域                   | ...              |
|                                                      |                  |
|                                                      | SWEEP CONTROL    |
|                                                      | [Continuous]     |
|                                                      | [Single]         |
|                                                      | [Refresh]        |
|                                                      | [Reset]          |
|                                                      |                  |
|                                                      | SYSTEM           |
|                                                      | [System]         |
+------------------------------------------------------+------------------+
| Status bar                                                               |
+---------------------------------------------------------------------------+
```

- 初期ウィンドウ幅は現状から広げない。右側パネルが既存幅の約15%を使用し、6つのサブウィンドウ側が自然に圧縮される構成とする。
- 右側パネル幅はPluto SAと同じ`240 px`を基準とする。
- 高DPIや低い画面でボタンが欠けないよう、メインページは必要時のみ縦スクロール可能にする。
- ステータスバーは処理中表示、エラー、完了通知に使うため残す。
- 6つのサブウィンドウは常時表示とし、Display設定から表示ON/OFF項目を削除する。Dockの移動・リサイズ、および各Plotの右クリックResetは維持する。
- 上部メニューバーは非表示にする。既存の`QAction`が持つショートカットは、到達不能な機能が生じないようウィンドウActionとして維持する。

## 3. 共通操作パネル

### 3.1 所有関係

`PlutoAnalysisWindow`が右側パネルを1つだけ所有する。

```text
PlutoAnalysisWindow
  +-- workspace stack
  |     +-- General VSA
  |     +-- Bluetooth
  |     +-- DECT
  |     +-- ADS-B 1090ES
  +-- VSAControlPanel
        +-- active workspace adapter
```

- 各Workspaceへ同じUIを複製しない。
- 共通パネルは表示とナビゲーションだけを担当する。
- 実際の設定ダイアログ、Capture、Analysis、Exportはactive workspaceの公開Action/adapterへ委譲する。
- モード切替時はadapterを差し替え、必ず`Main Menu`へ戻る。

想定する共通インターフェース:

```python
class VSAWorkspaceActions:
    mode_id: str
    mode_label: str

    def setup_items(self) -> list[PanelAction]: ...
    def run_single(self) -> None: ...
    def toggle_continuous(self) -> None: ...
    def refresh_analysis(self) -> None: ...
    def reset_statistics(self) -> None: ...
    def capture_state(self) -> CaptureState: ...
    def export_actions(self) -> list[PanelAction]: ...
    def collect_meas_config(self) -> dict: ...
    def apply_meas_config(self, settings: dict) -> None: ...
```

### 3.2 外観

- 背景色、GroupBox、ボタン、フォント、hover表示はPluto SAの右側パネルと同じスタイル定義を使う。
- 通常ボタンは最小高さ`50 px`、太字、SA相当の文字倍率を基準とする。
- ページタイトルは上部に表示する。
- 最下部右側に`Back`ボタンを置く。Topでは非表示とする。
- 右クリックは現在のページから1階層戻る。コンテキストメニューは表示しない。
- `Esc`は開いている設定ダイアログのCancelへ使用するため、パネルのBackには割り当てない。

### 3.3 階層

```text
Main Menu
  +-- Analyzer Mode
  +-- mode-specific setup dialogs
  +-- System
        +-- Preset
        +-- Device
        +-- Recall
        +-- Save
        +-- File
              +-- mode-specific file actions
```

- `Analyzer Mode`、`System`、`File`、`Preset`はパネル内の下位ページを開く。
- 各測定設定ボタンは、対応する設定ページをモーダルダイアログとして直接開く。
- 旧`Config Top Menu`と各ダイアログ内の`< Config Top`は廃止する。
- 設定ダイアログを閉じただけではCaptureも再Analysisも行わない。変更値は次の`Single`/`Continuous`に使い、現在IQの再解析は明示的な`Refresh Analysis`で行う。

## 4. Main Menu

### 4.1 General VSA

#### ANALYZER SETUP

1. `Analyzer Mode`
2. `Input / Frontend`
3. `Signal Description`
4. `Signal Capture`
5. `Trigger`
6. `Pattern Search`
7. `Result Range`
8. `Demodulation`
9. `Result Summary`
10. `Display`

`Input / Frontend`では次を変更する。

- `Input Source`を削除し、通常入力はPlutoに固定する。
- IQファイル入力は`System > File > Open IQ`へ集約する。
- ADALM-Plutoの接続先選択・Refresh等は`System > Device`へ移す。
- 周波数、Gain、External ATT/Gain、Analysis Channel等の測定条件は残す。

`Display`はDisplay Configページを直接開く。6つのDock表示ON/OFFは置かない。

既存の`Previous Result Range` / `Next Result Range`は削除せず、`Result Range`ページ内の操作と既存ショートカットへ移す。

### 4.2 Bluetooth Dedicated Analyzer

#### ANALYZER SETUP

1. `Analyzer Mode`
2. `Bluetooth Analysis`
3. `Input / Frontend`
4. `Signal Description`
5. `Trigger`
6. `Display`

- `Bluetooth Analysis`はProfile / Protocol / PHY / Access情報等、現在の専用設定を保持する。
- `Signal Description`はPHYから自動決定されるSymbol rate、Deviation、Measurement Filter等を確認できるようにする。自動値はread-onlyでよい。
- Pluto接続先操作は`System > Device`へ移す。

### 4.3 DECT Dedicated Analyzer

#### ANALYZER SETUP

1. `Analyzer Mode`
2. `DECT Analysis`
3. `Input / Frontend`
4. `Signal Description`
5. `Trigger`
6. `Display`

- `DECT Analysis`はCarrier plan、Direction、Packet/Modulation case等のDECT固有設定を保持する。
- `Signal Description`はSymbol rate、GFSK条件、Measurement reference等の設定・自動値を整理して表示する。
- Pluto接続先操作は`System > Device`へ移す。

### 4.4 ADS-B 1090ES

#### ANALYZER SETUP

1. `Analyzer Mode`
2. `ADS-B Analysis`
3. `Receiver Location`
4. `Display`

`ADS-B Analysis`には最低限、次を含める。

- Sample Rate
- Capture Time
- External Attenuation
- Internal Gain
- External Gain
- Preamble SNR Threshold

Pluto接続先操作は`System > Device`へ移す。

## 5. Analyzer Modeページ

全モードで同じページを使う。

1. `General VSA`
2. `Bluetooth`
3. `DECT`
4. `ADS-B 1090ES`

- 現在モードはchecked/active色で示す。
- 現在と同じモードを押した場合はMain Menuへ戻るだけとする。
- Capture、Continuous、Analysis workerが動作中の場合は、現行どおりモード変更を拒否して理由を表示する。
- モード変更成功後は新しいWorkspaceのMain Menuへ戻る。
- モード変更だけではCapture/Analysisを開始しない。

## 6. SWEEP CONTROL

全モードで同じ4ボタンを同じ順序で表示する。

1. `Continuous`
2. `Single`
3. `Refresh Analysis`
4. `Reset`

状態遷移:

| 状態 | Continuous | Single | Refresh Analysis | Reset |
|---|---|---|---|---|
| Idle | 実行可 | 実行可 | IQがあれば実行可 | 実行可 |
| Single Capture/Analysis中 | 無効 | 無効 | 無効 | 原則無効 |
| Continuous中 | 表示を`Stop`へ変更し実行可 | 無効 | 無効 | 無効 |
| Stop処理中 | `Stopping...`、無効 | 無効 | 無効 | 無効 |

- `Continuous`はCapture → Analysisをユーザーが停止するまで繰り返す。
- `Single`は1回だけCapture → Analysisする。
- `Refresh Analysis`は現在保持しているIQを再解析し、Captureは絶対に開始しない。IQがなければ無効にする。
- `Reset`は現行の`Reset All Packets Statistics`相当とする。Plot scaleのResetとは分離する。
- F5/F6/F7等の既存ショートカットは維持する。

## 7. SYSTEM

### 7.1 Systemページ

1. `Preset`
2. `Device`
3. `Recall`
4. `Save`
5. `File`

### 7.2 Preset

Presetは「現在モードの測定設定を既知の初期値へ戻す」機能とし、Device URI、現在IQ、解析履歴、Plotの手動レンジは変更しない。

各解析モードは`Default`を1つだけ持つ。

| モード | Preset |
|---|---|
| General VSA | `Default` |
| Bluetooth | `Default` |
| DECT | `Default` |
| ADS-B 1090ES | `Default` |

- Preset適用前に確認ダイアログを出す。
- 適用後もCapture/Analysisは自動実行しない。
- `Default`は各Workspaceが持つ現行の初期設定値を基準とする。
- Preset定義はUIコードへ散在させず、mode IDと設定payloadを持つデータとして管理する。
- 将来Presetを追加できるデータ構造にはしておくが、現時点では複数候補をUIへ表示しない。

### 7.3 Device

Deviceページは全モード共通で、共有`PlutoLiveSource`の接続先を操作する。

- Connection URI / 検出済みPluto一覧
- Refresh Devices
- 接続状態と短縮device identity
- 必要な共通Pluto receiver setting

周波数、Sample Rate、RF Bandwidth、Gain等の測定条件は各モードの`Input / Frontend`または`ADS-B Analysis`に残し、Deviceページへ重複配置しない。

### 7.4 Recall / Save

- `Recall`は全モードから共通Meas Configファイルを開く。
- `Save`はactive modeの全測定設定を保存する。
- Recallしたファイルの`analysis_mode`が現在モードと異なる場合、先にモードを切り替えてから対象Workspaceへ設定を適用する。
- Recall/SaveにはPlutoのdevice URI、IQデータ、統計履歴、Plotのzoom/pan状態を含めない。
- Recall後にCapture/Analysisを自動実行しない。

推奨ファイル形式:

```json
{
  "schema": "pluto-vsa-meas-config",
  "version": 2,
  "analysis_mode": "bluetooth",
  "settings": {
    "profile": "rf_phy_test",
    "protocol": "bluetooth.le",
    "phy": "LE 2M"
  }
}
```

mode IDは次に固定する。

- `generic`
- `bluetooth`
- `dect`
- `adsb1090`

互換性:

- 現行version 1のGeneric `.vsaconfig.json`は`analysis_mode = generic`として読み込む。
- Bluetooth / DECTのstartup QSettingsは既存schemaを当面読み込み、新形式へ移行後も旧値を破棄しない。
- ADS-Bの既存QSettingsも新しい外部Save/Recallとは独立に移行する。
- 未知のmode/versionは部分適用せず、エラー表示して現在設定を維持する。
- Recallは全項目を検証してから一括適用し、途中まで設定が変わる状態を作らない。

### 7.5 File

#### General VSA

1. `Open IQ`
2. `Export IQ`
3. `Export Symbol Table`

#### Bluetooth / DECT

1. `Open IQ`
2. `Export IQ`

#### ADS-B 1090ES

1. `Open IQ`
2. `Export IQ`
3. `Export Packet List`
4. `Import OpenSky CSV`
5. `Download / Update from OpenSky`

- 現在IQや結果が必要なExportは、対象データがない間disableする。
- `Open IQ`はファイルを読み込んで表示・解析してよいが、Pluto Captureを開始しない。
- ADS-BのOpenSky通信中は重複操作をdisableし、進捗はstatus barへ表示する。
- Windowを閉じる操作は標準のCloseボタンへ集約し、Fileページには置かない。

## 8. 設定ダイアログ

現行`HierarchicalMeasConfigDialog`はTop Menuを内包しているため、次の責務へ変更する。

```text
旧: Config Top + 複数設定ページ + < Config Top
新: 指定された設定ページ + OK/Apply/Cancel
```

- 右側パネルのボタンが設定ページ名を指定して直接開く。
- 同一の設定Widgetをパネルとダイアログへ二重登録しない。
- 数値入力は既存の共通validationを使い、編集中の一時的な範囲外を許容しつつ、無効値のままOK/Applyできない仕様を維持する。
- Cancel時は編集前の値へ戻す。
- 設定確定時は依存項目の表示とderived valueだけ更新し、Capture/Analysisは開始しない。

## 9. 既存機能の移設表

| 現行機能 | 新しい到達先 |
|---|---|
| Analysis Mode menu | Analyzer Setup > Analyzer Mode |
| Meas Config Top | 廃止。各設定ボタンから直接開く |
| Run Single | Sweep Control > Single |
| Run Continuous / Stop | Sweep Control > Continuous / Stop |
| Refresh Analysis | Sweep Control > Refresh Analysis |
| Reset All Packets Statistics | Sweep Control > Reset |
| Load Meas Config | System > Recall |
| Save Meas Config As | System > Save |
| Open / Export | System > File |
| Pluto device selector / Refresh | System > Device |
| Dock show/hide | 廃止。6 Dockを常時表示 |
| Reset Plot Scales | 各Plotの右クリックReset |
| Previous / Next Result Range | Result Rangeページ + 既存shortcut |
| Close | Window標準Close |

## 10. 実装順序

1. 共通`VSAControlPanel`とworkspace adapterを追加する。
2. `PlutoAnalysisWindow`をworkspace stack + control panelの横並び構成へ変更する。
3. General VSAを接続し、Sweep状態と設定ページ遷移を検証する。
4. Bluetooth / DECTを接続し、専用設定と統計Resetを検証する。
5. ADS-Bを接続し、OpenSky/File操作を検証する。
6. Config version 2、全モードSave/Recall、旧Generic config互換を追加する。
7. Presetを追加する。
8. メニューバーを非表示にし、全既存機能への到達性とshortcutを監査する。
9. 1600x960、125%/150% DPIでレイアウトを確認する。

## 11. Acceptance

1. 全4モードで同じ外観・位置・Back操作の右側パネルを使用する。
2. 初期ウィンドウ幅を増やさず、既存6 Dock領域が約15%圧縮される。
3. 上部メニューバーとConfig Top Menuが表示されない。
4. 右クリックまたは`Back`で1階層だけ戻る。
5. Analyzer Mode変更後はMain Menuへ戻る。
6. Single / Continuous / Stop / Refresh / Resetの表示と実状態が一致する。
7. 設定ダイアログを閉じても自動Capture/Analysisしない。
8. Refresh Analysisは保持IQだけを解析し、Captureしない。
9. 6つのサブウィンドウは常に表示される。
10. Meas Configを全モードでSave/Recallでき、異なるモードのConfigで自動遷移する。
11. Recall後も自動Capture/Analysisしない。
12. 旧Generic version 1 configを読み込める。
13. メニューバー廃止後も、明示的に廃止したDock ON/OFF以外の既存機能へ到達できる。
14. Device選択は4モードで同じ共有Pluto接続へ反映される。
15. 解析結果、Capture条件、Plotレンジ保持、Decode/RF measurementに回帰がない。

## 12. 確定事項

- PresetはGeneral VSA / Bluetooth / DECT / ADS-B 1090ESごとに`Default`を1つだけ持つ。
- `Default`はactive modeの測定設定だけを初期化し、Device URI、現在IQ、解析履歴、Plotの手動レンジには影響させない。
