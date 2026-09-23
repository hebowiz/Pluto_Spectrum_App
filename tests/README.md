# テスト

- `common/`: 共有SDR、取得ストリーム、Trigger、デバイス、共通UI・ユーティリティ。
- `rtsa/`: RTSA、Sweep、HSTA、スペクトラム処理。
- `calibration/`: 校正。
- `vsa/core/`: General VSA、パターン、表示、保存、セッション、UI。
- `vsa/bluetooth/`、`vsa/dect/`、`vsa/adsb/`: 各プロトコルの解析・専用VSA。
- `vsg/`: 波形生成、パケット構成、送信、VSG UI。
- [data/](data/README.md): IQ fixture、設定・出力・プロジェクト保存例。

リポジトリルートから `python -m pytest` で全件実行します。テストファイル名やテスト条件は配置変更で変えません。フォルダの運用ルールは [AGENTS.md](AGENTS.md) を参照してください。

## 分割したテストの入口

大きなテストファイルは、テスト関数・パラメータ・期待値を保ったまま責務別に分けています。分割先は同じサブシステムのフォルダに置き、fixture参照の深さを維持します。

| 対象 | テストモジュールと役割 |
| --- | --- |
| Bluetooth専用VSA | `vsa/bluetooth/test_vsa_bluetooth_*.py`: `hdt_analysis`（HDT解析）、`rf_analysis`（BR/LE RF解析）、`edr_analysis`（EDR解析）、`captures`（実測データ）、`hdt_ui`（HDT表示）、`workspace_ui`（画面操作・設定・連続実行） |
| General VSA UI | `vsa/core/test_vsa_*_ui.py`: `result`（結果表示・範囲選択）、`symbol`（シンボル・マーカー）、`persistence`（設定保存・復元）、`export`（出力）、`acquisition`（取得・非同期処理）。描画用関数は `test_vsa_display_helpers.py` |
| General VSAパターン解析 | `vsa/core/test_vsa_pattern_*.py`: `detected_data`（既知パターンを使わない同期）、`qam` / `fsk` / `psk`（変調方式別同期）、`ranges`（トリガー・結果範囲）、`session`（セッション・前処理再利用） |
| VSG | `vsg/test_vsg_*.py`: `model`（モデル・周波数・タイミング）、`generation`（波形生成）、`settings_dialogs`（設定画面）、`controls`（操作パネル）、`transmission_ui`（送信・校正操作）、`preview`（波形表示）、`project_io`（保存・出力） |

共通準備処理は同じフォルダの `_bluetooth_dedicated_test_helpers.py`、`_vsa_ui_test_helpers.py`、`_vsa_pattern_test_helpers.py` に置き、必要な関数だけを明示的にimportします。テスト関数を他のテストモジュールからimportして収集件数を増やさないでください。
