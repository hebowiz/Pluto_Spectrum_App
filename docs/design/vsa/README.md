# VSA設計文書の役割と参照先

主題ごとの担当文書を以下に示します。UIの操作経路、配置、入力source、16QAM、BluetoothのRF測定境界、連続取得について本文を現行実装へ合わせています。拡張案・初期ロードマップは該当節で区別し、全項目を実装済みとは扱いません。確認した根拠と改訂内容は [文書・実装の照合記録](../../verification/vsa/README.md) を参照してください。

## 参照する順序

1. 動作の要件は該当する `docs/spec/` を参照します。文書同士に矛盾がある場合、古い構想で現行仕様を上書きしません。
2. 操作名・操作順は [VSAユーザーマニュアル](../../user-manual/Pluto_VSA_User_Manual_JA.md)、設計理由は下表の担当文書を参照します。
3. 実装状況の確認には該当コードと回帰テストを使います。仕様と実装の食い違いは記録し、どちらかを無断で変更して解消しません。
4. 作業指示・監査・旧メニュー一覧は経緯を調べる資料です。日付が新しいだけで現行仕様の根拠にはしません。

## 主題ごとの担当文書

| 主題 | 担当文書 | 他の文書に重複して定義しない範囲・注意 |
| --- | --- | --- |
| VSA全体のsession・record・解析段階 | [vsa-architecture.md](vsa-architecture.md) | 共通概念と拡張方針。画面の最新メニュー、対応機能一覧、実装完了状況の正本にはしない |
| 統合VSAの外枠・右操作パネルとworkspaceの分担、設定Widget所有権 | [VSA_UI.md](VSA_UI.md) | 共通パネルは§3、設定ダイアログとWidget所有権は§8。現行の操作経路を記述し、個別設定値はマニュアルへ委ねる |
| General VSAのCFO・位相・ドリフト・タイミング補正 | [vsa-carrier-synchronization.md](vsa-carrier-synchronization.md) | 汎用同期の設計。Bluetooth規格別RF測定の参照信号・評価区間まで一般化しない |
| Acquisition Trigger・Burst Search・Pattern Searchの区別 | [vsa-iq-power-trigger.md](vsa-iq-power-trigger.md) | 検出と検索の責務。RX producerの生存期間・再アーム方針は共通取得設計へ委ねる |
| 専用解析モードの導入意図・RF解析とsemantic decodeの境界 | [VSA_Bluetooth_WiFi_Dedicated_Analyzer_Design_JA.md](VSA_Bluetooth_WiFi_Dedicated_Analyzer_Design_JA.md) | 初期提案と追記を含む構想資料。Wi-Fi案や初期実装中という表記を統合VSAの現在の対応表に使わない |
| Bluetoothの汎用解析・表示処理と専用RF測定の境界 | [bluetooth_dedicated_analysis_pipeline_ja.md](bluetooth/bluetooth_dedicated_analysis_pipeline_ja.md) | FSK/PSK表示・複数packetの座標系と、EDR DEVM / HDT EVMの専用経路を区別 |
| DECT PHY/RFの実装上の検討事項 | [DECT_PHY_RF_Tester_Implementation_Guide.md](dect/DECT_PHY_RF_Tester_Implementation_Guide.md) | 規格を実装視点で整理したプロジェクト資料。規格原文でも、記載する受信試験・HLM等の全実装を保証する一覧でもない |

## 別の担当領域へ委ねる事項

| 事項 | 参照先 |
| --- | --- |
| ウィンドウ位置・サイズ、dock・tab、再起動とモード切替 | [共通ウィンドウ仕様](../../spec/common/window-layout.md) |
| デバイス選択とIQ Power表示の状態 | [VSAデバイス・表示状態](../../spec/vsa/general/vsa-device-and-display-state.md) |
| 設定・パターン・IQ・symbolの保存と読込 | [VSAファイル操作](../../spec/vsa/general/vsa-file-workflows.md)、[共通フォルダ履歴](../../spec/common/file-dialog-folders.md) |
| RX producer、cursor、再アーム、Single/Continuousの取得 | [連続IQ取得](../acquisition/continuous-iq-acquisition.md) |
| IQから測定することと、復調済みbit列を解釈することの境界 | [共有Protocol Packet Analyzer](../common/shared_protocol_packet_analyzer_design.md) |
| Bluetooth packet形式・生成との共通前提 | [Bluetooth RF Test Packet設計メモ](../common/Bluetooth_RF_Test_Packet_Design_Memo_JA.md) |
| ADS-Bの機能と制約 | [ADS-B仕様](../../spec/vsa/adsb/adsb1090.md) |
| 利用者向けの解析フロー・アルゴリズム説明 | [VSA解析補足](../../user-manual/Pluto_VSA_Analysis_Guide_JA.md) |
| 実測差・性能・監査 | [VSA検証索引](../../verification/vsa/README.md)、[専用VSA性能記録](../../verification/performance/Dedicated_VSA_Performance_Profile.md) |

## 重複を増やさない更新方法

- 同じ設定項目の操作説明を設計書ごとに書き直さず、操作マニュアルへリンクします。設計書には理由・境界・制約を残します。
- 共通パネルの所有関係と設定Widgetの所有関係は対象を区別し、`VSA_UI.md` の§3と§8で扱います。旧Config Topを前提とする共通UI文書は、固有のWidget所有権の注意点を§8へ移して削除しました。
- 汎用同期、規格別RF測定、semantic decode、表示処理を区別します。共通コードを使うことと、測定値の定義が同じであることを混同しません。
- 初期構想の本文は、単に重複しているという理由で統合・削除しません。本文を改訂するときは、該当する [照合項目](../../verification/vsa/README.md) と参照先を同時に更新します。
- 旧メニューの棚卸しやCodex指示書へ、恒久的な仕様の追記を積み重ねません。

UIの責務・操作経路は`VSA_UI.md`、要件は該当する`docs/spec/`、個々の設定値と操作例はユーザーマニュアルで管理します。今後の変更でもこの担当範囲に追記し、同じ内容の仕様書を並立させないでください。
