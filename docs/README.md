# プロジェクト文書

Pluto RTSA / VSA / VSG自身の仕様・設計・検証・操作資料を管理します。メーカー文書・規格原文・ユーザー提供資料は [references](../references/README.md) に分離します。

| 配置 | 役割 | 主な入口 |
| --- | --- | --- |
| `spec/` | 現在の実装がどう動くべきかを示す現行仕様 | [共通設定](spec/common/common.md)、[ウィンドウ配置](spec/common/window-layout.md)、[ファイル選択フォルダ](spec/common/file-dialog-folders.md) |
| `design/` | アーキテクチャ、処理方式、設計理由・設計案 | [測定アーキテクチャ](design/common/measurement-architecture.md)、[共有パケット解析](design/common/shared_protocol_packet_analyzer_design.md) |
| `verification/` | 実測、性能評価、監査、手動比較記録 | [実機検証](verification/hardware/hardware-validation.md)、[専用VSA性能](verification/performance/Dedicated_VSA_Performance_Profile.md)、[比較画像](verification/assets/captures/) |
| `work-notes/` | 一時的な実装指示・調査計画・引き継ぎ | [VSA実装ノート](work-notes/vsa-implementation.md)、[Bluetooth適合性レビュー](work-notes/Bluetooth_Classic_BLE_VSA_SIG_Compliance_Review_and_Codex_Instructions.md) |
| `archive/` | 明確に置き換え済み・legacyの履歴 | [統一前のウィンドウ配置](archive/window-layout-current-state.md)、[旧Time Analyzer](archive/rtsa/time-analyzer-legacy.md) |
| `user-manual/` | 利用者向け操作説明・解析補足 | [ユーザーマニュアル一覧](user-manual/README.md) |
| `images/user-manual/` | マニュアル用画像 | [画面索引](user-manual/manual-screen-index.md) |

## アプリ・機能別の入口

- 共通: [連続IQ取得](design/acquisition/continuous-iq-acquisition.md)、[IQストリーム](design/acquisition/iq-streaming.md)、[デバイス所有権](spec/common/pluto-device-ownership.md)、[終了処理](spec/common/graceful-application-shutdown.md)。
- RTSA: [Realtime SA](spec/rtsa/realtime-sa.md)、[Wideband RTSA](spec/rtsa/wideband-rt-sa.md)、[Sweep SA](spec/rtsa/sweep-sa.md)、[HSTA](spec/rtsa/high-speed-ta.md)、[校正](spec/rtsa/calibration.md)、[RBW監査](verification/rtsa/rbw-processing.md)。
- General VSA: [設計文書の役割と参照先](design/vsa/README.md)、[文書・実装の照合](verification/vsa/README.md)、[構成](design/vsa/vsa-architecture.md)、[搬送波同期](design/vsa/vsa-carrier-synchronization.md)、[ファイル操作](spec/vsa/general/vsa-file-workflows.md)、[デバイス・表示状態](spec/vsa/general/vsa-device-and-display-state.md)。
- 専用VSA: [Bluetooth解析フロー](design/vsa/bluetooth/bluetooth_dedicated_analysis_pipeline_ja.md)、[DECT実装ガイド](design/vsa/dect/DECT_PHY_RF_Tester_Implementation_Guide.md)、[ADS-B仕様](spec/vsa/adsb/adsb1090.md)。
- VSG: [波形生成設計](design/vsg/iq_waveform_generator_design_spec.md)、[RFレベルとRMS](design/vsg/Pluto_VSG_RF_Level_RMS_Design.md)、[連続送信](design/vsg/pluto-vsg-continuous-transmission.md)、[フィールド階層](design/vsg/pluto_vsg_field_hierarchy.md)。

## 分類上の注意

名称だけでなく本文の目的に基づいて分類しています。[VSA UI](design/vsa/VSA_UI.md) はUI設計、[メニュー棚卸し](verification/vsa/VSA_menu_inventory.md) と [Bluetooth BRノート](verification/vsa/vsa-bluetooth-br.md) は調査・検証、[Bluetooth EDRデバッグ計画](work-notes/vsa-bluetooth-edr.md) は作業資料です。VSA設計資料の役割分担と旧Config文書の整理は [設計索引](design/vsa/README.md)、実装との差分は [照合記録](verification/vsa/README.md) を参照してください。

設計案や作業資料をそのまま現行仕様として扱わず、`archive/` を現行仕様の根拠にしないでください。新規資料を作る前に既存の該当資料を確認し、関係する資料だけを参照します。詳細な運用ルールは [AGENTS.md](AGENTS.md) にあります。
