# VSA検証資料と文書・実装の照合

この索引は検証記録です。正式仕様は `docs/spec/`、設計文書の担当範囲は [VSA設計索引](../../design/vsa/README.md) を参照してください。

## 既存の検証資料

- [FSK同期監査](vsa-fsk-synchronization-audit.md): 同期・フィルタの定量評価。
- [Bluetooth BR実装・検証ノート](vsa-bluetooth-br.md): BR解析と実測録音の検証。
- [VSAメニュー棚卸し](VSA_menu_inventory.md): メニュー再編時の調査記録。現行の操作名は [ユーザーマニュアル](../../user-manual/Pluto_VSA_User_Manual_JA.md) を参照。
- [専用VSA性能](../performance/Dedicated_VSA_Performance_Profile.md)、[VSA UI性能](../performance/vsa-ui-performance.md): 性能評価・描画改善の記録。

## 照合範囲

整理前のVSA設計8文書について、主題の重複と、操作・対応機能・取得・解析責務に関する記述を実装と照合しました。旧Config Top文書を整理し、担当文書は7つになっています。続いてUI、全体設計、Bluetooth設計案・解析補足、Trigger設計の本文を改訂しました。下記は発見した差分と修正根拠です。DSP数式・規格値の網羅的な再検証、実機測定、外部規格の再解釈はこの調査の対象ではありません。

### D01 設定画面の入口

旧 `vsa_config_ui_shared_design_ja.md` はConfig Topと戻るボタンを説明し、[右側操作UI設計](../../design/vsa/VSA_UI.md) はその廃止を説明していました。

統合VSAの通常操作では、右パネルから指定ページを直接開きます。[設定ダイアログ実装](../../../pluto_vsa/ui/measurement_config_dialog.py) の `open_page()` はTop表示とBackを隠します。一方、部品には `open_top()` も残っています。「内部クラスからTop機能を完全削除した」とは読み替えません。[設定経路テスト](../../../tests/vsa/core/test_vsa_setup_controls.py) の `test_setup_button_order_and_all_dialog_routes` で通常の呼出経路を確認できます。

旧Config文書の固有情報だった非表示Toolbar／QWidgetActionとWidget所有権の注意点を右側UI設計§8へ移し、旧文書を削除しました。共通パネルの所有は§3、設定Widgetの所有は§8を参照します。旧メニュー構成はGit履歴で確認できます。

### D02 State・File・Deviceと設定ページ名

右側UI設計にあった `System > Recall/Save/File/Device`、`Open IQ`、Bluetooth Analysis / DECT Analysis / ADS-B Analysisという旧一覧を改訂しました。

[共通パネル](../../../pluto_vsa/ui/control_panel.py) はSYSTEMグループに `State`、`File`、`Device` を直接並べ、StateにRecall・Save・Presetを置きます。[workspace接続](../../../pluto_vsa/ui/application_window.py) の `_panel_spec()` は全モードの基本設定をSignal Description、Input / Frontend、Signal Capture、Triggerとし、Fileの読込名をImport IQとします。追加ページはモードごとに異なります。

操作名は [マニュアル](../../user-manual/Pluto_VSA_User_Manual_JA.md) と [設定経路テスト](../../../tests/vsa/core/test_vsa_setup_controls.py) を参照します。右側UI設計は設定順序・File操作・現在の到達先を更新し、ResetがIQ・結果・履歴・プロットをクリアして設定を保持する点も実装へ合わせています。

### D03 配置の保存と復元

[全体設計](../../design/vsa/vsa-architecture.md) §10にあったsession全体のlayout保存・復元やclose/duplicateの構想を、現行の配置管理に置き換えました。右側UI設計と専用モード案§3.2も同じ扱いへ更新しました。

現在の要件は [共通ウィンドウ仕様](../../spec/common/window-layout.md) にあります。位置・サイズは再起動時に復元しますが、dock配置・分割比率は再起動時には復元せず、選択タブも永続保存しません。VSAのモード別配置は [外枠の `_workspace_layouts`](../../../pluto_vsa/ui/application_window.py) に実行中だけ保存します。統合VSAではdockを閉じる操作も無効です。[レイアウトテスト](../../../tests/common/test_window_layout.py) がこの区別を検証します。

### D04 General VSAという表示名

改訂対象の全体設計・専用モード案・Bluetooth補足の表示名をGeneral VSAへ統一しました。現在の表示は [パネルの `_build_mode_page()`](../../../pluto_vsa/ui/control_panel.py) に基づきます。内部mode IDの `generic` や `generic_workspace` は保持します。過去の記録や内部名を一律に置換する作業ではありません。

### D05 専用モードの構想と現在の選択肢

[Bluetooth/Wi-Fi専用モード案](../../design/vsa/VSA_Bluetooth_WiFi_Dedicated_Analyzer_Design_JA.md) の冒頭を現行モードに合わせ、「Bluetooth初期実装中」という状態表示を削除しました。初期モード図とWi-Fiの章は拡張案と明記しています。

[統合VSAの外枠](../../../pluto_vsa/ui/application_window.py) と [モード選択](../../../pluto_vsa/ui/control_panel.py) に登録されているのはGeneral VSA、Bluetooth、DECT、ADS-B 1090ESの4つです。Wi-Fiはこのメニューの選択肢ではありません。リポジトリ内のWi-Fi関連コードやVSGの対応を、統合VSAへの登録と同一視しません。

### D06 QAMは一律に将来機能ではない

全体設計§1・§4・§14にあるQAM全般を将来／対象外とする記述を改め、16QAMの実装と、それ以外の方式への拡張を区別しました。初期Phase一覧は当時のロードマップと明記しています。

現在の [modulation定義](../../../pluto_vsa/model.py) は16QAMを含み、[QAMパターン同期テスト](../../../tests/vsa/core/test_vsa_pattern_qam.py) と [detected-dataテスト](../../../tests/vsa/core/test_vsa_pattern_detected_data.py) が存在します。これは全QAM方式・全規格への対応を意味しません。初期の対象外一覧を現在の対応表として使わないでください。

### D07 汎用解析の再利用と専用RF測定

専用モード案§29と [Bluetooth解析補足](../../design/vsa/bluetooth/bluetooth_dedicated_analysis_pipeline_ja.md) を改訂し、汎用session・pattern・表示DSPと専用RF測定の呼出しを区別しました。「全RF測定値が汎用EVMから得られる」という旧説明を取り除きました。

[Bluetooth解析model](../../../pluto_vsa/protocol_modes/bluetooth/model.py) はEDRの `measure_edr_devm()` やHDTの `build_hdt_evm_result()` を呼び出し、[規格別RF測定](../../../pluto_vsa/protocol_modes/bluetooth/rf_measurement/) と表示用の汎用処理を区別しています。[RF測定テスト](../../../tests/vsa/bluetooth/test_bluetooth_rf_measurement.py) の `test_hdt_payload_phase_and_cfo_fit_is_independent_of_generic_display` もその境界を検証します。

汎用同期の詳細は [搬送波同期設計](../../design/vsa/vsa-carrier-synchronization.md)、利用者向けの経路の区別は [解析補足](../../user-manual/Pluto_VSA_Analysis_Guide_JA.md) を参照します。規格値・測定区間の正否をこの索引だけで判定しません。

### D08 Triggerと連続取得

[Trigger設計](../../design/vsa/vsa-iq-power-trigger.md) の「取得TriggerはRun Singleだけ」「最初のbufferはfresh-buffer経路」という初期記述を、Single / Continuousの共有producerとcursorの説明へ更新しました。Continuous / 再アームを未実装とする記述も修正しています。

現在のGeneral VSAは [UIの `_toggle_pluto_continuous()` / `_start_pluto_capture()`](../../../pluto_vsa/ui/main_window.py) でも同じ取得設定を渡し、[PlutoLiveSource](../../../pluto_vsa/pluto_source.py) がFree Run／I/Q Powerに対応するrecordを共通の連続producerから作ります。producerとcursorの扱いは [連続IQ取得設計](../../design/acquisition/continuous-iq-acquisition.md)、回帰は [Pluto sourceテスト](../../../tests/vsa/core/test_vsa_pluto_source.py) を参照します。

取得Trigger・取得後のBurst Search・Pattern Searchを区別する設計は維持されています。Continuous対応だけを理由に、Burst SearchのDrop-OutやHoldoffを取得Trigger設定へ移したとは解釈しません。

### D09 入力形式と将来のsource

全体設計§3のSigMF優先や `ScpiInstrumentSource` を将来構想と明記し、現行の入力形式を記述しました。[FileIQSource.load()](../../../pluto_vsa/sources.py) が形式別に扱うのはIQ-TAR、NPY、NPZで、それ以外はraw complex IQの経路です。rawとしてバイト列を読めることは、SigMFメタデータを解釈できることとは異なります。

統合VSAの実機取得は [PlutoLiveSource](../../../pluto_vsa/pluto_source.py) を使用しています。構想図だけを根拠にSCPI機器の取得操作やSigMF専用対応を案内しません。

### D10 DECT文書の対象範囲

[DECT実装ガイド](../../design/vsa/dect/DECT_PHY_RF_Tester_Implementation_Guide.md) はRF受信試験、HLM、将来のActive Lower Testerまで扱います。[現行解析入口](../../../pluto_vsa/protocol_modes/dect/analysis.py) はpassive Classic DECT GFSK同期・送信測定を対象としています。

ガイドの目次をそのまま実装済み機能の一覧にはせず、操作は [マニュアル](../../user-manual/Pluto_VSA_User_Manual_JA.md)、個別対応は [DECTテスト](../../../tests/vsa/dect/) と実装を確認します。ガイド全体の規格適合性や未実装項目を網羅的に判定したものではありません。

### D11 旧パッケージ名と提案中の構成図

専用モード案の実在ファイル参照に残っていた `pluto_rtsa/vsa/` は `pluto_vsa/`、連続IQ取得設計の共有SDRファイル参照は `pluto_common/sdr/` へ修正しています。専用モード案§19「package構成」は提案の記録であり、現在のツリーとしては扱いません。

## 回帰確認の対象

通常の設定ページへの経路、設定編集の分離、モード切替、配置保存、QAM同期、HDT測定と汎用表示の分離を、次の既存テストで確認します。

```text
python -m pytest -q
  tests/vsa/core/test_vsa_setup_controls.py
  tests/vsa/core/test_analysis_application.py
  tests/common/test_window_layout.py
  tests/vsa/core/test_vsa_pattern_qam.py
  tests/vsa/core/test_vsa_pluto_source.py
  tests/vsa/bluetooth/test_bluetooth_rf_measurement.py::test_hdt_payload_phase_and_cfo_fit_is_independent_of_generic_display
```

上記は引数を改行して示しています。実行時は1コマンドとして渡します。文書のローカルリンクと、削除した旧Config文書へのリンクが残っていないことも確認します。

本文改訂時の確認では、上記の既存テスト67件が成功しました。`QT_QPA_PLATFORM=offscreen`で実行し、実機による再測定は行っていません。`docs/`内のローカルMarkdownリンク311件の参照先が存在すること、設定Widget所有権の注意点が維持されていること、追跡ファイルの変更がMarkdownだけであることも確認しました。

## 本文改訂の範囲と限界

右側UIの旧画面遷移、Bluetoothの汎用処理と専用RF測定の境界、全体設計の構想と実装状況を本文へ反映しました。関連するTrigger設計も更新し、D02〜D09の既知の差分を注意書きだけに残さない形へ改訂しています。専用モード案§28の外部設定保存も共通version 2と起動時QSettingsを区別する説明に修正しました。

初期ロードマップ、Wi-Fi拡張案、DECTガイドの規格解説などは設計上の経緯・検討資料として残します。全規格への適合性確認や、全候補機能の実装監査を完了したという意味ではありません。測定アルゴリズム、テスト条件、外部資料、ユーザーマニュアルとPDFは変更していません。
