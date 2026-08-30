# VSA Meas Config 共通UI設計

更新日: 2026-08-30

## 方針

- Generic VSA と Bluetooth 専用解析モードは、`HierarchicalMeasConfigDialog` を共通利用する。
- Config Top Menu、2列の大ボタン、個別ページ、`< Config Top` による戻り操作、モーダル動作、文字サイズを共通仕様とする。
- 各解析モードはページ名とページ内容だけを供給し、階層ナビゲーションを個別実装しない。
- Bluetooth固有の Profile / Protocol / PHY 等は `Bluetooth Analysis` ページへ配置する。
- PHYから一意に決まる設定は表示専用、または編集不可とする。

## Widget所有権

Config内の入力Widgetを非表示Toolbarや`QWidgetAction`へ重複登録しない。Widgetの所有先はConfigページに一本化する。

Qtでは、非表示Toolbarが保持する`QWidgetAction`へ登録したWidgetを別レイアウトへ移しても、Action側の可視状態に影響されて入力欄が非表示になることがある。このため、設定値を操作するWidgetとメイン画面上の操作Widgetを共用しない。

## 現在の適用範囲

- Generic VSA Meas Config
- Bluetooth Dedicated Analyzer Meas Config

Bluetooth側では以下を確認する。

- `Bluetooth Analysis`: Profile / Protocol / PHYなどを表示・編集できる。
- `Input / Frontend`: Center Frequency / Capture Length / Internal Gainなどを表示・編集できる。
- PHY依存値は編集不可になる。
