# Pluto計測アプリケーション ユーザーマニュアル

ADALM-Plutoを使用する3アプリの操作マニュアルです。2026-09-23時点の実装（`b43f7e6`）に合わせたMarkdown版と、その内容を組版したPDF版があります。

| アプリケーション | 用途 | マニュアル |
|---|---|---|
| Pluto RTSA | スペクトラム／時間領域観測 | [Pluto RTSAユーザーマニュアル](Pluto_RTSA_User_Manual_JA.md) |
| Pluto VSA | 汎用および規格別の変調・パケット解析 | [Pluto VSAユーザーマニュアル](Pluto_VSA_User_Manual_JA.md) |
| Pluto VSG | パケット波形生成およびPluto送信 | [Pluto VSGユーザーマニュアル](Pluto_VSG_User_Manual_JA.md) |
| VSA補足 | 入力IQから測定値までの処理順・計算式・測定条件 | [VSA解析フロー・アルゴリズム補足](Pluto_VSA_Analysis_Guide_JA.md) |
| 共通セットアップ | ADALM-Pluto Windowsドライバの導入と認識確認 | [Plutoドライバ インストールガイド](Pluto_Driver_Installation_Guide_JA.md) |

## 読み進め方

1. 各マニュアルの第2章で画面の構成、第3章で操作手順を確認します。
2. 各設定の意味・単位・関連設定は、後続章の項目別表で確認します。
3. VSAの測定値を詳しく理解する場合は補足資料を読みます。同期・復号、RF測定、表示加工を区別して説明しています。

掲載した全画面・設定画面は[画面一覧](manual-screen-index.md)、入力データと確認範囲は[図版・確認記録](manual-validation.md)にまとめています。Mermaid対応のMarkdownビューアでは、補足資料の解析フローを図として表示できます。

## 今回反映した仕様

- RTSA / VSA / VSGそれぞれの現在のメニュー構成と設定項目。
- `General VSA`表記、VSAの4モード、VSGの各規格の設定。
- ウィンドウ位置・サイズ復元、960×640の最小サイズ、Dockとモード別レイアウトの保存範囲。
- ファイル種別ごとの初期フォルダ記憶。
- 保存済み受信IQを現在のアプリで解析した画面と、VSG・ADS-Bの生成データの画面。

図版作成では新たな実機送受信を行っていません。RTSAの図は開発用の再生経路を使っています。通常メニューのIQ読込手順ではありません。電力校正や規格判定に必要な条件は各本文で説明しています。

## PDF版

| 資料 | PDF | ページ数 |
|---|---|---|
| RTSA | [PDF](../../output/pdf/Pluto_RTSA_User_Manual_JA.pdf) | 7 |
| VSA | [PDF](../../output/pdf/Pluto_VSA_User_Manual_JA.pdf) | 18 |
| VSG | [PDF](../../output/pdf/Pluto_VSG_User_Manual_JA.pdf) | 16 |
| VSA解析補足 | [PDF](../../output/pdf/Pluto_VSA_Analysis_Guide_JA.pdf) | 8 |

章の開始、表紙、目次を理由にした固定改ページはありません。本文と表は続けて配置し、表がページをまたぐ場合は見出し行を繰り返します。図とキャプションは一体で配置し、1ページ内に収めます。補足資料の3つのフロー図は、MarkdownのMermaid定義からPDF内のベクター図へ変換しています。

再生成にはWindowsのメイリオフォントと、ReportLab・Pillowを利用できるPython環境が必要です。

```powershell
python -m tools.build_user_manual_pdfs
```

出力は`output/pdf/`です。ドライバガイドも再生成する場合は`--include-driver`を付けます。PDFは生成物であり、Git管理外です。
