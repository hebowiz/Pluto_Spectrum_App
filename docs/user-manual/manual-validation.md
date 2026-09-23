# 図版・確認記録

更新日: 2026-09-23 / アプリ仕様の基準: `b43f7e6`

対象: RTSA・VSA・VSGユーザーマニュアルとVSA解析補足のMarkdownレビュー版

## 1. 図版の入力と再現条件

画像は現在のQtウィジェットを実際に起動して撮影しています。測定値や波形を画像へ書き足していません。全体画面の赤枠と番号だけを操作説明用に重ねています。

| 図 | 入力・条件 | 読み方 |
|---|---|---|
| RTSA全体・設定画面 | `tests/fixtures/bluetooth_br_prbs9_pluto_16msps.npz`。保存済みPluto受信IQ、約16 MS/s・中心2441 MHz。Gain 0 dB、外部ATT 30 dBの収録情報を反映。RBW 100 kHz | 開発用スクリプトで受信部分を置換し、現在のFFT・表示処理へ投入。通常UIにIQ Importがあるという意味ではない。振幅はnominal換算であり新規校正ではない |
| General VSA全体・設定画面 | 同じBR受信IQ。FSK、1 MSym/s、参照偏移160 kHz、Gaussian BT=0.5。LAP C6967Eの72-symbol Access CodeでPattern Search、Result Length 256 | 汎用同期・変調表示の例。Bluetooth専用の適合判定と区別 |
| Bluetooth全体・設定画面 | `tests/fixtures/RT_Packet_TX_2DH1.npz`。General Packet、Classic Auto。Import IQの経路で読込 | LO offsetメタデータから解析チャネルを再構成。EDR 2M / 2-DH1を4 packet検出。HEC valid。General PacketのRF判定N/AをPASSと読み替えない |
| DECT全体・設定画面 | `tests/fixtures/DECT_PP_A5_OK.npz`。9.216 MS/s、JP-DECTの1902.528 MHz。Software DC removedの保存データ | ファイル名にPPを含むが、画面の検出結果はRFP P32Z。N/AやINCOMPLETEを含む例であり、ファイル名のOKは全RF項目の合格を意味しない |
| ADS-B全体・設定画面 | `tests/fixtures/adsb1090_multi_8msps.npz`。生成IQ、8 MS/s | 4 messageのparity確認と表示を確認。画面のOS Timeは解析時の時刻であり、現在の実在航空機の受信記録ではない |
| VSG全体 | 組込みBluetooth BR/EDRプロジェクト。DH1、PRBS-9、whitening OFF、8 MS/s。Verify Packet実行 | 生成bit列のDecode結果と生成IQのPreview。RF出力の受信測定ではない |
| VSG設定画面 | BR/EDR、LE、HDT、Wi-Fi、DECTの組込み初期プロジェクト | 各RF / TimingとFieldsを撮影。規格・PHYによって無効になる項目もそのまま掲載 |

全画像へのリンクは[画面一覧](manual-screen-index.md)を参照してください。設定画面の数値は撮影時の設定であり、すべての信号に適した推奨値や常に適用される初期値ではありません。

## 2. 生成方法

リポジトリ直下で、依存関係を導入済みの環境から実行します。

```powershell
$env:QT_QPA_PLATFORM='windows'
& .\.venv\Scripts\python.exe -m tools.generate_user_manual_screenshots
```

スクリプトは[generate_user_manual_screenshots.py](../../tools/generate_user_manual_screenshots.py)です。機器検索と取得を置き換え、QSettingsは一時ディレクトリに分離します。個人のRTSA校正CSVも読み込みません。RF送信は開始しません。GUIが一時的に開くため、編集中のアプリとは別に実行してください。

WindowsネイティブのQt描画を使用します。offscreenプラットフォームでは環境によりフォントが欠けるため、今回の掲載画像には使用していません。設定フォームの項目一覧は確認用の`tmp/manual-ui-inventory.json`へ出力します。

## 3. 内容の照合範囲

| 対象 | 確認内容 |
|---|---|
| 設定項目 | 現在のUIフォーム、選択肢、無効項目と説明表を照合。規格共通の項目は共通章から参照 |
| メニューとファイル | 操作パネル、State/File、各読込・保存処理、フォルダ記憶の実装を確認 |
| 起動・レイアウト | [ウィンドウ仕様](../window-layout.md)と現行実装に合わせて保存範囲を記載 |
| VSA解析 | 補足資料の各節から実装へリンクし、DDC、同期、FSK、EVM/DEVM、DECT電力、ADS-B復調の経路を説明 |
| 図版 | 現行アプリによる解析・生成、文字・波形・設定フォームの表示を確認 |
| Markdown | 相対リンク・掲載画像の存在と整合、差分の書式を確認 |

今回の作業は資料と図版生成ツールの改訂です。アプリ本体の挙動は変更していません。実機の取得・送信・校正を再実施した検証記録ではありません。

## 4. レビュー時の観点

初回利用者が第3章の手順を追えるか、各設定の説明に不足がないか、VSA補足の数式・処理粒度が適切かを本文で確認します。

## 5. PDF生成・確認（2026-09-23追記）

追加依頼に基づき、3アプリとVSA解析補足の4冊をPDF化しました。RTSA 7ページ、VSA 18ページ、VSG 16ページ、解析補足8ページです。

- 章・表紙・目次での固定改ページを廃止し、本文と表を連続配置。
- スクリーンショット19点をキャプションと一体化し、ページ境界で分割しないことを確認。
- 解析補足のMermaid 3図を、内容・接続を保持したベクター図へ変換。縦長ページに合わせて上から下へ配置。
- Popplerで全49ページを画像へ展開し、表の継続、画像、文字、ページ番号を目視確認。
- PDF抽出テキストとMarkdownの本文・表セル・図ラベルを照合し、掲載内容の欠落がないことを確認。
- 画像数とページ内の描画範囲、PDFの目次・しおりを確認。

生成処理は[build_user_manual_pdfs.py](../../tools/build_user_manual_pdfs.py)。今回はドライバガイドのPDFを更新していません。
