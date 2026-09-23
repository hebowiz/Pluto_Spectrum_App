# Pluto Spectrum App

ADALM-Pluto用のRTSA（スペクトラム解析）、VSA（ベクトル信号解析）、
VSG（信号生成）アプリケーションです。

## アプリの構成

3アプリはリポジトリ直下の独立したパッケージです。起動入口とUIの配置を揃えています。

```text
pluto_rtsa/                 RTSA
    __init__.py
    __main__.py             python -m pluto_rtsa
    main.py                 アプリの組み立て・起動
    ui/                     画面
    config/                 RTSAセッション保存
    modes/                  測定モード制御
    signal/                 スペクトラム処理
    utils/                  校正
pluto_vsa/                  VSA
    __init__.py
    __main__.py             python -m pluto_vsa
    main.py                 アプリの組み立て・起動
    ui/                     画面
    demod/                  復調
    profiles/               波形・解析プロファイル
    protocol_modes/         Bluetooth・DECT専用解析
    standards/              ADS-B解析
pluto_vsg/                  VSG
    __init__.py
    __main__.py             python -m pluto_vsg
    main.py                 アプリの組み立て・起動
    ui/                     画面
    engine/                 波形生成
    profiles/               波形プロファイル
    backends/               送信
pluto_common/               アプリ共通の機能
    config/                 受信設定・入力振幅補正・モード定義
    sdr/                    Pluto受信・IQストリーム・トリガ
pluto_protocol/             共通パケットデコード
pluto_cal/                  周波数校正アプリ
pluto_sa/                   旧Python参照・起動コマンドの互換窓口のみ
```

アプリ固有の処理は各アプリ内に配置します。RTSAとVSAで使う受信処理は
`pluto_common`に置き、VSAがRTSAのパッケージに依存しない構成です。
VSAとVSGで既に共有している波形処理・画面部品は、既存の呼び出し関係を維持しています。

## 起動

Windowsでは従来と同じ`Pluto_RTSA.bat`、`Pluto_VSA.bat`、`Pluto_VSG.bat`を使えます。
Pythonからは仮想環境とlibiioの実行環境を用意して、リポジトリ直下で実行してください。

```powershell
python -m pluto_rtsa
python -m pluto_vsa
python -m pluto_vsg
```

旧`python -m pluto_sa.main`、`python -m pluto_sa.vsa.main`と旧Python importは
互換窓口から新しい実装へ転送します。新規コードは新しいパッケージ名を使用してください。
設定の保存先・ファイル形式、測定処理、配布EXE名は従来どおりです。

詳しい仕様と運用手順は[ドキュメント一覧](docs/README.md)を参照してください。

## テスト

```powershell
$env:QT_QPA_PLATFORM = 'offscreen'
.\.venv\Scripts\python.exe -m pytest -q
```
