# テスト

- `common/`: 共有SDR、取得ストリーム、Trigger、デバイス、共通UI・ユーティリティ。
- `rtsa/`: RTSA、Sweep、HSTA、スペクトラム処理。
- `calibration/`: 校正。
- `vsa/core/`: General VSA、パターン、表示、保存、セッション、UI。
- `vsa/bluetooth/`、`vsa/dect/`、`vsa/adsb/`: 各プロトコルの解析・専用VSA。
- `vsg/`: 波形生成、パケット構成、送信、VSG UI。
- [data/](data/README.md): IQ fixture、設定・出力・プロジェクト保存例。

リポジトリルートから `python -m pytest` で全件実行します。テストファイル名やテスト条件は配置変更で変えません。フォルダの運用ルールは [AGENTS.md](AGENTS.md) を参照してください。
