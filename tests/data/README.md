# テストデータ

テストコードは対象サブシステムの `tests/common/`、`rtsa/`、`calibration/`、`vsa/`、`vsg/` に置き、共有する入力データ・保存例をここにまとめます。

| 配置 | 内容 |
| --- | --- |
| `fixtures/bluetooth/br-edr/` | Bluetooth BR/EDRのIQ録音・生成波形・IQ-TAR |
| `fixtures/bluetooth/le/` | Bluetooth LEのIQ録音 |
| `fixtures/bluetooth/hdt/` | HDTのIQ録音・生成波形・IQ-TAR |
| `fixtures/dect/` | DECTのIQ録音・生成波形 |
| `fixtures/adsb/` | ADS-B IQ波形と `packet-logs/` のパケット保存例 |
| `fixtures/general/` | プロトコルに依存しないIQ形式検証用波形 |
| `configs/` | VSA測定設定の保存例 |
| `patterns/` | VSA参照パターンの保存例 |
| `symbols/` | シンボル出力の保存例 |
| `vsg-projects/` | VSGプロジェクトの保存例 |

`configs/from-pattern/` は旧patternフォルダに置かれていた設定ファイルです。同名でも既存の設定とは内容が異なるため、上書き・統合せず分離して保持します。録音に付属するJSONは同じフォルダに置きます。保存データ内の採取元パスや作成時の情報は履歴として保持します。

代表的な入口は [BR録音](fixtures/bluetooth/br-edr/bluetooth_br_prbs9_pluto_16msps.npz)、[DECT波形](fixtures/dect/dect_rfp_p32_prbs9_9p216msps.npz)、[ADS-B波形](fixtures/adsb/adsb1090_multi_8msps.npz)、[汎用IQ-TAR](fixtures/general/rs_sample_gfsk_8msps.iq.tar) です。実際の用途はそれぞれを読むテスト・ツールを参照してください。未使用の手動測定データも削除せず保持しています。

人間が目視比較する画像は [docs/verification/assets/captures](../../docs/verification/assets/captures/) にあります。ユーザーマニュアルの画像とは分離します。

データはプロトコルを明示したパスで参照します。fixtureを再生成・変更する前に理由を確認し、配置変更だけで条件や期待値を変えないでください。詳細は [tests/AGENTS.md](../AGENTS.md) を参照してください。
