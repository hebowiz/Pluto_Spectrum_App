# WideBand RT SA 仕様

## 目的

PlutoSDRの瞬時帯域を超える周波数範囲を複数チャンクに分割して取得し、1本の広帯域スペクトラムとして合成表示します。

## 周波数範囲

- Start最小値: 80 MHz
- Stop最大値: 5.990 GHz
- 最小Span: 10 MHz
- 設計上の最大Span: 6 GHz
- Center/Span指定とStart/Stop指定の両方に対応

## チャンク構成

Frequencyメニューの`Chunk Width`から10／20／30／40 MHzを選択できます。初期値は10 MHzです。選択幅は合成結果へ採用する帯域幅とchunk間隔を兼ね、測定帯域の下端から上端方向へ順番に配置します。最後のchunkだけ測定Stop位置で切り詰めます。

採用帯域の左右へ各5 MHzを追加した範囲をFFTの表示解析Spanとし、さらに従来どおり外側4%をFFT guardとして確保します。このためPlutoへ設定するSR/RF BWは`(Chunk Width + 10 MHz) / 0.92`です。

| Chunk Width | FFT表示解析Span | SR / RF BW（公称） | LO中心 |
|---:|---:|---:|---:|
| 10 MHz | 20 MHz | 約21.739 MHz | chunk開始 + 5 MHz |
| 20 MHz | 30 MHz | 約32.609 MHz | chunk開始 + 10 MHz |
| 30 MHz | 40 MHz | 約43.478 MHz | chunk開始 + 15 MHz |
| 40 MHz | 50 MHz | 約54.348 MHz | chunk開始 + 20 MHz |

| 共通項目 | 値 |
|---|---:|
| 採用帯域外の固定guard | 左右各5 MHz |
| FFT外周guard | 左右各4% |
| LO settle | 200 µs |
| リチューン後flush | FFTサイズのブロックを5回 |

各チャンクではRealTime SAと同じGaussian FFT filter bankを実行し、選択したChunk Width部分だけを合成バッファへ書き込みます。全チャンクの取得完了後に1フレームとして公開します。RBWは両側3 dB bandwidth、ENBWは約`1.0645 × RBW`で、FFT電力化後のGaussian畳み込みは行いません。

## 処理フロー

1. 指定帯域を選択したChunk Width単位のチャンクへ分割
2. 対象チャンクのLOへリチューン
3. 200 µs待機
4. RXバッファを5回読み捨て
5. IQを1ブロック取得してFFT/RBW処理
6. 周波数補正を含む表示補正を適用
7. 対応する合成バッファ範囲へ格納
8. 全チャンク完了後にトレース、マーカー、Waterfall、Persistenceを更新

## 表示・操作

- Spectrum、Waterfall、Persistenceを利用できます。
- トレースとマーカーは完成した合成フレームを対象にします。
- Singleは全チャンクを1回取得した後に停止します。
- Continuousは全チャンクの巡回を繰り返します。

## 制限・注意事項

- 名称にRealTimeを含みますが、表示更新周期はチャンク数と各リチューン時間に比例します。
- チャンク境界で位相や取得時刻は連続しません。
- 各チャンクの同期取得結果は共通`IQBlock`として発行され、LO切替は新しい`stream_id`になります。
- 各チャンクは別時刻の測定値なので、瞬間的に変化する信号の広帯域スナップショットではありません。
- 各チャンク内も現段階では1 FFT frameであり、overlap STFTによる全sample被覆は未実装です。
- 狭いRBWではFFT Sizeを最大16384まで自動拡張し、なお不足する場合はステータスの`Eff RBW`へ`limited`を表示します。
- Spanが大きいほど1フレーム完成までの時間が長くなります。
- Chunk Widthを広げるほど1フレームのchunk数とリチューン回数は減りますが、Plutoのusable bandwidth中央部から離れた領域も採用します。ノイズフロアの盛り上がり、DC/LO由来の像、mirror、帯域端roll-off、周波数別振幅誤差が増える可能性を理解した上で使用してください。既定10 MHzは品質優先設定です。
