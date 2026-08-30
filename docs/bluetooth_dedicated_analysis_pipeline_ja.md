# Bluetooth専用解析パイプライン補足

更新日: 2026-08-30

## Generic VSAとの共通化方針

Bluetooth専用解析は、独自の簡易復調器を持たず、Generic VSAと同じ
`VSASession`、pattern解析、表示DSPを使用する。専用モードが担当するのは、
PHY・packet境界・既知同期列から解析条件を自動決定する部分と、Bluetooth
field decodeおよびBluetooth固有結果の表示である。

## FSK Measurement Filter

BR/LEのGFSKは送信信号自体にGaussian shapingが含まれる。専用解析で同じ
Gaussian特性をMeasurement Filterとして再適用すると、特に01交互列の
シンボル判定周波数偏移が過小になる。そのため専用Bluetooth FSK解析は
Measurement Filterを`None`とし、wide discriminator出力からシンボル判定を
行う。連続瞬時周波数トレースとConstellation Frequencyは同じ復調結果を
参照する。

EDR PSK部はPHYで規定したRoot Raised Cosine TX filterに対応するmatched
receive filterを`Auto`で適用する。

## EDR Vector / Symbol Plot

EDR部はdecoded packet境界に切り詰めた局所IQをGeneric VSA共通のPSK表示DSPへ
渡す。専用packetのシンボル数は表示負荷上十分小さいため、PSK Vectorの
サンプル間引きは行わない。pyqtgraphのauto downsamplingとclip-to-viewも
無効にし、filter通過後の全サンプル軌跡を描く。Symbol Plotは共通の正規化・
pi/4-DQPSK/8DPSK差動処理を使用する。

## EDR品質指標

Bluetooth専用画面のEDR品質指標は`Bluetooth DEVM RMS`のみを表示する。
Generic EVMおよびDifferential Symbol EVMは内部診断値として保持できるが、
専用Result Summaryには重複表示しない。

FSK部とPSK部の平均電力は、各部のdBm値を直接算術平均せず、いったんmWへ
戻して線形領域で平均した後にdBmへ変換する。Result Summaryには
`FSK Average Power`、`PSK Average Power`、および
`Relative Power (PSK - FSK)`を表示する。Relative Powerの正符号はPSK部が
FSK部より高いことを示す。

## 複数パケット解析と表示状態

初回のFSK同期探索で得た全候補のsample位置を保持し、2件目以降は候補近傍の
局所IQだけを解析する。全キャプチャをpacket indexごとに繰り返し探索しない。
局所IQ上のsample番号は`recording_sample_offset`と
`analysis_sample_offset`で元キャプチャの絶対sample番号へ戻し、Power、FSK
Modulation、Result Rangeを同じ時間軸に表示する。

左右キーによるpacket移動、およびDisplay Config変更に伴う再描画では、
Modulation/Symbol PlotのFSK/PSK tab選択を保持する。選択packetを変更した際は
FSK ModulationのX rangeだけを選択packetへ追従させる。

10個の内部生成2-DH1を使った開発時回帰では、10件すべてのCRCを確認し、候補
局所解析化後の処理時間は約1.4秒（開発機上、UI描画を除く）だった。実機IQの
処理時間はcapture長、SNR、候補数、PC性能に依存する。

## Packet Analysis表示

Decode treeは省略記号を使用しない。Payload/Meaningはセル内で折り返し、
ウィンドウ幅に応じてValue/Meaning列へ余白を配分する。全値はtooltipでも
確認できる。

## 回帰確認

- 生成LE 1M IQ: 同期、Length、CRC、FSK filter mode
- 生成2-DH1 IQ: BR header、EDR Length/CRC、PSK packet境界、DEVM
- UI: 全PSK trajectory sampleの描画、Decode tree非省略表示
- 複数生成2-DH1: 全packet CRC、絶対sample offset、FSK/PSK個別電力
- UI: packet移動後のFSK表示範囲、FSK/PSK tab選択保持
