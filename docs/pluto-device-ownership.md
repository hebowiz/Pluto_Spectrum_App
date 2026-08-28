# Pluto device identity and ownership

最終更新: 2026-08-28

## 目的

RTSA、VSA、VSGを同時に起動し、接続された複数のADALM-Plutoから任意の個体を
RX/TXへ割り当てる運用を安全にする。個体を恒久的なTX専用/RX専用にはしない。

## 個体識別

- 保存値は一時的な`usb:x.y.z`ではなく、取得可能なら`serial:<hardware serial>`を使う。
- 接続時にserial selectorから現在のlibiio URIを解決する。
- direct USBと`ip:pluto.local`が同一serialなら1台として扱い、direct USBを優先する。
- 各アプリのウィンドウタイトルには役割とserial末尾4桁を表示する。
  - `Pluto RTSA [RX: …1234]`
  - `Pluto VSA - Generic FSK / PSK [RX: …1234]`
  - `Pluto VSG - IQ Waveform Generator [TX: …5678]`

## 排他制御

`pluto_common.device_lease.PlutoDeviceLease`が物理個体単位のプロセス間lockを管理する。

- RTSA/VSAは`PlutoReceiver`がlibiio contextを所有する間、RX leaseを保持する。
- VSGはPrepareまたはTransmitでcontextを開いている間、TX leaseを保持する。
- RX/TXの別を問わず、同一serialの同時使用は拒否する。
- エラーには所有application、RX/TX role、PIDを含める。
- process crash時はOSがfile lockを解放するため、恒久的なlock残留は起こらない。
- lock metadata fileが残っても、実際のlock取得可否を所有判定の正とする。
- `LOCALAPPDATA`が書込不可の管理PCでは、共通の一時directoryへfallbackする。

この方式では、ある測定を停止してleaseが解放された後、同じPlutoを別applicationから
別roleで使用できる。将来、同一PlutoのRX/TX同時使用を明示的にサポートする場合は、
half-duplex state machineとRF safety条件を別途定義すること。

## VSA mode間の選択

Generic VSAで選択したPluto targetをADS-B 1090ES workspaceにも渡す。
これによりmode切替で暗黙に別個体のAuto選択へ戻らない。

## Windows BAT runtime

`_Pluto_Runtime.bat`は`.venv\Scripts`と`.venv\Library\bin`を`PATH`へ追加してから
`import iio`を確認する。activate済みterminalでは起動できるがBATからは
`libiio runtime missing`になる環境差を吸収する。必要ならvenv内、同梱runtime、PATH、
代表的なlibiio install先の順に`libiio.dll`を探索する。

## 検証

- 同一serialへの二重leaseが拒否されること。
- エラーから先行ownerのapplication/roleを取得できること。
- release後に別roleで再取得できること。
- explicit URIからもdescription内serialへ正規化されること。
- RTSA/VSA/VSGの既存Pluto backend/UI testsに回帰がないこと。
