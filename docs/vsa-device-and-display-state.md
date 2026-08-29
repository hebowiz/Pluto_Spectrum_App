# VSA device and IQ-power display state

## Pluto selection persistence

- The VSA stores the selected Pluto as a stable selector such as `serial:<serial>`.
- A selector change is written to `QSettings` immediately, rather than only during a clean application shutdown.
- The stored selector is restored at the next VSA startup and is passed to the first Pluto capture. Measurement configuration files do not contain or overwrite this device selection.
- Device discovery may replace a discovered URI with its stable serial selector and persists the normalized selection.

## IQ Power display floor

- The generic VSA IQ Power display floor is `-120 dBm`.
- Non-finite values (`NaN`, `-inf`) and values below the floor are clamped only in the display path before interpolation and decimation.
- The IQ Power `ViewBox` also has a `yMin=-120 dBm` limit, so Reset, View All, autorange, and manual navigation cannot expose a lower range.
- Trigger detection, DSP measurements, result-summary values, and IQ export continue to use the original unmodified data.
