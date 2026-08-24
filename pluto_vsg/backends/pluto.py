"""Finite ADALM-Pluto transmit backend for generated VSG waveforms."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time

import adi
import iio
import numpy as np

from pluto_vsg.backends.base import BackendCapabilities
from pluto_vsg.engine import GenerationResult


_PLUTO_DAC_SCALE = 2**14 - 1


@dataclass(frozen=True)
class PlutoTransmitSettings:
    center_frequency_hz: float
    sample_rate_hz: float
    rf_bandwidth_hz: float
    hardware_gain_db: float = -30.0
    connection_uri: str | None = None


class PlutoOutputBackend:
    """Own one Pluto TX session and send a transferred waveform once.

    Finite packet repetition is represented in ``GenerationResult.iq``. The
    backend deliberately disables the cyclic IIO buffer, so the requested
    sequence is emitted once and does not continue indefinitely.
    """

    def __init__(self, settings: PlutoTransmitSettings) -> None:
        self.settings = settings
        self._validate_settings(settings)
        self._buffer: np.ndarray | None = None
        self._sample_count = 0
        self._stop_event = threading.Event()
        self._sdr = None

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend_name="ADALM-Pluto",
            supports_rf_output=True,
            maximum_sample_rate_hz=61_440_000.0,
        )

    @staticmethod
    def discover() -> dict[str, str]:
        try:
            contexts = iio.scan_contexts()
        except Exception:
            return {}
        return {
            str(uri): str(description)
            for uri, description in contexts.items()
            if uri.startswith("usb:")
            or (uri.startswith("ip:") and "pluto" in description.lower())
        }

    @staticmethod
    def _validate_settings(settings: PlutoTransmitSettings) -> None:
        if not 520_833.0 <= settings.sample_rate_hz <= 61_440_000.0:
            raise ValueError("Pluto sample rate must be between 520.833 kS/s and 61.44 MS/s")
        if settings.center_frequency_hz <= 0.0:
            raise ValueError("Pluto TX center frequency must be positive")
        if not 200_000.0 <= settings.rf_bandwidth_hz <= 56_000_000.0:
            raise ValueError("Pluto TX RF bandwidth must be between 200 kHz and 56 MHz")
        if not -89.75 <= settings.hardware_gain_db <= 0.0:
            raise ValueError("Pluto TX hardware gain must be between -89.75 dB and 0 dB")

    @staticmethod
    def _resolve_connection_uri(configured_uri: str | None) -> str | None:
        if configured_uri is not None and configured_uri.strip():
            return configured_uri.strip()
        contexts = PlutoOutputBackend.discover()
        usb_uris = sorted(uri for uri in contexts if uri.startswith("usb:"))
        if usb_uris:
            return usb_uris[0]
        ip_uris = sorted(uri for uri in contexts if uri.startswith("ip:"))
        return ip_uris[0] if ip_uris else None

    def transfer(self, result: GenerationResult) -> None:
        iq = np.asarray(result.iq, dtype=np.complex128).reshape(-1)
        if not iq.size:
            raise ValueError("Cannot transmit an empty waveform")
        if not np.all(np.isfinite(iq)):
            raise ValueError("Pluto transmission requires finite IQ samples")
        peak = float(np.max(np.abs(iq)))
        if peak > 1.0 + 1e-6:
            raise ValueError("Pluto transmission requires normalized IQ magnitude <= 1.0")
        if not np.isclose(result.sample_rate_hz, self.settings.sample_rate_hz):
            raise ValueError("Transferred waveform sample rate differs from Pluto TX settings")
        real = np.clip(iq.real, -1.0, 1.0) * _PLUTO_DAC_SCALE
        imag = np.clip(iq.imag, -1.0, 1.0) * _PLUTO_DAC_SCALE
        self._buffer = (real + 1j * imag).astype(np.complex64)
        self._sample_count = int(iq.size)

    def start(self) -> None:
        if self._buffer is None:
            raise RuntimeError("Transfer a waveform before starting Pluto TX")
        uri = self._resolve_connection_uri(self.settings.connection_uri)
        sdr = adi.Pluto(uri=uri) if uri is not None else adi.Pluto()
        self._sdr = sdr
        try:
            sdr.tx_enabled_channels = [0]
            sdr.sample_rate = int(round(self.settings.sample_rate_hz))
            sdr.tx_lo = int(round(self.settings.center_frequency_hz))
            sdr.tx_rf_bandwidth = int(round(self.settings.rf_bandwidth_hz))
            sdr.tx_hardwaregain_chan0 = float(self.settings.hardware_gain_db)
            sdr.tx_cyclic_buffer = False
            if self._stop_event.is_set():
                return
            sdr.tx(self._buffer)

            # A non-cyclic push is finite. Keep the session alive until the
            # corresponding air time has elapsed, while allowing cancellation.
            duration_s = self._sample_count / self.settings.sample_rate_hz
            deadline = time.monotonic() + duration_s
            while not self._stop_event.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                self._stop_event.wait(min(0.05, remaining))
        finally:
            try:
                sdr.tx_destroy_buffer()
            except Exception:
                pass
            self._sdr = None

    def stop(self) -> None:
        """Request cancellation; cleanup is performed by the TX owner thread."""

        self._stop_event.set()
