"""Finite ADALM-Pluto transmit backend for generated VSG waveforms."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time

import adi
import iio
import numpy as np

from pluto_common import discover_pluto_devices, resolve_pluto_uri
from pluto_vsg.backends.base import BackendCapabilities
from pluto_vsg.engine import GenerationResult


_PLUTO_DAC_SCALE = 2**14 - 1
_PLUTO_MUTED_GAIN_DB = -89.75
_DEFAULT_LEAD_IN_GUARD_S = 0.010
_DEFAULT_STOP_GUARD_S = 0.100
_MAX_STOP_SETTLE_S = 0.005


@dataclass(frozen=True)
class PlutoTransmitSettings:
    center_frequency_hz: float
    sample_rate_hz: float
    rf_bandwidth_hz: float
    hardware_gain_db: float = -30.0
    connection_uri: str | None = None
    lead_in_guard_s: float = _DEFAULT_LEAD_IN_GUARD_S
    stop_guard_s: float = _DEFAULT_STOP_GUARD_S


class PlutoOutputBackend:
    """Send one finite waveform schedule through a guarded cyclic buffer.

    Finite packet repetition is represented in ``GenerationResult.iq``. A
    short non-cyclic Pluto buffer can underrun before the host tears down the
    TX path, so the backend wraps the requested samples in zero-IQ guards and
    uses one cyclic superframe. Normal cleanup occurs inside the trailing
    guard, before the superframe can wrap back to its first packet.
    """

    def __init__(self, settings: PlutoTransmitSettings) -> None:
        self.settings = settings
        self._validate_settings(settings)
        self._buffer: np.ndarray | None = None
        self._superframe: np.ndarray | None = None
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
    def discover_devices():
        return discover_pluto_devices(PlutoOutputBackend.discover())

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
        if settings.lead_in_guard_s < 0.0:
            raise ValueError("Pluto TX lead-in guard must not be negative")
        if settings.stop_guard_s < 0.010:
            raise ValueError("Pluto TX stop guard must be at least 10 ms")

    @staticmethod
    def _resolve_connection_uri(configured_uri: str | None) -> str | None:
        return resolve_pluto_uri(configured_uri, PlutoOutputBackend.discover())

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
        lead_in_count = int(
            round(self.settings.lead_in_guard_s * self.settings.sample_rate_hz)
        )
        stop_guard_count = int(
            round(self.settings.stop_guard_s * self.settings.sample_rate_hz)
        )
        self._superframe = np.concatenate(
            (
                np.zeros(lead_in_count, dtype=np.complex64),
                self._buffer,
                np.zeros(stop_guard_count, dtype=np.complex64),
            )
        )

    @property
    def waveform_duration_s(self) -> float:
        return self._sample_count / self.settings.sample_rate_hz

    @property
    def superframe_duration_s(self) -> float:
        if self._superframe is None:
            return 0.0
        return self._superframe.size / self.settings.sample_rate_hz

    @staticmethod
    def _mute_and_stop(sdr) -> None:
        """Mute RF first, then remove DMA data and select the DAC zero source."""

        try:
            sdr.tx_hardwaregain_chan0 = _PLUTO_MUTED_GAIN_DB
        except Exception:
            pass
        try:
            sdr.tx_destroy_buffer()
        except Exception:
            pass
        try:
            sdr.tx_enabled_channels = []
            sdr.tx()
        except Exception:
            pass

    def start(self) -> None:
        if self._buffer is None or self._superframe is None:
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
            sdr.tx_cyclic_buffer = True
            if self._stop_event.is_set():
                return
            sdr.tx(self._superframe)

            # Enter the trailing zero-IQ region before cleanup. The additional
            # settle interval absorbs host scheduling jitter while remaining
            # well inside the stop guard and therefore before cyclic wrap.
            settle_s = min(_MAX_STOP_SETTLE_S, self.settings.stop_guard_s / 2.0)
            active_end_s = self.settings.lead_in_guard_s + self.waveform_duration_s
            deadline = time.monotonic() + active_end_s + settle_s
            while not self._stop_event.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                self._stop_event.wait(min(0.05, remaining))
        finally:
            self._mute_and_stop(sdr)
            self._sdr = None

    def stop(self) -> None:
        """Request cancellation; cleanup is performed by the TX owner thread."""

        self._stop_event.set()
