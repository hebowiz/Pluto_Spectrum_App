"""PlutoSDR receiver."""

from __future__ import annotations

import threading
from typing import Optional

import adi
import numpy as np

from pluto_sa.config.spectrum_config import SpectrumConfig


class PlutoReceiver:
    """Own PlutoSDR access, streaming, and IQ buffering."""

    def __init__(self, config: SpectrumConfig) -> None:
        self.config = config
        self.sdr = adi.Pluto()

        self.sdr.rx_lo = config.center_freq_hz
        self.sdr.sample_rate = config.sample_rate_hz
        self.sdr.rx_rf_bandwidth = config.rx_bandwidth_hz
        self.sdr.rx_buffer_size = config.rx_buffer_size
        self.sdr.gain_control_mode_chan0 = "manual"
        self.sdr.rx_hardwaregain_chan0 = config.rx_gain_db

        self._iq_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._rx_thread = None

        self._capture_block_size = config.rx_buffer_size
        self._capture_buffer_size = config.rx_buffer_size * config.capture_buffer_blocks
        self._iq_ring_buffer = np.zeros(self._capture_buffer_size, dtype=np.complex64)
        self._write_index = 0
        self._stored_samples = 0

        self.received_samples_total = 0

    def start(self) -> None:
        if self._rx_thread is not None and self._rx_thread.is_alive():
            return

        self._stop_event.clear()
        self._rx_thread = threading.Thread(
            target=self._rx_worker,
            name="pluto-rx-worker",
            daemon=True,
        )
        self._rx_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._rx_thread is not None:
            self._rx_thread.join(timeout=1.0)
            self._rx_thread = None

    def get_latest_block(self) -> Optional[np.ndarray]:
        n = self.config.fft_size

        with self._iq_lock:
            if self._stored_samples < n:
                return None

            start_index = (self._write_index - n) % self._capture_buffer_size
            end_index = start_index + n

            if end_index <= self._capture_buffer_size:
                iq = self._iq_ring_buffer[start_index:end_index].copy()
            else:
                first_part = self._capture_buffer_size - start_index
                iq = np.empty(n, dtype=np.complex64)
                iq[:first_part] = self._iq_ring_buffer[start_index:]
                iq[first_part:] = self._iq_ring_buffer[: end_index % self._capture_buffer_size]

        return iq

    def get_received_sample_count(self) -> int:
        return self.received_samples_total

    def close(self) -> None:
        self.stop()
        del self.sdr

    def _rx_worker(self) -> None:
        while not self._stop_event.is_set():
            iq = self.sdr.rx().astype(np.complex64, copy=False)
            n = len(iq)

            with self._iq_lock:
                end_index = self._write_index + n

                if end_index <= self._capture_buffer_size:
                    self._iq_ring_buffer[self._write_index:end_index] = iq
                else:
                    first_part = self._capture_buffer_size - self._write_index
                    self._iq_ring_buffer[self._write_index:] = iq[:first_part]
                    self._iq_ring_buffer[: end_index % self._capture_buffer_size] = iq[first_part:]

                self._write_index = end_index % self._capture_buffer_size
                self._stored_samples = min(self._stored_samples + n, self._capture_buffer_size)
                self.received_samples_total += n
