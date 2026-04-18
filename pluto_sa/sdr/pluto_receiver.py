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
        self._iq_lock = threading.Lock()
        self._sdr_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._rx_thread = None
        self._sweep_config_signature: tuple[int, int, int] | None = None

        self.received_samples_total = 0
        self._configure_sdr(config)
        self._allocate_capture_buffers(config)

    def _configure_sdr(self, config: SpectrumConfig) -> None:
        self.config = config
        self._sweep_config_signature = None
        with self._sdr_lock:
            self.sdr.rx_lo = config.center_freq_hz
            self.sdr.sample_rate = config.sample_rate_hz
            self.sdr.rx_rf_bandwidth = config.rx_bandwidth_hz
            self.sdr.rx_buffer_size = config.rx_buffer_size
            self.sdr.gain_control_mode_chan0 = "manual"
            self.sdr.rx_hardwaregain_chan0 = config.rx_gain_db

    def _allocate_capture_buffers(self, config: SpectrumConfig) -> None:
        self._capture_block_size = config.rx_buffer_size
        self._capture_buffer_size = config.rx_buffer_size * config.capture_buffer_blocks
        self._iq_ring_buffer = np.zeros(self._capture_buffer_size, dtype=np.complex64)
        self._write_index = 0
        self._stored_samples = 0

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

    def retune_lo(self, center_freq_hz: int, *, update_config: bool = True) -> None:
        with self._iq_lock:
            with self._sdr_lock:
                self.sdr.rx_lo = center_freq_hz
            if update_config:
                self.config.center_freq_hz = center_freq_hz

    def get_current_lo_hz(self) -> int:
        with self._iq_lock:
            with self._sdr_lock:
                return int(self.sdr.rx_lo)

    def reconfigure_span(self, config: SpectrumConfig) -> None:
        with self._iq_lock:
            self._sweep_config_signature = None
            with self._sdr_lock:
                self.config = config
                self.sdr.sample_rate = config.sample_rate_hz
                self.sdr.rx_rf_bandwidth = config.rx_bandwidth_hz
                self.sdr.rx_buffer_size = config.rx_buffer_size
            self._allocate_capture_buffers(config)
            self.received_samples_total = 0

    def invalidate_sweep_configuration(self) -> None:
        with self._iq_lock:
            self._sweep_config_signature = None

    def configure_for_sweep(self, config: SpectrumConfig) -> None:
        """Apply the fixed SDR settings required by Sweep SA."""
        sweep_signature = (
            int(config.sweep_sample_rate_hz),
            int(config.sweep_rf_bandwidth_hz),
            int(config.rx_gain_db),
        )
        with self._iq_lock:
            self.config = config
            if self._sweep_config_signature == sweep_signature:
                return
            with self._sdr_lock:
                self.sdr.sample_rate = config.sweep_sample_rate_hz
                self.sdr.rx_rf_bandwidth = config.sweep_rf_bandwidth_hz
                self.sdr.gain_control_mode_chan0 = "manual"
                self.sdr.rx_hardwaregain_chan0 = config.rx_gain_db
            self._sweep_config_signature = sweep_signature

    def discard_block(self, num_samples: int) -> int:
        """Read and discard one SDR block for post-retune flushing."""
        discard_size = max(1, int(num_samples))

        with self._iq_lock:
            with self._sdr_lock:
                if self.sdr.rx_buffer_size != discard_size:
                    self.sdr.rx_buffer_size = discard_size
                chunks = []
                total_samples = 0
                raw_lengths: list[int] = []
                while total_samples < discard_size:
                    chunk = self.sdr.rx()
                    raw_lengths.append(len(chunk))
                    chunks.append(chunk)
                    total_samples += len(chunk)
            raw_returned = total_samples
            final_returned = discard_size
            self.received_samples_total += final_returned

        print(
            "ReceiverDiscard "
            f"requested={discard_size} "
            f"configured={discard_size} "
            f"raw_returned={raw_returned} "
            f"final_returned={final_returned} "
            f"chunks={raw_lengths}"
        )
        return final_returned

    def capture_block(self, num_samples: int) -> np.ndarray:
        """Capture one IQ block without starting the streaming worker."""
        capture_size = max(1, int(num_samples))

        with self._iq_lock:
            with self._sdr_lock:
                if self.sdr.rx_buffer_size != capture_size:
                    self.sdr.rx_buffer_size = capture_size
                chunks = []
                total_samples = 0
                raw_lengths: list[int] = []
                while total_samples < capture_size:
                    chunk = self.sdr.rx()
                    raw_lengths.append(len(chunk))
                    chunks.append(chunk)
                    total_samples += len(chunk)
                iq = np.concatenate(chunks)[:capture_size].astype(np.complex64, copy=True)
            raw_returned = total_samples
            final_returned = len(iq)
            self.received_samples_total += final_returned

        print(
            "ReceiverCapture "
            f"requested={capture_size} "
            f"configured={capture_size} "
            f"raw_returned={raw_returned} "
            f"final_returned={final_returned} "
            f"chunks={raw_lengths}"
        )
        return iq

    def set_gain_db(self, gain_db: int) -> None:
        with self._iq_lock:
            with self._sdr_lock:
                self.sdr.rx_hardwaregain_chan0 = gain_db
            self.config.rx_gain_db = gain_db
            self._sweep_config_signature = None

    def reconfigure(self, config: SpectrumConfig) -> None:
        was_running = self._rx_thread is not None and self._rx_thread.is_alive()
        self.stop()

        with self._iq_lock:
            self._configure_sdr(config)
            self._allocate_capture_buffers(config)
            self.received_samples_total = 0

        if was_running:
            self.start()

    def close(self) -> None:
        self.stop()
        del self.sdr

    def _rx_worker(self) -> None:
        while not self._stop_event.is_set():
            with self._sdr_lock:
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
