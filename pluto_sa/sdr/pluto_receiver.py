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
        self._closed = False
        self._iq_lock = threading.Lock()
        self._sdr_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._rx_thread = None
        self._sweep_config_signature: tuple[int, int, int] | None = None
        self._fast_capture_size: int | None = None
        self._high_speed_backend_enabled = False
        self._high_speed_backend_block_size: int | None = None

        self.received_samples_total = 0
        self._configure_sdr(config)
        self._allocate_capture_buffers(config)

    def _configure_sdr(self, config: SpectrumConfig) -> None:
        self.config = config
        self._sweep_config_signature = None
        self._fast_capture_size = None
        self._high_speed_backend_enabled = False
        self._high_speed_backend_block_size = None
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
            self._fast_capture_size = None
            self._high_speed_backend_enabled = False
            self._high_speed_backend_block_size = None
            with self._sdr_lock:
                self.config = config
                self.sdr.sample_rate = config.sample_rate_hz
                self.sdr.rx_rf_bandwidth = config.rx_bandwidth_hz
                self.sdr.rx_buffer_size = config.rx_buffer_size
                # rx_buffer_size setter does not recreate pyadi internal iio.Buffer.
                # Destroy explicitly so next buffered read reflects the new size.
                try:
                    self.sdr.rx_destroy_buffer()
                except Exception:
                    pass
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
            self._fast_capture_size = None
            self._high_speed_backend_enabled = False
            self._high_speed_backend_block_size = None
            if self._sweep_config_signature == sweep_signature:
                return
            with self._sdr_lock:
                self.sdr.sample_rate = config.sweep_sample_rate_hz
                self.sdr.rx_rf_bandwidth = config.sweep_rf_bandwidth_hz
                self.sdr.gain_control_mode_chan0 = "manual"
                self.sdr.rx_hardwaregain_chan0 = config.rx_gain_db
                try:
                    self.sdr.rx_destroy_buffer()
                except Exception:
                    pass
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
                while total_samples < discard_size:
                    chunk = self.sdr.rx()
                    chunks.append(chunk)
                    total_samples += len(chunk)
            final_returned = discard_size
            self.received_samples_total += final_returned

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
                while total_samples < capture_size:
                    chunk = self.sdr.rx()
                    chunks.append(chunk)
                    total_samples += len(chunk)
                iq = np.concatenate(chunks)[:capture_size].astype(np.complex64, copy=True)
            final_returned = len(iq)
            self.received_samples_total += final_returned

        return iq

    def prepare_fast_capture(self, num_samples: int) -> None:
        """Preconfigure SDR RX block size for High Speed TA fast capture path."""
        capture_size = max(1, int(num_samples))
        with self._sdr_lock:
            if self.sdr.rx_buffer_size != capture_size:
                self.sdr.rx_buffer_size = capture_size
            self._fast_capture_size = capture_size

    def capture_block_fast(self, num_samples: int) -> np.ndarray:
        """Capture one IQ block for High Speed TA with reduced copy/lock overhead."""
        if self._closed:
            return np.empty(0, dtype=np.complex64)
        capture_size = max(1, int(num_samples))
        with self._sdr_lock:
            if self._fast_capture_size != capture_size:
                if self.sdr.rx_buffer_size != capture_size:
                    self.sdr.rx_buffer_size = capture_size
                self._fast_capture_size = capture_size

            iq = np.empty(capture_size, dtype=np.complex64)
            filled = 0
            while filled < capture_size:
                chunk = self.sdr.rx()
                if chunk is None:
                    break
                chunk_len = int(len(chunk))
                if chunk_len <= 0:
                    break
                chunk_arr = (
                    chunk
                    if chunk.dtype == np.complex64
                    else chunk.astype(np.complex64, copy=False)
                )
                copy_len = min(chunk_len, capture_size - filled)
                iq[filled : filled + copy_len] = chunk_arr[:copy_len]
                filled += copy_len

        self.received_samples_total += filled
        if filled >= capture_size:
            return iq
        return iq[:filled]

    def set_gain_db(self, gain_db: int) -> None:
        with self._iq_lock:
            with self._sdr_lock:
                self.sdr.rx_hardwaregain_chan0 = gain_db
            self.config.rx_gain_db = gain_db
            self._sweep_config_signature = None
            self._fast_capture_size = None
            self._high_speed_backend_enabled = False
            self._high_speed_backend_block_size = None

    def start_high_speed_capture_backend(self, block_size: int) -> None:
        """Initialize High Speed TA dedicated buffered RX backend."""
        if self._closed:
            return
        size = max(1, int(block_size))
        with self._sdr_lock:
            self.sdr.rx_buffer_size = size
            try:
                self.sdr.rx_destroy_buffer()
            except Exception:
                pass
            # Prime internal pyadi buffer with desired size.
            _ = self.sdr.rx()
            self._high_speed_backend_enabled = True
            self._high_speed_backend_block_size = size
            self._fast_capture_size = size

    def stop_high_speed_capture_backend(self) -> None:
        """Stop High Speed TA dedicated backend and release pyadi RX buffer."""
        if self._closed:
            return
        with self._sdr_lock:
            self._high_speed_backend_enabled = False
            self._high_speed_backend_block_size = None
            try:
                self.sdr.rx_destroy_buffer()
            except Exception:
                pass

    def capture_block_high_speed_backend(self) -> np.ndarray:
        """Read one High Speed TA block via low-overhead buffered backend."""
        if self._closed:
            return np.empty(0, dtype=np.complex64)
        if not self._high_speed_backend_enabled:
            size = self._high_speed_backend_block_size or self._fast_capture_size or self.config.rx_buffer_size
            return self.capture_block_fast(int(size))

        with self._sdr_lock:
            try:
                raw_channels = self.sdr._rx__rx_buffered_data()
            except Exception:
                size = self._high_speed_backend_block_size or self.config.rx_buffer_size
                return self.capture_block_fast(int(size))

        # Complex Pluto path: two channels [I, Q] expected.
        if (
            not isinstance(raw_channels, list)
            or len(raw_channels) < 2
        ):
            size = self._high_speed_backend_block_size or self.config.rx_buffer_size
            return self.capture_block_fast(int(size))

        i_raw = np.asarray(raw_channels[0])
        q_raw = np.asarray(raw_channels[1])
        n = min(len(i_raw), len(q_raw))
        if n <= 0:
            return np.empty(0, dtype=np.complex64)

        out = np.empty(n, dtype=np.complex64)
        out.real = i_raw[:n].astype(np.float32, copy=False)
        out.imag = q_raw[:n].astype(np.float32, copy=False)
        self.received_samples_total += n
        return out

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
        self.stop_high_speed_capture_backend()
        self._closed = True

    def _rx_worker(self) -> None:
        while not self._stop_event.is_set() and not self._closed:
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
