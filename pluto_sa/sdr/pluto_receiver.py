"""PlutoSDR receiver."""

from __future__ import annotations

import threading
import time
from typing import Optional

import adi
import iio
import numpy as np

from pluto_common import PlutoDeviceLease, resolve_pluto_uri
from pluto_sa.config.spectrum_config import SpectrumConfig
from pluto_sa.sdr.iq_stream import (
    IQBlock,
    IQReadResult,
    IQStreamBuffer,
    IQStreamCursor,
    IQStreamStats,
)


class PlutoReceiver:
    """Own PlutoSDR access, streaming, and IQ buffering."""

    RX_KERNEL_BUFFER_COUNT = 8

    def __init__(
        self,
        config: SpectrumConfig,
        *,
        owner_application: str = "Pluto RTSA/VSA",
    ) -> None:
        self.config = config
        try:
            contexts = iio.scan_contexts()
        except Exception:
            contexts = {}
        self.connection_uri = resolve_pluto_uri(config.sdr_uri, contexts)
        self._device_lease = PlutoDeviceLease.acquire(
            config.sdr_uri,
            self.connection_uri,
            contexts,
            application=owner_application,
            role="RX",
        )
        self.device_selector = (
            f"serial:{self._device_lease.owner.serial}"
            if self._device_lease.owner.serial
            else (config.sdr_uri or self.connection_uri)
        )
        try:
            self.sdr = (
                adi.Pluto(uri=self.connection_uri)
                if self.connection_uri is not None
                else adi.Pluto()
            )
        except Exception:
            self._device_lease.release()
            raise
        self._closed = False
        self._iq_lock = threading.Lock()
        self._sdr_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._rx_thread: threading.Thread | None = None
        self._capture_max_blocks: int | None = None
        self._sweep_config_signature: tuple[int, int, int] | None = None
        self._stream_source = "continuous"
        self.rx_kernel_buffers_requested = int(self.RX_KERNEL_BUFFER_COUNT)
        self.rx_kernel_buffers_applied: int | None = None
        self.iq_stream = IQStreamBuffer(
            capacity_blocks=max(1, int(config.capture_buffer_blocks))
        )

        self.received_samples_total = 0
        try:
            self._configure_rx_kernel_buffers()
            self._configure_sdr(config)
            self._allocate_capture_buffers(config)
        except Exception:
            self._device_lease.release()
            raise

    @staticmethod
    def _resolve_connection_uri(configured_uri: str | None) -> str | None:
        """Resolve a stable serial selector, explicit URI, or automatic USB."""
        try:
            contexts = iio.scan_contexts()
        except Exception:
            contexts = {}
        return resolve_pluto_uri(configured_uri, contexts)

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

    def _configure_rx_kernel_buffers(self) -> None:
        """Increase DMA queue depth before pyadi creates its first RX buffer.

        libiio v0 refills a userspace buffer synchronously.  Extra kernel DMA
        buffers keep the Pluto producer running while Python converts and
        publishes the previous block.  Backends that do not expose this
        control retain their driver default.
        """

        rx_device = getattr(self.sdr, "_rxadc", None)
        setter = getattr(rx_device, "set_kernel_buffers_count", None)
        if not callable(setter):
            return
        try:
            setter(self.rx_kernel_buffers_requested)
        except Exception:
            return
        self.rx_kernel_buffers_applied = self.rx_kernel_buffers_requested

    def _allocate_capture_buffers(self, config: SpectrumConfig) -> None:
        self._capture_block_size = config.rx_buffer_size

    def start(
        self,
        *,
        block_size: int | None = None,
        source: str = "continuous",
        max_blocks: int | None = None,
        fresh: bool = False,
    ) -> IQStreamCursor:
        resolved_block_size = max(
            1,
            int(self.config.rx_buffer_size if block_size is None else block_size),
        )
        resolved_source = str(source)
        resolved_max_blocks = None if max_blocks is None else int(max_blocks)
        if resolved_max_blocks is not None and resolved_max_blocks <= 0:
            raise ValueError("max_blocks must be positive when provided")
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("receiver is closed")
            if self._rx_thread is not None and self._rx_thread.is_alive():
                if self._stop_event.is_set():
                    raise RuntimeError("previous receive worker is still stopping")
                if (
                    self._capture_block_size != resolved_block_size
                    or self._stream_source != resolved_source
                    or self._capture_max_blocks != resolved_max_blocks
                ):
                    raise RuntimeError(
                        "receive worker is already running with different settings"
                    )
                return self.iq_stream.create_cursor(start="latest")

            with self._iq_lock:
                with self._sdr_lock:
                    buffer_size_changed = (
                        int(self.sdr.rx_buffer_size) != resolved_block_size
                    )
                    if buffer_size_changed:
                        self.sdr.rx_buffer_size = resolved_block_size
                    if fresh or buffer_size_changed:
                        try:
                            self.sdr.rx_destroy_buffer()
                        except Exception:
                            pass
                self._capture_block_size = resolved_block_size
                self._stream_source = resolved_source
                self._capture_max_blocks = resolved_max_blocks
                self.iq_stream.begin_stream(clear=True)
                cursor = self.iq_stream.create_cursor(start="latest")
            self._stop_event.clear()
            self._rx_thread = threading.Thread(
                target=self._rx_worker,
                name="pluto-rx-worker",
                daemon=True,
            )
            self._rx_thread.start()
            return cursor

    def stop(self) -> bool:
        """Request worker shutdown without ever losing track of a live producer."""
        with self._lifecycle_lock:
            self._stop_event.set()
            thread = self._rx_thread
            if thread is None:
                return True
            sample_rate_hz = max(1, int(self.sdr.sample_rate))
            block_duration_s = self._capture_block_size / float(sample_rate_hz)
            thread.join(timeout=max(1.0, min(5.0, block_duration_s * 2.0 + 0.5)))
            if thread.is_alive():
                # Keep the reference: start() must not create a second SDR producer.
                return False
            self._rx_thread = None
            return True

    def is_streaming(self) -> bool:
        """Return whether the common IQ producer thread is actively running."""
        with self._lifecycle_lock:
            return bool(
                self._rx_thread is not None
                and self._rx_thread.is_alive()
                and not self._stop_event.is_set()
            )

    def get_latest_block(self) -> Optional[np.ndarray]:
        return self.iq_stream.latest_samples(self.config.fft_size)

    def create_iq_stream_cursor(self, *, start: str = "latest") -> IQStreamCursor:
        """Create an independent cursor for an analyzer-mode consumer."""
        return self.iq_stream.create_cursor(start=start)

    def read_iq_stream(
        self,
        cursor: IQStreamCursor,
        *,
        max_blocks: int | None = None,
    ) -> IQReadResult:
        """Read common IQ blocks without removing them for other consumers."""
        return self.iq_stream.read(cursor, max_blocks=max_blocks)

    def get_iq_stream_stats(self) -> IQStreamStats:
        return self.iq_stream.stats()

    def get_received_sample_count(self) -> int:
        return self.received_samples_total

    def retune_lo(self, center_freq_hz: int, *, update_config: bool = True) -> None:
        with self._iq_lock:
            with self._sdr_lock:
                self.sdr.rx_lo = center_freq_hz
            if update_config:
                self.config.center_freq_hz = center_freq_hz
            self.iq_stream.begin_stream()

    def get_current_lo_hz(self) -> int:
        with self._iq_lock:
            with self._sdr_lock:
                return int(self.sdr.rx_lo)

    def get_current_sample_rate_hz(self) -> int:
        with self._iq_lock:
            with self._sdr_lock:
                return int(self.sdr.sample_rate)

    def get_current_rf_bandwidth_hz(self) -> int:
        with self._iq_lock:
            with self._sdr_lock:
                return int(self.sdr.rx_rf_bandwidth)

    def reconfigure_span(self, config: SpectrumConfig) -> None:
        with self._iq_lock:
            self._sweep_config_signature = None
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
            self.iq_stream.begin_stream(clear=True)

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
                try:
                    self.sdr.rx_destroy_buffer()
                except Exception:
                    pass
            self._sweep_config_signature = sweep_signature
            self.iq_stream.begin_stream()

    def discard_block(self, num_samples: int) -> int:
        """Read and discard one SDR block for post-retune flushing."""
        discard_size = max(1, int(num_samples))

        with self._iq_lock:
            with self._sdr_lock:
                if self.sdr.rx_buffer_size != discard_size:
                    self.sdr.rx_buffer_size = discard_size
                    try:
                        self.sdr.rx_destroy_buffer()
                    except Exception:
                        pass
                chunks = []
                total_samples = 0
                while total_samples < discard_size:
                    chunk = self.sdr.rx()
                    chunks.append(chunk)
                    total_samples += len(chunk)
            final_returned = discard_size
            self.received_samples_total += final_returned
            # Samples consumed here are intentionally absent from the public
            # stream. The next published block must expose that discontinuity.
            self.iq_stream.mark_discontinuity()

        return final_returned

    def capture_iq_block(
        self,
        num_samples: int,
        *,
        source: str = "capture",
        fresh: bool = False,
    ) -> IQBlock:
        """Synchronously capture and publish one common IQ block.

        ``fresh`` recreates the IIO RX buffer before reading so a finite
        acquisition starts after the request instead of draining samples that
        accumulated in a reusable kernel/USB buffer.
        """
        capture_size = max(1, int(num_samples))

        with self._iq_lock:
            capture_started_at = time.perf_counter()
            with self._sdr_lock:
                buffer_size_changed = self.sdr.rx_buffer_size != capture_size
                if buffer_size_changed:
                    self.sdr.rx_buffer_size = capture_size
                if fresh or buffer_size_changed:
                    try:
                        self.sdr.rx_destroy_buffer()
                    except Exception:
                        pass
                    self.iq_stream.begin_stream()
                chunks = []
                total_samples = 0
                while total_samples < capture_size:
                    chunk = self.sdr.rx()
                    chunks.append(chunk)
                    total_samples += len(chunk)
                iq = np.concatenate(chunks)[:capture_size].astype(np.complex64, copy=True)
            capture_elapsed_s = time.perf_counter() - capture_started_at
            final_returned = len(iq)
            self.received_samples_total += final_returned
            block = self.iq_stream.publish(
                iq,
                timestamp_s=time.perf_counter(),
                source=source,
                capture_elapsed_s=capture_elapsed_s,
            )

        return block

    def capture_block(
        self,
        num_samples: int,
        *,
        source: str = "capture",
    ) -> np.ndarray:
        """Compatibility wrapper returning only IQ samples."""
        return self.capture_iq_block(num_samples, source=source).iq

    def set_gain_db(self, gain_db: int) -> None:
        with self._iq_lock:
            with self._sdr_lock:
                self.sdr.rx_hardwaregain_chan0 = gain_db
            self.config.rx_gain_db = gain_db
            self._sweep_config_signature = None
            self.iq_stream.begin_stream()

    def reconfigure(self, config: SpectrumConfig) -> None:
        was_running = self._rx_thread is not None and self._rx_thread.is_alive()
        if not self.stop():
            raise RuntimeError("receive worker did not stop before reconfiguration")

        with self._iq_lock:
            self._configure_sdr(config)
            self._allocate_capture_buffers(config)
            self.received_samples_total = 0
            self.iq_stream.begin_stream(clear=True)

        if was_running:
            self.start()

    def close(self) -> None:
        self._closed = True
        if not self.stop():
            return
        with self._sdr_lock:
            try:
                self.sdr.rx_destroy_buffer()
            except Exception:
                pass
            self._device_lease.release()

    def _rx_worker(self) -> None:
        published_blocks = 0
        while not self._stop_event.is_set() and not self._closed:
            with self._iq_lock:
                capture_started_at = time.perf_counter()
                with self._sdr_lock:
                    iq = self.sdr.rx().astype(np.complex64, copy=False)
                capture_elapsed_s = time.perf_counter() - capture_started_at
                n = len(iq)
                self.iq_stream.publish(
                    iq,
                    timestamp_s=time.perf_counter(),
                    source=self._stream_source,
                    capture_elapsed_s=capture_elapsed_s,
                )
                self.received_samples_total += n
                published_blocks += 1
                if (
                    self._capture_max_blocks is not None
                    and published_blocks >= self._capture_max_blocks
                ):
                    break
