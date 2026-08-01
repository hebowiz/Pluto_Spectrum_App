from __future__ import annotations

import time
import threading

import pytest

import numpy as np

from pluto_sa.config.spectrum_config import SpectrumConfig
from pluto_sa.sdr import pluto_receiver as receiver_module
from pluto_sa.sdr.pluto_receiver import PlutoReceiver


class FakePluto:
    def __init__(self) -> None:
        self.rx_lo = 0
        self.sample_rate = 0
        self.rx_rf_bandwidth = 0
        self.rx_buffer_size = 4
        self.gain_control_mode_chan0 = "manual"
        self.rx_hardwaregain_chan0 = 0
        self.destroy_count = 0
        self.next_sample = 0

    def rx_destroy_buffer(self) -> None:
        self.destroy_count += 1

    def rx(self) -> np.ndarray:
        time.sleep(0.001)
        count = int(self.rx_buffer_size)
        values = np.arange(
            self.next_sample,
            self.next_sample + count,
            dtype=np.float32,
        )
        self.next_sample += count
        return (values + 1j * -values).astype(np.complex64)


def build_receiver(monkeypatch, *, fft_size: int = 4) -> PlutoReceiver:
    fake = FakePluto()
    monkeypatch.setattr(receiver_module.adi, "Pluto", lambda: fake)
    config = SpectrumConfig(fft_size=fft_size, capture_buffer_blocks=16)
    return PlutoReceiver(config)


def test_synchronous_capture_publishes_common_iq_block(monkeypatch) -> None:
    receiver = build_receiver(monkeypatch)
    cursor = receiver.create_iq_stream_cursor(start="latest")

    captured = receiver.capture_iq_block(4, source="calibration")
    result = receiver.read_iq_stream(cursor)

    assert len(result.blocks) == 1
    block = result.blocks[0]
    assert block is captured
    assert block.source == "calibration"
    assert block.start_sample_index == 0
    assert block.discontinuity_before is True


def test_retune_starts_new_stream_epoch(monkeypatch) -> None:
    receiver = build_receiver(monkeypatch)
    first = receiver.capture_iq_block(4, source="sweep")
    assert first.sample_count == 4
    first_stream_id = receiver.get_iq_stream_stats().stream_id

    receiver.retune_lo(1_000_000, update_config=False)
    cursor = receiver.create_iq_stream_cursor(start="latest")
    receiver.capture_block(4, source="sweep")
    block = receiver.read_iq_stream(cursor).blocks[0]

    assert block.stream_id == first_stream_id + 1
    assert block.start_sample_index == 0
    assert block.discontinuity_before is True


def test_continuous_worker_publishes_to_common_stream(monkeypatch) -> None:
    receiver = build_receiver(monkeypatch)
    cursor = receiver.start(block_size=8, source="high_speed_ta")
    deadline = time.perf_counter() + 1.0
    result = receiver.read_iq_stream(cursor)
    while not result.blocks and time.perf_counter() < deadline:
        time.sleep(0.005)
        result = receiver.read_iq_stream(cursor)
    receiver.stop()

    assert result.overrun is False
    assert len(result.blocks) >= 1
    assert all(block.source == "high_speed_ta" for block in result.blocks)
    assert all(block.sample_count == 8 for block in result.blocks)
    for previous, current in zip(result.blocks, result.blocks[1:]):
        assert current.start_sample_index == previous.end_sample_index


def test_running_worker_rejects_incompatible_second_start(monkeypatch) -> None:
    receiver = build_receiver(monkeypatch)
    receiver.start(block_size=8, source="high_speed_ta")
    try:
        with pytest.raises(RuntimeError, match="different settings"):
            receiver.start(block_size=4, source="continuous")
    finally:
        assert receiver.stop()


def test_stop_keeps_reference_to_worker_that_is_still_blocked(monkeypatch) -> None:
    entered_rx = threading.Event()
    release_rx = threading.Event()

    class BlockingPluto(FakePluto):
        def rx(self) -> np.ndarray:
            entered_rx.set()
            release_rx.wait()
            return super().rx()

    fake = BlockingPluto()
    monkeypatch.setattr(receiver_module.adi, "Pluto", lambda: fake)
    receiver = PlutoReceiver(SpectrumConfig(fft_size=4, capture_buffer_blocks=16))
    receiver.start(block_size=4)
    assert entered_rx.wait(timeout=1.0)

    assert receiver.stop() is False
    assert receiver._rx_thread is not None
    assert receiver._rx_thread.is_alive()
    with pytest.raises(RuntimeError, match="still stopping"):
        receiver.start(block_size=4)

    release_rx.set()
    assert receiver.stop()
