"""Spectrum analyzer configuration."""

from dataclasses import dataclass
from typing import Optional

MAX_DISPLAY_SPAN_HZ = 55_000_000


@dataclass
class SpectrumConfig:
    """Configuration values migrated from the prototype."""

    center_freq_hz: int = 2_440_000_000
    display_span_hz: int = 20_000_000
    guard_ratio: float = 0.04
    fft_size: int = 2048
    rbw_hz: Optional[float] = 1e+6
    calibration_offset_db: float = -72.0
    rx_gain_db: int = 40
    update_interval_ms: int = 0
    waterfall_history: int = 300
    waterfall_decimation: int = 4
    capture_buffer_blocks: int = 512
    drop_threshold_factor: float = 2.5
    drop_judge_window: int = 30

    def __post_init__(self) -> None:
        if self.display_span_hz > MAX_DISPLAY_SPAN_HZ:
            print(f"[WARN] display_span clipped to {MAX_DISPLAY_SPAN_HZ / 1e6:.1f} MHz")
            self.display_span_hz = MAX_DISPLAY_SPAN_HZ

    @property
    def sample_rate_hz(self) -> int:
        return int(round(self.display_span_hz / (1.0 - 2.0 * self.guard_ratio)))

    @property
    def rx_bandwidth_hz(self) -> int:
        return self.sample_rate_hz

    @property
    def rx_buffer_size(self) -> int:
        return self.fft_size

    @property
    def bin_width_hz(self) -> float:
        return self.sample_rate_hz / self.fft_size
