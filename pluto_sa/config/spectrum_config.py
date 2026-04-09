"""Spectrum analyzer configuration."""

from dataclasses import dataclass
from typing import Optional

MAX_DISPLAY_SPAN_HZ = 55_000_000
MIN_INTERNAL_GAIN_DB = 0
MAX_INTERNAL_GAIN_DB = 40


@dataclass
class SpectrumConfig:
    """Configuration values migrated from the prototype."""

    center_freq_hz: int = 2_440_000_000
    center_freq_step_mhz: float = 1.0
    display_span_hz: int = 20_000_000
    use_start_stop_freq: bool = False
    display_start_freq_hz: Optional[int] = None
    display_stop_freq_hz: Optional[int] = None
    guard_ratio: float = 0.04
    fft_size: int = 4096
    rbw_hz: Optional[float] = 1e+6
    calibration_offset_db: float = -62.0
    rx_gain_db: int = 30
    ref_level_dbm: float = 20.0
    display_range_db: float = 100.0
    ext_att_db: float = 30.0
    ext_gain_db: float = 0.0
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
        if self.rx_gain_db < MIN_INTERNAL_GAIN_DB:
            print(f"[WARN] rx_gain_db clipped to {MIN_INTERNAL_GAIN_DB} dB")
            self.rx_gain_db = MIN_INTERNAL_GAIN_DB
        if self.rx_gain_db > MAX_INTERNAL_GAIN_DB:
            print(f"[WARN] rx_gain_db clipped to {MAX_INTERNAL_GAIN_DB} dB")
            self.rx_gain_db = MAX_INTERNAL_GAIN_DB

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

    @property
    def y_max_dbm(self) -> float:
        return self.ref_level_dbm

    @property
    def y_min_dbm(self) -> float:
        return self.ref_level_dbm - self.display_range_db

    @property
    def input_correction_db(self) -> float:
        return self.ext_att_db - self.rx_gain_db - self.ext_gain_db
