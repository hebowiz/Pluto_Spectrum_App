"""Spectrum analyzer configuration."""

from dataclasses import dataclass
import os
from typing import Optional

from pluto_sa.modes.analyzer_mode import AnalyzerMode
from pluto_sa.config.input_frontend import InputPowerCorrection

MAX_DISPLAY_SPAN_HZ = 55_000_000
MIN_INTERNAL_GAIN_DB = 0
MAX_INTERNAL_GAIN_DB = 40
WIDEBAND_CHUNK_WIDTH_OPTIONS_HZ = (
    10_000_000,
    20_000_000,
    30_000_000,
    40_000_000,
)


@dataclass
class SpectrumConfig:
    """Configuration values migrated from the prototype."""

    # Common analyzer settings
    analyzer_mode: AnalyzerMode = AnalyzerMode.REALTIME_SA
    center_freq_hz: int = 2_440_000_000
    center_freq_step_mhz: float = 1.0
    display_span_hz: int = 20_000_000
    use_start_stop_freq: bool = False
    display_start_freq_hz: Optional[int] = None
    display_stop_freq_hz: Optional[int] = None
    guard_ratio: float = 0.04
    rbw_hz: Optional[float] = 1e+6
    calibration_offset_db: float = -62.0
    rx_gain_db: int = 30
    ref_level_dbm: float = 20.0
    display_range_db: float = 100.0
    ext_att_db: float = 30.0
    ext_gain_db: float = 0.0
    # zero-mean化処理のON/OFF。TrueでON
    remove_dc_offset: bool = False
    # None: PLUTO_SDR_URI環境変数、次にdirect USBを自動選択する。
    sdr_uri: Optional[str] = None

    # Real-Time SA settings
    fft_size: int = 4096
    update_interval_ms: int = 0
    waterfall_history: int = 300
    persistence_decay_mode: str = "Medium"
    waterfall_decimation: int = 4
    capture_buffer_blocks: int = 512
    realtime_fft_parameter_mode: str = "Auto"
    realtime_min_display_bins: int = 1024
    realtime_overlap_ratio: float = 0.8
    realtime_fft_rate_limit_hz: float = 10_000.0
    realtime_stream_read_blocks: int = 16
    drop_threshold_factor: float = 2.5
    drop_judge_window: int = 30
    wideband_chunk_width_hz: int = 10_000_000

    # Sweep SA settings
    sweep_points: int = 201
    sweep_time_ms: float = 100.0
    sweep_detector_mode: str = "Sample"
    sweep_update_interval_ms: int = 1
    sweep_lo_settle_us: int = 200
    sweep_retune_flush_reads: int = 4
    sweep_flush_samples: int = 256
    sweep_capture_samples_override: Optional[int] = 1024
    sweep_ui_update_interval_points: int = 4
    sweep_profile_logging: bool = False
    sweep_sample_rate_hz: int = 10_000_000
    sweep_rf_bandwidth_hz: int = 20_000_000

    # Time Analyzer settings
    time_analyzer_sample_rate_hz: int = 2_000_000
    time_analyzer_rf_bandwidth_hz: int = 2_000_000
    time_analyzer_time_span_s: float = 0.010
    time_analyzer_display_points: int = 1000
    hsta_trigger_kind: str = "free_run"
    hsta_trigger_run_mode: str = "auto"
    hsta_trigger_slope: str = "rising"
    hsta_trigger_level_dbm: float = -20.0
    hsta_trigger_hysteresis_db: float = 1.0
    hsta_trigger_position_percent: float = 50.0
    hsta_trigger_auto_timeout_s: float = 1.0

    def __post_init__(self) -> None:
        if self.sdr_uri is None:
            self.sdr_uri = os.environ.get("PLUTO_SDR_URI") or None
        if self.analyzer_mode == AnalyzerMode.REALTIME_SA and self.display_span_hz > MAX_DISPLAY_SPAN_HZ:
            self.display_span_hz = MAX_DISPLAY_SPAN_HZ
        if self.rx_gain_db < MIN_INTERNAL_GAIN_DB:
            self.rx_gain_db = MIN_INTERNAL_GAIN_DB
        if self.rx_gain_db > MAX_INTERNAL_GAIN_DB:
            self.rx_gain_db = MAX_INTERNAL_GAIN_DB
        if int(self.wideband_chunk_width_hz) not in WIDEBAND_CHUNK_WIDTH_OPTIONS_HZ:
            self.wideband_chunk_width_hz = WIDEBAND_CHUNK_WIDTH_OPTIONS_HZ[0]
        self.realtime_overlap_ratio = min(
            0.95,
            max(0.0, float(self.realtime_overlap_ratio)),
        )
        self.realtime_fft_rate_limit_hz = max(
            1.0,
            float(self.realtime_fft_rate_limit_hz),
        )
        self.realtime_stream_read_blocks = max(
            1,
            int(self.realtime_stream_read_blocks),
        )
        normalized_fft_mode = str(self.realtime_fft_parameter_mode).strip().lower()
        self.realtime_fft_parameter_mode = (
            "Advanced" if normalized_fft_mode == "advanced" else "Auto"
        )
        self.realtime_min_display_bins = max(16, int(self.realtime_min_display_bins))

    @property
    def sample_rate_hz(self) -> int:
        if self.analyzer_mode in (
            AnalyzerMode.TIME_ANALYZER,
            AnalyzerMode.HIGH_SPEED_TIME_ANALYZER,
        ):
            return int(self.time_analyzer_sample_rate_hz)
        return int(round(self.display_span_hz / (1.0 - 2.0 * self.guard_ratio)))

    @property
    def rx_bandwidth_hz(self) -> int:
        if self.analyzer_mode in (
            AnalyzerMode.TIME_ANALYZER,
            AnalyzerMode.HIGH_SPEED_TIME_ANALYZER,
        ):
            return int(self.time_analyzer_rf_bandwidth_hz)
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
        return self.input_power_correction.input_correction_db

    @property
    def input_power_correction(self) -> InputPowerCorrection:
        """Return the correction contract shared with Pluto VSA capture."""

        return InputPowerCorrection(
            calibration_offset_db=self.calibration_offset_db,
            internal_gain_db=self.rx_gain_db,
            external_attenuation_db=self.ext_att_db,
            external_gain_db=self.ext_gain_db,
        )

    @property
    def sweep_start_freq_hz(self) -> int:
        if self.use_start_stop_freq and self.display_start_freq_hz is not None:
            return self.display_start_freq_hz
        return int(round(self.center_freq_hz - self.display_span_hz / 2.0))

    @property
    def sweep_stop_freq_hz(self) -> int:
        if self.use_start_stop_freq and self.display_stop_freq_hz is not None:
            return self.display_stop_freq_hz
        return int(round(self.center_freq_hz + self.display_span_hz / 2.0))

    @property
    def sweep_step_hz(self) -> float:
        if self.sweep_points <= 1:
            return float(self.display_span_hz)
        return self.display_span_hz / (self.sweep_points - 1)
