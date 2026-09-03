"""Application-neutral data models for Pluto CAL."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math


class CalibrationState(str, Enum):
    IDLE = "IDLE"
    SIGNAL_CHECK = "SIGNAL_CHECK"
    MEASURE = "MEASURE"
    ADJUST = "ADJUST"
    VERIFY = "VERIFY"
    PERSIST = "PERSIST"
    COMPLETE = "COMPLETE"
    ROLLBACK = "ROLLBACK"
    FAILED = "FAILED"


@dataclass(frozen=True)
class FrequencyCalibrationConfig:
    reference_frequency_hz: float = 2_440_000_000.0
    if_offset_hz: float = 500_000.0
    sample_rate_hz: float = 4_000_000.0
    rx_bandwidth_hz: float = 2_000_000.0
    rx_buffer_size: int = 65_536
    rx_gain_db: float = 30.0
    captures_per_measurement: int = 5
    verification_captures: int = 9
    minimum_snr_db: float = 18.0
    maximum_frequency_spread_hz: float = 30.0
    search_half_width_hz: float = 150_000.0
    maximum_iterations: int = 14
    convergence_error_hz: float = 1.0
    local_initial_step: int = 16
    deterioration_factor: float = 3.0
    deterioration_floor_hz: float = 20.0
    settle_time_s: float = 0.08

    def __post_init__(self) -> None:
        finite_positive = (
            "reference_frequency_hz",
            "if_offset_hz",
            "sample_rate_hz",
            "rx_bandwidth_hz",
            "maximum_frequency_spread_hz",
            "search_half_width_hz",
            "convergence_error_hz",
            "settle_time_s",
        )
        for name in finite_positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        for name in (
            "rx_buffer_size",
            "captures_per_measurement",
            "verification_captures",
            "maximum_iterations",
            "local_initial_step",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if abs(self.if_offset_hz) + self.search_half_width_hz >= (
            self.sample_rate_hz / 2.0
        ):
            raise ValueError("IF search interval must stay inside Nyquist")

    @property
    def rx_lo_hz(self) -> float:
        return self.reference_frequency_hz - self.if_offset_hz


@dataclass(frozen=True)
class CWToneEstimate:
    frequency_hz: float
    snr_db: float
    peak_dbfs: float


@dataclass(frozen=True)
class FrequencyMeasurement:
    xo_correction: int
    measured_if_hz: float
    measured_frequency_hz: float
    frequency_error_hz: float
    frequency_error_ppm: float
    snr_db: float
    spread_hz: float
    capture_frequencies_hz: tuple[float, ...] = ()


@dataclass(frozen=True)
class CalibrationSample:
    iteration: int
    measurement: FrequencyMeasurement


@dataclass(frozen=True)
class FrequencyCalibrationResult:
    state: CalibrationState
    original_xo_correction: int
    best_xo_correction: int
    best_frequency_error_hz: float
    best_frequency_error_ppm: float
    persisted: bool
    verified: bool
    samples: tuple[CalibrationSample, ...] = field(default_factory=tuple)
    message: str = ""
