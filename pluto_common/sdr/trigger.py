"""Trigger and acquisition-record contracts for analyzer-mode consumers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


DEFAULT_IQ_FULL_SCALE = 2048.0


def power_trigger_dbfs_to_display_dbm(
    level_dbfs: float,
    *,
    iq_full_scale: float = DEFAULT_IQ_FULL_SCALE,
    calibration_offset_db: float = 0.0,
    frequency_dependent_offset_db: float = 0.0,
    input_correction_db: float = 0.0,
) -> float:
    """Convert raw IQ magnitude dBFS into the calibrated display dBm scale."""
    if float(iq_full_scale) <= 0.0:
        raise ValueError("iq_full_scale must be positive")
    return float(
        float(level_dbfs)
        + 20.0 * np.log10(float(iq_full_scale))
        + float(calibration_offset_db)
        + float(frequency_dependent_offset_db)
        + float(input_correction_db)
    )


def power_trigger_display_dbm_to_dbfs(
    level_dbm: float,
    *,
    iq_full_scale: float = DEFAULT_IQ_FULL_SCALE,
    calibration_offset_db: float = 0.0,
    frequency_dependent_offset_db: float = 0.0,
    input_correction_db: float = 0.0,
) -> float:
    """Convert a calibrated display dBm threshold into raw IQ magnitude dBFS."""
    display_zero_dbfs_dbm = power_trigger_dbfs_to_display_dbm(
        0.0,
        iq_full_scale=iq_full_scale,
        calibration_offset_db=calibration_offset_db,
        frequency_dependent_offset_db=frequency_dependent_offset_db,
        input_correction_db=input_correction_db,
    )
    return float(level_dbm) - display_zero_dbfs_dbm


class TriggerKind(str, Enum):
    """Trigger sources supported now or reserved by the common architecture."""

    FREE_RUN = "free_run"
    POWER_LEVEL = "power_level"
    FREQUENCY_MASK = "frequency_mask"


class TriggerSlope(str, Enum):
    RISING = "rising"
    FALLING = "falling"
    EITHER = "either"


class TriggerRunMode(str, Enum):
    """How acquisition behaves after arming and trigger completion."""

    AUTO = "auto"
    NORMAL = "normal"
    SINGLE = "single"


class TriggerRearmMode(str, Enum):
    AUTO_REARM = "auto_rearm"
    STOP_ON_TRIGGER = "stop_on_trigger"


@dataclass(frozen=True)
class TriggerConfig:
    """Sample-domain trigger configuration independent of GUI time units."""

    kind: TriggerKind = TriggerKind.FREE_RUN
    run_mode: TriggerRunMode = TriggerRunMode.AUTO
    rearm_mode: TriggerRearmMode = TriggerRearmMode.AUTO_REARM
    slope: TriggerSlope = TriggerSlope.RISING
    level_dbfs: float = -20.0
    hysteresis_db: float = 1.0
    minimum_duration_samples: int = 1
    holdoff_samples: int = 0
    pretrigger_samples: int = 0
    posttrigger_samples: int = 0
    auto_timeout_samples: int | None = None

    def __post_init__(self) -> None:
        nonnegative = {
            "holdoff_samples": self.holdoff_samples,
            "pretrigger_samples": self.pretrigger_samples,
            "posttrigger_samples": self.posttrigger_samples,
        }
        for name, value in nonnegative.items():
            if int(value) < 0:
                raise ValueError(f"{name} must be non-negative")
        if int(self.minimum_duration_samples) <= 0:
            raise ValueError("minimum_duration_samples must be positive")
        if float(self.hysteresis_db) < 0.0:
            raise ValueError("hysteresis_db must be non-negative")
        if self.auto_timeout_samples is not None and int(self.auto_timeout_samples) <= 0:
            raise ValueError("auto_timeout_samples must be positive when provided")

    @property
    def record_samples(self) -> int:
        """Record length including the trigger sample itself."""
        return int(self.pretrigger_samples) + 1 + int(self.posttrigger_samples)


@dataclass(frozen=True)
class TriggerEvent:
    """A trigger located on the authoritative IQ sample timeline."""

    stream_id: int
    sample_index: int
    sequence: int
    offset_in_block: int
    kind: TriggerKind
    measured_value: float | None = None
    forced: bool = False


@dataclass(frozen=True)
class AcquisitionMetadata:
    """RF/sample settings required to reproduce analysis of a saved record."""

    sample_rate_hz: float
    center_freq_hz: float
    rf_bandwidth_hz: float
    gain_db: float
    source: str
    iq_full_scale: float = DEFAULT_IQ_FULL_SCALE

    def __post_init__(self) -> None:
        if float(self.sample_rate_hz) <= 0.0:
            raise ValueError("sample_rate_hz must be positive")
        if float(self.center_freq_hz) <= 0.0:
            raise ValueError("center_freq_hz must be positive")
        if float(self.rf_bandwidth_hz) <= 0.0:
            raise ValueError("rf_bandwidth_hz must be positive")
        if float(self.iq_full_scale) <= 0.0:
            raise ValueError("iq_full_scale must be positive")


@dataclass(frozen=True)
class IQAcquisitionRecord:
    """Immutable pre/post-trigger time record passed to SA/VSA analysis."""

    stream_id: int
    start_sample_index: int
    trigger_sample_index: int
    iq: np.ndarray
    trigger: TriggerEvent
    config: TriggerConfig
    metadata: AcquisitionMetadata
    discontinuity_reason: str | None = None

    def __post_init__(self) -> None:
        iq = np.asarray(self.iq)
        if iq.ndim != 1 or not np.issubdtype(iq.dtype, np.complexfloating):
            raise ValueError("iq must be a one-dimensional complex array")
        if self.trigger_sample_offset < 0 or self.trigger_sample_offset >= len(iq):
            raise ValueError("trigger sample must be inside the acquisition record")
        if self.trigger.stream_id != self.stream_id:
            raise ValueError("trigger and record stream_id must match")
        if self.trigger.sample_index != self.trigger_sample_index:
            raise ValueError("trigger event and record sample index must match")
        if len(iq) != self.config.record_samples:
            raise ValueError("record length must match trigger configuration")
        owned_iq = np.array(iq, dtype=np.complex64, copy=True)
        owned_iq.flags.writeable = False
        object.__setattr__(self, "iq", owned_iq)

    @property
    def sample_count(self) -> int:
        return int(len(self.iq))

    @property
    def end_sample_index(self) -> int:
        return self.start_sample_index + self.sample_count

    @property
    def trigger_sample_offset(self) -> int:
        return self.trigger_sample_index - self.start_sample_index

    @property
    def is_contiguous(self) -> bool:
        return self.discontinuity_reason is None
