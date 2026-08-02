from __future__ import annotations

import numpy as np
import pytest

from pluto_sa.sdr.trigger import (
    AcquisitionMetadata,
    IQAcquisitionRecord,
    TriggerConfig,
    TriggerEvent,
    TriggerKind,
    power_trigger_dbfs_to_display_dbm,
    power_trigger_display_dbm_to_dbfs,
)


def metadata() -> AcquisitionMetadata:
    return AcquisitionMetadata(
        sample_rate_hz=4_000_000,
        center_freq_hz=2_440_000_000,
        rf_bandwidth_hz=4_000_000,
        gain_db=30,
        source="high_speed_ta",
    )


def test_trigger_config_record_length_includes_trigger_sample() -> None:
    config = TriggerConfig(pretrigger_samples=10, posttrigger_samples=20)
    assert config.record_samples == 31


def test_power_trigger_display_dbm_conversion_round_trips() -> None:
    correction = {
        "iq_full_scale": 2048.0,
        "calibration_offset_db": -62.0,
        "frequency_dependent_offset_db": 1.25,
        "input_correction_db": -3.0,
    }

    level_dbfs = power_trigger_display_dbm_to_dbfs(-30.0, **correction)

    assert level_dbfs == pytest.approx(-32.476599, abs=1e-6)
    assert power_trigger_dbfs_to_display_dbm(level_dbfs, **correction) == pytest.approx(
        -30.0
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pretrigger_samples", -1),
        ("posttrigger_samples", -1),
        ("holdoff_samples", -1),
        ("minimum_duration_samples", 0),
        ("hysteresis_db", -0.1),
        ("auto_timeout_samples", 0),
    ],
)
def test_trigger_config_rejects_invalid_sample_domain_values(field, value) -> None:
    with pytest.raises(ValueError):
        TriggerConfig(**{field: value})


def test_acquisition_record_locates_trigger_on_sample_timeline() -> None:
    config = TriggerConfig(
        kind=TriggerKind.POWER_LEVEL,
        pretrigger_samples=2,
        posttrigger_samples=2,
    )
    event = TriggerEvent(
        stream_id=3,
        sample_index=102,
        sequence=8,
        offset_in_block=4,
        kind=TriggerKind.POWER_LEVEL,
        measured_value=-12.5,
    )
    record = IQAcquisitionRecord(
        stream_id=3,
        start_sample_index=100,
        trigger_sample_index=102,
        iq=np.arange(5, dtype=np.complex64),
        trigger=event,
        config=config,
        metadata=metadata(),
    )

    assert record.trigger_sample_offset == 2
    assert record.end_sample_index == 105
    assert record.is_contiguous is True
    with pytest.raises(ValueError, match="read-only"):
        record.iq[0] = 1.0 + 1.0j


def test_acquisition_record_rejects_trigger_outside_record() -> None:
    config = TriggerConfig()
    event = TriggerEvent(
        stream_id=1,
        sample_index=20,
        sequence=0,
        offset_in_block=0,
        kind=TriggerKind.FREE_RUN,
    )
    with pytest.raises(ValueError, match="inside"):
        IQAcquisitionRecord(
            stream_id=1,
            start_sample_index=0,
            trigger_sample_index=20,
            iq=np.zeros(4, dtype=np.complex64),
            trigger=event,
            config=config,
            metadata=metadata(),
        )


def test_acquisition_record_requires_exact_configured_length() -> None:
    config = TriggerConfig(pretrigger_samples=2, posttrigger_samples=2)
    event = TriggerEvent(
        stream_id=1,
        sample_index=2,
        sequence=0,
        offset_in_block=2,
        kind=TriggerKind.FREE_RUN,
    )
    with pytest.raises(ValueError, match="record length"):
        IQAcquisitionRecord(
            stream_id=1,
            start_sample_index=0,
            trigger_sample_index=2,
            iq=np.zeros(6, dtype=np.complex64),
            trigger=event,
            config=config,
            metadata=metadata(),
        )
