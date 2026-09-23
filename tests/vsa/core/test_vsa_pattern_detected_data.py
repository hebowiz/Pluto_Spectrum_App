from pathlib import Path
import numpy as np
import pytest
from pluto_vsa.model import IQRecording, ModulationKind, SignalDescription
from pluto_vsa.mapping import BLUETOOTH_HDT_MAPPING
from pluto_vsa.pattern import (
    BitOrdering,
    DemodulationSettings,
    IQPowerTriggerSettings,
    KnownPattern,
    MeasurementFilterMode,
    PatternAnalyzer,
    PatternSearchMode,
    PatternSearchSettings,
    ResultRangeSettings,
    SynchronizationSource,
)
from pluto_vsa.sources import FileIQSource, GeneratedIQSource
from pluto_vsa.session import VSASession


def test_measurement_filter_defaults_to_auto_and_accepts_none() -> None:
    assert DemodulationSettings().measurement_filter is MeasurementFilterMode.AUTO
    assert DemodulationSettings().bit_ordering is BitOrdering.LSB
    assert (
        DemodulationSettings(
            measurement_filter=MeasurementFilterMode.NONE
        ).measurement_filter
        is MeasurementFilterMode.NONE
    )


def test_detected_data_psk_sync_does_not_claim_a_pattern_match() -> None:
    recording, signal = GeneratedIQSource.psk(
        modulation=ModulationKind.DPSK8,
        symbol_count=160,
        seed=20260821,
    )
    session = VSASession(recording=recording, signal=signal)
    session.configure_pattern_analysis(
        PatternSearchSettings(
            pattern=KnownPattern((0,) * 8, name="Not used"),
            mode=PatternSearchMode.ON,
            meas_only_if_pattern_symbols_correct=True,
        ),
        ResultRangeSettings(result_length=120),
        DemodulationSettings(
            coarse_synchronization=SynchronizationSource.DETECTED_DATA,
            measurement_filter=MeasurementFilterMode.NONE,
        ),
    )

    session.analyze()

    result = session.pattern_result
    assert result is not None
    assert result.metadata["synchronization_source"] == "Detected Data"
    assert result.metadata["pattern_match_valid"] is False
    assert result.metadata["pattern_symbol_count"] == 0
    assert result.decoded_symbols.size == 120
    assert result.evm_rms_percent < 10.0
    assert session.pattern_error is None


def test_detected_data_psk_sync_runs_without_pattern_search() -> None:
    recording, signal = GeneratedIQSource.psk(
        modulation=ModulationKind.DPSK8,
        symbol_count=160,
        seed=20260822,
    )
    session = VSASession(recording=recording, signal=signal)
    session.configure_pattern_analysis(
        None,
        ResultRangeSettings(result_length=120),
        DemodulationSettings(
            coarse_synchronization=SynchronizationSource.AUTO,
            measurement_filter=MeasurementFilterMode.NONE,
        ),
        IQPowerTriggerSettings(enabled=False, search_start_offset_symbols=3.0),
    )

    session.analyze()

    result = session.pattern_result
    assert session.pattern_search is None
    assert result is not None
    assert result.metadata["synchronization_source"] == "Detected Data"
    assert result.metadata["pattern_name"] == "Detected Data"
    assert not result.metadata["pattern_match_valid"]
    assert result.decoded_symbols.size == 120
    assert session.pattern_error is None


@pytest.mark.parametrize("result_length", (120, 400))
def test_detected_data_qam_honors_requested_result_length(result_length: int) -> None:
    recording, signal = GeneratedIQSource.psk(
        modulation=ModulationKind.QAM16,
        symbol_count=512,
        seed=20260901,
    )
    result = PatternAnalyzer().detect_data(
        recording,
        signal,
        None,
        ResultRangeSettings(result_length=result_length),
        DemodulationSettings(measurement_filter=MeasurementFilterMode.NONE),
        iq_power_trigger=IQPowerTriggerSettings(enabled=False),
    )

    assert result.decoded_symbols.size == result_length
    assert result.metadata["carrier_symmetry_order"] == 4
    assert result.metadata["detected_psk_interval_start_symbol"] == 0
    assert result.metadata["detected_psk_interval_stop_symbol"] >= result_length
    assert result.evm_rms_percent < 2.0


def test_detected_data_qam_recovers_carrier_without_losing_amplitude() -> None:
    recording, signal = GeneratedIQSource.psk(
        modulation=ModulationKind.QAM16,
        symbol_count=512,
        seed=20260902,
    )
    carrier_offset_hz = 30_000.0
    phase_rad = 0.37
    sample_index = np.arange(recording.sample_count, dtype=np.float64)
    shifted = IQRecording(
        iq=recording.iq
        * np.exp(
            1j
            * (
                phase_rad
                + 2.0
                * np.pi
                * carrier_offset_hz
                * sample_index
                / recording.sample_rate_hz
            )
        ),
        sample_rate_hz=recording.sample_rate_hz,
        metadata=recording.metadata,
    )
    result = PatternAnalyzer().detect_data(
        shifted,
        signal,
        None,
        ResultRangeSettings(result_length=400),
        DemodulationSettings(measurement_filter=MeasurementFilterMode.NONE),
        iq_power_trigger=IQPowerTriggerSettings(enabled=False),
    )

    assert result.decoded_symbols.size == 400
    assert result.carrier_frequency_offset_hz == pytest.approx(
        carrier_offset_hz, abs=10.0
    )
    assert result.evm_rms_percent < 2.0


def test_detected_data_hdt_qam_fixture_is_not_truncated_by_psk_interval() -> None:
    path = Path(__file__).resolve().parents[2] / "data" / "fixtures" / "bluetooth" / "hdt" / "bluetooth_hdt7_5_prbs9_16msps.npz"
    recording = FileIQSource.load(path)
    signal = SignalDescription(
        modulation=ModulationKind.QAM16,
        symbol_rate_hz=2_000_000.0,
        tx_filter="Root Raised Cosine",
        filter_parameter=0.4,
        symbol_mapping=BLUETOOTH_HDT_MAPPING,
    )
    result = PatternAnalyzer().detect_data(
        recording,
        signal,
        None,
        ResultRangeSettings(result_length=500),
        DemodulationSettings(measurement_filter=MeasurementFilterMode.AUTO),
        iq_power_trigger=IQPowerTriggerSettings(enabled=False),
    )

    assert result.decoded_symbols.size == 500
    assert 145 <= result.metadata["detected_psk_interval_start_symbol"] <= 160
    assert result.evm_rms_percent < 5.0


def test_pattern_only_sync_does_not_run_without_pattern_search() -> None:
    recording, signal = GeneratedIQSource.psk(
        modulation=ModulationKind.PI4_DQPSK,
        symbol_count=80,
        seed=20260823,
    )
    session = VSASession(recording=recording, signal=signal)
    session.configure_pattern_analysis(
        None,
        demodulation=DemodulationSettings(
            coarse_synchronization=SynchronizationSource.PATTERN,
        ),
    )

    session.analyze()

    assert session.pattern_result is None
    assert session.pattern_error is None
