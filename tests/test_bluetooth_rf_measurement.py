from dataclasses import replace

import numpy as np
import pytest
from scipy.signal import freqz

from pluto_protocol.bluetooth.hdt import HDTRate
from pluto_sa.vsa.model import IQRecording
from pluto_sa.vsa.protocol_modes.bluetooth.model import (
    BluetoothAnalysisProfile,
    analyze_bluetooth_hdt_recording,
)
from pluto_sa.vsa.protocol_modes.bluetooth.rf_measurement import (
    BluetoothRFMeasurementFilterProfile,
    BluetoothRFTestAccumulator,
    BluetoothRFMeasurementResult,
    RFTestEligibility,
    RFTestVerdict,
    measure_burst_power,
    rf_test_channel_filter_taps,
)
from pluto_vsg.engine import BluetoothHDTWaveformEngine
from pluto_vsg.profiles import bluetooth_hdt_fields, bluetooth_hdt_project


@pytest.mark.parametrize(
    ("profile", "sample_rate_hz", "scale"),
    (
        (BluetoothRFMeasurementFilterProfile.BR_1M, 16e6, 1.0),
        (BluetoothRFMeasurementFilterProfile.LE_1M, 16e6, 1.0),
        (BluetoothRFMeasurementFilterProfile.LE_2M, 32e6, 2.0),
    ),
)
def test_sig_measurement_filter_meets_response_anchors(
    profile: BluetoothRFMeasurementFilterProfile,
    sample_rate_hz: float,
    scale: float,
) -> None:
    taps = rf_test_channel_filter_taps(sample_rate_hz, profile)
    frequency_hz, response = freqz(taps, worN=131072, fs=sample_rate_hz)
    response_db = 20.0 * np.log10(np.maximum(np.abs(response), 1e-12))

    passband = response_db[frequency_hz <= 550e3 * scale]
    assert np.max(passband) - np.min(passband) <= 0.5
    assert response_db[np.argmin(np.abs(frequency_hz - 650e3 * scale))] == pytest.approx(-3.0, abs=0.25)
    assert response_db[np.argmin(np.abs(frequency_hz - 1e6 * scale))] <= -14.0
    assert response_db[np.argmin(np.abs(frequency_hz - 2e6 * scale))] <= -44.0


def test_sig_power_window_excludes_burst_edges() -> None:
    iq = np.ones(1000, dtype=np.complex128)
    iq[:100] = 0.1
    iq[-100:] = 0.1
    measured = measure_burst_power(
        iq,
        full_scale=1.0,
        dbfs_to_dbm_offset_db=0.0,
        start_sample=0,
        stop_sample=iq.size,
        central_fraction=0.8,
    )
    assert measured.average_dbm == pytest.approx(0.0, abs=0.02)
    assert measured.start_sample == 99 or measured.start_sample == 100
    assert measured.stop_sample in {900, 901}


def _hdt_recording(payload_length: int = 32) -> tuple[IQRecording, object]:
    base = bluetooth_hdt_project(HDTRate.HDT7_5)
    settings = replace(base.bluetooth_hdt, payload_length_bytes=payload_length)
    project = replace(base, bluetooth_hdt=settings, fields=bluetooth_hdt_fields(settings))
    generated = BluetoothHDTWaveformEngine().generate(project)
    return (
        IQRecording(
            iq=generated.iq,
            sample_rate_hz=generated.sample_rate_hz,
            center_frequency_hz=project.center_frequency_hz,
        ),
        generated,
    )


def _hdt_result(recording: IQRecording):
    return analyze_bluetooth_hdt_recording(
        recording, profile=BluetoothAnalysisProfile.RF_PHY_TEST
    )


def test_hdt_header_evm_keeps_training_amplitude_reference() -> None:
    recording, generated = _hdt_recording()
    baseline = _hdt_result(recording)
    iq = np.array(recording.iq, copy=True)
    start = int(generated.metadata["data_start_sample"]) + 74 * 8
    stop = start + 62 * 8
    iq[start:stop] *= 0.90
    impaired = _hdt_result(replace(recording, iq=iq))

    assert impaired.metadata["hdt_header_evm_rms_percent"] > (
        baseline.metadata["hdt_header_evm_rms_percent"] + 5.0
    )


def test_hdt_payload_evm_keeps_training_amplitude_reference() -> None:
    recording, generated = _hdt_recording()
    baseline = _hdt_result(recording)
    iq = np.array(recording.iq, copy=True)
    start = int(generated.metadata["data_start_sample"]) + (74 + 62 + 2) * 8
    stop = int(generated.metadata["data_stop_sample"])
    iq[start:stop] *= 0.90
    impaired = _hdt_result(replace(recording, iq=iq))

    assert impaired.metadata["hdt_payload_evm_rms_percent"] > (
        baseline.metadata["hdt_payload_evm_rms_percent"] + 5.0
    )


def test_hdt_payload_evm_includes_final_terminating_symbols() -> None:
    recording, generated = _hdt_recording(payload_length=0)
    baseline = _hdt_result(recording)
    iq = np.array(recording.iq, copy=True)
    stop = int(generated.metadata["data_stop_sample"])
    iq[stop - 16 : stop] *= -1.0
    impaired = _hdt_result(replace(recording, iq=iq))

    assert impaired.metadata["hdt_payload_evm_symbol_count"] == (
        impaired.metadata["hdt_payload_symbol_count"] + 2
    )
    assert impaired.metadata["hdt_payload_evm_rms_percent"] > (
        baseline.metadata["hdt_payload_evm_rms_percent"] + 5.0
    )


@pytest.mark.parametrize("injected_cfo_hz", (10_000.0, -20_000.0))
def test_hdt_training_reference_tracks_cfo_without_hiding_payload_error(
    injected_cfo_hz: float,
) -> None:
    recording, _generated = _hdt_recording()
    sample_axis = np.arange(recording.sample_count, dtype=np.float64)
    rotated = recording.iq * np.exp(
        2j * np.pi * injected_cfo_hz * sample_axis / recording.sample_rate_hz
    )
    result = _hdt_result(replace(recording, iq=rotated))
    estimated_cfo_hz = (
        result.metadata["hdt_reference_phase_step_rad_per_symbol"]
        * 2_000_000.0
        / (2.0 * np.pi)
    )

    assert estimated_cfo_hz == pytest.approx(injected_cfo_hz, abs=150.0)
    assert result.metadata["hdt_payload_evm_rms_percent"] < 2.0


def test_hdt_fractional_timing_reference_tracks_subsample_shift() -> None:
    recording, _generated = _hdt_recording()
    shift_samples = 0.4
    axis = np.arange(recording.sample_count, dtype=np.float64)
    source = axis - shift_samples
    shifted = np.interp(source, axis, recording.iq.real) + 1j * np.interp(
        source, axis, recording.iq.imag
    )
    result = _hdt_result(replace(recording, iq=shifted))

    assert abs(result.metadata["hdt_reference_timing_offset_samples"]) > 0.1
    assert result.metadata["hdt_header_evm_rms_percent"] < 2.0
    assert result.metadata["hdt_payload_evm_rms_percent"] < 2.0


def test_hdt_payload_phase_ramp_is_not_removed_by_generic_resynchronization() -> None:
    recording, generated = _hdt_recording()
    baseline = _hdt_result(recording)
    iq = np.array(recording.iq, copy=True)
    start = int(generated.metadata["data_start_sample"]) + (74 + 62 + 2) * 8
    stop = int(generated.metadata["data_stop_sample"])
    phase = np.linspace(0.0, 0.35, stop - start)
    iq[start:stop] *= np.exp(1j * phase)
    impaired = _hdt_result(replace(recording, iq=iq))

    assert impaired.metadata["hdt_payload_evm_rms_percent"] > (
        baseline.metadata["hdt_payload_evm_rms_percent"] + 8.0
    )


def test_rf_accumulator_withholds_edr_verdict_until_200_blocks() -> None:
    packet = BluetoothRFMeasurementResult(
        "bluetooth.edr",
        RFTestEligibility(True),
        arrays={
            "block_rms_devm": np.full(10, 0.05),
            "block_peak_devm": np.full(10, 0.08),
        },
        metadata={"modulation": "8DPSK"},
    )
    accumulator = BluetoothRFTestAccumulator()
    accumulator.add(packet)
    aggregate = accumulator.aggregate_edr()

    assert aggregate.verdict is RFTestVerdict.NOT_APPLICABLE
    assert aggregate.eligibility.eligible is False
    assert "200 DEVM blocks" in aggregate.eligibility.reasons[-1]


def test_rf_accumulator_applies_rms_99_percent_and_peak_limits() -> None:
    packet = BluetoothRFMeasurementResult(
        "bluetooth.edr",
        RFTestEligibility(True),
        arrays={
            "block_rms_devm": np.full(200, 0.05),
            "block_peak_devm": np.full(200, 0.08),
            "symbol_devm": np.full(10_000, 0.06),
        },
        metadata={"modulation": "PI4_DQPSK"},
    )
    accumulator = BluetoothRFTestAccumulator()
    accumulator.add(packet)
    aggregate = accumulator.aggregate_edr()

    assert aggregate.eligibility.eligible is True
    assert aggregate.verdict is RFTestVerdict.PASS
    assert aggregate.metrics["devm_99_percentile"] == pytest.approx(0.06)
