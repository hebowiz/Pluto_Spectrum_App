from dataclasses import replace

import numpy as np
import pytest
from scipy.signal import freqz

from pluto_protocol.bluetooth.hdt import HDTRate, map_hdt_symbols
from pluto_sa.vsa.model import IQRecording
from pluto_sa.vsa.protocol_modes.bluetooth.model import (
    BluetoothAnalysisProfile,
    analyze_bluetooth_hdt_recording,
)
from pluto_sa.vsa.protocol_modes.bluetooth.rf_measurement import (
    BluetoothRFMeasurementFilterProfile,
    BluetoothFMMeasurementTrace,
    BluetoothRFTestAccumulator,
    BluetoothRFMeasurementResult,
    RFTestEligibility,
    RFTestVerdict,
    measure_burst_power,
    measure_observed_fsk_deviation,
    measure_pre_packet_emissions,
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


def test_observed_fsk_deviation_removes_cfo_without_payload_pattern() -> None:
    sample_rate_hz = 8e6
    symbol_rate_hz = 1e6
    carrier_offset_hz = 10_000.0
    frequency_hz = np.repeat(
        np.asarray((110_000.0, -190_000.0, 110_000.0, -190_000.0)), 8
    )
    trace = BluetoothFMMeasurementTrace(
        time_s=np.arange(frequency_hz.size) / sample_rate_hz,
        frequency_hz=frequency_hz,
        p0_sample=0.0,
        sample_rate_hz=sample_rate_hz,
        symbol_rate_hz=symbol_rate_hz,
        samples_per_symbol=8.0,
        filter_profile=BluetoothRFMeasurementFilterProfile.BR_1M,
    )

    measured = measure_observed_fsk_deviation(
        trace,
        payload_start_symbol=0,
        payload_symbol_count=4,
        carrier_frequency_offset_hz=carrier_offset_hz,
    )

    assert measured.mean_abs_hz == pytest.approx(150_000.0)
    assert measured.percentile_99_9_hz == pytest.approx(200_000.0)
    assert measured.max_abs_hz == pytest.approx(200_000.0)
    assert measured.deviations_hz.size == 4 * 32


def test_hdt_pre_packet_emissions_uses_linear_power_thresholds() -> None:
    iq = np.zeros(100, dtype=np.complex128)
    iq[17] = np.sqrt(1e-3)
    iq[18] = np.sqrt(0.5)
    iq[19] = np.sqrt(0.7)
    iq[20:] = 1.0

    duration = measure_pre_packet_emissions(
        iq,
        packet_start_sample=20,
        packet_stop_sample=100,
        sample_rate_hz=1e6,
    )

    assert duration == pytest.approx(3e-6)


def test_hdt_pre_packet_emissions_requires_pre_packet_idle() -> None:
    assert measure_pre_packet_emissions(
        np.ones(100, dtype=np.complex128),
        packet_start_sample=20,
        packet_stop_sample=100,
        sample_rate_hz=1e6,
    ) is None


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


def test_hdt_payload_evm_uses_fixed_reencoded_reference() -> None:
    recording, generated = _hdt_recording(payload_length=73)
    result = _hdt_result(recording)
    evm = result.metadata["hdt_evm_result"]
    expected = map_hdt_symbols(
        generated.metadata["coded_payload_bits"], HDTRate.HDT7_5
    )[: evm.payload_reference_symbols.size]

    np.testing.assert_array_equal(evm.payload_reference_symbols, expected)
    assert result.metadata["hdt_payload_reference_source"] == (
        "decoded_reencoded_bits"
    )


def test_hdt_payload_evm_holds_final_terminating_symbols_separately() -> None:
    recording, generated = _hdt_recording(payload_length=0)
    baseline = _hdt_result(recording)
    iq = np.array(recording.iq, copy=True)
    stop = int(generated.metadata["data_stop_sample"])
    iq[stop - 16 : stop] *= -1.0
    impaired = _hdt_result(replace(recording, iq=iq))

    assert impaired.metadata["hdt_payload_evm_symbol_count"] == (
        impaired.metadata["hdt_payload_symbol_count"]
    )
    baseline_evm = baseline.metadata["hdt_evm_result"]
    impaired_evm = impaired.metadata["hdt_evm_result"]
    assert baseline_evm.terminating_measured_symbols.size == 2
    assert baseline_evm.terminating_reference_symbols.size == 2
    baseline_error = np.linalg.norm(
        baseline_evm.terminating_measured_symbols
        - baseline_evm.terminating_reference_symbols
    )
    impaired_error = np.linalg.norm(
        impaired_evm.terminating_measured_symbols
        - impaired_evm.terminating_reference_symbols
    )
    assert impaired_error > baseline_error + 1.0
    rf_measurement = impaired.metadata["rf_measurements"][0]
    assert rf_measurement.metadata["terminating_symbols_included"] is False
    assert rf_measurement.metadata["terminating_symbols_held_separately"] is True


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


def test_hdt_payload_phase_and_cfo_fit_is_independent_of_generic_display() -> None:
    recording, generated = _hdt_recording()
    baseline = _hdt_result(recording)
    iq = np.array(recording.iq, copy=True)
    start = int(generated.metadata["data_start_sample"]) + (74 + 62 + 2) * 8
    stop = int(generated.metadata["data_stop_sample"])
    phase = np.linspace(0.0, 0.35, stop - start)
    iq[start:stop] *= np.exp(1j * phase)
    impaired = _hdt_result(replace(recording, iq=iq))

    assert impaired.metadata["hdt_payload_evm_rms_percent"] < (
        baseline.metadata["hdt_payload_evm_rms_percent"] + 1.0
    )
    assert abs(
        impaired.metadata["hdt_payload_carrier_error_hz"]
        - baseline.metadata["hdt_payload_carrier_error_hz"]
    ) > 100.0


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


def test_rf_accumulator_allows_irreversible_devm_failure_before_200_blocks() -> None:
    packet = BluetoothRFMeasurementResult(
        "bluetooth.edr",
        RFTestEligibility(True),
        arrays={
            "block_rms_devm": np.full(10, 0.21),
            "block_peak_devm": np.full(10, 0.22),
            "symbol_devm": np.full(500, 0.21),
        },
        metadata={"modulation": "PI4_DQPSK"},
    )
    accumulator = BluetoothRFTestAccumulator()
    accumulator.add(packet)

    aggregate = accumulator.aggregate_edr()

    assert aggregate.metrics["block_count"] == 10
    assert aggregate.verdict is RFTestVerdict.FAIL


def test_rf_accumulator_combines_fsk_patterns_and_all_delta_f2_samples() -> None:
    accumulator = BluetoothRFTestAccumulator()
    accumulator.add(
        BluetoothRFMeasurementResult(
            "bluetooth.fsk",
            RFTestEligibility(True),
            metrics={"delta_f1_avg_hz": 160_000.0, "delta_f2_avg_hz": None},
            metadata={"payload_pattern": "11110000", "filter_profile": "br_1m"},
        )
    )
    accumulator.add(
        BluetoothRFMeasurementResult(
            "bluetooth.fsk",
            RFTestEligibility(True),
            metrics={"delta_f1_avg_hz": None, "delta_f2_avg_hz": 140_000.0},
            arrays={"delta_f2_max_hz": np.array([100_000.0, 120_000.0, 130_000.0])},
            metadata={"payload_pattern": "10101010", "filter_profile": "br_1m"},
        )
    )

    aggregate = accumulator.aggregate_fsk()

    assert aggregate.eligibility.eligible is True
    assert aggregate.metrics["delta_f1_packet_count"] == 1
    assert aggregate.metrics["delta_f2_packet_count"] == 1
    assert aggregate.metrics["delta_f1_avg_hz"] == pytest.approx(160_000.0)
    assert aggregate.metrics["delta_f2_avg_hz"] == pytest.approx(140_000.0)
    assert aggregate.metrics["delta_f2_ratio"] == pytest.approx(0.875)
    assert aggregate.metrics["delta_f2_p999_floor_hz"] == pytest.approx(100_040.0)
    assert aggregate.verdict is RFTestVerdict.FAIL
    np.testing.assert_array_equal(
        aggregate.arrays["delta_f2_max_hz"],
        np.array([100_000.0, 120_000.0, 130_000.0]),
    )


def test_rf_accumulator_tracks_hdt_packet_evm_qualification() -> None:
    accumulator = BluetoothRFTestAccumulator()
    for header_pass, payload_pass, header_db, payload_db in (
        (True, True, -25.0, -28.0),
        (True, False, -24.0, -21.0),
    ):
        accumulator.add(
            BluetoothRFMeasurementResult(
                "bluetooth.hdt.evm",
                RFTestEligibility(True),
                RFTestVerdict.PASS if payload_pass else RFTestVerdict.FAIL,
                metrics={
                    "header_rms_evm_db": header_db,
                    "payload_rms_evm_db": payload_db,
                    "header_pass": header_pass,
                    "payload_pass": payload_pass,
                },
            )
        )

    aggregate = accumulator.aggregate_hdt(required_packets=2)

    assert aggregate.verdict is RFTestVerdict.FAIL
    assert aggregate.metrics["eligible_packet_count"] == 2
    assert aggregate.metrics["header_pass_count"] == 2
    assert aggregate.metrics["payload_pass_count"] == 1
    assert aggregate.metrics["first_failure"] == 2
    assert aggregate.metrics["worst_payload_rms_evm_db"] == -21.0


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
