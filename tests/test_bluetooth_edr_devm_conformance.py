from __future__ import annotations

import numpy as np
import pytest

from pluto_sa.vsa.mapping import BLUETOOTH_EDR_MAPPING, psk_constellation
from pluto_sa.vsa.model import ModulationKind
from pluto_sa.vsa.pattern import _root_raised_cosine_taps
from pluto_sa.vsa.protocol_modes.bluetooth.rf_measurement import (
    BluetoothFMMeasurementTrace,
    BluetoothRFMeasurementFilterProfile,
    BluetoothRFMeasurementResult,
    BluetoothRFTestAccumulator,
    RFTestEligibility,
    RFTestVerdict,
    appendix_c_devm_quantities,
    edr_measurement_filter_response_db,
    edr_measurement_filter_taps,
    measure_edr_devm,
    measure_edr_initial_carrier_frequency,
)
from pluto_sa.vsa.protocol_modes.bluetooth.rf_measurement.edr import (
    _interpolate_complex,
)
from pluto_sa.vsa.profiles.bluetooth_edr import generate_edr_dh1
from pluto_sa.vsa.protocol_modes.bluetooth.model import (
    BluetoothAnalysisProfile,
    analyze_bluetooth_classic_recording,
)


def _ideal_srrc_db(frequency_hz: np.ndarray) -> np.ndarray:
    frequency = np.abs(np.asarray(frequency_hz, dtype=np.float64))
    result = np.full(frequency.shape, -300.0)
    result[frequency <= 300_000.0] = 0.0
    transition = (frequency > 300_000.0) & (frequency < 700_000.0)
    magnitude = np.sqrt(
        0.5
        * (
            1.0
            + np.cos(
                np.pi * (frequency[transition] - 300_000.0) / 400_000.0
            )
        )
    )
    result[transition] = 20.0 * np.log10(magnitude)
    return result


@pytest.mark.parametrize("sample_rate_hz", (4e6, 8e6, 16e6, 32e6))
def test_edr_measurement_filter_meets_software_response_requirements(
    sample_rate_hz: float,
) -> None:
    audit_frequencies = np.asarray(
        [0, 100e3, 250e3, 500e3, 600e3, 650e3, 800e3, 1e6]
    )
    response = edr_measurement_filter_response_db(
        sample_rate_hz, audit_frequencies
    )
    assert response[3] == pytest.approx(-3.0103, abs=0.15)

    passband = np.linspace(0.0, 650_000.0, 2601)
    error = edr_measurement_filter_response_db(
        sample_rate_hz, passband
    ) - _ideal_srrc_db(passband)
    assert np.max(np.abs(error)) <= 0.25

    stopband = np.linspace(800_001.0, sample_rate_hz / 2.0, 8192)
    assert np.max(
        edr_measurement_filter_response_db(sample_rate_hz, stopband)
    ) <= -40.0
    assert edr_measurement_filter_taps(sample_rate_hz).size == (
        20 * int(sample_rate_hz / 1e6) + 1
    )


def test_edr_filter_does_not_change_common_generic_or_hdt_default() -> None:
    # Generic VSA and HDT continue to call the common 10-symbol default.
    assert _root_raised_cosine_taps(8, 0.4).size == 81
    assert edr_measurement_filter_taps(8_000_000.0).size == 161


def test_edr_omega_i_uses_only_consecutive_equal_header_centers() -> None:
    sample_rate_hz = 8_000_000.0
    sps = 8.0
    p0 = 3.37
    start_symbol = 72
    header = np.asarray(
        [0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0],
        dtype=np.uint8,
    )
    selected_local = np.flatnonzero(
        (header[1:-1] == header[:-2]) & (header[1:-1] == header[2:])
    ) + 1
    count = 800
    sample_axis = np.arange(count, dtype=np.float64)
    frequency = 31_000.0 + 50_000.0 * np.sin(
        2.0 * np.pi * 0.03 * sample_axis
    )
    selected_values = header[selected_local]
    expected_centers = p0 + (start_symbol + selected_local + 0.5) * sps
    expected_values = 31_000.0 + 50_000.0 * np.sin(
        2.0 * np.pi * 0.03 * expected_centers
    )
    trace = BluetoothFMMeasurementTrace(
        time_s=np.arange(count) / sample_rate_hz,
        frequency_hz=frequency,
        p0_sample=p0,
        sample_rate_hz=sample_rate_hz,
        symbol_rate_hz=1_000_000.0,
        samples_per_symbol=sps,
        filter_profile=BluetoothRFMeasurementFilterProfile.BR_1M,
    )
    result = measure_edr_initial_carrier_frequency(
        trace,
        header,
        nominal_frequency_hz=2_440_000_000.0,
        start_symbol=start_symbol,
    )
    np.testing.assert_array_equal(
        result.selected_bit_indices, start_symbol + selected_local
    )
    np.testing.assert_array_equal(result.selected_bit_values, selected_values)
    np.testing.assert_allclose(
        result.selected_bit_center_frequency_hz, expected_values, atol=2.0
    )
    expected_one = float(np.mean(expected_values[selected_values == 1]))
    expected_zero = float(np.mean(expected_values[selected_values == 0]))
    assert result.delta_omega_one_hz == pytest.approx(expected_one, abs=2.0)
    assert result.delta_omega_zero_hz == pytest.approx(expected_zero, abs=2.0)
    assert result.error_hz == pytest.approx(
        0.5 * (expected_one + expected_zero), abs=2.0
    )


@pytest.mark.parametrize("fraction", (0.1, 0.2, 0.37))
def test_edr_windowed_sinc_fractional_sampling_accuracy(fraction: float) -> None:
    axis = np.arange(512, dtype=np.float64)
    frequency_cycles_per_sample = 0.08
    waveform = np.exp(2j * np.pi * frequency_cycles_per_sample * axis)
    positions = np.arange(64, 448, 7, dtype=np.float64) + fraction
    expected = np.exp(2j * np.pi * frequency_cycles_per_sample * positions)
    measured = _interpolate_complex(waveform, positions)
    linear = np.interp(positions, axis, waveform.real) + 1j * np.interp(
        positions, axis, waveform.imag
    )
    np.testing.assert_allclose(measured, expected, atol=3e-5)
    assert np.max(np.abs(measured - expected)) < 0.01 * np.max(
        np.abs(linear - expected)
    )


def test_appendix_c_equation_matches_hand_calculation_without_fit() -> None:
    sk = np.asarray([1.0, 1j, -1.0, -1j], dtype=np.complex128)
    zk = np.asarray([1.2 + 0.1j, -0.1 + 0.9j, -0.8 - 0.2j, 0.2 - 1.1j])
    expected_qk = zk * np.conj(sk)
    expected_ek = np.diff(expected_qk)
    denominator = np.sum(np.abs(expected_qk[1:]) ** 2)
    qk, ek, symbol_devm, rms_devm = appendix_c_devm_quantities(zk, sk)
    np.testing.assert_allclose(qk, expected_qk)
    np.testing.assert_allclose(ek, expected_ek)
    np.testing.assert_allclose(
        symbol_devm,
        np.abs(expected_ek) / np.sqrt(denominator / expected_ek.size),
    )
    assert rms_devm == pytest.approx(
        np.sqrt(np.sum(np.abs(expected_ek) ** 2) / denominator)
    )


def test_appendix_c_naturally_rejects_common_gain_and_phase_only() -> None:
    phase = np.linspace(0.0, 5.0, 51)
    sk = np.exp(1j * phase)
    zk = 0.37 * np.exp(0.91j) * sk
    _qk, _ek, symbol_devm, rms_devm = appendix_c_devm_quantities(zk, sk)
    assert rms_devm < 1e-14
    assert np.max(symbol_devm) < 1e-13


def test_appendix_c_preserves_symbol_dependent_amplitude_and_phase_error() -> None:
    phase = np.linspace(0.0, 5.0, 51)
    sk = np.exp(1j * phase)
    impairment = np.where(
        np.arange(51) % 2 == 0,
        1.12 * np.exp(0.08j),
        0.88 * np.exp(-0.08j),
    )
    _qk, _ek, _symbol_devm, rms_devm = appendix_c_devm_quantities(
        sk * impairment, sk
    )
    assert rms_devm > 0.25


def _synthetic_edr_block(
    modulation: ModulationKind,
    *,
    timing_offset_symbols: float,
    residual_frequency_hz: float,
    symbol_rate_error_ppm: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    sample_rate_hz = 16_000_000.0
    sps = 16
    count = 100
    labels = np.arange(count, dtype=np.int16) % modulation.order
    differential = psk_constellation(modulation, BLUETOOTH_EDR_MAPPING)[labels]
    physical = np.concatenate((np.ones(1, dtype=np.complex128), np.cumprod(differential)))
    taps = edr_measurement_filter_taps(sample_rate_hz)
    half = (taps.size - 1) // 2
    margin = half + 4 * sps
    first_reference_center = margin + timing_offset_symbols * sps
    output = np.zeros(margin * 2 + physical.size * sps, dtype=np.complex128)
    tap_axis = np.arange(taps.size, dtype=np.float64) - half
    actual_sps = sps * (1.0 + symbol_rate_error_ppm * 1e-6)
    for index, symbol in enumerate(physical):
        center = first_reference_center + index * actual_sps
        positions = np.arange(output.size, dtype=np.float64) - center
        output += symbol * np.interp(positions, tap_axis, taps, left=0.0, right=0.0)
    sample_axis = np.arange(output.size, dtype=np.float64)
    output *= np.exp(
        2j * np.pi * residual_frequency_hz * sample_axis / sample_rate_hz
    )
    first_sync_center = first_reference_center + sps
    return output, labels, first_sync_center


@pytest.mark.parametrize(
    "modulation", (ModulationKind.PI4_DQPSK, ModulationKind.DPSK8)
)
@pytest.mark.parametrize("timing_offset", (0.1, 0.2, 0.37))
def test_edr_optimizer_recovers_fractional_timing_and_residual_cfo(
    modulation: ModulationKind, timing_offset: float
) -> None:
    iq, labels, first_center = _synthetic_edr_block(
        modulation,
        timing_offset_symbols=timing_offset,
        residual_frequency_hz=23_000.0,
    )
    result = measure_edr_devm(
        iq,
        sample_rate_hz=16_000_000.0,
        symbol_rate_hz=1_000_000.0,
        first_symbol_center_sample=first_center - timing_offset * 16.0,
        decoded_symbols=labels,
        reference_symbols=labels,
        modulation=modulation,
        symbol_mapping=BLUETOOTH_EDR_MAPPING,
        initial_frequency_error_hz=0.0,
        trailer_symbols=0,
    )
    assert len(result.blocks) == 2
    for block in result.blocks:
        assert block.timing_offset_symbols == pytest.approx(timing_offset, abs=0.015)
        # Correction sign is opposite the injected positive frequency rotation.
        assert block.residual_frequency_error_hz == pytest.approx(23_000.0, abs=500.0)
        assert block.rms_devm < 0.015
        assert block.optimizer_boundary_reached is False


def test_edr_optimizer_does_not_fit_symbol_rate_per_block() -> None:
    iq, labels, first_center = _synthetic_edr_block(
        ModulationKind.DPSK8,
        timing_offset_symbols=0.0,
        residual_frequency_hz=0.0,
        symbol_rate_error_ppm=8_000.0,
    )
    result = measure_edr_devm(
        iq,
        sample_rate_hz=16_000_000.0,
        symbol_rate_hz=1_000_000.0,
        first_symbol_center_sample=first_center,
        decoded_symbols=labels,
        reference_symbols=labels,
        modulation=ModulationKind.DPSK8,
        symbol_mapping=BLUETOOTH_EDR_MAPPING,
        initial_frequency_error_hz=0.0,
        trailer_symbols=0,
    )
    assert len(result.blocks) == 2
    assert result.rms_worst is not None and result.rms_worst > 0.03
    assert abs(
        result.blocks[1].timing_offset_symbols
        - result.blocks[0].timing_offset_symbols
    ) > 0.2


def test_known_reference_is_independent_of_received_symbol_decisions() -> None:
    iq, labels, first_center = _synthetic_edr_block(
        ModulationKind.PI4_DQPSK,
        timing_offset_symbols=0.2,
        residual_frequency_hz=4_000.0,
    )
    wrong_decisions = labels.copy()
    wrong_decisions[7] = (wrong_decisions[7] + 1) % 4
    common = dict(
        sample_rate_hz=16_000_000.0,
        symbol_rate_hz=1_000_000.0,
        first_symbol_center_sample=first_center - 0.2 * 16.0,
        reference_symbols=labels,
        modulation=ModulationKind.PI4_DQPSK,
        symbol_mapping=BLUETOOTH_EDR_MAPPING,
        initial_frequency_error_hz=0.0,
        trailer_symbols=0,
    )
    correct = measure_edr_devm(iq, decoded_symbols=labels, **common)
    wrong = measure_edr_devm(iq, decoded_symbols=wrong_decisions, **common)
    assert wrong.rms_worst == pytest.approx(correct.rms_worst, abs=1e-15)
    assert wrong.peak_worst == pytest.approx(correct.peak_worst, abs=1e-15)


@pytest.mark.parametrize(
    ("packet_name", "payload_length"), (("2-DH1", 31), ("3-DH1", 11))
)
def test_known_rf_test_reference_full_path_and_cfo_split(
    packet_name: str, payload_length: int
) -> None:
    generated = generate_edr_dh1(
        packet_name,
        payload_length_bytes=payload_length,
        whitening_enabled=False,
        carrier_frequency_offset_hz=30_000.0,
        edr_residual_frequency_offset_hz=7_000.0,
        snr_db=70.0,
    )
    result = analyze_bluetooth_classic_recording(
        generated.recording,
        profile=BluetoothAnalysisProfile.RF_PHY_TEST,
        lap=0xC6967E,
        uap=0x6B,
        clock_6_1=0x2B,
        whitening_enabled=False,
        expected_edr_rf_test_packet=packet_name,
        result_length=4096,
    )
    measurement = result.metadata["rf_measurements"][0]
    assert measurement.eligibility.eligible is True
    assert measurement.metadata["reference_source"] == "known_rf_test_packet"
    assert measurement.metrics["initial_frequency_error_hz"] == pytest.approx(
        30_000.0, abs=750.0
    )
    assert measurement.metrics["omega0_worst_hz"] == pytest.approx(
        7_000.0, abs=750.0
    )
    combined = measurement.arrays["omega_i_plus_omega0_hz"]
    np.testing.assert_allclose(combined, 37_000.0, atol=1_000.0)
    # RF.TS tester-validation maxima under the prescribed CFO range.
    assert measurement.metrics["rms_devm_worst"] < 0.03
    assert measurement.metrics["peak_devm_worst"] < 0.08


def test_formal_99_percent_verdict_uses_empirical_symbol_yield() -> None:
    # 9,899 / 10,000 symbols meet the 2M 30% limit: just below 99%.
    symbol_devm = np.concatenate((np.full(9_899, 0.1), np.full(101, 0.31)))
    packet = BluetoothRFMeasurementResult(
        "bluetooth.edr",
        RFTestEligibility(True),
        arrays={
            "block_rms_devm": np.full(200, 0.1),
            "block_peak_devm": np.full(200, 0.31),
            "symbol_devm": symbol_devm,
            "omega0_hz": np.zeros(200),
            "omega_i_plus_omega0_hz": np.zeros(200),
        },
        metrics={"initial_frequency_error_hz": 0.0},
        metadata={"modulation": "PI4_DQPSK"},
    )
    accumulator = BluetoothRFTestAccumulator()
    accumulator.add(packet)
    aggregate = accumulator.aggregate_edr()
    assert aggregate.metrics["devm_99_yield"] == pytest.approx(0.9899)
    assert aggregate.verdict is RFTestVerdict.FAIL
