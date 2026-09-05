import numpy as np
import pytest
from dataclasses import replace
from pathlib import Path

from pluto_protocol import PacketDecodeInput, analyze_packet
from pluto_protocol.dect import dect_p_range
from pluto_sa.vsa.model import IQRecording
from pluto_sa.vsa.protocol_modes.dect import (
    DECT_CARRIER_PLANS,
    DectModulationReference,
    analyze_dect_recording,
    generate_dect_packet,
)
from pluto_sa.vsa.protocol_modes.dect.analysis import _sync_packet
from pluto_sa.vsa.sources import FileIQSource


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("direction", ("RFP", "PP"))
def test_generated_p32_sync_and_rf_measurements(direction: str) -> None:
    recording = generate_dect_packet(
        direction=direction,
        frequency_error_hz=12_500.0,
        symbol_rate_error_ppm=8.0,
        power_dbm=-7.5,
    )
    result = analyze_dect_recording(recording)[0]
    assert result.direction == direction
    assert result.packet_type == "P32"
    assert result.modulation_case == "Case A (00001111)"
    assert result.carrier_test_eligible
    assert result.modulation_test_eligible
    assert result.carrier_error_hz == pytest.approx(12_500.0, abs=500.0)
    assert result.positive_deviation_hz == pytest.approx(288_000.0, abs=2_000.0)
    assert result.negative_deviation_hz == pytest.approx(-288_000.0, abs=2_000.0)
    assert result.symbol_rate_error_ppm == pytest.approx(8.0, abs=1.0)
    assert result.output_power == pytest.approx(-7.5, abs=0.1)
    assert result.output_power_unit == "dBm"
    assert result.power_time_pass is True
    assert result.sync_score > 0.98
    assert result.p0_sample == pytest.approx(
        recording.metadata["expected_p0_sample"], abs=0.6
    )
    carrier_row = next(
        row for row in result.summary_rows
        if row.test_item == "RF Carrier Frequency Accuracy"
    )
    assert carrier_row.result == "MEASURING"


def test_alternating_payload_uses_case_b_deviation_limit() -> None:
    result = analyze_dect_recording(
        generate_dect_packet(payload_pattern="0101")
    )[0]
    assert result.modulation_case == "Case B (0101)"
    assert result.modulation_test_eligible
    row = next(
        row for row in result.summary_rows if row.test_item == "GFSK Modulation Deviation"
    )
    assert "202 kHz" in row.limit


def test_arbitrary_payload_reports_reference_values_without_rf_test_verdict() -> None:
    result = analyze_dect_recording(
        generate_dect_packet(payload_pattern="prbs9")
    )[0]
    assert result.modulation_case == "Observed arbitrary payload"
    assert not result.carrier_test_eligible
    assert not result.modulation_test_eligible
    rows = {row.test_item: row for row in result.summary_rows}
    assert rows["RF Carrier Frequency Accuracy"].result == "N/A"
    assert rows["GFSK Modulation Deviation"].result == "N/A"
    assert "Observed carrier frequency error" in rows
    assert "Observed GFSK deviation" in rows
    reference_items = [
        row.test_item
        for row in result.summary_rows
        if row.section == "Reference Information"
    ]
    packet_index = reference_items.index("Detected packet type")
    assert reference_items[packet_index : packet_index + 3] == [
        "Detected packet type",
        "Observed carrier frequency error",
        "Observed GFSK deviation",
    ]


def test_missing_burst_is_rejected() -> None:
    recording = IQRecording(
        iq=np.zeros(4096, dtype=np.complex64),
        sample_rate_hz=9_216_000.0,
        center_frequency_hz=1_888_704_000.0,
    )
    with pytest.raises(RuntimeError, match="burst"):
        analyze_dect_recording(recording)


def test_carrier_plans_include_etsi_us_japan_and_extended_bands() -> None:
    plans = {plan.plan_id: plan for plan in DECT_CARRIER_PLANS}
    assert plans["etsi_1880"].carriers[0].center_frequency_hz == 1_897_344_000.0
    assert plans["dect_6_us"].carriers[-1].center_frequency_hz == 1_928_448_000.0
    assert len(plans["j_dect"].carriers) == 12
    assert [carrier.channel for carrier in plans["j_dect"].carriers] == [
        "F7",
        "F8",
        "F9",
        "Fa",
        "Fb",
        "F0",
        "F1",
        "F2",
        "F3",
        "F4",
        "F5",
        "F6",
    ]
    assert [
        carrier.center_frequency_hz for carrier in plans["j_dect"].carriers
    ] == [
        1_885_248_000.0,
        1_886_976_000.0,
        1_888_704_000.0,
        1_890_432_000.0,
        1_892_160_000.0,
        1_893_888_000.0,
        1_895_616_000.0,
        1_897_344_000.0,
        1_899_072_000.0,
        1_900_800_000.0,
        1_902_528_000.0,
        1_904_256_000.0,
    ]
    assert {"etsi_ext_1935", "etsi_ext_2010"} <= plans.keys()
    assert len(plans["etsi_ext_1935"].carriers) == 14
    assert plans["etsi_ext_1935"].carriers[-1].center_frequency_hz == 1_959_552_000.0


def test_multiple_generated_bursts_are_independently_detected() -> None:
    first = generate_dect_packet(direction="RFP", frequency_error_hz=4_000.0)
    second = generate_dect_packet(direction="PP", frequency_error_hz=-6_000.0)
    gap = np.zeros(160, dtype=np.complex64)
    combined = replace(first, iq=np.concatenate((first.iq, gap, second.iq)))
    results = analyze_dect_recording(combined)
    assert [result.direction for result in results] == ["RFP", "PP"]
    assert results[0].carrier_error_hz == pytest.approx(4_000.0, abs=500.0)
    assert results[1].carrier_error_hz == pytest.approx(-6_000.0, abs=500.0)


def test_peak_deviation_details_are_retained_per_bit() -> None:
    result = analyze_dect_recording(generate_dect_packet())[0]
    assert result.bit_peak_frequency_hz.shape == result.bits.shape
    assert result.bit_peak_frequency_hz.flags.writeable is False
    assert result.metadata["minimum_measured_deviation_hz"] > 259_000.0
    assert result.metadata["maximum_measured_deviation_hz"] < 403_000.0


@pytest.mark.parametrize("pattern", ("0101", "00110011", "00001111"))
def test_cts60_trace_preserves_pattern_dependent_fm_waveform(pattern: str) -> None:
    recording = generate_dect_packet(payload_pattern=pattern)
    result = analyze_dect_recording(recording)[0]
    assert result.cts60_trace_frequency_hz.size == result.bits.size * 6
    assert np.diff(result.cts60_trace_sample) == pytest.approx(
        result.metadata["samples_per_symbol"] / 6.0
    )
    assert result.cts60_trace_fraction[:12].tolist() == list(range(6)) * 2
    assert result.metadata["measurement_filter_applied"] is False
    assert result.measurement_bandwidth_hz >= 3_000_000.0
    assert result.raw_fm_frequency_hz == pytest.approx(
        result.measurement_fm_frequency_hz
    )
    assert result.symbol_frequency_hz.shape == result.bits.shape
    assert np.isfinite(result.fitted_deviation_hz)


def test_cts60_trace_uses_physical_p_grid_for_prolonged_preamble() -> None:
    result = analyze_dect_recording(
        generate_dect_packet(prolonged_preamble=True)
    )[0]
    assert result.cts60_trace_symbol[0] == -16
    assert result.cts60_trace_symbol[16 * 6] == 0
    assert result.cts60_trace_fraction[:6].tolist() == list(range(6))


def test_modulation_references_are_independent_of_carrier_accuracy_estimator() -> None:
    result = analyze_dect_recording(
        generate_dect_packet(payload_pattern="00001111", frequency_error_hz=17_000.0)
    )[0]
    loopback_start, loopback_stop = result.metadata["loopback_bit_range"]
    start = result.metadata["actual_preamble_start_sample"] + (loopback_start + 1) * result.metadata["samples_per_symbol"]
    stop = result.metadata["actual_preamble_start_sample"] + (loopback_stop - 1) * result.metadata["samples_per_symbol"]
    selected = result.measurement_fm_frequency_hz[
        (result.measurement_fm_sample >= start) & (result.measurement_fm_sample < stop)
    ]
    assert result.frequency_references.measured_hz == pytest.approx(np.mean(selected))
    assert result.frequency_references.nominal_hz == 0.0
    assert result.frequency_references.half_peak_hz == pytest.approx(
        0.5 * (np.max(selected) + np.min(selected))
    )
    assert result.modulation_reference_hz() == result.frequency_references.measured_hz
    assert result.modulation_reference_hz(DectModulationReference.NOMINAL) == 0.0
    assert result.carrier_error_hz == pytest.approx(17_000.0, abs=500.0)


def test_case_a_peak_search_excludes_transition_adjacent_bits() -> None:
    result = analyze_dect_recording(
        generate_dect_packet(payload_pattern="00001111")
    )[0]
    loopback_start, _ = result.metadata["loopback_bit_range"]
    assert np.isnan(result.bit_peak_frequency_hz[loopback_start])
    assert np.isfinite(result.bit_peak_frequency_hz[loopback_start + 1])
    assert np.isfinite(result.bit_peak_frequency_hz[loopback_start + 2])
    assert np.isnan(result.bit_peak_frequency_hz[loopback_start + 3])
    eligible_values = result.measurement_fm_frequency_hz[result.etsi_eligible_sample_mask]
    reference = result.carrier_error_hz
    assert result.positive_deviation_hz == pytest.approx(
        np.max(eligible_values) - reference
    )
    assert result.negative_deviation_hz == pytest.approx(
        np.min(eligible_values) - reference
    )


def test_rf_modulation_rejects_less_than_three_mhz_usable_bandwidth() -> None:
    recording = replace(
        generate_dect_packet(),
        usable_bandwidth_hz=2_999_999.0,
    )
    with pytest.raises(ValueError, match="3 MHz usable bandwidth"):
        analyze_dect_recording(recording)


@pytest.mark.parametrize(
    ("packet_type", "expected"),
    (
        ("P00", (("S-field", 0, 32), ("A-field", 32, 96))),
        (
            "P32",
            (
                ("S-field", 0, 32),
                ("A-field", 32, 96),
                ("B-field", 96, 416),
                ("X-field", 416, 420),
            ),
        ),
        (
            "P80Z",
            (
                ("S-field", 0, 32),
                ("A-field", 32, 96),
                ("B-field", 96, 896),
                ("X-field", 896, 900),
                ("Z-field", 900, 904),
            ),
        ),
    ),
)
def test_packet_field_boundaries_follow_physical_packet_layout(
    packet_type: str,
    expected: tuple[tuple[str, int, int], ...],
) -> None:
    result = analyze_dect_recording(
        generate_dect_packet(packet_type=packet_type)
    )[0]
    assert tuple(
        (field.name, field.start_bit, field.stop_bit) for field in result.fields
    ) == expected


def test_rf_summary_uses_requested_measurement_order() -> None:
    result = analyze_dect_recording(generate_dect_packet())[0]
    items = [
        row.test_item
        for row in result.summary_rows
        if row.section == "RF PHY Measurements"
    ]
    assert items[:4] == [
        "Transmit Power",
        "Power-Time Template",
        "GFSK Modulation Deviation",
        "Modulation Speed",
    ]


def test_committed_dect_case_a_fixture_is_analyzable() -> None:
    recording = FileIQSource.load(
        FIXTURES / "dect_rfp_p32_case_a_9p216msps.npz"
    )
    result = analyze_dect_recording(recording)[0]
    assert result.direction == "RFP"
    assert result.packet_type == "P32"
    assert result.modulation_case == "Case A (00001111)"
    assert result.carrier_error_hz == pytest.approx(12_500.0, abs=500.0)
    assert result.symbol_rate_error_ppm == pytest.approx(8.0, abs=1.0)
    assert result.output_power == pytest.approx(-7.5, abs=0.1)


def test_committed_dect_prbs9_fixture_is_analyzable_as_reference() -> None:
    recording = FileIQSource.load(
        FIXTURES / "dect_rfp_p32_prbs9_9p216msps.npz"
    )
    result = analyze_dect_recording(recording)[0]
    assert result.direction == "RFP"
    assert result.packet_type == "P32"
    assert result.modulation_case == "Observed arbitrary payload"
    assert result.carrier_test_eligible is False
    assert result.modulation_test_eligible is False
    rows = {row.test_item: row for row in result.summary_rows}
    assert "Observed carrier frequency error" in rows
    assert "Observed GFSK deviation" in rows


def test_symbol_clock_uses_known_s_field_with_arbitrary_payload() -> None:
    recording = generate_dect_packet(
        direction="PP",
        payload_pattern="prbs9",
        symbol_rate_error_ppm=-173.0,
    )
    result = analyze_dect_recording(recording)[0]
    assert result.symbol_rate_error_ppm == pytest.approx(-173.0, abs=6.0)
    assert result.metadata["samples_per_symbol"] == pytest.approx(
        recording.sample_rate_hz / (1_152_000.0 * (1.0 - 173e-6)),
        rel=6e-6,
    )


def test_nominal_live_pluto_power_is_displayed_in_dbm() -> None:
    calibrated = generate_dect_packet(power_dbm=-11.0)
    live = replace(
        calibrated,
        amplitude_calibrated=False,
        metadata={**dict(calibrated.metadata), "nominal_pluto_amplitude": True},
    )
    result = analyze_dect_recording(live)[0]
    assert result.output_power == pytest.approx(-11.0, abs=0.1)
    assert result.output_power_unit == "dBm"
    assert result.power_calibrated is False


def test_variable_p00j_exposes_physical_fields_instead_of_opaque_body() -> None:
    bits = np.resize(np.array([0, 1], dtype=np.uint8), 455)
    decoded = analyze_packet(
        PacketDecodeInput(
            bits,
            protocol_hint="dect.classic",
            context={"direction": "PP", "packet_type": "P00j (455 symbols)"},
        )
    )
    fields = decoded.root_fields[0].children
    assert tuple(
        (field.name, field.start_bit, field.stop_bit) for field in fields
    ) == (
        ("S-field", 0, 32),
        ("A-field", 32, 96),
        ("B-field", 96, 451),
        ("X-field", 451, 455),
    )


def test_noisy_live_like_capture_keeps_timing_and_power_time_measurements() -> None:
    generated = generate_dect_packet(
        direction="PP",
        payload_pattern="prbs9",
        power_dbm=-24.0,
        symbol_rate_error_ppm=-199.0,
    )
    random = np.random.default_rng(7)
    noise_amplitude = 10.0 ** (-70.0 / 20.0) / np.sqrt(2.0)
    noise = noise_amplitude * (
        random.normal(size=generated.sample_count)
        + 1j * random.normal(size=generated.sample_count)
    )
    recording = replace(
        generated,
        iq=(generated.iq + noise).astype(np.complex64),
        amplitude_calibrated=False,
        metadata={
            **dict(generated.metadata),
            "nominal_pluto_amplitude": True,
        },
    )
    result = analyze_dect_recording(recording)[0]
    assert result.symbol_rate_error_ppm == pytest.approx(-199.0, abs=4.0)
    assert result.output_power == pytest.approx(-24.0, abs=0.1)
    assert result.output_power_unit == "dBm"
    assert result.attack_time_s is not None
    assert result.release_time_s is not None
    assert result.power_time_pass is True


@pytest.mark.parametrize("direction", ("RFP", "PP"))
def test_prolonged_preamble_is_anchored_by_sync_word(direction: str) -> None:
    recording = generate_dect_packet(
        direction=direction,
        packet_type="P32",
        prolonged_preamble=True,
    )
    result = analyze_dect_recording(recording)[0]
    assert result.direction == direction
    assert result.preamble_mode == "Prolonged"
    assert result.p0_sample == pytest.approx(
        recording.metadata["expected_p0_sample"], abs=0.6
    )
    assert result.packet_type == "P32"
    assert result.bits.size == 436
    assert result.metadata["physical_packet_symbol_count"] == 420
    assert result.preamble_correlation > 0.98
    assert result.sync_word_correlation > 0.98
    assert result.sync_score > 0.98
    assert tuple(
        (
            field.name,
            field.start_bit,
            field.stop_bit,
            *dect_p_range(field.start_bit, field.stop_bit, 16),
        )
        for field in result.fields
    ) == (
        ("Prolonged Preamble", 0, 16, -16, 0),
        ("S-field", 16, 48, 0, 32),
        ("A-field", 48, 112, 32, 96),
        ("B-field", 112, 432, 96, 416),
        ("X-field", 432, 436, 416, 420),
    )


def test_normal_preamble_keeps_bit_and_dect_symbol_numbering_aligned() -> None:
    result = analyze_dect_recording(generate_dect_packet())[0]
    assert result.preamble_mode == "Normal"
    assert result.bits.size == 420
    s_field = result.fields[0]
    assert (s_field.start_bit, s_field.stop_bit) == (0, 32)
    assert dect_p_range(s_field.start_bit, s_field.stop_bit, 0) == (0, 32)


def test_alternating_preamble_without_sync_word_is_not_valid_s_field() -> None:
    sps = 8
    alternating = np.resize(np.array([0, 1], dtype=np.uint8), 64)
    levels = 2.0 * alternating.astype(np.float64) - 1.0
    frequency = np.repeat(levels, sps)
    positions = np.arange(frequency.size, dtype=np.float64) + 0.5
    with pytest.raises(RuntimeError, match="synchronization failed"):
        _sync_packet(frequency, positions, 0, float(sps))
