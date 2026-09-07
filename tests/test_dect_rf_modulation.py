from dataclasses import replace

import numpy as np
import pytest

from pluto_protocol.dect.rf_modulation import (
    DectCaseBFormat,
    DectScramblingMode,
    case_a_bits,
    case_b_bits,
    deviation_in_limits,
    identify_rf_pattern,
    maximum_dsv,
)
from pluto_sa.vsa.protocol_modes.dect.analysis import _measurement_mask
from pluto_sa.vsa.model import IQRecording
from pluto_sa.vsa.protocol_modes.dect.analysis import analyze_dect_recording
from pluto_vsg.engine.dect import DectWaveformEngine
from pluto_vsg.model import DectBFieldSource, DectPacketType
from pluto_vsg.profiles.dect import dect_fields, dect_project


def _project(packet_type, source, mode=DectScramblingMode.NONE, phase=0):
    base = dect_project()
    settings = replace(
        base.dect,
        packet_type=packet_type,
        b_field_source=source,
        scrambling_mode=mode,
        scrambling_phase=phase,
    )
    period = 960.0 if packet_type in {DectPacketType.P80, DectPacketType.P80Z} else 480.0
    return replace(base, dect=settings, fields=dect_fields(settings), period_symbols=period)


def test_case_a_320_is_exact_air_pattern_and_one_error_is_not_case_a() -> None:
    expected = np.tile(np.array([0, 0, 0, 0, 1, 1, 1, 1], np.uint8), 40)
    np.testing.assert_array_equal(case_a_bits(320), expected)
    assert identify_rf_pattern(expected, "P32").case == "A"
    damaged = expected.copy()
    damaged[17] ^= 1
    identified = identify_rf_pattern(damaged, "P32")
    assert identified.case != "A"
    assert identified.pattern_errors == 1


def test_etsi_case_b_figures_have_exact_independent_boundaries() -> None:
    f27 = case_b_bits(DectCaseBFormat.A_FIELD_ONLY)
    np.testing.assert_array_equal(f27[:16], 1)
    np.testing.assert_array_equal(f27[16:], 0)

    f28 = case_b_bits(DectCaseBFormat.HALF_SLOT)
    assert f28.size == 80
    np.testing.assert_array_equal(f28[:8], [1, 0] * 4)
    np.testing.assert_array_equal(f28[8:40], 1)
    np.testing.assert_array_equal(f28[40:72], 0)
    np.testing.assert_array_equal(f28[72:], [1, 0] * 4)

    f29 = case_b_bits(DectCaseBFormat.FULL_SLOT)
    assert f29.size == 320
    np.testing.assert_array_equal(f29[:128], [1, 0] * 64)
    np.testing.assert_array_equal(f29[128:192], 1)
    np.testing.assert_array_equal(f29[192:256], 0)
    np.testing.assert_array_equal(f29[256:], [1, 0] * 32)

    f30 = case_b_bits(DectCaseBFormat.VARIABLE_640)
    f31 = case_b_bits(DectCaseBFormat.DOUBLE_SLOT)
    assert (f30.size, f31.size) == (640, 800)
    np.testing.assert_array_equal(f30[:128], [1, 0] * 64)
    np.testing.assert_array_equal(f30[128:512], np.tile(np.r_[np.ones(64, np.uint8), np.zeros(64, np.uint8)], 3))
    np.testing.assert_array_equal(f30[512:], [1, 0] * 64)
    np.testing.assert_array_equal(f31[:144], [1, 0] * 72)
    np.testing.assert_array_equal(f31[144:656], np.tile(np.r_[np.ones(64, np.uint8), np.zeros(64, np.uint8)], 4))
    np.testing.assert_array_equal(f31[656:], [1, 0] * 72)
    assert all(maximum_dsv(bits) <= 64 for bits in (f27, f28, f29, f30, f31))


@pytest.mark.parametrize("mode", (DectScramblingMode.NONE, DectScramblingMode.STANDARD))
@pytest.mark.parametrize(
    "packet_type,source,expected",
    (
        (DectPacketType.P32, DectBFieldSource.CASE_A, case_a_bits(320)),
        (DectPacketType.P32, DectBFieldSource.CASE_B_ETSI, case_b_bits(DectCaseBFormat.FULL_SLOT)),
        (DectPacketType.P80, DectBFieldSource.CASE_B_ETSI, case_b_bits(DectCaseBFormat.DOUBLE_SLOT)),
    ),
)
def test_vsg_guarantees_required_air_bits_with_or_without_scrambling(packet_type, source, expected, mode) -> None:
    result = DectWaveformEngine().generate(_project(packet_type, source, mode, phase=3))
    np.testing.assert_array_equal(result.metadata["air_b_field_bits"], expected)
    if mode is DectScramblingMode.STANDARD:
        assert np.any(result.metadata["pre_scramble_b_field_bits"] != expected)
    identified = identify_rf_pattern(result.metadata["air_b_field_bits"], packet_type.value)
    assert identified.case == ("A" if source is DectBFieldSource.CASE_A else "B")


def test_a_field_only_uses_only_a16_to_a47_for_case_patterns() -> None:
    result = DectWaveformEngine().generate(_project(DectPacketType.P00, DectBFieldSource.CASE_B_ETSI))
    np.testing.assert_array_equal(result.packet_bits.bits[48:80], case_b_bits(DectCaseBFormat.A_FIELD_ONLY))
    np.testing.assert_array_equal(result.packet_bits.bits[:32], result.metadata["preamble_bits"].tolist() + result.metadata["sync_word_bits"].tolist())


def test_vsg_iq_round_trip_is_identified_as_etsi_figure_29() -> None:
    project = _project(DectPacketType.P32, DectBFieldSource.CASE_B_ETSI)
    generated = DectWaveformEngine().generate(project)
    result = analyze_dect_recording(
        IQRecording(
            iq=generated.iq,
            sample_rate_hz=generated.sample_rate_hz,
            center_frequency_hz=project.center_frequency_hz,
            usable_bandwidth_hz=0.8 * generated.sample_rate_hz,
            full_scale=1.0,
            source="DECT Case B round-trip",
            metadata={"dc_removal_recommended": False},
        )
    )[0]
    assert result.modulation_case == "Case B / Figure 29"
    assert result.modulation_test_eligible
    assert len(result.deviation_bit_results) == 124
    loopback_start, loopback_stop = result.metadata["loopback_bit_range"]
    assert result.metadata["clause_7_carrier_reference_hz"] == pytest.approx(
        np.mean(result.symbol_frequency_hz[loopback_start:loopback_stop])
    )
    assert all(item.valid for item in result.deviation_bit_results)


def test_case_b_signed_ranges_inside_limits_are_not_failed() -> None:
    project = _project(DectPacketType.P32, DectBFieldSource.CASE_B_ETSI)
    generated = DectWaveformEngine().generate(project)
    result = analyze_dect_recording(
        IQRecording(
            iq=generated.iq,
            sample_rate_hz=generated.sample_rate_hz,
            center_frequency_hz=project.center_frequency_hz,
            usable_bandwidth_hz=0.8 * generated.sample_rate_hz,
            full_scale=1.0,
            source="DECT deviation verdict regression",
            metadata={"dc_removal_recommended": False},
        )
    )[0]
    positive_index = 0
    negative_index = 0
    adjusted = []
    for item in result.deviation_bit_results:
        if item.bit_value:
            value = 362_000.0 if positive_index % 2 == 0 else 391_600.0
            positive_index += 1
        else:
            value = -360_500.0 if negative_index % 2 == 0 else -395_600.0
            negative_index += 1
        adjusted.append(replace(item, deviation_hz=value))
    checked = replace(result, deviation_bit_results=tuple(adjusted))
    rows = {row.test_item: row for row in checked.summary_rows}
    assert rows["Positive Peak Deviation"].value == "+362.0 to +391.6 kHz"
    assert rows["Negative Peak Deviation"].value == "-360.5 to -395.6 kHz"
    assert rows["Positive Worst Margin"].value == "+11.4 kHz"
    assert rows["Negative Worst Margin"].value == "+7.4 kHz"
    assert rows["GFSK Modulation Deviation"].result == "MEASURING"


def test_case_b_positive_and_negative_deviation_verdicts_are_independent() -> None:
    project = _project(DectPacketType.P32, DectBFieldSource.CASE_B_ETSI)
    generated = DectWaveformEngine().generate(project)
    result = analyze_dect_recording(
        IQRecording(
            iq=generated.iq,
            sample_rate_hz=generated.sample_rate_hz,
            center_frequency_hz=project.center_frequency_hz,
            usable_bandwidth_hz=0.8 * generated.sample_rate_hz,
            full_scale=1.0,
            source="DECT independent polarity verdict regression",
            metadata={"dc_removal_recommended": False},
        )
    )[0]
    positive_index = 0
    negative_index = 0
    adjusted = []
    for item in result.deviation_bit_results:
        if item.bit_value:
            value = 336_300.0 if positive_index % 2 == 0 else 381_800.0
            positive_index += 1
        else:
            value = -337_100.0 if negative_index % 2 == 0 else -405_600.0
            negative_index += 1
        adjusted.append(replace(item, deviation_hz=value))
    rows = {
        row.test_item: row
        for row in replace(
            result, deviation_bit_results=tuple(adjusted)
        ).summary_rows
    }
    assert rows["GFSK Modulation Deviation"].result == "FAIL"
    assert rows["Positive Peak Deviation"].result == "MEASURING"
    assert rows["Positive Worst Margin"].result == "MEASURING"
    assert rows["Negative Peak Deviation"].result == "FAIL"
    assert rows["Negative Worst Margin"].result == "FAIL"
    assert rows["Positive Worst Margin"].value == "+21.2 kHz"
    assert rows["Negative Worst Margin"].value == "-2.6 kHz"


def test_clause_11_window_excludes_first_and_last_bit_of_each_run() -> None:
    bits = case_a_bits(16)
    mask = _measurement_mask(bits, "Case A", 0, bits.size)
    np.testing.assert_array_equal(np.flatnonzero(mask), [1, 2, 5, 6, 9, 10, 13, 14])
    figure = case_b_bits(DectCaseBFormat.FULL_SLOT)
    mask = _measurement_mask(figure, "Case B / Figure 29", 0, figure.size)
    assert not np.any(mask[:128])
    np.testing.assert_array_equal(np.flatnonzero(mask[128:192]), np.arange(1, 63))
    np.testing.assert_array_equal(np.flatnonzero(mask[192:256]), np.arange(1, 63))


@pytest.mark.parametrize(
    "case,value,expected",
    (
        ("A", 259_000.0, False), ("A", 259_001.0, True),
        ("A", 403_000.0, False), ("B", 202_000.0, False),
        ("B", -202_001.0, True), ("B", -403_000.0, False),
    ),
)
def test_case_a_and_b_deviation_limits_are_strict(case, value, expected) -> None:
    assert deviation_in_limits(value, case) is expected
