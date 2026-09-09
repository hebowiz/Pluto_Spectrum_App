from dataclasses import replace

import numpy as np
import pytest

from pluto_sa.vsa.model import IQRecording
from pluto_sa.vsa.protocol_modes.dect.power_time import (
    _complex_measurement_receiver,
    build_dect_power_measurement_paths,
    measure_dect_ntp,
    measure_dect_power_time,
)


RATE = 20_000_000.0
START = 600.0
END = 1600.0
NEXT = 3200.0


def _power_time_recording() -> IQRecording:
    amplitude = np.zeros(3800, dtype=np.float64)
    active = 10.0 ** (-10.0 / 20.0)
    amplitude[500:600] = np.linspace(0.0, active, 100, endpoint=False)
    amplitude[600:1610] = active
    amplitude[1610:1710] = np.linspace(active, 0.0, 100, endpoint=False)
    return IQRecording(
        iq=amplitude.astype(np.complex64),
        sample_rate_hz=RATE,
        usable_bandwidth_hz=3_000_000.0,
        full_scale=1.0,
        amplitude_calibrated=True,
    )


def _measure(recording: IQRecording):
    return measure_dect_power_time(
        recording,
        p0_sample=START,
        power_time_start_sample=START,
        packet_end_sample=END,
        next_power_time_start_sample=NEXT,
        measurement_bandwidth_hz=3_000_000.0,
    )


def test_nominal_template_uses_linear_ntp_and_passes_all_regions() -> None:
    result = _measure(_power_time_recording())
    assert result.reference_power_db == pytest.approx(-10.0, abs=0.05)
    assert result.overall_status == "PASS"
    assert {item.status for item in result.criteria} <= {"PASS"}
    assert result.attack_time_s is not None
    assert result.release_time_s is not None


@pytest.mark.parametrize(
    ("criterion", "mutation"),
    (
        ("Minimum Packet Power", (900, 901, 0.0)),
        ("Maximum Packet Power", (900, 901, 1.0)),
        ("Attack Region Maximum", (560, 590, 1.0)),
        ("Post-Packet Maintenance", (1580, 1620, 0.0)),
        ("Idle Power", (2320, 2380, 0.02)),
    ),
)
def test_each_sample_region_reports_its_own_violation(
    criterion: str, mutation: tuple[int, int, float]
) -> None:
    recording = _power_time_recording()
    iq = np.array(recording.iq, copy=True)
    iq[mutation[0] : mutation[1]] = np.complex64(mutation[2])
    result = _measure(replace(recording, iq=iq))
    measured = result.criterion_map[criterion]
    assert measured.status == "FAIL"
    assert measured.failure_samples.size > 0
    assert result.overall_status == "FAIL"


def test_uncalibrated_capture_never_gets_an_absolute_template_pass() -> None:
    result = _measure(replace(_power_time_recording(), amplitude_calibrated=False))
    assert result.overall_status == "INCOMPLETE"
    assert result.criterion_map["Minimum Packet Power"].status == "PASS"
    assert result.criterion_map["Attack Time"].status == "INCOMPLETE"
    assert "absolute amplitude calibration is required" in result.incomplete_reasons


def test_missing_next_packet_keeps_idle_power_incomplete() -> None:
    recording = _power_time_recording()
    result = measure_dect_power_time(
        recording,
        p0_sample=START,
        power_time_start_sample=START,
        packet_end_sample=END,
        measurement_bandwidth_hz=3_000_000.0,
    )
    assert result.criterion_map["Idle Power"].status == "INCOMPLETE"
    assert result.overall_status == "INCOMPLETE"


def test_less_than_54_us_gap_makes_idle_limit_not_applicable() -> None:
    recording = _power_time_recording()
    result = measure_dect_power_time(
        recording,
        p0_sample=START,
        power_time_start_sample=START,
        packet_end_sample=END,
        next_power_time_start_sample=END + 53e-6 * RATE,
        measurement_bandwidth_hz=3_000_000.0,
    )
    assert result.criterion_map["Idle Power"].status == "NOT APPLICABLE"
    assert result.overall_status == "PASS"


def test_sub_three_mhz_bandwidth_marks_conformance_incomplete() -> None:
    result = measure_dect_power_time(
        _power_time_recording(),
        p0_sample=START,
        power_time_start_sample=START,
        packet_end_sample=END,
        next_power_time_start_sample=NEXT,
        measurement_bandwidth_hz=2_999_999.0,
    )
    assert result.overall_status == "INCOMPLETE"
    assert result.criterion_map["Minimum Packet Power"].status == "INCOMPLETE"


def test_attack_and_release_limits_are_strict_at_ten_microseconds() -> None:
    recording = _power_time_recording()
    iq = np.array(recording.iq, copy=True)
    threshold_amplitude = np.sqrt(25e-3)
    active = 10.0 ** (-10.0 / 20.0)
    iq[400:600] = active
    iq[399] = 0.0
    iq[400] = threshold_amplitude
    iq[1600:1801] = active
    iq[1800] = threshold_amplitude
    iq[1801:] = 0.0
    result = _measure(replace(recording, iq=iq))
    assert result.attack_time_s == pytest.approx(10e-6, abs=1e-12)
    assert result.release_time_s == pytest.approx(10e-6, abs=1e-12)
    assert result.criterion_map["Attack Time"].status == "FAIL"
    assert result.criterion_map["Release Time"].status == "FAIL"


def test_ntp_is_a_linear_power_mean_not_a_db_mean() -> None:
    recording = _power_time_recording()
    iq = np.array(recording.iq, copy=True)
    iq[600:1100] = np.complex64(10.0 ** (-10.0 / 20.0))
    iq[1100:1600] = np.complex64(10.0 ** (-20.0 / 20.0))
    changed = replace(recording, iq=iq)
    paths = build_dect_power_measurement_paths(changed)
    result = _measure(changed)
    expected = np.mean(paths.power_time_power[int(START) : int(END)])
    assert result.reference_power_linear == pytest.approx(expected, rel=1e-12)
    db_average = np.mean(10.0 * np.log10(paths.power_time_power[int(START) : int(END)]))
    assert result.reference_power_db != pytest.approx(db_average + 30.0, abs=0.1)


def test_parallel_measurement_receivers_preserve_center_cw_gain() -> None:
    recording = IQRecording(
        iq=np.ones(4096, dtype=np.complex64) * np.complex64(0.25 + 0.0j),
        sample_rate_hz=RATE,
        usable_bandwidth_hz=8_000_000.0,
        amplitude_calibrated=True,
    )
    paths = build_dect_power_measurement_paths(recording)
    middle = slice(512, -512)
    np.testing.assert_allclose(
        np.mean(paths.power_time_power[middle]),
        np.mean(paths.raw_power[middle]),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.mean(paths.one_mhz_power[middle]),
        np.mean(paths.raw_power[middle]),
        rtol=1e-6,
    )
    assert paths.power_time_filter_group_delay_samples > 0.0
    assert paths.ntp_filter_group_delay_samples > 0.0
    assert paths.filter_delay_compensated


def test_one_mhz_path_is_filtered_directly_from_common_iq_not_three_mhz_path() -> None:
    recording = _power_time_recording()
    paths = build_dect_power_measurement_paths(recording)
    direct_iq, _, _ = _complex_measurement_receiver(
        recording.iq / recording.full_scale,
        recording.sample_rate_hz,
        1_000_000.0,
    )
    direct_power_w = np.abs(direct_iq) ** 2 / 1000.0
    np.testing.assert_allclose(paths.one_mhz_power, direct_power_w, rtol=1e-12, atol=1e-18)


def test_clause_10_ntp_uses_one_mhz_and_p0_to_packet_end() -> None:
    paths = build_dect_power_measurement_paths(_power_time_recording())
    ntp = measure_dect_ntp(paths, p0_sample=START, packet_end_sample=END)
    assert ntp.available
    assert ntp.measurement_bandwidth_hz == 1_000_000.0
    assert ntp.start_sample == START
    assert ntp.stop_sample == END
    assert ntp.power_linear == pytest.approx(
        np.mean(paths.one_mhz_power[int(START) : int(END)]), rel=1e-12
    )


def test_narrow_common_iq_can_measure_ntp_but_not_power_time_template() -> None:
    recording = replace(_power_time_recording(), usable_bandwidth_hz=1_500_000.0)
    paths = build_dect_power_measurement_paths(recording)
    ntp = measure_dect_ntp(paths, p0_sample=START, packet_end_sample=END)
    power_time = measure_dect_power_time(
        recording,
        p0_sample=START,
        power_time_start_sample=START,
        packet_end_sample=END,
        next_power_time_start_sample=NEXT,
        measurement_paths=paths,
    )
    assert ntp.available
    assert power_time.overall_status == "INCOMPLETE"
    assert not paths.power_time_available
