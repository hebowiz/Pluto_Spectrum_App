import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace
import numpy as np
import pytest
from pluto_vsa.model import IQRecording
from pluto_vsa.pattern import MeasurementFilterMode
from pluto_vsa.protocol_modes.bluetooth.model import (
    BluetoothAnalysisProfile,
    analyze_bluetooth_classic_recording,
    analyze_bluetooth_classic_recordings,
)
import pluto_vsa.protocol_modes.bluetooth.model as bluetooth_model
from pluto_vsa.protocol_modes.bluetooth.rf_measurement import (
    BluetoothFMMeasurementTrace,
    BluetoothRFMeasurementFilterProfile,
    BluetoothRFTestAccumulator,
)
from pluto_vsg.engine import BluetoothBRWaveformEngine
from pluto_vsg.model import BluetoothPacketKind
from pluto_vsg.profiles import bluetooth_br_edr_project, bluetooth_br_fields


def test_dedicated_edr_length_crc_and_type_meaning_use_air_bit_order() -> None:
    base = bluetooth_br_edr_project()
    settings = replace(
        base.bluetooth_br,
        packet_kind=BluetoothPacketKind.DH1_2,
        payload_length_bytes=54,
    )
    generated = BluetoothBRWaveformEngine().generate(
        replace(base, bluetooth_br=settings, fields=bluetooth_br_fields(settings))
    )
    result = analyze_bluetooth_classic_recording(
        IQRecording(
            iq=generated.iq,
            sample_rate_hz=generated.sample_rate_hz,
            center_frequency_hz=base.center_frequency_hz,
            source="generated 2-DH1",
        ),
        profile=BluetoothAnalysisProfile.GENERAL_PACKET,
        lap=settings.lap,
        uap=settings.uap,
        clock_6_1=settings.clock_6_1,
        whitening_enabled=settings.whitening_enabled,
        result_length=1024,
    )
    assert result.packet.packet_type == "2-DH1"
    measurement_trace = result.metadata["fsk_measurement_trace"]
    assert isinstance(measurement_trace, BluetoothFMMeasurementTrace)
    assert (
        measurement_trace.filter_profile
        is BluetoothRFMeasurementFilterProfile.BR_1M
    )
    assert result.packet.integrity.crc_valid is True
    header = next(field for field in result.packet.root_fields if field.field_id == "header")
    type_field = next(field for field in header.children if field.field_id == "type")
    assert type_field.meaning == "2-DH1"
    payload = next(field for field in result.packet.root_fields if field.field_id == "payload")
    payload_header = next(field for field in payload.children if field.field_id == "payload_header")
    length = next(field for field in payload_header.children if field.field_id == "length")
    assert int(length.value) == 54
    edr_session = result.metadata["analysis_session"]
    # 10 sync + (16-bit enhanced header + 54-byte payload + 16-bit CRC) / 2
    # + 2 trailer symbols.  The configured discovery range was 1024 symbols,
    # but the result must end at the decoded packet boundary.
    assert edr_session.pattern_result.decoded_symbols.size == 244
    assert (
        edr_session.pattern_result.result_stop_sample
        - edr_session.pattern_result.result_start_sample
        == 244 * 8
    )
    assert result.vsa_result.iq.size == 244 * 8
    assert result.vsa_result.measured_symbols.size <= 244
    br_session = result.metadata["br_analysis_session"]
    expected_edr_search_start = int(
        br_session.pattern_result.pattern_start_sample
        + round(131.0 * generated.sample_rate_hz / 1_000_000.0)
    )
    search_guard = max(
        round(2.0 * generated.sample_rate_hz / 1_000_000.0),
        # The EDR synchronizer keeps 50 us of pre-roll for SRRC/carrier
        # settling while independently enforcing the +/-8-symbol timing
        # acceptance window.
        round(50.0 * generated.sample_rate_hz / 1_000_000.0),
    )
    assert result.metadata["analysis_sample_offset"] <= expected_edr_search_start
    assert (
        expected_edr_search_start - result.metadata["analysis_sample_offset"]
        <= search_guard
    )
    metrics = {metric.metric_id: metric.display for metric in result.metrics}
    assert metrics["bluetooth_devm_rms"] != "--"
    assert "evm_rms" not in metrics
    assert "differential_symbol_evm_rms" not in metrics
    assert (
        result.metadata["analysis_session"].demodulation.measurement_filter
        is MeasurementFilterMode.AUTO
    )
    assert (
        result.metadata["br_analysis_session"].demodulation.measurement_filter
        is MeasurementFilterMode.NONE
    )


def test_edr_sig_measurement_uses_five_us_guard_and_excludes_trailer() -> None:
    base = bluetooth_br_edr_project()
    settings = replace(
        base.bluetooth_br,
        packet_kind=BluetoothPacketKind.DH3_2,
        payload_length_bytes=356,
        whitening_enabled=False,
    )
    generated = BluetoothBRWaveformEngine().generate(
        replace(base, bluetooth_br=settings, fields=bluetooth_br_fields(settings))
    )
    result = analyze_bluetooth_classic_recording(
        IQRecording(
            iq=generated.iq,
            sample_rate_hz=generated.sample_rate_hz,
            center_frequency_hz=base.center_frequency_hz,
            source="generated 2-DH3 RF test packet",
        ),
        profile=BluetoothAnalysisProfile.RF_PHY_TEST,
        lap=settings.lap,
        uap=settings.uap,
        clock_6_1=settings.clock_6_1,
        whitening_enabled=False,
        expected_edr_rf_test_packet="2-DH3",
        result_length=4096,
    )

    measurement = result.metadata["rf_measurements"][0]
    assert measurement.eligibility.eligible is True
    assert measurement.metrics["guard_time_s"] == pytest.approx(5.0e-6, abs=0.2e-6)
    assert measurement.metrics["sync_symbol_errors"] == 0
    assert measurement.metrics["trailer_symbol_errors"] == 0
    assert measurement.metrics["block_count"] > 0
    assert measurement.metadata["trailer_excluded_from_devm"] is True
    assert measurement.metrics["rms_devm_worst"] < 0.05
    assert measurement.metrics["output_power_dbm"] is not None
    assert measurement.metrics["pgfsk_dbm"] is not None
    assert measurement.metrics["pdpsk_dbm"] is not None
    assert measurement.metadata["relative_power_measurement_fraction"] == 0.8
    assert measurement.metadata["pgfsk_region"] == "Access Code and Header"
    assert measurement.metadata["pdpsk_region"] == (
        "Synchronization sequence and payload"
    )
    assert measurement.metadata["output_power_window_start_sample"] > (
        result.metadata["packet_start_sample"]
    )
    assert measurement.metadata["output_power_window_stop_sample"] < (
        result.metadata["packet_stop_sample"]
    )
    assert measurement.metrics["payload_bit_errors"] == 0
    summary = {row.metric_id: row for row in result.summary_rows}
    assert summary["pgfsk"].value.endswith(" dBm")
    assert summary["pdpsk"].value.endswith(" dBm")
    assert summary["pgfsk"].limit == "\N{EM DASH}"
    assert summary["pdpsk"].limit == "\N{EM DASH}"
    assert summary["pgfsk"].section == "RF PHY Measurements"
    assert summary["pdpsk"].section == "RF PHY Measurements"
    assert not any(
        row.metric_id in {"pgfsk", "pdpsk"}
        for row in result.summary_rows
        if row.section == "Reference Information"
    )
    assert summary["omega_i"].result == "PASS"
    assert summary["omega_0"].result == "PASS"
    assert summary["omega_i_plus_omega_0"].result == "PASS"
    assert summary["rms_devm"].value != "N/A"
    assert summary["rms_devm"].result == "MEASURING"
    assert summary["p99_devm"].value != "N/A"
    assert summary["p99_devm"].result == "MEASURING"
    assert summary["peak_devm"].value != "N/A"
    assert summary["peak_devm"].result == "MEASURING"
    assert summary["guard_time"].limit == "4.60–5.40 µs"
    assert summary["guard_time"].result == "MEASURING"
    assert summary["differential_phase_encoding"].result == "MEASURING"
    assert summary["synchronization_sequence"].result == "MEASURING"
    assert summary["trailer"].limit == "≤ 1 bit error / 50 packets"
    assert summary["trailer"].result == "MEASURING"

    accumulator = BluetoothRFTestAccumulator()
    for _ in range(100):
        accumulator.add(measurement)
    aggregate = accumulator.aggregate_edr()
    aggregated = replace(
        result,
        metadata={**result.metadata, "rf_capture_aggregate": aggregate},
    )
    aggregated_summary = {row.metric_id: row for row in aggregated.summary_rows}
    assert aggregated_summary["rms_devm"].result == "PASS"
    assert aggregated_summary["p99_devm"].result == "PASS"
    assert aggregated_summary["peak_devm"].result == "PASS"
    assert aggregated_summary["guard_time"].result == "PASS"
    assert aggregated_summary["differential_phase_encoding"].result == "PASS"
    assert aggregated_summary["synchronization_sequence"].result == "PASS"
    assert aggregated_summary["trailer"].result == "PASS"
    assert aggregated_summary["devm_blocks_evaluated"].value == "200 / 200"
    assert aggregated_summary["guard_time_packets_evaluated"].value == "100 / 100"
    assert aggregated_summary["guard_time_valid_packets"].value == "100 / 100"

    bad_metrics = dict(measurement.metrics)
    bad_metrics.update(
        guard_time_s=6.0e-6,
        payload_bit_errors=1,
        sync_symbol_errors=1,
        sync_bit_errors=1,
        trailer_symbol_errors=1,
        trailer_bit_errors=1,
    )
    bad_measurement = replace(measurement, metrics=bad_metrics)
    failing_accumulator = BluetoothRFTestAccumulator()
    for _ in range(6):
        failing_accumulator.add(bad_measurement)
    failing = replace(
        result,
        metadata={
            **result.metadata,
            "rf_capture_aggregate": failing_accumulator.aggregate_edr(),
        },
    )
    failing_summary = {row.metric_id: row for row in failing.summary_rows}
    assert failing_summary["guard_time"].result == "FAIL"
    assert failing_summary["differential_phase_encoding"].result == "FAIL"
    assert failing_summary["synchronization_sequence"].result == "FAIL"
    assert failing_summary["trailer"].result == "FAIL"


def test_edr_sig_devm_uses_capture_level_zero_if_dc_correction() -> None:
    base = bluetooth_br_edr_project()
    settings = replace(
        base.bluetooth_br,
        packet_kind=BluetoothPacketKind.DH1_2,
        payload_length_bytes=54,
        whitening_enabled=False,
    )
    generated = BluetoothBRWaveformEngine().generate(
        replace(base, bluetooth_br=settings, fields=bluetooth_br_fields(settings))
    )

    def analyze(iq: np.ndarray, *, remove_dc: bool):
        result = analyze_bluetooth_classic_recording(
            IQRecording(
                iq=iq,
                sample_rate_hz=generated.sample_rate_hz,
                center_frequency_hz=base.center_frequency_hz,
                metadata={"dc_removal_recommended": remove_dc},
            ),
            profile=BluetoothAnalysisProfile.RF_PHY_TEST,
            lap=settings.lap,
            uap=settings.uap,
            clock_6_1=settings.clock_6_1,
            whitening_enabled=False,
            result_length=1024,
        )
        measurement = next(
            item
            for item in result.metadata["rf_measurements"]
            if item.test_case_id == "bluetooth.edr"
        )
        return result, measurement

    # A finite Pluto burst capture contains an idle/noise cluster used by the
    # robust frontend estimator.  Model that context around the generated
    # packet rather than asking the estimator to separate a continuous signal
    # centered exactly at zero IF.
    padded_iq = np.pad(generated.iq, (4000, 4000))
    baseline_result, baseline = analyze(padded_iq, remove_dc=False)
    dc_offset = np.complex64(0.18 - 0.11j)
    corrected_result, corrected = analyze(
        np.asarray(padded_iq + dc_offset, dtype=np.complex64),
        remove_dc=True,
    )

    assert baseline_result.packet.integrity.crc_valid is True
    assert corrected_result.packet.integrity.crc_valid is True
    prepared_recording = corrected_result.metadata["analysis_session"].recording
    assert prepared_recording.metadata["software_dc_removal_applied"] is True
    assert corrected.metrics["rms_devm_worst"] == pytest.approx(
        baseline.metrics["rms_devm_worst"], abs=5e-3
    )
    offset_lo_recording = IQRecording(
        iq=np.asarray(padded_iq + dc_offset, dtype=np.complex64),
        sample_rate_hz=generated.sample_rate_hz,
        metadata={
            "dc_removal_recommended": True,
            "experimental_lo_offset": True,
        },
    )
    assert (
        bluetooth_model._prepare_classic_frontend_recording(offset_lo_recording)
        is offset_lo_recording
    )


def test_edr_sig_devm_retains_symbol_dependent_phase_and_amplitude_error() -> None:
    base = bluetooth_br_edr_project()
    settings = replace(
        base.bluetooth_br,
        packet_kind=BluetoothPacketKind.DH3_2,
        payload_length_bytes=300,
        whitening_enabled=False,
    )
    generated = BluetoothBRWaveformEngine().generate(
        replace(base, bluetooth_br=settings, fields=bluetooth_br_fields(settings))
    )

    def analyze(iq: np.ndarray):
        return analyze_bluetooth_classic_recording(
            IQRecording(
                iq=iq,
                sample_rate_hz=generated.sample_rate_hz,
                center_frequency_hz=base.center_frequency_hz,
            ),
            profile=BluetoothAnalysisProfile.RF_PHY_TEST,
            lap=settings.lap,
            uap=settings.uap,
            clock_6_1=settings.clock_6_1,
            whitening_enabled=False,
            result_length=4096,
        ).metadata["rf_measurements"][0]

    baseline = analyze(generated.iq)
    impaired_iq = np.array(generated.iq, copy=True)
    start = int(generated.metadata["edr_start_sample"])
    stop = int(generated.metadata["data_stop_sample"]) - 2 * 8
    sample_axis = np.arange(stop - start, dtype=np.float64)
    symbol_index = (sample_axis // 8).astype(np.int64)
    amplitude = np.where((symbol_index & 1) == 0, 0.82, 1.0)
    phase_error = 0.16 * np.sin(2.0 * np.pi * sample_axis / (3.0 * 8.0))
    impaired_iq[start:stop] *= amplitude * np.exp(1j * phase_error)
    impaired = analyze(impaired_iq)

    assert impaired.metrics["rms_devm_worst"] > (
        baseline.metrics["rms_devm_worst"] + 0.08
    )
    assert impaired.metrics["peak_devm_worst"] > (
        baseline.metrics["peak_devm_worst"] + 0.12
    )


@pytest.mark.parametrize(
    ("packet_kind", "expected_phy", "expected_result_symbols"),
    (
        # BR: Access + Header + ACL header + 12-byte body + CRC.
        (BluetoothPacketKind.DH1, "BR", 72 + 54 + 8 + 12 * 8 + 16),
        (BluetoothPacketKind.DH3, "BR", 72 + 54 + 16 + 12 * 8 + 16),
        (BluetoothPacketKind.DH5, "BR", 72 + 54 + 16 + 12 * 8 + 16),
        # EDR: Sync + ceil((enhanced header + body + CRC) / modulation width)
        # + two trailer symbols.
        (BluetoothPacketKind.DH1_2, "EDR 2M", 10 + (16 + 12 * 8 + 16) // 2 + 2),
        (BluetoothPacketKind.DH3_2, "EDR 2M", 10 + (16 + 12 * 8 + 16) // 2 + 2),
        (BluetoothPacketKind.DH5_2, "EDR 2M", 10 + (16 + 12 * 8 + 16) // 2 + 2),
        (BluetoothPacketKind.DH1_3, "EDR 3M", 10 + 43 + 2),
        (BluetoothPacketKind.DH3_3, "EDR 3M", 10 + 43 + 2),
        (BluetoothPacketKind.DH5_3, "EDR 3M", 10 + 43 + 2),
    ),
)
def test_classic_type_is_only_a_phy_candidate_and_length_sets_result_range(
    packet_kind: BluetoothPacketKind,
    expected_phy: str,
    expected_result_symbols: int,
) -> None:
    base = bluetooth_br_edr_project()
    settings = replace(
        base.bluetooth_br,
        packet_kind=packet_kind,
        payload_length_bytes=12,
    )
    generated = BluetoothBRWaveformEngine().generate(
        replace(base, bluetooth_br=settings, fields=bluetooth_br_fields(settings))
    )
    result = analyze_bluetooth_classic_recording(
        IQRecording(
            iq=generated.iq,
            sample_rate_hz=generated.sample_rate_hz,
            center_frequency_hz=base.center_frequency_hz,
            source=f"generated {packet_kind.value}",
        ),
        profile=BluetoothAnalysisProfile.GENERAL_PACKET,
        lap=settings.lap,
        uap=settings.uap,
        clock_6_1=settings.clock_6_1,
        whitening_enabled=settings.whitening_enabled,
        result_length=4096,
    )

    metrics = {metric.metric_id: metric.display for metric in result.metrics}
    assert metrics["detected_phy"] == expected_phy
    assert result.packet.packet_type == packet_kind.value
    assert result.packet.integrity.crc_valid is True
    assert result.packet.integrity.complete is True
    assert (
        result.metadata["analysis_session"].pattern_result.decoded_symbols.size
        == expected_result_symbols
    )
    assert result.metadata["physical_packet_stop_sample"] == (
        result.metadata["packet_stop_sample"]
    )
    assert result.metadata["packet_stop_source"] == (
        "decoded_type_and_payload_length"
    )
    if expected_phy == "BR":
        expected_packet_stop = result.metadata["packet_start_sample"] + int(
            round(expected_result_symbols * generated.sample_rate_hz / 1_000_000.0)
        )
    else:
        expected_packet_stop = int(
            round(
                result.metadata["edr_sync_first_symbol_center_sample"]
                + (expected_result_symbols - 0.5)
                * generated.sample_rate_hz
                / 1_000_000.0
            )
        )
    assert result.metadata["physical_packet_stop_sample"] == expected_packet_stop
    if expected_phy.startswith("EDR"):
        assert [row.metric_id for row in result.summary_rows[:13]] == [
            "pgfsk",
            "pdpsk",
            "relative_transmit_power",
            "omega_i",
            "omega_0",
            "omega_i_plus_omega_0",
            "rms_devm",
            "p99_devm",
            "peak_devm",
            "guard_time",
            "differential_phase_encoding",
            "synchronization_sequence",
            "trailer",
        ]
        summary = {row.metric_id: row for row in result.summary_rows}
        expected_devm_limits = (
            ("≤ 20 %", "≤ 30 %", "≤ 35 %")
            if expected_phy == "EDR 2M"
            else ("≤ 13 %", "≤ 20 %", "≤ 25 %")
        )
        assert (
            summary["rms_devm"].limit,
            summary["p99_devm"].limit,
            summary["peak_devm"].limit,
        ) == expected_devm_limits
        assert summary["relative_transmit_power"].limit == (
            "-4 dB < value < +1 dB"
        )
        assert summary["omega_i"].limit == "-75 kHz < ωi < +75 kHz"
        assert summary["omega_0"].limit == "-10 kHz < ω0 < +10 kHz"
        assert summary["omega_i_plus_omega_0"].limit == (
            "-75 kHz < value < +75 kHz"
        )
        assert summary["rms_devm"].result == "N/A"
        assert all(
            row.limit == "—" and row.result == "—"
            for row in result.summary_rows
            if row.section == "Reference Information"
        )


def test_br_packet_is_not_promoted_by_a_later_edr_sync() -> None:
    base = bluetooth_br_edr_project()
    br_settings = replace(
        base.bluetooth_br,
        packet_kind=BluetoothPacketKind.DH1,
        payload_length_bytes=12,
    )
    edr_settings = replace(
        base.bluetooth_br,
        packet_kind=BluetoothPacketKind.DH1_2,
        payload_length_bytes=12,
    )
    br = BluetoothBRWaveformEngine().generate(
        replace(base, bluetooth_br=br_settings, fields=bluetooth_br_fields(br_settings))
    )
    edr = BluetoothBRWaveformEngine().generate(
        replace(base, bluetooth_br=edr_settings, fields=bluetooth_br_fields(edr_settings))
    )
    spacer = np.zeros(256, dtype=np.complex64)
    recording = IQRecording(
        iq=np.concatenate((spacer, br.iq, spacer, edr.iq, spacer)),
        sample_rate_hz=br.sample_rate_hz,
        center_frequency_hz=base.center_frequency_hz,
        source="generated DH1 followed by 2-DH1",
    )

    results = analyze_bluetooth_classic_recordings(
        recording,
        profile=BluetoothAnalysisProfile.GENERAL_PACKET,
        lap=br_settings.lap,
        uap=br_settings.uap,
        clock_6_1=br_settings.clock_6_1,
        whitening_enabled=br_settings.whitening_enabled,
        result_length=4096,
    )

    assert [result.packet.packet_type for result in results] == ["DH1", "2-DH1"]
    assert [
        {metric.metric_id: metric.display for metric in result.metrics}["detected_phy"]
        for result in results
    ] == ["BR", "EDR 2M"]
    assert all(result.packet.integrity.crc_valid is True for result in results)


@pytest.mark.parametrize(
    "packet_kind",
    (BluetoothPacketKind.DH3_3, BluetoothPacketKind.DH5_3),
)
def test_3m_edr_phy_detection_tolerates_realistic_sync_boundary_delay(
    packet_kind: BluetoothPacketKind,
) -> None:
    """A guard/ramp timing offset must not turn 3-DHx into BR DHx."""

    base = bluetooth_br_edr_project()
    settings = replace(
        base.bluetooth_br,
        packet_kind=packet_kind,
        payload_length_bytes=12,
    )
    generated = BluetoothBRWaveformEngine().generate(
        replace(base, bluetooth_br=settings, fields=bluetooth_br_fields(settings))
    )
    recording = IQRecording(
        iq=generated.iq,
        sample_rate_hz=generated.sample_rate_hz,
        center_frequency_hz=base.center_frequency_hz,
        source=f"delayed {packet_kind.value}",
    )
    initial = analyze_bluetooth_classic_recording(
        recording,
        profile=BluetoothAnalysisProfile.GENERAL_PACKET,
        lap=settings.lap,
        uap=settings.uap,
        clock_6_1=settings.clock_6_1,
        whitening_enabled=settings.whitening_enabled,
        result_length=4096,
    )
    access_start = int(
        initial.metadata["br_analysis_session"].pattern_result.pattern_start_sample
    )
    switch_boundary = int(
        round(
            access_start
            + 131.0 * generated.sample_rate_hz / 1_000_000.0
        )
    )
    delay_samples = int(round(4.0e-6 * generated.sample_rate_hz))
    delayed = replace(
        recording,
        iq=np.concatenate(
            (
                generated.iq[:switch_boundary],
                np.zeros(delay_samples, dtype=np.complex64),
                generated.iq[switch_boundary:],
            )
        ),
    )

    result = analyze_bluetooth_classic_recording(
        delayed,
        profile=BluetoothAnalysisProfile.GENERAL_PACKET,
        lap=settings.lap,
        uap=settings.uap,
        clock_6_1=settings.clock_6_1,
        whitening_enabled=settings.whitening_enabled,
        result_length=4096,
    )

    assert result.packet.packet_type == packet_kind.value
    assert result.packet.integrity.complete is True
    assert result.packet.integrity.crc_valid is True
    assert result.metadata["edr_candidate_error"] is None


def test_dedicated_edr_multi_packet_analysis_uses_local_ranges_and_reports_relative_power() -> None:
    base = bluetooth_br_edr_project()
    settings = replace(
        base.bluetooth_br,
        packet_kind=BluetoothPacketKind.DH1_2,
        payload_length_bytes=54,
    )
    generated = BluetoothBRWaveformEngine().generate(
        replace(base, bluetooth_br=settings, fields=bluetooth_br_fields(settings))
    )
    spacer = np.zeros(256, dtype=np.complex64)
    recording = IQRecording(
        iq=np.concatenate((spacer, generated.iq, spacer, generated.iq, spacer)),
        sample_rate_hz=generated.sample_rate_hz,
        center_frequency_hz=base.center_frequency_hz,
        source="two generated 2-DH1 packets",
    )
    results = analyze_bluetooth_classic_recordings(
        recording,
        profile=BluetoothAnalysisProfile.GENERAL_PACKET,
        lap=settings.lap,
        uap=settings.uap,
        clock_6_1=settings.clock_6_1,
        whitening_enabled=settings.whitening_enabled,
        result_length=1024,
    )
    assert len(results) == 2
    assert all(result.packet.packet_type == "2-DH1" for result in results)
    assert all(result.packet.integrity.crc_valid is True for result in results)
    assert results[1].metadata["recording_sample_offset"] > 0
    assert results[1].metadata["analysis_sample_offset"] > results[0].metadata["analysis_sample_offset"]
    assert results[1].metadata["packet_start_sample"] > results[0].metadata["packet_stop_sample"]
    metrics = {metric.metric_id: metric.display for metric in results[1].metrics}
    assert metrics["fsk_average_power"] != "--"
    assert metrics["psk_average_power"] != "--"
    assert metrics["psk_relative_power"] != "--"
    assert abs(float(metrics["psk_relative_power"].split()[0])) < 0.5
    aggregate = results[1].metadata["rf_capture_aggregate"]
    assert aggregate.metrics["packet_count"] == 2
    assert aggregate.metrics["block_count"] == sum(
        result.metadata["rf_measurements"][0].metrics["block_count"]
        for result in results
    )
