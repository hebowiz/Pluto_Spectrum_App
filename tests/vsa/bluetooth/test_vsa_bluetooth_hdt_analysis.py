import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace
import numpy as np
import pyqtgraph as pg
import pytest
from pyqtgraph.Qt import QtCore
from pluto_protocol.bluetooth.hdt import HDTRate
from pluto_vsa.pattern import IQPowerTriggerSettings
from pluto_vsa.protocol_modes.bluetooth.model import (
    BluetoothAnalysisProfile,
    analyze_bluetooth_hdt_recording,
    analyze_bluetooth_hdt_recordings,
)
import pluto_vsa.protocol_modes.bluetooth.model as bluetooth_model
from pluto_vsa.protocol_modes.bluetooth.ui import BluetoothAnalyzerWindow
from _bluetooth_dedicated_test_helpers import _hdt_recording


def test_hdt_vectorized_trellis_step_matches_scalar_reference() -> None:
    rng = np.random.default_rng(20260906)
    costs = rng.integers(0, 100, size=32, dtype=np.int64)
    for retained in (
        np.asarray([False, True]),
        np.asarray([True, False]),
        np.asarray([True, True]),
    ):
        for observed_code in range(4):
            received = np.asarray(
                [(observed_code >> 1) & 1, observed_code & 1], dtype=np.uint8
            )
            expected_costs = np.full(32, 1_000_000, dtype=np.int64)
            expected_states = np.zeros(32, dtype=np.int16)
            expected_bits = np.zeros(32, dtype=np.uint8)
            for state in range(32):
                history = np.asarray(
                    [(state >> index) & 1 for index in range(5)], dtype=np.uint8
                )
                for bit in (0, 1):
                    registers = np.concatenate(([bit], history))
                    encoded = np.asarray(
                        [
                            registers[0]
                            ^ registers[2]
                            ^ registers[4]
                            ^ registers[5],
                            registers[0]
                            ^ registers[1]
                            ^ registers[2]
                            ^ registers[3]
                            ^ registers[5],
                        ],
                        dtype=np.uint8,
                    )
                    next_state = ((state << 1) | bit) & 0x1F
                    candidate = int(costs[state]) + int(
                        np.count_nonzero(encoded[retained] != received[retained])
                    )
                    if candidate < expected_costs[next_state]:
                        expected_costs[next_state] = candidate
                        expected_states[next_state] = state
                        expected_bits[next_state] = bit
            actual = bluetooth_model._hdt_viterbi_step(
                costs, received, retained
            )
            np.testing.assert_array_equal(actual[0], expected_costs)
            np.testing.assert_array_equal(actual[1], expected_states)
            np.testing.assert_array_equal(actual[2], expected_bits)


@pytest.mark.parametrize("rate", tuple(HDTRate))
def test_dedicated_hdt_auto_detects_rate_length_and_exact_payload_range(
    rate: HDTRate,
) -> None:
    recording, _generated, project = _hdt_recording(rate, payload_length=73)

    result = analyze_bluetooth_hdt_recording(
        recording,
        profile=BluetoothAnalysisProfile.RF_PHY_TEST,
    )

    expected_payload_symbols = next(
        field.symbol_count
        for field in project.fields
        if field.name == "Coded PDU Header / Payload / CRC"
    )
    metrics = {metric.metric_id: metric.display for metric in result.metrics}
    assert result.packet.protocol_id == "bluetooth.hdt"
    assert result.packet.phy_name == rate.value
    assert result.packet.packet_type == rate.value
    assert result.packet.integrity.complete is True
    assert metrics["payload_length"] == "73 byte(s)"
    assert metrics["automatic_result_range"] == (
        f"{expected_payload_symbols} symbol(s) (automatic)"
    )
    assert result.metadata["hdt_payload_symbol_count"] == expected_payload_symbols
    assert result.metadata["hdt_header_evm_rms_percent"] < 1.0
    assert result.metadata["hdt_payload_evm_rms_percent"] < 2.0
    hdt_evm = result.metadata["hdt_evm_result"]
    samples_per_symbol = recording.sample_rate_hz / 2_000_000.0
    assert hdt_evm.header_corrected_symbols is hdt_evm.header_measured_symbols
    assert hdt_evm.payload_corrected_symbols is hdt_evm.payload_measured_symbols
    assert hdt_evm.header_corrected_waveform.size > (
        hdt_evm.header_corrected_symbols.size
    )
    assert hdt_evm.payload_corrected_waveform.size > (
        hdt_evm.payload_corrected_symbols.size
    )
    np.testing.assert_allclose(
        np.diff(hdt_evm.header_symbol_sample_positions), samples_per_symbol
    )
    np.testing.assert_allclose(
        np.diff(hdt_evm.payload_symbol_sample_positions), samples_per_symbol
    )
    np.testing.assert_array_equal(
        result.metadata["hdt_header_symbols"], hdt_evm.header_corrected_symbols
    )
    np.testing.assert_array_equal(
        result.metadata["hdt_payload_symbols"], hdt_evm.payload_corrected_symbols
    )
    plot_data = result.metadata["hdt_plot_data"]
    assert plot_data.evm is hdt_evm
    assert plot_data.payload_sample_range == (
        result.metadata["hdt_payload_start_sample"],
        result.metadata["hdt_payload_stop_sample"],
    )
    assert plot_data.payload_evm_sample_range == (
        result.metadata["hdt_payload_start_sample"],
        result.metadata["hdt_payload_evm_stop_sample"],
    )
    assert plot_data.packet_sample_range == (
        result.metadata["packet_start_sample"],
        result.metadata["packet_stop_sample"],
    )
    assert [row.metric_id for row in result.summary_rows] == [
        "sig_hdt_output_power",
        "sig_hdt_header_evm_rms",
        "sig_hdt_payload_evm_rms",
        "sig_hdt_center_frequency_deviation",
        "sig_hdt_frequency_offset_change",
        "sig_hdt_symbol_timing_accuracy",
        "sig_hdt_pre_packet_emissions",
        "detected_phy",
        "sig_eligibility",
        "sig_hdt_preamble_carrier_error",
        "sig_hdt_payload_carrier_error",
        "sig_hdt_header_average_power",
        "sig_hdt_payload_average_power",
        "sig_hdt_relative_power",
        "sig_hdt_training_correlation",
        "sig_hdt_evm_packets_evaluated",
    ]
    payload_evm_row = next(
        row
        for row in result.summary_rows
        if row.metric_id == "sig_hdt_payload_evm_rms"
    )
    expected_limit_db = {
        HDTRate.HDT2: -10,
        HDTRate.HDT3: -13,
        HDTRate.HDT4: -16,
        HDTRate.HDT6: -19,
        HDTRate.HDT7_5: -22,
    }[rate]
    assert payload_evm_row.limit == f"≤ {expected_limit_db} dB"
    assert all(
        row.limit == "—" and row.result == "—"
        for row in result.summary_rows
        if row.section == "Reference Information"
    )
    field_ids = {field.field_id for field in result.packet.root_fields}
    assert field_ids == {"training", "control_header", "payload"}
    control = next(
        field
        for field in result.packet.root_fields
        if field.field_id == "control_header"
    )
    children = {field.field_id: field for field in control.children}
    assert children["rate_indicator"].meaning.startswith(rate.value)
    assert children["rate_indicator"].value == (
        f"{rate.value} (0b{result.metadata['hdt_rate_indicator']:03b})"
    )
    assert children["pdu_control"].value == 74
    assert children["hec_c"].status.value == "valid"
    payload = next(
        field for field in result.packet.root_fields if field.field_id == "payload"
    )
    payload_body = next(
        field for field in payload.children if field.field_id == "payload_body"
    )
    assert " " in payload_body.value
    assert result.packet.integrity.hec_valid is True
    assert result.packet.integrity.crc_valid is True


@pytest.mark.parametrize("rate", (HDTRate.HDT2, HDTRate.HDT3))
def test_hdt_qpsk_long_packet_uses_1000_symbol_range_and_marks_boundary(
    rate: HDTRate, tmp_path
) -> None:
    pg.mkQApp(f"Bluetooth dedicated {rate.value} long range test")
    recording, _generated, _project = _hdt_recording(rate, payload_length=255)
    result = analyze_bluetooth_hdt_recording(
        recording, profile=BluetoothAnalysisProfile.RF_PHY_TEST
    )
    plot_data = result.metadata["hdt_plot_data"]
    assert plot_data.packet_sample_range[1] > plot_data.payload_evm_sample_range[1]
    window = BluetoothAnalyzerWindow(
        preferences=QtCore.QSettings(
            str(tmp_path / f"bluetooth-{rate.value}-long-range.ini"),
            QtCore.QSettings.Format.IniFormat,
        )
    )
    try:
        window._recording = recording
        window._classic_analysis_ready((result,))
        regions = [
            tuple(item.getRegion())
            for item in window.power_plot.getPlotItem().items
            if isinstance(item, pg.LinearRegionItem)
        ]
        expected_result_ms = tuple(
            sample / recording.sample_rate_hz * 1e3
            for sample in plot_data.payload_evm_sample_range
        )
        assert any(np.allclose(region, expected_result_ms) for region in regions)
        packet_stop_ms = (
            plot_data.packet_sample_range[1] / recording.sample_rate_hz * 1e3
        )
        boundary_lines = [
            item
            for item in window.power_plot.getPlotItem().items
            if isinstance(item, pg.InfiniteLine)
            and np.isclose(float(item.value()), packet_stop_ms)
        ]
        assert len(boundary_lines) == 1
        assert boundary_lines[0].label.format == "Packet End"
        assert boundary_lines[0].pen.style() == QtCore.Qt.PenStyle.SolidLine
        assert boundary_lines[0].pen.width() == 1
        assert window.power_plot.viewRange()[0][1] > packet_stop_ms
    finally:
        window.close()
        window.deleteLater()


def test_dedicated_hdt_returns_every_packet_in_capture() -> None:
    recording, generated, project = _hdt_recording(HDTRate.HDT7_5, 32)
    spacer = np.zeros(128, dtype=np.complex64)
    repeated = replace(
        recording,
        iq=np.concatenate((spacer, generated.iq, spacer, generated.iq, spacer)),
        source="two generated HDT7.5 packets",
    )

    results = analyze_bluetooth_hdt_recordings(
        repeated,
        profile=BluetoothAnalysisProfile.RF_PHY_TEST,
    )

    assert len(results) == 2
    assert all(result.packet.phy_name == HDTRate.HDT7_5.value for result in results)
    expected_payload_symbols = next(
        field.symbol_count
        for field in project.fields
        if field.name == "Coded PDU Header / Payload / CRC"
    )
    assert all(
        result.metadata["hdt_payload_symbol_count"]
        == expected_payload_symbols
        for result in results
    )
    first_plot = results[0].metadata["hdt_plot_data"]
    second_plot = results[1].metadata["hdt_plot_data"]
    assert first_plot.packet_sample_range[1] < second_plot.packet_sample_range[0]
    assert second_plot.packet_sample_range[0] == results[1].metadata[
        "packet_start_sample"
    ]
    assert second_plot.payload_evm_sample_range == (
        results[1].metadata["hdt_payload_start_sample"],
        results[1].metadata["hdt_payload_evm_stop_sample"],
    )
    assert all(
        result.metadata["hdt_rms_evm_aggregate_status"]
        == "MEASURING 2 / 1500"
        for result in results
    )
    for result in results:
        evaluated = next(
            metric
            for metric in result.metrics
            if metric.metric_id == "sig_hdt_evm_packets_evaluated"
        )
        assert evaluated.display == "2 / 1500"


def test_hdt_post_capture_trigger_gates_packets_without_changing_evm() -> None:
    recording, _generated, _project = _hdt_recording(HDTRate.HDT7_5, 32)
    baseline = analyze_bluetooth_hdt_recordings(recording, profile=BluetoothAnalysisProfile.RF_PHY_TEST)
    threshold = recording.dbfs_to_dbm_offset_db - 20
    gated = analyze_bluetooth_hdt_recordings(
        recording, profile=BluetoothAnalysisProfile.RF_PHY_TEST,
        iq_power_trigger=IQPowerTriggerSettings(enabled=True, level_dbm=threshold),
    )
    assert [item.metadata["packet_start_sample"] for item in gated] == [item.metadata["packet_start_sample"] for item in baseline]
    assert [item.metrics for item in gated] == [item.metrics for item in baseline]
    with pytest.raises(RuntimeError, match="synchronization pattern"):
        analyze_bluetooth_hdt_recordings(recording, profile=BluetoothAnalysisProfile.RF_PHY_TEST,
                                        iq_power_trigger=IQPowerTriggerSettings(enabled=True, level_dbm=100))
