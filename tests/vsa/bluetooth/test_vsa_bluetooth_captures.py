import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace
from pathlib import Path
import numpy as np
import pyqtgraph as pg
import pytest
from pyqtgraph.Qt import QtCore
from pluto_protocol.bluetooth.hdt import HDTRate
from pluto_vsa.model import IQRecording
from pluto_vsa.protocol_modes.bluetooth.model import (
    BluetoothAnalysisProfile,
    analyze_bluetooth_classic_recording,
    analyze_bluetooth_classic_recordings,
    analyze_bluetooth_hdt_recording,
    analyze_bluetooth_hdt_recordings,
    analyze_bluetooth_le_recording,
)
import pluto_vsa.protocol_modes.bluetooth.model as bluetooth_model
from pluto_vsa.profiles.bluetooth_br import access_code_bits
from pluto_vsa.protocol_modes.bluetooth.ui import BluetoothAnalyzerWindow
from pluto_vsa.sources import FileIQSource
from pluto_vsg.engine import BluetoothBRWaveformEngine
from pluto_vsg.model import BluetoothPacketKind
from pluto_vsg.profiles import bluetooth_br_edr_project, bluetooth_br_fields


def test_dedicated_hdt_decodes_real_hdt7_5_and_identifies_legacy_crc_init() -> None:
    recording = FileIQSource.load(
        Path(__file__).resolve().parents[2] / "data" / "fixtures" / "bluetooth" / "hdt" / "RT_HDT7_5.npz"
    )

    result = analyze_bluetooth_hdt_recording(
        recording,
        profile=BluetoothAnalysisProfile.RF_PHY_TEST,
    )

    assert result.packet.phy_name == HDTRate.HDT7_5.value
    assert result.packet.integrity.hec_valid is True
    assert result.packet.integrity.crc_valid is False
    assert result.metadata["hdt_pca_a"] == 0x9F15
    assert result.metadata["hdt_nesn"] == 1
    assert result.metadata["hdt_packet_format_indicator"] == 0
    assert result.metadata["hdt_rate_indicator"] == 0b101
    assert result.metadata["hdt_pdu_control_octets"] == 510
    assert result.metadata["hdt_received_hec_c"] == 0x13FB5A
    assert result.metadata["hdt_control_path_errors"] == 0
    assert result.metadata["hdt_payload_path_errors"] == 0
    assert result.metadata["hdt_received_crc32"] == 0xBB166D73
    assert result.metadata["hdt_calculated_crc32"] == 0xCDCA2EBD
    assert result.metadata["hdt_legacy_init_crc32_match"] is True
    assert result.metadata["hdt_header_evm_rms_percent"] == pytest.approx(
        5.47, abs=0.25
    )
    assert result.metadata["hdt_payload_evm_rms_percent"] == pytest.approx(
        4.01, abs=0.25
    )
    assert result.metadata["hdt_payload_evm_symbol_count"] == 1000
    assert result.metadata["hdt_payload_reference_source"] == (
        "decoded_reencoded_bits"
    )
    assert (
        result.metadata["hdt_preamble_carrier_error_hz"]
        - result.metadata["hdt_payload_carrier_error_hz"]
    ) == pytest.approx(189.0, abs=25.0)
    for key in (
        "hdt_alpha0",
        "hdt_phi0_rad",
        "hdt_delta_omega0_rad_per_symbol",
        "hdt_t0_sample",
        "hdt_phi1_rad",
        "hdt_delta_omega1_rad_per_symbol",
    ):
        assert np.isfinite(result.metadata[key])
    metrics = {metric.label: metric for metric in result.metrics if metric.group}
    assert metrics["Output power"].display.endswith(" dBm")
    assert metrics["Output power"].limit == "Power Class dependent"
    assert metrics["Output power"].result == "N/A"
    assert result.metadata["hdt_output_power_measurement_status"] == "provisional"
    assert result.metadata["hdt_output_power_window_start_sample"] < (
        result.metadata["hdt_output_power_window_stop_sample"]
    )
    header_trajectory = np.asarray(result.metadata["hdt_header_trajectory"])
    assert header_trajectory.size > result.metadata["hdt_header_symbols"].size
    assert np.all(np.isfinite(header_trajectory))
    assert np.quantile(np.abs(header_trajectory), 0.95) < 1.5
    assert metrics["Control Header RMS EVM"].result == "PASS"
    assert metrics["Control Header RMS EVM"].limit == "≤ -10 dB"
    assert metrics["PDU Header and payload RMS EVM"].result == "PASS"
    assert metrics["PDU Header and payload RMS EVM"].limit == "≤ -22 dB"
    assert metrics["Center frequency deviation"].display == "+14.103 kHz"
    assert metrics["Center frequency deviation"].result == "PASS"
    assert metrics[
        "Center frequency offset change between the preamble and the payload"
    ].display == "0.189 kHz"
    assert metrics["Symbol timing accuracy"].result == "N/A"
    assert metrics["Pre-packet emissions"].result == "N/A"


def test_dedicated_hdt_detects_packet_tx_lts_variant() -> None:
    recording = FileIQSource.load(
        Path(__file__).resolve().parents[2] / "data" / "fixtures" / "bluetooth" / "hdt" / "RT_Packet_TX_HDT7P5_temp.npz"
    )

    results = analyze_bluetooth_hdt_recordings(
        recording,
        profile=BluetoothAnalysisProfile.RF_PHY_TEST,
    )

    assert len(results) == 2
    assert all(result.packet.phy_name == HDTRate.HDT7_5.value for result in results)
    assert all(result.metadata["hdt_lts_root"] == 1 for result in results)
    assert all(result.metadata["hdt_lts_phase"] == 2 for result in results)
    assert all(result.metadata["hdt_control_path_errors"] == 0 for result in results)
    assert all(result.metadata["hdt_pdu_control_octets"] == 54 for result in results)
    assert all(result.metadata["hdt_pdu_control_includes_crc"] for result in results)
    assert all(result.metadata["hdt_payload_length_bytes"] == 50 for result in results)
    assert all(result.metadata["hdt_payload_symbol_count"] == 119 for result in results)
    assert all(
        result.metadata["hdt_payload_terminating_symbol_count"] == 0
        for result in results
    )
    assert all(result.metadata["hdt_payload_path_errors"] == 0 for result in results)
    assert all(
        result.metadata["hdt_payload_evm_rms_percent"] < 6.0
        for result in results
    )
    assert all(
        abs(
            result.metadata["hdt_preamble_carrier_error_hz"]
            - result.metadata["hdt_payload_carrier_error_hz"]
        )
        < 500.0
        for result in results
    )


@pytest.mark.parametrize(
    ("filename", "whitening", "expected_phy", "expected_packet", "expected_start"),
    (
        ("DH1_test.npz", False, "BR", "DH1", 96),
        ("bluetooth_br_prbs9_pluto_16msps.npz", False, "BR", "DH1", 15206),
        ("bluetooth_2dh1_prbs9_16msps.npz", True, "EDR 2M", "2-DH1", 32001),
        ("bluetooth_3dh1_prbs9_16msps.npz", True, "EDR 3M", "3-DH1", 32001),
        ("PLUTO_VSG_SMCV100B_2DH1.npz", False, "EDR 2M", "2-DH1", 2074),
    ),
)
def test_real_classic_fixtures_preserve_sync_decode_and_symbol_products(
    filename: str,
    whitening: bool,
    expected_phy: str,
    expected_packet: str,
    expected_start: int,
) -> None:
    recording = FileIQSource.load(Path(__file__).resolve().parents[2] / "data" / "fixtures" / "bluetooth" / "br-edr" / filename)
    result = analyze_bluetooth_classic_recording(
        recording,
        profile=BluetoothAnalysisProfile.RF_PHY_TEST,
        lap=0xC6967E,
        uap=0x6B,
        clock_6_1=0x2B,
        whitening_enabled=whitening,
        result_length=4096,
    )
    pattern = result.metadata["br_analysis_session"].pattern_result
    assert pattern is not None
    assert pattern.correlation > 0.98
    assert pattern.pattern_start_sample == pytest.approx(expected_start, abs=2)
    assert pattern.result_start_sample == pattern.pattern_start_sample
    assert pattern.metadata["eligible_match_count"] >= 1
    assert pattern.decoded_bits.size == 126
    np.testing.assert_array_equal(pattern.decoded_bits[:72], access_code_bits(0xC6967E))
    first_center = pattern.symbol_time_s[0] * recording.sample_rate_hz
    samples_per_symbol = recording.sample_rate_hz / 1_000_000.0
    assert first_center == pytest.approx(
        pattern.pattern_start_sample + 0.5 * samples_per_symbol, abs=1.0
    )
    assert result.packet.phy_name == expected_phy
    assert result.packet.packet_type == expected_packet
    assert result.packet.integrity.crc_valid is True
    assert result.packet.integrity.complete is True
    assert result.vsa_result.measured_symbols.size > 0
    if expected_phy.startswith("EDR"):
        assert result.metadata["analysis_session"].pattern_result.decoded_symbols.size > 10
        measurement = result.metadata["rf_measurements"][0]
        assert measurement.metrics["rms_devm_worst"] < 0.10
        assert measurement.metrics["peak_devm_worst"] < 0.15
        assert measurement.metrics["omega0_abs_worst_hz"] < 10_000.0


@pytest.mark.parametrize(
    ("filename", "whitening", "expected_packet"),
    (
        ("bluetooth_2dh1_prbs9_16msps.npz", True, "2-DH1"),
        ("bluetooth_3dh1_prbs9_16msps.npz", True, "3-DH1"),
        ("PLUTO_VSG_SMCV100B_2DH1.npz", False, "2-DH1"),
    ),
)
def test_edr_sig_devm_uses_reference_plus_50_physical_symbols_and_shared_centers(
    filename: str,
    whitening: bool,
    expected_packet: str,
) -> None:
    recording = FileIQSource.load(Path(__file__).resolve().parents[2] / "data" / "fixtures" / "bluetooth" / "br-edr" / filename)
    result = analyze_bluetooth_classic_recording(
        recording,
        profile=BluetoothAnalysisProfile.RF_PHY_TEST,
        lap=0xC6967E,
        uap=0x6B,
        clock_6_1=0x2B,
        whitening_enabled=whitening,
        result_length=4096,
    )
    assert result.packet.packet_type == expected_packet
    assert result.packet.integrity.crc_valid is True
    session = result.metadata["analysis_session"]
    pattern = session.pattern_result
    measurement = result.metadata["rf_measurements"][0]
    sps = recording.sample_rate_hz / session.signal.symbol_rate_hz

    sync_center = float(result.metadata["edr_sync_first_symbol_center_sample"])
    reference_center = float(result.metadata["edr_reference_symbol_center_sample"])
    payload_center = float(result.metadata["edr_payload_first_symbol_center_sample"])
    trailer_center = float(result.metadata["edr_trailer_first_symbol_center_sample"])
    expected_sync_center = (
        float(result.metadata["analysis_sample_offset"])
        + float(pattern.symbol_time_s[0]) * recording.sample_rate_hz
    )
    assert abs(sync_center - expected_sync_center) <= 0.5 * sps
    assert reference_center == pytest.approx(sync_center - sps, abs=1e-6)
    assert payload_center == pytest.approx(sync_center + 10 * sps, abs=1e-6)
    assert trailer_center == pytest.approx(
        sync_center + (pattern.decoded_symbols.size - 2) * sps,
        abs=1e-6,
    )

    block_centers = measurement.arrays["block_physical_symbol_center_samples"]
    corrected = measurement.arrays["block_corrected_received_symbols"]
    references = measurement.arrays["block_reference_symbols"]
    errors = measurement.arrays["block_differential_error_vectors"]
    assert block_centers.shape[1] == 51
    assert corrected.shape == references.shape == block_centers.shape
    assert errors.shape == (block_centers.shape[0], 50)
    assert block_centers[0, 0] == pytest.approx(
        reference_center - result.metadata["recording_sample_offset"], abs=1e-6
    )
    assert block_centers[0, 1] == pytest.approx(
        sync_center - result.metadata["recording_sample_offset"], abs=1e-6
    )
    assert block_centers[0, 11] == pytest.approx(
        payload_center - result.metadata["recording_sample_offset"], abs=1e-6
    )
    np.testing.assert_allclose(np.diff(block_centers, axis=1), sps, atol=1e-5)
    np.testing.assert_allclose(errors, np.diff(corrected * np.conj(references), axis=1))
    recalculated_rms = np.sqrt(
        np.sum(np.abs(errors) ** 2, axis=1)
        / np.sum(np.abs(corrected[:, 1:] * np.conj(references[:, 1:])) ** 2, axis=1)
    )
    np.testing.assert_allclose(
        recalculated_rms,
        measurement.arrays["block_rms_devm"],
        rtol=1e-10,
        atol=1e-12,
    )
    assert measurement.metadata["reference_source"] == "decoded_reencoded_packet"
    assert measurement.eligibility.eligible is False
    assert "complete known EDR RF Test reference unavailable" in (
        measurement.eligibility.reasons
    )


@pytest.mark.parametrize("filename", ("LE1M_FSK_error_raw.npz", "LE1M_FSK_error.npz"))
def test_real_le_rf_test_fixtures_preserve_sync_and_symbol_products(filename: str) -> None:
    recording = FileIQSource.load(Path(__file__).resolve().parents[2] / "data" / "fixtures" / "bluetooth" / "le" / filename)
    result = analyze_bluetooth_le_recording(
        recording,
        profile=BluetoothAnalysisProfile.RF_PHY_TEST,
        phy="LE 1M",
        access_address=0x71764129,
        channel_index=18,
        crc_init=0x555555,
        whitening_enabled=False,
        result_length=4096,
    )
    pattern = result.metadata["analysis_session"].pattern_result
    expected_access = np.unpackbits(
        np.frombuffer((0x71764129).to_bytes(4, "little"), dtype=np.uint8),
        bitorder="little",
    )
    expected_sync = np.concatenate(
        (np.resize(np.asarray([1, 0], dtype=np.uint8), 8), expected_access)
    )
    assert pattern is not None
    assert pattern.correlation > 0.99
    assert pattern.pattern_start_sample == pytest.approx(106, abs=2)
    assert pattern.result_start_sample == pattern.pattern_start_sample
    assert pattern.metadata["eligible_match_count"] >= 1
    np.testing.assert_array_equal(pattern.decoded_bits[:40], expected_sync)
    assert pattern.decoded_bits.size >= 408
    assert result.packet.phy_name == "LE 1M"
    assert result.vsa_result.measured_symbols.size > 0


def test_sync_is_independent_of_rf_measurement_profile_and_failure(monkeypatch) -> None:
    recording = FileIQSource.load(
        Path(__file__).resolve().parents[2] / "data" / "fixtures" / "bluetooth" / "br-edr" / "bluetooth_br_prbs9_pluto_16msps.npz"
    )
    options = {
        "lap": 0xC6967E,
        "uap": 0x6B,
        "clock_6_1": 0x2B,
        "whitening_enabled": False,
        "result_length": 4096,
    }
    general = analyze_bluetooth_classic_recording(
        recording, profile=BluetoothAnalysisProfile.GENERAL_PACKET, **options
    )
    general_pattern = general.metadata["br_analysis_session"].pattern_result

    def fail_measurement(*_args, **_kwargs):
        raise RuntimeError("injected RF measurement failure")

    monkeypatch.setattr(bluetooth_model, "build_fm_measurement_trace", fail_measurement)
    rf_test = analyze_bluetooth_classic_recording(
        recording, profile=BluetoothAnalysisProfile.RF_PHY_TEST, **options
    )
    rf_pattern = rf_test.metadata["br_analysis_session"].pattern_result
    assert rf_pattern.pattern_start_sample == general_pattern.pattern_start_sample
    assert rf_pattern.result_start_sample == general_pattern.result_start_sample
    np.testing.assert_array_equal(rf_pattern.decoded_bits, general_pattern.decoded_bits)
    assert rf_test.packet.packet_type == general.packet.packet_type == "DH1"
    assert rf_test.packet.integrity.crc_valid is True
    measurement = rf_test.metadata["rf_measurements"][0]
    assert measurement.eligibility.eligible is False
    assert "injected RF measurement failure" in measurement.metadata["reason"]


def test_real_le_packet_end_uses_decoded_length_not_available_result_tail(
    tmp_path,
) -> None:
    pg.mkQApp("Bluetooth LE exact packet-end regression")
    recording = FileIQSource.load(
        Path(__file__).resolve().parents[2] / "data" / "fixtures" / "bluetooth" / "le" / "LE1M_packet_length.npz"
    )
    result = analyze_bluetooth_le_recording(
        recording,
        profile=BluetoothAnalysisProfile.RF_PHY_TEST,
        phy="LE 1M",
        access_address=0x71764129,
        channel_index=18,
        crc_init=0x555555,
        whitening_enabled=False,
        result_length=4096,
    )
    pattern = result.metadata["analysis_session"].pattern_result
    assert result.packet.integrity.crc_valid is True
    assert result.metadata["packet_symbol_count"] == result.packet.raw_bits.size == 376
    assert pattern.decoded_bits.size > result.packet.raw_bits.size
    expected_stop = result.metadata["packet_start_sample"] + int(
        round(
            result.packet.raw_bits.size
            * recording.sample_rate_hz
            / 1_000_000.0
        )
    )
    assert result.metadata["packet_stop_sample"] == expected_stop
    assert result.metadata["physical_packet_stop_sample"] == expected_stop
    assert result.metadata["packet_stop_source"] == "decoded_pdu_length"
    assert result.packet.source.stop_sample == expected_stop

    window = BluetoothAnalyzerWindow(
        preferences=QtCore.QSettings(
            str(tmp_path / "le-packet-end.ini"),
            QtCore.QSettings.Format.IniFormat,
        )
    )
    try:
        window._recording = recording
        window._classic_analysis_ready((result,))
        expected_stop_ms = expected_stop / recording.sample_rate_hz * 1e3
        result_regions = [
            item
            for item in window.power_plot.getPlotItem().items
            if isinstance(item, pg.LinearRegionItem)
        ]
        assert any(
            np.isclose(float(region.getRegion()[1]), expected_stop_ms)
            for region in result_regions
        )
        packet_end_lines = [
            item
            for item in window.power_plot.getPlotItem().items
            if isinstance(item, pg.InfiniteLine)
            and item.label is not None
            and item.label.format == "Packet End"
        ]
        assert len(packet_end_lines) == 1
        assert float(packet_end_lines[0].value()) == pytest.approx(
            expected_stop_ms
        )
        fsk_items = window.fsk_modulation_plot.listDataItems()
        marker = next(item for item in fsk_items if item.opts.get("symbol") is not None)
        assert marker.xData.size == result.packet.raw_bits.size
    finally:
        window.close()
        window.deleteLater()


@pytest.mark.parametrize(
    "fixture_name",
    ("3-DH3_misjudge.npz", "3-DH3_misjudge2.npz"),
)
def test_real_3dh3_requires_edr_sync_before_br_fallback(fixture_name) -> None:
    recording = FileIQSource.load(
        Path(__file__).resolve().parents[2] / "data" / "fixtures" / "bluetooth" / "br-edr" / fixture_name
    )
    results = analyze_bluetooth_classic_recordings(
        recording,
        profile=BluetoothAnalysisProfile.RF_PHY_TEST,
        lap=0xC6967E,
        uap=0x6B,
        clock_6_1=0x2B,
        whitening_enabled=False,
        result_length=4096,
    )
    assert results
    assert all(result.packet.phy_name == "EDR 3M" for result in results)
    assert all(result.packet.packet_type == "3-DH3" for result in results)
    assert all(result.packet.integrity.crc_valid is True for result in results)
    assert all(result.metadata["edr_sync_confirmed"] is True for result in results)
    assert all(
        float(result.metadata["edr_sync_correlation"]) > 0.99
        for result in results
    )
    assert all(
        result.metadata["analysis_session"].pattern_result.correlation > 0.99
        for result in results
    )


def test_high_confidence_edr_sync_is_not_demoted_when_length_refinement_fails(
    monkeypatch,
) -> None:
    base = bluetooth_br_edr_project()
    settings = replace(
        base.bluetooth_br,
        packet_kind=BluetoothPacketKind.DH3_3,
        payload_length_bytes=54,
    )
    generated = BluetoothBRWaveformEngine().generate(
        replace(base, bluetooth_br=settings, fields=bluetooth_br_fields(settings))
    )
    recording = IQRecording(
        iq=generated.iq,
        sample_rate_hz=generated.sample_rate_hz,
        center_frequency_hz=base.center_frequency_hz,
    )
    monkeypatch.setattr(
        bluetooth_model,
        "_exact_edr_result_symbols",
        lambda *_args, **_kwargs: None,
    )

    result = analyze_bluetooth_classic_recording(
        recording,
        profile=BluetoothAnalysisProfile.GENERAL_PACKET,
        lap=settings.lap,
        uap=settings.uap,
        clock_6_1=settings.clock_6_1,
        whitening_enabled=settings.whitening_enabled,
        result_length=4096,
    )

    assert result.packet.phy_name == "EDR 3M"
    assert result.packet.packet_type == "3-DH3"
    assert result.metadata["edr_sync_confirmed"] is True
    assert float(result.metadata["edr_sync_correlation"]) > 0.99
    assert "Length was not decoded" in result.metadata["edr_candidate_error"]
