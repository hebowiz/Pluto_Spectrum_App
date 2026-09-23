import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace
import numpy as np
import pyqtgraph as pg
import pytest
from pyqtgraph.Qt import QtCore
from pluto_vsa.model import IQRecording
from pluto_vsa.pattern import IQPowerTriggerSettings
from pluto_vsa.protocol_modes.bluetooth.model import (
    BluetoothAnalysisProfile,
    analyze_bluetooth_classic_recording,
    analyze_bluetooth_le_recording,
    analyze_bluetooth_le_recordings,
    analyze_bluetooth_session,
)
from pluto_vsa.protocol_modes.bluetooth.rf_measurement import (
    BluetoothFMMeasurementTrace,
    BluetoothRFMeasurementFilterProfile,
)
from pluto_vsa.protocol_modes.bluetooth.ui import BluetoothAnalyzerWindow
from pluto_vsg.engine import BluetoothBRWaveformEngine, BluetoothLEWaveformEngine
from pluto_vsg.model import BluetoothLEPhy, BluetoothPacketKind
from pluto_vsg.model import PayloadSourceKind
from pluto_vsg.profiles import (
    bluetooth_br_edr_project,
    bluetooth_br_fields,
    bluetooth_le_project,
    bluetooth_le_test_project,
)
from _bluetooth_dedicated_test_helpers import _session_with_le_bits


def test_dedicated_model_combines_rf_metrics_and_shared_le_decode() -> None:
    session, context = _session_with_le_bits()
    result = analyze_bluetooth_session(
        session,
        profile=BluetoothAnalysisProfile.GENERAL_PACKET,
        protocol_id="bluetooth.le",
        phy_name="LE 1M",
        context=context,
    )
    assert result.packet.protocol_id == "bluetooth.le"
    assert result.packet.integrity.crc_valid is True
    assert any(metric.metric_id == "packet_power" for metric in result.metrics)


def test_le_rf_profile_uses_test_sync_word_and_general_hides_identity_inputs(
    tmp_path,
) -> None:
    pg.mkQApp("Bluetooth LE profile config test")
    preferences = QtCore.QSettings(
        str(tmp_path / "bluetooth-le-profile.ini"),
        QtCore.QSettings.Format.IniFormat,
    )
    window = BluetoothAnalyzerWindow(preferences=preferences)
    try:
        window.protocol_combo.setCurrentIndex(
            window.protocol_combo.findData("bluetooth.le")
        )
        assert window.profile_combo.currentData() == BluetoothAnalysisProfile.RF_PHY_TEST
        assert window.access_address_edit.text() == "71764129"
        assert window.crc_init_edit.text() == "555555"
        assert window.whitening_check.isChecked() is False
        assert window.access_address_edit.isEnabled() is False
        assert window.crc_init_edit.isEnabled() is False
        assert window.whitening_check.isEnabled() is False
        rf_options = window._le_options()
        assert rf_options["access_address"] == 0x71764129
        assert rf_options["crc_init"] == 0x555555
        assert rf_options["whitening_enabled"] is False

        window.profile_combo.setCurrentIndex(
            window.profile_combo.findData(BluetoothAnalysisProfile.GENERAL_PACKET)
        )
        assert window.access_address_edit.isEnabled() is True
        assert window.crc_init_edit.isEnabled() is True
        assert window.whitening_check.isEnabled() is True
        assert window.access_address_edit.isHidden()
        assert window.channel_spin.isHidden()
        assert window.crc_init_edit.isHidden()
        assert window.whitening_check.isHidden()
        assert all(
            window._bluetooth_config_form.labelForField(widget).isHidden()
            for widget in (
                window.access_address_edit,
                window.channel_spin,
                window.crc_init_edit,
                window.whitening_check,
            )
        )
        window.access_address_edit.setText("8E89BED6")
        window.crc_init_edit.setText("123456")
        window.whitening_check.setChecked(False)
        general_options = window._le_options()
        assert general_options["access_address"] is None
        assert general_options["channel_index"] == 17
        assert general_options["crc_init"] == 0x555555
        assert general_options["whitening_enabled"] is True
    finally:
        window.close()
        window.deleteLater()


def test_dedicated_le_analyzer_synchronizes_generated_iq_and_shared_crc() -> None:
    generated = BluetoothLEWaveformEngine().generate(bluetooth_le_project(BluetoothLEPhy.LE_1M))
    recording = IQRecording(
        iq=generated.iq,
        sample_rate_hz=generated.sample_rate_hz,
        center_frequency_hz=2_440e6,
        source="generated LE packet",
    )
    result = analyze_bluetooth_le_recording(
        recording,
        profile=BluetoothAnalysisProfile.GENERAL_PACKET,
        phy="LE 1M",
        access_address=0x8E89BED6,
        channel_index=37,
        crc_init=0x555555,
        whitening_enabled=True,
        result_length=512,
    )
    assert result.packet.protocol_id == "bluetooth.le"
    assert result.packet.integrity.crc_valid is True
    assert result.packet.raw_bits.size == generated.packet_bits.bits.size


@pytest.mark.parametrize(
    ("phy", "expected_deviation_hz"),
    ((BluetoothLEPhy.LE_1M, 250_000.0), (BluetoothLEPhy.LE_2M, 500_000.0)),
)
def test_le_rf_test_packet_produces_eligible_raw_sig_measurements(
    phy: BluetoothLEPhy, expected_deviation_hz: float
) -> None:
    project = bluetooth_le_test_project(phy)
    generated = BluetoothLEWaveformEngine().generate(project)
    result = analyze_bluetooth_le_recording(
        IQRecording(
            iq=generated.iq,
            sample_rate_hz=generated.sample_rate_hz,
            center_frequency_hz=project.center_frequency_hz,
            source=f"generated {phy.value} RF test packet",
        ),
        profile=BluetoothAnalysisProfile.RF_PHY_TEST,
        phy=phy.value,
        access_address=0x71764129,
        channel_index=0,
        crc_init=0x555555,
        whitening_enabled=False,
        result_length=512,
    )

    measurement = result.metadata["rf_measurements"][0]
    measurement_trace = result.metadata["fsk_measurement_trace"]
    assert isinstance(measurement_trace, BluetoothFMMeasurementTrace)
    assert measurement_trace.filter_profile is {
        BluetoothLEPhy.LE_1M: BluetoothRFMeasurementFilterProfile.LE_1M,
        BluetoothLEPhy.LE_2M: BluetoothRFMeasurementFilterProfile.LE_2M,
    }[phy]
    np.testing.assert_array_equal(
        measurement_trace.frequency_hz,
        measurement.arrays["frequency_hz"],
    )
    assert measurement.metadata["p0_method"].startswith("RF.TS/RFPHY.TS")
    assert measurement.metadata["p0_zero_crossing_count"] > 0
    assert np.isfinite(measurement.metadata["p0_sample"])
    assert measurement.metadata["p0_correction_samples"] == pytest.approx(
        -0.5, abs=0.1
    )
    assert measurement.eligibility.eligible is True
    assert measurement.metadata["payload_pattern"] == "10101010"
    assert measurement.metrics["delta_f2_avg_hz"] is not None
    assert result.metadata["analysis_session"].signal.frequency_deviation_hz == (
        expected_deviation_hz
    )
    assert [row.metric_id for row in result.summary_rows] == [
        "output_power",
        "delta_f1_avg",
        "delta_f1_min",
        "delta_f1_max",
        "delta_f2_avg",
        "delta_f2_min",
        "delta_f2_max",
        "delta_f2_p999",
        "delta_f2_ratio",
        "initial_carrier_frequency",
        "carrier_frequency_drift",
        "carrier_frequency_drift_rate",
        "detected_phy",
        "rf_test_eligibility",
        "payload_pattern",
        "sync_correlation",
        "packets_evaluated",
        "peak_power",
        "mean_abs_fsk_deviation",
        "p999_fsk_deviation",
        "max_fsk_deviation",
    ]
    summary = {row.metric_id: row for row in result.summary_rows}
    expected_f1_limit = {
        BluetoothLEPhy.LE_1M: "225 kHz ≤ Δf1avg ≤ 275 kHz",
        BluetoothLEPhy.LE_2M: "450 kHz ≤ Δf1avg ≤ 550 kHz",
    }[phy]
    assert summary["delta_f1_avg"].limit == expected_f1_limit
    assert summary["delta_f1_avg"].result == "MEASURING"
    assert summary["delta_f2_p999"].value != "N/A"
    assert summary["delta_f2_p999"].limit == {
        BluetoothLEPhy.LE_1M: "≥ 185 kHz",
        BluetoothLEPhy.LE_2M: "≥ 370 kHz",
    }[phy]
    assert summary["delta_f2_p999"].result == "MEASURING"
    assert summary["delta_f2_ratio"].result == "MEASURING"
    assert summary["initial_carrier_frequency"].limit == "±150 kHz"
    assert summary["carrier_frequency_drift"].limit == "< 50 kHz"
    assert summary["carrier_frequency_drift_rate"].limit == "≤ 20 kHz / 50 µs"
    for metric_id in (
        "mean_abs_fsk_deviation",
        "p999_fsk_deviation",
        "max_fsk_deviation",
    ):
        assert summary[metric_id].value.endswith(" kHz")
        assert summary[metric_id].limit == "—"
        assert summary[metric_id].result == "—"
    assert all(
        row.limit == "—" and row.result == "—"
        for row in result.summary_rows
        if row.section == "Reference Information"
    )


def test_br_rf_test_packet_produces_eligible_raw_sig_measurements() -> None:
    base = bluetooth_br_edr_project()
    settings = replace(
        base.bluetooth_br,
        packet_kind=BluetoothPacketKind.DH1,
        payload_length_bytes=27,
        payload_source=PayloadSourceKind.PATTERN,
        payload_pattern="10101010",
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
            source="generated BR RF test packet",
        ),
        profile=BluetoothAnalysisProfile.RF_PHY_TEST,
        lap=settings.lap,
        uap=settings.uap,
        clock_6_1=settings.clock_6_1,
        whitening_enabled=False,
    )

    measurement = result.metadata["rf_measurements"][0]
    assert measurement.eligibility.eligible is True
    assert measurement.metadata["p0_method"].startswith("RF.TS/RFPHY.TS")
    assert measurement.metadata["p0_zero_crossing_count"] > 0
    assert measurement.metadata["p0_correction_samples"] == pytest.approx(
        -0.5, abs=0.1
    )
    assert measurement.metadata["payload_pattern"] == "10101010"
    assert measurement.metrics["delta_f2_avg_hz"] is not None
    assert [row.metric_id for row in result.summary_rows] == [
        "output_power",
        "delta_f1_avg",
        "delta_f1_min",
        "delta_f1_max",
        "delta_f2_avg",
        "delta_f2_min",
        "delta_f2_max",
        "delta_f2_p999",
        "delta_f2_ratio",
        "initial_carrier_frequency",
        "carrier_frequency_drift",
        "carrier_frequency_drift_rate",
        "detected_phy",
        "rf_test_eligibility",
        "packet_type",
        "payload_pattern",
        "access_code_correlation",
        "packets_evaluated",
        "peak_power",
        "mean_abs_fsk_deviation",
        "p999_fsk_deviation",
        "max_fsk_deviation",
    ]
    summary = {row.metric_id: row for row in result.summary_rows}
    assert summary["delta_f1_avg"].result == "MEASURING"
    assert summary["delta_f2_p999"].value != "N/A"
    assert summary["delta_f2_p999"].limit == "≥ 115 kHz"
    assert summary["delta_f2_p999"].result == "MEASURING"
    assert summary["delta_f2_ratio"].result == "MEASURING"
    assert summary["initial_carrier_frequency"].limit == "±75 kHz"
    assert summary["carrier_frequency_drift_rate"].limit == "≤ 20 kHz / 50 µs"
    for metric_id in (
        "mean_abs_fsk_deviation",
        "p999_fsk_deviation",
        "max_fsk_deviation",
    ):
        assert summary[metric_id].value.endswith(" kHz")
        assert summary[metric_id].limit == "—"
        assert summary[metric_id].result == "—"
    assert all(
        row.limit == "—" and row.result == "—"
        for row in result.summary_rows
        if row.section == "Reference Information"
    )


def test_br_arbitrary_payload_keeps_reference_fsk_deviation_metrics() -> None:
    base = bluetooth_br_edr_project()
    settings = replace(
        base.bluetooth_br,
        packet_kind=BluetoothPacketKind.DH1,
        payload_length_bytes=27,
        payload_source=PayloadSourceKind.PRBS9,
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
        ),
        profile=BluetoothAnalysisProfile.RF_PHY_TEST,
        lap=settings.lap,
        uap=settings.uap,
        clock_6_1=settings.clock_6_1,
        whitening_enabled=False,
    )

    measurement = result.metadata["rf_measurements"][0]
    assert measurement.metadata["payload_pattern"] is None
    assert measurement.metadata["modulation_pattern_eligible"] is False
    assert measurement.eligibility.eligible is True
    summary = {row.metric_id: row for row in result.summary_rows}
    for metric_id in (
        "delta_f1_avg",
        "delta_f2_p999",
        "delta_f2_ratio",
    ):
        assert summary[metric_id].value == "N/A"
        assert summary[metric_id].result == "N/A"
    for metric_id in (
        "delta_f1_min",
        "delta_f1_max",
        "delta_f2_avg",
        "delta_f2_min",
        "delta_f2_max",
    ):
        assert summary[metric_id].value == "N/A"
        assert summary[metric_id].result == "\N{EM DASH}"
    for metric_id in (
        "initial_carrier_frequency",
        "carrier_frequency_drift",
        "carrier_frequency_drift_rate",
    ):
        assert summary[metric_id].value != "N/A"
        assert summary[metric_id].result in {"PASS", "FAIL"}
    for metric_id in (
        "mean_abs_fsk_deviation",
        "p999_fsk_deviation",
        "max_fsk_deviation",
    ):
        assert summary[metric_id].section == "Reference Information"
        assert summary[metric_id].value.endswith(" kHz")
        assert summary[metric_id].limit == "—"
        assert summary[metric_id].result == "—"


def test_le_sig_initial_carrier_tracks_injected_cfo() -> None:
    project = bluetooth_le_test_project(BluetoothLEPhy.LE_1M)
    generated = BluetoothLEWaveformEngine().generate(project)
    injected_cfo_hz = 50_000.0
    axis = np.arange(generated.iq.size, dtype=np.float64)
    shifted = generated.iq * np.exp(
        2j * np.pi * injected_cfo_hz * axis / generated.sample_rate_hz
    )
    result = analyze_bluetooth_le_recording(
        IQRecording(
            iq=shifted,
            sample_rate_hz=generated.sample_rate_hz,
            center_frequency_hz=project.center_frequency_hz,
        ),
        profile=BluetoothAnalysisProfile.RF_PHY_TEST,
        phy="LE 1M",
        access_address=0x71764129,
        channel_index=0,
        crc_init=0x555555,
        whitening_enabled=False,
        result_length=512,
    )

    measurement = result.metadata["rf_measurements"][0]
    assert measurement.metrics["initial_carrier_error_hz"] == pytest.approx(
        injected_cfo_hz, abs=2_000.0
    )


def test_br_sig_deviation_is_not_normalized_to_nominal() -> None:
    measured: list[float] = []
    for deviation_hz in (130_000.0, 180_000.0):
        base = bluetooth_br_edr_project()
        settings = replace(
            base.bluetooth_br,
            packet_kind=BluetoothPacketKind.DH1,
            payload_length_bytes=27,
            payload_source=PayloadSourceKind.PATTERN,
            payload_pattern="10101010",
            whitening_enabled=False,
            frequency_deviation_hz=deviation_hz,
        )
        generated = BluetoothBRWaveformEngine().generate(
            replace(base, bluetooth_br=settings, fields=bluetooth_br_fields(settings))
        )
        result = analyze_bluetooth_classic_recording(
            IQRecording(
                iq=generated.iq,
                sample_rate_hz=generated.sample_rate_hz,
                center_frequency_hz=base.center_frequency_hz,
            ),
            profile=BluetoothAnalysisProfile.RF_PHY_TEST,
            lap=settings.lap,
            uap=settings.uap,
            clock_6_1=settings.clock_6_1,
            whitening_enabled=False,
        )
        measurement_trace = result.metadata["fsk_measurement_trace"]
        assert isinstance(measurement_trace, BluetoothFMMeasurementTrace)
        assert (
            measurement_trace.filter_profile
            is BluetoothRFMeasurementFilterProfile.BR_1M
        )
        measured.append(
            float(
                result.metadata["rf_measurements"][0].metrics[
                    "delta_f2_avg_hz"
                ]
            )
        )

    assert measured[1] / measured[0] == pytest.approx(180.0 / 130.0, rel=0.08)


def test_classic_rf_profile_passes_explicit_known_edr_packet_context(tmp_path) -> None:
    pg.mkQApp("Bluetooth EDR RF Test reference config")
    preferences = QtCore.QSettings(
        str(tmp_path / "bluetooth-edr-reference.ini"),
        QtCore.QSettings.Format.IniFormat,
    )
    window = BluetoothAnalyzerWindow(preferences=preferences)
    try:
        assert window.edr_rf_test_packet_combo.currentData() is None
        options = window._classic_options()
        assert options["expected_edr_rf_test_packet"] is None
        assert options["phy_search"] == "auto"
        assert [window.phy_combo.itemData(index) for index in range(window.phy_combo.count())] == [
            "auto",
            "BR",
            "EDR 2M",
            "EDR 3M",
        ]
        window.phy_combo.setCurrentIndex(window.phy_combo.findData("EDR 3M"))
        window.edr_rf_test_packet_combo.setCurrentIndex(
            window.edr_rf_test_packet_combo.findData("3-DH3")
        )
        options = window._classic_options()
        assert options["expected_edr_rf_test_packet"] == "3-DH3"
        assert options["phy_search"] == "EDR 3M"
        window.phy_combo.setCurrentIndex(window.phy_combo.findData("EDR 2M"))
        with pytest.raises(ValueError, match="does not match"):
            window._classic_options()
        window.phy_combo.setCurrentIndex(window.phy_combo.findData("EDR 3M"))
        saved = window._meas_config_values()
        assert saved["expected_edr_rf_test_packet"] == "3-DH3"
        window.edr_rf_test_packet_combo.setCurrentIndex(0)
        window._apply_meas_config_values(saved)
        assert window.edr_rf_test_packet_combo.currentData() == "3-DH3"

        window.profile_combo.setCurrentIndex(
            window.profile_combo.findData(BluetoothAnalysisProfile.GENERAL_PACKET)
        )
        assert window.edr_rf_test_packet_combo.isEnabled() is False
        assert window.edr_rf_test_packet_combo.isHidden()
        assert window.lap_edit.isHidden()
        assert window.uap_edit.isHidden()
        assert window.clock_spin.isHidden()
        assert window.whitening_check.isHidden()
        assert all(
            window._bluetooth_config_form.labelForField(widget).isHidden()
            for widget in (
                window.lap_edit,
                window.uap_edit,
                window.clock_spin,
                window.edr_rf_test_packet_combo,
                window.whitening_check,
            )
        )
        general_options = window._classic_options()
        assert general_options["lap"] is None
        assert general_options["uap"] is None
        assert general_options["clock_6_1"] is None
        assert general_options["whitening_enabled"] is True
        assert general_options["expected_edr_rf_test_packet"] is None
        assert general_options["phy_search"] == "EDR 3M"
    finally:
        window.close()
        window.deleteLater()


def test_dedicated_le_analyzer_returns_every_packet_in_capture() -> None:
    generated = BluetoothLEWaveformEngine().generate(bluetooth_le_project(BluetoothLEPhy.LE_1M))
    spacer = np.zeros(128, dtype=np.complex64)
    iq = np.concatenate((generated.iq, spacer, generated.iq))
    results = analyze_bluetooth_le_recordings(
        IQRecording(
            iq=iq,
            sample_rate_hz=generated.sample_rate_hz,
            center_frequency_hz=2_440e6,
            source="two generated LE packets",
        ),
        profile=BluetoothAnalysisProfile.GENERAL_PACKET,
        phy="LE 1M",
        access_address=0x8E89BED6,
        channel_index=37,
        crc_init=0x555555,
        whitening_enabled=True,
        result_length=512,
    )
    assert len(results) == 2
    assert all(result.packet.integrity.crc_valid is True for result in results)


def test_dedicated_le_burst_search_gates_pattern_candidates() -> None:
    generated = BluetoothLEWaveformEngine().generate(
        bluetooth_le_project(BluetoothLEPhy.LE_1M)
    )
    spacer = np.zeros(256, dtype=np.complex64)
    recording = IQRecording(
        iq=np.concatenate((spacer, generated.iq, spacer, generated.iq, spacer)),
        sample_rate_hz=generated.sample_rate_hz,
        center_frequency_hz=2_440e6,
        source="trigger-gated LE packets",
    )
    results = analyze_bluetooth_le_recordings(
        recording,
        profile=BluetoothAnalysisProfile.GENERAL_PACKET,
        phy="LE 1M",
        access_address=0x8E89BED6,
        channel_index=37,
        crc_init=0x555555,
        whitening_enabled=True,
        result_length=512,
        iq_power_trigger=IQPowerTriggerSettings(
            enabled=True,
            level_dbm=-100.0,
            hysteresis_db=3.0,
            dropout_symbols=8.0,
        ),
    )
    assert len(results) == 2
    assert all(result.packet.integrity.crc_valid is True for result in results)
