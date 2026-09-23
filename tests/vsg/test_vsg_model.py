import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace
import pyqtgraph as pg
import numpy as np
from pyqtgraph.Qt import QtWidgets
from pluto_vsg.engine import BluetoothBRWaveformEngine
from pluto_vsg.model import create_default_project, validate_project
from pluto_vsg.profiles import (
    bluetooth_br_edr_project,
    bluetooth_hdt_project,
    bluetooth_le_project,
)
from pluto_vsg.ui.main_window import (
    _BluetoothHDTSettingsDialog,
    _BluetoothLESettingsDialog,
    _BluetoothSettingsDialog,
    _cw_generation_result,
)


def test_le_carrier_list_and_offset_define_generated_frequency() -> None:
    pg.mkQApp("Bluetooth LE VSG carrier offset")
    parent = QtWidgets.QWidget()
    dialog = _BluetoothLESettingsDialog(bluetooth_le_project(), parent)
    try:
        channel_37 = dialog.carrier_combo.findData(2_402_000_000.0)
        dialog.carrier_combo.setCurrentIndex(channel_37)
        dialog.frequency_offset_spin.setValue(125.0)
        dialog._accept_settings()
        assert dialog.project.center_frequency_hz == 2_402_125_000.0
        assert "2402.125000 MHz" in dialog.actual_frequency_label.text()
    finally:
        dialog.close()
        parent.close()


def test_hdt_period_replaces_direct_post_idle_and_preserves_ramp() -> None:
    pg.mkQApp("Bluetooth HDT VSG period settings")
    parent = QtWidgets.QWidget()
    dialog = _BluetoothHDTSettingsDialog(bluetooth_hdt_project(), parent)
    try:
        assert not hasattr(dialog, "post_idle_spin")
        dialog.period_spin.setValue(dialog.period_spin.minimum() + 40.0)
        dialog.rise_spin.setValue(2.5)
        dialog._accept_settings()
        assert dialog.project.period_symbols == dialog.period_spin.value()
        assert dialog.project.bluetooth_hdt.post_idle_symbols == 0
        assert dialog.project.power_envelope.rise_symbols == 2.5
        assert "us" in dialog.post_idle_value.text()
    finally:
        dialog.close()
        parent.close()


def test_classic_and_hdt_carrier_offsets_preserve_rf_semantics() -> None:
    pg.mkQApp("Bluetooth VSG carrier offset semantics")
    parent = QtWidgets.QWidget()
    classic = _BluetoothSettingsDialog(bluetooth_br_edr_project(), parent)
    hdt = _BluetoothHDTSettingsDialog(bluetooth_hdt_project(), parent)
    try:
        classic.carrier_combo.setCurrentIndex(
            classic.carrier_combo.findData(2_402_000_000.0)
        )
        classic.cfo_spin.setValue(25.0)
        classic._accept_settings()
        assert classic.project.center_frequency_hz == 2_402_000_000.0
        assert classic.project.bluetooth_br.carrier_frequency_offset_hz == 25_000.0

        hdt.carrier_combo.setCurrentIndex(
            hdt.carrier_combo.findData(2_402_000_000.0)
        )
        hdt.frequency_offset_spin.setValue(25.0)
        hdt._accept_settings()
        assert hdt.project.center_frequency_hz == 2_402_025_000.0
    finally:
        classic.close()
        hdt.close()
        parent.close()


def test_default_vsg_project_is_valid() -> None:
    project = create_default_project()

    assert project.samples_per_symbol == 8
    assert validate_project(project) == ()

    bluetooth_project = bluetooth_br_edr_project()
    assert bluetooth_project.bluetooth_br is not None
    assert bluetooth_project.center_frequency_hz == 2_440_000_000.0
    assert bluetooth_project.bluetooth_br.whitening_enabled is False
    assert bluetooth_project.power_envelope.rise_symbols == 1.0
    assert bluetooth_project.power_envelope.rise_delay_symbols == -1.0
    assert bluetooth_project.power_envelope.fall_symbols == 1.0
    assert bluetooth_project.power_envelope.fall_delay_symbols == 1.0
    assert bluetooth_project.power_envelope.shape == "Cosine"


def test_invalid_vsg_project_reports_model_path() -> None:
    project = replace(create_default_project(), sample_rate_hz=0.0)

    issues = validate_project(project)

    assert any(issue.path == "sample_rate_hz" for issue in issues)


def test_bluetooth_profile_expands_into_common_fields() -> None:
    project = bluetooth_br_edr_project()

    assert [packet_field.name for packet_field in project.fields] == [
        "Access Code",
        "Header",
        "Payload",
    ]
    assert [child.name for child in project.fields[0].children] == [
        "Preamble",
        "Sync Word",
        "Trailer",
    ]
    assert [child.name for child in project.fields[1].children] == [
        "LT_ADDR",
        "TYPE",
        "FLOW",
        "ARQN",
        "SEQN",
        "HEC",
    ]
    assert project.fields[1].logical_bit_count == 18
    assert project.fields[1].symbol_count == 54
    assert validate_project(project) == ()


def test_cw_generation_result_is_constant_normalized_zero_if() -> None:
    result = _cw_generation_result(8_000_000.0, sample_count=1024)

    assert result.sample_rate_hz == 8_000_000.0
    assert result.iq.dtype == np.complex64
    assert result.iq.size == 1024
    np.testing.assert_array_equal(result.iq, np.ones(1024, dtype=np.complex64))
    assert result.metadata["waveform_kind"] == "CW"
    assert result.metadata["baseband_frequency_hz"] == 0.0


def test_explicit_br_period_produces_equal_complete_repetitions() -> None:
    base = bluetooth_br_edr_project()
    project = replace(base, repeat_count=5, period_symbols=512.375)

    result = BluetoothBRWaveformEngine().generate(project)

    period_samples = round(512.375 * project.samples_per_symbol)
    assert result.metadata["period_sample_count"] == period_samples
    assert result.iq.size == 5 * period_samples
    starts = [start for start, _stop in result.metadata["packet_ranges_samples"]]
    assert np.diff(starts).tolist() == [period_samples] * 4
