import os
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pyqtgraph as pg
import numpy as np
from pyqtgraph.Qt import QtWidgets

from pluto_sa.vsa.iqtar import load_iq_tar
from pluto_sa.vsa.profiles.bluetooth_br import (
    BluetoothBRProfile,
    access_code_bits,
    decode_dh1_payload,
)
from pluto_sa.vsa.model import IQRecording
from pluto_sa.vsa.ui.measurement_chrome import FixedInteractionViewBox
from pluto_vsg.engine import BluetoothBRWaveformEngine
from pluto_vsg.export import save_iq_tar, save_npz
from pluto_vsg.model import create_default_project, validate_project
from pluto_vsg.model import PayloadSourceKind
from pluto_vsg.persistence import (
    load_project,
    project_from_dict,
    project_to_dict,
    save_project,
)
from pluto_vsg.profiles import bluetooth_br_edr_project, bluetooth_br_fields
from pluto_vsg.ui.main_window import (
    PlutoVSGWindow,
    _BluetoothSettingsDialog,
    _instantaneous_frequency_khz,
)


def test_default_vsg_project_is_valid() -> None:
    project = create_default_project()

    assert project.samples_per_symbol == 8
    assert validate_project(project) == ()


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


def test_vsg_window_starts_with_composer_shell() -> None:
    pg.mkQApp("Pluto VSG scaffold test")
    window = PlutoVSGWindow()
    try:
        assert "Pluto VSG" in window.windowTitle()
        assert window.field_table.topLevelItemCount() == 3
        assert window.field_table.topLevelItem(0).childCount() == 3
        assert window.result is not None
        iq_traces = window.iq_waveform_plot.listDataItems()
        assert [trace.name() for trace in iq_traces] == ["I", "Q"]
        assert [action.text() for action in window.menuBar().actions()] == [
            "File",
            "Edit",
            "Waveform",
            "Graphics",
            "Output",
            "Tools",
            "Help",
        ]
    finally:
        window.close()


def test_vsg_field_labels_are_present_and_keep_a_fixed_side() -> None:
    pg.mkQApp("Pluto VSG field label anchor test")
    window = PlutoVSGWindow()
    try:
        for plot in (
            window.iq_waveform_plot,
            window.power_plot,
            window.frequency_plot,
        ):
            lines = [
                item
                for item in plot.getPlotItem().items
                if isinstance(item, pg.InfiniteLine)
            ]
            assert lines
            assert all(line.label is not None for line in lines)
            assert all(
                line.label.anchors == [(0.0, 0.5), (0.0, 0.5)]
                for line in lines
            )
    finally:
        window.close()


def test_vsg_plots_use_fixed_vsa_interaction_and_reset() -> None:
    pg.mkQApp("Pluto VSG VSA plot interaction test")
    window = PlutoVSGWindow()
    try:
        for name, plot in window._plot_widgets():
            view_box = plot.getViewBox()
            assert isinstance(view_box, FixedInteractionViewBox)
            assert view_box.state["mouseMode"] == pg.ViewBox.RectMode
            menu = view_box.getMenu(None)
            assert menu is not None
            assert "Mouse Mode" not in [action.text() for action in menu.actions()]
            assert name in window._plot_context_actions

        power_initial = window._plot_initial_ranges["power"]
        window.power_plot.setRange(
            xRange=[0.0, 1.0], yRange=[-10.0, 10.0], padding=0.0
        )
        window._plot_context_actions["power"]["reset"].trigger()
        x_range, y_range = window.power_plot.viewRange()
        np.testing.assert_allclose(x_range, power_initial[0])
        np.testing.assert_allclose(y_range, power_initial[1])
    finally:
        window.close()


def test_vsg_payload_source_labels_explain_generation_behavior() -> None:
    pg.mkQApp("Pluto VSG payload source label test")
    parent = PlutoVSGWindow()
    dialog = _BluetoothSettingsDialog(parent.project, parent)
    try:
        labels = [
            dialog.payload_source_combo.itemText(index)
            for index in range(dialog.payload_source_combo.count())
        ]
        assert labels == [
            "Constant (All 0 / All 1)",
            "Repeating Bit Pattern",
            "PRBS-9",
        ]
        assert "PRBS-9 sequence" in dialog.payload_source_help.text()
        ok_button = dialog.findChild(QtWidgets.QDialogButtonBox).button(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
        )
        assert ok_button.text() == "Apply and Generate"
    finally:
        dialog.close()
        parent.close()


def test_vsg_settings_keep_composer_payload_source_in_sync() -> None:
    pg.mkQApp("Pluto VSG composer source sync test")
    parent = PlutoVSGWindow()
    dialog = _BluetoothSettingsDialog(parent.project, parent)
    try:
        dialog.payload_source_combo.setCurrentIndex(
            dialog.payload_source_combo.findData(PayloadSourceKind.PATTERN)
        )
        dialog.pattern_edit.setText("1100")
        dialog._accept_settings()

        payload_field = next(
            packet_field
            for packet_field in dialog.project.fields
            if packet_field.name == "Payload"
        )
        assert payload_field.data_source.value == "Pattern"
        assert payload_field.data == "1100"
    finally:
        dialog.close()
        parent.close()


def test_vsg_settings_edit_packet_header_and_recalculate_hec() -> None:
    pg.mkQApp("Pluto VSG packet header settings test")
    parent = PlutoVSGWindow()
    dialog = _BluetoothSettingsDialog(parent.project, parent)
    try:
        initial_hec = dialog.hec_value.text()
        dialog.lt_addr_spin.setValue(5)
        dialog.flow_combo.setCurrentIndex(dialog.flow_combo.findData(0))
        dialog.arqn_combo.setCurrentIndex(dialog.arqn_combo.findData(1))
        dialog.seqn_combo.setCurrentIndex(dialog.seqn_combo.findData(1))

        assert dialog.hec_value.text().endswith("(auto)")
        assert dialog.hec_value.text() != initial_hec

        dialog._accept_settings()
        settings = dialog.project.bluetooth_br
        assert settings is not None
        assert settings.lt_addr == 5
        assert settings.flow == 0
        assert settings.arqn == 1
        assert settings.seqn == 1
    finally:
        dialog.close()
        parent.close()


def test_vsg_frequency_preview_does_not_connect_burst_to_zero_hz() -> None:
    phase_step = 2.0 * np.pi * 100_000.0 / 8_000_000.0
    active = np.exp(1j * phase_step * np.arange(8))
    iq = np.concatenate((np.zeros(2), active, np.zeros(2)))

    frequency = _instantaneous_frequency_khz(iq, 8_000_000.0)

    assert np.isnan(frequency[0:2]).all()
    np.testing.assert_allclose(frequency[2:9], 100.0)
    assert np.isnan(frequency[9:]).all()


def test_vsg_dh1_generation_decodes_with_valid_hec_and_crc() -> None:
    project = bluetooth_br_edr_project()

    result = BluetoothBRWaveformEngine().generate(project)
    settings = project.bluetooth_br
    assert settings is not None
    packet_start = settings.pre_idle_symbols * project.samples_per_symbol
    packet_stop = packet_start + int(result.metadata["packet_sample_count"])
    recording = IQRecording(
        iq=result.iq[packet_start:packet_stop],
        sample_rate_hz=result.sample_rate_hz,
        center_frequency_hz=project.center_frequency_hz,
    )
    analyzed = BluetoothBRProfile(access_code_bits(settings.lap)).analyze(
        recording,
        clock_6_1=settings.clock_6_1,
        uap=settings.uap,
    )

    assert analyzed.header is not None
    assert analyzed.header.hec_valid is True
    assert analyzed.header.packet_type == 4
    payload = decode_dh1_payload(analyzed.payload_bits, uap=settings.uap)
    assert payload.length_bytes == 27
    assert payload.crc_valid is True


def test_vsg_generation_emits_hierarchical_sample_boundaries() -> None:
    project = bluetooth_br_edr_project()

    result = BluetoothBRWaveformEngine().generate(project)
    boundaries = result.field_boundaries
    access = next(item for item in boundaries if item.name == "Access Code")
    sync_word = next(item for item in boundaries if item.name == "Sync Word")
    header = next(item for item in boundaries if item.name == "Header")
    header_type = next(item for item in boundaries if item.name == "TYPE")
    payload_body = next(item for item in boundaries if item.name == "Payload Body")

    assert access.level == 0
    assert access.start_symbol == 0
    assert access.stop_symbol == 72
    assert sync_word.level == 1
    assert sync_word.parent_name == "Access Code"
    assert (sync_word.start_symbol, sync_word.stop_symbol) == (4, 68)
    assert header.logical_bit_count == 18
    assert header.stop_symbol - header.start_symbol == 54
    assert header_type.logical_bit_count == 4
    assert header_type.stop_symbol - header_type.start_symbol == 12
    assert payload_body.logical_bit_count == 27 * 8
    assert payload_body.stop_sample - payload_body.start_sample == 27 * 8 * 8


def test_vsg_payload_sources_generate_distinct_expected_bits() -> None:
    base = bluetooth_br_edr_project()
    assert base.bluetooth_br is not None
    engine = BluetoothBRWaveformEngine()
    fixed_settings = replace(
        base.bluetooth_br,
        payload_length_bytes=2,
        payload_source=PayloadSourceKind.FIXED,
        payload_pattern="0",
    )
    pattern_settings = replace(
        base.bluetooth_br,
        payload_length_bytes=2,
        payload_source=PayloadSourceKind.PATTERN,
        payload_pattern="10",
    )
    prbs_settings = replace(
        base.bluetooth_br,
        payload_length_bytes=2,
        payload_source=PayloadSourceKind.PRBS9,
        payload_pattern="0",
    )
    fixed = engine.generate(
        replace(
            base,
            bluetooth_br=fixed_settings,
            fields=bluetooth_br_fields(fixed_settings),
        )
    )
    pattern = engine.generate(
        replace(
            base,
            bluetooth_br=pattern_settings,
            fields=bluetooth_br_fields(pattern_settings),
        )
    )
    prbs = engine.generate(
        replace(
            base,
            bluetooth_br=prbs_settings,
            fields=bluetooth_br_fields(prbs_settings),
        )
    )

    np.testing.assert_array_equal(
        fixed.metadata["payload_body_bits"], np.zeros(16, dtype=np.uint8)
    )
    np.testing.assert_array_equal(
        pattern.metadata["payload_body_bits"], np.tile([1, 0], 8)
    )
    assert not np.array_equal(
        prbs.metadata["payload_body_bits"],
        fixed.metadata["payload_body_bits"],
    )
    assert not np.array_equal(
        prbs.metadata["payload_body_bits"],
        pattern.metadata["payload_body_bits"],
    )


def test_vsg_engine_coerces_ui_string_payload_source() -> None:
    base = bluetooth_br_edr_project()
    assert base.bluetooth_br is not None
    settings = replace(
        base.bluetooth_br,
        payload_length_bytes=2,
        payload_source=PayloadSourceKind.PRBS9.value,
        payload_pattern="0",
    )
    project = replace(
        base,
        bluetooth_br=settings,
        fields=bluetooth_br_fields(settings),
    )

    result = BluetoothBRWaveformEngine().generate(project)

    assert np.count_nonzero(result.metadata["payload_body_bits"]) > 0


def test_vsg_delayed_ramp_down_holds_last_symbol_frequency() -> None:
    base = bluetooth_br_edr_project()
    project = replace(
        base,
        power_envelope=replace(
            base.power_envelope,
            fall_delay_symbols=2.0,
            fall_symbols=1.0,
        ),
    )

    result = BluetoothBRWaveformEngine().generate(project)
    data_stop = int(result.metadata["data_stop_sample"])
    hold_samples = 2 * project.samples_per_symbol
    frequency = _instantaneous_frequency_khz(result.iq, result.sample_rate_hz)
    packet_bits = np.asarray(result.metadata["packet_bits"])
    expected_khz = (
        (2.0 * float(packet_bits[-1]) - 1.0)
        * project.bluetooth_br.frequency_deviation_hz
        + project.bluetooth_br.carrier_frequency_offset_hz
    ) / 1e3

    np.testing.assert_allclose(
        frequency[data_stop : data_stop + hold_samples - 1],
        expected_khz,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        np.abs(result.iq[data_stop : data_stop + hold_samples]),
        1.0,
        atol=1e-6,
    )
    assert result.metadata["edge_frequency_mode"] == "Hold first / last symbol"


def test_vsg_project_json_round_trip(tmp_path: Path) -> None:
    expected = bluetooth_br_edr_project()
    path = tmp_path / "dh1.pvsg.json"

    save_project(path, expected)
    actual = load_project(path)

    assert actual == expected


def test_vsg_legacy_flat_bluetooth_project_is_upgraded() -> None:
    document = project_to_dict(bluetooth_br_edr_project())
    project_payload = document["project"]
    assert isinstance(project_payload, dict)
    fields = project_payload["fields"]
    assert isinstance(fields, list)
    for packet_field in fields:
        packet_field.pop("children", None)
        packet_field.pop("logical_bit_count", None)

    actual = project_from_dict(document)

    assert actual.fields[0].children[1].name == "Sync Word"
    assert actual.fields[1].logical_bit_count == 18


def test_vsg_npz_and_iqtar_exports_are_readable(tmp_path: Path) -> None:
    project = bluetooth_br_edr_project()
    result = BluetoothBRWaveformEngine().generate(project)
    npz_path = tmp_path / "dh1.npz"
    iqtar_path = tmp_path / "dh1.iq.tar"

    save_npz(npz_path, result, project)
    save_iq_tar(iqtar_path, result, project)

    with np.load(npz_path, allow_pickle=False) as document:
        np.testing.assert_array_equal(document["iq"], result.iq)
        assert float(document["sample_rate_hz"]) == result.sample_rate_hz
    iqtar = load_iq_tar(iqtar_path)
    np.testing.assert_allclose(iqtar.iq, result.iq, rtol=0.0, atol=1e-7)
    assert iqtar.center_frequency_hz == project.center_frequency_hz
