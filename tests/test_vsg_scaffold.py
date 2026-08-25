import os
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pyqtgraph as pg
import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets

from pluto_sa.vsa.iqtar import load_iq_tar
from pluto_sa.vsa.profiles.bluetooth_br import (
    BluetoothBRProfile,
    access_code_bits,
    decode_dh1_payload,
)
from pluto_sa.vsa.profiles.bluetooth_edr import generate_edr_dh1
from pluto_sa.vsa.mapping import (
    BLUETOOTH_EDR_MAPPING,
    phase_indices_to_logical_symbols,
    reverse_symbol_bits,
)
from pluto_sa.vsa.model import IQRecording, ModulationKind, SignalDescription
from pluto_sa.vsa.pattern import (
    KnownPattern,
    PatternAnalyzer,
    PatternSearchMode,
    PatternSearchSettings,
    ResultRangeSettings,
)
from pluto_sa.vsa.ui.measurement_chrome import FixedInteractionViewBox
from pluto_vsg.engine import BluetoothBRWaveformEngine, GenerationResult
from pluto_vsg.export import save_iq_tar, save_npz, save_wv
from pluto_vsg.model import create_default_project, validate_project
from pluto_vsg.model import (
    BluetoothLEPayloadSourceKind,
    BluetoothLEPhy,
    BluetoothPacketKind,
    PayloadSourceKind,
    bluetooth_packet_is_edr,
    bluetooth_packet_properties,
)
from pluto_vsg.persistence import (
    load_project,
    project_from_dict,
    project_to_dict,
    save_project,
)
from pluto_vsg.profiles import (
    bluetooth_br_edr_project,
    bluetooth_br_fields,
    bluetooth_le_project,
)
from pluto_vsg.ui.main_window import (
    PlutoVSGWindow,
    _BluetoothLESettingsDialog,
    _BluetoothSettingsDialog,
    _instantaneous_frequency_khz,
)


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
            assert any(line.label.format == "Packet End" for line in lines)
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


def test_classic_rf_test_preset_populates_existing_payload_controls() -> None:
    pg.mkQApp("Pluto VSG Classic RF preset test")
    parent = PlutoVSGWindow()
    dialog = _BluetoothSettingsDialog(parent.project, parent)
    try:
        dialog.rf_test_payload_combo.setCurrentIndex(
            dialog.rf_test_payload_combo.findData("11110000")
        )
        dialog._apply_rf_test_preset()

        assert dialog.payload_source_combo.currentData() == PayloadSourceKind.PATTERN
        assert dialog.pattern_edit.text() == "11110000"
        assert dialog.whitening_check.isChecked() is False
    finally:
        dialog.close()
        parent.close()


def test_le_rf_test_preset_populates_editable_packet_controls() -> None:
    pg.mkQApp("Pluto VSG LE RF preset test")
    project = bluetooth_le_project(BluetoothLEPhy.LE_2M)
    parent = PlutoVSGWindow(project)
    dialog = _BluetoothLESettingsDialog(project, parent)
    try:
        dialog._apply_rf_test_preset()

        assert dialog.sync_edit.text() == "10010100100000100110111010001110"
        assert dialog.payload_source_combo.currentData() == (
            BluetoothLEPayloadSourceKind.PATTERN
        )
        assert dialog.payload_pattern_edit.text() == "10101010"
        assert dialog.crc_init_edit.text() == "555555"
        assert dialog.whitening_check.isChecked() is False
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
    base = bluetooth_br_edr_project()
    assert base.bluetooth_br is not None
    settings = replace(base.bluetooth_br, whitening_enabled=True)
    project = replace(
        base,
        bluetooth_br=settings,
        fields=bluetooth_br_fields(settings),
    )

    result = BluetoothBRWaveformEngine().generate(project)
    packet_start, packet_stop = result.metadata["packet_ranges_samples"][0]
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


@pytest.mark.parametrize(
    ("packet_kind", "payload_length"),
    (
        (BluetoothPacketKind.DH1_2, 54),
        (BluetoothPacketKind.DH1_3, 83),
    ),
)
def test_vsg_edr_generation_matches_validated_phase_sequence(
    packet_kind, payload_length
) -> None:
    base = bluetooth_br_edr_project()
    assert base.bluetooth_br is not None
    settings = replace(
        base.bluetooth_br,
        packet_kind=packet_kind,
        payload_length_bytes=payload_length,
        whitening_enabled=True,
    )
    project = replace(
        base,
        bluetooth_br=settings,
        fields=bluetooth_br_fields(settings),
    )

    result = BluetoothBRWaveformEngine().generate(project)
    reference = generate_edr_dh1(
        packet_kind.value,
        sample_rate_hz=project.sample_rate_hz,
        carrier_frequency_offset_hz=0.0,
        duration_s=0.001,
        packet_start_s=0.0001,
        snr_db=200.0,
    )

    assert result.metadata["packet_name"] == packet_kind.value
    np.testing.assert_array_equal(
        result.metadata["edr_phase_indices"],
        reference.differential_phase_indices,
    )
    assert result.metadata["edr_start_sample"] - result.metadata["gfsk_stop_sample"] == 5 * project.samples_per_symbol
    assert np.max(np.abs(result.iq)) <= 1.0 + 1e-6
    assert [field.name for field in project.fields] == [
        "Access Code",
        "Header",
        "Guard",
        "EDR Data",
    ]

    modulation = (
        ModulationKind.PI4_DQPSK
        if packet_kind == BluetoothPacketKind.DH1_2
        else ModulationKind.DPSK8
    )
    expected_symbols = phase_indices_to_logical_symbols(
        modulation,
        BLUETOOTH_EDR_MAPPING,
        np.asarray(result.metadata["edr_phase_indices"]),
    )
    pattern = reverse_symbol_bits(expected_symbols[:10], modulation.order)
    analyzed = PatternAnalyzer().search(
        IQRecording(
            iq=result.iq,
            sample_rate_hz=result.sample_rate_hz,
            center_frequency_hz=project.center_frequency_hz,
        ),
        SignalDescription(
            modulation=modulation,
            symbol_rate_hz=1_000_000.0,
            tx_filter="Root Raised Cosine",
            filter_parameter=settings.edr_rolloff,
            symbol_mapping="Bluetooth EDR",
        ),
        PatternSearchSettings(
            pattern=KnownPattern(tuple(map(int, pattern))),
            mode=PatternSearchMode.ON,
            correlation_threshold_auto=False,
            iq_correlation_threshold=0.9,
        ),
        ResultRangeSettings(result_length=244),
    )
    assert analyzed.pattern_symbol_errors == 0
    assert analyzed.correlation > 0.99
    np.testing.assert_array_equal(analyzed.decoded_symbols, expected_symbols)
    assert analyzed.evm_rms_percent < 5.0


def test_vsg_packet_type_dialog_updates_edr_project() -> None:
    pg.mkQApp("Pluto VSG EDR settings test")
    parent = PlutoVSGWindow()
    dialog = _BluetoothSettingsDialog(parent.project, parent)
    try:
        dialog.packet_type_combo.setCurrentIndex(
            dialog.packet_type_combo.findData(BluetoothPacketKind.DH1_3)
        )
        dialog.payload_length_spin.setValue(83)
        dialog._accept_settings()

        settings = dialog.project.bluetooth_br
        assert settings is not None
        assert settings.packet_kind == BluetoothPacketKind.DH1_3
        assert dialog.project.fields[-1].name == "EDR Data"
        assert dialog.project.fields[-1].modulation.kind.value == "8DPSK"
    finally:
        dialog.close()
        parent.close()


def test_vsg_dialog_preserves_maximum_edr_payload_when_reopened() -> None:
    pg.mkQApp("Pluto VSG payload persistence test")
    parent = PlutoVSGWindow()
    first = _BluetoothSettingsDialog(parent.project, parent)
    second = None
    try:
        first.packet_type_combo.setCurrentIndex(
            first.packet_type_combo.findData(BluetoothPacketKind.DH5_3)
        )
        first.payload_length_spin.setValue(1021)
        first._accept_settings()

        second = _BluetoothSettingsDialog(first.project, parent)

        assert second.payload_length_spin.maximum() == 1021
        assert second.payload_length_spin.value() == 1021
    finally:
        if second is not None:
            second.close()
        first.close()
        parent.close()


@pytest.mark.parametrize(
    ("packet_kind", "payload_max"),
    (
        (BluetoothPacketKind.DH1, 27),
        (BluetoothPacketKind.DH3, 183),
        (BluetoothPacketKind.DH5, 339),
        (BluetoothPacketKind.DH1_2, 54),
        (BluetoothPacketKind.DH3_2, 367),
        (BluetoothPacketKind.DH5_2, 679),
        (BluetoothPacketKind.DH1_3, 83),
        (BluetoothPacketKind.DH3_3, 552),
        (BluetoothPacketKind.DH5_3, 1021),
    ),
)
def test_vsg_dialog_packet_type_change_selects_maximum_payload(
    packet_kind, payload_max
) -> None:
    parent = QtWidgets.QWidget()
    dialog = _BluetoothSettingsDialog(bluetooth_br_edr_project(), parent)
    try:
        # Move away first so the DH1 case also emits currentIndexChanged.
        dialog.packet_type_combo.setCurrentIndex(
            dialog.packet_type_combo.findData(BluetoothPacketKind.DH5_3)
        )
        dialog.packet_type_combo.setCurrentIndex(
            dialog.packet_type_combo.findData(packet_kind)
        )
        assert dialog.payload_length_spin.maximum() == payload_max
        assert dialog.payload_length_spin.value() == payload_max
    finally:
        dialog.close()
        parent.close()


@pytest.mark.parametrize(
    ("packet_kind", "payload_max", "packet_type", "bits_per_symbol", "slots"),
    (
        (BluetoothPacketKind.DH1, 27, 0x4, 1, 1),
        (BluetoothPacketKind.DH3, 183, 0xB, 1, 3),
        (BluetoothPacketKind.DH5, 339, 0xF, 1, 5),
        (BluetoothPacketKind.DH1_2, 54, 0x4, 2, 1),
        (BluetoothPacketKind.DH3_2, 367, 0xA, 2, 3),
        (BluetoothPacketKind.DH5_2, 679, 0xE, 2, 5),
        (BluetoothPacketKind.DH1_3, 83, 0x8, 3, 1),
        (BluetoothPacketKind.DH3_3, 552, 0xB, 3, 3),
        (BluetoothPacketKind.DH5_3, 1021, 0xF, 3, 5),
    ),
)
def test_vsg_all_dhx_packet_definitions_generate(
    packet_kind, payload_max, packet_type, bits_per_symbol, slots
) -> None:
    base = bluetooth_br_edr_project()
    assert base.bluetooth_br is not None
    settings = replace(
        base.bluetooth_br,
        packet_kind=packet_kind,
        payload_length_bytes=payload_max,
    )
    project = replace(
        base,
        bluetooth_br=settings,
        fields=bluetooth_br_fields(settings),
    )

    assert bluetooth_packet_properties(packet_kind) == (
        payload_max,
        packet_type,
        bits_per_symbol,
        slots,
    )
    assert bluetooth_packet_is_edr(packet_kind) is (bits_per_symbol > 1)
    assert validate_project(project) == ()

    result = BluetoothBRWaveformEngine().generate(project)
    payload_header = np.asarray(result.metadata["payload_header_bits"])
    assert payload_header.size == (
        8 if packet_kind == BluetoothPacketKind.DH1 else 16
    )
    packed_payload_header = sum(
        int(bit) << index for index, bit in enumerate(payload_header)
    )
    assert (packed_payload_header >> 3) & (
        0x1F if packet_kind == BluetoothPacketKind.DH1 else 0x3FF
    ) == payload_max
    assert result.metadata["payload_body_bits"].size == payload_max * 8
    assert result.metadata["packet_name"] == packet_kind.value
    assert result.metadata["packet_sample_count"] > 0
    header_air = np.asarray(result.metadata["packet_bits"])[72:126]
    header_data = header_air.reshape(-1, 3)[:, 0]
    packed_header = sum(
        int(bit) << index for index, bit in enumerate(header_data[:10])
    )
    assert (packed_header >> 3) & 0xF == packet_type
    packet_duration_us = (
        result.metadata["packet_sample_count"] / result.sample_rate_hz * 1e6
    )
    assert packet_duration_us <= slots * 625.0


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
    packet_ranges = result.metadata["packet_ranges_samples"]
    expected_start = (
        project.bluetooth_br.pre_idle_symbols * project.samples_per_symbol
        - min(
            0,
            round(
                project.power_envelope.rise_delay_symbols
                * project.samples_per_symbol
            ),
        )
    )
    assert packet_ranges == ((
        expected_start,
        expected_start + result.metadata["packet_sample_count"],
    ),)


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


def test_vsg_wv_export_has_smu_header_alignment_and_iq(tmp_path: Path) -> None:
    project = bluetooth_br_edr_project()
    result = BluetoothBRWaveformEngine().generate(project)
    path = tmp_path / "dh1.wv"

    save_wv(path, result, project)

    document = path.read_bytes()
    waveform_offset = document.index(b"{WAVEFORM-")
    assert waveform_offset == 0x4000
    assert document.startswith(b"{TYPE: SMU-WV,")
    assert f"{{SAMPLES: {result.iq.size}}}".encode() in document[:waveform_offset]
    assert (
        f"{{CLOCK: {result.sample_rate_hz:.12g}}}".encode()
        in document[:waveform_offset]
    )
    marker = f"{{WAVEFORM-{result.iq.size * 4 + 1}:#".encode()
    data_start = waveform_offset + len(marker)
    assert document[waveform_offset:data_start] == marker
    raw = np.frombuffer(
        document[data_start : data_start + result.iq.size * 4], dtype="<i2"
    )
    restored = (
        raw[0::2].astype(np.float64) + 1j * raw[1::2].astype(np.float64)
    ) / 32767.0
    np.testing.assert_allclose(restored, result.iq, rtol=0.0, atol=1.6 / 32767.0)
    assert document[-1:] == b"}"


def test_vsg_wv_export_checksum_and_level_offsets(tmp_path: Path) -> None:
    project = bluetooth_br_edr_project()
    result = BluetoothBRWaveformEngine().generate(project)
    path = tmp_path / "dh1.wv"
    save_wv(path, result, project)
    document = path.read_bytes()

    type_value = document[len(b"{TYPE: SMU-WV,") : document.index(b"}")]
    expected_checksum = 0xA50F74FF
    waveform_offset = document.index(b"{WAVEFORM-")
    data_start = document.index(b"#", waveform_offset) + 1
    binary = document[data_start:-1]
    for offset in range(0, len(binary), 4):
        expected_checksum ^= int.from_bytes(binary[offset : offset + 4], "little")
    assert int(type_value) == expected_checksum

    level_start = document.index(b"{LEVEL OFFS: ") + len(b"{LEVEL OFFS: ")
    rms_text, peak_text = document[level_start : document.index(b"}", level_start)].split(
        b","
    )
    raw = np.frombuffer(binary, dtype="<i2")
    iq = (raw[0::2] + 1j * raw[1::2]) / 32767.0
    assert float(rms_text) == pytest.approx(-20.0 * np.log10(np.sqrt(np.mean(abs(iq) ** 2))))
    assert float(peak_text) == pytest.approx(-20.0 * np.log10(np.max(abs(iq))))


def test_vsg_wv_export_rejects_non_normalized_iq(tmp_path: Path) -> None:
    project = bluetooth_br_edr_project()
    result = GenerationResult(
        iq=np.asarray([1.1 + 0.0j], dtype=np.complex64),
        sample_rate_hz=project.sample_rate_hz,
    )

    with pytest.raises(ValueError, match="normalized IQ"):
        save_wv(tmp_path / "invalid.wv", result, project)
