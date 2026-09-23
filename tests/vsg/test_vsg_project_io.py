import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace
from pathlib import Path
import pyqtgraph as pg
import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtWidgets
from pluto_vsa.iqtar import load_iq_tar
from pluto_vsg.engine import BluetoothBRWaveformEngine, GenerationResult
from pluto_vsg.export import save_iq_tar, save_npz, save_wv
from pluto_vsg.persistence import (
    load_project,
    project_from_dict,
    project_to_dict,
    save_project,
)
from pluto_vsg.profiles import bluetooth_br_edr_project, bluetooth_hdt_project
from pluto_vsg.ui.main_window import PlutoVSGWindow


def test_vsg_restores_last_project_controls_and_window_state(tmp_path) -> None:
    pg.mkQApp("Pluto VSG startup state persistence")
    settings_path = tmp_path / "vsg-settings.ini"
    preferences = QtCore.QSettings(
        str(settings_path), QtCore.QSettings.Format.IniFormat
    )
    project = replace(
        bluetooth_hdt_project(),
        name="Restored HDT project",
        repeat_count=10_000,
        period_symbols=2_500.0,
    )
    first = PlutoVSGWindow(
        project,
        preferences=preferences,
        restore_startup_state=True,
    )
    try:
        first.project_path = tmp_path / "restored.pvsg.json"
        first._modulation_enabled = False
        first._continuous_enabled = False
        first._field_display_mode = "off"
        first._pluto_uri = "usb:persisted"
        first._pluto_lead_in_guard_s = 0.023
        first._power_step_db = 2.5
        first.resize(1234, 777)
    finally:
        first.close()

    restored = PlutoVSGWindow(
        preferences=preferences,
        restore_startup_state=True,
    )
    try:
        assert restored.project == project
        assert restored.project_path == tmp_path / "restored.pvsg.json"
        assert restored._modulation_enabled is False
        assert restored._continuous_enabled is False
        assert restored._field_display_mode == "all"
        assert restored._pluto_uri == "usb:persisted"
        assert restored._pluto_lead_in_guard_s == pytest.approx(0.023)
        assert restored._power_step_db == pytest.approx(2.5)
        assert restored._rf_enabled is False
        assert preferences.contains("startup/geometry")
        assert not preferences.contains("startup/window_state")
    finally:
        restored.close()


def test_vsg_save_always_prompts_for_a_filename(tmp_path, monkeypatch) -> None:
    pg.mkQApp("Pluto VSG named save")
    window = PlutoVSGWindow()
    previous = tmp_path / "previous.pvsg.json"
    selected = tmp_path / "selected.pvsg.json"
    window.project_path = previous
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (str(selected), ""),
    )
    try:
        window.project_save_button.click()
        assert window.project_path == selected
        assert selected.exists()
        assert not previous.exists()
        assert load_project(selected) == window.project
    finally:
        window.close()


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
