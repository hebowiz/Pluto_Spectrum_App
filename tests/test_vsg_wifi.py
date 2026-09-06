import os
from dataclasses import replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyqtgraph as pg

from pluto_vsg.engine.wifi_legacy_ofdm import (
    WiFiLegacyOFDMWaveformEngine, bcc_encode, interleave, map_constellation,
)
from pluto_vsg.model import StandardProfile, WiFiPSDUSource, WiFiSettings
from pluto_vsg.persistence import load_project, save_project
from pluto_vsg.profiles.wifi import wifi_fields, wifi_project
from pluto_vsg.ui.main_window import PlutoVSGWindow, _WiFiSettingsDialog
from pluto_vsg.wifi.mac import build_beacon_psdu


def test_wifi_beacon_contains_mac_fields_and_valid_fcs() -> None:
    import binascii
    settings = WiFiSettings()
    psdu = build_beacon_psdu(settings)
    assert psdu[:2] == b"\x80\x00"
    assert b"Pluto_Test_AP" in psdu
    assert b"\x05\x04\x00\x01\x00\x00" in psdu
    assert b"\x2a\x01\x00" in psdu
    assert psdu[-4:] == binascii.crc32(psdu[:-4]).to_bytes(4, "little")


def test_wifi_all_legacy_rates_generate_expected_ppdu_shape() -> None:
    for legacy_rate in (6, 9, 12, 18, 24, 36, 48, 54):
        settings = WiFiSettings(psdu_source=WiFiPSDUSource.RAW_HEX, raw_psdu_hex="001122334455", legacy_rate_mbps=legacy_rate, oversample_factor=1, packet_period_us=1000)
        project = wifi_project(settings)
        result = WiFiLegacyOFDMWaveformEngine().generate(project)
        assert result.sample_rate_hz == 20_000_000
        assert result.metadata["packet_sample_count"] == 400 + 80 * result.metadata["n_sym"]
        assert [boundary.name for boundary in result.field_boundaries] == ["L-STF", "L-LTF", "L-SIG", "DATA"]
        assert np.max(np.abs(result.iq)) <= 1.000001
        assert result.metadata["active_sample_count"] == result.metadata["packet_sample_count"]
        assert result.metadata["active_rms_dbfs"] < result.metadata["iq_peak_dbfs"]
        assert result.metadata["crest_factor_db"] > 0.0


def test_wifi_interleaver_and_constellations() -> None:
    values = np.arange(48, dtype=np.uint8) & 1
    assert sorted(interleave(values, 48, 1).tolist()) == sorted(values.tolist())
    np.testing.assert_allclose(map_constellation(np.array([0, 1], dtype=np.uint8), 1), [-1, 1])
    qpsk = map_constellation(np.array([0, 0, 1, 1], dtype=np.uint8), 2)
    np.testing.assert_allclose(qpsk, [(-1 - 1j) / np.sqrt(2), (1 + 1j) / np.sqrt(2)])


def test_wifi_bcc_matches_lsb_newest_reference_vector() -> None:
    bits = np.asarray([1, 0, 1, 1, 0, 0, 1], dtype=np.uint8)
    expected = np.asarray([1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1], dtype=np.uint8)
    np.testing.assert_array_equal(bcc_encode(bits), expected)


def test_wifi_project_json_round_trip(tmp_path) -> None:
    expected = wifi_project()
    path = tmp_path / "wifi.pvsg.json"
    save_project(path, expected)
    assert load_project(path) == expected


def test_wifi_dedicated_ui_and_main_window_dispatch() -> None:
    app = pg.mkQApp("Pluto VSG Wi-Fi UI test")
    project = wifi_project(replace(WiFiSettings(), packet_period_us=1000))
    parent = PlutoVSGWindow()
    dialog = _WiFiSettingsDialog(project, parent)
    try:
        assert [dialog.tabs.tabText(index) for index in range(dialog.tabs.count())] == [
            "RF / Timing",
            "Fields",
        ]
        assert dialog.rate_combo.count() == 8
        assert dialog.source_combo.findData(WiFiPSDUSource.BEACON) >= 0
        dialog.channel_combo.setCurrentIndex(
            dialog.channel_combo.findData(2_412_000_000.0)
        )
        dialog.frequency_offset_spin.setValue(-250.0)
        dialog._accept_settings()
        assert dialog.project.center_frequency_hz == 2_411_750_000.0
        parent.project = project
        parent._refresh_project_view()
        parent.generate_waveform()
        assert parent.result is not None
        assert parent.result.metadata["phy_format"] == "Non-HT OFDM"
        assert parent.settings_action.text().startswith("Wi-Fi")
        parent._pluto_bandwidth_hz = 8_000_000.0
        pluto_settings = parent._current_pluto_settings()
        assert pluto_settings.rf_bandwidth_hz == 20_000_000.0
        assert pluto_settings.waveform_active_rms_dbfs == parent.result.metadata["active_rms_dbfs"]
    finally:
        dialog.close(); parent.close(); app.processEvents()
