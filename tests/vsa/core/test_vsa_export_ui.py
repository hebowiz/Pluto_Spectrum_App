import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
from dataclasses import replace
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
from pluto_vsa.ui.main_window import VSAWindow
from pluto_vsa.model import ModulationKind
from pluto_vsa.pattern import DemodulationSettings, SynchronizationSource
from pluto_vsa.session import VSASession
from pluto_vsa.sources import FileIQSource, GeneratedIQSource
from _vsa_ui_test_helpers import _isolated_preferences, _wait_for_background_analysis


def test_large_symbol_result_limits_table_display_but_not_export(tmp_path) -> None:
    pg.mkQApp("VSA bounded symbol table test")
    recording, signal = GeneratedIQSource.psk(
        modulation=ModulationKind.PI4_DQPSK,
        symbol_count=1500,
        seed=91,
    )
    session = VSASession()
    session.set_recording(recording)
    session.set_signal(signal)
    # This test exercises bounded rendering of the unsynchronized/base result,
    # rather than the detected-data Result Range.
    session.configure_pattern_analysis(
        None,
        demodulation=DemodulationSettings(
            coarse_synchronization=SynchronizationSource.PATTERN,
        ),
    )
    session.analyze()
    window = VSAWindow(
        preferences=_isolated_preferences(tmp_path, "bounded-symbol-table")
    )
    try:
        window.session = session
        window._update_summary()
        window._update_plots(reset_ranges=True)

        assert window.symbol_plot_dock.windowTitle() == "Symbol Plot (Physical)"
        window.differential_iq_symbol_plot_action.trigger()
        assert window.symbol_plot_dock.windowTitle() == "Symbol Plot (Differential)"
        assert window.symbol_table.rowCount() == 100
        result_symbol_count = session.result.decoded_symbols.size
        assert f"Showing 1000 of {result_symbol_count}" in window.symbol_table.toolTip()
        assert (
            len(window._symbol_table_export_document()["rows"])
            == result_symbol_count
        )
    finally:
        window.close()


def test_symbol_table_json_export_document_contains_machine_readable_context(
    tmp_path, monkeypatch,
) -> None:
    pg.mkQApp("VSA Symbol Table export test")
    window = VSAWindow(
        preferences=_isolated_preferences(tmp_path, "symbol-export")
    )
    try:
        window._load_generated(ModulationKind.GFSK)
        _wait_for_background_analysis(window)
        expected = np.asarray(window.session.recording.metadata["generated_symbols"])
        window.pattern_search_check.setChecked(True)
        window._set_pattern_symbols(expected[20:52])
        window.result_length_spin.setValue(64)
        assert window._analyze()

        document = window._symbol_table_export_document()

        assert document["schema"] == "pluto-vsa-symbol-table"
        assert document["version"] == 1
        assert document["metadata"]["modulation"] == "FSK"
        assert document["metadata"]["symbol_mapping"] == "Natural"
        assert document["metadata"]["pattern"]["match_variant"] == "Normal"
        assert document["columns"] == [
            "index",
            "symbol",
            "bits",
            "time_s",
            "pattern_index",
            "pattern_status",
        ]
        assert len(document["rows"]) == 64
        assert document["rows"][0][0:3] == [0, int(expected[20]), [int(expected[20])]]
        assert document["rows"][0][4:] == [0, "matched"]
        assert all(row[5] == "matched" for row in document["rows"][:32])
        assert all(row[5] == "outside" for row in document["rows"][32:])
        export_stem = tmp_path / "symbol-exports" / "capture-symbols"
        export_stem.parent.mkdir()
        monkeypatch.setattr(
            QtWidgets.QFileDialog,
            "getSaveFileName",
            lambda *args, **kwargs: (str(export_stem), ""),
        )
        window._export_symbol_table()
        export_path = export_stem.with_suffix(".vsasymbols.json")
        written = json.loads(export_path.read_text(encoding="utf-8"))
        assert written == document
        assert window._last_directory("symbol_table") == str(
            export_stem.parent.resolve()
        )
    finally:
        window._meas_config_dialog.close()
        window.close()
        window.deleteLater()
        QtWidgets.QApplication.processEvents()


def test_iq_export_can_save_raw_or_software_dc_removed_capture(
    tmp_path, monkeypatch,
) -> None:
    pg.mkQApp("VSA IQ export test")
    generated, signal = GeneratedIQSource.fsk(symbol_count=96, seed=908)
    recording = replace(
        generated,
        iq=(np.asarray(generated.iq) + (0.27 - 0.14j)).astype(np.complex64),
        source="VSA Pluto Single",
        metadata={
            **dict(generated.metadata),
            "dc_removal_recommended": True,
        },
    )
    window = VSAWindow(
        preferences=_isolated_preferences(tmp_path, "iq-export")
    )
    try:
        window.load_recording(recording, signal)
        _wait_for_background_analysis(window)
        assert window.export_iq_action.isEnabled()

        export_dir = tmp_path / "iq-exports"
        export_dir.mkdir()
        raw_stem = export_dir / "raw-capture"
        monkeypatch.setattr(
            QtWidgets.QInputDialog,
            "getItem",
            lambda *args, **kwargs: ("Raw capture", True),
        )
        monkeypatch.setattr(
            QtWidgets.QFileDialog,
            "getSaveFileName",
            lambda *args, **kwargs: (str(raw_stem), ""),
        )
        window._export_iq_recording()
        raw = FileIQSource.load(raw_stem.with_suffix(".npz"))
        np.testing.assert_array_equal(raw.iq, recording.iq)
        assert raw.metadata["dc_removal_recommended"] is True

        corrected_stem = export_dir / "dc-removed"
        monkeypatch.setattr(
            QtWidgets.QInputDialog,
            "getItem",
            lambda *args, **kwargs: (
                "Software DC removed (full-rate capture)",
                True,
            ),
        )
        monkeypatch.setattr(
            QtWidgets.QFileDialog,
            "getSaveFileName",
            lambda *args, **kwargs: (str(corrected_stem), ""),
        )
        window._export_iq_recording()
        corrected = FileIQSource.load(corrected_stem.with_suffix(".npz"))
        offset = complex(
            corrected.metadata["software_dc_offset_real"],
            corrected.metadata["software_dc_offset_imag"],
        )
        np.testing.assert_allclose(
            corrected.iq,
            np.asarray(recording.iq) - offset,
            atol=2e-7,
        )
        assert corrected.sample_count == recording.sample_count
        assert corrected.sample_rate_hz == recording.sample_rate_hz
        assert corrected.metadata["software_dc_removal_applied"] is True
        assert corrected.metadata["dc_removal_recommended"] is False
        assert window._last_directory("iq") == str(export_dir.resolve())
    finally:
        window._meas_config_dialog.close()
        window.close()
        window.deleteLater()
        QtWidgets.QApplication.processEvents()


def test_lsb_bit_ordering_applies_to_psk_pattern_table_and_export(tmp_path) -> None:
    pg.mkQApp("VSA LSB Symbol Table test")
    window = VSAWindow(
        preferences=_isolated_preferences(tmp_path, "lsb-symbol-table")
    )
    try:
        recording, signal = GeneratedIQSource.psk(
            modulation=ModulationKind.PI4_DQPSK,
            symbol_count=180,
            seed=822,
        )
        expected = np.asarray(
            recording.metadata["generated_symbols"], dtype=np.int16
        )
        displayed = np.asarray([0, 2, 1, 3], dtype=np.int16)[expected]
        window.load_recording(recording, signal)
        _wait_for_background_analysis(window)
        window.pattern_search_check.setChecked(True)
        window.bit_order_combo.setCurrentText("LSB")
        window._set_pattern_symbols(displayed[30:54])
        window.result_length_spin.setValue(64)
        assert window._analyze()

        result = window.session.pattern_result
        assert result is not None
        np.testing.assert_array_equal(result.decoded_symbols, expected[30:94])
        table_values = np.asarray(
            [
                int(window.symbol_table.item(index // 10, index % 10).text())
                for index in range(result.decoded_symbols.size)
            ]
        )
        np.testing.assert_array_equal(table_values, displayed[30:94])
        document = window._symbol_table_export_document()
        assert document["metadata"]["bit_ordering"] == "LSB"
        assert document["rows"][0][1] == int(displayed[30])
        assert document["rows"][0][2] == [
            (int(displayed[30]) >> 1) & 1,
            int(displayed[30]) & 1,
        ]
        assert all(row[5] == "matched" for row in document["rows"][:24])
    finally:
        window._meas_config_dialog.close()
        window.close()
        window.deleteLater()
        QtWidgets.QApplication.processEvents()
