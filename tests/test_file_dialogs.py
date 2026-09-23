"""File menus retain independent folders across application restarts."""

import os
from pathlib import Path
from unittest.mock import MagicMock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import iio
import pyqtgraph as pg
import pytest
from pyqtgraph.Qt import QtCore, QtWidgets

from pluto_common.file_dialogs import file_dialog_path, remember_file_directory


def settings(path):
    return QtCore.QSettings(str(path), QtCore.QSettings.Format.IniFormat)


@pytest.fixture
def app(monkeypatch):
    application = pg.mkQApp("File dialog history tests")
    monkeypatch.setattr(iio, "scan_contexts", lambda: {})
    # A failed operation must fail the test instead of blocking on a message box.
    def unexpected_message(*args):
        pytest.fail(str(args[1:]))
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", unexpected_message)
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", unexpected_message)
    return application


def test_history_persists_per_kind_and_application(tmp_path):
    first = settings(tmp_path / "vsa.ini")
    for kind in ("iq", "pattern", "config"):
        folder = tmp_path / kind
        folder.mkdir()
        remember_file_directory(first, f"directories/{kind}", folder / "selected.json")
    reopened = settings(tmp_path / "vsa.ini")
    for kind in ("iq", "pattern", "config"):
        assert file_dialog_path(reopened, f"directories/{kind}") == str(tmp_path / kind)
    other_app = settings(tmp_path / "vsg.ini")
    assert file_dialog_path(other_app, "directories/config") == str(Path.cwd())
    remember_file_directory(reopened, "directories/config", "")
    assert file_dialog_path(reopened, "directories/config") == str(tmp_path / "config")


def test_missing_directory_falls_back_without_losing_suggested_filename(tmp_path):
    prefs = settings(tmp_path / "history.ini")
    prefs.setValue("directories/iq", str(tmp_path / "deleted"))
    assert file_dialog_path(
        prefs, "directories/iq", filename="waveform.iq.tar", default_directory=tmp_path
    ) == str(tmp_path / "waveform.iq.tar")
    assert file_dialog_path(prefs, "directories/iq", filename="waveform.npz") == str(
        Path.cwd() / "waveform.npz"
    )


def test_draft_remembers_file_navigation_without_persisting_measurement_edits(tmp_path):
    from pluto_vsa.ui.config_transaction import DraftPreferences

    prefs = settings(tmp_path / "vsa.ini")
    prefs.setValue("measurement/value", "original")
    draft = DraftPreferences(prefs)
    draft.setValue("measurement/value", "edited")
    remember_file_directory(draft, "directories/pattern", tmp_path / "pattern.json")
    reopened = settings(tmp_path / "vsa.ini")
    assert reopened.value("measurement/value") == "original"
    assert file_dialog_path(reopened, "directories/pattern") == str(tmp_path)


def test_vsg_file_menus_restore_independent_folders_after_restart(app, tmp_path, monkeypatch):
    from pluto_vsg.ui.main_window import PlutoVSGWindow

    ini = tmp_path / "vsg.ini"
    selections = {
        "Save Pluto VSG Project": tmp_path / "projects" / "selected.pvsg.json",
        "Export IQ": tmp_path / "numpy" / "selected.npz",
        "Export R&S IQ TAR": tmp_path / "tar" / "selected.iq.tar",
        "Export R&S WV": tmp_path / "wv" / "selected.wv",
    }
    for path in selections.values():
        path.parent.mkdir()
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getSaveFileName",
        lambda parent, title, initial, filters: (str(selections[title]), ""),
    )
    window = PlutoVSGWindow(preferences=settings(ini))
    try:
        window._save_project()
        window._export_npz()
        window._export_iq_tar()
        window._export_wv()
        assert all(path.is_file() for path in selections.values())
    finally:
        window.close()

    initial_paths = {}
    def cancel(parent, title, initial, filters):
        initial_paths[title] = Path(initial)
        return "", ""
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", cancel)
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", cancel)
    window = PlutoVSGWindow(preferences=settings(ini))
    try:
        before = {key: window._preferences.value(key) for key in window._preferences.allKeys()}
        window._open_project()
        window._save_project()
        window._export_npz()
        window._export_iq_tar()
        window._export_wv()
        for title, selected in selections.items():
            assert initial_paths[title].parent == selected.parent
        assert initial_paths["Open Pluto VSG Project"] == selections["Save Pluto VSG Project"].parent
        assert initial_paths["Save Pluto VSG Project"].name == "waveform.pvsg.json"
        assert initial_paths["Export R&S IQ TAR"].name == "waveform.iq.tar"
        assert {key: window._preferences.value(key) for key in window._preferences.allKeys()} == before
    finally:
        window.close()


def test_rtsa_state_and_calibration_menus_keep_separate_folders(app, tmp_path, monkeypatch):
    from pluto_common.config.spectrum_config import SpectrumConfig
    from pluto_rtsa.signal.spectrum_processor import SpectrumProcessor
    from pluto_rtsa.ui.calibration_controller import CalibrationPointResult
    from pluto_rtsa.ui.session_window import SessionRealtimeSpectrumWindow
    from pluto_rtsa.ui.main_window import RealtimeSpectrumWindow

    # Keep application settings and calibration output inside the test folder.
    monkeypatch.setattr(RealtimeSpectrumWindow, "_save_app_settings", lambda self: None)
    monkeypatch.setattr(RealtimeSpectrumWindow, "_load_app_settings", lambda self: None)
    ini = tmp_path / "rtsa.ini"
    def make_window():
        config = SpectrumConfig()
        receiver = MagicMock()
        receiver.device_selector = "file-dialog-test"
        receiver.get_received_sample_count.return_value = 0
        sweep = MagicMock()
        sweep.estimate_sweep_time_seconds.return_value = 0.1
        window = SessionRealtimeSpectrumWindow(
            config, receiver, SpectrumProcessor(config), sweep,
            calibration_offset_db=0.0, preferences=settings(ini),
        )
        window.timer.stop()
        window.calibration_controller.measurement_results = [
            CalibrationPointResult(100_000_000, -20.0, -19.0, 1.0)
        ]
        return window

    selections = {
        "Save RTSA State": tmp_path / "states" / "state.rtsastate.json",
        "Load Calibration Correction CSV": tmp_path / "corrections" / "correction.csv",
        "Load Calibration Reference CSV": tmp_path / "references" / "reference.csv",
        "Save Calibration Result CSV": tmp_path / "results" / "result.csv",
    }
    for path in selections.values():
        path.parent.mkdir()
    selections["Load Calibration Correction CSV"].write_text(
        "frequency_hz,calibration_offset_db\n100000000,1\n", encoding="utf-8"
    )
    selections["Load Calibration Reference CSV"].write_text(
        "frequency_hz,reference_power_dbm\n100000000,-19\n", encoding="utf-8"
    )
    def choose(parent, title, initial, filters):
        return str(selections[title]), ""
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", choose)
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", choose)
    window = make_window()
    try:
        window._on_state_save_clicked()
        window._on_calibration_load_correction_csv_clicked()
        window._on_calibration_load_reference_csv_clicked()
        # Loading a new reference intentionally discards old measurement results.
        window.calibration_controller.measurement_results = [
            CalibrationPointResult(100_000_000, -20.0, -19.0, 1.0)
        ]
        assert window._save_calibration_results_csv()
        assert all(path.is_file() for path in selections.values())
    finally:
        window.close()

    initial_paths = {}
    def cancel(parent, title, initial, filters):
        initial_paths[title] = Path(initial)
        return "", ""
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", cancel)
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", cancel)
    window = make_window()
    try:
        window._on_state_save_clicked()
        window._on_state_recall_clicked()
        window._on_calibration_load_correction_csv_clicked()
        window._on_calibration_load_reference_csv_clicked()
        window._save_calibration_results_csv()
        for title, selected in selections.items():
            actual = initial_paths[title]
            if title == "Save Calibration Result CSV":
                actual = actual.parent
            assert actual == selected.parent
        assert initial_paths["Recall RTSA State"] == selections["Save RTSA State"].parent
    finally:
        window.close()
