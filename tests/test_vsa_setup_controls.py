import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyqtgraph as pg
import pytest
from pyqtgraph.Qt import QtCore, QtWidgets

from pluto_vsa.ui.application_window import PlutoAnalysisWindow
from pluto_vsa.model import IQRecording
from pluto_vsa.protocol_modes.dect import analyze_dect_recording, generate_dect_packet
from pluto_vsa.pattern import IQPowerTriggerSettings
from pluto_vsa.standards.adsb1090 import ADSB1090Analyzer, ADSB1090Settings
from pluto_vsa.sources import FileIQSource
from pathlib import Path


@pytest.fixture
def shell(tmp_path):
    pg.mkQApp("Common VSA setup test")
    prefs = QtCore.QSettings(str(tmp_path / "setup.ini"), QtCore.QSettings.Format.IniFormat)
    prefs.setValue("metadata/sqlite_path", str(tmp_path / "aircraft.sqlite"))
    window = PlutoAnalysisWindow(preferences=prefs)
    yield window
    window.close()
    window.deleteLater()


def test_setup_button_order_and_all_dialog_routes(shell, monkeypatch):
    expected = {
        "generic": ("Signal Description", "Input / Frontend", "Signal Capture", "Trigger", "Pattern Search", "Result Range", "Demodulation", "Result Summary", "Display"),
        "bluetooth": ("Signal Description", "Input / Frontend", "Signal Capture", "Trigger", "Display"),
        "dect": ("Signal Description", "Input / Frontend", "Signal Capture", "Trigger", "Display"),
        "adsb1090": ("Signal Description", "Input / Frontend", "Signal Capture", "Trigger"),
    }
    for mode, names in expected.items():
        shell.set_analysis_mode(mode)
        workspace = shell._active_workspace()
        spec = shell.control_panel._spec
        assert tuple(command.label for command in spec.setup) == names
        dialog = getattr(workspace, "_meas_config_dialog", None) or getattr(workspace, "_config_dialog", None) or workspace._setup_dialog
        visited = []
        monkeypatch.setattr(dialog, "exec", lambda: visited.append(dialog.page_title.text()) or 0)
        for command in spec.setup:
            command.callback()
        assert visited == list(names)
        assert spec.files[0].label == "Import IQ"
        assert {"Device", "State", "File", "Recall", "Save", "Preset"}.issubset(shell.control_panel.buttons)
        assert "Default" not in shell.control_panel.buttons
        for page_name in ("Input / Frontend", "Trigger", "Signal Capture"):
            page = dialog.stack.widget(dialog.page_names.index(page_name))
            forms = page.findChildren(QtWidgets.QFormLayout)
            assert forms and all(form.verticalSpacing() == 10 for form in forms)


def test_adsb_moved_toolbar_controls_are_visible_and_editable(shell):
    workspace = shell.adsb1090_workspace
    dialog = workspace._setup_dialog
    dialog.show_page(dialog.page_names.index("Signal Description"))
    dialog.show()
    QtWidgets.QApplication.processEvents()
    assert workspace.preamble_snr_spin.isVisibleTo(dialog)
    assert workspace.preamble_snr_spin.isEnabled()
    workspace.preamble_snr_spin.setValue(7.5)
    assert workspace._analysis_settings().minimum_preamble_snr_db == 7.5
    for name, control in (("Input / Frontend", workspace.internal_gain_spin),
                          ("Signal Capture", workspace.sample_rate_combo)):
        dialog.show_page(dialog.page_names.index(name))
        assert control.isVisibleTo(dialog)
    dialog.hide()


@pytest.mark.parametrize("mode", ["generic", "bluetooth", "dect", "adsb1090"])
@pytest.mark.parametrize("finish", ["accept", "reject", "close", "escape"])
def test_config_edits_are_isolated_until_ok(shell, mode, finish, monkeypatch):
    from pluto_vsa.ui.config_transaction import collect_settings
    from pyqtgraph.Qt import QtGui

    shell.set_analysis_mode(mode)
    workspace = shell._active_workspace()
    dialog = (getattr(workspace, "_meas_config_dialog", None)
              or getattr(workspace, "_config_dialog", None)
              or workspace._setup_dialog)
    baseline = collect_settings(workspace)
    preferences = workspace._preferences
    saved = {key: preferences.value(key) for key in preferences.allKeys()}
    plot = workspace.zero_span_plot if mode == "generic" else workspace.power_plot
    item = plot.plot([0, 1], [1, 2])
    calls = []
    for method in ("refresh", "analyze_recording", "_request_analysis"):
        if hasattr(workspace, method):
            monkeypatch.setattr(workspace, method, lambda *args, **kwargs: calls.append(True))
    errors = []
    edited = []

    def edit():
        editor = QtWidgets.QApplication.activeModalWidget()
        try:
            assert editor._is_draft
            draft = editor.settings_owner
            draft._common_setup.external_gain.setValue(3.5)
            if mode == "adsb1090":
                draft.preamble_snr_spin.setValue(7.5)
            edited.append(collect_settings(draft))
            assert edited[0] != baseline
            assert collect_settings(workspace) == baseline
            assert {key: preferences.value(key) for key in preferences.allKeys()} == saved
            assert item in plot.listDataItems()
            assert not calls
        except Exception as error:
            errors.append(error)
            editor.reject()
            return
        if finish == "escape":
            event = QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress, QtCore.Qt.Key.Key_Escape,
                                   QtCore.Qt.KeyboardModifier.NoModifier)
            QtWidgets.QApplication.sendEvent(editor, event)
        else:
            getattr(editor, finish)()

    QtCore.QTimer.singleShot(0, edit)
    dialog.open_page("Input / Frontend")
    if errors:
        raise errors[0]
    assert collect_settings(workspace) == (edited[0] if finish == "accept" else baseline)
    assert item in plot.listDataItems()
    assert not calls
    if finish != "accept":
        assert {key: preferences.value(key) for key in preferences.allKeys()} == saved


def test_capture_units_bandwidth_and_power_correction_are_mode_local(shell):
    for mode in ("bluetooth", "dect"):
        shell.set_analysis_mode(mode)
        workspace = shell._active_workspace()
        common = workspace._common_setup
        common.unit.setCurrentText("Symbols")
        common.length.setValue(2304)
        symbol_rate = 1e6 if mode == "bluetooth" else 1.152e6
        assert workspace._capture_settings().capture_length_s == pytest.approx(2304 / symbol_rate)
        common.match_bandwidth.setChecked(True)
        assert not common.bandwidth.isEnabled()
        assert workspace._capture_settings().rf_bandwidth_hz == workspace._capture_settings().requested_sample_rate_hz
        common.external_gain.setValue(4)
        common.swap_iq.setChecked(True)
        assert workspace._capture_settings().swap_iq
        assert workspace._capture_settings().power_correction.external_gain_db == 4
        saved = shell._collect_meas_config(mode)
        common.unit.setCurrentText("ms")
        common.match_bandwidth.setChecked(False)
        common.external_gain.setValue(0)
        shell._apply_meas_config(mode, saved)
        assert common.unit.currentText() == "Symbols"
        assert common.match_bandwidth.isChecked()
        assert common.external_gain.value() == 4
        assert common.swap_iq.isChecked()
    assert shell.generic_workspace.external_gain_spin.value() == 0
    assert shell.adsb1090_workspace._common_setup.unit.count() == 1


def test_preset_confirmation_cancel_and_busy_state(shell, monkeypatch):
    shell.set_analysis_mode("bluetooth")
    workspace = shell.bluetooth_workspace
    workspace.center_spin.setValue(2420)
    shell.generic_workspace._set_selected_pluto_target("ip:192.0.2.1")
    shell._pluto_target_changed("")
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *args: QtWidgets.QMessageBox.StandardButton.Cancel)
    shell.control_panel.buttons["Preset"].click()
    assert workspace.center_spin.value() == 2420
    workspace.run_action.setText("Stop")
    assert not shell.control_panel.buttons["Preset"].isEnabled()
    assert not shell.control_panel.buttons["Reset"].isEnabled()
    workspace.run_action.setText("Run Single")
    assert shell.control_panel.buttons["Preset"].isEnabled()
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *args: QtWidgets.QMessageBox.StandardButton.Ok)
    shell.control_panel.buttons["Preset"].click()
    assert workspace.center_spin.value() == 2440
    assert workspace._pluto_target == "ip:192.0.2.1"


def test_linked_rf_bandwidth_reports_readback_and_blocks_unavailable_range(shell, monkeypatch):
    workspace = shell.generic_workspace
    common = workspace._common_setup
    workspace.session.recording = IQRecording(np.ones(128), 8e6,
                                              metadata={"actual_rf_bandwidth_hz": 7_900_000})
    common.match_bandwidth.setChecked(True)
    common.refresh()
    assert common.bandwidth.value() == 8.0
    assert common.applied_bandwidth.text().startswith("7.900 MHz")
    workspace.capture_oversampling_combo.setCurrentIndex(workspace.capture_oversampling_combo.findData(64))
    warnings = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args: warnings.append(args[1]))
    workspace._meas_config_dialog.accept()
    assert warnings == ["Invalid RF Bandwidth"]
    assert workspace._meas_config_dialog.result() != QtWidgets.QDialog.DialogCode.Accepted


def test_reset_clears_acquisition_and_display_without_changing_settings(shell):
    for mode in ("generic", "bluetooth", "dect", "adsb1090"):
        shell.set_analysis_mode(mode)
        workspace = shell._active_workspace()
        saved = shell._collect_meas_config(mode)
        recording = IQRecording(np.ones(1024), 8e6)
        if mode == "generic":
            workspace.session.recording = recording
            plot = workspace.zero_span_plot
        elif mode == "adsb1090":
            workspace.recording = recording
            plot = workspace.power_plot
        else:
            workspace._recording = recording
            workspace._capture_recording = recording
            plot = workspace.power_plot
        plot.plot([0, 1], [1, 2])
        workspace.export_iq_action.setEnabled(True)
        shell.control_panel.buttons["Reset"].click()
        assert not plot.listDataItems()
        assert not workspace.export_iq_action.isEnabled()
        assert shell._collect_meas_config(mode) == saved
        assert (workspace.session.recording if mode == "generic" else workspace.recording if mode == "adsb1090" else workspace._recording) is None


def test_dect_burst_search_disabled_is_identical_and_threshold_is_applied():
    recording = generate_dect_packet()
    baseline = analyze_dect_recording(recording)
    unchanged = analyze_dect_recording(recording, iq_power_trigger=IQPowerTriggerSettings(enabled=False))
    assert [result.p0_sample for result in unchanged] == [result.p0_sample for result in baseline]
    assert not analyze_dect_recording(recording, iq_power_trigger=IQPowerTriggerSettings(enabled=True, level_dbm=100))


def test_adsb_burst_search_uses_calibrated_capture_coordinates():
    recording = FileIQSource.load(Path(__file__).parent / "fixtures" / "adsb1090_multi_8msps.npz")
    baseline = ADSB1090Analyzer().analyze(recording, ADSB1090Settings(minimum_preamble_snr_db=5))
    peak = 20 * np.log10(np.max(np.abs(recording.iq)) / recording.full_scale) + recording.dbfs_to_dbm_offset_db
    gated = ADSB1090Analyzer().analyze(recording, ADSB1090Settings(
        minimum_preamble_snr_db=5,
        iq_power_trigger=IQPowerTriggerSettings(enabled=True, level_dbm=peak - 10,
                                               limit_result_to_active_interval=False),
    ))
    assert [(m.raw_hex, m.start_sample) for m in gated.messages] == [(m.raw_hex, m.start_sample) for m in baseline.messages]
    rejected = ADSB1090Analyzer().analyze(recording, ADSB1090Settings(
        minimum_preamble_snr_db=5,
        iq_power_trigger=IQPowerTriggerSettings(enabled=True, level_dbm=100),
    ))
    assert not rejected.messages
