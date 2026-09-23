"""Window geometry persists; instrument pane layouts remain session-local."""

import os
from unittest.mock import MagicMock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import iio
import pyqtgraph as pg
import pytest
from pyqtgraph.Qt import QtCore, QtWidgets

from pluto_common.config.spectrum_config import SpectrumConfig
from pluto_common.window_geometry import WINDOW_GEOMETRY_KEY, restore_window_geometry
from pluto_rtsa.signal.spectrum_processor import SpectrumProcessor
from pluto_rtsa.ui.session_window import SessionRealtimeSpectrumWindow
from pluto_vsa.ui.application_window import PlutoAnalysisWindow
from pluto_vsg.ui.main_window import PlutoVSGWindow


@pytest.fixture
def app(monkeypatch):
    application = pg.mkQApp("Window layout regression")
    monkeypatch.setattr(iio, "scan_contexts", lambda: {})
    return application


@pytest.fixture
def preferences(tmp_path):
    return QtCore.QSettings(str(tmp_path / "layout.ini"), QtCore.QSettings.Format.IniFormat)


def make_window(kind, preferences):
    if kind == "rtsa":
        config = SpectrumConfig()
        receiver = MagicMock()
        receiver.device_selector = "layout-test"
        receiver.get_received_sample_count.return_value = 0
        sweep = MagicMock()
        sweep.estimate_sweep_time_seconds.return_value = 0.1
        window = SessionRealtimeSpectrumWindow(
            config, receiver, SpectrumProcessor(config), sweep,
            calibration_offset_db=0.0, preferences=preferences,
        )
        window.timer.stop()
        return window
    if kind == "vsa":
        return PlutoAnalysisWindow(preferences=preferences, pluto_source=MagicMock())
    return PlutoVSGWindow(preferences=preferences, restore_startup_state=True)


@pytest.mark.parametrize("kind", ["rtsa", "vsa", "vsg"])
def test_main_window_geometry_round_trip_and_minimum(app, preferences, kind):
    first = make_window(kind, preferences)
    try:
        assert first.minimumSize() == QtCore.QSize(960, 640)
        first.resize(1080, 680)
        first.move(35, 45)
        saved = first.saveGeometry()
    finally:
        first.close()
    assert preferences.value(WINDOW_GEOMETRY_KEY) == saved

    # Qt may adjust geometry to the available test screen; compare against
    # Qt's own restore result rather than assuming a desktop resolution.
    expected = QtWidgets.QMainWindow()
    expected.setMinimumSize(960, 640)
    assert expected.restoreGeometry(saved)
    second = make_window(kind, preferences)
    try:
        assert second.geometry() == expected.geometry()
        assert second.minimumSize() == QtCore.QSize(960, 640)
    finally:
        second.close()
        expected.close()


def test_bad_geometry_does_not_prevent_startup(app, preferences):
    preferences.setValue(WINDOW_GEOMETRY_KEY, "not a geometry byte array")
    window = QtWidgets.QMainWindow()
    window.resize(1200, 700)
    restore_window_geometry(window, preferences)
    assert window.size() == QtCore.QSize(1200, 700)
    assert window.minimumSize() == QtCore.QSize(960, 640)
    window.close()


@pytest.mark.parametrize("kind", ["rtsa", "vsa", "vsg"])
def test_visible_windows_can_shrink_to_common_minimum(app, preferences, kind):
    window = make_window(kind, preferences)
    try:
        window.show()
        app.processEvents()
        window.resize(960, 640)
        app.processEvents()
        assert window.size() == QtCore.QSize(960, 640)
    finally:
        window.close()


def test_vsa_restores_per_mode_docks_only_during_current_session(app, preferences):
    window = make_window("vsa", preferences)
    window.show()
    app.processEvents()
    generic = window.generic_workspace
    try:
        generic.tabifyDockWidget(generic.zero_span_dock, generic.spectrum_dock)
        generic.modulation_dock.setFloating(True)
        generic.modulation_dock.resize(420, 310)
        generic.modulation_dock.show()
        app.processEvents()
        window.set_analysis_mode("bluetooth")
        app.processEvents()
        assert not generic.modulation_dock.isVisible()
        bluetooth = window.bluetooth_workspace
        bluetooth.packet_dock.setFloating(True)
        bluetooth.packet_dock.show()
        app.processEvents()
        window.set_analysis_mode("generic")
        app.processEvents()
        assert generic.modulation_dock.isFloating()
        assert generic.modulation_dock.isVisible()
        assert generic.spectrum_dock in generic.tabifiedDockWidgets(generic.zero_span_dock)
        assert not bluetooth.packet_dock.isVisible()
        window.resize(window.width() + 120, window.height() + 80)
        app.processEvents()
        window.set_analysis_mode("bluetooth")
        app.processEvents()
        assert bluetooth.packet_dock.isFloating()
        assert bluetooth.packet_dock.isVisible()
        assert not generic.modulation_dock.isVisible()
    finally:
        window.close()
    assert not generic.modulation_dock.isVisible()
    assert not bluetooth.packet_dock.isVisible()
    reopened = make_window("vsa", preferences)
    try:
        assert reopened._active_mode() == "generic"
        assert not reopened._workspace_layouts
        assert not reopened.generic_workspace.modulation_dock.isFloating()
        assert not reopened.generic_workspace.tabifiedDockWidgets(reopened.generic_workspace.zero_span_dock)
        assert not reopened.bluetooth_workspace.packet_dock.isFloating()
    finally:
        reopened.close()


@pytest.mark.parametrize("mode", ["generic", "bluetooth", "dect", "wifi", "adsb1090"])
def test_vsa_resizing_and_returning_to_mode_does_not_equalize(app, preferences, monkeypatch, mode):
    window = make_window("vsa", preferences)
    try:
        window.show()
        window.set_analysis_mode(mode)
        app.processEvents()
        workspace = window._active_workspace()
        equalize = "_equalize_docks" if mode in ("bluetooth", "dect", "wifi") else "_equalize_result_docks"
        callback = MagicMock()
        monkeypatch.setattr(workspace, equalize, callback)
        window.resize(window.width() + 100, window.height() + 80)
        app.processEvents()
        window.set_analysis_mode("bluetooth" if mode == "generic" else "generic")
        app.processEvents()
        window.set_analysis_mode(mode)
        app.processEvents()
        callback.assert_not_called()
    finally:
        window.close()


@pytest.mark.parametrize("mode", ["generic", "bluetooth", "dect", "wifi", "adsb1090"])
def test_vsa_mode_switch_restores_user_split_sizes(app, preferences, mode):
    window = make_window("vsa", preferences)
    try:
        window.show()
        window.set_analysis_mode(mode)
        app.processEvents()
        workspace = window._active_workspace()
        if mode == "generic":
            row = [workspace.zero_span_dock, workspace.spectrum_dock, workspace.result_summary_dock]
        elif mode == "adsb1090":
            row = [workspace.power_dock, workspace.packet_dock, workspace.aircraft_dock]
        else:
            row = [workspace.power_dock, workspace.spectrum_dock, workspace.summary_dock]
        workspace.resizeDocks(row, [300, 650, 350], QtCore.Qt.Orientation.Horizontal)
        app.processEvents()
        widths = [dock.width() for dock in row]
        window.set_analysis_mode("bluetooth" if mode == "generic" else "generic")
        app.processEvents()
        # Exercise a hidden workspace whose geometry was changed while away.
        workspace.resize(workspace.width() - 150, workspace.height() - 100)
        app.processEvents()
        window.set_analysis_mode(mode)
        app.processEvents()
        assert [dock.width() for dock in row] == pytest.approx(widths, abs=3)
    finally:
        window.close()


def test_vsg_docks_are_movable_and_reset_on_restart_without_saving_tabs(app, preferences):
    window = make_window("vsg", preferences)
    window.show()
    app.processEvents()
    try:
        docks = window.workspace.findChildren(QtWidgets.QDockWidget)
        assert len(docks) == 5
        for dock in docks:
            assert dock.features() & QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
            assert dock.features() & QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable
            assert not dock.features() & QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
        window.composer_dock.widget().setCurrentIndex(1)
        window.preview_dock.widget().setCurrentIndex(3)
        window.workspace.tabifyDockWidget(window.library_dock, window.inspector_dock)
        window.preview_dock.setFloating(True)
        window.preview_dock.show()
        app.processEvents()
        # Old releases wrote a main-window state. Even when it exists, only
        # the main geometry should be restored by the new implementation.
        preferences.setValue("startup/window_state", window.workspace.saveState())
    finally:
        window.close()
    assert not window.preview_dock.isVisible()
    assert not preferences.contains("startup/window_state")
    reopened = make_window("vsg", preferences)
    try:
        assert not reopened.preview_dock.isFloating()
        assert not reopened.workspace.tabifiedDockWidgets(reopened.library_dock)
        assert reopened.composer_dock.widget().currentIndex() == 0
        assert reopened.preview_dock.widget().currentIndex() == 0
    finally:
        reopened.close()
