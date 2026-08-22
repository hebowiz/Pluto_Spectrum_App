import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

from pluto_sa.vsa.ui.application_window import PlutoAnalysisWindow


class _SharedPlutoSource:
    def __init__(self) -> None:
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1


def test_single_window_switches_complete_workspaces_and_shares_pluto(tmp_path) -> None:
    pg.mkQApp("Pluto analysis shell test")
    preferences = QtCore.QSettings(
        str(tmp_path / "analysis-shell.ini"),
        QtCore.QSettings.Format.IniFormat,
    )
    source = _SharedPlutoSource()
    window = PlutoAnalysisWindow(pluto_source=source, preferences=preferences)
    try:
        assert window._stack.currentWidget() is window.generic_workspace
        assert "Generic" in window.windowTitle()
        assert window.generic_workspace._pluto_source is source
        assert window.adsb1090_workspace._pluto_source is source

        window.set_analysis_mode("adsb1090")
        assert window._stack.currentWidget() is window.adsb1090_workspace
        assert "ADS-B 1090ES" in window.windowTitle()

        window.set_analysis_mode("generic")
        assert window._stack.currentWidget() is window.generic_workspace
    finally:
        window.generic_workspace._meas_config_dialog.close()
        window.close()
        window.deleteLater()
    assert source.close_count == 1
