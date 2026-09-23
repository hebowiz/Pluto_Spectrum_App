import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
from pyqtgraph.Qt import QtCore, QtWidgets
from pluto_vsa.ui.main_window import VSAWindow


def _isolated_preferences(tmp_path: Path, name: str) -> QtCore.QSettings:
    return QtCore.QSettings(
        str(tmp_path / f"{name}.ini"), QtCore.QSettings.Format.IniFormat
    )


def _wait_for_background_analysis(window: VSAWindow) -> None:
    for _index in range(500):
        QtWidgets.QApplication.processEvents()
        thread = window._analysis_thread
        if thread is None and window._pending_analysis is None:
            QtWidgets.QApplication.processEvents()
            return
        if thread is not None:
            thread.wait(10)
    raise AssertionError("background VSA analysis did not finish")
