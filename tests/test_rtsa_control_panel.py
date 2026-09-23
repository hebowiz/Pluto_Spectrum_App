from types import SimpleNamespace

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from pluto_common.control_panel import CONTROL_BUTTON_HEIGHT
from pluto_common.config.analyzer_mode import AnalyzerMode
from pluto_rtsa.ui.main_window import RealtimeSpectrumWindow, SWEEP_STATE_RUNNING


def test_rtsa_arrow_control_uses_native_expanding_tool_button():
    pg.mkQApp("RTSA arrow controls")

    button = RealtimeSpectrumWindow._make_arrow_control_button(
        object(), QtCore.Qt.ArrowType.LeftArrow
    )

    assert isinstance(button, QtWidgets.QToolButton)
    assert button.arrowType() == QtCore.Qt.ArrowType.LeftArrow
    assert button.minimumHeight() == CONTROL_BUTTON_HEIGHT
    assert (
        button.sizePolicy().horizontalPolicy()
        == QtWidgets.QSizePolicy.Policy.Expanding
    )


def test_rtsa_mode_selection_uses_checked_state_without_text_prefix():
    pg.mkQApp("RTSA mode controls")
    window = SimpleNamespace()
    window.config = SimpleNamespace(analyzer_mode=AnalyzerMode.SWEEP_SA)
    window.analyzer_mode_option_buttons = {
        mode: QtWidgets.QPushButton()
        for mode in AnalyzerMode
        if mode != AnalyzerMode.CALIBRATION
    }
    for button in window.analyzer_mode_option_buttons.values():
        button.setCheckable(True)
    window.calibration_mode_entry_button = QtWidgets.QPushButton()
    window.calibration_mode_entry_button.setCheckable(True)
    window._analyzer_mode_display_name = (
        RealtimeSpectrumWindow._analyzer_mode_display_name
    )

    RealtimeSpectrumWindow._update_analyzer_mode_selection_page(window)

    assert window.analyzer_mode_option_buttons[AnalyzerMode.SWEEP_SA].isChecked()
    assert all(
        not button.text().startswith(">")
        for button in window.analyzer_mode_option_buttons.values()
    )
    assert not window.calibration_mode_entry_button.isChecked()


def test_rtsa_continuous_selection_uses_checked_state_without_text_prefix():
    pg.mkQApp("RTSA sweep controls")
    window = SimpleNamespace(
        sweep_state=SWEEP_STATE_RUNNING,
        cont_button=QtWidgets.QPushButton(),
    )
    window.cont_button.setCheckable(True)

    RealtimeSpectrumWindow._update_continuous_button(window)

    assert window.cont_button.text() == "Continuous"
    assert window.cont_button.isChecked()
