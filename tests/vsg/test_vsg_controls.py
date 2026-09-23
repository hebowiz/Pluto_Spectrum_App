import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from pluto_vsg.model import BluetoothLEPhy
from pluto_vsg.ui.main_window import PlutoVSGWindow, _Panel


def test_new_menu_uses_shared_le_and_plain_hdt_packet_actions() -> None:
    pg.mkQApp("Pluto VSG unified Bluetooth New menu")
    window = PlutoVSGWindow()
    try:
        assert window.new_le_action.text() == "New Bluetooth LE Packet"
        assert window.new_hdt_action.text() == "New Bluetooth HDT Packet"
        assert not hasattr(window, "new_le1m_action")
        assert not hasattr(window, "new_le2m_action")
        window.new_le_action.trigger()
        assert window.project.bluetooth_le is not None
        assert window.project.bluetooth_le.phy == BluetoothLEPhy.LE_1M
    finally:
        window.close()


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
        assert window.findChild(QtWidgets.QMenuBar) is None
        assert window.undo_action in window.actions()
        assert window.redo_action in window.actions()
        assert window.generate_action in window.actions()
    finally:
        window.close()


def test_vsg_control_panel_defaults_and_auto_bandwidth() -> None:
    pg.mkQApp("Pluto VSG control panel defaults")
    window = PlutoVSGWindow()
    try:
        assert window.size() == QtCore.QSize(1600, 960)
        assert window.rf_button.text() == "Calibration"
        assert window.rf_button.isCheckable() is True
        assert window.rf_button.isChecked() is False
        assert window.mod_button.text() == "Mod\nON"
        assert window.continuous_button.text() == "Continuous\nON"
        settings = window._current_pluto_settings()
        assert settings.rf_bandwidth_hz == settings.sample_rate_hz
        assert window.frequency_button.text().endswith("MHz")
        assert window.rf_button.minimumHeight() == 72
        assert window.mod_button.minimumHeight() == 72
        assert window.continuous_button.minimumHeight() == 72
        assert window.frequency_button.minimumHeight() == 72
        assert window.packet_settings_button.minimumHeight() == 50
        assert window.received_fields_button.minimumHeight() == 72
        assert window.project_button.minimumHeight() == 50
        assert window.vsg_control_back_button.minimumHeight() == 50
        assert window.vsg_control_panel.minimumWidth() == 240
        assert window.vsg_control_panel.maximumWidth() == 240
        workspace_panels = window.findChildren(_Panel)
        assert workspace_panels
        assert all(
            f"left: {_Panel.TITLE_LEFT_INSET_PX}px" in panel.styleSheet()
            for panel in workspace_panels
        )
        assert (
            window.estimated_peak_label.font().pointSizeF()
            > window.font().pointSizeF()
        )
        assert (
            window.estimated_peak_label.font().pointSizeF()
            < window.power_button.font().pointSizeF()
        )
        assert window.vsg_control_panel.title() == "Main Menu"
        assert window.vsg_control_page_title.isHidden()
        assert window.vsg_setup_group.title() == "VSG SETUP"
        assert window.packet_group.title() == "PACKET"
        assert window.system_group.title() == "SYSTEM"
        assert window.vsg_setup_group.layout().indexOf(window.rf_button) >= 0
        assert (
            window.vsg_setup_group.layout().indexOf(
                window.frequency_settings_button
            )
            >= 0
        )
        assert window.packet_group.layout().indexOf(window.packet_settings_button) >= 0
        assert window.packet_group.layout().indexOf(window.verify_packet_button) >= 0
        assert window.system_group.layout().indexOf(window.project_button) >= 0
        assert window.system_group.layout().indexOf(window.instrument_settings_button) >= 0
        assert window.power_up_button.height() == 34
        assert window.power_down_button.height() == 34
        assert (
            window.power_up_button.height()
            + window.power_down_button.height()
            + 4
            == window.power_button.minimumHeight()
        )
        assert not hasattr(window, "save_as_action")
    finally:
        window.close()


def test_vsg_control_panel_project_file_and_new_navigation() -> None:
    pg.mkQApp("Pluto VSG hierarchical controls")
    window = PlutoVSGWindow()
    try:
        assert window.vsg_control_stack.currentWidget() is window.vsg_main_control_page
        window.project_button.click()
        assert window.vsg_control_page_title.text() == "Project"
        assert window.vsg_control_panel.title() == "Project"
        assert window.vsg_control_stack.currentWidget() is window.vsg_project_page
        assert window.project_open_button.text() == "Open"
        assert window.project_save_button.text() == "Save"

        window.project_new_button.click()
        assert window.vsg_control_page_title.text() == "New Project"
        assert window.vsg_control_stack.currentWidget() is window.vsg_new_project_page
        assert [
            window.new_bluetooth_button.text(),
            window.new_bluetooth_le_button.text(),
            window.new_bluetooth_hdt_button.text(),
            window.new_wifi_button.text(),
            window.new_dect_button.text(),
        ] == ["Bluetooth BR/EDR", "Bluetooth LE", "Bluetooth HDT", "Wi-Fi", "DECT"]

        right_click = QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseButtonPress,
            QtCore.QPointF(2.0, 2.0),
            QtCore.QPointF(2.0, 2.0),
            QtCore.Qt.MouseButton.RightButton,
            QtCore.Qt.MouseButton.RightButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
        QtWidgets.QApplication.sendEvent(window.new_dect_button, right_click)
        assert window.vsg_control_stack.currentWidget() is window.vsg_project_page
        assert window.vsg_control_panel.title() == "Project"
        window._show_vsg_main_controls()
        window.file_button.click()
        assert window.vsg_control_page_title.text() == "File"
        assert window.vsg_control_stack.currentWidget() is window.vsg_file_page
        assert [
            window.export_npz_button.text(),
            window.export_iqtar_button.text(),
            window.export_wv_button.text(),
        ] == ["Export NPZ", "Export IQ TAR", "Export WV"]
    finally:
        window.close()


def test_vsg_control_navigation_restores_main_scroll_position() -> None:
    app = pg.mkQApp("Pluto VSG control scroll history")
    window = PlutoVSGWindow()
    window.show()
    app.processEvents()
    scroll = window.vsg_main_control_page.verticalScrollBar()
    try:
        position = min(120, scroll.maximum())
        assert position > 0
        scroll.setValue(position)

        window.project_button.click()
        app.processEvents()
        assert window.vsg_control_stack.currentWidget() is window.vsg_project_page

        window._navigate_vsg_control_back()
        app.processEvents()
        assert scroll.value() == position
    finally:
        window.close()


def test_vsg_calibration_state_keeps_control_scroll_position() -> None:
    app = pg.mkQApp("Pluto VSG calibration scroll stability")
    window = PlutoVSGWindow()
    window.show()
    app.processEvents()
    scroll = window.vsg_main_control_page.verticalScrollBar()
    try:
        scroll.setValue(0)
        window.rf_button.setFocus()
        window._calibration_in_progress = True
        window._set_pluto_busy(preparing=True, transmitting=False)
        app.processEvents()

        assert scroll.value() == 0
        assert window.rf_button.text() == "Calibrating..."
    finally:
        window._calibration_in_progress = False
        window._set_pluto_busy(preparing=False, transmitting=False)
        window.close()
