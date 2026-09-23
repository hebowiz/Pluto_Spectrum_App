import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')

import pyqtgraph as pg
from pluto_vsg.model import WiFiSettings, WiFiPSDUSource
from pluto_vsg.profiles.wifi import wifi_project
from pluto_vsg.ui.wifi_settings import WiFiSettingsDialog


def test_visible_rf_controls_field_groups_and_all_values_roundtrip():
    app = pg.mkQApp('Wi-Fi editor')
    dialog = WiFiSettingsDialog(wifi_project())
    try:
        dialog.show()
        app.processEvents()
        assert dialog.channel_combo.isVisible()
        assert dialog.frequency_offset_spin.isVisible()
        assert dialog.repeat_spin.isVisible()
        assert dialog.field_pages.count()==4
        dialog.channel_combo.setCurrentIndex(10)
        dialog.frequency_offset_spin.setValue(125)
        dialog.repeat_spin.setValue(2)
        dialog.ssid_edit.setText('Edited Beacon')
        dialog.timestamp_edit.setText('123456')
        dialog.sequence_spin.setValue(123)
        dialog.fragment_spin.setValue(4)
        dialog.beacon_interval_spin.setValue(50)
        dialog.interval_period_button.click()
        assert dialog.period_spin.value()==51200
        dialog._accept_settings()
        assert dialog.project.repeat_count==2
        assert dialog.project.center_frequency_hz==2462125000
        assert dialog.project.wifi.timestamp==123456
        assert dialog.project.wifi.fragment_number==4
        assert dialog.project.wifi.ssid=='Edited Beacon'
        assert 'Signal Extension: 6 us' in dialog.derived_label.text()
        assert 'static' in dialog.derived_label.text()
    finally:
        dialog.close()


def test_raw_fcs_editing_and_cancel_preserve_original():
    pg.mkQApp('Wi-Fi raw editor')
    original = wifi_project(WiFiSettings())
    dialog = WiFiSettingsDialog(original)
    try:
        dialog.source_combo.setCurrentIndex(dialog.source_combo.findData(WiFiPSDUSource.RAW_HEX))
        assert not dialog.fcs_check.isEnabled()
        dialog.raw_mode_combo.setCurrentIndex(1)
        assert dialog.fcs_check.isEnabled()
        dialog.fcs_check.setChecked(False)
        assert dialog.fcs_edit.isEnabled()
        dialog.fcs_edit.setText('11223344')
        dialog.reject()
        assert original.wifi == WiFiSettings()
        assert not hasattr(dialog,'project')
    finally:
        dialog.close()
