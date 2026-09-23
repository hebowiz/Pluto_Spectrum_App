import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')

import pyqtgraph as pg
import pytest
from dataclasses import replace
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
        assert dialog.field_pages.count()==6
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


def test_management_switching_defaults_and_user_fields_are_retained():
    pg.mkQApp('Wi-Fi management editor')
    dialog = WiFiSettingsDialog(wifi_project())
    try:
        dialog.source_address_edit.setText('02:AB:CD:EF:12:34')
        dialog.source_combo.setCurrentIndex(dialog.source_combo.findData(WiFiPSDUSource.PROBE_REQUEST))
        assert dialog.source_address_edit.isEnabled() and dialog.rates_edit.isEnabled()
        assert not dialog.timestamp_edit.isEnabled() and not dialog.tim_edit.isEnabled()
        assert not dialog.ds_auto_check.isEnabled() and not dialog.interval_period_button.isEnabled()
        assert dialog.source_address_edit.text() == '02:AB:CD:EF:12:34'
        assert '0x0040' in dialog.frame_control_label.text()
        assert not dialog.frame_control_spin.isEnabled()
        assert 'empty = wildcard' in dialog.ssid_hint.text()
        dialog.frame_control_mode.setCurrentIndex(1)
        dialog.frame_control_spin.setValue(0x1234)
        dialog.source_combo.setCurrentIndex(dialog.source_combo.findData(WiFiPSDUSource.PROBE_RESPONSE))
        assert dialog.frame_control_spin.value() == 0x1234
        assert dialog.timestamp_edit.isEnabled() and not dialog.tim_edit.isEnabled()
        assert dialog.ds_auto_check.isEnabled() and not dialog.interval_period_button.isEnabled()
        dialog.frame_control_mode.setCurrentIndex(0)
        assert '0x0050' in dialog.frame_control_label.text()
        dialog.frame_control_mode.setCurrentIndex(1)
        assert dialog.frame_control_spin.value() == 0x1234
        dialog.channel_combo.setCurrentIndex(10)
        dialog.name_edit.setText('My probe')
        dialog.defaults_button.click()
        assert dialog.destination_edit.text() == '02:11:22:33:44:66'
        assert dialog.source_address_edit.text() == dialog.bssid_edit.text() == '02:11:22:33:44:55'
        assert dialog.ds_channel_spin.value() == 11 and dialog.ds_auto_check.isChecked()
        assert dialog.extended_rates_edit.text() == 'B048606C'
        assert dialog.name_edit.text() == 'My probe'
        dialog.source_combo.setCurrentIndex(dialog.source_combo.findData(WiFiPSDUSource.PROBE_REQUEST))
        dialog.defaults_button.click()
        assert dialog.ssid_edit.text() == ''
        assert dialog.bssid_edit.text() == dialog.destination_edit.text() == 'FF:FF:FF:FF:FF:FF'
        assert dialog.source_address_edit.text() == '02:11:22:33:44:66'
        # Inactive invalid Beacon drafts must not prevent applying a Request.
        dialog.timestamp_edit.setText('invalid')
        dialog.tim_edit.setText('invalid')
        dialog.ssid_edit.setText('Target AP')
        dialog.additional_ies_edit.setText('DD050011220102')
        dialog._accept_settings()
        assert dialog.project.name == 'My probe'
        assert dialog.project.wifi.ssid == 'Target AP'
        assert dialog.project.wifi.additional_ies_hex == 'DD050011220102'
        assert 'Frame Type / body: Probe Request' in dialog.derived_label.text()
    finally:
        dialog.close()


@pytest.mark.parametrize('source', [WiFiPSDUSource.BEACON, WiFiPSDUSource.PROBE_REQUEST, WiFiPSDUSource.PROBE_RESPONSE])
def test_management_ui_manual_fcs_and_persistence(source):
    from pluto_vsg.persistence import project_from_dict, project_to_dict
    pg.mkQApp('Wi-Fi management persistence')
    original = replace(wifi_project(WiFiSettings(psdu_source=source)), name='Custom name')
    dialog = WiFiSettingsDialog(original)
    try:
        assert dialog.fcs_check.isEnabled()
        dialog.fcs_check.setChecked(False)
        assert dialog.fcs_edit.isEnabled()
        dialog.fcs_edit.setText('12345678')
        dialog.frame_control_mode.setCurrentIndex(1)
        dialog.frame_control_spin.setValue(0x4567)
        dialog._accept_settings()
        restored = project_from_dict(project_to_dict(dialog.project))
        assert restored.name == 'Custom name'
        assert restored.wifi.manual_fcs_hex == '12345678' and not restored.wifi.fcs_auto
        assert not restored.wifi.frame_control_auto and restored.wifi.frame_control == 0x4567
    finally:
        dialog.close()
