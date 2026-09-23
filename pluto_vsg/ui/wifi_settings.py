"""Non-HT packet editor using the common RF/Timing and Fields shell."""
from dataclasses import replace
import math

from pyqtgraph.Qt import QtCore, QtWidgets

from pluto_common.numeric_input import DeferredDoubleSpinBox, DeferredSpinBox, ensure_valid_numeric_inputs
from pluto_vsg.model import WiFiPSDUSource, WiFiScramblerSeedMode, WiFiSettings, validate_project, maximum_finite_repeat_count
from pluto_vsg.profiles.wifi import wifi_project
from pluto_vsg.wifi.common import LEGACY_RATES
from pluto_vsg.wifi.mac import build_psdu
from .packet_settings import carrier_selector, wifi_24ghz_carriers, packet_settings_tabs, packet_field_sections


def _integer(value, low, high, *, hexadecimal=False):
    box = DeferredSpinBox()
    box.setRange(low,high)
    box.setValue(value)
    if hexadecimal:
        box.setDisplayIntegerBase(16)
        box.setPrefix("0x")
    return box


class WiFiSettingsDialog(QtWidgets.QDialog):
    def __init__(self, project, parent=None):
        super().__init__(parent)
        if project.wifi is None:
            raise ValueError("Wi-Fi settings are required")
        self._project = project
        s = project.wifi
        self.setWindowTitle("Wi-Fi Packet / Waveform Settings")
        self.name_edit = QtWidgets.QLineEdit(project.name)
        self.rate_combo = QtWidgets.QComboBox()
        for rate in LEGACY_RATES:
            self.rate_combo.addItem(f"{rate} Mbps",rate)
        self.rate_combo.setCurrentIndex(self.rate_combo.findData(s.legacy_rate_mbps))
        self.sample_rate_combo = QtWidgets.QComboBox()
        for label,value in (("20 MS/s",1),("40 MS/s",2)):
            self.sample_rate_combo.addItem(label,value)
        self.sample_rate_combo.setCurrentIndex(self.sample_rate_combo.findData(s.oversample_factor))
        self.seed_mode_combo = QtWidgets.QComboBox()
        for mode in WiFiScramblerSeedMode:
            self.seed_mode_combo.addItem(mode.value,mode)
        self.seed_mode_combo.setCurrentIndex(self.seed_mode_combo.findData(s.scrambler_seed_mode))
        self.seed_spin = _integer(s.scrambler_seed,1,127,hexadecimal=True)
        nominal = (2407+5*s.channel)*1e6
        self.channel_combo = carrier_selector(wifi_24ghz_carriers(),nominal)
        self.frequency_offset_spin = DeferredDoubleSpinBox()
        self.frequency_offset_spin.setRange(-6e6,6e6)
        self.frequency_offset_spin.setDecimals(3)
        self.frequency_offset_spin.setSuffix(" kHz")
        self.frequency_offset_spin.setValue((project.center_frequency_hz-nominal)/1e3)
        self.center_label = QtWidgets.QLabel()
        self.period_spin = DeferredDoubleSpinBox()
        self.period_spin.setRange(1,10_000_000)
        self.period_spin.setDecimals(2)
        self.period_spin.setSuffix(" us")
        self.period_spin.setValue(s.packet_period_us)
        self.repeat_spin = _integer(project.repeat_count,1,maximum_finite_repeat_count(project))
        self.interval_period_button = QtWidgets.QPushButton("Use Beacon interval × 1024 us")
        self.interval_period_button.clicked.connect(lambda: self.period_spin.setValue(self.beacon_interval_spin.value()*1024))
        self.derived_label = QtWidgets.QLabel()
        self.derived_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.derived_label.setWordWrap(True)

        self.source_combo = QtWidgets.QComboBox()
        for source in WiFiPSDUSource:
            self.source_combo.addItem(source.value,source)
        self.source_combo.setCurrentIndex(self.source_combo.findData(s.psdu_source))
        self.raw_mode_combo = QtWidgets.QComboBox()
        self.raw_mode_combo.addItem("Raw PSDU including FCS (preserve bytes)",True)
        self.raw_mode_combo.addItem("Raw MAC frame without FCS (append Auto / Manual FCS)",False)
        self.raw_mode_combo.setCurrentIndex(self.raw_mode_combo.findData(s.raw_includes_fcs))
        self.raw_hex_edit = QtWidgets.QPlainTextEdit(s.raw_psdu_hex)
        self.raw_hex_edit.setMaximumHeight(90)
        self.length_spin = _integer(s.payload_length_bytes,1,4095)
        self.pattern_edit = QtWidgets.QLineEdit(s.payload_pattern_hex)
        self.ssid_edit = QtWidgets.QLineEdit(s.ssid)
        self.bssid_edit = QtWidgets.QLineEdit(s.bssid)
        self.destination_edit = QtWidgets.QLineEdit(s.destination_address)
        self.source_address_edit = QtWidgets.QLineEdit(s.source_address)
        self.source_address_edit.setPlaceholderText("Auto: same as BSSID")
        self.frame_control_spin = _integer(s.frame_control,0,65535,hexadecimal=True)
        self.duration_spin = _integer(s.duration_id,0,65535)
        self.sequence_spin = _integer(s.sequence_number,0,4095)
        self.fragment_spin = _integer(s.fragment_number,0,15)
        self.timestamp_edit = QtWidgets.QLineEdit(str(s.timestamp))
        self.beacon_interval_spin = _integer(s.beacon_interval_tu,1,65535)
        self.beacon_interval_spin.setSuffix(" TU")
        self.capability_spin = _integer(s.capability_information,0,65535,hexadecimal=True)
        self.rates_edit = QtWidgets.QLineEdit(s.supported_rates_hex)
        self.ds_auto_check = QtWidgets.QCheckBox("Auto: follow RF channel")
        self.ds_auto_check.setChecked(s.ds_channel_auto)
        self.ds_channel_spin = _integer(s.ds_channel,1,13)
        self.tim_edit = QtWidgets.QLineEdit(s.tim_hex)
        self.erp_spin = _integer(s.erp_information,0,255,hexadecimal=True)
        self.fcs_check = QtWidgets.QCheckBox("Auto: calculate IEEE CRC-32")
        self.fcs_check.setChecked(s.fcs_auto)
        self.fcs_edit = QtWidgets.QLineEdit(s.manual_fcs_hex)
        self.beacon_widgets = (self.ssid_edit,self.bssid_edit,self.destination_edit,self.source_address_edit,
            self.frame_control_spin,self.duration_spin,self.sequence_spin,self.fragment_spin,self.timestamp_edit,
            self.beacon_interval_spin,self.capability_spin,self.rates_edit,self.ds_auto_check,self.tim_edit,self.erp_spin)
        self.field_pages = packet_field_sections((
            ("Source / payload", (("Raw input meaning",self.raw_mode_combo),("Raw bytes [hex]",self.raw_hex_edit),
                ("Pattern / PRBS length [byte]",self.length_spin),("Pattern [hex]",self.pattern_edit),
                ("Pattern / PRBS",QtWidgets.QLabel("Exact synthetic PSDU bytes; no MAC header or FCS is added.")))),
            ("Beacon MAC header", (("Frame Control",self.frame_control_spin),("Duration / ID",self.duration_spin),
                ("Destination",self.destination_edit),("Source",self.source_address_edit),("BSSID",self.bssid_edit),
                ("Sequence Number (static)",self.sequence_spin),("Fragment Number",self.fragment_spin))),
            ("Beacon fixed fields / IEs", (("Timestamp (static, microseconds)",self.timestamp_edit),
                ("Beacon Interval",self.beacon_interval_spin),("Capability Information",self.capability_spin),
                ("SSID",self.ssid_edit),("Supported Rates [hex octets]",self.rates_edit),
                ("DS Parameter Set",self.ds_auto_check),("Manual DS channel",self.ds_channel_spin),
                ("TIM body [hex]",self.tim_edit),("ERP Information",self.erp_spin))),
            ("FCS", (("FCS mode",self.fcs_check),("Manual FCS [4 transmitted hex octets]",self.fcs_edit),
                ("L-SIG",QtWidgets.QLabel("LENGTH and parity: Auto from final PSDU. Reserved and tail: zero.")))),
        ))
        self.tabs = packet_settings_tabs((
            ("PHY Format",QtWidgets.QLabel("Non-HT OFDM / ERP-OFDM")),
            ("Bandwidth",QtWidgets.QLabel("20 MHz")),("Data Rate",self.rate_combo),
            ("Sample Rate",self.sample_rate_combo),("Channel",self.channel_combo),
            ("Frequency Offset",self.frequency_offset_spin),("Generated RF Frequency",self.center_label),
            ("Packet Period",self.period_spin),("Beacon period helper",self.interval_period_button),
            ("Repeat Count",self.repeat_spin),("Scrambler",self.seed_mode_combo),("Fixed Seed",self.seed_spin),
            ("Envelope",QtWidgets.QLabel("Common ramp disabled; OFDM uses cyclic prefixes, no optional overlap window.")),
            ("Derived timing",self.derived_label)),
            (("Project Name",self.name_edit),("PSDU Source",self.source_combo),("Field groups",self.field_pages)))
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText("Apply and Generate")
        buttons.accepted.connect(self._accept_settings)
        buttons.rejected.connect(self.reject)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.tabs)
        layout.addWidget(buttons)
        for widget in self.findChildren(QtWidgets.QComboBox):
            widget.currentIndexChanged.connect(self._refresh)
        for widget in self.findChildren(QtWidgets.QAbstractSpinBox):
            widget.valueChanged.connect(self._refresh)
        for widget in self.findChildren(QtWidgets.QLineEdit):
            widget.textChanged.connect(self._refresh)
        for widget in self.findChildren(QtWidgets.QCheckBox):
            widget.toggled.connect(self._refresh)
        self.raw_hex_edit.textChanged.connect(self._refresh)
        self.resize(860,760)
        self._refresh()

    def _settings(self) -> WiFiSettings:
        channel = round((float(self.channel_combo.currentData())/1e6-2407)/5)
        return replace(self._project.wifi,
            legacy_rate_mbps=int(self.rate_combo.currentData()),oversample_factor=int(self.sample_rate_combo.currentData()),
            scrambler_seed_mode=WiFiScramblerSeedMode(self.seed_mode_combo.currentData()),scrambler_seed=self.seed_spin.value(),
            channel=channel,packet_period_us=self.period_spin.value(),psdu_source=WiFiPSDUSource(self.source_combo.currentData()),
            raw_psdu_hex=self.raw_hex_edit.toPlainText(),raw_includes_fcs=bool(self.raw_mode_combo.currentData()),
            payload_length_bytes=self.length_spin.value(),payload_pattern_hex=self.pattern_edit.text(),
            ssid=self.ssid_edit.text(),bssid=self.bssid_edit.text(),destination_address=self.destination_edit.text(),
            source_address=self.source_address_edit.text(),frame_control=self.frame_control_spin.value(),duration_id=self.duration_spin.value(),
            sequence_number=self.sequence_spin.value(),fragment_number=self.fragment_spin.value(),timestamp=int(self.timestamp_edit.text(),0) if self.timestamp_edit.text().startswith('0x') else int(self.timestamp_edit.text()),
            beacon_interval_tu=self.beacon_interval_spin.value(),capability_information=self.capability_spin.value(),
            supported_rates_hex=self.rates_edit.text(),ds_channel_auto=self.ds_auto_check.isChecked(),ds_channel=self.ds_channel_spin.value(),
            tim_hex=self.tim_edit.text(),erp_information=self.erp_spin.value(),fcs_auto=self.fcs_check.isChecked(),manual_fcs_hex=self.fcs_edit.text())

    def _refresh(self, *_args):
        nominal = float(self.channel_combo.currentData())
        actual = nominal+self.frequency_offset_spin.value()*1e3
        self.center_label.setText(f"{actual/1e6:.6f} MHz")
        source = self.source_combo.currentData()
        beacon = source == WiFiPSDUSource.BEACON
        raw = source == WiFiPSDUSource.RAW_HEX
        self.raw_mode_combo.setEnabled(raw)
        self.raw_hex_edit.setEnabled(raw)
        self.length_spin.setEnabled(source in (WiFiPSDUSource.PATTERN,WiFiPSDUSource.PRBS9))
        self.pattern_edit.setEnabled(source == WiFiPSDUSource.PATTERN)
        self.seed_spin.setEnabled(self.seed_mode_combo.currentData() == WiFiScramblerSeedMode.FIXED)
        for widget in self.beacon_widgets:
            widget.setEnabled(beacon)
        self.ds_channel_spin.setEnabled(beacon and not self.ds_auto_check.isChecked())
        self.interval_period_button.setEnabled(beacon)
        append = beacon or (raw and not self.raw_mode_combo.currentData())
        self.fcs_check.setEnabled(append)
        self.fcs_edit.setEnabled(append and not self.fcs_check.isChecked())
        try:
            s = self._settings()
            rate = LEGACY_RATES[s.legacy_rate_mbps]
            length = len(build_psdu(s))
            count = math.ceil((16+8*length+6)/rate.n_dbps)
            duration = 20+4*count
            errors = validate_project(wifi_project(s))
            text = (f"PSDU: {length} bytes | {rate.modulation}, {rate.coding_rate}\n"
                f"N_BPSC / N_CBPS / N_DBPS: {rate.n_bpsc} / {rate.n_cbps} / {rate.n_dbps}\n"
                f"N_SYM / N_PAD: {count} / {count*rate.n_dbps-(16+8*length+6)} | L-SIG LENGTH / parity: Auto\n"
                f"PPDU Duration: {duration} us | Duty Cycle: {100*duration/s.packet_period_us:.3f} %\n"
                f"Signal Extension: 6 us (no transmission)\nMinimum Packet Period: {duration+6} us\n"
                f"Configured Packet Period: {s.packet_period_us:g} us")
            if beacon:
                text += f"\nBeacon Interval: {s.beacon_interval_tu*1024:g} us; Timestamp / Sequence: static"
                if not math.isclose(s.packet_period_us,s.beacon_interval_tu*1024):
                    text += "\nNotice: packet period differs from advertised Beacon Interval."
            if errors:
                text += "\n" + "\n".join(i.message for i in errors)
            self.derived_label.setText(text)
        except (ValueError,OverflowError,KeyError) as error:
            self.derived_label.setText(str(error))

    def _accept_settings(self):
        if not ensure_valid_numeric_inputs(self,title="Invalid Wi-Fi Setting"):
            return
        try:
            candidate = wifi_project(self._settings())
            candidate = replace(candidate,name=self.name_edit.text().strip(),repeat_count=self.repeat_spin.value(),
                center_frequency_hz=float(self.channel_combo.currentData())+self.frequency_offset_spin.value()*1e3)
            issues = validate_project(candidate)
            if issues:
                raise ValueError("\n".join(i.message for i in issues))
        except (ValueError,OverflowError,KeyError) as error:
            QtWidgets.QMessageBox.warning(self,"Wi-Fi Settings",str(error))
            return
        self.project = candidate
        self.accept()
