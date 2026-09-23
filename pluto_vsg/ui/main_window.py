"""Visual Composer shell and first Bluetooth BR vertical slice."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import re

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from pluto_common import short_pluto_identity
from pluto_common.control_panel import (
    CONTROL_BUTTON_FONT_SCALE,
    CONTROL_PANEL_WIDTH,
    CONTROL_VALUE_BUTTON_HEIGHT,
    ControlPanelNavigator,
    add_back_button_footer,
    configure_control_button,
    make_control_group,
)
from pluto_common.numeric_input import (
    DeferredDoubleSpinBox,
    DeferredSpinBox,
    ensure_valid_numeric_inputs,
    get_deferred_double,
    get_deferred_int,
)
from pluto_common.runtime_paths import diagnostic_log_path
from pluto_common.window_geometry import restore_window_geometry, save_window_geometry

from pluto_vsa.profiles.bluetooth_br import header_error_check
from pluto_vsa.ui.measurement_chrome import (
    install_measurement_plot_menu,
    make_measurement_plot,
    make_measurement_dock,
)
from pluto_vsa.ui.packet_decode import PacketDecodeTabs, apply_analysis_font
from pluto_vsg.protocol import analyze_generation_result
from pluto_vsg.backends import (
    PlutoOutputBackend,
    PlutoPlaybackMode,
    PlutoTransmitSettings,
    estimate_pluto_output_power_dbm,
    pluto_hardware_gain_for_output_power_dbm,
    pluto_output_power_range_dbm,
)
from pluto_vsg.composer import ComposerBlock, build_composer_graph
from pluto_vsg.engine import (
    BluetoothBRWaveformEngine,
    BluetoothHDTWaveformEngine,
    BluetoothLEWaveformEngine,
    DectWaveformEngine,
    GenerationResult,
    WiFiLegacyOFDMWaveformEngine,
)
from pluto_vsg.export import save_iq_tar, save_npz, save_wv
from pluto_vsg.model import (
    BluetoothHDTSettings,
    BluetoothLEPayloadType,
    BluetoothLEPayloadSourceKind,
    BluetoothLEPhy,
    BluetoothPacketKind,
    DectBFieldSource,
    DectPacketType,
    PayloadSourceKind,
    HDTPayloadSourceKind,
    hdt_is_rf_test_configuration,
    StandardProfile,
    WaveformProject,
    bluetooth_packet_is_edr,
    bluetooth_packet_properties,
    effective_post_idle_symbols,
    effective_period_symbols,
    maximum_finite_repeat_count,
    minimum_period_symbols,
    validate_project,
    WiFiPSDUSource,
    WiFiScramblerSeedMode,
    WiFiSettings,
)
from pluto_vsg.persistence import (
    load_project,
    project_from_dict,
    project_to_dict,
    save_project,
)
from pluto_vsg.rf_level import generation_result_iq_levels
from pluto_vsg.profiles import (
    bluetooth_br_edr_project,
    bluetooth_br_fields,
    bluetooth_le_fields,
    bluetooth_le_project,
    bluetooth_le_test_project,
    apply_bluetooth_le_rf_test_preset,
    bluetooth_hdt_fields,
    bluetooth_hdt_project,
    wifi_beacon_project,
    wifi_fields,
    wifi_project,
    dect_fields,
    dect_project,
)
from pluto_protocol.bluetooth.hdt import HDTRate, hdt_definition, hdt_rf_test_control_bits, hdt_rf_test_format0_bits
from pluto_vsg.engine.bluetooth_hdt import hdt_payload_bits
from pluto_vsg.ui.style import (
    ACCENT_COLOR,
    FIELD_BOUNDARY_COLOR,
    FIELD_MINOR_BOUNDARY_COLOR,
    PACKET_END_COLOR,
    TRACE_COLOR,
    panel_title_font,
)
from pluto_vsg.ui.composer_view import PacketComposerView
from pluto_vsg.ui.dect_settings import DectSettingsDialog
from pluto_vsg.ui.frequency_settings import (
    FrequencySettingsDialog,
    default_frequency_selection,
    effective_rf_frequency_hz,
    with_manual_rf_frequency,
)


_STARTUP_STATE_SCHEMA = "pluto-vsg-startup-state"
_STARTUP_STATE_VERSION = 1
_STARTUP_STATE_KEY = "startup/state"
_STARTUP_WINDOW_STATE_KEY = "startup/window_state"
from pluto_vsg.ui.packet_settings import (
    SymbolTimeControl,
    bluetooth_classic_carriers,
    bluetooth_le_carriers,
    carrier_selector,
    packet_settings_tabs,
    wifi_24ghz_carriers,
)


def _instantaneous_frequency_khz(
    iq: np.ndarray, sample_rate_hz: float, *, active_threshold: float = 1e-5
) -> np.ndarray:
    """Return phase-difference frequency, leaving RF-off samples undefined."""

    values = np.asarray(iq, dtype=np.complex128)
    if values.size < 2:
        return np.empty(0, dtype=np.float64)
    frequency = (
        np.angle(values[1:] * np.conj(values[:-1]))
        * float(sample_rate_hz)
        / (2.0 * np.pi * 1e3)
    )
    active = (np.abs(values[1:]) > active_threshold) & (
        np.abs(values[:-1]) > active_threshold
    )
    frequency[~active] = np.nan
    return frequency


def _preview_active_x_range_us(
    result: GenerationResult,
    preview_sample_count: int,
    *,
    total_margin_fraction: float = 0.10,
) -> tuple[float, float]:
    """Return a first-repeat view covering Active Window plus <=10% margin."""

    count = max(1, int(preview_sample_count))
    ranges = tuple(result.metadata.get("active_ranges_samples", ()))
    if ranges:
        start_sample, stop_sample = map(int, ranges[0])
    else:
        packet_ranges = tuple(result.metadata.get("packet_ranges_samples", ()))
        if packet_ranges:
            start_sample, stop_sample = map(int, packet_ranges[0])
        else:
            start_sample, stop_sample = 0, count
    start_sample = min(count, max(0, start_sample))
    stop_sample = min(count, max(start_sample + 1, stop_sample))
    active_span = max(1, stop_sample - start_sample)
    side_margin = 0.5 * max(0.0, float(total_margin_fraction)) * active_span
    view_start = max(0.0, start_sample - side_margin)
    view_stop = min(float(count), stop_sample + side_margin)
    scale = 1e6 / float(result.sample_rate_hz)
    return view_start * scale, view_stop * scale


def _cw_generation_result(
    sample_rate_hz: float, sample_count: int = 4096
) -> GenerationResult:
    """Build a normalized zero-IF period for continuous CW playback."""

    rate_hz = float(sample_rate_hz)
    count = int(sample_count)
    if not np.isfinite(rate_hz) or rate_hz <= 0.0:
        raise ValueError("CW sample rate must be positive and finite")
    if count < 4:
        raise ValueError("CW period must contain at least four samples")
    return GenerationResult(
        iq=np.ones(count, dtype=np.complex64),
        sample_rate_hz=rate_hz,
        metadata={"waveform_kind": "CW", "baseband_frequency_hz": 0.0},
    )


class _Panel(QtWidgets.QGroupBox):
    TITLE_LEFT_INSET_PX = 8

    def __init__(self, title: str, child: QtWidgets.QWidget) -> None:
        super().__init__(title)
        self.setObjectName("vsgWorkspacePanel")
        self.setStyleSheet(
            "QGroupBox#vsgWorkspacePanel::title { "
            "subcontrol-origin: margin; "
            f"left: {self.TITLE_LEFT_INSET_PX}px; "
            "padding: 0 1px; }"
        )
        self.setFont(panel_title_font(self.font()))
        child_font = QtGui.QFont(child.font())
        child_font.setBold(False)
        child.setFont(child_font)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 8, 4, 4)
        layout.addWidget(child)


class _RFStateButton(QtWidgets.QPushButton):
    """Checkable RF indicator whose state is controlled only by hardware events."""

    def nextCheckState(self) -> None:
        # QAbstractButton normally toggles before emitting clicked(). That can
        # paint a false blue ON state while the click handler decides whether
        # calibration is required. Programmatic setChecked() remains available.
        return


class _WiFiSettingsDialog(QtWidgets.QDialog):
    """Dedicated Non-HT OFDM packet, RF and Beacon editor."""

    def __init__(self, project: WaveformProject, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        if project.wifi is None:
            raise ValueError("Wi-Fi settings are required")
        self._project = project
        settings = project.wifi
        self.setWindowTitle("Wi-Fi Packet / Waveform Settings")
        self.name_edit = QtWidgets.QLineEdit(project.name)
        self.rate_combo = QtWidgets.QComboBox()
        for rate in (6, 9, 12, 18, 24, 36, 48, 54):
            self.rate_combo.addItem(f"{rate} Mbps", rate)
        self.rate_combo.setCurrentIndex(self.rate_combo.findData(settings.legacy_rate_mbps))
        self.sample_rate_combo = QtWidgets.QComboBox()
        self.sample_rate_combo.addItem("20 MS/s (native)", 1)
        self.sample_rate_combo.addItem("40 MS/s (2x oversampled)", 2)
        self.sample_rate_combo.setCurrentIndex(self.sample_rate_combo.findData(settings.oversample_factor))
        self.seed_mode_combo = QtWidgets.QComboBox()
        for mode in WiFiScramblerSeedMode:
            self.seed_mode_combo.addItem(mode.value, mode)
        self.seed_mode_combo.setCurrentIndex(self.seed_mode_combo.findData(WiFiScramblerSeedMode(settings.scrambler_seed_mode)))
        self.seed_spin = DeferredSpinBox(); self.seed_spin.setRange(1, 127); self.seed_spin.setValue(settings.scrambler_seed)
        self.seed_spin.setDisplayIntegerBase(16); self.seed_spin.setPrefix("0x")
        nominal_wifi_hz = (2407 + 5 * int(settings.channel)) * 1e6
        self.channel_combo = carrier_selector(
            wifi_24ghz_carriers(), nominal_wifi_hz
        )
        self.frequency_offset_spin = DeferredDoubleSpinBox()
        self.frequency_offset_spin.setRange(-3000.0, 3000.0)
        self.frequency_offset_spin.setDecimals(3)
        self.frequency_offset_spin.setSuffix(" kHz")
        self.frequency_offset_spin.setValue(
            (project.center_frequency_hz - nominal_wifi_hz) / 1e3
        )
        self.center_label = QtWidgets.QLabel()
        self.source_combo = QtWidgets.QComboBox()
        for source in WiFiPSDUSource:
            self.source_combo.addItem(source.value, source)
        self.source_combo.setCurrentIndex(self.source_combo.findData(WiFiPSDUSource(settings.psdu_source)))
        self.raw_hex_edit = QtWidgets.QPlainTextEdit(settings.raw_psdu_hex); self.raw_hex_edit.setMaximumHeight(100)
        self.length_spin = DeferredSpinBox(); self.length_spin.setRange(1, 4095); self.length_spin.setValue(settings.payload_length_bytes)
        self.pattern_edit = QtWidgets.QLineEdit(settings.payload_pattern_hex)
        self.period_spin = DeferredDoubleSpinBox(); self.period_spin.setRange(1.0, 10_000_000.0); self.period_spin.setDecimals(1); self.period_spin.setValue(settings.packet_period_us); self.period_spin.setSuffix(" us")
        self.ssid_edit = QtWidgets.QLineEdit(settings.ssid)
        self.bssid_edit = QtWidgets.QLineEdit(settings.bssid)
        self.sequence_spin = DeferredSpinBox(); self.sequence_spin.setRange(0, 4095); self.sequence_spin.setValue(settings.sequence_number)
        self.beacon_interval_spin = DeferredSpinBox(); self.beacon_interval_spin.setRange(1, 65535); self.beacon_interval_spin.setValue(settings.beacon_interval_tu); self.beacon_interval_spin.setSuffix(" TU")
        self.fcs_check = QtWidgets.QCheckBox("Append IEEE 802.11 FCS automatically"); self.fcs_check.setChecked(settings.fcs_auto)
        self.derived_label = QtWidgets.QLabel(); self.derived_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)

        self.tabs = packet_settings_tabs(
            (
                ("Format", QtWidgets.QLabel("Non-HT OFDM (802.11a/g)")),
                ("Bandwidth", QtWidgets.QLabel("20 MHz")),
                ("Data Rate / Modulation", self.rate_combo),
                ("Sample Rate", self.sample_rate_combo),
                ("Pattern / PRBS Length [byte]", self.length_spin),
                ("Packet Period", self.period_spin),
                ("Ramp", QtWidgets.QLabel("Disabled (automatic OFDM packet boundary)")),
                ("Calculated PHY values", self.derived_label),
            ),
            (
                ("Project Name", self.name_edit),
                ("Scrambler Seed", self.seed_mode_combo),
                ("Fixed Seed", self.seed_spin),
                ("Frame Source", self.source_combo),
                ("Raw PSDU [hex]", self.raw_hex_edit),
                ("Pattern [hex]", self.pattern_edit),
                ("SSID", self.ssid_edit),
                ("BSSID", self.bssid_edit),
                ("Sequence Number", self.sequence_spin),
                ("Beacon Interval", self.beacon_interval_spin),
                ("FCS", self.fcs_check),
            ),
        )
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText("Apply and Generate")
        buttons.accepted.connect(self._accept_settings); buttons.rejected.connect(self.reject)
        layout = QtWidgets.QVBoxLayout(self); layout.addWidget(self.tabs); layout.addWidget(buttons)
        for signal in (self.rate_combo.currentIndexChanged, self.sample_rate_combo.currentIndexChanged, self.seed_mode_combo.currentIndexChanged, self.channel_combo.currentIndexChanged, self.frequency_offset_spin.valueChanged, self.source_combo.currentIndexChanged, self.length_spin.valueChanged, self.period_spin.valueChanged, self.beacon_interval_spin.valueChanged):
            signal.connect(self._refresh)
        self.raw_hex_edit.textChanged.connect(self._refresh); self.ssid_edit.textChanged.connect(self._refresh); self.bssid_edit.textChanged.connect(self._refresh)
        self.resize(860, 760); self._refresh()

    def _settings(self) -> WiFiSettings:
        nominal_hz = float(self.channel_combo.currentData())
        channel = int(round((nominal_hz / 1e6 - 2407.0) / 5.0))
        return replace(self._project.wifi, legacy_rate_mbps=int(self.rate_combo.currentData()), oversample_factor=int(self.sample_rate_combo.currentData()), scrambler_seed_mode=WiFiScramblerSeedMode(self.seed_mode_combo.currentData()), scrambler_seed=self.seed_spin.value(), psdu_source=WiFiPSDUSource(self.source_combo.currentData()), raw_psdu_hex=self.raw_hex_edit.toPlainText(), payload_length_bytes=self.length_spin.value(), payload_pattern_hex=self.pattern_edit.text(), channel=channel, ssid=self.ssid_edit.text(), bssid=self.bssid_edit.text(), sequence_number=self.sequence_spin.value(), beacon_interval_tu=self.beacon_interval_spin.value(), fcs_auto=self.fcs_check.isChecked(), packet_period_us=self.period_spin.value())

    def _refresh(self, _value=None) -> None:
        nominal_hz = float(self.channel_combo.currentData())
        actual_hz = nominal_hz + self.frequency_offset_spin.value() * 1e3
        self.center_label.setText(
            f"{actual_hz / 1e6:.6f} MHz = {nominal_hz / 1e6:.3f} MHz "
            f"{self.frequency_offset_spin.value():+.3f} kHz"
        )
        source = WiFiPSDUSource(self.source_combo.currentData())
        self.raw_hex_edit.setEnabled(source == WiFiPSDUSource.RAW_HEX)
        self.length_spin.setEnabled(source in {WiFiPSDUSource.PATTERN, WiFiPSDUSource.PRBS9})
        self.pattern_edit.setEnabled(source == WiFiPSDUSource.PATTERN)
        for widget in (self.ssid_edit, self.bssid_edit, self.sequence_spin, self.beacon_interval_spin, self.fcs_check): widget.setEnabled(source == WiFiPSDUSource.BEACON)
        self.seed_spin.setEnabled(WiFiScramblerSeedMode(self.seed_mode_combo.currentData()) == WiFiScramblerSeedMode.FIXED)
        try:
            candidate = wifi_project(self._settings())
            from pluto_vsg.wifi.common import LEGACY_RATES
            from pluto_vsg.wifi.mac import build_psdu
            rate = LEGACY_RATES[int(candidate.wifi.legacy_rate_mbps)]; length = len(build_psdu(candidate.wifi))
            n_sym = int(np.ceil((16 + 8 * length + 6) / rate.n_dbps)); n_pad = n_sym * rate.n_dbps - (16 + 8 * length + 6); duration = 20 + 4 * n_sym
            duty = 100 * duration / self.period_spin.value()
            self.derived_label.setText(f"PSDU: {length} byte\nModulation: {rate.modulation}\nCoding Rate: {rate.coding_rate}\nN_BPSC / N_CBPS / N_DBPS: {rate.n_bpsc} / {rate.n_cbps} / {rate.n_dbps}\nN_SYM / N_PAD: {n_sym} / {n_pad}\nPPDU Duration: {duration} us\nDuty Cycle: {duty:.3f} %")
        except ValueError as error:
            self.derived_label.setText(str(error))

    def _accept_settings(self) -> None:
        if not ensure_valid_numeric_inputs(self, title="Invalid Wi-Fi Setting"):
            return
        try:
            settings = self._settings(); candidate = wifi_project(settings)
            candidate = replace(
                candidate,
                center_frequency_hz=(
                    float(self.channel_combo.currentData())
                    + self.frequency_offset_spin.value() * 1e3
                ),
            )
            candidate = replace(candidate, name=self.name_edit.text().strip(), repeat_count=self._project.repeat_count)
            issues = validate_project(candidate)
            if issues: raise ValueError("\n".join(issue.message for issue in issues))
        except ValueError as error:
            QtWidgets.QMessageBox.warning(self, "Wi-Fi Settings", str(error)); return
        self.project = candidate; self.accept()


class _BluetoothLESettingsDialog(QtWidgets.QDialog):
    """Edit an uncoded LE Direct Test Mode packet."""

    def __init__(self, project: WaveformProject, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        settings = project.bluetooth_le
        if settings is None:
            raise ValueError("Bluetooth LE settings are required")
        self._project = project
        self.setWindowTitle("Bluetooth LE Packet Settings")
        self.name_edit = QtWidgets.QLineEdit(project.name)
        self.phy_combo = QtWidgets.QComboBox()
        for phy in BluetoothLEPhy:
            self.phy_combo.addItem(phy.value, phy)
        self.phy_combo.setCurrentIndex(self.phy_combo.findData(BluetoothLEPhy(settings.phy)))
        self.payload_combo = QtWidgets.QComboBox()
        for payload_type in BluetoothLEPayloadType:
            self.payload_combo.addItem(payload_type.value, payload_type)
        self.payload_combo.setCurrentIndex(
            self.payload_combo.findData(BluetoothLEPayloadType(settings.payload_type))
        )
        self.apply_rf_test_button = QtWidgets.QPushButton(
            "Apply RF Test Packet Preset"
        )
        self.apply_rf_test_button.clicked.connect(self._apply_rf_test_preset)
        preset_row = QtWidgets.QWidget()
        preset_layout = QtWidgets.QHBoxLayout(preset_row)
        preset_layout.setContentsMargins(0, 0, 0, 0)
        preset_layout.addWidget(self.payload_combo)
        preset_layout.addWidget(self.apply_rf_test_button)
        self.preamble_edit = QtWidgets.QLineEdit(settings.preamble_bits)
        self.sync_edit = QtWidgets.QLineEdit(settings.sync_word_bits)
        self.header_edit = QtWidgets.QLineEdit(settings.pdu_header_bits)
        self.payload_source_combo = QtWidgets.QComboBox()
        for source in BluetoothLEPayloadSourceKind:
            self.payload_source_combo.addItem(source.value, source)
        self.payload_source_combo.setCurrentIndex(
            self.payload_source_combo.findData(
                BluetoothLEPayloadSourceKind(settings.payload_source)
            )
        )
        self.payload_pattern_edit = QtWidgets.QLineEdit(settings.payload_pattern)
        self.length_spin = DeferredSpinBox()
        self.length_spin.setRange(0, 255)
        self.length_spin.setValue(settings.payload_length_bytes)
        self.crc_check = QtWidgets.QCheckBox()
        self.crc_check.setChecked(settings.crc_enabled)
        self.crc_init_edit = QtWidgets.QLineEdit(f"{settings.crc_init:06X}")
        self.whitening_check = QtWidgets.QCheckBox()
        self.whitening_check.setChecked(settings.whitening_enabled)
        self.whitening_channel_spin = self._integer_spin(
            0, 39, settings.whitening_channel_index
        )
        le_carriers = bluetooth_le_carriers()
        le_nominal_hz = min(
            (frequency for _label, frequency in le_carriers),
            key=lambda frequency: abs(frequency - project.center_frequency_hz),
        )
        self.carrier_combo = carrier_selector(le_carriers, le_nominal_hz)
        self.frequency_offset_spin = self._double_spin(
            -3000.0, 3000.0,
            (project.center_frequency_hz - float(self.carrier_combo.currentData())) / 1e3,
            3,
        )
        self.frequency_offset_spin.setSuffix(" kHz")
        self.actual_frequency_label = QtWidgets.QLabel()
        self.sps_spin = self._integer_spin(4, 64, project.samples_per_symbol)
        self.sample_rate_label = QtWidgets.QLabel()
        self.packet_duration_label = QtWidgets.QLabel()
        self.deviation_spin = self._double_spin(1.0, 2000.0, settings.frequency_deviation_hz / 1e3, 3)
        self.bt_spin = self._double_spin(0.05, 2.0, settings.gaussian_bt, 3)
        self.pre_idle_spin = self._integer_spin(0, 1_000_000, settings.pre_idle_symbols)
        self._minimum_period_symbols = minimum_period_symbols(project)
        self.period_spin = self._double_spin(
            0.0, 1_000_000.0, effective_period_symbols(project), 3
        )
        self.post_idle_value = QtWidgets.QLabel()
        self.period_spin.valueChanged.connect(self._update_post_idle_reference)
        self._update_post_idle_reference()
        self.rise_spin = self._double_spin(0.0, 1000.0, project.power_envelope.rise_symbols, 3)
        self.rise_delay_spin = self._double_spin(-1000.0, 1000.0, project.power_envelope.rise_delay_symbols, 3)
        self.fall_spin = self._double_spin(0.0, 1000.0, project.power_envelope.fall_symbols, 3)
        self.fall_delay_spin = self._double_spin(-1000.0, 1000.0, project.power_envelope.fall_delay_symbols, 3)
        self.ramp_combo = QtWidgets.QComboBox()
        self.ramp_combo.addItems(["Cosine", "Linear"])
        self.ramp_combo.setCurrentText(project.power_envelope.shape)
        self._timing_controls = tuple(
            SymbolTimeControl(control, self._symbol_rate_hz)
            for control in (
                self.pre_idle_spin,
                self.period_spin,
                self.rise_spin,
                self.rise_delay_spin,
                self.fall_spin,
                self.fall_delay_spin,
            )
        )
        self.tabs = packet_settings_tabs(
            (
            ("PHY", self.phy_combo),
            ("Modulation", QtWidgets.QLabel("GFSK")),
            ("Payload Length [byte]", self.length_spin),
            ("Packet Length", self.packet_duration_label),
            ("Samples / Symbol", self.sps_spin),
            ("Sample Rate", self.sample_rate_label),
            ("FSK Deviation [kHz]", self.deviation_spin),
            ("Gaussian B*T", self.bt_spin),
            ("Pre Idle", self._timing_controls[0]),
            ("Packet Period", self._timing_controls[1]),
            ("Derived Post Idle", self.post_idle_value),
            ("Ramp Up", self._timing_controls[2]),
            ("Ramp Up Start rel. Packet", self._timing_controls[3]),
            ("Ramp Down", self._timing_controls[4]),
            ("Ramp Down Start rel. Packet End", self._timing_controls[5]),
            ("Ramp Shape", self.ramp_combo),
            ),
            (
            ("Project Name", self.name_edit),
            ("RF Test Payload Preset", preset_row),
            ("Preamble [air-order bits]", self.preamble_edit),
            ("Access Address / Sync [air-order bits]", self.sync_edit),
            ("PDU Header [air-order bits]", self.header_edit),
            ("Payload Source", self.payload_source_combo),
            ("Payload Pattern [bin]", self.payload_pattern_edit),
            ("CRC-24", self.crc_check),
            ("CRCInit [hex]", self.crc_init_edit),
            ("Whitening", self.whitening_check),
            ("Whitening Channel Index", self.whitening_channel_spin),
            ),
        )
        note = QtWidgets.QLabel(
            "All packet fields remain editable. Applying an RF Test Packet preset "
            "loads the Core test Sync Word/header/payload, CRCInit 0x555555, "
            "Whitening Off and the standard packet interval into these controls."
        )
        note.setWordWrap(True)
        fields_page = self.tabs.widget(1).widget()
        fields_page.layout().addRow(note)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText(
            "Apply and Generate"
        )
        buttons.accepted.connect(self._accept_settings)
        buttons.rejected.connect(self.reject)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.tabs)
        layout.addWidget(buttons)
        self.resize(860, 760)
        self.phy_combo.currentIndexChanged.connect(self._phy_changed)
        self.carrier_combo.currentIndexChanged.connect(self._update_rf_preview)
        self.frequency_offset_spin.valueChanged.connect(self._update_rf_preview)
        self.payload_source_combo.currentIndexChanged.connect(
            self._payload_source_changed
        )
        for signal in (
            self.phy_combo.currentIndexChanged,
            self.length_spin.valueChanged,
            self.sps_spin.valueChanged,
            self.pre_idle_spin.valueChanged,
            self.rise_spin.valueChanged,
            self.rise_delay_spin.valueChanged,
            self.fall_spin.valueChanged,
            self.fall_delay_spin.valueChanged,
        ):
            signal.connect(self._update_period_constraints)
        self._update_period_constraints()
        self._payload_source_changed()

    @staticmethod
    def _double_spin(minimum: float, maximum: float, value: float, decimals: int) -> QtWidgets.QDoubleSpinBox:
        control = DeferredDoubleSpinBox()
        control.setRange(minimum, maximum)
        control.setDecimals(decimals)
        control.setValue(value)
        control.setKeyboardTracking(False)
        return control

    @staticmethod
    def _integer_spin(minimum: int, maximum: int, value: int) -> QtWidgets.QSpinBox:
        control = DeferredSpinBox()
        control.setRange(minimum, maximum)
        control.setValue(value)
        return control

    def _phy_changed(self) -> None:
        phy = BluetoothLEPhy(self.phy_combo.currentData())
        self.deviation_spin.setValue(250.0 if phy == BluetoothLEPhy.LE_1M else 500.0)
        self._update_rf_preview()

    def _symbol_rate_hz(self) -> float:
        return (
            1_000_000.0
            if BluetoothLEPhy(self.phy_combo.currentData()) == BluetoothLEPhy.LE_1M
            else 2_000_000.0
        )

    def _update_rf_preview(self, _value=None) -> None:
        nominal_hz = float(self.carrier_combo.currentData())
        actual_hz = nominal_hz + self.frequency_offset_spin.value() * 1e3
        self.actual_frequency_label.setText(
            f"{actual_hz / 1e6:.6f} MHz = {nominal_hz / 1e6:.3f} MHz "
            f"{self.frequency_offset_spin.value():+.3f} kHz"
        )
        self.sample_rate_label.setText(
            f"{self._symbol_rate_hz() * self.sps_spin.value() / 1e6:.3f} MS/s"
        )
        for control in self._timing_controls:
            control.refresh()

    def _payload_source_changed(self) -> None:
        source = BluetoothLEPayloadSourceKind(self.payload_source_combo.currentData())
        self.payload_pattern_edit.setEnabled(
            source
            in {
                BluetoothLEPayloadSourceKind.FIXED,
                BluetoothLEPayloadSourceKind.PATTERN,
            }
        )

    def _apply_rf_test_preset(self) -> None:
        phy = BluetoothLEPhy(self.phy_combo.currentData())
        payload_type = BluetoothLEPayloadType(self.payload_combo.currentData())
        settings = apply_bluetooth_le_rf_test_preset(
            self._project.bluetooth_le,
            phy=phy,
            payload_type=payload_type,
            payload_length_bytes=self.length_spin.value(),
        )
        self.preamble_edit.setText(settings.preamble_bits)
        self.sync_edit.setText(settings.sync_word_bits)
        self.header_edit.setText(settings.pdu_header_bits)
        self.payload_source_combo.setCurrentIndex(
            self.payload_source_combo.findData(settings.payload_source)
        )
        self.payload_pattern_edit.setText(settings.payload_pattern)
        self.crc_check.setChecked(True)
        self.crc_init_edit.setText("555555")
        self.whitening_check.setChecked(False)
        self.deviation_spin.setValue(settings.frequency_deviation_hz / 1e3)
        self.bt_spin.setValue(settings.gaussian_bt)
        symbol_rate_hz = 1_000_000.0 if phy == BluetoothLEPhy.LE_1M else 2_000_000.0
        packet_symbols = sum(field.symbol_count for field in bluetooth_le_fields(settings))
        interval_us = np.ceil((packet_symbols / symbol_rate_hz * 1e6 + 249.0) / 625.0) * 625.0
        self.period_spin.setValue(interval_us * 1e-6 * symbol_rate_hz)

    def _update_post_idle_reference(self) -> None:
        post_idle = max(0.0, self.period_spin.value() - self._minimum_period_symbols)
        self.post_idle_value.setText(
            f"{post_idle:.3f} symbols = "
            f"{post_idle / self._symbol_rate_hz() * 1e6:.6g} us"
        )

    def _update_period_constraints(self, _value=None) -> None:
        phy = BluetoothLEPhy(self.phy_combo.currentData())
        settings = replace(
            self._project.bluetooth_le,
            phy=phy,
            payload_length_bytes=self.length_spin.value(),
            pre_idle_symbols=self.pre_idle_spin.value(),
        )
        symbol_rate_hz = 1_000_000.0 if phy == BluetoothLEPhy.LE_1M else 2_000_000.0
        candidate = replace(
            self._project,
            sample_rate_hz=symbol_rate_hz * self.sps_spin.value(),
            samples_per_symbol=self.sps_spin.value(),
            fields=bluetooth_le_fields(settings),
            bluetooth_le=settings,
            power_envelope=replace(
                self._project.power_envelope,
                rise_symbols=self.rise_spin.value(),
                rise_delay_symbols=self.rise_delay_spin.value(),
                fall_symbols=self.fall_spin.value(),
                fall_delay_symbols=self.fall_delay_spin.value(),
            ),
        )
        self._minimum_period_symbols = minimum_period_symbols(candidate)
        packet_symbols = sum(field.symbol_count for field in candidate.fields)
        self.packet_duration_label.setText(
            f"{packet_symbols} symbols / "
            f"{packet_symbols / symbol_rate_hz * 1e6:.6g} us"
        )
        self.period_spin.setMinimum(self._minimum_period_symbols)
        self._update_post_idle_reference()
        self._update_rf_preview()

    def _accept_settings(self) -> None:
        if not ensure_valid_numeric_inputs(
            self, title="Invalid Bluetooth LE Setting"
        ):
            return
        phy = BluetoothLEPhy(self.phy_combo.currentData())
        try:
            crc_init = int(self.crc_init_edit.text().strip(), 16)
        except ValueError:
            QtWidgets.QMessageBox.warning(
                self, "Bluetooth LE Settings", "CRCInit must be hexadecimal."
            )
            return
        previous = self._project.bluetooth_le
        rf_test_fields = (
            self.sync_edit.text().strip().replace(" ", "")
            == "10010100100000100110111010001110"
            and not self.whitening_check.isChecked()
            and crc_init == 0x555555
        )
        settings = replace(
            previous,
            phy=phy,
            preamble_bits=self.preamble_edit.text(),
            sync_word_bits=self.sync_edit.text(),
            pdu_header_bits=self.header_edit.text(),
            payload_type=BluetoothLEPayloadType(self.payload_combo.currentData()),
            payload_source=BluetoothLEPayloadSourceKind(
                self.payload_source_combo.currentData()
            ),
            payload_pattern=self.payload_pattern_edit.text(),
            payload_length_bytes=self.length_spin.value(),
            crc_enabled=self.crc_check.isChecked(),
            crc_init=crc_init,
            whitening_enabled=self.whitening_check.isChecked(),
            whitening_channel_index=self.whitening_channel_spin.value(),
            rf_test_interval_enabled=rf_test_fields,
            frequency_deviation_hz=self.deviation_spin.value() * 1e3,
            gaussian_bt=self.bt_spin.value(),
            pre_idle_symbols=self.pre_idle_spin.value(),
            post_idle_symbols=0,
        )
        symbol_rate_hz = 1_000_000.0 if phy == BluetoothLEPhy.LE_1M else 2_000_000.0
        project = replace(
            self._project,
            name=self.name_edit.text(),
            center_frequency_hz=(
                float(self.carrier_combo.currentData())
                + self.frequency_offset_spin.value() * 1e3
            ),
            samples_per_symbol=self.sps_spin.value(),
            sample_rate_hz=symbol_rate_hz * self.sps_spin.value(),
            repeat_count=self._project.repeat_count,
            period_symbols=self.period_spin.value(),
            fields=bluetooth_le_fields(settings),
            bluetooth_le=settings,
            power_envelope=replace(
                self._project.power_envelope,
                rise_symbols=self.rise_spin.value(),
                rise_delay_symbols=self.rise_delay_spin.value(),
                fall_symbols=self.fall_spin.value(),
                fall_delay_symbols=self.fall_delay_spin.value(),
                shape=self.ramp_combo.currentText(),
            ),
        )
        minimum_period = minimum_period_symbols(project)
        if project.period_symbols is not None and project.period_symbols < minimum_period:
            project = replace(project, period_symbols=minimum_period)
            self.period_spin.setValue(minimum_period)
            self._minimum_period_symbols = minimum_period
            self._update_post_idle_reference()
        issues = validate_project(project)
        if issues:
            QtWidgets.QMessageBox.warning(
                self,
                "Bluetooth LE Settings",
                "\n".join(f"{issue.path}: {issue.message}" for issue in issues),
            )
            return
        self._project = project
        self.accept()

    @property
    def project(self) -> WaveformProject:
        return self._project


class _BluetoothHDTSettingsDialog(QtWidgets.QDialog):
    """Compact HDT test-packet editor backed by the shared PHY definitions."""

    def __init__(self, project: WaveformProject, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Bluetooth HDT Settings")
        self._project = project
        settings = project.bluetooth_hdt
        if settings is None:
            raise ValueError("Bluetooth HDT settings are required")
        self.name_value = QtWidgets.QLabel(project.name)
        hdt_carriers = bluetooth_le_carriers()
        hdt_nominal_hz = min(
            (frequency for _label, frequency in hdt_carriers),
            key=lambda frequency: abs(frequency - project.center_frequency_hz),
        )
        self.carrier_combo = carrier_selector(hdt_carriers, hdt_nominal_hz)
        self.frequency_offset_spin = DeferredDoubleSpinBox()
        self.frequency_offset_spin.setRange(-3000.0, 3000.0)
        self.frequency_offset_spin.setDecimals(3)
        self.frequency_offset_spin.setSuffix(" kHz")
        self.frequency_offset_spin.setValue(
            (project.center_frequency_hz - float(self.carrier_combo.currentData())) / 1e3
        )
        self.actual_frequency_label = QtWidgets.QLabel()
        self.rate_combo = QtWidgets.QComboBox()
        for rate in HDTRate:
            definition = hdt_definition(rate)
            self.rate_combo.addItem(f"{rate.value} / {definition.modulation} / code {definition.payload_code_rate}", rate)
        self.rate_combo.setCurrentIndex(self.rate_combo.findData(HDTRate(settings.rate)))
        self.length_spin = DeferredSpinBox(); self.length_spin.setRange(1, 510); self.length_spin.setValue(settings.payload_length_bytes)
        self.source_combo = QtWidgets.QComboBox()
        for source in HDTPayloadSourceKind:
            self.source_combo.addItem(source.value, source)
        self.source_combo.setCurrentIndex(self.source_combo.findData(HDTPayloadSourceKind(settings.payload_source)))
        self.pattern_edit = QtWidgets.QLineEdit(settings.payload_pattern)
        self.rolloff_spin = DeferredDoubleSpinBox(); self.rolloff_spin.setRange(0.01, 1.0); self.rolloff_spin.setDecimals(3); self.rolloff_spin.setValue(settings.rrc_rolloff)
        self.sps_spin = DeferredSpinBox(); self.sps_spin.setRange(4, 64); self.sps_spin.setValue(project.samples_per_symbol)
        self.sample_rate_label = QtWidgets.QLabel()
        self.packet_duration_label = QtWidgets.QLabel()
        self.pre_idle_spin = DeferredSpinBox(); self.pre_idle_spin.setRange(0, 100000); self.pre_idle_spin.setValue(settings.pre_idle_symbols)
        self._minimum_period_symbols = minimum_period_symbols(project)
        self.period_spin = DeferredDoubleSpinBox(); self.period_spin.setRange(0.0, 1_000_000.0); self.period_spin.setDecimals(3); self.period_spin.setValue(effective_period_symbols(project))
        self.post_idle_value = QtWidgets.QLabel()
        self.rise_spin = DeferredDoubleSpinBox(); self.rise_spin.setRange(0.0, 1000.0); self.rise_spin.setValue(project.power_envelope.rise_symbols)
        self.rise_delay_spin = DeferredDoubleSpinBox(); self.rise_delay_spin.setRange(-1000.0, 1000.0); self.rise_delay_spin.setValue(project.power_envelope.rise_delay_symbols)
        self.fall_spin = DeferredDoubleSpinBox(); self.fall_spin.setRange(0.0, 1000.0); self.fall_spin.setValue(project.power_envelope.fall_symbols)
        self.fall_delay_spin = DeferredDoubleSpinBox(); self.fall_delay_spin.setRange(-1000.0, 1000.0); self.fall_delay_spin.setValue(project.power_envelope.fall_delay_symbols)
        self.ramp_combo = QtWidgets.QComboBox(); self.ramp_combo.addItems(["Cosine", "Linear"]); self.ramp_combo.setCurrentText(project.power_envelope.shape)
        self.training_value = QtWidgets.QLabel("STS x9 + GI + LTS x2: 74 symbols / 37 us (u=7, p=8; fixed)")
        self.packet_profile_label = QtWidgets.QLabel()
        self.pca_edit = QtWidgets.QLineEdit(f"{settings.pca:010X}")
        self.hec_edit = QtWidgets.QLineEdit(f"{settings.hec_manual:06X}")
        self.crc_init_edit = QtWidgets.QLineEdit(f"{settings.crc_init:08X}")
        self.crc_edit = QtWidgets.QLineEdit(f"{settings.crc_manual:08X}")
        for control, width in (
            (self.pca_edit, 10), (self.hec_edit, 6),
            (self.crc_init_edit, 8), (self.crc_edit, 8),
        ):
            control.setMaxLength(width)
            control.setValidator(QtGui.QRegularExpressionValidator(
                QtCore.QRegularExpression(f"[0-9A-Fa-f]{{0,{width}}}"), control,
            ))
            control.setToolTip(f"Enter exactly {width} hexadecimal digits (without 0x)")
        def choice(values, current):
            control = QtWidgets.QComboBox()
            for label, value in values:
                control.addItem(label, value)
            control.setCurrentIndex(control.findData(current))
            return control
        self.nesn_combo = choice([(str(i), i) for i in range(8)], settings.nesn)
        self.md_combo = choice([("0 — No more data", 0), ("1 — More data", 1)], settings.md)
        self.sn_combo = choice([(str(i), i) for i in range(8)], settings.sn)
        self.llid_combo = choice([(f"0b{i:02b}", i) for i in range(4)], settings.llid)
        self.hec_auto_check = QtWidgets.QCheckBox("Automatic HEC-C")
        self.hec_auto_check.setChecked(settings.hec_auto)
        self.crc_auto_check = QtWidgets.QCheckBox("Automatic CRC-32")
        self.crc_auto_check.setChecked(settings.crc_auto)
        self.control_readback = QtWidgets.QLabel()
        self.integrity_readback = QtWidgets.QLabel()
        self.address_readback = QtWidgets.QLabel()
        for label in (self.training_value, self.packet_profile_label, self.control_readback,
                      self.integrity_readback, self.address_readback):
            label.setWordWrap(True)
        self._timing_controls = tuple(
            SymbolTimeControl(control, lambda: 2_000_000.0)
            for control in (
                self.pre_idle_spin,
                self.period_spin,
                self.rise_spin,
                self.rise_delay_spin,
                self.fall_spin,
                self.fall_delay_spin,
            )
        )
        self.tabs = packet_settings_tabs(
            (
                ("Packet Format", QtWidgets.QLabel("Format 0; minimum Initial Portion (Short / Format 1 / CTE not generated)")),
                ("HDT Rate / Modulation", self.rate_combo),
                ("Payload Length [byte]", self.length_spin),
                ("Packet Length", self.packet_duration_label),
                ("Samples / Symbol", self.sps_spin),
                ("Sample Rate", self.sample_rate_label),
                ("SRRC Roll-off", self.rolloff_spin),
                ("Pre Idle", self._timing_controls[0]),
                ("Packet Period", self._timing_controls[1]),
                ("Derived Post Idle", self.post_idle_value),
                ("Ramp Up", self._timing_controls[2]),
                ("Ramp Up Start rel. Packet", self._timing_controls[3]),
                ("Ramp Down", self._timing_controls[4]),
                ("Ramp Down Start rel. Packet End", self._timing_controls[5]),
                ("Ramp Shape", self.ramp_combo),
            ),
            (
                ("Project Name", self.name_value),
                ("Training / Preamble", self.training_value),
                ("Packet Profile", self.packet_profile_label),
                ("PCA [40-bit hex]", self.pca_edit),
                ("PCA-A / HEC Init (auto)", self.address_readback),
                ("NESN", self.nesn_combo),
                ("Control Header (auto)", self.control_readback),
                ("HEC-C Mode", self.hec_auto_check),
                ("Manual HEC-C [hex]", self.hec_edit),
                ("XHP / RxPP (fixed)", QtWidgets.QLabel("0 / 0 — extended headers not generated")),
                ("MD", self.md_combo),
                ("SN", self.sn_combo),
                ("LLID", self.llid_combo),
                ("Payload Source", self.source_combo),
                ("Payload Pattern", self.pattern_edit),
                ("CRC-32 Init [hex]", self.crc_init_edit),
                ("CRC-32 Mode", self.crc_auto_check),
                ("Manual CRC-32 [hex]", self.crc_edit),
                ("Generated HEC-C / CRC-32", self.integrity_readback),
                ("Terminating Symbols (fixed)", QtWidgets.QLabel("2 zero-label symbols per stream; pi/4-QPSK parity continues")),
            ),
        )
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText("Apply and Generate")
        buttons.accepted.connect(self._accept_settings); buttons.rejected.connect(self.reject)
        layout = QtWidgets.QVBoxLayout(self); layout.addWidget(self.tabs); layout.addWidget(buttons)
        self.resize(860, 760)
        for signal in (
            self.rate_combo.currentIndexChanged,
            self.length_spin.valueChanged,
            self.sps_spin.valueChanged,
            self.pre_idle_spin.valueChanged,
            self.rise_spin.valueChanged,
            self.rise_delay_spin.valueChanged,
            self.fall_spin.valueChanged,
            self.fall_delay_spin.valueChanged,
        ):
            signal.connect(self._update_period_constraints)
        self.carrier_combo.currentIndexChanged.connect(self._update_preview)
        self.frequency_offset_spin.valueChanged.connect(self._update_preview)
        self.source_combo.currentIndexChanged.connect(self._source_changed)
        self.rolloff_spin.valueChanged.connect(self._update_preview)
        for control in (self.pca_edit, self.hec_edit, self.crc_init_edit, self.crc_edit, self.pattern_edit):
            control.textChanged.connect(self._update_preview)
        for control in (self.nesn_combo, self.md_combo, self.sn_combo, self.llid_combo):
            control.currentIndexChanged.connect(self._update_preview)
        self.hec_auto_check.toggled.connect(self._update_preview)
        self.crc_auto_check.toggled.connect(self._update_preview)
        self.period_spin.valueChanged.connect(self._update_post_idle_reference)
        self._source_changed()
        self._update_period_constraints()

    def _source_changed(self, _value=None) -> None:
        self.pattern_edit.setEnabled(
            HDTPayloadSourceKind(self.source_combo.currentData()) in {HDTPayloadSourceKind.FIXED, HDTPayloadSourceKind.PATTERN}
        )
        self._update_preview()

    def _update_preview(self, _value=None) -> None:
        self.hec_edit.setEnabled(not self.hec_auto_check.isChecked())
        self.crc_edit.setEnabled(not self.crc_auto_check.isChecked())
        definition = hdt_definition(self.rate_combo.currentData())
        received_pdu = self._project.manual_packet_fields.get("hdt_pdu_control")
        pdu_control = (
            sum(int(b) << i for i, b in enumerate(received_pdu))
            if received_pdu is not None else self.length_spin.value() + 1
        )
        rfu = int(self._project.manual_packet_fields.get("hdt_rfu", "0"))
        self.control_readback.setText(
            f"PFI=0; RI=0b{definition.rate_indicator:03b}; RFU={rfu}; "
            f"PDU Control={pdu_control} octets"
            f"{' (Manual)' if received_pdu is not None else ''}; FEC tail=00000"
        )
        rf_test = False
        try:
            fields = self._field_settings()
            self.address_readback.setText(
                f"PCA-A=0x{fields.pca >> 24:04X}; HEC Init=0x{fields.pca & 0xFFFFFF:06X}"
            )
            rf_test = hdt_is_rf_test_configuration(fields)
            control = hdt_rf_test_control_bits(
                fields.rate, fields.payload_length_bytes, pca=fields.pca, nesn=fields.nesn,
                hec_override=None if fields.hec_auto else fields.hec_manual,
                pdu_control_override=pdu_control if received_pdu is not None else None, rfu=rfu,
            )
            payload = hdt_payload_bits(replace(self._project, bluetooth_hdt=fields))
            pdu = hdt_rf_test_format0_bits(
                payload, md=fields.md, sn=fields.sn, llid=fields.llid, crc_init=fields.crc_init,
                crc_override=None if fields.crc_auto else fields.crc_manual,
            )
            msb = lambda bits: sum(int(bit) << (bits.size - 1 - i) for i, bit in enumerate(bits))
            self.integrity_readback.setText(
                f"HEC-C=0x{msb(control[-24:]):06X}; CRC-32=0x{msb(pdu[-32:]):08X}"
            )
        except ValueError:
            self.address_readback.setText("Incomplete or invalid field input")
            self.integrity_readback.setText("Incomplete or invalid field input")
        self.packet_profile_label.setText(
            "RF Test Format 0 (PRBS9 / PRBS15, SRRC 0.4)"
            if rf_test else "Custom Format 0 (not the RF Test configuration)"
        )
        nominal_hz = float(self.carrier_combo.currentData())
        actual_hz = nominal_hz + self.frequency_offset_spin.value() * 1e3
        self.actual_frequency_label.setText(
            f"{actual_hz / 1e6:.6f} MHz = {nominal_hz / 1e6:.3f} MHz "
            f"{self.frequency_offset_spin.value():+.3f} kHz"
        )
        self.sample_rate_label.setText(f"{2.0 * self.sps_spin.value():.3f} MS/s")
        for control in self._timing_controls:
            control.refresh()

    def _field_settings(self) -> BluetoothHDTSettings:
        def hexadecimal(control, width):
            text = control.text().strip()
            if not re.fullmatch(f"[0-9A-Fa-f]{{{width}}}", text):
                raise ValueError(f"Enter exactly {width} hexadecimal digits")
            return int(text, 16)
        return replace(
            self._project.bluetooth_hdt,
            rate=HDTRate(self.rate_combo.currentData()),
            payload_length_bytes=self.length_spin.value(),
            payload_source=HDTPayloadSourceKind(self.source_combo.currentData()),
            payload_pattern=self.pattern_edit.text(), rrc_rolloff=self.rolloff_spin.value(),
            pca=hexadecimal(self.pca_edit, 10), nesn=self.nesn_combo.currentData(),
            md=self.md_combo.currentData(), sn=self.sn_combo.currentData(), llid=self.llid_combo.currentData(),
            hec_auto=self.hec_auto_check.isChecked(),
            hec_manual=(hexadecimal(self.hec_edit, 6) if len(self.hec_edit.text()) == 6
                        or not self.hec_auto_check.isChecked() else self._project.bluetooth_hdt.hec_manual),
            crc_init=hexadecimal(self.crc_init_edit, 8), crc_auto=self.crc_auto_check.isChecked(),
            crc_manual=(hexadecimal(self.crc_edit, 8) if len(self.crc_edit.text()) == 8
                        or not self.crc_auto_check.isChecked() else self._project.bluetooth_hdt.crc_manual),
        )

    def _update_post_idle_reference(self, _value=None) -> None:
        post_idle = max(0.0, self.period_spin.value() - self._minimum_period_symbols)
        self.post_idle_value.setText(
            f"{post_idle:.3f} symbols = {post_idle / 2.0:.6g} us"
        )

    def _update_period_constraints(self, _value=None) -> None:
        settings = replace(
            self._project.bluetooth_hdt,
            rate=HDTRate(self.rate_combo.currentData()),
            payload_length_bytes=self.length_spin.value(),
            pre_idle_symbols=self.pre_idle_spin.value(),
            post_idle_symbols=0,
        )
        candidate = replace(
            self._project,
            sample_rate_hz=2_000_000.0 * self.sps_spin.value(),
            samples_per_symbol=self.sps_spin.value(),
            fields=bluetooth_hdt_fields(settings),
            bluetooth_hdt=settings,
            power_envelope=replace(
                self._project.power_envelope,
                rise_symbols=self.rise_spin.value(),
                rise_delay_symbols=self.rise_delay_spin.value(),
                fall_symbols=self.fall_spin.value(),
                fall_delay_symbols=self.fall_delay_spin.value(),
            ),
        )
        self._minimum_period_symbols = minimum_period_symbols(candidate)
        packet_symbols = sum(field.symbol_count for field in candidate.fields)
        self.packet_duration_label.setText(
            f"{packet_symbols} symbols / {packet_symbols / 2.0:.6g} us"
        )
        self.period_spin.setMinimum(self._minimum_period_symbols)
        self._update_post_idle_reference()
        self._update_preview()

    def _accept_settings(self) -> None:
        if not ensure_valid_numeric_inputs(
            self, title="Invalid Bluetooth HDT Setting"
        ):
            return
        try:
            field_settings = self._field_settings()
        except ValueError as error:
            QtWidgets.QMessageBox.warning(self, "Bluetooth HDT Settings", str(error))
            return
        settings = replace(field_settings,
            rate=HDTRate(self.rate_combo.currentData()), payload_length_bytes=self.length_spin.value(),
            payload_source=HDTPayloadSourceKind(self.source_combo.currentData()), payload_pattern=self.pattern_edit.text(),
            rrc_rolloff=self.rolloff_spin.value(), pre_idle_symbols=self.pre_idle_spin.value(),
            post_idle_symbols=0, training_enabled=True,
        )
        candidate = replace(
            self._project,
            name=f"Bluetooth {settings.rate.value} " + (
                "RF Test Packet" if hdt_is_rf_test_configuration(settings) else "Custom Format 0 Packet"
            ),
            center_frequency_hz=(float(self.carrier_combo.currentData()) + self.frequency_offset_spin.value() * 1e3),
            sample_rate_hz=2_000_000.0 * self.sps_spin.value(),
            samples_per_symbol=self.sps_spin.value(),
            repeat_count=self._project.repeat_count,
            period_symbols=self.period_spin.value(),
            fields=bluetooth_hdt_fields(settings),
            bluetooth_hdt=settings,
            power_envelope=replace(
                self._project.power_envelope,
                rise_symbols=self.rise_spin.value(),
                rise_delay_symbols=self.rise_delay_spin.value(),
                fall_symbols=self.fall_spin.value(),
                fall_delay_symbols=self.fall_delay_spin.value(),
                shape=self.ramp_combo.currentText(),
            ),
        )
        issues = validate_project(candidate)
        if issues:
            QtWidgets.QMessageBox.warning(self, "Bluetooth HDT Settings", "\n".join(f"{i.path}: {i.message}" for i in issues)); return
        self._project = candidate
        self.accept()

    @property
    def project(self) -> WaveformProject:
        return self._project


class _BluetoothSettingsDialog(QtWidgets.QDialog):
    def __init__(self, project: WaveformProject, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        settings = project.bluetooth_br
        if settings is None:
            raise ValueError("Bluetooth settings are required")
        self._project = project
        self.setWindowTitle("Bluetooth BR / EDR Settings")

        self.name_edit = QtWidgets.QLineEdit(project.name)
        self.carrier_combo = carrier_selector(
            bluetooth_classic_carriers(), project.center_frequency_hz
        )
        self.actual_frequency_label = QtWidgets.QLabel()
        self.sps_spin = self._integer_spin(4, 64, project.samples_per_symbol)
        self.sample_rate_label = QtWidgets.QLabel()
        self.packet_duration_label = QtWidgets.QLabel()
        self.lap_edit = QtWidgets.QLineEdit(f"{settings.lap:06X}")
        self.uap_edit = QtWidgets.QLineEdit(f"{settings.uap:02X}")
        self.clock_edit = QtWidgets.QLineEdit(f"{settings.clock_6_1:02X}")
        self.lt_addr_spin = self._integer_spin(0, 7, settings.lt_addr)
        self.packet_type_combo = QtWidgets.QComboBox()
        for kind in BluetoothPacketKind:
            self.packet_type_combo.addItem(kind.value, kind)
        self.packet_type_combo.setCurrentIndex(
            self.packet_type_combo.findData(BluetoothPacketKind(settings.packet_kind))
        )
        self.flow_combo = self._bit_combo(settings.flow)
        self.arqn_combo = self._bit_combo(settings.arqn)
        self.seqn_combo = self._bit_combo(settings.seqn)
        self.hec_mode_combo = QtWidgets.QComboBox()
        self.hec_mode_combo.addItem("Auto", True)
        self.hec_mode_combo.addItem("Manual", False)
        self.hec_mode_combo.setCurrentIndex(
            self.hec_mode_combo.findData(bool(settings.hec_auto))
        )
        self._hec_mode_auto = bool(settings.hec_auto)
        self.hec_value = QtWidgets.QLineEdit(f"0x{int(settings.hec_manual):02X}")
        self.hec_value.setMaximumWidth(72)
        self.hec_value.setValidator(
            QtGui.QRegularExpressionValidator(
                QtCore.QRegularExpression(r"0[xX][0-9A-Fa-f]{1,2}"),
                self.hec_value,
            )
        )
        self.hec_value.setToolTip(
            "Auto calculates HEC from Packet Header and UAP. Manual transmits "
            "the entered byte while retaining normal FEC and whitening."
        )
        self.payload_length_spin = self._integer_spin(
            0, 1021, settings.payload_length_bytes
        )
        self.payload_llid_spin = self._integer_spin(0, 3, settings.payload_llid)
        self.payload_flow_combo = self._bit_combo(settings.payload_flow)
        self.payload_header_length_value = QtWidgets.QLabel()
        self.rf_test_payload_combo = QtWidgets.QComboBox()
        for label, value in (
            ("PRBS-9", "prbs9"),
            ("Constant 0", "0"),
            ("Constant 1", "1"),
            ("Alternating 1010", "10"),
            ("Repeating 11110000", "11110000"),
        ):
            self.rf_test_payload_combo.addItem(label, value)
        self.rf_test_apply_button = QtWidgets.QPushButton("Apply RF Test Packet Preset")
        self.rf_test_apply_button.clicked.connect(self._apply_rf_test_preset)
        rf_test_row = QtWidgets.QWidget()
        rf_test_layout = QtWidgets.QHBoxLayout(rf_test_row)
        rf_test_layout.setContentsMargins(0, 0, 0, 0)
        rf_test_layout.addWidget(self.rf_test_payload_combo)
        rf_test_layout.addWidget(self.rf_test_apply_button)
        self.payload_source_combo = QtWidgets.QComboBox()
        for label, source in (
            ("Constant (All 0 / All 1)", PayloadSourceKind.FIXED),
            ("Repeating Bit Pattern", PayloadSourceKind.PATTERN),
            ("PRBS-9", PayloadSourceKind.PRBS9),
        ):
            self.payload_source_combo.addItem(label, source)
        self.payload_source_combo.setCurrentIndex(
            self.payload_source_combo.findData(settings.payload_source)
        )
        self.pattern_edit = QtWidgets.QLineEdit(settings.payload_pattern)
        self.payload_source_help = QtWidgets.QLabel()
        self.payload_source_help.setWordWrap(True)
        self.payload_source_help.setStyleSheet("color: #b8b8b8;")
        self.payload_source_combo.currentIndexChanged.connect(
            self._payload_source_changed
        )
        self.whitening_check = QtWidgets.QCheckBox()
        self.whitening_check.setChecked(settings.whitening_enabled)
        self.deviation_spin = self._double_spin(
            1.0, 1000.0, settings.frequency_deviation_hz / 1e3, 3
        )
        self.cfo_spin = self._double_spin(
            -1000.0, 1000.0, settings.carrier_frequency_offset_hz / 1e3, 3
        )
        self.cfo_spin.setSuffix(" kHz")
        self.bt_spin = self._double_spin(0.05, 2.0, settings.gaussian_bt, 3)
        self.edr_guard_spin = self._integer_spin(0, 1000, settings.edr_guard_symbols)
        self.edr_guard_power_spin = self._double_spin(
            -120.0, 20.0, settings.edr_guard_relative_power_db, 3
        )
        self.edr_guard_power_spin.setSuffix(" dB")
        self.edr_guard_power_spin.setToolTip(
            "Guard amplitude relative to the preceding GFSK section; 0 dB keeps "
            "the current level and negative values reduce it."
        )
        self.edr_guard_ramp_in_spin = self._double_spin(
            0.0, 1000.0, settings.edr_guard_ramp_in_symbols, 3
        )
        self.edr_guard_ramp_out_spin = self._double_spin(
            0.0, 1000.0, settings.edr_guard_ramp_out_symbols, 3
        )
        self.edr_guard_ramp_shape_combo = QtWidgets.QComboBox()
        self.edr_guard_ramp_shape_combo.addItems(["Cosine", "Linear"])
        self.edr_guard_ramp_shape_combo.setCurrentText(
            settings.edr_guard_ramp_shape
        )
        self.edr_rolloff_spin = self._double_spin(0.01, 1.0, settings.edr_rolloff, 3)
        self.edr_power_spin = self._double_spin(
            -60.0, 20.0, settings.edr_relative_power_db, 3
        )
        self.pre_idle_spin = self._integer_spin(
            0, 1_000_000, settings.pre_idle_symbols
        )
        self._minimum_period_symbols = minimum_period_symbols(project)
        self.period_spin = self._double_spin(
            0.0, 1_000_000.0, effective_period_symbols(project), 3
        )
        self.post_idle_value = QtWidgets.QLabel()
        self.period_spin.valueChanged.connect(self._update_post_idle_reference)
        self._update_post_idle_reference()
        self.rise_spin = self._double_spin(
            0.0, 1000.0, project.power_envelope.rise_symbols, 3
        )
        self.fall_spin = self._double_spin(
            0.0, 1000.0, project.power_envelope.fall_symbols, 3
        )
        self.rise_delay_spin = self._double_spin(
            -1000.0, 1000.0, project.power_envelope.rise_delay_symbols, 3
        )
        self.fall_delay_spin = self._double_spin(
            -1000.0, 1000.0, project.power_envelope.fall_delay_symbols, 3
        )
        self.ramp_combo = QtWidgets.QComboBox()
        self.ramp_combo.addItems(["Cosine", "Linear"])
        self.ramp_combo.setCurrentText(project.power_envelope.shape)
        ramp_timing_help = QtWidgets.QLabel(
            "Negative starts before the packet boundary; positive starts after it. "
            "Outside packet data, the first/last symbol frequency is held."
        )
        ramp_timing_help.setWordWrap(True)
        ramp_timing_help.setStyleSheet("color: #b8b8b8;")

        header_group = QtWidgets.QGroupBox("Packet Header")
        header_form = QtWidgets.QFormLayout(header_group)
        for label, widget in (
            ("LT_ADDR", self.lt_addr_spin),
            ("FLOW", self.flow_combo),
            ("ARQN", self.arqn_combo),
            ("SEQN", self.seqn_combo),
            ("HEC Mode", self.hec_mode_combo),
            ("HEC Value", self.hec_value),
        ):
            header_form.addRow(label, widget)
        payload_header_group = QtWidgets.QGroupBox("Payload Header")
        payload_header_form = QtWidgets.QFormLayout(payload_header_group)
        payload_header_form.addRow("LLID", self.payload_llid_spin)
        payload_header_form.addRow("FLOW", self.payload_flow_combo)
        payload_header_form.addRow("LENGTH", self.payload_header_length_value)
        self._timing_controls = tuple(
            SymbolTimeControl(control, lambda: 1_000_000.0)
            for control in (
                self.pre_idle_spin,
                self.period_spin,
                self.rise_spin,
                self.rise_delay_spin,
                self.fall_spin,
                self.fall_delay_spin,
                self.edr_guard_spin,
                self.edr_guard_ramp_in_spin,
                self.edr_guard_ramp_out_spin,
            )
        )
        self.tabs = packet_settings_tabs(
            (
            ("Packet Type / Modulation", self.packet_type_combo),
            ("Payload Length [byte]", self.payload_length_spin),
            ("Packet Length", self.packet_duration_label),
            ("Samples / Symbol", self.sps_spin),
            ("Sample Rate", self.sample_rate_label),
            ("FSK Deviation [kHz]", self.deviation_spin),
            ("Gaussian B*T", self.bt_spin),
            ("EDR Guard", self._timing_controls[6]),
            ("EDR Guard Power rel. GFSK", self.edr_guard_power_spin),
            ("EDR Guard Ramp In", self._timing_controls[7]),
            ("EDR Guard Ramp Out", self._timing_controls[8]),
            ("EDR Guard Ramp Shape", self.edr_guard_ramp_shape_combo),
            ("EDR SRRC Roll-off", self.edr_rolloff_spin),
            ("EDR Power rel. GFSK [dB]", self.edr_power_spin),
            ("Pre Idle", self._timing_controls[0]),
            ("Packet Period", self._timing_controls[1]),
            ("Derived Post Idle", self.post_idle_value),
            ("Ramp Up", self._timing_controls[2]),
            ("Ramp Up Start rel. Packet", self._timing_controls[3]),
            ("Ramp Down", self._timing_controls[4]),
            ("Ramp Down Start rel. Packet End", self._timing_controls[5]),
            ("Ramp Shape", self.ramp_combo),
            ("Ramp Timing", ramp_timing_help),
            ),
            (
            ("Project Name", self.name_edit),
            ("LAP [hex]", self.lap_edit),
            ("UAP [hex]", self.uap_edit),
            ("CLK 6-1 [hex]", self.clock_edit),
            ("Header Fields", header_group),
            ("Payload Header Fields", payload_header_group),
            ("RF Test Payload Preset", rf_test_row),
            ("Payload Source", self.payload_source_combo),
            ("Source Behavior", self.payload_source_help),
            ("Payload Data [bin]", self.pattern_edit),
            ("Whitening", self.whitening_check),
            ),
        )

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept_settings)
        buttons.rejected.connect(self.reject)
        apply_button = buttons.button(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
        )
        apply_button.setText("Apply and Generate")
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.tabs)
        layout.addWidget(buttons)
        self.resize(860, 760)
        self.uap_edit.textChanged.connect(self._update_header_preview)
        self.lt_addr_spin.valueChanged.connect(self._update_header_preview)
        self.flow_combo.currentIndexChanged.connect(self._update_header_preview)
        self.arqn_combo.currentIndexChanged.connect(self._update_header_preview)
        self.seqn_combo.currentIndexChanged.connect(self._update_header_preview)
        self.hec_mode_combo.currentIndexChanged.connect(self._hec_mode_changed)
        self.packet_type_combo.currentIndexChanged.connect(
            self._packet_type_changed
        )
        self.payload_length_spin.valueChanged.connect(
            self._update_payload_header_preview
        )
        self.carrier_combo.currentIndexChanged.connect(self._update_rf_preview)
        self.cfo_spin.valueChanged.connect(self._update_rf_preview)
        for signal in (
            self.payload_length_spin.valueChanged,
            self.sps_spin.valueChanged,
            self.pre_idle_spin.valueChanged,
            self.rise_spin.valueChanged,
            self.rise_delay_spin.valueChanged,
            self.fall_spin.valueChanged,
            self.fall_delay_spin.valueChanged,
            self.edr_guard_spin.valueChanged,
        ):
            signal.connect(self._update_period_constraints)
        self._payload_source_changed()
        # Opening Settings must preserve the saved payload length. Only a
        # subsequent user-initiated packet-type change selects the new
        # packet type's maximum payload.
        self._packet_type_changed(reset_payload=False)
        self._update_period_constraints()
        self._update_header_preview()
        self._update_payload_header_preview()

    @staticmethod
    def _double_spin(
        minimum: float, maximum: float, value: float, decimals: int
    ) -> QtWidgets.QDoubleSpinBox:
        control = DeferredDoubleSpinBox()
        control.setRange(minimum, maximum)
        control.setDecimals(decimals)
        control.setValue(value)
        control.setKeyboardTracking(False)
        return control

    @staticmethod
    def _integer_spin(minimum: int, maximum: int, value: int) -> QtWidgets.QSpinBox:
        control = DeferredSpinBox()
        control.setRange(minimum, maximum)
        control.setValue(value)
        return control

    @staticmethod
    def _bit_combo(value: int) -> QtWidgets.QComboBox:
        control = QtWidgets.QComboBox()
        control.addItem("0", 0)
        control.addItem("1", 1)
        control.setCurrentIndex(control.findData(int(value)))
        return control

    def _update_header_preview(self) -> None:
        hec_auto = bool(self.hec_mode_combo.currentData())
        self.hec_value.setEnabled(not hec_auto)
        if not hec_auto:
            return
        try:
            uap = int(self.uap_edit.text().strip(), 16)
            if not 0 <= uap <= 0xFF:
                raise ValueError
        except ValueError:
            self.hec_value.setText("Invalid UAP")
            return
        packet_kind = BluetoothPacketKind(self.packet_type_combo.currentData())
        packet_type = bluetooth_packet_properties(packet_kind)[1]
        packed = (
            self.lt_addr_spin.value()
            | (packet_type << 3)
            | (int(self.flow_combo.currentData()) << 7)
            | (int(self.arqn_combo.currentData()) << 8)
            | (int(self.seqn_combo.currentData()) << 9)
        )
        header_bits = np.asarray(
            [(packed >> index) & 1 for index in range(10)], dtype=np.uint8
        )
        self.hec_value.setText(f"0x{header_error_check(header_bits, uap):02X}")

    def _hec_mode_changed(self, _index=None) -> None:
        hec_auto = bool(self.hec_mode_combo.currentData())
        # Auto -> Manual deliberately retains the currently displayed correct
        # HEC as the editable starting point for negative-test packets.
        self._hec_mode_auto = hec_auto
        self._update_header_preview()

    def _update_payload_header_preview(self, _value=None) -> None:
        self.payload_header_length_value.setText(
            f"{self.payload_length_spin.value()} byte (auto)"
        )

    def _update_post_idle_reference(self) -> None:
        post_idle = max(0.0, self.period_spin.value() - self._minimum_period_symbols)
        self.post_idle_value.setText(
            f"{post_idle:.3f} symbols = {post_idle:.6g} us"
        )

    def _update_rf_preview(self, _value=None) -> None:
        nominal_hz = float(self.carrier_combo.currentData())
        actual_hz = nominal_hz + self.cfo_spin.value() * 1e3
        self.actual_frequency_label.setText(
            f"{actual_hz / 1e6:.6f} MHz = {nominal_hz / 1e6:.3f} MHz "
            f"{self.cfo_spin.value():+.3f} kHz"
        )
        self.sample_rate_label.setText(f"{self.sps_spin.value():.3f} MS/s")
        for control in self._timing_controls:
            control.refresh()

    def _update_period_constraints(self, _value=None) -> None:
        settings = replace(
            self._project.bluetooth_br,
            packet_kind=BluetoothPacketKind(self.packet_type_combo.currentData()),
            payload_length_bytes=self.payload_length_spin.value(),
            edr_guard_symbols=self.edr_guard_spin.value(),
            pre_idle_symbols=self.pre_idle_spin.value(),
        )
        candidate = replace(
            self._project,
            sample_rate_hz=self.sps_spin.value() * 1e6,
            samples_per_symbol=self.sps_spin.value(),
            fields=bluetooth_br_fields(settings),
            bluetooth_br=settings,
            power_envelope=replace(
                self._project.power_envelope,
                rise_symbols=self.rise_spin.value(),
                rise_delay_symbols=self.rise_delay_spin.value(),
                fall_symbols=self.fall_spin.value(),
                fall_delay_symbols=self.fall_delay_spin.value(),
            ),
        )
        self._minimum_period_symbols = minimum_period_symbols(candidate)
        packet_symbols = sum(field.symbol_count for field in candidate.fields)
        self.packet_duration_label.setText(
            f"{packet_symbols} symbols / {packet_symbols:.6g} us"
        )
        self.period_spin.setMinimum(self._minimum_period_symbols)
        self._update_post_idle_reference()
        self._update_rf_preview()

    def _accept_settings(self) -> None:
        if not ensure_valid_numeric_inputs(
            self, title="Invalid Bluetooth Setting"
        ):
            return
        try:
            lap = int(self.lap_edit.text().strip(), 16)
            uap = int(self.uap_edit.text().strip(), 16)
            clock = int(self.clock_edit.text().strip(), 16)
            hec_manual = int(self.hec_value.text().strip(), 16)
            if not 0 <= hec_manual <= 0xFF:
                raise ValueError
        except ValueError:
            QtWidgets.QMessageBox.warning(
                self,
                "Bluetooth Settings",
                "LAP, UAP, clock and HEC must be valid hexadecimal values.",
            )
            return
        settings = replace(
            self._project.bluetooth_br,
            packet_kind=BluetoothPacketKind(self.packet_type_combo.currentData()),
            lap=lap,
            uap=uap,
            clock_6_1=clock,
            lt_addr=self.lt_addr_spin.value(),
            flow=int(self.flow_combo.currentData()),
            arqn=int(self.arqn_combo.currentData()),
            seqn=int(self.seqn_combo.currentData()),
            hec_auto=bool(self.hec_mode_combo.currentData()),
            hec_manual=hec_manual,
            payload_length_bytes=self.payload_length_spin.value(),
            payload_llid=self.payload_llid_spin.value(),
            payload_flow=int(self.payload_flow_combo.currentData()),
            payload_source=PayloadSourceKind(
                self.payload_source_combo.currentData()
            ),
            payload_pattern=self.pattern_edit.text(),
            whitening_enabled=self.whitening_check.isChecked(),
            frequency_deviation_hz=self.deviation_spin.value() * 1e3,
            carrier_frequency_offset_hz=self.cfo_spin.value() * 1e3,
            gaussian_bt=self.bt_spin.value(),
            edr_guard_symbols=self.edr_guard_spin.value(),
            edr_guard_relative_power_db=self.edr_guard_power_spin.value(),
            edr_guard_ramp_in_symbols=self.edr_guard_ramp_in_spin.value(),
            edr_guard_ramp_out_symbols=self.edr_guard_ramp_out_spin.value(),
            edr_guard_ramp_shape=self.edr_guard_ramp_shape_combo.currentText(),
            edr_rolloff=self.edr_rolloff_spin.value(),
            edr_relative_power_db=self.edr_power_spin.value(),
            pre_idle_symbols=self.pre_idle_spin.value(),
            post_idle_symbols=0,
        )
        project = replace(
            self._project,
            name=self.name_edit.text(),
            center_frequency_hz=float(self.carrier_combo.currentData()),
            sample_rate_hz=self.sps_spin.value() * 1e6,
            samples_per_symbol=self.sps_spin.value(),
            repeat_count=self._project.repeat_count,
            period_symbols=self.period_spin.value(),
            fields=bluetooth_br_fields(settings),
            bluetooth_br=settings,
            power_envelope=replace(
                self._project.power_envelope,
                rise_symbols=self.rise_spin.value(),
                fall_symbols=self.fall_spin.value(),
                rise_delay_symbols=self.rise_delay_spin.value(),
                fall_delay_symbols=self.fall_delay_spin.value(),
                shape=self.ramp_combo.currentText(),
            ),
        )
        minimum_period = minimum_period_symbols(project)
        if project.period_symbols is not None and project.period_symbols < minimum_period:
            project = replace(project, period_symbols=minimum_period)
            self.period_spin.setValue(minimum_period)
            self._minimum_period_symbols = minimum_period
            self._update_post_idle_reference()
        issues = validate_project(project)
        if issues:
            QtWidgets.QMessageBox.warning(
                self,
                "Bluetooth Settings",
                "\n".join(f"{issue.path}: {issue.message}" for issue in issues),
            )
            return
        self._project = project
        self.accept()

    def _packet_type_changed(
        self, _index: int | None = None, *, reset_payload: bool = True
    ) -> None:
        kind = BluetoothPacketKind(self.packet_type_combo.currentData())
        payload_max = bluetooth_packet_properties(kind)[0]
        self.payload_length_spin.setMaximum(payload_max)
        if reset_payload:
            self.payload_length_spin.setValue(payload_max)
        is_edr = bluetooth_packet_is_edr(kind)
        for control in (
            self.edr_guard_spin,
            self.edr_guard_power_spin,
            self.edr_guard_ramp_in_spin,
            self.edr_guard_ramp_out_spin,
            self.edr_guard_ramp_shape_combo,
            self.edr_rolloff_spin,
            self.edr_power_spin,
        ):
            control.setEnabled(is_edr)
        self._update_header_preview()

    def _payload_source_changed(self) -> None:
        source = PayloadSourceKind(self.payload_source_combo.currentData())
        self.pattern_edit.setEnabled(source != PayloadSourceKind.PRBS9)
        if source == PayloadSourceKind.FIXED:
            self.pattern_edit.setPlaceholderText("0 or 1 (repeated for the payload)")
            self.payload_source_help.setText(
                "Uses the first entered bit as a constant value for the entire payload."
            )
        elif source == PayloadSourceKind.PATTERN:
            self.pattern_edit.setPlaceholderText("Binary pattern, repeated as needed")
            self.payload_source_help.setText(
                "Repeats the complete entered bit pattern until the payload is filled."
            )
        else:
            self.pattern_edit.setPlaceholderText("Not used by PRBS-9")
            self.payload_source_help.setText(
                "Generates the Bluetooth test PRBS-9 sequence; Payload Data is ignored."
            )

    def _apply_rf_test_preset(self) -> None:
        value = str(self.rf_test_payload_combo.currentData())
        if value == "prbs9":
            source = PayloadSourceKind.PRBS9
            pattern = self.pattern_edit.text()
        elif value in {"0", "1"}:
            source = PayloadSourceKind.FIXED
            pattern = value
        else:
            source = PayloadSourceKind.PATTERN
            pattern = value
        self.payload_source_combo.setCurrentIndex(
            self.payload_source_combo.findData(source)
        )
        self.pattern_edit.setText(pattern)
        self.whitening_check.setChecked(False)

    @property
    def project(self) -> WaveformProject:
        return self._project


_PLUTO_DEVICE_CACHE: tuple[object, ...] = ()


class _VSGPlutoDiscoveryThread(QtCore.QThread):
    discovery_ready = QtCore.Signal(object, str)

    def run(self) -> None:
        try:
            self.discovery_ready.emit(
                tuple(PlutoOutputBackend.discover_devices()), ""
            )
        except Exception as error:
            self.discovery_ready.emit((), str(error))


class _PlutoOutputDialog(QtWidgets.QDialog):
    def __init__(
        self,
        settings: PlutoTransmitSettings,
        packet_count: int,
        parent: QtWidgets.QWidget,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("ADALM-Pluto Instrument Settings")
        self._settings = settings
        self._packet_count = int(packet_count)
        form = QtWidgets.QFormLayout()
        self.playback_mode_combo = QtWidgets.QComboBox()
        self.playback_mode_combo.addItem("Finite", PlutoPlaybackMode.FINITE.value)
        self.playback_mode_combo.addItem(
            "Continuous", PlutoPlaybackMode.CONTINUOUS.value
        )
        playback_mode = PlutoPlaybackMode(settings.playback_mode)
        self.playback_mode_combo.setCurrentIndex(
            self.playback_mode_combo.findData(playback_mode.value)
        )
        self.uri_combo = QtWidgets.QComboBox()
        self.uri_combo.setEditable(True)
        self._discovery_thread: _VSGPlutoDiscoveryThread | None = None
        self.refresh_devices_button = QtWidgets.QPushButton("Refresh")
        self.refresh_devices_button.clicked.connect(self._refresh_devices)
        selector_row = QtWidgets.QWidget()
        selector_layout = QtWidgets.QHBoxLayout(selector_row)
        selector_layout.setContentsMargins(0, 0, 0, 0)
        selector_layout.addWidget(self.uri_combo, 1)
        selector_layout.addWidget(self.refresh_devices_button)
        self._populate_devices(_PLUTO_DEVICE_CACHE, settings.connection_uri or "")
        self.bandwidth_spin = DeferredDoubleSpinBox()
        self.bandwidth_spin.setRange(0.2, 56.0)
        self.bandwidth_spin.setDecimals(3)
        self.bandwidth_spin.setSuffix(" MHz")
        self.bandwidth_spin.setValue(settings.rf_bandwidth_hz / 1e6)
        self.bandwidth_spin.setKeyboardTracking(False)
        self.output_power_spin = DeferredDoubleSpinBox()
        self.output_power_spin.setDecimals(2)
        self.output_power_spin.setSingleStep(0.5)
        self.output_power_spin.setSuffix(" dBm")
        self.output_power_spin.setKeyboardTracking(False)
        self.digital_backoff_combo = QtWidgets.QComboBox()
        for label, value in (("0 dB (Full Scale)", 0.0), ("-3 dB", -3.0), ("-6 dB", -6.0)):
            self.digital_backoff_combo.addItem(label, value)
        backoff_index = self.digital_backoff_combo.findData(
            float(settings.digital_backoff_db)
        )
        if backoff_index < 0:
            self.digital_backoff_combo.addItem(
                f"{settings.digital_backoff_db:+.2f} dB", settings.digital_backoff_db
            )
            backoff_index = self.digital_backoff_combo.count() - 1
        self.digital_backoff_combo.setCurrentIndex(backoff_index)
        initial_output_power_dbm = (
            float(settings.output_power_dbm)
            if settings.output_power_dbm is not None
            else estimate_pluto_output_power_dbm(
                settings.hardware_gain_db,
                settings.digital_backoff_db,
                settings.center_frequency_hz,
                settings.waveform_active_rms_dbfs,
            )
        )
        self.applied_gain_label = QtWidgets.QLabel()
        self.iq_rms_label = QtWidgets.QLabel(
            f"{settings.waveform_active_rms_dbfs:+.2f} dBFS (active interval)"
        )
        self.iq_peak_label = QtWidgets.QLabel(
            f"{settings.waveform_peak_dbfs:+.2f} dBFS"
        )
        self.crest_factor_label = QtWidgets.QLabel(
            f"{settings.waveform_peak_dbfs - settings.waveform_active_rms_dbfs:.2f} dB"
        )
        self.peak_rf_level_label = QtWidgets.QLabel()
        self.digital_backoff_combo.currentIndexChanged.connect(
            lambda _index: self._update_output_level_constraints()
        )
        self.output_power_spin.valueChanged.connect(self._update_applied_gain_label)
        self._update_output_level_constraints(initial_output_power_dbm)
        self.lead_in_guard_spin = DeferredDoubleSpinBox()
        self.lead_in_guard_spin.setRange(0.0, 1000.0)
        self.lead_in_guard_spin.setDecimals(3)
        self.lead_in_guard_spin.setSuffix(" ms")
        self.lead_in_guard_spin.setValue(settings.lead_in_guard_s * 1e3)
        self.lead_in_guard_spin.setKeyboardTracking(False)
        self.dma_preroll_spin = DeferredDoubleSpinBox()
        self.dma_preroll_spin.setRange(0.0, 1000.0)
        self.dma_preroll_spin.setDecimals(3)
        self.dma_preroll_spin.setSuffix(" ms")
        self.dma_preroll_spin.setValue(settings.dma_preroll_s * 1e3)
        self.dma_preroll_spin.setKeyboardTracking(False)
        self.stop_guard_spin = DeferredDoubleSpinBox()
        self.stop_guard_spin.setRange(10.0, 5000.0)
        self.stop_guard_spin.setDecimals(3)
        self.stop_guard_spin.setSuffix(" ms")
        self.stop_guard_spin.setValue(settings.stop_guard_s * 1e3)
        self.stop_guard_spin.setKeyboardTracking(False)
        form.addRow("Connection URI", selector_row)
        form.addRow("Digital Backoff", self.digital_backoff_combo)
        self.lead_in_guard_spin.setToolTip(
            "Wait after enabling the TX LO while TX gain remains muted, "
            "before applying the requested output level."
        )
        form.addRow("LO Stabilization Wait (Muted)", self.lead_in_guard_spin)
        self.dma_preroll_label = QtWidgets.QLabel(
            "Finite TX Lead-in (Zero IQ)"
        )
        self.dma_preroll_spin.setToolTip(
            "Finite TX only: prepend zero-IQ samples before the generated "
            "packet schedule to protect its first packet from DMA startup."
        )
        form.addRow(self.dma_preroll_label, self.dma_preroll_spin)
        self.stop_guard_label = QtWidgets.QLabel("Finite TX Minimum Hold")
        self.stop_guard_spin.setToolTip(
            "Finite TX only: minimum time to keep the transmission active "
            "after DMA submission before muted cleanup."
        )
        form.addRow(self.stop_guard_label, self.stop_guard_spin)
        self.packet_count_label = QtWidgets.QLabel(str(packet_count))
        self.warning = QtWidgets.QLabel()
        self.warning.setWordWrap(True)
        self.warning.setStyleSheet("color: #e0b050;")
        self.playback_mode_combo.currentIndexChanged.connect(
            self._update_playback_mode_ui
        )
        self._update_playback_mode_ui()
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept_settings)
        buttons.rejected.connect(self.reject)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.warning)
        layout.addWidget(buttons)
        self.resize(660, 360)

    def _selected_playback_mode(self) -> PlutoPlaybackMode:
        return PlutoPlaybackMode(str(self.playback_mode_combo.currentData()))

    def _update_playback_mode_ui(self, _index=None) -> None:
        continuous = (
            self._selected_playback_mode() is PlutoPlaybackMode.CONTINUOUS
        )
        self.dma_preroll_spin.setEnabled(not continuous)
        self.stop_guard_spin.setEnabled(not continuous)
        self.dma_preroll_label.setEnabled(not continuous)
        self.stop_guard_label.setEnabled(not continuous)
        self.packet_count_label.setEnabled(not continuous)
        self.packet_count_label.setText(
            "Ignored; first packet period repeats"
            if continuous
            else str(self._packet_count)
        )
        common = (
            "Digital Backoff is applied to waveform samples before the Pluto DAC. "
            "The RF frequency, sample rate, TX RF bandwidth, output power and "
            "playback mode are controlled from the main VSG panel. "
        )
        if continuous:
            detail = (
                "Continuous repeats exactly the first generated packet period "
                "with Pluto cyclic DMA until Stop is requested. Project Repeat "
                "Count, Finite TX Lead-in and Finite TX Minimum Hold do not "
                "alter the continuous cycle. Stop may interrupt a packet. "
                "Residual LO leakage is not a calibrated RF-off state."
            )
        else:
            detail = (
                "Finite submits the complete requested packet schedule once in "
                "a non-cyclic DMA buffer. Finite TX Lead-in prepends zero IQ to "
                "protect the first packet from the DMA/DAC source-start "
                "transient. Finite TX Minimum Hold sets the minimum post-submit "
                "wait before cleanup. Residual LO leakage is not a calibrated "
                "RF-off state."
            )
        self.warning.setText(common + detail)

    def _populate_devices(self, devices: tuple[object, ...], selected: str) -> None:
        self.uri_combo.blockSignals(True)
        try:
            self.uri_combo.clear()
            self.uri_combo.addItem("Auto (USB preferred)", "")
            for device in devices:
                self.uri_combo.addItem(device.label, device.selector)
                self.uri_combo.setItemData(
                    self.uri_combo.count() - 1,
                    device.description,
                    QtCore.Qt.ItemDataRole.ToolTipRole,
                )
            matching_device = next(
                (device for device in devices if device.uri == selected), None
            )
            selector = matching_device.selector if matching_device is not None else selected
            index = self.uri_combo.findData(selector)
            if index < 0 and selector:
                self.uri_combo.addItem(selector, selector)
                index = self.uri_combo.count() - 1
            self.uri_combo.setCurrentIndex(max(0, index))
        finally:
            self.uri_combo.blockSignals(False)

    def _refresh_devices(self) -> None:
        if self._discovery_thread is not None:
            return
        selected = self.uri_combo.currentData() or self.uri_combo.currentText().strip()
        thread = _VSGPlutoDiscoveryThread(self)
        self._discovery_thread = thread
        self.refresh_devices_button.setEnabled(False)
        self.refresh_devices_button.setText("Scanning...")
        thread.discovery_ready.connect(
            lambda devices, error: self._devices_discovered(selected, devices, error)
        )
        thread.finished.connect(self._device_discovery_finished)
        thread.finished.connect(thread.deleteLater)
        thread.start()

    def _devices_discovered(
        self, selected: str, devices: object, error: str
    ) -> None:
        global _PLUTO_DEVICE_CACHE
        self.refresh_devices_button.setEnabled(True)
        self.refresh_devices_button.setText("Refresh")
        if error:
            self.refresh_devices_button.setToolTip(error)
            return
        _PLUTO_DEVICE_CACHE = tuple(devices)
        self._populate_devices(_PLUTO_DEVICE_CACHE, selected)

    def _device_discovery_finished(self) -> None:
        self._discovery_thread = None

    def done(self, result: int) -> None:
        # libiio discovery itself is not cancellable. Keep the QThread wrapper
        # alive until scan_contexts returns so closing the dialog cannot delete
        # a running native thread.
        if self._discovery_thread is not None and self._discovery_thread.isRunning():
            self._discovery_thread.wait()
        super().done(result)

    def _selected_backoff_db(self) -> float:
        return float(self.digital_backoff_combo.currentData())

    def _update_output_level_constraints(self, requested_value=None) -> None:
        backoff_db = self._selected_backoff_db()
        minimum_dbm, maximum_dbm = pluto_output_power_range_dbm(
            backoff_db,
            self._settings.center_frequency_hz,
            self._settings.waveform_active_rms_dbfs,
        )
        if isinstance(requested_value, (int, float)):
            level_dbm = float(requested_value)
        else:
            level_dbm = self.output_power_spin.value()
        self.output_power_spin.blockSignals(True)
        self.output_power_spin.setRange(minimum_dbm, maximum_dbm)
        self.output_power_spin.setValue(
            min(maximum_dbm, max(minimum_dbm, level_dbm))
        )
        self.output_power_spin.blockSignals(False)
        self._update_applied_gain_label()

    def _update_applied_gain_label(self, _value=None) -> None:
        gain_db = pluto_hardware_gain_for_output_power_dbm(
            self.output_power_spin.value(),
            self._selected_backoff_db(),
            self._settings.center_frequency_hz,
            self._settings.waveform_active_rms_dbfs,
        )
        self.applied_gain_label.setText(f"{gain_db:+.2f} dB (estimated)")
        peak_rf_dbm = self.output_power_spin.value() + (
            self._settings.waveform_peak_dbfs
            - self._settings.waveform_active_rms_dbfs
        )
        self.peak_rf_level_label.setText(f"{peak_rf_dbm:+.2f} dBm (estimated)")

    def _accept_settings(self) -> None:
        if not ensure_valid_numeric_inputs(
            self, title="Invalid Pluto Output Setting"
        ):
            return
        uri = self.uri_combo.currentData()
        if uri is None:
            uri = self.uri_combo.currentText().strip()
            if uri == "Auto (USB preferred)":
                uri = ""
        backoff_db = self._selected_backoff_db()
        output_power_dbm = self.output_power_spin.value()
        candidate = replace(
            self._settings,
            connection_uri=str(uri).strip() or None,
            rf_bandwidth_hz=self.bandwidth_spin.value() * 1e6,
            hardware_gain_db=pluto_hardware_gain_for_output_power_dbm(
                output_power_dbm,
                backoff_db,
                self._settings.center_frequency_hz,
                self._settings.waveform_active_rms_dbfs,
            ),
            digital_backoff_db=backoff_db,
            lead_in_guard_s=self.lead_in_guard_spin.value() * 1e-3,
            dma_preroll_s=self.dma_preroll_spin.value() * 1e-3,
            stop_guard_s=self.stop_guard_spin.value() * 1e-3,
            burst_count=self._packet_count,
            output_power_dbm=output_power_dbm,
            playback_mode=self._selected_playback_mode(),
        )
        try:
            PlutoOutputBackend(candidate)
        except ValueError as error:
            QtWidgets.QMessageBox.warning(self, "Pluto Output Settings", str(error))
            return
        self._settings = candidate
        self.accept()

    @property
    def settings(self) -> PlutoTransmitSettings:
        return self._settings


class _PlutoTransmitWorker(QtCore.QObject):
    finished = QtCore.Signal(bool, str)
    first_tx_completed = QtCore.Signal()

    def __init__(self, backend: PlutoOutputBackend, result: GenerationResult) -> None:
        super().__init__()
        self.backend = backend
        self.result = result
        self._cancel_requested = False

    @QtCore.Slot()
    def run(self) -> None:
        success = False
        message = ""
        try:
            self.backend.transfer(self.result)
            self.backend.start(
                on_first_tx_completed=self.first_tx_completed.emit,
            )
        except Exception as error:
            message = str(error)
        else:
            success = True
            continuous = (
                PlutoPlaybackMode(self.backend.settings.playback_mode)
                is PlutoPlaybackMode.CONTINUOUS
            )
            message = (
                "Pluto continuous transmission stopped"
                if self._cancel_requested and continuous
                else "Pluto transmission stopped"
                if self._cancel_requested
                else "Pluto transmission complete"
            )
        report = self.backend.diagnostic_report()
        report["success"] = success
        report["message"] = message
        log_path = diagnostic_log_path("PlutoVSG", "pluto_vsg_tx_trace.log")
        try:
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(report, ensure_ascii=False) + "\n")
        except OSError as error:
            message = f"{message} (diagnostic log failed: {error})"
        self.finished.emit(success, message)

    def cancel(self) -> None:
        self._cancel_requested = True
        self.backend.stop()


class _PlutoPrepareWorker(QtCore.QObject):
    finished = QtCore.Signal(bool, str)

    def __init__(self, backend: PlutoOutputBackend) -> None:
        super().__init__()
        self.backend = backend

    @QtCore.Slot()
    def run(self) -> None:
        success = False
        message = ""
        try:
            self.backend.prepare()
        except Exception as error:
            message = str(error)
        else:
            success = True
            message = "ADALM-Pluto READY (configuration calibrated)"
        report = self.backend.diagnostic_report()
        report["operation"] = "prepare"
        report["success"] = success
        report["message"] = message
        log_path = diagnostic_log_path("PlutoVSG", "pluto_vsg_tx_trace.log")
        try:
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(report, ensure_ascii=False) + "\n")
        except OSError as error:
            message = f"{message} (diagnostic log failed: {error})"
        self.finished.emit(success, message)


class _ProjectChangeCommand(QtGui.QUndoCommand):
    """Undoable replacement of the immutable WaveformProject snapshot."""

    def __init__(
        self,
        window: "PlutoVSGWindow",
        before: WaveformProject,
        after: WaveformProject,
        description: str,
    ) -> None:
        super().__init__(description)
        self._window = window
        self._before = before
        self._after = after

    def redo(self) -> None:
        self._window._restore_project_snapshot(self._after)

    def undo(self) -> None:
        self._window._restore_project_snapshot(self._before)


class PlutoVSGWindow(QtWidgets.QMainWindow):
    """Own project state, generation, preview and first export workflow."""

    def __init__(
        self,
        project: WaveformProject | None = None,
        *,
        preferences: QtCore.QSettings | None = None,
        restore_startup_state: bool = False,
    ) -> None:
        super().__init__()
        self._preferences = preferences or QtCore.QSettings(
            "PlutoSpectrumApp", "PlutoVSG"
        )
        self._persist_startup_state = bool(restore_startup_state)
        restored = (
            self._load_startup_state()
            if self._persist_startup_state and project is None
            else {}
        )
        restored_project = restored.get("project")
        self.project = (
            project
            or (
                restored_project
                if isinstance(restored_project, WaveformProject)
                else None
            )
            or bluetooth_br_edr_project()
        )
        self.result: GenerationResult | None = None
        restored_path = str(restored.get("project_path", "") or "")
        self.project_path: Path | None = Path(restored_path) if restored_path else None
        # Field boundaries are part of the standard VSG preview.  Keep this
        # compatibility attribute fixed so startup data written by older
        # versions cannot restore the retired "off" setting.
        self._field_display_mode = "all"
        self._plot_initial_ranges: dict[
            str, tuple[list[float], list[float]]
        ] = {}
        self._plot_context_actions: dict[str, dict[str, QtGui.QAction]] = {}
        self._tx_thread: QtCore.QThread | None = None
        self._tx_worker: _PlutoTransmitWorker | None = None
        self._prepare_thread: QtCore.QThread | None = None
        self._prepare_worker: _PlutoPrepareWorker | None = None
        self._pluto_prepared_signature: tuple[object, ...] | None = None
        self._preparing_signature: tuple[object, ...] | None = None
        self._calibration_in_progress = False
        self._close_after_tx = False
        self._shutdown_stop_requested = False
        self.undo_stack = QtGui.QUndoStack(self)
        self._selected_composer_block: ComposerBlock | None = None
        preferences = self._preferences
        self._pluto_uri = str(preferences.value("pluto_tx/uri", "") or "")
        self._pluto_digital_backoff_db = float(
            preferences.value("pluto_tx/digital_backoff_db", 0.0)
        )
        legacy_gain_db = float(
            preferences.value("pluto_tx/hardware_gain_db", -30.0)
        )
        if preferences.contains("pluto_tx/output_power_dbm"):
            self._pluto_output_power_dbm = float(
                preferences.value("pluto_tx/output_power_dbm")
            )
        else:
            self._pluto_output_power_dbm = estimate_pluto_output_power_dbm(
                legacy_gain_db,
                self._pluto_digital_backoff_db,
                self.project.center_frequency_hz,
            )
        self._pluto_bandwidth_hz = float(
            preferences.value("pluto_tx/rf_bandwidth_hz", 8_000_000.0)
        )
        self._pluto_lead_in_guard_s = float(
            preferences.value("pluto_tx/lead_in_guard_s", 0.010)
        )
        self._pluto_dma_preroll_s = float(
            preferences.value("pluto_tx/dma_preroll_s", 0.010)
        )
        self._pluto_stop_guard_s = float(
            preferences.value("pluto_tx/stop_guard_s", 0.100)
        )
        self._pluto_playback_mode = PlutoPlaybackMode.CONTINUOUS
        self._rf_enabled = False
        self._rf_transfer_pending = False
        self._modulation_enabled = bool(restored.get("modulation_enabled", True))
        self._continuous_enabled = bool(restored.get("continuous_enabled", True))
        self._power_step_db = float(preferences.value("pluto_tx/power_step_db", 10.0))
        self._frequency_selections = {
            self.project.standard: default_frequency_selection(self.project)
        }
        self._update_pluto_window_title()
        # Match the unified VSA shell so switching between instruments does
        # not resize or reposition the user's working area unexpectedly.
        self.resize(1600, 960)
        self._build_actions()
        self._install_window_shortcuts()
        self._build_workspace()
        self._configure_plot_interaction()
        restore_window_geometry(
            self, self._preferences, restore=self._persist_startup_state
        )
        self._refresh_project_view()
        self.generate_waveform()

    def _load_startup_state(self) -> dict[str, object]:
        serialized = self._preferences.value(_STARTUP_STATE_KEY, "", type=str)
        if not serialized:
            return {}
        try:
            document = json.loads(serialized)
            if not isinstance(document, dict):
                raise ValueError("startup state root must be an object")
            if document.get("schema") != _STARTUP_STATE_SCHEMA:
                raise ValueError("startup state schema is invalid")
            if int(document.get("version", 0)) != _STARTUP_STATE_VERSION:
                raise ValueError("startup state version is unsupported")
            project_document = document.get("project")
            if not isinstance(project_document, dict):
                raise ValueError("startup project is missing")
            return {
                "project": project_from_dict(project_document),
                "project_path": document.get("project_path", ""),
                "modulation_enabled": document.get("modulation_enabled", True),
                "continuous_enabled": document.get("continuous_enabled", True),
            }
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}

    def _save_startup_state(self) -> None:
        if not self._persist_startup_state:
            return
        document = {
            "schema": _STARTUP_STATE_SCHEMA,
            "version": _STARTUP_STATE_VERSION,
            "project": project_to_dict(self.project),
            "project_path": "" if self.project_path is None else str(self.project_path),
            "modulation_enabled": self._modulation_enabled,
            "continuous_enabled": self._continuous_enabled,
        }
        self._preferences.setValue(
            _STARTUP_STATE_KEY,
            json.dumps(document, ensure_ascii=False, separators=(",", ":")),
        )
        save_window_geometry(self, self._preferences)
        self._preferences.remove(_STARTUP_WINDOW_STATE_KEY)
        self._preferences.setValue("pluto_tx/uri", self._pluto_uri)
        self._preferences.setValue(
            "pluto_tx/digital_backoff_db", self._pluto_digital_backoff_db
        )
        self._preferences.setValue(
            "pluto_tx/output_power_dbm", self._pluto_output_power_dbm
        )
        self._preferences.setValue(
            "pluto_tx/rf_bandwidth_hz", self._pluto_bandwidth_hz
        )
        self._preferences.setValue(
            "pluto_tx/lead_in_guard_s", self._pluto_lead_in_guard_s
        )
        self._preferences.setValue(
            "pluto_tx/dma_preroll_s", self._pluto_dma_preroll_s
        )
        self._preferences.setValue(
            "pluto_tx/stop_guard_s", self._pluto_stop_guard_s
        )
        self._preferences.setValue("pluto_tx/power_step_db", self._power_step_db)
        self._preferences.sync()

    def _update_pluto_window_title(self) -> None:
        identity = short_pluto_identity(self._pluto_uri)
        self.setWindowTitle(
            f"Pluto VSG - IQ Waveform Generator [TX: {identity}]"
        )

    def _build_actions(self) -> None:
        self.undo_action = self.undo_stack.createUndoAction(self, "Undo")
        self.undo_action.setShortcut(QtGui.QKeySequence.StandardKey.Undo)
        self.redo_action = self.undo_stack.createRedoAction(self, "Redo")
        self.redo_action.setShortcut(QtGui.QKeySequence.StandardKey.Redo)
        self.new_action = QtGui.QAction("New Bluetooth BR / EDR Project", self)
        self.new_action.triggered.connect(self._new_bluetooth_project)
        self.new_le_action = QtGui.QAction("New Bluetooth LE Packet", self)
        self.new_le_action.triggered.connect(self._new_bluetooth_le_project)
        self.new_hdt_action = QtGui.QAction("New Bluetooth HDT Packet", self)
        self.new_hdt_action.triggered.connect(self._new_bluetooth_hdt_project)
        self.new_wifi_action = QtGui.QAction("New Wi-Fi Packet", self)
        self.new_wifi_action.triggered.connect(self._new_wifi_project)
        self.new_dect_action = QtGui.QAction("New DECT Packet", self)
        self.new_dect_action.triggered.connect(self._new_dect_project)
        self.open_action = QtGui.QAction("Open...", self)
        self.open_action.triggered.connect(self._open_project)
        self.save_action = QtGui.QAction("Save", self)
        self.save_action.triggered.connect(self._save_project)
        self.settings_action = QtGui.QAction("Bluetooth BR / EDR Settings...", self)
        self.settings_action.triggered.connect(self._edit_project_settings)
        self.packet_fields_action = QtGui.QAction(
            "Received Packet Fields...", self
        )
        self.packet_fields_action.triggered.connect(self._edit_received_fields)
        self.generate_action = QtGui.QAction("Generate Waveform", self)
        self.generate_action.setShortcut(QtGui.QKeySequence("F5"))
        self.generate_action.triggered.connect(self.generate_waveform)
        self.export_npz_action = QtGui.QAction("Export NPZ...", self)
        self.export_npz_action.triggered.connect(self._export_npz)
        self.export_iqtar_action = QtGui.QAction("Export R&S IQ TAR...", self)
        self.export_iqtar_action.triggered.connect(self._export_iq_tar)
        self.export_wv_action = QtGui.QAction("Export R&S WV...", self)
        self.export_wv_action.triggered.connect(self._export_wv)
        self.pluto_settings_action = QtGui.QAction("ADALM-Pluto Settings...", self)
        self.pluto_settings_action.triggered.connect(self._edit_pluto_settings)
        self.pluto_prepare_action = QtGui.QAction(
            "Prepare / Calibrate ADALM-Pluto", self
        )
        self.pluto_prepare_action.triggered.connect(self._start_pluto_preparation)
        self.pluto_transmit_action = QtGui.QAction("Transmit with ADALM-Pluto", self)
        self.pluto_transmit_action.triggered.connect(self._start_pluto_transmission)
        self.pluto_cw_action = QtGui.QAction(
            "Start CW with ADALM-Pluto (Current Frequency / Level)", self
        )
        self.pluto_cw_action.triggered.connect(self._start_pluto_cw)
        self.pluto_stop_action = QtGui.QAction("Stop Pluto Transmission", self)
        self.pluto_stop_action.setEnabled(False)
        self.pluto_stop_action.triggered.connect(self._stop_pluto_transmission)
    def _install_window_shortcuts(self) -> None:
        # These actions used to live in the menu bar.  Register them directly
        # on the window so their shortcuts remain available without reserving
        # any menu-bar space.
        self.addActions(
            [self.undo_action, self.redo_action, self.generate_action]
        )

    def _build_workspace(self) -> None:
        self.block_library = QtWidgets.QListWidget()
        self.block_library.addItems(
            ["Fixed Data", "Pattern", "PRBS-9", "Computed Field", "Guard / Idle", "Power Ramp"]
        )
        self.block_library.setEnabled(False)
        self.field_table = QtWidgets.QTreeWidget()
        apply_analysis_font(self.field_table)
        self.field_table.setColumnCount(6)
        self.field_table.setHeaderLabels(
            [
                "Field",
                "Logical Bits",
                "Tx Symbols",
                "Data Source",
                "Modulation",
                "Relative Power",
            ]
        )
        self.field_table.header().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        self.field_table.header().setStretchLastSection(True)
        self.field_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.field_table.currentItemChanged.connect(self._field_tree_selected)
        self.composer_view = PacketComposerView()
        self.composer_view.selected_block_changed.connect(
            self._composer_block_selected
        )
        self.composer_view.block_edit_requested.connect(
            self._edit_composer_block
        )
        composer_tabs = QtWidgets.QTabWidget()
        apply_analysis_font(composer_tabs)
        composer_tabs.addTab(self.composer_view, "Visual Composer")
        composer_tabs.addTab(self.field_table, "Field Tree")
        inspector_widget = QtWidgets.QWidget()
        inspector_layout = QtWidgets.QVBoxLayout(inspector_widget)
        self.inspector = QtWidgets.QTableWidget(0, 2)
        apply_analysis_font(self.inspector)
        self.inspector.setHorizontalHeaderLabels(["Parameter", "Current"])
        self.inspector.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self.inspector.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.edit_settings_button = QtWidgets.QPushButton(
            "Edit Bluetooth BR / EDR Settings..."
        )
        self.edit_settings_button.clicked.connect(self._edit_project_settings)
        inspector_layout.addWidget(self.inspector)
        inspector_layout.addWidget(self.edit_settings_button)

        previews = QtWidgets.QTabWidget()
        apply_analysis_font(previews)
        self.iq_waveform_plot = self._make_plot("Normalized Amplitude", "Time (us)")
        self.iq_waveform_legend = self.iq_waveform_plot.addLegend()
        self.power_plot = self._make_plot("IQ Power (dBFS)", "Time (us)")
        self.frequency_plot = self._make_plot("Frequency (kHz)", "Time (us)")
        self.spectrum_plot = self._make_plot(
            "Magnitude (dBFS)", "Frequency Offset (MHz)"
        )
        self.constellation_plot = self._make_plot("Q", "I")
        self.constellation_plot.setAspectLocked(True)
        self.constellation_legend = self.constellation_plot.addLegend()
        for widget, title in (
            (self.iq_waveform_plot, "IQ Waveform"),
            (self.power_plot, "IQ Power"),
            (self.frequency_plot, "Instantaneous Frequency"),
            (self.spectrum_plot, "Spectrum"),
            (self.constellation_plot, "Constellation"),
        ):
            previews.addTab(widget, title)
        self.packet_decode = PacketDecodeTabs()
        self._verified_packet = None
        self.workspace = QtWidgets.QMainWindow()
        self.workspace.setDockOptions(
            QtWidgets.QMainWindow.DockOption.AllowNestedDocks
            | QtWidgets.QMainWindow.DockOption.AllowTabbedDocks
        )

        def dock(title: str, widget: QtWidgets.QWidget) -> QtWidgets.QDockWidget:
            return make_measurement_dock(
                title, widget, self.workspace, object_prefix="vsg", closable=False
            )

        self.library_dock = dock("Block Library", self.block_library)
        self.composer_dock = dock("Packet Composer", composer_tabs)
        self.preview_dock = dock("Generated IQ Preview", previews)
        self.inspector_dock = dock("Inspector", inspector_widget)
        self.packet_decode_dock = dock("Packet Decode", self.packet_decode)
        self.workspace.addDockWidget(
            QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.library_dock
        )
        self.workspace.splitDockWidget(
            self.library_dock, self.inspector_dock, QtCore.Qt.Orientation.Horizontal
        )
        self.workspace.splitDockWidget(
            self.library_dock, self.preview_dock, QtCore.Qt.Orientation.Vertical
        )
        self.workspace.splitDockWidget(
            self.inspector_dock, self.packet_decode_dock, QtCore.Qt.Orientation.Vertical
        )
        self.workspace.splitDockWidget(
            self.library_dock, self.composer_dock, QtCore.Qt.Orientation.Horizontal
        )
        outer = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        outer.addWidget(self.workspace)
        outer.addWidget(self._build_vsg_control_panel())
        outer.setStretchFactor(0, 1)
        outer.setStretchFactor(1, 0)
        outer.setSizes([1360, 240])
        self.setCentralWidget(outer)
        QtCore.QTimer.singleShot(0, self._initialize_workspace_sizes)

    def _initialize_workspace_sizes(self) -> None:
        """Apply the original proportions once; subsequent resizing is Qt's job."""
        horizontal = QtCore.Qt.Orientation.Horizontal
        vertical = QtCore.Qt.Orientation.Vertical
        self.workspace.resizeDocks(
            [self.preview_dock, self.packet_decode_dock], [820, 410], horizontal
        )
        self.workspace.resizeDocks(
            [self.library_dock, self.composer_dock], [205, 615], horizontal
        )
        self.workspace.resizeDocks(
            [self.library_dock, self.preview_dock], [450, 450], vertical
        )
        self.workspace.resizeDocks(
            [self.inspector_dock, self.packet_decode_dock], [450, 450], vertical
        )

    @staticmethod
    def _make_control_button(
        text: str,
        *,
        value: bool | None = None,
        rf_indicator: bool = False,
    ) -> QtWidgets.QPushButton:
        button_class = _RFStateButton if rf_indicator else QtWidgets.QPushButton
        button = button_class(text)
        configure_control_button(button, value=value)
        button.setStyleSheet(
            "QPushButton { background-color: #303030; color: white; "
            "border: 1px solid #666; padding: 8px; }"
            "QPushButton:hover { background-color: #3c3c3c; }"
            "QPushButton:checked { background-color: #176b87; "
            "border-color: #37b7dc; }"
            "QPushButton:disabled { color: #888; background-color: #292929; }"
            "QPushButton:checked:disabled { color: white; "
            "background-color: #176b87; border-color: #37b7dc; }"
        )
        return button

    def _build_vsg_control_panel(self) -> QtWidgets.QWidget:
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(4, 2, 4, 4)
        layout.setSpacing(8)
        self.vsg_control_page_title = QtWidgets.QLabel("Main Menu")
        self.vsg_control_page_title.hide()
        self.vsg_control_stack = QtWidgets.QStackedWidget()
        layout.addWidget(self.vsg_control_stack, 1)

        main_content = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(main_content)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(8)
        self.rf_button = self._make_control_button(
            "Calibration", value=True, rf_indicator=True
        )
        self.rf_button.setCheckable(True)
        self.mod_button = self._make_control_button("Mod\nON", value=True)
        self.continuous_button = self._make_control_button("Continuous\nON", value=True)
        self.repetitions_button = self._make_control_button("Repeat Count", value=True)
        self.repetitions_button.setToolTip("Number of packets for Continuous OFF; ignored for Continuous ON and CW")
        self.power_button = self._make_control_button("Power", value=True)
        self.power_up_button = QtWidgets.QToolButton()
        self.power_up_button.setArrowType(QtCore.Qt.ArrowType.UpArrow)
        self.power_down_button = QtWidgets.QToolButton()
        self.power_down_button.setArrowType(QtCore.Qt.ArrowType.DownArrow)
        for button in (self.power_up_button, self.power_down_button):
            button.setStyleSheet(
                "QToolButton { background-color: #303030; color: white; "
                "border: 1px solid #666; }"
                "QToolButton:hover { background-color: #3c3c3c; }"
                "QToolButton:disabled { background-color: #292929; }"
            )
        power_arrow_height = (CONTROL_VALUE_BUTTON_HEIGHT - 4) // 2
        self.power_up_button.setMinimumWidth(38)
        self.power_down_button.setMinimumWidth(38)
        self.power_up_button.setFixedHeight(power_arrow_height)
        self.power_down_button.setFixedHeight(power_arrow_height)
        power_row = QtWidgets.QWidget()
        power_layout = QtWidgets.QHBoxLayout(power_row)
        power_layout.setContentsMargins(0, 0, 0, 0)
        power_layout.setSpacing(6)
        power_layout.addWidget(self.power_button, 1)
        arrows = QtWidgets.QVBoxLayout()
        arrows.setContentsMargins(0, 0, 0, 0)
        arrows.setSpacing(4)
        arrows.addWidget(self.power_up_button)
        arrows.addWidget(self.power_down_button)
        power_layout.addLayout(arrows)
        self.estimated_peak_label = QtWidgets.QLabel()
        self.estimated_peak_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        peak_font = QtGui.QFont(self.estimated_peak_label.font())
        if peak_font.pointSizeF() > 0.0:
            peak_font.setPointSizeF(
                peak_font.pointSizeF()
                * ((1.0 + CONTROL_BUTTON_FONT_SCALE) / 2.0)
            )
        peak_font.setBold(True)
        self.estimated_peak_label.setFont(peak_font)
        self.power_step_button = self._make_control_button("Power Step", value=True)
        self.frequency_button = self._make_control_button("Frequency", value=True)
        self.frequency_settings_button = self._make_control_button("Freq Settings")
        self.packet_settings_button = self._make_control_button("Packet Settings")
        self.received_fields_button = self._make_control_button(
            "Received Packet\nFields"
        )
        self.verify_packet_button = self._make_control_button("Verify Packet")
        self.project_button = self._make_control_button("Project")
        self.file_button = self._make_control_button("File")
        self.instrument_settings_button = self._make_control_button("Device")

        self.vsg_setup_group = make_control_group("VSG SETUP")
        for widget in (
            self.rf_button,
            self.mod_button,
            self.continuous_button,
            self.repetitions_button,
            power_row,
            self.estimated_peak_label,
            self.power_step_button,
            self.frequency_button,
            self.frequency_settings_button,
        ):
            self.vsg_setup_group.layout().addWidget(widget)

        self.packet_group = make_control_group("PACKET")
        for widget in (
            self.packet_settings_button,
            self.received_fields_button,
            self.verify_packet_button,
        ):
            self.packet_group.layout().addWidget(widget)

        self.system_group = make_control_group("SYSTEM")
        for widget in (
            self.project_button,
            self.file_button,
            self.instrument_settings_button,
        ):
            self.system_group.layout().addWidget(widget)
        for group in (
            self.vsg_setup_group,
            self.packet_group,
            self.system_group,
        ):
            main_layout.addWidget(group)
        main_layout.addStretch(1)

        main_page = QtWidgets.QScrollArea()
        main_page.setWidgetResizable(True)
        main_page.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        main_page.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        main_page.setWidget(main_content)
        self.vsg_main_control_page = main_page

        def simple_page() -> tuple[QtWidgets.QWidget, QtWidgets.QVBoxLayout]:
            page = QtWidgets.QWidget()
            page_layout = QtWidgets.QVBoxLayout(page)
            page_layout.setContentsMargins(0, 0, 0, 0)
            page_layout.setSpacing(10)
            return page, page_layout

        def action_button(
            label: str, action: QtGui.QAction
        ) -> QtWidgets.QPushButton:
            button = self._make_control_button(label)
            button.clicked.connect(action.trigger)

            def sync() -> None:
                button.setEnabled(action.isEnabled())
                button.setToolTip(action.toolTip())

            action.changed.connect(sync)
            sync()
            return button

        self.vsg_project_page, project_layout = simple_page()
        self.project_open_button = action_button("Open", self.open_action)
        self.project_save_button = action_button("Save", self.save_action)
        self.project_new_button = self._make_control_button("New")
        for button in (
            self.project_open_button,
            self.project_save_button,
            self.project_new_button,
        ):
            project_layout.addWidget(button)
        project_layout.addStretch(1)

        self.vsg_new_project_page, new_layout = simple_page()
        self.new_bluetooth_button = action_button(
            "Bluetooth BR/EDR", self.new_action
        )
        self.new_bluetooth_le_button = action_button(
            "Bluetooth LE", self.new_le_action
        )
        self.new_bluetooth_hdt_button = action_button(
            "Bluetooth HDT", self.new_hdt_action
        )
        self.new_wifi_button = action_button("Wi-Fi", self.new_wifi_action)
        self.new_dect_button = action_button("DECT", self.new_dect_action)
        for button in (
            self.new_bluetooth_button,
            self.new_bluetooth_le_button,
            self.new_bluetooth_hdt_button,
            self.new_wifi_button,
            self.new_dect_button,
        ):
            new_layout.addWidget(button)
        new_layout.addStretch(1)

        self.vsg_file_page, file_layout = simple_page()
        self.export_npz_button = action_button("Export NPZ", self.export_npz_action)
        self.export_iqtar_button = action_button(
            "Export IQ TAR", self.export_iqtar_action
        )
        self.export_wv_button = action_button("Export WV", self.export_wv_action)
        for button in (
            self.export_npz_button,
            self.export_iqtar_button,
            self.export_wv_button,
        ):
            file_layout.addWidget(button)
        file_layout.addStretch(1)

        for page in (
            self.vsg_main_control_page,
            self.vsg_project_page,
            self.vsg_new_project_page,
            self.vsg_file_page,
        ):
            self.vsg_control_stack.addWidget(page)

        self.vsg_control_back_button = self._make_control_button("Back")
        self.vsg_control_back_button.hide()
        add_back_button_footer(layout, self.vsg_control_back_button)

        self.rf_button.clicked.connect(self._toggle_rf)
        self.mod_button.clicked.connect(self._toggle_modulation)
        self.continuous_button.clicked.connect(self._toggle_continuous)
        self.repetitions_button.clicked.connect(self._edit_repetitions)
        self.power_button.clicked.connect(self._edit_output_power)
        self.power_up_button.clicked.connect(lambda: self._step_output_power(+1.0))
        self.power_down_button.clicked.connect(lambda: self._step_output_power(-1.0))
        self.power_step_button.clicked.connect(self._edit_power_step)
        self.frequency_button.clicked.connect(self._edit_frequency)
        self.frequency_settings_button.clicked.connect(self._edit_frequency_settings)
        self.packet_settings_button.clicked.connect(self._edit_project_settings)
        self.received_fields_button.clicked.connect(self._edit_received_fields)

        def sync_packet_setting_buttons() -> None:
            self.packet_settings_button.setEnabled(self.settings_action.isEnabled())
            self.received_fields_button.setEnabled(
                self.packet_fields_action.isEnabled()
            )

        self.settings_action.changed.connect(sync_packet_setting_buttons)
        self.packet_fields_action.changed.connect(sync_packet_setting_buttons)
        sync_packet_setting_buttons()
        self.verify_packet_button.clicked.connect(self._verify_packet)
        self.project_button.clicked.connect(
            lambda: self._show_vsg_control_page(
                "Project", self.vsg_project_page
            )
        )
        self.file_button.clicked.connect(
            lambda: self._show_vsg_control_page("File", self.vsg_file_page)
        )
        self.project_new_button.clicked.connect(
            lambda: self._show_vsg_control_page(
                "New Project", self.vsg_new_project_page
            )
        )
        self.instrument_settings_button.clicked.connect(self._edit_pluto_settings)
        self._update_vsg_control_labels()
        panel = _Panel("Main Menu", content)
        panel_font = QtGui.QFont(panel.font())
        if panel_font.pointSizeF() > 0.0:
            panel_font.setPointSizeF(panel_font.pointSizeF() * (1.45 / 1.3))
        panel.setFont(panel_font)
        panel.setFixedWidth(CONTROL_PANEL_WIDTH)
        self.vsg_control_panel = panel
        self._vsg_control_navigator = ControlPanelNavigator(
            panel=panel,
            title_label=self.vsg_control_page_title,
            stack=self.vsg_control_stack,
            main_page=self.vsg_main_control_page,
            back_button=self.vsg_control_back_button,
            apply_title=self._apply_vsg_control_title,
        )
        self._vsg_control_navigator.show_main()
        return panel

    def _apply_vsg_control_title(
        self, title: str, _page: QtWidgets.QWidget
    ) -> None:
        self.vsg_control_page_title.setText(title)
        self.vsg_control_panel.setTitle(title)

    def _show_vsg_control_page(
        self,
        title: str,
        page: QtWidgets.QWidget,
        *,
        remember: bool = True,
    ) -> None:
        self._vsg_control_navigator.show_page(
            title, page, remember=remember
        )

    def _show_vsg_main_controls(self) -> None:
        self._vsg_control_navigator.show_main()

    def _navigate_vsg_control_back(self) -> None:
        self._vsg_control_navigator.navigate_back()

    @staticmethod
    def _make_plot(left: str, bottom: str) -> pg.PlotWidget:
        return make_measurement_plot(left, bottom)

    def _plot_widgets(self) -> tuple[tuple[str, pg.PlotWidget], ...]:
        return (
            ("iq_waveform", self.iq_waveform_plot),
            ("power", self.power_plot),
            ("frequency", self.frequency_plot),
            ("spectrum", self.spectrum_plot),
            ("constellation", self.constellation_plot),
        )

    def _configure_plot_interaction(self) -> None:
        for name, plot in self._plot_widgets():
            actions = install_measurement_plot_menu(
                plot,
                reset=lambda plot_name=name, target=plot: self._reset_plot_scale(
                    plot_name, target
                ),
            )
            if actions:
                actions["reset"].setToolTip(
                    "Restore this plot's waveform-generation scale"
                )
                self._plot_context_actions[name] = actions

    def _remember_plot_scales(self) -> None:
        for name, plot in self._plot_widgets():
            plot.getViewBox().updateAutoRange()
            x_range, y_range = plot.viewRange()
            self._plot_initial_ranges[name] = (list(x_range), list(y_range))

    def _reset_plot_scale(self, name: str, plot: pg.PlotWidget) -> None:
        ranges = self._plot_initial_ranges.get(name)
        if ranges is None:
            return
        x_range, y_range = ranges
        plot.setRange(xRange=x_range, yRange=y_range, padding=0.0)

    def _new_bluetooth_project(self) -> None:
        previous_signature = self._pluto_configuration_signature()
        self.project = bluetooth_br_edr_project()
        self._frequency_selections[self.project.standard] = (
            default_frequency_selection(self.project)
        )
        self.project_path = None
        self.undo_stack.clear()
        self._refresh_project_view()
        self.generate_waveform()
        self._configuration_maybe_changed(previous_signature)

    def _new_bluetooth_le_project(self) -> None:
        previous_signature = self._pluto_configuration_signature()
        # LE 1M/2M share one settings dialog; start at the broadly compatible
        # 1M default and select PHY inside that dialog.
        self.project = bluetooth_le_project(BluetoothLEPhy.LE_1M)
        self._frequency_selections[self.project.standard] = (
            default_frequency_selection(self.project)
        )
        self.project_path = None
        self.undo_stack.clear()
        self._refresh_project_view()
        self.generate_waveform()
        self._configuration_maybe_changed(previous_signature)

    def _new_bluetooth_hdt_project(self) -> None:
        previous_signature = self._pluto_configuration_signature()
        self.project = bluetooth_hdt_project()
        self._frequency_selections[self.project.standard] = (
            default_frequency_selection(self.project)
        )
        self.project_path = None
        self.undo_stack.clear()
        self._refresh_project_view()
        self.generate_waveform()
        self._configuration_maybe_changed(previous_signature)

    def _new_wifi_project(self) -> None:
        previous_signature = self._pluto_configuration_signature()
        self.project = wifi_beacon_project()
        self._frequency_selections[self.project.standard] = (
            default_frequency_selection(self.project)
        )
        self.project_path = None
        self.undo_stack.clear()
        self._refresh_project_view()
        self.generate_waveform()
        self._configuration_maybe_changed(previous_signature)

    def _new_dect_project(self) -> None:
        previous_signature = self._pluto_configuration_signature()
        self.project = dect_project()
        self._frequency_selections[self.project.standard] = (
            default_frequency_selection(self.project)
        )
        self.project_path = None
        self.undo_stack.clear()
        self._refresh_project_view()
        self.generate_waveform()
        self._configuration_maybe_changed(previous_signature)

    def _apply_rf_test_preset(self) -> None:
        if self.project.standard == StandardProfile.DECT:
            current = self.project.dect
            if current is None:
                return
            settings = replace(
                current,
                b_field_source=DectBFieldSource.PATTERN,
                b_field_pattern="00001111",
                r_crc_auto=True,
                x_crc_auto=True,
                z_repeat_auto=True,
            )
            updated_project = replace(
                self.project,
                name=f"DECT {DectPacketType(settings.packet_type).value} RF Test Packet",
                fields=dect_fields(settings),
                dect=settings,
            )
        elif self.project.standard == StandardProfile.WIFI:
            updated_project = wifi_beacon_project()
        elif self.project.standard == StandardProfile.BLUETOOTH_HDT:
            current = self.project.bluetooth_hdt
            if current is None:
                return
            settings = replace(
                current,
                payload_source=PayloadSourceKind.PRBS9,
                payload_pattern="11111111100000111101",
            )
            updated_project = replace(
                self.project,
                name=f"Bluetooth {settings.rate.value} RF Test Packet",
                fields=bluetooth_hdt_fields(settings),
                bluetooth_hdt=settings,
            )
        elif self.project.standard == StandardProfile.BLUETOOTH_LE:
            current = self.project.bluetooth_le
            if current is None:
                return
            settings = apply_bluetooth_le_rf_test_preset(
                current,
                payload_type=BluetoothLEPayloadType(current.payload_type),
                payload_length_bytes=current.payload_length_bytes,
            )
            updated_project = replace(
                self.project,
                name=f"Bluetooth {BluetoothLEPhy(settings.phy).value} RF Test Packet",
                fields=bluetooth_le_fields(settings),
                bluetooth_le=settings,
            )
        else:
            current = self.project.bluetooth_br
            if current is None:
                return
            settings = replace(
                current,
                payload_source=PayloadSourceKind.PRBS9,
                whitening_enabled=False,
            )
            updated_project = replace(
                self.project,
                name=f"Bluetooth {BluetoothPacketKind(settings.packet_kind).value} RF Test Packet",
                fields=bluetooth_br_fields(settings),
                bluetooth_br=settings,
            )
        self._commit_project_change(updated_project, "Apply RF test packet preset")

    def _edit_project_settings(self) -> None:
        if self.project.standard == StandardProfile.DECT:
            dialog = DectSettingsDialog(self.project, self)
            if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                self._commit_project_change(dialog.project, "Edit DECT packet settings")
        elif self.project.standard == StandardProfile.WIFI:
            dialog = _WiFiSettingsDialog(self.project, self)
            if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                self._commit_project_change(dialog.project, "Edit Wi-Fi packet settings")
        elif self.project.standard == StandardProfile.BLUETOOTH_HDT:
            dialog = _BluetoothHDTSettingsDialog(self.project, self)
            if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                self._commit_project_change(dialog.project, "Edit Bluetooth HDT settings")
        elif self.project.standard == StandardProfile.BLUETOOTH_LE:
            self._edit_bluetooth_le_settings()
        else:
            self._edit_bluetooth_settings()

    def _edit_bluetooth_settings(self) -> None:
        dialog = _BluetoothSettingsDialog(self.project, self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        self._commit_project_change(dialog.project, "Edit Bluetooth BR / EDR settings")

    def _edit_bluetooth_le_settings(self) -> None:
        dialog = _BluetoothLESettingsDialog(self.project, self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        self._commit_project_change(dialog.project, "Edit Bluetooth LE settings")

    def _commit_project_change(
        self, updated_project: WaveformProject, description: str
    ) -> None:
        if updated_project == self.project:
            return
        self.undo_stack.push(
            _ProjectChangeCommand(self, self.project, updated_project, description)
        )

    def _restore_project_snapshot(self, project: WaveformProject) -> None:
        previous_signature = self._pluto_configuration_signature()
        self.project = project
        self._refresh_project_view()
        self.generate_waveform()
        self._configuration_maybe_changed(previous_signature)

    def _refresh_project_view(self) -> None:
        self.packet_fields_action.setEnabled(self.project.wifi is None)
        self.field_table.clear()
        self._field_items_by_block_id: dict[str, QtWidgets.QTreeWidgetItem] = {}

        def add_field(packet_field, parent=None, path="0") -> None:
            values = [
                packet_field.name,
                (
                    "-"
                    if packet_field.logical_bit_count is None
                    else str(packet_field.logical_bit_count)
                ),
                str(packet_field.symbol_count),
                packet_field.data_source.value,
                packet_field.modulation.kind.value,
                f"{packet_field.relative_power_db:+.3g} dB",
            ]
            item = QtWidgets.QTreeWidgetItem(values)
            block_id = f"field:{path}"
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, block_id)
            self._field_items_by_block_id[block_id] = item
            for column in range(1, len(values)):
                item.setTextAlignment(
                    column, QtCore.Qt.AlignmentFlag.AlignCenter
                )
            if parent is None:
                self.field_table.addTopLevelItem(item)
                font = item.font(0)
                font.setBold(True)
                item.setFont(0, font)
            else:
                parent.addChild(item)
            for child_index, child in enumerate(packet_field.children):
                add_field(child, item, f"{path}.{child_index}")

        for field_index, packet_field in enumerate(self.project.fields):
            add_field(packet_field, path=str(field_index))
        self.field_table.expandAll()
        self.composer_view.set_graph(build_composer_graph(self.project))
        settings = self.project.bluetooth_br
        le_settings = self.project.bluetooth_le
        hdt_settings = self.project.bluetooth_hdt
        wifi_settings = self.project.wifi
        dect_settings = self.project.dect
        parameters = [
            ("Project", self.project.name),
            ("Standard", self.project.standard.value),
            ("Center", f"{self.project.center_frequency_hz / 1e6:.6f} MHz"),
            ("Sample Rate", f"{self.project.sample_rate_hz / 1e6:.3f} MS/s"),
            ("Samples / Symbol", str(self.project.samples_per_symbol)),
            ("Period", f"{effective_period_symbols(self.project):.3f} symbols"),
            (
                "Post Idle",
                f"{effective_post_idle_symbols(self.project):.3f} symbols",
            ),
        ]
        if self.project.manual_packet_fields:
            parameters.insert(2, (
                "Received Fields", f"{len(self.project.manual_packet_fields)} Manual overrides (Edit menu)",
            ))
        self.edit_settings_button.setToolTip(
            "Received fields stay Manual when other settings change. "
            "Use Edit > Received Packet Fields to select Auto or edit their values."
            if self.project.manual_packet_fields else ""
        )
        if settings is not None:
            parameters.extend(
                [
                    ("Packet", BluetoothPacketKind(settings.packet_kind).value),
                    ("BD_ADDR", f"{settings.uap:02X}{settings.lap:06X}"),
                    ("Payload", f"{settings.payload_length_bytes} byte / {settings.payload_source.value}"),
                    ("Whitening", "On" if settings.whitening_enabled else "Off"),
                    ("Deviation", f"{settings.frequency_deviation_hz / 1e3:.3f} kHz"),
                    ("Gaussian B*T", f"{settings.gaussian_bt:.3f}"),
                ]
            )
            if bluetooth_packet_is_edr(settings.packet_kind):
                parameters.extend(
                    [
                        ("EDR Guard", f"{settings.edr_guard_symbols} symbols"),
                        (
                            "EDR Guard Power",
                            f"{settings.edr_guard_relative_power_db:+.3f} dB rel. GFSK",
                        ),
                        (
                            "EDR Guard Transition",
                            f"{settings.edr_guard_ramp_in_symbols:.3f} / "
                            f"{settings.edr_guard_ramp_out_symbols:.3f} symbols, "
                            f"{settings.edr_guard_ramp_shape}",
                        ),
                        (
                            "EDR Data Power",
                            f"{settings.edr_relative_power_db:+.3f} dB rel. GFSK",
                        ),
                    ]
                )
        elif le_settings is not None:
            parameters.extend(
                [
                    ("PHY", BluetoothLEPhy(le_settings.phy).value),
                    (
                        "Payload",
                        f"{le_settings.payload_length_bytes} byte / "
                        f"{BluetoothLEPayloadSourceKind(le_settings.payload_source).value}",
                    ),
                    (
                        "Whitening",
                        (
                            f"On / Channel {le_settings.whitening_channel_index}"
                            if le_settings.whitening_enabled
                            else "Off"
                        ),
                    ),
                    (
                        "CRCInit",
                        f"0x{le_settings.crc_init:06X}"
                        if le_settings.crc_enabled
                        else "Disabled",
                    ),
                    ("Deviation", f"{le_settings.frequency_deviation_hz / 1e3:.3f} kHz"),
                    ("Gaussian B*T", f"{le_settings.gaussian_bt:.3f}"),
                ]
            )
        elif hdt_settings is not None:
            definition = hdt_definition(hdt_settings.rate)
            parameters.extend([
                ("PHY", hdt_settings.rate.value), ("Modulation", definition.modulation),
                ("Code Rate", definition.payload_code_rate),
                ("Payload", f"{hdt_settings.payload_length_bytes} byte / {hdt_settings.payload_source.value}"),
                ("SRRC Roll-off", f"{hdt_settings.rrc_rolloff:.3f}"),
            ])
        elif wifi_settings is not None:
            parameters.extend([
                ("PHY", "Non-HT OFDM / 20 MHz"),
                ("Channel", f"{wifi_settings.channel} / {(2407 + 5 * wifi_settings.channel)} MHz"),
                ("Data Rate", f"{wifi_settings.legacy_rate_mbps} Mbps"),
                ("Frame Source", WiFiPSDUSource(wifi_settings.psdu_source).value),
                ("Packet Period", f"{wifi_settings.packet_period_us:g} us"),
                ("Scrambler", f"{WiFiScramblerSeedMode(wifi_settings.scrambler_seed_mode).value} / 0x{wifi_settings.scrambler_seed:02X}"),
                ("SSID", wifi_settings.ssid if WiFiPSDUSource(wifi_settings.psdu_source) == WiFiPSDUSource.BEACON else "-"),
                ("BSSID", wifi_settings.bssid if WiFiPSDUSource(wifi_settings.psdu_source) == WiFiPSDUSource.BEACON else "-"),
            ])
        elif dect_settings is not None:
            parameters.extend(
                [
                    ("Direction", dect_settings.direction.value),
                    ("Packet Type", dect_settings.packet_type.value),
                    ("Carrier Plan", dect_settings.carrier_plan_id),
                    ("Carrier", dect_settings.carrier_channel),
                    (
                        "Frequency Offset",
                        f"{dect_settings.carrier_frequency_offset_hz / 1e3:+.3f} kHz",
                    ),
                    (
                        "Generated RF Frequency",
                        f"{(self.project.center_frequency_hz + dect_settings.carrier_frequency_offset_hz) / 1e6:.6f} MHz",
                    ),
                    (
                        "Deviation",
                        f"{dect_settings.frequency_deviation_hz / 1e3:.3f} kHz",
                    ),
                    ("Gaussian B*T", f"{dect_settings.gaussian_bt:.3f}"),
                    (
                        "B-field Source",
                        DectBFieldSource(dect_settings.b_field_source).value,
                    ),
                    (
                        "B-field Scrambling",
                        f"{dect_settings.scrambling_mode.value} / phase {dect_settings.scrambling_phase}"
                        if str(dect_settings.scrambling_mode.value) == "Standard"
                        else "None",
                    ),
                    (
                        "R/X/Z Generation",
                        f"R-CRC {'Auto' if dect_settings.r_crc_auto else 'Manual'} / "
                        f"X {'Auto' if dect_settings.x_crc_auto else 'Manual'} / "
                        f"Z {'Auto' if dect_settings.z_repeat_auto else 'Manual'}",
                    ),
                ]
            )
        settings_label = ({
            StandardProfile.BLUETOOTH_LE: "Bluetooth LE Packet Settings...",
            StandardProfile.BLUETOOTH_HDT: "Bluetooth HDT Settings...",
            StandardProfile.WIFI: "Wi-Fi Packet / Waveform Settings...",
            StandardProfile.DECT: "DECT Packet / Waveform Settings...",
        }).get(self.project.standard, "Bluetooth BR / EDR Settings...")
        self.settings_action.setText(settings_label)
        self.edit_settings_button.setText(f"Edit {settings_label}")
        self._project_inspector_parameters = parameters
        self._populate_inspector(parameters)
        if hasattr(self, "frequency_button"):
            self._update_vsg_control_labels()
        status = "Ready" if not validate_project(self.project) else "Project has validation errors"
        self.statusBar().showMessage(status)

    def _populate_inspector(self, parameters: list[tuple[str, str]]) -> None:
        self.inspector.setRowCount(len(parameters))
        for row, values in enumerate(parameters):
            for column, value in enumerate(values):
                self.inspector.setItem(
                    row, column, QtWidgets.QTableWidgetItem(value)
                )

    def _composer_block_selected(self, block: ComposerBlock | None) -> None:
        self._selected_composer_block = block
        if block is None:
            self._populate_inspector(
                getattr(self, "_project_inspector_parameters", [])
            )
            return
        tree_item = self._field_items_by_block_id.get(block.block_id)
        with QtCore.QSignalBlocker(self.field_table):
            self.field_table.setCurrentItem(tree_item)
        parameters = [
            ("Block", block.name),
            ("Track", block.track.value),
            ("Role", block.role.value),
            ("Start", f"{block.start_symbol:g} symbols"),
            ("Duration", f"{block.symbol_count:g} symbols"),
            *list(block.properties),
        ]
        self._populate_inspector(parameters)

    def _field_tree_selected(
        self,
        current: QtWidgets.QTreeWidgetItem | None,
        _previous: QtWidgets.QTreeWidgetItem | None,
    ) -> None:
        if current is None:
            return
        block_id = current.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if block_id:
            self.composer_view.select_block(str(block_id))

    def _edit_composer_block(self, _block: ComposerBlock) -> None:
        # Standard-profile blocks are generated from the packet settings.  Route
        # editing through that source of truth until Experimental Profile field
        # mutation is introduced; editing graphics items directly would produce
        # a preview that cannot be regenerated consistently.
        self._edit_project_settings()

    def generate_waveform(self) -> None:
        self.packet_decode.clear_packet()
        self._verified_packet = None
        try:
            engine = {
                StandardProfile.BLUETOOTH_LE: BluetoothLEWaveformEngine,
                StandardProfile.BLUETOOTH_HDT: BluetoothHDTWaveformEngine,
                StandardProfile.WIFI: WiFiLegacyOFDMWaveformEngine,
                StandardProfile.DECT: DectWaveformEngine,
            }.get(self.project.standard, BluetoothBRWaveformEngine)()
            self.result = engine.generate(replace(self.project, repeat_count=1))
        except ValueError as error:
            self.result = None
            self.verify_packet_button.setEnabled(False)
            QtWidgets.QMessageBox.warning(self, "Waveform Generation", str(error))
            return
        self.verify_packet_button.setEnabled(self.result.packet_bits is not None)
        self.verify_packet_button.setToolTip(
            "Decode generated transmitted bits using the shared VSA packet decoder. "
            "This does not demodulate IQ or verify RF performance."
            if self.result.packet_bits is not None else
            "Packet decoding is not available for this waveform template."
        )
        self._update_previews(self.result)
        level_metrics = generation_result_iq_levels(self.result)
        minimum_power, maximum_power = self._power_limits_dbm()
        clamped_power = min(maximum_power, max(minimum_power, self._pluto_output_power_dbm))
        if not np.isclose(clamped_power, self._pluto_output_power_dbm):
            self._pluto_output_power_dbm = clamped_power
            self._save_power_preferences()
        level_names = {
            "IQ Active RMS",
            "IQ Peak",
            "Crest Factor",
            "EDR Block RMS",
        }
        parameters = [
            item
            for item in getattr(self, "_project_inspector_parameters", [])
            if item[0] not in level_names
        ]
        parameters.extend(
            [
                ("IQ Active RMS", f"{level_metrics.active_rms_dbfs:+.3f} dBFS"),
                ("IQ Peak", f"{level_metrics.peak_dbfs:+.3f} dBFS"),
                ("Crest Factor", f"{level_metrics.crest_factor_db:.3f} dB"),
            ]
        )
        block_rms = self.result.metadata.get("block_rms_dbfs")
        if isinstance(block_rms, dict) and block_rms:
            parameters.append(
                (
                    "EDR Block RMS",
                    ", ".join(
                        f"{name} {float(value):+.2f} dBFS"
                        for name, value in block_rms.items()
                    ),
                )
            )
        self._project_inspector_parameters = parameters
        if self._selected_composer_block is None:
            self._populate_inspector(parameters)
        duration_ms = 1e3 * self.result.iq.size / self.result.sample_rate_hz
        self.statusBar().showMessage(
            f"Generated {self.result.iq.size:,} samples | {duration_ms:.3f} ms | "
            f"{self.result.sample_rate_hz / 1e6:.3f} MS/s"
        )
        if hasattr(self, "power_button"):
            self._update_vsg_control_labels()

    def _verify_packet(self) -> None:
        if self.result is None or self.result.packet_bits is None:
            return
        self.packet_decode.clear_packet()
        self._verified_packet = None
        try:
            packet = analyze_generation_result(self.result)
        except (ValueError, RuntimeError) as error:
            QtWidgets.QMessageBox.warning(self, "Verify Packet", str(error))
            return
        self._verified_packet = packet
        self.packet_decode.render_packet(
            packet, p0_internal_bit=int(self.result.packet_bits.context.get("p0_internal_bit", 0)),
        )
        self.statusBar().showMessage(
            f"Packet decoded: {packet.protocol_name} / {packet.packet_type or packet.phy_name or '--'} "
            f"({len(packet.issues)} issue(s)); generated bits, not IQ demodulation"
        )

    def _power_limits_dbm(self) -> tuple[float, float]:
        active_rms_dbfs = 0.0
        if self.result is not None:
            active_rms_dbfs = generation_result_iq_levels(self.result).active_rms_dbfs
        return pluto_output_power_range_dbm(
            self._pluto_digital_backoff_db,
            effective_rf_frequency_hz(self.project),
            active_rms_dbfs,
        )

    def _update_vsg_control_labels(self) -> None:
        if not hasattr(self, "rf_button"):
            return
        calibration_required = (
            self._pluto_prepared_signature
            != self._pluto_configuration_signature()
        )
        if self._calibration_in_progress:
            rf_state = "Calibrating..."
        elif self._rf_enabled:
            rf_state = "ON"
        elif self._rf_transfer_pending:
            rf_state = "Transferring..."
        elif calibration_required:
            rf_state = "Calibration"
        else:
            rf_state = "RF\nOFF"
        self.rf_button.setText(
            f"RF\n{rf_state}" if rf_state in {"ON", "Transferring..."} else rf_state
        )
        self.rf_button.setChecked(self._rf_enabled)
        self.mod_button.setText(f"Mod\n{'ON' if self._modulation_enabled else 'OFF'}")
        self.continuous_button.setText(
            f"Continuous\n{'ON' if self._continuous_enabled else 'OFF'}"
        )
        self.repetitions_button.setText(f"Repeat Count\n{self.project.repeat_count}")
        self.power_button.setText(f"Power\n{self._pluto_output_power_dbm:.2f} dBm")
        self.power_step_button.setText(f"Power Step\n{self._power_step_db:g} dB")
        self.frequency_button.setText(
            f"Frequency\n{effective_rf_frequency_hz(self.project) / 1e6:.6f} MHz"
        )
        crest_db = 0.0
        if self.result is not None:
            crest_db = generation_result_iq_levels(self.result).crest_factor_db
        self.estimated_peak_label.setText(
            "Estimated Peak Power\n"
            f"{self._pluto_output_power_dbm + crest_db:.2f} dBm"
        )

    def _save_power_preferences(self) -> None:
        self._preferences.setValue(
            "pluto_tx/output_power_dbm", self._pluto_output_power_dbm
        )
        self._preferences.setValue("pluto_tx/power_step_db", self._power_step_db)

    def _apply_output_power(self, value_dbm: float) -> None:
        self._pluto_output_power_dbm = float(value_dbm)
        self._save_power_preferences()
        self._update_vsg_control_labels()
        if self._tx_worker is not None:
            active_rms_dbfs = generation_result_iq_levels(
                self._tx_worker.result
            ).active_rms_dbfs
            gain_db = pluto_hardware_gain_for_output_power_dbm(
                self._pluto_output_power_dbm,
                self._pluto_digital_backoff_db,
                effective_rf_frequency_hz(self.project),
                active_rms_dbfs,
            )
            self._tx_worker.backend.request_hardware_gain_db(gain_db)
            self.statusBar().showMessage(
                f"Pluto output power update requested: {value_dbm:.2f} dBm"
            )

    def _edit_output_power(self) -> None:
        minimum, maximum = self._power_limits_dbm()
        value, accepted = get_deferred_double(
            self,
            "RF Output Power",
            f"Power ({minimum:.2f} to {maximum:.2f} dBm)",
            self._pluto_output_power_dbm,
            minimum,
            maximum,
            2,
        )
        if accepted:
            self._apply_output_power(value)

    def _step_output_power(self, direction: float) -> None:
        target = self._pluto_output_power_dbm + float(direction) * self._power_step_db
        minimum, maximum = self._power_limits_dbm()
        if target < minimum or target > maximum:
            self.statusBar().showMessage(
                f"Power step ignored; valid range is {minimum:.2f} to {maximum:.2f} dBm"
            )
            return
        self._apply_output_power(target)

    def _edit_repetitions(self) -> None:
        maximum = maximum_finite_repeat_count(self.project)
        value, accepted = get_deferred_int(
            self,
            "Repeat Count",
            f"Number of packets (Continuous OFF; maximum {maximum} for this period)",
            self.project.repeat_count,
            1,
            maximum,
        )
        if accepted and value != self.project.repeat_count:
            self._commit_project_change(replace(self.project, repeat_count=value), "Change repeat count")

    def _edit_received_fields(self) -> None:
        from pluto_vsg.ui.packet_fields import PacketFieldsDialog
        dialog = PacketFieldsDialog(self.project, self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self._commit_project_change(
                replace(self.project, manual_packet_fields=dialog.manual_fields),
                "Change received packet fields",
            )

    def _edit_power_step(self) -> None:
        value, accepted = get_deferred_double(
            self, "Power Step", "Power step (dB)", self._power_step_db, 0.01, 100.0, 2
        )
        if accepted:
            self._power_step_db = value
            self._save_power_preferences()
            self._update_vsg_control_labels()

    def _edit_frequency(self) -> None:
        value, accepted = get_deferred_double(
            self,
            "RF Frequency",
            "Frequency (MHz)",
            effective_rf_frequency_hz(self.project) / 1e6,
            70.0,
            6000.0,
            6,
        )
        if accepted:
            self._commit_project_change(
                with_manual_rf_frequency(self.project, value * 1e6),
                "Set RF frequency",
            )

    def _edit_frequency_settings(self) -> None:
        selection = self._frequency_selections.get(self.project.standard)
        if selection is None:
            selection = default_frequency_selection(self.project)
        dialog = FrequencySettingsDialog(
            self.project, self, selection=selection
        )
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self._frequency_selections[self.project.standard] = dialog.selection
            self._commit_project_change(dialog.project, "Select RF carrier")

    def _toggle_modulation(self) -> None:
        if self._tx_thread is not None:
            return
        self._modulation_enabled = not self._modulation_enabled
        self._update_vsg_control_labels()

    def _toggle_continuous(self) -> None:
        if self._tx_thread is not None:
            return
        self._continuous_enabled = not self._continuous_enabled
        self._pluto_playback_mode = (
            PlutoPlaybackMode.CONTINUOUS
            if self._continuous_enabled
            else PlutoPlaybackMode.FINITE
        )
        self._update_vsg_control_labels()

    def _toggle_rf(self) -> None:
        # A checkable QPushButton flips itself before emitting ``clicked``.
        # Restore the hardware-backed state immediately so the calibration
        # decision dialog cannot briefly paint RF as ON. Only the worker's
        # first-TX-completed signal is allowed to assert this indicator.
        self.rf_button.setChecked(self._rf_enabled)
        if self._tx_thread is not None:
            self._stop_pluto_transmission()
            return
        if self._prepare_thread is not None:
            return
        if self._pluto_prepared_signature != self._pluto_configuration_signature():
            self._rf_enabled = False
            self._update_vsg_control_labels()
            self._start_pluto_preparation()
            return
        self._pluto_playback_mode = (
            PlutoPlaybackMode.CONTINUOUS
            if (not self._modulation_enabled or self._continuous_enabled)
            else PlutoPlaybackMode.FINITE
        )
        if self._modulation_enabled:
            self._start_pluto_transmission()
        else:
            self._start_pluto_cw()

    def _update_previews(self, result: GenerationResult) -> None:
        complete_iq = np.asarray(result.iq)
        metadata_period = int(result.metadata.get("period_sample_count", 0) or 0)
        if 0 < metadata_period <= complete_iq.size:
            preview_sample_count = metadata_period
        else:
            # Waveform engines are expected to return an integer number of
            # repetitions. Fall back to the complete result rather than hide
            # samples if a future engine has a different schedule model.
            preview_sample_count = complete_iq.size
        iq = complete_iq[:preview_sample_count]
        single_repeat_preview = preview_sample_count < complete_iq.size
        time_us = np.arange(iq.size) / result.sample_rate_hz * 1e6
        power_dbfs = 20.0 * np.log10(np.maximum(np.abs(iq), 1e-6))
        frequency_khz = _instantaneous_frequency_khz(iq, result.sample_rate_hz)
        self.iq_waveform_plot.clear()
        self.iq_waveform_legend.clear()
        self.power_plot.clear()
        self.frequency_plot.clear()
        self.iq_waveform_plot.plot(time_us, iq.real, pen=TRACE_COLOR, name="I")
        self.iq_waveform_plot.plot(time_us, iq.imag, pen=ACCENT_COLOR, name="Q")
        self.power_plot.plot(time_us, power_dbfs, pen=TRACE_COLOR)
        self.frequency_plot.plot(time_us[1:], frequency_khz, pen=TRACE_COLOR)
        for plot in (
            self.iq_waveform_plot,
            self.power_plot,
            self.frequency_plot,
        ):
            self._add_field_guides(
                plot,
                result,
                include_minor=True,
                include_labels=True,
                preview_stop_sample=preview_sample_count,
                single_repeat_preview=single_repeat_preview,
            )

        fft_size = min(
            16384, max(256, 1 << (max(1, iq.size) - 1).bit_length())
        )
        spectrum_input = iq[:fft_size]
        if spectrum_input.size < fft_size:
            spectrum_input = np.pad(
                spectrum_input, (0, fft_size - spectrum_input.size)
            )
        spectrum = np.fft.fftshift(
            np.fft.fft(spectrum_input * np.hanning(fft_size))
        )
        spectrum_dbfs = 20.0 * np.log10(
            np.maximum(np.abs(spectrum) / fft_size, 1e-12)
        )
        frequency_mhz = (
            np.fft.fftshift(np.fft.fftfreq(fft_size, 1.0 / result.sample_rate_hz))
            / 1e6
        )
        self.spectrum_plot.clear()
        self.spectrum_plot.plot(frequency_mhz, spectrum_dbfs, pen=TRACE_COLOR)

        self.constellation_plot.clear()
        self.constellation_legend.clear()
        constellation_colors = (
            TRACE_COLOR,
            ACCENT_COLOR,
            FIELD_BOUNDARY_COLOR,
            FIELD_MINOR_BOUNDARY_COLOR,
        )
        for index, trace in enumerate(result.constellation_traces):
            symbols = np.asarray(trace.symbols, dtype=np.complex128).reshape(-1)
            if symbols.size > 4096:
                display_indices = np.linspace(
                    0, symbols.size - 1, 4096, dtype=np.int64
                )
                symbols = symbols[display_indices]
            color = constellation_colors[index % len(constellation_colors)]
            self.constellation_plot.plot(
                symbols.real,
                symbols.imag,
                pen=None,
                symbol="o",
                symbolSize=7,
                symbolBrush=color,
                symbolPen=None,
                name=trace.label,
            )
        has_constellation = bool(result.constellation_traces)
        self.constellation_legend.setVisible(has_constellation)
        self.constellation_plot.setTitle(
            "Mapped symbols before pulse shaping; non-I/Q sections omitted"
            if has_constellation
            else "No I/Q symbol constellation for this waveform"
        )
        self.constellation_plot.setRange(
            xRange=[-1.25, 1.25], yRange=[-1.25, 1.25], padding=0.0
        )
        self._remember_plot_scales()
        active_x_range = _preview_active_x_range_us(result, preview_sample_count)
        for name, plot in (
            ("iq_waveform", self.iq_waveform_plot),
            ("power", self.power_plot),
            ("frequency", self.frequency_plot),
        ):
            plot.setXRange(*active_x_range, padding=0.0)
            self._plot_initial_ranges[name] = (
                list(active_x_range),
                list(plot.viewRange()[1]),
            )

    @staticmethod
    def _add_field_guides(
        plot: pg.PlotWidget,
        result: GenerationResult,
        *,
        include_minor: bool,
        include_labels: bool,
        preview_stop_sample: int | None = None,
        single_repeat_preview: bool = False,
    ) -> None:
        for boundary in result.field_boundaries:
            if (
                preview_stop_sample is not None
                and boundary.start_sample >= preview_stop_sample
            ):
                continue
            if boundary.level > 0 and not include_minor:
                continue
            is_major = boundary.level == 0
            color = (
                FIELD_BOUNDARY_COLOR if is_major else FIELD_MINOR_BOUNDARY_COLOR
            )
            pen = pg.mkPen(
                color,
                width=1.25 if is_major else 1.0,
                style=(
                    QtCore.Qt.PenStyle.DashLine
                    if is_major
                    else QtCore.Qt.PenStyle.DotLine
                ),
            )
            start_us = boundary.start_sample / result.sample_rate_hz * 1e6
            label = boundary.name if include_labels else None
            if label is not None and single_repeat_preview:
                label = re.sub(r" \[1\]$", "", label)
            label_options = None
            if label is not None:
                label_options = {
                    "position": 0.92 if is_major else 0.08,
                    "color": color,
                    "fill": (0, 0, 0, 150),
                    # Keep every label on the same side of its boundary. The
                    # pyqtgraph default swaps anchors at the view center.
                    "anchors": [(0.0, 0.5), (0.0, 0.5)],
                }
            line = pg.InfiniteLine(
                start_us,
                angle=90,
                pen=pen,
                span=(0.0, 1.0) if is_major else (0.0, 0.22),
                label=label,
                labelOpts=label_options,
            )
            plot.addItem(line)

        packet_ranges = result.metadata.get("packet_ranges_samples", ())
        visible_packet_ranges = [
            packet_range
            for packet_range in packet_ranges
            if isinstance(packet_range, (tuple, list))
            and len(packet_range) == 2
            and (
                preview_stop_sample is None
                or int(packet_range[0]) < preview_stop_sample
            )
        ]
        for index, packet_range in enumerate(visible_packet_ranges):
            stop_sample = int(packet_range[1])
            stop_us = stop_sample / result.sample_rate_hz * 1e6
            suffix = "" if len(visible_packet_ranges) == 1 else f" [{index + 1}]"
            line = pg.InfiniteLine(
                stop_us,
                angle=90,
                pen=pg.mkPen(PACKET_END_COLOR, width=1.75),
                span=(0.0, 1.0),
                label=f"Packet End{suffix}" if include_labels else None,
                labelOpts=(
                    {
                        # Align Packet End with the major-field labels.
                        "position": 0.92,
                        "color": PACKET_END_COLOR,
                        "fill": (0, 0, 0, 170),
                        "anchors": [(0.0, 0.5), (0.0, 0.5)],
                    }
                    if include_labels
                    else None
                ),
            )
            plot.addItem(line)

    def _open_project(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Pluto VSG Project",
            "",
            "Pluto VSG Project (*.pvsg.json);;JSON (*.json)",
        )
        if not path:
            return
        previous_signature = self._pluto_configuration_signature()
        try:
            self.project = load_project(path)
        except ValueError as error:
            QtWidgets.QMessageBox.critical(self, "Open Project", str(error))
            return
        self._frequency_selections[self.project.standard] = (
            default_frequency_selection(self.project)
        )
        self.project_path = Path(path)
        self.undo_stack.clear()
        self._refresh_project_view()
        self.generate_waveform()
        self._configuration_maybe_changed(previous_signature)

    def _save_project(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Pluto VSG Project",
            (
                str(self.project_path)
                if self.project_path is not None
                else "waveform.pvsg.json"
            ),
            "Pluto VSG Project (*.pvsg.json)",
        )
        if not path:
            return
        self.project_path = Path(path)
        save_project(self.project_path, self.project)
        self.statusBar().showMessage(f"Saved {self.project_path.name}")

    def _export_npz(self) -> None:
        if self.result is None:
            self.generate_waveform()
        if self.result is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export IQ", "waveform.npz", "NumPy IQ (*.npz)"
        )
        if path:
            save_npz(path, self.result, replace(self.project, repeat_count=1))
            self.statusBar().showMessage(f"Exported {Path(path).name}")

    def _export_iq_tar(self) -> None:
        if self.result is None:
            self.generate_waveform()
        if self.result is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export R&S IQ TAR",
            "waveform.iq.tar",
            "R&S IQ TAR (*.iq.tar)",
        )
        if path:
            save_iq_tar(path, self.result, replace(self.project, repeat_count=1))
            self.statusBar().showMessage(f"Exported {Path(path).name}")

    def _export_wv(self) -> None:
        if self.result is None:
            self.generate_waveform()
        if self.result is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export R&S WV",
            "waveform.wv",
            "R&S ARB Waveform (*.wv)",
        )
        if not path:
            return
        try:
            save_wv(path, self.result, replace(self.project, repeat_count=1))
        except (OSError, ValueError) as error:
            QtWidgets.QMessageBox.critical(self, "Export R&S WV", str(error))
            return
        self.statusBar().showMessage(f"Exported {Path(path).name}")

    def _current_pluto_settings(self) -> PlutoTransmitSettings:
        bandwidth_hz = min(
            56_000_000.0, max(200_000.0, self.project.sample_rate_hz)
        )
        if self.result is None:
            active_rms_dbfs = 0.0
            peak_dbfs = 0.0
        else:
            level_metrics = generation_result_iq_levels(self.result)
            active_rms_dbfs = level_metrics.active_rms_dbfs
            peak_dbfs = level_metrics.peak_dbfs
        hardware_gain_db = pluto_hardware_gain_for_output_power_dbm(
            self._pluto_output_power_dbm,
            self._pluto_digital_backoff_db,
            self.project.center_frequency_hz,
            active_rms_dbfs,
        )
        return PlutoTransmitSettings(
            center_frequency_hz=self.project.center_frequency_hz,
            sample_rate_hz=self.project.sample_rate_hz,
            rf_bandwidth_hz=bandwidth_hz,
            hardware_gain_db=hardware_gain_db,
            digital_backoff_db=self._pluto_digital_backoff_db,
            connection_uri=self._pluto_uri or None,
            lead_in_guard_s=self._pluto_lead_in_guard_s,
            dma_preroll_s=self._pluto_dma_preroll_s,
            stop_guard_s=self._pluto_stop_guard_s,
            burst_count=self.project.repeat_count,
            output_power_dbm=self._pluto_output_power_dbm,
            playback_mode=self._pluto_playback_mode,
            waveform_active_rms_dbfs=active_rms_dbfs,
            waveform_peak_dbfs=peak_dbfs,
            single_period_template=True,
        )

    def _pluto_configuration_signature(self) -> tuple[object, ...]:
        """Identify settings that require AD936x reconfiguration/calibration."""

        settings = self._current_pluto_settings()
        return (
            settings.connection_uri or "",
            int(round(settings.center_frequency_hz)),
            int(round(settings.sample_rate_hz)),
            int(round(settings.rf_bandwidth_hz)),
        )

    def _configuration_maybe_changed(
        self, previous_signature: tuple[object, ...]
    ) -> None:
        if previous_signature == self._pluto_configuration_signature():
            return
        self._pluto_prepared_signature = None
        self._update_vsg_control_labels()
        self.statusBar().showMessage(
            "ADALM-Pluto configuration changed; preparation required"
        )

    def _edit_pluto_settings(self) -> None:
        previous_signature = self._pluto_configuration_signature()
        dialog = _PlutoOutputDialog(
            self._current_pluto_settings(), self.project.repeat_count, self
        )
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        settings = dialog.settings
        self._pluto_uri = settings.connection_uri or ""
        self._update_pluto_window_title()
        self._pluto_digital_backoff_db = settings.digital_backoff_db
        self._pluto_lead_in_guard_s = settings.lead_in_guard_s
        self._pluto_dma_preroll_s = settings.dma_preroll_s
        self._pluto_stop_guard_s = settings.stop_guard_s
        minimum_power, maximum_power = self._power_limits_dbm()
        self._pluto_output_power_dbm = min(
            maximum_power, max(minimum_power, self._pluto_output_power_dbm)
        )
        preferences = self._preferences
        preferences.setValue("pluto_tx/uri", self._pluto_uri)
        # Preserve the derived legacy value for older application versions.
        preferences.setValue(
            "pluto_tx/hardware_gain_db", settings.resolved_hardware_gain_db
        )
        preferences.setValue(
            "pluto_tx/digital_backoff_db", self._pluto_digital_backoff_db
        )
        preferences.setValue(
            "pluto_tx/lead_in_guard_s", self._pluto_lead_in_guard_s
        )
        preferences.setValue(
            "pluto_tx/dma_preroll_s", self._pluto_dma_preroll_s
        )
        preferences.setValue("pluto_tx/stop_guard_s", self._pluto_stop_guard_s)
        configuration_changed = (
            previous_signature != self._pluto_configuration_signature()
        )
        if configuration_changed:
            self._pluto_prepared_signature = None
        self._update_vsg_control_labels()
        self.statusBar().showMessage(
            "ADALM-Pluto instrument settings saved"
            + ("; calibration required" if configuration_changed else "")
        )

    def _start_pluto_preparation(self) -> None:
        if self._prepare_thread is not None or self._tx_thread is not None:
            return
        try:
            backend = PlutoOutputBackend(self._current_pluto_settings())
        except ValueError as error:
            QtWidgets.QMessageBox.warning(self, "Pluto Preparation", str(error))
            return
        signature = self._pluto_configuration_signature()
        worker = _PlutoPrepareWorker(backend)
        thread = QtCore.QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._pluto_preparation_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._pluto_prepare_thread_finished)
        self._prepare_worker = worker
        self._prepare_thread = thread
        self._preparing_signature = signature
        self._calibration_in_progress = True
        self._set_pluto_busy(preparing=True, transmitting=False)
        self.statusBar().showMessage(
            "Preparing ADALM-Pluto: muted configuration and explicit TX calibration..."
        )
        thread.start()

    @QtCore.Slot(bool, str)
    def _pluto_preparation_finished(self, success: bool, message: str) -> None:
        self._calibration_in_progress = False
        if success and self._preparing_signature == self._pluto_configuration_signature():
            self._pluto_prepared_signature = self._preparing_signature
            self.statusBar().showMessage(message)
        else:
            self._pluto_prepared_signature = None
            if success:
                message = "Configuration changed during preparation; prepare again"
            self.statusBar().showMessage(f"Pluto preparation failed: {message}")
            if not self._close_after_tx:
                QtWidgets.QMessageBox.critical(self, "Pluto Preparation", message)
        self._set_pluto_busy(preparing=False, transmitting=False)

    @QtCore.Slot()
    def _pluto_prepare_thread_finished(self) -> None:
        self._prepare_worker = None
        self._prepare_thread = None
        self._preparing_signature = None
        if self._close_after_tx:
            self._close_after_tx = False
            self.close()

    def _start_pluto_transmission(self) -> None:
        if self._tx_thread is not None or self._prepare_thread is not None:
            return
        if self._pluto_prepared_signature != self._pluto_configuration_signature():
            QtWidgets.QMessageBox.warning(
                self,
                "Pluto Transmission",
                "ADALM-Pluto is not READY for the current RF/baseband settings. "
                "Run 'Prepare / Calibrate ADALM-Pluto' first. Transmit never "
                "changes these settings or launches calibration automatically.",
            )
            return
        if self.result is None:
            self.generate_waveform()
        if self.result is None:
            return
        issues = validate_project(self.project)
        if issues:
            QtWidgets.QMessageBox.warning(
                self,
                "Pluto Transmission",
                "\n".join(f"{issue.path}: {issue.message}" for issue in issues),
            )
            return
        try:
            backend = PlutoOutputBackend(self._current_pluto_settings())
        except ValueError as error:
            QtWidgets.QMessageBox.warning(self, "Pluto Transmission", str(error))
            return
        worker = _PlutoTransmitWorker(backend, self.result)
        thread = QtCore.QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._pluto_transmission_finished)
        worker.first_tx_completed.connect(self._pluto_first_tx_completed)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._pluto_thread_finished)
        self._tx_worker = worker
        self._tx_thread = thread
        self._rf_enabled = False
        self._rf_transfer_pending = True
        self._set_pluto_busy(preparing=False, transmitting=True)
        if self._pluto_playback_mode is PlutoPlaybackMode.CONTINUOUS:
            period_samples = int(
                self.result.metadata.get(
                    "period_sample_count", self.result.iq.size
                )
            )
            period_ms = 1e3 * period_samples / self.result.sample_rate_hz
            self.statusBar().showMessage(
                "Starting continuous Pluto TX: "
                f"{period_ms:.3f} ms period, "
                f"{self.project.center_frequency_hz / 1e6:.6f} MHz; use Stop to end"
            )
        else:
            period_samples = int(
                self.result.metadata.get(
                    "period_sample_count", self.result.iq.size
                )
            )
            duration_ms = (
                1e3
                * period_samples
                * self.project.repeat_count
                / self.result.sample_rate_hz
            )
            self.statusBar().showMessage(
                f"Starting finite Pluto TX: {self.project.repeat_count} packet(s), "
                f"{duration_ms:.3f} ms, "
                f"{self.project.center_frequency_hz / 1e6:.6f} MHz"
            )
        thread.start()

    def _start_pluto_cw(self) -> None:
        """Start a carrier at the current Pluto frequency and output level."""

        if self._tx_thread is not None or self._prepare_thread is not None:
            return
        if self._pluto_prepared_signature != self._pluto_configuration_signature():
            QtWidgets.QMessageBox.warning(
                self,
                "Pluto CW",
                "ADALM-Pluto is not READY for the current RF/baseband settings. "
                "Run 'Prepare / Calibrate ADALM-Pluto' first. CW start never "
                "changes RF/baseband settings or launches calibration.",
            )
            return
        try:
            settings = replace(
                self._current_pluto_settings(),
                burst_count=1,
                playback_mode=PlutoPlaybackMode.CONTINUOUS,
            )
            result = _cw_generation_result(settings.sample_rate_hz)
            backend = PlutoOutputBackend(settings)
        except ValueError as error:
            QtWidgets.QMessageBox.warning(self, "Pluto CW", str(error))
            return

        worker = _PlutoTransmitWorker(backend, result)
        thread = QtCore.QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._pluto_transmission_finished)
        worker.first_tx_completed.connect(self._pluto_first_tx_completed)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._pluto_thread_finished)
        self._tx_worker = worker
        self._tx_thread = thread
        self._rf_enabled = False
        self._rf_transfer_pending = True
        self._set_pluto_busy(preparing=False, transmitting=True)
        self.statusBar().showMessage(
            "Starting Pluto CW: "
            f"{settings.center_frequency_hz / 1e6:.6f} MHz, "
            f"target {float(settings.output_power_dbm):+.2f} dBm; use Stop to end"
        )
        thread.start()

    def _stop_pluto_transmission(self) -> None:
        if self._tx_worker is None:
            return
        self._tx_worker.cancel()
        self.pluto_stop_action.setEnabled(False)
        self.rf_button.setEnabled(False)
        self.statusBar().showMessage("Stopping Pluto transmission...")

    @QtCore.Slot()
    def _pluto_first_tx_completed(self) -> None:
        """Show RF ON only after the first host-to-Pluto TX call succeeds."""

        if self._tx_thread is None:
            return
        self._rf_transfer_pending = False
        self._rf_enabled = True
        self._update_vsg_control_labels()

    @QtCore.Slot(bool, str)
    def _pluto_transmission_finished(self, success: bool, message: str) -> None:
        self._rf_enabled = False
        self._rf_transfer_pending = False
        if success:
            self.statusBar().showMessage(message)
        else:
            self.statusBar().showMessage(f"Pluto transmission failed: {message}")
            if not self._close_after_tx:
                QtWidgets.QMessageBox.critical(self, "Pluto Transmission", message)
        self._set_pluto_busy(preparing=False, transmitting=False)

    @QtCore.Slot()
    def _pluto_thread_finished(self) -> None:
        self._tx_worker = None
        self._tx_thread = None
        if self._close_after_tx:
            self._close_after_tx = False
            self.close()

    def _set_pluto_busy(self, *, preparing: bool, transmitting: bool) -> None:
        main_scroll = (
            self.vsg_main_control_page.verticalScrollBar()
            if hasattr(self, "vsg_main_control_page")
            else None
        )
        main_scroll_value = main_scroll.value() if main_scroll is not None else 0
        if preparing and hasattr(self, "rf_button") and self.rf_button.hasFocus():
            # Disabling the clicked Calibration button can make QScrollArea
            # chase the next focusable child. Drop that focus before changing
            # enabled states and restore the user's exact scroll position.
            self.rf_button.clearFocus()
        active = preparing or transmitting
        for action in (
            self.new_action,
            self.new_le_action,
            self.new_hdt_action,
            self.new_wifi_action,
            self.new_dect_action,
            self.open_action,
            self.save_action,
            self.settings_action,
            self.generate_action,
            self.pluto_settings_action,
            self.pluto_prepare_action,
            self.pluto_transmit_action,
            self.pluto_cw_action,
        ):
            action.setEnabled(not active)
        self.pluto_stop_action.setEnabled(transmitting)
        if hasattr(self, "rf_button"):
            self.edit_settings_button.setEnabled(not active)
            self.verify_packet_button.setEnabled(
                not preparing and self.result is not None and self.result.packet_bits is not None
            )
            self.rf_button.setEnabled(not preparing)
            self.mod_button.setEnabled(not active)
            self.continuous_button.setEnabled(not active)
            self.repetitions_button.setEnabled(not active)
            self.packet_fields_action.setEnabled(not active and self.project.wifi is None)
            self.frequency_button.setEnabled(not active)
            self.frequency_settings_button.setEnabled(not active)
            self.project_button.setEnabled(not active)
            self.instrument_settings_button.setEnabled(not active)
            self.power_button.setEnabled(not preparing)
            self.power_up_button.setEnabled(not preparing)
            self.power_down_button.setEnabled(not preparing)
            self.power_step_button.setEnabled(not preparing)
            self._update_vsg_control_labels()
        if main_scroll is not None:
            main_scroll.setValue(main_scroll_value)
            QtCore.QTimer.singleShot(
                0,
                lambda bar=main_scroll, value=main_scroll_value: bar.setValue(
                    value
                ),
            )

    def _show_validation(self) -> None:
        issues = validate_project(self.project)
        text = (
            "Project settings are valid."
            if not issues
            else "\n".join(f"{issue.path}: {issue.message}" for issue in issues)
        )
        QtWidgets.QMessageBox.information(self, "Project Validation", text)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self._prepare_thread is not None:
            self._close_after_tx = True
            self.statusBar().showMessage(
                "Waiting for safe Pluto preparation/calibration completion before closing..."
            )
            event.ignore()
            return
        if self._tx_thread is not None:
            self._close_after_tx = True
            if not self._shutdown_stop_requested:
                self._shutdown_stop_requested = True
                self._stop_pluto_transmission()
            self.statusBar().showMessage(
                "Stopping Pluto transmission safely before closing..."
            )
            event.ignore()
            return
        self._shutdown_stop_requested = False
        self._save_startup_state()
        for dock in self.workspace.findChildren(QtWidgets.QDockWidget):
            if dock.isFloating():
                dock.hide()
        super().closeEvent(event)
