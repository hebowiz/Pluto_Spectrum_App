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

from pluto_sa.vsa.profiles.bluetooth_br import header_error_check
from pluto_sa.vsa.ui.measurement_chrome import (
    install_measurement_plot_menu,
    make_measurement_plot,
)
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
    BluetoothLEWaveformEngine,
    GenerationResult,
)
from pluto_vsg.export import save_iq_tar, save_npz, save_wv
from pluto_vsg.model import (
    BluetoothLEPayloadType,
    BluetoothLEPayloadSourceKind,
    BluetoothLEPhy,
    BluetoothPacketKind,
    PayloadSourceKind,
    StandardProfile,
    WaveformProject,
    bluetooth_packet_is_edr,
    bluetooth_packet_properties,
    effective_post_idle_symbols,
    effective_period_symbols,
    minimum_period_symbols,
    validate_project,
)
from pluto_vsg.persistence import load_project, save_project
from pluto_vsg.profiles import (
    bluetooth_br_edr_project,
    bluetooth_br_fields,
    bluetooth_le_fields,
    bluetooth_le_project,
    bluetooth_le_test_project,
    apply_bluetooth_le_rf_test_preset,
)
from pluto_vsg.ui.style import (
    ACCENT_COLOR,
    FIELD_BOUNDARY_COLOR,
    FIELD_MINOR_BOUNDARY_COLOR,
    PACKET_END_COLOR,
    TRACE_COLOR,
    panel_title_font,
)
from pluto_vsg.ui.composer_view import PacketComposerView


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


class _Panel(QtWidgets.QGroupBox):
    def __init__(self, title: str, child: QtWidgets.QWidget) -> None:
        super().__init__(title)
        self.setFont(panel_title_font(self.font()))
        child_font = QtGui.QFont(child.font())
        child_font.setBold(False)
        child.setFont(child_font)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 8, 4, 4)
        layout.addWidget(child)


class _BluetoothLESettingsDialog(QtWidgets.QDialog):
    """Edit an uncoded LE Direct Test Mode packet."""

    def __init__(self, project: WaveformProject, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        settings = project.bluetooth_le
        if settings is None:
            raise ValueError("Bluetooth LE settings are required")
        self._project = project
        self.setWindowTitle("Bluetooth LE Packet Settings")
        form = QtWidgets.QFormLayout()
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
        self.length_spin = QtWidgets.QSpinBox()
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
        self.center_spin = self._double_spin(0.0, 6000.0, project.center_frequency_hz / 1e6, 6)
        self.sps_spin = self._integer_spin(4, 64, project.samples_per_symbol)
        self.repeat_spin = self._integer_spin(1, 1000, project.repeat_count)
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
        for label, widget in (
            ("Project Name", self.name_edit),
            ("PHY", self.phy_combo),
            ("RF Test Payload Preset", preset_row),
            ("Preamble [air-order bits]", self.preamble_edit),
            ("Access Address / Sync [air-order bits]", self.sync_edit),
            ("PDU Header [air-order bits]", self.header_edit),
            ("Payload Length [byte]", self.length_spin),
            ("Payload Source", self.payload_source_combo),
            ("Payload Pattern [bin]", self.payload_pattern_edit),
            ("CRC-24", self.crc_check),
            ("CRCInit [hex]", self.crc_init_edit),
            ("Whitening", self.whitening_check),
            ("Whitening Channel Index", self.whitening_channel_spin),
            ("Center Frequency [MHz]", self.center_spin),
            ("Samples / Symbol", self.sps_spin),
            ("Repeat Count", self.repeat_spin),
            ("FSK Deviation [kHz]", self.deviation_spin),
            ("Gaussian B*T", self.bt_spin),
            ("Pre Idle [symbols]", self.pre_idle_spin),
            ("Period [symbols]", self.period_spin),
            ("Post Idle [symbols] (reference)", self.post_idle_value),
            ("Ramp Up [symbols]", self.rise_spin),
            ("Ramp Up Start rel. Packet [symbols]", self.rise_delay_spin),
            ("Ramp Down [symbols]", self.fall_spin),
            ("Ramp Down Start rel. Packet End [symbols]", self.fall_delay_spin),
            ("Ramp Shape", self.ramp_combo),
        ):
            form.addRow(label, widget)
        note = QtWidgets.QLabel(
            "All packet fields remain editable. Applying an RF Test Packet preset "
            "loads the Core test Sync Word/header/payload, CRCInit 0x555555, "
            "Whitening Off and the standard packet interval into these controls."
        )
        note.setWordWrap(True)
        form.addRow(note)
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
        layout.addLayout(form)
        layout.addWidget(buttons)
        self.resize(620, 650)
        self.phy_combo.currentIndexChanged.connect(self._phy_changed)
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
        control = QtWidgets.QDoubleSpinBox()
        control.setRange(minimum, maximum)
        control.setDecimals(decimals)
        control.setValue(value)
        control.setKeyboardTracking(False)
        return control

    @staticmethod
    def _integer_spin(minimum: int, maximum: int, value: int) -> QtWidgets.QSpinBox:
        control = QtWidgets.QSpinBox()
        control.setRange(minimum, maximum)
        control.setValue(value)
        return control

    def _phy_changed(self) -> None:
        phy = BluetoothLEPhy(self.phy_combo.currentData())
        self.deviation_spin.setValue(250.0 if phy == BluetoothLEPhy.LE_1M else 500.0)

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
        self.post_idle_value.setText(f"{post_idle:.3f}")

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
        self.period_spin.setMinimum(self._minimum_period_symbols)
        self._update_post_idle_reference()

    def _accept_settings(self) -> None:
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
            center_frequency_hz=self.center_spin.value() * 1e6,
            samples_per_symbol=self.sps_spin.value(),
            sample_rate_hz=symbol_rate_hz * self.sps_spin.value(),
            repeat_count=self.repeat_spin.value(),
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


class _BluetoothSettingsDialog(QtWidgets.QDialog):
    def __init__(self, project: WaveformProject, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        settings = project.bluetooth_br
        if settings is None:
            raise ValueError("Bluetooth settings are required")
        self._project = project
        self.setWindowTitle("Bluetooth BR / EDR Settings")
        form_widget = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(form_widget)

        self.name_edit = QtWidgets.QLineEdit(project.name)
        self.center_spin = self._double_spin(
            0.0, 6000.0, project.center_frequency_hz / 1e6, 6
        )
        self.sps_spin = self._integer_spin(4, 64, project.samples_per_symbol)
        self.repeat_spin = self._integer_spin(1, 1000, project.repeat_count)
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
        self.hec_value = QtWidgets.QLabel()
        self.hec_value.setToolTip("Automatically calculated from Header and UAP")
        self.payload_length_spin = self._integer_spin(
            0, 1021, settings.payload_length_bytes
        )
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

        for label, widget in (
            ("Project Name", self.name_edit),
            ("Center Frequency [MHz]", self.center_spin),
            ("Samples / Symbol", self.sps_spin),
            ("Repeat Count", self.repeat_spin),
            ("LAP [hex]", self.lap_edit),
            ("UAP [hex]", self.uap_edit),
            ("CLK 6-1 [hex]", self.clock_edit),
        ):
            form.addRow(label, widget)

        header_group = QtWidgets.QGroupBox("Packet Header")
        header_form = QtWidgets.QFormLayout(header_group)
        for label, widget in (
            ("LT_ADDR", self.lt_addr_spin),
            ("Packet Type / TYPE", self.packet_type_combo),
            ("FLOW", self.flow_combo),
            ("ARQN", self.arqn_combo),
            ("SEQN", self.seqn_combo),
            ("HEC", self.hec_value),
        ):
            header_form.addRow(label, widget)
        form.addRow(header_group)

        for label, widget in (
            ("RF Test Payload Preset", rf_test_row),
            ("Payload Length [byte]", self.payload_length_spin),
            ("Payload Source", self.payload_source_combo),
            ("Source Behavior", self.payload_source_help),
            ("Payload Data [bin]", self.pattern_edit),
            ("Whitening", self.whitening_check),
            ("FSK Deviation [kHz]", self.deviation_spin),
            ("Carrier Offset [kHz]", self.cfo_spin),
            ("Gaussian B*T", self.bt_spin),
            ("EDR Guard [symbols]", self.edr_guard_spin),
            ("EDR Guard Power rel. GFSK", self.edr_guard_power_spin),
            ("EDR Guard Ramp In [symbols]", self.edr_guard_ramp_in_spin),
            ("EDR Guard Ramp Out [symbols]", self.edr_guard_ramp_out_spin),
            ("EDR Guard Ramp Shape", self.edr_guard_ramp_shape_combo),
            ("EDR SRRC Roll-off", self.edr_rolloff_spin),
            ("EDR Power rel. GFSK [dB]", self.edr_power_spin),
            ("Pre Idle [symbols]", self.pre_idle_spin),
            ("Period [symbols]", self.period_spin),
            ("Post Idle [symbols] (reference)", self.post_idle_value),
            ("Ramp Up [symbols]", self.rise_spin),
            ("Ramp Up Start rel. Packet [symbols]", self.rise_delay_spin),
            ("Ramp Down [symbols]", self.fall_spin),
            ("Ramp Down Start rel. Packet End [symbols]", self.fall_delay_spin),
            ("Ramp Shape", self.ramp_combo),
            ("Ramp Timing", ramp_timing_help),
        ):
            form.addRow(label, widget)

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
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setWidget(form_widget)
        layout.addWidget(scroll)
        layout.addWidget(buttons)
        self.resize(620, 780)
        self.uap_edit.textChanged.connect(self._update_header_preview)
        self.lt_addr_spin.valueChanged.connect(self._update_header_preview)
        self.flow_combo.currentIndexChanged.connect(self._update_header_preview)
        self.arqn_combo.currentIndexChanged.connect(self._update_header_preview)
        self.seqn_combo.currentIndexChanged.connect(self._update_header_preview)
        self.packet_type_combo.currentIndexChanged.connect(
            self._packet_type_changed
        )
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

    @staticmethod
    def _double_spin(
        minimum: float, maximum: float, value: float, decimals: int
    ) -> QtWidgets.QDoubleSpinBox:
        control = QtWidgets.QDoubleSpinBox()
        control.setRange(minimum, maximum)
        control.setDecimals(decimals)
        control.setValue(value)
        control.setKeyboardTracking(False)
        return control

    @staticmethod
    def _integer_spin(minimum: int, maximum: int, value: int) -> QtWidgets.QSpinBox:
        control = QtWidgets.QSpinBox()
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
        self.hec_value.setText(f"0x{header_error_check(header_bits, uap):02X} (auto)")

    def _update_post_idle_reference(self) -> None:
        post_idle = max(0.0, self.period_spin.value() - self._minimum_period_symbols)
        self.post_idle_value.setText(f"{post_idle:.3f}")

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
        self.period_spin.setMinimum(self._minimum_period_symbols)
        self._update_post_idle_reference()

    def _accept_settings(self) -> None:
        try:
            lap = int(self.lap_edit.text().strip(), 16)
            uap = int(self.uap_edit.text().strip(), 16)
            clock = int(self.clock_edit.text().strip(), 16)
        except ValueError:
            QtWidgets.QMessageBox.warning(
                self, "Bluetooth Settings", "LAP, UAP and clock must be hexadecimal."
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
            payload_length_bytes=self.payload_length_spin.value(),
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
            center_frequency_hz=self.center_spin.value() * 1e6,
            sample_rate_hz=self.sps_spin.value() * 1e6,
            samples_per_symbol=self.sps_spin.value(),
            repeat_count=self.repeat_spin.value(),
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
        self.setWindowTitle("ADALM-Pluto Output Settings")
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
        self.bandwidth_spin = QtWidgets.QDoubleSpinBox()
        self.bandwidth_spin.setRange(0.2, 56.0)
        self.bandwidth_spin.setDecimals(3)
        self.bandwidth_spin.setSuffix(" MHz")
        self.bandwidth_spin.setValue(settings.rf_bandwidth_hz / 1e6)
        self.bandwidth_spin.setKeyboardTracking(False)
        self.output_power_spin = QtWidgets.QDoubleSpinBox()
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
            )
        )
        self.applied_gain_label = QtWidgets.QLabel()
        self.digital_backoff_combo.currentIndexChanged.connect(
            lambda _index: self._update_output_level_constraints()
        )
        self.output_power_spin.valueChanged.connect(self._update_applied_gain_label)
        self._update_output_level_constraints(initial_output_power_dbm)
        self.lead_in_guard_spin = QtWidgets.QDoubleSpinBox()
        self.lead_in_guard_spin.setRange(0.0, 1000.0)
        self.lead_in_guard_spin.setDecimals(3)
        self.lead_in_guard_spin.setSuffix(" ms")
        self.lead_in_guard_spin.setValue(settings.lead_in_guard_s * 1e3)
        self.lead_in_guard_spin.setKeyboardTracking(False)
        self.dma_preroll_spin = QtWidgets.QDoubleSpinBox()
        self.dma_preroll_spin.setRange(0.0, 1000.0)
        self.dma_preroll_spin.setDecimals(3)
        self.dma_preroll_spin.setSuffix(" ms")
        self.dma_preroll_spin.setValue(settings.dma_preroll_s * 1e3)
        self.dma_preroll_spin.setKeyboardTracking(False)
        self.stop_guard_spin = QtWidgets.QDoubleSpinBox()
        self.stop_guard_spin.setRange(10.0, 5000.0)
        self.stop_guard_spin.setDecimals(3)
        self.stop_guard_spin.setSuffix(" ms")
        self.stop_guard_spin.setValue(settings.stop_guard_s * 1e3)
        self.stop_guard_spin.setKeyboardTracking(False)
        form.addRow("Playback Mode", self.playback_mode_combo)
        form.addRow("Connection URI", selector_row)
        form.addRow(
            "Center Frequency",
            QtWidgets.QLabel(f"{settings.center_frequency_hz / 1e6:.6f} MHz (Project)"),
        )
        form.addRow(
            "Sample Rate",
            QtWidgets.QLabel(f"{settings.sample_rate_hz / 1e6:.3f} MS/s (Project)"),
        )
        form.addRow("TX RF Bandwidth", self.bandwidth_spin)
        form.addRow("RF Output Level", self.output_power_spin)
        form.addRow("Digital Backoff", self.digital_backoff_combo)
        form.addRow("Applied Tx Gain", self.applied_gain_label)
        form.addRow("Muted LO Settling Time", self.lead_in_guard_spin)
        self.dma_preroll_label = QtWidgets.QLabel("DMA Pre-roll")
        form.addRow(self.dma_preroll_label, self.dma_preroll_spin)
        self.stop_guard_label = QtWidgets.QLabel("Completion Margin")
        form.addRow(self.stop_guard_label, self.stop_guard_spin)
        self.packet_count_label = QtWidgets.QLabel(str(packet_count))
        form.addRow("Packets per transmission", self.packet_count_label)
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
        self.resize(620, 410)

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
            "RF Output Level uses a provisional 2440 MHz calibration measured "
            "with this Pluto and a constant-envelope FSK packet. Frequency "
            "response, device variation and residual nonlinearity are not yet "
            "corrected. Verify the conducted level when accuracy matters. "
            "Digital Backoff 0 dB drives the DMA/DAC path at full scale; use "
            "-3 or -6 dB when additional linearity margin is required. "
        )
        if continuous:
            detail = (
                "Continuous repeats exactly the first generated packet period "
                "with Pluto cyclic DMA until Stop is requested. Project Repeat "
                "Count, DMA Pre-roll and Completion Margin do not alter the "
                "continuous cycle. Stop may interrupt a packet. Residual LO "
                "leakage is not a calibrated RF-off state."
            )
        else:
            detail = (
                "Finite submits the complete requested packet schedule once in "
                "a non-cyclic DMA buffer. DMA Pre-roll protects the first packet "
                "from the DMA/DAC source-start transient, and Completion Margin "
                "defers cleanup after submission. Residual LO leakage is not a "
                "calibrated RF-off state."
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
        )
        self.applied_gain_label.setText(f"{gain_db:+.2f} dB (estimated)")

    def _accept_settings(self) -> None:
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
            self.backend.start()
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
        log_path = Path(__file__).resolve().parents[2] / "pluto_vsg_tx_trace.log"
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
        log_path = Path(__file__).resolve().parents[2] / "pluto_vsg_tx_trace.log"
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

    def __init__(self, project: WaveformProject | None = None) -> None:
        super().__init__()
        self.project = project or bluetooth_br_edr_project()
        self.result: GenerationResult | None = None
        self.project_path: Path | None = None
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
        self._close_after_tx = False
        self._shutdown_stop_requested = False
        self.undo_stack = QtGui.QUndoStack(self)
        self._selected_composer_block: ComposerBlock | None = None
        preferences = QtCore.QSettings("PlutoSpectrumApp", "PlutoVSG")
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
        try:
            self._pluto_playback_mode = PlutoPlaybackMode(
                str(
                    preferences.value(
                        "pluto_tx/playback_mode", PlutoPlaybackMode.FINITE.value
                    )
                )
            )
        except ValueError:
            self._pluto_playback_mode = PlutoPlaybackMode.FINITE
        self._update_pluto_window_title()
        self.resize(1500, 900)
        self._build_actions()
        self._build_menus()
        self._build_workspace()
        self._configure_plot_interaction()
        self._refresh_project_view()
        self.generate_waveform()

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
        self.new_le1m_action = QtGui.QAction("New Bluetooth LE 1M Packet", self)
        self.new_le1m_action.triggered.connect(
            lambda: self._new_bluetooth_le_project(BluetoothLEPhy.LE_1M)
        )
        self.new_le2m_action = QtGui.QAction("New Bluetooth LE 2M Packet", self)
        self.new_le2m_action.triggered.connect(
            lambda: self._new_bluetooth_le_project(BluetoothLEPhy.LE_2M)
        )
        self.open_action = QtGui.QAction("Open...", self)
        self.open_action.triggered.connect(self._open_project)
        self.save_action = QtGui.QAction("Save", self)
        self.save_action.triggered.connect(self._save_project)
        self.save_as_action = QtGui.QAction("Save As...", self)
        self.save_as_action.triggered.connect(self._save_project_as)
        self.settings_action = QtGui.QAction("Bluetooth BR / EDR Settings...", self)
        self.settings_action.triggered.connect(self._edit_project_settings)
        self.rf_test_preset_action = QtGui.QAction(
            "Apply Default Bluetooth RF Test Packet Preset", self
        )
        self.rf_test_preset_action.triggered.connect(self._apply_rf_test_preset)
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
        self.pluto_transmit_action.setShortcut(QtGui.QKeySequence("Ctrl+T"))
        self.pluto_transmit_action.triggered.connect(self._start_pluto_transmission)
        self.pluto_stop_action = QtGui.QAction("Stop Pluto Transmission", self)
        self.pluto_stop_action.setEnabled(False)
        self.pluto_stop_action.triggered.connect(self._stop_pluto_transmission)
        self.validate_action = QtGui.QAction("Validate Project", self)
        self.validate_action.triggered.connect(self._show_validation)
        self.exit_action = QtGui.QAction("Exit", self)
        self.exit_action.triggered.connect(self.close)
        self.field_display_group = QtGui.QActionGroup(self)
        self.field_display_group.setExclusive(True)
        self.field_display_actions: dict[str, QtGui.QAction] = {}
        for mode, label in (
            ("all", "Major + Minor Fields"),
            ("major", "Major Fields Only"),
            ("off", "Hide Field Boundaries"),
        ):
            action = QtGui.QAction(label, self, checkable=True)
            action.setData(mode)
            action.setChecked(mode == self._field_display_mode)
            action.triggered.connect(self._field_display_changed)
            self.field_display_group.addAction(action)
            self.field_display_actions[mode] = action

    def _build_menus(self) -> None:
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")
        new_menu = file_menu.addMenu("New")
        new_menu.addActions(
            [self.new_action, self.new_le1m_action, self.new_le2m_action]
        )
        file_menu.addActions([self.open_action, self.save_action, self.save_as_action])
        file_menu.addSeparator()
        file_menu.addActions(
            [self.export_npz_action, self.export_iqtar_action, self.export_wv_action]
        )
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)
        edit_menu = menu_bar.addMenu("Edit")
        edit_menu.addActions([self.undo_action, self.redo_action])
        waveform_menu = menu_bar.addMenu("Waveform")
        waveform_menu.addAction(self.settings_action)
        waveform_menu.addAction(self.rf_test_preset_action)
        waveform_menu.addSeparator()
        for label in (
            "Packet Composer",
            "Data Sources and Lists",
            "Modulation Profiles",
            "Filters",
            "Power Envelope and Control Tracks",
            "Impairments / Dirty Transmitter",
            "Recording Layout / Sequence",
        ):
            waveform_menu.addAction(label).setEnabled(False)
        waveform_menu.addSeparator()
        waveform_menu.addAction(self.generate_action)
        graphics_menu = menu_bar.addMenu("Graphics")
        graphics_menu.addAction("Save Layout").setEnabled(False)
        graphics_menu.addAction("Restore Layout").setEnabled(False)
        graphics_menu.addSeparator()
        field_menu = graphics_menu.addMenu("Field Boundaries")
        field_menu.addActions(self.field_display_group.actions())
        output_menu = menu_bar.addMenu("Output")
        output_menu.addAction(self.pluto_settings_action)
        output_menu.addSeparator()
        output_menu.addAction(self.pluto_prepare_action)
        output_menu.addAction(self.pluto_transmit_action)
        output_menu.addAction(self.pluto_stop_action)
        output_toolbar = self.addToolBar("Output")
        output_toolbar.setObjectName("PlutoVSGOutputToolbar")
        output_toolbar.addAction(self.pluto_settings_action)
        output_toolbar.addSeparator()
        output_toolbar.addAction(self.pluto_prepare_action)
        output_toolbar.addAction(self.pluto_transmit_action)
        output_toolbar.addAction(self.pluto_stop_action)
        tools_menu = menu_bar.addMenu("Tools")
        tools_menu.addAction(self.validate_action)
        tools_menu.addAction("Device Capabilities").setEnabled(False)
        menu_bar.addMenu("Help")

    def _build_workspace(self) -> None:
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        upper = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.block_library = QtWidgets.QListWidget()
        self.block_library.addItems(
            ["Fixed Data", "Pattern", "PRBS-9", "Computed Field", "Guard / Idle", "Power Ramp"]
        )
        self.block_library.setEnabled(False)
        self.field_table = QtWidgets.QTreeWidget()
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
        composer_tabs.addTab(self.composer_view, "Visual Composer")
        composer_tabs.addTab(self.field_table, "Field Tree")
        inspector_widget = QtWidgets.QWidget()
        inspector_layout = QtWidgets.QVBoxLayout(inspector_widget)
        self.inspector = QtWidgets.QTableWidget(0, 2)
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
        generate_button = QtWidgets.QPushButton("Generate Waveform (F5)")
        generate_button.clicked.connect(self.generate_waveform)
        inspector_layout.addWidget(self.inspector)
        inspector_layout.addWidget(self.edit_settings_button)
        inspector_layout.addWidget(generate_button)
        upper.addWidget(_Panel("Block Library", self.block_library))
        upper.addWidget(_Panel("Packet Composer", composer_tabs))
        upper.addWidget(_Panel("Inspector", inspector_widget))
        upper.setStretchFactor(0, 1)
        upper.setStretchFactor(1, 3)
        upper.setStretchFactor(2, 2)

        previews = QtWidgets.QTabWidget()
        self.iq_waveform_plot = self._make_plot("Normalized Amplitude", "Time (us)")
        self.iq_waveform_legend = self.iq_waveform_plot.addLegend()
        self.power_plot = self._make_plot("IQ Power (dBFS)", "Time (us)")
        self.frequency_plot = self._make_plot("Frequency (kHz)", "Time (us)")
        self.spectrum_plot = self._make_plot(
            "Magnitude (dBFS)", "Frequency Offset (MHz)"
        )
        self.constellation_plot = self._make_plot("Q", "I")
        self.constellation_plot.setAspectLocked(True)
        for widget, title in (
            (self.iq_waveform_plot, "IQ Waveform"),
            (self.power_plot, "IQ Power"),
            (self.frequency_plot, "Instantaneous Frequency"),
            (self.spectrum_plot, "Spectrum"),
            (self.constellation_plot, "Constellation"),
        ):
            previews.addTab(widget, title)
        splitter.addWidget(upper)
        splitter.addWidget(_Panel("Generated IQ Preview", previews))
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([450, 450])
        self.setCentralWidget(splitter)

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
        self.project_path = None
        self.undo_stack.clear()
        self._refresh_project_view()
        self.generate_waveform()
        self._configuration_maybe_changed(previous_signature)

    def _new_bluetooth_le_project(self, phy: BluetoothLEPhy) -> None:
        previous_signature = self._pluto_configuration_signature()
        self.project = bluetooth_le_project(phy)
        self.project_path = None
        self.undo_stack.clear()
        self._refresh_project_view()
        self.generate_waveform()
        self._configuration_maybe_changed(previous_signature)

    def _apply_rf_test_preset(self) -> None:
        if self.project.standard == StandardProfile.BLUETOOTH_LE:
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
        if self.project.standard == StandardProfile.BLUETOOTH_LE:
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

    def _field_display_changed(self, checked: bool) -> None:
        if not checked:
            return
        action = self.sender()
        if isinstance(action, QtGui.QAction):
            self._field_display_mode = str(action.data())
        if self.result is not None:
            self._update_previews(self.result)

    def _refresh_project_view(self) -> None:
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
        parameters = [
            ("Project", self.project.name),
            ("Standard", self.project.standard.value),
            ("Center", f"{self.project.center_frequency_hz / 1e6:.6f} MHz"),
            ("Sample Rate", f"{self.project.sample_rate_hz / 1e6:.3f} MS/s"),
            ("Samples / Symbol", str(self.project.samples_per_symbol)),
            ("Repeat Count", str(self.project.repeat_count)),
            ("Period", f"{effective_period_symbols(self.project):.3f} symbols"),
            (
                "Post Idle",
                f"{effective_post_idle_symbols(self.project):.3f} symbols",
            ),
        ]
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
        is_le = self.project.standard == StandardProfile.BLUETOOTH_LE
        settings_label = (
            "Bluetooth LE Packet Settings..."
            if is_le
            else "Bluetooth BR / EDR Settings..."
        )
        self.settings_action.setText(settings_label)
        self.edit_settings_button.setText(f"Edit {settings_label}")
        self._project_inspector_parameters = parameters
        self._populate_inspector(parameters)
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
        try:
            engine = (
                BluetoothLEWaveformEngine()
                if self.project.standard == StandardProfile.BLUETOOTH_LE
                else BluetoothBRWaveformEngine()
            )
            self.result = engine.generate(self.project)
        except ValueError as error:
            self.result = None
            QtWidgets.QMessageBox.warning(self, "Waveform Generation", str(error))
            return
        self._update_previews(self.result)
        duration_ms = 1e3 * self.result.iq.size / self.result.sample_rate_hz
        self.statusBar().showMessage(
            f"Generated {self.result.iq.size:,} samples | {duration_ms:.3f} ms | "
            f"{self.result.sample_rate_hz / 1e6:.3f} MS/s"
        )

    def _update_previews(self, result: GenerationResult) -> None:
        complete_iq = np.asarray(result.iq)
        repeat_count = max(1, int(self.project.repeat_count))
        if complete_iq.size % repeat_count == 0:
            preview_sample_count = complete_iq.size // repeat_count
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
        if self._field_display_mode != "off":
            for plot in (
                self.iq_waveform_plot,
                self.power_plot,
                self.frequency_plot,
            ):
                self._add_field_guides(
                    plot,
                    result,
                    include_minor=self._field_display_mode == "all",
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

        edr_start = result.metadata.get("edr_start_sample")
        edr_indices = np.asarray(
            result.metadata.get("edr_phase_indices", ()), dtype=np.int16
        )
        if edr_start is not None and edr_indices.size:
            sample_positions = (
                int(edr_start)
                + self.project.samples_per_symbol // 2
                + np.arange(edr_indices.size + 1) * self.project.samples_per_symbol
            )
            sample_positions = sample_positions[sample_positions < iq.size]
            symbol_samples = iq[sample_positions]
            typical_amplitude = float(np.median(np.abs(symbol_samples)))
            if typical_amplitude > 0.0:
                symbol_samples = symbol_samples / typical_amplitude
        else:
            symbol_samples = iq[:: self.project.samples_per_symbol]
        self.constellation_plot.clear()
        self.constellation_plot.plot(
            symbol_samples.real,
            symbol_samples.imag,
            pen=None,
            symbol="o",
            symbolSize=7,
            symbolBrush=TRACE_COLOR,
        )
        self.constellation_plot.setRange(
            xRange=[-1.25, 1.25], yRange=[-1.25, 1.25], padding=0.0
        )
        self._remember_plot_scales()

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
        self.project_path = Path(path)
        self.undo_stack.clear()
        self._refresh_project_view()
        self.generate_waveform()
        self._configuration_maybe_changed(previous_signature)

    def _save_project(self) -> None:
        if self.project_path is None:
            self._save_project_as()
            return
        save_project(self.project_path, self.project)
        self.statusBar().showMessage(f"Saved {self.project_path.name}")

    def _save_project_as(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Pluto VSG Project",
            "waveform.pvsg.json",
            "Pluto VSG Project (*.pvsg.json)",
        )
        if not path:
            return
        self.project_path = Path(path)
        self._save_project()

    def _export_npz(self) -> None:
        if self.result is None:
            self.generate_waveform()
        if self.result is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export IQ", "waveform.npz", "NumPy IQ (*.npz)"
        )
        if path:
            save_npz(path, self.result, self.project)
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
            save_iq_tar(path, self.result, self.project)
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
            save_wv(path, self.result, self.project)
        except (OSError, ValueError) as error:
            QtWidgets.QMessageBox.critical(self, "Export R&S WV", str(error))
            return
        self.statusBar().showMessage(f"Exported {Path(path).name}")

    def _current_pluto_settings(self) -> PlutoTransmitSettings:
        bandwidth_hz = min(
            56_000_000.0,
            max(200_000.0, self._pluto_bandwidth_hz),
        )
        hardware_gain_db = pluto_hardware_gain_for_output_power_dbm(
            self._pluto_output_power_dbm,
            self._pluto_digital_backoff_db,
            self.project.center_frequency_hz,
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
        was_prepared = self._pluto_prepared_signature is not None
        self._pluto_prepared_signature = None
        self.statusBar().showMessage(
            "ADALM-Pluto configuration changed; preparation required"
        )
        # Once this session has prepared a device, subsequent RF/baseband
        # edits automatically run the explicit calibration step.
        if was_prepared and self._prepare_thread is None and self._tx_thread is None:
            QtCore.QTimer.singleShot(0, self._start_pluto_preparation)

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
        self._pluto_output_power_dbm = float(settings.output_power_dbm)
        self._pluto_digital_backoff_db = settings.digital_backoff_db
        self._pluto_bandwidth_hz = settings.rf_bandwidth_hz
        self._pluto_lead_in_guard_s = settings.lead_in_guard_s
        self._pluto_dma_preroll_s = settings.dma_preroll_s
        self._pluto_stop_guard_s = settings.stop_guard_s
        self._pluto_playback_mode = PlutoPlaybackMode(settings.playback_mode)
        preferences = QtCore.QSettings("PlutoSpectrumApp", "PlutoVSG")
        preferences.setValue("pluto_tx/uri", self._pluto_uri)
        preferences.setValue(
            "pluto_tx/output_power_dbm", self._pluto_output_power_dbm
        )
        # Preserve the derived legacy value for older application versions.
        preferences.setValue(
            "pluto_tx/hardware_gain_db", settings.resolved_hardware_gain_db
        )
        preferences.setValue(
            "pluto_tx/digital_backoff_db", self._pluto_digital_backoff_db
        )
        preferences.setValue("pluto_tx/rf_bandwidth_hz", self._pluto_bandwidth_hz)
        preferences.setValue(
            "pluto_tx/lead_in_guard_s", self._pluto_lead_in_guard_s
        )
        preferences.setValue(
            "pluto_tx/dma_preroll_s", self._pluto_dma_preroll_s
        )
        preferences.setValue("pluto_tx/stop_guard_s", self._pluto_stop_guard_s)
        preferences.setValue(
            "pluto_tx/playback_mode", self._pluto_playback_mode.value
        )
        configuration_changed = (
            previous_signature != self._pluto_configuration_signature()
        )
        if configuration_changed:
            self._pluto_prepared_signature = None
        if self._pluto_prepared_signature != self._pluto_configuration_signature():
            self.statusBar().showMessage(
                "ADALM-Pluto output settings saved; preparing configuration..."
            )
            # Accepting RF/baseband device settings is the explicit
            # configuration action. Calibration may radiate an internal tone,
            # so it is never deferred to the later Transmit command.
            self._start_pluto_preparation()
        else:
            self.statusBar().showMessage(
                "ADALM-Pluto output settings saved; configuration remains READY"
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
        self._set_pluto_busy(preparing=True, transmitting=False)
        self.statusBar().showMessage(
            "Preparing ADALM-Pluto: muted configuration and explicit TX calibration..."
        )
        thread.start()

    @QtCore.Slot(bool, str)
    def _pluto_preparation_finished(self, success: bool, message: str) -> None:
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
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._pluto_thread_finished)
        self._tx_worker = worker
        self._tx_thread = thread
        self._set_pluto_busy(preparing=False, transmitting=True)
        if self._pluto_playback_mode is PlutoPlaybackMode.CONTINUOUS:
            period_samples = self.result.iq.size // self.project.repeat_count
            period_ms = 1e3 * period_samples / self.result.sample_rate_hz
            self.statusBar().showMessage(
                "Starting continuous Pluto TX: "
                f"{period_ms:.3f} ms period, "
                f"{self.project.center_frequency_hz / 1e6:.6f} MHz; use Stop to end"
            )
        else:
            duration_ms = 1e3 * self.result.iq.size / self.result.sample_rate_hz
            self.statusBar().showMessage(
                f"Starting finite Pluto TX: {self.project.repeat_count} packet(s), "
                f"{duration_ms:.3f} ms, "
                f"{self.project.center_frequency_hz / 1e6:.6f} MHz"
            )
        thread.start()

    def _stop_pluto_transmission(self) -> None:
        if self._tx_worker is None:
            return
        self._tx_worker.cancel()
        self.pluto_stop_action.setEnabled(False)
        self.statusBar().showMessage("Stopping Pluto transmission...")

    @QtCore.Slot(bool, str)
    def _pluto_transmission_finished(self, success: bool, message: str) -> None:
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
        active = preparing or transmitting
        for action in (
            self.new_action,
            self.open_action,
            self.settings_action,
            self.generate_action,
            self.pluto_settings_action,
            self.pluto_prepare_action,
            self.pluto_transmit_action,
        ):
            action.setEnabled(not active)
        self.pluto_stop_action.setEnabled(transmitting)

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
        super().closeEvent(event)
