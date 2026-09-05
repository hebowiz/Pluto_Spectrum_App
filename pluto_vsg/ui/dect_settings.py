"""Dedicated Classic DECT packet and RF settings dialog."""

from __future__ import annotations

from dataclasses import replace

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from pluto_protocol.dect.carriers import DECT_CARRIER_PLANS, carrier_by_identity
from pluto_protocol.dect.classic import BA_NAMES, PP_S_FIELD, RFP_S_FIELD, TA_NAMES
from pluto_vsg.model import (
    DectDirection,
    DectPacketType,
    DectSettings,
    PayloadSourceKind,
    WaveformProject,
    effective_period_symbols,
    minimum_period_symbols,
    validate_project,
)
from pluto_vsg.profiles.dect import DECT_B_FIELD_BITS, dect_fields


def _bit_text(bits) -> str:
    return "".join(str(int(bit)) for bit in bits)


_A_TAIL_CHOICES = (
    ("All zeros", "0" * 40),
    ("All ones", "1" * 40),
    ("Alternating 01", "01" * 20),
    ("Alternating 10", "10" * 20),
)


class DectSettingsDialog(QtWidgets.QDialog):
    """Edit DECT fields without exposing settings from other VSG profiles."""

    def __init__(self, project: WaveformProject, parent=None) -> None:
        super().__init__(parent)
        if project.dect is None:
            raise ValueError("DECT settings are required")
        self._base_project = project
        settings = project.dect
        initial_s_field = (
            RFP_S_FIELD
            if DectDirection(settings.direction) is DectDirection.RFP
            else PP_S_FIELD
        )
        initial_s_text = _bit_text(initial_s_field)
        self.setWindowTitle("DECT Packet / Waveform Settings")
        self.resize(860, 760)

        self.plan_combo = QtWidgets.QComboBox()
        for plan in DECT_CARRIER_PLANS:
            self.plan_combo.addItem(plan.label, plan.plan_id)
        self.plan_combo.setCurrentIndex(
            max(0, self.plan_combo.findData(settings.carrier_plan_id))
        )
        self.carrier_combo = QtWidgets.QComboBox()
        self.offset_spin = QtWidgets.QDoubleSpinBox()
        self.offset_spin.setRange(-3000.0, 3000.0)
        self.offset_spin.setDecimals(3)
        self.offset_spin.setSuffix(" kHz")
        self.offset_spin.setValue(settings.carrier_frequency_offset_hz / 1e3)
        self.actual_frequency_label = QtWidgets.QLabel()

        self.direction_combo = QtWidgets.QComboBox()
        for direction in DectDirection:
            self.direction_combo.addItem(direction.value, direction)
        self.direction_combo.setCurrentIndex(
            self.direction_combo.findData(DectDirection(settings.direction))
        )
        self.packet_type_combo = QtWidgets.QComboBox()
        for packet_type in DectPacketType:
            self.packet_type_combo.addItem(packet_type.value, packet_type)
        self.packet_type_combo.setCurrentIndex(
            self.packet_type_combo.findData(DectPacketType(settings.packet_type))
        )
        self.prolonged_check = QtWidgets.QCheckBox("Add p-16...p-1 preamble")
        self.prolonged_check.setChecked(settings.prolonged_preamble)
        self.samples_per_symbol_combo = QtWidgets.QComboBox()
        for sps in (4, 8, 16, 32):
            self.samples_per_symbol_combo.addItem(f"{sps} S/sym", sps)
        self.samples_per_symbol_combo.setCurrentIndex(
            max(0, self.samples_per_symbol_combo.findData(project.samples_per_symbol))
        )
        self.repeat_spin = QtWidgets.QSpinBox()
        self.repeat_spin.setRange(1, 1000)
        self.repeat_spin.setValue(project.repeat_count)
        self.deviation_spin = QtWidgets.QDoubleSpinBox()
        self.deviation_spin.setRange(1.0, 1500.0)
        self.deviation_spin.setDecimals(3)
        self.deviation_spin.setSuffix(" kHz")
        self.deviation_spin.setValue(settings.frequency_deviation_hz / 1e3)
        self.bt_spin = QtWidgets.QDoubleSpinBox()
        self.bt_spin.setRange(0.05, 2.0)
        self.bt_spin.setDecimals(3)
        self.bt_spin.setValue(settings.gaussian_bt)
        self.pre_idle_spin = QtWidgets.QSpinBox()
        self.pre_idle_spin.setRange(0, 10000)
        self.pre_idle_spin.setValue(settings.pre_idle_symbols)
        self._minimum_period_symbols = minimum_period_symbols(project)
        self.period_spin = QtWidgets.QDoubleSpinBox()
        self.period_spin.setRange(0.0, 1_000_000.0)
        self.period_spin.setDecimals(3)
        self.period_spin.setSuffix(" symbols")
        self.period_spin.setValue(effective_period_symbols(project))
        self.rise_spin = self._double_spin(
            0.0, 1000.0, project.power_envelope.rise_symbols, " symbols"
        )
        self.rise_delay_spin = self._double_spin(
            -1000.0,
            1000.0,
            project.power_envelope.rise_delay_symbols,
            " symbols",
        )
        self.fall_spin = self._double_spin(
            0.0, 1000.0, project.power_envelope.fall_symbols, " symbols"
        )
        self.fall_delay_spin = self._double_spin(
            -1000.0,
            1000.0,
            project.power_envelope.fall_delay_symbols,
            " symbols",
        )
        self.ramp_combo = QtWidgets.QComboBox()
        self.ramp_combo.addItems(["Cosine", "Linear"])
        self.ramp_combo.setCurrentText(project.power_envelope.shape)

        self.preamble_value = QtWidgets.QLabel(initial_s_text[:16])
        self.sync_value = QtWidgets.QLabel(initial_s_text[16:])
        fixed_font = QtGui.QFontDatabase.systemFont(
            QtGui.QFontDatabase.SystemFont.FixedFont
        )
        self.preamble_value.setFont(fixed_font)
        self.sync_value.setFont(fixed_font)
        header = settings.a_header_bits.replace(" ", "").replace("_", "")
        self.ta_combo = QtWidgets.QComboBox()
        self.q1_combo = self._bit_combo(int(header[3:4] or "0"))
        self.ba_combo = QtWidgets.QComboBox()
        for value, meaning in BA_NAMES.items():
            self.ba_combo.addItem(f"{value:03b} — {meaning}", value)
        self.ba_combo.setCurrentIndex(self.ba_combo.findData(int(header[4:7] or "0", 2)))
        self.q2_combo = self._bit_combo(int(header[7:8] or "0"))
        self.a_tail_combo = QtWidgets.QComboBox()
        for label, bits in _A_TAIL_CHOICES:
            self.a_tail_combo.addItem(f"{label} ({bits})", bits)
        tail_index = self.a_tail_combo.findData(settings.a_tail_bits)
        if tail_index < 0:
            self.a_tail_combo.addItem(
                f"Loaded project value ({settings.a_tail_bits})", settings.a_tail_bits
            )
            tail_index = self.a_tail_combo.count() - 1
        self.a_tail_combo.setCurrentIndex(tail_index)
        self.r_crc_value = QtWidgets.QLabel("Automatic from A Header + Tail")
        self.b_source_combo = QtWidgets.QComboBox()
        for label, source in (
            ("Constant (first bit)", PayloadSourceKind.FIXED),
            ("Repeating bit pattern", PayloadSourceKind.PATTERN),
            ("PRBS-9", PayloadSourceKind.PRBS9),
        ):
            self.b_source_combo.addItem(label, source)
        self.b_source_combo.setCurrentIndex(
            self.b_source_combo.findData(PayloadSourceKind(settings.b_field_source))
        )
        self.b_pattern_edit = self._bits_edit(settings.b_field_pattern, 4096)
        self.x_crc_auto = QtWidgets.QCheckBox("Calculate format-specific X-CRC")
        self.x_crc_auto.setChecked(settings.x_crc_auto)
        self.x_field_edit = self._bits_edit(settings.x_field_bits, 4)
        self.z_repeat_auto = QtWidgets.QCheckBox("Repeat generated X-field")
        self.z_repeat_auto.setChecked(settings.z_repeat_auto)
        self.z_field_edit = self._bits_edit(settings.z_field_bits, 4)
        self.packet_layout_label = QtWidgets.QLabel()
        self.packet_layout_label.setWordWrap(True)
        self._populate_ta(int(header[:3] or "0", 2))

        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._rf_tab(), "RF / Packet")
        tabs.addTab(self._fields_tab(), "Fields")
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(tabs)
        layout.addWidget(buttons)

        self.plan_combo.currentIndexChanged.connect(
            lambda _index: self._populate_carriers()
        )
        self.carrier_combo.currentIndexChanged.connect(self._update_derived)
        self.offset_spin.valueChanged.connect(self._update_derived)
        self.packet_type_combo.currentIndexChanged.connect(self._update_derived)
        self.direction_combo.currentIndexChanged.connect(
            lambda _index: self._direction_changed()
        )
        self.x_crc_auto.toggled.connect(
            lambda _enabled: self._update_derived()
        )
        self.z_repeat_auto.toggled.connect(
            lambda _enabled: self._update_derived()
        )
        self.b_source_combo.currentIndexChanged.connect(self._update_derived)
        for signal in (
            self.packet_type_combo.currentIndexChanged,
            self.prolonged_check.toggled,
            self.samples_per_symbol_combo.currentIndexChanged,
            self.pre_idle_spin.valueChanged,
            self.rise_spin.valueChanged,
            self.rise_delay_spin.valueChanged,
            self.fall_spin.valueChanged,
            self.fall_delay_spin.valueChanged,
        ):
            signal.connect(self._update_period_constraints)
        self._populate_carriers(settings.carrier_channel)
        self.x_field_edit.setDisabled(self.x_crc_auto.isChecked())
        self.z_field_edit.setDisabled(self.z_repeat_auto.isChecked())
        self._update_derived()
        self._update_period_constraints()

    @staticmethod
    def _bits_edit(value: str, maximum: int) -> QtWidgets.QLineEdit:
        edit = QtWidgets.QLineEdit(str(value))
        edit.setMaxLength(int(maximum))
        edit.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        edit.setValidator(QtGui.QRegularExpressionValidator(QtCore.QRegularExpression("[01 _]*")))
        return edit

    @staticmethod
    def _double_spin(
        minimum: float, maximum: float, value: float, suffix: str = ""
    ) -> QtWidgets.QDoubleSpinBox:
        control = QtWidgets.QDoubleSpinBox()
        control.setRange(minimum, maximum)
        control.setDecimals(3)
        control.setSuffix(suffix)
        control.setValue(value)
        control.setKeyboardTracking(False)
        return control

    @staticmethod
    def _bit_combo(value: int) -> QtWidgets.QComboBox:
        control = QtWidgets.QComboBox()
        control.addItem("0", 0)
        control.addItem("1", 1)
        control.setCurrentIndex(control.findData(int(value)))
        return control

    @staticmethod
    def _form(rows: tuple[tuple[str, QtWidgets.QWidget], ...]) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(widget)
        layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        for label, control in rows:
            layout.addRow(label, control)
        return widget

    def _rf_tab(self) -> QtWidgets.QWidget:
        return self._form(
            (
                ("Carrier Plan", self.plan_combo),
                ("Carrier", self.carrier_combo),
                ("Frequency Offset", self.offset_spin),
                ("Generated RF Frequency", self.actual_frequency_label),
                ("Direction", self.direction_combo),
                ("Packet Type", self.packet_type_combo),
                ("Prolonged Preamble", self.prolonged_check),
                ("Samples / Symbol", self.samples_per_symbol_combo),
                ("Repeat Count", self.repeat_spin),
                ("Peak Frequency Deviation", self.deviation_spin),
                ("Gaussian B*T", self.bt_spin),
                ("Pre Idle", self.pre_idle_spin),
                ("Packet Period", self.period_spin),
                ("Ramp Up Time", self.rise_spin),
                ("Ramp Up Start rel. Packet", self.rise_delay_spin),
                ("Ramp Down Time", self.fall_spin),
                ("Ramp Down Start rel. Packet End", self.fall_delay_spin),
                ("Ramp Shape", self.ramp_combo),
                ("Derived Layout", self.packet_layout_label),
            )
        )

    def _fields_tab(self) -> QtWidgets.QWidget:
        content = self._form(
            (
                ("Preamble (Direction-derived)", self.preamble_value),
                ("Packet Sync Word (Direction-derived)", self.sync_value),
                ("A Header / TA", self.ta_combo),
                ("A Header / Q1-BCK", self.q1_combo),
                ("A Header / BA", self.ba_combo),
                ("A Header / Q2", self.q2_combo),
                ("A Tail Pattern", self.a_tail_combo),
                ("R-CRC", self.r_crc_value),
                ("B-field Source", self.b_source_combo),
                ("B-field Data / Pattern", self.b_pattern_edit),
                ("X-field Auto", self.x_crc_auto),
                ("X-field (4 bits)", self.x_field_edit),
                ("Z-field Auto", self.z_repeat_auto),
                ("Z-field (4 bits)", self.z_field_edit),
            )
        )
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)
        return container

    def _populate_carriers(self, preferred: object = None) -> None:
        plan_id = str(self.plan_combo.currentData())
        plan = next(plan for plan in DECT_CARRIER_PLANS if plan.plan_id == plan_id)
        previous = str(preferred if preferred is not None else self.carrier_combo.currentData())
        self.carrier_combo.clear()
        for carrier in plan.carriers:
            self.carrier_combo.addItem(carrier.label, str(carrier.channel))
        index = self.carrier_combo.findData(previous)
        self.carrier_combo.setCurrentIndex(max(0, index))
        self._update_derived()

    def _direction_changed(self) -> None:
        field = (
            RFP_S_FIELD
            if DectDirection(self.direction_combo.currentData()) is DectDirection.RFP
            else PP_S_FIELD
        )
        text = _bit_text(field)
        self.preamble_value.setText(text[:16])
        self.sync_value.setText(text[16:])
        self._populate_ta(int(self.ta_combo.currentData()))

    def _populate_ta(self, selected: int) -> None:
        direction = DectDirection(self.direction_combo.currentData())
        self.ta_combo.clear()
        for value in range(8):
            if value == 0b010:
                meaning = "NT / ULE NT (context dependent)"
            elif value == 0b111:
                meaning = (
                    "PT / Paging Tail"
                    if direction is DectDirection.RFP
                    else "MT / First PP Transmission"
                )
            else:
                meaning = TA_NAMES.get(value, "Reserved")
            self.ta_combo.addItem(f"{value:03b} — {meaning}", value)
        self.ta_combo.setCurrentIndex(max(0, self.ta_combo.findData(selected)))

    def _update_derived(self) -> None:
        if self.carrier_combo.currentData() is None:
            return
        carrier = carrier_by_identity(
            str(self.plan_combo.currentData()), str(self.carrier_combo.currentData())
        )
        actual = carrier.center_frequency_hz + self.offset_spin.value() * 1e3
        self.actual_frequency_label.setText(
            f"{actual / 1e6:.6f} MHz = {carrier.center_frequency_hz / 1e6:.6f} MHz "
            f"{self.offset_spin.value():+.3f} kHz"
        )
        packet_type = DectPacketType(self.packet_type_combo.currentData())
        b_count = DECT_B_FIELD_BITS.get(packet_type, 0)
        has_b = b_count > 0
        has_z = packet_type in {DectPacketType.P32Z, DectPacketType.P80Z}
        for widget in (self.b_source_combo, self.b_pattern_edit, self.x_crc_auto):
            widget.setEnabled(has_b)
        self.x_field_edit.setEnabled(has_b and not self.x_crc_auto.isChecked())
        self.z_repeat_auto.setEnabled(has_z)
        self.z_field_edit.setEnabled(has_z and not self.z_repeat_auto.isChecked())
        source = PayloadSourceKind(self.b_source_combo.currentData())
        self.b_pattern_edit.setEnabled(has_b and source is not PayloadSourceKind.PRBS9)
        total = {DectPacketType.P00: 96, DectPacketType.P32: 420, DectPacketType.P32Z: 424, DectPacketType.P80: 900, DectPacketType.P80Z: 904}[packet_type]
        self.packet_layout_label.setText(
            f"S 32 + A 64"
            + (f" + B {b_count} + X 4" if has_b else "")
            + (" + Z 4" if has_z else "")
            + f" = {total} symbols from p0"
        )

    def _update_period_constraints(self, _value=None) -> None:
        settings = self._settings()
        sps = int(self.samples_per_symbol_combo.currentData())
        candidate = replace(
            self._base_project,
            sample_rate_hz=1_152_000.0 * sps,
            samples_per_symbol=sps,
            fields=dect_fields(settings),
            dect=settings,
            power_envelope=replace(
                self._base_project.power_envelope,
                rise_symbols=self.rise_spin.value(),
                rise_delay_symbols=self.rise_delay_spin.value(),
                fall_symbols=self.fall_spin.value(),
                fall_delay_symbols=self.fall_delay_spin.value(),
            ),
        )
        self._minimum_period_symbols = minimum_period_symbols(candidate)
        self.period_spin.setMinimum(self._minimum_period_symbols)

    def _settings(self) -> DectSettings:
        return DectSettings(
            direction=DectDirection(self.direction_combo.currentData()),
            packet_type=DectPacketType(self.packet_type_combo.currentData()),
            prolonged_preamble=self.prolonged_check.isChecked(),
            preamble_bits=self.preamble_value.text(),
            sync_word_bits=self.sync_value.text(),
            a_header_bits=(
                f"{int(self.ta_combo.currentData()):03b}"
                + str(int(self.q1_combo.currentData()))
                + f"{int(self.ba_combo.currentData()):03b}"
                + str(int(self.q2_combo.currentData()))
            ),
            a_tail_bits=str(self.a_tail_combo.currentData()),
            r_crc_auto=True,
            r_crc_bits=self._base_project.dect.r_crc_bits,
            b_field_source=PayloadSourceKind(self.b_source_combo.currentData()),
            b_field_pattern=self.b_pattern_edit.text(),
            x_crc_auto=self.x_crc_auto.isChecked(),
            x_field_bits=self.x_field_edit.text(),
            z_repeat_auto=self.z_repeat_auto.isChecked(),
            z_field_bits=self.z_field_edit.text(),
            carrier_plan_id=str(self.plan_combo.currentData()),
            carrier_channel=str(self.carrier_combo.currentData()),
            carrier_frequency_offset_hz=self.offset_spin.value() * 1e3,
            frequency_deviation_hz=self.deviation_spin.value() * 1e3,
            gaussian_bt=self.bt_spin.value(),
            pre_idle_symbols=self.pre_idle_spin.value(),
            post_idle_symbols=0,
        )

    @property
    def project(self) -> WaveformProject:
        settings = self._settings()
        carrier = carrier_by_identity(settings.carrier_plan_id, settings.carrier_channel)
        packet_type = DectPacketType(settings.packet_type)
        sps = int(self.samples_per_symbol_combo.currentData())
        project = replace(
            self._base_project,
            name=f"DECT {packet_type.value} Packet",
            sample_rate_hz=1_152_000.0 * sps,
            samples_per_symbol=sps,
            repeat_count=self.repeat_spin.value(),
            period_symbols=self.period_spin.value(),
            center_frequency_hz=carrier.center_frequency_hz,
            fields=dect_fields(settings),
            dect=settings,
            power_envelope=replace(
                self._base_project.power_envelope,
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
        return project

    def _accept(self) -> None:
        project = self.project
        issues = validate_project(project)
        if issues:
            QtWidgets.QMessageBox.warning(
                self,
                "DECT Settings",
                "\n".join(f"{issue.path}: {issue.message}" for issue in issues),
            )
            return
        self.accept()
