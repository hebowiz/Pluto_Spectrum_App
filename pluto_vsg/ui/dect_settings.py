"""Dedicated Classic DECT packet and RF settings dialog."""

from __future__ import annotations

from dataclasses import replace

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from pluto_common.numeric_input import (
    DeferredDoubleSpinBox,
    DeferredSpinBox,
    ensure_valid_numeric_inputs,
)
from pluto_protocol.dect.carriers import DECT_CARRIER_PLANS, carrier_by_identity
from pluto_protocol.dect.classic import BA_NAMES, PP_S_FIELD, RFP_S_FIELD, TA_NAMES
from pluto_protocol.dect.rf_modulation import (
    CASE_B_DEFINITIONS,
    DectScramblingMode,
    case_b_format_for_packet,
)
from pluto_vsg.model import (
    DectBFieldSource,
    DectDirection,
    DectPacketType,
    DectSettings,
    WaveformProject,
    effective_period_symbols,
    minimum_period_symbols,
    validate_project,
)
from pluto_vsg.profiles.dect import DECT_B_FIELD_BITS, dect_fields
from pluto_vsg.ui.packet_settings import SymbolTimeControl, packet_settings_tabs


def _bit_text(bits) -> str:
    return "".join(str(int(bit)) for bit in bits)


_TEST_BURST_TX_A_TAIL_HEX = "0x70736E6363"
_TEST_BURST_TX_A_TAIL_BITS = f"{int(_TEST_BURST_TX_A_TAIL_HEX, 16):040b}"

_A_TAIL_CHOICES = (
    ("All zeros", "0" * 40),
    ("All ones", "1" * 40),
    ("Alternating 01", "01" * 20),
    ("Alternating 10", "10" * 20),
    ("Test Burst Tx", _TEST_BURST_TX_A_TAIL_BITS),
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
        self.offset_spin = DeferredDoubleSpinBox()
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
        self.sample_rate_value = QtWidgets.QLabel()
        self.repeat_spin = DeferredSpinBox()
        self.repeat_spin.setRange(1, 1000)
        self.repeat_spin.setValue(project.repeat_count)
        self.deviation_spin = DeferredDoubleSpinBox()
        self.deviation_spin.setRange(1.0, 1500.0)
        self.deviation_spin.setDecimals(3)
        self.deviation_spin.setSuffix(" kHz")
        self.deviation_spin.setValue(settings.frequency_deviation_hz / 1e3)
        self.bt_spin = DeferredDoubleSpinBox()
        self.bt_spin.setRange(0.05, 2.0)
        self.bt_spin.setDecimals(3)
        self.bt_spin.setValue(settings.gaussian_bt)
        self.pre_idle_spin = DeferredSpinBox()
        self.pre_idle_spin.setRange(0, 10000)
        self.pre_idle_spin.setValue(settings.pre_idle_symbols)
        self._minimum_period_symbols = minimum_period_symbols(project)
        self.period_spin = DeferredDoubleSpinBox()
        self.period_spin.setRange(0.0, 1_000_000.0)
        self.period_spin.setDecimals(3)
        self.period_spin.setSuffix(" symbols")
        self.period_spin.setValue(effective_period_symbols(project))
        self.post_idle_value = QtWidgets.QLabel()
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
        self.a_tail_combo.addItem("Custom", None)
        for label, bits in _A_TAIL_CHOICES:
            self.a_tail_combo.addItem(
                f"{label} (0x{int(bits, 2):010X})", bits
            )
        tail_index = self.a_tail_combo.findData(settings.a_tail_bits)
        self.a_tail_combo.setCurrentIndex(max(0, tail_index))
        self.a_tail_edit = QtWidgets.QLineEdit(
            self._format_a_tail_value(settings.a_tail_bits)
        )
        self.a_tail_edit.setMaxLength(42)
        self.a_tail_edit.setFont(fixed_font)
        self.a_tail_edit.setPlaceholderText(
            "0x0000000000 or exactly 40 binary digits"
        )
        self.a_tail_edit.setValidator(
            QtGui.QRegularExpressionValidator(
                QtCore.QRegularExpression(
                    "(?:0[xX][0-9A-Fa-f]{0,10}|[01 _]{0,40})"
                )
            )
        )
        self.r_crc_value = QtWidgets.QLabel("Automatic from A Header + Tail")
        self.b_source_combo = QtWidgets.QComboBox()
        for label, source in (
            ("Constant (first bit)", DectBFieldSource.FIXED),
            ("Repeating bit pattern", DectBFieldSource.PATTERN),
            ("PRBS-9", DectBFieldSource.PRBS9),
            ("Case A — 00001111 (Air-side)", DectBFieldSource.CASE_A),
            ("Case B (ETSI, packet-derived)", DectBFieldSource.CASE_B_ETSI),
        ):
            self.b_source_combo.addItem(label, source)
        self.b_source_combo.setCurrentIndex(
            self.b_source_combo.findData(DectBFieldSource(settings.b_field_source))
        )
        self.b_pattern_edit = self._bits_edit(settings.b_field_pattern, 4096)
        self.test_pattern_value = QtWidgets.QLabel()
        self.test_pattern_value.setWordWrap(True)
        self.scrambling_combo = QtWidgets.QComboBox()
        for mode in DectScramblingMode:
            self.scrambling_combo.addItem(mode.value, mode)
        self.scrambling_combo.setCurrentIndex(
            self.scrambling_combo.findData(DectScramblingMode(settings.scrambling_mode))
        )
        self.scrambling_phase_spin = DeferredSpinBox()
        self.scrambling_phase_spin.setRange(0, 7)
        self.scrambling_phase_spin.setValue(int(settings.scrambling_phase or 0))
        self.x_crc_auto = QtWidgets.QCheckBox("Calculate format-specific X-CRC")
        self.x_crc_auto.setChecked(settings.x_crc_auto)
        self.x_field_edit = self._bits_edit(settings.x_field_bits, 4)
        self.z_repeat_auto = QtWidgets.QCheckBox("Repeat generated X-field")
        self.z_repeat_auto.setChecked(settings.z_repeat_auto)
        self.z_field_edit = self._bits_edit(settings.z_field_bits, 4)
        self.packet_layout_label = QtWidgets.QLabel()
        self.packet_layout_label.setWordWrap(True)
        self._populate_ta(int(header[:3] or "0", 2))

        self._timing_controls = tuple(
            SymbolTimeControl(control, lambda: 1_152_000.0)
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
            self._rf_rows(),
            self._field_rows(),
        )
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.tabs)
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
        self.scrambling_combo.currentIndexChanged.connect(self._update_derived)
        self.a_tail_combo.currentIndexChanged.connect(
            self._a_tail_preset_changed
        )
        self.a_tail_edit.textEdited.connect(self._a_tail_value_edited)
        self.period_spin.valueChanged.connect(self._update_period_constraints)
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
        control = DeferredDoubleSpinBox()
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

    def _rf_rows(self) -> tuple[tuple[str, QtWidgets.QWidget], ...]:
        timing = self._timing_controls
        return (
                ("Modulation", QtWidgets.QLabel("GFSK / 1.152 Msym/s")),
                ("Packet Type / Length", self.packet_type_combo),
                ("Prolonged Preamble", self.prolonged_check),
                ("Samples / Symbol", self.samples_per_symbol_combo),
                ("Sample Rate", self.sample_rate_value),
                ("Repeat Count", self.repeat_spin),
                ("Peak Frequency Deviation", self.deviation_spin),
                ("Gaussian B*T", self.bt_spin),
                ("Pre Idle", timing[0]),
                ("Packet Period", timing[1]),
                ("Derived Post Idle", self.post_idle_value),
                ("Ramp Up Time", timing[2]),
                ("Ramp Up Start rel. Packet", timing[3]),
                ("Ramp Down Time", timing[4]),
                ("Ramp Down Start rel. Packet End", timing[5]),
                ("Ramp Shape", self.ramp_combo),
                ("Derived Layout", self.packet_layout_label),
        )

    def _field_rows(self) -> tuple[tuple[str, QtWidgets.QWidget], ...]:
        return (
                ("Direction", self.direction_combo),
                ("Preamble (Direction-derived)", self.preamble_value),
                ("Packet Sync Word (Direction-derived)", self.sync_value),
                ("A Header / TA", self.ta_combo),
                ("A Header / Q1-BCK", self.q1_combo),
                ("A Header / BA", self.ba_combo),
                ("A Header / Q2", self.q2_combo),
                ("A Tail Preset", self.a_tail_combo),
                ("A Tail Value (40-bit)", self.a_tail_edit),
                ("R-CRC", self.r_crc_value),
                ("B-field Source", self.b_source_combo),
                ("B-field Data / Pattern", self.b_pattern_edit),
                ("RF Modulation Test Pattern", self.test_pattern_value),
                ("B-field Scrambling", self.scrambling_combo),
                ("Scrambling Frame Phase", self.scrambling_phase_spin),
                ("X-field Auto", self.x_crc_auto),
                ("X-field (4 bits)", self.x_field_edit),
                ("Z-field Auto", self.z_repeat_auto),
                ("Z-field (4 bits)", self.z_field_edit),
        )

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

    @staticmethod
    def _format_a_tail_value(bits: str) -> str:
        normalized = str(bits).replace(" ", "").replace("_", "")
        if len(normalized) == 40 and all(bit in "01" for bit in normalized):
            return f"0x{int(normalized, 2):010X}"
        return str(bits)

    @staticmethod
    def _a_tail_bits(value: str) -> str:
        normalized = str(value).replace(" ", "").replace("_", "")
        if normalized.lower().startswith("0x"):
            hexadecimal = normalized[2:]
            if len(hexadecimal) == 10 and all(
                character in "0123456789abcdefABCDEF"
                for character in hexadecimal
            ):
                return f"{int(hexadecimal, 16):040b}"
        return normalized

    def _a_tail_preset_changed(self, _index: int) -> None:
        bits = self.a_tail_combo.currentData()
        if bits is None:
            return
        self.a_tail_edit.setText(self._format_a_tail_value(str(bits)))

    def _a_tail_value_edited(self, _text: str) -> None:
        if self.a_tail_combo.currentData() is not None:
            blocker = QtCore.QSignalBlocker(self.a_tail_combo)
            self.a_tail_combo.setCurrentIndex(0)
            del blocker

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
        self.b_source_combo.setEnabled(True)
        for widget in (self.b_pattern_edit, self.x_crc_auto):
            widget.setEnabled(has_b)
        self.x_field_edit.setEnabled(has_b and not self.x_crc_auto.isChecked())
        self.z_repeat_auto.setEnabled(has_z)
        self.z_field_edit.setEnabled(has_z and not self.z_repeat_auto.isChecked())
        source = DectBFieldSource(self.b_source_combo.currentData())
        self.b_pattern_edit.setEnabled(
            has_b and source in {DectBFieldSource.FIXED, DectBFieldSource.PATTERN}
        )
        mode = DectScramblingMode(self.scrambling_combo.currentData())
        self.scrambling_phase_spin.setEnabled(has_b and mode is DectScramblingMode.STANDARD)
        if source is DectBFieldSource.CASE_A:
            self.test_pattern_value.setText("Case A / 00001111 repeat / Air-side")
        elif source is DectBFieldSource.CASE_B_ETSI:
            test_count = 32 if packet_type is DectPacketType.P00 else b_count
            format = case_b_format_for_packet(packet_type.value, test_count)
            self.test_pattern_value.setText(
                "Unsupported for this packet format"
                if format is None
                else f"{CASE_B_DEFINITIONS[format].label} / Air-side"
            )
        else:
            self.test_pattern_value.setText("Normal / user data")
        total = {DectPacketType.P00: 96, DectPacketType.P32: 420, DectPacketType.P32Z: 424, DectPacketType.P80: 900, DectPacketType.P80Z: 904}[packet_type]
        self.packet_layout_label.setText(
            f"S 32 + A 64"
            + (f" + B {b_count} + X 4" if has_b else "")
            + (" + Z 4" if has_z else "")
            + f" = {total} symbols / {total / 1.152:.6g} us from p0"
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
        self.sample_rate_value.setText(f"{1.152 * sps:.3f} MS/s")
        post_idle = max(0.0, self.period_spin.value() - self._minimum_period_symbols)
        self.post_idle_value.setText(
            f"{post_idle:.3f} symbols = {post_idle / 1.152:.6g} us"
        )
        for control in self._timing_controls:
            control.refresh()

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
            a_tail_bits=self._a_tail_bits(self.a_tail_edit.text()),
            r_crc_auto=True,
            r_crc_bits=self._base_project.dect.r_crc_bits,
            b_field_source=DectBFieldSource(self.b_source_combo.currentData()),
            b_field_pattern=self.b_pattern_edit.text(),
            scrambling_mode=DectScramblingMode(self.scrambling_combo.currentData()),
            scrambling_phase=self.scrambling_phase_spin.value(),
            x_crc_auto=self.x_crc_auto.isChecked(),
            x_field_bits=self.x_field_edit.text(),
            z_repeat_auto=self.z_repeat_auto.isChecked(),
            z_field_bits=self.z_field_edit.text(),
            carrier_plan_id=self._base_project.dect.carrier_plan_id,
            carrier_channel=self._base_project.dect.carrier_channel,
            carrier_frequency_offset_hz=(
                self._base_project.dect.carrier_frequency_offset_hz
            ),
            frequency_deviation_hz=self.deviation_spin.value() * 1e3,
            gaussian_bt=self.bt_spin.value(),
            pre_idle_symbols=self.pre_idle_spin.value(),
            post_idle_symbols=0,
        )

    @property
    def project(self) -> WaveformProject:
        settings = self._settings()
        packet_type = DectPacketType(settings.packet_type)
        sps = int(self.samples_per_symbol_combo.currentData())
        project = replace(
            self._base_project,
            name=f"DECT {packet_type.value} Packet",
            sample_rate_hz=1_152_000.0 * sps,
            samples_per_symbol=sps,
            repeat_count=self.repeat_spin.value(),
            period_symbols=self.period_spin.value(),
            # Carrier selection belongs to the main VSG Frequency Settings
            # dialog. Packet-field edits must preserve a manual Frequency.
            center_frequency_hz=self._base_project.center_frequency_hz,
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
        if not ensure_valid_numeric_inputs(self, title="Invalid DECT Setting"):
            return
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
