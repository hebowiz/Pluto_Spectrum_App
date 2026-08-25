"""Visual Composer shell and first Bluetooth BR vertical slice."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from pluto_sa.vsa.profiles.bluetooth_br import header_error_check
from pluto_sa.vsa.ui.measurement_chrome import (
    install_measurement_plot_menu,
    make_measurement_plot,
)
from pluto_vsg.backends import PlutoOutputBackend, PlutoTransmitSettings
from pluto_vsg.engine import BluetoothBRWaveformEngine, GenerationResult
from pluto_vsg.export import save_iq_tar, save_npz, save_wv
from pluto_vsg.model import (
    BluetoothPacketKind,
    PayloadSourceKind,
    WaveformProject,
    validate_project,
)
from pluto_vsg.persistence import load_project, save_project
from pluto_vsg.profiles import bluetooth_br_edr_project, bluetooth_br_fields
from pluto_vsg.ui.style import (
    ACCENT_COLOR,
    FIELD_BOUNDARY_COLOR,
    FIELD_MINOR_BOUNDARY_COLOR,
    TRACE_COLOR,
    panel_title_font,
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
            0, 27, settings.payload_length_bytes
        )
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
        self.edr_rolloff_spin = self._double_spin(0.01, 1.0, settings.edr_rolloff, 3)
        self.edr_power_spin = self._double_spin(
            -60.0, 20.0, settings.edr_relative_power_db, 3
        )
        self.pre_idle_spin = self._integer_spin(
            0, 1_000_000, settings.pre_idle_symbols
        )
        self.post_idle_spin = self._integer_spin(
            0, 1_000_000, settings.post_idle_symbols
        )
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
            ("Payload Length [byte]", self.payload_length_spin),
            ("Payload Source", self.payload_source_combo),
            ("Source Behavior", self.payload_source_help),
            ("Payload Data [bin]", self.pattern_edit),
            ("Whitening", self.whitening_check),
            ("FSK Deviation [kHz]", self.deviation_spin),
            ("Carrier Offset [kHz]", self.cfo_spin),
            ("Gaussian B*T", self.bt_spin),
            ("EDR Guard [symbols]", self.edr_guard_spin),
            ("EDR SRRC Roll-off", self.edr_rolloff_spin),
            ("EDR Power rel. GFSK [dB]", self.edr_power_spin),
            ("Pre Idle [symbols]", self.pre_idle_spin),
            ("Post Idle [symbols]", self.post_idle_spin),
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
        self._payload_source_changed()
        self._packet_type_changed()
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
        packet_type = 0x8 if packet_kind == BluetoothPacketKind.DH1_3 else 0x4
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
            edr_rolloff=self.edr_rolloff_spin.value(),
            edr_relative_power_db=self.edr_power_spin.value(),
            pre_idle_symbols=self.pre_idle_spin.value(),
            post_idle_symbols=self.post_idle_spin.value(),
        )
        project = replace(
            self._project,
            name=self.name_edit.text(),
            center_frequency_hz=self.center_spin.value() * 1e6,
            sample_rate_hz=self.sps_spin.value() * 1e6,
            samples_per_symbol=self.sps_spin.value(),
            repeat_count=self.repeat_spin.value(),
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

    def _packet_type_changed(self) -> None:
        kind = BluetoothPacketKind(self.packet_type_combo.currentData())
        payload_max = {
            BluetoothPacketKind.DH1: 27,
            BluetoothPacketKind.DH1_2: 54,
            BluetoothPacketKind.DH1_3: 83,
        }[kind]
        self.payload_length_spin.setMaximum(payload_max)
        is_edr = kind != BluetoothPacketKind.DH1
        for control in (
            self.edr_guard_spin,
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

    @property
    def project(self) -> WaveformProject:
        return self._project


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
        form = QtWidgets.QFormLayout()
        self.uri_combo = QtWidgets.QComboBox()
        self.uri_combo.setEditable(True)
        self.uri_combo.addItem("Auto (USB preferred)", "")
        devices = PlutoOutputBackend.discover_devices()
        for device in devices:
            self.uri_combo.addItem(device.label, device.selector)
            self.uri_combo.setItemData(
                self.uri_combo.count() - 1,
                device.description,
                QtCore.Qt.ItemDataRole.ToolTipRole,
            )
        configured_uri = settings.connection_uri or ""
        index = self.uri_combo.findData(configured_uri)
        if index < 0:
            matching_device = next(
                (device for device in devices if device.uri == configured_uri), None
            )
            if matching_device is not None:
                index = self.uri_combo.findData(matching_device.selector)
        if index < 0 and configured_uri:
            self.uri_combo.addItem(configured_uri, configured_uri)
            index = self.uri_combo.count() - 1
        self.uri_combo.setCurrentIndex(max(0, index))
        self.bandwidth_spin = QtWidgets.QDoubleSpinBox()
        self.bandwidth_spin.setRange(0.2, 56.0)
        self.bandwidth_spin.setDecimals(3)
        self.bandwidth_spin.setSuffix(" MHz")
        self.bandwidth_spin.setValue(settings.rf_bandwidth_hz / 1e6)
        self.bandwidth_spin.setKeyboardTracking(False)
        self.gain_spin = QtWidgets.QDoubleSpinBox()
        self.gain_spin.setRange(-89.75, 0.0)
        self.gain_spin.setDecimals(2)
        self.gain_spin.setSingleStep(0.25)
        self.gain_spin.setSuffix(" dB")
        self.gain_spin.setValue(settings.hardware_gain_db)
        self.gain_spin.setKeyboardTracking(False)
        self.lead_in_guard_spin = QtWidgets.QDoubleSpinBox()
        self.lead_in_guard_spin.setRange(0.0, 1000.0)
        self.lead_in_guard_spin.setDecimals(3)
        self.lead_in_guard_spin.setSuffix(" ms")
        self.lead_in_guard_spin.setValue(settings.lead_in_guard_s * 1e3)
        self.lead_in_guard_spin.setKeyboardTracking(False)
        self.stop_guard_spin = QtWidgets.QDoubleSpinBox()
        self.stop_guard_spin.setRange(10.0, 5000.0)
        self.stop_guard_spin.setDecimals(3)
        self.stop_guard_spin.setSuffix(" ms")
        self.stop_guard_spin.setValue(settings.stop_guard_s * 1e3)
        self.stop_guard_spin.setKeyboardTracking(False)
        form.addRow("Connection URI", self.uri_combo)
        form.addRow(
            "Center Frequency",
            QtWidgets.QLabel(f"{settings.center_frequency_hz / 1e6:.6f} MHz (Project)"),
        )
        form.addRow(
            "Sample Rate",
            QtWidgets.QLabel(f"{settings.sample_rate_hz / 1e6:.3f} MS/s (Project)"),
        )
        form.addRow("TX RF Bandwidth", self.bandwidth_spin)
        form.addRow("TX Hardware Gain", self.gain_spin)
        form.addRow("Lead-in Zero Guard", self.lead_in_guard_spin)
        form.addRow("Stop Zero Guard", self.stop_guard_spin)
        form.addRow("Packets per transmission", QtWidgets.QLabel(str(packet_count)))
        warning = QtWidgets.QLabel(
            "TX Hardware Gain is a relative Pluto setting, not calibrated dBm. "
            "Start with sufficient external attenuation and verify the output level. "
            "Zero-IQ guards suppress modulation but do not remove LO leakage."
        )
        warning.setWordWrap(True)
        warning.setStyleSheet("color: #e0b050;")
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept_settings)
        buttons.rejected.connect(self.reject)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(warning)
        layout.addWidget(buttons)
        self.resize(560, 330)

    def _accept_settings(self) -> None:
        uri = self.uri_combo.currentData()
        if uri is None:
            uri = self.uri_combo.currentText().strip()
            if uri == "Auto (USB preferred)":
                uri = ""
        candidate = replace(
            self._settings,
            connection_uri=str(uri).strip() or None,
            rf_bandwidth_hz=self.bandwidth_spin.value() * 1e6,
            hardware_gain_db=self.gain_spin.value(),
            lead_in_guard_s=self.lead_in_guard_spin.value() * 1e-3,
            stop_guard_s=self.stop_guard_spin.value() * 1e-3,
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
        try:
            self.backend.transfer(self.result)
            self.backend.start()
        except Exception as error:
            self.finished.emit(False, str(error))
            return
        self.finished.emit(
            True,
            "Pluto transmission stopped"
            if self._cancel_requested
            else "Pluto transmission complete",
        )

    def cancel(self) -> None:
        self._cancel_requested = True
        self.backend.stop()


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
        self._engine = BluetoothBRWaveformEngine()
        self._tx_thread: QtCore.QThread | None = None
        self._tx_worker: _PlutoTransmitWorker | None = None
        self._close_after_tx = False
        preferences = QtCore.QSettings("PlutoSpectrumApp", "PlutoVSG")
        self._pluto_uri = str(preferences.value("pluto_tx/uri", "") or "")
        self._pluto_gain_db = float(
            preferences.value("pluto_tx/hardware_gain_db", -30.0)
        )
        self._pluto_bandwidth_hz = float(
            preferences.value("pluto_tx/rf_bandwidth_hz", 8_000_000.0)
        )
        self._pluto_lead_in_guard_s = float(
            preferences.value("pluto_tx/lead_in_guard_s", 0.010)
        )
        self._pluto_stop_guard_s = float(
            preferences.value("pluto_tx/stop_guard_s", 0.100)
        )
        self.setWindowTitle("Pluto VSG - IQ Waveform Generator")
        self.resize(1500, 900)
        self._build_actions()
        self._build_menus()
        self._build_workspace()
        self._configure_plot_interaction()
        self._refresh_project_view()
        self.generate_waveform()

    def _build_actions(self) -> None:
        self.new_action = QtGui.QAction("New Project", self)
        self.new_action.triggered.connect(self._new_bluetooth_project)
        self.open_action = QtGui.QAction("Open...", self)
        self.open_action.triggered.connect(self._open_project)
        self.save_action = QtGui.QAction("Save", self)
        self.save_action.triggered.connect(self._save_project)
        self.save_as_action = QtGui.QAction("Save As...", self)
        self.save_as_action.triggered.connect(self._save_project_as)
        self.settings_action = QtGui.QAction("Bluetooth BR / EDR Settings...", self)
        self.settings_action.triggered.connect(self._edit_bluetooth_settings)
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
        file_menu.addActions(
            [self.new_action, self.open_action, self.save_action, self.save_as_action]
        )
        file_menu.addSeparator()
        file_menu.addActions(
            [self.export_npz_action, self.export_iqtar_action, self.export_wv_action]
        )
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)
        edit_menu = menu_bar.addMenu("Edit")
        edit_menu.addActions([QtGui.QAction("Undo", self), QtGui.QAction("Redo", self)])
        waveform_menu = menu_bar.addMenu("Waveform")
        waveform_menu.addAction(self.settings_action)
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
        output_menu.addAction(self.pluto_transmit_action)
        output_menu.addAction(self.pluto_stop_action)
        output_toolbar = self.addToolBar("Output")
        output_toolbar.setObjectName("PlutoVSGOutputToolbar")
        output_toolbar.addAction(self.pluto_settings_action)
        output_toolbar.addSeparator()
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
        self.field_table.setColumnCount(5)
        self.field_table.setHeaderLabels(
            ["Field", "Logical Bits", "Tx Symbols", "Data Source", "Modulation"]
        )
        self.field_table.header().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        self.field_table.header().setStretchLastSection(True)
        self.field_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
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
        edit_button = QtWidgets.QPushButton("Edit Bluetooth BR / EDR Settings...")
        edit_button.clicked.connect(self._edit_bluetooth_settings)
        generate_button = QtWidgets.QPushButton("Generate Waveform (F5)")
        generate_button.clicked.connect(self.generate_waveform)
        inspector_layout.addWidget(self.inspector)
        inspector_layout.addWidget(edit_button)
        inspector_layout.addWidget(generate_button)
        upper.addWidget(_Panel("Block Library", self.block_library))
        upper.addWidget(_Panel("Packet Composer", self.field_table))
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
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
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
        self.project = bluetooth_br_edr_project()
        self.project_path = None
        self._refresh_project_view()
        self.generate_waveform()

    def _edit_bluetooth_settings(self) -> None:
        dialog = _BluetoothSettingsDialog(self.project, self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        self.project = dialog.project
        self._refresh_project_view()
        self.generate_waveform()

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

        def add_field(packet_field, parent=None) -> None:
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
            ]
            item = QtWidgets.QTreeWidgetItem(values)
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
            for child in packet_field.children:
                add_field(child, item)

        for packet_field in self.project.fields:
            add_field(packet_field)
        self.field_table.expandAll()
        settings = self.project.bluetooth_br
        parameters = [
            ("Project", self.project.name),
            ("Standard", self.project.standard.value),
            ("Center", f"{self.project.center_frequency_hz / 1e6:.6f} MHz"),
            ("Sample Rate", f"{self.project.sample_rate_hz / 1e6:.3f} MS/s"),
            ("Samples / Symbol", str(self.project.samples_per_symbol)),
            ("Repeat Count", str(self.project.repeat_count)),
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
        self.inspector.setRowCount(len(parameters))
        for row, values in enumerate(parameters):
            for column, value in enumerate(values):
                self.inspector.setItem(
                    row, column, QtWidgets.QTableWidgetItem(value)
                )
        status = "Ready" if not validate_project(self.project) else "Project has validation errors"
        self.statusBar().showMessage(status)

    def generate_waveform(self) -> None:
        try:
            self.result = self._engine.generate(self.project)
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
        iq = np.asarray(result.iq)
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
    ) -> None:
        for boundary in result.field_boundaries:
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

    def _open_project(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Pluto VSG Project",
            "",
            "Pluto VSG Project (*.pvsg.json);;JSON (*.json)",
        )
        if not path:
            return
        try:
            self.project = load_project(path)
        except ValueError as error:
            QtWidgets.QMessageBox.critical(self, "Open Project", str(error))
            return
        self.project_path = Path(path)
        self._refresh_project_view()
        self.generate_waveform()

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
        return PlutoTransmitSettings(
            center_frequency_hz=self.project.center_frequency_hz,
            sample_rate_hz=self.project.sample_rate_hz,
            rf_bandwidth_hz=bandwidth_hz,
            hardware_gain_db=self._pluto_gain_db,
            connection_uri=self._pluto_uri or None,
            lead_in_guard_s=self._pluto_lead_in_guard_s,
            stop_guard_s=self._pluto_stop_guard_s,
        )

    def _edit_pluto_settings(self) -> None:
        dialog = _PlutoOutputDialog(
            self._current_pluto_settings(), self.project.repeat_count, self
        )
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        settings = dialog.settings
        self._pluto_uri = settings.connection_uri or ""
        self._pluto_gain_db = settings.hardware_gain_db
        self._pluto_bandwidth_hz = settings.rf_bandwidth_hz
        self._pluto_lead_in_guard_s = settings.lead_in_guard_s
        self._pluto_stop_guard_s = settings.stop_guard_s
        preferences = QtCore.QSettings("PlutoSpectrumApp", "PlutoVSG")
        preferences.setValue("pluto_tx/uri", self._pluto_uri)
        preferences.setValue("pluto_tx/hardware_gain_db", self._pluto_gain_db)
        preferences.setValue("pluto_tx/rf_bandwidth_hz", self._pluto_bandwidth_hz)
        preferences.setValue(
            "pluto_tx/lead_in_guard_s", self._pluto_lead_in_guard_s
        )
        preferences.setValue("pluto_tx/stop_guard_s", self._pluto_stop_guard_s)
        self.statusBar().showMessage("ADALM-Pluto output settings saved")

    def _start_pluto_transmission(self) -> None:
        if self._tx_thread is not None:
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
        self._set_transmitting(True)
        duration_ms = 1e3 * self.result.iq.size / self.result.sample_rate_hz
        self.statusBar().showMessage(
            f"Starting Pluto TX: {self.project.repeat_count} packet(s), "
            f"{duration_ms:.3f} ms, {self.project.center_frequency_hz / 1e6:.6f} MHz"
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
            QtWidgets.QMessageBox.critical(self, "Pluto Transmission", message)
        self._set_transmitting(False)

    @QtCore.Slot()
    def _pluto_thread_finished(self) -> None:
        self._tx_worker = None
        self._tx_thread = None
        if self._close_after_tx:
            self._close_after_tx = False
            self.close()

    def _set_transmitting(self, active: bool) -> None:
        for action in (
            self.new_action,
            self.open_action,
            self.settings_action,
            self.generate_action,
            self.pluto_settings_action,
            self.pluto_transmit_action,
        ):
            action.setEnabled(not active)
        self.pluto_stop_action.setEnabled(active)

    def _show_validation(self) -> None:
        issues = validate_project(self.project)
        text = (
            "Project settings are valid."
            if not issues
            else "\n".join(f"{issue.path}: {issue.message}" for issue in issues)
        )
        QtWidgets.QMessageBox.information(self, "Project Validation", text)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self._tx_thread is not None:
            self._close_after_tx = True
            self._stop_pluto_transmission()
            event.ignore()
            return
        super().closeEvent(event)
