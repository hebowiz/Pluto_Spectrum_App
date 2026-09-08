"""Protocol-aware carrier and offset selection for the VSG control pane."""

from __future__ import annotations

from dataclasses import dataclass, replace

from pyqtgraph.Qt import QtWidgets

from pluto_common.numeric_input import (
    DeferredDoubleSpinBox,
    ensure_valid_numeric_inputs,
)
from pluto_protocol.dect.carriers import DECT_CARRIER_PLANS, carrier_by_identity
from pluto_vsg.model import StandardProfile, WaveformProject
from pluto_vsg.ui.packet_settings import (
    bluetooth_classic_carriers,
    bluetooth_le_carriers,
    carrier_selector,
    wifi_24ghz_carriers,
)


def effective_rf_frequency_hz(project: WaveformProject) -> float:
    """Return the generated carrier, including a waveform-domain RF offset."""

    offset_hz = 0.0
    if project.bluetooth_br is not None:
        offset_hz = float(project.bluetooth_br.carrier_frequency_offset_hz)
    elif project.dect is not None:
        offset_hz = float(project.dect.carrier_frequency_offset_hz)
    return float(project.center_frequency_hz) + offset_hz


@dataclass(frozen=True)
class FrequencySelection:
    """Carrier-list state kept independently from the main Frequency value."""

    plan_id: str
    carrier: str
    nominal_frequency_hz: float
    offset_hz: float = 0.0


def default_frequency_selection(project: WaveformProject) -> FrequencySelection:
    """Build the carrier-list default supplied by a newly selected template."""

    if project.standard == StandardProfile.DECT and project.dect is not None:
        carrier = carrier_by_identity(
            project.dect.carrier_plan_id, project.dect.carrier_channel
        )
        return FrequencySelection(
            project.dect.carrier_plan_id,
            str(project.dect.carrier_channel),
            carrier.center_frequency_hz,
            project.dect.carrier_frequency_offset_hz,
        )
    if project.standard == StandardProfile.WIFI and project.wifi is not None:
        nominal_hz = (2407 + 5 * int(project.wifi.channel)) * 1e6
        return FrequencySelection("wifi_24", str(project.wifi.channel), nominal_hz)
    carriers = (
        bluetooth_le_carriers()
        if project.standard == StandardProfile.BLUETOOTH_LE
        else bluetooth_classic_carriers()
    )
    nominal_hz = min(
        (frequency for _label, frequency in carriers),
        key=lambda frequency: abs(frequency - project.center_frequency_hz),
    )
    channel = next(
        str(index)
        for index, (_label, frequency) in enumerate(carriers)
        if frequency == nominal_hz
    )
    offset_hz = (
        project.bluetooth_br.carrier_frequency_offset_hz
        if project.bluetooth_br is not None
        else 0.0
    )
    return FrequencySelection(
        "bluetooth_le"
        if project.standard == StandardProfile.BLUETOOTH_LE
        else "bluetooth_classic",
        channel,
        nominal_hz,
        offset_hz,
    )


def with_manual_rf_frequency(
    project: WaveformProject, frequency_hz: float
) -> WaveformProject:
    """Apply an arbitrary RF frequency without retaining a prior list offset."""

    bluetooth_br = project.bluetooth_br
    dect = project.dect
    if bluetooth_br is not None:
        bluetooth_br = replace(bluetooth_br, carrier_frequency_offset_hz=0.0)
    if dect is not None:
        dect = replace(dect, carrier_frequency_offset_hz=0.0)
    return replace(
        project,
        center_frequency_hz=float(frequency_hz),
        bluetooth_br=bluetooth_br,
        dect=dect,
    )


class FrequencySettingsDialog(QtWidgets.QDialog):
    """Select the carrier plan appropriate for the active packet template."""

    def __init__(
        self,
        project: WaveformProject,
        parent=None,
        *,
        selection: FrequencySelection | None = None,
    ) -> None:
        super().__init__(parent)
        self._project = project
        self._initial_selection = selection or default_frequency_selection(project)
        self.setWindowTitle("VSG Frequency Settings")
        self.plan_combo = QtWidgets.QComboBox()
        self.carrier_combo = QtWidgets.QComboBox()
        self.offset_spin = DeferredDoubleSpinBox()
        self.offset_spin.setRange(-3000.0, 3000.0)
        self.offset_spin.setDecimals(3)
        self.offset_spin.setSuffix(" kHz")
        self.generated_label = QtWidgets.QLabel()

        form = QtWidgets.QFormLayout()
        form.addRow("Carrier Plan", self.plan_combo)
        form.addRow("Carrier", self.carrier_combo)
        form.addRow("Carrier Offset", self.offset_spin)
        form.addRow("Generated RF Frequency", self.generated_label)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

        self._configure_for_project()
        self.plan_combo.currentIndexChanged.connect(self._plan_changed)
        self.carrier_combo.currentIndexChanged.connect(self._update_preview)
        self.offset_spin.valueChanged.connect(self._update_preview)
        self._update_preview()
        self.resize(610, self.sizeHint().height())

    def _configure_for_project(self) -> None:
        project = self._project
        selected = self._initial_selection
        if project.standard == StandardProfile.DECT and project.dect is not None:
            self._kind = "dect"
            for plan in DECT_CARRIER_PLANS:
                self.plan_combo.addItem(plan.label, plan.plan_id)
            self.plan_combo.setCurrentIndex(
                max(0, self.plan_combo.findData(selected.plan_id))
            )
            self._populate_dect_carriers(selected.carrier)
            self.offset_spin.setValue(selected.offset_hz / 1e3)
            return

        if project.standard == StandardProfile.WIFI:
            self._kind = "wifi"
            carriers = wifi_24ghz_carriers()
            plan_label = "2.4 GHz Wi-Fi"
        elif project.standard == StandardProfile.BLUETOOTH_LE:
            self._kind = "le"
            carriers = bluetooth_le_carriers()
            plan_label = "Bluetooth LE"
        else:
            self._kind = "bluetooth"
            carriers = bluetooth_classic_carriers()
            plan_label = "Bluetooth BR / EDR / HDT"
        self.plan_combo.addItem(plan_label, self._kind)
        self.plan_combo.setEnabled(False)
        selector = carrier_selector(carriers, selected.nominal_frequency_hz)
        for index in range(selector.count()):
            self.carrier_combo.addItem(
                selector.itemText(index), selector.itemData(index)
            )
        self.carrier_combo.setCurrentIndex(selector.currentIndex())
        self.offset_spin.setValue(selected.offset_hz / 1e3)

    def _populate_dect_carriers(self, preferred=None) -> None:
        plan_id = str(self.plan_combo.currentData())
        plan = next(plan for plan in DECT_CARRIER_PLANS if plan.plan_id == plan_id)
        previous = str(
            preferred if preferred is not None else self.carrier_combo.currentData()
        )
        self.carrier_combo.clear()
        for carrier in plan.carriers:
            self.carrier_combo.addItem(carrier.label, str(carrier.channel))
        index = self.carrier_combo.findData(previous)
        self.carrier_combo.setCurrentIndex(max(0, index))

    def _plan_changed(self, _index: int) -> None:
        if self._kind == "dect":
            self._populate_dect_carriers()
        self._update_preview()

    def _nominal_frequency_hz(self) -> float:
        if self._kind == "dect":
            carrier = carrier_by_identity(
                str(self.plan_combo.currentData()),
                str(self.carrier_combo.currentData()),
            )
            return float(carrier.center_frequency_hz)
        return float(self.carrier_combo.currentData())

    def _update_preview(self, _value=None) -> None:
        if self.carrier_combo.currentData() is None:
            return
        nominal_hz = self._nominal_frequency_hz()
        actual_hz = nominal_hz + self.offset_spin.value() * 1e3
        self.generated_label.setText(
            f"{actual_hz / 1e6:.6f} MHz = {nominal_hz / 1e6:.6f} MHz "
            f"{self.offset_spin.value():+.3f} kHz"
        )

    def _accept(self) -> None:
        if ensure_valid_numeric_inputs(self, title="Invalid Frequency Setting"):
            self.accept()

    @property
    def project(self) -> WaveformProject:
        nominal_hz = self._nominal_frequency_hz()
        offset_hz = self.offset_spin.value() * 1e3
        project = self._project
        if self._kind == "dect" and project.dect is not None:
            return replace(
                project,
                center_frequency_hz=nominal_hz,
                dect=replace(
                    project.dect,
                    carrier_plan_id=str(self.plan_combo.currentData()),
                    carrier_channel=str(self.carrier_combo.currentData()),
                    carrier_frequency_offset_hz=offset_hz,
                ),
            )
        if self._kind == "bluetooth" and project.bluetooth_br is not None:
            return replace(
                project,
                center_frequency_hz=nominal_hz,
                bluetooth_br=replace(
                    project.bluetooth_br, carrier_frequency_offset_hz=offset_hz
                ),
            )
        if self._kind == "wifi" and project.wifi is not None:
            channel = int(round((nominal_hz / 1e6 - 2407.0) / 5.0))
            return replace(
                project,
                center_frequency_hz=nominal_hz + offset_hz,
                wifi=replace(project.wifi, channel=channel),
            )
        return replace(project, center_frequency_hz=nominal_hz + offset_hz)

    @property
    def selection(self) -> FrequencySelection:
        return FrequencySelection(
            str(self.plan_combo.currentData()),
            str(self.carrier_combo.currentData()),
            self._nominal_frequency_hz(),
            self.offset_spin.value() * 1e3,
        )
