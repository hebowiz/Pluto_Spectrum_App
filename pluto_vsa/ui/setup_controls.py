"""Common receiver setup widgets; protocol DSP remains in each workspace."""

from collections.abc import Callable

from pyqtgraph.Qt import QtCore, QtWidgets

from pluto_common.numeric_input import DeferredDoubleSpinBox
from pluto_common.config.input_frontend import InputPowerCorrection
from pluto_common.sdr.trigger import TriggerKind, TriggerSlope
from pluto_vsa.pattern import IQPowerTriggerSettings


def configure_form(form: QtWidgets.QFormLayout) -> None:
    form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
    form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    form.setHorizontalSpacing(18)
    form.setVerticalSpacing(10)
    for row in range(form.rowCount()):
        item = form.itemAt(row, QtWidgets.QFormLayout.ItemRole.LabelRole)
        if item is not None and item.widget() is not None:
            item.widget().setMinimumWidth(222)


class ReceiverSetupControls:
    """Presentation and saved state shared by Generic and dedicated receivers.

    Dedicated duration models remain in milliseconds; the visible editor can
    use symbols without changing existing capture/decoder coordinate handling.
    """

    def __init__(self, owner, *, rate: Callable[[], float], bandwidth,
                 gain, attenuation, external_gain=None, duration=None,
                 oversampling=None, symbols=True, symbol_rate=None):
        self.owner = owner
        self.rate = rate
        self.symbol_rate = symbol_rate or rate
        self.bandwidth = bandwidth
        self.gain = gain
        self.attenuation = attenuation
        self.external_gain = external_gain or DeferredDoubleSpinBox()
        for control in (self.attenuation, self.external_gain):
            control.setRange(-200.0, 200.0)
            control.setDecimals(1)
            control.setSuffix(" dB")
        self.bandwidth.setRange(0.2, 56.0)
        self.bandwidth.setDecimals(3)
        self.bandwidth.setSuffix(" MHz")
        self.gain.setSuffix(" dB")
        if isinstance(self.gain, QtWidgets.QDoubleSpinBox):
            self.gain.setDecimals(1)
        self.match_bandwidth = QtWidgets.QCheckBox("Match Sample Rate")
        self.match_bandwidth.receiver_setup = self
        self.applied_bandwidth = QtWidgets.QLabel()
        self.correction = QtWidgets.QLabel()
        self.duration = duration
        self.oversampling = oversampling
        if symbols and oversampling is not None:
            for index in range(oversampling.count()):
                oversampling.setItemText(index, f"{oversampling.itemData(index)} samples/symbol")
        self.sample_rate_label = QtWidgets.QLabel()
        self.record_length_label = QtWidgets.QLabel()
        self.usable_bandwidth_label = QtWidgets.QLabel()
        self._syncing = False
        self._last_symbol_rate = self.symbol_rate()
        if duration is not None:
            duration.setRange(0.001, 1_000_000.0)
            duration.setDecimals(9)
            self.length = DeferredDoubleSpinBox()
            self.length.setRange(0.001, 1_000_000.0)
            self.length.setDecimals(3)
            self.length.setValue(duration.value())
            self.unit = QtWidgets.QComboBox()
            self.swap_iq = QtWidgets.QCheckBox("Swap I/Q")
            self.unit.addItems(("ms", "Symbols") if symbols else ("ms",))
            self.length.valueChanged.connect(self._length_changed)
            self.unit.currentTextChanged.connect(self._unit_changed)
            duration.valueChanged.connect(self._model_changed)
            duration.hide()
        for control in (self.bandwidth, self.gain, self.attenuation, self.external_gain):
            control.valueChanged.connect(self.refresh)
        self.match_bandwidth.toggled.connect(self.refresh)
        if oversampling is not None:
            oversampling.currentIndexChanged.connect(self.refresh)
        self.refresh()

    def power_rows(self, form):
        form.addRow("Internal Gain", self.gain)
        form.addRow("External ATT", self.attenuation)
        form.addRow("External Gain", self.external_gain)
        form.addRow("Input Correction", self.correction)

    def bandwidth_rows(self, form):
        form.addRow("RF Bandwidth", self.bandwidth)
        form.addRow(self.match_bandwidth)
        form.addRow("Applied RF Bandwidth", self.applied_bandwidth)

    def capture_page(self):
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)
        configure_form(form)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.length, 1)
        row.addWidget(self.unit)
        form.addRow("Capture Length", row)
        form.addRow("Sample Rate", self.oversampling)
        form.addRow("Resulting Sample Rate", self.sample_rate_label)
        form.addRow("Record Length", self.record_length_label)
        form.addRow("Usable I/Q Bandwidth", self.usable_bandwidth_label)
        form.addRow(self.swap_iq)
        return page

    def power_correction(self):
        return InputPowerCorrection(
            internal_gain_db=self.gain.value(),
            external_attenuation_db=self.attenuation.value(),
            external_gain_db=self.external_gain.value(),
        )

    def _length_changed(self, _value=None):
        if self._syncing:
            return
        milliseconds = (self.length.value() if self.unit.currentText() == "ms"
                        else self.length.value() / self.symbol_rate() * 1000.0)
        self.duration.setValue(milliseconds)
        self.refresh()

    def _model_changed(self, _value=None):
        self._syncing = True
        value = self.duration.value()
        if self.unit.currentText() == "Symbols":
            value *= self.symbol_rate() / 1000.0
        self.length.setValue(value)
        self._syncing = False
        self.refresh()

    def _unit_changed(self, unit):
        # Changing units preserves the duration, not the displayed number.
        factor = self.symbol_rate() / 1000.0 if unit == "Symbols" else 1.0
        self._syncing = True
        self.length.setRange(self.duration.minimum() * factor, self.duration.maximum() * factor)
        self._syncing = False
        self._model_changed()

    def refresh(self, _value=None):
        symbol_rate = self.symbol_rate()
        if self.duration is not None and symbol_rate != self._last_symbol_rate:
            self._last_symbol_rate = symbol_rate
            if self.unit.currentText() == "Symbols":
                self._length_changed()
        sample_rate = self.rate()
        self.bandwidth.setEnabled(not self.match_bandwidth.isChecked())
        if self.match_bandwidth.isChecked():
            blocker = QtCore.QSignalBlocker(self.bandwidth)
            self.bandwidth.setValue(sample_rate / 1e6)
            del blocker
        recording = getattr(self.owner, "_recording", None) or getattr(self.owner, "recording", None)
        session = getattr(self.owner, "session", None)
        if session is not None:
            recording = session.recording
        if recording is None:
            recording = getattr(self.owner, "_reference_capture_recording", None)
        actual = getattr(recording, "metadata", {}).get("actual_rf_bandwidth_hz") if recording is not None else None
        self.applied_bandwidth.setText(
            f"{float(actual) / 1e6:.3f} MHz (last capture)" if actual is not None else "— (not acquired)"
        )
        correction = self.power_correction()
        self.correction.setText(f"{correction.input_correction_db:+.1f} dB "
                                "(Ext ATT - Internal Gain - Ext Gain)")
        self.sample_rate_label.setText(f"{sample_rate / 1e6:.3f} MS/s")
        self.usable_bandwidth_label.setText(f"{min(0.8 * sample_rate, self.bandwidth.value() * 1e6) / 1e6:.3f} MHz")
        if self.duration is not None:
            self.record_length_label.setText(f"{round(self.duration.value() * sample_rate / 1000):,} samples")

    def values(self):
        values = {"match_rf_bandwidth": self.match_bandwidth.isChecked(),
                  "external_gain_db": self.external_gain.value()}
        if self.duration is not None:
            values["capture_unit"] = self.unit.currentText()
            values["swap_iq"] = self.swap_iq.isChecked()
        return values

    def apply(self, values):
        self.match_bandwidth.setChecked(bool(values.get("match_rf_bandwidth", False)))
        self.external_gain.setValue(float(values.get("external_gain_db", self.external_gain.value())))
        if self.duration is not None:
            self.swap_iq.setChecked(bool(values.get("swap_iq", False)))
            self.unit.setCurrentText(str(values.get("capture_unit", "ms")))
            self._model_changed()
        self.refresh()


def standardize_frontend(form, *extra_forms):
    """One shared ordering, spacing and checkbox presentation."""
    rows = []
    for source in (form,) + extra_forms:
        while source.rowCount():
            row = source.takeRow(0)
            label = row.labelItem.widget() if row.labelItem else None
            field = row.fieldItem.widget() if row.fieldItem else None
            if field is None:
                continue
            title = label.text() if label else ""
            if title == "Analysis Channel":
                label.hide()
                label.deleteLater()
                label = None
                title = ""
            if title == "LO Offset" and isinstance(field, QtWidgets.QCheckBox):
                field.setText("Enable (Experimental)")
            key = title or (field.text() if isinstance(field, QtWidgets.QCheckBox) else "")
            rows.append((key, label, field))
    order = ("Center Frequency", "RF Bandwidth", "Match Sample Rate", "Applied RF Bandwidth",
             "LO Offset", "Offset Frequency", "Resolved LO", "Internal Gain", "External ATT",
             "External Gain", "Input Correction", "Enable Analysis Channel", "Analysis Center",
             "Analysis Bandwidth", "Apply Analysis Bandwidth to Power", "Apply Analysis Bandwidth to Spectrum")
    rows.sort(key=lambda row: order.index(row[0]) if row[0] in order else len(order))
    for _key, label, field in rows:
        if label is None:
            form.addRow(field)
        else:
            form.addRow(label, field)
    configure_form(form)


def display_form(page, owner, *, psk=False, reference=None):
    layout = page.layout()
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.setParent(owner)
            widget.hide()
    # Qt cannot replace a widget's layout while the old layout is alive.
    holder = QtWidgets.QWidget()
    holder.setLayout(layout)
    form = QtWidgets.QFormLayout(page)
    configure_form(form)
    owner.config_show_symbols.setText("")
    form.addRow("Show Symbol Points", owner.config_show_symbols)
    trace = QtWidgets.QComboBox()
    trace.addItems(("Flat", "Density"))
    trace.setCurrentText("Density" if owner.config_density.isChecked() else "Flat")
    trace.currentTextChanged.connect(lambda value: owner.config_density.setChecked(value == "Density"))
    owner.config_density.toggled.connect(lambda checked: trace.setCurrentText("Density" if checked else "Flat"))
    owner.config_symbol_trace = trace
    form.addRow("Symbol Plot Trace", trace)
    form.addRow("Density Spread", owner.config_density_spread)
    if psk:
        form.addRow("PSK Symbol Plot", owner.config_psk_mode)
    form.addRow("FSK Symbol Plot", owner.config_fsk_mode)
    if reference is not None:
        form.addRow("GFSK Modulation Reference", reference)
    for index in range(form.rowCount()):
        item = form.itemAt(index, QtWidgets.QFormLayout.ItemRole.FieldRole)
        if item and item.widget():
            item.widget().show()


def build_trigger_page(owner, *, burst=False):
    """Expose common widgets through the existing save/DSP attribute names."""
    aliases = {"source": "acquisition_trigger_source_combo", "level": "acquisition_trigger_level_spin",
               "slope": "acquisition_trigger_slope_combo", "offset": "acquisition_trigger_offset_spin",
               "hysteresis": "acquisition_trigger_hysteresis_spin", "enabled": "iq_power_trigger_check",
               "burst_level": "iq_power_trigger_level_spin", "burst_hysteresis": "iq_power_trigger_hysteresis_spin",
               "average": "iq_power_trigger_average_spin", "dropout": "iq_power_trigger_dropout_spin",
               "holdoff": "iq_power_trigger_holdoff_spin", "start_offset": "iq_power_trigger_offset_spin",
               "limit": "iq_power_trigger_limit_result_check"}
    controls = TriggerControls(burst=burst)
    owner._trigger_setup = controls
    for key, name in aliases.items():
        setattr(owner, name, controls.controls[key])
    return controls.page


class TriggerControls:
    """Identical acquisition and post-capture trigger form for receivers."""

    def __init__(self, *, source=TriggerKind.FREE_RUN, level=-20.0, burst=False,
                 symbol_rate=lambda: 1e6, time_units=False):
        self.symbol_rate = symbol_rate
        self.time_units = time_units
        self.page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(self.page)
        configure_form(form)
        self.controls = {}
        for heading, fields in (
            ("Acquisition Trigger", ("source", "level", "slope", "offset", "hysteresis")),
            ("Post-capture Burst Search", ("enabled", "burst_level", "burst_hysteresis", "average", "dropout", "holdoff", "start_offset", "limit")),
        ):
            label = QtWidgets.QLabel(heading)
            label.setStyleSheet("font-weight: bold;")
            form.addRow(label)
            for key in fields:
                if key in {"source", "slope"}:
                    widget = QtWidgets.QComboBox()
                    choices = (("Free Run", TriggerKind.FREE_RUN.value), ("I/Q Power", TriggerKind.POWER_LEVEL.value)) if key == "source" else tuple((s.value.capitalize(), s.value) for s in TriggerSlope)
                    for title, value in choices:
                        widget.addItem(title, value)
                elif key in {"enabled", "limit"}:
                    widget = QtWidgets.QCheckBox("Burst Search On" if key == "enabled" else "Limit Result Range to Active Interval")
                    widget.setChecked(burst if key == "enabled" else True)
                else:
                    widget = DeferredDoubleSpinBox()
                    low, high, default, decimals, suffix = {
                        "level": (-200, 100, level, 2, " dBm"),
                        "burst_level": (-200, 100, level, 2, " dBm"),
                        "hysteresis": (0, 50, 3, 1, " dB"),
                        "burst_hysteresis": (0, 60, 3, 2, " dB"),
                        "offset": (-1e6, 1e6, 0, 3, " sym"),
                        "average": (0, 1000, 1, 2, " sym"),
                        "dropout": (0, 1e6, 8, 2, " sym"),
                        "holdoff": (0, 1e6, 0, 2, " sym"),
                        "start_offset": (-1e6, 1e6, 0, 3, " sym"),
                    }[key]
                    if time_units and suffix == " sym":
                        suffix = " ms"
                        default = default / symbol_rate() * 1000
                        decimals = 6
                    widget.setRange(low, high)
                    widget.setDecimals(decimals)
                    widget.setSuffix(suffix)
                    widget.setValue(default)
                self.controls[key] = widget
                labels = {"source": "Trigger Source", "level": "Level", "slope": "Slope", "offset": "Trigger Offset", "hysteresis": "Hysteresis", "burst_level": "Level", "burst_hysteresis": "Hysteresis", "average": "Envelope Average", "dropout": "Drop-Out Time", "holdoff": "Holdoff", "start_offset": "Search Start Offset"}
                if key in labels:
                    form.addRow(labels[key], widget)
                else:
                    form.addRow(widget)
        self.controls["source"].setCurrentIndex(self.controls["source"].findData(source.value))
        self.controls["offset"].setToolTip(
            "Positive offsets start after the trigger; negative offsets retain pretrigger IQ. "
            "At zero, Pluto capture retains 16 symbols to protect the first ramp/preamble."
        )
        self.controls["limit"].setToolTip(
            "Keep only packets/symbols ending inside the active power interval. "
            "Disable for OOK/PPM when valid data includes low-power gaps or a quiet final chip."
        )
        self.controls["source"].currentIndexChanged.connect(self.sync)
        self.sync()

    def sync(self, _value=None):
        enabled = self.controls["source"].currentData() == TriggerKind.POWER_LEVEL.value
        for key in ("level", "slope", "offset", "hysteresis"):
            self.controls[key].setEnabled(enabled)

    def values(self):
        return {key: (widget.currentData() if isinstance(widget, QtWidgets.QComboBox) else widget.isChecked() if isinstance(widget, QtWidgets.QCheckBox) else widget.value()) for key, widget in self.controls.items()}

    def apply(self, values):
        for key, value in values.items():
            if key not in self.controls:
                continue
            widget = self.controls[key]
            if isinstance(widget, QtWidgets.QComboBox):
                widget.setCurrentIndex(widget.findData(value))
            elif isinstance(widget, QtWidgets.QCheckBox):
                widget.setChecked(bool(value))
            else:
                widget.setValue(float(value))
        self.sync()

    def acquisition_settings(self):
        values = self.values()
        return dict(trigger_source=TriggerKind(values["source"]),
                    trigger_level_dbm=values["level"],
                    trigger_slope=TriggerSlope(values["slope"]),
                    trigger_offset_s=values["offset"] / (1000 if self.time_units else self.symbol_rate()),
                    trigger_hysteresis_db=values["hysteresis"])

    def burst_settings(self):
        v = self.values()
        factor = self.symbol_rate() / 1000 if self.time_units else 1
        return IQPowerTriggerSettings(enabled=v["enabled"], level_dbm=v["burst_level"], hysteresis_db=v["burst_hysteresis"], envelope_average_symbols=v["average"] * factor, dropout_symbols=v["dropout"] * factor, holdoff_symbols=v["holdoff"] * factor, search_start_offset_symbols=v["start_offset"] * factor, limit_result_to_active_interval=v["limit"])
