"""Wi-Fi settings built from the common transactional Meas Config controls."""
from pyqtgraph.Qt import QtWidgets
from pluto_common.numeric_input import DeferredDoubleSpinBox
from pluto_vsa.ui.setup_controls import ReceiverSetupControls, TriggerControls, standardize_frontend
from pluto_vsa.ui.measurement_config_dialog import HierarchicalMeasConfigDialog
from pluto_vsa.ui.measurement_chrome import SymbolDensitySpread


def number(value, low, high, suffix="", decimals=3):
    control = DeferredDoubleSpinBox()
    control.setRange(low, high)
    control.setDecimals(decimals)
    control.setSuffix(suffix)
    control.setValue(value)
    return control


def page(rows):
    widget = QtWidgets.QWidget()
    form = QtWidgets.QFormLayout(widget)
    for label, control in rows:
        form.addRow(label, control)
    return widget


def build_config(owner):
    owner.channel_combo = QtWidgets.QComboBox()
    for channel in range(1,14):
        owner.channel_combo.addItem(f"Channel {channel} — {2407+5*channel} MHz", channel)
    owner.channel_combo.setCurrentIndex(5)
    owner.center_spin = number(2437, 1, 6000, " MHz", 6)
    owner.channel_combo.currentIndexChanged.connect(lambda: owner.center_spin.setValue(2407+5*owner.channel_combo.currentData()))
    owner.sample_rate_combo = QtWidgets.QComboBox()
    for value in (20,40):
        owner.sample_rate_combo.addItem(f"{value} MS/s", value*1e6)
    owner.sample_rate_combo.setCurrentIndex(1)
    owner.duration_spin = number(10, .01, 500, " ms")
    owner.bandwidth_spin = number(30, .2, 56, " MHz")
    owner.gain_spin = number(30, -3, 73, " dB", 1)
    owner.attenuation_spin = number(0, -200, 200, " dB", 1)
    owner.analysis_check = QtWidgets.QCheckBox("Enable Analysis Channel")
    owner.analysis_center_spin = number(2437, 1, 6000, " MHz", 6)
    owner.analysis_bandwidth_spin = number(20, 16.25, 30, " MHz")
    owner.power_filter_check = QtWidgets.QCheckBox("Apply Analysis Bandwidth to Power")
    owner.spectrum_filter_check = QtWidgets.QCheckBox("Apply Analysis Bandwidth to Spectrum")
    owner.lo_offset_spin = number(0, -30, 30, " MHz")
    owner._common_setup = ReceiverSetupControls(owner, rate=lambda: owner.sample_rate_combo.currentData(),
        bandwidth=owner.bandwidth_spin, gain=owner.gain_spin, attenuation=owner.attenuation_spin,
        duration=owner.duration_spin, oversampling=owner.sample_rate_combo, symbols=False)
    owner._trigger_controls = TriggerControls(symbol_rate=lambda: 250e3, time_units=True)
    # Wi-Fi packet acquisition uses STF detection, not the FSK post-capture
    # burst search. Keep only the shared hardware acquisition trigger rows.
    trigger_form = owner._trigger_controls.page.layout()
    for row in range(6,trigger_form.rowCount()):
        trigger_form.setRowVisible(row,False)
    # Keep the persisted boolean compatible with existing Wi-Fi settings.
    owner.density_check = QtWidgets.QCheckBox(owner)
    owner.density_check.hide()
    owner.symbol_trace_combo = QtWidgets.QComboBox()
    owner.symbol_trace_combo.addItems(("Flat", "Density"))
    owner.symbol_trace_combo.currentTextChanged.connect(
        lambda value: owner.density_check.setChecked(value == "Density"))
    owner.density_check.toggled.connect(
        lambda checked: owner.symbol_trace_combo.setCurrentText("Density" if checked else "Flat"))
    owner.density_spread_combo = QtWidgets.QComboBox()
    for spread in SymbolDensitySpread:
        owner.density_spread_combo.addItem(spread.value, spread)
    owner.density_spread_combo.setCurrentIndex(owner.density_spread_combo.findData(SymbolDensitySpread.MAXIMUM))
    owner.diagnostics_check = QtWidgets.QCheckBox("Show synchronization diagnostics")
    frontend = page((("Center Frequency",owner.center_spin), ("LO Offset",owner.lo_offset_spin),
                     ("Analysis Channel",owner.analysis_check), ("Analysis Center",owner.analysis_center_spin),
                     ("Analysis Bandwidth",owner.analysis_bandwidth_spin),
                     ("",owner.power_filter_check), ("",owner.spectrum_filter_check)))
    owner._common_setup.bandwidth_rows(frontend.layout())
    owner._common_setup.power_rows(frontend.layout())
    standardize_frontend(frontend.layout())
    sweep = page(())
    for title, callback in (("Single",owner._toggle_capture), ("Continuous",owner._toggle_continuous_capture),
                            ("Refresh Analysis",owner.refresh)):
        button = QtWidgets.QPushButton(title)
        button.clicked.connect(callback)
        sweep.layout().addRow(button)
    owner._meas_config_dialog = HierarchicalMeasConfigDialog(owner, (
        ("Signal Description", page((("PHY",QtWidgets.QLabel("Non-HT OFDM / 20 MHz")),
                                    ("Channel / Nominal Center",owner.channel_combo),
                                    ("Rate",QtWidgets.QLabel("Auto from L-SIG"))))),
        ("Input / Frontend",frontend), ("Signal Capture",owner._common_setup.capture_page()),
        ("Trigger",owner._trigger_controls.page),
        ("Display",page((("Symbol Plot Trace",owner.symbol_trace_combo),
                         ("Density Spread",owner.density_spread_combo),
                         ("Modulation",QtWidgets.QLabel("L-SIG / DATA: OFDM index × subcarrier EVM")),
                         ("Result Summary",owner.diagnostics_check)))), ("Sweep / Run",sweep)),
        window_title="Wi-Fi Meas Config",
        standard_buttons=QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
    owner._meas_config_dialog.accepted.connect(owner._save_config)


NUMBERS = ("center_spin", "duration_spin", "bandwidth_spin", "gain_spin", "attenuation_spin",
           "analysis_center_spin", "analysis_bandwidth_spin", "lo_offset_spin")
CHECKS = ("analysis_check", "power_filter_check", "spectrum_filter_check",
          "density_check", "diagnostics_check")


def collect(owner):
    values = {name: getattr(owner,name).value() for name in NUMBERS}
    values.update({name: getattr(owner,name).isChecked() for name in CHECKS})
    values.update(channel=owner.channel_combo.currentData(), sample_rate=owner.sample_rate_combo.currentData(),
                  receiver=owner._common_setup.values(), trigger=owner._trigger_controls.values(),
                  symbol_density_spread=owner.density_spread_combo.currentData())
    return values


def apply(owner, values):
    # Validate before mutating live controls, including externally recalled files.
    spread = SymbolDensitySpread(values.get("symbol_density_spread", owner.density_spread_combo.currentData()))
    for name in NUMBERS:
        if name in values:
            value = float(values[name])
            control = getattr(owner,name)
            if not control.minimum() <= value <= control.maximum():
                raise ValueError(f"Out-of-range Wi-Fi setting: {name}")
    for name, control in (("channel",owner.channel_combo), ("sample_rate",owner.sample_rate_combo)):
        if name in values and control.findData(values[name]) < 0:
            raise ValueError(f"Unsupported Wi-Fi {name}")
    for name, control in (("channel",owner.channel_combo), ("sample_rate",owner.sample_rate_combo)):
        if name in values:
            control.setCurrentIndex(control.findData(values[name]))
    for name in NUMBERS:
        if name in values:
            getattr(owner,name).setValue(float(values[name]))
    for name in CHECKS:
        if name in values:
            getattr(owner,name).setChecked(bool(values[name]))
    owner.density_spread_combo.setCurrentIndex(owner.density_spread_combo.findData(spread))
    owner._common_setup.apply(values.get("receiver",{}))
    owner._trigger_controls.apply(values.get("trigger",{}))
