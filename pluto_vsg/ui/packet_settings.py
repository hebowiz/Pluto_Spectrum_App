"""Shared controls for dedicated packet/waveform settings dialogs."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from pyqtgraph.Qt import QtWidgets


RF_TIMING_TAB = "RF / Timing"
FIELDS_TAB = "Fields"


def scroll_form(
    rows: Iterable[tuple[str, QtWidgets.QWidget]],
) -> QtWidgets.QScrollArea:
    """Return the common scrollable, full-width packet-settings form."""

    content = QtWidgets.QWidget()
    form = QtWidgets.QFormLayout(content)
    form.setFieldGrowthPolicy(
        QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
    )
    for label, control in rows:
        form.addRow(label, control)
    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    scroll.setWidget(content)
    return scroll


def packet_settings_tabs(
    rf_timing_rows: Iterable[tuple[str, QtWidgets.QWidget]],
    field_rows: Iterable[tuple[str, QtWidgets.QWidget]],
) -> QtWidgets.QTabWidget:
    """Build the two-tab layout shared by every packet generator."""

    tabs = QtWidgets.QTabWidget()
    tabs.addTab(scroll_form(rf_timing_rows), RF_TIMING_TAB)
    tabs.addTab(scroll_form(field_rows), FIELDS_TAB)
    return tabs


class SymbolTimeControl(QtWidgets.QWidget):
    """Keep a symbol-domain editor and its time equivalent side by side."""

    def __init__(
        self,
        control: QtWidgets.QAbstractSpinBox,
        symbol_rate_hz: Callable[[], float],
    ) -> None:
        super().__init__()
        self.control = control
        self._symbol_rate_hz = symbol_rate_hz
        if isinstance(control, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
            if not control.suffix():
                control.setSuffix(" symbols")
        self.time_label = QtWidgets.QLabel()
        self.time_label.setMinimumWidth(115)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(control, 1)
        layout.addWidget(self.time_label)
        control.valueChanged.connect(self.refresh)
        self.refresh()

    def refresh(self, _value=None) -> None:
        rate = max(float(self._symbol_rate_hz()), 1.0)
        symbols = float(self.control.value())
        time_us = symbols / rate * 1e6
        if abs(time_us) >= 1000.0:
            self.time_label.setText(f"= {time_us / 1000.0:.6g} ms")
        else:
            self.time_label.setText(f"= {time_us:.6g} us")


def carrier_selector(
    carriers: Iterable[tuple[str, float]],
    nominal_frequency_hz: float,
) -> QtWidgets.QComboBox:
    """Build a carrier list and preserve a non-plan value as Custom."""

    combo = QtWidgets.QComboBox()
    selected = -1
    for label, frequency_hz in carriers:
        combo.addItem(label, float(frequency_hz))
        if abs(float(frequency_hz) - float(nominal_frequency_hz)) < 0.5:
            selected = combo.count() - 1
    if selected < 0:
        combo.addItem(
            f"Custom / loaded {float(nominal_frequency_hz) / 1e6:.6f} MHz",
            float(nominal_frequency_hz),
        )
        selected = combo.count() - 1
    combo.setCurrentIndex(selected)
    return combo


def bluetooth_classic_carriers() -> tuple[tuple[str, float], ...]:
    return tuple(
        (f"Channel {channel} — {2402 + channel} MHz", (2402 + channel) * 1e6)
        for channel in range(79)
    )


def bluetooth_le_carriers() -> tuple[tuple[str, float], ...]:
    frequencies = {
        **{channel: 2404 + 2 * channel for channel in range(11)},
        **{channel: 2428 + 2 * (channel - 11) for channel in range(11, 37)},
        37: 2402,
        38: 2426,
        39: 2480,
    }
    return tuple(
        (
            f"Channel {channel} — {frequencies[channel]} MHz"
            + (" (Advertising)" if channel >= 37 else ""),
            frequencies[channel] * 1e6,
        )
        for channel in range(40)
    )


def wifi_24ghz_carriers() -> tuple[tuple[str, float], ...]:
    return tuple(
        (f"Channel {channel} — {2407 + 5 * channel} MHz", (2407 + 5 * channel) * 1e6)
        for channel in range(1, 14)
    )
