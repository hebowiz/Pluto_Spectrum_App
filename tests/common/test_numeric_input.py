import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from pluto_common.numeric_input import (
    DeferredDoubleSpinBox,
    DeferredSpinBox,
    ensure_valid_numeric_inputs,
    revert_invalid_numeric_inputs,
)


def _enter_text(control, text: str) -> None:
    control.lineEdit().selectAll()
    control.lineEdit().setText(text)


def test_double_spin_retains_temporary_out_of_range_text() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    control = DeferredDoubleSpinBox()
    control.setRange(1.0, 100.0)
    control.setValue(50.0)
    assert control.property("numericInputValid") is True
    _enter_text(control, "999")
    assert control.text() == "999"
    assert control.value() == 50.0
    assert not control.has_valid_input()
    event = QtGui.QFocusEvent(QtCore.QEvent.Type.FocusOut)
    control.focusOutEvent(event)
    app.processEvents()
    assert control.text() == "999"


def test_integer_spin_commits_after_text_returns_to_range() -> None:
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    control = DeferredSpinBox()
    control.setRange(10, 20)
    control.setValue(12)
    _enter_text(control, "200")
    assert not ensure_valid_numeric_inputs(control, show_message=False)
    _enter_text(control, "18")
    control.interpretText()
    assert ensure_valid_numeric_inputs(control, show_message=False)
    assert control.value() == 18


def test_form_validation_finds_invalid_numeric_child() -> None:
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    form = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(form)
    first = DeferredDoubleSpinBox()
    first.setRange(-10.0, 10.0)
    first.setAccessibleName("Frequency offset")
    second = DeferredSpinBox()
    second.setRange(1, 5)
    layout.addWidget(first)
    layout.addWidget(second)
    _enter_text(first, "11")
    assert not ensure_valid_numeric_inputs(form, show_message=False)
    assert first.property("numericInputValid") is False
    revert_invalid_numeric_inputs(form)
    assert first.has_valid_input()
    assert first.value() == 0.0
