"""Qt numeric inputs that defer range enforcement until form submission."""

from __future__ import annotations

import re
from typing import TypeAlias

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets


_INTEGER_RE = re.compile(r"[+-]?\d+")
_FLOAT_RE = re.compile(r"[+-]?(?:(?:\d+(?:[.,]\d*)?)|(?:[.,]\d+))")
_INCOMPLETE_NUMBERS = {"", "+", "-", ".", ",", "+.", "-.", "+,", "-,"}
_INVALID_STYLE = """
QAbstractSpinBox[numericInputValid="false"] {
    border: 1px solid #d9534f;
    background-color: #4a2528;
}
"""


class _DeferredNumericMixin:
    """Keep parseable out-of-range text without committing it as a value."""

    def _initialize_deferred_validation(self) -> None:
        self.setKeyboardTracking(True)
        self.setProperty("numericInputValid", True)
        self.setStyleSheet(self.styleSheet() + _INVALID_STYLE)
        self.lineEdit().textChanged.connect(self._input_text_changed)
        self._refresh_input_validity()

    def _input_text_changed(self, _text: str) -> None:
        # User edits are evaluated immediately. Programmatic range/value
        # changes are refreshed again by their explicit overrides below,
        # after QAbstractSpinBox has finished updating its internal state.
        self._refresh_input_validity()

    def setRange(self, minimum, maximum) -> None:
        super().setRange(minimum, maximum)
        self._refresh_input_validity()

    def setMinimum(self, minimum) -> None:
        super().setMinimum(minimum)
        self._refresh_input_validity()

    def setMaximum(self, maximum) -> None:
        super().setMaximum(maximum)
        self._refresh_input_validity()

    def setValue(self, value) -> None:
        super().setValue(value)
        self._refresh_input_validity()

    def setPrefix(self, prefix: str) -> None:
        super().setPrefix(prefix)
        self._refresh_input_validity()

    def setSuffix(self, suffix: str) -> None:
        super().setSuffix(suffix)
        self._refresh_input_validity()

    def _numeric_text(self, text: str) -> str:
        value = str(text).strip()
        prefix = self.prefix()
        suffix = self.suffix()
        if prefix and value.startswith(prefix):
            value = value[len(prefix) :]
        if suffix and value.endswith(suffix):
            value = value[: -len(suffix)]
        return value.strip()

    def _is_numeric_or_incomplete(self, text: str) -> bool:
        value = self._numeric_text(text)
        if value in _INCOMPLETE_NUMBERS:
            return True
        expression = _INTEGER_RE if isinstance(self, QtWidgets.QSpinBox) else _FLOAT_RE
        return expression.fullmatch(value) is not None

    def validate(self, text: str, position: int):
        state, normalized, normalized_position = super().validate(text, position)
        if (
            state == QtGui.QValidator.State.Invalid
            and self._is_numeric_or_incomplete(normalized)
        ):
            state = QtGui.QValidator.State.Intermediate
        return state, normalized, normalized_position

    def has_valid_input(self) -> bool:
        state, _text, _position = self.validate(self.text(), len(self.text()))
        return state == QtGui.QValidator.State.Acceptable

    def validation_message(self) -> str:
        label = self.accessibleName() or self.objectName() or "Value"
        return (
            f"{label}: enter a value from {self.minimum():g} to "
            f"{self.maximum():g}{self.suffix()}."
        )

    def revert_invalid_text(self) -> None:
        if self.has_valid_input():
            return
        committed = self.textFromValue(self.value())
        self.lineEdit().setText(f"{self.prefix()}{committed}{self.suffix()}")

    def _refresh_input_validity(self, _text: str = "") -> None:
        valid = self.has_valid_input()
        if self.property("numericInputValid") == valid:
            return
        self.setProperty("numericInputValid", valid)
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def focusOutEvent(self, event: QtGui.QFocusEvent) -> None:
        if self.has_valid_input():
            super().focusOutEvent(event)
        else:
            # QAbstractSpinBox would otherwise replace Intermediate text with
            # the previous value.  Retain it so the form can point out the
            # exact invalid entry when the user presses Apply/OK/Run.
            QtWidgets.QWidget.focusOutEvent(self, event)

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        self._refresh_input_validity()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if (
            event.key()
            in {QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter}
            and not self.has_valid_input()
        ):
            self._refresh_input_validity()
            event.accept()
            return
        super().keyPressEvent(event)

    def stepBy(self, steps: int) -> None:
        if self.has_valid_input():
            super().stepBy(steps)


class DeferredSpinBox(_DeferredNumericMixin, QtWidgets.QSpinBox):
    """Integer spin box that permits temporary out-of-range editing."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._initialize_deferred_validation()


class DeferredDoubleSpinBox(_DeferredNumericMixin, QtWidgets.QDoubleSpinBox):
    """Floating-point spin box that permits temporary out-of-range editing."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._initialize_deferred_validation()


DeferredNumericInput: TypeAlias = DeferredSpinBox | DeferredDoubleSpinBox


def invalid_numeric_inputs(root: QtWidgets.QWidget) -> tuple[DeferredNumericInput, ...]:
    """Return every deferred numeric child whose current text is not valid."""

    controls: list[DeferredNumericInput] = []
    if isinstance(root, (DeferredSpinBox, DeferredDoubleSpinBox)):
        controls.append(root)
    controls.extend(root.findChildren(DeferredSpinBox))
    controls.extend(root.findChildren(DeferredDoubleSpinBox))
    return tuple(control for control in controls if not control.has_valid_input())


def revert_invalid_numeric_inputs(root: QtWidgets.QWidget) -> None:
    """Restore incomplete edits to their last committed values on Cancel/Close."""

    for control in invalid_numeric_inputs(root):
        control.revert_invalid_text()


def ensure_valid_numeric_inputs(
    root: QtWidgets.QWidget,
    *,
    title: str = "Invalid Numeric Input",
    show_message: bool = True,
) -> bool:
    """Block form completion while retaining and highlighting invalid text."""

    invalid = invalid_numeric_inputs(root)
    if not invalid:
        return True
    first = invalid[0]
    first.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)
    first.lineEdit().selectAll()
    if show_message:
        details = "\n".join(control.validation_message() for control in invalid[:8])
        if len(invalid) > 8:
            details += f"\n... and {len(invalid) - 8} more"
        QtWidgets.QMessageBox.warning(root, title, details)
    return False


def _run_numeric_dialog(
    parent: QtWidgets.QWidget | None,
    title: str,
    label: str,
    control: DeferredNumericInput,
) -> tuple[float | int, bool]:
    dialog = QtWidgets.QDialog(parent)
    dialog.setWindowTitle(str(title))
    layout = QtWidgets.QVBoxLayout(dialog)
    prompt = QtWidgets.QLabel(str(label))
    layout.addWidget(prompt)
    control.setAccessibleName(str(label))
    layout.addWidget(control)
    buttons = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.StandardButton.Ok
        | QtWidgets.QDialogButtonBox.StandardButton.Cancel
    )

    def accept_if_valid() -> None:
        if ensure_valid_numeric_inputs(dialog):
            dialog.accept()

    buttons.accepted.connect(accept_if_valid)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    control.selectAll()
    control.setFocus()
    accepted = dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted
    return control.value(), accepted


def get_deferred_double(
    parent: QtWidgets.QWidget | None,
    title: str,
    label: str,
    value: float = 0.0,
    minValue: float = -2147483647.0,
    maxValue: float = 2147483647.0,
    decimals: int = 1,
) -> tuple[float, bool]:
    """QInputDialog.getDouble-compatible entry with deferred range checks."""

    control = DeferredDoubleSpinBox()
    control.setRange(float(minValue), float(maxValue))
    control.setDecimals(int(decimals))
    control.setValue(float(value))
    result, accepted = _run_numeric_dialog(parent, title, label, control)
    return float(result), accepted


def get_deferred_int(
    parent: QtWidgets.QWidget | None,
    title: str,
    label: str,
    value: int = 0,
    minValue: int = -2147483647,
    maxValue: int = 2147483647,
    step: int = 1,
) -> tuple[int, bool]:
    """QInputDialog.getInt-compatible entry with deferred range checks."""

    control = DeferredSpinBox()
    control.setRange(int(minValue), int(maxValue))
    control.setSingleStep(int(step))
    control.setValue(int(value))
    result, accepted = _run_numeric_dialog(parent, title, label, control)
    return int(result), accepted
