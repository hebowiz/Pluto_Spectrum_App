"""Auto/manual editor for fields preserved from a received packet."""
from pyqtgraph.Qt import QtWidgets, QtGui, QtCore
from pluto_vsg.packet_fields import field_widths


class PacketFieldsDialog(QtWidgets.QDialog):
    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Received Packet Fields")
        self.controls = {}
        self.manual_fields = dict(project.manual_packet_fields)
        layout = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel(
            "Manual values are retained when other packet settings change. "
            "Select Auto to recalculate a field. Values below are canonical binary bit order; "
            "CRC/HEC failures are permitted. Verify Packet checks the regenerated structure."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        content = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(content)
        for name, width in field_widths(project).items():
            row = QtWidgets.QWidget()
            controls = QtWidgets.QHBoxLayout(row)
            controls.setContentsMargins(0, 0, 0, 0)
            auto = QtWidgets.QCheckBox("Auto")
            auto.setChecked(name not in project.manual_packet_fields)
            value = QtWidgets.QLineEdit(project.manual_packet_fields.get(name, "0"*width))
            value.setMaxLength(width)
            value.setValidator(QtGui.QRegularExpressionValidator(
                QtCore.QRegularExpression(f"[01]{{0,{width}}}"), value,
            ))
            value.setEnabled(not auto.isChecked())
            auto.toggled.connect(lambda checked, edit=value: edit.setEnabled(not checked))
            controls.addWidget(auto)
            controls.addWidget(value, 1)
            form.addRow(f"{name.replace('_', ' ')} ({width} bits)", row)
            self.controls[name] = (auto, value, width)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        layout.addWidget(scroll)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
        )
        buttons.accepted.connect(self._accept_fields)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.resize(820, 520)

    def _accept_fields(self):
        fields = {}
        for name, (auto, edit, width) in self.controls.items():
            if auto.isChecked():
                continue
            value = edit.text()
            if len(value) != width or any(b not in "01" for b in value):
                QtWidgets.QMessageBox.warning(self, "Received Packet Fields", f"{name} requires {width} binary bits")
                return
            fields[name] = value
        self.manual_fields = fields
        self.accept()
