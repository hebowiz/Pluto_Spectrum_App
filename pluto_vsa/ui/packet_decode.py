"""Packet content views shared by VSA and VSG (no DSP or instrument access)."""

from collections.abc import Iterable

import numpy as np
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from pluto_protocol.model import PacketField
from pluto_protocol.dect.common import dect_p_range
from .measurement_chrome import (
    DedicatedPacketAnalysisTree, apply_dedicated_table_style, dedicated_status_color,
)


def apply_analysis_font(widget):
    """Use the ordinary VSA content font, not the containing panel-title font."""
    widget.setFont(QtGui.QFont(QtWidgets.QApplication.font()))


def payload_field(fields: Iterable[PacketField]) -> PacketField | None:
    for field in fields:
        if field.field_id in {"payload", "payload_body", "b_field"}:
            return field
        found = payload_field(field.children)
        if found is not None:
            return found
    return None


class PacketAnalysisTree(DedicatedPacketAnalysisTree):
    def __init__(self, parent=None, *, dect=False):
        super().__init__(
            ("Field", "Value", "Bit Range", "DECT Symbols", "Status") if dect
            else ("Field", "Value", "Stream", "Bit Range", "Status"),
            (125, 115, 68, 92, 55) if dect else (120, 120, 105, 72, 55),
            expand_columns=(0, 1), parent=parent,
        )


class AutoHeightIssuesTable(QtWidgets.QTableWidget):
    def resizeEvent(self, event):
        super().resizeEvent(event)
        QtCore.QTimer.singleShot(0, self.resizeRowsToContents)


def field_bit_range(field, bit_offset=0):
    start, stop = int(field.start_bit) - bit_offset, int(field.stop_bit) - bit_offset - 1
    return "N/A" if stop < start else str(start) if start == stop else f"{start}\N{EN DASH}{stop}"


def bluetooth_tree_item(field, *, stream="Packet", bit_offset=0):
    value = "\N{EM DASH}" if field.field_id == "payload" and field.children else str(field.value)
    bit_range = field_bit_range(field, bit_offset)
    texts = (field.name, value, stream, bit_range, field.status.value)
    item = QtWidgets.QTreeWidgetItem(texts)
    item.setTextAlignment(1, QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
    color = dedicated_status_color(field.status)
    if color:
        item.setForeground(4, QtGui.QBrush(color))
    for column, text in enumerate((field.name, str(field.value),
                                  f"{stream}: {field.meaning}" if field.meaning else stream,
                                  bit_range, field.status.value)):
        item.setToolTip(column, text)
    for child in field.children:
        item.addChild(bluetooth_tree_item(child, stream=stream, bit_offset=bit_offset))
    return item


def dect_tree_item(field, bits, p0_internal_bit=0):
    stop = min(field.stop_bit, bits.size)
    value = "" if field.value is None else str(field.value)
    bit_range = "N/A" if stop <= field.start_bit else str(field.start_bit) if stop == field.start_bit + 1 else f"{field.start_bit}\N{EN DASH}{stop - 1}"
    start_p, stop_p = dect_p_range(field.start_bit, stop, p0_internal_bit)
    symbols = "N/A" if stop_p <= start_p else f"p{start_p}" if stop_p == start_p + 1 else f"p{start_p}\N{EN DASH}p{stop_p - 1}"
    texts = (field.name, value, bit_range, symbols, field.status.value)
    item = QtWidgets.QTreeWidgetItem(texts)
    color = dedicated_status_color(field.status)
    if color is not None:
        item.setForeground(4, QtGui.QBrush(color))
    for column, text in enumerate(texts):
        item.setToolTip(column, f"{text}\n{field.meaning}" if field.meaning else text)
    for child in field.children:
        item.addChild(dect_tree_item(child, bits, p0_internal_bit))
    return item


class PacketDecodeTabs(QtWidgets.QTabWidget):
    """One implementation of Decode / Payload Hex / Issues for both apps."""

    def __init__(self, parent=None, *, dect=False):
        super().__init__(parent)
        apply_analysis_font(self)
        self.decode_tree = PacketAnalysisTree(dect=dect)
        self.payload_text = QtWidgets.QPlainTextEdit(readOnly=True)
        self.issues_table = AutoHeightIssuesTable(0, 4)
        self.issues_table.setHorizontalHeaderLabels(("Severity", "Code", "Message", "Bit Range"))
        apply_dedicated_table_style(self.issues_table)
        self.issues_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.issues_table.setWordWrap(True)
        self.issues_table.setTextElideMode(QtCore.Qt.TextElideMode.ElideNone)
        self.issues_table.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header = self.issues_table.horizontalHeader()
        for column in (0, 1, 3):
            header.setSectionResizeMode(column, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Stretch)
        for label, widget in (("Decode", self.decode_tree), ("Payload Hex", self.payload_text), ("Issues", self.issues_table)):
            apply_analysis_font(widget)
            self.addTab(widget, label)

    def clear_packet(self):
        self.decode_tree.clear()
        self.payload_text.clear()
        self.issues_table.setRowCount(0)

    def render_packet(self, packet, *, p0_internal_bit=0, dect_bits=None):
        self.clear_packet()
        dect = packet.protocol_id.startswith("dect.")
        self.decode_tree.setHeaderLabels(("Field", "Value", "Bit Range", "DECT Symbols", "Status") if dect else ("Field", "Value", "Stream", "Bit Range", "Status"))
        for field in packet.root_fields:
            if dect:
                item = dect_tree_item(field, packet.raw_bits if dect_bits is None else dect_bits, p0_internal_bit)
            else:
                stream = {"training": "Training symbols", "control_header": "Control Header", "payload": "PDU+Payload"}.get(field.field_id, "Packet") if packet.protocol_id == "bluetooth.hdt" else "Packet"
                bit_offset = field.start_bit if packet.protocol_id == "bluetooth.hdt" and field.field_id == "payload" else 0
                item = bluetooth_tree_item(field, stream=stream, bit_offset=bit_offset)
            self.decode_tree.addTopLevelItem(item)
        self.decode_tree.expandToDepth(2 if dect else 1)
        QtCore.QTimer.singleShot(0, self.decode_tree._fit_columns)
        payload = payload_field(packet.root_fields)
        if payload is None:
            self.payload_text.setPlainText("Payload field was not decoded")
        else:
            values = np.packbits(np.pad(payload.raw_bits, (0, (-payload.raw_bits.size) % 8)), bitorder="big" if dect else "little")
            lines = [f"{offset:04X}: " + " ".join(f"{int(value):02X}" for value in values[offset:offset + 16]) for offset in range(0, values.size, 16)]
            self.payload_text.setPlainText("\n".join(lines) or "(empty payload)")
        self.issues_table.setRowCount(len(packet.issues))
        for row, issue in enumerate(packet.issues):
            bit_range = "--" if issue.start_bit is None else f"{issue.start_bit}:{issue.stop_bit}"
            for column, value in enumerate((issue.severity.value, issue.code, issue.message, bit_range)):
                self.issues_table.setItem(row, column, QtWidgets.QTableWidgetItem(str(value)))
        self.issues_table.resizeRowsToContents()
