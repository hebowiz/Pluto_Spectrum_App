"""Application-local visual constants pending a shared SA/VSA/VSG theme module."""

from __future__ import annotations

from pyqtgraph.Qt import QtGui


TRACE_COLOR = "#ffff00"
ACCENT_COLOR = "#00e5ff"
FIELD_BOUNDARY_COLOR = "#ff4de1"
FIELD_MINOR_BOUNDARY_COLOR = "#ff9f43"
PACKET_END_COLOR = "#f5f5f5"
PANEL_TITLE_SCALE = 1.3


def panel_title_font(base: QtGui.QFont) -> QtGui.QFont:
    font = QtGui.QFont(base)
    font.setBold(True)
    if font.pointSizeF() > 0.0:
        font.setPointSizeF(font.pointSizeF() * PANEL_TITLE_SCALE)
    return font
