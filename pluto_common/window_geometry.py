"""Shared main-window geometry policy, independent of internal pane layouts."""

from pyqtgraph.Qt import QtCore, QtWidgets


WINDOW_GEOMETRY_KEY = "startup/geometry"
MINIMUM_WINDOW_SIZE = (960, 640)


def restore_window_geometry(
    window: QtWidgets.QMainWindow, settings, *, restore: bool = True
) -> None:
    """Apply the common minimum and optionally restore the outer window only."""
    window.setMinimumSize(*MINIMUM_WINDOW_SIZE)
    if restore:
        geometry = settings.value(WINDOW_GEOMETRY_KEY)
        if isinstance(geometry, (QtCore.QByteArray, bytes, bytearray)):
            window.restoreGeometry(QtCore.QByteArray(geometry))


def save_window_geometry(window: QtWidgets.QMainWindow, settings) -> None:
    """Save position and size without persisting docks, splitters or tabs."""
    settings.setValue(WINDOW_GEOMETRY_KEY, window.saveGeometry())
    settings.sync()
