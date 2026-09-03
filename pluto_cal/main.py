"""Application entry point for Pluto CAL."""

from __future__ import annotations

import sys

from PySide6 import QtWidgets

from .ui import PlutoCalMainWindow


def main() -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Pluto CAL")
    app.setOrganizationName("Pluto Spectrum App")
    window = PlutoCalMainWindow()
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
