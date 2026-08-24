"""Pluto VSG application entry point."""

from __future__ import annotations

import pyqtgraph as pg

from pluto_vsg.ui.main_window import PlutoVSGWindow


def build_vsg_window() -> PlutoVSGWindow:
    return PlutoVSGWindow()


def main() -> int:
    app = pg.mkQApp("Pluto VSG")
    window = build_vsg_window()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
