"""Independent VSA application entry point."""

from __future__ import annotations

import pyqtgraph as pg

from pluto_sa.vsa.ui.main_window import VSAWindow


def build_vsa_window() -> VSAWindow:
    return VSAWindow()


def main() -> int:
    app = pg.mkQApp("Pluto VSA")
    window = build_vsa_window()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
