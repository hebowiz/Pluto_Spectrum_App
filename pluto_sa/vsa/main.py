"""Independent VSA application entry point."""

from __future__ import annotations

import pyqtgraph as pg

from pluto_sa.vsa.ui.application_window import PlutoAnalysisWindow


def build_vsa_window() -> PlutoAnalysisWindow:
    return PlutoAnalysisWindow()


def main() -> int:
    app = pg.mkQApp("Pluto VSA")
    window = build_vsa_window()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
