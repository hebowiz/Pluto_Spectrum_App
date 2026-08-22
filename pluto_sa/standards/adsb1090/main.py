"""Standalone entry point for the ADS-B 1090ES workspace."""

from __future__ import annotations

import pyqtgraph as pg

from pluto_sa.standards.adsb1090.ui import ADSB1090Window


def main() -> int:
    app = pg.mkQApp("Pluto VSA - ADS-B 1090ES")
    window = ADSB1090Window()
    window.application_close_requested.connect(window.close)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
