"""PyInstaller entry point for Pluto VSA."""

from __future__ import annotations

import os
import json
from pathlib import Path


def main() -> int:
    if os.environ.get("PLUTO_APP_SMOKE_TEST") == "1":
        import iio
        import pluto_sa.vsa.main  # noqa: F401
        from PySide6 import QtWebEngineCore, QtWebEngineWidgets  # noqa: F401

        report = os.environ.get("PLUTO_APP_SMOKE_REPORT")
        if report:
            Path(report).write_text(
                json.dumps(
                    {
                        "application": "Pluto_VSA",
                        "libiio_version": list(iio.version),
                        "qt_webengine": True,
                    }
                ),
                encoding="utf-8",
            )
        return 0
    from pluto_sa.vsa.main import main as application_main

    return application_main()


if __name__ == "__main__":
    raise SystemExit(main())
