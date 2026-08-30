"""PyInstaller entry point for Pluto VSG.

The file name intentionally differs from the ``pluto_vsg`` package name so
PyInstaller does not resolve this script in place of the application package.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def main() -> int:
    if os.environ.get("PLUTO_APP_SMOKE_TEST") == "1":
        import iio
        import pluto_vsg.main  # noqa: F401

        report = os.environ.get("PLUTO_APP_SMOKE_REPORT")
        if report:
            Path(report).write_text(
                json.dumps({"application": "Pluto_VSG", "libiio_version": list(iio.version)}),
                encoding="utf-8",
            )
        return 0
    from pluto_vsg.main import main as application_main

    return application_main()


if __name__ == "__main__":
    raise SystemExit(main())
