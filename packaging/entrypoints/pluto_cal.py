"""PyInstaller entry point for Pluto CAL."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys


def _prefer_application_package() -> None:
    """Prevent this required script name from shadowing ``pluto_cal/``.

    PyInstaller executes the file as ``__main__`` and is unaffected.  Direct
    source-tree smoke tests put ``packaging/entrypoints`` first on sys.path,
    where this file would otherwise be imported as a non-package named
    ``pluto_cal``.
    """

    if getattr(sys, "frozen", False):
        return
    entrypoint_dir = Path(__file__).resolve().parent
    sys.path[:] = [
        item
        for item in sys.path
        if Path(item or os.curdir).resolve() != entrypoint_dir
    ]
    repository_root = str(Path(__file__).resolve().parents[2])
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)


def main() -> int:
    _prefer_application_package()
    if os.environ.get("PLUTO_APP_SMOKE_TEST") == "1":
        import iio
        import pluto_cal.main  # noqa: F401

        report = os.environ.get("PLUTO_APP_SMOKE_REPORT")
        if report:
            Path(report).write_text(
                json.dumps(
                    {
                        "application": "Pluto_CAL",
                        "libiio_version": list(iio.version),
                    }
                ),
                encoding="utf-8",
            )
        return 0
    from pluto_cal.main import main as application_main

    return application_main()


if __name__ == "__main__":
    raise SystemExit(main())
