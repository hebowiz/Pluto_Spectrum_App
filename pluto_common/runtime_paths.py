"""Runtime-safe writable paths for source and frozen application builds."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def is_frozen_application() -> bool:
    """Return whether the process is running from a PyInstaller bundle."""

    return bool(getattr(sys, "frozen", False))


def application_data_dir(application_name: str) -> Path:
    """Return and create a per-user writable application data directory."""

    local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
    base = Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
    path = base / "PlutoSpectrumApp" / application_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def diagnostic_log_path(application_name: str, filename: str) -> Path:
    """Return a writable diagnostic log path for an application."""

    return application_data_dir(application_name) / filename
