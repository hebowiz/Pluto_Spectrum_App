"""Make bundled libiio and its native dependencies discoverable by ctypes."""

from __future__ import annotations

import os
import sys
from pathlib import Path


_DLL_DIRECTORY_HANDLES: list[object] = []


def _activate_bundled_dll_directory() -> None:
    bundle_root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    candidates = (bundle_root, bundle_root / "libiio")
    existing = [path for path in candidates if path.is_dir()]
    if not existing:
        return
    os.environ["PATH"] = os.pathsep.join(
        [*(str(path) for path in existing), os.environ.get("PATH", "")]
    )
    if hasattr(os, "add_dll_directory"):
        for path in existing:
            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(path)))


_activate_bundled_dll_directory()
