"""Package moves must preserve launch dispatch and Python object identity."""

from __future__ import annotations

import ast
import importlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("entrypoint", "implementation"),
    [
        ("pluto_rtsa", "pluto_rtsa.main"),
        ("pluto_vsa", "pluto_vsa.main"),
        ("pluto_vsg", "pluto_vsg.main"),
        ("pluto_sa.main", "pluto_rtsa.main"),
        ("pluto_sa.vsa.main", "pluto_vsa.main"),
        ("pluto_sa.standards.adsb1090.main", "pluto_vsa.standards.adsb1090.main"),
    ],
)
def test_module_launch_dispatches_to_application_main(entrypoint, implementation, monkeypatch):
    module = importlib.import_module(implementation)
    calls = []

    def main():
        calls.append(True)
        return 17

    monkeypatch.setattr(module, "main", main)
    with pytest.raises(SystemExit) as raised:
        runpy.run_module(entrypoint, run_name="__main__")
    assert raised.value.code == 17
    assert calls == [True]


@pytest.mark.parametrize("legacy_first", [True, False])
def test_legacy_imports_share_modules_classes_and_enums_in_fresh_process(legacy_first):
    # Loading an old path must not create a second Enum/dataclass definition.
    code = f'''
import importlib
import pickle

pairs = [
    ("pluto_sa.config.spectrum_config", "pluto_common.config.spectrum_config"),
    ("pluto_sa.config.input_frontend", "pluto_common.config.input_frontend"),
    ("pluto_sa.config.session_state", "pluto_rtsa.config.session_state"),
    ("pluto_sa.modes.analyzer_mode", "pluto_common.config.analyzer_mode"),
    ("pluto_sa.sdr.trigger", "pluto_common.sdr.trigger"),
    ("pluto_sa.signal.detector", "pluto_rtsa.signal.detector"),
    ("pluto_sa.vsa.model", "pluto_vsa.model"),
    ("pluto_sa.vsa.profiles.bluetooth_br", "pluto_vsa.profiles.bluetooth_br"),
    ("pluto_sa.standards.adsb1090.model", "pluto_vsa.standards.adsb1090.model"),
]
for old, new in pairs:
    first, second = (old, new) if {legacy_first!r} else (new, old)
    first_module = importlib.import_module(first)
    canonical_spec = first_module.__spec__
    second_module = importlib.import_module(second)
    assert first_module is second_module, (old, new)
    assert first_module.__spec__ is canonical_spec
    assert first_module.__spec__.name == new

from pluto_sa.modes.analyzer_mode import AnalyzerMode as OldMode
from pluto_common.config.analyzer_mode import AnalyzerMode
from pluto_sa.vsa.model import IQRecording as OldRecording
from pluto_vsa.model import IQRecording
assert OldMode is AnalyzerMode
assert OldRecording is IQRecording
assert pickle.loads(pickle.dumps(OldMode.REALTIME_SA)) is AnalyzerMode.REALTIME_SA
assert pickle.loads(b"cpluto_sa.modes.analyzer_mode\\nAnalyzerMode\\n.") is AnalyzerMode
import pluto_sa.vsa.model as old_model
assert old_model is importlib.import_module("pluto_vsa.model")
'''
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("application", ["rtsa", "vsa", "vsg"])
def test_packaging_entrypoint_imports_application_and_writes_smoke_report(application, tmp_path):
    entrypoint = ROOT / "packaging" / "entrypoints" / f"pluto_{application}_entry.py"
    report = tmp_path / "smoke.json"
    environment = dict(
        os.environ,
        PLUTO_APP_SMOKE_TEST="1",
        PLUTO_APP_SMOKE_REPORT=str(report),
        QT_QPA_PLATFORM="offscreen",
    )
    # Match PyInstaller's entrypoint-directory-before-project import order.
    code = (
        "import runpy, sys; "
        f"sys.path.insert(0, {str(entrypoint.parent)!r}); "
        f"runpy.run_path({str(entrypoint)!r}, run_name='__main__')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=environment,
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["application"] == f"Pluto_{application.upper()}"
    assert payload["libiio_version"]
    if application == "vsa":
        assert payload["qt_webengine"] is True


def test_applications_do_not_depend_on_legacy_namespace_or_common_on_apps():
    for package in ("pluto_rtsa", "pluto_vsa", "pluto_vsg", "pluto_common"):
        for path in (ROOT / package).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            for node in ast.walk(tree):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules = [node.module]
                for module in modules:
                    root = module.split(".")[0]
                    assert root != "pluto_sa", (path, module)
                    if package == "pluto_common":
                        assert root not in {"pluto_rtsa", "pluto_vsa", "pluto_vsg"}, (path, module)
                    if package == "pluto_vsa":
                        assert root != "pluto_rtsa", (path, module)
