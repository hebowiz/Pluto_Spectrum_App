from __future__ import annotations

from pathlib import Path

from pluto_common.runtime_paths import application_data_dir, diagnostic_log_path


def test_application_data_dir_uses_local_app_data(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

    path = application_data_dir("PlutoVSG")

    assert path == tmp_path / "PlutoSpectrumApp" / "PlutoVSG"
    assert path.is_dir()


def test_diagnostic_log_path_is_inside_application_directory(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

    path = diagnostic_log_path("PlutoVSG", "trace.log")

    assert path == tmp_path / "PlutoSpectrumApp" / "PlutoVSG" / "trace.log"
