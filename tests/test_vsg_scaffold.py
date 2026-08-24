import os
from dataclasses import replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pyqtgraph as pg

from pluto_vsg.model import create_default_project, validate_project
from pluto_vsg.profiles import bluetooth_br_edr_project
from pluto_vsg.ui.main_window import PlutoVSGWindow


def test_default_vsg_project_is_valid() -> None:
    project = create_default_project()

    assert project.samples_per_symbol == 8
    assert validate_project(project) == ()


def test_invalid_vsg_project_reports_model_path() -> None:
    project = replace(create_default_project(), sample_rate_hz=0.0)

    issues = validate_project(project)

    assert any(issue.path == "sample_rate_hz" for issue in issues)


def test_bluetooth_profile_expands_into_common_fields() -> None:
    project = bluetooth_br_edr_project()

    assert [packet_field.name for packet_field in project.fields] == [
        "Access Code",
        "Header",
        "Payload",
    ]
    assert validate_project(project) == ()


def test_vsg_window_starts_with_composer_shell() -> None:
    pg.mkQApp("Pluto VSG scaffold test")
    window = PlutoVSGWindow()
    try:
        assert "Pluto VSG" in window.windowTitle()
        assert window.field_table.rowCount() == 1
        assert [action.text() for action in window.menuBar().actions()] == [
            "File",
            "Edit",
            "Waveform",
            "Graphics",
            "Output",
            "Tools",
            "Help",
        ]
    finally:
        window.close()
