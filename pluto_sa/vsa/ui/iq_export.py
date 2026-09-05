"""Shared IQ-recording export workflow for Generic and dedicated VSA modes."""

from __future__ import annotations

from pathlib import Path

from pyqtgraph.Qt import QtCore, QtWidgets

from pluto_sa.vsa.dc import apply_robust_dc_removal
from pluto_sa.vsa.model import IQRecording
from pluto_sa.vsa.sources import FileIQSource


def export_iq_recording(
    parent: QtWidgets.QWidget,
    recording: IQRecording | None,
    preferences: QtCore.QSettings,
    *,
    directory_key: str = "directories/iq",
) -> Path | None:
    """Run the common raw/DC-removed NPZ export dialog and preserve metadata."""

    status_bar = getattr(parent, "statusBar")()
    if recording is None:
        status_bar.showMessage("No IQ recording is available to export")
        return None
    processing, accepted = QtWidgets.QInputDialog.getItem(
        parent,
        "Export IQ Recording",
        "IQ processing:",
        (
            "Raw capture",
            "Software DC removed (full-rate capture)",
        ),
        0,
        False,
    )
    if not accepted:
        return None
    stored = preferences.value(directory_key, "", type=str)
    initial_directory = (
        stored if stored and Path(stored).is_dir() else str(Path.cwd())
    )
    path_text, _ = QtWidgets.QFileDialog.getSaveFileName(
        parent,
        "Export IQ Recording",
        initial_directory,
        "NumPy IQ recording (*.npz)",
    )
    if not path_text:
        return None
    path = Path(path_text)
    if not path.suffix:
        path = path.with_suffix(".npz")
    try:
        remove_dc = str(processing).startswith("Software DC removed")
        exported = apply_robust_dc_removal(recording) if remove_dc else recording
        FileIQSource.save_npz(path, exported)
        preferences.setValue(directory_key, str(path.resolve().parent))
        preferences.sync()
        mode = "software DC removed" if remove_dc else "raw"
        status_bar.showMessage(
            f"IQ recording exported ({mode}) - {path.name}"
        )
    except (OSError, ValueError) as error:
        QtWidgets.QMessageBox.critical(parent, "IQ Export Error", str(error))
        return None
    return path
