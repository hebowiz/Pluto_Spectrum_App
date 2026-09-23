"""Shared VSA decoded-packet export dialog for BT and DECT."""
from pyqtgraph.Qt import QtWidgets
from pluto_common.file_dialogs import file_dialog_path, remember_file_directory
from pluto_vsg.packet_import import project_from_packet, packet_export_error
from pluto_vsg.persistence import save_project


def update_export_action(action, packet):
    error = packet_export_error(packet)
    action.setEnabled(error is None)
    action.setToolTip(error or "Save received fields; unknown RF/timing settings use template defaults")


def export_packet_project(parent, packet, preferences):
    try:
        project = project_from_packet(packet)
    except (ValueError, TypeError, KeyError, StopIteration) as error:
        QtWidgets.QMessageBox.warning(parent, "Export VSG Project", str(error))
        return
    path, _ = QtWidgets.QFileDialog.getSaveFileName(
        parent, "Export VSG Project",
        file_dialog_path(preferences, "directories/vsg_project", filename="received_packet.pvsg.json"),
        "Pluto VSG Project (*.pvsg.json)",
    )
    if not path:
        return
    if not path.lower().endswith(".pvsg.json"):
        path += ".pvsg.json"
    try:
        save_project(path, project)
    except (OSError, ValueError) as error:
        QtWidgets.QMessageBox.warning(parent, "Export VSG Project", str(error))
        return
    remember_file_directory(preferences, "directories/vsg_project", path)
    parent.statusBar().showMessage("VSG project saved; received fields retained, RF/timing uses template defaults")
