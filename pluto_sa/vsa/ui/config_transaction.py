"""Isolated settings editors using the same mode-specific UI builders.

Draft workspaces have no recording, physical receiver, or persistent settings.
Their callbacks can update dependent widgets without touching the live mode.
Only a successfully accepted draft is copied back to the live workspace.
"""

from copy import deepcopy
from inspect import signature

from pyqtgraph.Qt import QtCore, QtWidgets


class DraftPreferences:
    """The small QSettings interface used by VSA, backed only by memory."""

    def __init__(self, original):
        self._values = {key: original.value(key) for key in original.allKeys()}

    def value(self, key, default=None, *, type=None):
        value = self._values.get(key, default)
        if type is None or value is None:
            return value
        if type is bool and isinstance(value, str):
            return value.lower() in {"true", "1"}
        return type(value)

    def setValue(self, key, value):
        self._values[key] = value

    def contains(self, key):
        return key in self._values

    def remove(self, key):
        self._values.pop(key, None)

    def allKeys(self):
        return list(self._values)

    def sync(self):
        pass


class DraftReceiver:
    """No settings editor can accidentally operate the physical receiver."""

    def capture_single(self, *args, **kwargs):
        raise RuntimeError("Close Config and use SWEEP CONTROL to capture IQ")

    def stop_stream(self):
        pass

    def close(self):
        pass


def collect_settings(workspace):
    collector = getattr(workspace, "_config_values", None)
    return deepcopy(collector() if collector else workspace._meas_config_values())


def apply_settings(workspace, values):
    apply = getattr(workspace, "_apply_config_values", None)
    if apply is None:
        apply = workspace._apply_meas_config_values
    apply(deepcopy(values))


def create_draft_editor(workspace):
    kwargs = {
        "preferences": DraftPreferences(workspace._preferences),
        "pluto_source": DraftReceiver(),
    }
    if "owns_pluto_source" in signature(type(workspace)).parameters:
        kwargs["owns_pluto_source"] = False
    draft = type(workspace)(**kwargs)
    dialog = (getattr(draft, "_meas_config_dialog", None)
              or getattr(draft, "_config_dialog", None)
              or draft._setup_dialog)
    dialog._is_draft = True
    dialog.settings_owner = draft
    dialog.settings_validator = lambda: collect_settings(draft)
    apply_settings(draft, collect_settings(workspace))
    sync_display = getattr(draft, "_sync_display_config_controls", None)
    if sync_display is not None:
        sync_display()
    session = getattr(workspace, "session", None)
    draft._reference_capture_recording = (
        session.recording if session is not None else
        getattr(workspace, "_capture_recording", None)
        or getattr(workspace, "_recording", None)
        or getattr(workspace, "recording", None)
    )
    draft._common_setup.refresh()
    if "Sweep / Run" in dialog.page_names:
        page = dialog.stack.widget(dialog.page_names.index("Sweep / Run"))
        for button in page.findChildren(QtWidgets.QPushButton):
            button.setEnabled(False)
    dialog.setParent(workspace, QtCore.Qt.WindowType.Dialog)
    dialog.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
    return draft, dialog
