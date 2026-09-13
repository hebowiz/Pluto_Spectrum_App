"""Shared right-side controls for all VSA analysis workspaces."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets


CONTROL_PANEL_WIDTH = 240
BUTTON_FONT_SCALE = 1.45


@dataclass(frozen=True)
class PanelCommand:
    """One command displayed by the common VSA control panel."""

    label: str
    callback: Callable[[], None]
    action: QtGui.QAction | None = None


@dataclass(frozen=True)
class WorkspacePanelSpec:
    """Mode-specific commands consumed by :class:`VSAControlPanel`."""

    mode_id: str
    mode_label: str
    setup: Sequence[PanelCommand]
    single: PanelCommand
    continuous: PanelCommand
    refresh: PanelCommand
    reset: PanelCommand
    files: Sequence[PanelCommand]


class VSAControlPanel(QtWidgets.QFrame):
    """SA-style hierarchical panel shared by every VSA mode."""

    mode_requested = QtCore.Signal(str)
    preset_requested = QtCore.Signal()
    device_requested = QtCore.Signal()
    recall_requested = QtCore.Signal()
    save_requested = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("vsa-control-panel")
        self.setFixedWidth(CONTROL_PANEL_WIDTH)
        self.setStyleSheet(
            "QFrame#vsa-control-panel { background-color: #1c1c1c; }"
            "QGroupBox { color: white; border: 1px solid #555; margin-top: 12px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }"
            "QPushButton { background-color: #303030; color: white; "
            "border: 1px solid #666; padding: 8px; }"
            "QPushButton:hover { background-color: #3c3c3c; }"
            "QPushButton:checked { background-color: #176b87; border-color: #37b7dc; }"
            "QPushButton:disabled { color: #888; background-color: #252525; }"
        )
        self._spec: WorkspacePanelSpec | None = None
        self._history: list[QtWidgets.QWidget] = []
        self._bound_actions: list[tuple[QtGui.QAction, Callable[[], None]]] = []
        self.buttons: dict[str, QtWidgets.QPushButton] = {}

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        self.title_label = QtWidgets.QLabel("Main Menu")
        font = QtGui.QFont(self.title_label.font())
        font.setPointSizeF(font.pointSizeF() * 1.4)
        font.setBold(True)
        self.title_label.setFont(font)
        self.title_label.setStyleSheet("color: white; padding: 4px 2px;")
        layout.addWidget(self.title_label)

        self.stack = QtWidgets.QStackedWidget()
        layout.addWidget(self.stack, 1)
        footer = QtWidgets.QHBoxLayout()
        footer.addStretch(1)
        self.back_button = self._make_button("Back")
        self.back_button.clicked.connect(self.navigate_back)
        footer.addWidget(self.back_button)
        layout.addLayout(footer)
        self.back_button.hide()
        self._install_back_filter(self)

    def set_workspace(self, spec: WorkspacePanelSpec) -> None:
        self._disconnect_actions()
        self._spec = spec
        self._history.clear()
        self.buttons.clear()
        while self.stack.count():
            widget = self.stack.widget(0)
            self.stack.removeWidget(widget)
            widget.deleteLater()

        self.main_page = self._build_main_page(spec)
        self.mode_page = self._build_mode_page(spec.mode_id)
        self.system_page = self._build_system_page()
        self.preset_page = self._build_preset_page()
        self.file_page = self._build_file_page(spec.files)
        for page in (
            self.main_page,
            self.mode_page,
            self.system_page,
            self.preset_page,
            self.file_page,
        ):
            self.stack.addWidget(page)
            self._install_back_filter(page)
        self._show_page("Main Menu", self.main_page, remember=False)

    def show_main_menu(self) -> None:
        if self._spec is None:
            return
        self._history.clear()
        self._show_page("Main Menu", self.main_page, remember=False)

    def navigate_back(self) -> None:
        if not self._history:
            self.show_main_menu()
            return
        title, page = self._history.pop()
        self.title_label.setText(title)
        self.stack.setCurrentWidget(page)
        self.back_button.setVisible(bool(self._history) or page is not self.main_page)

    def _show_page(
        self,
        title: str,
        page: QtWidgets.QWidget,
        *,
        remember: bool = True,
    ) -> None:
        current = self.stack.currentWidget()
        if remember and current is not None and current is not page:
            self._history.append((self.title_label.text(), current))
        self.title_label.setText(title)
        self.stack.setCurrentWidget(page)
        self.back_button.setVisible(page is not self.main_page)

    def _build_main_page(self, spec: WorkspacePanelSpec) -> QtWidgets.QWidget:
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        analyzer = self._group("ANALYZER SETUP")
        analyzer_layout = analyzer.layout()
        mode_button = self._make_button("Analyzer Mode")
        mode_button.clicked.connect(
            lambda: self._show_page("Analyzer Mode", self.mode_page)
        )
        analyzer_layout.addWidget(mode_button)
        self.buttons["Analyzer Mode"] = mode_button
        for command in spec.setup:
            button = self._command_button(command)
            analyzer_layout.addWidget(button)
            self.buttons[command.label] = button
        layout.addWidget(analyzer)

        sweep = self._group("SWEEP CONTROL")
        sweep_layout = sweep.layout()
        for key, command in (
            ("Continuous", spec.continuous),
            ("Single", spec.single),
            ("Refresh Analysis", spec.refresh),
            ("Reset", spec.reset),
        ):
            button = self._command_button(command, display_label=key)
            sweep_layout.addWidget(button)
            self.buttons[key] = button
        watched_actions = tuple(
            item.action
            for item in (spec.single, spec.continuous, spec.refresh, spec.reset)
            if item.action is not None
        )

        def sync_sweep_group() -> None:
            single_running = (
                spec.single.action is not None
                and "Stop" in spec.single.action.text()
            )
            continuous_running = (
                spec.continuous.action is not None
                and "Stop" in spec.continuous.action.text()
            )
            busy = single_running or continuous_running
            refresh_enabled = (
                spec.refresh.action is None or spec.refresh.action.isEnabled()
            )
            reset_enabled = spec.reset.action is None or spec.reset.action.isEnabled()
            self.buttons["Refresh Analysis"].setEnabled(
                refresh_enabled and not busy
            )
            self.buttons["Reset"].setEnabled(reset_enabled and not busy)

        for action in watched_actions:
            action.changed.connect(sync_sweep_group)
            self._bound_actions.append((action, sync_sweep_group))
        sync_sweep_group()
        layout.addWidget(sweep)

        system = self._group("SYSTEM")
        system_button = self._make_button("System")
        system_button.clicked.connect(lambda: self._show_page("System", self.system_page))
        system.layout().addWidget(system_button)
        self.buttons["System"] = system_button
        layout.addWidget(system)
        layout.addStretch(1)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(content)
        return scroll

    def _build_mode_page(self, active_mode: str) -> QtWidgets.QWidget:
        page = self._simple_page()
        modes = (
            ("Generic VSA", "generic"),
            ("Bluetooth", "bluetooth"),
            ("DECT", "dect"),
            ("ADS-B 1090ES", "adsb1090"),
        )
        for label, mode_id in modes:
            button = self._make_button(label)
            button.setCheckable(True)
            button.setChecked(mode_id == active_mode)
            button.clicked.connect(
                lambda _checked=False, selected=mode_id: self.mode_requested.emit(selected)
            )
            page.layout().addWidget(button)
            self.buttons[f"mode:{mode_id}"] = button
        page.layout().addStretch(1)
        return page

    def _build_system_page(self) -> QtWidgets.QWidget:
        page = self._simple_page()
        preset = self._make_button("Preset")
        preset.clicked.connect(lambda: self._show_page("Preset", self.preset_page))
        device = self._make_button("Device")
        device.clicked.connect(self.device_requested.emit)
        recall = self._make_button("Recall")
        recall.clicked.connect(self.recall_requested.emit)
        save = self._make_button("Save")
        save.clicked.connect(self.save_requested.emit)
        file_button = self._make_button("File")
        file_button.clicked.connect(lambda: self._show_page("File", self.file_page))
        for key, button in (
            ("Preset", preset),
            ("Device", device),
            ("Recall", recall),
            ("Save", save),
            ("File", file_button),
        ):
            page.layout().addWidget(button)
            self.buttons[key] = button
        page.layout().addStretch(1)
        return page

    def _build_preset_page(self) -> QtWidgets.QWidget:
        page = self._simple_page()
        button = self._make_button("Default")
        button.clicked.connect(self.preset_requested.emit)
        page.layout().addWidget(button)
        page.layout().addStretch(1)
        self.buttons["Default"] = button
        return page

    def _build_file_page(self, commands: Sequence[PanelCommand]) -> QtWidgets.QWidget:
        page = self._simple_page()
        for command in commands:
            button = self._command_button(command)
            page.layout().addWidget(button)
            self.buttons[f"file:{command.label}"] = button
        page.layout().addStretch(1)
        return page

    def _command_button(
        self,
        command: PanelCommand,
        *,
        display_label: str | None = None,
    ) -> QtWidgets.QPushButton:
        button = self._make_button(display_label or command.label)
        # QPushButton.clicked emits its checked state.  Do not forward that
        # positional bool to commands whose callbacks use default arguments
        # (for example the measurement-config page name), otherwise False
        # replaces the requested page when the button is clicked.
        button.clicked.connect(
            lambda _checked=False, callback=command.callback: callback()
        )
        if command.action is not None:
            action = command.action

            def sync() -> None:
                button.setEnabled(action.isEnabled())
                if display_label == "Continuous":
                    button.setText(
                        "Stopping..."
                        if "Stop" in action.text() and not action.isEnabled()
                        else ("Stop" if "Stop" in action.text() else "Continuous")
                    )
                elif display_label == "Single":
                    button.setText(
                        "Stopping..."
                        if "Stop" in action.text() and not action.isEnabled()
                        else ("Stop" if "Stop" in action.text() else "Single")
                    )

            action.changed.connect(sync)
            self._bound_actions.append((action, sync))
            sync()
        return button

    def _disconnect_actions(self) -> None:
        for action, callback in self._bound_actions:
            try:
                action.changed.disconnect(callback)
            except (RuntimeError, TypeError):
                pass
        self._bound_actions.clear()

    def _make_button(self, text: str) -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton(text)
        font = QtGui.QFont(button.font())
        font.setPointSizeF(font.pointSizeF() * BUTTON_FONT_SCALE)
        font.setBold(True)
        button.setFont(font)
        button.setMinimumHeight(50)
        return button

    @staticmethod
    def _simple_page() -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        return page

    def _group(self, title: str) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox(title)
        font = QtGui.QFont(group.font())
        font.setBold(True)
        group.setFont(font)
        layout = QtWidgets.QVBoxLayout(group)
        layout.setSpacing(8)
        return group

    def _install_back_filter(self, root: QtWidgets.QWidget) -> None:
        root.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.NoContextMenu)
        root.installEventFilter(self)
        for child in root.findChildren(QtWidgets.QWidget):
            child.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.NoContextMenu)
            child.installEventFilter(self)

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if (
            event.type() == QtCore.QEvent.Type.MouseButtonPress
            and isinstance(event, QtGui.QMouseEvent)
            and event.button() == QtCore.Qt.MouseButton.RightButton
            and self.stack.currentWidget() is not self.main_page
        ):
            self.navigate_back()
            return True
        return super().eventFilter(watched, event)
