"""Shared right-side controls for all VSA analysis workspaces."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from pluto_common.control_panel import (
    CONTROL_PANEL_WIDTH,
    ControlPanelNavigator,
    add_back_button_footer,
    configure_control_button,
    make_control_group,
)



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
        self.back_button = self._make_button("Back")
        add_back_button_footer(layout, self.back_button)
        self.back_button.hide()

    def set_workspace(self, spec: WorkspacePanelSpec) -> None:
        self._disconnect_actions()
        self._spec = spec
        self.buttons.clear()
        while self.stack.count():
            widget = self.stack.widget(0)
            self.stack.removeWidget(widget)
            widget.deleteLater()

        self.main_page = self._build_main_page(spec)
        self.mode_page = self._build_mode_page(spec.mode_id)
        self.system_page = self._build_system_page()
        self.file_page = self._build_file_page(spec.files)
        for page in (
            self.main_page,
            self.mode_page,
            self.system_page,
            self.file_page,
        ):
            self.stack.addWidget(page)
        if hasattr(self, "_navigator"):
            self._navigator.reset(self.main_page)
        else:
            self._navigator = ControlPanelNavigator(
                panel=self,
                title_label=self.title_label,
                stack=self.stack,
                main_page=self.main_page,
                back_button=self.back_button,
            )
            self._navigator.show_main()
        self._sync_sweep_group()

    def show_main_menu(self) -> None:
        if self._spec is None:
            return
        self._navigator.show_main()

    def navigate_back(self) -> None:
        self._navigator.navigate_back()

    def _show_page(
        self,
        title: str,
        page: QtWidgets.QWidget,
        *,
        remember: bool = True,
    ) -> None:
        self._navigator.show_page(title, page, remember=remember)

    def _build_main_page(self, spec: WorkspacePanelSpec) -> QtWidgets.QWidget:
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        analyzer = self._group("ANALYZER SETUP")
        analyzer_layout = analyzer.layout()
        mode_button = self._make_button(
            f"Analyzer Mode\n{spec.mode_label}"
        )
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
            continuous_running = (
                spec.continuous.action is not None
                and "Stop" in spec.continuous.action.text()
            )
            # Dedicated Continuous capture temporarily gives its internal
            # Single/analysis action a Stop label while analysis is active.
            # Continuous is the owning operation in that state; treating both
            # actions as running disables the only button that can stop it.
            single_running = (
                not continuous_running
                and spec.single.action is not None
                and "Stop" in spec.single.action.text()
            )
            busy = single_running or continuous_running
            refresh_enabled = (
                spec.refresh.action is None or spec.refresh.action.isEnabled()
            )
            reset_enabled = spec.reset.action is None or spec.reset.action.isEnabled()
            continuous_enabled = (
                spec.continuous.action is None
                or spec.continuous.action.isEnabled()
            )
            self.buttons["Continuous"].setEnabled(
                continuous_enabled and not single_running
            )
            self.buttons["Continuous"].setChecked(continuous_running)
            self.buttons["Single"].setChecked(single_running)
            if continuous_running:
                self.buttons["Single"].setText("Single")
            self.buttons["Refresh Analysis"].setEnabled(
                refresh_enabled and not busy
            )
            self.buttons["Reset"].setEnabled(reset_enabled and not busy)
            # Capture settings and the shared Pluto target are immutable while
            # an acquisition owns the device. Disable every entry point that
            # would otherwise be rejected by the application, including mode
            # buttons when the user is already viewing the Analyzer Mode page.
            locked_labels = (
                "Analyzer Mode",
                "Device",
                "Preset",
                "Recall",
                *(command.label for command in spec.setup),
            )
            for label in locked_labels:
                button = self.buttons.get(label)
                if button is not None:
                    button.setEnabled(not busy)
            for label, button in self.buttons.items():
                if label.startswith("mode:"):
                    button.setEnabled(not busy)

        for action in watched_actions:
            action.changed.connect(sync_sweep_group)
            self._bound_actions.append((action, sync_sweep_group))
        self._sync_sweep_group = sync_sweep_group
        sync_sweep_group()
        layout.addWidget(sweep)

        system = self._group("SYSTEM")
        self.system_group = system
        for label, callback in (
            ("State", lambda: self._show_page("State", self.system_page)),
            ("File", lambda: self._show_page("File", self.file_page)),
            ("Device", self.device_requested.emit),
        ):
            button = self._make_button(label)
            button.clicked.connect(lambda _checked=False, callback=callback: callback())
            system.layout().addWidget(button)
            self.buttons[label] = button
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
            ("General VSA", "generic"),
            ("Bluetooth", "bluetooth"),
            ("DECT", "dect"),
            ("Wi-Fi", "wifi"),
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
        preset.clicked.connect(self.preset_requested.emit)
        recall = self._make_button("Recall")
        recall.clicked.connect(self.recall_requested.emit)
        save = self._make_button("Save")
        save.clicked.connect(self.save_requested.emit)
        for key, button in (
            ("Recall", recall),
            ("Save", save),
            ("Preset", preset),
        ):
            page.layout().addWidget(button)
            self.buttons[key] = button
        page.layout().addStretch(1)
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
        running_button = display_label in {"Continuous", "Single"}
        if running_button:
            # Reuse the exact checked-state styling used by Analyzer Mode.
            # State is driven by the QAction below rather than by click
            # toggling, so a rejected start cannot leave stale blue feedback.
            button.setCheckable(True)
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
                running = "Stop" in action.text()
                button.setEnabled(action.isEnabled())
                button.setToolTip(action.toolTip())
                if running_button:
                    button.setChecked(running)
                if display_label == "Continuous":
                    button.setText(
                        "Stopping..."
                        if running and not action.isEnabled()
                        else ("Stop" if running else "Continuous")
                    )
                elif display_label == "Single":
                    button.setText(
                        "Stopping..."
                        if running and not action.isEnabled()
                        else ("Stop" if running else "Single")
                    )

            action.changed.connect(sync)
            self._bound_actions.append((action, sync))
            sync()
            if running_button:
                button.clicked.connect(
                    lambda _checked=False, update=sync: update()
                )
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
        return configure_control_button(button)

    @staticmethod
    def _simple_page() -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        return page

    def _group(self, title: str) -> QtWidgets.QGroupBox:
        return make_control_group(title)
