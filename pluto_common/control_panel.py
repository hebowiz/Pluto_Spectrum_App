"""Shared sizing and hierarchical navigation for instrument control panels."""

from __future__ import annotations

from collections.abc import Callable

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets


CONTROL_PANEL_WIDTH = 240
CONTROL_BUTTON_FONT_SCALE = 1.45
CONTROL_BUTTON_HEIGHT = 50
CONTROL_VALUE_BUTTON_HEIGHT = 72


def configure_control_button(
    button: QtWidgets.QPushButton,
    *,
    value: bool | None = None,
) -> QtWidgets.QPushButton:
    """Apply the shared VSA/RTSA button font and one/two-line height policy."""

    font = QtGui.QFont(button.font())
    if font.pointSizeF() > 0.0:
        font.setPointSizeF(font.pointSizeF() * CONTROL_BUTTON_FONT_SCALE)
    font.setBold(True)
    button.setFont(font)
    is_value = "\n" in button.text() if value is None else bool(value)
    button.setMinimumHeight(
        CONTROL_VALUE_BUTTON_HEIGHT if is_value else CONTROL_BUTTON_HEIGHT
    )
    return button


def add_back_button_footer(
    panel_layout: QtWidgets.QVBoxLayout,
    back_button: QtWidgets.QPushButton,
) -> QtWidgets.QHBoxLayout:
    """Add the RTSA/VSA right-aligned Back-button footer."""

    footer = QtWidgets.QHBoxLayout()
    footer.addStretch(1)
    footer.addWidget(back_button)
    panel_layout.addLayout(footer)
    return footer


def make_control_group(
    title: str,
    *,
    spacing: int = 8,
) -> QtWidgets.QGroupBox:
    """Create the shared framed and titled control-panel button group."""

    group = QtWidgets.QGroupBox(title)
    group.setStyleSheet(
        "QGroupBox { color: white; border: 1px solid #555; "
        "margin-top: 12px; }"
        "QGroupBox::title { subcontrol-origin: margin; left: 10px; "
        "padding: 0 4px; }"
    )
    font = QtGui.QFont(group.font())
    font.setBold(True)
    group.setFont(font)
    layout = QtWidgets.QVBoxLayout(group)
    layout.setSpacing(spacing)
    return group


class ControlPanelNavigator(QtCore.QObject):
    """Shared stacked-page history, Back button, and right-click navigation."""

    def __init__(
        self,
        *,
        panel: QtWidgets.QWidget,
        title_label: QtWidgets.QLabel,
        stack: QtWidgets.QStackedWidget,
        main_page: QtWidgets.QWidget,
        back_button: QtWidgets.QPushButton,
        apply_title: Callable[[str, QtWidgets.QWidget], None] | None = None,
        scroll_area: QtWidgets.QScrollArea | None = None,
    ) -> None:
        super().__init__(panel)
        self.panel = panel
        self.title_label = title_label
        self.stack = stack
        self.main_page = main_page
        self.back_button = back_button
        self.apply_title = apply_title
        self.scroll_area = scroll_area
        self.history: list[tuple[str, QtWidgets.QWidget]] = []
        self._scroll_positions: dict[QtWidgets.QWidget, int] = {}
        self.back_button.clicked.connect(self.navigate_back)
        self.refresh_event_filters()
        self.update_back_button()

    def reset(self, main_page: QtWidgets.QWidget) -> None:
        self.main_page = main_page
        self.history.clear()
        self._scroll_positions.clear()
        self.refresh_event_filters()
        self.show_page("Main Menu", main_page, remember=False)

    def _scroll_bar_for(
        self, page: QtWidgets.QWidget | None
    ) -> QtWidgets.QScrollBar | None:
        if page is None:
            return None
        if self.scroll_area is not None:
            return self.scroll_area.verticalScrollBar()
        if isinstance(page, QtWidgets.QScrollArea):
            return page.verticalScrollBar()
        return None

    def _save_scroll_position(self, page: QtWidgets.QWidget | None) -> None:
        bar = self._scroll_bar_for(page)
        if page is not None and bar is not None:
            self._scroll_positions[page] = bar.value()

    def _restore_scroll_position(self, page: QtWidgets.QWidget) -> None:
        bar = self._scroll_bar_for(page)
        if bar is None:
            return
        value = self._scroll_positions.get(page, 0)
        bar.setValue(value)

        def restore_after_layout() -> None:
            try:
                bar.setValue(value)
            except RuntimeError:
                pass

        # A shared QScrollArea updates its range only after QStackedWidget has
        # laid out the destination page. Reapply once that range is current.
        QtCore.QTimer.singleShot(0, restore_after_layout)

    def refresh_event_filters(self) -> None:
        for widget in (self.panel, *self.panel.findChildren(QtWidgets.QWidget)):
            widget.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.NoContextMenu)
            widget.installEventFilter(self)

    def show_page(
        self,
        title: str,
        page: QtWidgets.QWidget,
        *,
        remember: bool = True,
    ) -> None:
        current = self.stack.currentWidget()
        if current is page:
            if self.apply_title is None:
                self.title_label.setText(title)
            else:
                self.apply_title(title, page)
            self._restore_scroll_position(page)
            self.update_back_button()
            return
        self._save_scroll_position(current)
        if remember and current is not None and current is not page:
            self.history.append((self.title_label.text(), current))
        if self.apply_title is None:
            self.title_label.setText(title)
        else:
            self.apply_title(title, page)
        self.stack.setCurrentWidget(page)
        self._restore_scroll_position(page)
        self.update_back_button()

    def show_main(self) -> None:
        self.history.clear()
        self.show_page("Main Menu", self.main_page, remember=False)

    def navigate_back(self) -> None:
        if not self.history:
            return
        title, page = self.history.pop()
        self.show_page(title, page, remember=False)

    def update_back_button(self) -> None:
        show = self.stack.currentWidget() is not self.main_page and bool(
            self.history
        )
        self.back_button.setEnabled(show)
        self.back_button.setVisible(show)

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if (
            event.type() == QtCore.QEvent.Type.MouseButtonPress
            and isinstance(event, QtGui.QMouseEvent)
            and event.button() == QtCore.Qt.MouseButton.RightButton
        ):
            if self.stack.currentWidget() is not self.main_page:
                self.navigate_back()
            event.accept()
            return True
        return super().eventFilter(watched, event)
