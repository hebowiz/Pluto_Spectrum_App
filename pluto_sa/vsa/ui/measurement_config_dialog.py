"""Shared hierarchical measurement-configuration dialog chrome."""

from __future__ import annotations

from collections.abc import Sequence

from pyqtgraph.Qt import QtCore, QtWidgets


class HierarchicalMeasConfigDialog(QtWidgets.QDialog):
    """Generic VSA-style Config Top menu shared by analysis workspaces."""

    def __init__(
        self,
        parent: QtWidgets.QWidget,
        pages: Sequence[tuple[str, QtWidgets.QWidget]],
        *,
        window_title: str = "Meas Config",
        size: tuple[int, int] = (620, 520),
        standard_buttons: QtWidgets.QDialogButtonBox.StandardButton = (
            QtWidgets.QDialogButtonBox.StandardButton.Close
        ),
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(window_title)
        self.setModal(True)
        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self.resize(*size)
        dialog_layout = QtWidgets.QVBoxLayout(self)

        navigation_layout = QtWidgets.QHBoxLayout()
        self.back_button = QtWidgets.QPushButton("< Config Top")
        self.back_button.clicked.connect(lambda: self.show_page(0))
        self.page_title = QtWidgets.QLabel()
        title_font = self.page_title.font()
        title_font.setBold(True)
        title_font.setPointSize(title_font.pointSize() + 2)
        self.page_title.setFont(title_font)
        navigation_layout.addWidget(self.back_button)
        navigation_layout.addWidget(self.page_title)
        navigation_layout.addStretch(1)
        dialog_layout.addLayout(navigation_layout)

        self.stack = QtWidgets.QStackedWidget()
        config_top = QtWidgets.QWidget()
        config_top_layout = QtWidgets.QVBoxLayout(config_top)
        self.top_title = QtWidgets.QLabel("Config Top Menu")
        top_title_font = self.top_title.font()
        top_title_font.setBold(True)
        top_title_font.setPointSizeF(max(16.0, top_title_font.pointSizeF() + 6.0))
        self.top_title.setFont(top_title_font)
        config_top_layout.addWidget(self.top_title)
        button_grid = QtWidgets.QGridLayout()
        button_grid.setHorizontalSpacing(14)
        button_grid.setVerticalSpacing(14)
        self.top_buttons: dict[str, QtWidgets.QPushButton] = {}
        self.page_names = ("Config Top Menu",) + tuple(name for name, _page in pages)
        self.stack.addWidget(config_top)
        for index, (name, page) in enumerate(pages, start=1):
            self.stack.addWidget(page)
            button = QtWidgets.QPushButton(name)
            button_font = button.font()
            button_font.setPointSizeF(max(18.0, button_font.pointSizeF() * 2.0))
            button_font.setBold(True)
            button.setFont(button_font)
            button.setMinimumHeight(84)
            button.setProperty("configPageIndex", index)
            button.clicked.connect(
                lambda _checked=False, value=index: self.show_page(value)
            )
            button_grid.addWidget(button, (index - 1) // 2, (index - 1) % 2)
            self.top_buttons[name] = button
        config_top_layout.addLayout(button_grid)
        config_top_layout.addStretch(1)
        dialog_layout.addWidget(self.stack, 1)

        self.button_box = QtWidgets.QDialogButtonBox(standard_buttons)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        dialog_layout.addWidget(self.button_box)
        self.show_page(0)

    def show_page(self, index: int) -> None:
        self.stack.setCurrentIndex(int(index))
        is_top = int(index) == 0
        self.back_button.setVisible(not is_top)
        self.page_title.setText("" if is_top else self.page_names[int(index)])

    def open_top(self) -> int:
        self.show_page(0)
        return self.exec()
