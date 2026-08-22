"""Shared plot interaction and dock-title styling for measurement workspaces."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets


DOCK_TITLE_SCALE = 1.3


class CenteredLabelAxisItem(pg.AxisItem):
    """Keep horizontal and rotated vertical labels visually centered."""

    def resizeEvent(self, event=None) -> None:
        super().resizeEvent(event)
        if not hasattr(self, "_linkedView") or self.label is None:
            return
        label_bounds = self.label.mapRectToParent(self.label.boundingRect())
        axis_center = QtCore.QPointF(
            self.size().width() / 2.0,
            self.size().height() / 2.0,
        )
        if self.orientation in {"left", "right"}:
            self.label.setY(self.label.y() + axis_center.y() - label_bounds.center().y())
        else:
            self.label.setX(self.label.x() + axis_center.x() - label_bounds.center().x())


class FixedInteractionViewBox(pg.ViewBox):
    """Left-drag rectangle zoom with middle-drag pan, without mode switching."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        super().setMouseMode(pg.ViewBox.RectMode)

    def setMouseMode(self, _mode: int) -> None:
        super().setMouseMode(pg.ViewBox.RectMode)

    def mouseDragEvent(self, event: object, axis: int | None = None) -> None:
        if event.button() != QtCore.Qt.MouseButton.MiddleButton:
            super().mouseDragEvent(event, axis=axis)
            return
        self.state["mouseMode"] = pg.ViewBox.PanMode
        try:
            super().mouseDragEvent(event, axis=axis)
        finally:
            self.state["mouseMode"] = pg.ViewBox.RectMode


def make_measurement_plot(left: str, bottom: str) -> pg.PlotWidget:
    """Create the common VSA plot surface and fixed three-button interaction."""

    plot = pg.PlotWidget(
        viewBox=FixedInteractionViewBox(),
        axisItems={
            "left": CenteredLabelAxisItem(orientation="left"),
            "bottom": CenteredLabelAxisItem(orientation="bottom"),
        },
    )
    plot.showGrid(x=True, y=True, alpha=0.25)
    plot.setLabel("left", left)
    plot.setLabel("bottom", bottom)
    plot.setDownsampling(auto=True, mode="peak")
    plot.setClipToView(True)
    return plot


def make_measurement_dock(
    title: str,
    widget: QtWidgets.QWidget,
    parent: QtWidgets.QMainWindow,
    *,
    object_prefix: str,
    closable: bool,
) -> QtWidgets.QDockWidget:
    """Create a dock whose title and content fonts match the generic VSA."""

    dock = QtWidgets.QDockWidget(title, parent)
    dock.setObjectName(f"{object_prefix}-{title.lower().replace(' ', '-')}")
    content_font = QtGui.QFont(widget.font())
    content_point_size = content_font.pointSizeF()
    content_font.setBold(False)
    if content_point_size > 0.0:
        content_font.setPointSizeF(content_point_size)
    title_font = QtGui.QFont(dock.font())
    title_font.setBold(True)
    if title_font.pointSizeF() > 0.0:
        title_font.setPointSizeF(title_font.pointSizeF() * DOCK_TITLE_SCALE)
    elif title_font.pixelSize() > 0:
        title_font.setPixelSize(max(1, round(title_font.pixelSize() * DOCK_TITLE_SCALE)))
    dock.setFont(title_font)
    dock.setWidget(widget)
    widget.setFont(content_font)
    features = (
        QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
        | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable
    )
    if closable:
        features |= QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
    dock.setFeatures(features)
    return dock


def trace_bounds(plot: pg.PlotWidget) -> tuple[float, float, float, float] | None:
    """Return finite visible trace bounds without non-data overlays."""

    x_min = y_min = np.inf
    x_max = y_max = -np.inf
    found = False
    for item in plot.listDataItems():
        if not item.isVisible():
            continue
        x_values, y_values = item.getOriginalDataset()
        if x_values is None or y_values is None:
            continue
        x_values = np.asarray(x_values)
        y_values = np.asarray(y_values)
        count = min(x_values.size, y_values.size)
        if count == 0:
            continue
        finite = np.isfinite(x_values[:count]) & np.isfinite(y_values[:count])
        if not np.any(finite):
            continue
        x_finite = x_values[:count][finite]
        y_finite = y_values[:count][finite]
        x_min = min(x_min, float(np.min(x_finite)))
        x_max = max(x_max, float(np.max(x_finite)))
        y_min = min(y_min, float(np.min(y_finite)))
        y_max = max(y_max, float(np.max(y_finite)))
        found = True
    if not found:
        return None
    return x_min, x_max, y_min, y_max


def padded_range(lower: float, upper: float) -> list[float]:
    span = upper - lower
    if span <= np.finfo(float).eps:
        span = max(abs(lower), abs(upper), 1.0) * 0.1
    margin = 0.05 * span
    return [lower - margin, upper + margin]


def view_all_traces(plot: pg.PlotWidget) -> None:
    bounds = trace_bounds(plot)
    if bounds is None:
        return
    x_min, x_max, y_min, y_max = bounds
    plot.setRange(
        xRange=padded_range(x_min, x_max),
        yRange=padded_range(y_min, y_max),
        padding=0.0,
    )


def install_measurement_plot_menu(
    plot: pg.PlotWidget,
    *,
    reset: Callable[[], None],
    view_all: Callable[[], None] | None = None,
) -> dict[str, QtGui.QAction]:
    """Install Reset/View All behavior and remove mutable mouse-mode controls."""

    menu = plot.getViewBox().getMenu(None)
    if menu is None:
        return {}
    reset_action = QtGui.QAction("Reset", menu)
    reset_action.triggered.connect(lambda _checked=False: reset())
    menu.insertAction(menu.viewAll, reset_action)
    menu.insertSeparator(menu.viewAll)
    menu.viewAll.triggered.disconnect(menu.autoRange)
    menu.viewAll.triggered.connect(
        lambda _checked=False: (view_all or (lambda: view_all_traces(plot)))()
    )
    for action in tuple(menu.actions()):
        if action.text() == "Mouse Mode":
            menu.removeAction(action)
    return {"reset": reset_action, "view_all": menu.viewAll}
