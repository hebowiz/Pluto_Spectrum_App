"""Shared plot interaction and dock-title styling for measurement workspaces."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from scipy.ndimage import gaussian_filter


DOCK_TITLE_SCALE = 1.3
TRACE_COLOR = "y"
IQ_PLANE_LIMIT = 1.25
SYMBOL_PLOT_FLAT_SIZE = 6.0
CONSTELLATION_DENSITY_BINS = 96
CONSTELLATION_DENSITY_SIGMA_BINS = 0.7
CONSTELLATION_DENSITY_RED_LEVEL = 0.75
FREQUENCY_CONSTELLATION_DENSITY_HALF_WIDTH = 0.22
FREQUENCY_CONSTELLATION_X_LIMIT = 1.0
TRACE_SYMBOL_SIZE = 5.5


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


def add_result_range_overlay(
    plot: pg.PlotWidget,
    *,
    result_start_ms: float,
    result_stop_ms: float,
    pattern_start_ms: float | None = None,
    pattern_stop_ms: float | None = None,
    label: str = "Pattern Start",
) -> None:
    """Draw the shared Generic-VSA result/pattern range colors."""

    result_region = pg.LinearRegionItem(
        values=(float(result_start_ms), float(result_stop_ms)),
        movable=False,
        brush=pg.mkBrush(60, 130, 255, 35),
        pen=pg.mkPen(80, 150, 255, 150),
    )
    result_region.setZValue(-5)
    plot.addItem(result_region)
    if pattern_start_ms is None:
        return
    if pattern_stop_ms is not None:
        pattern_region = pg.LinearRegionItem(
            values=(float(pattern_start_ms), float(pattern_stop_ms)),
            movable=False,
            brush=pg.mkBrush(40, 220, 100, 65),
            pen=pg.mkPen(40, 240, 120, 190),
        )
        pattern_region.setZValue(-4)
        plot.addItem(pattern_region)
    marker = pg.InfiniteLine(
        pos=float(pattern_start_ms),
        angle=90,
        movable=False,
        pen=pg.mkPen(80, 255, 130, 220, width=2),
        label=label,
        labelOpts={"position": 0.08, "color": (120, 255, 160)},
    )
    plot.addItem(marker)


def _density_levels(density: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(density, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    peak = float(np.max(finite)) if finite.size else 0.0
    return (0.0, 1.0) if peak <= 0.0 else (
        0.0,
        peak * CONSTELLATION_DENSITY_RED_LEVEL,
    )


def _density_image(density: np.ndarray, rect: QtCore.QRectF) -> pg.ImageItem:
    item = pg.ImageItem(axisOrder="row-major")
    lookup_table = np.array(
        pg.colormap.get("turbo").getLookupTable(nPts=256, alpha=True),
        copy=True,
    )
    lookup_table[0, 3] = 0
    item.setLookupTable(lookup_table)
    item.setImage(density, levels=_density_levels(density))
    item.setRect(rect)
    return item


def plot_frequency_symbol_distribution(
    plot: pg.PlotWidget,
    frequency_khz: np.ndarray,
    *,
    y_limit_khz: float,
    density: bool,
) -> pg.ImageItem | None:
    """Render the common flat/density FSK constellation-frequency view."""

    values = np.asarray(frequency_khz, dtype=np.float64)
    values = values[np.isfinite(values)]
    horizontal = np.zeros(values.size, dtype=np.float64)
    if not density or values.size < 2:
        plot.plot(
            horizontal,
            values,
            pen=None,
            symbol="o",
            symbolSize=SYMBOL_PLOT_FLAT_SIZE,
            symbolBrush=pg.mkBrush(TRACE_COLOR),
            symbolPen=pg.mkPen(TRACE_COLOR),
        )
        return None
    limit = max(float(y_limit_khz), np.finfo(np.float64).eps)
    vertical_bins = CONSTELLATION_DENSITY_BINS
    horizontal_bins = 16
    histogram, _edges = np.histogram(
        values,
        bins=vertical_bins,
        range=(-limit, limit),
    )
    image = np.zeros((vertical_bins, horizontal_bins), dtype=np.float64)
    image[:, horizontal_bins // 2] = histogram
    image = gaussian_filter(
        image,
        sigma=(CONSTELLATION_DENSITY_SIGMA_BINS,) * 2,
        mode="constant",
        cval=0.0,
        truncate=3.0,
    )
    image = np.log1p(image)
    item = _density_image(
        image,
        QtCore.QRectF(
            -FREQUENCY_CONSTELLATION_DENSITY_HALF_WIDTH,
            -limit,
            2.0 * FREQUENCY_CONSTELLATION_DENSITY_HALF_WIDTH,
            2.0 * limit,
        ),
    )
    plot.addItem(item)
    # Preserve finite trace bounds for View All without rendering extra dots.
    plot.plot(horizontal, values, pen=None, symbol=None)
    return item


def plot_complex_symbol_distribution(
    plot: pg.PlotWidget,
    symbols: np.ndarray,
    *,
    density: bool,
    minimum_limit: float = IQ_PLANE_LIMIT,
) -> pg.ImageItem | None:
    """Render the common flat/density complex-symbol view."""

    values = np.asarray(symbols, dtype=np.complex128)
    finite = np.isfinite(values.real) & np.isfinite(values.imag)
    values = values[finite]
    if not density or values.size < 2:
        plot.plot(
            values.real,
            values.imag,
            pen=None,
            symbol="o",
            symbolSize=SYMBOL_PLOT_FLAT_SIZE,
            symbolBrush=pg.mkBrush(TRACE_COLOR),
            symbolPen=pg.mkPen(TRACE_COLOR),
        )
        return None
    component_peak = (
        float(max(np.max(np.abs(values.real)), np.max(np.abs(values.imag))))
        if values.size
        else 0.0
    )
    limit = max(float(minimum_limit), 1.02 * component_peak, np.finfo(float).eps)
    histogram, _i_edges, _q_edges = np.histogram2d(
        values.real,
        values.imag,
        bins=CONSTELLATION_DENSITY_BINS,
        range=((-limit, limit), (-limit, limit)),
    )
    image = gaussian_filter(
        histogram.T,
        sigma=CONSTELLATION_DENSITY_SIGMA_BINS,
        mode="constant",
        cval=0.0,
        truncate=3.0,
    )
    image = np.log1p(image)
    item = _density_image(
        image,
        QtCore.QRectF(-limit, -limit, 2.0 * limit, 2.0 * limit),
    )
    plot.addItem(item)
    plot.plot(values.real, values.imag, pen=None, symbol=None)
    return item


def plot_trace_symbol_points(
    plot: pg.PlotWidget, x_values: np.ndarray, y_values: np.ndarray
) -> None:
    """Draw synchronized decisions using the common VSA overlay style."""
    plot.plot(
        np.asarray(x_values),
        np.asarray(y_values),
        pen=None,
        symbol="o",
        symbolSize=TRACE_SYMBOL_SIZE,
        symbolBrush=pg.mkBrush(70, 255, 145, 230),
        symbolPen=pg.mkPen(10, 35, 20, 230, width=1),
    )


def set_frequency_constellation_x_lock(
    plot: pg.PlotWidget, locked: bool
) -> None:
    """Apply the common frequency-constellation horizontal-axis contract."""
    view_box = plot.getViewBox()
    if locked:
        x_limit = FREQUENCY_CONSTELLATION_X_LIMIT
        view_box.setMouseEnabled(x=False, y=True)
        view_box.setLimits(
            xMin=-x_limit,
            xMax=x_limit,
            minXRange=2.0 * x_limit,
            maxXRange=2.0 * x_limit,
        )
        plot.setXRange(-x_limit, x_limit, padding=0.0)
        return
    view_box.setLimits(xMin=None, xMax=None, minXRange=None, maxXRange=None)
    view_box.setMouseEnabled(x=True, y=True)
