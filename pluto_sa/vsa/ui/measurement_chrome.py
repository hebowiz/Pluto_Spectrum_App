"""Shared plot interaction and dock-title styling for measurement workspaces."""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum

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
DEDICATED_TABLE_GRID_COLOR = "#606060"
IQ_POWER_DISPLAY_FLOOR_DBM = -120.0
DEDICATED_STATUS_COLORS = {
    "PASS": "#43f5a5",
    "FAIL": "#ff5b5b",
    "MEASURING": "#ffd166",
    "VALID": "#43f5a5",
    "INVALID": "#ff5b5b",
    "WARNING": "#ffd166",
    "N/A": "#a0a0a0",
    "—": "#a0a0a0",
}


def make_analysis_bandwidth_display_controls(
) -> tuple[QtWidgets.QCheckBox, QtWidgets.QCheckBox]:
    """Create the common Power/Spectrum Analysis Bandwidth selectors."""

    power = QtWidgets.QCheckBox()
    power.setChecked(True)
    power.setToolTip(
        "Use the DDC and Analysis Channel LPF output for Power vs Time."
    )
    spectrum = QtWidgets.QCheckBox()
    spectrum.setChecked(False)
    spectrum.setToolTip(
        "Use the DDC and Analysis Channel LPF output for Spectrum."
    )
    return power, spectrum


class CenteredDedicatedTableDelegate(QtWidgets.QStyledItemDelegate):
    """Center cell text consistently across dedicated-mode tables."""

    @staticmethod
    def _text_layout(
        text: str,
        font: QtGui.QFont,
        width: float,
    ) -> tuple[QtGui.QTextLayout, float]:
        layout = QtGui.QTextLayout(text, font)
        text_option = QtGui.QTextOption()
        text_option.setWrapMode(
            QtGui.QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere
        )
        text_option.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.setTextOption(text_option)
        height = 0.0
        layout.beginLayout()
        while True:
            line = layout.createLine()
            if not line.isValid():
                break
            line.setLineWidth(max(1.0, float(width)))
            line.setPosition(QtCore.QPointF(0.0, height))
            height += line.height()
        layout.endLayout()
        return layout, height

    def initStyleOption(
        self,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex,
    ) -> None:
        super().initStyleOption(option, index)
        option.displayAlignment = QtCore.Qt.AlignmentFlag.AlignCenter
        option.features |= QtWidgets.QStyleOptionViewItem.ViewItemFeature.WrapText

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex,
    ) -> None:
        """Paint unbroken values (for example long Hex) as real wrapped text."""

        styled = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(styled, index)
        text = styled.text
        style = styled.widget.style() if styled.widget is not None else QtWidgets.QApplication.style()
        text_rect = style.subElementRect(
            QtWidgets.QStyle.SubElement.SE_ItemViewItemText,
            styled,
            styled.widget,
        ).adjusted(2, 1, -2, -1)
        styled.text = ""
        style.drawControl(
            QtWidgets.QStyle.ControlElement.CE_ItemViewItem,
            styled,
            painter,
            styled.widget,
        )
        if not text or text_rect.width() <= 0 or text_rect.height() <= 0:
            return
        layout, height = self._text_layout(text, styled.font, text_rect.width())
        selected = bool(
            styled.state & QtWidgets.QStyle.StateFlag.State_Selected
        )
        enabled = bool(styled.state & QtWidgets.QStyle.StateFlag.State_Enabled)
        group = (
            QtGui.QPalette.ColorGroup.Normal
            if enabled
            else QtGui.QPalette.ColorGroup.Disabled
        )
        role = (
            QtGui.QPalette.ColorRole.HighlightedText
            if selected
            else QtGui.QPalette.ColorRole.Text
        )
        painter.save()
        painter.setPen(styled.palette.color(group, role))
        top = text_rect.top() + max(0.0, (text_rect.height() - height) / 2.0)
        painter.setClipRect(text_rect)
        layout.draw(painter, QtCore.QPointF(text_rect.left(), top))
        painter.restore()

    def sizeHint(
        self,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex,
    ) -> QtCore.QSize:
        """Return a row height that preserves wrapped dedicated-table text."""

        base = super().sizeHint(option, index)
        view = self.parent()
        if not isinstance(view, QtWidgets.QAbstractItemView):
            return base
        width = max(8, int(view.columnWidth(index.column())) - 8)
        if isinstance(view, QtWidgets.QTreeView) and index.column() == 0:
            depth = 1
            parent = index.parent()
            while parent.isValid():
                depth += 1
                parent = parent.parent()
            width = max(8, width - depth * view.indentation())
        text = str(index.data(QtCore.Qt.ItemDataRole.DisplayRole) or "")
        _layout, height = self._text_layout(text, option.font, width)
        return QtCore.QSize(base.width(), max(base.height(), int(np.ceil(height)) + 6))


def apply_dedicated_table_style(
    view: QtWidgets.QAbstractItemView,
) -> None:
    """Apply the shared dedicated-mode table alignment and grid styling."""

    view.setItemDelegate(CenteredDedicatedTableDelegate(view))
    view.setStyleSheet(
        "QTableView::item, QTreeView::item {"
        f" border-right: 1px solid {DEDICATED_TABLE_GRID_COLOR};"
        f" border-bottom: 1px solid {DEDICATED_TABLE_GRID_COLOR};"
        " padding: 2px;"
        "}"
        "QHeaderView::section {"
        f" border-right: 1px solid {DEDICATED_TABLE_GRID_COLOR};"
        f" border-bottom: 1px solid {DEDICATED_TABLE_GRID_COLOR};"
        " padding: 2px;"
        "}"
    )
    if isinstance(view, QtWidgets.QTableView):
        view.setShowGrid(False)
        header = view.horizontalHeader()
    elif isinstance(view, QtWidgets.QTreeView):
        header = view.header()
    else:
        return
    header.setDefaultAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)


def dedicated_status_color(status: object) -> QtGui.QColor | None:
    """Return the status color shared by all dedicated analyzer tables."""

    text = str(getattr(status, "value", status)).strip().upper()
    key = "MEASURING" if text.startswith("MEASURING") else text
    value = DEDICATED_STATUS_COLORS.get(key)
    return None if value is None else QtGui.QColor(value)


class DedicatedSummaryTable(QtWidgets.QTableWidget):
    """Four-column dedicated summary fitted to its dock without ellipsis."""

    def __init__(self, parent=None) -> None:
        super().__init__(0, 4, parent)
        self.setHorizontalHeaderLabels(("Test Item", "Value", "Limit", "Result"))
        self.verticalHeader().setVisible(False)
        self.verticalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        self.setWordWrap(True)
        self.setTextElideMode(QtCore.Qt.TextElideMode.ElideNone)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        apply_dedicated_table_style(self)
        header = self.horizontalHeader()
        header.setTextElideMode(QtCore.Qt.TextElideMode.ElideNone)
        header.setMinimumSectionSize(1)
        for column in range(self.columnCount()):
            header.setSectionResizeMode(column, QtWidgets.QHeaderView.ResizeMode.Fixed)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._fit_columns()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        QtCore.QTimer.singleShot(0, self._fit_columns)

    def _fit_columns(self) -> None:
        width = max(1, self.viewport().width())
        if width < 160:
            widths = [width // 4] * 4
        else:
            result_width = min(90, max(64, round(width * 0.17)))
            remaining = width - result_width
            item_width = round(remaining * 0.35)
            value_width = round(remaining * 0.27)
            widths = [
                item_width,
                value_width,
                remaining - item_width - value_width,
                result_width,
            ]
        widths[-1] += width - sum(widths)
        for column, column_width in enumerate(widths):
            self.setColumnWidth(column, max(1, column_width))
        self.resizeRowsToContents()


class DedicatedPacketAnalysisTree(QtWidgets.QTreeWidget):
    """Dedicated packet table that preserves complete cell text."""

    def __init__(
        self,
        headers: tuple[str, ...],
        minimum_widths: tuple[int, ...],
        *,
        expand_columns: tuple[int, ...] = (0, 1),
        parent=None,
    ) -> None:
        super().__init__(parent)
        if len(headers) != len(minimum_widths):
            raise ValueError("headers and minimum_widths must have equal length")
        self._minimum_widths = tuple(int(value) for value in minimum_widths)
        self._expand_columns = tuple(int(value) for value in expand_columns)
        self.setColumnCount(len(headers))
        self.setHeaderLabels(headers)
        self.setWordWrap(True)
        self.setUniformRowHeights(False)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setTextElideMode(QtCore.Qt.TextElideMode.ElideNone)
        apply_dedicated_table_style(self)
        header = self.header()
        header.setTextElideMode(QtCore.Qt.TextElideMode.ElideNone)
        for column in range(len(headers)):
            header.setSectionResizeMode(column, QtWidgets.QHeaderView.ResizeMode.Fixed)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._fit_columns()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        QtCore.QTimer.singleShot(0, self._fit_columns)

    def _fit_columns(self) -> None:
        width = max(1, self.viewport().width())
        widths = list(self._minimum_widths)
        extra = max(0, width - sum(widths))
        if extra and self._expand_columns:
            portion, remainder = divmod(extra, len(self._expand_columns))
            for offset, column in enumerate(self._expand_columns):
                widths[column] += portion + (1 if offset < remainder else 0)
        elif width < sum(widths) and self._expand_columns:
            deficit = sum(widths) - width
            for column in self._expand_columns:
                reduction = min(deficit, max(0, widths[column] - 80))
                widths[column] -= reduction
                deficit -= reduction
        for column, column_width in enumerate(widths):
            self.setColumnWidth(column, max(1, column_width))
        self.doItemsLayout()


class SymbolDensitySpread(StrEnum):
    """Shared density-kernel width for every VSA symbol-plot mode."""

    NONE = "None"
    MEDIUM = "Medium"
    MAXIMUM = "Maximum"


def add_symbol_density_menu(
    menu: QtWidgets.QMenu,
    owner: QtCore.QObject,
    *,
    enabled: bool,
    spread: SymbolDensitySpread,
    on_enabled: Callable[[bool], None],
    on_spread: Callable[[SymbolDensitySpread], None],
) -> tuple[QtGui.QAction, QtGui.QActionGroup, dict[SymbolDensitySpread, QtGui.QAction]]:
    """Install the density controls shared by every VSA Symbol Plot."""

    density_action = menu.addAction("Symbol Plot Density")
    density_action.setCheckable(True)
    density_action.setChecked(bool(enabled))
    density_action.toggled.connect(on_enabled)
    spread_menu = menu.addMenu("Density Spread")
    spread_group = QtGui.QActionGroup(owner)
    spread_group.setExclusive(True)
    spread_actions: dict[SymbolDensitySpread, QtGui.QAction] = {}
    for candidate in SymbolDensitySpread:
        action = spread_menu.addAction(candidate.value)
        action.setCheckable(True)
        action.setData(candidate.value)
        spread_group.addAction(action)
        spread_actions[candidate] = action
        action.triggered.connect(
            lambda _checked=False, selected=candidate: on_spread(selected)
        )
    spread_actions[spread].setChecked(True)
    return density_action, spread_group, spread_actions


def add_fsk_symbol_plot_menu(
    menu: QtWidgets.QMenu,
    owner: QtCore.QObject,
    *,
    mode: str,
    on_mode: Callable[[str], None],
) -> tuple[QtGui.QAction, QtGui.QAction, QtGui.QActionGroup]:
    """Install the common frequency/phase FSK Symbol Plot selector."""

    fsk_menu = menu.addMenu("FSK Symbol Plot")
    group = QtGui.QActionGroup(owner)
    group.setExclusive(True)
    frequency_action = fsk_menu.addAction("Constellation Frequency")
    phase_action = fsk_menu.addAction("Phase Difference")
    for action in (frequency_action, phase_action):
        action.setCheckable(True)
        group.addAction(action)
    frequency_action.setChecked(mode == "Constellation Frequency")
    phase_action.setChecked(mode == "Phase Difference")
    frequency_action.triggered.connect(
        lambda _checked=False: on_mode("Constellation Frequency")
    )
    phase_action.triggered.connect(
        lambda _checked=False: on_mode("Phase Difference")
    )
    return frequency_action, phase_action, group


_SYMBOL_DENSITY_SIGMA_BINS = {
    SymbolDensitySpread.NONE: 0.0,
    SymbolDensitySpread.MEDIUM: 0.45,
    SymbolDensitySpread.MAXIMUM: CONSTELLATION_DENSITY_SIGMA_BINS,
}


def symbol_density_sigma_bins(
    spread: SymbolDensitySpread | str,
) -> float:
    """Resolve a persisted density-spread choice to Gaussian sigma in bins."""

    try:
        resolved = SymbolDensitySpread(spread)
    except ValueError as error:
        raise ValueError(f"Unsupported symbol density spread: {spread!r}") from error
    return _SYMBOL_DENSITY_SIGMA_BINS[resolved]


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


def configure_iq_power_plot(plot: pg.PlotWidget) -> None:
    """Apply the shared finite IQ-power display floor."""

    plot.getViewBox().setLimits(yMin=IQ_POWER_DISPLAY_FLOOR_DBM)


def packet_time_view_range_ms(
    *,
    packet_start_ms: float,
    packet_stop_ms: float,
    capture_stop_ms: float,
    minimum_margin_ms: float = 0.0,
) -> tuple[float, float]:
    """Return the common packet-plus-context range for time-domain plots."""

    duration_ms = max(0.0, float(packet_stop_ms) - float(packet_start_ms))
    margin_ms = max(0.10 * duration_ms, float(minimum_margin_ms), 1e-9)
    lower = max(0.0, float(packet_start_ms) - margin_ms)
    upper = min(float(capture_stop_ms), float(packet_stop_ms) + margin_ms)
    if upper <= lower:
        upper = min(float(capture_stop_ms), lower + max(margin_ms, 1e-9))
    return lower, upper


def limit_iq_power_display_dbm(values: np.ndarray) -> np.ndarray:
    """Clamp display-only IQ power without changing measurement data."""

    display = np.asarray(values, dtype=np.float64).copy()
    display[~np.isfinite(display)] = IQ_POWER_DISPLAY_FLOOR_DBM
    np.maximum(display, IQ_POWER_DISPLAY_FLOOR_DBM, out=display)
    return display


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
    density_spread: SymbolDensitySpread | str = SymbolDensitySpread.MAXIMUM,
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
    sigma = symbol_density_sigma_bins(density_spread)
    if sigma > 0.0:
        image = gaussian_filter(
            image,
            sigma=(sigma,) * 2,
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
    density_spread: SymbolDensitySpread | str = SymbolDensitySpread.MAXIMUM,
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
    image = histogram.T
    sigma = symbol_density_sigma_bins(density_spread)
    if sigma > 0.0:
        image = gaussian_filter(
            image,
            sigma=sigma,
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


def plot_unit_circle(plot: pg.PlotWidget) -> None:
    """Draw the common IQ-plane unit reference circle."""

    angle = np.linspace(0.0, 2.0 * np.pi, 361)
    plot.plot(
        np.cos(angle),
        np.sin(angle),
        pen=pg.mkPen((120, 120, 120, 110), width=1),
    )


def set_iq_plane_range(plot: pg.PlotWidget) -> None:
    """Apply the common fixed initial IQ-plane range."""

    plot.setXRange(-IQ_PLANE_LIMIT, IQ_PLANE_LIMIT, padding=0.0)
    plot.setYRange(-IQ_PLANE_LIMIT, IQ_PLANE_LIMIT, padding=0.0)


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
