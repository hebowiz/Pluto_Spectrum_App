"""Read-only visual packet-composer canvas backed by ComposerGraph."""

from __future__ import annotations

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from pluto_vsg.composer import (
    ComposerBlock,
    ComposerBlockRole,
    ComposerGraph,
    ComposerTrackKind,
)


_TRACK_BACKGROUND = QtGui.QColor("#17191d")
_TRACK_BORDER = QtGui.QColor("#4a4d53")
_TEXT_COLOR = QtGui.QColor("#f1f1f1")
_MUTED_TEXT = QtGui.QColor("#aeb4bd")
_BLOCK_COLORS = {
    "Fixed": QtGui.QColor("#3264a8"),
    "Pattern": QtGui.QColor("#237a55"),
    "PRBS": QtGui.QColor("#704ca0"),
    "Computed": QtGui.QColor("#5b626d"),
}
_MODULATION_COLORS = {
    "GFSK": QtGui.QColor("#008c9e"),
    "pi/4-DQPSK": QtGui.QColor("#9a477b"),
    "8DPSK": QtGui.QColor("#8057b1"),
}
_POWER_COLOR = QtGui.QColor("#a86516")
_SELECTION_COLOR = QtGui.QColor("#55f2c2")


class _ComposerBlockItem(QtWidgets.QGraphicsRectItem):
    def __init__(self, block: ComposerBlock, rect: QtCore.QRectF, color: QtGui.QColor) -> None:
        # Keep the item's geometry in local coordinates.  Child labels are then
        # positioned relative to their own block instead of accumulating at the
        # scene origin when the timeline starts away from x=0.
        super().__init__(QtCore.QRectF(0.0, 0.0, rect.width(), rect.height()))
        self.setPos(rect.topLeft())
        self.block = block
        self.base_color = QtGui.QColor(color)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setPen(QtGui.QPen(self.base_color.lighter(150), 1.0))
        self.setBrush(QtGui.QBrush(self.base_color))
        details = "<br>".join(
            f"<b>{key}</b>: {value}" for key, value in block.properties
        )
        self.setToolTip(f"<b>{block.name}</b><br>{details}")

    def itemChange(self, change, value):  # noqa: N802 - Qt override
        if change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            selected = bool(value)
            self.setPen(
                QtGui.QPen(
                    _SELECTION_COLOR if selected else self.base_color.lighter(150),
                    2.5 if selected else 1.0,
                )
            )
            self.setBrush(
                QtGui.QBrush(
                    self.base_color.lighter(125) if selected else self.base_color
                )
            )
        return super().itemChange(change, value)


class PacketComposerView(QtWidgets.QGraphicsView):
    """Timeline view of packet fields, modulation regions and power controls."""

    selected_block_changed = QtCore.Signal(object)
    block_edit_requested = QtCore.Signal(object)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)
        self._graph: ComposerGraph | None = None
        self._block_items: list[_ComposerBlockItem] = []
        self._block_labels: list[QtWidgets.QGraphicsSimpleTextItem] = []
        self.setBackgroundBrush(QtGui.QColor("#0d0f12"))
        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(
            QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse
        )
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scene.selectionChanged.connect(self._selection_changed)

    @property
    def graph(self) -> ComposerGraph | None:
        return self._graph

    def set_graph(self, graph: ComposerGraph) -> None:
        self._graph = graph
        self._render_graph()

    def clear_graph(self) -> None:
        self._graph = None
        self._block_items.clear()
        self._block_labels.clear()
        self._scene.clear()
        self.selected_block_changed.emit(None)

    def select_block(self, block_id: str) -> bool:
        """Select a block by stable graph id, primarily for navigation/tests."""

        for item in self._scene.items():
            if isinstance(item, _ComposerBlockItem) and item.block.block_id == block_id:
                self._scene.clearSelection()
                item.setSelected(True)
                self.ensureVisible(item)
                return True
        return False

    def _selection_changed(self) -> None:
        selected = self._scene.selectedItems()
        block = selected[0].block if selected and isinstance(selected[0], _ComposerBlockItem) else None
        self.selected_block_changed.emit(block)

    def mouseDoubleClickEvent(  # noqa: N802 - Qt override
        self, event: QtGui.QMouseEvent
    ) -> None:
        item = self.itemAt(event.position().toPoint())
        while item is not None and not isinstance(item, _ComposerBlockItem):
            item = item.parentItem()
        if isinstance(item, _ComposerBlockItem):
            self._scene.clearSelection()
            item.setSelected(True)
            self.block_edit_requested.emit(item.block)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    @staticmethod
    def _block_color(block: ComposerBlock) -> QtGui.QColor:
        if block.role == ComposerBlockRole.MODULATION:
            return _MODULATION_COLORS.get(
                block.modulation_summary, QtGui.QColor("#356f8c")
            )
        if block.role == ComposerBlockRole.POWER:
            if block.relative_power_db is not None and block.relative_power_db < 0.0:
                darkness = 100 + min(80, round(abs(block.relative_power_db) * 2.0))
                return _POWER_COLOR.darker(darkness)
            return _POWER_COLOR
        return _BLOCK_COLORS.get(block.data_source, QtGui.QColor("#4d5968"))

    @staticmethod
    def _power_block_lanes(
        blocks: tuple[ComposerBlock, ...],
    ) -> dict[str, int]:
        """Place the active-window overview above non-overlapping controls."""

        lanes: dict[str, int] = {}
        detail_lane_stops: list[float] = []
        for block in sorted(
            blocks,
            key=lambda item: (item.start_symbol, item.stop_symbol, item.block_id),
        ):
            if block.block_id == "power:on-level":
                lanes[block.block_id] = 0
                continue
            for lane_index, stop_symbol in enumerate(detail_lane_stops):
                if block.start_symbol >= stop_symbol - 1e-9:
                    detail_lane_stops[lane_index] = block.stop_symbol
                    lanes[block.block_id] = lane_index + 1
                    break
            else:
                detail_lane_stops.append(block.stop_symbol)
                lanes[block.block_id] = len(detail_lane_stops)
        return lanes

    def _render_graph(self) -> None:
        self._block_items.clear()
        self._block_labels.clear()
        self._scene.clear()
        graph = self._graph
        if graph is None:
            return
        data_blocks = graph.track_blocks(ComposerTrackKind.DATA)
        power_blocks = graph.track_blocks(ComposerTrackKind.POWER)
        power_lanes = self._power_block_lanes(power_blocks)
        power_lane_count = max(power_lanes.values(), default=0) + 1
        max_depth = max((block.depth for block in data_blocks), default=0)
        minimum_symbol = min(
            (block.start_symbol for block in graph.blocks), default=0.0
        )
        maximum_symbol = max(
            (block.stop_symbol for block in graph.blocks), default=graph.total_symbols
        )
        span = max(1.0, maximum_symbol - minimum_symbol)
        pixels_per_symbol = max(0.35, min(6.0, 1150.0 / span))
        label_width = 138.0
        timeline_left = label_width + 12.0
        timeline_width = span * pixels_per_symbol

        major_y = 34.0
        major_height = 42.0
        child_y = major_y + major_height + 12.0
        child_height = 28.0
        data_height = major_height + 18.0 + max(1, max_depth) * (child_height + 5.0)
        modulation_y = major_y + data_height + 34.0
        power_y = modulation_y + 78.0
        power_lane_height = 34.0
        power_lane_gap = 7.0
        scene_height = (
            power_y
            + 24.0
            + power_lane_count * power_lane_height
            + max(0, power_lane_count - 1) * power_lane_gap
            + 16.0
        )

        tracks = (
            (ComposerTrackKind.DATA, 18.0, modulation_y - 12.0),
            (ComposerTrackKind.MODULATION, modulation_y - 12.0, power_y - 12.0),
            (ComposerTrackKind.POWER, power_y - 12.0, scene_height - 12.0),
        )
        for track, top, bottom in tracks:
            background = self._scene.addRect(
                timeline_left,
                top,
                timeline_width,
                bottom - top,
                QtGui.QPen(_TRACK_BORDER, 0.8),
                QtGui.QBrush(_TRACK_BACKGROUND),
            )
            background.setZValue(-10)
            label = self._scene.addSimpleText(track.value)
            label.setBrush(QtGui.QBrush(_TEXT_COLOR))
            font = QtGui.QFont(label.font())
            font.setBold(True)
            label.setFont(font)
            label.setPos(8.0, top + 8.0)

        zero_x = timeline_left + (0.0 - minimum_symbol) * pixels_per_symbol
        zero_line = self._scene.addLine(
            zero_x,
            18.0,
            zero_x,
            scene_height - 12.0,
            QtGui.QPen(QtGui.QColor("#70757d"), 1.0, QtCore.Qt.PenStyle.DashLine),
        )
        zero_line.setZValue(-2)

        for block in graph.blocks:
            x = timeline_left + (block.start_symbol - minimum_symbol) * pixels_per_symbol
            width = max(2.0, block.symbol_count * pixels_per_symbol)
            if block.track == ComposerTrackKind.DATA:
                if block.depth == 0:
                    y, height = major_y, major_height
                else:
                    y = child_y + (block.depth - 1) * (child_height + 5.0)
                    height = child_height
            elif block.track == ComposerTrackKind.MODULATION:
                y, height = modulation_y + 12.0, 42.0
            else:
                lane = power_lanes.get(block.block_id, 0)
                y = power_y + 12.0 + lane * (
                    power_lane_height + power_lane_gap
                )
                height = power_lane_height
            item = _ComposerBlockItem(
                block,
                QtCore.QRectF(x, y, width, height),
                self._block_color(block),
            )
            item.setZValue(2 + block.depth)
            self._scene.addItem(item)
            self._block_items.append(item)
            if width >= 34.0:
                text = self._scene.addSimpleText(block.name)
                text.setBrush(QtGui.QBrush(_TEXT_COLOR))
                text.setParentItem(item)
                text.setPos(5.0, max(1.0, (height - text.boundingRect().height()) / 2.0))
                text.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
                if text.boundingRect().width() > width - 8.0:
                    text.setScale(max(0.1, (width - 8.0) / text.boundingRect().width()))
                self._block_labels.append(text)

        title = self._scene.addSimpleText(
            f"{graph.project_name}  |  {graph.total_symbols:g} transmitted symbols"
        )
        title.setBrush(QtGui.QBrush(_MUTED_TEXT))
        title.setPos(timeline_left, 0.0)
        self._scene.setSceneRect(0.0, 0.0, timeline_left + timeline_width + 30.0, scene_height)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:  # noqa: N802 - Qt override
        if event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
            factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
            self.scale(factor, factor)
            event.accept()
            return
        super().wheelEvent(event)


__all__ = ["PacketComposerView"]
