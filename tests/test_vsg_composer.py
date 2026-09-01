import os
from dataclasses import replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

from pluto_vsg.composer import (
    ComposerBlockRole,
    ComposerTrackKind,
    build_composer_graph,
)
from pluto_vsg.model import BluetoothLEPhy, BluetoothPacketKind
from pluto_vsg.profiles import (
    bluetooth_br_edr_project,
    bluetooth_br_fields,
    bluetooth_le_project,
)
from pluto_vsg.ui.composer_view import PacketComposerView
from pluto_vsg.ui.main_window import PlutoVSGWindow


def test_le_composer_graph_has_three_time_aligned_tracks() -> None:
    project = bluetooth_le_project(BluetoothLEPhy.LE_1M)

    graph = build_composer_graph(project)

    data = graph.track_blocks(ComposerTrackKind.DATA)
    modulation = graph.track_blocks(ComposerTrackKind.MODULATION)
    power = graph.track_blocks(ComposerTrackKind.POWER)
    assert graph.total_symbols == sum(field.symbol_count for field in project.fields)
    assert [block.name for block in data if block.depth == 0] == [
        "Preamble",
        "Access Address / Sync Word",
        "PDU Header",
        "PDU Length",
        "PDU Payload",
        "CRC",
    ]
    assert len(modulation) == 1
    assert modulation[0].name == "GFSK"
    assert modulation[0].start_symbol == 0
    assert modulation[0].stop_symbol == graph.total_symbols
    assert [block.name for block in power] == [
        "Active Window",
        "Ramp Up",
        "Ramp Down",
    ]
    assert power[0].start_symbol == -1
    assert power[0].stop_symbol == graph.total_symbols + 2
    assert power[1].start_symbol == -1
    assert power[-1].start_symbol == graph.total_symbols + 1


def test_br_composer_graph_preserves_major_and_minor_field_hierarchy() -> None:
    graph = build_composer_graph(bluetooth_br_edr_project())

    data = graph.track_blocks(ComposerTrackKind.DATA)
    access_code = graph.block("field:0")
    preamble = graph.block("field:0.0")
    header = graph.block("field:1")
    lt_addr = graph.block("field:1.0")
    assert access_code is not None and access_code.name == "Access Code"
    assert preamble is not None and preamble.parent_id == access_code.block_id
    assert preamble.depth == 1
    assert header is not None and lt_addr is not None
    assert lt_addr.start_symbol == header.start_symbol
    assert any(block.logical_bit_count != block.symbol_count for block in data if block.depth)


def test_edr_composer_graph_exposes_mixed_modulation_regions() -> None:
    base = bluetooth_br_edr_project()
    assert base.bluetooth_br is not None
    settings = replace(
        base.bluetooth_br,
        packet_kind=BluetoothPacketKind.DH1_2,
        payload_length_bytes=54,
    )
    project = replace(
        base,
        bluetooth_br=settings,
        fields=bluetooth_br_fields(settings),
    )

    graph = build_composer_graph(project)
    modulation = graph.track_blocks(ComposerTrackKind.MODULATION)

    assert [block.name for block in modulation] == ["GFSK", "pi/4-DQPSK"]
    assert modulation[0].stop_symbol == modulation[1].start_symbol
    assert all(block.role == ComposerBlockRole.MODULATION for block in modulation)


def test_edr_composer_guard_exposes_relative_power() -> None:
    base = bluetooth_br_edr_project()
    assert base.bluetooth_br is not None
    settings = replace(
        base.bluetooth_br,
        packet_kind=BluetoothPacketKind.DH1_2,
        payload_length_bytes=54,
        edr_guard_relative_power_db=-15.0,
    )
    project = replace(
        base,
        bluetooth_br=settings,
        fields=bluetooth_br_fields(settings),
    )

    graph = build_composer_graph(project)
    guard = graph.block("field:2")
    ramp_in = graph.block("power:field:2:ramp-in")
    guard_level = graph.block("power:field:2:level")
    ramp_out = graph.block("power:field:2:ramp-out")

    assert guard is not None
    assert ("Relative Power", "-15 dB") in guard.properties
    assert ramp_in is not None
    assert ramp_in.start_symbol == guard.start_symbol
    assert ramp_in.symbol_count == 1.0
    assert ("Shape", "Cosine") in ramp_in.properties
    assert guard_level is not None
    assert guard_level.start_symbol == guard.start_symbol + 1.0
    assert guard_level.symbol_count == guard.symbol_count - 2.0
    assert guard_level.relative_power_db == -15.0
    assert ramp_out is not None
    assert ramp_out.start_symbol == guard.stop_symbol - 1.0
    assert ramp_out.symbol_count == 1.0


def test_visual_composer_selection_emits_graph_block() -> None:
    pg.mkQApp("Pluto VSG visual composer test")
    view = PacketComposerView()
    graph = build_composer_graph(bluetooth_le_project())
    selected = []
    view.selected_block_changed.connect(selected.append)
    try:
        view.set_graph(graph)
        assert view.select_block("field:0") is True
        assert selected[-1].name == "Preamble"
        assert view.select_block("missing") is False
    finally:
        view.close()


def test_visual_composer_labels_stay_inside_their_timeline_blocks() -> None:
    pg.mkQApp("Pluto VSG visual composer label geometry test")
    view = PacketComposerView()
    try:
        view.set_graph(build_composer_graph(bluetooth_br_edr_project()))
        block_labels = view._block_labels
        assert block_labels
        for label in block_labels:
            parent_rect = label.parentItem().sceneBoundingRect()
            label_rect = label.sceneBoundingRect()
            assert parent_rect.left() <= label_rect.left() <= parent_rect.right()
            assert parent_rect.top() <= label_rect.center().y() <= parent_rect.bottom()
    finally:
        view.close()


def test_visual_composer_power_controls_do_not_overlap_active_window() -> None:
    pg.mkQApp("Pluto VSG visual composer power lane test")
    view = PacketComposerView()
    try:
        view.set_graph(build_composer_graph(bluetooth_le_project()))
        items = {item.block.block_id: item for item in view._block_items}
        active_window = items["power:on-level"].sceneBoundingRect()
        ramp_up = items["power:ramp-up"].sceneBoundingRect()
        ramp_down = items["power:ramp-down"].sceneBoundingRect()

        assert active_window.intersects(ramp_up) is False
        assert active_window.intersects(ramp_down) is False
        assert ramp_up.top() == ramp_down.top()
    finally:
        view.close()


def test_field_tree_and_visual_composer_selection_are_synchronized() -> None:
    app = pg.mkQApp("Pluto VSG composer selection synchronization test")
    window = PlutoVSGWindow()
    try:
        tree_item = window._field_items_by_block_id["field:1.0"]
        window.field_table.setCurrentItem(tree_item)
        app.processEvents()
        assert window._selected_composer_block is not None
        assert window._selected_composer_block.block_id == "field:1.0"

        assert window.composer_view.select_block("field:0") is True
        app.processEvents()
        assert window.field_table.currentItem().data(
            0, QtCore.Qt.ItemDataRole.UserRole
        ) == "field:0"
    finally:
        window.close()


def test_project_settings_changes_support_undo_and_redo() -> None:
    pg.mkQApp("Pluto VSG composer undo test")
    window = PlutoVSGWindow()
    try:
        original = window.project
        updated = replace(original, repeat_count=2)
        window._commit_project_change(updated, "Change repeat count")
        assert window.project.repeat_count == 2
        assert window.undo_stack.canUndo() is True

        window.undo_stack.undo()
        assert window.project == original
        window.undo_stack.redo()
        assert window.project == updated
    finally:
        window.close()
