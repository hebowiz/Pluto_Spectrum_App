import os
from dataclasses import replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pyqtgraph as pg

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
    assert [block.name for block in power] == ["Ramp Up", "Packet ON", "Ramp Down"]
    assert power[0].start_symbol == -1
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
