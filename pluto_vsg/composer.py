"""Packet-composer graph derived from the device-independent waveform project."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pluto_vsg.model import FieldDefinition, WaveformProject


class ComposerTrackKind(StrEnum):
    DATA = "Packet / Data"
    MODULATION = "Modulation"
    POWER = "Power / Control"


class ComposerBlockRole(StrEnum):
    FIELD = "Field"
    MODULATION = "Modulation"
    POWER = "Power"


@dataclass(frozen=True)
class ComposerBlock:
    block_id: str
    track: ComposerTrackKind
    role: ComposerBlockRole
    name: str
    start_symbol: float
    symbol_count: float
    logical_bit_count: int | None = None
    depth: int = 0
    parent_id: str | None = None
    data_source: str = ""
    data_summary: str = ""
    modulation_summary: str = ""
    relative_power_db: float | None = None
    properties: tuple[tuple[str, str], ...] = ()

    @property
    def stop_symbol(self) -> float:
        return self.start_symbol + self.symbol_count


@dataclass(frozen=True)
class ComposerGraph:
    project_name: str
    standard: str
    total_symbols: float
    blocks: tuple[ComposerBlock, ...]

    def block(self, block_id: str) -> ComposerBlock | None:
        return next((item for item in self.blocks if item.block_id == block_id), None)

    def track_blocks(self, track: ComposerTrackKind) -> tuple[ComposerBlock, ...]:
        return tuple(item for item in self.blocks if item.track == track)


def _field_properties(packet_field: FieldDefinition) -> tuple[tuple[str, str], ...]:
    modulation = packet_field.modulation
    return (
        ("Field", packet_field.name),
        ("Logical Bits", "-" if packet_field.logical_bit_count is None else str(packet_field.logical_bit_count)),
        ("Tx Symbols", str(packet_field.symbol_count)),
        ("Data Source", packet_field.data_source.value),
        ("Data", packet_field.data or "-"),
        ("Relative Power", f"{packet_field.relative_power_db:+.3g} dB"),
        ("Modulation", modulation.kind.value),
        ("Symbol Rate", f"{modulation.symbol_rate_hz / 1e6:.6g} MSym/s"),
        ("TX Filter", modulation.filter_kind.value),
        ("Filter Parameter", f"{modulation.filter_parameter:.6g}"),
    )


def build_composer_graph(project: WaveformProject) -> ComposerGraph:
    """Compile a WaveformProject hierarchy into a visual, time-ordered graph."""

    blocks: list[ComposerBlock] = []
    leaf_fields: list[tuple[float, FieldDefinition]] = []

    def append_field(
        packet_field: FieldDefinition,
        *,
        start: float,
        path: str,
        depth: int,
        parent_id: str | None,
    ) -> None:
        block_id = f"field:{path}"
        blocks.append(
            ComposerBlock(
                block_id=block_id,
                track=ComposerTrackKind.DATA,
                role=ComposerBlockRole.FIELD,
                name=packet_field.name,
                start_symbol=start,
                symbol_count=float(packet_field.symbol_count),
                logical_bit_count=packet_field.logical_bit_count,
                depth=depth,
                parent_id=parent_id,
                data_source=packet_field.data_source.value,
                data_summary=packet_field.data,
                modulation_summary=packet_field.modulation.kind.value,
                properties=_field_properties(packet_field),
            )
        )
        if not packet_field.children:
            leaf_fields.append((start, packet_field))
            return
        child_start = start
        for child_index, child in enumerate(packet_field.children):
            append_field(
                child,
                start=child_start,
                path=f"{path}.{child_index}",
                depth=depth + 1,
                parent_id=block_id,
            )
            child_start += child.symbol_count

    cursor = 0.0
    for field_index, packet_field in enumerate(project.fields):
        append_field(
            packet_field,
            start=cursor,
            path=str(field_index),
            depth=0,
            parent_id=None,
        )
        cursor += packet_field.symbol_count

    modulation_runs: list[tuple[float, float, FieldDefinition]] = []
    for start, packet_field in sorted(leaf_fields, key=lambda item: item[0]):
        stop = start + packet_field.symbol_count
        if modulation_runs:
            previous_start, previous_stop, previous_field = modulation_runs[-1]
            previous = previous_field.modulation
            current = packet_field.modulation
            same_definition = (
                previous.kind == current.kind
                and previous.symbol_rate_hz == current.symbol_rate_hz
                and previous.filter_kind == current.filter_kind
                and previous.filter_parameter == current.filter_parameter
                and abs(previous_stop - start) < 1e-9
            )
            if same_definition:
                modulation_runs[-1] = (previous_start, stop, previous_field)
                continue
        modulation_runs.append((start, stop, packet_field))

    for index, (start, stop, packet_field) in enumerate(modulation_runs):
        modulation = packet_field.modulation
        filter_text = modulation.filter_kind.value
        if filter_text != "None":
            filter_text += f" ({modulation.filter_parameter:.3g})"
        blocks.append(
            ComposerBlock(
                block_id=f"modulation:{index}",
                track=ComposerTrackKind.MODULATION,
                role=ComposerBlockRole.MODULATION,
                name=modulation.kind.value,
                start_symbol=start,
                symbol_count=stop - start,
                modulation_summary=modulation.kind.value,
                properties=(
                    ("Modulation", modulation.kind.value),
                    ("Start", f"{start:g} symbols"),
                    ("Duration", f"{stop - start:g} symbols"),
                    ("Symbol Rate", f"{modulation.symbol_rate_hz / 1e6:.6g} MSym/s"),
                    ("TX Filter", filter_text),
                ),
            )
        )

    envelope = project.power_envelope
    if envelope.enabled:
        power_start = min(0.0, float(envelope.rise_delay_symbols))
        power_stop = max(
            cursor,
            cursor
            + float(envelope.fall_delay_symbols)
            + float(envelope.fall_symbols),
        )
        power_specs = (
            (
                "on-level",
                "Active Window",
                power_start,
                power_stop - power_start,
                (("Level", f"{envelope.on_level_db:.3g} dB"),),
            ),
            (
                "ramp-up",
                "Ramp Up",
                envelope.rise_delay_symbols,
                envelope.rise_symbols,
                (("Shape", envelope.shape), ("Target", f"{envelope.on_level_db:.3g} dB")),
            ),
            (
                "ramp-down",
                "Ramp Down",
                cursor + envelope.fall_delay_symbols,
                envelope.fall_symbols,
                (("Shape", envelope.shape), ("Target", f"{envelope.idle_level_db:.3g} dB")),
            ),
        )
        for block_id, name, start, duration, extra in power_specs:
            if duration <= 0.0:
                continue
            blocks.append(
                ComposerBlock(
                    block_id=f"power:{block_id}",
                    track=ComposerTrackKind.POWER,
                    role=ComposerBlockRole.POWER,
                    name=name,
                    start_symbol=float(start),
                    symbol_count=float(duration),
                    relative_power_db=(
                        envelope.on_level_db if block_id == "on-level" else None
                    ),
                    properties=(
                        ("Control", name),
                        ("Start", f"{start:g} symbols"),
                        ("Duration", f"{duration:g} symbols"),
                        *extra,
                    ),
                )
            )

    field_cursor = 0.0
    for field_index, packet_field in enumerate(project.fields):
        relative_power_db = float(packet_field.relative_power_db)
        is_edr_guard = (
            project.bluetooth_br is not None
            and packet_field.name == "Guard"
        )
        if is_edr_guard:
            settings = project.bluetooth_br
            ramp_in = min(
                float(packet_field.symbol_count),
                float(settings.edr_guard_ramp_in_symbols),
            )
            ramp_out = min(
                float(packet_field.symbol_count) - ramp_in,
                float(settings.edr_guard_ramp_out_symbols),
            )
            level_duration = max(
                0.0,
                float(packet_field.symbol_count) - ramp_in - ramp_out,
            )
            guard_specs = (
                ("ramp-in", "Guard Ramp In", field_cursor, ramp_in),
                (
                    "level",
                    "Guard Level",
                    field_cursor + ramp_in,
                    level_duration,
                ),
                (
                    "ramp-out",
                    "Guard Ramp Out",
                    field_cursor + float(packet_field.symbol_count) - ramp_out,
                    ramp_out,
                ),
            )
            for suffix, name, start, duration in guard_specs:
                if duration <= 0.0:
                    continue
                blocks.append(
                    ComposerBlock(
                        block_id=f"power:field:{field_index}:{suffix}",
                        track=ComposerTrackKind.POWER,
                        role=ComposerBlockRole.POWER,
                        name=name,
                        start_symbol=start,
                        symbol_count=duration,
                        relative_power_db=relative_power_db,
                        properties=(
                            ("Control", name),
                            ("Start", f"{start:g} symbols"),
                            ("Duration", f"{duration:g} symbols"),
                            ("Relative Power", f"{relative_power_db:+.3g} dB"),
                            ("Shape", settings.edr_guard_ramp_shape),
                        ),
                    )
                )
            field_cursor += packet_field.symbol_count
            continue
        if abs(relative_power_db) > 1e-12:
            blocks.append(
                ComposerBlock(
                    block_id=f"power:field:{field_index}",
                    track=ComposerTrackKind.POWER,
                    role=ComposerBlockRole.POWER,
                    name=f"{packet_field.name} Level",
                    start_symbol=field_cursor,
                    symbol_count=float(packet_field.symbol_count),
                    relative_power_db=relative_power_db,
                    properties=(
                        ("Control", f"{packet_field.name} relative level"),
                        ("Start", f"{field_cursor:g} symbols"),
                        ("Duration", f"{packet_field.symbol_count:g} symbols"),
                        ("Relative Power", f"{relative_power_db:+.3g} dB"),
                    ),
                )
            )
        field_cursor += packet_field.symbol_count

    return ComposerGraph(
        project_name=project.name,
        standard=project.standard.value,
        total_symbols=cursor,
        blocks=tuple(blocks),
    )


__all__ = [
    "ComposerBlock",
    "ComposerBlockRole",
    "ComposerGraph",
    "ComposerTrackKind",
    "build_composer_graph",
]
