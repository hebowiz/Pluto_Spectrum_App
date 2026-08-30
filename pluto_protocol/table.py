"""Common UI-neutral rows derived from hierarchical packet fields."""

from __future__ import annotations

from dataclasses import dataclass

from pluto_protocol.bitops import bits_hex_lsb
from pluto_protocol.model import FieldStatus, PacketAnalysisResult, PacketField


@dataclass(frozen=True)
class PacketTableRow:
    path: str
    depth: int
    name: str
    value: object
    display_value: str
    meaning: str
    status: FieldStatus
    start_bit: int
    stop_bit: int
    raw_hex: str


def _display(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return str(value)


def packet_table_rows(result: PacketAnalysisResult) -> tuple[PacketTableRow, ...]:
    """Flatten the authoritative field tree without discarding hierarchy."""

    rows: list[PacketTableRow] = []

    def append(field: PacketField, parent: str, depth: int) -> None:
        path = f"{parent}.{field.field_id}" if parent else field.field_id
        rows.append(
            PacketTableRow(
                path=path,
                depth=depth,
                name=field.name,
                value=field.value,
                display_value=_display(field.value),
                meaning=field.meaning,
                status=field.status,
                start_bit=field.start_bit,
                stop_bit=field.stop_bit,
                raw_hex=bits_hex_lsb(field.raw_bits),
            )
        )
        for child in field.children:
            append(child, path, depth + 1)

    for root in result.root_fields:
        append(root, "", 0)
    return tuple(rows)
