"""Manual received-field overrides, shared by packet engines and the editor."""

import numpy as np


def field_widths(project):
    from pluto_vsg.model import BluetoothPacketKind, bluetooth_packet_properties
    if project.bluetooth_br is not None:
        kind = project.bluetooth_br.packet_kind
        header = 8 if kind == BluetoothPacketKind.DH1 else 16
        bps = bluetooth_packet_properties(kind)[2]
        widths = {"br_access": 72, "br_type": 4, "br_length": 5 if header == 8 else 10,
                  "br_crc": 16}
        if header == 16:
            widths["br_rfu"] = 3
        if bps > 1:
            widths.update(edr_sync=10*bps, edr_trailer=2*bps,
                          edr_padding=(-((project.bluetooth_br.payload_length_bytes+4)*8)) % bps)
        return widths
    if project.bluetooth_le is not None:
        return {"le_length": 8, "le_crc": 24}
    if project.bluetooth_hdt is not None:
        return {"hdt_pdu_control": 9, "hdt_rfu": 1}
    if project.dect is not None:
        result = {"dect_s": 32}
        if project.dect.prolonged_preamble:
            result["dect_prolonged"] = 16
        return result
    return {}


def validate_manual_fields(project):
    from pluto_vsg.model import ValidationIssue
    widths = field_widths(project)
    issues = []
    if not isinstance(project.manual_packet_fields, dict):
        return [ValidationIssue("manual_packet_fields", "Received packet fields must be a mapping.")]
    for name, bits in project.manual_packet_fields.items():
        if name not in widths or not isinstance(bits, str) or len(bits) != widths[name] or any(b not in "01" for b in bits):
            issues.append(ValidationIssue(
                f"manual_packet_fields.{name}",
                "Invalid received-field override; select Auto or enter the required binary bits.",
            ))
    return issues


def packet_field_bits(project, name, automatic):
    text = project.manual_packet_fields.get(name)
    return np.asarray(automatic, dtype=np.uint8) if text is None else np.asarray([int(b) for b in text], dtype=np.uint8)
