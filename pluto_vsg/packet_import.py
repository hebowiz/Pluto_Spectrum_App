"""Restore supported, structurally complete decoded packets as VSG projects.

Integrity failures are allowed. No RF measurement is mistaken for an original
transmitter setting: RF/timing parameters start from each template's defaults.
"""
from dataclasses import replace

from pluto_vsg.model import (
    BluetoothPacketKind, BluetoothLEPhy, BluetoothLEPayloadSourceKind,
    PayloadSourceKind, HDTPayloadSourceKind, DectDirection, DectPacketType,
    DectBFieldSource, validate_project, bluetooth_packet_properties,
)
from pluto_vsg.profiles import (
    bluetooth_br_edr_project, bluetooth_br_fields, bluetooth_le_project,
    bluetooth_le_fields, bluetooth_hdt_project, bluetooth_hdt_fields,
    dect_project, dect_fields,
)
from pluto_protocol.bluetooth.hdt import HDTRate, HDT_RF_TEST_PCA, HDT_RF_TEST_CRC32_INIT
from pluto_vsa.profiles.bluetooth_br import recover_lap_from_access_code_bits


def text(bits):
    return "".join(str(int(b)) for b in bits)


def lsb(bits):
    return sum(int(b) << i for i, b in enumerate(bits))


def msb(bits):
    return sum(int(b) << (len(bits)-1-i) for i, b in enumerate(bits))


def project_from_packet(packet):
    if packet is None or not packet.integrity.complete:
        raise ValueError("A structurally complete decoded packet is required")
    context = packet.decode_context
    fields = {}
    def collect(items):
        for item in items:
            fields.setdefault(item.field_id, item)
            collect(item.children)
    collect(packet.root_fields)
    def bits(name, width=None):
        field = fields.get(name)
        if field is None or (width is not None and field.raw_bits.size != width):
            raise ValueError(f"Missing or incomplete packet field: {name}")
        return field.raw_bits
    def value(name):
        if name not in fields or fields[name].value is None:
            raise ValueError(f"Missing decoded value: {name}")
        return fields[name].value
    manual = {}
    def preserve(key, values):
        manual[key] = text(values)

    if packet.protocol_id == "bluetooth.br_edr":
        try:
            kind = BluetoothPacketKind(packet.packet_type)
        except ValueError as error:
            raise ValueError("This Classic packet type is not supported by VSG") from error
        if int(value("type")) != bluetooth_packet_properties(kind)[1]:
            raise ValueError("Decoded Classic TYPE does not match the supported packet layout")
        base = bluetooth_br_edr_project()
        body = bits("payload_body")
        header_width = 8 if kind == BluetoothPacketKind.DH1 else 16
        header = bits("payload_header", header_width)
        length = int(value("length"))
        if body.size != length*8 or bits("payload").size != header_width+length*8+16:
            raise ValueError("ACL length and decoded data disagree")
        if context.get("uap") is None:
            raise ValueError("UAP was not recovered")
        whitening = bool(context.get("whitening_enabled", True))
        if whitening and context.get("clock_6_1") is None:
            raise ValueError("Whitening clock was not recovered")
        recovered = recover_lap_from_access_code_bits(bits("access_code", 72))
        if recovered is None:
            raise ValueError("Access Code is not structurally recognizable")
        settings = replace(
            base.bluetooth_br, packet_kind=kind, lap=recovered[0],
            uap=int(context["uap"]), clock_6_1=int(context.get("clock_6_1") or 0),
            lt_addr=int(value("lt_addr")), flow=int(value("flow")),
            arqn=int(value("arqn")), seqn=int(value("seqn")),
            hec_auto=False, hec_manual=msb(bits("hec", 8)),
            whitening_enabled=whitening, payload_length_bytes=length,
            payload_llid=int(value("llid")), payload_flow=int(value("payload_flow")),
            payload_source=PayloadSourceKind.PATTERN, payload_pattern=text(body) or "0",
        )
        preserve("br_access", bits("access_code", 72))
        preserve("br_type", bits("type", 4))
        preserve("br_length", header[3:8 if header_width == 8 else 13])
        preserve("br_crc", bits("payload_crc", 16))
        if header_width == 16:
            preserve("br_rfu", header[13:])
        if kind.value.startswith(("2-", "3-")):
            bps = 2 if kind.value.startswith("2-") else 3
            preserve("edr_sync", bits("edr_sync", 10*bps))
            preserve("edr_trailer", bits("edr_trailer", 2*bps))
            padding = (-((length+4)*8)) % bps
            if padding:
                raw_padding = (
                    bits("edr_padding", padding) if "edr_padding" in fields else
                    packet.raw_bits[fields["payload"].stop_bit:fields["edr_trailer"].start_bit]
                )
                if raw_padding.size != padding:
                    raise ValueError("EDR padding length does not match the packet")
                preserve("edr_padding", raw_padding)
            expected_size = fields["payload"].stop_bit + padding + 2*bps
        else:
            expected_size = fields["payload"].stop_bit
        if packet.raw_bits.size != expected_size:
            raise ValueError("Classic packet contains undecoded or missing trailing bits")
        project = replace(base, bluetooth_br=settings, fields=bluetooth_br_fields(settings))
    elif packet.protocol_id == "bluetooth.le":
        phy = BluetoothLEPhy(packet.phy_name)
        base = bluetooth_le_project(phy)
        body = bits("payload")
        length = int(value("pdu_length"))
        if body.size != length*8 or bits("pdu").size != 16+length*8+24:
            raise ValueError("LE length and decoded data disagree")
        if packet.raw_bits.size != fields["pdu"].stop_bit:
            raise ValueError("LE packet contains undecoded trailing bits")
        whitening = bool(context.get("whitening_enabled", True))
        channel = context.get("whitening_channel_index")
        if whitening and channel is None:
            raise ValueError("LE whitening channel was not recovered")
        settings = replace(
            base.bluetooth_le, phy=phy, preamble_bits=text(bits("preamble", 8 if phy == BluetoothLEPhy.LE_1M else 16)),
            sync_word_bits=text(bits("access_address", 32)), pdu_header_bits=text(bits("pdu_header", 8)),
            payload_length_bytes=length, payload_source=BluetoothLEPayloadSourceKind.PATTERN,
            payload_pattern=text(body) or "0", crc_enabled=True,
            crc_init=int(context.get("crc_init", 0x555555)),
            whitening_enabled=whitening, whitening_channel_index=int(channel or 0),
        )
        preserve("le_length", bits("pdu_length", 8))
        preserve("le_crc", bits("crc", 24))
        rate = (1e6 if phy == BluetoothLEPhy.LE_1M else 2e6)
        project = replace(base, bluetooth_le=settings, sample_rate_hz=rate*base.samples_per_symbol,
                          fields=bluetooth_le_fields(settings))
    elif packet.protocol_id == "bluetooth.hdt":
        rate = HDTRate(packet.phy_name)
        base = bluetooth_hdt_project(rate)
        header = bits("pdu_header", 8)
        if header[0] or header[1] or int(value("pfi")):
            raise ValueError("Only minimum-header HDT Format 0 is supported")
        body = bits("payload_body")
        pdu_length = int(value("pdu_control"))
        if body.size != (pdu_length-1)*8:
            raise ValueError("HDT PDU length and decoded data disagree")
        if packet.raw_bits.size != 57+pdu_length*8+32:
            raise ValueError("HDT packet contains undecoded trailing bits")
        pca = int(context.get("pca", HDT_RF_TEST_PCA))
        pca = (lsb(bits("pca_a", 16)) << 24) | (pca & 0xFFFFFF)
        settings = replace(
            base.bluetooth_hdt, payload_length_bytes=body.size//8,
            payload_source=HDTPayloadSourceKind.PATTERN, payload_pattern=text(body) or "0",
            pca=pca, nesn=int(value("nesn")), md=int(header[2]),
            sn=lsb(header[3:6]), llid=lsb(header[6:]),
            hec_auto=False, hec_manual=msb(bits("hec_c", 24)),
            crc_auto=False, crc_manual=msb(bits("crc32", 32)),
            crc_init=int(context.get("crc_init", HDT_RF_TEST_CRC32_INIT)),
        )
        preserve("hdt_pdu_control", bits("pdu_control", 9))
        preserve("hdt_rfu", bits("rfu", 1))
        project = replace(base, bluetooth_hdt=settings, fields=bluetooth_hdt_fields(settings))
    elif packet.protocol_id == "dect.classic":
        kind = DectPacketType(packet.packet_type)
        base = dect_project()
        base = dect_project(replace(base.dect, packet_type=kind))
        direction = DectDirection(context.get("direction", next(s.value for s in packet.summary if s.key == "direction")))
        prolonged = fields.get("prolonged_preamble")
        preamble, sync = bits("preamble", 16), bits("sync_word", 16)
        a = bits("a_field", 64)
        settings = replace(
            base.dect, direction=direction, packet_type=kind, prolonged_preamble=prolonged is not None,
            preamble_bits=text(preamble), sync_word_bits=text(sync),
            a_header_bits=text(a[:8]), a_tail_bits=text(a[8:48]),
            r_crc_auto=False, r_crc_bits=text(a[48:]),
            b_field_source=DectBFieldSource.PATTERN,
            b_field_pattern=text(bits("b_field")) if kind != DectPacketType.P00 else "0",
            x_crc_auto=False, x_field_bits=text(bits("x_field", 4)) if kind != DectPacketType.P00 else "0000",
            z_repeat_auto=False, z_field_bits=text(bits("z_field", 4)) if kind in {DectPacketType.P32Z, DectPacketType.P80Z} else "0000",
        )
        b_count = {"P00": 0, "P32": 320, "P32Z": 320, "P80": 800, "P80Z": 800}[kind.value]
        expected_size = (16 if prolonged is not None else 0)+96+b_count+(4 if b_count else 0)+(4 if kind.value.endswith("Z") else 0)
        if packet.raw_bits.size != expected_size:
            raise ValueError("DECT packet contains undecoded or missing bits")
        preserve("dect_s", bits("s_field", 32))
        if prolonged is not None:
            preserve("dect_prolonged", bits("prolonged_preamble", 16))
        project = replace(base, dect=settings, fields=dect_fields(settings))
    else:
        raise ValueError("This decoded protocol is not supported by VSG")

    # Capture center is not an original transmitter setting. Keep template RF defaults.
    project = replace(project, name=f"Received {packet.packet_type or packet.phy_name} Packet",
                      manual_packet_fields=manual)
    issues = validate_project(project)
    if issues:
        raise ValueError("; ".join(f"{i.path}: {i.message}" for i in issues))
    return project


def packet_export_error(packet):
    try:
        project_from_packet(packet)
    except (ValueError, KeyError, TypeError, StopIteration) as error:
        return str(error)
    return None
