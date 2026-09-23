"""IEEE 802.11 MAC parsing; no dependency on packet generation code."""
from __future__ import annotations

import binascii
from dataclasses import replace
import numpy as np

from pluto_protocol.model import (
    DecodeProbeResult, FieldStatus, IssueSeverity, PacketAnalysisResult,
    PacketDecodeInput, PacketField, PacketIntegritySummary, PacketIssue, PacketSummaryItem,
)


_MANAGEMENT_NAMES = {4: "Probe Request", 5: "Probe Response", 8: "Beacon"}


def _parse_information_elements(frame, start, field, summary, issue, prefix, request):
    """Decode a common IE stream while preserving unknown TLVs and empty SSIDs."""
    cursor = start
    seen = set()
    elements = []
    complete = True
    while cursor < len(frame):
        if cursor + 2 > len(frame) or cursor + 2 + frame[cursor + 1] > len(frame):
            issue(prefix + ".ie_truncated", "Truncated information element")
            complete = False
            break
        ident, count = frame[cursor:cursor+2]
        value = frame[cursor+2:cursor+2+count]
        labels = {0: "SSID", 1: "Supported Rates", 3: "DS Parameter Set", 5: "TIM",
                  42: "ERP Information", 50: "Extended Supported Rates"}
        label = labels.get(ident, "Unknown IE")
        seen.add(ident)
        display = value.hex(" ")
        meaning = ""
        if ident == 0:
            display = value.decode("utf-8", errors="replace")
            meaning = "Wildcard / empty" if request and not value else ""
            summary.append(PacketSummaryItem("ssid", "SSID", display, meaning or display))
        elif ident in (1, 50):
            display = ", ".join(f"{(v & 127)/2:g} Mbps" + (" (basic)" if v & 128 and not request else "") for v in value)
        elif ident == 3 and count == 1:
            display = str(value[0])
            summary.append(PacketSummaryItem("channel", "Channel", value[0], display))
        element = field(f"wifi.ie.{ident}", label, cursor, cursor+2+count, display, meaning)
        if ident not in labels:
            element = replace(element, children=(
                field(f"wifi.ie.{ident}.id", "Element ID", cursor, cursor+1, ident),
                field(f"wifi.ie.{ident}.length", "Length", cursor+1, cursor+2, count),
                field(f"wifi.ie.{ident}.raw", "Raw Value", cursor+2, cursor+2+count)))
        elements.append(element)
        cursor += 2 + count
    return tuple(elements), seen, complete


class WiFiMACDecoder:
    protocol_id = "wifi.non_ht"
    protocol_name = "Wi-Fi Non-HT OFDM"

    def probe(self, packet: PacketDecodeInput) -> DecodeProbeResult:
        # Arbitrary raw bits are not evidence of a Wi-Fi PHY preamble.
        return DecodeProbeResult(self.protocol_id, 1.0 if packet.protocol_hint == self.protocol_id else 0.0)

    def decode(self, packet: PacketDecodeInput) -> PacketAnalysisResult:
        psdu = np.packbits(packet.bits[:packet.bits.size//8*8], bitorder="little").tobytes()
        fields = []
        issues = []
        summary = []
        complete = packet.bits.size % 8 == 0

        def issue(code, text, severity=IssueSeverity.ERROR):
            issues.append(PacketIssue(code, text, severity))

        def field(key, name, start, stop, value=None, meaning="", status=FieldStatus.INFO):
            end = min(stop, len(psdu))
            return PacketField(key, name, start*8, end*8, packet.bits[start*8:end*8],
                               psdu[start:end].hex(" ") if value is None else value,
                               meaning, status)

        if not complete:
            issue("wifi.mac.partial_octet", "PSDU contains a partial octet")
        fcs_valid = None
        frame = psdu
        if len(psdu) >= 4:
            frame = psdu[:-4]
            expected = binascii.crc32(frame)
            received = int.from_bytes(psdu[-4:], "little")
            fcs_valid = received == expected
            if not fcs_valid:
                issue("wifi.fcs", "FCS does not match the recovered PSDU")
            fcs_field = field("wifi.fcs", "FCS", len(psdu)-4, len(psdu),
                              f"0x{received:08X}", f"Calculated 0x{expected:08X}",
                              FieldStatus.VALID if fcs_valid else FieldStatus.INVALID)
        else:
            complete = False
            issue("wifi.mac.truncated", "PSDU is too short to contain FCS")
            fcs_field = None

        packet_type = "Unknown"
        header = []
        body = []
        header_len = 0
        frame_body = None
        if len(frame) < 2:
            complete = False
            issue("wifi.mac.header", "Missing Frame Control")
        else:
            fc = int.from_bytes(frame[:2], "little")
            version, kind, subtype = fc & 3, (fc >> 2) & 3, (fc >> 4) & 15
            names = {0: "Management", 1: "Control", 2: "Data", 3: "Extension"}
            packet_type = _MANAGEMENT_NAMES.get(subtype, names[kind]) if kind == 0 else names[kind]
            header.append(field("wifi.frame_control", "Frame Control",0,2,f"0x{fc:04X}"))
            header.extend((
                PacketField("wifi.frame_type", "Frame Type", 2, 4, packet.bits[2:4], names[kind]),
                PacketField("wifi.frame_subtype", "Frame Subtype", 4, 8, packet.bits[4:8],
                            f"{packet_type} ({subtype})" if kind == 0 and subtype in _MANAGEMENT_NAMES else str(subtype)),
            ))
            flags = [("protocol_version", "Protocol Version", 0, 2)] + [
                (key, label, bit, bit+1) for bit, key, label in (
                    (8,"to_ds","To DS"), (9,"from_ds","From DS"),
                    (10,"more_fragments","More Fragments"), (11,"retry","Retry"),
                    (12,"power_management","Power Management"), (13,"more_data","More Data"),
                    (14,"protected","Protected Frame"), (15,"order","Order"))]
            control_fields = tuple(
                PacketField("wifi."+key,label,a,b,packet.bits[a:b], (fc >> a) & ((1 << (b-a))-1))
                for key,label,a,b in flags)
            header[0] = replace(header[0], children=(control_fields[0], *header[1:], *control_fields[1:]))
            del header[1:]
            summary.extend((PacketSummaryItem("frame_type","Frame Type",kind,names[kind]),
                            PacketSummaryItem("frame_subtype","Frame Subtype",subtype,packet_type if kind == 0 and subtype in _MANAGEMENT_NAMES else str(subtype))))
            if version:
                issue("wifi.mac.version", "Unsupported MAC protocol version", IssueSeverity.WARNING)
            header_len = 24 if kind in (0,2) else (10 if kind == 1 and subtype in (12,13) else 16)
            if kind == 2:
                header_len += 6 if (fc & 0x0300) == 0x0300 else 0
                header_len += 2 if subtype & 8 else 0
                header_len += 4 if subtype & 8 and fc & 0x8000 else 0
            if len(frame) < header_len:
                complete = False
                issue("wifi.mac.header", f"Truncated MAC header: need {header_len} bytes")
            if len(frame) >= 4:
                header.append(field("wifi.duration", "Duration / ID",2,4,int.from_bytes(frame[2:4],"little")))
            for number, offset in ((1,4),(2,10),(3,16)):
                if offset+6 <= min(len(frame),header_len):
                    address = ":".join(f"{v:02X}" for v in frame[offset:offset+6])
                    header.append(field(f"wifi.address{number}",f"Address {number}",offset,offset+6,address))
                    if kind == 0:
                        key, label = {1: ("destination_address", "Destination"), 2: ("source_address", "Source"), 3: ("bssid", "BSSID")}[number]
                        summary.append(PacketSummaryItem(key, label, address, address))
                        header[-1] = replace(header[-1], meaning=label)
            if kind == 2 and fc & 0x0300 == 0x0300 and len(frame) >= 30:
                header.append(field("wifi.address4","Address 4",24,30,":".join(f"{v:02X}" for v in frame[24:30])))
            if kind in (0,2) and len(frame) >= 24:
                seq = int.from_bytes(frame[22:24],"little")
                header.append(field("wifi.sequence_control","Sequence Control",22,24,seq,
                                    f"Sequence {seq >> 4}, fragment {seq & 15}"))
                header[-1] = replace(header[-1], children=(
                    PacketField("wifi.fragment_number","Fragment Number",176,180,packet.bits[176:180],seq & 15),
                    PacketField("wifi.sequence_number","Sequence Number",180,192,packet.bits[180:192],seq >> 4)))
            if header_len < len(frame):
                frame_body = field("payload_body","Frame Body",header_len,len(frame),
                                   f"{len(frame)-header_len} bytes")
            if kind == 0 and subtype in _MANAGEMENT_NAMES:
                request = subtype == 4
                prefix = "wifi." + packet_type.lower().replace(" ", "_")
                ie_start = 24 if request else 36
                if len(frame) < ie_start:
                    complete = False
                    issue(prefix + ".fixed" if not request else "wifi.mac.header",
                          f"Truncated {packet_type} {'MAC header' if request else 'fixed parameters'}")
                else:
                    if not request:
                        fixed_fields = []
                        for key, name, a, b in (("timestamp", "Timestamp", 24, 32), ("beacon_interval", "Beacon Interval", 32, 34), ("capability", "Capability Information", 34, 36)):
                            val = int.from_bytes(frame[a:b], "little")
                            fixed_fields.append(field("wifi."+key, name, a, b, val, "TU (1024 us)" if key == "beacon_interval" else ""))
                        body.append(PacketField(prefix + ".fixed", packet_type + " Fixed Parameters", 192, 288,
                                                children=tuple(fixed_fields)))
                    elements, seen, ies_complete = _parse_information_elements(
                        frame, ie_start, field, summary, issue, prefix, request)
                    complete = complete and ies_complete
                    body.append(PacketField(prefix + ".ies", "Information Elements", ie_start*8, len(frame)*8,
                                            children=elements))
                    required = [(0, "SSID"), (1, "Supported Rates")]
                    if not request:
                        required.append((3, "DS Parameter Set"))
                    if subtype == 8:
                        required.append((5, "TIM"))
                    if subtype == 5:
                        required.append((42, "ERP Information"))
                    for ident, label in required:
                        if ident not in seen:
                            issue(prefix + ".missing_ie", f"{packet_type} is missing {label}", IssueSeverity.WARNING)
                    if not request:
                        capability = int.from_bytes(frame[34:36], "little")
                        if not capability & 1 or capability & 0x12:
                            issue(prefix + ".capability", f"{packet_type} is not the default open infrastructure BSS", IssueSeverity.WARNING)

        fields.append(PacketField("wifi.mac_header","MAC Header",0,min(len(frame),header_len)*8,children=tuple(header)))
        if frame_body is not None:
            fields.append(replace(frame_body, children=tuple(body)))
        if fcs_field is not None:
            fields.append(fcs_field)
        summary.append(PacketSummaryItem("fcs_valid","FCS Valid",fcs_valid,str(fcs_valid),
                                        FieldStatus.VALID if fcs_valid else FieldStatus.INVALID))
        return PacketAnalysisResult("1.0",self.protocol_id,self.protocol_name,"Non-HT OFDM",packet_type,
            tuple(summary),(PacketField("wifi.psdu","PSDU",0,len(psdu)*8,children=tuple(fields)),),
            tuple(issues),PacketIntegritySummary(crc_valid=fcs_valid,complete=complete),packet.source,packet.bits,
            {"verification_source":"PSDU bits only; PHY not verified"})
