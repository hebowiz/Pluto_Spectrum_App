"""Capture IQ -> independent shared PHY/MAC -> VSA measurements."""
from dataclasses import replace
import numpy as np
from pluto_protocol.model import PacketSourceInfo
from pluto_protocol.wifi.detection import detect_packets
from pluto_protocol.wifi.non_ht import analyze_iq
from pluto_vsa.model import IQRecording
from .measurement import measure_region, packet_power
from .model import WiFiCaptureResult, WiFiPacketResult


def analyze_wifi_recording(recording: IQRecording, *, max_packets=128, cancelled=None):
    fs = recording.sample_rate_hz
    if fs not in (20e6, 40e6):
        return WiFiCaptureResult((), ("Non-HT analysis requires 20 or 40 MS/s IQ",))
    if not np.all(np.isfinite(recording.iq)):
        return WiFiCaptureResult((), ("Capture contains non-finite IQ",))
    candidates = detect_packets(recording.iq, fs, max_candidates=max_packets*4)
    packets = []
    covered_until = -1
    factor = int(fs/20e6)
    for candidate in candidates:
        if cancelled is not None and cancelled():
            break
        if candidate.start_sample < covered_until:
            continue
        # Bound work per packet and keep the hint local; do not scan/copy the
        # entire capture again for every PPDU.
        origin = max(0, candidate.start_sample-64*factor)
        stop = min(recording.sample_count, origin+110400*factor)
        diagnostic = {}
        packet = analyze_iq(recording.iq[origin:stop], fs,
            start_hint=candidate.start_sample-origin, measurements=diagnostic,
            source=PacketSourceInfo(source_kind="vsa_capture_iq", center_frequency_hz=recording.center_frequency_hz))
        values = {s.key: s.value for s in packet.summary}
        # Keep recognized STF even if the capture ends inside LTF/SIG/DATA;
        # stationary tones fail LTF and are rejected as false candidates.
        truncated = any(i.code.startswith("wifi.truncated") for i in packet.issues)
        if not values.get("preamble") and not (values.get("stf") and truncated):
            continue
        start = origin+(packet.source.start_sample if packet.source.start_sample is not None
                        else candidate.start_sample-origin)
        count = packet.decode_context.get("n_sym")
        expected_stop = start+(400+80*int(count))*factor if count else start+400*factor
        end = min(recording.sample_count, expected_stop)
        packet = replace(packet, source=replace(packet.source, start_sample=start, stop_sample=end,
                                               packet_index=len(packets)))
        covered_until = end
        average, peak = packet_power(recording,start/fs,end/fs)
        symbols, pilots = diagnostic.get("symbols", []), diagnostic.get("pilots", [])
        signal = measure_region("L-SIG", "BPSK", symbols[:1], pilots[:1]) if symbols else None
        data = None
        if len(symbols) > 1:
            data = measure_region("DATA", str(values["modulation"]), symbols[1:], pilots[1:])
        packets.append(WiFiPacketResult(packet, start, end, candidate.confidence, float(average), float(peak),
            packet.decode_context.get("cfo_hz"), signal, data,
            diagnostic.get("channel", np.empty(0,dtype=complex)),
            diagnostic.get("channel_subcarriers", np.empty(0,dtype=int)),
            np.asarray(diagnostic.get("cpe", []))))
        if len(packets) >= max_packets:
            break
    issues = []
    if not packets:
        issues.append("No confirmed Non-HT packet detected")
    if len(packets) >= max_packets:
        issues.append(f"Capture analysis limited to {max_packets} packets")
    return WiFiCaptureResult(tuple(packets), tuple(issues))
