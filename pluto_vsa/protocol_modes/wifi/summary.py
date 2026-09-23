"""Information-only RF metrics and explicit packet integrity limits."""
def summary_rows(result, recording, *, diagnostics=False):
    packet = result.packet
    values = {s.key: s.display for s in packet.summary}
    info = lambda name, value: (name, str(value), "—", "Info")
    rows = [info("PHY Format", "Non-HT OFDM / ERP"), info("Channel Bandwidth", "20 MHz"),
            info("Center Frequency", f"{recording.center_frequency_hz/1e6:.6f} MHz"),
            info("Data Rate", values.get("rate", "—")+" Mbps"),
            info("Modulation", values.get("modulation", "—")),
            info("Coding Rate", values.get("coding", "—")),
            info("Packet Power", f"{result.packet_power_dbm:.3f} dBm"),
            info("Peak Power", f"{result.peak_power_dbm:.3f} dBm"),
            info("Power Calibration", "Calibrated" if recording.amplitude_calibrated else "Uncalibrated reference"),
            info("Carrier Frequency Error", "Not Available" if result.cfo_hz is None else f"{result.cfo_hz/1e3:.4f} kHz"),
            info("Symbol Clock Error", "Not Available"),
            info("L-SIG Rate", values.get("rate", "—")), info("L-SIG Length / PSDU", values.get("length", "—")),
            info("DATA OFDM Symbols", values.get("n_sym", "—")),
            info("Frame Type / Subtype", values.get("frame_type", "—")+" / "+values.get("frame_subtype", "—"))]
    for region in (result.signal, result.data):
        if region is not None:
            rows.extend((info(region.name+" EVM RMS", f"{region.evm_rms_percent:.4f} %"),
                         info(region.name+" EVM Peak", f"{region.evm_peak_percent:.4f} %"),
                         info(region.name+" Pilot Error RMS", f"{region.pilot_error_rms_percent:.4f} %")))
    for label, key in (("L-SIG Parity", "lsig_parity_valid"), ("PSDU Complete", "psdu_complete"), ("FCS", "fcs_valid")):
        valid = result.integrity[key]
        rows.append((label, "Not Available" if valid is None else str(valid), "Valid",
                     "—" if valid is None else "PASS" if valid else "FAIL"))
    if diagnostics:
        for key in ("stf_metric", "ltf_correlation", "coarse_cfo_hz", "fine_cfo_hz"):
            rows.append(info(key, packet.decode_context.get(key, "—")))
    return rows
