"""Separate available values from standards decisions and decode integrity."""
from dataclasses import dataclass, asdict
import numpy as np

from . import standards as std
from .results import MeasurementKind as Kind, MeasurementResult as Result, MeasurementStatus as Status


def capture_statistics(packets):
    """Same-rate qualifying packets within one capture; never mix rates.

    Printed Eq. (17-28) averages per-packet RMS (equal packet weighting).
    DATA length and random-payload evidence are separate observation conditions.
    Refresh/Continuous do not accumulate or recount a previous capture.
    """
    groups = {}
    for p in packets:
        rate = p.packet.decode_context.get("rate_mbps")
        rms = p.rf_details.get("constellation_rms")
        if (rate not in std.REFERENCE_RCE_LIMIT_DB or rms is None or p.data is None
                or len(p.data.measured)<std.REFERENCE_MIN_SYMBOLS or not p.measurement_eligible):
            continue
        groups.setdefault(rate,[]).append(rms)
    return {rate:dict(packet_count=len(values),rms=float(np.mean(values))) for rate,values in groups.items()}


@dataclass(frozen=True)
class MeasurementConditions:
    """User-confirmed conditions for the current test setup, never inferred from FCS.

    Confirmation does not calibrate IQ. In particular OTA channel response and
    receiver DC must not be attributed to the transmitter without characterization.
    """
    frequency_reference_verified: bool = False
    receiver_response_verified: bool = False
    random_payload_verified: bool = False
    non_vht_dut: bool = False


def measurement_results(p, recording, statistics, *, conditions=None):
    setup = conditions or MeasurementConditions()
    context = p.packet.decode_context
    decoded = {s.key:s.value for s in p.packet.summary}
    profile = std.reference_profile(recording.center_frequency_hz)
    rate = context.get("rate_mbps")
    reference_limit = std.REFERENCE_RCE_LIMIT_DB.get(rate)
    receiver_missing = (() if setup.receiver_response_verified else
        ("Receiver IQ/DC/phase noise/flatness accuracy and conducted path not verified",))
    bandwidth_missing = (() if (recording.usable_bandwidth_hz or recording.sample_rate_hz)>=16.25e6 else
        ("Usable acquisition bandwidth does not cover all 52 subcarriers",))

    def info(key,name,value,unit="",kind=Kind.PHY_DECODE,visible=True,**kwargs):
        return Result(key,name,kind,value,unit,default_visible=visible,**kwargs)

    def standard(key,name,value,unit,clause,*,limit,candidate=None,missing=(),
                 observations=(),insufficient=False,passed=None,metadata=None):
        reasons = list(missing)
        if profile is None:
            reasons.append("PHY/band profile unavailable")
        if not std.CURRENT_MEASUREMENT_CLAUSES_VERIFIED:
            reasons.append("Current standard clauses are not verified")
        if value is None or not np.isfinite(value):
            value = None
            reasons.append("Measurement value unavailable")
        eligible = value is not None and not reasons and not insufficient and passed is not None
        status = ((Status.PASS if passed else Status.FAIL) if eligible else
                  Status.INSUFFICIENT_DATA if insufficient else Status.NOT_MEASURED)
        meta = dict(metadata or {},reference_revision=std.REFERENCE_REVISION,
                    current_revision_verified=std.CURRENT_MEASUREMENT_CLAUSES_VERIFIED,
                    reference_limit=candidate,phy=profile.phy if profile else None,
                    band=profile.band if profile else None,confirmed_setup=asdict(setup))
        if profile and profile.inheritance_clause and clause.startswith("17."):
            clause += "; ERP inheritance " + profile.inheritance_clause
        return Result(key,name,Kind.STANDARD,value,unit,limit,status,
                      std.reference(clause),eligible,conditions=tuple((*observations,*reasons)),metadata=meta)

    rows = [info("packet_power","Packet Power",p.packet_power_dbm,"dBm",Kind.STANDARD,
        standard_reference=std.reference("18.4.7.2" if profile and profile.phy=="ERP-OFDM" else "17.3.9.2"),
        conditions=("No universal power limit: applicable regional regulations are not evaluated",))]
    tolerance = profile.frequency_tolerance_ppm if profile else None
    ppm = p.cfo_hz/recording.center_frequency_hz*1e6 if p.cfo_hz is not None and recording.center_frequency_hz else None
    frequency_limit = f"±{tolerance:g} ppm" if tolerance is not None else "N/A (unknown band)"
    rows.append(standard("carrier_frequency_error","Carrier Frequency Error",p.cfo_hz,"Hz",
        profile.frequency_clause if profile else "17.3.9.5 / 18.4.7.4",candidate=tolerance,limit=frequency_limit,
        missing=(() if setup.frequency_reference_verified else ("Receiver frequency reference not verified",))+
                (() if p.integrity["preamble_detected"] else ("Complete L-LTF synchronization unavailable",)),
        passed=abs(ppm)<=tolerance+1e-10 if ppm is not None and tolerance is not None else None,
        metadata=dict(reference_limit_unit="ppm",ppm=ppm)))
    rows.append(standard("symbol_clock_error","Symbol Clock Frequency Error",None,"ppm",
        profile.clock_clause if profile else "17.3.9.6 / 18.4.7.5",candidate=tolerance,limit=frequency_limit,
        missing=("OFDM sample-clock estimator is not implemented; CFO is not a clock-error substitute",)))

    group = statistics.get(rate,{})
    count = group.get("packet_count",0)
    rms = group.get("rms",p.rf_details.get("constellation_rms"))
    rce_db = float(20*np.log10(max(rms,1e-15))) if rms is not None else None
    rce = standard("relative_constellation_error","Relative Constellation Error",rce_db,"dB",
        "17.3.9.7.4, Table 17-20, 17.3.9.8 Eq. (17-28)",candidate=reference_limit,
        limit=f"≤ {reference_limit:g} dB" if reference_limit is not None else "N/A (unknown rate)",
        insufficient=count<std.REFERENCE_MIN_PACKETS,
        missing=receiver_missing+bandwidth_missing+(() if setup.random_payload_verified else ("Random test payload not verified",)),
        observations=(f"Capture: {count}/20 qualifying same-rate packets, >=16 DATA symbols each",),
        passed=rce_db<=reference_limit+1e-10 if rce_db is not None and reference_limit is not None else None,
        metadata=dict(canonical_rms=rms,packet_count=count,rate_mbps=rate,
                      scope="same-rate capture" if count else "selected packet",
                      min_packets=20,min_data_symbols=16,tones=52,random_payload_verified=setup.random_payload_verified))
    rows.append(rce)
    rows.append(info("evm_rms","EVM RMS",None if rms is None else 100*rms,"%",Kind.STANDARD,
        status=Status.INFO if rms is not None else Status.NOT_MEASURED,
        canonical_id=rce.measurement_id,conditions=("Representation of Relative Constellation Error; no separate decision",),
        metadata=dict(canonical_rms=rms)))

    training = p.rf_details.get("training",{})
    leakage = training.get("leakage_db")
    training_power = training.get("training_power")
    power_dbm = (float(10*np.log10(max(training_power/recording.full_scale**2,1e-30)))+
                 recording.dbfs_to_dbm_offset_db) if training_power is not None and recording.amplitude_calibrated else None
    leakage_limit = (max(std.REFERENCE_LEAKAGE_LIMIT_DB,std.REFERENCE_LEAKAGE_ABSOLUTE_DBM-power_dbm)
                     if power_dbm is not None else std.REFERENCE_LEAKAGE_LIMIT_DB)
    leakage_missing = receiver_missing+bandwidth_missing
    if not setup.non_vht_dut:
        leakage_missing += ("Confirm non-VHT DUT: VHT STAs require 21.3.17.4.2 for all PPDU formats (not implemented)",)
    # Without absolute calibration the relative branch can prove PASS, but
    # cannot prove FAIL: the -20 dBm exception may still permit the emission.
    if power_dbm is None and leakage is not None and leakage>leakage_limit+1e-10:
        leakage_missing += ("Calibrated transmitter-plane power required to evaluate the -20 dBm exception",)
    rows.append(standard("center_frequency_leakage","Transmit Center Frequency Leakage",leakage,"dB",
        "17.3.9.7.2",candidate=leakage_limit,
        limit=f"≤ {leakage_limit:g} dB" if power_dbm is not None else "≤ -15 dB / -20 dBm",
        missing=leakage_missing,passed=leakage<=leakage_limit+1e-10 if leakage is not None else None,
        observations=("Non-VHT DUT: max(P - 15, -20) dBm; P from channel-training power",),
        metadata=dict(training,reference_power="sum of active-tone and DC training energies",
                      training_power_dbm=power_dbm,leakage_dbm=None if power_dbm is None or leakage is None else power_dbm+leakage)))
    flatness = training.get("flatness_margin_db")
    rows.append(standard("spectral_flatness","Spectral Flatness",flatness,"dB margin",
        "17.3.9.7.3",candidate=dict(inner=(-4.,4.),edge=(-6.,4.)),limit="Inner ±4; edge -6/+4 dB",
        missing=receiver_missing+bandwidth_missing,passed=flatness>=-1e-10 if flatness is not None else None,metadata=training))
    spectrum = p.rf_details.get("spectrum",{})
    mask_missing = ["Equivalent Digital Measurement; 100 kHz ENBW Welch; 30 kHz instrument VBW/detector not reproduced",
                    *receiver_missing]
    span_missing = not spectrum.get("full_span",False) or spectrum.get("usable_bandwidth_hz",0)<60e6
    if span_missing:
        mask_missing.append("Capture cannot cover both +/-30 MHz mask edges and out-of-band regions")
    rows.append(standard("transmit_spectrum_mask","Transmit Spectrum Mask",spectrum.get("worst_margin_db"),"dB margin",
        "17.3.9.3, Figure 17-13",candidate="20 MHz default upper emission mask",limit="IEEE 17.3.9.3 (20 MHz)",
        missing=tuple(mask_missing),insufficient=bool(spectrum) and span_missing,metadata=spectrum))

    rows.extend((info("data_rate","Data Rate",rate,"Mbps"),
        info("modulation","Modulation",decoded.get("modulation")),
        info("coding_rate","Coding Rate",decoded.get("coding")),
        info("lsig_length","L-SIG Length",context.get("length",decoded.get("length")),"byte"),
        info("phy_format","PHY Format",profile.phy if profile else "Non-HT OFDM (band unknown)",visible=False),
        info("channel_bandwidth","Channel Bandwidth",20,"MHz",visible=False),
        info("lsig_rate","L-SIG Rate",rate,"Mbps",visible=False),
        info("data_symbols","DATA OFDM Symbol Count",context.get("n_sym"),visible=False)))
    parity = p.integrity["lsig_parity_valid"]
    fcs = p.integrity["fcs_valid"]
    for key,label,valid,kind in (("lsig_parity","L-SIG Parity",parity,Kind.PHY_DECODE),
                               ("fcs","FCS",fcs,Kind.MAC_DECODE)):
        status = Status.NOT_AVAILABLE if valid is None else Status.PASS if valid else Status.FAIL
        rows.append(Result(key,label,kind,None if valid is None else "Valid" if valid else "Invalid",
                           limit="Valid",status=status,measurement_conditions_satisfied=valid is not None))
    rows.extend((info("frame_type","Frame Type",decoded.get("frame_type"),kind=Kind.MAC_DECODE),
                 info("frame_subtype","Frame Subtype",decoded.get("frame_subtype"),kind=Kind.MAC_DECODE),
                 info("psdu_length","PSDU Length",len(p.packet.raw_bits)//8 if p.integrity["psdu_complete"] else None,"byte",Kind.MAC_DECODE)))
    diagnostics = [info("peak_power","Peak Power",p.peak_power_dbm,"dBm",Kind.DIAGNOSTIC,False),
        info("power_calibration","Power Calibration","Calibrated amplitude" if recording.amplitude_calibrated else "Uncalibrated reference",kind=Kind.DIAGNOSTIC,visible=False)]
    for region in (p.signal,p.data):
        if region is not None:
            for suffix,label,value in (("evm_rms_48","Data-tone EVM RMS (48 tones)",region.evm_rms_percent),
                    ("evm_peak","EVM Peak",region.evm_peak_percent),("pilot_error","Pilot Error RMS",region.pilot_error_rms_percent)):
                diagnostics.append(info(region.name.lower()+"_"+suffix,region.name+" "+label,value,"%",Kind.DIAGNOSTIC,False))
    for key,label,unit in (("stf_metric","L-STF Correlation",""),("ltf_correlation","L-LTF Correlation",""),
            ("coarse_cfo_hz","Coarse CFO","Hz"),("fine_cfo_hz","Fine CFO","Hz")):
        diagnostics.append(info(key,label,context.get(key),unit,Kind.DIAGNOSTIC,False))
    diagnostics.extend((info("residual_cfo","Residual CFO",None,"Hz",Kind.DIAGNOSTIC,False,status=Status.NOT_MEASURED),
        info("timing_offset","Timing Offset",p.start_sample,"sample",Kind.DIAGNOSTIC,False),
        info("packet_detection_metric","Packet Detection Metric",p.detection_confidence,kind=Kind.DIAGNOSTIC,visible=False),
        info("cpe_rms","Common Phase Error RMS",float(np.sqrt(np.mean(p.common_phase_error_rad**2))) if p.common_phase_error_rad.size else None,"rad",Kind.DIAGNOSTIC,False),
        info("channel_quality","Channel Estimate Quality",None,kind=Kind.DIAGNOSTIC,visible=False,status=Status.NOT_MEASURED)))
    return tuple(rows+diagnostics)
