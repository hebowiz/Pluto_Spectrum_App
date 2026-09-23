"""Independent IEEE legacy OFDM receiver for packet verification.

Only IQ and sample rate enter the PHY. No VSG tables, encoder helpers, packet
bits, seed, length, rate or generator metadata are used to recover the packet.
This is an offline first-packet validator, not a continuous RF receiver.
"""
from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
from scipy.signal import correlate

from pluto_protocol.model import (
    FieldStatus, IssueSeverity, PacketAnalysisResult, PacketDecodeInput, PacketField,
    PacketIntegritySummary, PacketIssue, PacketSourceInfo, PacketSummaryItem,
)
from .mac import WiFiMACDecoder

# Clause 17, Table 78 and Table 80, indexed by the four received RATE bits.
RATES = {
    (1,1,0,1):(6,1,"1/2",24), (1,1,1,1):(9,1,"3/4",36),
    (0,1,0,1):(12,2,"1/2",48), (0,1,1,1):(18,2,"3/4",72),
    (1,0,0,1):(24,4,"1/2",96), (1,0,1,1):(36,4,"3/4",144),
    (0,0,0,1):(48,6,"2/3",192), (0,0,1,1):(54,6,"3/4",216),
}
DATA = np.array([*range(-26,-21),*range(-20,-7),*range(-6,0),*range(1,7),*range(8,21),*range(22,27)])
PILOTS = np.array([-21,-7,7,21])
LONG = np.array([1,1,-1,-1,1,1,-1,1,-1,1,1,1,1,1,1,-1,-1,1,1,-1,1,-1,1,1,1,1,0,
                 1,-1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,1,1,1])
# Explicit published polarity sequence, independent of the TX scrambler.
POLARITY = np.array([
    1,1,1,1,-1,-1,-1,1,-1,-1,-1,-1,1,1,-1,1,-1,-1,1,1,-1,1,1,-1,1,1,1,1,1,1,-1,1,
    1,1,-1,1,1,-1,-1,1,1,1,-1,1,-1,-1,-1,1,-1,1,-1,-1,1,-1,-1,1,1,1,1,1,-1,-1,1,1,
    -1,-1,1,-1,1,-1,1,1,-1,-1,-1,1,1,-1,-1,-1,-1,1,-1,-1,1,-1,1,1,1,1,-1,1,-1,1,-1,1,
    -1,-1,-1,-1,-1,1,-1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,-1,-1])


def deinterleave(values: np.ndarray, n_bpsc: int) -> np.ndarray:
    """Clause 17.3.5.6 receiver equations (18)/(19), j -> i -> k."""
    count = len(values)
    j = np.arange(count)
    s = max(n_bpsc // 2, 1)
    i = s*(j//s) + (j + (16*j)//count) % s
    k = 16*i - (count-1)*((16*i)//count)
    output = np.empty_like(values)
    output[k] = values
    return output


def depuncture(values: np.ndarray, rate: str, bit_count: int) -> np.ndarray:
    # Erasures have zero soft evidence, not a guessed bit value.
    keep = np.resize({"1/2":[True,True],"2/3":[True,True,True,False],
                      "3/4":[True,True,True,False,False,True]}[rate],bit_count*2)
    if np.count_nonzero(keep) != len(values):
        raise ValueError("Punctured DATA length does not match decoded RATE/LENGTH")
    result = np.zeros(bit_count*2)
    result[keep] = values
    return result


def viterbi(soft: np.ndarray) -> np.ndarray:
    """64-state soft Viterbi, MSB-newest register and 133/171 octal taps.

    Starts at zero; final state is selected by metric because scrambled PAD
    follows the six zero DATA tail bits. TAIL is checked after decoding.
    """
    pairs = np.asarray(soft,dtype=float).reshape(-1,2)
    dest = np.arange(64)
    incoming = dest >> 5
    pred = np.stack(((dest & 31)*2,(dest & 31)*2+1),axis=1)
    word = (incoming[:,None] << 6) | pred
    labels = np.array([[[1 if int(v & mask).bit_count()%2 else -1 for mask in (0o133,0o171)]
                        for v in row] for row in word])
    cost = np.full(64,np.inf)
    cost[0] = 0
    history = np.empty((len(pairs),64),dtype=np.uint8)
    for index, pair in enumerate(pairs):
        branches = cost[pred] - np.sum(labels*pair,axis=2)
        choice = branches[:,1] < branches[:,0]
        history[index] = choice
        cost = branches[dest,choice.astype(int)]
        cost -= np.min(cost)
    state = int(np.argmin(cost))
    bits = np.empty(len(pairs),dtype=np.uint8)
    for index in range(len(bits)-1,-1,-1):
        bits[index] = state >> 5
        state = int(pred[state,history[index,state]])
    return bits


def demap(symbols: np.ndarray, width: int) -> np.ndarray:
    """Max-log soft bits from IEEE Tables 82-85, positive means bit one."""
    labels = ((np.arange(2**width)[:,None] >> np.arange(width-1,-1,-1)) & 1)
    if width == 1:
        constellation = np.array([-1,1],dtype=complex)
    else:
        axis = {2:[-1,1],4:[-3,-1,3,1],6:[-7,-5,-1,-3,7,5,1,3]}[width]
        axis = np.array(axis)
        half = width//2
        indices = np.arange(2**width)
        constellation = (axis[indices >> half] + 1j*axis[indices & ((1<<half)-1)]) / math.sqrt({2:2,4:10,6:42}[width])
    distances = abs(symbols[:,None]-constellation[None,:])**2
    return np.stack([np.min(distances[:,labels[:,b]==0],axis=1)-np.min(distances[:,labels[:,b]==1],axis=1)
                     for b in range(width)],axis=1).ravel()


def descramble(bits: np.ndarray) -> tuple[np.ndarray, int]:
    # Recover the unknown state from the first seven zero SERVICE bits.
    # Receiver register stores x1 at the MSB, independently of TX orientation.
    def sequence(seed, count):
        state = seed
        output = np.empty(count,dtype=np.uint8)
        for i in range(count):
            value = ((state >> 3) ^ state) & 1
            output[i] = value
            state = (state >> 1) | (value << 6)
        return output
    for seed in range(1,128):
        if np.array_equal(sequence(seed,7),bits[:7]):
            return np.bitwise_xor(bits,sequence(seed,len(bits))), seed
    raise ValueError("No nonzero scrambler state matches SERVICE")


def analyze_iq(iq: np.ndarray, sample_rate_hz: float, *, source: PacketSourceInfo | None = None,
               start_hint: int | None = None, measurements: dict | None = None) -> PacketAnalysisResult:
    """Recover the first PPDU from 20/40 MS/s IQ, including structured failures."""
    source = source or PacketSourceInfo(source_kind="iq")
    issues: list[PacketIssue] = []
    fields: list[PacketField] = []
    summary: list[PacketSummaryItem] = []
    context = {"verification_source":"IQ", "sample_rate_hz":float(sample_rate_hz)}

    def item(key, name, value, valid=True):
        status = FieldStatus.VALID if valid else FieldStatus.INVALID
        summary.append(PacketSummaryItem(key,name,value,str(value),status))
        # PHY fields have no invented MAC bit or contiguous DATA sample span.
        fields.append(PacketField("wifi.phy."+key,name,0,0,value=value,status=status))

    def fail(code, text):
        issues.append(PacketIssue(code,text,IssueSeverity.ERROR))
        return PacketAnalysisResult("1.0","wifi.non_ht","Wi-Fi Non-HT OFDM","Non-HT OFDM",None,
            tuple(summary),(PacketField("wifi.phy","PHY",0,0,children=tuple(fields)),),tuple(issues),
            PacketIntegritySummary(complete=False),source,np.empty(0,dtype=np.uint8),context)

    if sample_rate_hz not in (20_000_000,40_000_000):
        return fail("wifi.sample_rate","IQ verification requires 20 or 40 MS/s")
    factor = int(sample_rate_hz/20_000_000)
    x = np.asarray(iq)
    if x.ndim != 1 or not np.iscomplexobj(x) or not np.all(np.isfinite(x)):
        return fail("wifi.iq","IQ must be a finite one-dimensional complex array")
    x = x[::factor].astype(np.complex128)
    if not len(x) or np.max(abs(x)) < 1e-12:
        return fail("wifi.preamble","No preamble detected")
    # Energy gates a local STF search; packet contents never come from metadata.
    active = np.flatnonzero(abs(x) > np.max(abs(x))*0.04)
    begin = int(active[0]) if start_hint is None else max(0, int(start_hint)//factor)
    if len(x)-begin < 144:
        return fail("wifi.truncated.stf","Truncated L-STF")
    stf_offset = 16
    if start_hint is not None:
        # A detector's plateau can start slightly before the actual STF.
        # Select a complete periodic window; fine timing still comes from LTF.
        scores = []
        for offset in range(0, min(65, len(x)-begin-111), 4):
            u, v = x[begin+offset:begin+offset+96], x[begin+offset+16:begin+offset+112]
            scores.append((abs(np.vdot(u,v))/max(np.linalg.norm(u)*np.linalg.norm(v),1e-20), offset))
        stf_offset = max(scores)[1] if scores else 16
    a, b = x[begin+stf_offset:begin+stf_offset+96], x[begin+stf_offset+16:begin+stf_offset+112]
    score = abs(np.vdot(a,b))/max(np.linalg.norm(a)*np.linalg.norm(b),1e-20)
    if score < 0.8:
        return fail("wifi.preamble","L-STF repetition not detected")
    item("stf","L-STF",True)
    coarse = np.angle(np.vdot(a,b))/16
    context.update(stf_metric=float(score), coarse_cfo_hz=float(coarse*20e6/(2*np.pi)))
    bins = np.zeros(64,dtype=complex)
    bins[np.arange(-26,27)%64] = LONG
    reference = np.fft.ifft(bins)
    window = x[begin:min(len(x),begin+512)] * np.exp(-1j*coarse*np.arange(min(len(x)-begin,512)))
    if len(window) < 320:
        return fail("wifi.truncated.ltf","Truncated L-LTF")
    correlation = abs(correlate(window,reference,mode="valid"))
    power = np.sqrt(np.convolve(abs(window)**2,np.ones(64),mode="valid"))*np.linalg.norm(reference)
    correlation /= np.maximum(power,1e-20)
    paired = np.minimum(correlation[:-64],correlation[64:])
    # First long symbol follows 160 STF + 32 GI2. GI contains a repeated half;
    # search near this transition rather than selecting a later payload match.
    lo, hi = (125 if start_hint is not None else 150), min(285 if start_hint is not None else 235,len(paired))
    if hi <= lo:
        return fail("wifi.truncated.ltf","Incomplete L-LTF pair")
    ltf_start = lo + int(np.argmax(paired[lo:hi]))
    if paired[ltf_start] < 0.8:
        return fail("wifi.ltf","L-LTF correlation failed")
    packet_start = begin + ltf_start - 192
    context["ltf_correlation"] = float(paired[ltf_start])
    if packet_start < 0:
        return fail("wifi.truncated.preamble","Packet begins before available IQ")
    source = replace(source,start_sample=packet_start*factor)
    x = x[packet_start:]
    # Rebase coarse phase, estimate residual CFO using repeated long symbols.
    y = x[:320]*np.exp(-1j*coarse*np.arange(min(len(x),320)))
    fine = np.angle(np.vdot(y[192:256],y[256:320]))/64
    omega = coarse+fine
    context["cfo_hz"] = omega*20e6/(2*np.pi)
    context["fine_cfo_hz"] = fine*20e6/(2*np.pi)
    item("ltf","L-LTF",True)
    item("preamble","Preamble Detected",True)
    if len(x) < 400:
        item("lsig_complete", "L-SIG Complete", False, False)
        return fail("wifi.truncated.signal","Truncated L-SIG")
    item("lsig_complete", "L-SIG Complete", True)
    # Only a bounded PPDU prefix is needed, even when a cyclic waveform has
    # millions of idle/repeat samples. Maximum legacy LENGTH is 4095 octets.
    x = x[:110000]*np.exp(-1j*omega*np.arange(min(len(x),110000)))
    h = np.zeros(64,dtype=complex)
    used = np.r_[np.arange(-26,0),np.arange(1,27)]
    ltf_bins = (np.fft.fft(x[192:256])+np.fft.fft(x[256:320]))/2
    h[used%64] = ltf_bins[used%64]/LONG[used+26]
    if np.min(abs(h[used%64])) < 1e-10:
        return fail("wifi.channel","Degenerate channel estimate")
    if measurements is not None:
        measurements.update(channel=h[used%64].copy(), channel_subcarriers=used.copy(),
                            data_subcarriers=DATA.copy(), symbols=[], pilots=[], cpe=[])

    def symbol(start, pilot_index):
        z = np.fft.fft(x[start+16:start+80])
        equalized = np.zeros(64,dtype=complex)
        equalized[used%64] = z[used%64]/h[used%64]
        expected = np.array([1,1,1,-1])*POLARITY[pilot_index%127]
        phase = np.angle(np.vdot(expected,equalized[PILOTS%64]))
        corrected = equalized[DATA%64]*np.exp(-1j*phase)
        if measurements is not None:
            measurements["symbols"].append(corrected.copy())
            measurements["pilots"].append(equalized[PILOTS%64]*np.exp(-1j*phase)-expected)
            measurements["cpe"].append(float(phase))
        return corrected

    sig = viterbi(deinterleave(demap(symbol(320,0),1),1))
    parity = not bool(np.sum(sig[:18]) % 2)
    length = sum(int(sig[5+i]) << i for i in range(12))
    rate_info = RATES.get(tuple(sig[:4]))
    item("lsig_parity","L-SIG Parity Valid",parity,parity)
    item("length","L-SIG LENGTH",length)
    item("reserved", "L-SIG Reserved", int(sig[4]), not bool(sig[4]))
    item("signal_tail", "L-SIG Tail", "".join(map(str, sig[18:])), not bool(np.any(sig[18:])))
    context["lsig_bits"] = tuple(int(bit) for bit in sig)
    fields.append(PacketField("wifi.lsig", "L-SIG logical bits", 0, 24, sig, children=tuple(
        PacketField("wifi.lsig."+key, name, a, b, sig[a:b], "".join(map(str,sig[a:b])))
        for key,name,a,b in (("rate","RATE",0,4),("reserved","Reserved",4,5),
                            ("length","LENGTH",5,17),("parity","Parity",17,18),("tail","Tail",18,24)))))
    if rate_info is None or not parity or sig[4] or np.any(sig[18:]):
        return fail("wifi.signal","Invalid L-SIG RATE, reserved bit, parity or tail")
    mbps,width,coding,n_dbps = rate_info
    item("rate","L-SIG RATE",mbps)
    item("modulation","Modulation",{1:"BPSK",2:"QPSK",4:"16QAM",6:"64QAM"}[width])
    item("coding","Coding Rate",coding)
    count = math.ceil((16+8*length+6)/n_dbps)
    item("n_sym","N_SYM",count)
    stop = 400+80*count
    context.update(rate_mbps=mbps,length=length,n_sym=count)
    if length == 0:
        return fail("wifi.length","Zero PSDU length is not supported")
    if len(x) < stop:
        item("data_complete", "DATA Complete", False, False)
        item("psdu_complete","PSDU Complete",False,False)
        return fail("wifi.truncated.data",f"DATA needs {stop} native samples; only {len(x)} available")
    item("data_complete", "DATA Complete", True)
    soft = np.concatenate([deinterleave(demap(symbol(400+80*n,n+1),width),width) for n in range(count)])
    bits = viterbi(depuncture(soft,coding,count*n_dbps))
    tail_start = 16+8*length
    tail_valid = not bool(np.any(bits[tail_start:tail_start+6]))
    try:
        decoded, seed = descramble(bits)
    except ValueError as error:
        return fail("wifi.scrambler",str(error))
    service_valid = not bool(np.any(decoded[:16]))
    pad_valid = not bool(np.any(decoded[tail_start+6:]))
    for key, valid in (("service",service_valid),("tail",tail_valid),("pad",pad_valid)):
        item(key,key.upper()+" Valid",valid,valid)
        if not valid:
            issues.append(PacketIssue("wifi."+key,f"Invalid DATA {key.upper()}",IssueSeverity.ERROR))
    item("data_decode","DATA Decode",service_valid and tail_valid and pad_valid,service_valid and tail_valid and pad_valid)
    fields.append(PacketField("wifi.data", "DATA logical fields", 0, len(decoded), children=(
        PacketField("wifi.data.service","SERVICE",0,16,decoded[:16],"".join(map(str,decoded[:16]))),
        PacketField("wifi.data.tail","TAIL (encoder input)",tail_start,tail_start+6,bits[tail_start:tail_start+6],
                    "".join(map(str,bits[tail_start:tail_start+6]))),
        PacketField("wifi.data.pad","PAD",tail_start+6,len(decoded),decoded[tail_start+6:],
                    f"{len(decoded)-tail_start-6} bits"))))
    item("psdu_complete","PSDU Complete",True)
    source = replace(source,stop_sample=(packet_start+stop)*factor)
    psdu_bits = decoded[16:tail_start]
    packet = WiFiMACDecoder().decode(PacketDecodeInput(psdu_bits,protocol_hint="wifi.non_ht",source=source))
    context.update(scrambler_state_msb_first=seed,psdu_hex=np.packbits(psdu_bits,bitorder="little").tobytes().hex(),
                   phy_valid=not issues)
    return replace(packet,summary=tuple(summary)+packet.summary,
                   root_fields=(PacketField("wifi.phy","PHY",0,0,children=tuple(fields)),)+packet.root_fields,
                   issues=tuple(issues)+packet.issues,decode_context=context)
