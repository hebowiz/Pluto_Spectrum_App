"""Capture-wide STF candidates, independent of any waveform generator."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PacketCandidate:
    start_sample: int
    confidence: float
    coarse_cfo_hz: float


def detect_packets(iq, sample_rate_hz, *, threshold=0.75, max_candidates=256):
    if sample_rate_hz not in (20e6, 40e6):
        raise ValueError("Non-HT detection requires 20 or 40 MS/s")
    factor = int(sample_rate_hz/20e6)
    x = np.asarray(iq)[::factor].astype(np.complex128)
    if x.ndim != 1 or not np.all(np.isfinite(x)):
        raise ValueError("Capture IQ must be finite and one dimensional")
    if len(x) < 112:
        return ()

    def window_sum(values, count=96):
        total = np.r_[0, np.cumsum(values)]
        return total[count:]-total[:-count]

    correlation = window_sum(np.conj(x[:-16])*x[16:])
    p0 = window_sum(abs(x[:-16])**2)
    p1 = window_sum(abs(x[16:])**2)
    metric = abs(correlation)/np.maximum(np.sqrt(np.maximum(p0*p1,0)),1e-20)
    active = (metric >= threshold) & (np.minimum(p0,p1) > 1e-18)
    edges = np.diff(np.r_[False, active, False].astype(int))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    result = []
    for start, stop in zip(starts, stops):
        if stop-start < 24:
            continue
        peak = start+int(np.argmax(metric[start:stop]))
        result.append(PacketCandidate(int(start*factor), float(metric[peak]),
                                      float(np.angle(correlation[peak])*20e6/(2*np.pi*16))))
        if len(result) >= max_candidates:
            break
    return tuple(result)
