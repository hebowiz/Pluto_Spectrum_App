"""Post-equalization, decision-directed EVM; never raw time-domain I/Q EVM."""
import numpy as np
from .model import OFDMRegion


def packet_power(recording, start_time_s, stop_time_s):
    start = max(0, int(round(start_time_s*recording.sample_rate_hz)))
    stop = min(recording.sample_count, int(round(stop_time_s*recording.sample_rate_hz)))
    powers = abs(np.asarray(recording.iq[start:stop],dtype=complex)/recording.full_scale)**2
    if not powers.size:
        return float("nan"), float("nan")
    offset = recording.dbfs_to_dbm_offset_db
    return (float(10*np.log10(max(float(np.mean(powers)),1e-30))+offset),
            float(10*np.log10(max(float(np.max(powers)),1e-30))+offset))


def constellation(modulation):
    if modulation == "BPSK":
        return np.array([-1, 1], dtype=complex)
    levels, norm = {"QPSK": ([-1, 1], 2), "16QAM": ([-3, -1, 1, 3], 10),
                    "64QAM": ([-7, -5, -3, -1, 1, 3, 5, 7], 42)}[modulation]
    return np.array([i+1j*q for i in levels for q in levels])/np.sqrt(norm)


def measure_region(name, modulation, symbols, pilot_errors):
    """48 data tones, nominal constellation mean power=1; pilots separate.

    No per-packet amplitude fit and no reference from the transmitter. These
    single-packet diagnostics are not a 52-tone multi-frame conformance test.
    """
    measured = np.array(symbols, dtype=complex).reshape(-1, 48)
    ideal = constellation(modulation)
    nearest = np.argmin(abs(measured[..., None]-ideal)**2, axis=-1)
    reference = ideal[nearest]
    error = measured-reference
    squared = abs(error)**2
    return OFDMRegion(name, modulation, measured, reference, error,
        float(100*np.sqrt(np.mean(squared))), float(100*np.sqrt(np.max(squared))),
        100*np.sqrt(np.mean(squared,axis=0)), 100*np.sqrt(np.mean(squared,axis=1)),
        float(100*np.sqrt(np.mean(abs(np.asarray(pilot_errors))**2))))
