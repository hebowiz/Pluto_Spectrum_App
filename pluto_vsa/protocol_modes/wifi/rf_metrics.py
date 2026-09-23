"""OFDM measurements using IEEE Std 802.11-2024 reference procedures.

These functions return values and reference comparisons, not certification
decisions. Current-standard eligibility is handled separately in measurements.
"""
import numpy as np
from scipy.signal import welch

from . import standards as std


def relative_constellation_rms(region, pilot_errors):
    """Eq. (17-28), one packet; normalized constellation P0 = 1.

    Include 48 data and 4 known-pilot errors from each DATA symbol. L-SIG is
    excluded. Return one canonical linear RMS, with dB/% derived by consumers.
    """
    pilots = np.asarray(pilot_errors)
    if region is None or pilots.shape != (len(region.error),4) or not pilots.size:
        return None
    mse = (np.sum(abs(region.error)**2)+np.sum(abs(pilots)**2))/(52*len(pilots))
    return float(np.sqrt(mse)) if np.isfinite(mse) else None


def training_measurements(ltf_fft):
    """Pre-equalization channel-training energy, no data-FFT max/min shortcut.

    17.3.9.7.2/.3: two CFO-corrected LTF symbols; use their mean energy.
    Received channel/filter response and receiver DC remain in these values.
    """
    ltf = np.asarray(ltf_fft)
    if ltf.shape != (2,64) or not np.all(np.isfinite(ltf)):
        return {}
    carriers = np.r_[np.arange(-26,0),np.arange(1,27)]
    energies = np.mean(abs(ltf[:,carriers%64])**2,axis=0)
    inner = abs(carriers)<=16
    reference = float(np.mean(energies[inner]))
    if reference <= 0:
        return {}
    deviation = 10*np.log10(np.maximum(energies/reference,1e-30))
    lower = np.where(inner,std.REFERENCE_FLATNESS_INNER_DB[0],std.REFERENCE_FLATNESS_EDGE_DB[0])
    upper = np.full(52,std.REFERENCE_FLATNESS_INNER_DB[1])
    margin = np.minimum(deviation-lower,upper-deviation)
    dc = float(np.mean(abs(ltf[:,0])**2))
    total = float(np.sum(energies))+dc
    leakage = float(10*np.log10(max(dc/total,1e-30)))
    return dict(subcarriers=carriers,energy=energies,deviation_db=deviation,
                lower_db=lower,upper_db=upper,flatness_margin_db=float(np.min(margin)),
                flatness_reference_pass=bool(np.all(margin>=-1e-12)),
                flatness_worst_subcarrier=int(carriers[np.argmin(margin)]),
                training_power=total/64**2,leakage_db=leakage,
                leakage_relative_pass=leakage<=std.REFERENCE_LEAKAGE_LIMIT_DB+1e-12)


def compare_reference_mask(offset_hz, psd_dbm_mhz, *, amplitude_calibrated=False):
    """Compare an already resolved PSD against the 2024 *upper* mask.

    Positive margin means below mask. There is no minimum-power/lower mask;
    positive and negative frequency branches are both upper emission limits.
    Absolute -39 dBm/MHz exemption is available only with calibrated power.
    This comparison alone does not verify RBW, VBW, detector or RF bandwidth.
    """
    f,p = np.asarray(offset_hz),np.asarray(psd_dbm_mhz)
    valid = np.isfinite(f)&np.isfinite(p)
    inband = valid&(abs(f)<=9e6)
    if not np.any(inband):
        return {}
    peak = float(np.max(p[inband]))
    upper = peak+np.interp(abs(f),std.REFERENCE_MASK_OFFSETS_HZ,std.REFERENCE_MASK_LEVELS_DBR)
    if amplitude_calibrated:
        # The absolute exception is stated for >=30 MHz. Do not extend it
        # inward by interpolating a raised endpoint through the 20-30 MHz slope.
        outer = abs(f)>=30e6
        upper[outer] = np.maximum(upper[outer],std.REFERENCE_MASK_ABSOLUTE_DBM_MHZ)
    tested = valid&(abs(f)>=9e6)
    if not np.any(tested):
        return {}
    margin = upper-p
    worst = np.flatnonzero(tested)[np.argmin(margin[tested])]
    return dict(offset_hz=f,psd_dbm_mhz=p,upper_dbm_mhz=upper,
                worst_margin_db=float(margin[worst]),worst_offset_hz=float(f[worst]),
                violation_offsets_hz=f[tested&(margin < -1e-10)],
                reference_pass=bool(np.all(margin[tested]>=-1e-10)),
                full_span=bool(f[valid].min()<=-30e6 and f[valid].max()>=30e6))


def packet_spectrum(recording, start, stop, cfo_hz):
    """Equivalent digital PSD: Hann Welch, 100 kHz ENBW, linear averaging.

    30 kHz instrument VBW/detector is NOT reproduced. At 20/40 MS/s the
    +/-30 MHz mask span is unavailable, so a full-mask PASS is impossible.
    Use only active PPDU samples; never zero-pad a short packet to fake RBW.
    """
    fs = recording.sample_rate_hz
    nperseg = int(np.ceil(1.5*fs/std.REFERENCE_RBW_HZ))
    x = np.asarray(recording.iq[start:stop],dtype=complex)/recording.full_scale
    if len(x)<nperseg or cfo_hz is None:
        return {}
    x = x*np.exp(-2j*np.pi*cfo_hz*np.arange(len(x))/fs)
    f,psd = welch(x,fs=fs,window="hann",nperseg=nperseg,noverlap=nperseg//2,
                nfft=4*nperseg,detrend=False,return_onesided=False,scaling="density")
    f,psd = np.fft.fftshift(f),np.fft.fftshift(psd)
    dbm_mhz = 10*np.log10(np.maximum(psd,1e-30))+60+recording.dbfs_to_dbm_offset_db
    result = compare_reference_mask(f,dbm_mhz,amplitude_calibrated=recording.amplitude_calibrated)
    result.update(rbw_hz=1.5*fs/nperseg,vbw_reproduced=False,
                  method="Equivalent Digital Measurement: Hann Welch, linear power averaging",
                  usable_bandwidth_hz=recording.usable_bandwidth_hz or fs,
                  reference_revision=std.REFERENCE_REVISION)
    return result
