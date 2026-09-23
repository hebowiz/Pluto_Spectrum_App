"""IEEE 2024 measurement algorithms, boundaries and observation conditions."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.signal import resample_poly, firwin, fftconvolve

from pluto_vsa.model import IQRecording
from pluto_vsa.protocol_modes.wifi.analysis import analyze_wifi_recording
from pluto_vsa.protocol_modes.wifi.measurement import measure_region
from pluto_vsa.protocol_modes.wifi.measurements import capture_statistics, measurement_results, MeasurementConditions
from pluto_vsa.protocol_modes.wifi.results import MeasurementKind as Kind, MeasurementStatus as Status
from pluto_vsa.protocol_modes.wifi.rf_metrics import relative_constellation_rms, training_measurements, compare_reference_mask, packet_spectrum
from pluto_vsa.protocol_modes.wifi.standards import reference_profile
from pluto_vsa.protocol_modes.wifi.summary import visible_results
from pluto_vsg.engine.wifi_legacy_ofdm import WiFiLegacyOFDMWaveformEngine
from pluto_vsg.model import WiFiSettings, WiFiPSDUSource
from pluto_vsg.profiles.wifi import wifi_project


def waveform(rate=54, **changes):
    settings = WiFiSettings(legacy_rate_mbps=rate,oversample_factor=2,packet_period_us=1500,
                           psdu_source=WiFiPSDUSource.PRBS9,payload_length_bytes=600,**changes)
    return WiFiLegacyOFDMWaveformEngine().generate(wifi_project(settings))


def results(iq,fs=40e6,center=2437e6):
    recording = IQRecording(iq,fs,center)
    capture = analyze_wifi_recording(recording)
    assert capture.packets
    return recording,capture,capture.packets[0],{m.measurement_id:m for m in capture.packets[0].measurements}


@pytest.mark.parametrize('rate,limit',[(6,-5),(9,-8),(12,-10),(18,-13),(24,-16),(36,-19),(48,-22),(54,-25)])
def test_all_rates_choose_current_limits_but_require_observation_conditions(rate,limit):
    wave = waveform(rate)
    _,_,p,m = results(wave.iq)
    assert m['relative_constellation_error'].metadata['reference_limit']==limit
    assert m['relative_constellation_error'].limit==f'≤ {limit} dB'
    assert m['relative_constellation_error'].metadata['rate_mbps']==rate
    assert m['relative_constellation_error'].status==Status.INSUFFICIENT_DATA
    assert m['evm_rms'].value<.0001
    assert m['evm_rms'].canonical_id=='relative_constellation_error'
    assert m['evm_rms'].value==pytest.approx(100*10**(m['relative_constellation_error'].value/20))
    assert all(x.status not in (Status.PASS,Status.FAIL) for x in m.values() if x.measurement_kind==Kind.STANDARD)
    assert p.data.modulation=={6:'BPSK',9:'BPSK',12:'QPSK',18:'QPSK',24:'16QAM',36:'16QAM',48:'64QAM',54:'64QAM'}[rate]


@pytest.mark.parametrize('center,tolerance',[(2437e6,25),(5180e6,20),(915e6,None)])
def test_phy_band_profiles_do_not_guess_unknown_band(center,tolerance):
    _,_,_,m = results(waveform().iq,center=center)
    assert m['carrier_frequency_error'].metadata['reference_limit']==tolerance
    assert not m['carrier_frequency_error'].measurement_conditions_satisfied
    assert 'Unverified' not in m['carrier_frequency_error'].limit
    assert '802.11-2024' in m['carrier_frequency_error'].standard_reference
    if tolerance is not None:
        assert m['carrier_frequency_error'].limit==f'±{tolerance} ppm'
    if tolerance is None:
        assert reference_profile(center) is None


def test_gain_phase_cfo_multipath_correction_and_deliberate_distortion():
    wave = waveform()
    x = np.convolve(wave.iq,[1,0,0,.15+.07j])[:len(wave.iq)]
    x = x*.16*np.exp(1j*(.83+2*np.pi*37000*np.arange(len(x))/wave.sample_rate_hz))
    _,_,_,m = results(x)
    assert m['evm_rms'].value<.001
    assert m['carrier_frequency_error'].value==pytest.approx(37000,abs=.1)
    distorted = x.copy()
    # Distort only DATA, preserving channel training and synchronization.
    wave_packet = results(wave.iq)[2]
    a,b = wave_packet.start_sample+800,wave_packet.stop_sample
    distorted[a:b] += .002*np.random.default_rng(301).standard_normal(b-a)
    _,_,_,bad = results(distorted)
    assert bad['evm_rms'].value>m['evm_rms'].value+1


def test_canonical_error_includes_pilots_but_not_lsig_and_uses_nominal_power():
    ideal = np.tile([-1.,1.],24).reshape(1,48)
    region = measure_region('DATA','BPSK',1.1*ideal,np.full((1,4),.03))
    rms = relative_constellation_rms(region,np.full((1,4),.03))
    assert rms==pytest.approx(np.sqrt((48*.1**2+4*.03**2)/52))
    assert relative_constellation_rms(region,np.zeros((2,4))) is None


def test_packet_statistics_weight_each_packet_and_do_not_promote_count_to_pass():
    rec,_,p,_ = results(waveform().iq)
    packets = tuple(replace(p,rf_details={**p.rf_details,'constellation_rms':r}) for r in ([.01]*10+[.03]*10))
    stats = capture_statistics(packets)
    assert stats[54]['packet_count']==20
    assert stats[54]['rms']==pytest.approx(.02)  # printed 2024 Eq. (17-28)
    m = {x.measurement_id:x for x in measurement_results(p,rec,stats)}
    assert m['relative_constellation_error'].status==Status.NOT_MEASURED
    assert not m['relative_constellation_error'].measurement_conditions_satisfied
    # A good FCS is neither required nor sufficient for the RF decision.
    bad_mac = replace(p,packet=replace(p.packet,integrity=replace(p.packet.integrity,crc_valid=False)))
    assert capture_statistics((bad_mac,))[54]['packet_count']==1


def test_mixed_rates_short_frames_and_missing_fcs_are_separate():
    first,second = waveform(24),waveform(54)
    rec,capture,_,_ = results(np.r_[first.iq,second.iq])
    assert set(capture.measurement_statistics)=={24,54}
    assert all(v['packet_count']==1 for v in capture.measurement_statistics.values())
    short = WiFiLegacyOFDMWaveformEngine().generate(wifi_project(WiFiSettings(legacy_rate_mbps=54)))
    _,capture,p,m = results(short.iq)
    assert len(p.data.measured)<16
    assert not capture.measurement_statistics
    assert m['relative_constellation_error'].status==Status.INSUFFICIENT_DATA
    _,_,p,m = results(first.iq[:500])
    assert m['fcs'].status==Status.NOT_AVAILABLE
    assert m['fcs'].value is None
    assert m['relative_constellation_error'].value is None


@pytest.mark.parametrize('carrier,db,passed',[(16,-4,True),(16,-4.01,False),(16,4,True),(16,4.01,False),
                                         (17,-6,True),(17,-6.01,False),(26,4,True),(26,4.01,False)])
def test_training_flatness_inner_edge_boundaries(carrier,db,passed):
    ltf = np.ones((2,64),complex)
    ltf[:,0] = 0
    ltf[:,carrier] = 10**(db/20)
    if abs(carrier)<=16:
        others = [k%64 for k in range(-16,17) if k not in (0,carrier)]
        ltf[:,others] = np.sqrt((32-10**(db/10))/31)
    measured = training_measurements(ltf)
    assert measured['flatness_reference_pass']==passed
    i = list(measured['subcarriers']).index(carrier)
    assert measured['deviation_db'][i]==pytest.approx(db)


@pytest.mark.parametrize('db,passed',[(-100,True),(-15.001,True),(-15,True),(-14.999,False)])
def test_training_leakage_relative_to_total_power(db,passed):
    ltf = np.ones((2,64),complex)
    ratio = 10**(db/10)
    ltf[:,0] = np.sqrt(52*ratio/(1-ratio))
    measured = training_measurements(ltf)
    assert measured['leakage_db']==pytest.approx(db)
    assert measured['leakage_relative_pass']==passed


def test_waveform_leakage_without_reading_an_arbitrary_dc_fft_bin():
    wave = waveform()
    _,_,p,_ = results(wave.iq)
    raw = wave.iq.astype(complex).copy()
    # Use the active OFDM training mean power to add known -15 dB leakage.
    a = p.start_sample+384
    signal_power = float(np.mean(abs(raw[a:a+256])**2))
    db = -14.8
    ratio = 10**(db/10)
    raw[p.start_sample:p.stop_sample] += np.sqrt(signal_power*ratio/(1-ratio))
    _,_,_,m = results(raw)
    assert m['center_frequency_leakage'].value==pytest.approx(db,abs=.03)
    assert m['center_frequency_leakage'].status==Status.NOT_MEASURED


def test_reference_mask_known_spectrum_violation_location_and_absolute_floor():
    f = np.array([-35,-30,-20,-11,-9,0,9,11,20,30,35])*1e6
    psd = np.array([-45,-45,-33,-25,-5,0,-5,-25,-33,-45,-45],float)
    ok = compare_reference_mask(f,psd)
    assert ok['reference_pass'] and ok['full_span']
    assert ok['worst_margin_db']==pytest.approx(5)
    psd[7] = -19
    bad = compare_reference_mask(f,psd)
    assert not bad['reference_pass']
    assert bad['worst_offset_hz']==11e6
    assert bad['worst_margin_db']==pytest.approx(-1)
    np.testing.assert_array_equal(bad['violation_offsets_hz'],[11e6])
    floor = compare_reference_mask(np.array([-30e6,0,25e6,30e6]),np.array([-54,-30,-65,-54]),amplitude_calibrated=True)
    assert floor['reference_pass']
    assert floor['upper_dbm_mhz'][0]==pytest.approx(-39)
    assert floor['upper_dbm_mhz'][2]==pytest.approx(-64)


def test_equivalent_digital_mask_detects_added_out_of_band_tone():
    # Wider synthetic acquisition exercises the reference algorithm; the
    # production 20/40 MS/s analyzer still cannot cover the complete mask.
    fs = 80e6
    rng = np.random.default_rng(45)
    noise = rng.normal(size=80000)+1j*rng.normal(size=80000)
    x = fftconvolve(noise,firwin(601,8e6,fs=fs),mode='same')
    base = packet_spectrum(IQRecording(x,fs),0,len(x),0.)
    assert base['reference_pass'] and base['full_span']
    x += .05*np.exp(2j*np.pi*22e6*np.arange(len(x))/fs)
    bad = packet_spectrum(IQRecording(x,fs),0,len(x),0.)
    assert not bad['reference_pass']
    assert bad['worst_offset_hz']==pytest.approx(22e6,abs=100e3)
    assert bad['worst_margin_db']<0


def test_mask_conditions_and_decode_diagnostic_classification():
    rec,_,p,m = results(waveform().iq)
    mask = m['transmit_spectrum_mask']
    assert mask.status==Status.INSUFFICIENT_DATA
    assert mask.metadata['rbw_hz']==pytest.approx(100e3)
    assert not mask.metadata['vbw_reproduced']
    assert not mask.metadata['full_span']
    assert m['data_rate'].measurement_kind==Kind.PHY_DECODE
    assert m['fcs'].measurement_kind==Kind.MAC_DECODE
    assert m['peak_power'].measurement_kind==Kind.DIAGNOSTIC
    assert m['data_evm_peak'].measurement_kind==Kind.DIAGNOSTIC
    assert m['data_pilot_error'].limit=='—'
    assert m['packet_power'].status==Status.INFO and m['packet_power'].limit=='—'
    assert all(x.measurement_kind!=Kind.DIAGNOSTIC for x in visible_results(p,rec))
    assert any(x.measurement_kind==Kind.DIAGNOSTIC for x in visible_results(p,rec,diagnostics=True))


def test_known_sample_clock_offset_is_explicitly_not_measured():
    # +100 ppm rate error: no substitution of carrier CFO for sample clock.
    wave = waveform()
    x = resample_poly(wave.iq,10001,10000)
    _,_,_,m = results(x)
    assert m['symbol_clock_error'].value is None
    assert m['symbol_clock_error'].status==Status.NOT_MEASURED


CONFIRMED = MeasurementConditions(True,True,True,True)


def measured(p,rec,stats=None,conditions=CONFIRMED):
    return {x.measurement_id:x for x in measurement_results(p,rec,stats or {},conditions=conditions)}


@pytest.mark.parametrize('center,ppm,expected',[(2437e6,25,Status.PASS),(2437e6,-25,Status.PASS),
    (2437e6,25.001,Status.FAIL),(2437e6,-25.001,Status.FAIL),(5180e6,20,Status.PASS),
    (5180e6,20.001,Status.FAIL),(915e6,0,Status.NOT_MEASURED)])
def test_current_cfo_limits_apply_only_with_frequency_reference(center,ppm,expected):
    rec,_,p,_ = results(waveform().iq,center=center)
    p = replace(p,cfo_hz=ppm*center/1e6)
    row = measured(p,rec)['carrier_frequency_error']
    assert row.status==expected
    assert row.measurement_conditions_satisfied==(expected in (Status.PASS,Status.FAIL))
    assert measured(p,rec,conditions=replace(CONFIRMED,frequency_reference_verified=False))['carrier_frequency_error'].status==Status.NOT_MEASURED


@pytest.mark.parametrize('rate,limit',[(6,-5),(9,-8),(12,-10),(18,-13),(24,-16),(36,-19),(48,-22),(54,-25)])
def test_current_rce_boundary_and_confirmation_gates(rate,limit):
    rec,_,p,_ = results(waveform(rate).iq)
    for db,expected in [(limit,Status.PASS),(limit+.001,Status.FAIL)]:
        stats = {rate:dict(packet_count=20,rms=10**(db/20))}
        row = measured(p,rec,stats)['relative_constellation_error']
        assert row.status==expected
        assert row.measurement_conditions_satisfied
        assert 'Table 17-20' in row.standard_reference
        for condition in [replace(CONFIRMED,random_payload_verified=False),replace(CONFIRMED,receiver_response_verified=False)]:
            assert measured(p,rec,stats,condition)['relative_constellation_error'].status==Status.NOT_MEASURED
        stats[rate]['packet_count']=19
        assert measured(p,rec,stats)['relative_constellation_error'].status==Status.INSUFFICIENT_DATA


def test_twenty_real_ppdus_qualify_without_fcs_and_refresh_does_not_accumulate():
    wave = waveform()
    recording = IQRecording(np.tile(wave.iq,20),wave.sample_rate_hz,2437e6)
    capture = analyze_wifi_recording(recording,measurement_conditions=CONFIRMED)
    assert len(capture.packets)==20
    assert capture.measurement_statistics[54]['packet_count']==20
    for p in capture.packets:
        row = next(x for x in p.measurements if x.measurement_id=='relative_constellation_error')
        assert row.status==Status.PASS
        assert row.value < -80
    assert capture_statistics(capture.packets)==capture.measurement_statistics


@pytest.mark.parametrize('power_dbm,leakage_db,status',[(0,-15,Status.PASS),(0,-14.99,Status.FAIL),
    (-10,-10,Status.PASS),(-10,-9.99,Status.FAIL),(-25,-1,Status.PASS)])
def test_current_leakage_absolute_exception_and_dut_gate(power_dbm,leakage_db,status):
    rec,_,p,_ = results(waveform().iq)
    # Training energy measured at the calibrated transmitter reference plane.
    training = dict(p.rf_details['training'],training_power=10**(power_dbm/10),leakage_db=leakage_db)
    p = replace(p,rf_details={**p.rf_details,'training':training})
    rec = replace(rec,amplitude_calibrated=True)
    row = measured(p,rec)['center_frequency_leakage']
    assert row.status==status
    assert row.metadata['leakage_dbm']==pytest.approx(power_dbm+leakage_db)
    assert measured(p,rec,conditions=replace(CONFIRMED,non_vht_dut=False))['center_frequency_leakage'].status==Status.NOT_MEASURED
    uncalibrated = measured(p,replace(rec,amplitude_calibrated=False))['center_frequency_leakage']
    assert uncalibrated.status==(Status.PASS if leakage_db<=-15 else Status.NOT_MEASURED)


@pytest.mark.parametrize('margin,status',[(0,Status.PASS),(-.001,Status.FAIL),(1,Status.PASS)])
def test_flatness_decision_requires_characterized_path_and_full_active_band(margin,status):
    rec,_,p,_ = results(waveform().iq)
    p = replace(p,rf_details={**p.rf_details,'training':dict(p.rf_details['training'],flatness_margin_db=margin)})
    assert measured(p,rec)['spectral_flatness'].status==status
    assert measured(p,rec,conditions=replace(CONFIRMED,receiver_response_verified=False))['spectral_flatness'].status==Status.NOT_MEASURED
    assert measured(p,replace(rec,usable_bandwidth_hz=15e6))['spectral_flatness'].status==Status.NOT_MEASURED


def test_mask_current_absolute_floor_is_not_the_old_revision_limit():
    f = np.array([-35e6,0,35e6])
    psd = np.array([-40,-20,-40])
    calibrated = compare_reference_mask(f,psd,amplitude_calibrated=True)
    assert calibrated['reference_pass']
    np.testing.assert_allclose(calibrated['upper_dbm_mhz'][[0,2]],-39)
    assert not compare_reference_mask(f,psd)['reference_pass']
    rec,_,p,_ = results(waveform().iq)
    assert measured(p,rec)['transmit_spectrum_mask'].status==Status.INSUFFICIENT_DATA


def test_current_erp_references_and_unknown_band_fail_closed():
    assert reference_profile(2437e6).frequency_clause=='18.4.7.4'
    assert reference_profile(2437e6).clock_clause=='18.4.7.5'
    assert reference_profile(5180e6).frequency_clause=='17.3.9.5'
    rec,_,p,_ = results(waveform().iq,center=915e6)
    rows = measured(p,rec,{54:dict(packet_count=20,rms=.001)})
    assert all(x.status not in (Status.PASS,Status.FAIL) for x in rows.values() if x.measurement_kind==Kind.STANDARD)
