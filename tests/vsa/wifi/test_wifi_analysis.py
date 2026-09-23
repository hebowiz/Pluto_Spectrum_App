"""Digital integration uses only IQ across the VSG/VSA boundary."""
from dataclasses import replace
import numpy as np
import pytest
from pluto_vsa.model import IQRecording
from pluto_vsa.protocol_modes.wifi.analysis import analyze_wifi_recording
from pluto_vsa.protocol_modes.wifi.measurement import measure_region
from pluto_vsg.engine.wifi_legacy_ofdm import WiFiLegacyOFDMWaveformEngine
from pluto_vsg.model import WiFiSettings
from pluto_vsg.profiles.wifi import wifi_project
from pluto_vsg.wifi.mac import build_psdu


def generated(rate=6, factor=1, **kwargs):
    settings = WiFiSettings(legacy_rate_mbps=rate,oversample_factor=factor,packet_period_us=1000,**kwargs)
    waveform = WiFiLegacyOFDMWaveformEngine().generate(wifi_project(settings))
    return waveform, settings


@pytest.mark.parametrize("rate",[6,9,12,18,24,36,48,54])
@pytest.mark.parametrize("factor",[1,2])
def test_all_rates_and_sample_rates(rate,factor):
    wave, settings = generated(rate,factor)
    result = analyze_wifi_recording(IQRecording(wave.iq,wave.sample_rate_hz,2437e6))
    assert result.counts == dict(detected=1,complete=1,measurement_eligible=1,decode_success=1,fcs_valid=1)
    p = result.packets[0]
    assert p.packet.decode_context['psdu_hex'] == build_psdu(settings).hex()
    assert p.packet.decode_context['rate_mbps'] == rate
    assert p.signal.modulation == 'BPSK'
    assert p.data.evm_rms_percent < .0001
    assert p.data.evm_peak_percent < .001
    assert p.channel.shape == (52,)
    assert all(p.integrity.values())
    assert p.symbol_clock_error_ppm is None


@pytest.mark.parametrize("rate",[6,12,24,54])
@pytest.mark.parametrize("factor",[1,2])
def test_noise_fractional_timing_cfo_multipath_and_multiple_packets(rate,factor):
    wave, settings = generated(rate,factor)
    rng = np.random.default_rng(102)
    x = np.r_[np.zeros(901),wave.iq,wave.iq,np.zeros(100)]
    # Fractional sample delay by a two-tap interpolator plus an independent
    # delayed path. Delay spread is within the 0.8 us guard interval.
    x = np.convolve(x,[.7,.3])[:len(x)]
    impulse = np.zeros(4*factor+1,dtype=complex)
    impulse[0],impulse[-1] = 1,.12+.08j
    x = np.convolve(x,impulse)[:len(x)]
    x *= .23*np.exp(1j*(1.2+2*np.pi*85000*np.arange(len(x))/wave.sample_rate_hz))
    x += .00035*(rng.normal(size=len(x))+1j*rng.normal(size=len(x)))
    result = analyze_wifi_recording(IQRecording(x,wave.sample_rate_hz,2437e6))
    assert result.counts['fcs_valid'] == 2, result
    for p in result.packets:
        assert p.packet.decode_context['psdu_hex'] == build_psdu(settings).hex()
        assert abs(p.cfo_hz-85000) < 600
        assert 0 < p.data.evm_rms_percent < 5
        assert np.ptp(abs(p.channel)) > .02
        assert p.data.evm_per_subcarrier.shape == (48,)


@pytest.mark.parametrize("cut",[100,230,350,500])
def test_truncated_capture_returns_partial_or_no_detection_issue(cut):
    wave,_ = generated()
    result = analyze_wifi_recording(IQRecording(wave.iq[:cut],20e6))
    assert result.issues or result.packets
    for p in result.packets:
        assert p.packet.issues
        assert not p.integrity['psdu_complete']
        assert p.packet.integrity.crc_valid is None


def test_bad_fcs_and_unknown_ie_are_not_phy_failure():
    wave,_ = generated(fcs_auto=False,manual_fcs_hex='12345678')
    p = analyze_wifi_recording(IQRecording(wave.iq,20e6)).packets[0]
    assert p.integrity['data_complete']
    assert p.integrity['fcs_valid'] is False
    assert any(i.code=='wifi.fcs' for i in p.packet.issues)


@pytest.mark.parametrize("kind",['zero','noise','tone'])
def test_false_detection_rejected(kind):
    rng = np.random.default_rng(4)
    x = {'zero':np.zeros(4000,dtype=complex),'noise':rng.normal(size=4000)+1j*rng.normal(size=4000),
         'tone':np.exp(1j*np.arange(4000)*.13)}[kind]
    result = analyze_wifi_recording(IQRecording(x,20e6))
    assert not result.packets
    assert result.issues


def test_evm_normalization_and_pilot_measurement_are_explicit():
    # Known radial +10% gain error: no hidden amplitude fitting allowed.
    ideal = np.tile(np.array([-1,1]),24).reshape(1,48)
    r = measure_region('DATA','BPSK',ideal*1.1,np.ones((1,4))*.03)
    assert r.evm_rms_percent == pytest.approx(10)
    assert r.evm_peak_percent == pytest.approx(10)
    assert r.pilot_error_rms_percent == pytest.approx(3)


def test_power_uses_active_packet_and_recording_corrections():
    wave,_ = generated()
    base = IQRecording(wave.iq,20e6,full_scale=2,calibration_offset_db=-10,input_correction_db=7)
    p = analyze_wifi_recording(base).packets[0]
    active = base.iq[p.start_sample:p.stop_sample]
    assert p.packet_power_dbm == pytest.approx(10*np.log10(np.mean(abs(active.astype(complex))**2))-3)
    assert p.peak_power_dbm > p.packet_power_dbm


@pytest.mark.parametrize('corruption',['parity','rate'])
def test_invalid_lsig_returns_fields_and_issue(corruption):
    from pluto_vsg.engine.wifi_legacy_ofdm import _l_sig_bits, bcc_encode, interleave, map_constellation, _ifft_symbol
    from pluto_vsg.wifi.common import LEGACY_RATES
    wave,settings = generated()
    bits = _l_sig_bits(LEGACY_RATES[6],len(build_psdu(settings)))
    if corruption=='parity':
        bits[17] ^= 1
    else:
        bits[:4] = 0
        bits[17] = np.sum(bits[:17]) % 2
    symbol = _ifft_symbol(map_constellation(interleave(bcc_encode(bits),48,1),1),0,1)
    # Match the TX's common amplitude normalization without importing any RX logic.
    scale = np.sqrt(np.mean(abs(wave.iq[320:400])**2)/np.mean(abs(symbol)**2))
    x = wave.iq.copy()
    x[320:400] = symbol*scale
    result = analyze_wifi_recording(IQRecording(x,20e6))
    assert len(result.packets)==1
    p = result.packets[0]
    assert p.integrity['lsig_complete']
    assert not p.integrity['data_complete']
    assert any(i.code=='wifi.signal' for i in p.packet.issues)
    assert 'lsig_bits' in p.packet.decode_context


def test_unknown_ie_and_mac_control_flags_remain_available():
    import binascii
    from pluto_protocol.wifi.mac import WiFiMACDecoder
    from pluto_protocol.model import PacketDecodeInput
    frame = build_psdu(WiFiSettings(sequence_number=25,fragment_number=3,frame_control=0x2880))[:-4]
    frame += bytes.fromhex('dd03010203')
    frame += binascii.crc32(frame).to_bytes(4,'little')
    p = WiFiMACDecoder().decode(PacketDecodeInput(np.unpackbits(np.frombuffer(frame,dtype=np.uint8),bitorder='little')))
    def walk(nodes):
        for node in nodes:
            yield node.field_id,node
            yield from walk(node.children)
    fields = dict(walk(p.root_fields))
    assert p.integrity.crc_valid
    assert fields['wifi.ie.221.id'].value==221
    assert fields['wifi.ie.221.raw'].raw_bits.size==24
    assert fields['wifi.retry'].value==1
    assert fields['wifi.more_data'].value==1
    assert fields['wifi.sequence_number'].value==25
    assert fields['wifi.fragment_number'].value==3


def test_vsa_wifi_modules_do_not_import_vsg():
    import ast
    from pathlib import Path
    for path in (Path(__file__).parents[3]/'pluto_vsa/protocol_modes/wifi').glob('*.py'):
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if isinstance(node,ast.ImportFrom):
                assert not (node.module or '').startswith('pluto_vsg'), path
            if isinstance(node,ast.Import):
                assert all(not item.name.startswith('pluto_vsg') for item in node.names), path
