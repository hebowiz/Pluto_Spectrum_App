from dataclasses import replace
import numpy as np
import pytest

from pluto_protocol.wifi import analyze_iq
from pluto_vsg.engine.wifi_legacy_ofdm import WiFiLegacyOFDMWaveformEngine
from pluto_vsg.model import WiFiSettings
from pluto_vsg.profiles.wifi import wifi_project
from pluto_vsg.protocol import analyze_generation_result
from pluto_vsg.wifi.mac import build_psdu


@pytest.mark.parametrize('rate',[6,9,12,18,24,36,48,54])
@pytest.mark.parametrize('factor',[1,2])
def test_iq_round_trip_all_rates_and_sample_rates(rate,factor):
    settings = WiFiSettings(legacy_rate_mbps=rate,oversample_factor=factor,packet_period_us=1000)
    result = WiFiLegacyOFDMWaveformEngine().generate(wifi_project(settings))
    packet = analyze_generation_result(result)
    assert not packet.issues, packet.issues
    assert packet.integrity.crc_valid is True
    assert packet.integrity.complete
    assert packet.decode_context['rate_mbps'] == rate
    assert packet.decode_context['psdu_hex'] == build_psdu(settings).hex()
    values = {v.key:v.value for v in packet.summary}
    assert values['lsig_parity'] is True
    assert values['ssid'] == 'Pluto_Test_AP'
    assert values['bssid'] == '02:11:22:33:44:55'
    assert values['channel'] == 6


def test_verification_ignores_generator_psdu_seed_rate_and_boundaries():
    result = WiFiLegacyOFDMWaveformEngine().generate(wifi_project(WiFiSettings(packet_period_us=1000)))
    poisoned = replace(result,metadata={'phy_format':'Non-HT OFDM','psdu':b'fake','scrambler_seed':0,
        'legacy_rate_mbps':54,'sample_ranges':{},'packet_ranges_samples':((999,1000),)})
    packet = analyze_generation_result(poisoned)
    assert packet.integrity.crc_valid is True
    assert packet.decode_context['rate_mbps'] == 6


@pytest.mark.parametrize('stop',[0,80,200,350,440])
def test_truncated_packet_returns_structured_issue(stop):
    result = WiFiLegacyOFDMWaveformEngine().generate(wifi_project(WiFiSettings(oversample_factor=1,packet_period_us=1000)))
    packet = analyze_iq(result.iq[:stop],result.sample_rate_hz)
    assert packet.issues
    assert not packet.integrity.complete


def test_corrupt_data_cannot_verify_from_metadata():
    result = WiFiLegacyOFDMWaveformEngine().generate(wifi_project(WiFiSettings(oversample_factor=1,packet_period_us=1000)))
    bad = result.iq.copy()
    bad[400:int(result.metadata['packet_sample_count'])] = 0
    packet = analyze_generation_result(replace(result,iq=bad))
    assert packet.issues
    assert packet.integrity.crc_valid is not True


def test_iq_receiver_timing_cfo_and_common_complex_gain():
    result = WiFiLegacyOFDMWaveformEngine().generate(wifi_project(WiFiSettings(oversample_factor=1,packet_period_us=1000)))
    x = np.pad(result.iq[:int(result.metadata['packet_sample_count'])],(37,0))
    x = x*(0.3+0.2j)*np.exp(2j*np.pi*75000*np.arange(len(x))/20e6)
    packet = analyze_iq(x,20e6)
    assert not packet.issues, packet.issues
    assert packet.integrity.crc_valid is True
    assert abs(packet.decode_context['cfo_hz']-75000) < 1


@pytest.mark.parametrize('seed',[1,2,4,8,16,32,64,93,127])
def test_scrambler_is_recovered_from_service(seed):
    s = WiFiSettings(scrambler_seed=seed,legacy_rate_mbps=54,packet_period_us=1000)
    p = analyze_generation_result(WiFiLegacyOFDMWaveformEngine().generate(wifi_project(s)))
    assert not p.issues
    assert p.decode_context['psdu_hex']==build_psdu(s).hex()


@pytest.mark.parametrize('length,rate',[(1,9),(2,48),(7,54),(511,18),(4095,6),(4095,54)])
def test_arbitrary_psdu_lengths_tail_pad_and_pilot_wrap(length,rate):
    from pluto_vsg.model import WiFiPSDUSource
    raw = bytes((i*17+3)%256 for i in range(length))
    s = WiFiSettings(psdu_source=WiFiPSDUSource.RAW_HEX,raw_psdu_hex=raw.hex(),legacy_rate_mbps=rate,
                     oversample_factor=1,packet_period_us=6000)
    p = analyze_generation_result(WiFiLegacyOFDMWaveformEngine().generate(wifi_project(s)))
    assert p.decode_context['psdu_hex']==raw.hex()
    assert p.decode_context['phy_valid'] is True
    # Synthetic bytes are not promised to be valid MAC/FCS.


def test_wrong_manual_fcs_is_recovered_but_not_verified():
    p = analyze_generation_result(WiFiLegacyOFDMWaveformEngine().generate(wifi_project(
        WiFiSettings(fcs_auto=False,manual_fcs_hex='11223344',packet_period_us=1000))))
    assert p.decode_context['phy_valid'] is True
    assert p.integrity.crc_valid is False
    assert any(i.code=='wifi.fcs' for i in p.issues)


def test_corrupt_signal_does_not_guess_rate_or_length():
    result = WiFiLegacyOFDMWaveformEngine().generate(wifi_project(WiFiSettings(oversample_factor=1,packet_period_us=1000)))
    bad = result.iq.copy()
    bad[320:400] = 0
    p = analyze_generation_result(replace(result,iq=bad))
    assert any(i.code=='wifi.signal' for i in p.issues)
    assert not p.integrity.complete
