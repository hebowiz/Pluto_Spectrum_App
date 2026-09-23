from dataclasses import replace
import numpy as np
import pytest

from pluto_protocol.wifi.non_ht import deinterleave, depuncture, viterbi, demap, POLARITY
from pluto_vsg.engine.wifi_legacy_ofdm import WiFiLegacyOFDMWaveformEngine, map_constellation, _pilot_polarities
from pluto_vsg.model import WiFiSettings, minimum_period_symbols, validate_project
from pluto_vsg.profiles.wifi import wifi_project


@pytest.mark.parametrize('width',[1,2,4,6])
def test_normative_constellation_tables(width):
    # IEEE Tables 82-85: axis lookup is a literal published truth table.
    bits = ((np.arange(2**width)[:,None] >> np.arange(width-1,-1,-1)) & 1).astype(np.uint8)
    if width == 1:
        expected = np.array([-1,1])
    else:
        axis = np.array({2:[-1,1],4:[-3,-1,3,1],6:[-7,-5,-1,-3,7,5,1,3]}[width])
        half = 2**(width//2)
        expected = (np.repeat(axis,half)+1j*np.tile(axis,half))/np.sqrt({2:2,4:10,6:42}[width])
    np.testing.assert_allclose(map_constellation(bits.ravel(),width),expected)
    np.testing.assert_allclose(np.mean(abs(expected)**2),1)
    np.testing.assert_array_equal(demap(expected,width)>0,bits.ravel())


def test_pilot_sequence_full_period_and_wrap():
    assert len(POLARITY)==127
    np.testing.assert_array_equal(_pilot_polarities(381),np.tile(POLARITY,3))


@pytest.mark.parametrize('factor',[1,2])
def test_erp_extension_is_silence_and_minimum_repetition_spacing(factor):
    engine = WiFiLegacyOFDMWaveformEngine()
    base = wifi_project(WiFiSettings(oversample_factor=factor,packet_period_us=1000))
    result = engine.generate(base)
    ppdu_us = result.metadata['ppdu_duration_us']
    exact = wifi_project(replace(base.wifi,packet_period_us=ppdu_us+6))
    repeated = engine.generate(replace(exact,repeat_count=2))
    packet = int(result.metadata['packet_sample_count'])
    period = int(repeated.metadata['period_sample_count'])
    assert period-packet==120*factor
    assert not np.any(repeated.iq[packet:period])
    np.testing.assert_array_equal(repeated.iq[:packet],repeated.iq[period:period+packet])
    assert minimum_period_symbols(exact)*4 == ppdu_us+6
    bad = wifi_project(replace(base.wifi,packet_period_us=ppdu_us+5.9))
    assert validate_project(bad)
    with pytest.raises(ValueError):
        engine.generate(bad)


def test_large_diagnostics_are_opt_in():
    project = wifi_project(WiFiSettings(packet_period_us=1000))
    engine = WiFiLegacyOFDMWaveformEngine()
    assert 'scrambled_data_bits' not in engine.generate(project).metadata
    debug = engine.generate(project,diagnostics=True).metadata
    assert debug['bcc_bits'].size == debug['scrambled_data_bits'].size*2
    assert len(debug['interleaved_bits_per_symbol']) == debug['n_sym']
