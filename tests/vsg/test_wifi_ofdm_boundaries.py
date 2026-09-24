"""Rectangular Non-HT construction against IEEE 802.11-2024 Fourier sums.

The reference does not call the generator's IFFT, CP, training or pilot helpers.
DATA constellation traces are inputs here; coding/mapping have separate IEEE
fixed-vector tests. Abrupt adjacent sample values are valid for rectangular
subfields: continuity must not be imposed by an unrequested smoothing window.
"""
from dataclasses import replace
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import pytest

from pluto_protocol.wifi.non_ht import analyze_iq
from pluto_vsg.engine.wifi_legacy_ofdm import WiFiLegacyOFDMWaveformEngine
from pluto_vsg.model import PowerEnvelopeDefinition, WiFiPSDUSource, WiFiSettings
from pluto_vsg.profiles.wifi import wifi_project


RATES = (6, 9, 12, 18, 24, 36, 48, 54)
# IEEE 802.11-2024 17.3.3, equations (17-6) and (17-8).
SHORT_CARRIERS = (-24, -20, -16, -12, -8, -4, 4, 8, 12, 16, 20, 24)
SHORT_SIGNS = np.array([1, -1, 1, -1, -1, 1, -1, -1, 1, 1, 1, 1])
LONG = np.array([
    1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1,
    1, -1, 1, -1, 1, 1, 1, 1, 0, 1, -1, -1, 1, 1, -1, 1, -1, 1,
    -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1,
])
# First 32 terms of (17-25), sufficient for the default Beacon at every rate.
POLARITIES = (1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, -1, 1,
              -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1, 1, 1, -1, 1)


@lru_cache(None)
def fourier_kernel(carriers, duration_us, guard_us, factor):
    # Sample the analytic definition on a half-open time interval. No IFFT,
    # prefix copying, overlapping endpoints, or window multiplication.
    time_us = np.arange(round(duration_us * 20 * factor)) / (20 * factor)
    return np.exp(2j * np.pi * np.outer(time_us - guard_us, carriers) * 0.3125) / np.sqrt(52)


def reference_ppdu(result, factor):
    short = fourier_kernel(SHORT_CARRIERS, 8, 0, factor) @ (SHORT_SIGNS * (1+1j) * np.sqrt(13/6))
    long = fourier_kernel(tuple(range(-26, 27)), 8, 1.6, factor) @ LONG
    # Independent explicit subcarrier assignment, not DATA_SUBCARRIERS import.
    data_carriers = tuple(k for k in range(-26, 27) if k not in (-21, -7, 0, 7, 21))
    carriers = data_carriers + (-21, -7, 7, 21)
    data = result.constellation_traces[1].symbols.reshape(-1, 48)
    rows = [result.constellation_traces[0].symbols, *data]
    assert len(rows) <= len(POLARITIES)
    symbols = [fourier_kernel(carriers, 4, 0.8, factor) @ np.r_[row, np.array([1, 1, 1, -1])*POLARITIES[n]]
               for n, row in enumerate(rows)]
    return np.concatenate((short, long, *symbols))


@pytest.fixture(scope='module', params=[(r, f) for r in RATES for f in (1, 2)])
def waveform(request):
    rate, factor = request.param
    project = wifi_project(WiFiSettings(legacy_rate_mbps=rate, oversample_factor=factor, packet_period_us=1000))
    return WiFiLegacyOFDMWaveformEngine().generate(project), factor


def test_field_lengths_prefix_copies_and_exclusive_boundaries(waveform):
    result, factor = waveform
    n_sym = result.constellation_traces[1].symbols.size // 48
    expected = [('L-STF', 0, 160), ('L-LTF', 160, 320), ('L-SIG', 320, 400), ('DATA', 400, 400+80*n_sym)]
    assert [(b.name, b.start_sample, b.stop_sample) for b in result.field_boundaries] == [
        (name, start*factor, stop*factor) for name, start, stop in expected]
    assert result.metadata['packet_sample_count'] == (400+80*n_sym)*factor
    assert result.metadata['ppdu_duration_us'] == pytest.approx(20+4*n_sym)
    # Both L-SIG and every DATA symbol have exactly 0.8 us CP + 3.2 us useful.
    symbols = result.iq[320*factor:(400+80*n_sym)*factor].reshape(-1, 80*factor)
    np.testing.assert_array_equal(symbols[:, :16*factor], symbols[:, 64*factor:])
    ltf = result.iq[160*factor:320*factor]
    np.testing.assert_array_equal(ltf[:32*factor], ltf[64*factor:96*factor])
    np.testing.assert_array_equal(ltf[32*factor:96*factor], ltf[96*factor:])
    stf = result.iq[:160*factor].reshape(10, 16*factor)
    np.testing.assert_array_equal(stf, np.tile(stf[0], (10, 1)))
    stop = (400+80*n_sym)*factor
    assert not np.any(result.iq[stop:stop+120*factor])  # ERP extension stays silent.


def test_full_ppdu_direct_fourier_reference_and_spectrum(waveform):
    result, factor = waveform
    expected = reference_ppdu(result, factor)
    actual = result.iq[:expected.size].astype(np.complex128)
    # Permit only a single real amplitude scale, never phase/sample alignment.
    gain = np.vdot(expected, actual) / np.vdot(expected, expected)
    assert gain.real > 0 and abs(gain.imag) < 1e-8
    expected *= gain.real
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=6e-8)
    # Include both sides of every join; no skipped half-weight endpoint sample.
    joins = [160*factor, 320*factor, *range(400*factor, expected.size, 80*factor)]
    for boundary in joins:
        np.testing.assert_allclose(actual[boundary-1:boundary+2], expected[boundary-1:boundary+2], atol=6e-8)
    # Digital spectrum regression against the independent rectangular reference.
    # This is not an RF mask measurement (finite Fs/RBW and no DAC/RF filtering).
    nfft = 16384
    reference_psd = abs(np.fft.fft(expected, nfft))**2
    measured_psd = abs(np.fft.fft(actual, nfft))**2
    peak = reference_psd.max()
    np.testing.assert_allclose(measured_psd/peak, reference_psd/peak, rtol=2e-5, atol=1e-7)
    frequency = abs(np.fft.fftfreq(nfft, 1/result.sample_rate_hz))
    out_of_band = frequency >= 9e6
    assert measured_psd[out_of_band].sum()/measured_psd.sum() == pytest.approx(
        reference_psd[out_of_band].sum()/reference_psd.sum(), rel=2e-6, abs=1e-9)


@pytest.mark.parametrize('rate', RATES)
def test_40_msps_samples_same_waveform_as_20_msps(rate):
    engine = WiFiLegacyOFDMWaveformEngine()
    a, b = [engine.generate(wifi_project(WiFiSettings(legacy_rate_mbps=rate, oversample_factor=factor,
                                                    packet_period_us=1000))) for factor in (1, 2)]
    # Oversampled peaks can change existing peak normalization; remove only
    # that real gain. Odd 40-MS/s samples are also checked by the Fourier test.
    gain = np.vdot(b.iq[::2].astype(complex), a.iq) / np.vdot(b.iq[::2].astype(complex), b.iq[::2])
    assert gain.real > 0 and abs(gain.imag) < 1e-8
    np.testing.assert_allclose(b.iq[::2]*gain.real, a.iq, rtol=2e-6, atol=8e-8)
    assert b.metadata['packet_sample_count'] == a.metadata['packet_sample_count']*2


def test_receiver_fft_uses_useful_interval_not_cyclic_prefix(waveform):
    result, factor = waveform
    baseline, changed = {}, {}
    packet = analyze_iq(result.iq, result.sample_rate_hz, measurements=baseline)
    iq = result.iq.copy()
    for start in range(320*factor, int(result.metadata['packet_sample_count']), 80*factor):
        iq[start:start+16*factor] = 0  # Deliberate nonstandard CP corruption.
    recovered = analyze_iq(iq, result.sample_rate_hz, measurements=changed)
    assert recovered.integrity.crc_valid and recovered.integrity.complete
    assert recovered.decode_context['psdu_hex'] == packet.decode_context['psdu_hex']
    np.testing.assert_array_equal(np.array(changed['symbols']), np.array(baseline['symbols']))


def test_rectangular_boundary_metadata_and_outer_envelope_independence():
    project = wifi_project(WiFiSettings(packet_period_us=1000))
    assert project.power_envelope.enabled is False
    engine = WiFiLegacyOFDMWaveformEngine()
    result = engine.generate(project)
    assert result.metadata['symbol_boundary_processing'] == 'IEEE standard rectangular OFDM symbol boundary'
    assert result.metadata['windowing_standard_reference'] == 'IEEE Std 802.11-2024, 17.3.2.5; 17.3.2.6 (informational)'
    forced = engine.generate(replace(project, power_envelope=PowerEnvelopeDefinition(enabled=True)))
    np.testing.assert_array_equal(forced.iq, result.iq)


@pytest.mark.parametrize('factor', [1, 2])
def test_published_ieee_signal_and_data_bins_to_all_time_samples(factor):
    # Reuse existing external G.11/G.22 values; do not create/rewrite fixtures.
    vector = json.loads((Path(__file__).parents[1]/'data/fixtures/wifi/ieee_80211a_annex_g.json').read_text())
    settings = WiFiSettings(legacy_rate_mbps=36, oversample_factor=factor, psdu_source=WiFiPSDUSource.RAW_HEX,
                            raw_psdu_hex=vector['psdu_hex'], scrambler_seed=93, packet_period_us=1000)
    result = WiFiLegacyOFDMWaveformEngine().generate(wifi_project(settings))
    references = []
    for key in ('signal_bins', 'first_data_bins'):
        values = np.array(vector[key])
        references.append(fourier_kernel(tuple(range(-32, 32)), 4, 0.8, factor) @ (values[:, 0]+1j*values[:, 1]))
    expected = np.concatenate(references)
    actual = result.iq[320*factor:480*factor]
    gain = np.vdot(expected, actual)/np.vdot(expected, expected)
    assert gain.real > 0 and abs(gain.imag) < 1e-4  # Three-decimal source rounding.
    # Check all samples including n=0 and the SIGNAL/DATA join, not just interiors.
    np.testing.assert_allclose(actual, expected*gain.real, atol=8e-4, rtol=0)
