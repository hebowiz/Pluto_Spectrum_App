import numpy as np

from pluto_vsa.protocol_modes.wifi.display import power_display_indices


def test_low_duty_packets_keep_every_sample_and_exact_peak_times():
    power = np.full(2_000_003,-90.)
    start,stop = 1_432_101,1_433_101
    power[start:stop] = -30+np.sin(np.arange(stop-start))
    power[-2] = 5  # Last partial bucket must also survive.
    before = power.copy()
    indices = power_display_indices(power,0,len(power),[(start,stop)])
    assert np.isin(np.arange(start-1,stop+1),indices).all()
    assert len(power)-2 in indices
    assert indices[0]==0 and indices[-1]==len(power)-1
    assert np.all(np.diff(indices)>0)
    assert len(indices)<6000
    np.testing.assert_array_equal(power,before)


def test_zoom_recovers_original_samples_even_without_detected_packets():
    power = np.sin(np.arange(1_000_007)*.37)
    overview = power_display_indices(power,0,len(power))
    zoomed = power_display_indices(power,500_000,501_000)
    assert len(overview)<5000
    np.testing.assert_array_equal(zoomed,np.arange(500_000,501_000))


def test_many_long_packets_have_bounded_display_and_preserve_edges():
    power = np.sin(np.arange(4_000_007)*.19)
    ranges = [(i*30_000,i*30_000+25_000) for i in range(128)]
    indices = power_display_indices(power,0,len(power),ranges,64)
    assert len(indices)<105_000
    for start,stop in ranges:
        assert max(0,start-1) in indices
        assert stop in indices


def test_empty_nan_and_outside_view():
    assert not power_display_indices(np.array([]),0,1).size
    assert not power_display_indices(np.ones(10),20,30).size
    power = np.full(100_003,np.nan)
    indices = power_display_indices(power,-10,len(power)+10)
    assert indices[0]==0 and indices[-1]==len(power)-1
