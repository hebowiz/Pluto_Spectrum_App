"""Display-only sampling; measurement arrays are never modified."""
import numpy as np


def _peak_indices(power, start, stop, buckets):
    """Keep real sample coordinates, extrema, boundaries and the final bucket."""
    size = stop-start
    if size <= 2*buckets:
        return np.arange(start, stop, dtype=np.int64)
    width = int(np.ceil(size/buckets))
    count = size//width
    blocks = np.nan_to_num(power[start:start+count*width], nan=-np.inf).reshape(count,width)
    offsets = np.arange(count)*width+start
    indices = [offsets+np.argmin(blocks,axis=1), offsets+np.argmax(blocks,axis=1),
               np.array([start,stop-1])]
    tail = start+count*width
    if tail < stop:
        values = np.nan_to_num(power[tail:stop],nan=-np.inf)
        indices.append(np.array([tail+np.argmin(values),tail+np.argmax(values)]))
    return np.concatenate(indices)


def power_display_indices(power, start, stop, packet_ranges=(), selected_packet=0):
    """Bounded peak envelope with extra detail in visible detected packets.

    Call again on zoom using the original power array. Packet ranges are in
    this power array's sample coordinates, independent of analysis sample rate.
    """
    start = max(0, min(len(power), int(start)))
    stop = max(start, min(len(power), int(stop)))
    if start == stop:
        return np.empty(0,dtype=np.int64)
    indices = [_peak_indices(power,start,stop,2048)]
    visible = [(i,max(start,a-1),min(stop,b+1))
               for i,(a,b) in enumerate(packet_ranges) if b >= start and a < stop]
    budget = min(8192,max(1,32768//max(1,len(visible))))
    for i,a,b in visible:
        indices.append(_peak_indices(power,a,b,16384 if i == selected_packet else budget))
    return np.unique(np.concatenate(indices))
