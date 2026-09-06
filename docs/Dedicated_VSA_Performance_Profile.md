# Bluetooth / DECT Dedicated VSA performance profile

## Scope and method

Measurements were made with the repository virtual environment on 2026-09-06.
Each multi-packet input is ten concatenated copies of the named IQ fixture.  The
normal analyzer entry points, RF PHY measurement profile, exact packet decode,
and all conformance optimizers were used.  Wall-clock values are representative
single runs and are intended for before/after comparison on the same machine.

## Baseline hotspots

| Analyzer | Profiled workload | Dominant baseline work |
| --- | ---: | --- |
| BR / EDR | 10 packets | Repeated full pattern search and provisional 4096-symbol analysis; EDR PSK waveform least-squares fit dominated runtime |
| HDT 7.5 | 10 packets | Hard-decision punctured Viterbi decode (7.45 s of 8.44 s profiled) |
| DECT | 10 packets | Nested symbol-rate / half-sample / direction synchronization loop (8.47 s of 8.93 s profiled) |
| LE 1M | 64 packets | One packet-local GFSK demodulation and distortion fit per packet; already approximately 29 packets/s |

The profiles separately exposed capture preprocessing/power envelope, candidate
search, coarse/fine synchronization, filtering/resampling, FM/PSK demodulation,
carrier/timing estimation, decode, RF measurement/EVM, result assembly, and VSA
plot-product analysis.  The three hotspots above accounted for the large
majority of the slow workloads; decode/result assembly outside HDT was small.

## Changes

- DECT S-field coarse synchronization evaluates the identical rate, half-sample
  and direction candidate grid with batched NumPy interpolation/correlation.
- DECT power, instantaneous frequency and sample-position arrays are computed
  once per capture and shared read-only by packet results.  Bit-window selection
  uses sorted-index boundaries instead of rebuilding a full-capture Boolean mask
  for every bit.
- HDT matched-filter output and training candidates are computed once per
  capture.  The K=6 Viterbi trellis and branch metrics are precomputed and each
  exact hard-decision step is vectorized without changing tie handling.
- Classic BR/EDR discovery reads only the payload header required to determine
  Length; the final pass still analyzes the exact complete packet.
- Packet TYPE values shared by BR and EDR always check the deterministic EDR
  PSK synchronization position before falling back to BR.  A temporary
  CRC-first shortcut was rejected because it could misclassify valid EDR.
- The confirmed local EDR sync supplies the provisional enhanced ACL Length.
  The exact payload pass retains the existing fine synchronization and optimizer.
- Candidate order is fixed before packet-local work.  Independent Classic/EDR
  and LE packets run in a bounded four-worker pool; output order remains capture
  order.
- Only the initially selected packet builds every Generic VSA plot derivative.
  Non-selected packets build the same corrected selected-range result used by
  summary values; the remaining display-only products are materialized when the
  user selects that packet.

## Representative wall-clock result

| Fixture / packets | Before | After | Speed-up |
| --- | ---: | ---: | ---: |
| `bluetooth_br_prbs9_pluto_16msps.npz` / 20 | 36.6780 s | 4.5907 s | 8.0x |
| `bluetooth_2dh1_prbs9_16msps.npz` / 10 | 20.5496 s | 2.8991 s | 7.1x |
| `bluetooth_3dh1_prbs9_16msps.npz` / 10 | 21.5344 s | 2.7586 s | 7.8x |
| `LE1M_FSK_error.npz` / 64 (candidate cap) | 2.2148 s | 1.6664 s | 1.3x |
| `bluetooth_hdt7_5_prbs9_16msps.npz` / 10 | 5.6608 s | 0.6181 s | 9.2x |
| `dect_rfp_p32_prbs9_9p216msps.npz` / 10 | 4.4715 s | 0.3545 s | 12.6x |

EDR remains limited by the required full-result differential PSK least-squares
synchronization.  That optimizer, its candidate grid, and its stopping conditions
were deliberately retained, so the 10x target was not claimed for EDR.  LE was
already fast and its RF measurement/distortion fit was not approximated merely
to improve the ratio.

## Equivalence checks

Regression compares packet count/order, PHY and packet type, decoded fields,
CRC/HEC, synchronization products, RF measurement values, EVM/DEVM and plot
coordinates.  Dedicated analyzer and RF measurement suites pass after the
changes.  Additional tests compare the vectorized HDT trellis step with the
previous scalar state transition/tie rule and the DECT sorted bit windows with
the original Boolean-window definition.

No RF measurement filter, sample rate, synchronization threshold, search
resolution, EVM/DEVM definition, measurement window, or conformance optimizer
condition was reduced.
