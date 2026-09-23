# VSA Acquisition Trigger, Burst Search, and Pattern Search

## Purpose

The VSA separates three operations, following the R&S measurement flow:

1. **Acquisition Trigger** decides where a new Pluto record starts.
2. **Post-capture Burst Search** finds every active interval in the acquired
   record (or in a loaded file).
3. **Pattern Search** aligns a Result Range inside each eligible burst.

No-signal intervals can produce high normalized pattern correlation because
normalization removes absolute amplitude. Burst Search prevents those intervals
from becoming pattern candidates, while Acquisition Trigger reduces unnecessary
capture data before the first event.

Burst Search and Pattern Search apply equally to Pluto captures and loaded I/Q
files. Acquisition Trigger applies to Pluto `Run Single` only. The separation is
intentional: one triggered record can still contain multiple bursts, and every
qualifying event must remain navigable.

## Acquisition Trigger (Pluto Run Single)

The Trigger page offers `Free Run` and `I/Q Power`. I/Q Power uses calibrated
display dBm as its input and converts it back to the common raw-IQ dBFS detector
reference. It uses the same internal gain, external attenuation, external gain,
base calibration, and frequency-dependent correction as the IQ Power trace.

Implemented controls:

- Level in dBm.
- Rising, Falling, or Either slope.
- Hysteresis in dB.
- Signed Trigger Offset in symbols. A negative value retains pretrigger data;
  a positive value starts the returned record after the crossing.
- Operator cancellation. While waiting, invoking `Run Single` again requests
  cancellation, equivalent to aborting the highlighted R&S Run Single action.

The returned record always has the configured capture length. Trigger Offset
does not change record length. The first Pluto buffer is acquired with the
fresh-buffer path so samples queued by a previous acquisition are not reused.

This stage intentionally does not perform Burst Search or Pattern Search. A
power crossing is not a protocol boundary and does not establish symbol timing.
Drop-Out and Holdoff remain post-capture Burst Search controls; they become
acquisition controls only when Continuous acquisition/rearming is implemented.

## R&S-aligned post-capture behavior

The design follows the relevant R&S VSA concepts:

- I/Q Power is evaluated using the usable I/Q acquisition bandwidth.
- Level is expressed in the same dBm reference plane as the I/Q Power trace.
- Hysteresis defines the lower re-arm level as `Level - Hysteresis`.
- Envelope Average smooths linear I/Q power before falling-edge detection.
- Drop-Out Time requires power to remain below the re-arm level before a new
  event is permitted.
- Holdoff specifies a minimum interval between events.
- Search Start Offset is signed. Positive values delay search from the trigger;
  negative values include data before the trigger.
- Pattern search returns the first eligible pattern in each detected active
  interval, corresponding to R&S burst-gated pattern search behavior.

Unlike the acquisition trigger, Burst Search scans the entire already-acquired
buffer. This is required to detect all packets in one Pluto or file recording.

## Processing

The trigger detector uses the same amplitude conversion as the VSA trace:

```text
power_dbfs = 20 log10(|IQ| / full_scale)
power_dbm  = power_dbfs + dbfs_to_dbm_offset_db
```

For each rising crossing of `Level`:

1. Record the trigger sample.
2. Keep the interval active through short power dips.
3. End the interval only after `Drop-Out Time` continuously below
   `Level - Hysteresis`.
4. Apply Holdoff before accepting another crossing.
5. Begin pattern search at `trigger sample + Search Start Offset`.
6. Accept only a pattern whose start remains inside that trigger's active
   interval.
7. Add the first eligible match from the interval to the chronological Result
   Range list.
8. When `Limit Result Range to Active Interval` is enabled, limit the waveform
   to the detected falling edge before demodulation, synchronization, amplitude
   normalization, and EVM/frequency-error evaluation. Discard any final symbol
   whose complete interval still extends beyond that edge.

The pre-demodulation limit is important when the configured Result Length is
deliberately longer than a burst. Before the 2026-08-18 fix, the application
could normalize PSK and calculate EVM over the requested length (for example
3500 symbols), then trim only the displayed arrays to the detected active count
(for example 699 symbols). `Result Symbols` was correct, but Symbol Plot scale
and EVM retained the inactive tail as their measurement population. The active
interval is now the common population for demodulation, normalization, EVM,
Symbol Plot, and Symbol Table.

The rising trigger uses unsmoothed power so the averaging filter does not move
Pattern Search earlier than the physical crossing. The falling edge uses the
averaged envelope and compensates its nominal half-window delay.

New IQ acquisition or file load selects the first triggered match. Refresh
Analysis retains the current one-based match index. The existing Left/Right
navigation changes between successful triggered matches.

When a triggered Pattern or Detected Data result is selected, reset/default
time-domain plot ranges begin one existing display margin before that result's
power-trigger sample. The margin remains 10% of the selected Result Range
duration, matching the non-triggered layout. The right edge remains one 10%
margin after Result Range end. This changes only the initial view; the capture
trace is retained and remains available through zoom/pan.

## Configuration and defaults

The Meas Config `Trigger` page contains two independent sections.

Acquisition Trigger defaults:

| Setting | Default | Unit |
| --- | ---: | --- |
| Trigger Source | Free Run | — |
| Level | -20.00 | dBm |
| Slope | Rising | — |
| Trigger Offset | 0.000 | symbols（0では先頭burst保護のため16 symbolsを自動prestore） |
| Hysteresis | 3.0 | dB |

Post-capture Burst Search defaults:

| Setting | Default | Unit |
| --- | ---: | --- |
| Burst Search | Off | — |
| Level | -20.00 | dBm |
| Hysteresis | 3.00 | dB |
| Envelope Average | 1.00 | symbols |
| Drop-Out Time | 8.00 | symbols |
| Holdoff | 0.00 | symbols |
| Search Start Offset | 0.000 | symbols |
| Limit Result Range to Active Interval | On | — |

Symbol-based durations scale automatically with Signal Description > Symbol
Rate. Settings are persisted in both `.vsaconfig.json` and the startup
configuration. The acquisition section is stored as `acquisition_trigger` and
Burst Search as `burst_search`. Older files using `iq_power_trigger` are
accepted as a compatibility alias.

For constant-envelope and filtered FSK/PSK, one-symbol envelope averaging plus
a Drop-Out longer than expected short gaps prevents ordinary modulation ripple
from splitting a burst. For OOK, a zero symbol is physically indistinguishable
from no transmission using power alone. Set Drop-Out longer than the maximum
valid zero run, or disable result limiting. Unknown OOK without a bounded zero
run requires protocol/preamble knowledge or a separate timing model; power-only
end detection cannot determine the boundary uniquely.

## Result metadata

`PatternSearchResult.metadata` includes the trigger level, trigger/search/stop
sample positions, total detected event count, matched event count, and selected
trigger-event index. `eligible_match_count` is the number of trigger intervals
that produced an eligible pattern, so existing Result Range navigation remains
compatible.

## Remaining acquisition work

Continuous acquisition, trigger rearming, external hardware trigger, and
acquisition-stage Drop-Out/Holdoff are not implemented.
They must reuse the common stream/trigger contracts and must not replace the
post-capture multi-event Burst Search.
