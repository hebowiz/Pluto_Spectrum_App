# VSA I/Q Power Trigger and Pattern Search Gate

## Purpose

No-signal intervals can produce high normalized correlation because correlation
normalization removes absolute amplitude. The VSA therefore supports a
post-capture I/Q Power Trigger that limits known-pattern search to intervals
that contain sufficient calibrated I/Q power.

This feature applies equally to Pluto captures and loaded I/Q files. It is
separate from a future acquisition trigger that controls when Pluto starts a
record. The separation is intentional: a stored capture can contain multiple
bursts, and every qualifying event must remain navigable.

## R&S-aligned behavior

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

Unlike an R&S hardware acquisition trigger, this implementation scans the
entire already-acquired buffer. This is required to detect all packets in one
Pluto or file recording.

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
8. When `Limit Result Range to Active Interval` is enabled, discard symbols
   whose complete symbol interval extends beyond the detected falling edge.

The rising trigger uses unsmoothed power so the averaging filter does not move
Pattern Search earlier than the physical crossing. The falling edge uses the
averaged envelope and compensates its nominal half-window delay.

New IQ acquisition or file load selects the first triggered match. Refresh
Analysis retains the current one-based match index. The existing Left/Right
navigation changes between successful triggered matches.

## Configuration and defaults

The Meas Config `Trigger` page contains:

| Setting | Default | Unit |
| --- | ---: | --- |
| I/Q Power Trigger | Off | — |
| Level | -20.00 | dBm |
| Hysteresis | 3.00 | dB |
| Envelope Average | 1.00 | symbols |
| Drop-Out Time | 8.00 | symbols |
| Holdoff | 0.00 | symbols |
| Search Start Offset | 0.000 | symbols |
| Limit Result Range to Active Interval | On | — |

Symbol-based durations scale automatically with Signal Description > Symbol
Rate. Settings are persisted in both `.vsaconfig.json` and the startup
configuration. Older configuration files without the `iq_power_trigger`
section load with the defaults above.

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

Pluto `Run Single` still acquires a finite Free Run record. A future acquisition
trigger may align that record to the first event and provide pre-trigger data,
but it must not replace this post-capture multi-event scan.
