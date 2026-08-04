# VSA pattern, configuration, and IQ-trajectory workflow

This note records the implementation state introduced on 2026-08-04 so a
future maintainer or AI can continue the VSA work without reconstructing the
UI contracts.

## Known-pattern editor and file

`Meas Config > Pattern Search` contains a 10-column editable Symbol Table. Each
cell represents one modulation symbol; the vertical header is the zero-based
index of the first symbol in that row. `Add Row` and `Remove Last Row` change
the available table capacity. Empty cells are permitted only after the final
defined symbol.

`Symbol Format` controls both editing and file metadata:

- `Binary`: one zero-padded binary value per symbol cell
- `Decimal`: one decimal value per cell
- `Hexadecimal`: one hexadecimal value per cell

Changing the format changes only the representation, not the stored symbol
values. A loaded pattern is rejected when a symbol is outside the order of the
currently selected modulation.

Pattern files use UTF-8 JSON with the preferred extension
`.vsapattern.json`. The version-1 fields are:

```json
{
  "schema": "pluto-vsa-pattern",
  "version": 1,
  "name": "Known Pattern",
  "symbol_format": "Binary",
  "symbols": [0, 1, 0, 1]
}
```

The Symbol Table result view uses a green cell background for decoded symbols
whose symbol-centre times are inside the matched Pattern Waveform interval.
The comparison uses `PatternSearchResult.symbol_time_s` and the measured
pattern start/stop times, so Result Range offsets and alignment are respected.

## Measurement configuration file

`Meas Config` files use UTF-8 JSON with the preferred extension
`.vsaconfig.json`. They are loaded or saved from the main-window `Meas Config`
menu. Config file buttons are not duplicated in `Config Top Menu` or individual
pages, avoiding confusion with the Pattern-specific file buttons on the Pattern
Search page.

Version 1 stores all currently exposed measurement controls:

- Input / Frontend analysis-channel enable, center, and bandwidth
- Signal Description modulation, symbol rate, FSK deviation, mapping, TX
  filter, and filter parameter
- Pattern Search enable, name, format, symbols, threshold, and correctness rule
- Result Range length, reference, alignment, offset, and reference numbering
- Demodulation synchronization contracts, bit ordering, CFO-drift
  compensation, and FSK deviation compensation

Loading a configuration applies the controls and immediately reruns analysis
against the currently loaded IQ capture. Display-only window layout and
Carrier Display selection are intentionally not measurement configuration.

Serialization and schema validation live in `pluto_sa/vsa/persistence.py`.

## Last-used folders

Qt `QSettings` organization `PlutoSA`, application `PlutoVSA`, stores three
independent directory keys:

- `directories/iq`
- `directories/pattern`
- `directories/config`

Each corresponding Open/Save dialog starts in its own last selected folder.
These preferences persist between application runs and do not form
part of a measurement configuration file.

The selected directory is recorded as soon as a non-empty filename is returned
by the dialog, even if parsing or writing the selected file later fails. Before
a file type has its own history, the application passes the current working
directory explicitly. It never passes an empty start path because the Windows
native dialog would then fall back to a process-wide folder history and make
the Pattern and Config locations appear to be shared.

## IQ trajectory

The former `Reserved` dock is now `IQ Trajectory`. It draws a connected path
of complex samples on the I/Q plane, comparable to the R&S VSA IQ trajectory
view.

When Pattern Search succeeds, the trajectory uses only the selected Result
Range. It follows `Display Config > Carrier Display`, choosing raw or
carrier-corrected Result Range IQ. Without a pattern result it uses the full
analysis result. Display samples are RMS-normalized and capped at 20,000 plot
vertices; this affects only rendering and never modifies stored IQ or DSP
results. The I/Q axes use equal scale and a symmetric range based on the 99.5th
amplitude percentile.

## Tests

`tests/test_vsa_persistence.py` covers versioned JSON round trips and schema
rejection. `tests/test_vsa_ui.py` covers the editable pattern table, Config
control round trip, separate folder preferences, matched-symbol highlighting,
and IQ trajectory population.
