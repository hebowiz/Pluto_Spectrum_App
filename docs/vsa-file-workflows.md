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
currently selected modulation. Existing and newly entered table cells are
always center-aligned; alignment is applied by the edit handler as well as the
file/table population path.

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
- Pattern Search enable, name, format, symbols, threshold, correctness rule,
  match-selection policy, and one-based match index
- Result Range length, reference, alignment, offset, reference numbering, and
  incomplete-range exclusion
- Demodulation synchronization contracts, bit ordering, CFO-drift
  compensation, and FSK deviation compensation
- Result Summary visible item IDs. Missing sections from older version-1 files
  restore the current default selection; unknown future IDs are ignored.

Loading a configuration applies the controls and immediately reruns analysis
against the currently loaded IQ capture. Display-only window layout and
Carrier Display selection are intentionally not measurement configuration.

Serialization and schema validation live in `pluto_sa/vsa/persistence.py`.

Existing version-1 configuration files that predate multiple-match selection
remain valid. Missing fields load as `First`, match index `1`, and
`Exclude incomplete Result Range = Off`.

### Automatic startup configuration

On a clean window close, the same measurement-control values are serialized
as a compact schema/version-tagged JSON document in Qt `QSettings` under
`startup/measurement_config`. The next VSA launch applies those controls but
starts with an empty `VSASession`: no IQ array, IQ filename, source recording,
or analysis result is persisted or reopened. The separate last-directory
preferences remain available only as file-dialog starting locations.

Invalid, truncated, wrong-schema, or unsupported-version startup JSON is
removed and the application opens with widget defaults and no capture. Startup
restore does not run analysis; `Run Single`, `Open IQ`, or a generated test
signal must provide the next recording.

## Multiple pattern matches in one capture

Pattern Search now keeps all above-threshold local correlation peaks and
collapses detections from adjacent symbol-timing phases into one physical
packet candidate. This behavior is shared by FSK and PSK. The UI has no match
selection policy: eligible candidates are always numbered in capture-time order.
A newly loaded file, generated recording, or Pluto capture always selects index
1. Refreshing or reconfiguring analysis for the same immutable IQ recording
keeps the current one-based index. `Meas only if Pattern Symbols Correct` and
the incomplete-range option determine which detections enter this eligible list.

The Result Summary displays the selected/eligible index as `Selected Result`,
for example `2 / 8`. Result metadata also records
`detected_match_count`, `eligible_match_count`, and `selected_match_index`.

`Result Range > Exclude incomplete Result Range` controls captures that end
before the requested result length is available. When Off (the backward-
compatible default), the selected result is returned with the available
symbols only. When On, incomplete candidates are removed before indexing.
Consequently, the selected index numbers the remaining eligible candidates,
not the raw detections. If none remain, pattern
analysis reports that no match satisfies the result-range requirements.

The completeness requirement includes Result Range alignment and offset. FSK
also respects the detected burst end; PSK uses the available demodulated
symbols up to the capture end. The current analyzer still returns one selected
result at a time. `Sweep / Run > Previous Result Range` (`Left`) and
`Next Result Range` (`Right`) change to the adjacent eligible match without a
new IQ acquisition and rerun analysis against the same immutable capture. The
actions stop at the first/last result rather than wrapping. Match policy and
index are no longer measurement Config properties; legacy Config fields are
accepted but ignored.

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

## Modulation and Symbol Plot layout

The lower-left dock keeps the window name `Modulation`; the former `Reserved`
dock is named `Symbol Plot`. Their contents depend on the modulation family:

| Modulation | Modulation dock | Symbol Plot dock |
| --- | --- | --- |
| PSK | connected IQ trajectory | constellation decision points |
| FSK | instantaneous frequency vs time | per-symbol phase difference |

For PSK, the IQ trajectory uses the selected Result Range when Pattern Search
succeeds. It follows `Display Config > Carrier Display`, choosing raw or
carrier-corrected IQ. The IQ is resampled to 8 samples/symbol and, when the TX
filter is Root Raised Cosine, passed through the same-beta SRRC matched receive
filter before plotting. The continuous waveform is then cropped to Result
Range; without a pattern result the full filtered waveform is used. Its scale
is normalized by the RMS amplitude of the filtered IQ at recovered symbol
times. Display samples are capped at 20,000 plot vertices; this affects only
rendering and never modifies stored IQ or DSP results. The I/Q axes use equal
scale and a symmetric range based on the 99.5th amplitude percentile.

For differential PSK, the trajectory markers are absolute filtered IQ samples,
while the constellation contains differential products between adjacent
symbols. Their amplitude distributions can therefore still differ slightly,
but the earlier pre-filter/post-filter mismatch has been removed.

When Root Raised Cosine is selected, PSK pattern analysis uses two carrier
recovery passes. The first pass estimates coarse CFO. The resampled IQ is then
frequency-centered before the SRRC matched/measurement filter, and the second
pass estimates residual CFO, phase, timing, and drift. Reported carrier error is
the sum of coarse and residual CFO and remains referenced to the original IQ
recording. This ordering is modulation-generic; Bluetooth EDR merely supplies
the conventional rolloff value 0.4.

### Normalization difference from R&S FPL1-K70

The current Pluto VSA constellation normalization is an interim implementation,
not an exact reproduction of R&S processing. After carrier/phase correction it
divides the selected measured (differential, for DPSK) symbols by their measured
RMS magnitude, while the ideal PSK alphabet has unit magnitude.

The FPL1-K70 manual rev.12 describes a different full measurement model:

- the physical differential-PSK constellation contains decision points after
  ISI-free demodulation and is de-rotated for the configured standard (pp.86-88,
  302);
- analyzer scaling is optimized to minimize mean-square error-vector magnitude,
  or to minimize EVM, according to the Demodulation `Optimization` setting
  (pp.135-136, 222);
- `Normalize EVM to` independently selects Max/Mean Reference Power or Max/Mean
  Constellation Power for the EVM denominator (pp.128, 222). This setting is not
  simply the constellation display scaling.

For constant-envelope, high-SNR PSK the current measured-RMS scale can be
numerically close to the R&S optimum global scale, but it is not equivalent in
the presence of noise, gain error, amplitude distortion, filtering mismatch, or
non-constant-envelope modulation. A future R&S-compatible implementation must
fit one global complex gain against the reconstructed reference sequence and
implement `Optimization` and `Normalize EVM to` as separate settings. It must
not normalize every symbol independently.

For FSK, each demodulated symbol-frequency value `f[k]` sets the phase of the
display vector using `exp(j * 2*pi*f[k]/symbol_rate)`. The +I axis is zero
phase; the point angle is the phase accumulated over one symbol. Its magnitude
is sampled from the analysis IQ waveform at the recovered symbol instant. One
global RMS magnitude over the selected symbols is used for normalization, as
for the PSK symbol display; individual points are not projected onto the unit
circle. Ideal constant-envelope 2FSK therefore forms two clusters near the
positive and negative deviation angles, while real amplitude variation remains
visible as radial spread. A unit-circle guide is drawn behind the measured
points, and the plot range expands when the measured radial spread exceeds the
default +/-1.25 range.

## Symbol-position overlay

`Display Config > Show Symbol Points` is an independent display-only toggle and
defaults to off. When enabled, bright green circular markers distinguish symbol
centres from the yellow Power/IQ traces and the magenta FSK trace.

- IQ Power: interpolate dBm at each decoded symbol-centre time.
- FSK Modulation: interpolate instantaneous frequency at each symbol centre.
- PSK Modulation: interpolate matched-filter output at each symbol centre and
  apply the same RMS normalization as the displayed IQ trajectory.

With a successful Pattern Search, the markers cover the current Result Range;
otherwise they cover the symbol decisions from the normal analysis result.
Rendering is capped at 20,000 markers and does not alter DSP data or results.

## Graph scale and mouse interaction

Every completed analysis establishes a new initial X/Y range for IQ Power,
Spectrum, Modulation, and Symbol Plot. Manual zoom/pan does not modify this
snapshot. `Display Config > Reset Graph Scales` restores all four plots to the
snapshot; the `Home` key is its shortcut. Re-running analysis discards the old
snapshot and calculates a new one from the new result.

Each plot's standard right-click menu is retained and extended at the top:

- `Reset` restores only that plot to the latest analysis-complete snapshot.
- `View All` fits all finite points in the visible data traces with 5% margin.
  Result-range regions, pattern markers, infinite boundary lines, and other
  overlays are deliberately excluded from the bounds calculation. IQ-plane
  plots keep equal I/Q scaling and at least the default +/-1.25 range.

Mouse interaction is fixed and has no mode-selection menu:

- left drag selects a rectangle and zooms to it;
- middle-button (mouse-wheel) drag pans the visible range;
- right click opens the plot context menu; right drag retains pyqtgraph's axis
  scaling behavior.

This is implemented by a VSA-specific ViewBox: its persistent left-button mode
is Rect Zoom, while a middle-button drag alone uses pyqtgraph's three-button
Pan path. `Display Config > Mouse Interaction` and the standard context-menu
`Mouse Mode` submenu are removed so all four plots keep the same behavior.

Display-only refreshes, such as toggling symbol points or raw/carrier-corrected
display, preserve the current manual view instead of silently resetting it.

`Display Config > Symbol Plot Trace` switches both PSK constellation symbols
and FSK phase-difference vectors between `Flat` and `Density`. Flat draws every
final symbol as a yellow point. Density bins the same symbols into a 96 by 96
I/Q histogram. PSK uses the fixed +/-1.25 plane; FSK uses a symmetric plane
large enough for its RMS-normalized phase-difference amplitude. Each
observation is spread with a Gaussian kernel having a
standard deviation of 0.7 bins, and maps `log(1 + density)` through a
blue-to-red color map. Density at 75% of the current peak and above saturates
to red, making a tightly converged cluster core clearly visible without
widening the density kernel; zero-density cells are transparent. The smoothing makes
small single-packet result sets appear as a continuous occurrence field while
preserving relative occurrence. It is a display-only R&S-inspired view, not a
new DSP result, so decoded symbols and EVM are unchanged. The selected mode is
stored in manual and automatic startup Config. Older Config files without
`display_config` default to Flat.

## Result Summary item selection

The Result Summary context menu and `Meas Config > Result Summary` use the same
canonical item registry in `pluto_sa/vsa/result_summary.py`. The context menu is
grouped into Common, PSK, FSK, and Synchronization Diagnostics submenus. The
Config page exposes the same entries as a persistent checklist. `Show All`,
`Measurement Results Only`, `Diagnostics Only`, and `Restore Defaults` presets
are available from both locations.

R&S-defined items that are not calculated yet remain visible as disabled `Not
implemented` entries. This avoids silently presenting synchronization metrics as
standards-compatible measurement results. In particular, `Sync EVM RMS` and
`Frequency Fit RMS` remain diagnostics distinct from `EVM RMS` and `Frequency
Error RMS`.

Current implemented measurement rows include linear-mean result power, carrier
frequency error, PSK EVM RMS and symbol-rate error, and FSK frequency-error RMS,
measured/reference/deviation error, and carrier drift. FSK deviation error is
reported in Hz and carrier drift in Hz/symbol to match the R&S Result Summary
convention. These values remain development measurements, not conformance data.

For Pattern Search PSK results, `EVM RMS` now uses the exact final complex symbol
array supplied to the Constellation (`PatternSearchResult.measured_symbols`) and
the ideal alphabet points selected by the corresponding decoded symbols. It is
calculated over the current Result Range as
`100 * sqrt(sum(|measured-reference|^2) / sum(|reference|^2))`. The denominator
therefore implements mean-reference-power normalization for a constant number of
symbols. The display rotation is already included in both arrays and does not
change the vector error.

This makes the displayed EVM and Constellation internally consistent at one
decision point per symbol. It follows the R&S principle that EVM is evaluated
between the synchronized, corrected measurement and reference at the selected
evaluation points. It is not yet full R&S equivalence: Pluto VSA does not expose
an independent Evaluation Range, Display Points/Symbol, Optimization, Measurement
Filter, or `Normalize EVM to` selector. `Sync EVM RMS` remains a separate fitting
diagnostic and is not used as the displayed measurement EVM.

## Tests

`tests/test_vsa_persistence.py` covers versioned JSON round trips and schema
rejection. `tests/test_vsa_ui.py` covers the editable pattern table, Config
control round trip, automatic Config-only startup restore, empty-session and
corrupt-preference fallback, separate folder preferences, matched-symbol
highlighting, PSK IQ trajectory/constellation placement, and FSK
phase-difference plotting.
