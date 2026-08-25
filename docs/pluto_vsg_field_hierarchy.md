# Pluto VSG hierarchical field model

## Purpose

The waveform composer distinguishes protocol structure from the samples displayed
or exported by the generator. A field therefore keeps both its logical bit count
and its transmitted-symbol count. Generated field spans additionally contain the
absolute IQ sample range.

This avoids incorrect boundaries when coding expands logical data, such as the
Bluetooth BR header's rate-1/3 FEC.

## Model

`FieldDefinition` is recursive:

- `logical_bit_count`: number of protocol bits before coding, when known
- `symbol_count`: number of transmitted symbols represented by the field
- `children`: ordered subdivisions which must exactly fill the parent span
- `data_source`, `data`, `modulation`: generation information already used by the
  common composer model

`FieldBoundary` is produced by the waveform engine and maps a field to:

- packet-relative transmitted-symbol start/stop
- absolute generated-IQ sample start/stop
- hierarchy level and parent name
- logical bit count

The project validator rejects a hierarchy when child symbol counts do not fill
the parent or when all logical counts are known but do not sum to the parent.

## Bluetooth BR / EDR DHx hierarchy currently implemented

- Access Code: Preamble, Sync Word, Trailer
- Header: LT_ADDR, TYPE, FLOW, ARQN, SEQN, HEC
- Payload: Payload Header, Payload Body, Payload CRC

The Header records 18 logical bits and 54 transmitted symbols. Each subfield's
transmitted span includes its rate-1/3 FEC expansion.

The Bluetooth settings dialog exposes LT_ADDR, FLOW, ARQN and SEQN. TYPE is
read-only and follows the selected DHx packet definition. HEC is read-only and
is recalculated immediately from the ten Header data bits and UAP; the waveform
engine uses the same calculation when generating IQ.

BR supports `DH1`, `DH3` and `DH5`. DH1 uses an 8-bit payload header; DH3 and
DH5 use a 16-bit payload header. All three use uncoded GFSK payloads with CRC-16.

EDR supports `2-DH1`, `2-DH3`, `2-DH5`, `3-DH1`, `3-DH3` and `3-DH5`. These
projects keep the same Access Code and rate-1/3 coded Header hierarchy, followed
by:

- Guard (default 5 symbols, continuous final-GFSK phase)
- EDR Data
  - EDR Sync (one differential reference symbol plus the sync word)
  - EDR Payload (16-bit header, body and CRC-16)
  - EDR Trailer (two symbols)

`2-DHx` uses Bluetooth differential mapping with pi/4-DQPSK; `3-DHx` uses the
Bluetooth 8DPSK mapping. Both use an SRRC transmit filter with configurable
roll-off (default 0.4). Payloads which do not end on an EDR symbol boundary are
zero-padded internally; the padding count is recorded in generation metadata.
The complete mixed-modulation packet is one continuous IQ array, not two preview
fragments.

EDR Guard has an independent `Guard Power rel. GFSK` setting (default 0 dB,
range -120 to +20 dB). `Guard Ramp In` and `Guard Ramp Out` (default one symbol
each) smoothly connect the final GFSK amplitude to the Guard level and the Guard
level to the actual first EDR sample amplitude. The transition shape is selectable
between cosine (default) and linear, and the two transition durations must not
exceed the Guard duration in total. The engine holds the final GFSK phase through
Guard, so the amplitude transition does not introduce a phase discontinuity. These
values are saved in the project, included in generation metadata and shown as
separate Ramp In, level and Ramp Out sections on the Visual Composer Power track.

## UI behavior

The Packet Composer uses an expanded tree and displays both Logical Bits and Tx
Symbols. Preview plots use:

- major fields: full-height magenta dashed guides
- minor fields: lower-lane orange dotted guides
- field labels: IQ Waveform, IQ Power and Instantaneous Frequency previews
- packet end: a full-height white solid guide labelled `Packet End`; repeated
  packets use a 1-based suffix so each generated packet endpoint is explicit
- packet-end labels use the same upper label lane as major-field labels
- label anchoring: fixed to the right of each boundary; labels do not switch
  sides when a boundary crosses the center of the visible plot

New projects default to a one-symbol cosine ramp at each edge. Ramp Up starts
one symbol before Packet Start (`-1.000`), while Ramp Down starts one symbol
after Packet End (`+1.000`). A loaded project keeps its saved ramp values.

New Bluetooth BR/EDR projects use 2440 MHz as the center frequency. Changing
Packet Type in Settings selects that type's maximum payload: DH1/DH3/DH5 use
27/183/339 bytes, 2-DH1/2-DH3/2-DH5 use 54/367/679 bytes, and
3-DH1/3-DH3/3-DH5 use 83/552/1021 bytes. Reopening Settings preserves the
currently saved payload length until Packet Type is changed by the user.

## Bluetooth LE RF Test Packet hierarchy

`File > New` offers editable LE 1M and LE 2M packet projects. Direct Test Mode
is a preset which populates this same model rather than a separate engine. The
hierarchy is:

- Preamble (8 bits for LE 1M, 16 bits for LE 2M)
- Sync Word (32 bits)
- PDU Header (Payload Type, RFU, CP=0, RFU)
- PDU Length (8 bits)
- PDU Payload (0 to 255 octets; omitted from the tree when length is zero)
- CRC-24 (CRCInit 0x555555)

The settings dialog exposes editable air-order Preamble, Access Address/Sync,
PDU Header, payload source/pattern/length, CRC enable/CRCInit, whitening and
channel index in addition to PHY, RF parameters, idle and power ramps. Applying
an RF Test Packet preset overwrites these controls with the fixed Sync Word,
selected Core test payload type, CRCInit 0x555555, whitening Off and standard
625-us-based interval. A PHY change selects the nominal 250 kHz (LE 1M) or
500 kHz (LE 2M) deviation and sample rate remains symbol rate times samples per
symbol.

BR/EDR uses the same preset concept: Settings can load the RF test payload
patterns into the existing Payload Source/Data controls. The preset is not a
parallel waveform representation, so users can inspect and edit every loaded
value before generation.

`Graphics > Field Boundaries` selects `Major + Minor Fields`, `Major Fields
Only`, or `Hide Field Boundaries`. The default is Major + Minor.

All VSG preview plots use the shared VSA interaction surface:

- left-button rectangle drag: zoom to the selected range
- middle-button drag: pan
- right-click `Reset`: restore the scale captured after waveform generation
- right-click `View All`: fit all finite trace data
- mouse interaction mode is fixed; the mutable pyqtgraph Mouse Mode menu is hidden

## Visual Packet Composer foundation

The first Visual Packet Composer phase is implemented as a read-only graph view.
It is the default tab in the Packet Composer panel; the existing `Field Tree` is
retained as a second tab for detailed inspection and compatibility. The graph is
derived from the same `WaveformProject` and `FieldDefinition` objects used by the
waveform engines, so it does not introduce a display-only packet definition.

The canvas has three time-aligned tracks:

- `Packet / Data`: major fields and their minor-field hierarchy
- `Modulation`: adjacent leaf fields with identical modulation are combined into
  one region, while BR-to-EDR changes remain separate regions
- `Power`: ramp-up, packet ON level and ramp-down controls

Every visual block has a stable path-based ID (`field:0`, `field:0.1`, etc.) plus
its start, duration, logical-bit count, transmitted-symbol count, data source and
modulation properties. Selecting a block populates the existing Inspector with
those values. The canvas supports horizontal scrolling and Ctrl+wheel zoom.

This phase deliberately does not mutate the project. The next editor phase must
operate on the graph/model layer and then regenerate `WaveformProject.fields`;
it must not edit graphics items as an independent source of truth. Planned editor
operations are field insertion/removal/reordering, property editing, validation,
and undo/redo. Standard-owned computed fields must remain distinguishable from
freely editable user fields.

The initial editing bridge now provides the following behavior:

- Visual Composer and Field Tree selections are synchronized by stable block ID.
- Double-clicking a visual block opens the owning Bluetooth packet/signal
  settings. Power blocks use the same settings source because the envelope is
  part of the immutable project snapshot. A separate Power Envelope button is
  intentionally not shown while it would only duplicate Packet Settings.
- Accepted settings changes and RF-test-preset application are project-level
  undoable commands. `Ctrl+Z` / `Ctrl+Y` restore the complete project snapshot,
  rebuild the graph, regenerate IQ and refresh every preview.
- New/open project operations clear the history so undo never crosses a project
  boundary.

Bluetooth standard blocks remain protected at this stage. Directly changing a
graphics item would desynchronize it from HEC, CRC, FEC, whitening and packet
length calculation, so edits are routed through the standard profile settings.
Free add/insert/reorder/delete will be enabled together with the Experimental
Profile/generic waveform engine, where the field graph itself is the generation
source of truth.

## Compatibility and next steps

Version-1 Bluetooth project files without children are upgraded to the current
DH1 hierarchy when loaded. Other profiles remain valid with flat fields.

Future packet profiles should generate their hierarchy from the same settings
used to generate bits. HDT, mixed modulation and coding stages should add
their own logical/transmitted mappings instead of deriving UI boundaries from
display-only constants.
