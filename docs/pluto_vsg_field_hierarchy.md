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

## Bluetooth BR / DH1 hierarchy currently implemented

- Access Code: Preamble, Sync Word, Trailer
- Header: LT_ADDR, TYPE, FLOW, ARQN, SEQN, HEC
- Payload: Payload Header, Payload Body, Payload CRC

The Header records 18 logical bits and 54 transmitted symbols. Each subfield's
transmitted span includes its rate-1/3 FEC expansion.

The Bluetooth settings dialog exposes LT_ADDR, FLOW, ARQN and SEQN. TYPE is
currently read-only as `4 (DH1)` because the first vertical slice only generates
DH1. HEC is read-only and is recalculated immediately from the ten Header data
bits and UAP; the waveform engine uses the same calculation when generating IQ.

The current DH1 vertical slice still uses the existing uncoded-payload generator.
The Payload child spans deliberately describe that generated waveform; this
change does not silently alter its coding or modulation behavior.

## UI behavior

The Packet Composer uses an expanded tree and displays both Logical Bits and Tx
Symbols. Preview plots use:

- major fields: full-height magenta dashed guides
- minor fields: lower-lane orange dotted guides
- field labels: IQ Waveform, IQ Power and Instantaneous Frequency previews
- label anchoring: fixed to the right of each boundary; labels do not switch
  sides when a boundary crosses the center of the visible plot

`Graphics > Field Boundaries` selects `Major + Minor Fields`, `Major Fields
Only`, or `Hide Field Boundaries`. The default is Major + Minor.

All VSG preview plots use the shared VSA interaction surface:

- left-button rectangle drag: zoom to the selected range
- middle-button drag: pan
- right-click `Reset`: restore the scale captured after waveform generation
- right-click `View All`: fit all finite trace data
- mouse interaction mode is fixed; the mutable pyqtgraph Mouse Mode menu is hidden

## Compatibility and next steps

Version-1 Bluetooth project files without children are upgraded to the current
DH1 hierarchy when loaded. Other profiles remain valid with flat fields.

Future packet profiles should generate their hierarchy from the same settings
used to generate bits. EDR, HDT, mixed modulation and coding stages should add
their own logical/transmitted mappings instead of deriving UI boundaries from
display-only constants.
