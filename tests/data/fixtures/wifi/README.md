# IEEE legacy OFDM reference

`ieee_80211a_annex_g.json` contains numeric test data extracted from IEEE Std
802.11a-1999, Annex G (36 Mbps, 100-byte PSDU, scrambler state 1011101).
Source URL and SHA-256 are recorded in the JSON. Expected values were extracted
from the publication, not produced by Pluto VSG. G.8/G.9 were transcribed from
rendered printed pages 61/62 because PDF text extraction merges their columns.

- G.1: PSDU; G.7-G.12: SIGNAL bits, BCC, interleaving, bins and time samples.
- G.13-G.18: first/last uncoded/scrambled data, scrambler sequence, punctured BCC.
- G.21/G.22: interleaving and first DATA constellation/frequency bins.
- G.24 is deliberately excluded: its published samples are not a reliable
  whole-packet golden waveform. DATA time samples are checked against the IFFT
  of the independently published G.22 bins instead.

The 1999 informative Annex contains erroneous STF numbers (G.2/G.3/G.24),
recorded in the [IEEE July 2000 minutes](https://grouper.ieee.org/groups/802/11/Minutes/Cons_Minutes_July-2000.pdf).
STF tests use normative Clause 17.3.3 Equation (6), confirmed against
[gr-ieee802-11's training sequence](https://github.com/bastibl/gr-ieee802-11/blob/maint-3.10/examples/wifi_phy_hier.grc).
No incorrect Annex STF is substituted into the generator.
Annex time samples are rounded to three decimal places and use a 1/64 IFFT;
tests account for rounding, VSG's common amplitude scale, and the explicitly
non-normative half-weight one-sample boundary overlap.

This fixture covers numerical PHY stages, not RF spectral-mask certification.
The published example's MAC header is historical test data, not a Beacon preset.
