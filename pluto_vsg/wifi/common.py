"""Shared Non-HT OFDM definitions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LegacyRate:
    rate_bits: tuple[int, int, int, int]
    modulation: str
    coding_rate: str
    n_bpsc: int
    n_cbps: int
    n_dbps: int


LEGACY_RATES = {
    6: LegacyRate((1, 1, 0, 1), "BPSK", "1/2", 1, 48, 24),
    9: LegacyRate((1, 1, 1, 1), "BPSK", "3/4", 1, 48, 36),
    12: LegacyRate((0, 1, 0, 1), "QPSK", "1/2", 2, 96, 48),
    18: LegacyRate((0, 1, 1, 1), "QPSK", "3/4", 2, 96, 72),
    24: LegacyRate((1, 0, 0, 1), "16QAM", "1/2", 4, 192, 96),
    36: LegacyRate((1, 0, 1, 1), "16QAM", "3/4", 4, 192, 144),
    48: LegacyRate((0, 0, 0, 1), "64QAM", "2/3", 6, 288, 192),
    54: LegacyRate((0, 0, 1, 1), "64QAM", "3/4", 6, 288, 216),
}
