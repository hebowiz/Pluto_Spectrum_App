"""Blind/coarse acquisition helpers for Bluetooth General Packet mode.

The helpers in this module recover packet identity only.  They intentionally
do not provide final symbol timing, CFO, drift, or measurement values; callers
must reconstruct the exact known pattern and run the existing fine analyzer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pluto_sa.vsa.demod.gfsk import _detect_bursts, prepare_fsk_frequency
from pluto_sa.vsa.model import IQRecording
from pluto_sa.vsa.profiles.bluetooth_br import (
    BLUETOOTH_ACCESS_CODE_BITS,
    access_code_bits,
    recover_lap_from_access_code_bits,
)


_COARSE_SAMPLES_PER_SYMBOL = 8
_LE_ADVERTISING_ACCESS_ADDRESS = 0x8E89BED6
_LE_STANDARD_ACCESS_ADDRESSES = {
    _LE_ADVERTISING_ACCESS_ADDRESS,
    0x71764129,  # Uncoded RF PHY Test Packet
}


@dataclass(frozen=True)
class ClassicAcquisitionCandidate:
    start_sample: int
    lap: int
    access_bit_errors: int
    correlation: float
    polarity_inverted: bool


@dataclass(frozen=True)
class LEAcquisitionCandidate:
    start_sample: int
    phy: str
    access_address: int
    sync_bit_errors: int
    correlation: float
    polarity_inverted: bool
    preamble_first_bit: int


def _bits_to_int_lsb(bits: np.ndarray) -> int:
    return sum(int(bit) << index for index, bit in enumerate(bits))


def _maximum_run(bits: np.ndarray) -> int:
    values = np.asarray(bits, dtype=np.uint8)
    if values.size == 0:
        return 0
    changes = np.flatnonzero(values[1:] != values[:-1]) + 1
    edges = np.concatenate(([0], changes, [values.size]))
    return int(np.max(np.diff(edges)))


def valid_le_access_address(access_address: int) -> bool:
    """Apply the Core constraints usable without connection state."""

    value = int(access_address) & 0xFFFFFFFF
    if value == _LE_ADVERTISING_ACCESS_ADDRESS:
        return True
    bits = np.asarray([(value >> index) & 1 for index in range(32)], dtype=np.uint8)
    octets = value.to_bytes(4, byteorder="little")
    if len(set(octets)) == 1:
        return False
    if _maximum_run(bits) > 6:
        return False
    if int(np.count_nonzero(bits[1:] != bits[:-1])) > 24:
        return False
    if int(np.count_nonzero(bits[27:32] != bits[26:31])) < 2:
        return False
    if int((value ^ _LE_ADVERTISING_ACCESS_ADDRESS).bit_count()) == 1:
        return False
    return True


def _le_sync_bits(
    phy: str,
    access_address: int,
    *,
    preamble_first_bit: int | None = None,
) -> np.ndarray:
    access = np.asarray(
        [(int(access_address) >> index) & 1 for index in range(32)],
        dtype=np.uint8,
    )
    preamble_count = 16 if "2M" in str(phy).upper().replace(" ", "") else 8
    if preamble_first_bit is not None:
        first = int(preamble_first_bit) & 1
        preamble = (first + np.arange(preamble_count, dtype=np.uint8)) & 1
        return np.concatenate((preamble, access))
    if int(access_address) == 0x71764129:
        preamble = np.resize(np.asarray([1, 0], dtype=np.uint8), preamble_count)
        return np.concatenate((preamble, access))
    preamble = (1 - int(access[0]) + np.arange(preamble_count, dtype=np.uint8)) & 1
    return np.concatenate((preamble, access))


def _coarse_bitstreams(
    recording: IQRecording,
    *,
    symbol_rate_hz: float,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    frequency_hz, analysis_rate_hz = prepare_fsk_frequency(
        recording.iq,
        sample_rate_hz=recording.sample_rate_hz,
        symbol_rate_hz=float(symbol_rate_hz),
        gaussian_bt=0.5,
        samples_per_symbol=_COARSE_SAMPLES_PER_SYMBOL,
    )
    bursts = _detect_bursts(
        np.asarray(recording.iq),
        sample_rate_hz=recording.sample_rate_hz,
        symbol_rate_hz=float(symbol_rate_hz),
        minimum_symbols=32,
    )
    if not bursts:
        bursts = ((0, recording.sample_count),)
    scale_to_analysis = analysis_rate_hz / recording.sample_rate_hz
    scale_to_source = recording.sample_rate_hz / analysis_rate_hz
    streams: list[tuple[np.ndarray, np.ndarray]] = []
    for source_start, source_stop in bursts:
        start = max(0, int(np.floor(source_start * scale_to_analysis)))
        stop = min(frequency_hz.size, int(np.ceil(source_stop * scale_to_analysis)))
        for phase in range(_COARSE_SAMPLES_PER_SYMBOL):
            first = start + phase
            count = (stop - first) // _COARSE_SAMPLES_PER_SYMBOL
            if count < 40:
                continue
            blocks = frequency_hz[
                first : first + count * _COARSE_SAMPLES_PER_SYMBOL
            ].reshape(count, _COARSE_SAMPLES_PER_SYMBOL)
            edge = max(1, _COARSE_SAMPLES_PER_SYMBOL // 8)
            values = np.mean(blocks[:, edge:-edge], axis=1)
            # The packet payload need not contain equal numbers of ones and
            # zeros, so its median is not a reliable discriminator.  The
            # midpoint of robust low/high clusters preserves an unknown CFO
            # while avoiding a data-pattern-dependent decision threshold.
            low, high = np.percentile(values, (10.0, 90.0))
            threshold = 0.5 * float(low + high)
            bits = (values >= threshold).astype(np.uint8)
            positions = (
                first
                + np.arange(count, dtype=np.float64) * _COARSE_SAMPLES_PER_SYMBOL
            ) * scale_to_source
            streams.append((bits, positions))
    return tuple(streams)


def _deduplicate_classic(
    candidates: list[ClassicAcquisitionCandidate],
    *,
    tolerance_samples: int,
    lap_hint: int | None,
) -> tuple[ClassicAcquisitionCandidate, ...]:
    selected: list[ClassicAcquisitionCandidate] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (
            item.start_sample,
            0 if lap_hint is not None and item.lap == (int(lap_hint) & 0xFFFFFF) else 1,
            item.access_bit_errors,
        ),
    ):
        duplicate = next(
            (
                index
                for index, existing in enumerate(selected)
                if existing.lap == candidate.lap
                and abs(existing.start_sample - candidate.start_sample)
                <= tolerance_samples
            ),
            None,
        )
        if duplicate is None:
            selected.append(candidate)
        elif candidate.access_bit_errors < selected[duplicate].access_bit_errors:
            selected[duplicate] = candidate
    return tuple(sorted(selected, key=lambda item: item.start_sample))


def detect_classic_identities(
    recording: IQRecording,
    *,
    lap_hint: int | None = None,
    maximum_access_bit_errors: int = 4,
    max_candidates: int = 64,
) -> tuple[ClassicAcquisitionCandidate, ...]:
    """Recover Classic LAP candidates without using a configured pattern."""

    found: list[ClassicAcquisitionCandidate] = []
    for stream, positions in _coarse_bitstreams(
        recording, symbol_rate_hz=1_000_000.0
    ):
        for start in range(0, stream.size - BLUETOOTH_ACCESS_CODE_BITS + 1):
            window = stream[start : start + BLUETOOTH_ACCESS_CODE_BITS]
            for inverted in (False, True):
                candidate_bits = window ^ int(inverted)
                recovered = recover_lap_from_access_code_bits(
                    candidate_bits,
                    maximum_bit_errors=maximum_access_bit_errors,
                )
                if recovered is None:
                    continue
                lap, errors = recovered
                found.append(
                    ClassicAcquisitionCandidate(
                        start_sample=int(round(float(positions[start]))),
                        lap=lap,
                        access_bit_errors=errors,
                        correlation=1.0 - errors / float(BLUETOOTH_ACCESS_CODE_BITS),
                        polarity_inverted=inverted,
                    )
                )
    tolerance = max(1, int(round(recording.sample_rate_hz / 1_000_000.0)))
    return _deduplicate_classic(
        found,
        tolerance_samples=tolerance,
        lap_hint=lap_hint,
    )[: max(1, int(max_candidates))]


def _deduplicate_le(
    candidates: list[LEAcquisitionCandidate],
    *,
    tolerance_samples: int,
    access_address_hint: int | None,
) -> tuple[LEAcquisitionCandidate, ...]:
    selected: list[LEAcquisitionCandidate] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (
            item.start_sample,
            item.polarity_inverted,
            0
            if access_address_hint is not None
            and item.access_address == (int(access_address_hint) & 0xFFFFFFFF)
            else 1,
            item.sync_bit_errors,
        ),
    ):
        duplicate = next(
            (
                index
                for index, existing in enumerate(selected)
                if existing.phy == candidate.phy
                and existing.access_address == candidate.access_address
                and abs(existing.start_sample - candidate.start_sample)
                <= tolerance_samples
            ),
            None,
        )
        if duplicate is None:
            selected.append(candidate)
        elif candidate.sync_bit_errors < selected[duplicate].sync_bit_errors:
            selected[duplicate] = candidate
    return tuple(sorted(selected, key=lambda item: item.start_sample))


def detect_le_identities(
    recording: IQRecording,
    *,
    phy: str,
    access_address_hint: int | None = None,
    maximum_sync_bit_errors: int = 2,
    max_candidates: int = 64,
) -> tuple[LEAcquisitionCandidate, ...]:
    """Recover LE Access Addresses from preamble plus 32-bit AA candidates."""

    phy_name = str(phy)
    symbol_rate_hz = 2_000_000.0 if "2M" in phy_name.upper().replace(" ", "") else 1_000_000.0
    preamble_count = 16 if symbol_rate_hz == 2_000_000.0 else 8
    sync_count = preamble_count + 32
    found: list[LEAcquisitionCandidate] = []
    for stream, positions in _coarse_bitstreams(
        recording, symbol_rate_hz=symbol_rate_hz
    ):
        for start in range(0, stream.size - sync_count + 1):
            window = stream[start : start + sync_count]
            for inverted in (False, True):
                candidate_bits = window ^ int(inverted)
                access_address = _bits_to_int_lsb(candidate_bits[preamble_count:])
                if not valid_le_access_address(access_address):
                    continue
                expected = _le_sync_bits(phy_name, access_address)
                alternate = _le_sync_bits(
                    phy_name,
                    access_address,
                    preamble_first_bit=1 - int(expected[0]),
                )
                standard_errors = int(np.count_nonzero(candidate_bits != expected))
                alternate_errors = int(np.count_nonzero(candidate_bits != alternate))
                expected_first_bit = int(expected[0])
                if alternate_errors < standard_errors:
                    errors = alternate_errors
                    expected_first_bit = int(alternate[0])
                else:
                    errors = standard_errors
                if errors > max(0, int(maximum_sync_bit_errors)):
                    continue
                found.append(
                    LEAcquisitionCandidate(
                        start_sample=int(round(float(positions[start]))),
                        phy=phy_name,
                        access_address=access_address,
                        sync_bit_errors=errors,
                        correlation=1.0 - errors / float(sync_count),
                        polarity_inverted=inverted,
                        preamble_first_bit=expected_first_bit,
                    )
                )
    tolerance = max(1, int(round(recording.sample_rate_hz / symbol_rate_hz)))
    deduplicated = _deduplicate_le(
        found,
        tolerance_samples=tolerance,
        access_address_hint=access_address_hint,
    )
    # An arbitrary 32-bit payload window can itself look like a legal Access
    # Address.  A real LE sync is the first qualifying preamble+AA at the
    # leading edge of each detected RF burst.  Keep one polarity/identity per
    # burst and leave exact timing to the known-pattern fine synchronizer.
    bursts = _detect_bursts(
        np.asarray(recording.iq),
        sample_rate_hz=recording.sample_rate_hz,
        symbol_rate_hz=symbol_rate_hz,
        minimum_symbols=32,
    )
    selected: list[LEAcquisitionCandidate] = []
    for burst_start, burst_stop in bursts or ((0, recording.sample_count),):
        inside = [
            candidate
            for candidate in deduplicated
            if burst_start <= candidate.start_sample < burst_stop
        ]
        if not inside:
            continue
        best_errors = min(candidate.sync_bit_errors for candidate in inside)
        error_matched = [
            candidate
            for candidate in inside
            if candidate.sync_bit_errors == best_errors
        ]
        preferred_polarity = min(
            candidate.polarity_inverted for candidate in error_matched
        )
        error_matched = [
            candidate
            for candidate in error_matched
            if candidate.polarity_inverted is preferred_polarity
        ]
        hinted = [
            candidate
            for candidate in error_matched
            if access_address_hint is not None
            and candidate.access_address
            == (int(access_address_hint) & 0xFFFFFFFF)
        ]
        if hinted:
            error_matched = hinted
        else:
            leading_edge = min(
                candidate.start_sample for candidate in error_matched
            )
            standardized = [
                candidate
                for candidate in error_matched
                if candidate.access_address in _LE_STANDARD_ACCESS_ADDRESSES
                and candidate.start_sample
                <= leading_edge + preamble_count * tolerance
            ]
            if standardized:
                error_matched = standardized
        earliest = min(candidate.start_sample for candidate in error_matched)
        leading = [
            candidate
            for candidate in error_matched
            if candidate.start_sample <= earliest + tolerance
        ]
        selected.append(
            min(
                leading,
                key=lambda candidate: (
                    candidate.sync_bit_errors,
                    candidate.polarity_inverted,
                    0
                    if access_address_hint is not None
                    and candidate.access_address
                    == (int(access_address_hint) & 0xFFFFFFFF)
                    else 1,
                ),
            )
        )
    return tuple(sorted(selected, key=lambda item: item.start_sample))[
        : max(1, int(max_candidates))
    ]


__all__ = [
    "ClassicAcquisitionCandidate",
    "LEAcquisitionCandidate",
    "detect_classic_identities",
    "detect_le_identities",
    "valid_le_access_address",
]
