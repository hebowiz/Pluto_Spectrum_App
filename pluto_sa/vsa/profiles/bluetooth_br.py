"""Bluetooth Basic Rate packet profile and deterministic test vectors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pluto_sa.vsa.demod.gfsk import GFSKDemodulationResult, demodulate_gfsk
from pluto_sa.vsa.demod.fsk_reference import fsk_reference_frequency_levels
from pluto_sa.vsa.model import IQRecording


BLUETOOTH_BR_SYMBOL_RATE_HZ = 1_000_000.0
BLUETOOTH_BR_BT = 0.5
BLUETOOTH_GIAC_LAP = 0x9E8B33
BLUETOOTH_GIAC_SYNC_WORD_HEX = "475c58cc73345e72"
BLUETOOTH_ACCESS_CODE_BITS = 72
BLUETOOTH_HEADER_AIR_BITS = 54


def _bits_from_hex_msb(value: str) -> np.ndarray:
    return np.asarray(
        [int(bit) for digit in value for bit in f"{int(digit, 16):04b}"],
        dtype=np.uint8,
    )


def _bits_to_int_lsb(bits: np.ndarray) -> int:
    return sum(int(bit) << index for index, bit in enumerate(bits))


def _bits_to_int_msb(bits: np.ndarray) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def _reverse_byte(value: int) -> int:
    return int(f"{int(value) & 0xFF:08b}"[::-1], 2)


def _polynomial_remainder(value: int, divisor: int) -> int:
    remainder = int(value)
    while remainder.bit_length() >= divisor.bit_length():
        remainder ^= divisor << (remainder.bit_length() - divisor.bit_length())
    return remainder


def access_code_bits(lap: int, *, include_trailer: bool = True) -> np.ndarray:
    """Generate an access code from a 24-bit LAP in over-the-air order."""
    if not 0 <= int(lap) <= 0xFFFFFF:
        raise ValueError("lap must be a 24-bit value")
    lap_bits = [(int(lap) >> index) & 1 for index in range(24)]
    barker = [0, 0, 1, 1, 0, 1] if lap_bits[23] == 0 else [1, 1, 0, 0, 1, 0]
    information = np.asarray(lap_bits + barker, dtype=np.uint8)
    pn_overlay = _bits_from_hex_msb("3F2A33DD69B121C1")
    covered = information ^ pn_overlay[34:]
    information_polynomial = sum(
        int(bit) << index for index, bit in enumerate(covered)
    )
    primitive_bch = 0x37CD0EB67
    generator = primitive_bch ^ (primitive_bch << 1)
    parity = _polynomial_remainder(information_polynomial << 34, generator)
    codeword = parity | (information_polynomial << 34)
    codeword_bits = np.asarray(
        [(codeword >> index) & 1 for index in range(64)], dtype=np.uint8
    )
    sync_word = codeword_bits ^ pn_overlay
    preamble = _bits_from_hex_msb("a" if sync_word[0] else "5")
    if not include_trailer:
        return np.concatenate((preamble, sync_word))
    trailer = _bits_from_hex_msb("5" if sync_word[-1] else "a")
    return np.concatenate((preamble, sync_word, trailer))


def giac_access_code_bits(*, include_trailer: bool = True) -> np.ndarray:
    return access_code_bits(BLUETOOTH_GIAC_LAP, include_trailer=include_trailer)


def whitening_sequence(clock_6_1: int, count: int) -> np.ndarray:
    """Generate BR/EDR whitening bits for a six-bit native-clock value."""
    if not 0 <= int(clock_6_1) <= 0x3F:
        raise ValueError("clock_6_1 must be a six-bit value")
    if int(count) < 0:
        raise ValueError("count must be non-negative")
    # State order is D7..D1. The fixed extension bit occupies D7 and
    # CLK_6..CLK_1 occupy D6..D1, matching the Core sample-data notation.
    state = np.asarray(
        [1] + [(int(clock_6_1) >> shift) & 1 for shift in range(5, -1, -1)],
        dtype=np.uint8,
    )
    output = np.empty(int(count), dtype=np.uint8)
    for index in range(int(count)):
        output[index] = state[0]
        state = np.asarray(
            [state[1], state[2], state[3] ^ state[0], state[4], state[5], state[6], state[0]],
            dtype=np.uint8,
        )
    return output


def header_error_check(data_10_bits: np.ndarray, uap: int) -> int:
    """Return the transmitted HEC byte for ten header data bits."""
    bits = np.asarray(data_10_bits, dtype=np.uint8)
    if bits.shape != (10,) or np.any(bits > 1):
        raise ValueError("data_10_bits must contain exactly ten binary bits")
    if not 0 <= int(uap) <= 0xFF:
        raise ValueError("uap must be an eight-bit value")
    register = int(uap)
    for bit in bits:
        feedback = ((register >> 7) & 1) ^ int(bit)
        register = (register << 1) & 0xFF
        if feedback:
            register ^= 0xA7
    return _reverse_byte(register)


def fec13_encode(bits: np.ndarray) -> np.ndarray:
    values = np.asarray(bits, dtype=np.uint8)
    if values.ndim != 1 or np.any(values > 1):
        raise ValueError("bits must be a one-dimensional binary array")
    return np.repeat(values, 3)


def fec13_decode(bits: np.ndarray) -> tuple[np.ndarray, int]:
    values = np.asarray(bits, dtype=np.uint8)
    if values.ndim != 1 or values.size % 3 or np.any(values > 1):
        raise ValueError("rate 1/3 FEC input must contain complete binary triplets")
    triplets = values.reshape(-1, 3)
    decoded = (np.sum(triplets, axis=1) >= 2).astype(np.uint8)
    corrected = int(np.count_nonzero(np.any(triplets != decoded[:, None], axis=1)))
    return decoded, corrected


@dataclass(frozen=True)
class BluetoothHeader:
    lt_addr: int
    packet_type: int
    flow: int
    arqn: int
    seqn: int
    hec: int
    hec_valid: bool | None
    uap: int | None
    clock_6_1: int | None
    whitening_enabled: bool
    corrected_fec_triplets: int
    bits: np.ndarray

    def __post_init__(self) -> None:
        owned = np.array(self.bits, dtype=np.uint8, copy=True)
        owned.flags.writeable = False
        object.__setattr__(self, "bits", owned)


@dataclass(frozen=True)
class BluetoothBRPacketResult:
    demodulation: GFSKDemodulationResult
    access_code_bits: np.ndarray
    header_air_bits: np.ndarray
    header: BluetoothHeader | None
    payload_bits: np.ndarray

    def __post_init__(self) -> None:
        for name in ("access_code_bits", "header_air_bits", "payload_bits"):
            owned = np.array(getattr(self, name), dtype=np.uint8, copy=True)
            owned.flags.writeable = False
            object.__setattr__(self, name, owned)


@dataclass(frozen=True)
class BluetoothACLPayload:
    logical_channel: int
    flow: int
    length_bytes: int
    body: bytes
    received_crc: bytes
    expected_crc: bytes
    crc_valid: bool


@dataclass(frozen=True)
class BluetoothDH1Candidate:
    header: BluetoothHeader
    payload: BluetoothACLPayload


@dataclass(frozen=True)
class PRBS9Match:
    bit_errors: int
    bit_count: int
    phase: int
    inverted: bool
    time_reversed: bool

    @property
    def ber(self) -> float:
        return self.bit_errors / self.bit_count if self.bit_count else 0.0


def decode_header_air_bits(
    header_air_bits: np.ndarray,
    *,
    uap: int | None,
    clock_6_1: int | None,
    whitening_enabled: bool = True,
) -> BluetoothHeader:
    """Decode one complete 54-air-bit BR packet header."""
    values = np.asarray(header_air_bits, dtype=np.uint8)
    if values.shape != (BLUETOOTH_HEADER_AIR_BITS,) or np.any(values > 1):
        raise ValueError("header_air_bits must contain exactly 54 binary bits")
    if whitening_enabled and clock_6_1 is None:
        raise ValueError("clock_6_1 is required when whitening is enabled")
    fec_bits, corrected = fec13_decode(values)
    if whitening_enabled:
        header_bits = fec_bits ^ whitening_sequence(int(clock_6_1), fec_bits.size)
    else:
        header_bits = fec_bits
    data = header_bits[:10]
    packed = _bits_to_int_lsb(data)
    received_hec = _bits_to_int_msb(header_bits[10:18])
    hec_valid = (
        None if uap is None else header_error_check(data, int(uap)) == received_hec
    )
    return BluetoothHeader(
        lt_addr=packed & 0x7,
        packet_type=(packed >> 3) & 0xF,
        flow=(packed >> 7) & 1,
        arqn=(packed >> 8) & 1,
        seqn=(packed >> 9) & 1,
        hec=received_hec,
        hec_valid=hec_valid,
        uap=None if uap is None else int(uap),
        clock_6_1=None if not whitening_enabled else int(clock_6_1),
        whitening_enabled=bool(whitening_enabled),
        corrected_fec_triplets=corrected,
        bits=header_bits,
    )


def find_header_candidates(
    header_air_bits: np.ndarray,
    *,
    uap: int,
    include_unwhitened: bool = True,
) -> tuple[BluetoothHeader, ...]:
    """Return CLK_6-1/no-whitening candidates whose decoded HEC is valid."""
    candidates = [
        header
        for clock in range(64)
        if (
            header := decode_header_air_bits(
                header_air_bits,
                uap=int(uap),
                clock_6_1=clock,
            )
        ).hec_valid
    ]
    if include_unwhitened:
        unwhitened = decode_header_air_bits(
            header_air_bits,
            uap=int(uap),
            clock_6_1=None,
            whitening_enabled=False,
        )
        if unwhitened.hec_valid:
            candidates.append(unwhitened)
    return tuple(candidates)


def payload_crc_bytes(bits: np.ndarray, uap: int) -> bytes:
    """Calculate the transmitted BR/EDR payload CRC bytes."""
    values = np.asarray(bits, dtype=np.uint8)
    if values.ndim != 1 or np.any(values > 1):
        raise ValueError("bits must be a one-dimensional binary array")
    if not 0 <= int(uap) <= 0xFF:
        raise ValueError("uap must be an eight-bit value")
    register = int(uap)
    for bit in values:
        feedback = ((register >> 15) & 1) ^ int(bit)
        register = (register << 1) & 0xFFFF
        if feedback:
            register ^= 0x1021
    return bytes((_reverse_byte(register >> 8), _reverse_byte(register)))


def _air_bits_to_bytes(bits: np.ndarray) -> bytes:
    values = np.asarray(bits, dtype=np.uint8)
    if values.ndim != 1 or values.size % 8 or np.any(values > 1):
        raise ValueError("air bits must contain complete binary octets")
    return bytes(
        _bits_to_int_lsb(values[start : start + 8])
        for start in range(0, values.size, 8)
    )


def decode_dh1_payload(payload_bits: np.ndarray, *, uap: int) -> BluetoothACLPayload:
    """Decode an unwhitened DH1 payload header, body and CRC."""
    values = np.asarray(payload_bits, dtype=np.uint8)
    if values.ndim != 1 or np.any(values > 1) or values.size < 24:
        raise ValueError("DH1 payload must contain a header and CRC")
    payload_header = values[:8]
    logical_channel = _bits_to_int_lsb(payload_header[:2])
    flow = int(payload_header[2])
    length_bytes = _bits_to_int_lsb(payload_header[3:8])
    if length_bytes > 27:
        raise ValueError("DH1 payload length exceeds 27 bytes")
    payload_stop = 8 + length_bytes * 8
    crc_stop = payload_stop + 16
    if values.size < crc_stop:
        raise ValueError("DH1 payload is shorter than its length field")
    body_bits = values[8:payload_stop]
    received_crc = _air_bits_to_bytes(values[payload_stop:crc_stop])
    expected_crc = payload_crc_bytes(values[:payload_stop], int(uap))
    return BluetoothACLPayload(
        logical_channel=logical_channel,
        flow=flow,
        length_bytes=length_bytes,
        body=_air_bits_to_bytes(body_bits),
        received_crc=received_crc,
        expected_crc=expected_crc,
        crc_valid=received_crc == expected_crc,
    )


def find_dh1_candidates(
    header_air_bits: np.ndarray,
    payload_air_bits: np.ndarray,
    *,
    uaps: range | tuple[int, ...] = range(256),
    include_unwhitened: bool = True,
    require_crc: bool = True,
) -> tuple[BluetoothDH1Candidate, ...]:
    """Find HEC+CRC-valid DH1 candidates across unknown UAP/clock settings."""
    payload_air = np.asarray(payload_air_bits, dtype=np.uint8)
    matches: list[BluetoothDH1Candidate] = []
    for uap in uaps:
        for header in find_header_candidates(
            header_air_bits,
            uap=int(uap),
            include_unwhitened=include_unwhitened,
        ):
            if header.packet_type != 4:
                continue
            if header.whitening_enabled:
                whitening = whitening_sequence(
                    int(header.clock_6_1), 18 + payload_air.size
                )
                payload = payload_air ^ whitening[18:]
            else:
                payload = payload_air
            try:
                decoded = decode_dh1_payload(payload, uap=int(uap))
            except ValueError:
                continue
            if decoded.crc_valid or not require_crc:
                matches.append(BluetoothDH1Candidate(header=header, payload=decoded))
    return tuple(matches)


def prbs9_period() -> np.ndarray:
    """Return one 511-bit period of x^9 + x^5 + 1 from an all-one state."""
    state = 0x1FF
    output = np.empty(511, dtype=np.uint8)
    for index in range(output.size):
        output[index] = state & 1
        feedback = ((state >> 0) ^ (state >> 4)) & 1
        state = (state >> 1) | (feedback << 8)
    output.flags.writeable = False
    return output


def match_prbs9(bits: np.ndarray) -> PRBS9Match:
    """Find the best cyclic PRBS-9 phase, polarity and time direction."""
    values = np.asarray(bits, dtype=np.uint8)
    if values.ndim != 1 or values.size == 0 or np.any(values > 1):
        raise ValueError("bits must be a non-empty one-dimensional binary array")
    base = prbs9_period()
    indices = np.arange(values.size, dtype=np.int64)
    best: PRBS9Match | None = None
    for time_reversed in (False, True):
        sequence = base[::-1] if time_reversed else base
        for inverted in (False, True):
            candidate_sequence = sequence ^ int(inverted)
            for phase in range(sequence.size):
                expected = candidate_sequence[(indices + phase) % sequence.size]
                errors = int(np.count_nonzero(values != expected))
                candidate = PRBS9Match(
                    bit_errors=errors,
                    bit_count=int(values.size),
                    phase=phase,
                    inverted=inverted,
                    time_reversed=time_reversed,
                )
                if best is None or candidate.bit_errors < best.bit_errors:
                    best = candidate
    assert best is not None
    return best


def _header_data_bits(
    *, lt_addr: int, packet_type: int, flow: int, arqn: int, seqn: int
) -> np.ndarray:
    if not 0 <= int(lt_addr) <= 7:
        raise ValueError("lt_addr must fit in three bits")
    if not 0 <= int(packet_type) <= 15:
        raise ValueError("packet_type must fit in four bits")
    for name, value in (("flow", flow), ("arqn", arqn), ("seqn", seqn)):
        if int(value) not in (0, 1):
            raise ValueError(f"{name} must be zero or one")
    packed = (
        int(lt_addr)
        | (int(packet_type) << 3)
        | (int(flow) << 7)
        | (int(arqn) << 8)
        | (int(seqn) << 9)
    )
    return np.asarray([(packed >> index) & 1 for index in range(10)], dtype=np.uint8)


def build_packet_bits(
    *,
    clock_6_1: int,
    uap: int,
    payload_bits: np.ndarray | None = None,
    lt_addr: int = 1,
    packet_type: int = 4,
    flow: int = 1,
    arqn: int = 0,
    seqn: int = 0,
    lap: int | None = None,
) -> np.ndarray:
    """Build a BR-like bitstream for receiver tests (uncoded payload)."""
    payload = np.asarray(
        np.empty(0, dtype=np.uint8) if payload_bits is None else payload_bits,
        dtype=np.uint8,
    )
    if payload.ndim != 1 or np.any(payload > 1):
        raise ValueError("payload_bits must be a one-dimensional binary array")
    data = _header_data_bits(
        lt_addr=lt_addr,
        packet_type=packet_type,
        flow=flow,
        arqn=arqn,
        seqn=seqn,
    )
    hec = header_error_check(data, uap)
    hec_bits = _bits_from_hex_msb(f"{hec:02x}")
    header = np.concatenate((data, hec_bits))
    whitening = whitening_sequence(clock_6_1, header.size + payload.size)
    header_air = fec13_encode(header ^ whitening[: header.size])
    payload_air = payload ^ whitening[header.size :]
    access = (
        giac_access_code_bits()
        if lap is None
        else access_code_bits(int(lap))
    )
    return np.concatenate((access, header_air, payload_air))


def modulate_packet_bits(
    bits: np.ndarray,
    *,
    sample_rate_hz: float = 8_000_000.0,
    frequency_deviation_hz: float = 160_000.0,
    carrier_frequency_offset_hz: float = 0.0,
    carrier_frequency_drift_hz_per_s: float = 0.0,
    prefix_samples: int = 0,
    suffix_samples: int = 0,
    snr_db: float | None = None,
    seed: int = 1,
) -> np.ndarray:
    """Generate a continuous-phase BT=0.5 test waveform for packet tests."""
    values = np.asarray(bits, dtype=np.uint8)
    if values.ndim != 1 or np.any(values > 1):
        raise ValueError("bits must be a one-dimensional binary array")
    samples_per_symbol = int(round(float(sample_rate_hz) / BLUETOOTH_BR_SYMBOL_RATE_HZ))
    if samples_per_symbol < 4 or not np.isclose(
        samples_per_symbol * BLUETOOTH_BR_SYMBOL_RATE_HZ, sample_rate_hz
    ):
        raise ValueError("sample_rate_hz must be an integer multiple of 1 MHz")
    shaped = fsk_reference_frequency_levels(
        values,
        samples_per_symbol=samples_per_symbol,
        transmit_gaussian_bt=BLUETOOTH_BR_BT,
    )
    packet_time_s = np.arange(shaped.size, dtype=np.float64) / float(sample_rate_hz)
    frequency = (
        shaped * float(frequency_deviation_hz)
        + float(carrier_frequency_offset_hz)
        + float(carrier_frequency_drift_hz_per_s) * packet_time_s
    )
    phase = 2.0 * np.pi * np.cumsum(frequency) / float(sample_rate_hz)
    packet = np.exp(1j * phase)
    rng = np.random.default_rng(int(seed))
    prefix = np.zeros(max(0, int(prefix_samples)), dtype=np.complex128)
    suffix = np.zeros(max(0, int(suffix_samples)), dtype=np.complex128)
    iq = np.concatenate((prefix, packet, suffix))
    if snr_db is not None:
        noise_sigma = np.sqrt(0.5 / (10.0 ** (float(snr_db) / 10.0)))
        noise = noise_sigma * (
            rng.standard_normal(iq.size) + 1j * rng.standard_normal(iq.size)
        )
        iq = iq + noise
    return iq.astype(np.complex64)


class BluetoothBRProfile:
    """Decode a known-access-code Basic Rate capture."""

    def __init__(self, access_bits: np.ndarray | None = None) -> None:
        self.access_bits = np.array(
            giac_access_code_bits() if access_bits is None else access_bits,
            dtype=np.uint8,
            copy=True,
        )

    def analyze(
        self,
        recording: IQRecording,
        *,
        clock_6_1: int | None = None,
        uap: int | None = None,
        whitening_enabled: bool = True,
        minimum_correlation: float = 0.65,
    ) -> BluetoothBRPacketResult:
        demodulation = demodulate_gfsk(
            recording.iq,
            sample_rate_hz=recording.sample_rate_hz,
            access_bits=self.access_bits,
            symbol_rate_hz=BLUETOOTH_BR_SYMBOL_RATE_HZ,
            minimum_correlation=minimum_correlation,
        )
        packet_bits = demodulation.bits
        access_stop = self.access_bits.size
        header_stop = access_stop + BLUETOOTH_HEADER_AIR_BITS
        access = packet_bits[:access_stop]
        header_air = packet_bits[access_stop:header_stop]
        payload_air = packet_bits[header_stop:]
        header: BluetoothHeader | None = None
        payload = payload_air
        can_decode_header = header_air.size == BLUETOOTH_HEADER_AIR_BITS and (
            clock_6_1 is not None or not whitening_enabled
        )
        if can_decode_header:
            header = decode_header_air_bits(
                header_air,
                uap=uap,
                clock_6_1=clock_6_1,
                whitening_enabled=whitening_enabled,
            )
            if whitening_enabled:
                whitening = whitening_sequence(
                    int(clock_6_1), 18 + payload_air.size
                )
                payload = payload_air ^ whitening[18:]
            else:
                payload = payload_air
        return BluetoothBRPacketResult(
            demodulation=demodulation,
            access_code_bits=access,
            header_air_bits=header_air,
            header=header,
            payload_bits=payload,
        )
