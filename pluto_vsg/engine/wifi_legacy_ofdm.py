"""IEEE 802.11 Non-HT OFDM 20 MHz waveform generation."""

from __future__ import annotations

import math

import numpy as np

from pluto_vsg.engine.base import FieldBoundary, GenerationResult
from pluto_vsg.model import WiFiScramblerSeedMode, WaveformProject, validate_project
from pluto_vsg.rf_level import iq_level_metadata, measure_iq_levels
from pluto_vsg.wifi.common import LEGACY_RATES, LegacyRate
from pluto_vsg.wifi.mac import build_psdu, bytes_to_air_bits


DATA_SUBCARRIERS = tuple(k for k in range(-26, 27) if k and k not in {-21, -7, 7, 21})
PILOT_SUBCARRIERS = (-21, -7, 7, 21)
PILOT_BASE = np.asarray((1.0, 1.0, 1.0, -1.0))
LTF_VALUES = np.asarray((
    1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1,
    1, -1, 1, -1, 1, 1, 1, 1, 0, 1, -1, -1, 1, 1, -1, 1, -1, 1,
    -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1,
), dtype=np.complex128)
STF_TONES = {
    -24: 1 + 1j, -20: -1 - 1j, -16: 1 + 1j, -12: -1 - 1j,
    -8: -1 - 1j, -4: 1 + 1j, 4: -1 - 1j, 8: -1 - 1j,
    12: 1 + 1j, 16: 1 + 1j, 20: 1 + 1j, 24: 1 + 1j,
}


def _parity(value: int) -> int:
    return int(value).bit_count() & 1


def bcc_encode(bits: np.ndarray) -> np.ndarray:
    register = 0
    output = np.empty(int(bits.size) * 2, dtype=np.uint8)
    for index, bit in enumerate(np.asarray(bits, dtype=np.uint8)):
        register = ((register << 1) | int(bit)) & 0x7F
        # The standard names the generators 133/171 octal with the newest
        # bit at the opposite end of the shift register.  With the LSB-newest
        # representation used here the bit-reversed masks are 155/117.
        output[2 * index] = _parity(register & 0o155)
        output[2 * index + 1] = _parity(register & 0o117)
    return output


def puncture(bits: np.ndarray, coding_rate: str) -> np.ndarray:
    masks = {
        "1/2": np.asarray((1, 1), dtype=bool),
        "2/3": np.asarray((1, 1, 1, 0), dtype=bool),
        "3/4": np.asarray((1, 1, 1, 0, 0, 1), dtype=bool),
    }
    mask = masks[coding_rate]
    keep = np.resize(mask, bits.size)
    return np.asarray(bits, dtype=np.uint8)[keep]


def interleave(bits: np.ndarray, n_cbps: int, n_bpsc: int) -> np.ndarray:
    values = np.asarray(bits, dtype=np.uint8)
    if values.size != n_cbps:
        raise ValueError("Interleaver input must contain exactly N_CBPS bits")
    s = max(n_bpsc // 2, 1)
    output = np.empty_like(values)
    for k in range(n_cbps):
        i = (n_cbps // 16) * (k % 16) + k // 16
        j = s * (i // s) + (i + n_cbps - (16 * i) // n_cbps) % s
        output[j] = values[k]
    return output


def map_constellation(bits: np.ndarray, n_bpsc: int) -> np.ndarray:
    groups = np.asarray(bits, dtype=np.uint8).reshape(-1, n_bpsc)
    if n_bpsc == 1:
        return (2.0 * groups[:, 0] - 1.0).astype(np.complex128)
    if n_bpsc == 2:
        return ((2 * groups[:, 0].astype(int) - 1) + 1j * (2 * groups[:, 1].astype(int) - 1)) / math.sqrt(2)
    def axis(values: np.ndarray) -> np.ndarray:
        sign = 2 * values[:, 0].astype(int) - 1
        if n_bpsc == 4:
            magnitude = 3 - 2 * values[:, 1].astype(int)
        else:
            b1, b2 = values[:, 1].astype(int), values[:, 2].astype(int)
            magnitude = (1 - b1) * (7 - 2 * b2) + b1 * (1 + 2 * b2)
        return sign * magnitude
    half = n_bpsc // 2
    scale = math.sqrt(10 if n_bpsc == 4 else 42)
    return (axis(groups[:, :half]) + 1j * axis(groups[:, half:])) / scale


def scramble(bits: np.ndarray, seed: int) -> np.ndarray:
    state = [(int(seed) >> index) & 1 for index in range(7)]
    output = np.empty_like(np.asarray(bits, dtype=np.uint8))
    for index, bit in enumerate(bits):
        feedback = state[3] ^ state[6]
        output[index] = int(bit) ^ feedback
        state = [feedback, *state[:6]]
    return output


def _pilot_polarities(count: int) -> np.ndarray:
    state = [1] * 7
    values = np.empty(count, dtype=np.float64)
    for index in range(count):
        feedback = state[3] ^ state[6]
        values[index] = 1.0 if feedback == 0 else -1.0
        state = [feedback, *state[:6]]
    return values


def _ifft_symbol(data: np.ndarray, pilot_index: int, oversample: int) -> np.ndarray:
    nfft = 64 * oversample
    bins = np.zeros(nfft, dtype=np.complex128)
    for carrier, value in zip(DATA_SUBCARRIERS, data, strict=True):
        bins[carrier % nfft] = value
    polarity = _pilot_polarities(pilot_index + 1)[-1]
    for carrier, value in zip(PILOT_SUBCARRIERS, PILOT_BASE * polarity, strict=True):
        bins[carrier % nfft] = value
    useful = np.fft.ifft(bins) * nfft / math.sqrt(52)
    return np.concatenate((useful[-16 * oversample :], useful))


def _training_fields(oversample: int) -> tuple[np.ndarray, np.ndarray]:
    nfft = 64 * oversample
    stf_bins = np.zeros(nfft, dtype=np.complex128)
    for carrier, value in STF_TONES.items():
        stf_bins[carrier % nfft] = value * math.sqrt(13 / 6)
    stf_symbol = np.fft.ifft(stf_bins) * nfft / math.sqrt(52)
    short = stf_symbol[: 16 * oversample]
    l_stf = np.tile(short, 10)

    ltf_bins = np.zeros(nfft, dtype=np.complex128)
    for carrier, value in zip(range(-26, 27), LTF_VALUES, strict=True):
        ltf_bins[carrier % nfft] = value
    long_symbol = np.fft.ifft(ltf_bins) * nfft / math.sqrt(52)
    l_ltf = np.concatenate((long_symbol[-32 * oversample :], long_symbol, long_symbol))
    return l_stf, l_ltf


def _l_sig_bits(rate: LegacyRate, length: int) -> np.ndarray:
    bits = list(rate.rate_bits) + [0] + [(length >> index) & 1 for index in range(12)]
    bits.append(sum(bits) & 1)
    bits.extend((0,) * 6)
    return np.asarray(bits, dtype=np.uint8)


def _encode_symbol(input_bits: np.ndarray, rate: LegacyRate) -> np.ndarray:
    return interleave(puncture(bcc_encode(input_bits), rate.coding_rate), rate.n_cbps, rate.n_bpsc)


class WiFiLegacyOFDMWaveformEngine:
    """Generate a standards-structured Non-HT OFDM PPDU and packet schedule."""

    def generate(self, project: WaveformProject) -> GenerationResult:
        issues = validate_project(project)
        if issues:
            raise ValueError("Invalid waveform project: " + "; ".join(f"{i.path}: {i.message}" for i in issues))
        settings = project.wifi
        if settings is None:
            raise ValueError("Wi-Fi settings are required")
        rate = LEGACY_RATES[int(settings.legacy_rate_mbps)]
        oversample = int(settings.oversample_factor)
        expected_sample_rate = 20_000_000.0 * oversample
        if not np.isclose(project.sample_rate_hz, expected_sample_rate):
            raise ValueError("Wi-Fi sample rate must be 20 or 40 MS/s according to the oversample setting")
        psdu = build_psdu(settings)
        if not 1 <= len(psdu) <= 4095:
            raise ValueError("Non-HT OFDM PSDU length must be between 1 and 4095 bytes")
        n_sym = math.ceil((16 + 8 * len(psdu) + 6) / rate.n_dbps)
        n_pad = n_sym * rate.n_dbps - (16 + 8 * len(psdu) + 6)
        data_bits = np.concatenate((np.zeros(16, dtype=np.uint8), bytes_to_air_bits(psdu), np.zeros(6 + n_pad, dtype=np.uint8)))
        seed = (
            int(np.random.default_rng().integers(1, 128))
            if WiFiScramblerSeedMode(settings.scrambler_seed_mode) == WiFiScramblerSeedMode.AUTO
            else int(settings.scrambler_seed)
        )
        scrambled = scramble(data_bits, seed)
        tail_start = 16 + 8 * len(psdu)
        scrambled[tail_start : tail_start + 6] = 0

        l_stf, l_ltf = _training_fields(oversample)
        sig_rate = LEGACY_RATES[6]
        l_sig_bits = _l_sig_bits(rate, len(psdu))
        l_sig = _ifft_symbol(map_constellation(_encode_symbol(l_sig_bits, sig_rate), 1), 0, oversample)
        coded = puncture(bcc_encode(scrambled), rate.coding_rate)
        symbols = []
        interleaved_symbols = []
        for index in range(n_sym):
            chunk = coded[index * rate.n_cbps : (index + 1) * rate.n_cbps]
            interleaved_bits = interleave(chunk, rate.n_cbps, rate.n_bpsc)
            interleaved_symbols.append(interleaved_bits)
            symbols.append(_ifft_symbol(map_constellation(interleaved_bits, rate.n_bpsc), index + 1, oversample))
        data_iq = np.concatenate(symbols)
        ppdu = np.concatenate((l_stf, l_ltf, l_sig, data_iq))
        peak = float(np.max(np.abs(ppdu)))
        if peak > 1.0:
            ppdu = ppdu / peak
        samples_per_ofdm = 80 * oversample
        period_count = round(float(settings.packet_period_us) * 1e-6 * expected_sample_rate)
        if period_count < ppdu.size:
            raise ValueError(f"Packet period is shorter than PPDU duration ({ppdu.size / expected_sample_rate * 1e6:.1f} us)")
        single = np.pad(ppdu, (0, period_count - ppdu.size))
        iq = np.tile(single, int(project.repeat_count)).astype(np.complex64)
        field_sizes = (("L-STF", 160 * oversample), ("L-LTF", 160 * oversample), ("L-SIG", 80 * oversample), ("DATA", 80 * oversample * n_sym))
        boundaries: list[FieldBoundary] = []
        packet_ranges = []
        sample_ranges: dict[str, tuple[int, int]] = {}
        for repeat in range(int(project.repeat_count)):
            cursor = repeat * period_count
            packet_ranges.append((cursor, cursor + ppdu.size))
            for name, size in field_sizes:
                start = cursor
                boundaries.append(FieldBoundary(name, start, start + size, start // samples_per_ofdm, (start + size) // samples_per_ofdm))
                if repeat == 0:
                    sample_ranges[name.lower().replace("-", "_")] = (start, start + size)
                cursor += size
        level_metrics = measure_iq_levels(iq, packet_ranges)
        return GenerationResult(
            iq=iq,
            sample_rate_hz=expected_sample_rate,
            field_boundaries=tuple(boundaries),
            metadata={
                "project_name": project.name, "standard": project.standard.value,
                "center_frequency_hz": project.center_frequency_hz, "packet_name": "Wi-Fi Non-HT OFDM",
                "phy_format": settings.phy_format.value, "legacy_rate_mbps": int(settings.legacy_rate_mbps),
                "channel_bandwidth_hz": 20_000_000, "oversample_factor": oversample,
                "psdu": psdu, "psdu_length_bytes": len(psdu), "scrambler_seed": seed,
                "n_bpsc": rate.n_bpsc, "n_cbps": rate.n_cbps, "n_dbps": rate.n_dbps,
                "coding_rate": rate.coding_rate, "modulation": rate.modulation,
                "n_sym": n_sym, "n_pad": n_pad, "l_sig_bits": l_sig_bits,
                "scrambled_data_bits": scrambled, "interleaved_bits_per_symbol": tuple(interleaved_symbols),
                "sample_ranges": sample_ranges, "packet_ranges_samples": tuple(packet_ranges),
                "active_ranges_samples": tuple(packet_ranges),
                "packet_sample_count": ppdu.size, "period_sample_count": period_count,
                "period_symbols": period_count / samples_per_ofdm, "samples_per_symbol": samples_per_ofdm,
                "symbol_rate_hz": 250_000.0, "ppdu_duration_us": ppdu.size / expected_sample_rate * 1e6,
                **iq_level_metadata(level_metrics),
            },
        )


__all__ = ["WiFiLegacyOFDMWaveformEngine", "bcc_encode", "interleave", "map_constellation", "puncture", "scramble"]
