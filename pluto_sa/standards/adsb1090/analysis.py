"""1090 MHz Mode S preamble detection and PPM demodulation."""

from __future__ import annotations

import numpy as np

from pluto_sa.standards.adsb1090.decoder import (
    bits_to_hex,
    bits_to_int,
    decode_adsb_fields,
    mode_s_crc_remainder,
)
from pluto_sa.standards.adsb1090.model import (
    ADSB1090AnalysisResult,
    ADSB1090Message,
    ADSB1090Settings,
)
from pluto_sa.vsa.model import IQRecording


_EPSILON = np.finfo(np.float64).tiny
_PREAMBLE_US = 8.0
_LONG_MESSAGE_BITS = 112
_SHORT_MESSAGE_BITS = 56


def _window_mean(values: np.ndarray, start: float, stop: float) -> float:
    lo = max(0, int(np.floor(start)))
    hi = min(values.size, max(lo + 1, int(np.ceil(stop))))
    return float(np.mean(values[lo:hi]))


def _preamble_template(sample_rate_hz: float) -> np.ndarray:
    count = max(16, int(round(_PREAMBLE_US * 1e-6 * sample_rate_hz)))
    time_us = np.arange(count, dtype=np.float64) * 1e6 / sample_rate_hz
    pulse = (
        ((time_us >= 0.0) & (time_us < 0.5))
        | ((time_us >= 1.0) & (time_us < 1.5))
        | ((time_us >= 3.5) & (time_us < 4.0))
        | ((time_us >= 4.5) & (time_us < 5.0))
    )
    template = np.where(pulse, 1.0, -float(np.mean(pulse)) / max(1.0 - float(np.mean(pulse)), _EPSILON))
    norm = float(np.linalg.norm(template))
    return template / max(norm, _EPSILON)


def _normalized_correlation(power: np.ndarray, template: np.ndarray) -> np.ndarray:
    centered = np.asarray(power, dtype=np.float64)
    numerator = np.correlate(centered, template, mode="valid")
    energy = np.convolve(centered * centered, np.ones(template.size), mode="valid")
    return numerator / np.sqrt(np.maximum(energy, _EPSILON))


def _candidate_starts(
    power: np.ndarray,
    sample_rate_hz: float,
    settings: ADSB1090Settings,
) -> tuple[np.ndarray, np.ndarray]:
    template = _preamble_template(sample_rate_hz)
    if power.size < template.size:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    correlation = _normalized_correlation(power, template)
    possible = np.flatnonzero(correlation >= settings.minimum_preamble_correlation)
    if possible.size == 0:
        return np.empty(0, dtype=np.int64), correlation
    # Retain one local maximum per 8 us neighborhood. Payload pulses can mimic
    # fragments of the preamble; SNR/shape validation later rejects them.
    separation = max(1, int(round(_PREAMBLE_US * 1e-6 * sample_rate_hz)))
    ordered = possible[np.argsort(correlation[possible])[::-1]]
    selected: list[int] = []
    for index in ordered:
        if all(abs(int(index) - prior) >= separation for prior in selected):
            selected.append(int(index))
    selected.sort()
    return np.asarray(selected, dtype=np.int64), correlation


def _preamble_quality(
    power: np.ndarray, start: int, sample_rate_hz: float
) -> tuple[float, bool]:
    samples_per_us = sample_rate_hz * 1e-6
    segment = power[start : start + int(np.ceil(8.0 * samples_per_us))]
    pulse_ranges = ((0.0, 0.5), (1.0, 1.5), (3.5, 4.0), (4.5, 5.0))
    quiet_ranges = ((0.5, 1.0), (1.5, 3.5), (4.0, 4.5), (5.0, 8.0))
    pulse = np.asarray(
        [_window_mean(segment, a * samples_per_us, b * samples_per_us) for a, b in pulse_ranges]
    )
    quiet = np.asarray(
        [_window_mean(segment, a * samples_per_us, b * samples_per_us) for a, b in quiet_ranges]
    )
    snr_db = 10.0 * np.log10(max(float(np.mean(pulse)), _EPSILON) / max(float(np.mean(quiet)), _EPSILON))
    shape_ok = bool(float(np.min(pulse)) > float(np.max(quiet)))
    return float(snr_db), shape_ok


def _decode_ppm(
    power: np.ndarray, start: int, sample_rate_hz: float, bit_count: int
) -> tuple[np.ndarray, np.ndarray]:
    samples_per_us = sample_rate_hz * 1e-6
    data_start = start + 8.0 * samples_per_us
    bits = np.empty(bit_count, dtype=np.uint8)
    confidence = np.empty(bit_count, dtype=np.float64)
    for bit_index in range(bit_count):
        symbol_start = data_start + bit_index * samples_per_us
        first = _window_mean(power, symbol_start, symbol_start + 0.5 * samples_per_us)
        second = _window_mean(
            power,
            symbol_start + 0.5 * samples_per_us,
            symbol_start + samples_per_us,
        )
        bits[bit_index] = 1 if first >= second else 0
        confidence[bit_index] = abs(first - second) / max(first + second, _EPSILON)
    return bits, confidence


class ADSB1090Analyzer:
    """Detect every Mode S burst in one immutable IQ recording."""

    def analyze(
        self,
        recording: IQRecording,
        settings: ADSB1090Settings | None = None,
    ) -> ADSB1090AnalysisResult:
        resolved = settings or ADSB1090Settings()
        sample_rate_hz = float(recording.sample_rate_hz)
        if sample_rate_hz < 4_000_000.0:
            raise ValueError("1090ES analysis requires at least 4 MS/s")
        iq = np.asarray(recording.iq, dtype=np.complex128)
        power = (np.abs(iq) / float(recording.full_scale)) ** 2
        power_dbfs = 10.0 * np.log10(np.maximum(power, _EPSILON))
        time_s = np.arange(iq.size, dtype=np.float64) / sample_rate_hz
        starts, correlations = _candidate_starts(power, sample_rate_hz, resolved)
        messages: list[ADSB1090Message] = []
        for start in starts:
            if len(messages) >= resolved.maximum_messages:
                break
            snr_db, shape_ok = _preamble_quality(power, int(start), sample_rate_hz)
            if not shape_ok or snr_db < resolved.minimum_preamble_snr_db:
                continue
            available_bits = int(
                np.floor((power.size - (int(start) + 8e-6 * sample_rate_hz)) / (1e-6 * sample_rate_hz))
            )
            if available_bits < _SHORT_MESSAGE_BITS:
                continue
            bit_count = _LONG_MESSAGE_BITS if available_bits >= _LONG_MESSAGE_BITS else _SHORT_MESSAGE_BITS
            bits, confidence = _decode_ppm(power, int(start), sample_rate_hz, bit_count)
            downlink_format = bits_to_int(bits[0:5])
            # DF16 and above use the 112-bit long format.  In a streaming
            # analysis window, 56 bits may be available before the remainder
            # crosses the next block boundary; defer rather than publishing a
            # false short message for the same preamble.
            if downlink_format >= 16 and bit_count < _LONG_MESSAGE_BITS:
                continue
            # Mode S formats 16 and above normally use the long message. Decode
            # the short form for lower DFs even if extra capture data follows.
            if downlink_format < 16 and bit_count == _LONG_MESSAGE_BITS:
                bits = bits[:_SHORT_MESSAGE_BITS]
                confidence = confidence[:_SHORT_MESSAGE_BITS]
            remainder = mode_s_crc_remainder(bits)
            crc_ok = remainder == 0
            if resolved.require_valid_crc and not crc_ok:
                continue
            fields = decode_adsb_fields(bits)
            icao = (
                f"{bits_to_int(bits[8:32]):06X}"
                if bits.size == 112 and downlink_format in {17, 18}
                else None
            )
            type_code = int(fields["type_code"]) if "type_code" in fields else None
            message = ADSB1090Message(
                start_sample=int(start),
                sample_rate_hz=sample_rate_hz,
                raw_hex=bits_to_hex(bits),
                bits=bits,
                downlink_format=downlink_format,
                capability=bits_to_int(bits[5:8]),
                icao_address=icao,
                type_code=type_code,
                crc_remainder=remainder,
                crc_ok=crc_ok,
                preamble_snr_db=snr_db,
                preamble_correlation=float(correlations[int(start)]),
                fields={
                    **fields,
                    "mean_bit_confidence": float(np.mean(confidence)),
                    "minimum_bit_confidence": float(np.min(confidence)),
                },
            )
            messages.append(message)
        return ADSB1090AnalysisResult(
            time_s=time_s,
            power_dbfs=power_dbfs,
            messages=tuple(messages),
            metadata={
                "source": recording.source,
                "sample_rate_hz": sample_rate_hz,
                "center_frequency_hz": recording.center_frequency_hz,
                "candidate_count": int(starts.size),
                "valid_message_count": sum(message.crc_ok for message in messages),
            },
        )
