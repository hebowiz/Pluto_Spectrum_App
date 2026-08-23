from dataclasses import replace

import numpy as np
import pytest
from pathlib import Path

from pluto_sa.standards.adsb1090 import (
    ADSB1090Analyzer,
    ADSB1090Settings,
)
from pluto_sa.standards.adsb1090.decoder import (
    classify_mode_s_parity,
    decode_adsb_fields,
    decode_global_airborne_cpr,
    decode_mode_s_header_fields,
    mode_s_crc_remainder,
)
from pluto_sa.vsa.model import IQRecording
from pluto_sa.vsa.sources import FileIQSource


KNOWN_DF17 = "8D40621D58C382D690C8AC2863A7"
KNOWN_DF17_ODD = "8D40621D58C386435CC412692AD6"
KNOWN_VELOCITY = "8D485020994409940838175B284F"
KNOWN_IDENTIFICATION = "8D4840D6202CC371C32CE0576098"
OBSERVED_DF5_ADDRESS_PARITY = "28201507C8E5CD"


def _hex_bits(value: str) -> np.ndarray:
    return np.asarray(
        [int(bit) for bit in f"{int(value, 16):0{len(value) * 4}b}"],
        dtype=np.uint8,
    )


def _mode_s_recording(
    messages: list[str], *, sample_rate_hz: float = 8e6
) -> IQRecording:
    samples_per_half_bit = int(round(sample_rate_hz * 0.5e-6))
    samples_per_us = 2 * samples_per_half_bit
    packet_samples = int(round(120e-6 * sample_rate_hz))
    gap_samples = int(round(80e-6 * sample_rate_hz))
    start = gap_samples
    count = gap_samples + len(messages) * (packet_samples + gap_samples)
    rng = np.random.default_rng(42)
    iq = (
        rng.normal(scale=0.005, size=count)
        + 1j * rng.normal(scale=0.005, size=count)
    ).astype(np.complex64)
    for raw in messages:
        bits = _hex_bits(raw)
        for pulse_us in (0.0, 1.0, 3.5, 4.5):
            pulse_start = start + int(round(pulse_us * samples_per_us))
            iq[pulse_start : pulse_start + samples_per_half_bit] += 1.0
        data_start = start + 8 * samples_per_us
        for bit_index, bit in enumerate(bits):
            pulse_start = data_start + bit_index * samples_per_us
            if not bit:
                pulse_start += samples_per_half_bit
            iq[pulse_start : pulse_start + samples_per_half_bit] += 1.0
        start += packet_samples + gap_samples
    return IQRecording(
        iq=iq,
        sample_rate_hz=sample_rate_hz,
        center_frequency_hz=1_090_000_000.0,
        source="Synthetic 1090ES",
    )


def _short_reply_with_overlay(
    *, downlink_format: int, first_field: int, address: int, overlay: int
) -> str:
    prefix = (downlink_format << 27) | (first_field << 24) | address
    zero_parity = np.asarray(
        [int(bit) for bit in f"{prefix:032b}{0:024b}"], dtype=np.uint8
    )
    parity = mode_s_crc_remainder(zero_parity) ^ overlay
    return f"{prefix:08X}{parity:06X}"


def test_known_df17_crc_and_airborne_position_fields() -> None:
    bits = _hex_bits(KNOWN_DF17)

    fields = decode_adsb_fields(bits)

    assert mode_s_crc_remainder(bits) == 0
    assert fields["type_code"] == 11
    assert fields["icao_address"] == "40621D"
    assert fields["altitude_ft"] == 38_000
    assert fields["cpr_format"] == "even"
    assert fields["air_ground"] == "airborne"


def test_global_airborne_cpr_decodes_even_odd_pair() -> None:
    even = decode_adsb_fields(_hex_bits(KNOWN_DF17))
    odd = decode_adsb_fields(_hex_bits(KNOWN_DF17_ODD))

    even_position = decode_global_airborne_cpr(
        even["cpr_latitude"],
        even["cpr_longitude"],
        odd["cpr_latitude"],
        odd["cpr_longitude"],
        use_odd=False,
    )
    odd_position = decode_global_airborne_cpr(
        even["cpr_latitude"],
        even["cpr_longitude"],
        odd["cpr_latitude"],
        odd["cpr_longitude"],
        use_odd=True,
    )

    assert even_position == pytest.approx((52.257202, 3.919373), abs=1e-6)
    assert odd_position == pytest.approx((52.265780, 3.938913), abs=1e-6)


def test_velocity_message_decodes_vertical_rate_and_airborne_state() -> None:
    fields = decode_adsb_fields(_hex_bits(KNOWN_VELOCITY))

    assert fields["air_ground"] == "airborne"
    assert fields["vertical_rate_fpm"] == -832.0
    assert fields["vertical_rate_source"] == "barometric"


def test_analyzer_detects_and_decodes_every_burst() -> None:
    recording = _mode_s_recording([KNOWN_DF17, KNOWN_DF17])

    result = ADSB1090Analyzer().analyze(recording)

    valid = [message for message in result.messages if message.crc_ok]
    assert len(valid) == 2
    assert [message.raw_hex for message in valid] == [KNOWN_DF17, KNOWN_DF17]
    assert all(message.downlink_format == 17 for message in valid)
    assert all(message.icao_address == "40621D" for message in valid)
    assert valid[1].start_time_s > valid[0].start_time_s


def test_identification_message_decodes_callsign() -> None:
    recording = _mode_s_recording([KNOWN_IDENTIFICATION])

    result = ADSB1090Analyzer().analyze(recording)

    assert len(result.messages) == 1
    assert result.messages[0].crc_ok is True
    assert result.messages[0].icao_address == "4840D6"
    assert result.messages[0].fields["callsign"] == "KLM1023"


def test_crc_filter_rejects_corrupt_message() -> None:
    corrupt = f"{int(KNOWN_DF17, 16) ^ 1:028X}"
    recording = _mode_s_recording([corrupt])

    unfiltered = ADSB1090Analyzer().analyze(recording)
    filtered = ADSB1090Analyzer().analyze(
        recording,
        ADSB1090Settings(require_valid_crc=True),
    )

    assert len(unfiltered.messages) == 1
    assert unfiltered.messages[0].crc_ok is False
    assert filtered.messages == ()


def test_df5_address_parity_recovers_and_confirms_icao() -> None:
    raw = _short_reply_with_overlay(
        downlink_format=5,
        first_field=2,
        address=0x123456,
        overlay=0x40621D,
    )
    bits = _hex_bits(raw)

    parity = classify_mode_s_parity(bits)
    result = ADSB1090Analyzer().analyze(
        _mode_s_recording([raw, KNOWN_DF17]),
        ADSB1090Settings(require_valid_crc=True),
    )

    assert parity.kind == "address"
    assert parity.valid is None
    assert parity.recovered_icao == "40621D"
    assert decode_mode_s_header_fields(bits) == {"flight_status": 2}
    assert len(result.messages) == 2
    reply = result.messages[0]
    assert reply.icao_address == "40621D"
    assert reply.icao_address_source == "address_parity"
    assert reply.icao_confirmed is True
    assert reply.parity_ok is True
    assert reply.parity_display == "AP 40621D Confirmed"


def test_observed_df5_syndrome_is_an_icao_address_not_crc_failure() -> None:
    bits = _hex_bits(OBSERVED_DF5_ADDRESS_PARITY)

    parity = classify_mode_s_parity(bits)

    assert parity.kind == "address"
    assert parity.valid is None
    assert parity.recovered_icao == "84D28E"
    assert decode_mode_s_header_fields(bits) == {"flight_status": 0}


def test_df11_interrogator_parity_and_capability_are_df_aware() -> None:
    raw = _short_reply_with_overlay(
        downlink_format=11,
        first_field=5,
        address=0xABCDEF,
        overlay=0x2A,
    )
    bits = _hex_bits(raw)

    parity = classify_mode_s_parity(bits)

    assert parity.kind == "interrogator"
    assert parity.valid is True
    assert parity.interrogator_identifier == 0x2A
    assert decode_mode_s_header_fields(bits) == {"capability": 5}


def test_analyzer_requires_enough_time_resolution() -> None:
    recording = IQRecording(
        iq=np.ones(100, dtype=np.complex64),
        sample_rate_hz=2e6,
    )

    with pytest.raises(ValueError, match="at least 4 MS/s"):
        ADSB1090Analyzer().analyze(recording)


def test_stream_window_defers_incomplete_long_message() -> None:
    recording = _mode_s_recording([KNOWN_DF17])
    message_start = int(round(80e-6 * recording.sample_rate_hz))
    partial_stop = message_start + int(round((8 + 70) * 1e-6 * recording.sample_rate_hz))

    result = ADSB1090Analyzer().analyze(
        replace(recording, iq=recording.iq[:partial_stop])
    )

    assert result.messages == ()


def test_saved_multi_packet_fixture_decodes_expected_frames() -> None:
    path = Path(__file__).parent / "fixtures" / "adsb1090_multi_8msps.npz"
    recording = FileIQSource.load(path)

    result = ADSB1090Analyzer().analyze(recording)

    valid_hex = [message.raw_hex for message in result.messages if message.crc_ok]
    assert valid_hex == [
        "8D4840D6202CC371C32CE0576098",
        "8D40621D58C382D690C8AC2863A7",
        "8D40621D58C386435CC412692AD6",
        "8D485020994409940838175B284F",
    ]
