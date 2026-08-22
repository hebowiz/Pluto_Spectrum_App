"""Generate a deterministic multi-packet 1090ES IQ regression fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pluto_sa.standards.adsb1090 import ADSB1090Analyzer
from pluto_sa.vsa.model import IQRecording
from pluto_sa.vsa.sources import FileIQSource


MESSAGES = (
    # Aircraft identification: ICAO 4840D6, callsign KLM1023.
    "8D4840D6202CC371C32CE0576098",
    # Known even/odd airborne-position pair: ICAO 40621D, altitude 38000 ft.
    "8D40621D58C382D690C8AC2863A7",
    "8D40621D58C386435CC412692AD6",
    # Airborne velocity example.
    "8D485020994409940838175B284F",
)


def _hex_bits(value: str) -> np.ndarray:
    width = len(value) * 4
    return np.asarray(
        [int(bit) for bit in f"{int(value, 16):0{width}b}"],
        dtype=np.uint8,
    )


def generate_fixture(
    *,
    sample_rate_hz: float = 8_000_000.0,
    duration_s: float = 0.002,
    center_frequency_hz: float = 1_090_000_000.0,
    carrier_offset_hz: float = 27_500.0,
    seed: int = 1090,
) -> IQRecording:
    if sample_rate_hz < 8_000_000.0:
        raise ValueError("fixture generation expects at least 8 MS/s")
    sample_count = int(round(duration_s * sample_rate_hz))
    samples_per_half_bit = int(round(0.5e-6 * sample_rate_hz))
    samples_per_us = 2 * samples_per_half_bit
    starts_us = (180.0, 570.0, 960.0, 1350.0)
    amplitudes = (0.82, 1.00, 0.73, 0.90)
    envelope = np.zeros(sample_count, dtype=np.float64)
    for raw, start_us, amplitude in zip(MESSAGES, starts_us, amplitudes):
        start = int(round(start_us * 1e-6 * sample_rate_hz))
        for pulse_us in (0.0, 1.0, 3.5, 4.5):
            pulse_start = start + int(round(pulse_us * samples_per_us))
            envelope[pulse_start : pulse_start + samples_per_half_bit] += amplitude
        data_start = start + 8 * samples_per_us
        for bit_index, bit in enumerate(_hex_bits(raw)):
            pulse_start = data_start + bit_index * samples_per_us
            if not bit:
                pulse_start += samples_per_half_bit
            envelope[pulse_start : pulse_start + samples_per_half_bit] += amplitude

    # Approximate a finite analog edge without changing the 0.5 us pulse
    # positions. This makes the fixture less ideal than rectangular unit tests.
    edge_kernel = np.asarray([0.08, 0.17, 0.50, 0.17, 0.08], dtype=np.float64)
    envelope = np.convolve(envelope, edge_kernel, mode="same")
    time_s = np.arange(sample_count, dtype=np.float64) / sample_rate_hz
    carrier = np.exp(1j * (2.0 * np.pi * carrier_offset_hz * time_s + 0.37))
    rng = np.random.default_rng(seed)
    noise = (
        rng.normal(scale=0.006, size=sample_count)
        + 1j * rng.normal(scale=0.006, size=sample_count)
    )
    dc_offset = 0.0025 - 0.0015j
    iq = (envelope * carrier + noise + dc_offset).astype(np.complex64)
    return IQRecording(
        iq=iq,
        sample_rate_hz=sample_rate_hz,
        center_frequency_hz=center_frequency_hz,
        usable_bandwidth_hz=4_000_000.0,
        source="Generated ADS-B 1090ES multi-packet fixture",
        full_scale=1.0,
        amplitude_calibrated=False,
        metadata={
            "generated_fixture": True,
            "carrier_offset_hz": carrier_offset_hz,
            "dc_removal_recommended": False,
            "seed": seed,
        },
    )


def save_fixture(recording: IQRecording, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        iq=recording.iq,
        sample_rate_hz=np.float64(recording.sample_rate_hz),
        center_frequency_hz=np.float64(recording.center_frequency_hz),
        usable_bandwidth_hz=np.float64(recording.usable_bandwidth_hz),
        full_scale=np.float64(recording.full_scale),
        calibration_offset_db=np.float64(recording.calibration_offset_db),
        frequency_dependent_offset_db=np.float64(
            recording.frequency_dependent_offset_db
        ),
        input_correction_db=np.float64(recording.input_correction_db),
        amplitude_calibrated=np.bool_(recording.amplitude_calibrated),
        dc_removal_recommended=np.bool_(False),
    )


def write_manifest(output: Path) -> Path:
    loaded = FileIQSource.load(output)
    result = ADSB1090Analyzer().analyze(loaded)
    manifest = {
        "schema": "pluto-vsa-adsb1090-fixture",
        "version": 1,
        "iq_file": output.name,
        "sample_rate_hz": loaded.sample_rate_hz,
        "center_frequency_hz": loaded.center_frequency_hz,
        "duration_s": loaded.duration_s,
        "expected_messages": [
            {
                "raw_hex": message.raw_hex,
                "start_time_us": message.start_time_s * 1e6,
                "downlink_format": message.downlink_format,
                "icao_address": message.icao_address,
                "type_code": message.type_code,
                "crc_ok": message.crc_ok,
                "fields": dict(message.fields),
            }
            for message in result.messages
            if message.crc_ok
        ],
    }
    manifest_path = output.with_suffix(output.suffix + ".json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/fixtures/adsb1090_multi_8msps.npz"),
    )
    args = parser.parse_args()
    recording = generate_fixture()
    save_fixture(recording, args.output)
    manifest = write_manifest(args.output)
    print(f"Wrote {args.output} and {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
