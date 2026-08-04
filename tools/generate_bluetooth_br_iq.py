"""Generate a deterministic maximum-length Bluetooth BR DH1 IQ fixture."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pluto_sa.vsa.profiles.bluetooth_br_waveform import generate_br_dh1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("tests/fixtures"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    waveform = generate_br_dh1(seed=11)
    recording = waveform.recording
    path = args.output_dir / "bluetooth_dh1_prbs9_16msps.npz"
    np.savez(
        path,
        iq=recording.iq,
        sample_rate_hz=np.float64(recording.sample_rate_hz),
        center_frequency_hz=np.float64(recording.center_frequency_hz),
        usable_bandwidth_hz=np.float64(recording.usable_bandwidth_hz),
        full_scale=np.float64(recording.full_scale),
        amplitude_calibrated=np.bool_(recording.amplitude_calibrated),
        packet_name=np.asarray(waveform.packet_name),
        modulation=np.asarray(waveform.modulation.value),
        packet_type=np.uint8(waveform.packet_type),
        payload_length_bytes=np.uint16(waveform.payload_length_bytes),
        packet_bits=waveform.packet_bits,
        access_bits=waveform.access_bits,
        header_air_bits=waveform.header_air_bits,
        payload_header_bits=waveform.payload_header_bits,
        payload_body_bits=waveform.payload_body_bits,
        payload_crc_bits=waveform.payload_crc_bits,
        payload_air_bits=waveform.payload_air_bits,
        packet_start_sample=np.int64(waveform.packet_start_sample),
        packet_stop_sample=np.int64(waveform.packet_stop_sample),
        lap=np.uint32(recording.metadata["lap"]),
        uap=np.uint8(recording.metadata["uap"]),
        clock_6_1=np.uint8(recording.metadata["clock_6_1"]),
        carrier_frequency_offset_hz=np.float64(
            recording.metadata["carrier_frequency_offset_hz"]
        ),
        frequency_deviation_hz=np.float64(
            recording.metadata["frequency_deviation_hz"]
        ),
        tx_filter=np.asarray(recording.metadata["tx_filter"]),
        bt=np.float64(recording.metadata["bt"]),
    )
    print(path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
