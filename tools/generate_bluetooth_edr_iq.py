"""Generate deterministic maximum-length Bluetooth 2-DH1 and 3-DH1 IQ fixtures."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pluto_sa.vsa.profiles.bluetooth_edr import generate_edr_dh1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("tests/fixtures"))
    return parser


def _save(path: Path, packet_name: str, seed: int) -> None:
    waveform = generate_edr_dh1(packet_name, seed=seed)
    recording = waveform.recording
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
        access_bits=waveform.access_bits,
        header_air_bits=waveform.header_air_bits,
        sync_bits=waveform.sync_bits,
        payload_header_bits=waveform.payload_header_bits,
        payload_body_bits=waveform.payload_body_bits,
        payload_crc_bits=waveform.payload_crc_bits,
        payload_air_bits=waveform.payload_air_bits,
        trailer_bits=waveform.trailer_bits,
        differential_phase_indices=waveform.differential_phase_indices,
        logical_symbols=waveform.logical_symbols,
        packet_start_sample=np.int64(waveform.packet_start_sample),
        gfsk_stop_sample=np.int64(waveform.gfsk_stop_sample),
        edr_start_sample=np.int64(waveform.edr_start_sample),
        packet_stop_sample=np.int64(waveform.packet_stop_sample),
        lap=np.uint32(recording.metadata["lap"]),
        uap=np.uint8(recording.metadata["uap"]),
        clock_6_1=np.uint8(recording.metadata["clock_6_1"]),
        carrier_frequency_offset_hz=np.float64(
            recording.metadata["carrier_frequency_offset_hz"]
        ),
        tx_filter=np.asarray(recording.metadata["tx_filter"]),
        rolloff=np.float64(recording.metadata["rolloff"]),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for packet_name, filename, seed in (
        ("2-DH1", "bluetooth_2dh1_prbs9_16msps.npz", 21),
        ("3-DH1", "bluetooth_3dh1_prbs9_16msps.npz", 31),
    ):
        path = args.output_dir / filename
        _save(path, packet_name, seed)
        print(path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
