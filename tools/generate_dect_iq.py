"""Generate a deterministic Classic DECT GFSK IQ recording."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pluto_sa.vsa.protocol_modes.dect.generator import generate_dect_packet


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("dect_p32_case_a.npz"))
    parser.add_argument("--direction", choices=("RFP", "PP"), default="RFP")
    parser.add_argument("--packet", choices=("P00", "P32", "P32Z", "P80", "P80Z"), default="P32")
    parser.add_argument(
        "--pattern", choices=("00001111", "0101", "prbs9", "random"), default="00001111"
    )
    parser.add_argument("--frequency-mhz", type=float, default=1888.704)
    parser.add_argument("--frequency-error-hz", type=float, default=0.0)
    parser.add_argument("--symbol-rate-error-ppm", type=float, default=0.0)
    parser.add_argument("--power-dbm", type=float, default=-10.0)
    parser.add_argument("--samples-per-symbol", type=int, default=8)
    parser.add_argument("--prolonged-preamble", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    recording = generate_dect_packet(
        direction=args.direction,
        packet_type=args.packet,
        payload_pattern=args.pattern,
        center_frequency_hz=args.frequency_mhz * 1e6,
        samples_per_symbol=args.samples_per_symbol,
        frequency_error_hz=args.frequency_error_hz,
        symbol_rate_error_ppm=args.symbol_rate_error_ppm,
        power_dbm=args.power_dbm,
        prolonged_preamble=args.prolonged_preamble,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        iq=recording.iq,
        sample_rate_hz=np.float64(recording.sample_rate_hz),
        center_frequency_hz=np.float64(recording.center_frequency_hz),
        usable_bandwidth_hz=np.float64(recording.usable_bandwidth_hz),
        full_scale=np.float64(recording.full_scale),
        amplitude_calibrated=np.bool_(recording.amplitude_calibrated),
        generated_bits=recording.metadata["generated_bits"],
        direction=np.asarray(recording.metadata["direction"]),
        packet_type=np.asarray(recording.metadata["packet_type"]),
        payload_pattern=np.asarray(recording.metadata["payload_pattern"]),
        preamble_mode=np.asarray(recording.metadata["preamble_mode"]),
        expected_p0_sample=np.float64(recording.metadata["expected_p0_sample"]),
        expected_packet_stop_sample=np.int64(
            recording.metadata["expected_packet_stop_sample"]
        ),
    )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
