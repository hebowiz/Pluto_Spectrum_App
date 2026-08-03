"""Analyze a saved Bluetooth BR GFSK capture from the command line."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pluto_sa.vsa.profiles.bluetooth_br import BluetoothBRProfile, access_code_bits
from pluto_sa.vsa.sources import FileIQSource


def _parse_hex(value: str) -> int:
    return int(str(value), 0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help=".npz/.npy/raw complex IQ file")
    parser.add_argument("--sample-rate", type=float, default=None)
    parser.add_argument("--center-frequency", type=float, default=0.0)
    parser.add_argument("--raw-dtype", default="complex64")
    parser.add_argument(
        "--lap",
        type=_parse_hex,
        default=0x9E8B33,
        help="24-bit LAP used to construct the access code (default: GIAC)",
    )
    parser.add_argument(
        "--shortened-access-code",
        action="store_true",
        help="use a 68-bit access code without trailer (inquiry/page ID)",
    )
    parser.add_argument("--clock", type=_parse_hex, default=None, help="CLK_6-1")
    parser.add_argument("--uap", type=_parse_hex, default=None)
    parser.add_argument("--minimum-correlation", type=float, default=0.65)
    parser.add_argument("--show-bits", type=int, default=256)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    recording = FileIQSource.load(
        args.path,
        sample_rate_hz=args.sample_rate,
        center_frequency_hz=args.center_frequency,
        raw_dtype=args.raw_dtype,
    )
    profile = BluetoothBRProfile(
        access_code_bits(
            args.lap,
            include_trailer=not args.shortened_access_code,
        )
    )
    result = profile.analyze(
        recording,
        clock_6_1=args.clock,
        uap=args.uap,
        minimum_correlation=args.minimum_correlation,
    )
    demod = result.demodulation
    summary: dict[str, object] = {
        "source": recording.source,
        "sample_rate_hz": recording.sample_rate_hz,
        "lap": f"0x{args.lap:06X}",
        "access_start_sample": demod.access_start_sample,
        "access_start_time_s": demod.access_start_sample / recording.sample_rate_hz,
        "access_correlation": demod.access_correlation,
        "access_bit_errors": demod.access_bit_errors,
        "carrier_frequency_offset_hz": demod.carrier_frequency_offset_hz,
        "carrier_frequency_drift_hz_per_s": demod.carrier_frequency_drift_hz_per_s,
        "frequency_deviation_hz": demod.frequency_deviation_hz,
        "iq_inverted": demod.iq_inverted,
        "burst_ranges": demod.burst_ranges,
        "recovered_bit_count": int(demod.bits.size),
    }
    if result.header is not None:
        summary["header"] = {
            "lt_addr": result.header.lt_addr,
            "packet_type": result.header.packet_type,
            "flow": result.header.flow,
            "arqn": result.header.arqn,
            "seqn": result.header.seqn,
            "hec": f"0x{result.header.hec:02X}",
            "hec_valid": result.header.hec_valid,
            "corrected_fec_triplets": result.header.corrected_fec_triplets,
        }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    shown = demod.bits[: max(0, int(args.show_bits))]
    if shown.size:
        print("bits=" + "".join(str(int(bit)) for bit in shown))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
