"""Analyze a saved Bluetooth BR GFSK capture from the command line."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pluto_sa.vsa.channel import extract_analysis_channel
from pluto_sa.vsa.profiles.bluetooth_br import (
    BluetoothBRProfile,
    access_code_bits,
    find_dh1_candidates,
    find_header_candidates,
    match_prbs9,
)
from pluto_sa.vsa.sources import FileIQSource


def _parse_hex(value: str) -> int:
    return int(str(value), 0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help=".npz/.npy/raw complex IQ file")
    parser.add_argument("--sample-rate", type=float, default=None)
    parser.add_argument("--center-frequency", type=float, default=0.0)
    parser.add_argument(
        "--analysis-center-frequency",
        type=float,
        default=None,
        help="absolute center frequency selected for demodulation",
    )
    parser.add_argument(
        "--analysis-bandwidth",
        type=float,
        default=None,
        help="DDC/FIR channel bandwidth; omitted to analyze the full recording",
    )
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
    parser.add_argument(
        "--no-whitening",
        action="store_true",
        help="decode header/payload without the standard whitening sequence",
    )
    parser.add_argument("--minimum-correlation", type=float, default=0.65)
    parser.add_argument(
        "--search-all-uap",
        action="store_true",
        help="diagnose DH1 candidates across every UAP when transmitter settings are unknown",
    )
    parser.add_argument(
        "--payload-pattern",
        choices=("unknown", "prbs9"),
        default="unknown",
    )
    parser.add_argument("--show-bits", type=int, default=256)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.analysis_center_frequency is not None and args.analysis_bandwidth is None:
        raise SystemExit(
            "--analysis-bandwidth is required with --analysis-center-frequency"
        )
    if args.no_whitening and args.clock is not None:
        raise SystemExit("--clock cannot be combined with --no-whitening")
    recording = FileIQSource.load(
        args.path,
        sample_rate_hz=args.sample_rate,
        center_frequency_hz=args.center_frequency,
        raw_dtype=args.raw_dtype,
    )
    input_recording = recording
    if args.analysis_bandwidth is not None:
        recording = extract_analysis_channel(
            recording,
            center_frequency_hz=(
                recording.center_frequency_hz
                if args.analysis_center_frequency is None
                else args.analysis_center_frequency
            ),
            bandwidth_hz=args.analysis_bandwidth,
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
        whitening_enabled=not args.no_whitening,
        minimum_correlation=args.minimum_correlation,
    )
    clock_candidates = ()
    if (
        result.header is None
        and args.clock is None
        and not args.no_whitening
        and args.uap is not None
        and result.header_air_bits.size == 54
    ):
        clock_candidates = find_header_candidates(
            result.header_air_bits,
            uap=args.uap,
        )
        if len(clock_candidates) == 1:
            selected = clock_candidates[0]
            result = profile.analyze(
                recording,
                clock_6_1=selected.clock_6_1,
                uap=args.uap,
                whitening_enabled=selected.whitening_enabled,
                minimum_correlation=args.minimum_correlation,
            )
    demod = result.demodulation
    summary: dict[str, object] = {
        "source": recording.source,
        "input_sample_rate_hz": input_recording.sample_rate_hz,
        "input_center_frequency_hz": input_recording.center_frequency_hz,
        "sample_rate_hz": recording.sample_rate_hz,
        "analysis_center_frequency_hz": recording.center_frequency_hz,
        "analysis_bandwidth_hz": recording.usable_bandwidth_hz,
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
            "clock_6_1": result.header.clock_6_1,
            "whitening_enabled": result.header.whitening_enabled,
            "corrected_fec_triplets": result.header.corrected_fec_triplets,
        }
    if clock_candidates:
        summary["header_candidates"] = [
            {
                "clock_6_1": candidate.clock_6_1,
                "whitening_enabled": candidate.whitening_enabled,
                "lt_addr": candidate.lt_addr,
                "packet_type": candidate.packet_type,
                "hec": f"0x{candidate.hec:02X}",
            }
            for candidate in clock_candidates
        ]
    dh1_candidates = ()
    if result.header_air_bits.size == 54 and result.payload_bits.size >= 24:
        candidate_uaps: range | tuple[int, ...]
        if args.search_all_uap:
            candidate_uaps = range(256)
        elif args.uap is not None:
            candidate_uaps = (int(args.uap),)
        else:
            candidate_uaps = ()
        if candidate_uaps:
            dh1_candidates = find_dh1_candidates(
                result.header_air_bits,
                result.payload_bits,
                uaps=candidate_uaps,
                require_crc=False,
            )
    if dh1_candidates:
        diagnostics = []
        for candidate in dh1_candidates:
            payload = candidate.payload
            item: dict[str, object] = {
                "uap": f"0x{candidate.header.uap:02X}",
                "clock_6_1": candidate.header.clock_6_1,
                "whitening_enabled": candidate.header.whitening_enabled,
                "lt_addr": candidate.header.lt_addr,
                "length_bytes": payload.length_bytes,
                "crc_valid": payload.crc_valid,
                "received_crc": payload.received_crc.hex(),
                "expected_crc": payload.expected_crc.hex(),
            }
            if args.payload_pattern == "prbs9" and payload.body:
                body_bits = np.asarray(
                    [
                        (byte >> index) & 1
                        for byte in payload.body
                        for index in range(8)
                    ],
                    dtype=np.uint8,
                )
                match = match_prbs9(body_bits)
                item["prbs9"] = {
                    "bit_errors": match.bit_errors,
                    "bit_count": match.bit_count,
                    "ber": match.ber,
                    "phase": match.phase,
                    "inverted": match.inverted,
                    "time_reversed": match.time_reversed,
                }
            diagnostics.append(item)
        summary["dh1_candidates"] = diagnostics
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    shown = demod.bits[: max(0, int(args.show_bits))]
    if shown.size:
        print("bits=" + "".join(str(int(bit)) for bit in shown))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
