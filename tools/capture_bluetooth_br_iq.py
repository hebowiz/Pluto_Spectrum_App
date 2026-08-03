"""Capture one Bluetooth BR channel with Pluto and search for a known access code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import adi
import iio
import numpy as np

from pluto_sa.vsa.channel import extract_analysis_channel
from pluto_sa.vsa.model import IQRecording
from pluto_sa.vsa.profiles.bluetooth_br import BluetoothBRProfile, access_code_bits
from pluto_sa.vsa.sources import FileIQSource


def _parse_hex(value: str) -> int:
    return int(str(value), 0)


def _default_uri() -> str | None:
    try:
        contexts = iio.scan_contexts()
    except Exception:
        return None
    usb = sorted(uri for uri in contexts if uri.startswith("usb:"))
    return usb[0] if usb else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uri", default=None)
    parser.add_argument("--center-frequency", type=int, default=2_441_000_000)
    parser.add_argument("--sample-rate", type=int, default=16_000_000)
    parser.add_argument("--rf-bandwidth", type=int, default=16_000_000)
    parser.add_argument(
        "--analysis-center-frequency",
        type=float,
        default=None,
        help="absolute center frequency selected from the captured bandwidth",
    )
    parser.add_argument(
        "--analysis-bandwidth",
        type=float,
        default=None,
        help="DDC/FIR channel bandwidth; omitted to search the full capture",
    )
    parser.add_argument("--duration-ms", type=float, default=3.0)
    parser.add_argument("--gain", type=int, default=30)
    parser.add_argument("--attempts", type=int, default=50)
    parser.add_argument("--lap", type=_parse_hex, default=0x9E8B33)
    parser.add_argument("--shortened-access-code", action="store_true")
    parser.add_argument("--clock", type=_parse_hex, default=None)
    parser.add_argument("--uap", type=_parse_hex, default=None)
    parser.add_argument("--minimum-correlation", type=float, default=0.65)
    parser.add_argument("--output", type=Path, default=Path("bluetooth_capture.npz"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.duration_ms <= 0.0 or args.attempts <= 0:
        raise SystemExit("duration-ms and attempts must be positive")
    if args.analysis_center_frequency is not None and args.analysis_bandwidth is None:
        raise SystemExit(
            "--analysis-bandwidth is required with --analysis-center-frequency"
        )
    samples = max(1024, int(round(args.sample_rate * args.duration_ms / 1000.0)))
    uri = args.uri or _default_uri()
    sdr = adi.Pluto(uri=uri) if uri is not None else adi.Pluto()
    sdr.rx_lo = int(args.center_frequency)
    sdr.sample_rate = int(args.sample_rate)
    sdr.rx_rf_bandwidth = int(args.rf_bandwidth)
    sdr.rx_buffer_size = samples
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = int(args.gain)
    try:
        sdr.rx_destroy_buffer()
    except Exception:
        pass

    profile = BluetoothBRProfile(
        access_code_bits(
            args.lap, include_trailer=not args.shortened_access_code
        )
    )
    last_error = "not attempted"
    try:
        for attempt in range(1, int(args.attempts) + 1):
            iq = np.asarray(sdr.rx(), dtype=np.complex64).copy()
            recording = IQRecording(
                iq,
                sample_rate_hz=float(sdr.sample_rate),
                center_frequency_hz=float(sdr.rx_lo),
                usable_bandwidth_hz=float(sdr.rx_rf_bandwidth),
                source=f"Pluto Bluetooth attempt {attempt}",
                full_scale=2048.0,
                metadata={
                    "uri": uri,
                    "gain_db": int(args.gain),
                    "attempt": attempt,
                },
            )
            analysis_recording = recording
            if args.analysis_bandwidth is not None:
                analysis_recording = extract_analysis_channel(
                    recording,
                    center_frequency_hz=(
                        recording.center_frequency_hz
                        if args.analysis_center_frequency is None
                        else args.analysis_center_frequency
                    ),
                    bandwidth_hz=args.analysis_bandwidth,
                )
            try:
                result = profile.analyze(
                    analysis_recording,
                    clock_6_1=args.clock,
                    uap=args.uap,
                    minimum_correlation=args.minimum_correlation,
                )
            except ValueError as error:
                last_error = str(error)
                continue
            FileIQSource.save_npz(args.output, recording)
            demod = result.demodulation
            summary = {
                "matched": True,
                "attempt": attempt,
                "uri": uri,
                "output": str(args.output.resolve()),
                "center_frequency_hz": int(sdr.rx_lo),
                "sample_rate_hz": int(sdr.sample_rate),
                "analysis_sample_rate_hz": analysis_recording.sample_rate_hz,
                "analysis_center_frequency_hz": analysis_recording.center_frequency_hz,
                "analysis_bandwidth_hz": analysis_recording.usable_bandwidth_hz,
                "samples": int(iq.size),
                "lap": f"0x{args.lap:06X}",
                "access_correlation": demod.access_correlation,
                "access_bit_errors": demod.access_bit_errors,
                "access_start_sample": demod.access_start_sample,
                "carrier_frequency_offset_hz": demod.carrier_frequency_offset_hz,
                "frequency_deviation_hz": demod.frequency_deviation_hz,
                "iq_inverted": demod.iq_inverted,
                "header": (
                    None
                    if result.header is None
                    else {
                        "lt_addr": result.header.lt_addr,
                        "packet_type": result.header.packet_type,
                        "hec_valid": result.header.hec_valid,
                    }
                ),
            }
            print(json.dumps(summary, indent=2))
            return 0
    finally:
        try:
            sdr.rx_destroy_buffer()
        except Exception:
            pass
    print(
        json.dumps(
            {
                "matched": False,
                "attempts": int(args.attempts),
                "uri": uri,
                "center_frequency_hz": int(args.center_frequency),
                "last_error": last_error,
            },
            indent=2,
        )
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
