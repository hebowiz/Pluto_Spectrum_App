"""Debug helper for probing Sweep SA point measurements without the UI."""

from __future__ import annotations

import argparse

import numpy as np

from pluto_sa.config.spectrum_config import SpectrumConfig
from pluto_sa.modes.analyzer_mode import AnalyzerMode
from pluto_sa.modes.sweep_controller import SweepController
from pluto_sa.sdr.pluto_receiver import PlutoReceiver


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe Sweep SA point measurements around a frequency window.",
    )
    parser.add_argument("--center-mhz", type=float, default=2440.0, help="Center frequency [MHz]")
    parser.add_argument(
        "--span-khz",
        type=float,
        default=2000.0,
        help="Probe span around the center frequency [kHz]",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=21,
        help="Number of probe points across the requested span",
    )
    parser.add_argument(
        "--rbw-khz",
        type=float,
        default=1000.0,
        help="RBW setting used for the point measurement [kHz]",
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="Sample",
        choices=["Sample", "Peak", "RMS"],
        help="Detector mode used for one-point reduction",
    )
    parser.add_argument(
        "--flush-reads",
        type=int,
        default=2,
        help="Number of post-retune capture flush reads before measurement",
    )
    parser.add_argument(
        "--capture-samples",
        type=int,
        default=0,
        help="Override capture sample count (0 = automatic RBW-based sizing)",
    )
    parser.add_argument(
        "--int-gain-db",
        type=int,
        default=30,
        help="Internal Pluto gain [dB] for the debug capture path",
    )
    return parser


def format_result_line(center_hz: int, result) -> str:
    offset_khz = (result.frequency_hz - center_hz) / 1e3
    return (
        f"{result.frequency_hz / 1e6:10.6f} MHz | "
        f"{offset_khz:10.1f} kHz | "
        f"{result.measured_power_db:9.2f} dB | "
        f"{result.capture_samples:6d} samples | "
        f"RBW {result.effective_rbw_hz / 1e3:8.1f} kHz | "
        f"bin {result.bin_width_hz:8.1f} Hz"
    )


def format_debug_block(result) -> list[str]:
    detector_values = ", ".join(f"{value:.6e}" for value in result.detector_input_values)
    return [
        f"  peak bin (unshifted): {result.peak_bin_index_unshifted}",
        f"  peak bin (shifted):   {result.peak_bin_index_shifted}",
        f"  peak rel freq:        {result.peak_frequency_relative_hz / 1e3:10.3f} kHz",
        f"  RBW center bin:       {result.rbw_center_bin_index}",
        f"  RBW center freq:      {result.rbw_center_frequency_hz / 1e3:10.3f} kHz",
        f"  detector input:       [{detector_values}]",
    ]


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    center_hz = int(round(args.center_mhz * 1e6))
    span_hz = float(args.span_khz * 1e3)
    rbw_hz = None if args.rbw_khz <= 0.0 else float(args.rbw_khz * 1e3)

    config = SpectrumConfig(
        analyzer_mode=AnalyzerMode.SWEEP_SA,
        center_freq_hz=center_hz,
        display_span_hz=max(1, int(round(span_hz))),
        rbw_hz=rbw_hz,
        rx_gain_db=args.int_gain_db,
        sweep_points=max(1, args.points),
        sweep_detector_mode=args.detector,
        sweep_retune_flush_reads=max(0, args.flush_reads),
        sweep_capture_samples_override=(
            None if args.capture_samples <= 0 else int(args.capture_samples)
        ),
    )

    receiver = PlutoReceiver(config)
    controller = SweepController(config, receiver)

    try:
        if args.points <= 1:
            frequencies_hz = np.array([center_hz], dtype=np.int64)
        else:
            half_span_hz = span_hz / 2.0
            frequencies_hz = np.linspace(
                center_hz - half_span_hz,
                center_hz + half_span_hz,
                args.points,
                dtype=np.int64,
            )

        print(
            f"flush_reads={config.sweep_retune_flush_reads}, "
            f"detector={config.sweep_detector_mode}, "
            f"rbw={config.rbw_hz if config.rbw_hz is not None else 'None'} Hz, "
            f"capture_override={config.sweep_capture_samples_override}"
        )
        print("Frequency       |   Offset     |  Measured | Capture        | Effective RBW")
        print("-" * 89)
        for result in controller.measure_points(frequencies_hz):
            print(format_result_line(center_hz, result))
            for line in format_debug_block(result):
                print(line)
    finally:
        receiver.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
