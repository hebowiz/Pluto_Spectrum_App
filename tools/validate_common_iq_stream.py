"""Receive-only validation of PlutoReceiver and the common IQ stream."""

from __future__ import annotations

import argparse
import json
import time

from pluto_sa.config.spectrum_config import SpectrumConfig
from pluto_sa.modes.analyzer_mode import AnalyzerMode
from pluto_sa.sdr.pluto_receiver import PlutoReceiver


def consume(receiver: PlutoReceiver, duration_s: float, block_samples: int) -> dict:
    cursor = receiver.start(block_size=block_samples, source="hardware_validation")
    started = time.perf_counter()
    blocks = 0
    samples = 0
    overruns = 0
    missed_blocks = 0
    sequence_errors = 0
    sample_index_errors = 0
    expected_sequence = None
    expected_sample_index = None
    capture_elapsed: list[float] = []
    first_stream_id = None
    while (time.perf_counter() - started) < duration_s:
        result = receiver.read_iq_stream(cursor, max_blocks=8)
        cursor = result.cursor
        overruns += int(result.overrun)
        missed_blocks += int(result.missed_blocks)
        for block in result.blocks:
            first_stream_id = block.stream_id if first_stream_id is None else first_stream_id
            if expected_sequence is not None and block.sequence != expected_sequence:
                sequence_errors += 1
            if expected_sample_index is not None and block.start_sample_index != expected_sample_index:
                sample_index_errors += 1
            expected_sequence = block.sequence + 1
            expected_sample_index = block.end_sample_index
            blocks += 1
            samples += block.sample_count
            capture_elapsed.append(block.capture_elapsed_s)
        time.sleep(0.001)
    stopped_cleanly = receiver.stop()
    elapsed_s = time.perf_counter() - started
    return {
        "stream_id": first_stream_id,
        "blocks": blocks,
        "samples": samples,
        "elapsed_s": elapsed_s,
        "effective_sample_rate_hz": samples / elapsed_s,
        "overruns": overruns,
        "missed_blocks": missed_blocks,
        "sequence_errors": sequence_errors,
        "sample_index_errors": sample_index_errors,
        "max_capture_elapsed_ms": max(capture_elapsed, default=0.0) * 1000.0,
        "stopped_cleanly": stopped_cleanly,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default=None)
    parser.add_argument("--sample-rate", type=int, default=4_000_000)
    parser.add_argument("--block-samples", type=int, default=65_536)
    parser.add_argument("--duration", type=float, default=3.0)
    args = parser.parse_args()

    config = SpectrumConfig(
        analyzer_mode=AnalyzerMode.HIGH_SPEED_TIME_ANALYZER,
        time_analyzer_sample_rate_hz=args.sample_rate,
        time_analyzer_rf_bandwidth_hz=min(args.sample_rate, 20_000_000),
        fft_size=4096,
        capture_buffer_blocks=512,
        sdr_uri=args.uri,
    )
    receiver = PlutoReceiver(config)
    first = consume(receiver, args.duration, args.block_samples)
    second = consume(receiver, min(1.0, args.duration), args.block_samples)
    receiver.close()
    print(
        json.dumps(
            {
                "connection_uri": receiver.connection_uri,
                "requested_sample_rate_hz": args.sample_rate,
                "block_samples": args.block_samples,
                "first_run": first,
                "restart_run": second,
                "stream_id_advanced": (
                    first["stream_id"] is not None
                    and second["stream_id"] is not None
                    and second["stream_id"] > first["stream_id"]
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
