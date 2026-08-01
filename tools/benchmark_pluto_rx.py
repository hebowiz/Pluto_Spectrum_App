"""Short, receive-only PlutoSDR throughput benchmark.

This changes RX settings temporarily but never enables transmission. Device settings
remain at the last benchmark point when the process exits.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import adi


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))
    return float(ordered[index])


def benchmark_rate(sdr, sample_rate_hz: int, block_samples: int, duration_s: float) -> dict:
    sdr.sample_rate = int(sample_rate_hz)
    sdr.rx_rf_bandwidth = int(min(sample_rate_hz, 20_000_000))
    sdr.rx_buffer_size = int(block_samples)
    try:
        sdr.rx_destroy_buffer()
    except Exception:
        pass

    warmup = sdr.rx()
    if len(warmup) != block_samples:
        raise RuntimeError(f"warmup returned {len(warmup)} samples, expected {block_samples}")

    elapsed_calls: list[float] = []
    total_samples = 0
    started = time.perf_counter()
    while (time.perf_counter() - started) < duration_s:
        call_started = time.perf_counter()
        iq = sdr.rx()
        elapsed_calls.append(time.perf_counter() - call_started)
        total_samples += int(len(iq))
        if len(iq) != block_samples:
            raise RuntimeError(f"rx returned {len(iq)} samples, expected {block_samples}")
    elapsed_total = time.perf_counter() - started
    expected_call_s = block_samples / float(sample_rate_hz)
    slow_calls = sum(value > expected_call_s * 1.2 for value in elapsed_calls)
    effective_sample_rate_hz = total_samples / elapsed_total
    throughput_ratio = effective_sample_rate_hz / float(sample_rate_hz)
    return {
        "requested_sample_rate_hz": int(sample_rate_hz),
        "block_samples": int(block_samples),
        "calls": len(elapsed_calls),
        "samples": int(total_samples),
        "elapsed_s": elapsed_total,
        "effective_sample_rate_hz": effective_sample_rate_hz,
        "throughput_ratio": throughput_ratio,
        "sustainable_98pct": throughput_ratio >= 0.98,
        "mean_refill_ms": statistics.fmean(elapsed_calls) * 1000.0,
        "p95_refill_ms": percentile(elapsed_calls, 0.95) * 1000.0,
        "max_refill_ms": max(elapsed_calls) * 1000.0,
        "expected_refill_ms": expected_call_s * 1000.0,
        "slow_call_count": int(slow_calls),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", required=True)
    parser.add_argument("--rates", default="1000000,2000000,4000000,6000000,8000000")
    parser.add_argument("--block-samples", type=int, default=65_536)
    parser.add_argument("--duration", type=float, default=1.5)
    args = parser.parse_args()

    rates = [int(value) for value in args.rates.split(",") if value.strip()]
    sdr = adi.Pluto(uri=args.uri)
    sdr.rx_lo = 2_440_000_000
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = 30
    results = [
        benchmark_rate(sdr, rate, args.block_samples, args.duration)
        for rate in rates
    ]
    print(json.dumps({"uri": args.uri, "results": results}, indent=2))


if __name__ == "__main__":
    main()
