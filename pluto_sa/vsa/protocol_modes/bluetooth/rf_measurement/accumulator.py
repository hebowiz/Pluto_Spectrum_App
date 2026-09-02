"""Multi-packet aggregation for Bluetooth RF test measurements."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .model import (
    BluetoothRFMeasurementResult,
    RFTestEligibility,
    RFTestVerdict,
)


@dataclass
class BluetoothRFTestAccumulator:
    """Collect evidence without converting incomplete tests into a verdict."""

    _results: list[BluetoothRFMeasurementResult] = field(default_factory=list)

    def add(self, result: BluetoothRFMeasurementResult) -> None:
        self._results.append(result)

    @property
    def results(self) -> tuple[BluetoothRFMeasurementResult, ...]:
        return tuple(self._results)

    def aggregate_edr(self, *, required_blocks: int = 200) -> BluetoothRFMeasurementResult:
        edr = [result for result in self._results if result.test_case_id == "bluetooth.edr"]
        reasons = tuple(dict.fromkeys(
            reason
            for result in edr
            for reason in result.eligibility.reasons
        ))
        rms = np.concatenate(
            [result.arrays.get("block_rms_devm", np.empty(0)) for result in edr]
        ) if edr else np.empty(0)
        peak = np.concatenate(
            [result.arrays.get("block_peak_devm", np.empty(0)) for result in edr]
        ) if edr else np.empty(0)
        symbol_devm = np.concatenate(
            [result.arrays.get("symbol_devm", np.empty(0)) for result in edr]
        ) if edr else np.empty(0)
        block_count = int(rms.size)
        if block_count < int(required_blocks):
            reasons = (*reasons, f"requires {required_blocks} DEVM blocks; {block_count} available")
        eligibility = RFTestEligibility.from_reasons(reasons)
        modulation = next(
            (
                str(result.metadata.get("modulation"))
                for result in edr
                if result.metadata.get("modulation")
            ),
            "",
        )
        if "8" in modulation:
            rms_limit, percentile_limit, peak_limit = 0.13, 0.20, 0.25
        else:
            rms_limit, percentile_limit, peak_limit = 0.20, 0.30, 0.35
        rms_worst = float(np.max(rms)) if rms.size else None
        peak_worst = float(np.max(peak)) if peak.size else None
        percentile_99 = (
            float(np.percentile(symbol_devm, 99.0)) if symbol_devm.size else None
        )
        verdict = RFTestVerdict.NOT_APPLICABLE
        if (
            eligibility.eligible
            and rms_worst is not None
            and percentile_99 is not None
            and peak_worst is not None
        ):
            verdict = (
                RFTestVerdict.PASS
                if rms_worst <= rms_limit
                and percentile_99 <= percentile_limit
                and peak_worst <= peak_limit
                else RFTestVerdict.FAIL
            )
        return BluetoothRFMeasurementResult(
            "bluetooth.edr.aggregate",
            eligibility,
            verdict,
            metrics={
                "packet_count": len(edr),
                "block_count": block_count,
                "rms_devm_worst": rms_worst,
                "devm_99_percentile": percentile_99,
                "peak_devm_worst": peak_worst,
                "sync_symbol_errors": sum(
                    int(result.metrics.get("sync_symbol_errors", 0)) for result in edr
                ),
                "trailer_symbol_errors": sum(
                    int(result.metrics.get("trailer_symbol_errors", 0)) for result in edr
                ),
            },
            arrays={
                "block_rms_devm": rms,
                "block_peak_devm": peak,
                "symbol_devm": symbol_devm,
            },
            metadata={"required_blocks": int(required_blocks), "modulation": modulation},
        )

    def aggregate_fsk(self) -> BluetoothRFMeasurementResult:
        fsk = [result for result in self._results if result.test_case_id == "bluetooth.fsk"]
        reasons = tuple(dict.fromkeys(
            reason
            for result in fsk
            for reason in result.eligibility.reasons
        ))
        f1 = np.asarray(
            [
                float(result.metrics["delta_f1_avg_hz"])
                for result in fsk
                if result.metrics.get("delta_f1_avg_hz") is not None
            ],
            dtype=np.float64,
        )
        f2 = np.asarray(
            [
                float(result.metrics["delta_f2_avg_hz"])
                for result in fsk
                if result.metrics.get("delta_f2_avg_hz") is not None
            ],
            dtype=np.float64,
        )
        if not f1.size:
            reasons = (*reasons, "requires at least one 11110000 packet")
        if not f2.size:
            reasons = (*reasons, "requires at least one 10101010 packet")
        eligibility = RFTestEligibility.from_reasons(reasons)
        f1_avg = float(np.mean(f1)) if f1.size else None
        f2_avg = float(np.mean(f2)) if f2.size else None
        ratio = (
            f2_avg / f1_avg
            if f1_avg is not None and f2_avg is not None and f1_avg > 0.0
            else None
        )
        return BluetoothRFMeasurementResult(
            "bluetooth.fsk.aggregate",
            eligibility,
            RFTestVerdict.NOT_APPLICABLE,
            metrics={
                "packet_count": len(fsk),
                "delta_f1_avg_hz": f1_avg,
                "delta_f2_avg_hz": f2_avg,
                "delta_f2_ratio": ratio,
            },
            arrays={"delta_f1_packet_hz": f1, "delta_f2_packet_hz": f2},
            metadata={"limits_require_phy_and_test_case_selection": True},
        )
